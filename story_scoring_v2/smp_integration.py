# ============================================================
# SHORTMAKERPRO INTEGRATION ADAPTER
#
# The ONE file a ShortMakerPro developer needs to import. It does not modify
# any ShortMakerPro file and does not assume where Stable Audio runs — it
# only defines the boundary between the validated V2 scoring engine
# (scoring_engine.py / music_generation.py / pipeline.py, all unchanged in
# architecture) and however ShortMakerPro actually talks to Stable Audio.
#
# Two supported setups:
#
#   1. Stable Audio already loaded in-process (this is what the validated
#      Kaggle notebook does today, and what dist/story_generalization_v2_
#      exact_waveform.py does when pasted into a Kaggle cell). Nothing to
#      configure — this is the default provider (StableAudioDirectProvider
#      in music_generation.py). Just call run_scoring_job(...).
#
#   2. ShortMakerPro talks to a separate, already-running Kaggle session
#      over a WebSocket — the SAME pattern used for voice (kaggle_bridge.py /
#      kaggle_persistent_bridge.py), but for music it is a SEPARATE, already
#      existing bridge: kaggle_music_bridge.py (verified against the real
#      file in Desktop/story_scoring_kaggle_music_bundle/
#      story_scoring_kaggle_music_bundle/, paired with
#      stable_audio_3_music_kaggle.ipynb). Install
#      KaggleBridgeMusicProvider(bridge_dir) — see its docstring for the
#      REAL contract this was verified against, and its one real limitation
#      (no conditioned/reference generation support today).
#
# Entry point:
#
#     from smp_integration import run_scoring_job
#     result = run_scoring_job(narration_paths=[...], out_root=...)
#     # result = {"run_seed": int, "manifests": [...], "summary": [...]}
#
# `run_scoring_job` is the ONLY function ShortMakerPro's worker code needs to
# call. Everything else in this file exists to let it plug in its own
# Stable-Audio transport without editing scoring_engine.py, music_generation.py
# or pipeline.py.
# ============================================================

import importlib.util
import tempfile
import uuid
from pathlib import Path

import numpy as np

from music_generation import MusicProvider, set_provider, get_provider
from run_seed import generate_run_seed
from pipeline import process_story, summarize, save_json
from scoring_engine import to_numpy_audio


# ============================================================
# CALLABLE PROVIDER — wrap two plain functions, no subclassing needed
# ============================================================

class CallableMusicProvider(MusicProvider):
    """Wraps two plain functions as a MusicProvider — the fastest way to
    plug in a custom transport without subclassing.

        provider = CallableMusicProvider(
            fresh=lambda prompt, duration, seed: my_bridge.music(prompt, duration, seed),
            conditioned=lambda prompt, duration, reference, noise, seed:
                my_bridge.music(prompt, duration, seed, reference=reference, noise=noise),
        )
    """

    def __init__(self, fresh, conditioned):
        self._fresh = fresh
        self._conditioned = conditioned

    def generate_fresh(self, prompt, duration, seed):
        return to_numpy_audio(self._fresh(prompt, duration, seed))

    def generate_conditioned(self, prompt, duration, reference, noise, seed):
        return to_numpy_audio(self._conditioned(prompt, duration, reference, noise, seed))


# ============================================================
# KAGGLE-BRIDGE PROVIDER — the REAL music bridge, verified against
# Desktop/story_scoring_kaggle_music_bundle/story_scoring_kaggle_music_bundle/
# kaggle_music_bridge.py (paired with stable_audio_3_music_kaggle.ipynb).
#
# This is a SEPARATE bridge from ShortMakerPro's voice bridge
# (kaggle_bridge.py / kaggle_persistent_bridge.py) — voice and music each
# keep their own persistent Kaggle kernel/session, exactly as the bundle's
# own README says ("Keep it separate from the existing ShortMakerPro voice
# kernel."). This class was written by reading the actual bridge file, not
# assumed: its one public entry point is
#
#     KaggleMusicKernel(ws_url).generate(prompt, duration, seed, output: Path) -> Path
#
# i.e. exactly the same (prompt, duration, seed) shape as
# MusicProvider.generate_fresh() — good news for the motif. There is NO
# init_audio / init_noise_level parameter anywhere in the real bridge: its
# remote generation code (kaggle_music_bridge.build_generation_code) always
# calls music_model.generate(prompt=..., duration=..., steps=8, cfg_scale=1.0,
# seed=..., batch_size=1) with no reference clip. So generate_conditioned()
# — which every layer phrase needs, to stay locked to the motif — cannot be
# routed through this bridge today. See this class's docstring and item F of
# FINAL_SMP_INTEGRATION_REPORT.md.
# ============================================================

def load_kaggle_music_bridge(bridge_dir):
    """Loads the real kaggle_music_bridge.py module from its actual location
    (Desktop/story_scoring_kaggle_music_bundle/story_scoring_kaggle_music_bundle/
    on this machine) — same importlib mechanism story_scoring_lab_kaggle.py's
    own load_kaggle_bridge() uses for the voice bridge."""
    bridge_dir = Path(bridge_dir)
    bridge_path = bridge_dir / "kaggle_music_bridge.py"
    if not bridge_path.is_file():
        raise FileNotFoundError(
            f"Could not find {bridge_path}. Pass the folder that contains "
            "kaggle_music_bridge.py (the story_scoring_kaggle_music_bundle "
            "inner folder)."
        )
    spec = importlib.util.spec_from_file_location("smp_kaggle_music_bridge", bridge_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {bridge_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class KaggleBridgeMusicProvider(MusicProvider):
    """Routes generate_fresh() (the motif) through the REAL, already-existing
    remote Stable Audio 3 Kaggle bridge — kaggle_music_bridge.KaggleMusicKernel
    — the exact class story_scoring_lab_kaggle.py's own KaggleStableAudioProvider
    uses. Connects and loads the model eagerly in __init__ (not lazily on the
    first call), so a bad URL or a cold kernel fails fast instead of burning a
    Kaggle GPU minute mid-job.

    generate_conditioned() (every core/pressure/climax phrase) is NOT
    supported by the deployed bridge — see the module-level note above — and
    raises a clear RuntimeError rather than silently generating unconditioned
    (motif-drifting) phrases. That would be a quiet behaviour change to the
    validated V2 design, which this integration must not introduce. Use
    StableAudioDirectProvider (the default; Stable Audio loaded in the SAME
    Kaggle kernel that runs the scoring job) for a real end-to-end job today.
    """

    def __init__(self, bridge_dir, url_file=None, log=print):
        self.bridge_dir = Path(bridge_dir)
        self.bridge = load_kaggle_music_bridge(self.bridge_dir)
        url_path = Path(url_file) if url_file is not None else self.bridge.DEFAULT_URL_FILE
        url = self.bridge.load_url(url_path)
        self._kernel = self.bridge.KaggleMusicKernel(url)
        log("[KaggleBridgeMusicProvider] connecting to the remote music kernel...")
        self._kernel.connect()
        self._kernel.ensure_model()

    def generate_fresh(self, prompt, duration, seed):
        with tempfile.TemporaryDirectory(prefix="smp_music_bridge_") as tmp:
            output = Path(tmp) / f"fresh_{uuid.uuid4().hex}.flac"
            result_path = self._kernel.generate(
                prompt=prompt, duration=float(duration), seed=int(seed), output=output
            )
            return to_numpy_audio(_load_bridge_result(result_path))

    def generate_conditioned(self, prompt, duration, reference, noise, seed):
        raise RuntimeError(
            "KaggleBridgeMusicProvider cannot perform conditioned (phrase) "
            "generation: the deployed kaggle_music_bridge.py / "
            "stable_audio_3_music_kaggle.ipynb generation code calls "
            "music_model.generate(prompt=..., duration=..., steps=8, "
            "cfg_scale=1.0, seed=..., batch_size=1) with no init_audio or "
            "init_noise_level argument, so the remote kernel has no way to "
            "condition on a reference clip today. Every layer phrase "
            "(core/pressure/climax) needs generate_conditioned() to stay "
            "locked to the motif, so this provider cannot run a full scoring "
            "job by itself. Use the default StableAudioDirectProvider "
            "(Stable Audio loaded in the SAME Kaggle kernel that runs the "
            "scoring job, e.g. by pasting dist/story_generalization_v2_"
            "exact_waveform.py into the stable_audio_3_music_kaggle.ipynb "
            "kernel after 'STABLE AUDIO 3 SMALL MUSIC READY'), or extend "
            "kaggle_music_bridge.build_generation_code() to accept "
            "init_audio/init_noise_level and re-verify."
        )

    def close(self):
        self._kernel.close()


def _load_bridge_result(result):
    """kaggle_music_bridge.KaggleMusicKernel.generate() always writes real
    FLAC bytes (sf.write(..., format="FLAC")), regardless of the output
    path's extension. audio_io.read_audio() only has a WAV fallback when
    soundfile isn't installed, and neither ShortMakerPro's requirements.txt
    nor the music bundle's requirements list soundfile — so on a machine
    without it, decoding the bridge's real FLAC output would otherwise fail
    outright. ffmpeg is already a hard ShortMakerPro dependency (kaggle_bridge.py
    imports the `ffmpeg` package; requirements.txt pins ffmpeg-python), so this
    converts FLAC -> WAV with it when soundfile isn't available, instead of
    requiring a new dependency."""
    if isinstance(result, (str, Path)):
        from audio_io import read_audio, SOUNDFILE_OK
        result = Path(result)
        if SOUNDFILE_OK or result.suffix.lower() == ".wav":
            audio, _sr = read_audio(result)
            return audio
        return _decode_via_ffmpeg(result)
    return result


def _decode_via_ffmpeg(path):
    import subprocess
    from audio_io import read_audio
    with tempfile.TemporaryDirectory(prefix="smp_music_bridge_decode_") as tmp:
        wav_path = Path(tmp) / "decoded.wav"
        proc = subprocess.run(
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
             "-i", str(path), "-ar", "44100", "-ac", "2", "-c:a", "pcm_s16le", str(wav_path)],
            capture_output=True, text=True, timeout=120,
        )
        if proc.returncode != 0 or not wav_path.is_file():
            raise RuntimeError(
                f"Could not decode {path} (soundfile is not installed and the "
                f"ffmpeg fallback failed): {proc.stderr.strip()}"
            )
        audio, _sr = read_audio(wav_path)
        return audio


# ============================================================
# ENTRY POINT
# ============================================================

def run_scoring_job(narration_paths, out_root, run_seed=None, provider=None, log=print):
    """Scores one or more narration files as ONE production job.

    narration_paths: list of audio file paths (already-finished narration —
                      the same input the validated pipeline always took).
    out_root:         output folder; one story_NN_<stem>/ subfolder per file,
                      same layout as the Kaggle pipeline
                      (score_raw.wav / score_ducked.wav / voice_music_preview.wav
                      / manifest.json), plus a job-level summary.
    run_seed:         omit for a fresh production job (recommended — every
                      call to run_scoring_job() then generates new material).
                      Pass a previous job's run_seed to reproduce it exactly.
    provider:         a MusicProvider (e.g. KaggleBridgeMusicProvider(bridge_dir)
                      or CallableMusicProvider(...)). Omit to use whatever
                      provider is already active — the default is
                      StableAudioDirectProvider, correct when Stable Audio is
                      loaded in the same process (the validated Kaggle-notebook
                      case, and the only provider that supports conditioned
                      phrase generation today — see KaggleBridgeMusicProvider's
                      docstring).

    Returns {"run_seed": int, "manifests": [...], "summary": [...]}.
    Raises on any story that fails — this function does not silently skip a
    story (matches pipeline.main()'s behaviour: a scoring job either produces
    validated output for every input or it stops).
    """
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    resolved_seed = run_seed if run_seed is not None else generate_run_seed()

    previous_provider = None
    if provider is not None:
        previous_provider = set_provider(provider)
    try:
        log(f"Scoring job run_seed={resolved_seed} "
            f"({'explicit — reproducing a previous job' if run_seed is not None else 'fresh'}), "
            f"provider={type(get_provider()).__name__}")
        manifests = [
            process_story(i, path, out_root, resolved_seed, log=log)
            for i, path in enumerate(narration_paths, start=1)
        ]
    finally:
        if previous_provider is not None:
            set_provider(previous_provider)

    summary = summarize(manifests)
    for row in summary:
        row["run_seed"] = resolved_seed
    save_json(out_root / "generalization_summary.json", summary)
    return {"run_seed": resolved_seed, "manifests": manifests, "summary": summary}
