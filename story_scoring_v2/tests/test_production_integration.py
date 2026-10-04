"""Production integration tests: run-seed handling, the MusicProvider
plug point, and the ShortMakerPro adapter (smp_integration.py).

Fully offline — a small deterministic FakeMusicProvider stands in for Stable
Audio, exactly like the earlier mocked-Kaggle smoke test, but wired through
the real MusicProvider interface instead of monkeypatching stable_audio.

Run:  python tests/test_production_integration.py
"""

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import scoring_engine as E  # noqa: E402
import music_generation as MG  # noqa: E402
import run_seed as RS  # noqa: E402
from audio_io import write_audio  # noqa: E402
from pipeline import process_story  # noqa: E402
from smp_integration import (  # noqa: E402
    run_scoring_job, CallableMusicProvider, KaggleBridgeMusicProvider,
)


# ============================================================
# FAKE PROVIDER — deterministic function of (seed, duration), no torch/GPU
# ============================================================

class FakeMusicProvider(MG.MusicProvider):
    """Bit-identical output for the same seed; different output for a
    different seed. Ignores `prompt`/`reference`/`noise` content (only their
    shapes matter) so tests isolate seed behaviour specifically."""

    def __init__(self):
        self.calls = []

    def _tone(self, duration, seed):
        rng = np.random.default_rng(seed)
        n = int(round(duration * E.SR))
        t = np.arange(n) / E.SR
        freq = rng.uniform(80, 320)
        env = 0.5 + 0.5 * np.sin(2 * np.pi * t / rng.uniform(2, 6) + rng.uniform(0, 6))
        x = 0.3 * np.sin(2 * np.pi * freq * t) * env + 0.03 * rng.standard_normal(n)
        x = x / (np.max(np.abs(x)) + 1e-9) * 0.85
        return E.f32(np.stack([x, 0.85 * x], axis=1))

    def generate_fresh(self, prompt, duration, seed):
        self.calls.append(("fresh", seed))
        return self._tone(duration, seed)

    def generate_conditioned(self, prompt, duration, reference, noise, seed):
        self.calls.append(("conditioned", seed))
        return self._tone(duration, seed)


def make_narration(path, seconds=42.0, seed=1):
    rng = np.random.default_rng(seed)
    n = int(seconds * E.SR)
    x = np.zeros(n, dtype=np.float32)
    pos = 0
    while pos < n:
        L = int(rng.uniform(0.8, 3.0) * E.SR)
        L = min(L, n - pos)
        x[pos:pos + L] = 0.2 * rng.standard_normal(L) * np.hanning(max(L, 1))
        pos += L + int(rng.uniform(0.1, 0.6) * E.SR)
    write_audio(path, np.stack([x, x], axis=1), E.SR)
    return path


class _Provider:
    """Context manager: installs a provider, restores the previous one."""

    def __init__(self, provider):
        self.provider = provider
        self.previous = None

    def __enter__(self):
        self.previous = MG.set_provider(self.provider)
        return self.provider

    def __exit__(self, *exc):
        MG.set_provider(self.previous)


# ============================================================
# run_seed.py — pure logic
# ============================================================

def test_generate_run_seed_in_range_and_unpredictable():
    seeds = {RS.generate_run_seed() for _ in range(50)}
    assert len(seeds) == 50  # no collisions in 50 draws
    assert all(RS.SEED_MIN <= s <= RS.SEED_MAX for s in seeds)


def test_derive_seed_deterministic():
    a = RS.derive_seed(12345, "motif", 1)
    b = RS.derive_seed(12345, "motif", 1)
    assert a == b


def test_derive_seed_distinguishes_parts_and_run_seed():
    base = RS.derive_seed(1, "layer", "core", 1, 0, 0)
    diff_run = RS.derive_seed(2, "layer", "core", 1, 0, 0)
    diff_layer = RS.derive_seed(1, "layer", "pressure", 1, 0, 0)
    diff_phrase = RS.derive_seed(1, "layer", "core", 1, 1, 0)
    assert len({base, diff_run, diff_layer, diff_phrase}) == 4


def test_derive_seed_requires_run_seed():
    try:
        RS.derive_seed(None, "motif", 1)
    except ValueError:
        return
    raise AssertionError("expected ValueError for missing run_seed")


# ============================================================
# MusicProvider plug point
# ============================================================

def test_default_provider_is_stable_audio_direct():
    assert isinstance(MG.get_provider(), MG.StableAudioDirectProvider)


def test_set_provider_restores_previous():
    original = MG.get_provider()
    with _Provider(FakeMusicProvider()) as fake:
        assert MG.get_provider() is fake
    assert MG.get_provider() is original


def test_generate_motif_same_run_seed_bit_identical():
    with _Provider(FakeMusicProvider()):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            m1, _, _, seed1 = MG.generate_motif(1, "financial pressure", tmp, run_seed=999)
            m2, _, _, seed2 = MG.generate_motif(1, "financial pressure", tmp, run_seed=999)
            assert seed1 == seed2
            assert np.array_equal(m1, m2)


def test_generate_motif_different_run_seed_differs():
    with _Provider(FakeMusicProvider()):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            m1, _, _, seed1 = MG.generate_motif(1, "financial pressure", tmp, run_seed=1)
            m2, _, _, seed2 = MG.generate_motif(1, "financial pressure", tmp, run_seed=2)
            assert seed1 != seed2
            assert not np.array_equal(m1, m2)


def test_generate_motif_same_story_index_different_story_in_same_run_differs():
    # Two stories in the SAME job (same run_seed) must still get different material.
    with _Provider(FakeMusicProvider()):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            m1, _, _, seed1 = MG.generate_motif(1, "financial pressure", tmp, run_seed=42)
            m2, _, _, seed2 = MG.generate_motif(2, "financial pressure", tmp, run_seed=42)
            assert seed1 != seed2
            assert not np.array_equal(m1, m2)


# ============================================================
# Full process_story() with the fake provider — same run_seed vs fresh
# ============================================================

def _run_story(out_root, run_seed, narration_path, story_aware=False):
    # These tests pin the validated legacy engine; the story-aware path has
    # its own tests in test_story_scoring.py.
    with _Provider(FakeMusicProvider()):
        return process_story(1, narration_path, out_root, run_seed, log=lambda *_: None,
                             story_aware=story_aware)


def test_process_story_same_run_seed_reproduces_bit_identical_output():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        narration = make_narration(tmp / "narr.wav")
        out_a = tmp / "run_a"
        out_b = tmp / "run_b"
        m1 = _run_story(out_a, 555, narration)
        m2 = _run_story(out_b, 555, narration)
        assert m1["motif"]["seed"] == m2["motif"]["seed"]
        assert m1["optimizer"]["orders"] == m2["optimizer"]["orders"]
        assert m1["final_metrics"]["objective"] == m2["final_metrics"]["objective"]
        raw_a = (out_a / "story_01_narr" / "score_raw.wav").read_bytes()
        raw_b = (out_b / "story_01_narr" / "score_raw.wav").read_bytes()
        assert raw_a == raw_b


def test_process_story_different_run_seed_produces_different_output():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        narration = make_narration(tmp / "narr.wav")
        out_a = tmp / "run_a"
        out_b = tmp / "run_b"
        m1 = _run_story(out_a, 1, narration)
        m2 = _run_story(out_b, 2, narration)
        assert m1["motif"]["seed"] != m2["motif"]["seed"]
        raw_a = (out_a / "story_01_narr" / "score_raw.wav").read_bytes()
        raw_b = (out_b / "story_01_narr" / "score_raw.wav").read_bytes()
        assert raw_a != raw_b


def test_process_story_alignment_still_holds_with_fake_provider():
    """The part that matters most: even with fresh generated material, the
    optimizer's reported metrics still equal the final render exactly — this
    is scoring_engine.py behaviour, untouched by the run-seed change."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        narration = make_narration(tmp / "narr.wav", seed=7)
        manifest = _run_story(tmp / "run", 4242, narration)
        assert manifest["alignment"]["optimizer_vs_final_render"]["aligned"]
        assert manifest["alignment"]["optimizer_vs_final_render"]["worst_db_diff"] == 0.0
        assert manifest["ducking"]["max_duck_db"] == E.MAX_DUCK_DB
        assert manifest["ducking"]["attack_seconds"] == E.DUCK_ATTACK_SECONDS
        assert manifest["ducking"]["release_seconds"] == E.DUCK_RELEASE_SECONDS


# ============================================================
# smp_integration.run_scoring_job()
# ============================================================

def test_run_scoring_job_fresh_vs_explicit_seed():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        narration = make_narration(tmp / "narr.wav", seed=3)
        fake = FakeMusicProvider()
        result1 = run_scoring_job([narration], tmp / "job1", run_seed=None,
                                   provider=fake, log=lambda *_: None)
        result2 = run_scoring_job([narration], tmp / "job2", run_seed=None,
                                   provider=fake, log=lambda *_: None)
        assert result1["run_seed"] != result2["run_seed"]  # two fresh jobs differ
        assert MG.get_provider() is not fake  # provider restored after each job

        result3 = run_scoring_job([narration], tmp / "job3", run_seed=result1["run_seed"],
                                   provider=fake, log=lambda *_: None)
        assert result3["manifests"][0]["motif"]["seed"] == result1["manifests"][0]["motif"]["seed"]


def test_callable_music_provider():
    provider = CallableMusicProvider(
        fresh=lambda prompt, duration, seed: FakeMusicProvider().generate_fresh(prompt, duration, seed),
        conditioned=lambda prompt, duration, reference, noise, seed:
            FakeMusicProvider().generate_conditioned(prompt, duration, reference, noise, seed),
    )
    out = provider.generate_fresh("x", 2.0, 1)
    assert out.shape[1] == 2 and out.shape[0] == int(2.0 * E.SR)


# ============================================================
# KaggleBridgeMusicProvider — the REAL kaggle_music_bridge.py contract
# (KaggleMusicKernel(ws_url).generate(prompt, duration, seed, output) -> Path,
# verified against Desktop/story_scoring_kaggle_music_bundle/
# story_scoring_kaggle_music_bundle/kaggle_music_bridge.py — NOT an assumed
# signature). The stub module below reproduces that exact shape: a
# KaggleMusicKernel class with connect()/ensure_model()/generate()/close(),
# plus load_url()/DEFAULT_URL_FILE, so these tests exercise the real
# integration boundary rather than a fictional one.
# ============================================================

_STUB_BRIDGE_CODE = '''
import subprocess
import tempfile
from pathlib import Path
import numpy as np
from scipy.io import wavfile

DEFAULT_URL_FILE = Path(__file__).resolve().parent / "kaggle_music_session_url.txt"

def load_url(path=DEFAULT_URL_FILE):
    return "wss://stub-kaggle-music-kernel.example/fake"

class KaggleMusicKernel:
    """Stub matching the REAL kaggle_music_bridge.KaggleMusicKernel contract:
    generate() always writes real FLAC bytes to `output`, exactly like the
    real bridge's sf.write(request_output, audio.T, sample_rate, format="FLAC")
    (regardless of what extension `output` was given)."""

    def __init__(self, ws_url):
        self.ws_url = ws_url
        self.connected = False
        self.model_ready = False

    def connect(self):
        self.connected = True

    def ensure_model(self):
        self.model_ready = True

    def generate(self, prompt, duration, seed, output):
        assert self.connected and self.model_ready
        rng = np.random.default_rng(seed)
        n = int(round(duration * 44100))
        x = (0.1 * rng.standard_normal(n)).astype(np.float32)
        pcm = np.clip(np.round(np.stack([x, x], axis=1) * 32767), -32768, 32767).astype(np.int16)
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as td:
            wav_tmp = Path(td) / "raw.wav"
            wavfile.write(wav_tmp, 44100, pcm)
            subprocess.run(
                ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                 "-i", str(wav_tmp), str(output)],
                check=True, timeout=60,
            )
        return output

    def close(self):
        self.connected = False
'''


def test_kaggle_bridge_missing_file_raises():
    with tempfile.TemporaryDirectory() as tmp:
        try:
            KaggleBridgeMusicProvider(Path(tmp))
        except FileNotFoundError:
            return
        raise AssertionError("expected FileNotFoundError")


def test_kaggle_bridge_present_generates_fresh_via_real_contract():
    with tempfile.TemporaryDirectory() as tmp:
        bridge_dir = Path(tmp)
        (bridge_dir / "kaggle_music_bridge.py").write_text(_STUB_BRIDGE_CODE)
        provider = KaggleBridgeMusicProvider(bridge_dir)
        try:
            assert provider._kernel.connected
            assert provider._kernel.model_ready
            audio = provider.generate_fresh("prompt", 1.0, 123)
            assert audio.shape == (44100, 2)
        finally:
            provider.close()


def test_kaggle_bridge_conditioned_generation_raises_actionable_error():
    """The real bridge has no init_audio/init_noise_level path — this must
    fail loudly, not silently drop motif conditioning."""
    with tempfile.TemporaryDirectory() as tmp:
        bridge_dir = Path(tmp)
        (bridge_dir / "kaggle_music_bridge.py").write_text(_STUB_BRIDGE_CODE)
        provider = KaggleBridgeMusicProvider(bridge_dir)
        try:
            reference = np.zeros((44100, 2), dtype=np.float32)
            try:
                provider.generate_conditioned("prompt", 1.0, reference, 0.1, 123)
            except RuntimeError as exc:
                assert "init_audio" in str(exc) and "generate_conditioned" in str(exc)
                return
            raise AssertionError("expected RuntimeError")
        finally:
            provider.close()


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"{len(tests)} tests passed")
