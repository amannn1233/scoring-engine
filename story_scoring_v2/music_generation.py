# ============================================================
# MUSIC GENERATION — Stable Audio 3 Small Music, motif, references,
# phrase generation and quality gates.
#
# Two prompt paths:
#   cue=None  -> the original V1/V2 prompts and first-valid-attempt logic,
#                byte-for-byte (kept for the validated benchmark).
#   cue=Cue   -> story-aware: prompts composed from the story's cue sheet
#                (cue_sheet.py: palette, tempo, key, descriptors), several
#                takes per motif/phrase, and the best take kept by a ranking
#                that adds tempo match, key match, loop-seam smoothness and
#                spectral fullness (music_analysis.py) to the quality gates.
# torch / stable_audio_3 / librosa are imported lazily so the scoring engine
# and the offline replay never need them.
#
# Seeds are the one thing this file deliberately changed from the original
# V1/V2 draft: every seed is now derived from a per-job `run_seed` via
# run_seed.derive_seed() instead of a fixed literal, so production runs
# generate fresh material each job while staying reproducible within a job.
# See run_seed.py for why.
# ============================================================

import sys

import numpy as np

# <local-imports>
from scoring_engine import (
    SR, PHRASE_DURATION, PHRASES_PER_LAYER, LAYER_NAMES,
    f32, fit_length, normalize_peak, rms_db, to_mono, to_numpy_audio,
)
from audio_io import write_audio
from run_seed import derive_seed
from cue_sheet import motif_prompt_from_cue, layer_prompts_from_cue
from music_analysis import (MODE_FAMILY, analyse as analyse_music, chroma as music_chroma,
                            estimate_key as music_key, estimate_tempo as music_tempo,
                            grid_lock as music_grid_lock)
from arrangement import plan_grid
# </local-imports>


MAX_ATTEMPTS = 2
MOTIF_TAKES = 3      # story-aware path: motif candidates ranked, best kept
PHRASE_TAKES = 3     # story-aware path: takes per phrase, all ranked
# Real Stable Audio 3 Small renders (CPU test, 2026-10-04) treat the
# requested duration as a whole piece: they open strong and decay or fade
# to silence by the end (a 12 s "climax" fell from -20 to -74 dB). So the
# story-aware path asks for OVERHANG_SECONDS more than it needs and keeps
# the most sustained window, never the model's ending.
OVERHANG_SECONDS = 6.0


def sustain_profile(audio, window_s=0.5):
    """Level (dB) per half second, the head-to-tail drop and the spread."""
    mono = to_mono(audio)
    w = max(1, int(round(window_s * SR)))
    n = len(mono) // w
    if n < 2:
        return {"levels": [], "drop_db": 0.0, "spread_db": 0.0}
    lv = 10 * np.log10(np.mean(mono[: n * w].reshape(n, w) ** 2, axis=1) + 1e-12)
    k = max(1, n // 4)
    return {"levels": lv, "drop_db": float(np.median(lv[:k]) - np.median(lv[-k:])),
            "spread_db": float(np.percentile(lv, 90) - np.percentile(lv, 10))}


def best_sustained_window(audio, seconds, step_s=0.25):
    """The `seconds`-long window with the least decay and level spread,
    preferring early windows (before any fade the model wrote)."""
    audio = f32(audio)
    need = int(round(seconds * SR))
    if len(audio) <= need:
        return fit_length(audio, need), 0.0
    step = max(1, int(round(step_s * SR)))
    best = None
    for start in range(0, len(audio) - need + 1, step):
        prof = sustain_profile(audio[start:start + need])
        lv = np.asarray(prof["levels"])
        dead = float(np.mean(lv < np.median(lv) - 20.0)) if len(lv) else 1.0
        score = abs(prof["drop_db"]) * 2.0 + prof["spread_db"] + 40.0 * dead + 0.2 * start / SR
        if best is None or score < best[0]:
            best = (score, start)
    start = best[1]
    return f32(audio[start:start + need]), start / SR
MOTIF_SECONDS = 28.0
STEPS = 8
CFG_SCALE = 1.0

LAYER_NOISE = {"core": 0.10, "pressure": 0.13, "climax": 0.16}
# Story-aware path: the legacy 0.10-0.16 makes phrases near-copies of the
# motif window (real renders copied its silent gaps and ignored the layer
# prompt). More freedom lets each layer play its own part around the motif.
STORY_LAYER_NOISE = {"core": 0.30, "pressure": 0.40, "climax": 0.50}
MAX_EXTRA_MOTIF_TAKES = 3     # retry while every motif take has dead air
LAYER_REFERENCE_OFFSET = {"core": 0, "pressure": 1, "climax": 2}


# ============================================================
# MUSIC PROVIDER — pluggable generation backend
#
# The validated Kaggle notebook path (load Stable Audio in-process, call
# music_model.generate(...) directly) is the DEFAULT provider below and is
# unchanged in behaviour. A production integration (e.g. ShortMakerPro's
# existing Kaggle-bridge pattern) can install a different provider with
# set_provider() without touching any prompt/gate/generation logic here —
# see smp_integration.py.
# ============================================================

class MusicProvider:
    """Minimal interface the rest of this module generates through."""

    def generate_fresh(self, prompt, duration, seed):
        raise NotImplementedError

    def generate_conditioned(self, prompt, duration, reference, noise, seed):
        raise NotImplementedError


_MODEL = {"music_model": None}


def get_music_model(log=print):
    """Loads Stable Audio once; reuses a `music_model` already in the Kaggle
    notebook namespace (so re-running the cell never reloads the model)."""
    if _MODEL["music_model"] is not None:
        return _MODEL["music_model"]
    main = sys.modules.get("__main__")
    existing = globals().get("music_model") or getattr(main, "music_model", None)
    if existing is None:
        import torch
        from stable_audio_3 import StableAudioModel
        log("Loading Stable Audio 3 Small Music...")
        existing = StableAudioModel.from_pretrained(
            "small-music", device="cuda" if torch.cuda.is_available() else "cpu"
        )
        if main is not None:
            setattr(main, "music_model", existing)
    _MODEL["music_model"] = existing
    try:
        import torch
        log(f"Model: {type(existing).__name__}")
        log(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    except Exception:
        pass
    return existing


def _generate(**kwargs):
    import torch
    model = get_music_model()
    with torch.inference_mode():
        output = model.generate(**kwargs)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return to_numpy_audio(output)


class StableAudioDirectProvider(MusicProvider):
    """The validated path: Stable Audio loaded in-process (Kaggle notebook).

    Identical to what Generalization V1/V2 always did — this class only
    packages the existing module-level calls behind the MusicProvider
    interface so a production provider can be swapped in instead.
    """

    def generate_fresh(self, prompt, duration, seed):
        return _generate(prompt=prompt, duration=float(duration), steps=STEPS,
                         cfg_scale=CFG_SCALE, seed=int(seed), batch_size=1)

    def generate_conditioned(self, prompt, duration, reference, noise, seed):
        reference = np.ascontiguousarray(np.asarray(reference, dtype=np.float32), dtype=np.float32)
        return _generate(prompt=prompt, duration=float(duration), steps=STEPS,
                         cfg_scale=CFG_SCALE, seed=int(seed), batch_size=1,
                         init_audio=(SR, reference), init_noise_level=float(noise))


_ACTIVE_PROVIDER = [StableAudioDirectProvider()]


def get_provider():
    return _ACTIVE_PROVIDER[0]


def set_provider(provider):
    """Installs a MusicProvider for all subsequent generate_motif() /
    generate_layer_phrases() calls. Returns the previous provider so callers
    can restore it (tests do this)."""
    previous = _ACTIVE_PROVIDER[0]
    _ACTIVE_PROVIDER[0] = provider
    return previous


def generate_fresh(prompt, duration, seed):
    """Back-compat direct call — goes through the active provider."""
    return get_provider().generate_fresh(prompt, duration, seed)


def generate_conditioned(prompt, duration, reference, noise, seed):
    """Back-compat direct call — goes through the active provider."""
    return get_provider().generate_conditioned(prompt, duration, reference, noise, seed)


# ============================================================
# MOTIF
# ============================================================

def motif_prompt(mood):
    return (
        "cinematic musical motif for a "
        f"{mood} story, "
        "dark modern psychological score, "
        "clear memorable three-note "
        "descending felt piano motif, "
        "motif repeats naturally, "
        "warm deep bass, "
        "restrained dark pad, "
        "subtle rhythmic pulse, "
        "emotionally intelligent and understated, "
        "instrumental only, "
        "no vocals, no speech, "
        "no choir, "
        "no trailer bombast"
    )


def _motif_seed(run_seed, story_index, take):
    # Take 0 keeps the original seed so the legacy path is unchanged.
    if take == 0:
        return derive_seed(run_seed, "motif", story_index)
    return derive_seed(run_seed, "motif", story_index, "take", take)


def rank_motif(motif, cue):
    """Higher is better. Tempo/key match against the cue, how many stable
    12 s reference windows the motif offers, and spectral fullness."""
    info = analyse_music(motif, SR, cue.tempo_bpm, cue.key, cue.mode)
    reference_samples = int(round((plan_grid(cue).generate_seconds + OVERHANG_SECONDS) * SR))
    step = int(round(1.0 * SR))
    scores = sorted(window_stability(motif[s:s + reference_samples])["score"]
                    for s in range(0, max(1, len(motif) - reference_samples + 1), step))
    stability = max(0.0, 1.0 - float(np.mean(scores[:4])) / 40.0) if scores else 0.0
    info["reference_stability"] = stability
    # How much of the motif is usable before the model's own ending fades it.
    prof = sustain_profile(motif)
    lv = np.asarray(prof["levels"])
    usable = float(np.mean(lv > np.median(lv[: max(1, len(lv) // 3)]) - 12.0)) if len(lv) else 0.0
    info["usable_fraction"] = usable
    # Dead air: share of half-seconds more than 20 dB under the motif's
    # typical level (stabs separated by silence are useless as a bed).
    dead = float(np.mean(lv < np.median(lv) - 20.0)) if len(lv) else 1.0
    info["dead_air_fraction"] = dead
    # The grid adopts the motif's own tempo/key afterwards, so what matters
    # is a clear pulse and the story's mode family (minor vs major), not
    # whether the model obeyed the BPM/key words.
    fam, _ = MODE_FAMILY.get(cue.mode, ("minor", 0))
    info["mode_family_match"] = float(info["key"].split()[-1] == fam)
    info["pulse_clarity"] = float(min(1.0, info["tempo_confidence"] / 0.5))
    info["rank"] = (30.0 * info["pulse_clarity"] + 20.0 * info["mode_family_match"]
                    + 20.0 * stability + 10.0 * info["fullness"] + 20.0 * usable - 60.0 * dead)
    return info


def generate_motif(story_index, mood, output_dir, run_seed, cue=None, takes=None,
                   return_report=False, log=print):
    if cue is None:
        prompt = motif_prompt(mood)
        takes = 1
    else:
        prompt = motif_prompt_from_cue(cue)
        takes = MOTIF_TAKES if takes is None else int(takes)
    provider = get_provider()
    best, rows = None, []
    take = 0
    while take < takes or (cue is not None and best is not None
                           and best[2].get("dead_air_fraction", 0.0) > 0.05
                           and take < takes + MAX_EXTRA_MOTIF_TAKES):
        seed = _motif_seed(run_seed, story_index, take)
        motif = f32(provider.generate_fresh(prompt, MOTIF_SECONDS, seed))
        row = {"take": take + 1, "seed": seed}
        if cue is not None:
            row.update(rank_motif(motif, cue))
            log(f"  motif take {take + 1}: rank={row['rank']:.1f} dead_air={row['dead_air_fraction']:.2f} "
                f"tempo={row['tempo_bpm']:.1f} key={row['key']} seam={row['seam']:.2f}")
        rows.append(row)
        if best is None or row.get("rank", 0.0) > best[2].get("rank", 0.0):
            best = (motif, seed, row)
        take += 1
    motif, seed, _ = best
    path = output_dir / "motif_seed_28s.wav"
    write_audio(path, motif, SR)
    if return_report:
        return motif, str(path), prompt, seed, rows
    return motif, str(path), prompt, seed


# ============================================================
# REFERENCE WINDOWS
# ============================================================

def envelope_levels(audio, window_seconds=0.25, hop_seconds=0.125):
    mono = to_mono(audio)
    window = max(1, int(round(window_seconds * SR)))
    hop = max(1, int(round(hop_seconds * SR)))
    starts, values = [], []
    for start in range(0, max(1, len(mono) - window + 1), hop):
        chunk = mono[start:start + window]
        value = np.sqrt(np.mean(chunk * chunk) + 1e-12)
        starts.append(start)
        values.append(20 * np.log10(value + 1e-9))
    return np.asarray(starts, dtype=np.int64), np.asarray(values, dtype=np.float32)


def window_stability(audio):
    _, levels = envelope_levels(audio)
    if len(levels) < 3:
        return {"score": 999.0, "valley": 99.0}
    median = float(np.median(levels))
    p10 = float(np.percentile(levels, 10))
    valley = median - p10
    dropout = float(np.mean(levels < (median - 10)))
    spread = float(np.percentile(levels, 95) - np.percentile(levels, 5))
    score = valley * 3.0 + dropout * 100.0 + max(0, spread - 24) * 1.5
    return {"score": float(score), "valley": float(valley), "dropout": dropout, "spread": spread}


def repair_reference(audio):
    audio = f32(audio)
    starts, levels = envelope_levels(audio, 0.10, 0.05)
    if len(levels) < 3:
        return audio
    median = float(np.median(levels))
    correction = np.minimum(np.maximum((median - 7.0) - levels, 0.0), 8.0)
    kernel = np.ones(12, dtype=np.float32) / np.float32(12)  # ~600 ms
    correction = np.convolve(correction, kernel, mode="same")
    correction_db = np.interp(np.arange(len(audio), dtype=np.float64), starts, correction)
    gain = (10 ** (correction_db / 20)).astype(np.float32)
    return normalize_peak(audio * gain[:, None], 0.85)


def choose_references(motif, seconds=None):
    """seconds: reference length (story-aware path generates whole-bar
    phrases, so its references match that length). Default: PHRASE_DURATION."""
    reference_samples = int(round((PHRASE_DURATION if seconds is None else seconds) * SR))
    step = int(round(0.50 * SR))
    candidates = []
    for start in range(0, max(1, len(motif) - reference_samples + 1), step):
        window = motif[start:start + reference_samples]
        if len(window) < reference_samples:
            continue
        candidates.append((window_stability(window)["score"], start))
    if not candidates:
        candidates = [(0.0, 0)]
    candidates.sort(key=lambda x: x[0])

    selected = []
    min_distance = int(round(2.0 * SR))
    for _, start in candidates:
        if all(abs(start - previous) >= min_distance for previous in selected):
            selected.append(start)
        if len(selected) == 4:
            break
    while len(selected) < 4:
        selected.append(candidates[len(selected) % len(candidates)][1])

    references = [
        f32(repair_reference(fit_length(motif[s:s + reference_samples], reference_samples)))
        for s in selected
    ]
    return references, selected


# ============================================================
# CHROMA + PHRASE QUALITY GATES
# ============================================================

def chroma_profile(audio, numpy_fallback=False):
    """librosa chroma (legacy behaviour: None when librosa is missing).
    numpy_fallback=True (story-aware path) uses music_analysis.chroma so the
    motif-similarity gate works on any machine."""
    try:
        import librosa
    except Exception:
        if not numpy_fallback:
            return None
        return f32(music_chroma(audio, SR))
    mono = to_mono(audio)
    if len(mono) < SR * 2:
        mono = np.pad(mono, (0, int(SR * 2) - len(mono)))
    chroma = librosa.feature.chroma_stft(y=mono, sr=SR, n_fft=4096, hop_length=1024)
    profile = f32(chroma.mean(axis=1))
    profile /= np.linalg.norm(profile) + 1e-9
    return profile


def phrase_quality(audio, motif_profile):
    audio = f32(audio)
    _, levels = envelope_levels(audio)
    if len(levels) < 4:
        return {"valid": False, "quality": -999.0, "failures": ["too_short"], "motif_similarity": None}

    median = float(np.median(levels))
    p05 = float(np.percentile(levels, 5))
    local_valley = median - p05
    bad = levels < (median - 10)
    bad_fraction = float(np.mean(bad))

    longest = current = 0
    for value in bad:
        current = current + 1 if value else 0
        longest = max(longest, current)
    longest_seconds = longest * 0.125

    spread = float(np.percentile(levels, 95) - np.percentile(levels, 5))
    edge = int(round(0.75 * SR))
    mono = to_mono(audio)
    head_drop = median - rms_db(mono[:edge])
    tail_drop = median - rms_db(mono[-edge:])

    similarity = None
    if motif_profile is not None:
        candidate_profile = chroma_profile(audio, numpy_fallback=True)
        if candidate_profile is not None:
            similarity = float(np.dot(motif_profile, candidate_profile))

    failures = []
    if local_valley > 11.0:
        failures.append("deep_local_valley")
    if bad_fraction > 0.08:
        failures.append("dropout_fraction")
    if longest_seconds > 0.75:
        failures.append("long_valley")
    if head_drop > 12.0:
        failures.append("head_collapse")
    if tail_drop > 12.0:
        failures.append("tail_collapse")
    if spread > 28.0:
        failures.append("level_swing")
    if similarity is not None and similarity < 0.88:
        failures.append("weak_motif")

    quality = 100.0
    quality -= local_valley * 2.5
    quality -= bad_fraction * 160.0
    quality -= max(0.0, longest_seconds - 0.25) * 20.0
    quality -= max(0.0, spread - 20.0) * 1.2
    if similarity is not None:
        quality += (similarity - 0.90) * 70.0

    return {
        "valid": len(failures) == 0,
        "quality": float(quality),
        "local_valley_db": float(local_valley),
        "dropout_fraction": bad_fraction,
        "longest_valley_seconds": float(longest_seconds),
        "head_drop_db": float(head_drop),
        "tail_drop_db": float(tail_drop),
        "level_spread_db": spread,
        "motif_similarity": similarity,
        "failures": failures,
    }


# ============================================================
# LAYER PHRASES
# ============================================================

def layer_prompts(mood):
    return {
        "core": (
            f"cinematic {mood} story "
            "underscore, "
            "same memorable three-note "
            "descending motif, "
            "felt piano, "
            "warm restrained bass, "
            "dark soft pad, "
            "sparse continuous musical bed, "
            "no silence, no pauses, "
            "instrumental only, no vocals"
        ),
        "pressure": (
            f"cinematic {mood} thriller "
            "pressure layer, "
            "same three-note descending "
            "felt piano motif, "
            "deeper bass pulse, "
            "subtle rhythmic movement, "
            "dark pad, "
            "increasing suspense, "
            "continuous musical bed, "
            "no silence, "
            "instrumental only, no vocals"
        ),
        "climax": (
            f"cinematic {mood} revelation "
            "climax layer, "
            "same three-note descending motif, "
            "wider piano register, "
            "deep bass pulse, "
            "controlled dramatic percussion, "
            "serious revelation, "
            "powerful but restrained, "
            "continuous musical bed, "
            "no silence, "
            "instrumental only, no vocals"
        ),
    }


def _fmt_similarity(value):
    return "n/a" if value is None else f"{value:.3f}"


def rank_phrase(audio, report, cue):
    """Story-aware take score: the original quality score (gates, valleys,
    motif chroma similarity) plus musical fit. Invalid takes keep their
    score but always lose to a valid one."""
    info = analyse_music(audio, SR, cue.tempo_bpm, cue.key, cue.mode)
    prof = sustain_profile(audio)
    info["sustain_drop_db"] = prof["drop_db"]
    info["sustain_spread_db"] = prof["spread_db"]
    sustain = max(0.0, 1.0 - max(0.0, prof["drop_db"]) / 12.0)
    info["sustain"] = sustain
    # Pulse steadiness at the take's own tempo (folded toward the cue), so a
    # take that's 2 % fast but rock steady still scores: the arranger
    # conforms it to the grid by beat slicing.
    own = info["tempo_bpm"]
    if own > 0:
        own = min((own * k for k in (0.5, 1.0, 2.0)), key=lambda b: abs(np.log(b / cue.tempo_bpm)))
    from_grid = music_grid_lock(audio, SR, own) if own > 0 else 0.0
    info["grid_lock"] = from_grid
    rank = (report["quality"] + 15.0 * info["seam"] + 15.0 * info["tempo_match"]
            + 10.0 * info["key_match"] + 5.0 * info["fullness"] + 20.0 * sustain + 10.0 * from_grid)
    info["rank"] = float(rank if report["valid"] else rank - 1000.0)
    return info


def generate_layer_phrases(story_index, mood, references, motif_profile, story_dir, run_seed, log=print,
                           cue=None, takes=None):
    prompts = layer_prompts(mood) if cue is None else layer_prompts_from_cue(cue)
    attempts = MAX_ATTEMPTS if cue is None else (PHRASE_TAKES if takes is None else int(takes))
    # Story-aware phrases are a whole number of bars (+ crossfade beat and
    # one alignment beat) at the cue's tempo, so the arranger can cut on bars.
    phrase_seconds = PHRASE_DURATION if cue is None else plan_grid(cue).generate_seconds
    phrase_samples = int(round(phrase_seconds * SR))
    model_seconds = phrase_seconds if cue is None else phrase_seconds + OVERHANG_SECONDS
    layer_phrases = {name: [] for name in LAYER_NAMES}
    quality_rows = []
    provider = get_provider()

    for layer in LAYER_NAMES:
        log(f"\nGenerating {layer}...")
        for phrase_index in range(PHRASES_PER_LAYER):
            best_audio, best_score, best_row, accepted = None, None, None, False
            ref_index = (phrase_index + LAYER_REFERENCE_OFFSET[layer]) % len(references)
            reference = references[ref_index]

            for attempt in range(attempts):
                seed = derive_seed(run_seed, "layer", layer, story_index, phrase_index, attempt)
                log(f"  phrase {phrase_index + 1}/{PHRASES_PER_LAYER} "
                    f"attempt {attempt + 1}/{attempts} ref={ref_index + 1}")
                try:
                    noise = LAYER_NOISE[layer] if cue is None else STORY_LAYER_NOISE[layer]
                    generated = provider.generate_conditioned(prompts[layer], model_seconds, reference,
                                                               noise, seed)
                    window_start = 0.0
                    if cue is not None:
                        generated, window_start = best_sustained_window(generated, phrase_seconds)
                    generated = normalize_peak(fit_length(generated, phrase_samples), 0.85)
                    report = phrase_quality(generated, motif_profile)
                    row = {"layer": layer, "phrase": phrase_index + 1,
                           "attempt": attempt + 1, "seed": seed,
                           "reference": ref_index + 1, **report}
                    if cue is None:
                        score = report["quality"]
                    else:
                        fit = rank_phrase(generated, report, cue)
                        fit["window_start_s"] = window_start
                        row["musical_fit"] = fit
                        score = fit["rank"]
                    quality_rows.append(row)
                    log(f"    quality={report['quality']:.1f} "
                        f"valley={report.get('local_valley_db', float('nan')):.1f}dB "
                        f"motif={_fmt_similarity(report.get('motif_similarity'))}"
                        + ("" if cue is None else
                           f" tempo={row['musical_fit']['tempo_bpm']:.1f} seam={row['musical_fit']['seam']:.2f}"))
                    row["selected"] = False
                    if best_score is None or score > best_score:
                        best_audio, best_score, best_row = generated, score, row
                    if report["valid"]:
                        accepted = True
                        if cue is None:
                            break          # legacy: first valid attempt wins
                except Exception as exc:
                    log(f"    generation error: {exc!r}")

            if best_audio is None:
                raise RuntimeError(f"No candidate for {layer} phrase {phrase_index + 1}")
            if not accepted:
                log("    fallback to best candidate")
            best_row["selected"] = True

            write_audio(story_dir / f"{layer}_phrase_{phrase_index + 1:02d}.wav", best_audio, SR)
            layer_phrases[layer].append(best_audio)

    return layer_phrases, quality_rows
