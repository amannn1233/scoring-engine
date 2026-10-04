# ============================================================
# STORY SCORING LAB — V2 EXACT-WAVEFORM SCORING ENGINE
#
# Pure numpy. No torch, no Stable Audio, no file I/O.
#
# The single most important property of this module:
#
#   render_score()  is the ONLY function that builds the music mix
#   measure_score() is the ONLY function that measures it
#
# The optimizer, the final render and the manifest all call these same two
# functions, so the optimizer's reported metrics ARE the final waveform's
# metrics (bit-identical), not an estimate of them.
# ============================================================

import math
import time
from itertools import permutations

import numpy as np


# ============================================================
# CONSTANTS (unchanged from V8 / Generalization V1)
# ============================================================

SR = 44100

PHRASE_DURATION = 12.0
PHRASES_PER_LAYER = 3

PHRASE_CROSSFADE = 1.25
SEQUENCE_CROSSFADE = 2.25

LAYER_RMS_DB = -24.0
MIX_PEAK = 0.93

MAX_DUCK_DB = 5.0
DUCK_ATTACK_SECONDS = 0.080
DUCK_RELEASE_SECONDS = 1.000
DUCK_FRAME_SECONDS = 0.050
DUCK_HOP_SECONDS = 0.010
DUCKED_PEAK = 0.96
PREVIEW_PEAK = 0.96

LAYER_NAMES = ("core", "pressure", "climax")

STORY_LABELS = (
    "HOOK",
    "SETUP / CONFLICT",
    "CONSEQUENCE",
    "HIDDEN DANGER",
    "DISCOVERY",
    "FORGERY REVELATION",
    "AFTERMATH / RESOLUTION",
)

SECTION_TARGETS_DB = np.array(
    [-26.0, -24.8, -23.9, -22.7, -22.0, -20.0, -25.3],
    dtype=np.float32,
)

# Gain automation (dB) at the 8 story boundaries b0..b7.
LAYER_AUTOMATION_DB = {
    "core": (-1.0, -0.5, 0.0, 0.0, 0.5, 1.0, -1.0, -4.0),
    "pressure": (-18.0, -16.0, -13.0, -8.0, -4.0, -1.5, -8.0, -20.0),
    "climax": (-32.0, -32.0, -32.0, -30.0, -21.0, -7.0, -1.5, -34.0),
}

# Objective (unchanged).
BOUNDARY_TOLERANCE_DB = 2.5
REVELATION_MIN_LIFT_DB = 1.0
RELEASE_MIN_DROP_DB = 2.0
LOCAL_VALLEY_LIMIT_DB = 9.0
JUMP_WINDOW_SECONDS = 0.5
VALLEY_WINDOW_SECONDS = 0.5

W_ENERGY = 1.20
W_BOUNDARY = 8.0
W_REVELATION = 8.0
W_RELEASE = 5.0
W_LOCAL = 1.50

# Search.
STAGE1_PHASE_STEP = 1.0
STAGE1_PASSES = 2
STAGE2_PHASE_DELTAS = (-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0)
STAGE2_PASSES = 2

# Optimizer-vs-final agreement required to call the run aligned.
ALIGNMENT_TOLERANCE_DB = 0.01


# ============================================================
# BASIC AUDIO HELPERS
# ============================================================

def f32(x):
    return np.ascontiguousarray(np.asarray(x, dtype=np.float32), dtype=np.float32)


def fit_length(audio, samples):
    audio = f32(audio)
    if len(audio) >= samples:
        return f32(audio[:samples])
    pad = [(0, samples - len(audio))] + [(0, 0)] * (audio.ndim - 1)
    return f32(np.pad(audio, pad, mode="constant"))


def to_mono(audio):
    """Bit-identical to audio.mean(axis=1) for stereo, ~10x faster."""
    audio = f32(audio)
    if audio.ndim == 1:
        return audio
    if audio.shape[1] == 2:
        return f32((audio[:, 0] + audio[:, 1]) * np.float32(0.5))
    return f32(audio.mean(axis=1))


def peak_abs(audio):
    audio = f32(audio)
    if audio.size == 0:
        return 0.0
    return float(max(float(audio.max()), -float(audio.min())))


def normalize_peak(audio, peak=0.96):
    # Never mutates the caller's array (the V1/V2 version could, through
    # np.asarray aliasing).
    audio = f32(audio)
    maximum = peak_abs(audio) + 1e-9
    if maximum > peak:
        return f32(audio * np.float32(peak / maximum))
    return audio.copy()


def db_from_mean_power(mean_power):
    # Identical formula to rms_db(): 20*log10(sqrt(mean + 1e-12) + 1e-9)
    return 20.0 * np.log10(np.sqrt(np.asarray(mean_power) + 1e-12) + 1e-9)


def rms_db(audio):
    mono = to_mono(audio)
    value = np.sqrt(np.mean(mono * mono) + 1e-12)
    return float(20.0 * np.log10(value + 1e-9))


def set_rms_db(audio, target_db):
    audio = f32(audio)
    gain = 10.0 ** ((target_db - rms_db(audio)) / 20.0)
    return f32(audio * np.float32(gain))


def to_numpy_audio(x):
    """Any model output / list / tensor / array -> (samples, 2) float32."""
    if isinstance(x, (tuple, list)):
        x = x[0]
    if hasattr(x, "detach"):
        x = x.detach().float().cpu().numpy()
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 3:
        x = x[0]
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim == 2:
        if x.shape[0] <= 8 and x.shape[1] > x.shape[0]:
            x = x.T
        elif x.shape[1] <= 8:
            pass
        else:
            x = x.T
    if x.shape[1] == 1:
        x = np.repeat(x, 2, axis=1)
    if x.shape[1] > 2:
        x = x[:, :2]
    return normalize_peak(x)


# ============================================================
# CROSSFADE / SEQUENCE / REPEAT
# ============================================================

def crossfade_join(a, b, seconds):
    a = f32(a)
    b = f32(b)
    fade = int(round(seconds * SR))
    fade = min(fade, len(a) // 3, len(b) // 3)
    if fade <= 0:
        return f32(np.concatenate([a, b], axis=0))
    x = np.linspace(0.0, 1.0, fade, dtype=np.float32)
    fi = np.sin(x * np.pi / 2.0).astype(np.float32)
    fo = np.cos(x * np.pi / 2.0).astype(np.float32)
    blend = a[-fade:] * fo[:, None] + b[:fade] * fi[:, None]
    return f32(np.concatenate([a[:-fade], blend, b[fade:]], axis=0))


def build_sequence(items):
    result = f32(items[0])
    for item in items[1:]:
        result = crossfade_join(result, item, PHRASE_CROSSFADE)
    return result


def phase_to_samples(phase_seconds, sequence_samples):
    return int(round((phase_seconds % (sequence_samples / SR)) * SR))


def loop_sequence(sequence, min_length):
    """Crossfade-repeat `sequence` until it is at least min_length long.

    Each crossfade only rewrites the last SEQUENCE_CROSSFADE of the running
    result, so a longer loop has exactly the same samples as a shorter one
    everywhere a repeat_sequence() slice can read. That lets the optimizer
    build one loop per (layer, order) and slice every phase from it, while
    staying bit-identical to repeat_sequence().
    """
    sequence = f32(sequence)
    result = sequence.copy()
    while len(result) < min_length:
        result = crossfade_join(result, sequence, SEQUENCE_CROSSFADE)
    return result


def repeat_sequence(sequence, target_samples, phase_seconds=0.0, looped=None):
    sequence = f32(sequence)
    phase = phase_to_samples(phase_seconds, len(sequence))
    required = target_samples + phase + len(sequence)
    if looped is None or len(looped) < required:
        looped = loop_sequence(sequence, required)
    return fit_length(looped[phase:phase + target_samples], target_samples)


def prepare_layer_sequence(phrases, order):
    sequence = build_sequence([phrases[i] for i in order])
    return set_rms_db(sequence, LAYER_RMS_DB)


# ============================================================
# STORY AUTOMATION
# ============================================================

def layer_gain_curves(boundaries, total_samples):
    """Linear per-sample gains for each layer (float32)."""
    boundaries = [float(x) for x in boundaries]
    time_axis = np.arange(total_samples) / SR
    gains = {}
    for name in LAYER_NAMES:
        env_db = f32(np.interp(time_axis, boundaries, LAYER_AUTOMATION_DB[name]))
        gains[name] = f32(10 ** (env_db / 20.0))
    return gains


def render_layer(sequence, phase_seconds, gain, total_samples, looped=None):
    layer = repeat_sequence(sequence, total_samples, phase_seconds, looped=looped)
    return f32(layer * gain[:, None])


def mix_layers(layer_signals):
    """Canonical summation: zeros + core + pressure + climax, then peak."""
    total = np.zeros_like(layer_signals[0])
    for signal in layer_signals:
        total += signal
    return normalize_peak(total, MIX_PEAK)


def render_score(layer_phrases, orders, phases, boundaries, total_samples,
                 gains=None, sequences=None):
    """THE renderer. Used by the optimizer and by the final export."""
    if gains is None:
        gains = layer_gain_curves(boundaries, total_samples)
    signals = []
    for name in LAYER_NAMES:
        order = tuple(orders[name])
        if sequences is not None and (name, order) in sequences:
            sequence = sequences[(name, order)]
        else:
            sequence = prepare_layer_sequence(layer_phrases[name], order)
        signals.append(render_layer(sequence, phases[name], gains[name], total_samples))
    return mix_layers(signals)


# ============================================================
# MEASUREMENT (THE ONLY METRIC CODE)
# ============================================================

def section_sample_ranges(boundaries):
    ranges = []
    for i in range(7):
        start = int(round(float(boundaries[i]) * SR))
        end = int(round(float(boundaries[i + 1]) * SR))
        ranges.append((start, max(start + 1, end)))
    return ranges


def measure_score(music, boundaries, targets=SECTION_TARGETS_DB):
    """Measure the exact waveform. Returns a JSON-friendly dict."""
    mono = to_mono(music)
    n = len(mono)
    ranges = section_sample_ranges(boundaries)
    half = int(round(JUMP_WINDOW_SECONDS * SR))
    window = int(round(VALLEY_WINDOW_SECONDS * SR))

    # Every window edge the metrics need; sums between consecutive edges are
    # taken in one float64 pass, then any window sum is a difference of the
    # cumulative edge sums (exact partition, no per-window re-scan).
    edges = {0, n}
    for start, end in ranges:
        edges.update(np.clip(np.arange(start, end, window), 0, n).tolist())
        edges.add(min(max(end, 0), n))
    for i in range(1, 7):
        center = int(round(float(boundaries[i]) * SR))
        edges.update((min(max(center - half, 0), n), min(max(center, 0), n), min(max(center + half, 0), n)))
    edges = np.array(sorted(edges), dtype=np.int64)
    power = np.square(mono, dtype=np.float64)
    seg = np.add.reduceat(power, edges[:-1]) if n > 0 else np.zeros(0)
    seg[edges[:-1] == edges[1:]] = 0.0
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    position = {int(e): i for i, e in enumerate(edges)}

    def total_to(index):
        return cum[position[min(max(int(index), 0), n)]]

    def window_db(start, end):
        start = max(0, min(n, start))
        end = max(0, min(n, end))
        if end <= start:
            return float(db_from_mean_power(0.0))
        return float(db_from_mean_power((total_to(end) - total_to(start)) / (end - start)))

    section_db = np.array([window_db(s, e) for s, e in ranges], dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    energy_error = float(np.mean((section_db - targets) ** 2))

    jumps = []
    for i in range(1, 7):
        center = int(round(float(boundaries[i]) * SR))
        jumps.append(window_db(center, center + half) - window_db(center - half, center))
    jumps = np.asarray(jumps, dtype=np.float64)
    boundary_penalty = float(np.mean(np.maximum(np.abs(jumps) - BOUNDARY_TOLERANCE_DB, 0.0) ** 2))

    peak_gap = float(section_db[5] - section_db[4])
    revelation_penalty = max(0.0, REVELATION_MIN_LIFT_DB - peak_gap) ** 2

    release_drop = float(section_db[6] - section_db[5])
    release_penalty = max(0.0, release_drop + RELEASE_MIN_DROP_DB) ** 2

    local_penalty = 0.0
    valley_depths = []
    for start, end in ranges:
        starts = np.arange(start, end, window)
        if len(starts) < 3:
            valley_depths.append(None)
            continue
        stops = np.minimum(starts + window, end)
        db = np.array([window_db(int(s), int(e)) for s, e in zip(starts, stops)])
        depth = float(np.median(db) - np.percentile(db, 10))
        valley_depths.append(depth)
        if depth > LOCAL_VALLEY_LIMIT_DB:
            local_penalty += (depth - LOCAL_VALLEY_LIMIT_DB) ** 2

    objective = (
        W_ENERGY * energy_error
        + W_BOUNDARY * boundary_penalty
        + W_REVELATION * revelation_penalty
        + W_RELEASE * release_penalty
        + W_LOCAL * local_penalty
    )

    return {
        "objective": float(objective),
        "section_db": [float(x) for x in section_db],
        "section_target_db": [float(x) for x in targets],
        "boundary_jumps_db": [float(x) for x in jumps],
        "max_abs_boundary_jump_db": float(np.max(np.abs(jumps))),
        "energy_error": energy_error,
        "boundary_penalty": boundary_penalty,
        "revelation_penalty": float(revelation_penalty),
        "release_penalty": float(release_penalty),
        "local_penalty": float(local_penalty),
        "local_valley_depth_db": valley_depths,
        "peak_gap_db": peak_gap,
        "release_drop_db": release_drop,
    }


# ============================================================
# OPTIMIZER — EXACT WAVEFORM, BOTH STAGES
# ============================================================

def normalize_layer_input(layers_audio):
    """Accepts {name: [phrase, ...]} (the real pipeline structure) or a
    single array per layer. Fixes the V1 'list has no attribute mean' error."""
    result = {}
    for name in LAYER_NAMES:
        value = layers_audio[name]
        if isinstance(value, (list, tuple)):
            if not value:
                raise ValueError(f"Layer '{name}' contains no audio phrases.")
            result[name] = [to_numpy_audio(p) for p in value]
        else:
            result[name] = [to_numpy_audio(value)]
        if len(result[name]) != PHRASES_PER_LAYER:
            raise ValueError(
                f"Layer '{name}' has {len(result[name])} phrases; expected {PHRASES_PER_LAYER}."
            )
    return result


def _clamp_phase(value):
    return float(min(max(value, 0.0), PHRASE_DURATION - 0.25))


def optimize_story(layers_audio, boundaries, total_samples=None, log=print):
    """Two-stage coordinate search, every candidate measured on the exact
    44.1 kHz mix produced by the same code as render_score().

    Stage 1: all 6 orders x phases 0..11 s (1 s grid), 2 passes.
    Stage 2: all 6 orders x phase +-1, +-0.5, +-0.25, 0 s, 2 passes.
    """
    started = time.time()
    phrases = normalize_layer_input(layers_audio)
    boundaries = [float(x) for x in boundaries]
    if total_samples is None:
        total_samples = int(round((boundaries[-1] - boundaries[0]) * SR))

    gains = layer_gain_curves(boundaries, total_samples)
    orders = list(permutations(range(PHRASES_PER_LAYER)))

    sequences = {
        (name, order): prepare_layer_sequence(phrases[name], order)
        for name in LAYER_NAMES
        for order in orders
    }

    max_phase_samples = int(round(PHRASE_DURATION * SR)) + 1

    def loop_for(name, order):
        sequence = sequences[(name, order)]
        return loop_sequence(sequence, total_samples + max_phase_samples + len(sequence))

    state_orders = {name: orders[0] for name in LAYER_NAMES}
    state_phases = {name: 0.0 for name in LAYER_NAMES}
    evaluations = [0]

    def signal(name, order, phase, looped=None):
        return render_layer(sequences[(name, order)], phase, gains[name], total_samples, looped=looped)

    def search_layer(name, phase_candidates):
        fixed = {
            other: signal(other, state_orders[other], state_phases[other])
            for other in LAYER_NAMES if other != name
        }
        best = None
        for order in orders:
            looped = loop_for(name, order)
            for phase in phase_candidates:
                active = signal(name, order, phase, looped=looped)
                music = mix_layers([active if n == name else fixed[n] for n in LAYER_NAMES])
                metrics = measure_score(music, boundaries)
                evaluations[0] += 1
                if best is None or metrics["objective"] < best[0]:
                    best = (metrics["objective"], order, float(phase), metrics)
            del looped
        state_orders[name] = best[1]
        state_phases[name] = best[2]
        return best

    def current_metrics():
        music = render_score(phrases, state_orders, state_phases, boundaries,
                             total_samples, gains=gains, sequences=sequences)
        return measure_score(music, boundaries)

    history = []

    stage1_grid = [float(x) for x in np.arange(0.0, PHRASE_DURATION, STAGE1_PHASE_STEP)]
    for pass_index in range(STAGE1_PASSES):
        for name in LAYER_NAMES:
            best = search_layer(name, stage1_grid)
            log(f"  Stage 1 pass {pass_index + 1} {name:<9} score={best[0]:.4f} "
                f"order={best[1]} phase={best[2]:.2f}s")
        stage1 = current_metrics()
        history.append({"stage": 1, "pass": pass_index + 1, "objective": stage1["objective"]})
        log(f"Stage 1 pass {pass_index + 1}: score={stage1['objective']:.4f} "
            f"peak={stage1['peak_gap_db']:+.2f}dB boundary={stage1['boundary_penalty']:.4f}")
    stage1_orders = {k: list(v) for k, v in state_orders.items()}
    stage1_phases = dict(state_phases)

    for pass_index in range(STAGE2_PASSES):
        for name in LAYER_NAMES:
            center = state_phases[name]
            candidates = sorted(dict.fromkeys(_clamp_phase(center + d) for d in STAGE2_PHASE_DELTAS))
            best = search_layer(name, candidates)
            log(f"  Stage 2 pass {pass_index + 1} {name:<9} score={best[0]:.4f} "
                f"order={best[1]} phase={best[2]:.2f}s")
        stage2 = current_metrics()
        history.append({"stage": 2, "pass": pass_index + 1, "objective": stage2["objective"]})
        log(f"Stage 2 exact pass {pass_index + 1}: score={stage2['objective']:.4f} "
            f"peak={stage2['peak_gap_db']:+.2f}dB boundary={stage2['boundary_penalty']:.4f}")

    return {
        "optimizer_version": "V2_EXACT_WAVEFORM",
        "orders": {k: list(v) for k, v in state_orders.items()},
        "phases_seconds": {k: float(v) for k, v in state_phases.items()},
        "objective": stage2["objective"],
        "metrics": stage2,
        "stage1": {
            "orders": stage1_orders,
            "phases_seconds": stage1_phases,
            "objective": stage1["objective"],
            "metrics": stage1,
        },
        "history": history,
        "evaluations": evaluations[0],
        "seconds": round(time.time() - started, 2),
        "total_samples": int(total_samples),
    }


# ============================================================
# ALIGNMENT CHECK (optimizer report vs final waveform)
# ============================================================

def alignment_report(optimizer_metrics, final_metrics, tolerance_db=ALIGNMENT_TOLERANCE_DB):
    def diffs(key):
        a = np.asarray(optimizer_metrics[key], dtype=np.float64)
        b = np.asarray(final_metrics[key], dtype=np.float64)
        return np.abs(a - b)

    section = diffs("section_db")
    jumps = diffs("boundary_jumps_db")
    peak = float(abs(optimizer_metrics["peak_gap_db"] - final_metrics["peak_gap_db"]))
    objective = float(abs(optimizer_metrics["objective"] - final_metrics["objective"]))
    boundary = float(abs(optimizer_metrics["boundary_penalty"] - final_metrics["boundary_penalty"]))
    worst_db = float(max(section.max(), jumps.max(), peak))
    return {
        "tolerance_db": tolerance_db,
        "max_section_db_diff": float(section.max()),
        "max_boundary_jump_db_diff": float(jumps.max()),
        "peak_gap_db_diff": peak,
        "objective_diff": objective,
        "boundary_penalty_diff": boundary,
        "worst_db_diff": worst_db,
        "aligned": bool(worst_db <= tolerance_db and objective <= 0.01),
    }


# ============================================================
# ADAPTIVE VOICE DUCKING (unchanged behaviour)
# ============================================================

def duck_envelope(voice_mono, total_samples):
    voice_mono = f32(voice_mono)
    frame = int(round(DUCK_FRAME_SECONDS * SR))
    hop = int(round(DUCK_HOP_SECONDS * SR))
    starts = np.arange(0, max(1, len(voice_mono) - frame), hop, dtype=np.int64)

    csum = np.concatenate([[0.0], np.cumsum(voice_mono.astype(np.float64) ** 2)])
    ends = np.minimum(starts + frame, len(voice_mono))
    lengths = np.maximum(ends - starts, 1)
    levels = np.where(
        ends > starts,
        db_from_mean_power((csum[ends] - csum[starts]) / lengths),
        -80.0,
    ).astype(np.float32)

    p10, p20, p50, p80, p90 = np.percentile(levels, [10, 20, 50, 80, 90])
    activity = np.clip((levels - p20) / max(1e-6, p80 - p20), 0.0, 1.0) ** 1.25
    target = -MAX_DUCK_DB * activity

    attack = math.exp(-hop / (SR * DUCK_ATTACK_SECONDS))
    release = math.exp(-hop / (SR * DUCK_RELEASE_SECONDS))
    smooth = np.zeros_like(target)
    current = 0.0
    for i, value in enumerate(target):
        coeff = attack if value < current else release
        current = coeff * current + (1.0 - coeff) * value
        smooth[i] = current

    env = np.interp(np.arange(total_samples), starts, smooth,
                    left=smooth[0], right=smooth[-1]).astype(np.float32)
    env = np.clip(env, -MAX_DUCK_DB, 0.0)
    stats = {
        "max_duck_db": MAX_DUCK_DB,
        "attack_seconds": DUCK_ATTACK_SECONDS,
        "release_seconds": DUCK_RELEASE_SECONDS,
        "average_duck_db": float(np.mean(env)),
        "max_actual_duck_db": float(np.min(env)),
        "voice_p10_db": float(p10),
        "voice_p20_db": float(p20),
        "voice_p50_db": float(p50),
        "voice_p80_db": float(p80),
        "voice_p90_db": float(p90),
    }
    return env, stats


def apply_ducking(music, env_db):
    return normalize_peak(f32(music) * (10 ** (env_db / 20.0))[:, None], DUCKED_PEAK)


def make_preview(voice_mono, ducked):
    voice_stereo = np.repeat(f32(voice_mono)[:, None], 2, axis=1)
    return normalize_peak(voice_stereo + fit_length(ducked, len(voice_stereo)), PREVIEW_PEAK)
