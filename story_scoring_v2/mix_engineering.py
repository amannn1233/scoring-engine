# ============================================================
# STORY SCORING LAB — MIX / MASTER ENGINEERING STAGE
#
# Everything a mix/mastering engineer would check or do between "the score
# is rendered" and "the file ships", in pure numpy + scipy.signal:
#
#   Metering (standards-based, not ad-hoc RMS):
#     - ITU-R BS.1770-4 K-weighted loudness: integrated (gated), momentary
#       (400 ms), short-term (3 s)
#     - EBU Tech 3342 loudness range (LRA)
#     - ITU-R BS.1770-4 Annex 2 true peak (4x oversampled, dBTP)
#     - PLR, crest factor, clipping runs, DC offset, noise floor
#     - stereo phase correlation, mid/side width, mono fold-down loss
#     - octave-band spectral balance
#     - dialogue masking margin (voice vs music in the 1-4 kHz speech band)
#
#   Processing (the dialogue mix + master bus):
#     - subsonic high-pass on music, rumble high-pass on voice
#     - stem gain staging: narration to a fixed dialogue anchor, music bed
#       trimmed relative to it, so the balance never depends on how hot the
#       TTS or the generator happened to render
#     - dialogue leveling: a slow phrase rider (evens out loud/quiet lines)
#       followed by a gentle 3:1 compressor on the narration
#     - dialogue-to-music-ratio ducking with lookahead (the music moves
#       BEFORE the line starts, the way a mixer rides the fader), hold
#       between words so it doesn't pump
#     - speech-band "carve": a dynamic presence dip in the music only while
#       the narrator is talking, so less broadband ducking is needed
#     - loudness normalization to a delivery spec + true-peak lookahead
#       limiter at the spec's ceiling (gain solved against ONE limiter pass,
#       never a stack of limiters)
#
# This module is ADDITIVE. scoring_engine.render_score()/measure_score()
# (the validated optimizer path) and the legacy score_ducked /
# voice_music_preview outputs are untouched.
# ============================================================

import math
from dataclasses import dataclass, asdict

import numpy as np
from scipy.ndimage import maximum_filter1d, minimum_filter1d, uniform_filter1d
from scipy.signal import butter, lfilter, resample_poly, sosfilt, welch


# ============================================================
# DELIVERY SPECS
# ============================================================

@dataclass(frozen=True)
class DeliverySpec:
    name: str
    integrated_lufs: float
    true_peak_dbtp: float
    tolerance_lu: float = 1.0


DELIVERY_SPECS = {
    # YouTube / Shorts / TikTok / Reels / Spotify normalise to ~-14 LUFS.
    "shorts": DeliverySpec("shorts", -14.0, -1.0),
    # Apple Podcasts / AES TD1004 spoken-word guidance.
    "podcast": DeliverySpec("podcast", -16.0, -1.0),
    # EBU R128 broadcast.
    "ebu_r128": DeliverySpec("ebu_r128", -23.0, -1.0, 0.5),
    # ATSC A/85 (US broadcast).
    "atsc_a85": DeliverySpec("atsc_a85", -24.0, -2.0, 2.0),
}
DEFAULT_DELIVERY = "shorts"


@dataclass(frozen=True)
class MixSettings:
    # Dialogue-to-music ratio while the narrator speaks: speech-gated voice
    # loudness minus the music bed's short-term loudness. 12-18 LU is the
    # usual range for narration over score; 15 keeps the score present.
    target_dmr_lu: float = 15.0
    max_duck_db: float = 12.0
    duck_lookahead_s: float = 0.20     # music starts dipping before the line
    duck_attack_s: float = 0.12
    duck_hold_s: float = 0.35          # bridges inter-word gaps (no pumping)
    duck_release_s: float = 0.90
    # Presence carve on the music under speech (dynamic peaking cut).
    carve_center_hz: float = 2500.0
    carve_q: float = 0.9
    carve_depth_db: float = 4.0
    # Filtering.
    music_highpass_hz: float = 30.0
    voice_highpass_hz: float = 70.0
    # Gain staging. The narration is normalised to a speech-gated anchor and
    # the score is trimmed so its integrated loudness sits this far under it
    # BEFORE ducking: between lines the music opens up to anchor - offset,
    # under lines the ducker takes it the rest of the way to target_dmr_lu.
    dialogue_anchor_lufs: float = -20.0
    music_open_lu_below_dialogue: float = 6.0
    max_music_trim_db: float = 30.0
    # Dialogue leveling: phrase rider (slow) then compressor (fast).
    level_window_s: float = 1.0
    level_max_ride_db: float = 9.0
    level_smooth_s: float = 0.4
    comp_threshold_over_speech_db: float = 3.0   # vs median active-frame RMS
    comp_ratio: float = 3.0
    comp_knee_db: float = 6.0
    comp_attack_s: float = 0.005
    comp_release_s: float = 0.12
    # Limiter.
    limiter_window_s: float = 0.015    # +-15 ms lookahead, ~30 ms ramps
    limiter_release_s: float = 0.08    # recovery after the ramp (no 30 Hz flutter)


# QC thresholds. Each one is a widely used engineering rule of thumb,
# not a law; they produce warnings in the report, never hard failures.
QC = {
    "clip_run_samples": 3,           # 3+ consecutive full-scale samples = clip
    "clip_level": 0.999,
    "dc_offset_max": 0.001,           # ~ -60 dBFS
    "min_plr_db": 8.0,                # below this the master is crushed
    "min_correlation": -0.2,          # 5th-pct window below this = phase problem
    "max_mono_loss_lu": 3.0,
    "min_masking_margin_db": 6.0,     # voice over music in the speech band
    "max_sub_share_db": -12.0,        # <40 Hz energy vs full band
    "max_lra_lu_narration": 15.0,
}

SPEECH_BAND = (1000.0, 4000.0)
OCTAVE_CENTERS = (31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000)

ABSOLUTE_GATE_LUFS = -70.0
SILENCE_LUFS = -120.0


# ============================================================
# SMALL HELPERS
# ============================================================

def _f64(audio):
    audio = np.asarray(audio, dtype=np.float64)
    return audio[:, None] if audio.ndim == 1 else audio


def _db(power):
    return 10.0 * np.log10(np.maximum(power, 1e-12))   # floor -120 dB


def _lufs_from_power(power):
    # BS.1770: L = -0.691 + 10 log10(sum_c G_c * z_c); G = 1 for L/R/C.
    power = np.asarray(power, dtype=np.float64)
    return np.where(power > 0, -0.691 + _db(power), SILENCE_LUFS)


def as_stereo(audio):
    audio = _f64(audio)
    if audio.shape[1] == 1:
        return np.repeat(audio, 2, axis=1)
    return audio[:, :2]


# ============================================================
# K-WEIGHTING (BS.1770-4) — coefficients derived for any sample rate
# ============================================================

def k_weighting_coeffs(sr):
    """Two biquads: high-shelf ('head' pre-filter) and RLB high-pass. At 48 kHz
    these reproduce the coefficient tables printed in BS.1770-4."""
    # Stage 1: high shelf.
    f0, gain_db, q = 1681.974450955533, 3.999843853973347, 0.7071752369554196
    k = math.tan(math.pi * f0 / sr)
    vh = 10.0 ** (gain_db / 20.0)
    vb = vh ** 0.4996667741545416
    a0 = 1.0 + k / q + k * k
    b1 = np.array([(vh + vb * k / q + k * k) / a0,
                   2.0 * (k * k - vh) / a0,
                   (vh - vb * k / q + k * k) / a0])
    a1 = np.array([1.0, 2.0 * (k * k - 1.0) / a0, (1.0 - k / q + k * k) / a0])
    # Stage 2: RLB high-pass.
    f0, q = 38.13547087602444, 0.5003270373238773
    k = math.tan(math.pi * f0 / sr)
    a0 = 1.0 + k / q + k * k
    b2 = np.array([1.0, -2.0, 1.0])
    a2 = np.array([1.0, 2.0 * (k * k - 1.0) / a0, (1.0 - k / q + k * k) / a0])
    return (b1, a1), (b2, a2)


def k_weight(audio, sr):
    audio = _f64(audio)
    (b1, a1), (b2, a2) = k_weighting_coeffs(sr)
    return lfilter(b2, a2, lfilter(b1, a1, audio, axis=0), axis=0)


def _block_power(weighted, sr, window_s, hop_s):
    """Mean-square per channel over sliding windows, summed over channels."""
    n = weighted.shape[0]
    win = int(round(window_s * sr))
    hop = int(round(hop_s * sr))
    summed = np.sum(weighted * weighted, axis=1)
    if n < win:
        return np.array([summed.mean() if n else 0.0]), np.array([0])
    csum = np.concatenate([[0.0], np.cumsum(summed)])
    starts = np.arange(0, n - win + 1, hop)
    return (csum[starts + win] - csum[starts]) / win, starts


def _gated_integrated(block_power):
    """Two-stage gate of BS.1770-4 applied to a set of block powers."""
    block_power = np.asarray(block_power, dtype=np.float64)
    loud = _lufs_from_power(block_power)
    stage1 = block_power[loud > ABSOLUTE_GATE_LUFS]
    if stage1.size == 0:
        return SILENCE_LUFS
    relative_gate = float(_lufs_from_power(stage1.mean())) - 10.0
    stage2 = block_power[(loud > ABSOLUTE_GATE_LUFS) & (loud > relative_gate)]
    if stage2.size == 0:
        return SILENCE_LUFS
    return float(_lufs_from_power(stage2.mean()))


def integrated_loudness(audio, sr, weighted=None):
    """Gated integrated loudness, LUFS (400 ms blocks, 75 % overlap)."""
    weighted = k_weight(audio, sr) if weighted is None else weighted
    power, _ = _block_power(weighted, sr, 0.4, 0.1)
    return _gated_integrated(power)


def loudness_curve(audio, sr, window_s, hop_s=0.1, weighted=None):
    """(times_s, LUFS) — window 0.4 = momentary, 3.0 = short-term."""
    weighted = k_weight(audio, sr) if weighted is None else weighted
    power, starts = _block_power(weighted, sr, window_s, hop_s)
    centers = (starts + int(round(window_s * sr)) / 2.0) / sr
    return centers, _lufs_from_power(power)


def loudness_range(audio, sr, weighted=None):
    """EBU Tech 3342 LRA: P95 - P10 of gated short-term loudness."""
    weighted = k_weight(audio, sr) if weighted is None else weighted
    power, _ = _block_power(weighted, sr, 3.0, 0.1)
    loud = _lufs_from_power(power)
    gated = power[loud > ABSOLUTE_GATE_LUFS]
    if gated.size < 2:
        return 0.0
    rel = float(_lufs_from_power(gated.mean())) - 20.0
    values = _lufs_from_power(gated)
    values = values[values > rel]
    if values.size < 2:
        return 0.0
    return float(np.percentile(values, 95) - np.percentile(values, 10))


# ============================================================
# TRUE PEAK (BS.1770-4 Annex 2)
# ============================================================

def true_peak_envelope(audio, sr, block=1 << 18, pad=64):
    """Per-sample inter-sample peak magnitude (max over all channels and the
    oversampled points belonging to each input sample). Processed in padded
    blocks so a 90 s file doesn't need a 4x float64 copy in memory."""
    audio = _f64(audio)
    factor = 4 if sr < 96000 else (2 if sr < 192000 else 1)
    n = audio.shape[0]
    if factor == 1 or n == 0:
        return np.max(np.abs(audio), axis=1) if n else np.zeros(0)
    out = np.empty(n)
    for start in range(0, n, block):
        stop = min(n, start + block)
        lo, hi = max(0, start - pad), min(n, stop + pad)
        up = resample_poly(audio[lo:hi], factor, 1, axis=0)
        up = np.abs(up[(start - lo) * factor:(stop - lo) * factor]).max(axis=1)
        out[start:stop] = up.reshape(-1, factor).max(axis=1)
    return out


def true_peak_dbtp(audio, sr):
    peak = float(np.max(true_peak_envelope(audio, sr))) if len(audio) else 0.0
    return 20.0 * math.log10(max(peak, 1e-12))


# ============================================================
# FILTERS
# ============================================================

def highpass(audio, sr, hz, order=2):
    sos = butter(order, hz, btype="highpass", fs=sr, output="sos")
    return sosfilt(sos, _f64(audio), axis=0)


def _bandpass_rbj(sr, f0, q):
    """RBJ constant 0 dB peak-gain band-pass."""
    w0 = 2.0 * math.pi * f0 / sr
    alpha = math.sin(w0) / (2.0 * q)
    a0 = 1.0 + alpha
    b = np.array([alpha, 0.0, -alpha]) / a0
    a = np.array([1.0, -2.0 * math.cos(w0) / a0, (1.0 - alpha) / a0])
    return b, a


def dynamic_presence_cut(audio, sr, gain_db_curve, f0, q):
    """Time-varying peaking EQ: y = x + (A(t) - 1) * BP(x). With A=1 the
    signal is untouched; with A<1 it dips around f0 by exactly 20log10(A)."""
    audio = _f64(audio)
    b, a = _bandpass_rbj(sr, f0, q)
    band = lfilter(b, a, audio, axis=0)
    amount = 10.0 ** (np.asarray(gain_db_curve, dtype=np.float64) / 20.0) - 1.0
    return audio + band * amount[:, None]


# ============================================================
# SPEECH ACTIVITY
# ============================================================

def speech_activity(voice_mono, sr, frame_s=0.02, hangover_s=0.25):
    """Boolean speech mask per frame + frame times. Threshold adapts to the
    recording: 12 dB over the noise floor, but never more than 30 dB under
    the loud speech level."""
    voice = highpass(np.asarray(voice_mono, dtype=np.float64), sr, 100.0)[:, 0]
    frame = int(round(frame_s * sr))
    n_frames = max(1, len(voice) // frame)
    power = np.mean(voice[: n_frames * frame].reshape(n_frames, frame) ** 2, axis=1)
    level = _db(power)
    floor = np.percentile(level, 10)
    loud = np.percentile(level, 95)
    threshold = min(floor + 12.0, loud - 6.0)
    threshold = max(threshold, loud - 30.0)
    active = level > threshold
    hang = max(1, int(round(hangover_s / frame_s)))
    active = maximum_filter1d(active.astype(np.uint8), size=2 * hang + 1) > 0
    times = (np.arange(n_frames) + 0.5) * frame / sr
    return active, times, {"threshold_db": float(threshold), "floor_db": float(floor),
                            "speech_p95_db": float(loud)}


# ============================================================
# DIALOGUE DUCKING (target DMR, lookahead, hold, asymmetric smoothing)
# ============================================================

def _smooth_asymmetric(target_db, frame_s, attack_s, release_s):
    """One-pole smoothing at frame rate: fast into gain reduction, slow out."""
    att = math.exp(-frame_s / max(attack_s, 1e-6))
    rel = math.exp(-frame_s / max(release_s, 1e-6))
    out = np.empty_like(target_db)
    current = 0.0
    for i, value in enumerate(target_db):
        coeff = att if value < current else rel
        current = coeff * current + (1.0 - coeff) * value
        out[i] = current
    return out


def _lookahead_min(values, look):
    """out[t] = min(values[t .. t+look])."""
    if look <= 0:
        return values
    padded = np.concatenate([values, np.full(look, values[-1])])
    return np.lib.stride_tricks.sliding_window_view(padded, look + 1).min(axis=1)


def dialogue_duck_curve(voice_mono, music, sr, settings=MixSettings()):
    """Per-sample gain (dB, <= 0) for the music bed, plus the per-sample
    speech-activity weight (0..1) used to drive the presence carve."""
    n = len(music)
    frame_s = 0.02
    active, times, vad = speech_activity(voice_mono, sr, frame_s)

    voice_w = k_weight(as_stereo(voice_mono), sr)   # as it sits in the mix
    voice_power, _ = _block_power(voice_w, sr, 0.4, 0.1)
    # Speech-gated narration loudness: the dialogue anchor of the mix.
    voice_lufs = _gated_integrated(voice_power)

    # Music short-term loudness (3 s) — slow enough that ducking follows
    # the score's arc, not its individual notes.
    m_times, m_lufs = loudness_curve(music, sr, 3.0, 0.1)
    music_st = np.interp(times, m_times, m_lufs)

    needed = settings.target_dmr_lu - (voice_lufs - music_st)
    target = -np.clip(needed, 0.0, settings.max_duck_db) * active

    # Hold the deepest duck across short gaps (no pumping between words),
    # then look ahead so the music is already moving when the line starts.
    hold = max(1, int(round(settings.duck_hold_s / frame_s)))
    look = int(round(settings.duck_lookahead_s / frame_s))
    target = _lookahead_min(minimum_filter1d(target, size=hold, mode="nearest"), look)
    smooth = _smooth_asymmetric(target, frame_s, settings.duck_attack_s, settings.duck_release_s)

    speaking = _lookahead_min(-active.astype(np.float64), look)
    activity = _smooth_asymmetric(speaking, frame_s, settings.duck_attack_s, settings.duck_release_s)
    sample_t = np.arange(n) / sr
    duck_db = np.interp(sample_t, times, smooth)
    weight = -np.interp(sample_t, times, activity)
    stats = {
        "voice_speech_lufs": float(voice_lufs),
        "speech_fraction": float(np.mean(active)),
        "average_duck_db_during_speech": float(np.mean(smooth[active])) if active.any() else 0.0,
        "max_duck_db": float(np.min(smooth)),
        "vad": vad,
    }
    return duck_db, weight, stats


# ============================================================
# STEM GAIN STAGING + DIALOGUE LEVELING
# ============================================================

def speech_gated_lufs(voice_mono, sr):
    """Narration loudness as it sits in a stereo mix (dual mono), gated."""
    weighted = k_weight(as_stereo(voice_mono), sr)
    power, _ = _block_power(weighted, sr, 0.4, 0.1)
    return _gated_integrated(power)


def _compress(voice, sr, settings, active, times):
    """Feed-forward soft-knee compressor on 5 ms RMS, threshold set from the
    narration's own median active-frame level. Control signal runs at 1 ms."""
    hop = max(1, int(round(0.001 * sr)))
    env = np.sqrt(np.maximum(uniform_filter1d(voice * voice, size=max(1, int(0.005 * sr)), mode="nearest")[::hop], 0.0))
    level = 20.0 * np.log10(np.maximum(env, 1e-9))
    ctl_t = np.arange(len(level)) * hop / sr
    speech = np.interp(ctl_t, times, active.astype(np.float64)) > 0.5
    if not speech.any():
        return voice, 0.0
    threshold = float(np.median(level[speech])) + settings.comp_threshold_over_speech_db
    over = level - threshold
    knee = settings.comp_knee_db
    slope = 1.0 - 1.0 / settings.comp_ratio
    gr = np.where(over <= -knee / 2, 0.0,
                  np.where(over >= knee / 2, slope * over,
                           slope * (over + knee / 2) ** 2 / (2 * knee)))
    smooth = -_smooth_asymmetric(-gr, hop / sr, settings.comp_attack_s, settings.comp_release_s)
    gain_db = np.interp(np.arange(len(voice)) / sr, ctl_t, -smooth)
    return voice * 10.0 ** (gain_db / 20.0), float(np.max(smooth))


def level_dialogue(voice_mono, sr, settings=MixSettings()):
    """Ride each phrase toward the anchor (like a vocal rider), compress the
    peaks, then normalise the narration to settings.dialogue_anchor_lufs."""
    voice = np.asarray(voice_mono, dtype=np.float64)
    frame_s = 0.02
    active, times, _ = speech_activity(voice, sr, frame_s)
    stats = {"rider_range_db": [0.0, 0.0], "compressor_max_gr_db": 0.0}
    if not active.any():
        return voice, stats

    # Phrase rider: speech-only K-weighted level over ~1 s, gaps don't count.
    weighted = k_weight(voice, sr)[:, 0]
    frame = int(round(frame_s * sr))
    nf = len(active)
    power = np.mean(weighted[: nf * frame].reshape(nf, frame) ** 2, axis=1) * active
    win = max(1, int(round(settings.level_window_s / frame_s)))
    num = uniform_filter1d(power, win, mode="nearest")
    den = uniform_filter1d(active.astype(np.float64), win, mode="nearest")
    local = _db(num / np.maximum(den, 1e-6))
    target = float(np.median(local[active]))
    ride = np.clip(target - local, -settings.level_max_ride_db, settings.level_max_ride_db)
    # Hold the ride through pauses (no gain moves on breaths / room tone).
    idx = np.where(active, np.arange(nf), 0)
    np.maximum.accumulate(idx, out=idx)
    ride = ride[idx]
    first = int(np.argmax(active))
    ride[:first] = ride[first]
    alpha = math.exp(-frame_s / settings.level_smooth_s)
    ride = lfilter([1 - alpha], [1, -alpha], ride, zi=[ride[0] * alpha])[0]
    voice = voice * 10.0 ** (np.interp(np.arange(len(voice)) / sr, times, ride) / 20.0)
    stats["rider_range_db"] = [float(ride.min()), float(ride.max())]

    voice, comp_gr = _compress(voice, sr, settings, active, times)
    stats["compressor_max_gr_db"] = comp_gr
    lufs = speech_gated_lufs(voice, sr)
    if lufs > SILENCE_LUFS:
        voice = voice * 10.0 ** ((settings.dialogue_anchor_lufs - lufs) / 20.0)
    return voice, stats


def gain_stage_music(music, sr, settings=MixSettings()):
    """Static trim so the score's integrated loudness sits
    music_open_lu_below_dialogue under the dialogue anchor. Static, so the
    score's own arc (the optimizer's section energy) is preserved exactly."""
    current = integrated_loudness(music, sr)
    if current <= SILENCE_LUFS:
        return music, 0.0
    trim = settings.dialogue_anchor_lufs - settings.music_open_lu_below_dialogue - current
    trim = float(np.clip(trim, -settings.max_music_trim_db, settings.max_music_trim_db))
    return music * 10.0 ** (trim / 20.0), trim


# ============================================================
# LIMITER (true-peak aware, lookahead, zero overshoot by construction)
# ============================================================

def _release_hold(gain, sr, release_s, block=64):
    """Slow the limiter's recovery without ever raising the gain above its
    input: per block, r = min(block_min(gain), r_prev recovering toward 1).
    Expanded back as a step per block, so r <= gain at every sample."""
    if release_s <= 0:
        return gain
    n = len(gain)
    nb = -(-n // block)
    padded = np.concatenate([gain, np.full(nb * block - n, gain[-1])])
    blocks = padded.reshape(nb, block).min(axis=1)
    if blocks.min() >= 1.0:
        return gain
    coeff = math.exp(-block / (release_s * sr))
    out = np.empty(nb)
    current = 1.0
    for i, value in enumerate(blocks):
        current = 1.0 - (1.0 - current) * coeff     # recover toward unity
        if value < current:
            current = value
        out[i] = current
    return np.repeat(out, block)[:n]


def true_peak_limit(audio, sr, ceiling_dbtp, window_s=0.015, release_s=0.08):
    """Lookahead brick-wall limiter driven by the true-peak envelope.

    gain = moving_average_{+-w}( release( min_{+-w}(required) ) ). Every value
    in the average window around a peak is <= a minimum over a window that
    contains that peak, so gain <= required there: the ceiling holds by
    construction, with smooth ~2w attack ramps and an exponential release
    (no per-sample gain ripple, no fast flutter on sustained material)."""
    audio = _f64(audio)
    ceiling = 10.0 ** (ceiling_dbtp / 20.0)
    required = np.minimum(1.0, ceiling / np.maximum(true_peak_envelope(audio, sr), 1e-12))
    if required.min() >= 1.0:
        return audio.copy(), 0.0
    size = 2 * max(1, int(round(window_s * sr))) + 1
    gain = minimum_filter1d(required, size=size, mode="nearest")
    gain = _release_hold(gain, sr, release_s)
    gain = uniform_filter1d(gain, size=size, mode="nearest")
    out = audio * gain[:, None]
    # Resampler ringing can leave a few hundredths of a dB; trim it.
    over = true_peak_dbtp(out, sr) - ceiling_dbtp
    if over > 0:
        out *= 10.0 ** (-(over + 0.01) / 20.0)
    return out, float(-20.0 * np.log10(max(gain.min(), 1e-12)))


def master_to_spec(audio, sr, spec, settings=MixSettings(), max_iter=8, tol_lu=0.05):
    """Find the ONE input gain g such that integrated(limit(g * x)) hits the
    spec, with the limiter always run on the original mix (re-limiting an
    already limited file stacks distortion and undershoots on peaky
    narration). Loudness after limiting is monotonic in g, so a secant search
    on dB converges in a few passes."""
    audio = _f64(audio)
    start = integrated_loudness(audio, sr)
    if start <= SILENCE_LUFS:
        return audio.copy(), 0.0

    def run(gain_db):
        out, gr = true_peak_limit(audio * 10.0 ** (gain_db / 20.0), sr, spec.true_peak_dbtp,
                                  settings.limiter_window_s, settings.limiter_release_s)
        return out, gr, integrated_loudness(out, sr) - spec.integrated_lufs

    g0 = spec.integrated_lufs - start
    out, gr, err0 = run(g0)
    best = (abs(err0), out, gr)
    g1 = g0 - err0
    for _ in range(max_iter - 1):
        if best[0] < tol_lu:
            break
        out, gr, err1 = run(g1)
        if abs(err1) < best[0]:
            best = (abs(err1), out, gr)
        slope = (err1 - err0) / (g1 - g0) if abs(g1 - g0) > 1e-9 else 1.0
        slope = min(max(slope, 0.05), 1.0)          # limiter only ever eats gain
        g0, err0 = g1, err1
        g1 = g1 - err1 / slope
    return best[1], best[2]


# ============================================================
# QC REPORT
# ============================================================

def _clip_runs(audio, level, run):
    hot = np.any(np.abs(audio) >= level, axis=1).astype(np.int8)
    if not hot.any():
        return 0
    edges = np.diff(np.concatenate([[0], hot, [0]]))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return int(np.sum((ends - starts) >= run))


def _correlation(audio, sr, window_s=0.4):
    left, right = audio[:, 0], audio[:, 1]
    win = int(round(window_s * sr))
    n = (len(left) // win) * win
    if n == 0:
        return 1.0, 1.0
    l = left[:n].reshape(-1, win)
    r = right[:n].reshape(-1, win)
    num = np.sum(l * r, axis=1)
    den = np.sqrt(np.sum(l * l, axis=1) * np.sum(r * r, axis=1))
    valid = den > 1e-9
    if not valid.any():
        return 1.0, 1.0
    corr = num[valid] / den[valid]
    return float(np.mean(corr)), float(np.percentile(corr, 5))


def octave_balance(audio, sr):
    """Octave-band energy relative to total (dB) from a Welch LTAS."""
    mono = _f64(audio).mean(axis=1)
    freqs, psd = welch(mono, fs=sr, nperseg=8192)
    total = np.sum(psd) + 1e-30
    bands = {}
    for fc in OCTAVE_CENTERS:
        lo, hi = fc / math.sqrt(2), fc * math.sqrt(2)
        sel = (freqs >= lo) & (freqs < min(hi, sr / 2))
        bands[f"{fc:g}Hz"] = float(_db(np.sum(psd[sel]) / total)) if sel.any() else None
    sub = float(_db(np.sum(psd[freqs < 40.0]) / total))
    # Tilt: slope of octave-band energy 125 Hz - 8 kHz, dB per octave.
    xs = [math.log2(fc) for fc in OCTAVE_CENTERS[2:9]]
    ys = [bands[f"{fc:g}Hz"] for fc in OCTAVE_CENTERS[2:9]]
    tilt = float(np.polyfit(xs, ys, 1)[0]) if all(y is not None for y in ys) else None
    return bands, sub, tilt


def masking_margin(voice_mono, music, sr, active_mask, mask_times):
    """Median voice-minus-music level in the 1-4 kHz speech band over
    speech frames. Below ~6 dB, consonants start getting lost."""
    sos = butter(4, SPEECH_BAND, btype="bandpass", fs=sr, output="sos")
    v = sosfilt(sos, _f64(voice_mono), axis=0).mean(axis=1)
    m = sosfilt(sos, _f64(music), axis=0).mean(axis=1)
    frame = int(round((mask_times[1] - mask_times[0]) * sr)) if len(mask_times) > 1 else len(v)
    n = min(len(active_mask), len(v) // frame)
    if n == 0 or not active_mask[:n].any():
        return None
    vp = np.mean(v[: n * frame].reshape(n, frame) ** 2, axis=1)
    mp = np.mean(m[: n * frame].reshape(n, frame) ** 2, axis=1)
    sel = active_mask[:n] & (vp > 0)
    # Integrate over 200 ms so single plosive frames don't dominate.
    vp = uniform_filter1d(vp, 10)
    mp = uniform_filter1d(mp, 10)
    return float(np.median(_db(vp[sel]) - _db(mp[sel])))


def qc_report(audio, sr, spec=None, voice_mono=None, music=None, label=""):
    """Engineering QC for one rendered file. Returns metrics + warnings."""
    stereo = as_stereo(audio)
    weighted = k_weight(stereo, sr)
    integrated = integrated_loudness(stereo, sr, weighted)
    _, short_term = loudness_curve(stereo, sr, 3.0, 0.1, weighted)
    _, momentary = loudness_curve(stereo, sr, 0.4, 0.1, weighted)
    tp = true_peak_dbtp(stereo, sr)
    sample_peak = 20.0 * math.log10(max(float(np.max(np.abs(stereo))), 1e-12))
    rms = 10.0 * math.log10(max(float(np.mean(stereo ** 2)), 1e-30))
    mono_fold = stereo.mean(axis=1, keepdims=True)
    mono_lufs = integrated_loudness(np.repeat(mono_fold, 2, axis=1), sr)
    corr_mean, corr_p5 = _correlation(stereo, sr)
    mid = stereo.mean(axis=1)
    side = (stereo[:, 0] - stereo[:, 1]) * 0.5
    width_db = float(_db(np.mean(side ** 2)) - _db(np.mean(mid ** 2)))
    bands, sub_share, tilt = octave_balance(stereo, sr)
    frame = int(round(0.05 * sr))
    nf = len(stereo) // frame
    frame_db = _db(np.mean(stereo[: nf * frame].reshape(nf, frame, 2) ** 2, axis=(1, 2))) if nf else np.array([-120.0])

    report = {
        "label": label,
        "integrated_lufs": integrated,
        "loudness_range_lu": loudness_range(stereo, sr, weighted),
        "max_short_term_lufs": float(np.max(short_term)),
        "max_momentary_lufs": float(np.max(momentary)),
        "true_peak_dbtp": tp,
        "sample_peak_dbfs": sample_peak,
        "plr_db": tp - integrated,
        "crest_factor_db": sample_peak - rms,
        "clip_runs": _clip_runs(stereo, QC["clip_level"], QC["clip_run_samples"]),
        "dc_offset": [float(x) for x in stereo.mean(axis=0)],
        "noise_floor_dbfs": float(np.percentile(frame_db, 5)),
        "phase_correlation_mean": corr_mean,
        "phase_correlation_p5": corr_p5,
        "side_to_mid_db": width_db,
        "mono_fold_loss_lu": integrated - mono_lufs,
        "octave_balance_db": bands,
        "sub40_share_db": sub_share,
        "spectral_tilt_db_per_octave": tilt,
    }
    if voice_mono is not None and music is not None:
        active, times, _ = speech_activity(voice_mono, sr)
        report["speech_band_masking_margin_db"] = masking_margin(voice_mono, music, sr, active, times)

    warnings = []
    if spec is not None:
        report["delivery_spec"] = asdict(spec)
        if abs(integrated - spec.integrated_lufs) > spec.tolerance_lu:
            warnings.append(f"integrated {integrated:.1f} LUFS outside {spec.name} "
                            f"{spec.integrated_lufs:.0f} +/-{spec.tolerance_lu:g}")
        if tp > spec.true_peak_dbtp + 0.05:
            warnings.append(f"true peak {tp:.2f} dBTP over {spec.true_peak_dbtp:.1f} ceiling")
        if report["plr_db"] < QC["min_plr_db"]:
            warnings.append(f"PLR {report['plr_db']:.1f} dB (over-limited)")
    if report["clip_runs"]:
        warnings.append(f"{report['clip_runs']} clipped runs")
    if max(abs(x) for x in report["dc_offset"]) > QC["dc_offset_max"]:
        warnings.append("DC offset")
    if corr_p5 < QC["min_correlation"]:
        warnings.append(f"phase correlation drops to {corr_p5:+.2f} (5th pct) (mono risk)")
    if report["mono_fold_loss_lu"] > QC["max_mono_loss_lu"]:
        warnings.append(f"mono fold-down loses {report['mono_fold_loss_lu']:.1f} LU")
    if sub_share > QC["max_sub_share_db"]:
        warnings.append(f"sub-40 Hz energy {sub_share:.1f} dB of total (wasted headroom)")
    margin = report.get("speech_band_masking_margin_db")
    if margin is not None and margin < QC["min_masking_margin_db"]:
        warnings.append(f"music masks speech band (margin {margin:.1f} dB)")
    report["warnings"] = warnings
    return report


# ============================================================
# THE MIX: voice + score -> delivery master
# ============================================================

def mix_and_master(voice_mono, music, sr, delivery=DEFAULT_DELIVERY, settings=MixSettings()):
    """Narration-led dialogue mix and master.

    Returns (master float32 (n, 2), report dict)."""
    spec = DELIVERY_SPECS[delivery] if isinstance(delivery, str) else delivery
    n = len(voice_mono)
    music = as_stereo(music)[:n]
    if len(music) < n:
        music = np.pad(music, ((0, n - len(music)), (0, 0)))

    voice = highpass(np.asarray(voice_mono, dtype=np.float64), sr, settings.voice_highpass_hz)[:, 0]
    music = highpass(music, sr, settings.music_highpass_hz)
    raw_voice_lufs = speech_gated_lufs(voice, sr)
    voice, level_stats = level_dialogue(voice, sr, settings)
    music, music_trim_db = gain_stage_music(music, sr, settings)

    duck_db, speech_weight, duck_stats = dialogue_duck_curve(voice, music, sr, settings)
    carve_db = -settings.carve_depth_db * speech_weight
    bed = dynamic_presence_cut(music, sr, carve_db, settings.carve_center_hz, settings.carve_q)
    bed = bed * (10.0 ** (duck_db / 20.0))[:, None]

    voice_st = np.repeat(voice[:, None], 2, axis=1)
    mix = voice_st + bed
    master, limiter_gr = master_to_spec(mix, sr, spec, settings)

    # Everything downstream of the mix bus gain is linear except the limiter,
    # so measure the achieved DMR on the pre-master stems.
    active, times, _ = speech_activity(voice, sr)
    bed_times, bed_st = loudness_curve(bed, sr, 3.0, 0.1)
    bed_under_speech = np.interp(times[active], bed_times, bed_st) if active.any() else np.array([np.nan])
    duck_stats["achieved_dmr_lu_median"] = float(duck_stats["voice_speech_lufs"] - np.nanmedian(bed_under_speech))
    duck_stats["target_dmr_lu"] = settings.target_dmr_lu
    duck_stats["gain_staging"] = {"raw_voice_speech_lufs": float(raw_voice_lufs),
                                  "music_trim_db": music_trim_db, **level_stats}
    duck_stats["carve"] = {"center_hz": settings.carve_center_hz, "q": settings.carve_q,
                           "depth_db": settings.carve_depth_db}
    duck_stats["limiter_max_gain_reduction_db"] = limiter_gr

    master = np.ascontiguousarray(master, dtype=np.float32)
    report = {
        "settings": asdict(settings),
        "dialogue_mix": duck_stats,
        "master_qc": qc_report(master, sr, spec, voice_mono=voice, music=bed, label="master"),
    }
    return master, report


def format_qc(report):
    """Short human-readable QC lines for logs."""
    r = report
    lines = [
        f"{r['label'] or 'audio'}: {r['integrated_lufs']:.1f} LUFS  LRA {r['loudness_range_lu']:.1f} LU  "
        f"TP {r['true_peak_dbtp']:+.2f} dBTP  PLR {r['plr_db']:.1f} dB",
        f"  corr {r['phase_correlation_mean']:+.2f} (p5 {r['phase_correlation_p5']:+.2f})  "
        f"mono loss {r['mono_fold_loss_lu']:.1f} LU  tilt {r['spectral_tilt_db_per_octave'] or 0:+.1f} dB/oct",
    ]
    if r.get("speech_band_masking_margin_db") is not None:
        lines.append(f"  speech-band margin {r['speech_band_masking_margin_db']:.1f} dB")
    for w in r["warnings"]:
        lines.append(f"  WARNING: {w}")
    return lines
