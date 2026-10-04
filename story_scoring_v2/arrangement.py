# ============================================================
# ARRANGEMENT — the story-aware scoring stage.
#
# The legacy renderer (scoring_engine.render_score) loops 12 s phrases with
# 1.25 s / 2.25 s crossfades, so the beat grid slips at every join and the
# section automation moves at arbitrary times. This stage arranges the same
# kind of material the way a music editor cuts a score to picture:
#
#   - a tempo grid from the cue sheet: phrases are generated to a whole
#     number of bars (+ crossfade beat + spare bar), conformed to the exact
#     grid tempo (pitch-preserving WSOLA, only when the model drifted a
#     little), cut to start on their downbeat, and
#     joined with a crossfade exactly one beat long, so every join lands on
#     a bar line and the groove never slips;
#   - story sections snapped to bar lines, with the twist / climax /
#     resolution taken from the cue sheet's hit points;
#   - layer automation derived from the story's own intensity arc (not a
#     fixed table), every move ramping into a bar line;
#   - story hits: a two-beat "breath" (the bed drops out) right before the
#     twist lands, a reversed swell plus a two-bar build into the climax
#     and a low cinematic impact on its downbeat;
#   - a real outro: the score keeps playing past the last word and fades
#     over two bars to end on a downbeat, instead of being cut off.
#
# Measurement reuses scoring_engine.measure_score (the validated measurer),
# with per-story section targets from the arc. Pure numpy.
# ============================================================

import math
from dataclasses import dataclass, asdict
from itertools import permutations

import numpy as np

# <local-imports>
from scoring_engine import (
    SR, LAYER_NAMES, LAYER_RMS_DB, MIX_PEAK, f32, normalize_peak, set_rms_db, to_mono,
    measure_score,
)
from music_analysis import onset_envelope, estimate_tempo, grid_lock
# </local-imports>


# ============================================================
# TEMPO GRID
# ============================================================

@dataclass(frozen=True)
class Grid:
    bpm: float
    beats_per_bar: int
    phrase_bars: int
    xfade_beats: int

    @property
    def beat(self):
        return 60.0 / self.bpm

    @property
    def bar(self):
        return self.beat * self.beats_per_bar

    @property
    def phrase_seconds(self):
        return self.phrase_bars * self.bar

    @property
    def xfade_seconds(self):
        return self.xfade_beats * self.beat

    @property
    def generate_seconds(self):
        # Phrase + crossfade tail + one spare bar: room to find the downbeat
        # and to conform a slightly-off tempo without running out of audio.
        return self.phrase_seconds + self.xfade_seconds + self.bar

    def to_dict(self):
        d = asdict(self)
        d.update(beat=self.beat, bar=self.bar, phrase_seconds=self.phrase_seconds,
                 xfade_seconds=self.xfade_seconds, generate_seconds=self.generate_seconds)
        return d


def plan_grid(cue, target_phrase_seconds=10.5, max_generate_seconds=20.0):
    """Bars per phrase: the even count (2, 4, 6, 8) closest to ~10 s, so
    loops are musically square, within what the model generates well."""
    bpm = float(cue.tempo_bpm)
    bar = 4 * 60.0 / bpm
    best = None
    for bars in (2, 4, 6, 8):
        grid = Grid(bpm, 4, bars, 1)
        if grid.generate_seconds > max_generate_seconds and bars > 2:
            continue
        err = abs(grid.phrase_seconds - target_phrase_seconds)
        if best is None or err < best[0]:
            best = (err, grid)
    return best[1]


def snap(grid, t, how="nearest"):
    bars = t / grid.bar
    k = {"nearest": round, "floor": math.floor, "ceil": math.ceil}[how](bars)
    return k * grid.bar


# ============================================================
# PHRASE PREPARATION (beat alignment)
# ============================================================

def wsola_stretch(audio, rate, win_s=0.046, tol_s=0.012):
    """Time-stretch without changing pitch (WSOLA). rate > 1 = longer.
    Each output frame takes the input segment, within +-tol of its nominal
    position, that best continues the previous frame's waveform, so
    transients and pitch survive small tempo corrections intact."""
    audio = np.asarray(audio, dtype=np.float64)
    if abs(rate - 1.0) < 1e-6:
        return audio.copy()
    win = int(round(win_s * SR)) // 2 * 2
    hop_out = win // 2
    hop_in = hop_out / rate
    tol = int(round(tol_s * SR))
    window = np.hanning(win)
    mono = audio.mean(axis=1)
    n_out = int(len(audio) * rate)
    frames = n_out // hop_out
    out = np.zeros((frames * hop_out + win, audio.shape[1]))
    norm = np.zeros(frames * hop_out + win)
    prev = 0
    for k in range(frames):
        nominal = int(round(k * hop_in))
        if k == 0:
            pos = 0
        else:
            target = prev + hop_out           # natural continuation of the last frame
            lo, hi = max(0, nominal - tol), min(len(audio) - win, nominal + tol)
            if hi <= lo or target + win > len(audio):
                pos = min(max(lo, nominal), max(0, len(audio) - win))
            else:
                ref = mono[target:target + win]
                cand = mono[lo:hi + win]
                corr = np.correlate(cand, ref, mode="valid")
                pos = lo + int(np.argmax(corr))
        if pos + win > len(audio):
            break
        out[k * hop_out:k * hop_out + win] += audio[pos:pos + win] * window[:, None]
        norm[k * hop_out:k * hop_out + win] += window
        prev = pos
    out /= np.maximum(norm, 1e-3)[:, None]
    return out[:n_out]


def _beat_phase(env, fps, beat_s):
    beat_frames = beat_s * fps
    best = (-1.0, 0.0)
    for k in range(64):
        phase = k / 64.0 * beat_s
        idx = (phase * fps + beat_frames * np.arange(int(len(env) / beat_frames) + 1)).astype(int)
        idx = idx[idx < len(env)]
        score = sum(env[np.clip(idx + d, 0, len(env) - 1)].sum() for d in (-1, 0, 1))
        if score > best[0]:
            best = (score, phase)
    return best[1]


def beat_slice_conform(audio, src_bpm, dst_bpm, keep_s=0.045, snap_s=0.03, xf_s=0.004):
    """Tempo-conform the way a music editor does (REX-style beat slicing):
    find every beat in the source, keep each beat's first `keep_s` (the
    transient) untouched, and WSOLA-stretch only the rest of the beat so it
    fills exactly one destination beat. Transients land exactly on the new
    grid and are never smeared or doubled."""
    audio = np.asarray(audio, dtype=np.float64)
    env, fps = onset_envelope(audio, SR)
    src_beat, dst_beat = 60.0 / src_bpm, 60.0 / dst_bpm
    phase = _beat_phase(env, fps, src_beat)
    # Sample-accurate onset strength: rise of a 5 ms RMS envelope.
    mono = audio.mean(axis=1)
    w = max(1, int(0.005 * SR))
    rms = np.sqrt(np.maximum(np.convolve(mono * mono, np.ones(w) / w, mode="same"), 0.0))
    rise = np.maximum(np.diff(rms, prepend=rms[0]), 0.0)
    rise = np.convolve(rise, np.ones(w) / w, mode="same")
    snap = int(round(snap_s * SR))
    # The onset envelope's frames lag the audio by about one analysis window.
    lag = 1536.0 / SR
    starts = []
    t = phase + lag
    while t * SR < len(audio):
        c = int(round(t * SR))
        lo, hi = max(0, c - snap), min(len(rise), c + snap + 1)
        if hi > lo and rise[lo:hi].max() > 0:
            c = lo + int(np.argmax(rise[lo:hi]))       # snap to the actual onset
        starts.append(max(0, c - int(0.002 * SR)))    # 2 ms pre-roll keeps the attack whole
        t += src_beat
    starts = sorted(set(starts))
    keep = int(round(keep_s * SR))
    dst = int(round(dst_beat * SR))
    xf = max(1, int(round(xf_s * SR)))
    ramp = np.linspace(0.0, 1.0, xf)
    out = [audio[:starts[0]]] if starts and starts[0] > 0 else []
    for i, a in enumerate(starts):
        b = starts[i + 1] if i + 1 < len(starts) else min(len(audio), a + int(round(src_beat * SR)))
        seg = audio[a:b]
        if len(seg) <= keep + xf * 2:
            piece = np.pad(seg, ((0, max(0, dst + xf - len(seg))), (0, 0)))[:dst + xf]
        else:
            body = seg[keep:]
            rate = (dst - keep + 2 * xf) / len(body)
            stretched = wsola_stretch(body, rate, win_s=0.02, tol_s=0.006) if abs(rate - 1) > 1e-4 else body
            # dst - keep + 2*xf: one xf is eaten by the keep/body joint, one by
            # the crossfade into the next slice, so each slice spans exactly dst.
            need = dst - keep + 2 * xf
            stretched = np.pad(stretched, ((0, max(0, need - len(stretched))), (0, 0)))[:need]
            joint = seg[keep - xf:keep] * (1 - ramp)[:, None] + stretched[:xf] * ramp[:, None]
            piece = np.concatenate([seg[:keep - xf], joint, stretched[xf:]])
        if out and xf < len(piece):
            prev = out[-1]
            if len(prev) >= xf:
                tail = prev[-xf:] * (1 - ramp)[:, None] + piece[:xf] * ramp[:, None]
                out[-1] = prev[:-xf]
                piece = np.concatenate([tail, piece[xf:]])
        out.append(piece)
    return np.concatenate(out) if out else audio.copy()


def conform_tempo(audio, grid, max_correction=0.06):
    """Bring a phrase to exactly the grid tempo when the model drifted a
    little (half/double-time readings are folded first), by beat slicing.
    Returns (audio, measured_bpm, stretch_rate)."""
    bpm, conf = estimate_tempo(audio, SR)
    if bpm <= 0 or conf < 0.2:
        return audio, bpm, 1.0
    folded = min((bpm * k for k in (0.5, 1.0, 2.0)), key=lambda b: abs(math.log(b / grid.bpm)))
    rate = folded / grid.bpm                 # 87 BPM -> 86 BPM grid: 1.0116x longer
    if abs(rate - 1.0) < 0.002 or abs(rate - 1.0) > max_correction:
        return audio, bpm, 1.0
    if grid_lock(audio, SR, folded) < 0.4:
        return audio, bpm, 1.0             # no steady pulse (pads, rubato): nothing to slice
    return f32(beat_slice_conform(audio, folded, grid.bpm)), bpm, rate


def _low_onsets(audio):
    from scipy.signal import butter, sosfilt
    low = sosfilt(butter(2, 200.0, fs=SR, output="sos"), np.asarray(audio, dtype=np.float64), axis=0)
    return onset_envelope(low, SR)


def beat_offset(audio, grid):
    """Seconds to trim so the phrase starts on a DOWNBEAT: first the
    click-train phase that best matches the onset envelope, then which of
    the four beats carries the most low-end weight (kick / bass on 'one')."""
    env, fps = onset_envelope(audio, SR)
    if len(env) < 8 or not env.any():
        return 0.0
    beat_frames = grid.beat * fps
    best = (-1.0, 0.0)
    for k in range(48):
        phase = k / 48.0 * grid.beat
        idx = (phase * fps + beat_frames * np.arange(int(len(env) / beat_frames) + 1)).astype(int)
        idx = idx[idx < len(env)]
        # Sum a +-1 frame neighbourhood so frame quantisation doesn't hide a hit.
        score = sum(env[np.clip(idx + d, 0, len(env) - 1)].sum() for d in (-1, 0, 1))
        if score > best[0]:
            best = (score, phase)
    phase = best[1]
    low, lfps = _low_onsets(audio)
    bar_frames = grid.bar * lfps
    weights = []
    for j in range(grid.beats_per_bar):
        start = (phase + j * grid.beat) * lfps
        idx = (start + bar_frames * np.arange(int(len(low) / bar_frames) + 1)).astype(int)
        idx = idx[idx < len(low)]
        weights.append(sum(low[np.clip(idx + d, 0, len(low) - 1)].sum() for d in (-1, 0, 1)))
    j = int(np.argmax(weights)) if max(weights) > 0 else 0
    return float(phase + j * grid.beat)


def prepare_phrase(audio, grid, info=None):
    """Conform tempo, align to the downbeat, then cut to phrase + crossfade
    tail (exact samples). info (dict) receives what was done."""
    audio, bpm, rate = conform_tempo(f32(audio), grid)
    off_s = beat_offset(audio, grid)
    off = int(round(off_s * SR))
    need = int(round((grid.phrase_seconds + grid.xfade_seconds) * SR))
    out = audio[off:off + need]
    short = need - len(out)
    if short > 0:
        out = np.pad(out, ((0, short), (0, 0)))
    if info is not None:
        info.update(measured_bpm=float(bpm), stretch_rate=float(rate), downbeat_offset_s=off_s,
                    padded_samples=int(max(short, 0)))
    return f32(out)


# ============================================================
# BEDS (one layer laid on the grid)
# ============================================================

def build_bed(phrases, order, grid, total, offset_bars=0):
    """Phrases placed every `phrase_bars` bars in `order`, cycling. Each
    new phrase enters ON its bar line (10 ms de-click, so its downbeat is
    intact) while the previous phrase rings out under it for one beat on a
    cosine fade: the way an editor cuts music on the downbeat.
    offset_bars shifts which bar of the cycle sits at t = 0."""
    period = int(round(grid.phrase_seconds * SR))
    xf = int(round(grid.xfade_seconds * SR))
    declick = max(1, int(round(0.010 * SR)))
    fin = 0.5 - 0.5 * np.cos(np.pi * (np.arange(declick) + 0.5) / declick)
    fout = np.cos(0.5 * np.pi * (np.arange(xf) + 0.5) / max(xf, 1)) ** 2
    out = np.zeros((total, 2), dtype=np.float64)
    shift = int(round(offset_bars * grid.bar * SR))
    k = 0
    while True:
        s0 = k * period - shift
        if s0 >= total:
            break
        p = np.asarray(phrases[order[k % len(order)]], dtype=np.float64)
        seg = p[: period + xf].copy()
        if len(seg) < period + xf:
            seg = np.pad(seg, ((0, period + xf - len(seg)), (0, 0)))
        seg[:declick] *= fin[:, None]
        seg[period:] *= fout[:, None]
        lo, hi = max(0, s0), min(total, s0 + period + xf)
        if hi > lo:
            out[lo:hi] += seg[lo - s0:hi - s0]
        k += 1
    return out


# ============================================================
# STORY SECTIONS, TARGETS AND AUTOMATION
# ============================================================

def story_boundaries(cue, base_boundaries, grid, duration):
    """8 bar-snapped boundaries. The first half keeps story_analysis's
    keyword sections; the turn comes from the cue sheet:
      b5 (revelation) = the twist: the moment the story turns,
      b6 (aftermath)  = the resolution, at least 2 bars after the climax
                        and 4 bars after the twist,
      b4 (discovery)  = the build: up to 8 bars before the twist.
    Sections stay at least one bar long."""
    b = [float(x) for x in base_boundaries]
    hp = cue.hit_points
    bar = grid.bar
    end = snap(grid, duration, "floor")
    b5 = snap(grid, hp["twist"])
    b6 = max(snap(grid, hp["resolution"]), snap(grid, hp["climax"]) + 2 * bar, b5 + 4 * bar)
    b6 = min(b6, end - 2 * bar)
    b4 = b5 - min(8 * bar, max(2 * bar, snap(grid, 0.5 * (b5 - b[3]), "floor")))
    raw = [0.0, snap(grid, b[1]), snap(grid, b[2]), snap(grid, b[3]), b4, b5, b6]
    out = [0.0]
    for i in range(1, 7):
        out.append(max(raw[i], out[-1] + bar))
    out.append(max(duration, out[-1] + bar))
    # If enforcing gaps pushed sections past the end, compress from the back.
    for i in range(6, 0, -1):
        if out[i] > out[i + 1] - bar:
            out[i] = max(out[i - 1] + 0.5 * bar, out[i + 1] - bar)
    # Keep b1..b3 before the build if the keyword sections ran late.
    for i in (3, 2, 1):
        if out[i] > out[i + 1] - bar:
            out[i] = max(out[i - 1] + 0.5 * bar, out[i + 1] - bar)
    return out


def section_intensity(cue):
    """Per-section 0..1 intensity, reshaped so the story turns: the
    revelation (climax section) is the peak and the aftermath releases."""
    a = np.clip(np.asarray(cue.arc, dtype=np.float64), 0.0, 1.0)
    a = 0.15 + 0.7 * (a - a.min()) / max(1e-6, a.max() - a.min())
    # A score under narration opens present but modest (the hook is told,
    # not scored at full tilt) and moves in steps a mixer would ride, not
    # jumps: at most 0.22 (~2 dB of target) between neighbouring sections
    # until the turn, which is allowed to hit.
    a[0] = float(np.clip(a[0], 0.3, 0.55))
    for i in range(1, 5):
        a[i] = float(np.clip(a[i], a[i - 1] - 0.22, a[i - 1] + 0.22))
    a[4] = max(a[4], a[3])
    a[5] = max(a[5], a[:5].max() + 0.12, 0.9)
    a[4] = min(a[4], a[5] - 0.08)
    a[6] = float(np.clip(a[6], a[5] - 0.5, a[5] - 0.35))
    return np.clip(a, 0.05, 1.0)


def section_targets(intensity):
    """Relative dB targets: 9 dB of story arc between quietest and peak,
    with the scoring engine's revelation lift / release drop honoured."""
    t = -28.0 + 9.0 * np.asarray(intensity)
    t[5] = max(t[5], t[4] + 1.5)
    t[6] = min(t[6], t[5] - 3.0)
    return t


def _layer_levels(intensity):
    core = -2.0 + 2.0 * np.minimum(intensity / 0.6, 1.0)          # always present
    pressure = -22.0 + 21.0 * np.clip((intensity - 0.2) / 0.7, 0, 1) ** 1.3
    climax = np.full(7, -40.0)
    climax[5] = -1.0
    climax[4] = -40.0 + 20.0 * np.clip((intensity[4] - 0.5) / 0.4, 0, 1)
    return {"core": core, "pressure": pressure, "climax": climax}


def layer_gains(cue, boundaries, grid, total, music_end, section_trim_db=None):
    """Per-sample linear gains. Each section holds its level; changes ride
    over two bars, from the bar line before the section boundary to the bar
    line after it, like a mixer pushing a fader through the downbeat. The
    climax layer instead builds over the two bars INTO the turn, so the turn
    hits at full weight. section_trim_db: per-section level trims applied to
    every layer (the arc fader ride solved by fit_section_trims)."""
    levels = _layer_levels(section_intensity(cue))
    if section_trim_db is not None:
        levels = {k: v + np.asarray(section_trim_db) for k, v in levels.items()}
    t = np.arange(total) / SR
    gains = {}
    for name in LAYER_NAMES:
        db = np.full(total, levels[name][0])
        for i in range(1, 7):
            if i == 5 and name == "climax":
                ramp, end = 2.0 * grid.bar, boundaries[i]
            else:
                ramp, end = 2.0 * grid.bar, boundaries[i] + grid.bar
            start = end - ramp
            frac = np.clip((t - start) / ramp, 0.0, 1.0)
            frac = 0.5 - 0.5 * np.cos(np.pi * frac)               # S-curve
            db = np.where(t >= start, db + (levels[name][i] - levels[name][i - 1]) * frac, db)
        gains[name] = 10.0 ** (db / 20.0)
    # Outro: pressure/climax out over the last 4 bars, everything fades over
    # the last 2 bars to silence exactly on the final bar line.
    fade_all = np.clip((music_end - t) / (2.0 * grid.bar), 0.0, 1.0)
    fade_hi = np.clip((music_end - 2.0 * grid.bar - t) / (2.0 * grid.bar), 0.0, 1.0)
    for name in LAYER_NAMES:
        g = gains[name] * np.sin(fade_all * np.pi / 2.0)
        if name != "core":
            g = g * np.sin(fade_hi * np.pi / 2.0)
        gains[name] = g
    return gains


# ============================================================
# STORY HITS
# ============================================================

def _breath_curve(grid, twist_t, total, depth_db=-14.0):
    """Bed ducks out over the two beats before the twist's bar line and
    snaps back (30 ms) on it: the classic pre-line 'drop'."""
    t = np.arange(total) / SR
    start = twist_t - 2.0 * grid.beat
    down = np.clip((t - start) / grid.beat, 0.0, 1.0)
    back = np.clip((t - (twist_t - 0.03)) / 0.03, 0.0, 1.0)
    amount = np.where(t < twist_t - 0.03, down, 1.0 - back)
    return 10.0 ** (depth_db * amount / 20.0)


def _impact(grid, level_rms, seed=0):
    """Low cinematic impact: sub sweep 58 -> 36 Hz + soft filtered noise
    thump, ~1.5 bars of decay. Level set relative to the bed."""
    n = int(round(min(4.0, 1.5 * grid.bar) * SR))
    t = np.arange(n) / SR
    f = 36.0 + 22.0 * np.exp(-t / 0.18)
    phase = 2 * np.pi * np.cumsum(f) / SR
    boom = np.sin(phase) * np.exp(-t / (0.45 * grid.bar)) * (1 - np.exp(-t / 0.004))
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(n)
    a = math.exp(-2 * math.pi * 900.0 / SR)
    from scipy.signal import lfilter
    noise = lfilter([1 - a], [1, -a], noise) * np.exp(-t / 0.09)
    x = boom + 0.35 * noise / (np.max(np.abs(noise)) + 1e-9)
    # Felt more than heard: ~bed level, mostly sub energy under the narration.
    x *= level_rms * 10 ** (-2.0 / 20.0) / (np.sqrt(np.mean(x[: int(0.3 * SR)] ** 2)) + 1e-12)
    return np.stack([x, x], axis=1)


def _reverse_swell(music, grid, climax_t):
    """The climax's first bar, reversed, swelling in over the bar before
    it (reverse-reverb style riser built from the score itself)."""
    bar = int(round(grid.bar * SR))
    c = int(round(climax_t * SR))
    if c - bar < 0 or c + bar > len(music):
        return None, 0
    src = music[c:c + bar][::-1].copy()
    env = (np.arange(bar) / bar) ** 2.5
    return src * env[:, None] * 10 ** (-3.0 / 20.0), c - bar


# ============================================================
# DEAD-AIR GUARD
# ============================================================

def dead_air_regions(mix, min_s=0.4, below_db=18.0, frame_s=0.05):
    """(start_s, end_s) runs where the score falls more than below_db under
    its own median level for at least min_s."""
    mono = np.asarray(mix, dtype=np.float64).mean(axis=1)
    f = int(round(frame_s * SR))
    n = len(mono) // f
    if n < 4:
        return []
    lv = 10 * np.log10(np.mean(mono[: n * f].reshape(n, f) ** 2, axis=1) + 1e-12)
    low = lv < np.median(lv) - below_db
    out, i = [], 0
    while i < n:
        if low[i]:
            j = i
            while j < n and low[j]:
                j += 1
            if (j - i) * frame_s >= min_s:
                out.append((i * frame_s, j * frame_s))
            i = j
        else:
            i += 1
    return out


def _freeze(source, seconds, seed):
    """Spectral freeze: the source's average magnitude spectrum, random
    phase, overlap-added. Same harmony and timbre as the music around the
    gap, with no attacks: it reads as the chord ringing on."""
    n_fft, hop = 4096, 1024
    rng = np.random.default_rng(seed)
    win = np.hanning(n_fft)
    mono_frames = []
    for ch in range(source.shape[1]):
        x = source[:, ch]
        if len(x) < n_fft:
            x = np.pad(x, (0, n_fft - len(x)))
        frames = np.lib.stride_tricks.sliding_window_view(x, n_fft)[::hop] * win
        mono_frames.append(np.abs(np.fft.rfft(frames, axis=1)).mean(axis=0))
    total = int(round(seconds * SR)) + n_fft
    out = np.zeros((total, source.shape[1]))
    norm = np.zeros(total)
    for start in range(0, total - n_fft + 1, hop):
        phase = np.exp(2j * np.pi * rng.random(len(mono_frames[0])))
        for ch, mag in enumerate(mono_frames):
            out[start:start + n_fft, ch] += np.fft.irfft(mag * phase, n_fft) * win
        norm[start:start + n_fft] += win ** 2
    out /= np.maximum(norm, 1e-3)[:, None]
    return out[: int(round(seconds * SR))]


def fill_dead_air(mix, protect=(), level_db=-4.0, fade_s=0.25):
    """Fill any stretch where the bed dropped out (real model outputs can
    contain silences) with a spectral freeze of the second before it, so
    the narration never sits on dead air. `protect` lists (start, end)
    regions that are silent on purpose (the breath, the outro)."""
    out = np.array(mix, dtype=np.float64, copy=True)
    filled = []
    for k, (a, b) in enumerate(dead_air_regions(mix)):
        if any(a < pe and b > ps for ps, pe in protect):
            continue
        src0 = max(0, int(round((a - 1.0) * SR)))
        src = out[src0:int(round(a * SR))]
        if len(src) < int(0.25 * SR) or not np.any(src):
            continue
        g0, g1 = max(0.0, a - fade_s), min(len(out) / SR, b + fade_s)
        length = g1 - g0
        fz = _freeze(src, length, seed=k)
        rms_src = np.sqrt(np.mean(src ** 2) + 1e-12)
        fz *= rms_src * 10 ** (level_db / 20) / (np.sqrt(np.mean(fz ** 2)) + 1e-12)
        nfade = int(round(fade_s * SR))
        env = np.ones(len(fz))
        env[:nfade] = np.sin(0.5 * np.pi * np.arange(nfade) / nfade) ** 2
        env[-nfade:] = env[:nfade][::-1]
        i0 = int(round(g0 * SR))
        out[i0:i0 + len(fz)] += fz[: len(out) - i0] * env[: len(out) - i0, None]
        filled.append((a, b))
    return out, filled


# ============================================================
# RENDER + MEASURE + OPTIMISE
# ============================================================

@dataclass
class ArrangementPlan:
    orders: dict
    offsets_bars: dict

    def to_dict(self):
        return {"orders": {k: list(v) for k, v in self.orders.items()},
                "offsets_bars": dict(self.offsets_bars)}


def music_length(grid, voice_seconds):
    """End on the first bar line at least one bar after the last word."""
    return snap(grid, voice_seconds + grid.bar, "ceil")


def _beds(phrases, grid, total, max_cached=8):
    """All (layer, order, offset) beds, RMS-normalised per layer/order.
    float32 and a small LRU: a long story's full candidate set (3 layers x
    6 orders x offsets x minutes of stereo) does not fit in RAM."""
    from collections import OrderedDict
    cache = OrderedDict()

    def get(name, order, offset):
        key = (name, tuple(order), int(offset))
        if key in cache:
            cache.move_to_end(key)
            return cache[key]
        bed = build_bed(phrases[name], order, grid, total, offset)
        cache[key] = np.asarray(set_rms_db(bed, LAYER_RMS_DB), dtype=np.float32)
        while len(cache) > max_cached:
            cache.popitem(last=False)
        return cache[key]
    return get


def render_arrangement(phrases, plan, cue, grid, boundaries, voice_seconds, hits=True, bed_cache=None,
                       section_trim_db=None):
    """THE story-aware renderer (used by the optimiser and the export)."""
    end_t = music_length(grid, voice_seconds)
    total = int(round(end_t * SR))
    get = bed_cache or _beds(phrases, grid, total)
    gains = layer_gains(cue, boundaries, grid, total, end_t, section_trim_db)
    mix = np.zeros((total, 2), dtype=np.float64)
    for name in LAYER_NAMES:
        mix += get(name, plan.orders[name], plan.offsets_bars[name]) * gains[name][:, None]
    protect = [(boundaries[5] - 2.0 * grid.beat, boundaries[5] + 0.05), (end_t - 2.0 * grid.bar, end_t)]
    mix, _ = fill_dead_air(mix, protect)
    if hits:
        mix = apply_hits(mix, cue, grid, boundaries)
    return normalize_peak(f32(mix), MIX_PEAK)


def apply_hits(mix, cue, grid, boundaries):
    """The turn is the revelation boundary (b5): breath before it, reversed
    swell into it, impact on its downbeat."""
    total = len(mix)
    turn_t = boundaries[5]
    out = mix.copy()
    swell, at = _reverse_swell(mix, grid, turn_t)
    if swell is not None:
        out[at:at + len(swell)] += swell
    # The swell rises, then everything is sucked out for the last beat: the
    # silence makes the turn's downbeat land.
    out *= _breath_curve(grid, turn_t, total)[:, None]
    c = int(round(turn_t * SR))
    bed_rms = float(np.sqrt(np.mean(mix[max(0, c - int(grid.bar * SR)):c] ** 2) + 1e-12))
    boom = _impact(grid, bed_rms)
    hi = min(total, c + len(boom))
    if hi > c:
        out[c:hi] += boom[: hi - c]
    return out


def measure_arrangement(music, boundaries, targets):
    """measure_score with the story's own targets, offset-invariant (the
    arc's SHAPE matters, not its absolute level, which mastering sets)."""
    first = measure_score(music, boundaries, targets)
    offset = float(np.mean(np.asarray(first["section_db"]) - np.asarray(targets)))
    return measure_score(music, boundaries, np.asarray(targets) + offset)


def optimise_arrangement(phrases, cue, grid, boundaries, voice_seconds, log=print, passes=2):
    """Coordinate search over phrase order (all permutations) and start
    offset (whole bars within one cycle) per layer, every candidate measured
    on the exact render (hits off while searching, on for the final)."""
    end_t = music_length(grid, voice_seconds)
    total = int(round(end_t * SR))
    get = _beds(phrases, grid, total)
    targets = section_targets(section_intensity(cue))
    n_phr = len(phrases[LAYER_NAMES[0]])
    orders = list(permutations(range(n_phr)))
    cycle_bars = grid.phrase_bars * n_phr
    offsets = list(range(0, cycle_bars, max(1, grid.phrase_bars // 2)))
    plan = ArrangementPlan({n: orders[0] for n in LAYER_NAMES}, {n: 0 for n in LAYER_NAMES})
    gains = layer_gains(cue, boundaries, grid, total, end_t)
    evals = 0
    best_metrics = None
    for _ in range(passes):
        for name in LAYER_NAMES:
            fixed = sum(get(o, plan.orders[o], plan.offsets_bars[o]) * gains[o][:, None]
                        for o in LAYER_NAMES if o != name)
            best = None
            for order in orders:
                for off in offsets:
                    mix = fixed + get(name, order, off) * gains[name][:, None]
                    m = measure_arrangement(normalize_peak(f32(mix), MIX_PEAK), boundaries, targets)
                    evals += 1
                    if best is None or m["objective"] < best[0]:
                        best = (m["objective"], order, off, m)
            plan.orders[name], plan.offsets_bars[name] = best[1], best[2]
            best_metrics = best[3]
    log(f"Arrangement search: {evals} exact renders, objective {best_metrics['objective']:.4f}")
    return plan, best_metrics, targets, get


def fit_section_trims(prepared, plan, cue, grid, boundaries, voice_seconds, targets, get,
                      rounds=4, limit_db=4.0):
    """After the phrase search, ride the per-section level toward the arc's
    targets (offset-invariant), damped and clamped, keeping the trim that
    measured best. This is the mixer's fader pass over the arrangement."""
    trims = np.zeros(7)
    best = None
    for _ in range(rounds + 1):
        music = render_arrangement(prepared, plan, cue, grid, boundaries, voice_seconds, hits=False,
                                   bed_cache=get, section_trim_db=trims)
        m = measure_arrangement(music, boundaries, targets)
        if best is None or m["objective"] < best[0]:
            best = (m["objective"], trims.copy(), m)
        err = np.asarray(m["section_db"]) - np.asarray(m["section_target_db"])
        trims = np.clip(trims - 0.7 * err, -limit_db, limit_db)
    return best[1], best[2]


def arrange_story(phrases, cue, base_boundaries, voice_seconds, log=print):
    """Full story-aware scoring stage. Returns (music, report)."""
    grid = plan_grid(cue)
    prep_info = {n: [{} for _ in phrases[n]] for n in LAYER_NAMES}
    prepared = {n: [prepare_phrase(p, grid, prep_info[n][i]) for i, p in enumerate(phrases[n])]
                for n in LAYER_NAMES}
    boundaries = story_boundaries(cue, base_boundaries, grid, voice_seconds)
    plan, search_metrics, targets, get = optimise_arrangement(
        prepared, cue, grid, boundaries, voice_seconds, log=log)
    trims, search_metrics = fit_section_trims(prepared, plan, cue, grid, boundaries, voice_seconds, targets, get)
    log(f"Section fader ride (dB): {' '.join(f'{x:+.1f}' for x in trims)}, "
        f"objective {search_metrics['objective']:.4f}")
    music = render_arrangement(prepared, plan, cue, grid, boundaries, voice_seconds, bed_cache=get,
                               section_trim_db=trims)
    final = measure_arrangement(music, boundaries, targets)
    unhit = render_arrangement(prepared, plan, cue, grid, boundaries, voice_seconds, hits=False, bed_cache=get,
                               section_trim_db=trims)
    search_check = measure_arrangement(unhit, boundaries, targets)
    report = {
        "grid": grid.to_dict(),
        "phrase_preparation": prep_info,
        "boundaries": boundaries,
        "plan": plan.to_dict(),
        "section_trim_db": [float(x) for x in trims],
        "section_intensity": [float(x) for x in section_intensity(cue)],
        "section_targets_db": [float(x) for x in targets],
        "search_objective": search_metrics["objective"],
        "search_metrics": search_metrics,
        "final_without_hits_objective": search_check["objective"],
        "search_vs_final_aligned": abs(search_metrics["objective"] - search_check["objective"]) < 1e-6,
        "final_metrics": final,
        "music_seconds": len(music) / SR,
        "hit_points_on_grid": {"build": boundaries[4], "turn": boundaries[5],
                               "aftermath": boundaries[6], "end": len(music) / SR},
    }
    return music, report, prepared
