# ============================================================
# MUSIC ANALYSIS — tempo, key, loop seams, spectral fullness.
# Pure numpy (no librosa needed) so take ranking works anywhere and the
# same checks can be run on real Kaggle renders afterwards.
# ============================================================

import math

import numpy as np


_KEYS = ("C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")
_ENHARMONIC = {"Db": "C#", "D#": "Eb", "Gb": "F#", "G#": "Ab", "A#": "Bb"}
# Krumhansl-Kessler key profiles.
_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
# Modes that read as major- or minor-family for key matching, and the
# semitone offset of their relative major/minor tonic.
MODE_FAMILY = {
    "major": ("major", 0), "lydian": ("major", 0), "mixolydian": ("major", 0),
    "aeolian": ("minor", 0), "harmonic minor": ("minor", 0), "dorian": ("minor", 0),
    "phrygian": ("minor", 0), "minor": ("minor", 0),
}


def _mono(audio):
    audio = np.asarray(audio, dtype=np.float64)
    return audio.mean(axis=1) if audio.ndim == 2 else audio


def _stft_mag(mono, n_fft=2048, hop=512):
    if len(mono) < n_fft:
        mono = np.pad(mono, (0, n_fft - len(mono)))
    frames = 1 + (len(mono) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(frames)[:, None]
    win = np.hanning(n_fft)
    return np.abs(np.fft.rfft(mono[idx] * win, axis=1))


def onset_envelope(audio, sr, hop=512):
    mag = _stft_mag(_mono(audio), 2048, hop)
    logmag = np.log1p(100.0 * mag)
    flux = np.maximum(np.diff(logmag, axis=0), 0.0).sum(axis=1)
    flux = flux - np.convolve(flux, np.ones(16) / 16, mode="same")   # local mean removal
    return np.maximum(flux, 0.0), sr / hop


def estimate_tempo(audio, sr, lo_bpm=50.0, hi_bpm=190.0):
    """(bpm, confidence 0..1) from the onset-envelope autocorrelation,
    weighted toward 70-130 BPM the way a listener taps along."""
    env, fps = onset_envelope(audio, sr)
    if len(env) < 16 or not env.any():
        return 0.0, 0.0
    env = env - env.mean()
    ac = np.correlate(env, env, mode="full")[len(env) - 1:]
    ac /= ac[0] + 1e-12
    lags = np.arange(len(ac))
    bpm = np.where(lags > 0, 60.0 * fps / np.maximum(lags, 1), 0)
    sel = (bpm >= lo_bpm) & (bpm <= hi_bpm)
    if not sel.any():
        return 0.0, 0.0
    weight = np.exp(-0.5 * (np.log2(np.maximum(bpm, 1) / 100.0) / 0.9) ** 2)
    score = np.where(sel, ac * weight, -np.inf)
    best = int(np.argmax(score))
    # Parabolic refinement for sub-lag precision.
    if 1 <= best < len(ac) - 1:
        a, b, c = ac[best - 1], ac[best], ac[best + 1]
        denom = a - 2 * b + c
        shift = 0.5 * (a - c) / denom if abs(denom) > 1e-12 else 0.0
        lag = best + float(np.clip(shift, -0.5, 0.5))
    else:
        lag = float(best)
    return float(60.0 * fps / lag), float(np.clip(ac[best], 0.0, 1.0))


def tempo_match(estimated_bpm, target_bpm):
    """1.0 for a perfect match, octave errors (half/double time) accepted
    as matches, falling to 0 at ~6 % off."""
    if estimated_bpm <= 0 or target_bpm <= 0:
        return 0.0
    ratios = [estimated_bpm / (target_bpm * k) for k in (0.5, 1.0, 2.0, 2.0 / 3.0, 1.5)]
    err = min(abs(math.log2(r)) for r in ratios)
    return float(max(0.0, 1.0 - err / 0.085))


def chroma(audio, sr, n_fft=8192, hop=4096):
    mag = _stft_mag(_mono(audio), n_fft, hop) ** 2
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    sel = (freqs >= 55.0) & (freqs <= 4200.0)
    pitch = np.round(12 * np.log2(freqs[sel] / 440.0) + 69).astype(int) % 12
    out = np.zeros(12)
    np.add.at(out, pitch, mag[:, sel].sum(axis=0))
    # MIDI 60 is C, so pitch class 0 == C.
    return out / (np.linalg.norm(out) + 1e-12)


def estimate_key(audio, sr):
    """(tonic, 'major'|'minor', correlation)."""
    c = chroma(audio, sr)
    best = (-2.0, "C", "major")
    for shift in range(12):
        for name, prof in (("major", _MAJOR), ("minor", _MINOR)):
            r = float(np.corrcoef(np.roll(prof, shift), c)[0, 1])
            if r > best[0]:
                best = (r, _KEYS[shift], name)
    return best[1], best[2], best[0]


def key_match(est_tonic, est_family, target_tonic, target_mode):
    """1.0 same key; 0.8 relative major/minor (same notes); 0.5 dominant or
    subdominant neighbour; else 0."""
    target_tonic = _ENHARMONIC.get(target_tonic, target_tonic)
    est_tonic = _ENHARMONIC.get(est_tonic, est_tonic)
    fam, _ = MODE_FAMILY.get(target_mode, ("minor", 0))
    t = _KEYS.index(target_tonic)
    e = _KEYS.index(est_tonic)
    if fam == est_family and t == e:
        return 1.0
    # relative: minor tonic = major tonic - 3 semitones
    if fam == "minor" and est_family == "major" and (t + 3) % 12 == e:
        return 0.8
    if fam == "major" and est_family == "minor" and (t - 3) % 12 == e:
        return 0.8
    if fam == est_family and (e - t) % 12 in (5, 7):
        return 0.5
    return 0.0


def _band_db(segment, sr, edges=(60, 150, 400, 1000, 2500, 6000, 12000)):
    mag = _stft_mag(_mono(segment), 2048, 1024) ** 2
    freqs = np.fft.rfftfreq(2048, 1.0 / sr)
    spec = mag.mean(axis=0)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (freqs >= lo) & (freqs < hi)
        out.append(10 * np.log10(spec[sel].sum() + 1e-12))
    return np.array(out)


def seam_score(audio, sr, seconds=1.25):
    """How well the phrase's tail joins its own head (the crossfade the
    renderer makes). 1.0 = indistinguishable level and tone, 0 = obvious
    jump. Uses level difference and band-energy (timbre) distance."""
    n = int(round(seconds * sr))
    mono = _mono(audio)
    if len(mono) < 3 * n:
        return 0.0
    head, tail = mono[:n], mono[-n:]
    lvl = abs(10 * np.log10((np.mean(head ** 2) + 1e-12) / (np.mean(tail ** 2) + 1e-12)))
    timbre = float(np.sqrt(np.mean((_band_db(head, sr) - _band_db(tail, sr)) ** 2)))
    return float(max(0.0, 1.0 - lvl / 8.0 - timbre / 16.0))


def fullness(audio, sr):
    """Fraction of 7 bands within 30 dB of the loudest band (thin vs full)."""
    bands = _band_db(audio, sr)
    return float(np.mean(bands > bands.max() - 30.0))


def analyse(audio, sr, target_bpm=None, target_key=None, target_mode=None):
    bpm, conf = estimate_tempo(audio, sr)
    tonic, family, kr = estimate_key(audio, sr)
    out = {"tempo_bpm": bpm, "tempo_confidence": conf, "key": f"{tonic} {family}",
           "key_correlation": kr, "seam": seam_score(audio, sr), "fullness": fullness(audio, sr)}
    if target_bpm:
        out["tempo_match"] = tempo_match(bpm, target_bpm)
    if target_key and target_mode:
        out["key_match"] = key_match(tonic, family, target_key, target_mode)
    return out
