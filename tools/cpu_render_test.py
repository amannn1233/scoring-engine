#!/usr/bin/env python3
"""Render short Stable Audio 3 Small Music clips on CPU and measure them.

Standalone: needs only numpy + scipy for the measurements, plus torch and
stable_audio_3 for rendering (pip install
git+https://github.com/Stability-AI/stable-audio-3.git). The model repo
stabilityai/stable-audio-3-small-music is gated on Hugging Face, so HF_TOKEN
must belong to an account that accepted its license.

Per clip it reports render time, real-time factor, tempo (BPM), key,
integrated loudness (LUFS, ITU-R BS.1770-4), true peak (dBTP, 4x
oversampled) and sample peak, and writes the WAV plus a JSON report.

Examples:
  # measurement code only, no model needed
  python tools/cpu_render_test.py --self-test

  # two pipeline-style 12 s phrases (same prompts as story_scoring_v2)
  python tools/cpu_render_test.py --preset core --preset climax --mood "guilty confession"

  # free text, 6 s, plus one init_audio (conditioned) render for timing
  python tools/cpu_render_test.py -p "dark felt piano motif" --duration 6 --conditioned

  # measure an existing file
  python tools/cpu_render_test.py --analyze some.wav
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.io import wavfile

# Generation settings from story_scoring_v2/music_generation.py.
STEPS = 8
CFG_SCALE = 1.0
PHRASE_DURATION = 12.0
LAYER_NOISE = {"core": 0.10, "pressure": 0.13, "climax": 0.16}


# ============================================================
# PROMPTS: copied from story_scoring_v2/music_generation.py
# (motif_prompt / layer_prompts) so this script stays standalone.
# ============================================================

def preset_prompt(name, mood):
    if name == "motif":
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
    }[name]


# ============================================================
# MEASUREMENTS (numpy / scipy only)
# ============================================================

def as_stereo(x):
    """Any array -> (samples, channels) float64."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 3:
        x = x[0]
    if x.ndim == 1:
        x = x[:, None]
    if x.shape[0] <= 8 and x.shape[1] > x.shape[0]:
        x = x.T
    return x


def _biquad_high_shelf(fs, fc, gain_db, q):
    a = 10 ** (gain_db / 40.0)
    w0 = 2 * np.pi * fc / fs
    cw, alpha = np.cos(w0), np.sin(w0) / (2 * q)
    sa = 2 * np.sqrt(a) * alpha
    b = [a * ((a + 1) + (a - 1) * cw + sa),
         -2 * a * ((a - 1) + (a + 1) * cw),
         a * ((a + 1) + (a - 1) * cw - sa)]
    den = [(a + 1) - (a - 1) * cw + sa,
           2 * ((a - 1) - (a + 1) * cw),
           (a + 1) - (a - 1) * cw - sa]
    return np.array(b) / den[0], np.array(den) / den[0]


def _biquad_high_pass(fs, fc, q):
    w0 = 2 * np.pi * fc / fs
    cw, alpha = np.cos(w0), np.sin(w0) / (2 * q)
    b = [(1 + cw) / 2, -(1 + cw), (1 + cw) / 2]
    den = [1 + alpha, -2 * cw, 1 - alpha]
    return np.array(b) / den[0], np.array(den) / den[0]


def integrated_lufs(x, sr):
    """ITU-R BS.1770-4 integrated loudness with absolute and relative gating
    (same K-weighting constants as pyloudnorm)."""
    x = as_stereo(x)
    b1, a1 = _biquad_high_shelf(sr, 1681.974450955533, 3.999843853973347, 0.7071752369554196)
    b2, a2 = _biquad_high_pass(sr, 38.13547087602444, 0.5003270373238773)
    y = signal.lfilter(b2, a2, signal.lfilter(b1, a1, x, axis=0), axis=0)
    block, hop = int(0.4 * sr), int(0.1 * sr)
    if len(y) < block:
        return float("-inf")
    starts = range(0, len(y) - block + 1, hop)
    z = np.array([np.mean(y[s:s + block] ** 2, axis=0).sum() for s in starts])
    with np.errstate(divide="ignore"):
        lj = -0.691 + 10 * np.log10(z)
    z_abs = z[lj > -70.0]
    if not len(z_abs):
        return float("-inf")
    gamma_r = -0.691 + 10 * np.log10(z_abs.mean()) - 10.0
    z_rel = z[(lj > -70.0) & (lj > gamma_r)]
    return float(-0.691 + 10 * np.log10(z_rel.mean()))


def true_peak_dbtp(x, oversample=4):
    x = as_stereo(x)
    up = signal.resample_poly(x, oversample, 1, axis=0)
    peak = float(np.max(np.abs(up)))
    return 20 * np.log10(peak) if peak > 0 else float("-inf")


def sample_peak_dbfs(x):
    peak = float(np.max(np.abs(as_stereo(x))))
    return 20 * np.log10(peak) if peak > 0 else float("-inf")


def _mono_stft(x, sr, n_fft=4096, hop=512):
    mono = as_stereo(x).mean(axis=1)
    f, _, spec = signal.stft(mono, fs=sr, nperseg=n_fft, noverlap=n_fft - hop,
                             boundary=None, padded=False)
    return f, np.abs(spec), hop


def estimate_tempo(x, sr, lo=60.0, hi=180.0):
    """Spectral-flux onset envelope + autocorrelation, weighted toward
    ~110 BPM the way librosa's prior is. Returns (bpm, confidence 0..1)."""
    _, mag, hop = _mono_stft(x, sr, n_fft=2048)
    logmag = np.log1p(100 * mag)
    flux = np.maximum(np.diff(logmag, axis=1), 0).sum(axis=0)
    flux = flux - flux.mean()
    if len(flux) < 8 or not np.any(flux):
        return None, 0.0
    frame_rate = sr / hop
    acf = np.correlate(flux, flux, mode="full")[len(flux) - 1:]
    acf /= acf[0]
    lags = np.arange(len(acf))
    lo_lag = max(1, int(np.floor(frame_rate * 60 / hi)))
    hi_lag = min(len(acf) - 1, int(np.ceil(frame_rate * 60 / lo)))
    if hi_lag <= lo_lag:
        return None, 0.0
    bpm = 60 * frame_rate / lags[lo_lag:hi_lag + 1]
    prior = np.exp(-0.5 * (np.log2(bpm / 110.0) / 1.0) ** 2)
    score = acf[lo_lag:hi_lag + 1] * prior
    i = int(np.argmax(score))
    # parabolic refinement of the peak lag
    lag = float(lags[lo_lag + i])
    if 0 < i < len(score) - 1:
        y0, y1, y2 = score[i - 1], score[i], score[i + 1]
        denom = y0 - 2 * y1 + y2
        if denom:
            lag += 0.5 * (y0 - y2) / denom
    return float(60 * frame_rate / lag), float(max(0.0, acf[lo_lag + i]))


_KK_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_KK_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
_NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def estimate_key(x, sr):
    """Chroma from the STFT (55 Hz..5 kHz) correlated against
    Krumhansl-Kessler profiles. Returns (key, correlation, runner_up)."""
    f, mag, _ = _mono_stft(x, sr)
    keep = (f >= 55) & (f <= 5000)
    pitch_class = np.round(12 * np.log2(f[keep] / 440.0) + 69).astype(int) % 12
    energy = (mag[keep] ** 2).sum(axis=1)
    chroma = np.bincount(pitch_class, weights=energy, minlength=12)
    if not chroma.any():
        return None, 0.0, None
    scores = []
    for tonic in range(12):
        for mode, profile in (("major", _KK_MAJOR), ("minor", _KK_MINOR)):
            r = np.corrcoef(chroma, np.roll(profile, tonic))[0, 1]
            scores.append((float(r), f"{_NOTES[tonic]} {mode}"))
    scores.sort(reverse=True)
    return scores[0][1], scores[0][0], scores[1][1]


def measure(x, sr):
    x = as_stereo(x)
    bpm, bpm_conf = estimate_tempo(x, sr)
    key, key_r, key_2 = estimate_key(x, sr)
    rms = float(np.sqrt(np.mean(x ** 2)))
    return {
        "duration_s": round(len(x) / sr, 3),
        "channels": int(x.shape[1]),
        "tempo_bpm": None if bpm is None else round(bpm, 1),
        "tempo_confidence": round(bpm_conf, 3),
        "key": key,
        "key_correlation": round(key_r, 3),
        "key_runner_up": key_2,
        "integrated_lufs": round(integrated_lufs(x, sr), 2),
        "true_peak_dbtp": round(true_peak_dbtp(x), 2),
        "sample_peak_dbfs": round(sample_peak_dbfs(x), 2),
        "rms_dbfs": round(20 * np.log10(rms), 2) if rms > 0 else float("-inf"),
    }


# ============================================================
# I/O
# ============================================================

def read_wav(path):
    try:
        import soundfile as sf
        data, sr = sf.read(str(path), always_2d=True, dtype="float32")
        return data, sr
    except ImportError:
        sr, data = wavfile.read(str(path))
        if data.dtype.kind == "i":
            data = data / float(np.iinfo(data.dtype).max)
        return as_stereo(data), sr


def write_wav(path, x, sr):
    wavfile.write(str(path), sr, as_stereo(x).astype(np.float32))


# ============================================================
# SELF-TEST: synthetic clip with known tempo / key / level
# ============================================================

def synthetic_clip(sr=44100, seconds=12.0, bpm=96.0):
    """A minor triad pad plus a click on every beat, at a known level."""
    t = np.arange(int(sr * seconds)) / sr
    pad = sum(np.sin(2 * np.pi * f * t) for f in (220.0, 261.63, 329.63, 110.0))
    pad *= 0.08
    clicks = np.zeros_like(t)
    beat = 60.0 / bpm
    env = np.exp(-np.arange(int(0.05 * sr)) / (0.008 * sr))
    burst = env * np.sin(2 * np.pi * 1000 * np.arange(len(env)) / sr) * 0.5
    for k in range(int(seconds / beat)):
        s = int(k * beat * sr)
        e = min(len(t), s + len(burst))
        clicks[s:e] += burst[:e - s]
    mono = pad + clicks
    return np.stack([mono, mono], axis=1), sr


def self_test():
    x, sr = synthetic_clip()
    m = measure(x, sr)
    print(json.dumps(m, indent=2))
    # 1 kHz sine at -20 dBFS (stereo) is -20.0 +/- ~0.1 LUFS + 3 dB for 2 channels
    t = np.arange(sr * 5) / sr
    sine = 0.1 * np.sin(2 * np.pi * 997 * t)
    lufs = integrated_lufs(np.stack([sine, sine], axis=1), sr)
    print(f"997 Hz sine at -20 dBFS peak, stereo: {lufs:.2f} LUFS (expect about -20.0)")
    ok = (m["key"] == "A minor" and abs(m["tempo_bpm"] - 96.0) < 3.0 and abs(lufs + 20.0) < 0.3)
    print("SELF-TEST", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


# ============================================================
# RENDERING
# ============================================================

def load_model(threads, log=print):
    import torch
    if threads:
        torch.set_num_threads(threads)
    from stable_audio_3 import StableAudioModel
    if not os.environ.get("HF_TOKEN") and not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        log("note: HF_TOKEN is not set; the gated stabilityai repo will refuse the "
            "download unless the weights are already in the HF cache")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.perf_counter()
    model = StableAudioModel.from_pretrained("small-music", device=device)
    load_s = time.perf_counter() - t0
    log(f"loaded small-music on {device} ({torch.get_num_threads()} threads) in {load_s:.1f}s")
    return model, device, load_s


def render(model, prompt, duration, seed, init_audio=None, noise=None):
    import torch
    kwargs = dict(prompt=prompt, duration=float(duration), steps=STEPS,
                  cfg_scale=CFG_SCALE, seed=int(seed), batch_size=1)
    if init_audio is not None:
        kwargs.update(init_audio=(model.model.sample_rate, np.ascontiguousarray(init_audio, dtype=np.float32)),
                      init_noise_level=float(noise))
    t0 = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(**kwargs)
    seconds = time.perf_counter() - t0
    audio = out[0].detach().float().cpu().numpy().T  # (samples, channels)
    return audio, seconds


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-p", "--prompt", action="append", default=[], help="free-text prompt (repeatable)")
    ap.add_argument("--preset", action="append", default=[], choices=["motif", "core", "pressure", "climax"],
                    help="pipeline prompt from story_scoring_v2 (repeatable)")
    ap.add_argument("--mood", default="guilty confession", help="mood word(s) for presets")
    ap.add_argument("--duration", type=float, default=PHRASE_DURATION, help="seconds per clip (pipeline phrase = 12)")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (default: torch's choice)")
    ap.add_argument("--conditioned", action="store_true",
                    help="also time one init_audio render from the first clip (the pipeline's phrase path)")
    ap.add_argument("--out-dir", default="cpu_render_out")
    ap.add_argument("--analyze", nargs="+", help="only measure existing audio files")
    ap.add_argument("--self-test", action="store_true", help="check the measurement code on a synthetic clip")
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test()

    if args.analyze:
        for path in args.analyze:
            x, sr = read_wav(path)
            print(json.dumps({"file": path, **measure(x, sr)}, indent=2))
        return 0

    jobs = [(f"preset_{p}", preset_prompt(p, args.mood)) for p in args.preset]
    jobs += [(f"prompt_{i + 1}", p) for i, p in enumerate(args.prompt)]
    if not jobs:
        jobs = [("preset_core", preset_prompt("core", args.mood)),
                ("preset_climax", preset_prompt("climax", args.mood))]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model, device, load_s = load_model(args.threads)
    sr = int(model.model.sample_rate)

    results = []
    for n, (name, prompt) in enumerate(jobs):
        audio, secs = render(model, prompt, args.duration, args.seed + n)
        path = out_dir / f"{name}.wav"
        write_wav(path, audio, sr)
        row = {"name": name, "file": str(path), "prompt": prompt, "seed": args.seed + n,
               "render_s": round(secs, 2), "realtime_factor": round(secs / args.duration, 2),
               "first_call_includes_warmup": n == 0, **measure(audio, sr)}
        results.append(row)
        print(f"{name}: {secs:.1f}s for {args.duration:.0f}s audio "
              f"(x{secs / args.duration:.2f} realtime) | {row['tempo_bpm']} BPM | {row['key']} | "
              f"{row['integrated_lufs']} LUFS | {row['true_peak_dbtp']} dBTP", flush=True)

    if args.conditioned and results:
        ref, _ = read_wav(results[0]["file"])
        noise = LAYER_NOISE["core"]
        audio, secs = render(model, jobs[0][1], args.duration, args.seed + 100, init_audio=ref, noise=noise)
        path = out_dir / f"{jobs[0][0]}_conditioned.wav"
        write_wav(path, audio, sr)
        row = {"name": f"{jobs[0][0]}_conditioned", "file": str(path), "init_noise_level": noise,
               "render_s": round(secs, 2), "realtime_factor": round(secs / args.duration, 2),
               **measure(audio, sr)}
        results.append(row)
        print(f"{row['name']}: {secs:.1f}s (x{secs / args.duration:.2f} realtime)", flush=True)

    report = {"device": device, "model": "stabilityai/stable-audio-3-small-music",
              "steps": STEPS, "cfg_scale": CFG_SCALE, "sample_rate": sr,
              "cpu_count": os.cpu_count(), "model_load_s": round(load_s, 1), "clips": results}
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    print(f"report: {out_dir / 'report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
