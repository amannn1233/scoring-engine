# ============================================================
# AUDIO I/O — soundfile when available, scipy.io.wavfile fallback
# ============================================================

import numpy as np

try:
    import soundfile as _sf
    SOUNDFILE_OK = True
except Exception:  # pragma: no cover - depends on environment
    _sf = None
    SOUNDFILE_OK = False


def read_audio(path):
    """Returns (float32 array of shape (samples, channels), sample_rate)."""
    path = str(path)
    if SOUNDFILE_OK:
        data, sr = _sf.read(path, always_2d=True, dtype="float32")
        return np.ascontiguousarray(data, dtype=np.float32), int(sr)
    if not path.lower().endswith(".wav"):
        raise RuntimeError(f"soundfile is not installed; cannot read non-WAV file {path}")
    from scipy.io import wavfile
    sr, data = wavfile.read(path)
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.uint8:
        data = (data.astype(np.float32) - 128.0) / 128.0
    else:
        data = data.astype(np.float32)
    if data.ndim == 1:
        data = data[:, None]
    return np.ascontiguousarray(data, dtype=np.float32), int(sr)


def write_audio(path, audio, sr):
    """Writes 16-bit PCM WAV."""
    path = str(path)
    audio = np.asarray(audio, dtype=np.float32)
    if SOUNDFILE_OK:
        _sf.write(path, audio, sr, subtype="PCM_16")
        return
    from scipy.io import wavfile
    pcm = np.clip(np.round(np.clip(audio, -1.0, 1.0) * 32767.0), -32768, 32767).astype(np.int16)
    wavfile.write(path, sr, pcm)


def resample_mono(mono, orig_sr, target_sr):
    if orig_sr == target_sr:
        return np.asarray(mono, dtype=np.float32)
    try:
        import librosa
        return np.asarray(librosa.resample(mono, orig_sr=orig_sr, target_sr=target_sr), dtype=np.float32)
    except Exception:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(int(orig_sr), int(target_sr))
        return np.asarray(resample_poly(mono, target_sr // g, orig_sr // g), dtype=np.float32)
