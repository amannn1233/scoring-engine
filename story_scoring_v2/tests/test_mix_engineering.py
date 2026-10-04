"""Mix/master stage tests against published reference values.
Run:  python tests/test_mix_engineering.py"""

import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import mix_engineering as M  # noqa: E402

SR = 44100


def sine(freq, seconds, amp, sr=SR, phase=0.0):
    t = np.arange(int(seconds * sr)) / sr
    return amp * np.sin(2 * np.pi * freq * t + phase)


def test_k_weighting_matches_bs1770_48k_table():
    (b1, a1), (b2, a2) = M.k_weighting_coeffs(48000)
    assert np.allclose(b1, [1.53512485958697, -2.69169618940638, 1.19839281085285], atol=1e-8)
    assert np.allclose(a1, [1.0, -1.69065929318241, 0.73248077421585], atol=1e-8)
    assert np.allclose(a2, [1.0, -1.99004745483398, 0.99007225036621], atol=1e-8)
    assert np.allclose(b2, [1.0, -2.0, 1.0])


def test_1k_sine_reads_its_level_in_lufs():
    # EBU Tech 3341: a 1 kHz stereo sine at -23 dBFS reads -23.0 LUFS (+-0.1).
    for sr in (44100, 48000):
        x = sine(1000, 10, 10 ** (-23 / 20), sr)
        lufs = M.integrated_loudness(np.stack([x, x], axis=1), sr)
        assert abs(lufs - (-23.0)) < 0.1, (sr, lufs)


def test_relative_gate_ignores_quiet_passage():
    # Tech 3341 case 3-style: 10 s at -36, 60 s at -23, 10 s at -36 -> -23.
    a = sine(1000, 10, 10 ** (-36 / 20))
    b = sine(1000, 60, 10 ** (-23 / 20))
    x = np.concatenate([a, b, a])
    lufs = M.integrated_loudness(np.stack([x, x], axis=1), SR)
    assert abs(lufs - (-23.0)) < 0.1, lufs


def test_lra_of_two_level_signal():
    # Tech 3342 case 1: 20 s at -20 then 20 s at -30 LUFS -> LRA 10 LU (+-1).
    x = np.concatenate([sine(1000, 20, 10 ** (-20 / 20)), sine(1000, 20, 10 ** (-30 / 20))])
    lra = M.loudness_range(np.stack([x, x], axis=1), SR)
    assert abs(lra - 10.0) < 1.0, lra


def test_true_peak_catches_intersample_peak():
    # fs/4 sine at 45 deg phase: every sample is at 0.707 of the real peak.
    x = sine(SR / 4, 1, 1.0, phase=math.pi / 4)
    sample_peak = 20 * math.log10(np.max(np.abs(x)))
    tp = M.true_peak_dbtp(np.stack([x, x], axis=1), SR)
    assert abs(sample_peak - (-3.01)) < 0.05
    assert abs(tp) < 0.3, tp


def test_limiter_holds_ceiling_and_master_hits_spec():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((SR * 20, 2)) * 0.05
    x[SR * 5: SR * 5 + 200] *= 15  # transient
    spec = M.DELIVERY_SPECS["shorts"]
    out, gr = M.master_to_spec(x, SR, spec)
    assert M.true_peak_dbtp(out, SR) <= spec.true_peak_dbtp + 0.02
    assert abs(M.integrated_loudness(out, SR) - spec.integrated_lufs) < 0.2
    assert gr > 0


def test_ducking_hits_target_dmr_and_leaves_gaps_alone():
    rng = np.random.default_rng(2)
    n = SR * 20
    voice = np.zeros(n)
    for start in (2, 8, 14):  # three 3 s lines
        s = start * SR
        voice[s: s + 3 * SR] = sine(220, 3, 0.2) * (1 + 0.3 * np.sin(np.arange(3 * SR) / SR * 2 * np.pi * 4))
    music = rng.standard_normal((n, 2)) * 0.1
    settings = M.MixSettings()
    duck_db, weight, stats = M.dialogue_duck_curve(voice, music, SR, settings)
    assert duck_db.max() <= 0.0 and duck_db.min() >= -settings.max_duck_db - 1e-9
    mid_line = int(3.5 * SR)
    gap = int(6.2 * SR)
    assert duck_db[mid_line] < -3.0
    assert duck_db[gap] > duck_db[mid_line] + 3.0
    # Lookahead: already ducking before the first line starts.
    assert duck_db[int(1.98 * SR)] < -0.5
    assert weight[mid_line] > 0.9


def test_mix_and_master_report():
    rng = np.random.default_rng(3)
    n = SR * 15
    voice = np.zeros(n)
    voice[SR: 6 * SR] = rng.standard_normal(5 * SR) * 0.1
    voice[8 * SR: 13 * SR] = rng.standard_normal(5 * SR) * 0.1
    music = rng.standard_normal((n, 2)) * 0.04
    master, report = M.mix_and_master(voice, music, SR, "podcast")
    qc = report["master_qc"]
    assert master.dtype == np.float32 and master.shape == (n, 2)
    assert abs(qc["integrated_lufs"] - (-16.0)) < 0.5
    assert qc["true_peak_dbtp"] <= -0.98
    assert abs(report["dialogue_mix"]["achieved_dmr_lu_median"] - 15.0) < 2.0
    assert qc["clip_runs"] == 0


def test_qc_flags_problems():
    rng = np.random.default_rng(4)
    x = rng.standard_normal((SR * 5, 2)) * 0.3
    x[:, 1] = -x[:, 0]                         # polarity-flipped right channel
    x[SR: SR + 10] = 1.0                        # hard clip run
    report = M.qc_report(x, SR, M.DELIVERY_SPECS["shorts"])
    text = " ".join(report["warnings"])
    assert "clipped" in text and "phase correlation" in text and "true peak" in text


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_")]
    for name, fn in tests:
        fn()
        print("PASS", name)
    print(f"{len(tests)} tests passed")
