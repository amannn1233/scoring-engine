"""V2 engine tests. Run:  python -m pytest tests  (or)  python tests/test_v2.py"""

import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import scoring_engine as E  # noqa: E402
from story_analysis import discover_stories, infer_story_boundaries, normalized_story_id  # noqa: E402

RNG = np.random.default_rng(7)


def synthetic_phrase(seed):
    rng = np.random.default_rng(seed)
    n = int(E.PHRASE_DURATION * E.SR)
    t = np.arange(n) / E.SR
    env = 0.55 + 0.45 * np.sin(2 * np.pi * t / rng.uniform(2.0, 6.0) + rng.uniform(0, 6))
    x = (0.3 * np.sin(2 * np.pi * rng.uniform(80, 300) * t) * env + 0.05 * rng.standard_normal(n))
    return E.f32(np.stack([x, 0.8 * x], axis=1) * 0.85 / np.max(np.abs(x)))


def synthetic_layers():
    return {name: [synthetic_phrase(10 * i + j) for j in range(3)] for i, name in enumerate(E.LAYER_NAMES)}


def reference_measure(music, boundaries):
    """The original V2 exact_metrics() formula, written with rms_db() per chunk."""
    section = []
    for i in range(7):
        s, e = int(round(boundaries[i] * E.SR)), int(round(boundaries[i + 1] * E.SR))
        section.append(E.rms_db(music[s:e]))
    half = int(round(0.5 * E.SR))
    jumps = []
    for i in range(1, 7):
        c = int(round(boundaries[i] * E.SR))
        jumps.append(E.rms_db(music[c:min(len(music), c + half)]) - E.rms_db(music[max(0, c - half):c]))
    return np.array(section), np.array(jumps)


def test_to_mono_bit_identical():
    x = E.f32(RNG.standard_normal((50000, 2)))
    assert np.array_equal(E.to_mono(x), x.mean(axis=1))


def test_normalize_peak_does_not_mutate():
    x = E.f32(RNG.standard_normal((1000, 2)) * 3)
    before = x.copy()
    y = E.normalize_peak(x, 0.5)
    assert np.array_equal(x, before)
    assert abs(float(np.max(np.abs(y))) - 0.5) < 1e-5


def test_loop_slicing_is_exact():
    seq = E.prepare_layer_sequence(synthetic_layers()["core"], (2, 0, 1))
    total = int(40 * E.SR)
    looped = E.loop_sequence(seq, total + int(13 * E.SR) + len(seq))
    for phase in (0.0, 0.25, 5.0, 11.75, 40.0):
        assert np.array_equal(E.repeat_sequence(seq, total, phase),
                              E.repeat_sequence(seq, total, phase, looped=looped))


def test_measure_matches_original_formula():
    layers = E.normalize_layer_input(synthetic_layers())
    boundaries = list(np.array([0.0, 0.07, 0.26, 0.40, 0.60, 0.69, 0.88, 1.0]) * 45.0)
    total = int(round(45.0 * E.SR))
    music = E.render_score(layers, {n: (0, 1, 2) for n in E.LAYER_NAMES},
                           {n: 3.0 for n in E.LAYER_NAMES}, boundaries, total)
    m = E.measure_score(music, boundaries)
    section, jumps = reference_measure(music, boundaries)
    assert np.max(np.abs(section - m["section_db"])) < 1e-3
    assert np.max(np.abs(jumps - m["boundary_jumps_db"])) < 1e-3


def test_optimizer_report_equals_final_render():
    layers = synthetic_layers()
    boundaries = list(np.array([0.0, 0.07, 0.26, 0.40, 0.60, 0.69, 0.88, 1.0]) * 40.0)
    total = int(round(40.0 * E.SR))
    opt = E.optimize_story(layers, boundaries, total_samples=total, log=lambda *_: None)
    final = E.measure_score(
        E.render_score(E.normalize_layer_input(layers), opt["orders"], opt["phases_seconds"], boundaries, total),
        boundaries,
    )
    align = E.alignment_report(opt["metrics"], final)
    assert align["aligned"]
    assert align["worst_db_diff"] == 0.0
    assert opt["objective"] <= opt["stage1"]["objective"] + 1e-12


def test_layer_input_accepts_lists_and_rejects_wrong_count():
    layers = synthetic_layers()
    out = E.normalize_layer_input(layers)  # the V1 'list has no attribute mean' input
    assert all(len(v) == 3 for v in out.values())
    bad = dict(layers)
    bad["core"] = layers["core"][:2]
    try:
        E.normalize_layer_input(bad)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_boundaries_are_increasing_and_span_duration():
    for segments in ([], [{"start": i * 3.0, "end": i * 3.0 + 2, "text": w} for i, w in enumerate(
            ["hi", "he told me", "money gone", "risk", "I found it", "we opened", "it was forged",
             "the truth", "finally", "now", "since", "end"])]):
        b = infer_story_boundaries(segments, 40.0)
        assert len(b) == 8 and b[0] == 0.0 and abs(b[-1] - 40.0) < 1e-9
        assert np.all(np.diff(b) > 0)


def test_discovery_prefers_wav_and_excludes_85():
    with tempfile.TemporaryDirectory() as tmp:
        for name in ("71_extracted.wav", "86_audio.mp3", "86_extracted.wav", "85_extracted.wav", "85_audio.mp3"):
            (Path(tmp) / name).write_bytes(b"x")
        picked = discover_stories([tmp], ["71", "86"], log=lambda *_: None)
        assert [Path(p).name for p in picked] == ["71_extracted.wav", "86_extracted.wav"]
        assert normalized_story_id("86_audio.mp3") == "86"


def test_ducking_bounds():
    n = int(20 * E.SR)
    voice = E.f32(np.where((np.arange(n) // E.SR) % 2 == 0, 0.2, 0.0) * RNG.standard_normal(n))
    env, stats = E.duck_envelope(voice, n)
    assert env.shape == (n,)
    assert -E.MAX_DUCK_DB - 1e-6 <= env.min() <= env.max() <= 0.0
    assert stats["average_duck_db"] < 0


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"{len(tests)} tests passed")
