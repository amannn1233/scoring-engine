"""Story-aware scoring tests: cue sheet, music analysis, take ranking,
bar-grid arrangement, story hits and the engineered mix.

Run:  python tests/test_story_scoring.py      (fully offline, no GPU)"""

import json
import math
import re
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import arrangement as AR  # noqa: E402
import cue_sheet as C  # noqa: E402
import mix_engineering as M  # noqa: E402
import music_analysis as A  # noqa: E402
import music_generation as MG  # noqa: E402
import scoring_engine as E  # noqa: E402
import story_analysis as SA  # noqa: E402
from audio_io import write_audio  # noqa: E402
from fake_music import PromptFollowingProvider, render  # noqa: E402
from pipeline import process_story  # noqa: E402

REPO = HERE.parent.parent
SR = E.SR


def _stories():
    d = json.load(open(REPO / "85_transcript_clean.json"))
    workshop = (SA.normalize_text(d["segments"]), float(d["duration_sec"]))
    text = " ".join(re.sub(r"\[music\]", "", (REPO / "guiltyconfession.txt").read_text()).split())
    t, segs = 0.0, []
    for sent in re.split(r"(?<=[.!?])\s+", text):
        dur = len(sent.split()) / 2.7
        segs.append({"text": sent, "start": t, "end": t + dur})
        t += dur
    return {"workshop": workshop, "undercover": (segs, t)}


STORIES = _stories()
CUES = {k: C.build_cue_sheet(s, d, list(SA.infer_story_boundaries(s, d))) for k, (s, d) in STORIES.items()}


# ============================================================
# cue sheet
# ============================================================

def test_cue_sheets_differ_per_story():
    a, b = CUES["workshop"], CUES["undercover"]
    assert sum(a.palette[r] != b.palette[r] for r in C.ROLES) >= 3
    assert a.style != b.style
    assert (a.tempo_bpm, a.key_label) != (b.tempo_bpm, b.key_label)


def test_cue_sheet_reads_each_story():
    # The undercover-biker confession is gritty / deceptive; the engineer's
    # workshop story is procedural pressure. Neither gets the old hardcoded
    # "dark felt piano" palette.
    u, w = CUES["undercover"], CUES["workshop"]
    assert max(u.colour, key=u.colour.get) == "grit"
    assert max(w.colour, key=w.colour.get) == "pressure"
    assert "guitar" in u.palette["lead"]
    for cue in CUES.values():
        assert 60 <= cue.tempo_bpm <= 118


def test_hit_points_find_the_twist():
    # Twist lines as written in the transcripts.
    w = CUES["workshop"].hit_points
    u = CUES["undercover"].hit_points
    segs_w, _ = STORIES["workshop"]
    twist_w = next(s for s in segs_w if "signature that was not mine" in s["text"])
    assert abs(w["twist"] - twist_w["start"]) < 1.0
    segs_u, _ = STORIES["undercover"]
    twist_u = next(s for s in segs_u if "the whole thing was a test" in s["text"])
    assert abs(u["twist"] - twist_u["start"]) < 1.0
    for hp in (w, u):
        assert hp["twist"] < hp["climax"] <= hp["resolution"] <= hp["end"]


def test_stem_matching_ignores_false_prefixes():
    assert C._stem_ok("diesel", "die") is False
    assert C._stem_ok("started", "star") is False
    assert C._stem_ok("stopped", "stop") is True
    assert C._stem_ok("betrayal", "betray") is True


def test_prompts_carry_tempo_key_and_palette():
    for cue in CUES.values():
        prompts = [C.motif_prompt_from_cue(cue)] + list(C.layer_prompts_from_cue(cue).values())
        tag = f"{cue.tempo_bpm} BPM, {cue.key_label}"
        for p in prompts:
            assert tag in p and "no vocals" in p and len(p.split()) <= 60
            assert sum(v in p for v in cue.palette.values()) >= 3


def test_empty_transcript_still_gives_a_cue():
    cue = C.build_cue_sheet([], 40.0)
    assert 60 <= cue.tempo_bpm <= 118 and len(cue.arc) == 7


# ============================================================
# music analysis
# ============================================================

def test_tempo_and_key_estimates_on_rendered_music():
    for bpm, key, mode, fam in ((92, "D", "dorian", "minor"), (120, "E", "major", "major"),
                                (70, "G", "aeolian", "minor")):
        x = render(f"x, {bpm} BPM, {key} {mode}", 12.0, 1, 0.0)
        est, conf = A.estimate_tempo(x, SR)
        assert A.tempo_match(est, bpm) > 0.9, (bpm, est)
        tonic, family, _ = A.estimate_key(x, SR)
        assert (tonic, family) == (key, fam), (key, tonic, family)


def test_key_match_relative_keys():
    assert A.key_match("D", "minor", "D", "dorian") == 1.0
    assert A.key_match("F", "major", "D", "aeolian") == 0.8
    assert A.key_match("F#", "major", "C", "major") == 0.0


# ============================================================
# arrangement
# ============================================================

def test_wsola_changes_length_not_pitch():
    t = np.arange(SR * 4) / SR
    x = np.stack([np.sin(2 * np.pi * 440 * t)] * 2, 1)
    y = AR.wsola_stretch(x, 1.05)
    assert abs(len(y) / len(x) - 1.05) < 0.01
    seg = y[SR:3 * SR, 0]
    f = np.fft.rfftfreq(len(seg), 1 / SR)[np.argmax(np.abs(np.fft.rfft(seg * np.hanning(len(seg)))))]
    assert abs(f - 440) < 2.0


def test_prepare_phrase_conforms_tempo():
    grid = AR.Grid(86, 4, 4, 1)
    x = render("x, 88.5 BPM, G aeolian", grid.generate_seconds, 3, 0.0)
    info = {}
    y = AR.prepare_phrase(x, grid, info)
    assert len(y) == int(round((grid.phrase_seconds + grid.xfade_seconds) * SR))
    assert abs(info["stretch_rate"] - 88.5 / 86) < 0.01
    assert A.tempo_match(A.estimate_tempo(y, SR)[0], 86) > 0.9


def test_bed_joins_land_on_bars():
    grid = AR.Grid(96, 4, 4, 1)
    n = int(round((grid.phrase_seconds + grid.xfade_seconds) * SR))
    # Three constant "phrases" at different levels: joins show up as level steps.
    phrases = [np.full((n, 2), v) for v in (0.1, 0.2, 0.3)]
    total = int(30 * SR)
    bed = AR.build_bed(phrases, (0, 1, 2), grid, total)
    period = int(round(grid.phrase_seconds * SR))
    for k in (1, 2):
        j = k * period
        assert abs(bed[j + 600, 0] - phrases[k][0, 0] - phrases[k - 1][0, 0]) < 0.05   # tail rings under
        assert abs(bed[j + int(round(grid.beat * SR)) + 5, 0] - phrases[k][0, 0]) < 1e-6  # gone a beat later


def test_story_boundaries_on_bar_lines_and_turn_on_twist():
    for name, (segs, dur) in STORIES.items():
        cue = CUES[name]
        grid = AR.plan_grid(cue)
        b = AR.story_boundaries(cue, list(SA.infer_story_boundaries(segs, dur)), grid, dur)
        assert all(b[i + 1] - b[i] >= 0.5 * grid.bar - 1e-9 for i in range(7))
        for x in b[1:7]:
            assert abs(x / grid.bar - round(x / grid.bar)) < 1e-9
        assert abs(b[5] - cue.hit_points["twist"]) <= 0.51 * grid.bar


# ============================================================
# take ranking + full story-aware run
# ============================================================

def _narration(path, seconds, segs):
    rng = np.random.default_rng(5)
    n = int(seconds * SR)
    v = np.zeros(n, np.float32)
    for s in segs:
        a, b = int(s["start"] * SR), min(n, int(s["end"] * SR))
        tt = np.arange(b - a) / SR
        v[a:b] = 0.15 * np.sin(2 * np.pi * 140 * tt) * np.clip(np.sin(2 * np.pi * 4 * tt), 0, None) \
            + 0.01 * rng.standard_normal(b - a)
    write_audio(path, np.stack([v, v], 1), SR)


def _short_story():
    segs, dur = STORIES["workshop"]
    keep = [s for s in segs if s["end"] <= 60.0]
    return keep, keep[-1]["end"] + 1.0


def test_story_aware_process_story_end_to_end():
    segs, dur = _short_story()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        _narration(tmp / "s.wav", dur, segs)
        json.dump({"segments": segs}, open(tmp / "s.json", "w"))
        prev = MG.set_provider(PromptFollowingProvider())
        try:
            m = process_story(1, str(tmp / "s.wav"), tmp / "out", 99, log=lambda *_: None)
        finally:
            MG.set_provider(prev)
        a = m["arrangement"]
        cue = m["cue_sheet"]
        # Every phrase: several takes, and the top-ranked one was kept.
        groups = {}
        for r in m["quality_report"]:
            groups.setdefault((r["layer"], r["phrase"]), []).append(r)
        assert all(len(g) == MG.PHRASE_TAKES for g in groups.values())
        assert all(max(g, key=lambda r: r["musical_fit"]["rank"])["selected"] for g in groups.values())
        assert len(m["motif"]["takes"]) == MG.MOTIF_TAKES
        # Prompts were the story's own.
        assert f"{cue['tempo_bpm']} BPM" in m["motif"]["prompt"]
        # Arrangement: deterministic, on the grid, runs past the last word.
        assert m["arrangement_deterministic"] and a["search_vs_final_aligned"]
        assert a["music_seconds"] >= dur + a["grid"]["bar"] - 1e-6
        # Master meets the delivery spec.
        qc = m["mix_master"]["master_qc"]
        assert abs(qc["integrated_lufs"] - (-14.0)) <= 0.5 and qc["true_peak_dbtp"] <= -0.95


# ============================================================
# mix / master
# ============================================================

def _voice(n, amps):
    from self_review import make_voice
    return make_voice(n, amps)


def test_dmr_independent_of_stem_levels():
    from self_review import make_music
    n = SR * 32
    for va, ma in ((0.01, 0.25), (0.3, 0.002)):
        _, rep = M.mix_and_master(_voice(n, (va,) * 3), make_music(n, ma), SR, "shorts")
        assert abs(rep["dialogue_mix"]["achieved_dmr_lu_median"] - 15.0) < 1.5


def test_dialogue_leveling_evens_out_lines():
    n = SR * 32
    v = M.highpass(_voice(n, (0.2, 0.03, 0.1)), SR, 70)[:, 0]
    lev, stats = M.level_dialogue(v, SR)
    lines = [M.integrated_loudness(M.as_stereo(lev[s * SR:(s + 6) * SR]), SR) for s in (2, 12, 22)]
    assert max(lines) - min(lines) < 2.0
    assert stats["compressor_max_gr_db"] < 10.0


def test_master_hits_target_with_one_limiter_pass():
    from self_review import make_music
    n = SR * 32
    master, rep = M.mix_and_master(_voice(n, (0.1,) * 3), make_music(n, 0.03), SR, "shorts")
    q = rep["master_qc"]
    assert abs(q["integrated_lufs"] - (-14.0)) < 0.2
    assert q["true_peak_dbtp"] <= -0.98
    assert rep["dialogue_mix"]["limiter_max_gain_reduction_db"] <= 6.0


def test_music_longer_than_voice_is_not_cut():
    n = SR * 10
    v = np.zeros(n)
    v[SR:4 * SR] = 0.1 * np.random.default_rng(0).standard_normal(3 * SR)
    music = 0.05 * np.random.default_rng(1).standard_normal((n + 3 * SR, 2))
    master, _ = M.mix_and_master(v, music, SR, "podcast")
    assert len(master) == n + 3 * SR


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_")]
    for name, fn in tests:
        fn()
        print("PASS", name)
    print(f"{len(tests)} tests passed")
