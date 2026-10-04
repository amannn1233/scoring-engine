#!/usr/bin/env python3
# ============================================================
# OFFLINE V2 REPLAY — no GPU, no Stable Audio
#
# Re-scores stories whose phrases were already generated (e.g. the
# Generalization V1 run) with the V2 exact-waveform optimizer, then checks
# that the optimizer's reported metrics match the final waveform.
#
#   python replay_v2.py --source ../story_generalization_v1 --out ../story_generalization_v2_replay
#
# Voice for ducking/preview (first match wins):
#   1. --voice-dir containing the original narration (71_extracted.wav, ...)
#   2. recovered from the source run's voice_music_preview.wav and
#      score_ducked.wav (preview = k * (voice + ducked)); the recovery is
#      validated by re-deriving the source run's own ducking numbers.
# ============================================================

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from scoring_engine import (  # noqa: E402
    SR, LAYER_NAMES, PHRASES_PER_LAYER, f32, fit_length, to_mono,
    render_score, measure_score, duck_envelope, normalize_layer_input,
)
from audio_io import read_audio  # noqa: E402
from story_analysis import normalized_story_id  # noqa: E402
from pipeline import score_story, summarize, save_json, load_voice_mono  # noqa: E402


def load_phrases(story_dir):
    phrases = {}
    for name in LAYER_NAMES:
        phrases[name] = []
        for i in range(1, PHRASES_PER_LAYER + 1):
            audio, sr = read_audio(story_dir / f"{name}_phrase_{i:02d}.wav")
            if sr != SR:
                raise RuntimeError(f"{story_dir}: phrase sample rate {sr} != {SR}")
            phrases[name].append(audio)
    return phrases


def recover_voice(story_dir, total_samples):
    preview, _ = read_audio(story_dir / "voice_music_preview.wav")
    ducked, _ = read_audio(story_dir / "score_ducked.wav")
    preview = fit_length(preview, total_samples).astype(np.float64)
    ducked = fit_length(ducked, total_samples).astype(np.float64)
    # Voice is identical in both channels, so L-R isolates k * (ducked_L - ducked_R).
    dp = preview[:, 0] - preview[:, 1]
    dd = ducked[:, 0] - ducked[:, 1]
    k = float(np.dot(dp, dd) / max(np.dot(dd, dd), 1e-12))
    voice = (preview / k - ducked).mean(axis=1)
    return f32(voice), k


def find_voice_file(voice_dir, story_id):
    if not voice_dir:
        return None
    candidates = sorted(Path(voice_dir).glob("*"))
    wavs = [p for p in candidates if p.is_file() and normalized_story_id(p) == story_id]
    wavs.sort(key=lambda p: (p.suffix.lower() != ".wav", p.name))
    return wavs[0] if wavs else None


def replay_story(index, src_dir, out_root, voice_dir=None, log=print):
    src_dir = Path(src_dir)
    source = json.load(open(src_dir / "manifest.json", encoding="utf-8"))
    boundaries = [float(x) for x in source["boundaries"]]
    total_samples = int(round(float(source["duration_seconds"]) * SR))
    story_id = normalized_story_id(source["story_file"])

    log("\n" + "=" * 80)
    log(f"REPLAY {index}: {src_dir.name}  (story {story_id}, {total_samples / SR:.2f}s)")
    log("=" * 80)

    phrases = normalize_layer_input(load_phrases(src_dir))

    # --- Sanity: V2 renderer reproduces the source run's own score_raw.wav.
    src_opt = source["optimizer"]
    rerender = render_score(phrases, src_opt["orders"], src_opt["phases_seconds"], boundaries, total_samples)
    src_raw, _ = read_audio(src_dir / "score_raw.wav")
    src_raw = fit_length(src_raw, total_samples)
    pcm_diff = float(np.max(np.abs(np.clip(np.round(rerender * 32767), -32768, 32767) / 32768.0 - src_raw)) * 32768)
    rerender_metrics = measure_score(rerender, boundaries)
    src_final = [r["rms_db"] for r in source["final_story_energy"]]
    src_reported_opt = src_opt.get("metrics", {})
    source_check = {
        "renderer_reproduces_source_score_raw_max_pcm16_lsb": pcm_diff,
        "source_final_section_db": src_final,
        "v2_measure_of_source_config_section_db": rerender_metrics["section_db"],
        "max_section_db_diff": float(np.max(np.abs(np.array(src_final) - rerender_metrics["section_db"]))),
        "source_final_jumps_db": [r["jump_db"] for r in source["final_boundary_continuity"]],
        "source_optimizer_reported": {
            "objective": src_opt.get("objective"),
            "peak_gap_db": src_reported_opt.get("peak_gap_db"),
            "boundary_penalty": src_reported_opt.get("boundary_penalty"),
        },
        "source_config_true_metrics": {
            "objective": rerender_metrics["objective"],
            "peak_gap_db": rerender_metrics["peak_gap_db"],
            "boundary_penalty": rerender_metrics["boundary_penalty"],
            "max_abs_boundary_jump_db": rerender_metrics["max_abs_boundary_jump_db"],
        },
    }
    log(f"Source render reproduced within {pcm_diff:.0f} LSB; source optimizer claimed "
        f"objective {src_opt.get('objective'):.4f}, true {rerender_metrics['objective']:.4f}")

    # --- Voice.
    voice_file = find_voice_file(voice_dir, story_id)
    if voice_file is not None:
        voice = fit_length(load_voice_mono(voice_file), total_samples)
        voice_info = {"source": "original_narration_file", "path": str(voice_file)}
    else:
        voice, k = recover_voice(src_dir, total_samples)
        env, stats = duck_envelope(voice, total_samples)
        src_duck = source["ducking"]["average_duck_db"]
        voice_info = {
            "source": "recovered_from_source_preview_minus_ducked",
            "preview_normalization_gain": k,
            "check_recomputed_source_average_duck_db": stats["average_duck_db"],
            "check_source_manifest_average_duck_db": src_duck,
            "check_abs_diff_db": abs(stats["average_duck_db"] - src_duck),
        }
        log(f"Voice recovered from preview (k={k:.5f}); ducking check "
            f"{stats['average_duck_db']:.4f} vs source {src_duck:.4f} dB")

    out_dir = Path(out_root) / src_dir.name
    manifest_base = {
        "mode": "offline_replay_on_existing_phrases",
        "story_file": source["story_file"],
        "source_run_dir": str(src_dir),
        "source_phrase_files": {
            name: [str(src_dir / f"{name}_phrase_{i:02d}.wav") for i in range(1, PHRASES_PER_LAYER + 1)]
            for name in LAYER_NAMES
        },
        "transcript_file": source.get("transcript_file"),
        "detected_mood": source.get("detected_mood"),
        "motif": source.get("motif"),
        "quality_report": source.get("quality_report", []),
        "voice": voice_info,
        "source_run_check": source_check,
    }
    return score_story(phrases, boundaries, voice, out_dir, manifest_base, log=log)


def write_report(rows, manifests, out_root):
    lines = [
        "# Generalization V2 — offline replay on the V1 phrases",
        "",
        "Same phrases, same story boundaries and same objective as Generalization V1.",
        "Only the optimizer and render/measurement path changed.",
        "",
        "## Does the optimizer now match the final waveform?",
        "",
        "| Story | Optimizer objective | Final objective | Optimizer peak gap | Final peak gap | Worst diff vs render | Worst diff vs 16-bit file | Aligned |",
        "|---|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['file']} | {r['optimizer_objective']:.4f} | {r['final_objective']:.4f} | "
            f"{r['optimizer_peak_gap_db']:+.3f} dB | {r['revelation_minus_discovery_db']:+.3f} dB | "
            f"{r['alignment_worst_db_diff']:.6f} dB | {r['alignment_pcm16_worst_db_diff']:.4f} dB | "
            f"{'yes' if r['aligned'] else 'NO'} |"
        )
    lines += [
        "",
        "## V1 vs V2 on identical material",
        "",
        "V1 numbers are what the V1 optimizer reported, then the V1 final waveform. V2 numbers are the final waveform.",
        "",
        "| Story | V1 claimed peak gap | V1 actual peak gap | V2 peak gap | V1 max jump | V2 max jump | V1 true objective | V2 objective |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r, m in zip(rows, manifests):
        sc = m["source_run_check"]
        v1_claim = sc["source_optimizer_reported"]["peak_gap_db"]
        v1_true = sc["source_config_true_metrics"]
        lines.append(
            f"| {r['file']} | {v1_claim:+.2f} dB | {v1_true['peak_gap_db']:+.2f} dB | "
            f"{r['revelation_minus_discovery_db']:+.2f} dB | {v1_true['max_abs_boundary_jump_db']:.2f} dB | "
            f"{r['max_boundary_jump_db']:.2f} dB | {v1_true['objective']:.4f} | {r['final_objective']:.4f} |"
        )
    lines += ["", "## Final story energy (dB) — V2", "",
              "| Story | Hook | Setup | Consequence | Hidden | Discovery | Revelation | Aftermath |",
              "|---|---:|---:|---:|---:|---:|---:|---:|",
              "| target | -26.0 | -24.8 | -23.9 | -22.7 | -22.0 | -20.0 | -25.3 |"]
    for r in rows:
        lines.append(f"| {r['file']} | " + " | ".join(f"{x:.2f}" for x in r["section_db"]) + " |")
    lines += ["", "## Boundary jumps (dB) — V2", "",
              "| Story | Hook→Setup | Setup→Cons. | Cons.→Hidden | Hidden→Disc. | Disc.→Rev. | Rev.→After. |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['file']} | " + " | ".join(f"{x:+.2f}" for x in r["boundary_jumps_db"]) + " |")
    lines += ["", "## Other checks", "",
              "| Story | Local valley penalty | Motif sim. min / mean | Avg duck | Voice source | Renderer reproduces V1 file |",
              "|---|---:|---:|---:|---|---:|"]
    for r, m in zip(rows, manifests):
        v = m["voice"]
        voice = "original narration" if v["source"] == "original_narration_file" else \
            f"recovered (duck check Δ {v['check_abs_diff_db']:.4f} dB)"
        sims = "n/a" if r["motif_similarity_min"] is None else \
            f"{r['motif_similarity_min']:.3f} / {r['motif_similarity_mean']:.3f}"
        lines.append(
            f"| {r['file']} | {r['local_penalty']:.4f} | {sims} | {r['average_duck_db']:.2f} dB | {voice} | "
            f"≤ {m['source_run_check']['renderer_reproduces_source_score_raw_max_pcm16_lsb']:.0f} LSB |"
        )
    (Path(out_root) / "REPLAY_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Offline V2 replay on existing phrases")
    parser.add_argument("--source", default=str(HERE.parent / "story_generalization_v1"))
    parser.add_argument("--out", default=str(HERE.parent / "story_generalization_v2_replay"))
    parser.add_argument("--voice-dir", default=None)
    args = parser.parse_args(argv)

    source = Path(args.source)
    story_dirs = sorted(p for p in source.iterdir() if p.is_dir() and (p / "manifest.json").exists())
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    started = time.time()
    manifests = [replay_story(i, d, out_root, args.voice_dir) for i, d in enumerate(story_dirs, start=1)]
    rows = summarize(manifests)
    save_json(out_root / "replay_summary.json", rows)
    write_report(rows, manifests, out_root)
    print(f"\nReplay finished in {time.time() - started:.1f}s -> {out_root}")
    for r in rows:
        print(f"  {r['file']}: optimizer {r['optimizer_objective']:.4f} final {r['final_objective']:.4f} "
              f"peak {r['revelation_minus_discovery_db']:+.2f} dB max jump {r['max_boundary_jump_db']:.2f} dB "
              f"worst diff {r['alignment_worst_db_diff']:.2e} dB aligned={r['aligned']}")
    return rows


if __name__ == "__main__":
    main()
