# ============================================================
# STORY SCORING LAB — GENERALIZATION V2 PIPELINE
#
# narration -> transcript -> story cue sheet (vibe, palette, tempo, key,
# arc, hit points) -> 7 story sections -> fresh motif (best of N takes) ->
# reference windows -> core/pressure/climax phrases (best of N takes) ->
# V2 exact-waveform optimizer -> exact render -> adaptive ducking ->
# score_raw / score_ducked / voice_music_preview / manifest.json
#
# Kaggle: paste the built single file (dist/) into a cell, or run
#   python pipeline.py --input-root /kaggle/input --stories 71,72,86
# ============================================================

import argparse
import json
import os
import zipfile
from pathlib import Path

import numpy as np

# <local-imports>
from scoring_engine import (
    SR, LAYER_NAMES, STORY_LABELS, SECTION_TARGETS_DB, ALIGNMENT_TOLERANCE_DB,
    f32, fit_length, normalize_layer_input, optimize_story, render_score, measure_score,
    alignment_report, duck_envelope, apply_ducking, make_preview, to_mono,
)
from audio_io import read_audio, write_audio, resample_mono
from story_analysis import (
    discover_stories, find_matching_json, load_transcript_json, transcribe_with_whisper,
    normalize_text, build_story_mood, infer_story_boundaries, structural_turn_estimate,
)
from music_generation import (
    generate_motif, choose_references, chroma_profile, generate_layer_phrases, OVERHANG_SECONDS,
)
from run_seed import generate_run_seed
from cue_sheet import adopt_motif_tempo_key, build_cue_sheet, check_turn, describe as describe_cue
from music_analysis import estimate_key as music_key, estimate_tempo as music_tempo
from arrangement import arrange_story, plan_grid
from mix_engineering import mix_and_master, qc_report, format_qc, DEFAULT_DELIVERY, DELIVERY_SPECS
# </local-imports>


PIPELINE_VERSION = "GENERALIZATION_V2_EXACT_WAVEFORM"

DEFAULT_INPUT_ROOTS = ["/kaggle/input"]
DEFAULT_OUT_ROOT = "/kaggle/working/story_generalization_v2_exact_waveform"
DEFAULT_TEST_STORY_IDS = ["71", "72", "86"]
DEFAULT_MAX_STORIES = 3


def _json_default(value):
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value)}")


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=_json_default)


def load_voice_mono(path):
    audio, sr = read_audio(path)
    return f32(resample_mono(to_mono(audio), sr, SR))


# ============================================================
# SHARED SCORING STAGE (used by the full run AND by the offline replay)
# ============================================================

def score_story(layer_phrases, boundaries, voice_mono, story_dir, manifest_base, log=print,
                delivery=DEFAULT_DELIVERY):
    """Optimize -> exact render -> measure -> duck -> export.

    The manifest's optimizer metrics and final metrics come from the same
    render_score()/measure_score() code, then the written score_raw.wav is
    read back and measured a third time (16-bit file)."""
    story_dir = Path(story_dir)
    story_dir.mkdir(parents=True, exist_ok=True)
    voice_mono = f32(voice_mono)
    total_samples = len(voice_mono)
    boundaries = [float(x) for x in boundaries]
    phrases = normalize_layer_input(layer_phrases)

    log("\nOptimizing actual generated material (V2 exact waveform)...")
    opt = optimize_story(phrases, boundaries, total_samples=total_samples, log=log)

    log("\nBest phrase orders:")
    for name in LAYER_NAMES:
        log(f"  {name:<10}{tuple(opt['orders'][name])} phase={opt['phases_seconds'][name]:.2f}s")

    # Final exact render — fresh, from the phrases, not from optimizer caches.
    final_music = render_score(phrases, opt["orders"], opt["phases_seconds"], boundaries, total_samples)
    final_metrics = measure_score(final_music, boundaries)

    raw_path = story_dir / "score_raw.wav"
    write_audio(raw_path, final_music, SR)
    file_audio, _ = read_audio(raw_path)
    file_metrics = measure_score(fit_length(file_audio, total_samples), boundaries)

    duck_env, duck_stats = duck_envelope(voice_mono, total_samples)
    ducked = apply_ducking(final_music, duck_env)
    ducked_path = story_dir / "score_ducked.wav"
    write_audio(ducked_path, ducked, SR)

    preview = make_preview(voice_mono, ducked)
    preview_path = story_dir / "voice_music_preview.wav"
    write_audio(preview_path, preview, SR)

    # Engineered dialogue mix + delivery master (additive: the legacy
    # ducked/preview files above are unchanged).
    master, mix_report = mix_and_master(voice_mono, final_music, SR, delivery)
    master_path = story_dir / "voice_music_master.wav"
    write_audio(master_path, master, SR)
    mix_report["score_raw_qc"] = qc_report(final_music, SR, label="score_raw")
    mix_report["legacy_preview_qc"] = qc_report(
        preview, SR, DELIVERY_SPECS[delivery], label="legacy_preview")

    align = alignment_report(opt["metrics"], final_metrics)
    align_file = alignment_report(opt["metrics"], file_metrics)

    section_rows = [
        {
            "section": label,
            "start": boundaries[i],
            "end": boundaries[i + 1],
            "rms_db": final_metrics["section_db"][i],
            "target_db": float(SECTION_TARGETS_DB[i]),
            "error_db": final_metrics["section_db"][i] - float(SECTION_TARGETS_DB[i]),
        }
        for i, label in enumerate(STORY_LABELS)
    ]
    boundary_rows = [
        {
            "boundary": f"{STORY_LABELS[i - 1]} -> {STORY_LABELS[i]}",
            "time": boundaries[i],
            "jump_db": final_metrics["boundary_jumps_db"][i - 1],
        }
        for i in range(1, 7)
    ]

    manifest = dict(manifest_base)
    manifest.update({
        "optimizer_version": opt["optimizer_version"],
        "pipeline_version": PIPELINE_VERSION,
        "duration_seconds": total_samples / SR,
        "boundaries": boundaries,
        "optimizer": opt,
        "final_metrics": final_metrics,
        "final_file_metrics_pcm16": file_metrics,
        "alignment": {
            "optimizer_vs_final_render": align,
            "optimizer_vs_written_pcm16_file": align_file,
        },
        "final_story_energy": section_rows,
        "final_boundary_continuity": boundary_rows,
        "ducking": duck_stats,
        "mix_master": mix_report,
        "files": {"raw": str(raw_path), "ducked": str(ducked_path), "preview": str(preview_path),
                  "master": str(master_path)},
    })
    save_json(story_dir / "manifest.json", manifest)

    log("\nFINAL STORY RESULT")
    log(f"Duration: {total_samples / SR:.3f}s")
    log(f"Optimizer exact objective: {opt['objective']:.4f}   final objective: {final_metrics['objective']:.4f}")
    log(f"Peak gap Discovery -> Revelation: optimizer {opt['metrics']['peak_gap_db']:+.3f} dB, "
        f"final {final_metrics['peak_gap_db']:+.3f} dB")
    log("\nEnergy:")
    for row in section_rows:
        log(f"  {row['section']:<28}{row['rms_db']:8.2f} dB   (target {row['target_db']:.1f})")
    log("\nBoundaries:")
    for row in boundary_rows:
        log(f"  {row['boundary']}: {row['jump_db']:+.2f} dB")
    log(f"\nAlignment optimizer vs final: worst {align['worst_db_diff']:.6f} dB -> "
        f"{'ALIGNED' if align['aligned'] else 'MISMATCH'}")
    log(f"Alignment optimizer vs 16-bit file: worst {align_file['worst_db_diff']:.6f} dB")
    log(f"Average duck: {duck_stats['average_duck_db']:.2f} dB")
    dm = mix_report["dialogue_mix"]
    log(f"\nDialogue mix: voice {dm['voice_speech_lufs']:.1f} LUFS, DMR {dm['achieved_dmr_lu_median']:.1f} LU "
        f"(target {dm['target_dmr_lu']:.0f}), duck under speech {dm['average_duck_db_during_speech']:.1f} dB, "
        f"limiter {dm['limiter_max_gain_reduction_db']:.1f} dB")
    for line in format_qc(mix_report["master_qc"]) + format_qc(mix_report["legacy_preview_qc"]):
        log(line)
    return manifest


def score_story_arranged(layer_phrases, cue, boundaries, voice_mono, story_dir, manifest_base, log=print,
                         delivery=DEFAULT_DELIVERY):
    """Story-aware scoring: bar-grid arrangement driven by the cue sheet
    (arrangement.py), then the engineered dialogue mix + master. The score
    runs past the narration and ends on a bar line, so the narration is
    padded with silence to the music's length for the mix."""
    story_dir = Path(story_dir)
    story_dir.mkdir(parents=True, exist_ok=True)
    voice_mono = f32(voice_mono)
    voice_seconds = len(voice_mono) / SR

    log("\nArranging the score on the story's tempo grid...")
    music, arrangement, _ = arrange_story(layer_phrases, cue, boundaries, voice_seconds, log=log)
    again, _, _ = arrange_story(layer_phrases, cue, boundaries, voice_seconds, log=lambda *_: None)
    deterministic = bool(np.array_equal(music, again))
    del again

    raw_path = story_dir / "score_raw.wav"
    write_audio(raw_path, music, SR)
    voice_padded = fit_length(voice_mono, len(music))

    master, mix_report = mix_and_master(voice_padded, music, SR, delivery)
    master_path = story_dir / "voice_music_master.wav"
    write_audio(master_path, master, SR, dither=True)
    mix_report["score_raw_qc"] = qc_report(music, SR, label="score_raw")

    g = arrangement["grid"]
    manifest = dict(manifest_base)
    manifest.update({
        "pipeline_version": PIPELINE_VERSION + "+STORY_ARRANGEMENT",
        "duration_seconds": voice_seconds,
        "music_seconds": arrangement["music_seconds"],
        "arrangement": arrangement,
        "arrangement_deterministic": deterministic,
        "mix_master": mix_report,
        "files": {"raw": str(raw_path), "master": str(master_path)},
    })
    save_json(story_dir / "manifest.json", manifest)

    log("\nFINAL STORY RESULT (story-aware)")
    log(f"Grid: {g['bpm']:.0f} BPM, {g['phrase_bars']}-bar phrases, bar {g['bar']:.3f}s")
    hp = arrangement["hit_points_on_grid"]
    log(f"Hits on bar lines: build {hp['build']:.2f}s, turn {hp['turn']:.2f}s, "
        f"aftermath {hp['aftermath']:.2f}s, end {hp['end']:.2f}s")
    fm = arrangement["final_metrics"]
    log(f"Arc objective {fm['objective']:.4f}, peak gap {fm['peak_gap_db']:+.2f} dB, "
        f"release {fm['release_drop_db']:+.2f} dB, deterministic render: {deterministic}")
    dm = mix_report["dialogue_mix"]
    log(f"Dialogue mix: DMR {dm['achieved_dmr_lu_median']:.1f} LU (target {dm['target_dmr_lu']:.0f}), "
        f"limiter {dm['limiter_max_gain_reduction_db']:.1f} dB")
    for line in format_qc(mix_report["master_qc"]):
        log(line)
    return manifest


# ============================================================
# FULL STORY (Stable Audio generation + scoring)
# ============================================================

def process_story(story_index, story_path, out_root, run_seed, log=print, story_aware=True):
    """run_seed: the ONE seed for this whole production job (all stories in
    the job share it). Every motif/phrase seed is derived from it, so this
    job's material differs from any other job's, while re-running this exact
    story_index with this exact run_seed reproduces it bit-for-bit."""
    story_dir = Path(out_root) / f"story_{story_index:02d}_{Path(story_path).stem}"
    story_dir.mkdir(parents=True, exist_ok=True)
    log("\n" + "=" * 80)
    log(f"STORY {story_index}: {os.path.basename(story_path)}  (run_seed={run_seed})")
    log("=" * 80)

    voice_mono = load_voice_mono(story_path)
    duration = len(voice_mono) / SR

    transcript_path = find_matching_json(story_path)
    segments = []
    if transcript_path is not None:
        try:
            segments = load_transcript_json(transcript_path)
            log(f"Transcript: {transcript_path}")
        except Exception as exc:
            log(f"Transcript JSON failed: {exc!r}")
    if not segments:
        segments = transcribe_with_whisper(story_path, log=log)
    segments = normalize_text(segments)

    mood = build_story_mood(segments)
    boundaries = infer_story_boundaries(segments, duration)
    log(f"Detected mood: {mood}")
    log("Story boundaries:")
    for i, label in enumerate(STORY_LABELS):
        log(f"  {label:<28}{boundaries[i]:7.2f}s -> {boundaries[i + 1]:7.2f}s")

    cue = build_cue_sheet(segments, duration, list(boundaries)) if story_aware else None
    turn_check = None
    if cue is not None:
        log("Story cue sheet:")
        for line in describe_cue(cue):
            log(f"  {line}")
        turn_check = check_turn(cue.hit_points["twist"], structural_turn_estimate(segments, duration), duration)
        log(f"  Turn check: {turn_check['note']}")

    motif, motif_path, prompt, motif_seed, motif_takes = generate_motif(
        story_index, mood, story_dir, run_seed, cue=cue, return_report=True, log=log)
    if cue is not None:
        # Every layer is conditioned on the motif, so the grid and layer
        # prompts follow the motif's measured pulse and key.
        bpm, conf = music_tempo(motif, SR)
        tonic, family, _ = music_key(motif, SR)
        cue = adopt_motif_tempo_key(cue, bpm, conf, tonic, family)
        log(f"Motif reads {bpm:.1f} BPM (confidence {conf:.2f}), {tonic} {family}: "
            f"grid {cue.tempo_bpm:g} BPM, {cue.key_label} (story asked {cue.story_tempo_bpm:g} BPM, "
            f"{cue.story_key_label})")
    references, reference_starts = choose_references(
        motif, None if cue is None else plan_grid(cue).generate_seconds + OVERHANG_SECONDS)
    for i, reference in enumerate(references, start=1):
        write_audio(story_dir / f"reference_{i:02d}.wav", reference, SR)
    motif_profile = chroma_profile(motif, numpy_fallback=cue is not None)

    if cue is None:
        layer_phrases, quality_rows = generate_layer_phrases(
            story_index, mood, references, motif_profile, story_dir, run_seed, log=log)
    else:
        layer_phrases, quality_rows, cue = generate_layer_phrases(
            story_index, mood, references, motif_profile, story_dir, run_seed, log=log, cue=cue,
            return_cue=True)

    manifest_base = {
        "story_file": str(story_path),
        "run_seed": run_seed,
        "transcript_file": transcript_path,
        "transcript_segments": len(segments),
        # The words the cue sheet read, so the self-review can re-read them.
        "transcript": [{"start": s["start"], "end": s["end"], "text": s["text"]} for s in segments],
        "turn_check": turn_check,
        "detected_mood": mood,
        "story_aware": story_aware,
        "cue_sheet": cue.to_dict() if cue is not None else None,
        "motif": {
            "path": motif_path,
            "prompt": prompt,
            "seed": motif_seed,
            "takes": motif_takes,
            "reference_starts_seconds": [float(x / SR) for x in reference_starts],
        },
        "quality_report": quality_rows,
    }
    if cue is not None:
        return score_story_arranged(layer_phrases, cue, boundaries, voice_mono, story_dir,
                                    manifest_base, log=log)
    return score_story(layer_phrases, boundaries, voice_mono, story_dir, manifest_base, log=log)


# ============================================================
# SUMMARY
# ============================================================

def _summarize_arranged(index, m):
    a = m["arrangement"]
    final = a["final_metrics"]
    qc = m["mix_master"]["master_qc"]
    takes = [r for r in m.get("quality_report", []) if r.get("selected")]
    fits = [r["musical_fit"] for r in takes if "musical_fit" in r]
    return {
        "story": index,
        "file": os.path.basename(m["story_file"]),
        "story_aware": True,
        "duration_sec": m["duration_seconds"],
        "music_sec": m["music_seconds"],
        "style": m["cue_sheet"]["style"],
        "tempo_bpm": m["cue_sheet"]["tempo_bpm"],
        "key": m["cue_sheet"]["key_label"],
        "palette": m["cue_sheet"]["palette"],
        "final_objective": final["objective"],
        "section_db": final["section_db"],
        "revelation_minus_discovery_db": final["peak_gap_db"],
        "release_drop_db": final["release_drop_db"],
        "max_boundary_jump_db": final["max_abs_boundary_jump_db"],
        "hit_points_on_grid": a["hit_points_on_grid"],
        "deterministic": m["arrangement_deterministic"],
        "selected_tempo_match_mean": float(np.mean([f["tempo_match"] for f in fits])) if fits else None,
        "selected_key_match_mean": float(np.mean([f["key_match"] for f in fits])) if fits else None,
        "selected_seam_mean": float(np.mean([f["seam"] for f in fits])) if fits else None,
        "master_integrated_lufs": qc["integrated_lufs"],
        "master_true_peak_dbtp": qc["true_peak_dbtp"],
        "master_lra_lu": qc["loudness_range_lu"],
        "dialogue_to_music_lu": m["mix_master"]["dialogue_mix"]["achieved_dmr_lu_median"],
        "speech_band_margin_db": qc.get("speech_band_masking_margin_db"),
        "master_warnings": qc["warnings"],
    }


def summarize(manifests):
    rows = []
    for index, m in enumerate(manifests, start=1):
        if "arrangement" in m:
            rows.append(_summarize_arranged(index, m))
            continue
        final = m["final_metrics"]
        opt = m["optimizer"]["metrics"]
        sims = [r.get("motif_similarity") for r in m.get("quality_report", [])
                if r.get("motif_similarity") is not None]
        valid = [r for r in m.get("quality_report", []) if r.get("valid")]
        rows.append({
            "story": index,
            "file": os.path.basename(m["story_file"]),
            "duration_sec": m["duration_seconds"],
            "optimizer_objective": m["optimizer"]["objective"],
            "final_objective": final["objective"],
            "stage1_objective": m["optimizer"]["stage1"]["objective"],
            "section_db": final["section_db"],
            "discovery_db": final["section_db"][4],
            "revelation_db": final["section_db"][5],
            "revelation_minus_discovery_db": final["peak_gap_db"],
            "optimizer_peak_gap_db": opt["peak_gap_db"],
            "aftermath_db": final["section_db"][6],
            "boundary_jumps_db": final["boundary_jumps_db"],
            "max_boundary_jump_db": final["max_abs_boundary_jump_db"],
            "optimizer_boundary_penalty": opt["boundary_penalty"],
            "final_boundary_penalty": final["boundary_penalty"],
            "local_penalty": final["local_penalty"],
            "local_valley_depth_db": final["local_valley_depth_db"],
            "alignment_worst_db_diff": m["alignment"]["optimizer_vs_final_render"]["worst_db_diff"],
            "alignment_pcm16_worst_db_diff": m["alignment"]["optimizer_vs_written_pcm16_file"]["worst_db_diff"],
            "aligned": m["alignment"]["optimizer_vs_final_render"]["aligned"],
            "motif_similarity_min": min(sims) if sims else None,
            "motif_similarity_mean": float(np.mean(sims)) if sims else None,
            "valid_phrase_reports": len(valid),
            "average_duck_db": m["ducking"]["average_duck_db"],
            "master_integrated_lufs": m["mix_master"]["master_qc"]["integrated_lufs"],
            "master_true_peak_dbtp": m["mix_master"]["master_qc"]["true_peak_dbtp"],
            "master_lra_lu": m["mix_master"]["master_qc"]["loudness_range_lu"],
            "dialogue_to_music_lu": m["mix_master"]["dialogue_mix"]["achieved_dmr_lu_median"],
            "speech_band_margin_db": m["mix_master"]["master_qc"].get("speech_band_masking_margin_db"),
            "master_warnings": m["mix_master"]["master_qc"]["warnings"],
        })
    return rows


def print_summary(rows, log=print):
    log("\n" + "=" * 80)
    log("GENERALIZATION V2 SUMMARY")
    log("=" * 80)
    for row in rows:
        log(f"\nStory {row['story']}: {row['file']}")
        if row.get("story_aware"):
            log(f"  {row['style']}")
            log(f"  {row['tempo_bpm']} BPM, {row['key']}; arc objective {row['final_objective']:.4f}, "
                f"peak gap {row['revelation_minus_discovery_db']:+.2f} dB")
            log(f"  Master {row['master_integrated_lufs']:.1f} LUFS, TP {row['master_true_peak_dbtp']:+.2f} dBTP, "
                f"DMR {row['dialogue_to_music_lu']:.1f} LU, warnings: {row['master_warnings'] or 'none'}")
            continue
        log(f"  Objective (optimizer / final): {row['optimizer_objective']:.4f} / {row['final_objective']:.4f}")
        log(f"  Discovery: {row['discovery_db']:.2f} dB   Revelation: {row['revelation_db']:.2f} dB")
        log(f"  Revelation - Discovery (optimizer / final): "
            f"{row['optimizer_peak_gap_db']:+.3f} / {row['revelation_minus_discovery_db']:+.3f} dB")
        log(f"  Aftermath: {row['aftermath_db']:.2f} dB")
        log(f"  Max boundary jump: {row['max_boundary_jump_db']:.2f} dB")
        log(f"  Optimizer vs final worst diff: {row['alignment_worst_db_diff']:.6f} dB "
            f"({'ALIGNED' if row['aligned'] else 'MISMATCH'})")
        log(f"  Average duck: {row['average_duck_db']:.2f} dB")


def zip_folder(folder, zip_path):
    folder = Path(folder)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in folder.rglob("*"):
            if path.is_file():
                zf.write(path, arcname=path.relative_to(folder))
    return zip_path


# ============================================================
# MAIN
# ============================================================

def main(argv=None):
    parser = argparse.ArgumentParser(description="Story Scoring Lab — Generalization V2")
    parser.add_argument("--input-root", action="append", default=None,
                        help="Folder(s) searched recursively for narration audio.")
    parser.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    parser.add_argument("--stories", default=",".join(DEFAULT_TEST_STORY_IDS),
                        help="Comma-separated story ids; empty string = first 3 eligible.")
    parser.add_argument("--max-stories", type=int, default=DEFAULT_MAX_STORIES)
    parser.add_argument("--no-zip", action="store_true")
    parser.add_argument("--run-seed", type=int, default=None,
                        help="Seed for this whole job (all stories in it). Omit for a fresh "
                             "production job (default); pass an explicit value to reproduce "
                             "a previous job's exact motif/phrases.")
    parser.add_argument("--legacy-prompts", action="store_true",
                        help="Use the original fixed V1/V2 prompts and first-valid-take logic "
                             "instead of the story cue sheet (reproduces the validated benchmark).")
    # parse_known_args: Kaggle/Jupyter pass their own -f argument.
    args, _ = parser.parse_known_args(argv)

    input_roots = args.input_root or DEFAULT_INPUT_ROOTS
    story_ids = [s.strip() for s in args.stories.split(",") if s.strip()]
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    run_seed = args.run_seed if args.run_seed is not None else generate_run_seed()
    print(f"\nRun seed for this job: {run_seed}"
          f"{' (explicit)' if args.run_seed is not None else ' (fresh — record this to reproduce this job)'}")

    story_files = discover_stories(input_roots, story_ids, args.max_stories)
    print("\nSelected stories:")
    for i, path in enumerate(story_files, start=1):
        print(f"{i}. {os.path.basename(path)}")

    manifests = [process_story(i, path, out_root, run_seed, story_aware=not args.legacy_prompts)
                 for i, path in enumerate(story_files, start=1)]

    rows = summarize(manifests)
    for row, m in zip(rows, manifests):
        row["run_seed"] = run_seed
    save_json(out_root / "generalization_summary.json", rows)
    print_summary(rows)

    if not args.no_zip:
        zip_path = out_root.parent / f"{out_root.name}.zip"
        zip_folder(out_root, zip_path)
        print(f"ZIP: {zip_path}")

    print("\n" + "=" * 80)
    print("GENERALIZATION V2 COMPLETE — EXACT WAVEFORM OPTIMIZER")
    print("=" * 80)
    print(f"Summary: {out_root / 'generalization_summary.json'}")
    return rows


if __name__ == "__main__":
    main()
