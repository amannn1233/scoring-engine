#!/usr/bin/env python3
"""
Story Scoring Lab
=================
Standalone research harness for experimenting with adaptive AI scoring before
integrating anything into ShortMakerPro.

Design:
    script -> story beat analysis -> per-beat score generation -> score assembly
           -> optional Kaggle voice -> voice + score mix -> optional test video

Kaggle:
    --smp-root points at the existing ShortMakerPro folder. The lab imports
    kaggle_bridge.generate_from_kaggle from there, so it reuses the existing
    persistent WebSocket/bridge instead of copying credentials.

Music:
    Stability AI Stable Audio 2.5 is supported through its documented REST API.
    Set STABILITY_API_KEY in the environment. The provider is intentionally
    isolated so a local/Kaggle music model can replace it later without
    changing the experiment pipeline.

This is intentionally a research tool, not an SMP integration.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional


LAB_VERSION = "0.1.0"
CACHE_DIRNAME = "cache"
OUTPUT_DIRNAME = "outputs"


def log(msg: str) -> None:
    print(f"[SCORE LAB] {msg}", flush=True)


def require_ffmpeg() -> None:
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
            timeout=10,
        )
    except Exception as exc:
        raise RuntimeError(
            "FFmpeg is required and must be available on PATH."
        ) from exc


def run_cmd(cmd: list[str], *, timeout: int = 900) -> subprocess.CompletedProcess:
    log("RUN " + " ".join(f'"{x}"' if " " in x else x for x in cmd))
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          text=True, timeout=timeout)


@dataclass
class StoryBeat:
    index: int
    start_sentence: int
    end_sentence: int
    text: str
    label: str
    emotion: str
    energy: float
    tension: float
    density: float
    music_direction: str


def _sentences(script: str) -> list[str]:
    text = re.sub(r"\s+", " ", script.strip())
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


TENSION_WORDS = {
    "afraid", "anxious", "anxiety", "blood", "caught", "danger", "dead",
    "death", "discover", "discovered", "fear", "followed", "hiding", "lie",
    "lied", "missing", "panic", "secret", "scared", "scream", "suspicious",
    "threat", "trapped", "truth", "warning", "wrong",
}
REVEAL_WORDS = {
    "actually", "finally", "found", "realized", "realised", "revealed",
    "reveal", "turns", "turned", "then", "truth", "until", "discovered",
    "confession", "because", "but",
}
EMOTIONAL_WORDS = {
    "love", "loved", "hate", "heart", "cry", "cried", "sad", "happy",
    "regret", "sorry", "lonely", "hurt", "miss", "missed", "family",
}
ACTION_WORDS = {
    "ran", "run", "hit", "fight", "fought", "chased", "opened", "grabbed",
    "called", "left", "arrived", "saw", "heard", "screamed",
}


def analyze_story(script: str, target_beats: int = 5) -> list[StoryBeat]:
    """
    Deliberately transparent baseline analyzer.
    It is not pretending to understand the story like an LLM. It creates
    reproducible beat candidates so the music experiment can start now.
    """
    sentences = _sentences(script)
    if not sentences:
        return []

    n = max(3, min(int(target_beats), len(sentences)))
    beats: list[StoryBeat] = []

    # Evenly partition sentences, then adjust labels from lexical signals.
    cuts = [round(i * len(sentences) / n) for i in range(n + 1)]
    for i in range(n):
        a, b = cuts[i], cuts[i + 1]
        chunk = sentences[a:b]
        if not chunk:
            continue
        text = " ".join(chunk)
        words = re.findall(r"[a-z']+", text.lower())
        wordset = set(words)

        tension_hits = len(wordset & TENSION_WORDS)
        reveal_hits = len(wordset & REVEAL_WORDS)
        emotion_hits = len(wordset & EMOTIONAL_WORDS)
        action_hits = len(wordset & ACTION_WORDS)

        position = i / max(1, n - 1)
        tension = min(1.0, 0.15 + 0.16 * i + 0.08 * tension_hits + 0.05 * reveal_hits)
        energy = min(1.0, 0.20 + 0.05 * i + 0.07 * action_hits + 0.04 * reveal_hits)
        density = min(1.0, 0.20 + min(0.45, len(words) / 220.0) + 0.05 * action_hits)

        if reveal_hits >= 2 or (i == n - 2 and tension > 0.65):
            label = "reveal"
            emotion = "shock"
        elif i == 0:
            label = "setup"
            emotion = "curiosity"
        elif tension_hits >= 1 or tension > 0.55:
            label = "tension"
            emotion = "anxiety"
        elif emotion_hits >= 2:
            label = "aftermath"
            emotion = "emotion"
        elif i == n - 1:
            label = "aftermath"
            emotion = "reflection"
        else:
            label = "conflict"
            emotion = "unease"

        # Keep the music restrained because narration is primary.
        if label == "setup":
            direction = "dark ambient bed, sparse piano, soft texture, low energy, no vocals"
        elif label == "tension":
            direction = "dark cinematic ambient, low pulse, subtle sub bass, restrained percussion, rising tension, no vocals"
        elif label == "reveal":
            direction = "dark cinematic tension, deep bass pulse, dramatic texture, controlled impact, no vocals, no bright melody"
        elif label == "aftermath":
            direction = "minimal emotional ambient, soft piano texture, warm pad, sparse arrangement, reflective, no vocals"
        else:
            direction = "moody cinematic underscore, restrained pulse, atmospheric texture, moderate energy, no vocals"

        beats.append(
            StoryBeat(
                index=i,
                start_sentence=a,
                end_sentence=b,
                text=text,
                label=label,
                emotion=emotion,
                energy=round(energy, 3),
                tension=round(tension, 3),
                density=round(density, 3),
                music_direction=direction,
            )
        )

    return beats


def load_kaggle_bridge(smp_root: Path):
    bridge = smp_root / "kaggle_bridge.py"
    if not bridge.is_file():
        raise FileNotFoundError(
            f"Could not find {bridge}. Pass --smp-root pointing at the "
            "ShortMakerPro installation containing kaggle_bridge.py."
        )
    spec = importlib.util.spec_from_file_location("smp_kaggle_bridge", bridge)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {bridge}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class VoiceProvider:
    def __init__(self, smp_root: Optional[Path]):
        self.smp_root = smp_root

    def generate(self, voice: str, text: str, seed: int, output: Path) -> Path:
        if self.smp_root is None:
            raise RuntimeError("--smp-root is required for Kaggle voice generation.")
        bridge = load_kaggle_bridge(self.smp_root)
        # Existing production bridge accepts A/B. Keep this exact and explicit
        # rather than pretending arbitrary SMP voice IDs are supported here.
        voice = str(voice).strip().upper()
        if voice not in {"A", "B"}:
            raise ValueError(
                "The currently verified kaggle_bridge.generate_from_kaggle "
                "accepts voice A or B. Add another bridge adapter when needed."
            )
        result = bridge.generate_from_kaggle(voice=voice, text=text, seed=seed)
        result = Path(result).resolve()
        if not result.is_file():
            raise FileNotFoundError(f"Kaggle bridge returned missing WAV: {result}")
        output.parent.mkdir(parents=True, exist_ok=True)
        if result.resolve() != output.resolve():
            output.write_bytes(result.read_bytes())
        return output


class StabilityAudioProvider:
    """
    Stable Audio 2.5 text-to-audio adapter.
    The endpoint/model are isolated here so the lab can later swap in a
    Kaggle-hosted/local provider.
    """

    ENDPOINT = "https://api.stability.ai/v2beta/audio/stable-audio-2/text-to-audio"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("STABILITY_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "STABILITY_API_KEY is not set. Set it in the environment or "
                "use --music-provider folder for offline assembly tests."
            )

    def generate(
        self,
        prompt: str,
        duration: float,
        seed: int,
        output: Path,
    ) -> Path:
        try:
            import requests
        except ImportError as exc:
            raise RuntimeError(
                "The music provider needs requests. Install requirements.txt."
            ) from exc

        output.parent.mkdir(parents=True, exist_ok=True)
        response = requests.post(
            self.ENDPOINT,
            headers={
                "authorization": f"Bearer {self.api_key}",
                "accept": "audio/*",
            },
            files={"none": ""},
            data={
                "prompt": prompt,
                "output_format": "wav",
                "duration": max(1, min(180, int(round(duration)))),
                "model": "stable-audio-2.5",
                "seed": int(seed),
            },
            timeout=600,
        )
        if response.status_code != 200:
            try:
                detail = response.json()
            except Exception:
                detail = response.text
            raise RuntimeError(f"Stable Audio request failed ({response.status_code}): {detail}")
        output.write_bytes(response.content)
        return output


class FolderMusicProvider:
    """Offline provider: use pre-generated music clips for assembly experiments."""

    def __init__(self, folder: Path):
        self.folder = folder
        self.files = sorted(
            [p for p in folder.iterdir() if p.suffix.lower() in {".wav", ".mp3", ".flac", ".m4a"}]
        )
        if not self.files:
            raise RuntimeError(f"No audio files found in {folder}")

    def generate(self, prompt: str, duration: float, seed: int, output: Path) -> Path:
        # Deterministic selection is intentional for controlled A/B tests.
        idx = int(hashlib.sha256(prompt.encode()).hexdigest(), 16) % len(self.files)
        src = self.files[idx]
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(src.read_bytes())
        return output


def cache_key(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode()
    return hashlib.sha256(raw).hexdigest()[:24]


def media_duration(path: Path) -> float:
    cp = run_cmd([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path)
    ], timeout=30)
    if cp.returncode != 0:
        raise RuntimeError(cp.stderr.strip() or f"ffprobe failed for {path}")
    return float(cp.stdout.strip())


def render_segment_to_duration(src: Path, duration: float, out: Path) -> Path:
    """
    Make a segment exactly duration seconds. If it is longer, trim it.
    If shorter, loop it. This is only a duration tool; musical continuity
    is handled by crossfading in assemble_music().
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    src_dur = media_duration(src)
    if src_dur >= duration - 0.02:
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(src), "-t", f"{duration:.3f}",
            "-ar", "44100", "-ac", "2", "-c:a", "pcm_s16le", str(out),
        ]
    else:
        loops = max(1, math.ceil(duration / max(0.01, src_dur)))
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-stream_loop", str(loops - 1), "-i", str(src),
            "-t", f"{duration:.3f}",
            "-ar", "44100", "-ac", "2", "-c:a", "pcm_s16le", str(out),
        ]
    cp = run_cmd(cmd, timeout=300)
    if cp.returncode != 0:
        raise RuntimeError(cp.stderr.strip() or "FFmpeg segment render failed")
    return out


def assemble_music(segment_paths: list[Path], durations: list[float], out: Path, crossfade: float = 1.5) -> Path:
    if not segment_paths:
        raise ValueError("No music segments.")
    out.parent.mkdir(parents=True, exist_ok=True)

    normalized = []
    with tempfile.TemporaryDirectory(prefix="smp_score_norm_") as td:
        td_path = Path(td)
        for i, (src, dur) in enumerate(zip(segment_paths, durations)):
            p = td_path / f"seg_{i:02d}.wav"
            render_segment_to_duration(src, dur, p)
            normalized.append(p)

        if len(normalized) == 1:
            cmd = [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-i", str(normalized[0]), "-ar", "44100", "-ac", "2",
                "-c:a", "pcm_s16le", str(out),
            ]
        else:
            # Build sequential acrossfades. We keep the requested segment
            # lengths, subtracting each crossfade once from the total.
            inputs = []
            for p in normalized:
                inputs += ["-i", str(p)]
            filters = []
            current = "[0:a]"
            elapsed = durations[0]
            for i in range(1, len(normalized)):
                nxt = f"[{i}:a]"
                label = f"[xf{i}]"
                d = min(crossfade, durations[i - 1] / 2, durations[i] / 2)
                offset = max(0.0, elapsed - d)
                filters.append(
                    f"{current}{nxt}acrossfade=d={d:.3f}:c1=tri:c2=tri{label}"
                )
                current = label
                elapsed += durations[i] - d
            filter_complex = ";".join(filters)
            cmd = [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                *inputs,
                "-filter_complex", filter_complex,
                "-map", current,
                "-ar", "44100", "-ac", "2", "-c:a", "pcm_s16le", str(out),
            ]

        cp = run_cmd(cmd, timeout=600)
        if cp.returncode != 0:
            raise RuntimeError(cp.stderr.strip() or "FFmpeg music assembly failed")
    return out


def mix_voice_and_score(voice: Path, score: Path, out: Path, music_db: float = -28.0) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    score_gain = 10 ** (music_db / 20.0)
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(voice), "-i", str(score),
        "-filter_complex",
        (
            f"[1:a]volume={score_gain:.8f}[m];"
            "[0:a]aresample=44100[voice];"
            "[m]aresample=44100[music];"
            "[voice][music]amix=inputs=2:duration=first:dropout_transition=0:"
            "normalize=0[mix]"
        ),
        "-map", "[mix]", "-ar", "44100", "-ac", "2", "-c:a", "pcm_s16le",
        str(out),
    ]
    cp = run_cmd(cmd, timeout=600)
    if cp.returncode != 0:
        raise RuntimeError(cp.stderr.strip() or "FFmpeg voice/music mix failed")
    return out


def build_prompts(beats: list[StoryBeat], global_style: str) -> list[str]:
    prompts = []
    for beat in beats:
        prompts.append(
            f"{global_style}. "
            f"Scene function: {beat.label}. Emotion: {beat.emotion}. "
            f"Energy {beat.energy:.2f}/1, tension {beat.tension:.2f}/1, density {beat.density:.2f}/1. "
            f"Musical direction: {beat.music_direction}. "
            "Instrumental underscore for spoken narration. "
            "No vocals, no spoken words, no dominant lead melody, no abrupt ending."
        )
    return prompts


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def run_experiment(args) -> Path:
    require_ffmpeg()

    script_path = Path(args.script).resolve()
    script = script_path.read_text(encoding="utf-8")
    beats = analyze_story(script, args.beats)

    run_id = time.strftime("%Y%m%d_%H%M%S")
    root = Path(args.output).resolve() / run_id
    cache = root / CACHE_DIRNAME
    outputs = root / OUTPUT_DIRNAME
    cache.mkdir(parents=True, exist_ok=True)
    outputs.mkdir(parents=True, exist_ok=True)

    prompts = build_prompts(beats, args.style)

    save_json(root / "story_analysis.json", {
        "lab_version": LAB_VERSION,
        "script": str(script_path),
        "beat_count": len(beats),
        "beats": [asdict(b) for b in beats],
        "prompts": prompts,
        "settings": vars(args),
    })

    log(f"Story analyzed into {len(beats)} beats.")
    for b in beats:
        log(
            f"Beat {b.index + 1}: {b.label} | {b.emotion} | "
            f"energy={b.energy:.2f} tension={b.tension:.2f}"
        )

    voice_path = None
    if args.generate_voice:
        voice_cache = cache / f"voice_{args.voice}_{cache_key({'script': script, 'voice': args.voice, 'seed': args.seed})}.wav"
        if voice_cache.exists():
            log(f"Voice cache hit: {voice_cache.name}")
            voice_path = voice_cache
        else:
            log(f"Generating voice via existing SMP Kaggle bridge: Voice {args.voice}")
            voice_path = VoiceProvider(Path(args.smp_root)).generate(
                args.voice, script, args.seed, voice_cache
            )

    provider = None
    if args.music_provider == "stability":
        provider = StabilityAudioProvider()
    elif args.music_provider == "folder":
        provider = FolderMusicProvider(Path(args.music_folder))
    elif args.music_provider == "none":
        provider = None

    segment_paths: list[Path] = []
    durations: list[float] = []

    if provider:
        total_duration = media_duration(voice_path) if voice_path else args.target_duration
        # Initial experiment: equal story-beat time allocation. Later we can
        # derive exact timing from WhisperX word timings.
        weights = [max(0.25, 0.6 + b.density) for b in beats]
        weight_sum = sum(weights)
        durations = [total_duration * w / weight_sum for w in weights]

        for beat, prompt, duration in zip(beats, prompts, durations):
            key = cache_key({
                "provider": args.music_provider,
                "prompt": prompt,
                "duration": round(duration, 2),
                "seed": args.seed + beat.index,
            })
            seg = cache / f"score_{beat.index:02d}_{key}.wav"
            if seg.exists():
                log(f"Score cache hit: {seg.name}")
            else:
                log(f"Generating score beat {beat.index + 1} ({duration:.1f}s)")
                provider.generate(
                    prompt=prompt,
                    duration=duration,
                    seed=args.seed + beat.index,
                    output=seg,
                )
            segment_paths.append(seg)

        score_path = outputs / "adaptive_score.wav"
        assemble_music(segment_paths, durations, score_path, args.crossfade)
    else:
        score_path = None

    mixed_path = None
    if voice_path and score_path:
        mixed_path = outputs / "voice_plus_score.wav"
        mix_voice_and_score(voice_path, score_path, mixed_path, args.music_db)

    manifest = {
        "run_id": run_id,
        "voice_path": str(voice_path) if voice_path else None,
        "score_path": str(score_path) if score_path else None,
        "mixed_path": str(mixed_path) if mixed_path else None,
        "beats": [asdict(b) for b in beats],
        "segment_durations": durations,
        "music_db": args.music_db,
    }
    save_json(root / "manifest.json", manifest)

    log(f"Experiment complete: {root}")
    if voice_path:
        log(f"VOICE : {voice_path}")
    if score_path:
        log(f"SCORE : {score_path}")
    if mixed_path:
        log(f"MIX   : {mixed_path}")
    return root


def main():
    p = argparse.ArgumentParser(description="ShortMakerPro Story Scoring Lab")
    p.add_argument("--script", required=True, help="UTF-8 text file containing the story")
    p.add_argument("--smp-root", help="ShortMakerPro folder containing kaggle_bridge.py")
    p.add_argument("--voice", default="A", choices=["A", "B"])
    p.add_argument("--seed", type=int, default=48103)
    p.add_argument("--beats", type=int, default=5)
    p.add_argument("--style", default="cinematic dark minimal underscore")
    p.add_argument("--music-provider", choices=["stability", "folder", "none"], default="stability")
    p.add_argument("--music-folder", help="Folder used by --music-provider folder")
    p.add_argument("--target-duration", type=float, default=142.0,
                   help="Used only when no voice is generated")
    p.add_argument("--crossfade", type=float, default=1.5)
    p.add_argument("--music-db", type=float, default=-28.0)
    p.add_argument("--generate-voice", action="store_true")
    p.add_argument("--output", default="score_lab_runs")

    args = p.parse_args()

    if args.music_provider == "folder" and not args.music_folder:
        p.error("--music-folder is required with --music-provider folder")
    run_experiment(args)


if __name__ == "__main__":
    main()
