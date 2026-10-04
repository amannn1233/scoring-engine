#!/usr/bin/env python3
"""Run the story-aware pipeline end to end on REAL Stable Audio 3 Small
output (CPU or GPU), then score it with self_review.py --renders.

Narration audio isn't in the repo, so a synthetic narration track is timed
to each real transcript (the cue sheet, sections and hits come from the
real text; only the voice timbre is synthetic). The music is 100 % model
output.

    pip install torch git+https://github.com/Stability-AI/stable-audio-3.git
    export HF_TOKEN=...            # account that accepted the model licence
    export HF_HUB_DISABLE_XET=1    # if the xet CDN is blocked
    python real_render_e2e.py --out /mnt/project-files/real_e2e
    python real_render_e2e.py --out ... --stories workshop --takes 2   # quicker pass

Writes <out>/<story>/story_01_*/{score_raw.wav, voice_music_master.wav,
manifest.json, *_phrase_*.wav, motif_seed_28s.wav} and <out>/self_review.json.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import music_generation as MG  # noqa: E402
import self_review as SR_  # noqa: E402
from audio_io import write_audio  # noqa: E402
from pipeline import process_story  # noqa: E402

SR = 44100


def narration_for(segs, dur, seed=0):
    rng = np.random.default_rng(seed)
    n = int(dur * SR)
    v = np.zeros(n, np.float32)
    for i, s in enumerate(segs):
        a, b = int(s["start"] * SR), min(n, int(s["end"] * SR))
        if b > a:
            x = SR_.speechlike((b - a + 1) / SR, 0.1 * rng.uniform(0.6, 1.4), i)
            v[a:b] = np.pad(x, (0, max(0, b - a - len(x))))[: b - a]
    return v


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--stories", default="workshop,undercover",
                    help="comma list from: workshop, undercover")
    ap.add_argument("--takes", type=int, default=None, help="takes per phrase/motif (default 3)")
    ap.add_argument("--run-seed", type=int, default=2026)
    args = ap.parse_args(argv)
    if args.takes:
        MG.MOTIF_TAKES = MG.PHRASE_TAKES = int(args.takes)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    all_stories = SR_.load_stories()
    pick = {"workshop": "workshop (85)", "undercover": "undercover (confession)"}
    t0 = time.time()
    MG.get_music_model()
    print(f"model ready in {time.time() - t0:.0f}s")
    for key in [s.strip() for s in args.stories.split(",") if s.strip()]:
        segs, dur = all_stories[pick[key]]
        sdir = out / key
        sdir.mkdir(parents=True, exist_ok=True)
        narr = sdir / f"{key}.wav"
        v = narration_for(segs, dur)
        write_audio(narr, np.stack([v, v], 1), SR)
        json.dump({"segments": segs}, open(sdir / f"{key}.json", "w"))
        t = time.time()
        process_story(1, str(narr), sdir, args.run_seed, log=print)
        print(f"{key}: done in {time.time() - t:.0f}s")
    SR_.main(["--renders", str(out), "--json", str(out / "self_review.json")])


if __name__ == "__main__":
    main()
