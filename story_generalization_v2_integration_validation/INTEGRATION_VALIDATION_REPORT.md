# V2 Production Integration — Validation Report

All checks below were run **offline**, in this container, with a mocked Stable Audio
model (deterministic function of `seed`, no GPU/network). They validate the run-seed
and provider-injection code paths and reconfirm the scoring engine is untouched.
**No real Stable Audio / Kaggle generation was run** — see item F in the final report
for what still needs a real GPU run.

## 1. Unit / integration tests (offline)

```
python tests/test_v2.py                        9 tests  -> 9 passed
python tests/test_production_integration.py    17 tests -> 17 passed
```
26/26 pass. `test_v2.py` is the original engine-exactness suite (untouched).
`test_production_integration.py` is new and covers: run-seed derivation, the
`MusicProvider` plug point, same/different-run-seed reproducibility at both the
`generate_motif()` level and the full `process_story()` level, `run_scoring_job()`,
and the `KaggleBridgeMusicProvider` contract (missing-file / missing-function /
working cases).

## 2. 71/72/86 alignment benchmark — reconfirmed unaffected

`scoring_engine.py` (the canonical `render_score()`/`measure_score()`) was not
touched by this integration work. Re-running `replay_v2.py` against the same V1
phrases produces **byte-identical** objectives to the previously validated run:

| Story | Previously validated objective | Re-run objective | Match | Aligned |
|---|---:|---:|:---:|:---:|
| 71_extracted.wav | 0.9546042002066979 | 0.9546042002066979 | yes | yes |
| 72_extracted.wav | 0.39577805701353574 | 0.39577805701353574 | yes | yes |
| 86_audio.mp3 | 0.8447514694002037 | 0.8447514694002037 | yes | yes |

Worst optimizer-vs-final diff: `0.0` dB for all three (`replay_recheck_71_72_86_summary.json`).

## 3. End-to-end mocked production run (`dist/story_generalization_v2_exact_waveform.py`)

Ran the actual single-file Kaggle build three times against two synthetic narration
files (`71_extracted.wav`, `72_extracted.wav`, mocked Stable Audio):

| Job | `--run-seed` | Result |
|---|---|---|
| A | omitted (fresh) | `run_seed=838368497` |
| B | omitted (fresh) | `run_seed=421381610` — **different from A** |
| C | `838368497` (= A's seed) | reproduces A |

**Reproducibility (A vs C, same run_seed):**
```
cmp jobA/story_01_71_extracted/score_raw.wav jobC/story_01_71_extracted/score_raw.wav -> IDENTICAL
cmp jobA/story_02_72_extracted/score_raw.wav jobC/story_02_72_extracted/score_raw.wav -> IDENTICAL
```
Motif seed, phrase seeds, optimizer orders/phases and every manifest metric matched exactly.

**Freshness (A vs B, different run_seed):**
```
cmp jobA/story_01_71_extracted/score_raw.wav jobB/story_01_71_extracted/score_raw.wav -> DIFFERENT
```

**Within one job, different stories still get different material** (both share job A's
`run_seed=838368497`):
```
story_01_71_extracted motif seed: 1227941148
story_02_72_extracted motif seed: 912067277
```

**Alignment held with freshly generated (mocked) material, not just replayed V1 phrases:**
```
71_extracted.wav: optimizer 0.6349 == final 0.6349, worst diff 0.000000 dB, ALIGNED
72_extracted.wav: optimizer 1.4614 == final 1.4614, worst diff 0.000000 dB, ALIGNED
```

**Ducking unchanged** (`manifest["ducking"]`): `max_duck_db=5.0`, `attack_seconds=0.08`,
`release_seconds=1.0` in every job — identical to the original V1/V2 settings.

**Model loaded once per job** (`"Loading Stable Audio 3 Small Music..."` appears exactly
once in each job log, confirming the existing model-reuse pattern is preserved).

## 4. Files in this folder

- `jobA.log` / `jobB.log` / `jobC.log` — full console logs of the three mocked runs.
- `summary_jobA_fresh.json`, `summary_jobB_fresh.json`, `summary_jobC_reproduced.json` —
  `generalization_summary.json` from each job.
- `manifest_jobA_story01.json`, `manifest_jobA_story02.json` — full per-story manifests
  from job A (includes `run_seed`, per-phrase derived seeds, optimizer/final/file
  alignment, ducking stats).
- `replay_recheck_71_72_86_summary.json` — the re-run of the original validated
  benchmark, confirming no drift.

No audio from the mocked runs is included — mocked tones have no listening value;
what matters here is the metrics and the bit-for-bit file comparisons above.
