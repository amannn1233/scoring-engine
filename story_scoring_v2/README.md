# Story Scoring Lab — Generalization V2 (exact-waveform optimizer + production integration)

Status: **integrated into the real ShortMakerPro codebase.** The scoring architecture
(validated on 71/72/86) is unchanged. A later pass (see
`../story_scoring_v2_integration_validation/FINAL_SMP_INTEGRATION_REPORT.md` — the
authoritative document for the real integration) inspected ShortMakerPro's actual source,
the actual `kaggle_music_bridge.py`, and added one new file,
`Desktop/ShortMakerPro/music_scoring_bridge.py`, as the real entry point. This file
(2a/2b/2c below) documents the scoring-lab side of that integration; read the final report
for what changed on the ShortMakerPro side and what was actually tested.

## Part 1 — the validated V2 scoring architecture (unchanged)

The V1 optimizer reported metrics the final WAV did not have (story 72: "0 boundary
penalty" but a 5.9 dB jump; story 86: "+1.34 dB peak gap" but −0.10 dB). V2 fixed this with
**one** renderer (`render_score()`) and **one** measurer (`measure_score()`) in
`scoring_engine.py`; both optimizer stages evaluate every candidate on the exact 44.1 kHz
stereo mix built by that renderer, so the optimizer's numbers *are* the final waveform's
numbers.

Unchanged on purpose (per the original handover, and again per this integration pass):
Stable Audio prompts, motif design, phrase/layer design, quality gates, 7-section targets,
gain automation, objective weights, tolerances, and ducking (5 dB / 80 ms / 1.0 s).

**Validated result** (offline replay on the real V1 phrases for 71, 72, 86 — see
`../story_generalization_v2_replay/REPLAY_REPORT.md`): optimizer objective == in-memory
final objective == reloaded-WAV objective, worst diff `0.000000` dB (PCM-rounding only,
≤ 0.0003 dB against the 16-bit file). This benchmark used fixed seeds intentionally, for
reproducibility, and still does — the seed change below only affects fresh production runs,
not this benchmark. Re-run in this session to reconfirm zero drift after the integration
changes: see `../story_generalization_v2_integration_validation/INTEGRATION_VALIDATION_REPORT.md`
section 2 (byte-identical objectives, still 0.0 dB worst diff).

## Part 2 — what this pass added

### 2a. Fresh run seeds for production (`run_seed.py`)

The original V2 draft used fixed literal seeds (`motif = 20000 + story_index*1000`,
`layer = 30000/31000/32000 + story_index*1000 + phrase_index*100 + attempt`) — correct for a
reproducible benchmark, wrong for production: every new run would regenerate the exact same
motif and phrases for the same story.

`run_seed.py` adds two functions:

- `generate_run_seed()` — a fresh, unpredictable seed (`secrets.randbelow`), called **once
  per production job** (one job = one call to `pipeline.main()` / `smp_integration.run_scoring_job()`,
  which may cover several stories).
- `derive_seed(run_seed, *parts)` — a stable hash (`blake2b`) of `run_seed` and identifying
  parts (e.g. `"motif", story_index` or `"layer", layer_name, story_index, phrase_index, attempt`).
  Same inputs → same seed, always; different `run_seed` → an independent seed even for
  identical `parts`.

`music_generation.py`'s `generate_motif()` and `generate_layer_phrases()` now take a
`run_seed` argument and derive every seed through `derive_seed()` instead of a literal.
Nothing else about generation changed — same prompts, same noise levels, same attempt/gate
logic; only *which seed* each call uses.

Net effect: two calls to `run_scoring_job()` (or two runs of `pipeline.main()`) produce
different music, even for the same story file and story index. The same explicit
`run_seed` reproduces a previous job's motif and phrases bit-for-bit (every manifest now
records `run_seed` and each generated item's derived `seed` for exactly this purpose).

### 2b. `MusicProvider` — a plug point for however Stable Audio actually runs (`music_generation.py`)

`music_generation.py` now defines a two-method interface:

```python
class MusicProvider:
    def generate_fresh(self, prompt, duration, seed): ...
    def generate_conditioned(self, prompt, duration, reference, noise, seed): ...
```

`StableAudioDirectProvider` — the **default**, installed automatically — is exactly what
V1/V2 always did (load Stable Audio in-process, call `music_model.generate(...)`,
reusing an already-loaded `music_model` if the Kaggle notebook has one). Behaviour is
byte-for-byte the same as before this change when nothing overrides the provider.

`set_provider(provider)` / `get_provider()` let a caller install a different transport —
this is the ONLY change needed to point generation at a different Stable Audio session
(a separate Kaggle bridge, a local GPU service, whatever ShortMakerPro already uses) without
touching prompts, gates, the optimizer, or the renderer.

### 2c. ShortMakerPro adapter (`smp_integration.py`) and the real entry point

The one function a ShortMakerPro developer needs to call:

```python
from smp_integration import run_scoring_job
result = run_scoring_job(narration_paths=[...], out_root=...)
# result = {"run_seed": int, "manifests": [...], "summary": [...]}
```

But the actual ShortMakerPro-side entry point ShortMakerPro's own code should import is
`Desktop/ShortMakerPro/music_scoring_bridge.py`'s `generate_story_score(narration_path,
out_dir, run_seed=None, provider=None, log=print)` — a thin, new file that adds this
package to `sys.path` and calls `run_scoring_job()`, the music equivalent of
`kaggle_bridge.generate_from_kaggle()` for voice. See
`../story_scoring_v2_integration_validation/FINAL_SMP_INTEGRATION_REPORT.md` for how that
file was verified.

- Omit `run_seed` for a fresh production job (the default — recommended for real use).
- Pass a previous job's `run_seed` to reproduce it exactly (support/debugging, or re-running
  a benchmark).
- Omit `provider` when Stable Audio is loaded in the same process (the validated Kaggle
  notebook case) — `StableAudioDirectProvider` is used automatically, and is the only
  provider that supports the validated conditioned (motif-locked) phrase generation today.
- Pass `provider=KaggleBridgeMusicProvider(bridge_dir)` to route the **motif** through
  ShortMakerPro's REAL, already-existing remote Stable Audio 3 Kaggle bridge —
  `kaggle_music_bridge.KaggleMusicKernel`, verified against the actual file in
  `Desktop/story_scoring_kaggle_music_bundle/story_scoring_kaggle_music_bundle/` (paired
  with `stable_audio_3_music_kaggle.ipynb`). This is a **separate** bridge/kernel from
  ShortMakerPro's voice bridge (`kaggle_bridge.py`), by the bundle's own design. Its real
  `generate(prompt, duration, seed, output) -> Path` has **no `init_audio`/
  `init_noise_level` parameter**, so `generate_conditioned()` — needed for every
  core/pressure/climax phrase — raises a clear, actionable `RuntimeError` rather than
  silently generating unconditioned (motif-drifting) phrases. See
  `KaggleBridgeMusicProvider`'s docstring in `smp_integration.py` and item F below.
- Or pass `provider=CallableMusicProvider(fresh=..., conditioned=...)` to wire up any other
  transport in one line without subclassing.

`run_scoring_job()` installs the given provider for the duration of the job and restores
whatever was active before, so it composes safely with other code using this module.

## Files

```
story_scoring_v2/
  scoring_engine.py       render_score / measure_score / optimize_story / ducking  — UNCHANGED
  story_analysis.py       input discovery, transcripts, mood, 7 story boundaries    — UNCHANGED
  audio_io.py             soundfile, with scipy fallback                           — UNCHANGED
  replay_v2.py            offline V2 re-score of existing phrases (no GPU)          — UNCHANGED
  run_seed.py             NEW — generate_run_seed() / derive_seed()
  music_generation.py     MODIFIED — MusicProvider + run_seed-derived seeds (prompts/gates unchanged)
  pipeline.py             MODIFIED — threads run_seed through, --run-seed CLI flag, records it in manifests
  build_kaggle_script.py  MODIFIED — includes run_seed.py in the single-file build
  smp_integration.py      NEW — ShortMakerPro-facing entry point (run_scoring_job, providers)
  dist/story_generalization_v2_exact_waveform.py   REGENERATED — paste this into Kaggle
  tests/test_v2.py                       9 tests  — UNCHANGED, still 9/9 passing
  tests/test_production_integration.py   NEW — 17 tests: run-seed, provider, integration
../story_generalization_v2_replay/                    UNCHANGED — the validated 71/72/86 benchmark artifacts
../story_generalization_v2_integration_validation/     validation report from the run-seed/provider pass
../story_scoring_v2_integration_validation/            NEW — real-ShortMakerPro-codebase integration report
../../ShortMakerPro/music_scoring_bridge.py            NEW — the real SMP-side entry point (music_scoring_bridge.generate_story_score)
```
(`../../ShortMakerPro/` because `Desktop/ShortMakerPro/` is a sibling of `Desktop/scoring/`,
not of `Desktop/scoring/story_scoring_v2/`.)

## Run

**Kaggle (production, fresh generation each run):** paste
`dist/story_generalization_v2_exact_waveform.py` into one cell and run it — behaves exactly
as before, except it now prints and records a fresh `run_seed` every run:
```
Run seed for this job: 1928374650 (fresh — record this to reproduce this job)
```
To reproduce a specific past run: `main(["--run-seed", "1928374650"])`, or edit
`DEFAULT_TEST_STORY_IDS`/pass `--stories` as before. Story 86 uses `86_extracted.wav`
automatically (WAV preferred over MP3, `85_*` excluded — `story_analysis.discover_stories()`,
unchanged).

**ShortMakerPro integration (real entry point, from ShortMakerPro's own code):**
```python
# from inside Desktop/ShortMakerPro/ (e.g. a QThread worker in src/ui/main_window.py)
from music_scoring_bridge import generate_story_score

result = generate_story_score(
    narration_path=finished_narration_wav,
    out_dir=job_output_dir,
    # run_seed=None -> fresh production job (recommended)
    # provider=None -> StableAudioDirectProvider (the only provider that supports
    #                  conditioned phrase generation today; see item F)
)
```

**Offline replay (any PC, no GPU) — reconfirms scoring-engine alignment:**
```
python replay_v2.py --source ../story_generalization_v1 --out ../story_generalization_v2_replay
```

**Tests:**
```
python tests/test_v2.py                      # 9  — scoring-engine exactness (unchanged)
python tests/test_production_integration.py  # 17 — run-seed, provider, integration (new)
```

**Rebuild the Kaggle file after editing any module:** `python build_kaggle_script.py`

## A — exact files created / modified / untouched

Full, current list (including the real ShortMakerPro-side file) is in
`../story_scoring_v2_integration_validation/FINAL_SMP_INTEGRATION_REPORT.md`. Summary for
this package:

**Created:**
- `story_scoring_v2/run_seed.py`
- `story_scoring_v2/smp_integration.py`
- `story_scoring_v2/tests/test_production_integration.py`
- `story_generalization_v2_integration_validation/` (validation report from the run-seed pass)
- `story_scoring_v2_integration_validation/` (validation report from the real-SMP-integration pass)
- `Desktop/ShortMakerPro/music_scoring_bridge.py` — the real SMP-side entry point (**outside**
  this package; see the final report)

**Modified:**
- `story_scoring_v2/music_generation.py` — seed derivation + `MusicProvider`; also fixed the
  Stable Audio import (`stable_audio` -> `stable_audio_3`, matching the real, verified
  `stable_audio_3_music_kaggle.ipynb` — a real integration bug, not a design change). Prompts,
  noise levels, gates, attempt logic byte-for-byte unchanged otherwise.
- `story_scoring_v2/pipeline.py` — `run_seed` threaded through `process_story()`, `--run-seed`
  CLI flag, `run_seed` + per-item `seed` recorded in manifests.
- `story_scoring_v2/build_kaggle_script.py` — added `run_seed.py` to the single-file build.
- `story_scoring_v2/dist/story_generalization_v2_exact_waveform.py` — regenerated from the
  modules above (mechanical; not hand-edited).
- `story_scoring_v2/requirements.txt` — corrected the `stable_audio` -> `stable_audio_3`
  comment to match the real package.
- `story_scoring_v2/README.md` — this file.

**Explicitly untouched (byte-identical to the validated V2 handoff):**
- `story_scoring_v2/scoring_engine.py` (the canonical renderer/measurer — reconfirmed via
  re-run after every change in this pass too; worst diff `0.0` dB on 71/72/86)
- `story_scoring_v2/story_analysis.py`
- `story_scoring_v2/audio_io.py`
- `story_scoring_v2/replay_v2.py`
- `story_scoring_v2/tests/test_v2.py`
- `../story_generalization_v1/` and `../story_generalization_v2_replay/` (existing benchmark
  artifacts — not deleted, not modified)
- Every existing ShortMakerPro file — `main_window.py`, `production_audio.py`,
  `kaggle_bridge.py`, `kaggle_persistent_bridge.py`, `speed_engine.py`, everything else. See
  the final report for the exact list inspected and confirmed unchanged.

## B — exact integration entry point

Real ShortMakerPro-side entry point (new file):
```python
# Desktop/ShortMakerPro/music_scoring_bridge.py
music_scoring_bridge.generate_story_score(narration_path, out_dir, run_seed=None, provider=None, log=print)
```
which calls, unchanged:
```python
smp_integration.run_scoring_job(narration_paths, out_root, run_seed=None, provider=None, log=print)
```
Both return `{"run_seed": int, "manifests": [...], "summary": [...]}`. See
`../story_scoring_v2_integration_validation/FINAL_SMP_INTEGRATION_REPORT.md` for the full
account of the real-codebase integration (files touched, real bridge contract, tests run).

## F — remaining issue that needs a real Kaggle/Stable Audio run

See `../story_scoring_v2_integration_validation/FINAL_SMP_INTEGRATION_REPORT.md` for the
full, current account (this section is kept only as a short pointer, since the earlier
version of this section — written before ShortMakerPro's real source was inspected — is now
superseded). In short:

1. **`KaggleBridgeMusicProvider` can only do the motif, not phrases**, against the REAL,
   verified `kaggle_music_bridge.py` — its live `generate()` has no `init_audio`/
   `init_noise_level` parameter, so conditioned generation isn't possible over that bridge
   today. `StableAudioDirectProvider` (Stable Audio in the same Kaggle kernel) is the only
   provider that can run a full scoring job right now.
2. **No real Stable Audio generation, and no real Kaggle WebSocket connection, has been run
   through this code** — only mocked offline validation (both here and in the final SMP
   report) and the unchanged 71/72/86 benchmark replay. A real Kaggle run of
   `dist/story_generalization_v2_exact_waveform.py`, and a real `KaggleBridgeMusicProvider`
   connection against a live `kaggle_music_session_url.txt`, are the two checks that
   genuinely need live infrastructure.
