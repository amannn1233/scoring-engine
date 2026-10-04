# FINAL ShortMakerPro Integration Report — Story Scoring Lab V2

This is the authoritative report for the real-codebase integration pass. It supersedes the
"item F" notes in `story_scoring_v2/README.md` and
`story_generalization_v2_integration_validation/INTEGRATION_VALIDATION_REPORT.md`, which were
written before ShortMakerPro's actual source and the actual Kaggle music bridge had been
inspected (access was declined in an earlier session; it was granted and used in this pass).

## 0. What was inspected (real source, not assumed)

With `Desktop/ShortMakerPro` and `Desktop/story_scoring_kaggle_music_bundle` access granted,
the following REAL files were read in full before any change was made:

- `Desktop/ShortMakerPro/kaggle_bridge.py`, `kaggle_persistent_bridge.py` — ShortMakerPro's
  **voice** bridge (Chatterbox TTS over a persistent Kaggle WebSocket kernel). Contains
  `generate_from_kaggle(voice, text, seed=DEFAULT_SEED)` and `KagglePersistentKernel`.
- `Desktop/ShortMakerPro/production_audio.py` (89 KB) — the locked V2 + Stage 3 audio
  processing chain (voice normalization/ducking-adjacent code, unrelated to music scoring).
- `Desktop/ShortMakerPro/src/ui/main_window.py` (729 KB, ~16.7k lines) — the entire desktop
  UI, including `KaggleVoiceWorker(QThread)` (the existing background-worker pattern for a
  long-running Kaggle call: acquires a concurrency-limiting gate, calls
  `generate_from_kaggle(...)`, emits `finished_ok`/`failed` Qt signals) and
  `KaggleStartupWorker(QThread)` (kernel/model readiness check).
- `Desktop/ShortMakerPro/main.py`, `requirements.txt`, and the manual/diagnostic scripts at
  the repo root (`startup_test.py`, `test_generation.py`, `test_kaggle_remote.py`,
  `caption_alignment_test.py`, `benchmark_step9.py`, `shortmakerpro_quick_diagnostic.py`,
  `shortmakerpro_state_flow_diagnostic.py`).
- **Result: grepped `main_window.py`, `production_audio.py`, `kaggle_bridge.py`, and
  `kaggle_persistent_bridge.py` for `music`, `stable_audio`, `motif`, `scoring`, `duck` (case
  insensitive) — zero matches anywhere except one unrelated comment ("pre-duck" limiting) in
  `production_audio.py`.** ShortMakerPro has no existing music-scoring feature, code path, UI,
  or worker of any kind today. This integration adds one, rather than hooking into one.
- `Desktop/story_scoring_kaggle_music_bundle/story_scoring_kaggle_music_bundle/
  kaggle_music_bridge.py`, `story_scoring_lab_kaggle.py`, `stable_audio_3_music_kaggle.ipynb`,
  `README_KAGGLE_MUSIC_SETUP.md` — the REAL, already-existing Stable Audio 3 Kaggle bridge
  (separate from the voice bridge; see section 2).
- `Desktop/scoring/story_scoring_v2/` — the package completed and validated in the previous
  session (unchanged unless noted in section 4).

## 1. The real music bridge contract (not the assumed one)

An earlier session (without ShortMakerPro access) designed `KaggleBridgeMusicProvider`
against a **guessed** signature, `generate_music_from_kaggle(prompt, duration, seed,
init_audio=None, init_noise_level=None) -> str`, modelled on the voice bridge's
`generate_from_kaggle(voice, text, seed)`. That guess was wrong in two ways, found by reading
the real file:

1. **The music bridge is a separate module** — `kaggle_music_bridge.py` — not a new function
   added to `kaggle_bridge.py`. It lives in a third folder,
   `Desktop/story_scoring_kaggle_music_bundle/story_scoring_kaggle_music_bundle/`, paired with
   its own notebook, `stable_audio_3_music_kaggle.ipynb`, and its own persistent Kaggle
   kernel/session — the bundle's own README says explicitly: "Keep it separate from the
   existing ShortMakerPro voice kernel."
2. **Its real public entry point takes no reference/conditioning input:**
   ```python
   class KaggleMusicKernel:
       def __init__(self, ws_url: str): ...
       def connect(self) -> None: ...
       def ensure_model(self) -> None: ...
       def generate(self, prompt: str, duration: float, seed: int, output: Path) -> Path: ...
       def close(self) -> None: ...

   def load_url(path: Path = DEFAULT_URL_FILE) -> str: ...
   ```
   `generate()`'s remote-side code (`kaggle_music_bridge.build_generation_code`) always calls
   `music_model.generate(prompt=..., duration=..., steps=8, cfg_scale=1.0, seed=..., batch_size=1)`
   — **no `init_audio` or `init_noise_level` argument anywhere.** The bridge can do a fresh,
   unconditioned generation (right shape for the motif) but **cannot** condition on a reference
   clip (needed for every core/pressure/climax phrase, to stay locked to the motif).

This is a real, load-bearing finding, not a technicality: V2's phrase generation
(`music_generation.generate_layer_phrases()`) calls `generate_conditioned()` for all 9 phrases
per story (3 layers × 3 phrases), every time, by design (`LAYER_NOISE`, `LAYER_REFERENCE_OFFSET`
— unchanged). The real remote bridge cannot do that today. `KaggleBridgeMusicProvider` was
rewritten to the real contract and its `generate_conditioned()` now raises a clear,
actionable `RuntimeError` explaining exactly this, rather than silently dropping conditioning
(which would be an undocumented behaviour change to the validated V2 design). See section 6.

A second real, smaller bug was found the same way: `story_scoring_v2/music_generation.py`
imported `from stable_audio import StableAudioModel`, but the verified notebook
(`stable_audio_3_music_kaggle.ipynb`) installs and imports `stable_audio_3`
(`pip install git+https://github.com/Stability-AI/stable-audio-3.git`, `from stable_audio_3
import StableAudioModel`). Fixed — see section 4.

## 2. Exact SMP files created

- **`Desktop/ShortMakerPro/music_scoring_bridge.py`** (new, ~125 lines) — the real
  ShortMakerPro-side integration entry point, the music equivalent of `kaggle_bridge.py` for
  voice. Adds `Desktop/scoring/story_scoring_v2` to `sys.path` (resolved relative to its own
  location, overridable via `SHORTMAKERPRO_SCORING_LAB_DIR`) and exposes:
  - `generate_story_score(narration_path, out_dir, run_seed=None, provider=None, log=print)`
    → calls `smp_integration.run_scoring_job(...)`, returns
    `{"run_seed": int, "manifests": [...], "summary": [...]}`.
  - `make_kaggle_bridge_music_provider(bridge_dir=None, url_file=None, log=print)` — builds a
    `KaggleBridgeMusicProvider` pointed at
    `Desktop/story_scoring_kaggle_music_bundle/story_scoring_kaggle_music_bundle` by default
    (overridable via `SHORTMAKERPRO_MUSIC_BRIDGE_DIR`).

  This file imports nothing from, and is imported by nothing in, ShortMakerPro's existing
  modules — it is additive. No UI button or menu entry was added to `main_window.py`; see
  section 8 for why, and what the natural hookup point is (`KaggleVoiceWorker`'s pattern).

## 3. Exact SMP files modified

**None.** `main_window.py`, `production_audio.py`, `kaggle_bridge.py`,
`kaggle_persistent_bridge.py`, `speed_engine.py`, `tsm_backends.py`, `main.py`,
`requirements.txt`, and every other existing ShortMakerPro file are byte-for-byte unchanged.
Confirmed by never writing to any of them, and by `py_compile`-checking untouched local copies
of `kaggle_bridge.py`, `kaggle_persistent_bridge.py`, `production_audio.py`, `main.py`,
`benchmark_step9.py`, `src/ui/main_window.py`, and `src/ui/workers.py` (all still valid Python,
because none were touched).

## 4. Files modified inside `story_scoring_v2/` this pass

- **`music_generation.py`** — one-line import fix: `from stable_audio import StableAudioModel`
  → `from stable_audio_3 import StableAudioModel` (section 1). This is the only functional
  change; it is a correction to match the real, verified package, not a design change, and is
  exactly the kind of "actual integration bug" fix the brief allows. Comment updated to match.
- **`requirements.txt`** — corrected the `stable_audio` → `stable_audio_3` comment, with the
  real pip install command.
- **`smp_integration.py`** — `KaggleBridgeMusicProvider` rewritten against the real
  `kaggle_music_bridge.py` contract (section 1); `load_kaggle_bridge()` renamed
  `load_kaggle_music_bridge()` and now looks for `kaggle_music_bridge.py` in a `bridge_dir`
  (not `smp_root/kaggle_bridge.py`, which was the wrong file); added an ffmpeg-based FLAC
  decode fallback for machines without `soundfile` installed (see section 7).
- **`tests/test_production_integration.py`** — the 3 Kaggle-bridge tests rewritten to stub the
  REAL `KaggleMusicKernel` contract (`connect()`/`ensure_model()`/`generate()`/`close()`,
  `load_url()`, `DEFAULT_URL_FILE`) instead of the previous invented
  `generate_music_from_kaggle()` function; added a test that `generate_conditioned()` raises
  the documented error. Test count unchanged (17).
- **`dist/story_generalization_v2_exact_waveform.py`** — regenerated (mechanical) via
  `build_kaggle_script.py` to pick up the `stable_audio_3` fix.
- **`README.md`** — sections 2c/A/B/F updated to point at this report and the real entry point.

`scoring_engine.py`, `story_analysis.py`, `audio_io.py`, `replay_v2.py`, `pipeline.py`,
`run_seed.py`, `build_kaggle_script.py`, `tests/test_v2.py` — **unchanged** in this pass
(`pipeline.py`/`run_seed.py`/`build_kaggle_script.py` were already correct from the previous
pass and needed no further change here).

## 5. Run-seed flow (unchanged from the previous pass, re-verified below)

One unpredictable `run_seed` (`secrets.randbelow`) per production job
(`run_scoring_job()`/`generate_story_score()` call). Every Stable Audio seed used inside that
job — the motif, and each layer/phrase/attempt — is derived from it via
`blake2b(run_seed, ...identifying parts...)` in `run_seed.derive_seed()`. Same `run_seed` →
same derived seeds → bit-identical regeneration. Fresh jobs (the default — `run_seed` omitted)
get a new `run_seed` every call, so every new scoring job generates new music, while staying
deterministic within that one job. Nothing about this changed in this pass; section 6 verifies
it again end-to-end through the new real entry point.

## 6. Tests executed and results

### 6a. Existing ShortMakerPro tests

**Finding: ShortMakerPro has no offline-runnable automated test suite.** Every
`test_*.py`/`*_test.py`/diagnostic script at its root was inspected:

| File | What it needs |
|---|---|
| `startup_test.py` | live authenticated Kaggle WebSocket session |
| `test_generation.py` | live Kaggle voice generation (real network + credentials) |
| `test_kaggle_remote.py` | live Kaggle WebSocket session |
| `caption_alignment_test.py` | WhisperX models + real audio/ffmpeg |
| `benchmark_step9.py` | live persistent Kaggle voice kernel |
| `shortmakerpro_quick_diagnostic.py` / `shortmakerpro_state_flow_diagnostic.py` | live app/session state |

None import `unittest`/`pytest`, none are runnable offline in this sandbox (no live Kaggle
session, no network beyond the allowlisted proxy, no GPU), and none were faked. This is stated
plainly rather than skipped silently. What **was** run instead: `py_compile` on every existing
ShortMakerPro file this integration touches the neighbourhood of (section 3) to confirm they
are still syntactically valid — trivially true since none were edited, but verified rather
than assumed.

### 6b. The 26 offline scoring-lab tests

```
python tests/test_v2.py                      ->  9/9 passed  (unchanged)
python tests/test_production_integration.py  -> 17/17 passed (3 Kaggle-bridge tests rewritten
                                                                to the real contract; see 6d)
```
26/26 pass. (Compute-bound and slow in this sandbox — the full `test_production_integration.py`
run takes ~2.5 minutes here, not because anything hangs, but because `optimize_story()`'s
coordinate search is CPU-heavy without a GPU; confirmed not-hung with `faulthandler` stack
dumps mid-run before simply raising the timeout.)

### 6c. Mocked end-to-end job through the REAL `music_scoring_bridge.py` entry point

A folder layout mirroring the real Desktop was built (`ShortMakerPro/`, `scoring/
story_scoring_v2/`, `story_scoring_kaggle_music_bundle/story_scoring_kaggle_music_bundle/`,
each a sibling of the others, matching the real machine), with fake `torch` and `stable_audio_3`
modules (seed-deterministic synthetic audio, no GPU/network — the same mocking approach as the
previous pass, updated to the corrected `stable_audio_3` package name) on `PYTHONPATH`. Three
jobs were run by literally calling `music_scoring_bridge.generate_story_score(...)` from inside
the simulated `ShortMakerPro/` folder — the same function ShortMakerPro's own code would call:

| Job | `run_seed` | Result |
|---|---|---|
| A | omitted (fresh) | `265185709` |
| B | omitted (fresh) | `1291116904` — different from A |
| C | `265185709` (= A's) | reproduces A |

```
cmp jobA/story_01_narration_71/score_raw.wav jobC/.../score_raw.wav  -> IDENTICAL (same run_seed)
cmp jobA/story_01_narration_71/score_raw.wav jobB/.../score_raw.wav  -> DIFFERENT (different run_seed)
```
Job A's manifest: `alignment.optimizer_vs_final_render = {worst_db_diff: 0.0, aligned: true}`,
`optimizer.objective == final_metrics.objective` exactly (`1.190395026470733` both), `ducking =
{max_duck_db: 5.0, attack_seconds: 0.08, release_seconds: 1.0}` — unchanged from the validated
design.

### 6d. Real Kaggle bridge call signature and error handling

- `smp_integration.load_kaggle_music_bridge()` successfully `importlib`-loads the **real,
  unmodified** `kaggle_music_bridge.py` file (with only a fake `websocket` module standing in
  for the `websocket-client` package, so the import itself succeeds offline — no protocol
  behaviour is faked).
- `music_scoring_bridge.make_kaggle_bridge_music_provider()` was called with no
  `kaggle_music_session_url.txt` present. It correctly propagated the **real** bridge's own
  real error: `FileNotFoundError: Missing .../kaggle_music_session_url.txt. Paste the exact
  authenticated Kaggle Jupyter WebSocket URL into it.` — confirming the loading, URL
  resolution, and error-propagation path all work against the actual file, stopping exactly at
  the point a live network connection would be required (see section 9 — that connection
  itself was not, and could not be, tested here).
- `KaggleBridgeMusicProvider.generate_fresh()` was tested against a stub that reproduces the
  real contract exactly (`connect()`/`ensure_model()`/`generate()` writing real FLAC bytes via
  `ffmpeg`, `close()`) — passes, including the ffmpeg-fallback FLAC decode path (section 7).
- `KaggleBridgeMusicProvider.generate_conditioned()` raises the documented `RuntimeError`
  explaining the real bridge's lack of `init_audio` support — verified by test.

### 6e. Benchmark mode / `score_raw` measurement agreement

Re-ran `replay_v2.py` against the same V1 phrases used in the original 71/72/86 benchmark
after every change in this pass:

| Story | Objective | Match | Aligned | Worst diff |
|---|---:|:---:|:---:|---:|
| 71_extracted.wav | 0.9546042002066979 | yes (byte-identical) | yes | 0.0 dB |
| 72_extracted.wav | 0.39577805701353574 | yes (byte-identical) | yes | 0.0 dB |
| 86_audio.mp3 | 0.8447514694002037 | yes (byte-identical) | yes | 0.0 dB |

`scoring_engine.py` was not touched by this pass, so this is expected — and reconfirmed rather
than assumed.

### 6f. Compile/import checks

```
python -m py_compile story_scoring_v2/*.py story_scoring_v2/tests/*.py   -> OK (all files)
python -m py_compile Desktop/ShortMakerPro/music_scoring_bridge.py       -> OK
```
Every `.py` file created or modified in this pass compiles cleanly.

## 7. A real dependency gap found and closed

Decoding the real bridge's output needs an audio library that understands FLAC.
`audio_io.read_audio()` only has a WAV fallback when `soundfile` isn't installed — and
**neither ShortMakerPro's `requirements.txt` nor the music bundle's
`requirements_story_scoring_kaggle.txt` lists `soundfile`.** Without it, `KaggleBridgeMusicProvider`
would fail outright on a machine that doesn't happen to have `soundfile` installed. Since
`ffmpeg` is already a hard ShortMakerPro dependency (`kaggle_bridge.py` imports the `ffmpeg`
package; `requirements.txt` pins `ffmpeg-python`; `main.py` refuses to start without it),
`smp_integration._load_bridge_result()` now falls back to an `ffmpeg`-based FLAC→WAV decode
when `soundfile` isn't available, instead of requiring a new dependency. Verified in 6d.

## 8. Scope decision: no UI wiring added

`main_window.py` (16.7k lines, PySide6) has an established pattern for a long-running Kaggle
call — `KaggleVoiceWorker(QThread)` (background thread, a concurrency gate, `finished_ok`/
`failed` Qt signals) — and a future "Add Music Score" feature would naturally follow it: a new
`MusicScoreWorker(QThread)` calling `music_scoring_bridge.generate_story_score(...)` in
`run()`. That worker, plus a menu/button to trigger it, was **not** added in this pass. This
was a deliberate scope decision, not an oversight: the brief asked for "the smallest surgical
integration possible" and "do not modify unrelated SMP functionality," this environment cannot
render or interact with a Qt GUI to verify such a change, and `main_window.py` is a single
728,999-byte file with no separation between the music feature and everything else in it —
editing it blind, untestable here, is a real risk to "existing ShortMakerPro behaviour outside
the scoring/music feature," which the brief explicitly protects. `music_scoring_bridge.py` is
written so that adding that worker later is a small, mechanical step (one new `QThread`
subclass following the existing `KaggleVoiceWorker` pattern, calling one function).

## 9. What was actually tested vs. mocked — summary

**Actually tested, against real files, offline:**
- The real `kaggle_music_bridge.py` module loads and its `load_url()`/`DEFAULT_URL_FILE`
  behave exactly as written (real missing-file error surfaced end-to-end through our code).
- `KaggleBridgeMusicProvider`'s plumbing (connect → ensure_model → generate → decode) against
  a stub that reproduces the real `KaggleMusicKernel.generate()` contract exactly, including
  writing genuine FLAC bytes and decoding them via the new ffmpeg fallback.
- The full scoring pipeline (motif, 9 phrases, two-stage optimizer, render, duck, manifest)
  through the real `music_scoring_bridge.generate_story_score()` entry point, in a folder
  layout mirroring the real Desktop, with mocked Stable Audio.
- Run-seed freshness/reproducibility, alignment, and ducking, all through that same real entry
  point (section 6c) — not just through `smp_integration.run_scoring_job()` directly.
- The 71/72/86 benchmark, unaffected (section 6e).
- Every changed file's compilation (section 6f).

**Mocked (necessarily — no GPU/network/live Kaggle session in this environment):**
- Actual Stable Audio 3 generation (`torch`/`stable_audio_3` are fake, seed-deterministic
  synthetic audio, exactly as in the previous validation pass).
- An actual Kaggle Jupyter WebSocket connection (`websocket-client` itself is faked; no real
  `wss://` handshake, no real remote kernel, no real generation over that channel).

**Not tested at all (out of scope for this environment):**
- ShortMakerPro's desktop GUI (no Qt runtime/display here).
- The existing voice bridge (`kaggle_bridge.py`) end to end — not touched, not exercised; its
  behaviour is asserted unchanged only by never having modified it.

## 10. Is this ready to integrate?

- **The scoring engine, run-seed/provider mechanics, and the new real SMP entry point
  (`music_scoring_bridge.generate_story_score`) are validated and ready**, using the default
  `StableAudioDirectProvider` — i.e. Stable Audio loaded in the same Kaggle kernel that runs
  the scoring job (paste `dist/story_generalization_v2_exact_waveform.py` into
  `stable_audio_3_music_kaggle.ipynb`'s kernel after "STABLE AUDIO 3 SMALL MUSIC READY", or
  call `generate_story_score()` from code already running inside that kernel).
- **`KaggleBridgeMusicProvider` (routing through ShortMakerPro's separate WebSocket session
  from the desktop process, the same way voice already works) is real and tested for the
  motif, but cannot run a full scoring job** — every layer phrase needs conditioned
  generation, which the deployed `kaggle_music_bridge.py` does not support (section 1). Two
  ways forward, both requiring genuinely new work, not more inspection: (a) extend
  `kaggle_music_bridge.build_generation_code()` to accept `init_audio`/`init_noise_level` and
  re-verify against a live kernel, or (b) run the scoring job in-process inside the Kaggle
  kernel itself (already fully supported) rather than driving it from the desktop over the
  bridge.
- **No UI hookup in ShortMakerPro yet** (section 8) — the entry point is ready to be called by
  one, but nothing calls it today.
- **Nothing here has touched a real GPU or a real Kaggle session** — see section 9.
