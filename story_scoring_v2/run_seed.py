# ============================================================
# PRODUCTION RUN-SEED HANDLING
#
# Problem: the validated V2 generation code used fixed literal seeds
# (motif = 20000 + story_index*1000, layer = 30000/31000/32000 + story_index*1000
# + phrase_index*100 + attempt) — perfect for a reproducible benchmark, wrong
# for production, where repeated runs would keep generating the same motif and
# phrases for the same story.
#
# Fix: every child seed (motif, each phrase/attempt) is derived from one
# `run_seed` for the whole job. A fresh job calls generate_run_seed() once and
# reuses it for every story/layer/phrase in that job, so:
#   - two different jobs (two different run_seeds) -> different music, even
#     for the same story file, mood and index.
#   - the SAME run_seed reproduces the SAME motif/phrases/seeds exactly
#     (useful for support, debugging, and re-running the 71/72/86 benchmark).
#
# This module has no dependency on the rest of the engine, so it is trivial to
# unit test without any audio/model code.
# ============================================================

import hashlib
import secrets

# Seeds are kept in this range for compatibility with model/backends that
# expect a 32-bit non-negative seed (0 is avoided since some samplers treat
# seed 0 as "unset").
SEED_MIN = 1
SEED_MAX = 2_147_483_646  # 2**31 - 2


def generate_run_seed():
    """A fresh, unpredictable seed for one production scoring job.

    Call this ONCE per job (e.g. once per `pipeline.main()` invocation, or
    once per ShortMakerPro scoring request) and pass the same value to every
    story/layer/phrase generated within that job.
    """
    return SEED_MIN + secrets.randbelow(SEED_MAX - SEED_MIN + 1)


def derive_seed(run_seed, *parts):
    """Deterministically derives a child seed from `run_seed` and `parts`.

    Same (run_seed, parts) -> same seed, always (needed for within-run
    reproducibility and for tests). Different run_seed -> a different,
    effectively independent seed even for identical parts (needed so a new
    production job doesn't regenerate the same motif/phrases).

    `parts` should fully identify what is being generated, e.g.
    derive_seed(run_seed, "motif", story_index) or
    derive_seed(run_seed, "layer", layer_name, story_index, phrase_index, attempt).
    """
    if run_seed is None:
        raise ValueError("run_seed is required (call generate_run_seed() for a fresh production job)")
    payload = "|".join(["run_seed", str(int(run_seed))] + [str(p) for p in parts])
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, "big")
    return SEED_MIN + (value % (SEED_MAX - SEED_MIN + 1))
