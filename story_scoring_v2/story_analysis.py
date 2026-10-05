# ============================================================
# STORY ANALYSIS — discovery, transcripts, mood, 7 story boundaries
# (logic unchanged from Generalization V1/V2), plus an independent turn
# estimate used to cross-check the cue sheet's twist
# ============================================================

import glob
import json
import math
import os
import re
from pathlib import Path

import numpy as np


SUPPORTED_AUDIO = (".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac")
EXCLUDE_NAMES = {"85_extracted.wav"}
EXCLUDE_STORY_IDS = {"85"}

GENERIC_BOUNDARY_FRACTIONS = np.array(
    [0.00, 0.07, 0.26, 0.40, 0.60, 0.69, 0.88, 1.00], dtype=np.float64
)


# ============================================================
# INPUT DISCOVERY
# ============================================================

def normalized_story_id(path):
    stem = Path(path).stem.strip().lower()
    for suffix in ("_extracted", "_audio", "_narration"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return stem


def discover_stories(input_roots, test_story_ids=(), max_stories=3, prefer_wav=True, log=print):
    all_audio = []
    excluded = {x.lower() for x in EXCLUDE_NAMES}
    for root in input_roots:
        for path in glob.glob(os.path.join(str(root), "**", "*"), recursive=True):
            if not os.path.isfile(path):
                continue
            if Path(path).suffix.lower() not in SUPPORTED_AUDIO:
                continue
            name = os.path.basename(path).lower()
            story_id = normalized_story_id(path)
            if name in excluded or story_id in EXCLUDE_STORY_IDS:
                continue
            all_audio.append(path)

    # De-duplicate alternate encodings; WAV preferred over lossy.
    unique = {}
    for path in sorted(dict.fromkeys(all_audio)):
        story_id = normalized_story_id(path)
        existing = unique.get(story_id)
        if existing is None:
            unique[story_id] = path
        elif (prefer_wav and Path(path).suffix.lower() == ".wav"
              and Path(existing).suffix.lower() != ".wav"):
            unique[story_id] = path

    all_audio = sorted(unique.values())
    log("\nEligible de-duplicated story files:")
    for path in all_audio:
        log(f"  {path}")

    if test_story_ids:
        selected = []
        for story_id in test_story_ids:
            matches = [p for p in all_audio if normalized_story_id(p) == str(story_id).lower()]
            if not matches:
                raise RuntimeError(f"Required test story '{story_id}' was not found in {list(input_roots)}.")
            selected.append(matches[0])
        return selected

    if len(all_audio) < max_stories:
        raise RuntimeError(
            f"Found only {len(all_audio)} eligible NEW story files. "
            f"This benchmark requires at least {max_stories}."
        )
    return all_audio[:max_stories]


# ============================================================
# TRANSCRIPTS
# ============================================================

def find_matching_json(audio_path):
    directory = os.path.dirname(audio_path)
    stem = Path(audio_path).stem
    for candidate in (stem + ".json", stem + "_transcript.json", stem + "_clean.json"):
        path = os.path.join(directory, candidate)
        if os.path.exists(path):
            return path
    return None


def load_transcript_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        if "segments" in data:
            return data["segments"]
        if "transcript" in data:
            return [{"start": 0.0, "end": 0.0, "text": data["transcript"]}]
    if isinstance(data, list):
        return data
    return []


def transcribe_with_whisper(path, log=print):
    """Returns a list of segments; [] (never None) when Whisper is missing."""
    try:
        import whisper
        import torch
    except Exception:
        log("  Whisper not available; using generic story boundaries.")
        return []
    log("  Whisper transcription...")
    model = whisper.load_model("base")
    result = model.transcribe(path, fp16=torch.cuda.is_available(), verbose=False)
    return result.get("segments", []) or []


# ============================================================
# KEYWORDS
# ============================================================

HOOK_WORDS = {"but", "then", "until", "suddenly", "days", "later", "one", "never",
              "nobody", "didn't", "did not"}
CONFLICT_WORDS = {"fight", "fought", "argument", "angry", "said", "told", "refused", "stop",
                  "leave", "left", "problem", "conflict", "betray", "betrayed"}
STAKE_WORDS = {"danger", "risk", "fire", "money", "debt", "loss", "dead", "death", "illegal",
               "fraud", "police", "court", "job", "business", "ruined", "threat"}
DISCOVERY_WORDS = {"found", "discover", "noticed", "realized", "saw", "opened", "inside",
                   "hidden", "learned", "discovered"}
REVEAL_WORDS = {"actually", "forged", "fake", "truth", "revealed", "confessed", "because",
                "secret", "wasn't", "was not", "never knew", "license", "signature"}
RESOLUTION_WORDS = {"finally", "after", "since", "ended", "closed", "returned", "left", "now",
                    "sleep", "resolution", "aftermath"}


def text_score(text, keywords):
    text = text.lower()
    return sum(1 for word in keywords if word in text)


def normalize_text(segments):
    clean = []
    for item in segments or []:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        start = float(item.get("start", 0.0))
        end = float(item.get("end", start))
        clean.append({"text": text, "start": start, "end": end})
    return clean


# ============================================================
# 7 STORY SECTIONS
# ============================================================

def infer_story_boundaries(segments, duration):
    segments = normalize_text(segments)
    if len(segments) < 7:
        return GENERIC_BOUNDARY_FRACTIONS * duration

    n = len(segments)
    hook_idx = min(n - 1, max(1, int(round(n * 0.06))))

    conflict = [text_score(s["text"], CONFLICT_WORDS) for s in segments]
    stake = [text_score(s["text"], STAKE_WORDS) for s in segments]
    discovery = [text_score(s["text"], DISCOVERY_WORDS) for s in segments]
    reveal = [text_score(s["text"], REVEAL_WORDS) for s in segments]
    resolution = [text_score(s["text"], RESOLUTION_WORDS) for s in segments]

    def best_in_range(scores, lo, hi, default_fraction):
        lo = max(0, int(round(n * lo)))
        hi = min(n - 1, int(round(n * hi)))
        if hi <= lo:
            return int(round(n * default_fraction))
        window = np.asarray(scores[lo:hi + 1])
        if np.max(window) <= 0:
            return int(round(n * default_fraction))
        return lo + int(np.argmax(window))

    raw_indices = [
        0,
        hook_idx,
        best_in_range(conflict, 0.08, 0.30, 0.26),
        best_in_range(stake, 0.25, 0.52, 0.40),
        best_in_range(discovery, 0.48, 0.72, 0.60),
        best_in_range(reveal, 0.62, 0.88, 0.69),
        best_in_range(resolution, 0.78, 0.96, 0.88),
        n - 1,
    ]

    fixed = [0]
    for index in raw_indices[1:-1]:
        minimum = fixed[-1] + 1
        maximum = n - (len(raw_indices) - len(fixed) - 1)
        fixed.append(max(minimum, min(index, maximum)))
    fixed.append(n - 1)

    boundaries = [segments[i]["start"] for i in fixed]
    boundaries[0] = 0.0
    boundaries[-1] = duration
    boundaries = np.asarray(boundaries, dtype=np.float64)

    minimum_gap = duration * 0.025
    for i in range(1, len(boundaries)):
        boundaries[i] = max(boundaries[i], boundaries[i - 1] + minimum_gap)
    boundaries[-1] = duration

    if not np.all(np.diff(boundaries) > 0):
        return GENERIC_BOUNDARY_FRACTIONS * duration
    return boundaries


# ============================================================
# MOOD
# ============================================================

MOOD_MAPPINGS = [
    (["betray", "cheat", "affair", "husband", "wife", "marriage", "relationship"], "relationship betrayal"),
    (["murder", "kill", "death", "dead", "police", "crime"], "dark crime mystery"),
    (["money", "debt", "business", "bank", "loan", "contract"], "financial pressure"),
    (["family", "brother", "sister", "mother", "father", "parent"], "family conflict"),
    (["job", "boss", "office", "company", "work", "fired"], "workplace conflict"),
    (["fire", "danger", "accident", "injury", "risk"], "physical danger"),
    (["lawyer", "court", "legal", "fraud", "forged", "signature", "police"], "legal thriller"),
]


def build_story_mood(segments):
    text = " ".join(str(s.get("text", "")) for s in segments or []).lower()
    mood = []
    for words, label in MOOD_MAPPINGS:
        if any(word in text for word in words):
            mood.append(label)
    if not mood:
        mood.append("psychological suspense")
    return ", ".join(list(dict.fromkeys(mood))[:3])


# ============================================================
# INDEPENDENT TURN ESTIMATE (for cross-checking the cue sheet)
#
# A second, coarse reading of where the story turns that shares nothing
# with cue_sheet's affect lexicon or intensity curve. It reads how the story
# is BUILT around the turn:
#   payoff     the line pays off details planted well before it (rare words
#              or numbers that appeared once, earlier: "the whole thing was a
#              test" after "what kind of loyalty test was that?")
#   aftermath  the next lines react or explain ("Nobody laughed", "Graham
#              stared", "Why would he forge it", "They wanted to see if")
#   place      narrated twists sit in the last third, before the aftermath
# It is deliberately a REGION, not a line: the score-weighted centre of the
# best candidates. On the three reference stories it lands within 13 % of
# the story length of the real twist; a turn placed a third of the story
# early (the 292 Kaggle run: 98 s for a 147 s reveal) falls outside it.
# ============================================================

_STRUCT_STOP = set("""a an the and or but so to of in on at for with from by as is was were be been being it its
this that these those i me my mine we our us you your he him his she her they them their there here then than
when what who whom which why how if not no yes do did does done have has had would could should will can just
like up down out over into about after before again all any each few more most other some such only own same too
very now one two said says say told tell asked ask went go goes got get gets came come back off still even ever
never time day days thing things man woman men women people way while where because i'd i'm i'll he's she's
it's that's didn't don't wasn't couldn't wouldn't know knew think thought every really right made make let look
looked first last next around through across away put took take give gave left long little much many good great
""".split())
_STRUCT_TOKEN = re.compile(r"[a-z0-9][a-z0-9']*(?:[.:][0-9]+)?")
_AFTERMATH_CUES = ("nobody", "no one", "silence", "silent", "froze", "pale", "stared", "staring", "speechless",
                   "lost it", "scream", "gasp", "couldn't believe", "tell me", "shaking", "went white",
                   "blinked", "his face", "her face", "laughing", "let go",
                   "because", "why ", "wanted to see", "the reason", "that's why", "which meant", "it meant",
                   "the whole", "all along", "way of")


def _struct_words(text):
    out = set()
    for tok in _STRUCT_TOKEN.findall(text.lower()):
        if tok in _STRUCT_STOP or not (tok[0].isdigit() or len(tok) >= 4):
            continue
        tok = tok.strip("'")
        if tok.endswith("'s"):
            tok = tok[:-2]
        out.add(tok if tok[0].isdigit() else tok[:6])
    return out


def structural_turn_estimate(segments, duration):
    """{"time": centre of the turn region (s), "candidates": best lines (s)},
    or None when the transcript is too short to read."""
    segs = normalize_text(segments)
    n = len(segs)
    if n < 7 or duration <= 0:
        return None
    words = [_struct_words(s["text"]) for s in segs]
    df = {}
    for ws in words:
        for w in ws:
            df[w] = df.get(w, 0) + 1
    pos = (np.arange(n) + 0.5) / n
    score = np.full(n, -np.inf)
    for i in range(n):
        if not 0.4 <= pos[i] <= 0.93:
            continue
        planted = [w for w in words[i] if df[w] <= 3 and any(
            w in words[j] and segs[j]["start"] <= segs[i]["start"] - 0.1 * duration for j in range(i))]
        after = " ".join(s["text"].lower() for s in segs[i + 1:i + 4])
        reacts = sum(after.count(c) for c in _AFTERMATH_CUES)
        place = math.exp(-0.5 * ((pos[i] - 0.78) / 0.12) ** 2)
        score[i] = 0.3 * min(3, len(planted)) + 0.3 * min(3, reacts) + place
    ok = np.isfinite(score)
    if not ok.any():
        return None
    weight = np.where(ok, np.exp((np.where(ok, score, 0.0) - score[ok].max()) / 0.15), 0.0)
    starts = np.array([s["start"] for s in segs])
    order = [int(i) for i in np.argsort(-np.where(ok, score, -1e9))[:3]]
    return {"time": float(np.sum(weight * starts) / weight.sum()),
            "candidates": [float(starts[i]) for i in order]}
