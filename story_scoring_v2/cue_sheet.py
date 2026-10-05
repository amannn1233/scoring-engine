# ============================================================
# STORY CUE SHEET — read the story's vibe from its own words and turn it
# into a music brief: palette, tempo, key/mode, motif, per-section
# intensity arc and hit points. Free and local: pure Python + numpy.
#
# There is no list of moods to pick from. Every segment of the transcript is
# scored on continuous affect dimensions (valence, arousal, tension) and on
# open "colour" dimensions (darkness, warmth, wonder, grit, intimacy,
# urgency, melancholy, triumph, eeriness, playfulness, and setting cues).
# Those numbers are then composed into a brief:
#
#   tempo   <- arousal + urgency           (continuous, rounded to whole BPM)
#   mode    <- valence x tension x wonder  (aeolian, dorian, phrygian, ...)
#   key     <- story fingerprint + brightness
#   palette <- instruments scored against the story's colour vector, one per
#              orchestral role (lead / harmony / bass / pulse / texture / hits)
#   style   <- the strongest colour + setting descriptors, worded freshly
#   arc     <- smoothed per-segment intensity; climax, twist and resolution
#              located from the curve itself, not from fixed fractions
#
# Two stories with different words get different briefs; the same story
# always gets the same brief (deterministic, so runs are reproducible).
# ============================================================

import hashlib
import math
import re
from dataclasses import dataclass, field, asdict

import numpy as np

# <local-imports>
from story_analysis import GENERIC_BOUNDARY_FRACTIONS
# </local-imports>


# ============================================================
# LEXICON
#
# word stem -> (valence, arousal, tension) in [-1, 1], plus colour tags.
# Hand-built from common narration vocabulary (Reddit-style confessions,
# revenge / betrayal / workplace / crime / family stories, wholesome
# stories). Stems match by prefix, so "betray" covers betrayed/betrayal.
# ============================================================

_AFFECT = {
    # --- threat, fear, danger
    "afraid": (-0.7, 0.6, 0.8), "terrif": (-0.9, 0.9, 0.9), "scare": (-0.6, 0.7, 0.7),
    "fear": (-0.7, 0.6, 0.8), "panic": (-0.8, 0.95, 0.9), "danger": (-0.6, 0.7, 0.9),
    "threat": (-0.7, 0.7, 0.9), "risk": (-0.4, 0.5, 0.7), "gun": (-0.7, 0.9, 0.95),
    "knife": (-0.7, 0.8, 0.9), "blood": (-0.8, 0.8, 0.8), "kill": (-0.95, 0.9, 0.95),
    "murder": (-0.95, 0.85, 0.95), "dead": (-0.9, 0.5, 0.7), "death": (-0.9, 0.5, 0.7),
    "die": (-0.9, 0.6, 0.7), "fire": (-0.6, 0.85, 0.85), "explo": (-0.7, 0.95, 0.9),
    "crash": (-0.7, 0.9, 0.8), "scream": (-0.7, 0.95, 0.85), "attack": (-0.8, 0.9, 0.9),
    "beat": (-0.5, 0.8, 0.7), "punch": (-0.6, 0.9, 0.8), "slam": (-0.4, 0.85, 0.7),
    "shove": (-0.5, 0.8, 0.7), "pound": (-0.3, 0.8, 0.7), "trap": (-0.6, 0.7, 0.85),
    "hide": (-0.3, 0.5, 0.75), "hidden": (-0.2, 0.4, 0.7), "chase": (-0.4, 0.9, 0.8),
    "escape": (-0.1, 0.85, 0.8), "run": (0.0, 0.7, 0.4), "disaster": (-0.8, 0.8, 0.85),
    "collapse": (-0.7, 0.8, 0.8), "injur": (-0.7, 0.6, 0.7), "hospital": (-0.6, 0.4, 0.6),
    "police": (-0.3, 0.6, 0.7), "cop": (-0.2, 0.6, 0.6), "arrest": (-0.5, 0.7, 0.8),
    "undercover": (-0.1, 0.6, 0.85), "gang": (-0.5, 0.7, 0.8), "biker": (-0.2, 0.7, 0.6),
    "loyal": (0.3, 0.4, 0.5), "test": (-0.1, 0.5, 0.6), "cover": (-0.1, 0.4, 0.6),
    "secret": (-0.2, 0.4, 0.75), "lie": (-0.6, 0.5, 0.7), "lied": (-0.6, 0.5, 0.7),
    "fake": (-0.5, 0.4, 0.65), "forg": (-0.6, 0.5, 0.75), "fraud": (-0.7, 0.6, 0.8),
    "steal": (-0.7, 0.6, 0.7), "stole": (-0.7, 0.6, 0.7), "illegal": (-0.6, 0.5, 0.75),
    "court": (-0.3, 0.5, 0.7), "lawyer": (-0.2, 0.4, 0.6), "sue": (-0.5, 0.6, 0.7),
    "inspector": (-0.3, 0.4, 0.6), "marshal": (-0.3, 0.5, 0.65), "shut": (-0.5, 0.6, 0.6),
    "deadline": (-0.3, 0.6, 0.7), "pending": (-0.2, 0.3, 0.55), "warn": (-0.4, 0.6, 0.7),
    # --- conflict, anger, betrayal
    "angry": (-0.7, 0.85, 0.7), "anger": (-0.7, 0.85, 0.7), "furious": (-0.8, 0.95, 0.8),
    "rage": (-0.85, 0.95, 0.8), "yell": (-0.6, 0.9, 0.7), "shout": (-0.5, 0.9, 0.65),
    "argu": (-0.5, 0.7, 0.6), "fight": (-0.6, 0.85, 0.7), "fought": (-0.6, 0.85, 0.7),
    "betray": (-0.9, 0.6, 0.8), "cheat": (-0.85, 0.6, 0.75), "affair": (-0.7, 0.5, 0.7),
    "revenge": (-0.3, 0.7, 0.75), "petty": (-0.2, 0.5, 0.4), "humiliat": (-0.8, 0.6, 0.6),
    "embarrass": (-0.5, 0.5, 0.4), "incompetent": (-0.5, 0.4, 0.4), "blame": (-0.6, 0.6, 0.6),
    "refus": (-0.4, 0.5, 0.5), "demand": (-0.4, 0.6, 0.6), "threaten": (-0.8, 0.8, 0.9),
    "fired": (-0.7, 0.6, 0.6), "quit": (-0.3, 0.5, 0.4), "divorce": (-0.7, 0.5, 0.6),
    "entitled": (-0.6, 0.5, 0.4), "karen": (-0.4, 0.6, 0.4), "toxic": (-0.7, 0.5, 0.5),
    "manipulat": (-0.8, 0.5, 0.7), "abuse": (-0.9, 0.6, 0.75), "control": (-0.3, 0.4, 0.5),
    # --- sadness, loss
    "sad": (-0.7, 0.2, 0.2), "cry": (-0.7, 0.5, 0.4), "cried": (-0.7, 0.5, 0.4),
    "tears": (-0.6, 0.4, 0.3), "grief": (-0.85, 0.3, 0.3), "funeral": (-0.8, 0.2, 0.3),
    "lonely": (-0.7, 0.1, 0.2), "alone": (-0.5, 0.1, 0.3), "miss": (-0.4, 0.2, 0.2),
    "lost": (-0.6, 0.3, 0.4), "loss": (-0.7, 0.3, 0.4), "broke": (-0.6, 0.4, 0.4),
    "regret": (-0.6, 0.3, 0.3), "sorry": (-0.3, 0.3, 0.2), "guilt": (-0.6, 0.4, 0.5),
    "confess": (-0.2, 0.5, 0.6), "ashamed": (-0.6, 0.3, 0.3), "hurt": (-0.7, 0.5, 0.4),
    "debt": (-0.6, 0.4, 0.6), "loan": (-0.2, 0.3, 0.4), "poor": (-0.5, 0.2, 0.3),
    "sick": (-0.6, 0.3, 0.4), "cancer": (-0.85, 0.4, 0.6), "silence": (-0.2, 0.0, 0.4),
    "quiet": (0.0, -0.4, 0.2), "never knew": (-0.2, 0.4, 0.6),
    # --- discovery, reveal, twist
    "found": (0.0, 0.5, 0.5), "discover": (0.0, 0.6, 0.6), "realiz": (0.0, 0.6, 0.6),
    "notic": (0.0, 0.4, 0.4), "truth": (0.0, 0.6, 0.6), "reveal": (0.0, 0.7, 0.7),
    "actually": (0.0, 0.4, 0.4), "suddenly": (-0.1, 0.8, 0.6), "turns out": (0.0, 0.6, 0.6),
    "shock": (-0.3, 0.9, 0.7), "stunn": (-0.1, 0.8, 0.6), "froze": (-0.3, 0.7, 0.8),
    "pale": (-0.4, 0.5, 0.6), "gasp": (-0.2, 0.8, 0.6), "silent": (-0.2, 0.1, 0.5),
    # --- warmth, joy, relief, triumph
    "love": (0.8, 0.4, -0.2), "hug": (0.7, 0.3, -0.3), "kiss": (0.7, 0.5, 0.0),
    "laugh": (0.7, 0.6, -0.3), "smile": (0.7, 0.3, -0.3), "happy": (0.8, 0.5, -0.3),
    "joy": (0.85, 0.6, -0.3), "proud": (0.7, 0.5, -0.1), "thank": (0.6, 0.3, -0.3),
    "grateful": (0.7, 0.3, -0.3), "kind": (0.6, 0.2, -0.3), "gentle": (0.5, -0.2, -0.4),
    "safe": (0.6, -0.2, -0.5), "relief": (0.6, 0.2, -0.6), "calm": (0.4, -0.5, -0.5),
    "peace": (0.6, -0.5, -0.6), "home": (0.5, -0.1, -0.2), "family": (0.4, 0.1, 0.1),
    "wedding": (0.6, 0.5, 0.1), "baby": (0.6, 0.3, 0.0), "friend": (0.5, 0.2, -0.1),
    "win": (0.8, 0.7, 0.0), "won": (0.8, 0.7, 0.0), "victory": (0.85, 0.8, 0.0),
    "justice": (0.6, 0.6, 0.2), "karma": (0.4, 0.5, 0.2), "promot": (0.7, 0.5, -0.1),
    "finally": (0.4, 0.3, -0.3), "free": (0.6, 0.4, -0.3), "hope": (0.6, 0.3, 0.0),
    "beautiful": (0.8, 0.3, -0.3), "amazing": (0.8, 0.6, -0.2), "magic": (0.7, 0.4, 0.0),
    "wonder": (0.6, 0.3, 0.0), "dream": (0.5, 0.1, 0.0), "star": (0.5, 0.2, 0.0),
    "funny": (0.7, 0.6, -0.4), "joke": (0.6, 0.5, -0.4), "classy": (0.2, 0.3, 0.1),
    "flirt": (0.3, 0.6, 0.3), "date": (0.4, 0.4, 0.2), "drunk": (-0.1, 0.6, 0.3),
    "bar": (0.0, 0.5, 0.3), "party": (0.5, 0.8, 0.0),
    # --- work / money / mundane (low arousal, mild tension)
    "work": (0.0, 0.2, 0.2), "job": (0.0, 0.2, 0.2), "boss": (-0.2, 0.3, 0.4),
    "office": (-0.1, 0.0, 0.2), "client": (0.0, 0.2, 0.3), "business": (0.0, 0.2, 0.3),
    "shop": (0.0, 0.2, 0.2), "workshop": (0.0, 0.3, 0.2), "money": (0.0, 0.4, 0.4),
    "contract": (-0.1, 0.3, 0.4), "engineer": (0.1, 0.2, 0.1), "licens": (0.0, 0.2, 0.3),
    "signature": (0.0, 0.3, 0.4), "sign": (0.0, 0.2, 0.3),
    # --- added after the first self-review pass (missed plot words)
    "felon": (-0.8, 0.6, 0.8), "weapon": (-0.7, 0.7, 0.85), "traffick": (-0.8, 0.6, 0.8),
    "smuggl": (-0.6, 0.6, 0.8), "testif": (-0.2, 0.6, 0.7), "sentenc": (-0.4, 0.6, 0.7),
    "confront": (-0.5, 0.8, 0.8), "accus": (-0.6, 0.7, 0.7), "suspicio": (-0.4, 0.5, 0.75),
    "caught": (-0.3, 0.7, 0.7), "report": (-0.1, 0.4, 0.5), "stare": (-0.2, 0.5, 0.6),
    "staring": (-0.2, 0.5, 0.6), "whisper": (-0.1, 0.3, 0.6), "penalt": (-0.5, 0.5, 0.6),
    "flagged": (-0.5, 0.5, 0.65), "insurance": (-0.1, 0.3, 0.4), "garbage": (-0.6, 0.6, 0.4),
    "operation": (-0.1, 0.5, 0.6), "evict": (-0.7, 0.6, 0.7), "ruin": (-0.8, 0.6, 0.7),
    "not mine": (-0.4, 0.6, 0.8), "blow your cover": (-0.3, 0.7, 0.85), "took down": (0.2, 0.8, 0.6),
}

# Colour dimensions: what the story FEELS like beyond affect. Prefix stems.
_COLOUR = {
    "darkness": ["dark", "night", "shadow", "blood", "kill", "murder", "dead", "death", "grave",
                 "crime", "gang", "gun", "threat", "cold", "basement", "abuse", "secret"],
    "warmth": ["love", "hug", "family", "home", "mother", "mom", "dad", "father", "grandm",
               "grandp", "child", "kid", "baby", "kind", "gentle", "thank", "smile", "wedding"],
    "wonder": ["magic", "wonder", "star", "sky", "dream", "miracle", "beautiful", "amazing",
               "discover", "ancient", "space", "light"],
    "grit": ["biker", "motorcycle", "gang", "bar", "beer", "whiskey", "truck", "garage",
             "workshop", "street", "fight", "punch", "rusty", "dust", "desert", "phoenix"],
    "intimacy": ["i felt", "my heart", "alone", "quiet", "whisper", "letter", "diary",
                 "remember", "memory", "confess", "never told", "nobody knew"],
    "urgency": ["suddenly", "run", "ran", "chase", "seconds", "minutes", "deadline", "rush",
                "now", "immediately", "hurry", "pounding", "racing", "three seconds"],
    "melancholy": ["miss", "lost", "loss", "regret", "funeral", "grief", "lonely", "used to",
                   "years ago", "anymore", "goodbye", "cried", "tears", "empty"],
    "triumph": ["win", "won", "victory", "justice", "karma", "finally", "promot", "proved",
                "fired him", "fired her", "got what", "applause", "cheer"],
    "eeriness": ["strange", "weird", "creepy", "noise", "footstep", "ghost", "haunt",
                 "watching", "stare", "staring", "basement", "attic", "woods", "missing"],
    "playfulness": ["funny", "laugh", "joke", "silly", "prank", "ridiculous", "classy",
                    "lol", "petty", "karen", "entitled"],
    "deceit": ["undercover", "fake", "lie", "lied", "forg", "fraud", "pretend", "cover",
               "disguise", "secret", "double", "spy", "scam", "forged", "signature that"],
    "pressure": ["deadline", "inspector", "marshal", "court", "loan", "debt", "boss",
                 "recertif", "audit", "pending", "shut down", "evict", "fired"],
}

# Setting cues add style words (not instruments forced on the story).
_SETTING = {
    "urban night": ["city", "street", "night", "club", "bar", "alley", "apartment", "subway"],
    "small-town americana": ["town", "truck", "diner", "farm", "county", "highway", "phoenix",
                             "texas", "biker", "motorcycle", "rusty"],
    "corporate": ["office", "boss", "meeting", "company", "corporate", "hr ", "manager", "email"],
    "domestic": ["house", "kitchen", "bedroom", "home", "husband", "wife", "married", "mom",
                 "dad", "brother", "sister"],
    "industrial": ["workshop", "factory", "warehouse", "rigging", "wiring", "mezzanine",
                   "engineer", "structural", "machine", "lacquer"],
    "legal / institutional": ["court", "lawyer", "judge", "police", "inspector", "marshal",
                              "license", "licens", "permit", "county"],
    "nature": ["forest", "woods", "lake", "ocean", "mountain", "river", "rain", "snow", "field"],
    "medical": ["hospital", "doctor", "nurse", "surgery", "diagnos", "er ", "icu"],
}

_NEGATORS = {"not", "never", "no", "didn't", "don't", "wasn't", "isn't", "couldn't", "won't",
             "nobody", "nothing", "without"}
_INTENSIFIERS = {"very": 1.3, "really": 1.25, "so": 1.2, "extremely": 1.5, "totally": 1.3,
                 "completely": 1.3, "absolutely": 1.4, "whole": 1.15, "hard": 1.2}
_REVEAL_CUES = ("never knew", "turns out", "truth", "actually", "realized", "found out",
                "the reason", "secret", "had been", "all along", "that's when", "was not mine",
                "wasn't mine", "the whole thing was", "it was him", "it was her", "already filed",
                "what i didn't know", "little did")
_RESOLUTION_CUES = ("now", "since then", "finally", "these days", "to this day", "ever since",
                    "in the end", "ended up", "lesson", "today", "anymore", "moved on")

_WORD = re.compile(r"[a-z']+")


def _tokens(text):
    return _WORD.findall(text.lower())


_SUFFIXES = ("", "s", "es", "ed", "d", "ing", "er", "ers", "y", "ly", "al", "ion", "ions",
             "ation", "ment", "ful", "ous", "ened", "en", "ence", "ent", "ies", "ied", "ery",
             "ive", "ity", "ness", "ure", "ures", "ured", "ated", "ating", "ate", "ish", "ic")


def _stem_ok(token, stem):
    """Prefix match that only accepts real inflections, so 'die' does not
    match 'diesel', 'star' does not match 'start', 'con' does not match
    'confirmed'. Stems of 6+ letters match any continuation."""
    if not token.startswith(stem):
        return False
    rest = token[len(stem):]
    if len(stem) >= 6:
        return True
    if rest in _SUFFIXES:
        return True
    # doubled final consonant: stop -> stopped, run -> running
    return len(rest) > 1 and rest[0] == stem[-1] and rest[1:] in _SUFFIXES


def _match_stem(token, stems):
    return [s for s in stems if " " not in s and _stem_ok(token, s)]


def _phrase_hits(text, phrases):
    return sum(1 for p in phrases if " " in p and p in text)


# ============================================================
# PER-SEGMENT FEATURES
# ============================================================

COLOUR_NAMES = tuple(_COLOUR.keys())


def segment_features(text):
    """Affect + colour vector for one transcript segment."""
    low = " " + text.lower() + " "
    toks = _tokens(text)
    v = a = t = 0.0
    hits = 0
    for i, tok in enumerate(toks):
        stems = _match_stem(tok, _AFFECT)
        if not stems:
            continue
        stem = max(stems, key=len)
        val, aro, ten = _AFFECT[stem]
        scale = 1.0
        window = toks[max(0, i - 3):i]
        if any(w in _NEGATORS for w in window):
            val, ten = -0.5 * val, 0.8 * ten          # "not safe" is tense, not happy
        for w in window:
            scale *= _INTENSIFIERS.get(w, 1.0)
        v += val * scale
        a += aro * scale
        t += ten * scale
        hits += 1
    for phrase in (p for p in _AFFECT if " " in p):
        if phrase in low:
            val, aro, ten = _AFFECT[phrase]
            v, a, t, hits = v + val, a + aro, t + ten, hits + 1
    norm = max(1.0, math.sqrt(hits))
    punct = text.count("!") * 0.15 + text.count("?") * 0.08 + text.count('"') * 0.03
    words = max(1, len(toks))
    terse = 0.15 if words <= 7 else 0.0               # short, punchy lines read as tension
    colour = {}
    for name, stems in _COLOUR.items():
        c = sum(1 for tok in toks if _match_stem(tok, stems)) + _phrase_hits(low, stems)
        colour[name] = c / math.sqrt(words)
    setting = {}
    for name, stems in _SETTING.items():
        setting[name] = sum(1 for tok in toks if _match_stem(tok, [s.strip() for s in stems]))
    return {
        "valence": float(np.tanh(v / norm)),
        "arousal": float(np.tanh(a / norm + punct)),
        "tension": float(np.tanh(t / norm + terse + punct * 0.5)),
        "affect_hits": hits,
        "colour": colour,
        "setting": setting,
        "reveal_cue": sum(1 for c in _REVEAL_CUES if c in low),
        "resolution_cue": sum(1 for c in _RESOLUTION_CUES if c in low),
        "words": words,
    }


# ============================================================
# MUSICAL VOCABULARY — composed from the story vector, not picked by mood
# ============================================================

# Instrument -> (role, colour affinities, affect affinities, prompt wording).
# Affinity is a dot product with the story's colour/affect vector, so a new
# combination of colours gives a new ensemble.
_INSTRUMENTS = [
    # lead voices
    ("muted prepared piano ostinato", "lead", {"pressure": 1.0, "deceit": 0.4, "urgency": 0.3}, {"tension": 0.3}),
    ("felt piano", "lead", {"intimacy": 1.0, "melancholy": 0.8, "warmth": 0.4}, {"arousal": -0.5}),
    ("solo cello", "lead", {"melancholy": 1.0, "intimacy": 0.6, "darkness": 0.4}, {"valence": -0.4}),
    ("clean electric guitar with tremolo", "lead", {"grit": 0.9, "deceit": 0.5, "darkness": 0.3}, {"tension": 0.3}),
    ("baritone guitar", "lead", {"grit": 1.0, "darkness": 0.5}, {"tension": 0.4}),
    ("muted trumpet", "lead", {"deceit": 0.6, "darkness": 0.3, "playfulness": 0.3}, {"arousal": -0.2}),
    ("pizzicato strings", "lead", {"playfulness": 1.0, "deceit": 0.3}, {"valence": 0.3}),
    ("music box", "lead", {"wonder": 0.6, "eeriness": 0.8, "intimacy": 0.4}, {}),
    ("upright piano", "lead", {"warmth": 0.8, "intimacy": 0.5}, {"valence": 0.4}),
    ("acoustic guitar fingerpicking", "lead", {"warmth": 0.9, "intimacy": 0.4}, {"valence": 0.5}),
    ("analog synth lead", "lead", {"urgency": 0.6, "deceit": 0.4}, {"arousal": 0.5}),
    ("french horn", "lead", {"triumph": 1.0, "wonder": 0.5}, {"valence": 0.4}),
    # harmony beds
    ("warm string ensemble", "harmony", {"warmth": 0.8, "melancholy": 0.6, "triumph": 0.5}, {}),
    ("dark analog pad", "harmony", {"darkness": 0.9, "deceit": 0.6, "eeriness": 0.4}, {"valence": -0.3}),
    ("low brass chords", "harmony", {"darkness": 0.5, "triumph": 0.6, "pressure": 0.5}, {"tension": 0.4}),
    ("hammond organ", "harmony", {"grit": 0.7, "warmth": 0.3}, {}),
    ("glassy granular pad", "harmony", {"eeriness": 0.9, "wonder": 0.6}, {}),
    ("soft choir pad without words", "harmony", {"wonder": 0.7, "melancholy": 0.5}, {"valence": 0.2}),
    ("rhodes electric piano", "harmony", {"intimacy": 0.6, "warmth": 0.5, "deceit": 0.2}, {"arousal": -0.3}),
    # bass
    ("deep sub bass", "bass", {"darkness": 0.8, "pressure": 0.6, "urgency": 0.3}, {"tension": 0.4}),
    ("upright bass", "bass", {"deceit": 0.6, "warmth": 0.5, "playfulness": 0.4}, {}),
    ("overdriven bass guitar", "bass", {"grit": 1.0, "urgency": 0.4}, {"arousal": 0.4}),
    ("low cello and contrabass", "bass", {"melancholy": 0.6, "darkness": 0.5, "triumph": 0.3}, {}),
    ("pulsing analog bass", "bass", {"urgency": 0.8, "pressure": 0.7, "deceit": 0.3}, {"arousal": 0.4}),
    # pulse / rhythm
    ("ticking clock-like percussion", "pulse", {"pressure": 1.0, "urgency": 0.6}, {"tension": 0.5}),
    ("brushed snare", "pulse", {"deceit": 0.6, "warmth": 0.4, "playfulness": 0.3}, {}),
    ("low tom ostinato", "pulse", {"urgency": 0.7, "darkness": 0.5, "triumph": 0.4}, {"arousal": 0.5}),
    ("stomping kick and handclap groove", "pulse", {"grit": 0.8, "triumph": 0.4}, {"arousal": 0.5}),
    ("staccato string ostinato", "pulse", {"urgency": 0.9, "pressure": 0.6}, {"tension": 0.5}),
    ("soft heartbeat pulse", "pulse", {"intimacy": 0.6, "eeriness": 0.4, "pressure": 0.4}, {"arousal": -0.2}),
    ("light shaker groove", "pulse", {"warmth": 0.6, "playfulness": 0.6}, {"valence": 0.4}),
    # texture / colour
    ("vinyl crackle and tape hiss", "texture", {"melancholy": 0.6, "intimacy": 0.6, "grit": 0.3}, {}),
    ("bowed metal and reversed swells", "texture", {"eeriness": 1.0, "darkness": 0.5}, {}),
    ("distant guitar feedback", "texture", {"grit": 0.8, "darkness": 0.3}, {}),
    ("shimmering celesta", "texture", {"wonder": 1.0, "warmth": 0.3}, {"valence": 0.3}),
    ("airy wind textures", "texture", {"melancholy": 0.5, "eeriness": 0.4}, {"arousal": -0.4}),
    ("low drones", "texture", {"darkness": 0.7, "pressure": 0.5}, {"tension": 0.3}),
    # hits (climax colour)
    ("big cinematic drum hits", "hits", {"triumph": 0.7, "urgency": 0.6, "darkness": 0.3}, {"arousal": 0.6}),
    ("taiko and low brass stabs", "hits", {"darkness": 0.6, "pressure": 0.6, "urgency": 0.5}, {"tension": 0.5}),
    ("driving rock drums", "hits", {"grit": 0.9, "urgency": 0.5}, {"arousal": 0.5}),
    ("string swells and cymbal rolls", "hits", {"melancholy": 0.5, "wonder": 0.5, "triumph": 0.5}, {}),
    ("orchestral hits with timpani", "hits", {"triumph": 0.8, "pressure": 0.3}, {"valence": 0.2}),
]
ROLES = ("lead", "harmony", "bass", "pulse", "texture", "hits")

# Colour -> adjectives used to word the style line freshly.
_COLOUR_WORDS = {
    "darkness": ("brooding", "shadowed", "nocturnal"),
    "warmth": ("tender", "warm", "heartfelt"),
    "wonder": ("luminous", "awe-struck", "glowing"),
    "grit": ("gritty", "dusty", "raw"),
    "intimacy": ("intimate", "close", "confessional"),
    "urgency": ("driving", "restless", "propulsive"),
    "melancholy": ("bittersweet", "wistful", "aching"),
    "triumph": ("rising", "defiant", "victorious"),
    "eeriness": ("uneasy", "eerie", "haunted"),
    "playfulness": ("wry", "playful", "cheeky"),
    "deceit": ("noir", "double-edged", "sly"),
    "pressure": ("tense", "taut", "tightening"),
}
_SETTING_WORDS = {
    "urban night": "neo-noir city-at-night atmosphere",
    "small-town americana": "dusty americana edge",
    "corporate": "sleek modern minimal sheen",
    "domestic": "close, human, living-room scale",
    "industrial": "mechanical, metallic undertone",
    "legal / institutional": "procedural, measured restraint",
    "nature": "open, organic air",
    "medical": "clinical, sterile stillness",
}

_KEYS = ("C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")
# Low-register keys read darker on most instruments; bright ones lift.
_DARK_KEYS = ("C", "D", "Eb", "F", "G", "Bb")
_BRIGHT_KEYS = ("D", "E", "F#", "A", "B", "G")


# ============================================================
# CUE SHEET
# ============================================================

@dataclass
class CueSheet:
    style: str
    tempo_bpm: int
    key: str
    mode: str
    palette: dict                 # role -> instrument wording
    motif: str
    descriptors: list             # leading adjectives, strongest first
    affect: dict                  # story-level valence/arousal/tension
    colour: dict                  # story-level colour vector (normalised)
    settings: list
    arc: list                     # per-section intensity 0..1 (7 values)
    section_moods: list           # per-section descriptor words
    hit_points: dict              # name -> seconds
    segment_intensity: list = field(default_factory=list)
    # What the text asked for, kept when the grid adopts the motif's own
    # tempo/key (adopt_motif_tempo_key); None until then.
    story_tempo_bpm: float = None
    story_key_label: str = None

    @property
    def key_label(self):
        return f"{self.key} {self.mode}"

    def to_dict(self):
        d = asdict(self)
        d["key_label"] = self.key_label
        return d


def fold_tempo(bpm, hint, lo=60.0, hi=130.0):
    """The octave (x0.5, x1, x2, x4) of bpm nearest the hint, inside lo..hi."""
    cands = [bpm * k for k in (0.25, 0.5, 1.0, 2.0, 4.0) if lo <= bpm * k <= hi]
    if not cands:
        return float(hint)
    return float(min(cands, key=lambda b: abs(np.log(b / hint))))


def adopt_motif_tempo_key(cue, tempo_bpm, confidence, tonic, family, min_confidence=0.08):
    """The composer keeps the theme's key and pulse. The model follows the
    BPM/key words loosely, but every layer is conditioned on the motif, so
    the bar grid and the layer prompts take the motif's measured tempo and
    key; the story still decides the mode colour (dorian stays dorian when
    the motif reads minor-family) and everything else. Returns a new cue."""
    from dataclasses import replace
    # Bundled Kaggle file: music_analysis is inlined after this module.
    mode_family = globals().get("MODE_FAMILY") or __import__("music_analysis").MODE_FAMILY
    new = replace(cue, story_tempo_bpm=cue.tempo_bpm, story_key_label=cue.key_label)
    if tempo_bpm > 0 and confidence >= min_confidence:
        folded = fold_tempo(tempo_bpm, cue.tempo_bpm)
        # Within 1 % is estimator noise: the motif already sits on the grid.
        if abs(np.log(folded / cue.tempo_bpm)) > np.log(1.01):
            new.tempo_bpm = round(folded, 1)
    fam, _ = mode_family.get(cue.mode, ("minor", 0))
    new.key = tonic
    if family != fam:
        new.mode = "major" if family == "major" else "aeolian"
    return new


def _smooth(values, width):
    if len(values) == 0:
        return np.zeros(0)
    width = max(1, int(width))
    kernel = np.hanning(2 * width + 3)[1:-1]
    kernel /= kernel.sum()
    padded = np.pad(values, (width, width), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: len(values)]


def _fingerprint(text):
    return int.from_bytes(hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest(), "big")


def _choose_mode(valence, tension, colour):
    # Continuous rules, documented so a musician can audit them.
    if colour.get("eeriness", 0) > 0.6 and valence < 0:
        return "phrygian"
    if valence < -0.25 and tension > 0.55:
        return "harmonic minor" if colour.get("darkness", 0) > colour.get("deceit", 0) else "aeolian"
    if valence < -0.1:
        return "dorian" if colour.get("grit", 0) + colour.get("deceit", 0) > 0.8 else "aeolian"
    if colour.get("wonder", 0) > 0.6:
        return "lydian"
    if valence < 0.25:
        return "mixolydian" if colour.get("grit", 0) > 0.5 else "dorian"
    return "major"


def _motif_shape(valence, tension, colour):
    if colour.get("triumph", 0) > 0.6:
        contour = "rising perfect fourth then a step up, resolving upward"
    elif tension > 0.6 and valence < 0:
        contour = "falling minor second then a tritone drop, unresolved"
    elif valence < -0.2:
        contour = "descending minor third then a step down"
    elif colour.get("wonder", 0) > 0.5:
        contour = "rising major sixth then falling back a step"
    else:
        contour = "rising major second then a falling fourth"
    return contour


def _score_instrument(entry, colour, affect):
    _, _, caff, aaff = entry
    s = sum(colour.get(k, 0.0) * w for k, w in caff.items())
    s += sum(affect.get(k, 0.0) * w for k, w in aaff.items()) * 0.5
    return s


def _section_ranges(boundaries, segments):
    out = []
    for i in range(len(boundaries) - 1):
        lo, hi = boundaries[i], boundaries[i + 1]
        idx = [j for j, s in enumerate(segments) if lo <= 0.5 * (s["start"] + s["end"]) < hi]
        out.append(idx)
    return out


def build_cue_sheet(segments, duration, boundaries=None):
    """segments: [{"text","start","end"}...] (story_analysis.normalize_text).
    boundaries: the 8 story-section boundaries (seconds), optional."""
    segments = [s for s in segments if str(s.get("text", "")).strip()]
    full_text = " ".join(s["text"] for s in segments)
    feats = [segment_features(s["text"]) for s in segments] or [segment_features("")]
    n = len(feats)
    weights = np.array([max(1, f["words"]) for f in feats], dtype=np.float64)

    def wmean(key):
        return float(np.average([f[key] for f in feats], weights=weights))

    affect = {"valence": wmean("valence"), "arousal": wmean("arousal"), "tension": wmean("tension")}
    colour_raw = {c: float(np.average([f["colour"][c] for f in feats], weights=weights)) for c in COLOUR_NAMES}
    top = max(colour_raw.values()) or 1.0
    colour = {c: v / top for c, v in colour_raw.items()}          # strongest colour = 1.0
    setting_tot = {k: sum(f["setting"][k] for f in feats) for k in _SETTING}
    settings = [k for k, v in sorted(setting_tot.items(), key=lambda kv: -kv[1]) if v >= 2][:1]

    # Intensity curve: arousal + tension + local reveal cues, smoothed.
    raw = np.array([0.45 * f["arousal"] + 0.45 * f["tension"] + 0.1 * min(1, f["reveal_cue"])
                    for f in feats])
    curve = _smooth(raw, max(1, n // 12))
    lo, hi = float(curve.min()), float(curve.max())
    curve = (curve - lo) / (hi - lo) if hi - lo > 1e-6 else np.full(n, 0.5)

    starts = [s["start"] for s in segments] or [0.0]
    pos = (np.arange(n) + 0.5) / max(n, 1)
    # Climax: intensity weighted by where stories usually pay off (a soft
    # prior peaking around 75 %, never the first 45 % or last 8 %).
    prior = np.exp(-0.5 * ((pos - 0.75) / 0.15) ** 2)
    allowed = (pos >= 0.45) & (pos <= 0.92)
    climax_score = np.where(allowed, curve + 0.35 * prior, -np.inf)
    climax_i = int(np.argmax(climax_score)) if n > 2 else n - 1
    # Twist: where the story turns: strongest reveal cue or the steepest rise
    # in intensity, before the climax and after 35 %.
    rise = np.diff(curve, prepend=curve[0])
    # Novelty: deceit/darkness colour appearing where the story had little
    # of it so far is what a turn usually sounds like ("a signature that
    # was not mine", "it was a test").
    turn_col = np.array([f["colour"]["deceit"] + f["colour"]["darkness"] + f["colour"]["eeriness"]
                         for f in feats])
    seen = np.concatenate([[0.0], np.cumsum(turn_col)[:-1]]) / np.maximum(np.arange(n), 1)
    novelty = np.maximum(turn_col - seen, 0.0)
    twist_score = np.array([feats[i]["reveal_cue"] * 0.3 + rise[i] * 2.0 + feats[i]["tension"] * 0.2
                            + novelty[i] * 0.8 for i in range(n)])
    twist_ok = (pos >= 0.35) & (np.arange(n) < climax_i)
    twist_i = int(np.argmax(np.where(twist_ok, twist_score, -np.inf))) if twist_ok.any() else max(0, climax_i - 1)
    # Resolution: first resolution cue after the climax, else halfway from
    # the climax to the end.
    res_cues = [i for i in range(climax_i + 1, n) if feats[i]["resolution_cue"] > 0]
    end_t = float(duration)
    climax_t = float(starts[climax_i])
    resolution_t = float(starts[res_cues[0]]) if res_cues else climax_t + 0.5 * (end_t - climax_t)
    hit_points = {
        "twist": float(starts[twist_i]),
        "climax": climax_t,
        "resolution": max(resolution_t, climax_t + 0.04 * end_t) if resolution_t < end_t else resolution_t,
        "end": end_t,
    }

    # Tempo: calm 62 -> frantic 118 BPM, nudged by urgency; whole BPM.
    # Narration lexica skew positive on arousal, so 0.1 is "calm" and 0.7
    # "frantic"; urgency colour adds drive.
    base = float(np.clip((affect["arousal"] - 0.1) / 0.6, 0, 1))
    energy = 0.75 * base + 0.25 * float(colour["urgency"])
    tempo = int(round(62 + 56 * energy))

    mode = _choose_mode(affect["valence"], affect["tension"], colour)
    fp = _fingerprint(full_text)
    dark = mode in ("aeolian", "harmonic minor", "phrygian", "dorian")
    pool = _DARK_KEYS if dark else _BRIGHT_KEYS
    key = pool[fp % len(pool)]

    # Palette: best instrument per role against this story's vector.
    palette = {}
    for role in ROLES:
        cands = [e for e in _INSTRUMENTS if e[1] == role]
        ranked = sorted(cands, key=lambda e: (-_score_instrument(e, colour, affect), e[0]))
        palette[role] = ranked[0][0]

    order = sorted(COLOUR_NAMES, key=lambda c: -colour[c])
    descriptors = []
    for c in order[:3]:
        words = _COLOUR_WORDS[c]
        descriptors.append(words[(fp >> (3 * len(descriptors))) % len(words)])
    style_bits = [", ".join(descriptors), "cinematic underscore"]
    style_bits += [_SETTING_WORDS[s] for s in settings]
    style = ", ".join(style_bits)

    # Per-section arc and moods.
    if boundaries is None:
        boundaries = list(GENERIC_BOUNDARY_FRACTIONS * duration)
    seg_for = _section_ranges(boundaries, segments)
    arc, section_moods = [], []
    for idx in seg_for:
        if idx:
            arc.append(float(np.mean(curve[idx])))
            sec_col = {c: float(np.mean([feats[j]["colour"][c] for j in idx])) for c in COLOUR_NAMES}
            best = max(sec_col, key=lambda c: (sec_col[c], colour[c]))
            sec_val = float(np.mean([feats[j]["valence"] for j in idx]))
            mood_word = _COLOUR_WORDS[best][0] if sec_col[best] > 0 else ("hopeful" if sec_val > 0.2 else "restrained")
            section_moods.append(mood_word)
        else:
            arc.append(arc[-1] if arc else 0.3)
            section_moods.append(section_moods[-1] if section_moods else "restrained")

    return CueSheet(
        style=style, tempo_bpm=tempo, key=key, mode=mode, palette=palette,
        motif=_motif_shape(affect["valence"], affect["tension"], colour),
        descriptors=descriptors, affect=affect, colour=colour, settings=settings,
        arc=arc, section_moods=section_moods, hit_points=hit_points,
        segment_intensity=[float(x) for x in curve],
    )


# ============================================================
# PROMPTS (Stable Audio 3 Small Music — text-conditioned, T5 encoder)
#
# Stable Audio responds best to comma-separated tags: genre/style, then
# instruments, then mood, then BPM and key. Every prompt for one story
# carries the SAME tempo and key so phrases line up when they're joined.
# ============================================================

NEGATIVE_TAIL = "instrumental only, no vocals, no speech, no lyrics"


# Real Stable Audio renders (2026-10-04) of a sparse palette came out as
# isolated stabs separated by seconds of silence, and the low-noise
# conditioned phrases copied the gaps. Under narration the bed must never
# stop, so every prompt states a sustained harmony bed explicitly.
BED = "legato, continuous sustained bed, no gaps, no silence"


def motif_prompt_from_cue(cue):
    p = cue.palette
    return (f"{cue.style}, memorable three-note motif on {p['lead']} over sustained {p['harmony']}, "
            f"{p['bass']}, {BED}, {cue.tempo_bpm:g} BPM, {cue.key_label}, no fade out, {NEGATIVE_TAIL}")


def layer_prompts_from_cue(cue):
    p = cue.palette
    d = cue.descriptors
    tail = f"{BED}, {cue.tempo_bpm:g} BPM, {cue.key_label}, steady loop, no fade out, {NEGATIVE_TAIL}"
    return {
        "core": (f"{d[0]} {d[-1]} cinematic underscore, same motif on {p['lead']}, sustained "
                 f"{p['harmony']}, {p['bass']}, understated, room for narration, {tail}"),
        "pressure": (f"{d[min(1, len(d) - 1)]} rising tension, same motif, {p['pulse']}, sustained "
                     f"{p['harmony']}, {p['bass']}, {p['texture']}, {tail}"),
        "climax": (f"{d[0]} cinematic climax, same motif on {p['lead']}, {p['hits']}, "
                   f"{p['pulse']}, full sustained {p['harmony']}, powerful but controlled, {tail}"),
    }


def describe(cue):
    """Short human-readable lines for logs/manifests."""
    return [
        f"Style: {cue.style}",
        f"Tempo/key: {cue.tempo_bpm:g} BPM, {cue.key_label}",
        "Palette: " + "; ".join(f"{r}={cue.palette[r]}" for r in ROLES),
        f"Motif: {cue.motif}",
        "Arc: " + " ".join(f"{x:.2f}" for x in cue.arc),
        "Hit points: " + ", ".join(f"{k} {v:.1f}s" for k, v in cue.hit_points.items()),
    ]
