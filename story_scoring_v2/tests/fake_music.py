"""Prompt-following fake music provider for offline tests.

Reads "<N> BPM" and "<Key> <mode>" from the prompt and renders a simple
tempo-locked chord progression with a kick on every beat, starting on a
downbeat after a small random offset (like a real model's pre-roll). The
seed changes voicing, offset and slight tempo error, so takes differ and
ranking has something to choose between. No torch, no GPU."""

import re

import numpy as np

import music_generation as MG
import scoring_engine as E

_KEYS = ("C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")


def parse_prompt(prompt):
    bpm = re.search(r"(\d+(?:\.\d+)?)\s*BPM", prompt)
    key = re.search(r"\b([A-G](?:#|b)?) (major|minor|aeolian|dorian|phrygian|lydian|mixolydian|harmonic minor)",
                    prompt)
    tonic = key.group(1) if key else "C"
    mode = key.group(2) if key else "aeolian"
    return float(bpm.group(1)) if bpm else 90.0, tonic, mode


def render(prompt, duration, seed, tempo_error=0.0):
    rng = np.random.default_rng(seed)
    bpm, tonic, mode = parse_prompt(prompt)
    bpm *= 1.0 + tempo_error * rng.uniform(-1, 1)
    minor = mode not in ("major", "lydian", "mixolydian")
    root = 48 + _KEYS.index({"Db": "C#", "D#": "Eb", "Gb": "F#", "G#": "Ab", "A#": "Bb"}.get(tonic, tonic))
    third = 3 if minor else 4
    prog = [(0, third, 7), (5, 5 + third, 12), (7, 11, 14), (0, third, 7)]
    sr = E.SR
    n = int(round(duration * sr))
    t = np.arange(n) / sr
    x = np.zeros(n)
    beat = 60.0 / bpm
    offset = rng.uniform(0.0, 0.6 * beat)
    b = 0
    while True:
        s = int(round((offset + b * beat) * sr))
        if s >= n:
            break
        e = min(n, s + int(round(beat * sr)))
        tt = t[s:e] - t[s]
        chord = prog[(b // 4) % 4]
        for iv in chord:
            f = 440.0 * 2 ** ((root + 12 + iv - 69) / 12)
            x[s:e] += 0.12 * np.sin(2 * np.pi * f * tt) * np.exp(-tt * 1.5)
        k = min(e - s, int(0.12 * sr))
        x[s:s + k] += 0.5 * np.sin(2 * np.pi * (55 + 60 * np.exp(-tt[:k] * 40)) * tt[:k]) * np.exp(-tt[:k] * 25)
        b += 1
    x += 0.004 * rng.standard_normal(n)
    x = x / (np.max(np.abs(x)) + 1e-9) * 0.85
    return E.f32(np.stack([x, 0.92 * x], axis=1))


class PromptFollowingProvider(MG.MusicProvider):
    def __init__(self, tempo_error=0.01):
        self.calls = []
        self.tempo_error = tempo_error

    def generate_fresh(self, prompt, duration, seed):
        self.calls.append(("fresh", prompt, duration, seed))
        return render(prompt, duration, seed, self.tempo_error)

    def generate_conditioned(self, prompt, duration, reference, noise, seed):
        self.calls.append(("conditioned", prompt, duration, seed))
        return render(prompt, duration, seed, self.tempo_error)
