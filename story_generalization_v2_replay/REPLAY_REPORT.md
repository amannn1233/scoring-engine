# Generalization V2 — offline replay on the V1 phrases

Same phrases, same story boundaries and same objective as Generalization V1.
Only the optimizer and render/measurement path changed.

> Audio in this folder is lossless **FLAC** (bit-identical to the 16-bit WAVs the code writes; verified sample-for-sample) because the WAVs exceed the file-transfer limit. Running `python story_scoring_v2/replay_v2.py` from `Desktop/scoring` regenerates the WAVs locally (~5 min).

## Does the optimizer now match the final waveform?

| Story | Optimizer objective | Final objective | Optimizer peak gap | Final peak gap | Worst diff vs render | Worst diff vs 16-bit file | Aligned |
|---|---:|---:|---:|---:|---:|---:|:---:|
| 71_extracted.wav | 0.9546 | 0.9546 | +1.399 dB | +1.399 dB | 0.000000 dB | 0.0003 dB | yes |
| 72_extracted.wav | 0.3958 | 0.3958 | +1.214 dB | +1.214 dB | 0.000000 dB | 0.0003 dB | yes |
| 86_audio.mp3 | 0.8448 | 0.8448 | +1.492 dB | +1.492 dB | 0.000000 dB | 0.0003 dB | yes |

## V1 vs V2 on identical material

V1 numbers are what the V1 optimizer reported, then the V1 final waveform. V2 numbers are the final waveform.

| Story | V1 claimed peak gap | V1 actual peak gap | V2 peak gap | V1 max jump | V2 max jump | V1 true objective | V2 objective |
|---|---:|---:|---:|---:|---:|---:|---:|
| 71_extracted.wav | +1.42 dB | +2.14 dB | +1.40 dB | 5.15 dB | 2.33 dB | 22.7737 | 0.9546 |
| 72_extracted.wav | +2.11 dB | +0.72 dB | +1.21 dB | 5.90 dB | 2.77 dB | 17.3468 | 0.3958 |
| 86_audio.mp3 | +1.34 dB | -0.10 dB | +1.49 dB | 4.32 dB | 2.59 dB | 15.2617 | 0.8448 |

## Final story energy (dB) — V2

| Story | Hook | Setup | Consequence | Hidden | Discovery | Revelation | Aftermath |
|---|---:|---:|---:|---:|---:|---:|---:|
| target | -26.0 | -24.8 | -23.9 | -22.7 | -22.0 | -20.0 | -25.3 |
| 71_extracted.wav | -28.13 | -25.63 | -23.72 | -22.77 | -21.48 | -20.08 | -25.51 |
| 72_extracted.wav | -25.25 | -23.94 | -23.65 | -22.70 | -21.52 | -20.31 | -25.02 |
| 86_audio.mp3 | -24.16 | -24.38 | -23.59 | -22.10 | -21.75 | -20.25 | -26.15 |

## Boundary jumps (dB) — V2

| Story | Hook→Setup | Setup→Cons. | Cons.→Hidden | Hidden→Disc. | Disc.→Rev. | Rev.→After. |
|---|---:|---:|---:|---:|---:|---:|
| 71_extracted.wav | +1.94 | +1.57 | +1.09 | -1.00 | -1.60 | -2.33 |
| 72_extracted.wav | -2.77 | +0.93 | +1.79 | -0.59 | +2.09 | +2.40 |
| 86_audio.mp3 | -2.21 | -1.57 | +2.49 | -2.59 | -2.23 | -2.31 |

## Other checks

| Story | Local valley penalty | Motif sim. min / mean | Avg duck | Voice source | Renderer reproduces V1 file |
|---|---:|---:|---:|---|---:|
| 71_extracted.wav | 0.0000 | 0.950 / 0.978 | -4.31 dB | recovered (duck check Δ 0.0000 dB) | ≤ 2 LSB |
| 72_extracted.wav | 0.0000 | 0.956 / 0.970 | -4.29 dB | recovered (duck check Δ 0.0000 dB) | ≤ 1 LSB |
| 86_audio.mp3 | 0.0000 | 0.893 / 0.951 | -4.29 dB | recovered (duck check Δ 0.0000 dB) | ≤ 2 LSB |
