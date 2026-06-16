---
license: mit
library_name: pytorch
tags:
  - modular-arithmetic
  - scratchpad
  - grokking
  - transformer
  - sair-challenge
---

# modmul-tier3 — learned modular multiplication

A submission to the **SAIR Modular Arithmetic Challenge**: compute `(a · b) mod p`
with the arithmetic produced entirely by **trained parameters** — no `%`, `//`,
Barrett, Montgomery, or CRT on the product.

## What it does

Two trained networks behind one interface, routed by the size of `p`:

| `p` range | Model |
|---|---|
| `p < 512` (tiers 1–2) | joint-attention Transformer with a classification head over the residue |
| `512 ≤ p < 65536` (tier 3) | **autoregressive "abacus" decoder** that emits an interleaved modular-multiply scratchpad |
| `p ≥ 65536` | out of trained range → returns `[0]` |

Operands are reduced per-argument (`a mod p`, `b mod p`) before the network runs;
the **product's** reduction is learned.

## The tier-3 idea

A one-shot 5-digit × 5-digit multiply is a hard learnability wall. Instead the model
runs **Horner's method**, reducing mod `p` at every step so no intermediate ever
exceeds ~6 digits — the regime the decoder handles reliably. Every intermediate is
written to the scratchpad and supervised:

```
BOS x MUL y MOD p EQ  d:q1:r1:pp:t:q2:r2  STEP  …  EOS
```

per `y`-digit: shift-reduce (`q1,r1`), single-digit partial product (`pp`),
the explicit add `t = r1 + pp`, then add-reduce (`q2,r2`). Emitting `t` explicitly
was the decisive fix — leaving that addition implicit capped end-to-end accuracy.

## Results

Verified through the official `modchallenge evaluate-hf` pipeline:

| Tier | Accuracy |
|---|---|
| 1 | 1.00 |
| 2 | 0.99 |
| **3** | **1.00** |

**`highest_tier_above_90 = 3`**, deterministic, static-compliance check passed.
A 30-seed local sweep gave htop90 = 3 on all 30 (tier-3 min 0.95 / mean 0.986).

## Files

- `model.py` — the `EBMModMul` entry class (loader + abacus decoder)
- `manifest.json` — entry class, `output_base=10`, descriptions
- `weights.pt` — trained weights for both routed models
