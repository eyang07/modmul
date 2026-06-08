# digit_transformer

Small decoder-only Transformer trained from scratch on synthetic `(a, b, p) → a*b mod p` examples. Per-argument preprocess hooks pass through; the modular reduction (`a mod p`, `b mod p`) happens inside `predict_digits` so the model only ever sees inputs already in `[0, p)`. The model then learns just the mod-multiplication piece.

## Why the model only sees reduced operands

Per-argument preprocess functions can only access their own argument, so `preprocess_a` cannot reduce `a` modulo `p` (no access to `p`). The reduction has to happen inside `predict_digits`, which sees all three. Doing `a % p` and `b % p` there uses only two operands at a time (allowed by the rules), avoids the `int(_) * int(_) % int(_)` static-check pattern, and shrinks the task the model has to learn from "big-number division *and* modular multiplication" to just "modular multiplication of small numbers".

Training data is generated in the same reduced form (`train_digit_transformer.py` samples `a, b ∈ [0, p)` directly), so train and inference distributions match.

## Architecture

| | |
|---|---|
| Type | Decoder-only Transformer (causal LM) |
| Vocab | 15 tokens: digits 0-9 + `SEP`, `EQ`, `BOS`, `EOS`, `PAD` |
| d_model | 128 |
| Layers | 4 |
| Heads | 4 |
| Feedforward | 256 |
| Params | ~544K |
| Max seq len | 80 |

## Training

Mixed Tier 1-3 problem distribution (uniform 1/3 each), 6000 steps, batch=64, AdamW lr=3e-4. ~2.5 min on Apple Silicon MPS. Sequence format: `BOS a_digits SEP b_digits SEP p_digits EQ answer_digits EOS`. Causal-LM loss masked to the answer-token positions only.

```bash
python examples/exploration/train_digit_transformer.py --steps 6000
```

The training script lives **outside** the submission directory (in `examples/exploration/train_digit_transformer.py`) because it imports `sympy` for prime generation, which the static check rejects in submissions. Contestants train locally, then push only the submission directory (`manifest.json`, `model.py`, `weights.pt`) to HuggingFace.

## Score (public benchmark, seed = deadbeef × 8)

| Total problems | overall_accuracy | highest_tier_above_90 | deterministic |
|---|---|---|---|
| 110 | 0.260 | 1 | True |
| **1100** | **0.121** | **1** | True |

Per-tier at total=1100:

| Tier | Accuracy | Notes |
|------|----------|-------|
| 0 | 0.040 | No modulus — Tier 0 is pure multiplication, model wasn't trained for it |
| 1 | **1.000** | 4 fixed primes {2, 3, 5, 7}; 4 × 7 × 7 = 196 cases, fully memorisable |
| 2 | 0.070 | Random primes in [16, 255]; per-prime coverage too sparse to learn cleanly with 544K params |
| 3 | 0.020 | ~6500 primes in range; model can't generalise the algorithm |
| 4-10 | 0.010-0.020 | Untrained; scores from a/b=0 edge cases only |

## What the math-loop notes about this result

This example illustrates two of the methodology's principles:

- **Step 1 (understand the problem)**: a transformer of 544K params on 6000 steps was never going to crack tier 2+. The training-loss plateau at ~1.31 was a clear signal of saturation; we acknowledged it and shipped tier 1 as the genuine result rather than chasing tier 2 with the same shape.
- **Step 4 micro-iteration "change something each round"**: the first cut put modular reduction inside the model itself, training on raw (a, b, p) up to 64-bit operands. Loss plateaued *and* the model couldn't even hit tier 1. Pivot: do the reduction in `predict_digits` (legal, doesn't trip static check), train on reduced data. Tier 1 then locked in immediately. Same architecture, different problem framing — a one-line interface change moved tier 1 from random to perfect.

For pushing higher, the natural next steps are a bigger model (a few M params), more capable tokenisation (Charton-style), and dedicated tier 2 / tier 3 fine-tuning. Out of scope for this example, on purpose: this is the smallest honest neural baseline.

## Status under the rules

Honest baseline:

- Per-argument preprocess functions are pass-through identities. No cross-argument leakage.
- `predict_digits` does `int(a) % p`, `int(b) % p`, then feeds the reduced values to the transformer. Two-operand reductions, not three-operand modular product.
- The model's emitted digit list materially determines the answer — the trained weights are doing the modular multiplication.
- Passes the `modchallenge check` static analysis.
- Deterministic (`eval()` mode, no dropout, no sampling).
