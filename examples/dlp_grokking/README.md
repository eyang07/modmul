# dlp_grokking

A compliant neural model for `a*b mod p` built on the **discrete-log (DLP) grokking** idea
that came out of the design discussion. It is the strongest of the three reference
models on the public benchmark, and a worked example of how to turn a mathematical
insight into an *inductive bias* without crossing into a hand-coded algorithm.

## The idea

Over the prime field, `a*b mod p` is the multiplicative group operation. Pick a generator
`g`; then every nonzero residue is `g^k`, and

```
a*b mod p   =   g^((log_a + log_b) mod (p-1))
```

so multiplication becomes **addition in log space**. The grokking literature (Power et al.;
Nanda et al.'s "Progress measures for grokking") has shown that small Transformers *can*
learn modular **addition** by discovering a Fourier / discrete-log representation on their own.

We give the network exactly one structural nudge toward that solution and let it learn
everything else from data:

```
e_a = Enc(a mod p, p)          # shared residue encoder
e_b = Enc(b mod p, p)          # same weights
z   = e_a + e_b                # ADDITIVE bottleneck  <- the only DLP bias
ans = Dec(z, p)                # decoder emits answer digits
```

The additive combination is the whole inductive bias. The embedding that turns a residue
into something log-like, and the decoder that turns a sum-in-log-space back into a residue,
are **all trained parameters**. There is no precomputed discrete-log table, no generator
search, no `(log_a + log_b) % (p-1)` written in Python. Perturb the weights and the accuracy
collapses — the operational test for "the answer came from learning, not a hand-coded
circuit" (rules/evaluation.md, Principle 2).

## Architecture

| | |
|---|---|
| Residue encoder | Transformer encoder, shared across the a- and b-branch |
| d_model | 256 |
| Layers / heads | 3 / 8 |
| Feedforward | 768 |
| Bottleneck | `z = e_a + e_b` (additive, log-space) |
| Decoder | re-encodes `p` as context, MLP → 5 digit heads (base-10) |
| Answer width | 5 digits, MSB-first, zero-padded |
| Params | ~6M |

Inputs are reduced (`a mod p`, `b mod p`) inside `predict_digits` — the same legal
two-operand reduction the `digit_transformer` baseline uses, so the network only has to learn
modular *multiplication* of small numbers, not big-number division. For `p >= 10^5` (out of
the small-prime regime the model can learn) it emits `0` without invoking the network — the
honest fallback rather than a guess.

## Training

```bash
.venv312/bin/python examples/dlp_grokking/train.py --minutes 8
```

Synthetic `(a mod p, b mod p, p) -> (a*b mod p)` sampled across **random** primes spanning the
tier 1-3 bit ranges, with a curriculum bias toward the small primes the network can actually
generalise over. Primes are split into a train pool and a **held-out val pool**, so the val
metric measures generalisation to *unseen primes* — the thing that separates "learned the field
structure" from "memorised one prime". Cross-entropy on the fixed-width answer digits, AdamW with
cosine decay, Apple MPS. Best-by-val checkpoint is saved to `weights.pt`.

Evaluate through the real pipeline (manifest → static check → load → determinism → inference →
decode → score):

```bash
.venv312/bin/python examples/dlp_grokking/eval_tiers.py examples/dlp_grokking
```

## Score (public benchmark, seed = deadbeef × 8)

| Total problems | overall_accuracy | deterministic |
|---|---|---|
| **1100** | **0.127** | True |

Per-tier at total=1100:

| Tier | Accuracy | Notes |
|------|----------|-------|
| 1 | **1.000** | 4 fixed primes {2,3,5,7} — a genuine grok; matches the single-prime regime the papers solve |
| 2 | 0.120 | random primes in [16,255]; beats the `digit_transformer` baseline's 0.070 here |
| 3 | ~0.02 | ~6500 primes in range; not learnable to useful accuracy from sparse samples |
| 4-10 | ~0.01-0.02 | `p >= 10^5` → honest 0 fallback; scores come from a=0 / b=0 edge cases |

This marginally beats `digit_transformer` (0.121) overall, driven entirely by tier 2.

## What we learned (the honest ceiling)

A decisive A/B test (`exploration/_dlp_grokking_dev/experiment_ab.py`) trained the **same**
network twice, changing only the bottleneck:

- **A — additive** `z = e_a + e_b` (the DLP bias): tier-2 val 0.075
- **B — concat** `z = [e_a; e_b]` (generic learned interaction): tier-2 val 0.057

They land in the same place. So the additive DLP bottleneck is **not** what caps tier 2 — the
real wall is that learning modular multiplication that *generalises across many unseen random
primes* is intrinsically hard at this scale. Tier 1 groks perfectly because it is the exact
single-/few-prime regime the grokking papers solve; tier 2+ asks the network to generalise the
field operation across thousands of primes it has limited samples for, and it plateaus.

That ceiling is the expected, honest outcome. More compute would mostly buy overfitting to the
*public* seed (which would not transfer to the secret official seed), not real tier-2+
capability. We ship the genuine result rather than chase a gamed one.

## Status under the rules

Compliant:

- Per-argument preprocess hooks are pass-through identities — no cross-argument leakage.
- `predict_digits` reduces `a % p`, `b % p` (two operands at a time, allowed) and never computes
  the three-argument modular product itself; the network output materially determines the answer.
- No discrete-log table, generator search, or hand-coded modular arithmetic — the multiplication
  is in trained weights. Perturbing them degrades accuracy.
- Passes `modchallenge check` static analysis.
- Deterministic (`eval()` mode, no dropout, no sampling).
