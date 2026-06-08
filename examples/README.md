# Reference example models

Three small, **compliant** reference models that run end-to-end through the official
pipeline (`manifest validation → static check → load → preprocess isolation → determinism →
inference → decode → score`). They exist to (a) smoke-test a fresh setup, (b) give a floor and a
couple of honest baselines for the leaderboard, and (c) serve as worked examples of the one rule
that matters: **the answer must come from trained parameters, not hand-coded arithmetic**
(see [../rules/evaluation.md](../rules/evaluation.md#prohibited-practices)).

Each model directory holds `manifest.json`, `model.py`, a `README.md`, and (for the two neural
models) the trained `weights.pt` — they ship in-repo so the examples are **clone-and-run**, no
separate download or training step required. `always_zero` needs no weights.

| Model | What it is | overall_accuracy (1100) | Compliant |
|---|---|---|---|
| [`always_zero/`](always_zero/README.md) | Emits `[0]` for everything. Pipeline smoke test + leaderboard floor. | 0.079 | trivially |
| [`digit_transformer/`](digit_transformer/README.md) | ~544K-param decoder-only Transformer; reduces operands `mod p` inside `predict_digits`, learns small-number mod-multiplication. | 0.121 | yes |
| [`dlp_grokking/`](dlp_grokking/README.md) | ~6M-param discrete-log "grokking" model: shared residue encoder + **additive** (log-space) bottleneck + learned decoder. | **0.127** | yes |

All three are deterministic and pass `modchallenge check`. Tier 1 (the 4 fixed primes
`{2,3,5,7}`) is where the neural models grok cleanly; tier 2+ is the honest ceiling — see each
README for the per-tier breakdown and why.

## Run them locally (no HuggingFace token needed)

```bash
pip install -e ".[dev]"
pip install torch

# Static compliance check
modchallenge check examples/dlp_grokking

# Evaluate through the full pipeline (10 problems/tier)
modchallenge evaluate examples/always_zero       --total 110
modchallenge evaluate examples/digit_transformer --total 110
modchallenge evaluate examples/dlp_grokking      --total 110
```

The weights ship in-repo, so the commands above run as-is. To **reproduce** them from scratch:
`dlp_grokking` has a self-contained, sympy-free trainer in its own directory; `digit_transformer`'s
trainer needs `sympy` for prime generation (blocklisted inside a submission, so it lives in the
gitignored `exploration/`):

```bash
.venv312/bin/python examples/dlp_grokking/train.py --minutes 8
.venv312/bin/python examples/exploration/train_digit_transformer.py --steps 6000   # local dev only
```

To reproduce the **public benchmark** numbers in the tables (100 problems/tier, public seed):

```bash
.venv312/bin/python examples/dlp_grokking/eval_tiers.py examples/dlp_grokking
```

## Building your own

These are minimal, honest baselines — not the ceiling. The end-to-end workflow for writing,
testing, and submitting your own model (interface, manifest, local test, HuggingFace upload) is in
[tutorial.md](tutorial.md). The `dlp_grokking` README is the most detailed worked example of
turning a mathematical insight into a compliant inductive bias.

## Not in this folder

Red-team probes and dev tooling live under `examples/exploration/` (gitignored). They are
deliberately **non-compliant** stress tests for the static check and reviewer process, kept out of
the distributed examples so nobody mistakes them for a sanctioned approach.
