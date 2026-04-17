# Modular Arithmetic Challenge

Can a neural network learn modular multiplication?

Given two large integers `a`, `b` and a prime `p`, compute `(a * b) mod p`. Operands can be hundreds of digits long — far beyond what fits in a 64-bit integer. Your model must learn to compute the answer without symbolic math libraries or built-in arbitrary-precision arithmetic.

See **[RULES.md](RULES.md)** for full competition rules, background, scoring, and submission workflow.

This repository contains the open-source evaluation system used for both local testing and official evaluation.

## Quick Start

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pip install torch  # required for example models

# Run example models from examples/examples.json (private models via HF token)
modchallenge evaluate-example --group private --total 110

# Evaluate a single HuggingFace submission
modchallenge evaluate-hf user/my-model <commit_hash> --token hf_xxx

# Quick-test any HuggingFace LLM (exploratory, not ranked)
modchallenge evaluate-llm Qwen/Qwen2.5-0.5B-Instruct

# Run tests
pytest
```

## Project Structure

```
src/modchallenge/
  interface/        Model ABC + manifest schema (what contestants implement)
  testgen/          Test case generation (CSPRNG, 11 difficulty tiers)
  evaluation/       Pipeline, scorer, LLM wrapper, model loader
  leaderboard/      JSON-based ranking store
  cli.py            CLI entry point

examples/
  examples.json       Model registry (public + private HF repos)
  tutorial.md         Step-by-step guide from setup to submission

RULES.md            Competition rules, background, and scoring
public_benchmark/   1100 test cases (fixed seed, answers included)
tests/              pytest test suite
```

## Difficulty Tiers

11 tiers. Tier 0 is a diagnostic (pure multiplication, unscored). Tiers 1-10 are scored.

Each tier independently controls both the prime size and the operand size. Operands grow progressively across tiers, so every tier tests a distinct combination of multiplication scale and modular reduction complexity.

| Tier | Prime p (bits) | Operands a, b (bits) | ~Decimal digits of a, b |
|------|---------------|----------------------|-------------------------|
| 0    | diagnostic    | 1-4096               | 1-1233 (10 sub-levels)  |
| 1    | 1-3           | up to 32             | up to 10                |
| 2    | 4-8           | up to 48             | up to 15                |
| 3    | 9-16          | up to 64             | up to 19                |
| 4    | 17-32         | up to 96             | up to 29                |
| 5    | 33-64         | up to 128            | up to 39                |
| 6    | 65-128        | up to 256            | up to 77                |
| 7    | 129-256       | up to 512            | up to 154               |
| 8    | 257-512       | up to 1024           | up to 309               |
| 9    | 513-1024      | up to 2048           | up to 617               |
| 10   | 1025-2048     | up to 4096           | up to 1233              |

Tier 0 covers the full operand range (10 sub-levels from 1-digit to 1233-digit) as pure multiplication without modular reduction. This allows diagnosing whether a model fails at high tiers due to multiplication capacity or modular reduction ability.

## CLI Commands

| Command | Description |
|---------|-------------|
| `modchallenge evaluate <dir>` | Evaluate a local submission directory |
| `modchallenge evaluate-hf <repo> <hash>` | Evaluate a HuggingFace submission (trusted-use only, no sandbox) |
| `modchallenge evaluate-example` | Run all models from examples/examples.json |
| `modchallenge evaluate-llm <model_id>` | Quick-test any HF LLM (exploratory, not ranked) |
| `modchallenge leaderboard` | Display the leaderboard |
| `modchallenge generate-public <dir>` | Generate the public benchmark test set |

## Test Generation

- Master seed: `secrets.token_bytes(32)` (secret) or fixed (public benchmark)
- Per-tier seeds derived via `HMAC-SHA256(master_seed, tier_id)`
- 5 different primes per tier to prevent overfitting
- Edge cases: a=0, b=0, a=1, b=1 in every tier
- Public benchmark uses a fixed seed so results are reproducible
