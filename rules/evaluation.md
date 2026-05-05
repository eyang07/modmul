# Modular Arithmetic Challenge - Evaluation Setup

> **We want your feedback.** The evaluation plan below — including the evaluation hardware, time budget, problem-set size, and final scoring details — is still being refined, and items marked **TBD** will be decided based on community input. Please share suggestions on the [SAIR Foundation Zulip](https://zulip.sair.foundation/).

This page specifies how submissions to the Modular Arithmetic Challenge are evaluated: submission format, model interface, sandbox, evaluation pipeline, scoring, and prohibited practices.

For the high-level task description, key dates, eligibility, and team / anti-cheating policy, see **[overview.md](overview.md)**.

## Submission Format

A submission is a **single HuggingFace repository** identified by `repo_id` + an immutable `commit_hash`.

| File | Purpose |
|------|---------|
| `manifest.json` | Declares the entry class (e.g. `{"entry_class": "model.MyModel"}`) and any other submission metadata. Validated against a Pydantic schema. |
| `model.py` | The model implementation. Must define the entry class declared in `manifest.json`. |
| Weight files | The trained weights (`.pt`, `.pth`, `.safetensors`, `.bin`, `.gguf`, `.onnx`, etc.). |
| Other files | Any additional code, configuration, or assets needed at inference time. |

| Limit | Reference value |
|-------|-----------------|
| Total artifact size (weights + code + all files) | **20 GB** |
| Total inference time for the full evaluation set | **5 minutes** for 1100 problems (TBD; final value confirmed before competition opens) |

The repository may be private during development and must be made **public** before the official submission deadline. Private repositories are not ranked on the official leaderboard.

## Model Interface

The entry class must subclass `ModularMultiplicationModel` from `modchallenge.interface.base_model`:

```python
from modchallenge.interface.base_model import ModularMultiplicationModel

class MyModel(ModularMultiplicationModel):
    def load(self, model_dir: str) -> None:
        # Load weights, initialize the model. Called once before any predict call.
        ...

    def predict(self, a: str, b: str, p: str) -> str:
        # Return (a * b mod p) as a decimal string.
        ...

    def predict_batch(self, inputs: list[tuple[str, str, str]]) -> list[str]:
        # Optional. Override to enable GPU batching.
        ...

    def max_batch_size(self) -> int:
        return 64
```

Inputs are decimal strings of arbitrary length (`a >= 0`, `b >= 0`, `p >= 2` prime). The output must be a decimal string equal to `(a * b) mod p`.

Any architecture, tokenization, and preprocessing strategy is allowed, subject to the restrictions in **Prohibited Practices** below.

## Submission Workflow

**Phase 1 — Development (private):**

```
Train model locally
  -> Push to a private HuggingFace repo
  -> Create a fine-grained read-only access token for that repo
  -> Test:  modchallenge evaluate-hf <repo_id> <commit_hash> --token hf_xxx
  -> Iterate
```

Local-only testing is also supported, without a HuggingFace round trip:

```
modchallenge evaluate ./my-local-model --total 110
```

**Phase 2 — Final submission (public):**

1. Make the HuggingFace repo **public**.
2. Submit `repo_id` + `commit_hash` to organizers (submission portal: TBD).
3. Organizers verify the repo is public, accessible, and passes the manifest schema.
4. Organizers run official evaluation in the sandboxed environment with a **secret random seed**.
5. Results are posted to the leaderboard.

Multiple submissions per team are allowed; each submission is locked by its commit hash, and the leaderboard keeps the best result per team.

## Solver Environment

The submitted model runs in an isolated sandbox during evaluation:

- **No network access** — outbound and inbound networking is blocked at the sandbox boundary.
- **No secrets** — the HuggingFace token used to download the repository is not propagated to the model process; no other API keys or environment variables are exposed beyond a minimal allowlist (`PATH`, `HOME`, `LANG`, etc.).
- **Restricted filesystem** — read access is limited to the submission directory and standard system paths required by the runtime; write access is limited to a scratch directory. Reading other locations on the host is blocked.
- **No subprocess / no dynamic code execution** — the model may not spawn subprocesses, invoke `eval`/`exec`, or otherwise execute code outside the loaded module.
- **Deterministic seeding** — the pipeline sets a fixed RNG seed before each batch; the model is expected to be deterministic.

```
HuggingFace repo --(loader, manifest validation)--> Sandbox --(predict / predict_batch)--> Scorer
```

The sandbox is a Docker container at official evaluation time. Local testing via `modchallenge evaluate-hf` currently runs **without** the sandbox (trusted-use only); contestants are expected to test against the sandbox before final submission once the Docker image is published. Sandbox image and configuration: **TBD**.

## Evaluation Pipeline

The pipeline runs three steps for each submission:

1. **Test generation** — generate the test set for this run from the master seed (see below).
2. **Inference** — run the model over all problems, batched up to `max_batch_size()` where applicable.
3. **Determinism check** — re-run a sampled subset of problems and compare outputs; nondeterministic submissions are flagged and excluded from the ranked leaderboard.

The full pipeline is implemented in `src/modchallenge/evaluation/pipeline.py` and is identical between local testing and official evaluation.

## Test Generation

- **Master seed** — `secrets.token_bytes(32)` for the official run (secret, not disclosed); a fixed seed for the public benchmark and reproducible local testing.
- **Per-tier seeds** — derived via `HMAC-SHA256(master_seed, tier_id)` so that tiers are independent and reproducible from the master seed.
- **Primes per tier** — 5 different primes per scored tier, to prevent overfitting to a single modulus. Tier 0 uses pre-selected Mersenne primes for the higher sub-tiers (where `sympy.nextprime` is prohibitively slow).
- **Edge cases** — `a = 0`, `b = 0`, `a = 1`, `b = 1` are included in every scored tier.
- **Default size** — 1100 test cases (100 per tier).

Implementation: `src/modchallenge/testgen/generator.py`.

## Difficulty Tiers

Evaluation uses 11 tiers. Tier 0 is a **diagnostic** (pure multiplication, unscored). Tiers 1-10 are **scored**.

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

Each scored tier independently controls **both** the prime size and the operand size. Operands grow progressively across tiers, so each tier tests a distinct combination of multiplication scale and modular reduction complexity. Operands `a, b` may be much larger than `p`, requiring genuine modular reduction (not just memorization of `a * b mod p` for `a, b < p`).

Tier 0 covers the full operand range as **pure multiplication without modular reduction**. It is unscored and exists to diagnose whether a model fails at higher tiers due to multiplication capacity or due to modular reduction ability.

The single source of truth for tier geometry is `src/modchallenge/config.py`.

## Scoring

- **Primary metric:** `overall_accuracy` = average accuracy across Tiers 1-10, with **equal weight per tier**.
- **Secondary metric:** `highest_tier_above_90` = the highest tier index at which accuracy `>= 90%`.
- **Tier 0:** diagnostic only; not counted toward the score.
- **Incomplete tiers:** tiers that fail to complete (timeout, error, crash) score 0% for that tier.

**Leaderboard ranking:** first by `highest_tier_above_90` (descending), then by `overall_accuracy` as tiebreaker. Final scoring rules and tiebreakers are subject to community feedback before the competition officially opens.

## Determinism Requirement

Submissions must be **deterministic**: the same `(a, b, p)` input must produce the same output across runs.

The pipeline performs an automated determinism check on a sampled subset of problems by running them twice and comparing outputs. Submissions that fail the check are flagged and excluded from the ranked leaderboard.

If a model uses any source of randomness internally (sampling, dropout at inference, MCMC, etc.), it is the contestant's responsibility to seed it deterministically inside `load()` or `predict()`.

## Time and Resource Budget

| Resource | Reference value | Notes |
|----------|-----------------|-------|
| Total inference wall-clock | 5 minutes for 1100 problems | TBD; may be tuned. |
| Per-problem soft target | ~273 ms | Use `predict_batch()` and GPU batching to amortize. |
| Total artifact size | 20 GB | Weights + code + all files. |
| Memory | TBD | Will be set based on chosen evaluation hardware. |

A submission that exceeds the wall-clock budget for a tier scores 0% on that tier.

## Prohibited Practices

The following are **not allowed at inference time**:

- Symbolic-math libraries: `sympy`, `gmpy2`, `mpmath`, `flint`, etc.
- Using Python's built-in arbitrary-precision integer arithmetic to compute the answer directly (e.g. `str(int(a) * int(b) % int(p))`).
- `eval`, `exec`, `compile`, `__import__` of arbitrary modules, or any other dynamic code generation used to perform the computation.
- Network access of any kind.
- Reading files outside the submission directory (other than standard runtime libraries on `sys.path`).
- Spawning subprocesses or invoking system commands.
- Hard-coding test answers or fingerprints of the evaluation set into the model or code.

The model must **learn** to compute modular multiplication. It may not delegate the computation to an external library, an external service, or a hard-coded lookup table indexed by the evaluation inputs.

**Enforcement:**

- The official evaluation runs in a sandboxed environment that enforces network, filesystem, and subprocess restrictions.
- Submissions are subject to organizer review (manual reading of `model.py` and any auxiliary code).
- Planned additional checks: static analysis for prohibited imports, weight-perturbation tests for memorization detection, and timing analysis. Details: TBD.
- Any submission found to violate these rules is disqualified and removed from the leaderboard.

## Evaluation Hardware

**TBD.** The evaluation pipeline supports CPU, CUDA, and Apple MPS backends. The official evaluation hardware (CPU model, GPU model, memory, disk) and the corresponding per-tier wall-clock budget will be announced before the competition officially opens.

## Evaluation Problem Sets

- **Public benchmark** (in repository, `public_benchmark/`): 1100 test cases, fixed seed, answers included. Reproducible bit-for-bit. Use this for development and local testing.
- **Private evaluation set**: generated at official evaluation time from a secret master seed unknown to contestants. **Separate from the public benchmark.** Used to compute the official leaderboard score.

The public benchmark and the private evaluation set share the same tier geometry and the same generation algorithm; only the master seed differs.

## Official Repository

The official GitHub repository for the Modular Arithmetic Challenge:

- [https://github.com/SAIRcompetition/modular-arithmetic-challenge](https://github.com/SAIRcompetition/modular-arithmetic-challenge)

The repository includes:

- the evaluation pipeline (`src/modchallenge/evaluation/`)
- the test-case generator (`src/modchallenge/testgen/`)
- the model interface and submission schema (`src/modchallenge/interface/`)
- the leaderboard store (`src/modchallenge/leaderboard/`)
- the public benchmark (`public_benchmark/`)
- demo / baseline submissions (`examples/examples.json`)
- a step-by-step contestant tutorial (`examples/tutorial.md`)

## Local Testing

A typical local-testing workflow:

```bash
# One-time setup
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pip install torch                 # required for example models

# Run tests against your own model directory
modchallenge evaluate ./my-local-model --total 110

# Run against a HuggingFace submission (private repo + read-only token)
modchallenge evaluate-hf user/my-model <commit_hash> --token hf_xxx

# Quick-test any HuggingFace LLM (exploratory; not used for ranking)
modchallenge evaluate-llm Qwen/Qwen2.5-0.5B-Instruct

# Run all reference example models from examples/examples.json
modchallenge evaluate-example --group public --total 110

# Run the full pipeline test suite
pytest
```

A recommended workflow:

1. Implement and unit-test your model locally on a few problems per tier (`--total 11`).
2. Once your model passes the determinism check on a small sample, run the full public benchmark (`--total 1100`) and read off `overall_accuracy` and `highest_tier_above_90`.
3. Push to a private HuggingFace repo and re-test with `evaluate-hf` to verify the manifest, the loader path, and the artifact size.
4. Once stable, make the repo public and submit `repo_id` + `commit_hash`.

## Changes and Versioning

The evaluation pipeline (test generation, scoring, sandbox configuration) is versioned in the official repository. Material changes to the rules, the scoring formula, the tier geometry, or the prohibited-practices list will be announced through the official repository and the SAIR Foundation Zulip community before they take effect.
