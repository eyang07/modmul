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
| Total inference time for the full evaluation set | **5 minutes** for 1100 problems (TBD; subject to community input) |

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

Inputs are decimal strings of arbitrary length (`a >= 0`, `b >= 0`, `p >= 2` prime). The output must be a canonical decimal string equal to `(a * b) mod p` (full output specification in **Output Format** below).

The decimal-string boundary is the **only** fixed contract. Inside `predict` (and `predict_batch`) you may use any internal representation — digit-level tokens, p-adic, CRT decomposition, other bases, custom embeddings, etc. Encoding the inputs into your representation and decoding the model's output back to a decimal string both happen inside your code. See **Prohibited Practices** below for the boundary between *re-encoding* (allowed) and *computing the answer outside the model* (prohibited).

Any architecture implementable within the supported sandbox runtime is allowed, subject to the restrictions in **Prohibited Practices** below.

## Output Format

The output of `predict()` (and each element of `predict_batch()`) must be a **canonical decimal string** representing `(a * b) mod p`:

- digits `0`-`9` only
- no leading zeros, except for the literal string `"0"` representing zero
- no whitespace, no signs, no separators, no scientific notation
- type must be `str` (not `None`, not `bytes`, not `int`)

Outputs that violate the canonical format are scored as **incorrect** for that problem. They do not raise an error and do not abort the run.

**Batch contract:** if `predict_batch(inputs)` returns a list whose length differs from `len(inputs)`, the entire tier is marked incomplete and scores **0%**. The pipeline does not attempt to re-align partial results.

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

Multiple submissions per team are allowed; each submission is locked by its commit hash. The leaderboard keeps the best result **per team** (not per repo); see [overview.md](overview.md#team-participation-and-anti-cheating-policy) for how team identity is bound to HuggingFace accounts at registration time.

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

The pipeline is implemented in `src/modchallenge/evaluation/pipeline.py` and is shared between local testing and official evaluation. Official evaluation additionally executes the model inside the sandbox described in **Solver Environment**; local testing via `modchallenge evaluate-hf` currently runs the model in the host process (trusted-use only). Apart from the sandbox boundary and the package allowlist that comes with it, the test-generation, inference loop, scoring, and determinism check are the same code path in both settings.

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

**Reported metrics** (computed for every submission):

- `overall_accuracy` — average accuracy across Tiers 1-10, with **equal weight per tier**.
- `highest_tier_above_90` — the highest tier index at which accuracy `>= 90%`.

**Leaderboard ranking keys** (sort order):

1. `highest_tier_above_90` (descending)
2. `overall_accuracy` (descending) as tiebreaker

**Tier 0** is diagnostic only and is not counted toward either metric.
**Incomplete tiers** (timeout, error, crash, batch-contract failure) score 0% for that tier.

Final scoring rules and tiebreakers are subject to community feedback.

## Determinism Requirement

Submissions must be **deterministic**: the same `(a, b, p)` input must produce the same output across runs.

The pipeline performs an automated determinism check on a sampled subset of problems by running them twice and comparing outputs. Submissions that fail the check are flagged and excluded from the ranked leaderboard.

If a model uses any source of randomness internally (sampling, dropout at inference, MCMC, etc.), it is the contestant's responsibility to seed it deterministically inside `load()` or `predict()`.

## Time and Resource Budget

| Resource | Reference value | Notes |
|----------|-----------------|-------|
| Total inference wall-clock | 5 minutes for 1100 problems | TBD; may be tuned. |
| Per-problem soft target | ~273 ms | Use `predict_batch()` and GPU batching to amortize. |
| `load()` budget | TBD | Bounded separately; not counted against the inference wall-clock. |
| Determinism check | TBD | Bounded separately; not counted against the inference wall-clock. |
| Total artifact size | 20 GB | Weights + code + all files. |
| Memory | TBD | Will be set based on chosen evaluation hardware. |

**Wall-clock measurement (mechanical policy):**

- The inference timer **starts** when the pipeline issues the first `predict_batch` call after `load()` completes.
- The timer **excludes** `load()` (separate bounded budget) and the determinism check (separate bounded budget on a small sample).
- Tiers run in order: Tier 0 first, then Tiers 1, 2, ..., 10.
- **On timeout:**
  - The tier currently being processed scores **0%**; partial results within that tier are discarded.
  - All subsequent tiers score **0%**.
  - Tiers fully completed before the timeout retain their actual scores.

## Prohibited Practices

The principle: **the model must learn to compute `(a * b) mod p`. It may not delegate, look up, or hard-code the computation.**

The following are **not allowed at inference time**:

- **Computing the final answer** `(a * b) mod p` using:
  - symbolic-math libraries: `sympy`, `gmpy2`, `mpmath`, `flint`, etc.
  - Python's built-in arbitrary-precision integer arithmetic on the full operands (e.g. `str(int(a) * int(b) % int(p))`)
- **Hard-coding** test answers, lookup tables indexed by the evaluation inputs, or fingerprints / hashes of the evaluation set into the model weights or code.
- Dynamic code execution: `eval`, `exec`, `compile`, `__import__` of arbitrary modules, `ctypes`, or any other mechanism used to load or execute computation outside the model.
- Network access of any kind.
- Reading files outside the submission directory (other than standard runtime libraries on `sys.path`).
- Spawning subprocesses or invoking system commands.

**Explicitly allowed: representation conversion.** Using `int()`, modular arithmetic on small intermediate quantities, base conversion, p-adic decomposition, CRT splitting, byte-level encoding, or any other standard transformation to **re-encode** inputs into your model's internal representation, or to **decode** the model's output back to a decimal string, is **not a violation**.

**Structural test for the boundary.** The issue is not how the code is spelled but where the answer comes from. If pre- or post-processing code combines information from `a`, `b`, **and** `p` to derive part of the final residue digits — whether by chunked / streaming multiplication, Karatsuba, CRT recombination of model outputs against `p`, or any other algorithm — it is treated as *computing the answer outside the model*, even if no expression of the form `int(a) * int(b) % int(p)` ever appears. The model's output must materially determine the answer digits; conversion code must be representational only, not computational.

Borderline examples (allowed):

- `int(a)` to parse the decimal string into a Python int as a step in computing its base-`q` digits — uses only `a`, no combination with `b` or `p`
- `int(p) % q` to set up CRT moduli — uses only `p`, no combination with `a` or `b`
- a base-`b` representation of `a` and `b` fed to the model, with the answer reconstructed digit-by-digit from the model's output — the model output materially determines the answer
- the model emits the answer as CRT residues (or any other representation), and the post-processor deterministically decodes those residues into the canonical decimal string — decoding consumes only the model output (and `p` for canonicalization), not `a` or `b`

Borderline examples (prohibited):

- Computing `int(a) * int(b) % int(p)` and then "encoding" it as the answer
- A neural model whose forward pass returns garbage but whose output post-processor computes the answer from `int(a)`, `int(b)`, `int(p)` regardless of the model output
- Splitting `a` and `b` into chunks, asking the model only for individual chunk products, and recombining them modulo `p` in the post-processor — the recombination is the modular multiplication, not the model
- The post-processor computes CRT residues itself from `int(a)`, `int(b)` against small prime factors and uses those residues — residues come from the algorithm, not the model
- Per-prime-factor CRT in the post-processor: model produces residues against small primes, post-processor recombines using `p` — recombination is the answer

The principle is symmetric: **decoding model-produced residues / digits / encodings into the answer is fine; computing residues or the final modular product from `a`, `b`, `p` outside the model is not.**

**Enforcement layers** (the design below is the target for the official evaluation; layers marked *planned* are under active development and will be published as the implementation matures, in time for the official evaluation runs):

1. **Sandbox package allowlist** *(planned)*. The evaluation sandbox is a published Docker image whose package list will be fixed and disclosed as soon as the image is finalized, so contestants can build and test against it during the competition window. The runtime initially centers on PyTorch; broader runtime support (JAX, TensorFlow, ONNX, etc.) may be added based on contestant demand. The image will not include `sympy`, `gmpy2`, `mpmath`, `flint`, or networking / subprocess libraries; `import` of any package not in the image fails at load time. Any further restrictions on the Python stdlib (if needed) will be listed in the published image spec.
2. **Static analysis** *(planned)*. Every submission's source is AST-scanned before evaluation. The scanner flags disallowed imports, calls to `eval` / `exec` / `compile` / `__import__` / `ctypes`, and obvious patterns of computing the modular product from `(a, b, p)` directly (e.g. `int(_) * int(_) % int(_)` and arithmetic equivalents). Submissions matching these patterns are rejected before the model is loaded. Static analysis is intentionally narrow — it catches the easy cases; subtler code paths (e.g. chunked multiplication, CRT recombination in post-processing) are caught by Layer 4.
3. **Behavioral signals for review** *(planned)*. Submissions may be subject to investigative checks that produce **signals for organizer review**, not automatic disqualification:
   - **Weight perturbation**: random small perturbation of the model weights; if accuracy is essentially unchanged, the model may not be using its weights to produce the answer. *Caveat:* this can miss code-based solvers and may false-flag fragile or aggressively quantized models.
   - **Distribution shift**: re-evaluation on a separately seeded test set drawn from the same distribution; large divergence from the official score is a possible indicator of over-fitting to a leaked seed. *Caveat:* legitimate models can also show non-trivial variance.
   - **Latency profile**: per-problem inference time vs operand size; an essentially flat curve across tiers is a possible indicator of a constant-time shortcut. *Caveat:* aggressive batching can flatten the curve for legitimate models.

   These checks inform the manual review in Layer 4 — they do not by themselves disqualify a submission.
4. **Manual code review.** Required for top leaderboard entries and any submission flagged by Layer 2 (static analysis) or Layer 3 (behavioral signals). The reviewer reads `model.py` and any auxiliary code, applying the **structural test** above to decide whether the submission is computing the answer in the model or outside it.

Any submission found to violate the rules is disqualified and removed from the leaderboard.

## Evaluation Hardware

**TBD.** The evaluation pipeline supports CPU, CUDA, and Apple MPS backends. The official evaluation hardware (CPU model, GPU model, memory, disk) and the corresponding per-tier wall-clock budget will be announced before the official evaluation runs.

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
