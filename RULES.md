# Modular Arithmetic Challenge

## What is This Competition?

The Modular Arithmetic Challenge asks a simple question: **can a neural network learn to do modular multiplication?**

Given two large integers `a`, `b` and a prime `p`, compute `(a * b) mod p`. All values are decimal strings. The operands can be hundreds of digits long — far beyond what fits in a 64-bit integer.

This sounds trivial for a calculator. For a neural network, it is an open problem.

## Background

Modular arithmetic is the foundation of modern cryptography, number theory, and large parts of algebra. Humans learn it through algorithms — long multiplication, then division for the remainder. But neural networks don't execute algorithms. They learn patterns from data and generalize.

Recent work has shown that small Transformers can learn modular addition (`a + b mod p`) for small primes, and even discover internal representations resembling Fourier analysis on cyclic groups. But multiplication is fundamentally harder: it requires carrying, multi-digit coordination, and the output length scales with the input length. Modular reduction on top of that adds another layer of difficulty.

**The key questions this competition explores:**

- Can neural networks learn exact multi-digit multiplication, not just approximate it?
- Can they generalize to operands much larger than the prime — requiring genuine modular reduction, not just memorization?
- What architectures, tokenizations, and training strategies work best?
- Where does the scaling break? At 10 digits? 100? 1000?

The difficulty tiers are designed to probe exactly this: at which point does your model stop being able to compute the correct answer?

## What We Expect to Learn

This is as much a research challenge as it is a competition. We expect:

- **Baselines from existing LLMs** will fail beyond small numbers, since they were not trained for exact arithmetic at scale.
- **Custom small models** with the right tokenization and training data may significantly outperform much larger general-purpose models on this task.
- **Novel approaches** — p-adic representations, CRT decomposition, number-theoretic tokenizations — could unlock capabilities that standard digit-level approaches cannot.
- **A clear picture of the frontier**: at what operand size and prime size do current neural approaches break down?

The results, techniques, and insights from all participants will be compiled and shared through the [SAIR Contributor Network](https://competition.sair.foundation/contributor-network). All contributions are recognized and cited.

## The Task

Your model receives three decimal strings `a`, `b`, `p` and must return `(a * b) mod p` as a decimal string.

**Input:**
- `a >= 0`, `b >= 0` (can be up to ~1233 decimal digits)
- `p >= 2` is prime (up to ~617 decimal digits)
- `a` and `b` can be much larger than `p`

**Output:** A decimal string equal to `(a * b) mod p`

**Example:**
```
a = "123456789", b = "987654321", p = "97"
answer = "52"   # because (123456789 * 987654321) % 97 == 52
```

This is a pure mathematical reasoning challenge. No internet access, no external APIs, no symbolic math libraries at inference time. Your model must learn to compute the answer.

## Task Specification

**Input:** Three decimal strings `a`, `b`, `p` where:
- `a >= 0`, `b >= 0` (can be up to ~1233 decimal digits)
- `p >= 2` is prime (up to ~617 decimal digits)
- `a` and `b` can be much larger than `p`

**Output:** A decimal string equal to `(a * b) mod p`

**Example:**
```
a = "123456789", b = "987654321", p = "97"
answer = "52"   # because (123456789 * 987654321) % 97 == 52
```

## Difficulty Tiers

Evaluation uses 11 tiers. Tier 0 is a diagnostic (pure multiplication, unscored). Tiers 1-10 are scored.

| Tier | Prime p (bits) | Operands a, b (bits) | ~Decimal digits of a, b |
|------|---------------|----------------------|-------------------------|
| 0    | diagnostic    | 1-4096               | 1-1233 (unscored)       |
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

Each tier uses 5 different primes to prevent overfitting to a single modulus. Edge cases (a=0, b=0, a=1, b=1) are included in every tier.

Tier 0 tests pure multiplication (no modular reduction) across the full operand range. It helps diagnose whether your model fails at high tiers due to multiplication capacity or modular reduction ability.

## Scoring

- **Primary metric:** `overall_accuracy` = average accuracy across Tiers 1-10 (equal weight per tier)
- **Secondary metric:** `highest_tier_above_90` = highest tier where accuracy >= 90%
- Tier 0 is diagnostic only and does not count toward the score
- Incomplete tiers (timeout, errors) score 0%
- Default evaluation: 1100 problems (100 per tier)

Leaderboard ranking: first by `highest_tier_above_90` (descending), then by `overall_accuracy` as tiebreaker.

## Model Requirements

- Your model must implement the `ModularMultiplicationModel` interface
- Maximum artifact size: **20 GB** (weights + code + all files)
- Total inference time limit: **5 minutes** for 1100 problems
- Model must be **deterministic** (same input produces the same output across runs)
- Any architecture is allowed: Transformer, RNN, CNN, hybrid, etc.
- Any tokenization strategy is allowed: digit-level, BPE, p-adic, CRT decomposition, etc.

### Model Interface

```python
from modchallenge.interface.base_model import ModularMultiplicationModel

class MyModel(ModularMultiplicationModel):
    def load(self, model_dir: str) -> None:
        # Load weights, initialize model
        ...

    def predict(self, a: str, b: str, p: str) -> str:
        # Return (a * b mod p) as a decimal string
        ...

    def predict_batch(self, inputs: list[tuple[str, str, str]]) -> list[str]:
        # Optional: override for GPU batching
        ...

    def max_batch_size(self) -> int:
        return 64
```

### Submission Files

Your HuggingFace repo must contain:

```
manifest.json     # {"entry_class": "model.MyModel"}
model.py          # Your model implementation
weights.pt        # Your trained weights (or any model files)
```

## Submission Workflow

### Phase 1: Development (Private)

During development, keep your model in a **private** HuggingFace repo to prevent leaking weights and code.

```
Train model locally
  -> Upload to private HF repo
  -> Create a fine-grained read-only access token
  -> Test: modchallenge evaluate-hf user/my-model <commit_hash> --token hf_xxx
  -> Iterate: update model, push new version, re-test
```

You can also test purely locally:
```
modchallenge evaluate ./my-local-model --total 110
```

### Phase 2: Final Submission (Public)

Before the deadline, **make your HF repo public** (or create a new public repo).

1. Set your HF repo to **public**
2. Submit `repo_id` + `commit_hash` to organizers
3. Organizers verify the repo is public and accessible
4. Organizers run official evaluation with a **secret random seed**
5. Results posted to leaderboard

### Submission Rules

- Final submission must be a **public** HuggingFace repo — private repos are not ranked
- Commit hash locks the submission version immutably
- Multiple submissions are allowed; leaderboard keeps your best result
- Official evaluation uses a secret random seed unknown to contestants
- The evaluation logic (test generation, scoring, tier definitions) is open source and identical between local testing and official evaluation. Official evaluation will additionally run in a sandboxed environment

## Prohibited Practices

The following are **not allowed** at inference time:

- Using symbolic math libraries (`sympy`, `gmpy2`, `mpmath`, etc.)
- Using Python's built-in arbitrary-precision arithmetic to compute the answer directly (`int(a) * int(b) % int(p)`)
- Using `eval()`, `exec()`, or dynamic code generation to perform computation
- Network access of any kind
- Reading files outside your submission directory
- Subprocess calls or system commands

Your model must **learn** to compute modular multiplication, not hard-code or delegate the computation.

**Enforcement:**
- Official evaluation runs in a sandboxed environment with no network access and restricted system calls
- Submissions are subject to organizer review; any submission found to violate these rules will be disqualified
- Nondeterministic submissions are excluded from the ranked leaderboard

## SAIR Contributor Network

This competition is integrated with the [SAIR Contributor Network](https://competition.sair.foundation/contributor-network).

During the competition, you can share your model through the Contributor Network:

1. Make your HF repo public (or create a separate public repo for sharing)
2. Submit `repo_id` + `commit_hash` via the Contributor Network
3. Organizers will evaluate your shared model and publish the results
4. Community members can review and give feedback on your work
5. All contributions are recognized and cited in the competition report

Sharing is **optional** but encouraged. It does not replace the final submission — you still need to submit before the deadline to be ranked on the official leaderboard.

## Timeline

| Event | Date |
|-------|------|
| Competition opens | TBD |
| Contributor Network sharing opens | TBD |
| Submission deadline | TBD |
| Final evaluation and results | TBD |

## FAQ

**Q: Can I use a fine-tuned LLM?**
A: Yes. Any model architecture is allowed, including fine-tuned LLMs, custom Transformers, or entirely novel architectures. The model just needs to implement the `ModularMultiplicationModel` interface.

**Q: Can I use pre-training data that includes modular arithmetic?**
A: Yes. How you train your model is entirely up to you. The restriction is on inference-time behavior only.

**Q: Why are operands larger than the prime?**
A: This tests whether your model can handle large multiplication followed by modular reduction. A model that only memorizes `a * b mod p` for small `a, b < p` will fail when operands grow beyond the prime.

**Q: What hardware is used for official evaluation?**
A: TBD. The evaluation pipeline supports CPU, CUDA, and Apple MPS. Official eval hardware will be announced before the competition opens.

**Q: Can I submit multiple times?**
A: Yes. Each submission is identified by a unique commit hash. The leaderboard keeps your best result.

**Q: What if my model times out?**
A: Tiers that don't complete within the time limit score 0%. The timeout is 5 minutes total for 1100 problems. Optimize your inference speed, use batching, and consider `predict_batch()` with GPU acceleration.
