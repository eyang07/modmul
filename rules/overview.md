# Modular Arithmetic Challenge

## What is This Competition?

The Modular Arithmetic Challenge asks a simple question: **can a neural network learn to do modular multiplication?**

Given two large integers `a`, `b` and a prime `p`, compute `(a * b) mod p`. All values are decimal strings. The operands can be hundreds of digits long — far beyond what fits in a 64-bit integer.

This sounds trivial for a calculator. For a neural network, it is an open problem.

## Organizers

Alberto Alfarano, François Charton, Yongzheng Jia, Kristin Lauter, Cathy Li, Terence Tao, Emily Wenger

## Background

Modular arithmetic is the foundation of modern cryptography, number theory, and large parts of algebra. Humans learn it through algorithms — long multiplication, then division for the remainder. But neural networks don't execute algorithms. They learn patterns from data and generalize.

Recent work has shown that small Transformers can learn modular addition (`a + b mod p`) for small primes, and even discover internal representations resembling Fourier analysis on cyclic groups. But multiplication is fundamentally harder: it requires carrying, multi-digit coordination, and the output length scales with the input length. Modular reduction on top of that adds another layer of difficulty.

**The key questions this competition explores:**

- Can neural networks learn exact multi-digit multiplication, not just approximate it?
- Can they generalize to operands much larger than the prime — requiring genuine modular reduction, not just memorization?
- What architectures, tokenizations, and training strategies work best?
- Where does the scaling break? At 10 digits? 100? 1000?

The difficulty tiers are designed to probe exactly this: at which point does your model stop being able to compute the correct answer?

A non-comprehensive bibliography of related prior work is available in [literature.md](literature.md).

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

This is a pure mathematical reasoning challenge. The decimal-string input/output is the official contract. **Internally your model may use any representation** — digit-level tokens, p-adic, CRT decomposition, other bases, custom embeddings — as long as the function from decimal `(a, b, p)` to decimal answer is performed by your model, not delegated to an external library. See **Prohibited Practices** below for the precise boundary.

For the breakdown of difficulty tiers, the test-generation procedure, and the scoring rules used by the official evaluation, see [evaluation.md](evaluation.md).

## Timeline

| Event | Date |
|-------|------|
| Competition opens | May 12, 2026 |
| Submission deadline | September 30, 2026, 23:59 AoE |

## Model Requirements

- Submissions must implement the `ModularMultiplicationModel` Python interface. The `predict()` boundary takes the canonical decimal strings `(a, b, p)` and returns the canonical decimal answer; encoding and decoding inside the model is your choice.
- Submissions must be deterministic.
- Any architecture **implementable within the supported sandbox runtime** is allowed: Transformer, RNN, CNN, hybrid, or novel approaches. The runtime contract (initially centered on PyTorch; broader runtime support such as JAX, TensorFlow, ONNX may be added based on contestant demand) is documented in [evaluation.md](evaluation.md).
- Any internal representation of inputs and outputs is allowed: digit-level tokens, p-adic, CRT decomposition, other bases, learned embeddings, etc.

The interface signature, required file layout, artifact size limit, output format, and inference time budget are specified in [evaluation.md](evaluation.md).

## Submission Workflow

A submission is a **public HuggingFace repository** containing your model, identified by `repo_id` + an immutable `commit_hash`.

1. Push your model (`manifest.json` + `model.py` + weight files) to a public HuggingFace repo.
2. Submit `repo_id` + `commit_hash` to organizers.
3. Organizers verify the repo is accessible and run the official evaluation with a secret random seed.
4. Results are posted to the leaderboard.

### Submission Rules

- The HuggingFace repo must be **public** at submission time — private repos are not ranked.
- The commit hash locks the submission version immutably.
- Multiple submissions are allowed; the leaderboard keeps your best result per team.
- Official evaluation uses a secret random seed unknown to contestants.
- The test-generation, scoring, and model-interface code is open source and shared by both local testing and official evaluation. Official evaluation additionally runs inside the sandboxed environment described in [evaluation.md](evaluation.md), so the two setups differ in sandbox enforcement and in the available package surface — contestants should test their submission against the published sandbox image before final submission.

For local testing during development (including the option of using a private HF repo with a read-only token) and full evaluation details, see [evaluation.md](evaluation.md).

## Prohibited Practices

The prohibitions below all share one principle: **the model must learn to compute `(a * b) mod p`; it may not delegate, look up, or hard-code the computation.**

The following are **not allowed at inference time**:

- **Computing the final answer** `(a * b) mod p` using symbolic-math libraries (`sympy`, `gmpy2`, `mpmath`, `flint`, etc.) or Python's built-in arbitrary-precision integer arithmetic (e.g. `int(a) * int(b) % int(p)`)
- **Hard-coded answers** or lookup tables indexed by the evaluation inputs (or by hashes/fingerprints of them)
- Dynamic code execution: `eval`, `exec`, `compile`, `__import__`, `ctypes`
- Network access of any kind
- Reading files outside the submission directory
- Subprocess calls or system commands

**What is explicitly allowed:** using `int()`, base conversion, modular arithmetic on small intermediate values, or other standard operations to **convert representations** (decimal ↔ p-adic, decimal ↔ CRT components, decimal ↔ other bases, etc.) is fine. Pre-/post-processing, tokenization, and decoding all fall on the allowed side.

**Structural test for the boundary.** The issue is not how the code is spelled but where the answer comes from. If pre- or post-processing code combines information from `a`, `b`, **and** `p` to derive part of the final residue digits — whether by chunked / streaming multiplication, Karatsuba, CRT recombination of model outputs against `p`, or any other algorithm — it is treated as *computing the answer outside the model*, even if no expression of the form `int(a) * int(b) % int(p)` ever appears. The model's output must materially determine the answer digits; conversion code must be representational only, not computational.

Submissions found to violate these rules will be disqualified. Sandbox configuration, static-analysis checks, and the planned behavioral signals (weight perturbation, distribution shift, latency profile) are documented in [evaluation.md](evaluation.md).

## Team Participation and Anti-Cheating Policy

**Team registration:**

- Each individual or organization can participate in only one team.
- Teams must register members and sponsors with organizers in advance (registration portal: TBD).
- At registration, a team declares one or more HuggingFace accounts or organizations as their submission identities.

**Team identity binding:**

- A submission is attributed to a team if its `repo_id` is owned by one of that team's declared HuggingFace accounts or organizations.
- A team may submit from multiple repositories across any of their declared accounts.
- Leaderboard deduplication is **per team**, not per repo: the leaderboard keeps each team's best result across all their submissions, regardless of source repo or account.
- Repo ownership is verified against the team's declared accounts at submission time.

**Anti-cheating:**

- If coordinated cheating is detected — including sockpuppet teams, weight laundering across accounts, or hard-coding of evaluation answers — all related teams will be disqualified.

## SAIR Contributor Network

This competition is integrated with the [SAIR Contributor Network](https://competition.sair.foundation/contributor-network).

During the competition, you can share your model through the Contributor Network:

1. Make your HF repo public (or create a separate public repo for sharing)
2. Submit `repo_id` + `commit_hash` via the Contributor Network
3. Organizers will evaluate your shared model and publish the results
4. Community members can review and give feedback on your work
5. All contributions are recognized and cited in the competition report

Sharing is **optional** but encouraged. It does not replace the final submission — you still need to submit before the deadline to be ranked on the official leaderboard.

## Community Feedback

Rules, scoring details, and evaluation procedures are still being refined and will be shaped by community input. Community contributions are welcome.

Join the SAIR Foundation Zulip community for discussion and collaboration:

- [https://zulip.sair.foundation/](https://zulip.sair.foundation/)

## Experimental Status

This challenge is currently in an experimental phase. Rules, scoring details, and evaluation procedures may be adjusted based on implementation experience and community feedback. Items marked **TBD** in this document are open questions where we explicitly welcome community input — see **Community Feedback** above for where to weigh in.
