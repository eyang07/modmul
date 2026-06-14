# Tier 3 experiment plan

**Goal:** push `highest_tier_above_90` from 2 → 3. Tier 3 primes live in `[2^9, 2^16) = [512, 65536)`
(~6000 primes); eval draws 5 of them at random (unseen seed) and reduces operands to residues in
`[0, p)`. Beating it means a model that genuinely *computes* `x·y mod p` for tier-3 primes, not one
that memorizes a table.

## Why tier 3 is a regime change

- **Memorization dies.** For `p ~ 65535` there are `p² ≈ 4·10⁹` residue pairs — can't enumerate per
  prime (tier 2 worked precisely because its ~48 primes had small enough tables to cover).
- So we need **generalization**, along two axes:
  - *within-prime*: learn the multiplication function for a prime from a **sample** of its pairs.
  - *cross-prime*: transfer to primes unseen in training.
- **Two routes to "coverage":**
  - **(A)** train on *all* ~6000 tier-3 primes (a sample of pairs each) → eval primes are seen →
    only need *within-prime* generalization. **More tractable; start here.**
  - **(B)** train on a subset → need *cross-prime* generalization (harder; this is what the
    `dlp_grokking` baseline failed at, scoring 1% on tier 3).

This is the make-or-break research bet: single-prime modular-arithmetic grokking is established in the
literature, but **multi-prime tier-3 grokking is genuinely open.**

## New ingredients (implemented + validated)

| Ingredient | Where | Status |
|---|---|---|
| Angular output head — circle encoding `(cos2πt/p, sin2πt/p)`, Saxena-Charton | `model.JointModMulNetAngular`, `--arch angular` | exact decode verified at p=65521 |
| Custom angular loss `α(r²+1/r²)+‖pred−tgt‖²`, α=1e-4 | `train.angular_loss` | smoke-tested |
| Grokking recipe — fixed finite train set + weight decay | `--fixed-per-prime N`, `--wd` | smoke-tested |
| Grokking metrics — train-fit / within-prime-unseen / cross-prime | `train.py` eval branch | smoke-tested |

Angular is the key bet: a `cls` head at tier 3 needs `p_max=65536` classes (16.7M-param head, slow),
while angular is 2 outputs and **scales** — the paper hit 99% at q≈10⁶.

## Experiments (ordered by leverage; E2 is the linchpin — run it first)

### E2 — Can we grok within-prime multiplication at tier-3 scale? (de-risk everything)
Cheap: 8 fixed tier-3 primes, fixed train set, sweep weight decay. Watch `within-prime-unseen` jump
(the grokking phase transition) while `train-fit`→1.0 early.
```
# pick 8 primes near the top of the range; sweep wd ∈ {0.1, 1.0, 3.0}, fixed-per-prime ∈ {1000, 5000}
.venv/bin/python -u training/train.py --arch angular --tiers 3 --fixed-per-prime 2000 \
    --wd 1.0 --steps 20000 --eval-every 500 --batch 1024 --tag e2_ang_wd1
```
**Decision:** if `within-prime-unseen` groks to ≥90% on a few primes → proceed to E3. If nothing
groks after a long run across the wd/size grid → tier 3 is likely beyond this approach; bank tier 2.

### E1 — Output representation head-to-head (run alongside E2)
Same fixed setup, compare `--arch angular` vs `cls --p-max 65536` vs `joint` (digit heads).
Hypothesis: angular wins on speed + final accuracy; cls is too heavy; digit heads coordinate poorly.

### E3 — Prime coverage scaling
Once a config groks on 8 primes, scale the prime pool 8 → 64 → 512 → all ~6000, holding `--fixed-per-prime`.
Track whether within-prime generalization survives more primes, and whether **cross-prime** (held-out,
via `--holdout`) starts to emerge — if it does, route (B) becomes viable and training gets cheaper.

### E4 — Curriculum + warm start
Warm-start from the tier-2 checkpoint; curriculum over prime size (small→large within tier 3) and over
operand magnitude. Expectation: faster grok, better cross-prime transfer.

### E5 — EBM verifier (last mile to 90%)
Once a config reaches ~80% within-prime, attach the margin-energy head on the shared encoder rep
(`ModMulNet.encode`-style hook) with hard negatives (`t±1`, near-residues, the model's own mistakes)
to push borderline residues over the 90% line. EBM only earns its keep here, not before.

## Metrics & success criteria
- **train-fit**: accuracy on the fixed train pairs (overfit signal; should hit ~1.0 early).
- **within-prime-unseen**: accuracy on *unseen pairs of training primes* — the grokking signal and the
  thing that decides tier 3 under route (A).
- **cross-prime**: accuracy on held-out primes (`--holdout`) — decides whether route (B) is possible.
- **Tier 3 cleared** = within-prime-unseen ≥ 90% on the full prime pool, confirmed end-to-end via
  `modchallenge evaluate` with `output_base="p"`.

## Practical notes
- **Compute is the bottleneck.** Grokking wants tens of thousands of steps; MPS is ~0.2–0.5 s/step.
  A single cloud GPU (A100/H100) is ~10–50× faster and likely necessary to sweep E2/E3 before the
  Aug 12 deadline. Decide early.
- Each run writes a unique checkpoint (`--tag`, else `arch_tNN`), so experiments don't collide.
- `WIDTH=5` already covers tier-3 values (<65536). Tier 4+ (p up to 2³²) will need a wider input
  encoding and rules out both `cls` and fine-angular — a later problem.

---

# Energy-based ideas (from Kona / IREM / IRED)

Kona ([Logical Intelligence](https://logicalintelligence.com/blog/energy-based-model-sudoku-demo),
open repro [Enso](https://github.com/MVPandey/Enso)) solves Sudoku at 96% via an energy landscape +
Langevin "thinking". **Honest scope:** that headline mechanism (gradient descent over the answer)
does *not* transfer here — Sudoku is a constraint-satisfaction problem with local, decomposable,
cheap-to-verify constraints, so its energy has a navigable landscape; `x·y mod p` has none (verifying
a candidate residue = recomputing the whole answer, so the landscape is a spike). EBMs win when
verification ≪ generation; for us verification ≈ generation. So we borrow the *parts that fit*, not
the Langevin search.

## E6 — Algebraic-consistency losses (Kona's real lever: the differentiable constraint loss)

Kona's power is its **constraint loss** (Sudoku uniqueness), not the sampling. Our analog: `x·y mod p`
obeys ring/group axioms, which are **differentiable self-consistency constraints on the model's own
outputs** — no ground-truth labels, so they apply to *unlimited unlabeled* `(x,y,p)` triples
(including held-out residue pairs) and push within-prime generalization (= grokking) directly.
**Training-time only; inference stays a single forward pass → fully compliant** (no hand-coded
arithmetic; these are regularizers on the model's outputs). This is the highest-value borrow and
the most direct "make it learn the algorithm" lever.

Let `f(x,y,p)` be the model's predicted answer. Add (weight λ each):

- **Commutativity** (self-supervised, label-free): `f(x,y,p) ≈ f(y,x,p)`. Penalize disagreement
  between the two forward passes — symmetric KL on `cls` logits, or `‖·‖²` on angular unit vectors.
  Applies to *any* `(x,y)`, so it regularizes the whole grid, not just sampled/labeled pairs.
- **Identity / absorbing** (cheap labeled anchors): `f(x,1,p)=x`, `f(x,0,p)=0` (+ symmetric). CE to
  the known residue.
- **Distributivity** (self-supervised; cleanest with `--arch angular`): `x·y + x·z ≡ x·(y+z) (mod p)`.
  With angular outputs `u(·) = e^{2πi f(·)/p}` (unit vectors), this is exactly the **complex-product
  identity** `u(x,y)·u(x,z) = u(x,(y+z) mod p)`, so
  `L_distrib = ‖cmul(u(x,y,p), u(x,z,p)) − u(x,(y+z)%p,p)‖²` over three forward passes. (The angular
  encoding turns a modular *additive* identity into a clean differentiable constraint — a strong
  reason to favour angular for tiers 3+.)

**Hypothesis:** commutativity + distributivity force *global* algebraic consistency (which is what a
real multiplication algorithm has), closing the train-fit → within-prime-unseen gap faster and
lifting cross-prime transfer.

**Build:** `--alg-consistency λ` in train.py; run the extra forward passes (commutativity: swap x,y;
distributivity: sample z, compute `(y+z)%p` in the *data pipeline*, never in the submission) and add
the penalties. **Measure** within-prime-unseen / cross-prime vs the no-consistency baseline at matched
steps on the E2 8-prime setup. **Decision:** if it materially lifts within-prime-unseen, make it
default and scale to E3.

## E7 — Energy re-ranking verifier (Kona's "pick the lowest-energy chain", minus Langevin)

This is E5 made concrete. An energy head `g(rep, c)` scores a candidate residue `c` against the
shared encoder representation `rep`; output `argmin_c g`.

- **Candidates:** the predictor's top-K residues (top-K `cls` logits; for angular, the K residues
  nearest the predicted angle).
- **Training:** margin/contrastive — `g(rep, true) + m ≤ g(rep, c)` for hard negatives
  `c ∈ {true±1, near-residues, the predictor's own top-K mistakes}`.
- **Inference:** deterministic argmin over K candidates (one extra small-head forward) — fits the
  ~273 ms/problem budget.

**When:** pure last-mile exactness — only pays off once the predictor is ~80–88% on a tier
(converts near-miss → ≥90%). Build right after a tier lands in that band. **Decision:** does
re-ranking lift exact-match over plain argmax near the 90% boundary?

## E8 — (frontier) Manufacture a CSP so iterative "thinking" actually applies

The only way to port Kona's iterative energy descent is to give the problem the *local constraint
structure* it natively lacks: reframe the computation over an intermediate representation with local
consistencies — a **digit/carry lattice** (learned carry-consistency potentials) or a **CRT/RNS
residue system** — and do unrolled energy minimisation (IREM: backprop through T refinement steps;
IRED: anneal + adapt step count to difficulty).

**Risk/compliance — high:** the local potentials must be *learned* (randomising weights must break
it); hand-coded carry/Barrett/CRT logic makes it a forbidden circuit. Seed any noise (determinism);
T steps × batch must fit the latency budget. **Park until tier 3 is in hand and E6/E7 are
exhausted** — this is the high-risk/high-reward novel-approach bet for tiers 4–5.

Reading: IREM ([2206.15448](https://arxiv.org/abs/2206.15448)),
IRED ([2406.11179](https://arxiv.org/abs/2406.11179)),
Energy-Based Transformers ([2507.02092](https://arxiv.org/pdf/2507.02092)).

## Priority order
E6 (algebraic-consistency) is the next build after the tier-2 / E2 numbers land — biggest expected
lift on generalisation, low risk, directly "learns the algorithm". E7 (verifier) follows the moment a
tier sits at 80–88%. E8 is a later frontier bet.
