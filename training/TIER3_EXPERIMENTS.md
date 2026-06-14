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
