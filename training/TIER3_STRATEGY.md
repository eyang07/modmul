# Tier 3 strategy (locked decision)

**Aim:** clear tier 3 (`highest_tier_above_90 = 3`). Tier 2 is the banked floor; tier 4 is
upside only. See [TIER3_EXPERIMENTS.md](TIER3_EXPERIMENTS.md) for the experiment mechanics and
the earlier E2–E8 notes.

## Why the tier-2 playbook is dead at tier 3
Tier 2 was won by **coverage** (memorise small residue tables). Tier 3 primes reach 65535, so a
table has `p² ≈ 4e9` entries — sampling 2000/prime is ~0% coverage. The E2 probe confirmed it:
**flat at zero, the model couldn't even fit its training set.** The only route is learning the
**general algorithm** — multiply then reduce — that generalises across all residues and primes.
After the free per-operand reduction, the real task is small: `x·y mod p` with `x,y,p < 2¹⁶`
(5-digit × 5-digit → 10-digit product, reduce mod a 5-digit prime).

## The decision: looped abacus digit-transformer that learns multiply-then-reduce, regularised by E6

Combination (each piece is literature-backed; the synthesis + E6 is ours):

1. **Digit-level I/O + abacus embeddings** — per-digit position-relative embeddings
   ([McLeish 2024](https://arxiv.org/abs/2405.17399): 99% on 100-digit addition from 20-digit
   training; "unlocks multiplication"). Reversed-digit output for carry direction.
2. **Scratchpad decomposition** — emit the intermediate product `x·y`, then the reduction `mod p`,
   then the answer (Nye et al.). The reduction is **learned**, never Python `%` (the compliance line).
3. **Looped / recurrent core** (input injection + recurrence) — iterative compute to *execute* the
   algorithm. This is the practical, proven form of "thinking", and where the EBM/Kona idea
   genuinely fits (multiplication IS iterative, unlike tier-2 residue lookup).
4. **Infinite fresh, shaped data** across ALL tier-3 primes/residues (learn the algorithm, not a
   table) with a **tier-0 → tier-3 curriculum** (digit length small → large). Distribution shaping
   matters most ([Saxena–Charton](https://arxiv.org/abs/2410.03569)); consider reciprocal-operand
   augmentation, which produced grokking across moduli in
   [modular exponentiation](https://arxiv.org/abs/2506.23679).
5. **E6 algebraic-consistency regularisation (our differentiator)** — self-supervised
   commutativity/distributivity on the whole grid. No cited paper does this; it's free
   structure-signal and our original contribution.

**Why this can edge out:** prior work does multiplication *or* modular arithmetic in isolation. We
combine a looped abacus multiplier + a learned modular reducer + E6 self-consistency, aimed at the
multiply-and-reduce structure of tier 3.

## Alternatives held in reserve
- **Two-net decomposition:** a learned multiplier (tier-0 task) feeding a learned reducer. Easier to
  train/debug independently; same compliance care on the reducer (must be learned, not `%`).
- **EBM iterative refinement** instead of a plain loop — the energy "thinking" variant; a research
  flourish only after the plain looped model works.

## Compliance guardrails
- Per-operand reduction `int(a)%p` is allowed; the **product's** reduction mod p must come from
  trained parameters (no `%`, no `pow(_,_,_)`, no bignum on the product).
- Scratchpad/intermediate digits are fine — they're the model's own generation; we still return only
  the answer digits via `predict_digits`. Determinism: greedy decode, seed any noise.

## Go/no-go gate (keeps the bet bounded)
**Tier-0 multiplication probe first.** Tier 0 is pure multiplication, unscored — the free diagnostic.
The make-or-break question: *can an abacus + scratchpad model learn 5-digit × 5-digit multiplication
that generalises?*
- **Yes →** add the learned reducer, push to tier 3 (curriculum + E6 + looping).
- **No →** tier 3 is out of reach; bank tier 2, write tier 3 up as future work. Stop early.

## Next concrete build
Specify and run the **Tier-0 multiplication probe**: a digit/abacus model + scratchpad data format,
reusing the existing trainer/data infra. Bounded effort with a hard early kill-switch.
