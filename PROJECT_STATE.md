# PROJECT STATE — SAIR Modular Arithmetic Challenge

> **Start-here file.** Point a fresh session at this. It holds the current standing,
> what's proven, where the artifacts are, the hard constraints, and the next move.
> Last updated: **2026-06-15.**

---

## The task

Learn `(a * b) mod p` with a neural model. The arithmetic must come from **trained
parameters**, not hand-coded math.

- **Primary metric:** `highest_tier_above_90` (htop90) — the highest tier where the
  model scores ≥90% accuracy.
- **Tiebreaker:** `overall_accuracy`.
- **Tier ladder** (`src/modchallenge/config.py`): 11 tiers; tier 0 is unscored
  multiplication-only, tiers 1–10 are scored. Bit-width of `p` doubles each tier.
  Tier 3 = `[2^9, 2^16)` (~5-digit primes); tier 4 = `[2^17, 2^32)`; tier 5+ keeps
  doubling. Operands are reduced mod p first, then multiplied.
- **Difficulty is ML-execution hardness**, not math hardness: the arithmetic is
  trivial poly-time, but exact long-chain algorithmic execution gets exponentially
  harder because end-to-end accuracy ≈ (per-token accuracy)^(chain length), and the
  chain length doubles per tier.

## Hard constraints (MUST hold — a static AST scanner checks every `.py` in the submission)

- **No Python `%`, `//`, division, Barrett, Montgomery, or CRT on the PRODUCT.** The
  product's modular reduction must be *learned*.
- The **only** blessed reduction is per-operand `int(a) % p` / `int(b) % p` (reducing
  the inputs before the model sees them is allowed).
- Submission must be **deterministic**.
- **No reinforcement learning. No hand-coded arithmetic.** (Standing instruction.)

---

## Current standing

| Tier | Status | Notes |
|---|---|---|
| 1 | ✅ trivial | fixed small primes {2,3,5,7} |
| **2** | ✅ **BANKED + SUBMITTED** | `cire77/ebm-modmul @15211043` — htop90=2 is locked |
| **3** | 🟢 **both halves proven, compose next** | see below — this is the live push |
| 4 | ⚪ stretch, one cheap data-driven swing later | frontier coin-flip |
| 5–10 | ⚪ out of scope | open / unsolved-ML territory |

**Baselines on the leaderboard sit at htop90=1.** Breaking tier 3 is a genuine result.

---

## Tier 3 — what's proven (the breakthrough)

Tier 3 = learn the **general multiply-then-reduce algorithm**, not a residue table.
(Residue-table memorization fails completely at tier 3 — p up to 65535 means sampled
pairs are ~0% coverage. That dead-end is documented in `memory/tier3-wall.md`.)

The algorithm is split into two scratchpad halves, **both now saturated at ~0.95+**
with the same training recipe:

1. **Multiply** (`training/tier0_probe.py`) — LSB-first digit transformer, abacus
   embeddings, autoregressive greedy decode.
   `BOS x_lsb MUL y_lsb EQ prod_lsb EOS`.
   **Result (2026-06-15):** exact@5d saturated **~0.96–0.97, peak 0.982** (d384/L8,
   80k steps). Broke the old 0.85 ceiling.

2. **Reduce** (`training/longdiv_probe.py`) — MSB-first **long-division** scratchpad
   with per-digit intermediate supervision: `r=0; for each digit d of N: r=r*10+d;
   q=r//p; r-=q*p`. (The `//` here is in the *data generator* that builds the
   supervision target — that is allowed; the *model* never calls `%`/`//`.)
   **Result (2026-06-15):** on the full tier-3 range `[8192, 65536)`, rem saturated
   **~0.95 (peak 0.960)** (d384/L8, 80k steps).

**Composition math (favorable):** composed accuracy ≈ P(mult) × P(reduce)
≈ 0.97 × 0.95 ≈ **0.92 — above the 0.90 bar with a small margin.**

### The recipe that unlocked both (KEY — reuse this)

The same training recipe saturated *both* halves. Depth bought nothing; the recipe
is the unlock:

- **LR schedule gates the grokking liftoff.** Cosine `T_max = steps`, so liftoff
  happens *during* annealing. Pick `steps` so the schedule anneals meaningfully over
  the run (we used 80k). Flattening the schedule (e.g. 120k) keeps LR near peak too
  long and the model stalls.
- **`tf_tok` (teacher-forced per-token accuracy) is the leading indicator.** The final
  answer is the tail of a ~50-token chain, so autoregressive accuracy ≈ tf_tok^50 —
  it stays ~0 until tf_tok crosses ~0.94. **Watch tf_tok during the rem=0 warmup**,
  not the end-to-end metric (which looks flat then suddenly lifts off — that's
  grokking, not divergence).
- **Stability:** LR warmup (linear 2000 steps via `SequentialLR`) + grad-norm clip 1.0.
  Without these, deep models (L12) diverge early (loss jumps, tf_tok collapses to
  chance). Both are defaults in the probes (`--warmup`, `--grad-clip`).
- **`--amp`** for bf16 autocast (~2× throughput on Ada/Ampere, e.g. 4090).

---

## Artifacts (on the Mac)

`training/checkpoints/`:

- **`tier0_mult_best.pt`** (133M) — multiply half. step 44k, exact@5d **0.982**.
  config: d_model 384, layers 8, max_len 28, abacus_max 14, max_digits 5.
- **`longdiv_t3_L8_best.pt`** (171M) — reduce half. step 78k, rem **0.96**.
  config: d384/L8, full tier-3 prime range.

Both verified loadable. (Older checkpoints in that dir — `e2_*`, `predictor.pt`,
`smoke_*` — are superseded EBM/residue-table experiments, not part of the tier-3 path.)

---

## NEXT STEP — compose multiply + reduce into one model (the htop90=3 path)

Build **one scratchpad model** that multiplies `a*b` to get the product `N`, then
long-divides `N` by `p` to get the remainder — end to end, all digits net-generated.

`BOS a MUL b MOD p EQ <product digits> SEP <long-division chain> EOS`

Plan:
1. Merge the two scratchpad formats into one sequence; supervise both regions.
2. Train on the full tier-3 range with the proven recipe (L8/d384, annealed cosine
   over the run length, warmup 2000, grad-clip 1.0, `--amp`). Watch `tf_tok`.
3. **Risk:** joint training must hold *both* halves simultaneously — composed acc
   ≈ P(mult) × P(reduce), so any regression in either half drops below the bar.
   If joint training fights itself, consider curriculum (multiply first, then add the
   division chain) or a two-pass / shared-trunk design.
4. Once the composed model clears ~0.90 on held-out tier-3 pairs, **integrate the
   autoregressive readout into `EBMModMul`** and produce the tier-3 submission.

**This is pure upside — htop90=2 is already banked.** Don't risk the tier-2 submission.

### Stretch (only after tier 3 is banked)
One cheap data-driven swing at tier 4: train/eval the composed model on tier-4-sized
data and read the chain-survival curve. Stop at tier 4 — tier 5+ is unsolved ML.

---

## Compute / ops

- **GPU:** RunPod pod (4090-class). Setup: `training/RUNPOD_SETUP.md`,
  fallback `training/COMPUTE_BACKUP_PLAN.md`.
- **JupyterLab terminals persist** on the pod independent of the browser — no tmux
  needed. A long run survives laptop sleep / browser disconnect.
- **File transfer off the pod:** `runpodctl send <path>` on the pod prints a code;
  `runpodctl receive <code>` on the Mac (installed via `brew install
  runpod/runpodctl/runpodctl`). This sidesteps dead upload hosts and the pod's stale
  CA bundle. **Always pull checkpoints off the pod** — network-volume data can be
  wiped, and a *stopped* pod hides `/workspace` from the file browser (restart to
  see it again; a stop is not a terminate).

## Map of the other docs

- `training/TIER3_STRATEGY.md` — the original locked strategy writeup.
- `training/TIER3_EXPERIMENTS.md` — experiment log / history.
- `training/RUNPOD_SETUP.md` — click-by-click pod bootstrap.
- `training/COMPUTE_BACKUP_PLAN.md` — alternate compute.
- `memory/` (auto-loaded each session) — `tier3-wall.md` has the full blow-by-blow of
  the tier-3 investigation, including every dead-end and the saturation runs.
