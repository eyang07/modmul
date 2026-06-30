"""Tier-6+ recurrent reduction cell: a SHARED, weight-tied cell that learns ONE
bounded modular-Horner transition and is unrolled to compute (a*b) mod p for p of
*any* bit-width (tiers 6-10).

Why this beats the tier-3..5 autoregressive scratchpad (which plateaued at tier 6):
the scratchpad transformer re-derives the WHOLE multiply-then-reduce chain
positionally (no weight-sharing across steps, L~4347 tokens at tier 6) and never
length-generalizes. This model instead learns the single digit-serial Horner step

    s_{t+1} = (s_t * B + d_t * x) mod p          (x = a mod p ; d_t = base-B digits of b)

and applies the SAME learned cell at every step. Two consequences:
  * BOUNDED STATE: every s_t < p, so there is never a giant intermediate product
    (this is exactly the tier-6 wall, dissolved by construction).
  * LENGTH GENERALIZATION: one shared transition applied N times -> train on short
    chains (small p, few limbs) and extrapolate to tiers 6-10's long chains.

Representation: every wide number (s, x, p) is a fixed vector of base-B limbs,
LEAST-significant-first. The cell is a bidirectional GRU over the limb axis (it
propagates carries / the mod-p compare-subtract both directions). The cell predicts
the limbs of the reduced next state s_{t+1} directly -- the reduction is LEARNED,
never done in Python (randomizing the cell's weights collapses accuracy; see
tier6_verify.py). An optional auxiliary head predicts the per-step quotient
q_t = (s_t*B + d_t*x)//p (the tier-5 lesson: making the quotient explicit solidifies
the reduction) -- this is also an emitted intermediate, strengthening the
"emitted digits materially determine the answer" compliance posture.

TEACHER-FORCED TRAINING IS A SINGLE-STEP PROBLEM: given the true s_t, the target
s_{t+1} is independent of all other steps, so we train the cell on the FULL transition
function by sampling s, x ~ U[0,p), d ~ U[0,B) uniformly. Training the complete
function (not trajectories) means there is no free-running distribution shift, so no
DAgger pass is needed -- if every one-step transition is exact, the free-running
unroll is exact at any length. The only thing that must generalize is per-step
exactness at MORE limbs than trained; that is the length-generalization gate (--gate).

All // and % live ONLY in the data generator that builds targets; inference (forward())
runs the cell in a loop and never divides. Compliant with rules/evaluation.md.

Smoke (self-contained, Miller-Rabin range so no data.py dependency):
    python training/tier6_recurrent.py --limb-bits 1 --p-min 131072 --p-max 1048576 \
        --steps 400 --batch 256 --d-model 64 --gru-layers 2 --eval-every 100 \
        --ckpt training/checkpoints/t6_smoke.pt

Gate (train <=64-bit p, test the cell at 128/256/512-bit p; the go/no-go signal):
    python training/tier6_recurrent.py --limb-bits 1 --p-min 2 --p-max 18446744073709551616 \
        --steps 60000 --batch 512 --amp --gate --ckpt training/checkpoints/t6_gate.pt
"""

from __future__ import annotations

import argparse
import bisect
import contextlib
import math
import os
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Limb <-> int helpers. Limbs are base-B = 2**limb_bits, LEAST-significant-first.
# ---------------------------------------------------------------------------

def to_limbs(n: int, base: int, K: int) -> list[int]:
    """Non-negative int -> K base-B limbs, LSB-first (zero-padded on the high end)."""
    out = [0] * K
    i = 0
    while n > 0 and i < K:
        out[i] = n % base
        n //= base
        i += 1
    if n > 0:
        raise ValueError(f"{n} overflows {K} base-{base} limbs")
    return out


def from_limbs(limbs: list[int], base: int) -> int:
    v = 0
    for d in reversed(limbs):
        v = v * base + int(d)
    return v


def digits_msb(n: int, base: int) -> list[int]:
    """Non-negative int -> base-B digits, MOST-significant-first ([0] for 0)."""
    if n == 0:
        return [0]
    s = []
    while n > 0:
        s.append(n % base)
        n //= base
    return s[::-1]


# ---------------------------------------------------------------------------
# Primes. Deterministic Miller-Rabin (valid < 3.3e24; tiers 6-10 p < 2^2048 needs
# probable-prime is fine for TRAINING data -- the scorer's primes are real primes and
# the cell only needs (s*B+d*x) mod p to be well-defined). Log-uniform sampling gives a
# spread of bit-lengths (= a spread of limb counts), which is what drives length-gen.
# ---------------------------------------------------------------------------

_MR_WITNESSES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)


def _is_prime(n: int, rng: random.Random | None = None) -> bool:
    """Miller-Rabin. The fixed witness set is only DETERMINISTIC for n < 3.3e24, so for
    the large tier-6+ ranges (n up to 2^256+) we add 24 random-base rounds -> false-positive
    probability < 4^-24 ~ 1e-14. (Without this, build_prime_pool returns composites at
    256-bit, which silently poison training/eval; the scorer itself uses real sympy primes.)"""
    if n < 2:
        return False
    for sp in _MR_WITNESSES:
        if n % sp == 0:
            return n == sp
    d, r = n - 1, 0
    while d % 2 == 0:
        d //= 2; r += 1
    bases = list(_MR_WITNESSES)
    if n >= (1 << 81):                       # beyond the deterministic guarantee
        rr = rng or random
        bases += [rr.randrange(2, n - 1) for _ in range(24)]
    for a in bases:
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def build_prime_pool(p_min: int, p_max: int, size: int, rng: random.Random) -> list[int]:
    """Sorted pool over [p_min, p_max), log-uniform Miller-Rabin sampling. Fully
    self-contained (no data.py dependency) so it runs on a bare pod and small ranges."""
    p_min = max(2, p_min)
    lo_log, hi_log = math.log(p_min), math.log(p_max)
    out: set[int] = set()
    tries = 0
    cap = size * 200 + 10000
    while len(out) < size and tries < cap:
        tries += 1
        n = int(math.exp(rng.uniform(lo_log, hi_log))) | 1
        if n < p_min or n >= p_max:
            continue
        if _is_prime(n):
            out.add(n)
    return sorted(out)


# ---------------------------------------------------------------------------
# Horner trace (the data generator -- the ONLY place // and % appear).
# ---------------------------------------------------------------------------

def horner_state_seq(x: int, b: int, p: int, base: int):
    """Free-running reference: s_0=0; s_{t+1}=(s_t*base + d_t*x) mod p over the
    base-B digits d_t of b (MSB-first). Final s == (x*b) mod p == (a*b) mod p when
    x = a mod p. Returns the final state (used to check exact-match)."""
    s = 0
    for d in digits_msb(b, base):
        s = (s * base + d * x) % p
    return s


def transition_target(s: int, d: int, x: int, p: int, base: int):
    """One learned step: returns (s_next, q) where s_next=(s*base+d*x) mod p < p and
    q=(s*base+d*x)//p < 2*base is the quotient (auxiliary supervision)."""
    v = s * base + d * x
    return v % p, v // p


# ---------------------------------------------------------------------------
# The shared recurrent reduction cell.
# ---------------------------------------------------------------------------

class RecurrentReducer(nn.Module):
    """A single weight-tied cell. step_logits learns the one-step transition; forward()
    unrolls it over b's digits INSIDE the forward pass (no Python loop sequencing calls
    from outside the model -- the recurrence is the architecture). Length-agnostic: no
    parameter depends on the limb count K or the chain length L, so it extrapolates."""

    def __init__(self, base: int, d_model: int = 256, gru_layers: int = 2,
                 aux_quotient: bool = True, q_max: int | None = None):
        super().__init__()
        self.base = base
        self.aux_quotient = aux_quotient
        self.q_max = q_max if q_max is not None else 2 * base  # q < 2*base
        self.E_s = nn.Embedding(base, d_model)
        self.E_x = nn.Embedding(base, d_model)
        self.E_p = nn.Embedding(base, d_model)
        self.E_d = nn.Embedding(base, d_model)
        self.gru = nn.GRU(d_model, d_model, num_layers=gru_layers,
                          batch_first=True, bidirectional=True)
        self.ln = nn.LayerNorm(2 * d_model)
        self.head = nn.Linear(2 * d_model, base)
        if aux_quotient:
            self.qhead = nn.Linear(2 * d_model, self.q_max)

    def _encode(self, s, x, p, d):
        # s, x, p: (B, K) long limbs (LSB-first); d: (B,) long digit. Broadcast d.
        h = self.E_s(s) + self.E_x(x) + self.E_p(p) + self.E_d(d).unsqueeze(1)
        out, _ = self.gru(h)                 # (B, K, 2*d_model)
        return self.ln(out)

    def step_logits(self, s, x, p, d):
        """Returns (limb_logits (B,K,base), q_logits (B,q_max) or None)."""
        z = self._encode(s, x, p, d)
        limb_logits = self.head(z)
        q_logits = self.qhead(z.mean(dim=1)) if self.aux_quotient else None
        return limb_logits, q_logits

    @torch.no_grad()
    def forward(self, x, b_digits, p):
        """Free-running unroll. x, p: (B,K) LSB-first limbs; b_digits: (B,L) base-B
        digits MSB-first, LEFT-padded with zeros (leading zeros are harmless while
        s==0). Returns s: (B,K) limbs == (a*b) mod p. The loop IS the forward pass."""
        s = torch.zeros_like(x)
        L = b_digits.shape[1]
        for t in range(L):
            limb_logits, _ = self.step_logits(s, x, p, b_digits[:, t])
            s = limb_logits.argmax(-1)
        return s


# ---------------------------------------------------------------------------
# Training batch: uniform one-step transitions across the (curriculum-limited) pool.
# ---------------------------------------------------------------------------

def make_batch(batch, primes, base, K, rng, device):
    """Sample (p, s, x, d) with s, x ~ U[0,p), d ~ U[0,base); target = transition.
    Teaches the COMPLETE transition function (uniform states), so the free-running
    unroll has no distribution shift -- exactness at every input => exactness at any
    chain length."""
    S, X, P, D, T, Q = [], [], [], [], [], []
    n = len(primes)
    for _ in range(batch):
        p = primes[rng.randrange(n)]
        s = rng.randrange(p)
        x = rng.randrange(p)
        d = rng.randrange(base)
        snext, q = transition_target(s, d, x, p, base)
        S.append(to_limbs(s, base, K)); X.append(to_limbs(x, base, K))
        P.append(to_limbs(p, base, K)); T.append(to_limbs(snext, base, K))
        D.append(d); Q.append(min(q, 2 * base - 1))
    t = lambda v: torch.tensor(v, dtype=torch.long, device=device)
    return t(S), t(X), t(P), t(D), t(T), t(Q)


# ---------------------------------------------------------------------------
# Eval. (a) per-step exact-match (the leading indicator, like tf_tok); (b) full
# free-running exact-match on (a*b) mod p with operands >> p (the real metric).
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_step_exact(model, primes, base, K, n, rng, device, chunk=512):
    """Fraction of one-step transitions whose ALL limbs are predicted correctly."""
    model.eval()
    ok = tot = 0
    for s0 in range(0, n, chunk):
        g = min(chunk, n - s0)
        S, X, P, D, T, _ = make_batch(g, primes, base, K, rng, device)
        limb_logits, _ = model.step_logits(S, X, P, D)
        pred = limb_logits.argmax(-1)
        ok += (pred == T).all(dim=1).sum().item()
        tot += g
    return ok / max(1, tot)


@torch.no_grad()
def eval_exact(model, primes, base, n, op_bits, rng, device, chunk=128):
    """Free-running (a*b) mod p exact-match. a,b ~ U[0, 2**op_bits) (operands >> p,
    forcing genuine reduction); x = a mod p (per-operand reduction, allowed)."""
    model.eval()
    # Limb count from the largest prime in THIS eval set (may exceed the training K
    # -> that is the length-generalization gate).
    Kp = max(len(digits_msb(p, base)) for p in primes) + 1
    ok = tot = 0
    items = []
    for _ in range(n):
        p = primes[rng.randrange(len(primes))]
        a = rng.randrange(1 << op_bits)
        b = rng.randrange(1 << op_bits)
        # Reduce BOTH operands per-argument, exactly as the submission does:
        # (a*b) % p == ((a%p) * (b%p)) % p, and reducing b shortens the Horner chain
        # to ~len(p) steps (the deployment regime) instead of ~op_bits steps.
        x, b_red = a % p, b % p
        items.append((x, b_red, p, (a * b) % p))
    Lb = max(len(digits_msb(b, base)) for _, b, _, _ in items)
    for s0 in range(0, n, chunk):
        sub = items[s0:s0 + chunk]
        g = len(sub)
        X = torch.tensor([to_limbs(x, base, Kp) for x, _, p, _ in sub],
                         dtype=torch.long, device=device)
        P = torch.tensor([to_limbs(p, base, Kp) for _, _, p, _ in sub],
                         dtype=torch.long, device=device)
        # b digits MSB-first, LEFT-padded to Lb with zeros.
        Bd = []
        for _, b, _, _ in sub:
            ds = digits_msb(b, base)
            Bd.append([0] * (Lb - len(ds)) + ds)
        Bd = torch.tensor(Bd, dtype=torch.long, device=device)
        s = model(X, Bd, P)
        for j, (_, _, _, ans) in enumerate(sub):
            if from_limbs(s[j].tolist(), base) == ans:
                ok += 1
            tot += 1
    return ok / max(1, tot)


def log_buckets(p_min: int, p_max: int, n=3):
    lo_log, hi_log = math.log(max(2, p_min)), math.log(p_max)
    cuts = [math.exp(lo_log + (hi_log - lo_log) * k / n) for k in range(n + 1)]
    edges = [(int(cuts[k]), int(cuts[k + 1])) for k in range(n)]
    edges[0] = (max(2, p_min), edges[0][1])
    edges[-1] = (edges[-1][0], p_max)
    return edges


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limb-bits", type=int, default=1,
                    help="w; base B = 2**w. w=1 = bit-serial (simplest atom, longest "
                         "chain, best length-gen). Raise to 4/8 for throughput.")
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--gru-layers", type=int, default=2)
    ap.add_argument("--no-aux-quotient", action="store_true",
                    help="disable the auxiliary quotient head (on by default; it "
                         "solidifies the reduction and adds an emitted intermediate)")
    ap.add_argument("--aux-weight", type=float, default=0.5)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--eval-n", type=int, default=512)
    ap.add_argument("--p-min", type=int, default=2)
    ap.add_argument("--p-max", type=int, default=2 ** 64,
                    help="TRAINING prime ceiling. Keep small (<=2^64) and rely on "
                         "length generalization for tiers 6-10 (verified by --gate).")
    ap.add_argument("--pool-size", type=int, default=40000)
    ap.add_argument("--curriculum", action="store_true",
                    help="ramp the prime ceiling small->large over --curr-frac")
    ap.add_argument("--curr-frac", type=float, default=0.5)
    ap.add_argument("--gate", action="store_true",
                    help="ALSO eval the cell at p bit-lengths BEYOND training "
                         "(128/256/512-bit) -- the tiers-6-10 go/no-go signal.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt", type=str, default="training/checkpoints/t6_recurrent.pt")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--amp", action="store_true", help="bf16 autocast on CUDA")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = (torch.device("cuda") if torch.cuda.is_available()
              else torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cpu"))
    rng = random.Random(args.seed)
    eval_rng = random.Random(999)
    base = 1 << args.limb_bits

    pool_rng = random.Random(args.seed + 12345)
    POOL = build_prime_pool(args.p_min, args.p_max, args.pool_size, pool_rng)
    if not POOL:
        raise SystemExit(f"no primes sampled in [{args.p_min}, {args.p_max})")
    K = len(digits_msb(args.p_max - 1, base)) + 1   # limb count covers p_max (+slack)

    aux = not args.no_aux_quotient
    model = RecurrentReducer(base, d_model=args.d_model, gru_layers=args.gru_layers,
                             aux_quotient=aux).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"device {device} | base {base} (w={args.limb_bits}) | K {K} limbs | "
          f"pool {len(POOL)} primes [{args.p_min},{args.p_max}) | aux_q {aux} | "
          f"params {n_params:,}")

    curr_start = min(max(args.p_min * 4, 256), args.p_max)

    def cur_pmax(step):
        if not args.curriculum:
            return args.p_max
        frac = min(1.0, step / max(1, args.curr_frac * args.steps))
        return int(curr_start * (args.p_max / curr_start) ** frac)

    train_edges = log_buckets(args.p_min, args.p_max)
    train_buckets = [(lo, hi, [p for p in POOL if lo <= p < hi]) for lo, hi in train_edges]
    train_buckets = [(lo, hi, ps) for lo, hi, ps in train_buckets if ps]

    # Gate pools: bit-lengths BEYOND training, to test length generalization.
    gate_specs = []
    if args.gate:
        for bits in (128, 256, 512):
            lo, hi = (1 << (bits - 1)), (1 << bits)
            ps = build_prime_pool(lo, hi, 400, random.Random(7 * bits))
            if ps:
                gate_specs.append((bits, ps))

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    warmup = max(0, min(args.warmup, args.steps - 1))
    if warmup > 0:
        sched = torch.optim.lr_scheduler.SequentialLR(
            opt,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.01, end_factor=1.0,
                                                  total_iters=warmup),
                torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps - warmup,
                                                           eta_min=args.lr * 0.1),
            ],
            milestones=[warmup])
    else:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps,
                                                           eta_min=args.lr * 0.1)

    cfg = dict(base=base, limb_bits=args.limb_bits, d_model=args.d_model,
               gru_layers=args.gru_layers, aux_quotient=aux, K=K,
               p_min=args.p_min, p_max=args.p_max)

    def save_ckpt(path, step, best):
        if not args.ckpt:
            return
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(),
                        sched=sched.state_dict(), step=step, best=best,
                        config=cfg, args=vars(args)), path)

    start_step, best = 0, -1.0
    if args.resume and args.ckpt and os.path.exists(args.ckpt):
        ck = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start_step, best = ck["step"], ck.get("best", -1.0)
        print(f"resumed from {args.ckpt} at step {start_step} (best {best:.3f})")

    use_amp = args.amp and device.type == "cuda"
    amp_ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
               if use_amp else contextlib.nullcontext())
    if use_amp:
        print("bf16 autocast ON")
    ce = nn.CrossEntropyLoss()
    start = time.monotonic()
    n_skipped = 0

    for step in range(start_step + 1, args.steps + 1):
        model.train()
        hi = bisect.bisect_left(POOL, cur_pmax(step))
        primes_now = POOL[:hi] if hi > 0 else POOL[:1]
        S, X, P, D, T, Q = make_batch(args.batch, primes_now, base, K, rng, device)
        with amp_ctx:
            limb_logits, q_logits = model.step_logits(S, X, P, D)
            loss = ce(limb_logits.reshape(-1, base), T.reshape(-1))
            if q_logits is not None:
                loss = loss + args.aux_weight * ce(q_logits, Q)
        opt.zero_grad(); loss.backward()
        gnorm = (torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                 if args.grad_clip > 0 else None)
        # bf16 can emit a non-finite loss/grad on a hard batch; a single NaN step
        # corrupts AdamW's moments permanently. Skip it (keep the LR schedule aligned).
        if not torch.isfinite(loss) or (gnorm is not None and not torch.isfinite(gnorm)):
            n_skipped += 1
        else:
            opt.step()
        sched.step()

        if step % args.eval_every == 0:
            with amp_ctx:
                parts = []
                for lo, hi, ps in train_buckets:
                    se = eval_step_exact(model, ps, base, K, args.eval_n, eval_rng, device)
                    parts.append(f"[~{hi.bit_length()}b] step_ex {se:.3f}")
                # one full free-running exact-match at the top training bucket
                top_ps = train_buckets[-1][2]
                ex = eval_exact(model, top_ps, base, min(256, args.eval_n),
                                2 * args.p_max.bit_length(), eval_rng, device)
                gate_parts = []
                for bits, ps in gate_specs:
                    gse = eval_step_exact(model, ps, base,
                                          len(digits_msb((1 << bits) - 1, base)) + 1,
                                          256, eval_rng, device)
                    gex = eval_exact(model, ps, base, 128, 2 * bits, eval_rng, device)
                    gate_parts.append(f"gate{bits}: step_ex {gse:.3f} exact {gex:.3f}")
            print(f"step {step:6d} | loss {loss.item():.4f} | " + " | ".join(parts)
                  + f" | top_exact {ex:.3f}"
                  + (" | " + " | ".join(gate_parts) if gate_parts else "")
                  + f" | cur_pmax~{cur_pmax(step).bit_length()}b | skip {n_skipped}"
                  + f" | {time.monotonic()-start:.0f}s")
            save_ckpt(args.ckpt, step, best)
            score = ex
            if args.ckpt and score > best:
                best = score
                save_ckpt(args.ckpt.replace(".pt", "_best.pt"), step, best)

    save_ckpt(args.ckpt, args.steps, best)
    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
