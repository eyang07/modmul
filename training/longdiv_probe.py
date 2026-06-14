"""Long-division reduction probe: can a transformer learn N mod p the RIGHT way?

The tier-3 probes proved multiplication is learnable but the modular REDUCTION
(product mod p) is a hard wall: emitting the remainder LSB-first in one shot stays
flat at chance even on small primes with a perfect product in hand. Reason: division
is inherently MSB-first (subtract multiples of p from the top, carrying a running
remainder), so a one-shot LSB remainder has no learnable local structure.

This probe ISOLATES reduction and gives it the known-correct framing: an explicit
schoolbook LONG-DIVISION scratchpad. We feed N and p directly and supervise the full
chain of (quotient-digit, running-remainder) pairs, MSB-first across N's digits:

    r = 0
    for each digit d of N (most significant first):
        r = r*10 + d
        q = r // p            # one quotient digit, 0..9
        r = r - q*p           # new running remainder, < p
    answer = final r = N mod p

Each step is a tiny bounded computation (r < p, d < 10) with intermediate
supervision — the recipe that makes division tractable for transformers (Lee et al.,
"Teaching Arithmetic"; Nye et al. scratchpads). Everything is net-generated, so it
stays submission-compliant (no Python % on the product at inference).

Sequence:
    BOS N_msb MOD p_msb EQ  q1 : r1_msb STEP q2 : r2_msb STEP ... qk : rk_msb  EOS
Loss is on the whole scratchpad (everything after EQ).

Eval: greedy-decode from "...EQ", take the digits after the LAST ':' as the final
remainder, compare to N % p. Bucketed by prime size.

We feed N = a*b for a,b ~ U[0,p) (the real product distribution) so this transfers to
the tier-3 task once composed with the (already learnable) multiplication stage.

Usage:
    python training/longdiv_probe.py --p-max 2048 --steps 40000 --eval-every 2000
"""

from __future__ import annotations

import argparse
import bisect
import random
import time

import torch
import torch.nn as nn

from data import primes_for_tier

# Vocab: 0-9 digits, then specials.
PAD, BOS, MOD, EQ, COLON, STEP, EOS = 10, 11, 12, 13, 14, 15, 16
VOCAB = 17
SPECIALS = {PAD, BOS, MOD, EQ, COLON, STEP, EOS}


def digits_msb(n: int) -> list[int]:
    """Non-negative int -> decimal digits, most-significant-first ([0] for 0)."""
    if n == 0:
        return [0]
    s = []
    while n > 0:
        s.append(n % 10)
        n //= 10
    return s[::-1]


def msb_to_int(ds: list[int]) -> int:
    v = 0
    for d in ds:
        v = v * 10 + d
    return v


def long_division_steps(N: int, p: int):
    """The (quotient-digit, running-remainder) chain; final remainder == N % p."""
    r, steps = 0, []
    for d in digits_msb(N):
        r = r * 10 + d
        q = r // p          # single digit in [0, 9] since prev r < p
        r = r - q * p       # running remainder, < p
        steps.append((q, r))
    return steps


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Example construction
# ---------------------------------------------------------------------------

def build_example(N: int, p: int, max_len: int):
    """Return (tokens, abacus_idx, is_output) padded to max_len.

    Abacus index = position within the current number/segment (resets after every
    special token). The whole scratchpad after EQ is supervised.
    """
    Nd, pd = digits_msb(N), digits_msb(p)
    toks = [BOS] + Nd + [MOD] + pd + [EQ]
    abac = [0] + list(range(len(Nd))) + [0] + list(range(len(pd))) + [0]
    is_out = [False] + [False] * len(Nd) + [False] + [False] * len(pd) + [False]
    for i, (q, r) in enumerate(long_division_steps(N, p)):
        if i > 0:
            toks.append(STEP); abac.append(0); is_out.append(True)
        rd = digits_msb(r)
        toks.append(q); abac.append(0); is_out.append(True)            # quotient digit
        toks.append(COLON); abac.append(0); is_out.append(True)
        toks += rd; abac += list(range(len(rd))); is_out += [True] * len(rd)  # remainder
    toks.append(EOS); abac.append(0); is_out.append(True)
    pad = max_len - len(toks)
    if pad < 0:
        raise ValueError(f"max_len {max_len} too small for sequence of {len(toks)}")
    toks += [PAD] * pad
    abac += [0] * pad
    is_out += [False] * pad
    return toks, abac, is_out


def make_batch(batch: int, primes: list[int], max_len: int, rng, device):
    """N = a*b for a,b ~ U[0,p): the real tier-3 product distribution."""
    T, A, M = [], [], []
    for _ in range(batch):
        p = primes[rng.randrange(len(primes))]
        a, b = rng.randrange(p), rng.randrange(p)
        t, ab, m = build_example(a * b, p, max_len)
        T.append(t); A.append(ab); M.append(m)
    tt = lambda v, dt: torch.tensor(v, dtype=dt, device=device)
    return tt(T, torch.long), tt(A, torch.long), tt(M, torch.bool)


# ---------------------------------------------------------------------------
# Model (abacus decoder, same as the other probes)
# ---------------------------------------------------------------------------

class AbacusDecoder(nn.Module):
    def __init__(self, max_len: int, abacus_max: int, d_model=384, nhead=8,
                 num_layers=8, dim_ff=1536):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.abacus_emb = nn.Embedding(abacus_max, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=0.0, batch_first=True, activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, VOCAB, bias=False)
        self.max_len = max_len
        self.register_buffer("pos_ids", torch.arange(max_len), persistent=False)

    def forward(self, toks, abacus):
        b, t = toks.shape
        x = self.tok_emb(toks) + self.pos_emb(self.pos_ids[:t]) + self.abacus_emb(abacus)
        mask = torch.triu(torch.full((t, t), float("-inf"), device=toks.device), diagonal=1)
        x = self.transformer(x, mask=mask, is_causal=True)
        return self.head(self.ln(x))


# ---------------------------------------------------------------------------
# Eval: greedy-decode the scratchpad, exact-match on the final remainder
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_remainder(model, primes, n, max_len, abacus_max, rng, device):
    """Returns final-remainder exact-match over n freshly sampled products."""
    model.eval()
    ok = 0
    for _ in range(n):
        p = primes[rng.randrange(len(primes))]
        a, b = rng.randrange(p), rng.randrange(p)
        N = a * b
        true_rem = N % p
        toks = [BOS] + digits_msb(N) + [MOD] + digits_msb(p) + [EQ]
        abac = ([0] + list(range(len(digits_msb(N)))) + [0]
                + list(range(len(digits_msb(p)))) + [0])
        gen, seg = [], 0
        while len(toks) < max_len:
            tt = torch.tensor([toks], dtype=torch.long, device=device)
            aa = torch.tensor([abac], dtype=torch.long, device=device)
            nxt = int(model(tt, aa)[0, -1].argmax())
            if nxt == EOS:
                break
            toks.append(nxt); gen.append(nxt)
            if nxt in SPECIALS:
                abac.append(0); seg = 0
            else:
                abac.append(min(seg, abacus_max - 1)); seg += 1
        # Final remainder = digits after the LAST colon.
        if COLON in gen:
            j = len(gen) - 1 - gen[::-1].index(COLON)
            ans = [d for d in gen[j + 1:] if d < 10]
            if ans and msb_to_int(ans) == true_rem:
                ok += 1
    return ok / n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d-model", type=int, default=384)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--nhead", type=int, default=8)
    ap.add_argument("--dim-ff", type=int, default=1536)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--eval-n", type=int, default=300)
    ap.add_argument("--p-min", type=int, default=512)
    ap.add_argument("--p-max", type=int, default=2048,
                    help="start small (match the range where the one-shot probe failed)")
    ap.add_argument("--curriculum", action="store_true",
                    help="ramp the prime ceiling small->large over --curr-frac of training")
    ap.add_argument("--curr-frac", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device()
    rng = random.Random(args.seed)
    eval_rng = random.Random(999)

    POOL = [p for p in primes_for_tier(3) if args.p_min <= p < args.p_max]
    if not POOL:
        raise SystemExit(f"no primes in [{args.p_min}, {args.p_max})")

    # Sequence sizing: N up to (p_max-1)^2; running remainders < p_max.
    nd = len(str((args.p_max - 1) ** 2))           # max digits in the product N
    pd = len(str(args.p_max - 1))                  # max digits in a prime / remainder
    abacus_max = nd + 2
    # worst case: N + MOD + p + EQ + nd*(q + ':' + remainder + STEP) + EOS
    max_len = (nd + 1 + pd + 1) + nd * (pd + 3) + 2 + 8

    curr_start = min(max(args.p_min * 4, 1024), args.p_max)

    def cur_pmax(step: int) -> int:
        if not args.curriculum:
            return args.p_max
        frac = min(1.0, step / max(1, args.curr_frac * args.steps))
        return int(curr_start * (args.p_max / curr_start) ** frac)

    edges = [(512, 2048), (2048, 8192), (8192, 65536)]
    buckets = []
    for lo, hi in edges:
        lo2, hi2 = max(lo, args.p_min), min(hi, args.p_max)
        if lo2 < hi2:
            ps = [p for p in POOL if lo2 <= p < hi2]
            if ps:
                buckets.append((lo2, hi2, ps))

    model = AbacusDecoder(max_len, abacus_max, d_model=args.d_model, nhead=args.nhead,
                          num_layers=args.layers, dim_ff=args.dim_ff).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"device {device} | pool {len(POOL)} primes [{args.p_min},{args.p_max}) | "
          f"max_len {max_len} | abacus_max {abacus_max} | curriculum {args.curriculum} | "
          f"params {n_params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=args.lr * 0.1)
    loss_fn = nn.CrossEntropyLoss()
    start = time.monotonic()

    for step in range(1, args.steps + 1):
        model.train()
        hi = bisect.bisect_left(POOL, cur_pmax(step))
        primes_now = POOL[:hi] if hi > 0 else POOL[:1]
        toks, abac, mask = make_batch(args.batch, primes_now, max_len, rng, device)
        logits = model(toks, abac)
        logits = logits[:, :-1].reshape(-1, VOCAB)
        target = toks[:, 1:].reshape(-1)
        m = mask[:, 1:].reshape(-1)
        loss = loss_fn(logits[m], target[m])
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()

        if step % args.eval_every == 0:
            parts = []
            for lo, hi, ps in buckets:
                rem = eval_remainder(model, ps, args.eval_n, max_len,
                                     abacus_max, eval_rng, device)
                parts.append(f"[{lo}-{hi}) rem {rem:.3f}")
            print(f"step {step:6d} | loss {loss.item():.4f} | " + " | ".join(parts)
                  + f" | cur_pmax {cur_pmax(step)} | {time.monotonic()-start:.0f}s")

    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
