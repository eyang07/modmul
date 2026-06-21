"""Tier-5 modular-multiply scratchpad, base-parametrized: learn (x*y) mod p for
p in [2^33, 2^64) at a DENSER numeric base so the Horner chain stays tier-4-short.

This is modmul_probe.py (the proven tier-3/tier-4 recipe) generalized from hard-coded
base 10 to an arbitrary base B via --base. Probe A (tier5_pp_probe.py) settled the base:
base-100 GROKS the per-step partial product pp = x*d (widest-x bucket ~0.89 and still
climbing, tf_tok 0.988), base-1000 STALLS (one-shot 3x3-decimal multiply hits the wall).
So tier 5 runs at base 100, where:

  * p < 2^64 is ~10 base-100 digits  -> ~10 Horner blocks  -> max_len ~800
    (IDENTICAL chain length/width to base-10 tier-4 -- tier 5 becomes tier-4 difficulty
    in a wider base, instead of the ~1700-token base-10 chain that busts the 300s budget).
  * the only genuinely new sub-skill vs tier 4 is the single base-B digit multiply
    inside pp = x*d: (<=2-decimal)x(<=2-decimal), under the 4x4 learnable edge.

Algorithm (process y MSB-first; result stays < p between digits), q in 0..B-1:

    result = 0
    for each base-B digit d of y:
        s  = result * B                 # shift the accumulator   (< B*p)
        q1 = s // p ; m1 = q1*p ; r1 = s - m1     # shift-reduce (q1 single base-B digit)
        pp = x * d                                # single base-B-digit multiply  (< B*p)
        t  = r1 + pp                              # explicit add (< B*p)
        q2 = t // p ; m2 = q2*p ; r2 = t - m2     # add-reduce  (q2 single base-B digit)
        result = r2
    answer = result = (x * y) mod p

Every intermediate (incl. explicit m1=q1*p, t=r1+pp, m2=q2*p) is written down and
supervised -- the lesson from tier 3/4: any hidden intermediate caps end-to-end acc.
All // and % live ONLY in this data generator that builds the target; the model emits
digit tokens and never divides, so inference stays submission-compliant.

Sequence (all numbers base-B, MSB-first; fixed colon-separated fields per y-digit):
    BOS x MUL y MOD p EQ  d:q1:m1:r1:pp:t:q2:m2:r2 STEP ... EOS
The final answer is r2 of the last block = digits after the LAST colon (format-agnostic
parse, same as the tier-3/4 decoder).

Usage (on a CUDA pod, after Probe A picked base 100):
    python training/tier5_modmul.py --base 100 --amp --steps 80000 --batch 192 \
        --p-min 8589934592 --p-max 18446744073709551616 \
        --curriculum --ckpt training/checkpoints/modmul_t5_b100.pt > t5_b100.log 2>&1 &
  (p-min = 2^33, p-max = 2^64)
"""

from __future__ import annotations

import argparse
import bisect
import contextlib
import math
import os
import random
import time
from collections import defaultdict

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Base-B tokenization. Digits 0..B-1 are tokens 0..B-1; specials follow.
# ---------------------------------------------------------------------------

def make_vocab(base: int):
    PAD, BOS, MUL, MOD, EQ, COLON, STEP, EOS = (base, base + 1, base + 2, base + 3,
                                                base + 4, base + 5, base + 6, base + 7)
    vocab = base + 8
    specials = {PAD, BOS, MUL, MOD, EQ, COLON, STEP, EOS}
    return dict(PAD=PAD, BOS=BOS, MUL=MUL, MOD=MOD, EQ=EQ, COLON=COLON, STEP=STEP,
                EOS=EOS, VOCAB=vocab, SPECIALS=specials)


def digits_base(n: int, base: int) -> list[int]:
    """Non-negative int -> base-B digits, most-significant-first ([0] for 0)."""
    if n == 0:
        return [0]
    s = []
    while n > 0:
        s.append(n % base)
        n //= base
    return s[::-1]


def base_to_int(ds: list[int], base: int) -> int:
    v = 0
    for d in ds:
        v = v * base + d
    return v


# ---------------------------------------------------------------------------
# Prime pool: tier 5 [2^33, 2^64) is non-enumerable, so log-uniform Miller-Rabin
# sampling (deterministic witnesses are valid < 3.3e24; 2^64 ~ 1.8e19 is well under).
# Log-uniform gives a spread of chain lengths rather than ~all max-width.
# ---------------------------------------------------------------------------
SIEVE_LIMIT = 65536


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    for sp in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % sp == 0:
            return n == sp
    d, r = n - 1, 0
    while d % 2 == 0:
        d //= 2; r += 1
    for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):  # deterministic < 3.3e24
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
    """Sorted prime pool over [p_min, p_max), log-uniform Miller-Rabin sampling."""
    if p_max <= SIEVE_LIMIT:
        from data import primes_for_tier
        return sorted(p for p in primes_for_tier(3) if p_min <= p < p_max)
    lo_log, hi_log = math.log(p_min), math.log(p_max)
    out: set[int] = set()
    while len(out) < size:
        n = int(math.exp(rng.uniform(lo_log, hi_log))) | 1
        if n < p_min or n >= p_max:
            continue
        while not _is_prime(n):
            n += 2
            if n >= p_max:
                break
        else:
            out.add(n)
    return sorted(out)


def modmul_rows(x: int, y: int, p: int, base: int):
    """Per-y-digit (d, q1, m1, r1, pp, t, q2, m2, r2) chain; final r2 == (x*y) % p.
    q1, q2 are single base-B digits (0..B-1); m1,pp,m2 < B*p; t < B*p."""
    result, rows = 0, []
    for d in digits_base(y, base):
        s = result * base
        q1 = s // p; m1 = q1 * p; r1 = s - m1   # shift-reduce: q*p explicit, then subtract
        pp = x * d                              # single base-B-digit multiply
        t = r1 + pp                             # explicit add (supervised)
        q2 = t // p; m2 = q2 * p; r2 = t - m2   # add-reduce: q*p explicit, then subtract
        rows.append((d, q1, m1, r1, pp, t, q2, m2, r2))
        result = r2
    return rows


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Example construction. Abacus index = place within the current number (0 at MSB),
# reset to 0 after every special token. Everything after EQ is supervised.
# ---------------------------------------------------------------------------

def build_example(x: int, y: int, p: int, base: int, V: dict, max_len: int):
    xd, yd, pd = digits_base(x, base), digits_base(y, base), digits_base(p, base)
    toks = [V["BOS"]] + xd + [V["MUL"]] + yd + [V["MOD"]] + pd + [V["EQ"]]
    abac = ([0] + list(range(len(xd))) + [0] + list(range(len(yd)))
            + [0] + list(range(len(pd))) + [0])
    is_out = [False] * len(toks)

    def emit(tok, ab):
        toks.append(tok); abac.append(ab); is_out.append(True)

    def emit_num(n):
        for i, d in enumerate(digits_base(n, base)):
            emit(d, i)

    def emit_digit(v):          # single 0..B-1 value
        emit(v, 0)

    for i, (d, q1, m1, r1, pp, t, q2, m2, r2) in enumerate(modmul_rows(x, y, p, base)):
        if i > 0:
            emit(V["STEP"], 0)
        emit_digit(d)
        emit(V["COLON"], 0); emit_digit(q1)
        emit(V["COLON"], 0); emit_num(m1)
        emit(V["COLON"], 0); emit_num(r1)
        emit(V["COLON"], 0); emit_num(pp)
        emit(V["COLON"], 0); emit_num(t)
        emit(V["COLON"], 0); emit_digit(q2)
        emit(V["COLON"], 0); emit_num(m2)
        emit(V["COLON"], 0); emit_num(r2)
    emit(V["EOS"], 0)

    pad = max_len - len(toks)
    if pad < 0:
        raise ValueError(f"max_len {max_len} too small for sequence of {len(toks)}")
    toks += [V["PAD"]] * pad
    abac += [0] * pad
    is_out += [False] * pad
    return toks, abac, is_out


def make_batch(batch, primes, base, V, max_len, rng, device, wcum=None):
    """x, y ~ U[0, p) for p sampled from the pool. If ``wcum`` (cumulative weights
    aligned to ``primes``) is given, p is drawn weighted by it -- weight p^alpha makes
    the training prime distribution value-uniform (alpha=1) to match the scorer, which
    draws nextprime(randrange(2^33,2^64)) and so lands ~50% at 64-bit. Without weights,
    p is uniform over the pool (= uniform in bits, which under-samples the high end)."""
    T, A, M = [], [], []
    n = len(primes)
    for _ in range(batch):
        if wcum is not None:
            r = rng.uniform(0.0, wcum[n - 1])
            p = primes[bisect.bisect_left(wcum, r, 0, n)]
        else:
            p = primes[rng.randrange(n)]
        x, y = rng.randrange(p), rng.randrange(p)
        t, ab, m = build_example(x, y, p, base, V, max_len)
        T.append(t); A.append(ab); M.append(m)
    tt = lambda v, dt: torch.tensor(v, dtype=dt, device=device)
    return tt(T, torch.long), tt(A, torch.long), tt(M, torch.bool)


# ---------------------------------------------------------------------------
# Model (abacus decoder, same backbone; vocab depends on base).
# ---------------------------------------------------------------------------

class AbacusDecoder(nn.Module):
    def __init__(self, vocab, max_len, abacus_max, d_model=384, nhead=8,
                 num_layers=8, dim_ff=1536):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.abacus_emb = nn.Embedding(abacus_max, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=0.0, batch_first=True, activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.max_len = max_len
        self.register_buffer("pos_ids", torch.arange(max_len), persistent=False)

    def forward(self, toks, abacus):
        b, t = toks.shape
        x = self.tok_emb(toks) + self.pos_emb(self.pos_ids[:t]) + self.abacus_emb(abacus)
        mask = torch.triu(torch.full((t, t), float("-inf"), device=toks.device), diagonal=1)
        x = self.transformer(x, mask=mask, is_causal=True)
        return self.head(self.ln(x))


# ---------------------------------------------------------------------------
# Eval: greedy-decode the scratchpad, exact-match on the final answer.
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_answer(model, primes, base, V, n, max_len, abacus_max, rng, device, chunk=48):
    """Final-answer exact-match over n products, batched by prompt length (sub-chunked
    so the autoregressive decode doesn't OOM once chains grow to full max_len)."""
    model.eval()
    specials_t = torch.tensor(sorted(V["SPECIALS"]), device=device)

    samples, groups = [], defaultdict(list)
    for _ in range(n):
        p = primes[rng.randrange(len(primes))]
        x, y = rng.randrange(p), rng.randrange(p)
        xd, yd, pd = digits_base(x, base), digits_base(y, base), digits_base(p, base)
        toks = [V["BOS"]] + xd + [V["MUL"]] + yd + [V["MOD"]] + pd + [V["EQ"]]
        abac = ([0] + list(range(len(xd))) + [0] + list(range(len(yd)))
                + [0] + list(range(len(pd))) + [0])
        groups[len(toks)].append(len(samples))
        samples.append((toks, abac, (x * y) % p))

    chunks = []
    for L, idxs in groups.items():
        for cs in range(0, len(idxs), chunk):
            chunks.append(idxs[cs:cs + chunk])

    ok = 0
    for idxs in chunks:
        g = len(idxs)
        toks = torch.tensor([samples[i][0] for i in idxs], dtype=torch.long, device=device)
        abac = torch.tensor([samples[i][1] for i in idxs], dtype=torch.long, device=device)
        seg = torch.zeros(g, dtype=torch.long, device=device)
        done = torch.zeros(g, dtype=torch.bool, device=device)
        gen = [[] for _ in range(g)]
        while toks.shape[1] < max_len and not bool(done.all()):
            nxt = model(toks, abac)[:, -1].argmax(-1)
            nxt = torch.where(done, torch.full_like(nxt, V["PAD"]), nxt)
            is_special = (nxt.unsqueeze(1) == specials_t).any(1)
            new_abac = torch.where(is_special, torch.zeros_like(seg),
                                   torch.clamp(seg, max=abacus_max - 1))
            seg = torch.where(is_special, torch.zeros_like(seg), seg + 1)
            nxt_cpu, done_cpu = nxt.tolist(), done.tolist()
            for j in range(g):
                if not done_cpu[j] and nxt_cpu[j] != V["EOS"] and nxt_cpu[j] != V["PAD"]:
                    gen[j].append(nxt_cpu[j])
            toks = torch.cat([toks, nxt.unsqueeze(1)], dim=1)
            abac = torch.cat([abac, new_abac.unsqueeze(1)], dim=1)
            done = done | (nxt == V["EOS"])
        for j, i in enumerate(idxs):
            gj = gen[j]
            if V["COLON"] in gj:
                k = len(gj) - 1 - gj[::-1].index(V["COLON"])
                ans = [d for d in gj[k + 1:] if d < base]
                if ans and base_to_int(ans, base) == samples[i][2]:
                    ok += 1
    return ok / n


@torch.no_grad()
def eval_teacher_forced(model, primes, base, V, n, max_len, rng, device, chunk=48):
    """Cheap smooth signal: (token_acc, seq_acc) over supervised positions. tf_tok is
    the leading indicator -- autoregressive acc ~= tf_tok^chain. Forward is chunked so
    the eval (fp32, no autocast) doesn't OOM at long max_len / large eval-n."""
    model.eval()
    toks, abac, mask = make_batch(n, primes, base, V, max_len, rng, device)
    target, m = toks[:, 1:], mask[:, 1:]
    tot_hit, tot_tok, seq_ok = 0, 0, 0
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        pred = model(toks[s:e], abac[s:e])[:, :-1].argmax(-1)
        hit = (pred == target[s:e]) & m[s:e]
        tot_hit += hit.sum().item()
        tot_tok += m[s:e].sum().item()
        seq_ok += ((hit == m[s:e]).all(dim=1)).float().sum().item()
    return tot_hit / max(1, tot_tok), seq_ok / max(1, n)


def log_buckets(p_min: int, p_max: int):
    """3 log-spaced [lo, hi) buckets over [p_min, p_max), hardest last."""
    lo_log, hi_log = math.log(p_min), math.log(p_max)
    cuts = [math.exp(lo_log + (hi_log - lo_log) * k / 3) for k in range(4)]
    edges = [(int(cuts[k]), int(cuts[k + 1])) for k in range(3)]
    edges[0] = (p_min, edges[0][1])
    edges[-1] = (edges[-1][0], p_max)
    return edges


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=int, required=True, help="numeric base B (tier 5 -> 100)")
    ap.add_argument("--steps", type=int, default=80000)
    ap.add_argument("--batch", type=int, default=192)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--d-model", type=int, default=384)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--nhead", type=int, default=8)
    ap.add_argument("--dim-ff", type=int, default=1536)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--eval-n", type=int, default=300)
    ap.add_argument("--p-min", type=int, default=2 ** 33)
    ap.add_argument("--p-max", type=int, default=2 ** 64, help="tier 5 = [2^33, 2^64)")
    ap.add_argument("--pool-size", type=int, default=40000,
                    help="sampled prime-pool size for non-enumerable ranges")
    ap.add_argument("--curriculum", action="store_true",
                    help="ramp the prime ceiling small->large over --curr-frac of training")
    ap.add_argument("--curr-frac", type=float, default=0.6)
    ap.add_argument("--prime-pow", type=float, default=0.0,
                    help="weight prime sampling by p^alpha (1.0 = value-uniform, "
                         "scorer-matched; 0.0 = uniform over pool = uniform in bits)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt", type=str, default="training/checkpoints/modmul_t5.pt")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--amp", action="store_true", help="bf16 autocast on CUDA")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device()
    rng = random.Random(args.seed)
    eval_rng = random.Random(999)
    B = args.base
    V = make_vocab(B)

    pool_rng = random.Random(args.seed + 12345)
    POOL = build_prime_pool(args.p_min, args.p_max, args.pool_size, pool_rng)
    if not POOL:
        raise SystemExit(f"no primes in [{args.p_min}, {args.p_max})")

    # Sequence sizing in base-B digits. y up to pd digits -> pd Horner blocks.
    pd = len(digits_base(args.p_max - 1, B))            # prime / operand / remainder width
    ppd = len(digits_base((B - 1) * (args.p_max - 1), B))  # m1, pp, m2 (< B*p)
    td = len(digits_base(B * (args.p_max - 1), B))      # t = r1 + pp (< B*p)
    abacus_max = max(ppd, td, pd) + 2
    # per block: d : q1 : m1 : r1 : pp : t : q2 : m2 : r2  (+ STEP)
    #   single digits d,q1,q2 ; wide m1,pp,m2 (ppd) ; r1,r2 (pd) ; t (td) ; 8 colons + STEP
    block = (1 + 1 + 1            # d, q1, q2
             + 3 * ppd            # m1, pp, m2
             + 2 * pd             # r1, r2
             + td                 # t
             + 8 + 1)             # colons + STEP
    header = 1 + pd + 1 + pd + 1 + pd + 1               # BOS x MUL y MOD p EQ
    max_len = header + pd * block + 1 + 8               # EOS + slack

    curr_start = min(max(args.p_min * 4, 1024), args.p_max)

    def cur_pmax(step: int) -> int:
        if not args.curriculum:
            return args.p_max
        frac = min(1.0, step / max(1, args.curr_frac * args.steps))
        return int(curr_start * (args.p_max / curr_start) ** frac)

    edges = log_buckets(args.p_min, args.p_max)
    buckets = []
    for lo, hi in edges:
        ps = [p for p in POOL if lo <= p < hi]
        if ps:
            buckets.append((lo, hi, ps))

    # Optional value-weighted prime sampling (scorer-matched). Weight p^alpha; the
    # log-uniform pool has density ~1/p in value, so alpha=1 -> uniform in value.
    # Cumulative weights over the sorted POOL; primes_now = POOL[:hi] uses wcum[:hi].
    wcum = None
    if args.prime_pow > 0:
        acc, wcum = 0.0, []
        lmax = args.prime_pow * math.log(args.p_max)
        for p in POOL:
            acc += math.exp(args.prime_pow * math.log(p) - lmax)
            wcum.append(acc)
        n64 = sum(1 for p in POOL if p.bit_length() == 64)
        print(f"prime-pow {args.prime_pow}: value-weighted sampling on "
              f"(pool 64-bit fraction {n64/len(POOL):.3f}; sampling weight ~p^{args.prime_pow})")

    model = AbacusDecoder(V["VOCAB"], max_len, abacus_max, d_model=args.d_model,
                          nhead=args.nhead, num_layers=args.layers, dim_ff=args.dim_ff).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"device {device} | base {B} | vocab {V['VOCAB']} | pool {len(POOL)} primes "
          f"[{args.p_min},{args.p_max}) | pd {pd} blocks | max_len {max_len} | "
          f"abacus_max {abacus_max} | curriculum {args.curriculum} | params {n_params:,}")
    print(f"eval buckets: {[(lo, hi) for lo, hi, _ in buckets]}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    warmup = max(0, min(args.warmup, args.steps - 1))
    if warmup > 0:
        sched = torch.optim.lr_scheduler.SequentialLR(
            opt,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(
                    opt, start_factor=0.01, end_factor=1.0, total_iters=warmup),
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    opt, T_max=args.steps - warmup, eta_min=args.lr * 0.1),
            ],
            milestones=[warmup])
    else:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=args.steps, eta_min=args.lr * 0.1)
    loss_fn = nn.CrossEntropyLoss()

    cfg = dict(base=B, d_model=args.d_model, layers=args.layers, nhead=args.nhead,
               dim_ff=args.dim_ff, max_len=max_len, abacus_max=abacus_max,
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
    start = time.monotonic()

    for step in range(start_step + 1, args.steps + 1):
        model.train()
        hi = bisect.bisect_left(POOL, cur_pmax(step))
        primes_now = POOL[:hi] if hi > 0 else POOL[:1]
        toks, abac, mask = make_batch(args.batch, primes_now, B, V, max_len, rng, device,
                                      wcum=wcum)
        with amp_ctx:
            logits = model(toks, abac)
            logits = logits[:, :-1].reshape(-1, V["VOCAB"])
            target = toks[:, 1:].reshape(-1)
            m = mask[:, 1:].reshape(-1)
            loss = loss_fn(logits[m], target[m])
        opt.zero_grad(); loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step(); sched.step()

        if step % args.eval_every == 0:
            parts, accs = [], []
            # Eval under the SAME bf16 autocast as training. In fp32 the decode's
            # attention kernel can hit "unspecified launch failure" at long max_len
            # (base-16 -> 1853); bf16 matches the proven-working training forward and
            # halves attention memory/kernel size.
            with amp_ctx:
                for lo, hi, ps in buckets:
                    acc = eval_answer(model, ps, B, V, args.eval_n, max_len, abacus_max,
                                      eval_rng, device)
                    accs.append(acc)
                    parts.append(f"[{lo}-{hi}) acc {acc:.3f}")
                tf_tok, tf_seq = eval_teacher_forced(model, POOL, B, V, args.eval_n, max_len,
                                                     eval_rng, device)
            print(f"step {step:6d} | loss {loss.item():.4f} | " + " | ".join(parts)
                  + f" | tf_tok {tf_tok:.3f} tf_seq {tf_seq:.3f}"
                  + f" | cur_pmax {cur_pmax(step)} | {time.monotonic()-start:.0f}s")
            save_ckpt(args.ckpt, step, best)
            score = accs[-1] if accs else 0.0                  # hardest bucket
            if args.ckpt and score > best:
                best = score
                save_ckpt(args.ckpt.replace(".pt", "_best.pt"), step, best)

    save_ckpt(args.ckpt, args.steps, best)
    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
