"""Composed tier-3 probe: multiply THEN reduce, end to end, in one model.

Both halves are already proven independently (see PROJECT_STATE.md):
  * multiply  (tier0_probe.py)  : LSB-first digit transformer, exact@5d ~0.98
  * reduce    (longdiv_probe.py): MSB-first long-division scratchpad, rem ~0.96

This script fuses them into ONE autoregressive scratchpad model that takes a, b, p
and emits the product, then long-divides it by p to land on (a*b) mod p. Everything
after EQ is net-generated and supervised, so it stays submission-compliant (the model
never calls Python %/// on the product; the only % in this file is in the data
generator that builds the supervision target, and in eval scoring -- neither ships).

Sequence (the multiply half is LSB-first, the division half MSB-first; the N_msb block
is the product re-emitted most-significant-first -- a cheap supervised reversal that
decouples the two algorithms so neither has to do the other's job):

    BOS a_lsb MUL b_lsb MOD p_msb EQ  prod_lsb  PSEP  N_msb  DSEP \
        q1 : r1_msb STEP q2 : r2_msb STEP ... qk : rk_msb  EOS

where the long-division chain is, MSB-first across N's digits:
    r = 0; for each digit d of N: r = r*10 + d; q = r//p; r -= q*p   (final r = N mod p)

Abacus index = place within the current segment, reset to 0 after every special token
(identical scheme to both probes, so the greedy-decode abacus update is shared).

Composition math: composed acc ~= P(mult) * P(reduce) ~= 0.98 * 0.95 ~= 0.93 -> over
the 0.90 tier-3 bar. The risk is joint training holding BOTH halves at once; watch
tf_tok (teacher-forced per-token acc) as the leading indicator -- the end-to-end metric
sits near 0 until tf_tok crosses ~0.94, then groks. prod_acc / rem_acc in the eval line
show which half is limiting.

Usage (proven recipe -- annealed cosine over the run, warmup, grad-clip, amp):
    python training/compose_probe.py --steps 80000 --amp \
        --p-min 512 --p-max 65536 --eval-every 2000 \
        --ckpt training/checkpoints/compose_t3.pt
"""

from __future__ import annotations

import argparse
import bisect
import contextlib
import os
import random
import time
from collections import defaultdict

import torch
import torch.nn as nn

from data import primes_for_tier

# Vocab: 0-9 digits, then specials.
PAD, BOS, MUL, MOD, EQ, PSEP, DSEP, COLON, STEP, EOS = 10, 11, 12, 13, 14, 15, 16, 17, 18, 19
VOCAB = 20
SPECIALS = {PAD, BOS, MUL, MOD, EQ, PSEP, DSEP, COLON, STEP, EOS}


def digits_lsb(n: int) -> list[int]:
    """Non-negative int -> decimal digits, least-significant-first ([0] for 0)."""
    if n == 0:
        return [0]
    out = []
    while n > 0:
        out.append(n % 10)
        n //= 10
    return out


def digits_msb(n: int) -> list[int]:
    """Non-negative int -> decimal digits, most-significant-first ([0] for 0)."""
    return digits_lsb(n)[::-1]


def lsb_to_int(ds: list[int]) -> int:
    v = 0
    for d in reversed(ds):
        v = v * 10 + d
    return v


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

def build_example(a: int, b: int, p: int, max_len: int):
    """Return (tokens, abacus_idx, is_output) padded to max_len.

    Supervises everything after EQ: product (LSB), the reversal block (N MSB), and
    the long-division chain. Abacus resets to 0 on every special token.
    """
    N = a * b
    al, bl, pm = digits_lsb(a), digits_lsb(b), digits_msb(p)
    prodl, Nm = digits_lsb(N), digits_msb(N)

    # prompt: BOS a MUL b MOD p EQ  (nothing here is supervised)
    toks = [BOS] + al + [MUL] + bl + [MOD] + pm + [EQ]
    abac = ([0] + list(range(len(al))) + [0] + list(range(len(bl)))
            + [0] + list(range(len(pm))) + [0])
    is_out = [False] * len(toks)

    def emit(tok, ab, out=True):
        toks.append(tok); abac.append(ab); is_out.append(out)

    # product, LSB-first
    for i, d in enumerate(prodl):
        emit(d, i)
    emit(PSEP, 0)
    # reversal bridge: same number, MSB-first
    for i, d in enumerate(Nm):
        emit(d, i)
    emit(DSEP, 0)
    # long-division chain
    for i, (q, r) in enumerate(long_division_steps(N, p)):
        if i > 0:
            emit(STEP, 0)
        emit(q, 0)
        emit(COLON, 0)
        for j, d in enumerate(digits_msb(r)):
            emit(d, j)
    emit(EOS, 0)

    pad = max_len - len(toks)
    if pad < 0:
        raise ValueError(f"max_len {max_len} too small for sequence of {len(toks)}")
    toks += [PAD] * pad
    abac += [0] * pad
    is_out += [False] * pad
    return toks, abac, is_out


def make_batch(batch: int, primes: list[int], max_len: int, rng, device):
    """a, b ~ U[0, p) for p sampled uniformly from the pool."""
    T, A, M = [], [], []
    for _ in range(batch):
        p = primes[rng.randrange(len(primes))]
        a, b = rng.randrange(p), rng.randrange(p)
        t, ab, m = build_example(a, b, p, max_len)
        T.append(t); A.append(ab); M.append(m)
    tt = lambda v, dt: torch.tensor(v, dtype=dt, device=device)
    return tt(T, torch.long), tt(A, torch.long), tt(M, torch.bool)


# ---------------------------------------------------------------------------
# Model (abacus decoder, same backbone as the two probes)
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
# Eval: greedy-decode the full scratchpad; score the product and the remainder
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_compose(model, primes, n, max_len, abacus_max, rng, device):
    """Returns (prod_acc, rem_acc) over n samples, batched by prompt length.

    prod_acc = product (digits before PSEP, LSB) == a*b.
    rem_acc  = final remainder (digits after the LAST COLON, MSB) == (a*b) % p.
               This is the real end-to-end tier-3 metric.
    """
    model.eval()
    specials_t = torch.tensor(sorted(SPECIALS), device=device)

    samples, groups = [], defaultdict(list)
    for _ in range(n):
        p = primes[rng.randrange(len(primes))]
        a, b = rng.randrange(p), rng.randrange(p)
        N = a * b
        al, bl, pm = digits_lsb(a), digits_lsb(b), digits_msb(p)
        toks = [BOS] + al + [MUL] + bl + [MOD] + pm + [EQ]
        abac = ([0] + list(range(len(al))) + [0] + list(range(len(bl)))
                + [0] + list(range(len(pm))) + [0])
        groups[len(toks)].append(len(samples))
        samples.append((toks, abac, N, N % p))

    prod_ok = rem_ok = 0
    for L, idxs in groups.items():
        g = len(idxs)
        toks = torch.tensor([samples[i][0] for i in idxs], dtype=torch.long, device=device)
        abac = torch.tensor([samples[i][1] for i in idxs], dtype=torch.long, device=device)
        seg = torch.zeros(g, dtype=torch.long, device=device)
        done = torch.zeros(g, dtype=torch.bool, device=device)
        gen = [[] for _ in range(g)]
        while toks.shape[1] < max_len and not bool(done.all()):
            nxt = model(toks, abac)[:, -1].argmax(-1)              # [g]
            nxt = torch.where(done, torch.full_like(nxt, PAD), nxt)
            is_special = (nxt.unsqueeze(1) == specials_t).any(1)
            new_abac = torch.where(is_special, torch.zeros_like(seg),
                                   torch.clamp(seg, max=abacus_max - 1))
            seg = torch.where(is_special, torch.zeros_like(seg), seg + 1)
            nxt_cpu, done_cpu = nxt.tolist(), done.tolist()
            for j in range(g):
                if not done_cpu[j] and nxt_cpu[j] != EOS and nxt_cpu[j] != PAD:
                    gen[j].append(nxt_cpu[j])
            toks = torch.cat([toks, nxt.unsqueeze(1)], dim=1)
            abac = torch.cat([abac, new_abac.unsqueeze(1)], dim=1)
            done = done | (nxt == EOS)
        for j, i in enumerate(idxs):
            gj = gen[j]
            _, _, N_true, rem_true = samples[i]
            # product: digits before the first PSEP, LSB-first
            if PSEP in gj:
                pe = [d for d in gj[:gj.index(PSEP)] if d < 10]
                if pe and lsb_to_int(pe) == N_true:
                    prod_ok += 1
            # remainder: digits after the last COLON, MSB-first
            if COLON in gj:
                k = len(gj) - 1 - gj[::-1].index(COLON)
                ans = [d for d in gj[k + 1:] if d < 10]
                if ans and msb_to_int(ans) == rem_true:
                    rem_ok += 1
    return prod_ok / n, rem_ok / n


@torch.no_grad()
def eval_teacher_forced(model, primes, n, max_len, rng, device):
    """Cheap smooth signal during warmup: one batched teacher-forced forward.

    Returns (token_acc, seq_acc) over supervised positions. tf_tok is the leading
    indicator -- autoregressive acc ~= tf_tok^(chain length), so it climbs first.
    """
    model.eval()
    toks, abac, mask = make_batch(n, primes, max_len, rng, device)
    pred = model(toks, abac)[:, :-1].argmax(-1)
    target, m = toks[:, 1:], mask[:, 1:]
    hit = (pred == target) & m
    tok_acc = hit.sum().item() / max(1, m.sum().item())
    seq_acc = ((hit == m).all(dim=1)).float().mean().item()
    return tok_acc, seq_acc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=80000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=2000,
                    help="linear LR warmup steps (stabilizes deep models)")
    ap.add_argument("--grad-clip", type=float, default=1.0,
                    help="max grad norm; <=0 disables")
    ap.add_argument("--d-model", type=int, default=384)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--nhead", type=int, default=8)
    ap.add_argument("--dim-ff", type=int, default=1536)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--eval-n", type=int, default=300)
    ap.add_argument("--p-min", type=int, default=512)
    ap.add_argument("--p-max", type=int, default=65536, help="full tier-3 is [512, 65536)")
    ap.add_argument("--curriculum", action="store_true",
                    help="ramp the prime ceiling small->large over --curr-frac of training")
    ap.add_argument("--curr-frac", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt", type=str, default="training/checkpoints/compose_t3.pt",
                    help="save state (latest) + a _best.pt copy; '' disables")
    ap.add_argument("--resume", action="store_true",
                    help="resume model/opt/sched/step from --ckpt if it exists")
    ap.add_argument("--amp", action="store_true",
                    help="bf16 autocast on CUDA (~1.5-2x faster on Ada/Ampere)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device()
    rng = random.Random(args.seed)
    eval_rng = random.Random(999)

    POOL = [p for p in primes_for_tier(3) if args.p_min <= p < args.p_max]
    if not POOL:
        raise SystemExit(f"no primes in [{args.p_min}, {args.p_max})")

    # Sequence sizing: N up to (p_max-1)^2; operands/remainders < p_max.
    nd = len(str((args.p_max - 1) ** 2))     # max digits in the product N
    pd = len(str(args.p_max - 1))            # max digits in a prime / operand / remainder
    abacus_max = nd + 2
    header = 1 + pd + 1 + pd + 1 + pd + 1    # BOS a MUL b MOD p EQ
    mid = nd + 1 + nd + 1                    # prod PSEP N DSEP
    chain = nd * (1 + 1 + pd + 1)            # q COLON r STEP per N-digit (overcounts a STEP)
    max_len = header + mid + chain + 1 + 8   # EOS + slack

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

    cfg = dict(d_model=args.d_model, layers=args.layers, nhead=args.nhead,
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
        toks, abac, mask = make_batch(args.batch, primes_now, max_len, rng, device)
        with amp_ctx:
            logits = model(toks, abac)
            logits = logits[:, :-1].reshape(-1, VOCAB)
            target = toks[:, 1:].reshape(-1)
            m = mask[:, 1:].reshape(-1)
            loss = loss_fn(logits[m], target[m])
        opt.zero_grad(); loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step(); sched.step()

        if step % args.eval_every == 0:
            parts, rems = [], []
            for lo, hi, ps in buckets:
                prod_acc, rem_acc = eval_compose(model, ps, args.eval_n, max_len,
                                                 abacus_max, eval_rng, device)
                rems.append(rem_acc)
                parts.append(f"[{lo}-{hi}) prod {prod_acc:.3f} rem {rem_acc:.3f}")
            tf_tok, tf_seq = eval_teacher_forced(model, POOL, args.eval_n, max_len,
                                                 eval_rng, device)
            print(f"step {step:6d} | loss {loss.item():.4f} | " + " | ".join(parts)
                  + f" | tf_tok {tf_tok:.3f} tf_seq {tf_seq:.3f}"
                  + f" | cur_pmax {cur_pmax(step)} | {time.monotonic()-start:.0f}s")
            save_ckpt(args.ckpt, step, best)                       # latest (for resume)
            score = rems[-1] if rems else 0.0                      # hardest bucket
            if args.ckpt and score > best:
                best = score
                save_ckpt(args.ckpt.replace(".pt", "_best.pt"), step, best)

    save_ckpt(args.ckpt, args.steps, best)
    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
