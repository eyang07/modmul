"""Tier-5 Probe A: is the per-step partial product pp = x * d learnable one-shot
at a denser numeric base?

Tier 5 is (a*b) mod p with p in [2^33, 2^64) (~20 decimal digits). In base-10 the
Horner scratchpad is ~1700 tokens, which busts BOTH the chain-survival math and the
300s decode budget (see PROJECT_STATE / memory/tier4-cracked). The fix is a denser
base B: base-100 -> ~700-token chain, base-1000 -> ~400. But a denser base makes the
ONLY genuinely new sub-skill harder:

    pp = x * d     with  x < 2^64  and  d a single base-B digit (0 <= d < B)

This is "long number x single base-B digit". The elementary op inside it is a
base-B single-digit multiply = a (<=3-decimal)x(<=3-decimal) cross product for B=1000,
(<=2-decimal)x(<=2-decimal) for B=100. We KNOW one-shot 5x5 decimal multiply is a hard
wall (tier-3 history) and 4x4 is the learnable edge. So the make-or-break question for
tier 5 is: at base B, can the model emit pp = x*d in one shot at >= ~0.999?

This probe trains exactly that, nothing else, parametrized by --base. It is cheap
(short sequences, no Horner chain) so the grok/no-grok pattern shows fast.

    BOS  x_baseB  MUL  d  EQ  pp_baseB  EOS          (loss on pp + EOS)

GO/NO-GO:
  * base-100 groks (full-width acc -> ~1.0) and base-1000 stalls  -> use base 100.
  * both grok                                                     -> use base 1000
                                                                     (shorter chain,
                                                                      faster decode).
  * base-100 also stalls -> pp itself needs a sub-scratchpad (partial products per
    base-B digit of x, with carries); that is Probe A.2, write it only if needed.

Watch tf_tok (teacher-forced per-token acc) -- the leading indicator. The per-bucket
exact-match (esp. the full-width x bucket) is the honest verdict.

Usage (on a CUDA pod):
    python training/tier5_pp_probe.py --base 100  --amp --ckpt training/checkpoints/pp_b100.pt  > pp_b100.log 2>&1 &
    python training/tier5_pp_probe.py --base 1000 --amp --ckpt training/checkpoints/pp_b1000.pt > pp_b1000.log 2>&1 &
    # optional sanity baseline (should nail it almost immediately):
    python training/tier5_pp_probe.py --base 10   --amp --ckpt training/checkpoints/pp_b10.pt   > pp_b10.log 2>&1 &
"""

from __future__ import annotations

import argparse
import contextlib
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
    PAD, BOS, MUL, EQ, EOS = base, base + 1, base + 2, base + 3, base + 4
    vocab = base + 5
    specials = {PAD, BOS, MUL, EQ, EOS}
    return dict(PAD=PAD, BOS=BOS, MUL=MUL, EQ=EQ, EOS=EOS, VOCAB=vocab, SPECIALS=specials)


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


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Example construction:  BOS x MUL d EQ pp EOS   (pp = x * d, all base-B, MSB-first)
# Abacus index = place within the current number (0 at MSB), reset on specials --
# same convention as modmul_probe.
# ---------------------------------------------------------------------------

def build_example(x: int, d: int, base: int, V: dict, max_len: int):
    xd = digits_base(x, base)
    ppd = digits_base(x * d, base)
    toks: list[int] = [V["BOS"]]
    abac: list[int] = [0]
    is_out: list[bool] = [False]
    for i, dig in enumerate(xd):
        toks.append(dig); abac.append(i); is_out.append(False)
    toks += [V["MUL"], d, V["EQ"]]; abac += [0, 0, 0]; is_out += [False, False, False]
    for i, dig in enumerate(ppd):
        toks.append(dig); abac.append(i); is_out.append(True)
    toks.append(V["EOS"]); abac.append(0); is_out.append(True)

    pad = max_len - len(toks)
    if pad < 0:
        raise ValueError(f"max_len {max_len} too small for sequence of {len(toks)}")
    toks += [V["PAD"]] * pad
    abac += [0] * pad
    is_out += [False] * pad
    return toks, abac, is_out


def sample_xd(base: int, bits_min: int, bits_max: int, rng: random.Random):
    """x ~ random bit-length in [bits_min, bits_max] (spread of operand widths, like
    the log-uniform pool); d ~ U[0, base) (a single base-B digit)."""
    bits = rng.randint(bits_min, bits_max)
    lo = 1 if bits <= 1 else 1 << (bits - 1)
    x = rng.randrange(lo, 1 << bits)
    d = rng.randrange(0, base)
    return x, d


def make_batch(batch, base, V, bits_min, bits_max, max_len, rng, device):
    T, A, M = [], [], []
    for _ in range(batch):
        x, d = sample_xd(base, bits_min, bits_max, rng)
        t, ab, m = build_example(x, d, base, V, max_len)
        T.append(t); A.append(ab); M.append(m)
    tt = lambda v, dt: torch.tensor(v, dtype=dt, device=device)
    return tt(T, torch.long), tt(A, torch.long), tt(M, torch.bool)


# ---------------------------------------------------------------------------
# Model: same abacus decoder backbone as the other probes (vocab depends on base).
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
# Eval: greedy-decode pp, exact-match, bucketed by x bit-length (full-width is
# the honest tier-5 signal). Plus teacher-forced token acc (leading indicator).
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_pp(model, base, V, n, bit_edges, max_len, abacus_max, rng, device):
    model.eval()
    specials_t = torch.tensor(sorted(V["SPECIALS"]), device=device)
    per = defaultdict(lambda: [0, 0])   # bucket idx -> [correct, total]

    samples, groups = [], defaultdict(list)
    for _ in range(n):
        x, d = sample_xd(base, bit_edges[0][0], bit_edges[-1][1], rng)
        xd = digits_base(x, base)
        toks = [V["BOS"]] + xd + [V["MUL"], d, V["EQ"]]
        abac = [0] + list(range(len(xd))) + [0, 0, 0]
        b = x.bit_length()
        bk = next((i for i, (lo, hi) in enumerate(bit_edges) if lo <= b < hi), len(bit_edges) - 1)
        groups[len(toks)].append(len(samples))
        samples.append((toks, abac, x * d, bk))

    for L, idxs in groups.items():
        g = len(idxs)
        toks = torch.tensor([samples[i][0] for i in idxs], dtype=torch.long, device=device)
        abac = torch.tensor([samples[i][1] for i in idxs], dtype=torch.long, device=device)
        seg = torch.zeros(g, dtype=torch.long, device=device)
        done = torch.zeros(g, dtype=torch.bool, device=device)
        gen = [[] for _ in range(g)]
        while toks.shape[1] < max_len and not bool(done.all()):
            nxt = model(toks, abac)[:, -1].argmax(-1)
            nxt = torch.where(done, torch.full_like(nxt, V["PAD"]), nxt)
            is_sp = (nxt.unsqueeze(1) == specials_t).any(1)
            new_abac = torch.where(is_sp, torch.zeros_like(seg),
                                   torch.clamp(seg, max=abacus_max - 1))
            seg = torch.where(is_sp, torch.zeros_like(seg), seg + 1)
            nc, dc = nxt.tolist(), done.tolist()
            for j in range(g):
                if not dc[j] and nc[j] != V["EOS"] and nc[j] != V["PAD"]:
                    gen[j].append(nc[j])
            toks = torch.cat([toks, nxt.unsqueeze(1)], dim=1)
            abac = torch.cat([abac, new_abac.unsqueeze(1)], dim=1)
            done = done | (nxt == V["EOS"])
        for j, i in enumerate(idxs):
            ans = [t for t in gen[j] if t < base]
            bk = samples[i][3]
            per[bk][1] += 1
            if ans and base_to_int(ans, base) == samples[i][2]:
                per[bk][0] += 1
    return per


@torch.no_grad()
def eval_tf(model, base, V, n, bits_min, bits_max, max_len, rng, device):
    model.eval()
    toks, abac, mask = make_batch(n, base, V, bits_min, bits_max, max_len, rng, device)
    pred = model(toks, abac)[:, :-1].argmax(-1)
    target, m = toks[:, 1:], mask[:, 1:]
    hit = (pred == target) & m
    tok_acc = hit.sum().item() / max(1, m.sum().item())
    seq_acc = ((hit == m).all(dim=1)).float().mean().item()
    return tok_acc, seq_acc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=int, required=True, help="numeric base B (e.g. 10, 100, 1000)")
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--d-model", type=int, default=384)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--nhead", type=int, default=8)
    ap.add_argument("--dim-ff", type=int, default=1536)
    ap.add_argument("--bits-min", type=int, default=8)
    ap.add_argument("--bits-max", type=int, default=64, help="tier-5 operands are < 2^64")
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--eval-n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt", type=str, default="training/checkpoints/pp_probe.pt")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--amp", action="store_true", help="bf16 autocast on CUDA")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device()
    rng = random.Random(args.seed)
    eval_rng = random.Random(999)
    B = args.base
    V = make_vocab(B)

    # Sizing: x < 2^bits_max ; pp = x*d < 2^bits_max * B.
    x_max = (1 << args.bits_max) - 1
    pp_max = x_max * (B - 1)
    xlen = len(digits_base(x_max, B))
    pplen = len(digits_base(pp_max, B))
    abacus_max = max(xlen, pplen) + 2
    max_len = 1 + xlen + 3 + pplen + 1 + 4        # BOS x MUL d EQ pp EOS + slack

    # Eval buckets by x bit-length; the last (widest) bucket is the honest signal.
    lo, hi = args.bits_min, args.bits_max
    third = max(1, (hi - lo) // 3)
    bit_edges = [(lo, lo + third), (lo + third, lo + 2 * third), (lo + 2 * third, hi + 1)]

    model = AbacusDecoder(V["VOCAB"], max_len, abacus_max, d_model=args.d_model,
                          nhead=args.nhead, num_layers=args.layers, dim_ff=args.dim_ff).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"device {device} | base {B} | vocab {V['VOCAB']} | xlen<= {xlen} pplen<= {pplen} | "
          f"max_len {max_len} | abacus_max {abacus_max} | bits [{lo},{hi}] | params {n_params:,}")
    print(f"eval buckets (x bits): {bit_edges}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    warmup = max(0, min(args.warmup, args.steps - 1))
    if warmup > 0:
        sched = torch.optim.lr_scheduler.SequentialLR(
            opt,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.01, end_factor=1.0, total_iters=warmup),
                torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps - warmup, eta_min=args.lr * 0.1),
            ],
            milestones=[warmup])
    else:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=args.lr * 0.1)
    loss_fn = nn.CrossEntropyLoss()

    cfg = dict(base=B, d_model=args.d_model, layers=args.layers, nhead=args.nhead,
               dim_ff=args.dim_ff, max_len=max_len, abacus_max=abacus_max,
               bits_min=args.bits_min, bits_max=args.bits_max)

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
        toks, abac, mask = make_batch(args.batch, B, V, args.bits_min, args.bits_max,
                                      max_len, rng, device)
        with amp_ctx:
            logits = model(toks, abac)[:, :-1].reshape(-1, V["VOCAB"])
            target = toks[:, 1:].reshape(-1)
            m = mask[:, 1:].reshape(-1)
            loss = loss_fn(logits[m], target[m])
        opt.zero_grad(); loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step(); sched.step()

        if step % args.eval_every == 0:
            per = eval_pp(model, B, V, args.eval_n, bit_edges, max_len, abacus_max,
                          eval_rng, device)
            parts = []
            for bk, (loe, hie) in enumerate(bit_edges):
                c, t = per[bk]
                parts.append(f"[{loe}-{hie}b) {c/max(1,t):.3f}")
            full_acc = per[len(bit_edges) - 1][0] / max(1, per[len(bit_edges) - 1][1])
            tf_tok, tf_seq = eval_tf(model, B, V, args.eval_n, args.bits_min,
                                     args.bits_max, max_len, eval_rng, device)
            print(f"step {step:6d} | loss {loss.item():.4f} | " + " ".join(parts)
                  + f" | tf_tok {tf_tok:.4f} tf_seq {tf_seq:.3f} | {time.monotonic()-start:.0f}s")
            save_ckpt(args.ckpt, step, best)
            if args.ckpt and full_acc > best:
                best = full_acc
                save_ckpt(args.ckpt.replace(".pt", "_best.pt"), step, best)

    save_ckpt(args.ckpt, args.steps, best)
    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
