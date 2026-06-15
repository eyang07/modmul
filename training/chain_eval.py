"""Chain the two PROVEN halves and measure end-to-end tier-3 accuracy.

The single-model composition (compose_probe.py) stalls: joint training starves the
multiply sub-skill. But both halves are already saturated as separate checkpoints:
  * model A  tier0_mult_best.pt    : a,b -> product digits (LSB), exact@5d ~0.98
  * model B  longdiv_t3_L8_best.pt : product digits + p -> remainder (long division), ~0.96

Here we run A then B: decode the product with A, reverse its digit tokens to MSB
(a string op on already-generated tokens -- NOT arithmetic on the product), feed those
digits to B as N, decode the remainder. The product's modular reduction stays LEARNED
(B does it via long division); we never apply %/// to the product. Composed acc should
be ~= P(A) * P(B) ~= 0.98 * 0.96 ~= 0.94.

Per bucket we report:
  prod    = A's product == a*b
  chain   = end-to-end A->B remainder == (a*b) % p   (the real tier-3 metric)
  oracle  = B fed the TRUE product == (a*b) % p       (isolates B; chain<=oracle)

Usage:
    python training/chain_eval.py --n 500
"""

from __future__ import annotations

import argparse
import random
from collections import defaultdict

import torch

import tier0_probe as A
import longdiv_probe as B
from data import primes_for_tier


def load_model(module, path, device):
    ck = torch.load(path, map_location=device)
    c = ck["config"]
    m = module.AbacusDecoder(
        max_len=c["max_len"], abacus_max=c["abacus_max"], d_model=c["d_model"],
        num_layers=c["layers"],
        **({"nhead": c["nhead"], "dim_ff": c["dim_ff"]} if "nhead" in c else {}),
    ).to(device)
    m.load_state_dict(ck["model"]); m.eval()
    return m, c


@torch.no_grad()
def decode_products(model, cfg, ab_pairs, device):
    """Model A: greedy-decode product digits (LSB) for each (a, b). Batched by prompt
    length. Returns list of LSB digit lists (model's raw output)."""
    max_len, abmax = cfg["max_len"], cfg["abacus_max"]
    out = [None] * len(ab_pairs)
    groups = defaultdict(list)
    prompts = []
    for i, (a, b) in enumerate(ab_pairs):
        xr, yr = A.digits_lsb(a), A.digits_lsb(b)
        toks = [A.BOS] + xr + [A.MUL] + yr + [A.EQ]
        abac = [0] + list(range(len(xr))) + [0] + list(range(len(yr))) + [0]
        groups[len(toks)].append(i)
        prompts.append((toks, abac))
    for L, idxs in groups.items():
        g = len(idxs)
        toks = torch.tensor([prompts[i][0] for i in idxs], dtype=torch.long, device=device)
        abac = torch.tensor([prompts[i][1] for i in idxs], dtype=torch.long, device=device)
        seg = torch.zeros(g, dtype=torch.long, device=device)
        done = torch.zeros(g, dtype=torch.bool, device=device)
        gen = [[] for _ in range(g)]
        while toks.shape[1] < max_len and not bool(done.all()):
            nxt = model(toks, abac)[:, -1].argmax(-1)
            nxt = torch.where(done, torch.full_like(nxt, A.PAD), nxt)
            new_abac = torch.clamp(seg, max=abmax - 1)
            seg = seg + 1
            nxt_cpu, done_cpu = nxt.tolist(), done.tolist()
            for j in range(g):
                if not done_cpu[j] and nxt_cpu[j] != A.EOS and nxt_cpu[j] != A.PAD:
                    gen[j].append(nxt_cpu[j])
            toks = torch.cat([toks, nxt.unsqueeze(1)], dim=1)
            abac = torch.cat([abac, new_abac.unsqueeze(1)], dim=1)
            done = done | (nxt == A.EOS)
        for j, i in enumerate(idxs):
            out[i] = [d for d in gen[j] if d < 10]
    return out


@torch.no_grad()
def decode_remainders(model, cfg, N_p_pairs, device):
    """Model B: greedy-decode the long-division scratchpad for each (N_digits_msb, p).
    N_digits_msb is a digit list (model A's product, reversed). Returns list of ints
    (final remainder = digits after the last COLON), or None if unparseable."""
    max_len, abmax = cfg["max_len"], cfg["abacus_max"]
    specials = torch.tensor(sorted(B.SPECIALS), device=device)
    out = [None] * len(N_p_pairs)
    groups = defaultdict(list)
    prompts = []
    for i, (Nd, p) in enumerate(N_p_pairs):
        pd = B.digits_msb(p)
        toks = [B.BOS] + Nd + [B.MOD] + pd + [B.EQ]
        abac = [0] + list(range(len(Nd))) + [0] + list(range(len(pd))) + [0]
        groups[len(toks)].append(i)
        prompts.append((toks, abac))
    for L, idxs in groups.items():
        g = len(idxs)
        toks = torch.tensor([prompts[i][0] for i in idxs], dtype=torch.long, device=device)
        abac = torch.tensor([prompts[i][1] for i in idxs], dtype=torch.long, device=device)
        seg = torch.zeros(g, dtype=torch.long, device=device)
        done = torch.zeros(g, dtype=torch.bool, device=device)
        gen = [[] for _ in range(g)]
        while toks.shape[1] < max_len and not bool(done.all()):
            nxt = model(toks, abac)[:, -1].argmax(-1)
            nxt = torch.where(done, torch.full_like(nxt, B.PAD), nxt)
            is_special = (nxt.unsqueeze(1) == specials).any(1)
            new_abac = torch.where(is_special, torch.zeros_like(seg),
                                   torch.clamp(seg, max=abmax - 1))
            seg = torch.where(is_special, torch.zeros_like(seg), seg + 1)
            nxt_cpu, done_cpu = nxt.tolist(), done.tolist()
            for j in range(g):
                if not done_cpu[j] and nxt_cpu[j] != B.EOS and nxt_cpu[j] != B.PAD:
                    gen[j].append(nxt_cpu[j])
            toks = torch.cat([toks, nxt.unsqueeze(1)], dim=1)
            abac = torch.cat([abac, new_abac.unsqueeze(1)], dim=1)
            done = done | (nxt == B.EOS)
        for j, i in enumerate(idxs):
            gj = gen[j]
            if B.COLON in gj:
                k = len(gj) - 1 - gj[::-1].index(B.COLON)
                ans = [d for d in gj[k + 1:] if d < 10]
                if ans:
                    out[i] = B.msb_to_int(ans)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500, help="samples per bucket")
    ap.add_argument("--mult-ckpt", default="training/checkpoints/tier0_mult_best.pt")
    ap.add_argument("--div-ckpt", default="training/checkpoints/longdiv_t3_L8_best.pt")
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    device = A.pick_device()
    rng = random.Random(args.seed)
    mA, cA = load_model(A, args.mult_ckpt, device)
    mB, cB = load_model(B, args.div_ckpt, device)
    print(f"device {device} | A {args.mult_ckpt} (max_digits {cA['max_digits']}) | "
          f"B {args.div_ckpt} (p [{cB['p_min']},{cB['p_max']}))")

    POOL = primes_for_tier(3)
    buckets = [(512, 2048), (2048, 8192), (8192, 65536)]
    print(f"{'bucket':>16} | {'prod(A)':>8} | {'chain':>8} | {'oracle(B)':>9}")
    for lo, hi in buckets:
        ps = [p for p in POOL if lo <= p < hi]
        samples = []
        for _ in range(args.n):
            p = ps[rng.randrange(len(ps))]
            a, b = rng.randrange(p), rng.randrange(p)
            samples.append((a, b, p))
        # Stage A: products
        prods = decode_products(mA, cA, [(a, b) for a, b, _ in samples], device)
        prod_ok = sum(1 for (a, b, _), pl in zip(samples, prods)
                      if pl and A.digits_lsb(a * b) == pl)
        # Stage B: chain (A's product, reversed to MSB) and oracle (true product MSB)
        chain_in = [(pl[::-1] if pl else [0], p) for (a, b, p), pl in zip(samples, prods)]
        oracle_in = [(B.digits_msb(a * b), p) for a, b, p in samples]
        rem_chain = decode_remainders(mB, cB, chain_in, device)
        rem_oracle = decode_remainders(mB, cB, oracle_in, device)
        chain_ok = sum(1 for (a, b, p), r in zip(samples, rem_chain) if r == (a * b) % p)
        oracle_ok = sum(1 for (a, b, p), r in zip(samples, rem_oracle) if r == (a * b) % p)
        n = args.n
        print(f"[{lo:>5}-{hi:<5}) | {prod_ok/n:8.3f} | {chain_ok/n:8.3f} | {oracle_ok/n:9.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
