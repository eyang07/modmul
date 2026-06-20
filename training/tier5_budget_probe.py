"""Budget probe: how long does base-16 tier-5 decode take on CUDA?

base-16 cleared the 0.90 learnability gate (final hard bucket 0.93, tf_tok 1.000),
but our AbacusDecoder has NO KV-cache, so greedy decode is ~O(L^3) over the
~1853-token chain. The official scorer gives 300s TOTAL for 1100 problems
(100/tier x 11 tiers, 5 primes/tier). This script answers the make-or-break
question BEFORE we invest in integration: can 100 tier-5 problems decode in a
small-enough slice of 300s (leaving room for tiers 1-4)?

It mirrors the scorer's exact shape -- 5 distinct primes near 2^64, 100 random
(x, y) reduced into [0, p) -- and runs the SAME length-grouped greedy decode as
eval_answer/the submission, under bf16 autocast (matching training + the eval
fix). It reports wall-clock, ms/problem, projected 100-problem time, peak VRAM,
and (free bonus) decoded exact-match accuracy on this fresh sample.

Usage (on the 5090, after git pull):
    python training/tier5_budget_probe.py \
        --ckpt training/checkpoints/modmul_t5_b16_d512_best.pt --amp
"""

from __future__ import annotations

import argparse
import contextlib
import random
import time

import torch

from tier5_modmul import (
    AbacusDecoder, build_prime_pool, eval_answer, make_vocab, pick_device,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=100, help="problems (scorer uses 100/tier)")
    ap.add_argument("--primes", type=int, default=5, help="distinct primes (scorer uses 5)")
    ap.add_argument("--chunks", type=int, nargs="+", default=[32, 64, 128],
                    help="decode sub-batch sizes to sweep")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--amp", action="store_true", help="bf16 autocast on CUDA")
    args = ap.parse_args()

    device = pick_device()
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ck["config"]
    B = cfg["base"]
    V = make_vocab(B)
    max_len, abmax = cfg["max_len"], cfg["abacus_max"]
    model = AbacusDecoder(V["VOCAB"], max_len, abmax, d_model=cfg["d_model"],
                          nhead=cfg["nhead"], num_layers=cfg["layers"],
                          dim_ff=cfg["dim_ff"]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"loaded {args.ckpt} | step {ck.get('step','?')} best {ck.get('best',-1):.3f} "
          f"| base {B} | max_len {max_len} | params {n_params:,} | device {device}")

    use_amp = args.amp and device.type == "cuda"
    amp_ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
               if use_amp else contextlib.nullcontext())

    # Scorer shape: a handful of distinct primes near the top of the tier-5 range.
    pool_rng = random.Random(args.seed + 777)
    hard_lo = cfg["p_max"] - cfg["p_max"] // 8        # top eighth of [.., 2^64)
    primes = build_prime_pool(hard_lo, cfg["p_max"], args.primes, pool_rng)
    print(f"primes ({len(primes)}, near 2^64): {primes}")

    for chunk in args.chunks:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        rng = random.Random(args.seed)        # same sample across chunk sizes
        t0 = time.time()
        with amp_ctx:
            acc = eval_answer(model, primes, B, V, args.n, max_len, abmax,
                              rng, device, chunk=chunk)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0
        peak = (torch.cuda.max_memory_allocated() / 1e9
                if device.type == "cuda" else float("nan"))
        print(f"chunk {chunk:4d} | {args.n} probs | {dt:7.2f}s total | "
              f"{1000*dt/args.n:7.1f} ms/prob | acc {acc:.3f} | peak {peak:.1f} GB")

    print("\nBudget gate: tier-5 must fit a slice of 300s TOTAL (with tiers 1-4). "
          "If total >> ~150s here, KV-cache is mandatory before any submit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
