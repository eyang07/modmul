"""Clean high-n held-out eval + decode timing for a tier-4 modmul checkpoint.

Run on the pod after training finishes (short, paste-safe):
    PYTHONPATH=training python training/eval_t4.py
    PYTHONPATH=training python training/eval_t4.py --ckpt training/checkpoints/modmul_t4_v2_best.pt --n 500

Reports per-bucket final-answer accuracy over fresh held-out problems (low noise at
n=500, ~+/-0.02) and wall-clock decode time per bucket so we can size the 300s budget.
"""

from __future__ import annotations

import argparse
import random
import time

import torch

import modmul_probe as M


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="training/checkpoints/modmul_t4_v2_best.pt")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--pool-size", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    device = M.pick_device()
    ck = torch.load(args.ckpt, map_location=device)
    c = ck["config"]
    m = M.AbacusDecoder(max_len=c["max_len"], abacus_max=c["abacus_max"],
                        d_model=c["d_model"], nhead=c["nhead"],
                        num_layers=c["layers"], dim_ff=c["dim_ff"]).to(device)
    m.load_state_dict(ck["model"]); m.eval()
    print(f"loaded {args.ckpt} | step {ck.get('step')} | best {ck.get('best')}")
    print(f"max_len {c['max_len']} | abacus_max {c['abacus_max']} | n {args.n}/bucket")

    buckets = [(2**17, 2**22), (2**22, 2**27), (2**27, 2**32)]
    for lo, hi in buckets:
        ps = M.build_prime_pool(lo, hi, args.pool_size, random.Random(1))
        rng = random.Random(args.seed)
        t0 = time.time()
        acc = M.eval_answer(m, ps, args.n, c["max_len"], c["abacus_max"], rng, device)
        dt = time.time() - t0
        print(f"[{lo}-{hi}) acc {acc:.3f} | {dt:6.1f}s for {args.n} "
              f"({1000*dt/args.n:5.1f} ms/problem)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
