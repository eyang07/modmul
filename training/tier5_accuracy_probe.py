"""Accuracy probe: what tier-5 score does the OFFICIAL scorer's prime draw give?

The budget probe showed time fits (~125s/100). The binding risk is now accuracy.
testgen/primes.py draws each tier-5 prime as nextprime(randrange(2^33, 2^64)) --
UNIFORM IN VALUE, so ~50% of tier-5 primes are exactly 64-bit and ~75% are 63-64
bit: the scorer's tier 5 lives near the 2^64 ceiling. The scorer then uses only
5 primes/tier (100 problems => 20/prime), so the tier score has real small-sample
variance.

This samples K primes the SAME way the scorer does (uniform in value over
[2^33, 2^64), via the trainer's Miller-Rabin rejection sampler -- same bit-size
distribution as nextprime), runs M problems per prime with the SAME greedy decode
as the submission, and reports:
  * per-prime accuracy (+ bit length) -- shows the spread that drives variance,
  * overall mean -- the best estimate of EXPECTED tier-5 score,
  * a bootstrap over 5-prime x 20-problem draws -> P(tier >= 0.90), the number
    that actually decides whether htop90=5 is real or a coin flip.

Per-prime lines flush as they finish, so you can Ctrl-C once the picture is clear.

Usage (on the 5090, after git pull):
    python training/tier5_accuracy_probe.py \
        --ckpt training/checkpoints/modmul_t5_b16_d512_best.pt --amp \
        --primes 25 --per-prime 40
"""

from __future__ import annotations

import argparse
import contextlib
import random

import torch

from tier5_modmul import (
    AbacusDecoder, build_prime_pool, eval_answer, make_vocab, pick_device,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--primes", type=int, default=25, help="distinct primes to test")
    ap.add_argument("--per-prime", type=int, default=40, help="problems per prime")
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gate", type=float, default=0.90)
    ap.add_argument("--tier-primes", type=int, default=5, help="primes/tier in scorer")
    ap.add_argument("--tier-problems", type=int, default=100, help="problems/tier in scorer")
    ap.add_argument("--boot", type=int, default=20000, help="bootstrap resamples")
    ap.add_argument("--amp", action="store_true")
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
    print(f"loaded {args.ckpt} | step {ck.get('step','?')} best {ck.get('best',-1):.3f} "
          f"| base {B} | max_len {max_len} | device {device}", flush=True)

    use_amp = args.amp and device.type == "cuda"
    amp_ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
               if use_amp else contextlib.nullcontext())

    # Scorer-faithful prime draw: uniform in value over [2^33, 2^64) -> same
    # bit-size distribution (heavily 63-64 bit) as nextprime(randrange(...)).
    pool_rng = random.Random(args.seed + 777)
    primes = build_prime_pool(cfg["p_min"], cfg["p_max"], args.primes, pool_rng)

    per_prime = []
    rng = random.Random(args.seed)
    print(f"\ntesting {len(primes)} primes x {args.per_prime} problems "
          f"(uniform-in-value draw, ~scorer distribution):", flush=True)
    for k, p in enumerate(primes):
        with amp_ctx:
            acc = eval_answer(model, [p], B, V, args.per_prime, max_len, abmax,
                              rng, device, chunk=args.chunk)
        per_prime.append(acc)
        print(f"  prime {k+1:3d}/{len(primes)} | {p.bit_length()}-bit | "
              f"acc {acc:.3f} | running mean {sum(per_prime)/len(per_prime):.3f}",
              flush=True)

    mean = sum(per_prime) / len(per_prime)
    print(f"\noverall mean accuracy (expected tier-5 score): {mean:.4f} "
          f"over {len(primes)*args.per_prime} problems", flush=True)

    # Bootstrap the scorer's actual draw: pick `tier_primes` primes (with
    # replacement) from our measured set, weight each by tier_problems/tier_primes,
    # treat each prime's measured acc as its success rate -> simulate the tier score.
    boot_rng = random.Random(args.seed + 999)
    ppd = args.tier_problems // args.tier_primes
    ge_gate = 0
    scores = []
    for _ in range(args.boot):
        chosen = [per_prime[boot_rng.randrange(len(per_prime))]
                  for _ in range(args.tier_primes)]
        # binomial draw of correct answers per prime at its measured rate
        correct = sum(sum(1 for _ in range(ppd) if boot_rng.random() < a)
                      for a in chosen)
        s = correct / (ppd * args.tier_primes)
        scores.append(s)
        if s >= args.gate:
            ge_gate += 1
    scores.sort()
    p_pass = ge_gate / args.boot
    lo = scores[int(0.05 * len(scores))]
    hi = scores[int(0.95 * len(scores))]
    print(f"bootstrap tier score: median {scores[len(scores)//2]:.3f} | "
          f"90% CI [{lo:.3f}, {hi:.3f}]", flush=True)
    print(f"P(tier-5 >= {args.gate:.2f}) ~= {p_pass:.2f}", flush=True)
    print("\nDecision: P(pass) >> 0.5 => integrate & ship; ~0.5 => coin flip "
          "(more training or bank tier 4); << 0.5 => bank htop90=4.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
