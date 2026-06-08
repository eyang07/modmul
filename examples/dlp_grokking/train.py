"""Train the compliant DLP-grokking model.

Samples synthetic (a mod p, b mod p, p) -> (a*b mod p) examples across many
small primes. Primes are split into a train pool and a held-out val pool, so
the val metric measures generalisation to *unseen primes* — the thing that
distinguishes "learned the field structure" from "memorised one prime".

Usage:
    .venv312/bin/python examples/dlp_grokking/train.py [--minutes 8]
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import torch
import torch.nn as nn

# Import the shared architecture from the sibling model.py.
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from model import DLPGrokNet, WIDTH, _digits_fixed  # noqa: E402


def sieve_primes(limit: int) -> list[int]:
    is_p = bytearray([1]) * limit
    is_p[0] = is_p[1] = 0
    for i in range(2, int(limit ** 0.5) + 1):
        if is_p[i]:
            is_p[i * i :: i] = bytearray(len(is_p[i * i :: i]))
    return [i for i in range(2, limit) if is_p[i]]


# Prime ceiling: covers tiers 1-3 (p < 2^16) and the low end of tier 4.
PRIME_LIMIT = 1 << 16

# Tier bit-ranges -> prime value ranges (see config.TIERS).
#   tier 1: fixed {2,3,5,7}
#   tier 2: 4-8 bit   -> [8, 256)
#   tier 3: 9-16 bit  -> [256, 65536)
# We sample with a curriculum bias toward the small primes the model can
# actually learn and generalise over; large primes get exposure but are
# (expectedly) not learnable to high accuracy from sparse samples.
# Focus capacity where compliant learning is actually possible: tiers 1-2.
# Tier 3 (primes up to 2^16) is not learnable to useful accuracy from sparse
# samples, so it gets only a token weight rather than starving tier 2.
BUCKET_WEIGHTS = (0.22, 0.70, 0.08)  # tier1, tier2, tier3


def split_primes(seed: int = 0) -> tuple[dict, dict]:
    """Bucket primes by tier range and split each bucket into train/val.
    Returns (train_buckets, val_buckets), each a dict tier -> list[int].
    The four tier-1 primes are fixed and known at eval time, so all in train."""
    primes = sieve_primes(PRIME_LIMIT)
    t1 = [2, 3, 5, 7]
    t2 = [p for p in primes if 8 <= p < 256]
    t3 = [p for p in primes if 256 <= p < PRIME_LIMIT]
    rng = random.Random(seed)
    rng.shuffle(t2)
    rng.shuffle(t3)
    train = {1: t1, 2: t2[len(t2) // 10 :], 3: t3[len(t3) // 10 :]}
    val = {1: t1, 2: sorted(t2[: len(t2) // 10]), 3: sorted(t3[: len(t3) // 10])}
    return train, val


def make_batch(buckets: dict, batch_size: int, rng: random.Random, device):
    a_rows, b_rows, p_rows, y_rows = [], [], [], []
    tiers = (1, 2, 3)
    for _ in range(batch_size):
        tier = rng.choices(tiers, weights=BUCKET_WEIGHTS, k=1)[0]
        pool = buckets[tier]
        p = pool[rng.randrange(len(pool))]
        a = rng.randrange(p)
        b = rng.randrange(p)
        ans = (a * b) % p
        a_rows.append(_digits_fixed(a))
        b_rows.append(_digits_fixed(b))
        p_rows.append(_digits_fixed(p))
        y_rows.append(_digits_fixed(ans))
    t = lambda r: torch.tensor(r, dtype=torch.long, device=device)
    return t(a_rows), t(b_rows), t(p_rows), t(y_rows)


@torch.no_grad()
def _exact_on_pool(model, pool, device, n, rng) -> float:
    a, b, p, y = make_batch({1: pool, 2: pool, 3: pool}, n, rng, device)
    pred = model(a, b, p).argmax(dim=-1)
    return (pred == y).all(dim=1).float().mean().item()


@torch.no_grad()
def evaluate(model, buckets, device, n: int = 3000) -> dict:
    """Per-tier exact-match on freshly sampled problems from each bucket."""
    rng = random.Random(12345)
    return {t: _exact_on_pool(model, buckets[t], device, n, rng) for t in (1, 2, 3)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=8.0)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()

    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    print(f"device: {device}")

    train_b, val_b = split_primes()
    print(
        f"train primes: t1={len(train_b[1])} t2={len(train_b[2])} t3={len(train_b[3])} | "
        f"val primes: t2={len(val_b[2])} t3={len(val_b[3])}"
    )

    model = DLPGrokNet().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = int(args.minutes * 60 * 4.0)  # ~4 steps/s budget (bigger net)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(total_steps, 1), eta_min=args.lr * 0.15
    )
    loss_fn = nn.CrossEntropyLoss()
    rng = random.Random(0)

    out_path = HERE / "weights.pt"
    # Score weighted toward the tiers we expect to learn (and that are scored).
    def combined(d):
        return 0.34 * d[1] + 0.5 * d[2] + 0.16 * d[3]

    best = -1.0
    deadline = time.monotonic() + args.minutes * 60
    start = time.monotonic()
    step = 0

    while time.monotonic() < deadline:
        model.train()
        a, b, p, y = make_batch(train_b, args.batch, rng, device)
        logits = model(a, b, p)  # (B, WIDTH, 10)
        loss = loss_fn(logits.reshape(-1, 10), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        step += 1

        if step % 200 == 0:
            model.eval()
            vacc = evaluate(model, val_b, device)   # generalisation (unseen primes)
            tacc = evaluate(model, train_b, device) # fit
            elapsed = time.monotonic() - start
            print(
                f"step {step:5d} | loss {loss.item():.4f} | "
                f"val t1 {vacc[1]:.3f} t2 {vacc[2]:.3f} t3 {vacc[3]:.3f} | "
                f"train t2 {tacc[2]:.3f} t3 {tacc[3]:.3f} | {elapsed:.0f}s"
            )
            score = combined(vacc)
            if score > best:
                best = score
                torch.save(
                    {"state_dict": model.state_dict(), "config": model.config},
                    out_path,
                )

    print(f"done. best combined val={best:.3f}. saved -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
