"""Synthetic data for learning ``x * y mod p`` (the reduced core of the task).

The harness reduces operands ``a, b`` modulo ``p`` for us inside the model
(``a' = a % p`` etc. — the blessed two-args-at-a-time step), so the *learnable*
problem is purely ``(x, y, p) -> x * y mod p`` with ``x, y in [0, p)``.

Eval primes are drawn from finite, enumerable pools (see ``modchallenge.config``
and ``testgen/primes.py``: ``lo = 2**min_bits``, ``hi = 2**max_bits``):

    tier 1 : fixed {2, 3, 5, 7}
    tier 2 : primes in [2**4,  2**8)  = [16,   256)     (~42 primes)
    tier 3 : primes in [2**9,  2**16) = [512, 65536)    (~6000 primes)

Because eval can only ever draw from these pools, training the network to compute
``x * y mod p`` for the *whole* pool is compliant (genuine learned modular
multiplication, not a lookup table). For tier 2 that means full coverage is
feasible; for tier 3 we must generalise across residues within each prime.

Encoding: fixed-width ``WIDTH`` decimal digits, MSB-first, zero-padded. ``WIDTH=5``
covers every value < 2**16 = 65536, i.e. tiers 1-3.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch

# 5 decimal digits cover every residue/prime/answer below 65536 (tiers 1-3).
WIDTH = 5

# Tier prime ranges, matching testgen/primes.py exactly (lo = 2**min_bits).
TIER_PRIME_RANGES: dict[int, tuple[int, int]] = {
    2: (2 ** 4, 2 ** 8),    # [16, 256)
    3: (2 ** 9, 2 ** 16),   # [512, 65536)
}
TIER1_FIXED_PRIMES: tuple[int, ...] = (2, 3, 5, 7)


def sieve_primes(limit: int) -> list[int]:
    """All primes < ``limit`` via a simple sieve of Eratosthenes."""
    if limit < 3:
        return []
    is_p = bytearray([1]) * limit
    is_p[0] = is_p[1] = 0
    for i in range(2, int(limit ** 0.5) + 1):
        if is_p[i]:
            is_p[i * i :: i] = bytearray(len(is_p[i * i :: i]))
    return [i for i in range(2, limit) if is_p[i]]


def primes_for_tier(tier: int) -> list[int]:
    """The full enumerable prime pool a given tier can draw from at eval time."""
    if tier == 1:
        return list(TIER1_FIXED_PRIMES)
    lo, hi = TIER_PRIME_RANGES[tier]
    return [p for p in sieve_primes(hi) if p >= lo]


def digits_fixed(n: int, width: int = WIDTH) -> list[int]:
    """Non-negative int -> fixed-width zero-padded decimal digits, MSB-first."""
    out = [0] * width
    i = width - 1
    while n > 0 and i >= 0:
        out[i] = n % 10
        n //= 10
        i -= 1
    return out


@dataclass
class PrimeSplit:
    """Train / held-out-prime pools per tier.

    ``train`` primes are seen during training; ``val_primes`` are held out to
    measure generalisation to *unseen* primes. For tiers where full coverage is
    the goal (tier 2), pass ``holdout_frac=0`` so every prime is in ``train``.
    """

    train: dict[int, list[int]]
    val_primes: dict[int, list[int]]


def build_prime_split(
    tiers: tuple[int, ...],
    holdout_frac: float = 0.1,
    seed: int = 0,
) -> PrimeSplit:
    """Bucket each tier's prime pool into train / held-out-prime sets.

    Tier 1's four primes are fixed and known, so they always stay fully in train.
    """
    rng = random.Random(seed)
    train: dict[int, list[int]] = {}
    val: dict[int, list[int]] = {}
    for t in tiers:
        pool = primes_for_tier(t)
        if t == 1 or holdout_frac <= 0:
            train[t] = list(pool)
            val[t] = list(pool)
            continue
        shuffled = pool[:]
        rng.shuffle(shuffled)
        k = max(1, int(len(shuffled) * holdout_frac))
        val[t] = sorted(shuffled[:k])
        train[t] = sorted(shuffled[k:])
    return PrimeSplit(train=train, val_primes=val)


# Probability of injecting an edge operand (0 or 1) per slot — eval includes
# a=0, b=0, a=1, b=1 as explicit edge cases, so we over-sample them slightly.
EDGE_PROB = 0.05


def _sample_operand(p: int, rng: random.Random) -> int:
    """A residue in [0, p), with a small chance of an edge value (0 or 1)."""
    if rng.random() < EDGE_PROB:
        return rng.choice((0, 1)) % p
    return rng.randrange(p)


def make_batch(
    pools: dict[int, list[int]],
    tier_weights: dict[int, float],
    batch_size: int,
    rng: random.Random,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample a training batch of (x_digits, y_digits, p_digits, ans_digits).

    ``pools`` maps tier -> list of primes to sample from; ``tier_weights`` maps
    tier -> sampling weight (need not be normalised).
    """
    tiers = list(pools.keys())
    weights = [tier_weights[t] for t in tiers]
    x_rows, y_rows, p_rows, ans_rows = [], [], [], []
    for _ in range(batch_size):
        t = rng.choices(tiers, weights=weights, k=1)[0]
        pool = pools[t]
        p = pool[rng.randrange(len(pool))]
        x = _sample_operand(p, rng)
        y = _sample_operand(p, rng)
        ans = (x * y) % p
        x_rows.append(digits_fixed(x))
        y_rows.append(digits_fixed(y))
        p_rows.append(digits_fixed(p))
        ans_rows.append(digits_fixed(ans))
    t_ = lambda r: torch.tensor(r, dtype=torch.long, device=device)
    return t_(x_rows), t_(y_rows), t_(p_rows), t_(ans_rows)


def make_eval_batch(
    primes: list[int],
    n: int,
    rng: random.Random,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Freshly sampled (unseen residue pairs) eval batch over the given primes."""
    x_rows, y_rows, p_rows, ans_rows = [], [], [], []
    for _ in range(n):
        p = primes[rng.randrange(len(primes))]
        x = rng.randrange(p)
        y = rng.randrange(p)
        ans = (x * y) % p
        x_rows.append(digits_fixed(x))
        y_rows.append(digits_fixed(y))
        p_rows.append(digits_fixed(p))
        ans_rows.append(digits_fixed(ans))
    t_ = lambda r: torch.tensor(r, dtype=torch.long, device=device)
    return t_(x_rows), t_(y_rows), t_(p_rows), t_(ans_rows)


# ---------------------------------------------------------------------------
# Fixed finite dataset (the grokking recipe: overfit a fixed set -> generalise)
# ---------------------------------------------------------------------------

@dataclass
class FixedDataset:
    """A fixed set of (x, y, p, ans) rows held constant across the whole run.

    Grokking (sudden generalisation) needs a *finite* train set the model can
    overfit, not the infinite fresh stream ``make_batch`` provides. ``held_pairs``
    records, per prime, the residue pairs that are IN the train set so eval can
    sample *unseen* pairs on the same primes (within-prime generalisation).
    """

    X: torch.Tensor   # (M, WIDTH)
    Y: torch.Tensor
    P: torch.Tensor
    ANS: torch.Tensor
    held_pairs: dict[int, set]


def build_fixed_dataset(
    primes: list[int],
    per_prime: int,
    seed: int,
    device: torch.device,
) -> FixedDataset:
    """Pre-generate ``per_prime`` distinct (x, y) residue pairs for each prime.

    For small primes (p*p <= per_prime) the full multiplication table is used;
    for large primes a random distinct sample of size ``per_prime`` is drawn.
    """
    rng = random.Random(seed)
    x_rows, y_rows, p_rows, ans_rows = [], [], [], []
    held: dict[int, set] = {}
    for p in primes:
        pairs: set = set()
        full = p * p
        if full <= per_prime:
            pairs = {(x, y) for x in range(p) for y in range(p)}
        else:
            while len(pairs) < per_prime:
                pairs.add((rng.randrange(p), rng.randrange(p)))
        held[p] = pairs
        for (x, y) in pairs:
            ans = (x * y) % p
            x_rows.append(digits_fixed(x))
            y_rows.append(digits_fixed(y))
            p_rows.append(digits_fixed(p))
            ans_rows.append(digits_fixed(ans))
    t_ = lambda r: torch.tensor(r, dtype=torch.long, device=device)
    # Shuffle row order once so batches are well-mixed across primes.
    perm = list(range(len(x_rows)))
    rng.shuffle(perm)
    X, Y, P, A = t_(x_rows)[perm], t_(y_rows)[perm], t_(p_rows)[perm], t_(ans_rows)[perm]
    return FixedDataset(X=X, Y=Y, P=P, ANS=A, held_pairs=held)


def sample_fixed_batch(ds: FixedDataset, batch_size: int, rng: random.Random):
    """Draw a random minibatch (with replacement) from the fixed dataset."""
    m = ds.X.shape[0]
    idx = torch.tensor([rng.randrange(m) for _ in range(batch_size)], device=ds.X.device)
    return ds.X[idx], ds.Y[idx], ds.P[idx], ds.ANS[idx]


def make_unseen_pair_eval(
    ds: FixedDataset,
    primes: list[int],
    n: int,
    rng: random.Random,
    device: torch.device,
):
    """Eval batch of residue pairs NOT in the fixed train set (same primes).

    Measures within-prime generalisation — the grokking signal.
    """
    x_rows, y_rows, p_rows, ans_rows = [], [], [], []
    tries = 0
    while len(x_rows) < n and tries < n * 50:
        tries += 1
        p = primes[rng.randrange(len(primes))]
        x, y = rng.randrange(p), rng.randrange(p)
        if (x, y) in ds.held_pairs.get(p, ()):  # skip seen pairs
            continue
        ans = (x * y) % p
        x_rows.append(digits_fixed(x))
        y_rows.append(digits_fixed(y))
        p_rows.append(digits_fixed(p))
        ans_rows.append(digits_fixed(ans))
    t_ = lambda r: torch.tensor(r, dtype=torch.long, device=device)
    return t_(x_rows), t_(y_rows), t_(p_rows), t_(ans_rows)
