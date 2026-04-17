"""Prime generation for test cases.

Supports primes from tiny (2, 3, 5, 7) up to ~2^2048.
Uses sympy.nextprime for reliable primality across all sizes.
"""

from __future__ import annotations

import hashlib
import hmac
import random

from sympy import nextprime

from modchallenge.config import TierConfig


def generate_primes_for_tier(
    tier: TierConfig,
    count: int,
    seed: bytes,
) -> list[int]:
    """Generate `count` distinct primes for the given tier.

    For tier 0 (fixed_primes), returns those primes directly.
    For other tiers, generates random primes in [2^min_bits, 2^max_bits).

    Args:
        tier: Tier configuration.
        count: Number of distinct primes to generate.
        seed: Seed bytes for deterministic generation.

    Returns:
        List of distinct primes.
    """
    if tier.fixed_primes:
        # Tier 0: use the fixed small primes, cycle if count > len
        primes = []
        for i in range(count):
            primes.append(tier.fixed_primes[i % len(tier.fixed_primes)])
        return primes

    # Derive a deterministic RNG from the seed + tier_id
    tier_seed = hmac.new(seed, f"primes-tier-{tier.tier_id}".encode(), hashlib.sha256).digest()
    rng = random.Random(tier_seed)

    primes: list[int] = []
    seen: set[int] = set()

    lo = 2 ** tier.min_bits
    hi = 2 ** tier.max_bits

    while len(primes) < count:
        # Pick a random number in [lo, hi) and find the next prime
        candidate = rng.randrange(lo, hi)
        p = nextprime(candidate)
        # nextprime might overshoot the range; if so, try from the bottom
        if p >= hi:
            p = nextprime(lo)
        if p not in seen:
            seen.add(p)
            primes.append(p)

    return primes
