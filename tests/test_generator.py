"""Tests for test case generation."""

from modchallenge.config import EvalConfig, PublicBenchmarkConfig, TIERS, NUM_TIERS
from modchallenge.testgen.generator import (
    generate_private_test_set,
    generate_public_test_set,
)
from modchallenge.testgen.primes import generate_primes_for_tier


def test_generate_primes_tier1():
    """Tier 1 should return the fixed small primes."""
    primes = generate_primes_for_tier(TIERS[1], count=4, seed=b"test")
    assert primes == [2, 3, 5, 7]


def test_generate_primes_tier2():
    """Tier 2 primes should be in [2^4, 2^8) range."""
    primes = generate_primes_for_tier(TIERS[2], count=5, seed=b"test")
    assert len(primes) == 5
    for p in primes:
        assert 2**4 <= p < 2**8, f"Prime {p} out of range for tier 2"


def test_generate_primes_deterministic():
    """Same seed should produce same primes."""
    a = generate_primes_for_tier(TIERS[4], count=5, seed=b"seed1")
    b = generate_primes_for_tier(TIERS[4], count=5, seed=b"seed1")
    assert a == b


def test_generate_primes_different_seeds():
    """Different seeds should produce different primes."""
    a = generate_primes_for_tier(TIERS[4], count=5, seed=b"seed1")
    b = generate_primes_for_tier(TIERS[4], count=5, seed=b"seed2")
    assert a != b


def test_private_test_set():
    """Generate a private test set and verify structure."""
    config = EvalConfig(total_problems=110)
    ts = generate_private_test_set(master_seed=b"x" * 32, config=config)

    assert len(ts.tiers) == NUM_TIERS  # 11
    for tier_ts in ts.tiers:
        assert len(tier_ts.cases) == 10  # 110 / 11 tiers


def test_private_test_set_correctness():
    """Verify that ground truth answers are actually correct."""
    config = EvalConfig(total_problems=110)
    ts = generate_private_test_set(master_seed=b"y" * 32, config=config)

    for tier_ts in ts.tiers:
        for case in tier_ts.cases:
            a, b, p = int(case.a), int(case.b), int(case.p)
            expected = (a * b) % p
            assert case.expected == str(expected), (
                f"Wrong answer for {a}*{b} mod {p}: "
                f"got {case.expected}, want {expected}"
            )


def test_tier0_pure_multiplication():
    """Tier 0 cases should be pure multiplication (a*b < p, answer = a*b)."""
    config = EvalConfig(total_problems=110)
    ts = generate_private_test_set(master_seed=b"m" * 32, config=config)

    tier0 = ts.tiers[0]
    assert tier0.tier_id == 0
    for case in tier0.cases:
        a, b, p = int(case.a), int(case.b), int(case.p)
        assert a * b < p, f"Tier 0: a*b ({a*b}) should be < p ({p})"
        assert case.expected == str(a * b)
        assert (a * b) % p == a * b  # mod p is identity


def test_modular_tiers_a_b_can_exceed_p():
    """Tiers 1-10: a and b can be larger than p."""
    config = EvalConfig(total_problems=110)
    ts = generate_private_test_set(master_seed=b"a" * 32, config=config)

    found_exceeding = False
    for tier_ts in ts.tiers[1:]:  # skip tier 0
        for case in tier_ts.cases:
            a, b, p = int(case.a), int(case.b), int(case.p)
            if a > p or b > p:
                found_exceeding = True
                break
        if found_exceeding:
            break
    assert found_exceeding, "Expected some cases where a or b > p"


def test_public_test_set():
    """Public test set should be deterministic and have correct structure."""
    ts1 = generate_public_test_set()
    ts2 = generate_public_test_set()

    assert ts1.total_cases == ts2.total_cases == 1100  # 100 * 11 tiers
    for t1, t2 in zip(ts1.tiers, ts2.tiers):
        for c1, c2 in zip(t1.cases, t2.cases):
            assert c1.a == c2.a
            assert c1.b == c2.b
            assert c1.p == c2.p
            assert c1.expected == c2.expected


def test_edge_cases_present():
    """Scored tiers should contain edge cases (a=0, b=0, a=1, b=1)."""
    config = EvalConfig(total_problems=110)
    ts = generate_private_test_set(master_seed=b"z" * 32, config=config)

    for tier_ts in ts.tiers[1:]:  # skip tier 0
        a_values = [c.a for c in tier_ts.cases[:4]]
        b_values = [c.b for c in tier_ts.cases[:4]]
        assert "0" in a_values or "0" in b_values
