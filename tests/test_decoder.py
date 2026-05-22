"""Tests for the answer decoder."""

from __future__ import annotations

import pytest

from modchallenge.evaluation.decoder import (
    MalformedOutput,
    decode_answer,
    encode_answer,
    resolve_base,
)


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "value,base,prime",
    [
        (0, 10, 7),
        (1, 10, 7),
        (6, 10, 7),       # max value for prime=7
        (52, 10, 97),
        (123456789, 10, 1_000_000_007),
        (255, 256, 257),
        (0xdeadbeef, 16, 0xffffffff),
        (0, 2, 5),
        (4, 2, 5),
    ],
)
def test_roundtrip_decimal_bases(value: int, base: int, prime: int) -> None:
    digits = encode_answer(value, base=base, prime=prime)
    decoded = decode_answer(digits, base=base, prime=prime)
    assert decoded == value


def test_roundtrip_base_p_sentinel() -> None:
    """base = "p" should resolve to the current prime."""
    prime = 97
    value = 52
    digits = encode_answer(value, base="p", prime=prime)
    assert digits == [52]  # single digit in base-97
    decoded = decode_answer(digits, base="p", prime=prime)
    assert decoded == value


def test_roundtrip_with_explicit_width_pads_zeros() -> None:
    """encode_answer pads with leading zeros up to requested width."""
    digits = encode_answer(7, base=10, prime=997, width=3)
    assert digits == [0, 0, 7]
    # Decoder ignores leading zeros — value is still 7.
    assert decode_answer(digits, base=10, prime=997) == 7


# ---------------------------------------------------------------------------
# Tier 0 relaxation
# ---------------------------------------------------------------------------

def test_tier_zero_allows_value_above_prime() -> None:
    """Tier 0 is pure multiplication, no modular reduction — value can be > p."""
    huge_value = 12345 * 67890  # 838102050
    prime = 97  # unused for the check, but provided since signature requires it
    digits = encode_answer(huge_value, base=10, prime=prime)
    # Tier 0 path: should decode fine.
    decoded = decode_answer(digits, base=10, prime=prime, is_tier_zero=True)
    assert decoded == huge_value


def test_scored_tier_rejects_value_above_prime() -> None:
    """On scored tiers, a value >= p is malformed."""
    digits = encode_answer(100, base=10, prime=97)
    with pytest.raises(MalformedOutput, match="value 100 >= prime 97"):
        decode_answer(digits, base=10, prime=97, is_tier_zero=False)


# ---------------------------------------------------------------------------
# Malformed input
# ---------------------------------------------------------------------------

def test_non_list_digits_rejected() -> None:
    with pytest.raises(MalformedOutput, match="list"):
        decode_answer("123", base=10, prime=997)


def test_non_int_digit_rejected() -> None:
    with pytest.raises(MalformedOutput, match=r"position 1.*str"):
        decode_answer([1, "2", 3], base=10, prime=997)


def test_bool_digit_rejected() -> None:
    """bool is technically int but rejected to avoid surprise."""
    with pytest.raises(MalformedOutput, match=r"position 0.*bool"):
        decode_answer([True, 0], base=10, prime=997)


def test_digit_above_base_rejected() -> None:
    with pytest.raises(MalformedOutput, match=r"out of range \[0, 10\)"):
        decode_answer([1, 2, 10], base=10, prime=997)


def test_negative_digit_rejected() -> None:
    with pytest.raises(MalformedOutput, match=r"out of range"):
        decode_answer([-1, 0], base=10, prime=997)


def test_empty_digit_list_decodes_to_zero() -> None:
    """An empty list represents the number 0."""
    assert decode_answer([], base=10, prime=997) == 0


# ---------------------------------------------------------------------------
# Base validation
# ---------------------------------------------------------------------------

def test_invalid_base_string_rejected() -> None:
    with pytest.raises(MalformedOutput, match="invalid output_base string"):
        decode_answer([1, 2, 3], base="x", prime=997)


def test_base_too_small_rejected() -> None:
    with pytest.raises(MalformedOutput, match=r"out of range \[2,"):
        decode_answer([0], base=1, prime=997)


def test_base_too_large_rejected() -> None:
    too_big = 2**32 + 1
    with pytest.raises(MalformedOutput, match="out of range"):
        decode_answer([0], base=too_big, prime=997)


def test_base_bool_rejected() -> None:
    with pytest.raises(MalformedOutput, match="must be int or string"):
        decode_answer([0], base=True, prime=997)


# ---------------------------------------------------------------------------
# resolve_base helper
# ---------------------------------------------------------------------------

def test_resolve_base_p_sentinel() -> None:
    assert resolve_base("p", prime=97) == 97


def test_resolve_base_integer() -> None:
    assert resolve_base(16, prime=997) == 16


def test_resolve_base_p_sentinel_with_invalid_prime() -> None:
    with pytest.raises(MalformedOutput):
        resolve_base("p", prime=1)
