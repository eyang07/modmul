"""Tests for the scorer."""

import json
import tempfile
from pathlib import Path

from modchallenge.config import EvalConfig
from modchallenge.evaluation.results import TierResult, EvalResult
from modchallenge.evaluation.scorer import (
    is_canonical_decimal,
    score_tier_from_files,
    score_tier_in_memory,
    score_full_in_memory,
)
from modchallenge.testgen.generator import generate_private_test_set


def test_is_canonical_decimal():
    assert is_canonical_decimal("0")
    assert is_canonical_decimal("42")
    assert is_canonical_decimal("12345678901234567890")
    assert not is_canonical_decimal("")
    assert not is_canonical_decimal("042")  # leading zero
    assert not is_canonical_decimal("-1")
    assert not is_canonical_decimal("3.14")
    assert not is_canonical_decimal(" 5")


def test_tier_result_accuracy():
    r = TierResult(tier_id=0, total=100, correct=90, completed=True)
    assert r.accuracy == 0.9

    r2 = TierResult(tier_id=0, total=0, correct=0, completed=True)
    assert r2.accuracy == 0.0


def test_score_tier_in_memory_perfect():
    expected = ["0", "42", "100"]
    predictions = ["0", "42", "100"]
    result = score_tier_in_memory(0, predictions, expected)
    assert result.correct == 3
    assert result.accuracy == 1.0


def test_score_tier_in_memory_partial():
    expected = ["0", "42", "100"]
    predictions = ["0", "99", "100"]
    result = score_tier_in_memory(0, predictions, expected)
    assert result.correct == 2


def test_score_tier_from_files_complete():
    expected = ["0", "42", "100"]
    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = Path(tmpdir) / "tier_0.jsonl"
        with open(output_file, "w") as f:
            for val in ["0", "42", "100"]:
                f.write(json.dumps({"result": val}) + "\n")
            f.write(json.dumps({"tier_complete": True}) + "\n")

        result = score_tier_from_files(0, output_file, expected)
        assert result.completed
        assert result.correct == 3


def test_score_tier_from_files_incomplete():
    """Incomplete tier (no marker) should score 0."""
    expected = ["0", "42", "100"]
    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = Path(tmpdir) / "tier_0.jsonl"
        with open(output_file, "w") as f:
            for val in ["0", "42"]:
                f.write(json.dumps({"result": val}) + "\n")
            # No tier_complete marker

        result = score_tier_from_files(0, output_file, expected)
        assert not result.completed
        assert result.correct == 0


def test_score_tier_from_files_missing():
    """Missing output file should score 0."""
    result = score_tier_from_files(0, Path("/nonexistent"), ["0", "42"])
    assert not result.completed
    assert result.correct == 0


def test_eval_result_overall_accuracy():
    """overall_accuracy only counts scored tiers (1+), not tier 0."""
    result = EvalResult(tier_results=[
        TierResult(tier_id=0, total=10, correct=10, completed=True),  # diagnostic, excluded
        TierResult(tier_id=1, total=10, correct=5, completed=True),
        TierResult(tier_id=2, total=10, correct=0, completed=False),
    ])
    # Scored tiers: tier 1 (50%) + tier 2 incomplete (0%)
    # Overall = (0.5 + 0.0) / 2 = 0.25
    assert abs(result.overall_accuracy - 0.25) < 0.001


def test_eval_result_diagnostic_tier():
    """Tier 0 should be accessible as diagnostic_tier."""
    result = EvalResult(tier_results=[
        TierResult(tier_id=0, total=10, correct=10, completed=True),
        TierResult(tier_id=1, total=10, correct=9, completed=True),
    ])
    assert result.diagnostic_tier is not None
    assert result.diagnostic_tier.accuracy == 1.0
    assert len(result.scored_tiers) == 1


def test_score_full_in_memory():
    """End-to-end: generate test set, simulate perfect predictions, score."""
    config = EvalConfig(total_problems=110)
    ts = generate_private_test_set(master_seed=b"test" * 8, config=config)

    # Perfect predictions
    predictions = {}
    for tier_ts in ts.tiers:
        predictions[tier_ts.tier_id] = [c.expected for c in tier_ts.cases]

    result = score_full_in_memory(ts, predictions)
    assert result.overall_accuracy == 1.0
