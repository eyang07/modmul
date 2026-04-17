"""Scorer: compare model predictions against ground truth.

Reads per-tier output files from the sandbox and scores them.
Only fully completed tiers (with tier_complete marker) are scored.
"""

from __future__ import annotations

import json
from pathlib import Path

from modchallenge.config import NUM_TIERS
from modchallenge.evaluation.results import EvalResult, TierResult
from modchallenge.testgen.generator import FullTestSet


def is_canonical_decimal(s: str) -> bool:
    """Check if a string is a canonical decimal integer (no leading zeros except '0')."""
    if not s:
        return False
    if not s.isdigit():
        return False
    if len(s) > 1 and s[0] == "0":
        return False
    return True


def score_tier_from_files(
    tier_id: int,
    output_file: Path,
    expected: list[str],
) -> TierResult:
    """Score a single tier by reading the output JSONL file.

    The output file is written by entrypoint.py inside the sandbox.
    Each line is a JSON object with a "result" key.
    The last line should be {"tier_complete": true} to mark completion.

    Args:
        tier_id: The tier being scored.
        output_file: Path to the tier's output JSONL file.
        expected: List of expected answer strings in order.

    Returns:
        TierResult with accuracy and completion status.
    """
    if not output_file.exists():
        return TierResult(tier_id=tier_id, total=len(expected), correct=0, completed=False)

    lines = output_file.read_text().strip().splitlines()
    if not lines:
        return TierResult(tier_id=tier_id, total=len(expected), correct=0, completed=False)

    # Check if the tier completed
    completed = False
    prediction_lines = lines
    try:
        last = json.loads(lines[-1])
        if last.get("tier_complete") is True:
            completed = True
            prediction_lines = lines[:-1]
    except json.JSONDecodeError:
        pass

    # If not completed, this tier scores 0
    if not completed:
        return TierResult(tier_id=tier_id, total=len(expected), correct=0, completed=False)

    # Parse predictions and compare
    correct = 0
    for i, exp in enumerate(expected):
        if i >= len(prediction_lines):
            break
        try:
            pred_obj = json.loads(prediction_lines[i])
            pred = pred_obj.get("result", "")
        except (json.JSONDecodeError, AttributeError):
            continue

        if pred == exp:
            correct += 1

    return TierResult(
        tier_id=tier_id,
        total=len(expected),
        correct=correct,
        completed=True,
    )


def score_tier_in_memory(
    tier_id: int,
    predictions: list[str],
    expected: list[str],
) -> TierResult:
    """Score a single tier from in-memory prediction list.

    Used for local evaluation (no sandbox).
    A tier is considered completed only if predictions cover all expected cases.

    Args:
        tier_id: The tier being scored.
        predictions: List of predicted answer strings.
        expected: List of expected answer strings.

    Returns:
        TierResult with accuracy. Incomplete tiers score 0.
    """
    completed = len(predictions) == len(expected) and len(expected) > 0

    if not completed:
        return TierResult(
            tier_id=tier_id,
            total=len(expected),
            correct=0,
            completed=False,
        )

    correct = sum(1 for pred, exp in zip(predictions, expected) if pred == exp)

    return TierResult(
        tier_id=tier_id,
        total=len(expected),
        correct=correct,
        completed=True,
    )


def score_full(
    test_set: FullTestSet,
    output_dir: Path,
) -> EvalResult:
    """Score a full evaluation run from sandbox output files.

    Args:
        test_set: The test set with ground truth.
        output_dir: Directory containing tier_N.jsonl output files.

    Returns:
        EvalResult with per-tier and aggregate scores.
    """
    tier_results = []
    for tier_ts in test_set.tiers:
        expected = [case.expected for case in tier_ts.cases]
        output_file = output_dir / f"tier_{tier_ts.tier_id}.jsonl"
        result = score_tier_from_files(tier_ts.tier_id, output_file, expected)
        tier_results.append(result)

    return EvalResult(tier_results=tier_results)


def score_full_in_memory(
    test_set: FullTestSet,
    all_predictions: dict[int, list[str]],
) -> EvalResult:
    """Score a full evaluation from in-memory predictions.

    Args:
        test_set: The test set with ground truth.
        all_predictions: Dict mapping tier_id -> list of predicted strings.

    Returns:
        EvalResult with per-tier and aggregate scores.
    """
    tier_results = []
    for tier_ts in test_set.tiers:
        expected = [case.expected for case in tier_ts.cases]
        predictions = all_predictions.get(tier_ts.tier_id, [])
        result = score_tier_in_memory(tier_ts.tier_id, predictions, expected)
        tier_results.append(result)

    return EvalResult(tier_results=tier_results)
