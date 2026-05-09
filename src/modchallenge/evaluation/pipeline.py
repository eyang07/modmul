"""Main evaluation pipeline.

Orchestrates: load model -> generate tests -> run inference -> score.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from modchallenge.config import EvalConfig
from modchallenge.evaluation.loader import (
    check_artifact_size,
    load_model,
    validate_manifest,
)
from modchallenge.evaluation.results import EvalResult
from modchallenge.evaluation.scorer import score_full_in_memory
from modchallenge.interface.base_model import ModularMultiplicationModel
from modchallenge.testgen.generator import (
    FullTestSet,
    generate_private_test_set,
)

logger = logging.getLogger(__name__)


def run_inference(
    model: ModularMultiplicationModel,
    test_set: FullTestSet,
    timeout_seconds: float = 300,
) -> dict[int, list[str]]:
    """Run model inference on the full test set.

    Timeout is cooperative: checked between batches.
    Only fully completed tiers are recorded. If timeout hits mid-tier,
    that tier's partial results are discarded.

    Returns dict mapping tier_id -> list of prediction strings.
    """
    start_time = time.monotonic()
    batch_size = max(1, model.max_batch_size())
    all_predictions: dict[int, list[str]] = {}

    for tier_ts in test_set.tiers:
        elapsed = time.monotonic() - start_time
        if elapsed >= timeout_seconds:
            logger.warning(
                "Timeout reached (%.1fs). Skipping tier %d and beyond.",
                elapsed, tier_ts.tier_id,
            )
            break

        tier_predictions: list[str] = []
        inputs = [(c.a, c.b, c.p) for c in tier_ts.cases]
        tier_complete = True

        for batch_start in range(0, len(inputs), batch_size):
            elapsed = time.monotonic() - start_time
            if elapsed >= timeout_seconds:
                logger.warning(
                    "Timeout during tier %d batch %d.",
                    tier_ts.tier_id, batch_start,
                )
                tier_complete = False
                break

            batch = inputs[batch_start : batch_start + batch_size]
            try:
                results = model.predict_batch(batch)
                if len(results) != len(batch):
                    logger.warning(
                        "Tier %d batch %d: predict_batch returned %d results, expected %d. "
                        "Marking tier incomplete.",
                        tier_ts.tier_id, batch_start, len(results), len(batch),
                    )
                    tier_complete = False
                    break
                tier_predictions.extend(results)
            except Exception as e:
                logger.error(
                    "Error in tier %d batch %d: %s", tier_ts.tier_id, batch_start, e
                )
                tier_predictions.extend([""] * len(batch))

        if tier_complete:
            all_predictions[tier_ts.tier_id] = tier_predictions
            logger.info(
                "Tier %d complete: %d predictions in %.1fs",
                tier_ts.tier_id, len(tier_predictions),
                time.monotonic() - start_time,
            )
        else:
            logger.warning("Tier %d incomplete, scored as 0.", tier_ts.tier_id)

    return all_predictions


def check_determinism(
    model: ModularMultiplicationModel,
    test_set: FullTestSet,
    num_checks: int = 10,
) -> bool:
    """Spot-check determinism by running random problems twice via predict_batch().

    Returns True if all checked predictions are identical across two runs.
    """
    import random as _rng

    all_cases: list[tuple[str, str, str]] = []
    for tier_ts in test_set.tiers:
        for c in tier_ts.cases:
            all_cases.append((c.a, c.b, c.p))

    if not all_cases:
        return True

    sample_size = min(num_checks, len(all_cases))
    sample = _rng.Random(42).sample(all_cases, sample_size)

    batch_size = max(1, model.max_batch_size())

    results_a: list[str] = []
    results_b: list[str] = []
    for i in range(0, len(sample), batch_size):
        batch = sample[i : i + batch_size]
        ra = model.predict_batch(batch)
        rb = model.predict_batch(batch)

        if len(ra) != len(batch) or len(rb) != len(batch):
            logger.warning(
                "predict_batch returned wrong length: expected %d, got %d and %d",
                len(batch), len(ra), len(rb),
            )
            return False

        results_a.extend(ra)
        results_b.extend(rb)

    for i, (ra, rb) in enumerate(zip(results_a, results_b)):
        if ra != rb:
            logger.warning(
                "Non-deterministic output on check %d: %r vs %r", i, ra, rb
            )
            return False

    return True


def evaluate_local(
    model_dir: Path,
    master_seed: bytes | None = None,
    config: EvalConfig = EvalConfig(),
) -> EvalResult:
    """Run a full local evaluation.

    Args:
        model_dir: Path to the submission directory.
        master_seed: Secret seed for test generation. None = random.
        config: Evaluation configuration.

    Returns:
        EvalResult with scores and metadata.
    """
    logger.info("Validating submission at %s", model_dir)

    manifest = validate_manifest(model_dir)
    logger.info("Manifest OK: entry_class=%s", manifest.entry_class)

    total_bytes = check_artifact_size(model_dir, config.max_artifact_bytes)
    logger.info("Artifact size: %.2f GB", total_bytes / 1e9)

    logger.info("Generating test set: %d problems", config.total_problems)
    test_set = generate_private_test_set(master_seed=master_seed, config=config)

    logger.info("Loading model...")
    load_start = time.monotonic()
    model = load_model(model_dir, manifest)
    load_time = time.monotonic() - load_start
    logger.info("Model loaded in %.1fs", load_time)

    det_start = time.monotonic()
    is_deterministic = check_determinism(model, test_set)
    det_time = time.monotonic() - det_start
    if not is_deterministic:
        logger.warning("Model is NON-DETERMINISTIC. Results will not be ranked.")
    logger.info("Determinism check in %.1fs", det_time)

    logger.info("Running inference (%.1fs budget)...", config.timeout_seconds)
    predictions = run_inference(model, test_set, timeout_seconds=config.timeout_seconds)

    result = score_full_in_memory(test_set, predictions)
    result.deterministic = is_deterministic

    logger.info(
        "Evaluation complete: overall_accuracy=%.2f%%",
        result.overall_accuracy * 100,
    )

    return result
