"""Main evaluation pipeline.

Orchestrates: validate manifest -> static-check submission -> generate tests
-> load model -> check determinism -> run inference -> decode -> score.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Union

from modchallenge.config import EvalConfig
from modchallenge.evaluation.decoder import MalformedOutput, decode_answer
from modchallenge.evaluation.loader import (
    check_artifact_size,
    load_model,
    validate_manifest,
)
from modchallenge.evaluation.results import EvalResult
from modchallenge.evaluation.scorer import score_full_in_memory
from modchallenge.interface.base_model import ModularMultiplicationModel
from modchallenge.security.static_check import check_submission
from modchallenge.testgen.generator import (
    FullTestSet,
    generate_private_test_set,
)


class StaticCheckError(RuntimeError):
    """Raised when the submission fails the pre-load static-analysis check."""

    def __init__(self, findings: list, message: str) -> None:
        super().__init__(message)
        self.findings = findings


logger = logging.getLogger(__name__)


def _preprocess_batch(
    model: ModularMultiplicationModel,
    batch: list[tuple[str, str, str]],
) -> list[tuple[Any, Any, Any]]:
    """Apply the per-argument preprocess hooks to a batch.

    Each hook may only access its own argument; the loop here enforces that
    by construction (each call takes one argument only).
    """
    return [
        (
            model.preprocess_a(a),
            model.preprocess_b(b),
            model.preprocess_p(p),
        )
        for a, b, p in batch
    ]


def _decode_batch(
    digits_list: list[list[int]],
    raw_inputs: list[tuple[str, str, str]],
    *,
    output_base: Union[int, str],
    is_tier_zero: bool,
) -> list[str]:
    """Decode a batch of digit lists into canonical decimal answer strings.

    Malformed outputs become ``""`` (empty string), which scores 0 for that
    problem. The pipeline does not abort on malformed output — that is per
    the rules.
    """
    out: list[str] = []
    for digits, (_a, _b, p) in zip(digits_list, raw_inputs):
        try:
            value = decode_answer(
                digits,
                base=output_base,
                prime=int(p),
                is_tier_zero=is_tier_zero,
            )
            out.append(str(value))
        except MalformedOutput as exc:
            logger.debug("Malformed output: %s", exc)
            out.append("")
    return out


def run_inference(
    model: ModularMultiplicationModel,
    test_set: FullTestSet,
    *,
    output_base: Union[int, str],
    timeout_seconds: float = 300,
) -> dict[int, list[str]]:
    """Run model inference on the full test set.

    Timeout is cooperative: checked between batches.
    Only fully completed tiers are recorded. If timeout hits mid-tier,
    that tier's partial results are discarded.

    Returns dict mapping tier_id -> list of decoded prediction strings.
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

        is_tier_zero = tier_ts.tier_id == 0
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
                encoded = _preprocess_batch(model, batch)
                digits_list = model.predict_digits_batch(encoded)
                if len(digits_list) != len(batch):
                    logger.warning(
                        "Tier %d batch %d: predict_digits_batch returned %d "
                        "results, expected %d. Marking tier incomplete.",
                        tier_ts.tier_id, batch_start, len(digits_list), len(batch),
                    )
                    tier_complete = False
                    break
                decoded = _decode_batch(
                    digits_list, batch,
                    output_base=output_base,
                    is_tier_zero=is_tier_zero,
                )
                tier_predictions.extend(decoded)
            except Exception as e:
                logger.error(
                    "Error in tier %d batch %d: %s",
                    tier_ts.tier_id, batch_start, e,
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
    *,
    output_base: Union[int, str],
    num_checks: int = 10,
) -> bool:
    """Spot-check determinism by running random problems twice end-to-end.

    Uses the same preprocess + predict_digits + decode chain as the real
    inference pass, so both encoding *and* model output are exercised.
    Returns True if all checked decoded predictions are identical across the
    two runs.
    """
    import random as _rng

    all_cases: list[tuple[str, str, str, bool]] = []
    for tier_ts in test_set.tiers:
        is_tier_zero = tier_ts.tier_id == 0
        for c in tier_ts.cases:
            all_cases.append((c.a, c.b, c.p, is_tier_zero))

    if not all_cases:
        return True

    sample_size = min(num_checks, len(all_cases))
    sample = _rng.Random(42).sample(all_cases, sample_size)

    batch_size = max(1, model.max_batch_size())

    def _run_once() -> list[str]:
        out: list[str] = []
        for i in range(0, len(sample), batch_size):
            sub = sample[i : i + batch_size]
            batch = [(a, b, p) for a, b, p, _ in sub]
            encoded = _preprocess_batch(model, batch)
            digits_list = model.predict_digits_batch(encoded)
            if len(digits_list) != len(batch):
                logger.warning(
                    "predict_digits_batch returned wrong length: "
                    "expected %d, got %d",
                    len(batch), len(digits_list),
                )
                return []
            # All items in `sub` share the same tier-zero flag at the
            # batch boundary only when batch_size <= len(tier); for the
            # determinism check we just use the per-item flag.
            decoded: list[str] = []
            for digits, (a, b, p, t0) in zip(digits_list, sub):
                try:
                    value = decode_answer(
                        digits,
                        base=output_base,
                        prime=int(p),
                        is_tier_zero=t0,
                    )
                    decoded.append(str(value))
                except MalformedOutput:
                    decoded.append("")
            out.extend(decoded)
        return out

    results_a = _run_once()
    results_b = _run_once()

    if len(results_a) != len(sample) or len(results_b) != len(sample):
        return False

    for i, (ra, rb) in enumerate(zip(results_a, results_b)):
        if ra != rb:
            logger.warning(
                "Non-deterministic output on check %d: %r vs %r", i, ra, rb
            )
            return False

    return True


def check_preprocess_isolation(
    model: ModularMultiplicationModel,
    test_set: FullTestSet,
    *,
    num_checks: int = 5,
) -> bool:
    """Sanity check that ``preprocess_a/b/p`` are stateless / per-argument.

    Calls each preprocess function with the same input twice (with calls to
    the other preprocess hooks interleaved in between) and verifies the two
    outputs are equal. This catches the simplest forms of cross-argument
    cheating where one hook stashes state that another hook reads.

    Returns True if all checks pass. False otherwise, with a warning logged.
    """
    cases: list[tuple[str, str, str]] = []
    for tier_ts in test_set.tiers:
        for c in tier_ts.cases:
            cases.append((c.a, c.b, c.p))
    if not cases:
        return True

    import random as _rng
    sample = _rng.Random(123).sample(cases, min(num_checks, len(cases)))

    def _eq(x: Any, y: Any) -> bool:
        # Best-effort equality. Most encoders return strings, lists of ints,
        # or numpy/tensor objects; fall back to str() comparison for the
        # latter to avoid framework imports here.
        try:
            return x == y or str(x) == str(y)
        except Exception:
            return False

    ok = True
    for (a, b, p), (a2, b2, p2) in zip(sample, sample[::-1]):
        # First call
        ea1 = model.preprocess_a(a)
        eb1 = model.preprocess_b(b)
        ep1 = model.preprocess_p(p)
        # Inject other-argument calls in between (with different values)
        _ = model.preprocess_a(a2)
        _ = model.preprocess_b(b2)
        _ = model.preprocess_p(p2)
        # Repeat the original calls
        ea2 = model.preprocess_a(a)
        eb2 = model.preprocess_b(b)
        ep2 = model.preprocess_p(p)
        if not (_eq(ea1, ea2) and _eq(eb1, eb2) and _eq(ep1, ep2)):
            logger.warning(
                "Preprocess function appears stateful or cross-argument: "
                "repeated call on the same input produced different output."
            )
            ok = False
            break
    return ok


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
    logger.info(
        "Manifest OK: entry_class=%s output_base=%r",
        manifest.entry_class, manifest.output_base,
    )

    total_bytes = check_artifact_size(model_dir, config.max_artifact_bytes)
    logger.info("Artifact size: %.2f GB", total_bytes / 1e9)

    if config.skip_static_check:
        logger.warning("Static check SKIPPED (trusted-use mode)")
    else:
        findings = check_submission(model_dir)
        if findings:
            for f in findings:
                logger.error("static-check: %s", f.format())
            raise StaticCheckError(
                findings=findings,
                message=(
                    f"Submission failed static analysis with {len(findings)} "
                    f"finding(s); see logs for details."
                ),
            )
        logger.info("Static check passed")

    logger.info("Generating test set: %d problems", config.total_problems)
    test_set = generate_private_test_set(master_seed=master_seed, config=config)

    logger.info("Loading model...")
    load_start = time.monotonic()
    model = load_model(model_dir, manifest)
    load_time = time.monotonic() - load_start
    logger.info("Model loaded in %.1fs", load_time)

    iso_start = time.monotonic()
    preprocess_ok = check_preprocess_isolation(model, test_set)
    iso_time = time.monotonic() - iso_start
    if not preprocess_ok:
        logger.warning(
            "Preprocess isolation check failed (model preprocess appears "
            "stateful / cross-argument). Flagged for review."
        )
    logger.info("Preprocess isolation check in %.1fs", iso_time)

    det_start = time.monotonic()
    is_deterministic = check_determinism(
        model, test_set, output_base=manifest.output_base
    )
    det_time = time.monotonic() - det_start
    if not is_deterministic:
        logger.warning("Model is NON-DETERMINISTIC. Results will not be ranked.")
    logger.info("Determinism check in %.1fs", det_time)

    logger.info("Running inference (%.1fs budget)...", config.timeout_seconds)
    predictions = run_inference(
        model, test_set,
        output_base=manifest.output_base,
        timeout_seconds=config.timeout_seconds,
    )

    result = score_full_in_memory(test_set, predictions)
    result.deterministic = is_deterministic

    logger.info(
        "Evaluation complete: overall_accuracy=%.2f%%",
        result.overall_accuracy * 100,
    )

    return result
