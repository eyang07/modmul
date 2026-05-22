"""Sandbox entrypoint for the Modular Arithmetic Challenge.

Runs `evaluate_local` on the submission mounted at /sandbox/submission and
writes the result JSON to the path given by MODCHALLENGE_OUTPUT.

Configuration via environment variables (set by the harness on the host):

    MODCHALLENGE_SUBMISSION  Path to the mounted submission dir (required).
    MODCHALLENGE_OUTPUT      Path where the result JSON is written (required).
    MODCHALLENGE_SEED        Master seed as hex (optional; empty -> random).
    MODCHALLENGE_TOTAL       Total number of test problems (default 1100).
    MODCHALLENGE_TIMEOUT     Inference wall-clock budget, seconds (default 300).
    MODCHALLENGE_SKIP_STATIC Set to "1" to skip the static check (trusted-use).

Exit codes:

    0   evaluation completed (result.json was written)
    2   bad configuration / missing inputs
    3   submission rejected by static analysis
    4   evaluation crashed unexpectedly
"""

from __future__ import annotations

import json
import logging
import os
import sys
import traceback
from pathlib import Path


def _get_env(name: str, default: str | None = None) -> str:
    val = os.environ.get(name, default)
    if val is None:
        sys.stderr.write(f"missing required env var: {name}\n")
        sys.exit(2)
    return val


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )
    log = logging.getLogger("sandbox")

    submission = Path(_get_env("MODCHALLENGE_SUBMISSION"))
    output = Path(_get_env("MODCHALLENGE_OUTPUT"))
    seed_hex = os.environ.get("MODCHALLENGE_SEED", "")
    total = int(os.environ.get("MODCHALLENGE_TOTAL", "1100"))
    timeout = int(os.environ.get("MODCHALLENGE_TIMEOUT", "300"))
    skip_static = os.environ.get("MODCHALLENGE_SKIP_STATIC", "0") == "1"

    if not submission.is_dir():
        log.error("submission path is not a directory: %s", submission)
        return 2

    output.parent.mkdir(parents=True, exist_ok=True)

    # Imports are deferred until after env validation so failures here are
    # easier to triage.
    from modchallenge.config import EvalConfig
    from modchallenge.evaluation.pipeline import StaticCheckError, evaluate_local

    config = EvalConfig(
        total_problems=total,
        timeout_seconds=timeout,
        skip_static_check=skip_static,
    )
    master_seed = bytes.fromhex(seed_hex) if seed_hex else None

    try:
        result = evaluate_local(submission, master_seed=master_seed, config=config)
    except StaticCheckError as exc:
        log.error("submission rejected by static analysis: %s", exc)
        # Write the findings out so the host can surface them.
        output.write_text(
            json.dumps(
                {
                    "status": "rejected",
                    "reason": "static-check",
                    "findings": [
                        {
                            "file": f.file,
                            "line": f.line,
                            "col": f.col,
                            "rule": f.rule,
                            "message": f.message,
                        }
                        for f in exc.findings
                    ],
                },
                indent=2,
            )
        )
        return 3
    except Exception:
        log.error("evaluation crashed:\n%s", traceback.format_exc())
        output.write_text(
            json.dumps(
                {
                    "status": "error",
                    "reason": "exception",
                    "traceback": traceback.format_exc(),
                },
                indent=2,
            )
        )
        return 4

    summary = result.summary()
    summary["status"] = "completed"
    output.write_text(json.dumps(summary, indent=2))
    log.info(
        "evaluation complete: overall_accuracy=%.4f, highest_tier_above_90=%s",
        result.overall_accuracy,
        getattr(result, "highest_tier_above_90", None),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
