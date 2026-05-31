"""End-to-end pipeline test with a dummy model.

After the structural lockdown, the contestant interface returns a list of
base-b digits and the harness decodes. The dummy submissions below use
base 10 (one digit per list entry, MSB-first).
"""

import json
import tempfile
from pathlib import Path

import pytest

from modchallenge.evaluation.pipeline import StaticCheckError, evaluate_local
from modchallenge.config import EvalConfig


def _create_dummy_submission(tmpdir: Path, cheat: bool = False) -> Path:
    """Create a minimal submission directory with a dummy model."""
    model_dir = tmpdir / "submission"
    model_dir.mkdir()

    manifest = {
        "entry_class": "model.DummyModel",
        "framework": "none",
        "model_description": "dummy model for testing",
        "output_base": 10,
        "training_description": "test fixture; no real training",
    }
    (model_dir / "manifest.json").write_text(json.dumps(manifest))

    if cheat:
        # Cheater: with identity preprocess, predict_digits receives the
        # raw decimal strings directly and computes the answer with
        # built-in int arithmetic. This is exactly the modmul-shortcut
        # pattern the static check rejects pre-load.
        model_code = '''
from modchallenge.interface.base_model import ModularMultiplicationModel

class DummyModel(ModularMultiplicationModel):
    def load(self, model_dir):
        pass

    def predict_digits(self, a_enc, b_enc, p_enc):
        answer = int(a_enc) * int(b_enc) % int(p_enc)
        if answer == 0:
            return [0]
        digits = []
        while answer > 0:
            digits.append(answer % 10)
            answer //= 10
        return list(reversed(digits))
'''
    else:
        # Constant model that always returns "0" (digit list [0]).
        model_code = '''
from modchallenge.interface.base_model import ModularMultiplicationModel

class DummyModel(ModularMultiplicationModel):
    def load(self, model_dir):
        pass

    def predict_digits(self, a_enc, b_enc, p_enc):
        return [0]
'''
    (model_dir / "model.py").write_text(model_code)
    return model_dir


def test_pipeline_dummy_model():
    """Dummy model (always returns 0) gets partial credit on edge cases only."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model_dir = _create_dummy_submission(Path(tmpdir))
        config = EvalConfig(total_problems=110, timeout_seconds=60)
        result = evaluate_local(model_dir, master_seed=b"e2e-test" * 4, config=config)

        assert result.deterministic
        assert 0 <= result.overall_accuracy <= 1.0
        assert len(result.tier_results) == 11
        # Returns 0 → correct only on a=0/b=0 edge cases, low overall.
        assert result.overall_accuracy < 0.5


def test_pipeline_cheater_model_blocked_by_static_check():
    """The cheater pattern `int(a)*int(b)%int(p)` is rejected before load."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model_dir = _create_dummy_submission(Path(tmpdir), cheat=True)
        config = EvalConfig(total_problems=110, timeout_seconds=60)
        with pytest.raises(StaticCheckError) as excinfo:
            evaluate_local(model_dir, master_seed=b"e2e-test" * 4, config=config)
        rules = [f.rule for f in excinfo.value.findings]
        assert "modmul-shortcut" in rules


def test_pipeline_cheater_model_runs_when_static_check_skipped():
    """With the static check skipped, this cheater pattern (stash a/b/p in
    instance state across preprocess hooks, recompute the answer in
    predict_digits) actually runs and scores 100%.

    The isolation check does NOT catch this variant because the preprocess
    functions return the same output for the same input — they just have a
    side effect on instance state. Static analysis is the layer that catches
    this in practice; the bypass test confirms manual review is needed when
    the static check is intentionally skipped (trusted-use mode).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        model_dir = _create_dummy_submission(Path(tmpdir), cheat=True)
        config = EvalConfig(
            total_problems=110, timeout_seconds=60, skip_static_check=True
        )
        result = evaluate_local(model_dir, master_seed=b"e2e-test" * 4, config=config)

        assert result.deterministic
        assert result.overall_accuracy == 1.0
