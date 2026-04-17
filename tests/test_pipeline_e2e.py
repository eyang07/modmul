"""End-to-end pipeline test with a dummy model."""

import json
import tempfile
from pathlib import Path

from modchallenge.evaluation.pipeline import evaluate_local
from modchallenge.config import EvalConfig


def _create_dummy_submission(tmpdir: Path, cheat: bool = False) -> Path:
    """Create a minimal submission directory with a dummy model."""
    model_dir = tmpdir / "submission"
    model_dir.mkdir()

    # manifest.json
    manifest = {
        "entry_class": "model.DummyModel",
        "framework": "none",
        "model_description": "dummy model for testing",
    }
    (model_dir / "manifest.json").write_text(json.dumps(manifest))

    # model.py
    if cheat:
        # "Cheater" model that computes the correct answer directly
        model_code = '''
from modchallenge.interface.base_model import ModularMultiplicationModel

class DummyModel(ModularMultiplicationModel):
    def load(self, model_dir):
        pass

    def predict(self, a, b, p):
        return str((int(a) * int(b)) % int(p))
'''
    else:
        # Random model that always returns "0"
        model_code = '''
from modchallenge.interface.base_model import ModularMultiplicationModel

class DummyModel(ModularMultiplicationModel):
    def load(self, model_dir):
        pass

    def predict(self, a, b, p):
        return "0"
'''
    (model_dir / "model.py").write_text(model_code)
    return model_dir


def test_pipeline_dummy_model():
    """Dummy model (always returns 0) should get partial credit on edge cases."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model_dir = _create_dummy_submission(Path(tmpdir))
        config = EvalConfig(total_problems=110, timeout_seconds=60)
        result = evaluate_local(model_dir, master_seed=b"e2e-test" * 4, config=config)

        assert result.deterministic
        assert 0 <= result.overall_accuracy <= 1.0
        assert len(result.tier_results) == 11
        # A model that always returns "0" gets some correct on a=0 or b=0 cases
        # but overall accuracy should be low
        assert result.overall_accuracy < 0.5


def test_pipeline_cheater_model():
    """Cheater model should get 100% accuracy."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model_dir = _create_dummy_submission(Path(tmpdir), cheat=True)
        config = EvalConfig(total_problems=110, timeout_seconds=60)
        result = evaluate_local(model_dir, master_seed=b"e2e-test" * 4, config=config)

        assert result.deterministic
        assert result.overall_accuracy == 1.0
