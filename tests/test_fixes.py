"""Tests for the four issues raised in review (now adapted to the
post-lockdown digit-emitting interface)."""

import json
import tempfile
import time
from pathlib import Path

import pytest

from modchallenge.config import EvalConfig
from modchallenge.evaluation.pipeline import (
    check_determinism,
    evaluate_local,
    run_inference,
)
from modchallenge.interface.base_model import ModularMultiplicationModel
from modchallenge.testgen.generator import generate_private_test_set


# --- Fix 1: determinism check uses predict_digits_batch ---

class NonDeterministicBatchModel(ModularMultiplicationModel):
    """Model whose batch output flips between two answers on alternate calls."""

    def __init__(self):
        self._call_count = 0

    def load(self, model_dir):
        pass

    def predict_digits(self, a_enc, b_enc, p_enc):
        return [0]

    def predict_digits_batch(self, inputs):
        self._call_count += 1
        # Alternate between two different answers on each batch call.
        if self._call_count % 2 == 0:
            return [[0]] * len(inputs)
        return [[1]] * len(inputs)

    def max_batch_size(self):
        return 10


def test_determinism_catches_batch_nondeterminism():
    config = EvalConfig(total_problems=110)
    test_set = generate_private_test_set(master_seed=b"det-test" * 4, config=config)
    model = NonDeterministicBatchModel()
    model.load("")
    assert check_determinism(model, test_set, output_base=10) is False


# --- Fix 2: hard timeout ---

class SlowModel(ModularMultiplicationModel):
    """Model that sleeps 2 seconds per predict_digits call."""

    def load(self, model_dir):
        pass

    def predict_digits(self, a_enc, b_enc, p_enc):
        time.sleep(2)
        return [0]

    def max_batch_size(self):
        return 1


def test_cooperative_timeout():
    config = EvalConfig(total_problems=110)
    test_set = generate_private_test_set(master_seed=b"timeout!" * 4, config=config)
    model = SlowModel()
    model.load("")

    start = time.monotonic()
    result = run_inference(model, test_set, output_base=10, timeout_seconds=5)
    elapsed = time.monotonic() - start

    assert elapsed < 10, f"run_inference took {elapsed:.1f}s, expected <10s"
    assert len(result) < 10


# --- Fix 3: module isolation across submissions ---

def test_module_isolation():
    """Two submissions with same-named helper.py should not cross-contaminate."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dir_a = Path(tmpdir) / "sub_a"
        dir_a.mkdir()
        (dir_a / "manifest.json").write_text(json.dumps({
            "entry_class": "model_a.ModelA",
            "output_base": 10,
            "training_description": "test fixture",
        }))
        (dir_a / "helper.py").write_text('def get_digits(): return [1, 2, 3]\n')
        (dir_a / "model_a.py").write_text(
            'from modchallenge.interface.base_model import ModularMultiplicationModel\n'
            'import helper\n'
            'class ModelA(ModularMultiplicationModel):\n'
            '    def load(self, d): pass\n'
            '    def predict_digits(self, a_enc, b_enc, p_enc): return helper.get_digits()\n'
        )

        dir_b = Path(tmpdir) / "sub_b"
        dir_b.mkdir()
        (dir_b / "manifest.json").write_text(json.dumps({
            "entry_class": "model_b.ModelB",
            "output_base": 10,
            "training_description": "test fixture",
        }))
        (dir_b / "helper.py").write_text('def get_digits(): return [4, 5, 6]\n')
        (dir_b / "model_b.py").write_text(
            'from modchallenge.interface.base_model import ModularMultiplicationModel\n'
            'import helper\n'
            'class ModelB(ModularMultiplicationModel):\n'
            '    def load(self, d): pass\n'
            '    def predict_digits(self, a_enc, b_enc, p_enc): return helper.get_digits()\n'
        )

        from modchallenge.evaluation.loader import load_model, validate_manifest

        manifest_a = validate_manifest(dir_a)
        model_a = load_model(dir_a, manifest_a)
        assert model_a.predict_digits("1", "2", "3") == [1, 2, 3]

        manifest_b = validate_manifest(dir_b)
        model_b = load_model(dir_b, manifest_b)
        assert model_b.predict_digits("1", "2", "3") == [4, 5, 6]


# --- Fix 4: total_problems must be divisible by NUM_TIERS ---

def test_total_problems_not_divisible():
    with pytest.raises(ValueError, match="divisible"):
        EvalConfig(total_problems=1003)
