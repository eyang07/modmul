"""Tests for the preprocess-isolation sanity check in the pipeline."""

from __future__ import annotations

from modchallenge.config import EvalConfig
from modchallenge.evaluation.pipeline import check_preprocess_isolation
from modchallenge.interface.base_model import ModularMultiplicationModel
from modchallenge.testgen.generator import generate_private_test_set


class PureModel(ModularMultiplicationModel):
    """Honest, stateless preprocess — should pass the isolation check."""

    def load(self, model_dir): pass

    def preprocess_a(self, a):
        return [int(c) for c in a]

    def preprocess_b(self, b):
        return [int(c) for c in b]

    def preprocess_p(self, p):
        return [int(c) for c in p]

    def predict_digits(self, a_enc, b_enc, p_enc):
        return [0]


class StatefulPreprocessModel(ModularMultiplicationModel):
    """Caches the last input and tags subsequent encodings with it.

    The isolation check should catch this: calling preprocess_a(x) twice
    with the same x produces different output because the inner counter
    advances on each call.
    """

    def __init__(self):
        self._counter = 0

    def load(self, model_dir): pass

    def preprocess_a(self, a):
        self._counter += 1
        return (a, self._counter)

    def preprocess_b(self, b):
        self._counter += 1
        return (b, self._counter)

    def preprocess_p(self, p):
        self._counter += 1
        return (p, self._counter)

    def predict_digits(self, a_enc, b_enc, p_enc):
        return [0]


def _tiny_test_set():
    return generate_private_test_set(
        master_seed=b"isolate!" * 4,
        config=EvalConfig(total_problems=110),
    )


def test_pure_preprocess_passes_isolation_check():
    ts = _tiny_test_set()
    assert check_preprocess_isolation(PureModel(), ts) is True


def test_stateful_preprocess_fails_isolation_check():
    ts = _tiny_test_set()
    assert check_preprocess_isolation(StatefulPreprocessModel(), ts) is False
