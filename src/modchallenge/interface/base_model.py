"""Unified model interface for the Modular Arithmetic Challenge.

All submissions must implement ModularMultiplicationModel.
The evaluator calls load() once, then predict_batch() repeatedly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class ModularMultiplicationModel(ABC):
    """Abstract base class that all submissions must implement.

    Input/output are decimal strings to support primes up to ~2^2000.
    Tokenization strategy is entirely up to the contestant.
    """

    @abstractmethod
    def load(self, model_dir: str) -> None:
        """Load model weights and any config from model_dir.

        model_dir is the local path of the downloaded HuggingFace repo.
        Called exactly once before any predict calls.
        """

    def predict(self, a: str, b: str, p: str) -> str:
        """Compute a*b mod p.

        Args:
            a: First operand as a decimal string, a >= 0. Can be much larger than p.
            b: Second operand as a decimal string, b >= 0. Can be much larger than p.
            p: Prime modulus as a decimal string, p >= 2.

        Returns:
            The result (a*b mod p) as a canonical decimal string.
            Must contain only digits 0-9, no leading zeros (except "0" itself),
            no spaces, newlines, signs, or separators.
        """
        raise NotImplementedError

    def predict_batch(self, inputs: list[tuple[str, str, str]]) -> list[str]:
        """Batch prediction. Override to leverage GPU batching.

        Default implementation calls predict() one by one.

        Args:
            inputs: List of (a, b, p) tuples, each as decimal strings.

        Returns:
            List of results in the same order as inputs.
        """
        return [self.predict(a, b, p) for a, b, p in inputs]

    def max_batch_size(self) -> int:
        """Hint to the evaluator on the maximum comfortable batch size.

        The evaluator will not send batches larger than this.
        Default is 1 (sequential). Override for GPU-batched models.
        """
        return 1
