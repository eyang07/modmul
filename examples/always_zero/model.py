"""Trivial baseline: always emit [0] as the answer.

Scores whatever fraction of the test set happens to have answer = 0
(edge cases where a == 0 or b == 0, plus a few random hits in tiers
that reuse a small set of primes).

Purpose: simplest possible submission that exercises the full pipeline
(manifest validation -> static check -> load -> preprocess -> predict_digits
-> decode -> score). Useful as a smoke test and as a floor reference for
the leaderboard.
"""

from __future__ import annotations

from modchallenge.interface.base_model import ModularMultiplicationModel


class AlwaysZero(ModularMultiplicationModel):
    def load(self, model_dir: str) -> None:
        pass

    def predict_digits(self, a_enc, b_enc, p_enc):
        return [0]
