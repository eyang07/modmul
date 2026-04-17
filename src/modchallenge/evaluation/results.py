"""Result data structures for evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TierResult:
    """Evaluation result for a single tier."""

    tier_id: int
    total: int
    correct: int
    completed: bool  # True if the tier ran to completion (tier_complete marker found)

    @property
    def accuracy(self) -> float:
        if self.total == 0:
            return 0.0
        return self.correct / self.total

    def to_dict(self) -> dict:
        return {
            "tier_id": self.tier_id,
            "total": self.total,
            "correct": self.correct,
            "accuracy": round(self.accuracy, 4),
            "completed": self.completed,
        }


@dataclass
class EvalResult:
    """Full evaluation result across all tiers."""

    tier_results: list[TierResult] = field(default_factory=list)
    deterministic: bool = True
    eval_period: str = ""
    evaluator_version: str = ""
    image_digest: str = ""
    repo_id: str = ""
    revision: str = ""

    @property
    def scored_tiers(self) -> list[TierResult]:
        """Tiers 1-10 (modular arithmetic, scored). Tier 0 is diagnostic only."""
        return [t for t in self.tier_results if t.tier_id > 0]

    @property
    def diagnostic_tier(self) -> TierResult | None:
        """Tier 0 (pure multiplication, unscored)."""
        for t in self.tier_results:
            if t.tier_id == 0:
                return t
        return None

    @property
    def overall_accuracy(self) -> float:
        """Average accuracy across scored tiers 1-10 (equal weight).

        Incomplete tiers count as 0 accuracy.
        """
        scored = self.scored_tiers
        if not scored:
            return 0.0
        total_acc = sum(t.accuracy for t in scored if t.completed)
        return total_acc / len(scored)

    @property
    def highest_tier_above_90(self) -> int:
        """Highest scored tier_id where accuracy >= 90%. Returns -1 if none.

        Only considers scored tiers (1-10), not diagnostic tier 0.
        """
        best = -1
        for t in self.scored_tiers:
            if t.completed and t.accuracy >= 0.9:
                best = max(best, t.tier_id)
        return best

    def summary(self) -> dict:
        return {
            "overall_accuracy": round(self.overall_accuracy, 4),
            "highest_tier_above_90": self.highest_tier_above_90,
            "deterministic": self.deterministic,
            "tiers": [t.to_dict() for t in self.tier_results],
            "repo_id": self.repo_id,
            "revision": self.revision,
            "eval_period": self.eval_period,
        }
