"""JSON file-based leaderboard store.

Stores evaluation results and provides ranking, history, and display.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from modchallenge.evaluation.results import EvalResult


@dataclass
class LeaderboardEntry:
    """A single leaderboard entry."""

    repo_id: str
    revision: str
    overall_accuracy: float
    highest_tier_above_90: int
    deterministic: bool
    eval_period: str
    timestamp: str  # ISO 8601
    tier_accuracies: dict[str, float] = field(default_factory=dict)

    @staticmethod
    def from_eval_result(result: EvalResult) -> LeaderboardEntry:
        """Create an entry from an EvalResult."""
        tier_accs = {}
        for t in result.tier_results:
            tier_accs[str(t.tier_id)] = round(t.accuracy, 4) if t.completed else 0.0

        return LeaderboardEntry(
            repo_id=result.repo_id,
            revision=result.revision,
            overall_accuracy=round(result.overall_accuracy, 4),
            highest_tier_above_90=result.highest_tier_above_90,
            deterministic=result.deterministic,
            eval_period=result.eval_period or datetime.now(timezone.utc).strftime("%Y-%m"),
            timestamp=datetime.now(timezone.utc).isoformat(),
            tier_accuracies=tier_accs,
        )


class Leaderboard:
    """JSON file-based leaderboard.

    Each entry is a single evaluation run. Supports multiple entries
    per submission (different eval periods, re-runs, etc.).
    """

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.entries: list[LeaderboardEntry] = []
        if self.path.exists():
            self._load()

    def _load(self) -> None:
        text = self.path.read_text().strip()
        if not text:
            return
        data = json.loads(text)
        self.entries = [LeaderboardEntry(**e) for e in data.get("entries", [])]

    def _save(self) -> None:
        data = {
            "version": 1,
            "entries": [asdict(e) for e in self.entries],
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, indent=2, ensure_ascii=False))

    def add(self, result: EvalResult) -> LeaderboardEntry:
        """Add an evaluation result to the leaderboard."""
        entry = LeaderboardEntry.from_eval_result(result)
        self.entries.append(entry)
        self._save()
        return entry

    def current_rankings(self, eval_period: str | None = None) -> list[LeaderboardEntry]:
        """Get ranked entries, optionally filtered by eval period.

        - Only deterministic submissions are ranked
        - Best result per repo_id (across all revisions)
        - Sorted by highest_tier_above_90 desc, then overall_accuracy desc
        """
        filtered = self.entries
        if eval_period:
            filtered = [e for e in filtered if e.eval_period == eval_period]

        # Exclude non-deterministic submissions
        filtered = [e for e in filtered if e.deterministic]

        # Best result per repo_id: highest tier first, then accuracy
        best: dict[str, LeaderboardEntry] = {}
        for e in filtered:
            key = e.repo_id
            if key not in best or (
                e.highest_tier_above_90, e.overall_accuracy
            ) > (
                best[key].highest_tier_above_90, best[key].overall_accuracy
            ):
                best[key] = e

        ranked = sorted(
            best.values(),
            key=lambda e: (e.highest_tier_above_90, e.overall_accuracy),
            reverse=True,
        )
        return ranked

    def history(self, repo_id: str) -> list[LeaderboardEntry]:
        """Get all entries for a specific repo, newest first."""
        entries = [e for e in self.entries if e.repo_id == repo_id]
        return sorted(entries, key=lambda e: e.timestamp, reverse=True)

    def display(self, eval_period: str | None = None) -> str:
        """Format the leaderboard as a text table."""
        ranked = self.current_rankings(eval_period)
        if not ranked:
            return "No entries found."

        lines = []
        header = f"{'#':>3}  {'Repo':30s}  {'Rev':12s}  {'Accuracy':>8s}  {'Best Tier':>9s}  {'Det':>3s}  {'Period':7s}"
        lines.append(header)
        lines.append("-" * len(header))

        for i, e in enumerate(ranked, 1):
            det_flag = "Y" if e.deterministic else "N"
            rev_short = e.revision[:12] if e.revision else ""
            repo_short = e.repo_id[:30] if e.repo_id else "(local)"
            lines.append(
                f"{i:>3}  {repo_short:30s}  {rev_short:12s}  "
                f"{e.overall_accuracy:>7.1%}  "
                f"{'T' + str(e.highest_tier_above_90) if e.highest_tier_above_90 >= 0 else '-':>9s}  "
                f"{det_flag:>3s}  {e.eval_period:7s}"
            )

        return "\n".join(lines)
