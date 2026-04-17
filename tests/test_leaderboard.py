"""Tests for the leaderboard store."""

import json
import tempfile
from pathlib import Path

import pytest

from modchallenge.evaluation.results import EvalResult, TierResult
from modchallenge.leaderboard.store import Leaderboard, LeaderboardEntry


def _make_eval_result(
    overall_acc_tiers: list[tuple[int, int, int]] | None = None,
    deterministic: bool = True,
    repo_id: str = "user/model",
    revision: str = "a" * 40,
    eval_period: str = "2026-04",
) -> EvalResult:
    """Create an EvalResult with specified tier results.

    overall_acc_tiers: list of (tier_id, correct, total) tuples.
    """
    if overall_acc_tiers is None:
        # Default: 10 scored tiers (1-10) + tier 0 diagnostic, all 100%
        overall_acc_tiers = [(i, 100, 100) for i in range(11)]

    tier_results = [
        TierResult(tier_id=tid, correct=c, total=t, completed=True)
        for tid, c, t in overall_acc_tiers
    ]
    return EvalResult(
        tier_results=tier_results,
        deterministic=deterministic,
        repo_id=repo_id,
        revision=revision,
        eval_period=eval_period,
    )


class TestLeaderboardEntry:
    def test_from_eval_result_basic(self):
        result = _make_eval_result()
        entry = LeaderboardEntry.from_eval_result(result)

        assert entry.repo_id == "user/model"
        assert entry.revision == "a" * 40
        assert entry.overall_accuracy == 1.0
        assert entry.highest_tier_above_90 == 10
        assert entry.deterministic is True
        assert entry.eval_period == "2026-04"
        assert len(entry.tier_accuracies) == 11  # tier 0-10

    def test_from_eval_result_partial(self):
        # Only tiers 0-3 correct, rest wrong
        tiers = [(0, 100, 100)] + [(i, 100, 100) for i in range(1, 4)] + \
                [(i, 0, 100) for i in range(4, 11)]
        result = _make_eval_result(overall_acc_tiers=tiers)
        entry = LeaderboardEntry.from_eval_result(result)

        assert entry.highest_tier_above_90 == 3
        # overall_accuracy = average of tiers 1-10 = (3*1.0 + 7*0.0)/10 = 0.3
        assert abs(entry.overall_accuracy - 0.3) < 0.01


class TestLeaderboard:
    def test_empty_leaderboard(self, tmp_path):
        lb = Leaderboard(tmp_path / "lb.json")
        assert lb.entries == []
        assert lb.display() == "No entries found."

    def test_add_and_persist(self, tmp_path):
        db_path = tmp_path / "lb.json"
        result = _make_eval_result()

        lb = Leaderboard(db_path)
        entry = lb.add(result)
        assert entry.repo_id == "user/model"
        assert len(lb.entries) == 1

        # Reload from disk
        lb2 = Leaderboard(db_path)
        assert len(lb2.entries) == 1
        assert lb2.entries[0].repo_id == "user/model"

    def test_json_format(self, tmp_path):
        db_path = tmp_path / "lb.json"
        lb = Leaderboard(db_path)
        lb.add(_make_eval_result())

        data = json.loads(db_path.read_text())
        assert data["version"] == 1
        assert len(data["entries"]) == 1
        assert "overall_accuracy" in data["entries"][0]

    def test_current_rankings_sort(self, tmp_path):
        lb = Leaderboard(tmp_path / "lb.json")

        # Add two submissions with different accuracies
        r1 = _make_eval_result(
            overall_acc_tiers=[(0, 100, 100)] + [(i, 50, 100) for i in range(1, 11)],
            repo_id="user/bad",
            revision="b" * 40,
        )
        r2 = _make_eval_result(
            overall_acc_tiers=[(i, 100, 100) for i in range(11)],
            repo_id="user/good",
            revision="c" * 40,
        )
        lb.add(r1)
        lb.add(r2)

        ranked = lb.current_rankings()
        assert len(ranked) == 2
        assert ranked[0].repo_id == "user/good"
        assert ranked[1].repo_id == "user/bad"

    def test_current_rankings_dedup(self, tmp_path):
        lb = Leaderboard(tmp_path / "lb.json")

        # Add same repo+revision twice, should keep latest
        r1 = _make_eval_result(repo_id="user/m", revision="d" * 40)
        r2 = _make_eval_result(repo_id="user/m", revision="d" * 40)
        lb.add(r1)
        lb.add(r2)

        ranked = lb.current_rankings()
        assert len(ranked) == 1  # deduplicated

    def test_current_rankings_filter_period(self, tmp_path):
        lb = Leaderboard(tmp_path / "lb.json")
        lb.add(_make_eval_result(eval_period="2026-03", repo_id="a/1", revision="1" * 40))
        lb.add(_make_eval_result(eval_period="2026-04", repo_id="a/2", revision="2" * 40))

        r_march = lb.current_rankings(eval_period="2026-03")
        assert len(r_march) == 1
        assert r_march[0].repo_id == "a/1"

        r_april = lb.current_rankings(eval_period="2026-04")
        assert len(r_april) == 1
        assert r_april[0].repo_id == "a/2"

    def test_history(self, tmp_path):
        lb = Leaderboard(tmp_path / "lb.json")
        lb.add(_make_eval_result(repo_id="user/m", revision="a" * 40))
        lb.add(_make_eval_result(repo_id="user/m", revision="b" * 40))
        lb.add(_make_eval_result(repo_id="other/m", revision="c" * 40))

        hist = lb.history("user/m")
        assert len(hist) == 2
        # Should be newest first
        assert hist[0].timestamp >= hist[1].timestamp

    def test_load_empty_file(self, tmp_path):
        """Should handle an existing but empty JSON file gracefully."""
        db_path = tmp_path / "lb.json"
        db_path.write_text("")
        lb = Leaderboard(db_path)
        assert lb.entries == []

    def test_display_format(self, tmp_path):
        lb = Leaderboard(tmp_path / "lb.json")
        lb.add(_make_eval_result())

        output = lb.display()
        assert "user/model" in output
        assert "100.0%" in output
        assert "T10" in output
