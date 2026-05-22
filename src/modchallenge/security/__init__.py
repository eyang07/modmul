"""Anti-cheat / safety checks for submitted code."""

from __future__ import annotations

from modchallenge.security.static_check import (
    Finding,
    check_file,
    check_submission,
)

__all__ = ["Finding", "check_file", "check_submission"]
