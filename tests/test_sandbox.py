"""Smoke tests for the sandbox layer.

Two layers:

1. **Entrypoint test** (always runs): invokes `docker/entrypoint.py` directly
   with environment variables, so it exercises the env -> EvalConfig ->
   evaluate_local flow without needing Docker. This catches most regressions
   in the entrypoint script itself.

2. **Docker container test** (skipped unless `MODCHALLENGE_SANDBOX_TEST=1`):
   actually runs the built sandbox image. Requires Docker + a built
   `modchallenge-sandbox` image. Set the env var to opt in.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
ENTRYPOINT = REPO_ROOT / "docker" / "entrypoint.py"


def _make_dummy_submission(tmp: Path, *, cheat: bool = False) -> Path:
    """Create a tiny submission directory with a `DummyModel`."""
    sub = tmp / "submission"
    sub.mkdir()
    (sub / "manifest.json").write_text(
        json.dumps({
            "entry_class": "model.DummyModel",
            "framework": "none",
            "output_base": 10,
            "model_description": "test fixture; trivial dummy model",
            "training_description": "test fixture; no real training",
        })
    )
    if cheat:
        # Triggers the modmul-shortcut static-check rule pre-load.
        body = (
            "from modchallenge.interface.base_model import "
            "ModularMultiplicationModel\n"
            "class DummyModel(ModularMultiplicationModel):\n"
            "    def load(self, model_dir): pass\n"
            "    def predict_digits(self, a_enc, b_enc, p_enc):\n"
            "        ans = int(a_enc) * int(b_enc) % int(p_enc)\n"
            "        if ans == 0: return [0]\n"
            "        digits = []\n"
            "        while ans > 0:\n"
            "            digits.append(ans % 10); ans //= 10\n"
            "        return list(reversed(digits))\n"
        )
    else:
        body = (
            "from modchallenge.interface.base_model import "
            "ModularMultiplicationModel\n"
            "class DummyModel(ModularMultiplicationModel):\n"
            "    def load(self, model_dir): pass\n"
            "    def predict_digits(self, a_enc, b_enc, p_enc): return [0]\n"
        )
    (sub / "model.py").write_text(body)
    return sub


# ---------------------------------------------------------------------------
# Entrypoint test (no Docker required)
# ---------------------------------------------------------------------------

def _run_entrypoint(env: dict[str, str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(ENTRYPOINT)],
        env={**os.environ, **env},
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_entrypoint_clean_submission_completes() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        sub = _make_dummy_submission(tmp_path)
        out = tmp_path / "result.json"

        proc = _run_entrypoint({
            "MODCHALLENGE_SUBMISSION": str(sub),
            "MODCHALLENGE_OUTPUT": str(out),
            "MODCHALLENGE_TOTAL": "11",
            "MODCHALLENGE_TIMEOUT": "60",
            "MODCHALLENGE_SEED": "deadbeef" * 8,
        })

        assert proc.returncode == 0, proc.stderr
        payload = json.loads(out.read_text())
        assert payload["status"] == "completed"
        assert "overall_accuracy" in payload


def test_entrypoint_cheater_submission_rejected() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        sub = _make_dummy_submission(tmp_path, cheat=True)
        out = tmp_path / "result.json"

        proc = _run_entrypoint({
            "MODCHALLENGE_SUBMISSION": str(sub),
            "MODCHALLENGE_OUTPUT": str(out),
            "MODCHALLENGE_TOTAL": "11",
            "MODCHALLENGE_TIMEOUT": "60",
        })

        assert proc.returncode == 3, (proc.returncode, proc.stderr)
        payload = json.loads(out.read_text())
        assert payload["status"] == "rejected"
        assert payload["reason"] == "static-check"
        assert any(f["rule"] == "modmul-shortcut" for f in payload["findings"])


def test_entrypoint_missing_env_var_fails() -> None:
    proc = _run_entrypoint({"MODCHALLENGE_SUBMISSION": "/nonexistent"})
    # Output env var missing -> exit 2
    assert proc.returncode == 2


def test_entrypoint_invalid_submission_fails() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "result.json"
        proc = _run_entrypoint({
            "MODCHALLENGE_SUBMISSION": "/nonexistent/path",
            "MODCHALLENGE_OUTPUT": str(out),
        })
        assert proc.returncode == 2


# ---------------------------------------------------------------------------
# Full Docker container test (opt-in)
# ---------------------------------------------------------------------------

_SANDBOX_OPT_IN = os.environ.get("MODCHALLENGE_SANDBOX_TEST") == "1"
_DOCKER_AVAILABLE = shutil.which("docker") is not None


def _image_exists(tag: str) -> bool:
    proc = subprocess.run(
        ["docker", "image", "inspect", tag],
        capture_output=True,
        text=True,
    )
    return proc.returncode == 0


@pytest.mark.skipif(
    not _SANDBOX_OPT_IN,
    reason="set MODCHALLENGE_SANDBOX_TEST=1 to run Docker sandbox tests",
)
@pytest.mark.skipif(not _DOCKER_AVAILABLE, reason="docker not on PATH")
def test_docker_sandbox_runs_clean_submission() -> None:
    image = "modchallenge-sandbox"
    if not _image_exists(image):
        pytest.skip(
            f"image {image!r} not built — run `modchallenge build-sandbox` first"
        )

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp).resolve()
        sub = _make_dummy_submission(tmp_path)
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        cmd = [
            "docker", "run", "--rm",
            "--network", "none",
            "--read-only",
            "--tmpfs", "/tmp:size=1g,mode=1777",
            "--memory", "2g",
            "--cpus", "2",
            "-v", f"{sub}:/sandbox/submission:ro",
            "-v", f"{out_dir}:/sandbox/output",
            "-e", "MODCHALLENGE_SUBMISSION=/sandbox/submission",
            "-e", "MODCHALLENGE_OUTPUT=/sandbox/output/result.json",
            "-e", "MODCHALLENGE_TOTAL=11",
            "-e", "MODCHALLENGE_TIMEOUT=60",
            "-e", "MODCHALLENGE_SEED=" + "ab" * 32,
            image,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        assert proc.returncode == 0, proc.stderr
        payload = json.loads((out_dir / "result.json").read_text())
        assert payload["status"] == "completed"
