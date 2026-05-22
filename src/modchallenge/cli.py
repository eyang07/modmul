"""CLI entry point for the evaluation system."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import typer

app = typer.Typer(help="Modular Arithmetic Challenge - Evaluation System")


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )


def _output_result(summary: dict, output: Path | None) -> None:
    typer.echo(json.dumps(summary, indent=2))
    if output:
        output.write_text(json.dumps(summary, indent=2))
        typer.echo(f"\nResult written to {output}")


# ---------------------------------------------------------------------------
# Local evaluation
# ---------------------------------------------------------------------------

@app.command()
def evaluate(
    model_dir: Path = typer.Argument(..., help="Path to the submission directory"),
    total: int = typer.Option(1100, help="Total number of test problems (must be divisible by 11)"),
    seed: str = typer.Option("", help="Master seed hex string (empty = random)"),
    timeout: int = typer.Option(300, help="Total timeout in seconds"),
    output: Path = typer.Option(None, help="Write result JSON to this file"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Evaluate a local submission directory."""
    _setup_logging(verbose)

    from modchallenge.config import EvalConfig
    from modchallenge.evaluation.pipeline import evaluate_local

    master_seed = bytes.fromhex(seed) if seed else None
    config = EvalConfig(total_problems=total, timeout_seconds=timeout)
    result = evaluate_local(model_dir, master_seed=master_seed, config=config)
    _output_result(result.summary(), output)


# ---------------------------------------------------------------------------
# HuggingFace submission evaluation
# ---------------------------------------------------------------------------

@app.command()
def evaluate_hf(
    repo_id: str = typer.Argument(..., help="HuggingFace repo ID (e.g. 'user/my-model')"),
    revision: str = typer.Argument(..., help="Full 40-character commit SHA"),
    token: str = typer.Option("", help="HuggingFace access token (for private repos)"),
    total: int = typer.Option(1100, help="Total number of test problems (must be divisible by 11)"),
    seed: str = typer.Option("", help="Master seed hex string (empty = random)"),
    timeout: int = typer.Option(300, help="Total timeout in seconds"),
    output: Path = typer.Option(None, help="Write result JSON to this file"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Evaluate a HuggingFace submission by repo ID and commit hash.

    WARNING: This command loads and executes contestant code directly in the
    current process without sandboxing. Use only with trusted submissions.
    Official evaluation will use a sandboxed environment (not yet implemented).
    """
    _setup_logging(verbose)

    from modchallenge.config import EvalConfig
    from modchallenge.evaluation.loader import download_and_validate
    from modchallenge.evaluation.pipeline import evaluate_local
    from modchallenge.interface.submission_schema import SubmissionRef

    ref = SubmissionRef(repo_id=repo_id, revision=revision)
    hf_token = token or None
    model_dir = download_and_validate(ref, token=hf_token)

    master_seed = bytes.fromhex(seed) if seed else None
    config = EvalConfig(total_problems=total, timeout_seconds=timeout)
    result = evaluate_local(model_dir, master_seed=master_seed, config=config)
    result.repo_id = repo_id
    result.revision = revision

    _output_result(result.summary(), output)


# ---------------------------------------------------------------------------
# Quick-test a HuggingFace LLM (exploratory, not ranked)
# ---------------------------------------------------------------------------

@app.command()
def evaluate_llm(
    model_id: str = typer.Argument(..., help="HuggingFace model ID (e.g. 'google/gemma-3-1b-it')"),
    revision: str = typer.Option("", help="Model revision (commit SHA or branch)"),
    dtype: str = typer.Option("bfloat16", help="Model dtype (bfloat16, float16, float32)"),
    total: int = typer.Option(1100, help="Total number of test problems (must be divisible by 11)"),
    seed: str = typer.Option("", help="Master seed hex string (empty = random)"),
    timeout: int = typer.Option(300, help="Total timeout in seconds"),
    output: Path = typer.Option(None, help="Write result JSON to this file"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Quick-test any HuggingFace LLM (exploratory only, not for official ranking)."""
    _setup_logging(verbose)

    from modchallenge.config import EvalConfig
    from modchallenge.evaluation.llm_wrapper import GenericLLMWrapper
    from modchallenge.evaluation.pipeline import run_inference, check_determinism
    from modchallenge.evaluation.scorer import score_full_in_memory
    from modchallenge.testgen.generator import generate_private_test_set

    rev = revision or None
    master_seed = bytes.fromhex(seed) if seed else None
    config = EvalConfig(total_problems=total, timeout_seconds=timeout)

    test_set = generate_private_test_set(master_seed=master_seed, config=config)

    wrapper = GenericLLMWrapper(model_id=model_id, revision=rev, dtype=dtype)
    wrapper.load("")

    # LLM wrapper always emits base-10 digits; see llm_wrapper.predict_digits.
    is_deterministic = check_determinism(wrapper, test_set, output_base=10)
    predictions = run_inference(
        wrapper, test_set, output_base=10, timeout_seconds=config.timeout_seconds,
    )

    result = score_full_in_memory(test_set, predictions)
    result.deterministic = is_deterministic
    result.repo_id = model_id
    result.revision = revision

    _output_result(result.summary(), output)


# ---------------------------------------------------------------------------
# Evaluate from examples.json config
# ---------------------------------------------------------------------------

@app.command()
def evaluate_example(
    name: str = typer.Argument("", help="Example name (empty = run all examples)"),
    group: str = typer.Option("", help="Filter by group: 'public', 'private', or empty for all"),
    config_file: Path = typer.Option(
        "examples/examples.json",
        help="Path to examples.json config (default: examples/examples.json in project root)",
    ),
    total: int = typer.Option(110, help="Total number of test problems (must be divisible by 11)"),
    seed: str = typer.Option("", help="Master seed hex string (empty = random)"),
    timeout: int = typer.Option(300, help="Total timeout in seconds"),
    output: Path = typer.Option(None, help="Write result JSON to this file"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Evaluate models from examples.json config.

    Public models are evaluated via the LLM wrapper (exploratory).
    Private models are evaluated via HF download + submission interface.
    """
    _setup_logging(verbose)

    if not config_file.exists():
        typer.echo(f"Config not found: {config_file}", err=True)
        raise typer.Exit(1)

    data = json.loads(config_file.read_text())
    # Collect entries tagged with their group
    entries: list[tuple[str, dict]] = []
    for g in ("public", "private"):
        if group and group != g:
            continue
        for ex in data.get(g, []):
            entries.append((g, ex))

    if name:
        entries = [(g, ex) for g, ex in entries if ex["name"] == name]
        if not entries:
            typer.echo(f"Example '{name}' not found in {config_file}", err=True)
            raise typer.Exit(1)

    import os
    import secrets

    from modchallenge.config import EvalConfig
    from modchallenge.testgen.generator import generate_private_test_set

    # Use a single shared seed so all models are compared on the same benchmark
    master_seed = bytes.fromhex(seed) if seed else secrets.token_bytes(32)
    config = EvalConfig(total_problems=total, timeout_seconds=timeout)
    shared_test_set = generate_private_test_set(master_seed=master_seed, config=config)

    typer.echo(f"Benchmark seed: {master_seed.hex()}")

    for grp, ex in entries:
        typer.echo(f"\n{'='*60}")
        typer.echo(f"[{grp}] {ex['name']} ({ex['repo_id']})")
        typer.echo(f"{'='*60}")

        if grp == "public":
            # Public models: evaluate via LLM wrapper (exploratory, not ranked)
            from modchallenge.evaluation.llm_wrapper import GenericLLMWrapper
            from modchallenge.evaluation.pipeline import run_inference, check_determinism
            from modchallenge.evaluation.scorer import score_full_in_memory

            wrapper = GenericLLMWrapper(
                model_id=ex["repo_id"], revision=ex.get("revision"), dtype="bfloat16",
            )
            wrapper.load("")
            # LLM wrapper always emits base-10 digits.
            is_deterministic = check_determinism(
                wrapper, shared_test_set, output_base=10,
            )
            predictions = run_inference(
                wrapper, shared_test_set,
                output_base=10,
                timeout_seconds=config.timeout_seconds,
            )
            result = score_full_in_memory(shared_test_set, predictions)
            result.deterministic = is_deterministic
            result.repo_id = ex["repo_id"]
            result.revision = ex.get("revision", "")
        else:
            # Private models: evaluate via HF download + submission contract
            from modchallenge.evaluation.loader import download_and_validate
            from modchallenge.evaluation.pipeline import evaluate_local
            from modchallenge.interface.submission_schema import SubmissionRef

            ref = SubmissionRef(repo_id=ex["repo_id"], revision=ex["revision"])
            token_env = ex.get("token_env", "")
            hf_token = os.environ.get(token_env, "") if token_env else ""
            hf_token = hf_token or ex.get("token") or None
            model_dir = download_and_validate(ref, token=hf_token)
            result = evaluate_local(model_dir, master_seed=master_seed, config=config)
            result.repo_id = ex["repo_id"]
            result.revision = ex["revision"]

        _output_result(result.summary(), output)


# ---------------------------------------------------------------------------
# Leaderboard
# ---------------------------------------------------------------------------

@app.command()
def leaderboard(
    db: Path = typer.Option("leaderboard.json", help="Path to leaderboard JSON file"),
    period: str = typer.Option("", help="Filter by eval period (e.g. '2026-04')"),
) -> None:
    """Display the leaderboard."""
    from modchallenge.leaderboard.store import Leaderboard

    lb = Leaderboard(db)
    typer.echo(lb.display(period or None))


# ---------------------------------------------------------------------------
# Public benchmark
# ---------------------------------------------------------------------------

@app.command()
def generate_public(
    output_dir: Path = typer.Argument(..., help="Output directory for public benchmark"),
    problems_per_tier: int = typer.Option(100, help="Problems per tier"),
) -> None:
    """Generate the public benchmark test set (with answers)."""
    from modchallenge.config import PublicBenchmarkConfig
    from modchallenge.testgen.generator import generate_public_test_set, write_test_full

    config = PublicBenchmarkConfig(problems_per_tier=problems_per_tier)
    test_set = generate_public_test_set(config)
    write_test_full(test_set, output_dir)
    typer.echo(f"Public benchmark written to {output_dir} ({test_set.total_cases} cases)")


# ---------------------------------------------------------------------------
# Sandboxed evaluation
# ---------------------------------------------------------------------------

DEFAULT_SANDBOX_IMAGE = "modchallenge-sandbox"


@app.command()
def evaluate_sandboxed(
    submission_dir: Path = typer.Argument(..., help="Local submission directory"),
    image: str = typer.Option(
        DEFAULT_SANDBOX_IMAGE,
        help="Docker image tag to run (build with `modchallenge build-sandbox`)",
    ),
    total: int = typer.Option(1100, help="Total number of test problems"),
    seed: str = typer.Option("", help="Master seed hex string (empty = random)"),
    timeout: int = typer.Option(300, help="Inference wall-clock budget (s)"),
    memory: str = typer.Option("8g", help="Container memory limit"),
    cpus: str = typer.Option("4", help="Container CPU limit"),
    tmpfs_size: str = typer.Option("2g", help="Size of the /tmp tmpfs mount"),
    skip_static_check: bool = typer.Option(
        False, "--skip-static-check", help="Trusted-use bypass of the static check"
    ),
    output: Path = typer.Option(None, help="Write result JSON to this file"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Evaluate a local submission inside the sandboxed Docker image.

    The container runs with `--network none --read-only`, a writable tmpfs at
    /tmp, and bind mounts the submission read-only. The evaluation result is
    written to a temporary file inside an output volume and surfaced here.
    """
    import shutil
    import subprocess
    import tempfile

    _setup_logging(verbose)

    if shutil.which("docker") is None:
        typer.echo("error: docker is not on PATH", err=True)
        raise typer.Exit(code=2)

    submission_dir = submission_dir.resolve()
    if not submission_dir.is_dir():
        typer.echo(f"error: {submission_dir} is not a directory", err=True)
        raise typer.Exit(code=2)

    with tempfile.TemporaryDirectory(prefix="modchallenge-sandbox-") as tmp:
        host_output_dir = Path(tmp)
        result_file_in_container = "/sandbox/output/result.json"

        cmd = [
            "docker", "run", "--rm",
            "--network", "none",
            "--read-only",
            "--tmpfs", f"/tmp:size={tmpfs_size},mode=1777",
            "--memory", memory,
            "--cpus", cpus,
            "-v", f"{submission_dir}:/sandbox/submission:ro",
            "-v", f"{host_output_dir}:/sandbox/output",
            "-e", "MODCHALLENGE_SUBMISSION=/sandbox/submission",
            "-e", f"MODCHALLENGE_OUTPUT={result_file_in_container}",
            "-e", f"MODCHALLENGE_TOTAL={total}",
            "-e", f"MODCHALLENGE_TIMEOUT={timeout}",
        ]
        if seed:
            cmd += ["-e", f"MODCHALLENGE_SEED={seed}"]
        if skip_static_check:
            cmd += ["-e", "MODCHALLENGE_SKIP_STATIC=1"]
        cmd.append(image)

        if verbose:
            typer.echo("docker cmd: " + " ".join(cmd), err=True)
        proc = subprocess.run(cmd, capture_output=False)

        result_path = host_output_dir / "result.json"
        if result_path.exists():
            payload = json.loads(result_path.read_text())
            _output_result(payload, output)
        else:
            typer.echo(
                "error: container produced no result.json (exit code "
                f"{proc.returncode})", err=True,
            )

        if proc.returncode != 0:
            raise typer.Exit(code=proc.returncode)


@app.command()
def build_sandbox(
    image: str = typer.Option(DEFAULT_SANDBOX_IMAGE, help="Tag for the built image"),
    repo_root: Path = typer.Option(
        Path.cwd(), help="Repo root (build context for the Docker build)"
    ),
) -> None:
    """Build the sandbox Docker image (`docker/Dockerfile`)."""
    import shutil
    import subprocess

    if shutil.which("docker") is None:
        typer.echo("error: docker is not on PATH", err=True)
        raise typer.Exit(code=2)

    dockerfile = repo_root / "docker" / "Dockerfile"
    if not dockerfile.exists():
        typer.echo(f"error: {dockerfile} not found", err=True)
        raise typer.Exit(code=2)

    cmd = ["docker", "build", "-f", str(dockerfile), "-t", image, str(repo_root)]
    typer.echo("docker cmd: " + " ".join(cmd))
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise typer.Exit(code=proc.returncode)
    typer.echo(f"built {image}")


# ---------------------------------------------------------------------------
# Static analysis
# ---------------------------------------------------------------------------

@app.command()
def check(
    submission_dir: Path = typer.Argument(..., help="Path to a submission directory"),
) -> None:
    """Static-analyze a submission for prohibited code patterns.

    Exit code 0 if clean, 1 if any findings are reported. Findings are printed
    one per line in `path:line:col [rule] message` format.
    """
    from modchallenge.security.static_check import check_submission

    if not submission_dir.is_dir():
        typer.echo(f"error: {submission_dir} is not a directory", err=True)
        raise typer.Exit(code=2)

    findings = check_submission(submission_dir)
    for f in findings:
        typer.echo(f.format())
    if findings:
        typer.echo(f"\n{len(findings)} finding(s)", err=True)
        raise typer.Exit(code=1)
    typer.echo(f"clean ({submission_dir})")


if __name__ == "__main__":
    app()
