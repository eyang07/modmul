"""Load and validate submissions from HuggingFace."""

from __future__ import annotations

import importlib
import json
import logging
import sys
from pathlib import Path

from modchallenge.interface.base_model import ModularMultiplicationModel
from modchallenge.interface.submission_schema import SubmissionManifest, SubmissionRef

logger = logging.getLogger(__name__)


def download_submission(
    repo_id: str, revision: str, local_dir: Path, token: str | None = None,
) -> Path:
    """Download a submission from HuggingFace by repo_id + commit SHA.

    Args:
        token: HuggingFace access token for private repos.

    Returns the local path to the downloaded repo.
    """
    from huggingface_hub import snapshot_download

    path = snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=str(local_dir),
        token=token,
    )
    return Path(path)


def download_and_validate(
    ref: SubmissionRef,
    cache_dir: Path | None = None,
    max_bytes: int = 20 * 1024 ** 3,
    token: str | None = None,
) -> Path:
    """Download a submission from HuggingFace and validate it.

    Uses a local cache to avoid re-downloading. Validates manifest
    and artifact size after download.

    Args:
        ref: HuggingFace repo reference (repo_id + 40-char SHA).
        cache_dir: Where to cache downloads. Default: ~/.cache/modchallenge/
        max_bytes: Maximum allowed artifact size.

    Returns:
        Path to the validated local submission directory.
    """
    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "modchallenge"

    local_dir = cache_dir / ref.repo_id / ref.revision[:12]

    if local_dir.exists() and (local_dir / "manifest.json").exists():
        logger.info("Using cached submission at %s", local_dir)
    else:
        logger.info("Downloading %s @ %s ...", ref.repo_id, ref.revision[:12])
        download_submission(ref.repo_id, ref.revision, local_dir, token=token)

    validate_manifest(local_dir)
    check_artifact_size(local_dir, max_bytes)
    return local_dir


def validate_manifest(model_dir: Path) -> SubmissionManifest:
    """Read and validate manifest.json from the submission directory."""
    manifest_path = model_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json not found in {model_dir}")

    with open(manifest_path) as f:
        data = json.load(f)

    return SubmissionManifest(**data)


def check_artifact_size(model_dir: Path, max_bytes: int) -> int:
    """Check total size of submission artifacts. Raises if over limit."""
    total = sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file())
    if total > max_bytes:
        raise ValueError(
            f"Submission size {total / 1e9:.1f} GB exceeds limit "
            f"{max_bytes / 1e9:.1f} GB"
        )
    return total


# Tracks all submission directories ever added to sys.path by load_model.
_loaded_submission_dirs: list[str] = []


def _purge_submission_modules() -> None:
    """Remove all cached modules whose source lives under ANY previous submission dir.

    This ensures that same-named files (helper.py, utils.py) from different
    submissions never cross-contaminate.
    """
    resolved_dirs = [str(Path(d).resolve()) for d in _loaded_submission_dirs]
    if not resolved_dirs:
        return
    to_remove = []
    for name, mod in sys.modules.items():
        mod_file = getattr(mod, "__file__", None)
        if not mod_file:
            continue
        resolved_file = str(Path(mod_file).resolve())
        if any(resolved_file.startswith(d) for d in resolved_dirs):
            to_remove.append(name)
    for name in to_remove:
        del sys.modules[name]


def load_model(
    model_dir: Path,
    manifest: SubmissionManifest,
) -> ModularMultiplicationModel:
    """Dynamically import and instantiate the contestant's model class.

    Args:
        model_dir: Path to the downloaded submission.
        manifest: Validated manifest.

    Returns:
        An instance of the contestant's model, with load() already called.
    """
    # Remove old submission dirs from sys.path and purge their modules
    for old_dir in _loaded_submission_dirs:
        if old_dir in sys.path:
            sys.path.remove(old_dir)
    _purge_submission_modules()

    # Add new model_dir to sys.path
    model_dir_str = str(model_dir)
    sys.path.insert(0, model_dir_str)
    _loaded_submission_dirs.append(model_dir_str)

    try:
        parts = manifest.entry_class.rsplit(".", 1)
        module_path, class_name = parts[0], parts[1]
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        raise ImportError(
            f"Cannot import entry_class {manifest.entry_class!r}: {e}"
        ) from e

    if not (isinstance(cls, type) and issubclass(cls, ModularMultiplicationModel)):
        raise TypeError(
            f"{manifest.entry_class} must be a subclass of ModularMultiplicationModel"
        )

    instance = cls()
    instance.load(str(model_dir))
    return instance
