# Modular Arithmetic Challenge — evaluation sandbox
#
# Build (from the repo root, NOT from this directory):
#   docker build -f docker/Dockerfile -t modchallenge-sandbox .
#
# Run (typically via `modchallenge evaluate-sandboxed`):
#   docker run --rm \
#     --network none \
#     --read-only \
#     --tmpfs /tmp:size=2g \
#     --memory=8g \
#     --cpus=4 \
#     -v /path/to/submission:/sandbox/submission:ro \
#     -v /path/to/output:/sandbox/output \
#     -e MODCHALLENGE_SUBMISSION=/sandbox/submission \
#     -e MODCHALLENGE_OUTPUT=/sandbox/output/result.json \
#     -e MODCHALLENGE_SEED=<hex> \
#     modchallenge-sandbox
#
# Package allowlist (everything else is intentionally absent):
#   - torch (CPU build)
#   - numpy
#   - transformers
#   - huggingface_hub (offline-mode only; no network at runtime)
#   - typer, pydantic (modchallenge runtime deps)
#   - safetensors (model loading)
#
# Explicitly NOT installed:
#   - sympy, gmpy2, mpmath, flint  (cryptographic / arbitrary-precision math)
#   - requests, urllib3, httpx     (HTTP clients; pip's bundled urllib3 is
#                                   the only network-capable lib present)
#   - subprocess capability is not removed at the OS level, but the sandbox
#     is run with --network none and --read-only by the harness.

FROM python:3.12-slim

# Install minimal system packages. We need libgomp for PyTorch's OpenMP
# routines; everything else is intentionally not installed.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Non-root user so the container runs unprivileged.
RUN useradd -m -u 1000 -s /bin/bash sandbox

# Python deps. Pin major versions to keep the image reproducible-ish.
# PyTorch CPU build keeps the image around 1 GB instead of 5 GB+ for CUDA.
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.5.1

RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    transformers==4.46.3 \
    huggingface_hub==0.26.5 \
    safetensors==0.4.5 \
    typer==0.13.1 \
    pydantic==2.10.3

# Install the modchallenge package itself. Build context is expected to be
# the repository root, so these paths refer to the repo layout.
WORKDIR /opt/modchallenge
COPY pyproject.toml ./
COPY src/ ./src/
RUN pip install --no-cache-dir -e .

# Sandbox entrypoint. Lives in /opt/modchallenge so it's not in the
# read-only mounted submission tree and is owned by root (immutable to the
# sandbox user).
COPY docker/entrypoint.py /opt/modchallenge/entrypoint.py

# Switch to the sandbox user. The submission will be mounted at
# /sandbox/submission (read-only) and outputs written under /sandbox/output.
USER sandbox
WORKDIR /sandbox

# Disable HF telemetry / cache writes outside /tmp by default.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "/opt/modchallenge/entrypoint.py"]
