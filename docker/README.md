# Sandbox

Docker image and entrypoint for sandboxed evaluation of submitted models.

## What this is for

The Modular Arithmetic Challenge runs contestant code as part of evaluation.
The sandbox isolates that code so it cannot:

- reach the network (`--network none`)
- write to the host filesystem (`--read-only`, only `/tmp` tmpfs is writable)
- spawn host processes
- access secrets (no env vars beyond the harness-supplied ones)
- exceed configured CPU / memory limits

It also enforces the **package allowlist** that backs the rules' "Sandbox
package allowlist" enforcement layer: `sympy`, `gmpy2`, `mpmath`, `flint`,
and the various networking libraries are **not installed**, so an attempt to
`import` them fails at load time.

## Image contents

| Category | Installed |
|----------|-----------|
| Runtime | Python 3.12 (slim) |
| Math | `torch` (CPU), `numpy` |
| Model loading | `transformers`, `huggingface_hub` (offline mode), `safetensors` |
| Evaluation harness | `modchallenge` (this repo), `typer`, `pydantic` |

Explicitly **NOT** installed: `sympy`, `gmpy2`, `mpmath`, `flint`,
`requests`, `httpx`, `urllib3` (beyond pip's bundled copy). The Dockerfile
header has the up-to-date list.

## Build

From the repository root:

```bash
modchallenge build-sandbox
# or directly:
docker build -f docker/Dockerfile -t modchallenge-sandbox .
```

First build is slow (pip downloads PyTorch CPU wheels — ~250 MB). The image
ends up around 1 GB.

## Run

The convenience wrapper is the recommended path:

```bash
modchallenge evaluate-sandboxed ./my-submission \
    --total 1100 \
    --timeout 300 \
    --memory 8g \
    --cpus 4
```

This shells out to `docker run` with the right isolation flags, parses the
result JSON, and surfaces it as the command's output.

For ad-hoc / debug runs you can drive the container directly:

```bash
mkdir -p /tmp/sandbox-out
docker run --rm \
    --network none \
    --read-only \
    --tmpfs /tmp:size=2g,mode=1777 \
    --memory 8g \
    --cpus 4 \
    -v /path/to/submission:/sandbox/submission:ro \
    -v /tmp/sandbox-out:/sandbox/output \
    -e MODCHALLENGE_SUBMISSION=/sandbox/submission \
    -e MODCHALLENGE_OUTPUT=/sandbox/output/result.json \
    -e MODCHALLENGE_SEED=$(openssl rand -hex 32) \
    -e MODCHALLENGE_TOTAL=110 \
    -e MODCHALLENGE_TIMEOUT=60 \
    modchallenge-sandbox
cat /tmp/sandbox-out/result.json
```

## Entrypoint protocol

`entrypoint.py` reads everything from environment variables:

| Variable | Required | Default | Meaning |
|----------|----------|---------|---------|
| `MODCHALLENGE_SUBMISSION` | yes | — | Path to the mounted submission dir |
| `MODCHALLENGE_OUTPUT`     | yes | — | Path where the result JSON is written |
| `MODCHALLENGE_SEED`       | no  | random | Master seed (hex) for test generation |
| `MODCHALLENGE_TOTAL`      | no  | 1100 | Total problems |
| `MODCHALLENGE_TIMEOUT`    | no  | 300 | Inference budget (s) |
| `MODCHALLENGE_SKIP_STATIC`| no  | 0 | Set to `1` to bypass the static check |

Exit codes:

| Code | Meaning |
|------|---------|
| 0 | Evaluation completed; result.json contains `"status": "completed"` |
| 2 | Bad configuration (missing env var, bad submission path) |
| 3 | Submission rejected by static analysis; result.json has `"status": "rejected"` and the findings |
| 4 | Evaluation crashed; result.json has `"status": "error"` and a traceback |

## Testing the sandbox

Entrypoint-level tests run on every `pytest` invocation (no Docker needed):

```bash
pytest tests/test_sandbox.py
```

Full container tests (require a built image) are opt-in:

```bash
MODCHALLENGE_SANDBOX_TEST=1 pytest tests/test_sandbox.py -v
```

## Limitations / TODOs

- **CPU only.** The current image installs PyTorch CPU. For an evaluation
  pass that needs GPU, we'll need a `modchallenge-sandbox-gpu` tag based on
  a CUDA image.
- **No GPU device exposure.** `evaluate_sandboxed` does not pass
  `--gpus all` because the image doesn't ship CUDA. Add when GPU image
  exists.
- **No HF download inside.** `HF_HUB_OFFLINE=1` is set; the harness is
  expected to download the submission to a local directory first and mount
  it in. Keeps the sandbox attack surface minimal.
- **Behavioural checks are not yet wired in.** Weight perturbation,
  distribution shift, latency profile — see `rules/evaluation.md` Enforcement
  Layer 3 (planned).
