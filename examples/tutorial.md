# Tutorial: End-to-End Workflow

## Quick Start: Run the Examples

The fastest way to get started — run the built-in example models right away. All models live on HuggingFace; `examples/examples.json` contains the repo IDs, commit hashes, and read-only tokens.

```bash
pip install -e ".[dev]"
pip install torch

# Set HF tokens for private examples (see examples/examples.json for token_env names)
export HF_TOKEN_TINY_TRANSFORMER=hf_xxx
export HF_TOKEN_AXOLVER_FORKED=hf_xxx

# Run private example models (downloads from HF using tokens from env vars)
modchallenge evaluate-example --group private --total 110

# Run a specific example by name
modchallenge evaluate-example tiny-transformer --total 110

# Run public baseline LLMs (no token needed, downloads Qwen models ~1-3 GB)
modchallenge evaluate-example --group public --total 110

# Run everything
modchallenge evaluate-example --total 110
```

`--total 110` = 110 problems = 10 per tier (11 tiers). Use `--total 1100` for a full 100-per-tier run.

**What happens under the hood:**

- Private models: downloaded from HF with read-only token -> loaded via `manifest.json` -> evaluated through the submission contract (`ModularMultiplicationModel`). This is the same contract used for official evaluation.
- Public models: downloaded from HF -> evaluated via the LLM wrapper with a fixed prompt. This is an **exploratory smoke test only**, not the official ranking pipeline.

If the private examples run, your setup is correct and your own submissions will work the same way.

## Build Your Own Model

### 1. Implement the Interface

```python
# model.py
from modchallenge.interface.base_model import ModularMultiplicationModel

class MyModel(ModularMultiplicationModel):
    def load(self, model_dir: str) -> None:
        # Load your trained weights
        ...

    def predict(self, a: str, b: str, p: str) -> str:
        # Compute (a * b) mod p
        # a, b can be much larger than p — all are decimal strings
        return result_string

    def predict_batch(self, inputs: list[tuple[str, str, str]]) -> list[str]:
        # Optional: override for GPU batching (default calls predict() in a loop)
        ...

    def max_batch_size(self) -> int:
        return 64
```

Create a `manifest.json`:

```json
{
  "entry_class": "model.MyModel"
}
```

Your submission directory:

```
my-model/
├── manifest.json
├── model.py
└── weights.pt (or any model files)
```

### 2. Test Locally

```bash
modchallenge evaluate ./my-model --total 110
```

### 3. Upload to Private HuggingFace Repo

```bash
pip install huggingface_hub
huggingface-cli login

python -c "
from huggingface_hub import HfApi
api = HfApi()
api.create_repo('your-username/my-model', private=True)
api.upload_folder(folder_path='./my-model', repo_id='your-username/my-model')
"
```

Create a **fine-grained read-only token** for your repo at https://huggingface.co/settings/tokens — this lets you test without exposing write access.

### 4. Test via HuggingFace (Private)

Get your commit hash:

```bash
python -c "
from huggingface_hub import HfApi
api = HfApi()
info = api.repo_info('your-username/my-model')
print(info.sha)
"
```

Run evaluation with the same pipeline organizers will use:

```bash
modchallenge evaluate-hf your-username/my-model <commit_hash> --token hf_xxx --total 110
```

If it works here, it will work in official evaluation.

### 5. Iterate

```
Improve model -> push to HF -> get new commit hash -> re-test
```

Each push creates a new commit hash. The old hash still points to the old version — nothing is overwritten.

### 6. Share via Contributor Network (Optional)

During the competition, share your model through the [SAIR Contributor Network](https://competition.sair.foundation/contributor-network):

- Make your HF repo public (or create a new public repo for sharing)
- Submit repo_id + commit_hash via the Contributor Network
- Organizers will evaluate your model and publish results
- Community can give feedback; your work will be recognized and cited

### 7. Final Submission

Before the deadline:

1. Make your HF repo **public**
2. Submit `repo_id` + `commit_hash` to organizers
3. Organizers run official evaluation with a secret random seed
4. Results posted to leaderboard

## Tips

- Start with `--total 110` (10 per tier) for fast iteration, use `--total 1100` for thorough testing
- Use `--seed <hex>` to get reproducible results across runs
- Check Tier 0 results to diagnose whether failures come from multiplication or modular reduction
- The `predict()` method receives decimal strings — handle arbitrarily large numbers
- If your model uses GPU batching, override `predict_batch()` and `max_batch_size()`
