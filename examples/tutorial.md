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

The interface is intentionally narrow: three per-argument preprocessing hooks (each only ever sees its own argument) and one `predict_digits` method that emits the answer as a list of base-`b` digits. The harness — not your code — decodes those digits into the final answer.

```python
# model.py
from modchallenge.interface.base_model import ModularMultiplicationModel


class MyModel(ModularMultiplicationModel):
    def load(self, model_dir: str) -> None:
        # Load your trained weights
        ...

    # Per-argument preprocessing. Each hook only sees its own argument.
    # Defaults are identity (pass the string through unchanged); override
    # to tokenise, embed, do base conversion, etc.
    def preprocess_a(self, a: str):
        return [int(c) for c in a]   # e.g. digit-level tokens

    def preprocess_b(self, b: str):
        return [int(c) for c in b]

    def preprocess_p(self, p: str):
        return [int(c) for c in p]

    def predict_digits(self, a_enc, b_enc, p_enc) -> list[int]:
        # Run the model on the encoded inputs. Return the answer
        # (a * b mod p) as base-b digits, MOST-SIGNIFICANT-FIRST.
        # b = the value declared in manifest.json's output_base field.
        # For example, with output_base = 10 and answer = 52, return [5, 2].
        ...

    def predict_digits_batch(self, inputs) -> list[list[int]]:
        # Optional: override for GPU batching (default loops over predict_digits).
        ...

    def max_batch_size(self) -> int:
        return 64
```

Create a `manifest.json` declaring your entry class and the base you emit answers in:

```json
{
  "entry_class": "model.MyModel",
  "output_base": 10
}
```

Allowed `output_base` values:

- any integer in `[2, 2^32]` — for example `2` (binary), `10` (decimal digits), `256` (bytes), `65536` (word-level), etc.
- the string `"p"` — answers are emitted in base equal to the current prime, so each answer is a single digit in `[0, p)`

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
- The preprocessing hooks each see decimal strings — `a` and `b` can be hundreds of digits long (much larger than `p`); `p` is up to ~617 digits
- If your model uses GPU batching, override `predict_digits_batch()` and `max_batch_size()`
- The pipeline runs a sanity check that catches obviously stateful preprocessing (e.g. caching `a` in instance state so a later hook can read it) — design your preprocessing as pure per-argument functions
