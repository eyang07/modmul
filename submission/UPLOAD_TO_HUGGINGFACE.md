# How to upload the submission to HuggingFace

The competition submission is a **public HuggingFace repo** identified by `repo_id` + a
`commit_hash`. This guide takes the three files in `submission/ebm_modmul/` and gets them onto
HuggingFace, then submitted. Do this when we have a checkpoint worth submitting (it's fine to
upload a "safety" version early and re-upload better weights later — each upload is a new commit).

## What gets uploaded

From `submission/ebm_modmul/`:

| File | What it is | Required |
|------|------------|----------|
| `manifest.json` | declares entry class, `output_base`, descriptions | yes |
| `model.py` | the `EBMModMul` model code | yes |
| `weights.pt` | the trained weights (~13 MB; limit is 20 GB) | yes |

> ⚠️ Do **not** upload anything else into the repo. The static check scans every `.py` in the repo —
> only `model.py` should be there. (Our trainers live in `training/`, never in `submission/`.)

## One-time setup

1. Create a free account at https://huggingface.co/join (if you don't have one).
2. Pick a repo name, e.g. `ebm-modmul`. Your `repo_id` will be `<your-username>/ebm-modmul`.

---

## Method A — Web UI (easiest, no terminal)

1. Go to https://huggingface.co/new
2. **Owner** = your username. **Model name** = `ebm-modmul`.
3. Set visibility to **Private** for now (we make it Public only for the final submission). Click
   **Create model**.
4. On the repo page, click **Files and versions → Add file → Upload files**.
5. Drag in **all three** files: `manifest.json`, `model.py`, `weights.pt`
   (from `submission/ebm_modmul/` on your computer).
6. Add a commit message (e.g. "first upload") and click **Commit changes to main**.
7. **Get the commit hash:** click the **History** link (or the commit message) — the 40-character
   hash is your `commit_hash`. (Short hash is shown; click it for the full 40 chars, or use Method B's
   one-liner below.)

---

## Method B — CLI (repeatable; better when re-uploading new weights)

```bash
# one-time: install the client and log in
pip install -U huggingface_hub
huggingface-cli login        # paste a WRITE token from https://huggingface.co/settings/tokens

# create the repo (private for now) and upload the folder
python -c "
from huggingface_hub import HfApi
api = HfApi()
repo = 'YOUR_USERNAME/ebm-modmul'
api.create_repo(repo, private=True, exist_ok=True, repo_type='model')
api.upload_folder(folder_path='submission/ebm_modmul', repo_id=repo, repo_type='model')
print('uploaded', repo)
"

# get the full 40-char commit hash to submit
python -c "
from huggingface_hub import HfApi
print(HfApi().repo_info('YOUR_USERNAME/ebm-modmul').sha)
"
```

---

## Test it before submitting (recommended)

You can run the exact pipeline the organizers use, against your HF repo, with a **read-only** token:

1. Create a **fine-grained read-only token** at https://huggingface.co/settings/tokens
   (scope it to just this repo).
2. Run locally:

```bash
modchallenge evaluate-hf YOUR_USERNAME/ebm-modmul <commit_hash> --token hf_xxx --total 1100
```

If it reports the same numbers we see locally (`overall_accuracy`, per-tier), the repo, manifest, and
loader path are all good.

---

## Final submission

1. Make the repo **Public** (Settings → change visibility). *Private repos are not ranked.*
2. Submit your **`repo_id` + `commit_hash`** to the organizers (per the competition page /
   Contributor Network).
3. Organizers run the official eval with a secret seed and post results to the leaderboard.

## Updating later (new/better weights)

Each upload creates a **new commit hash**; the old one still points to the old version (nothing is
overwritten). So to submit improved weights:

1. Replace `submission/ebm_modmul/weights.pt` locally with the new checkpoint.
2. Re-run Method B's `upload_folder` (or drag the new `weights.pt` in via the web UI).
3. Grab the new `commit_hash` and submit that. The leaderboard keeps your **best** result per team.

## Current status / checklist

- [x] `model.py` + `manifest.json` written and committed (passes `modchallenge check`)
- [x] Validated locally: `overall_accuracy 0.175` (beats the 0.131 best baseline)
- [ ] `weights.pt` upgraded to a tier-2 ≥90% checkpoint (from the Colab grokking run)
- [ ] Uploaded to a HuggingFace repo
- [ ] Tested via `modchallenge evaluate-hf`
- [ ] Repo made public + submitted to organizers
