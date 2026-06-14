# RunPod GPU — click-by-click setup

Goal: a fast GPU (A100 ~$1–2/hr, or RTX 4090 ~$0.3–0.5/hr) running our training in minutes
instead of hours. Our `training/` code is standalone (torch only), so it's just clone-and-run.

> What I (Claude) set up for you: the repo, the bootstrap script, and the exact commands.
> What only you can do: create the account, add credit, launch the pod (needs your login + card).

---

## 1. Account + credit (one-time, ~3 min)
1. Go to **runpod.io** → sign up.
2. **Billing → add credit** — $10 is plenty to start (a few hours of A100, or ~20 h of a 4090).

## 2. Launch a pod
1. Left menu → **Pods → + Deploy** (or **Deploy → GPU Pod**).
2. Pick a GPU:
   - **RTX 4090** (~$0.3–0.5/hr) — cheapest, ~5× a T4. Great default.
   - **A100 80GB** (~$1.6–2/hr) — fastest, use if you want sweeps to fly.
3. **Template:** choose a **PyTorch** template (e.g. "RunPod PyTorch 2.x"). Torch comes preinstalled.
4. Leave disk defaults. Click **Deploy On-Demand**. Wait ~1 min until status = **Running**.

## 3. Connect to the pod
On the running pod, click **Connect**, then either:
- **Start Web Terminal** (simplest — a terminal in your browser), or
- **Connect to Jupyter Lab** (gives a terminal **and** a file browser to download checkpoints).

I recommend **Jupyter Lab** (you'll want the file browser later). In Jupyter: **File → New → Terminal**.

## 4. Bootstrap (paste this one block into the terminal)
```bash
git clone -b ebm-dev https://github.com/eyang07/modular-arithmetic-challenge.git
cd modular-arithmetic-challenge
bash training/runpod_setup.sh
```
This checks the GPU, ensures deps, and prints the recommended run commands. You should see your GPU
name and `cuda: True`.

## 5. Run training
Paste one of the commands the bootstrap printed. The first one to run:
```bash
python training/train.py --arch cls_pp --tiers 1 2 --wd 0.05 \
    --steps 40000 --eval-every 2000 --batch 4096 --lr 5e-4 \
    --d-model 256 --layers 6 --tag t2_full
```
Watch **`t2@seen`** climb toward **≥ 0.90** (this is the full-coverage path that banks tier 2).
Paste the `step …` lines back to me and I'll read the trajectory.

> Tip: to keep a run alive if your laptop disconnects, prefix with `nohup ... &` and `tail -f`,
> or run it inside `tmux`. For a first run, just leaving the browser tab open is fine.

## 6. Get the checkpoint back
The trained weights land at `training/checkpoints/t2_full.pt`. To retrieve:
- **Easiest (Jupyter):** in the Jupyter file browser, navigate to
  `modular-arithmetic-challenge/training/checkpoints/`, right-click the `.pt` → **Download**.
- **Or push to HuggingFace** from the pod (see `submission/UPLOAD_TO_HUGGINGFACE.md`).

Then drop it into `submission/ebm_modmul/weights.pt` locally and we re-run `modchallenge evaluate`.

## 7. ⚠️ STOP the pod when done
Billing runs while the pod exists. When finished: **Pods → … → Stop** (pauses, small disk fee) or
**Terminate** (deletes, $0). Terminate once you've downloaded the checkpoint.

---

## Cost sanity
- A full tier-2 + tier-0-probe exploration burst ≈ a few GPU-hours ≈ **$5–15** on a 4090.
- The whole tier-3 phase, even with sweeps, is a few tens of dollars. See
  [COMPUTE_BACKUP_PLAN.md](COMPUTE_BACKUP_PLAN.md).
