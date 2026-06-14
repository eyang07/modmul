# Compute backup plan (escalate to a paid GPU when the wait is the bottleneck)

## Current setup (default)
Free **Colab T4**. Cost $0. Reality:
- ~0.6 s/step with E6 (`--alg-consistency` does ~4 forward passes/step), ~0.18 s/step without E6.
- A 20k-step E6 run ≈ **3.4 h**; a no-E6 run ≈ **1 h**.
- Session timeouts; not great for unattended multi-run sweeps.

Good enough for **single long runs** we can launch and walk away from.

## When to escalate (the trigger)
Switch to a paid GPU when **iteration speed becomes the bottleneck**, i.e. any of:
- We need to **sweep** hyperparameters (weight decay, E6 weight, fixed-per-prime, lr) — many runs.
- The **tier-0 multiplication probe** + tier-3 experiments need fast turnaround to stay bounded.
- A single run is too long to iterate on the same day (E6 runs at ~3.4 h each).
- We're running several configs in parallel (Colab gives one GPU; timeouts interrupt).

If we're just letting one run cook overnight, **stay on free Colab**.

## The backup: rent an A100/4090 (RunPod, or Lambda / Vast.ai)
- **Speed:** ~5–10× a T4. A 3.4 h E6 run → **~20–30 min**. Sweeps become same-day.
- **Cost:** ~$0.2–0.5/hr (RTX 4090) or ~$1–2/hr (A100). A full exploration burst of
  ~20–40 GPU-hours ≈ **$20–80** total — trivial vs. the time saved.
- **Why it's easy for us:** `training/` is standalone (torch + stdlib only), so the exact same
  commands run anywhere — no project-specific setup.

### RunPod quickstart (the recommended path)
1. runpod.io → add a few $ credit → **Deploy** a Community-Cloud pod with a 4090 or A100,
   "PyTorch" template (torch preinstalled).
2. Connect (web terminal or SSH).
3. Same three commands as Colab:
   ```bash
   git clone -b ebm-dev https://github.com/eyang07/modular-arithmetic-challenge.git
   cd modular-arithmetic-challenge
   python training/train.py --arch cls_pp --tiers 1 2 ... --tag <run>    # our usual commands
   ```
4. Copy checkpoints back: `runpodctl send training/checkpoints/<tag>.pt`, or push to HuggingFace,
   or `scp`. **Stop/terminate the pod when done** so billing stops.

> Tip: bigger GPU → use a **bigger batch** (`--batch 4096+`) to actually use the hardware.

## Speed levers we control (independent of hardware)
- Drop E6 (`--alg-consistency 0`) for coverage/memorization runs → ~3× faster.
- Larger `--batch` on a bigger GPU.
- Reserve E6 (the 4×-cost regularizer) for the runs where structure/generalization is the point.

## Decision summary
- **Single overnight run →** free Colab T4.
- **Sweeps / fast iteration / the tier-0 probe loop →** spin up a RunPod 4090 (~$0.3/hr) for a few
  hours, run the batch of experiments, pull the checkpoints, terminate. Budget a few tens of dollars
  for the whole tier-3 exploration phase.
