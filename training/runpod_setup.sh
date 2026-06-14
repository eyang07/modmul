#!/usr/bin/env bash
# One-shot environment check + readiness for a RunPod (or any cloud) GPU box.
# Usage (inside the pod terminal):
#   git clone -b ebm-dev https://github.com/eyang07/modular-arithmetic-challenge.git
#   cd modular-arithmetic-challenge && bash training/runpod_setup.sh
set -e

echo "==================== GPU ===================="
nvidia-smi -L || { echo "!! No GPU visible — did you launch a GPU pod?"; exit 1; }

echo "==================== Python / Torch ===================="
python -c "import torch; print('torch', torch.__version__, '| cuda:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
# Our training code is torch + stdlib + numpy. Torch is preinstalled on the
# PyTorch template; make sure numpy is present (no-op if already there).
python -c "import numpy" 2>/dev/null || pip install -q numpy
echo "numpy OK"

echo "==================== Ready ===================="
cat <<'EOF'
Recommended first runs (a fast GPU lets you use a bigger --batch):

# (A) Bank tier 2 — full coverage, no E6, fast (~minutes on an A100):
python training/train.py --arch cls_pp --tiers 1 2 --wd 0.05 \
    --steps 40000 --eval-every 2000 --batch 4096 --lr 5e-4 \
    --d-model 256 --layers 6 --tag t2_full
#   watch t2@seen -> aim >= 0.90

# (B) Give grokking a real shot — longer + E6 (still fast on A100):
python training/train.py --arch cls_pp --tiers 1 2 \
    --fixed-per-prime 2000 --holdout --wd 0.1 --alg-consistency 0.5 \
    --steps 50000 --eval-every 2000 --batch 2048 --lr 5e-4 \
    --d-model 256 --layers 6 --tag t2_grok_long
#   watch within-prime-unseen -> the jump = grok

Checkpoints are written to training/checkpoints/<tag>.pt
Retrieve them via the RunPod file browser (Jupyter), or push to HuggingFace.
REMEMBER to STOP/TERMINATE the pod when done so billing stops.
EOF
