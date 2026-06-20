"""Bundle the trained base-16 tier-5 decoder into the submission weights.pt.

The submission's weights.pt already carries tiers 1-2 (cls head) + tier3 + tier4.
This adds a ``tier5`` entry = {config, state_dict} from the base-16 modmul checkpoint,
so submission/ebm_modmul/model.py routes p in [2^32, 2^64) to the base-16 scratchpad.

It (1) backs up the current weights.pt, (2) verifies the base-16 state_dict loads
into the submission's AbacusDecoder(vocab=24) cleanly, (3) writes the merged weights.

Run on the pod (where the tier-5 checkpoint lives):
    python training/bundle_tier5.py \
        --weights submission/ebm_modmul/weights.pt \
        --tier5-ckpt training/checkpoints/modmul_t5_b16_d512_best.pt
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="submission/ebm_modmul/weights.pt")
    ap.add_argument("--tier5-ckpt", default="training/checkpoints/modmul_t5_b16_d512_best.pt")
    ap.add_argument("--backup", default="submission/ebm_modmul/weights_tier1234_backup.pt")
    args = ap.parse_args()

    w = torch.load(args.weights, map_location="cpu", weights_only=False)
    t5 = torch.load(args.tier5_ckpt, map_location="cpu", weights_only=False)
    cfg = t5["config"]
    print(f"weights.pt tiers present: arch={w.get('arch')} "
          f"tier3={'tier3' in w} tier4={'tier4' in w} tier5={'tier5' in w}")
    print(f"tier5 ckpt: step {t5.get('step')} best {t5.get('best'):.3f} "
          f"| base {cfg['base']} max_len {cfg['max_len']} d_model {cfg['d_model']} "
          f"layers {cfg['layers']}")

    # Parity check: the base-16 state_dict must load into the submission decoder.
    sys.path.insert(0, str(Path(args.weights).resolve().parent))
    from model import AbacusDecoder  # noqa: E402
    m = AbacusDecoder(max_len=cfg["max_len"], abacus_max=cfg["abacus_max"],
                      d_model=cfg["d_model"], nhead=cfg["nhead"],
                      num_layers=cfg["layers"], dim_ff=cfg["dim_ff"],
                      vocab=cfg["base"] + 8)
    missing, unexpected = m.load_state_dict(t5["model"], strict=True)
    print(f"state_dict parity OK (missing={list(missing)}, unexpected={list(unexpected)})")

    w["tier5"] = {"config": cfg, "state_dict": t5["model"]}

    if not Path(args.backup).exists():
        shutil.copy2(args.weights, args.backup)
        print(f"backed up original -> {args.backup}")
    else:
        print(f"backup already exists ({args.backup}); not overwriting")

    torch.save(w, args.weights)
    sz = Path(args.weights).stat().st_size / 1e6
    print(f"wrote {args.weights} ({sz:.0f} MB) with tiers 1-2 + tier3 + tier4 + tier5")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
