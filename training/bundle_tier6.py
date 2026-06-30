"""Bundle the trained tier-6+ recurrent reduction cell into the submission weights.pt.

The submission's weights.pt carries tiers 1-2 (cls head) + tier3 + tier4 + tier5.
This adds a ``tier6`` entry = {config, state_dict} from the recurrent-cell checkpoint,
so submission/ebm_modmul/model.py routes p >= 2^64 (tiers 6-10) to the shared cell
instead of emitting the [0] fallback.

It (1) verifies the cell's state_dict loads into the submission-shape RecurrentReducer
cleanly, (2) backs up the current weights.pt, (3) writes the merged weights.

Run on the pod (where the tier-6 checkpoint lives):
    python training/bundle_tier6.py \
        --weights submission/ebm_modmul/weights.pt \
        --tier6-ckpt training/checkpoints/t6_gate_best.pt
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="submission/ebm_modmul/weights.pt")
    ap.add_argument("--tier6-ckpt", default="training/checkpoints/t6_gate_best.pt")
    ap.add_argument("--backup", default="submission/ebm_modmul/weights_pre_tier6_backup.pt")
    args = ap.parse_args()

    w = torch.load(args.weights, map_location="cpu", weights_only=False)
    t6 = torch.load(args.tier6_ckpt, map_location="cpu", weights_only=False)
    cfg = t6["config"]
    print(f"weights.pt tiers present: arch={w.get('arch')} "
          f"tier3={'tier3' in w} tier4={'tier4' in w} tier5={'tier5' in w} "
          f"tier6={'tier6' in w}")
    print(f"tier6 ckpt: step {t6.get('step')} best {t6.get('best')} "
          f"| base {cfg['base']} d_model {cfg['d_model']} gru_layers {cfg['gru_layers']} "
          f"aux_q {cfg.get('aux_quotient', True)} K {cfg.get('K')}")

    # Parity check: load into the training RecurrentReducer (identical module structure
    # / state_dict keys to the submission's, importable with only torch).
    from tier6_recurrent import RecurrentReducer  # noqa: E402
    m = RecurrentReducer(cfg["base"], d_model=cfg["d_model"], gru_layers=cfg["gru_layers"],
                         aux_quotient=cfg.get("aux_quotient", True))
    m.load_state_dict(t6["model"], strict=True)
    print("state_dict parity OK")

    w["tier6"] = {"config": cfg, "state_dict": t6["model"]}

    if not Path(args.backup).exists():
        shutil.copy2(args.weights, args.backup)
        print(f"backed up original -> {args.backup}")
    else:
        print(f"backup already exists ({args.backup}); not overwriting")

    torch.save(w, args.weights)
    sz = Path(args.weights).stat().st_size / 1e6
    print(f"wrote {args.weights} ({sz:.0f} MB) with tiers 1-2 + tier3 + tier4 + tier5 + tier6")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
