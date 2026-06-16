"""Per-field teacher-forced accuracy for a modmul checkpoint.

The aggregate tf_tok can't tell us WHICH part of the d:q1:r1:pp:q2:r2 scratchpad is
dragging. This loads a (latest) modmul checkpoint and breaks teacher-forced accuracy
down by field, so we can see if pp = x*d (the single-digit multiply, the one new
sub-skill) is the bottleneck. Non-disruptive: only reads the checkpoint.

Usage (on the pod, while training runs):
    python training/modmul_diag.py --ckpt training/checkpoints/modmul.pt --n 500
"""

from __future__ import annotations

import argparse
import random
from collections import defaultdict

import torch

import modmul_probe as M
from data import primes_for_tier

FIELDS = {0: "d", 1: "q1", 2: "r1", 3: "pp", 4: "t", 5: "q2", 6: "r2"}


def field_ids(toks: list[int]) -> list[int]:
    """Per-position field id for a built example: 0..5 within a block (d,q1,r1,pp,q2,r2),
    or -1 (prompt / pad / delimiter / eos)."""
    ids = [-1] * len(toks)
    after_eq = False
    field = 0
    for i, t in enumerate(toks):
        if t == M.EQ:
            after_eq = True; field = 0; continue
        if not after_eq:
            continue
        if t == M.STEP:
            field = 0; continue
        if t == M.COLON:
            field += 1; continue
        if t in (M.EOS, M.PAD):
            continue
        ids[i] = field
    return ids


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="training/checkpoints/modmul.pt")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--p-min", type=int, default=512)
    ap.add_argument("--p-max", type=int, default=65536)
    ap.add_argument("--seed", type=int, default=999)
    args = ap.parse_args()

    device = M.pick_device()
    ck = torch.load(args.ckpt, map_location=device)
    c = ck["config"]
    model = M.AbacusDecoder(max_len=c["max_len"], abacus_max=c["abacus_max"],
                            d_model=c["d_model"], nhead=c["nhead"],
                            num_layers=c["layers"], dim_ff=c["dim_ff"]).to(device)
    model.load_state_dict(ck["model"]); model.eval()
    print(f"loaded {args.ckpt} | step {ck.get('step')} | best {ck.get('best')}")

    POOL = [p for p in primes_for_tier(3) if args.p_min <= p < args.p_max]
    rng = random.Random(args.seed)
    toks, abac, mask = M.make_batch(args.n, POOL, c["max_len"], rng, device)
    with torch.no_grad():
        pred = model(toks, abac)[:, :-1].argmax(-1)
    target = toks[:, 1:]
    hit = (pred == target)

    # bucket by the field of the TARGET token (toks[:,1:])
    tg = target.tolist()
    hh = hit.tolist()
    mm = mask[:, 1:].tolist()
    per = defaultdict(lambda: [0, 0])   # field -> [correct, total]
    tok_lists = toks.tolist()
    for b in range(args.n):
        ids = field_ids(tok_lists[b])
        for t in range(len(tg[b])):
            if not mm[b][t]:
                continue
            fid = ids[t + 1]            # target token is toks[t+1]
            if fid < 0:
                fid = -1               # delimiters/eos that are supervised
            per[fid][0] += int(hh[b][t]); per[fid][1] += 1

    print(f"{'field':>6} | {'tok-acc':>8} | {'count':>7}")
    overall_c = overall_t = 0
    for fid in sorted(per, key=lambda k: (k < 0, k)):
        c_, t_ = per[fid]
        overall_c += c_; overall_t += t_
        name = FIELDS.get(fid, "delim/eos" if fid == -1 else str(fid))
        print(f"{name:>6} | {c_/max(1,t_):8.4f} | {t_:7d}")
    print(f"{'ALL':>6} | {overall_c/max(1,overall_t):8.4f} | {overall_t:7d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
