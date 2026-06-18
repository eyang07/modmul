"""Per-field teacher-forced error breakdown for a tier5_modmul checkpoint.

After an 80k run plateaus with tf_tok ~0.996 but the hard bucket stuck well below
0.90, the question is WHERE the residual ~0.4%/token error lives. End-to-end answer
survival ~= prod over tokens of (per-token correctness), so a single weak field caps
the whole chain. This loads a checkpoint, runs TEACHER-FORCED prediction over each
log-spaced prime bucket, and splits token accuracy by scratchpad field:

    d : q1 : m1 : r1 : pp : t : q2 : m2 : r2     (+ structural COLON/STEP/EOS)

Reading the result:
  * error concentrated in ONE field (likely pp = x*d, or the quotient digits q1/q2)
    -> targeted fix (sub-scratchpad / more supervision on that field) beats scaling.
  * error spread roughly evenly across fields -> global capacity/sharpening limit
    -> scale the model (bigger d_model / more layers) is the right lever.
Also prints, for the weakest wide field, accuracy by within-number digit position
(abacus index) -- tells us if failures are the high-order digits / carry positions.

Usage (on the pod, after the run finishes):
    python training/tier5_field_diag.py --ckpt training/checkpoints/modmul_t5_b100_best.pt \
        --n 400 --amp
"""

from __future__ import annotations

import argparse
import contextlib
import random
from collections import defaultdict

import torch

from tier5_modmul import (
    AbacusDecoder, build_prime_pool, digits_base, log_buckets, make_vocab,
    modmul_rows, pick_device,
)


def build_example_labeled(x, y, p, base, V, max_len):
    """Same token stream as tier5_modmul.build_example, but also returns a per-position
    field label (None for prompt/pad, else the field name)."""
    xd, yd, pd = digits_base(x, base), digits_base(y, base), digits_base(p, base)
    toks = [V["BOS"]] + xd + [V["MUL"]] + yd + [V["MOD"]] + pd + [V["EQ"]]
    abac = ([0] + list(range(len(xd))) + [0] + list(range(len(yd)))
            + [0] + list(range(len(pd))) + [0])
    labels = [None] * len(toks)

    def emit(tok, ab, lab):
        toks.append(tok); abac.append(ab); labels.append(lab)

    def emit_num(n, lab):
        for i, d in enumerate(digits_base(n, base)):
            emit(d, i, lab)

    def emit_digit(v, lab):
        emit(v, 0, lab)

    for i, (d, q1, m1, r1, pp, t, q2, m2, r2) in enumerate(modmul_rows(x, y, p, base)):
        if i > 0:
            emit(V["STEP"], 0, "STEP")
        emit_digit(d, "d")
        emit(V["COLON"], 0, "COLON"); emit_digit(q1, "q1")
        emit(V["COLON"], 0, "COLON"); emit_num(m1, "m1")
        emit(V["COLON"], 0, "COLON"); emit_num(r1, "r1")
        emit(V["COLON"], 0, "COLON"); emit_num(pp, "pp")
        emit(V["COLON"], 0, "COLON"); emit_num(t, "t")
        emit(V["COLON"], 0, "COLON"); emit_digit(q2, "q2")
        emit(V["COLON"], 0, "COLON"); emit_num(m2, "m2")
        emit(V["COLON"], 0, "COLON"); emit_num(r2, "r2")
    emit(V["EOS"], 0, "EOS")

    pad = max_len - len(toks)
    if pad < 0:
        raise ValueError(f"max_len {max_len} too small for sequence of {len(toks)}")
    toks += [V["PAD"]] * pad
    abac += [0] * pad
    labels += [None] * pad
    return toks, abac, labels


FIELD_ORDER = ["d", "q1", "m1", "r1", "pp", "t", "q2", "m2", "r2",
               "COLON", "STEP", "EOS"]


@torch.no_grad()
def field_breakdown(model, primes, base, V, n, max_len, rng, device, amp_ctx):
    """Teacher-forced token accuracy split by field, plus (field, within-num pos)."""
    model.eval()
    T, A, L = [], [], []
    for _ in range(n):
        p = primes[rng.randrange(len(primes))]
        x, y = rng.randrange(p), rng.randrange(p)
        toks, abac, labels = build_example_labeled(x, y, p, base, V, max_len)
        T.append(toks); A.append(abac); L.append(labels)
    toks = torch.tensor(T, dtype=torch.long, device=device)
    abac = torch.tensor(A, dtype=torch.long, device=device)
    with amp_ctx:
        pred = model(toks, abac)[:, :-1].argmax(-1)        # predicts positions 1..
    hit = (pred == toks[:, 1:]).to("cpu")                   # [n, L-1]

    correct = defaultdict(int); total = defaultdict(int)
    pos_correct = defaultdict(int); pos_total = defaultdict(int)  # (field, abac_idx)
    for b in range(n):
        labs, abs_ = L[b], A[b]
        for j in range(1, len(labs)):                       # target j-1 <-> token j
            lab = labs[j]
            if lab is None:
                continue
            h = bool(hit[b, j - 1])
            total[lab] += 1
            correct[lab] += h
            if lab in ("m1", "r1", "pp", "t", "m2", "r2"):  # multi-digit fields
                key = (lab, abs_[j])
                pos_total[key] += 1
                pos_correct[key] += h
    return correct, total, pos_correct, pos_total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=400, help="examples per bucket")
    ap.add_argument("--pool-size", type=int, default=40000)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--amp", action="store_true")
    args = ap.parse_args()

    device = pick_device()
    ck = torch.load(args.ckpt, map_location=device)
    cfg = ck["config"]
    B = cfg["base"]
    V = make_vocab(B)
    model = AbacusDecoder(V["VOCAB"], cfg["max_len"], cfg["abacus_max"],
                          d_model=cfg["d_model"], nhead=cfg["nhead"],
                          num_layers=cfg["layers"], dim_ff=cfg["dim_ff"]).to(device)
    model.load_state_dict(ck["model"])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"loaded {args.ckpt} | step {ck.get('step','?')} best {ck.get('best',-1):.3f} "
          f"| base {B} | d_model {cfg['d_model']} layers {cfg['layers']} "
          f"| params {n_params:,} | device {device}")

    pool_rng = random.Random(args.seed + 12345)
    POOL = build_prime_pool(cfg["p_min"], cfg["p_max"], args.pool_size, pool_rng)
    edges = log_buckets(cfg["p_min"], cfg["p_max"])
    rng = random.Random(args.seed)
    use_amp = args.amp and device.type == "cuda"
    amp_ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
               if use_amp else contextlib.nullcontext())

    for lo, hi in edges:
        ps = [p for p in POOL if lo <= p < hi]
        if not ps:
            continue
        correct, total, pc, pt = field_breakdown(
            model, ps, B, V, args.n, cfg["max_len"], rng, device, amp_ctx)
        tok_hit = sum(correct.values()); tok_tot = sum(total.values())
        print(f"\n=== bucket [{lo}-{hi}) | {len(ps)} primes | tf_tok "
              f"{tok_hit/max(1,tok_tot):.4f} ===")
        for f in FIELD_ORDER:
            if total.get(f):
                acc = correct[f] / total[f]
                bar = "#" * int(round((1 - acc) * 200))   # error magnitude bar
                print(f"  {f:6s} acc {acc:.4f}  err {1-acc:.4f}  n={total[f]:6d} {bar}")
        # within-number position breakdown for the weakest wide field
        wide = [f for f in ("m1", "r1", "pp", "t", "m2", "r2") if total.get(f)]
        if wide:
            worst = min(wide, key=lambda f: correct[f] / total[f])
            print(f"  -- {worst} by digit position (abacus idx, 0=MSB):")
            keys = sorted(k for k in pt if k[0] == worst)
            row = "     " + "  ".join(
                f"p{k[1]}:{pc[k]/pt[k]:.3f}" for k in keys)
            print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
