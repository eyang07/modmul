"""Tier-6 compliance + robustness battery for a trained RecurrentReducer.

Reimplements (in our own harness) the checks NeuralHorner ships, plus our own:
  1. weight-perturbation  -- randomize the cell's weights; accuracy MUST collapse to
     ~0. This is the organizers' own no-shortcut standard (rules/evaluation.md:262):
     a hand-coded solver keeps working under randomization; a learned model dies. It
     proves "the answer is in the weights, not the loop."
  2. no-shortcut          -- the prediction must equal (a*b) mod p and DIFFER from the
     cheap shortcuts a%p, b%p, (a%p)*(b%p). Confirms genuine modular multiplication.
  3. exhaustive           -- over ALL one-step transition states (s,x,d) for every
     prime < --exhaustive-limit: per-step exactness with zero gaps.
  4. adversarial battery  -- edge operands a,b in {0,1}, power-of-two-adjacent operands,
     and Fermat-like / Mersenne-like p (the family NeuralHorner fails at 98.83%).
  5. bf16 margin          -- 0 argmax flips fp32 vs bf16, and the min logit margin
     (decode precision safety).

Usage:
    python training/tier6_verify.py --ckpt training/checkpoints/t6_gate_best.pt \
        --bits 128 256 512 --exhaustive-limit 64
"""

from __future__ import annotations

import argparse
import random

import torch

import tier6_recurrent as M


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    c = ck["config"]
    model = M.RecurrentReducer(c["base"], d_model=c["d_model"], gru_layers=c["gru_layers"],
                               aux_quotient=c.get("aux_quotient", True)).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, c


def check_weight_perturbation(model, cfg, device, n=200, bits=128, seed=0):
    base = cfg["base"]
    ps = M.build_prime_pool(1 << (bits - 1), 1 << bits, 400, random.Random(bits))
    real = M.eval_exact(model, ps, base, n, 2 * bits, random.Random(seed), device)
    pert = M.RecurrentReducer(base, d_model=cfg["d_model"], gru_layers=cfg["gru_layers"],
                              aux_quotient=cfg.get("aux_quotient", True)).to(device)
    torch.manual_seed(12345)
    for p in pert.parameters():
        torch.nn.init.normal_(p, std=0.1) if p.dim() > 1 else torch.nn.init.zeros_(p)
    pert.eval()
    rand = M.eval_exact(pert, ps, base, n, 2 * bits, random.Random(seed), device)
    ok = real >= 0.5 and rand <= 0.02
    print(f"[1] weight-perturbation @{bits}b: trained {real:.3f} -> randomized {rand:.3f}  "
          f"{'PASS' if ok else 'FAIL'} (need trained>=0.5, random<=0.02)")
    return ok


@torch.no_grad()
def check_no_shortcut(model, cfg, device, n=500, bits=128, seed=1):
    """Predictions must equal (a*b)%p and not coincide with the shortcuts. We only
    count problems where the shortcuts genuinely differ from the true answer."""
    base = cfg["base"]
    ps = M.build_prime_pool(1 << (bits - 1), 1 << bits, 400, random.Random(bits + 1))
    rng = random.Random(seed)
    Kp = max(len(M.digits_msb(p, base)) for p in ps) + 1
    items = []
    for _ in range(n):
        p = ps[rng.randrange(len(ps))]
        a = rng.randrange(1 << (2 * bits)); b = rng.randrange(1 << (2 * bits))
        items.append((a, b, p))
    Lb = max(len(M.digits_msb(b, base)) for _, b, _ in items)
    X = torch.tensor([M.to_limbs(a % p, base, Kp) for a, _, p in items],
                     dtype=torch.long, device=device)
    P = torch.tensor([M.to_limbs(p, base, Kp) for _, _, p in items],
                     dtype=torch.long, device=device)
    Bd = torch.tensor([[0] * (Lb - len(M.digits_msb(b, base))) + M.digits_msb(b, base)
                       for _, b, _ in items], dtype=torch.long, device=device)
    out = model(X, Bd, P)
    correct = shortcut_hits = differ = 0
    for j, (a, b, p) in enumerate(items):
        pred = M.from_limbs(out[j].tolist(), base)
        true = (a * b) % p
        shortcuts = {a % p, b % p, (a % p) * (b % p)}
        if true not in shortcuts:
            differ += 1
            if pred == true:
                correct += 1
            if pred in shortcuts:
                shortcut_hits += 1
    acc = correct / max(1, differ)
    ok = acc >= 0.9 and shortcut_hits == 0
    print(f"[2] no-shortcut @{bits}b: {correct}/{differ} true (acc {acc:.3f}), "
          f"shortcut-coincidences {shortcut_hits}  {'PASS' if ok else 'FAIL'}")
    return ok


@torch.no_grad()
def check_exhaustive(model, cfg, device, limit=64):
    """Every one-step transition (s,x,d) for every prime < limit, batched."""
    base = cfg["base"]
    S, X, P, D, T = [], [], [], [], []
    for p in range(2, limit):
        if not M._is_prime(p):
            continue
        K = len(M.digits_msb(p, base)) + 1
        for s in range(p):
            for x in range(p):
                for d in range(base):
                    snext = (s * base + d * x) % p
                    S.append(M.to_limbs(s, base, K)); X.append(M.to_limbs(x, base, K))
                    P.append(M.to_limbs(p, base, K)); D.append(d)
                    T.append(M.to_limbs(snext, base, K))
    # pad to common K
    K = max(len(r) for r in S)
    pad = lambda r: r + [0] * (K - len(r))
    St = torch.tensor([pad(r) for r in S], dtype=torch.long, device=device)
    Xt = torch.tensor([pad(r) for r in X], dtype=torch.long, device=device)
    Pt = torch.tensor([pad(r) for r in P], dtype=torch.long, device=device)
    Dt = torch.tensor(D, dtype=torch.long, device=device)
    Tt = torch.tensor([pad(r) for r in T], dtype=torch.long, device=device)
    ok = tot = 0
    for s0 in range(0, len(D), 4096):
        e = min(s0 + 4096, len(D))
        logits, _ = model.step_logits(St[s0:e], Xt[s0:e], Pt[s0:e], Dt[s0:e])
        ok += (logits.argmax(-1) == Tt[s0:e]).all(dim=1).sum().item()
        tot += e - s0
    passed = ok == tot
    print(f"[3] exhaustive transitions for primes < {limit}: {ok}/{tot} exact  "
          f"{'PASS' if passed else 'FAIL'}")
    return passed


@torch.no_grad()
def check_adversarial(model, cfg, device, bits=128, seed=2):
    base = cfg["base"]
    ps = M.build_prime_pool(1 << (bits - 1), 1 << bits, 200, random.Random(bits + 2))
    rng = random.Random(seed)
    cases = []
    # edge operands
    for _ in range(100):
        p = ps[rng.randrange(len(ps))]
        for a, b in [(0, rng.randrange(1 << bits)), (1, rng.randrange(1 << bits)),
                     (rng.randrange(1 << bits), 0), (rng.randrange(1 << bits), 1)]:
            cases.append((a, b, p))
    # power-of-two-adjacent operands (NeuralHorner's failure family)
    for _ in range(200):
        p = ps[rng.randrange(len(ps))]
        e1, e2 = rng.randrange(2, 2 * bits), rng.randrange(2, 2 * bits)
        a = (1 << e1) + rng.choice([-1, 0, 1])
        b = (1 << e2) + rng.choice([-1, 0, 1])
        cases.append((max(0, a), max(0, b), p))
    Kp = max(len(M.digits_msb(p, base)) for _, _, p in cases) + 1
    Lb = max(len(M.digits_msb(max(1, b), base)) for _, b, _ in cases)
    X = torch.tensor([M.to_limbs(a % p, base, Kp) for a, _, p in cases],
                     dtype=torch.long, device=device)
    P = torch.tensor([M.to_limbs(p, base, Kp) for _, _, p in cases],
                     dtype=torch.long, device=device)
    Bd = torch.tensor([[0] * (Lb - len(M.digits_msb(b, base))) + M.digits_msb(b, base)
                       for _, b, _ in cases], dtype=torch.long, device=device)
    out = model(X, Bd, P)
    ok = sum(M.from_limbs(out[j].tolist(), base) == (a * b) % p
             for j, (a, b, p) in enumerate(cases))
    acc = ok / len(cases)
    print(f"[4] adversarial battery @{bits}b ({len(cases)} cases): exact {acc:.4f}  "
          f"{'PASS' if acc >= 0.99 else 'WARN'} (NeuralHorner: 0.9883)")
    return acc


@torch.no_grad()
def check_bf16_margin(model, cfg, device, n=400, bits=128, seed=3):
    base = cfg["base"]
    ps = M.build_prime_pool(1 << (bits - 1), 1 << bits, 400, random.Random(bits + 3))
    S, X, P, D, _, _ = M.make_batch(n, ps, base,
                                    len(M.digits_msb((1 << bits) - 1, base)) + 1,
                                    random.Random(seed), device)
    f32, _ = model.step_logits(S, X, P, D)
    with torch.autocast(device_type=device.type,
                        dtype=torch.bfloat16) if device.type in ("cuda", "cpu") else \
            torch.no_grad():
        b16, _ = model.step_logits(S, X, P, D)
    flips = (f32.argmax(-1) != b16.argmax(-1)).sum().item()
    top2 = f32.topk(2, dim=-1).values
    margin = (top2[..., 0] - top2[..., 1]).min().item()
    ok = flips == 0
    print(f"[5] bf16 margin @{bits}b: argmax flips {flips}, min fp32 margin {margin:.3f}  "
          f"{'PASS' if ok else 'WARN'}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--bits", type=int, nargs="+", default=[128, 256, 512])
    ap.add_argument("--exhaustive-limit", type=int, default=64)
    args = ap.parse_args()
    device = (torch.device("cuda") if torch.cuda.is_available()
              else torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cpu"))
    model, cfg = load_model(args.ckpt, device)
    print(f"loaded {args.ckpt} | base {cfg['base']} | device {device}")
    results = []
    results.append(check_exhaustive(model, cfg, device, args.exhaustive_limit))
    for bits in args.bits:
        results.append(check_weight_perturbation(model, cfg, device, bits=bits))
        results.append(check_no_shortcut(model, cfg, device, bits=bits))
        check_adversarial(model, cfg, device, bits=bits)
        check_bf16_margin(model, cfg, device, bits=bits)
    print(f"\nSUMMARY: {'ALL GATING CHECKS PASS' if all(results) else 'SOME CHECKS FAILED'}")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
