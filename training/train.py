"""Train the modular-multiplication predictor (tiers 1-2 first).

Two validation metrics, both exact-match (all WIDTH digits correct):
  - val@seen-primes  : unseen residue pairs on primes that were in training
                       (within-prime generalisation)
  - val@unseen-primes: held-out primes (cross-prime generalisation)

For tier 2 the prime pool is small and fully enumerable, so the default trains on
*all* tier-2 primes (holdout_frac=0) — full coverage is the legitimate route to
>=90%. Pass --holdout to instead hold some primes out and measure cross-prime
generalisation (the signal that matters for tier 3+).

Usage:
    .venv/bin/python training/train.py --minutes 10
    .venv/bin/python training/train.py --steps 8000 --holdout
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import torch
import torch.nn as nn

import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from data import (  # noqa: E402
    build_prime_split,
    make_batch,
    make_eval_batch,
    build_fixed_dataset,
    sample_fixed_batch,
    make_unseen_pair_eval,
)
import math  # noqa: E402

from model import (  # noqa: E402
    ModMulNet, JointModMulNet, JointModMulNetCls, JointModMulNetAngular,
    JointModMulNetClsPP,
)
from data import WIDTH  # noqa: E402

ARCHS = {
    "additive": ModMulNet,
    "joint": JointModMulNet,
    "cls": JointModMulNetCls,
    "cls_pp": JointModMulNetClsPP,   # per-prime embedding
    "angular": JointModMulNetAngular,
}

# Place values to turn fixed-width MSB-first decimal digits into an integer.
_PV = [10 ** (WIDTH - 1 - i) for i in range(WIDTH)]


def digits_to_int(ans_dig: torch.Tensor) -> torch.Tensor:
    """(B, WIDTH) MSB-first decimal digits -> (B,) integer residue labels."""
    pv = torch.tensor(_PV, dtype=ans_dig.dtype, device=ans_dig.device)
    return (ans_dig * pv).sum(dim=1)


# -- angular (Saxena-Charton) target / loss / decode --------------------------

def angular_target(ans_int: torch.Tensor, p_int: torch.Tensor) -> torch.Tensor:
    """Residue t (mod p) -> unit-circle point (cos 2pi t/p, sin 2pi t/p)."""
    theta = (2 * math.pi) * ans_int.float() / p_int.float()
    return torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)  # (B, 2)


def angular_loss(pred: torch.Tensor, ans_int, p_int, alpha: float = 1e-4) -> torch.Tensor:
    """alpha*(r^2 + 1/r^2) anti-collapse term + squared distance to target."""
    tgt = angular_target(ans_int, p_int)
    r2 = (pred ** 2).sum(dim=1).clamp_min(1e-8)
    anti_collapse = alpha * (r2 + 1.0 / r2)
    dist = ((pred - tgt) ** 2).sum(dim=1)
    return (anti_collapse + dist).mean()


def angular_decode(pred: torch.Tensor, p_int: torch.Tensor) -> torch.Tensor:
    """(x',y') -> nearest residue t_hat in [0, p): round(angle * p / 2pi) mod p."""
    theta = torch.atan2(pred[:, 1], pred[:, 0])  # (-pi, pi]
    t = torch.round(theta * p_int.float() / (2 * math.pi))
    return (t % p_int.float()).long()


# -- E6: algebraic-consistency losses (ring/group axioms; Kona-inspired) ------

def _ints_to_digits(n: torch.Tensor) -> torch.Tensor:
    """(B,) int tensor -> (B, WIDTH) MSB-first decimal digits."""
    pv = torch.tensor(_PV, device=n.device)
    return ((n.unsqueeze(1) // pv) % 10).long()


def _unit_vec(out: torch.Tensor, p_int: torch.Tensor, mode: str) -> torch.Tensor:
    """Map a model output to a unit vector on the circle (the predicted residue's
    angle). For angular: normalise the 2-vector. For cls/cls_pp: circular mean of
    the softmax over residue classes (class c -> angle 2*pi*c/p), then normalise."""
    if mode == "angular":
        u = out
    else:
        k = out.shape[1]
        cls_idx = torch.arange(k, device=out.device).unsqueeze(0).float()   # (1,K)
        ang = (2 * math.pi) * cls_idx / p_int.unsqueeze(1).float()          # (B,K)
        probs = torch.softmax(out, dim=-1) * (cls_idx < p_int.unsqueeze(1).float())
        ux = (probs * torch.cos(ang)).sum(1)
        uy = (probs * torch.sin(ang)).sum(1)
        u = torch.stack([ux, uy], dim=1)
    return u / u.norm(dim=1, keepdim=True).clamp_min(1e-6)


def _cmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Complex multiplication of (B,2) unit vectors: adds their angles."""
    return torch.stack(
        [a[:, 0] * b[:, 0] - a[:, 1] * b[:, 1], a[:, 0] * b[:, 1] + a[:, 1] * b[:, 0]],
        dim=1,
    )


def alg_consistency_loss(model, out_xy, x, y, p, mode: str) -> torch.Tensor:
    """Self-supervised structural penalties (no labels):

    - commutativity:  f(x,y) == f(y,x)
    - distributivity: x*y + x*z == x*(y+z) (mod p), which in angle space is the
      complex-product identity u(x,y) * u(x,z) == u(x,(y+z) mod p).

    Both regularise the model on the *whole* residue grid, pushing it to satisfy
    the ring axioms (= learn the algorithm) rather than memorise. Adds 3 forwards.
    """
    p_int = digits_to_int(p)
    u_xy = _unit_vec(out_xy, p_int, mode)
    # commutativity
    u_yx = _unit_vec(model(y, x, p), p_int, mode)
    comm = ((u_xy - u_yx) ** 2).sum(1).mean()
    # distributivity: sample z in [0,p), build s = (y+z) mod p
    y_int = digits_to_int(y)
    z_int = (torch.rand(x.shape[0], device=x.device) * p_int.float()).floor().long()
    z_int = torch.minimum(z_int, p_int - 1)
    s_int = (y_int + z_int) % p_int
    u_xz = _unit_vec(model(x, _ints_to_digits(z_int), p), p_int, mode)
    u_xs = _unit_vec(model(x, _ints_to_digits(s_int), p), p_int, mode)
    dist = ((_cmul(u_xy, u_xz) - u_xs) ** 2).sum(1).mean()
    return comm + dist

CKPT_DIR = HERE / "checkpoints"


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def score_batch(model, x, y, p, ans, mode: str) -> float:
    """Exact-match accuracy on a ready-made batch, per output mode."""
    if x.shape[0] == 0:
        return float("nan")
    out = model(x, y, p)
    if mode in ("cls", "cls_pp"):
        return (out.argmax(dim=-1) == digits_to_int(ans)).float().mean().item()
    if mode == "angular":
        return (angular_decode(out, digits_to_int(p)) == digits_to_int(ans)).float().mean().item()
    return (out.argmax(dim=-1) == ans).all(dim=1).float().mean().item()  # digit heads


@torch.no_grad()
def exact_match(model, primes: list[int], device, n: int, rng, mode: str) -> float:
    """Fresh-sample exact-match over the given primes (infinite-stream eval)."""
    if not primes:
        return float("nan")
    x, y, p, ans = make_eval_batch(primes, n, rng, device)
    return score_batch(model, x, y, p, ans, mode)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=10.0)
    ap.add_argument("--steps", type=int, default=0, help="override: train N steps instead of by time")
    ap.add_argument("--tiers", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--holdout", action="store_true", help="hold out 10%% of primes for cross-prime val")
    ap.add_argument("--arch", choices=list(ARCHS), default="joint")
    ap.add_argument("--p-max", type=int, default=256, help="num residue classes for --arch cls")
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.01, help="weight decay (key grokking lever)")
    ap.add_argument("--alg-consistency", type=float, default=0.0,
                    help="weight for E6 algebraic-consistency loss (commutativity + "
                         "distributivity); 0 = off. Only for cls/cls_pp/angular.")
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--fixed-per-prime", type=int, default=0,
                    help="0 = infinite fresh stream; >0 = grokking recipe on a fixed "
                         "set of this many (x,y) pairs per prime")
    ap.add_argument("--max-primes", type=int, default=0,
                    help="0 = all; else cap each tier's TRAIN pool to this many primes "
                         "(for the cheap E2 linchpin: e.g. 8). Held-out val primes unaffected.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", type=str, default="", help="checkpoint name; default derives from arch+tiers")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device()
    tiers = tuple(args.tiers)
    holdout = 0.1 if args.holdout else 0.0
    split = build_prime_split(tiers, holdout_frac=holdout, seed=args.seed)
    if args.max_primes > 0:
        for t in tiers:
            split.train[t] = split.train[t][: args.max_primes]

    # Curriculum weights: bias toward the harder/larger tier once tier 1 is easy.
    # Tier 1 has only 4 primes so it groks fast; give tier 2 the bulk of samples.
    default_w = {1: 0.2, 2: 0.8, 3: 1.0}
    tier_weights = {t: default_w.get(t, 1.0) for t in tiers}

    print(f"device: {device}")
    for t in tiers:
        print(f"  tier {t}: train primes={len(split.train[t])} "
              f"val(unseen) primes={len(split.val_primes[t]) if holdout else 0}")

    mode = args.arch
    is_cls = mode in ("cls", "cls_pp")
    kw = dict(d_model=args.d_model, num_layers=args.layers)
    if is_cls:
        kw["p_max"] = args.p_max
    model = ARCHS[args.arch](**kw).to(device)
    n_params = sum(q.numel() for q in model.parameters())
    print(f"arch: {args.arch} | params: {n_params:,}" + (f" | p_max={args.p_max}" if is_cls else ""))

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    if args.steps > 0:
        total_steps = args.steps
    else:
        total_steps = max(1, int(args.minutes * 60 * 6))  # ~6 steps/s rough budget
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=total_steps, eta_min=args.lr * 0.1
    )
    loss_fn = nn.CrossEntropyLoss()
    rng = random.Random(args.seed)
    eval_rng = random.Random(12345)

    CKPT_DIR.mkdir(exist_ok=True)
    tag = args.tag or f"{args.arch}_t{''.join(map(str, tiers))}"
    out_path = CKPT_DIR / f"{tag}.pt"
    best = -1.0

    # Grokking recipe: a fixed finite train set the model can overfit, instead of
    # the infinite fresh stream. Pool all train primes across the requested tiers.
    fixed_ds = None
    if args.fixed_per_prime > 0:
        all_train_primes = sorted({p for t in tiers for p in split.train[t]})
        fixed_ds = build_fixed_dataset(all_train_primes, args.fixed_per_prime, args.seed, device)
        print(f"FIXED dataset: {fixed_ds.X.shape[0]:,} rows over {len(all_train_primes)} primes "
              f"({args.fixed_per_prime} pairs/prime)")

    deadline = time.monotonic() + args.minutes * 60
    start = time.monotonic()
    step = 0

    def time_left() -> bool:
        return step < total_steps if args.steps > 0 else time.monotonic() < deadline

    while time_left():
        model.train()
        if fixed_ds is not None:
            x, y, p, ans = sample_fixed_batch(fixed_ds, args.batch, rng)
        else:
            x, y, p, ans = make_batch(split.train, tier_weights, args.batch, rng, device)
        out = model(x, y, p)
        if mode in ("cls", "cls_pp"):
            loss = loss_fn(out, digits_to_int(ans))            # (B, p_max) vs (B,)
        elif mode == "angular":
            loss = angular_loss(out, digits_to_int(ans), digits_to_int(p))
        else:
            loss = loss_fn(out.reshape(-1, 10), ans.reshape(-1))  # digit heads

        if args.alg_consistency > 0 and mode in ("cls", "cls_pp", "angular"):
            loss = loss + args.alg_consistency * alg_consistency_loss(model, out, x, y, p, mode)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        step += 1

        if step % args.eval_every == 0:
            model.eval()
            if fixed_ds is not None:
                # Grokking signals: fit on the fixed train set, within-prime
                # generalisation to unseen pairs, cross-prime to held-out primes.
                idx = torch.randint(0, fixed_ds.X.shape[0], (4000,), device=device)
                fit = score_batch(model, fixed_ds.X[idx], fixed_ds.Y[idx],
                                  fixed_ds.P[idx], fixed_ds.ANS[idx], mode)
                seen_primes = sorted({p for t in tiers for p in split.train[t]})
                xe, ye, pe, ae = make_unseen_pair_eval(fixed_ds, seen_primes, 4000, eval_rng, device)
                gen = score_batch(model, xe, ye, pe, ae, mode)  # within-prime grok signal
                line = (f"step {step:5d} | loss {loss.item():.4f} | "
                        f"train-fit {fit:.3f} | within-prime-unseen {gen:.3f}")
                if holdout:
                    cross = {t: exact_match(model, split.val_primes[t], device, 4000, eval_rng, mode)
                             for t in tiers if t != 1}
                    line += " | " + " ".join(f"t{t}@cross-prime {cross[t]:.3f}" for t in cross)
                score = gen
            else:
                seen = {t: exact_match(model, split.train[t], device, 4000, eval_rng, mode) for t in tiers}
                line = (f"step {step:5d} | loss {loss.item():.4f} | "
                        + " ".join(f"t{t}@seen {seen[t]:.3f}" for t in tiers))
                if holdout:
                    unseen = {t: exact_match(model, split.val_primes[t], device, 4000, eval_rng, mode)
                              for t in tiers if t != 1}
                    line += " | " + " ".join(f"t{t}@unseen {unseen[t]:.3f}" for t in unseen)
                    score = sum(unseen.values()) / len(unseen)
                else:
                    score = sum(seen[t] for t in tiers if t != 1) / max(1, len([t for t in tiers if t != 1]))
            line += f" | {time.monotonic() - start:.0f}s"
            print(line)

            if score > best:
                best = score
                torch.save({"state_dict": model.state_dict(),
                            "config": model.config,
                            "arch": args.arch,
                            "tiers": list(tiers)}, out_path)

    print(f"done. best score={best:.3f}. saved -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
