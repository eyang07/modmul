"""Tier-0 multiplication probe: can a digit transformer LEARN to multiply?

This is the go/no-go gate for the whole tier-3 bet (see TIER3_STRATEGY.md). Tier 0
is pure multiplication (unscored). If a small transformer with abacus-style
position embeddings can learn x*y exactly and generalise across values (and ideally
to one extra digit of length), tier 3's multiply-then-reduce is plausible; if it
can't even multiply, tier 3 is out of reach and we bank tier 2.

Design (standard arithmetic-transformer recipe, McLeish et al. 2024):
- Digits emitted LSB-first (reversed) so carries flow left-to-right.
- **Abacus embeddings**: each digit also gets an embedding for its position *within
  its own number* (its place value), reset per operand/result. This is the key trick
  that makes digit arithmetic learnable.
- Decoder-only causal transformer; autoregressive greedy decode of the product.

Sequence:  BOS  x_lsb...  MUL  y_lsb...  EQ  prod_lsb...  EOS
Loss is on the product region + EOS only.

Usage:
    python training/tier0_probe.py --max-digits 5 --steps 20000
"""

from __future__ import annotations

import argparse
import contextlib
import os
import random
import time

import torch
import torch.nn as nn

# Vocab: 0-9 digits, then specials.
PAD, BOS, MUL, EQ, EOS = 10, 11, 12, 13, 14
VOCAB = 15


def digits_lsb(n: int) -> list[int]:
    """Non-negative int -> decimal digits, least-significant-first ([0] for 0)."""
    if n == 0:
        return [0]
    out = []
    while n > 0:
        out.append(n % 10)
        n //= 10
    return out


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Example construction
# ---------------------------------------------------------------------------

def build_example(x: int, y: int, max_len: int):
    """Return (tokens, abacus_idx, is_output) padded to max_len."""
    xr, yr, pr = digits_lsb(x), digits_lsb(y), digits_lsb(x * y)
    toks = [BOS] + xr + [MUL] + yr + [EQ] + pr + [EOS]
    # Abacus index = position within the current number (resets each segment).
    abacus = (
        [0] + list(range(len(xr))) + [0] + list(range(len(yr)))
        + [0] + list(range(len(pr))) + [0]
    )
    # We predict (and supervise) the product digits and the final EOS.
    is_out = (
        [False] + [False] * len(xr) + [False] + [False] * len(yr)
        + [False] + [True] * len(pr) + [True]
    )
    pad = max_len - len(toks)
    if pad < 0:
        raise ValueError("max_len too small")
    toks += [PAD] * pad
    abacus += [0] * pad
    is_out += [False] * pad
    return toks, abacus, is_out


def sample_operand(n_digits: int, rng: random.Random) -> int:
    if n_digits <= 1:
        return rng.randrange(0, 10)
    return rng.randrange(10 ** (n_digits - 1), 10 ** n_digits)


def make_batch(batch: int, min_d: int, max_d: int, max_len: int, rng, device,
               hard_frac: float = 0.0):
    """Sample operands with digit-lengths uniform in [min_d, max_d] (balances
    lengths; plain integer sampling over-represents the longest length).

    hard_frac: fraction of examples that FORCE both operands to max_d digits. The
    uniform scheme makes max_d x max_d only 1/(span^2) of the batch -- the hardest,
    rarest corner -- which is exactly where one-shot multiply hits a length cliff.
    Oversampling it drills that corner without losing coverage of shorter operands.
    """
    T, A, M = [], [], []
    for _ in range(batch):
        if hard_frac > 0.0 and rng.random() < hard_frac:
            dx = dy = max_d
        else:
            dx = rng.randint(min_d, max_d)
            dy = rng.randint(min_d, max_d)
        x, y = sample_operand(dx, rng), sample_operand(dy, rng)
        t, a, m = build_example(x, y, max_len)
        T.append(t); A.append(a); M.append(m)
    tt = lambda v, dt: torch.tensor(v, dtype=dt, device=device)
    return tt(T, torch.long), tt(A, torch.long), tt(M, torch.bool)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class AbacusDecoder(nn.Module):
    def __init__(self, max_len: int, abacus_max: int, d_model=256, nhead=8,
                 num_layers=4, dim_ff=1024):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)          # absolute position
        self.abacus_emb = nn.Embedding(abacus_max, d_model)    # place within number
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=0.0, batch_first=True, activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, VOCAB, bias=False)
        self.max_len = max_len
        self.register_buffer("pos_ids", torch.arange(max_len), persistent=False)

    def forward(self, toks, abacus):
        b, t = toks.shape
        x = self.tok_emb(toks) + self.pos_emb(self.pos_ids[:t]) + self.abacus_emb(abacus)
        mask = torch.triu(torch.full((t, t), float("-inf"), device=toks.device), diagonal=1)
        x = self.transformer(x, mask=mask, is_causal=True)
        return self.head(self.ln(x))


# ---------------------------------------------------------------------------
# Eval: autoregressive greedy decode, exact-match by operand length
# ---------------------------------------------------------------------------

@torch.no_grad()
def exact_match_at_length(model, n_digits, n, max_len, abacus_max, rng, device,
                          pure=False) -> float:
    """pure=True forces BOTH operands to exactly n_digits (the hard corner); else
    operand lengths are uniform in [1, n_digits]."""
    model.eval()
    correct = 0
    for _ in range(n):
        if pure:
            dx = dy = n_digits
        else:
            dx = rng.randint(1, n_digits); dy = rng.randint(1, n_digits)
        x, y = sample_operand(dx, rng), sample_operand(dy, rng)
        target = digits_lsb(x * y)
        xr, yr = digits_lsb(x), digits_lsb(y)
        toks = [BOS] + xr + [MUL] + yr + [EQ]
        abac = [0] + list(range(len(xr))) + [0] + list(range(len(yr))) + [0]
        out = []
        for step in range(2 * n_digits + 2):
            if len(toks) >= max_len:
                break
            tt = torch.tensor([toks], dtype=torch.long, device=device)
            aa = torch.tensor([abac], dtype=torch.long, device=device)
            nxt = int(model(tt, aa)[0, -1].argmax())
            if nxt == EOS:
                break
            toks.append(nxt)
            abac.append(min(len(out), abacus_max - 1))  # product place index
            out.append(nxt)
        if out == target:
            correct += 1
    return correct / n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-digits", type=int, default=5, help="max operand digits in training")
    ap.add_argument("--hard-frac", type=float, default=0.0,
                    help="fraction of batch forced to max_d x max_d (drills the length cliff)")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=2000,
                    help="linear LR warmup steps (stabilizes deep models)")
    ap.add_argument("--grad-clip", type=float, default=1.0,
                    help="max grad norm; <=0 disables")
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--eval-n", type=int, default=400)
    ap.add_argument("--amp", action="store_true",
                    help="bf16 autocast on CUDA (~1.5-2x faster on Ada/Ampere)")
    ap.add_argument("--ckpt", type=str, default="training/checkpoints/tier0.pt")
    ap.add_argument("--resume", action="store_true",
                    help="resume model/opt/sched/step from --ckpt if it exists")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device()
    maxd = args.max_digits
    max_len = 4 * maxd + 8
    abacus_max = 2 * maxd + 4
    rng = random.Random(args.seed)
    eval_rng = random.Random(999)

    model = AbacusDecoder(max_len, abacus_max, d_model=args.d_model,
                          num_layers=args.layers).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"device {device} | max_digits {maxd} | params {n_params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    warmup = max(0, min(args.warmup, args.steps - 1))
    if warmup > 0:
        sched = torch.optim.lr_scheduler.SequentialLR(
            opt,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(
                    opt, start_factor=0.01, end_factor=1.0, total_iters=warmup),
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    opt, T_max=args.steps - warmup, eta_min=args.lr * 0.1),
            ],
            milestones=[warmup])
    else:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=args.steps, eta_min=args.lr * 0.1)
    loss_fn = nn.CrossEntropyLoss()

    cfg = dict(d_model=args.d_model, layers=args.layers, max_len=max_len,
               abacus_max=abacus_max, max_digits=maxd)

    def save_ckpt(path, step, best):
        if not args.ckpt:
            return
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(),
                        sched=sched.state_dict(), step=step, best=best,
                        config=cfg, args=vars(args)), path)

    start_step, best = 0, -1.0
    if args.resume and args.ckpt and os.path.exists(args.ckpt):
        ck = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start_step, best = ck["step"], ck.get("best", -1.0)
        print(f"resumed from {args.ckpt} at step {start_step} (best {best:.3f})")

    use_amp = args.amp and device.type == "cuda"
    amp_ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
               if use_amp else contextlib.nullcontext())
    if use_amp:
        print("bf16 autocast ON")
    start = time.monotonic()

    for step in range(start_step + 1, args.steps + 1):
        model.train()
        toks, abac, mask = make_batch(args.batch, 1, maxd, max_len, rng, device,
                                      hard_frac=args.hard_frac)
        with amp_ctx:
            logits = model(toks, abac)          # (B, T, V)
            # predict token t+1 from t; supervise product region + EOS only
            logits = logits[:, :-1].reshape(-1, VOCAB)
            target = toks[:, 1:].reshape(-1)
            m = mask[:, 1:].reshape(-1)
            loss = loss_fn(logits[m], target[m])
        opt.zero_grad(); loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step(); sched.step()

        if step % args.eval_every == 0:
            # exact-match in-distribution (=maxd) and one digit longer (extrapolation)
            acc = {d: exact_match_at_length(model, d, args.eval_n, max_len, abacus_max, eval_rng, device)
                   for d in (maxd, maxd + 1)}
            pure = exact_match_at_length(model, maxd, args.eval_n, max_len, abacus_max,
                                         eval_rng, device, pure=True)
            print(f"step {step:6d} | loss {loss.item():.4f} | "
                  f"exact@{maxd}d {acc[maxd]:.3f} | pure{maxd}x{maxd} {pure:.3f} | "
                  f"exact@{maxd+1}d(extrap) {acc[maxd+1]:.3f} | "
                  f"{time.monotonic()-start:.0f}s")
            save_ckpt(args.ckpt, step, best)
            if args.ckpt and pure > best:          # track the hard corner, not the average
                best = pure
                save_ckpt(args.ckpt.replace(".pt", "_best.pt"), step, best)

    save_ckpt(args.ckpt, args.steps, best)
    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
