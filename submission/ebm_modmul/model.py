"""Submission entry point: learned modular multiplication.

Compliance contract (see rules/evaluation.md):
- ``preprocess_*`` are per-argument identities (each sees only its own argument).
- Inside ``predict_digits_batch`` we reduce each operand modulo p — ``int(a) % p``
  and ``int(b) % p`` — the same two-args-at-a-time normalisation the reference
  baselines use. We never form ``a * b`` or ``(a*b) % p`` in Python/tensors; the
  modular product is produced by the trained network, whose output (a residue in
  ``[0, p)``) materially determines the answer.
- We emit the residue as base-10 digits (``output_base = 10``); the harness decodes.

Out of regime (``p >= 10**WIDTH``, i.e. tiers >= 4) the network's fixed-width
residue encoding cannot represent the operands, so we emit ``[0]`` — an honest
fallback, not a guess. This model targets the low tiers (1-3).

The architecture (encoder + classification/angular head) is loaded from the
checkpoint's ``arch`` field, so the same wrapper serves either trained head.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn

from modchallenge.interface.base_model import ModularMultiplicationModel

# ---------------------------------------------------------------------------
# Fixed dimensions (must match the training code that produced the weights)
# ---------------------------------------------------------------------------

VOCAB_SIZE = 10           # decimal digits 0-9; fixed-width inputs, no PAD token
WIDTH = 5                 # values < 10**5 = 100000 -> covers tiers 1-3
SEG_X, SEG_Y, SEG_P, SEG_ANS = 0, 1, 2, 3


def digits_fixed(n: int, width: int = WIDTH) -> list[int]:
    """Non-negative int -> fixed-width zero-padded decimal digits, MSB-first."""
    out = [0] * width
    i = width - 1
    while n > 0 and i >= 0:
        out[i] = n % 10
        n //= 10
        i -= 1
    return out


def int_to_decimal_digits(n: int) -> list[int]:
    """Non-negative int -> base-10 digit list, MSB-first ([0] for zero)."""
    if n == 0:
        return [0]
    return [int(c) for c in str(n)]


# ---------------------------------------------------------------------------
# Architectures (copied verbatim from training/model.py for state_dict match)
# ---------------------------------------------------------------------------

class JointModMulNetCls(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_layers=6, dim_ff=1024, p_max=256):
        super().__init__()
        self.p_max = p_max
        self.tok_emb = nn.Embedding(VOCAB_SIZE, d_model)
        self.cls_query = nn.Parameter(torch.randn(1, d_model) * 0.02)
        self.seg_emb = nn.Embedding(4, d_model)
        self.pos_emb = nn.Embedding(3 * WIDTH + 1, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=0.0, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, p_max)
        seg = torch.tensor([SEG_X] * WIDTH + [SEG_Y] * WIDTH + [SEG_P] * WIDTH + [SEG_ANS])
        self.register_buffer("seg_ids", seg, persistent=False)
        self.register_buffer("pos_ids", torch.arange(3 * WIDTH + 1), persistent=False)

    def forward(self, x_digits, y_digits, prime_digits):
        b = x_digits.shape[0]
        inp = torch.cat([x_digits, y_digits, prime_digits], dim=1)
        tok = self.tok_emb(inp)
        cls = self.cls_query.unsqueeze(0).expand(b, 1, -1)
        x = torch.cat([tok, cls], dim=1)
        x = x + self.seg_emb(self.seg_ids.unsqueeze(0)) + self.pos_emb(self.pos_ids.unsqueeze(0))
        x = self.encoder(x)
        x = self.ln(x)
        return self.head(x[:, -1, :])  # (B, p_max)


class JointModMulNetAngular(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_layers=6, dim_ff=1024):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB_SIZE, d_model)
        self.cls_query = nn.Parameter(torch.randn(1, d_model) * 0.02)
        self.seg_emb = nn.Embedding(4, d_model)
        self.pos_emb = nn.Embedding(3 * WIDTH + 1, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=0.0, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 2)
        seg = torch.tensor([SEG_X] * WIDTH + [SEG_Y] * WIDTH + [SEG_P] * WIDTH + [SEG_ANS])
        self.register_buffer("seg_ids", seg, persistent=False)
        self.register_buffer("pos_ids", torch.arange(3 * WIDTH + 1), persistent=False)

    def forward(self, x_digits, y_digits, prime_digits):
        b = x_digits.shape[0]
        inp = torch.cat([x_digits, y_digits, prime_digits], dim=1)
        tok = self.tok_emb(inp)
        cls = self.cls_query.unsqueeze(0).expand(b, 1, -1)
        x = torch.cat([tok, cls], dim=1)
        x = x + self.seg_emb(self.seg_ids.unsqueeze(0)) + self.pos_emb(self.pos_ids.unsqueeze(0))
        x = self.encoder(x)
        x = self.ln(x)
        return self.head(x[:, -1, :])  # (B, 2)


_ARCHS = {"cls": JointModMulNetCls, "angular": JointModMulNetAngular}


def _angular_decode(pred: torch.Tensor, p_int: torch.Tensor) -> torch.Tensor:
    theta = torch.atan2(pred[:, 1], pred[:, 0])
    t = torch.round(theta * p_int.float() / (2 * math.pi))
    return (t % p_int.float()).long()


# ---------------------------------------------------------------------------
# Submission entry class
# ---------------------------------------------------------------------------

class EBMModMul(ModularMultiplicationModel):
    def __init__(self):
        self.model = None
        self.device = None
        self.arch = None

    def load(self, model_dir: str) -> None:
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        ckpt = torch.load(Path(model_dir) / "weights.pt",
                          map_location=self.device, weights_only=False)
        self.arch = ckpt.get("arch", "cls")
        self.model = _ARCHS[self.arch](**ckpt["config"]).to(self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

    # Per-argument identity preprocessing (each hook sees only its own argument).
    def preprocess_a(self, a): return a
    def preprocess_b(self, b): return b
    def preprocess_p(self, p): return p

    @torch.no_grad()
    def predict_digits(self, a_enc, b_enc, p_enc):
        return self.predict_digits_batch([(a_enc, b_enc, p_enc)])[0]

    @torch.no_grad()
    def predict_digits_batch(self, inputs):
        out: list[list[int] | None] = [None] * len(inputs)
        x_rows, y_rows, p_rows, p_ints, idx = [], [], [], [], []

        for i, (a_enc, b_enc, p_enc) in enumerate(inputs):
            p = int(p_enc)
            # Out of the model's regime (residues don't fit WIDTH digits): honest 0.
            if p >= 10 ** WIDTH:
                out[i] = [0]
                continue
            a_red = int(a_enc) % p          # per-operand reduction (allowed)
            b_red = int(b_enc) % p
            x_rows.append(digits_fixed(a_red))
            y_rows.append(digits_fixed(b_red))
            p_rows.append(digits_fixed(p))
            p_ints.append(p)
            idx.append(i)

        if idx:
            t = lambda r: torch.tensor(r, dtype=torch.long, device=self.device)
            logits = self.model(t(x_rows), t(y_rows), t(p_rows))
            if self.arch == "angular":
                residues = _angular_decode(logits, t(p_ints)).tolist()
            else:
                residues = logits.argmax(dim=-1).tolist()
            for j, i in enumerate(idx):
                out[i] = int_to_decimal_digits(int(residues[j]))

        return [o if o is not None else [0] for o in out]

    def max_batch_size(self) -> int:
        return 512
