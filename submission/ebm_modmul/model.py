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
from collections import defaultdict
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


PRIME_ENUM_LIMIT = 65536


def _sieve_primes(limit: int) -> list[int]:
    is_p = bytearray([1]) * limit
    is_p[0] = is_p[1] = 0
    for i in range(2, int(limit ** 0.5) + 1):
        if is_p[i]:
            is_p[i * i :: i] = bytearray(len(is_p[i * i :: i]))
    return [i for i in range(2, limit) if is_p[i]]


class JointModMulNetClsPP(nn.Module):
    """Joint-attention classifier with a learned per-prime embedding.
    Mirrors training/model.py for state_dict compatibility."""

    def __init__(self, d_model=256, nhead=8, num_layers=6, dim_ff=1024, p_max=256):
        super().__init__()
        self.p_max = p_max
        self.limit = PRIME_ENUM_LIMIT
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
        primes = _sieve_primes(self.limit)
        self.prime_emb = nn.Embedding(len(primes), d_model)
        idx = torch.zeros(self.limit, dtype=torch.long)
        valid = torch.zeros(self.limit, dtype=torch.float)
        for rank, p in enumerate(primes):
            idx[p] = rank
            valid[p] = 1.0
        self.register_buffer("idx_lookup", idx, persistent=False)
        self.register_buffer("valid_lookup", valid, persistent=False)
        self.register_buffer(
            "place_value",
            torch.tensor([10 ** (WIDTH - 1 - i) for i in range(WIDTH)], dtype=torch.long),
            persistent=False,
        )
        seg = torch.tensor([SEG_X] * WIDTH + [SEG_Y] * WIDTH + [SEG_P] * WIDTH + [SEG_ANS])
        self.register_buffer("seg_ids", seg, persistent=False)
        self.register_buffer("pos_ids", torch.arange(3 * WIDTH + 1), persistent=False)

    def forward(self, x_digits, y_digits, prime_digits):
        b = x_digits.shape[0]
        p_int = (prime_digits * self.place_value).sum(dim=1)
        safe = p_int.clamp(0, self.limit - 1)
        p_emb = self.prime_emb(self.idx_lookup[safe]) * self.valid_lookup[safe].unsqueeze(-1)
        inp = torch.cat([x_digits, y_digits, prime_digits], dim=1)
        tok = self.tok_emb(inp)
        cls = self.cls_query.unsqueeze(0).expand(b, 1, -1)
        x = torch.cat([tok, cls], dim=1)
        x = x + self.seg_emb(self.seg_ids.unsqueeze(0)) + self.pos_emb(self.pos_ids.unsqueeze(0))
        x = x + p_emb.unsqueeze(1)
        x = self.encoder(x)
        x = self.ln(x)
        return self.head(x[:, -1, :])


_ARCHS = {
    "cls": JointModMulNetCls,
    "cls_pp": JointModMulNetClsPP,
    "angular": JointModMulNetAngular,
}


def _angular_decode(pred: torch.Tensor, p_int: torch.Tensor) -> torch.Tensor:
    theta = torch.atan2(pred[:, 1], pred[:, 0])
    t = torch.round(theta * p_int.float() / (2 * math.pi))
    return (t % p_int.float()).long()


# ---------------------------------------------------------------------------
# Tier-3 interleaved modular-multiply scratchpad (autoregressive).
#
# Self-contained copy of the trained training/modmul_probe.py decoder + greedy
# decode. The network emits the schoolbook computation digit by digit:
#   BOS x MUL y MOD p EQ  d:q1:r1:pp:t:q2:r2 STEP ... EOS
# folding multiply and reduction into one Horner pass so no intermediate exceeds
# ~6 digits. Compliance: the only modular reduction in shipped code is the
# per-operand int(a)%p / int(b)%p done BEFORE the network runs; the product's
# reduction is produced entirely by trained parameters (greedy argmax over digit
# tokens). There is no %, //, Barrett, Montgomery or CRT applied to a*b anywhere.
# ---------------------------------------------------------------------------

MM_PAD, MM_BOS, MM_MUL, MM_MOD, MM_EQ, MM_COLON, MM_STEP, MM_EOS = 10, 11, 12, 13, 14, 15, 16, 17
MM_VOCAB = 18
MM_SPECIALS = {MM_PAD, MM_BOS, MM_MUL, MM_MOD, MM_EQ, MM_COLON, MM_STEP, MM_EOS}


def _digits_msb(n: int) -> list[int]:
    if n == 0:
        return [0]
    s = []
    while n > 0:
        s.append(n % 10)
        n //= 10
    return s[::-1]


class AbacusDecoder(nn.Module):
    """Decoder-only transformer with abacus (place-within-number) embeddings.
    Architecture identical to training/modmul_probe.py for state_dict match."""

    def __init__(self, max_len, abacus_max, d_model=384, nhead=8, num_layers=8, dim_ff=1536):
        super().__init__()
        self.tok_emb = nn.Embedding(MM_VOCAB, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.abacus_emb = nn.Embedding(abacus_max, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=0.0, batch_first=True, activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, MM_VOCAB, bias=False)
        self.max_len = max_len
        self.register_buffer("pos_ids", torch.arange(max_len), persistent=False)

    def forward(self, toks, abacus):
        b, t = toks.shape
        x = self.tok_emb(toks) + self.pos_emb(self.pos_ids[:t]) + self.abacus_emb(abacus)
        mask = torch.triu(torch.full((t, t), float("-inf"), device=toks.device), diagonal=1)
        x = self.transformer(x, mask=mask, is_causal=True)
        return self.head(self.ln(x))


@torch.no_grad()
def _modmul_decode(model, cfg, xyp, device, chunk=128):
    """Greedy-decode (x*y) mod p for each (x, y, p) with x,y already in [0, p).
    Returns a list of residue digit-lists (MSB-first), or [0] if unparseable.
    Decodes in length-grouped chunks to bound memory."""
    max_len, abmax = cfg["max_len"], cfg["abacus_max"]
    specials = torch.tensor(sorted(MM_SPECIALS), device=device)
    out: list[list[int] | None] = [None] * len(xyp)

    groups = defaultdict(list)
    prompts = []
    for i, (x, y, p) in enumerate(xyp):
        xd, yd, pd = _digits_msb(x), _digits_msb(y), _digits_msb(p)
        toks = [MM_BOS] + xd + [MM_MUL] + yd + [MM_MOD] + pd + [MM_EQ]
        abac = ([0] + list(range(len(xd))) + [0] + list(range(len(yd)))
                + [0] + list(range(len(pd))) + [0])
        groups[len(toks)].append(i)
        prompts.append((toks, abac))

    for L, idxs in groups.items():
        for s in range(0, len(idxs), chunk):
            sub = idxs[s:s + chunk]
            g = len(sub)
            toks = torch.tensor([prompts[i][0] for i in sub], dtype=torch.long, device=device)
            abac = torch.tensor([prompts[i][1] for i in sub], dtype=torch.long, device=device)
            seg = torch.zeros(g, dtype=torch.long, device=device)
            done = torch.zeros(g, dtype=torch.bool, device=device)
            gen = [[] for _ in range(g)]
            while toks.shape[1] < max_len and not bool(done.all()):
                nxt = model(toks, abac)[:, -1].argmax(-1)
                nxt = torch.where(done, torch.full_like(nxt, MM_PAD), nxt)
                is_sp = (nxt.unsqueeze(1) == specials).any(1)
                new_abac = torch.where(is_sp, torch.zeros_like(seg),
                                       torch.clamp(seg, max=abmax - 1))
                seg = torch.where(is_sp, torch.zeros_like(seg), seg + 1)
                nc, dc = nxt.tolist(), done.tolist()
                for j in range(g):
                    if not dc[j] and nc[j] != MM_EOS and nc[j] != MM_PAD:
                        gen[j].append(nc[j])
                toks = torch.cat([toks, nxt.unsqueeze(1)], dim=1)
                abac = torch.cat([abac, new_abac.unsqueeze(1)], dim=1)
                done = done | (nxt == MM_EOS)
            for j, i in enumerate(sub):
                gj = gen[j]
                if MM_COLON in gj:
                    k = len(gj) - 1 - gj[::-1].index(MM_COLON)
                    ans = [d for d in gj[k + 1:] if d < 10]
                    out[i] = ans if ans else [0]
                else:
                    out[i] = [0]
    return [o if o is not None else [0] for o in out]


# ---------------------------------------------------------------------------
# Submission entry class
# ---------------------------------------------------------------------------

class EBMModMul(ModularMultiplicationModel):
    def __init__(self):
        self.model = None
        self.device = None
        self.arch = None
        self.mm = None          # tier-3 modmul scratchpad
        self.mm_cfg = None

    def load(self, model_dir: str) -> None:
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        ckpt = torch.load(Path(model_dir) / "weights.pt",
                          map_location=self.device, weights_only=False)
        # Tiers 1-2: the classification/angular head (banked).
        self.arch = ckpt.get("arch", "cls")
        self.model = _ARCHS[self.arch](**ckpt["config"]).to(self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        # Tier 3: the interleaved modular-multiply scratchpad (optional bundle).
        if "tier3" in ckpt:
            c = ckpt["tier3"]["config"]
            self.mm_cfg = c
            self.mm = AbacusDecoder(
                max_len=c["max_len"], abacus_max=c["abacus_max"], d_model=c["d_model"],
                nhead=c["nhead"], num_layers=c["layers"], dim_ff=c["dim_ff"],
            ).to(self.device)
            self.mm.load_state_dict(ckpt["tier3"]["state_dict"])
            self.mm.eval()

    # Per-argument identity preprocessing (each hook sees only its own argument).
    def preprocess_a(self, a): return a
    def preprocess_b(self, b): return b
    def preprocess_p(self, p): return p

    @torch.no_grad()
    def predict_digits(self, a_enc, b_enc, p_enc):
        return self.predict_digits_batch([(a_enc, b_enc, p_enc)])[0]

    # Prime routing: tiers 1-2 (p < 512) use the classification head; tier 3
    # (512 <= p < 65536) uses the modmul scratchpad; p >= 65536 (tiers 4+) is out
    # of regime. 512 = 2**9 is exactly the tier-3 floor (see config TIERS).
    TIER3_LO = 512
    TIER3_HI = 65536

    @torch.no_grad()
    def predict_digits_batch(self, inputs):
        out: list[list[int] | None] = [None] * len(inputs)
        x_rows, y_rows, p_rows, p_ints, idx = [], [], [], [], []   # tiers 1-2
        mm_items, mm_idx = [], []                                  # tier 3

        for i, (a_enc, b_enc, p_enc) in enumerate(inputs):
            p = int(p_enc)
            # Out of regime (residues don't fit the fixed-width / trained range): honest 0.
            if p >= self.TIER3_HI:
                out[i] = [0]
                continue
            a_red = int(a_enc) % p          # per-operand reduction (allowed)
            b_red = int(b_enc) % p
            if p >= self.TIER3_LO and self.mm is not None:
                mm_items.append((a_red, b_red, p)); mm_idx.append(i)
            else:
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

        if mm_items:
            res = _modmul_decode(self.mm, self.mm_cfg, mm_items, self.device)
            for j, i in enumerate(mm_idx):
                out[i] = res[j]

        return [o if o is not None else [0] for o in out]

    def max_batch_size(self) -> int:
        return 512
