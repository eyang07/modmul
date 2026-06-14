"""Feedforward predictor for ``x * y mod p`` with an additive (log-space) bottleneck.

Architecture (the dlp_grokking skeleton, made more expressive and reusable):

    e_x = Enc(x_digits, p_digits)     # shared residue encoder, p-conditioned
    e_y = Enc(y_digits, p_digits)     # same weights
    z   = e_x + e_y                   # ADDITIVE bottleneck  <- logs add (DLP bias)
    logits = Dec(z, p_digits)         # WIDTH x 10 digit logits, MSB-first

The additive combination is the one structural prior: over the prime field,
``x * y mod p`` is the multiplicative group operation, which becomes *addition*
in discrete-log coordinates. Everything that turns a residue into something
log-like (the encoder) and a sum-of-logs back into the product residue (the
decoder) is in trained parameters — randomising the weights collapses accuracy,
which is the operational compliance test (rules/evaluation.md, Principle 2).

This module is deliberately self-contained (no ``modchallenge`` import) so it can
be dropped into the submission package later. An EBM verifier head will be added
on top of ``z`` as a training-time exactness signal.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# Decimal digits 0-9; fixed-width inputs so no PAD token is needed.
VOCAB_SIZE = 10
WIDTH = 5  # keep in sync with data.WIDTH (values < 65536 -> <= 5 decimal digits)

SLOT_RESIDUE = 0
SLOT_PRIME = 1


class ResidueEncoder(nn.Module):
    """Encode a (residue, prime) pair into one vector via a small Transformer.

    Shared between the x-branch and the y-branch so a residue is embedded the
    same way regardless of which operand it came from. Conditioned on the prime
    because the group structure differs for every modulus.
    """

    def __init__(self, d_model: int, nhead: int, num_layers: int, dim_ff: int):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB_SIZE, d_model)
        self.pos_emb = nn.Embedding(2 * WIDTH, d_model)
        self.seg_emb = nn.Embedding(2, d_model)  # residue vs prime slot
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.ln = nn.LayerNorm(d_model)

        seg = torch.tensor([SLOT_RESIDUE] * WIDTH + [SLOT_PRIME] * WIDTH)
        self.register_buffer("seg_ids", seg, persistent=False)
        self.register_buffer("pos_ids", torch.arange(2 * WIDTH), persistent=False)

    def forward(self, residue_digits: torch.Tensor, prime_digits: torch.Tensor) -> torch.Tensor:
        tokens = torch.cat([residue_digits, prime_digits], dim=1)  # (B, 2*WIDTH)
        x = (
            self.tok_emb(tokens)
            + self.pos_emb(self.pos_ids.unsqueeze(0))
            + self.seg_emb(self.seg_ids.unsqueeze(0))
        )
        x = self.encoder(x)
        x = self.ln(x)
        return x.mean(dim=1)  # (B, d_model)


class AnswerDecoder(nn.Module):
    """Map the additive code ``z`` (plus the re-encoded prime) to WIDTH digit
    distributions. Fixed-width, MSB-first, zero-padded answer."""

    def __init__(self, d_model: int, dim_ff: int):
        super().__init__()
        self.prime_tok_emb = nn.Embedding(VOCAB_SIZE, d_model)
        self.prime_pos_emb = nn.Embedding(WIDTH, d_model)
        self.net = nn.Sequential(
            nn.Linear(2 * d_model, dim_ff),
            nn.GELU(),
            nn.Linear(dim_ff, dim_ff),
            nn.GELU(),
            nn.Linear(dim_ff, dim_ff),
            nn.GELU(),
        )
        self.heads = nn.ModuleList([nn.Linear(dim_ff, 10) for _ in range(WIDTH)])
        self.register_buffer("prime_pos_ids", torch.arange(WIDTH), persistent=False)

    def forward(self, z: torch.Tensor, prime_digits: torch.Tensor) -> torch.Tensor:
        pe = self.prime_tok_emb(prime_digits) + self.prime_pos_emb(
            self.prime_pos_ids.unsqueeze(0)
        )
        p_ctx = pe.mean(dim=1)  # (B, d_model)
        h = self.net(torch.cat([z, p_ctx], dim=1))
        return torch.stack([head(h) for head in self.heads], dim=1)  # (B, WIDTH, 10)


class ModMulNet(nn.Module):
    """The predictor: shared encoder + additive bottleneck + decoder.

    ``encode`` exposes the bottleneck ``z`` so a future EBM verifier head can be
    attached to the same representation.
    """

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_ff: int = 768,
    ):
        super().__init__()
        self.encoder = ResidueEncoder(d_model, nhead, num_layers, dim_ff)
        self.decoder = AnswerDecoder(d_model, dim_ff)
        self.config = dict(
            d_model=d_model, nhead=nhead, num_layers=num_layers, dim_ff=dim_ff
        )

    def encode(
        self, x_digits: torch.Tensor, y_digits: torch.Tensor, prime_digits: torch.Tensor
    ) -> torch.Tensor:
        e_x = self.encoder(x_digits, prime_digits)
        e_y = self.encoder(y_digits, prime_digits)
        return e_x + e_y  # additive (log-space) bottleneck

    def forward(
        self, x_digits: torch.Tensor, y_digits: torch.Tensor, prime_digits: torch.Tensor
    ) -> torch.Tensor:
        z = self.encode(x_digits, y_digits, prime_digits)
        return self.decoder(z, prime_digits)  # (B, WIDTH, 10)


# ---------------------------------------------------------------------------
# Joint-attention predictor (no additive bottleneck)
# ---------------------------------------------------------------------------

# Segment ids for the joint sequence: x-digits, y-digits, p-digits, answer slots.
SEG_X, SEG_Y, SEG_P, SEG_ANS = 0, 1, 2, 3


class JointModMulNet(nn.Module):
    """Non-autoregressive Transformer that sees x, y, p *jointly*.

    The input sequence is ``[x(WIDTH), y(WIDTH), p(WIDTH), ans_query(WIDTH)]``.
    Full self-attention lets the answer-query positions attend to all of x, y, p
    (and each other), so the network can form the bilinear ``x_i * y_j`` cross
    terms that pure additive combination forbids. Logits are read in parallel
    from the WIDTH answer-query positions — fast and deterministic (no sampling).

    Strictly more expressive than the additive bottleneck: it can still discover
    a discrete-log circuit if that is what generalises, but is not forced into
    one. This is the architecture for the tier-2 coverage regime.
    """

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_ff: int = 1024,
    ):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB_SIZE, d_model)
        # Learned content for the WIDTH answer-query slots (no digit value yet).
        self.ans_query = nn.Parameter(torch.randn(WIDTH, d_model) * 0.02)
        self.seg_emb = nn.Embedding(4, d_model)
        self.pos_emb = nn.Embedding(4 * WIDTH, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 10)
        self.config = dict(
            d_model=d_model, nhead=nhead, num_layers=num_layers, dim_ff=dim_ff
        )

        seg = torch.tensor(
            [SEG_X] * WIDTH + [SEG_Y] * WIDTH + [SEG_P] * WIDTH + [SEG_ANS] * WIDTH
        )
        self.register_buffer("seg_ids", seg, persistent=False)
        self.register_buffer("pos_ids", torch.arange(4 * WIDTH), persistent=False)

    def forward(
        self, x_digits: torch.Tensor, y_digits: torch.Tensor, prime_digits: torch.Tensor
    ) -> torch.Tensor:
        b = x_digits.shape[0]
        inp = torch.cat([x_digits, y_digits, prime_digits], dim=1)  # (B, 3*WIDTH)
        tok = self.tok_emb(inp)  # (B, 3*WIDTH, d)
        ans = self.ans_query.unsqueeze(0).expand(b, WIDTH, -1)  # (B, WIDTH, d)
        x = torch.cat([tok, ans], dim=1)  # (B, 4*WIDTH, d)
        x = x + self.seg_emb(self.seg_ids.unsqueeze(0)) + self.pos_emb(self.pos_ids.unsqueeze(0))
        x = self.encoder(x)
        x = self.ln(x)
        ans_out = x[:, 3 * WIDTH :, :]  # (B, WIDTH, d) — the answer slots
        return self.head(ans_out)  # (B, WIDTH, 10)


class JointModMulNetCls(nn.Module):
    """Joint-attention encoder with a single residue-classification head.

    Instead of decoding the answer as several correlated decimal digits, predict
    the residue directly as one class in ``[0, p_max)`` — the representation the
    grokking literature uses for small modular arithmetic, which avoids the
    multi-digit coordination the digit decoder struggles with. At submission
    time this pairs with ``output_base = "p"`` (emit the single residue digit).

    Only usable while ``p_max`` is a tractable number of classes (low/mid tiers).
    """

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_ff: int = 1024,
        p_max: int = 256,
    ):
        super().__init__()
        self.p_max = p_max
        self.tok_emb = nn.Embedding(VOCAB_SIZE, d_model)
        self.cls_query = nn.Parameter(torch.randn(1, d_model) * 0.02)
        self.seg_emb = nn.Embedding(4, d_model)  # x, y, p, cls
        self.pos_emb = nn.Embedding(3 * WIDTH + 1, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, p_max)
        self.config = dict(
            d_model=d_model, nhead=nhead, num_layers=num_layers,
            dim_ff=dim_ff, p_max=p_max,
        )

        seg = torch.tensor([SEG_X] * WIDTH + [SEG_Y] * WIDTH + [SEG_P] * WIDTH + [SEG_ANS])
        self.register_buffer("seg_ids", seg, persistent=False)
        self.register_buffer("pos_ids", torch.arange(3 * WIDTH + 1), persistent=False)

    def forward(
        self, x_digits: torch.Tensor, y_digits: torch.Tensor, prime_digits: torch.Tensor
    ) -> torch.Tensor:
        b = x_digits.shape[0]
        inp = torch.cat([x_digits, y_digits, prime_digits], dim=1)  # (B, 3*WIDTH)
        tok = self.tok_emb(inp)
        cls = self.cls_query.unsqueeze(0).expand(b, 1, -1)
        x = torch.cat([tok, cls], dim=1)  # (B, 3*WIDTH+1, d)
        x = x + self.seg_emb(self.seg_ids.unsqueeze(0)) + self.pos_emb(self.pos_ids.unsqueeze(0))
        x = self.encoder(x)
        x = self.ln(x)
        return self.head(x[:, -1, :])  # (B, p_max) — logits over residue classes


class JointModMulNetAngular(nn.Module):
    """Joint-attention encoder with an *angular* output head (Saxena-Charton).

    The answer residue ``t in [0, p)`` is represented as a point on the unit
    circle ``(cos 2*pi*t/p, sin 2*pi*t/p)``. The head regresses a 2-vector
    ``(x', y')`` and training minimises distance to that circle point plus an
    anti-collapse term. This encodes the cyclic structure of Z/pZ directly and,
    unlike a p-way classifier, **scales to large p** (no per-residue class).

    Decode (deterministic, compliant): ``t_hat = round(atan2(y',x') * p / 2pi)
    mod p`` — the angular analogue of argmax.
    """

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_ff: int = 1024,
    ):
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
        self.head = nn.Linear(d_model, 2)  # (x', y') on/near the unit circle
        self.config = dict(d_model=d_model, nhead=nhead, num_layers=num_layers, dim_ff=dim_ff)

        seg = torch.tensor([SEG_X] * WIDTH + [SEG_Y] * WIDTH + [SEG_P] * WIDTH + [SEG_ANS])
        self.register_buffer("seg_ids", seg, persistent=False)
        self.register_buffer("pos_ids", torch.arange(3 * WIDTH + 1), persistent=False)

    def forward(
        self, x_digits: torch.Tensor, y_digits: torch.Tensor, prime_digits: torch.Tensor
    ) -> torch.Tensor:
        b = x_digits.shape[0]
        inp = torch.cat([x_digits, y_digits, prime_digits], dim=1)
        tok = self.tok_emb(inp)
        cls = self.cls_query.unsqueeze(0).expand(b, 1, -1)
        x = torch.cat([tok, cls], dim=1)
        x = x + self.seg_emb(self.seg_ids.unsqueeze(0)) + self.pos_emb(self.pos_ids.unsqueeze(0))
        x = self.encoder(x)
        x = self.ln(x)
        return self.head(x[:, -1, :])  # (B, 2)


# ---------------------------------------------------------------------------
# Per-prime conditioning (a learned embedding per prime)
# ---------------------------------------------------------------------------

# Covers every prime that can appear in tiers 1-3 (p < 2**16). Tier-3 max prime
# is < 65536; tier-4 primes start at 2**17, so nothing falls in [65536, 131072).
PRIME_ENUM_LIMIT = 65536


def _sieve_primes(limit: int) -> list[int]:
    is_p = bytearray([1]) * limit
    is_p[0] = is_p[1] = 0
    for i in range(2, int(limit ** 0.5) + 1):
        if is_p[i]:
            is_p[i * i :: i] = bytearray(len(is_p[i * i :: i]))
    return [i for i in range(2, limit) if is_p[i]]


class JointModMulNetClsPP(nn.Module):
    """Joint-attention classifier with a **learned per-prime embedding**.

    The shared network conditioned only on prime *digits* caps ~0.71 on tier 2
    because it must encode ~44 different multiplication tables through one
    p-digit pathway. Here each prime additionally gets a dedicated learned vector
    (indexed by the prime's rank among all primes < PRIME_ENUM_LIMIT), broadcast
    over the sequence — giving every prime its own "slot" while sharing the
    multiplication circuit. Compliant: the embedding is learned conditioning, not
    an answer lookup (randomising weights still collapses accuracy).

    The prime index is computed *inside* forward from the prime digits, so the
    signature stays ``(x, y, p)`` — a drop-in for the plain cls head.
    """

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
        self.config = dict(d_model=d_model, nhead=nhead, num_layers=num_layers,
                           dim_ff=dim_ff, p_max=p_max)

        # Per-prime embedding + a (non-trained) prime -> index lookup.
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
        # Reconstruct p from its digits and fetch the learned per-prime vector.
        p_int = (prime_digits * self.place_value).sum(dim=1)        # (B,)
        safe = p_int.clamp(0, self.limit - 1)
        p_emb = self.prime_emb(self.idx_lookup[safe])              # (B, d)
        p_emb = p_emb * self.valid_lookup[safe].unsqueeze(-1)      # zero if out of range

        inp = torch.cat([x_digits, y_digits, prime_digits], dim=1)
        tok = self.tok_emb(inp)
        cls = self.cls_query.unsqueeze(0).expand(b, 1, -1)
        x = torch.cat([tok, cls], dim=1)
        x = x + self.seg_emb(self.seg_ids.unsqueeze(0)) + self.pos_emb(self.pos_ids.unsqueeze(0))
        x = x + p_emb.unsqueeze(1)  # broadcast per-prime conditioning to all positions
        x = self.encoder(x)
        x = self.ln(x)
        return self.head(x[:, -1, :])  # (B, p_max)
