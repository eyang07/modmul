"""Submission entry point: learned modular multiplication.

Compliance contract (see rules/evaluation.md):
- ``preprocess_*`` are per-argument identities (each sees only its own argument).
- Inside ``predict_digits_batch`` we reduce each operand modulo p — ``int(a) % p``
  and ``int(b) % p`` — the same two-args-at-a-time normalisation the reference
  baselines use. We never form ``a * b`` or ``(a*b) % p`` in Python/tensors; the
  modular product is produced by the trained network, whose output (a residue in
  ``[0, p)``) materially determines the answer.
- We emit the residue as base-10 digits (``output_base = 10``); the harness decodes.

Routing by prime size: tiers 1-2 (p < 512) use the classification head; tiers 3-5
use the interleaved modular-multiply scratchpad decoder (same architecture,
separately trained weights) — tier 3 (512 <= p < 65536) and tier 4
(65536 <= p < 2**32) in numeric base 10, tier 5 (2**32 <= p < 2**64) in numeric
base 16 (shorter Horner chain at large prime sizes). p >= 2**64 (tiers 6+) is out
of regime, so we emit ``[0]`` — an honest fallback, not a guess.

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
    Architecture identical to training/modmul_probe.py / tier5_modmul.py for
    state_dict match. ``vocab`` defaults to the base-10 scratchpad vocab (18) used
    by tiers 3-4; the tier-5 base-16 scratchpad passes vocab=24 (16 digits + 8
    specials)."""

    def __init__(self, max_len, abacus_max, d_model=384, nhead=8, num_layers=8,
                 dim_ff=1536, vocab=MM_VOCAB):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.abacus_emb = nn.Embedding(abacus_max, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=0.0, batch_first=True, activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
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
            steps = 0
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
                # The sequence grows one token per step, so the caching allocator
                # holds a distinct buffer for every length (~800 on tier-4 chains)
                # and OOMs mid-decode. Periodically release them.
                steps += 1
                if steps % 32 == 0:
                    if device.type == "mps":
                        torch.mps.empty_cache()
                    elif device.type == "cuda":
                        torch.cuda.empty_cache()
            for j, i in enumerate(sub):
                gj = gen[j]
                if MM_COLON in gj:
                    k = len(gj) - 1 - gj[::-1].index(MM_COLON)
                    ans = [d for d in gj[k + 1:] if d < 10]
                    out[i] = ans if ans else [0]
                else:
                    out[i] = [0]
            # Release the chunk's activations: the caching allocator otherwise
            # accumulates across length-groups/chunks (MPS in particular never
            # frees mid-run) and OOMs on long tier-4 chains.
            del toks, abac, seg, done, gen
            if device.type == "mps":
                torch.mps.empty_cache()
            elif device.type == "cuda":
                torch.cuda.empty_cache()
    return [o if o is not None else [0] for o in out]


# ---------------------------------------------------------------------------
# Tier-5 base-16 modular-multiply scratchpad (autoregressive).
#
# Same AbacusDecoder architecture and 9-field Horner scratchpad as tiers 3-4, but
# trained in numeric BASE 16 (so the per-step partial products / quotient digits
# stay easy while the chain length is bounded). tier-5 primes are 33-64 bit, so the
# chain is ~16 base-16 Horner blocks (~1853 tokens). Vocab: digits 0..15 then
# PAD,BOS,MUL,MOD,EQ,COLON,STEP,EOS = base..base+7 (see tier5_modmul.make_vocab).
# The decoded answer is a BASE-16 residue; we convert it to base-10 digits with
# multiply-add only (no %, //, Barrett/Montgomery/CRT on the product) so it matches
# the global output_base=10. Compliance is unchanged: the only modular reduction in
# shipped code is the per-operand int(a)%p / int(b)%p done before the network runs.
# ---------------------------------------------------------------------------


def _make_vocab_base(base: int) -> dict:
    """Base-B scratchpad vocab, matching training/tier5_modmul.make_vocab."""
    PAD, BOS, MUL, MOD, EQ, COLON, STEP, EOS = (
        base, base + 1, base + 2, base + 3, base + 4, base + 5, base + 6, base + 7)
    return dict(PAD=PAD, BOS=BOS, MUL=MUL, MOD=MOD, EQ=EQ, COLON=COLON, STEP=STEP,
                EOS=EOS, VOCAB=base + 8,
                SPECIALS={PAD, BOS, MUL, MOD, EQ, COLON, STEP, EOS})


def _digits_base_msb(n: int, base: int) -> list[int]:
    if n == 0:
        return [0]
    s = []
    while n > 0:
        s.append(n % base)
        n //= base
    return s[::-1]


def _base_to_int(ds: list[int], base: int) -> int:
    v = 0
    for d in ds:
        v = v * base + d        # multiply-add only; no %/// on the product
    return v


@torch.no_grad()
def _modmul_decode_base(model, cfg, xyp, device, base, chunk=64):
    """Greedy-decode (x*y) mod p in numeric base ``base`` for each (x, y, p) with
    x, y already in [0, p). Returns base-10 digit-lists (MSB-first), or [0] if
    unparseable. Mirrors _modmul_decode but base-parametrized; the final base-``base``
    residue is re-expressed in base 10 via multiply-add (compliant)."""
    V = _make_vocab_base(base)
    PAD, EOS, COLON = V["PAD"], V["EOS"], V["COLON"]
    max_len, abmax = cfg["max_len"], cfg["abacus_max"]
    specials = torch.tensor(sorted(V["SPECIALS"]), device=device)
    out: list[list[int] | None] = [None] * len(xyp)

    groups = defaultdict(list)
    prompts = []
    for i, (x, y, p) in enumerate(xyp):
        xd, yd, pd = (_digits_base_msb(x, base), _digits_base_msb(y, base),
                      _digits_base_msb(p, base))
        toks = [V["BOS"]] + xd + [V["MUL"]] + yd + [V["MOD"]] + pd + [V["EQ"]]
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
                nxt = torch.where(done, torch.full_like(nxt, PAD), nxt)
                is_sp = (nxt.unsqueeze(1) == specials).any(1)
                new_abac = torch.where(is_sp, torch.zeros_like(seg),
                                       torch.clamp(seg, max=abmax - 1))
                seg = torch.where(is_sp, torch.zeros_like(seg), seg + 1)
                nc, dc = nxt.tolist(), done.tolist()
                for j in range(g):
                    if not dc[j] and nc[j] != EOS and nc[j] != PAD:
                        gen[j].append(nc[j])
                toks = torch.cat([toks, nxt.unsqueeze(1)], dim=1)
                abac = torch.cat([abac, new_abac.unsqueeze(1)], dim=1)
                done = done | (nxt == EOS)
            for j, i in enumerate(sub):
                gj = gen[j]
                if COLON in gj:
                    k = len(gj) - 1 - gj[::-1].index(COLON)
                    ans = [d for d in gj[k + 1:] if d < base]
                    out[i] = int_to_decimal_digits(_base_to_int(ans, base)) if ans else [0]
                else:
                    out[i] = [0]
            del toks, abac, seg, done, gen
            # Only MPS needs the cache drop (it never frees mid-run and OOMs on
            # long chains). On CUDA, empty_cache() synchronizes + forces the
            # allocator to re-acquire buffers every chunk -- a ~5x slowdown over
            # the ~1853-step base-16 decode -- and tier-5 peak is <3 GB anyway.
            if device.type == "mps":
                torch.mps.empty_cache()
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
        self.mm4 = None         # tier-4 modmul scratchpad
        self.mm4_cfg = None
        self.mm5 = None         # tier-5 base-16 modmul scratchpad
        self.mm5_cfg = None

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
        # Tier 4: same scratchpad architecture, trained on [2**17, 2**32).
        if "tier4" in ckpt:
            c4 = ckpt["tier4"]["config"]
            self.mm4_cfg = c4
            self.mm4 = AbacusDecoder(
                max_len=c4["max_len"], abacus_max=c4["abacus_max"], d_model=c4["d_model"],
                nhead=c4["nhead"], num_layers=c4["layers"], dim_ff=c4["dim_ff"],
            ).to(self.device)
            self.mm4.load_state_dict(ckpt["tier4"]["state_dict"])
            self.mm4.eval()
        # Tier 5: base-16 scratchpad, trained on primes in [2**33, 2**64).
        if "tier5" in ckpt:
            c5 = ckpt["tier5"]["config"]
            self.mm5_cfg = c5
            self.mm5 = AbacusDecoder(
                max_len=c5["max_len"], abacus_max=c5["abacus_max"], d_model=c5["d_model"],
                nhead=c5["nhead"], num_layers=c5["layers"], dim_ff=c5["dim_ff"],
                vocab=c5["base"] + 8,
            ).to(self.device)
            self.mm5.load_state_dict(ckpt["tier5"]["state_dict"])
            self.mm5.eval()

    # Per-argument identity preprocessing (each hook sees only its own argument).
    def preprocess_a(self, a): return a
    def preprocess_b(self, b): return b
    def preprocess_p(self, p): return p

    @torch.no_grad()
    def predict_digits(self, a_enc, b_enc, p_enc):
        return self.predict_digits_batch([(a_enc, b_enc, p_enc)])[0]

    # Prime routing: tiers 1-2 (p < 512) use the classification head; tier 3
    # (512 <= p < 65536) and tier 4 (65536 <= p < 2**32) use the modmul scratchpad
    # (separate trained weights); p >= 2**32 (tiers 5+) is out of regime.
    # 512 = 2**9 is the tier-3 floor, 65536 = 2**16 the tier-3/4 boundary (TIERS).
    TIER3_LO = 512
    TIER3_HI = 65536
    TIER4_HI = 2 ** 32
    TIER5_HI = 2 ** 64

    @torch.no_grad()
    def predict_digits_batch(self, inputs):
        out: list[list[int] | None] = [None] * len(inputs)
        x_rows, y_rows, p_rows, p_ints, idx = [], [], [], [], []   # tiers 1-2
        mm_items, mm_idx = [], []                                  # tier 3
        mm4_items, mm4_idx = [], []                                # tier 4
        mm5_items, mm5_idx = [], []                                # tier 5

        for i, (a_enc, b_enc, p_enc) in enumerate(inputs):
            p = int(p_enc)
            # Out of regime (residues don't fit the trained range): honest 0.
            if p >= self.TIER5_HI:
                out[i] = [0]
                continue
            a_red = int(a_enc) % p          # per-operand reduction (allowed)
            b_red = int(b_enc) % p
            if p >= self.TIER4_HI:
                if self.mm5 is not None:
                    mm5_items.append((a_red, b_red, p)); mm5_idx.append(i)
                else:
                    out[i] = [0]
            elif p >= self.TIER3_HI:
                if self.mm4 is not None:
                    mm4_items.append((a_red, b_red, p)); mm4_idx.append(i)
                else:
                    out[i] = [0]
            elif p >= self.TIER3_LO and self.mm is not None:
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

        if mm4_items:
            # Tier-4 chains are ~800 tokens; without a KV-cache the per-step
            # forward is O(L^2), so decode in small sub-batches to bound peak
            # memory (a single batch of 100 OOMs on a 20 GB device).
            res = _modmul_decode(self.mm4, self.mm4_cfg, mm4_items, self.device, chunk=16)
            for j, i in enumerate(mm4_idx):
                out[i] = res[j]

        if mm5_items:
            # Tier-5 base-16 chains are ~1853 tokens. The model was trained under
            # bf16 autocast, and decoding in bf16 (not fp32) is what keeps the long
            # attention both fast (~125s/100 vs ~470-660s in fp32) and within memory
            # (<3 GB vs a fp32 OOM at 1853-length attention). Match training precision.
            if self.device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    res = _modmul_decode_base(self.mm5, self.mm5_cfg, mm5_items,
                                              self.device, base=self.mm5_cfg["base"],
                                              chunk=64)
            else:
                res = _modmul_decode_base(self.mm5, self.mm5_cfg, mm5_items,
                                          self.device, base=self.mm5_cfg["base"], chunk=64)
            for j, i in enumerate(mm5_idx):
                out[i] = res[j]

        return [o if o is not None else [0] for o in out]

    def max_batch_size(self) -> int:
        return 512
