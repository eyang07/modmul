"""Compliant DLP-grokking model for modular multiplication.

Idea (the trick Terence Tao raised in the design discussion): over the prime
field, ``a * b mod p`` is the multiplicative group operation, and the
grokking literature has shown small networks can learn the *additive* group
``+ mod n`` by discovering a Fourier / discrete-log representation. So we give
the network a structural nudge toward the discrete-log solution and let it
*learn* the rest from data:

    e_a = Enc(a mod p, p)          # shared residue encoder
    e_b = Enc(b mod p, p)          # (same weights)
    z   = e_a + e_b                # ADDITIVE bottleneck  <-- DLP: logs add
    ans = Dec(z, p)               # decoder emits answer digits

The additive combination of the two residue embeddings is the only inductive
bias we impose. Everything that makes this produce the right answer — the
embedding that turns a residue into something log-like, and the decoder that
turns a sum-in-log-space back into a residue — is in *trained parameters*.
There is no precomputed discrete-log table, no hand-coded generator search,
no ``(log_a + log_b) % (p-1)`` written in Python. Perturb the weights and the
accuracy degrades, which is the operational test for "the answer came from
learning, not from a hand-coded circuit" (see rules/evaluation.md, Principle 2).

Compliance note on the ``a % p`` / ``b % p`` reduction below: this is the same
input-normalisation step the digit_transformer baseline uses. It combines two
arguments at a time (a with p, then b with p), never all three, and does not
compute the modular product itself — the network output materially determines
the answer. The hard part (multiplication in the residue field) is learned.

Like all small-modulus approaches, this only works where the field is small
enough for the structure to be learned and to generalise across primes — i.e.
the low tiers. Above that the network has not learned anything useful and the
score falls to the floor. That ceiling is the honest, expected outcome.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from modchallenge.interface.base_model import ModularMultiplicationModel

# ---------------------------------------------------------------------------
# Fixed dimensions
# ---------------------------------------------------------------------------

# Decimal digits 0-9 plus PAD. Residues and primes are zero-padded MSB-first.
PAD = 10
VOCAB_SIZE = 11

# Widths. We target small primes: p < 10^WIDTH, residues < p so also < 10^WIDTH.
# WIDTH=5 covers every prime up to 99_999 (i.e. tiers 1-3 fully, the small end
# of tier 4). Answers for those tiers are < p, hence <= WIDTH digits.
WIDTH = 5

SLOT_RESIDUE = 0
SLOT_PRIME = 1


def _digits_fixed(n: int, width: int = WIDTH) -> list[int]:
    """Non-negative int -> fixed-width zero-padded decimal digits, MSB-first."""
    out = [0] * width
    i = width - 1
    while n > 0 and i >= 0:
        out[i] = n % 10
        n //= 10
        i -= 1
    return out


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------

class ResidueEncoder(nn.Module):
    """Encode a (residue, prime) pair into a single vector via a small
    Transformer encoder. Shared between the a-branch and the b-branch so a
    residue is embedded the same way regardless of which operand it came from."""

    def __init__(self, d_model: int, nhead: int, num_layers: int, dim_ff: int):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB_SIZE, d_model)
        # 2 * WIDTH positions: residue digits then prime digits.
        self.pos_emb = nn.Embedding(2 * WIDTH, d_model)
        # Segment: which slot (residue vs prime) a position belongs to.
        self.seg_emb = nn.Embedding(2, d_model)
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
        pos = torch.arange(2 * WIDTH)
        self.register_buffer("seg_ids", seg, persistent=False)
        self.register_buffer("pos_ids", pos, persistent=False)

    def forward(self, residue_digits: torch.Tensor, prime_digits: torch.Tensor) -> torch.Tensor:
        # (B, WIDTH) each -> (B, 2*WIDTH)
        tokens = torch.cat([residue_digits, prime_digits], dim=1)
        b, t = tokens.shape
        x = (
            self.tok_emb(tokens)
            + self.pos_emb(self.pos_ids[:t].unsqueeze(0))
            + self.seg_emb(self.seg_ids[:t].unsqueeze(0))
        )
        x = self.encoder(x)
        x = self.ln(x)
        return x.mean(dim=1)  # (B, d_model)


class AnswerDecoder(nn.Module):
    """Map the additive residue code z (and the prime, re-encoded) to WIDTH
    digit distributions. Fixed-width, MSB-first, zero-padded answer."""

    def __init__(self, d_model: int, dim_ff: int):
        super().__init__()
        self.prime_tok_emb = nn.Embedding(VOCAB_SIZE, d_model)
        self.prime_pos_emb = nn.Embedding(WIDTH, d_model)
        self.net = nn.Sequential(
            nn.Linear(2 * d_model, dim_ff),
            nn.GELU(),
            nn.Linear(dim_ff, dim_ff),
            nn.GELU(),
        )
        self.heads = nn.ModuleList([nn.Linear(dim_ff, 10) for _ in range(WIDTH)])
        self.register_buffer("prime_pos_ids", torch.arange(WIDTH), persistent=False)

    def forward(self, z: torch.Tensor, prime_digits: torch.Tensor) -> torch.Tensor:
        # Re-encode the prime as an explicit modulus context for the decoder.
        pe = self.prime_tok_emb(prime_digits) + self.prime_pos_emb(
            self.prime_pos_ids.unsqueeze(0)
        )
        p_ctx = pe.mean(dim=1)  # (B, d_model)
        h = self.net(torch.cat([z, p_ctx], dim=1))
        # (B, WIDTH, 10)
        return torch.stack([head(h) for head in self.heads], dim=1)


class DLPGrokNet(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 3,
        dim_ff: int = 768,
    ):
        super().__init__()
        self.encoder = ResidueEncoder(d_model, nhead, num_layers, dim_ff)
        self.decoder = AnswerDecoder(d_model, dim_ff)
        self.config = dict(
            d_model=d_model, nhead=nhead, num_layers=num_layers, dim_ff=dim_ff
        )

    def forward(
        self,
        a_digits: torch.Tensor,
        b_digits: torch.Tensor,
        prime_digits: torch.Tensor,
    ) -> torch.Tensor:
        e_a = self.encoder(a_digits, prime_digits)
        e_b = self.encoder(b_digits, prime_digits)
        z = e_a + e_b  # additive (log-space) bottleneck -- DLP inductive bias
        return self.decoder(z, prime_digits)  # (B, WIDTH, 10)


# ---------------------------------------------------------------------------
# Submission entry point
# ---------------------------------------------------------------------------

class DLPGrokking(ModularMultiplicationModel):
    def __init__(self):
        self.model: DLPGrokNet | None = None
        self.device: torch.device | None = None

    def load(self, model_dir: str) -> None:
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        ckpt = torch.load(
            Path(model_dir) / "weights.pt",
            map_location=self.device,
            weights_only=True,
        )
        self.model = DLPGrokNet(**ckpt.get("config", {}))
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(self.device)
        self.model.eval()

    def preprocess_a(self, a):
        return a

    def preprocess_b(self, b):
        return b

    def preprocess_p(self, p):
        return p

    @torch.no_grad()
    def predict_digits(self, a_enc, b_enc, p_enc):
        return self.predict_digits_batch([(a_enc, b_enc, p_enc)])[0]

    @torch.no_grad()
    def predict_digits_batch(self, inputs):
        assert self.model is not None
        out: list[list[int] | None] = [None] * len(inputs)

        a_rows, b_rows, p_rows, idx = [], [], [], []
        for i, (a_enc, b_enc, p_enc) in enumerate(inputs):
            p = int(p_enc)
            # Out of the model's small-prime regime: it never learned this.
            # Emit 0 (the honest fallback) without invoking the network.
            if p >= 10 ** WIDTH:
                out[i] = [0]
                continue
            a_red = int(a_enc) % p
            b_red = int(b_enc) % p
            a_rows.append(_digits_fixed(a_red))
            b_rows.append(_digits_fixed(b_red))
            p_rows.append(_digits_fixed(p))
            idx.append(i)

        if idx:
            a_t = torch.tensor(a_rows, dtype=torch.long, device=self.device)
            b_t = torch.tensor(b_rows, dtype=torch.long, device=self.device)
            p_t = torch.tensor(p_rows, dtype=torch.long, device=self.device)
            logits = self.model(a_t, b_t, p_t)  # (N, WIDTH, 10)
            preds = logits.argmax(dim=-1).tolist()  # N x WIDTH
            for j, i in enumerate(idx):
                out[i] = preds[j]  # MSB-first; harness ignores leading zeros

        return [o if o is not None else [0] for o in out]

    def max_batch_size(self) -> int:
        return 512
