"""The cell encoder: a set transformer over a cell's own token and its top-k partners'.

Each cell is a set of tokens: its own feature row, and one per strongest partner
holding the partner's feature row, `log1p(synapses)` and the edge direction. A
learned query pools the set into one embedding. No token carries partner
identity, so the encoder runs on any volume with the same feature columns.

The seed token also carries a class code: one fixed vector per class, plus `UNK`.
Training (`scripts/train.py`) encodes each cell twice with a frozen random copy of
this network, once with the class and once with `UNK`, and the difference is the
displacement the trainable encoder's predictor learns to output. The seed's own
features are in both branches, so the two differ in the class code alone.

**The class codes are fixed, not learned.** A trainable class embedding has a
degenerate optimum: send every class to one vector and the two branches become
identical. A frozen code cannot collapse that way.

Paper terms: the trained CellEncoder is WireType; the query token is the summary
token; a cell (the seed) is a neuron.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class _Block(nn.Module):
    """Pre-LN transformer block. `pad` is True where a slot is padding."""

    def __init__(self, d: int, heads: int, ff_mult: int = 4, dropout: float = 0.0) -> None:
        """Attention and a feed-forward network, each preceded by its own layer norm."""
        super().__init__()
        self.norm_attn = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.norm_ff = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, d * ff_mult), nn.GELU(), nn.Linear(d * ff_mult, d))

    def forward(self, x: Tensor, pad: Tensor) -> Tensor:
        """Self-attention then feed-forward, each added back as a residual; padded slots
        are ignored as keys.
        """
        h = self.norm_attn(x)
        a, _ = self.attn(h, h, h, key_padding_mask=pad, need_weights=False)
        x = x + a
        return x + self.ff(self.norm_ff(x))


class CellEncoder(nn.Module):
    """A cell in, one embedding out. Both branches are this class.

    Tokens are the seed followed by its k strongest connections. A connection
    token is `[partner features, log1p(synapses), direction, 0]`; the seed token
    is `[seed features, 0, 0, 1]` -- the trailing flag is what lets attention
    tell the cell from its partners, since the feature block alone does not.

    The seed token then carries the class code. Call `forward` with `label =
    n_classes` for the label-free branch; `UNK` is a code like any other rather
    than an absence, so the two branches differ by a displacement between two
    symbols and not by one symbol against zero.

    Readout is a learned query token, not a mean: averaging would reduce the set
    to its marginal moments, which an earlier pooled encoder showed carry little.
    """

    def __init__(self, n_features: int, n_classes: int, d: int = 512, heads: int = 8,
                 layers: int = 4, dropout: float = 0.0, ff_mult: int = 4) -> None:
        """The token map, the fixed class codes, the learned query (summary) token, the
        attention blocks and a final layer norm.
        """
        super().__init__()
        self.d = d
        self.n_classes = n_classes
        self.unk = n_classes
        self.token = nn.Linear(n_features + 3, d)
        # A buffer, not a Parameter: it is carried by state_dict so the two
        # branches share one code book, and it can never receive a gradient.
        self.register_buffer("label_code", F.normalize(torch.randn(n_classes + 1, d), dim=-1))
        self.query = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.normal_(self.query, std=0.02)
        # ff_mult drives memory, not parameter count: each layer holds
        # batch x tokens x (ff_mult * d) activations for the backward pass, which
        # at 1024 x 66 x 2048 is ~554 MB per layer per copy, against a 9.5 GiB
        # MIG slice (a 1g.10gb profile).
        self.blocks = nn.ModuleList(
            [_Block(d, heads, ff_mult=ff_mult, dropout=dropout) for _ in range(layers)])
        self.norm = nn.LayerNorm(d)

    def _tokens(self, seed_features: Tensor, features: Tensor,
                weight: Tensor, sign: Tensor) -> Tensor:
        """Project the rows to tokens: the seed row [features, 0, 0, 1] first, then one
        row per partner [features, log synapses, direction, 0].
        """
        b, k, _ = features.shape
        zeros = torch.zeros(b, 1, device=features.device, dtype=features.dtype)
        ones = torch.ones(b, k, 1, device=features.device, dtype=features.dtype)
        seed = torch.cat([seed_features, zeros, zeros, ones[:, :1, 0]], dim=-1).unsqueeze(1)
        nbr = torch.cat([features, weight.unsqueeze(-1), sign.unsqueeze(-1),
                         torch.zeros_like(ones)], dim=-1)
        return self.token(torch.cat([seed, nbr], dim=1))

    @torch.no_grad()
    def calibrate_label_code(self, seed_features: Tensor, features: Tensor,
                             weight: Tensor, sign: Tensor) -> float:
        """Scale the class codes to the norm of a seed token, and return it.

        Without this the code's magnitude is an unnamed constant that decides how
        much the label moves the target -- and the blocks are pre-LN, so it is the
        *ratio* to the token that survives, not the absolute size. Too small and
        the label channel is inert and the run measures nothing; too large and the
        target is the label alone with the neighbourhood normalised away. Matching
        the seed token's own norm is the one choice that is not arbitrary, and
        `label_sensitivity` in the run report says whether it worked.
        """
        scale = self._tokens(seed_features, features, weight, sign)[:, 0].norm(dim=-1).mean()
        self.label_code.copy_(F.normalize(self.label_code, dim=-1) * scale)
        return float(scale)

    def forward(self, seed_features: Tensor, features: Tensor, weight: Tensor,
                sign: Tensor, valid: Tensor, label: Tensor) -> Tensor:
        """seed_features (B,F) · features (B,k,F) · weight/sign (B,k) ·
        valid (B,k) bool · label (B,) long, `self.unk` for the label-free branch."""
        x = self._tokens(seed_features, features, weight, sign)
        x = torch.cat([x[:, :1] + self.label_code[label].unsqueeze(1).to(x.dtype), x[:, 1:]], dim=1)
        x = torch.cat([self.query.expand(x.shape[0], -1, -1).to(x.dtype), x], dim=1)
        # Query and seed are always attendable, which is also what keeps a cell
        # with no connections finite: 623 cells in FAFB have none, and a row whose
        # keys are all masked makes softmax over all -inf return NaN.
        pad = torch.cat([torch.zeros_like(valid[:, :2]), ~valid], dim=1)
        for block in self.blocks:
            x = block(x, pad)
        return self.norm(x[:, 0])


class Predictor(nn.Module):
    """Label-free embedding in, label-carrying embedding out.

    What it cannot recover from the neighbourhood, the displacement read-out
    cannot carry. It is deliberately an MLP and not another transformer
    -- it operates on one pooled vector, and giving it more capacity than the
    encoder would let it memorise the class marginal instead of reading structure.
    """

    def __init__(self, d: int, hidden_mult: int = 2, layers: int = 2) -> None:
        """An MLP of `layers` hidden layers of width d x hidden_mult, each with layer
        norm and GELU.
        """
        super().__init__()
        h = d * hidden_mult
        net: list[nn.Module] = []
        for i in range(layers):
            net += [nn.Linear(d if i == 0 else h, h), nn.LayerNorm(h), nn.GELU()]
        net += [nn.Linear(h if layers else d, d)]
        self.net = nn.Sequential(*net)

    def forward(self, z: Tensor) -> Tensor:
        """Map the label-free embedding E_c(x, UNK) to a predicted displacement."""
        return self.net(z)
