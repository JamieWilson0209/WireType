"""From a cell to the encoder's input, and the encoder's read-outs.

A cell's tokens are its own feature row plus one per top-k partner (the partner's
features, `log1p` synapse count and edge direction). `gather` builds them for a
batch of cells; `embed_all` runs a read-out over every cell of a volume.

The read-outs, all functions of `(tokens, batch)`:
- `context_fn`: `E_c(x, UNK)`, the plain embedding. This is what is frozen,
  transferred and probed.
- `oracle_fn`: the true displacement from the frozen target encoder (a check that
  the read-out is not inert).

Paper terms: `context_fn` gives WireType's embedding; a cell (the seed) is a neuron.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def gather(topk, features_t, seeds, device):
    """The token block for a batch of seeds: seed features, partners, weights, direction, validity.

    No masking. The two branches differ in the class channel and in nothing else;
    hiding connections in one of them would add a second difference, and the
    displacement would stop being the class.
    """
    idx = torch.from_numpy(seeds.astype(np.int64)).to(device)
    partner = topk["partner"][seeds]
    return (features_t[idx],
            features_t[torch.from_numpy(np.maximum(partner, 0).astype(np.int64)).to(device)],
            torch.from_numpy(topk["weight"][seeds]).to(device),
            torch.from_numpy(topk["sign"][seeds].astype(np.float32)).to(device),
            torch.from_numpy(partner >= 0).to(device))


def batch_for(k: int, at_64: int) -> int:
    """Scale a batch size against k, anchored on a size known to fit at k=64.

    Attention is O(k^2), so a batch tuned at k=64 needs 16x the memory at k=256.
    """
    return max(256, at_64 * 64 // max(k, 1))


def embed_all(fn, topk, features_t, n_nodes, device, amp, chunk=None):
    """Apply `fn(tokens, batch)` over every node and concatenate. `fn` returns (B, d)."""
    chunk = chunk or batch_for(topk["partner"].shape[1], 2048)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        return torch.cat([
            fn(gather(topk, features_t, np.arange(i, min(i + chunk, n_nodes)), device),
               np.arange(i, min(i + chunk, n_nodes)))
            for i in range(0, n_nodes, chunk)]).float()


def unk_of(tokens, encoder):
    """A batch-sized tensor holding the encoder's UNK class code."""
    return torch.full((tokens[0].shape[0],), encoder.unk, dtype=torch.long,
                      device=tokens[0].device)


def context_fn(encoder):
    """E_c(x, UNK): the plain embedding, the artefact that is frozen and transferred."""
    def fn(tokens, _batch):
        return encoder(*tokens, unk_of(tokens, encoder))
    return fn


def displacement(target_encoder, tokens, y):
    """LN E_t(x, y) - LN E_t(x, UNK): the class-attributable part of the target.

    The two terms share every input except the class code, so everything the class
    does not touch cancels exactly.
    """
    z = target_encoder(*tokens, y)
    u = target_encoder(*tokens, unk_of(tokens, target_encoder))
    return F.layer_norm(z, z.shape[-1:]) - F.layer_norm(u, u.shape[-1:])


def oracle_fn(target_encoder, y_index):
    """The true displacement, LN E_t(x, y) - LN E_t(x, UNK); zero where unlabelled."""
    def fn(tokens, batch):
        y = y_index[torch.from_numpy(batch.astype(np.int64))].to(tokens[0].device)
        known = y >= 0
        z = target_encoder(*tokens, torch.where(known, y, target_encoder.unk))
        u = target_encoder(*tokens, unk_of(tokens, target_encoder))
        d = F.layer_norm(z, z.shape[-1:]) - F.layer_norm(u, u.shape[-1:])
        return d * known.unsqueeze(-1).to(d.dtype)
    return fn
