#!/usr/bin/env python3
"""External baselines: FAFB-fitted embeddings of both volumes, for the comparison table.

Each method here produces a pair of `.npy` embeddings, FAFB and MCNS, fitted on
FAFB only, and nothing else. Scoring is `scripts/score_embeddings.py`'s job, so every row
of the comparison table (the paper's Figure 2) comes out of the same probe code.

| tier | method | fitted on |
|---|---|---|
| A | `degree`: the 8 degree columns | nothing |
| A | `raw`: the 31 input columns the encoder sees | nothing |
| B | `sage`, `gin`, `gat`: supervised GNNs | FAFB transmitter labels |
| C | `dgi`, `bgrl`, `graphmae`: self-supervised GNNs | FAFB wiring only |
| D | `composition`: each cell's partner super-class profile | FAFB labels, **plus MCNS super-class** |

**Same inputs for every method.** The GNNs take the 31 columns of the `refined`
set, standardised with FAFB's training-split statistics and applied unchanged to
MCNS (`source`, as for the encoder). Only the method differs.

**Directed.** A connectome's input and output partners are nearly disjoint (Jaccard
0.141), so every GNN layer aggregates the two directions with separate
weights, as in Dir-GNN (Rossi et al. 2024), with log1p synapse counts as edge
weights. `sage` averages, `gin` sums, `gat` attends, with log synapse count as a
term in the attention logit.

**Whole-graph training.** FAFB has 15M edges and MCNS 25M. Sparse products keep
memory at nodes x width, so a full-graph step fits a MIG slice and needs neither
neighbour sampling nor `pyg_lib`/`torch_sparse`, which the training env lacks.
`gat` is the exception on memory: its per-edge attention is checkpointed.

**Frozen at transfer.** Batch-norm statistics are FAFB's; nothing is refitted on
MCNS. Supervised methods are selected on FAFB validation macro-F1, and their own
classifier head is also scored on MCNS (the `head` entry of `<name>_build.json`), since a
probe on an embedding is not the only fair reading of a supervised model.

**`composition` is not zero-shot.** It reads MCNS partners' super-class
annotations, so it belongs to the "partial labels" setting. It is excluded from the paper.

**Reverse direction.** `--source mcns --target fafb --scope brain_neurons` fits on MCNS
brain neurons with synapses and applies to FAFB; the output names end
`_mcns_to_fafb` (used for the raw-features floor, job `raw_s0_mcns_to_fafb`).

    PYTHONPATH=src python scripts/baselines.py --method sage --supervise-on nt_best
    PYTHONPATH=src python scripts/score_embeddings.py --name sage_ntbest_s0 \\
        --source-emb experiments/baselines/sage_ntbest_s0_embeddings_fafb.npy \\
        --target-emb experiments/baselines/sage_ntbest_s0_embeddings_mcns.npy
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.checkpoint import checkpoint
from torch_geometric.utils import softmax

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wiretype.data.features import (DEFAULT_BLOCKS, degree_features, effective_rank, feature_summary,
                                    standardise_with, wiring_features)
from wiretype.data.scope import EXCLUDED, SCOPES, restrict_split
from wiretype.eval.transfer import load_volume
from wiretype.log import log
from wiretype.eval.probes import _score, label_strings

FEATURE_ONLY = ("degree", "raw", "composition")
SUPERVISED = ("sage", "gin", "gat")
SELF_SUPERVISED = ("dgi", "bgrl", "graphmae")
METHODS = FEATURE_ONLY + SUPERVISED + SELF_SUPERVISED

# Per-method defaults, from each paper's large-graph settings where it gives one.
EPOCHS = {"sage": 300, "gin": 300, "gat": 300, "dgi": 300, "bgrl": 500, "graphmae": 500}
LR = {"sage": 5e-3, "gin": 5e-3, "gat": 5e-3, "dgi": 1e-3, "bgrl": 5e-4, "graphmae": 1e-3}
SHORT = {"nt_best": "ntbest", "nt_train": "nttrain", "nt_known": "ntknown"}


# --------------------------------------------------------------------------- graph

class Graph:
    """A weighted directed graph as sparse operators, both directions.

    `in` rows aggregate over a cell's presynaptic partners, `out` rows over its
    postsynaptic ones. Operators are built on first use and cached; `drop_edges`
    returns a new graph, for BGRL's views.
    """

    def __init__(self, src: torch.Tensor, dst: torch.Tensor, w: torch.Tensor, n: int):
        self.src, self.dst, self.w, self.n = src, dst, w, n
        self._cache: dict = {}

    @classmethod
    def from_edges(cls, edges: pd.DataFrame, n: int, device) -> "Graph":
        """Build from an edge table on `device`, with log(1 + synapses) as each edge's
        weight.
        """
        as_long = lambda a: torch.from_numpy(a.astype(np.int64)).to(device)
        return cls(as_long(edges.pre.to_numpy()), as_long(edges.post.to_numpy()),
                   torch.from_numpy(np.log1p(edges.w.to_numpy(np.float32))).to(device), n)

    def edges(self, direction: str):
        """(row, col, w), sorted by (row, col) so sparse tensors need no coalescing."""
        key = ("edges", direction)
        if key not in self._cache:
            row, col = (self.dst, self.src) if direction == "in" else (self.src, self.dst)
            order = torch.argsort(row * self.n + col)
            self._cache[key] = (row[order], col[order], self.w[order])
        return self._cache[key]

    def operator(self, direction: str, norm: str) -> torch.Tensor:
        """Sparse (n, n): `mean` divides each row by its total weight, `sum` does not."""
        key = ("op", direction, norm)
        if key not in self._cache:
            row, col, w = self.edges(direction)
            if norm == "mean":
                total = torch.zeros(self.n, device=w.device).index_add_(0, row, w)
                w = w / total[row].clamp_min(1e-12)
            self._cache[key] = torch.sparse_coo_tensor(
                torch.stack([row, col]), w, (self.n, self.n), is_coalesced=True)
        return self._cache[key]

    def drop_edges(self, p: float) -> "Graph":
        """A copy with each edge dropped independently with probability `p` (BGRL's
        augmentation).
        """
        keep = torch.rand(len(self.w), device=self.w.device) >= p
        return Graph(self.src[keep], self.dst[keep], self.w[keep], self.n)


def spmm(a: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Sparse matrix times dense matrix: one round of aggregation over the operator `a`."""
    return torch.sparse.mm(a, x)


class EdgeWeightedSum(torch.autograd.Function):
    """out[i] = sum over edges e with row[e] = i of values[e] * x[col[e]].

    The same product as `spmm` with a sparse matrix of `values`, but with a
    backward pass that stays sparse. `torch.sparse.mm` returns the gradient for a
    sparse matrix's values as a dense n x n matrix (72 GiB for FAFB), which is
    what killed the first GAT run. Here the values' gradient is one dot product per
    edge, computed in chunks, and x's gradient is the transposed sparse product.
    `row, col` must be sorted by (row, col), as `Graph.edges` returns them.
    """

    CHUNK = 1 << 20

    @staticmethod
    def forward(ctx, values, x, row, col, n):
        ctx.save_for_backward(values, x, row, col)
        ctx.n = n
        a = torch.sparse_coo_tensor(torch.stack([row, col]), values, (n, n), is_coalesced=True)
        return torch.sparse.mm(a, x)

    @staticmethod
    def backward(ctx, grad):
        values, x, row, col = ctx.saved_tensors
        grad_values = grad_x = None
        if ctx.needs_input_grad[0]:
            grad_values = torch.empty_like(values)
            for start in range(0, len(values), EdgeWeightedSum.CHUNK):
                stop = start + EdgeWeightedSum.CHUNK
                grad_values[start:stop] = (grad[row[start:stop]] * x[col[start:stop]]).sum(-1)
        if ctx.needs_input_grad[1]:
            at = torch.sparse_coo_tensor(torch.stack([col, row]), values, (ctx.n, ctx.n)).coalesce()
            grad_x = torch.sparse.mm(at, grad)
        return grad_values, grad_x, None, None, None


# --------------------------------------------------------------------------- layers

class DirSAGE(nn.Module):
    """GraphSAGE-mean (Hamilton et al. 2017), one weight per direction."""

    def __init__(self, d_in, d_out, **_):
        super().__init__()
        self.self_, self.in_, self.out_ = (nn.Linear(d_in, d_out), nn.Linear(d_in, d_out, bias=False),
                                           nn.Linear(d_in, d_out, bias=False))

    def forward(self, h, g):
        return (self.self_(h) + self.in_(spmm(g.operator("in", "mean"), h))
                + self.out_(spmm(g.operator("out", "mean"), h)))


class DirGIN(nn.Module):
    """GIN (Xu et al. 2019): weighted sum aggregation into an MLP, per direction."""

    def __init__(self, d_in, d_out, **_):
        super().__init__()
        self.eps = nn.Parameter(torch.zeros(2))
        self.mlp = nn.Sequential(nn.Linear(2 * d_in, d_out), nn.BatchNorm1d(d_out), nn.ReLU(),
                                 nn.Linear(d_out, d_out))

    def forward(self, h, g):
        return self.mlp(torch.cat([(1 + self.eps[0]) * h + spmm(g.operator("in", "sum"), h),
                                   (1 + self.eps[1]) * h + spmm(g.operator("out", "sum"), h)], dim=1))


class DirGAT(nn.Module):
    """GAT (Veličković et al. 2018), per direction, with log synapse count in the logit.

    Attention is one scalar per edge and head, and aggregation is a sparse product
    per head, so the edge-sized tensors are E x heads, never E x width.
    """

    def __init__(self, d_in, d_out, heads=4, attn_dropout=0.1, **_):
        super().__init__()
        if d_out % heads:
            raise ValueError(f"width {d_out} is not divisible by {heads} heads")
        self.heads, self.c, self.attn_dropout = heads, d_out // heads, attn_dropout
        self.self_ = nn.Linear(d_in, d_out)
        self.lin = nn.ModuleDict({d: nn.Linear(d_in, d_out, bias=False) for d in ("in", "out")})
        self.att = nn.ParameterDict({f"{d}_{r}": nn.Parameter(torch.randn(heads, self.c) * self.c ** -0.5)
                                     for d in ("in", "out") for r in ("src", "dst")})
        self.att_w = nn.ParameterDict({d: nn.Parameter(torch.zeros(heads)) for d in ("in", "out")})

    def attend(self, h, g, direction):
        """Attention-weighted aggregation over one direction's partners, per head,
        concatenated across heads.
        """
        n = h.shape[0]
        x = self.lin[direction](h).view(n, self.heads, self.c)
        row, col, w = g.edges(direction)
        s_src = (x * self.att[f"{direction}_src"]).sum(-1)
        s_dst = (x * self.att[f"{direction}_dst"]).sum(-1)
        e = F.leaky_relu(s_src[col] + s_dst[row] + w[:, None] * self.att_w[direction], 0.2)
        alpha = F.dropout(softmax(e, row, num_nodes=n), self.attn_dropout, self.training)
        return torch.stack([EdgeWeightedSum.apply(alpha[:, k].contiguous(), x[:, k].contiguous(), row, col, n)
                            for k in range(self.heads)], dim=1).reshape(n, -1)

    def forward(self, h, g):
        return self.self_(h) + self.attend(h, g, "in") + self.attend(h, g, "out")


LAYERS = {"sage": DirSAGE, "gin": DirGIN, "gat": DirGAT}


class GNN(nn.Module):
    """`layers` message-passing layers, each followed by batch norm, ReLU and dropout.

    The last layer's output is the embedding. Batch norm keeps FAFB's running
    statistics at transfer, so MCNS is not renormalised by its own.
    """

    def __init__(self, kind, d_in, width, layers=2, dropout=0.2, heads=4):
        super().__init__()
        self.kind, self.dropout = kind, dropout
        dims = [d_in] + [width] * layers
        self.convs = nn.ModuleList(LAYERS[kind](a, b, heads=heads) for a, b in zip(dims[:-1], dims[1:]))
        self.norms = nn.ModuleList(nn.BatchNorm1d(width) for _ in range(layers))

    def forward(self, x, g):
        h = x
        for conv, norm in zip(self.convs, self.norms):
            # GAT's edge-sized intermediates are recomputed in the backward pass
            # rather than held: 15M edges x heads x several tensors per layer
            # would not fit a 9.5 GiB MIG slice otherwise.
            h = (checkpoint(conv, h, g, use_reentrant=False) if self.kind == "gat" and self.training
                 else conv(h, g))
            h = F.dropout(F.relu(norm(h)), self.dropout, self.training)
        return h


# --------------------------------------------------------------------------- training

def macro_f1(logits, y, idx, n_classes):
    """Macro-F1 of the classifier head on the neurons `idx` (used to pick the best epoch
    on FAFB validation).
    """
    pred = logits[idx].argmax(1).cpu().numpy()
    return float(f1_score(y[idx].cpu().numpy(), pred, labels=range(n_classes), average="macro",
                          zero_division=0))


def train_supervised(model, x, g, nodes, split, label, args, device):
    """Class-balanced cross-entropy on FAFB's training split; keep the best validation epoch."""
    values = label_strings(nodes[label])
    classes = np.array(sorted(set(values[(split == "train").to_numpy() & values.notna().to_numpy()])))
    code = {c: i for i, c in enumerate(classes)}
    y = torch.tensor([code.get(v, -1) if pd.notna(v) else -1 for v in values], device=device)
    train = torch.from_numpy(np.flatnonzero((split == "train").to_numpy())).to(device)
    val = torch.from_numpy(np.flatnonzero((split == "val").to_numpy())).to(device)
    train, val = train[y[train] >= 0], val[y[val] >= 0]
    counts = torch.bincount(y[train], minlength=len(classes)).float()
    weight = counts.sum() / (len(classes) * counts.clamp_min(1))

    head = nn.Linear(args.width, len(classes)).to(device)
    params = list(model.parameters()) + list(head.parameters())
    opt = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    best, history = (-1.0, 0, None), []
    for epoch in range(1, args.epochs + 1):
        model.train(); head.train()
        opt.zero_grad()
        loss = F.cross_entropy(head(model(x, g))[train], y[train], weight=weight)
        loss.backward()
        opt.step()
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            model.eval(); head.eval()
            with torch.no_grad():
                logits = head(model(x, g))
            f1 = macro_f1(logits, y, val, len(classes))
            history.append({"epoch": epoch, "loss": float(loss), "val_macro_f1": f1})
            log(f"  epoch {epoch:>4}  loss {float(loss):.4f}  val macro-F1 {f1:.4f}")
            if f1 > best[0]:
                best = (f1, epoch, (copy.deepcopy(model.state_dict()), copy.deepcopy(head.state_dict())))
    model.load_state_dict(best[2][0]); head.load_state_dict(best[2][1])
    log(f"  best val macro-F1 {best[0]:.4f} at epoch {best[1]}")
    return head, classes, {"best_val_macro_f1": best[0], "best_epoch": best[1], "history": history}


def train_dgi(model, x, g, args, device):
    """Deep Graph Infomax (Veličković et al. 2019): real cells against a row-shuffled graph."""
    # The bilinear discriminator h_i^T W s + b, written out: `nn.Bilinear`'s backward
    # materialises cells x width x width (136 GiB for FAFB). The summary s is one
    # vector for every cell, so the score is h_i . (W s) + b.
    bound = args.width ** -0.5
    weight = nn.Parameter(torch.empty(args.width, args.width, device=device).uniform_(-bound, bound))
    bias = nn.Parameter(torch.empty(1, device=device).uniform_(-bound, bound))
    opt = torch.optim.Adam(list(model.parameters()) + [weight, bias],
                           lr=args.lr, weight_decay=args.weight_decay)
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        opt.zero_grad()
        h = model(x, g)
        corrupt = model(x[torch.randperm(len(x), device=device)], g)
        ws = weight @ torch.sigmoid(h.mean(0))
        pos, neg = h @ ws + bias, corrupt @ ws + bias
        loss = (F.binary_cross_entropy_with_logits(pos, torch.ones_like(pos))
                + F.binary_cross_entropy_with_logits(neg, torch.zeros_like(neg)))
        loss.backward()
        opt.step()
        history.append(_log_ssl(epoch, loss, args))
    return {"history": [h for h in history if h]}


def train_graphmae(model, x, g, args, device):
    """GraphMAE (Hou et al. 2022): reconstruct masked cells' features, scaled cosine error."""
    mask_token = nn.Parameter(torch.zeros(1, x.shape[1], device=device))
    to_decoder = nn.Linear(args.width, args.width, bias=False).to(device)
    decoder = DirSAGE(args.width, x.shape[1]).to(device)
    opt = torch.optim.Adam(list(model.parameters()) + [mask_token] + list(to_decoder.parameters())
                           + list(decoder.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    n_mask = int(args.mask_rate * len(x))
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        opt.zero_grad()
        masked = torch.randperm(len(x), device=device)[:n_mask]
        keep = torch.ones(len(x), 1, device=device)
        keep[masked] = 0.0
        z = to_decoder(model(x * keep + mask_token * (1 - keep), g)) * keep   # re-mask before decoding
        cos = F.cosine_similarity(decoder(z, g)[masked], x[masked], dim=1)
        loss = ((1 - cos) ** args.sce_gamma).mean()
        loss.backward()
        opt.step()
        history.append(_log_ssl(epoch, loss, args))
    return {"history": [h for h in history if h]}


def train_bgrl(model, x, g, args, device):
    """BGRL (Thakoor et al. 2022): predict an EMA target's view of a second augmentation."""
    target = copy.deepcopy(model)
    for p in target.parameters():
        p.requires_grad_(False)
    predictor = nn.Sequential(nn.Linear(args.width, 512), nn.BatchNorm1d(512), nn.PReLU(),
                              nn.Linear(512, args.width)).to(device)
    opt = torch.optim.AdamW(list(model.parameters()) + list(predictor.parameters()),
                            lr=args.lr, weight_decay=args.weight_decay)

    def view(p_feature, p_edge):
        keep = (torch.rand(1, x.shape[1], device=device) >= p_feature).float()
        return x * keep, g.drop_edges(p_edge)

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train(); target.train()
        opt.zero_grad()
        (x1, g1), (x2, g2) = view(0.2, 0.2), view(0.1, 0.3)
        p1, p2 = predictor(model(x1, g1)), predictor(model(x2, g2))
        with torch.no_grad():
            t1, t2 = target(x1, g1), target(x2, g2)
        loss = (2 - F.cosine_similarity(p1, t2, dim=1).mean() - F.cosine_similarity(p2, t1, dim=1).mean())
        loss.backward()
        opt.step()
        tau = 1 - (1 - 0.99) * (math.cos(math.pi * epoch / args.epochs) + 1) / 2
        with torch.no_grad():
            for pt, po in zip(target.parameters(), model.parameters()):
                pt.mul_(tau).add_(po.detach(), alpha=1 - tau)
        del g1, g2
        history.append(_log_ssl(epoch, loss, args))
    return {"history": [h for h in history if h]}


def _log_ssl(epoch, loss, args):
    """Log a self-supervised epoch's loss every `eval_every` epochs; returns the logged
    row or None.
    """
    if epoch % args.eval_every == 0 or epoch == args.epochs:
        log(f"  epoch {epoch:>4}  loss {float(loss):.4f}")
        return {"epoch": epoch, "loss": float(loss)}
    return None


# --------------------------------------------------------------------------- inputs

def volume_inputs(processed, volume, blocks, columns, split_column, stats=None, scope="all"):
    """Nodes, split, edges and the standardised input columns for one volume.

    `scope` (source only; wiretype.data.scope) marks out-of-scope cells `excluded`,
    so the scaler is fitted on in-scope training cells, as scripts/train.py does."""
    nodes, split = load_volume(processed, volume, split_column)
    split = restrict_split(nodes, split, scope)
    edges = pd.read_parquet(processed / f"{volume}_edges.parquet")
    raw, names = wiring_features(processed, volume, edges, len(nodes), blocks=blocks, columns=columns)
    x, stats = standardise_with(raw, (split == "train").to_numpy(), stats)
    return nodes, split, edges, x, names, stats


def composition(nodes, edges, vocab, label="super_class"):
    """Share of each cell's input and output synapses by partner class, plus `unknown`."""
    index = {c: i for i, c in enumerate(vocab)}
    k = len(vocab) + 1
    code = np.array([index.get(v, len(vocab)) for v in nodes[label].to_numpy()], dtype=np.int64)
    pre, post, w = edges.pre.to_numpy(), edges.post.to_numpy(), edges.w.to_numpy(np.float64)
    n = len(nodes)
    blocks = []
    for cell, partner in ((post, pre), (pre, post)):          # inputs, then outputs
        counts = np.bincount(cell * k + code[partner], weights=w, minlength=n * k).reshape(n, k)
        blocks.append(counts / np.maximum(counts.sum(1, keepdims=True), 1.0))
    return np.hstack(blocks).astype(np.float32)


def head_report(logits_of, nodes, classes, targets=("nt_known", "nt_train", "nt_best")):
    """The supervised model's own classifier, scored on MCNS brain cells."""
    pred = classes[logits_of.argmax(1)]
    brain = nodes["region"].to_numpy() == "brain"
    out = {}
    for target in targets:
        values = label_strings(nodes[target])
        keep = brain & values.notna().to_numpy() & values.isin(set(classes)).to_numpy()
        out[f"{target}/brain"] = {"n": int(keep.sum()), **_score(values[keep].to_numpy(), pred[keep], classes)}
    return out


# --------------------------------------------------------------------------- main

def main() -> None:
    """Build one baseline: fit on the source volume, embed both volumes, write
    embeddings and the build report.
    """
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--method", choices=METHODS, required=True)
    p.add_argument("--supervise-on", default="nt_best", help="label for sage/gin/gat")
    p.add_argument("--processed", type=Path, default=Path("data/processed"))
    p.add_argument("--reports", type=Path, default=Path("experiments/baselines"))
    p.add_argument("--source", default="fafb")
    p.add_argument("--target", default="mcns")
    p.add_argument("--scope", choices=SCOPES, default="brain_neurons",
                   help="source cells that take part: brain neurons with at least one connection "
                        "(brain_neurons, the paper's setting) or every cell (all)")
    p.add_argument("--split-column", default="split_random")
    p.add_argument("--features", default=DEFAULT_BLOCKS,
                   help="feature blocks; the default is the paper encoder's")
    p.add_argument("--column-set", default="refined")
    p.add_argument("--backbone", choices=tuple(LAYERS), default="sage",
                   help="encoder for the self-supervised methods")
    p.add_argument("--width", type=int, default=512, help="embedding width; the encoder's is 512")
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=None, help="default depends on the method")
    p.add_argument("--lr", type=float, default=None, help="default depends on the method")
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--mask-rate", type=float, default=0.5, help="graphmae")
    p.add_argument("--sce-gamma", type=float, default=2.0, help="graphmae")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="")
    args = p.parse_args()

    method = args.method
    # --- 1. Settings and the run's name, e.g. gat_ntbest_s0, raw_s0_mcns_to_fafb
    args.epochs = args.epochs or EPOCHS.get(method, 0)
    args.lr = args.lr or LR.get(method, 0.0)
    name = method
    if method in SUPERVISED:
        name += f"_{SHORT.get(args.supervise_on, args.supervise_on)}"
    elif method in SELF_SUPERVISED and args.backbone != "sage":
        name += f"_{args.backbone}"
    name += f"_s{args.seed}" + (f"_{args.tag}" if args.tag else "")
    if (args.source, args.target) != ("fafb", "mcns"):
        name += f"_{args.source}_to_{args.target}"

    # --- 2. Seeds, device, and the source volume's standardised inputs
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.reports.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    started = time.time()
    log(f"baseline {name}: method {method} · device {device}")

    src_nodes, src_split, src_edges, src_x, names, stats = volume_inputs(
        args.processed, args.source, args.features, args.column_set, args.split_column, scope=args.scope)
    log(f"  {args.source}: {len(src_nodes):,} cells · {len(src_edges):,} edges · {feature_summary(names)}")
    report = {"name": name, "method": method, "args": vars(args), "inputs": names}

    # --- 3a. Feature-only baselines (raw, degree, composition): the embedding is the input itself
    if method in FEATURE_ONLY:
        tgt_nodes, _, tgt_edges, tgt_x, _, _ = volume_inputs(
            args.processed, args.target, args.features, args.column_set, args.split_column, stats=stats)
        if method == "raw":
            embs = (src_x, tgt_x)
        elif method == "degree":
            embs = (degree_features(src_edges, len(src_nodes)), degree_features(tgt_edges, len(tgt_nodes)))
        else:
            train = (src_split == "train").to_numpy()
            vocab = sorted(set(src_nodes["super_class"][train].dropna()), key=str)
            embs = (composition(src_nodes, src_edges, vocab), composition(tgt_nodes, tgt_edges, vocab))
            report["setting"] = "partial labels: MCNS partner super-class is read"
            report["vocab"] = vocab
    else:
        # --- 3b. Graph networks: train on the source's in-scope graph, then embed it.
        # With --scope brain_neurons the cells outside the scope (in FAFB, the neurons
        # without connections) are removed before training, so no label, loss term or
        # batch-norm statistic comes from them. Their
        # embedding rows are left at zero: the readout never fits or scores them.
        # With --scope all every cell is kept.
        model = GNN(method if method in SUPERVISED else args.backbone, src_x.shape[1], args.width,
                    layers=args.layers, dropout=args.dropout, heads=args.heads).to(device)
        keep = np.flatnonzero((src_split != EXCLUDED).to_numpy())
        remap = np.full(len(src_nodes), -1, dtype=np.int64)
        remap[keep] = np.arange(len(keep))
        pre, post = remap[src_edges.pre.to_numpy()], remap[src_edges.post.to_numpy()]
        inside = (pre >= 0) & (post >= 0)
        sub_edges = src_edges[inside].assign(pre=pre[inside], post=post[inside])
        log(f"  training graph: {len(keep):,} of {len(src_nodes):,} cells, {len(sub_edges):,} of "
            f"{len(src_edges):,} edges (scope {args.scope})")
        g = Graph.from_edges(sub_edges, len(keep), device)
        x = torch.from_numpy(src_x[keep]).to(device)
        sub_nodes, sub_split = src_nodes.iloc[keep].reset_index(drop=True), src_split.iloc[keep].reset_index(drop=True)
        del src_edges, sub_edges
        log(f"  training {method} for {args.epochs} epochs · lr {args.lr} · width {args.width} · "
            f"{sum(p.numel() for p in model.parameters()):,} parameters")
        if method in SUPERVISED:
            head, classes, report["training"] = train_supervised(
                model, x, g, sub_nodes, sub_split, args.supervise_on, args, device)
        else:
            report["training"] = {"dgi": train_dgi, "bgrl": train_bgrl,
                                  "graphmae": train_graphmae}[method](model, x, g, args, device)
        model.eval()
        with torch.no_grad():
            src_h = model(x, g)
            src_emb = np.zeros((len(src_nodes), src_h.shape[1]), dtype=np.float32)
            src_emb[keep] = src_h.cpu().numpy()
            if method in SUPERVISED:
                src_logits = np.zeros((len(src_nodes), len(classes)), dtype=np.float32)
                src_logits[keep] = head(src_h).cpu().numpy()
        del g, x, src_h
        if device.type == "cuda":
            report["peak_gpu_gib"] = torch.cuda.max_memory_allocated() / 2 ** 30
            torch.cuda.empty_cache()

        tgt_nodes, _, tgt_edges, tgt_x, _, _ = volume_inputs(
            args.processed, args.target, args.features, args.column_set, args.split_column, stats=stats)
        # Embed the target with the frozen network (its inputs scaled with the source's statistics)
        g = Graph.from_edges(tgt_edges, len(tgt_nodes), device)
        del tgt_edges
        with torch.no_grad():
            tgt_h = model(torch.from_numpy(tgt_x).to(device), g)
            if method in SUPERVISED:
                tgt_logits = head(tgt_h).cpu().numpy()
            tgt_emb = tgt_h.cpu().numpy()
        del g, tgt_h
        embs = (src_emb, tgt_emb)

        if method in SUPERVISED:
            # Supervised networks: also score their own classifier head, on FAFB test and MCNS
            test = (src_split == "test").to_numpy()
            values = label_strings(src_nodes[args.supervise_on])
            ok = test & values.isin(set(classes)).to_numpy()
            report["head"] = {
                "classes": classes.tolist(),
                "source_test": _score(values[ok].to_numpy(), classes[src_logits[ok].argmax(1)], classes),
                "zero_shot": head_report(tgt_logits, tgt_nodes, classes)}
            log(f"  own head, source test macro-F1 {report['head']['source_test']['macro_f1']:.4f}")
            for key, res in report["head"]["zero_shot"].items():
                log(f"  own head, zero-shot {key:<16} {res['macro_f1']:.4f}   n={res['n']:,}")

    # --- 4. Write both volumes' embeddings (float16) and the build report
    for vol, emb in zip((args.source, args.target), embs):
        path = args.reports / f"{name}_embeddings_{vol}.npy"
        np.save(path, emb.astype(np.float16))
        log(f"  {vol}: embeddings {emb.shape} · effective rank {effective_rank(emb.astype(np.float64)):.2f} "
            f"-> {path}")
    report["effective_rank"] = {vol: effective_rank(e.astype(np.float64))
                                for vol, e in zip((args.source, args.target), embs)}
    report["seconds"] = time.time() - started
    out = args.reports / f"{name}_build.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    log(f"wrote {out}")
    log(f"score it: PYTHONPATH=src python scripts/score_embeddings.py --name {name} "
        f"--source-emb {args.reports}/{name}_embeddings_{args.source}.npy "
        f"--target-emb {args.reports}/{name}_embeddings_{args.target}.npy --dump-predictions")


if __name__ == "__main__":
    main()
