"""Where a cell sits on the feedforward axis: trophic level and SpringRank.

Every other feature is a local statistic of a cell's own connections: how many,
how strong, how distributed. This is a **global** position, inferred from edge
directions across the whole graph, placing a cell between sensory input and motor
output. It is one sparse linear solve, isomorphism-invariant (no node identity),
and computed identically in every volume.

## The two methods, and the fact that they are one method

**Trophic levels** (MacKay, Johnson & Sansom 2020) assign each node a height `h`
minimising

    F = sum_ij w_ij (h_j - h_i - 1)^2 / sum_ij w_ij

so that every edge i→j would ideally climb exactly one level. The minimiser
solves the sparse linear system

    L h = d_in - d_out,      L = diag(d_in + d_out) - (A + A^T)

with `L` the Laplacian of the symmetrised weighted graph. `F` itself is the
**trophic incoherence**: 0 for a perfectly stratified feedforward network, 1 for
one with no directional structure at all.

**SpringRank** (De Bacco, Larremore & Moore 2018) treats each edge as a spring of
rest length 1 pulling `i` one unit above `j`, and minimises

    H = 1/2 sum_ij A_ij (s_i - s_j - 1)^2

which solves `(alpha I + L) s = d_out - d_in`.

**Substituting s = -h makes the two objectives identical, and at alpha = 0 the
two linear systems are the same system with the right-hand side negated.** They
are one quantity under two sign conventions, from two literatures.
`scripts/flow.py` verifies this numerically rather than assuming it.

What is *not* shared is the regularisation. SpringRank's `alpha > 0` adds a
spring to the origin, which makes the system non-singular, shrinks poorly
determined nodes towards zero, and comes with the generative model and the
significance test that trophic levels do not have. That is where any real
difference between the two has to come from, so it is swept rather than assumed.

## Two choices this module makes explicit

**Weighting.** The solve can run on raw synapse counts, on `log1p` of them, or on
a binary adjacency, and they are different questions: raw counts let the few
2,405-synapse connections set the axis, binary lets the 49.7% single-synapse
edges (about 42% reproducible between brains) vote equally with them. `log1p` is
the default because the encoder's partner tokens use it too.
All three are computed and reported; if the axis moves between them, that is a
result about the noise floor and not a nuisance to be averaged away.

**Disconnected nodes.** `L` has one null direction per weakly connected
component, so levels are fixed only up to a constant per component and are
centred to zero mean within each. A node with no edges at all is its own
component with no constraint on it; those get 0 and are counted in the report
rather than quietly imputed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import cg

# The three readings of an edge weight. Named rather than passed as a flag so a
# report says which one it used.
WEIGHTINGS = {
    "binary": lambda w: np.ones(len(w), dtype=np.float64),
    "raw": lambda w: w.astype(np.float64),
    "log1p": lambda w: np.log1p(w.astype(np.float64)),
}
DEFAULT_WEIGHTING = "log1p"


def laplacian_system(edges: pd.DataFrame, n_nodes: int, weighting: str = DEFAULT_WEIGHTING):
    """`(L, d_in, d_out)` — the shared machinery of both methods.

    `L` is the combinatorial Laplacian of the **symmetrised** weighted graph,
    which is where the direction information goes: the left-hand side forgets
    direction entirely and the right-hand side is the net imbalance `d_in -
    d_out` that carries it. A cell with balanced traffic contributes nothing to
    the right-hand side and takes its level from its neighbours.
    """
    if weighting not in WEIGHTINGS:
        raise ValueError(f"weighting must be one of {sorted(WEIGHTINGS)}, got {weighting!r}")
    pre = edges["pre"].to_numpy(np.int64)
    post = edges["post"].to_numpy(np.int64)
    w = WEIGHTINGS[weighting](edges["w"].to_numpy())

    adjacency = sp.csr_matrix((w, (pre, post)), shape=(n_nodes, n_nodes))
    symmetrised = (adjacency + adjacency.T).tocsr()
    d_out = np.asarray(adjacency.sum(axis=1)).ravel()
    d_in = np.asarray(adjacency.sum(axis=0)).ravel()
    laplacian = (sp.diags(d_in + d_out) - symmetrised).tocsr()
    return laplacian, d_in, d_out


def components(edges: pd.DataFrame, n_nodes: int) -> np.ndarray:
    """Weak components, as the label per node. Direction is dropped on purpose —
    the Laplacian's null space is one constant per *weakly* connected part."""
    pre = edges["pre"].to_numpy(np.int64)
    post = edges["post"].to_numpy(np.int64)
    ones = np.ones(len(pre), dtype=np.int8)
    graph = sp.csr_matrix((ones, (pre, post)), shape=(n_nodes, n_nodes))
    _, labels = connected_components(graph, directed=True, connection="weak")
    return labels


def _centre_per_component(values: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Zero mean within each component — the only gauge choice available.

    Levels are differences, so a component's absolute offset means nothing and
    comparing two components' raw values would be comparing two arbitrary
    constants. Centring makes the feature mean what it says within a component
    and admits that it says nothing across them.
    """
    sums = np.bincount(labels, weights=values, minlength=labels.max() + 1)
    counts = np.bincount(labels, minlength=labels.max() + 1)
    return values - (sums / np.maximum(counts, 1))[labels]


def _solve(laplacian, rhs, diagonal, rtol: float, maxiter: int):
    """Jacobi-preconditioned CG, and the residual it reached.

    `L` is singular — one null direction per component — but `rhs` is orthogonal
    to every one of them by construction (a component's in-weight and out-weight
    are the same sum counted twice), so the system is consistent and CG stays in
    the range. Drift along the null space is removed afterwards by centring, so
    it costs nothing.

    Isolated nodes have a zero row, a zero right-hand side and a zero
    preconditioner entry, so they never move from the zero start.
    """
    inverse = np.where(diagonal > 0, 1.0 / np.maximum(diagonal, 1e-300), 0.0)
    solution, status = cg(
        laplacian, rhs, rtol=rtol, maxiter=maxiter, M=sp.diags(inverse)
    )
    scale = np.linalg.norm(rhs)
    residual = float(np.linalg.norm(laplacian @ solution - rhs) / max(scale, 1e-300))
    return solution, {"cg_status": int(status), "relative_residual": residual}


def trophic_levels(edges, n_nodes, weighting=DEFAULT_WEIGHTING, rtol=1e-10, maxiter=5000,
                   labels=None):
    """`(h, info)` — MacKay, Johnson & Sansom trophic levels, centred per component.

    High `h` means a cell is far downstream: it receives more than it sends,
    relative to everything it is connected to.
    """
    laplacian, d_in, d_out = laplacian_system(edges, n_nodes, weighting)
    if labels is None:
        labels = components(edges, n_nodes)
    levels, info = _solve(laplacian, d_in - d_out, d_in + d_out, rtol, maxiter)
    levels = _centre_per_component(levels, labels)
    info |= {
        "weighting": weighting,
        "incoherence": trophic_incoherence(edges, levels, weighting),
        "isolated_nodes": int((d_in + d_out == 0).sum()),
    }
    return levels, info


def springrank(edges, n_nodes, alpha=0.0, weighting=DEFAULT_WEIGHTING, rtol=1e-10,
               maxiter=5000, labels=None):
    """`(s, info)` — SpringRank. High `s` means upstream, the opposite of `h`.

    At `alpha = 0` this is the trophic system with the right-hand side negated,
    and the returned vector should equal `-h` to solver tolerance. `alpha > 0`
    adds the spring to the origin: the system becomes non-singular, no centring
    is needed or applied, and nodes whose position the edges barely determine are
    pulled towards zero.
    """
    laplacian, d_in, d_out = laplacian_system(edges, n_nodes, weighting)
    diagonal = d_in + d_out
    if alpha > 0:
        laplacian = (laplacian + alpha * sp.eye(n_nodes, format="csr")).tocsr()
        diagonal = diagonal + alpha
    ranks, info = _solve(laplacian, d_out - d_in, diagonal, rtol, maxiter)
    if alpha == 0:
        if labels is None:
            labels = components(edges, n_nodes)
        ranks = _centre_per_component(ranks, labels)
    info |= {"weighting": weighting, "alpha": float(alpha)}
    return ranks, info


def trophic_incoherence(edges: pd.DataFrame, levels: np.ndarray,
                        weighting: str = DEFAULT_WEIGHTING) -> float:
    """`F` — the weighted mean squared deviation from a one-level climb per edge.

    A single number for the whole graph, and a reportable result in its own
    right: it says how close the connectome is to a feedforward hierarchy. 0 is
    perfectly stratified, 1 is no directional structure at all. It is also the
    honest health check on the feature — if `F` is near 1, the axis the levels
    describe is not one the graph actually has, and a probe is unlikely to find
    anything on it.
    """
    pre = edges["pre"].to_numpy(np.int64)
    post = edges["post"].to_numpy(np.int64)
    w = WEIGHTINGS[weighting](edges["w"].to_numpy())
    gap = levels[post] - levels[pre] - 1.0
    return float((w * gap ** 2).sum() / w.sum())
