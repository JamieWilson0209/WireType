#!/usr/bin/env python3
"""Random-walk structural encoding: exact return probabilities after 1..K steps.

A return probability measures cycles and local clustering around a cell, which
degree counts do not. The encoder's `refined` set keeps four of them.

**Exact, by propagating a block of unit vectors.** RWSE is `diag(P^k)`, and
forming `P^k` densifies: P has 15.1M non-zeros in FAFB, but a typical cell has
about 24,600 cells within two hops, so P² has order 3×10⁹. The rejected
alternatives:
- Hutchinson's estimator is noisiest exactly where return probabilities are
  smallest.
- Pruned sparse powers bias the diagonal downward.

Here `X ← P X` is iterated over blocks of unit vectors, reading each block's own
diagonal entries at every step. Fixed memory (0.6 GB at block 512); about 3 h
for K = 8 on FAFB.

The transition matrix is synapse-weighted, symmetrised and row-normalised.

    qsub hpc/jobs/rwse.sh
    qsub -v MAX_K=4 hpc/jobs/rwse.sh
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from wiretype.data.features import effective_rank
from wiretype.log import log


def transition_matrix(edges: pd.DataFrame, n_nodes: int) -> sp.csr_matrix:
    """Row-normalised, synapse-weighted, symmetrised — the sampler's walk."""
    pre = edges["pre"].to_numpy(np.int64)
    post = edges["post"].to_numpy(np.int64)
    w = edges["w"].to_numpy(np.float64)
    src = np.concatenate([pre, post])
    dst = np.concatenate([post, pre])
    weight = np.concatenate([w, w])

    A = sp.coo_matrix((weight, (src, dst)), shape=(n_nodes, n_nodes)).tocsr()
    A.sum_duplicates()
    totals = np.asarray(A.sum(axis=1)).ravel()
    # A cell with no connections keeps an all-zero row, so its return
    # probabilities are zero rather than undefined.
    inverse = np.divide(1.0, totals, out=np.zeros_like(totals), where=totals > 0)
    return (sp.diags(inverse) @ A).astype(np.float32).tocsr()


def main() -> None:
    """Compute and write the random-walk return probabilities for one volume."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--processed", type=Path, default=Path("data/processed"))
    p.add_argument("--reports", type=Path, default=Path("experiments/features"))
    p.add_argument("--volume", default="fafb")
    p.add_argument("--max-k", type=int, default=8)
    p.add_argument("--block", type=int, default=512)
    args = p.parse_args()

    args.reports.mkdir(parents=True, exist_ok=True)
    nodes = pd.read_parquet(args.processed / f"{args.volume}_nodes.parquet").sort_values("node_id")
    edges = pd.read_parquet(args.processed / f"{args.volume}_edges.parquet")
    n = len(nodes)
    log(f"{n:,} cells, {len(edges):,} connections -> {2*len(edges):,} traversable steps")

    P = transition_matrix(edges, n)
    log(f"transition matrix built: {P.nnz:,} non-zeros")

    # Checkpoint every few hundred blocks so a wall-clock kill costs the time
    # since the last save rather than the whole run. The MCNS pass is ~9 hours
    # and the first attempt was sized against FAFB's benchmark, so it was killed
    # at 60% with nothing written.
    checkpoint = args.processed / f"{args.volume}_rwse_partial.npy"
    progress = args.processed / f"{args.volume}_rwse_partial.json"
    rwse = np.zeros((n, args.max_k), dtype=np.float32)
    resume_from = 0
    if checkpoint.exists() and progress.exists():
        state = json.loads(progress.read_text())
        if state.get("n") == n and state.get("max_k") == args.max_k:
            rwse = np.load(checkpoint)
            resume_from = int(state["done_upto"])
            log(f"resuming from cell {resume_from:,} of {n:,}")

    started = time.time()
    for start in range(resume_from, n, args.block):
        ids = np.arange(start, min(start + args.block, n))
        X = np.zeros((n, len(ids)), dtype=np.float32)
        X[ids, np.arange(len(ids))] = 1.0
        for k in range(args.max_k):
            X = P @ X
            rwse[ids, k] = X[ids, np.arange(len(ids))]
        done = start + len(ids)
        if (start // args.block) % 20 == 0:
            rate = max(done - resume_from, 1) / max(time.time() - started, 1e-9)
            log(f"  {done:>7,} / {n:,} cells   eta {(n-done)/rate/60:>5.1f} min")
        if (start // args.block) % 200 == 0 or done >= n:
            np.save(checkpoint, rwse)
            progress.write_text(json.dumps({"n": n, "max_k": args.max_k, "done_upto": done}))

    frame = pd.DataFrame({"node_id": nodes.node_id.to_numpy()})
    for k in range(args.max_k):
        frame[f"rwse_{k+1}"] = rwse[:, k]
    out = args.processed / f"{args.volume}_rwse.parquet"
    frame.to_parquet(out, index=False)
    log(f"wrote {out}")
    checkpoint.unlink(missing_ok=True)
    progress.unlink(missing_ok=True)

    # The point of the exercise: does this carry directions the degree features
    # do not? A second rank-2 vector would mean the feature problem is deeper
    # than a missing eight numbers.
    report = {
        "volume": args.volume,
        "max_k": args.max_k,
        "per_step": {
            f"k={k+1}": {
                "mean": float(rwse[:, k].mean()),
                "median": float(np.median(rwse[:, k])),
                "nonzero_fraction": float((rwse[:, k] > 0).mean()),
                "max": float(rwse[:, k].max()),
            }
            for k in range(args.max_k)
        },
        "effective_rank_rwse": effective_rank(np.log1p(rwse * 1e4)),
    }

    # The number that actually decides whether this was worth computing. RWSE at
    # even k is partly a degree feature by construction: on a symmetrised graph
    # diag(P^2)_i = sum_j w_ij^2 / (W_i W_j), which carries a 1/W_i scaling. So
    # RWSE having its own rank is not enough — it has to add directions the
    # degree features do not already have, and only the combined matrix says so.
    from wiretype.data.features import degree_features, standardise

    combined = standardise(
        np.hstack([degree_features(edges, n), np.log1p(rwse * 1e4)])
    )
    degree_only = standardise(degree_features(edges, n))
    report["effective_rank_degree_only"] = effective_rank(degree_only)
    report["effective_rank_combined"] = effective_rank(combined)
    log("")
    for k, v in report["per_step"].items():
        log(f"  {k:<5} mean {v['mean']:.3e}  median {v['median']:.3e}  "
            f"nonzero {v['nonzero_fraction']:.1%}  max {v['max']:.3e}")
    log("")
    log(f"  effective rank, degree features alone   {report['effective_rank_degree_only']:>6.2f} of 8")
    log(f"  effective rank, RWSE alone              {report['effective_rank_rwse']:>6.2f} of {args.max_k}")
    log(f"  effective rank, both together           {report['effective_rank_combined']:>6.2f} "
        f"of {8 + args.max_k}")
    gain = report["effective_rank_combined"] - report["effective_rank_degree_only"]
    log("")
    log(f"  RWSE adds {gain:.2f} directions to what degree already carried.")
    if gain < 1.0:
        log("  That is almost nothing. RWSE at even k is partly a degree feature by")
        log("  construction — diag(P^2) carries a 1/degree scaling — so if the odd")
        log("  steps are near-zero too, the feature problem is deeper than these")
        log("  eight numbers and the next lever is the pooling, not more features.")
    (args.reports / f"rwse_{args.volume}.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
