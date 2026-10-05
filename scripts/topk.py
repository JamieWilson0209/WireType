#!/usr/bin/env python3
"""Each cell's k strongest connections, as the encoder's partner tokens.

A cell is its k strongest partners, each one a token, and the encoder attends over
them. Nothing is pooled before the encoder.

**Top-k also denoises.** k = 64 keeps 26.1% of FAFB connections and truncates
80.7% of cells. What it discards is the weak tail: single-synapse edges are about
42% consistent between brains, and edges above 10 synapses exceed 90%.

**Direction is a field on the token.** A cell's presynaptic and postsynaptic
partner sets barely overlap (Jaccard 0.14), so each token says which it is.

Writes data/processed/{volume}_topk{k}.npz:
    partner (n, k) int32   neighbour node_id index, -1 where padded
    weight  (n, k) float32 log1p(synapses)
    sign    (n, k) int8    +1 if the seed is presynaptic, -1 if postsynaptic
    length  (n,)   int32   how many of the k slots are real

    qsub -v VOLUME=mcns hpc/jobs/topk.sh
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from wiretype.log import log


def main() -> None:
    """Write each neuron's top-k partners by synapse count (partner, log weight,
    direction).
    """
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--processed", type=Path, default=Path("data/processed"))
    p.add_argument("--volume", default="fafb")
    p.add_argument("-k", type=int, default=64)
    args = p.parse_args()

    nodes = pd.read_parquet(args.processed / f"{args.volume}_nodes.parquet").sort_values("node_id")
    edges = pd.read_parquet(args.processed / f"{args.volume}_edges.parquet")
    n, k = len(nodes), args.k
    log(f"{args.volume}: {n:,} nodes, {len(edges):,} edges, k={k}")

    pre = edges["pre"].to_numpy(np.int64)
    post = edges["post"].to_numpy(np.int64)
    w = edges["w"].to_numpy(np.float64)
    src = np.concatenate([pre, post])
    dst = np.concatenate([post, pre])
    sign = np.concatenate([np.ones(len(pre), np.int8), -np.ones(len(pre), np.int8)])
    weight = np.concatenate([w, w])

    # Strongest first within each cell, so the first k of each block are the
    # top-k. Sorting on (src, -weight) once beats a per-node partition.
    order = np.lexsort((-weight, src))
    src, dst, sign, weight = src[order], dst[order], sign[order], weight[order]

    boundary = np.empty(len(src), dtype=bool)
    boundary[0] = True
    np.not_equal(src[1:], src[:-1], out=boundary[1:])
    starts = np.flatnonzero(boundary)
    rank = np.arange(len(src)) - starts[np.cumsum(boundary) - 1]
    keep = rank < k

    partner = np.full((n, k), -1, dtype=np.int32)
    out_w = np.zeros((n, k), dtype=np.float32)
    out_s = np.zeros((n, k), dtype=np.int8)
    partner[src[keep], rank[keep]] = dst[keep]
    out_w[src[keep], rank[keep]] = np.log1p(weight[keep])
    out_s[src[keep], rank[keep]] = sign[keep]
    length = (partner >= 0).sum(1).astype(np.int32)

    degree = np.bincount(src, minlength=n)
    log(f"  connections per cell: median {np.median(degree):.0f}, p90 {np.percentile(degree,90):.0f}, "
        f"p99 {np.percentile(degree,99):.0f}, max {degree.max():,}")
    log(f"  truncated cells: {(degree > k).mean():6.1%}   connections kept: {length.sum()/len(src):6.1%}")
    log(f"  isolated cells (length 0): {(length == 0).sum():,}")
    log(f"  slots filled: {length.mean():.1f} of {k}")

    # **Written atomically.** np.savez_compressed streams into the destination, so
    # a reader that opens the path while this is running -- or after a killed job
    # -- gets a file that exists, passes an existence check, and raises BadZipFile
    # on load, which kills any training job queued alongside this one. A temp file
    # plus os.replace means the final path never exists in a partial state.
    # numpy appends '.npz' to any path that does not already end in it, so the
    # temp name has to carry the extension itself or os.replace below looks for
    # a file that was never written.
    out = args.processed / f"{args.volume}_topk{k}.npz"
    tmp = out.parent / f"{out.stem}.partial{os.getpid()}.npz"
    np.savez_compressed(tmp, partner=partner, weight=out_w, sign=out_s, length=length)
    os.replace(tmp, out)
    log(f"wrote {out}")


if __name__ == "__main__":
    main()
