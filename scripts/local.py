#!/usr/bin/env python3
"""Local topology: reciprocity, core number, partner degree and directed triangles.

The degree and connection columns count partners and describe weight
distributions. None measures reciprocity, clustering or core membership, so this
block adds them. Every column is isomorphism-invariant and identity-free, so the
block is computed identically on any volume and transfers by construction.

**Two groups, because their costs differ by two orders of magnitude:**
- `cheap`, O(nnz), about a minute: reciprocity by count and by weight
  (`A_bin ⊙ A_binᵀ`), core number, and the degree statistics of a cell's
  partners.
- `triangles`: the masked product `(A[B] @ A) ⊙ A[B]` over row blocks,
  checkpointed, because forming `A @ A` outright densifies to order 3×10⁹
  non-zeros.

**Directed triangles are two statistics, and both are kept.** A cyclic triangle
`i→j→k→i` is a feedback loop; a transitive one (`i→j→k` with `i→k`) is a
feed-forward motif. Counts are fractions of what the cell's degree permits, since
a raw triangle count mostly restates degree.

Writes `data/processed/{volume}_local{suffix}.parquet`. The encoder reads
`_local` and `_local_tri`.

    qsub -v GROUPS=cheap hpc/jobs/local.sh
    qsub -v GROUPS=triangles,SUFFIX=_tri hpc/jobs/local.sh
"""
from __future__ import annotations

import argparse
import atexit
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from wiretype.data.features import effective_rank
from wiretype.log import log


def core_numbers(A: sp.csr_matrix) -> np.ndarray:
    """Core number per node, by peeling (Batagelj & Zaversnik).

    O(V + E) with a bucket ordering. networkx would do this too but wants the
    whole 15.1M-edge graph as Python objects, which is not a trade worth making
    for thirty lines of array code.
    """
    n = A.shape[0]
    deg = np.diff(A.indptr).astype(np.int64)
    order = np.argsort(deg, kind="stable")
    pos = np.empty(n, dtype=np.int64)
    pos[order] = np.arange(n)
    order = order.copy()
    deg = deg.copy()
    # bin[d] = first index in `order` holding a node of degree d
    max_deg = int(deg.max()) if n else 0
    counts = np.bincount(deg, minlength=max_deg + 2)
    bins = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64)
    core = deg.copy()
    for i in range(n):
        v = order[i]
        for u in A.indices[A.indptr[v]:A.indptr[v + 1]]:
            if deg[u] > deg[v]:
                du, pu = deg[u], pos[u]
                pw = bins[du]
                w = order[pw]
                if u != w:
                    pos[u], pos[w] = pw, pu
                    order[pu], order[pw] = w, u
                bins[du] += 1
                deg[u] -= 1
        core[v] = deg[v]
    return core


def cheap_block(A: sp.csr_matrix, n: int) -> dict[str, np.ndarray]:
    """Reciprocity, core number and partner-degree statistics. All O(nnz)."""
    binary = A.copy()
    binary.data = np.ones_like(binary.data)
    binary = binary.tocsr()
    recip = binary.multiply(binary.T).tocsr()          # 1 iff i->j and j->i
    recip_w = A.multiply(recip).tocsr()

    out_n = np.asarray(binary.sum(1)).ravel()
    in_n = np.asarray(binary.sum(0)).ravel()
    out_w = np.asarray(A.sum(1)).ravel()
    in_w = np.asarray(A.sum(0)).ravel()
    r_n = np.asarray(recip.sum(1)).ravel()
    r_w_out = np.asarray(recip_w.sum(1)).ravel()
    r_w_in = np.asarray(recip_w.sum(0)).ravel()
    safe = lambda a, b: np.divide(a, b, out=np.zeros(n, np.float64), where=b > 0)

    sym = ((binary + binary.T) > 0).astype(np.int8).tocsr()
    sym.setdiag(0); sym.eliminate_zeros()
    core = core_numbers(sym)
    sym_deg = np.diff(sym.indptr).astype(np.float64)

    # What kind of cell do I attach to, by degree? Classic assortativity, absent.
    logdeg = np.log1p(out_n + in_n)
    out_norm = sp.diags(safe(1.0, out_n)) @ binary
    in_norm = sp.diags(safe(1.0, in_n)) @ binary.T
    mean_out = np.asarray(out_norm @ logdeg).ravel()
    mean_in = np.asarray(in_norm @ logdeg).ravel()
    sq_out = np.asarray(out_norm @ (logdeg ** 2)).ravel()
    sq_in = np.asarray(in_norm @ (logdeg ** 2)).ravel()

    return {
        "recip_frac_out": safe(r_n, out_n),
        "recip_frac_in": safe(r_n, in_n),
        "recip_wfrac_out": safe(r_w_out, out_w),
        "recip_wfrac_in": safe(r_w_in, in_w),
        "log_kcore": np.log1p(core.astype(np.float64)),
        "kcore_frac": safe(core.astype(np.float64), sym_deg),
        "nbr_logdeg_out": mean_out,
        "nbr_logdeg_in": mean_in,
        "nbr_logdeg_out_sd": np.sqrt(np.maximum(sq_out - mean_out ** 2, 0)),
        "nbr_logdeg_in_sd": np.sqrt(np.maximum(sq_in - mean_in ** 2, 0)),
    }


def triangle_block(A: sp.csr_matrix, n: int, block: int,
                   checkpoint: Path, progress: Path) -> dict[str, np.ndarray]:
    """Directed triangle participation, by masked product in row blocks.

    For a block of rows B, `(A[B] @ M) . A[B]` keeps only the two-step paths that
    close on an existing edge, so nothing outside the graph is ever materialised.
    Cyclic uses `A @ A` masked by `A.T` (i->j->k and k->i); transitive uses
    `A @ A` masked by `A` (i->j->k and i->k).
    """
    binary = A.copy(); binary.data = np.ones_like(binary.data); binary = binary.tocsr()
    bt = binary.T.tocsr()
    sym = ((binary + bt) > 0).astype(np.float32).tocsr()
    sym.setdiag(0); sym.eliminate_zeros()

    out = np.zeros((n, 3), dtype=np.float64)
    resume = 0
    if checkpoint.exists() and progress.exists():
        state = json.loads(progress.read_text())
        if state.get("n") == n:
            out = np.load(checkpoint); resume = int(state["done_upto"])
            log(f"  resuming triangles from cell {resume:,} of {n:,}")

    started = time.time()
    for start in range(resume, n, block):
        stop = min(start + block, n)
        rows = slice(start, stop)
        b_out = binary[rows]
        two = (b_out @ binary).tocsr()                       # i -> j -> k
        out[start:stop, 0] = np.asarray(two.multiply(bt[rows]).sum(1)).ravel()   # cyclic
        out[start:stop, 1] = np.asarray(two.multiply(b_out).sum(1)).ravel()      # transitive
        s = sym[rows]
        out[start:stop, 2] = np.asarray((s @ sym).multiply(s).sum(1)).ravel()    # undirected
        done = stop
        if (start // block) % 20 == 0:
            rate = max(done - resume, 1) / max(time.time() - started, 1e-9)
            log(f"  triangles: {done:>7,} / {n:,}   eta {(n-done)/rate/60:>5.1f} min")
        if (start // block) % 100 == 0 or done >= n:
            np.save(checkpoint, out)
            progress.write_text(json.dumps({"n": n, "done_upto": done}))
    checkpoint.unlink(missing_ok=True); progress.unlink(missing_ok=True)

    sym_deg = np.diff(sym.indptr).astype(np.float64)
    pairs = sym_deg * (sym_deg - 1.0)
    safe = lambda a, b: np.divide(a, b, out=np.zeros(n, np.float64), where=b > 0)
    return {
        # Fractions of what the degree permits, not raw counts -- a raw count is
        # mostly a restatement of degree and the vector has eight of those.
        "tri_cycle_frac": safe(out[:, 0], pairs),
        "tri_trans_frac": safe(out[:, 1], pairs),
        "clust_coef": safe(out[:, 2], pairs),
        "log_tri_undirected": np.log1p(out[:, 2] / 2.0),
    }


def main() -> None:
    """Compute and write the requested local-topology feature groups for one volume."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--processed", type=Path, default=Path("data/processed"))
    p.add_argument("--reports", type=Path, default=Path("experiments/features"))
    p.add_argument("--volume", default="fafb")
    p.add_argument("--groups", default="cheap",
                   help="cheap (O(nnz), ~1 min) and/or triangles (blockwise, benchmark "
                        "before requesting wall clock). They are independent.")
    p.add_argument("--block", type=int, default=512)
    p.add_argument("--suffix", default="")
    args = p.parse_args()

    groups = [g.strip() for g in args.groups.split(",") if g.strip()]
    unknown = set(groups) - {"cheap", "triangles"}
    if unknown:
        raise SystemExit(f"unknown groups: {sorted(unknown)} — choose from cheap, triangles")
    if not groups:
        raise SystemExit("--groups selected nothing to compute")
    args.reports.mkdir(parents=True, exist_ok=True)

    owner = os.environ.get("JOB_ID", f"pid{os.getpid()}")
    lock = args.processed / f"{args.volume}_local{args.suffix}.lock"
    if lock.exists():
        raise SystemExit(f"{lock} is held by {lock.read_text().strip()}. Give this run a "
                         f"different --suffix, or clear the lock if that job died.")
    lock.write_text(owner)

    def release():
        """Remove this job's lock file, if it still owns it."""
        if lock.exists() and lock.read_text().strip() == owner:
            lock.unlink(missing_ok=True)
    atexit.register(release)

    nodes = pd.read_parquet(args.processed / f"{args.volume}_nodes.parquet").sort_values("node_id")
    edges = pd.read_parquet(args.processed / f"{args.volume}_edges.parquet")
    n = len(nodes)
    log(f"{n:,} cells, {len(edges):,} connections, groups={groups}")

    A = sp.coo_matrix((edges["w"].to_numpy(np.float64),
                       (edges["pre"].to_numpy(np.int64), edges["post"].to_numpy(np.int64))),
                      shape=(n, n)).tocsr()
    A.sum_duplicates()
    A.setdiag(0); A.eliminate_zeros()

    frame = pd.DataFrame({"node_id": nodes.node_id.to_numpy()})
    report = {"volume": args.volume, "groups": groups}

    if "cheap" in groups:
        t0 = time.time()
        for name, values in cheap_block(A, n).items():
            frame[f"local_{name}"] = values.astype(np.float32)
        log(f"  cheap block done in {time.time()-t0:.1f}s")
        log(f"  reciprocity: {frame['local_recip_frac_out'].mean():.1%} of out-partners "
            f"are reciprocal on average")

    if "triangles" in groups:
        t0 = time.time()
        for name, values in triangle_block(
                A, n, args.block,
                args.processed / f"{args.volume}_local{args.suffix}_tri.partial.npy",
                args.processed / f"{args.volume}_local{args.suffix}_tri.partial.json").items():
            frame[f"local_{name}"] = values.astype(np.float32)
        log(f"  triangle block done in {(time.time()-t0)/60:.1f} min")
        log(f"  mean clustering coefficient {frame['local_clust_coef'].mean():.4f}")

    cols = [c for c in frame.columns if c != "node_id"]
    out = args.processed / f"{args.volume}_local{args.suffix}.parquet"
    tmp = out.with_suffix(f".parquet.partial{os.getpid()}")
    frame.to_parquet(tmp, index=False)
    os.replace(tmp, out)
    log(f"wrote {out}  ({len(cols)} columns)")

    from wiretype.data.features import load_features, standardise
    base, _ = load_features(args.processed, args.volume, edges, n)
    mask = pd.read_parquet(args.processed / f"{args.volume}_splits.parquet") \
             .sort_values("node_id")["split"].to_numpy() == "train"
    new = frame[cols].to_numpy(np.float32)
    report["effective_rank_base"] = effective_rank(standardise(base, mask=mask))
    report["effective_rank_new_alone"] = effective_rank(standardise(new, mask=mask))
    report["effective_rank_combined"] = effective_rank(
        standardise(np.hstack([base, new]), mask=mask))
    report["columns"] = cols
    gain = report["effective_rank_combined"] - report["effective_rank_base"]
    log("")
    log(f"  effective rank, existing {base.shape[1]}        {report['effective_rank_base']:>6.2f}")
    log(f"  effective rank, these {len(cols)} alone      {report['effective_rank_new_alone']:>6.2f}")
    log(f"  effective rank, together            {report['effective_rank_combined']:>6.2f}")
    log(f"  directions added: {gain:+.2f}  (reported, not a decision)")
    (args.reports / f"local_{args.volume}{args.suffix}.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
