#!/usr/bin/env python3
"""Flow-hierarchy position (trophic level, SpringRank), with checks on what it adds.

Every other input column is a local statistic of a cell's own connections. A
trophic level is global: it places a cell between sensory input and motor
output, which no aggregate over its partners can reach. The encoder's `refined`
set keeps `trophic_log1p`.

Writes `data/processed/{volume}_flow.parquet` and a report answering four
questions:
1. **Are trophic level and SpringRank the same number?** Algebraically they are
   (`wiretype.data.flow`): with `s = -h` the objectives coincide, and at
   `alpha = 0` the linear systems differ only in the sign of the right-hand
   side. The report checks this on the real graph.
2. **Is the graph coherent enough for the axis to mean anything?** Trophic
   incoherence `F` is 0 for a perfect hierarchy and 1 for no directional
   structure.
3. **Does it restate degree?** Its correlation with each degree column, and the
   R² of a regression on all eight.
4. **Does it add effective directions** to the degree and connection columns?

`--probe` adds the degree-only probe with and without the flow columns, with
super-class the target to watch.

About 20 s per configuration on FAFB (one sparse CG solve, relative residual 1e-10).

    qsub hpc/jobs/flow.sh
    qsub -v VOLUME=mcns hpc/jobs/flow.sh
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from wiretype.data.features import (DEGREE_FEATURE_NAMES, degree_features,
                                           effective_rank, load_features, standardise)
from wiretype.data.flow import (WEIGHTINGS, components, springrank,
                                       trophic_levels)
from wiretype.log import log

# SpringRank's spring-to-the-origin. 0 is the unregularised MLE and the value at
# which the method coincides with trophic levels; the rest is the sweep that
# says whether regularisation makes it a genuinely different feature or a shrunk
# copy of the same one.
DEFAULT_ALPHAS = (0.0, 0.1, 1.0, 10.0)


def self_test() -> None:
    """Known answers on graphs small enough to check by hand.

    There is no test suite in this project, and a silently wrong linear solve
    would produce a plausible-looking column of numbers that a probe would
    dutifully score. These five cases cost milliseconds and are run before every
    real solve rather than kept as a separate command nobody invokes.
    """
    def edges(pairs, w=1):
        """An edge table from (pre, post) pairs, all with weight `w`."""
        return pd.DataFrame({"pre": [a for a, _ in pairs], "post": [b for _, b in pairs],
                             "w": [w] * len(pairs)})

    # A directed chain climbs exactly one level per edge, and is perfectly coherent.
    levels, info = trophic_levels(edges([(0, 1), (1, 2), (2, 3)]), 4, weighting="binary")
    assert np.allclose(levels, [-1.5, -0.5, 0.5, 1.5]), levels
    assert info["incoherence"] < 1e-12, info

    # A directed cycle has no hierarchy at all: every node flat, F exactly 1.
    levels, info = trophic_levels(edges([(0, 1), (1, 2), (2, 0)]), 3, weighting="binary")
    assert np.allclose(levels, 0.0) and abs(info["incoherence"] - 1.0) < 1e-12, info

    # Two sources feeding two sinks: one clean level of climb.
    levels, info = trophic_levels(edges([(0, 2), (0, 3), (1, 2), (1, 3)]), 4, weighting="binary")
    assert np.allclose(levels, [-0.5, -0.5, 0.5, 0.5]), levels

    # A node with no edges is its own component and is not imputed from anything.
    levels, info = trophic_levels(edges([(0, 1), (1, 2)]), 4, weighting="binary")
    assert levels[3] == 0.0 and info["isolated_nodes"] == 1, info

    # The claim `data/flow.py` makes in prose, checked as arithmetic.
    graph = edges([(0, 1), (1, 2), (2, 3), (0, 3), (3, 1)], w=3)
    levels, _ = trophic_levels(graph, 4)
    ranks, _ = springrank(graph, 4, alpha=0.0)
    assert np.abs(ranks + levels).max() < 1e-9, (ranks, levels)
    log("self-test: 5/5 known answers reproduced")


def correlations(x: np.ndarray, y: np.ndarray) -> dict:
    """Pearson and Spearman. Both, because they answer different questions here:
    Pearson says whether the feature is a linear restatement of another, which is
    what a linear probe would exploit, and Spearman says whether it carries the
    same *ordering*, which is what a k-NN probe would."""
    return {"pearson": float(stats.pearsonr(x, y).statistic),
            "spearman": float(stats.spearmanr(x, y).statistic)}


def explained_by(target: np.ndarray, block: np.ndarray) -> float:
    """R-squared of `target` regressed on `block` — question 3, done properly.

    Correlating against each degree feature one at a time understates the
    overlap: eight features that each correlate at 0.4 can still span the target
    completely between them. This is the number that decides whether the level is
    already implicit in the degree block.
    """
    design = np.hstack([standardise(block), np.ones((len(block), 1), dtype=np.float32)])
    centred = target - target.mean()
    coefficients, *_ = np.linalg.lstsq(design, centred, rcond=None)
    residual = centred - design @ coefficients
    total = float((centred ** 2).sum())
    return float(1.0 - (residual ** 2).sum() / total) if total > 0 else 0.0


def main() -> None:
    """Compute the trophic level and SpringRank for one volume, compare them, and write
    the flow block.
    """
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--processed", type=Path, default=Path("data/processed"))
    parser.add_argument("--reports", type=Path, default=Path("experiments/features"))
    parser.add_argument("--volume", default="fafb")
    parser.add_argument("--alphas", type=float, nargs="*", default=list(DEFAULT_ALPHAS))
    parser.add_argument("--probe", action="store_true",
                        help="run the linear and k-NN probes; the expensive half")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    args.reports.mkdir(parents=True, exist_ok=True)
    self_test()

    nodes = pd.read_parquet(args.processed / f"{args.volume}_nodes.parquet").sort_values("node_id")
    edges = pd.read_parquet(args.processed / f"{args.volume}_edges.parquet")
    n_nodes = len(nodes)
    log(f"{args.volume}: {n_nodes:,} nodes, {len(edges):,} edges")

    labels = components(edges, n_nodes)
    sizes = np.bincount(labels)
    log(f"  {len(sizes):,} weak components, largest {sizes.max():,} "
        f"({sizes.max() / n_nodes:.1%} of cells)")

    report = {"volume": args.volume, "n_nodes": int(n_nodes), "n_edges": int(len(edges)),
              "n_components": int(len(sizes)), "largest_component": int(sizes.max()),
              "trophic": {}, "springrank": {}}

    # ---- the solves -------------------------------------------------------
    levels: dict[str, np.ndarray] = {}
    for weighting in WEIGHTINGS:
        start = time.time()
        values, info = trophic_levels(edges, n_nodes, weighting=weighting, labels=labels)
        levels[weighting] = values
        info["seconds"] = round(time.time() - start, 1)
        info["std"] = float(values.std())
        report["trophic"][weighting] = info
        log(f"  trophic · {weighting:<6} F = {info['incoherence']:.4f}  "
            f"sd {info['std']:.3f}  residual {info['relative_residual']:.1e}  "
            f"{info['seconds']:.0f}s")

    ranks: dict[float, np.ndarray] = {}
    for alpha in args.alphas:
        values, info = springrank(edges, n_nodes, alpha=alpha, labels=labels)
        ranks[alpha] = values
        report["springrank"][str(alpha)] = info
        log(f"  springrank · alpha={alpha:<5} sd {values.std():.3f}  "
            f"residual {info['relative_residual']:.1e}")

    # Everything below is scored on cells that have at least one connection. The
    # 623 isolated cells are all exactly 0 by construction, and leaving them in
    # would put a spike of identical values into every correlation.
    connected = np.bincount(edges.pre.to_numpy(), minlength=n_nodes) + \
                np.bincount(edges.post.to_numpy(), minlength=n_nodes) > 0
    report["n_connected"] = int(connected.sum())
    log(f"  {connected.sum():,} cells carry at least one connection; "
        f"{n_nodes - connected.sum():,} isolated and excluded from the statistics below")

    # ---- question 1: are they the same number? ----------------------------
    default = levels["log1p"]
    report["equivalence"] = {
        "max_abs_springrank_plus_trophic": float(np.abs(ranks[0.0] + default).max())
        if 0.0 in ranks else None,
        "vs_springrank": {str(a): correlations(default[connected], r[connected])
                          for a, r in ranks.items()},
        "across_weightings": {w: correlations(default[connected], v[connected])
                              for w, v in levels.items() if w != "log1p"},
    }
    log("")
    log("  trophic (log1p) against SpringRank, Pearson / Spearman:")
    for alpha, corr in report["equivalence"]["vs_springrank"].items():
        log(f"    alpha={alpha:<5} {corr['pearson']:+.4f} / {corr['spearman']:+.4f}")
    log("  trophic (log1p) against the same solve at other weightings:")
    for weighting, corr in report["equivalence"]["across_weightings"].items():
        log(f"    {weighting:<9} {corr['pearson']:+.4f} / {corr['spearman']:+.4f}")

    # ---- question 3: does it restate degree? ------------------------------
    degree = degree_features(edges, n_nodes)
    degree_names = list(DEGREE_FEATURE_NAMES)
    report["vs_degree"] = {
        "per_feature": {name: correlations(default[connected], degree[connected, i])
                        for i, name in enumerate(degree_names)},
        "r_squared_on_degree_block": explained_by(default[connected], degree[connected]),
    }
    log("")
    log("  trophic (log1p) against each degree feature, Pearson / Spearman:")
    for name, corr in report["vs_degree"]["per_feature"].items():
        log(f"    {name:<12} {corr['pearson']:+.4f} / {corr['spearman']:+.4f}")
    log(f"  R^2 of the trophic level on all 8 degree features together: "
        f"{report['vs_degree']['r_squared_on_degree_block']:.4f}")

    # ---- question 4: does it add directions? ------------------------------
    # The trophic level and one regularised SpringRank, which is the only form in
    # which SpringRank can be a second feature rather than a sign flip.
    extra_alpha = next((a for a in args.alphas if a > 0), None)
    flow_block = default[:, None] if extra_alpha is None else \
        np.stack([default, ranks[extra_alpha]], axis=1)
    flow_names = ["trophic_log1p"] + ([] if extra_alpha is None
                                      else [f"springrank_log1p_a{extra_alpha:g}"])
    flow_block = flow_block.astype(np.float32)

    # v1.0's 24 explicitly: this script appends the flow block itself, and the
    # defaults now include it. Its arms are the 24 with and without flow.
    all_features, all_names = load_features(args.processed, args.volume, edges, n_nodes,
                                            rwse=False, flow=False)
    ranks_before_after = {
        "degree_only": effective_rank(standardise(degree)),
        "degree_plus_flow": effective_rank(standardise(np.hstack([degree, flow_block]))),
        "all_features": effective_rank(standardise(all_features)),
        "all_features_plus_flow": effective_rank(standardise(np.hstack([all_features, flow_block]))),
    }
    report["effective_rank"] = ranks_before_after
    report["flow_columns"] = flow_names
    log("")
    log(f"  effective rank, degree only          {ranks_before_after['degree_only']:>6.2f} of 8")
    log(f"  effective rank, degree + flow        {ranks_before_after['degree_plus_flow']:>6.2f} "
        f"of {8 + len(flow_names)}")
    log(f"  effective rank, all 24               {ranks_before_after['all_features']:>6.2f} of 24")
    log(f"  effective rank, all 24 + flow        {ranks_before_after['all_features_plus_flow']:>6.2f} "
        f"of {24 + len(flow_names)}")
    gain = ranks_before_after["all_features_plus_flow"] - ranks_before_after["all_features"]
    log("")
    log(f"  flow adds {gain:+.2f} directions to the 24-feature vector "
        f"(RWSE was declined at +0.65).")

    # ---- write the feature, whatever the verdict --------------------------
    frame = pd.DataFrame({"node_id": nodes.node_id.to_numpy()})
    for weighting, values in levels.items():
        frame[f"trophic_{weighting}"] = values.astype(np.float32)
    for alpha, values in ranks.items():
        if alpha > 0:
            frame[f"springrank_log1p_a{alpha:g}"] = values.astype(np.float32)
    frame["component"] = labels.astype(np.int32)
    out = args.processed / f"{args.volume}_flow.parquet"
    frame.to_parquet(out, index=False)
    log(f"wrote {out}")

    # ---- the probe, if asked ----------------------------------------------
    if args.probe:
        from wiretype.eval.probes import probe_all

        split = pd.read_parquet(args.processed / f"{args.volume}_splits.parquet") \
            .sort_values("node_id")["split"].reset_index(drop=True)
        arms = {
            # The sharpest statement of "a different kind of information": one
            # global number, against the eight local ones it is being added to.
            "flow_only": flow_block,
            # The clean comparison, where a single added feature cannot hide
            # behind 24 others.
            "degree_only": degree,
            "degree_plus_flow": np.hstack([degree, flow_block]),
            # And the arm that would actually ship, since the 24-feature vector
            # is what every trained arm consumes.
            "all_features": all_features,
            "all_features_plus_flow": np.hstack([all_features, flow_block]),
        }
        report["probes"] = {}
        for name, block in arms.items():
            log("")
            log(f"probing {name} ({block.shape[1]} dims)")
            report["probes"][name] = probe_all(block, nodes, split, seed=args.seed)
            (args.reports / f"flow_{args.volume}.json").write_text(
                json.dumps(report, indent=2, default=str))

        log("")
        log("macro-F1, linear / k-NN — super-class is the target to watch:")
        for key in ("super_class/whole_brain", "nt_train/whole_brain",
                    "nt_train/central_brain", "hemilineage/whole_brain"):
            log(f"  {key}")
            for name in arms:
                result = report["probes"][name].get(key, {})
                if "linear" in result:
                    log(f"    {name:<24} {result['linear']['macro_f1']:.4f} / "
                        f"{result['knn']['macro_f1']:.4f}")

    (args.reports / f"flow_{args.volume}.json").write_text(json.dumps(report, indent=2, default=str))
    log(f"wrote {args.reports / f'flow_{args.volume}.json'}")


if __name__ == "__main__":
    main()
