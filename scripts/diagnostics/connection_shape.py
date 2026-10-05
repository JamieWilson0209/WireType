#!/usr/bin/env python3
"""Share of single-synapse connections by super-class, for the Figure 1-figure supplement 1 legend.

In FAFB the legend's 36% and 67% are the median shares of the INPUTS of motor
and sensory neurons (outputs 0.85 and 0.43). This recomputes the `fracW1` connection-weight
feature, unstandardised, per neuron and reports its median and mean by
super-class, for outputs and inputs, in each volume. Needs the edge tables, so it
runs on the cluster (CPU, a few minutes).

    PYTHONPATH=src python scripts/diagnostics/connection_shape.py
    # writes experiments/features/connection_shape.json and prints the table
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from wiretype.data.features import connection_feature_names, connection_features
from wiretype.eval.transfer import load_volume


def shares(processed: Path, volume: str) -> pd.DataFrame:
    """Each neuron's single-synapse shares of outputs and inputs, with its super-class
    (MCNS: brain only).
    """
    nodes, _ = load_volume(processed, volume)
    edges = pd.read_parquet(processed / f"{volume}_edges.parquet")
    block = connection_features(edges, len(nodes))
    names = connection_feature_names()
    frame = pd.DataFrame(block, columns=names)[["fracW1_out", "fracW1_in"]]
    frame["super_class"] = nodes["super_class"].to_numpy()
    frame["n_out"], frame["n_in"] = nodes["n_out"].to_numpy(), nodes["n_in"].to_numpy()
    if volume == "mcns":
        frame = frame[nodes["region"].to_numpy() == "brain"]
    return frame


def main() -> None:
    """Write the single-synapse shares by super-class for both volumes to
    connection_shape.json.
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--processed", type=Path, default=Path("data/processed"))
    ap.add_argument("--out", type=Path, default=Path("experiments/features/connection_shape.json"))
    args = ap.parse_args()
    report = {}
    for volume in ("fafb", "mcns"):
        frame = shares(args.processed, volume)
        rows = {}
        for side in ("out", "in"):
            has = frame[f"n_{side}"] > 0  # a neuron with no partners on this side has no share
            g = frame[has].groupby("super_class")[f"fracW1_{side}"]
            rows[side] = {sc: {"neurons": int(n), "median": float(m), "mean": float(a)}
                          for sc, n, m, a in zip(g.size().index, g.size(), g.median(), g.mean())}
        report[volume] = rows
        print(f"\n{volume}: share of single-synapse connections, median (mean), by super-class")
        print(f"  {'super-class':20s} {'outputs':>16s} {'inputs':>16s}")
        for sc in sorted(rows["out"], key=lambda s: -rows["out"][s]["neurons"]):
            o, i = rows["out"][sc], rows["in"].get(sc, {"median": np.nan, "mean": np.nan})
            print(f"  {sc:20s} {o['median']:7.2f} ({o['mean']:.2f}) {i['median']:9.2f} ({i['mean']:.2f})"
                  f"   n={o['neurons']:,}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
