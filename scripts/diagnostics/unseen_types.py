#!/usr/bin/env python3
"""Transmitter on cell types the encoder never saw.

Reads the `_predictions.parquet` of a transfer whose checkpoint was trained on FAFB's
type-blocked split (`split`), and scores MCNS brain cells with experimental labels in
three groups, by where their named cell type falls in that split:
- `seen`: the type is in FAFB's training types;
- `unseen`: the type is in FAFB's validation or test types, so neither the encoder
  nor the read-out saw any cell of it;
- `absent`: the type does not occur in FAFB.

It also reports FAFB's own held-out (unseen-type) cells from the same file. Compare
`unseen` with `seen`: if transmitter transfers only through type matching, `unseen`
falls to near the floor; if wiring carries transmitter beyond type, it stays well
above it. Run it on the untrained arm's predictions for the floor.

    PYTHONPATH=src .venv/bin/python scripts/diagnostics/unseen_types.py \\
        experiments/transfer/<typeblocked report>_predictions.parquet \\
        [--floor experiments/transfer/<typeblocked untrained>_predictions.parquet]
"""
from __future__ import annotations

import argparse
import contextlib
import io
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score

from wiretype.data.labels import harmonise


def groups(pred: pd.DataFrame, processed: Path) -> pd.DataFrame:
    """The confident-label MCNS rows, each tagged seen, unseen or absent by where its
    type falls in FAFB's type-blocked split.
    """
    with contextlib.redirect_stdout(io.StringIO()):
        fafb = harmonise(pd.read_parquet(processed / "fafb_nodes.parquet"), "fafb")
        mcns = harmonise(pd.read_parquet(processed / "mcns_nodes.parquet"), "mcns")
    split = pd.read_parquet(processed / "fafb_splits.parquet")[["node_id", "split"]]
    fafb = fafb.merge(split, on="node_id")
    seen = set(fafb.loc[fafb["split"] == "train", "cell_type"].dropna())
    held = set(fafb.loc[fafb["split"].isin(["val", "test"]), "cell_type"].dropna()) - seen

    rows = pred[(pred["probe_target"] == "nt_known") & (pred["where"] == "target")]
    rows = rows.merge(mcns[["node_id", "cell_type"]], on="node_id", how="left")
    rows["group"] = rows["cell_type"].map(
        lambda t: "seen" if t in seen else "unseen" if t in held else "absent")
    return rows


def score(frame: pd.DataFrame, col: str) -> dict:
    """Neurons, types, macro-F1 and accuracy of one prediction column."""
    labels = sorted(frame["true"].unique())
    return {"cells": len(frame), "types": frame["cell_type"].nunique(),
            "macro_f1": round(f1_score(frame["true"], frame[col], labels=labels,
                                       average="macro", zero_division=0), 3),
            "accuracy": round(float((frame["true"] == frame[col]).mean()), 3)}


def report(path: Path, processed: Path, title: str) -> None:
    """Print the scores and per-class recall of each group for one predictions file."""
    pred = pd.read_parquet(path, columns=["probe_target", "node_id", "where", "true",
                                          "pred_linear", "pred_knn"])
    rows = groups(pred, processed)
    print(f"\n{title}: {path.name}")
    for group in ("seen", "unseen", "absent"):
        part = rows[rows["group"] == group]
        if len(part):
            print(f"  MCNS {group:7s} linear {score(part, 'pred_linear')}  k-NN macro "
                  f"{score(part, 'pred_knn')['macro_f1']}")
            hit = (part["true"] == part["pred_linear"]).groupby(part["true"]).mean().round(3)
            print(f"          recall by class {hit.to_dict()}")
    src = pred[(pred["probe_target"] == "nt_known") & (pred["where"] == "source_test")]
    if len(src):
        labels = sorted(src["true"].unique())
        print(f"  FAFB held-out cells of the checkpoint's split, linear macro-F1 "
              f"{f1_score(src['true'], src['pred_linear'], labels=labels, average='macro', zero_division=0):.3f}"
              f" over {len(src):,} cells")


def main() -> None:
    """Score the trained arm, and optionally the untrained floor, on seen, unseen and
    absent types.
    """
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("predictions", type=Path)
    p.add_argument("--floor", type=Path, help="the untrained arm's predictions parquet")
    p.add_argument("--processed", type=Path, default=Path("data/processed"))
    args = p.parse_args()
    report(args.predictions, args.processed, "trained")
    if args.floor:
        report(args.floor, args.processed, "floor (untrained)")


if __name__ == "__main__":
    main()
