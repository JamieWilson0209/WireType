#!/usr/bin/env python3
"""Where the zero-shot errors are: slice per-cell predictions every way the metadata allows.

Reads the `_predictions.parquet` that `transfer.py --dump-predictions` writes
(one row per scored cell: the source's held-out test cells as `source_test`, every
labelled target cell as `target`) and prints markdown tables. There is no model
here and no whole-graph work, so it runs locally.

    PYTHONPATH=src python scripts/breakdown.py \\
        experiments/transfer/<report>_predictions.parquet \\
        --floor experiments/transfer/<untrained report>_predictions.parquet \\
        --out breakdown_<tag>.md

For each probe target it reports:
- per class: support, precision, recall, F1, source recall beside target recall;
- the confusion matrix, row-normalised (the true class is the row);
- macro-F1, support-weighted F1 and accuracy per slice: region, trace status,
  super-class, whether the MCNS cell type exists in FAFB, whether the label is
  confident (`nt_known`), degree quintile, top-k tokens held, side;
- per cell type: how many types are mostly wrong, and which cost the most cells;
- probabilities: mean probability given to the true class, and the top
  probability on wrong calls (a confident wrong call is a label or transfer
  problem, an unsure one is a representation problem), per class, slice and type;
- triage: the share of cells kept at a given precision when only confident
  predictions are accepted, overall and per class, plus calibration error.

`--floor` adds the untrained encoder's figures beside each slice, because every
arm is read against its own floor.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support

SLICES = ("region", "status", "super_class", "type_in_source", "confident_label",
          "degree_quintile", "tokens", "side")
MIN_SLICE = 30  # smaller slices are printed as counts only


def attach_metadata(pred: pd.DataFrame, processed: Path, source: str, target: str) -> pd.DataFrame:
    """Join each row to its own volume's node table, and derive the slice columns."""
    source_types = set(pd.read_parquet(processed / f"{source}_nodes.parquet")["cell_type"].dropna())
    frames = []
    for where, volume in (("source_test", source), ("target", target)):
        part = pred[pred["where"] == where]
        if part.empty:
            continue
        nodes = pd.read_parquet(processed / f"{volume}_nodes.parquet")
        topk = np.load(processed / f"{volume}_topk64.npz")
        nodes["tokens"] = pd.cut(topk["length"][nodes["node_id"].to_numpy()],
                                 [-1, 7, 15, 31, 63, 64], labels=["1-7", "8-15", "16-31", "32-63", "64"])
        nodes["degree"] = nodes["n_in"] + nodes["n_out"]
        nodes["type_in_source"] = np.where(nodes["cell_type"].isna(), "no type",
                                           np.where(nodes["cell_type"].isin(source_types), "seen", "unseen"))
        nodes["confident_label"] = np.where(nodes["nt_known"].notna(), "nt_known", "predicted only")
        cols = ["node_id", "region", "status", "super_class", "cell_type", "side", "tokens",
                "degree", "type_in_source", "confident_label"]
        part = part.merge(nodes[cols], on="node_id", how="left")
        # Quintiles of the scored cells, per volume and probe target: MCNS has
        # thousands of zero-degree unlabelled bodies that would collapse the edges.
        part["degree_quintile"] = part.groupby("probe_target")["degree"].transform(
            lambda d: pd.qcut(d.rank(method="first"), 5, labels=False) + 1).map(
            {1: "q1 lowest", 2: "q2", 3: "q3", 4: "q4", 5: "q5 highest"})
        frames.append(part)
    out = pd.concat(frames, ignore_index=True)
    out["status"] = out["status"].astype(str)
    return out


def md(table: pd.DataFrame, index: bool = True) -> str:
    """A pipe table without the optional `tabulate` dependency."""
    if index:
        table = table.reset_index()
    fmt = lambda v: f"{v:.3f}" if isinstance(v, (float, np.floating)) else str(v)
    head = "| " + " | ".join(map(str, table.columns)) + " |"
    rule = "|" + "---|" * len(table.columns)
    body = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in table.itertuples(index=False)]
    return "\n".join([head, rule, *body])


def macro(frame: pd.DataFrame, col: str = "pred_linear") -> float:
    """Macro-F1 of one prediction column against the true labels."""
    labels = sorted(frame["true"].unique())
    return f1_score(frame["true"], frame[col], labels=labels, average="macro", zero_division=0)


def weighted(frame: pd.DataFrame, col: str = "pred_linear") -> float:
    """Per-class F1 weighted by class share: the F1 expected for a random cell."""
    labels = sorted(frame["true"].unique())
    return f1_score(frame["true"], frame[col], labels=labels, average="weighted", zero_division=0)


def per_class(frame: pd.DataFrame) -> pd.DataFrame:
    """Per transmitter: precision, recall and F1 on the target, k-NN recall, and the
    source's held-out scores.
    """
    tgt = frame[frame["where"] == "target"]
    src = frame[frame["where"] == "source_test"]
    classes = sorted(set(frame["true"]))
    p, r, f, n = precision_recall_fscore_support(tgt["true"], tgt["pred_linear"], labels=classes,
                                                 zero_division=0)
    _, rs, fs, ns = precision_recall_fscore_support(src["true"], src["pred_linear"], labels=classes,
                                                    zero_division=0)
    _, rk, _, _ = precision_recall_fscore_support(tgt["true"], tgt["pred_knn"], labels=classes,
                                                  zero_division=0)
    right = tgt["true"] == tgt["pred_linear"]
    p_true = tgt.groupby("true")["p_true"].mean().reindex(classes) if "p_true" in tgt else np.nan
    p_wrong = tgt[~right].groupby("true")["p_linear"].mean().reindex(classes) if "p_true" in tgt else np.nan
    return pd.DataFrame({"n target": n, "precision": p, "recall": r, "F1": f, "k-NN recall": rk,
                         "mean p(true class)": p_true, "mean top p when wrong": p_wrong,
                         "n source": ns, "source recall": rs, "source F1": fs,
                         "F1 lost in transfer": fs - f}, index=classes).round(3)


def confusion(frame: pd.DataFrame) -> pd.DataFrame:
    """Row-normalised confusion matrix of the linear probe on the target (rows: true
    label).
    """
    tgt = frame[frame["where"] == "target"]
    table = pd.crosstab(tgt["true"], tgt["pred_linear"], normalize="index")
    return table.reindex(columns=sorted(set(tgt["true"]) | set(table.columns)), fill_value=0).round(3)


def slices(frame: pd.DataFrame, floor: pd.DataFrame | None) -> dict[str, pd.DataFrame]:
    """Scores within each value of every slicing column (for example trace status),
    beside the untrained arm's.
    """
    tgt = frame[frame["where"] == "target"]
    ftgt = None if floor is None else floor[floor["where"] == "target"]
    out = {}
    for col in SLICES:
        rows = []
        for value, part in tgt.groupby(col, observed=True, dropna=False):
            row = {col: value, "n": len(part), "share of errors": (part["true"] != part["pred_linear"]).sum()}
            if len(part) >= MIN_SLICE:
                row |= {"macro-F1": macro(part), "weighted F1": weighted(part), "accuracy": (part["true"] == part["pred_linear"]).mean(),
                        "k-NN macro-F1": macro(part, "pred_knn"), "mean top p": part["p_linear"].mean()}
                if "p_true" in part:
                    row["mean p(true class)"] = part["p_true"].mean()
                if ftgt is not None:
                    fpart = ftgt[ftgt["node_id"].isin(part["node_id"])]
                    row["floor macro-F1"] = macro(fpart) if len(fpart) else np.nan
            rows.append(row)
        table = pd.DataFrame(rows).sort_values("n", ascending=False)
        table["share of errors"] = table["share of errors"] / max(1, (tgt["true"] != tgt["pred_linear"]).sum())
        out[col] = table.head(12).round(3)
    return out


def by_type(frame: pd.DataFrame, min_cells: int = 10) -> tuple[dict, pd.DataFrame]:
    """Per cell type: neurons, accuracy, errors and the modal true and predicted labels,
    with a summary of failing types.
    """
    tgt = frame[(frame["where"] == "target") & frame["cell_type"].notna()]
    grouped = tgt.assign(right=tgt["true"] == tgt["pred_linear"]).groupby("cell_type")
    table = pd.DataFrame({"cells": grouped.size(), "accuracy": grouped["right"].mean(),
                          "wrong": grouped.size() - grouped["right"].sum(),
                          "mean top p": grouped["p_linear"].mean(),
                          "true": grouped["true"].agg(lambda s: s.mode().iat[0]),
                          "predicted as": grouped["pred_linear"].agg(lambda s: s.mode().iat[0]),
                          "seen": grouped["type_in_source"].first()})
    big = table[table["cells"] >= min_cells]
    summary = {"types with >= %d cells" % min_cells: len(big),
               "accuracy < 0.5": int((big["accuracy"] < 0.5).sum()),
               "accuracy == 0": int((big["accuracy"] == 0).sum()),
               "cells in types < 0.5": int(big.loc[big["accuracy"] < 0.5, "cells"].sum()),
               "share of all errors in those types": round(
                   big.loc[big["accuracy"] < 0.5, "wrong"].sum() / max(1, table["wrong"].sum()), 3)}
    return summary, table.sort_values("wrong", ascending=False).head(20).round(3)


def triage(frame: pd.DataFrame, targets=(0.80, 0.90, 0.95)) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Accept a prediction only when the probe's top probability clears a threshold.

    Coverage is the share of cells accepted; precision is accuracy among them. The
    threshold for each precision target is the lowest that reaches it.
    """
    tgt = frame[frame["where"] == "target"].sort_values("p_linear", ascending=False)
    right = (tgt["true"] == tgt["pred_linear"]).to_numpy()
    running = np.cumsum(right) / np.arange(1, len(right) + 1)
    overall = []
    for goal in targets:
        ok = np.flatnonzero(running >= goal)
        cover = (ok.max() + 1) / len(right) if len(ok) else 0.0
        overall.append({"precision target": goal, "coverage": round(cover, 3),
                        "threshold": round(float(tgt["p_linear"].iat[ok.max()]), 3) if len(ok) else np.nan})
    rows = []
    for c, part in tgt.groupby("pred_linear"):
        part = part.sort_values("p_linear", ascending=False)
        hit = (part["true"] == c).to_numpy()
        run = np.cumsum(hit) / np.arange(1, len(hit) + 1)
        ok = np.flatnonzero(run >= 0.95)
        n_true = (tgt["true"] == c).sum()
        rows.append({"class": c, "predicted": len(part), "precision": round(hit.mean(), 3),
                     "cells kept at 95% precision": int(ok.max() + 1) if len(ok) else 0,
                     "as share of true class": round((ok.max() + 1) / n_true, 3) if len(ok) and n_true else 0.0})
    bins = np.clip((tgt["p_linear"].to_numpy() * 10).astype(int), 0, 9)
    ece = sum(abs(right[bins == b].mean() - tgt["p_linear"].to_numpy()[bins == b].mean()) * (bins == b).mean()
              for b in range(10) if (bins == b).any())
    return pd.DataFrame(overall), pd.DataFrame(rows), round(float(ece), 3)


def main() -> None:
    """Write the markdown breakdown of one predictions file, optionally against the
    untrained arm's.
    """
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("predictions", type=Path)
    p.add_argument("--floor", type=Path, help="the untrained arm's predictions parquet")
    p.add_argument("--processed", type=Path, default=Path("data/processed"))
    p.add_argument("--source", default="fafb")
    p.add_argument("--target", default="mcns")
    p.add_argument("--out", type=Path, help="write the markdown here as well as to stdout")
    args = p.parse_args()

    pred = attach_metadata(pd.read_parquet(args.predictions), args.processed, args.source, args.target)
    floor = None if args.floor is None else \
        attach_metadata(pd.read_parquet(args.floor), args.processed, args.source, args.target)

    lines = [f"# Breakdown: `{args.predictions.name}`", ""]
    for target, frame in pred.groupby("probe_target"):
        ffloor = None if floor is None else floor[floor["probe_target"] == target]
        tgt = frame[frame["where"] == "target"]
        lines += [f"## {target}", "",
                  f"Zero-shot macro-F1 {macro(tgt):.4f} (k-NN {macro(tgt, 'pred_knn'):.4f}), weighted F1 "
                  f"{weighted(tgt):.4f} (k-NN {weighted(tgt, 'pred_knn'):.4f}), accuracy "
                  f"{(tgt['true'] == tgt['pred_linear']).mean():.4f}, n={len(tgt):,}; source in-volume "
                  f"{macro(frame[frame['where'] == 'source_test']):.4f}"
                  + ("" if ffloor is None else f"; untrained floor {macro(ffloor[ffloor['where'] == 'target']):.4f}"), ""]
        if tgt["true"].nunique() <= 12:
            lines += ["### Per class", "", md(per_class(frame)), "",
                      "### Confusion (row = true class)", "", md(confusion(frame)), ""]
        for col, table in slices(frame, ffloor).items():
            lines += [f"### By {col}", "", md(table, index=False), ""]
        summary, worst = by_type(frame)
        lines += ["### By cell type", "", "  ".join(f"{k}: **{v}**" for k, v in summary.items()), "",
                  md(worst), ""]
        overall, per, ece = triage(frame)
        lines += ["### Triage: accept only confident predictions", "", md(overall, index=False), "",
                  md(per, index=False), "", f"Expected calibration error: {ece}", ""]
    text = "\n".join(lines)
    print(text)
    if args.out:
        args.out.write_text(text)


if __name__ == "__main__":
    main()
