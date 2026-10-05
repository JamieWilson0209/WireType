#!/usr/bin/env python3
"""The type-blocked split scored on experimental labels.

The seen and held-out MCNS groups of `unseen_types.groups` are different populations:
experimental labels sit in the large optic types, which FAFB's type-blocked split dealt
to training, so 92% of the seen neurons are optic against 28% of the held-out ones.
Comparing their scores mixes the cost of an unseen type with a change of population.
Two comparisons hold the population fixed:

1. **Within super-class.** Seen against held-out types, one MCNS super-class at a time.
2. **On the same neurons.** The main model (random split) saw every type in training;
   scored on the held-out neurons, it gives the score those same neurons reach when
   their type was seen. The gap to the type-blocked model is the cost of the unseen
   type. The untrained arms make the same comparison with the encoder removed, so the
   gap there is what the probe alone gains from having seen the type.

Weighted F1 over the classes present, linear probe for WireType and k-NN for WireType
(untrained), mean and range over the three seeds; seed k of the type-blocked arm is
paired with seed k of the main arm.

    PYTHONPATH=src .venv/bin/python scripts/diagnostics/typeblocked_matched.py
"""
from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

sys.path.insert(0, str(Path(__file__).resolve().parent))
from regions_and_floors import COLUMNS, PROCESSED, TRANSFER, commonest, weighted  # noqa: E402
from unseen_types import groups  # noqa: E402

from wiretype.data.labels import harmonise  # noqa: E402
from wiretype.data.scope import in_scope  # noqa: E402

SEEDS = {0: "", 1: "_seed1", 2: "_seed2"}
BLOCKED = "transfer_fafb_to_mcns_checkpoint_fafb_{arm}_refined_s24k_ntbest_typeblocked{seed}_source_predictions.parquet"
MAIN = "transfer_fafb_to_mcns_checkpoint_fafb_{arm}_refined_split_random_s24k_ntbest{seed}_source_predictions.parquet"
ARMS = {"trained": ("displacement", "pred_linear"), "untrained": ("untrained", "pred_knn")}
# Super-classes with enough neurons in both groups to score; the rest are pooled.
REGIONS = ("optic", "central", "sensory", "visual_projection", "ascending")


def load(template: str, arm: str, seed: int, mcns: pd.DataFrame) -> pd.DataFrame:
    """The MCNS experimental-label rows of one predictions file, tagged by group and region."""
    rows = groups(pd.read_parquet(TRANSFER / template.format(arm=arm, seed=SEEDS[seed]), columns=COLUMNS),
                  PROCESSED)
    rows = rows.merge(mcns[["node_id", "super_class"]], on="node_id", how="left")
    rows["region"] = rows["super_class"].where(rows["super_class"].isin(REGIONS), "other")
    return rows.set_index("node_id")


def fmt(values: list[float]) -> str:
    """Mean (min-max) over seeds."""
    return f"{np.mean(values):.3f} ({min(values):.3f}-{max(values):.3f})"


def composition(fafb: pd.DataFrame, mcns: pd.DataFrame) -> None:
    """Where the split put the experimentally labelled neurons, and what that does to the
    seen and held-out MCNS groups.
    """
    split = pd.read_parquet(PROCESSED / "fafb_splits.parquet").set_index("node_id")["split"]
    fafb = fafb[in_scope(fafb, "brain_neurons")].assign(split=lambda d: d["node_id"].map(split))
    labelled = fafb["nt_known"].notna()
    print("FAFB connected brain neurons by split: neurons (types) | experimentally labelled (types, share)")
    for part in ("train", "val", "test"):
        d, e = fafb[fafb["split"] == part], fafb[(fafb["split"] == part) & labelled]
        print(f"  {part:5s} {len(d):7,d} ({d['cell_type'].nunique():,}) | {len(e):6,d} "
              f"({e['cell_type'].nunique()}, {len(e) / len(d):.0%})")
    typed = fafb[fafb["cell_type"].notna()]
    size = typed.groupby("cell_type").agg(n=("node_id", "size"), split=("split", "first"))
    big = size[size["n"] > 1000]
    print(f"  types over 1,000 neurons: {len(big)}, in training {int((big['split'] == 'train').sum())}; "
          f"optic neurons in training {(fafb.loc[fafb['super_class'] == 'optic', 'split'] == 'train').mean():.0%}; "
          f"experimentally labelled neurons in training {(fafb.loc[labelled, 'split'] == 'train').mean():.0%}")
    lab_types = typed.loc[typed["nt_known"].notna(), "cell_type"].unique()
    print(f"  types with an experimental label {len(lab_types):,} of {typed['cell_type'].nunique():,}, "
          f"holding {typed['cell_type'].isin(lab_types).sum() / len(fafb):.0%} of neurons")
    rows = mcns[in_scope(mcns, "brain_neurons") & mcns["nt_known"].notna()]
    seen = set(fafb.loc[fafb["split"] == "train", "cell_type"].dropna())
    held = set(fafb.loc[fafb["split"].isin(["val", "test"]), "cell_type"].dropna()) - seen
    for name, types in (("seen", seen), ("held-out", held)):
        d = rows[rows["cell_type"].isin(types)]
        print(f"  MCNS {name:8s} {len(d):6,d} labelled neurons, {d['cell_type'].nunique()} types, "
              f"optic {(d['super_class'] == 'optic').mean():.0%}")


def fafb_heldout(fafb: pd.DataFrame) -> None:
    """Weighted F1 on FAFB's own held-out (test) types, by FAFB super-class, both arms."""
    print("\nFAFB held-out (test) types, weighted F1: WireType linear | WireType (untrained) k-NN | rule")
    scores: dict[str, dict[str, list[float]]] = {}
    for arm, (name, col) in ARMS.items():
        for s in SEEDS:
            pred = pd.read_parquet(TRANSFER / BLOCKED.format(arm=name, seed=SEEDS[s]), columns=COLUMNS)
            pred = pred[(pred["probe_target"] == "nt_known") & (pred["where"] == "source_test")]
            pred = pred.merge(fafb[["node_id", "super_class", "cell_type"]], on="node_id", how="left")
            pred["region"] = pred["super_class"].where(pred["super_class"].isin(REGIONS), "other")
            for region in (*REGIONS, "other", "all"):
                part = pred if region == "all" else pred[pred["region"] == region]
                if len(part):
                    scores.setdefault(region, {"n": len(part), "types": part["cell_type"].nunique(),
                                               "rule": commonest(part["true"])}).setdefault(arm, []).append(
                        weighted(part["true"], part[col]))
    for region, e in scores.items():
        print(f"  {region:18s} {e['n']:6,d} ({e['types']:3d} types)  {fmt(e['trained'])}  "
              f"{fmt(e['untrained'])}  {e['rule']:.3f}")


def main() -> None:
    """Print the split's composition, FAFB's held-out scores, and the within-super-class
    and same-neuron comparisons, trained and untrained.
    """
    with contextlib.redirect_stdout(io.StringIO()):
        fafb = harmonise(pd.read_parquet(PROCESSED / "fafb_nodes.parquet"), "fafb")
        mcns = harmonise(pd.read_parquet(PROCESSED / "mcns_nodes.parquet"), "mcns")
    composition(fafb, mcns)
    fafb_heldout(fafb)

    for arm, (name, col) in ARMS.items():
        runs = {s: (load(BLOCKED, name, s, mcns), load(MAIN, name, s, mcns)) for s in SEEDS}
        blocked0 = runs[0][0]
        held_ids = blocked0.index[blocked0["group"] == "unseen"]
        seen_ids = blocked0.index[blocked0["group"] == "seen"]
        print(f"\n=== WireType ({arm}), {col}")
        print(f"{'':18s} {'seen n':>7s} {'held n':>7s} {'types':>5s}  {'blocked, seen':>21s}  "
              f"{'blocked, held-out':>21s}  {'main, held-out':>21s}  rule")
        for region in (*REGIONS, "other", "all"):
            s_ids = seen_ids if region == "all" else seen_ids[blocked0.loc[seen_ids, "region"] == region]
            h_ids = held_ids if region == "all" else held_ids[blocked0.loc[held_ids, "region"] == region]
            if not len(h_ids):
                continue
            true = blocked0.loc[h_ids, "true"]
            seen_f1, held_f1, main_f1 = [], [], []
            for s, (blocked, full) in runs.items():
                seen_f1.append(weighted(blocked.loc[s_ids, "true"], blocked.loc[s_ids, col]))
                held_f1.append(weighted(true, blocked.loc[h_ids, col]))
                main_f1.append(weighted(true, full.loc[h_ids, col]))
            print(f"{region:18s} {len(s_ids):7,d} {len(h_ids):7,d} {blocked0.loc[h_ids, 'cell_type'].nunique():5d}  "
                  f"{fmt(seen_f1):>21s}  {fmt(held_f1):>21s}  {fmt(main_f1):>21s}  {commonest(true):.3f}")

        # Per transmitter on the held-out neurons, and paired changes of outcome.
        true = blocked0.loc[held_ids, "true"]
        labels = sorted(true.unique())
        per = {"blocked": [], "main": []}
        flips = []
        for s, (blocked, full) in runs.items():
            b, m = blocked.loc[held_ids, col], full.loc[held_ids, col]
            per["blocked"].append(precision_recall_fscore_support(true, b, labels=labels, zero_division=0)[2])
            per["main"].append(precision_recall_fscore_support(true, m, labels=labels, zero_division=0)[2])
            flips.append((float(((m == true) & (b != true)).mean()), float(((m != true) & (b == true)).mean())))
        types = blocked0.loc[held_ids].groupby("true")["cell_type"].nunique()
        print(f"  held-out neurons by transmitter: F1 blocked / main (mean of seeds), neurons, types")
        for i, label in enumerate(labels):
            print(f"    {label:14s} {np.mean([f[i] for f in per['blocked']]):.2f} / "
                  f"{np.mean([f[i] for f in per['main']]):.2f}  {int((true == label).sum()):6,d}  {types[label]:4d}")
        print(f"  held-out neurons right under main but wrong under blocked {np.mean([f[0] for f in flips]):.1%}, "
              f"the reverse {np.mean([f[1] for f in flips]):.1%}")


if __name__ == "__main__":
    main()
