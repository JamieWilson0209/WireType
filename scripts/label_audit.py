#!/usr/bin/env python3
"""Audit FAFB's predicted transmitter labels against experimental ones, and against MCNS.

The paper trains on `nt_best`: FAFB's experimental label (`known_nt`) where one
exists, else the EM classifier's prediction (`top_nt`). That is only defensible if
the experimental labels are the more accurate ones where the two disagree. This
script produces the evidence, from the released annotation tables alone, with no
model involved:

1. **What changes.** Every FAFB cell where `top_nt` and `known_nt` disagree,
   grouped by the change, the cell class and the cell type, with the share of each
   type that disagrees.
2. **What the experimental labels rest on.** The sources FlyWire cites in
   `known_nt_source` for each group.
3. **What the EM classifier thought.** Its own confidence on agreeing and on
   disagreeing cells.
4. **What a second animal says.** For each disagreeing FAFB cell whose cell type
   exists in MCNS by name, whether MCNS's transmitter for that type (predicted,
   and experimental where it exists) sides with FAFB's experimental label or with
   its predicted one. MCNS was reconstructed and annotated independently of FAFB,
   but its "predicted" label (`consensusNt`) is its per-type image prediction
   replaced by the experimental label wherever the literature has one (Berg et al.
   2026), so for types with literature labels this shows consistency between the
   volumes, not independent confirmation. The Kenyon cells, which MCNS labels by
   image prediction alone, are the independent part (paper-verification.md C1).

Writes `<reports>/label_audit.json` and prints a summary.

    PYTHONPATH=src python scripts/label_audit.py \\
        --annotations data/raw/fafb_v783/neuron_annotations_v783.tsv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wiretype.data.labels import harmonise
from wiretype.log import log


def mode_or_none(values: pd.Series):
    """The most common non-missing value, or None if there is none."""
    values = values.dropna()
    return values.mode().iat[0] if len(values) else None


def main() -> None:
    """Compare FAFB's predicted and experimental transmitter labels and write
    label_audit.json.
    """
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--processed", type=Path, default=Path("data/processed"))
    p.add_argument("--annotations", type=Path, default=Path("data/raw/fafb_v783/neuron_annotations_v783.tsv"),
                   help="FlyWire's annotation table, for top_nt_conf and known_nt_source")
    p.add_argument("--reports", type=Path, default=Path("experiments/labels"))
    args = p.parse_args()

    raw = pd.read_csv(args.annotations, sep="\t", low_memory=False,
                      usecols=["root_id", "top_nt_conf", "known_nt", "known_nt_source"])
    fafb = harmonise(pd.read_parquet(args.processed / "fafb_nodes.parquet"), "fafb") \
        .merge(raw, left_on="source_id", right_on="root_id", how="left")
    both = fafb[fafb.nt_known.notna() & fafb.nt_train.notna()]
    changed = both[both.nt_known != both.nt_train]
    report = {"fafb_cells": len(fafb), "cells_with_both_labels": len(both), "cells_changed": len(changed)}
    log(f"FAFB: {len(both):,} cells carry both a predicted and an experimental label; {len(changed):,} disagree")

    # 1. what changes
    by_change = changed.groupby(["nt_train", "nt_known"]).size().sort_values(ascending=False)
    report["by_change"] = [{"predicted": a, "experimental": b, "cells": int(n)} for (a, b), n in by_change.items()]
    share = both.assign(dis=both.nt_known != both.nt_train).groupby("cell_type").dis.agg(["mean", "size"])
    typed = share[share["size"] >= 5]
    report["types"] = {
        "with_any_change": int((typed["mean"] > 0).sum()),
        "over_90pct_changed": int((typed["mean"] > 0.9).sum()),
        "under_10pct_changed": int(((typed["mean"] > 0) & (typed["mean"] < 0.1)).sum()),
        "largest": [{"cell_type": t, "cells_changed": int(n), "share_of_type": round(float(share.at[t, "mean"]), 3),
                     "type_size": int(share.at[t, "size"])}
                    for t, n in changed.cell_type.value_counts().head(15).items()],
    }
    log("  largest changes (predicted -> experimental):")
    for row in report["by_change"][:6]:
        log(f"    {row['predicted']:>13} -> {row['experimental']:<13} {row['cells']:>6,}")

    # 2. sources, and 3. classifier confidence, per headline group
    groups = {
        "Kenyon cells, dopamine -> acetylcholine":
            (changed.cell_type.fillna("").str.startswith("KC")) & (changed.nt_train == "dopamine"),
        "ORNs, serotonin -> acetylcholine":
            (changed.cell_type.fillna("").str.startswith("ORN")) & (changed.nt_train == "serotonin"),
        "L1, GABA -> glutamate": (changed.cell_type == "L1") & (changed.nt_train == "gaba"),
        "other GABA -> glutamate":
            (changed.cell_type != "L1") & (changed.nt_train == "gaba") & (changed.nt_known == "glutamate"),
    }
    report["groups"] = {}
    for name, sel in groups.items():
        g = changed[sel]
        report["groups"][name] = {
            "cells": int(len(g)), "types": int(g.cell_type.nunique()),
            "sources": {k: int(v) for k, v in g.known_nt_source.value_counts().head(4).items()},
            "median_classifier_confidence": round(float(g.top_nt_conf.median()), 3),
        }
    agree = both[both.nt_known == both.nt_train]
    report["median_classifier_confidence"] = {"agree": round(float(agree.top_nt_conf.median()), 3),
                                              "disagree": round(float(changed.top_nt_conf.median()), 3)}
    report["left_as_predicted"] = {k: int(v) for k, v in
                                   fafb[fafb.nt_known.isna()].nt_train.value_counts()
                                   .reindex(["serotonin", "dopamine", "octopamine"]).fillna(0).items()}

    # MCNS's nt_train is consensusNt, which already contains the literature label where one
    # exists: agreement is consistency except for types MCNS labels by image alone.
    # 4. a second animal
    mcns = harmonise(pd.read_parquet(args.processed / "mcns_nodes.parquet"), "mcns")
    mcns = mcns[mcns.region == "brain"]
    per_type = mcns.groupby("cell_type").agg(mcns_predicted=("nt_train", mode_or_none),
                                             mcns_experimental=("nt_known", mode_or_none))
    joined = changed.merge(per_type, left_on="cell_type", right_index=True, how="inner")
    pred = joined[joined.mcns_predicted.notna()]
    expt = joined[joined.mcns_experimental.notna()]
    report["mcns"] = {
        "changed_cells_with_type_in_mcns": int(len(pred)),
        "mcns_predicted_agrees_with_fafb_experimental": round(float((pred.mcns_predicted == pred.nt_known).mean()), 3),
        "mcns_predicted_agrees_with_fafb_predicted": round(float((pred.mcns_predicted == pred.nt_train).mean()), 3),
        "changed_cells_with_mcns_experimental": int(len(expt)),
        "mcns_experimental_agrees_with_fafb_experimental": round(float((expt.mcns_experimental == expt.nt_known).mean()), 3),
        "mcns_experimental_agrees_with_fafb_predicted": round(float((expt.mcns_experimental == expt.nt_train).mean()), 3),
        "per_group": {},
    }
    for name, sel in groups.items():
        g = changed[sel].merge(per_type, left_on="cell_type", right_index=True, how="inner")
        report["mcns"]["per_group"][name] = {
            "mcns_cells_of_these_types": int(mcns.cell_type.isin(changed[sel].cell_type.unique()).sum()),
            "mcns_predicted": {k: int(v) for k, v in
                               mcns[mcns.cell_type.isin(changed[sel].cell_type.unique())].nt_train.value_counts().head(3).items()},
            "mcns_experimental": {str(k): int(v) for k, v in
                                  mcns[mcns.cell_type.isin(changed[sel].cell_type.unique())].nt_known
                                  .value_counts(dropna=False).head(3).items()},
        }
    m = report["mcns"]
    log(f"  MCNS, same-named types: its predicted label sides with FAFB's experimental label for "
        f"{m['mcns_predicted_agrees_with_fafb_experimental']:.1%} of {m['changed_cells_with_type_in_mcns']:,} "
        f"changed cells, with FAFB's predicted label for {m['mcns_predicted_agrees_with_fafb_predicted']:.1%}")
    log(f"  MCNS experimental labels, where they exist ({m['changed_cells_with_mcns_experimental']:,} cells): "
        f"{m['mcns_experimental_agrees_with_fafb_experimental']:.1%} side with FAFB's experimental label")

    args.reports.mkdir(parents=True, exist_ok=True)
    out = args.reports / "label_audit.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    log(f"wrote {out}")


if __name__ == "__main__":
    main()
