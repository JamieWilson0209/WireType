#!/usr/bin/env python3
"""Pipeline step 2: the train/val/test splits, and how much they leak.

Reads `data/processed/{volume}_nodes.parquet` and writes `{volume}_splits.parquet`
with two columns:
- `split_random`: per cell. The paper's protocol.
- `split`: whole cell types held out together, a stress test for unseen types.

It also writes `splits_report.json` with the leak audit: the share of test cells
that have a same-type sibling in training, measured with no model involved.
Under `split_random` it is high, which is why in-volume numbers are diagnostics
and only zero-shot transfer is a result.

    qsub hpc/jobs/splits.sh
    qsub -v SEED=1 hpc/jobs/splits.sh

Running it overwrites `{volume}_splits.parquet`. The paper's FAFB splits (released as
`fafb_splits`) were drawn with an earlier `--stratify-on` default, and this script does not
redraw them (`wiretype.data.splits`); to reproduce the paper, use the released split.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from wiretype.log import log
from wiretype.data.labels import harmonise
from wiretype.data.splits import (
    assign_splits,
    label_coverage,
    leak_audit,
    random_splits,
)


def main() -> None:
    """Build and write the random and type-blocked splits for one volume, with their
    leak audit.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed", type=Path, default=Path("data/processed"))
    parser.add_argument("--reports", type=Path, default=Path("experiments/intake"),
                        help="small JSON reports; tracked in git, unlike data/processed")
    parser.add_argument("--volume", choices=("fafb", "mcns", "both"), default="fafb")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stratify-on", default="nt_best,super_class",
                        help="comma-separated columns, balanced jointly")
    args = parser.parse_args()

    volumes = ("fafb", "mcns") if args.volume == "both" else (args.volume,)
    report: dict = {}

    for volume in volumes:
        path = args.processed / f"{volume}_nodes.parquet"
        if not path.exists():
            raise SystemExit(f"{path} not found — run intake first (qsub hpc/jobs/intake.sh)")
        # `nt_best` to stratify on, and MCNS's side in FAFB's spelling for the
        # twin audit. Only the audit and the assignment see this; the node table
        # on disk is unchanged.
        nodes = harmonise(pd.read_parquet(path), volume)
        log(f"{volume}: {len(nodes):,} nodes")

        blocked = assign_splits(nodes, stratify_on=tuple(args.stratify_on.split(",")), seed=args.seed)
        naive = random_splits(nodes, seed=args.seed)

        frame = pd.DataFrame({"node_id": nodes.node_id, "split": blocked, "split_random": naive})
        frame.to_parquet(args.processed / f"{volume}_splits.parquet", index=False)

        report[volume] = {
            "seed": args.seed,
            "stratify_on": args.stratify_on,
            "blocked": leak_audit(nodes, blocked),
            "random": leak_audit(nodes, naive),
            "coverage": label_coverage(nodes, blocked),
            "class_balance": {
                target: pd.crosstab(nodes[target], blocked, normalize="index").round(4).to_dict()
                for target in ("nt_best", "super_class")
                if target in nodes
            },
        }
        b, r = report[volume]["blocked"], report[volume]["random"]
        log(f"  leak: same-type sibling in train — blocked {b['test_with_cell_type_sibling_in_train']:.1%}, "
            f"random {r['test_with_cell_type_sibling_in_train']:.1%}")
        log(f"  leak: bilateral twin in train   — blocked {b['test_twins_with_partner_in_train']:.1%}, "
            f"random {r['test_twins_with_partner_in_train']:.1%}")

    args.reports.mkdir(parents=True, exist_ok=True)
    out = args.reports / "splits_report.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    log(f"wrote {out}")


if __name__ == "__main__":
    main()
