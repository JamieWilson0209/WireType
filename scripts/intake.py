#!/usr/bin/env python3
"""Pipeline step 1: both volumes' raw dumps into one node/edge shape.

Reads the raw dumps and writes `data/processed/{volume}_nodes.parquet` and
`{volume}_edges.parquet`, plus `intake_report.json` with every count the paper
quotes about the data, so each is reproducible by re-running this.

What it enforces:
- the six per-edge neurotransmitter probability columns are never read, since
  they are the prediction target in another form;
- the label space is seven classes, histamine included;
- neurons with no edges keep a row but are not seeds;
- an MCNS super-class missing from the hand-written map to FAFB's is a hard
  error, not a silent null.

    qsub hpc/jobs/intake.sh
    qsub -v VOLUME=fafb hpc/jobs/intake.sh
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from wiretype.data.intake import check_schema, load_fafb, load_mcns, log


def main() -> None:
    """Read the raw release of FAFB, MCNS or both; write canonical node and edge tables and an intake report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/raw"),
                        help="on the cluster this normally points at scratch, since data/raw is gitignored")
    parser.add_argument("--out", type=Path, default=Path("data/processed"))
    parser.add_argument("--reports", type=Path, default=Path("experiments/intake"),
                        help="small JSON reports; tracked in git, unlike data/processed")
    parser.add_argument("--volume", choices=("fafb", "mcns", "both"), default="both")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    reports = {}

    if args.volume in ("fafb", "both"):
        fafb = args.data / "fafb_v783"
        nodes, edges, report = load_fafb(
            fafb / "neuron_annotations_v783.tsv",
            fafb / "proofread_connections_783.feather",
        )
        check_schema(nodes, edges)
        nodes.to_parquet(args.out / "fafb_nodes.parquet", index=False)
        edges.to_parquet(args.out / "fafb_edges.parquet", index=False)
        reports["fafb"] = report
        log(f"fafb: {report['n_nodes']:,} nodes, {report['n_edges']:,} edges, "
            f"{report['n_seeds']:,} seeds ({report['n_edgeless_annotated']:,} edgeless)")

    if args.volume in ("mcns", "both"):
        mcns = args.data / "malecns_v1.0"
        nodes, edges, report = load_mcns(
            mcns / "body-annotations-v1.0-minconf-0.5.feather",
            mcns / "connectome-weights-v1.0-minconf-0.5-traced-only.feather",
            mcns / "body-neurotransmitters-v1.0.feather",
        )
        check_schema(nodes, edges)
        nodes.to_parquet(args.out / "mcns_nodes.parquet", index=False)
        edges.to_parquet(args.out / "mcns_edges.parquet", index=False)
        reports["mcns"] = report
        log(f"mcns: {report['n_nodes']:,} nodes, {report['n_edges']:,} edges, "
            f"{report['n_vnc']:,} in the VNC, {report['n_superclass_unmapped']:,} super-classes with no FAFB counterpart")

    args.reports.mkdir(parents=True, exist_ok=True)
    path = args.reports / "intake_report.json"
    path.write_text(json.dumps(reports, indent=2, default=str))
    log(f"wrote {path}")


if __name__ == "__main__":
    main()
