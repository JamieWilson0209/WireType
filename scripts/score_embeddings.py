#!/usr/bin/env python3
"""Score any pair of frozen embeddings, source and target, by the zero-shot protocol.

**Every baseline goes through this, so every number in the comparison table is
produced by the same code** as the encoder's own (`wiretype.eval.transfer.zero_shot`):

1. The probe (`probes_fit_apply`, linear and k-NN) is fitted on FAFB's
   `split_random` training split, with a FAFB-fitted scaler.
2. It is applied to MCNS brain cells unchanged. No MCNS label is seen.
3. The same six targets are scored in the same order, experimental labels first,
   plus transmitter by reconstruction status.
4. `--dump-predictions` writes the per-cell rows `scripts/breakdown.py` reads.

Embeddings are `(n_cells, d)` arrays in `node_id` order, the order
`scripts/transfer.py --save-embeddings` writes. The row count is checked against each
volume's nodes before anything is fitted.

    PYTHONPATH=src python scripts/score_embeddings.py --name sage_ntbest_s0 \\
        --source-emb experiments/baselines/sage_ntbest_s0_embeddings_fafb.npy \\
        --target-emb experiments/baselines/sage_ntbest_s0_embeddings_mcns.npy

Reproducing the encoder's row from its saved embeddings is the check on this script:
it should match `scripts/transfer.py`'s zero-shot numbers up to float16 rounding.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wiretype.data.scope import SCOPES, restrict_split
from wiretype.eval.probes import probe_all
from wiretype.eval.transfer import PLAN, TARGETS, load_volume, zero_shot
from wiretype.log import log


def main() -> None:
    """Score one pair of saved embeddings by the zero-shot protocol and write the
    report.
    """
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True, help="report stem, e.g. sage_ntbest_s0")
    p.add_argument("--source-emb", type=Path, required=True)
    p.add_argument("--target-emb", type=Path, required=True)
    p.add_argument("--processed", type=Path, default=Path("data/processed"))
    p.add_argument("--reports", type=Path, default=Path("experiments/baselines"))
    p.add_argument("--source", default="fafb")
    p.add_argument("--target", default="mcns")
    p.add_argument("--scope", choices=SCOPES, default="brain_neurons",
                   help="source cells that take part in the probe's fit (wiretype.data.scope); "
                        "brain_neurons is the paper's setting")
    p.add_argument("--split-column", default="split_random")
    p.add_argument("--targets", default=",".join(t for t, _ in PLAN),
                   help="comma-separated subset of the plan, for a quick check")
    p.add_argument("--dump-predictions", action="store_true")
    p.add_argument("--no-status", action="store_true", help="skip the by-status table")
    p.add_argument("--in-volume", action="store_true",
                   help="also refit the probe on the target's own split (a diagnostic, ~1 h)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    # --- 1. Both volumes' neurons and splits; the source's out-of-scope neurons are excluded
    args.reports.mkdir(parents=True, exist_ok=True)
    src_nodes, src_split = load_volume(args.processed, args.source, args.split_column)
    src_split = restrict_split(src_nodes, src_split, args.scope)
    tgt_nodes, tgt_split = load_volume(args.processed, args.target, args.split_column)
    # --- 2. The embeddings, checked against the node tables
    src_emb = np.load(args.source_emb).astype(np.float32)
    tgt_emb = np.load(args.target_emb).astype(np.float32)
    for vol, emb, nodes in ((args.source, src_emb, src_nodes), (args.target, tgt_emb, tgt_nodes)):
        if emb.shape[0] != len(nodes):
            raise SystemExit(f"{vol} embeddings have {emb.shape[0]:,} rows but {len(nodes):,} cells")
    if src_emb.shape[1] != tgt_emb.shape[1]:
        raise SystemExit(f"widths differ: {src_emb.shape[1]} vs {tgt_emb.shape[1]}")
    log(f"{args.name}: {args.source} {src_emb.shape} -> {args.target} {tgt_emb.shape} · "
        f"split {args.split_column}")

    # --- 3. The zero-shot probes (the same code as scripts/transfer.py), and optionally in-volume
    wanted = {t.strip() for t in args.targets.split(",") if t.strip()}
    report, predictions = zero_shot(src_emb, src_nodes, src_split, tgt_emb, tgt_nodes, seed=args.seed,
                                    dump=args.dump_predictions,
                                    plan=[(t, tier) for t, tier in PLAN if t in wanted],
                                    by_status=not args.no_status)
    report = {"name": args.name, "source_emb": str(args.source_emb), "target_emb": str(args.target_emb),
              "source": args.source, "target": args.target, "split_column": args.split_column,
              "width": int(src_emb.shape[1]), **report}
    if args.in_volume:
        log(f"in-volume: probe refitted on {args.target}'s own training split.")
        report["in_volume"] = probe_all(tgt_emb, tgt_nodes, tgt_split, targets=TARGETS, seed=args.seed)
        for key, res in report["in_volume"].items():
            if "linear" in res:
                log(f"  {key:<26} {res['linear']['macro_f1']:.4f} / {res['knn']['macro_f1']:.4f}")

    # --- 4. Write the report and, if asked, the per-neuron predictions
    out = args.reports / f"{args.name}.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    if predictions is not None:
        predictions.to_parquet(out.with_name(out.stem + "_predictions.parquet"))
        log(f"wrote {out.stem}_predictions.parquet")
    log(f"wrote {out}")


if __name__ == "__main__":
    main()
