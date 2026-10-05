#!/usr/bin/env python3
"""Build the released prediction tables from the release run.

Reads the seed-0 forward and reverse release runs in `experiments/release/`
(`ONLY=release bash hpc/submit_paper.sh`, checked by `tests/check_release.py`) and
writes, for each target volume, one row per connected brain neuron:

    wiretype_predictions_mcns.{parquet,csv.gz}   MCNS neurons, FAFB-trained WireType
    wiretype_predictions_fafb.{parquet,csv.gz}   FAFB neurons, MCNS-trained WireType
    fafb_splits.{parquet,csv.gz}                 the FAFB splits every FAFB-trained model used
    README_predictions.md                        what every column means

with the calls of both released probes (fitted on experimental labels, `explabel_`;
fitted on training labels, `trainlabel_`), each by its linear and k-NN readout.

    PYTHONPATH=src .venv/bin/python scripts/release.py
"""
from __future__ import annotations

import argparse
import contextlib
import io
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wiretype.data.labels import harmonise   # noqa: E402
from wiretype.data.scope import in_scope     # noqa: E402

RUNS = {"mcns": ("fafb", "transfer_fafb_to_mcns_checkpoint_fafb_displacement_refined_split_random_s24k_ntbest_source"),
        "fafb": ("mcns", "transfer_mcns_to_fafb_checkpoint_mcns_displacement_refined_split_random_s24k_ntbest_brain_source")}
PROBES = {"nt_known": "explabel", "nt_best": "trainlabel"}
ID_COLUMN = {"mcns": "mcns_body_id", "fafb": "flywire_root_id_v783"}

README = """# WireType transmitter predictions

One table per volume, one row per brain neuron with at least one connection:

- `wiretype_predictions_mcns`: every connected MCNS brain neuron ({n_mcns:,}), predicted by
  WireType trained on FAFB (FlyWire v783) and applied unchanged to MCNS v1.0.
- `wiretype_predictions_fafb`: every connected FAFB brain neuron ({n_fafb:,}), predicted by
  WireType trained on MCNS brain neurons and applied unchanged to FAFB.

Neither table uses a label from the volume it predicts. Neurons without connections are
absent (the model has no wiring to read), as is the MCNS nerve cord (FAFB has none).
Both models are seed 0; the transmitter scores in the paper come from these runs.

## Columns

| column | meaning |
|---|---|
| `{id_mcns}` / `{id_fafb}` | the neuron's ID in its release |
| `embedding_row` | its row in the released embeddings of that volume |
| `cell_type`, `super_class` | the volume's own annotations |
| `experimental_label` | the experimentally established transmitter (FAFB: `known_nt` parsed to one fast transmitter; MCNS: `ground_truth`), where one exists |
| `predicted_label` | the volume's image-based prediction as used in the paper (FAFB: `top_nt`, histamine where the literature names it; MCNS: `consensusNt`) |
| `scored` | the neuron has an experimental label, so it is in the set the paper scores |

Then, for each of two probes fitted on the source volume's embeddings:

- `explabel_`: fitted on the source's neurons with an experimental label. Every score in
  the paper comes from this probe.
- `trainlabel_`: fitted on the source's training labels (the experimental label where one
  exists, the predicted label otherwise), so it has also seen types without an
  experimental label.

and each of two readouts:

| column | meaning |
|---|---|
| `<probe>_linear_call` | the class of highest probability under the logistic-regression readout |
| `<probe>_linear_p` | that probability |
| `<probe>_linear_p_<class>` | the probability of each class |
| `<probe>_knn_call` | the majority class of the nearest source neurons (k in the probe file) |
| `<probe>_knn_share` | the share of those neighbours voting for the call |
| `<probe>_knn_share_<class>` | the share voting for each class |

The fitted probes are in the `_probes.npz` files, with how to re-apply them in the
matching `.json`; `wiretype.eval.release.apply_probe` in the code does it.

## The FAFB splits

`fafb_splits` gives every FAFB neuron ({n_split:,}) its set in the two splits the paper uses:

| column | meaning |
|---|---|
| `{id_fafb}` | the neuron's ID in FlyWire v783 |
| `split_random` | training, validation or test, drawn per neuron at random (70:15:15, seed 0); every FAFB-trained result except the type-blocked analysis |
| `split_type_blocked` | the same three sets, assigned by whole cell type, largest types first, balanced on the predicted label; the type-blocked analysis |

This file, not the current default of `scripts/splits.py`, defines the type-blocked split:
the code now balances training label and super-class jointly, which deals types
differently, and the node table has changed since the split was drawn.
"""


def table(volume: str, release: Path, nodes: pd.DataFrame) -> pd.DataFrame:
    """One row per connected brain neuron of `volume`, with both probes' calls."""
    _, stem = RUNS[volume]
    pred = pd.read_parquet(release / f"{stem}_predictions.parquet")
    pred = pred[pred["where"].isin(["target", "target_unlabelled"])]
    connected = nodes[in_scope(nodes, "brain_neurons")]
    out = pd.DataFrame({ID_COLUMN[volume]: connected["source_id"].to_numpy(),
                        "embedding_row": connected["node_id"].to_numpy(),
                        "cell_type": connected["cell_type"].to_numpy(),
                        "super_class": connected["super_class_raw"].to_numpy(),
                        "experimental_label": connected["nt_known"].to_numpy(),
                        "predicted_label": connected["nt_train"].to_numpy(),
                        "scored": connected["nt_known"].notna().to_numpy()})
    for target, prefix in PROBES.items():
        rows = pred[pred.probe_target == target].set_index("node_id")
        if rows.index.duplicated().any() or set(rows.index) != set(out.embedding_row):
            raise SystemExit(f"{stem}: the {target} probe does not call each connected {volume} brain "
                             f"neuron once; run tests/check_release.py")
        rows = rows.reindex(out.embedding_row)
        # This probe's classes: the vote-share columns it fills (other probes' are empty)
        classes = sorted({c[len("knn_p_"):] for c in rows.columns
                          if c.startswith("knn_p_") and rows[c].notna().all()})
        if not (rows.pred_linear.isin(classes) & rows.pred_knn.isin(classes)).all():
            raise SystemExit(f"{stem}: a {target} call is not one of {classes}; run tests/check_release.py")
        out[f"{prefix}_linear_call"] = rows.pred_linear.to_numpy()
        out[f"{prefix}_linear_p"] = rows.p_linear.to_numpy()
        for c in classes:
            out[f"{prefix}_linear_p_{c}"] = rows[f"p_{c}"].to_numpy()
        out[f"{prefix}_knn_call"] = rows.pred_knn.to_numpy()
        out[f"{prefix}_knn_share"] = rows.knn_share.to_numpy()
        for c in classes:
            out[f"{prefix}_knn_share_{c}"] = rows[f"knn_p_{c}"].to_numpy()
    return out


def main() -> None:
    """Write both tables and the README."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--release", type=Path, default=Path("experiments/release"))
    ap.add_argument("--out", type=Path, help="default: the release folder")
    args = ap.parse_args()
    out_dir = args.out or args.release
    out_dir.mkdir(parents=True, exist_ok=True)

    counts = {}
    for volume in RUNS:
        with contextlib.redirect_stdout(io.StringIO()):
            nodes = harmonise(pd.read_parquet(f"data/processed/{volume}_nodes.parquet")
                              .sort_values("node_id").reset_index(drop=True), volume)
        t = table(volume, args.release, nodes)
        t.to_parquet(out_dir / f"wiretype_predictions_{volume}.parquet", index=False)
        t.to_csv(out_dir / f"wiretype_predictions_{volume}.csv.gz", index=False)
        counts[volume] = len(t)
        print(f"{volume}: {len(t):,} neurons, {t.scored.sum():,} scored; "
              f"linear calls of the two probes agree on {(t.explabel_linear_call == t.trainlabel_linear_call).mean():.1%}")
    # The splits the FAFB-trained models used, keyed by FlyWire root ID like the tables above
    ids = pd.read_parquet("data/processed/fafb_nodes.parquet", columns=["node_id", "source_id"])
    splits = (pd.read_parquet("data/processed/fafb_splits.parquet").merge(ids, on="node_id", how="left")
              .rename(columns={"source_id": ID_COLUMN["fafb"], "split": "split_type_blocked"}))
    splits = splits[[ID_COLUMN["fafb"], "split_random", "split_type_blocked"]]
    splits.to_parquet(out_dir / "fafb_splits.parquet", index=False)
    splits.to_csv(out_dir / "fafb_splits.csv.gz", index=False)
    (out_dir / "README_predictions.md").write_text(
        README.format(n_mcns=counts["mcns"], n_fafb=counts["fafb"], id_mcns=ID_COLUMN["mcns"], id_fafb=ID_COLUMN["fafb"],
                      n_split=len(splits)))
    print(f"wrote {out_dir}/wiretype_predictions_{{mcns,fafb}}.{{parquet,csv.gz}}, fafb_splits.{{parquet,csv.gz}} "
          f"({len(splits):,} neurons) and README_predictions.md")


if __name__ == "__main__":
    main()
