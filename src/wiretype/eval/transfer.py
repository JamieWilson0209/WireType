"""The zero-shot transfer readout: probe fitted on the source volume, applied to the target.

Every row of the paper's comparison table goes through `zero_shot`, whether the
embeddings come from the encoder (`scripts/transfer.py`) or a baseline
(`scripts/score_embeddings.py`), so the numbers are comparable by construction.

- The probe (`probes_fit_apply`, linear and k-NN) is fitted on the source's
  `split_random` training split, with a source-fitted scaler, and tuned on its
  validation split.
- It is applied to the target's brain cells unchanged. No target label is used.
- Only neurons with at least one connection take part, on both sides (scope
  `brain_neurons`, `wiretype.data.scope`): the probe is fitted, tuned and tested on
  connected source neurons, and only connected target brain neurons are scored. A
  neuron without connections gives the encoder no wiring to read, so its call says
  nothing about transfer. The input scaler and the embeddings are unchanged.
- Experimental labels (`nt_known`) are scored first. `side` is scored but must not
  be quoted: MCNS and FAFB use different side vocabularies for midline cells.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..data.labels import harmonise
from ..data.scope import in_scope, restrict_split
from ..log import log
from .probes import probes_fit_apply

# Targets for the in-volume diagnostic.
TARGETS = ("nt_train", "hemilineage", "super_class", "nt_known", "side")
# The zero-shot plan, in reporting order. The target volume's brain only: FAFB is a
# brain, so the MCNS VNC is not "the same cells in another animal".
PLAN = [("nt_known", "brain"), ("nt_train", "brain"), ("nt_best", "brain"),
        ("super_class", "brain"), ("hemilineage", "brain"), ("side", "brain")]
# The transmitter probes whose calls and fitted parameters are released: the one
# fitted on experimental labels (scored in the paper) and the one fitted on training
# labels (which also calls the neurons without an experimental label).
RELEASE_PROBES = ("nt_known", "nt_best")


def load_volume(processed: Path, volume: str, split_column: str = "split_random"):
    """Harmonised nodes and the split, both in `node_id` order."""
    nodes = harmonise(
        pd.read_parquet(processed / f"{volume}_nodes.parquet").sort_values("node_id").reset_index(drop=True),
        volume)
    splits = pd.read_parquet(processed / f"{volume}_splits.parquet").sort_values("node_id").reset_index(drop=True)
    if split_column not in splits.columns:
        raise SystemExit(f"{volume}_splits.parquet has no column {split_column}; rebuild the splits")
    return nodes, splits[split_column]


def zero_shot(src_emb, src_nodes, src_split, tgt_emb, tgt_nodes, seed=0, dump=False,
              plan=PLAN, by_status=True, predict_all=False, probes=None):
    """The zero-shot readout. Returns the report sections and, if `dump`, per-cell rows.

    With `dump`, the `nt_best` probe also predicts the target's unlabelled brain
    cells that have synapses (`where == "target_unlabelled"`): the application case.
    With `predict_all` as well, every probe in `RELEASE_PROBES` does, so each calls
    every connected target brain neuron. If `probes` is a dict, it is filled with the
    fitted `RELEASE_PROBES` as plain arrays, keyed by target, for release; the k-NN
    reference rows become the source's `node_id`s.
    """
    # Connected neurons only, on both sides: the source's unconnected neurons are marked
    # `excluded` in the split the probes use (fit, tuning, held-out score), and the
    # target tier is its connected brain neurons (FAFB has no nerve cord)
    report = {"zero_shot": {}, "by_status": {}}
    src_split = restrict_split(src_nodes, src_split, "brain_neurons")
    tiers = {"brain": in_scope(tgt_nodes, "brain_neurons")}
    predictions: list[pd.DataFrame] = []
    log("zero-shot: probe fitted on the source, applied to the target unchanged.")
    log("  macro-F1 linear / k-NN, then support-weighted F1 linear / k-NN, with the source's")
    log("  own held-out macro-F1 beside it.")
    # One probe per target label: fitted on the source, applied to the target's brain.
    # With `dump`, per-neuron rows are kept; the nt_best probe also predicts unlabelled
    # target neurons with synapses (`target_unlabelled`).
    for target, tier in plan:
        calls_all = dump and (target == "nt_best" or (predict_all and target in RELEASE_PROBES))
        res = probes_fit_apply(src_emb, src_nodes, src_split, tgt_emb, tgt_nodes, tiers[tier],
                               target=target, scope="whole_brain", seed=seed,
                               return_predictions=dump,
                               unlabelled_mask=(tiers["brain"] & (tgt_nodes["n_in"] + tgt_nodes["n_out"]).gt(0).to_numpy())
                               if calls_all else None,
                               return_probe=probes is not None and target in RELEASE_PROBES)
        if "probe" in res:
            probe = res.pop("probe")
            probe["knn_reference_node_id"] = src_nodes["node_id"].to_numpy()[probe.pop("knn_reference_rows")]
            probes[target] = probe
        if "predictions" in res:
            frame = res.pop("predictions")
            nodes_of = {"source_test": src_nodes, "target": tgt_nodes, "target_unlabelled": tgt_nodes}
            frame.insert(0, "node_id", [int(nodes_of[w]["node_id"].iat[r])
                                        for w, r in zip(frame["where"], frame["row"])])
            frame.insert(0, "probe_target", target)
            predictions.append(frame.drop(columns="row"))
        report["zero_shot"][f"{target}/{tier}"] = res
        if "linear" in res:
            log(f"  {target + '/' + tier:<22} {res['linear']['macro_f1']:.4f} / "
                f"{res['knn']['macro_f1']:.4f}   weighted {res['linear']['weighted_f1']:.4f} / "
                f"{res['knn']['weighted_f1']:.4f}   (source {res['source_linear_macro_f1']:.4f}) "
                f"  n={res['n_target']:,}  dropped={res['target_cells_dropped']:,}")
        else:
            log(f"  {target + '/' + tier:<22} {res.get('skipped', '—')}")

    # Transmitter (predicted label) by the target's reconstruction status, for statuses
    # with at least 500 neurons
    if by_status:
        log("zero-shot transmitter by reconstruction status (brain only):")
        for status in pd.Series(tgt_nodes["status"]).dropna().value_counts().index[:5]:
            mask = tiers["brain"] & (tgt_nodes["status"].to_numpy() == status)
            if mask.sum() < 500:
                continue
            res = probes_fit_apply(src_emb, src_nodes, src_split, tgt_emb, tgt_nodes, mask,
                                   target="nt_train", scope="whole_brain", seed=seed)
            if "linear" in res:
                report["by_status"][str(status)] = res
                log(f"  {str(status):<24} {res['linear']['macro_f1']:.4f} / {res['knn']['macro_f1']:.4f}"
                    f"   n={res['n_target']:,}")
    return report, (pd.concat(predictions, ignore_index=True) if predictions else None)
