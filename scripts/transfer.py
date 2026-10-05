#!/usr/bin/env python3
"""Apply a trained encoder, frozen, to another volume: zero-shot and in-volume readings.

**Zero-shot** (`wiretype.eval.transfer.zero_shot`). The probe is fitted on the
source volume's embeddings and labels, tuned on its validation split, and applied
to the target volume unchanged: no refitting, no rescaling to target statistics,
no target label. The probe's scaler travels with it, so the target has to arrive
in the source's scaled space. The source's own held-out score under the same
fitted probe comes back beside it, and the drop between them is the quantity of
interest.

**In-volume** (`--in-volume`). The encoder is still frozen, but the probe is fitted
on the target's own training split, restricted like the zero-shot readout to brain
neurons with at least one connection. This asks whether the representation is
useful when a few target labels exist. It is not zero-shot, and under
`split_random` it leaks through same-type siblings, so it is a diagnostic only.

**Input scaling.** The encoder's 31 input columns are standardised before
encoding.
- `source` (default): the target is scaled with the source's training-split
  statistics.
- `own_neurons`: the target is scaled with its own brain neurons that carry a
  transmitter label, the population the source's statistics describe.
- `own`: the target's whole training split. MCNS's includes 22% zero-synapse
  bodies, which inflate every column's spread, so this is kept only as an ablation.

    qsub -v CKPT=experiments/train/checkpoint_fafb_displacement_refined_split_random_s24k_ntbest.pt,DUMP=1,EMB=1 \\
        hpc/jobs/transfer.sh

Writes `<reports>/transfer_<source>_to_<target>_<checkpoint>_<scaling>.json`, and with
`--dump-predictions` the per-cell rows `scripts/breakdown.py` reads.

**Release** (`--predict-all --save-probes --embeddings-dtype float32`, `RELEASE=1` in
`hpc/jobs/transfer.sh`). Both released transmitter probes (`RELEASE_PROBES`: fitted
on experimental labels, and on training labels) call every connected target brain
neuron, the fitted probes are written to `<report>_probes.npz` and `.json`, and the
embeddings are saved in float32 so the probes re-apply exactly. `scripts/release.py`
builds the released tables from these files.

Paper terms: the encoder is WireType; `source` scaling is input scaling and
`own_neurons` (forward) or `own` (reverse) is target scaling; the probes are the
linear and k-NN read-outs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wiretype.data.scope import restrict_split
from wiretype.data.features import effective_rank, feature_summary, standardise_with, wiring_features
from wiretype.eval.probes import probe_all
from wiretype.eval.release import write_probes
from wiretype.eval.transfer import TARGETS, load_volume, zero_shot
from wiretype.log import log
from wiretype.model.encoder import CellEncoder
from wiretype.model.tokens import context_fn, embed_all


def load_encoder(path: Path, device):
    """The frozen `E_c` from a training checkpoint, and the checkpoint's settings."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    settings = ckpt["settings"]
    encoder = CellEncoder(ckpt["encoder"]["token.weight"].shape[1] - 3, len(ckpt["classes"]),
                          d=settings["width"], heads=settings["heads"],
                          layers=settings["layers"], ff_mult=settings["ff_mult"]).to(device)
    encoder.load_state_dict(ckpt["encoder"])
    encoder.eval()
    return encoder, settings


def embed_volume(volume, processed, settings, encoder, device, split_column, amp,
                 stats=None, fit_neurons_only=False, scope="all"):
    """Every cell of `volume` through the frozen encoder. Returns nodes, split, embeddings, stats.

    With `stats`, the input columns are scaled with them (another volume's);
    otherwise they are fitted on this volume's training split, restricted to brain
    neurons with a transmitter label if `fit_neurons_only`. `scope` (the source's,
    from the checkpoint) marks out-of-scope cells `excluded` in the split, so the
    scaler, the probe and the source's held-out score skip them as training did.
    """
    nodes, split = load_volume(processed, volume, split_column)
    split = restrict_split(nodes, split, scope)
    edges = pd.read_parquet(processed / f"{volume}_edges.parquet")
    raw, names = wiring_features(processed, volume, edges, len(nodes),
                                 blocks=settings["features"], columns=settings["column_set"])
    fit_on = (split == "train").to_numpy()
    if stats is None and fit_neurons_only:
        fit_on = fit_on & (nodes["region"].to_numpy() == "brain") & nodes["nt_train"].notna().to_numpy()
    features, stats = standardise_with(raw, fit_on, stats)
    log(f"  {volume}: {len(nodes):,} cells · {features.shape[1]} wiring columns · "
        f"{feature_summary(names)} · effective rank {effective_rank(features):.2f} · split {split_column}")
    del edges, raw

    expected = encoder.token.in_features - 3
    if features.shape[1] != expected:
        raise SystemExit(f"{volume} builds {features.shape[1]} columns but the encoder takes {expected}")

    k = settings["k"]
    path = processed / f"{volume}_topk{k}.npz"
    if not path.exists():
        raise SystemExit(f"no {path} — build it: qsub -v VOLUME={volume},K={k} hpc/jobs/topk.sh")
    topk = dict(np.load(path))
    if topk["partner"].shape != (len(nodes), k):
        raise SystemExit(f"{path} holds {topk['partner'].shape}, wanted {(len(nodes), k)}")
    features_t = torch.from_numpy(features).to(device)
    emb = embed_all(context_fn(encoder), topk, features_t, len(nodes), device, amp).cpu().numpy()
    del features_t, topk
    return nodes, split, emb, stats


def main() -> None:
    """Embed both volumes with a frozen checkpoint and write the zero-shot (and optional
    in-volume) report.
    """
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--processed", type=Path, default=Path("data/processed"))
    p.add_argument("--reports", type=Path, default=Path("experiments/transfer"))
    p.add_argument("--source", default="fafb")
    p.add_argument("--target", default="mcns")
    p.add_argument("--target-split-column", default="split_random",
                   help="the target's split, used by the in-volume probe")
    p.add_argument("--standardise-with", choices=("source", "own_neurons", "own"), default="source",
                   help="how the target's input columns are scaled; see the module docstring")
    p.add_argument("--dump-predictions", action="store_true",
                   help="also write per-cell predictions to <report>_predictions.parquet, "
                        "for scripts/breakdown.py")
    p.add_argument("--save-embeddings", action="store_true",
                   help="also write both volumes' embeddings, for scripts/score_embeddings.py")
    p.add_argument("--embeddings-dtype", choices=("float16", "float32"), default="float16",
                   help="precision of the saved embeddings; float32 for the release")
    p.add_argument("--predict-all", action="store_true",
                   help="with --dump-predictions: both released transmitter probes call every "
                        "connected target brain neuron, labelled or not")
    p.add_argument("--save-probes", action="store_true",
                   help="also write the released transmitter probes to <report>_probes.npz/.json")
    p.add_argument("--in-volume", action="store_true",
                   help="also refit the probe on the target's own split (a diagnostic, about 1 h)")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    if args.predict_all and not args.dump_predictions:
        p.error("--predict-all needs --dump-predictions")

    # --- 1. The frozen encoder (WireType) and its training settings
    args.reports.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = not args.no_amp and device.type == "cuda"
    encoder, settings = load_encoder(args.checkpoint, device)
    if str(settings.get("partner_labels", "none")) not in ("", "none"):
        raise SystemExit(f"{args.checkpoint.name} was trained with partner labels, which this "
                         f"code does not support")
    log(f"{args.checkpoint.name}: trained on {settings.get('volume', 'fafb')} · set "
        f"{settings['column_set']} · split {settings['split_column']} · supervised on "
        f"{settings['supervise_on']} · device {device}")

    # --- 2. Embed both volumes. The target is scaled with the source's statistics
    #     (input scaling) unless --standardise-with asks for its own (target scaling).
    scope = settings.get("scope", "all")  # checkpoints from before --scope trained on every cell
    if scope != "all":
        log(f"source scope {scope}: as in training (wiretype.data.scope)")
    src_nodes, src_split, src_emb, stats = embed_volume(
        args.source, args.processed, settings, encoder, device, settings["split_column"], amp, scope=scope)
    tgt_nodes, tgt_split, tgt_emb, _ = embed_volume(
        args.target, args.processed, settings, encoder, device, args.target_split_column, amp,
        stats=stats if args.standardise_with == "source" else None,
        fit_neurons_only=args.standardise_with == "own_neurons")

    log("")
    # --- 3. Zero-shot probes: fitted on the source, applied to the target unchanged
    probes = {} if args.save_probes else None
    sections, predictions = zero_shot(src_emb, src_nodes, src_split, tgt_emb, tgt_nodes,
                                      seed=args.seed, dump=args.dump_predictions,
                                      predict_all=args.predict_all, probes=probes)
    report = {"checkpoint": str(args.checkpoint), "settings": settings,
              "source": args.source, "target": args.target,
              "standardise_with": args.standardise_with, "source_scope": scope,
              "target_split_column": args.target_split_column, **sections}

    # --- 4. Optional in-volume diagnostic: probes refitted on the target's own split
    if args.in_volume:
        log("")
        log(f"in-volume: the same frozen encoder, probe refitted on {args.target}'s own training split.")
        # Connected brain neurons only, as the zero-shot readout
        report["in_volume"] = probe_all(tgt_emb, tgt_nodes, restrict_split(tgt_nodes, tgt_split, "brain_neurons"),
                                        targets=TARGETS, seed=args.seed)
        for key, res in report["in_volume"].items():
            if "linear" in res:
                log(f"  {key:<26} {res['linear']['macro_f1']:.4f} / {res['knn']['macro_f1']:.4f}")

    # --- 5. Write the report, and optionally predictions and embeddings
    out = args.reports / (f"transfer_{args.source}_to_{args.target}_{args.checkpoint.stem}"
                          f"_{args.standardise_with}.json")
    out.write_text(json.dumps(report, indent=2, default=str))
    if predictions is not None:
        # Probability columns differ per target, so the union is sparse; that is fine.
        predictions.to_parquet(out.with_name(out.stem + "_predictions.parquet"))
        log(f"wrote {out.stem}_predictions.parquet")
    if args.save_embeddings:
        for vol, emb in ((args.source, src_emb), (args.target, tgt_emb)):
            np.save(out.with_name(f"{out.stem}_embeddings_{vol}.npy"), emb.astype(args.embeddings_dtype))
        log(f"wrote both volumes' embeddings ({args.embeddings_dtype})")
    if probes:
        write_probes(out, probes, report)
        log(f"wrote the fitted probes ({', '.join(probes)}) to {out.stem}_probes.npz/.json")
    log(f"wrote {out}")


if __name__ == "__main__":
    main()
