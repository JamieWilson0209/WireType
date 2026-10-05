"""The released transmitter probes: write them, read them back, and re-apply them.

A release run (`scripts/transfer.py --predict-all --save-probes`) fits the probes in
`wiretype.eval.transfer.RELEASE_PROBES` on the source volume and writes them as plain
arrays, so anyone can re-apply them to the released embeddings without scikit-learn's
pickles:

- `<report>_probes.npz`: per probe, keys `<label>__<array>`: `classes`, `scaler_mean`,
  `scaler_scale`, `linear_coef`, `linear_intercept`, `knn_reference_node_id` (source
  neurons) and `knn_reference_codes` (their classes, as indices into `classes`);
- `<report>_probes.json`: how to apply them, and each probe's C, k and size.

`apply_probe` is the reference implementation of that recipe; `tests/check_release.py`
uses it to show that the released probes reproduce the released calls.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from .probes import KNN_K

APPLY = ("x = (embedding - scaler_mean) / scaler_scale. Linear: probabilities are "
         "softmax(x @ linear_coef.T + linear_intercept) over `classes`; the call is the largest. "
         "k-NN: the k reference neurons (knn_reference_node_id, in the source volume's embeddings, "
         "scaled the same way) nearest to x by Euclidean distance vote with their class codes "
         "(knn_reference_codes, indices into `classes`); a tie goes to the first class in `classes`.")


def write_probes(out: Path, probes: dict, report: dict) -> None:
    """Write the fitted probes beside the report `out` (`<stem>_probes.npz` and `.json`)."""
    meta = {"checkpoint": report["checkpoint"], "source": report["source"], "target": report["target"],
            "standardise_with": report["standardise_with"], "apply": APPLY, "probes": {}}
    arrays = {}
    for target, probe in probes.items():
        res = report["zero_shot"][f"{target}/brain"]
        meta["probes"][target] = {"classes": [str(c) for c in probe["classes"]],
                                  "logistic_C": res["linear"]["C"], "knn_k": res["knn"]["k"],
                                  "n_reference": int(len(probe["knn_reference_codes"]))}
        for key, value in probe.items():
            arrays[f"{target}__{key}"] = np.asarray(value).astype(str) if key == "classes" else np.asarray(value)
    np.savez_compressed(out.with_name(out.stem + "_probes.npz"), **arrays)
    out.with_name(out.stem + "_probes.json").write_text(json.dumps(meta, indent=2))


def read_probes(npz: Path) -> dict:
    """The probes written by `write_probes`, keyed by label, each with its k."""
    meta = json.loads(npz.with_suffix(".json").read_text())
    arrays = np.load(npz, allow_pickle=False)
    probes = {}
    for target, info in meta["probes"].items():
        probe = {key.split("__", 1)[1]: arrays[key] for key in arrays.files if key.startswith(f"{target}__")}
        probe["k"] = int(info["knn_k"])
        probes[target] = probe
    return probes


def apply_probe(probe: dict, source_embeddings: np.ndarray, target_embeddings: np.ndarray,
                rows: np.ndarray) -> pd.DataFrame:
    """Re-apply one released probe to target neurons `rows` (embedding rows = `node_id`s).

    Returns, per row, the linear call and probabilities and the k-NN call and vote shares,
    computed only from the released arrays.
    """
    classes = probe["classes"]

    def scaled(embeddings: np.ndarray) -> np.ndarray:
        # As scikit-learn's StandardScaler.transform does in place: the result keeps the
        # embeddings' own precision (float32 in a release run), so near-ties resolve alike
        x = embeddings.astype(np.float32)
        x -= probe["scaler_mean"]
        x /= probe["scaler_scale"]
        return x

    x = scaled(target_embeddings[rows])
    logits = x @ probe["linear_coef"].T + probe["linear_intercept"]
    logits -= logits.max(axis=1, keepdims=True)
    proba = np.exp(logits)
    proba /= proba.sum(axis=1, keepdims=True)

    reference = scaled(source_embeddings[probe["knn_reference_node_id"]])
    k = probe["k"]
    # The fitted probe queried the largest candidate k and voted among the first k
    searcher = NearestNeighbors(n_neighbors=min(max(KNN_K), len(reference))).fit(reference)
    neighbours = searcher.kneighbors(x, return_distance=False)[:, :k]
    codes = probe["knn_reference_codes"][neighbours]
    counts = np.zeros((len(rows), len(classes)))
    np.add.at(counts, (np.repeat(np.arange(len(rows)), k), codes.ravel()), 1)

    frame = pd.DataFrame({"node_id": rows, "pred_linear": classes[proba.argmax(axis=1)],
                          "pred_knn": classes[counts.argmax(axis=1)]})
    for j, c in enumerate(classes):
        frame[f"p_{c}"] = proba[:, j]
        frame[f"knn_p_{c}"] = counts[:, j] / k
    return frame
