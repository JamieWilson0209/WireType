#!/usr/bin/env python3
"""Numbers in the paper: transfer by region, reference floors, anatomy by region.

Reads saved `_predictions.parquet` files only (runs locally; no model is run) and prints:
- Table 1: transmitter weighted F1 on MCNS experimental labels by MCNS super-class, for
  the three paper-encoder seeds, beside the rule that gives every neuron of a group
  the group's commonest transmitter;
- the same rule on the whole scored set;
- the type-blocked groups (seen and held-out types) in weighted F1 for the trained and
  untrained arms over three seeds, the predominant-transmitter baseline, and per-class
  precision, recall and F1 on held-out types (mean of seeds; untyped and name-unmatched
  MCNS neurons are left out);
- super-class recall per MCNS super-class, and hemilineage weighted F1 by region
  (seed 0);
- MCNS neurons with no label of either kind that the probe calls at >= 0.9;
- type matching by region: for up to 4,000 scored MCNS neurons per super-class whose
  type occurs among FAFB's training neurons, whether the nearest FAFB training neuron
  (cosine, FAFB-standardised embeddings, seed 0) has the same named type, and
  whether it has the same transmitter;
- temperature scaling fitted on FAFB's held-out neurons and applied to MCNS: the
  calibration error and coverage at 95% and 99% precision;
- the reverse direction over three seeds beside the forward direction, with the
  untrained and raw-features references and the dopaminergic output-synapse test.

    PYTHONPATH=src .venv/bin/python scripts/diagnostics/regions_and_floors.py
"""
from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support

sys.path.insert(0, str(Path(__file__).resolve().parent))
from unseen_types import groups  # noqa: E402

from wiretype.data.labels import harmonise  # noqa: E402

PROCESSED = Path("data/processed")
TRANSFER = Path("experiments/transfer")
PAPER = "transfer_fafb_to_mcns_checkpoint_fafb_displacement_refined_split_random_s24k_ntbest_{}source_predictions.parquet"
SEEDS = {"seed 0": "", "seed 1": "seed1_", "seed 2": "seed2_"}
# The type-blocked runs, one file per seed (seed 0 carries no suffix)
TYPEBLOCKED = {arm: [f"transfer_fafb_to_mcns_checkpoint_fafb_{key}_refined_s24k_ntbest_typeblocked{seed}_source_predictions.parquet"
                     for seed in ("", "_seed1", "_seed2")]
               for arm, key in (("trained", "displacement"), ("untrained", "untrained"))}
COLUMNS = ["probe_target", "node_id", "where", "true", "pred_linear", "pred_knn", "p_linear"]


def weighted(true: pd.Series, pred) -> float:
    """Support-weighted F1 of `pred` against `true`, over the classes present in `true`."""
    labels = sorted(true.unique())
    return f1_score(true, pred, labels=labels, average="weighted", zero_division=0)


def commonest(true: pd.Series) -> float:
    """Weighted F1 of calling every neuron the group's commonest class."""
    return weighted(true, [true.value_counts().idxmax()] * len(true))


def rows(path: Path, target: str, where: str = "target") -> pd.DataFrame:
    """The prediction rows for one probe target and one population (`where`)."""
    pred = pd.read_parquet(path, columns=COLUMNS)
    return pred[(pred["probe_target"] == target) & (pred["where"] == where)]


def by_region(mcns: pd.DataFrame) -> None:
    """Table 1: transmitter weighted F1 by MCNS super-class for the three seeds, beside
    the commonest-transmitter rule.
    """
    table: dict[str, dict] = {}
    for seed, tag in SEEDS.items():
        part = rows(TRANSFER / PAPER.format(tag), "nt_known").merge(
            mcns[["node_id", "super_class", "cell_type"]], on="node_id", how="left")
        if seed == "seed 0":
            print(f"all scored neurons: {len(part):,}; commonest-transmitter rule {commonest(part['true']):.3f}")
        for region, group in part.groupby("super_class"):
            entry = table.setdefault(region, {"neurons": len(group), "types": group["cell_type"].nunique(),
                                              "rule": commonest(group["true"]), "linear": [], "knn": []})
            entry["linear"].append(weighted(group["true"], group["pred_linear"]))
            entry["knn"].append(weighted(group["true"], group["pred_knn"]))
    print("\nTable 1: MCNS super-class, neurons, types, linear mean (min-max), k-NN mean, rule")
    for region, e in sorted(table.items(), key=lambda kv: -kv[1]["neurons"]):
        lin = e["linear"]
        print(f"  {region:20s} {e['neurons']:7,d} {e['types']:5d}  {np.mean(lin):.3f} "
              f"({min(lin):.3f}-{max(lin):.3f})  {np.mean(e['knn']):.3f}  {e['rule']:.3f}")


def typeblocked() -> None:
    """The type-blocked split over three seeds: weighted F1 on seen and held-out types,
    trained and untrained, with the predominant-transmitter baseline, and per-class
    scores on held-out types.
    """
    print("\nType-blocked split, MCNS experimental labels, weighted F1 mean (min-max) over seeds, linear / k-NN:")
    span = lambda v: f"{np.mean(v):.3f} ({min(v):.3f}-{max(v):.3f})"  # noqa: E731
    for arm, names in TYPEBLOCKED.items():
        runs = [groups(pd.read_parquet(TRANSFER / name, columns=COLUMNS), PROCESSED) for name in names]
        for group, label in (("seen", "seen"), ("unseen", "held-out")):
            parts = [r[r["group"] == group] for r in runs]
            line = (f"  {arm:9s} {label:8s} {len(parts[0]):6,d} neurons {parts[0]['cell_type'].nunique():4d} types  "
                    f"{span([weighted(p['true'], p['pred_linear']) for p in parts])} / "
                    f"{span([weighted(p['true'], p['pred_knn']) for p in parts])}")
            if arm == "trained":
                line += f"  baseline {commonest(parts[0]['true']):.3f}"
            print(line)
            if arm == "trained" and group == "unseen":
                labels = sorted(parts[0]["true"].unique())
                scores = np.mean([precision_recall_fscore_support(p["true"], p["pred_linear"], labels=labels,
                                                                  zero_division=0)[:3] for p in parts], axis=0)
                n = parts[0]["true"].value_counts()
                for label_, (pr, rc, f) in zip(labels, scores.T):
                    print(f"      {label_:14s} precision {pr:.2f} recall {rc:.2f} F1 {f:.2f} ({n[label_]:,d})")



def anatomy(mcns: pd.DataFrame) -> None:
    """Super-class recall per MCNS super-class, and hemilineage weighted F1 by region
    (mean over three seeds, with the range).
    """
    def span(values: list[float]) -> str:
        return f"{np.mean(values):.3f} ({min(values):.3f}-{max(values):.3f})"

    recalls, regions = [], {}
    for tag in SEEDS.values():
        path = TRANSFER / PAPER.format(tag)
        sup = rows(path, "super_class")
        recalls.append((sup["true"] == sup["pred_linear"]).groupby(sup["true"]).agg(["mean", "size"]))
        hem = rows(path, "hemilineage").merge(mcns[["node_id", "super_class"]], on="node_id", how="left")
        for region, part in hem.groupby("super_class"):
            if len(part) >= 200:
                regions.setdefault(region, {"n": len(part), "share": len(part) / len(hem), "linear": [], "knn": []})
                regions[region]["linear"].append(weighted(part["true"], part["pred_linear"]))
                regions[region]["knn"].append(weighted(part["true"], part["pred_knn"]))
    print("\nSuper-class recall per MCNS super-class (linear; three-seed mean and range):")
    for cls in recalls[0].index:
        values = [r.loc[cls, "mean"] for r in recalls]
        print(f"  {cls:20s} {int(recalls[0].loc[cls, 'size']):6,d}  {span(values)}")
    print("\nHemilineage by super-class (weighted F1 linear / k-NN; three-seed mean and range):")
    for region, r in regions.items():
        print(f"  {region:20s} {r['n']:6,d} ({r['share']:.0%})  {span(r['linear'])} / {span(r['knn'])}")


def unlabelled() -> None:
    """How many MCNS neurons with no label of either kind the probe calls with
    confidence >= 0.9.
    """
    part = rows(TRANSFER / PAPER.format(""), "nt_best", where="target_unlabelled")
    print(f"\nNo label of either kind: {len(part):,} neurons, {(part['p_linear'] >= 0.9).sum():,} called at >= 0.9")


def type_matching(mcns: pd.DataFrame) -> None:
    """Per MCNS super-class, the share of neurons whose nearest FAFB training neuron has
    the same named type, and the share whose nearest FAFB training neuron has the same
    transmitter (its training label against the MCNS neuron's experimental label).
    """
    from sklearn.preprocessing import StandardScaler
    with contextlib.redirect_stdout(io.StringIO()):
        fafb = harmonise(pd.read_parquet(PROCESSED / "fafb_nodes.parquet"), "fafb")
    split = pd.read_parquet(PROCESSED / "fafb_splits.parquet").set_index("node_id").reindex(fafb["node_id"])
    # connected neurons only on both sides, as the transfer scores
    train = ((split["split_random"] == "train").to_numpy()
             & ((fafb["n_in"] + fafb["n_out"]) > 0).to_numpy())
    stem = TRANSFER / PAPER.format("").replace("_predictions.parquet", "_embeddings_")
    ef, em = (np.load(f"{stem}{v}.npy").astype(np.float32) for v in ("fafb", "mcns"))
    scaler = StandardScaler().fit(ef[train])
    ref = scaler.transform(ef[train])
    ref /= np.linalg.norm(ref, axis=1, keepdims=True)
    types = fafb["cell_type"].to_numpy()[train]
    transmitters = fafb["nt_best"].to_numpy()[train]  # each FAFB neighbour's training label
    scored =((mcns["region"] == "brain") & mcns["nt_known"].notna() & ((mcns["n_in"] + mcns["n_out"]) > 0)
              & mcns["cell_type"].isin(set(pd.Series(types).dropna()))).to_numpy()
    rng = np.random.default_rng(0)
    print("\nType matching by region (seed 0): nearest FAFB training neuron has the same named type, "
          "and the same transmitter")
    for region in ("optic", "sensory", "central", "visual_projection", "ascending", "descending"):
        q = np.flatnonzero(scored & (mcns["super_class"] == region).to_numpy())
        q = rng.choice(q, min(4000, len(q)), replace=False)
        x = scaler.transform(em[q])
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        nn = np.concatenate([np.argmax(x[i:i + 1000] @ ref.T, axis=1) for i in range(0, len(x), 1000)])
        same = (types[nn] == mcns["cell_type"].to_numpy()[q]).mean()
        same_nt = (transmitters[nn] == mcns["nt_known"].to_numpy()[q]).mean()
        print(f"  {region:20s} {len(q):5,d} neurons  same type {same:.3f}  same transmitter {same_nt:.3f}")


def temperature() -> None:
    """Fit one temperature on FAFB held-out neurons and report calibration and coverage
    on MCNS before and after.
    """
    from scipy.optimize import minimize_scalar
    classes = ["acetylcholine", "dopamine", "gaba", "glutamate", "histamine", "octopamine", "serotonin"]
    pred = pd.read_parquet(TRANSFER / PAPER.format(""),
                           columns=["probe_target", "where", "true"] + [f"p_{c}" for c in classes])

    def get(where):
        """Log-probabilities and integer labels of the confident-label rows in one
        population.
        """
        d = pred[(pred["probe_target"] == "nt_known") & (pred["where"] == where)]
        logp = np.log(np.clip(d[[f"p_{c}" for c in classes]].to_numpy(float), 1e-12, 1))
        return logp, d["true"].map({c: i for i, c in enumerate(classes)}).to_numpy()

    def soft(logp, t):
        """Softmax of log-probabilities divided by temperature `t`."""
        z = logp / t
        z -= z.max(1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(1, keepdims=True)

    def ece(p, y):
        """Expected calibration error over ten equal-width confidence bins."""
        conf, ok, out = p.max(1), p.argmax(1) == y, 0.0
        for lo in np.arange(0, 1, 0.1):
            m = (conf > lo) & (conf <= lo + 0.1)
            if m.any():
                out += m.mean() * abs(ok[m].mean() - conf[m].mean())
        return out

    def coverage(p, y, precision):
        """The largest share of neurons, most confident first, whose running precision
        reaches `precision`.
        """
        order = np.argsort(-p.max(1))
        ok = (p.argmax(1) == y)[order]
        running = np.cumsum(ok) / np.arange(1, len(ok) + 1)
        hit = np.flatnonzero(running >= precision)
        return (hit.max() + 1) / len(ok) if len(hit) else 0.0

    ls, ys = get("source_test")
    lt, yt = get("target")
    fitted = minimize_scalar(lambda t: -np.log(np.clip(soft(ls, t)[np.arange(len(ys)), ys], 1e-12, 1)).mean(),
                             bounds=(0.05, 20), method="bounded").x
    print("\nTemperature scaling fitted on FAFB held-out neurons (seed 0, linear probe):")
    for name, t in (("none", 1.0), (f"T = {fitted:.2f}", fitted)):
        pt = soft(lt, t)
        print(f"  {name:9s} FAFB ECE {ece(soft(ls, t), ys):.3f} | MCNS ECE {ece(pt, yt):.3f}, coverage at 95% "
              f"{coverage(pt, yt, 0.95):.3f}, at 99% {coverage(pt, yt, 0.99):.4f}")


def reverse() -> None:
    """The reverse direction (MCNS-trained, scored on FAFB) over three seeds, beside the
    forward direction: weighted and macro-F1, dopamine recall, the untrained and
    raw-features references, and the output synapses of FAFB's dopaminergic neurons
    called correctly against those missed (one-sided Mann-Whitney, per seed).
    """
    from scipy.stats import mannwhitneyu
    rev = "transfer_mcns_to_fafb_checkpoint_mcns_{}_refined_split_random_s24k_ntbest_brain_{}{}"
    fwd = "transfer_fafb_to_mcns_checkpoint_fafb_{}_refined_split_random_s24k_ntbest{}_{}"
    span = lambda v: f"{np.mean(v):.3f} ({min(v):.3f}-{max(v):.3f})"  # noqa: E731
    with contextlib.redirect_stdout(io.StringIO()):
        fafb = harmonise(pd.read_parquet(PROCESSED / "fafb_nodes.parquet"), "fafb")
    syn_out = fafb.set_index("node_id")["syn_out"]

    def summary(stems: list[str]) -> dict:
        out = {"lin": [], "knn": [], "macro": [], "da": []}
        for stem in stems:
            part = rows(TRANSFER / f"{stem}_predictions.parquet", "nt_known")
            out["lin"].append(weighted(part["true"], part["pred_linear"]))
            out["knn"].append(weighted(part["true"], part["pred_knn"]))
            labels = sorted(part["true"].unique())
            out["macro"].append(f1_score(part["true"], part["pred_linear"], labels=labels, average="macro", zero_division=0))
            da = part[part["true"] == "dopamine"]
            out["da"].append(float((da["pred_linear"] == "dopamine").mean()))
        return out

    seeds = ("", "seed1_", "seed2_")
    runs = {
        "reverse, input scaling": [rev.format("displacement", s, "source") for s in seeds],
        "reverse, target scaling": [rev.format("displacement", s, "own") for s in seeds],
        "reverse, untrained": [rev.format("untrained", s, "source") for s in seeds],
        "forward, input scaling": [fwd.format("displacement", s, "source") for s in ("", "_seed1", "_seed2")],
        "forward, target scaling (seed 0)": [fwd.format("displacement", "", "own_neurons")],
    }
    print("\nReverse direction, FAFB experimental labels: mean (min-max) over seeds")
    for name, stems in runs.items():
        s = summary(stems)
        print(f"  {name:34s} weighted lin {span(s['lin'])}  k-NN {span(s['knn'])}  macro lin {span(s['macro'])}  "
              f"DA recall {span(s['da'])}")
        if name == "reverse, input scaling":
            for stem in stems:
                part = rows(TRANSFER / f"{stem}_predictions.parquet", "nt_known")
                da = part[part["true"] == "dopamine"]
                hit, miss = syn_out.reindex(da.loc[da["pred_linear"] == "dopamine", "node_id"]), \
                    syn_out.reindex(da.loc[da["pred_linear"] != "dopamine", "node_id"])
                p = mannwhitneyu(hit, miss, alternative="greater").pvalue
                print(f"      {stem[-20:]:20s} DA output synapses, correct {hit.median():.0f} ({len(hit)}) against missed "
                      f"{miss.median():.0f} ({len(miss)}): {hit.median() / miss.median():.2f}x, p = {p:.1e}")
    raw = pd.read_parquet("experiments/baselines/raw_s0_mcns_to_fafb_predictions.parquet",
                          columns=["probe_target", "where", "true", "pred_linear", "pred_knn"])
    raw = raw[(raw["probe_target"] == "nt_known") & (raw["where"] == "target")]
    da = raw[raw["true"] == "dopamine"]
    print(f"  raw features (one deterministic run)  weighted lin {weighted(raw['true'], raw['pred_linear']):.3f}  "
          f"k-NN {weighted(raw['true'], raw['pred_knn']):.3f}  DA recall {(da['pred_linear'] == 'dopamine').mean():.3f}")


def main() -> None:
    """Print every block of numbers above, in order."""
    with contextlib.redirect_stdout(io.StringIO()):
        mcns = harmonise(pd.read_parquet(PROCESSED / "mcns_nodes.parquet"), "mcns")
    by_region(mcns)
    typeblocked()
    anatomy(mcns)
    unlabelled()
    type_matching(mcns)
    temperature()
    reverse()


if __name__ == "__main__":
    main()
