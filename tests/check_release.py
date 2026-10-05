#!/usr/bin/env python3
"""Check the release run (`ONLY=release bash hpc/submit_paper.sh`) before building the tables.

Run locally after copying `experiments/release/` from the cluster:

    PYTHONPATH=src .venv/bin/python tests/check_release.py

For the seed-0 forward (FAFB -> MCNS) and reverse (MCNS -> FAFB) runs it checks
1. scores: every metric of the two released probes (`nt_known/brain`, `nt_best/brain`)
   equals the paper's report in `experiments/transfer` to within `--tol`;
2. calls: on every row the paper's predictions file holds for those probes (FAFB or
   MCNS held-out neurons, and scored target neurons), the release run makes the same
   linear and k-NN call; any flip is listed;
3. coverage: each released probe calls every connected target brain neuron exactly
   once, and no other neuron (none without connections, no MCNS nerve cord);
4. the probe files: `wiretype.eval.release.apply_probe` re-applies the saved probes to
   the saved float32 embeddings and reproduces the release run's calls exactly and its
   probabilities to within 1e-5, on `--sample` target neurons per probe (0 for all).
Exits 1 on any failure.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wiretype.data.labels import harmonise            # noqa: E402
from wiretype.data.scope import in_scope              # noqa: E402
from wiretype.eval.release import apply_probe, read_probes  # noqa: E402
from wiretype.eval.transfer import RELEASE_PROBES     # noqa: E402

RUNS = {"forward": ("fafb", "mcns", "transfer_fafb_to_mcns_checkpoint_fafb_displacement_refined_split_random_s24k_ntbest_source"),
        "reverse": ("mcns", "fafb", "transfer_mcns_to_fafb_checkpoint_mcns_displacement_refined_split_random_s24k_ntbest_brain_source")}
METRICS = ("macro_f1", "weighted_f1", "accuracy")


def main() -> None:
    """Run the four checks on both directions."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--release", type=Path, default=Path("experiments/release"))
    ap.add_argument("--paper", type=Path, default=Path("experiments/transfer"))
    ap.add_argument("--tol", type=float, default=1e-6, help="largest allowed score difference")
    ap.add_argument("--sample", type=int, default=20_000, help="target neurons per probe for check 4; 0 = all")
    args = ap.parse_args()

    with contextlib.redirect_stdout(io.StringIO()):
        nodes = {v: harmonise(pd.read_parquet(f"data/processed/{v}_nodes.parquet").sort_values("node_id")
                              .reset_index(drop=True), v) for v in ("fafb", "mcns")}
    failures = []
    for direction, (src, tgt, stem) in RUNS.items():
        print(f"== {direction}: {src} -> {tgt}")
        rel, pap = args.release / f"{stem}.json", args.paper / f"{stem}.json"
        if not rel.exists():
            failures.append(f"{direction}: missing {rel}")
            continue
        zr, zp = json.loads(rel.read_text())["zero_shot"], json.loads(pap.read_text())["zero_shot"]

        # 1. scores
        for target in RELEASE_PROBES:
            key = f"{target}/brain"
            for probe in ("linear", "knn"):
                for metric in METRICS:
                    a, b = zr[key][probe][metric], zp[key][probe][metric]
                    if abs(a - b) > args.tol:
                        failures.append(f"{direction} {key} {probe} {metric}: release {a:.6f}, paper {b:.6f}")
            print(f"  1. {key}: weighted F1 {zr[key]['linear']['weighted_f1']:.4f} / {zr[key]['knn']['weighted_f1']:.4f} "
                  f"(paper {zp[key]['linear']['weighted_f1']:.4f} / {zp[key]['knn']['weighted_f1']:.4f})")

        # 2. calls on the paper's rows
        cols = ["probe_target", "where", "node_id", "pred_linear", "pred_knn"]
        pr = pd.read_parquet(args.release / f"{stem}_predictions.parquet")
        pp = pd.read_parquet(args.paper / f"{stem}_predictions.parquet", columns=cols)
        pp = pp[pp.probe_target.isin(RELEASE_PROBES) & pp["where"].isin(["source_test", "target"])]
        both = pp.merge(pr[cols], on=["probe_target", "where", "node_id"], how="left", suffixes=("_paper", ""))
        missing = int(both.pred_linear.isna().sum())
        flips = both[(both.pred_linear != both.pred_linear_paper) | (both.pred_knn != both.pred_knn_paper)]
        print(f"  2. {len(both):,} paper rows: {missing} missing from the release run, {len(flips)} calls changed")
        if missing or len(flips):
            failures.append(f"{direction}: {missing} paper rows missing, {len(flips)} calls changed")
            print(flips.head(10).to_string())

        # 3. coverage: every connected target brain neuron once per probe, nothing else, and
        #    every call one of the probe's classes (guards against a corrupted copy)
        probes = read_probes(args.release / f"{stem}_probes.npz")
        connected = set(nodes[tgt].loc[in_scope(nodes[tgt], "brain_neurons"), "node_id"])
        for target in RELEASE_PROBES:
            called = pr[(pr.probe_target == target) & pr["where"].isin(["target", "target_unlabelled"])]
            dup, extra = int(called.node_id.duplicated().sum()), set(called.node_id) - connected
            absent = connected - set(called.node_id)
            known = set(probes[target]["classes"])
            odd = int((~called.pred_linear.isin(known) | ~called.pred_knn.isin(known)).sum())
            print(f"  3. {target}: {called.node_id.nunique():,} of {len(connected):,} connected {tgt} brain neurons "
                  f"called; {odd} calls outside the probe's classes")
            if dup or extra or absent or odd:
                failures.append(f"{direction} {target}: {dup} duplicates, {len(extra)} outside scope, "
                                f"{len(absent)} missing, {odd} calls not a class")

        # 4. the saved probes reproduce the release run
        emb = {v: np.load(args.release / f"{stem}_embeddings_{v}.npy", mmap_mode="r") for v in (src, tgt)}
        if emb[tgt].dtype != np.float32:
            failures.append(f"{direction}: embeddings are {emb[tgt].dtype}, the release needs float32")
        rng = np.random.default_rng(0)
        for target, probe in probes.items():
            called = pr[(pr.probe_target == target) & pr["where"].isin(["target", "target_unlabelled"])]
            if args.sample and len(called) > args.sample:
                called = called.iloc[np.sort(rng.choice(len(called), args.sample, replace=False))]
            again = apply_probe(probe, np.asarray(emb[src]), np.asarray(emb[tgt]), called.node_id.to_numpy())
            p_cols = [f"p_{c}" for c in probe["classes"]]
            gap = float(np.abs(again[p_cols].to_numpy() - called[p_cols].to_numpy()).max())
            lin = int((again.pred_linear.to_numpy() != called.pred_linear.to_numpy()).sum())
            knn = int((again.pred_knn.to_numpy() != called.pred_knn.to_numpy()).sum())
            print(f"  4. {target}: re-applied to {len(called):,} neurons: {lin} linear and {knn} k-NN calls "
                  f"differ; largest probability gap {gap:.2e}")
            if lin or knn or gap > 1e-5:
                failures.append(f"{direction} {target}: re-applied probe differs ({lin} linear, {knn} k-NN, gap {gap:.1e})")

    print()
    if failures:
        print(f"FAILED ({len(failures)}):")
        for f in failures:
            print("  " + f)
        sys.exit(1)
    print("OK: the release run matches the paper, covers every connected brain neuron, and its probes re-apply")


if __name__ == "__main__":
    main()
