#!/usr/bin/env python3
"""Supplementary file 1: the 47 candidate connectivity features, their definitions, and
which 31 the redundancy filter kept.

Names come from the code (`DEGREE_FEATURE_NAMES`, `connection_feature_names`, the
random-walk steps, `FLOW_COLUMNS`, the local-topology columns) and kept/dropped from the
frozen column set `refined.json`; the script stops if the two disagree, so the table
cannot drift from what the encoder reads. The filter was backward elimination: each step
dropped the feature best predicted by a linear regression on all remaining features,
until none reached R^2 >= 0.95 (`refined.json` `rule`); a dropped row gives its step and
that R^2. Definitions follow `wiretype.data.features`, `wiretype.data.flow`,
`scripts/rwse.py` and `scripts/local.py`.

    PYTHONPATH=src .venv/bin/python scripts/supplementary_file_1.py
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from wiretype.data.features import (COLUMN_SETS, DEGREE_FEATURE_NAMES, FLOW_COLUMNS,
                                    connection_feature_names)

SIDE = {"out": "output", "in": "input", "Out": "output", "In": "input"}
# out = connections the neuron sends; in = connections it receives
CONNECTION = {
    "fracW1": "share of the neuron's {side} connections that have exactly one synapse",
    "fracGeP75": "share of the neuron's {side} connections with at least as many synapses as the "
                 "volume's 75th percentile of connection weight (percentile over all connections "
                 "of that volume)",
    "fracGeP90": "as fracGeP75, at the volume's 90th percentile",
    "logMeanW": "log(1 + mean synapses per {side} connection)",
    "cv": "coefficient of variation (standard deviation / mean) of synapses per {side} connection",
    "herfindahl": "Herfindahl concentration of {side} synapses over connections: "
                  "sum of squared synapse counts / (total synapses)^2",
    "topShare": "strongest {side} connection's share of the neuron's {side} synapses",
    "meanLogW": "mean of log(1 + synapses) over {side} connections",
}
LOCAL = {
    "local_recip_frac_out": "share of output partners that also connect back to the neuron",
    "local_recip_frac_in": "share of input partners that the neuron also connects to",
    "local_recip_wfrac_out": "share of output synapses that go to reciprocated partners",
    "local_recip_wfrac_in": "share of input synapses that come from reciprocated partners",
    "local_log_kcore": "log(1 + core number) in the graph with direction and weights removed",
    "local_kcore_frac": "core number / number of partners (direction and weights removed)",
    "local_nbr_logdeg_out": "mean, over output partners, of the partner's log(1 + input partners + "
                            "output partners)",
    "local_nbr_logdeg_in": "mean, over input partners, of the partner's log(1 + input partners + "
                           "output partners)",
    "local_nbr_logdeg_out_sd": "standard deviation of the same over output partners",
    "local_nbr_logdeg_in_sd": "standard deviation of the same over input partners",
    "local_tri_cycle_frac": "directed 3-cycles through the neuron (neuron → j → k → neuron) / k(k − 1), "
                            "k = number of partners with direction removed",
    "local_tri_trans_frac": "feed-forward triangles in which the neuron is the source "
                            "(neuron → j → k and neuron → k) / k(k − 1)",
    "local_clust_coef": "clustering coefficient of the graph with direction removed: "
                        "2 × triangles / k(k − 1)",
    "local_log_tri_undirected": "log(1 + number of triangles through the neuron), direction removed",
}
FLOW = {
    "trophic_log1p": "trophic level h (MacKay et al., 2020): minimises sum_ij w_ij (h_j − h_i − 1)^2 / "
                     "sum_ij w_ij over connections i → j, w = log(1 + synapses); centred to zero mean "
                     "within each weakly connected component; 0 for neurons without connections",
    "springrank_log1p_a1": "SpringRank s (De Bacco et al., 2018) on the same weights with regularisation "
                           "alpha = 1; at alpha = 0 it equals −h exactly. Correlation with trophic level "
                           "r = −0.990 in FAFB, −0.999 in MCNS (experiments/features/flow_*.json)",
}
RWSE = ("probability that a random walk starting at the neuron is back at it after {k} steps; "
        "the walk ignores direction and steps to a partner with probability proportional to "
        "synapses; computed exactly by propagating blocks of unit vectors; entered as "
        "log(1 + 10^4 p). The one-step return is always 0 (no self-connections) and is not a candidate")


def rows() -> list[dict]:
    """One row per candidate feature, in the code's block order."""
    out = []
    for name in DEGREE_FEATURE_NAMES:
        if "@" in name:
            side, t = name[4:].split("@")
            d = f"log(1 + number of {SIDE[side]} partners joined by at least {t} synapse{'s' * (t != '1')})"
        else:
            d = f"log(1 + total {SIDE[name[6:]]} synapses)"
        out.append(("degree", name, d))
    for name in connection_feature_names():
        stat, side = name.rsplit("_", 1)
        out.append(("connection weight", name, CONNECTION[stat].format(side=SIDE[side])))
    for k in range(2, 9):
        out.append(("random-walk return", f"rwse_{k}", RWSE.format(k=k)))
    for name in FLOW_COLUMNS:
        out.append(("feed-forward hierarchy", name, FLOW[name]))
    for name, d in LOCAL.items():
        out.append(("local topology", name, d))
    return [{"family": f, "feature": n, "definition": d} for f, n, d in out]


def main() -> None:
    """Write the table and check it against the frozen column set."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("docs/supplementary/supplementary_file_1_features.tsv"))
    args = ap.parse_args()
    refined = json.loads((COLUMN_SETS / "refined.json").read_text())
    kept, dropped = set(refined["columns"]), set(refined["dropped"])
    table = rows()
    names = [r["feature"] for r in table]
    if len(names) != 47 or set(names) != kept | dropped or kept & dropped or len(kept) != 31:
        raise SystemExit(f"names disagree with refined.json: extra {set(names) - kept - dropped}, "
                         f"missing {(kept | dropped) - set(names)}")
    if list(refined["dropped_r2"]) != refined["dropped"]:
        raise SystemExit("refined.json: dropped_r2 does not follow the removal order in dropped")
    step = {name: i + 1 for i, name in enumerate(refined["dropped"])}
    for r in table:
        name = r["feature"]
        r["kept"] = ("yes" if name in kept else
                     f"no: removed at step {step[name]} of {len(step)}, "
                     f"R² = {refined['dropped_r2'][name]:.4f} from the remaining features")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["family", "feature", "kept", "definition"], delimiter="\t")
        w.writeheader()
        w.writerows(table)
    by = {}
    for r in table:
        by.setdefault(r["family"], [0, 0])[r["kept"] == "yes"] += 1
    print(f"{args.out}: {len(table)} features, {len(kept)} kept; "
          + ", ".join(f"{f} {k + d} ({k} kept)" for f, (d, k) in by.items()))


if __name__ == "__main__":
    main()
