"""Train/val/test assignment, and a measurement of what a per-cell split leaks.

Two splits are written for every volume:

- **`split_random`, per cell: the paper's protocol.** It lets every type be seen
  in training. Inside a volume it leaks: most test cells have a same-type sibling
  (often the mirror-image cell) in training, and some labels are assigned per
  type in the source papers. That is why in-volume numbers are diagnostics only.
  Zero-shot transfer has no such leak, because no target cell is in any training
  set.
- **`split`, type-blocked: a check that the transmitter result survives unseen
  types.** Whole `cell_type` groups go to one split, balanced on `nt_best`.
  Blocking on the label itself would make the task impossible (the probe could
  never emit a held-out class); `cell_type` is one level finer, and a
  transmitter class spans about 1,100 types. Anatomy targets are not evaluated
  under this split.

`leak_audit` measures the leak without a model: the share of test cells with a
same-type or same-hemilineage neighbour in training.

**The paper's splits are the released file, not this code.** The paper's FAFB splits
(released as `fafb_splits`) were drawn with `stratify_on="nt_train"`, the predicted label
alone, by an earlier version of this code. The default is now `("nt_best", "super_class")`,
so `assign_splits` does not reproduce the paper's type-blocked split, and the earlier
version does not run on the current node table (328 neurons lack a predicted label). To
reproduce the paper, use the released split.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SPLITS = ("train", "val", "test")
DEFAULT_FRACTIONS = (0.70, 0.15, 0.15)
GROUP_COLUMN = "cell_type"


def _groups(nodes: pd.DataFrame, group_column: str) -> pd.Series:
    """Group key per neuron, with un-typed neurons in singleton groups.

    1.1% of FAFB neurons carry no `cell_type`. Putting them all in one shared
    group would bind thousands of unrelated cells to a single split; giving each
    its own group lets them distribute freely, which is right, because a neuron
    with no type has no sibling to leak to.
    """
    key = nodes[group_column].astype("object")
    missing = key.isna()
    return key.where(~missing, "__ungrouped_" + nodes.node_id.astype(str)).astype(str)


def assign_splits(
    nodes: pd.DataFrame,
    stratify_on: str | tuple[str, ...] = ("nt_best", "super_class"),
    group_column: str = GROUP_COLUMN,
    fractions: tuple[float, float, float] = DEFAULT_FRACTIONS,
    seed: int = 0,
) -> pd.Series:
    """Assign every neuron to 'train', 'val' or 'test', keeping groups intact.

    Groups are dealt largest-first to whichever split is furthest below its quota
    for that group's dominant label — a greedy longest-processing-time heuristic.
    Largest-first matters: the type-size distribution is extreme (231 types hold
    73.5% of all neurons), so dealing the big optic families at the end would
    leave no room to balance them.

    `stratify_on` may name several columns, which are balanced jointly. The
    default balances transmitter within each super-class. Balancing transmitter
    alone let the big optic types pile into train (FAFB's test set was 5.5% optic,
    and MCNS's val set had no endocrine cells), because nothing asked for optic
    cells in test.

    Other targets are not stratified, and `label_coverage` reports any class that
    a split happens to miss.
    """
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError(f"fractions must sum to 1, got {fractions}")

    rng = np.random.default_rng(seed)
    group = _groups(nodes, group_column)
    columns = [stratify_on] if isinstance(stratify_on, str) else list(stratify_on)
    label = nodes[columns[0]].astype("object")
    if len(columns) > 1:
        # One stratum per combination. A missing value is its own stratum rather
        # than a reason to drop the cell from the balance.
        label = nodes[columns].astype(str).agg("|".join, axis=1)

    frame = pd.DataFrame({"group": group, "label": label})
    # Dominant label per group, and group size. Groups are usually label-pure —
    # a cell type has one transmitter — so the mode is not a lossy summary here.
    summary = frame.groupby("group", sort=False).agg(
        size=("label", "size"),
        label=("label", lambda s: s.dropna().mode().iat[0] if s.notna().any() else None),
    )
    # Shuffle before the stable size sort, so that equal-sized groups — and there
    # are thousands of two-member types — are not dealt in table order.
    summary = summary.iloc[rng.permutation(len(summary))]
    summary = summary.sort_values("size", ascending=False, kind="stable")

    targets = {s: f for s, f in zip(SPLITS, fractions)}
    total_by_label: dict[object, float] = summary.groupby("label", dropna=False)["size"].sum().to_dict()
    allocated: dict[tuple[str, object], float] = {(s, l): 0.0 for s in SPLITS for l in total_by_label}

    assignment: dict[str, str] = {}
    for group_key, row in summary.iterrows():
        lab, size = row["label"], row["size"]
        # Deficit = how far this split still is from its quota for this label.
        deficit = {s: targets[s] * total_by_label[lab] - allocated[(s, lab)] for s in SPLITS}
        chosen = max(SPLITS, key=lambda s: deficit[s])
        assignment[str(group_key)] = chosen
        allocated[(chosen, lab)] += size

    return group.map(assignment).rename("split")


def random_splits(
    nodes: pd.DataFrame, fractions: tuple[float, float, float] = DEFAULT_FRACTIONS, seed: int = 0
) -> pd.Series:
    """The naive per-neuron split, for the comparison the thesis has to report."""
    rng = np.random.default_rng(seed)
    draw = rng.random(len(nodes))
    train, val = fractions[0], fractions[0] + fractions[1]
    return pd.Series(
        np.where(draw < train, "train", np.where(draw < val, "val", "test")),
        index=nodes.index,
        name="split",
    )


def leak_audit(nodes: pd.DataFrame, split: pd.Series, sibling_columns=("cell_type", "hemilineage")) -> dict:
    """How many test neurons have a sibling sitting in the training set.

    This is the leak, measured without a model: for each grouping, the share of
    test neurons whose group also appears in train. Under a type-blocked split it
    is 0 for `cell_type` by construction.

    Reported next to the same quantity under a random split, where it is the size
    of the advantage a naive protocol would have handed the model.

    `side` must be in the harmonised spelling (`labels.harmonise`): MCNS spells
    it L/R, which matches no twin and reports NaN.
    """
    out: dict[str, float] = {}
    is_test, is_train = split == "test", split == "train"
    out["n_train"] = int(is_train.sum())
    out["n_val"] = int((split == "val").sum())
    out["n_test"] = int(is_test.sum())

    for column in sibling_columns:
        if column not in nodes:
            continue
        values = nodes[column]
        train_groups = set(values[is_train].dropna().unique())
        tested = values[is_test].dropna()
        share = float(tested.isin(train_groups).mean()) if len(tested) else float("nan")
        out[f"test_with_{column}_sibling_in_train"] = share
        out[f"n_test_labelled_{column}"] = int(len(tested))

    # The sharpest form: a bilateral twin. Types with exactly one neuron per side.
    if {"cell_type", "side"} <= set(nodes.columns):
        pairs = nodes[nodes.cell_type.notna() & nodes.side.isin(["left", "right"])]
        sizes = pairs.groupby("cell_type").size()
        twin_types = set(sizes[sizes == 2].index)
        twins = pairs[pairs.cell_type.isin(twin_types)]
        twin_test = twins[split.reindex(twins.index) == "test"]
        twin_train_types = set(twins[split.reindex(twins.index) == "train"].cell_type)
        out["n_bilateral_twin_neurons"] = int(len(twins))
        out["test_twins_with_partner_in_train"] = (
            float(twin_test.cell_type.isin(twin_train_types).mean()) if len(twin_test) else float("nan")
        )
    return out


def label_coverage(nodes: pd.DataFrame, split: pd.Series, targets=("nt_best", "super_class")) -> dict:
    """Classes present in train but missing from val or test, per target.

    A class absent from test is simply unscored; a class absent from train cannot
    be predicted at all. Both are consequences of blocking on a grouping rather
    than on the label, both are expected to be rare, and both have to be declared
    rather than discovered in the results table.
    """
    out: dict[str, dict] = {}
    for target in targets:
        if target not in nodes:
            continue
        present = {s: set(nodes.loc[split == s, target].dropna().unique()) for s in SPLITS}
        every = set().union(*present.values())
        out[target] = {
            "n_classes": len(every),
            "missing_from_train": sorted(map(str, every - present["train"])),
            "missing_from_val": sorted(map(str, every - present["val"])),
            "missing_from_test": sorted(map(str, every - present["test"])),
        }
    return out
