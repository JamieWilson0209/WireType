"""Read both volumes into the one node/edge shape defined in `schema.py`.

Build-order step 1. Everything downstream — samplers, encoders, probes — sees
the output of this module and never the raw tables, so the volume-specific
awkwardness is confined here.

Three rules are enforced rather than documented, because each is a place the
project could go wrong silently:

1. **The six per-edge neurotransmitter probability columns are never read.**
   They come from the same Eckstein model that produced the label, averaged over
   a neuron's own incident edges, so feeding them to an encoder would make the
   headline task circular in the most direct way available. They are excluded by
   column selection at the pyarrow level, not dropped after loading — the bytes
   never enter the process.
2. **The label space is seven classes.** Histamine is added from `known_nt`,
   because `top_nt` has six outputs and cannot emit it. See `labels.py`.
3. **Neurons with no edges are not seeds.** The encoder consumes a
   neighbourhood and 623 annotated FAFB neurons have none. They stay in the node
   table, so the count is visible, and are flagged out of the seed set.

This is a cluster job. The FAFB edge table is 852 MB on disk and aggregating it
builds intermediates several times that; `hpc/jobs/intake.sh` has the sizing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
import pyarrow.feather as feather

from ..log import log
from .labels import clean_hemilineage, parse_known_nt, unrecognised_tokens
from .schema import EDGE_COLUMNS, MCNS_TO_FAFB_SUPER_CLASS, NODE_COLUMNS

# Read only these. The six `*_avg` neurotransmitter probability columns and the
# per-connection `neuropil` are deliberately absent: the first is rule 1 above,
# the second is aggregated away by definition of the graph: one edge per
# connected pair, summed over neuropils.
FAFB_EDGE_COLUMNS = ["pre_pt_root_id", "post_pt_root_id", "syn_count"]
MCNS_EDGE_COLUMNS = ["body_pre", "body_post", "weight"]

SIX_PREDICTED = ("acetylcholine", "glutamate", "gaba", "dopamine", "serotonin", "octopamine")


def _aggregate_edges(frame: pd.DataFrame, already_unique: bool) -> pd.DataFrame:
    """One row per directed pair, summing synapse counts over whatever split the
    source table used (neuropil, on the FAFB side).

    `already_unique` skips the group-by for a table that is one row per pair
    already, but verifies the claim rather than trusting it — a silent duplicate
    would double a weight and there would be no way to notice downstream.
    """
    frame.columns = ["pre", "post", "w"]
    if already_unique:
        n_unique = len(frame.drop_duplicates(subset=["pre", "post"]))
        if n_unique == len(frame):
            return frame
        log(f"  table claimed one row per pair but has {len(frame) - n_unique:,} duplicates; aggregating")
    grouped: pd.DataFrame = frame.groupby(["pre", "post"], sort=False, as_index=False)["w"].sum()
    return grouped


def _index_nodes(edges: pd.DataFrame, annotated: "npt.ArrayLike") -> pd.Series:
    """Map every identifier to a contiguous `node_id`.

    The node set is the union of everything in the edge list and everything
    carrying an annotation, so that edgeless annotated neurons keep a row and
    can be counted rather than vanishing.
    """
    present = pd.unique(
        np.concatenate([edges.pre.values, edges.post.values, np.asarray(annotated)])
    )
    return pd.Series(np.arange(len(present), dtype=np.int32), index=present)


def _degree_table(edges: pd.DataFrame, n_nodes: int) -> pd.DataFrame:
    """Partner counts and synapse totals per neuron, both directions.

    Computed here because the aggregation pass already has the edge list in
    memory and a second pass over 15.1M rows later would be pure waste. These
    are also the raw material for the structural node features of §2.
    """
    out = np.zeros(n_nodes, dtype=np.int32)
    inn = np.zeros(n_nodes, dtype=np.int32)
    syn_out = np.zeros(n_nodes, dtype=np.int64)
    syn_in = np.zeros(n_nodes, dtype=np.int64)
    np.add.at(out, edges.pre.values, 1)
    np.add.at(inn, edges.post.values, 1)
    np.add.at(syn_out, edges.pre.values, edges.w.values)
    np.add.at(syn_in, edges.post.values, edges.w.values)
    return pd.DataFrame({"n_in": inn, "n_out": out, "syn_in": syn_in, "syn_out": syn_out})


def _neurotransmitter(annotations: pd.DataFrame, report: dict) -> pd.DataFrame:
    """The seven-class training label, the literature label, and their provenance.

    `nt_train` is `top_nt` everywhere except the histaminergic neurons, which
    `top_nt` cannot express and therefore labels wrong with certainty. Those take
    the literature value instead, and their `nt_conf` is NaN because no predictor
    confidence applies to a label the predictor did not produce.

    A neuron whose only named transmitter is out of scope (tyramine, glycine)
    gets no training label at all rather than falling back to a guess that is
    known to be wrong — the same argument as histamine, applied consistently.
    """
    parsed = annotations["known_nt"].apply(parse_known_nt)
    nt_known = pd.Series([p[0] for p in parsed], index=annotations.index, dtype="object")
    mentions_histamine = pd.Series([("histamine" in p[1]) for p in parsed], index=annotations.index)
    is_clean = pd.Series([p[2] for p in parsed], index=annotations.index)

    # Out of scope and nothing else named: the literature says something the
    # label space cannot hold, so the predicted value must not stand in for it.
    named_nothing_usable = pd.Series(
        [(p[0] is None and not p[1]) for p in parsed], index=annotations.index
    )
    out_of_scope = annotations["known_nt"].notna() & named_nothing_usable & annotations[
        "known_nt"
    ].str.contains("tyramine|glycine", case=False, na=False)

    nt_train = annotations["top_nt"].astype("object").where(~mentions_histamine, "histamine")
    nt_train = nt_train.where(~out_of_scope, None)
    # `top_nt` is absent for 602 neurons and pandas spells that NaN while the
    # two masks above spell it None. Normalise, so that a null label has one
    # representation and `notna()` below means what it says.
    nt_train = nt_train.where(nt_train.notna(), None)

    source = pd.Series("predicted", index=annotations.index, dtype="object")
    source = source.where(~mentions_histamine, "known")
    source = source.where(nt_train.notna(), None)

    # Confidence belongs to a prediction. It is dropped for the histamine
    # neurons, whose label the predictor did not produce, and for the
    # out-of-scope neurons, which have no training label at all — carrying a
    # confidence for a label that is not there would be read as calibration
    # information by anything downstream that filters on it.
    conf = annotations["top_nt_conf"].astype("float32")
    conf = conf.where(nt_train.notna() & ~mentions_histamine, np.nan)

    report["nt_histamine_from_known"] = int(mentions_histamine.sum())
    report["nt_histamine_unambiguous"] = int((nt_known == "histamine").sum())
    report["nt_dropped_out_of_scope"] = int(out_of_scope.sum())
    report["nt_known_unambiguous"] = int(nt_known.notna().sum())
    report["nt_known_clean_subset"] = int((nt_known.notna() & is_clean).sum())
    return pd.DataFrame(
        {
            "nt_train": nt_train,
            "nt_train_source": source,
            "nt_conf": conf,
            "nt_known": nt_known,
            "nt_known_clean": is_clean & nt_known.notna(),
        }
    )


def load_fafb(annotations_path: Path, connections_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """FAFB v783 into the canonical shape. Returns (nodes, edges, report)."""
    report: dict = {"volume": "fafb"}

    # Edges: one row per connected pair, synapse counts summed over neuropils
    log(f"reading {connections_path.name}")
    raw = feather.read_table(connections_path, columns=FAFB_EDGE_COLUMNS, memory_map=True).to_pandas()
    report["edge_rows_raw"] = len(raw)
    edges = _aggregate_edges(raw, already_unique=False)
    del raw
    log(f"  {len(edges):,} directed pairs, {int(edges.w.sum()):,} synapses")

    # Annotations, and a count of any `known_nt` tokens the parser does not recognise
    log(f"reading {annotations_path.name}")
    ann = pd.read_csv(annotations_path, sep="\t", low_memory=False)
    report["annotation_rows"] = len(ann)

    audit: dict[str, int] = {}
    for text in ann["known_nt"].dropna().unique():
        for token in unrecognised_tokens(text):
            audit[token] = audit.get(token, 0) + 1
    report["unrecognised_nt_tokens"] = audit

    # Renumber neurons 0..n-1 (annotated neurons plus any edge endpoint) and remap the edges
    index = _index_nodes(edges, ann["root_id"].values)
    edges["pre"] = index[edges.pre.values].values
    edges["post"] = index[edges.post.values].values
    edges = edges.astype({"pre": np.int32, "post": np.int32, "w": np.int32})

    n_nodes = len(index)
    nodes = pd.DataFrame({"node_id": np.arange(n_nodes, dtype=np.int32), "source_id": index.index.values})
    nodes = nodes.join(_degree_table(edges, n_nodes))

    # Transmitter labels (predicted, confident) from the annotation table (_neurotransmitter)
    ann = ann.assign(node_id=index[ann["root_id"].values].values)
    ann = ann.set_index("node_id")
    nt = _neurotransmitter(ann, report)

    def col(name: str) -> pd.Series:
        """One annotation column in node order; all-missing if the release lacks it."""
        if name not in ann.columns:
            return pd.Series([None] * n_nodes)
        column: pd.Series = ann[name]
        return column.reindex(nodes.node_id.values).reset_index(drop=True)

    # The canonical node columns. FAFB is a brain, so every neuron is region `brain`;
    # hemilineage placeholders (putative_primary, __prim) are dropped by clean_hemilineage.
    nodes["volume"] = "fafb"
    nodes["region"] = "brain"
    for name in ("nt_train", "nt_train_source", "nt_conf", "nt_known", "nt_known_clean"):
        nodes[name] = nt[name].reindex(nodes.node_id.values).reset_index(drop=True)
    nodes["super_class"] = col("super_class")
    nodes["super_class_raw"] = nodes["super_class"]
    nodes["cell_class"] = col("cell_class")
    nodes["supertype"] = col("supertype")
    nodes["cell_type"] = col("cell_type")
    nodes["hemibrain_type"] = col("hemibrain_type")
    nodes["fbbt_id"] = col("fbbt_id")
    nodes["hemilineage"] = col("ito_lee_hemilineage").apply(clean_hemilineage)
    nodes["side"] = col("side")
    nodes["status"] = None

    # is_seed: at least one edge. Only these neurons are training seeds (scripts/train.py);
    # edgeless neurons keep their row
    nodes["is_seed"] = (nodes.n_in + nodes.n_out) > 0
    report["n_nodes"] = int(n_nodes)
    report["n_edges"] = int(len(edges))
    report["n_synapses"] = int(edges.w.sum())
    report["n_edgeless_annotated"] = int((~nodes.is_seed).sum())
    report["n_seeds"] = int(nodes.is_seed.sum())
    return nodes, edges, report


def load_mcns(
    annotations_path: Path, connections_path: Path, neurotransmitters_path: Path
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """MCNS v1.0 into the canonical shape. Returns (nodes, edges, report).

    Two differences from FAFB that are handled here and nowhere else: the
    super-class vocabulary is region-prefixed and 26-valued, so it is mapped
    through `schema.MCNS_TO_FAFB_SUPER_CLASS` with `None` meaning *no FAFB
    counterpart exists*; and the animal has a ventral nerve cord, flagged in
    `region` so that the VNC can be reported as its own transfer tier.
    """
    report: dict = {"volume": "mcns"}

    # Edges: the release has one row per connected pair (checked; duplicates would be summed)
    log(f"reading {connections_path.name}")
    raw = feather.read_table(connections_path, columns=MCNS_EDGE_COLUMNS, memory_map=True).to_pandas()
    report["edge_rows_raw"] = len(raw)
    edges = _aggregate_edges(raw, already_unique=True)
    del raw
    log(f"  {len(edges):,} directed pairs, {int(edges.w.sum()):,} synapses")

    # Annotations joined to the transmitter table. `consensus_nt` is MCNS's per-type
    # image prediction, replaced by the experimental label where one exists (Berg et al. 2026);
    # `ground_truth` is the experimental label alone.
    log(f"reading {annotations_path.name}")
    ann = pd.read_feather(annotations_path)
    body_col = "bodyId" if "bodyId" in ann.columns else ann.columns[0]
    nt_table = pd.read_feather(neurotransmitters_path, columns=["body", "consensus_nt", "ground_truth"])
    ann = ann.merge(nt_table, left_on=body_col, right_on="body", how="left")
    report["annotation_rows"] = len(ann)

    # Renumber neurons 0..n-1 and remap the edges, as for FAFB
    index = _index_nodes(edges, ann[body_col].values)
    edges["pre"] = index[edges.pre.values].values
    edges["post"] = index[edges.post.values].values
    edges = edges.astype({"pre": np.int32, "post": np.int32, "w": np.int32})

    n_nodes = len(index)
    nodes = pd.DataFrame({"node_id": np.arange(n_nodes, dtype=np.int32), "source_id": index.index.values})
    nodes = nodes.join(_degree_table(edges, n_nodes))

    ann = ann.assign(node_id=index[ann[body_col].values].values).set_index("node_id")

    def col(name: str) -> pd.Series:
        """One annotation column in node order; all-missing if the release lacks it."""
        if name not in ann.columns:
            return pd.Series([None] * n_nodes)
        column: pd.Series = ann[name]
        return column.reindex(nodes.node_id.values).reset_index(drop=True)

    # Every MCNS super-class must have an explicit entry in the FAFB mapping
    raw_super = col("superclass")
    unknown = set(raw_super.dropna().unique()) - set(MCNS_TO_FAFB_SUPER_CLASS)
    if unknown:
        raise ValueError(
            f"MCNS super-classes absent from the hand-written map: {sorted(unknown)}. "
            "Add them to schema.MCNS_TO_FAFB_SUPER_CLASS with an explicit decision — "
            "mapping by guess would be scored as a model error."
        )

    # `unclear` is MCNS's explicit reject class and FAFB has no counterpart, so it
    # is dropped to null rather than becoming an extra class.
    consensus = col("consensus_nt").replace("unclear", None)
    truth = col("ground_truth").replace("unclear", None)

    # The canonical node columns. The nerve cord is every neuron whose super-class starts
    # with `vnc`; everything else is the brain. nt_train (predicted) = consensus_nt,
    # nt_known (confident) = ground_truth; cell_type is the FlyWire-matched type.
    nodes["volume"] = "mcns"
    nodes["region"] = np.where(raw_super.fillna("").str.startswith("vnc"), "vnc", "brain")
    nodes["nt_train"] = consensus
    nodes["nt_train_source"] = np.where(consensus.notna(), "predicted", None)
    nodes["nt_conf"] = np.nan
    nodes["nt_known"] = truth
    nodes["nt_known_clean"] = truth.notna()
    nodes["super_class"] = raw_super.map(MCNS_TO_FAFB_SUPER_CLASS)
    nodes["super_class_raw"] = raw_super
    nodes["cell_class"] = None
    nodes["supertype"] = None
    nodes["cell_type"] = col("flywireType")
    nodes["hemibrain_type"] = col("type")
    nodes["fbbt_id"] = None
    nodes["hemilineage"] = col("itoleeHl").apply(clean_hemilineage)
    nodes["side"] = col("somaSide")
    nodes["status"] = col("statusLabel")

    # is_seed: at least one edge. Only these neurons are training seeds (scripts/train.py);
    # edgeless neurons keep their row
    nodes["is_seed"] = (nodes.n_in + nodes.n_out) > 0
    report["n_nodes"] = int(n_nodes)
    report["n_edges"] = int(len(edges))
    report["n_synapses"] = int(edges.w.sum())
    report["n_edgeless_annotated"] = int((~nodes.is_seed).sum())
    report["n_seeds"] = int(nodes.is_seed.sum())
    report["n_vnc"] = int((nodes.region == "vnc").sum())
    report["n_superclass_unmapped"] = int(raw_super.notna().sum() - nodes.super_class.notna().sum())
    return nodes, edges, report


def check_schema(nodes: pd.DataFrame, edges: pd.DataFrame) -> None:
    """Assert the canonical shape rather than trusting it."""
    missing = set(NODE_COLUMNS) - set(nodes.columns)
    if missing:
        raise ValueError(f"node table is missing canonical columns: {sorted(missing)}")
    if list(edges.columns) != list(EDGE_COLUMNS):
        raise ValueError(f"edge table columns are {list(edges.columns)}, expected {list(EDGE_COLUMNS)}")
    if not nodes.node_id.is_unique:
        raise ValueError("node_id is not unique")
    n = len(nodes)
    if edges.pre.max() >= n or edges.post.max() >= n or edges.pre.min() < 0:
        raise ValueError("edge endpoints fall outside the node table")
    if (edges.w <= 0).any():
        raise ValueError("edge list contains non-positive synapse counts")
