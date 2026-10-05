"""Structural node features: the only thing a cell arrives with.

The graph is featureless by premise, so whatever stands in for features is
computed from the wiring, identically in every volume. There is no neuron
identity, coordinate, soma, morphology or per-node learned embedding: a new volume
is a different node set, and a lookup table cannot survive the move.

The blocks, before the column set is applied:

    8   degree                  how big a cell's weight profile is
    16  connection-distribution what shape that profile is
    7   RWSE                    whether a weighted walk comes back (scripts/rwse.py)
    2   flow                    where the cell sits in the whole graph (scripts/flow.py)
    14  local                   reciprocity, core number, partner degree, triangles
                                (scripts/local.py)

The encoder reads the `refined` column set (`column_sets/refined.json`): 31 of
these 47, with redundant columns removed.

`rwse_1` is always dropped. A one-step return needs an autapse, which neither
connectome has, so the column is identically zero.

`wiring_features` builds the encoder's input for a volume; `standardise_with`
scales it and returns the statistics so that they can be applied to another
volume unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..log import log

THRESHOLDS = (1, 3, 5)
N_DEGREE_FEATURES = 2 * len(THRESHOLDS) + 2

# **Out before in, because that is the order the columns are built in.**
# `degree_features` iterates `(pre[keep], post[keep])`, and a cell counted as
# `pre` is the one sending the connection — so column 0 is out-degree. The names
# used to be generated as `for d in ("Din", "Dout")`, which labelled all eight
# columns with the opposite quantity: every name said in where the column held
# out and vice versa. Nothing trained was affected, because the encoder reads
# columns and never names, but the labels reached `checkpoint["feature_names"]`,
# `transfer.py`'s mismatch error and M20's correlation table, where an inference
# was drawn from them.
#
# Generated once here rather than rebuilt at each call site, so a name and the
# column it describes cannot drift apart again.
DEGREE_FEATURE_NAMES = tuple(
    f"log{direction}@{threshold}"
    for threshold in THRESHOLDS
    for direction in ("Dout", "Din")
) + ("logSynOut", "logSynIn")

# Per side, in construction order. `connection_features` groups by `pre` first,
# which describes a cell's outgoing connections, then by `post`.
CONNECTION_STATS = ("logMeanW", "cv", "herfindahl", "topShare", "meanLogW")


# The two flow columns, in the order flow_hierarchy.py writes them.
FLOW_COLUMNS = ("trophic_log1p", "springrank_log1p_a1")

# A one-step return needs an autapse; neither volume has one, so rwse_1 is a
# column of zeros. Dropped by name rather than by testing for constancy, so that
# both volumes get the identical vector whatever their reconstruction contains.
RWSE_DROP = ("rwse_1",)


def connection_feature_names(percentiles=(75, 90)) -> tuple[str, ...]:
    """The 16 connection-distribution names, in the order the block builds them.

    Takes `percentiles` so the names cannot claim a cut the block did not use;
    they were previously `conn0` to `conn15`, which could not be wrong but also
    said nothing about what the sixteen numbers were.
    """
    stats = ("fracW1",) + tuple(f"fracGeP{p}" for p in percentiles) + CONNECTION_STATS
    return tuple(f"{stat}_{side}" for side in ("out", "in") for stat in stats)


def effective_rank(features: np.ndarray) -> float:
    """How many directions a feature matrix actually carries.

    exp of the entropy of the normalised covariance spectrum — the same
    quantity `eval/collapse.py` computes on embeddings, applied to the input
    instead. It exists because nobody checked it before the first training run
    and the answer was **1.74 of 8** (M16): every pairwise correlation among the
    degree features is 0.69-0.96, so thresholding degree at 1, 3 and 5 measures
    the same thing three times. An encoder cannot represent more directions than
    its input carries, so this is the ceiling on everything downstream and it
    should be read before a run, not after one collapses.
    """
    centred = features - features.mean(axis=0, keepdims=True)
    eigenvalues = np.linalg.eigvalsh(np.cov(centred.T)).clip(min=0.0)
    total = eigenvalues.sum()
    if total <= 0:
        return 0.0
    spectrum = eigenvalues / total
    entropy = -(spectrum * np.log(np.maximum(spectrum, 1e-12))).sum()
    return float(np.exp(entropy))


def degree_features(edges: pd.DataFrame, n_nodes: int) -> np.ndarray:
    """(n_nodes, 8): log degrees at three weight thresholds, plus synapse totals.

    The threshold sweep lives *inside* the feature rather than being applied to
    the graph. That is the point: it hands the encoder what it needs to discount
    weak connections without anyone hard-coding the discount, which would
    hand-solve the question the thesis is asking.
    """
    pre, post, w = edges.pre.to_numpy(), edges.post.to_numpy(), edges.w.to_numpy()
    columns = []
    for threshold in THRESHOLDS:
        keep = w >= threshold
        for endpoints in (pre[keep], post[keep]):
            counts = np.bincount(endpoints, minlength=n_nodes).astype(np.float32)
            columns.append(np.log1p(counts))
    for endpoints in (pre, post):
        totals = np.bincount(endpoints, weights=w, minlength=n_nodes).astype(np.float32)
        columns.append(np.log1p(totals))
    return np.stack(columns, axis=1)


def _require(processed, volume: str, kind: str, how: str) -> pd.DataFrame:
    """Load a precomputed block, or stop with the command that builds it.

    **Nothing here falls back to a shorter vector.** M18's accident was a block
    joining itself in silently whenever its file happened to exist; the mirror
    image — a run quietly training on 24 columns because a parquet was missing —
    is just as unfalsifiable after the fact and just as expensive. A missing
    input is a failure at startup, not a different experiment.
    """
    path = Path(processed) / f"{volume}_{kind}.parquet"
    if not path.exists():
        raise SystemExit(
            f"{path} is missing and the v2 feature vector requires it.\n"
            f"Build it first:  {how}\n"
            "Pass rwse=False / flow=False only to reproduce a v1.0 arm deliberately."
        )
    return pd.read_parquet(path).sort_values("node_id")


def feature_summary(names) -> str:
    """The line every trainer prints at startup, and the check that catches a
    mismatched vector before a job spends an hour on it.

    Reading it is the habit that would have caught M18's silent RWSE join and
    M21's swapped names. It states the width and every block by name, so a run
    on the wrong vector is visible in the first line of its log.
    """
    known = set(DEGREE_FEATURE_NAMES) | set(connection_feature_names())
    n_base = sum(n in known for n in names)
    n_rwse = sum(n.startswith("rwse_") for n in names)
    n_flow = sum(n in FLOW_COLUMNS for n in names)
    parts = [f"{n_base} degree+connection",
             f"{n_rwse} RWSE" if n_rwse else "NO RWSE",
             f"{n_flow} flow" if n_flow else "NO flow"]
    # **Counted, not inferred by exclusion.** This used to report everything that
    # was neither RWSE nor flow as "degree+connection", so a v4 line appending its
    # own block saw `22 node features - 22 degree+connection` for 22 partner-
    # composition columns. This line exists to make a wrong vector visible in the
    # first line of a log; a category that absorbs anything unrecognised cannot do
    # that. Extra blocks are now named by their own prefix.
    other = [n for n in names if n not in known
             and not n.startswith("rwse_") and n not in FLOW_COLUMNS]
    if other:
        groups: dict[str, int] = {}
        for n in other:
            groups[n.split("_")[0]] = groups.get(n.split("_")[0], 0) + 1
        parts += [f"{c} {prefix}" for prefix, c in sorted(groups.items())]
    return f"{len(names)} node features — " + ", ".join(parts)


def load_features(processed, volume: str, edges, n_nodes: int,
                  rwse: bool = True, flow: bool = True):
    """The node feature matrix. Returns `(features, names)`.

    **Both blocks default ON, which is the v2 vector and reverses M18.** A caller
    that wants v1.0's 24 columns — to reproduce an arm in `experiments/` as it was
    measured — has to ask for it with `rwse=False, flow=False`, and should say so
    in its report. The defaults changed once before, in the other direction and by
    accident, and cost a supervised run that was comparable to nothing.

    Requires `{volume}_rwse.parquet` and `{volume}_flow.parquet`. Missing files
    stop the run rather than shortening the vector.
    """
    block = np.hstack([degree_features(edges, n_nodes), connection_features(edges, n_nodes)])
    names = list(DEGREE_FEATURE_NAMES) + list(connection_feature_names())
    if len(names) != block.shape[1]:
        raise AssertionError(f"{len(names)} names for {block.shape[1]} columns")

    if rwse:
        frame = _require(processed, volume, "rwse", f"qsub -v VOLUME={volume} hpc/jobs/rwse.sh")
        columns = [c for c in frame.columns if c.startswith("rwse_") and c not in RWSE_DROP]
        # Return probabilities span orders of magnitude and are mostly tiny, so
        # they enter on a log scale for the same reason synapse counts do.
        block = np.hstack([block, np.log1p(frame[columns].to_numpy(np.float64) * 1e4).astype(np.float32)])
        names += columns

    if flow:
        frame = _require(processed, volume, "flow", f"qsub -v VOLUME={volume} hpc/jobs/flow.sh")
        block = np.hstack([block, frame[list(FLOW_COLUMNS)].to_numpy(np.float32)])
        names += list(FLOW_COLUMNS)

    return block, names


def connection_features(edges: pd.DataFrame, n_nodes: int, percentiles=(75, 90)) -> np.ndarray:
    """Describe a cell by the *shape* of its connection-weight distribution.

    The degree features answer "how connected am I" eight times over (together
    about 1.7 effective directions in FAFB). These answer how that connectivity is
    *distributed*, and every one is degree-normalised, so they add directions
    rather than restating one: effective rank 6.33 on their own in FAFB.

    The shape carries biology: the single-synapse fraction of a cell's inputs runs
    36% for motor cells and 67% for sensory ones (FAFB medians; for outputs 85% and
    43%; experiments/features/connection_shape.json).

    **Thresholds are percentile-matched, not absolute.** MCNS is reconstructed to
    a denser standard (24.4% of its connected pairs have w >= 5, against FAFB's
    17.9%), so a fixed cut means something different in each brain. Percentiles computed per volume mean the same thing in both.
    Note the degree features do *not* yet do this and carry the same latent
    transfer problem at w >= 1, 3, 5.
    """
    w = edges.w.to_numpy(np.float64)
    cuts = [np.percentile(w, p) for p in percentiles]
    columns = []
    for side in ("pre", "post"):
        key = edges[side].to_numpy()
        count = np.bincount(key, minlength=n_nodes).astype(np.float64)
        safe = np.maximum(count, 1.0)
        total = np.bincount(key, weights=w, minlength=n_nodes)

        # w == 1 stays absolute: a single-synapse connection is a specific
        # biological claim -- the 42%-reproducible class -- not a percentile.
        columns.append(np.bincount(key, weights=(w == 1).astype(float), minlength=n_nodes) / safe)
        for cut in cuts:
            columns.append(np.bincount(key, weights=(w >= cut).astype(float), minlength=n_nodes) / safe)

        mean = total / safe
        second = np.bincount(key, weights=w ** 2, minlength=n_nodes) / safe
        spread = np.sqrt(np.maximum(second - mean ** 2, 0.0))
        strongest = np.zeros(n_nodes)
        np.maximum.at(strongest, key, w)
        columns += [
            np.log1p(mean),                                   # typical strength
            spread / np.maximum(mean, 1e-9),                  # dispersion
            np.bincount(key, weights=w ** 2, minlength=n_nodes) / np.maximum(total, 1e-9) ** 2,
            strongest / np.maximum(total, 1e-9),              # strongest partner's share
            np.bincount(key, weights=np.log1p(w), minlength=n_nodes) / safe,
        ]
    return np.stack(columns, axis=1).astype(np.float32)


def standardise(features: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Zero mean, unit variance, with statistics taken over `mask` only.

    The mask exists so that the statistics can be computed on training cells
    alone. It matters less here than it would for a learned feature — these are
    deterministic functions of the graph, which pretraining sees in full anyway
    (§6, transductive) — but taking them from the training split keeps one rule
    for every scaling decision in the project rather than two.
    """
    subset = features if mask is None else features[mask]
    mean = subset.mean(axis=0, keepdims=True)
    std = subset.std(axis=0, keepdims=True)
    return ((features - mean) / np.maximum(std, 1e-6)).astype(np.float32)


def standardise_with(raw: np.ndarray, mask: np.ndarray | None, stats=None):
    """`standardise`, returning its statistics so that they can travel to another volume.

    With `stats` given, those are applied instead of being fitted: this is how the
    target volume arrives in the source's scale (`source` scaling).
    """
    if stats is None:
        subset = raw if mask is None else raw[mask]
        stats = (subset.mean(axis=0, keepdims=True),
                 np.maximum(subset.std(axis=0, keepdims=True), 1e-6))
    mean, std = stats
    return ((raw - mean) / std).astype(np.float32), stats


# ---------------------------------------------------------------- the encoder's input

LOCAL_PREFIX = "local_"
# Built by scripts/local.py as two jobs: the cheap group and the triangle group.
# Read in this order, which is the order the development runs read them in.
LOCAL_FILES = ("local", "local_tri")
COLUMN_SETS = Path(__file__).resolve().parent / "column_sets"

BLOCKS = {
    "local": lambda n: n.startswith(LOCAL_PREFIX),
    "degree": lambda n: n in set(DEGREE_FEATURE_NAMES),
    "connection": lambda n: n in set(connection_feature_names()),
    "rwse": lambda n: n.startswith("rwse_"),
    "flow": lambda n: n in set(FLOW_COLUMNS),
}
DEFAULT_BLOCKS = "degree,connection,rwse,flow,local"


def attach_local(processed, volume: str, raw: np.ndarray, names: list):
    """Append the local-topology columns from `<volume>_local.parquet` and `_local_tri.parquet`."""
    cols, added = [], []
    for kind in LOCAL_FILES:
        frame = _require(processed, volume, kind,
                         f"qsub -v VOLUME={volume},GROUPS={'triangles,SUFFIX=_tri' if kind == 'local_tri' else 'cheap'} "
                         f"hpc/jobs/local.sh")
        if len(frame) != raw.shape[0]:
            raise SystemExit(f"{volume}_{kind}.parquet has {len(frame):,} rows for {raw.shape[0]:,} cells")
        new = [c for c in frame.columns if c.startswith(LOCAL_PREFIX) and c not in cols]
        cols += new
        added.append(frame[new].to_numpy(np.float32))
        log(f"  {volume}_{kind}.parquet: {len(new)} columns")
    return np.hstack([raw] + added), names + cols


def select_blocks(raw: np.ndarray, names: list, wanted):
    """Keep only the named blocks, in their original column order."""
    unknown = set(wanted) - set(BLOCKS)
    if unknown:
        raise SystemExit(f"unknown feature blocks: {sorted(unknown)}. choose from {sorted(BLOCKS)}")
    tests = [BLOCKS[b] for b in wanted]
    keep = [i for i, n in enumerate(names) if any(t(n) for t in tests)]
    if not keep:
        raise SystemExit(f"feature blocks {','.join(wanted)} selected no columns")
    return raw[:, keep], [names[i] for i in keep]


def column_set(name: str) -> list[str]:
    """The frozen column list `name` (e.g. `refined`, the 31 columns the encoder reads)."""
    path = COLUMN_SETS / f"{name}.json"
    if not path.exists():
        raise SystemExit(f"no column set {name!r}; have {sorted(p.stem for p in COLUMN_SETS.glob('*.json'))}")
    return json.loads(path.read_text())["columns"]


def wiring_features(processed, volume: str, edges: pd.DataFrame, n_nodes: int,
                    blocks: str = DEFAULT_BLOCKS, columns: str = "refined"):
    """The encoder's input for `volume`: the named blocks, then the column set. Raw, unscaled."""
    raw, names = load_features(processed, volume, edges, n_nodes)
    wanted_blocks = [b.strip() for b in blocks.split(",") if b.strip()]
    if "local" in wanted_blocks:
        raw, names = attach_local(processed, volume, raw, names)
    if set(wanted_blocks) != set(BLOCKS):
        raw, names = select_blocks(raw, names, wanted_blocks)
    wanted = column_set(columns)
    missing = [c for c in wanted if c not in names]
    if missing:
        raise SystemExit(f"{volume} is missing {len(missing)} of the {columns} columns: "
                         f"{missing[:6]}{'…' if len(missing) > 6 else ''}. Build the feature "
                         f"tables first (README, 'Pipeline').")
    keep = [i for i, n in enumerate(names) if n in set(wanted)]
    return raw[:, keep], [names[i] for i in keep]
