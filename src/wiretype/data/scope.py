"""Which of a volume's cells take part when it is the *source* (the volume trained on).

`brain_neurons`, the default and the paper's setting, keeps the brain neurons with at
least one connection: a neuron without connections gives the model no wiring to read,
and FAFB has no counterpart for the MCNS nerve cord. `all` keeps every cell.

The scope works through the split: cells outside it are marked `excluded`, so
the scaler (fitted on `train`), the training pool, the probe's fit and tuning
(`train`, `val`) and the source's held-out score (`test`) all leave them out,
with nothing else to keep in step. The cells stay in the graph: their features
and top-k partners are unchanged, as they are when MCNS is the target.

The probes and scores apply `brain_neurons` to both volumes whatever the source's
scope (`wiretype.eval.transfer.zero_shot`), so no probe is fitted on, and no score
counts, a neuron without connections.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SCOPES = ("all", "brain_neurons")
EXCLUDED = "excluded"


def in_scope(nodes: pd.DataFrame, scope: str) -> np.ndarray:
    """Boolean mask, in node order, of the cells a source volume keeps."""
    if scope == "all":
        return np.ones(len(nodes), dtype=bool)
    if scope == "brain_neurons":
        return ((nodes["region"] == "brain") & ((nodes["n_in"] + nodes["n_out"]) > 0)).to_numpy()
    raise ValueError(f"unknown scope {scope!r}; choose from {SCOPES}")


def restrict_split(nodes: pd.DataFrame, split: pd.Series, scope: str) -> pd.Series:
    """The split with every out-of-scope cell marked `excluded`."""
    if scope == "all":
        return split
    return split.astype(object).where(in_scope(nodes, scope), EXCLUDED)
