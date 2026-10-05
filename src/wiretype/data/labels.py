"""Deriving a usable label from what the annotation tables actually say.

Two of the label columns need real work rather than a rename, and both are
places where a careless read changes the label space:

`known_nt` is free text with two levels of separator. Semicolons separate
*sources* (different papers reporting on the same neuron), commas separate
*co-transmitters* within one source. It also carries explicit negatives
(`gaba-negative`, 4,913 of them), neuropeptides (sNPF, tachykinin, MIP, Dh44)
and nitric oxide. So `"acetylcholine; sNPF; acetylcholine, sNPF"` is one
cholinergic neuron that also releases a peptide, reported twice — not three
labels, and not an ambiguous case.

`ito_lee_hemilineage` carries primary-neuron placeholders — `putative_primary`,
or a `__prim` suffix — which are not hemilineages. They are 10.7% of the column
and dropping them is what takes 194 strings down to the 173 usable classes.
"""

from __future__ import annotations

from .schema import NON_TRANSMITTER_TOKENS, OUT_OF_SCOPE_TRANSMITTERS, TRANSMITTERS

_TRANSMITTERS = frozenset(TRANSMITTERS)
_OUT_OF_SCOPE = frozenset(OUT_OF_SCOPE_TRANSMITTERS)
_IGNORED = frozenset(NON_TRANSMITTER_TOKENS)


def parse_known_nt(text: str | None) -> tuple[str | None, frozenset[str], bool]:
    """Read one `known_nt` cell.

    Returns `(unambiguous, mentioned, clean)`:

        unambiguous   the single transmitter this neuron can be said to use, or
                      None if the sources name none, name more than one, or name
                      only an out-of-scope transmitter such as tyramine.
        mentioned     every canonical transmitter named anywhere in the cell,
                      including in cells that are ambiguous overall.
        clean         True only when every source named the same single
                      transmitter and nothing else — no co-transmitter, no
                      peptide, no out-of-scope token, no disagreement.

    The two differ for co-transmitting or disputed neurons, and the difference
    matters: `mentioned` is what decides the histamine override for the training
    label (see `neurotransmitter_labels`), while `unambiguous` is the stricter
    quantity used for the literature evaluation tier, where a neuron nobody
    agrees about should simply be left out.

    `clean` exists because agreement between this column and `top_nt` depends
    sharply on which neurons are admitted. Scoring only the clean subset measures
    agreement on the easy cases; scoring everything admits co-transmitting and
    disputed neurons, which is where a predictor and the literature part company.
    Both are legitimate, so intake reports both and the gap between them is
    itself a statement about label quality.

    Unrecognised tokens are returned in neither set, and the caller is expected
    to count them — a token nobody anticipated should surface as unparsed rather
    than be silently discarded.
    """
    if not text or not isinstance(text, str):
        return None, frozenset(), False

    found: set[str] = set()
    out_of_scope = False
    impure = False
    for source in text.split(";"):
        for raw in source.split(","):
            token = raw.strip().lower()
            if not token:
                continue
            if token.endswith("-negative") or token in _IGNORED:
                impure = True
                continue
            if token in _TRANSMITTERS:
                found.add(token)
            elif token in _OUT_OF_SCOPE:
                impure = True
                # A real transmitter the label space cannot express. It makes the
                # cell unlabelable rather than being ignored, so that the neuron
                # is dropped instead of falling back to a prediction known to be
                # wrong. See schema.OUT_OF_SCOPE_TRANSMITTERS.
                out_of_scope = True

    if out_of_scope and not found:
        return None, frozenset(), False
    if len(found) == 1:
        return next(iter(found)), frozenset(found), not impure
    return None, frozenset(found), False


def unrecognised_tokens(text: str | None) -> set[str]:
    """Tokens in a `known_nt` cell that are neither a transmitter nor a known
    non-transmitter. Used to audit the vocabulary rather than in the pipeline."""
    if not text or not isinstance(text, str):
        return set()
    out = set()
    for source in text.split(";"):
        for raw in source.split(","):
            token = raw.strip().lower()
            if not token or token.endswith("-negative"):
                continue
            if token not in _TRANSMITTERS and token not in _OUT_OF_SCOPE and token not in _IGNORED:
                out.add(token)
    return out


def clean_hemilineage(text: str | None) -> str | None:
    """Drop the primary-neuron placeholders, keep everything else.

    `putative_primary` is a statement that the neuron is primary, not a
    hemilineage name; a `__prim` suffix marks the primary neurons *of* a
    hemilineage, which is a different grouping from the hemilineage itself.
    """
    if not text or not isinstance(text, str):
        return None
    value = text.strip()
    if not value or value == "putative_primary" or value.endswith("__prim"):
        return None
    return value


# MCNS spells side L/M/R where FAFB spells it left/center/right. FAFB's `na` has no
# MCNS counterpart and is left as it is.
SIDE_MAP = {"L": "left", "R": "right", "M": "center"}


def harmonise(nodes, volume: str):
    """Make label vocabularies comparable across volumes, and derive `nt_best`.

    `nt_best`, the training label, is the experimental label (`nt_known`) where one
    exists and the predicted one (`nt_train`) otherwise. In FAFB it replaces the
    predicted label for whole types where the two disagree: Kenyon cells predicted
    dopamine, ORNs predicted serotonin, L1 predicted GABA (8,708 cells). In MCNS the two never disagree, so
    there it equals `nt_train`.
    """
    from ..log import log

    nodes = nodes.assign(nt_best=nodes["nt_known"].where(nodes["nt_known"].notna(), nodes["nt_train"]))
    if "side" in nodes.columns:
        hits = int(nodes["side"].isin(SIDE_MAP).sum())
        if hits:
            nodes = nodes.assign(side=nodes["side"].replace(SIDE_MAP))
            log(f"  {volume}: side recoded L/M/R -> left/center/right for {hits:,} cells")
    return nodes
