"""The one shape both volumes are read into, and the vocabularies that define it.

FAFB v783 and MCNS v1.0 disagree about almost everything at the surface — column
names, identifier types, how many super-classes there are, whether the animal has
a ventral nerve cord. Everything downstream sees the canonical shape below
instead, so no model or probe ever contains a branch on which volume it is
looking at.

Two tables per volume:

    nodes   one row per neuron, indexed by a contiguous `node_id` that is also
            the row/column index of the edge list. Carries the labels, the
            provenance of each label, and the flags that decide what is eligible
            to be a training seed.
    edges   one row per directed pair, `(pre, post, w)` with `w` the synapse
            count. Already aggregated over neuropil on the FAFB side.

The vocabularies here are reference data as much as code: the transmitter list
and the super-class map are both things the thesis has to state, so they are
written once, here, with the reasoning attached.
"""

from __future__ import annotations

# --- neurotransmitters -------------------------------------------------------

# Seven, not six. The Eckstein model that produced FAFB's `top_nt` has six
# outputs and cannot emit histamine, so it assigns all 11,129 histaminergic
# neurons — 65.8% of the whole `sensory` super-class — to a class that is wrong
# with certainty rather than with noise. Histamine labels come from annotation.
TRANSMITTERS = (
    "acetylcholine",
    "glutamate",
    "gaba",
    "dopamine",
    "serotonin",
    "octopamine",
    "histamine",
)

# Genuine fast transmitters that are nonetheless NOT in the label space, and the
# reason is transfer rather than biology. Tyramine is named in `known_nt` for 946
# neurons — more than octopamine's 216 — and it is in exactly the position
# histamine was in: `top_nt` has no output for it, so those neurons are labelled
# wrong with certainty. The difference is that MCNS's vocabulary has no tyramine
# either, so promoting it to an eighth class would label FAFB neurons that the
# transfer benchmark could never score.
#
# So a neuron whose only named transmitter is out-of-scope gets **no** training
# label rather than `top_nt`'s guess. That is the histamine argument applied
# consistently: better to drop 946 neurons than to train on 946 known-wrong ones.
OUT_OF_SCOPE_TRANSMITTERS = ("tyramine", "glycine")

# Tokens that appear inside FAFB's free-text `known_nt` and are *not* a fast
# transmitter at all. Neuropeptides and gases co-released with one; recording
# them would be a different project. Listed rather than inferred so that a token
# nobody anticipated shows up as unparsed instead of being silently dropped —
# `labels.unrecognised_tokens` audits this and intake reports what it finds.
NON_TRANSMITTER_TOKENS = (
    "snpf", "dnpf", "npf", "tachykinin", "mip", "dh44", "dh31", "dh", "itp",
    "nitric oxide", "no", "proctolin", "allatostatin", "allatostatin-a",
    "allatostatin-c", "corazonin", "hugin", "leucokinin", "sifamide",
    "myosuppressin", "myosupressin", "fmrfamide", "fmrfa", "pdf", "ilp2",
    "dilp3", "nplp1", "cnma", "drosulfakinin", "eclosion hormone", "capability",
    "space blanket", "unknown", "unclear", "none",
)

# --- super-classes -----------------------------------------------------------

# FAFB's ten, flat.
FAFB_SUPER_CLASSES = (
    "optic", "central", "sensory", "visual_projection", "ascending",
    "descending", "sensory_ascending", "visual_centrifugal", "motor", "endocrine",
)

# MCNS is a whole central nervous system and FAFB is a brain, so MCNS
# super-classes are region-prefixed and there are 26 of them. This map is a
# hand-written judgement call, reported in full in the paper's methods.
#
# `None` means *no FAFB counterpart exists* — not "unmapped". Those bodies are
# excluded from super-class transfer rather than forced into a nearest class,
# because a wrong mapping would be scored as a model error. Everything in the
# ventral nerve cord is in that position by construction, which is why the VNC
# is a neurotransmitter-only tier: transmitter identity is defined everywhere,
# super-class and hemilineage are not.
#
# The `_tbc` suffix is MCNS's "to be confirmed". Those groups are tiny (2–38
# bodies) and are mapped to their base class rather than excluded, so that they
# do not leave a hole; the flag survives in `super_class_raw` if it ever matters.
MCNS_TO_FAFB_SUPER_CLASS: dict[str, str | None] = {
    # --- brain, direct counterparts ---
    "ol_intrinsic": "optic",
    "cb_intrinsic": "central",
    "ol_sensory": "sensory",            # photoreceptors; FAFB calls these sensory too
    "cb_sensory": "sensory",
    "cb_sensory_tbc": "sensory",
    "visual_projection": "visual_projection",
    "visual_projection_tbc": "visual_projection",
    "visual_centrifugal": "visual_centrifugal",
    "ascending_neuron": "ascending",
    "descending_neuron": "descending",
    "descending_neuron_tbc": "descending",
    "sensory_ascending": "sensory_ascending",
    "sensory_ascending_tbc": "sensory_ascending",
    "cb_motor": "motor",
    "cb_endocrine": "endocrine",
    # --- brain, no clean counterpart ---
    # FAFB has no `efferent` category and no `sensory_descending`. Folding them
    # into motor/descending would be a guess scored as model error, so they are
    # excluded from super-class transfer and kept for the other targets.
    "cb_efferent": None,
    "sensory_descending": None,
    "efferent_ascending": None,
    "efferent_descending": None,
    "ENS": None,                        # enteric nervous system; absent from FAFB
    # --- ventral nerve cord: absent from FAFB entirely ---
    "vnc_intrinsic": None,
    "vnc_sensory": None,
    "vnc_sensory_tbc": None,
    "vnc_motor": None,
    "vnc_efferent": None,
    "vnc_endocrine": None,
    "vnc_tbc": None,
}

# --- canonical columns -------------------------------------------------------

# Written out so that a reader can see the whole shape in one place, and so that
# both loaders can be asserted against it rather than trusted.
NODE_COLUMNS = (
    "node_id",          # int32, contiguous, indexes the edge list
    "source_id",        # int64, the volume's own identifier (root_id / bodyId)
    "volume",           # 'fafb' | 'mcns'
    "region",           # 'brain' | 'vnc'  — MCNS only has both
    "nt_train",         # 7-class label used for training. Predicted, except histamine
    "nt_train_source",  # 'predicted' | 'known'  — provenance of nt_train, per neuron
    "nt_conf",          # float32, the predictor's confidence; NaN where source is 'known'
    "nt_known",         # 7-class literature label where unambiguous, else null
    "super_class",      # harmonised to FAFB's ten; null where MCNS has no counterpart
    "super_class_raw",  # what the volume actually said
    "cell_class",       # 49 classes, FAFB only
    "supertype",        # 1,898 classes, FAFB only
    "cell_type",        # 8,840 FAFB / `flywireType` on MCNS — the shared vocabulary
    "hemibrain_type",   # defined on a different volume; a circularity control
    "fbbt_id",          # FlyBase ontology term; the other circularity control
    "hemilineage",      # 173 usable classes, `__prim` placeholders removed
    "side",             # 'left' | 'right' | 'center'
    "status",           # MCNS `statusLabel`; transfer numbers stratify on it
    "n_in", "n_out",    # int32 partner counts, unweighted
    "syn_in", "syn_out",# int32 synapse totals
    "is_seed",          # bool — False for neurons with no edges at all
)

EDGE_COLUMNS = ("pre", "post", "w")
