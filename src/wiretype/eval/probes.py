"""Reading a label off a frozen representation, and scoring it.

Every row of the results (the encoder, its untrained floor, every baseline) is
scored by this code, so the numbers differ only because the representations do.

Two probes, because they fail differently:

    linear    multinomial logistic regression, the standard read-out.
    k-NN      no fitted parameters. A partially collapsed representation can still
              support a linear probe through a few surviving directions; k-NN
              notices. k-NN above linear means the information is present but not
              linearly organised.

Rules enforced here rather than left to the caller:
1. **Never bare accuracy.** Acetylcholine is most cells, so always answering it
   scores well. Macro-F1 (every class equal) and support-weighted F1 (the F1 of a
   random cell) lead, with per-class recall and F1 beside them.
2. **Hyperparameters are chosen on validation, never on test.** Test is scored
   once, with the setting validation picked.
3. **Classes absent from training are dropped, and the drop is reported.** A class
   the probe never saw cannot be predicted.
4. **Scopes are reported separately.** The optic lobe is over half the graph, so
   a whole-brain average is largely an average over the medulla; central brain is
   reported beside it.

Class weighting is `balanced` by default. With octopamine at a few hundred cells
against acetylcholine's tens of thousands, an unweighted fit never emits the rare
classes. It also optimises macro-F1 at the common classes' expense, so read
weighted F1 from an unweighted probe (`class_weight=None`) too.

Paper terms: these are the probes (linear and k-NN); a cell is a neuron.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# Swept on validation. Small grids on purpose: these are floors and baselines,
# and an extravagant search on the baseline arm would flatter it relative to the
# arms that come later.
LOGISTIC_C = (0.01, 0.1, 1.0, 10.0)
KNN_K = (1, 5, 15, 50)

# Raised from 200 after the first learned-representation runs hit the cap. The
# floors did not: every SVD fit in M14 converged in 35-194 iterations, because
# singular vectors are orthogonal by construction and stay well-conditioned after
# scaling. A learned encoder's dimensions are correlated, which elongates the
# loss surface and slows lbfgs — and a fit that stops early *understates* the
# score, so leaving the cap at 200 would have handed the floors a converged fit
# and the trained arms a truncated one. That is a bias in the one comparison the
# project exists to make.
#
# Non-convergence is still reported per fit rather than assumed away: a probe
# that needs more than this is saying something about the representation.
MAX_ITER = 1000
TOLERANCE = 1e-4

SCOPES = {
    "whole_brain": None,
    "central_brain": ("super_class", "central"),
    # Added after M14. The k-NN probe beat the linear one on neurotransmitter
    # whole-brain at every width, and lost to it central-brain from width 256 —
    # opposite verdicts from the same representation. The optic lobe is 56% of
    # the graph and is built from columnar families repeating near-identically
    # across hundreds of columns, which is exactly the regime a nearest-neighbour
    # probe wins in and a linear boundary does not care about. Scoring it on its
    # own turns that explanation from a story into a measurement.
    "optic_lobe": ("super_class", "optic"),
}


def label_strings(values: pd.Series) -> pd.Series:
    """Labels as strings, missing kept missing.

    `supertype` is stored as float IDs (13390.0), and an object array of floats is
    type "unknown" to sklearn, which then refuses to score it against a float
    prediction. Job 58639614 died on exactly that in its first probe.
    """
    return values.map(
        lambda v: v if isinstance(v, str) or pd.isna(v)
        else str(int(v)) if float(v).is_integer() else str(v)
    ).astype("object")


def _scope_mask(nodes: pd.DataFrame, scope: str) -> pd.Series:
    """The rows inside a named scope (for example `central_brain`), as defined in
    SCOPES.
    """
    rule = SCOPES[scope]
    if rule is None:
        return pd.Series(True, index=nodes.index)
    column, value = rule
    return nodes[column] == value


def _score(y_true: np.ndarray, y_pred: np.ndarray, classes: np.ndarray) -> dict:
    """Macro-F1 first, then the per-class recalls that make it interpretable.

    `macro_f1` averages over the *probe's* vocabulary, so a class the probe can
    emit but that has no instance in this evaluation set scores F1 = 0 and drags
    the mean down. That is the right denominator when comparing arms on a fixed
    vocabulary, and the wrong one when reading a tier's absolute difficulty: the
    MCNS VNC carries no dopamine and the central brain no histamine, which costs
    macro-F1 there on nothing but absence.
    `macro_f1_present` restricts the average to the classes actually present, and
    `classes_absent` names what the two figures differ by. Neither replaces the
    other and both are reported.
    """
    per_class = recall_score(y_true, y_pred, labels=classes, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, labels=classes, average=None, zero_division=0)
    support = np.array([(y_true == c).sum() for c in classes])
    present = support > 0
    return {
        "macro_f1": float(f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0)),
        "macro_f1_present": float(np.mean(f1_per_class[present])) if present.any() else 0.0,
        # Per-class F1 weighted by each class's share of the evaluated cells: the F1
        # expected for a cell drawn at random. Macro-F1 answers "how good is it on
        # every class"; this answers "how good is it on the cells an annotator meets".
        "weighted_f1": float(np.sum(f1_per_class * support) / support.sum()) if support.sum() else 0.0,
        "n_classes_present": int(present.sum()),
        "classes_absent": [str(c) for c in classes[~present]],
        "balanced_accuracy": float(np.mean(per_class)),
        "balanced_accuracy_present": float(np.mean(per_class[present])) if present.any() else 0.0,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "per_class_recall": {str(c): float(r) for c, r in zip(classes, per_class)},
        "per_class_f1": {str(c): float(f) for c, f in zip(classes, f1_per_class)},
        "support": {str(c): int(v) for c, v in zip(classes, support)},
    }


def probe(
    embeddings: np.ndarray,
    nodes: pd.DataFrame,
    split: pd.Series,
    target: str,
    scope: str = "whole_brain",
    class_weight: str | None = "balanced",
    seed: int = 0,
) -> dict:
    """Fit both probes on train, choose settings on val, score once on test.

    `embeddings` is (n_nodes, n_dims), row-aligned with `nodes`. Returns a nested
    dict of metrics plus the bookkeeping needed to read them: how many neurons
    each split contributed, and which classes were dropped for being absent from
    train.
    """
    usable = nodes[target].notna() & _scope_mask(nodes, scope)
    labels = label_strings(nodes[target])

    parts = {}
    for name in ("train", "val", "test"):
        parts[name] = np.flatnonzero((split == name).to_numpy() & usable.to_numpy())
    if min(len(v) for v in parts.values()) == 0:
        return {"error": f"a split is empty for target={target} scope={scope}"}

    train_classes = np.array(sorted(set(labels.iloc[parts["train"]])))
    if len(train_classes) < 2:
        # Degenerate rather than broken. `super_class` inside the central-brain
        # scope is the standing example: the scope is *defined* by that label, so
        # there is one class and nothing to predict. Reported rather than raised,
        # so the combination stays visible in the results table as a thing that
        # was considered and is meaningless, instead of silently missing.
        return {
            "target": target,
            "scope": scope,
            "skipped": f"only {len(train_classes)} class present in train",
            "n_train": int(len(parts["train"])),
        }
    dropped = {}
    for name in ("val", "test"):
        present = set(labels.iloc[parts[name]])
        missing = sorted(str(c) for c in present - set(train_classes))
        keep = np.array([labels.iat[i] in set(train_classes) for i in parts[name]])
        dropped[name] = {
            "classes_absent_from_train": missing,
            "neurons_dropped": int((~keep).sum()),
        }
        parts[name] = parts[name][keep]

    # Standardise on train statistics only. Logistic regression is scale
    # sensitive and the SVD arms in particular arrive with wildly unequal
    # component variances, so skipping this would make C mean something
    # different in every arm.
    scaler = StandardScaler().fit(embeddings[parts["train"]])
    x = {name: scaler.transform(embeddings[idx]) for name, idx in parts.items()}
    y = {name: labels.iloc[idx].to_numpy() for name, idx in parts.items()}

    results: dict = {
        "target": target,
        "scope": scope,
        "n_dims": int(embeddings.shape[1]),
        "n_classes": int(len(train_classes)),
        "n_train": int(len(parts["train"])),
        "n_val": int(len(parts["val"])),
        "n_test": int(len(parts["test"])),
        "dropped": dropped,
    }

    # Two reference points scored the same way, so that a macro-F1 is readable
    # without going and looking up the class priors. `majority` is what a model
    # that has learned nothing but the most common class scores — 62% accuracy
    # on neurotransmitter, and a macro-F1 far below it, which is the whole
    # reason bare accuracy is banned above. `stratified` draws from the training
    # priors and is the chance level for macro-F1 specifically.
    rng = np.random.default_rng(seed)
    majority = train_classes[np.argmax([(y["train"] == c).sum() for c in train_classes])]
    priors = np.array([(y["train"] == c).mean() for c in train_classes])
    results["baselines"] = {
        "majority": _score(y["test"], np.full(len(y["test"]), majority), train_classes),
        "stratified": _score(
            y["test"], rng.choice(train_classes, size=len(y["test"]), p=priors), train_classes
        ),
    }

    best_c, best_val = None, -1.0
    for c in LOGISTIC_C:
        model = LogisticRegression(
            C=c, max_iter=MAX_ITER, tol=TOLERANCE, class_weight=class_weight, random_state=seed
        ).fit(x["train"], y["train"])
        score = f1_score(y["val"], model.predict(x["val"]), labels=train_classes,
                         average="macro", zero_division=0)
        if score > best_val:
            best_c, best_val = c, score
    model = LogisticRegression(
        C=best_c, max_iter=MAX_ITER, tol=TOLERANCE, class_weight=class_weight, random_state=seed
    ).fit(x["train"], y["train"])
    results["linear"] = {"C": best_c, "val_macro_f1": float(best_val),
                         "converged": bool(np.all(model.n_iter_ < MAX_ITER)),
                         "n_iter": int(np.max(model.n_iter_)),
                         **_score(y["test"], model.predict(x["test"]), train_classes)}

    # One neighbour search per split, not one per k. The searches dominate the
    # cost of this whole module at the widths being swept — 97k training
    # neurons against 2,048 dimensions — and every k in the grid is answerable
    # from a single top-max(k) query, so refitting per k would be paying four
    # times for the same distances.
    candidate_k = [k for k in KNN_K if k < len(parts["train"])]
    searcher = NearestNeighbors(n_neighbors=max(candidate_k), n_jobs=-1).fit(x["train"])

    # Vote by accumulating class counts over the neighbour window, rather than
    # constructing a pandas Series per row: hemilineage has 173 classes and tens
    # of thousands of query neurons, and the per-row version was a measurable
    # share of the job's wall clock.
    code_of = {c: i for i, c in enumerate(train_classes)}
    train_codes = np.array([code_of[c] for c in y["train"]], dtype=np.int32)

    def vote(neighbour_codes: np.ndarray, k: int) -> np.ndarray:
        """Majority vote among each row's k nearest training neurons; a tie goes to the
        first class.
        """
        counts = np.zeros((len(neighbour_codes), len(train_classes)), dtype=np.int32)
        rows = np.repeat(np.arange(len(neighbour_codes)), k)
        np.add.at(counts, (rows, neighbour_codes[:, :k].ravel()), 1)
        return train_classes[counts.argmax(axis=1)]

    val_neighbours = train_codes[searcher.kneighbors(x["val"], return_distance=False)]
    best_k, best_val = None, -1.0
    for k in candidate_k:
        score = f1_score(y["val"], vote(val_neighbours, k), labels=train_classes,
                         average="macro", zero_division=0)
        if score > best_val:
            best_k, best_val = k, score

    test_neighbours = train_codes[searcher.kneighbors(x["test"], return_distance=False)]
    results["knn"] = {"k": best_k, "val_macro_f1": float(best_val),
                      **_score(y["test"], vote(test_neighbours, best_k), train_classes)}

    return results


def probe_all(
    embeddings: np.ndarray,
    nodes: pd.DataFrame,
    split: pd.Series,
    targets=("nt_train", "hemilineage", "super_class"),
    scopes=("whole_brain", "central_brain"),
    **kwargs,
) -> dict:
    """Every target at every scope. Cell type is absent by design — §6.1."""
    out: dict = {}
    for target in targets:
        if target not in nodes.columns:
            continue
        for scope in scopes:
            out[f"{target}/{scope}"] = probe(embeddings, nodes, split, target, scope, **kwargs)
    return out


def _confidence(proba: np.ndarray, y: np.ndarray, classes: np.ndarray) -> dict:
    """How far to trust the probe's probabilities: a summary for the JSON report.

    Accepting only predictions whose top probability clears a threshold trades
    coverage for precision; `coverage_at_precision` is the largest share of cells
    that can be accepted while staying at or above each precision. Calibration
    error (ECE) is the mean gap between stated confidence and actual accuracy
    over ten equal-width confidence bins, weighted by the cells in each.
    """
    top = proba.max(axis=1)
    right = classes[proba.argmax(axis=1)] == y
    order = np.argsort(-top)
    running = np.cumsum(right[order]) / np.arange(1, len(right) + 1)
    coverage = {}
    for goal in (0.80, 0.90, 0.95, 0.99):
        ok = np.flatnonzero(running >= goal)
        coverage[str(goal)] = {"coverage": float((ok.max() + 1) / len(right)) if len(ok) else 0.0,
                               "threshold": float(top[order][ok.max()]) if len(ok) else None}
    bins = np.minimum((top * 10).astype(int), 9)
    ece = sum(abs(right[bins == b].mean() - top[bins == b].mean()) * (bins == b).mean()
              for b in range(10) if (bins == b).any())
    code = {c: i for i, c in enumerate(classes)}
    p_true = proba[np.arange(len(y)), [code[c] for c in y]]
    return {"mean_top_p_correct": float(top[right].mean()) if right.any() else None,
            "mean_top_p_wrong": float(top[~right].mean()) if (~right).any() else None,
            "mean_p_true_class": float(p_true.mean()),
            "per_class_mean_p_true": {c: float(p_true[y == c].mean()) for c in classes if (y == c).any()},
            "ece": float(ece), "coverage_at_precision": coverage}


def probes_fit_apply(
    source_embeddings: np.ndarray,
    source_nodes: pd.DataFrame,
    source_split: pd.Series,
    target_embeddings: np.ndarray,
    target_nodes: pd.DataFrame,
    target_mask: np.ndarray,
    target: str,
    scope: str = "whole_brain",
    class_weight: str | None = "balanced",
    seed: int = 0,
    return_predictions: bool = False,
    unlabelled_mask: np.ndarray | None = None,
    return_probe: bool = False,
) -> dict:
    """Fit a probe on one volume and apply it, unchanged, to another.

    The zero-shot transfer readout. The encoder is already frozen; this freezes
    the probe too — fitted on the source's training split, tuned on the source's
    validation split, and then applied to the target volume without refitting,
    rescaling to target statistics, or seeing a single target label.

    Two decisions that make it honest rather than flattering:

    **The scaler is fitted on the source and applied to the target.** Standardising
    the target embeddings with the target's own mean and variance would be a form
    of unsupervised adaptation — cheap, label-free, and no longer zero-shot. The
    probe's weights live in the source's scaled space and the target has to
    arrive in that space.

    **Classes absent from the source's training split are dropped**, and the count
    reported. A probe cannot emit a class it never saw, and charging it for one
    would measure the label vocabularies rather than the representation.

    `source_linear_macro_f1` comes back alongside, scored on the source's own
    held-out test split by the identical fitted probe — so the transfer number
    always has its within-volume reference next to it, and the drop between them
    is the quantity of interest.

    With `return_probe`, `result["probe"]` holds the fitted probe as plain arrays, for
    release: the scaler, the logistic regression and the k-NN reference set (rows of
    the source and their class codes). Not JSON: the caller pops it.
    """
    src_label = label_strings(source_nodes[target])
    tgt_label = label_strings(target_nodes[target])

    usable_src = src_label.notna() & _scope_mask(source_nodes, scope)
    train = np.flatnonzero((source_split == "train").to_numpy() & usable_src.to_numpy())
    val = np.flatnonzero((source_split == "val").to_numpy() & usable_src.to_numpy())
    test = np.flatnonzero((source_split == "test").to_numpy() & usable_src.to_numpy())
    if min(len(train), len(val), len(test)) == 0:
        return {"skipped": f"source split empty for {target}/{scope}"}

    classes = np.array(sorted(set(src_label.iloc[train])))
    if len(classes) < 2:
        return {"skipped": f"only {len(classes)} class in the source training split"}

    known = set(classes)
    tgt_usable = np.flatnonzero(target_mask & tgt_label.notna().to_numpy())
    keep = np.array([tgt_label.iat[i] in known for i in tgt_usable], dtype=bool)
    dropped = int((~keep).sum())
    tgt_usable = tgt_usable[keep]
    if len(tgt_usable) == 0:
        return {"skipped": f"no target cells carry a class the source trained on"}

    scaler = StandardScaler().fit(source_embeddings[train])
    x_train = scaler.transform(source_embeddings[train])
    x_val = scaler.transform(source_embeddings[val])
    x_test = scaler.transform(source_embeddings[test])
    x_tgt = scaler.transform(target_embeddings[tgt_usable])
    y_train = src_label.iloc[train].to_numpy()
    y_val = src_label.iloc[val].to_numpy()
    y_test = src_label.iloc[test].to_numpy()
    y_tgt = tgt_label.iloc[tgt_usable].to_numpy()

    best_c, best_val = None, -1.0
    for c in LOGISTIC_C:
        model = LogisticRegression(C=c, max_iter=MAX_ITER, tol=TOLERANCE,
                                   class_weight=class_weight, random_state=seed).fit(x_train, y_train)
        score = f1_score(y_val, model.predict(x_val), labels=classes, average="macro", zero_division=0)
        if score > best_val:
            best_c, best_val = c, score
    linear = LogisticRegression(C=best_c, max_iter=MAX_ITER, tol=TOLERANCE,
                                class_weight=class_weight, random_state=seed).fit(x_train, y_train)

    candidate_k = [k for k in KNN_K if k < len(train)]
    searcher = NearestNeighbors(n_neighbors=max(candidate_k), n_jobs=-1).fit(x_train)
    code_of = {c: i for i, c in enumerate(classes)}
    train_codes = np.array([code_of[c] for c in y_train], dtype=np.int32)

    def vote(neighbour_codes: np.ndarray, k: int) -> np.ndarray:
        """Majority vote among each row's k nearest training neurons; a tie goes to the
        first class.
        """
        counts = np.zeros((len(neighbour_codes), len(classes)), dtype=np.int32)
        rows = np.repeat(np.arange(len(neighbour_codes)), k)
        np.add.at(counts, (rows, neighbour_codes[:, :k].ravel()), 1)
        return classes[counts.argmax(axis=1)]

    val_neighbours = train_codes[searcher.kneighbors(x_val, return_distance=False)]
    best_k, best_knn_val = None, -1.0
    for k in candidate_k:
        score = f1_score(y_val, vote(val_neighbours, k), labels=classes, average="macro", zero_division=0)
        if score > best_knn_val:
            best_k, best_knn_val = k, score
    tgt_neighbours = train_codes[searcher.kneighbors(x_tgt, return_distance=False)]

    rng = np.random.default_rng(seed)
    priors = np.array([(y_train == c).mean() for c in classes])
    majority = classes[np.argmax([(y_train == c).sum() for c in classes])]

    result = {
        "target": target, "scope": scope,
        "n_source_train": int(len(train)), "n_target": int(len(tgt_usable)),
        "target_cells_dropped": dropped,
        "n_classes": int(len(classes)),
        "source_linear_macro_f1": float(
            f1_score(y_test, linear.predict(x_test), labels=classes, average="macro", zero_division=0)
        ),
        "linear": {"C": best_c, "converged": bool(np.all(linear.n_iter_ < MAX_ITER)),
                   "n_iter": int(np.max(linear.n_iter_)),
                   **_score(y_tgt, linear.predict(x_tgt), classes),
                   "confidence": _confidence(linear.predict_proba(x_tgt), y_tgt, linear.classes_)},
        "knn": {"k": best_k, **_score(y_tgt, vote(tgt_neighbours, best_k), classes)},
        "baselines": {
            "majority": _score(y_tgt, np.full(len(y_tgt), majority), classes),
            "stratified": _score(y_tgt, rng.choice(classes, size=len(y_tgt), p=priors), classes),
        },
    }
    if return_probe:
        # Everything needed to re-apply the probe without scikit-learn's pickles: scale
        # with the scaler; the linear call is the argmax of coef @ x + intercept (softmax
        # for probabilities); the k-NN call is a vote among the k nearest reference rows,
        # Euclidean in the scaled space, a tie going to the first class.
        result["probe"] = {"classes": classes,
                           "scaler_mean": scaler.mean_, "scaler_scale": scaler.scale_,
                           "linear_coef": linear.coef_, "linear_intercept": linear.intercept_,
                           "knn_reference_rows": train, "knn_reference_codes": train_codes}
    if return_predictions:
        # Per-cell rows for the breakdown script: the source's held-out test cells
        # (in-volume) and every scored target cell (zero-shot), under one probe.
        # Not JSON: the caller writes it to parquet and pops it from the report.
        test_neighbours = train_codes[searcher.kneighbors(x_test, return_distance=False)]
        rows = []
        parts = [("source_test", test, x_test, y_test, test_neighbours),
                 ("target", tgt_usable, x_tgt, y_tgt, tgt_neighbours)]
        if unlabelled_mask is not None:
            # The application case: target cells with no label at all, predicted by
            # the same probe. Nothing is scored on them; `true` is empty.
            free = np.flatnonzero(unlabelled_mask & tgt_label.isna().to_numpy())
            if len(free):
                x_free = scaler.transform(target_embeddings[free])
                parts.append(("target_unlabelled", free, x_free, np.full(len(free), None, dtype=object),
                              train_codes[searcher.kneighbors(x_free, return_distance=False)]))
        for where, idx, x, y, nbrs in parts:
            proba = linear.predict_proba(x)
            ranked = np.sort(proba, axis=1)
            code = {c: i for i, c in enumerate(linear.classes_)}
            # k-NN "probability": the share of the k nearest FAFB cells voting for each class.
            shares = np.zeros((len(nbrs), len(classes)), dtype=np.float32)
            np.add.at(shares, (np.repeat(np.arange(len(nbrs)), best_k), nbrs[:, :best_k].ravel()), 1.0 / best_k)
            frame = pd.DataFrame({
                "where": where, "row": idx, "true": y,
                "pred_linear": linear.classes_[proba.argmax(axis=1)],
                "p_linear": ranked[:, -1],                                   # top probability
                "p_true": np.where([c in code for c in y],                   # probability of the true class
                                   proba[np.arange(len(y)), [code.get(c, 0) for c in y]], np.nan),
                "margin": ranked[:, -1] - ranked[:, -2],                     # top minus runner-up
                "entropy": -(proba * np.log(np.clip(proba, 1e-12, 1))).sum(axis=1),
                "pred_knn": vote(nbrs, best_k),
                "knn_share": shares.max(axis=1),
            })
            for j, c in enumerate(linear.classes_):
                frame[f"p_{c}"] = proba[:, j].astype(np.float32)
            for j, c in enumerate(classes):
                frame[f"knn_p_{c}"] = shares[:, j]
            rows.append(frame)
        result["predictions"] = pd.concat(rows, ignore_index=True)
    return result
