#!/usr/bin/env python3
"""Train the wiretype encoder on one volume (FAFB), with the displacement objective.

The design, in short: a transformer encoder `E_c` reads a cell's tokens (its own
feature row and its top-k partners') and pools them to one embedding. A frozen,
randomly initialised copy `E_t` encodes the same cell twice, once with its class
code on the seed token and once with `UNK`; the difference is the
**displacement**, what the class does to this cell's encoding. A predictor `P`
reads `E_c(x, UNK)` and is trained to output that displacement. `E_c` never sees
the label. **The artefact is `E_c`'s plain embedding**, which is frozen,
transferred to other volumes and probed.

Arms, each written to its own checkpoint and embeddings:
- `untrained`: `E_c` at initialisation. Every result is read against this floor,
  on the same inputs.
- `oracle`: the true displacement, probed on the supervised class. It needs no
  training, and if it does not clear the floor the read-out is broken.
- `displacement`: the trained encoder.

The paper's model:

    qsub -v SET=refined,SPLIT=split_random,STEPS=24000,SUPERVISE_ON=nt_best,TAG=s24k_ntbest \
        hpc/jobs/train.sh

Inputs are the `refined` column set (`wiretype/data/column_sets/`), standardised
with the training split's statistics, and `<volume>_topk<k>.npz` for the partners.

Paper terms: the trained `E_c` (the `displacement` arm) is WireType; the `untrained`
arm is WireType (untrained); a cell is a neuron; the learned query token that pools
the set is the summary token; the probes are the linear and k-NN read-outs.
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wiretype.data.scope import SCOPES, in_scope, restrict_split
from wiretype.data.features import (BLOCKS, attach_local, column_set, effective_rank,
                                    feature_summary, load_features, select_blocks, standardise)
from wiretype.data.labels import harmonise
from wiretype.log import log
from wiretype.eval.collapse import collapse_metrics
from wiretype.eval.probes import MAX_ITER, probe_all
from wiretype.model.encoder import CellEncoder, Predictor
from wiretype.model.tokens import (batch_for, context_fn, displacement, embed_all, gather, oracle_fn,
                                   unk_of)

# Fixed, not swept: a curve must not move because a different k was chosen.
CURVE_K = 15
CURVE_C = 1.0
# The spread in FAFB macro-F1 between two runs differing only in seed. A delta
# inside it is not a result.
SPREAD = 0.026

def cohort_of(nodes, split, target, size, rng):
    """A fixed set of labelled cells for the curve. Fixed is the whole point."""
    labelled = nodes[target].notna().to_numpy()
    return {part: rng.choice(np.flatnonzero(labelled & (split == part).to_numpy()),
                             size=min(size // 2 if part == "train" else size // 4,
                                      int((labelled & (split == part).to_numpy()).sum())),
                             replace=False)
            for part in ("train", "val")}


def curve_point(embeddings, cohort, labels):
    """One cheap probe reading. Deliberately not the protocol number."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler

    x_tr = embeddings[cohort["train"]].cpu().numpy()
    x_va = embeddings[cohort["val"]].cpu().numpy()
    y_tr, y_va = labels[cohort["train"]], labels[cohort["val"]]
    classes = np.array(sorted(set(y_tr)))
    scaler = StandardScaler().fit(x_tr)
    x_tr, x_va = scaler.transform(x_tr), scaler.transform(x_va)
    linear = LogisticRegression(C=CURVE_C, max_iter=MAX_ITER, class_weight="balanced",
                                random_state=0).fit(x_tr, y_tr)
    lin = f1_score(y_va, linear.predict(x_va), labels=classes, average="macro", zero_division=0)
    k = min(CURVE_K, len(x_tr) - 1)
    nn = NearestNeighbors(n_neighbors=k, n_jobs=-1).fit(x_tr)
    code = {c: i for i, c in enumerate(classes)}
    codes = np.array([code[c] for c in y_tr])
    votes = codes[nn.kneighbors(x_va, return_distance=False)]
    pred = classes[np.apply_along_axis(lambda r: np.bincount(r, minlength=len(classes)).argmax(), 1, votes)]
    return float(lin), float(f1_score(y_va, pred, labels=classes, average="macro", zero_division=0))


def target_stats(target_encoder, topk, features_t, y_index, batch, device, amp):
    """What the class is worth inside the target, and the loss a class-blind predictor reaches.

    `constant_predictor_loss` is the reference line for training: a predictor that
    has learned nothing about the cell's class, and outputs the mean displacement
    for every cell, sits at it. Every bit of loss below it is class recovery. A
    falling loss is not evidence until it is below this line.

    `batch` must be drawn the way training draws, so that the reference matches.
    """
    tokens = gather(topk, features_t, batch, device)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        y = y_index[torch.from_numpy(batch.astype(np.int64))].to(device)
        u = target_encoder(*tokens, unk_of(tokens, target_encoder))
        u = F.layer_norm(u.float(), u.shape[-1:])
        d = displacement(target_encoder, tokens, y).float()
    spread = (u - u.mean(0)).norm(dim=-1).mean()
    rms = d.pow(2).mean().sqrt()
    # What a predictor that has learned nothing about this cell's class achieves
    # under --predict displacement: emit the mean displacement for every cell.
    std = d / rms
    const = F.smooth_l1_loss(std.mean(0).expand_as(std), std)
    return {"constant_predictor_loss": float(const),
            "displacement": float(d.norm(dim=-1).mean()),
            "anchor_spread": float(spread),
            "ratio": float(d.norm(dim=-1).mean() / spread),
            "class_share": float(d.pow(2).mean() / (d.pow(2).mean() + (u - u.mean(0)).pow(2).mean())),
            "class_blind_loss": float(0.5 * d.pow(2).mean()),
            "rms": float(rms)}


def train_displacement(encoder, target_encoder, predictor, args, ctx):
    """Fit E_c and P so that P(E_c(x,UNK)) lands on LN E_t(x,y). Returns the curve."""
    device, topk, features_t, rng = ctx["device"], ctx["topk"], ctx["features_t"], ctx["rng"]
    params = list(encoder.parameters()) + list(predictor.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    pool = ctx["supervised_pool"]
    log(f"  {sum(p.numel() for p in params)/1e6:.2f}M trainable · {len(pool):,} labelled "
        f"train seeds · target frozen")
    scale = ctx["target_rms"]

    curve = []
    for step in range(args.steps + 1):
        if step % args.probe_every == 0:
            encoder.eval(), predictor.eval()
            # The curve tracks E_c, the artefact that is transferred.
            emb = embed_all(context_fn(encoder), topk, features_t,
                            ctx["n_nodes"], device, ctx["amp"])
            lin, knn = curve_point(emb, ctx["cohort"], ctx["curve_labels"])
            m = collapse_metrics(emb[ctx["held_out"]])
            curve.append({"step": step, "linear": lin, "knn": knn,
                          "effective_rank": m["effective_rank"],
                          "collapse_ratio": m["collapse_ratio"]})
            log(f"    step {step:6,}  encoder lin {lin:.4f}  knn {knn:.4f}  "
                f"rank {m['effective_rank']:7.1f}  ratio {m['collapse_ratio']:.3f}")
            del emb
            encoder.train(), predictor.train()
        if step == args.steps:
            break

        batch = rng.choice(pool, args.batch, replace=False)
        tokens = gather(topk, features_t, batch, device)
        y = ctx["y_index"][torch.from_numpy(batch.astype(np.int64))].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=ctx["amp"]):
            with torch.no_grad():
                # Everything the class does not touch cancels here exactly, so the
                # predictor has nothing but the class to spend capacity on.
                # Standardised because the raw displacement is ~0.046 per dimension
                # and would barely move at the default lr.
                t = displacement(target_encoder, tokens, y) / scale
            loss = F.smooth_l1_loss(predictor(encoder(*tokens, unk_of(tokens, encoder))), t)
        opt.zero_grad(set_to_none=True)
        loss.float().backward()
        opt.step()
        if step % args.log_every == 0:
            log(f"      step {step:6,}  loss {float(loss):.4f}{ctx['loss_note']}")
    return curve


def main() -> None:
    """Train the requested arms on one volume and write their checkpoints, embeddings
    and report.
    """
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--processed", type=Path, default=Path("data/processed"))
    p.add_argument("--reports", type=Path, default=Path("experiments/train"))
    p.add_argument("--volume", default="fafb")
    p.add_argument("--arms", default="untrained,oracle,displacement",
                   help="comma-separated from untrained,oracle,displacement. `oracle` is the "
                        "read-out's own ceiling and costs no training; read it first.")
    p.add_argument("--supervise-on", default="nt_best",
                   help="the class that enters the target branch's input: a node column, or "
                        "nt_best (nt_known where it exists, else nt_train). Every target is "
                        "still probed; the matched one is the headline.")
    p.add_argument("--features", default="degree,connection,rwse,flow,local",
                   help="feature blocks to load before the column set is applied: degree, "
                        "connection, rwse, flow, local.")
    p.add_argument("--column-set", default="refined",
                   help="name of a frozen column list in wiretype/data/column_sets/ (the paper "
                        "uses `refined`, 31 columns). Applied after --features.")
    p.add_argument("-k", type=int, default=64)
    p.add_argument("--steps", type=int, default=24000)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--ff-mult", type=int, default=4)
    p.add_argument("--predictor-layers", type=int, default=2)
    p.add_argument("--predictor-mult", type=int, default=2)
    p.add_argument("--probe-every", type=int, default=1000)
    p.add_argument("--log-every", type=int, default=500,
                   help="loss logging cadence. The loss is the fastest read on whether the "
                        "predictor is beating its learns-nothing reference; 500 is too coarse "
                        "for a smoke.")
    p.add_argument("--cohort", type=int, default=12288)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--no-final-probe", action="store_true",
                   help="skip probe_all. Pre-flight only; the report is then not an arm.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--scope", choices=SCOPES, default="brain_neurons",
                   help="which cells of --volume take part: 'brain_neurons' (the paper's setting) keeps "
                        "brain neurons with at least one connection; 'all' keeps every cell. Stored in "
                        "the checkpoint, so transfer.py applies the same.")
    p.add_argument("--split-column", default="split_random",
                   help="'split_random' (per cell) is the paper's protocol; 'split' holds out "
                        "whole cell types, a stress test for unseen types.")
    p.add_argument("--tag", default="")
    args = p.parse_args()

    # --- 1. Arms, device and random state
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    unknown = set(arms) - {"untrained", "oracle", "displacement"}
    if unknown:
        raise SystemExit(f"unknown arms: {sorted(unknown)}; choose from untrained, oracle, displacement")
    args.reports.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = (not args.no_amp) and device.type == "cuda"
    rng = np.random.default_rng(args.seed)
    log(f"device {device} · arms {arms} · supervise_on {args.supervise_on} · amp {amp}")

    # --- 2. Neurons, labels and the split (out-of-scope neurons marked `excluded`)
    nodes = harmonise(pd.read_parquet(args.processed / f"{args.volume}_nodes.parquet")
                      .sort_values("node_id").reset_index(drop=True), args.volume)
    if args.supervise_on == "nt_best":
        changed = int((nodes["nt_best"] != nodes["nt_train"]).sum())
        log(f"nt_best: {changed:,} cells relabelled from nt_train; "
            + ", ".join(f"{c} {n:,}" for c, n in nodes["nt_best"].value_counts().items()))
    edges = pd.read_parquet(args.processed / f"{args.volume}_edges.parquet")
    split = pd.read_parquet(args.processed / f"{args.volume}_splits.parquet").sort_values("node_id")[args.split_column].reset_index(drop=True)
    if set(split.unique()) - {"train", "val", "test"}:
        raise SystemExit(f"--split-column {args.split_column} holds values other than train/val/test")
    # Out-of-scope cells leave every split (see wiretype.data.scope): the scaler,
    # the training pool and the probes then all skip them.
    split = restrict_split(nodes, split, args.scope)
    if args.scope != "all":
        log(f"--scope {args.scope}: {int(in_scope(nodes, args.scope).sum()):,} of {len(nodes):,} cells kept")
    log(f"split column {args.split_column}: " + ", ".join(f"{s} {int((split == s).sum()):,}" for s in ("train", "val", "test")))
    n_nodes = len(nodes)
    # --- 3. Input features: the column set, standardised with training-split statistics
    raw, names = load_features(args.processed, args.volume, edges, n_nodes)
    blocks = [b.strip() for b in args.features.split(",") if b.strip()]
    if "local" in blocks:
        raw, names = attach_local(args.processed, args.volume, raw, names)
    if set(blocks) != set(BLOCKS):
        raw, names = select_blocks(raw, names, blocks)
        log(f"--features {','.join(blocks)}: {raw.shape[1]} of {len(names)} columns")
    # The column set, read verbatim. Order follows the loaded matrix, not the file,
    # so a set is a column subset of the same vector and never a reordering of it.
    wanted = column_set(args.column_set)
    missing = [c for c in wanted if c not in names]
    if missing:
        raise SystemExit(f"column set {args.column_set} names columns this vector does not have: "
                         f"{missing}. Widen --features.")
    wanted_set = set(wanted)
    if len(wanted_set) != len(wanted):
        raise SystemExit(f"column set {args.column_set} lists a column twice")
    dropped = [n for n in names if n not in wanted_set]
    keep = [i for i, n in enumerate(names) if n in wanted_set]
    raw, names = raw[:, keep], [names[i] for i in keep]
    log(f"--column-set {args.column_set}: {len(names)} columns kept, {len(dropped)} dropped"
        + (f": {', '.join(dropped)}" if dropped else ""))
    features = standardise(raw, mask=(split == "train").to_numpy())
    log(f"{feature_summary(names)}, effective rank {effective_rank(features):.2f}")
    features_t = torch.from_numpy(features).to(device)
    del edges, raw

    # --- 4. Partner tokens: the top-k archive, validated before use
    # Existence is not readiness: a job queued alongside the top-k build can open a
    # half-written archive. Validate rather than assume, and say which problem it is.
    path = args.processed / f"{args.volume}_topk{args.k}.npz"
    build = f"qsub -v VOLUME={args.volume},K={args.k} hpc/jobs/topk.sh"
    if not path.exists():
        raise SystemExit(f"no {path} — build it first: {build}")
    try:
        topk = dict(np.load(path))
    except Exception as exc:
        raise SystemExit(
            f"{path} is unreadable ({type(exc).__name__}: {exc}). It is most likely still "
            f"being written, or was left truncated by a killed job. Delete it and rebuild:\n"
            f"  rm {path} && {build}\n"
            f"To chain the two, submit with: qsub -hold_jid <topk-jobid> hpc/jobs/train.sh")
    missing = {"partner", "weight", "sign", "length"} - set(topk)
    if missing:
        raise SystemExit(f"{path} is missing {sorted(missing)} — rebuild it: {build}")
    if topk["partner"].shape != (n_nodes, args.k):
        raise SystemExit(
            f"{path} holds {topk['partner'].shape} but this run wants "
            f"({n_nodes}, {args.k}). Training on a k the filename does not match is silent, "
            f"so this stops here.")

    # --- 5. Which neurons train, which are held out, and the labelled pool the predictor fits on
    seeds_all = np.flatnonzero(nodes["is_seed"].to_numpy() & in_scope(nodes, args.scope))
    held_out = rng.choice(seeds_all, size=min(max(4 * args.width, 2048), len(seeds_all) // 4), replace=False)
    trainable = np.setdiff1d(seeds_all, held_out)

    # The predictor may only be fitted on train-split cells that carry the class.
    # A pool drawn from val or test would leak the label the probe is scored on.
    labelled = nodes[args.supervise_on].notna().to_numpy().copy()
    supervised_pool = np.intersect1d(trainable, np.flatnonzero(labelled & (split == "train").to_numpy()))
    classes = np.array(sorted(set(nodes[args.supervise_on].dropna())))
    index = {c: i for i, c in enumerate(classes)}
    y_index = torch.full((n_nodes,), -1, dtype=torch.long)
    y_index[labelled] = torch.tensor(
        np.array([index[c] for c in nodes[args.supervise_on][labelled]], dtype=np.int64))
    log(f"{len(trainable):,} seeds · {len(held_out):,} held out · "
        f"{len(supervised_pool):,} labelled train cells over {len(classes)} classes")

    per_class = [int((y_index[torch.from_numpy(supervised_pool.astype(np.int64))] == c).sum())
                 for c in range(len(classes))]
    log("  labelled train cells per class: "
        + ", ".join(f"{c}={n:,}" for c, n in zip(classes, per_class)))

    curve_labels = nodes[args.supervise_on].astype("object").to_numpy()
    cohort = cohort_of(nodes, split, args.supervise_on, args.cohort, np.random.default_rng(args.seed + 7))

    # --- 6. One initialisation for every arm; the frozen target E_t and its label sensitivity
    # One initialisation, shared by every arm and by both branches, so `untrained`
    # is not an approximation of the trained arm's starting point -- it is exactly
    # it, and E_t differs from E_c at step 0 only in what it is fed.
    torch.manual_seed(args.seed)
    init = CellEncoder(features.shape[1], len(classes), d=args.width, heads=args.heads,
                       layers=args.layers, ff_mult=args.ff_mult).to(device)
    scale = init.calibrate_label_code(
        *gather(topk, features_t,
                 rng.choice(trainable, batch_for(args.k, 4096), replace=False), device)[:4])
    init_state = copy.deepcopy(init.state_dict())
    log(f"encoder {sum(q.numel() for q in init.parameters())/1e6:.2f}M parameters · "
        f"class codes scaled to the mean seed-token norm, {scale:.3f}")

    # E_t, frozen at init. It can never learn to ignore its class channel, which is
    # the degenerate optimum the whole design has to avoid.
    target_encoder = CellEncoder(features.shape[1], len(classes), d=args.width, heads=args.heads,
                                 layers=args.layers, ff_mult=args.ff_mult).to(device)
    target_encoder.load_state_dict(init_state)
    target_encoder.eval()
    for q in target_encoder.parameters():
        q.requires_grad_(False)

    stat_rng = np.random.default_rng(args.seed + 13)
    stat_batch = stat_rng.choice(
        supervised_pool, size=min(batch_for(args.k, 4096), len(supervised_pool)), replace=False)
    sens = target_stats(target_encoder, topk, features_t, y_index, stat_batch, device, amp)
    log(f"label sensitivity: displacement {sens['displacement']:.3f} over anchor spread "
        f"{sens['anchor_spread']:.3f} = {sens['ratio']:.3f}")
    log(f"the class is {sens['class_share']:.2%} of the target's variance; a class-blind "
        f"predictor of it reaches loss {sens['class_blind_loss']:.6f}")
    log("  the x-dependent part is cancelled inside the target. A predictor that learns")
    log(f"  nothing sits at {sens['constant_predictor_loss']:.4f}; every bit of loss below that "
        f"is class recovery.")
    if sens["ratio"] < 0.05:
        log("  WARNING: the class channel barely moves the target. Read `oracle` before "
            "trusting anything else in this report; the read-out may be inert.")

    # --- 7. Train each arm in turn, embed every neuron, probe, and save
    ctx = {"device": device, "topk": topk, "features_t": features_t, "rng": rng,
           "n_nodes": n_nodes, "trainable": trainable, "supervised_pool": supervised_pool,
           "held_out": held_out, "y_index": y_index, "cohort": cohort,
           "curve_labels": curve_labels, "amp": amp, "target_rms": sens["rms"],
           "loss_note": f"   (learns-nothing {sens['constant_predictor_loss']:.4f})"}

    report = {"settings": vars(args) | {"arms": arms}, "columns": names, "target_stats": sens,
              "label_code_scale": scale, "arms": {}}
    split_tag = "" if args.split_column == "split" else "_" + args.split_column
    suffix = "_" + args.column_set + split_tag + (("_" + args.tag) if args.tag else "")
    out = args.reports / f"train_{args.volume}{suffix}.json"

    def save(name, emb, probes):
        np.save(args.reports / f"embeddings_{args.volume}_{name}{suffix}.npy", emb.astype(np.float16))
        report["arms"][name] = probes
        out.write_text(json.dumps(report, indent=2, default=str))

    floor = None
    for arm in arms:
        log("")
        log(f"=== {arm} ===")
        t0 = time.time()
        if arm == "untrained":
            encoder = CellEncoder(features.shape[1], len(classes), d=args.width, heads=args.heads,
                                  layers=args.layers, ff_mult=args.ff_mult).to(device)
            encoder.load_state_dict(init_state)
            encoder.eval()
            # The untrained floor has to be transferable, not merely probeable, so it
            # is saved with the keys scripts/transfer.py reads. The weights are
            # `init_state`: the exact starting point of the trained arm.
            torch.save({"encoder": encoder.state_dict(), "settings": vars(args),
                        "classes": classes.tolist()},
                       args.reports / f"checkpoint_{args.volume}_untrained{suffix}.pt")
            emb = embed_all(context_fn(encoder), topk, features_t, n_nodes, device, amp).cpu().numpy()
            entry = {"curve": [], "probes": {} if args.no_final_probe
                     else probe_all(emb, nodes, split, seed=args.seed)}
            floor = entry["probes"]
        elif arm == "oracle":
            emb = embed_all(oracle_fn(target_encoder, y_index), topk, features_t,
                            n_nodes, device, amp).cpu().numpy()
            # Only the supervised class is in this displacement, so only it is
            # informative -- the other targets would be scoring a constant.
            entry = {"curve": [], "probes": {} if args.no_final_probe
                     else probe_all(emb, nodes, split, targets=(args.supervise_on,), seed=args.seed)}
        else:
            encoder = CellEncoder(features.shape[1], len(classes), d=args.width, heads=args.heads,
                                  layers=args.layers, ff_mult=args.ff_mult).to(device)
            encoder.load_state_dict(init_state)
            predictor = Predictor(args.width, hidden_mult=args.predictor_mult,
                                  layers=args.predictor_layers).to(device)
            curve = train_displacement(encoder, target_encoder, predictor, args, ctx)
            encoder.eval(), predictor.eval()
            torch.save({"encoder": encoder.state_dict(), "predictor": predictor.state_dict(),
                        "target_encoder": target_encoder.state_dict(), "settings": vars(args),
                        "classes": classes.tolist()},
                       args.reports / f"checkpoint_{args.volume}_displacement{suffix}.pt")
            # **E_c is the artefact.** The predictor and the target encoder are
            # training-time scaffolding; they are saved but never transferred.
            emb = embed_all(context_fn(encoder), topk, features_t,
                            n_nodes, device, amp).cpu().numpy()
            entry = {"curve": curve, "probes": {} if args.no_final_probe
                     else probe_all(emb, nodes, split, seed=args.seed)}
        entry["minutes"] = (time.time() - t0) / 60
        save(arm, emb, entry)
        log(f"  {arm} done in {entry['minutes']:.1f} min")
        del emb

    log("")
    # --- 8. Summary: each arm against the untrained floor
    if args.no_final_probe:
        log("--no-final-probe: no probes in this report; it is a pre-flight, not an arm.")
        log(f"wrote {out}")
        return
    log("macro-F1, linear / k-NN, and the delta against the untrained floor.")
    log("`oracle` is scored on the supervised class only.")
    for key in dict.fromkeys((f"{args.supervise_on}/whole_brain", "nt_train/whole_brain",
                              "nt_train/central_brain", "hemilineage/whole_brain",
                              "super_class/whole_brain")):
        if not floor or "linear" not in floor.get(key, {}):
            continue
        log(f"  {key}")
        for arm, entry in report["arms"].items():
            r = entry["probes"].get(key, {})
            if "linear" not in r:
                continue
            dl = r["linear"]["macro_f1"] - floor[key]["linear"]["macro_f1"]
            dk = r["knn"]["macro_f1"] - floor[key]["knn"]["macro_f1"]
            flag = ""
            if arm != "untrained":
                flag = "   <-- clears the floor" if dl > SPREAD else "   (inside run-to-run spread)"
            log(f"    {arm:<20} {r['linear']['macro_f1']:.4f} / {r['knn']['macro_f1']:.4f}   "
                f"({dl:+.4f} / {dk:+.4f}){flag}")
    log("")
    log(f"{SPREAD} is the FAFB run-to-run spread between two seeds.")
    log("If `oracle` did not clear the floor comfortably, the read-out is broken and the")
    log("`displacement` arm says nothing.")
    log(f"wrote {out}")


if __name__ == "__main__":
    main()
