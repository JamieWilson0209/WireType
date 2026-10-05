#!/bin/bash
# =============================================================================
# Submit every model and score behind the paper (SGE). Run from a login node,
# from the repo root, after the feature jobs (README, Running the pipeline).
#
#   ONLY=smoke   bash hpc/submit_paper.sh   # optional: tiny runs through every path
#   bash hpc/submit_paper.sh                # the main groups below
#   ONLY=seeds   bash hpc/submit_paper.sh   # seeds 1-2 of the type-blocked and reverse runs
#   ONLY=release bash hpc/submit_paper.sh   # the released predictions and probes; needs
#                                           # the seed-0 checkpoints from the main groups
#   (ONLY="release seeds" submits both.)
#
# Main groups (ONLY picks some; transfers and scorings hold on their builds):
#   forward     WireType seeds 0-2, then transfers of the trained and untrained
#               arms; seed 0 also with the in-volume diagnostic, target scaling
#               (own_neurons) and saved embeddings for both arms.
#   typeblocked the type-blocked split (SPLIT=split), then both transfers.
#   reverse     WireType on MCNS brain neurons, then input scaling (in-volume,
#               embeddings), target scaling (own) and untrained transfers.
#   predicted   the model trained on predicted labels (SUPERVISE_ON=nt_train,
#               displacement arm only), then its transfer.
#   baselines   GAT, GraphSAGE, GIN, GraphMAE, DGI (seeds 0-2) and BGRL (seeds 0, 2),
#               each scored after its build; raw features and degree only built
#               and scored; the reverse raw-feature run.
#   checks      score_embeddings on the saved forward and reverse embeddings; it
#               must match the transfer reports to within 1e-3 (it reads float16
#               embeddings).
#
# Every job trains, scales and probes on connected brain neurons only
# (SCOPE=brain_neurons; src/wiretype/data/scope.py). Outputs go to
# experiments/{train,transfer,baselines,release} under fixed names, so a second
# run overwrites the first.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

ONLY="${ONLY:-forward typeblocked reverse predicted baselines checks}"
wanted() { [[ " ${ONLY} " == *" $1 "* ]]; }
q() { qsub -terse "$@" | cut -d. -f1; }
SC=SCOPE=brain_neurons
T=experiments/train
C=${T}/checkpoint_fafb_
M=${T}/checkpoint_mcns_
FWD_EMB=experiments/transfer/transfer_fafb_to_mcns_checkpoint_fafb_displacement_refined_split_random_s24k_ntbest_source_embeddings
REV_MCNS=${T}/embeddings_mcns_displacement_refined_split_random_s24k_ntbest_brain.npy
REV_FAFB=experiments/transfer/transfer_mcns_to_fafb_checkpoint_mcns_displacement_refined_split_random_s24k_ntbest_brain_source_embeddings_fafb.npy

mkdir -p logs

# ---------------------------------------------------------------- 1. smoke
if wanted smoke; then
    tr=$(q -v ${SC},STEPS=100,PROBE_EVERY=50,NO_PROBE=1,TAG=smoke_rerun hpc/jobs/train.sh)
    x=$(q -hold_jid "${tr}" -v CKPT=${C}displacement_refined_split_random_smoke_rerun.pt,DUMP=1,IN_VOLUME=1 hpc/jobs/transfer.sh)
    echo "smoke train ${tr} -> transfer ${x}"
    for method in gat dgi graphmae bgrl; do
        b=$(q -v METHOD=${method},EPOCHS=2,${SC},TAG=smoke hpc/jobs/baseline.sh)
        name=$([ ${method} = gat ] && echo gat_ntbest_s0_smoke || echo ${method}_s0_smoke)
        s=$(q -hold_jid "${b}" -v NAME=${name},${SC} hpc/jobs/score_embeddings.sh)
        echo "smoke ${method} build ${b} -> score ${s}"
    done
    echo "smoke raw $(q -v NAME=raw_s0_smoke,BUILD=raw,${SC} hpc/jobs/score_embeddings.sh)"
    echo
    echo "Check: each baseline log prints 'training graph: 138,639 of 139,262 cells';"
    echo "the train log prints scope=brain_neurons; every job ends 'finished'."
    echo "Then: bash hpc/submit_paper.sh"
    exit 0
fi

# ---------------------------------------------------------------- release
# The released predictions: the seed-0 forward and reverse (input scaling)
# transfers again, into experiments/release, with both released transmitter
# probes calling every connected target brain neuron, the fitted probes saved and
# float32 embeddings. No retraining: the checkpoints exist. Their scores equal the
# main groups' reports; tests/check_release.py checks this before
# scripts/release.py builds the tables.
if wanted release; then
    mkdir -p experiments/release
    rel="DUMP=1,EMB=1,RELEASE=1,REPORTS=experiments/release"
    fwd=$(q -v CKPT=${C}displacement_refined_split_random_s24k_ntbest.pt,${rel} hpc/jobs/transfer.sh)
    rev=$(q -v CKPT=${M}displacement_refined_split_random_s24k_ntbest_brain.pt,SOURCE=mcns,TARGET=fafb,${rel} hpc/jobs/transfer.sh)
    printf '%-28s %s\n' run job release_forward_s0 "${fwd}" release_reverse_s0 "${rev}"
    wanted seeds || exit 0
fi

# ---------------------------------------------------------------- seeds
# Seeds 1 and 2 of the type-blocked and reverse runs (seed 0 is in the main
# groups). The seed changes the initialisation and training order; the
# type-blocked split itself (which types are held out) stays fixed.
if wanted seeds; then
    rows=()
    for seed in 1 2; do
        tr=$(q -v SPLIT=split,SEED=${seed},${SC},TAG=s24k_ntbest_typeblocked_seed${seed} hpc/jobs/train.sh)
        for arm in displacement untrained; do
            x=$(q -hold_jid "${tr}" -v CKPT=${C}${arm}_refined_s24k_ntbest_typeblocked_seed${seed}.pt,SEED=${seed},DUMP=1 hpc/jobs/transfer.sh)
            rows+=("typeblocked_${arm}_s${seed} ${tr}->${x}")
        done
        tr=$(q -v VOLUME=mcns,SEED=${seed},${SC},TAG=s24k_ntbest_brain_seed${seed} hpc/jobs/train.sh)
        rv=",SOURCE=mcns,TARGET=fafb,SEED=${seed},DUMP=1"
        ck=${M}displacement_refined_split_random_s24k_ntbest_brain_seed${seed}.pt
        rows+=("reverse_input_scaling_s${seed} ${tr}->$(q -hold_jid "${tr}" -v CKPT=${ck}${rv} hpc/jobs/transfer.sh)")
        rows+=("reverse_target_scaling_s${seed} ${tr}->$(q -hold_jid "${tr}" -v CKPT=${ck}${rv},STD=own hpc/jobs/transfer.sh)")
        rows+=("reverse_untrained_s${seed} ${tr}->$(q -hold_jid "${tr}" -v CKPT=${M}untrained_refined_split_random_s24k_ntbest_brain_seed${seed}.pt${rv} hpc/jobs/transfer.sh)")
    done
    printf '%-28s %s\n' run "job(s)"
    for row in "${rows[@]}"; do printf '%-28s %s\n' ${row}; done
    exit 0
fi

# ---------------------------------------------------------------- main groups
submitted=()

transfer() {  # hold_job ckpt extra_vars label
    submitted+=("$4 $1->$(q -hold_jid "$1" -v CKPT=$2$3 hpc/jobs/transfer.sh)")
}

if wanted forward; then
    for seed in 0 1 2; do
        tag=s24k_ntbest$([ ${seed} = 0 ] || echo "_seed${seed}")
        tr=$(q -v SEED=${seed},${SC},TAG=${tag} hpc/jobs/train.sh)
        ck=${C}displacement_refined_split_random_${tag}.pt
        if [ ${seed} = 0 ]; then
            transfer "${tr}" "${ck}" ",DUMP=1,EMB=1,IN_VOLUME=1" "forward_displacement_s0"
            transfer "${tr}" "${ck}" ",STD=own_neurons,DUMP=1" "forward_own_neurons_s0"
        else
            transfer "${tr}" "${ck}" ",SEED=${seed},DUMP=1" "forward_displacement_s${seed}"
        fi
        # seed 0 untrained also saves embeddings: forward_mechanism.py compares them
        transfer "${tr}" "${C}untrained_refined_split_random_${tag}.pt" ",SEED=${seed},DUMP=1$([ ${seed} = 0 ] && echo ,EMB=1)" "forward_untrained_s${seed}"
    done
fi

if wanted typeblocked; then
    tr=$(q -v SPLIT=split,${SC},TAG=s24k_ntbest_typeblocked hpc/jobs/train.sh)
    for arm in displacement untrained; do
        transfer "${tr}" "${C}${arm}_refined_s24k_ntbest_typeblocked.pt" ",DUMP=1" "typeblocked_${arm}"
    done
fi

if wanted reverse; then
    tr=$(q -v VOLUME=mcns,${SC},TAG=s24k_ntbest_brain hpc/jobs/train.sh)
    rv=",SOURCE=mcns,TARGET=fafb,DUMP=1"
    transfer "${tr}" "${M}displacement_refined_split_random_s24k_ntbest_brain.pt" "${rv},EMB=1,IN_VOLUME=1" "reverse_input_scaling"
    transfer "${tr}" "${M}displacement_refined_split_random_s24k_ntbest_brain.pt" "${rv},STD=own" "reverse_target_scaling"
    transfer "${tr}" "${M}untrained_refined_split_random_s24k_ntbest_brain.pt" "${rv}" "reverse_untrained"
fi

if wanted predicted; then
    tr=$(q -v SUPERVISE_ON=nt_train,ARMS=displacement,${SC},TAG=s24k_nttrain hpc/jobs/train.sh)
    transfer "${tr}" "${C}displacement_refined_split_random_s24k_nttrain.pt" ",DUMP=1" "predicted_labels"
fi

if wanted baselines; then
    for seed in 0 1 2; do
        for method in gat sage gin graphmae dgi bgrl; do
            [ ${method} = bgrl ] && [ ${seed} = 1 ] && continue   # BGRL seed 1 does not fit the MIG GPU
            case ${method} in gat|sage|gin) name=${method}_ntbest_s${seed} ;; *) name=${method}_s${seed} ;; esac
            b=$(q -v METHOD=${method},SEED=${seed},${SC} hpc/jobs/baseline.sh)
            submitted+=("baseline_${name} ${b}->$(q -hold_jid "${b}" -v NAME=${name},SEED=${seed},${SC} hpc/jobs/score_embeddings.sh)")
        done
    done
    for method in raw degree; do
        submitted+=("baseline_${method}_s0 $(q -v NAME=${method}_s0,BUILD=${method},${SC} hpc/jobs/score_embeddings.sh)")
    done
    submitted+=("baseline_raw_reverse $(q -v NAME=raw_s0_mcns_to_fafb,BUILD=raw,SOURCE=mcns,TARGET=fafb,${SC} hpc/jobs/score_embeddings.sh)")
fi

if wanted checks; then
    # Hold on the transfers that write the embeddings: find their job ids from this run
    fwd=$(printf '%s\n' "${submitted[@]}" | awk '$1=="forward_displacement_s0"{split($2,a,"->"); print a[2]}')
    rev=$(printf '%s\n' "${submitted[@]}" | awk '$1=="reverse_input_scaling"{split($2,a,"->"); print a[2]}')
    [ -n "${fwd}" ] && [ -n "${rev}" ] || { echo "ERROR: checks need the forward and reverse groups in the same run"; exit 1; }
    submitted+=("check_encoder $(q -hold_jid "${fwd}" -v NAME=encoder_check,SRC=${FWD_EMB}_fafb.npy,TGT=${FWD_EMB}_mcns.npy,DUMP= hpc/jobs/score_embeddings.sh)")
    submitted+=("check_reverse $(q -hold_jid "${rev}" -v NAME=reverse_check,SRC=${REV_MCNS},TGT=${REV_FAFB},SOURCE=mcns,TARGET=fafb,${SC},DUMP= hpc/jobs/score_embeddings.sh)")
fi

echo
printf '%-28s %s\n' run "job(s)"
for row in "${submitted[@]}"; do printf '%-28s %s\n' ${row}; done
echo
echo "When all jobs finish, copy experiments/ to wherever the local analyses run (README)."
