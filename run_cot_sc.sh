#!/bin/bash
# =============================================================================
#  Fine-tuned model + chain-of-thought + self-consistency.
#
#  1. pip install -r requirements-eval.txt
#  2. python download_data.py           <- once, on a login node
#  3. Fill in the 2 settings below, put the models in place.
#  4. bash run_cot_sc.sh          <- runs seed 3, which is the one we need
#
#  Seed 1 is done and seed 2 may be covered on our side, so seed 3 is the
#  only one to run right now. Details: COT_SC_NOTES.md
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")"

# ----------------------------- FILL THESE IN ---------------------------------

# Your email, for SLURM job notifications.
EMAIL="you@example.edu"

# Same as in run_morebench.sh.
CKPT_PREFIX="pyro_rej"            # or "qwen_pyro_rej" for Qwen2-7B

# ------------------------- nothing to edit below -----------------------------

SEED="${1:-3}"
case "$SEED" in
    3) ;;
    2)  # Seed 2 may already be running on our side; don't duplicate ~137 GPU-h.
        if [ "${2:-}" != "confirm" ]; then
            echo "Seed 2 is probably already covered — check before running it."
            echo "If it has been confirmed, run:  bash run_cot_sc.sh 2 confirm"
            exit 1
        fi ;;
    1)  echo "Seed 1 is already finished — do not re-run it."; exit 1 ;;
    *)  echo "usage: bash run_cot_sc.sh          # seed 3, the one we need"
        exit 1 ;;
esac

DATA_ROOT="$PWD/data_root"
CKPT_ROOT="$DATA_ROOT/ckpt"
export DATA_ROOT CKPT_ROOT CKPT_PREFIX EMAIL

[ -d "$DATA_ROOT/hg_cache" ] || { echo "Run 'python download_data.py' first."; exit 1; }

echo "== 1/3  pointing the repo at $DATA_ROOT"
# `|| true`: after the first run grep finds nothing and exits 1, which
# `set -o pipefail` would otherwise treat as a fatal error.
grep -rl '<DATA_ROOT>' data_evaluation data_processing 2>/dev/null \
    | xargs -r sed -i "s|<DATA_ROOT>|$DATA_ROOT|g" || true

echo "== 2/3  checking what seed $SEED needs"
missing=0
check () { [ -e "$2" ] && echo "   ok      $1" || { echo "   MISSING $1 -> $2"; missing=1; }; }
check "fine-tuned model seed $SEED" \
      "$CKPT_ROOT/${CKPT_PREFIX}_lora_dist_r8_all_seed${SEED}_bracket_lora8_dist/epoch_0"
check "Bayesian Teaching data" "data_processing/bayesian_teaching_test_base.jsonl"
check "OpenEstimate data"      "data_processing/openestimate_test.json"
for d in mmlu truthfulqa_mc hellaswag winogrande arc_challenge; do
    check "$d validation split" "$DATA_ROOT/hg_cache/${d}_validation_disk"
done
[ "$missing" -eq 0 ] || { echo "Fix the above, then re-run."; exit 1; }

echo "== 3/3  submitting 40 jobs for seed $SEED"
python torchtune/_build_cot_ft_sc_slurm.py \
    --seeds "$SEED" \
    --email "$EMAIL" \
    --submit

cat <<EOF

Done. Each job runs up to ~12 h (BT-guided is the long one); the whole seed is
about 137 GPU-hours and finishes in roughly a day at 40 concurrent.

Watch:   squeue -u \$USER

Don't start another seed alongside this one — 40 jobs fits the cluster's
44-job cap, 80 does not.

Re-running this script is safe: jobs whose output already exists are skipped,
so if something dies just run it again for that seed.

When all 40 outputs exist, combine the 5 reasoning paths into one row:

    python data_evaluation/combine_sc_samples.py --auto \\
        --tag cotsc_pyrorej_all_s${SEED}_bracket
EOF
