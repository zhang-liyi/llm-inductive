#!/bin/bash
# =============================================================================
#  Moral benchmark (MoReBench) evaluation.
#
#  1. pip install -r requirements-eval.txt
#  2. python download_data.py                          <- login node
#  3. python download_models.py                        <- login node
#  4. unzip finetuned_checkpoints.zip -d data_root/
#  5. Put your email below, then: bash run_morebench.sh
#
#  Submits 8 SLURM jobs (base + 3 fine-tuned seeds x 2 datasets), under an
#  hour each. Results go to data_evaluation/results/text_cls/.
#  Details: MOREBENCH_HANDOFF.md
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")"

# ----------------------------- FILL THIS IN ----------------------------------

# Your email, for SLURM job notifications.
EMAIL="akshay.jagadish@princeton.edu"

# ------------------------- nothing to edit below -----------------------------
#
# Base model: downloaded by download_models.py. To run the Qwen row instead,
# switch both lines below to the commented alternatives.
#
# Fine-tuned models: unzip finetuned_checkpoints.zip inside data_root/, which
# gives data_root/ckpt/<model-name>/epoch_0/ for each of the 3 seeds.

BASE_SUBDIR="base_models/Meta-Llama-3-8B-Instruct"
CKPT_PREFIX="pyro_rej"
# BASE_SUBDIR="base_models/Qwen2-7B-Instruct"
# CKPT_PREFIX="qwen_pyro_rej"

DATA_ROOT="$PWD/data_root"
BASE_MODEL="$DATA_ROOT/$BASE_SUBDIR"
CKPT_ROOT="$DATA_ROOT/ckpt"
export DATA_ROOT BASE_MODEL CKPT_ROOT CKPT_PREFIX EMAIL

[ -d "$DATA_ROOT/morebench" ] || { echo "Run 'python download_data.py' first."; exit 1; }

echo "== 1/3  pointing the repo at $DATA_ROOT"
# The eval and data-prep sources ship with a '<DATA_ROOT>' placeholder. Scoped
# to those two directories on purpose: a repo-wide sed would also rewrite this
# script's own text. Harmless to re-run — after the first pass there is nothing
# left to replace.
# The `|| true` matters: on any run after the first, grep finds nothing and
# exits 1, which `set -o pipefail` would otherwise treat as a fatal error.
grep -rl '<DATA_ROOT>' data_evaluation data_processing 2>/dev/null \
    | xargs -r sed -i "s|<DATA_ROOT>|$DATA_ROOT|g" || true

echo "== 2/3  checking the models"
missing=0
for s in 1 2 3; do
    d="$CKPT_ROOT/${CKPT_PREFIX}_lora_dist_r8_all_seed${s}_bracket_lora8_dist/epoch_0"
    if [ -d "$d" ]; then echo "   ok      fine-tuned seed $s"
    else echo "   MISSING fine-tuned seed $s -> $d"; missing=1; fi
done
[ -d "$BASE_MODEL" ] && echo "   ok      base model" \
                     || { echo "   MISSING base model -> $BASE_MODEL"; missing=1; }
[ "$missing" -eq 0 ] || { echo "Fix the paths above, then re-run."; exit 1; }

echo "== 3/3  submitting 8 jobs"
python torchtune/_build_morebench_eval_slurm.py \
    --seeds 1 2 3 \
    --base_model "$BASE_MODEL" \
    --ckpt_root "$CKPT_ROOT" \
    --ckpt_prefix "$CKPT_PREFIX" \
    --email "$EMAIL" \
    --submit

cat <<EOF

Done submitting. Watch them with:   squeue -u \$USER

When everything finishes, print the comparison table with:

    python data_evaluation/summarize_morebench.py \\
        --tags base pyrorej_all_s1_bracket pyrorej_all_s2_bracket pyrorej_all_s3_bracket

Sanity check: chance accuracy is 0.500 and the majority-letter baseline is
0.510, so anything near 0.51 means the model is answering by letter rather than
by content. The summariser's "P(say Yes)" line is the one to look at first.
EOF
