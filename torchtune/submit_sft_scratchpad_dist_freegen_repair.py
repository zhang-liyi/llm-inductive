"""
Rerun the 2238-point Bayesian Teaching free-gen eval on the
scratchpad-distribution SFT checkpoint with:
  - larger max_new_tokens (640)
  - two-stage answer-repair when <N> parsing fails
    (search for "the answer is" phrase, else append a forcing suffix)
"""

import datetime
import os
import subprocess
import time

tm = str(datetime.datetime.now())
TMSTR = tm[:10] + "-" + tm[11:13] + tm[14:16] + tm[17:19]

BASE_DIR      = "."
TORCHTUNE_DIR = f"{BASE_DIR}/torchtune"
DATA_EVAL_DIR = f"{BASE_DIR}/data_evaluation"
ARCHIVE_DIR   = f"{BASE_DIR}/archive"

CKPT_EPOCH = (
    f"{TORCHTUNE_DIR}/ckpt/llama3_8B/"
    "sft_scratchpad_distribution_lora8_sft30-sft_answer_sftlr1e-05-p2lr1e-05/"
    "epoch_0"
)

BT_DATA = f"{BASE_DIR}/data_processing/bayesian_teaching_test_base.jsonl"
BT_OUT  = f"{CKPT_EPOCH}/bayesian_teaching_eval_base_freegen_repair.json"


def submit(job_name, cmd, log_suffix):
    script = (
        f"#!/bin/bash\n"
        f"#SBATCH --job-name={job_name}\n"
        f"#SBATCH --time=23:59:59\n"
        f"#SBATCH --gres=gpu:1\n"
        f"#SBATCH --constraint=gpu80\n"
        f"#SBATCH --nodes=1\n"
        f"#SBATCH --ntasks=1\n"
        f"#SBATCH --cpus-per-task=1\n"
        f"#SBATCH --mem-per-cpu=50G\n"
        f"#SBATCH --mail-type=end\n"
        f"#SBATCH --mail-user=anonymous@example.com
        f"{cmd} >{ARCHIVE_DIR}/{TMSTR}_{log_suffix}.out 2>&1\n"
    )
    with open("temp.sh", "w") as f:
        f.write(script)
    subprocess.call("chmod +x temp.sh", shell=True)
    time.sleep(0.1)
    subprocess.call("sbatch temp.sh", shell=True)


bt_cmd = (
    f"python {DATA_EVAL_DIR}/evaluate_bayesian_teaching.py "
    f"--ckpt_dir {CKPT_EPOCH} "
    f"--data_path {BT_DATA} "
    f"--mode generate "
    f"--free_gen "
    f"--max_new_tokens 640 "
    f"--max_seq_len 2048 "
    f"--output_file {BT_OUT}"
)
print("Submitting BT free-gen repair rerun on scratchpad-distribution ckpt")
submit("bt-b-sftsd-r", bt_cmd, "bt_base_sftsd_freegen_repair")
print("Done.")
