"""
Emit SLURM scripts for the MoReBench binary-criterion evals.

Compares the pretrained Llama-3-8B-Instruct base against the pyro rejection-
sampling "all-domains" distribution LoRA (the posterior-all-domain model),
on both MoReBench datasets:

    morebench          1880 theory-neutral items
    morebench_theory    642 framework-conditional items

One job per (model, dataset).  Default is base + seed 1 => 4 jobs; pass
``--seeds 1 2 3`` for the full 3-seed row (8 jobs).

Prerequisite (run once, on a node with internet — a login node):

    python data_processing/prepare_morebench.py --data_root <DATA_ROOT>

Usage
-----
    python torchtune/_build_morebench_eval_slurm.py            # write scripts
    python torchtune/_build_morebench_eval_slurm.py --submit   # write + sbatch
"""

import argparse
import os
import stat
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
# This file *writes* other files, so it cannot rely on the repo-wide
# `sed s|<DATA_ROOT>|...|` pass — that runs once over the source tree, before
# these scripts exist. Resolve the root at emit time instead.
BASE_DIR = os.path.dirname(HERE)
DATA_ROOT = os.environ.get("DATA_ROOT", "")
EVAL_DIR = f"{BASE_DIR}/data_evaluation"
RESULTS = f"{EVAL_DIR}/results"
ARCHIVE = f"{BASE_DIR}/archive"
# The public repo ships no checkpoints, so this defaults to $CKPT_ROOT and
# falls back to the in-repo path only if that is unset. On the lab account the
# LoRA adapters live in the sibling working repo, not here — see the handoff.
CKPT_ROOT = os.environ.get("CKPT_ROOT", f"{BASE_DIR}/torchtune/ckpt/llama3_8B")
PRETRAINED_REL = (
    "resources/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/"
    "e1945c40cd546c78e41f1151f4db032b271faeaa"
)

DATASETS = ("morebench", "morebench_theory")

# Both datasets are short (max 819 tokens with the chat template) and small,
# so each job is well under the 1 h gpu-test limit: ~1880 forward passes.
SBATCH_HEADER = """\
#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --time=0:59:59
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=50G
#SBATCH --mail-type=end
#SBATCH --mail-user=anonymous@example.com
"""


def write_script(path: str, body: str) -> None:
    with open(path, "w") as f:
        f.write(body)
    os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def build_job(tag: str, ckpt: str, dataset: str, pretrained: bool) -> str:
    out = f"{RESULTS}/text_cls/{tag}_{dataset}.json"
    log = f"{ARCHIVE}/{tag}_{dataset}.out"
    job = f"mb-{tag}-{dataset}".replace("_", "-")[:40]
    flag = " \\\n    --pretrained" if pretrained else ""
    body = SBATCH_HEADER.format(job_name=job) + (
        f"mkdir -p {RESULTS}/text_cls {ARCHIVE}\n"
        f"cd {EVAL_DIR}\n"
        f"python {EVAL_DIR}/evaluate_text_classification.py \\\n"
        f"    --ckpt_dir {ckpt} \\\n"
        f"    --dataset {dataset} \\\n"
        f"    --output_file {out}{flag} \\\n"
        f"    >{log} 2>&1\n"
    )
    path = f"{HERE}/submit_{tag}_{dataset}.sh"
    write_script(path, body)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=[1],
                    help="Fine-tuned seeds to evaluate (default: 1).")
    ap.add_argument("--no_base", action="store_true",
                    help="Skip the pretrained-base jobs.")
    ap.add_argument("--submit", action="store_true",
                    help="sbatch each script after writing it.")
    ap.add_argument("--data_root", default=DATA_ROOT,
                    help="External-data root holding resources/ (default: "
                         "$DATA_ROOT).")
    ap.add_argument("--ckpt_root", default=CKPT_ROOT,
                    help="Directory holding the LoRA checkpoint dirs "
                         "(default: $CKPT_ROOT, else <repo>/torchtune/ckpt/"
                         "llama3_8B).")
    ap.add_argument("--base_model", default=os.environ.get("BASE_MODEL", ""),
                    help="Path to the pretrained base snapshot (Llama-3-8B-"
                         "Instruct or Qwen2-7B-Instruct). Default: $BASE_MODEL, "
                         "else <data_root>/" + PRETRAINED_REL)
    ap.add_argument("--ckpt_prefix", default=os.environ.get("CKPT_PREFIX",
                                                            "pyro_rej"),
                    help="Leading part of the LoRA checkpoint dir name; "
                         "'pyro_rej' for Llama-3, 'qwen_pyro_rej' for Qwen2.")
    ap.add_argument("--email", default=os.environ.get("EMAIL", ""),
                    help="SLURM --mail-user address.")
    args = ap.parse_args()

    if args.email:
        global SBATCH_HEADER
        SBATCH_HEADER = SBATCH_HEADER.replace("anonymous@example.com",
                                              args.email)

    base_model = args.base_model
    if not args.no_base and not base_model:
        if not args.data_root:
            ap.error("--base_model (or $BASE_MODEL) is required, or give "
                     "--data_root; pass --no_base to skip the base row.")
        base_model = f"{args.data_root}/{PRETRAINED_REL}"

    models = []
    if not args.no_base:
        if not os.path.isdir(base_model):
            print(f"WARN: base model not found: {base_model}")
        models.append(("base", base_model, True))
    for s in args.seeds:
        ckpt = (f"{args.ckpt_root}/{args.ckpt_prefix}_lora_dist_r8_all_seed{s}"
                f"_bracket_lora8_dist/epoch_0")
        if not os.path.isdir(ckpt):
            print(f"WARN: checkpoint not found: {ckpt}\n"
                  f"      pass --ckpt_root / set $CKPT_ROOT to where the "
                  f"adapters actually live.")
        models.append((f"pyrorej_all_s{s}_bracket", ckpt, False))

    paths = []
    for tag, ckpt, pretrained in models:
        for dataset in DATASETS:
            paths.append(build_job(tag, ckpt, dataset, pretrained))

    wrapper = f"{HERE}/submit_morebench_all.sh"
    write_script(wrapper, "#!/bin/bash\n" +
                 "".join(f"sbatch {p}\n" for p in paths))
    paths.append(wrapper)

    for p in paths:
        print(f"wrote {p}")
    if args.submit:
        for p in paths[:-1]:
            print(subprocess.run(["sbatch", p], capture_output=True,
                                 text=True).stdout.strip())
    else:
        print(f"\nNot submitted. Run: bash {wrapper}")


if __name__ == "__main__":
    main()
