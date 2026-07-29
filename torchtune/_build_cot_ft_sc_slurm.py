"""
Emit + submit SLURM scripts for **fine-tuned model + CoT + self-consistency**.

This is the cell `REBUTTAL_STATUS.md` calls "the single highest-value remaining
experiment on line 1".  The existing rows leave one confound standing:

    Base + CoT + SC     k=10 sampled paths, mixture-of-softmaxes
    Posterior + CoT     1 greedy path

so every Base-SC-beats-fine-tuning cell (SC currently wins ECE 7/7, accuracy
6/7, NLL 4-3) is also a k=10-beats-k=1 cell.  Running the fine-tuned
checkpoints under the *same* decoding scheme removes it, and is the fairer
comparison the reviewer's framing actually asks for.

Clone of `_build_cot_sc_persample_slurm.py` (one path per job, recombined by
`combine_sc_samples.py`) with three changes:

  * **fine-tuned checkpoints** — `--ckpt_dir` from `CKPT_FMT`, looped over
    seeds, so the row can carry a seed mean like every other fine-tuned row.
  * **path-major job ordering** — jobs are emitted path 0 of every benchmark,
    then path 1 of every benchmark, ... instead of benchmark-by-benchmark.
    Under a drip this matters: an interruption leaves *every* benchmark at the
    same k, which is a publishable row at k=7, whereas benchmark-major
    ordering leaves some benchmarks at k=10 and others at k=0, which is not.
  * **one long job per (benchmark, path)** — see below.

Sharding: the "seed 3" layout
-----------------------------
The defaults here are the ones the seed-3 run settled on, not the original
per-shard ones.  `--budget_min 720` is the smallest budget that makes the most
expensive benchmark (BT-guided, 2238 examples x 18.9 s = 11.8 h) fit in a
single shard, so **nothing is sharded at all**: the batch is exactly

    8 benchmarks x 5 paths = 40 jobs per seed

Two reasons this beats the many-short-jobs layout:

  * 40 <= gpu-short's 44-job-per-user cap, so a whole seed runs as one
    concurrent wave and waits in the queue once rather than five times.
  * Shard-boundary rounding disappears, which cut total compute from 154 to
    ~137 GPU-h per seed.

The walltime is `23:59:59` — the top of the gpu-short bracket.  Anything in
(1 h, 24 h] lands in gpu-short; asking for the full 24 h costs nothing extra
here because the batch is already at the concurrency cap, and it leaves the
11.8 h BT-guided job a wide margin.  Do **not** drop it to ~12 h: that leaves
about 15 min of headroom on that job, and the per-example rates in `RATE` were
calibrated on the *base* model.

Consequence for downstream code: **outputs carry no `_shard` suffix**
(`{tag}_{bench}_s{j}.json`).  `combine_sc_samples.py` groups by `global_idx`
so it handles both layouts, but an ad-hoc `*_shard*.json` glob would miss
every file this builder writes.

Decoding is paired with the base row by construction: `sc_seed` keys the RNG
on (base_seed, absolute dataset index, path index) and ignores the model, so
path j of example i is drawn from the same seed here as in `cot_sc`.  Keep
`--seed 0` unless you mean to break that.

Cost: 40 jobs and ~137 GPU-hours per seed.

Usage
-----
    python _build_cot_ft_sc_slurm.py --seeds 1            # write + report
    python _build_cot_ft_sc_slurm.py --seeds 1 --submit   # write + sbatch
"""
import argparse
import math
import os
import stat
import subprocess
import time

HERE = os.path.dirname(os.path.abspath(__file__))
# Resolved at emit time, like the MoReBench builder: this script *writes* other
# scripts, so it cannot rely on the repo-wide `sed s|<DATA_ROOT>|...|` pass.
BASE_DIR = os.path.dirname(HERE)
EVAL_DIR = f"{BASE_DIR}/data_evaluation"
RESULTS = f"{EVAL_DIR}/results"
ARCHIVE = f"{BASE_DIR}/archive"
BT_DATA = f"{BASE_DIR}/data_processing/bayesian_teaching_test_base.jsonl"
SCRIPT_DIR = f"{HERE}/slurm_cmd/cot_ft_sc"

CKPT_ROOT = os.environ.get("CKPT_ROOT", f"{BASE_DIR}/torchtune/ckpt/llama3_8B")
CKPT_PREFIX = os.environ.get("CKPT_PREFIX", "pyro_rej")
CKPT_FMT = (CKPT_ROOT + "/" + CKPT_PREFIX +
            "_lora_dist_r8_all_seed{S}_bracket_lora8_dist/epoch_0")
# Deliberately NOT "cot_sc_..." — `combine_sc_samples.py --auto --tag cot_sc`
# globs `cot_sc_*_s*.json`, which would sweep these files into the base row's
# run.  They land in separate groups so nothing is corrupted, but the base
# combine would silently gain eight extra output files.
TAG_FMT = "cotsc_pyrorej_all_s{S}_bracket"

# Measured k=1 seconds per example from completed base-model CoT/SC shards.
# One job runs ONE path, so shard size = budget / rate (no k factor).  The
# seed-1 FT smoke test measured mean_gen_tokens 200.4 against the base model's
# 207.9, so these transfer at --rate_scale 1.0.
RATE = {
    "bt_base_tf": 13.5,
    "bayesian_teaching_base_guided": 18.9,
    "openestimate": 10.0,
    "mmlu": 10.1, "truthfulqa": 10.1, "hellaswag": 10.1,
    "winogrande": 10.1, "arc_challenge": 10.1,
}
MCQ_TASKS = ("mmlu", "truthfulqa", "hellaswag", "winogrande", "arc_challenge")
TOTALS = {"bt_base_tf": 2238, "bayesian_teaching_base_guided": 2238,
          "openestimate": 181}

SBATCH_HEADER = """\
#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --time={walltime}
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=50G
"""

JOB_PREFIX = "sf"


def write_script(path, body):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(body)
    os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def rate(args, name):
    return RATE[name] * args.rate_scale


def n_per_shard(args, name):
    """Examples per job so one path finishes inside the compute budget."""
    return max(1, int(args.budget_min * 60 / rate(args, name)))


def shards(total, per):
    n = max(1, math.ceil(total / per))
    for k in range(n):
        s = k * per
        yield k, s, min(per, total - s), n


def emit(args, seed, j, sub, name, script, extra, total, max_new, jobkey):
    """Scripts for one (seed, path) x one benchmark, sharded over examples."""
    tag = TAG_FMT.format(S=seed)
    ckpt = CKPT_FMT.format(S=seed)
    per = n_per_shard(args, name)
    out_scripts = []
    for k, s, n, n_shards in shards(total, per):
        suffix = "" if n_shards == 1 else f"_shard{k}"
        out = f"{RESULTS}/{sub}/{tag}_{name}_s{j}{suffix}.json"
        tail = "" if n_shards == 1 else f"_{k}"
        log = f"{ARCHIVE}/{tag}_{jobkey}_s{j}{tail}.out"
        body = SBATCH_HEADER.format(
            job_name=f"{JOB_PREFIX}{seed}-{jobkey[:7]}-s{j}{tail}",
            walltime=args.walltime) + (
            f"cd {EVAL_DIR}\n"
            f"python {EVAL_DIR}/{script} \\\n"
            f"    --ckpt_dir {ckpt} \\\n"
            f"    --sample_idx {j} \\\n"
            f"    --sc_defaults \\\n"
            f"    --seed {args.seed} \\\n"
            f"{extra}"
            f"    --start_idx {s} \\\n"
            f"    --n_examples {n} \\\n"
            f"    --max_new_tokens {max_new} \\\n"
            f"    --output_file {out} \\\n"
            f"    >{log} 2>&1\n"
        )
        path = f"{SCRIPT_DIR}/submit_{tag}_{jobkey}_s{j}{tail}.sh"
        write_script(path, body)
        out_scripts.append((path, out, name, n))
    return out_scripts


def emit_benchmarks(args, seed, j):
    """Every benchmark for one (seed, path), in cheapest-first order."""
    jobs = []
    if args.only is None or "oe" in args.only:
        jobs += emit(args, seed, j, "openestimate", "openestimate",
                     "evaluate_openestimate_cot.py", "    --split all \\\n",
                     TOTALS["openestimate"], args.max_new_tokens, "oe")
    for ds in MCQ_TASKS:
        if args.only is not None and ds not in args.only:
            continue
        total = 299 if ds == "arc_challenge" else args.subsample_n
        extra = (f"    --dataset {ds} \\\n"
                 f"    --subsample_n {args.subsample_n} \\\n"
                 f"    --subsample_seed {args.subsample_seed} \\\n")
        jobs += emit(args, seed, j, "text_cls", ds,
                     "evaluate_text_classification_cot.py", extra,
                     total, args.max_new_tokens, ds)
    if args.only is None or "bt" in args.only:
        jobs += emit(args, seed, j, "bayesian_teaching", "bt_base_tf",
                     "evaluate_bayesian_teaching_cot.py",
                     f"    --data_path {BT_DATA} \\\n",
                     TOTALS["bt_base_tf"], args.max_new_tokens, "btn")
    if args.only is None or "bt_guided" in args.only:
        jobs += emit(args, seed, j, "bayesian_teaching",
                     "bayesian_teaching_base_guided",
                     "evaluate_bayesian_teaching_cot.py",
                     f"    --data_path {BT_DATA} \\\n    --guided \\\n",
                     TOTALS["bayesian_teaching_base_guided"],
                     args.max_new_tokens_guided, "btg")
    return jobs


def build_all(args):
    """Path-major: every benchmark at path 0, then path 1, ...

    So an interrupted run yields a uniform k across benchmarks rather than a
    ragged one.  Seeds are the outer loop — seed 1 complete beats three seeds
    at k=3.
    """
    jobs = []
    for seed in args.seeds:
        for j in range(args.n_samples):
            jobs += emit_benchmarks(args, seed, j)
    return jobs


def n_queued(prefix=JOB_PREFIX):
    out = subprocess.run(["squeue", "-u", os.environ.get("USER", ""), "-h",
                          "-o", "%j"], capture_output=True, text=True).stdout
    return sum(1 for line in out.splitlines() if line.strip().startswith(prefix))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1],
                    help="Fine-tuned checkpoint seeds. 40 jobs per seed; "
                         "submit one seed at a time so each fits the 44-job cap.")
    ap.add_argument("--n_samples", type=int, default=5,
                    help="Reasoning paths k. 5 is what this line runs at.")
    ap.add_argument("--seed", type=int, default=0,
                    help="SC base seed. 0 pairs the drawn paths with the "
                         "base `cot_sc` row; change it and they diverge.")
    ap.add_argument("--subsample_n", type=int, default=500)
    ap.add_argument("--subsample_seed", type=int, default=1234)
    ap.add_argument("--budget_min", type=float, default=720.0,
                    help="Target compute minutes per job. 720 is the smallest "
                         "value that leaves every benchmark unsharded, giving "
                         "40 jobs per seed.")
    ap.add_argument("--rate_scale", type=float, default=1.0,
                    help="Multiply the measured base-model per-example rates.")
    ap.add_argument("--walltime", default="23:59:59",
                    help="Top of the gpu-short bracket (44 concurrent). The "
                         "longest job here is BT-guided at ~11.8 h, so this is "
                         "2x headroom. Do not drop below ~14h.")
    ap.add_argument("--email", default=os.environ.get("EMAIL", ""),
                    help="SLURM --mail-user address.")
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--max_new_tokens_guided", type=int, default=768)
    ap.add_argument("--only", nargs="+", default=None)
    ap.add_argument("--submit", action="store_true",
                    help="sbatch every pending job at once. Correct for the "
                         "40-job layout; use --drip only if you rebuild with "
                         "a smaller --budget_min.")
    ap.add_argument("--drip", action="store_true",
                    help="Submit continuously, keeping at most --max_queued "
                         "of this batch's jobs in the queue.")
    ap.add_argument("--max_queued", type=int, default=60)
    ap.add_argument("--poll", type=int, default=120)
    ap.add_argument("--force", action="store_true",
                    help="Re-submit even where a non-empty output exists.")
    args = ap.parse_args()

    if args.email:
        global SBATCH_HEADER
        SBATCH_HEADER += (f"#SBATCH --mail-type=end\n"
                          f"#SBATCH --mail-user={args.email}\n")

    for seed in args.seeds:
        ckpt = CKPT_FMT.format(S=seed)
        if not os.path.isdir(ckpt):
            print(f"WARN: checkpoint not found: {ckpt}\n"
                  f"      set $CKPT_ROOT / $CKPT_PREFIX to where the "
                  f"fine-tuned models actually live.")

    jobs = build_all(args)
    print(f"Wrote {len(jobs)} submit scripts under {SCRIPT_DIR}")
    by = {}
    for _, _, name, n in jobs:
        d = by.setdefault(name, [0, n_per_shard(args, name), 0])
        d[0] += 1
        d[2] = max(d[2], n)
    print(f"\n{'benchmark':<32}{'jobs':>6}{'ex/shard':>10}{'est min/job':>13}")
    gpu_h = 0.0
    for name, (cnt, per, mx) in sorted(by.items()):
        est = mx * rate(args, name) / 60 + 3
        gpu_h += cnt * est / 60
        print(f"{name:<32}{cnt:>6}{per:>10}{est:>13.1f}")
    print(f"{'TOTAL':<32}{len(jobs):>6}{'':>10}{gpu_h:>12.0f}h")

    pending = [(p, o) for p, o, _, _ in jobs
               if args.force or not (os.path.isfile(o) and os.path.getsize(o) > 0)]
    print(f"{len(jobs) - len(pending)} of {len(jobs)} outputs already exist.")

    if args.submit:
        # 40 jobs is under the 44-job cap, so there is nothing to meter.
        # Re-running is safe: anything with a non-empty output is skipped.
        n = 0
        for path, _ in pending:
            subprocess.check_call(["sbatch", path], stdout=subprocess.DEVNULL)
            n += 1
        print(f"\nSubmitted {n} jobs.")
        for seed in args.seeds:
            print(f"Then: python {EVAL_DIR}/combine_sc_samples.py --auto "
                  f"--tag {TAG_FMT.format(S=seed)}")
        return

    if not args.drip:
        print("\nRe-run with --submit to sbatch these.")
        return

    print(f"\nDripping {len(pending)} jobs, max {args.max_queued} queued at a time.")
    submitted = 0
    while pending:
        room = args.max_queued - n_queued()
        if room <= 0:
            time.sleep(args.poll)
            continue
        for _ in range(min(room, len(pending))):
            path, out = pending.pop(0)
            try:
                subprocess.check_call(["sbatch", path], stdout=subprocess.DEVNULL)
                submitted += 1
            except subprocess.CalledProcessError:
                pending.insert(0, (path, out))
                break
        print(f"  submitted {submitted}/{submitted + len(pending)}", flush=True)
        if pending:
            time.sleep(args.poll)
    print(f"All {submitted} jobs submitted.")
    for seed in args.seeds:
        print(f"Then: python {EVAL_DIR}/combine_sc_samples.py --auto "
              f"--tag {TAG_FMT.format(S=seed)}")


if __name__ == "__main__":
    main()
