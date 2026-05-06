"""Submit Slurm jobs that run the pyro MCMC programs and dump posterior samples.

For a given (domain, diverse_mode, sampling) configuration this enumerates
every program combination that exists on disk, edits the `out_path` line of
each program to a timestamped JSON filename, and submits one sbatch job per
(P, C, R, Nind, Nepi) group (or (P, C, R) group in diverse mode). Each job
loops sequentially over all N values and all seeds.
"""
import argparse
import datetime
import glob
import os
import subprocess
import time


def parse_args():
    parser = argparse.ArgumentParser(description='Inductive-LLM')
    parser.add_argument('--diverse_mode',
                        default=True,
                        type=lambda x: str(x).lower() == 'true')
    parser.add_argument('--base_seed', default=0, type=int)
    parser.add_argument('--num_seeds', default=5, type=int,
                        help='Number of consecutive seeds per program, '
                             'starting at base_seed.')
    parser.add_argument('--domain',
                        default='sports',
                        choices=['sports', 'healthcare', 'general'])
    parser.add_argument('--sampling',
                        default='NUTS',
                        choices=['NUTS', 'REJ'],
                        help='NUTS runs the original MCMC programs; REJ runs '
                             'the rejection-sampling variants in '
                             'programs/pg-gemini-REJ-*.py.')
    parser.add_argument('--per_seed_jobs', action='store_true',
                        help='One sbatch per (group_id, seed): the job loops '
                             'over the N values only. Default packs all '
                             '(N, seed) pairs into a single sbatch per '
                             'group_id.')
    parser.add_argument('--walltime', default='23:59:59',
                        help='SLURM --time. NUTS multi-chain CPU runs may '
                             'need 71:59:59.')
    parser.add_argument('--cpus_per_task', default=1, type=int,
                        help='SLURM --cpus-per-task. Pyro NUTS num_chains>1 '
                             'parallelizes across CPUs; pass 4 for '
                             'num_chains=4 in the diverse programs.')
    parser.add_argument('--mem_per_cpu', default='6G',
                        help='SLURM --mem-per-cpu (e.g. 6G, 8G).')
    return parser.parse_args()


args = parse_args()

DIVERSE_MODE = args.diverse_mode
BASE_SEED = args.base_seed
NUM_SEEDS = args.num_seeds
DOMAIN = args.domain
SAMPLING = args.sampling
PER_SEED_JOBS = args.per_seed_jobs
WALLTIME = args.walltime
CPUS_PER_TASK = args.cpus_per_task
MEM_PER_CPU = args.mem_per_cpu

tm = str(datetime.datetime.now())
TMSTR = tm[:10] + '-' + tm[11:13] + tm[14:16] + tm[17:19]


def _domain_infix(domain):
    return '' if domain == 'sports' else f'{domain}-'


# ── Build groups ─────────────────────────────────────────────────────────────
# Each group is one sbatch job. Every (N, seed) combination inside a group is
# run sequentially in that job. group_id identifies the group (omits N so a
# single job can iterate over all N values).
#   diverse:     group_id = '{domain_infix}diverse-P-{P}-C-{C}-R-{R}'
#                full program_id = '{group_id}-N-{N}'
#   non-diverse: group_id = '{domain_infix}P-{P}-C-{C}-R-{R}-Nind-{Nind}-Nepi-{Nepi}'
#                full program_id = '{domain_infix}P-{P}-C-{C}-R-{R}-N-{N}-Nind-{Nind}-Nepi-{Nepi}'
groups = []  # list of (group_id, [program_id, ...] )

if DIVERSE_MODE:
    for P in range(4):
        for C in range(2):
            for R in range(2):
                group_id = f'{_domain_infix(DOMAIN)}diverse-P-{P}-C-{C}-R-{R}'
                program_ids = [f'{group_id}-N-{N}' for N in range(6)]
                groups.append((group_id, program_ids))
else:
    for P in range(4):
        for C in range(2):
            for R in range(2):
                for N_ind in [1, 2]:
                    for N_epi in [1, 2, 3, 4]:
                        group_id = (
                            f'{_domain_infix(DOMAIN)}'
                            f'P-{P}-C-{C}-R-{R}-'
                            f'Nind-{N_ind}-Nepi-{N_epi}'
                        )
                        program_ids = [
                            f'{_domain_infix(DOMAIN)}'
                            f'P-{P}-C-{C}-R-{R}-N-{N}-'
                            f'Nind-{N_ind}-Nepi-{N_epi}'
                            for N in range(8)
                        ]
                        groups.append((group_id, program_ids))


def _program_filename(program_id, seed):
    prefix = 'pg-gemini-REJ-' if SAMPLING == 'REJ' else 'pg-gemini-'
    return f'programs/{prefix}{program_id}-{seed}.py'


def _result_filename(program_id, seed, tmstr):
    return (f'inference_results/result-gemini-{SAMPLING}-'
            f'{program_id}-{seed}-{tmstr}.json')


def _already_has_result(program_id, seed):
    pattern = (f'inference_results/result-gemini-{SAMPLING}-'
               f'{program_id}-{seed}-*.json')
    return bool(glob.glob(pattern))


def write_run_sequential(job_tag, pending):
    """Write and submit an sbatch script that runs every (program_id, seed)
    in `pending` sequentially, in order. `job_tag` is the SLURM job-name
    suffix and the script-filename infix (group_id alone, or
    f'{group_id}-s{seed}' under --per_seed_jobs).
    """
    if not pending:
        return

    script_name = f'temp-{SAMPLING}-{job_tag}-{TMSTR}.sh'
    prefix = 'pg-gemini-REJ-' if SAMPLING == 'REJ' else 'pg-gemini-'
    with open(script_name, 'w') as f:
        f.write(
            f"#!/bin/bash\n"
            f"#SBATCH --job-name={SAMPLING}-{job_tag}\n"
            f"#SBATCH --time={WALLTIME}\n"
            "#SBATCH --nodes=1\n"
            "#SBATCH --ntasks=1\n"
            f"#SBATCH --cpus-per-task={CPUS_PER_TASK}\n"
            f"#SBATCH --mem-per-cpu={MEM_PER_CPU}\n"
            "#SBATCH --mail-type=fail\n"
            "#SBATCH --mail-user=anonymous@example.com
        )
        for program_id, seed in pending:
            cmd = f"python programs/{prefix}{program_id}-{seed}.py"
            cmd += (f" >archive/{prefix}{program_id}-{seed}-{TMSTR}.out\n")
            f.write(cmd)

    subprocess.call(f'chmod +x {script_name}', shell=True)
    time.sleep(0.05)
    subprocess.call(f'sbatch {script_name}', shell=True)


def _prepare_one(program_id, seed):
    """Edit the program's out_path to a timestamped result JSON. Returns
    True if the program was prepared (file exists, no prior result), False
    otherwise."""
    program_filename = _program_filename(program_id, seed)
    if not os.path.isfile(program_filename):
        return False
    if _already_has_result(program_id, seed):
        return False
    result_json_filename = _result_filename(program_id, seed, TMSTR)
    with open(program_filename, "r", encoding="utf-8") as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if line.strip().startswith("out_path"):
            lines[i] = f'    out_path = "{result_json_filename}"\n'
            break
    with open(program_filename, "w", encoding="utf-8") as f:
        f.writelines(lines)
    return True


if PER_SEED_JOBS:
    # One sbatch per (group_id, seed); the job loops the 6 N's.
    for group_id, program_ids in groups:
        for seed in range(BASE_SEED, BASE_SEED + NUM_SEEDS):
            pending = []
            for program_id in program_ids:
                if _prepare_one(program_id, seed):
                    pending.append((program_id, seed))
            write_run_sequential(f'{group_id}-s{seed}', pending)
else:
    # Original behavior: pack all (program_id, seed) for a group into one sbatch.
    for group_id, program_ids in groups:
        pending = []
        for program_id in program_ids:
            for seed in range(BASE_SEED, BASE_SEED + NUM_SEEDS):
                if _prepare_one(program_id, seed):
                    pending.append((program_id, seed))
        write_run_sequential(group_id, pending)
