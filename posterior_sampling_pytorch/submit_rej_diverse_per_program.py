"""Submit one SLURM job per pg-gemini-REJ-diverse-*.py program with a 2h cap.

The original temp-REJ-diverse-*.sh scripts batched 30 programs sequentially
under a single 24h walltime, so a single low-acceptance run at the start of
a batch killed every program after it. This submitter atomizes the run:
each (P,C,R,N,seed) program gets its own SLURM job with --time=02:00:00.
A stuck low-acceptance program just dies after 2h and frees the slot —
the rest of the seeds still get to run.

Skipping rule: a program is considered "done" if any
``inference_results/result-gemini-REJ-diverse-P-{P}-C-{C}-R-{R}-N-{N}-{S}-*.json``
exists (the trailing piece is the build-time timestamp baked into the program's
hardcoded out_path; it differs across the 165011 / 165142 / 165148 batches).

Usage:
    python submit_rej_diverse_per_program.py --dry-run
    python submit_rej_diverse_per_program.py
    python submit_rej_diverse_per_program.py --motif P-0-C-0-R-0-N-0
    python submit_rej_diverse_per_program.py --max 200
"""

import argparse
import glob
import os
import re
import subprocess
import time

ROOT = './posterior_sampling_pytorch'
PROGRAMS_DIR = f'{ROOT}/programs'
RESULTS_DIR = f'{ROOT}/inference_results'
ARCHIVE_DIR = f'{ROOT}/archive'
SUBMIT_DIR = f'{ROOT}/submit_per_program'

PG_RE = re.compile(
    r'^pg-gemini-REJ-diverse-'
    r'P-(\d+)-C-(\d+)-R-(\d+)-N-(\d+)-(\d+)\.py$'
)

SLURM_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=REJ-diverse-{motif}-S{seed}
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=6G
#SBATCH --output={archive}/{stem}.out
#SBATCH --mail-type=fail
#SBATCH --mail-user=anonymous@example.com

cd {root}
python programs/{stem}.py
"""


def parse_program_filename(fname):
    m = PG_RE.match(fname)
    if not m:
        return None
    P, C, R, N, S = (int(x) for x in m.groups())
    motif = f'P-{P}-C-{C}-R-{R}-N-{N}'
    return motif, S


def already_done(motif, seed):
    glob_pat = f'{RESULTS_DIR}/result-gemini-REJ-diverse-{motif}-{seed}-*.json'
    matches = glob.glob(glob_pat)
    return any(os.path.getsize(p) > 0 for p in matches)


def submit_one(stem, motif, seed, dry_run=False):
    os.makedirs(SUBMIT_DIR, exist_ok=True)
    os.makedirs(ARCHIVE_DIR, exist_ok=True)
    sh_path = f'{SUBMIT_DIR}/{stem}.sh'
    with open(sh_path, 'w') as f:
        f.write(SLURM_TEMPLATE.format(
            motif=motif, seed=seed, stem=stem,
            archive=ARCHIVE_DIR, root=ROOT,
        ))
    os.chmod(sh_path, 0o755)
    if dry_run:
        print(f'[dry-run] sbatch {sh_path}')
    else:
        subprocess.check_call(['sbatch', sh_path])
        time.sleep(0.02)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--motif', default=None,
                   help='Only submit programs for this motif (e.g. P-0-C-0-R-0-N-0).')
    p.add_argument('--max', type=int, default=None,
                   help='Cap the total number of jobs submitted (debug).')
    p.add_argument('--force', action='store_true',
                   help='Submit even if a result JSON for this seed already exists.')
    args = p.parse_args()

    fnames = sorted(os.listdir(PROGRAMS_DIR))
    n_eligible = n_skip_done = n_skip_filter = n_submit = 0
    for fname in fnames:
        parsed = parse_program_filename(fname)
        if parsed is None:
            continue
        motif, seed = parsed
        n_eligible += 1
        if args.motif and motif != args.motif:
            n_skip_filter += 1
            continue
        if not args.force and already_done(motif, seed):
            n_skip_done += 1
            continue
        stem = fname[:-len('.py')]
        submit_one(stem, motif, seed, dry_run=args.dry_run)
        n_submit += 1
        if args.max is not None and n_submit >= args.max:
            print(f'[max={args.max}] stopping')
            break

    print(f'eligible={n_eligible}  '
          f'skipped_done={n_skip_done}  '
          f'skipped_filter={n_skip_filter}  '
          f'submitted={n_submit}')


if __name__ == '__main__':
    main()
