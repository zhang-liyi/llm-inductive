"""Submit one slurm job per scenario that runs the rejection-sampling variant.

NUTS source programs live in ``benchmarking/programs/``. For each scenario we
convert the NUTS source to a rejection-sampling program on the fly (via
``utils.nuts_to_rej.convert_text``), patch in the seed and output path, and
submit a slurm job. Rejection sampling is i.i.d., so there's no benefit to
multiple seeds/chains -- each job draws 4000 accepted samples and writes a
combined-posterior-format JSON directly.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time

# Make sibling `utils/` importable when this script is run from the repo root
# (e.g. ``python benchmarking/run_rejection_programs.py``).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from utils.nuts_to_rej import convert_text  # noqa: E402

BENCH_DIR = './benchmarking'
SRC_DIR = f'{BENCH_DIR}/programs'
RESULTS_DIR = f'{BENCH_DIR}/inference_results'
LOGS_DIR = f'{BENCH_DIR}/logs'
SLURM_DIR = f'{BENCH_DIR}/slurm_scripts'

# Per-mode runtime artefacts. The on-the-fly converted source for each
# scenario is patched (seed/out_path) and dropped here, then sbatch'd.
MODE_DIRS = {
    'hard': {
        'patched': f'{BENCH_DIR}/programs_rejection_patched',
        'suffix':  'rej-hard-combined',
    },
    'soft': {
        'patched': f'{BENCH_DIR}/programs_rejection_soft_patched',
        'suffix':  'rej-soft-combined',
    },
}

NUM_SAMPLES = 4000

SLURM_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --time=02:59:59
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --output={log_path}
#SBATCH --mail-type=fail
#SBATCH --mail-user=anonymous@example.com

cd {bench_dir}
python {program_path}
"""


def patch_program(src_text, out_path, seed):
    text = src_text

    if re.search(r'pyro\.set_rng_seed\s*\(', text):
        text = re.sub(r'pyro\.set_rng_seed\s*\([^)]*\)',
                      f'pyro.set_rng_seed({seed})', text, count=1)

    out_path_literal = f'out_path = "{out_path}"'
    new_text, n = re.subn(
        r'out_path\s*=\s*[^\n]+', out_path_literal, text, count=1,
    )
    if n == 0:
        new_text = text + (
            '\n\nimport json as _json, os as _os\n'
            f'_os.makedirs({json.dumps(os.path.dirname(out_path))}, exist_ok=True)\n'
            f'out_path = "{out_path}"\n'
            "with open(out_path, 'w') as _f:\n"
            "    _json.dump({'seeds': [0], 'queries': result}, _f)\n"
        )
    return new_text


def write_patched_program(scenario_id, seed, mode):
    dirs = MODE_DIRS[mode]
    src_path = f'{SRC_DIR}/pg-{scenario_id}.py'
    if not os.path.isfile(src_path):
        return None, None
    with open(src_path) as f:
        src = f.read()
    rej_src = convert_text(src, mode=mode, num_samples=NUM_SAMPLES,
                           wrap_json=True)
    out_json = f'{RESULTS_DIR}/result-{scenario_id}-{dirs["suffix"]}.json'
    patched = patch_program(rej_src, out_json, seed)
    os.makedirs(dirs['patched'], exist_ok=True)
    patched_path = f'{dirs["patched"]}/pg-{scenario_id}.py'
    with open(patched_path, 'w') as f:
        f.write(patched)
    return patched_path, out_json


def submit_slurm(scenario_id, program_path, mode, dry_run=False):
    tag = f'rej-{mode}'
    job_name = f'{tag}-{scenario_id[:40]}'
    log_path = f'{LOGS_DIR}/{scenario_id}-{tag}.out'
    script_path = f'{SLURM_DIR}/{scenario_id}-{tag}.sh'
    script = SLURM_TEMPLATE.format(
        job_name=job_name,
        log_path=log_path,
        bench_dir=BENCH_DIR,
        program_path=program_path,
    )
    os.makedirs(SLURM_DIR, exist_ok=True)
    with open(script_path, 'w') as f:
        f.write(script)
    os.chmod(script_path, 0o755)
    if dry_run:
        print(f'[dry-run] sbatch {script_path}')
    else:
        subprocess.check_call(['sbatch', script_path])
        time.sleep(0.05)


def submit_all(mode, dry_run=False, scenarios=None, seed=0, force=False):
    with open(f'{BENCH_DIR}/msa_cogsci_human_data.json') as f:
        human = json.load(f)
    scenario_ids = list(human['e2_implicit'].keys())
    if scenarios:
        scenario_ids = [s for s in scenario_ids if s in scenarios]

    n_submitted = 0
    for sid in scenario_ids:
        program_path, out_json = write_patched_program(sid, seed, mode)
        if program_path is None:
            print(f'No source program for {sid}; skipping.')
            continue
        if os.path.isfile(out_json) and not force:
            print(f'[SKIP] {sid}: {out_json} exists')
            continue
        submit_slurm(sid, program_path, mode, dry_run=dry_run)
        n_submitted += 1
    print(f'Submitted {n_submitted} {mode} job(s).')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=tuple(MODE_DIRS.keys()), default='hard',
                   help='Which rejection-sampling variant to submit (hard=prior '
                        'rejection on diff>0; soft=importance-weighted on the '
                        'same target as NUTS).')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--scenarios', nargs='*', default=None)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--force', action='store_true')
    args = p.parse_args()
    submit_all(mode=args.mode, dry_run=args.dry_run, scenarios=args.scenarios,
               seed=args.seed, force=args.force)


if __name__ == '__main__':
    main()
