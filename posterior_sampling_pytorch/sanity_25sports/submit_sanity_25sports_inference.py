"""Submit one SLURM job per scenario in sanity_25sports_queries.json.
Each job runs rejection sampling on the matching pg-gemini-REJ-{...}.py
program once and computes bins for the 4 strict-different queries.
Cap of 0:59:59 per program (short-queue lane); skip if result JSON already exists.
"""
import json
import os
import subprocess
import time

ROOT = './posterior_sampling_pytorch'
QUERIES_JSON = f'{ROOT}/sanity_25sports/sanity_25sports_queries.json'
RESULTS_DIR = f'{ROOT}/sanity_25sports/inference_results'
SUBMIT_DIR = f'{ROOT}/sanity_25sports/submit_per_scenario'
ARCHIVE_DIR = f'{ROOT}/sanity_25sports/archive'

TEMPLATE = """#!/bin/bash
#SBATCH --job-name=san25rej-{stem}
#SBATCH --time=00:59:59
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --output={archive}/{stem}.out
#SBATCH --mail-type=fail
#SBATCH --mail-user=anonymous@example.com

cd {root}
python run_sanity_25sports_one.py {sid}
"""


def main():
    for d in (RESULTS_DIR, SUBMIT_DIR, ARCHIVE_DIR):
        os.makedirs(d, exist_ok=True)
    spec = json.load(open(QUERIES_JSON))
    n_skip = n_submit = 0
    for s in spec:
        sid = s['scenario_id']
        if os.path.isfile(f'{RESULTS_DIR}/{sid}.json'):
            n_skip += 1
            continue
        # SLURM job names need to be reasonably short; use the seed-suffix end.
        stem = sid.split('gemini-', 1)[-1]
        sh = f'{SUBMIT_DIR}/{stem}.sh'
        with open(sh, 'w') as f:
            f.write(TEMPLATE.format(stem=stem, archive=ARCHIVE_DIR,
                                    root=ROOT, sid=sid))
        os.chmod(sh, 0o755)
        subprocess.check_call(['sbatch', sh])
        time.sleep(0.05)
        n_submit += 1
    print(f'submitted={n_submit}  skipped_existing={n_skip}')


if __name__ == '__main__':
    main()
