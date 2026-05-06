"""Submit slurm jobs to run the generated pyro programs.

For each scenario in e2_implicit we produce 4 per-seed copies of the program
(seeds 0..3), each with ``num_chains=1``, and submit one slurm job per copy.
After all jobs finish, call :func:`combine_seeds` (or run this script with
``--combine``) to merge the 4 seeds into a single posterior per scenario.

Example slurm script (written to ``benchmarking/slurm_scripts/<id>-seed<k>.sh``):

    #!/bin/bash
    #SBATCH --job-name=pg-<scenario_id>-s<seed>
    #SBATCH --time=23:59:59
    #SBATCH --nodes=1
    #SBATCH --ntasks=1
    #SBATCH --cpus-per-task=1
    #SBATCH --mem-per-cpu=8G
    #SBATCH --output=benchmarking/logs/<scenario_id>-s<seed>.out

    cd ./benchmarking
    python programs_seeded/pg-<scenario_id>-s<seed>.py
"""

import argparse
import json
import os
import re
import subprocess
import time

BENCH_DIR = './benchmarking'
RESULTS_DIR = f'{BENCH_DIR}/inference_results'
LOGS_DIR = f'{BENCH_DIR}/logs'
SLURM_DIR = f'{BENCH_DIR}/slurm_scripts'

# Inference modes. The default ``nuts`` mode reproduces the original
# 4-seed multi-chain combine flow.  ``rej-hard`` and ``rej-soft`` route
# program sources through the rejection-sampling trees and tag per-seed
# result/seeded paths so they don't collide with the NUTS files.
MODE_CONFIG = {
    'nuts':     {'programs': f'{BENCH_DIR}/programs',
                 'seeded':   f'{BENCH_DIR}/programs_seeded',
                 'tag':      None},
    'rej-hard': {'programs': f'{BENCH_DIR}/programs_rejection',
                 'seeded':   f'{BENCH_DIR}/programs_rejection_seeded',
                 'tag':      'rej-hard'},
    'rej-soft': {'programs': f'{BENCH_DIR}/programs_rejection_soft',
                 'seeded':   f'{BENCH_DIR}/programs_rejection_soft_seeded',
                 'tag':      'rej-soft'},
}

DEFAULT_NUTS_SEEDS = [0, 1, 2, 3]
DEFAULT_NUTS_WALLTIME = '23:59:59'
DEFAULT_REJ_SEEDS = list(range(15))
DEFAULT_REJ_WALLTIME = '02:00:00'

SLURM_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --time={walltime}
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


def per_seed_paths(scenario_id, seed, mode):
    """Return (out_json, seeded_program_path) for a (scenario, seed, mode)."""
    cfg = MODE_CONFIG[mode]
    tag = cfg['tag']
    suffix = f'-{tag}' if tag else ''
    out_json = f'{RESULTS_DIR}/result-{scenario_id}{suffix}-s{seed}.json'
    seeded_path = f'{cfg["seeded"]}/pg-{scenario_id}{suffix}-s{seed}.py'
    return out_json, seeded_path


def combined_path(scenario_id, mode):
    """Return the combined-result JSON path for a (scenario, mode)."""
    tag = MODE_CONFIG[mode]['tag']
    combined_suffix = f'-{tag}-combined' if tag else '-combined'
    return f'{RESULTS_DIR}/result-{scenario_id}{combined_suffix}.json'


def patch_program(src_program, out_path, seed):
    """Return program text with num_chains=1, seed set, out_path set."""
    text = src_program

    # Force single chain.
    text = re.sub(r'num_chains\s*=\s*\d+', 'num_chains=1', text)

    # Force a specific seed. Replace the first pyro.set_rng_seed(...) call;
    # if none exists, inject one right after the pyro import block.
    if re.search(r'pyro\.set_rng_seed\s*\(', text):
        text = re.sub(r'pyro\.set_rng_seed\s*\([^)]*\)',
                      f'pyro.set_rng_seed({seed})', text, count=1)
    else:
        text = re.sub(
            r'(import pyro[^\n]*\n)',
            r'\1pyro.set_rng_seed(' + str(seed) + ')\n',
            text, count=1,
        )

    # Rewrite the first out_path assignment (the one that controls the JSON
    # dump at the end of the generated program).
    out_path_literal = f'out_path = "{out_path}"'
    new_text, n = re.subn(
        r'out_path\s*=\s*[^\n]+', out_path_literal, text, count=1,
    )
    if n == 0:
        # The generator didn't emit an out_path. Append an explicit dump.
        new_text = text + (
            '\n\nimport json as _json, os as _os\n'
            f'_os.makedirs({json.dumps(os.path.dirname(out_path))}, exist_ok=True)\n'
            f'out_path = "{out_path}"\n'
            'with open(out_path, "w") as _f:\n'
            '    _json.dump(result, _f)\n'
        )
    return new_text


def write_seeded_program(scenario_id, seed, mode):
    cfg = MODE_CONFIG[mode]
    src_path = f'{cfg["programs"]}/pg-{scenario_id}.py'
    if not os.path.isfile(src_path):
        return None, None
    with open(src_path) as f:
        src = f.read()
    out_json, seeded_path = per_seed_paths(scenario_id, seed, mode)
    os.makedirs(os.path.dirname(seeded_path), exist_ok=True)
    patched = patch_program(src, out_json, seed)
    with open(seeded_path, 'w') as f:
        f.write(patched)
    return seeded_path, out_json


def submit_slurm(scenario_id, seed, program_path, mode, walltime, dry_run=False):
    tag = MODE_CONFIG[mode]['tag']
    suffix = f'-{tag}' if tag else ''
    job_name = f'pg-{scenario_id[:40]}{suffix}-s{seed}'
    log_path = f'{LOGS_DIR}/{scenario_id}{suffix}-s{seed}.out'
    script_path = f'{SLURM_DIR}/{scenario_id}{suffix}-s{seed}.sh'
    script = SLURM_TEMPLATE.format(
        job_name=job_name,
        log_path=log_path,
        walltime=walltime,
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


def filter_scenarios(scenario_ids, scenarios=None, name_contains=None):
    if scenarios:
        scenario_ids = [s for s in scenario_ids if s in scenarios]
    if name_contains:
        needles = [n.lower() for n in name_contains]
        scenario_ids = [s for s in scenario_ids
                        if all(n in s.lower() for n in needles)]
    return scenario_ids


def submit_all(mode='nuts', dry_run=False, scenarios=None, name_contains=None,
               force=False, seeds=None, walltime=None):
    if seeds is None:
        seeds = DEFAULT_REJ_SEEDS if mode != 'nuts' else DEFAULT_NUTS_SEEDS
    if walltime is None:
        walltime = DEFAULT_REJ_WALLTIME if mode != 'nuts' else DEFAULT_NUTS_WALLTIME

    with open(f'{BENCH_DIR}/msa_cogsci_human_data.json') as f:
        human = json.load(f)
    scenario_ids = filter_scenarios(
        list(human['e2_implicit'].keys()),
        scenarios=scenarios, name_contains=name_contains,
    )
    print(f'Mode={mode}  walltime={walltime}  seeds={seeds}  '
          f'scenarios={len(scenario_ids)}')

    n_submitted = 0
    n_skipped_existing = 0
    n_skipped_combined = 0
    n_missing_src = 0
    for sid in scenario_ids:
        # Scenario-level skip: combined JSON already covers all seeds.
        if not force and os.path.isfile(combined_path(sid, mode)):
            n_skipped_combined += len(seeds)
            continue
        for seed in seeds:
            out_json, _ = per_seed_paths(sid, seed, mode)
            if os.path.isfile(out_json) and not force:
                n_skipped_existing += 1
                continue
            program_path, _ = write_seeded_program(sid, seed, mode)
            if program_path is None:
                n_missing_src += 1
                continue
            submit_slurm(sid, seed, program_path, mode, walltime,
                         dry_run=dry_run)
            n_submitted += 1
    print(f'Submitted {n_submitted} job(s); skipped {n_skipped_combined} '
          f'with existing combined results; skipped {n_skipped_existing} '
          f'with existing per-seed results; {n_missing_src} missing source programs.')


def combine_seeds(mode='nuts', scenarios=None, name_contains=None, seeds=None):
    """Concatenate samples across seeds and write one JSON per scenario."""
    if seeds is None:
        seeds = DEFAULT_REJ_SEEDS if mode != 'nuts' else DEFAULT_NUTS_SEEDS

    tag = MODE_CONFIG[mode]['tag']
    suffix = f'-{tag}' if tag else ''
    combined_suffix = f'-{tag}-combined' if tag else '-combined'

    with open(f'{BENCH_DIR}/msa_cogsci_human_data.json') as f:
        human = json.load(f)
    scenario_ids = filter_scenarios(
        list(human['e2_implicit'].keys()),
        scenarios=scenarios, name_contains=name_contains,
    )

    import numpy as np
    for sid in scenario_ids:
        merged = {}
        ok_seeds = []
        for seed in seeds:
            path = f'{RESULTS_DIR}/result-{sid}{suffix}-s{seed}.json'
            if not os.path.isfile(path) or os.path.getsize(path) == 0:
                continue
            try:
                with open(path) as f:
                    d = json.load(f)
            except json.JSONDecodeError as e:
                print(f'  [WARN] {os.path.basename(path)}: invalid JSON ({e}); skipping')
                continue
            # Soft-rejection writes {"seeds": [...], "queries": {...}};
            # NUTS programs write the bare query dict. Normalize.
            queries = d.get('queries', d) if isinstance(d, dict) else {}
            ok_seeds.append(seed)
            for q, qd in queries.items():
                if q not in merged:
                    merged[q] = {
                        'description': qd.get('description', ''),
                        'samples': [],
                    }
                samples = qd.get('samples', [])
                merged[q]['samples'].extend(samples)
        for q, qd in merged.items():
            arr = np.asarray(qd['samples'], dtype=float)
            qd['mean'] = float(arr.mean()) if arr.size else float('nan')
            qd['std'] = float(arr.std()) if arr.size else float('nan')
            qd['n_samples'] = int(arr.size)
        out_path = f'{RESULTS_DIR}/result-{sid}{combined_suffix}.json'
        with open(out_path, 'w') as f:
            json.dump({'seeds': ok_seeds, 'queries': merged}, f)
        print(f'{sid}: combined {len(ok_seeds)} seed(s) -> {out_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=tuple(MODE_CONFIG.keys()),
                        default='nuts',
                        help='nuts (default, NUTS programs in programs/) or '
                             'rej-hard / rej-soft (rejection-sampling programs).')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--combine', action='store_true',
                        help='Merge existing per-seed JSONs instead of submitting jobs.')
    parser.add_argument('--force', action='store_true',
                        help='Resubmit even if the per-seed JSON already exists.')
    parser.add_argument('--scenarios', nargs='*', default=None)
    parser.add_argument('--name-contains', nargs='*', default=None,
                        help='Filter scenario IDs to those containing all of '
                             'these substrings (case-insensitive). E.g. '
                             '--name-contains diverse')
    parser.add_argument('--n-seeds', type=int, default=None,
                        help='Override the default number of seeds (NUTS=4, '
                             'rejection=15).')
    parser.add_argument('--seeds', nargs='*', type=int, default=None,
                        help='Explicit seed list; overrides --n-seeds.')
    parser.add_argument('--walltime', default=None,
                        help='SLURM walltime override (default: NUTS=23:59:59, '
                             'rejection=02:00:00).')
    args = parser.parse_args()

    if args.seeds is not None:
        seeds = args.seeds
    elif args.n_seeds is not None:
        seeds = list(range(args.n_seeds))
    else:
        seeds = None

    if args.combine:
        combine_seeds(mode=args.mode, scenarios=args.scenarios,
                      name_contains=args.name_contains, seeds=seeds)
    else:
        submit_all(mode=args.mode, dry_run=args.dry_run,
                   scenarios=args.scenarios,
                   name_contains=args.name_contains,
                   force=args.force, seeds=seeds,
                   walltime=args.walltime)


if __name__ == '__main__':
    main()
