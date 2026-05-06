"""Run rejection-sampling inference for one sanity-check scenario and compute
the 4 new queries. Called by the SLURM submitter.

Loads the matching `pg-gemini-REJ-{...}.py` program (importance-weighted
rejection sampling on the soft-likelihood model), not the NUTS variant.

Usage:
    python run_sanity_25sports_one.py <scenario_id>
"""
import argparse
import importlib.util
import json
import os
import sys
from collections import Counter

import pyro
import torch

ROOT = './posterior_sampling_pytorch'
QUERIES_JSON = f'{ROOT}/sanity_25sports/sanity_25sports_queries.json'
RESULTS_DIR = f'{ROOT}/sanity_25sports/inference_results'

os.makedirs(RESULTS_DIR, exist_ok=True)


def is_who_wins(samples):
    return max(samples) <= 1.0 + 1e-9


def samples_to_bins(samples):
    """101-bin histogram + mean, matching convert_rej_to_finetune.py."""
    scaled = [s * 100 for s in samples] if is_who_wins(samples) else list(samples)
    int_samples = [max(0, min(100, round(s))) for s in scaled]
    counts = Counter(int_samples)
    total = len(int_samples)
    bins = [counts.get(i, 0) / total for i in range(101)]
    mean_val = sum(scaled) / len(scaled)
    return bins, mean_val


def rej_program_id(scenario_id):
    """Map e.g. 'gemini-P-1-C-1-...-2-1' -> 'gemini-REJ-P-1-C-1-...-2-1'."""
    if scenario_id.startswith('gemini-REJ-'):
        return scenario_id
    if scenario_id.startswith('gemini-'):
        return 'gemini-REJ-' + scenario_id[len('gemini-'):]
    raise ValueError(f'Unexpected scenario_id format: {scenario_id}')


def load_program_module(scenario_id):
    rej_id = rej_program_id(scenario_id)
    path = f'{ROOT}/programs/pg-{rej_id}.py'
    spec = importlib.util.spec_from_file_location(f'pg_{rej_id.replace("-", "_")}', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    p = argparse.ArgumentParser()
    p.add_argument('scenario_id')
    p.add_argument('--num_samples', type=int, default=1000)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    out_path = f'{RESULTS_DIR}/{args.scenario_id}.json'
    if os.path.isfile(out_path):
        print(f'Skip (exists): {out_path}')
        return

    # Find this scenario's spec.
    spec_list = json.load(open(QUERIES_JSON))
    spec = next((s for s in spec_list if s['scenario_id'] == args.scenario_id), None)
    if spec is None:
        raise SystemExit(f'No spec found for {args.scenario_id}')

    print(f'Loading REJ program for {args.scenario_id} ...')
    mod = load_program_module(args.scenario_id)

    pyro.set_rng_seed(args.seed)
    torch.manual_seed(args.seed)
    import random as _r
    _r.seed(args.seed)
    print(f'Running rejection sampling (num_samples={args.num_samples}) ...')
    samples = mod.run_inference(num_samples=args.num_samples)

    results = []
    for nq in spec['new_queries']:
        helper = getattr(mod, nq['helper'])
        # Pass the helper's positional args as in the original __main__ block:
        # all helpers take (samples, *args).
        out = helper(samples, *nq['args'])
        raw = out.get('raw') or out.get('samples') or []
        if not raw:
            # Some helpers (e.g. rank) return scalar dict; degrade gracefully.
            raise RuntimeError(f'Helper {nq["helper"]} returned no raw samples.')
        bins, mean_val = samples_to_bins(raw)
        results.append({
            'type': nq['type'],
            'helper': nq['helper'],
            'args': nq['args'],
            'nl': nq['nl'],
            'mean': mean_val,
            'bins': bins,
        })
        print(f'  {nq["type"]:9s} {nq["helper"]} {nq["args"]}  mean={mean_val:.2f}')

    with open(out_path, 'w') as f:
        json.dump({
            'scenario_id': args.scenario_id,
            'queries': results,
        }, f)
    print(f'Wrote {out_path}')


if __name__ == '__main__':
    main()
