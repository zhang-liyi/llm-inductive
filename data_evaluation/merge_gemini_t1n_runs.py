"""Merge multiple T=1 sample-and-mean Gemini runs into a single n-sample
aggregate file.

For each (group, prompt) cell, takes:
  results/gemini_argmax_bars/gemini3pro_<group>_<mode>_t1_n5.json        (run 1)
  results/gemini_argmax_bars/gemini3pro_<group>_<mode>_t1_n5_run2.json   (run 2)
  ...
and writes:
  results/gemini_argmax_bars/gemini3pro_<group>_<mode>_t1_n10.json

Per-item: concatenates `answers` lists (10 entries), recomputes
`mean_answer` = mean of not-None entries, recomputes `n_parsed`. The
top-level `metrics.mean_abs_error_mean` is the MAE of those recomputed
per-item means vs gt_mean.

Usage (default — merge run1 + run2 → n10):
    python merge_gemini_t1n_runs.py

Usage (custom — pick which runs to merge, name the output):
    python merge_gemini_t1n_runs.py --runs '' run2 --out_n 10

Items where every sample is None across all runs are excluded from MAE.
"""
import argparse
import json
import os

import numpy as np

ROOT = '.'
RES_DIR = f'{ROOT}/data_evaluation/results/gemini_argmax_bars'

CELLS = [
    ('sanity_25sports',   'old'),
    ('sanity_25sports',   'bracket'),
    ('rej_sports_val',    'old'),
    ('rej_sports_val',    'bracket'),
    ('rej_healthcare_val','old'),
    ('rej_healthcare_val','bracket'),
]


def _path(group, mode, run_id):
    suf = f'_{run_id}' if run_id else ''
    return f'{RES_DIR}/gemini3pro_{group}_{mode}_t1_n5{suf}.json'


def merge_cell(group, mode, run_ids, out_n):
    paths = [_path(group, mode, r) for r in run_ids]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        return None, f'missing inputs: {missing}'
    docs = [json.load(open(p)) for p in paths]
    base = docs[0]
    n_items = base['n_items']
    # All inputs must align item-for-item.
    for d in docs[1:]:
        if d['n_items'] != n_items:
            return None, f'n_items mismatch ({d["n_items"]} vs {n_items})'

    merged_items = []
    for i in range(n_items):
        all_answers = []
        gt = base['per_item'][i]['gt_mean']
        for d in docs:
            all_answers.extend(d['per_item'][i]['answers'])
            # double-check gt_mean alignment (val data is the same, so should match)
            if abs(d['per_item'][i]['gt_mean'] - gt) > 1e-6:
                return None, f'gt_mean mismatch at item {i}'
        parsed = [a for a in all_answers if a is not None]
        mean_ans = float(np.mean(parsed)) if parsed else None
        merged_items.append({
            'answers':     all_answers,
            'mean_answer': mean_ans,
            'n_parsed':    len(parsed),
            'gt_mean':     gt,
        })

    means = np.array([it['mean_answer'] for it in merged_items
                      if it['mean_answer'] is not None], dtype=float)
    gts = np.array([it['gt_mean'] for it in merged_items
                    if it['mean_answer'] is not None], dtype=float)
    mae = float(np.mean(np.abs(means - gts))) if len(means) else None

    out_path = f'{RES_DIR}/gemini3pro_{group}_{mode}_t1_n{out_n}.json'
    out_doc = {
        'run_label':   f'gemini3pro_{group}_{mode}_t1_n{out_n}',
        'val_data':    base['val_data'],
        'n_items':     n_items,
        'prompt_mode': mode,
        'temperature': base.get('temperature', 1.0),
        'n_samples':   out_n,
        'gemini_model':base.get('gemini_model'),
        'merged_from': paths,
        'metrics': {
            'mean_abs_error_mean':  mae,
            'n_items_with_parse':   int(len(means)),
            'n_total_items':        n_items,
            'avg_samples_per_item': float(np.mean([it['n_parsed'] for it in merged_items])),
        },
        'per_item': merged_items,
    }
    with open(out_path, 'w') as f:
        json.dump(out_doc, f, indent=2)
    return out_doc['metrics'], out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='+', default=['', 'run2'],
                    help="Run-id suffixes to merge. '' = the unsuffixed first run. "
                         "Default: '' run2 → 5+5 = 10 samples.")
    ap.add_argument('--out_n', type=int, default=None,
                    help='Total samples per item in the merged file (default: 5*len(runs)).')
    args = ap.parse_args()

    out_n = args.out_n if args.out_n is not None else 5 * len(args.runs)

    print(f'Merging runs {args.runs} -> n={out_n} per item')
    print()
    for group, mode in CELLS:
        metrics, info = merge_cell(group, mode, args.runs, out_n)
        if metrics is None:
            print(f'  [skip] {group} {mode}: {info}')
            continue
        print(f'  {group:<20s} {mode:<8s} '
              f'MAE={metrics["mean_abs_error_mean"]:6.3f}  '
              f'parsed_items={metrics["n_items_with_parse"]}/{metrics["n_total_items"]}  '
              f'avg_samples={metrics["avg_samples_per_item"]:.2f}  '
              f'-> {info.split("/")[-1]}')


if __name__ == '__main__':
    main()
