"""Compare combined pyro posterior samples to pooled human responses.

For each e2_implicit scenario and each query we compute:
  - Human mean / std (pool all entries across all respondents, all samples).
  - Pyro mean / std (from the combined per-seed posterior samples).

Output:
  - benchmarking/analysis/per_query_metrics.csv
  - benchmarking/analysis/scatter_pyro_vs_human.pdf

The scatter plot is square, axes 0-100; each dot is one (scenario, query).
Transparent error bars show the human and pyro standard deviations.
"""

import argparse
import csv
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BENCH_DIR = './benchmarking'
RESULTS_DIR = f'{BENCH_DIR}/inference_results'
ANALYSIS_DIR = f'{BENCH_DIR}/analysis'

SPORT_ORDER = ('tug_of_war', 'canoe', 'biathalon')
SPORT_LABEL = {'tug_of_war': 'Tug of War',
               'canoe':      'Canoe',
               'biathalon':  'Biathlon'}

TYPE_ORDER = ('constant', 'temporal', 'prediction')
TYPE_LABEL = {'constant':   'Constant',
              'temporal':   'Temporal',
              'prediction': 'Prediction'}


def sport_of(scenario_id):
    if scenario_id.startswith('tug-of-war'):
        return 'tug_of_war'
    if scenario_id.startswith('canoe-race'):
        return 'canoe'
    if scenario_id.startswith('biathalon'):
        return 'biathalon'
    return None


def query_type_of(query_key):
    idx = int(query_key.replace('query', ''))
    if idx <= 3:
        return 'constant'
    if idx <= 6:
        return 'temporal'
    return 'prediction'


def human_query_stats(responses, query_key):
    """Pool every number any respondent gave for this query and return mean/std."""
    vals = []
    for r in responses:
        if query_key in r:
            vals.extend(r[query_key])
    vals = np.asarray(vals, dtype=float)
    if vals.size == 0:
        return float('nan'), float('nan'), 0
    return float(vals.mean()), float(vals.std()), int(vals.size)


def pyro_query_stats(samples):
    arr = np.asarray(samples, dtype=float)
    if arr.size == 0:
        return float('nan'), float('nan'), 0
    # Some programs encode "who wins by how much" queries as a probability in
    # [0, 1]; rescale to the 0-100 scale used by humans so every query lives
    # on a common axis.
    if np.nanmax(arr) <= 1.0 and np.nanmin(arr) >= 0.0:
        arr = arr * 100.0
    return float(arr.mean()), float(arr.std()), int(arr.size)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=None,
                        help='If set, use only the per-seed result-{sid}-s{seed}.json '
                             'instead of the combined posterior.')
    args = parser.parse_args()

    os.makedirs(ANALYSIS_DIR, exist_ok=True)
    with open(f'{BENCH_DIR}/msa_cogsci_human_data.json') as f:
        human = json.load(f)
    e2 = human['e2_implicit']

    suffix = '' if args.seed is None else f'_seed{args.seed}'
    source_label = 'Pyro' if args.seed is None else f'Pyro (seed {args.seed})'

    rows = []
    for scenario_id, responses in e2.items():
        if args.seed is None:
            src_path = f'{RESULTS_DIR}/result-{scenario_id}-combined.json'
        else:
            src_path = f'{RESULTS_DIR}/result-{scenario_id}-s{args.seed}.json'
        if not os.path.isfile(src_path) or os.path.getsize(src_path) == 0:
            print(f'Missing {src_path}; skipping {scenario_id}')
            continue
        with open(src_path) as f:
            src = json.load(f)
        queries = src['queries'] if args.seed is None else src

        # Iterate in the human-data ordering (query1..query8).
        query_keys = sorted(
            set(k for r in responses for k in r.keys()),
            key=lambda q: int(q.replace('query', '')),
        )
        for q in query_keys:
            h_mean, h_std, h_n = human_query_stats(responses, q)
            pyro_key = q  # scripts use query1..queryN too
            if pyro_key not in queries:
                print(f'{scenario_id}: pyro missing {pyro_key}')
                p_mean, p_std, p_n = float('nan'), float('nan'), 0
            else:
                p_mean, p_std, p_n = pyro_query_stats(queries[pyro_key]['samples'])
            rows.append({
                'scenario_id': scenario_id,
                'query': q,
                'sport': sport_of(scenario_id),
                'query_type': query_type_of(q),
                'human_mean': h_mean,
                'human_std': h_std,
                'human_n': h_n,
                'pyro_mean': p_mean,
                'pyro_std': p_std,
                'pyro_n': p_n,
            })

    # Write CSV.
    csv_path = f'{ANALYSIS_DIR}/per_query_metrics{suffix}.csv'
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f'Wrote {csv_path} ({len(rows)} rows).')

    # Scatter plot.
    pts = [r for r in rows
           if np.isfinite(r['human_mean']) and np.isfinite(r['pyro_mean'])]
    # Axis convention: pyro on x, human on y.
    xs = np.array([r['pyro_mean'] for r in pts])
    ys = np.array([r['human_mean'] for r in pts])
    xerr = np.array([r['pyro_std'] for r in pts])
    yerr = np.array([r['human_std'] for r in pts])

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.errorbar(xs, ys, xerr=xerr, yerr=yerr,
                fmt='none', ecolor='#4a148c', alpha=0.2,
                elinewidth=1, capsize=0, zorder=1)
    ax.scatter(xs, ys, s=55, c='#4a148c', edgecolor='none',
               alpha=0.9, zorder=2)
    ax.plot([0, 100], [0, 100], linestyle='--', color='gray',
            linewidth=1, zorder=0)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel(source_label, fontsize=15)
    ax.set_ylabel('Human', fontsize=15)
    ax.tick_params(axis='both', labelsize=12)

    # Correlation stats: R² as title, r / MAE / RMSE in corner.
    if len(pts) >= 2:
        r = float(np.corrcoef(xs, ys)[0, 1])
        r2 = r * r
        mae = float(np.mean(np.abs(xs - ys)))
        rmse = float(np.sqrt(np.mean((xs - ys) ** 2)))
        ax.set_title(f'$R^2$ = {r2:.3f}', fontsize=16)
        ax.text(0.03, 0.97, f'r = {r:.3f}\nMAE = {mae:.2f}\nRMSE = {rmse:.2f}',
                transform=ax.transAxes, ha='left', va='top',
                fontsize=11, bbox=dict(facecolor='white', edgecolor='none',
                                       alpha=0.7))
    fig.tight_layout()
    out_pdf = f'{ANALYSIS_DIR}/scatter_pyro_vs_human{suffix}.pdf'
    fig.savefig(out_pdf)
    print(f'Wrote {out_pdf}.')

    fig2, axes = plt.subplots(3, 3, figsize=(14, 14), sharex=True, sharey=True)
    for i, qtype in enumerate(TYPE_ORDER):
        for j, sport in enumerate(SPORT_ORDER):
            ax = axes[i, j]
            cell = [r for r in pts
                    if sport_of(r['scenario_id']) == sport
                    and query_type_of(r['query']) == qtype]
            if cell:
                cx = np.array([r['pyro_mean']  for r in cell])
                cy = np.array([r['human_mean'] for r in cell])
                ce_x = np.array([r['pyro_std']  for r in cell])
                ce_y = np.array([r['human_std'] for r in cell])
                ax.errorbar(cx, cy, xerr=ce_x, yerr=ce_y, fmt='none',
                            ecolor='#4a148c', alpha=0.2,
                            elinewidth=1, capsize=0, zorder=1)
                ax.scatter(cx, cy, s=55, c='#4a148c',
                           edgecolor='none', alpha=0.9, zorder=2)
                if len(cell) >= 2:
                    rval = float(np.corrcoef(cx, cy)[0, 1])
                    r2_v = rval * rval
                    mae_v = float(np.mean(np.abs(cx - cy)))
                    rmse_v = float(np.sqrt(np.mean((cx - cy) ** 2)))
                    ax.set_title(f'$R^2$ = {r2_v:.3f}', fontsize=20)
                    ax.text(0.03, 0.97,
                            f'n={len(cell)}\nr={rval:.2f}\nMAE={mae_v:.1f}\nRMSE={rmse_v:.1f}',
                            transform=ax.transAxes, ha='left', va='top',
                            fontsize=11,
                            bbox=dict(facecolor='white', edgecolor='none',
                                      alpha=0.7))
                else:
                    ax.set_title(f'$R^2$ = n/a', fontsize=20)
                    ax.text(0.03, 0.97, f'n={len(cell)}',
                            transform=ax.transAxes, ha='left', va='top',
                            fontsize=11)
            else:
                ax.set_title(f'$R^2$ = n/a', fontsize=20)
                ax.text(0.5, 0.5, 'no data', transform=ax.transAxes,
                        ha='center', va='center', color='gray')

            ax.plot([0, 100], [0, 100], linestyle='--', color='gray',
                    linewidth=1, zorder=0)
            ax.set_xlim(0, 100)
            ax.set_ylim(0, 100)
            ax.set_aspect('equal', adjustable='box')
            ax.tick_params(axis='both', labelsize=12)

            if j == 0:
                ax.set_ylabel('Human', fontsize=20)
            if i == len(TYPE_ORDER) - 1:
                ax.set_xlabel(source_label, fontsize=20)

    fig2.tight_layout(rect=[0.04, 0, 1, 0.96])
    for j, sport in enumerate(SPORT_ORDER):
        bbox = axes[0, j].get_position()
        fig2.text((bbox.x0 + bbox.x1) / 2, 0.965, SPORT_LABEL[sport],
                  ha='center', va='bottom', fontsize=20, fontweight='bold')
    for i, qtype in enumerate(TYPE_ORDER):
        bbox = axes[i, 0].get_position()
        fig2.text(0.015, (bbox.y0 + bbox.y1) / 2, TYPE_LABEL[qtype],
                  ha='left', va='center', fontsize=20, fontweight='bold',
                  rotation=90)
    out_grid = f'{ANALYSIS_DIR}/scatter_pyro_vs_human_grid{suffix}.pdf'
    fig2.savefig(out_grid)
    print(f'Wrote {out_grid}.')


if __name__ == '__main__':
    main()
