"""Scatter-plot rejection-sampling posterior means vs pooled human responses.

Mirrors compare_to_humans.py but reads the rejection-sampling JSONs
(`-rej-{mode}-combined.json`, falling back to the legacy `-rej-combined.json`
for hard mode) and emits distinct output names so the NUTS plots are not
overwritten.

Outputs (per --mode):
  benchmarking/analysis/per_query_metrics_rej_{mode}.csv
  benchmarking/analysis/scatter_rej_{mode}_vs_human.pdf
  benchmarking/analysis/scatter_rej_{mode}_vs_human_grid.pdf
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

SOURCE_LABEL = 'Pyro (Rejection Sampling)'


def sport_of(sid):
    if sid.startswith('tug-of-war'):    return 'tug_of_war'
    if sid.startswith('canoe-race'):    return 'canoe'
    if sid.startswith('biathalon'):     return 'biathalon'
    return None


def query_type_of(q):
    idx = int(q.replace('query', ''))
    if idx <= 3: return 'constant'
    if idx <= 6: return 'temporal'
    return 'prediction'


def rej_path(sid, mode):
    candidates = [f'result-{sid}-rej-{mode}-combined.json']
    if mode == 'hard':
        candidates.append(f'result-{sid}-rej-combined.json')
    for name in candidates:
        p = f'{RESULTS_DIR}/{name}'
        if os.path.isfile(p) and os.path.getsize(p) > 0:
            return p
    return None


def human_stats(responses, q):
    vals = []
    for r in responses:
        if q in r:
            vals.extend(r[q])
    arr = np.asarray(vals, dtype=float)
    if arr.size == 0:
        return float('nan'), float('nan'), 0
    return float(arr.mean()), float(arr.std()), int(arr.size)


def pyro_stats(samples):
    arr = np.asarray(samples, dtype=float)
    if arr.size == 0:
        return float('nan'), float('nan'), 0
    # Rescale who-wins probabilities ∈ [0,1] onto the 0-100 scale.
    if np.nanmax(arr) <= 1.0 and np.nanmin(arr) >= 0.0:
        arr = arr * 100.0
    return float(arr.mean()), float(arr.std()), int(arr.size)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('hard', 'soft'), default='hard',
                        help='Which rejection-sampling variant to plot.')
    args = parser.parse_args()
    mode = args.mode
    suffix = f'rej_{mode}'

    os.makedirs(ANALYSIS_DIR, exist_ok=True)
    with open(f'{BENCH_DIR}/msa_cogsci_human_data.json') as f:
        e2 = json.load(f)['e2_implicit']

    rows = []
    for sid, responses in e2.items():
        p = rej_path(sid, mode)
        if p is None:
            print(f'Missing rejection result for {sid}; skipping')
            continue
        with open(p) as f:
            src = json.load(f)
        queries = src.get('queries', src)
        qkeys = sorted(set(k for r in responses for k in r.keys()),
                       key=lambda q: int(q.replace('query', '')))
        for q in qkeys:
            h_mean, h_std, h_n = human_stats(responses, q)
            if q not in queries:
                print(f'{sid}: rejection missing {q}')
                p_mean, p_std, p_n = float('nan'), float('nan'), 0
            else:
                samples = queries[q].get('samples') or queries[q].get('raw') or []
                p_mean, p_std, p_n = pyro_stats(samples)
            rows.append({
                'scenario_id': sid,
                'query': q,
                'sport': sport_of(sid),
                'query_type': query_type_of(q),
                'human_mean': h_mean,
                'human_std':  h_std,
                'human_n':    h_n,
                'pyro_mean':  p_mean,
                'pyro_std':   p_std,
                'pyro_n':     p_n,
            })

    csv_path = f'{ANALYSIS_DIR}/per_query_metrics_{suffix}.csv'
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f'Wrote {csv_path} ({len(rows)} rows).')

    pts = [r for r in rows
           if np.isfinite(r['human_mean']) and np.isfinite(r['pyro_mean'])]
    xs = np.array([r['pyro_mean']  for r in pts])
    ys = np.array([r['human_mean'] for r in pts])
    xerr = np.array([r['pyro_std']  for r in pts])
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
    ax.set_xlabel(SOURCE_LABEL, fontsize=15)
    ax.set_ylabel('Human', fontsize=15)
    ax.tick_params(axis='both', labelsize=12)

    if len(pts) >= 2:
        r = float(np.corrcoef(xs, ys)[0, 1])
        r2 = r * r
        mae = float(np.mean(np.abs(xs - ys)))
        rmse = float(np.sqrt(np.mean((xs - ys) ** 2)))
        ax.set_title(f'$R^2$ = {r2:.3f}', fontsize=16)
        ax.text(0.03, 0.97, f'r = {r:.3f}\nMAE = {mae:.2f}\nRMSE = {rmse:.2f}',
                transform=ax.transAxes, ha='left', va='top',
                fontsize=11,
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
    fig.tight_layout()
    out_pdf = f'{ANALYSIS_DIR}/scatter_{suffix}_vs_human.pdf'
    fig.savefig(out_pdf)
    print(f'Wrote {out_pdf}.')
    plt.close(fig)

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
                    r2v  = rval * rval
                    mae_v = float(np.mean(np.abs(cx - cy)))
                    rmse_v = float(np.sqrt(np.mean((cx - cy) ** 2)))
                    ax.set_title(f'$R^2$ = {r2v:.3f}', fontsize=20)
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
                ax.set_xlabel(SOURCE_LABEL, fontsize=20)

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
    out_grid = f'{ANALYSIS_DIR}/scatter_{suffix}_vs_human_grid.pdf'
    fig2.savefig(out_grid)
    print(f'Wrote {out_grid}.')
    plt.close(fig2)


if __name__ == '__main__':
    main()
