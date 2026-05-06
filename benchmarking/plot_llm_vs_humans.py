"""Scatter LLM answer (mean of the 101-way softmax) vs pooled human answer
for each (scenario, query) pair in e2_implicit.

Mirrors the format of compare_to_humans.py but swaps the pyro-program posterior
for the LLM's teacher-forced distribution (as produced by
``llm_inference_per_query.py``).

Inputs
------
  --llm_results  Path to the JSON produced by llm_inference_per_query.py.
  --out_dir      Output directory (default: benchmarking/analysis).

Outputs
-------
  per_query_metrics_{label}.csv
  scatter_llm_vs_human_{label}.pdf            (single panel, all pairs)
  scatter_llm_vs_human_{label}_grid.pdf       (3x3: sport x query type)

Grid layout:
  Columns (left -> right): Tug of War, Canoe, Biathlon
  Rows    (top  -> bottom): Constant (queries 1-3, individual rank),
                            Temporal (queries 4-6, per-match effort/accuracy),
                            Prediction (queries 7-8, future match outcome).
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
HUMAN_DATA = f'{BENCH_DIR}/msa_cogsci_human_data.json'


def human_query_stats(responses, query_key):
    """Pool all numbers any respondent gave for this query; return mean/std/n."""
    vals = []
    for r in responses:
        if query_key in r:
            vals.extend(r[query_key])
    vals = np.asarray(vals, dtype=float)
    if vals.size == 0:
        return float('nan'), float('nan'), 0
    return float(vals.mean()), float(vals.std()), int(vals.size)


def llm_std(dist):
    """Std of a discrete distribution over 0..100."""
    p = np.asarray(dist, dtype=float)
    xs = np.arange(101)
    mean = float((xs * p).sum())
    var = float(((xs - mean) ** 2 * p).sum())
    return float(np.sqrt(max(var, 0.0)))


# ── sport / query-type classification ────────────────────────────────────────

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
    """queries 1-3 -> constant (rank), 4-6 -> temporal (per-match), 7-8 -> prediction."""
    idx = int(query_key.replace('query', ''))
    if idx <= 3:
        return 'constant'
    if idx <= 6:
        return 'temporal'
    return 'prediction'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--llm_results', required=True,
                        help="JSON produced by llm_inference_per_query.py.")
    parser.add_argument('--out_dir', default=f'{BENCH_DIR}/analysis')
    parser.add_argument('--label', default=None,
                        help="Short label for the plot title / file names "
                             "(default: derived from the LLM results filename).")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    label = args.label or os.path.splitext(os.path.basename(args.llm_results))[0]

    with open(args.llm_results) as f:
        llm_data = json.load(f)
    llm_results = llm_data['results']

    with open(HUMAN_DATA) as f:
        human = json.load(f)
    e2 = human['e2_implicit']

    rows = []
    for scenario_id, responses in e2.items():
        llm_per_query = llm_results.get(scenario_id, {})
        if not llm_per_query:
            print(f'No LLM results for {scenario_id}; skipping')
            continue
        query_keys = sorted(
            set(k for r in responses for k in r.keys()),
            key=lambda q: int(q.replace('query', '')),
        )
        for q in query_keys:
            h_mean, h_std, h_n = human_query_stats(responses, q)
            if q not in llm_per_query:
                print(f'{scenario_id}: LLM missing {q}')
                continue
            llm_entry = llm_per_query[q]
            rows.append({
                'scenario_id': scenario_id,
                'query':       q,
                'sport':       sport_of(scenario_id),
                'query_type':  query_type_of(q),
                'human_mean':  h_mean,
                'human_std':   h_std,
                'human_n':     h_n,
                'llm_mean':    llm_entry['llm_mean'],
                'llm_std':     llm_std(llm_entry['llm_dist']),
                'llm_mode':    llm_entry.get('llm_mode', -1),
            })

    if not rows:
        print('No matched rows; nothing to plot.')
        return

    csv_path = os.path.join(args.out_dir, f'per_query_metrics_{label}.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f'Wrote {csv_path} ({len(rows)} rows).')

    # ── scatter plot ─────────────────────────────────────────────────────────
    pts = [r for r in rows if np.isfinite(r['human_mean']) and np.isfinite(r['llm_mean'])]
    # Axis convention: LLM on x, human on y.
    xs = np.array([r['llm_mean']   for r in pts])
    ys = np.array([r['human_mean'] for r in pts])
    xerr = np.array([r['llm_std']   for r in pts])
    yerr = np.array([r['human_std'] for r in pts])

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.errorbar(xs, ys, xerr=xerr, yerr=yerr, fmt='none',
                ecolor='#4a148c', alpha=0.2, elinewidth=1, capsize=0, zorder=1)
    ax.scatter(xs, ys, s=55, c='#4a148c', edgecolor='none', alpha=0.9, zorder=2)
    ax.plot([0, 100], [0, 100], linestyle='--', color='gray', linewidth=1, zorder=0)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('LLM', fontsize=15)
    ax.set_ylabel('Human', fontsize=15)
    ax.tick_params(axis='both', labelsize=12)

    if len(pts) >= 2:
        r = float(np.corrcoef(xs, ys)[0, 1])
        r2 = r * r
        mae = float(np.mean(np.abs(xs - ys)))
        rmse = float(np.sqrt(np.mean((xs - ys) ** 2)))
        ax.set_title(f'$R^2$ = {r2:.3f}', fontsize=16)
        ax.text(0.03, 0.97, f'r = {r:.3f}\nMAE = {mae:.2f}\nRMSE = {rmse:.2f}',
                transform=ax.transAxes, ha='left', va='top', fontsize=11,
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
    else:
        ax.set_title('', fontsize=16)

    fig.tight_layout()
    out_pdf = os.path.join(args.out_dir, f'scatter_llm_vs_human_{label}.pdf')
    fig.savefig(out_pdf)
    print(f'Wrote {out_pdf}.')

    # ── 3x3 facet grid: sport (cols) x query type (rows) ─────────────────────
    fig2, axes = plt.subplots(3, 3, figsize=(14, 14), sharex=True, sharey=True)
    for i, qtype in enumerate(TYPE_ORDER):
        for j, sport in enumerate(SPORT_ORDER):
            ax = axes[i, j]
            cell = [r for r in pts
                    if r['sport'] == sport and r['query_type'] == qtype]
            if cell:
                cx = np.array([r['llm_mean']   for r in cell])
                cy = np.array([r['human_mean'] for r in cell])
                ce_x = np.array([r['llm_std']   for r in cell])
                ce_y = np.array([r['human_std'] for r in cell])
                ax.errorbar(cx, cy, xerr=ce_x, yerr=ce_y, fmt='none',
                            ecolor='#4a148c', alpha=0.2,
                            elinewidth=1, capsize=0, zorder=1)
                ax.scatter(cx, cy, s=55, c='#4a148c',
                           edgecolor='none', alpha=0.9, zorder=2)
                if len(cell) >= 2:
                    rval = float(np.corrcoef(cx, cy)[0, 1])
                    r2_v = rval * rval
                    ax.set_title(f'$R^2$ = {r2_v:.3f}', fontsize=20)
                else:
                    ax.set_title(f'$R^2$ = n/a', fontsize=20)
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
                ax.set_xlabel('LLM', fontsize=20)

    # Column headers (sport) above the top-row R² titles, and row headers
    # (query type) on the left edge — both larger than the per-axes titles.
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
    out_grid = os.path.join(args.out_dir, f'scatter_llm_vs_human_{label}_grid.pdf')
    fig2.savefig(out_grid)
    print(f'Wrote {out_grid}.')


if __name__ == '__main__':
    main()
