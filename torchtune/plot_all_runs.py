"""
Plot MAE and CE training curves (val) for all 6 runs:
  pyro mean-only, pyro dist, forward (mean-only),
  prob mean-only, prob dist, fusion (pyro_fusion dist).

All solid lines are evaluated on probabilistic_reasoning_val.json:
  - Prob runs:    _maes_val / _ces_*_val  (this IS prob-reasoning val)
  - Forward run:  _prob_maes_val / _prob_ces_*_val
  - Pyro/Fusion:  _webppl_maes_val / _webppl_ces_*_val

Pyro and Fusion also show dashed lines (same colour) for pyro MCMC val:
  - _maes_val / _ces_*_val

Four subplots:
  1. MAE mean    2. MAE dist    3. CE mean    4. CE dist
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

BASE = './torchtune'
P = 'llama-{}-normal-5t10neg5-stepeval0_False_{}_{}-seed1'

def pfx(dataset, loss, rank='r8'):
    return os.path.join(BASE, P.format(dataset, loss, rank))

def load(path):
    try:
        d = np.loadtxt(path)
        if len(d) == 501:
            d = d[:500]
        return d
    except Exception:
        return None

# ── metric suffixes ────────────────────────────────────────────────────────────
# (key, prob_reasoning_suffix, pyro_suffix, ylabel, title)
METRICS = [
    ('mae_mean', '_webppl_maes_val.csv',      '_maes_val.csv',      '_prob_maes_val.csv',      '_maes_val.csv',      'MAE (mean)',      'MAE'),
    ('mae_dist', '_webppl_maes_dist_val.csv', '_maes_dist_val.csv', '_prob_maes_dist_val.csv', '_maes_dist_val.csv', 'MAE (dist L1)',   'MAE dist'),
    ('ce_mean',  '_webppl_ces_mean_val.csv',  '_ces_mean_val.csv',  '_prob_ces_mean_val.csv',  '_ces_mean_val.csv',  'Cross-entropy',   'CE mean'),
    ('ce_dist',  '_webppl_ces_dist_val.csv',  '_ces_dist_val.csv',  '_prob_ces_dist_val.csv',  '_ces_dist_val.csv',  'Cross-entropy',   'CE dist'),
]
# columns: key, pyro/fusion solid suffix (webppl), pyro/fusion dashed suffix (pyro mcmc),
#          forward solid suffix (prob_*), prob solid suffix, ylabel, title

# ── run definitions ────────────────────────────────────────────────────────────
# kind: 'pyro_fusion' = solid webppl + dashed pyro-mcmc
#       'prob'        = solid prob-reasoning (_maes_val)
#       'forward'     = solid prob-reasoning (_prob_maes_val)
RUNS = [
    ('Pyro mean-only',    'steelblue',   'pyro_fusion', pfx('pyro',          'mean_only')),
    ('Pyro dist',         'royalblue',   'pyro_fusion', pfx('pyro',          'distribution')),
    ('Forward mean-only', 'crimson',     'forward',     pfx('forward',       'mean_only')),
    ('Prob mean-only',    'darkorange',  'prob',        pfx('probabilistic', 'mean_only')),
    ('Prob dist',         'goldenrod',   'prob',        pfx('probabilistic', 'distribution')),
    ('Fusion dist',       'forestgreen', 'pyro_fusion', pfx('pyro_fusion',   'distribution')),
]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for ax_i, (key, webppl_sfx, pyro_sfx, fwd_sfx, prob_sfx, ylabel, title) in enumerate(METRICS):
    ax = axes[ax_i]

    for label, color, kind, base_pfx in RUNS:
        if kind == 'pyro_fusion':
            # solid = probabilistic_reasoning (webppl)
            d_solid = load(base_pfx + webppl_sfx)
            # dashed = pyro MCMC val
            d_dash  = load(base_pfx + pyro_sfx)
            if d_solid is not None:
                ax.plot(np.arange(1, len(d_solid) + 1), d_solid,
                        color=color, linewidth=1.6, label=label)
            if d_dash is not None:
                ax.plot(np.arange(1, len(d_dash) + 1), d_dash,
                        color=color, linewidth=1.0, linestyle='--', alpha=0.55,
                        label=f'{label} (pyro-val)')
        elif kind == 'forward':
            d = load(base_pfx + fwd_sfx)
            if d is not None:
                ax.plot(np.arange(1, len(d) + 1), d,
                        color=color, linewidth=1.6, label=label)
        else:  # prob
            d = load(base_pfx + prob_sfx)
            if d is not None:
                ax.plot(np.arange(1, len(d) + 1), d,
                        color=color, linewidth=1.6, label=label)

    ax.set_title(title, fontsize=12)
    ax.set_xlabel('Epoch')
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

fig.suptitle(
    'Validation on probabilistic_reasoning_val — all 6 runs (seed 1, LoRA r8)\n'
    'Solid = prob-reasoning val · Dashed = pyro MCMC val (pyro/fusion only)',
    fontsize=12)
plt.tight_layout()

out = os.path.join(BASE, 'plot_all_runs.png')
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved: {out}')

# ── summary stats ──────────────────────────────────────────────────────────────
print('\n=== Summary — prob-reasoning val (solid lines) ===')
for key, webppl_sfx, pyro_sfx, fwd_sfx, prob_sfx, ylabel, title in METRICS:
    print(f'\n  {title}:')
    for label, color, kind, base_pfx in RUNS:
        sfx = webppl_sfx if kind == 'pyro_fusion' else (fwd_sfx if kind == 'forward' else prob_sfx)
        d = load(base_pfx + sfx)
        if d is not None:
            print(f'    {label:22s}  n={len(d):3d}  min={d.min():.4f} @ ep {d.argmin()+1:3d}  final={d[-1]:.4f}')
        else:
            print(f'    {label:22s}  MISSING')
