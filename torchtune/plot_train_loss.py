"""
Plot training inner loss curves for all methods.

Usage:
    python plot_train_loss.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = './torchtune/'
PP = BASE + 'llama-probabilistic-normal-5t10neg5-stepeval0_False_'
FP = BASE + 'llama-forward-normal-5t10neg5-stepeval0_False_'

def load(path):
    try:
        return np.loadtxt(path)
    except:
        print(f"Missing: {path.split('/')[-1]}")
        return None

labels = {
    'prob_dist':     'Prob + Dist',
    'prob_mean_r8':  'Prob + Mean-only r8',
    'prob_mean_r64': 'Prob + Mean-only r64',
    'fwd_r8':        'Forward r8',
    'fwd_r64':       'Forward r64',
}
colors = {
    'prob_dist':     'royalblue',
    'prob_mean_r8':  'darkorange',
    'prob_mean_r64': 'green',
    'fwd_r8':        'crimson',
    'fwd_r64':       'purple',
}

# Training inner loss
# - prob_mean_r8: uses mean_only_r8 explicit run (has innerloss)
train_inner = {
    'prob_dist':     load(PP + 'distribution-seed1_innerloss.csv'),
    'prob_mean_r8':  load(PP + 'mean_only_r8-seed1_innerloss.csv'),
    'prob_mean_r64': load(PP + 'mean_only_r64-seed1_innerloss.csv'),
    'fwd_r8':        load(FP + 'mean_only_r8-seed1_innerloss.csv'),    # updated r8 run
    'fwd_r64':       load(FP + 'mean_only_r64-seed1_innerloss.csv'),
}

fig, ax = plt.subplots(figsize=(8, 5))

for key in labels:
    d = train_inner[key]
    if d is not None:
        ax.plot(d, color=colors[key], label=labels[key], linewidth=1.8)

ax.set_title('Training Inner Loss', fontsize=13)
ax.set_xlabel('Step')
ax.set_ylabel('Loss')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()

out = BASE + 'train_loss.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Saved {out}")
