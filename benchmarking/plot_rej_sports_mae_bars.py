"""Grouped bar chart on two validation sets (sports rejection-sampling heldout
+ healthcare rejection heldout), metric = MAE.

Series:
  Base (pretrained Llama-3-8B-Instruct)
  Sports-FT          (pyro-rej-sports, *with* diverse-data augmentation)
  Sports-FT-no-div   (pyro-rej-sports, no diverse-data augmentation)
  All-FT             (pyro-rej-all, with diverse)
  All-FT-no-div      (pyro-rej-all, no diverse)
  Forward-sampling   (lora_forward_sampling_dist_*_bracket)

Sports MAEs come from training-time val CSVs (no new eval needed):
  torchtune/llama-pyro_rej_sports_bracket-...-seed{1,2,3}_maes_val.csv          (sports-FT, w/ diverse)
  torchtune/llama-pyro_rej_sports_no_diverse_bracket-...-seed{1,2,3}_..._val.csv (sports-FT, no diverse)
  torchtune/llama-pyro_rej_all_bracket-...-seed{1,2,3}_..._val.csv              (all-FT, w/ diverse)
  torchtune/llama-pyro_rej_all_no_diverse_bracket-...-seed{1,2,3}_..._val.csv   (all-FT, no diverse)
  torchtune/llama-forward_sampling_bracket-...-seed{1,2,3}_..._val.csv          (forward-sampling)
  - pretrained:  first row (val MAE before any training)
  - fine-tuned:  min across rows (best epoch), averaged over seeds

Healthcare MAEs come from teacher-forced evals (prepend "<", read 0..100 token dist):
  data_evaluation/results/healthcare/pretrained_rej_healthcare_val_tf.json
  data_evaluation/results/healthcare/pyro_rej_sports_bracket_seed{1,2,3}_rej_healthcare_val_tf.json
  data_evaluation/results/healthcare/pyrorej_sports_no_diverse_s{1,2,3}_bracket_rej_healthcare_val_tf.json
  data_evaluation/results/healthcare/pyrorej_all_s{1,2,3}_bracket_rej_healthcare_val_tf.json
  data_evaluation/results/healthcare/pyrorej_all_no_diverse_s{1,2,3}_bracket_rej_healthcare_val_tf.json
  data_evaluation/results/healthcare/forward_sampling_s{1,2,3}_bracket_rej_healthcare_val_tf.json
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = '.'
TORCH = f'{ROOT}/torchtune'
RES = f'{ROOT}/data_evaluation/results/healthcare'
RES_SAN25 = f'{ROOT}/data_evaluation/results/sanity_25sports'
OUT = f'{ROOT}/benchmarking/analysis'

CSV_TPL = (f'{TORCH}/llama-{{tag}}-normal-5t10neg5-stepeval0_False_'
           '{kind}_r8-seed{seed}_maes_val.csv')

# Canonical "Sports-FT" / "All-Domain-FT" labels = with-diverse training.
# (No-diverse runs remain available as `*-no-div` aux series.)
SPORTS_TAGS = [
    ('Sports-FT',                 'pyro_rej_sports_bracket',           'distribution'),
    ('All-Domain-FT',             'pyro_rej_all_bracket',              'distribution'),
    ('Sports-FT-no-div',          'pyro_rej_sports_no_diverse_bracket','distribution'),
    ('All-Domain-FT-no-div',      'pyro_rej_all_no_diverse_bracket',   'distribution'),
    ('Forward-sampling',          'forward_sampling_bracket',          'distribution'),
]

# Healthcare eval paths — instruction-matched TF on the full 1258-item val:
#   Base: old-prompt TF (BRACKET_INSTRUCTION pushes base OOD, hurting it).
#   FT bracket models: bracket-prompt TF (matches the BRACKET_INSTRUCTION the
#                      models actually saw at training time).
#   No-diverse / forward-sampling: only old-prompt TF runs exist for now.
HC_PRETRAINED = f'{RES}/pretrained_rej_healthcare_val_tf.json'
HC_TPLS = {
    'Sports-FT':                f'{RES}/pyro_rej_sports_bracket_seed{{seed}}_rej_healthcare_val_brkprm_tf.json',
    'All-Domain-FT':            f'{RES}/pyrorej_all_s{{seed}}_bracket_rej_healthcare_val_brkprm_tf.json',
    'Sports-FT-no-div':         f'{RES}/pyrorej_sports_no_diverse_s{{seed}}_bracket_rej_healthcare_val_brkprm_tf.json',
    'All-Domain-FT-no-div':     f'{RES}/pyrorej_all_no_diverse_s{{seed}}_bracket_rej_healthcare_val_brkprm_tf.json',
    'Forward-sampling':         f'{RES}/forward_sampling_s{{seed}}_bracket_rej_healthcare_val_tf.json',
    'Llama-3 Gemini-Answer-FT': f'{RES}/gemini_direct_all_s{{seed}}_bracket_rej_healthcare_val_brkprm_tf.json',
}

LEGEND_NAMES = {
    'Sports-FT':     'Meta-MSA (Sports)',
    'All-Domain-FT': 'Meta-MSA (All Domain)',
}

COLORS = {
    'Llama-3-8B':            '#c6dbef',  # lighter light blue (untrained base)
    'Llama-3 Gemini-Answer-FT': '#6baed6',  # darker light blue (LLama trained on Gemini answers; reserved)
    'Gemini-3-pro':          '#bdbdbd',  # lighter grey (argmax @ T=0)
    'Gemini-3-pro (sampled)':'#7f7f7f',  # darker grey (T=1 sampled mean)
    'Sports-FT':             '#fd8d3c',  # bright-light orange
    'Sports-FT-no-div':      '#9ecae1',  # light blue
    'All-Domain-FT':         '#d62728',  # dark red
    'All-Domain-FT-no-div':  '#fcae91',  # very light red
    'Forward-sampling':      '#fc9272',  # light red
}


def _csv_first_and_best(tag, kind):
    """Return (first_row, best_row) means/stds across seeds 1-3. Returns None
    if any seed CSV is missing — series with in-flight retraining are simply
    dropped from the chart."""
    firsts, bests = [], []
    for seed in (1, 2, 3):
        p = CSV_TPL.format(tag=tag, kind=kind, seed=seed)
        if not os.path.exists(p):
            return None, None
        v = np.atleast_1d(np.loadtxt(p))
        if v.size < 2:
            # Training in flight: only the pretrain row exists so far.
            return None, None
        firsts.append(float(v[0]))
        bests.append(float(v.min()))
    return (float(np.mean(firsts)), float(np.std(firsts))), \
           (float(np.mean(bests)),  float(np.std(bests)))


# Dedicated TF-eval JSONs of FT models on the strict-sports val
# `pytorch_rej_sports_val.json` (1265 items), with bracket prompt — what the
# Llama-base + Gemini bars are also evaluated against. Apples-to-apples.
RES_SPORTS_VAL = f'{ROOT}/data_evaluation/results/sports_val'
SPORTS_VAL_TPLS = {
    'Sports-FT':                f'{RES_SPORTS_VAL}/pyro_rej_sports_bracket_seed{{seed}}_rej_sports_val_brkprm_tf.json',
    'All-Domain-FT':            f'{RES_SPORTS_VAL}/pyrorej_all_s{{seed}}_bracket_rej_sports_val_brkprm_tf.json',
    'Sports-FT-no-div':         f'{RES_SPORTS_VAL}/pyrorej_sports_no_diverse_s{{seed}}_bracket_rej_sports_val_brkprm_tf.json',
    'All-Domain-FT-no-div':     f'{RES_SPORTS_VAL}/pyrorej_all_no_diverse_s{{seed}}_bracket_rej_sports_val_brkprm_tf.json',
    # Gemini-direct (Llama-3 trained on Gemini integer answers, mean-only loss)
    'Llama-3 Gemini-Answer-FT': f'{RES_SPORTS_VAL}/gemini_direct_all_s{{seed}}_bracket_rej_sports_val_brkprm_tf.json',
}


def sports_mae():
    """Returns dict label -> (mean, std) for the (strict) Sports val
    `pytorch_rej_sports_val.json`. FT bars come from the dedicated TF-eval
    JSONs (apples-to-apples with the Llama-base + Gemini bars). Falls back
    to the training-time CSV (which is on the mixed eval val) for any
    series that doesn't have a dedicated JSON yet."""
    out = {}
    # Preferred: dedicated TF-eval JSONs on pytorch_rej_sports_val.json
    for label, tpl in SPORTS_VAL_TPLS.items():
        paths = [tpl.format(seed=s) for s in (1, 2, 3)]
        if not all(os.path.exists(p) for p in paths):
            continue
        vs = [_json_mae(p) for p in paths]
        out[label] = (float(np.mean(vs)), float(np.std(vs)))

    # Fallback: training-time CSV (on `pytorch_rej_eval_val.json`, the mixed
    # eval val) for any FT series not yet evaluated on strict sports val.
    base_first = None
    for label, tag, kind in SPORTS_TAGS:
        first, best = _csv_first_and_best(tag, kind)
        if first is None:
            continue
        if base_first is None:
            base_first = first
        if label not in out:
            out[label] = best
    if base_first is not None:
        out['Base'] = base_first
    return out


def _json_mae(path):
    with open(path) as f:
        return float(json.load(f)['metrics']['mean_abs_error'])


def healthcare_mae():
    """Returns dict label -> (mean, std) on the healthcare val set. Series
    without all 3 seeds present are silently skipped."""
    out = {'Base': (_json_mae(HC_PRETRAINED), 0.0)}
    for label, tpl in HC_TPLS.items():
        paths = [tpl.format(seed=s) for s in (1, 2, 3)]
        if not all(os.path.exists(p) for p in paths):
            continue
        vs = [_json_mae(p) for p in paths]
        out[label] = (float(np.mean(vs)), float(np.std(vs)))
    return out


# Sanity-25-sports val: 25 train scenarios x 4 strict-different queries, GT
# bins from rejection sampling on the matching pg-gemini-REJ-{...}.py programs.
SAN25_PRETRAINED = f'{RES_SAN25}/pretrained_sanity_25sports_val.json'
SAN25_TPLS = {
    'Sports-FT':                f'{RES_SAN25}/pyro_rej_sports_bracket_seed{{seed}}_sanity_25sports_val.json',
    'All-Domain-FT':            f'{RES_SAN25}/pyrorej_all_s{{seed}}_bracket_sanity_25sports_val.json',
    'Sports-FT-no-div':         f'{RES_SAN25}/pyrorej_sports_no_diverse_s{{seed}}_bracket_sanity_25sports_val.json',
    'All-Domain-FT-no-div':     f'{RES_SAN25}/pyrorej_all_no_diverse_s{{seed}}_bracket_sanity_25sports_val.json',
    'Forward-sampling':         f'{RES_SAN25}/forward_sampling_s{{seed}}_bracket_sanity_25sports_val.json',
    'Llama-3 Gemini-Answer-FT': f'{RES_SAN25}/gemini_direct_all_s{{seed}}_bracket_sanity_25sports_val.json',
}


def sanity_25sports_mae():
    """Returns dict label -> (mean, std) on the 25-train-scenarios x new
    queries val set. Series with any missing seed are silently skipped."""
    out = {}
    if os.path.exists(SAN25_PRETRAINED):
        out['Base'] = (_json_mae(SAN25_PRETRAINED), 0.0)
    for label, tpl in SAN25_TPLS.items():
        paths = [tpl.format(seed=s) for s in (1, 2, 3)]
        if not all(os.path.exists(p) for p in paths):
            continue
        vs = [_json_mae(p) for p in paths]
        out[label] = (float(np.mean(vs)), float(np.std(vs)))
    return out


# ── Closed-API + base-model helpers ───────────────────────────────────────
# OLD prompt only for Llama-3-8B base + Gemini (we found these models do
# better with the OLD single-query instruction). FT models still use the
# BRACKET-prompt eval JSONs, since that's what they were trained on — those
# are loaded by the existing sports_mae / healthcare_mae / sanity_25sports_mae
# helpers and are unchanged here.
#
# Llama-3-8B base bar = best of {mean, mode} on the OLD-prompt TF JSON:
#   mean = `metrics.mean_abs_error`  (pred_mean of the 0..100 dist)
#   mode = MAE of per-item `greedy` (argmax of next-token dist at "<")
# Gemini argmax       = OLD prompt, T=0
# Gemini sampled-mean = OLD prompt, T=1, mean of N parsed samples
GEMINI_RES = f'{ROOT}/data_evaluation/results/gemini_argmax_bars'

# Pretrained-base TF JSON for each group, OLD prompt.
LLAMA_TF_OLD_PATHS = {
    'Sports (train)':   f'{RES_SAN25}/pretrained_sanity_25sports_val_oldprm.json',
    'Sports (val)':     f'{ROOT}/data_evaluation/results/sports_val/pretrained_rej_sports_val_oldprm_tf.json',
    'Healthcare (val)': f'{RES}/pretrained_rej_healthcare_val_tf.json',
}

# Gemini argmax (T=0) JSONs, OLD prompt.
GEMINI_ARGMAX_OLD_PATHS = {
    'Sports (train)':   f'{GEMINI_RES}/gemini3pro_sanity_25sports_old.json',
    'Sports (val)':     f'{GEMINI_RES}/gemini3pro_rej_sports_val_old.json',
    'Healthcare (val)': f'{GEMINI_RES}/gemini3pro_rej_healthcare_val_old.json',
}


def _t1n_old_path(group_stem):
    """Prefer the largest merged-n OLD-prompt sampled-mean file available
    (e.g. n=20 over n=10 over n=5). Search descending."""
    for n in (200, 150, 100, 75, 50, 40, 30, 25, 20, 15, 10, 5):
        p = f'{GEMINI_RES}/gemini3pro_{group_stem}_old_t1_n{n}.json'
        if os.path.exists(p):
            return p
    # Fallback to whatever exists with the n5 name (legacy default).
    return f'{GEMINI_RES}/gemini3pro_{group_stem}_old_t1_n5.json'


GEMINI_SAMPLED_OLD_PATHS = {
    'Sports (train)':   _t1n_old_path('sanity_25sports'),
    'Sports (val)':     _t1n_old_path('rej_sports_val'),
    'Healthcare (val)': _t1n_old_path('rej_healthcare_val'),
}


def _llama_tf_metrics(tf_json_path):
    """Return (mean_mae, argmax_mae) for a pretrained TF JSON, or (None,None)."""
    if not os.path.exists(tf_json_path):
        return None, None
    with open(tf_json_path) as f:
        d = json.load(f)
    mean_mae = d.get('metrics', {}).get('mean_abs_error')
    pi = d.get('per_item') or []
    if pi and 'greedy' in pi[0]:
        g = np.array([it['greedy'] for it in pi], dtype=float)
        gt = np.array([it['gt_mean'] for it in pi], dtype=float)
        argmax_mae = float(np.mean(np.abs(g - gt)))
    else:
        argmax_mae = None
    return mean_mae, argmax_mae


def _gemini_argmax_mae(json_path):
    if not os.path.exists(json_path):
        return None
    with open(json_path) as f:
        d = json.load(f)
    return d.get('metrics', {}).get('mean_abs_error_argmax')


def _gemini_sample_mean_mae(json_path):
    if not os.path.exists(json_path):
        return None
    with open(json_path) as f:
        d = json.load(f)
    return d.get('metrics', {}).get('mean_abs_error_mean')


def llama_base_mae_per_group():
    """Best of {mean, mode} on the OLD-prompt TF JSON per group. Returns
    (out, label) where label encodes the winning estimator."""
    out, label = {}, {}
    for g, p_old in LLAMA_TF_OLD_PATHS.items():
        m_old, a_old = _llama_tf_metrics(p_old)
        cands = []
        for v, tag in [(m_old, 'mean+OLD'), (a_old, 'mode+OLD')]:
            if v is not None:
                cands.append((v, tag))
        if not cands:
            continue
        v, tag = min(cands, key=lambda x: x[0])
        out[g] = (v, 0.0)
        label[g] = tag
    return out, label


def gemini_argmax_mae_per_group():
    """Gemini argmax (T=0) MAE per group, OLD prompt only."""
    out = {}
    for g, p in GEMINI_ARGMAX_OLD_PATHS.items():
        v = _gemini_argmax_mae(p)
        if v is not None:
            out[g] = (v, 0.0)
    return out


def gemini_sampled_mae_per_group():
    """Gemini sampled-mean (T=1, N=5 or N=10) MAE per group, OLD prompt only.
    Returns (out, n_used) so the plot can label the bar with how many
    samples ended up in the average."""
    out, n_used = {}, {}
    for g, p in GEMINI_SAMPLED_OLD_PATHS.items():
        if not os.path.exists(p):
            continue
        with open(p) as f:
            d = json.load(f)
        v = d.get('metrics', {}).get('mean_abs_error_mean')
        if v is None:
            continue
        out[g] = (v, 0.0)
        n_used[g] = d.get('n_samples', 5)
    return out, n_used


ALL_SERIES = ['Sports-FT', 'All-Domain-FT']  # FT-only; Llama base + Gemini are added as extras
PRINT_SERIES = ['Base',
                'Sports-FT', 'Sports-FT-no-div',
                'All-Domain-FT', 'All-Domain-FT-no-div',
                'Llama-3 Gemini-Answer-FT']  # printed table only


def main():
    os.makedirs(OUT, exist_ok=True)
    sp = sports_mae()
    hc = healthcare_mae()
    san25 = sanity_25sports_mae()

    # Series union: the legend is the same as the original chart (5 series);
    # any series missing data in a particular group renders an empty slot.
    series = [s for s in ALL_SERIES if s in sp and s in hc]

    def _print(group_name, src):
        print(group_name + ':')
        for s in PRINT_SERIES:
            v = src.get(s)
            if v is None:
                print(f'  {s:<22s} (no data)')
            else:
                print(f'  {s:<22s} MAE = {v[0]:.3f} (±{v[1]:.3f})')

    _print('Sports (train, new queries) val', san25)
    _print('Sports val', sp)
    _print('Healthcare val', hc)

    # OLD-prompt baselines: Llama-3-8B base + Gemini bars.
    llama_base, llama_lab = llama_base_mae_per_group()
    gem_arg                = gemini_argmax_mae_per_group()
    gem_smp,    gem_smp_n  = gemini_sampled_mae_per_group()
    print('Llama-3-8B base (OLD prompt; best of mean/mode):')
    for g, v in llama_base.items():
        print(f'  {g:<22s} MAE = {v[0]:.3f}  ({llama_lab[g]})')
    print('Gemini-3-pro (OLD prompt, argmax T=0):')
    for g, v in gem_arg.items():
        print(f'  {g:<22s} MAE = {v[0]:.3f}')
    print('Gemini-3-pro (OLD prompt, sampled-mean T=1):')
    for g, v in gem_smp.items():
        print(f'  {g:<22s} MAE = {v[0]:.3f}  (n={gem_smp_n[g]})')

    groups = ['Sports (train)', 'Sports (val)', 'Healthcare (val)']
    group_sources = [san25, sp, hc]

    # Final order: Gemini-3-pro → Gemini-3-pro (sampled) → Llama-3-8B
    # → Llama-3-8B (gemini-direct) → FT bars. Each extra-series slot only
    # appears when its data is present.
    base_extras = ['Gemini-3-pro']
    if any(g in gem_smp for g in groups):
        base_extras.append('Gemini-3-pro (sampled)')
    base_extras.append('Llama-3-8B')
    # gemini-direct lives in the per-group source dicts (san25 / sp / hc),
    # not extra_series_sources, since it has 3 seeds aggregated by the
    # existing helpers — same shape as the FT bars.
    GD = 'Llama-3 Gemini-Answer-FT'
    if GD in san25 and GD in sp and GD in hc:
        base_extras.append(GD)
    series = base_extras + list(series)
    extra_series_sources = {
        'Llama-3-8B':              [llama_base.get(g) for g in groups],
        'Gemini-3-pro':            [gem_arg.get(g)    for g in groups],
        'Gemini-3-pro (sampled)':  [gem_smp.get(g)    for g in groups],
    }

    n_series = len(series)
    n_groups = len(groups)
    group_spacing = 0.55   # less empty space between groups
    x = np.arange(n_groups) * group_spacing
    w = (0.85 / n_series) * 0.55  # thinner bars to keep group spans inside group_spacing
    fig, ax = plt.subplots(figsize=(13, 6))

    all_vals = []
    for i, s in enumerate(series):
        if s in extra_series_sources:
            pairs = extra_series_sources[s]
            vals = [p[0] if p is not None else np.nan for p in pairs]
            errs = [p[1] if p is not None else np.nan for p in pairs]
        else:
            vals = [src.get(s, (np.nan, np.nan))[0] for src in group_sources]
            errs = [src.get(s, (np.nan, np.nan))[1] for src in group_sources]
        offset = (i - (n_series - 1) / 2) * w
        bars = ax.bar(x + offset, vals, w, yerr=errs, capsize=3,
                      color=COLORS[s], label=LEGEND_NAMES.get(s, s))
        for bar, v in zip(bars, vals):
            if np.isnan(v):
                continue
            ax.text(bar.get_x() + bar.get_width() / 2, v,
                    f'{v:.2f}', ha='center', va='bottom', fontsize=14)
        all_vals.extend(v for v in vals if not np.isnan(v))

    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=20)
    ax.set_ylabel('MAE', fontsize=22)
    ax.tick_params(axis='y', labelsize=18)
    ax.legend(fontsize=14, frameon=False, ncol=2)

    ymax = max(all_vals) * 1.22 if all_vals else 1.0
    ax.set_ylim(0, ymax)

    fig.tight_layout()
    out_pdf = f'{OUT}/bars_rej_sports_mae.pdf'
    fig.savefig(out_pdf)
    print(f'Wrote {out_pdf}')


if __name__ == '__main__':
    main()
