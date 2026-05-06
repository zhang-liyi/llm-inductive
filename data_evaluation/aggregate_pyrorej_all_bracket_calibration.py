"""Aggregate the (re-run) pyro-rej-all-bracket calibration eval JSONs and
replace the existing `Pyro (Dist)` row in
`results/calibration_tables_all.tex` (NOT `calibration_tables.tex`).

Mirrors aggregate_gemini_direct_calibration.py: 3 seeds × {OE, BT-nonG,
BT-guided, MMLU, TruthfulQA, HellaSwag-h1+h2, ARC-C, Winogrande}, mean ± SE.
Idempotent — strips any existing Pyro (Dist) row block (NOT the
"Pyro (Dist) + TS" rows) before inserting the new one.

Usage:
    python aggregate_pyrorej_all_bracket_calibration.py            # print + insert
    python aggregate_pyrorej_all_bracket_calibration.py --print    # print only
"""
import argparse
import os
import re

# Reuse the helpers from the gemini-direct aggregator to stay consistent.
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aggregate_gemini_direct_calibration import (
    _f, _load, oe_metrics, cls_metrics, bt_guided_summary, hellaswag_combined,
    aggregate as _aggregate_template,
)

ROOT = './data_evaluation'
RES = f'{ROOT}/results'
TEX = f'{RES}/calibration_tables_all.tex'

SEEDS = (1, 2, 3)
TAG_FMT = 'pyrorej_all_s{seed}_bracket'
DISPLAY = 'Pyro (Dist)'
TS_DISPLAY = 'Pyro (Dist) + TS'


def _metrics_for_seed_with_prefix(seed, prefix=''):
    """If prefix='', read pyrorej_all_s{seed}_bracket_*.json (raw model evals).
    If prefix='ts_', read TS-applied versions ts_pyrorej_all_s{seed}_bracket_*.json."""
    tag = TAG_FMT.format(seed=seed)
    fname = lambda subdir, base: f'{RES}/{subdir}/{prefix}{base}.json'
    out = {}
    out['OE_MAE'], out['OE_CE'] = oe_metrics(_load(
        fname('openestimate', f'{tag}_openestimate')))
    out['BT-nonG_acc'], out['BT-nonG_ce'], out['BT-nonG_ece'] = cls_metrics(_load(
        fname('bayesian_teaching', f'{tag}_bt_base_tf')))
    out['BT-guided_acc'], out['BT-guided_ce'], out['BT-guided_ece'] = bt_guided_summary(_load(
        fname('bayesian_teaching', f'{tag}_bayesian_teaching_base_guided')))
    for ds in ('mmlu', 'truthfulqa', 'arc_challenge', 'winogrande'):
        a, c, e = cls_metrics(_load(fname('text_cls', f'{tag}_{ds}')))
        out[f'{ds}_acc'], out[f'{ds}_ce'], out[f'{ds}_ece'] = a, c, e
    h1 = _load(fname('text_cls', f'{tag}_hellaswag_h1'))
    h2 = _load(fname('text_cls', f'{tag}_hellaswag_h2'))
    a, c, e = hellaswag_combined(h1, h2)
    out['hellaswag_acc'], out['hellaswag_ce'], out['hellaswag_ece'] = a, c, e
    return out


def metrics_for_seed(seed):
    return _metrics_for_seed_with_prefix(seed, prefix='')


def metrics_for_seed_ts(seed):
    return _metrics_for_seed_with_prefix(seed, prefix='ts_')


def _aggregate_from(per_seed):
    from math import sqrt
    from statistics import mean, pstdev
    keys = sorted({k for m in per_seed for k in m})
    agg = {}
    for k in keys:
        vals = [m.get(k) for m in per_seed]
        if any(v is None for v in vals):
            agg[k] = (None, None)
        else:
            mu = mean(vals)
            se = (pstdev(vals) / sqrt(len(vals))) if len(vals) > 1 else 0.0
            agg[k] = (mu, se)
    return agg


def aggregate():
    return _aggregate_from([metrics_for_seed(s) for s in SEEDS])


def aggregate_ts():
    return _aggregate_from([metrics_for_seed_ts(s) for s in SEEDS])


def fmt_acc_row(agg):
    cells = [
        _f(*agg['OE_MAE'], digits=1),
        _f(*agg['BT-nonG_acc'],   digits=1, percent=True),
        _f(*agg['BT-guided_acc'], digits=1, percent=True),
        _f(*agg['mmlu_acc'],      digits=1, percent=True),
        _f(*agg['truthfulqa_acc'],digits=1, percent=True),
        _f(*agg['hellaswag_acc'], digits=1, percent=True),
        _f(*agg['arc_challenge_acc'], digits=1, percent=True),
        _f(*agg['winogrande_acc'],digits=1, percent=True),
    ]
    return f'{DISPLAY:<15}' + ' & ' + ' & '.join(cells) + r' \\'


def fmt_nll_row(agg):
    cells = [
        _f(*agg['OE_CE'], digits=3),
        _f(*agg['BT-nonG_ce'],    digits=3),
        _f(*agg['BT-guided_ce'],  digits=3),
        _f(*agg['mmlu_ce'],       digits=3),
        _f(*agg['truthfulqa_ce'], digits=3),
        _f(*agg['hellaswag_ce'],  digits=3),
        _f(*agg['arc_challenge_ce'], digits=3),
        _f(*agg['winogrande_ce'], digits=3),
    ]
    return f'{DISPLAY:<15}' + ' & ' + ' & '.join(cells) + r' \\'


def fmt_ece_row(agg):
    cells = [
        _f(*agg['BT-nonG_ece'],   digits=3),
        _f(*agg['BT-guided_ece'], digits=3),
        _f(*agg['mmlu_ece'],      digits=3),
        _f(*agg['truthfulqa_ece'],digits=3),
        _f(*agg['hellaswag_ece'], digits=3),
        _f(*agg['arc_challenge_ece'], digits=3),
        _f(*agg['winogrande_ece'],digits=3),
    ]
    return f'{DISPLAY:<15}' + ' & ' + ' & '.join(cells) + r' \\'


def fmt_ts_nll_row(agg):
    """NLL +TS row: OE column is `---`, then 7 classification CEs."""
    cells = [
        '---',
        _f(*agg['BT-nonG_ce'],    digits=3),
        _f(*agg['BT-guided_ce'],  digits=3),
        _f(*agg['mmlu_ce'],       digits=3),
        _f(*agg['truthfulqa_ce'], digits=3),
        _f(*agg['hellaswag_ce'],  digits=3),
        _f(*agg['arc_challenge_ce'], digits=3),
        _f(*agg['winogrande_ce'], digits=3),
    ]
    return f'{TS_DISPLAY:<20s}' + ' & ' + ' & '.join(cells) + r' \\'


def fmt_ts_ece_row(agg):
    """ECE +TS row: 7 classification ECEs (no OE)."""
    cells = [
        _f(*agg['BT-nonG_ece'],   digits=3),
        _f(*agg['BT-guided_ece'], digits=3),
        _f(*agg['mmlu_ece'],      digits=3),
        _f(*agg['truthfulqa_ece'],digits=3),
        _f(*agg['hellaswag_ece'], digits=3),
        _f(*agg['arc_challenge_ece'], digits=3),
        _f(*agg['winogrande_ece'],digits=3),
    ]
    return f'{TS_DISPLAY:<20s}' + ' & ' + ' & '.join(cells) + r' \\'


# ── Replace existing Pyro (Dist) rows in calibration_tables_all.tex ─────────
def _replace_rows(text, pattern, rows):
    """Replace the first len(rows) matches of `pattern` in `text` with `rows`,
    in document order. Pre-compute offsets, then patch in reverse so earlier
    offsets stay valid."""
    matches = list(pattern.finditer(text))
    if len(matches) < len(rows):
        return text, len(matches)
    targets = matches[:len(rows)]
    for m, row in zip(reversed(targets), reversed(rows)):
        text = text[:m.start()] + row + '\n' + text[m.end():]
    return text, len(targets)


def replace_in_tex(tex_path, acc_row, nll_row, ece_row,
                   ts_nll_row=None, ts_ece_row=None):
    """Replace 3 base `Pyro (Dist)` rows. If ts_nll_row / ts_ece_row are
    given, also replace the 2 `Pyro (Dist) + TS` rows in the NLL and ECE
    subtables."""
    with open(tex_path) as f:
        text = f.read()
    base_pat = re.compile(
        r'^Pyro \(Dist\)(?! \+ TS).*?\\\\\n', re.MULTILINE | re.DOTALL)
    text, n_base = _replace_rows(text, base_pat, [acc_row, nll_row, ece_row])
    if n_base < 3:
        print(f'WARN: replaced {n_base} of 3 Pyro (Dist) base rows')

    n_ts = 0
    if ts_nll_row is not None and ts_ece_row is not None:
        ts_pat = re.compile(
            r'^Pyro \(Dist\) \+ TS.*?\\\\\n', re.MULTILINE | re.DOTALL)
        text, n_ts = _replace_rows(text, ts_pat, [ts_nll_row, ts_ece_row])
        if n_ts < 2:
            print(f'WARN: replaced {n_ts} of 2 Pyro (Dist) + TS rows')

    with open(tex_path, 'w') as f:
        f.write(text)
    return n_base, n_ts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--print', action='store_true')
    ap.add_argument('--no_ts', action='store_true',
                    help='Skip the +TS rows (use only when TS jobs not yet done)')
    args = ap.parse_args()

    agg = aggregate()
    acc_row = fmt_acc_row(agg)
    nll_row = fmt_nll_row(agg)
    ece_row = fmt_ece_row(agg)

    print('--- Accuracy/MAE row ---')
    print(acc_row)
    print('\n--- NLL/CE row ---')
    print(nll_row)
    print('\n--- ECE row ---')
    print(ece_row)

    ts_nll_row = ts_ece_row = None
    if not args.no_ts:
        agg_ts = aggregate_ts()
        # TS rows don't use OE_CE / OE_MAE (OE column is `---`). Only flag
        # missing keys we actually need.
        REQUIRED = [
            'BT-nonG_ce', 'BT-guided_ce', 'mmlu_ce', 'truthfulqa_ce',
            'hellaswag_ce', 'arc_challenge_ce', 'winogrande_ce',
            'BT-nonG_ece', 'BT-guided_ece', 'mmlu_ece', 'truthfulqa_ece',
            'hellaswag_ece', 'arc_challenge_ece', 'winogrande_ece',
        ]
        ts_missing = [k for k in REQUIRED if agg_ts.get(k, (None, None))[0] is None]
        if ts_missing:
            print(f'\nWARN: TS metrics missing for keys: {ts_missing}')
            print('Skipping +TS rows; pass --no_ts to suppress this warning.')
            ts_nll_row = ts_ece_row = None
        else:
            ts_nll_row = fmt_ts_nll_row(agg_ts)
            ts_ece_row = fmt_ts_ece_row(agg_ts)
            print('\n--- NLL/CE +TS row ---')
            print(ts_nll_row)
            print('\n--- ECE +TS row ---')
            print(ts_ece_row)

    missing = [k for k, v in agg.items() if v[0] is None]
    if missing:
        print('\nWARNING: missing base metrics across all seeds:', missing)
        print('Aborting tex update; rerun once jobs finish.')
        return

    if not args.print:
        n_base, n_ts = replace_in_tex(TEX, acc_row, nll_row, ece_row,
                                      ts_nll_row, ts_ece_row)
        print(f'\nReplaced {n_base} Pyro (Dist) base rows + {n_ts} +TS rows in {TEX}')


if __name__ == '__main__':
    main()
