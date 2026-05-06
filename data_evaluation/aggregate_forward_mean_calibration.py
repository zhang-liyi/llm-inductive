"""Aggregate Llama-3 forward-sampling-mean calibration eval JSONs (3 seeds)
and replace the legacy single-shot `Forward (Mean)` row in
`results/calibration_tables_all.tex` with a 3-seed mean ± SE row.

Source JSONs (under data_evaluation/results/):
  openestimate/forward_sampling_mean_s{1,2,3}_bracket_openestimate.json
  bayesian_teaching/forward_sampling_mean_s{1,2,3}_bracket_bt_base_tf.json
  bayesian_teaching/forward_sampling_mean_s{1,2,3}_bracket_bayesian_teaching_base_guided.json
  text_cls/forward_sampling_mean_s{1,2,3}_bracket_{mmlu,truthfulqa,hellaswag_h1,hellaswag_h2,arc_challenge,winogrande}.json

Idempotent: if the table already has a 3-seed Forward (Mean) row (with SE),
it gets replaced again.
"""
import argparse
import json
import os
import re
import sys
from math import sqrt
from statistics import mean, pstdev

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aggregate_gemini_direct_calibration import (
    _f, _load, oe_metrics, cls_metrics, bt_guided_summary, hellaswag_combined,
)

ROOT = './data_evaluation'
RES = f'{ROOT}/results'
TEX = f'{RES}/calibration_tables_all.tex'

SEEDS = (1, 2, 3)
LABEL = 'Forward (Mean)'


def _metrics_for_tag(tag):
    out = {}
    out['OE_MAE'], out['OE_CE'] = oe_metrics(_load(
        f'{RES}/openestimate/{tag}_openestimate.json'))
    out['BT-nonG_acc'], out['BT-nonG_ce'], out['BT-nonG_ece'] = cls_metrics(_load(
        f'{RES}/bayesian_teaching/{tag}_bt_base_tf.json'))
    out['BT-guided_acc'], out['BT-guided_ce'], out['BT-guided_ece'] = bt_guided_summary(_load(
        f'{RES}/bayesian_teaching/{tag}_bayesian_teaching_base_guided.json'))
    for ds in ('mmlu', 'truthfulqa', 'arc_challenge', 'winogrande'):
        a, c, e = cls_metrics(_load(f'{RES}/text_cls/{tag}_{ds}.json'))
        out[f'{ds}_acc'], out[f'{ds}_ce'], out[f'{ds}_ece'] = a, c, e
    h1 = _load(f'{RES}/text_cls/{tag}_hellaswag_h1.json')
    h2 = _load(f'{RES}/text_cls/{tag}_hellaswag_h2.json')
    a, c, e = hellaswag_combined(h1, h2)
    out['hellaswag_acc'], out['hellaswag_ce'], out['hellaswag_ece'] = a, c, e
    return out


def aggregate_ft():
    per = [_metrics_for_tag(f'forward_sampling_mean_s{s}_bracket') for s in SEEDS]
    keys = sorted({k for m in per for k in m})
    agg = {}
    for k in keys:
        vals = [m.get(k) for m in per]
        if any(v is None for v in vals):
            agg[k] = (None, None)
        else:
            mu = mean(vals)
            se = (pstdev(vals) / sqrt(len(vals))) if len(vals) > 1 else 0.0
            agg[k] = (mu, se)
    return agg


def _fmt_acc(agg, label):
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
    return f'{label:<15s} & ' + ' & '.join(cells) + r' \\'


def _fmt_nll(agg, label):
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
    return f'{label:<15s} & ' + ' & '.join(cells) + r' \\'


def _fmt_ece(agg, label):
    cells = [
        _f(*agg['BT-nonG_ece'],   digits=3),
        _f(*agg['BT-guided_ece'], digits=3),
        _f(*agg['mmlu_ece'],      digits=3),
        _f(*agg['truthfulqa_ece'],digits=3),
        _f(*agg['hellaswag_ece'], digits=3),
        _f(*agg['arc_challenge_ece'], digits=3),
        _f(*agg['winogrande_ece'],digits=3),
    ]
    return f'{label:<15s} & ' + ' & '.join(cells) + r' \\'


def _replace_forward_mean(text, rows):
    """Replace each `Forward (Mean) ... \\` row with the corresponding rows
    entry, in document order (one per subtable: acc / NLL / ECE)."""
    pat = re.compile(r'^Forward \(Mean\).*?\\\\\n', re.MULTILINE | re.DOTALL)
    matches = list(pat.finditer(text))
    if len(matches) != len(rows):
        print(f'WARN: found {len(matches)} `Forward (Mean)` anchors; '
              f'expected {len(rows)}. Aborting in-place edit.')
        return text
    for m, row in zip(reversed(matches), reversed(rows)):
        text = text[:m.start()] + row + '\n' + text[m.end():]
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--print', action='store_true')
    args = ap.parse_args()

    ft = aggregate_ft()
    print(f'--- {LABEL} — 3-seed mean ± SE ---')
    print(_fmt_acc(ft, LABEL))
    print(_fmt_nll(ft, LABEL))
    print(_fmt_ece(ft, LABEL))

    missing = [k for k, (v, _) in ft.items() if v is None]
    if missing:
        print(f'\nWARN: missing metrics for: {missing}')

    if args.print:
        return

    rows = [_fmt_acc(ft, LABEL), _fmt_nll(ft, LABEL), _fmt_ece(ft, LABEL)]
    with open(TEX) as f:
        text = f.read()
    text = _replace_forward_mean(text, rows)
    with open(TEX, 'w') as f:
        f.write(text)
    print(f'\nReplaced Forward (Mean) row in {TEX}')


if __name__ == '__main__':
    main()
