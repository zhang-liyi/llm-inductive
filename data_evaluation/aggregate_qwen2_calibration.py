"""Aggregate Qwen2-7B calibration eval JSONs (base + 3 ft seeds) and
insert/update rows in `results/calibration_tables_all.tex`:

  Qwen2 (Base)   — single-shot (no SE) from `qwen2_pretrained_*` JSONs
  Qwen2 (Dist)   — 3-seed mean ± SE from `qwen2_pyrorej_all_s{1,2,3}_bracket_*` JSONs

Inserted right after the existing `Forward (Mean)` row in each of the 3
subtables (accuracy / NLL / ECE). Idempotent: any prior `Qwen2 (Base)` /
`Qwen2 (Dist)` rows are stripped before insertion.
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
    per = [_metrics_for_tag(f'qwen2_pyrorej_all_s{s}_bracket') for s in SEEDS]
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


def aggregate_base():
    """Base is single-shot; SE is None (formatter then prints just the value)."""
    m = _metrics_for_tag('qwen2_pretrained')
    return {k: (v, None) for k, v in m.items()}


# ── row formatters ─────────────────────────────────────────────────────────
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


# ── tex insertion ─────────────────────────────────────────────────────────
QWEN_LABELS = ('Qwen2 (Base)', 'Qwen2 (Dist)')


def _strip_existing(text):
    for label in QWEN_LABELS:
        pat = re.compile(r'^' + re.escape(label) + r'.*?\\\\\n',
                         re.MULTILINE | re.DOTALL)
        text = pat.sub('', text)
    return text


def _insert_after_forward_mean(text, rows):
    """rows = list of strings to insert (in document-order). Inserts each
    one immediately after the FIRST occurrence of `Forward (Mean)` row in
    each subtable. Patches in REVERSE document-order so earlier offsets
    stay valid."""
    pat = re.compile(r'^Forward \(Mean\).*?\\\\\n', re.MULTILINE | re.DOTALL)
    matches = list(pat.finditer(text))
    if len(matches) < len(rows):
        print(f'WARN: found {len(matches)} Forward (Mean) anchors; '
              f'expected ≥{len(rows)}')
        return text
    for m, row in zip(reversed(matches[:len(rows)]), reversed(rows)):
        text = text[:m.end()] + row + '\n' + text[m.end():]
    return text


def insert_into_tex(rows_per_subtable):
    with open(TEX) as f:
        text = f.read()
    text = _strip_existing(text)
    text = _insert_after_forward_mean(text, rows_per_subtable)
    with open(TEX, 'w') as f:
        f.write(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--print', action='store_true')
    args = ap.parse_args()

    base = aggregate_base()
    ft = aggregate_ft()

    # Print
    print('--- Qwen2 (Base) — single-shot ---')
    print(_fmt_acc(base, 'Qwen2 (Base)'))
    print(_fmt_nll(base, 'Qwen2 (Base)'))
    print(_fmt_ece(base, 'Qwen2 (Base)'))
    print()
    print('--- Qwen2 (Dist) — 3-seed mean ± SE ---')
    print(_fmt_acc(ft, 'Qwen2 (Dist)'))
    print(_fmt_nll(ft, 'Qwen2 (Dist)'))
    print(_fmt_ece(ft, 'Qwen2 (Dist)'))

    REQUIRED_FT = [k for k in ft if k.endswith('_acc') or k.endswith('_ce') or k.endswith('_ece')
                   or k.startswith('OE')]
    missing_ft = [k for k in REQUIRED_FT if ft[k][0] is None]
    if missing_ft:
        print(f'\nWARN: missing FT metrics for: {missing_ft}')

    REQUIRED_BASE = REQUIRED_FT
    missing_base = [k for k in REQUIRED_BASE if base[k][0] is None]
    if missing_base:
        print(f'WARN: missing Base metrics for: {missing_base}')

    if args.print:
        return

    # Insert order in each subtable: Qwen2 (Base) then Qwen2 (Dist)
    acc_rows = (_fmt_acc(base, 'Qwen2 (Base)') + '\n' +
                _fmt_acc(ft, 'Qwen2 (Dist)'))
    nll_rows = (_fmt_nll(base, 'Qwen2 (Base)') + '\n' +
                _fmt_nll(ft, 'Qwen2 (Dist)'))
    ece_rows = (_fmt_ece(base, 'Qwen2 (Base)') + '\n' +
                _fmt_ece(ft, 'Qwen2 (Dist)'))

    insert_into_tex([acc_rows, nll_rows, ece_rows])
    print(f'\nInserted Qwen2 rows into {TEX}')


if __name__ == '__main__':
    main()
