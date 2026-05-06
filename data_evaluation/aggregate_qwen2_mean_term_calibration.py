"""Aggregate Qwen2 mean-only + dist+term calibration eval JSONs and insert
``Qwen2 (Mean)`` and ``Qwen2 (Dist+Term)`` rows into
``calibration_tables_all.tex`` immediately after the existing
``Qwen2 (Dist)`` rows. Idempotent.
"""
import argparse
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


def aggregate(tag_fmt):
    per = [_metrics_for_tag(tag_fmt.format(s=s)) for s in SEEDS]
    keys = sorted({k for m in per for k in m})
    agg = {}
    for k in keys:
        vs = [m.get(k) for m in per]
        if any(v is None for v in vs):
            agg[k] = (None, None)
        else:
            mu = mean(vs)
            se = (pstdev(vs) / sqrt(len(vs))) if len(vs) > 1 else 0.0
            agg[k] = (mu, se)
    return agg


def fmt_acc(agg, label):
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
    return f'{label:<18s} & ' + ' & '.join(cells) + r' \\'


def fmt_nll(agg, label):
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
    return f'{label:<18s} & ' + ' & '.join(cells) + r' \\'


def fmt_ece(agg, label):
    cells = [
        _f(*agg['BT-nonG_ece'],   digits=3),
        _f(*agg['BT-guided_ece'], digits=3),
        _f(*agg['mmlu_ece'],      digits=3),
        _f(*agg['truthfulqa_ece'],digits=3),
        _f(*agg['hellaswag_ece'], digits=3),
        _f(*agg['arc_challenge_ece'], digits=3),
        _f(*agg['winogrande_ece'],digits=3),
    ]
    return f'{label:<18s} & ' + ' & '.join(cells) + r' \\'


LABELS = ('Qwen2 (Mean)', 'Qwen2 (Dist+Term)')


def _strip(text):
    for l in LABELS:
        pat = re.compile(r'^' + re.escape(l) + r'.*?\\\\\n',
                         re.MULTILINE | re.DOTALL)
        text = pat.sub('', text)
    return text


def insert(rows_per_subtable):
    with open(TEX) as f:
        text = f.read()
    text = _strip(text)
    # Anchor: the existing Qwen2 (Dist) row in each subtable. Insert after.
    pat = re.compile(r'^Qwen2 \(Dist\).*?\\\\\n', re.MULTILINE | re.DOTALL)
    matches = list(pat.finditer(text))
    if len(matches) < 3:
        print(f'WARN: found {len(matches)} Qwen2 (Dist) anchors; expected ≥3. Skipping insert.')
        return
    targets = matches[:3]
    for m, row in zip(reversed(targets), reversed(rows_per_subtable)):
        text = text[:m.end()] + row + '\n' + text[m.end():]
    with open(TEX, 'w') as f:
        f.write(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--print', action='store_true')
    args = ap.parse_args()

    mean_agg = aggregate('qwen2_pyrorej_all_mean_s{s}_bracket')
    term_agg = aggregate('qwen2_pyrorej_all_term_s{s}_bracket')

    rows_acc = (fmt_acc(mean_agg, 'Qwen2 (Mean)') + '\n' +
                fmt_acc(term_agg, 'Qwen2 (Dist+Term)'))
    rows_nll = (fmt_nll(mean_agg, 'Qwen2 (Mean)') + '\n' +
                fmt_nll(term_agg, 'Qwen2 (Dist+Term)'))
    rows_ece = (fmt_ece(mean_agg, 'Qwen2 (Mean)') + '\n' +
                fmt_ece(term_agg, 'Qwen2 (Dist+Term)'))

    print('--- Qwen2 (Mean) ---')
    print(fmt_acc(mean_agg, 'Qwen2 (Mean)'))
    print(fmt_nll(mean_agg, 'Qwen2 (Mean)'))
    print(fmt_ece(mean_agg, 'Qwen2 (Mean)'))
    print()
    print('--- Qwen2 (Dist+Term) ---')
    print(fmt_acc(term_agg, 'Qwen2 (Dist+Term)'))
    print(fmt_nll(term_agg, 'Qwen2 (Dist+Term)'))
    print(fmt_ece(term_agg, 'Qwen2 (Dist+Term)'))

    REQUIRED_M = [k for k in mean_agg if k.endswith(('_acc','_ce','_ece','_MAE'))]
    miss_m = [k for k in REQUIRED_M if mean_agg.get(k, (None, None))[0] is None]
    miss_t = [k for k in REQUIRED_M if term_agg.get(k, (None, None))[0] is None]
    if miss_m: print(f'\nWARN missing mean: {miss_m}')
    if miss_t: print(f'WARN missing term: {miss_t}')

    if not args.print and not (miss_m or miss_t):
        insert([rows_acc, rows_nll, rows_ece])
        print(f'\nInserted Qwen2 (Mean) + Qwen2 (Dist+Term) rows into {TEX}')


if __name__ == '__main__':
    main()
