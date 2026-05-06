"""Aggregate the gemini-direct calibration eval JSONs (3 seeds × {OE, BT-nonG,
BT-guided, MMLU, TruthfulQA, HellaSwag-h1, HellaSwag-h2, ARC-C, Winogrande})
into mean ± SE rows, and insert a "Gemini-Direct" row into
`results/calibration_tables_all.tex` (NOT `calibration_tables.tex`).

Inserted directly after the Pyro (Dist) row in each of the 3 subtables
(accuracy / NLL / ECE). Idempotent — re-running deletes any prior
"Gemini-Direct" row before inserting the freshly-computed one.

Usage:
    python aggregate_gemini_direct_calibration.py            # print + insert
    python aggregate_gemini_direct_calibration.py --print    # print only
"""
import argparse
import json
import os
import re
from math import sqrt
from statistics import mean, pstdev

ROOT = './data_evaluation'
RES = f'{ROOT}/results'
TEX = f'{RES}/calibration_tables_all.tex'

SEEDS = (1, 2, 3)
TAG_FMT = 'gemini_direct_all_s{seed}_bracket'
DISPLAY = 'Gemini-Direct'


# ── per-eval JSON loaders ───────────────────────────────────────────────────
def _load(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def oe_metrics(d):
    """Return (mae_mean, ce_mean) from openestimate JSON."""
    if d is None:
        return None, None
    s = d['summary']
    return s['mae']['mean'], s['ce_mean']['mean']


def cls_metrics(d):
    """Return (acc, ce_mean, ece) from a classification (BT or text_cls) JSON."""
    if d is None:
        return None, None, None
    s = d['summary']['overall']
    return s.get('accuracy'), s.get('ce_mean'), s.get('ece')


def bt_guided_summary(d):
    """Same as cls_metrics but tolerant to BT-guided file shape variations."""
    if d is None:
        return None, None, None
    s = d.get('summary', {})
    overall = s.get('overall', s)
    return (overall.get('accuracy'),
            overall.get('ce_mean'),
            overall.get('ece'))


def hellaswag_combined(h1, h2):
    """Combine HellaSwag h1 + h2 by re-aggregating raw per-item lists.
    Falls back to weighted average of summaries if per-item is missing."""
    if h1 is None or h2 is None:
        return None, None, None
    # Prefer per_item arrays when present.
    pi = (h1.get('per_item') or []) + (h2.get('per_item') or [])
    if pi and all(('correct' in it or 'is_correct' in it) for it in pi):
        n = len(pi)
        acc = mean(float(it.get('correct', it.get('is_correct'))) for it in pi)
        ce_per = [it['ce'] for it in pi if 'ce' in it]
        ce = mean(ce_per) if ce_per else None
        # ECE not trivial to recompute without bins; weighted avg as fallback.
        ece1 = h1['summary']['overall'].get('ece')
        ece2 = h2['summary']['overall'].get('ece')
        n1 = h1['summary']['overall'].get('n_total', 5021)
        n2 = h2['summary']['overall'].get('n_total', 5021)
        ece = ((ece1 * n1 + ece2 * n2) / (n1 + n2)
               if ece1 is not None and ece2 is not None else None)
        return acc, ce, ece
    # Fallback: weighted average of overall metrics.
    o1, o2 = h1['summary']['overall'], h2['summary']['overall']
    n1 = o1.get('n_total', 5021); n2 = o2.get('n_total', 5021)
    acc = (o1['accuracy'] * n1 + o2['accuracy'] * n2) / (n1 + n2)
    ce  = ((o1.get('ce_mean', 0) * n1 + o2.get('ce_mean', 0) * n2) / (n1 + n2)
           if o1.get('ce_mean') is not None and o2.get('ce_mean') is not None
           else None)
    ece = ((o1.get('ece', 0) * n1 + o2.get('ece', 0) * n2) / (n1 + n2)
           if o1.get('ece') is not None and o2.get('ece') is not None
           else None)
    return acc, ce, ece


# ── per-seed loader → metrics ──────────────────────────────────────────────
def metrics_for_seed(seed):
    tag = TAG_FMT.format(seed=seed)
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


def aggregate():
    """For each metric key, return (mean, SE) across seeds (None if any
    seed missing the metric)."""
    per_seed = [metrics_for_seed(s) for s in SEEDS]
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


# ── LaTeX row formatting ───────────────────────────────────────────────────
def _f(mu, se, digits=3, percent=False, with_se=True):
    if mu is None:
        return '---'
    if percent:
        body = rf'{mu * 100:.{digits}f}\%'
        se_body = rf'\pm {se * 100:.{digits}f}\%' if (se is not None and with_se) else ''
    else:
        body = rf'{mu:.{digits}f}'
        se_body = rf'\pm {se:.{digits}f}' if (se is not None and with_se) else ''
    if not se_body:
        return f'${body}$'
    return f'${body}$\n                  \\scriptsize{{${se_body}$}}'


def fmt_acc_row(agg):
    """Accuracy/MAE subtable: OE-MAE | BT-nonG | BT-guided | MMLU | TruthfulQA |
    HellaSwag | ARC-C | Winogrande."""
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
    """NLL subtable: OE-CE | BT-nonG-CE | BT-guided-CE | MMLU-CE | TruthfulQA-CE |
    HellaSwag-CE | ARC-C-CE | Winogrande-CE."""
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
    """ECE subtable: BT-nonG | BT-guided | MMLU | TruthfulQA | HellaSwag | ARC-C |
    Winogrande (no OE column)."""
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


# ── .tex insertion ─────────────────────────────────────────────────────────
PYRO_HEADER_RE = re.compile(r'^Pyro \(Dist\)', re.MULTILINE)


def _strip_existing_gd_row(text):
    """Remove any prior Gemini-Direct row block (single or multi-line)."""
    pattern = re.compile(
        r'\n' + re.escape(DISPLAY) + r'.*?\\\\\n', re.DOTALL)
    return pattern.sub('\n', text)


def _find_pyro_row_end(text, start):
    """From `start`, walk forward to the end of the Pyro (Dist) row (end-of-line
    after `\\\\`). Returns the index of the newline immediately following."""
    # Pyro (Dist) row spans multiple lines and ends with a `\\` terminator
    # followed by newline.
    i = text.index(r'\\', start)
    j = text.index('\n', i) + 1
    return j


def insert_into_tex(tex_path, acc_row, nll_row, ece_row):
    with open(tex_path) as f:
        text = f.read()
    text = _strip_existing_gd_row(text)

    # Find the 3 Pyro (Dist) row starts; use header position to identify subtable.
    # Subtable ordering in the .tex matches the row order of acc / nll / ece.
    # The NLL subtable also has a "Pyro (Dist) + TS" row — we want the FIRST
    # Pyro (Dist) (not the +TS variant) within each subtable.
    ms = list(PYRO_HEADER_RE.finditer(text))
    # Filter out "+ TS" variants by checking whether next non-space chars after
    # the row start include "+ TS".
    base_rows = [m for m in ms
                 if not text[m.start():m.start()+30].lstrip().startswith('Pyro (Dist) + TS')]
    if len(base_rows) < 3:
        raise RuntimeError(f'Expected >=3 Pyro (Dist) rows; found {len(base_rows)}')
    # Use first 3 (acc, nll, ece subtables).
    acc_m, nll_m, ece_m = base_rows[:3]

    rows_in_order = [acc_row, nll_row, ece_row]
    matches_in_order = [acc_m, nll_m, ece_m]
    # Insert from BOTTOM to TOP so earlier offsets remain valid.
    for m, row in zip(reversed(matches_in_order), reversed(rows_in_order)):
        end = _find_pyro_row_end(text, m.start())
        text = text[:end] + row + '\n' + text[end:]

    with open(tex_path, 'w') as f:
        f.write(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--print', action='store_true',
                    help='Print rows only; do not modify the .tex.')
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
    print()

    # Sanity check: if any major metric is missing across seeds, warn.
    missing = [k for k, v in agg.items() if v[0] is None]
    if missing:
        print('WARNING: missing metrics across all seeds:', missing)

    if not args.print:
        insert_into_tex(TEX, acc_row, nll_row, ece_row)
        print(f'Inserted Gemini-Direct rows into {TEX}')


if __name__ == '__main__':
    main()
