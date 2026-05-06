"""Run Gemini-3-pro on each item of a single-query val set (the same
pytorch_rej_*_val.json / pytorch_sanity_25sports_val.json that the Llama TF
evals consume), parse one integer answer per item, and report MAE vs gt_mean.

Reuses run_gemini_async + parse_brackets + BRACKET_INSTRUCTION from
gemini_direct/gemini_direct_answer.py. Single-query, temperature=0 (argmax).

Two prompt modes (matching the existing Llama TF conventions):
  --prompt_mode old      → send the val item's stored 'input' verbatim (which
                          carries the OLD "Answer the query…return only an
                          integer" instruction). Parse the first integer in
                          Gemini's reply.
  --prompt_mode bracket  → swap the first paragraph for BRACKET_INSTRUCTION
                          (mirroring _swap_bracket_instruction in
                          evaluate_healthcare.py). Parse the first <N>.

Output JSON shape mirrors evaluate_healthcare.py's so plot_rej_sports_mae_bars
can read the existing fields:
  {
    "run_label":  "...",
    "val_data":   "...path/to/val.json",
    "n_items":    1258,
    "prompt_mode":"bracket"|"old",
    "metrics":    {"mean_abs_error_argmax": <float>},
    "per_item":   [{"answer": int|None, "gt_mean": float}, ...]
  }

We only fill mean_abs_error_argmax (no token distribution available from a
closed API). Items where Gemini returned no integer are dropped from the
mean. Per-item field is named 'answer' (not 'greedy') to make the API source
explicit.
"""
import argparse
import asyncio
import json
import os
import re
import sys

import numpy as np

_ROOT = '.'
sys.path.insert(0, os.path.join(_ROOT, 'gemini_direct'))
from gemini_direct_answer import (  # type: ignore
    BRACKET_INSTRUCTION,
    call_with_retries,
    parse_brackets,
)


_FIRST_INT_RE = re.compile(r"-?\d+")


def parse_first_int(text):
    """Return the first integer in text clamped to [0, 100], or None."""
    if text is None:
        return None
    m = _FIRST_INT_RE.search(text)
    if m is None:
        return None
    try:
        return max(0, min(100, int(m.group(0))))
    except ValueError:
        return None


def _swap_bracket(prompt):
    parts = prompt.split("\n\n", 1)
    return BRACKET_INSTRUCTION + ("\n\n" + parts[1] if len(parts) > 1 else "")


def _gt_mean(item):
    """Mean of the item's bins[0] (single-query items have one bin row of
    101 probabilities). Falls back to int(output) for one-hot bins."""
    bins = item.get('bins')
    if bins is None:
        return float(int(item['output']))
    if isinstance(bins[0], list):
        bins = bins[0]
    return float(sum(p * n for n, p in enumerate(bins)))


def _parse_answer(raw, prompt_mode):
    if prompt_mode == 'bracket':
        nums = parse_brackets(raw, n_expected=1)
        return nums[0] if nums else None
    return parse_first_int(raw)


async def _process_one(idx, item, prompt_mode, semaphore, temperature,
                       max_tokens, retries, n_samples):
    """Issue n_samples parallel API calls for one item; return all parsed
    answers (None for failures) and the per-sample raw text."""
    raw_input = item['input']
    if prompt_mode == 'bracket':
        prompt = _swap_bracket(raw_input)
    else:
        prompt = raw_input
    sample_results = await asyncio.gather(*[
        call_with_retries(prompt, semaphore, temperature, max_tokens, retries)
        for _ in range(n_samples)
    ])
    answers, errors, raws = [], [], []
    for raw, err in sample_results:
        answers.append(_parse_answer(raw, prompt_mode))
        errors.append(err)
        raws.append(raw)
    return idx, answers, errors, raws


async def main_async(args):
    with open(args.val_data) as f:
        data = json.load(f)
    if args.limit is not None:
        data = data[: args.limit]

    semaphore = asyncio.Semaphore(args.max_concurrent)
    tasks = [
        _process_one(i, item, args.prompt_mode, semaphore,
                     args.temperature, args.max_tokens, args.retries,
                     args.n_samples)
        for i, item in enumerate(data)
    ]

    per_item = [None] * len(data)
    n_done = 0
    for coro in asyncio.as_completed(tasks):
        idx, answers, errors, raws = await coro
        gt = _gt_mean(data[idx])
        parsed = [a for a in answers if a is not None]
        mean_ans = float(np.mean(parsed)) if parsed else None
        rec = {
            'answers': answers,
            'mean_answer': mean_ans,
            'n_parsed': len(parsed),
            'gt_mean': gt,
        }
        if any(e is not None for e in errors):
            rec['errors'] = errors
        if args.keep_raw:
            rec['raws'] = raws
        per_item[idx] = rec
        n_done += 1
        if n_done % 50 == 0 or n_done == len(data):
            n_ok = sum(1 for r in per_item if r and r['mean_answer'] is not None)
            print(f'  {n_done}/{len(data)} items done  ({n_ok} with >=1 parsed sample)')

    # Aggregate metrics: MAE of per-item mean vs gt_mean. Items with zero
    # parsed samples are excluded.
    means = np.array([r['mean_answer'] for r in per_item if r['mean_answer'] is not None],
                     dtype=float)
    gts = np.array([r['gt_mean'] for r in per_item if r['mean_answer'] is not None],
                   dtype=float)
    mae_mean = float(np.mean(np.abs(means - gts))) if len(means) else None

    metrics = {
        'mean_abs_error_mean':   mae_mean,
        'n_items_with_parse':    int(len(means)),
        'n_total_items':         int(len(per_item)),
        'avg_samples_per_item':  float(np.mean([r['n_parsed'] for r in per_item])),
    }
    # Back-compat: when n_samples == 1, also expose `mean_abs_error_argmax`
    # so existing readers (plot script's old field) keep working.
    if args.n_samples == 1:
        metrics['mean_abs_error_argmax'] = mae_mean
        metrics['n_parsed'] = int(len(means))
        metrics['n_total']  = int(len(per_item))

    out = {
        'run_label': args.run_label or os.path.splitext(os.path.basename(args.output))[0],
        'val_data': os.path.abspath(args.val_data),
        'n_items': len(per_item),
        'prompt_mode': args.prompt_mode,
        'temperature': args.temperature,
        'n_samples': args.n_samples,
        'gemini_model': os.environ.get('GEMINI_MODEL', 'gemini-3-pro-preview'),
        'metrics': metrics,
        'per_item': per_item,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'  -> wrote {args.output}')
    print(f'  MAE(mean answer) = {mae_mean}  '
          f'(items with parse: {metrics["n_items_with_parse"]}/{metrics["n_total_items"]}, '
          f'avg {metrics["avg_samples_per_item"]:.2f} parsed samples/item)')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--val_data', required=True,
                   help='Path to a pytorch_rej_*_val.json / pytorch_sanity_25sports_val.json.')
    p.add_argument('--output', required=True,
                   help='Where to write the JSON result.')
    p.add_argument('--run_label', default=None)
    p.add_argument('--prompt_mode', choices=('old', 'bracket'), default='old')
    p.add_argument('--max_concurrent', type=int, default=32)
    p.add_argument('--temperature', type=float, default=0.0)
    p.add_argument('--max_tokens', type=int, default=128)
    p.add_argument('--retries', type=int, default=4)
    p.add_argument('--limit', type=int, default=None)
    p.add_argument('--n_samples', type=int, default=1,
                   help='Number of API calls per query (averaged before MAE). '
                        'Use temperature>0 with n_samples>1 for sampled estimates.')
    p.add_argument('--keep_raw', action='store_true',
                   help='Persist Gemini.text for each sample (debugging only; bigger output).')
    args = p.parse_args()

    if not os.environ.get('GEMINI_API_KEY'):
        print('ERROR: GEMINI_API_KEY env var is not set.', file=sys.stderr)
        sys.exit(1)
    asyncio.run(main_async(args))


if __name__ == '__main__':
    main()
