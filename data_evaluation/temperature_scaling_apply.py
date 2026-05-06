"""
temperature_scaling_apply.py

Apply a fitted temperature T to an existing pretrained eval JSON, producing
the temperature-scaled calibration baseline.

For each per-example `probs` vector p in the input eval JSON, we compute
    p' = softmax(log(p) / T)
and recompute overall / per-task acc, NLL (CE over the true class),
MAE (not meaningful for TS since acc is preserved — included for parity),
and ECE (15-bin) using the same aggregator as the original eval scripts.

Usage examples:
  # Standard datasets
  python temperature_scaling_apply.py \\
      --fit_file results/text_cls/ts_fit_mmlu.json \\
      --eval_file results/text_cls/pretrained_mmlu.json \\
      --output_file results/text_cls/ts_mmlu.json

  # TruthfulQA: no train split, so T=1 (TS row == pretrained row)
  python temperature_scaling_apply.py \\
      --T 1.0 \\
      --eval_file results/text_cls/pretrained_truthfulqa.json \\
      --output_file results/text_cls/ts_truthfulqa.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS))

from evaluate_text_classification import compute_ece  # noqa: E402


# ── math ─────────────────────────────────────────────────────────────────────

def apply_T(probs, T):
    """Row-wise softmax(log(p)/T). probs: (N, K) ndarray. Returns (N, K)."""
    p = np.clip(np.asarray(probs, dtype=np.float64), 1e-12, 1.0)
    z = np.log(p) / T
    z = z - z.max(axis=-1, keepdims=True)
    ez = np.exp(z)
    return ez / ez.sum(axis=-1, keepdims=True)


def _aggregate(probs, true_idx, n_bins=15):
    """Reproduce the aggregator used by evaluate_text_classification.py /
    evaluate_bayesian_teaching.py (probs-only path)."""
    if len(probs) == 0:
        return {"n": 0}
    probs = np.asarray(probs)
    true_idx = np.asarray(true_idx)
    pred_idx = probs.argmax(axis=1)
    correct = (pred_idx == true_idx).astype(float)
    p_true = probs[np.arange(len(probs)), true_idx]
    nll = -np.log(np.clip(p_true, 1e-12, 1.0)).mean()
    conf = probs.max(axis=1)
    ece = compute_ece(conf, correct, n_bins=n_bins)
    return {
        "n": int(len(probs)),
        "accuracy": float(correct.mean()),
        "ce_mean": float(nll),
        "mae": float(np.abs(pred_idx - true_idx).mean()),
        "ece": float(ece),
    }


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--fit_file", help="ts_fit_<dataset>.json produced by "
                   "temperature_scaling_fit.py (reads T from it).")
    g.add_argument("--T", type=float, help="Explicit temperature value "
                   "(use 1.0 for TruthfulQA / no-fit case).")
    ap.add_argument("--eval_file", required=True,
                    help="Existing pretrained eval JSON with per_example probs.")
    ap.add_argument("--output_file", required=True)
    ap.add_argument("--n_bins", type=int, default=15)
    args = ap.parse_args()

    if args.fit_file:
        with open(args.fit_file) as fh:
            T = float(json.load(fh)["T"])
        T_src = args.fit_file
    else:
        T = float(args.T)
        T_src = f"explicit T={T}"

    with open(args.eval_file) as fh:
        eval_data = json.load(fh)

    per_ex = eval_data["per_example"]
    probs_old = np.array([ex["probs"] for ex in per_ex])
    true_idx = np.array([ex["true_idx"] if "true_idx" in ex
                         else int(ex["gt"]) - 1
                         for ex in per_ex])

    probs_new = apply_T(probs_old, T)

    # Rebuild per_example with new probs (keep all other fields)
    new_per_ex = []
    for i, ex in enumerate(per_ex):
        new_ex = dict(ex)
        new_ex["probs"] = probs_new[i].tolist()
        new_ex["pred_idx" if "pred_idx" in ex else "pred"] = (
            int(probs_new[i].argmax()) + (0 if "pred_idx" in ex else 1)
        )
        new_per_ex.append(new_ex)

    # Overall summary
    overall = _aggregate(probs_new, true_idx, n_bins=args.n_bins)

    # Per-task summary (group by the same key the original eval used)
    task_key = "task" if "task" in per_ex[0] else None
    by_task = {}
    if task_key is not None:
        tasks = np.array([ex[task_key] for ex in per_ex])
        for t in sorted(set(tasks.tolist())):
            mask = tasks == t
            by_task[t] = _aggregate(probs_new[mask], true_idx[mask],
                                    n_bins=args.n_bins)

    new_summary = {
        "overall": overall,
        "by_task": by_task,
    }
    for k in ("skipped",):
        if "summary" in eval_data and k in eval_data["summary"]:
            new_summary[k] = eval_data["summary"][k]

    payload = dict(eval_data)
    payload["ts_T"] = T
    payload["ts_source"] = T_src
    payload["summary"] = new_summary
    payload["per_example"] = new_per_ex

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)),
                exist_ok=True)
    with open(args.output_file, "w") as fh:
        json.dump(payload, fh, indent=2, default=str)

    ov = overall
    print(f"T = {T:.4f}  (from {T_src})")
    print(f"  n={ov['n']}  acc={ov['accuracy']:.3f}  "
          f"NLL={ov['ce_mean']:.3f}  ECE={ov['ece']:.3f}")
    print(f"Saved {args.output_file}")


if __name__ == "__main__":
    main()
