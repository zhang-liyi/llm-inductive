"""
Combine two hellaswag half-result JSONs into a single result file,
re-computing the summary statistics from the merged per_example lists.

Usage:
    python combine_hellaswag_halves.py <h1.json> <h2.json> <output.json>
"""
import argparse
import json
import sys
from collections import defaultdict

import numpy as np


def compute_ece(confidences, correct, n_bins=15):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    N = len(confidences)
    if N == 0:
        return 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)
        n_in = int(mask.sum())
        if n_in == 0:
            continue
        bin_conf = confidences[mask].mean()
        bin_acc = correct[mask].mean()
        ece += (n_in / N) * abs(bin_conf - bin_acc)
    return float(ece)


def aggregate(items):
    if not items:
        return {"n": 0}
    probs = np.array([it["probs"] for it in items])
    true_idx = np.array([it["true_idx"] for it in items])
    pred_idx = probs.argmax(axis=1)
    correct = (pred_idx == true_idx).astype(float)
    p_true = probs[np.arange(len(items)), true_idx]
    nll = -np.log(np.clip(p_true, 1e-12, 1.0)).mean()
    conf = probs.max(axis=1)
    ece = compute_ece(conf, correct)
    return {
        "n": len(items),
        "accuracy": float(correct.mean()),
        "ce_mean": float(nll),
        "mae": float(np.abs(pred_idx - true_idx).mean()),
        "ece": ece,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("h1")
    parser.add_argument("h2")
    parser.add_argument("output")
    args = parser.parse_args()

    with open(args.h1) as f:
        d1 = json.load(f)
    with open(args.h2) as f:
        d2 = json.load(f)

    all_examples = d1["per_example"] + d2["per_example"]

    by_task = defaultdict(list)
    for ex in all_examples:
        by_task[ex.get("task", "unknown")].append(ex)

    summary = {
        "overall": aggregate(all_examples),
        "by_task": {t: aggregate(exs) for t, exs in sorted(by_task.items())},
    }

    out = {
        "ckpt_dir": d1["ckpt_dir"],
        "dataset": d1["dataset"],
        "n_examples": len(all_examples),
        "max_seq_len": d1.get("max_seq_len"),
        "pretrained": d1.get("pretrained"),
        "summary": summary,
        "per_example": all_examples,
    }

    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Combined {len(d1['per_example'])} + {len(d2['per_example'])} = {len(all_examples)} examples")
    print(f"Overall accuracy: {summary['overall']['accuracy']:.4f}")
    print(f"  ce_mean: {summary['overall']['ce_mean']:.4f}, ece: {summary['overall']['ece']:.4f}")
    print(f"Written to {args.output}")


if __name__ == "__main__":
    main()
