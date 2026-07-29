"""
summarize_morebench.py

Read the MoReBench result JSONs written by ``evaluate_text_classification.py``
and print a base-vs-fine-tuned comparison: accuracy / NLL / ECE overall, plus
the slices that only MoReBench has (rubric dimension, criterion weight, gold
side, dilemma source or moral theory).

Usage
-----
    python data_evaluation/summarize_morebench.py \\
        --results_dir data_evaluation/results/text_cls \\
        --tags base pyrorej_all_s1_bracket

Any tag whose JSON is missing is skipped with a warning, so this is safe to run
while jobs are still queued.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

DATASETS = ("morebench", "morebench_theory")


def compute_ece(conf: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece, N = 0.0, len(conf)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf >= lo) & (conf <= hi) if i == n_bins - 1 else (conf >= lo) & (conf < hi)
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / N) * abs(conf[mask].mean() - correct[mask].mean())
    return float(ece)


def metrics(items: List[dict]) -> dict:
    if not items:
        return {"n": 0}
    probs = np.array([it["probs"] for it in items])
    true_idx = np.array([it["true_idx"] for it in items])
    correct = (probs.argmax(axis=1) == true_idx).astype(float)
    p_true = probs[np.arange(len(items)), true_idx]
    return {
        "n": len(items),
        "acc": float(correct.mean()),
        "nll": float(-np.log(np.clip(p_true, 1e-12, 1.0)).mean()),
        "ece": compute_ece(probs.max(axis=1), correct),
        # Fraction of probability mass the model puts outside {A, B} at all.
        # Both datasets are 2-way but the softmax is over A..Z, so this is a
        # useful format-adherence check rather than a calibration number.
        "p_offchoice": float(1.0 - probs[:, :2].sum(axis=1).mean()),
    }


def load(results_dir: Path, tag: str, dataset: str):
    path = results_dir / f"{tag}_{dataset}.json"
    if not path.exists():
        print(f"  !! missing {path}")
        return None
    with open(path) as fh:
        return json.load(fh)["per_example"]


def slice_by(items: List[dict], key) -> Dict[str, List[dict]]:
    out = defaultdict(list)
    for it in items:
        out[str(key(it))].append(it)
    return dict(sorted(out.items()))


ROW = "  {:<34s} {:>6s} {:>8s} {:>8s} {:>8s}"


def print_block(title: str, per_tag: Dict[str, List[dict]], tags: List[str],
                slicer=None) -> None:
    print(f"\n{title}")
    print(ROW.format("", "n", "acc", "NLL", "ECE"))
    if slicer is None:
        groups = {"overall": None}
    else:
        # Slice keys are taken from the first available tag; every tag scores
        # the identical item set, so the groups line up.
        first = per_tag[tags[0]]
        groups = {k: None for k in slice_by(first, slicer)}
    for g in groups:
        for tag in tags:
            items = per_tag[tag]
            if slicer is not None:
                items = [it for it in items if str(slicer(it)) == g]
            m = metrics(items)
            label = f"{g[:22]:<22s} {tag[:11]}" if slicer is not None else tag
            print(ROW.format(label, str(m["n"]), f"{m['acc']:.3f}",
                             f"{m['nll']:.3f}", f"{m['ece']:.3f}"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="data_evaluation/results/text_cls")
    ap.add_argument("--tags", nargs="+",
                    default=["base", "pyrorej_all_s1_bracket"])
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    args = ap.parse_args()

    rd = Path(args.results_dir)
    for dataset in args.datasets:
        print(f"\n{'=' * 78}\n{dataset}\n{'=' * 78}")
        per_tag = {}
        for tag in args.tags:
            items = load(rd, tag, dataset)
            if items is not None:
                per_tag[tag] = items
        tags = [t for t in args.tags if t in per_tag]
        if not tags:
            continue

        ns = {t: len(per_tag[t]) for t in tags}
        if len(set(ns.values())) > 1:
            print(f"  !! tags disagree on n: {ns} — slices may not be paired")

        print_block("overall", per_tag, tags)
        print_block("by gold side (include = weight>0, avoid = weight<0)",
                    per_tag, tags, lambda it: it["meta"]["gold"])
        print_block("by rubric dimension", per_tag, tags,
                    lambda it: it["meta"]["dimension"])
        print_block("by criterion weight", per_tag, tags,
                    lambda it: f"{it['meta']['weight']:+d}")
        print_block("by task", per_tag, tags, lambda it: it["task"])

        # Yes/No bias: how often each model picks the "Yes" letter regardless
        # of gold.  A model that always says Yes lands at 0.5 accuracy here by
        # construction, so this is the diagnostic that tells the two apart.
        print("\nanswer bias (fraction of items answered 'Yes')")
        for tag in tags:
            items = per_tag[tag]
            yes = np.mean([
                (it["pred_idx"] == 0) == (it["meta"]["yes_letter"] == "A")
                for it in items
            ])
            off = metrics(items)["p_offchoice"]
            print(f"  {tag:<34s} P(say Yes)={yes:.3f}   "
                  f"mass outside A/B={off:.4f}")


if __name__ == "__main__":
    main()
