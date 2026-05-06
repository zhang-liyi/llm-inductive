"""
Aggregate pyrorej_all_s{1,2,3} results and compare against the single-seed
baselines that were previously evaluated on the same datasets.

  - OE: MAE (mean) and CE (ce_mean) — from {tag}_openestimate.json summaries.
  - BT-guided: accuracy, ce_mean, ece — from {tag}_bayesian_teaching_base_guided.json
    (base test set, n=2238). Matches calibration_tables.tex; uses `_base_guided`,
    NOT `_guided`, so BT-guided and BT-nonG are on the same 2238 examples.
  - Classification: accuracy, ce_mean, ece — from {tag}_{dataset}.json
    (hellaswag = combined halves via combine_hellaswag_halves.py).

Reports pyrorej_all as mean ± SE where SE = std / sqrt(3).
"""
import json
import os
from statistics import mean, pstdev
from math import sqrt

import numpy as np


def compute_ece(probs, correct, n_bins=15):
    """15-bin ECE on max-prob confidence, matching combine_hellaswag_halves.py."""
    probs = np.asarray(probs, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    if probs.size == 0:
        return None
    conf = probs.max(axis=1)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    N = len(conf)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        n_in = int(mask.sum())
        if n_in == 0:
            continue
        bin_conf = conf[mask].mean()
        bin_acc = correct[mask].mean()
        ece += (n_in / N) * abs(bin_conf - bin_acc)
    return float(ece)


def bt_guided_summary(d):
    """Return (acc, ce_mean, ece) for a BT-guided result, computing ECE on-the-fly
    if the summary doesn't already carry it."""
    if d is None:
        return None, None, None
    s = d["summary"]["overall"]
    acc, ce, ece = s.get("accuracy"), s.get("ce_mean"), s.get("ece")
    if ece is None and "per_example" in d:
        pe = d["per_example"]
        probs = [e["probs"] for e in pe]
        correct = [e["correct"] for e in pe]
        ece = compute_ece(probs, correct)
    return acc, ce, ece

BASE = "./data_evaluation/results"

# Baselines in the existing calibration_tables.tex + the ones we have BT-guided for.
# Maps display name -> (openestimate tag, bayesian_teaching tag, text_cls tag).
BASELINES = {
    "pretrained":     ("pretrained",    "pretrained",    "base"),
    "fusion":         ("fusion",        "fusion",        "fusion"),
    "fwd_mean":       ("fwd_mean_only", "fwd_mean_only", "fwd"),
    "prob_mean":      ("prob_mean_only","prob_mean_only","prob_mean_only"),
    "prob_dist":      ("prob_dist",     "prob_dist",     "probdist"),
    "pyro_mean":      ("pyro_mean_only","pyro_mean_only","pyro_mean_only"),
    "pyro_dist":      ("pyro_dist",     "pyro_dist",     "pyrodist"),
}

PYROREJ_SEEDS = (1, 2, 3)
TEXT_DATASETS = ("mmlu", "truthfulqa", "hellaswag", "arc_challenge", "winogrande")


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def oe_metrics(d):
    """(mae_mean, ce_mean) from OpenEstimate summary, or (None, None)."""
    if d is None:
        return None, None
    s = d["summary"]
    return s["mae"]["mean"], s["ce_mean"]["mean"]


def classif_metrics(d):
    """(accuracy, ce_mean, ece) from BT/text_cls summary, or Nones."""
    if d is None:
        return None, None, None
    s = d["summary"]["overall"]
    return s.get("accuracy"), s.get("ce_mean"), s.get("ece")


def fmt(x, digits=3):
    if x is None:
        return "  —  "
    return f"{x:.{digits}f}"


def fmt_pair(m, se, digits=3):
    if m is None:
        return "     —     "
    return f"{m:.{digits}f}±{se:.{digits}f}"


# ── pyrorej_all across seeds ──────────────────────────────────────────────────
def pyrorej_all_stats():
    per_ds = {}
    # OE
    oe = [load_json(f"{BASE}/openestimate/pyrorej_all_s{s}_openestimate.json")
          for s in PYROREJ_SEEDS]
    mae = [oe_metrics(d)[0] for d in oe]
    ce = [oe_metrics(d)[1] for d in oe]
    per_ds["openestimate"] = {
        "MAE": (mean(mae), pstdev(mae) / sqrt(len(mae))),
        "CE":  (mean(ce),  pstdev(ce)  / sqrt(len(ce))),
    }
    # BT-guided
    bt = [load_json(f"{BASE}/bayesian_teaching/pyrorej_all_s{s}_bayesian_teaching_base_guided.json")
          for s in PYROREJ_SEEDS]
    acc = [bt_guided_summary(d)[0] for d in bt]
    ce  = [bt_guided_summary(d)[1] for d in bt]
    ece = [bt_guided_summary(d)[2] for d in bt]
    per_ds["BT-guided"] = {
        "acc": (mean(acc), pstdev(acc) / sqrt(len(acc))),
        "CE":  (mean(ce),  pstdev(ce)  / sqrt(len(ce))),
        "ECE": (mean(ece), pstdev(ece) / sqrt(len(ece))),
    }
    # Classification
    for ds in TEXT_DATASETS:
        rs = [load_json(f"{BASE}/text_cls/pyrorej_all_s{s}_{ds}.json")
              for s in PYROREJ_SEEDS]
        acc = [classif_metrics(r)[0] for r in rs]
        ce  = [classif_metrics(r)[1] for r in rs]
        ece = [classif_metrics(r)[2] for r in rs]
        per_ds[ds] = {
            "acc": (mean(acc), pstdev(acc) / sqrt(len(acc))),
            "CE":  (mean(ce),  pstdev(ce)  / sqrt(len(ce))),
            "ECE": (mean(ece), pstdev(ece) / sqrt(len(ece))),
        }
    return per_ds


# ── baseline stats (single seed, so SE = 0) ───────────────────────────────────
def baseline_stats(label, oe_tag, bt_tag, tc_tag):
    per_ds = {}
    d = load_json(f"{BASE}/openestimate/{oe_tag}_openestimate.json")
    m, c = oe_metrics(d)
    per_ds["openestimate"] = {"MAE": (m, None), "CE": (c, None)}

    d = load_json(f"{BASE}/bayesian_teaching/{bt_tag}_bayesian_teaching_base_guided.json")
    a, c, e = bt_guided_summary(d)
    per_ds["BT-guided"] = {"acc": (a, None), "CE": (c, None), "ECE": (e, None)}

    for ds in TEXT_DATASETS:
        d = load_json(f"{BASE}/text_cls/{tc_tag}_{ds}.json")
        a, c, e = classif_metrics(d)
        per_ds[ds] = {"acc": (a, None), "CE": (c, None), "ECE": (e, None)}
    return per_ds


# ── print a table ─────────────────────────────────────────────────────────────
def print_table(title, rows, datasets, metric_key, digits=3):
    print(f"\n### {title}")
    header = f"{'method':<15}" + "".join(f"{ds:>15}" for ds in datasets)
    print(header)
    print("-" * len(header))
    for method, per_ds in rows:
        cells = []
        for ds in datasets:
            v = per_ds.get(ds, {}).get(metric_key)
            if v is None:
                cells.append("     —     ")
            else:
                m, se = v
                if m is None:
                    cells.append("     —     ")
                elif se is None:
                    cells.append(f"{m:.{digits}f}")
                else:
                    cells.append(fmt_pair(m, se, digits))
        line = f"{method:<15}" + "".join(f"{c:>15}" for c in cells)
        print(line)


def main():
    rows = []
    for label, (oe_tag, bt_tag, tc_tag) in BASELINES.items():
        rows.append((label, baseline_stats(label, oe_tag, bt_tag, tc_tag)))
    rows.append(("pyrorej_all*", pyrorej_all_stats()))

    class_datasets = ["BT-guided", "mmlu", "truthfulqa", "hellaswag", "arc_challenge", "winogrande"]
    oe_datasets = ["openestimate"]

    # Accuracy (OE uses MAE — lower better; classification uses acc — higher better).
    print("\n========== pyrorej_all* = 3 seeds, mean ± SE (SE = std/sqrt(3)) ==========")
    print_table("Accuracy (classification) / MAE (OpenEstimate ↓)",
                rows, oe_datasets, "MAE", digits=2)
    print_table("Accuracy (classification) / — ",
                rows, class_datasets, "acc", digits=3)

    # NLL / CE
    print_table("NLL / CE_mean (↓)",
                rows, oe_datasets, "CE", digits=2)
    print_table("NLL / CE_mean (↓, classification)",
                rows, class_datasets, "CE", digits=3)

    # ECE
    print_table("ECE (↓, classification; OE is regression so skipped)",
                rows, class_datasets, "ECE", digits=3)


if __name__ == "__main__":
    main()
