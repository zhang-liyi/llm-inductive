"""Paired comparison of two self-consistency rows on identical examples.

`summarize_selfconsistency.py` compares *greedy vs SC* for one model.  This
compares *two models under the same decoding scheme* — the comparison line 1
of the rebuttal needs:

    Base + CoT + SC              (tag `cot_sc`)
    Posterior + CoT + SC         (tag `cotsc_pyrorej_all_s{S}_bracket`)

Both rows are scored on the same 500-example MCQ subsample, the same full BT
and OpenEstimate sets, and — because `sc_seed` keys the RNG on (base seed,
absolute dataset index, path index) and ignores the model — the same k random
draws per example.  So the comparison is paired at the example level and the
uncertainty that matters is a paired bootstrap over examples, not the
between-seed SE.  Both are reported: `+/-` is the seed SE where more than one
fine-tuned seed is present, `[lo, hi]` is the paired-bootstrap CI on the
difference.

Metrics follow the paper's aggregators exactly (accuracy = argmax of the
mixture of per-path softmaxes, ce_mean = mean NLL of the mixture, ece =
15-bin ECE on the mixture's max probability), so the printed level numbers
reproduce `summary.overall` up to the paired-index restriction.  The
self-consistency extras (`vote_acc`, `vote_ece`) are printed alongside.

Usage
-----
    python compare_sc_rows.py                      # base vs Posterior seed 1
    python compare_sc_rows.py --seeds 1 2 3
    python compare_sc_rows.py --a cot_sc --b 'cotsc_pyrorej_all_s{S}_bracket'
"""
import argparse
import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
RES = f"{ROOT}/results"

from cot_eval_utils import binned_ece  # noqa: E402

# (subdir, file stem, display name)
BENCHES = [
    ("openestimate", "openestimate", "OpenEstimate"),
    ("bayesian_teaching", "bt_base_tf", "BT-nonG"),
    ("bayesian_teaching", "bayesian_teaching_base_guided", "BT-guided"),
    ("text_cls", "mmlu", "MMLU"),
    ("text_cls", "truthfulqa", "TruthfulQA"),
    ("text_cls", "hellaswag", "HellaSwag"),
    ("text_cls", "arc_challenge", "ARC-C"),
    ("text_cls", "winogrande", "Winogrande"),
]

# name -> (higher_is_better, format spec)
CLS_METRICS = [("acc", True, ">7.4f"), ("vote_acc", True, ">7.4f"),
               ("nll", False, ">7.4f"), ("ece", False, ">7.4f"),
               ("vote_ece", False, ">7.4f")]
OE_METRICS = [("mae", False, ">7.3f"), ("ce_mean", False, ">7.3f"),
              ("ce_dist", False, ">7.3f")]


def load(path: str) -> Optional[dict]:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None
    with open(path) as f:
        return json.load(f)


def _key(rec: dict) -> Optional[int]:
    """Absolute dataset index, or None for records that predate index logging.

    Greedy shards written before ``--sample_idx`` support carry neither key;
    they cannot be paired at all, so they are dropped rather than aligned by
    position (which would silently pair different questions).
    """
    k = rec.get("global_idx")
    if k is None:
        k = rec.get("idx")
    return int(k) if k is not None else None


def by_index(payload: dict) -> Dict[int, dict]:
    out = {}
    for r in payload.get("per_example", []):
        k = _key(r)
        if k is not None:
            out[k] = r
    return out


# ── per-example quantities ────────────────────────────────────────────────
def _cls_vectors(recs: List[dict], family: str) -> dict:
    """Per-example arrays for a classification benchmark.

    `probs` is the mixture over reasoning paths, so these reproduce the
    aggregators' definitions rather than re-deriving anything new.
    """
    probs = [np.asarray(r["probs"], dtype=np.float64) for r in recs]
    true = [int(r["true_idx"]) if family == "text_cls" else int(r["gt"]) - 1
            for r in recs]
    return {
        "correct": np.array([float(np.argmax(p) == t)
                             for p, t in zip(probs, true)]),
        "conf": np.array([float(np.max(p)) for p in probs]),
        "nll": np.array([float(-np.log(p[t] + 1e-10))
                         for p, t in zip(probs, true)]),
        "vote_correct": np.array([float(bool(r.get("vote_correct")))
                                  for r in recs]),
        "vote_conf": np.array([float(r.get("vote_margin") or 0.0)
                               for r in recs]),
    }


def _oe_vectors(recs: List[dict]) -> dict:
    def col(k):
        return np.array([float(r.get(k, np.nan)) for r in recs])
    return {"mae": col("mae"), "ce_mean": col("ce_mean"),
            "ce_dist": col("ce_dist")}


def _cls_stats(v: dict, sel: np.ndarray) -> dict:
    return {
        "acc": float(v["correct"][sel].mean()),
        "vote_acc": float(v["vote_correct"][sel].mean()),
        "nll": float(v["nll"][sel].mean()),
        "ece": binned_ece(v["conf"][sel], v["correct"][sel]),
        "vote_ece": binned_ece(v["vote_conf"][sel], v["vote_correct"][sel]),
    }


def _oe_stats(v: dict, sel: np.ndarray) -> dict:
    return {k: float(np.nanmean(v[k][sel])) for k in ("mae", "ce_mean", "ce_dist")}


# ── one benchmark ─────────────────────────────────────────────────────────
def compare_benchmark(sub: str, stem: str, tag_a: str, tags_b: List[str],
                      n_boot: int, rng: np.random.Generator):
    a_pay = load(f"{RES}/{sub}/{tag_a}_{stem}.json")
    b_pays = [(t, load(f"{RES}/{sub}/{t}_{stem}.json")) for t in tags_b]
    present = [(t, p) for t, p in b_pays if p is not None]
    if a_pay is None or not present:
        missing = ([tag_a] if a_pay is None else []) + \
                  [t for t, p in b_pays if p is None]
        return {"missing": missing}

    # A greedy row has no per-path vote, and `vote_correct` would silently
    # read as 0 for every example rather than erroring.
    thin = [t for t, p in [(tag_a, a_pay)] + present
            if (p.get("n_samples") or 1) < 2]
    if thin and sub != "openestimate":
        print(f"  [WARNING] {stem}: {thin} have n_samples<2 — the vote_* rows "
              f"are meaningless for them. Use summarize_selfconsistency.py "
              f"for greedy-vs-SC.")

    family = "openestimate" if sub == "openestimate" else (
        "text_cls" if sub == "text_cls" else "bayesian_teaching")
    a_idx = by_index(a_pay)
    b_idxs = [by_index(p) for _, p in present]

    # Paired on the intersection: a seed still running contributes only the
    # examples it has, and every row is then scored on that same subset.
    common = sorted(set(a_idx) & set.intersection(*[set(d) for d in b_idxs]))
    if not common:
        return {"missing": ["no shared example indices"]}

    a_recs = [a_idx[i] for i in common]
    b_recs = [[d[i] for i in common] for d in b_idxs]

    if family == "openestimate":
        vec, stats = _oe_vectors, _oe_stats
        metrics = OE_METRICS
    else:
        vec = lambda rs: _cls_vectors(rs, family)  # noqa: E731
        stats, metrics = _cls_stats, CLS_METRICS
        # Pairing sanity: the same absolute index must be the same question.
        gold = "true_idx" if family == "text_cls" else "gt"
        for rs in b_recs:
            mism = sum(1 for x, y in zip(a_recs, rs)
                       if x.get(gold) != y.get(gold))
            if mism:
                return {"missing": [f"{mism} gold-label mismatches — "
                                    f"pairing is wrong"]}

    a_v = vec(a_recs)
    b_vs = [vec(rs) for rs in b_recs]
    n = len(common)
    full = np.arange(n)

    out = {"n": n, "k_a": a_pay.get("n_samples"),
           "k_b": present[0][1].get("n_samples"),
           "seeds": [t for t, _ in present], "metrics": {}}

    a_stat = stats(a_v, full)
    b_per_seed = [stats(v, full) for v in b_vs]

    # Bootstrap over examples, resampling both rows on the same draw so the
    # pairing (and the shared random reasoning paths) is preserved.
    boot = {m: [] for m, _, _ in metrics}
    for _ in range(n_boot):
        sel = rng.integers(0, n, size=n)
        sa = stats(a_v, sel)
        sb = [stats(v, sel) for v in b_vs]
        for m, _, _ in metrics:
            boot[m].append(float(np.mean([s[m] for s in sb])) - sa[m])

    for m, higher_better, spec in metrics:
        vals = [s[m] for s in b_per_seed]
        b_mean = float(np.mean(vals))
        b_se = (float(np.std(vals, ddof=0) / np.sqrt(len(vals)))
                if len(vals) > 1 else None)
        d = np.asarray(boot[m])
        lo, hi = np.percentile(d, [2.5, 97.5])
        p = 2.0 * min(float(np.mean(d <= 0)), float(np.mean(d >= 0)))
        out["metrics"][m] = {
            "a": a_stat[m], "b": b_mean, "b_se": b_se,
            "delta": b_mean - a_stat[m], "lo": float(lo), "hi": float(hi),
            "p": min(1.0, p), "better": higher_better, "spec": spec,
        }
    return out


def _fmt(v, spec):
    return format(v, spec) if isinstance(v, (int, float)) and not (
        isinstance(v, float) and np.isnan(v)) else "    ---"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="cot_sc",
                    help="Reference row tag (Base + CoT + SC).")
    ap.add_argument("--b", default="cotsc_pyrorej_all_s{S}_bracket",
                    help="Comparison row tag; {S} is filled from --seeds.")
    ap.add_argument("--seeds", nargs="+", type=int, default=[1])
    ap.add_argument("--label_a", default="Base+CoT+SC")
    ap.add_argument("--label_b", default="Post+CoT+SC")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--boot_seed", type=int, default=0)
    args = ap.parse_args()

    tags_b = ([args.b.format(S=s) for s in args.seeds] if "{S}" in args.b
              else [args.b])
    rng = np.random.default_rng(args.boot_seed)

    print(f"{args.label_a}  ({args.a})   vs   {args.label_b}  "
          f"({', '.join(tags_b)})")
    print(f"paired bootstrap over examples, B={args.n_boot}; "
          f"+/- is the between-seed SE\n")

    wins = {"b": 0, "a": 0, "tie": 0}
    for sub, stem, name in BENCHES:
        r = compare_benchmark(sub, stem, args.a, tags_b, args.n_boot, rng)
        if "missing" in r:
            print(f"{name:<14} --- missing: {', '.join(map(str, r['missing']))}")
            continue
        head = f"{name} (n={r['n']}, k={r['k_a']} vs {r['k_b']}"
        if len(r["seeds"]) > 1:
            head += f", {len(r['seeds'])} seeds"
        print(head + ")")
        print(f"  {'metric':<10}{args.label_a:>14}{args.label_b:>16}"
              f"{'delta':>10}{'95% CI':>20}{'p':>8}")
        for m, cell in r["metrics"].items():
            se = (f" +/-{cell['b_se']:.4f}" if cell["b_se"] is not None
                  else "        ")
            star = ""
            if cell["lo"] > 0 or cell["hi"] < 0:
                improved = (cell["delta"] > 0) == cell["better"]
                star = "  <<" if improved else "  >>"
                wins["b" if improved else "a"] += 1
            else:
                wins["tie"] += 1
            print(f"  {m:<10}{_fmt(cell['a'], cell['spec']):>14}"
                  f"{_fmt(cell['b'], cell['spec']):>9}{se}"
                  f"{cell['delta']:>+10.4f}"
                  f"   [{cell['lo']:+.4f}, {cell['hi']:+.4f}]"
                  f"{cell['p']:>8.3f}{star}")
        print()

    print(f"Significant cells: {wins['b']} favour {args.label_b}, "
          f"{wins['a']} favour {args.label_a}, {wins['tie']} inconclusive "
          f"(95% CI on the paired difference straddles 0).")
    print("'<<' = the comparison row is significantly better on that metric, "
          "'>>' = significantly worse.")


if __name__ == "__main__":
    main()
