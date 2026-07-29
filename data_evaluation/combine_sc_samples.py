"""Combine k single-path runs into one self-consistency result.

Self-consistency needs k reasoning paths per example.  Running them inside one
process (``--n_samples k``) makes each job k times longer; running them as k
separate jobs (``--sample_idx j``) keeps every job short and lets partial
results still be used — if 7 of 10 paths finish, you get self-consistency at
k=7 instead of nothing.  This script does the recombination.

It expects per-sample files produced with ``--sample_idx j`` (optionally also
sharded over examples), groups their per-example records by benchmark and
absolute dataset index, and applies the same aggregation the in-process path
uses:

    mixture  mean of the per-path restricted softmaxes -> accuracy
    vote     majority vote over per-path answers       -> vote_accuracy
    vote_ECE calibration from the fraction of paths backing the winner

Usage
-----
    # one benchmark
    python combine_sc_samples.py \\
        --inputs 'results/text_cls/cot_sc_mmlu_s*_shard*.json' \\
        --output_file results/text_cls/cot_sc_mmlu.json

    # every benchmark for a tag
    python combine_sc_samples.py --auto --tag cot_sc
"""
import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict
from typing import List, Optional

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)
RES = f"{_THIS_DIR}/results"

from cot_eval_utils import (  # noqa: E402
    majority_vote, marginalize, sc_extras, vote_distribution,
)

# "<tag>_<name>_s<sample>[_shard<k>].json"
_SAMPLE_RE = re.compile(r"_s(\d+)(?:_shard\d+)?\.json$")


def _family(rec: dict) -> str:
    if "true_idx" in rec:
        return "text_cls"
    if "gt_mean" in rec:
        return "openestimate"
    if "gt" in rec:
        return "bayesian_teaching"
    raise ValueError(f"Cannot infer family from keys: {sorted(rec)}")


def _key(rec: dict):
    k = rec.get("global_idx")
    return k if k is not None else rec.get("idx")


def combine(paths: List[str], output_file: str,
            expected: Optional[int] = None) -> dict:
    paths = [p for p in sorted(paths) if not p.endswith("_partial.json")]
    if not paths:
        raise SystemExit("No per-sample files matched.")

    payloads = [json.load(open(p)) for p in paths]
    groups = defaultdict(list)
    for pay in payloads:
        for r in pay.get("per_example", []):
            k = _key(r)
            if k is None:
                raise SystemExit(
                    "Records lack global_idx/idx — they predate --sample_idx "
                    "support and cannot be aligned across samples.")
            groups[k].append(r)
    if not groups:
        raise SystemExit("No per-example records found.")

    fam = _family(next(iter(groups.values()))[0])
    counts = {len(v) for v in groups.values()}
    if len(counts) > 1:
        print(f"  [WARNING] uneven path counts across examples: {sorted(counts)}"
              f" — some sample jobs are missing or incomplete. Every example "
              f"is aggregated over whatever paths it has, so cells are not "
              f"directly comparable.")
    k_paths = max(counts)

    # OpenEstimate needs ground-truth bins to recompute ce_dist over the
    # mixture; load them once, keyed by absolute dataset index.
    oe_bins = None
    if fam == "openestimate":
        from evaluate_openestimate import load_data as _oe_load
        head0 = payloads[0]
        data = _oe_load(head0["data_path"], split=head0.get("split", "all"),
                        n_examples=None)
        oe_bins = {i: ex["bins"][0] for i, ex in enumerate(data)}

    # The answer distribution lives under a different key per family: the
    # classification evals emit ``probs`` over letters/choices, OpenEstimate
    # emits ``pred_dist`` over the 101 integer bins.  Reading the wrong one
    # silently yields no records at all rather than an error.
    probs_key = "pred_dist" if fam == "openestimate" else "probs"

    merged = []
    for gid in sorted(groups):
        recs = sorted(groups[gid], key=lambda r: r.get("sample_idx") or 0)
        probs_list = [np.asarray(r[probs_key], dtype=np.float64) for r in recs
                      if r.get(probs_key)]
        base = dict(recs[0])
        n_here = len(recs)

        if fam == "openestimate":
            # Continuous quantity: the mixture over paths *is* the marginal;
            # there is no vote to take. ce_dist needs the ground-truth bins,
            # which per-example records do not carry, so they are reloaded
            # from the dataset and indexed by absolute position.
            from evaluate_openestimate_cot import metrics_from_probs
            if not probs_list:
                continue
            pm = marginalize(probs_list)
            pred_std = float(np.mean([r.get("pred_std", float("nan"))
                                      for r in recs]))
            bins = oe_bins.get(gid) if oe_bins else None
            if bins is None:
                raise SystemExit(
                    f"No ground-truth bins for OpenEstimate index {gid}; "
                    f"cannot recompute ce_dist.")
            out = dict(base)
            out.update(metrics_from_probs(pm, pred_std, bins,
                                          float(base.get("gt_std", 0.0))))
            out.update({
                "n_samples": n_here,
                "sample_pred_means": [r.get("pred_mean") for r in recs],
                "pred_mean_spread": (float(np.std([r["pred_mean"] for r in recs]))
                                     if n_here > 1 else float("nan")),
                "n_gen_tokens": float(np.mean([r["n_gen_tokens"] for r in recs])),
                "truncated": all(bool(r.get("truncated")) for r in recs),
            })
            merged.append(out)
            continue

        # Discrete: vote over per-path answers, mixture for the distribution.
        if fam == "text_cls":
            per_pred = [r["pred_idx"] if r.get("parsed_letter") is None
                        else _letter_idx(r["parsed_letter"], r["pred_idx"])
                        for r in recs]
            n_classes = len(probs_list[0])
            true_idx = base["true_idx"]
        else:
            per_pred = [(r["parsed_pred"] - 1) if r.get("parsed_pred") else (r["pred"] - 1)
                        for r in recs]
            n_classes = len(probs_list[0])
            true_idx = base["gt"] - 1

        pm = marginalize(probs_list)
        pred_idx = int(np.argmax(pm))
        vote_idx = majority_vote(per_pred, pm)
        votes = vote_distribution(per_pred, n_classes)
        out = dict(base)
        out.update({
            "n_samples": n_here,
            "probs": pm.tolist(),
            "vote_probs": votes,
            "vote_pred": vote_idx,
            "pred_idx_marginal": pred_idx,
            "vote_correct": vote_idx == true_idx,
            "vote_margin": max(votes) if votes else None,
            "sample_preds": per_pred,
            "n_gen_tokens": float(np.mean([r["n_gen_tokens"] for r in recs])),
            "truncated": all(bool(r.get("truncated")) for r in recs),
        })
        if fam == "text_cls":
            from evaluate_text_classification import CHOICE_LETTERS
            out.update({"pred_idx": pred_idx,
                        "pred_letter": CHOICE_LETTERS[pred_idx]})
        else:
            out.update({"pred": pred_idx + 1,
                        "correct": (pred_idx + 1) == base["gt"],
                        "ce": float(-np.log(pm[true_idx] + 1e-10))})
        merged.append(out)

    summary = _summarize(fam, merged)
    head = dict(payloads[0])
    head.pop("per_example", None)
    head.pop("sample_idx", None)
    head.pop("start_idx", None)
    head.update({
        "mode": "cot_sc",
        "n_samples": k_paths,
        "n_source_files": len(paths),
        "n_examples": len(merged),
        "summary": summary,
        "per_example": merged,
    })
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    with open(output_file, "w") as fh:
        json.dump(head, fh, indent=2, default=str)

    ov = summary.get("overall", summary)
    n = ov.get("n", summary.get("n"))
    if expected and n != expected:
        print(f"  [WARNING] {os.path.basename(output_file)}: n={n} but expected "
              f"{expected} — sample jobs are missing or still running.")
    msg = (f"  {os.path.basename(output_file)}: {len(paths)} files, k={k_paths} "
           f"-> n={n}")
    if "accuracy" in ov:
        msg += (f"  vote_acc={ov.get('vote_accuracy', float('nan')):.4f}"
                f"  marg_acc={ov['accuracy']:.4f}"
                f"  vote_ECE={ov.get('vote_ece', float('nan')):.4f}")
    else:
        msg += f"  MAE={summary['mae']['mean']:.3f}"
    print(msg)
    return summary


def _letter_idx(letter, fallback):
    from evaluate_text_classification import CHOICE_LETTERS
    return CHOICE_LETTERS.index(letter) if letter in CHOICE_LETTERS else fallback


def _summarize(fam, items):
    if fam == "text_cls":
        from evaluate_text_classification_cot import summarize as f
        return f(items, 0)
    if fam == "bayesian_teaching":
        from evaluate_bayesian_teaching_cot import summarize as f
        return f(items)
    from evaluate_openestimate import aggregate_metrics
    s = aggregate_metrics(items)
    s["valid_rate"] = float(np.mean([bool(r.get("valid")) for r in items]))
    s["trunc_rate"] = float(np.mean([bool(r.get("truncated")) for r in items]))
    s["mean_gen_tokens"] = float(np.mean([r.get("n_gen_tokens", 0) for r in items]))
    sp = [r.get("pred_mean_spread") for r in items]
    sp = [v for v in sp if isinstance(v, (int, float)) and not np.isnan(v)]
    if sp:
        s["mean_pred_spread"] = float(np.mean(sp))
    return s


def auto(tag: str) -> None:
    from merge_cot_shards import EXPECTED_SC
    groups = defaultdict(list)
    for sub in ("text_cls", "bayesian_teaching", "openestimate"):
        for p in glob.glob(f"{RES}/{sub}/{tag}_*_s*.json"):
            if p.endswith("_partial.json") or not _SAMPLE_RE.search(p):
                continue
            base = _SAMPLE_RE.sub(".json", p)
            groups[base].append(p)
    if not groups:
        print(f"No per-sample files found for tag {tag!r} under {RES}.")
        return
    for base, files in sorted(groups.items()):
        stem = os.path.basename(base)[:-len(".json")]
        if stem.startswith(tag + "_"):
            stem = stem[len(tag) + 1:]
        combine(files, base, expected=EXPECTED_SC.get(stem))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs")
    ap.add_argument("--output_file")
    ap.add_argument("--auto", action="store_true")
    ap.add_argument("--tag", default="cot_sc")
    ap.add_argument("--expect", type=int, default=None)
    args = ap.parse_args()
    if args.auto:
        auto(args.tag)
        return
    if not args.inputs or not args.output_file:
        ap.error("Provide --inputs and --output_file, or --auto.")
    combine(glob.glob(args.inputs), args.output_file, expected=args.expect)


if __name__ == "__main__":
    main()
