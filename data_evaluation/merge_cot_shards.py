"""
merge_cot_shards.py

CoT evaluation is generation-bound, so the long benchmarks are run as
``--start_idx/--n_examples`` shards across many SLURM jobs.  This merges a
set of shard JSONs back into a single results file with the same schema the
non-sharded scripts produce, so the calibration aggregators can read it
unchanged.

The benchmark family is inferred from the per-example schema:
    ``true_idx``  → text classification (MCQ)
    ``gt``        → Bayesian Teaching
    ``gt_mean``   → OpenEstimate

Usage
-----
    python merge_cot_shards.py \\
        --shards 'results/text_cls/cot_base_mmlu_shard*.json' \\
        --output_file results/text_cls/cot_base_mmlu.json

    # merge every shard family found under results/ in one go
    python merge_cot_shards.py --auto --tag cot_base
"""

import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))

RES = str(_THIS_DIR / "results")
_SHARD_RE = re.compile(r"_shard\d+\.json$")

# Expected example count per benchmark, keyed by the ``{tag}_{name}.json`` stem.
# A merge that comes up short means a shard job died or is still running; the
# aggregators cannot tell a half-finished benchmark from a finished one, so the
# check happens here.
EXPECTED = {
    "mmlu": 1531, "truthfulqa": 817, "hellaswag_h1": 5021,
    "hellaswag_h2": 5021, "winogrande": 1267, "arc_challenge": 299,
    "bt_base_tf": 2238, "bayesian_teaching_base_guided": 2238,
    "openestimate": 181,
}

# The self-consistency run costs k times as much generation, so OE and BT are
# run whole while the MCQ benchmarks are cut to a deterministic 500-example
# subsample (ARC-C has only 299, so it runs whole). Same stems, different
# expected counts — hence a separate table rather than one shared dict.
EXPECTED_SC = {
    "mmlu": 500, "truthfulqa": 500, "hellaswag": 500,
    "winogrande": 500, "arc_challenge": 299,
    "bt_base_tf": 2238, "bayesian_teaching_base_guided": 2238,
    "openestimate": 181,
}


def _expected_for(output_file: str, tag: str) -> Optional[int]:
    stem = os.path.basename(output_file)[:-len(".json")]
    if stem.startswith(tag + "_"):
        stem = stem[len(tag) + 1:]
    table = EXPECTED_SC if tag.startswith("cot_sc") else EXPECTED
    return table.get(stem)


def _load(path: str) -> dict:
    with open(path) as fh:
        return json.load(fh)


def _family(items: List[dict]) -> str:
    probe = items[0]
    if "true_idx" in probe:
        return "text_cls"
    if "gt_mean" in probe:
        return "openestimate"
    if "gt" in probe:
        return "bayesian_teaching"
    raise ValueError(f"Cannot infer benchmark family from keys: {sorted(probe)}")


def merge(shard_paths: List[str], output_file: str,
          expected: int = None) -> dict:
    # A user-supplied glob like "*_shard*.json" also matches the in-progress
    # "*_shard0_partial.json" checkpoints, whose contents are a prefix of the
    # finished shard — merging both double-counts those examples. Filter here
    # rather than only in auto_merge so both entry points are safe.
    dropped = [p for p in shard_paths if p.endswith("_partial.json")]
    shard_paths = [p for p in shard_paths if not p.endswith("_partial.json")]
    if dropped:
        print(f"  (ignoring {len(dropped)} in-progress _partial.json file(s))")
    shard_paths = sorted(shard_paths, key=_shard_index)
    if not shard_paths:
        raise SystemExit("No shard files matched.")

    payloads = [_load(p) for p in shard_paths]
    items: List[dict] = []
    skipped = 0
    for p in payloads:
        items.extend(p.get("per_example", []))
        skipped += p.get("summary", {}).get("skipped", 0) or 0
    if not items:
        raise SystemExit("Shards contained no per-example results.")

    fam = _family(items)
    if fam == "text_cls":
        from evaluate_text_classification_cot import summarize
        summary = summarize(items, skipped)
    elif fam == "bayesian_teaching":
        from evaluate_bayesian_teaching_cot import summarize
        summary = summarize(items)
    else:
        from evaluate_openestimate import aggregate_metrics
        summary = aggregate_metrics(items)
        summary["valid_rate"] = float(np.mean([r.get("valid", False) for r in items]))
        summary["trunc_rate"] = float(
            np.mean([bool(r.get("truncated")) for r in items]))
        summary["mean_gen_tokens"] = float(
            np.mean([r.get("n_gen_tokens", 0) for r in items]))

    head = dict(payloads[0])
    head.pop("per_example", None)
    head.pop("start_idx", None)
    head.update({
        "n_examples": len(items),
        "n_shards": len(shard_paths),
        "shards": [os.path.basename(p) for p in shard_paths],
        "summary": summary,
        "per_example": items,
    })

    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    with open(output_file, "w") as fh:
        json.dump(head, fh, indent=2, default=str)

    ov = summary.get("overall", summary)
    n = ov.get("n", summary.get("n"))
    if expected and n != expected:
        print(f"  [WARNING] {os.path.basename(output_file)}: merged n={n} but "
              f"expected {expected} — a shard is missing, still running, or "
              f"died. Do not aggregate this into a table yet.")
    print(f"  {os.path.basename(output_file)}: {len(shard_paths)} shards → "
          f"n={n}", end="")
    if "accuracy" in ov:
        print(f"  acc={ov['accuracy']:.4f}  CE={ov['ce_mean']:.4f}  "
              f"ECE={ov['ece']:.4f}")
    else:
        print(f"  MAE={summary['mae']['mean']:.3f}  "
              f"CE={summary['ce_mean']['mean']:.3f}")
    return summary


def _shard_index(path: str) -> int:
    m = re.search(r"_shard(\d+)\.json$", path)
    return int(m.group(1)) if m else -1


def auto_merge(tag: str) -> None:
    """Find every ``<subdir>/<tag>_<name>_shard*.json`` group under results/
    and merge each into ``<subdir>/<tag>_<name>.json``."""
    groups = {}
    for sub in ("text_cls", "bayesian_teaching", "openestimate"):
        for path in glob.glob(f"{RES}/{sub}/{tag}_*_shard*.json"):
            if path.endswith("_partial.json"):
                continue
            base = _SHARD_RE.sub(".json", path)
            groups.setdefault(base, []).append(path)
    if not groups:
        print(f"No shard groups found for tag {tag!r} under {RES}.")
        return
    for base, shards in sorted(groups.items()):
        merge(shards, base, expected=_expected_for(base, tag))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", help="Glob for the shard JSONs.")
    ap.add_argument("--output_file")
    ap.add_argument("--auto", action="store_true",
                    help="Merge every shard group found under results/.")
    ap.add_argument("--tag", default="cot_base",
                    help="Run tag to match when using --auto.")
    ap.add_argument("--expect", type=int, default=None,
                    help="Warn if the merged example count differs. Inferred "
                         "from the benchmark name when omitted.")
    args = ap.parse_args()

    if args.auto:
        auto_merge(args.tag)
        return
    if not args.shards or not args.output_file:
        ap.error("Provide --shards and --output_file, or use --auto.")
    merge(glob.glob(args.shards), args.output_file,
          expected=args.expect or _expected_for(args.output_file, args.tag))


if __name__ == "__main__":
    main()
