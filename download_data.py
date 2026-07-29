"""
download_data.py

One-shot data setup. Run this once, before either eval script:

    python download_data.py

Everything lands in ``data_root/`` next to this file, and both
``run_morebench.sh`` and ``run_cot_sc.sh`` read from there by default — there
are no data paths to fill in anywhere.

What it fetches (~70 MB total):

    data_root/morebench/        MoReBench, converted to binary A/B items
    data_root/hg_cache/         the 5 multiple-choice validation splits

Bayesian Teaching and OpenEstimate data already ship in ``data_processing/``,
so nothing is downloaded for those; this script only checks they are present.

Model weights are NOT downloaded here — the base model and the fine-tuned
checkpoints are yours to place. See the guide for where.

Needs internet, so run it on a login node. It is safe to re-run: anything
already present is left alone.
"""

import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA_ROOT = HERE / "data_root"

# (local name, HF repo, config, expected n). The local names are what
# evaluate_text_classification.py looks for under hg_cache/.
MCQ_SPLITS = [
    ("mmlu",           "cais/mmlu",        "all",                 1531),
    ("truthfulqa_mc",  "truthful_qa",      "multiple_choice",      817),
    ("hellaswag",      "Rowan/hellaswag",  None,                 10042),
    ("winogrande",     "allenai/winogrande", "winogrande_debiased", 1267),
    ("arc_challenge",  "allenai/ai2_arc",  "ARC-Challenge",        299),
]

SHIPPED = [
    ("Bayesian Teaching", "data_processing/bayesian_teaching_test_base.jsonl"),
    ("OpenEstimate",      "data_processing/openestimate_test.json"),
]


def fetch_mcq(force: bool) -> int:
    from datasets import load_dataset, load_from_disk

    cache = DATA_ROOT / "hg_cache"
    cache.mkdir(parents=True, exist_ok=True)
    failures = 0
    for name, repo, config, expected in MCQ_SPLITS:
        dest = cache / f"{name}_validation_disk"
        if dest.exists() and not force:
            n = len(load_from_disk(str(dest)))
            print(f"  cached   {name:15s} n={n}")
            continue
        print(f"  fetching {name:15s} from {repo}"
              f"{'' if config is None else ' [' + config + ']'} ...", flush=True)
        try:
            ds = (load_dataset(repo, config, split="validation") if config
                  else load_dataset(repo, split="validation"))
        except Exception as e:
            print(f"  FAILED   {name:15s} {type(e).__name__}: {str(e)[:120]}")
            failures += 1
            continue
        if len(ds) != expected:
            print(f"  WARNING  {name}: got n={len(ds)}, expected {expected}. "
                  f"The upstream dataset may have changed; results will not be "
                  f"comparable to the published rows.")
        ds.save_to_disk(str(dest))
        print(f"  saved    {name:15s} n={len(ds)} -> {dest.relative_to(HERE)}")
    return failures


def fetch_morebench(force: bool) -> int:
    sys.path.insert(0, str(HERE / "data_processing"))
    import prepare_morebench as pm

    dest = DATA_ROOT / "morebench" / "morebench_binary.json"
    if dest.exists() and not force:
        print(f"  cached   morebench       -> {dest.relative_to(HERE)}")
        return 0

    import pandas as pd

    pm.THEORY_DEFINITIONS = pm._load_theory_definitions()
    out_dir = DATA_ROOT / "morebench"
    try:
        csvs = pm.fetch_csvs(out_dir / "raw", offline=False)
    except Exception as e:
        print(f"  FAILED   morebench       {type(e).__name__}: {str(e)[:120]}")
        return 1
    for name, csv_key, task_field, with_theory, fname in [
        ("morebench", "public", "DILEMMA_SOURCE", False, "morebench_binary.json"),
        ("morebench_theory", "theory", "THEORY", True,
         "morebench_theory_binary.json"),
    ]:
        df = pd.read_csv(csvs[csv_key])
        items = pm.build_items(df, task_field, with_theory, balanced=True)
        pm.report(name, items)
        import json
        with open(out_dir / fname, "w") as fh:
            json.dump({"dataset": name, "source_csv": csvs[csv_key].name,
                       "balanced": True, "n_items": len(items),
                       "positive_sampling_seed": pm.POSITIVE_SAMPLING_SEED,
                       "items": items}, fh, indent=1)
        print(f"  saved    {name:15s} n={len(items)}")
    return 0


def check_shipped() -> int:
    missing = 0
    for label, rel in SHIPPED:
        p = HERE / rel
        if p.exists():
            print(f"  present  {label:15s} {p.stat().st_size / 1e6:.1f} MB")
        else:
            print(f"  MISSING  {label:15s} expected at {rel}")
            missing += 1
    return missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="Re-download even where the data is already present.")
    args = ap.parse_args()

    print(f"Data root: {DATA_ROOT}\n")
    print("Multiple-choice validation splits")
    bad = fetch_mcq(args.force)
    print("\nMoReBench")
    bad += fetch_morebench(args.force)
    print("\nAlready in the repo")
    bad += check_shipped()

    if bad:
        print(f"\n{bad} item(s) failed. Re-run on a machine with internet "
              f"access, or fetch them by hand into {DATA_ROOT}.")
        sys.exit(1)

    print(f"\nAll data ready under {DATA_ROOT.relative_to(HERE.parent)}/")
    print("Next: put your models in place, then run the eval script.")


if __name__ == "__main__":
    main()
