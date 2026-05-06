"""
Build combined REJ datasets for the two SFT training groups.

Produces, under ../torchtune/data/pyro-rej/:
    pytorch_rej_sports_plus_diverse_train.json  (sports + sports_diverse train)
    pytorch_rej_all_train.json                  (all four categories train)
    pytorch_rej_eval_val.json                   (<=200/category, interleaved)
    pytorch_rej_eval_test.json                  (<=200/category, interleaved)

The interleaved eval files let early batches (e.g. when PYRO_VAL_BATCHES caps
the eval) stay stratified across the four categories.

Invoke:  python build_rej_combined.py
"""

import json
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "torchtune", "data", "pyro-rej")

CATEGORIES = ["sports", "sports_diverse", "healthcare", "general"]
MAX_PER_CAT = 200
SHUFFLE_SEED = 42  # deterministic subsampling


def load(category, split):
    path = os.path.join(DATA_DIR, f"pytorch_rej_{category}_{split}.json")
    with open(path) as f:
        return json.load(f)


def take_subset(data, k, rng):
    """Deterministic subset of up to k items."""
    if len(data) <= k:
        return list(data)
    idx = sorted(rng.sample(range(len(data)), k))
    return [data[i] for i in idx]


def interleave(lists):
    """Round-robin interleave of equal- or unequal-length lists."""
    out = []
    i = 0
    while True:
        added = False
        for lst in lists:
            if i < len(lst):
                out.append(lst[i])
                added = True
        if not added:
            break
        i += 1
    return out


def main():
    rng = random.Random(SHUFFLE_SEED)

    # ── Combined train sets (full, not subsampled) ──────────────────────────
    sports_train = load("sports", "train")
    sports_div_train = load("sports_diverse", "train")
    healthcare_train = load("healthcare", "train")
    general_train = load("general", "train")

    sports_plus_diverse = sports_train + sports_div_train
    all_domains = sports_train + sports_div_train + healthcare_train + general_train

    path_a = os.path.join(DATA_DIR, "pytorch_rej_sports_plus_diverse_train.json")
    with open(path_a, "w") as f:
        json.dump(sports_plus_diverse, f, indent=2)
    print(f"{path_a}  ({len(sports_plus_diverse)} examples)")

    path_b = os.path.join(DATA_DIR, "pytorch_rej_all_train.json")
    with open(path_b, "w") as f:
        json.dump(all_domains, f, indent=2)
    print(f"{path_b}  ({len(all_domains)} examples)")

    # ── Combined eval val / test: <=200 per category, interleaved ───────────
    for split, out_name in [("val", "pytorch_rej_eval_val.json"),
                            ("test", "pytorch_rej_eval_test.json")]:
        per_cat = []
        sizes = {}
        for cat in CATEGORIES:
            data = load(cat, split)
            subset = take_subset(data, MAX_PER_CAT, rng)
            per_cat.append(subset)
            sizes[cat] = len(subset)
        combined = interleave(per_cat)
        out_path = os.path.join(DATA_DIR, out_name)
        with open(out_path, "w") as f:
            json.dump(combined, f, indent=2)
        sizes_str = "  ".join(f"{k}={v}" for k, v in sizes.items())
        print(f"{out_path}  ({len(combined)} examples)  [{sizes_str}]")


if __name__ == "__main__":
    main()
