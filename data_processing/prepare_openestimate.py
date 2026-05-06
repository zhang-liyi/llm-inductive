"""
Prepare OpenEstimate benchmark data for testing with the probabilistic reasoning pipeline.

Downloads OpenEstimate variable files from GitHub and converts them to the format
expected by ProbabilisticReasoningDataset:
    {
        "input":    <prompt string>,
        "output":   <integer 0-100 as string>,
        "bins":     [[...101 float probabilities...]],
        "metadata": {...}
    }

Scale conventions
-----------------
Beta variables (proportions 0-1):
    Mapped directly to 0-100 percentage scale.
    Ground-truth bins: N(mean*100, (se*100)^2) discretised to 101 integer buckets.

Normal variables (continuous measurements):
    Mapped to 0-100 with a scale anchored to each base variable.
    The raw range is [base_mean - 3σ, base_mean + 3σ], but the lower bound is
    clamped to 0 for inherently non-negative quantities (salary, funding, employees,
    blood analyte concentrations):
        lo = max(0, base_mean - 3σ),  hi = base_mean + 3σ
        centre = (lo + hi) / 2,       full_range = hi - lo
        normalised = 50 + (value - centre) / full_range * 100
    Ground-truth bins: N(normalised_mean, normalised_se^2).
"""

import json
import os
import urllib.request

import numpy as np
from scipy.stats import norm


# ── GitHub raw URLs ────────────────────────────────────────────────────────────
VARIABLE_URLS = {
    "glassdoor": (
        "https://raw.githubusercontent.com/alanarenda/openestimate"
        "/main/data/variables/glassdoor_variables.json"
    ),
    "nhanes": (
        "https://raw.githubusercontent.com/alanarenda/openestimate"
        "/main/data/variables/nhanes_variables.json"
    ),
    "pitchbook": (
        "https://raw.githubusercontent.com/alanarenda/openestimate"
        "/main/data/variables/pitchbook_variables.json"
    ),
}

RAW_DIR = "openestimate_raw"
OUTPUT_FILE = "openestimate_test.json"


# ── Download helpers ───────────────────────────────────────────────────────────

def download_variables(raw_dir: str = RAW_DIR) -> dict:
    """Download (or load from cache) all three variable JSON files."""
    os.makedirs(raw_dir, exist_ok=True)
    datasets = {}
    for name, url in VARIABLE_URLS.items():
        path = os.path.join(raw_dir, f"{name}_variables.json")
        if not os.path.exists(path):
            print(f"  Downloading {name} variables from GitHub…")
            urllib.request.urlretrieve(url, path)
        with open(path) as f:
            datasets[name] = json.load(f)
        print(f"  {name}: {len(datasets[name])} entries loaded from {path}")
    return datasets


# ── Normalisation helpers ──────────────────────────────────────────────────────

def build_base_scales(variables: dict) -> dict:
    """
    For each *base* (unconditional, normal-type) variable entry, record
    (base_mean, base_std) so every conditional variant can be normalised
    on the same symmetric scale.

    Returns  {base_variable_name: (base_mean, 6*base_std)}
    """
    scales = {}
    for key, entry in variables.items():
        if not isinstance(entry, dict):
            continue
        # Base entries have no 'difficulty' key and empty conditions list.
        if "difficulty" in entry:
            continue
        if entry.get("conditions", []):
            continue
        if entry.get("ground_truth_distribution_type") != "normal":
            continue

        bv = entry.get("base_variable", key)
        mu = entry["mean"]
        sigma = entry["std"]
        if sigma <= 0:
            sigma = abs(mu) * 0.1 or 1.0  # safe fallback
        lo = max(0.0, mu - 3.0 * sigma)
        hi = mu + 3.0 * sigma
        scales[bv] = ((lo + hi) / 2.0, hi - lo)  # (centre, full_range)
    return scales


def normalise_normal(mean: float, se: float, base_mean: float, full_range: float):
    """Map a normal-type value onto the 0-100 scale."""
    norm_mean = 50.0 + (mean - base_mean) / full_range * 100.0
    norm_se   = se / full_range * 100.0
    return norm_mean, norm_se


def normalise_beta(mean: float, se: float):
    """Map a beta-type (proportion) value to 0-100 percentage scale."""
    return mean * 100.0, se * 100.0


# ── Bin construction ───────────────────────────────────────────────────────────

MIN_SE = 0.5   # floor: avoids near-zero sigma in scipy.stats.norm


def make_bins(norm_mean: float, norm_se: float, n: int = 101) -> list:
    """
    Discretise N(norm_mean, norm_se^2) into n integer buckets 0..n-1.

    Each bucket i captures probability mass in [i-0.5, i+0.5],
    with the outermost half-bins treated as [−∞, 0.5) and [n−1.5, +∞).
    The result is renormalised to sum to 1.
    """
    se = max(norm_se, MIN_SE)
    edges = np.arange(-0.5, n, 1.0)          # n+1 edges
    cdf   = norm.cdf(edges, loc=norm_mean, scale=se)
    bins  = np.diff(cdf).tolist()             # length n

    total = sum(bins)
    if total > 1e-12:
        bins = [b / total for b in bins]
    else:
        # Degenerate: put all mass at the nearest valid bucket
        idx = int(round(max(0.0, min(float(n - 1), norm_mean))))
        bins = [0.0] * n
        bins[idx] = 1.0

    return bins


# ── Prompt construction ────────────────────────────────────────────────────────

_UNIT_MAP = {
    "Salary":      "USD",
    "Cm":          "cm",
    "Mgdl":        "mg/dL",
    "Ugdl":        "ug/dL",
    "Ugl":         "ug/L",
    "McgL":        "mcg/L",
    "Kg":          "kg",
    "Raised":      "million USD",
    "Employees":   "employees",
}


def _infer_unit(base_variable: str) -> str:
    for key, unit in _UNIT_MAP.items():
        if key in base_variable:
            return unit
    return ""


def _fmt_number(x: float) -> str:
    """Format a number compactly (no unnecessary trailing zeros)."""
    if abs(x) >= 1000:
        return f"{x:,.0f}"
    if abs(x) >= 10:
        return f"{x:.1f}"
    return f"{x:.3g}"


def build_prompt(variable_desc: str, dist_type: str, base_variable: str,
                 base_mean: float, full_range: float) -> str:
    """Construct the model input prompt.

    Asks for two integers on the same 0-100 scale:
      1. Mean estimate
      2. Standard deviation (spread) estimate
    """
    if dist_type == "beta":
        instruction = (
            "Answer the query and return two integers each wrapped in < and >, "
            "separated by a space. For example, <mean> <std>. "
            "The first integer is your mean estimate (0 = 0%, 100 = 100%). "
            "The second integer is your standard deviation estimate on the same scale."
        )
        scale_note = "Express both values on a percentage scale (0–100)."
    else:
        unit = _infer_unit(base_variable)
        lo  = base_mean - full_range / 2.0
        hi  = base_mean + full_range / 2.0
        mid = base_mean
        unit_str = f" {unit}" if unit else ""
        instruction = (
            "Answer the query and return two integers each wrapped in < and >, "
            "separated by a space. For example, <mean> <std>. "
            "The first integer is your mean estimate. "
            "The second integer is your standard deviation estimate on the same scale."
        )
        scale_note = (
            f"Scale: 0 = {_fmt_number(lo)}{unit_str}, "
            f"50 = {_fmt_number(mid)}{unit_str}, "
            f"100 = {_fmt_number(hi)}{unit_str}."
        )

    return (
        f"{instruction}\n\n"
        f"Here is the query:\n\n"
        f"<START_QUERY>\n{variable_desc}\n<END_QUERY>\n\n"
        f"{scale_note}"
    )


# ── Per-dataset processing ─────────────────────────────────────────────────────

def process_dataset(name: str, variables: dict) -> list:
    """Convert all variable entries in one dataset to example dicts."""
    scales = build_base_scales(variables)
    print(f"    {name}: {len(scales)} base-variable scales found.")

    examples = []
    skipped  = 0

    for key, entry in variables.items():
        if not isinstance(entry, dict):
            continue

        mean  = entry.get("mean")
        se    = entry.get("se")
        desc  = entry.get("variable", "")
        dtype = entry.get("ground_truth_distribution_type", "normal")
        bv    = entry.get("base_variable", key)

        if None in (mean, se) or not desc:
            skipped += 1
            continue

        # ── normalise ──────────────────────────────────────────────────────────
        original_std = entry.get("std") or 0.0

        if dtype == "beta":
            norm_mean, norm_se = normalise_beta(mean, se)
            norm_std = original_std * 100.0          # proportion → percentage scale
            base_mean_for_prompt  = 0.5
            full_range_for_prompt = 1.0
        else:
            if bv not in scales:
                sigma_fallback = original_std or abs(mean) * 0.1 or 1.0
                lo_fb = max(0.0, mean - 3.0 * sigma_fallback)
                hi_fb = mean + 3.0 * sigma_fallback
                scales[bv] = ((lo_fb + hi_fb) / 2.0, hi_fb - lo_fb)
            base_mean_for_prompt, full_range_for_prompt = scales[bv]
            norm_mean, norm_se = normalise_normal(
                mean, se,
                base_mean_for_prompt,
                full_range_for_prompt,
            )
            # Normalise population std onto the same 0-100 scale as the mean
            norm_std = (original_std / full_range_for_prompt * 100.0
                        if full_range_for_prompt > 0 else 0.0)

        mean_int = int(round(max(0.0, min(100.0, norm_mean))))
        std_int  = int(round(max(0.0, min(100.0, norm_std))))
        bins = make_bins(norm_mean, norm_se)

        prompt = build_prompt(
            desc, dtype, bv,
            base_mean_for_prompt, full_range_for_prompt,
        )

        examples.append({
            "input":  prompt,
            "output": f"<{mean_int}> <{std_int}>",  # two angle-bracket-wrapped integers
            "bins":   [bins],
            "metadata": {
                "dataset":           name,
                "variable_key":      key,
                "base_variable":     bv,
                "distribution_type": dtype,
                "difficulty":        entry.get("difficulty", "base"),
                "conditions":        entry.get("conditions", []),
                "nat_langs":         entry.get("nat_langs", []),
                "original_mean":     mean,
                "original_se":       se,
                "original_std":      original_std,
                "normalised_mean":   norm_mean,
                "normalised_se":     norm_se,
                "normalised_std":    norm_std,   # population std on 0-100 scale
            },
        })

    if skipped:
        print(f"    {name}: skipped {skipped} entries with missing fields.")
    return examples


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=== Preparing OpenEstimate test data ===\n")

    print("Step 1: Loading variable files")
    datasets = download_variables()

    print("\nStep 2: Building examples")
    all_examples = []
    for name, variables in datasets.items():
        print(f"  Processing '{name}'…")
        exs = process_dataset(name, variables)
        print(f"    → {len(exs)} examples")
        all_examples.extend(exs)

    # ── Summary statistics ─────────────────────────────────────────────────────
    print(f"\nTotal examples: {len(all_examples)}")

    by_dataset = {}
    by_dtype   = {}
    by_diff    = {}
    for ex in all_examples:
        m = ex["metadata"]
        by_dataset[m["dataset"]]           = by_dataset.get(m["dataset"], 0) + 1
        by_dtype[m["distribution_type"]]   = by_dtype.get(m["distribution_type"], 0) + 1
        by_diff[m["difficulty"]]           = by_diff.get(m["difficulty"], 0) + 1

    print(f"  By dataset:           {by_dataset}")
    print(f"  By dist. type:        {by_dtype}")
    print(f"  By difficulty:        {by_diff}")

    # Sanity-check a random example
    import random
    rng = random.Random(42)
    sample = rng.choice(all_examples)
    print(f"\nSample example ({sample['metadata']['dataset']} / "
          f"{sample['metadata']['difficulty']}):")
    print(f"  input (truncated): {sample['input'][:200]!r}")
    print(f"  output: {sample['output']}")
    print(f"  bins sum: {sum(sample['bins'][0]):.6f}")
    peak = sample['bins'][0].index(max(sample['bins'][0]))
    print(f"  bins peak at: {peak}")

    # ── Write output ───────────────────────────────────────────────────────────
    print(f"\nStep 3: Writing {OUTPUT_FILE}")
    with open(OUTPUT_FILE, "w") as f:
        json.dump(all_examples, f, indent=2)
    print(f"Done. Saved {len(all_examples)} examples to {OUTPUT_FILE}.")


if __name__ == "__main__":
    main()
