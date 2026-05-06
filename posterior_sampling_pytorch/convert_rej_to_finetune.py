"""
Convert rejection-sampling inference results from inference_results/ into
finetuning data format, mirroring convert_mcmc_to_finetune.py.

Reads result-gemini-REJ-*.json files, groups them by category, and writes
motif-split train/val/test JSON datasets to ../torchtune/data/pyro-rej/.

Usage
-----
    python convert_rej_to_finetune.py                      # all four categories
    python convert_rej_to_finetune.py --mode healthcare    # single category
"""

import argparse
import json
import os
import re
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
INFERENCE_DIR = os.path.join(HERE, "inference_results")
SCENARIO_DIR = os.path.join(HERE, "scenarios")
OUTPUT_DIR = os.path.join(HERE, "..", "torchtune", "data", "pyro-rej")

PROMPT_PREFIX = (
    "Answer the query in the scenario and return only an integer. "
    "Use 0-100 scale. For a query on individual rank or performance, "
    "a higher number means more strength (e.g. 100 is stronger than 1). "
    "For a query on which team wins, a smaller number means the first team more likely wins."
    "\n\nHere is the scenario:\n\n"
)

# Category → (output base name, filename filter predicate on a REJ result filename).
# The filter receives the basename (no leading dir). It is run only on files that
# already start with "result-gemini-REJ-".
CATEGORIES = {
    "sports":         ("pytorch_rej_sports",
                       lambda f: f.startswith("result-gemini-REJ-P-")),
    "sports_diverse": ("pytorch_rej_sports_diverse",
                       lambda f: f.startswith("result-gemini-REJ-diverse-")),
    "healthcare":     ("pytorch_rej_healthcare",
                       lambda f: f.startswith("result-gemini-REJ-healthcare-")),
    "general":        ("pytorch_rej_general",
                       lambda f: f.startswith("result-gemini-REJ-general-")),
}


def is_who_wins_query(samples):
    """Samples in 0-1 range are who-wins probabilities; 0-100 range are strengths."""
    return max(samples) <= 1.0


def samples_to_bins_and_output(samples):
    """Convert posterior samples to a 101-element histogram over 0..100 and a mean.

    Returns (None, None) if the samples shape isn't a flat list of scalars (or
    a list of 1-element lists). Some malformed result files have shape
    (N, K, 1) with K > 1 — those are skipped upstream.
    """
    if samples and isinstance(samples[0], list) and all(len(s) == 1 for s in samples):
        samples = [s[0] for s in samples]
    if not samples or not isinstance(samples[0], (int, float)):
        return None, None
    if is_who_wins_query(samples):
        scaled = [s * 100 for s in samples]
    else:
        scaled = list(samples)

    int_samples = [max(0, min(100, round(s))) for s in scaled]
    counts = Counter(int_samples)
    total = len(int_samples)
    bins = [counts.get(i, 0) / total for i in range(101)]

    mean_val = sum(scaled) / len(scaled)
    output = str(round(mean_val))
    return bins, output


def parse_scenario_file(path):
    """Parse a scenario .txt file into (background, conditions, [query_texts])."""
    with open(path) as f:
        text = f.read()

    bg_match = re.search(r'BACKGROUND\n(.*?)(?=CONDITIONS)', text, re.DOTALL)
    cond_match = re.search(r'CONDITIONS\n(.*?)(?=QUERIES)', text, re.DOTALL)

    if bg_match and cond_match:
        q_match = re.search(r'QUERIES\n(.*?)(?=<END_SCENARIO>)', text, re.DOTALL)
        if not q_match:
            return None, None, None
        background = bg_match.group(1).strip()
        conditions = cond_match.group(1).strip()
        queries_block = q_match.group(1)
    else:
        labeled = list(re.finditer(r'\n(?:Query|Question)\s*1\s*[.:]', text))
        numbered = list(re.finditer(r'\n\s*1\.', text))
        candidates = labeled or numbered
        first_query = candidates[-1] if candidates else None
        if first_query is None:
            return None, None, None
        context = re.sub(r'^<START_SCENARIO>\s*', '', text[:first_query.start()]).strip()
        background = context
        conditions = None
        queries_block = text[first_query.start():]

    queries = re.findall(r'(?:Query|Question)\s*\d+\s*[.:]\s*(.+)', queries_block)
    if not queries:
        queries = re.findall(r'^\s*\d+\.\s*(.+)', queries_block, re.MULTILINE)
    if not queries:
        return None, None, None
    return background, conditions, queries


def build_input(background, conditions, query_text):
    if conditions is not None:
        context_block = (
            f"BACKGROUND\n{background}\n\n"
            f"CONDITIONS\n{conditions}\n\n"
        )
    else:
        context_block = f"{background}\n\n"

    scenario_block = (
        "<START_SCENARIO>\n"
        + context_block
        + f"QUERIES\nQuery: {query_text}\n"
        "<END_SCENARIO>"
    )
    return PROMPT_PREFIX + scenario_block


def find_scenario_file(result_filename):
    """
    result-gemini-REJ-<rest>-YYYY-MM-DD-HHMMSS.json
      → scenarios/gemini-<rest>.txt
    """
    base = result_filename.replace(".json", "")
    m = re.match(r'^result-gemini-REJ-(.+)-\d{4}-\d{2}-\d{2}-\d{6}$', base)
    if not m:
        return None
    path = os.path.join(SCENARIO_DIR, f"gemini-{m.group(1)}.txt")
    return path if os.path.exists(path) else None


def extract_motif_params(result_filename):
    """Extract (P, C, R, N, seed) — returns None if not a motif-style filename."""
    m = re.search(r'P-(\d+)-C-(\d+)-R-(\d+)-N-(\d+)-.*?(\d+)-\d{4}-\d{2}-\d{2}-\d{6}', result_filename)
    if m:
        return (int(m.group(1)), int(m.group(2)), int(m.group(3)),
                int(m.group(4)), int(m.group(5)))
    return None


def split_dataset(dataset, output_base_path, seed=42):
    """Motif-based split: holdout P=1,C=1,R=0 and P=2,C=1,R=0 for val/test."""
    np.random.seed(seed)
    holdout_configs = {(1, 1, 0), (2, 1, 0)}
    train_data, holdout_data = [], []
    for sample in dataset:
        params = extract_motif_params(sample.get('_result_filename', ''))
        if params and (params[0], params[1], params[2]) in holdout_configs:
            holdout_data.append(sample)
        else:
            train_data.append(sample)

    np.random.shuffle(holdout_data)
    mid = len(holdout_data) // 2
    val_data = holdout_data[:mid]
    test_data = holdout_data[mid:]
    np.random.shuffle(train_data)

    splits = [('train', train_data), ('val', val_data), ('test', test_data)]
    for name, data in splits:
        clean = [{k: v for k, v in s.items() if k != '_result_filename'} for s in data]
        path = f"{output_base_path}_{name}.json"
        with open(path, 'w') as f:
            json.dump(clean, f, indent=2)
        print(f"  {name}: {len(clean)} samples → {path}")


def build_category_dataset(category, output_base, filter_fn, all_rej_files):
    """Collect datapoints for a single category and write its train/val/test split."""
    result_files = [f for f in all_rej_files if filter_fn(f)]
    print(f"\n[{category}] {len(result_files)} inference files")

    dataset = []
    n_no_scenario = 0
    n_parse_failed = 0
    n_processed = 0

    for result_filename in sorted(result_files):
        scenario_path = find_scenario_file(result_filename)
        if scenario_path is None:
            n_no_scenario += 1
            continue

        with open(os.path.join(INFERENCE_DIR, result_filename)) as f:
            result_data = json.load(f)

        background, conditions, query_texts = parse_scenario_file(scenario_path)
        if query_texts is None:
            n_parse_failed += 1
            continue

        for i, (query_key, query_info) in enumerate(result_data.items()):
            if i >= len(query_texts):
                continue
            samples = query_info.get("samples")
            if not samples:
                continue
            bins, output = samples_to_bins_and_output(samples)
            if bins is None:
                continue
            input_str = build_input(background, conditions, query_texts[i])
            dataset.append({
                "input": input_str,
                "output": output,
                "bins": [bins],
                "_result_filename": result_filename,
            })
        n_processed += 1

    output_base_path = os.path.join(OUTPUT_DIR, output_base)
    split_dataset(dataset, output_base_path)

    print(f"  processed={n_processed} no_scenario={n_no_scenario} "
          f"parse_failed={n_parse_failed} datapoints={len(dataset)}")


def main():
    parser = argparse.ArgumentParser(description='Convert REJ inference results to fine-tuning datasets.')
    parser.add_argument('--mode', default='all',
                        choices=['all'] + list(CATEGORIES.keys()),
                        help='Category to process (default: all four).')
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_rej_files = [
        f for f in os.listdir(INFERENCE_DIR)
        if f.startswith("result-gemini-REJ-") and f.endswith(".json")
        and os.path.isfile(os.path.join(INFERENCE_DIR, f))
    ]
    print(f"Found {len(all_rej_files)} REJ inference files in {INFERENCE_DIR}")

    if args.mode == 'all':
        to_run = list(CATEGORIES.items())
    else:
        to_run = [(args.mode, CATEGORIES[args.mode])]

    for category, (output_base, filter_fn) in to_run:
        build_category_dataset(category, output_base, filter_fn, all_rej_files)


if __name__ == "__main__":
    main()
