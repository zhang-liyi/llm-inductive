"""
Convert MCMC posterior inference results from inference_results/ into
finetuning data format matching ../torchtune/data/data_webppl/probabilistic_reasoning.json.

One datapoint = one scenario x one query. Each scenario is duplicated for each query.

Output: ../torchtune/data/pyro/pytorch_mcmc_dataset.json
"""

import argparse
import json
import os
import re
from collections import Counter

import numpy as np

INFERENCE_DIR = os.path.join(os.path.dirname(__file__), "inference_results")
SCENARIO_DIR = os.path.join(os.path.dirname(__file__), "scenarios")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "torchtune", "data", "pyro", "pytorch_mcmc_dataset.json")
OUTPUT_PATH_HEALTHCARE = os.path.join(os.path.dirname(__file__), "..", "torchtune", "data", "pyro", "pytorch_mcmc_healthcare.json")
OUTPUT_PATH_DIVERSE_SPORTS = os.path.join(os.path.dirname(__file__), "..", "torchtune", "data", "pyro", "pytorch_mcmc_diverse_sports.json")
OUTPUT_PATH_GENERAL = os.path.join(os.path.dirname(__file__), "..", "torchtune", "data", "pyro", "pytorch_mcmc_general.json")

PROMPT_PREFIX = (
    "Answer the query in the scenario and return only an integer. "
    "Use 0-100 scale. For a query on individual rank or performance, "
    "a higher number means more strength (e.g. 100 is stronger than 1). "
    "For a query on which team wins, a smaller number means the first team more likely wins."
    "\n\nHere is the scenario:\n\n"
)


def is_who_wins_query(samples):
    """Detect if samples are in 0-1 range (who-wins probability) vs 0-100 range."""
    return max(samples) <= 1.0


def samples_to_bins_and_output(samples):
    """
    Convert MCMC samples to a 101-element probability distribution over integers 0-100,
    and compute the rounded-mean point estimate as the output string.

    For strength/effort queries: samples are already in 0-100 range.
    For who-wins queries: samples are probabilities in 0-1; scale to 0-100.
    """
    if is_who_wins_query(samples):
        scaled = [s * 100 for s in samples]
    else:
        scaled = list(samples)

    # Round to nearest integer and clamp to [0, 100]
    int_samples = [max(0, min(100, round(s))) for s in scaled]

    # Build normalized histogram
    counts = Counter(int_samples)
    total = len(int_samples)
    bins = [counts.get(i, 0) / total for i in range(101)]

    # Point estimate: rounded mean
    mean_val = sum(scaled) / len(scaled)
    output = str(round(mean_val))

    return bins, output


def parse_scenario_file(path):
    """
    Parse a scenario .txt file and return (background, conditions, list_of_query_texts),
    or (None, None, None) if the file cannot be parsed.

    Handles two formats:
      Structured: explicit BACKGROUND / CONDITIONS / QUERIES sections
      Narrative:  free-form prose with queries at the end in one of three styles:
                    "Query N:"  / "Question N:"  /  plain numbered "N."
    """
    with open(path) as f:
        text = f.read()

    bg_match = re.search(r'BACKGROUND\n(.*?)(?=CONDITIONS)', text, re.DOTALL)
    cond_match = re.search(r'CONDITIONS\n(.*?)(?=QUERIES)', text, re.DOTALL)

    if bg_match and cond_match:
        # Structured format
        q_match = re.search(r'QUERIES\n(.*?)(?=<END_SCENARIO>)', text, re.DOTALL)
        if not q_match:
            return None, None, None
        background = bg_match.group(1).strip()
        conditions = cond_match.group(1).strip()
        queries_block = q_match.group(1)
    else:
        # Narrative format: find where queries start
        # Accepts: "Query 1:" / "Question 1:" / plain "1."
        labeled = list(re.finditer(r'\n(?:Query|Question)\s*1\s*[.:]', text))
        numbered = list(re.finditer(r'\n\s*1\.', text))
        candidates = labeled or numbered  # prefer labeled; fall back to numbered
        first_query = candidates[-1] if candidates else None  # use last match for safety
        if first_query is None:
            return None, None, None
        context = re.sub(r'^<START_SCENARIO>\s*', '', text[:first_query.start()]).strip()
        background = context
        conditions = None
        queries_block = text[first_query.start():]

    # Extract query texts — try labeled style first, then plain numbered list
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
    Map a result filename to its corresponding scenario .txt file.

    Default (gemini-NUTS-diverse):
        result-gemini-NUTS-diverse-P-0-C-0-R-0-N-0-0-2026-03-27-073809.json
        → scenarios/gemini-diverse-P-0-C-0-R-0-N-0-0.txt

    Fallback (benchmarks):
        result-biathlon-pytorch.json → scenarios/benchmarks/sc-biathlon.txt
    """
    base = result_filename.replace(".json", "")

    # Default: gemini-NUTS-* with timestamp suffix
    m = re.match(r'^result-gemini-NUTS-(.+)-\d{4}-\d{2}-\d{2}-\d{6}$', base)
    if m:
        scenario_name = f"gemini-{m.group(1)}.txt"
        path = os.path.join(SCENARIO_DIR, scenario_name)
        if os.path.exists(path):
            return path

    # Fallback: benchmark files (result-*-pytorch.json → sc-*.txt)
    name = base.replace("result-", "").replace("-pytorch", "")
    name_no_hyphens = name.replace("-", "")
    benchmark_dir = os.path.join(SCENARIO_DIR, "benchmarks")
    for fname in os.listdir(benchmark_dir):
        if not fname.endswith(".txt"):
            continue
        sc_base = fname[len("sc-"):-len(".txt")]
        if sc_base == name or sc_base == name_no_hyphens:
            return os.path.join(benchmark_dir, fname)

    return None


def is_motif_filename(filename):
    """Return True if the filename follows the motif-based naming convention
    (result-gemini-NUTS-...-P-C-R-N-seed-YYYY-MM-DD-HHMMSS.json), as opposed
    to benchmark-style names like result-biathlon-pytorch.json."""
    return bool(re.search(r'P-\d+-C-\d+-R-\d+-N-\d+.*\d{4}-\d{2}-\d{2}-\d{6}', filename))


def extract_motif_params(result_filename):
    """
    Extract motif parameters (P, C, R, N, seed) from a result filename.
    Returns (P, C, R, N, seed) tuple or None if not a motif-based file.
    """
    m = re.search(r'P-(\d+)-C-(\d+)-R-(\d+)-N-(\d+)-.*?(\d+)-\d{4}-\d{2}-\d{2}-\d{6}', result_filename)
    if m:
        return (int(m.group(1)), int(m.group(2)), int(m.group(3)),
                int(m.group(4)), int(m.group(5)))
    return None


def split_dataset(dataset, output_path, split_mode='random', seed=42):
    """
    Split dataset into train/val/test and write to disk.

    Modes:
      random — 80/10/10 split
      motif  — holdout P=1,C=1,R=0 and P=2,C=1,R=0 for val/test;
               all other motif combos go to train
    """
    np.random.seed(seed)
    base = output_path.replace('.json', '')

    if split_mode == 'motif':
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

        print(f"Motif-based split: holdout P=1,C=1,R=0 and P=2,C=1,R=0 for val/test")
    else:
        indices = np.random.permutation(len(dataset))
        shuffled = [dataset[i] for i in indices]
        n = len(shuffled)
        train_end = int(n * 0.8)
        val_end = train_end + int(n * 0.1)
        train_data = shuffled[:train_end]
        val_data = shuffled[train_end:val_end]
        test_data = shuffled[val_end:]

    splits = [('train', train_data), ('val', val_data), ('test', test_data)]
    for name, data in splits:
        # Strip internal tracking key before saving
        clean = [{k: v for k, v in s.items() if k != '_result_filename'} for s in data]
        path = f"{base}_{name}.json"
        with open(path, 'w') as f:
            json.dump(clean, f, indent=2)
        print(f"  {name}: {len(clean)} samples → {path}")


def main():
    parser = argparse.ArgumentParser(description='Convert Pyro/MCMC inference results to fine-tuning dataset')
    parser.add_argument('--output', '-o', default=None,
                        help='Output JSON file path (default depends on --mode)')
    parser.add_argument('--split', action='store_true',
                        help='Split into train/val/test sets after generation')
    parser.add_argument('--split-mode', default='random', choices=['random', 'motif'],
                        help='Split strategy: random (80/10/10) or motif (holdout P=1,C=1,R=0 and P=2,C=1,R=0)')
    parser.add_argument('--mode', default='all', choices=['all', 'healthcare', 'diverse', 'sports', 'general'],
                        help=(
                            'Filter mode: '
                            '"all" processes all files (default); '
                            '"sports" processes original sports + diverse (no healthcare, no general); '
                            '"general" processes only files with "general" in the filename; '
                            '"healthcare" processes only files with "healthcare" in the filename; '
                            '"diverse" processes only files with "diverse" in the filename'
                        ))
    args = parser.parse_args()

    # Resolve default output path based on mode
    _output_defaults = {
        'sports': OUTPUT_PATH,
        'healthcare': OUTPUT_PATH_HEALTHCARE,
        'diverse': OUTPUT_PATH_DIVERSE_SPORTS,
        'general': OUTPUT_PATH_GENERAL,
        'all': OUTPUT_PATH,
    }
    output_path = args.output if args.output is not None else _output_defaults[args.mode]

    # Collect only JSON files directly in inference_results/ (no subdirectories)
    all_result_files = [
        f for f in os.listdir(INFERENCE_DIR)
        if f.endswith(".json") and os.path.isfile(os.path.join(INFERENCE_DIR, f))
    ]

    if args.mode == 'healthcare':
        result_files = [f for f in all_result_files if 'healthcare' in f]
    elif args.mode == 'general':
        result_files = [f for f in all_result_files if 'general' in f and 'healthcare' not in f]
    elif args.mode == 'diverse':
        result_files = [f for f in all_result_files if 'diverse' in f and 'healthcare' not in f]
    elif args.mode == 'sports':
        # Original sports + diverse, exclude healthcare and general
        result_files = [f for f in all_result_files
                        if 'healthcare' not in f and 'general' not in f and is_motif_filename(f)]
    else:
        result_files = all_result_files
    print(f"Mode: {args.mode} — {len(result_files)} matching files")

    dataset = []
    n_no_scenario = 0
    n_parse_failed = 0
    n_processed = 0

    for result_filename in sorted(result_files):
        result_path = os.path.join(INFERENCE_DIR, result_filename)
        scenario_path = find_scenario_file(result_filename)

        if scenario_path is None:
            print(f"WARNING: No scenario file found for {result_filename}, skipping.")
            n_no_scenario += 1
            continue

        with open(result_path) as f:
            result_data = json.load(f)

        background, conditions, query_texts = parse_scenario_file(scenario_path)

        if query_texts is None:
            print(f"WARNING: Could not parse {os.path.basename(scenario_path)}, skipping.")
            n_parse_failed += 1
            continue

        print(f"Processing {result_filename} with scenario {os.path.basename(scenario_path)}")

        # result_data keys are "query1", "query2", ...; query_texts is 0-indexed
        added = 0
        for i, (query_key, query_info) in enumerate(result_data.items()):
            if i >= len(query_texts):
                print(f"  WARNING: More result queries than scenario queries at {query_key}, skipping.")
                continue

            if "samples" not in query_info:
                print(f"  WARNING: No samples for {query_key}, skipping.")
                continue

            samples = query_info["samples"]
            query_text = query_texts[i]

            bins, output = samples_to_bins_and_output(samples)
            input_str = build_input(background, conditions, query_text)

            dataset.append({
                "input": input_str,
                "output": output,
                "bins": [bins],
                "_result_filename": result_filename,  # used for motif split, stripped before saving
            })
            added += 1

        print(f"  Added {added} datapoints.")
        n_processed += 1

    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    if args.split:
        split_dataset(dataset, output_path, split_mode=args.split_mode)
    else:
        clean = [{k: v for k, v in s.items() if k != '_result_filename'} for s in dataset]
        with open(output_path, 'w') as f:
            json.dump(clean, f, indent=2)
        print(f"\nDone. Wrote {len(clean)} datapoints to {output_path}")

    print(f"\nSummary:")
    print(f"  Processed:           {n_processed}")
    print(f"  No scenario file:    {n_no_scenario}")
    print(f"  Scenario parse fail: {n_parse_failed}")
    print(f"  Total datapoints:    {len(dataset)}")


if __name__ == "__main__":
    main()
