"""
Generate fine-tuning dataset from forward_sample_single_scenario.py output.

Reads JSON files from single_scenario_samples/ and converts them into the same
input-output JSON format used by generate_finetuning_dataset.py, so the same
torchtune training scripts work without modification.

Each (accepted posterior sample, query) pair becomes its own datapoint.  The
scenario text in the input contains only that one query; the output is a single
integer (the sampled value rounded to the nearest integer).

Output format:
[
    {
        "input":    "<instruction>\n\n<single-query scenario text>",
        "output":   "42",   # single integer string
        "metadata": {...}
    },
    ...
]
One source JSON file with N accepted samples and Q queries → N×Q dataset records
(e.g. 100 000 samples × 4 queries = 400 000 records).
"""

import json
import os
import glob
import re
from typing import Optional

# Reuse the multi-query instruction and split utility from the forward-sampling pipeline.
from generate_finetuning_dataset import INSTRUCTION, split_dataset

# Single-query variant of the instruction.
INSTRUCTION_SINGLE = (
    "Answer the query in the scenario and return only an integer. Use 0-100 scale. "
    "For a query on individual rank or performance, a higher number means more strength "
    "(e.g. 100 is stronger than 1). For a query on which team wins, a smaller number "
    "means the first team more likely wins."
    "\n\nHere is the scenario:\n\n"
)


def split_scenario_queries(scenario_text: str) -> list:
    """Split a multi-query scenario into individual single-query scenario texts.

    Returns a list where element i is the full scenario text containing only
    Query (i+1).
    """
    queries_start = scenario_text.find("QUERIES\n")
    if queries_start == -1:
        raise ValueError("No QUERIES section found in scenario text")

    preamble = scenario_text[:queries_start]  # everything up to (not including) "QUERIES\n"

    end_tag = scenario_text.find("<END_SCENARIO>")
    queries_body = scenario_text[queries_start + len("QUERIES\n"):end_tag].rstrip()

    # Split on "Query N:" boundaries, keeping the delimiter with each part.
    parts = re.split(r"(?=Query \d+:)", queries_body)
    parts = [p.strip() for p in parts if p.strip()]

    return [f"{preamble}QUERIES\n{part}\n<END_SCENARIO>" for part in parts]


# ── Per-file conversion ───────────────────────────────────────────────────────

def process_single_scenario_json(json_path: str) -> list:
    """
    Convert one forward_sample_single_scenario output JSON into a list of
    dataset records — one record per (accepted sample, query) pair.

    Each record targets a single query: the input contains only that query's
    scenario text and the output is a single integer (the sampled value).
    N accepted samples × Q queries → N×Q records.
    """
    with open(json_path) as f:
        data = json.load(f)

    scenario_text = data["scenario_text"]
    raw_answers   = data["raw_answers"]   # {"query1": [v0, v1, ...], ...}
    scenario_name = data["scenario_name"]
    queries       = data.get("queries", [])
    n_accepted    = data["metadata"].get("n_accepted")
    accept_rate   = data["metadata"].get("accept_rate")

    query_keys = sorted(raw_answers.keys())  # query1, query2, ...

    # One scenario text per query (uses split_scenario_queries defined above).
    single_query_texts = split_scenario_queries(scenario_text)

    records = []
    for sample_idx in range(len(raw_answers[query_keys[0]])):
        for q_i, q_key in enumerate(query_keys):
            answer_int = int(round(float(raw_answers[q_key][sample_idx])))
            input_text = INSTRUCTION_SINGLE + single_query_texts[q_i]
            records.append({
                "input":  input_text,
                "output": str(answer_int),
                "metadata": {
                    "source_file":   os.path.basename(json_path),
                    "scenario_name": scenario_name,
                    "sample_idx":    sample_idx,
                    "query_key":     q_key,
                    "n_accepted":    n_accepted,
                    "accept_rate":   accept_rate,
                    "query":         queries[q_i] if q_i < len(queries) else None,
                },
            })
    return records


# ── Dataset generation ────────────────────────────────────────────────────────

def generate_dataset(
    input_dir: str = "single_scenario_samples",
    output_file: str = "../torchtune/data/single_scenario_dataset.json",
    max_samples: Optional[int] = None,
    verbose: bool = True,
) -> int:
    """
    Convert all JSON files in input_dir into a single fine-tuning dataset JSON.

    Args:
        input_dir:    Directory containing *.json files from
                      forward_sample_single_scenario.py.
        output_file:  Path to write the JSON dataset.
        max_samples:  Cap total records (None = all).
        verbose:      Print progress.

    Returns:
        Number of records written.
    """
    json_files = sorted(glob.glob(os.path.join(input_dir, "*.json")))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {input_dir!r}")

    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

    dataset = []
    for path in json_files:
        if verbose:
            print(f"Reading {path} …")
        records = process_single_scenario_json(path)
        if verbose:
            m = records[0]["metadata"]
            n_queries = len(set(r["metadata"]["query_key"] for r in records))
            n_samples = m["n_accepted"]
            print(f"  scenario: {m['scenario_name']}, "
                  f"{n_queries} queries × {n_samples} samples = {len(records)} records")
        dataset.extend(records)
        if max_samples and len(dataset) >= max_samples:
            dataset = dataset[:max_samples]
            break

    with open(output_file, "w") as f:
        json.dump(dataset, f, indent=2)

    if verbose:
        print(f"\nWrote {len(dataset)} records → {output_file}")

    return len(dataset)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate fine-tuning dataset from single-scenario posterior JSONs."
    )
    parser.add_argument(
        "--input-dir", "-i", default="single_scenario_samples",
        help="Directory containing *.json files (default: single_scenario_samples/)"
    )
    parser.add_argument(
        "--output", "-o", default="../torchtune/data/single_scenario_dataset.json",
        help="Output JSON file path"
    )
    parser.add_argument(
        "--max-samples", "-n", type=int, default=None,
        help="Cap total records (default: all)"
    )
    parser.add_argument(
        "--split", action="store_true",
        help="Split into train/val/test after generation"
    )
    parser.add_argument(
        "--train-ratio", type=float, default=0.8,
        help="Training set fraction for random split (default: 0.8)"
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.1,
        help="Validation set fraction for random split (default: 0.1)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for splitting (default: 42)"
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="Suppress progress output"
    )

    args = parser.parse_args()

    n = generate_dataset(
        input_dir=args.input_dir,
        output_file=args.output,
        max_samples=args.max_samples,
        verbose=not args.quiet,
    )

    if args.split and n > 0:
        print()
        split_dataset(
            input_file=args.output,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed,
            split_mode="random",
        )
