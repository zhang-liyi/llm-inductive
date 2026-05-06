"""
Generate fine-tuning dataset from forward sampling data.

Reads JSONL files from samples/ directory and converts them into the same
input-output JSON format used by the posterior_sampling pipeline, so the
same torchtune training scripts work without modification.

Each forward-sampled record has exact ground-truth answers (point estimates
from the generative model), unlike MCMC posteriors which are full distributions.
The `bins` field is filled with a Gaussian centered at the exact answer.

Output format (same as posterior_sampling/generate_finetuning_dataset.py):
[
    {
        "input":    "<instruction>\n\n<scenario text>",
        "output":   "[q1, q2, q3, q4]",          # list of rounded integers
        "bins":     [[p0..p100], ...],             # one 101-length array per query
        "metadata": {...}
    },
    ...
]
"""

import json
import os
import glob
import numpy as np
from pathlib import Path
from typing import Optional


# ── Instruction prompt (matches posterior_sampling pipeline) ─────────────────

INSTRUCTION = (
    "Answer the queries in the scenario and return only a list of integers, "
    "each wrapped in < and >. For example, an output can be: "
    "[<mean1>, <mean2>]. For queries on individual rank, a higher number means "
    "a higher ranking (e.g. 100 means the individual ranks highest in that "
    "criterion; 1 is lowest). For queries on which of the two teams wins, a "
    "smaller number means the first team more likely wins."
    "\n\nHere is the scenario:\n\n"
)


# ── Bin construction ─────────────────────────────────────────────────────────

def make_bins(answer: float, sigma: float = 5.0) -> list:
    """
    Create a 101-length probability distribution over integer bins 0-100.

    If sigma == 0: pure delta mass at round(answer).
    Otherwise: Gaussian centred at answer with std=sigma, clipped to [0,100].
    """
    bins = np.zeros(101)
    idx = int(round(float(answer)))
    idx = max(0, min(100, idx))

    if sigma == 0:
        bins[idx] = 1.0
    else:
        xs = np.arange(101, dtype=float)
        raw = np.exp(-0.5 * ((xs - answer) / sigma) ** 2)
        total = raw.sum()
        bins = raw / total if total > 0 else bins

    return [round(float(v), 6) for v in bins]


# ── Dataset generation ───────────────────────────────────────────────────────

def process_jsonl(
    jsonl_path: str,
    sigma: float = 5.0,
) -> list:
    """Convert one JSONL file into a list of dataset records."""
    records = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)

            scenario = sample["scenario"]
            answers = sample["answers"]
            meta = sample.get("metadata", {})

            # Build input
            input_text = INSTRUCTION + scenario

            # Build output: sorted query keys → list of rounded integers
            query_keys = sorted(answers.keys())  # query1, query2, ...
            answer_ints = [int(round(float(answers[k]))) for k in query_keys]
            output_text = json.dumps(answer_ints)

            # Build bins: one 101-length array per query
            bins = [make_bins(float(answers[k]), sigma=sigma) for k in query_keys]

            records.append({
                "input": input_text,
                "output": output_text,
                "bins": bins,
                "metadata": {
                    "source_file": os.path.basename(jsonl_path),
                    "motifs": meta.get("motifs", {}),
                    "num_queries": len(query_keys),
                    "raw_answers": {k: answers[k] for k in query_keys},
                },
            })

    return records


def generate_dataset(
    samples_dir: str = "samples",
    output_file: str = "../torchtune/data/forward_sampling_dataset.json",
    sigma: float = 5.0,
    max_samples: Optional[int] = None,
    verbose: bool = True,
) -> int:
    """
    Generate fine-tuning dataset from all JSONL files in samples_dir.

    Args:
        samples_dir:  Directory containing *.jsonl forward-sampling files.
        output_file:  Path to write the JSON dataset.
        sigma:        Std-dev for Gaussian bins (0 = delta mass).
        max_samples:  Cap total records (None = all).
        verbose:      Print progress.

    Returns:
        Number of records written.
    """
    jsonl_files = sorted(glob.glob(os.path.join(samples_dir, "*.jsonl")))
    if not jsonl_files:
        raise FileNotFoundError(f"No JSONL files found in {samples_dir!r}")

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

    dataset = []
    for path in jsonl_files:
        if verbose:
            print(f"Reading {path} …")
        records = process_jsonl(path, sigma=sigma)
        dataset.extend(records)
        if verbose:
            print(f"  {len(records)} records")
        if max_samples and len(dataset) >= max_samples:
            break

    if max_samples:
        dataset = dataset[:max_samples]

    with open(output_file, "w") as f:
        json.dump(dataset, f, indent=2)

    if verbose:
        print(f"\nWrote {len(dataset)} records → {output_file}")

    return len(dataset)


# ── Train / val / test split ─────────────────────────────────────────────────

def split_dataset(
    input_file: str,
    train_file: str = None,
    val_file: str = None,
    test_file: str = None,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
    split_mode: str = "random",
):
    """
    Split a generated dataset JSON into train / val / test.

    split_mode options:
        'random'  – shuffle then slice by ratio.
        'motif'   – hold out specific (C, R) combinations for val/test.
                    Val/test: C=1, R=1. Train: everything else.
    """
    with open(input_file) as f:
        dataset = json.load(f)

    rng = np.random.default_rng(seed)

    if split_mode == "motif":
        holdout = lambda m: (m.get("C") == 1 and m.get("R") == 0)
        train_data, hold_data = [], []
        for s in dataset:
            motifs = s.get("metadata", {}).get("motifs", {})
            (hold_data if holdout(motifs) else train_data).append(s)
        rng.shuffle(hold_data)
        mid = len(hold_data) // 2
        val_data, test_data = hold_data[:mid], hold_data[mid:]
        rng.shuffle(train_data)
        print(f"Motif split — holdout: C=1,R=0")
    else:
        indices = rng.permutation(len(dataset))
        dataset = [dataset[i] for i in indices]
        n = len(dataset)
        t = int(n * train_ratio)
        v = t + int(n * val_ratio)
        train_data, val_data, test_data = dataset[:t], dataset[t:v], dataset[v:]

    def _save(data, path):
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  {len(data):>5} samples → {path}")

    base = input_file.replace(".json", "")
    _save(train_data, train_file or f"{base}_train.json")
    _save(val_data,   val_file   or f"{base}_val.json")
    _save(test_data,  test_file  or f"{base}_test.json")


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate fine-tuning dataset from forward-sampling JSONL files."
    )
    parser.add_argument(
        "--samples-dir", default="samples",
        help="Directory containing *.jsonl files (default: samples/)"
    )
    parser.add_argument(
        "--output", "-o", default="../torchtune/data/forward_sampling_dataset.json",
        help="Output JSON file path"
    )
    parser.add_argument(
        "--sigma", type=float, default=5.0,
        help="Gaussian σ for bins (0 = pure delta mass, default: 5.0)"
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
        "--split-mode", default="random", choices=["random", "motif"],
        help="Split strategy: 'random' or 'motif' (default: random)"
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
        samples_dir=args.samples_dir,
        output_file=args.output,
        sigma=args.sigma,
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
            split_mode=args.split_mode,
        )
