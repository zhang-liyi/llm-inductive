"""Convert gemini-direct answer JSONs into per-(scenario, query) torchtune
finetuning entries, with the same motif-based train/val/test split used by
pyro-rej (so direct comparisons across the two targets are clean).

Each input file (gemini_direct/gemini-{P,diverse,healthcare,general}-...json) has
shape:
    {
      "scenario_id": "P-0-C-0-R-0-N-0-Nind-1-Nepi-1-0",   # or diverse-...
      "scenario_path": "scenarios/...",
      "queries":         [str, str, str, str],
      "gemini_prompt":   "...full prompt incl. instruction + scenario...",
      "gemini_raw":      "[<a>, <b>, <c>, <d>]",
      "gemini_answers":  [int, int, int, int]
    }

We emit one entry per query, matching pyro-rej's per-query layout:
    {
      "input":  PROMPT_PREFIX + scenario_block_with_one_query,
      "output": str(gemini_answers[i]),
      "bins":   [[101 floats — one-hot at gemini_answers[i] for compat with the
                  ProbabilisticReasoningDataset class; mean-only training does
                  not actually use bins]]
    }

Categories:
    sports          → gemini-P-...
    sports_diverse  → gemini-diverse-...
    healthcare      → gemini-healthcare-...
    general         → gemini-general-...

Motif-based holdout: scenarios with motif (P=1,C=1,R=0) or (P=2,C=1,R=0) go
to val+test, everything else to train. Matches convert_rej_to_finetune.py.

Usage:
    python convert_to_finetune.py                # all four categories
    python convert_to_finetune.py --mode sports  # one category
"""
import argparse
import json
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR  = HERE
OUTPUT_DIR = os.path.join(HERE, "..", "torchtune", "data", "gemini-direct")

PROMPT_PREFIX = (
    "Answer the query in the scenario and return only an integer. "
    "Use 0-100 scale. For a query on individual rank or performance, "
    "a higher number means more strength (e.g. 100 is stronger than 1). "
    "For a query on which team wins, a smaller number means the first team more likely wins."
    "\n\nHere is the scenario:\n\n"
)

CATEGORIES = {
    # display name → (output base name, filename predicate on basename of *.json)
    "sports":         ("gemini_sports",
                       lambda f: f.startswith("gemini-P-")),
    "sports_diverse": ("gemini_sports_diverse",
                       lambda f: f.startswith("gemini-diverse-")),
    "healthcare":     ("gemini_healthcare",
                       lambda f: f.startswith("gemini-healthcare-")),
    "general":        ("gemini_general",
                       lambda f: f.startswith("gemini-general-")),
}

SCENARIO_OPEN  = "<START_SCENARIO>"
SCENARIO_CLOSE = "<END_SCENARIO>"


def parse_scenario_block(gemini_prompt: str):
    """Return (background, conditions) extracted from the bracket-format
    `gemini_prompt`. `conditions` is None for free-form (diverse) scenarios."""
    # Tolerate trailing whitespace after <START_SCENARIO> and leading whitespace
    # before <END_SCENARIO> (some healthcare/general files have " \n" instead of "\n").
    m = re.search(rf'{SCENARIO_OPEN}\s*\n?(.*?)\s*{SCENARIO_CLOSE}', gemini_prompt, re.DOTALL)
    if not m:
        return None, None
    body = m.group(1)
    # Truncate at first "Query 1" / "Query 1:" / "Query  1." inside the body.
    qmatch = re.search(r'\n(?:Query|Question)\s*1\s*[.:]', body)
    if qmatch:
        scen_text = body[:qmatch.start()].rstrip()
    else:
        scen_text = body.strip()

    # Tolerate "BACKGROUND " / "CONDITIONS " trailing whitespace too.
    bg_m   = re.search(r'BACKGROUND\s*\n(.*?)(?=\n\s*CONDITIONS\b)', scen_text, re.DOTALL)
    cond_m = re.search(r'CONDITIONS\s*\n(.*?)$',                      scen_text, re.DOTALL)
    if bg_m and cond_m:
        return bg_m.group(1).strip(), cond_m.group(1).strip()
    # Free-form (diverse): take the whole pre-query block as the background.
    return scen_text, None


def build_input(background: str, conditions, query_text: str) -> str:
    if conditions is not None:
        ctx = f"BACKGROUND\n{background}\n\nCONDITIONS\n{conditions}\n\n"
    else:
        ctx = f"{background}\n\n"
    scenario_block = (
        f"{SCENARIO_OPEN}\n"
        f"{ctx}"
        f"QUERIES\nQuery: {query_text}\n"
        f"{SCENARIO_CLOSE}"
    )
    return PROMPT_PREFIX + scenario_block


def one_hot_bin(value: int) -> list:
    v = max(0, min(100, int(value)))
    out = [0.0] * 101
    out[v] = 1.0
    return out


def extract_motif_params(scenario_id: str):
    """Return (P, C, R, N, seed) — same shape as pyro-rej's extract_motif_params,
    so the same holdout rule (P=1,C=1,R=0) / (P=2,C=1,R=0) selects the same
    (scenario-family, motif) entries."""
    m = re.search(r'P-(\d+)-C-(\d+)-R-(\d+)-N-(\d+)-.*?(\d+)$', scenario_id)
    if m:
        return tuple(int(x) for x in m.groups())
    return None


def split_dataset(dataset, output_base_path: str, seed: int = 42):
    """Motif-based split holding out P=1,C=1,R=0 and P=2,C=1,R=0 for val/test.
    Mirrors convert_rej_to_finetune.split_dataset."""
    np.random.seed(seed)
    holdout_configs = {(1, 1, 0), (2, 1, 0)}
    train, holdout = [], []
    for sample in dataset:
        params = extract_motif_params(sample.get('_scenario_id', ''))
        if params and (params[0], params[1], params[2]) in holdout_configs:
            holdout.append(sample)
        else:
            train.append(sample)
    np.random.shuffle(holdout)
    mid = len(holdout) // 2
    val, test = holdout[:mid], holdout[mid:]
    np.random.shuffle(train)
    for name, data in (('train', train), ('val', val), ('test', test)):
        clean = [{k: v for k, v in s.items() if k != '_scenario_id'} for s in data]
        path = f"{output_base_path}_{name}.json"
        with open(path, 'w') as f:
            json.dump(clean, f, indent=2)
        print(f"  {name}: {len(clean)} samples → {path}")


def build_category_dataset(category: str, output_base: str, predicate, all_files: list):
    files = [f for f in all_files if predicate(f)]
    print(f"\n[{category}] {len(files)} input files")

    dataset = []
    n_no_scenario = n_query_mismatch = n_processed = 0

    for fname in sorted(files):
        path = os.path.join(INPUT_DIR, fname)
        try:
            with open(path) as f:
                d = json.load(f)
        except Exception as e:
            print(f"  load fail: {fname}: {e}")
            continue
        background, conditions = parse_scenario_block(d.get('gemini_prompt', ''))
        if background is None:
            n_no_scenario += 1
            continue

        queries = d.get('queries') or []
        answers = d.get('gemini_answers') or []
        if len(queries) != len(answers):
            n_query_mismatch += 1
            continue

        sid = d.get('scenario_id', fname.replace('.json', ''))
        for q, a in zip(queries, answers):
            entry = {
                'input':  build_input(background, conditions, q),
                'output': str(int(a)),
                'bins':   [one_hot_bin(a)],
                '_scenario_id': sid,
            }
            dataset.append(entry)
        n_processed += 1

    output_base_path = os.path.join(OUTPUT_DIR, output_base)
    split_dataset(dataset, output_base_path)

    print(f"  processed={n_processed}  no_scenario={n_no_scenario}  "
          f"query_mismatch={n_query_mismatch}  datapoints={len(dataset)}")


def main():
    parser = argparse.ArgumentParser(description='Convert gemini_direct JSONs to torchtune SFT format.')
    parser.add_argument('--mode', default='all',
                        choices=['all'] + list(CATEGORIES.keys()))
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_files = [
        f for f in os.listdir(INPUT_DIR)
        if f.startswith('gemini-') and f.endswith('.json')
        and os.path.isfile(os.path.join(INPUT_DIR, f))
    ]
    print(f'Found {len(all_files)} gemini_direct JSON files in {INPUT_DIR}')

    if args.mode == 'all':
        to_run = list(CATEGORIES.items())
    else:
        to_run = [(args.mode, CATEGORIES[args.mode])]

    for category, (output_base, predicate) in to_run:
        build_category_dataset(category, output_base, predicate, all_files)


if __name__ == '__main__':
    main()
