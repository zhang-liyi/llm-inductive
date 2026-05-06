"""
Transform forward_sampling_dataset files from 4-queries-per-scenario
to 1-query-per-scenario (expanding to 4x the number of entries).
"""

import json
import re
import ast
import os

OLD_HEADER = (
    "Answer the queries in the scenario and return only a list of integers, "
    "each wrapped in < and >. For example, an output can be: "
    "[<mean1>, <mean2>]. For queries on individual rank, a higher number means "
    "a higher ranking (e.g. 100 means the individual ranks highest in that "
    "criterion; 1 is lowest). For queries on which of the two teams wins, a "
    "smaller number means the first team more likely wins."
)

NEW_HEADER = (
    "Answer the query in the scenario and return only an integer wrapped in "
    "< and >. For example, <x>. Use 0-100 scale. For a query on individual "
    "rank, a higher number means a higher ranking (e.g. 100 means the "
    "individual ranks highest in that criterion; 1 is lowest). For a query on "
    "which of the two teams wins, a smaller number means the first team more "
    "likely wins."
)

QUERIES_PATTERN = re.compile(r'QUERIES\n(.+?)\n<END_SCENARIO>', re.DOTALL)
QUERY_SPLIT = re.compile(r'\nQuery \d+: ')


def split_queries(queries_text):
    """Split 'Query 1: text1\nQuery 2: text2\n...' into individual query texts."""
    # Strip the leading 'Query 1: ' prefix from the first part
    # prepend a newline so the split pattern matches uniformly
    parts = QUERY_SPLIT.split('\n' + queries_text)
    # parts[0] is empty string before 'Query 1: '
    return [p.strip() for p in parts if p.strip()]


def transform_entry(entry):
    """Transform a 4-query entry into 4 single-query entries."""
    input_text = entry['input']
    output_vals = ast.literal_eval(entry['output'])  # e.g. [1, 100, 81, 76]
    bins_list = entry['bins']                          # list of 4 bin arrays
    metadata = entry['metadata']

    assert len(output_vals) == 4, f"Expected 4 outputs, got {len(output_vals)}"
    assert len(bins_list) == 4, f"Expected 4 bin arrays, got {len(bins_list)}"

    # Replace header
    new_input_base = input_text.replace(OLD_HEADER, NEW_HEADER, 1)

    # Extract prefix (everything up to and including 'QUERIES\n')
    q_start = new_input_base.index('QUERIES\n') + len('QUERIES\n')
    scenario_prefix = new_input_base[:q_start]

    # Extract individual queries
    m = QUERIES_PATTERN.search(new_input_base)
    queries_text = m.group(1)
    query_texts = split_queries(queries_text)

    assert len(query_texts) == 4, f"Expected 4 queries, got {len(query_texts)}: {query_texts}"

    results = []
    for i in range(4):
        new_input = scenario_prefix + f"Query: {query_texts[i]}\n<END_SCENARIO>"
        new_entry = {
            "input": new_input,
            "output": f"<{int(round(float(output_vals[i])))}>",
            "bins": [bins_list[i]],
            "metadata": {
                "source_file": metadata["source_file"],
                "motifs": metadata["motifs"],
                "num_queries": 1,
                "query_index": i + 1,
                "raw_answers": {f"query{i + 1}": metadata["raw_answers"][f"query{i + 1}"]},
            },
        }
        results.append(new_entry)

    return results


def transform_file(input_path, output_path):
    print(f"Loading {input_path} ...", flush=True)
    with open(input_path) as f:
        data = json.load(f)
    print(f"  Loaded {len(data)} entries", flush=True)

    new_data = []
    for i, entry in enumerate(data):
        new_data.extend(transform_entry(entry))
        if (i + 1) % 50000 == 0:
            print(f"  Processed {i + 1}/{len(data)} entries ({len(new_data)} output entries so far)", flush=True)

    # Write to a temp file first, then atomically rename to avoid corrupting
    # the original if interrupted mid-write.
    tmp_path = output_path + ".tmp"
    print(f"  Writing {len(new_data)} entries to {tmp_path} ...", flush=True)
    with open(tmp_path, 'w') as f:
        json.dump(new_data, f)
    os.replace(tmp_path, output_path)
    print(f"  Done. Output size: {os.path.getsize(output_path) / 1e9:.2f} GB", flush=True)


DATA_DIR = os.path.dirname(os.path.abspath(__file__))

FILES = [
    ("forward_sampling_dataset.json",       "forward_sampling_dataset.json"),
    ("forward_sampling_dataset_train.json", "forward_sampling_dataset_train.json"),
    ("forward_sampling_dataset_val.json",   "forward_sampling_dataset_val.json"),
    ("forward_sampling_dataset_test.json",  "forward_sampling_dataset_test.json"),
]

if __name__ == "__main__":
    import sys

    # Quick sanity check on 5 entries before processing all files
    print("=== Sanity check ===")
    sample_path = os.path.join(DATA_DIR, "forward_sampling_dataset.json")
    with open(sample_path) as f:
        sample = json.load(f)

    for entry in sample[:5]:
        transformed = transform_entry(entry)
        assert len(transformed) == 4
        for j, t in enumerate(transformed):
            assert t['metadata']['num_queries'] == 1
            assert 'Query: ' in t['input']
            assert 'Query 1:' not in t['input']
            assert isinstance(t['output'], str)
            assert t['output'].startswith('<') and t['output'].endswith('>')
            assert t['output'][1:-1].lstrip('-').isdigit()
            assert len(t['bins']) == 1 and len(t['bins'][0]) == 101
        # Make sure distinct queries per entry
        queries = [t['input'].split('Query: ')[1].split('\n')[0] for t in transformed]
        assert len(set(queries)) == 4, f"Expected 4 distinct queries, got: {queries}"

    print("Sanity check passed.")
    print()

    # Process all files
    for fname, outname in FILES:
        inpath = os.path.join(DATA_DIR, fname)
        outpath = os.path.join(DATA_DIR, outname)
        transform_file(inpath, outpath)
        print()

    print("All done.")
