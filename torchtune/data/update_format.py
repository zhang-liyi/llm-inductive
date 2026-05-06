"""
Update all JSON dataset files to use new <x> answer format.

Changes:
1. Replace old instruction with new instruction in input fields
2. Wrap bare integer outputs: "output": "77" -> "output": "<77>"
"""

import os
import re

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

OLD_INSTRUCTION = (
    "Answer the query in the scenario and return only an integer. "
    "Use 0-100 scale. For a query on individual rank or performance, "
    "a higher number means more strength (e.g. 100 is stronger than 1). "
    "For a query on which team wins, a smaller number means the first "
    "team more likely wins."
)

NEW_INSTRUCTION = (
    "Answer the query in the scenario and return only an integer wrapped "
    "in < and >. For example, <x>. Use 0-100 scale. For a query on "
    "individual rank, a higher number means a higher ranking (e.g. 100 "
    "means the individual ranks highest in that criterion; 1 is lowest). "
    "For a query on which of the two teams wins, a smaller number means "
    "the first team more likely wins."
)

# JSON-escaped versions (the instruction appears inside a JSON string field)
OLD_BYTES = OLD_INSTRUCTION.encode('utf-8')
NEW_BYTES = NEW_INSTRUCTION.encode('utf-8')

# Pattern for bare-integer output fields: "output": "77" or "output": "-1" etc.
# Must NOT match already-wrapped outputs like "output": "<77>"
OUTPUT_RE = re.compile(rb'"output":\s*"(-?\d+)"')
OUTPUT_REPL = rb'"output": "<\1>"'

CHUNK_SIZE = 64 * 1024 * 1024   # 64 MB
OVERLAP    =  4 * 1024           # 4 KB overlap to catch cross-boundary matches

FILES = [
    "probabilistic_reasoning.json",
    "probabilistic_reasoning_train.json",
    "probabilistic_reasoning_val.json",
    "probabilistic_reasoning_test.json",
    "forward_sampling_dataset.json",
    "forward_sampling_dataset_train.json",
    "forward_sampling_dataset_val.json",
    "forward_sampling_dataset_test.json",
    "single_scenario_dataset.json",
    "pyro/pytorch_mcmc_healthcare_train.json",
    "pyro/pytorch_mcmc_healthcare_val.json",
    "pyro/pytorch_mcmc_healthcare_test.json",
    "pyro/pytorch_mcmc_dataset_train.json",
    "pyro/pytorch_mcmc_dataset_val.json",
    "pyro/pytorch_mcmc_dataset_test.json",
]


def process_chunk(chunk: bytes) -> bytes:
    chunk = chunk.replace(OLD_BYTES, NEW_BYTES)
    chunk = OUTPUT_RE.sub(OUTPUT_REPL, chunk)
    return chunk


def update_file(path: str):
    size = os.path.getsize(path)
    print(f"  {path}  ({size / 1e9:.3f} GB)", flush=True)

    tmp_path = path + ".tmp"

    with open(path, 'rb') as fin, open(tmp_path, 'wb') as fout:
        tail = b""
        bytes_read = 0
        while True:
            raw = fin.read(CHUNK_SIZE)
            if not raw:
                # Flush remaining tail
                if tail:
                    fout.write(process_chunk(tail))
                break

            bytes_read += len(raw)
            block = tail + raw

            if fin.read(1) == b"":
                # Last chunk — process whole block
                fout.write(process_chunk(block))
                break
            else:
                # Rewind the one byte we peeked
                fin.seek(-1, 1)

            # Keep OVERLAP bytes as tail for next iteration
            safe = block[:-OVERLAP]
            tail = block[-OVERLAP:]
            fout.write(process_chunk(safe))

            if bytes_read % (256 * 1024 * 1024) < CHUNK_SIZE:
                print(f"    {bytes_read / 1e9:.2f} GB processed...", flush=True)

    os.replace(tmp_path, path)
    new_size = os.path.getsize(path)
    print(f"  Done. {size / 1e6:.1f} MB -> {new_size / 1e6:.1f} MB", flush=True)


def sanity_check(path: str):
    """Quick check: old instruction gone, new instruction present, outputs wrapped."""
    import json
    with open(path) as f:
        data = json.load(f)
    assert len(data) > 0
    entry = data[0]
    assert OLD_INSTRUCTION not in entry['input'], "Old instruction still present!"
    assert NEW_INSTRUCTION in entry['input'], "New instruction not found!"
    assert entry['output'].startswith('<'), f"Output not wrapped: {entry['output']!r}"
    assert entry['output'].endswith('>'), f"Output not wrapped: {entry['output']!r}"
    print(f"  Sanity check passed for {os.path.basename(path)}")


if __name__ == "__main__":
    print("=== Updating dataset files to <x> format ===\n")

    for fname in FILES:
        fpath = os.path.join(DATA_DIR, fname)
        if not os.path.exists(fpath):
            print(f"  SKIP (not found): {fpath}", flush=True)
            continue
        update_file(fpath)

    print("\n=== Sanity checks ===")
    small_files = [f for f in FILES if "forward_sampling" not in f and "single_scenario" not in f]
    for fname in small_files:
        fpath = os.path.join(DATA_DIR, fname)
        if os.path.exists(fpath):
            sanity_check(fpath)

    print("\nAll done.")
