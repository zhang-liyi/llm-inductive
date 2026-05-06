"""
Fix bins in already-transformed forward_sampling files:
wrap each bins from a flat list of 101 floats to [[101 floats]].
"""
import json
import os

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

FILES = [
    "forward_sampling_dataset.json",
    "forward_sampling_dataset_train.json",
    "forward_sampling_dataset_val.json",
    "forward_sampling_dataset_test.json",
]

for fname in FILES:
    path = os.path.join(DATA_DIR, fname)
    print(f"Loading {path} ...", flush=True)
    with open(path) as f:
        data = json.load(f)
    print(f"  {len(data)} entries. bins[0] type check: {type(data[0]['bins'][0])}", flush=True)

    # If already wrapped, skip
    if isinstance(data[0]['bins'][0], list):
        print("  Already wrapped, skipping.", flush=True)
        continue

    for entry in data:
        entry['bins'] = [entry['bins']]

    tmp = path + ".tmp"
    print(f"  Writing to {tmp} ...", flush=True)
    with open(tmp, 'w') as f:
        json.dump(data, f)
    os.replace(tmp, path)
    print(f"  Done. Size: {os.path.getsize(path) / 1e9:.2f} GB", flush=True)

print("All done.")
