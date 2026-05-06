"""
prepare_bayesian_teaching.py

Download, parse, and build evaluation prompts for the Bayesian Teaching dataset
(from "Bayesian teaching enables probabilistic reasoning in large language models",
Zenodo: https://zenodo.org/records/17677329).

Dataset structure
-----------------
Each interaction record has exactly 5 rounds.  We use rounds 0–3 as evidence
(each shows 3 options + the user's preferred option) and treat round 4 as the
held-out prediction target (the model sees the 3 options but not the label).

Output format
-------------
Each prepared example:
    {
        "task":       "flight" | "hotel" | "webshop",
        "source":     filename (webshop category or task name),
        "idx":        original record index,
        "input":      prompt string,
        "output":     "<N>"  where N ∈ {1, 2, 3}  (1-indexed choice),
        "metadata":   {...}
    }

The output follows the same <x> format used elsewhere in this codebase.
"""

import json
import os
import urllib.request

RAW_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bayesian_teaching_raw")
OUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bayesian_teaching_test.jsonl")

ZENODO_URL = "https://zenodo.org/records/17677329/files/data.zip?download=1"
ZIP_PATH   = os.path.join(RAW_DIR, "data.zip")

INTERACTION_DIR = os.path.join(RAW_DIR, "data", "eval", "interaction")

# Number of evidence rounds to show (remainder is the target)
N_EVIDENCE = 4  # rounds 0-3 as evidence, round 4 as target


# ── Download / extract ─────────────────────────────────────────────────────────

def ensure_data():
    """Download and extract data.zip if not already present."""
    os.makedirs(RAW_DIR, exist_ok=True)
    if not os.path.exists(ZIP_PATH):
        print(f"Downloading dataset from Zenodo …")
        urllib.request.urlretrieve(ZENODO_URL, ZIP_PATH)
        print(f"  Saved to {ZIP_PATH}")
    else:
        print(f"  data.zip already present at {ZIP_PATH}")

    interaction_path = os.path.join(RAW_DIR, "data", "eval", "interaction")
    if not os.path.isdir(interaction_path):
        import zipfile
        print("Extracting data.zip …")
        with zipfile.ZipFile(ZIP_PATH, "r") as zf:
            zf.extractall(RAW_DIR)
        print("  Extracted.")
    else:
        print("  Already extracted.")


# ── File inventory ─────────────────────────────────────────────────────────────

def inventory():
    """Return dict mapping task → list of (label, path) pairs."""
    inv = {"flight": [], "hotel": [], "webshop": []}

    for fname in sorted(os.listdir(INTERACTION_DIR)):
        fpath = os.path.join(INTERACTION_DIR, fname)
        if os.path.isfile(fpath) and fname.endswith(".jsonl"):
            if fname.startswith("flight"):
                inv["flight"].append((fname[:-6], fpath))
            elif fname.startswith("hotel"):
                inv["hotel"].append((fname[:-6], fpath))

    ws_dir = os.path.join(INTERACTION_DIR, "webshop")
    if os.path.isdir(ws_dir):
        for fname in sorted(os.listdir(ws_dir)):
            if fname.endswith(".jsonl"):
                fpath = os.path.join(ws_dir, fname)
                inv["webshop"].append((fname[:-6], fpath))

    return inv


# ── Prompt construction ────────────────────────────────────────────────────────

_FLIGHT_HEADER = (
    "You are helping a user choose among flights. "
    "The user has fixed preferences they apply consistently. "
    "Learn from the feedback below and choose the option the user would prefer."
)

_HOTEL_HEADER = (
    "You are helping a user choose among hotels. "
    "The user has fixed preferences they apply consistently. "
    "Learn from the feedback below and choose the option the user would prefer."
)

_WEBSHOP_HEADER = (
    "You are helping a user choose among products. "
    "The user has fixed preferences they apply consistently. "
    "Learn from the feedback below and choose the option the user would prefer."
)


def _task_header(task: str) -> str:
    return {"flight": _FLIGHT_HEADER, "hotel": _HOTEL_HEADER}.get(task, _WEBSHOP_HEADER)


def _format_options(options: list) -> str:
    lines = []
    for i, opt in enumerate(options, 1):
        # Options are already formatted strings like "Flight 1: …" or "Product 1: …"
        # Re-label consistently as Option 1/2/3 to avoid leaking the numbering
        # embedded in the string (some have "Flight 1" which matches position).
        # Actually keep original text since stripping it would lose info; just
        # present as-is and prepend the index.
        lines.append(f"  Option {i}: {opt}")
    return "\n".join(lines)


def build_prompt(task: str, rounds: list, evidence_count: int = N_EVIDENCE) -> str:
    """
    Build a single prompt from the interaction record.

    rounds[0..evidence_count-1]  → shown with feedback
    rounds[evidence_count]        → shown without feedback (the target)
    """
    instruction = (
        f"{_task_header(task)}\n\n"
        "Output only the number of the best option wrapped in < and >. "
        "For example: <1> or <2> or <3>. No other text."
    )

    lines = [instruction, ""]

    # Evidence rounds
    for i in range(evidence_count):
        r = rounds[i]
        lines.append(f"--- Round {i + 1} ---")
        lines.append(_format_options(r["options"]))
        preferred_1indexed = r["user_idx"] + 1   # dataset uses 0-indexed
        lines.append(f"User feedback: preferred option = {preferred_1indexed}")
        lines.append("")

    # Target round
    target_round = rounds[evidence_count]
    lines.append(f"--- Final round ---")
    lines.append(_format_options(target_round["options"]))
    lines.append("")
    lines.append("Which option does the user prefer? Answer with <1>, <2>, or <3> only.")

    return "\n".join(lines)


# ── Parsing ────────────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> list:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def parse_file(task: str, label: str, path: str) -> list:
    """Parse one interaction JSONL file into prepared examples."""
    records = load_jsonl(path)
    examples = []
    for rec in records:
        rounds = rec.get("rounds", [])
        if len(rounds) < N_EVIDENCE + 1:
            continue   # need at least N_EVIDENCE evidence + 1 target

        target_round  = rounds[N_EVIDENCE]
        gt_0indexed   = target_round["user_idx"]
        gt_1indexed   = gt_0indexed + 1

        prompt = build_prompt(task, rounds, evidence_count=N_EVIDENCE)

        examples.append({
            "task":   task,
            "source": label,
            "idx":    rec.get("idx", len(examples)),
            "input":  prompt,
            "output": f"<{gt_1indexed}>",
            "metadata": {
                "reward_fn":   rec.get("reward_fn"),
                "seed":        rec.get("seed"),
                "n_evidence":  N_EVIDENCE,
                "gt_0indexed": gt_0indexed,
            },
        })

    return examples


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=== Preparing Bayesian Teaching evaluation data ===\n")
    ensure_data()

    print("\nInventory:")
    inv = inventory()
    for task, files in inv.items():
        print(f"  {task}: {len(files)} file(s)")
        for label, _ in files[:3]:
            print(f"    {label}")
        if len(files) > 3:
            print(f"    … ({len(files) - 3} more)")

    print("\nBuilding examples …")
    all_examples = []
    counts = {}
    for task, files in inv.items():
        task_examples = []
        for label, path in files:
            exs = parse_file(task, label, path)
            task_examples.extend(exs)
        counts[task] = len(task_examples)
        all_examples.extend(task_examples)
        print(f"  {task}: {len(task_examples)} examples")

    print(f"\nTotal: {len(all_examples)} examples")

    # Write JSONL
    with open(OUT_FILE, "w") as f:
        for ex in all_examples:
            f.write(json.dumps(ex) + "\n")
    print(f"Saved to {OUT_FILE}")

    # ── Print 3 sample prompts ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Sample prompts (truncated to 800 chars each):")
    import random
    rng = random.Random(42)
    # One from each task
    for task in ["flight", "hotel", "webshop"]:
        task_exs = [e for e in all_examples if e["task"] == task]
        if not task_exs:
            continue
        ex = rng.choice(task_exs)
        print(f"\n--- {task.upper()} (source={ex['source']}, gt={ex['output']}) ---")
        print(ex["input"][:800])
        if len(ex["input"]) > 800:
            print("[… truncated …]")


if __name__ == "__main__":
    main()
