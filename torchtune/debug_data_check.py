"""
debug_data_check.py

Meticulous data check for the SFT trajectory → SFT answer pipeline.
Loads one example from each phase and prints:

  Phase 1 (SFT on trajectory):
    - Full decoded sequence
    - Token-by-token listing of the response portion (label != -100),
      confirming which tokens receive CE loss.

  Phase 2 (SFT on answer):
    - Full decoded prompt (what the model will see before generation)
    - Ground-truth integer and the peak of the gt_bins posterior

No model is loaded. No GPU needed. No CSVs or checkpoints written.

Usage (from torchtune/):
    python debug_data_check.py
"""

import json
import sys
import os
from functools import partial
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

# ── make sure local modules resolve ──────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from torchtune.models.llama3 import llama3_tokenizer
from torchtune.data import Message

from custom_lora_trajectory import SFTTrajDataset, sft_traj_collate_fn, SingleQueryDataset
from custom_lora_trajectory import single_query_collate_fn
from probabilistic_reasoning_utils import get_number_token_ids

# ── config ────────────────────────────────────────────────────────────────────

TOKENIZER_PATH = (
    "<DATA_ROOT>/resources/"
    "models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/"
    "e1945c40cd546c78e41f1151f4db032b271faeaa/tokenizer.model"
)
PHASE1_JSON = "data/pyro/sft_thinking_trajectory_train.json"
PHASE2_JSON = "data/pyro/pytorch_mcmc_dataset_train.json"
MAX_SEQ_LEN = 2048

SEP = "=" * 80


def decode_token(tokenizer, tok_id: int) -> str:
    """Decode a single token id to a human-readable string."""
    try:
        return repr(tokenizer.decode([tok_id]))
    except Exception:
        return f"<tok {tok_id}>"


def print_phase1(tokenizer, batch: dict, number_token_set: set) -> None:
    print(SEP)
    print("PHASE 1 — SFT on thinking trajectory (one example)")
    print(SEP)

    # Take first example in batch
    input_ids = batch["input_ids"][0].tolist()
    labels    = batch["labels"][0].tolist()

    # Split into prompt region (label == -100) and response region
    response_start = next(
        (i for i, l in enumerate(labels) if l != -100),
        len(labels),
    )

    prompt_ids    = input_ids[:response_start]
    response_ids  = input_ids[response_start:]
    response_lbls = labels[response_start:]

    prompt_text   = tokenizer.decode(prompt_ids)
    response_text = tokenizer.decode(response_ids)

    print("\n── PROMPT (masked, no loss) ──────────────────────────────────────")
    print(prompt_text)

    print("\n── RESPONSE (assistant output; some header/special tokens may still be masked) ───────")
    print(response_text)

    print("\n── TOKEN-BY-TOKEN: response portion ────────────────────────────────")
    print(f"  {'pos':>5}  {'label_id':>9}  {'trained':>7}  {'is_num':>6}  token_str")
    print(f"  {'---':>5}  {'---------':>9}  {'-------':>7}  {'------':>6}  ---------")

    # Print first 60 and last 20 response tokens to keep output manageable
    n = len(response_ids)
    indices = list(range(min(60, n))) + (
        list(range(max(60, n - 20), n)) if n > 60 else []
    )
    prev = -1
    for i in indices:
        if prev != -1 and i != prev + 1:
            print(f"  {'...':>5}")
        tok_id  = response_ids[i]
        lbl_id  = response_lbls[i]
        trained = lbl_id != -100
        is_num  = tok_id in number_token_set
        tok_str = decode_token(tokenizer, tok_id)
        print(f"  {response_start + i:>5}  {lbl_id:>9}  {'YES' if trained else 'masked':>7}  {'YES' if is_num else '':>6}  {tok_str}")
        prev = i

    # Sanity checks
    n_trained = sum(1 for l in labels if l != -100)
    n_masked  = sum(1 for l in labels if l == -100)
    print(f"\n  Total tokens : {len(labels)}")
    print(f"  Masked (prompt): {n_masked}  |  Trained (response): {n_trained}")
    mismatches = [
        (response_start + i, r, inp)
        for i, (r, inp) in enumerate(zip(response_lbls, response_ids))
        if r != -100 and r != inp
    ]
    if mismatches:
        print(f"  ✗ MISMATCH at {len(mismatches)} position(s):")
        for pos, lbl, inp in mismatches[:5]:
            print(f"      pos={pos}  label={lbl}  input_id={inp}")
    else:
        print("  ✓ All trained positions (label != -100) match input ids")


def print_phase2(tokenizer, batch: dict) -> None:
    print(SEP)
    print("PHASE 2 — SFT on answer token only (one example)")
    print(SEP)

    # Take first example in batch
    prompt_ids = batch["prompt_ids"][0].tolist()
    # Strip pad tokens at the end
    pad_id  = tokenizer.pad_id
    non_pad = [i for i, t in enumerate(prompt_ids) if t != pad_id]
    if non_pad:
        prompt_ids = prompt_ids[: non_pad[-1] + 1]

    gt_int  = batch["ground_truth_single"][0].item()
    gt_bins = batch["ground_truth_bins"][0, 0].tolist()  # [101]

    prompt_text = tokenizer.decode(prompt_ids)

    print("\n── PROMPT (model input before generation) ──────────────────────────")
    print(prompt_text)

    # GT bins summary
    peak_val  = int(torch.tensor(gt_bins).argmax().item())
    gt_mean   = sum(i * p for i, p in enumerate(gt_bins))
    nonzero   = [(i, round(p, 4)) for i, p in enumerate(gt_bins) if p > 1e-4]

    print("\n── GROUND TRUTH ────────────────────────────────────────────────────")
    print(f"  ground_truth_single : {gt_int}")
    print(f"  gt_bins peak        : {peak_val}  (argmax of posterior)")
    print(f"  gt_bins mean        : {gt_mean:.2f}")
    print(f"  gt_bins non-zero    : {nonzero[:20]}{'...' if len(nonzero) > 20 else ''}")

    print("\n── TOKEN-BY-TOKEN: last 20 prompt tokens ───────────────────────────")
    print(f"  {'pos':>5}  {'tok_id':>7}  token_str")
    print(f"  {'---':>5}  {'------':>7}  ---------")
    for i, tok_id in enumerate(prompt_ids[-20:]):
        abs_pos = len(prompt_ids) - 20 + i
        tok_str = decode_token(tokenizer, tok_id)
        print(f"  {abs_pos:>5}  {tok_id:>7}  {tok_str}")

    print(f"\n  Prompt length (tokens, no pad): {len(prompt_ids)}")
    print(
        "  Note: during training the model generates from this prompt, "
        "then the loss is applied only at the <N> answer token position."
    )


def main() -> None:
    print("Loading tokenizer …")
    tokenizer = llama3_tokenizer(TOKENIZER_PATH)
    number_token_ids = get_number_token_ids(tokenizer)
    number_token_set = set(number_token_ids.tolist())
    print(f"  Tokenizer pad_id : {tokenizer.pad_id}")
    print(f"  Number token ids : {number_token_ids[:10].tolist()} … (101 total)")

    # ── Phase 1 ───────────────────────────────────────────────────────────────
    print(f"\nLoading phase-1 dataset from {PHASE1_JSON} …")
    ds1 = SFTTrajDataset(PHASE1_JSON, tokenizer, max_seq_len=MAX_SEQ_LEN)
    print(f"  {len(ds1)} examples")
    item1 = ds1[0]
    # Wrap in fake batch of size 1
    collate1 = partial(sft_traj_collate_fn, pad_id=tokenizer.pad_id)
    batch1   = collate1([item1])
    print_phase1(tokenizer, batch1, number_token_set)

    # ── Phase 2 ───────────────────────────────────────────────────────────────
    print(f"\nLoading phase-2 dataset from {PHASE2_JSON} …")
    ds2 = SingleQueryDataset(PHASE2_JSON, tokenizer, max_seq_len=MAX_SEQ_LEN)
    print(f"  {len(ds2)} examples")
    item2 = ds2[0]
    collate2 = partial(single_query_collate_fn, pad_id=tokenizer.pad_id)
    batch2   = collate2([item2])
    print_phase2(tokenizer, batch2)

    print(f"\n{SEP}")
    print("Data check complete.")


if __name__ == "__main__":
    main()
