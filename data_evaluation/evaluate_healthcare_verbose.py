"""
Evaluate a pretrained or fine-tuned Llama3-8B on the healthcare val set.

Same as evaluate_healthcare.py but with --log_responses N:
  - records the full decoded model response for the first N items
  - records the complete parse trace (every number token hit, its position,
    and which one was ultimately selected as the answer)

Usage:
    python evaluate_healthcare_verbose.py --pretrained --log_responses 20 -o results/healthcare_eval/pretrained_verbose.json
    python evaluate_healthcare_verbose.py --checkpoint_dir ckpt/... --log_responses 20 -o results/healthcare_eval/ckpt_verbose.json
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

from torchtune import training
from torchtune.models.llama3 import llama3_8b, lora_llama3_8b, llama3_tokenizer
from torchtune.modules.peft import get_adapter_params, set_trainable_params
from torchtune.training.checkpointing._checkpointer import FullModelHFCheckpointer

_TORCHTUNE_DIR = "./torchtune"
sys.path.insert(0, _TORCHTUNE_DIR)
from probabilistic_reasoning_utils import evaluate_predictions
from answer_parsing_utils import get_number_token_ids, infer_answer, infer_answer_verbose

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_MODEL_PATH = (
    "<DATA_ROOT>/resources/"
    "models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/"
    "e1945c40cd546c78e41f1151f4db032b271faeaa"
)
DEFAULT_VAL_DATA = os.path.join(
    _TORCHTUNE_DIR, "data", "pyro", "pytorch_mcmc_healthcare_val.json"
)

LORA_CONFIG = {
    "lora_attn_modules": ["q_proj", "v_proj", "output_proj"],
    "apply_lora_to_mlp": True,
    "apply_lora_to_output": False,
    "lora_rank": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.0,
}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _ckpt_files(directory, exclude="adapter"):
    for ext in (".safetensors", ".bin"):
        files = sorted(
            f for f in os.listdir(directory)
            if f.endswith(ext) and exclude not in f.lower()
        )
        if files:
            return files
    raise FileNotFoundError(f"No checkpoint files found in {directory}")


def load_pretrained_model(model_path, device="cuda", dtype=torch.bfloat16):
    print(f"Loading pretrained base model from: {model_path}")
    tokenizer = llama3_tokenizer(path=os.path.join(model_path, "tokenizer.model"))

    checkpointer = FullModelHFCheckpointer(
        checkpoint_dir=model_path,
        checkpoint_files=_ckpt_files(model_path, exclude="__none__"),
        model_type="LLAMA3",
        output_dir=os.path.join(model_path, os.pardir),
    )
    ckpt = checkpointer.load_checkpoint()

    with training.set_default_dtype(dtype), torch.device(device):
        model = llama3_8b()
    model.load_state_dict(ckpt[training.MODEL_KEY], strict=True)
    model.eval()
    print("Pretrained model ready.")
    return model, tokenizer


def load_finetuned_model(checkpoint_dir, model_path, device="cuda", dtype=torch.bfloat16):
    print(f"Loading fine-tuned checkpoint from: {checkpoint_dir}")
    tokenizer = llama3_tokenizer(path=os.path.join(model_path, "tokenizer.model"))

    checkpointer = FullModelHFCheckpointer(
        checkpoint_dir=checkpoint_dir,
        checkpoint_files=_ckpt_files(checkpoint_dir),
        model_type="LLAMA3",
        output_dir=os.path.join(checkpoint_dir, os.pardir),
    )
    ckpt = checkpointer.load_checkpoint()

    with training.set_default_dtype(dtype), torch.device(device):
        model = lora_llama3_8b(
            lora_attn_modules=LORA_CONFIG["lora_attn_modules"],
            apply_lora_to_mlp=LORA_CONFIG["apply_lora_to_mlp"],
            apply_lora_to_output=LORA_CONFIG["apply_lora_to_output"],
            lora_rank=LORA_CONFIG["lora_rank"],
            lora_alpha=LORA_CONFIG["lora_alpha"],
            lora_dropout=LORA_CONFIG["lora_dropout"],
        )

    set_trainable_params(model, get_adapter_params(model))
    model.load_state_dict(ckpt[training.MODEL_KEY], strict=False)
    if training.ADAPTER_KEY in ckpt:
        model.load_state_dict(ckpt[training.ADAPTER_KEY], strict=False)
        print("Loaded adapter weights.")
    model.eval()
    print("Fine-tuned model ready.")
    return model, tokenizer


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
# Delegates to answer_parsing_utils.infer_answer / infer_answer_verbose.
# Uses the FIRST number token (query_idx=0), matching the training-time
# evaluator in custom_lora_answer_only.py.


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate(model, tokenizer, val_data, device, max_new_tokens, verbose, log_responses):
    number_token_ids = get_number_token_ids(tokenizer)

    pred_dists = []
    gt_dists = []
    greedy_values = []
    gt_means = []
    response_logs = []
    n_skipped = 0

    t0 = time.time()
    for idx, item in enumerate(val_data):
        prompt = item["input"]
        gt_bins = np.array(item["bins"][0])

        record = (log_responses > 0 and idx < log_responses)

        if record:
            pred_dist, greedy_val, all_hits, full_response = infer_answer_verbose(
                model, tokenizer, prompt, number_token_ids,
                device=device, max_new_tokens=max_new_tokens,
                query_idx=0,
            )
        else:
            pred_dist, greedy_val, all_hits = infer_answer(
                model, tokenizer, prompt, number_token_ids,
                device=device, max_new_tokens=max_new_tokens,
                query_idx=0,
            )
            full_response = None

        if pred_dist is None:
            print(f"  [{idx}] WARNING: no number token produced, skipping.")
            n_skipped += 1
            continue

        pred_dists.append(pred_dist)
        gt_dists.append(gt_bins)
        greedy_values.append(greedy_val)
        gt_mean = float(np.sum(np.arange(101) * gt_bins))
        gt_means.append(gt_mean)

        if record:
            selected_step = all_hits[0].step  # query_idx=0 → first hit
            number_hits_log = [
                {
                    "step": h.step,
                    "token_id": number_token_ids[h.value],
                    "value": h.value,
                    "selected": (h.step == selected_step),
                }
                for h in all_hits
            ]
            response_logs.append({
                "idx": idx,
                "prompt_tail": prompt[-300:],
                "gt_output": item["output"],
                "gt_mean": gt_mean,
                "greedy_value": greedy_val,
                "pred_mean": float(np.sum(np.arange(101) * pred_dist)),
                "full_response": full_response,
                "number_hits": number_hits_log,
                "answer_step": selected_step,
                "answer_value": greedy_val,
                "n_number_hits": len(all_hits),
            })
            print(f"\n[{idx:4d}] === VERBOSE RESPONSE LOG ===")
            print(f"  full_response   : {repr(full_response)}")
            print(f"  n_number_hits   : {len(all_hits)}")
            print(f"  number_hits     : {[(h.value, h.step == selected_step) for h in all_hits]}")
            print(f"  answer_step     : {selected_step}  →  value={greedy_val}")
            print(f"  gt_output       : {item['output']}   gt_mean={gt_mean:.2f}")
        elif verbose or (idx % 50 == 0):
            pred_mean = float(np.sum(np.arange(101) * pred_dist))
            print(f"  [{idx:4d}] greedy={greedy_val:3d}  pred_mean={pred_mean:5.1f}  gt_mean={gt_mean:5.1f}")

    elapsed = time.time() - t0
    print(f"\nInference done: {len(pred_dists)} items in {elapsed:.1f}s  ({n_skipped} skipped)")

    metrics = evaluate_predictions(pred_dists, gt_dists)
    return metrics, pred_dists, gt_dists, greedy_values, response_logs


def main():
    parser = argparse.ArgumentParser(description="Evaluate Llama3-8B on healthcare val set (verbose)")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--val_data", type=str, default=DEFAULT_VAL_DATA)
    parser.add_argument("--output", "-o", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--log_responses", type=int, default=20,
                        help="Record full response + parse trace for the first N items (default: 20)")
    args = parser.parse_args()

    if not args.pretrained and args.checkpoint_dir is None:
        parser.error("Provide --pretrained or --checkpoint_dir.")

    with open(args.val_data) as f:
        val_data = json.load(f)
    print(f"Loaded {len(val_data)} val items from {args.val_data}")

    if args.pretrained:
        model, tokenizer = load_pretrained_model(args.model_path, device=args.device)
        run_label = "pretrained"
    else:
        model, tokenizer = load_finetuned_model(
            args.checkpoint_dir, args.model_path, device=args.device
        )
        run_label = os.path.normpath(args.checkpoint_dir).replace(os.sep, "_")

    metrics, pred_dists, gt_dists, greedy_values, response_logs = evaluate(
        model, tokenizer, val_data, args.device, args.max_new_tokens,
        args.verbose, args.log_responses,
    )

    print("\n" + "=" * 60)
    print(f"Results — {run_label}")
    print(f"  Val file:          {args.val_data}")
    print(f"  Items evaluated:   {len(pred_dists)}")
    print(f"  MAE (mean):        {metrics['mean_abs_error']:.3f}")
    print(f"  MAE (dist L1):     {metrics['mean_abs_error_dist']:.3f}")
    print(f"  KL divergence:     {metrics['kl_divergence']:.4f}")
    print(f"  CE (vs dist):      {metrics['cross_entropy_dist']:.4f}")
    print(f"  CE (vs mean tok):  {metrics['cross_entropy_mean']:.4f}")
    print("=" * 60)

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        out = {
            "run_label": run_label,
            "val_data": args.val_data,
            "n_items": len(pred_dists),
            "metrics": {k: v for k, v in metrics.items()
                        if k not in ("pred_means", "gt_means")},
            "per_item": [
                {
                    "greedy": int(g),
                    "pred_mean": float(np.sum(np.arange(101) * p)),
                    "gt_mean": float(np.sum(np.arange(101) * gt)),
                }
                for g, p, gt in zip(greedy_values, pred_dists, gt_dists)
            ],
            "response_logs": response_logs,
        }
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Full results saved to {args.output}")


if __name__ == "__main__":
    main()
