"""
Evaluate a pretrained or fine-tuned Llama3-8B on the healthcare val set,
using the two-stage LLM-parser pipeline from answer_parsing_utils.

For each val item:
  1. The main model generates a full response (greedy, max_new_tokens=32).
  2. A separate pretrained Llama3-8B (the "parser LLM") reads that response
     and outputs a clean integer 0-100.
  3. Metrics are computed from the PARSER's distribution, not the main model's.

Also records, per item:
  - main_val:    first-hit integer from the main model (for comparison)
  - parser_val:  integer extracted by the parser LLM
  - full_response: raw main-model output string

Usage:
    python evaluate_healthcare_llm_parsed.py --pretrained
    python evaluate_healthcare_llm_parsed.py --checkpoint_dir ckpt/llama3_8B/.../epoch_0
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
from answer_parsing_utils import get_number_token_ids, LLMParser, infer_answer_llm_parsed

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
    print(f"Loading pretrained main model from: {model_path}")
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
    print("Main model ready.")
    return model, tokenizer


def load_finetuned_model(checkpoint_dir, model_path, device="cuda", dtype=torch.bfloat16):
    print(f"Loading fine-tuned main model from: {checkpoint_dir}")
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
    print("Fine-tuned main model ready.")
    return model, tokenizer


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def evaluate(model, tokenizer, val_data, parser, device, max_new_tokens, verbose):
    number_token_ids = get_number_token_ids(tokenizer)

    pred_dists = []
    gt_dists = []
    parser_values = []
    main_values = []
    gt_means = []
    n_skipped = 0

    t0 = time.time()
    for idx, item in enumerate(val_data):
        prompt = item["input"]
        gt_bins = np.array(item["bins"][0])

        parser_dist, parser_val, _, full_response, main_val = infer_answer_llm_parsed(
            model, tokenizer, prompt, number_token_ids, parser,
            device=device, max_new_tokens=max_new_tokens, query_idx=0,
        )

        if parser_dist is None:
            print(f"  [{idx}] WARNING: parser produced no number token "
                  f"(main_val={main_val}, response={repr(full_response[:60])}), skipping.")
            n_skipped += 1
            continue

        pred_dists.append(parser_dist)
        gt_dists.append(gt_bins)
        parser_values.append(parser_val)
        main_values.append(main_val if main_val is not None else -1)
        gt_mean = float(np.sum(np.arange(101) * gt_bins))
        gt_means.append(gt_mean)

        if verbose or (idx % 50 == 0):
            pred_mean = float(np.sum(np.arange(101) * parser_dist))
            print(f"  [{idx:4d}] main={main_val!s:>4}  parser={parser_val:3d}"
                  f"  pred_mean={pred_mean:5.1f}  gt_mean={gt_mean:5.1f}"
                  f"  response={repr(full_response[:50])}")

    elapsed = time.time() - t0
    print(f"\nInference done: {len(pred_dists)} items in {elapsed:.1f}s  ({n_skipped} skipped)")

    metrics = evaluate_predictions(pred_dists, gt_dists)
    return metrics, pred_dists, gt_dists, parser_values, main_values


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser_arg = argparse.ArgumentParser(
        description="Evaluate Llama3-8B on healthcare val set using LLM parser")
    parser_arg.add_argument("--pretrained", action="store_true")
    parser_arg.add_argument("--checkpoint_dir", type=str, default=None)
    parser_arg.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    parser_arg.add_argument("--parser_model_path", type=str, default=None,
                            help="Path to parser LLM (defaults to --model_path)")
    parser_arg.add_argument("--val_data", type=str, default=DEFAULT_VAL_DATA)
    parser_arg.add_argument("--output", "-o", type=str, default=None)
    parser_arg.add_argument("--device", type=str, default="cuda")
    parser_arg.add_argument("--max_new_tokens", type=int, default=32)
    parser_arg.add_argument("--verbose", action="store_true")
    args = parser_arg.parse_args()

    if not args.pretrained and args.checkpoint_dir is None:
        parser_arg.error("Provide --pretrained or --checkpoint_dir.")

    parser_model_path = args.parser_model_path or args.model_path

    # Load data
    with open(args.val_data) as f:
        val_data = json.load(f)
    print(f"Loaded {len(val_data)} val items from {args.val_data}")

    # Load main model
    if args.pretrained:
        model, tokenizer = load_pretrained_model(args.model_path, device=args.device)
        run_label = "pretrained"
    else:
        model, tokenizer = load_finetuned_model(
            args.checkpoint_dir, args.model_path, device=args.device)
        run_label = os.path.normpath(args.checkpoint_dir).replace(os.sep, "_")

    # Load parser LLM
    llm_parser = LLMParser.from_pretrained(parser_model_path, device=args.device)

    # Evaluate
    metrics, pred_dists, gt_dists, parser_values, main_values = evaluate(
        model, tokenizer, val_data, llm_parser,
        args.device, args.max_new_tokens, args.verbose,
    )

    # Print summary
    print("\n" + "=" * 60)
    print(f"Results (LLM-parsed) — {run_label}")
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
            "parser": "llm",
            "parser_model_path": parser_model_path,
            "val_data": args.val_data,
            "n_items": len(pred_dists),
            "metrics": {k: v for k, v in metrics.items()
                        if k not in ("pred_means", "gt_means")},
            "per_item": [
                {
                    "parser_val": int(pv),
                    "main_val": int(mv),
                    "pred_mean": float(np.sum(np.arange(101) * p)),
                    "gt_mean": float(np.sum(np.arange(101) * gt)),
                }
                for pv, mv, p, gt in zip(parser_values, main_values, pred_dists, gt_dists)
            ],
        }
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Full results saved to {args.output}")


if __name__ == "__main__":
    main()
