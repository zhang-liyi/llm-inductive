"""
Evaluate a pretrained or fine-tuned Llama3-8B on the healthcare val set.

Loads pytorch_mcmc_healthcare_val.json (one datapoint = one scenario × one query),
runs greedy inference on each item, records the model's number-token distribution
at the answer position, and computes aggregate metrics against ground truth bins.

Usage:
    # Pretrained base model
    python evaluate_healthcare.py --pretrained

    # Fine-tuned LoRA checkpoint
    python evaluate_healthcare.py --checkpoint_dir ckpt/llama3_8B/pyro_lora_dist_r8_seed1/epoch_0

    # Custom val file
    python evaluate_healthcare.py --checkpoint_dir ckpt/... --val_data data/pyro/pytorch_mcmc_healthcare_test.json
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
from probabilistic_reasoning_utils import evaluate_predictions, BRACKET_INSTRUCTION
from answer_parsing_utils import get_number_token_ids, infer_answer, build_chat_context_tokens


def _swap_bracket_instruction(prompt: str) -> str:
    """Replace the first paragraph (the instruction prefix) with
    BRACKET_INSTRUCTION, mirroring the substitution done at training time in
    ProbabilisticReasoningDataset.__getitem__."""
    parts = prompt.split("\n\n", 1)
    return BRACKET_INSTRUCTION + ("\n\n" + parts[1] if len(parts) > 1 else "")

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
# Model loading (mirrors inference_benchmarks.py)
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
# Inference helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def teacher_force_answer(model, tokenizer, prompt, number_token_ids, device="cuda"):
    """
    Teacher-force mode: build the full chat context (user turn + assistant
    header) then append "<", and extract the distribution over number tokens
    (0-100) at the next-token prediction position.

    The chat context matches the tokenize_messages format used during training:
        [BOS] <|start_header_id|>user<|end_header_id|>\\n\\n
              {prompt} <|eot_id|>
              <|start_header_id|>assistant<|end_header_id|>\\n\\n  <

    Returns:
        pred_dist:  np.ndarray [101] — renormalized softmax over 0-100.
        greedy_val: int — argmax of pred_dist.
    """
    number_token_ids_t = torch.tensor(number_token_ids, dtype=torch.long, device=device)

    context_ids = build_chat_context_tokens(tokenizer, prompt)
    lt_ids = tokenizer.encode("<", add_bos=False, add_eos=False)  # typically [27]
    input_tensor = torch.tensor([context_ids + lt_ids], dtype=torch.long, device=device)

    logits = model(input_tensor)
    if isinstance(logits, list):
        logits = torch.cat(logits, dim=1)

    next_logits = logits[0, -1, :]          # [vocab] — position after "<"
    probs = torch.softmax(next_logits, dim=0)
    num_probs = probs[number_token_ids_t].cpu().numpy()
    num_probs = num_probs / num_probs.sum()  # renorm to 101-token subspace
    greedy_val = int(np.argmax(num_probs))

    return num_probs, greedy_val


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate(model, tokenizer, val_data, device, max_new_tokens, verbose,
             mode="generate", bracket_prompt=False):
    number_token_ids = get_number_token_ids(tokenizer)

    pred_dists = []
    gt_dists = []
    greedy_values = []
    gt_means = []
    n_skipped = 0

    t0 = time.time()
    for idx, item in enumerate(val_data):
        prompt = item["input"]
        if bracket_prompt:
            prompt = _swap_bracket_instruction(prompt)
        gt_bins = np.array(item["bins"][0])  # shape [101]

        if mode == "teacher_force":
            pred_dist, greedy_val = teacher_force_answer(
                model, tokenizer, prompt, number_token_ids, device=device,
            )
        else:
            pred_dist, greedy_val, _ = infer_answer(
                model, tokenizer, prompt, number_token_ids,
                device=device, max_new_tokens=max_new_tokens,
                query_idx=0,
            )

        if pred_dist is None:
            print(f"  [{idx}] WARNING: no number token produced, skipping.")
            n_skipped += 1
            continue

        pred_dists.append(pred_dist)
        gt_dists.append(gt_bins)
        greedy_values.append(greedy_val)
        gt_mean = float(np.sum(np.arange(101) * gt_bins))
        gt_means.append(gt_mean)

        if verbose or (idx % 50 == 0):
            pred_mean = float(np.sum(np.arange(101) * pred_dist))
            print(f"  [{idx:4d}] greedy={greedy_val:3d}  pred_mean={pred_mean:5.1f}  gt_mean={gt_mean:5.1f}")

    elapsed = time.time() - t0
    print(f"\nInference done: {len(pred_dists)} items in {elapsed:.1f}s  ({n_skipped} skipped)")

    metrics = evaluate_predictions(pred_dists, gt_dists)
    return metrics, pred_dists, gt_dists, greedy_values


def main():
    parser = argparse.ArgumentParser(description="Evaluate Llama3-8B on healthcare val set")
    parser.add_argument("--pretrained", action="store_true",
                        help="Evaluate the pretrained base model (no fine-tuning)")
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                        help="Path to fine-tuned checkpoint dir (e.g. ckpt/llama3_8B/.../epoch_0)")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH,
                        help="Path to base HF model directory")
    parser.add_argument("--val_data", type=str, default=DEFAULT_VAL_DATA,
                        help="Path to val JSON file")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Save full results to this JSON path (optional)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_new_tokens", type=int, default=32,
                        help="Max tokens to generate per item (answer is a single number)")
    parser.add_argument("--mode", choices=["generate", "teacher_force"], default="generate",
                        help="generate: free-form decoding, parse first number token; "
                             "teacher_force: encode prompt+'<', extract distribution at next position")
    parser.add_argument("--bracket_prompt", action="store_true",
                        help="Swap the val item's first paragraph (instruction) for "
                             "BRACKET_INSTRUCTION — matches the prompt the FT bracket "
                             "models saw at training time.")
    parser.add_argument("--verbose", action="store_true",
                        help="Print every item (default: every 50)")
    args = parser.parse_args()

    if not args.pretrained and args.checkpoint_dir is None:
        parser.error("Provide --pretrained or --checkpoint_dir.")

    # Load data
    with open(args.val_data) as f:
        val_data = json.load(f)
    print(f"Loaded {len(val_data)} val items from {args.val_data}")

    # Load model
    if args.pretrained:
        model, tokenizer = load_pretrained_model(args.model_path, device=args.device)
        run_label = "pretrained"
    else:
        model, tokenizer = load_finetuned_model(
            args.checkpoint_dir, args.model_path, device=args.device
        )
        run_label = os.path.normpath(args.checkpoint_dir).replace(os.sep, "_")

    # Evaluate
    metrics, pred_dists, gt_dists, greedy_values = evaluate(
        model, tokenizer, val_data, args.device, args.max_new_tokens, args.verbose,
        mode=args.mode, bracket_prompt=args.bracket_prompt,
    )

    # Print summary
    print("\n" + "=" * 60)
    print(f"Results — {run_label}  [{args.mode}]")
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
        }
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Full results saved to {args.output}")


if __name__ == "__main__":
    main()
