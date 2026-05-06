#!/usr/bin/env python3
"""
evaluate_openestimate_pretrained.py

Evaluate the off-the-shelf (unmodified) Llama 3-8B Instruct model on the
OpenEstimate benchmark.  Identical pipeline to evaluate_openestimate.py
except the model is loaded directly from the base model directory using
llama3_8b() + FullModelHFCheckpointer, with no LoRA adapters.

Usage
-----
    python evaluate_openestimate_pretrained.py \\
        [--model_path /path/to/Meta-Llama-3-8B-Instruct] \\
        [--data_path openestimate_test.json] \\
        [--split dev|test|all] \\
        [--n_examples 50] \\
        [--mode teacher_force|generate] \\
        [--batch_size 4] \\
        [--max_seq_len 1024] \\
        [--device cuda] \\
        [--dtype bfloat16] \\
        [--output_file openestimate_eval_pretrained.json]
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from torchtune import training
from torchtune.models.llama3 import llama3_8b
from torchtune.training.checkpointing._checkpointer import FullModelHFCheckpointer

# ── constants ─────────────────────────────────────────────────────────────────

DEFAULT_MODEL_PATH = (
    "<DATA_ROOT>/resources/"
    "models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/"
    "e1945c40cd546c78e41f1151f4db032b271faeaa"
)

_SPLIT_DEV  = "dev"
_SPLIT_TEST = "test"
_SPLIT_ALL  = "all"


# ── answer parser ─────────────────────────────────────────────────────────────

def parse_answer(text: str) -> Optional[int]:
    text = text.strip()
    if not text:
        return None
    for tag in ("mean", "answer", "value", "estimate", "result"):
        m = re.search(
            rf"<{tag}>\s*([0-9]+(?:\.[0-9]+)?)\s*</{tag}>",
            text,
            re.IGNORECASE,
        )
        if m:
            val = float(m.group(1))
            return max(0, min(100, int(round(val))))
    candidates = [
        int(x)
        for x in re.findall(r"\b([0-9]{1,3})\b", text)
        if 0 <= int(x) <= 100
    ]
    return candidates[-1] if candidates else None


# ── tokenizer helpers ─────────────────────────────────────────────────────────

def get_number_token_ids(tokenizer) -> torch.Tensor:
    context = "[" + ", ".join(str(i) for i in range(101)) + "]"
    ctx_ids = tokenizer.encode(context, add_special_tokens=False)
    id_to_str: Dict[int, str] = {}
    for tid in ctx_ids:
        try:
            id_to_str[tid] = tokenizer.decode([tid]).strip()
        except Exception:
            pass
    number_token_ids = []
    for n in range(101):
        s = str(n)
        found = None
        for tid, decoded in id_to_str.items():
            if decoded == s:
                found = tid
                break
        if found is None:
            ids = tokenizer.encode(" " + s, add_special_tokens=False)
            found = ids[0] if ids else 0
        number_token_ids.append(found)
    return torch.tensor(number_token_ids, dtype=torch.long)


# ── model loading ─────────────────────────────────────────────────────────────

def load_model_and_tokenizer(
    model_path: str = DEFAULT_MODEL_PATH,
    device: str = "cuda",
    dtype: str = "bfloat16",
) -> Tuple[object, AutoTokenizer]:
    """
    Load the base Llama 3-8B Instruct model using FullModelHFCheckpointer +
    llama3_8b(), following the same pattern as custom_lora_answer_only.py and
    inference_benchmarks.py.
    """
    torch_dtype = {
        "bfloat16": torch.bfloat16,
        "float16":  torch.float16,
        "float32":  torch.float32,
    }.get(dtype, torch.bfloat16)

    print(f"Loading tokenizer from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model from {model_path}  (dtype={dtype}, device={device})")
    ckpt_files = sorted([
        f for f in os.listdir(model_path)
        if f.endswith('.safetensors')
    ])
    if not ckpt_files:
        raise FileNotFoundError(f"No safetensors files found in {model_path}")
    print(f"  Checkpoint files: {ckpt_files}")

    checkpointer = FullModelHFCheckpointer(
        checkpoint_dir=model_path,
        checkpoint_files=ckpt_files,
        model_type="LLAMA3",
        output_dir=os.path.join(model_path, os.pardir),
    )
    checkpoint_dict = checkpointer.load_checkpoint()

    with training.set_default_dtype(torch_dtype), torch.device(device):
        model = llama3_8b()

    missing, unexpected = model.load_state_dict(
        checkpoint_dict[training.MODEL_KEY], strict=True
    )
    if missing:
        print(f"  Missing keys: {missing}")
    if unexpected:
        print(f"  Unexpected keys: {unexpected}")

    model.eval()
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
    return model, tokenizer


# ── data loading / splitting ──────────────────────────────────────────────────

def load_data(
    data_path: str,
    split: str = _SPLIT_ALL,
    n_examples: Optional[int] = None,
) -> List[dict]:
    with open(data_path) as f:
        data = json.load(f)
    if split == _SPLIT_DEV:
        data = [ex for i, ex in enumerate(data) if i % 2 == 0]
    elif split == _SPLIT_TEST:
        data = [ex for i, ex in enumerate(data) if i % 2 == 1]
    if n_examples is not None:
        data = data[:n_examples]
    print(f"Loaded {len(data)} examples  (split={split!r})")
    return data


# ── prompt helper ─────────────────────────────────────────────────────────────

def build_chat_prefix(prompt: str, tokenizer) -> str:
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


# ── answer-position finder ────────────────────────────────────────────────────

def find_answer_positions(
    input_ids: torch.Tensor,
    prefix_len: int,
    number_token_set: set,
    tokenizer,
    n: int = 2,
) -> list:
    """Return the first n numeric-answer token positions after prefix_len."""
    seq = input_ids.tolist()
    positions = []
    for pos in range(prefix_len, len(seq)):
        if len(positions) >= n:
            break
        tok = seq[pos]
        if tok in number_token_set:
            positions.append(pos)
            continue
        try:
            decoded = tokenizer.decode([tok]).strip()
            if decoded.isdigit() and 0 <= int(decoded) <= 100:
                positions.append(pos)
        except Exception:
            pass
    fallback = positions[-1] if positions else prefix_len
    while len(positions) < n:
        positions.append(fallback)
    return positions


# ── per-example metric computation ────────────────────────────────────────────

def metrics_at_position(
    mean_logit_vec: torch.Tensor,
    std_logit_vec: torch.Tensor,
    gt_bins: List[float],
    gt_std: float,
    number_token_ids: torch.Tensor,
    device: str,
) -> dict:
    number_token_ids = number_token_ids.to(device)
    gt = np.array(gt_bins, dtype=np.float64)
    values = np.arange(101, dtype=np.float64)
    gt_mean = float(np.dot(values, gt))
    gt_mode = int(np.argmax(gt))
    gt_mean_int = max(0, min(100, int(round(gt_mean))))
    eps = 1e-10

    number_logits = mean_logit_vec[number_token_ids].float()
    pred_probs    = F.softmax(number_logits, dim=0).cpu().numpy().astype(np.float64)
    pred_mean     = float(np.dot(values, pred_probs))
    pred_mode     = int(np.argmax(pred_probs))

    ce_dist = float(-np.sum(gt * np.log(pred_probs + eps)))
    ce_mean = float(-np.log(pred_probs[gt_mean_int] + eps))

    std_number_logits = std_logit_vec[number_token_ids].float()
    pred_std = float(torch.argmax(std_number_logits).item())

    return {
        "ce_mean":   ce_mean,
        "ce_dist":   ce_dist,
        "mae":       abs(pred_mean - gt_mean),
        "mae_std":   abs(pred_std - gt_std),
        "pred_mean": pred_mean,
        "gt_mean":   gt_mean,
        "pred_std":  pred_std,
        "gt_std":    gt_std,
        "pred_mode": pred_mode,
        "gt_mode":   gt_mode,
        "pred_dist": pred_probs.tolist(),
    }


# ── trivial baseline (always predict 50 with probability 1) ──────────────────

def run_trivial_eval(data: List[dict]) -> List[dict]:
    """Baseline that always places all probability mass on answer = 50."""
    eps = 1e-10
    values = np.arange(101, dtype=np.float64)
    pred_probs = np.zeros(101, dtype=np.float64)
    pred_probs[50] = 1.0

    results = []
    for ex in data:
        gt_bins = np.array(ex["bins"][0], dtype=np.float64)
        gt_mean = float(np.dot(values, gt_bins))
        gt_mean_int = max(0, min(100, int(round(gt_mean))))

        ce_dist = float(-np.sum(gt_bins * np.log(pred_probs + eps)))
        ce_mean = float(-np.log(pred_probs[gt_mean_int] + eps))

        results.append({
            "mode":      "trivial",
            "ce_mean":   ce_mean,
            "ce_dist":   ce_dist,
            "mae":       abs(50.0 - gt_mean),
            "pred_mean": 50.0,
            "gt_mean":   gt_mean,
            "pred_mode": 50,
            "gt_mode":   int(np.argmax(gt_bins)),
            "pred_dist": pred_probs.tolist(),
            "metadata":  ex.get("metadata", {}),
        })
    return results


# ── teacher-force evaluation ──────────────────────────────────────────────────

@torch.no_grad()
def run_teacher_force_eval(
    model,
    tokenizer,
    data: List[dict],
    number_token_ids: torch.Tensor,
    device: str,
    batch_size: int = 4,
    max_seq_len: int = 1024,
) -> List[dict]:
    number_token_set = set(number_token_ids.tolist())
    results = []
    skipped = 0

    for i in range(0, len(data), batch_size):
        batch = data[i : i + batch_size]

        batch_inputs      = []
        batch_gt_bins     = []
        batch_prefix_lens = []
        batch_meta        = []

        for ex in batch:
            prompt  = ex["input"]
            answer  = ex["output"]
            gt_bins = ex["bins"][0]

            prefix_text = build_chat_prefix(prompt, tokenizer)
            full_text   = prefix_text + answer

            prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
            full_ids   = tokenizer.encode(full_text,   add_special_tokens=False)

            if len(full_ids) > max_seq_len:
                full_ids   = full_ids[:max_seq_len]
                prefix_ids = prefix_ids[:max_seq_len]

            batch_inputs.append(full_ids)
            batch_gt_bins.append(gt_bins)
            batch_prefix_lens.append(len(prefix_ids))
            batch_meta.append(ex.get("metadata", {}))

        max_len = max(len(x) for x in batch_inputs)
        pad_id  = tokenizer.pad_token_id or tokenizer.eos_token_id

        input_ids_list = []
        for ids in batch_inputs:
            pad_len = max_len - len(ids)
            input_ids_list.append(ids + [pad_id] * pad_len)

        input_ids_t = torch.tensor(input_ids_list, dtype=torch.long, device=device)

        try:
            logits = model(tokens=input_ids_t)
            if isinstance(logits, list):
                logits = torch.cat(logits, dim=1)
        except Exception as exc:
            print(f"  [WARNING] Forward pass failed on batch {i//batch_size}: {exc}")
            skipped += len(batch)
            continue

        for b_idx in range(len(batch)):
            ids_tensor = torch.tensor(batch_inputs[b_idx], dtype=torch.long)
            meta       = batch_meta[b_idx]

            ans_positions = find_answer_positions(
                ids_tensor, batch_prefix_lens[b_idx], number_token_set, tokenizer, n=2
            )
            mean_pos, std_pos = ans_positions[0], ans_positions[1]

            mean_logit_vec = logits[b_idx, max(0, mean_pos - 1)]
            std_logit_vec  = logits[b_idx, max(0, std_pos  - 1)]

            gt_std = float(meta.get("normalised_std", 0.0))

            metrics = metrics_at_position(
                mean_logit_vec, std_logit_vec,
                batch_gt_bins[b_idx], gt_std,
                number_token_ids, device,
            )
            metrics["mode"]       = "teacher_force"
            metrics["metadata"]   = meta
            metrics["mean_pos"]   = mean_pos
            metrics["std_pos"]    = std_pos
            metrics["prefix_len"] = batch_prefix_lens[b_idx]
            results.append(metrics)

        if (i // batch_size) % 10 == 0:
            n_done = min(i + batch_size, len(data))
            print(f"  {n_done}/{len(data)} examples done …", end="\r", flush=True)

    print()
    if skipped:
        print(f"  [WARNING] Skipped {skipped} examples due to forward-pass errors.")
    return results


# ── generation-based evaluation ───────────────────────────────────────────────

@torch.no_grad()
def run_generate_eval(
    model,
    tokenizer,
    data: List[dict],
    number_token_ids: torch.Tensor,
    device: str,
    max_new_tokens: int = 64,
    max_seq_len: int = 1024,
) -> List[dict]:
    results = []
    eos_id  = tokenizer.eos_token_id

    for i, ex in enumerate(data):
        prompt  = ex["input"]
        gt_bins = ex["bins"][0]
        gt_mean = sum(j * gt_bins[j] for j in range(101))

        prefix_text = build_chat_prefix(prompt, tokenizer)
        prefix_ids  = tokenizer.encode(prefix_text, add_special_tokens=False)
        if len(prefix_ids) > max_seq_len:
            prefix_ids = prefix_ids[:max_seq_len]

        try:
            generated = torch.tensor([prefix_ids], dtype=torch.long, device=device)
            for _ in range(max_new_tokens):
                logits = model(tokens=generated)
                if isinstance(logits, list):
                    logits = torch.cat(logits, dim=1)
                next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                generated  = torch.cat([generated, next_token], dim=1)
                if next_token.item() == eos_id:
                    break
            generated_ids  = generated[0, len(prefix_ids):].tolist()
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        except Exception as exc:
            print(f"  [WARNING] Generation failed on example {i}: {exc}")
            continue

        parsed    = parse_answer(generated_text)
        pred_mean = float(parsed) if parsed is not None else float("nan")

        results.append({
            "mode":           "generate",
            "generated_text": generated_text,
            "parsed_answer":  parsed,
            "pred_mean":      pred_mean,
            "gt_mean":        gt_mean,
            "mae":            abs(pred_mean - gt_mean) if parsed is not None else float("nan"),
            "ce_mean":        float("nan"),
            "ce_dist":        float("nan"),
            "metadata":       ex.get("metadata", {}),
        })

        if i % 10 == 0:
            print(f"  {i}/{len(data)} examples done …", end="\r", flush=True)

    print()
    return results


# ── metric aggregation ────────────────────────────────────────────────────────

def aggregate_metrics(results: List[dict]) -> dict:
    def _safe_mean(vals):
        valid = [v for v in vals if not (isinstance(v, float) and np.isnan(v))]
        return float(np.mean(valid)) if valid else float("nan")

    def _safe_std(vals):
        valid = [v for v in vals if not (isinstance(v, float) and np.isnan(v))]
        return float(np.std(valid)) if len(valid) > 1 else float("nan")

    def _safe_median(vals):
        valid = [v for v in vals if not (isinstance(v, float) and np.isnan(v))]
        return float(np.median(valid)) if valid else float("nan")

    scalar_keys = ("ce_mean", "ce_dist", "mae", "mae_std")
    agg: dict = {"n": len(results)}

    for k in scalar_keys:
        vals = [r[k] for r in results if k in r]
        agg[k] = {
            "mean":    _safe_mean(vals),
            "std":     _safe_std(vals),
            "median":  _safe_median(vals),
            "n_valid": sum(1 for v in vals if not (isinstance(v, float) and np.isnan(v))),
        }

    def _group_stats(group_key: str, value_key: str) -> dict:
        groups: dict = {}
        for r in results:
            label = r.get("metadata", {}).get(group_key, "unknown")
            val   = r.get(value_key)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                continue
            groups.setdefault(label, []).append(val)
        return {grp: {"mean": _safe_mean(vs), "n": len(vs)} for grp, vs in sorted(groups.items())}

    agg["by_dataset"]    = _group_stats("dataset",           "mae")
    agg["by_difficulty"] = _group_stats("difficulty",        "mae")
    agg["by_dist_type"]  = _group_stats("distribution_type", "mae")

    return agg


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the off-the-shelf Llama 3-8B Instruct on OpenEstimate."
    )
    parser.add_argument(
        "--model_path",
        default=DEFAULT_MODEL_PATH,
        help=f"Path to the base Llama 3-8B Instruct directory  [default: {DEFAULT_MODEL_PATH}].",
    )
    parser.add_argument(
        "--data_path",
        default=os.path.join(os.path.dirname(__file__), "openestimate_test.json"),
        help="Path to openestimate_test.json  [default: same folder as this script].",
    )
    parser.add_argument(
        "--split", choices=["dev", "test", "all"], default="all",
        help="Which portion of the data to evaluate.  dev=even-indexed examples, "
             "test=odd-indexed examples, all=everything  [default: all].",
    )
    parser.add_argument(
        "--n_examples", type=int, default=None,
        help="Limit evaluation to the first N examples after splitting (for quick runs).",
    )
    parser.add_argument(
        "--mode", choices=["teacher_force", "generate"], default="teacher_force",
        help="Evaluation mode.  teacher_force gives CE+MAE; generate gives MAE only  "
             "[default: teacher_force].",
    )
    parser.add_argument(
        "--batch_size", type=int, default=4,
        help="Batch size for teacher-force mode  [default: 4].",
    )
    parser.add_argument(
        "--max_seq_len", type=int, default=1024,
        help="Maximum token sequence length; longer inputs are truncated  [default: 1024].",
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=64,
        help="Maximum new tokens to generate (generate mode only)  [default: 64].",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on  [default: cuda if available].",
    )
    parser.add_argument(
        "--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"],
        help="Model dtype  [default: bfloat16].",
    )
    parser.add_argument(
        "--output_file",
        default=None,
        help="Path for the JSON results file.  "
             "Defaults to results/openestimate_eval_pretrained_<split>_<mode>.json.",
    )
    args = parser.parse_args()

    # ── output file ───────────────────────────────────────────────────────────
    if args.output_file is None:
        args.output_file = os.path.join(
            os.path.dirname(__file__),
            "results",
            f"openestimate_eval_pretrained_{args.split}_{args.mode}.json",
        )
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)

    # ── load data ─────────────────────────────────────────────────────────────
    data = load_data(args.data_path, split=args.split, n_examples=args.n_examples)
    if not data:
        print("No data loaded — check --data_path and --split.")
        sys.exit(1)

    # ── load model ────────────────────────────────────────────────────────────
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.device, args.dtype)
    number_token_ids = get_number_token_ids(tokenizer)
    print(f"Number token IDs (sample 0-5): {number_token_ids[:6].tolist()}")

    # ── run evaluation ────────────────────────────────────────────────────────
    print(f"\nRunning {args.mode!r} evaluation on {len(data)} examples …")
    if args.mode == "teacher_force":
        results = run_teacher_force_eval(
            model, tokenizer, data, number_token_ids,
            device=args.device,
            batch_size=args.batch_size,
            max_seq_len=args.max_seq_len,
        )
    else:
        results = run_generate_eval(
            model, tokenizer, data, number_token_ids,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            max_seq_len=args.max_seq_len,
        )

    if not results:
        print("No results produced — evaluation may have failed entirely.")
        sys.exit(1)

    # ── trivial baseline ──────────────────────────────────────────────────────
    print("\nComputing trivial baseline (always predict 50) …")
    trivial_results = run_trivial_eval(data)
    trivial_summary = aggregate_metrics(trivial_results)

    # ── aggregate ─────────────────────────────────────────────────────────────
    summary = aggregate_metrics(results)

    def _print_summary(label, s, mode):
        print(f"\n=== {label} ===")
        print(f"  MAE (mean) : {s['mae']['mean']:.3f} ± {s['mae']['std']:.3f}  (median {s['mae']['median']:.3f})")
        print(f"  MAE (std)  : {s['mae_std']['mean']:.3f} ± {s['mae_std']['std']:.3f}  (median {s['mae_std']['median']:.3f})  [explicit model output]")
        if mode == "teacher_force":
            print(f"  CE_mean    : {s['ce_mean']['mean']:.3f} ± {s['ce_mean']['std']:.3f}  [at mean answer position]")
            print(f"  CE_dist    : {s['ce_dist']['mean']:.3f} ± {s['ce_dist']['std']:.3f}  [at mean answer position]")
        print("  MAE by dataset:")
        for ds, st in s["by_dataset"].items():
            print(f"    {ds:<12} {st['mean']:.3f}  (n={st['n']})")
        print("  MAE by difficulty:")
        for diff, st in s["by_difficulty"].items():
            print(f"    {diff:<12} {st['mean']:.3f}  (n={st['n']})")
        print("  MAE by distribution type:")
        for dt, st in s["by_dist_type"].items():
            print(f"    {dt:<12} {st['mean']:.3f}  (n={st['n']})")

    _print_summary(
        f"Pretrained Llama 3-8B Instruct  [mode={args.mode}, split={args.split}, n={len(results)}]",
        summary, args.mode,
    )
    _print_summary(
        f"Trivial baseline (always predict 50)  [n={len(trivial_results)}]",
        trivial_summary, "teacher_force",
    )

    # ── save ──────────────────────────────────────────────────────────────────
    output = {
        "model_path": args.model_path,
        "data_path":  args.data_path,
        "split":      args.split,
        "mode":       args.mode,
        "n_examples": len(results),
        "pretrained": {
            "summary":     summary,
            "per_example": results,
        },
        "trivial": {
            "summary":     trivial_summary,
            "per_example": trivial_results,
        },
    }
    with open(args.output_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.output_file}")


if __name__ == "__main__":
    main()
