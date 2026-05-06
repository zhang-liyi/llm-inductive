"""
evaluate_bayesian_teaching_tf_reasoning.py

Teacher-force-reasoning eval for scratchpad/program-trained models on the
Bayesian Teaching benchmark.

For each BT test example (task, idx) we:
  1. Build the BT user prompt (free-gen rewrite, same as --free_gen generate mode)
  2. Prepend the assistant turn with the ground-truth reasoning text:
       scratchpad: posterior_sampling_pytorch/bt_scratchpads/bt-{task}-{idx}.json
                   (field "scratchpad")
       program:    posterior_sampling_pytorch/bt_programs/pg-bt-{task}-{idx}.py
                   (full file)
  3. Append "\n\nThe answer is: <" to the assistant text
  4. Single forward pass; score logits at the final position over {1,2,3}
     to pick the predicted choice.

This gives an oracle-ish upper bound: does the model produce the right answer
given the target reasoning? (The scratchpad/program includes the final round's
chosen option, so this mostly tests format compliance + calibration.)
"""

import argparse
import json
import os
import random
import re
import sys
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from evaluate_bayesian_teaching import (
    DEFAULT_MODEL_PATH,
    build_chat_prefix,
    compute_ece,
    get_choice_token_ids,
    load_data,
    load_model_and_tokenizer,
    load_pretrained_model_and_tokenizer,
    rewrite_prompt_free_gen,
)


DEFAULT_SCRATCHPADS_DIR = (
    "./"
    "posterior_sampling_pytorch/bt_scratchpads"
)
DEFAULT_PROGRAMS_DIR = (
    "./"
    "posterior_sampling_pytorch/bt_programs"
)


def load_reasoning_text(
    reasoning_mode: str,
    task: str,
    idx: int,
    scratchpads_dir: str,
    programs_dir: str,
    source: Optional[str] = None,
) -> Optional[str]:
    # Filenames: bt-{source}-{idx} (for flight/hotel, source==task;
    # for webshop, source is the sub-category).
    stem = source if source else task
    if task == "webshop":
        scenario_id = f"bt-webshop-{stem}-{idx}" if stem != "webshop" else f"bt-webshop-{idx}"
    else:
        scenario_id = f"bt-{stem}-{idx}"
    if reasoning_mode == "scratchpad":
        path = os.path.join(scratchpads_dir, f"{scenario_id}.json")
        if not os.path.exists(path):
            return None
        with open(path) as f:
            sp = json.load(f)
        text = sp.get("scratchpad", "")
        return text.strip() if text else None
    elif reasoning_mode == "program":
        path = os.path.join(programs_dir, f"pg-{scenario_id}.py")
        if not os.path.exists(path):
            return None
        with open(path) as f:
            return f.read().strip()
    else:
        raise ValueError(f"Unknown reasoning_mode: {reasoning_mode}")


@torch.no_grad()
def run_tf_reasoning_eval(
    model,
    tokenizer,
    data: List[dict],
    choice_token_ids: Dict[int, int],
    device: str,
    reasoning_mode: str,
    scratchpads_dir: str,
    programs_dir: str,
    max_seq_len: int = 10240,
    checkpoint_path: Optional[str] = None,
    checkpoint_every: int = 50,
) -> List[dict]:
    choice_tensor = torch.tensor(
        [choice_token_ids[1], choice_token_ids[2], choice_token_ids[3]],
        dtype=torch.long, device=device,
    )
    # Suffix appended to the reasoning text before the choice token.
    answer_prompt = "\n\nThe answer is: <"

    results = []
    n_skip_no_reasoning = 0
    n_skip_truncated    = 0

    for i, ex in enumerate(data):
        gt_str = ex["output"]
        m = re.search(r"<([123])>", gt_str)
        if not m:
            continue
        gt = int(m.group(1))

        task = ex.get("task")
        idx  = ex.get("idx")
        source = ex.get("source")
        reasoning = load_reasoning_text(
            reasoning_mode, task, idx, scratchpads_dir, programs_dir,
            source=source,
        )
        if not reasoning:
            n_skip_no_reasoning += 1
            continue

        prompt = rewrite_prompt_free_gen(ex["input"])
        prefix_text = build_chat_prefix(prompt, tokenizer)
        # Assistant turn is already opened by add_generation_prompt=True.
        full_text = prefix_text + reasoning + answer_prompt

        full_ids = tokenizer.encode(full_text, add_special_tokens=False)
        if len(full_ids) > max_seq_len:
            n_skip_truncated += 1
            # Truncate from the left to keep the tail (answer prompt) intact.
            full_ids = full_ids[-max_seq_len:]

        try:
            inp = torch.tensor([full_ids], dtype=torch.long, device=device)
            logits = model(tokens=inp)
            if isinstance(logits, list):
                logits = torch.cat(logits, dim=1)
        except Exception as exc:
            print(f"  [WARNING] Forward pass failed on example {i}: {exc}")
            continue

        last = logits[0, -1, :].float()
        choice_logits = last[choice_tensor]
        probs = F.softmax(choice_logits, dim=0).cpu().numpy()
        pred  = int(np.argmax(probs)) + 1

        eps = 1e-10
        ce  = float(-np.log(probs[gt - 1] + eps))

        results.append({
            "mode":      "tf_reasoning",
            "reasoning": reasoning_mode,
            "task":      task,
            "source":    ex.get("source"),
            "idx":       idx,
            "pred":      pred,
            "gt":        gt,
            "correct":   pred == gt,
            "ce":        ce,
            "probs":     probs.tolist(),
            "n_tokens":  len(full_ids),
            "metadata":  ex.get("metadata", {}),
        })

        if i % 10 == 0:
            print(f"  {i}/{len(data)} examples done …", end="\r", flush=True)

        if checkpoint_path and (i + 1) % checkpoint_every == 0:
            _summary = aggregate(results)
            ov = _summary["overall"]
            print(f"\n  [checkpoint @ {i+1}] acc={ov['accuracy']:.3f} "
                  f"CE={ov['ce_mean']:.3f} MAE={ov['mae']:.3f} n={ov['n']}",
                  flush=True)
            try:
                tmp = checkpoint_path + ".tmp"
                with open(tmp, "w") as f:
                    json.dump({"progress": i + 1, "total": len(data),
                               "summary": _summary, "per_example": results}, f)
                os.replace(tmp, checkpoint_path)
            except Exception as exc:
                print(f"  [WARNING] Checkpoint write failed: {exc}")

    print()
    if n_skip_no_reasoning:
        print(f"  [WARNING] Skipped {n_skip_no_reasoning} examples with no reasoning file.")
    if n_skip_truncated:
        print(f"  [WARNING] Left-truncated {n_skip_truncated} examples to fit max_seq_len={max_seq_len}.")
    return results


def aggregate(results: List[dict]) -> dict:
    def _stats(items):
        if not items:
            return {
                "n": 0, "accuracy": float("nan"), "ce_mean": float("nan"),
                "mae": float("nan"), "ece": float("nan"),
            }
        n        = len(items)
        accuracy = sum(r["correct"] for r in items) / n
        ces      = [r["ce"] for r in items]
        ce_mean  = float(np.mean(ces))
        mae_vals = [abs(r["pred"] - r["gt"]) for r in items]
        mae      = float(np.mean(mae_vals))
        ece      = compute_ece(items)
        return {
            "n": n, "accuracy": accuracy, "ce_mean": ce_mean,
            "mae": mae, "ece": ece,
        }

    overall = _stats(results)
    by_task = {}
    for task in ("flight", "hotel", "webshop"):
        subset = [r for r in results if r.get("task") == task]
        by_task[task] = _stats(subset)
    return {"overall": overall, "by_task": by_task}


def main():
    parser = argparse.ArgumentParser(
        description="Teacher-force-reasoning BT eval for scratchpad/program models."
    )
    parser.add_argument("--ckpt_dir", default=None)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data_path",
        default="./data_processing/bayesian_teaching_test_base.jsonl")
    parser.add_argument("--reasoning", choices=["scratchpad", "program"], required=True,
        help="Which reasoning trace to teacher-force.")
    parser.add_argument("--scratchpads_dir", default=DEFAULT_SCRATCHPADS_DIR)
    parser.add_argument("--programs_dir", default=DEFAULT_PROGRAMS_DIR)
    parser.add_argument("--tasks", nargs="+", default=None,
        choices=["flight", "hotel", "webshop"])
    parser.add_argument("--n_examples", type=int, default=None)
    parser.add_argument("--shuffle_seed", type=int, default=None)
    parser.add_argument("--max_seq_len", type=int, default=10240)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16",
        choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--output_file", default=None)
    parser.add_argument("--checkpoint_every", type=int, default=50,
        help="Write a partial results JSON every N examples.")
    args = parser.parse_args()

    if not args.pretrained and args.ckpt_dir is None:
        parser.error("Provide --ckpt_dir or --pretrained.")

    if args.output_file is None:
        task_tag = "_".join(args.tasks) if args.tasks else "all"
        tag = f"bayesian_teaching_eval_{task_tag}_tf_{args.reasoning}.json"
        if args.pretrained:
            out_dir = os.path.join(os.path.dirname(__file__), "results", "bayesian_teaching")
            args.output_file = os.path.join(out_dir, f"pretrained_{tag}")
        else:
            args.output_file = os.path.join(args.ckpt_dir, tag)
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)

    data = load_data(args.data_path, tasks=args.tasks, n_examples=args.n_examples,
                     shuffle_seed=args.shuffle_seed)
    if not data:
        print("No data loaded.")
        sys.exit(1)

    if args.pretrained:
        model, tokenizer = load_pretrained_model_and_tokenizer(
            args.model_path, args.device, args.dtype)
    else:
        model, tokenizer = load_model_and_tokenizer(
            args.ckpt_dir, args.device, args.dtype)
    choice_token_ids = get_choice_token_ids(tokenizer)
    print(f"Choice token IDs: {choice_token_ids}")

    print(f"\nReasoning mode: {args.reasoning}")
    print(f"Running teacher-force-reasoning eval on {len(data)} examples …")
    checkpoint_path = args.output_file.replace(".json", "_partial.json")
    results = run_tf_reasoning_eval(
        model, tokenizer, data, choice_token_ids,
        device=args.device,
        reasoning_mode=args.reasoning,
        scratchpads_dir=args.scratchpads_dir,
        programs_dir=args.programs_dir,
        max_seq_len=args.max_seq_len,
        checkpoint_path=checkpoint_path,
        checkpoint_every=args.checkpoint_every,
    )
    if not results:
        print("No results.")
        sys.exit(1)

    summary = aggregate(results)

    print("\n=== BT Teacher-Force-Reasoning Summary ===")
    ov = summary["overall"]
    print(f"  Reasoning: {args.reasoning}")
    print(f"  Overall  : accuracy={ov['accuracy']:.3f}  CE={ov['ce_mean']:.3f}  "
          f"MAE={ov['mae']:.3f}  ECE={ov['ece']:.3f}  n={ov['n']}")
    for task, s in summary["by_task"].items():
        if s["n"] > 0:
            print(f"  {task:<8}: accuracy={s['accuracy']:.3f}  CE={s['ce_mean']:.3f}  "
                  f"MAE={s['mae']:.3f}  ECE={s['ece']:.3f}  n={s['n']}")

    output = {
        "ckpt_dir":    args.ckpt_dir,
        "data_path":   args.data_path,
        "reasoning":   args.reasoning,
        "tasks":       args.tasks,
        "n_examples":  args.n_examples,
        "max_seq_len": args.max_seq_len,
        "summary":     summary,
        "per_example": results,
    }
    with open(args.output_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nWrote {args.output_file}")


if __name__ == "__main__":
    main()
