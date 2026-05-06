"""
evaluate_text_classification_verbalized.py

Verbalized-confidence variant of evaluate_text_classification.py for the
ABC-labelled benchmarks (cls45, chembench, legalbench, MMLU, TruthfulQA,
HellaSwag, ARC-Challenge, Winogrande).

Instead of softmaxing the model's logits over the {A,B,C,...} answer
tokens, we prompt the model to output an integer percentage for *each*
choice in the format
    <A> <a%> <B> <b%> <C> <c%> ... <Z> <z%>
and parse those integers as the model's distribution.  Sequential teacher
forcing: at each slot the parsed value of the previous slot(s) is shown
to the model, then we parse / fall back at the next slot.

Procedure (per example):
1. Decide the number of choices N from the existing ex["input"] prompt
   (count the leading "L) " patterns).
2. Replace the standard PROMPT_INSTRUCTION with a verbalized-format one
   (variable-length, generated from the actual N).
3. Build chat prefix; assistant prefill starts with "<A> <".
4. Greedy-generate up to a few tokens; parse first integer in [0,100].
   Fall back to first-token-argmax over integers 0..100.
5. Append "{val}%> <{next_letter}> <" and repeat.
6. Normalise the N parsed integers; treat as the model's prob distribution.
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

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))
# Reuse loaders, model loading, prompt helpers, ECE.
from evaluate_text_classification import (  # noqa: E402
    LOADERS, CHOICE_LETTERS, compute_ece,
)
from evaluate_bayesian_teaching import (  # noqa: E402
    load_model_and_tokenizer, load_pretrained_model_and_tokenizer,
    build_chat_prefix, DEFAULT_MODEL_PATH,
)
# Reuse the BT verbalized helpers (greedy-generate, integer fallback, parser).
from evaluate_bayesian_teaching_verbalized import (  # noqa: E402
    _greedy_generate, _argmax_integer_0_to_100, _parse_first_number,
)


def count_choices(prompt: str) -> int:
    """Count consecutive `L) ...` choices starting at letter A in *prompt*."""
    n = 0
    for line in prompt.split("\n"):
        s = line.strip()
        # match `<letter>) ` where letter is the next expected one (A, B, ...)
        if len(s) >= 2 and s[1] == ")" and s[0] == CHOICE_LETTERS[n]:
            n += 1
            if n >= len(CHOICE_LETTERS):
                break
    return n


def make_verbalized_instruction(n_choices: int) -> str:
    letters = CHOICE_LETTERS[:n_choices]
    fmt = " ".join(f"<{L}> <{L.lower()}%>" for L in letters)
    return (
        f"Output your probability for each choice being the correct answer, "
        f"in the format {fmt}. The {n_choices} percentages must sum to 100."
    )


@torch.no_grad()
def eval_one(
    model, tokenizer, task: str, ex: dict, device: str,
    max_seq_len: int, max_new_tokens: int,
) -> Optional[dict]:
    out = ex["output"].strip()
    if not out or out[0] not in CHOICE_LETTERS:
        return None
    true_idx = CHOICE_LETTERS.index(out[0])

    prompt = ex["input"]
    n_choices = count_choices(prompt)
    if n_choices < 2:
        return None
    if true_idx >= n_choices:
        # malformed example — gold letter outside the listed choices
        return None

    instruction = make_verbalized_instruction(n_choices)
    user_text = f"{instruction}\n\n{prompt}"
    chat_prefix = build_chat_prefix(user_text, tokenizer)

    parsed: List[float] = []
    parse_modes: List[str] = []
    raw_gens: List[str] = []

    prefill_text = ""
    for k in range(n_choices):
        L = CHOICE_LETTERS[k]
        if k == 0:
            prefill_text += f"<{L}> <"
        else:
            prefill_text += f"%> <{L}> <"

        full_text = chat_prefix + prefill_text
        cur_ids = tokenizer.encode(full_text, add_special_tokens=False)
        if len(cur_ids) > max_seq_len - max_new_tokens:
            cur_ids = cur_ids[-(max_seq_len - max_new_tokens):]

        gen_ids, gen_text = _greedy_generate(
            model, cur_ids, device,
            max_new_tokens=max_new_tokens,
            stop_chars=("%", ">"),
            tokenizer=tokenizer, max_seq_len=max_seq_len,
        )
        val = _parse_first_number(gen_text)
        mode = "gen"
        if val is None:
            val = float(_argmax_integer_0_to_100(
                model, cur_ids, device, tokenizer, max_seq_len=max_seq_len,
            ))
            mode = "argmax"

        rendered = str(int(val)) if float(val).is_integer() else f"{val:g}"
        prefill_text += rendered
        parsed.append(float(val))
        parse_modes.append(mode)
        raw_gens.append(gen_text)

    total = sum(parsed)
    if total == 0:
        probs = [1.0 / n_choices] * n_choices
        valid = False
    else:
        probs = [p / total for p in parsed]
        valid = True
    pred_idx = int(np.argmax(probs))

    return {
        "task": task,
        "n_choices": n_choices,
        "probs": probs,
        "true_idx": true_idx,
        "true_letter": out[0],
        "pred_idx": pred_idx,
        "raw_percents": parsed,
        "parse_modes": parse_modes,
        "raw_gens": raw_gens,
        "valid": valid,
    }


def aggregate(items: List[dict]) -> dict:
    if not items:
        return {"n": 0}
    n = len(items)
    correct = np.array([float(it["pred_idx"] == it["true_idx"]) for it in items])
    # Variable-length probs → compute NLL and confidence per example, then mean.
    nll_vals = []
    confs = []
    mae_vals = []
    for it in items:
        p_true = it["probs"][it["true_idx"]]
        nll_vals.append(-np.log(max(p_true, 1e-12)))
        confs.append(max(it["probs"]))
        mae_vals.append(abs(it["pred_idx"] - it["true_idx"]))
    return {
        "n": n,
        "accuracy": float(correct.mean()),
        "ce_mean": float(np.mean(nll_vals)),
        "mae": float(np.mean(mae_vals)),
        "ece": float(compute_ece(np.array(confs), correct)),
        "valid_rate": float(np.mean([it["valid"] for it in items])),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Verbalized-confidence text-classification evaluation."
    )
    parser.add_argument("--ckpt_dir", default=None)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--dataset", required=True, choices=list(LOADERS.keys()))
    parser.add_argument("--n_examples", type=int, default=None)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--max_seq_len", type=int, default=4096)
    parser.add_argument("--max_new_tokens", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16",
        choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--checkpoint_every", type=int, default=100)
    args = parser.parse_args()

    if not args.pretrained and args.ckpt_dir is None:
        parser.error("Provide --ckpt_dir or --pretrained.")
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)

    examples = LOADERS[args.dataset]()
    examples = examples[args.start_idx:]
    if args.n_examples is not None:
        examples = examples[: args.n_examples]
    print(f"Loaded {len(examples)} examples for {args.dataset}.")

    if args.pretrained:
        model, tokenizer = load_pretrained_model_and_tokenizer(
            args.model_path, args.device, args.dtype)
    else:
        model, tokenizer = load_model_and_tokenizer(
            args.ckpt_dir, args.device, args.dtype)

    items: List[dict] = []
    skipped = 0
    ckpt_path = args.output_file.replace(".json", "_partial.json")
    for i, (task, ex) in enumerate(examples):
        try:
            r = eval_one(model, tokenizer, task, ex, args.device,
                         max_seq_len=args.max_seq_len,
                         max_new_tokens=args.max_new_tokens)
        except Exception as exc:
            print(f"  [WARNING] eval failed on example {i}: {exc}")
            continue
        if r is None:
            skipped += 1
            continue
        items.append(r)
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(examples)} done", flush=True)
        if (i + 1) % args.checkpoint_every == 0:
            with open(ckpt_path, "w") as f:
                json.dump({"summary": {"overall": aggregate(items)},
                           "per_example": items}, f)

    if skipped:
        print(f"  Skipped {skipped} examples (malformed / no choices).")
    if not items:
        print("No results."); sys.exit(1)

    overall = aggregate(items)
    by_task: Dict[str, dict] = {}
    for t in sorted({it["task"] for it in items}):
        by_task[t] = aggregate([it for it in items if it["task"] == t])

    out = {
        "ckpt_dir": args.ckpt_dir or args.model_path,
        "dataset": args.dataset,
        "n_examples": len(items),
        "max_seq_len": args.max_seq_len,
        "max_new_tokens": args.max_new_tokens,
        "summary": {"overall": overall, "by_task": by_task},
        "per_example": items,
    }
    with open(args.output_file, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {args.output_file}")
    print(f"Overall: acc={overall['accuracy']:.4f}  ce={overall['ce_mean']:.4f}  "
          f"ece={overall['ece']:.4f}  n={overall['n']}")


if __name__ == "__main__":
    main()
