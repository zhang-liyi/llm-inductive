"""
evaluate_bayesian_teaching_verbalized.py

Verbalized-confidence evaluation for the BT 123 benchmark.

Instead of softmaxing the model's logits over the {1,2,3} answer tokens, we
prompt the model to *output* a probability for each option in the form
    <1> <x%> <2> <y%> <3> <z%>
and then parse x, y, z as integer percentages.  These are normalised to sum
to 1 and treated as the model's probability distribution over the three
options.  Accuracy and ECE are then computed in the same way as the
teacher-forced eval.

Procedure for one example
-------------------------
1.  Replace the original output instruction in the prompt with one that
    asks for the <i> <p%> format.
2.  Build the chat prefix and append the assistant prefill `<1> <`.
3.  Greedy-generate up to a few tokens; parse the first integer in [0,100].
    If the parse fails, fall back to scoring every integer 0..100 at the
    same position (joint log-prob of its tokenisation) and take the argmax.
4.  Replace the generated tokens with the *parsed* value (teacher-forcing
    the prior outputs), append `%> <2> <`, and repeat.
5.  Same for option 3 with prefix `%> <3> <`.
6.  Normalise (x, y, z) -> (px, py, pz).  If they all parse to 0, fall
    back to a uniform distribution.

Usage
-----
    python evaluate_bayesian_teaching_verbalized.py \\
        --ckpt_dir <hf-model-dir> \\
        [--data_path bayesian_teaching_test_base.jsonl] \\
        [--guided] \\
        [--output_file ...]
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

# Re-use loaders, prompt helpers and aggregation from the standard BT eval.
_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))
from evaluate_bayesian_teaching import (  # noqa: E402
    load_model_and_tokenizer,
    load_pretrained_model_and_tokenizer,
    build_chat_prefix,
    inject_task_instructions,
    load_data,
    aggregate,
    DEFAULT_MODEL_PATH,
)


# Original instructions present in bayesian_teaching_test_base.jsonl.  We
# replace BOTH to avoid a contradictory "answer with one option" line at the
# end of the prompt.
_ORIG_INSTRUCTION = (
    "Output only the number of the best option wrapped in < and >. "
    "For example: <1> or <2> or <3>. No other text."
)
_ORIG_TRAILING = (
    "Which option does the user prefer? Answer with <1>, <2>, or <3> only."
)
_VERBALIZED_INSTRUCTION = (
    "Output your probability for each option being the correct answer, in "
    "the format <1> <x%> <2> <y%> <3> <z%>. The three percentages must sum "
    "to 100."
)
_VERBALIZED_TRAILING = (
    "Output your probability for each option being the correct answer, in "
    "the format <1> <x%> <2> <y%> <3> <z%>."
)


def modify_prompt_for_verbalized(prompt: str) -> str:
    out = prompt
    if _ORIG_INSTRUCTION in out:
        out = out.replace(_ORIG_INSTRUCTION, _VERBALIZED_INSTRUCTION)
    else:
        out = out.rstrip() + "\n\n" + _VERBALIZED_INSTRUCTION
    if _ORIG_TRAILING in out:
        out = out.replace(_ORIG_TRAILING, _VERBALIZED_TRAILING)
    return out


def _parse_first_number(text: str) -> Optional[float]:
    """Return the first non-negative number (int or float) in *text*, or None.
    Values are clamped into [0, 100]."""
    m = re.search(r"\d+(?:\.\d+)?|\.\d+", text)
    if not m:
        return None
    try:
        v = float(m.group(0))
    except ValueError:
        return None
    if v < 0:
        return None
    return min(v, 100.0)


# ── core: greedy generate then optional argmax-fallback ───────────────────────

@torch.no_grad()
def _greedy_generate(
    model, context_ids: List[int], device: str,
    max_new_tokens: int, stop_chars: Tuple[str, ...], tokenizer,
    max_seq_len: int = 4096,
) -> Tuple[List[int], str]:
    """Greedy-generate up to *max_new_tokens* after *context_ids*. Stop early
    once any character in *stop_chars* appears in the decoded suffix."""
    if len(context_ids) > max_seq_len - max_new_tokens:
        context_ids = context_ids[-(max_seq_len - max_new_tokens):]
    inp = torch.tensor([context_ids], dtype=torch.long, device=device)
    gen_ids: List[int] = []
    for _ in range(max_new_tokens):
        logits = model(tokens=inp)
        if isinstance(logits, list):
            logits = torch.cat(logits, dim=1)
        next_tok = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        tid = int(next_tok.item())
        gen_ids.append(tid)
        inp = torch.cat([inp, next_tok], dim=1)
        partial = tokenizer.decode(gen_ids, skip_special_tokens=True)
        if any(c in partial for c in stop_chars):
            break
    return gen_ids, tokenizer.decode(gen_ids, skip_special_tokens=True)


@torch.no_grad()
def _argmax_integer_0_to_100(
    model, context_ids: List[int], device: str, tokenizer,
    max_seq_len: int = 4096,
) -> int:
    """Fallback (parse failed): at the next position after *context_ids*,
    score each integer 0..100 and return the argmax.  In the Llama-3
    tokenizer all of "0".."100" are exactly one token each, so a single
    forward pass on the context gives an exact score from
    ``logits[last][token_id_for(str(i))]``."""
    if len(context_ids) > max_seq_len:
        context_ids = context_ids[-max_seq_len:]
    inp = torch.tensor([context_ids], dtype=torch.long, device=device)
    logits = model(tokens=inp)
    if isinstance(logits, list):
        logits = torch.cat(logits, dim=1)
    last_logits = logits[0, -1, :].float()
    log_p = F.log_softmax(last_logits, dim=-1)

    int_token_ids = []
    for i in range(0, 101):
        ids = tokenizer.encode(str(i), add_special_tokens=False)
        # Llama-3: each integer 0..100 is a single token. Defensive guard.
        int_token_ids.append(ids[0] if ids else 0)
    int_tensor = torch.tensor(int_token_ids, dtype=torch.long, device=device)
    return int(torch.argmax(log_p[int_tensor]).item())


# ── per-example pipeline ──────────────────────────────────────────────────────

@torch.no_grad()
def eval_one(
    model, tokenizer, ex: dict, device: str,
    guided: bool, max_seq_len: int, max_new_tokens: int,
) -> Optional[dict]:
    gt_str = ex["output"]
    m = re.search(r"<([123])>", gt_str)
    if not m:
        return None
    gt = int(m.group(1))

    prompt = ex["input"]
    if guided:
        prompt = inject_task_instructions(prompt, ex.get("task", ""))
    prompt = modify_prompt_for_verbalized(prompt)
    prefix_text = build_chat_prefix(prompt, tokenizer)
    prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
    if len(prefix_ids) > max_seq_len - 64:
        prefix_ids = prefix_ids[-(max_seq_len - 64):]

    parsed: List[float] = []      # x, y, z (allowed to be float)
    parse_modes: List[str] = []   # "gen" | "argmax"
    raw_gens: List[str] = []      # what the model actually emitted at each slot

    # We accumulate the assistant's response as a *string* and re-encode the
    # full chat each iteration. Re-encoding (rather than concatenating token
    # ids) avoids BPE-merge drift across boundaries.
    prefill_text = ""
    for opt_idx, choice in enumerate([1, 2, 3]):
        if opt_idx == 0:
            prefill_text += f"<{choice}> <"
        else:
            # close the previous slot ("...%> ") and open this one
            prefill_text += f"%> <{choice}> <"

        full_text = prefix_text + prefill_text
        cur_ids = tokenizer.encode(full_text, add_special_tokens=False)
        if len(cur_ids) > max_seq_len - max_new_tokens:
            cur_ids = cur_ids[-(max_seq_len - max_new_tokens):]

        # Greedy generate (stop on '%' or '>')
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

        # Teacher-force the parsed value into the running prefill.
        # Render ints without trailing .0; floats keep their digits.
        rendered = str(int(val)) if float(val).is_integer() else f"{val:g}"
        prefill_text += rendered
        parsed.append(float(val))
        parse_modes.append(mode)
        raw_gens.append(gen_text)

    x, y, z = parsed
    total = x + y + z
    if total == 0:
        probs = [1.0 / 3] * 3
        valid = False
    else:
        probs = [x / total, y / total, z / total]
        valid = True
    pred = int(np.argmax(probs)) + 1
    eps = 1e-10
    ce = float(-np.log(probs[gt - 1] + eps))

    return {
        "mode": "verbalized",
        "task": ex.get("task"),
        "source": ex.get("source"),
        "idx": ex.get("idx"),
        "pred": pred,
        "gt": gt,
        "correct": pred == gt,
        "ce": ce,
        "probs": probs,
        "raw_percents": parsed,
        "parse_modes": parse_modes,
        "raw_gens": raw_gens,
        "valid": valid,
        "metadata": ex.get("metadata", {}),
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Verbalized-confidence BT evaluation."
    )
    parser.add_argument("--ckpt_dir", default=None)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data_path",
        default="./data_processing/bayesian_teaching_test_base.jsonl")
    parser.add_argument("--tasks", nargs="+", default=None,
        choices=["flight", "hotel", "webshop"])
    parser.add_argument("--n_examples", type=int, default=None)
    parser.add_argument("--shuffle_seed", type=int, default=None)
    parser.add_argument("--guided", action="store_true")
    parser.add_argument("--max_seq_len", type=int, default=4096)
    parser.add_argument("--max_new_tokens", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16",
        choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--checkpoint_every", type=int, default=50)
    args = parser.parse_args()

    if not args.pretrained and args.ckpt_dir is None:
        parser.error("Provide --ckpt_dir or --pretrained.")

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)

    data = load_data(args.data_path, tasks=args.tasks,
                     n_examples=args.n_examples, shuffle_seed=args.shuffle_seed)
    if not data:
        print("No data loaded."); sys.exit(1)

    if args.pretrained:
        model, tokenizer = load_pretrained_model_and_tokenizer(
            args.model_path, args.device, args.dtype)
    else:
        model, tokenizer = load_model_and_tokenizer(
            args.ckpt_dir, args.device, args.dtype)

    if args.guided:
        print("Guided mode: injecting task-specific instructions.")
    print(f"Running verbalized-confidence eval on {len(data)} examples …")

    results: List[dict] = []
    ckpt_path = args.output_file.replace(".json", "_partial.json")
    for i, ex in enumerate(data):
        try:
            r = eval_one(
                model, tokenizer, ex, args.device,
                guided=args.guided,
                max_seq_len=args.max_seq_len,
                max_new_tokens=args.max_new_tokens,
            )
        except Exception as exc:
            print(f"  [WARNING] eval failed on example {i}: {exc}")
            continue
        if r is None:
            continue
        results.append(r)
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(data)} done", flush=True)
        if (i + 1) % args.checkpoint_every == 0:
            with open(ckpt_path, "w") as f:
                json.dump({"summary": aggregate(results),
                           "per_example": results}, f)

    if not results:
        print("No results."); sys.exit(1)

    summary = aggregate(results)
    out = {
        "ckpt_dir": args.ckpt_dir or args.model_path,
        "data_path": args.data_path,
        "guided": args.guided,
        "n_examples": len(results),
        "max_seq_len": args.max_seq_len,
        "max_new_tokens": args.max_new_tokens,
        "summary": summary,
        "per_example": results,
    }
    with open(args.output_file, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {args.output_file}")
    print(f"Overall: acc={summary['overall']['accuracy']:.4f}  "
          f"ce={summary['overall']['ce_mean']:.4f}  "
          f"ece={summary['overall']['ece']:.4f}  "
          f"n={summary['overall']['n']}")


if __name__ == "__main__":
    main()
