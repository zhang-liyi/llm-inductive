"""
evaluate_bayesian_teaching.py

Evaluate a fine-tuned LLM checkpoint on the Bayesian Teaching benchmark.

Each example is a 3-option choice task (flight / hotel / webshop).
The model sees 4 evidence rounds (options + user's preferred choice) and must
predict the user's preferred option in the 5th (held-out) round.

Answer format: <1>, <2>, or <3>  (angle-bracket wrapped integer, matching
the <x> format used throughout this codebase).
With --abc: the prompt is rewritten so options and feedback use A/B/C and
the model is asked to answer with a bare letter A, B, or C.

Two evaluation modes
--------------------
teacher_force (default)
    Feed the full prompt + ground-truth answer token in one forward pass.
    Extract the logit distribution over {1, 2, 3} (or {A, B, C}) at the
    answer position.  Computes accuracy, CE, and valid_rate.

generate
    Autoregressive generation; parse the answer from output.
    Computes accuracy and valid_rate only.

Usage
-----
    python evaluate_bayesian_teaching.py \\
        --ckpt_dir ../torchtune/ckpt/llama3_8B/<run>/epoch_0 \\
        [--data_path bayesian_teaching_test.jsonl] \\
        [--tasks flight hotel webshop] \\
        [--n_examples 100] \\
        [--mode teacher_force|generate] \\
        [--guided] \\
        [--abc] \\
        [--batch_size 8] \\
        [--max_seq_len 2048] \\
        [--device cuda] \\
        [--dtype bfloat16] \\
        [--output_file ...]
"""

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from torchtune import training
from torchtune.models.llama3 import llama3_8b, lora_llama3_8b
from torchtune.training.checkpointing._checkpointer import FullModelHFCheckpointer


# ── token helpers ──────────────────────────────────────────────────────────────

def get_choice_token_ids(tokenizer) -> Dict[int, int]:
    """
    Return a dict mapping integer choice (1, 2, 3) → its token ID,
    using the same <N> context as the rest of the codebase.

    Tokenises "<1> <2> <3>" and finds the token ID for each digit.
    """
    context = "<1> <2> <3>"
    ctx_ids = tokenizer.encode(context, add_special_tokens=False)

    id_to_str: Dict[int, str] = {}
    for tid in ctx_ids:
        try:
            id_to_str[tid] = tokenizer.decode([tid]).strip()
        except Exception:
            pass

    result = {}
    for n in (1, 2, 3):
        s = str(n)
        found = None
        for tid, decoded in id_to_str.items():
            if decoded == s:
                found = tid
                break
        if found is None:
            ids = tokenizer.encode(" " + s, add_special_tokens=False)
            found = ids[0] if ids else 0
        result[n] = found

    return result   # {1: tok_id, 2: tok_id, 3: tok_id}


def get_abc_token_ids(tokenizer) -> Dict[str, int]:
    """
    Return {'A': tok_id, 'B': tok_id, 'C': tok_id}.

    Uses the '\n\n' prefix that the chat template leaves before the answer
    token, so the returned IDs match what actually gets tokenized when a bare
    letter is appended to the prefix text.
    """
    prefix = "\n\n"
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    result = {}
    for letter in ("A", "B", "C"):
        full_ids = tokenizer.encode(prefix + letter, add_special_tokens=False)
        answer_ids = full_ids[len(prefix_ids):]
        result[letter] = answer_ids[0] if answer_ids else 0
    return result   # {'A': tok_id, 'B': tok_id, 'C': tok_id}


# ── ABC prompt rewriting ───────────────────────────────────────────────────────

_NUM_TO_LETTER = {"1": "A", "2": "B", "3": "C"}
_LETTER_TO_NUM = {"A": 1, "B": 2, "C": 3}


def rewrite_prompt_free_gen(prompt: str) -> str:
    """
    Rewrite a BT prompt to allow reasoning before the final <N> answer.

    Removes the "Output only ... No other text." directive and the "Answer with
    <1>, <2>, or <3> only." closing line, replacing them with instructions that
    require a bracketed integer at the end of the answer.
    """
    prompt = prompt.replace(
        "Output only the number of the best option wrapped in < and >. "
        "For example: <1> or <2> or <3>. No other text.",
        "At the end of your answer, give the number of the best option "
        "wrapped in < and >, like <1> or <2> or <3>.",
    )
    prompt = prompt.replace(
        "Answer with <1>, <2>, or <3> only.",
        "End your answer with the best option choice <1>, <2>, or <3>.",
    )
    return prompt


def remap_prompt_to_abc(prompt: str) -> str:
    """
    Rewrite a numeric-choice prompt so that options and feedback use A/B/C.

    Replacements applied (in order):
      - Instruction line referencing <1>/<2>/<3> → letter equivalents
      - "Option 1/2/3:" → "Option A/B/C:"
      - "preferred option = 1/2/3" → "preferred option = A/B/C"
      - Final answer instruction → A/B/C wording
    """
    # Instruction at the top of the prompt
    prompt = prompt.replace(
        "Output only the number of the best option wrapped in < and >. "
        "For example: <1> or <2> or <3>. No other text.",
        "Output only the letter of the best option: A, B, or C. No other text.",
    )
    # Final question at the bottom
    prompt = prompt.replace(
        "Answer with <1>, <2>, or <3> only.",
        "Answer with A, B, or C only.",
    )
    # Option labels in each round
    for n, l in _NUM_TO_LETTER.items():
        prompt = prompt.replace(f"Option {n}:", f"Option {l}:")
    # User feedback lines
    for n, l in _NUM_TO_LETTER.items():
        prompt = prompt.replace(f"preferred option = {n}", f"preferred option = {l}")
    return prompt


DEFAULT_MODEL_PATH = (
    "<DATA_ROOT>/resources/"
    "models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/"
    "e1945c40cd546c78e41f1151f4db032b271faeaa"
)


# ── model loading ──────────────────────────────────────────────────────────────
# Auto-dispatch on architecture (Llama-3 / Qwen-2) via path inspection.
# Choice tokens {1, 2, 3} are single-token in both vocabularies, so no other
# logic in this script needs Qwen-specific changes.
from qwen2_eval_loaders import (  # noqa: E402
    load_pretrained_model_and_tokenizer as _qw_load_pretrained,
    load_lora_model_and_tokenizer as _qw_load_lora,
    default_pretrained_path as _qw_default_pretrained_path,
)


def load_pretrained_model_and_tokenizer(
    model_path: str = DEFAULT_MODEL_PATH,
    device: str = "cuda",
    dtype: str = "bfloat16",
):
    model, tokenizer, _arch = _qw_load_pretrained(
        model_path, arch='auto', device=device, dtype=dtype)
    return model, tokenizer


def load_model_and_tokenizer(
    ckpt_dir: str,
    device: str = "cuda",
    dtype: str = "bfloat16",
):
    model, tokenizer, _arch = _qw_load_lora(
        ckpt_dir, arch='auto', device=device, dtype=dtype)
    return model, tokenizer


# ── data loading ───────────────────────────────────────────────────────────────

def load_data(
    data_path: str,
    tasks: Optional[List[str]] = None,
    n_examples: Optional[int] = None,
    shuffle_seed: Optional[int] = None,
) -> List[dict]:
    examples = []
    with open(data_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            if tasks and ex.get("task") not in tasks:
                continue
            examples.append(ex)

    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(examples)

    if n_examples is not None:
        examples = examples[:n_examples]

    task_counts = {}
    for ex in examples:
        t = ex.get("task", "unknown")
        task_counts[t] = task_counts.get(t, 0) + 1
    print(f"Loaded {len(examples)} examples: {task_counts}")
    return examples


# ── task-specific instruction injection ───────────────────────────────────────

_TASK_INSTRUCTIONS = {
    "flight": (
        "The choice of a flight depends on 4 features:\n"
        "1. departure time\n"
        "2. duration\n"
        "3. number of stops\n"
        "4. cost\n\n"
        "For each feature, the user's preference weight can be one of 5 levels:\n"
        "1. strongly prefers lower values\n"
        "2. somewhat prefers lower values\n"
        "3. indifferent\n"
        "4. somewhat prefers higher values\n"
        "5. strongly prefers higher values"
    ),
    "hotel": (
        "The choice of a hotel depends on 4 features: distance to downtown, price, "
        "rating, and amenities.\n"
        "Distance and price are numeric features. Rating and amenities are ordinal features.\n\n"
        "For each feature, the user's preference weight can be one of 5 levels:\n"
        "1. strongly prefers lower values\n"
        "2. somewhat prefers lower values\n"
        "3. indifferent\n"
        "4. somewhat prefers higher values\n"
        "5. strongly prefers higher values"
    ),
    "webshop": (
        "A shopping goal consists of: desired attributes, such as \"waterproof\" or "
        "\"soft sole\"; desired options, such as \"color = black and blue\" or "
        "\"size = XL\"; a price limit\n\n"
        "For each feature, the user's preference weight can be one of 5 levels:\n"
        "1. irrelevant\n"
        "2. slightly important\n"
        "3. moderately important\n"
        "4. very important\n"
        "5. essential"
    ),
}

# Sentence after which the task instructions are injected.
_INJECT_AFTER = "The user has fixed preferences they apply consistently."


def inject_task_instructions(prompt: str, task: str) -> str:
    """
    Insert task-specific instructions into *prompt* immediately after the
    sentence "The user has fixed preferences they apply consistently."

    If the anchor sentence is not found, the prompt is returned unchanged.
    """
    instructions = _TASK_INSTRUCTIONS.get(task)
    if not instructions:
        return prompt
    idx = prompt.find(_INJECT_AFTER)
    if idx == -1:
        return prompt
    insert_at = idx + len(_INJECT_AFTER)
    return prompt[:insert_at] + "\n\n" + instructions + "\n\n" + prompt[insert_at:].lstrip(" ")


# ── prompt / tokenisation ──────────────────────────────────────────────────────

def build_chat_prefix(prompt: str, tokenizer) -> str:
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )


def find_answer_position(
    input_ids: torch.Tensor,
    prefix_len: int,
    choice_token_set: set,
    tokenizer,
    valid_chars: Tuple[str, ...] = ("1", "2", "3"),
) -> int:
    """Find the first token in the assistant response that is a choice token."""
    seq = input_ids.tolist()
    for pos in range(prefix_len, len(seq)):
        tok = seq[pos]
        if tok in choice_token_set:
            return pos
        try:
            decoded = tokenizer.decode([tok]).strip()
            if decoded in valid_chars:
                return pos
        except Exception:
            pass
    return prefix_len   # fallback


# ── answer parsing ─────────────────────────────────────────────────────────────

def parse_answer(text: str, abc: bool = False, last: bool = False) -> Optional[int]:
    """
    Extract the 1-indexed choice from generated text.

    If *last* is set, take the last bracketed (or bare) choice instead of the
    first — appropriate for free-generation / reasoning outputs where the final
    answer appears at the end.
    """
    if abc:
        chars = text.strip()
        if last:
            for ch in reversed(chars):
                if ch in "ABC":
                    return _LETTER_TO_NUM[ch]
        else:
            for ch in chars:
                if ch in "ABC":
                    return _LETTER_TO_NUM[ch]
        return None
    # Primary: <N> format
    matches = re.findall(r"<([123])>", text)
    if matches:
        return int(matches[-1] if last else matches[0])
    # Fallback: bare digit
    stripped = text.strip()
    if last:
        for ch in reversed(stripped):
            if ch in "123":
                return int(ch)
    else:
        for ch in stripped:
            if ch in "123":
                return int(ch)
    return None


# ── constrained teacher-forced argmax over {1,2,3} or {A,B,C} ─────────────────

@torch.no_grad()
def force_choice_argmax(
    model,
    context_ids: List[int],
    choice_tensor: torch.Tensor,
    device: str,
    max_seq_len: int = 4096,
) -> int:
    """Run one forward pass over *context_ids* and return the choice (1-indexed)
    whose corresponding token has the highest logit at the final position."""
    if len(context_ids) > max_seq_len:
        context_ids = context_ids[-max_seq_len:]
    inp = torch.tensor([context_ids], dtype=torch.long, device=device)
    logits = model(tokens=inp)
    if isinstance(logits, list):
        logits = torch.cat(logits, dim=1)
    last = logits[0, -1, :]
    choice_logits = last[choice_tensor]
    return int(torch.argmax(choice_logits).item()) + 1


def _find_last_phrase_end(text: str, phrase: str) -> Optional[int]:
    """Return the char index just after the last case-insensitive match of *phrase*."""
    lower = text.lower()
    idx = lower.rfind(phrase.lower())
    if idx < 0:
        return None
    return idx + len(phrase)


# ── teacher-force evaluation ──────────────────────────────────────────────────

@torch.no_grad()
def run_teacher_force_eval(
    model,
    tokenizer,
    data: List[dict],
    choice_token_ids: Dict[int, int],
    device: str,
    batch_size: int = 8,
    max_seq_len: int = 2048,
    guided: bool = False,
    abc: bool = False,
    abc_token_ids: Optional[Dict[str, int]] = None,
) -> List[dict]:
    if abc:
        assert abc_token_ids is not None
        choice_token_set = set(abc_token_ids.values())
        choice_tensor = torch.tensor(
            [abc_token_ids["A"], abc_token_ids["B"], abc_token_ids["C"]],
            dtype=torch.long, device=device,
        )
        valid_chars = ("A", "B", "C")
    else:
        choice_token_set = set(choice_token_ids.values())
        choice_tensor = torch.tensor(
            [choice_token_ids[1], choice_token_ids[2], choice_token_ids[3]],
            dtype=torch.long, device=device,
        )
        valid_chars = ("1", "2", "3")

    results = []
    skipped = 0

    for i in range(0, len(data), batch_size):
        batch = data[i: i + batch_size]

        batch_inputs    = []
        batch_prefix_lens = []
        batch_gt        = []
        batch_meta      = []

        for ex in batch:
            # gt is "<N>" → extract N (always numeric internally)
            gt_str = ex["output"]   # e.g. "<2>"
            m = re.search(r"<([123])>", gt_str)
            if not m:
                continue
            gt = int(m.group(1))   # 1-indexed

            prompt = ex["input"]
            if guided:
                prompt = inject_task_instructions(prompt, ex.get("task", ""))
            if abc:
                prompt = remap_prompt_to_abc(prompt)
            prefix_text = build_chat_prefix(prompt, tokenizer)
            # Append the ground-truth answer token
            answer_tok  = ["A", "B", "C"][gt - 1] if abc else gt_str
            full_text   = prefix_text + answer_tok

            prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
            full_ids   = tokenizer.encode(full_text,   add_special_tokens=False)

            if len(full_ids) > max_seq_len:
                full_ids   = full_ids[:max_seq_len]
                prefix_ids = prefix_ids[:max_seq_len]

            batch_inputs.append(full_ids)
            batch_prefix_lens.append(len(prefix_ids))
            batch_gt.append(gt)
            batch_meta.append(ex.get("metadata", {}))

        if not batch_inputs:
            continue

        # Pad
        max_len = max(len(x) for x in batch_inputs)
        pad_id  = tokenizer.pad_token_id or tokenizer.eos_token_id
        input_ids_t = torch.tensor(
            [ids + [pad_id] * (max_len - len(ids)) for ids in batch_inputs],
            dtype=torch.long, device=device,
        )

        try:
            logits = model(tokens=input_ids_t)
            if isinstance(logits, list):
                logits = torch.cat(logits, dim=1)
        except Exception as exc:
            print(f"  [WARNING] Forward pass failed on batch {i // batch_size}: {exc}")
            skipped += len(batch)
            continue

        for b_idx in range(len(batch_inputs)):
            ids_tensor = torch.tensor(batch_inputs[b_idx], dtype=torch.long)
            prefix_len = batch_prefix_lens[b_idx]
            gt         = batch_gt[b_idx]

            ans_pos   = find_answer_position(
                ids_tensor, prefix_len, choice_token_set, tokenizer, valid_chars
            )
            logit_vec = logits[b_idx, max(0, ans_pos - 1)]  # predict token at ans_pos

            choice_logits = logit_vec[choice_tensor].float()   # [3]
            probs         = F.softmax(choice_logits, dim=0).cpu().numpy()
            pred          = int(np.argmax(probs)) + 1          # 1-indexed

            gt_idx = gt - 1   # 0-indexed for CE
            eps    = 1e-10
            ce     = float(-np.log(probs[gt_idx] + eps))

            results.append({
                "mode":       "teacher_force",
                "task":       data[i + b_idx].get("task"),
                "source":     data[i + b_idx].get("source"),
                "idx":        data[i + b_idx].get("idx"),
                "pred":       pred,
                "gt":         gt,
                "correct":    pred == gt,
                "ce":         ce,
                "probs":      probs.tolist(),
                "ans_pos":    ans_pos,
                "prefix_len": prefix_len,
                "metadata":   batch_meta[b_idx],
            })

        if (i // batch_size) % 10 == 0:
            n_done = min(i + batch_size, len(data))
            print(f"  {n_done}/{len(data)} examples done …", end="\r", flush=True)

    print()
    if skipped:
        print(f"  [WARNING] Skipped {skipped} examples.")
    return results


# ── generation-based evaluation ───────────────────────────────────────────────

@torch.no_grad()
def run_generate_eval(
    model,
    tokenizer,
    data: List[dict],
    device: str,
    max_new_tokens: int = 16,
    max_seq_len: int = 2048,
    guided: bool = False,
    abc: bool = False,
    free_gen: bool = False,
    choice_token_ids: Optional[Dict[int, int]] = None,
    abc_token_ids: Optional[Dict[str, int]] = None,
    checkpoint_path: Optional[str] = None,
    checkpoint_every: int = 25,
) -> List[dict]:
    results = []
    eos_id = tokenizer.eos_token_id

    # Build the choice tensor used by the repair-pass argmax.
    if abc:
        assert abc_token_ids is not None
        choice_tensor = torch.tensor(
            [abc_token_ids["A"], abc_token_ids["B"], abc_token_ids["C"]],
            dtype=torch.long, device=device,
        )
    else:
        assert choice_token_ids is not None
        choice_tensor = torch.tensor(
            [choice_token_ids[1], choice_token_ids[2], choice_token_ids[3]],
            dtype=torch.long, device=device,
        )
    # Suffix appended when no "the answer is" phrase is found in generation.
    repair_suffix_str = "\n\nThe answer is: <" if not abc else "\n\nThe answer is: "

    for i, ex in enumerate(data):
        gt_str = ex["output"]
        m = re.search(r"<([123])>", gt_str)
        if not m:
            continue
        gt = int(m.group(1))

        prompt = ex["input"]
        if guided:
            prompt = inject_task_instructions(prompt, ex.get("task", ""))
        if free_gen:
            prompt = rewrite_prompt_free_gen(prompt)
        if abc:
            prompt = remap_prompt_to_abc(prompt)
        prefix_text = build_chat_prefix(prompt, tokenizer)
        prefix_ids  = tokenizer.encode(prefix_text, add_special_tokens=False)
        if len(prefix_ids) > max_seq_len:
            prefix_ids = prefix_ids[:max_seq_len]

        try:
            generated = torch.tensor([prefix_ids], dtype=torch.long, device=device)
            for _ in range(max_new_tokens):
                logits     = model(tokens=generated)
                if isinstance(logits, list):
                    logits = torch.cat(logits, dim=1)
                next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                generated  = torch.cat([generated, next_token], dim=1)
                if next_token.item() == eos_id:
                    break
            gen_ids  = generated[0, len(prefix_ids):].tolist()
            gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        except Exception as exc:
            print(f"  [WARNING] Generation failed on example {i}: {exc}")
            continue

        pred = parse_answer(gen_text, abc=abc, last=free_gen)

        repair_used = None  # None | "the_answer_is" | "append_suffix"
        repair_text = None
        if pred is None:
            # Repair pass 1: if the model already wrote "the answer is …",
            # truncate generation right after that phrase and force a choice
            # token via a constrained argmax over {1,2,3} (or {A,B,C}).
            phrase_end = _find_last_phrase_end(gen_text, "the answer is")
            if phrase_end is not None:
                truncated = gen_text[:phrase_end]
                if not abc and not truncated.rstrip().endswith("<"):
                    truncated = truncated.rstrip() + " <"
                elif abc and not truncated.endswith(" "):
                    truncated = truncated + " "
                ctx_ids = prefix_ids + tokenizer.encode(
                    truncated, add_special_tokens=False
                )
                try:
                    pred = force_choice_argmax(
                        model, ctx_ids, choice_tensor, device, max_seq_len=max_seq_len,
                    )
                    repair_used = "the_answer_is"
                    repair_text = truncated
                except Exception as exc:
                    print(f"  [WARNING] Repair-1 failed on example {i}: {exc}")

            # Repair pass 2: append an explicit "The answer is: <" suffix and
            # teacher-force the choice token.
            if pred is None:
                suffix_text = gen_text.rstrip() + repair_suffix_str
                ctx_ids = prefix_ids + tokenizer.encode(
                    suffix_text, add_special_tokens=False
                )
                try:
                    pred = force_choice_argmax(
                        model, ctx_ids, choice_tensor, device, max_seq_len=max_seq_len,
                    )
                    repair_used = "append_suffix"
                    repair_text = suffix_text
                except Exception as exc:
                    print(f"  [WARNING] Repair-2 failed on example {i}: {exc}")

        results.append({
            "mode":           "generate",
            "task":           ex.get("task"),
            "source":         ex.get("source"),
            "idx":            ex.get("idx"),
            "pred":           pred,
            "gt":             gt,
            "correct":        pred == gt if pred is not None else False,
            "valid":          pred is not None,
            "generated_text": gen_text,
            "repair_used":    repair_used,
            "repair_text":    repair_text,
            "ce":             float("nan"),
            "metadata":       ex.get("metadata", {}),
        })

        if i % 10 == 0:
            print(f"  {i}/{len(data)} examples done …", end="\r", flush=True)

        if checkpoint_path and (i + 1) % checkpoint_every == 0:
            _summary = aggregate(results)
            ov = _summary["overall"]
            print(f"\n  [checkpoint @ {i+1}] acc={ov['accuracy']:.3f} "
                  f"MAE={ov['mae']:.3f} valid={ov['valid_rate']:.3f} n={ov['n']}",
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
    return results


# ── aggregation ───────────────────────────────────────────────────────────────

def compute_ece(items: List[dict], n_bins: int = 10) -> float:
    """Expected Calibration Error using equal-width bins on max-prob confidence."""
    tf_items = [r for r in items if r.get("probs") is not None]
    if not tf_items:
        return float("nan")
    confidences = np.array([float(np.max(r["probs"])) for r in tf_items])
    corrects    = np.array([float(r["correct"])       for r in tf_items])
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece  = 0.0
    n    = len(tf_items)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidences >= lo) & (confidences <= hi if hi == 1.0 else confidences < hi)
        if mask.sum() == 0:
            continue
        ece += mask.sum() / n * abs(corrects[mask].mean() - confidences[mask].mean())
    return float(ece)


def aggregate(results: List[dict]) -> dict:
    def _stats(items):
        if not items:
            return {
                "n": 0, "accuracy": float("nan"), "ce_mean": float("nan"),
                "mae": float("nan"), "ece": float("nan"), "valid_rate": float("nan"),
            }
        n         = len(items)
        accuracy  = sum(r["correct"] for r in items) / n
        ces       = [r["ce"] for r in items if not (isinstance(r["ce"], float) and np.isnan(r["ce"]))]
        ce_mean   = float(np.mean(ces)) if ces else float("nan")
        mae_vals  = [abs(r["pred"] - r["gt"]) for r in items if r.get("pred") is not None]
        mae       = float(np.mean(mae_vals)) if mae_vals else float("nan")
        ece       = compute_ece(items)
        valid     = [r for r in items if r.get("valid", True)]
        valid_rate = len(valid) / n if "valid" in items[0] else float("nan")
        return {
            "n": n, "accuracy": accuracy, "ce_mean": ce_mean,
            "mae": mae, "ece": ece, "valid_rate": valid_rate,
        }

    overall = _stats(results)
    by_task  = {}
    for task in ("flight", "hotel", "webshop"):
        subset = [r for r in results if r.get("task") == task]
        by_task[task] = _stats(subset)

    return {"overall": overall, "by_task": by_task}


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a fine-tuned LLM on the Bayesian Teaching benchmark."
    )
    parser.add_argument("--ckpt_dir", default=None,
        help="Checkpoint epoch directory.")
    parser.add_argument("--pretrained", action="store_true",
        help="Evaluate the pretrained base model instead of a fine-tuned checkpoint.")
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH,
        help="Path to the base HF model directory (used with --pretrained).")
    parser.add_argument("--data_path",
        default="./data_processing/bayesian_teaching_test.jsonl",
        help="Path to bayesian_teaching_test.jsonl.")
    parser.add_argument("--tasks", nargs="+", default=None,
        choices=["flight", "hotel", "webshop"],
        help="Which tasks to evaluate (default: all).")
    parser.add_argument("--n_examples", type=int, default=None)
    parser.add_argument("--shuffle_seed", type=int, default=None,
        help="If set, randomly shuffle examples with this seed before taking --n_examples.")
    parser.add_argument("--mode", choices=["teacher_force", "generate"],
        default="teacher_force")
    parser.add_argument("--guided", action="store_true",
        help="Inject task-specific feature/preference instructions into each prompt.")
    parser.add_argument("--abc", action="store_true",
        help="Rewrite prompts to use A/B/C labels instead of 1/2/3.")
    parser.add_argument("--free_gen", action="store_true",
        help="With --mode generate, rewrite the prompt to allow reasoning "
             "before the final <N> answer, and parse the last bracketed choice.")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--max_new_tokens", type=int, default=640)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16",
        choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--output_file", default=None)
    parser.add_argument("--checkpoint_every", type=int, default=25,
        help="Generate-mode: write a partial results JSON every N examples.")
    args = parser.parse_args()

    if not args.pretrained and args.ckpt_dir is None:
        parser.error("Provide --ckpt_dir or --pretrained.")

    if args.output_file is None:
        if args.pretrained:
            out_dir = os.path.join(os.path.dirname(__file__), "results", "bayesian_teaching")
            task_tag = "_".join(args.tasks) if args.tasks else "all"
            abc_tag  = "_abc" if args.abc else ""
            args.output_file = os.path.join(
                out_dir,
                f"pretrained_bayesian_teaching_{task_tag}_{args.mode}{abc_tag}.json",
            )
        else:
            task_tag = "_".join(args.tasks) if args.tasks else "all"
            abc_tag  = "_abc" if args.abc else ""
            args.output_file = os.path.join(
                args.ckpt_dir,
                f"bayesian_teaching_eval_{task_tag}_{args.mode}{abc_tag}.json",
            )
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)

    data = load_data(args.data_path, tasks=args.tasks, n_examples=args.n_examples,
                     shuffle_seed=args.shuffle_seed)
    if not data:
        print("No data loaded.")
        sys.exit(1)

    if args.pretrained:
        model, tokenizer = load_pretrained_model_and_tokenizer(args.model_path, args.device, args.dtype)
    else:
        model, tokenizer = load_model_and_tokenizer(args.ckpt_dir, args.device, args.dtype)
    choice_token_ids = get_choice_token_ids(tokenizer)
    print(f"Choice token IDs: {choice_token_ids}")
    abc_token_ids = None
    if args.abc:
        abc_token_ids = get_abc_token_ids(tokenizer)
        print(f"ABC token IDs: {abc_token_ids}")

    if args.guided:
        print("Guided mode: injecting task-specific instructions into prompts.")
    if args.abc:
        print("ABC mode: prompts rewritten to use A/B/C labels.")
    print(f"\nRunning {args.mode!r} evaluation on {len(data)} examples …")
    if args.mode == "teacher_force":
        results = run_teacher_force_eval(
            model, tokenizer, data, choice_token_ids,
            device=args.device, batch_size=args.batch_size, max_seq_len=args.max_seq_len,
            guided=args.guided, abc=args.abc, abc_token_ids=abc_token_ids,
        )
    else:
        checkpoint_path = args.output_file.replace(".json", "_partial.json")
        results = run_generate_eval(
            model, tokenizer, data, device=args.device,
            max_new_tokens=args.max_new_tokens, max_seq_len=args.max_seq_len,
            guided=args.guided, abc=args.abc, free_gen=args.free_gen,
            choice_token_ids=choice_token_ids, abc_token_ids=abc_token_ids,
            checkpoint_path=checkpoint_path,
            checkpoint_every=args.checkpoint_every,
        )

    if not results:
        print("No results.")
        sys.exit(1)

    summary = aggregate(results)

    print("\n=== Bayesian Teaching Evaluation Summary ===")
    ov = summary["overall"]
    print(f"  Mode    : {args.mode}")
    print(f"  Overall : accuracy={ov['accuracy']:.3f}  CE={ov['ce_mean']:.3f}  MAE={ov['mae']:.3f}  ECE={ov['ece']:.3f}  n={ov['n']}")
    for task, s in summary["by_task"].items():
        if s["n"] > 0:
            print(f"  {task:<8}: accuracy={s['accuracy']:.3f}  CE={s['ce_mean']:.3f}  MAE={s['mae']:.3f}  ECE={s['ece']:.3f}  n={s['n']}")

    output = {
        "ckpt_dir":    args.ckpt_dir,
        "data_path":   args.data_path,
        "tasks":       args.tasks,
        "mode":        args.mode,
        "guided":      args.guided,
        "abc":         args.abc,
        "free_gen":    args.free_gen,
        "n_examples":  len(results),
        "summary":     summary,
        "per_example": results,
    }
    with open(args.output_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.output_file}")


if __name__ == "__main__":
    main()
