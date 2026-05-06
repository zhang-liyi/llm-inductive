#!/usr/bin/env python3
"""
evaluate_openestimate.py

Evaluate a fine-tuned LLM checkpoint on the OpenEstimate benchmark.

Metrics are computed *only at the numeric answer position*, ignoring any
verbal-reasoning tokens that precede or follow the answer.

Two evaluation modes
--------------------
teacher_force (default)
    Feed the full prompt + ground-truth answer through the model in a single
    forward pass.  At the position where the model predicts the answer token,
    extract the logit distribution over numbers 0-100 and compute:
      • CE_mean  – cross-entropy against the GT mean token only
      • CE_dist  – cross-entropy against the full 101-bin GT distribution
      • MAE      – |predicted_mean − GT_mean|  (on 0-100 scale)
    This is the fastest option and yields the full distributional metrics.

generate
    Run autoregressive generation, then *parse* the integer answer from the
    generated text with a robust regex that skips over reasoning prose.
    Computes MAE only (CE requires the full logit distribution).

Usage
-----
    python evaluate_openestimate.py \\
        --ckpt_dir ../torchtune/ckpt/llama3_8B/<run>/epoch_0 \\
        --data_path openestimate_test.json \\
        [--split dev|test|all]   \\   # default: all
        [--n_examples 50]         \\   # limit for quick runs
        [--mode teacher_force|generate]  \\
        [--batch_size 4]          \\
        [--max_seq_len 1024]      \\
        [--device cuda]           \\
        [--dtype bfloat16]        \\
        [--output_file openestimate_eval.json]
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
from torchtune.models.llama3 import llama3_8b, lora_llama3_8b
from torchtune.training.checkpointing._checkpointer import FullModelHFCheckpointer

DEFAULT_MODEL_PATH = (
    "<DATA_ROOT>/resources/"
    "models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/"
    "e1945c40cd546c78e41f1151f4db032b271faeaa"
)

# ── constants ─────────────────────────────────────────────────────────────────

# Deterministic dev/test split: even indices → dev, odd indices → test.
# ("all" skips the filter.)
_SPLIT_DEV  = "dev"
_SPLIT_TEST = "test"
_SPLIT_ALL  = "all"


# ── answer parser ─────────────────────────────────────────────────────────────

def parse_answer(text: str) -> Optional[int]:
    """
    Extract the 0-100 integer answer from model output, ignoring reasoning prose.

    Handles these output formats (in priority order):
      1. XML tags: <mean>42</mean>, <answer>42</answer>
      2. Plain last integer 0-100 anywhere in the text

    The *last* valid integer is taken as the final answer (model's conclusion
    usually appears at the end after chain-of-thought reasoning).
    """
    text = text.strip()
    if not text:
        return None

    # 1. XML-style tags used in the OpenEstimate elicitation format
    for tag in ("mean", "answer", "value", "estimate", "result"):
        m = re.search(
            rf"<{tag}>\s*([0-9]+(?:\.[0-9]+)?)\s*</{tag}>",
            text,
            re.IGNORECASE,
        )
        if m:
            val = float(m.group(1))
            return max(0, min(100, int(round(val))))

    # 2. All integers 0-100 in the text; take the last one
    candidates = [
        int(x)
        for x in re.findall(r"\b([0-9]{1,3})\b", text)
        if 0 <= int(x) <= 100
    ]
    return candidates[-1] if candidates else None


# ── tokenizer helpers ─────────────────────────────────────────────────────────

def get_number_token_ids(tokenizer) -> torch.Tensor:
    """
    Return a [101] tensor mapping integer i → its representative token ID.

    Uses the same heuristic as the training code: tokenise the string
    "<0> <1> ... <100>" and find the single token that decodes to each
    digit string.  Falls back to direct encoding if not found in context.
    """
    context = " ".join(f"<{i}>" for i in range(101))
    ctx_ids = tokenizer.encode(context, add_special_tokens=False)

    # Build decoded-token lookup for fast search
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
        # Prefer a token that appears in context and decodes exactly to s
        for tid, decoded in id_to_str.items():
            if decoded == s:
                found = tid
                break
        if found is None:
            # Fallback: encode directly (space-prefixed for consistency)
            ids = tokenizer.encode(" " + s, add_special_tokens=False)
            found = ids[0] if ids else 0
        number_token_ids.append(found)

    return torch.tensor(number_token_ids, dtype=torch.long)


# ── model / tokenizer loading (auto-dispatch on arch via path) ────────────────
from qwen2_eval_loaders import (  # noqa: E402
    load_pretrained_model_and_tokenizer as _qw_load_pretrained,
    load_lora_model_and_tokenizer as _qw_load_lora,
    setup_number_tokens as _setup_number_tokens,
    infer_arch_from_path,
)


def load_pretrained_model_and_tokenizer(
    model_path: str = DEFAULT_MODEL_PATH,
    device: str = "cuda",
    dtype: str = "bfloat16",
) -> Tuple[object, AutoTokenizer]:
    model, tokenizer, _arch = _qw_load_pretrained(
        model_path, arch='auto', device=device, dtype=dtype)
    return model, tokenizer


def load_model_and_tokenizer(
    ckpt_dir: str,
    device: str = "cuda",
    dtype: str = "bfloat16",
) -> Tuple[object, AutoTokenizer]:
    """Auto-dispatch on architecture (Llama-3 / Qwen-2)."""
    model, tokenizer, _arch = _qw_load_lora(
        ckpt_dir, arch='auto', device=device, dtype=dtype)
    return model, tokenizer


# ── data loading / splitting ──────────────────────────────────────────────────

def load_data(
    data_path: str,
    split: str = _SPLIT_ALL,
    n_examples: Optional[int] = None,
) -> List[dict]:
    """
    Load openestimate_test.json and optionally subset to dev / test split.

    Split strategy (deterministic, no randomness):
      dev  → even-indexed examples   (indices 0, 2, 4, …)
      test → odd-indexed examples    (indices 1, 3, 5, …)
      all  → every example
    """
    with open(data_path) as f:
        data = json.load(f)

    if split == _SPLIT_DEV:
        data = [ex for i, ex in enumerate(data) if i % 2 == 0]
    elif split == _SPLIT_TEST:
        data = [ex for i, ex in enumerate(data) if i % 2 == 1]
    # else: all

    if n_examples is not None:
        data = data[:n_examples]

    print(f"Loaded {len(data)} examples  (split={split!r})")
    return data


# ── prompt / tokenisation helpers ─────────────────────────────────────────────

def build_chat_prefix(prompt: str, tokenizer) -> str:
    """
    Return the text of the chat-formatted user turn with the assistant header
    appended (i.e. everything up to and including the '\n\n' after the header).
    This is the model's context for prediction.
    """
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,   # appends <|start_header_id|>assistant…
    )


# ── standard prompt rewriting ─────────────────────────────────────────────────

_ORIG_INSTRUCTION = (
    "Answer the query and return two integers each wrapped in < and >, "
    "separated by a space. For example, <mean> <std>. The first integer "
    "is your mean estimate. The second integer is your standard deviation "
    "estimate on the same scale."
)

_STRICT_INSTRUCTION = (
    "Answer the query and return only two integers each wrapped in < and >, "
    "separated by a space. For example, <mean> <std>. The first integer "
    "is your mean estimate. The second integer is your standard deviation "
    "estimate on the same scale. "
    "IMPORTANT: use the 0-100 scale as shown in the instruction."
)


def rewrite_prompt_strict(prompt: str) -> str:
    """Tighten the answer-only instruction with 'only' and a scale reminder."""
    if _ORIG_INSTRUCTION in prompt:
        return prompt.replace(_ORIG_INSTRUCTION, _STRICT_INSTRUCTION)
    return prompt


# ── free-generation prompt rewriting ──────────────────────────────────────────

_FREE_GEN_INSTRUCTION = (
    "Answer the query. At the end of your answer, return two integers each "
    "wrapped in < and >, separated by a space, like <mean> <std>. The first "
    "integer is your mean estimate. The second integer is your standard "
    "deviation estimate on the same scale. "
    "IMPORTANT: use two angle brackets <mean> <std>, not <mean std>. "
    "Use the 0-100 scale, as shown in the instruction."
)


def rewrite_prompt_free_gen(prompt: str) -> str:
    """Replace the answer-only instruction with a version that permits reasoning."""
    if _ORIG_INSTRUCTION in prompt:
        return prompt.replace(_ORIG_INSTRUCTION, _FREE_GEN_INSTRUCTION)
    return _FREE_GEN_INSTRUCTION + "\n\n" + prompt


def find_answer_positions(
    input_ids: torch.Tensor,       # [seq_len]
    prefix_len: int,
    number_token_set: set,
    tokenizer,
    n: int = 2,
) -> list:
    """
    Return the sequence positions of the first *n* numeric-answer tokens (0-100)
    in the assistant's response (mean, then std).

    Scans forward from *prefix_len* looking for tokens whose ID is in
    *number_token_set* or that decode to a digit string 0-100.

    Returns a list of up to *n* positions; missing positions are filled with
    the last found position (or prefix_len as a fallback).
    """
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
    # Pad to length n using the last found position or prefix_len
    fallback = positions[-1] if positions else prefix_len
    while len(positions) < n:
        positions.append(fallback)
    return positions


# ── span detection (multi-token) ──────────────────────────────────────────────

def find_digit_span(input_ids, start_pos, digit_token_set, max_len=3):
    """Walk forward from start_pos collecting positions whose token is a
    digit (0-9). Stops at the first non-digit position (e.g. '>'). Returns
    list of length 1..max_len."""
    seq = input_ids.tolist() if hasattr(input_ids, 'tolist') else list(input_ids)
    span = [start_pos]
    p = start_pos + 1
    while len(span) < max_len and p < len(seq) and seq[p] in digit_token_set:
        span.append(p)
        p += 1
    return span


# ── per-example metric computation ────────────────────────────────────────────

def metrics_at_span_multitoken(
    logits: torch.Tensor,            # [seq_len, vocab_size]
    mean_span: list, std_span: list,
    gt_bins: list, gt_std: float,
    digit_token_ids: list, number_token_seqs: list,
    device: str,
) -> dict:
    """Multi-token analogue of metrics_at_position: works when integers
    0..100 are NOT all single-token (e.g. Qwen-2 has 10..100 multi-token).

    pred_mean uses predict_integer_from_span(method='expected') over span-
    matching candidates (multi-token product, conditional approximation
    under teacher forcing — same approximation used in training-time eval).
    pred_mode uses predict_integer_from_span(method='greedy').
    ce_dist is the position-decomposed CE: marginal first-digit CE +
    conditional second-digit (and third-digit if span length 3) CE,
    matching evaluate_multitoken_metrics in the recipe.
    """
    import sys as _sys
    _sys.path.insert(0, "./torchtune")
    from probabilistic_reasoning_utils import (
        predict_integer_from_span, _build_first_second_third_idx,
    )

    digit_t = torch.tensor(digit_token_ids, dtype=torch.long, device=device)
    first_idx, second_idx, third_idx = _build_first_second_third_idx(
        number_token_seqs, digit_token_ids
    )
    first_idx_t = torch.tensor(first_idx, dtype=torch.long, device=device)
    second_idx_t = torch.tensor(second_idx, dtype=torch.long, device=device)
    third_idx_t = torch.tensor(third_idx, dtype=torch.long, device=device)
    digit_to_idx = {tid: idx for idx, tid in enumerate(digit_token_ids)}

    gt = np.array(gt_bins, dtype=np.float64)
    values = np.arange(101, dtype=np.float64)
    gt_mean = float(np.dot(values, gt))
    gt_mode = int(np.argmax(gt))
    gt_mean_int = max(0, min(100, int(round(gt_mean))))

    # ── pred_mean (expected over multi-token candidates) ──────────────────────
    pred_mean = float(predict_integer_from_span(
        logits, mean_span, digit_token_ids, method='expected',
        number_token_seqs=number_token_seqs,
    ))
    pred_mode = int(predict_integer_from_span(
        logits, mean_span, digit_token_ids, method='greedy',
        number_token_seqs=number_token_seqs,
    ))

    # ── pred_std (greedy integer over std_span) ───────────────────────────────
    pred_std = int(predict_integer_from_span(
        logits, std_span, digit_token_ids, method='greedy',
        number_token_seqs=number_token_seqs,
    ))

    # ── ce_dist: position-decomposed (first marginal + conditional next) ──────
    gt_t = torch.tensor(gt, dtype=torch.float64, device=device)
    # Marginal first-digit target
    target_0 = torch.zeros(10, dtype=torch.float64, device=device)
    valid = first_idx_t >= 0
    if valid.any():
        target_0.index_add_(0, first_idx_t[valid], gt_t[valid])
    lp0 = F.log_softmax(logits[mean_span[0], digit_t].float(), dim=-1).double()
    ce_pos0 = float(-(target_0 * lp0).sum().item()) if target_0.sum() > 1e-12 else 0.0
    # Conditional second digit (use observed first digit at mean_span[0])
    ce_pos1 = ce_pos2 = 0.0
    if len(mean_span) >= 2:
        gt_d0 = digit_to_idx.get(int(torch.argmax(logits[mean_span[0], digit_t]).item()), -1)
        # NB: at TF, the "observed" first digit *should* be the one that was
        # forced. Read it from input ids if available; here we use argmax
        # since this metric mirrors training-time decomposition.
        if gt_d0 >= 0:
            target_1 = torch.zeros(10, dtype=torch.float64, device=device)
            m1 = (first_idx_t == gt_d0) & (second_idx_t >= 0)
            if m1.any():
                target_1.index_add_(0, second_idx_t[m1], gt_t[m1])
            if target_1.sum() > 1e-12:
                target_1 = target_1 / target_1.sum()
                lp1 = F.log_softmax(logits[mean_span[1], digit_t].float(), dim=-1).double()
                ce_pos1 = float(-(target_1 * lp1).sum().item())
    if len(mean_span) >= 3:
        gt_d0 = digit_to_idx.get(int(torch.argmax(logits[mean_span[0], digit_t]).item()), -1)
        gt_d1 = digit_to_idx.get(int(torch.argmax(logits[mean_span[1], digit_t]).item()), -1)
        if gt_d0 >= 0 and gt_d1 >= 0:
            target_2 = torch.zeros(10, dtype=torch.float64, device=device)
            m2 = ((first_idx_t == gt_d0) & (second_idx_t == gt_d1)
                  & (third_idx_t >= 0))
            if m2.any():
                target_2.index_add_(0, third_idx_t[m2], gt_t[m2])
            if target_2.sum() > 1e-12:
                target_2 = target_2 / target_2.sum()
                lp2 = F.log_softmax(logits[mean_span[2], digit_t].float(), dim=-1).double()
                ce_pos2 = float(-(target_2 * lp2).sum().item())
    ce_dist = ce_pos0 + ce_pos1 + ce_pos2

    # ── ce_mean: NLL of round(gt_mean) under multi-token product ──────────────
    seq_gt = number_token_seqs[gt_mean_int]
    eps = 1e-10
    if len(seq_gt) > len(mean_span):
        # Span is shorter than GT integer's tokenization — extremely
        # rare; just clamp.
        ce_mean = 50.0
    else:
        log_p = 0.0
        for k, tok in enumerate(seq_gt):
            if tok not in digit_to_idx:
                log_p = float('nan')
                break
            lp_k = F.log_softmax(logits[mean_span[k], digit_t].float(), dim=-1).double()
            log_p += float(lp_k[digit_to_idx[tok]].item())
        ce_mean = -log_p if not np.isnan(log_p) else 50.0

    return {
        "ce_mean":   ce_mean,
        "ce_dist":   ce_dist,
        "mae":       abs(pred_mean - gt_mean),
        "mae_std":   abs(float(pred_std) - gt_std),
        "pred_mean": pred_mean,
        "gt_mean":   gt_mean,
        "pred_std":  float(pred_std),
        "gt_std":    gt_std,
        "pred_mode": pred_mode,
        "gt_mode":   gt_mode,
        # No 101-bin pred_dist for multi-token (not directly available);
        # downstream aggregator (aggregate_pyrorej_all.py) only reads
        # `mae` and `ce_mean` from OE results, so this is fine.
        "pred_dist": [],
    }

def metrics_at_position(
    mean_logit_vec: torch.Tensor,   # [vocab_size]  logits at mean answer position
    std_logit_vec: torch.Tensor,    # [vocab_size]  logits at std answer position
    gt_bins: List[float],           # 101 floats summing to 1
    gt_std: float,                  # ground truth normalised std (from metadata)
    number_token_ids: torch.Tensor, # [101]
    device: str,
) -> dict:
    """
    Compute CE_mean, CE_dist, MAE (mean), and MAE_std (explicit std output) for one query.

    Mean metrics use the full softmax distribution over 0-100 at the mean position.
    Std MAE compares the model's greedy std token against the GT normalised_std.

    Returns dict with keys:
        ce_mean   – cross-entropy of mean prediction against the GT mean token
        ce_dist   – cross-entropy of mean prediction against the full 101-bin GT distribution
        mae       – |predicted_mean − GT_mean|
        mae_std   – |predicted_std − GT_std|  (explicit model output vs GT norm_std)
        pred_mean, gt_mean, pred_std, gt_std, pred_mode, gt_mode
        pred_dist – list[float] of 101 predicted probabilities (mean position)
    """
    number_token_ids = number_token_ids.to(device)
    gt = np.array(gt_bins, dtype=np.float64)
    values = np.arange(101, dtype=np.float64)
    gt_mean = float(np.dot(values, gt))
    gt_mode = int(np.argmax(gt))
    gt_mean_int = max(0, min(100, int(round(gt_mean))))
    eps = 1e-10

    # ── Mean position: full distributional metrics ────────────────────────────
    number_logits = mean_logit_vec[number_token_ids].float()   # [101]
    pred_probs    = F.softmax(number_logits, dim=0).cpu().numpy().astype(np.float64)
    pred_mean     = float(np.dot(values, pred_probs))
    pred_mode     = int(np.argmax(pred_probs))

    ce_dist = float(-np.sum(gt * np.log(pred_probs + eps)))
    ce_mean = float(-np.log(pred_probs[gt_mean_int] + eps))

    # ── Std position: greedy token as explicit integer prediction ─────────────
    std_number_logits = std_logit_vec[number_token_ids].float()
    pred_std = float(torch.argmax(std_number_logits).item())   # greedy integer 0-100

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
    nt_helpers: Optional[dict] = None,
) -> List[dict]:
    """
    Teacher-forced evaluation: single forward pass per example (or batch).

    For each example:
      1. Build full token sequence: chat_prefix + answer_tokens
      2. Forward pass → logits
      3. Locate the first numeric-answer token in the assistant response
      4. Compute CE_mean, CE_dist, MAE from the logits at that position

    Returns a list of per-example result dicts.
    """
    number_token_set = set(number_token_ids.tolist())
    multitoken = (nt_helpers is not None) and (not nt_helpers.get('single_token', True))
    digit_token_set = set(nt_helpers['digit_token_ids']) if multitoken else None
    results = []
    skipped = 0

    for i in range(0, len(data), batch_size):
        batch = data[i : i + batch_size]

        # ── build inputs ──────────────────────────────────────────────────────
        batch_inputs   = []
        batch_gt_bins  = []
        batch_gt_means = []
        batch_prefix_lens = []
        batch_meta     = []

        for ex in batch:
            prompt = rewrite_prompt_strict(ex["input"])
            answer = ex["output"]   # e.g. "<50> <17>" (mean std)
            gt_bins = ex["bins"][0]  # list of 101 floats

            prefix_text = build_chat_prefix(prompt, tokenizer)
            full_text   = prefix_text + answer

            prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
            full_ids   = tokenizer.encode(full_text,   add_special_tokens=False)

            if len(full_ids) > max_seq_len:
                full_ids   = full_ids[:max_seq_len]
                prefix_ids = prefix_ids[:max_seq_len]

            batch_inputs.append(full_ids)
            batch_gt_bins.append(gt_bins)
            batch_gt_means.append(sum(j * gt_bins[j] for j in range(101)))
            batch_prefix_lens.append(len(prefix_ids))
            batch_meta.append(ex.get("metadata", {}))
            # gt_std comes from metadata normalised_std (population std on 0-100 scale)

        # ── pad + forward pass ────────────────────────────────────────────────
        max_len = max(len(x) for x in batch_inputs)
        pad_id  = tokenizer.pad_token_id or tokenizer.eos_token_id

        input_ids_list = []
        for ids in batch_inputs:
            pad_len = max_len - len(ids)
            input_ids_list.append(ids + [pad_id] * pad_len)

        input_ids_t = torch.tensor(input_ids_list, dtype=torch.long, device=device)

        try:
            logits = model(tokens=input_ids_t)   # torchtune: returns logits directly
            if isinstance(logits, list):
                logits = torch.cat(logits, dim=1)
            # logits: [B, seq_len, vocab_size]
        except Exception as exc:
            print(f"  [WARNING] Forward pass failed on batch {i//batch_size}: {exc}")
            skipped += len(batch)
            continue

        # ── extract metrics per example ───────────────────────────────────────
        for b_idx in range(len(batch)):
            ids_for_ex = batch_inputs[b_idx]
            prefix_len = batch_prefix_lens[b_idx]
            meta       = batch_meta[b_idx]

            ids_tensor = torch.tensor(ids_for_ex, dtype=torch.long)
            gt_std = float(meta.get("normalised_std", 0.0))

            if multitoken:
                # For multi-token tokenizers, locate first-digit positions
                # and walk forward through consecutive digit tokens to get
                # the full multi-digit span. find_answer_positions already
                # finds the FIRST digit token (since digit tokens are in
                # number_token_set's first-token fallback).
                ans_positions = find_answer_positions(
                    ids_tensor, prefix_len, number_token_set, tokenizer, n=2,
                )
                mean_pos, std_pos = ans_positions[0], ans_positions[1]
                mean_span = find_digit_span(ids_for_ex, mean_pos, digit_token_set)
                std_span  = find_digit_span(ids_for_ex, std_pos,  digit_token_set)
                # Logits at position (start-1) ... (end-1) predict the
                # tokens at positions [start..end] respectively.
                # We need logits AT the answer positions (since logits[pos]
                # predicts the token at pos+1 in causal LMs — but
                # `predict_integer_from_span` reads logits[span[k]] as the
                # logit predicting the token at position span[k]+1, NOT
                # the one already there). Adjust by -1.
                mean_span_lh = [max(0, p - 1) for p in mean_span]
                std_span_lh  = [max(0, p - 1) for p in std_span]
                metrics = metrics_at_span_multitoken(
                    logits[b_idx], mean_span_lh, std_span_lh,
                    batch_gt_bins[b_idx], gt_std,
                    nt_helpers['digit_token_ids'],
                    nt_helpers['number_token_seqs'],
                    device,
                )
                metrics['mean_pos'] = mean_pos
                metrics['std_pos']  = std_pos
                metrics['mean_span'] = mean_span
                metrics['std_span']  = std_span
            else:
                # Single-token (Llama) — original logic unchanged.
                ans_positions = find_answer_positions(
                    ids_tensor, prefix_len, number_token_set, tokenizer, n=2
                )
                mean_pos, std_pos = ans_positions[0], ans_positions[1]
                mean_logit_vec = logits[b_idx, max(0, mean_pos - 1)]
                std_logit_vec  = logits[b_idx, max(0, std_pos  - 1)]
                metrics = metrics_at_position(
                    mean_logit_vec, std_logit_vec,
                    batch_gt_bins[b_idx], gt_std,
                    number_token_ids, device,
                )
                metrics["mean_pos"] = mean_pos
                metrics["std_pos"]  = std_pos
            metrics["mode"]       = "teacher_force"
            metrics["metadata"]   = meta
            metrics["prefix_len"] = prefix_len
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
    free_gen: bool = False,
) -> List[dict]:
    """
    Generation-based evaluation: generate, then parse the integer answer.

    Computes MAE only (CE requires full logit distribution and is not well
    defined for arbitrary generation positions).

    If *free_gen* is set, the prompt is rewritten to allow reasoning before the
    final answer, and parsing takes the *last* <mean> <std> pair rather than
    the first two integers.
    """
    results = []
    eos_id = tokenizer.eos_token_id

    for i, ex in enumerate(data):
        prompt  = ex["input"]
        if free_gen:
            prompt = rewrite_prompt_free_gen(prompt)
        else:
            prompt = rewrite_prompt_strict(prompt)
        gt_bins = ex["bins"][0]
        gt_mean = sum(j * gt_bins[j] for j in range(101))
        gt_std  = float(ex.get("metadata", {}).get("normalised_std", 0.0))

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
                generated = torch.cat([generated, next_token], dim=1)
                if next_token.item() == eos_id:
                    break
            generated_ids = generated[0, len(prefix_ids):].tolist()
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        except Exception as exc:
            print(f"  [WARNING] Generation failed on example {i}: {exc}")
            continue

        if free_gen:
            parsed_mean = None
            parsed_std  = None

            # 1. Two separate brackets: <mean> <std> (integers or decimals)
            bracket_ints = [
                int(round(float(x)))
                for x in re.findall(r"<([0-9]{1,3}(?:\.[0-9]+)?)>", generated_text)
                if 0 <= float(x) <= 100
            ]
            if len(bracket_ints) >= 2:
                parsed_mean = bracket_ints[-2]
                parsed_std  = bracket_ints[-1]
            elif len(bracket_ints) == 1:
                parsed_mean = bracket_ints[-1]

            # 2. Single bracket with two numbers: <mean std> or <mean, std>
            if parsed_mean is None or parsed_std is None:
                pair_matches = re.findall(
                    r"<\s*([0-9]+(?:\.[0-9]+)?)[,\s]+([0-9]+(?:\.[0-9]+)?)\s*>",
                    generated_text,
                )
                if pair_matches:
                    m_str, s_str = pair_matches[-1]
                    m_val = int(round(float(m_str)))
                    s_val = int(round(float(s_str)))
                    if 0 <= m_val <= 100:
                        parsed_mean = m_val
                    if 0 <= s_val <= 100:
                        parsed_std = s_val

            # 3. Fallback: last two plain integers 0-100 in the text.
            if parsed_mean is None:
                all_ints = [
                    int(x) for x in re.findall(r"\b([0-9]{1,3})\b", generated_text)
                    if 0 <= int(x) <= 100
                ]
                parsed_mean = all_ints[-2] if len(all_ints) >= 2 else (all_ints[-1] if all_ints else None)
                if parsed_std is None:
                    parsed_std = all_ints[-1] if len(all_ints) >= 2 else None
        else:
            # Original: first two integers are mean, std.
            all_ints = [
                int(x) for x in re.findall(r"\b([0-9]{1,3})\b", generated_text)
                if 0 <= int(x) <= 100
            ]
            parsed_mean = all_ints[0]  if len(all_ints) >= 1 else None
            parsed_std  = all_ints[1]  if len(all_ints) >= 2 else None
        pred_mean = float(parsed_mean) if parsed_mean is not None else float("nan")
        pred_std  = float(parsed_std)  if parsed_std  is not None else float("nan")

        result = {
            "mode":           "generate",
            "generated_text": generated_text,
            "parsed_mean":    parsed_mean,
            "parsed_std":     parsed_std,
            "pred_mean":      pred_mean,
            "pred_std":       pred_std,
            "gt_mean":        gt_mean,
            "gt_std":         gt_std,
            "mae":            abs(pred_mean - gt_mean) if parsed_mean is not None else float("nan"),
            "mae_std":        abs(pred_std  - gt_std)  if parsed_std  is not None else float("nan"),
            "ce_mean":        float("nan"),
            "ce_dist":        float("nan"),
            "metadata":       ex.get("metadata", {}),
        }
        results.append(result)

        if i % 10 == 0:
            print(f"  {i}/{len(data)} examples done …", end="\r", flush=True)

    print()
    return results


# ── metric aggregation ────────────────────────────────────────────────────────

def aggregate_metrics(results: List[dict]) -> dict:
    """
    Compute aggregate statistics from per-example results.

    Returns a summary dict with mean / std / median for each scalar metric,
    plus breakdowns by dataset, difficulty, and distribution type.
    """
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
    # mae     = MAE on mean estimate
    # mae_std = MAE on explicit std estimate (model's second output integer)

    agg: dict = {"n": len(results)}

    for k in scalar_keys:
        vals = [r[k] for r in results if k in r]
        agg[k] = {
            "mean":   _safe_mean(vals),
            "std":    _safe_std(vals),
            "median": _safe_median(vals),
            "n_valid": sum(1 for v in vals if not (isinstance(v, float) and np.isnan(v))),
        }

    # ── breakdowns ────────────────────────────────────────────────────────────
    def _group_stats(group_key: str, value_key: str) -> dict:
        groups: dict = {}
        for r in results:
            label = r.get("metadata", {}).get(group_key, "unknown")
            val   = r.get(value_key)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                continue
            groups.setdefault(label, []).append(val)
        return {grp: {"mean": _safe_mean(vs), "n": len(vs)} for grp, vs in sorted(groups.items())}

    agg["by_dataset"]   = _group_stats("dataset",           "mae")
    agg["by_difficulty"] = _group_stats("difficulty",        "mae")
    agg["by_dist_type"]  = _group_stats("distribution_type", "mae")

    return agg


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a fine-tuned LLM on the OpenEstimate benchmark."
    )
    parser.add_argument(
        "--ckpt_dir", default=None,
        help="Path to the checkpoint epoch directory (e.g. ckpt/llama3_8B/<run>/epoch_0).",
    )
    parser.add_argument(
        "--pretrained", action="store_true",
        help="Evaluate the pretrained base model instead of a fine-tuned checkpoint.",
    )
    parser.add_argument(
        "--model_path", default=DEFAULT_MODEL_PATH,
        help="Path to the base HF model directory (used with --pretrained).",
    )
    parser.add_argument(
        "--data_path",
        default="./data_processing/openestimate_test.json",
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
        "--free_gen", action="store_true",
        help="With --mode generate, rewrite the prompt to allow reasoning "
             "before the final answer, and parse the last <mean> <std> pair.",
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
        help="Path for the JSON results file.  Defaults to "
             "<ckpt_dir>/openestimate_eval_<split>.json.",
    )
    args = parser.parse_args()

    if not args.pretrained and args.ckpt_dir is None:
        parser.error("Provide --ckpt_dir or --pretrained.")

    # ── output file ───────────────────────────────────────────────────────────
    if args.output_file is None:
        if args.pretrained:
            out_dir = os.path.join(os.path.dirname(__file__), "results")
            args.output_file = os.path.join(out_dir, f"openestimate_eval_pretrained_{args.split}_{args.mode}.json")
        else:
            args.output_file = os.path.join(
                args.ckpt_dir,
                f"openestimate_eval_{args.split}_{args.mode}.json",
            )
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)

    # ── load data ─────────────────────────────────────────────────────────────
    data = load_data(args.data_path, split=args.split, n_examples=args.n_examples)
    if not data:
        print("No data loaded — check --data_path and --split.")
        sys.exit(1)

    # ── load model ────────────────────────────────────────────────────────────
    if args.pretrained:
        model, tokenizer = load_pretrained_model_and_tokenizer(args.model_path, args.device, args.dtype)
    else:
        model, tokenizer = load_model_and_tokenizer(args.ckpt_dir, args.device, args.dtype)
    nt_helpers = _setup_number_tokens(tokenizer)
    if nt_helpers['single_token']:
        number_token_ids = get_number_token_ids(tokenizer)  # legacy 101-id tensor for Llama
    else:
        number_token_ids = torch.tensor(nt_helpers['number_token_ids'],
                                        dtype=torch.long)
        print(f"Multi-token tokenizer detected — dispatching OE TF eval to "
              f"multi-token path. (digit token ids: "
              f"{nt_helpers['digit_token_ids']})")
    print(f"Number token IDs (sample 0-5): {number_token_ids[:6].tolist()}")

    # ── run evaluation ────────────────────────────────────────────────────────
    print(f"\nRunning {args.mode!r} evaluation on {len(data)} examples …")
    if args.mode == "teacher_force":
        results = run_teacher_force_eval(
            model, tokenizer, data, number_token_ids,
            device=args.device,
            batch_size=args.batch_size,
            max_seq_len=args.max_seq_len,
            nt_helpers=nt_helpers,
        )
    else:
        results = run_generate_eval(
            model, tokenizer, data, number_token_ids,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            max_seq_len=args.max_seq_len,
            free_gen=args.free_gen,
        )

    if not results:
        print("No results produced — evaluation may have failed entirely.")
        sys.exit(1)

    # ── aggregate ─────────────────────────────────────────────────────────────
    summary = aggregate_metrics(results)

    print("\n=== OpenEstimate Evaluation Summary ===")
    print(f"  Mode    : {args.mode}")
    print(f"  Split   : {args.split}  ({len(results)} examples)")
    print(f"  MAE (mean) : {summary['mae']['mean']:.3f} ± {summary['mae']['std']:.3f}  (median {summary['mae']['median']:.3f})")
    print(f"  MAE (std)  : {summary['mae_std']['mean']:.3f} ± {summary['mae_std']['std']:.3f}  (median {summary['mae_std']['median']:.3f})  [explicit model output]")
    if args.mode == "teacher_force":
        print(f"  CE_mean    : {summary['ce_mean']['mean']:.3f} ± {summary['ce_mean']['std']:.3f}  [at mean answer position]")
        print(f"  CE_dist    : {summary['ce_dist']['mean']:.3f} ± {summary['ce_dist']['std']:.3f}  [at mean answer position]")
    print()
    print("  MAE by dataset:")
    for ds, s in summary["by_dataset"].items():
        print(f"    {ds:<12} {s['mean']:.3f}  (n={s['n']})")
    print("  MAE by difficulty:")
    for diff, s in summary["by_difficulty"].items():
        print(f"    {diff:<12} {s['mean']:.3f}  (n={s['n']})")
    print("  MAE by distribution type:")
    for dt, s in summary["by_dist_type"].items():
        print(f"    {dt:<12} {s['mean']:.3f}  (n={s['n']})")

    # ── save ──────────────────────────────────────────────────────────────────
    output = {
        "ckpt_dir":   args.ckpt_dir,
        "data_path":  args.data_path,
        "split":      args.split,
        "mode":       args.mode,
        "free_gen":   args.free_gen,
        "n_examples": len(results),
        "summary":    summary,
        "per_example": results,
    }
    with open(args.output_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.output_file}")


if __name__ == "__main__":
    main()
