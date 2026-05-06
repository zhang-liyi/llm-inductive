"""
Utilities for probabilistic reasoning fine-tuning with ground truth distributions.

This module provides tools to compare LLM outputs with ground truth probability
distributions over tokens 0-100 using cross-entropy loss.
"""

import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

from torchtune.data import Message

# Bracket-format instruction (matches data/update_format.py NEW_INSTRUCTION).
# Swapped in when bracket_format=True; the stored "return only an integer" prefix
# in each example's ``input`` is replaced by this text at tokenize time.
BRACKET_INSTRUCTION = (
    "Answer the query in the scenario and return only an integer wrapped "
    "in < and >. For example, <x>. Use 0-100 scale. For a query on "
    "individual rank, a higher number means a higher ranking (e.g. 100 "
    "means the individual ranks highest in that criterion; 1 is lowest). "
    "For a query on which of the two teams wins, a smaller number means "
    "the first team more likely wins."
)


def number_to_english(n: int) -> str:
    """
    Convert integer 0-100 to English word representation.

    Examples: 0 -> "zero", 21 -> "twenty-one", 45 -> "forty-five", 100 -> "one hundred"
    """
    ones = [
        'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven',
        'eight', 'nine', 'ten', 'eleven', 'twelve', 'thirteen', 'fourteen',
        'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen',
    ]
    tens = [
        '', '', 'twenty', 'thirty', 'forty', 'fifty',
        'sixty', 'seventy', 'eighty', 'ninety',
    ]

    if n < 0 or n > 100:
        raise ValueError(f"n must be 0-100, got {n}")
    if n < 20:
        return ones[n]
    if n == 100:
        return 'one hundred'
    if n % 10 == 0:
        return tens[n // 10]
    return f'{tens[n // 10]}-{ones[n % 10]}'


def get_english_number_token_info(tokenizer) -> dict:
    """
    Get token information for English number words 0-100.

    Returns:
        dict with:
        - 'words': list of 101 English word strings
        - 'token_ids': list of 101 token ID lists (full tokenization per word)
        - 'first_token_ids': tensor of first token IDs [101]
        - 'is_single_token': list of 101 booleans
        - 'first_token_words': list of decoded first tokens (for metadata)
    """
    words = []
    token_ids = []
    first_token_ids = []
    is_single_token = []
    first_token_words = []

    for n in range(101):
        word = number_to_english(n)
        words.append(word)

        # Tokenize with space prefix to match typical output context
        tokens = tokenizer.encode(' ' + word, add_bos=False, add_eos=False)

        token_ids.append(tokens)
        first_token_ids.append(tokens[0])
        is_single_token.append(len(tokens) == 1)

        # Decode first token for metadata
        try:
            first_word = tokenizer.decode([tokens[0]]).strip()
        except Exception:
            first_word = f"[{tokens[0]}]"
        first_token_words.append(first_word)

    return {
        'words': words,
        'token_ids': token_ids,
        'first_token_ids': torch.tensor(first_token_ids, dtype=torch.long),
        'is_single_token': is_single_token,
        'first_token_words': first_token_words,
    }


def get_number_token_ids(tokenizer) -> torch.Tensor:
    """
    Get token IDs for numbers 0-100 as they appear in <N> format.

    Args:
        tokenizer: HuggingFace tokenizer

    Returns:
        Tensor of token IDs [101] for numbers 0-100
    """
    number_token_ids = []

    # Encode all numbers in context to match actual output format <N>.
    # Using "<N>" context ensures token IDs match what the model sees at
    # inference time when the output is formatted as "<77>".
    test_output = " ".join(f"<{i}>" for i in range(101))
    tokens = tokenizer.encode(test_output, add_bos=False, add_eos=False)

    # Create mapping from number to token ID
    # Decode each token and check if it's a number
    for i in range(101):
        found = False
        str_num = str(i)

        for tok in tokens:
            decoded = tokenizer.decode([tok]).strip()
            if decoded == str_num:
                number_token_ids.append(tok)
                found = True
                break

        if not found:
            # Fallback: try encoding the number directly
            direct_tokens = tokenizer.encode(str(i), add_bos=False, add_eos=False)
            if len(direct_tokens) >= 1:
                number_token_ids.append(direct_tokens[0])
            else:
                number_token_ids.append(-1)

    return torch.tensor(number_token_ids, dtype=torch.long)


# ---------------------------------------------------------------------------
# Multi-token integer support (Qwen-2/2.5 etc.)
# ---------------------------------------------------------------------------
#
# For tokenizers that single-token every integer 0..100 (e.g. Llama-3), the
# legacy ``get_number_token_ids`` works as-is and the existing distributional
# loss is correct.
#
# For tokenizers that don't (Qwen-2/2.5: only 0..9 are single-token; 10..99
# are 2-token digit pairs; 100 is 3-token), the legacy lookup silently
# collapses 91 of 101 entries onto the single-digit prefix tokens. We need a
# proper per-integer tokenization plus a position-decomposed distributional
# loss.

DIGIT_STRS = [str(d) for d in range(10)]


def _no_special_encode(tokenizer, text: str) -> List[int]:
    """Encode ``text`` without any added special tokens. Works for torchtune
    wrappers (which take ``add_bos``/``add_eos``) and HF tokenizers (which take
    ``add_special_tokens``)."""
    try:
        return list(tokenizer.encode(text, add_bos=False, add_eos=False))
    except TypeError:
        return list(tokenizer.encode(text, add_special_tokens=False))


def get_number_token_seqs(tokenizer) -> List[List[int]]:
    """Return the actual token-id sequence for each integer 0..100.

    For Llama-3-style tokenizers all returned sequences have length 1.
    For Qwen-2/2.5: length 1 for 0..9, length 2 for 10..99, length 3 for 100.
    """
    return [_no_special_encode(tokenizer, str(i)) for i in range(101)]


def get_digit_token_ids(tokenizer) -> List[int]:
    """Return the 10 single-digit token IDs ['0',..,'9'] in digit order."""
    digit_ids: List[int] = []
    for d in range(10):
        ids = _no_special_encode(tokenizer, str(d))
        if len(ids) != 1:
            raise ValueError(
                f"Single digit '{d}' did not single-token (got {len(ids)} ids: {ids}). "
                "All major LLM tokenizers handle 0..9 as single tokens; "
                "this should not happen in practice."
            )
        digit_ids.append(ids[0])
    return digit_ids


def tokenizer_single_tokens_integers(tokenizer) -> bool:
    """True iff every integer 0..100 is a single token under this tokenizer."""
    seqs = get_number_token_seqs(tokenizer)
    return all(len(s) == 1 for s in seqs)


def find_answer_spans_with_term(
    labels: torch.Tensor,
    digit_token_set,
    term_token_id: int,
) -> List[List[Tuple[List[int], Optional[int]]]]:
    """Like find_answer_spans but also returns the ``>`` terminator position
    immediately after each digit run when present.

    Returns: per example, list of (digit_span, term_pos | None) pairs.
    """
    bs = labels.size(0)
    out: List[List[Tuple[List[int], Optional[int]]]] = []
    for b in range(bs):
        spans: List[Tuple[List[int], Optional[int]]] = []
        cur: List[int] = []
        for pos in range(labels.size(1)):
            tok = labels[b, pos].item()
            if tok == -100:
                if cur:
                    spans.append((cur, None))
                    cur = []
                continue
            if tok in digit_token_set:
                cur.append(pos)
            elif tok == term_token_id and cur:
                spans.append((cur, pos))
                cur = []
            else:
                if cur:
                    spans.append((cur, None))
                    cur = []
        if cur:
            spans.append((cur, None))
        out.append(spans)
    return out


def find_answer_spans(
    labels: torch.Tensor,
    digit_token_set,
) -> List[List[List[int]]]:
    """Group consecutive digit-token positions into spans (one span per query).

    A span is a maximal run of label positions whose label token id is in
    ``digit_token_set``. Spans are separated by either non-digit response
    tokens (e.g. ``<``, ``>``, EOT, ``,``) or by ``-100`` ignore tokens.

    For Llama-3 single-token integers each span is length 1 (legacy behaviour).
    For Qwen multi-token integers spans have length 1, 2, or 3.

    Returns:
        List of length batch_size; each element is a list of spans for that
        example; each span is a list of int positions (indices into seq_len).
    """
    bs = labels.size(0)
    out: List[List[List[int]]] = []
    for b in range(bs):
        spans: List[List[int]] = []
        cur: List[int] = []
        for pos in range(labels.size(1)):
            tok = labels[b, pos].item()
            is_digit = tok in digit_token_set if tok != -100 else False
            if is_digit:
                cur.append(pos)
            else:
                if cur:
                    spans.append(cur)
                    cur = []
        if cur:
            spans.append(cur)
        out.append(spans)
    return out


def _build_first_second_third_idx(
    number_token_seqs: List[List[int]],
    digit_token_ids: List[int],
):
    """Precompute, for each integer 0..100:
       first_idx[i]  = digit index (0..9) of seq[0]
       second_idx[i] = digit index of seq[1] or -1 if 1-token
       third_idx[i]  = digit index of seq[2] or -1
    """
    digit_to_idx = {tid: idx for idx, tid in enumerate(digit_token_ids)}
    first  = [-1] * 101
    second = [-1] * 101
    third  = [-1] * 101
    for i, seq in enumerate(number_token_seqs):
        if len(seq) >= 1 and seq[0] in digit_to_idx:
            first[i]  = digit_to_idx[seq[0]]
        if len(seq) >= 2 and seq[1] in digit_to_idx:
            second[i] = digit_to_idx[seq[1]]
        if len(seq) >= 3 and seq[2] in digit_to_idx:
            third[i]  = digit_to_idx[seq[2]]
    return first, second, third


def compute_distribution_loss_multitoken(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ground_truth_bins: torch.Tensor,
    number_token_seqs: List[List[int]],
    digit_token_ids: List[int],
    num_queries: Optional[torch.Tensor] = None,
    eps: float = 1e-10,
) -> torch.Tensor:
    """Multi-token-aware distributional cross-entropy loss.

    Decomposes the 101-bin CE into a sum over digit positions of per-position
    CE against the digit marginal/conditional of ``gt_bins``. Equivalent to
    the legacy single-token loss when every integer is a single token; faithful
    generalisation for tokenizers like Qwen-2.

    For each (example, query) span:
      - pos 0 target: P(d_0 = d) = sum gt_bins[i] over i with first_digit_token == d.
      - pos 1 target: P(d_1 = d | d_0 = gt_d_0) — conditional under the GT's
        actually-teacher-forced first digit.
      - pos 2 target (only when GT = 100): P(d_2 = d | (gt_d_0, gt_d_1)).
    Total loss = sum of per-position CEs averaged across queries.
    """
    device = logits.device
    digit_token_ids_t = torch.tensor(digit_token_ids, dtype=torch.long, device=device)
    digit_token_set   = set(digit_token_ids)
    first_idx, second_idx, third_idx = _build_first_second_third_idx(
        number_token_seqs, digit_token_ids
    )
    first_idx_t  = torch.tensor(first_idx,  dtype=torch.long, device=device)
    second_idx_t = torch.tensor(second_idx, dtype=torch.long, device=device)
    third_idx_t  = torch.tensor(third_idx,  dtype=torch.long, device=device)

    digit_to_idx = {tid: idx for idx, tid in enumerate(digit_token_ids)}

    spans = find_answer_spans(labels, digit_token_set)

    total_loss = torch.tensor(0.0, device=device)
    n_terms = 0
    bs = labels.size(0)
    for b in range(bs):
        nq = (num_queries[b].item() if num_queries is not None
              else ground_truth_bins[b].size(0))
        ex_spans = spans[b][:nq]
        for q_idx, span in enumerate(ex_spans):
            gt_bin = ground_truth_bins[b, q_idx].to(device)  # [101]
            if not span:
                continue

            # ── pos 0 ──────────────────────────────────────────────────────────
            target_0 = torch.zeros(10, device=device)
            valid_first = first_idx_t >= 0
            if valid_first.any():
                target_0.index_add_(0, first_idx_t[valid_first], gt_bin[valid_first])
            t0_sum = target_0.sum()
            if t0_sum > 1e-12:
                target_0 = target_0 / t0_sum
            else:
                continue
            pos_0 = span[0]
            ml0 = logits[b, pos_0, digit_token_ids_t]
            md0 = F.log_softmax(ml0, dim=-1)
            total_loss = total_loss - (target_0 * md0).sum()
            n_terms += 1

            if len(span) < 2:
                continue

            # ── pos 1 (conditional on GT d_0) ──────────────────────────────────
            gt_first_token = labels[b, span[0]].item()
            gt_d0 = digit_to_idx.get(gt_first_token, -1)
            if gt_d0 < 0:
                continue
            target_1 = torch.zeros(10, device=device)
            mask1 = (first_idx_t == gt_d0) & (second_idx_t >= 0)
            if mask1.any():
                target_1.index_add_(0, second_idx_t[mask1], gt_bin[mask1])
            t1_sum = target_1.sum()
            if t1_sum > 1e-12:
                target_1 = target_1 / t1_sum
            else:
                continue
            pos_1 = span[1]
            ml1 = logits[b, pos_1, digit_token_ids_t]
            md1 = F.log_softmax(ml1, dim=-1)
            total_loss = total_loss - (target_1 * md1).sum()
            n_terms += 1

            if len(span) < 3:
                continue

            # ── pos 2 (only "100" reaches here in 0..100) ──────────────────────
            gt_second_token = labels[b, span[1]].item()
            gt_d1 = digit_to_idx.get(gt_second_token, -1)
            if gt_d1 < 0:
                continue
            target_2 = torch.zeros(10, device=device)
            mask2 = (first_idx_t == gt_d0) & (second_idx_t == gt_d1) & (third_idx_t >= 0)
            if mask2.any():
                target_2.index_add_(0, third_idx_t[mask2], gt_bin[mask2])
            t2_sum = target_2.sum()
            if t2_sum > 1e-12:
                target_2 = target_2 / t2_sum
            else:
                continue
            pos_2 = span[2]
            ml2 = logits[b, pos_2, digit_token_ids_t]
            md2 = F.log_softmax(ml2, dim=-1)
            total_loss = total_loss - (target_2 * md2).sum()
            n_terms += 1

    if n_terms > 0:
        return total_loss / n_terms
    return torch.tensor(0.0, device=device)


def compute_distribution_loss_multitoken_with_term(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ground_truth_bins: torch.Tensor,
    number_token_seqs: List[List[int]],
    digit_token_ids: List[int],
    term_token_id: int,
    num_queries: Optional[torch.Tensor] = None,
    eps: float = 1e-10,
    fallback_threshold: float = 1e-4,
) -> torch.Tensor:
    """Distribution loss with terminator (``>``) supervision.

    Same factorization as compute_distribution_loss_multitoken but at every
    position k ≥ 1 the target is an 11-bin distribution over ``{'0',..,'9', '>'}``:

      target[k]['>'] = P(integer ends at length k | prefix gt_d_0..d_{k-1})
      target[k][d]  = P(d_k = d, integer continues | prefix)
      sum target[k] = 1

    Conditional fallback: if the conditional digit-mass at position k is
    below ``fallback_threshold``, target becomes
        target['>'] = p,   target[d] = (1-p) / 10  uniform over digits
    where p is the proper termination probability under the GT bins.

    Position 0 is unchanged (10-bin digit marginal).
    """
    device = logits.device
    digit_t = torch.tensor(digit_token_ids, dtype=torch.long, device=device)
    digit_token_set = set(digit_token_ids)
    first_idx, second_idx, third_idx = _build_first_second_third_idx(
        number_token_seqs, digit_token_ids
    )
    first_idx_t = torch.tensor(first_idx, dtype=torch.long, device=device)
    second_idx_t = torch.tensor(second_idx, dtype=torch.long, device=device)
    third_idx_t = torch.tensor(third_idx, dtype=torch.long, device=device)
    digit_to_idx = {tid: idx for idx, tid in enumerate(digit_token_ids)}
    seq_lens = torch.tensor([len(s) for s in number_token_seqs],
                            dtype=torch.long, device=device)

    # 11-bin token slot ordering: indices 0..9 = digit_token_ids, index 10 = >
    eleven_t = torch.cat([digit_t, torch.tensor([term_token_id], device=device)])

    spans = find_answer_spans_with_term(labels, digit_token_set, term_token_id)

    total_loss = torch.tensor(0.0, device=device)
    n_terms = 0

    bs = labels.size(0)
    for b in range(bs):
        nq = (num_queries[b].item() if num_queries is not None
              else ground_truth_bins[b].size(0))
        ex_spans = spans[b][:nq]
        for q_idx, (digit_span, term_pos) in enumerate(ex_spans):
            if not digit_span:
                continue
            gt_bin = ground_truth_bins[b, q_idx].to(device)  # [101]

            # ── Position 0 (10-bin digit marginal, no `>` slot) ──────────────
            target_0 = torch.zeros(10, device=device)
            valid_first = first_idx_t >= 0
            if valid_first.any():
                target_0.index_add_(0, first_idx_t[valid_first], gt_bin[valid_first])
            t0_sum = target_0.sum()
            if t0_sum > 1e-12:
                target_0 = target_0 / t0_sum
                ml0 = logits[b, digit_span[0], digit_t]
                lp0 = F.log_softmax(ml0, dim=-1)
                total_loss = total_loss - (target_0 * lp0).sum()
                n_terms += 1

            # ── Subsequent positions: 11-bin (digits + `>`) ──────────────────
            # Walk through positions 1..L (L = len(digit_span)). Position k
            # is digit_span[k] if k < L (a digit position), or term_pos if
            # k == L (the `>` position).
            L_span = len(digit_span)
            # Precompute prefix digit indices (length up to L_span)
            prefix_digit_idx: List[int] = []
            for k in range(L_span):
                gt_tok = labels[b, digit_span[k]].item()
                d_k = digit_to_idx.get(gt_tok, -1)
                if d_k < 0:
                    break
                prefix_digit_idx.append(d_k)
            if len(prefix_digit_idx) < L_span:
                continue  # malformed prefix; skip downstream supervision

            # For each subsequent position k in 1..L_span (inclusive of term)
            for k in range(1, L_span + 1):
                # Build mask over integers 0..100 consistent with the
                # observed prefix prefix_digit_idx[:k].
                # For k=1: mask = first_idx == prefix[0]
                # For k=2: mask = first_idx == prefix[0] AND second_idx == prefix[1]
                # For k=3: mask = ... AND third_idx == prefix[2]
                pref = prefix_digit_idx[:k]
                m = (first_idx_t == pref[0])
                if k >= 2:
                    m = m & (second_idx_t == pref[1])
                if k >= 3:
                    m = m & (third_idx_t == pref[2])
                # Mass on integers consistent with prefix.
                M_total = float(gt_bin[m].sum().item()) if m.any() else 0.0
                if M_total < 1e-12:
                    continue  # no GT mass — skip

                # Termination probability at position k = P(integer length == k | prefix).
                # That's the integer whose tokenization exactly equals the prefix.
                # Such an integer i satisfies seq_lens[i] == k AND prefix matches.
                term_mask = m & (seq_lens == k)
                p_term = float(gt_bin[term_mask].sum().item()) / M_total
                # Continuation conditional: integers whose seq_len > k AND prefix matches
                cont_mask = m & (seq_lens > k)
                M_cont = float(gt_bin[cont_mask].sum().item())

                # Build 11-bin target.
                target = torch.zeros(11, device=device)
                target[10] = p_term
                if M_cont >= fallback_threshold:
                    # Distribute (1 - p_term) over digits proportionally to
                    # next-digit mass under the conditional posterior.
                    # target[d] = (mass on integers continuing with digit d) / M_total
                    if k == 1:
                        next_idx_t = second_idx_t
                    elif k == 2:
                        next_idx_t = third_idx_t
                    else:
                        next_idx_t = None  # k>=3 has no next digit in 0..100
                    if next_idx_t is not None:
                        for d in range(10):
                            d_mask = cont_mask & (next_idx_t == d)
                            if d_mask.any():
                                target[d] = float(gt_bin[d_mask].sum().item()) / M_total
                else:
                    # Fallback: (1 - p_term) uniform across digits 0..9.
                    target[:10] = (1.0 - p_term) / 10.0

                # Pick logit position: digit_span[k] if k < L_span, else term_pos.
                if k < L_span:
                    pos = digit_span[k]
                else:
                    if term_pos is None:
                        continue   # no `>` label — skip terminator supervision
                    pos = term_pos
                ml = logits[b, pos, eleven_t]
                lp = F.log_softmax(ml, dim=-1)
                total_loss = total_loss - (target * lp).sum()
                n_terms += 1

    if n_terms > 0:
        return total_loss / n_terms
    return torch.tensor(0.0, device=device)


# ---------------------------------------------------------------------------
# Multi-token-aware MAE / pred_mean for evaluation
# ---------------------------------------------------------------------------

def predict_integer_from_span(
    logits: torch.Tensor,           # [seq_len, vocab_size]
    span: List[int],                # positions in the answer span
    digit_token_ids: List[int],
    method: str = "greedy",         # "greedy" or "expected"
    number_token_seqs: Optional[List[List[int]]] = None,
) -> float:
    """Return a single integer prediction for an answer span.

    method="greedy"  : argmax-of-digit-tokens at each position, concatenated.
                       Returns an int.
    method="expected": expected value over candidates whose tokenization length
                       matches the span length. Uses position-independent
                       digit-token softmax under teacher forcing — correct
                       up to conditioning approximation (the second-digit
                       softmax is conditioned on GT's actually-forced first
                       digit). Returns a float.

    For 1-token spans (Llama or Qwen 0..9), 'expected' reduces to the standard
    pred_mean over the 1-digit candidate set.
    """
    device = logits.device
    digit_token_ids_t = torch.tensor(digit_token_ids, dtype=torch.long, device=device)
    if method == "greedy":
        digits = []
        for pos in span:
            ml = logits[pos, digit_token_ids_t]
            digits.append(int(ml.argmax().item()))
        return int("".join(str(d) for d in digits))
    if number_token_seqs is None:
        raise ValueError("method='expected' requires number_token_seqs")
    first_idx, second_idx, third_idx = _build_first_second_third_idx(
        number_token_seqs, digit_token_ids
    )
    p0 = F.softmax(logits[span[0], digit_token_ids_t], dim=-1).detach().float().cpu().numpy()
    p1 = (F.softmax(logits[span[1], digit_token_ids_t], dim=-1).detach().float().cpu().numpy()
          if len(span) >= 2 else None)
    p2 = (F.softmax(logits[span[2], digit_token_ids_t], dim=-1).detach().float().cpu().numpy()
          if len(span) >= 3 else None)
    L = len(span)
    sum_p, sum_ip = 0.0, 0.0
    for i in range(101):
        if len(number_token_seqs[i]) != L:
            continue
        if first_idx[i] < 0:
            continue
        prob = float(p0[first_idx[i]])
        if L >= 2:
            if second_idx[i] < 0:
                continue
            prob *= float(p1[second_idx[i]])
        if L >= 3:
            if third_idx[i] < 0:
                continue
            prob *= float(p2[third_idx[i]])
        sum_p += prob
        sum_ip += i * prob
    return sum_ip / sum_p if sum_p > 0 else 0.0


def evaluate_multitoken_metrics(
    logits: torch.Tensor,
    ground_truth_bins: torch.Tensor,
    number_token_seqs: List[List[int]],
    digit_token_ids: List[int],
    num_queries: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """Span-based multi-token eval for Qwen-2 / Qwen-2.5 style tokenizers.

    Returns the same metric keys as the single-token Llama path so the recipe
    code paths are interchangeable. Per (example, query) span:
      cross_entropy       : sum of per-token CE on the GT span tokens
                            ("CE = sum of these tokens").
      cross_entropy_mean  : -log P(round(gt_mean)) under the multi-token product.
      cross_entropy_dist  : sum of per-position decomposed CE against the
                            digit marginal/conditional of gt_bins.
      mean_abs_error      : |pred_mean - gt_mean_float|, where pred_mean is
                            the expected integer over span-length-matching
                            candidates (multi-token product, conditional
                            approximation under teacher forcing).
      mean_abs_error_dist : 0.0 placeholder (not directly comparable to the
                            single-token L1 distance over 101 bins).
    """
    if labels is None:
        raise ValueError("evaluate_multitoken_metrics needs labels for span detection")
    device = logits.device
    digit_set = set(digit_token_ids)
    digit_t   = torch.tensor(digit_token_ids, dtype=torch.long, device=device)
    first_idx, second_idx, third_idx = _build_first_second_third_idx(
        number_token_seqs, digit_token_ids
    )
    first_idx_t  = torch.tensor(first_idx,  dtype=torch.long, device=device)
    second_idx_t = torch.tensor(second_idx, dtype=torch.long, device=device)
    third_idx_t  = torch.tensor(third_idx,  dtype=torch.long, device=device)
    digit_to_idx = {tid: idx for idx, tid in enumerate(digit_token_ids)}

    spans = find_answer_spans(labels, digit_set)

    total_ce = total_ce_mean = total_ce_dist = 0.0
    total_mae = total_mae_dist = 0.0
    total_queries = 0
    pred_means: List[float] = []
    gt_means:   List[float] = []

    bs = logits.size(0)
    for i in range(bs):
        gt_bins = ground_truth_bins[i]
        n_q = num_queries[i].item() if num_queries is not None else gt_bins.size(0)
        ex_spans = spans[i][:n_q]
        for q, span in enumerate(ex_spans):
            if not span:
                continue
            gt_dist = gt_bins[q].cpu().numpy()
            gt_mean_float = float(sum(idx * gt_dist[idx] for idx in range(101)))

            # Per-token CE on GT
            ce_sum = 0.0
            for pos in span:
                gt_tid = labels[i, pos].item()
                lp = F.log_softmax(logits[i, pos], dim=0)
                ce_sum += -lp[gt_tid].item()
            total_ce += ce_sum

            # Pred mean (expected integer over matching-length candidates)
            pred_mean = float(predict_integer_from_span(
                logits[i], span, digit_token_ids, method="expected",
                number_token_seqs=number_token_seqs,
            ))
            pred_means.append(pred_mean)
            gt_means.append(gt_mean_float)
            total_mae += abs(pred_mean - gt_mean_float)

            # Decomposed CE_dist: pos 0 marginal + pos 1 conditional + pos 2 (if "100")
            gt_bin_t = gt_bins[q].to(device)
            target_0 = torch.zeros(10, device=device)
            valid = first_idx_t >= 0
            if valid.any():
                target_0.index_add_(0, first_idx_t[valid], gt_bin_t[valid])
            ml0 = logits[i, span[0], digit_t]
            lp0 = F.log_softmax(ml0, dim=-1)
            ce_pos0 = float(-(target_0 * lp0).sum().item()) if target_0.sum() > 1e-12 else 0.0
            ce_pos1 = ce_pos2 = 0.0
            if len(span) >= 2:
                gt_d0 = digit_to_idx.get(labels[i, span[0]].item(), -1)
                if gt_d0 >= 0:
                    target_1 = torch.zeros(10, device=device)
                    m1 = (first_idx_t == gt_d0) & (second_idx_t >= 0)
                    if m1.any():
                        target_1.index_add_(0, second_idx_t[m1], gt_bin_t[m1])
                    if target_1.sum() > 1e-12:
                        target_1 = target_1 / target_1.sum()
                        lp1 = F.log_softmax(logits[i, span[1], digit_t], dim=-1)
                        ce_pos1 = float(-(target_1 * lp1).sum().item())
            if len(span) >= 3:
                gt_d0 = digit_to_idx.get(labels[i, span[0]].item(), -1)
                gt_d1 = digit_to_idx.get(labels[i, span[1]].item(), -1)
                if gt_d0 >= 0 and gt_d1 >= 0:
                    target_2 = torch.zeros(10, device=device)
                    m2 = ((first_idx_t == gt_d0) & (second_idx_t == gt_d1)
                          & (third_idx_t >= 0))
                    if m2.any():
                        target_2.index_add_(0, third_idx_t[m2], gt_bin_t[m2])
                    if target_2.sum() > 1e-12:
                        target_2 = target_2 / target_2.sum()
                        lp2 = F.log_softmax(logits[i, span[2], digit_t], dim=-1)
                        ce_pos2 = float(-(target_2 * lp2).sum().item())
            total_ce_dist += ce_pos0 + ce_pos1 + ce_pos2

            # CE-mean target: round(gt_mean) integer's multi-token product
            gt_mean_int = max(0, min(100, int(round(gt_mean_float))))
            seq = number_token_seqs[gt_mean_int]
            if len(seq) == len(span) and all(t in digit_to_idx for t in seq):
                log_p = 0.0
                ok = True
                for k, tid in enumerate(seq):
                    d_idx = digit_to_idx[tid]
                    lp_k  = F.log_softmax(logits[i, span[k], digit_t], dim=-1)
                    log_p += float(lp_k[d_idx].item())
                if ok:
                    total_ce_mean += -log_p

            total_queries += 1

    if total_queries > 0:
        return {
            'kl_divergence': 0.0,
            'cross_entropy':       total_ce       / total_queries,
            'cross_entropy_mean':  total_ce_mean  / total_queries,
            'cross_entropy_dist':  total_ce_dist  / total_queries,
            'mean_abs_error':      total_mae      / total_queries,
            'mean_abs_error_dist': total_mae_dist / total_queries,
            'pred_means': pred_means,
            'gt_means':   gt_means,
        }
    return {'kl_divergence': 0.0, 'cross_entropy': 0.0, 'cross_entropy_mean': 0.0,
            'cross_entropy_dist': 0.0, 'mean_abs_error': 0.0, 'mean_abs_error_dist': 0.0,
            'pred_means': [], 'gt_means': []}


def reconstruct_int_from_label_span(
    labels_row: torch.Tensor,
    span: List[int],
    digit_token_ids: List[int],
) -> int:
    """Recover the GT integer from a span of label tokens.

    Each label position must be one of the 10 single-digit token IDs; otherwise
    raises (since spans are constructed by find_answer_spans which only includes
    digit-token positions)."""
    digit_to_idx = {tid: idx for idx, tid in enumerate(digit_token_ids)}
    digits = []
    for pos in span:
        tid = labels_row[pos].item()
        if tid not in digit_to_idx:
            raise ValueError(
                f"Span position {pos} has label token id {tid} which is not "
                "a single-digit token; this should not happen if the span came "
                "from find_answer_spans."
            )
        digits.append(digit_to_idx[tid])
    return int("".join(str(d) for d in digits))


class ProbabilisticReasoningDataset(Dataset):
    """
    Dataset for probabilistic reasoning with ground truth posterior distributions.

    Each example contains:
    - input: Instruction + scenario text
    - output: Ground truth means and stds as text
    - bins: Ground truth probability distributions [num_queries, 101]
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_seq_length: int = 2048,
        instruction_override: Optional[str] = None,
        bracket_format: bool = False,
        train_terminator: bool = False,
    ):
        """
        Args:
            data_path: Path to JSON file with probabilistic reasoning data
            tokenizer: Tokenizer for encoding text
            max_seq_length: Maximum sequence length
            instruction_override: If provided, replaces the instruction prefix
                (everything before the first blank line) in each prompt.
            bracket_format: If True, swap the stored instruction prefix for
                BRACKET_INSTRUCTION and wrap each bare-integer output as
                ``<N>``. Takes precedence over ``instruction_override``.
            train_terminator: If True, also keep the closing ``>`` token
                immediately following each digit span. Used by the
                multi-token distribution-loss sub-mode that supervises
                P(terminate at this length) jointly with the digit
                marginal/conditional. No-op when bracket_format is False
                (bare integer outputs have no ``>`` to keep).
        """
        with open(data_path, 'r') as f:
            self.data = json.load(f)

        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.bracket_format = bracket_format
        self.train_terminator = train_terminator
        if bracket_format and instruction_override is not None:
            raise ValueError("bracket_format and instruction_override are mutually exclusive")
        self.instruction_override = (
            BRACKET_INSTRUCTION if bracket_format else instruction_override
        )
        # Precompute number token IDs so __getitem__ can mask non-number output tokens
        self._number_token_set = set(get_number_token_ids(tokenizer).tolist())
        # Closing-bracket ``>`` token id (for train_terminator). Encode an
        # actual ``<N>`` snippet so we get the same bracketing context the
        # dataset emits at training time.
        if train_terminator:
            term_ids = _no_special_encode(tokenizer, "<5>")
            # Last token of "<5>" should decode to ``>``. Defensive check.
            self._term_token_id = term_ids[-1] if term_ids else -1
            try:
                _decoded = tokenizer.decode([self._term_token_id])
                if '>' not in _decoded:
                    raise ValueError(
                        f"Expected last token of '<5>' to decode to '>', "
                        f"got {_decoded!r} (id={self._term_token_id})"
                    )
            except Exception:
                pass
        else:
            self._term_token_id = -1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns tokenized example with ground truth distributions.
        """
        example = self.data[idx]

        # Tokenize with chat template to match inference format
        input_text = example['input']
        if self.instruction_override is not None:
            parts = input_text.split("\n\n", 1)
            input_text = self.instruction_override + ("\n\n" + parts[1] if len(parts) > 1 else "")
        output_text = example['output']
        if self.bracket_format and not (output_text.startswith('<') and output_text.endswith('>')):
            output_text = f"<{output_text}>"

        messages = [
            Message(role="user", content=input_text, masked=True, eot=True),
            Message(role="assistant", content=output_text, masked=False, eot=True),
        ]
        tokens, mask = self.tokenizer.tokenize_messages(messages)

        # Truncate if needed
        tokens = tokens[:self.max_seq_length]
        mask = mask[:self.max_seq_length]

        # Create labels: in the output region (mask=False), supervise only on
        # number tokens (0-100); mask everything else (<, >, EOT, spaces) with
        # -100 so we finetune purely on the integer value, not on the <> format.
        # If train_terminator: also keep the ``>`` token immediately after a
        # digit run, so the loss can train the terminate-vs-continue decision.
        labels = []
        keep_term_next = False
        for t, m in zip(tokens, mask):
            if m:
                labels.append(-100)          # input region: always masked
                keep_term_next = False
            elif t in self._number_token_set:
                labels.append(t)             # output region: number token — keep
                keep_term_next = self.train_terminator
            elif keep_term_next and t == self._term_token_id:
                labels.append(t)             # ``>`` immediately after digits
                keep_term_next = False
            else:
                labels.append(-100)          # output region: format token — mask
                keep_term_next = False

        # Get ground truth distributions
        bins = example['bins']  # List of 101-length arrays

        return {
            'tokens': torch.tensor(tokens, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.bool),  # True = prompt (masked), False = response
            'ground_truth_bins': torch.tensor(bins, dtype=torch.float32),  # [num_queries, 101]
        }


def probabilistic_reasoning_collate_fn(batch, padding_idx=0, ignore_idx=-100):
    """
    Collate function for batching examples.

    Handles variable-length sequences and variable number of queries.
    All outputs are tensors to ensure compatibility with batch_to_device and other utilities.
    """
    # Find max length
    max_len = max(len(x['tokens']) for x in batch)

    # Find max number of queries across batch
    max_queries = max(x['ground_truth_bins'].size(0) for x in batch)

    # Pad sequences
    tokens_list = []
    labels_list = []
    mask_list = []
    bins_list = []
    num_queries_list = []

    for example in batch:
        tokens = example['tokens']
        labels = example['labels']
        mask = example['mask']
        bins = example['ground_truth_bins']

        # Pad tokens
        pad_len = max_len - len(tokens)
        padded_tokens = torch.cat([
            tokens,
            torch.full((pad_len,), padding_idx, dtype=torch.long)
        ])
        tokens_list.append(padded_tokens)

        # Pad labels
        padded_labels = torch.cat([
            labels,
            torch.full((pad_len,), ignore_idx, dtype=torch.long)
        ])
        labels_list.append(padded_labels)

        # Pad mask (True = masked/prompt, pad positions are masked)
        padded_mask = torch.cat([
            mask,
            torch.ones(pad_len, dtype=torch.bool)
        ])
        mask_list.append(padded_mask)

        # Pad bins to max_queries
        num_queries = bins.size(0)
        num_queries_list.append(num_queries)
        if num_queries < max_queries:
            # Pad with zeros for additional queries
            pad_bins = torch.zeros((max_queries - num_queries, 101), dtype=torch.float32)
            padded_bins = torch.cat([bins, pad_bins], dim=0)
        else:
            padded_bins = bins
        bins_list.append(padded_bins)

    return {
        'tokens': torch.stack(tokens_list),
        'labels': torch.stack(labels_list),
        'mask': torch.stack(mask_list),
        'ground_truth_bins': torch.stack(bins_list),  # [batch_size, max_queries, 101]
        'num_queries': torch.tensor(num_queries_list, dtype=torch.long),  # [batch_size]
    }


class SingleScenarioDataset(Dataset):
    """
    Dataset for single-scenario data without ground truth distributions.

    Loads examples with 'input' and 'output' fields and returns tokenized
    tokens/labels only (no bins). Intended for fine-tuning with standard CE loss,
    with validation/testing on a separate probabilistic reasoning dataset.
    """

    has_bins = False

    def __init__(self, data_path: str, tokenizer, max_seq_length: int = 2048):
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self._number_token_set = set(get_number_token_ids(tokenizer).tolist())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        input_text = example['input']
        output_text = example['output']

        messages = [
            Message(role="user", content=input_text, masked=True, eot=True),
            Message(role="assistant", content=output_text, masked=False, eot=True),
        ]
        tokens, mask = self.tokenizer.tokenize_messages(messages)

        tokens = tokens[:self.max_seq_length]
        mask = mask[:self.max_seq_length]

        # Supervise only on number tokens in the output region; mask <> and
        # other format tokens so we finetune purely on the integer value.
        labels = []
        for t, m in zip(tokens, mask):
            if m:
                labels.append(-100)
            elif t in self._number_token_set:
                labels.append(t)
            else:
                labels.append(-100)

        return {
            'tokens': torch.tensor(tokens, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
        }


def single_scenario_collate_fn(batch, padding_idx=0, ignore_idx=-100):
    """
    Collate function for SingleScenarioDataset batches (no bins).

    Pads tokens and labels to the longest sequence in the batch.
    Does not return ground_truth_bins or num_queries.
    """
    max_len = max(len(x['tokens']) for x in batch)

    tokens_list = []
    labels_list = []

    for example in batch:
        tokens = example['tokens']
        labels = example['labels']
        pad_len = max_len - len(tokens)

        tokens_list.append(torch.cat([
            tokens,
            torch.full((pad_len,), padding_idx, dtype=torch.long)
        ]))
        labels_list.append(torch.cat([
            labels,
            torch.full((pad_len,), ignore_idx, dtype=torch.long)
        ]))

    return {
        'tokens': torch.stack(tokens_list),
        'labels': torch.stack(labels_list),
    }


def _find_answer_positions(
    labels: torch.Tensor,
    number_token_ids: torch.Tensor,
    tokenizer=None,
) -> List[List[int]]:
    """
    Find answer positions (number token positions) for each example in the batch.

    Args:
        labels: Labels tensor [batch_size, seq_len]
        number_token_ids: Token IDs for numbers 0-100 [101]
        tokenizer: Tokenizer for decoding tokens

    Returns:
        List of lists of answer positions, one per example in the batch
    """
    number_token_set = set(number_token_ids.tolist())
    batch_size = labels.size(0)
    all_positions = []

    for i in range(batch_size):
        example_labels = labels[i]
        positions = []

        for pos in range(example_labels.size(0)):
            label_val = example_labels[pos].item()
            if label_val == -100:
                continue

            if label_val in number_token_set:
                positions.append(pos)
                continue

            if tokenizer is not None:
                try:
                    decoded = tokenizer.decode([label_val]).strip()
                    if decoded.isdigit() and 0 <= int(decoded) <= 100:
                        positions.append(pos)
                except:
                    pass

        all_positions.append(positions)

    return all_positions


def compute_probabilistic_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ground_truth_bins: torch.Tensor,
    number_token_ids: torch.Tensor,
    ce_weight: float = 1.0,
    dist_weight: float = 1.0,
    ignore_index: int = -100,
    num_queries: Optional[torch.Tensor] = None,
    tokenizer = None,
    loss_mode: str = "distribution",
    number_token_seqs: Optional[List[List[int]]] = None,
    digit_token_ids: Optional[List[int]] = None,
    train_terminator: bool = False,
    term_token_id: int = -1,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute loss with two components:
    - Standard CE on non-answer tokens (structural tokens like '[', ',', ']')
    - Distributional CE over 101 number tokens at answer positions

    Args:
        logits: Model logits [batch_size, seq_len, vocab_size]
        labels: Ground truth labels [batch_size, seq_len]
        ground_truth_bins: Ground truth distributions [batch_size, max_queries, 101]
        number_token_ids: Token IDs for numbers 0-100 [101]
        ce_weight: Weight for standard cross-entropy loss on non-answer tokens
        dist_weight: Weight for distribution matching loss on answer tokens
        ignore_index: Index to ignore in loss computation
        num_queries: Number of valid queries per example [batch_size]
        tokenizer: Tokenizer for decoding tokens
        loss_mode: "distribution" for full posterior matching, "mean_only" for CE on
                   the mean token only (standard next-token prediction on answer tokens)

    Returns:
        loss: Total loss (scalar)
        loss_dict: Dictionary with loss components
    """
    device = logits.device
    loss_dict = {}

    # Find answer positions for each example in the batch
    answer_positions = _find_answer_positions(labels, number_token_ids, tokenizer)

    # --- Standard CE on non-answer positions ---
    if ce_weight > 0:
        # Mask out answer positions so CE only trains structural tokens
        labels_no_answers = labels.clone()
        for i, positions in enumerate(answer_positions):
            for pos in positions:
                labels_no_answers[i, pos] = ignore_index

        ce_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels_no_answers.view(-1),
            ignore_index=ignore_index,
            reduction='mean',
        )
        loss_dict['ce_loss'] = ce_loss.item()
    else:
        ce_loss = torch.tensor(0.0, device=device)
        loss_dict['ce_loss'] = 0.0

    # --- Loss at answer positions ---
    if ground_truth_bins is not None:
        if loss_mode == "mean_only":
            # Mean-only mode: standard CE on the answer tokens (the mean values)
            # Only keep answer positions in labels, mask everything else
            labels_answers_only = torch.full_like(labels, ignore_index)
            for i, positions in enumerate(answer_positions):
                for pos in positions:
                    labels_answers_only[i, pos] = labels[i, pos]

            dist_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels_answers_only.view(-1),
                ignore_index=ignore_index,
                reduction='mean',
            )
            loss_dict['dist_loss'] = dist_loss.item()
        else:
            # Distribution mode: CE over 101 number tokens against ground truth posterior.
            # When number_token_seqs is supplied with any multi-token sequence
            # (e.g. Qwen-2), compute_distribution_loss internally dispatches to
            # the position-decomposed multi-token variant.
            dist_loss = compute_distribution_loss(
                logits,
                ground_truth_bins,
                number_token_ids.to(device),
                num_queries=num_queries,
                labels=labels,
                tokenizer=tokenizer,
                number_token_seqs=number_token_seqs,
                digit_token_ids=digit_token_ids,
                train_terminator=train_terminator,
                term_token_id=term_token_id,
            )
            loss_dict['dist_loss'] = dist_loss.item()
    else:
        dist_loss = torch.tensor(0.0, device=device)
        loss_dict['dist_loss'] = 0.0

    total_loss = ce_weight * ce_loss + dist_weight * dist_loss
    loss_dict['total_loss'] = total_loss.item()

    return total_loss, loss_dict


def compute_distribution_loss(
    logits: torch.Tensor,
    ground_truth_bins: torch.Tensor,
    number_token_ids: torch.Tensor,
    num_queries: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
    tokenizer = None,
    number_token_seqs: Optional[List[List[int]]] = None,
    digit_token_ids: Optional[List[int]] = None,
    train_terminator: bool = False,
    term_token_id: int = -1,
) -> torch.Tensor:
    """
    Compute cross-entropy loss between predicted and ground truth distributions
    over tokens 0-100 at answer positions only.

    For each example:
    1. Find positions where labels indicate a number token answer
    2. Extract logits for tokens 0-100 at those specific positions
    3. Compare to corresponding ground truth distribution for each query

    Args:
        logits: Model logits [batch_size, seq_len, vocab_size]
        ground_truth_bins: Ground truth distributions [batch_size, max_queries, 101]
        number_token_ids: Token IDs for numbers 0-100 [101]
        num_queries: Number of valid queries per example [batch_size]
        labels: Labels tensor [batch_size, seq_len] to identify answer positions
        tokenizer: Tokenizer for decoding tokens (optional but recommended for robustness)
        number_token_seqs: Per-integer token-id sequences (Qwen-2 etc.).
            When supplied AND any sequence is longer than 1 token, the position-
            decomposed multi-token loss is used instead of the single-token path.
        digit_token_ids: 10-element list of single-digit token IDs (required when
            number_token_seqs has any multi-token sequence).

    Returns:
        Average cross-entropy loss across all queries
    """
    if (number_token_seqs is not None
        and digit_token_ids is not None
        and any(len(s) > 1 for s in number_token_seqs)):
        if labels is None:
            raise ValueError("labels are required for multi-token distributional loss")
        if train_terminator:
            if term_token_id < 0:
                raise ValueError("train_terminator=True requires term_token_id ≥ 0")
            return compute_distribution_loss_multitoken_with_term(
                logits=logits,
                labels=labels,
                ground_truth_bins=ground_truth_bins,
                number_token_seqs=number_token_seqs,
                digit_token_ids=digit_token_ids,
                term_token_id=term_token_id,
                num_queries=num_queries,
            )
        return compute_distribution_loss_multitoken(
            logits=logits,
            labels=labels,
            ground_truth_bins=ground_truth_bins,
            number_token_seqs=number_token_seqs,
            digit_token_ids=digit_token_ids,
            num_queries=num_queries,
        )

    batch_size = logits.size(0)
    device = logits.device

    # Create a set of number token IDs for fast lookup
    number_token_set = set(number_token_ids.tolist())

    total_loss = 0.0
    num_queries_total = 0

    for i in range(batch_size):
        # Get ground truth for this example [max_queries, 101]
        gt_bins = ground_truth_bins[i]

        # Determine number of valid queries for this example
        if num_queries is not None:
            n_queries = num_queries[i].item()
        else:
            n_queries = gt_bins.size(0)

        example_logits = logits[i]  # [seq_len, vocab_size]

        # Find answer positions: where labels != -100 AND label is a number token
        # These are the positions where we want to evaluate the distribution
        if labels is not None:
            example_labels = labels[i]  # [seq_len]
            answer_positions = []

            for pos in range(example_labels.size(0)):
                label_val = example_labels[pos].item()

                # Skip masked positions
                if label_val == -100:
                    continue

                # Method 1: Direct token ID match (fast)
                if label_val in number_token_set:
                    answer_positions.append(pos)
                    continue

                # Method 2: Decode token and check if it's a number 0-100 (robust for context-dependent tokenization)
                if tokenizer is not None:
                    try:
                        decoded = tokenizer.decode([label_val]).strip()
                        # Check if decoded string is a number 0-100
                        if decoded.isdigit():
                            num_val = int(decoded)
                            if 0 <= num_val <= 100:
                                answer_positions.append(pos)
                                continue
                    except:
                        pass

            # Fallback: if no number token positions found, find positions that decode to numbers
            # This handles multi-token numbers or any tokenization edge cases
            if len(answer_positions) == 0 and tokenizer is not None:
                for pos in range(example_labels.size(0)):
                    label_val = example_labels[pos].item()
                    if label_val == -100:
                        continue

                    try:
                        decoded = tokenizer.decode([label_val]).strip()
                        # Accept any numeric-looking token as potential answer position
                        if any(c.isdigit() for c in decoded):
                            answer_positions.append(pos)
                            if len(answer_positions) >= n_queries:
                                break
                    except:
                        # If decode fails, skip this position
                        continue

            # Last resort fallback: use first n_queries valid positions
            if len(answer_positions) == 0:
                for pos in range(example_labels.size(0)):
                    if example_labels[pos].item() != -100:
                        answer_positions.append(pos)
                        if len(answer_positions) >= n_queries:
                            break
        else:
            # Fallback: use last n_queries positions (less accurate)
            seq_len = example_logits.size(0)
            answer_positions = list(range(max(0, seq_len - n_queries), seq_len))

        # Use the last n_queries number positions — the final answer list
        # is always at the end of the response, after any scratchpad numbers.
        tail_positions = answer_positions[-n_queries:] if len(answer_positions) >= n_queries else answer_positions
        for q, pos in enumerate(tail_positions):
            gt_dist = gt_bins[q]  # [101]

            # Compute cross-entropy over full vocabulary distribution
            # Softmax over all vocab tokens, then extract log-probs for tokens 0-100
            # This is equivalent to CE against a target where tokens 0-100 have
            # ground truth probabilities and all other tokens have 0 probability
            full_logits = example_logits[pos]  # [vocab_size]
            log_pred = F.log_softmax(full_logits, dim=0)
            log_pred_numbers = log_pred[number_token_ids]  # [101]
            ce = -(gt_dist * log_pred_numbers).sum()

            total_loss += ce
            num_queries_total += 1

        # Log details for first example in first batch (for debugging)
        if i == 0 and num_queries_total > 0:
            import logging
            log = logging.getLogger(__name__)
            log.debug(f"Distribution loss details (example 0):")
            log.debug(f"  Found {len(answer_positions)} answer positions for {n_queries} queries")
            log.debug(f"  Position → Query mapping: {list(enumerate(tail_positions))}")
            if tokenizer is not None and labels is not None:
                for q, pos in enumerate(tail_positions[:min(3, n_queries)]):
                    label_val = labels[i][pos].item()
                    try:
                        decoded = tokenizer.decode([label_val]).strip()
                    except:
                        decoded = f"[{label_val}]"
                    gt_mean = sum(idx * gt_bins[q][idx].item() for idx in range(101))
                    log.debug(f"  Query {q} @ pos {pos}: label='{decoded}', GT mean={gt_mean:.1f}")

    if num_queries_total > 0:
        return total_loss / num_queries_total
    else:
        return torch.tensor(0.0, device=device)


def extract_number_probabilities(
    logits: torch.Tensor,
    number_token_ids: torch.Tensor,
    method: str = 'mean',
    labels: Optional[torch.Tensor] = None,
    query_idx: int = 0,
) -> torch.Tensor:
    """
    Extract probability distribution over tokens 0-100 from model logits.

    Args:
        logits: Model logits [seq_len, vocab_size] or [batch_size, seq_len, vocab_size]
        number_token_ids: Token IDs for numbers 0-100 [101]
        method: Method for aggregating across sequence
                'mean': Average logits across positions (fallback)
                'last': Use last position
                'answer_position': Use the position corresponding to query_idx
        labels: Labels tensor to identify answer positions [seq_len] or [batch_size, seq_len]
        query_idx: Which query's answer position to use (0-indexed)

    Returns:
        Probability distribution over 0-100 [101] or [batch_size, 101]
    """
    device = logits.device
    number_token_ids = number_token_ids.to(device)

    # Create a set of number token IDs for fast lookup
    number_token_set = set(number_token_ids.tolist())

    if logits.dim() == 3:
        # Batch mode [batch_size, seq_len, vocab_size]
        batch_size = logits.size(0)
        number_logits = logits[:, :, number_token_ids]  # [batch_size, seq_len, 101]

        if method == 'mean':
            avg_logits = number_logits.mean(dim=1)  # [batch_size, 101]
        elif method == 'last':
            avg_logits = number_logits[:, -1, :]  # [batch_size, 101]
        else:
            avg_logits = number_logits.mean(dim=1)

        return F.softmax(avg_logits, dim=-1)

    else:
        # Single example [seq_len, vocab_size]
        number_logits = logits[:, number_token_ids]  # [seq_len, 101]

        if method == 'answer_position' and labels is not None:
            # Find answer positions where labels are number tokens
            answer_positions = []
            for pos in range(labels.size(0)):
                label_val = labels[pos].item()
                if label_val != -100 and label_val in number_token_set:
                    answer_positions.append(pos)

            # Fallback: if no number token positions found, use first valid label positions
            if len(answer_positions) == 0:
                for pos in range(labels.size(0)):
                    if labels[pos].item() != -100:
                        answer_positions.append(pos)

            if query_idx < len(answer_positions):
                pos = answer_positions[query_idx]
                return F.softmax(number_logits[pos], dim=0)
            else:
                # Fallback to last valid position or mean
                if answer_positions:
                    return F.softmax(number_logits[answer_positions[-1]], dim=0)
                return F.softmax(number_logits.mean(dim=0), dim=0)
        elif method == 'mean':
            avg_logits = number_logits.mean(dim=0)  # [101]
        elif method == 'last':
            avg_logits = number_logits[-1, :]  # [101]
        else:
            avg_logits = number_logits.mean(dim=0)

        return F.softmax(avg_logits, dim=0)


def evaluate_predictions(
    predicted_dists: List[np.ndarray],
    ground_truth_dists: List[np.ndarray],
) -> Dict[str, float]:
    """
    Evaluate predicted distributions against ground truth.

    Args:
        predicted_dists: List of predicted distributions [num_queries x 101]
        ground_truth_dists: List of ground truth distributions [num_queries x 101]

    Returns:
        Dictionary with evaluation metrics:
        - kl_divergence: Average KL divergence
        - cross_entropy: Average cross-entropy
        - mean_abs_error: Average absolute error between means
    """
    metrics = {
        'kl_divergence': 0.0,
        'cross_entropy': 0.0,
        'cross_entropy_mean': 0.0,
        'cross_entropy_dist': 0.0,
        'mean_abs_error': 0.0,
        'mean_abs_error_dist': 0.0,
    }

    num_queries = len(predicted_dists)
    if num_queries == 0:
        metrics['pred_means'] = []
        metrics['gt_means'] = []
        return metrics

    pred_means = []
    gt_means = []

    for pred, gt in zip(predicted_dists, ground_truth_dists):
        # Ensure arrays are numpy
        pred = np.array(pred)
        gt = np.array(gt)

        # KL divergence: sum(p * log(p / q))
        # Add small epsilon to avoid log(0)
        eps = 1e-10
        kl = np.sum(gt * np.log((gt + eps) / (pred + eps)))
        metrics['kl_divergence'] += kl

        # Cross-entropy against full 101-bin distribution: -sum(p * log(q))
        ce_dist = -np.sum(gt * np.log(pred + eps))
        metrics['cross_entropy'] += ce_dist
        metrics['cross_entropy_dist'] += ce_dist

        # Cross-entropy against mean token only: -log q[gt_mean_int]
        values = np.arange(101)
        pred_mean = float(np.sum(values * pred))
        gt_mean = float(np.sum(values * gt))
        gt_mean_int = max(0, min(100, int(round(gt_mean))))
        ce_mean = -np.log(pred[gt_mean_int] + eps)
        metrics['cross_entropy_mean'] += ce_mean

        # MAE between expected values (scalar)
        metrics['mean_abs_error'] += abs(pred_mean - gt_mean)

        # MAE as L1 distance between full distributions
        metrics['mean_abs_error_dist'] += float(np.sum(np.abs(pred - gt)))

        pred_means.append(pred_mean)
        gt_means.append(gt_mean)

    # Average scalar metrics across queries
    for key in ('kl_divergence', 'cross_entropy', 'cross_entropy_mean', 'cross_entropy_dist',
                'mean_abs_error', 'mean_abs_error_dist'):
        metrics[key] /= num_queries

    metrics['pred_means'] = pred_means
    metrics['gt_means'] = gt_means

    return metrics
