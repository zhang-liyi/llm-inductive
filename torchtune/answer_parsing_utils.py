"""
answer_parsing_utils.py
-----------------------
Inference-time answer parsing for probabilistic-reasoning evaluations.

Mirrors the training-time logic in custom_lora_answer_only.py /
probabilistic_reasoning_utils.py:

  Training path (custom_lora_answer_only.py):
    _find_answer_positions(labels, number_token_ids)
        → collects every position in the ground-truth label sequence where a
          number token (0-100) appears, in order.
    extract_number_probabilities(method='answer_position', query_idx=q)
        → reads the model's softmax distribution over 0-100 at position q
          in that ordered list.

  Inference path (this file):
    The same ordered-position logic is reproduced for auto-regressive
    generation: every generated number token is recorded in the order it
    appears, and `query_idx=0` (the FIRST hit) is used as the answer for
    single-query scenarios.

Key design choice — use the FIRST number token hit, not the last:
  - Matches the training evaluator (query_idx=0).
  - Avoids decimal-suffix artifacts: "Answer: 100.0" tokenises as
    ..., 100, '.', 0, ... so last-hit = 0 (wrong), first-hit = 100 (correct).
  - Multi-query scenarios can be supported via query_idx > 0.
"""

from __future__ import annotations
from typing import List, NamedTuple, Optional, Tuple

import numpy as np
import torch
import os


# ---------------------------------------------------------------------------
# Chat-format context builder
# ---------------------------------------------------------------------------

def build_chat_context_tokens(tokenizer, user_prompt: str) -> List[int]:
    """
    Return token IDs for the full chat-formatted prefix ending with the
    assistant header, i.e.:

        [BOS] <|start_header_id|>user<|end_header_id|>\\n\\n
              {user_prompt}
              <|eot_id|>
              <|start_header_id|>assistant<|end_header_id|>\\n\\n

    This matches exactly what tokenize_messages produces up to (but not
    including) the answer token, so the model is in the correct
    "assistant responding" state — the same context it saw during training.

    The assistant header is always 4 tokens in Llama3:
        <|start_header_id|>  assistant  <|end_header_id|>  \\n\\n
    These are the first unmasked tokens in the tokenize_messages output.
    """
    from torchtune.data import Message
    msgs = [
        Message(role="user",      content=user_prompt, masked=True,  eot=True),
        Message(role="assistant", content="X",         masked=False, eot=True),
    ]
    tokens, mask = tokenizer.tokenize_messages(msgs)
    first_unmasked = next(i for i, m in enumerate(mask) if not m)
    context_end = first_unmasked + 4  # include up to and including \n\n
    return list(tokens[:context_end])


# ---------------------------------------------------------------------------
# Token-ID helpers
# ---------------------------------------------------------------------------

def get_number_token_ids(tokenizer) -> List[int]:
    """
    Return a list of 101 token IDs, one per integer 0-100.

    Matches the logic in probabilistic_reasoning_utils.get_number_token_ids
    but returns a plain Python list so it can be used in both tensor and
    set-lookup contexts without an extra .tolist() call.
    """
    test_output = " ".join(f"<{i}>" for i in range(101))
    tokens = tokenizer.encode(test_output, add_bos=False, add_eos=False)

    ids: List[int] = []
    for i in range(101):
        str_num = str(i)
        found = False
        for tok in tokens:
            if tokenizer.decode([tok]).strip() == str_num:
                ids.append(tok)
                found = True
                break
        if not found:
            direct = tokenizer.encode(str_num, add_bos=False, add_eos=False)
            ids.append(direct[0] if direct else -1)

    return ids


# ---------------------------------------------------------------------------
# Per-hit record
# ---------------------------------------------------------------------------

class NumberHit(NamedTuple):
    """One number token encountered during generation."""
    step: int            # decoding step index (0-based)
    value: int           # integer 0-100 that was generated
    dist: np.ndarray     # softmax over all 101 number tokens at this position


# ---------------------------------------------------------------------------
# Core inference function
# ---------------------------------------------------------------------------

@torch.no_grad()
def infer_answer(
    model,
    tokenizer,
    prompt: str,
    number_token_ids: List[int],
    device: str = "cuda",
    max_new_tokens: int = 32,
    query_idx: int = 0,
) -> Tuple[Optional[np.ndarray], Optional[int], List[NumberHit]]:
    """
    Run greedy decoding and extract the answer distribution using the
    FIRST number token (query_idx=0), matching the training-time evaluator.

    This mirrors:
        extract_number_probabilities(method='answer_position', query_idx=query_idx)
    from probabilistic_reasoning_utils.py, adapted for auto-regressive
    generation where labels are not available.

    Args:
        model:             The language model.
        tokenizer:         The tokenizer.
        prompt:            Full input prompt string.
        number_token_ids:  List[int] of length 101 from get_number_token_ids().
        device:            Torch device string.
        max_new_tokens:    Maximum tokens to generate.
        query_idx:         Which number-token occurrence to use as the answer
                           (0 = first, 1 = second, ...). Use 0 for all
                           single-query healthcare scenarios.

    Returns:
        pred_dist:   np.ndarray [101] — softmax probability over 0-100 at
                     the selected hit position, or None if not enough hits.
        greedy_val:  int — the greedy argmax at that position, or None.
        all_hits:    List[NumberHit] — every number token seen, in order.
                     Useful for verbose logging and sanity checks.
    """
    number_token_set = set(number_token_ids)
    number_token_ids_t = torch.tensor(number_token_ids, dtype=torch.long, device=device)

    # Build the full chat-formatted context (user turn + assistant header)
    # so the model is in the correct "assistant responding" state, matching
    # the tokenize_messages context it saw during training.
    context_ids = build_chat_context_tokens(tokenizer, prompt)
    generated = torch.tensor([context_ids], dtype=torch.long, device=device)

    all_hits: List[NumberHit] = []

    for step in range(max_new_tokens):
        logits = model(generated)
        if isinstance(logits, list):
            logits = torch.cat(logits, dim=1)
        next_logits = logits[:, -1, :]           # [1, vocab]
        next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
        next_id = next_token.item()

        if next_id in number_token_set:
            probs = torch.softmax(next_logits[0], dim=0)
            num_probs = probs[number_token_ids_t].cpu().numpy()
            num_probs = num_probs / num_probs.sum()   # renormalise to 101-token subspace
            value = number_token_ids.index(next_id)
            all_hits.append(NumberHit(step=step, value=value, dist=num_probs))

            # Return immediately once we have enough hits for query_idx
            # (matches _find_answer_positions which stops scanning once the
            # q-th position is found and passed to extract_number_probabilities)
            if len(all_hits) > query_idx:
                hit = all_hits[query_idx]
                # Continue generation to EOS so verbose callers can see full
                # response, but we already have the answer — generate the rest
                # only for the purpose of building the full response string.
                # To keep it cheap, finish the loop naturally.
                # (We still need to append this token and keep going for EOS.)

        generated = torch.cat([generated, next_token], dim=1)
        if next_id in tokenizer.stop_tokens:
            break

    if len(all_hits) <= query_idx:
        return None, None, all_hits

    hit = all_hits[query_idx]
    return hit.dist, hit.value, all_hits


# ---------------------------------------------------------------------------
# Verbose response decoder (for logging)
# ---------------------------------------------------------------------------

@torch.no_grad()
def infer_answer_verbose(
    model,
    tokenizer,
    prompt: str,
    number_token_ids: List[int],
    device: str = "cuda",
    max_new_tokens: int = 32,
    query_idx: int = 0,
) -> Tuple[Optional[np.ndarray], Optional[int], List[NumberHit], str]:
    """
    Same as infer_answer but also returns the full decoded response string.

    Returns:
        pred_dist, greedy_val, all_hits, full_response
    """
    number_token_set = set(number_token_ids)
    number_token_ids_t = torch.tensor(number_token_ids, dtype=torch.long, device=device)

    context_ids = build_chat_context_tokens(tokenizer, prompt)
    generated = torch.tensor([context_ids], dtype=torch.long, device=device)

    all_hits: List[NumberHit] = []
    decoded_tokens: List[str] = []

    for step in range(max_new_tokens):
        logits = model(generated)
        if isinstance(logits, list):
            logits = torch.cat(logits, dim=1)
        next_logits = logits[:, -1, :]
        next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
        next_id = next_token.item()

        decoded_tokens.append(tokenizer.decode([next_id]))

        if next_id in number_token_set:
            probs = torch.softmax(next_logits[0], dim=0)
            num_probs = probs[number_token_ids_t].cpu().numpy()
            num_probs = num_probs / num_probs.sum()
            value = number_token_ids.index(next_id)
            all_hits.append(NumberHit(step=step, value=value, dist=num_probs))

        generated = torch.cat([generated, next_token], dim=1)
        if next_id in tokenizer.stop_tokens:
            break

    full_response = "".join(decoded_tokens)

    if len(all_hits) <= query_idx:
        return None, None, all_hits, full_response

    hit = all_hits[query_idx]
    return hit.dist, hit.value, all_hits, full_response


# ---------------------------------------------------------------------------
# LLM-based answer parser
# ---------------------------------------------------------------------------

# Prompt given to the parser LLM.  The raw model response is inserted at {response}.
# We keep the instruction minimal and direct so the base model reliably outputs
# only an integer.
_PARSER_PROMPT_TEMPLATE = """\
The following is a model's response to a probabilistic reasoning question. \
The answer must be a single integer from 0 to 100. \
Extract that integer and output it alone, with no other text.

Model response: {response}

Integer (0-100):"""


class LLMParser:
    """
    Uses a pretrained Llama3-8B as a secondary model whose sole job is to
    read a raw model response and return the integer answer (0-100).

    Typical usage
    -------------
    parser = LLMParser.from_pretrained(model_path, device="cuda")

    # During evaluation, after obtaining the main model's full response:
    pred_dist, greedy_val = parser.parse(full_response)

    # Or drive the full pipeline in one call:
    pred_dist, greedy_val, all_hits, full_response, parser_val = (
        infer_answer_llm_parsed(main_model, tokenizer, prompt,
                                number_token_ids, parser, device)
    )

    Design notes
    ------------
    - The parser's own first number token is used as the answer (same
      first-hit logic as infer_answer), so the parser only needs to output
      a single integer.
    - The distribution returned is the parser's softmax over 0-100 at that
      position, not the main model's.  This is the probability the parser
      assigns to each possible integer interpretation.
    - The parser model is kept on the same device as the main model and
      reuses the same number_token_ids list.
    - max_parser_tokens=8 is enough for any integer 0-100 plus an EOS token.
    """

    def __init__(self, model, tokenizer, number_token_ids: List[int], device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.number_token_ids = number_token_ids
        self.device = device

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        device: str = "cuda",
        dtype=None,
    ) -> "LLMParser":
        """
        Load a pretrained Llama3-8B from model_path and return a ready LLMParser.

        Args:
            model_path: Path to the HF-format model directory (same path used
                        by load_pretrained_model in evaluate_healthcare.py).
            device:     Torch device string.
            dtype:      torch dtype; defaults to bfloat16.
        """
        import torch
        from torchtune import training
        from torchtune.models.llama3 import llama3_8b, llama3_tokenizer
        from torchtune.training.checkpointing._checkpointer import FullModelHFCheckpointer

        if dtype is None:
            dtype = torch.bfloat16

        print(f"[LLMParser] Loading pretrained parser model from: {model_path}")
        tokenizer = llama3_tokenizer(path=os.path.join(model_path, "tokenizer.model"))

        # Find checkpoint files (same helper logic as evaluate_healthcare.py)
        ckpt_files = None
        for ext in (".safetensors", ".bin"):
            files = sorted(
                f for f in os.listdir(model_path)
                if f.endswith(ext)
            )
            if files:
                ckpt_files = files
                break
        if ckpt_files is None:
            raise FileNotFoundError(f"No checkpoint files found in {model_path}")

        checkpointer = FullModelHFCheckpointer(
            checkpoint_dir=model_path,
            checkpoint_files=ckpt_files,
            model_type="LLAMA3",
            output_dir=os.path.join(model_path, os.pardir),
        )
        ckpt = checkpointer.load_checkpoint()

        with training.set_default_dtype(dtype), torch.device(device):
            model = llama3_8b()
        model.load_state_dict(ckpt[training.MODEL_KEY], strict=True)
        model.eval()
        print("[LLMParser] Parser model ready.")

        number_token_ids = get_number_token_ids(tokenizer)
        return cls(model, tokenizer, number_token_ids, device)

    @torch.no_grad()
    def parse(
        self,
        response_text: str,
        max_parser_tokens: int = 8,
    ) -> Tuple[Optional[np.ndarray], Optional[int]]:
        """
        Feed response_text to the parser LLM and return the extracted integer
        answer as (pred_dist, greedy_val).

        Args:
            response_text:     The raw string the main model generated.
            max_parser_tokens: Max tokens to generate; 8 is enough for any
                               0-100 integer plus EOS.

        Returns:
            pred_dist:  np.ndarray [101] — parser's softmax over 0-100 at
                        the answer token position, or None if the parser
                        produces no number token.
            greedy_val: int (0-100) the parser chose, or None.
        """
        prompt = _PARSER_PROMPT_TEMPLATE.format(response=response_text.strip())
        pred_dist, greedy_val, _ = infer_answer(
            self.model,
            self.tokenizer,
            prompt,
            self.number_token_ids,
            device=self.device,
            max_new_tokens=max_parser_tokens,
            query_idx=0,
        )
        return pred_dist, greedy_val


# ---------------------------------------------------------------------------
# Full pipeline: main model → LLM parser
# ---------------------------------------------------------------------------

@torch.no_grad()
def infer_answer_llm_parsed(
    model,
    tokenizer,
    prompt: str,
    number_token_ids: List[int],
    parser: LLMParser,
    device: str = "cuda",
    max_new_tokens: int = 32,
    query_idx: int = 0,
) -> Tuple[Optional[np.ndarray], Optional[int], List[NumberHit], str, Optional[int]]:
    """
    Two-stage pipeline:
      1. Run the main model with infer_answer_verbose to get the full response
         string and the first-hit distribution/value.
      2. Pass the full response to the LLMParser to obtain a clean integer.

    The distribution returned (pred_dist) comes from the PARSER's softmax at
    its answer position, not the main model's.  This is the right choice when
    the main model's output format is not a clean 0-100 integer (e.g. decimal
    probabilities) — the parser re-grounds the distribution in integer space.

    Args:
        model:             Main language model.
        tokenizer:         Tokenizer for the main model.
        prompt:            Full input prompt.
        number_token_ids:  From get_number_token_ids(tokenizer).
        parser:            A loaded LLMParser instance.
        device:            Torch device string.
        max_new_tokens:    Max tokens for the main model.
        query_idx:         Which number-token hit from the main model to use
                           as the first-hit fallback (passed through to
                           infer_answer_verbose).

    Returns:
        parser_dist:    np.ndarray [101] from the parser (or None).
        parser_val:     int chosen by the parser (or None).
        main_hits:      List[NumberHit] from the main model (for logging).
        full_response:  The main model's full decoded response string.
        main_val:       int chosen by the first-hit parser on the main model
                        (useful as a comparison baseline).
    """
    # Stage 1 — main model
    _, main_val, main_hits, full_response = infer_answer_verbose(
        model, tokenizer, prompt, number_token_ids,
        device=device, max_new_tokens=max_new_tokens,
        query_idx=query_idx,
    )

    # Stage 2 — parser LLM
    parser_dist, parser_val = parser.parse(full_response)

    return parser_dist, parser_val, main_hits, full_response, main_val
