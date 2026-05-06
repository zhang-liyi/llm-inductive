"""Shared helpers for adding Qwen2-7B-Instruct support to the calibration
eval scripts (evaluate_openestimate.py / evaluate_bayesian_teaching.py /
evaluate_text_classification.py).

Provides:
  infer_arch_from_path(path) → 'llama3' | 'qwen2_7b'
  load_pretrained_model_and_tokenizer(path, arch, device, dtype)
  load_lora_model_and_tokenizer(ckpt_dir, arch, device, dtype)
  setup_number_tokens(tokenizer) → dict with multi-token-aware helpers

Path-based arch detection: anything containing 'qwen' (case-insensitive) →
qwen2_7b; otherwise llama3.

Tokenizer setup mirrors `custom_lora_answer_only.py:_setup_number_token_helpers`:
  - number_token_ids   : list[int] (length 101) — first-digit-only fallback
                         in multi-token tokenizers (kept for back-compat).
  - number_token_seqs  : list[list[int]] — true per-integer tokenizations.
  - digit_token_ids    : list[int] — the 10 digits "0".."9".
  - single_token       : bool — True iff every integer 0..100 is a single
                         token (Llama-3) — single-token code paths can
                         use the legacy 101-id lookup. False (Qwen-2)
                         dispatches to multi-token logic.
"""
from __future__ import annotations

import os
from typing import Tuple

import torch
from transformers import AutoTokenizer

from torchtune import training
from torchtune.training.checkpointing._checkpointer import FullModelHFCheckpointer

# Llama-3
from torchtune.models.llama3 import (
    llama3_8b, lora_llama3_8b, llama3_tokenizer,
)
# Qwen-2
from torchtune.models.qwen2 import (
    qwen2_7b, lora_qwen2_7b, qwen2_tokenizer,
)
from torchtune.modules.peft import get_adapter_params, set_trainable_params

DEFAULT_LLAMA_PATH = (
    "<DATA_ROOT>/resources/"
    "models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/"
    "e1945c40cd546c78e41f1151f4db032b271faeaa"
)
DEFAULT_QWEN_PATH = (
    "<DATA_ROOT>/resources/qwen/Qwen2-7B-Instruct"
)

_LORA_KW = dict(
    lora_attn_modules=["q_proj", "v_proj", "output_proj"],
    apply_lora_to_mlp=True,
    apply_lora_to_output=False,
    lora_rank=8,
    lora_alpha=16,
    lora_dropout=0.0,
)


# ── arch detection ─────────────────────────────────────────────────────────
def infer_arch_from_path(path: str) -> str:
    """Return 'qwen2_7b' if path contains 'qwen' (case-insensitive); else
    'llama3'. Pass `path` as either the model_path (pretrained) or
    ckpt_dir (LoRA)."""
    return 'qwen2_7b' if 'qwen' in path.lower() else 'llama3'


def default_pretrained_path(arch: str) -> str:
    return DEFAULT_QWEN_PATH if arch == 'qwen2_7b' else DEFAULT_LLAMA_PATH


# ── model loading ──────────────────────────────────────────────────────────
def _torch_dtype(dtype: str):
    return {"bfloat16": torch.bfloat16, "float16": torch.float16,
            "float32": torch.float32}.get(dtype, torch.bfloat16)


def load_pretrained_model_and_tokenizer(
    model_path: str, arch: str = 'auto',
    device: str = "cuda", dtype: str = "bfloat16",
) -> Tuple[object, object]:
    if arch == 'auto':
        arch = infer_arch_from_path(model_path)
    torch_dtype = _torch_dtype(dtype)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ckpt_files = sorted(f for f in os.listdir(model_path) if f.endswith('.safetensors'))
    print(f"[{arch}] loading pretrained model from {model_path}  ({len(ckpt_files)} shards)")
    checkpointer = FullModelHFCheckpointer(
        checkpoint_dir=model_path,
        checkpoint_files=ckpt_files,
        model_type='QWEN2' if arch == 'qwen2_7b' else 'LLAMA3',
        output_dir=os.path.join(model_path, os.pardir),
    )
    ckpt = checkpointer.load_checkpoint()

    with training.set_default_dtype(torch_dtype), torch.device(device):
        model = qwen2_7b() if arch == 'qwen2_7b' else llama3_8b()
    model.load_state_dict(ckpt[training.MODEL_KEY], strict=True)
    model.eval()
    return model, tokenizer, arch


def load_lora_model_and_tokenizer(
    ckpt_dir: str, arch: str = 'auto',
    device: str = "cuda", dtype: str = "bfloat16",
) -> Tuple[object, object]:
    if arch == 'auto':
        arch = infer_arch_from_path(ckpt_dir)
    torch_dtype = _torch_dtype(dtype)

    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ckpt_files = sorted(
        f for f in os.listdir(ckpt_dir)
        if f.endswith('.safetensors') and 'adapter' not in f.lower()
    )
    if not ckpt_files:
        raise FileNotFoundError(f"No merged safetensors files found in {ckpt_dir}")
    print(f"[{arch}] loading LoRA ckpt from {ckpt_dir}  ({len(ckpt_files)} shards)")
    checkpointer = FullModelHFCheckpointer(
        checkpoint_dir=ckpt_dir,
        checkpoint_files=ckpt_files,
        model_type='QWEN2' if arch == 'qwen2_7b' else 'LLAMA3',
        output_dir=os.path.join(ckpt_dir, os.pardir),
    )
    ckpt = checkpointer.load_checkpoint()

    with training.set_default_dtype(torch_dtype), torch.device(device):
        model = (lora_qwen2_7b(**_LORA_KW) if arch == 'qwen2_7b'
                 else lora_llama3_8b(**_LORA_KW))
    adapter_params = get_adapter_params(model)
    set_trainable_params(model, adapter_params)
    model.load_state_dict(ckpt[training.MODEL_KEY], strict=False)
    model.eval()
    return model, tokenizer, arch


# ── number-token setup (multi-token aware) ─────────────────────────────────
def setup_number_tokens(tokenizer):
    """Return a dict with:
      number_token_ids   : list[int] of length 101 (first-token fallback)
      number_token_seqs  : list[list[int]] — true per-integer token sequences
      digit_token_ids    : list[int] — 10 digit tokens "0".."9"
      single_token       : True iff every integer 0..100 is a single token

    Mirrors what custom_lora_answer_only.py / probabilistic_reasoning_utils.py
    use; works for both Llama-3 (all single-token) and Qwen-2 (10..100
    multi-token).
    """
    number_token_seqs = []
    for i in range(101):
        ids = tokenizer.encode(str(i), add_special_tokens=False)
        if not ids:
            ids = [0]
        number_token_seqs.append(list(ids))

    single_token = all(len(s) == 1 for s in number_token_seqs)

    digit_token_ids = []
    for d in range(10):
        ids = tokenizer.encode(str(d), add_special_tokens=False)
        digit_token_ids.append(ids[0] if ids else 0)

    # First-token fallback (legacy 101-id lookup).
    number_token_ids = [s[0] for s in number_token_seqs]

    return {
        'number_token_ids': number_token_ids,
        'number_token_seqs': number_token_seqs,
        'digit_token_ids': digit_token_ids,
        'single_token': single_token,
    }
