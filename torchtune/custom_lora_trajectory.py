# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys
import time
import glob

from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union
from warnings import warn

import torch
import torch.nn.functional as F
import torchtune.modules.common_utils as common_utils
from omegaconf import DictConfig, OmegaConf

from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Subset
from torchtune import config, modules, training, utils
from torchtune.data import Message
from torchtune.config._utils import _get_component_from_path
from torchtune.modules.peft import (
    get_adapter_params,
    get_adapter_state_dict,
    get_lora_module_names,
    get_merged_lora_ckpt,
    set_trainable_params,
    validate_missing_and_unexpected_for_lora,
)
from torchtune.recipe_interfaces import FTRecipeInterface
from torchtune.training import DummyProfiler, PROFILER_KEY

from transformers import AutoTokenizer

from tqdm import tqdm
from copy import deepcopy
import numpy as np

import logging
import json
import os
import re
import shutil

from probabilistic_reasoning_utils import (
    get_number_token_ids,
    _find_answer_positions,
    evaluate_predictions,
    extract_number_probabilities,
    compute_probabilistic_loss,
    compute_distribution_loss,
    ProbabilisticReasoningDataset,
    probabilistic_reasoning_collate_fn,
)
from torchtune.rlhf.rewards import masked_mean
from analyze_validation_predictions import (
    compute_direction_metrics,
    analyze_learning_vs_centering,
    plot_prediction_vs_ground_truth,
    plot_direction_analysis,
    plot_comparison,
)

VAL_SUBSET_SIZE = 1000
VAL_SUBSET_SEED = 42


def _build_bins_index(json_path: str) -> Dict[str, List[List[float]]]:
    """
    Build a mapping  scenario_id -> [bins_q0, bins_q1, ...]  from a
    probabilistic_reasoning JSON file.

    Each entry in the JSON corresponds to one query of one scenario.
    Entries for the same scenario are consecutive and in query order.
    ``scenario_id`` is the stem of ``metadata.scenario_file``, e.g.
    ``scenarios/gemini-P-0-C-1-R-1-N-4-2.txt`` → ``gemini-P-0-C-1-R-1-N-4-2``.
    """
    if not os.path.exists(json_path):
        return {}
    with open(json_path) as f:
        entries = json.load(f)
    index: Dict[str, List[List[float]]] = {}
    for entry in entries:
        meta = entry.get("metadata", {})
        scenario_file = meta.get("scenario_file", "")
        if not scenario_file:
            continue
        scenario_id = os.path.splitext(os.path.basename(scenario_file))[0]
        bins = entry.get("bins", None)
        if bins is None:
            continue
        # Each single-query entry has bins = [[v0, ..., v100]]
        bins_flat: List[float] = bins[0] if isinstance(bins[0], list) else bins
        index.setdefault(scenario_id, []).append(bins_flat)
    return index


def _get_val_subset(ds, size: int = VAL_SUBSET_SIZE):
    """Return a fixed random subset of ``size`` examples from ds."""
    if len(ds) <= size:
        return ds
    rng = np.random.RandomState(VAL_SUBSET_SEED)
    indices = sorted(rng.choice(len(ds), size, replace=False).tolist())
    return Subset(ds, indices)


def _parse_answer(text: str) -> List[int]:
    """
    Parse the final answer from a generated response.

    Primary format: "The answer is: <N>, <M>, ..."
        Single:  "The answer is: <77>"               → [77]
        Multi:   "The answer is: <60>, <75>, <33>"   → [60, 75, 33]

    Legacy fallbacks (backward compatibility):
        "[<N>, <M>, ...]"   → [N, M, ...]
        "[60, 75, 33]"      → [60, 75, 33]

    Returns [] if no parseable answer is found.
    """
    # Primary: "The answer is: <N>, <M>, ..." — parse all <N> on that answer line
    m = re.search(r'The answer is:\s*((?:<\d+>(?:,\s*)?)+)', text)
    if m:
        return [int(x) for x in re.findall(r'<(\d+)>', m.group(1))]
    # Legacy: angle-bracket list "[<N>, <M>, ...]"
    ab_lists = re.findall(r'\[\s*<\d+>(?:\s*,\s*<\d+>)*\s*\]', text)
    if ab_lists:
        return [int(x) for x in re.findall(r'<(\d+)>', ab_lists[-1])]
    # Legacy: bare bracketed integer list "[60, 75, 33]"
    all_lists = re.findall(r'\[\s*\d+(?:\s*,\s*\d+)*\s*\]', text)
    if all_lists:
        return [int(x) for x in re.findall(r'\d+', all_lists[-1])]
    return []


# ---------------------------------------------------------------------------
# DPO Dataset
# ---------------------------------------------------------------------------

class DPOPreferencePairDataset(Dataset):
    """
    Dataset for DPO training that loads preference pairs from a data directory.

    Expected file structure::

        {data_dir}/
            {prompt_id}_positive.json   # chosen (positive) responses for this prompt
            {prompt_id}_negative.json   # rejected (negative) responses for this prompt

    Each JSON file has the schema::

        {
            "prompt": "<full prompt text>",
            "responses": ["response text 1", "response text 2", ...]
        }

    For each prompt, we create one triplet (prompt, chosen, rejected) per
    positive response, sampling a random negative from that prompt's negatives.
    The caller is responsible for passing the correct split subdirectory
    (e.g. dpo_positives/train); no internal split is applied.
    """

    def __init__(
        self,
        data_dir: str,
        tokenizer,
        max_seq_len: int = 2048,
        seed: int = 42,
        neg_dir: Optional[str] = None,
        bins_index: Optional[Dict[str, List[List[float]]]] = None,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.rng = np.random.RandomState(seed)
        _bins_index = bins_index or {}

        # Discover all positive-response files and match with negatives
        positive_files = sorted(glob.glob(os.path.join(data_dir, "*_positive.json")))
        if not positive_files:
            raise FileNotFoundError(
                f"No '*_positive.json' files found under '{data_dir}'. "
                "Check that dpo_data_dir is set correctly."
            )

        _neg_dir = neg_dir if neg_dir is not None else data_dir

        # triplet: (prompt, chosen, rejected, answers, format_ok, bins_or_none)
        self.triplets: List[Tuple[str, str, str, List[int], bool, Optional[List[List[float]]]]] = []
        for pos_file in positive_files:
            prompt_id = os.path.basename(pos_file).replace("_positive.json", "")
            neg_file = os.path.join(_neg_dir, f"{prompt_id}_negative.json")
            if not os.path.exists(neg_file):
                continue

            with open(pos_file) as f:
                pos_data = json.load(f)
            with open(neg_file) as f:
                neg_data = json.load(f)

            prompt = pos_data["prompt"]
            positives: List[str] = pos_data["responses"]
            negatives: List[str] = neg_data["responses"]

            if not positives or not negatives:
                continue

            # bins_list: [[101 floats], [101 floats], ...] one per query, or None
            bins_list: Optional[List[List[float]]] = _bins_index.get(prompt_id, None)

            for chosen in positives:
                rejected = negatives[self.rng.randint(len(negatives))]
                answers = _parse_answer(chosen)
                # format_ok: True when the angle-bracket format was used
                format_ok = bool(re.search(r'The answer is:\s*<\d+>', chosen))
                if not answers:
                    answers = [-1]
                self.triplets.append((prompt, chosen, rejected, answers, format_ok, bins_list))

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        prompt, chosen, rejected, answers, format_ok, bins_list = self.triplets[idx]

        def _tokenize(response: str) -> Tuple[List[int], List[int]]:
            messages = [
                Message(role="user", content=prompt, masked=True, eot=True),
                Message(role="assistant", content=response, masked=False, eot=True),
            ]
            tokens, mask = self.tokenizer.tokenize_messages(messages)
            tokens = tokens[: self.max_seq_len]
            mask = mask[: self.max_seq_len]
            labels = [-100 if m else t for t, m in zip(tokens, mask)]
            return tokens, labels

        chosen_ids, chosen_labels = _tokenize(chosen)
        rejected_ids, rejected_labels = _tokenize(rejected)

        n_queries = len(answers) if answers != [-1] else 0
        if bins_list is not None and n_queries > 0:
            # bins_list: [[101 floats] * n_queries] in query order
            gt_bins = torch.tensor(bins_list[:n_queries], dtype=torch.float32)  # [n_queries, 101]
        else:
            gt_bins = torch.zeros(max(n_queries, 1), 101, dtype=torch.float32)

        return {
            "input_ids_chosen": torch.tensor(chosen_ids, dtype=torch.long),
            "labels_chosen": torch.tensor(chosen_labels, dtype=torch.long),
            "input_ids_rejected": torch.tensor(rejected_ids, dtype=torch.long),
            "labels_rejected": torch.tensor(rejected_labels, dtype=torch.long),
            "ground_truth": torch.tensor(answers, dtype=torch.long),  # [N_queries]
            "format_ok": torch.tensor(format_ok, dtype=torch.bool),
            "ground_truth_bins": gt_bins,                              # [n_queries, 101]
            "num_queries": torch.tensor(n_queries, dtype=torch.long),
        }


def dpo_collate_fn(
    batch: List[Dict[str, torch.Tensor]],
    pad_id: int,
    ignore_index: int = -100,
) -> Dict[str, torch.Tensor]:
    """Pad chosen and rejected sequences to the longest in the batch."""

    def _pad(seqs: List[torch.Tensor], pad_val: int) -> torch.Tensor:
        max_len = max(s.size(0) for s in seqs)
        return torch.stack(
            [F.pad(s, (0, max_len - s.size(0)), value=pad_val) for s in seqs]
        )

    # Pad ground_truth_bins to [B, max_queries, 101]
    max_queries = max(b["ground_truth_bins"].size(0) for b in batch)
    gt_bins_padded = torch.stack([
        F.pad(b["ground_truth_bins"], (0, 0, 0, max_queries - b["ground_truth_bins"].size(0)))
        for b in batch
    ])  # [B, max_queries, 101]

    return {
        "input_ids_chosen": _pad([b["input_ids_chosen"] for b in batch], pad_id),
        "labels_chosen": _pad([b["labels_chosen"] for b in batch], ignore_index),
        "input_ids_rejected": _pad([b["input_ids_rejected"] for b in batch], pad_id),
        "labels_rejected": _pad([b["labels_rejected"] for b in batch], ignore_index),
        "ground_truth": _pad([b["ground_truth"] for b in batch], -1),  # [B, max_queries]
        "format_ok": torch.stack([b["format_ok"] for b in batch]),     # [B]
        "ground_truth_bins": gt_bins_padded,                           # [B, max_queries, 101]
        "num_queries": torch.stack([b["num_queries"] for b in batch]), # [B]
    }


# ---------------------------------------------------------------------------
# Single-query dataset (probabilistic_reasoning_val.json / _test.json)
# ---------------------------------------------------------------------------

class SingleQueryDataset(Dataset):
    """
    Evaluation-only dataset.  Loads JSON files of the form::

        [{"input": "<prompt text>", "output": <int>, ...}, ...]

    Each item contains exactly one query with one integer ground-truth answer.
    Returns prompt tokens only (no reference response).
    """

    def __init__(
        self,
        json_path: str,
        tokenizer,
        max_seq_len: int = 2048,
        instruction_override: Optional[str] = None,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        with open(json_path) as f:
            data = json.load(f)

        items = []
        for item in data:
            prompt = item["input"]
            if instruction_override is not None:
                # Replace the instruction prefix (everything before the first
                # blank line) with the override, leaving the scenario intact.
                parts = prompt.split("\n\n", 1)
                prompt = instruction_override + ("\n\n" + parts[1] if len(parts) > 1 else "")
            bins = item.get("bins", None)
            # bins is [[v0, ..., v100]] for a single-query entry; take the inner list
            bins_flat: Optional[List[float]] = bins[0] if (bins and isinstance(bins[0], list)) else bins
            items.append((prompt, int(str(item["output"]).strip("<>")), bins_flat))
        self.items: List[Tuple[str, int, Optional[List[float]]]] = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        prompt_text, gt_int, bins_flat = self.items[idx]
        # Workaround for torchtune tokenize_messages quirk: empty assistant
        # content with masked=True/eot=False appends trailing EOM/EOT tokens
        # that close the conversation.  Use dummy content "X" + eot=True and
        # slice off everything from the assistant content onward, keeping only
        # the 4-token assistant header (<|start_header_id|>assistant<|end_header_id|>\n\n).
        messages = [
            Message(role="user",      content=prompt_text, masked=True,  eot=True),
            Message(role="assistant", content="X",         masked=False, eot=True),
        ]
        tokens, mask = self.tokenizer.tokenize_messages(messages)
        first_unmasked = next((i for i, m in enumerate(mask) if not m), len(tokens))
        # Keep the 4-token assistant header, drop the dummy content + eot
        tokens = list(tokens[: first_unmasked + 4])
        tokens = tokens[: self.max_seq_len]
        if bins_flat is not None:
            gt_bins = torch.tensor([bins_flat], dtype=torch.float32)  # [1, 101]
        else:
            gt_bins = torch.zeros(1, 101, dtype=torch.float32)
        return {
            "prompt_ids": torch.tensor(tokens, dtype=torch.long),
            "ground_truth_single": torch.tensor(gt_int, dtype=torch.long),
            "ground_truth_bins": gt_bins,  # [1, 101]
        }


def single_query_collate_fn(
    batch: List[Dict[str, torch.Tensor]],
    pad_id: int,
) -> Dict[str, torch.Tensor]:
    """Pad prompt sequences to the longest in the batch."""
    max_len = max(b["prompt_ids"].size(0) for b in batch)
    prompt_ids = torch.stack([
        F.pad(b["prompt_ids"], (0, max_len - b["prompt_ids"].size(0)), value=pad_id)
        for b in batch
    ])
    gt = torch.stack([b["ground_truth_single"] for b in batch])
    gt_bins = torch.stack([b["ground_truth_bins"] for b in batch])  # [B, 1, 101]
    return {"prompt_ids": prompt_ids, "ground_truth_single": gt, "ground_truth_bins": gt_bins}


# ---------------------------------------------------------------------------
# SFT Trajectory dataset (thinking-trajectory phase-1 data)
# ---------------------------------------------------------------------------

class SFTTrajDataset(Dataset):
    """
    Dataset for supervised fine-tuning on thinking-trajectory examples.

    Loads JSON files of the form::

        [{"scenario_id": "...", "input": "<prompt>", "output": "<scratchpad+answer>",
          "bins": [[101 floats], ...], "means": [int, ...]}, ...]

    The ``input`` is used as the user turn and ``output`` as the assistant turn.
    Labels are masked on the prompt (user turn); CE loss is computed on the full
    assistant output (scratchpad + "The answer is: <x>, <y>").

    ``bins`` (per-query 101-bin posterior) and ``means`` are passed through for
    use in distributional loss mode.  If the JSON lacks these keys (legacy
    files), zero bins and -1 means are returned.
    """

    def __init__(
        self,
        json_path: str,
        tokenizer,
        max_seq_len: int = 2048,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        with open(json_path) as f:
            data = json.load(f)

        self.items = []
        for entry in data:
            bins = entry.get("bins", None)   # [[101], [101], ...] or None
            means = entry.get("means", None) # [int, ...] or None
            if bins is not None:
                bins_t = torch.tensor(bins, dtype=torch.float32)    # [n_queries, 101]
            else:
                bins_t = torch.zeros(1, 101, dtype=torch.float32)
            n_queries = bins_t.size(0)
            self.items.append((entry["input"], entry["output"], bins_t, n_queries))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        prompt_text, response_text, bins, n_queries = self.items[idx]
        messages = [
            Message(role="user",      content=prompt_text,   masked=True,  eot=True),
            Message(role="assistant", content=response_text, masked=False, eot=True),
        ]
        tokens, mask = self.tokenizer.tokenize_messages(messages)
        tokens = tokens[: self.max_seq_len]
        mask   = mask[: self.max_seq_len]
        labels = [-100 if m else t for t, m in zip(tokens, mask)]
        return {
            "input_ids": torch.tensor(tokens, dtype=torch.long),
            "labels":    torch.tensor(labels, dtype=torch.long),
            "ground_truth_bins": bins,                                  # [n_queries, 101]
            "num_queries": torch.tensor(n_queries, dtype=torch.long),
        }


def sft_traj_collate_fn(
    batch: List[Dict[str, torch.Tensor]],
    pad_id: int,
    ignore_index: int = -100,
) -> Dict[str, torch.Tensor]:
    """Pad sequences and bins to the longest in the batch."""
    max_len = max(b["input_ids"].size(0) for b in batch)
    input_ids = torch.stack([
        F.pad(b["input_ids"], (0, max_len - b["input_ids"].size(0)), value=pad_id)
        for b in batch
    ])
    labels = torch.stack([
        F.pad(b["labels"], (0, max_len - b["labels"].size(0)), value=ignore_index)
        for b in batch
    ])
    # Pad bins to max queries across batch
    max_queries = max(b["ground_truth_bins"].size(0) for b in batch)
    bins_list = []
    for b in batch:
        bq = b["ground_truth_bins"]
        if bq.size(0) < max_queries:
            bq = torch.cat([bq, torch.zeros(max_queries - bq.size(0), 101)], dim=0)
        bins_list.append(bq)
    gt_bins = torch.stack(bins_list)  # [B, max_queries, 101]
    num_queries = torch.stack([b["num_queries"] for b in batch])
    return {
        "input_ids": input_ids,
        "labels": labels,
        "ground_truth_bins": gt_bins,
        "num_queries": num_queries,
    }


# ---------------------------------------------------------------------------
# Bayesian Teaching program-SFT dataset
# ---------------------------------------------------------------------------

DEFAULT_BT_DATA_PATH = (
    "./"
    "data_processing/bayesian_teaching_test_base.jsonl"
)
DEFAULT_BT_PROGRAMS_DIR = (
    "./"
    "posterior_sampling_pytorch/bt_programs"
)
DEFAULT_BT_SCRATCHPADS_DIR = (
    "./"
    "posterior_sampling_pytorch/bt_scratchpads"
)


def _bt_rewrite_prompt_free_gen(prompt: str) -> str:
    """Free-gen rewrite matching evaluate_bayesian_teaching.rewrite_prompt_free_gen."""
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


def _bt_load_jsonl(path: str) -> List[dict]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _bt_reasoning_path(
    reasoning_dir: str,
    reasoning: str,
    task: str,
    idx: int,
    source: Optional[str] = None,
) -> str:
    stem = source if source else task
    if task == "webshop":
        scenario_id = f"bt-webshop-{stem}-{idx}" if stem != "webshop" else f"bt-webshop-{idx}"
    else:
        scenario_id = f"bt-{stem}-{idx}"
    if reasoning == "program":
        return os.path.join(reasoning_dir, f"pg-{scenario_id}.py")
    return os.path.join(reasoning_dir, f"{scenario_id}.json")


def _bt_load_reasoning(path: str, reasoning: str) -> Optional[str]:
    if not os.path.exists(path):
        return None
    if reasoning == "program":
        with open(path) as f:
            txt = f.read().strip()
        return txt or None
    # scratchpad: JSON with "scratchpad" field
    with open(path) as f:
        obj = json.load(f)
    txt = (obj.get("scratchpad") or "").strip()
    return txt or None


def _bt_build_items(
    data_path: str,
    reasoning_dir: str,
    reasoning: str = "program",
    tasks: Optional[List[str]] = None,
) -> List[dict]:
    """Return BT examples that have a reasoning file (program or scratchpad)."""
    raw = _bt_load_jsonl(data_path)
    items = []
    for ex in raw:
        task = ex.get("task")
        if tasks is not None and task not in tasks:
            continue
        path = _bt_reasoning_path(
            reasoning_dir, reasoning, task, ex.get("idx"), source=ex.get("source"),
        )
        text = _bt_load_reasoning(path, reasoning)
        if text is None:
            continue
        gt_match = re.search(r"<([123])>", ex.get("output", ""))
        if not gt_match:
            continue
        items.append({
            "task":     task,
            "idx":      ex.get("idx"),
            "source":   ex.get("source"),
            "prompt":   _bt_rewrite_prompt_free_gen(ex["input"]),
            "program":  text,   # keyed "program" but holds scratchpad text in scratchpad mode
            "gt":       int(gt_match.group(1)),
        })
    return items


class BTProgramSFTDataset(Dataset):
    """
    Training dataset for BT-program SFT.

    Each example: user = BT question (free-gen rewrite); assistant = program
    source (no final-answer line).  Loss is computed over program tokens only
    (user tokens are masked).
    """

    def __init__(self, items: List[dict], tokenizer, max_seq_len: int = 8192,
                 include_answer: bool = False):
        self.items = items
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.include_answer = include_answer

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        it = self.items[i]
        if self.include_answer:
            content = f"{it['program']}\n\nThe answer is: <{int(it['gt'])}>"
        else:
            content = it["program"]
        messages = [
            Message(role="user",      content=it["prompt"], masked=True,  eot=True),
            Message(role="assistant", content=content,      masked=False, eot=True),
        ]
        tokens, mask = self.tokenizer.tokenize_messages(messages)
        tokens = tokens[: self.max_seq_len]
        mask   = mask[: self.max_seq_len]
        labels = [-100 if m else t for t, m in zip(tokens, mask)]
        if not self.include_answer:
            # Mask the trailing <|eot_id|>: we don't want to train the model to
            # terminate after the program — at eval/inference time the assistant
            # turn should continue with "\n\nThe answer is: <N>".
            for j in range(len(labels) - 1, -1, -1):
                if labels[j] != -100:
                    labels[j] = -100
                    break
        return {
            "input_ids": torch.tensor(tokens, dtype=torch.long),
            "labels":    torch.tensor(labels, dtype=torch.long),
            "ground_truth_bins": torch.zeros(1, 101, dtype=torch.float32),
            "num_queries":       torch.tensor(0, dtype=torch.long),
        }


class BTProgramEvalDataset(Dataset):
    """
    Teacher-forced evaluation dataset for BT programs.

    Sequence = user-prompt + assistant-header + program + "\\n\\nThe answer is: <".
    At the final token position, we score logits over {1,2,3} for accuracy/CE.
    """

    ANSWER_PROMPT = "\n\nThe answer is: <"

    def __init__(self, items: List[dict], tokenizer, max_seq_len: int = 10240):
        self.items = items
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        it = self.items[i]
        # Build prompt + assistant-header by tokenising with a dummy assistant
        # content (matches SingleQueryDataset trick).
        header_msgs = [
            Message(role="user",      content=it["prompt"], masked=True,  eot=True),
            Message(role="assistant", content="X",          masked=False, eot=True),
        ]
        header_tokens, header_mask = self.tokenizer.tokenize_messages(header_msgs)
        first_unmasked = next(
            (j for j, m in enumerate(header_mask) if not m), len(header_tokens)
        )
        # Keep the 4-token assistant header only.
        prefix = list(header_tokens[: first_unmasked + 4])
        body_ids = self.tokenizer.encode(
            it["program"] + self.ANSWER_PROMPT, add_bos=False, add_eos=False,
        )
        tokens = prefix + body_ids
        if len(tokens) > self.max_seq_len:
            # Left-truncate (keep tail intact) to preserve the answer prompt.
            tokens = tokens[-self.max_seq_len:]
        return {
            "input_ids": torch.tensor(tokens, dtype=torch.long),
            "gt_choice": torch.tensor(it["gt"], dtype=torch.long),
            "task_id":   torch.tensor(
                {"flight": 0, "hotel": 1, "webshop": 2}.get(it["task"], -1),
                dtype=torch.long,
            ),
        }


def bt_eval_collate_fn(
    batch: List[Dict[str, torch.Tensor]],
    pad_id: int,
) -> Dict[str, torch.Tensor]:
    max_len = max(b["input_ids"].size(0) for b in batch)
    # Left-pad so the last position is always the "<" token (next-token = choice).
    input_ids = torch.stack([
        F.pad(b["input_ids"], (max_len - b["input_ids"].size(0), 0), value=pad_id)
        for b in batch
    ])
    gt = torch.stack([b["gt_choice"] for b in batch])
    task = torch.stack([b["task_id"] for b in batch])
    return {"input_ids": input_ids, "gt_choice": gt, "task_id": task}


# ---------------------------------------------------------------------------
# DPO helpers
# ---------------------------------------------------------------------------

def compute_sequence_logps(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Sum log-probabilities over response (non-masked) tokens.

    Args:
        logits:  [batch, seq_len, vocab_size]
        labels:  [batch, seq_len]  (-100 for prompt / padding tokens)

    Returns:
        logps:   [batch]  (scalar per sequence)
    """
    # Shift so logits[t] predicts labels[t+1]
    shift_logits = logits[:, :-1, :].contiguous()   # [B, T-1, V]
    shift_labels = labels[:, 1:].contiguous()        # [B, T-1]

    log_probs = F.log_softmax(shift_logits, dim=-1)  # [B, T-1, V]

    # Gather the log-prob of the actual label token
    labels_clamped = shift_labels.clamp(min=0)       # avoid -100 indexing
    token_logps = log_probs.gather(
        2, labels_clamped.unsqueeze(2)
    ).squeeze(2)                                      # [B, T-1]

    # Zero out masked (prompt / padding) positions
    mask = (shift_labels != ignore_index).float()
    token_logps = token_logps * mask

    return token_logps.sum(dim=-1)                   # [B]


# ---------------------------------------------------------------------------
# Recipe
# ---------------------------------------------------------------------------

class LoRADPORecipeSingleDevice(FTRecipeInterface):
    """
    LoRA fine-tuning with Direct Preference Optimization (DPO).

    Parallel to ``custom_lora_answer_only.py`` but replaces the SFT loss with the
    DPO objective:

        L_DPO = -E[ log σ( β*(log π_θ(y_w|x) - log π_ref(y_w|x))
                           - β*(log π_θ(y_l|x) - log π_ref(y_l|x)) ) ]

    where y_w is the chosen (positive) response and y_l is the rejected
    (negative) response.

    **Data format** (see ``DPOPreferencePairDataset``):
        ``{dpo_data_dir}/{train,val,test}/{prompt_id}_positive.json``
        ``{dpo_neg_dir}/{train,val,test}/{prompt_id}_negative.json``

    **Additional config keys** (on top of the normal LoRA config):
        ``dpo_data_dir``    – root directory containing train/val/test split subdirs with positive files
        ``dpo_beta``        – KL-penalty coefficient β (default 0.1)
        ``reference_free``  – if True, skip the reference model (β acts on raw
                               log-probs rather than KL-regularised rewards)

    Args:
        cfg (DictConfig): OmegaConf object parsed from yaml file
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.filename = FILENAME
        self._device = utils.get_device(device=cfg.device)
        self._dtype = training.get_dtype(cfg.dtype, device=self._device)

        if self._dtype == torch.float16:
            raise ValueError(
                "fp16 is not supported in this recipe. Please use fp32 or bf16."
            )

        # Logging
        self._output_dir = cfg.output_dir
        self._log_every_n_steps = cfg.get("log_every_n_steps", 1)
        self._log_peak_memory_stats = cfg.get("log_peak_memory_stats", False)
        if self._log_peak_memory_stats and self._device.type == "cpu":
            log.info("log_peak_memory_stats set to False (CPU training).")
            self._log_peak_memory_stats = False

        # Training state
        self.seed = training.set_seed(seed=cfg.seed)
        self.epochs_run = 0
        self.total_epochs = cfg.epochs
        self.max_steps_per_epoch = cfg.max_steps_per_epoch
        self.global_step = 0
        self._resume_from_checkpoint = cfg.resume_from_checkpoint
        self._save_adapter_weights_only = cfg.get("save_adapter_weights_only", False)
        self._gradient_accumulation_steps = cfg.gradient_accumulation_steps
        self._clip_grad_norm = cfg.get("clip_grad_norm", None)

        # SFT mode: plain next-token-prediction on the positive response only,
        # optionally followed by a second phase for the remaining epochs.
        # Early stopping: stop training if validation CE hasn't improved for
        # this many consecutive eval epochs. Set to None or 0 to disable.
        # Default disabled — opt-in via the config field when desired.
        self._early_stopping_patience = cfg.get("early_stopping_patience", 0)
        self._sft_mode = cfg.get("sft_mode", False)
        # How many leading epochs use the SFT objective (remainder use phase 2).
        # Defaults to all epochs (pure SFT) when not specified.
        self._sft_epochs = cfg.get("sft_epochs", cfg.epochs)
        # Phase 2 objective: "sft_answer" (default) or "grpo"
        # "sft_answer": generate response, find answer position, CE loss on
        #   the answer token against the ground-truth integer.
        self._phase2 = cfg.get("phase2", "sft_answer")
        # Phase-1 trajectory mode: which thinking data to use for SFT.
        #   "scratchpad"         – natural-language scratchpad only
        #   "program"            – pyro program source only
        #   "scratchpad_program" – scratchpad followed by program
        #   null / false         – no trajectory data (fall back to DPO positives)
        # When set, phase 1 trains on sft_{mode}_{train,val,test}.json with
        # plain CE; phase 2 trains on pytorch_mcmc_dataset_train.json with
        # sft_answer loss. Validation uses both pyro and prob-reasoning val sets.
        _traj_mode_raw = cfg.get("sft_traj_mode", cfg.get("use_sft_traj", True))
        # Backward compat: True -> "scratchpad", False/None -> disabled
        if _traj_mode_raw is True:
            self._sft_traj_mode = "scratchpad"
        elif _traj_mode_raw in (False, None):
            self._sft_traj_mode = None
        else:
            self._sft_traj_mode = str(_traj_mode_raw)
        self._use_sft_traj = self._sft_traj_mode is not None
        # BT-program SFT mode: fine-tune a program-pretrained checkpoint on
        # Bayesian Teaching programs.  Training set = 200 flight-task examples;
        # eval on the remaining BT examples.  Loss is CE on the program tokens
        # only (assistant turn has no final-answer line).
        # bt_program_mode toggles BT-reasoning SFT.  The reasoning source is
        # controlled by bt_reasoning ("program" | "scratchpad").
        self._bt_program_mode = bool(cfg.get("bt_program_mode", False)) or \
                                bool(cfg.get("bt_scratchpad_mode", False))
        self._bt_reasoning    = str(cfg.get(
            "bt_reasoning",
            "scratchpad" if cfg.get("bt_scratchpad_mode", False) else "program",
        ))
        if self._bt_reasoning not in ("program", "scratchpad"):
            raise ValueError(
                f"bt_reasoning must be 'program' or 'scratchpad', got {self._bt_reasoning!r}"
            )
        self._bt_data_path       = cfg.get("bt_data_path",        DEFAULT_BT_DATA_PATH)
        self._bt_programs_dir    = cfg.get("bt_programs_dir",     DEFAULT_BT_PROGRAMS_DIR)
        self._bt_scratchpads_dir = cfg.get("bt_scratchpads_dir",  DEFAULT_BT_SCRATCHPADS_DIR)
        self._bt_n_train      = int(cfg.get("bt_n_train", 200))
        self._bt_train_task   = cfg.get("bt_train_task", "flight")
        self._bt_eval_cap     = int(cfg.get("bt_eval_cap", VAL_SUBSET_SIZE))
        self._bt_include_answer = bool(cfg.get("bt_include_answer", False))
        if self._bt_program_mode:
            # Route phase-1 through the trajectory loss path; force mean-only CE.
            self._sft_mode = True
            self._use_sft_traj = True
            self._sft_traj_mode = f"bt_{self._bt_reasoning}"
        # Per-phase learning rates (only used in sft_mode).
        self._sft_lr = cfg.get("sft_lr", None)
        self._grpo_lr = cfg.get("grpo_lr", None)
        self._phase2_lr = cfg.get("phase2_lr", self._grpo_lr)

        # DPO-specific
        self._beta = cfg.get("dpo_beta", 0.1)
        self._reference_free = cfg.get("reference_free", False)

        # GRPO-specific (runs after dpo_epochs DPO epochs)
        self._dpo_epochs = cfg.get("dpo_epochs", 5)
        self._grpo_num_completions = cfg.get("grpo_num_completions", 8)
        self._grpo_beta_kl = cfg.get("grpo_beta_kl", 0.0)
        self._grpo_temperature = cfg.get("grpo_temperature", 1.0)
        self._grpo_clip_eps = cfg.get("grpo_clip_eps", 0.2)

        # Prompt-only validation generation
        self._val_max_new_tokens = cfg.get("val_max_new_tokens", 1024)

        # Evaluation mode: "distribution" (full posterior CE) or "mean_only"
        self._loss_mode = cfg.get("loss_mode", "distribution")
        if self._bt_program_mode:
            # BT programs have no per-token posterior; force mean-only CE.
            self._loss_mode = "mean_only"

        # Activation checkpointing / offloading
        self._enable_activation_checkpointing = cfg.get(
            "enable_activation_checkpointing", False
        )
        self._enable_activation_offloading = cfg.get(
            "enable_activation_offloading", False
        )
        if self._enable_activation_offloading:
            if self._device.type != "cuda":
                raise RuntimeError(
                    "enable_activation_offloading requires CUDA."
                )
            if not self._enable_activation_checkpointing:
                raise RuntimeError(
                    "enable_activation_offloading requires "
                    "enable_activation_checkpointing=True."
                )

    # ------------------------------------------------------------------
    # Checkpoint helpers (identical to custom_lora_answer_only.py)
    # ------------------------------------------------------------------

    def load_checkpoint(self, cfg_checkpointer: DictConfig) -> Dict[str, Any]:
        self._checkpointer = config.instantiate(
            cfg_checkpointer,
            should_load_recipe_state=self._resume_from_checkpoint,
        )
        checkpoint_dict = self._checkpointer.load_checkpoint()
        if self._resume_from_checkpoint:
            if training.ADAPTER_KEY not in checkpoint_dict:
                raise ValueError(
                    "Adapter weights not found. Provide a valid adapter checkpoint."
                )
            self._update_recipe_state(checkpoint_dict)
        return checkpoint_dict

    def _update_recipe_state(self, ckpt_dict: Dict[str, Any]) -> None:
        try:
            self.epochs_run = ckpt_dict[training.EPOCHS_KEY]
            if self.seed != ckpt_dict[training.SEED_KEY]:
                warn(
                    f"Seed mismatch; using checkpoint value: {ckpt_dict[training.SEED_KEY]}"
                )
                self.seed = ckpt_dict[training.SEED_KEY]
            if self.max_steps_per_epoch != ckpt_dict[training.MAX_STEPS_KEY]:
                warn(
                    f"max_steps_per_epoch mismatch; using checkpoint value: "
                    f"{ckpt_dict[training.MAX_STEPS_KEY]}"
                )
                self.max_steps_per_epoch = ckpt_dict[training.MAX_STEPS_KEY]
            if self.total_epochs != ckpt_dict[training.TOTAL_EPOCHS_KEY]:
                warn(
                    f"total_epochs mismatch; using config value: {self.total_epochs}"
                )
        except KeyError as e:
            raise KeyError(
                "Checkpoint is missing required keys for recipe state."
            ) from e

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self._metric_logger = config.instantiate(cfg.metric_logger)
        self._metric_logger.log_config(cfg)
        self._compile = cfg.compile

        if cfg.device == "npu" and cfg.compile:
            raise ValueError("NPU does not support model compilation.")

        # Compute final output directory before checkpointer is instantiated
        # so only one directory is ever created.
        lr = cfg.optimizer.get("lr", 0)
        if self._bt_program_mode:
            sft_lr = self._sft_lr if self._sft_lr is not None else lr
            tag = "btprog" if self._bt_reasoning == "program" else "btscratch"
            dir_suffix = (
                f"_lora{cfg.model.lora_rank}_{tag}_n{self._bt_n_train}"
                f"_{self._bt_train_task}_lr{sft_lr}"
            )
        elif self._sft_mode:
            sft_lr = self._sft_lr if self._sft_lr is not None else lr
            p2_lr  = self._phase2_lr if self._phase2_lr is not None else lr
            dir_suffix = (
                f"_lora{cfg.model.lora_rank}_sft{self._sft_epochs}"
                f"-{self._phase2}_sftlr{sft_lr}-p2lr{p2_lr}"
            )
        else:
            dir_suffix = f"_lora{cfg.model.lora_rank}_dpo-grpo_beta{self._beta}_lr{lr}"
        self._output_dir = os.path.join(
            os.path.dirname(self._output_dir),
            os.path.basename(self._output_dir) + dir_suffix,
        )
        os.makedirs(self._output_dir, exist_ok=True)
        OmegaConf.update(cfg, "checkpointer.output_dir", self._output_dir)

        checkpoint_dict = self.load_checkpoint(cfg_checkpointer=cfg.checkpointer)
        common_utils._use_low_cpu_ram = cfg.get("low_cpu_ram", False)

        # Policy model (trainable LoRA)
        self._model = self._setup_model(
            cfg_model=cfg.model,
            enable_activation_checkpointing=self._enable_activation_checkpointing,
            enable_activation_offloading=self._enable_activation_offloading,
            compile_model=cfg.compile,
            base_model_state_dict=checkpoint_dict[training.MODEL_KEY],
            lora_weights_state_dict=(
                checkpoint_dict[training.ADAPTER_KEY]
                if self._resume_from_checkpoint
                else None
            ),
        )

        # Reference model: frozen snapshot of the policy at initialisation.
        # Because LoRA B-matrices are zero at init, this is equivalent to the
        # base model.  Deepcopy is taken BEFORE any training steps.
        if not self._reference_free:
            self._ref_model = self._setup_ref_model(self._model)
        else:
            self._ref_model = None
            log.info("Reference-free DPO: no reference model will be used.")

        self._tokenizer = config.instantiate(cfg.tokenizer)
        log.info("Tokenizer initialized.")

        # Token IDs for numbers 0-100 (used by GRPO reward computation)
        self._number_token_ids = get_number_token_ids(self._tokenizer).to(self._device)
        # Multi-token integer support (Qwen-2/2.5 etc.). When any integer 0..100
        # is multi-token, the legacy 101-id table has duplicates; the recipe
        # passes _number_token_seqs + _digit_token_ids into the loss/eval helpers
        # which dispatch to a position-decomposed multi-token implementation.
        from probabilistic_reasoning_utils import (
            get_number_token_seqs as _get_seqs,
            get_digit_token_ids as _get_digit_ids,
        )
        self._number_token_seqs = _get_seqs(self._tokenizer)
        self._digit_token_ids   = _get_digit_ids(self._tokenizer)
        self._single_token_integers = all(len(s) == 1 for s in self._number_token_seqs)
        if not self._single_token_integers:
            multi = [i for i, s in enumerate(self._number_token_seqs) if len(s) > 1]
            log.warning(
                f"Tokenizer is multi-token for {len(multi)} of 101 integers "
                f"(first 5: {multi[:5]}, last 5: {multi[-5:]}). Using "
                "position-decomposed multi-token loss/eval path."
            )

        self._cfg_optimizer = cfg.optimizer
        self._optimizer = self._setup_optimizer(
            cfg_optimizer=cfg.optimizer,
            model=self._model,
            opt_state_dict=None,
        )

        self._ignore_index = -100

        # DPO data
        self._setup_data(cfg=cfg)

        self._steps_per_epoch = (
            len(self._dataloader_train) // self._gradient_accumulation_steps
        )
        if (
            self.max_steps_per_epoch is not None
            and self.max_steps_per_epoch < self._steps_per_epoch
        ):
            self._steps_per_epoch = self.max_steps_per_epoch
            self.global_step = self.epochs_run * self._steps_per_epoch

        self._profiler = self._setup_profiler(cfg.get(PROFILER_KEY, None))

    def _setup_model(
        self,
        cfg_model: DictConfig,
        enable_activation_checkpointing: bool,
        enable_activation_offloading: bool,
        compile_model: bool,
        base_model_state_dict: Dict[str, Any],
        lora_weights_state_dict: Optional[Dict[str, Any]] = None,
    ) -> nn.Module:
        with training.set_default_dtype(self._dtype), self._device:
            model = config.instantiate(cfg_model)

        self._lora_rank = cfg_model.lora_rank
        self._lora_alpha = cfg_model.lora_alpha
        self._lora_attn_modules = list(cfg_model.lora_attn_modules)
        self._apply_lora_to_mlp = cfg_model.apply_lora_to_mlp
        self._apply_lora_to_output = getattr(cfg_model, "apply_lora_to_output", False)
        self.adapter_params = get_adapter_params(model)
        self._is_dora = any("magnitude" in k for k in self.adapter_params.keys())
        set_trainable_params(model, self.adapter_params)

        if compile_model:
            training.compile_model(model)
        if enable_activation_checkpointing:
            training.set_activation_checkpointing(
                model, auto_wrap_policy={modules.TransformerSelfAttentionLayer}
            )

        base_missing, base_unexpected = model.load_state_dict(
            base_model_state_dict, strict=False
        )
        if self._is_dora:
            for m in model.modules():
                if hasattr(m, "initialize_dora_magnitude"):
                    m.initialize_dora_magnitude()
        if lora_weights_state_dict:
            lora_missing, lora_unexpected = model.load_state_dict(
                lora_weights_state_dict, strict=False
            )
        else:
            lora_missing, lora_unexpected = None, None

        validate_missing_and_unexpected_for_lora(
            lora_attn_modules=self._lora_attn_modules,
            apply_lora_to_mlp=self._apply_lora_to_mlp,
            apply_lora_to_output=self._apply_lora_to_output,
            base_missing=base_missing,
            base_unexpected=base_unexpected,
            lora_missing=lora_missing,
            lora_unexpected=lora_unexpected,
        )
        training.validate_expected_param_dtype(
            self.adapter_params.items(), dtype=self._dtype
        )

        self.activations_handling_ctx = training.get_act_offloading_ctx_manager(
            model, enable_activation_offloading
        )
        log.info(f"Policy model initialized with precision {self._dtype}.")
        if self._device.type != "cpu":
            training.log_memory_stats(training.get_memory_stats(device=self._device))
        return model

    def _setup_ref_model(self, policy_model: nn.Module) -> nn.Module:
        """
        Create a frozen reference model as a deep copy of the policy model
        at initialisation time.  Because LoRA B-matrices start at zero the
        frozen copy is equivalent to running with base weights only.
        """
        ref_model = deepcopy(policy_model)
        for param in ref_model.parameters():
            param.requires_grad_(False)
        ref_model.eval()
        log.info("Reference model created (frozen copy of initial policy).")
        if self._device.type != "cpu":
            training.log_memory_stats(training.get_memory_stats(device=self._device))
        return ref_model

    def _setup_optimizer(
        self,
        cfg_optimizer: DictConfig,
        model: nn.Module,
        opt_state_dict: Optional[Dict[str, Any]] = None,
    ) -> Optimizer:
        optimizer = config.instantiate(cfg_optimizer, model.parameters())
        if opt_state_dict:
            optimizer.load_state_dict(opt_state_dict)
        log.info("Optimizer initialized.")
        return optimizer

    def _setup_data(self, cfg: DictConfig) -> None:
        """Instantiate DPO preference-pair datasets and dataloaders."""
        batch_size = cfg.batch_size
        max_seq_len = cfg.get("max_seq_len", 2048)
        if self._bt_program_mode:
            # Dedicated BT-program SFT path: no DPO/prob-reasoning data needed.
            self._dataloader_sft_traj_train = None
            self._dataloader_sft_traj_val = None
            self._dataloader_sft_traj_test = None
            self._dataloader_bt_eval_val = None
            self._dataloader_bt_eval_test = None
            self._setup_bt_program_data(cfg, batch_size, max_seq_len)
            # train() uses self._dataloader_train solely for step-count sizing
            # (len(self._dataloader_train) // grad_accum). Point it at the train
            # BT loader so _steps_per_epoch is well-defined.
            self._dataloader_train = self._dataloader_sft_traj_train
            self._sampler_train    = self._sampler_sft_traj_train
            return
        data_dir = cfg.get("dpo_data_dir", "data/dpo")
        neg_dir = cfg.get("dpo_neg_dir", None)

        def _neg_split_dir(base, split):
            return os.path.join(base, split) if base is not None else None

        # Build bins indices for each split so DPO datasets can use distribution targets
        # prob_train_json: phase-2 train data (default: pyro MCMC train for sft_answer)
        prob_train_json = cfg.get("prob_train_json", "data/pyro/pytorch_mcmc_dataset_train.json")
        prob_val_json   = cfg.get("prob_val_json",   "data/probabilistic_reasoning_val.json")
        prob_test_json  = cfg.get("prob_test_json",  "data/probabilistic_reasoning_test.json")
        # Pyro MCMC val/test (validated alongside probabilistic_reasoning)
        pyro_val_json  = cfg.get("pyro_val_json",  "data/pyro/pytorch_mcmc_dataset_val.json")
        pyro_test_json = cfg.get("pyro_test_json", "data/pyro/pytorch_mcmc_dataset_test.json")
        # Phase-1 trajectory train data path (derived from sft_traj_mode)
        _default_traj_paths = {
            "scratchpad":         "data/pyro/sft_scratchpad_train.json",
            "program":            "data/pyro/sft_program_train.json",
            "scratchpad_program": "data/pyro/sft_scratchpad_program_train.json",
        }
        _default_traj = _default_traj_paths.get(self._sft_traj_mode, "")
        sft_traj_train_json = cfg.get("sft_traj_train_json", _default_traj)
        bins_index_train = _build_bins_index(prob_train_json)
        bins_index_val   = _build_bins_index(prob_val_json)
        bins_index_test  = _build_bins_index(prob_test_json)
        log.info(
            f"Bins index: {len(bins_index_train)} train / {len(bins_index_val)} val / "
            f"{len(bins_index_test)} test scenarios"
        )

        ds_train = DPOPreferencePairDataset(
            os.path.join(data_dir, "train"), self._tokenizer, max_seq_len=max_seq_len,
            seed=self.seed, neg_dir=_neg_split_dir(neg_dir, "train"),
            bins_index=bins_index_train,
        )
        ds_val = _get_val_subset(
            DPOPreferencePairDataset(
                os.path.join(data_dir, "val"), self._tokenizer, max_seq_len=max_seq_len,
                seed=self.seed, neg_dir=_neg_split_dir(neg_dir, "val"),
                bins_index=bins_index_val,
            )
        )
        ds_test = _get_val_subset(
            DPOPreferencePairDataset(
                os.path.join(data_dir, "test"), self._tokenizer, max_seq_len=max_seq_len,
                seed=self.seed, neg_dir=_neg_split_dir(neg_dir, "test"),
                bins_index=bins_index_test,
            )
        )
        log.info(f'DATA LENGTH: {len(ds_train)}, {len(ds_val)}, {len(ds_test)}')

        collate = partial(
            dpo_collate_fn,
            pad_id=self._tokenizer.pad_id,
            ignore_index=self._ignore_index,
        )

        def _make_loader(ds, shuffle):
            sampler = DistributedSampler(
                ds, num_replicas=1, rank=0, shuffle=shuffle, seed=0
            )
            loader = DataLoader(
                dataset=ds,
                sampler=sampler,
                batch_size=batch_size,
                drop_last=True,
                collate_fn=collate,
            )
            return sampler, loader

        self._sampler_train, self._dataloader_train = _make_loader(
            ds_train, shuffle=cfg.shuffle
        )
        self._sampler_val, self._dataloader_val = _make_loader(ds_val, shuffle=False)
        self._sampler_test, self._dataloader_test = _make_loader(ds_test, shuffle=False)

        log.info(
            f"DPO dataset: {len(ds_train)} train / {len(ds_val)} val / "
            f"{len(ds_test)} test triplets."
        )
        log.info(
            f"Batches: {len(self._dataloader_train)} train / "
            f"{len(self._dataloader_val)} val / "
            f"{len(self._dataloader_test)} test."
        )

        # Phase-2 training: prompt-only (SingleQueryDataset).
        # When use_sft_traj is True (trajectory → answer-SFT), override the
        # instruction so the model is asked to think before answering, matching
        # what it learned in phase 1.
        freeform_instruction = cfg.get("freeform_instruction", None)
        if freeform_instruction is None and self._use_sft_traj:
            freeform_instruction = (
                "Answer the query or queries in the scenario, and at the end of your answer, "
                "return only integers in angle-bracket format, each integer corresponding "
                "to a query answer, separated by commas. For example: <w>, <x>, <y>, <z>. "
                "Use 0-100 scale. For a query on individual rank, a higher number means a higher ranking "
                "(e.g. 100 means the individual ranks highest in that criterion; 1 is lowest). "
                "For a query on which of the two teams wins, a smaller number means the first "
                "team more likely wins. "
                "IMPORTANT: end your answer with these integers in angle-bracket format (there may be only one query, "
                "in which case a single integer, e.g. <x>)."
            )
            log.info(f"Phase-2 instruction override (thinking): {freeform_instruction[:80]}...")
        train_sq_collate = partial(single_query_collate_fn, pad_id=self._tokenizer.pad_id)
        ds_prob_train = SingleQueryDataset(
            prob_train_json, self._tokenizer, max_seq_len=max_seq_len,
            instruction_override=freeform_instruction,
        )
        self._sampler_prob_train = DistributedSampler(
            ds_prob_train, num_replicas=1, rank=0, shuffle=cfg.shuffle, seed=0
        )
        self._dataloader_prob_train = DataLoader(
            dataset=ds_prob_train,
            sampler=self._sampler_prob_train,
            batch_size=batch_size,
            drop_last=True,
            collate_fn=train_sq_collate,
        )

        # Val/test evaluation: full prompt+response with bins (ProbabilisticReasoningDataset)
        def _make_prob_eval_loader(json_path):
            ds = _get_val_subset(
                ProbabilisticReasoningDataset(json_path, self._tokenizer, max_seq_length=max_seq_len,
                                             instruction_override=freeform_instruction)
            )
            sampler = DistributedSampler(ds, num_replicas=1, rank=0, shuffle=False, seed=0)
            loader = DataLoader(
                dataset=ds,
                sampler=sampler,
                batch_size=batch_size,
                drop_last=False,
                collate_fn=probabilistic_reasoning_collate_fn,
            )
            return loader

        self._dataloader_prob_val  = _make_prob_eval_loader(prob_val_json)
        self._dataloader_prob_test = _make_prob_eval_loader(prob_test_json)
        log.info(
            f"Prob-reasoning: {len(self._dataloader_prob_train.dataset)} train / "
            f"{len(self._dataloader_prob_val.dataset)} val / "
            f"{len(self._dataloader_prob_test.dataset)} test examples."
        )

        # Pyro MCMC val/test — evaluated alongside prob_val/test in both phases
        self._dataloader_pyro_val  = _make_prob_eval_loader(pyro_val_json) \
            if os.path.exists(pyro_val_json) else None
        self._dataloader_pyro_test = _make_prob_eval_loader(pyro_test_json) \
            if os.path.exists(pyro_test_json) else None
        if self._dataloader_pyro_val is not None:
            log.info(
                f"Pyro MCMC: {len(self._dataloader_pyro_val.dataset)} val / "
                f"{len(self._dataloader_pyro_test.dataset)} test examples."
            )
        else:
            log.warning(f"Pyro MCMC val not found at '{pyro_val_json}'; skipping.")

        # Phase-1 SFT trajectory loaders: train from the train split only,
        # validate teacher-forced on the matching val/test splits.  The
        # heldout motifs (P=1,C=1,R=0 and P=2,C=1,R=0) are never seen in train.
        self._dataloader_sft_traj_train = None
        self._dataloader_sft_traj_val = None
        self._dataloader_sft_traj_test = None
        if self._use_sft_traj:
            sft_traj_val_json  = sft_traj_train_json.replace("_train.json", "_val.json")
            sft_traj_test_json = sft_traj_train_json.replace("_train.json", "_test.json")
            if os.path.exists(sft_traj_train_json):
                traj_collate = partial(
                    sft_traj_collate_fn,
                    pad_id=self._tokenizer.pad_id,
                    ignore_index=self._ignore_index,
                )

                # Train loader
                ds_sft_traj = SFTTrajDataset(
                    sft_traj_train_json, self._tokenizer, max_seq_len=max_seq_len
                )
                traj_sampler = DistributedSampler(
                    ds_sft_traj, num_replicas=1, rank=0, shuffle=cfg.shuffle, seed=0
                )
                self._dataloader_sft_traj_train = DataLoader(
                    dataset=ds_sft_traj,
                    sampler=traj_sampler,
                    batch_size=batch_size,
                    drop_last=True,
                    collate_fn=traj_collate,
                )
                self._sampler_sft_traj_train = traj_sampler
                log.info(
                    f"SFT trajectory [{self._sft_traj_mode}] train: {len(ds_sft_traj)} examples "
                    f"({len(self._dataloader_sft_traj_train)} batches)."
                )

                # Val loader (teacher-forced evaluation)
                if os.path.exists(sft_traj_val_json):
                    ds_val = SFTTrajDataset(
                        sft_traj_val_json, self._tokenizer, max_seq_len=max_seq_len
                    )
                    val_sampler = DistributedSampler(
                        ds_val, num_replicas=1, rank=0, shuffle=False, seed=0
                    )
                    self._dataloader_sft_traj_val = DataLoader(
                        dataset=ds_val, sampler=val_sampler, batch_size=batch_size,
                        drop_last=False, collate_fn=traj_collate,
                    )
                    log.info(f"SFT trajectory [{self._sft_traj_mode}] val: {len(ds_val)} examples.")

                # Test loader (teacher-forced evaluation)
                if os.path.exists(sft_traj_test_json):
                    ds_test = SFTTrajDataset(
                        sft_traj_test_json, self._tokenizer, max_seq_len=max_seq_len
                    )
                    test_sampler = DistributedSampler(
                        ds_test, num_replicas=1, rank=0, shuffle=False, seed=0
                    )
                    self._dataloader_sft_traj_test = DataLoader(
                        dataset=ds_test, sampler=test_sampler, batch_size=batch_size,
                        drop_last=False, collate_fn=traj_collate,
                    )
                    log.info(f"SFT trajectory [{self._sft_traj_mode}] test: {len(ds_test)} examples.")
            else:
                log.warning(
                    f"sft_traj_train_json not found: '{sft_traj_train_json}'; "
                    "falling back to DPO data for phase-1."
                )
                self._use_sft_traj = False

    def _setup_bt_program_data(
        self, cfg: DictConfig, batch_size: int, max_seq_len: int,
    ) -> None:
        """Build train + eval loaders for the BT-program SFT mode."""
        reasoning_dir = (
            self._bt_programs_dir if self._bt_reasoning == "program"
            else self._bt_scratchpads_dir
        )
        # Training pool: first N examples of the chosen task with reasoning files.
        all_train_task = _bt_build_items(
            self._bt_data_path, reasoning_dir, reasoning=self._bt_reasoning,
            tasks=[self._bt_train_task],
        )
        all_train_task.sort(key=lambda it: (it["task"], it["idx"]))
        train_items = all_train_task[: self._bt_n_train]
        train_ids = {(it["task"], it["idx"]) for it in train_items}

        # Eval pool: everything else with reasoning files (all tasks).
        all_items = _bt_build_items(
            self._bt_data_path, reasoning_dir, reasoning=self._bt_reasoning,
        )
        eval_items = [
            it for it in all_items if (it["task"], it["idx"]) not in train_ids
        ]
        # Split eval into val (same task, held-out) and test (other tasks).
        val_items  = [it for it in eval_items if it["task"] == self._bt_train_task]
        test_items = [it for it in eval_items if it["task"] != self._bt_train_task]
        # Cap val size for speed.
        if self._bt_eval_cap and len(val_items) > self._bt_eval_cap:
            rng = np.random.RandomState(VAL_SUBSET_SEED)
            keep = sorted(rng.choice(len(val_items), self._bt_eval_cap, replace=False).tolist())
            val_items = [val_items[i] for i in keep]

        log.info(
            f"BT-{self._bt_reasoning} SFT: train={len(train_items)} ({self._bt_train_task}) / "
            f"val={len(val_items)} ({self._bt_train_task} held-out) / "
            f"test={len(test_items)} (other tasks)."
        )

        # --- Train loader (teacher-forced on program, loss on program only) ---
        ds_train = BTProgramSFTDataset(
            train_items, self._tokenizer, max_seq_len=max_seq_len,
            include_answer=self._bt_include_answer,
        )
        traj_collate = partial(
            sft_traj_collate_fn,
            pad_id=self._tokenizer.pad_id,
            ignore_index=self._ignore_index,
        )
        traj_sampler = DistributedSampler(
            ds_train, num_replicas=1, rank=0, shuffle=cfg.shuffle, seed=0
        )
        self._dataloader_sft_traj_train = DataLoader(
            dataset=ds_train, sampler=traj_sampler, batch_size=batch_size,
            drop_last=False, collate_fn=traj_collate,
        )
        self._sampler_sft_traj_train = traj_sampler

        # --- Answer-accuracy eval loaders (batch_size=1 for variable-length seqs) ---
        bt_eval_collate = partial(bt_eval_collate_fn, pad_id=self._tokenizer.pad_id)
        def _make_bt_eval_loader(items):
            ds = BTProgramEvalDataset(items, self._tokenizer, max_seq_len=max_seq_len)
            sampler = DistributedSampler(ds, num_replicas=1, rank=0, shuffle=False, seed=0)
            return DataLoader(
                dataset=ds, sampler=sampler, batch_size=1,
                drop_last=False, collate_fn=bt_eval_collate,
            )
        self._dataloader_bt_eval_train = _make_bt_eval_loader(train_items)
        self._dataloader_bt_eval_val   = _make_bt_eval_loader(val_items)
        self._dataloader_bt_eval_test  = _make_bt_eval_loader(test_items)
        # Also set traj_{val,test} to the train loader so the pre-train eval path
        # (which expects sft_traj loaders) has something to call; we override the
        # metric call in train() when BT mode is active.
        self._dataloader_sft_traj_val  = self._dataloader_bt_eval_val
        self._dataloader_sft_traj_test = self._dataloader_bt_eval_test

    def _setup_profiler(
        self, cfg_profiler: Optional[DictConfig] = None
    ) -> Union[torch.profiler.profile, DummyProfiler]:
        if cfg_profiler is None:
            cfg_profiler = DictConfig({"enabled": False})
        if cfg_profiler.get("_component_", None) is None:
            cfg_profiler["_component_"] = "torchtune.training.setup_torch_profiler"
        profiler, profiler_cfg = config.instantiate(cfg_profiler)
        log.info(f"Profiler config: {profiler_cfg}")
        self.profiler_profile_memory = profiler_cfg.get("profile_memory", False)
        if profiler_cfg["enabled"]:
            self.profiler_wait_steps = profiler_cfg["wait_steps"]
            self.profiler_warmup_steps = profiler_cfg["warmup_steps"]
            self.profiler_active_steps = profiler_cfg["active_steps"]
        return profiler

    # ------------------------------------------------------------------
    # Core DPO loss
    # ------------------------------------------------------------------

    def _compute_logps(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through *model* and compute per-sequence log-probabilities
        over the response (unmasked label) tokens.

        Args:
            model:     The language model (policy or reference).
            input_ids: [batch, seq_len]
            labels:    [batch, seq_len]  (-100 for prompt / padding)

        Returns:
            logps: [batch]
        """
        with self.activations_handling_ctx:
            logits = model(tokens=input_ids)

        if isinstance(logits, list):
            logits = torch.cat(logits, dim=1)

        return compute_sequence_logps(logits, labels, self._ignore_index)

    def _dpo_loss_step(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute DPO loss and reward metrics for one batch.

        Args:
            batch: dict with keys
                ``input_ids_chosen``, ``labels_chosen``,
                ``input_ids_rejected``, ``labels_rejected``

        Returns:
            loss:    scalar tensor (gradients flow through the policy model)
            metrics: dict with ``reward_accuracy``, ``reward_margin``,
                     ``chosen_reward``, ``rejected_reward``
        """
        input_ids_chosen  = torch.atleast_2d(batch["input_ids_chosen"])
        labels_chosen     = torch.atleast_2d(batch["labels_chosen"])
        input_ids_rejected = torch.atleast_2d(batch["input_ids_rejected"])
        labels_rejected   = torch.atleast_2d(batch["labels_rejected"])

        # ---- Policy log-probs (batch chosen + rejected together for efficiency) ----
        # Chosen and rejected may have different sequence lengths; pad to the same
        # length before concatenating along the batch dimension.
        max_len = max(input_ids_chosen.size(1), input_ids_rejected.size(1))

        def _pad_to(t: torch.Tensor, length: int, pad_value: int) -> torch.Tensor:
            if t.size(1) == length:
                return t
            pad = torch.full(
                (t.size(0), length - t.size(1)), pad_value,
                dtype=t.dtype, device=t.device,
            )
            return torch.cat([t, pad], dim=1)

        input_ids_chosen   = _pad_to(input_ids_chosen,   max_len, 0)
        input_ids_rejected = _pad_to(input_ids_rejected, max_len, 0)
        labels_chosen      = _pad_to(labels_chosen,      max_len, self._ignore_index)
        labels_rejected    = _pad_to(labels_rejected,    max_len, self._ignore_index)

        input_ids_all = torch.cat([input_ids_chosen, input_ids_rejected], dim=0)
        labels_all = torch.cat([labels_chosen, labels_rejected], dim=0)
        policy_logps_all = self._compute_logps(self._model, input_ids_all, labels_all)
        policy_chosen_logps, policy_rejected_logps = policy_logps_all.chunk(2)

        # ---- Reference log-probs (no gradient) ----
        if self._ref_model is not None:
            with torch.no_grad():
                ref_logps_all = self._compute_logps(
                    self._ref_model, input_ids_all, labels_all
                )
            ref_chosen_logps, ref_rejected_logps = ref_logps_all.chunk(2)
        else:
            # Reference-free: treat reference as uniform (log-ratio = raw log-prob)
            ref_chosen_logps = torch.zeros_like(policy_chosen_logps)
            ref_rejected_logps = torch.zeros_like(policy_rejected_logps)

        # ---- DPO objective ----
        chosen_rewards = self._beta * (policy_chosen_logps - ref_chosen_logps)
        rejected_rewards = self._beta * (policy_rejected_logps - ref_rejected_logps)
        loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()

        # ---- Metrics (detached) ----
        with torch.no_grad():
            reward_accuracy = (chosen_rewards > rejected_rewards).float().mean().item()
            reward_margin = (chosen_rewards - rejected_rewards).mean().item()

        metrics = {
            "reward_accuracy": reward_accuracy,
            "reward_margin": reward_margin,
            "chosen_reward": chosen_rewards.mean().item(),
            "rejected_reward": rejected_rewards.mean().item(),
        }

        return loss, metrics

    # ------------------------------------------------------------------
    # Shared GRPO helpers
    # ------------------------------------------------------------------

    def _greedy_generate(
        self,
        prompt_tokens: torch.Tensor,
        max_new_tokens: int,
    ) -> torch.Tensor:
        """
        Autoregressively generate from *prompt_tokens* [P] using greedy
        (argmax) decoding.  Stops on any stop token.

        Returns the full sequence [P + response_len] as a 1-D tensor.
        No gradient is computed.
        """
        stop_tokens = set(self._tokenizer.stop_tokens)
        generated = prompt_tokens.unsqueeze(0)  # [1, P]
        with torch.no_grad():
            for _ in range(max_new_tokens):
                with self.activations_handling_ctx:
                    logits_gen = self._model(tokens=generated)
                if isinstance(logits_gen, list):
                    logits_gen = torch.cat(logits_gen, dim=1)
                next_tok = torch.argmax(logits_gen[0, -1, :]).reshape(1, 1)
                generated = torch.cat([generated, next_tok], dim=1)
                if next_tok.item() in stop_tokens:
                    break
        return generated[0]  # [P + response_len]

    def _grpo_single_position_loss(
        self,
        logits: torch.Tensor,
        logit_pos: int,
        gt_q: float,
        ref_logits: Optional[torch.Tensor] = None,
        accuracy_weight: float = 1.0,
        format_reward: float = 0.0,
    ) -> Tuple[torch.Tensor, float, float]:
        """
        Compute the GRPO clipped surrogate loss for one answer position.

        Samples G integers from the policy distribution at *logit_pos*,
        computes group-relative advantages, and returns the PPO-style loss.

        Args:
            logits:          [T, V] or [1, T, V] — policy logits (grad retained).
            logit_pos:       Index into the time dimension (= label_pos - 1).
            gt_q:            Ground-truth integer in [0, 100].
            ref_logits:      [T, V] reference logits (no grad), or None.
            accuracy_weight: Weight on accuracy reward (default 1.0).
            format_reward:   Additive scalar format reward component.

        Returns:
            (loss_q, mean_reward, adv_std)
        """
        if logits.dim() == 3:
            logits = logits[0]  # [T, V]

        G = self._grpo_num_completions

        number_logits = logits[logit_pos, self._number_token_ids]  # [101]
        if self._grpo_temperature != 1.0:
            number_logits = number_logits / self._grpo_temperature
        log_probs = F.log_softmax(number_logits, dim=0)  # [101]
        probs     = log_probs.exp()

        with torch.no_grad():
            sampled = torch.multinomial(
                probs.detach().float(), G, replacement=True
            )  # [G]

        gt_f    = torch.tensor(gt_q, dtype=torch.float, device=self._device)
        accuracy_reward = 1.0 - (sampled.float() - gt_f) ** 2 / (100.0 ** 2)  # [G]
        rewards = accuracy_weight * accuracy_reward + (1.0 - accuracy_weight) * format_reward

        r_mean = rewards.mean()
        r_std  = rewards.std() + 1e-8
        advantages = (rewards - r_mean) / r_std  # [G]

        # NOTE: old_log_probs and new_log_probs come from the same forward
        # pass, so ratio == 1.0 identically and clipping never activates.
        # The loss is equivalent to REINFORCE with group-relative baseline.
        old_log_probs = log_probs[sampled].detach()  # [G]
        new_log_probs = log_probs[sampled]            # [G]
        ratio = (new_log_probs - old_log_probs).exp()
        adv   = advantages.detach()
        clipped_ratio = ratio.clamp(
            1.0 - self._grpo_clip_eps, 1.0 + self._grpo_clip_eps
        )
        loss_q = -torch.min(ratio * adv, clipped_ratio * adv).mean()

        if ref_logits is not None:
            if ref_logits.dim() == 3:
                ref_logits = ref_logits[0]
            ref_num_logits = ref_logits[logit_pos, self._number_token_ids].detach()
            ref_log_probs  = F.log_softmax(ref_num_logits, dim=0)
            kl = (probs * (log_probs - ref_log_probs)).sum()
            loss_q = loss_q + self._grpo_beta_kl * kl

        return loss_q, rewards.mean().item(), advantages.std().item()

    # ------------------------------------------------------------------
    # GRPO loss (accuracy-based, squared-error reward on answer token)
    # ------------------------------------------------------------------

    def _grpo_loss_step(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        GRPO on teacher-forced DPO-format batches (dpo_positives).

        For each example in the batch we:
          1. Find the answer-token positions in ``labels_chosen``.
          2. Use the last n_queries positions (skipping scratchpad numbers).
          3. For each position, call _grpo_single_position_loss with a combined
             accuracy + format reward (0.8 / 0.2 split).

        Args:
            batch: dict with keys ``input_ids_chosen``, ``labels_chosen``,
                   ``ground_truth`` (integer list per example), ``format_ok``.

        Returns:
            loss:    scalar tensor (gradients flow through the policy model).
            metrics: dict with ``grpo_loss``, ``grpo_reward``, ``grpo_advantage_std``.
        """
        input_ids_chosen = torch.atleast_2d(batch["input_ids_chosen"])  # [B, T]
        labels_chosen    = torch.atleast_2d(batch["labels_chosen"])    # [B, T]
        ground_truth     = batch["ground_truth"]                        # [B, max_queries]
        format_ok_batch  = batch["format_ok"]                           # [B]

        B = input_ids_chosen.size(0)

        with self.activations_handling_ctx:
            logits = self._model(tokens=input_ids_chosen)  # [B, T, V]
        if isinstance(logits, list):
            logits = torch.cat(logits, dim=1)

        if self._ref_model is not None and self._grpo_beta_kl > 0.0:
            with torch.no_grad():
                ref_logits = self._ref_model(tokens=input_ids_chosen)
            if isinstance(ref_logits, list):
                ref_logits = torch.cat(ref_logits, dim=1)
        else:
            ref_logits = None

        answer_positions = _find_answer_positions(
            labels_chosen, self._number_token_ids, self._tokenizer
        )

        total_loss  = torch.tensor(0.0, device=self._device, dtype=self._dtype)
        n_valid     = 0
        sum_reward  = 0.0
        sum_adv_std = 0.0

        for b in range(B):
            positions = answer_positions[b]
            if not positions:
                continue

            gt_vec = ground_truth[b]
            n_queries = int((gt_vec >= 0).sum().item())
            tail_positions = positions[-n_queries:] if len(positions) >= n_queries else positions
            query_pairs = [
                (pos, gt_val.item())
                for pos, gt_val in zip(tail_positions, gt_vec)
                if pos > 0 and gt_val.item() >= 0
            ]
            if not query_pairs:
                continue

            format_reward = 1.0 if format_ok_batch[b].item() else 0.0
            loss_queries = []
            reward_sum_b = 0.0
            adv_std_sum_b = 0.0

            for pos, gt_q in query_pairs:
                loss_q, mean_r, adv_std = self._grpo_single_position_loss(
                    logits[b],
                    logit_pos=pos - 1,
                    gt_q=gt_q,
                    ref_logits=ref_logits[b] if ref_logits is not None else None,
                    accuracy_weight=0.8,
                    format_reward=format_reward,
                )
                loss_queries.append(loss_q)
                reward_sum_b  += mean_r
                adv_std_sum_b += adv_std

            n_q = len(loss_queries)
            total_loss  = total_loss + sum(loss_queries) / n_q
            n_valid    += 1
            sum_reward  += reward_sum_b  / n_q
            sum_adv_std += adv_std_sum_b / n_q

        if n_valid > 0:
            total_loss = total_loss / n_valid

        metrics = {
            "grpo_loss":          total_loss.detach().item(),
            "grpo_reward":        sum_reward  / max(n_valid, 1),
            "grpo_advantage_std": sum_adv_std / max(n_valid, 1),
        }
        return total_loss, metrics

    # ------------------------------------------------------------------
    # SFT loss (plain next-token prediction on the positive response)
    # ------------------------------------------------------------------

    def _sft_loss_step(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Supervised fine-tuning loss on the positive (chosen) trajectory.

        At non-answer token positions: standard CE.
        At answer token positions: distributional CE over 101 bins against the
        ground-truth posterior (from probabilistic_reasoning_train.json).

        Args:
            batch: dict with ``input_ids_chosen``, ``labels_chosen``,
                   ``ground_truth_bins`` [B, n_queries, 101], ``num_queries`` [B].

        Returns:
            loss:    scalar tensor (gradients flow through the policy model).
            metrics: dict with ``sft_loss``, ``sft_ce_loss``, ``sft_dist_loss``.
        """
        input_ids        = torch.atleast_2d(batch["input_ids_chosen"])   # [B, T]
        labels           = torch.atleast_2d(batch["labels_chosen"])      # [B, T]
        ground_truth_bins = batch["ground_truth_bins"]                    # [B, n_q, 101]
        num_queries      = batch["num_queries"]                           # [B]

        with self.activations_handling_ctx:
            logits = self._model(tokens=input_ids)                        # [B, T, V]
        if isinstance(logits, list):
            logits = torch.cat(logits, dim=1)

        # Shift labels left by 1 so that shifted_labels[t] is the target for logits[t]
        B = labels.size(0)
        ignore_col = torch.full((B, 1), self._ignore_index, dtype=labels.dtype, device=labels.device)
        shifted_labels = torch.cat([labels[:, 1:], ignore_col], dim=1)   # [B, T]

        # number_token_seqs/digit_token_ids enable the multi-token-aware
        # distributional loss for tokenizers like Qwen-2 (no-op for Llama-3
        # where every integer 0..100 is single-token).
        loss, loss_dict = compute_probabilistic_loss(
            logits=logits,
            labels=shifted_labels,
            ground_truth_bins=ground_truth_bins,
            number_token_ids=self._number_token_ids,
            ce_weight=1.0,
            dist_weight=1.0,
            ignore_index=self._ignore_index,
            num_queries=num_queries,
            tokenizer=self._tokenizer,
            number_token_seqs=self._number_token_seqs,
            digit_token_ids=self._digit_token_ids,
        )

        metrics = {
            "sft_loss":      loss.detach().item(),
            "sft_ce_loss":   loss_dict.get("ce_loss", 0.0),
            "sft_dist_loss": loss_dict.get("dist_loss", 0.0),
        }
        return loss, metrics

    # ------------------------------------------------------------------
    # GRPO on single-query prob data (probabilistic_reasoning_train.json)
    # ------------------------------------------------------------------

    def _grpo_prob_loss_step(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        GRPO on SingleQueryDataset batches (probabilistic_reasoning_train.json).

        For each example in the batch:
          1. Strip padding and call _greedy_generate to obtain the full
             prompt+response context (no gradient).
          2. Find the answer token position in the generated sequence.
          3. Run a gradient-retaining forward pass and call
             _grpo_single_position_loss with accuracy-only reward.

        Args:
            batch: dict with keys ``prompt_ids`` [B, P] and
                   ``ground_truth_single`` [B].

        Returns:
            loss:    scalar tensor (gradients flow through the policy model).
            metrics: dict with ``grpo_loss``, ``grpo_reward``,
                     ``grpo_advantage_std``.
        """
        prompt_ids   = torch.atleast_2d(batch["prompt_ids"])  # [B, P]
        ground_truth = batch["ground_truth_single"]            # [B]
        pad_id       = self._tokenizer.pad_id
        B = prompt_ids.size(0)

        total_loss  = torch.tensor(0.0, device=self._device, dtype=self._dtype)
        n_valid     = 0
        n_skipped   = 0
        sum_reward  = 0.0
        sum_adv_std = 0.0

        for b in range(B):
            gt = ground_truth[b].item()

            # Strip trailing padding
            tokens = prompt_ids[b]
            non_pad = (tokens != pad_id).nonzero(as_tuple=True)[0]
            if len(non_pad) == 0:
                n_skipped += 1
                continue
            tokens = tokens[: non_pad[-1] + 1]   # [P]
            prompt_len = tokens.size(0)

            # Greedy generation (no gradient) to build context
            generated_1d = self._greedy_generate(tokens, self._val_max_new_tokens)
            generated = generated_1d.unsqueeze(0)  # [1, T]

            # Labels: -100 for prompt, token ids for response
            labels_gen = torch.full(
                (generated_1d.size(0),), -100, dtype=torch.long, device=self._device
            )
            labels_gen[prompt_len:] = generated_1d[prompt_len:]

            answer_positions = _find_answer_positions(
                labels_gen.unsqueeze(0), self._number_token_ids, self._tokenizer,
            )
            positions = answer_positions[0]
            if not positions:
                n_skipped += 1
                continue
            pos = positions[-1]  # last number token = final answer

            # Policy forward pass (gradient retained) on full generated sequence
            with self.activations_handling_ctx:
                logits = self._model(tokens=generated)  # [1, T, V]
            if isinstance(logits, list):
                logits = torch.cat(logits, dim=1)

            if self._ref_model is not None and self._grpo_beta_kl > 0.0:
                with torch.no_grad():
                    ref_logits = self._ref_model(tokens=generated)
                if isinstance(ref_logits, list):
                    ref_logits = torch.cat(ref_logits, dim=1)
            else:
                ref_logits = None

            loss_q, mean_r, adv_std = self._grpo_single_position_loss(
                logits[0],
                logit_pos=pos - 1,
                gt_q=gt,
                ref_logits=ref_logits[0] if ref_logits is not None else None,
                accuracy_weight=1.0,  # accuracy-only, no format reward
                format_reward=0.0,
            )

            total_loss  = total_loss + loss_q
            n_valid    += 1
            sum_reward  += mean_r
            sum_adv_std += adv_std
            del logits

        if n_valid > 0:
            total_loss = total_loss / n_valid
        else:
            log.warning(
                f"GRPO: all {B} examples in batch had no parseable answer; "
                "zero-gradient step."
            )

        if n_skipped > 0:
            log.warning(f"GRPO: skipped {n_skipped}/{B} examples (no answer token found).")

        metrics = {
            "grpo_loss":          total_loss.detach().item(),
            "grpo_reward":        sum_reward  / max(n_valid, 1),
            "grpo_advantage_std": sum_adv_std / max(n_valid, 1),
            "grpo_skip_rate":     n_skipped / B,
        }
        return total_loss, metrics

    # ------------------------------------------------------------------
    # SFT loss on full thinking-trajectory output (phase-1)
    # ------------------------------------------------------------------

    def _sft_traj_loss_step(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Loss on the full assistant output (scratchpad + answer).

        loss_mode="mean_only" (default behaviour):
            Standard CE on all unmasked response tokens.  Answer tokens are
            the rounded posterior mean, so this trains on the argmax.

        loss_mode="distribution":
            CE on non-answer response tokens (structural tokens) PLUS
            distributional 101-bin CE at each answer-token position against
            the ground-truth posterior from ``ground_truth_bins``.

        Args:
            batch: dict with keys ``input_ids`` [B, T], ``labels`` [B, T],
                   ``ground_truth_bins`` [B, max_queries, 101],
                   ``num_queries`` [B].

        Returns:
            loss:    scalar tensor (gradients flow through the policy model).
            metrics: dict with ``sft_traj_loss`` and optionally ``sft_traj_dist_loss``.
        """
        input_ids         = torch.atleast_2d(batch["input_ids"])       # [B, T]
        labels            = torch.atleast_2d(batch["labels"])          # [B, T]
        ground_truth_bins = batch.get("ground_truth_bins", None)       # [B, Q, 101]
        num_queries       = batch.get("num_queries", None)             # [B]

        with self.activations_handling_ctx:
            logits = self._model(tokens=input_ids)                     # [B, T, V]
        if isinstance(logits, list):
            logits = torch.cat(logits, dim=1)

        # Shift: logits[t] predicts labels[t+1]
        shift_logits = logits[:, :-1, :].contiguous()                  # [B, T-1, V]
        shift_labels = labels[:, 1:].contiguous()                      # [B, T-1]

        if self._loss_mode == "distribution" and ground_truth_bins is not None:
            # Find answer positions in shifted labels
            answer_positions = _find_answer_positions(
                shift_labels, self._number_token_ids, self._tokenizer
            )

            # CE on non-answer tokens (mask out answer positions)
            labels_no_answers = shift_labels.clone()
            for i, positions in enumerate(answer_positions):
                for pos in positions:
                    labels_no_answers[i, pos] = self._ignore_index

            ce_loss = F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                labels_no_answers.reshape(-1),
                ignore_index=self._ignore_index,
            )

            # Distributional CE at answer positions
            B = input_ids.size(0)
            dist_loss = torch.tensor(0.0, device=self._device, dtype=self._dtype)
            n_answers = 0
            for i in range(B):
                positions = answer_positions[i]
                nq = int(num_queries[i].item()) if num_queries is not None else len(positions)
                # Use the last nq positions (the final answer line)
                tail = positions[-nq:] if len(positions) >= nq else positions
                for q, pos in enumerate(tail):
                    if q >= ground_truth_bins.size(1):
                        break
                    gt_dist = ground_truth_bins[i, q].to(
                        dtype=self._dtype, device=self._device
                    )
                    log_pred = F.log_softmax(
                        shift_logits[i, pos], dim=0
                    )[self._number_token_ids]  # [101]
                    dist_loss = dist_loss - (gt_dist * log_pred).sum()
                    n_answers += 1

            if n_answers > 0:
                dist_loss = dist_loss / n_answers

            loss = ce_loss + dist_loss
            metrics = {
                "sft_traj_loss": loss.detach().item(),
                "sft_traj_ce_loss": ce_loss.detach().item(),
                "sft_traj_dist_loss": dist_loss.detach().item(),
            }
        else:
            # mean_only: standard CE on all tokens (answer tokens are the mean)
            loss = F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
                ignore_index=self._ignore_index,
            )
            metrics = {"sft_traj_loss": loss.detach().item()}

        return loss, metrics

    # ------------------------------------------------------------------
    # Phase-2 loss: answer-only SFT on trajectory data
    # ------------------------------------------------------------------

    def _sft_traj_answer_only_loss_step(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Phase-2 loss on the trajectory train set.

        The model sees the full assistant turn (reasoning + "The answer is: ")
        teacher-forced, and loss is computed ONLY at the final answer-token
        positions (the last ``num_queries`` number tokens).  No loss on
        reasoning/format tokens.

        loss_mode="distribution":
            Distributional 101-bin CE at each answer position against the
            ground-truth posterior in ``ground_truth_bins[i, q]``.

        loss_mode="mean_only":
            Standard CE targeting the single GT mean token (= the token that
            appears in the trajectory's answer line).

        Args:
            batch: SFTTrajDataset batch — ``input_ids`` [B, T],
                   ``labels`` [B, T], ``ground_truth_bins`` [B, Q, 101],
                   ``num_queries`` [B].

        Returns:
            loss: scalar tensor (gradients flow through the model).
            metrics: dict with ``phase2_loss``.
        """
        input_ids         = torch.atleast_2d(batch["input_ids"])        # [B, T]
        labels            = torch.atleast_2d(batch["labels"])           # [B, T]
        ground_truth_bins = batch["ground_truth_bins"]                  # [B, Q, 101]
        num_queries       = batch["num_queries"]                        # [B]

        with self.activations_handling_ctx:
            logits = self._model(tokens=input_ids)
        if isinstance(logits, list):
            logits = torch.cat(logits, dim=1)

        # Shift so logits[t] predicts labels[t+1]
        B = labels.size(0)
        ignore_col = torch.full(
            (B, 1), -100, dtype=labels.dtype, device=labels.device
        )
        labels_shifted = torch.cat([labels[:, 1:], ignore_col], dim=1)

        answer_positions = _find_answer_positions(
            labels_shifted, self._number_token_ids, self._tokenizer
        )

        total_loss = torch.tensor(0.0, device=self._device, dtype=self._dtype)
        n_answers = 0
        for i in range(B):
            positions = answer_positions[i]
            nq = int(num_queries[i].item())
            if nq <= 0 or not positions:
                continue
            tail = positions[-nq:] if len(positions) >= nq else positions
            for q, pos in enumerate(tail):
                if q >= ground_truth_bins.size(1):
                    break
                if self._loss_mode == "distribution":
                    gt_dist = ground_truth_bins[i, q].to(
                        dtype=self._dtype, device=self._device
                    )  # [101]
                    log_pred = F.log_softmax(
                        logits[i, pos], dim=0
                    )[self._number_token_ids]  # [101]
                    total_loss = total_loss - (gt_dist * log_pred).sum()
                else:
                    # mean_only: CE targeting the GT token at this position
                    gt_token_id = labels_shifted[i, pos]
                    total_loss = total_loss + F.cross_entropy(
                        logits[i, pos].unsqueeze(0),
                        gt_token_id.unsqueeze(0),
                    )
                n_answers += 1

        if n_answers > 0:
            total_loss = total_loss / n_answers
        else:
            log.warning("phase2: no answer positions in batch; zero-gradient step.")

        metrics = {"phase2_loss": total_loss.detach().item()}
        return total_loss, metrics

    # ------------------------------------------------------------------
    # Checkpoint (identical structure to custom_lora_answer_only.py)
    # ------------------------------------------------------------------

    def save_checkpoint(self, epoch: int) -> None:
        ckpt_dict: Dict[str, Any] = {}
        intermediate_checkpoint = epoch + 1 < self.total_epochs
        if intermediate_checkpoint:
            ckpt_dict.update(
                {
                    training.OPT_KEY: self._optimizer.state_dict(),
                    training.SEED_KEY: self.seed,
                    training.EPOCHS_KEY: self.epochs_run,
                    training.TOTAL_EPOCHS_KEY: self.total_epochs,
                    training.MAX_STEPS_KEY: self.max_steps_per_epoch,
                }
            )

        adapter_state_dict = get_adapter_state_dict(self._model.state_dict())
        ckpt_dict.update({training.ADAPTER_KEY: adapter_state_dict})

        if not self._save_adapter_weights_only:
            state_dict = {k: v.cpu() for k, v in self._model.state_dict().items()}
            merged_state_dict = get_merged_lora_ckpt(
                state_dict, rank=self._lora_rank, alpha=self._lora_alpha
            )
            ckpt_dict.update({training.MODEL_KEY: merged_state_dict})

        adapter_config = {
            "r": self._lora_rank,
            "lora_alpha": self._lora_alpha,
            "target_modules": get_lora_module_names(
                self._lora_attn_modules,
                self._apply_lora_to_mlp,
                self._apply_lora_to_output,
            ),
            "peft_type": "LORA",
        }
        ckpt_dict.update({training.ADAPTER_CONFIG: adapter_config})

        # Always overwrite epoch_0 so there is always one best checkpoint
        self._checkpointer.save_checkpoint(
            ckpt_dict,
            epoch=0,
            intermediate_checkpoint=intermediate_checkpoint,
            adapter_only=self._save_adapter_weights_only,
        )
        ckpt_dir = os.path.join(self._output_dir, "epoch_0")
        with open(os.path.join(ckpt_dir, "checkpoint_epoch.txt"), "w") as f:
            f.write(f"{epoch}\n")
        log.info(f"Checkpoint saved (actual epoch: {epoch}, written to epoch_0 folder)")

        if not self._bt_program_mode:
            # run_checkpoint_analysis depends on DPO/prob_reasoning loaders
            # that aren't instantiated in BT mode.
            self.run_checkpoint_analysis(epoch=epoch)

    # ------------------------------------------------------------------
    # Checkpoint analysis helpers
    # ------------------------------------------------------------------

    def _collect_val_preds_for_analysis(self, n_responses: int = 5):
        """
        Forward pass over the val set.  For each example, find the answer
        position (last number token in the response), read off the model's
        argmax prediction over 0-100, and compare to ground_truth.

        Returns:
            pred_vals      – list of predicted integers
            gt_vals        – list of ground-truth integers
            sample_texts   – list of (prompt_text, gt_int, pred_int) for the
                             first n_responses valid examples
        """
        pred_vals: List[int] = []
        gt_vals:   List[int] = []
        sample_texts: List[tuple] = []

        self._model.eval()
        with torch.no_grad():
            for batch in self._dataloader_val:
                utils.batch_to_device(batch, self._device)
                input_ids = torch.atleast_2d(batch["input_ids_chosen"])  # [B, T]
                labels    = torch.atleast_2d(batch["labels_chosen"])    # [B, T]
                gt        = batch["ground_truth"]                        # [B]

                with self.activations_handling_ctx:
                    logits = self._model(tokens=input_ids)
                if isinstance(logits, list):
                    logits = torch.cat(logits, dim=1)

                answer_positions = _find_answer_positions(
                    labels, self._number_token_ids, self._tokenizer
                )

                for b in range(input_ids.size(0)):
                    positions = answer_positions[b]
                    gt_vec = gt[b]   # [max_queries]
                    n_queries = int((gt_vec >= 0).sum().item())
                    tail_positions = positions[-n_queries:] if len(positions) >= n_queries else positions
                    for pos, gt_val in zip(tail_positions, gt_vec):
                        gt_b = gt_val.item()
                        if pos == 0 or gt_b < 0:
                            continue
                        number_logits = logits[b, pos - 1, self._number_token_ids]
                        pred_int = int(torch.argmax(number_logits).item())
                        pred_vals.append(pred_int)
                        gt_vals.append(gt_b)

                        if len(sample_texts) < n_responses:
                            resp_start = (labels[b] != -100).nonzero(as_tuple=True)[0]
                            prompt_ids = input_ids[b][:resp_start[0]].cpu().tolist() if len(resp_start) else []
                            try:
                                prompt_text = self._tokenizer.decode(
                                    prompt_ids, skip_special_tokens=True
                                )
                            except Exception:
                                prompt_text = "<decode error>"
                            sample_texts.append((prompt_text, gt_b, pred_int))

                del logits

        self._model.train()
        return pred_vals, gt_vals, sample_texts

    def _generate_val_samples(
        self,
        n_samples: int = 5,
        max_new_tokens: int = 1024,
    ) -> List[Tuple[str, str, List[int], List[int]]]:
        """
        Generate up to *n_samples* responses from the val set (dpo_positives/val).
        The model receives only the prompt; no reference response is shown.

        Returns list of (prompt_text, generated_text, gt_list, pred_list).
        """
        results: List[Tuple[str, str, List[int], List[int]]] = []

        self._model.eval()
        with torch.no_grad():
            for batch in self._dataloader_val:
                if len(results) >= n_samples:
                    break
                utils.batch_to_device(batch, self._device)

                for b in range(batch["input_ids_chosen"].size(0)):
                    if len(results) >= n_samples:
                        break

                    labels = batch["labels_chosen"][b]
                    gt_vec = batch["ground_truth"][b]
                    valid_gts = gt_vec[gt_vec >= 0]
                    if valid_gts.numel() == 0:
                        continue
                    gt_list = valid_gts.tolist()

                    resp_positions = (labels != -100).nonzero(as_tuple=True)[0]
                    if len(resp_positions) == 0:
                        continue
                    prompt_tokens = batch["input_ids_chosen"][b][:resp_positions[0]]
                    if prompt_tokens.size(0) == 0:
                        continue

                    try:
                        prompt_text = self._tokenizer.decode(
                            prompt_tokens.cpu().tolist(), skip_special_tokens=True
                        )
                    except Exception:
                        prompt_text = "<decode error>"

                    generated_1d = self._greedy_generate(prompt_tokens, max_new_tokens)
                    response_tokens = generated_1d[prompt_tokens.size(0):].cpu().tolist()
                    try:
                        generated_text = self._tokenizer.decode(
                            response_tokens, skip_special_tokens=True
                        )
                    except Exception:
                        generated_text = "<decode error>"

                    pred_list = _parse_answer(generated_text)

                    results.append((prompt_text, generated_text, gt_list, pred_list))

        self._model.train()
        return results

    def _compute_prompt_only_mae(
        self,
        dataloader,
        max_new_tokens: int = 1024,
    ) -> float:
        """
        Compute MAE over *dataloader* by generating from the prompt only.
        The model must produce "The answer is: <a>, <b>, <c>, <d>" itself.
        Examples where no parseable answer is found are skipped.
        """
        abs_errors: List[float] = []

        self._model.eval()
        with torch.no_grad():
            for batch in dataloader:
                utils.batch_to_device(batch, self._device)

                for b in range(batch["input_ids_chosen"].size(0)):
                    labels = batch["labels_chosen"][b]
                    gt_vec = batch["ground_truth"][b]
                    valid_gts = gt_vec[gt_vec >= 0]
                    if valid_gts.numel() == 0:
                        continue
                    gt_list = valid_gts.tolist()

                    resp_positions = (labels != -100).nonzero(as_tuple=True)[0]
                    if len(resp_positions) == 0:
                        continue
                    prompt_tokens = batch["input_ids_chosen"][b][:resp_positions[0]]
                    if prompt_tokens.size(0) == 0:
                        continue

                    generated_1d = self._greedy_generate(prompt_tokens, max_new_tokens)
                    response_tokens = generated_1d[prompt_tokens.size(0):].cpu().tolist()
                    try:
                        generated_text = self._tokenizer.decode(
                            response_tokens, skip_special_tokens=True
                        )
                    except Exception:
                        continue

                    pred_list = _parse_answer(generated_text)

                    for p, g in zip(pred_list, gt_list):
                        abs_errors.append(abs(p - g))

        self._model.train()
        if not abs_errors:
            log.warning("_compute_prompt_only_mae: no examples produced a parseable answer.")
            return float('nan')
        return float(np.mean(abs_errors))

    def _evaluate_multitoken_metrics(self, logits, ground_truth_bins, num_queries=None, labels=None):
        """Multi-token-aware eval (Qwen-2 etc.). Delegates to the shared helper
        in ``probabilistic_reasoning_utils.evaluate_multitoken_metrics``."""
        from probabilistic_reasoning_utils import evaluate_multitoken_metrics
        return evaluate_multitoken_metrics(
            logits=logits,
            ground_truth_bins=ground_truth_bins,
            number_token_seqs=self._number_token_seqs,
            digit_token_ids=self._digit_token_ids,
            num_queries=num_queries,
            labels=labels,
        )

    def evaluate_probabilistic_predictions(
        self,
        logits: torch.Tensor,
        ground_truth_bins: torch.Tensor,
        num_queries: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate predicted distributions against ground truth bins.
        Mirrors the method in custom_lora_answer_only.py.

        Returns dict with kl_divergence, cross_entropy, mean_abs_error,
        pred_means, gt_means.
        """
        # Multi-token tokenizers (Qwen-2 etc.) need the span-based eval.
        if not getattr(self, "_single_token_integers", True):
            return self._evaluate_multitoken_metrics(
                logits, ground_truth_bins, num_queries, labels
            )

        batch_size = logits.size(0)
        all_metrics = []

        for i in range(batch_size):
            gt_bins = ground_truth_bins[i]
            n_queries = num_queries[i].item() if num_queries is not None else gt_bins.size(0)
            example_labels = labels[i] if labels is not None else None

            pred_dists = []
            gt_dists = []
            for q in range(n_queries):
                pred_probs = extract_number_probabilities(
                    logits[i],
                    self._number_token_ids,
                    method='answer_position',
                    labels=example_labels,
                    query_idx=q,
                )
                pred_dists.append(pred_probs.detach().float().cpu().numpy())
                gt_dists.append(gt_bins[q].cpu().numpy())

            if pred_dists:
                all_metrics.append(evaluate_predictions(pred_dists, gt_dists))

        if not all_metrics:
            return {
                'kl_divergence': 0.0, 'cross_entropy': 0.0,
                'mean_abs_error': 0.0, 'pred_means': [], 'gt_means': [],
            }

        avg_metrics: Dict[str, Any] = {}
        for key in all_metrics[0]:
            if key in ('pred_means', 'gt_means'):
                avg_metrics[key] = [v for m in all_metrics for v in m[key]]
            else:
                avg_metrics[key] = sum(m[key] for m in all_metrics) / len(all_metrics)
        return avg_metrics

    def _compute_prob_metrics(
        self,
        dataloader,
    ) -> Tuple[float, float]:
        """
        Compute MAE and cross-entropy over a ProbabilisticReasoningDataset
        dataloader using a teacher-forced forward pass (same as custom_lora_answer_only.py).

        Returns (mae, cross_entropy).
        """
        all_mae: List[float] = []
        all_ce:  List[float] = []

        self._model.eval()
        with torch.no_grad():
            for batch in dataloader:
                utils.batch_to_device(batch, self._device)
                tokens           = batch["tokens"]
                labels           = batch["labels"]
                ground_truth_bins = batch["ground_truth_bins"]
                num_queries      = batch.get("num_queries", None)

                with self.activations_handling_ctx:
                    logits = self._model(tokens=tokens)
                if isinstance(logits, list):
                    logits = torch.cat(logits, dim=1)

                # Shift labels by 1 to align with logit positions
                B = labels.size(0)
                ignore_col = torch.full(
                    (B, 1), -100, dtype=labels.dtype, device=labels.device
                )
                labels_shifted = torch.cat([labels[:, 1:], ignore_col], dim=1)

                prob_metrics = self.evaluate_probabilistic_predictions(
                    logits, ground_truth_bins, num_queries, labels=labels_shifted
                )
                all_mae.append(prob_metrics['mean_abs_error'])
                all_ce.append(prob_metrics['cross_entropy'])
                del logits

        self._model.train()
        mae = float(np.mean(all_mae)) if all_mae else 0.0
        ce  = float(np.mean(all_ce))  if all_ce  else 0.0
        return mae, ce

    def _compute_traj_metrics(
        self,
        dataloader,
        label: str = "val",
    ) -> Tuple[float, float]:
        """
        Teacher-forced evaluation on an SFTTrajDataset dataloader.

        For each batch:
          1. Forward pass on the full (prompt + trajectory + answer) sequence.
          2. Shift labels by 1 to align logits with labels.
          3. Find the per-query answer positions via _find_answer_positions on
             the shifted labels (the last ``num_queries`` number-token positions
             correspond to the final-answer line).
          4. At each answer position, extract the 101-way softmax over number
             tokens and compute MAE (|pred_mean - gt_mean|) and distributional
             CE against ``ground_truth_bins[i, q]``.

        Returns (mae, cross_entropy) averaged across all query positions.
        """
        all_mae: List[float] = []
        all_ce:  List[float] = []
        eps = 1e-10

        self._model.eval()
        with torch.no_grad():
            for batch in dataloader:
                utils.batch_to_device(batch, self._device)
                input_ids         = torch.atleast_2d(batch["input_ids"])        # [B, T]
                labels            = torch.atleast_2d(batch["labels"])           # [B, T]
                ground_truth_bins = batch["ground_truth_bins"]                  # [B, Q, 101]
                num_queries       = batch.get("num_queries", None)              # [B]

                with self.activations_handling_ctx:
                    logits = self._model(tokens=input_ids)
                if isinstance(logits, list):
                    logits = torch.cat(logits, dim=1)

                # Shift labels so labels_shifted[t] == original_labels[t+1]
                B = labels.size(0)
                ignore_col = torch.full(
                    (B, 1), -100, dtype=labels.dtype, device=labels.device
                )
                labels_shifted = torch.cat([labels[:, 1:], ignore_col], dim=1)

                answer_positions = _find_answer_positions(
                    labels_shifted, self._number_token_ids, self._tokenizer
                )

                for i in range(B):
                    positions = answer_positions[i]
                    nq = int(num_queries[i].item()) if num_queries is not None else len(positions)
                    if nq <= 0 or not positions:
                        continue
                    # Take the last nq positions (the final-answer line)
                    tail = positions[-nq:] if len(positions) >= nq else positions
                    for q, pos in enumerate(tail):
                        if q >= ground_truth_bins.size(1):
                            break
                        number_logits = logits[i, pos, self._number_token_ids]
                        pred_probs = F.softmax(number_logits, dim=0).float().cpu().numpy()
                        gt_np = ground_truth_bins[i, q].float().cpu().numpy()

                        pred_mean = float(np.sum(np.arange(101) * pred_probs))
                        gt_mean = float(np.sum(np.arange(101) * gt_np))
                        all_mae.append(abs(pred_mean - gt_mean))

                        log_pred = np.log(pred_probs + eps)
                        all_ce.append(float(-np.sum(gt_np * log_pred)))

                del logits

        self._model.train()
        mae = float(np.mean(all_mae)) if all_mae else 0.0
        ce  = float(np.mean(all_ce))  if all_ce  else 0.0
        log.info(f"[TRAJ-EVAL {label}] scored {len(all_mae)} query positions — MAE={mae:.3f} CE={ce:.4f}")
        return mae, ce

    def _compute_bt_answer_metrics(
        self, dataloader, label: str = "bt_val",
    ) -> Tuple[float, float]:
        """
        Teacher-forced BT-program eval.

        For each example, the sequence is prompt + program + "\\n\\nThe answer
        is: <".  We score logits at the final position over the three choice
        tokens {1, 2, 3}, compare argmax to ``gt_choice`` for accuracy, and
        report mean CE = -log softmax(choice_logits)[gt-1].
        """
        if dataloader is None or len(dataloader.dataset) == 0:
            return 0.0, 0.0

        # Resolve choice token IDs using the "<1> <2> <3>" context trick.
        ctx_ids = self._tokenizer.encode("<1> <2> <3>", add_bos=False, add_eos=False)
        id_to_str = {}
        for tid in ctx_ids:
            try:
                id_to_str[tid] = self._tokenizer.decode([tid]).strip()
            except Exception:
                pass
        choice_ids = []
        for n in (1, 2, 3):
            found = None
            for tid, decoded in id_to_str.items():
                if decoded == str(n):
                    found = tid
                    break
            if found is None:
                enc = self._tokenizer.encode(" " + str(n), add_bos=False, add_eos=False)
                found = enc[0] if enc else 0
            choice_ids.append(found)
        choice_tensor = torch.tensor(choice_ids, dtype=torch.long, device=self._device)

        correct = 0
        ces: List[float] = []
        n = 0
        self._model.eval()
        with torch.no_grad():
            for batch in dataloader:
                utils.batch_to_device(batch, self._device)
                input_ids = batch["input_ids"]   # [B, T]
                gt        = batch["gt_choice"]   # [B]
                with self.activations_handling_ctx:
                    logits = self._model(tokens=input_ids)
                if isinstance(logits, list):
                    logits = torch.cat(logits, dim=1)
                last = logits[:, -1, :].float()              # [B, V]
                choice_logits = last[:, choice_tensor]       # [B, 3]
                log_probs = F.log_softmax(choice_logits, dim=-1)
                pred = choice_logits.argmax(dim=-1) + 1      # 1..3
                correct += int((pred == gt).sum().item())
                for b in range(input_ids.size(0)):
                    gt_b = int(gt[b].item())
                    ces.append(float(-log_probs[b, gt_b - 1].item()))
                n += input_ids.size(0)
                del logits
        self._model.train()
        acc = correct / n if n else 0.0
        ce  = float(np.mean(ces)) if ces else 0.0
        log.info(
            f"[BT-EVAL {label}] n={n}  acc={acc:.3f}  CE={ce:.4f}"
        )
        return acc, ce

    def _compute_dpo_val_metrics(self) -> Tuple[float, float]:
        """
        Teacher-forced MAE and CE over the dpo_positives val split.

        Ground truth is the integer answer stored in each batch's ``ground_truth``
        field.  CE = -log P(gt_token) over the full vocabulary.  MAE = absolute
        error between the predicted expected value (from number-token softmax) and
        the integer ground truth.

        Returns (mae, cross_entropy).
        """
        all_mae: List[float] = []
        all_ce:  List[float] = []

        self._model.eval()
        with torch.no_grad():
            for batch in self._dataloader_val:
                utils.batch_to_device(batch, self._device)
                input_ids = torch.atleast_2d(batch["input_ids_chosen"])  # [B, T]
                labels    = torch.atleast_2d(batch["labels_chosen"])     # [B, T]
                gt        = batch["ground_truth"]                         # [B, max_queries]

                with self.activations_handling_ctx:
                    logits = self._model(tokens=input_ids)
                if isinstance(logits, list):
                    logits = torch.cat(logits, dim=1)

                answer_positions = _find_answer_positions(
                    labels, self._number_token_ids, self._tokenizer
                )

                for b in range(input_ids.size(0)):
                    positions = answer_positions[b]
                    gt_vec    = gt[b]
                    n_queries = int((gt_vec >= 0).sum().item())
                    tail_pos  = (
                        positions[-n_queries:]
                        if len(positions) >= n_queries
                        else positions
                    )

                    for pos, gt_val in zip(tail_pos, gt_vec):
                        gt_b = gt_val.item()
                        if pos == 0 or gt_b < 0:
                            continue
                        # labels are 1-shifted relative to logits in this loader
                        logit_pos = pos - 1

                        # CE: -log P(gt_token) over full vocabulary
                        gt_token_id = self._number_token_ids[gt_b]
                        log_probs = torch.nn.functional.log_softmax(
                            logits[b, logit_pos], dim=0
                        )
                        all_ce.append(-log_probs[gt_token_id].item())

                        # MAE: |E[pred] - gt_b| using number-token subspace softmax
                        number_logits = logits[b, logit_pos, self._number_token_ids]
                        pred_probs = (
                            torch.softmax(number_logits, dim=0).float().cpu().numpy()
                        )
                        pred_mean = float(
                            sum(idx * pred_probs[idx] for idx in range(101))
                        )
                        all_mae.append(abs(pred_mean - gt_b))

                del logits

        self._model.train()
        mae = float(np.mean(all_mae)) if all_mae else 0.0
        ce  = float(np.mean(all_ce))  if all_ce  else 0.0
        return mae, ce

    def _generate_traj_batch_samples(
        self,
        dataloader,
        n_batches: int = 5,
        max_new_tokens: int = 1024,
    ) -> List[Tuple[str, str, List[int], List[int]]]:
        """
        Generate responses from the first *n_batches* batches of *dataloader*
        (which loads from dpo_positives).  Used for qualitative logging only;
        no metrics are computed on these examples.

        Returns list of (prompt_text, generated_text, gt_list, pred_list).
        """
        results: List[Tuple[str, str, List[int], List[int]]] = []

        self._model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= n_batches:
                    break
                utils.batch_to_device(batch, self._device)

                for b in range(batch["input_ids_chosen"].size(0)):
                    labels = batch["labels_chosen"][b]
                    gt_vec = batch["ground_truth"][b]
                    valid_gts = gt_vec[gt_vec >= 0]
                    if valid_gts.numel() == 0:
                        continue
                    gt_list = valid_gts.tolist()

                    resp_positions = (labels != -100).nonzero(as_tuple=True)[0]
                    if len(resp_positions) == 0:
                        continue
                    prompt_tokens = batch["input_ids_chosen"][b][:resp_positions[0]]
                    if prompt_tokens.size(0) == 0:
                        continue

                    try:
                        prompt_text = self._tokenizer.decode(
                            prompt_tokens.cpu().tolist(), skip_special_tokens=True
                        )
                    except Exception:
                        prompt_text = "<decode error>"

                    generated_1d = self._greedy_generate(prompt_tokens, max_new_tokens)
                    response_tokens = generated_1d[prompt_tokens.size(0):].cpu().tolist()
                    try:
                        generated_text = self._tokenizer.decode(
                            response_tokens, skip_special_tokens=True
                        )
                    except Exception:
                        generated_text = "<decode error>"

                    pred_list = _parse_answer(generated_text)

                    results.append((prompt_text, generated_text, gt_list, pred_list))

        self._model.train()
        return results

    def run_checkpoint_analysis(self, epoch: int) -> float:
        """
        Collect val predictions, save scatter / direction PNGs, write 5 sample
        responses to a text file, and return the mean absolute error.
        """
        ckpt_dir = os.path.join(self._output_dir, "epoch_0")
        pred_dir = os.path.join(ckpt_dir, "predictions")
        os.makedirs(pred_dir, exist_ok=True)

        pred_vals, gt_vals, sample_texts = self._collect_val_preds_for_analysis(
            n_responses=5
        )
        if not pred_vals:
            log.warning("run_checkpoint_analysis: no valid val predictions found.")
            return 0.0

        pred_means = [float(p) for p in pred_vals]
        gt_means   = [float(g) for g in gt_vals]

        # --- Scatter + direction PNGs ---
        try:
            plot_prediction_vs_ground_truth(
                pred_means, gt_means,
                title=f"DPO-GRPO val predictions (epoch {epoch})",
                output_path=os.path.join(pred_dir, f"scatter_epoch_{epoch}.png"),
            )
            plot_direction_analysis(
                pred_means, gt_means,
                title=f"DPO-GRPO val directions (epoch {epoch})",
                output_path=os.path.join(pred_dir, f"direction_epoch_{epoch}.png"),
            )
            log.info(f"Prediction PNGs saved to: {pred_dir}")
        except Exception as e:
            log.warning(f"run_checkpoint_analysis: plot error: {e}")

        # --- Learning-vs-centering text summary ---
        try:
            summary = analyze_learning_vs_centering(pred_means, gt_means)
            log.info(f"\nCheckpoint analysis (epoch {epoch}):\n{summary}")
        except Exception as e:
            log.warning(f"run_checkpoint_analysis: analysis error: {e}")

        # --- 5 sample responses ---
        responses_path = os.path.join(pred_dir, f"sample_responses_epoch_{epoch}.txt")
        with open(responses_path, "w") as f:
            f.write(f"=== 5 val sample responses — epoch {epoch} ===\n\n")
            for i, (prompt, gt_int, pred_int) in enumerate(sample_texts):
                f.write(f"--- Example {i + 1} ---\n")
                f.write(f"PROMPT (last 500 chars):\n...{prompt[-500:]}\n\n")
                f.write(f"GROUND TRUTH:      {gt_int}\n")
                f.write(f"MODEL PREDICTION:  {pred_int}\n")
                f.write(f"ABSOLUTE ERROR:    {abs(pred_int - gt_int)}\n\n")
        log.info(f"Sample responses written to: {responses_path}")

        # --- Autoregressive generations (prompt-only → free-form response) ---
        try:
            gen_samples = self._generate_val_samples(n_samples=5, max_new_tokens=1024)
            gen_path = os.path.join(pred_dir, f"generated_responses_epoch_{epoch}.txt")
            with open(gen_path, "w") as f:
                f.write(f"=== Autoregressive val generations — epoch {epoch} ===\n")
                f.write("(Model sees scenario + queries only; no reference response)\n\n")
                for i, (prompt, gen_text, gt_list, pred_list) in enumerate(gen_samples):
                    f.write(f"--- Example {i + 1} ---\n")
                    f.write(f"PROMPT (last 500 chars):\n...{prompt[-500:]}\n\n")
                    f.write(f"GENERATED RESPONSE:\n{gen_text}\n\n")
                    f.write(f"GROUND TRUTH:     {gt_list}\n")
                    f.write(f"EXTRACTED ANSWER: {pred_list}\n")
                    if pred_list and gt_list:
                        per_query_errs = [
                            abs(p - g)
                            for p, g in zip(pred_list, gt_list)
                        ]
                        avg_err = sum(per_query_errs) / len(per_query_errs)
                        f.write(f"PER-QUERY ERRORS: {per_query_errs}\n")
                        f.write(f"AVG ABSOLUTE ERROR: {avg_err:.2f}\n\n")
                    else:
                        f.write(f"AVG ABSOLUTE ERROR: N/A (no answers extracted)\n\n")
            log.info(f"Autoregressive generations written to: {gen_path}")
        except Exception as e:
            log.warning(f"run_checkpoint_analysis: generation error: {e}")

        # --- 5 autoregressive generations from probabilistic_reasoning_val ---
        try:
            prob_gen_samples = []
            self._model.eval()
            with torch.no_grad():
                for batch in self._dataloader_prob_val:
                    if len(prob_gen_samples) >= 5:
                        break
                    utils.batch_to_device(batch, self._device)
                    tokens  = batch["tokens"]            # [B, T]
                    masks   = batch["mask"]              # [B, T] True=prompt, False=response
                    gt_bins = batch["ground_truth_bins"] # [B, num_queries, 101]
                    num_queries = batch.get("num_queries", None)
                    for b in range(tokens.size(0)):
                        if len(prob_gen_samples) >= 5:
                            break
                        # First False in mask = start of assistant response
                        resp_start = (~masks[b]).nonzero(as_tuple=True)[0]
                        if len(resp_start) == 0:
                            continue
                        prompt_tokens = tokens[b, :resp_start[0]]
                        if prompt_tokens.size(0) == 0:
                            continue
                        try:
                            prompt_text = self._tokenizer.decode(
                                prompt_tokens.cpu().tolist(), skip_special_tokens=True
                            )
                        except Exception:
                            prompt_text = "<decode error>"
                        generated_1d = self._greedy_generate(prompt_tokens, self._val_max_new_tokens)
                        response_tokens = generated_1d[prompt_tokens.size(0):].cpu().tolist()
                        try:
                            generated_text = self._tokenizer.decode(
                                response_tokens, skip_special_tokens=True
                            )
                        except Exception:
                            generated_text = "<decode error>"
                        pred_list = _parse_answer(generated_text)
                        bins = gt_bins[b].cpu().numpy()  # [num_queries, 101]
                        nq = int(num_queries[b].item()) if num_queries is not None else bins.shape[0]
                        gt_list = [
                            float(sum(i * bins[q][i] for i in range(101)))
                            for q in range(nq)
                        ]
                        prob_gen_samples.append((prompt_text, generated_text, gt_list, pred_list))
            self._model.train()
            prob_gen_path = os.path.join(pred_dir, f"prob_val_generations_epoch_{epoch}.txt")
            with open(prob_gen_path, "w") as f:
                f.write(f"=== prob_val autoregressive generations — epoch {epoch} ===\n")
                f.write("(Model sees scenario + query only; no reference response)\n\n")
                for i, (prompt, gen_text, gt_list, pred_list) in enumerate(prob_gen_samples):
                    f.write(f"--- Example {i + 1} ---\n")
                    f.write(f"PROMPT (last 500 chars):\n...{prompt[-500:]}\n\n")
                    f.write(f"GENERATED RESPONSE:\n{gen_text}\n\n")
                    f.write(f"GROUND TRUTH (expected value): {[round(g, 1) for g in gt_list]}\n")
                    f.write(f"EXTRACTED ANSWER: {pred_list}\n")
                    if pred_list and gt_list:
                        errs = [abs(p - g) for p, g in zip(pred_list, gt_list)]
                        f.write(f"PER-QUERY ERRORS: {[round(e, 1) for e in errs]}\n")
                        f.write(f"AVG ABSOLUTE ERROR: {sum(errs) / len(errs):.2f}\n\n")
                    else:
                        f.write("AVG ABSOLUTE ERROR: N/A\n\n")
            log.info(f"prob_val generations written to: {prob_gen_path}")
        except Exception as e:
            log.warning(f"run_checkpoint_analysis: prob_val generation error: {e}")

        # --- MAE ---
        mae = float(np.mean([abs(p - g) for p, g in zip(pred_vals, gt_vals)]))
        log.info(f"Checkpoint val MAE (epoch {epoch}): {mae:.3f}")
        return mae

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self) -> None:
        if self._compile:
            log.info(
                "torch.compile is enabled; expect a slow first iteration."
            )

        saved_losses: List[float] = []
        train_reward_accs: List[float] = []
        val_maes:  List[float] = []   # teacher-forced trajectory val MAE
        test_maes: List[float] = []
        val_ces:   List[float] = []   # teacher-forced trajectory val CE
        test_ces:  List[float] = []
        train_maes: List[float] = []  # BT-mode: accuracy on the 200 training examples
        train_ces:  List[float] = []

        # Phase-2 metric buffers
        grpo_losses: List[float] = []
        grpo_rewards: List[float] = []
        sft_answer_losses: List[float] = []

        # Early-stopping state: bail out of the per-epoch loop once val CE has
        # stalled for `_early_stopping_patience` consecutive eval epochs.
        # Compatible across modes since it only watches `mean_val_ce`.
        best_val_ce = float("inf")
        epochs_since_improvement = 0

        # Persistent iterators: survive epoch boundaries so long dataloaders
        # are consumed continuously rather than restarted each epoch.
        MAX_BATCHES_PER_EPOCH = 200
        _iter_epoch = 0
        # Phase-1: use trajectory data when available, else DPO positives
        if self._use_sft_traj and self._dataloader_sft_traj_train is not None:
            self._sampler_sft_traj_train.set_epoch(_iter_epoch)
            train_iter = iter(self._dataloader_sft_traj_train)
        else:
            self._sampler_train.set_epoch(_iter_epoch)
            train_iter = iter(self._dataloader_train)

        # Separate iterator for the GRPO phase (prob_train data)
        _prob_iter_epoch = 0
        prob_train_iter: Optional[object] = None   # initialised on first GRPO epoch

        self._model.train()

        if self._sft_mode and self._sft_lr is not None:
            for pg in self._optimizer.param_groups:
                pg["lr"] = self._sft_lr
            log.info(f"Learning rate set to {self._sft_lr} for SFT phase.")

        # ------------------------------------------------------------------
        # Pre-training validation run (before any training, epoch = -1)
        # ------------------------------------------------------------------
        log.info("Running pre-training validation (teacher-forced trajectory)...")
        self._model.eval()
        torch.cuda.empty_cache()
        mean_train_mae = mean_train_ce = None
        if self._bt_program_mode:
            mean_train_mae, mean_train_ce = self._compute_bt_answer_metrics(
                self._dataloader_bt_eval_train, label="bt_train_pretrain")
            mean_val_mae,  mean_val_ce  = self._compute_bt_answer_metrics(
                self._dataloader_bt_eval_val,  label="bt_val_pretrain")
            mean_test_mae, mean_test_ce = self._compute_bt_answer_metrics(
                self._dataloader_bt_eval_test, label="bt_test_pretrain")
        else:
            if self._dataloader_sft_traj_train is not None:
                mean_train_mae, mean_train_ce = self._compute_traj_metrics(
                    self._dataloader_sft_traj_train, label="traj_train_pretrain")
            mean_val_mae,  mean_val_ce  = self._compute_traj_metrics(self._dataloader_sft_traj_val,  label="traj_val_pretrain")
            mean_test_mae, mean_test_ce = self._compute_traj_metrics(self._dataloader_sft_traj_test, label="traj_test_pretrain")
        val_maes.append(mean_val_mae)
        test_maes.append(mean_test_mae)
        val_ces.append(mean_val_ce)
        test_ces.append(mean_test_ce)
        np.savetxt(self.filename + "_maes_val.csv",  np.array(val_maes))
        np.savetxt(self.filename + "_maes_test.csv", np.array(test_maes))
        np.savetxt(self.filename + "_ces_val.csv",   np.array(val_ces))
        np.savetxt(self.filename + "_ces_test.csv",  np.array(test_ces))
        if mean_train_mae is not None:
            train_maes.append(mean_train_mae)
            train_ces.append(mean_train_ce)
            np.savetxt(self.filename + "_maes_train.csv", np.array(train_maes))
            np.savetxt(self.filename + "_ces_train.csv",  np.array(train_ces))
        torch.cuda.empty_cache()
        log.info("")
        log.info("----------- PRE-TRAINING VALIDATION -----------")
        if mean_train_mae is not None:
            tag = "BT-TRAIN" if self._bt_program_mode else "TRAJ-TRAIN"
            log.info(f"{tag} MAE={mean_train_mae:.2f}  CE={mean_train_ce:.4f}")
        log.info(f"TRAJ-VAL  MAE={mean_val_mae:.2f}  CE={mean_val_ce:.4f}")
        log.info(f"TRAJ-TEST MAE={mean_test_mae:.2f}  CE={mean_test_ce:.4f}")
        log.info("----------- PRE-TRAINING VALIDATION -----------")
        log.info("")
        self._model.train()

        for curr_epoch in range(self.total_epochs):
            if self._sft_mode:
                use_grpo = curr_epoch >= self._sft_epochs
                if use_grpo:
                    if self._phase2 == "sft_answer":
                        phase   = "SFT_ANSWER"
                        # Phase 2 uses the trajectory train loader (same as
                        # phase 1) but with answer-only loss on final <N> positions.
                        loss_fn = self._sft_traj_answer_only_loss_step
                    else:
                        phase   = "GRPO"
                        loss_fn = self._grpo_prob_loss_step
                    if curr_epoch == self._sft_epochs:
                        log.info(
                            f"SFT phase complete after {self._sft_epochs} epoch(s). "
                            f"Switching to {phase}."
                        )
                        # Reset optimizer to clear phase-1 momentum/second-moment state.
                        del self._optimizer
                        self._optimizer = self._setup_optimizer(
                            cfg_optimizer=self._cfg_optimizer,
                            model=self._model,
                            opt_state_dict=None,
                        )
                        if self._phase2_lr is not None:
                            for pg in self._optimizer.param_groups:
                                pg["lr"] = self._phase2_lr
                            log.info(f"Learning rate set to {self._phase2_lr} for {phase} phase.")
                        else:
                            log.info(f"Optimizer reset for {phase} phase (lr unchanged).")
                    # GRPO still uses the prob_train loader; sft_answer reuses train_iter (trajectory).
                    if self._phase2 != "sft_answer" and prob_train_iter is None:
                        self._sampler_prob_train.set_epoch(_prob_iter_epoch)
                        prob_train_iter = iter(self._dataloader_prob_train)
                else:
                    phase   = "SFT"
                    loss_fn = (
                        self._sft_traj_loss_step
                        if self._use_sft_traj and self._dataloader_sft_traj_train is not None
                        else self._sft_loss_step
                    )
            else:
                use_grpo = curr_epoch >= self._dpo_epochs
                phase    = "GRPO" if use_grpo else "DPO"
                loss_fn  = self._grpo_prob_loss_step if use_grpo else self._dpo_loss_step

                if use_grpo and curr_epoch == self._dpo_epochs:
                    log.info(
                        f"DPO phase complete after {self._dpo_epochs} epoch(s). "
                        "Switching to GRPO on probabilistic_reasoning_train.json."
                    )
                    if prob_train_iter is None:
                        self._sampler_prob_train.set_epoch(_prob_iter_epoch)
                        prob_train_iter = iter(self._dataloader_prob_train)

            self._optimizer.zero_grad()

            if not NOTRAIN:
                n_batches = MAX_BATCHES_PER_EPOCH

                # Phase 2 sft_answer reuses the trajectory iter (same data
                # as phase 1).  GRPO uses the separate prob_train_iter.
                use_prob_iter = use_grpo and self._phase2 != "sft_answer"

                for idx in range(n_batches):
                    # Advance the correct iterator, refreshing when exhausted
                    if use_prob_iter:
                        try:
                            batch = next(prob_train_iter)
                        except StopIteration:
                            _prob_iter_epoch += 1
                            self._sampler_prob_train.set_epoch(_prob_iter_epoch)
                            prob_train_iter = iter(self._dataloader_prob_train)
                            batch = next(prob_train_iter)
                    else:
                        try:
                            batch = next(train_iter)
                        except StopIteration:
                            _iter_epoch += 1
                            if (
                                self._use_sft_traj
                                and self._dataloader_sft_traj_train is not None
                            ):
                                self._sampler_sft_traj_train.set_epoch(_iter_epoch)
                                train_iter = iter(self._dataloader_sft_traj_train)
                            else:
                                self._sampler_train.set_epoch(_iter_epoch)
                                train_iter = iter(self._dataloader_train)
                            batch = next(train_iter)

                    if (
                        self.max_steps_per_epoch is not None
                        and (idx // self._gradient_accumulation_steps)
                        == self.max_steps_per_epoch
                    ):
                        break

                    utils.batch_to_device(batch, self._device)

                    loss, metrics = loss_fn(batch)

                    # Gradient accumulation (skip if no valid examples produced a grad_fn)
                    scaled_loss = loss / self._gradient_accumulation_steps
                    if scaled_loss.requires_grad:
                        scaled_loss.backward()

                    if (idx + 1) % self._gradient_accumulation_steps == 0:
                        if (
                            self._clip_grad_norm is not None
                            and self._clip_grad_norm != "inf"
                        ):
                            torch.nn.utils.clip_grad_norm_(
                                self._model.parameters(),
                                float(self._clip_grad_norm),
                            )
                        self._optimizer.step()
                        self._optimizer.zero_grad()

                    peak_memory = torch.cuda.max_memory_allocated()

                    if self._sft_mode and not use_grpo:
                        saved_losses.append(loss.detach().cpu().item())
                        log.info(
                            f"[{phase} Epoch {curr_epoch} Step {idx}] "
                            f"loss={loss.item():.4f} | "
                            f"peak_mem={peak_memory / (1024**2):.0f}MB"
                        )
                        if np.sum(np.isnan(saved_losses)) == 0:
                            np.savetxt(
                                self.filename + "_sft_loss.csv", np.array(saved_losses)
                            )
                    elif phase == "SFT_ANSWER":
                        sft_answer_losses.append(loss.detach().cpu().item())
                        log.info(
                            f"[{phase} Epoch {curr_epoch} Step {idx}] "
                            f"loss={loss.item():.4f} | "
                            f"peak_mem={peak_memory / (1024**2):.0f}MB"
                        )
                        if np.sum(np.isnan(sft_answer_losses)) == 0:
                            np.savetxt(
                                self.filename + "_sft_answer_loss.csv",
                                np.array(sft_answer_losses),
                            )
                    elif use_grpo:
                        grpo_losses.append(loss.detach().cpu().item())
                        grpo_rewards.append(metrics["grpo_reward"])
                        log.info(
                            f"[{phase} Epoch {curr_epoch} Step {idx}] "
                            f"loss={loss.item():.4f} | "
                            f"reward={metrics['grpo_reward']:.4f} | "
                            f"adv_std={metrics['grpo_advantage_std']:.4f} | "
                            f"peak_mem={peak_memory / (1024**2):.0f}MB"
                        )
                        if np.sum(np.isnan(grpo_losses)) == 0:
                            np.savetxt(
                                self.filename + "_grpo_loss.csv", np.array(grpo_losses)
                            )
                            np.savetxt(
                                self.filename + "_grpo_reward.csv", np.array(grpo_rewards)
                            )
                    else:
                        saved_losses.append(loss.detach().cpu().item())
                        train_reward_accs.append(metrics["reward_accuracy"])
                        log.info(
                            f"[{phase} Epoch {curr_epoch} Step {idx}] "
                            f"loss={loss.item():.4f} | "
                            f"reward_acc={metrics['reward_accuracy']:.3f} | "
                            f"margin={metrics['reward_margin']:.4f} | "
                            f"chosen_r={metrics['chosen_reward']:.4f} | "
                            f"rejected_r={metrics['rejected_reward']:.4f} | "
                            f"peak_mem={peak_memory / (1024**2):.0f}MB"
                        )
                        if np.sum(np.isnan(saved_losses)) == 0:
                            np.savetxt(
                                self.filename + "_dpo_loss.csv", np.array(saved_losses)
                            )

                torch.cuda.empty_cache()

            # ----------------------------------------------------------
            # Evaluation
            # ----------------------------------------------------------
            self._model.eval()
            torch.cuda.empty_cache()

            # --- MAE/acc + CE from SFT trajectory train/val/test (teacher-forced) ---
            mean_train_mae = mean_train_ce = None
            if self._bt_program_mode:
                mean_train_mae, mean_train_ce = self._compute_bt_answer_metrics(
                    self._dataloader_bt_eval_train, label=f"bt_train_ep{curr_epoch}")
                mean_val_mae,  mean_val_ce  = self._compute_bt_answer_metrics(
                    self._dataloader_bt_eval_val,  label=f"bt_val_ep{curr_epoch}")
                mean_test_mae, mean_test_ce = self._compute_bt_answer_metrics(
                    self._dataloader_bt_eval_test, label=f"bt_test_ep{curr_epoch}")
            else:
                if self._dataloader_sft_traj_train is not None:
                    mean_train_mae, mean_train_ce = self._compute_traj_metrics(
                        self._dataloader_sft_traj_train, label=f"traj_train_ep{curr_epoch}")
                mean_val_mae,  mean_val_ce  = self._compute_traj_metrics(self._dataloader_sft_traj_val,  label=f"traj_val_ep{curr_epoch}")
                mean_test_mae, mean_test_ce = self._compute_traj_metrics(self._dataloader_sft_traj_test, label=f"traj_test_ep{curr_epoch}")
            val_maes.append(mean_val_mae)
            test_maes.append(mean_test_mae)
            val_ces.append(mean_val_ce)
            test_ces.append(mean_test_ce)
            np.savetxt(self.filename + "_maes_val.csv",  np.array(val_maes))
            np.savetxt(self.filename + "_maes_test.csv", np.array(test_maes))
            np.savetxt(self.filename + "_ces_val.csv",   np.array(val_ces))
            np.savetxt(self.filename + "_ces_test.csv",  np.array(test_ces))
            if mean_train_mae is not None:
                train_maes.append(mean_train_mae)
                train_ces.append(mean_train_ce)
                np.savetxt(self.filename + "_maes_train.csv", np.array(train_maes))
                np.savetxt(self.filename + "_ces_train.csv",  np.array(train_ces))
            torch.cuda.empty_cache()

            log.info("")
            log.info(f"----------- END OF EPOCH {curr_epoch} -----------")
            if mean_train_mae is not None:
                tag = "BT-TRAIN" if self._bt_program_mode else "TRAJ-TRAIN"
                log.info(f"{tag} MAE={mean_train_mae:.2f}  CE={mean_train_ce:.4f}")
            log.info(f"TRAJ-VAL  MAE={mean_val_mae:.2f}  CE={mean_val_ce:.4f}")
            log.info(f"TRAJ-TEST MAE={mean_test_mae:.2f}  CE={mean_test_ce:.4f}")
            log.info(f"----------- END OF EPOCH {curr_epoch} -----------")
            log.info("")

            # Track val CE for early stopping. Save checkpoint either way so the
            # latest weights remain on disk (existing behavior).
            if mean_val_ce is not None and mean_val_ce < best_val_ce:
                log.info(f"Val CE improved: {best_val_ce:.4f} -> {mean_val_ce:.4f}")
                best_val_ce = mean_val_ce
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1
                log.info(f"Val CE did not improve: {mean_val_ce:.4f} (best: {best_val_ce:.4f}) "
                         f"[epochs since improvement: {epochs_since_improvement}]")

            self.save_checkpoint(epoch=curr_epoch)

            self.epochs_run += 1
            self._model.train()

            if (self._early_stopping_patience
                    and epochs_since_improvement >= self._early_stopping_patience):
                log.info(
                    f"Early stopping at epoch {curr_epoch}: val CE did not "
                    f"improve for {epochs_since_improvement} consecutive eval "
                    f"epochs (patience={self._early_stopping_patience}). "
                    f"Best val CE: {best_val_ce:.4f}."
                )
                break

        log.info("Training complete.")

    def cleanup(self) -> None:
        self._metric_logger.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

@config.parse
def recipe_main(cfg: DictConfig) -> None:
    """
    Entry point for the DPO recipe.

    Configurable parameters are read in the following order:
        - Parameters specified in config (see available configs via ``tune ls``)
        - Overwritten by arguments from the command-line
    """
    config.log_config(recipe_name="LoRADPORecipeSingleDevice", cfg=cfg)

    global MODEL
    global NOTRAIN
    global FILENAME
    global log

    if "llama" in str(cfg.model._component_):
        MODEL = "llama"
        model_path = (
            "<DATA_ROOT>/resources/"
            "models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/"
            "e1945c40cd546c78e41f1151f4db032b271faeaa"
        )
    elif "qwen" in str(cfg.model._component_):
        MODEL = "qwen"
        model_path = (
            "<DATA_ROOT>/resources/qwen/Qwen2-7B-Instruct"
        )

    NOTRAIN = False
    sft_mode = cfg.get("sft_mode", False)
    bt_program_mode = bool(cfg.get("bt_program_mode", False)) or \
                      bool(cfg.get("bt_scratchpad_mode", False))
    bt_reasoning = str(cfg.get(
        "bt_reasoning",
        "scratchpad" if cfg.get("bt_scratchpad_mode", False) else "program",
    ))
    beta = cfg.get("dpo_beta", 0.1)
    ref_free = cfg.get("reference_free", False)
    ref_suffix = "_reffree" if ref_free else ""
    lr = cfg.optimizer.get("lr", 0)
    if bt_program_mode:
        sft_lr = cfg.get("sft_lr", lr)
        n_train = int(cfg.get("bt_n_train", 200))
        task_tag = str(cfg.get("bt_train_task", "flight"))
        tag = "btprog" if bt_reasoning == "program" else "btscratch"
        run_tag = cfg.get("run_tag", None)
        base = f"{MODEL}-{tag}-n{n_train}-{task_tag}-lr{sft_lr}-seed{cfg.seed}"
        filename = f"{base}-{run_tag}" if run_tag else base
    elif sft_mode:
        sft_epochs = cfg.get("sft_epochs", cfg.epochs)
        sft_lr = cfg.get("sft_lr", lr)
        phase2 = cfg.get("phase2", "grpo")
        phase2_lr = cfg.get("phase2_lr", cfg.get("grpo_lr", lr))
        traj_mode = cfg.get("sft_traj_mode", cfg.get("use_sft_traj", True))
        if traj_mode is True:
            traj_tag = "scratchpad"
        elif traj_mode in (False, None):
            traj_tag = "notraj"
        else:
            traj_tag = str(traj_mode)
        loss_mode = cfg.get("loss_mode", "distribution")
        filename = f"{MODEL}-sft{sft_epochs}-{traj_tag}-{loss_mode}-{phase2}-sftlr{sft_lr}-p2lr{phase2_lr}-seed{cfg.seed}"
    else:
        filename = (
            f"{MODEL}-dpo-grpo-beta{beta}{ref_suffix}-lr{lr}-seed{cfg.seed}"
        )

    FILENAME = filename
    log = logging.getLogger(__name__)
    logging.basicConfig(
        filename=f"{FILENAME}.log", encoding="utf-8", level=logging.DEBUG
    )
    log.info("Starting DPO recipe")

    recipe = LoRADPORecipeSingleDevice(cfg=cfg)
    recipe.setup(cfg=cfg)
    recipe.train()
    recipe.cleanup()


if __name__ == "__main__":
    sys.exit(recipe_main())
