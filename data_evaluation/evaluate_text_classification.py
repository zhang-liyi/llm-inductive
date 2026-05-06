"""
evaluate_text_classification.py

Evaluate a (LoRA or pretrained) Llama3-8B checkpoint on cls45 / chembench /
legalbench *validation* splits used in ../bayes-llm/torchtune-normal/.

For each example the model sees the prompt (which already ends with
``Answer:``), we take the logits at the final position and softmax over the
4 single-token choice ids ``{A,B,C,D}``.  Argmax among those 4 is the
prediction.  The ground-truth answer is a single letter.

Metrics reported per task and overall: accuracy, NLL (CE over true letter),
MAE (|pred_letter_idx - true_letter_idx|), ECE (15-bin over predicted-class
confidence), and valid_rate (always 1 here).

Usage
-----
    python evaluate_text_classification.py \\
        --ckpt_dir <lora-ckpt-dir-or-pretrained-base> \\
        --dataset cls45|chembench|legalbench \\
        --output_file out.json
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

# Re-use loaders from the BT eval so we stay bug-for-bug compatible.
_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))
from evaluate_bayesian_teaching import (  # noqa: E402
    load_model_and_tokenizer,
    load_pretrained_model_and_tokenizer,
    build_chat_prefix,
)


BAYES_DATA_DIR = Path("<DATA_ROOT>/bayes-llm/data")
PROMPT_INSTRUCTION = (
    "Output only the answer choice in angular brackets, for example <LETTER>, "
    "where LETTER is one of A, B, C, D, etc."
)
# Teacher-force up to and including the opening '<' of the bracketed answer;
# the next token is then the bare letter A..Z (ids 32..57 in Llama3 tokenizer).
ASSISTANT_PREFILL = "The answer is: <"
DEFAULT_MODEL_PATH = (
    "<DATA_ROOT>/resources/"
    "models--meta-llama--Meta-Llama-3-8B-Instruct/"
    "snapshots/e1945c40cd546c78e41f1151f4db032b271faeaa"
)

CHOICE_LETTERS = tuple(chr(ord("A") + i) for i in range(26))

CLS45_TASKS = [
    "tweet_eval-stance_feminist", "ethos-national_origin", "tweet_eval-hate",
    "ag_news", "anli", "hate_speech18", "poem_sentiment", "climate_fever",
    "medical_questions_pairs", "tweet_eval-stance_atheism", "ethos-race",
    "ethos-religion", "superglue-cb", "wiki_qa", "yelp_polarity",
]
CHEMBENCH_TASKS = [
    "analytical_chemistry", "chemical_preference", "general_chemistry",
    "inorganic_chemistry", "materials_science", "organic_chemistry",
    "physical_chemistry", "technical_chemistry", "toxicity_and_safety",
]
LEGALBENCH_HELDOUT_TASKS = [
    "contract_nli_inclusion_of_verbally_conveyed_information",
    "contract_nli_limited_use", "contract_nli_no_licensing",
    "cuad_anti-assignment", "cuad_audit_rights",
    "cuad_competitive_restriction_exception", "cuad_effective_date",
    "cuad_joint_ip_ownership", "cuad_rofr-rofo-rofn",
    "cuad_unlimited-all-you-can-eat-license", "cuad_warranty_duration",
    "diversity_1", "diversity_4", "learned_hands_benefits",
    "maud_includes_consistent_with_past_practice",
    "opp115_user_access,_edit_and_deletion", "successor_liability",
    "supply_chain_disclosure_best_practice_training",
    "textualism_tool_dictionaries", "unfair_tos",
]
CHEMBENCH_SHUFFLE_SEED = 42


# ── dataset loading ────────────────────────────────────────────────────────────

def _read_json(path: Path) -> List[dict]:
    with open(path) as fh:
        return json.load(fh)


def load_cls45_val() -> List[Tuple[str, dict]]:
    out = []
    for task in CLS45_TASKS:
        p = BAYES_DATA_DIR / f"{task}_ins_inputoutput_icl_dev.json"
        for ex in _read_json(p):
            out.append((task, ex))
    return out


def load_chembench_val() -> List[Tuple[str, dict]]:
    # Replicate torchtune-normal/custom_chembench_icl: shuffle with seed 42,
    # take first half.  We use datasets.Dataset.shuffle for byte-identical
    # semantics.
    from datasets import load_dataset as hf_load_dataset
    out = []
    for task in CHEMBENCH_TASKS:
        p = BAYES_DATA_DIR / "chembench" / f"chembench_{task}_inputoutput_icl_dev.json"
        raw = hf_load_dataset("json", data_files=str(p), split="train")
        raw = raw.shuffle(seed=CHEMBENCH_SHUFFLE_SEED)
        half = len(raw) // 2
        for i in range(half):
            out.append((task, dict(raw[i])))
    return out


def load_legalbench_val() -> List[Tuple[str, dict]]:
    out = []
    for task in LEGALBENCH_HELDOUT_TASKS:
        p = BAYES_DATA_DIR / "legalbench" / f"legalbench_{task}_inputoutput_icl_val.json"
        for ex in _read_json(p):
            out.append((task, ex))
    return out


def load_mmlu_val() -> List[Tuple[str, dict]]:
    """MMLU validation split (1531 examples across 57 subjects), zero-shot.

    The standard prompt instruction + assistant-prefill '<' is applied at
    scoring time, same as for the other datasets.
    """
    from datasets import load_from_disk
    val = load_from_disk("<DATA_ROOT>/hg_cache/mmlu_validation_disk")
    letters = ["A", "B", "C", "D"]
    out = []
    for ex in val:
        body = [f"Question: {ex['question']}", "Choices:"]
        for L, c in zip(letters, ex["choices"]):
            body.append(f"{L}) {c}")
        body.append("Answer:")
        prompt = "\n".join(body)
        out.append((ex["subject"],
                    {"input": prompt, "output": letters[ex["answer"]]}))
    return out


def load_hellaswag_val() -> List[Tuple[str, dict]]:
    """HellaSwag validation split (10042 examples, 4-way MC)."""
    from datasets import load_from_disk
    val = load_from_disk(
        "<DATA_ROOT>/hg_cache/hellaswag_validation_disk"
    )
    out = []
    letters = ["A", "B", "C", "D"]
    for ex in val:
        body = [
            f"Context: {ex['ctx']}",
            "Which is the best continuation?",
            "Choices:",
        ]
        for L, e in zip(letters, ex["endings"]):
            body.append(f"{L}) {e}")
        body.append("Answer:")
        prompt = "\n".join(body)
        out.append((
            ex["activity_label"],
            {"input": prompt, "output": letters[int(ex["label"])]},
        ))
    return out


def load_winogrande_val() -> List[Tuple[str, dict]]:
    """Winogrande (debiased) validation split (1267 examples, 2-way MC)."""
    from datasets import load_from_disk
    val = load_from_disk(
        "<DATA_ROOT>/hg_cache/winogrande_validation_disk"
    )
    letters = ["A", "B"]
    out = []
    for ex in val:
        body = [
            f"Sentence: {ex['sentence']}",
            "What does '_' refer to?",
            "Choices:",
            f"A) {ex['option1']}",
            f"B) {ex['option2']}",
            "Answer:",
        ]
        prompt = "\n".join(body)
        # answer is '1' or '2' (1-indexed).
        out.append((
            "winogrande",
            {"input": prompt, "output": letters[int(ex["answer"]) - 1]},
        ))
    return out


def load_arc_challenge_val() -> List[Tuple[str, dict]]:
    """ARC-Challenge validation split (299 examples, mostly 4-way MC).

    A handful of examples use numeric answerKeys ('1', '2', …) — we map those
    to letters positionally (1→A, 2→B, …).
    """
    from datasets import load_from_disk
    val = load_from_disk(
        "<DATA_ROOT>/hg_cache/arc_challenge_validation_disk"
    )
    out = []
    for ex in val:
        texts = ex["choices"]["text"]
        labels = ex["choices"]["label"]
        # Normalize labels so answerKey always maps to a letter A..Z
        norm_labels = []
        for i, L in enumerate(labels):
            if L.isdigit():
                norm_labels.append(CHOICE_LETTERS[i])
            else:
                norm_labels.append(L)
        ak = ex["answerKey"]
        if ak.isdigit():
            idx = int(ak) - 1
            if idx < 0 or idx >= len(texts):
                continue
            gold_letter = CHOICE_LETTERS[idx]
        else:
            if ak not in norm_labels:
                continue
            gold_letter = ak
        body = [f"Question: {ex['question']}", "Choices:"]
        for L, t in zip(norm_labels, texts):
            body.append(f"{L}) {t}")
        body.append("Answer:")
        prompt = "\n".join(body)
        out.append(("arc_challenge",
                    {"input": prompt, "output": gold_letter}))
    return out


def load_truthfulqa_val() -> List[Tuple[str, dict]]:
    """TruthfulQA MC1 validation split (817 questions, variable # choices
    2..13).  Exactly one choice per question is correct (MC1)."""
    from datasets import load_from_disk
    val = load_from_disk(
        "<DATA_ROOT>/hg_cache/truthfulqa_mc_validation_disk"
    )
    out = []
    for ex in val:
        choices = ex["mc1_targets"]["choices"]
        labels = ex["mc1_targets"]["labels"]
        try:
            correct_idx = labels.index(1)
        except ValueError:
            continue  # malformed; shouldn't happen for MC1
        body = [f"Question: {ex['question']}", "Choices:"]
        for i, c in enumerate(choices):
            body.append(f"{CHOICE_LETTERS[i]}) {c}")
        body.append("Answer:")
        prompt = "\n".join(body)
        out.append((
            "truthfulqa",
            {"input": prompt, "output": CHOICE_LETTERS[correct_idx]},
        ))
    return out


LOADERS = {
    "cls45": load_cls45_val,
    "chembench": load_chembench_val,
    "legalbench": load_legalbench_val,
    "mmlu": load_mmlu_val,
    "truthfulqa": load_truthfulqa_val,
    "hellaswag": load_hellaswag_val,
    "winogrande": load_winogrande_val,
    "arc_challenge": load_arc_challenge_val,
}


# ── token helpers ──────────────────────────────────────────────────────────────

def get_letter_token_ids(tokenizer) -> List[int]:
    """Return the token ids for each capital letter A..Z as the next token
    after ``ASSISTANT_PREFILL`` (which ends with an opening ``<``).
    """
    base_ids = tokenizer.encode(ASSISTANT_PREFILL, add_special_tokens=False)
    ids = []
    for L in CHOICE_LETTERS:
        full = tokenizer.encode(ASSISTANT_PREFILL + L + ">",
                                add_special_tokens=False)
        if full[: len(base_ids)] != base_ids:
            raise RuntimeError(
                f"Prefix re-tokenization mismatch for {L!r}: "
                f"{full} vs base {base_ids}"
            )
        tail = full[len(base_ids):]
        if len(tail) != 2:
            raise RuntimeError(
                f"Expected 2 tokens after prefill for {L!r} (letter + '>'), "
                f"got {tail}"
            )
        ids.append(tail[0])
    if len(set(ids)) != len(CHOICE_LETTERS):
        raise RuntimeError(f"Letter tokens collide: {ids}")
    return ids


# ── evaluation ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def score_example(model, tokenizer, prompt: str, letter_ids: List[int],
                  device: str, max_seq_len: int) -> np.ndarray:
    """Return softmax probabilities over the 26 capital letters A..Z as the
    next token after the assistant prefill."""
    user_text = f"{PROMPT_INSTRUCTION}\n\n{prompt}"
    chat = build_chat_prefix(user_text, tokenizer) + ASSISTANT_PREFILL
    input_ids = tokenizer.encode(chat, add_special_tokens=False)
    if len(input_ids) > max_seq_len:
        input_ids = input_ids[-max_seq_len:]
    inp = torch.tensor([input_ids], dtype=torch.long, device=device)
    logits = model(tokens=inp)
    if isinstance(logits, list):
        logits = torch.cat(logits, dim=1)
    last_logits = logits[0, -1, :].float()
    choice_logits = last_logits[torch.tensor(letter_ids, device=device)]
    probs = F.softmax(choice_logits, dim=-1).cpu().numpy()
    return probs


def compute_ece(confidences: np.ndarray, correct: np.ndarray,
                n_bins: int = 15) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    N = len(confidences)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)
        if mask.sum() == 0:
            continue
        bin_conf = confidences[mask].mean()
        bin_acc = correct[mask].mean()
        ece += (mask.sum() / N) * abs(bin_conf - bin_acc)
    return float(ece)


def aggregate(items: List[dict]) -> dict:
    if not items:
        return {"n": 0}
    probs = np.array([it["probs"] for it in items])       # (n, 26)
    true_idx = np.array([it["true_idx"] for it in items]) # (n,)
    pred_idx = probs.argmax(axis=1)
    correct = (pred_idx == true_idx).astype(float)
    p_true = probs[np.arange(len(items)), true_idx]
    nll = -np.log(np.clip(p_true, 1e-12, 1.0)).mean()
    conf = probs.max(axis=1)
    ece = compute_ece(conf, correct)
    return {
        "n": len(items),
        "accuracy": float(correct.mean()),
        "ce_mean": float(nll),
        "mae": float(np.abs(pred_idx - true_idx).mean()),
        "ece": float(ece),
    }


def evaluate(model, tokenizer, examples: List[Tuple[str, dict]],
             device: str, max_seq_len: int, progress_every: int) -> dict:
    letter_ids = get_letter_token_ids(tokenizer)
    print(f"Letter token ids (A..Z): {letter_ids}")
    items = []
    skipped = 0
    for i, (task, ex) in enumerate(examples):
        prompt = ex["input"]
        out = ex["output"].strip()
        if not out or out[0] not in CHOICE_LETTERS:
            skipped += 1
            continue
        true_idx = CHOICE_LETTERS.index(out[0])
        probs = score_example(model, tokenizer, prompt, letter_ids,
                              device, max_seq_len)
        items.append({
            "task": task,
            "probs": probs.tolist(),
            "true_idx": true_idx,
            "true_letter": out[0],
            "pred_idx": int(probs.argmax()),
        })
        if (i + 1) % progress_every == 0:
            print(f"  {i + 1}/{len(examples)} done", flush=True)

    overall = aggregate(items)
    by_task = {}
    for task in sorted({it["task"] for it in items}):
        by_task[task] = aggregate([it for it in items if it["task"] == task])
    return {
        "overall": overall,
        "by_task": by_task,
        "skipped": skipped,
        "per_example": items,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", required=True,
                    help="LoRA ckpt dir (epoch_N), or pretrained base dir for Llama3-8B-Instruct baseline.")
    ap.add_argument("--dataset", required=True, choices=list(LOADERS.keys()))
    ap.add_argument("--output_file", required=True)
    ap.add_argument("--pretrained", action="store_true",
                    help="Treat --ckpt_dir as the pretrained base (no LoRA).")
    ap.add_argument("--max_seq_len", type=int, default=4096)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--progress_every", type=int, default=50)
    ap.add_argument("--start_idx", type=int, default=0,
                    help="Start index into the (sorted) dataset. Default 0.")
    ap.add_argument("--n_examples", type=int, default=None,
                    help="Number of examples starting at --start_idx.")
    args = ap.parse_args()

    print(f"Loading {args.dataset} validation split ...")
    examples = LOADERS[args.dataset]()
    print(f"  {len(examples)} examples across "
          f"{len({t for t,_ in examples})} tasks")
    if args.start_idx or args.n_examples is not None:
        end = args.start_idx + args.n_examples if args.n_examples else None
        examples = examples[args.start_idx:end]
        print(f"  sliced [{args.start_idx}:{end}] → {len(examples)}")

    if args.pretrained:
        model, tokenizer = load_pretrained_model_and_tokenizer(
            args.ckpt_dir, args.device, args.dtype)
    else:
        model, tokenizer = load_model_and_tokenizer(
            args.ckpt_dir, args.device, args.dtype)

    result = evaluate(model, tokenizer, examples,
                      args.device, args.max_seq_len, args.progress_every)
    payload = {
        "ckpt_dir": args.ckpt_dir,
        "dataset": args.dataset,
        "n_examples": len(examples),
        "max_seq_len": args.max_seq_len,
        "pretrained": args.pretrained,
        "summary": {
            "overall": result["overall"],
            "by_task": result["by_task"],
            "skipped": result["skipped"],
        },
        "per_example": result["per_example"],
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    with open(args.output_file, "w") as fh:
        json.dump(payload, fh, indent=2, default=str)

    ov = result["overall"]
    print("\n=== Text-Classification Eval Summary ===")
    print(f"  Dataset : {args.dataset}")
    print(f"  Overall : acc={ov.get('accuracy', float('nan')):.3f}  "
          f"NLL={ov.get('ce_mean', float('nan')):.3f}  "
          f"MAE={ov.get('mae', float('nan')):.3f}  "
          f"ECE={ov.get('ece', float('nan')):.3f}  n={ov['n']}")
    for task, s in result["by_task"].items():
        print(f"  {task:50s}: acc={s['accuracy']:.3f}  NLL={s['ce_mean']:.3f}  "
              f"ECE={s['ece']:.3f}  n={s['n']}")
    print(f"\nResults saved to {args.output_file}")


if __name__ == "__main__":
    main()
