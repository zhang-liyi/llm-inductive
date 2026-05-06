"""
temperature_scaling_fit.py

Fit a single temperature T for either the pretrained Llama-3-8B-Instruct or
a LoRA-fine-tuned checkpoint (program-trained / pyro-dist / fusion / …) as a
post-hoc calibration baseline.

For a given --dataset, we:
  1. load <=--n_max train-split examples (for BT: flight_Nfeatures / flight_human
     records that are NOT in the 2238-point test set),
  2. run the model in the same teacher-force scoring protocol used for the
     validation eval,
  3. fit T by minimizing NLL of softmax(log(probs)/T) on the collected probs.

Output: results/<group>/ts_fit_<dataset>[_<tag>].json with fields:
  { dataset, tag, ckpt_dir, T, n_fit, nll_at_1, nll_at_T, records: [...] }

Usage (pretrained):
  python temperature_scaling_fit.py --dataset mmlu
Usage (LoRA ckpt):
  python temperature_scaling_fit.py --dataset mmlu --tag program \\
      --ckpt_dir /path/to/sft_program_distribution_msl8192_lora8_.../epoch_0
TruthfulQA is intentionally not supported: no train split exists, and its TS
row is defined as T=1 (handled by temperature_scaling_apply.py).
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import minimize_scalar

_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS))

from evaluate_bayesian_teaching import (  # noqa: E402
    build_chat_prefix,
    find_answer_position,
    get_choice_token_ids,
    load_model_and_tokenizer,
    load_pretrained_model_and_tokenizer,
)
from evaluate_text_classification import (  # noqa: E402
    ASSISTANT_PREFILL as TC_ASSISTANT_PREFILL,
    CHOICE_LETTERS,
    PROMPT_INSTRUCTION as TC_PROMPT_INSTRUCTION,
    get_letter_token_ids,
    score_example as tc_score_example,
)

HF_CACHE = "<DATA_ROOT>/hg_cache"
BT_TEST_FULL_JSONL = (
    _THIS.parent / "data_processing" / "bayesian_teaching_test.jsonl"
)
BT_EXTRA_SOURCES = {
    "flight_2features", "flight_3features", "flight_5features",
    "flight_6features", "flight_7features", "flight_8features",
    "flight_human",
}

# Dataset group -> results subdir
_GROUP = {
    "mmlu": "text_cls", "hellaswag": "text_cls", "arc_challenge": "text_cls",
    "winogrande": "text_cls", "bt": "bayesian_teaching",
}


# ── train-split loaders ──────────────────────────────────────────────────────

def _tc_example(task, prompt, gold_letter):
    return (task, {"input": prompt, "output": gold_letter})


def load_mmlu_train():
    from datasets import load_dataset
    ds = load_dataset(
        "cais/mmlu", "all", cache_dir=HF_CACHE,
        split="auxiliary_train", trust_remote_code=True,
    )
    letters = ["A", "B", "C", "D"]
    out = []
    for ex in ds:
        body = [f"Question: {ex['question']}", "Choices:"]
        for L, c in zip(letters, ex["choices"]):
            body.append(f"{L}) {c}")
        body.append("Answer:")
        out.append(_tc_example(ex.get("subject", "mmlu"),
                               "\n".join(body),
                               letters[ex["answer"]]))
    return out


def load_hellaswag_train():
    from datasets import load_from_disk
    ds = load_from_disk(os.path.join(HF_CACHE, "hellaswag_train_disk"))
    letters = ["A", "B", "C", "D"]
    out = []
    for ex in ds:
        body = [
            f"Context: {ex['ctx']}",
            "Which is the best continuation?",
            "Choices:",
        ]
        for L, e in zip(letters, ex["endings"]):
            body.append(f"{L}) {e}")
        body.append("Answer:")
        out.append(_tc_example(ex["activity_label"],
                               "\n".join(body),
                               letters[int(ex["label"])]))
    return out


def load_arc_challenge_train():
    from datasets import load_dataset
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge",
                      cache_dir=HF_CACHE, split="train")
    out = []
    for ex in ds:
        texts = ex["choices"]["text"]
        labels = ex["choices"]["label"]
        norm_labels = []
        for i, L in enumerate(labels):
            norm_labels.append(CHOICE_LETTERS[i] if L.isdigit() else L)
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
        out.append(_tc_example("arc_challenge", "\n".join(body), gold_letter))
    return out


def load_winogrande_train():
    from datasets import load_from_disk
    ds = load_from_disk(os.path.join(HF_CACHE, "winogrande_train_disk"))
    letters = ["A", "B"]
    out = []
    for ex in ds:
        body = [
            f"Sentence: {ex['sentence']}",
            "What does '_' refer to?",
            "Choices:",
            f"A) {ex['option1']}",
            f"B) {ex['option2']}",
            "Answer:",
        ]
        out.append(_tc_example(
            "winogrande",
            "\n".join(body),
            letters[int(ex["answer"]) - 1],
        ))
    return out


def load_bt_train():
    """BT 'train' for T fitting: extras from bayesian_teaching_test.jsonl that
    are NOT in the 2238-point base test set. Same build_prompt template,
    different feature-count/human variants (flight-only)."""
    out = []
    with open(BT_TEST_FULL_JSONL) as fh:
        for line in fh:
            ex = json.loads(line)
            if ex.get("source") in BT_EXTRA_SOURCES:
                out.append(ex)
    return out


_TC_LOADERS = {
    "mmlu": load_mmlu_train,
    "hellaswag": load_hellaswag_train,
    "arc_challenge": load_arc_challenge_train,
    "winogrande": load_winogrande_train,
}


# ── BT scoring (teacher-force, batched) ──────────────────────────────────────

@torch.no_grad()
def score_bt_batch(model, tokenizer, batch, choice_token_ids, device, max_seq_len):
    """Return list of (task, probs[3], true_idx) for a batch of BT examples."""
    import re
    choice_token_set = set(choice_token_ids.values())
    choice_tensor = torch.tensor(
        [choice_token_ids[1], choice_token_ids[2], choice_token_ids[3]],
        dtype=torch.long, device=device,
    )
    valid_chars = ("1", "2", "3")

    inputs, prefix_lens, gts, tasks = [], [], [], []
    for ex in batch:
        m = re.search(r"<([123])>", ex["output"])
        if not m:
            continue
        gt = int(m.group(1))
        prompt = ex["input"]
        prefix_text = build_chat_prefix(prompt, tokenizer)
        full_text = prefix_text + ex["output"]
        prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
        full_ids = tokenizer.encode(full_text, add_special_tokens=False)
        if len(full_ids) > max_seq_len:
            full_ids = full_ids[:max_seq_len]
            prefix_ids = prefix_ids[:max_seq_len]
        inputs.append(full_ids)
        prefix_lens.append(len(prefix_ids))
        gts.append(gt)
        tasks.append(ex.get("task", "bt"))

    if not inputs:
        return []

    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    max_len = max(len(x) for x in inputs)
    input_ids_t = torch.tensor(
        [ids + [pad_id] * (max_len - len(ids)) for ids in inputs],
        dtype=torch.long, device=device,
    )
    logits = model(tokens=input_ids_t)
    if isinstance(logits, list):
        logits = torch.cat(logits, dim=1)

    results = []
    for b_idx in range(len(inputs)):
        ids_tensor = torch.tensor(inputs[b_idx], dtype=torch.long)
        ans_pos = find_answer_position(
            ids_tensor, prefix_lens[b_idx], choice_token_set,
            tokenizer, valid_chars,
        )
        logit_vec = logits[b_idx, max(0, ans_pos - 1)]
        choice_logits = logit_vec[choice_tensor].float()
        probs = F.softmax(choice_logits, dim=0).cpu().numpy()
        results.append((tasks[b_idx], probs.tolist(), gts[b_idx] - 1))
    return results


# ── T fit ────────────────────────────────────────────────────────────────────

def fit_temperature(probs, true_idx):
    probs = np.clip(np.asarray(probs, dtype=np.float64), 1e-12, 1.0)
    logp = np.log(probs)
    true_idx = np.asarray(true_idx)
    N = len(probs)

    def neg_ll(T):
        z = logp / T
        z = z - z.max(axis=1, keepdims=True)
        logZ = np.log(np.exp(z).sum(axis=1))
        return float(-(z[np.arange(N), true_idx] - logZ).mean())

    res = minimize_scalar(neg_ll, bounds=(0.05, 100.0),
                          method="bounded", options={"xatol": 1e-4})
    return float(res.x), float(res.fun), neg_ll(1.0)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    choices=list(_TC_LOADERS.keys()) + ["bt"])
    ap.add_argument("--n_max", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model_path", default=None,
                    help="Pretrained model dir. Defaults to Llama-3-8B-Instruct.")
    ap.add_argument("--ckpt_dir", default=None,
                    help="LoRA ckpt dir (epoch_N). If set, overrides --model_path "
                         "and the model is loaded via load_model_and_tokenizer.")
    ap.add_argument("--tag", default=None,
                    help="Optional model tag added to the output filename "
                         "(e.g. 'program' -> ts_fit_<dataset>_program.json).")
    ap.add_argument("--max_seq_len", type=int, default=4096)
    ap.add_argument("--batch_size", type=int, default=8,
                    help="Only used for BT (teacher-force batched).")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--progress_every", type=int, default=50)
    ap.add_argument("--output_file", default=None)
    args = ap.parse_args()

    # Load & subsample examples
    if args.dataset == "bt":
        examples = load_bt_train()
    else:
        examples = _TC_LOADERS[args.dataset]()
    print(f"[{args.dataset}] {len(examples)} candidate examples; "
          f"sampling {min(args.n_max, len(examples))} (seed={args.seed}).")
    rng = random.Random(args.seed)
    rng.shuffle(examples)
    examples = examples[: args.n_max]

    # Load model
    if args.ckpt_dir:
        print(f"[{args.dataset}] loading LoRA ckpt from {args.ckpt_dir}"
              + (f"  tag={args.tag}" if args.tag else ""))
        model, tokenizer = load_model_and_tokenizer(
            args.ckpt_dir, device=args.device, dtype=args.dtype)
    else:
        kwargs = {}
        if args.model_path:
            kwargs["model_path"] = args.model_path
        model, tokenizer = load_pretrained_model_and_tokenizer(
            device=args.device, dtype=args.dtype, **kwargs)

    # Score
    records = []
    if args.dataset == "bt":
        choice_token_ids = get_choice_token_ids(tokenizer)
        print(f"Choice token ids: {choice_token_ids}")
        for i in range(0, len(examples), args.batch_size):
            batch = examples[i : i + args.batch_size]
            outs = score_bt_batch(model, tokenizer, batch, choice_token_ids,
                                  args.device, args.max_seq_len)
            for task, probs, true_idx in outs:
                records.append({"task": task, "probs": probs,
                                "true_idx": int(true_idx)})
            if (i // args.batch_size) % 5 == 0:
                print(f"  scored {min(i+args.batch_size, len(examples))}/"
                      f"{len(examples)}", flush=True)
    else:
        letter_ids = get_letter_token_ids(tokenizer)
        print(f"Letter token ids (A..Z): {letter_ids}")
        for i, (task, ex) in enumerate(examples):
            out_letter = ex["output"].strip()
            if not out_letter or out_letter[0] not in CHOICE_LETTERS:
                continue
            true_idx = CHOICE_LETTERS.index(out_letter[0])
            probs = tc_score_example(
                model, tokenizer, ex["input"], letter_ids,
                args.device, args.max_seq_len,
            )
            records.append({
                "task": task,
                "probs": probs.tolist(),
                "true_idx": int(true_idx),
            })
            if (i + 1) % args.progress_every == 0:
                print(f"  scored {i + 1}/{len(examples)}", flush=True)

    # Fit T
    probs_arr = [r["probs"] for r in records]
    true_arr = [r["true_idx"] for r in records]
    T, nll_T, nll_1 = fit_temperature(probs_arr, true_arr)
    print(f"\n[{args.dataset}] n_fit={len(records)}  T={T:.4f}  "
          f"NLL(T=1)={nll_1:.4f}  NLL(T*)={nll_T:.4f}")

    # Save
    tag_suffix = f"_{args.tag}" if args.tag else ""
    out_path = args.output_file or str(
        _THIS / "results" / _GROUP[args.dataset]
        / f"ts_fit_{args.dataset}{tag_suffix}.json"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    payload = {
        "dataset": args.dataset,
        "tag": args.tag,
        "ckpt_dir": args.ckpt_dir,
        "model_path": args.model_path,
        "T": T,
        "n_fit": len(records),
        "nll_at_1": nll_1,
        "nll_at_T": nll_T,
        "seed": args.seed,
        "records": records,
    }
    with open(out_path, "w") as fh:
        json.dump(payload, fh)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
