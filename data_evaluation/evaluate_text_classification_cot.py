"""
evaluate_text_classification_cot.py

Chain-of-thought variant of ``evaluate_text_classification.py`` for the
letter-labelled MCQ benchmarks (mmlu, truthfulqa, hellaswag, winogrande,
arc_challenge, cls45, chembench, legalbench).

Where the base script reads the answer distribution off the very first
assistant token, this script lets the model reason first:

    user:      <CoT instruction>\\n\\n<question + choices>
    assistant: <freely generated reasoning …>
               The answer is: <?          ← restricted softmax over A..Z read here

Reported metrics are identical in definition to the non-CoT script
(accuracy / NLL / MAE / ECE over the same 26-way restricted softmax), so
rows can be compared directly.  Two extras are recorded:

    valid_rate   fraction of examples where an answer letter could be parsed
                 straight out of the free generation
    agree_rate   fraction (of those) where the parsed letter matches the
                 letter picked by the scored softmax

Usage
-----
    python evaluate_text_classification_cot.py \\
        --pretrained --dataset mmlu \\
        --output_file results/text_cls/cot_base_mmlu.json
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

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))

from evaluate_text_classification import (  # noqa: E402
    LOADERS, CHOICE_LETTERS, ASSISTANT_PREFILL,
    aggregate, get_letter_token_ids,
)
from evaluate_bayesian_teaching import (  # noqa: E402
    DEFAULT_MODEL_PATH, build_chat_prefix,
    load_model_and_tokenizer, load_pretrained_model_and_tokenizer,
)
from cot_eval_utils import (  # noqa: E402
    ANSWER_CUE, SC_TEMPERATURE, SC_TOP_K, FailureTracker, build_scoring_context,
    eos_token_ids, free_generate, last_position_logits, majority_vote,
    marginalize, restricted_probs, sc_extras, sc_seed,
    setup_generation_caches, strip_self_answer, vote_distribution,
)


COT_INSTRUCTION = (
    "Think step by step. First work through the question in your own words, "
    "then finish with your final answer on a new line in the form "
    "The answer is: <LETTER>, where LETTER is one of A, B, C, D, etc."
)

# The model's own final answer, e.g. "<C>" — used to cut the self-generated
# answer off the reasoning before the canonical cue is appended.
_SELF_ANSWER_RE = re.compile(r"<\s*([A-Z])\s*>")
# Fallback for models that answer with a bare letter after a lead-in phrase.
# The lead-in is matched case-insensitively but the letter itself must be
# upper-case: with (?i) on the letter too, ordinary prose like "the answer is a
# multiple of 6" parses as answer "A".
_BARE_ANSWER_RE = re.compile(
    r"(?i:(?:the\s+|my\s+)?(?:final\s+)?answer\s*(?:is)?\s*:?\s*\**\s*)([A-Z])\b")
# What counts as the model's final answer for *stripping* purposes: either
# form. Llama-3 ends CoT with a bare "The answer is: D" about as often as with
# "<D>", and leaving the bare form in place would append our cue directly after
# the model's own — giving those examples a duplicated question and a different
# scoring context from the bracketed ones.
_STRIP_RE = re.compile(
    r"<\s*[A-Z]\s*>"
    r"|(?i:(?:the\s+|my\s+)?(?:final\s+)?answer\s*(?:is)?\s*:?\s*\**\s*)[A-Z]\b")

_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16,
           "float32": torch.float32}


def parse_letter(text: str) -> Optional[str]:
    """The answer letter the model committed to last.

    Both the bracketed form ``<C>`` and the lead-in form ``the answer is C``
    count, and whichever appears *latest* wins — reasoning traces often quote
    "<A>" while discussing a choice they go on to reject.
    """
    cands = [(m.start(), m.group(1)) for m in _SELF_ANSWER_RE.finditer(text)]
    cands += [(m.start(), m.group(1)) for m in _BARE_ANSWER_RE.finditer(text)]
    if not cands:
        return None
    return max(cands, key=lambda t: t[0])[1].upper()


@torch.no_grad()
def eval_one(
    model, tokenizer, task: str, ex: dict, letter_ids: List[int],
    device: str, caches_on: bool, max_seq_len: int, max_new_tokens: int,
    eos_ids: List[int], temperature: float, top_k: Optional[int],
    n_samples: int = 1, global_idx: int = 0, base_seed: int = 0,
    sample_offset: int = 0,
) -> Optional[dict]:
    out = ex["output"].strip()
    if not out or out[0] not in CHOICE_LETTERS:
        return None
    true_idx = CHOICE_LETTERS.index(out[0])

    user_text = f"{COT_INSTRUCTION}\n\n{ex['input']}"
    prefix_text = build_chat_prefix(user_text, tokenizer)
    prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
    # Leave room for the reasoning + cue inside the context window.
    budget = max_seq_len - max_new_tokens - 16
    if len(prefix_ids) > budget:
        prefix_ids = prefix_ids[-budget:]

    per_probs, per_pred, n_tok = [], [], []
    n_trunc = 0
    first_text = first_self = None
    first_parsed = None

    for k in range(n_samples):
        if n_samples > 1 or sample_offset:
            torch.manual_seed(sc_seed(base_seed, global_idx,
                                          sample_offset + k))
        gen_ids = free_generate(
            model, prefix_ids, device, max_new_tokens, eos_ids, caches_on,
            temperature=temperature, top_k=top_k,
        )
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        reasoning, self_answer = strip_self_answer(gen_text, _STRIP_RE)
        parsed = parse_letter(gen_text)

        ctx_ids = build_scoring_context(tokenizer, prefix_ids, reasoning,
                                        ANSWER_CUE)
        logit_vec = last_position_logits(model, ctx_ids, device, max_seq_len)
        probs_k = restricted_probs(logit_vec, letter_ids)

        per_probs.append(probs_k)
        # A path votes for its own parsed conclusion, falling back to its
        # scored argmax when the text could not be parsed.
        per_pred.append(CHOICE_LETTERS.index(parsed)
                        if parsed in CHOICE_LETTERS else int(np.argmax(probs_k)))
        n_tok.append(len(gen_ids))
        n_trunc += int(len(gen_ids) >= max_new_tokens)
        if k == 0:
            first_text, first_self, first_parsed = gen_text, self_answer, parsed

    probs = marginalize(per_probs)
    pred_idx = int(np.argmax(probs))
    vote_idx = majority_vote(per_pred, probs)

    rec = {
        "global_idx": global_idx,
        "sample_idx": sample_offset,
        "task": task,
        "idx": global_idx,
        "probs": probs.tolist(),
        "true_idx": true_idx,
        "true_letter": out[0],
        "pred_idx": pred_idx,
        "pred_letter": CHOICE_LETTERS[pred_idx],
        "parsed_letter": first_parsed,
        "valid": first_parsed is not None,
        "agrees": (first_parsed == CHOICE_LETTERS[pred_idx]) if first_parsed else None,
        "self_answer": first_self,
        "n_gen_tokens": float(np.mean(n_tok)),
        "truncated": n_trunc == n_samples,
        "cot_text": first_text,
    }
    if n_samples > 1:
        votes = vote_distribution(per_pred, len(letter_ids))
        rec.update({
            "n_samples": n_samples,
            "vote_probs": votes,
            "vote_pred": vote_idx,
            "pred_idx_marginal": pred_idx,
            "vote_correct": vote_idx == true_idx,
            "vote_margin": max(votes) if votes else None,
            "sample_preds": per_pred,
            "trunc_frac": n_trunc / n_samples,
        })
    return rec


def cot_extras(items: List[dict]) -> dict:
    """valid_rate / agree_rate / truncation / mean CoT length, for a list of items."""
    if not items:
        return {}
    valid = [it for it in items if it.get("valid")]
    agree = [it for it in valid if it.get("agrees")]
    return {
        "valid_rate": len(valid) / len(items),
        "agree_rate": (len(agree) / len(valid)) if valid else float("nan"),
        # Reasoning that hit the token budget never reached a conclusion — the
        # answer cue is appended mid-sentence, so a high rate here means the
        # numbers reflect interrupted reasoning, not CoT.
        "trunc_rate": float(np.mean([bool(it.get("truncated")) for it in items])),
        "mean_gen_tokens": float(np.mean([it["n_gen_tokens"] for it in items])),
    }


def _vote_accuracy(items: List[dict]) -> dict:
    """Canonical self-consistency accuracy (majority vote over sampled paths),
    reported next to the mixture-based accuracy the other metrics use."""
    v = [it for it in items if it.get("vote_correct") is not None]
    if not v:
        return {}
    return {"vote_accuracy": sum(bool(it["vote_correct"]) for it in v) / len(v)}


def summarize(items: List[dict], skipped: int) -> dict:
    overall = aggregate(items)
    overall.update(cot_extras(items))
    overall.update(sc_extras(items))
    overall.update(_vote_accuracy(items))
    by_task = {}
    for task in sorted({it["task"] for it in items}):
        sub = [it for it in items if it["task"] == task]
        s = aggregate(sub)
        s.update(cot_extras(sub))
        s.update(sc_extras(sub))
        s.update(_vote_accuracy(sub))
        by_task[task] = s
    return {"overall": overall, "by_task": by_task, "skipped": skipped}


def main():
    ap = argparse.ArgumentParser(
        description="Chain-of-thought MCQ evaluation (free reasoning, then "
                    "restricted-softmax answer).")
    ap.add_argument("--ckpt_dir", default=None,
                    help="LoRA checkpoint dir (epoch_N).")
    ap.add_argument("--pretrained", action="store_true",
                    help="Evaluate the pretrained base model (no LoRA).")
    ap.add_argument("--model_path", default=DEFAULT_MODEL_PATH,
                    help="Base HF model dir (used with --pretrained).")
    ap.add_argument("--dataset", required=True, choices=list(LOADERS.keys()))
    ap.add_argument("--output_file", required=True)
    ap.add_argument("--start_idx", type=int, default=0)
    ap.add_argument("--n_examples", type=int, default=None)
    ap.add_argument("--subsample_n", type=int, default=None,
                    help="Evaluate a deterministic random subset of this many "
                         "examples (by absolute dataset index) instead of the "
                         "full split. Sharding then applies to the subset.")
    ap.add_argument("--subsample_seed", type=int, default=1234,
                    help="Seed for --subsample_n. Must match across runs for "
                         "greedy and self-consistency to be comparable.")
    ap.add_argument("--sample_idx", type=int, default=None,
                    help="Run exactly ONE reasoning path — path j of a "
                         "k-way self-consistency run — so the k paths can "
                         "be spread over k separate jobs and recombined by "
                         "combine_sc_samples.py. Seeds match what an "
                         "in-process --n_samples k run would have drawn.")
    ap.add_argument("--n_samples", type=int, default=1,
                    help="Self-consistency: sample this many reasoning paths "
                         "per example and marginalise over them.")
    ap.add_argument("--sc_defaults", action="store_true",
                    help="Use Wang et al. self-consistency decoding settings "
                         f"(temperature={SC_TEMPERATURE}, top_k={SC_TOP_K}).")
    ap.add_argument("--max_seq_len", type=int, default=4096)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=1,
                    help="1 = greedy decoding (default). >1 samples.")
    ap.add_argument("--seed", type=int, default=0,
                    help="RNG seed, only relevant when --top_k > 1.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="bfloat16", choices=list(_DTYPES))
    ap.add_argument("--no_kv_cache", action="store_true",
                    help="Disable KV-cached generation (much slower).")
    ap.add_argument("--progress_every", type=int, default=25)
    ap.add_argument("--checkpoint_every", type=int, default=100)
    args = ap.parse_args()

    if not args.pretrained and args.ckpt_dir is None:
        ap.error("Provide --ckpt_dir or --pretrained.")
    if args.sc_defaults:
        args.temperature, args.top_k = SC_TEMPERATURE, SC_TOP_K
    if (args.n_samples > 1 or args.sample_idx is not None) and args.top_k == 1:
        # top_k=1 collapses sampling onto the argmax, so every "sample" would
        # be the identical greedy trace — self-consistency would silently
        # reduce to plain CoT at k times the cost.
        ap.error("Sampled decoding is required for --n_samples > 1 or "
                 "--sample_idx: pass --sc_defaults, or set --top_k > 1.")
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)

    print(f"Loading {args.dataset} validation split …")
    examples = LOADERS[args.dataset]()
    print(f"  {len(examples)} examples across {len({t for t, _ in examples})} tasks")

    # Carry the absolute dataset index through subsampling and sharding, so a
    # subsampled run can be paired example-for-example against a full run of
    # the same benchmark.
    indexed = list(enumerate(examples))
    if args.subsample_n and args.subsample_n < len(indexed):
        rng = random.Random(args.subsample_seed)
        keep = sorted(rng.sample(range(len(indexed)), args.subsample_n))
        indexed = [indexed[i] for i in keep]
        print(f"  subsampled {len(indexed)} of {len(examples)} "
              f"(seed={args.subsample_seed}); first idxs {keep[:5]}")
    elif args.subsample_n:
        print(f"  --subsample_n {args.subsample_n} >= split size "
              f"{len(indexed)}; using the full split")

    if args.start_idx or args.n_examples is not None:
        end = args.start_idx + args.n_examples if args.n_examples else None
        indexed = indexed[args.start_idx:end]
        print(f"  sliced [{args.start_idx}:{end}] → {len(indexed)}")
    examples = indexed

    if args.pretrained:
        model, tokenizer = load_pretrained_model_and_tokenizer(
            args.model_path, args.device, args.dtype)
    else:
        model, tokenizer = load_model_and_tokenizer(
            args.ckpt_dir, args.device, args.dtype)

    letter_ids = get_letter_token_ids(tokenizer)
    print(f"Letter token ids (A..Z): {letter_ids}")
    eos_ids = eos_token_ids(tokenizer)
    print(f"Stop token ids: {eos_ids}")

    caches_on = False
    if not args.no_kv_cache:
        caches_on = setup_generation_caches(
            model, _DTYPES[args.dtype], args.max_seq_len + 16)
    print(f"KV-cached generation: {caches_on}")
    if args.top_k and args.top_k > 1:
        torch.manual_seed(args.seed)

    items: List[dict] = []
    skipped = 0
    failures = FailureTracker()
    ckpt_path = args.output_file.replace(".json", "_partial.json")
    for i, (abs_idx, (task, ex)) in enumerate(examples):
        try:
            r = eval_one(
                model, tokenizer, task, ex, letter_ids, args.device, caches_on,
                args.max_seq_len, args.max_new_tokens, eos_ids,
                args.temperature, args.top_k,
                n_samples=args.n_samples,
                global_idx=abs_idx, base_seed=args.seed,
                sample_offset=args.sample_idx or 0,
            )
        except Exception as exc:
            failures.record(i, exc, len(items))
            skipped += 1
            continue
        if r is None:
            skipped += 1
            continue
        items.append(r)

        if (i + 1) % args.progress_every == 0:
            ov = aggregate(items)
            print(f"  {i + 1}/{len(examples)}  acc={ov['accuracy']:.3f} "
                  f"NLL={ov['ce_mean']:.3f}  n={ov['n']}", flush=True)
        if (i + 1) % args.checkpoint_every == 0:
            tmp = ckpt_path + ".tmp"
            with open(tmp, "w") as fh:
                json.dump({"progress": i + 1, "total": len(examples),
                           "summary": summarize(items, skipped),
                           "per_example": items}, fh)
            os.replace(tmp, ckpt_path)

    if not items:
        print("No results.")
        sys.exit(1)

    summary = summarize(items, skipped)
    payload = {
        "ckpt_dir": args.ckpt_dir or args.model_path,
        "dataset": args.dataset,
        "mode": "cot" if args.n_samples == 1 else "cot_sc",
        "n_samples": args.n_samples,
        "sample_idx": args.sample_idx,
        "subsample_n": args.subsample_n,
        "subsample_seed": args.subsample_seed,
        "n_examples": len(items),
        "start_idx": args.start_idx,
        "max_seq_len": args.max_seq_len,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "pretrained": args.pretrained,
        "summary": summary,
        "per_example": items,
    }
    with open(args.output_file, "w") as fh:
        json.dump(payload, fh, indent=2, default=str)

    ov = summary["overall"]
    print("\n=== CoT Text-Classification Eval Summary ===")
    print(f"  Dataset : {args.dataset}")
    print(f"  Overall : acc={ov['accuracy']:.3f}  NLL={ov['ce_mean']:.3f}  "
          f"MAE={ov['mae']:.3f}  ECE={ov['ece']:.3f}  n={ov['n']}")
    print(f"  CoT     : valid={ov['valid_rate']:.3f}  "
          f"agree={ov['agree_rate']:.3f}  trunc={ov['trunc_rate']:.3f}  "
          f"mean_tokens={ov['mean_gen_tokens']:.1f}")
    if args.n_samples > 1:
        print(f"  SC(k={ov['n_samples']}): vote_acc={ov['vote_accuracy']:.3f}  "
              f"marginal_acc={ov['accuracy']:.3f}  "
              f"vote_ECE={ov.get('vote_ece', float('nan')):.3f}  "
              f"mean_vote_margin={ov['mean_vote_margin']:.3f}  "
              f"vote!=marginal={ov['vote_vs_marginal_disagree']:.3f}")
    for task, s in summary["by_task"].items():
        print(f"  {task:50s}: acc={s['accuracy']:.3f}  NLL={s['ce_mean']:.3f}  "
              f"ECE={s['ece']:.3f}  n={s['n']}")
    # Drop the checkpoint now that the real output exists: a leftover
    # "*_shard0_partial.json" is a prefix of this file, and any glob broad
    # enough to collect the shards also collects it.
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
    print(f"\nResults saved to {args.output_file}")


if __name__ == "__main__":
    main()
