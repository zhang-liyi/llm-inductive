"""
evaluate_bayesian_teaching_cot.py

Chain-of-thought variant of ``evaluate_bayesian_teaching.py``.

The model freely reasons about the four observed preference rounds, then the
answer is read off a restricted softmax over {1,2,3} (or {A,B,C} with
``--abc``) at a canonical ``"The answer is: <"`` cue appended after its
reasoning.

This differs from the existing ``--mode generate --free_gen`` path in two
ways that matter for the calibration tables:

  * ``generate`` only yields accuracy + valid_rate; CE and ECE are NaN
    because there is no distribution over the choices.  Here every example
    produces a 3-way distribution, so accuracy / CE / MAE / ECE are all
    defined and directly comparable to the teacher-forced rows.
  * The prompt explicitly asks for step-by-step reasoning rather than merely
    permitting it.

Usage
-----
    python evaluate_bayesian_teaching_cot.py \\
        --pretrained --guided \\
        --data_path ../data_processing/bayesian_teaching_test_base.jsonl \\
        --output_file results/bayesian_teaching/cot_base_bt_base_guided.json
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))

from evaluate_bayesian_teaching import (  # noqa: E402
    DEFAULT_MODEL_PATH, aggregate, build_chat_prefix, get_abc_token_ids,
    get_choice_token_ids, inject_task_instructions, load_data,
    load_model_and_tokenizer, load_pretrained_model_and_tokenizer,
    remap_prompt_to_abc, rewrite_prompt_free_gen,
)
from cot_eval_utils import (  # noqa: E402
    ANSWER_CUE, SC_TEMPERATURE, SC_TOP_K, FailureTracker, build_scoring_context,
    eos_token_ids, free_generate, last_position_logits, majority_vote,
    marginalize, restricted_probs, sc_extras, sc_seed,
    setup_generation_caches, strip_self_answer, vote_distribution,
)


_COT_DIRECTIVE_NUM = (
    "Think step by step: work out from the earlier rounds which features the "
    "user cares about and in which direction, then apply those preferences to "
    "the final round. Finish with your conclusion on a new line in the form "
    "The answer is: <N>, where N is 1, 2, or 3."
)
_COT_DIRECTIVE_ABC = (
    "Think step by step: work out from the earlier rounds which features the "
    "user cares about and in which direction, then apply those preferences to "
    "the final round. Finish with your conclusion on a new line in the form "
    "The answer is: <X>, where X is A, B, or C."
)

_SELF_ANSWER_NUM_RE = re.compile(r"<\s*([123])\s*>")
_SELF_ANSWER_ABC_RE = re.compile(r"<\s*([ABC])\s*>")
_LEAD = r"(?i:(?:the\s+|my\s+)?(?:final\s+)?answer\s*(?:is)?\s*:?\s*\**\s*)"
_BARE_NUM_RE = re.compile(_LEAD + r"([123])\b")
_BARE_ABC_RE = re.compile(_LEAD + r"([ABC])\b")
# What counts as the model's final answer for *stripping*: either the bracketed
# form or the bare lead-in form. Leaving a bare "The answer is: 2" in the
# reasoning would put our cue straight after the model's own answer, giving
# those examples a duplicated question and a different scoring context.
_STRIP_NUM_RE = re.compile(r"<\s*[123]\s*>|" + _LEAD + r"[123]\b")
_STRIP_ABC_RE = re.compile(r"<\s*[ABC]\s*>|" + _LEAD + r"[ABC]\b")

_LETTER_TO_NUM = {"A": 1, "B": 2, "C": 3}
_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16,
           "float32": torch.float32}


def rewrite_prompt_cot(prompt: str, abc: bool) -> str:
    """Relax the answer-only instruction, then append an explicit CoT directive."""
    prompt = rewrite_prompt_free_gen(prompt)
    if abc:
        prompt = remap_prompt_to_abc(prompt)
        # remap_prompt_to_abc only knows the answer-only wording; patch the
        # free-gen wording it leaves behind.
        prompt = prompt.replace(
            "End your answer with the best option choice <1>, <2>, or <3>.",
            "End your answer with the best option choice <A>, <B>, or <C>.",
        ).replace(
            "At the end of your answer, give the number of the best option "
            "wrapped in < and >, like <1> or <2> or <3>.",
            "At the end of your answer, give the letter of the best option "
            "wrapped in < and >, like <A> or <B> or <C>.",
        )
    directive = _COT_DIRECTIVE_ABC if abc else _COT_DIRECTIVE_NUM
    return prompt.rstrip() + "\n\n" + directive


def parse_choice(text: str, abc: bool) -> Optional[int]:
    """Last choice the model committed to in its free generation, 1-indexed.

    Both the bracketed form ``<2>`` and the lead-in form ``the answer is 2``
    count, and whichever appears *latest* wins.  Position rather than format
    decides because BT reasoning routinely refers to "option <1>" mid-trace
    before committing to a different option at the end.
    """
    br_re, bare_re = ((_SELF_ANSWER_ABC_RE, _BARE_ABC_RE) if abc
                      else (_SELF_ANSWER_NUM_RE, _BARE_NUM_RE))
    cands = [(m.start(), m.group(1)) for m in br_re.finditer(text)]
    cands += [(m.start(), m.group(1)) for m in bare_re.finditer(text)]
    if not cands:
        return None
    val = max(cands, key=lambda t: t[0])[1]
    return _LETTER_TO_NUM[val.upper()] if abc else int(val)


@torch.no_grad()
def eval_one(
    model, tokenizer, ex: dict, choice_ids: List[int], device: str,
    caches_on: bool, max_seq_len: int, max_new_tokens: int,
    eos_ids: List[int], guided: bool, abc: bool,
    temperature: float, top_k: Optional[int],
    n_samples: int = 1, global_idx: int = 0, base_seed: int = 0,
    sample_offset: int = 0,
) -> Optional[dict]:
    m = re.search(r"<([123])>", ex["output"])
    if not m:
        return None
    gt = int(m.group(1))

    prompt = ex["input"]
    if guided:
        prompt = inject_task_instructions(prompt, ex.get("task", ""))
    prompt = rewrite_prompt_cot(prompt, abc)

    prefix_text = build_chat_prefix(prompt, tokenizer)
    prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
    budget = max_seq_len - max_new_tokens - 16
    if len(prefix_ids) > budget:
        prefix_ids = prefix_ids[-budget:]

    strip_re = _STRIP_ABC_RE if abc else _STRIP_NUM_RE
    per_probs, per_parsed, per_pred, texts = [], [], [], []
    n_tok, n_trunc = [], 0

    for k in range(n_samples):
        if n_samples > 1 or sample_offset:
            torch.manual_seed(sc_seed(base_seed, global_idx,
                                          sample_offset + k))
        gen_ids = free_generate(
            model, prefix_ids, device, max_new_tokens, eos_ids, caches_on,
            temperature=temperature, top_k=top_k,
        )
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        reasoning, self_answer = strip_self_answer(gen_text, strip_re)
        parsed = parse_choice(gen_text, abc)

        ctx_ids = build_scoring_context(tokenizer, prefix_ids, reasoning,
                                        ANSWER_CUE)
        logit_vec = last_position_logits(model, ctx_ids, device, max_seq_len)
        probs_k = restricted_probs(logit_vec, choice_ids)

        per_probs.append(probs_k)
        per_parsed.append(parsed)
        # A path's answer is its own parsed conclusion; fall back to its scored
        # argmax when the text could not be parsed, so the path still votes.
        per_pred.append((parsed - 1) if parsed is not None
                        else int(np.argmax(probs_k)))
        n_tok.append(len(gen_ids))
        n_trunc += int(len(gen_ids) >= max_new_tokens)
        if k == 0:
            first_text, first_self = gen_text, self_answer

    probs = marginalize(per_probs)
    pred = int(np.argmax(probs)) + 1
    ce = float(-np.log(probs[gt - 1] + 1e-10))
    vote_idx = majority_vote(per_pred, probs)
    vote_pred = None if vote_idx is None else vote_idx + 1
    votes = vote_distribution(per_pred, len(choice_ids))
    parsed_first = per_parsed[0]

    rec = {
        "mode": "cot" if n_samples == 1 else "cot_sc",
        "global_idx": global_idx,
        "sample_idx": sample_offset,
        "task": ex.get("task"),
        "source": ex.get("source"),
        "idx": ex.get("idx"),
        "pred": pred,
        "gt": gt,
        "correct": pred == gt,
        "ce": ce,
        "probs": probs.tolist(),
        "parsed_pred": parsed_first,
        "valid": parsed_first is not None,
        "agrees": (parsed_first == pred) if parsed_first is not None else None,
        "self_answer": first_self,
        "n_gen_tokens": float(np.mean(n_tok)),
        "truncated": n_trunc == n_samples,
        "generated_text": first_text,
        "metadata": ex.get("metadata", {}),
    }
    if n_samples > 1:
        rec.update({
            "n_samples": n_samples,
            "vote_probs": votes,
            "vote_pred": vote_idx,
            "pred_idx_marginal": pred - 1,
            "vote_correct": vote_pred == gt,
            "vote_margin": max(votes) if votes else None,
            "sample_preds": [None if p is None else p + 1 for p in per_pred],
            "trunc_frac": n_trunc / n_samples,
        })
    return rec


def cot_extras(items: List[dict]) -> dict:
    if not items:
        return {}
    valid = [it for it in items if it.get("valid")]
    agree = [it for it in valid if it.get("agrees")]
    return {
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


def summarize(results: List[dict]) -> dict:
    summary = aggregate(results)
    summary["overall"].update(cot_extras(results))
    summary["overall"].update(sc_extras(results))
    summary["overall"].update(_vote_accuracy(results))
    for task in summary["by_task"]:
        sub = [r for r in results if r.get("task") == task]
        summary["by_task"][task].update(cot_extras(sub))
        summary["by_task"][task].update(sc_extras(sub))
        summary["by_task"][task].update(_vote_accuracy(sub))
    return summary


def main():
    ap = argparse.ArgumentParser(
        description="Chain-of-thought Bayesian Teaching evaluation.")
    ap.add_argument("--ckpt_dir", default=None)
    ap.add_argument("--pretrained", action="store_true")
    ap.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    ap.add_argument("--data_path",
        default="/home/aj9225/llm-inductive/data_processing/"
                "bayesian_teaching_test_base.jsonl")
    ap.add_argument("--tasks", nargs="+", default=None,
                    choices=["flight", "hotel", "webshop"])
    ap.add_argument("--start_idx", type=int, default=0)
    ap.add_argument("--n_examples", type=int, default=None)
    ap.add_argument("--shuffle_seed", type=int, default=None)
    ap.add_argument("--guided", action="store_true",
                    help="Inject task-specific feature/preference instructions.")
    ap.add_argument("--abc", action="store_true",
                    help="Rewrite prompts to use A/B/C labels instead of 1/2/3.")
    ap.add_argument("--sample_idx", type=int, default=None,
                    help="Run exactly ONE reasoning path — path j of a "
                         "k-way self-consistency run — so the k paths can "
                         "be spread over k separate jobs and recombined by "
                         "combine_sc_samples.py. Seeds match what an "
                         "in-process --n_samples k run would have drawn.")
    ap.add_argument("--n_samples", type=int, default=1,
                    help="Self-consistency: sample this many reasoning paths "
                         "per example and marginalise over them. >1 implies "
                         "sampled decoding (see --sc_defaults).")
    ap.add_argument("--sc_defaults", action="store_true",
                    help=f"Use the Wang et al. self-consistency decoding "
                         f"settings (temperature={SC_TEMPERATURE}, "
                         f"top_k={SC_TOP_K}).")
    ap.add_argument("--max_seq_len", type=int, default=4096)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=1,
                    help="1 = greedy decoding (default). >1 samples.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="bfloat16", choices=list(_DTYPES))
    ap.add_argument("--no_kv_cache", action="store_true")
    ap.add_argument("--output_file", required=True)
    ap.add_argument("--checkpoint_every", type=int, default=25)
    ap.add_argument("--progress_every", type=int, default=25)
    args = ap.parse_args()

    if not args.pretrained and args.ckpt_dir is None:
        ap.error("Provide --ckpt_dir or --pretrained.")
    if args.sc_defaults:
        args.temperature, args.top_k = SC_TEMPERATURE, SC_TOP_K
    if (args.n_samples > 1 or args.sample_idx is not None) and args.top_k == 1:
        # top_k=1 collapses sampling onto the argmax, so every "sample" would
        # be the identical greedy trace and self-consistency would reduce to
        # plain CoT at k times the cost — with no visible symptom.
        ap.error("Sampled decoding is required for --n_samples > 1 or "
                 "--sample_idx: pass --sc_defaults, or set --top_k > 1.")
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)

    # load_data has no start_idx; slice after loading so shards line up with
    # the deterministic file order.
    data = load_data(args.data_path, tasks=args.tasks, n_examples=None,
                     shuffle_seed=args.shuffle_seed)
    end = args.start_idx + args.n_examples if args.n_examples else None
    data = data[args.start_idx:end]
    print(f"  sliced [{args.start_idx}:{end}] → {len(data)} examples")
    if not data:
        print("No data loaded.")
        sys.exit(1)

    if args.pretrained:
        model, tokenizer = load_pretrained_model_and_tokenizer(
            args.model_path, args.device, args.dtype)
    else:
        model, tokenizer = load_model_and_tokenizer(
            args.ckpt_dir, args.device, args.dtype)

    if args.abc:
        abc_ids = get_abc_token_ids(tokenizer)
        choice_ids = [abc_ids["A"], abc_ids["B"], abc_ids["C"]]
        print(f"ABC token IDs: {abc_ids}")
    else:
        num_ids = get_choice_token_ids(tokenizer)
        choice_ids = [num_ids[1], num_ids[2], num_ids[3]]
        print(f"Choice token IDs: {num_ids}")
    eos_ids = eos_token_ids(tokenizer)
    print(f"Stop token ids: {eos_ids}")

    caches_on = False
    if not args.no_kv_cache:
        caches_on = setup_generation_caches(
            model, _DTYPES[args.dtype], args.max_seq_len + 16)
    print(f"KV-cached generation: {caches_on}")
    if args.top_k and args.top_k > 1:
        torch.manual_seed(args.seed)
    if args.n_samples > 1:
        print(f"Self-consistency: {args.n_samples} paths/example, "
              f"temperature={args.temperature}, top_k={args.top_k}, "
              f"seed={args.seed}")

    results: List[dict] = []
    failures = FailureTracker()
    ckpt_path = args.output_file.replace(".json", "_partial.json")
    for i, ex in enumerate(data):
        try:
            r = eval_one(
                model, tokenizer, ex, choice_ids, args.device, caches_on,
                args.max_seq_len, args.max_new_tokens, eos_ids,
                args.guided, args.abc, args.temperature, args.top_k,
                n_samples=args.n_samples,
                global_idx=args.start_idx + i, base_seed=args.seed,
                sample_offset=args.sample_idx or 0,
            )
        except Exception as exc:
            failures.record(i, exc, len(results))
            continue
        if r is None:
            continue
        results.append(r)

        if (i + 1) % args.progress_every == 0:
            ov = aggregate(results)["overall"]
            print(f"  {i + 1}/{len(data)}  acc={ov['accuracy']:.3f} "
                  f"CE={ov['ce_mean']:.3f}  n={ov['n']}", flush=True)
        if (i + 1) % args.checkpoint_every == 0:
            tmp = ckpt_path + ".tmp"
            with open(tmp, "w") as fh:
                json.dump({"progress": i + 1, "total": len(data),
                           "summary": summarize(results),
                           "per_example": results}, fh)
            os.replace(tmp, ckpt_path)

    if not results:
        print("No results.")
        sys.exit(1)

    summary = summarize(results)
    ov = summary["overall"]
    print("\n=== BT Chain-of-Thought Evaluation Summary ===")
    print(f"  Overall : accuracy={ov['accuracy']:.3f}  CE={ov['ce_mean']:.3f}  "
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
        if s["n"] > 0:
            print(f"  {task:<8}: accuracy={s['accuracy']:.3f}  "
                  f"CE={s['ce_mean']:.3f}  MAE={s['mae']:.3f}  "
                  f"ECE={s['ece']:.3f}  n={s['n']}")

    output = {
        "ckpt_dir": args.ckpt_dir or args.model_path,
        "data_path": args.data_path,
        "tasks": args.tasks,
        "mode": "cot" if args.n_samples == 1 else "cot_sc",
        "n_samples": args.n_samples,
        "sample_idx": args.sample_idx,
        "guided": args.guided,
        "abc": args.abc,
        "start_idx": args.start_idx,
        "n_examples": len(results),
        "max_seq_len": args.max_seq_len,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "summary": summary,
        "per_example": results,
    }
    with open(args.output_file, "w") as fh:
        json.dump(output, fh, indent=2)
    # Drop the checkpoint now that the real output exists: a leftover
    # "*_shard0_partial.json" is a prefix of this file, and any glob broad
    # enough to collect the shards also collects it.
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
    print(f"\nResults saved to {args.output_file}")


if __name__ == "__main__":
    main()
