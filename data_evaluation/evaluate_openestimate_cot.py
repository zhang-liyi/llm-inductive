"""
evaluate_openestimate_cot.py

Chain-of-thought variant of ``evaluate_openestimate.py``.

The model freely reasons about the estimation query, then the answer
distribution is read off a restricted softmax over the integers 0-100 at a
canonical ``"The answer is: <"`` cue appended after its reasoning:

    assistant: <freely generated reasoning …>
               The answer is: <?      ← 101-way softmax → CE_mean, CE_dist, MAE
                            <mean>> <?  ← std slot, conditioned on the model's
                                          own modal mean

This upgrades the existing ``--mode generate --free_gen`` path, which can
only report MAE from a parsed integer (CE is NaN there), to the full
distributional metric set used by the teacher-forced rows.

Note on ``mae_std``: the teacher-forced eval reads the std slot after the
*ground-truth* mean has been forced into the sequence.  There is no ground
truth in a generation setting, so the std slot here is conditioned on the
model's own modal mean.  ``mae`` / ``ce_mean`` / ``ce_dist`` are unaffected
and remain directly comparable.

Multi-token tokenizers (Qwen-2, where 10..100 are not single tokens) fall
back to parse-only scoring: MAE from the generated text, CE = NaN.

Usage
-----
    python evaluate_openestimate_cot.py \\
        --pretrained \\
        --output_file results/openestimate/cot_base_openestimate.json
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

from evaluate_openestimate import (  # noqa: E402
    DEFAULT_MODEL_PATH, aggregate_metrics, build_chat_prefix,
    get_number_token_ids, load_data, load_model_and_tokenizer,
    load_pretrained_model_and_tokenizer, metrics_at_position,
    rewrite_prompt_free_gen,
)
from qwen2_eval_loaders import setup_number_tokens  # noqa: E402
from cot_eval_utils import (  # noqa: E402
    ANSWER_CUE, SC_TEMPERATURE, SC_TOP_K, FailureTracker, build_scoring_context,
    eos_token_ids, free_generate, last_position_logits, marginalize,
    restricted_probs, sc_seed, setup_generation_caches, strip_self_answer,
)


_COT_DIRECTIVE = (
    "Think step by step: reason about what you know that bears on the "
    "quantity, and about how uncertain you are, before committing to a "
    "number. Finish with your conclusion on a new line in the form "
    "The answer is: <mean> <std>, using the 0-100 scale given above."
)

# Bracketed integers in the generation, e.g. "<42>" — used for parsing.
_SELF_ANSWER_RE = re.compile(r"<\s*([0-9]{1,3}(?:\.[0-9]+)?)\s*>")
# The model's own final answer, which is a *pair* "<mean> <std>" (or the
# malformed single-bracket variant "<mean, std>") — cut off as one unit
# before the canonical cue is appended.
_NUM = r"[0-9]{1,3}(?:\.[0-9]+)?"
_STRIP_RE = re.compile(
    rf"<\s*{_NUM}\s*>(?:\s*<\s*{_NUM}\s*>)?"
    rf"|<\s*{_NUM}[,\s]+{_NUM}\s*>"
)

_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16,
           "float32": torch.float32}


def rewrite_prompt_cot(prompt: str) -> str:
    """Relax the answer-only instruction, then append an explicit CoT directive."""
    return rewrite_prompt_free_gen(prompt).rstrip() + "\n\n" + _COT_DIRECTIVE


def _pair_from_unit(unit: str):
    """Parse the ``<mean> <std>`` / ``<mean, std>`` unit matched by _STRIP_RE.

    Returns ``(mean, std)`` with std possibly None, or ``(None, None)`` if a
    value falls outside the 0-100 scale (e.g. a dollar amount the model
    happened to bracket)."""
    nums = [float(x) for x in re.findall(_NUM, unit)]
    if not nums or any(not 0.0 <= v <= 100.0 for v in nums[:2]):
        return None, None
    m_val = int(round(nums[0]))
    s_val = int(round(nums[1])) if len(nums) > 1 else None
    return m_val, s_val


def parse_mean_std(text: str, tail_window: int = 80):
    """The ``<mean> <std>`` answer the model committed to.

    Prefers a well-formed answer *unit* near the end of the generation — the
    same span ``strip_self_answer`` cuts off — because a bare last-two-brackets
    scan mis-reads reasoning that brackets an intermediate figure and then
    gives only a mean ("... <50> is the midpoint ... The answer is: <62>"
    would otherwise parse as mean=50, std=62).  Falls back to the scan used by
    ``evaluate_openestimate.run_generate_eval`` when no trailing unit exists.
    """
    units = list(_STRIP_RE.finditer(text))
    if units and len(text.rstrip()) - units[-1].end() <= tail_window:
        m_val, s_val = _pair_from_unit(units[-1].group(0))
        if m_val is not None:
            return m_val, s_val

    vals = [float(x) for x in _SELF_ANSWER_RE.findall(text)
            if 0.0 <= float(x) <= 100.0]
    if len(vals) >= 2:
        return int(round(vals[-2])), int(round(vals[-1]))
    if len(vals) == 1:
        return int(round(vals[-1])), None
    pair = re.findall(
        r"<\s*([0-9]+(?:\.[0-9]+)?)[,\s]+([0-9]+(?:\.[0-9]+)?)\s*>", text)
    if pair:
        m_val, s_val = (int(round(float(v))) for v in pair[-1])
        return (m_val if 0 <= m_val <= 100 else None,
                s_val if 0 <= s_val <= 100 else None)
    plain = [int(x) for x in re.findall(r"\b([0-9]{1,3})\b", text)
             if 0 <= int(x) <= 100]
    if len(plain) >= 2:
        return plain[-2], plain[-1]
    if plain:
        return plain[-1], None
    return None, None



def metrics_from_probs(pred_probs: np.ndarray, pred_std: float,
                       gt_bins: List[float], gt_std: float) -> dict:
    """Same quantities as ``evaluate_openestimate.metrics_at_position``, but
    computed from an already-normalised distribution over 0-100.

    Self-consistency marginalises over reasoning paths by *averaging* the
    per-path softmaxes; that mixture is not the softmax of any single logit
    vector, so the metrics have to be computed from probabilities directly.
    With one path this reduces exactly to ``metrics_at_position``.
    """
    gt = np.asarray(gt_bins, dtype=np.float64)
    values = np.arange(101, dtype=np.float64)
    gt_mean = float(np.dot(values, gt))
    gt_mean_int = max(0, min(100, int(round(gt_mean))))
    eps = 1e-10
    pred_mean = float(np.dot(values, pred_probs))
    return {
        "ce_mean": float(-np.log(pred_probs[gt_mean_int] + eps)),
        "ce_dist": float(-np.sum(gt * np.log(pred_probs + eps))),
        "mae": abs(pred_mean - gt_mean),
        "mae_std": abs(float(pred_std) - float(gt_std)),
        "pred_mean": pred_mean,
        "gt_mean": gt_mean,
        "pred_std": float(pred_std),
        "gt_std": float(gt_std),
        "pred_mode": int(np.argmax(pred_probs)),
        "gt_mode": int(np.argmax(gt)),
        "pred_dist": pred_probs.tolist(),
    }


@torch.no_grad()
def eval_one(
    model, tokenizer, ex: dict, number_token_ids: torch.Tensor, device: str,
    caches_on: bool, max_seq_len: int, max_new_tokens: int,
    eos_ids: List[int], temperature: float, top_k: Optional[int],
    single_token: bool,
    n_samples: int = 1, global_idx: int = 0, base_seed: int = 0,
    sample_offset: int = 0,
) -> dict:
    gt_bins = ex["bins"][0]
    gt_mean = float(sum(j * gt_bins[j] for j in range(101)))
    gt_std = float(ex.get("metadata", {}).get("normalised_std", 0.0))

    prompt = rewrite_prompt_cot(ex["input"])
    prefix_text = build_chat_prefix(prompt, tokenizer)
    prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
    budget = max_seq_len - max_new_tokens - 16
    if len(prefix_ids) > budget:
        prefix_ids = prefix_ids[-budget:]

    # Each sampled reasoning path yields its own 101-way distribution over the
    # mean slot; self-consistency marginalises over paths by averaging them.
    # There is no majority vote here because the answer is continuous — the
    # mixture *is* the marginal, and its spread across paths is a usable
    # uncertainty signal in its own right.
    reasonings, gen_texts, parsed_means, parsed_stds = [], [], [], []
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
        reasoning, self_answer_k = strip_self_answer(gen_text, _STRIP_RE)
        pm, ps = parse_mean_std(gen_text)
        reasonings.append(reasoning)
        gen_texts.append(gen_text)
        parsed_means.append(pm)
        parsed_stds.append(ps)
        n_tok.append(len(gen_ids))
        n_trunc += int(len(gen_ids) >= max_new_tokens)
        if k == 0:
            self_answer = self_answer_k

    gen_text = gen_texts[0]
    parsed_mean, parsed_std = parsed_means[0], parsed_stds[0]

    common = {
        "mode": "cot" if n_samples == 1 else "cot_sc",
        "global_idx": global_idx,
        "sample_idx": sample_offset,
        "generated_text": gen_text,
        "self_answer": self_answer,
        "parsed_mean": parsed_mean,
        "parsed_std": parsed_std,
        "valid": parsed_mean is not None,
        "n_gen_tokens": float(np.mean(n_tok)),
        "truncated": n_trunc == n_samples,
        "metadata": ex.get("metadata", {}),
    }
    if n_samples > 1:
        pm_valid = [float(v) for v in parsed_means if v is not None]
        common.update({
            "n_samples": n_samples,
            "sample_parsed_means": parsed_means,
            # Disagreement between reasoning paths about the quantity: a
            # self-consistency-derived uncertainty estimate.
            "parsed_mean_spread": (float(np.std(pm_valid))
                                   if len(pm_valid) > 1 else float("nan")),
            "trunc_frac": n_trunc / n_samples,
        })

    if not single_token:
        # Multi-token vocabulary: no single-position 101-way softmax exists.
        # Fall back to parse-only MAE, matching run_generate_eval's contract.
        pred_mean = float(parsed_mean) if parsed_mean is not None else float("nan")
        pred_std = float(parsed_std) if parsed_std is not None else float("nan")
        common.update({
            "ce_mean": float("nan"), "ce_dist": float("nan"),
            "mae": abs(pred_mean - gt_mean) if parsed_mean is not None else float("nan"),
            "mae_std": abs(pred_std - gt_std) if parsed_std is not None else float("nan"),
            "pred_mean": pred_mean, "gt_mean": gt_mean,
            "pred_std": pred_std, "gt_std": gt_std,
            "pred_mode": parsed_mean, "gt_mode": int(np.argmax(gt_bins)),
            "pred_dist": [],
        })
        return common

    nt_list = number_token_ids.tolist()
    per_mean_probs, per_std_pred = [], []
    for reasoning in reasonings:
        # ── mean slot ────────────────────────────────────────────────────────
        ctx_ids = build_scoring_context(tokenizer, prefix_ids, reasoning,
                                        ANSWER_CUE)
        mean_logits = last_position_logits(model, ctx_ids, device, max_seq_len)
        probs_k = restricted_probs(mean_logits, nt_list)

        # ── std slot: continue past this path's own modal mean ───────────────
        pred_mode_k = int(np.argmax(probs_k))
        std_ctx = list(ctx_ids) + tokenizer.encode(
            f"{pred_mode_k}> <", add_special_tokens=False)
        std_logits = last_position_logits(model, std_ctx, device, max_seq_len)
        std_probs_k = restricted_probs(std_logits, nt_list)

        per_mean_probs.append(probs_k)
        per_std_pred.append(float(np.argmax(std_probs_k)))

    probs_mean = marginalize(per_mean_probs)
    metrics = metrics_from_probs(
        probs_mean, float(np.mean(per_std_pred)), gt_bins, gt_std)
    metrics.update(common)
    return metrics


def main():
    ap = argparse.ArgumentParser(
        description="Chain-of-thought OpenEstimate evaluation.")
    ap.add_argument("--ckpt_dir", default=None)
    ap.add_argument("--pretrained", action="store_true")
    ap.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    ap.add_argument("--data_path",
        default="/home/aj9225/llm-inductive/data_processing/"
                "openestimate_test.json")
    ap.add_argument("--split", choices=["dev", "test", "all"], default="all")
    ap.add_argument("--start_idx", type=int, default=0)
    ap.add_argument("--n_examples", type=int, default=None)
    ap.add_argument("--sample_idx", type=int, default=None,
                    help="Run exactly ONE reasoning path — path j of a "
                         "k-way self-consistency run — so the k paths can "
                         "be spread over k separate jobs and recombined by "
                         "combine_sc_samples.py. Seeds match what an "
                         "in-process --n_samples k run would have drawn.")
    ap.add_argument("--n_samples", type=int, default=1,
                    help="Self-consistency: sample this many reasoning paths "
                         "per example and average their 0-100 distributions.")
    ap.add_argument("--sc_defaults", action="store_true",
                    help="Use Wang et al. self-consistency decoding settings "
                         f"(temperature={SC_TEMPERATURE}, top_k={SC_TOP_K}).")
    ap.add_argument("--max_seq_len", type=int, default=2048)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=1,
                    help="1 = greedy decoding (default). >1 samples.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="bfloat16", choices=list(_DTYPES))
    ap.add_argument("--no_kv_cache", action="store_true")
    ap.add_argument("--output_file", required=True)
    ap.add_argument("--progress_every", type=int, default=10)
    ap.add_argument("--checkpoint_every", type=int, default=50)
    args = ap.parse_args()

    if not args.pretrained and args.ckpt_dir is None:
        ap.error("Provide --ckpt_dir or --pretrained.")
    if args.sc_defaults:
        args.temperature, args.top_k = SC_TEMPERATURE, SC_TOP_K
    if (args.n_samples > 1 or args.sample_idx is not None) and args.top_k == 1:
        # top_k=1 collapses sampling onto the argmax: k identical paths.
        ap.error("Sampled decoding is required for --n_samples > 1 or "
                 "--sample_idx: pass --sc_defaults, or set --top_k > 1.")
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)

    data = load_data(args.data_path, split=args.split, n_examples=None)
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

    nt = setup_number_tokens(tokenizer)
    single_token = bool(nt["single_token"])
    if single_token:
        number_token_ids = get_number_token_ids(tokenizer)
    else:
        number_token_ids = torch.tensor(nt["number_token_ids"], dtype=torch.long)
        print("[WARNING] Multi-token tokenizer detected — CoT OE eval falls "
              "back to parse-only MAE (CE will be NaN).")
    print(f"Number token IDs (sample 0-5): {number_token_ids[:6].tolist()}")

    eos_ids = eos_token_ids(tokenizer)
    print(f"Stop token ids: {eos_ids}")

    caches_on = False
    if not args.no_kv_cache:
        caches_on = setup_generation_caches(
            model, _DTYPES[args.dtype], args.max_seq_len + 32)
    print(f"KV-cached generation: {caches_on}")
    if args.top_k and args.top_k > 1:
        torch.manual_seed(args.seed)

    results: List[dict] = []
    failures = FailureTracker()
    ckpt_path = args.output_file.replace(".json", "_partial.json")
    for i, ex in enumerate(data):
        try:
            r = eval_one(
                model, tokenizer, ex, number_token_ids, args.device, caches_on,
                args.max_seq_len, args.max_new_tokens, eos_ids,
                args.temperature, args.top_k, single_token,
                n_samples=args.n_samples,
                global_idx=args.start_idx + i, base_seed=args.seed,
                sample_offset=args.sample_idx or 0,
            )
        except Exception as exc:
            failures.record(i, exc, len(results))
            continue
        results.append(r)

        if (i + 1) % args.progress_every == 0:
            maes = [r["mae"] for r in results
                    if not (isinstance(r["mae"], float) and np.isnan(r["mae"]))]
            print(f"  {i + 1}/{len(data)}  MAE={np.mean(maes):.3f} "
                  f"n={len(results)}", flush=True)
        if (i + 1) % args.checkpoint_every == 0:
            tmp = ckpt_path + ".tmp"
            with open(tmp, "w") as fh:
                json.dump({"progress": i + 1, "total": len(data),
                           "summary": aggregate_metrics(results),
                           "per_example": results}, fh)
            os.replace(tmp, ckpt_path)

    if not results:
        print("No results.")
        sys.exit(1)

    summary = aggregate_metrics(results)
    summary["valid_rate"] = float(np.mean([r["valid"] for r in results]))
    summary["trunc_rate"] = float(
        np.mean([bool(r.get("truncated")) for r in results]))
    summary["mean_gen_tokens"] = float(
        np.mean([r["n_gen_tokens"] for r in results]))

    print("\n=== OpenEstimate Chain-of-Thought Summary ===")
    print(f"  Split      : {args.split}  ({len(results)} examples)")
    print(f"  MAE (mean) : {summary['mae']['mean']:.3f} ± "
          f"{summary['mae']['std']:.3f}  (median {summary['mae']['median']:.3f})")
    print(f"  MAE (std)  : {summary['mae_std']['mean']:.3f} ± "
          f"{summary['mae_std']['std']:.3f}")
    print(f"  CE_mean    : {summary['ce_mean']['mean']:.3f}")
    print(f"  CE_dist    : {summary['ce_dist']['mean']:.3f}")
    print(f"  CoT        : valid={summary['valid_rate']:.3f}  "
          f"trunc={summary['trunc_rate']:.3f}  "
          f"mean_tokens={summary['mean_gen_tokens']:.1f}")
    print("  MAE by dataset:")
    for ds, s in summary["by_dataset"].items():
        print(f"    {ds:<12} {s['mean']:.3f}  (n={s['n']})")

    output = {
        "ckpt_dir": args.ckpt_dir or args.model_path,
        "data_path": args.data_path,
        "split": args.split,
        "mode": "cot" if args.n_samples == 1 else "cot_sc",
        "n_samples": args.n_samples,
        "sample_idx": args.sample_idx,
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
