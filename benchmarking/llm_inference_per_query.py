"""Run a fine-tuned LLM (e.g. pyro_lora_dist) on the e2_implicit benchmarking
scenarios, one query per prompt, mirroring the SFT validation loop:

  prompt   = INSTRUCTION + "Here is the scenario:\\n\\n<START_SCENARIO>\\n
             {single-query scenario}\\n<END_SCENARIO>"
  response = "<X>"  (bracket mode, default) or "X" (--no_bracket_format)

The full chat sequence is forwarded once.  At the position where the model
predicts the integer digit (between '<' and '>'), we restrict the vocab logits
to the 101 number-token IDs (0..100), softmax, and take that as the model's
posterior over the answer scale.  This is the same teacher-forced extraction
used by ``probabilistic_reasoning_utils.extract_number_probabilities`` and
``custom_lora_trajectory._compute_prob_metrics``.

Output: ``benchmarking/inference_results/llm-{ckpt_tag}.json``
    {
      scenario_id: {
        "query1": {"llm_dist": [101 floats], "llm_mean": float, "llm_mode": int},
        ...
      },
      ...
    }
"""

import argparse
import json
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

# Reuse the model / tokenizer / number-token helpers from the OpenEstimate eval.
sys.path.insert(0, './data_evaluation')
from evaluate_openestimate import (
    DEFAULT_MODEL_PATH,
    build_chat_prefix,
    get_number_token_ids,
    load_model_and_tokenizer,
    load_pretrained_model_and_tokenizer,
)

BENCH_DIR = './benchmarking'
SCENARIOS_DIR = f'{BENCH_DIR}/scenarios'
RESULTS_DIR = f'{BENCH_DIR}/inference_results'
HUMAN_DATA = f'{BENCH_DIR}/msa_cogsci_human_data.json'

# Two SFT response formats coexist in this project:
#   bracket  → response wrapped as "<N>" (prob_reasoning_r8_dist_seed1,
#              pyrorej_all_*_bracket family, sports_bracket, etc.)
#   plain    → response is a bare "N"   (pre-bracket pyro_lora_dist / pyro_rej
#              sports-non-bracket family)
# The eval instruction and the teacher-forced answer tokens must match the
# format the checkpoint was fine-tuned on — otherwise we read logits off a
# position the model has never seen during training.
BRACKET_INSTRUCTION = (
    "Answer the query in the scenario and return only an integer wrapped in "
    "< and >. For example, <x>. Use 0-100 scale. For a query on individual "
    "rank, a higher number means a higher ranking (e.g. 100 means the "
    "individual ranks highest in that criterion; 1 is lowest). For a query "
    "on which of the two teams wins, a smaller number means the first team "
    "more likely wins."
)
PLAIN_INSTRUCTION = (
    "Answer the query in the scenario and return only an integer. Use 0-100 "
    "scale. For a query on individual rank or performance, a higher number "
    "means more strength (e.g. 100 is stronger than 1). For a query on which "
    "team wins, a smaller number means the first team more likely wins."
)
DEFAULT_INSTRUCTION = BRACKET_INSTRUCTION  # back-compat default

# Placeholder integer used as the teacher-forced response (any single number
# token in 0-100 works; the digit position is what we read off, not its value).
PLACEHOLDER_INT = 0


def default_instruction_for(bracket_format: bool) -> str:
    return BRACKET_INSTRUCTION if bracket_format else PLAIN_INSTRUCTION


def build_answer_text(bracket_format: bool) -> str:
    return f"<{PLACEHOLDER_INT}>" if bracket_format else f"{PLACEHOLDER_INT}"


# ── scenario parsing ──────────────────────────────────────────────────────────

def split_queries(scenario_text: str) -> Tuple[str, List[Tuple[str, str]]]:
    """Split a scenario file into (header, [(query_key, query_text), ...]).

    `header` is everything up to (and including) the line "QUERIES".
    Each query entry is "Query N: <text>" — we keep the text part only and use
    "queryN" as the key (matching the human-data convention).
    """
    if 'QUERIES' not in scenario_text:
        raise ValueError("Scenario missing QUERIES section")
    header, queries_block = scenario_text.split('QUERIES', 1)
    # Preserve the blank line between CONDITIONS and QUERIES (training format).
    header = header.rstrip() + '\n\nQUERIES'

    pattern = re.compile(r'Query\s+(\d+)\s*:\s*(.+?)(?=(?:\n\s*Query\s+\d+\s*:)|\Z)',
                         re.DOTALL)
    out = []
    for m in pattern.finditer(queries_block):
        idx = int(m.group(1))
        text = m.group(2).strip()
        out.append((f'query{idx}', text))
    if not out:
        raise ValueError("No queries parsed from scenario")
    return header, out


def build_single_query_input(header: str, query_text: str, instruction: str) -> str:
    """Construct the user prompt for a single-query scenario.

    Mirrors the format in data/probabilistic_reasoning_train.json: instruction,
    blank line, "Here is the scenario:", then the scenario wrapped in
    <START_SCENARIO> ... <END_SCENARIO> with QUERIES containing exactly one
    "Query: ..." line (no number prefix, matching the training data).
    """
    body = f"{header}\nQuery: {query_text}"
    return (
        f"{instruction}\n\n"
        f"Here is the scenario:\n\n"
        f"<START_SCENARIO>\n{body}\n<END_SCENARIO>"
    )


# ── teacher-forced inference for a single (scenario, query) ──────────────────

def find_first_number_position(input_ids: List[int], prefix_len: int,
                                number_token_set: set, tokenizer) -> int:
    """Position of the first numeric (0-100) token at or after prefix_len."""
    for pos in range(prefix_len, len(input_ids)):
        tok = input_ids[pos]
        if tok in number_token_set:
            return pos
        try:
            decoded = tokenizer.decode([tok]).strip()
            if decoded.isdigit() and 0 <= int(decoded) <= 100:
                return pos
        except Exception:
            pass
    return -1


@torch.no_grad()
def predict_distribution(model, tokenizer, prompt: str,
                          number_token_ids: torch.Tensor,
                          number_token_set: set, device: str,
                          max_seq_len: int,
                          bracket_format: bool = True) -> Optional[np.ndarray]:
    """Return the 101-way softmax over number tokens at the answer position."""
    prefix_text = build_chat_prefix(prompt, tokenizer)
    answer_text = build_answer_text(bracket_format)
    full_text = prefix_text + answer_text

    prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
    full_ids = tokenizer.encode(full_text, add_special_tokens=False)

    if len(full_ids) > max_seq_len:
        # Truncate the front of the prompt (keep the tail with the query+answer).
        drop = len(full_ids) - max_seq_len
        full_ids = full_ids[drop:]
        prefix_ids = prefix_ids[drop:] if len(prefix_ids) > drop else []

    pos = find_first_number_position(full_ids, len(prefix_ids),
                                     number_token_set, tokenizer)
    if pos <= 0:
        return None

    input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
    logits = model(tokens=input_ids)
    if isinstance(logits, list):
        logits = torch.cat(logits, dim=1)
    # Logits at (pos - 1) predict the token at `pos`.
    number_logits = logits[0, pos - 1, number_token_ids.to(device)].float()
    probs = F.softmax(number_logits, dim=0).cpu().numpy().astype(np.float64)
    return probs


# ── main loop ────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ckpt_dir', default=None,
                   help="Path to torchtune epoch checkpoint dir "
                        "(e.g. ckpt/llama3_8B/<run>_lora8_dist/epoch_0).")
    p.add_argument('--pretrained', action='store_true',
                   help="Evaluate the base Llama-3-8B-Instruct instead.")
    p.add_argument('--model_path', default=DEFAULT_MODEL_PATH,
                   help="Base HF model dir (used with --pretrained).")
    p.add_argument('--instruction', default=None,
                   help="Instruction prefix prepended to each prompt. "
                        "Defaults to BRACKET_INSTRUCTION if --bracket_format, "
                        "else PLAIN_INSTRUCTION.")
    p.add_argument('--bracket_format', dest='bracket_format',
                   action='store_true', default=True,
                   help="Teacher-force answer as '<N>' and use the bracket "
                        "instruction (default; matches pyro_rej_*_bracket / "
                        "prob_reasoning bracket-trained ckpts).")
    p.add_argument('--no_bracket_format', dest='bracket_format',
                   action='store_false',
                   help="Teacher-force answer as bare 'N' and use the plain "
                        "instruction (for pre-bracket / sports-non-bracket "
                        "ckpts that emit integers without angle brackets).")
    p.add_argument('--max_seq_len', type=int, default=2048)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--dtype', default='bfloat16',
                   choices=['bfloat16', 'float16', 'float32'])
    p.add_argument('--output_file', default=None,
                   help="Output JSON path. Default: "
                        "inference_results/llm-{ckpt_tag}.json")
    p.add_argument('--limit_scenarios', type=int, default=None,
                   help="For quick smoke tests: only run the first N scenarios.")
    args = p.parse_args()

    if not args.pretrained and args.ckpt_dir is None:
        p.error("Provide --ckpt_dir or --pretrained.")

    if args.instruction is None:
        args.instruction = default_instruction_for(args.bracket_format)

    # Output path
    if args.output_file is None:
        if args.pretrained:
            tag = 'pretrained'
        else:
            # ckpt_dir typically ends in .../<run_name>/epoch_N
            parts = os.path.normpath(args.ckpt_dir).split(os.sep)
            tag = parts[-2] if len(parts) >= 2 else 'ft'
        os.makedirs(RESULTS_DIR, exist_ok=True)
        args.output_file = os.path.join(RESULTS_DIR, f'llm-{tag}.json')

    # Load model
    if args.pretrained:
        model, tokenizer = load_pretrained_model_and_tokenizer(
            args.model_path, args.device, args.dtype)
    else:
        model, tokenizer = load_model_and_tokenizer(
            args.ckpt_dir, args.device, args.dtype)

    number_token_ids = get_number_token_ids(tokenizer)
    number_token_set = set(number_token_ids.tolist())
    print(f"Number token IDs (0-5): {number_token_ids[:6].tolist()}")

    # Load human data: drives the scenario set and the query keys we report.
    with open(HUMAN_DATA) as f:
        human = json.load(f)
    e2 = human['e2_implicit']
    scenario_ids = list(e2.keys())
    if args.limit_scenarios is not None:
        scenario_ids = scenario_ids[:args.limit_scenarios]

    results: Dict[str, Dict[str, dict]] = {}
    n_pairs = 0
    for s_idx, sid in enumerate(scenario_ids, 1):
        sc_path = os.path.join(SCENARIOS_DIR, f'{sid}.txt')
        if not os.path.isfile(sc_path):
            print(f"  [WARN] missing scenario file: {sc_path}")
            continue
        with open(sc_path) as f:
            scenario_text = f.read()
        try:
            header, queries = split_queries(scenario_text)
        except ValueError as e:
            print(f"  [WARN] {sid}: {e}")
            continue

        # Restrict to queries the human data has (avoids extras/typos).
        human_query_keys = set(k for r in e2[sid] for k in r.keys())
        scenario_results: Dict[str, dict] = {}
        for q_key, q_text in queries:
            if q_key not in human_query_keys:
                continue
            prompt = build_single_query_input(header, q_text, args.instruction)
            probs = predict_distribution(
                model, tokenizer, prompt, number_token_ids,
                number_token_set, args.device, args.max_seq_len,
                bracket_format=args.bracket_format)
            if probs is None:
                print(f"  [WARN] {sid}/{q_key}: failed to locate answer position")
                continue
            mean_val = float(np.dot(np.arange(101), probs))
            mode_val = int(np.argmax(probs))
            scenario_results[q_key] = {
                'llm_dist': probs.tolist(),
                'llm_mean': mean_val,
                'llm_mode': mode_val,
            }
            n_pairs += 1
        results[sid] = scenario_results
        print(f"  [{s_idx}/{len(scenario_ids)}] {sid}: "
              f"{len(scenario_results)} queries done")

    output = {
        'ckpt_dir': args.ckpt_dir,
        'pretrained': args.pretrained,
        'instruction': args.instruction,
        'bracket_format': args.bracket_format,
        'placeholder_int': PLACEHOLDER_INT,
        'n_scenario_query_pairs': n_pairs,
        'results': results,
    }
    with open(args.output_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nWrote {args.output_file}  ({n_pairs} scenario-query pairs).")


if __name__ == '__main__':
    main()
