"""Direct scenario -> Gemini -> bracketed-integer answers pipeline.

Parallel to the scenario -> program -> answer route: instead of compiling
the scenario into a Pyro program and running rejection sampling, we hand
the full scenario (with all 4 queries) straight to Gemini with the
multi-query bracket-format instruction and parse the list of integers
it returns inside `[<a>, <b>, <c>, <d>]`.

The instruction is verbatim from
`torchtune/data/transform_forward_sampling.py:OLD_HEADER`, which is the
prompt used for the existing `forward_sampling_dataset_*.json` training
data. So one Gemini call per scenario, four answers per call.

Default behavior (zero-arg run): scan every `*.txt` scenario in `scenarios/`
(top-level only), emit one `*.json` answer file per scenario into
`gemini_direct_answers/`. Skips scenarios whose output file already
exists, so you can re-run to fill gaps.

Self-contained:
  - Single dependency: `google-genai` (`pip install google-genai`).
  - Stdlib only otherwise; no project imports.
  - Async with a 32-way semaphore, mirroring `msa_get_program_part1_async.py`.

Per-scenario output shape:
  {
    "scenario_id":  "diverse-P-0-C-0-R-0-N-0-0",
    "scenario_path": "scenarios/gemini-diverse-P-0-C-0-R-0-N-0-0.txt",
    "queries":       ["Out of 100 random athletes, ...", ...],
    "gemini_prompt": "Answer the queries...",
    "gemini_raw":    "[<42>, <100>, <29>, <71>]",
    "gemini_answers": [42, 100, 29, 71]
  }

Usage:
    export GEMINI_API_KEY=...
    python gemini_direct_answer.py                         # scan scenarios/
    python gemini_direct_answer.py --scenarios_dir foo/    # custom dir
    python gemini_direct_answer.py --limit 4               # smoke test
"""
import argparse
import asyncio
import glob
import json
import os
import re
import sys

from google import genai
from google.genai import types


DEFAULT_MODEL = "gemini-3-pro-preview"
MAX_CONCURRENT = 32  # semaphore limit for parallel Gemini calls

# Lazy client init so importing the module (and unit-testing the parser/
# prompt-builder helpers) doesn't require a GEMINI_API_KEY.
_CLIENT = None


def _get_client():
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = genai.Client(http_options=types.HttpOptions(timeout=900_000))
    return _CLIENT


# ── 4-answer bracket-format instruction (verbatim from
#    torchtune/data/transform_forward_sampling.py:OLD_HEADER, used to build
#    the `forward_sampling_dataset_*.json` training data) ────────────────────
BRACKET_INSTRUCTION = (
    "Answer the queries in the scenario and return only a list of integers, "
    "each wrapped in < and >. For example, an output can be: "
    "[<mean1>, <mean2>]. For queries on individual rank, a higher number means "
    "a higher ranking (e.g. 100 means the individual ranks highest in that "
    "criterion; 1 is lowest). For queries on which of the two teams wins, a "
    "smaller number means the first team more likely wins."
)


# ── Scenario parsing ────────────────────────────────────────────────────────
_QUERY_LINE_RE = re.compile(r"^\s*Query\s*\d+\s*:\s*(.+?)\s*$")


def parse_scenario_txt(text):
    """Return (prefix_before_queries, [query_nl, ...]) for a scenario .txt
    using the standard `<START_SCENARIO>...QUERIES\\nQuery N: ...\\n<END_SCENARIO>`
    format. The prefix retains BACKGROUND/CONDITIONS up to (but excluding)
    the `QUERIES` header."""
    pre, _, queries_block = text.partition("QUERIES\n")
    queries_block = queries_block.replace("<END_SCENARIO>", "").strip()
    queries = []
    for line in queries_block.splitlines():
        m = _QUERY_LINE_RE.match(line)
        if m:
            queries.append(m.group(1).strip())
    return pre.rstrip(), queries


def scenario_id_from_path(path):
    """`scenarios/gemini-diverse-P-...-{seed}.txt` -> `diverse-P-...-{seed}`."""
    stem = os.path.splitext(os.path.basename(path))[0]
    return stem[len("gemini-"):] if stem.startswith("gemini-") else stem


# ── Prompt construction ─────────────────────────────────────────────────────
def build_prompt(scenario_text):
    """Prepend BRACKET_INSTRUCTION to the full scenario (which already
    contains BACKGROUND/CONDITIONS + the `Query 1: ... Query 4: ...`
    block ending in `<END_SCENARIO>`)."""
    return (
        f"{BRACKET_INSTRUCTION}\n\n"
        f"Here is the scenario:\n\n{scenario_text.strip()}"
    )


# ── Response parsing ────────────────────────────────────────────────────────
_BRACKET_RE = re.compile(r"<\s*(-?\d+)\s*>")


def parse_brackets(text, n_expected=None):
    """Extract every `<N>` integer (clamped to [0, 100]) from `text`,
    in order. If `n_expected` is given, returns exactly that many entries,
    padding with `None` if too few were found and truncating if too many.
    Returns None on totally malformed input."""
    if text is None:
        return None
    matches = _BRACKET_RE.findall(text)
    nums = []
    for m in matches:
        try:
            nums.append(max(0, min(100, int(m))))
        except ValueError:
            nums.append(None)
    if n_expected is None:
        return nums
    if len(nums) >= n_expected:
        return nums[:n_expected]
    return nums + [None] * (n_expected - len(nums))


# ── Gemini call ─────────────────────────────────────────────────────────────
async def run_gemini_async(prompt, temperature, max_tokens,
                           system_prompt="You are a helpful assistant."):
    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        temperature=temperature,
        max_output_tokens=int(max_tokens),
    )
    resp = await _get_client().aio.models.generate_content(
        model=DEFAULT_MODEL,
        contents=prompt,
        config=config,
    )
    return resp.text


async def call_with_retries(prompt, semaphore, temperature, max_tokens, retries):
    last_err = None
    async with semaphore:
        for attempt in range(retries):
            try:
                return await run_gemini_async(prompt, temperature, max_tokens), None
            except Exception as e:
                last_err = e
                await asyncio.sleep(2 ** (attempt + 1))
    return None, repr(last_err)


# ── Per-scenario task ───────────────────────────────────────────────────────
async def process_scenario(path, output_dir, semaphore,
                           temperature, max_tokens, retries):
    sid = scenario_id_from_path(path)
    out_path = os.path.join(output_dir, f"gemini-{sid}.json")
    if os.path.isfile(out_path):
        return  # already done; skip silently to keep logs short

    with open(path) as f:
        scenario_text = f.read()
    _prefix, queries = parse_scenario_txt(scenario_text)
    if not queries:
        print(f"[skip] no queries parsed: {path}")
        return

    prompt = build_prompt(scenario_text)
    raw, err = await call_with_retries(
        prompt, semaphore, temperature, max_tokens, retries,
    )
    answers = parse_brackets(raw, n_expected=len(queries))

    record = {
        "scenario_id": sid,
        "scenario_path": path,
        "queries": queries,
        "gemini_prompt": prompt,
        "gemini_raw": raw,
        "gemini_answers": answers,
    }
    if err:
        record["gemini_error"] = err

    os.makedirs(output_dir, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(record, f, indent=2)
    n_ok = sum(1 for a in (answers or []) if a is not None)
    print(f"{sid}: {n_ok}/{len(queries)} parsed -> {out_path}")


# ── Driver ──────────────────────────────────────────────────────────────────
async def main_async(args):
    paths = sorted(glob.glob(os.path.join(args.scenarios_dir, "*.txt")))
    if args.limit is not None:
        paths = paths[: args.limit]
    print(f"Found {len(paths)} scenarios in {args.scenarios_dir}")

    semaphore = asyncio.Semaphore(args.max_concurrent)
    tasks = [
        process_scenario(p, args.output_dir, semaphore,
                         args.temperature, args.max_tokens, args.retries)
        for p in paths
    ]
    await asyncio.gather(*tasks)
    print("All done.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenarios_dir", default="scenarios",
                   help="Directory of scenario *.txt files (default: scenarios/).")
    p.add_argument("--output_dir", default="gemini_direct_answers",
                   help="Where per-scenario answer JSONs go.")
    p.add_argument("--max_concurrent", type=int, default=MAX_CONCURRENT)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max_tokens", type=int, default=256,
                   help="Bracket list `[<a>, <b>, <c>, <d>]` is short.")
    p.add_argument("--retries", type=int, default=4)
    p.add_argument("--limit", type=int, default=None,
                   help="Only process the first N scenarios (smoke test).")
    args = p.parse_args()

    if not os.environ.get("GEMINI_API_KEY"):
        print("WARNING: GEMINI_API_KEY env var is not set.", file=sys.stderr)

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
