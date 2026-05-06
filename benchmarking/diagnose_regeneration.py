"""Regenerate a single pyro program while logging Gemini's finish_reason and
token usage at every stage, to diagnose the truncation of
pg-tug-of-war_effort_team-confounded-partner-10-....py.

Usage:
    python diagnose_regeneration.py <scenario_id>
"""

import argparse
import asyncio
import json
import os
import sys

from google import genai
from google.genai import types


PYTORCH_DIR = './posterior_sampling_pytorch'
BENCH_DIR = './benchmarking'

DEFAULT_MODEL = "gemini-3-pro-preview"

os.environ['GEMINI_API_KEY'] = '<GEMINI_API_KEY_PLACEHOLDER>'

client = genai.Client(http_options=types.HttpOptions(timeout=900000))


def read_file(path):
    with open(path) as f:
        return f.read()


async def run_with_diag(stage, prompt, temperature, max_tokens,
                        system_prompt='You are a helpful assistant.'):
    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        temperature=temperature,
        max_output_tokens=int(max_tokens),
    )
    resp = await client.aio.models.generate_content(
        model=DEFAULT_MODEL, contents=prompt, config=config,
    )
    cand = resp.candidates[0] if resp.candidates else None
    fr = getattr(cand, 'finish_reason', None)
    usage = getattr(resp, 'usage_metadata', None)
    text = resp.text or ''
    print(f'[{stage}] finish_reason={fr}  '
          f'usage={usage}  response_chars={len(text)}')
    return text, fr, usage


async def process(scenario_id):
    input_filename = f'{BENCH_DIR}/scenarios/{scenario_id}.txt'
    output_program_filename = f'{BENCH_DIR}/programs/pg-{scenario_id}.py'
    scratchpad_filename = f'{BENCH_DIR}/scratchpads/{scenario_id}.json'
    raw_filename = f'{BENCH_DIR}/logs/raw-part3-{scenario_id}.txt'

    if not os.path.isfile(input_filename):
        print(f'Missing scenario file: {input_filename}')
        return

    if os.path.isfile(output_program_filename):
        backup = output_program_filename + '.broken'
        os.rename(output_program_filename, backup)
        print(f'Backed up existing program to {backup}')

    temperature = 0.0
    max_tokens = 3e5

    scenario_raw = read_file(input_filename).rstrip()
    if '<START_SCENARIO>' not in scenario_raw:
        scenario = f'<START_SCENARIO>\n{scenario_raw}\n<END_SCENARIO>'
    else:
        scenario = scenario_raw

    part1prompt = read_file(f'{PYTORCH_DIR}/pg-orig-part1.txt')
    resp1, _, _ = await run_with_diag(
        'PART I', part1prompt + '\n\n' + scenario, temperature, max_tokens)
    current_program = scenario + '\n\n' + resp1

    part2prompt = read_file(f'{PYTORCH_DIR}/pg-orig-part2.txt')
    resp2, _, _ = await run_with_diag(
        'PART II', part2prompt + '\n\n' + current_program, temperature, max_tokens)
    current_program = current_program + '\n\n' + resp2

    scratchpad_full = None
    concept_trace = None
    if '<START_SCRATCHPAD>' in resp2 and '<END_SCRATCHPAD>' in resp2:
        scratchpad_full = resp2.split('<START_SCRATCHPAD>')[1] \
                               .split('<END_SCRATCHPAD>')[0].strip()
    if scratchpad_full and '<START_CONCEPT_TRACE>' in scratchpad_full \
            and '<END_CONCEPT_TRACE>' in scratchpad_full:
        concept_trace = scratchpad_full.split('<START_CONCEPT_TRACE>')[1] \
                                       .split('<END_CONCEPT_TRACE>')[0].strip()
    with open(scratchpad_filename, 'w') as f:
        json.dump({
            'scenario_id': scenario_id,
            'scratchpad': scratchpad_full,
            'concept_trace': concept_trace,
        }, f, indent=2)

    part3prompt = read_file(f'{PYTORCH_DIR}/pg-orig-part3.txt')
    system_prompt = read_file(f'{PYTORCH_DIR}/pg-system-prompt.txt')
    resp3, fr3, usage3 = await run_with_diag(
        'PART III', part3prompt + '\n\n' + current_program,
        temperature, max_tokens, system_prompt)

    os.makedirs(os.path.dirname(raw_filename), exist_ok=True)
    with open(raw_filename, 'w') as f:
        f.write(resp3)
    print(f'Wrote raw PART III response to {raw_filename}')

    if '<START_PYRO_MODEL>' in resp3 and '<END_PYRO_MODEL>' in resp3:
        pyro_program = resp3.split('<START_PYRO_MODEL>\n')[1] \
                            .split('\n<END_PYRO_MODEL>')[0]
        with open(output_program_filename, 'w') as f:
            f.write(pyro_program)
        print(f'Wrote {output_program_filename} ({len(pyro_program)} chars, '
              f'{pyro_program.count(chr(10))+1} lines)')
    else:
        print('No <START_PYRO_MODEL>/<END_PYRO_MODEL> tags in PART III response; '
              'program NOT written. finish_reason=' + str(fr3))


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('scenario_id')
    args = parser.parse_args()
    await process(args.scenario_id)


if __name__ == '__main__':
    asyncio.run(main())
