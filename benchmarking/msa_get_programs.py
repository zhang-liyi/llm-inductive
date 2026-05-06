"""Generate pyro programs for the 21 e2_implicit benchmarking scenarios.

Adapted from posterior_sampling_pytorch/msa_get_program_part2_async.py.
Only processes scenarios that have human responses in msa_cogsci_human_data.json
under the 'e2_implicit' key.
"""

import argparse
import asyncio
import json
import os

from google import genai
from google.genai import types


PYTORCH_DIR = './posterior_sampling_pytorch'
BENCH_DIR = './benchmarking'

DEFAULT_MODEL = "gemini-3-pro-preview"
MAX_CONCURRENT = 8

os.environ['GEMINI_API_KEY'] = '<GEMINI_API_KEY_PLACEHOLDER>'

client = genai.Client(http_options=types.HttpOptions(timeout=900000))


def read_file(path):
    with open(path) as f:
        return f.read()


async def run_gemini_async(prompt, temperature, max_tokens,
                           system_prompt='You are a helpful assistant.'):
    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        temperature=temperature,
        max_output_tokens=int(max_tokens),
    )
    resp = await client.aio.models.generate_content(
        model=DEFAULT_MODEL, contents=prompt, config=config,
    )
    return resp.text


async def process_scenario(scenario_id, semaphore):
    input_filename = f'{BENCH_DIR}/scenarios/{scenario_id}.txt'
    output_program_filename = f'{BENCH_DIR}/programs/pg-{scenario_id}.py'
    scratchpad_filename = f'{BENCH_DIR}/scratchpads/{scenario_id}.json'

    if not os.path.isfile(input_filename):
        print(f'Missing scenario file: {input_filename}')
        return

    if os.path.isfile(output_program_filename):
        print(f'Skipping {scenario_id} (already done)')
        return

    temperature = 0.0
    max_tokens = 3e5

    async with semaphore:
        print(f'Processing {scenario_id} ...')
        scenario_raw = read_file(input_filename).rstrip()
        # Benchmark scenarios don't include the START/END tags used by the
        # few-shot examples inside pg-part*.txt, so wrap them for consistency.
        if '<START_SCENARIO>' not in scenario_raw:
            scenario = f'<START_SCENARIO>\n{scenario_raw}\n<END_SCENARIO>'
        else:
            scenario = scenario_raw

        # PART I - Parse
        part1prompt = read_file(f'{PYTORCH_DIR}/pg-orig-part1.txt')
        response = await run_gemini_async(part1prompt + '\n\n' + scenario,
                                          temperature, max_tokens)
        current_program = scenario + '\n\n' + response

        # PART II - Scratchpad / knowledge
        part2prompt = read_file(f'{PYTORCH_DIR}/pg-orig-part2.txt')
        response = await run_gemini_async(part2prompt + '\n\n' + current_program,
                                          temperature, max_tokens)
        current_program = current_program + '\n\n' + response

        scratchpad_full = None
        concept_trace = None
        if '<START_SCRATCHPAD>' in response and '<END_SCRATCHPAD>' in response:
            scratchpad_full = response.split('<START_SCRATCHPAD>')[1] \
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

        # PART III - Write model
        part3prompt = read_file(f'{PYTORCH_DIR}/pg-orig-part3.txt')
        system_prompt = read_file(f'{PYTORCH_DIR}/pg-system-prompt.txt')
        response = await run_gemini_async(
            part3prompt + '\n\n' + current_program,
            temperature, max_tokens, system_prompt,
        )

        try:
            pyro_program = response.split('<START_PYRO_MODEL>\n')[1] \
                                   .split('\n<END_PYRO_MODEL>')[0]
        except IndexError:
            print(f'WARNING: malformed response for {scenario_id}')
            with open(f'{BENCH_DIR}/logs/malformed-{scenario_id}.txt', 'w') as f:
                f.write(response)
            return

        with open(output_program_filename, 'w') as f:
            f.write(pyro_program)
        print(f'Done {scenario_id}')


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenarios', nargs='*', default=None,
                        help='Optional subset of scenario IDs to process.')
    args = parser.parse_args()

    with open(f'{BENCH_DIR}/msa_cogsci_human_data.json') as f:
        human = json.load(f)
    scenario_ids = list(human['e2_implicit'].keys())

    if args.scenarios:
        scenario_ids = [s for s in scenario_ids if s in args.scenarios]

    print(f'Processing {len(scenario_ids)} scenario(s).')
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    tasks = [process_scenario(sid, semaphore) for sid in scenario_ids]
    await asyncio.gather(*tasks)
    print('All done.')


if __name__ == '__main__':
    asyncio.run(main())
