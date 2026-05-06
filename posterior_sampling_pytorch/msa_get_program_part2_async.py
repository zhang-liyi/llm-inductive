import asyncio
import itertools
import json
import os

from google import genai
from google.genai import types

DEFAULT_MODEL = "gemini-3-pro-preview"
MAX_CONCURRENT = 16  # simultaneous program_id pipelines

os.environ['GEMINI_API_KEY'] = '<GEMINI_API_KEY_PLACEHOLDER>'

client = genai.Client(http_options=types.HttpOptions(timeout=900000))


def read_file(filename):
    with open(filename) as f:
        return f.read()


async def run_gemini_async(prompt, temperature, max_tokens, system_prompt='You are a helpful assistant.'):
    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        temperature=temperature,
        max_output_tokens=int(max_tokens),
    )
    resp = await client.aio.models.generate_content(
        model=DEFAULT_MODEL,
        contents=prompt,
        config=config,
    )
    return resp.text


async def process_program_id(program_id, semaphore):
    input_filename = f'scenarios/gemini-{program_id}.txt'
    output_program_filename = f'programs/pg-gemini-{program_id}.py'
    scratchpad_filename = f'scratchpads/gemini-{program_id}.json'
    negatives_filename = f'negatives/gemini-{program_id}.json'

    if not os.path.isfile(input_filename):
        return

    if os.path.isfile(output_program_filename):
        print(f'Skipping {program_id} (already done)')
        return

    temperature = 0.0
    max_tokens = 3e5

    async with semaphore:
        print(f'Processing {program_id} ...')
        scenario = read_file(input_filename)

        # Direct response (negative DPO candidate)
        direct_response = await run_gemini_async(scenario, temperature, max_tokens)
        os.makedirs('negatives', exist_ok=True)
        with open(negatives_filename, 'w') as f:
            json.dump({
                'scenario_id': f'gemini-{program_id}',
                'prompt': scenario,
                'response': direct_response,
            }, f, indent=2)

        # PART I - Parse
        part1prompt = read_file('pg-part1.txt')
        response = await run_gemini_async(part1prompt + '\n\n' + scenario, temperature, max_tokens)
        current_program = scenario + '\n\n' + response

        # PART II - Knowledge / scratchpad
        part2prompt = read_file('pg-part2.txt')
        response = await run_gemini_async(part2prompt + '\n\n' + current_program, temperature, max_tokens)
        tmp_program1 = current_program
        current_program = current_program + '\n\n' + response

        scratchpad_full = None
        concept_trace = None
        if '<START_SCRATCHPAD>' in response and '<END_SCRATCHPAD>' in response:
            scratchpad_full = response.split('<START_SCRATCHPAD>')[1].split('<END_SCRATCHPAD>')[0].strip()
        if scratchpad_full and '<START_CONCEPT_TRACE>' in scratchpad_full and '<END_CONCEPT_TRACE>' in scratchpad_full:
            concept_trace = scratchpad_full.split('<START_CONCEPT_TRACE>')[1].split('<END_CONCEPT_TRACE>')[0].strip()

        os.makedirs('scratchpads', exist_ok=True)
        with open(scratchpad_filename, 'w') as f:
            json.dump({
                'scenario_id': f'gemini-{program_id}',
                'prompt': tmp_program1,
                'scratchpad': scratchpad_full,
                'concept_trace': concept_trace,
            }, f, indent=2)

        # PART III - Write model
        part3prompt = read_file('pg-part3.txt')
        system_prompt = read_file('pg-system-prompt.txt')
        response = await run_gemini_async(
            part3prompt + '\n\n' + current_program, temperature, max_tokens, system_prompt
        )

        try:
            pyro_program = response.split('<START_PYRO_MODEL>\n')[1].split('\n<END_PYRO_MODEL>')[0]
        except IndexError:
            print(f'WARNING: malformed response for {program_id}, skipping program write')
            return

        os.makedirs('programs', exist_ok=True)
        with open(output_program_filename, 'w') as f:
            f.write(pyro_program)

        print(f'Done {program_id}')


async def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', default='diverse', type=str)
    parser.add_argument('--sweep_file', default='msa_get_program_part2_sweep.json', type=str)
    args = parser.parse_args()

    with open(args.sweep_file) as f:
        sweep = json.load(f)

    keys = list(sweep.keys())
    combos = list(itertools.product(*[sweep[k] for k in keys]))
    print(f'Total combinations: {len(combos)}, domain={args.domain}')

    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    tasks = []
    for combo in combos:
        params = dict(zip(keys, combo))
        P, C, R = params['P'], params['C'], params['R']

        if args.domain == 'diverse':
            seed = params['seed']
            for N in range(6):
                program_id = f'diverse-P-{P}-C-{C}-R-{R}-N-{N}-{seed}'
                tasks.append(process_program_id(program_id, semaphore))
        elif args.domain == 'healthcare':
            N = params['N']
            N_ind = params['N_ind']
            N_epi = params['N_epi']
            seed = params['seed']
            for j in range(5):
                program_id = f'healthcare-P-{P}-C-{C}-R-{R}-N-{N}-Nind-{N_ind}-Nepi-{N_epi}-{j}'
                tasks.append(process_program_id(program_id, semaphore))
        elif args.domain == 'general':
            N = params['N']
            N_ind = params['N_ind']
            N_epi = params['N_epi']
            seed = params['seed']
            program_id = f'general-P-{P}-C-{C}-R-{R}-N-{N}-Nind-{N_ind}-Nepi-{N_epi}-{seed}'
            tasks.append(process_program_id(program_id, semaphore))

    print(f'Total program_ids to process: {len(tasks)}')
    await asyncio.gather(*tasks)
    print('All done.')


if __name__ == '__main__':
    asyncio.run(main())
