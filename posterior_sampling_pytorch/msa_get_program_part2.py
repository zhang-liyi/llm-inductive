import subprocess
import time
import json
import datetime
import numpy as np
import argparse
import os
import sys
import random

from google import genai
from google.genai import types

tm = str(datetime.datetime.now())
TMSTR = tm[:10]+'-'+tm[11:13]+tm[14:16]+tm[17:19]


def read_file(filename):
    with open(filename) as f:
        lines = f.readlines()
    s = ''
    for l in lines:
        s += l
    return s

def parse_args():

    parser = argparse.ArgumentParser(description='Inductive-LLM')

    ### Data loading / data preparation arguments -------------+
    parser.add_argument('--P', 
                        default=0, 
                        type=int)
    parser.add_argument('--C', 
                        default=0, 
                        type=int)
    parser.add_argument('--R', 
                        default=0, 
                        type=int)
    parser.add_argument('--N', 
                        default=0, 
                        type=int)
    parser.add_argument('--N_ind', 
                        default=1, 
                        type=int)
    parser.add_argument('--N_epi', 
                        default=1, 
                        type=int)
    parser.add_argument('--seed',
                        default=0,
                        type=int)
    parser.add_argument('--domain',
                        default='sports',
                        type=str)
    parser.add_argument('--diverse',
                        action='store_true',
                        default=False)
    parser.add_argument('--part2_file',
                        default='pg-part2.txt',
                        type=str)
    parser.add_argument('--suffix',
                        default='',
                        type=str)

    args = parser.parse_args()
    return args

args = parse_args()

DEFAULT_MODEL = "gemini-3-pro-preview"  # example model used in official quickstarts :contentReference[oaicite:3]{index=3}
os.environ['GEMINI_API_KEY'] = '<GEMINI_API_KEY_PLACEHOLDER>'
#'<GEMINI_API_KEY_PLACEHOLDER>'


client = genai.Client()  # reads API key from env

def run_gemini(prompt, temperature, max_tokens, system_prompt='You are a helpful assistant.'):
    config = types.GenerateContentConfig(
            system_instruction=system_prompt,  # supported field :contentReference[oaicite:4]{index=4}
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

    resp = client.models.generate_content(
        model=DEFAULT_MODEL,
        contents=prompt,
        config=config
    )

    return resp.text

START_SEED = 0
program_ids = []

n_range = range(6) if args.diverse else range(2)
for N in n_range:
    if args.diverse:
        program_ids.append(f'diverse-P-{args.P}-C-{args.C}-R-{args.R}-N-{N}-{args.seed}')
    else:
        program_ids.append(f'{args.domain}-P-{args.P}-C-{args.C}-R-{args.R}-N-{N}-Nind-{args.N_ind}-Nepi-{args.N_epi}-{args.seed}')

for j, program_id in enumerate(program_ids):

    input_filename = f'scenarios/gemini-{program_id}.txt'
    output_program_filename = f'programs/pg-gemini-{program_id}{args.suffix}.py'
    output_json_filename = f'inference_results/result-gemini-{program_id}{args.suffix}.json'

    if not os.path.isfile(input_filename):
        continue

    # input_filename = 'pg-scenario.txt'
    # output_program_filename = 'pg.wppl'
    # output_json_filename = 'result-existing-canoe-10000.json'

    temperature = 0.0
    max_tokens = 1e5

    scenario = read_file(input_filename)

    # Direct Gemini call — no chain-of-thought prompting, just the raw scenario.
    # Serves as a negative DPO example (poor reasoning, weak estimates).
    direct_response = run_gemini(scenario, temperature, max_tokens)
    os.makedirs('negatives', exist_ok=True)
    with open(f'negatives/gemini-{program_id}.json', 'w') as f:
        json.dump({
            'scenario_id': f'gemini-{program_id}',
            'prompt': scenario,
            'response': direct_response,
        }, f, indent=2)

    # PART I - Parse

    part1prompt = read_file('pg-part1.txt')
    part1prompt_program = part1prompt + '\n\n' + scenario

    response = run_gemini(part1prompt_program, temperature, max_tokens)

    current_program = scenario + '\n\n' + response

    # PART II - Knowledge

    part2prompt = read_file(args.part2_file)
    part2prompt_program = part2prompt + '\n\n' + current_program

    response = run_gemini(part2prompt_program, temperature, max_tokens)

    tmp_program1 = current_program
    current_program = current_program + '\n\n' + response

    # Parse and save scratchpad + concept trace from part 2 as DPO positive candidate
    scratchpad_full = None
    concept_trace = None

    if '<START_SCRATCHPAD>' in response and '<END_SCRATCHPAD>' in response:
        scratchpad_full = response.split('<START_SCRATCHPAD>')[1].split('<END_SCRATCHPAD>')[0].strip()

    if scratchpad_full and '<START_CONCEPT_TRACE>' in scratchpad_full and '<END_CONCEPT_TRACE>' in scratchpad_full:
        concept_trace = scratchpad_full.split('<START_CONCEPT_TRACE>')[1].split('<END_CONCEPT_TRACE>')[0].strip()

    os.makedirs('scratchpads', exist_ok=True)
    scratchpad_filename = f'scratchpads/gemini-{program_id}{args.suffix}.json'
    with open(scratchpad_filename, 'w') as f:
        json.dump({
            'scenario_id': f'gemini-{program_id}',
            'prompt': tmp_program1,       # scenario + part1 parse (input context for part 2)
            'scratchpad': scratchpad_full, # full <START_SCRATCHPAD>…<END_SCRATCHPAD> body
            'concept_trace': concept_trace # <START_CONCEPT_TRACE>…<END_CONCEPT_TRACE> body
        }, f, indent=2)

    # PART III - Write Model

    part3prompt = read_file('pg-part3.txt')
    part3prompt_program = part3prompt + '\n\n' + current_program
    system_prompt = read_file('pg-system-prompt.txt')

    response = run_gemini(part3prompt_program, temperature, max_tokens, system_prompt)

    tmp_program2 = current_program
    current_program = current_program + '\n\n' + response
    pyro_program = response.split('<START_PYRO_MODEL>\n')[1].split('\n<END_PYRO_MODEL>')[0]

    #print(pyro_program)
    with open(output_program_filename, "w") as f:
        f.write(pyro_program)

    print(pyro_program)
