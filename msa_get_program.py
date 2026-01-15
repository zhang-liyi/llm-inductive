import subprocess
import time
import json
import datetime
import numpy as np
import argparse
import os
import sys

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
    parser.add_argument('--seed', 
                        default=0, 
                        type=int)

    args = parser.parse_args()
    return args

args = parse_args()

DEFAULT_MODEL = "gemini-3-pro-preview"  # example model used in official quickstarts :contentReference[oaicite:3]{index=3}
os.environ['GEMINI_API_KEY'] = 'AIzaSyAYc4GTOQYyPKvU7OuvG7Q-aXpIyhQFCkk'


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

# program_ids = list(range(50))
program_ids = ['canoe','tugofwar','biathlon']


# Hyperparameters, Filenames
# program_ids = [f'P-{args.P}-C-{args.C}-R-{args.R}-N-{args.N}-{args.seed}']
for program_id in program_ids:

    input_filename = f'scenarios/benchmarks/sc-{program_id}.txt'
    output_program_filename = f'programs/pg-gemini-{program_id}.wppl'
    output_json_filename = f'inference_results/result-gemini-{program_id}.json'

    # input_filename = 'pg-scenario.txt'
    # output_program_filename = 'pg.wppl'
    # output_json_filename = 'result-existing-canoe-10000.json'

    temperature = 0.0
    max_tokens = 1e5

    scenario = read_file(input_filename)
    # print(scenario)
    # print('=================================')

    # PART I - Parse

    part1prompt = read_file('pg-part1.txt')
    part1prompt_program = part1prompt + '\n\n' + scenario

    response = run_gemini(part1prompt_program, temperature, max_tokens)

    current_program = scenario + '\n\n' + response

    # PART II - Knowledge

    part2prompt = read_file('pg-part2.txt')
    part2prompt_program = part2prompt + '\n\n' + current_program

    response = run_gemini(part2prompt_program, temperature, max_tokens)

    tmp_program1 = current_program
    current_program = current_program + '\n\n' + response

    # PART III - Write Model

    part3prompt = read_file('pg-part3.txt')
    part3prompt_program = part3prompt + '\n\n' + current_program
    system_prompt = read_file('pg-system-prompt.txt')

    response = run_gemini(part3prompt_program, temperature, max_tokens, system_prompt)

    tmp_program2 = current_program
    current_program = current_program + '\n\n' + response
    webppl_program = response.split('<START_WEBPPL_MODEL>\n')[1].split('\n<END_WEBPPL_MODEL>')[0]
    # webppl_program += f'\njson.write(\'{output_json_filename}\', posterior);'
    additional_helpers = read_file('additional_helpers.txt')
    webppl_program = additional_helpers + '\n' + webppl_program

    #print(webppl_program)
    with open(output_program_filename, "w") as f:
        f.write(webppl_program)
