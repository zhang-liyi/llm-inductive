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

DIVERSE_MODE = False

if DIVERSE_MODE:
    from generate_scenario_diverse import generate_scenario
else:
    from generate_scenario import generate_scenario

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

    args = parser.parse_args()
    return args

args = parse_args()

# DEFAULT_MODEL = "gemini-3-pro-preview"  # example model used in official quickstarts :contentReference[oaicite:3]{index=3}
# os.environ['GEMINI_API_KEY'] = '<GEMINI_API_KEY_PLACEHOLDER>'


# client = genai.Client()  # reads API key from env

# def run_gemini(prompt, temperature, max_tokens, system_prompt='You are a helpful assistant.'):
#     config = types.GenerateContentConfig(
#             system_instruction=system_prompt,  # supported field :contentReference[oaicite:4]{index=4}
#             temperature=temperature,
#             max_output_tokens=max_tokens,
#         )

#     resp = client.models.generate_content(
#         model=DEFAULT_MODEL,
#         contents=prompt,
#         config=config
#     )

#     return resp.text



# program_ids = list(range(50))
# program_ids = ['canoe','tugofwar','biathlon']

# Hyperparameters, Filenames

generate_scenario(args.P, args.C, args.R, args.N, args.N_ind, args.N_epi, seed=args.seed, domain=args.domain)

'''
    "N_ind":[1,2],
    "N_epi":[1,2,3,4],
'''