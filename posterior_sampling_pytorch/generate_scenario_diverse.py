import subprocess
import time
import json
import datetime
import numpy as np
import argparse
import os
import sys
import random
import json
from typing import List, Tuple, Optional

from google import genai
from google.genai import types

def read_file(filename):
    with open(filename) as f:
        lines = f.readlines()
    s = ''
    for l in lines:
        s += l
    return s

GENERATION_PROMPT_FILE = '../generation_prompts/diverse-1-motif-1sce.txt'

DEFAULT_MODEL = "gemini-3-pro-preview"  # example model used in official quickstarts :contentReference[oaicite:3]{index=3}
# os.environ['GEMINI_API_KEY'] = '<GEMINI_API_KEY_PLACEHOLDER>'
os.environ['GEMINI_API_KEY'] = "<GEMINI_API_KEY_PLACEHOLDER>"

client = genai.Client(http_options=types.HttpOptions(
        timeout=900000,  # milliseconds = 900 seconds
    ))  # reads API key from env

def run_gemini(prompt, temperature, max_tokens, system_prompt='You are a helpful assistant.'):
    config = types.GenerateContentConfig(
            system_instruction=system_prompt,  # supported field :contentReference[oaicite:4]{index=4}
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

    # resp = client.models.generate_content(
    #     model=DEFAULT_MODEL,
    #     contents=prompt,
    #     config=config
    # )

    full_text = []
    for chunk in client.models.generate_content_stream(
        model=DEFAULT_MODEL,
        contents=prompt,
        config=config
    ):
        if chunk.text:
            print(chunk.text, end="", flush=True)
            full_text.append(chunk.text)

    result = "".join(full_text)

    return result

def _parse_open_tag(s: str, i: int) -> Optional[Tuple[str, int]]:
    """
    If s[i:] starts with an opening tag like <tag ...>, return (tag_name, idx_after_>).
    Otherwise return None.
    """
    if i >= len(s) or s[i] != "<":
        return None
    j = i + 1
    if j < len(s) and s[j] == "/":  # closing tag, not opening
        return None

    # tag name = run of non-space and not '>'
    name_start = j
    while j < len(s) and s[j] not in (" ", "\t", "\n", "\r", ">"):
        j += 1
    tag = s[name_start:j]
    if not tag:
        return None

    # advance to end of tag '>'
    while j < len(s) and s[j] != ">":
        j += 1
    if j >= len(s):
        return None
    return tag, j + 1  # position right after '>'

def _parse_close_tag(s: str, i: int, tag: str) -> Optional[int]:
    """
    If s[i:] starts with a closing tag like </tag ...>, return idx_after_>.
    Otherwise return None.
    """
    if i + 2 >= len(s) or s[i:i+2] != "</":
        return None
    j = i + 2

    # read tag name
    name_start = j
    while j < len(s) and s[j] not in (" ", "\t", "\n", "\r", ">"):
        j += 1
    close_tag = s[name_start:j]
    if close_tag != tag:
        return None

    # advance to end of tag '>'
    while j < len(s) and s[j] != ">":
        j += 1
    if j >= len(s):
        return None
    return j + 1

def extract_between_angle_tags_no_re(text: str) -> List[str]:
    """
    Extract inner text for each <tag> ... </tag> block (tag name can vary),
    without using re. Returns a list of inner contents (tags excluded).
    """
    out: List[str] = []
    i = 0
    n = len(text)

    while i < n:
        opened = _parse_open_tag(text, i)
        if not opened:
            i += 1
            continue

        tag, content_start = opened

        # Find the matching closing tag for this tag name
        j = content_start
        while j < n:
            if text[j] == "<":
                close_end = _parse_close_tag(text, j, tag)
                if close_end is not None:
                    out.append(text[content_start:j])  # inner content
                    i = close_end  # continue scanning after closing tag
                    break
            j += 1
        else:
            # no closing tag found; stop or skip (here we stop scanning)
            break

    return out

def prompt_insert(prompt, placeholder, value):

    prompt = prompt.split(placeholder)
    prompt = prompt[0] + value + prompt[1]

    return prompt

def generate_motif_scenarios(P, C, R, N):

    subdomains = read_file('../generation_prompts/sports-subdomains.txt')
    subdomains = subdomains.split(', ')
    random.shuffle(subdomains)

    one_v_one = 'You can optionally create one match where it is just a 1v1.'

    prompt = read_file(GENERATION_PROMPT_FILE)

    prompt = prompt_insert(prompt, '<SUBDOMAINS>', str(subdomains[:10]))

    prompt = prompt_insert(prompt, '<P>', P)
    prompt = prompt_insert(prompt, '<C>', C)
    prompt = prompt_insert(prompt, '<R>', R)
    prompt = prompt_insert(prompt, '<N>', N)

    if int(N) <= 3:
        prompt = prompt_insert(prompt, '<1v1 OPTION>', one_v_one)
    else:
        prompt = prompt_insert(prompt, '<1v1 OPTION>', '')

    # response = client.responses.create(
    #     model="gpt-5.1",
    #     reasoning={"effort": "medium"},  # thinking mode
    #     input=[
    #         {
    #             "role": "user",
    #             "content": prompt
    #         }
    #     ],
    # )

    # txt = response.output_text

    # txt = run_gemini('output a random word', 0.0, 3e5)
    # print(txt)
    # print()

    txt = run_gemini(prompt, 0.5, 3e5)
    # print(txt)
    # print()

    scenarios = txt.split('<START_SCENARIO>')[1:]
    scenarios = [s.split('<END_SCENARIO>')[0] for s in scenarios]
    scenarios = ['<START_SCENARIO>' + s + '<END_SCENARIO>' for s in scenarios]
    # scenarios = extract_between_angle_tags_no_re(txt)
    # scenarios = ['<START_SCENARIO>' + s + '<END_SCENARIO>' for s in scenarios]

    return scenarios

def generate_scenario(P, C, R, N, N_ind, N_episodic, seed=0):
    # probability / difficulty
    sports_P = [
        'X consistently wins',
        'X consistently loses',
        'X wins all but one match',
        'X loses all but one match'
    ]

    # confounded teammates
    sports_C = [
        'X always teams up with the same teammate(s)',
        'X teams up with different player(s) most of the times'
    ]

    # round-robin
    sports_R = [
        'players rotate across teams',
        'players have fixed teams',
        'players generally have fixed teams, except X'
    ]

    # team-size
    sports_N = list(range(2, 10)) 

    pi = sports_P[P]
    ci = sports_C[C]
    if R >= 1:
        if C == 0:
            ri = sports_R[1]
        else:
            ri = sports_R[2]
    else:
        ri = sports_R[0]
    

    for N_i in range(N, N+6):

        ni = str(sports_N[N_i])

        if not os.path.isfile(f"scenarios/gemini-diverse-P-{P}-C-{C}-R-{R}-N-{N_i}-{0}.txt") or \
            not os.path.isfile(f"scenarios/gemini-diverse-P-{P}-C-{C}-R-{R}-N-{N_i}-{1}.txt") or \
            not os.path.isfile(f"scenarios/gemini-diverse-P-{P}-C-{C}-R-{R}-N-{N_i}-{2}.txt") or \
            not os.path.isfile(f"scenarios/gemini-diverse-P-{P}-C-{C}-R-{R}-N-{N_i}-{3}.txt") or \
            not os.path.isfile(f"scenarios/gemini-diverse-P-{P}-C-{C}-R-{R}-N-{N_i}-{4}.txt"):

            # currently just one scenario in the list of 'scenarios'
            scenarios = generate_motif_scenarios(pi, ci, ri, ni)
            print(scenarios)
            
            for j, sce in enumerate(scenarios):
                with open(f"scenarios/gemini-diverse-P-{P}-C-{C}-R-{R}-N-{N_i}-{seed}.txt", "w") as f:
                    f.write(sce)

