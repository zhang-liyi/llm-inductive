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

DEFAULT_MODEL = "gemini-3-pro-preview"  # example model used in official quickstarts :contentReference[oaicite:3]{index=3}
os.environ['GEMINI_API_KEY'] = "<GEMINI_API_KEY_PLACEHOLDER>"

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

    if placeholder == '<SUBDOMAINS>':
        prompt = prompt.split('<SUBDOMAINS>')
        prompt = prompt[0] + '[' + value + ']' + prompt[1]

    else:
        prompt = prompt.split(placeholder)
        prompt = prompt[0] + value + prompt[1]

    return prompt

def generate_motif_scenarios(P, C, R, N, N_ind, N_episodic, domain='sports'):

    if domain == 'sports':
        GENERATION_PROMPT_FILE = '../generation_prompts/motif-4.txt'
    elif domain == 'healthcare':
        GENERATION_PROMPT_FILE = '../generation_prompts/healthcare.txt'

    if domain == 'sports':
        subdomains = read_file('../generation_prompts/sports-subdomains.txt')
        subdomains = subdomains.split(', ')
        random.shuffle(subdomains)
        subdomain = subdomains[0]

        with open('../generation_prompts/subdomains-to-variables.txt', 'r') as file:
            subdomain_variables = json.load(file)
        ind_variables = subdomain_variables[subdomain][0]
        episodic_variables = subdomain_variables[subdomain][1]
        random.shuffle(ind_variables)
        random.shuffle(episodic_variables)
        ind_variables = str(ind_variables[:N_ind])
        episodic_variables = str(episodic_variables[:N_episodic])

        one_v_one = 'You can optionally create one match where it is just a 1v1.'

        prompt = read_file(GENERATION_PROMPT_FILE)

        prompt = prompt_insert(prompt, '<SUBDOMAINS>', subdomain)
        prompt = prompt_insert(prompt, '<MAIN_VARIABLE>', ind_variables)
        prompt = prompt_insert(prompt, '<EPISODIC_VARIABLE>', episodic_variables)

        prompt = prompt_insert(prompt, '<P>', P)
        prompt = prompt_insert(prompt, '<C>', C)
        prompt = prompt_insert(prompt, '<R>', R)
        prompt = prompt_insert(prompt, '<N>', N)
    
    elif domain == 'healthcare':

        one_v_one = 'You can optionally create one observation that just compares one patient with another patient.'

        prompt = read_file(GENERATION_PROMPT_FILE)

        prompt = prompt_insert(prompt, '<P>', P)
        prompt = prompt_insert(prompt, '<C>', C)
        prompt = prompt_insert(prompt, '<R>', R)
        prompt = prompt_insert(prompt, '<N>', N)
        prompt = prompt_insert(prompt, '<N_ind>', str(N_ind))
        prompt = prompt_insert(prompt, '<N_epi>', str(N_episodic))

    if int(N) <= 3:
        prompt = prompt_insert(prompt, '<1v1 OPTION>', one_v_one)
    else:
        prompt = prompt_insert(prompt, '<1v1 OPTION>', '')

    txt = run_gemini(prompt, 0.5, 1e5)

    print(txt)
    print()

    scenarios = txt.split('<START_SCENARIO>')[1:]
    scenarios = [s.split('<END_SCENARIO>')[0] for s in scenarios]
    scenarios = ['<START_SCENARIO>' + s + '<END_SCENARIO>' for s in scenarios]
    # scenarios = extract_between_angle_tags_no_re(txt)
    # scenarios = ['<START_SCENARIO>' + s + '<END_SCENARIO>' for s in scenarios]

    return scenarios

def generate_scenario(P, C, R, N, N_ind, N_episodic, seed=0, domain="sports"):
    # probability / difficulty
    if domain == 'sports':
        motifs_P = [
            'X consistently wins',
            'X consistently loses',
            'X wins all but one match',
            'X loses all but one match'
        ]

        # confounded teammates
        motifs_C = [
            'X always teams up with the same teammate(s)',
            'X teams up with different player(s) most of the times'
        ]

        # round-robin
        motifs_R = [
            'players rotate across teams',
            'players have fixed teams',
            'players generally have fixed teams, except X'
        ]
    
    elif domain == 'healthcare':
        motifs_P = [
            'X consistently recovers better',
            'X consistently recovers worse',
            'X recovers better in all observations except one',
            'X recovers worse in all observations except one'
        ]

        # confounded teammates
        motifs_C = [
            'X is always observed and treated with the same patient(s)',
            'X is observed or treated different patient(s) most of the times'
        ]

        # round-robin
        motifs_R = [
            'patients rotate across groups in observations or treatments',
            'patients have their fixed groups in observations',
            'patients generally have their fixed groups in observations, except X'
        ]

    # team-size
    motifs_N = list(range(2, 10)) 

    pi = motifs_P[P]
    ci = motifs_C[C]
    if R >= 1:
        if C == 0:
            ri = motifs_R[1]
        else:
            ri = motifs_R[2]
    else:
        ri = motifs_R[0]
    

    for N_i in range(6):

        ni = str(motifs_N[N_i])

        if not os.path.isfile(f"scenarios/gemini-{domain}-P-{P}-C-{C}-R-{R}-N-{N_i}-Nind-{N_ind}-Nepi-{N_episodic}-{0}.txt") or \
            not os.path.isfile(f"scenarios/gemini-{domain}-P-{P}-C-{C}-R-{R}-N-{N_i}-Nind-{N_ind}-Nepi-{N_episodic}-{1}.txt") or \
            not os.path.isfile(f"scenarios/gemini-{domain}-P-{P}-C-{C}-R-{R}-N-{N_i}-Nind-{N_ind}-Nepi-{N_episodic}-{2}.txt") or \
            not os.path.isfile(f"scenarios/gemini-{domain}-P-{P}-C-{C}-R-{R}-N-{N_i}-Nind-{N_ind}-Nepi-{N_episodic}-{3}.txt") or \
            not os.path.isfile(f"scenarios/gemini-{domain}-P-{P}-C-{C}-R-{R}-N-{N_i}-Nind-{N_ind}-Nepi-{N_episodic}-{4}.txt"):

            # for i in range(5):
            # # currently, there's just one scenario in the list of scenarios
            scenarios = generate_motif_scenarios(pi, ci, ri, ni, N_ind, N_episodic, domain=domain)
            print(scenarios)
            
            for j, sce in enumerate(scenarios):
                with open(f"scenarios/gemini-{domain}-P-{P}-C-{C}-R-{R}-N-{N_i}-Nind-{N_ind}-Nepi-{N_episodic}-{j}.txt", "w") as f:
                    f.write(sce)

