import argparse
import asyncio
import itertools
import json
import os
import random

from google import genai
from google.genai import types

DEFAULT_MODEL = "gemini-3-pro-preview"
MAX_CONCURRENT = 32  # semaphore limit for parallel Gemini calls

SPORTS_DIVERSE_PROMPT_FILE = '../generation_prompts/diverse-1-motif-1sce.txt'
SPORTS_PROMPT_FILE = '../generation_prompts/motif-4.txt'
HEALTHCARE_PROMPT_FILE = '../generation_prompts/healthcare.txt'
GENERAL_DOMAINS_PROMPT_FILE = '../generation_prompts/general-domains.txt'
GENERAL_DOMAINS_FILE = '../generation_prompts/domains.txt'

os.environ['GEMINI_API_KEY'] = "<GEMINI_API_KEY_PLACEHOLDER>"

client = genai.Client(http_options=types.HttpOptions(timeout=900000))

# Sports param mappings
SPORTS_P = [
    'X consistently wins',
    'X consistently loses',
    'X wins all but one match',
    'X loses all but one match',
]
SPORTS_C = [
    'X always teams up with the same teammate(s)',
    'X teams up with different player(s) most of the times',
]
SPORTS_R = [
    'players rotate across teams',
    'players have fixed teams',
    'players generally have fixed teams, except X',
]

# Healthcare param mappings
HEALTHCARE_P = [
    'X consistently recovers better',
    'X consistently recovers worse',
    'X recovers better in all observations except one',
    'X recovers worse in all observations except one',
]
HEALTHCARE_C = [
    'X is always observed and treated with the same patient(s)',
    'X is observed or treated with different patient(s) most of the times',
]
HEALTHCARE_R = [
    'patients rotate across groups in observations or treatments',
    'patients have their fixed groups in observations',
    'patients generally have their fixed groups in observations, except X',
]

NEUTRAL_P = [
    'X consistently outperforms',
    'X consistently underperforms',
    'X outperforms in all observations except one',
    'X underperforms worse in all observations except one',
]
NEUTRAL_C = [
    'X is always observed with the same entity(s)',
    'X is observed with different entity(s) most of the times',
]
NEUTRAL_R = [
    'entities rotate across groups in observations',
    'entities have their fixed groups in observations',
    'entities generally have their fixed groups in observations, except X',
]
NEUTRAL_ONE_V_ONE = 'You can optionally create one observation that just compares one entity with another entity.'

SPORTS_N = list(range(2, 10))


def read_file(filename):
    with open(filename) as f:
        return f.read()


def prompt_insert(prompt, placeholder, value):
    parts = prompt.split(placeholder)
    return parts[0] + value + parts[1]


async def run_gemini_async(prompt, temperature, max_tokens):
    config = types.GenerateContentConfig(
        system_instruction='You are a helpful assistant.',
        temperature=temperature,
        max_output_tokens=int(max_tokens),
    )
    resp = await client.aio.models.generate_content(
        model=DEFAULT_MODEL,
        contents=prompt,
        config=config,
    )
    return resp.text


async def generate_motif_scenarios_async(P, C, R, N, N_ind, N_episodic, domain='sports', diverse=True):
    if domain == 'sports':
        one_v_one = 'You can optionally create one match where it is just a 1v1.'
        if diverse:
            prompt = read_file(SPORTS_DIVERSE_PROMPT_FILE)
            subdomains = read_file('../generation_prompts/sports-subdomains.txt')
            subdomains = subdomains.split(', ')
            random.shuffle(subdomains)
            prompt = prompt_insert(prompt, '<SUBDOMAINS>', str(subdomains[:15]))
        else:
            prompt = read_file(SPORTS_PROMPT_FILE)
            subdomains = read_file('../generation_prompts/sports-subdomains.txt')
            subdomains = subdomains.split(', ')
            random.shuffle(subdomains)
            subdomain = subdomains[0]
            with open('../generation_prompts/subdomains-to-variables.txt', 'r') as f:
                subdomain_variables = json.load(f)
            ind_variables = subdomain_variables[subdomain][0]
            episodic_variables = subdomain_variables[subdomain][1]
            random.shuffle(ind_variables)
            random.shuffle(episodic_variables)
            prompt = prompt_insert(prompt, '<SUBDOMAINS>', subdomain)
            prompt = prompt_insert(prompt, '<MAIN_VARIABLE>', str(ind_variables[:N_ind]))
            prompt = prompt_insert(prompt, '<EPISODIC_VARIABLE>', str(episodic_variables[:N_episodic]))
    elif domain == 'healthcare':
        one_v_one = 'You can optionally create one observation that just compares one patient with another patient.'
        prompt = read_file(HEALTHCARE_PROMPT_FILE)
        prompt = prompt_insert(prompt, '<N_ind>', str(N_ind))
        prompt = prompt_insert(prompt, '<N_epi>', str(N_episodic))
    elif domain == 'general':
        one_v_one = NEUTRAL_ONE_V_ONE
        prompt = read_file(GENERAL_DOMAINS_PROMPT_FILE)
        domains_text = read_file(GENERAL_DOMAINS_FILE)
        domains = [d.strip() for d in domains_text.split(',') if d.strip()]
        domain_name = random.choice(domains)
        prompt = prompt_insert(prompt, '<DOMAIN>', domain_name)
        prompt = prompt_insert(prompt, '<N_ind>', str(N_ind))
        prompt = prompt_insert(prompt, '<N_epi>', str(N_episodic))

    prompt = prompt_insert(prompt, '<P>', P)
    prompt = prompt_insert(prompt, '<C>', C)
    prompt = prompt_insert(prompt, '<R>', R)
    prompt = prompt_insert(prompt, '<N>', N)

    if int(N) <= 3:
        prompt = prompt_insert(prompt, '<1v1 OPTION>', one_v_one)
    else:
        prompt = prompt_insert(prompt, '<1v1 OPTION>', '')

    txt = await run_gemini_async(prompt, 0.5, 3e5)

    scenarios = txt.split('<START_SCENARIO>')[1:]
    scenarios = [s.split('<END_SCENARIO>')[0] for s in scenarios]
    scenarios = ['<START_SCENARIO>' + s + '<END_SCENARIO>' for s in scenarios]
    return scenarios


async def generate_scenario_async(P, C, R, N, N_ind, N_episodic, seed, semaphore, domain='sports', diverse=True):
    if domain == 'sports':
        motifs_P, motifs_C, motifs_R = SPORTS_P, SPORTS_C, SPORTS_R
    elif domain == 'healthcare':
        motifs_P, motifs_C, motifs_R = HEALTHCARE_P, HEALTHCARE_C, HEALTHCARE_R
    elif domain == 'general':
        motifs_P, motifs_C, motifs_R = NEUTRAL_P, NEUTRAL_C, NEUTRAL_R

    pi = motifs_P[P]
    ci = motifs_C[C]
    if R >= 1:
        ri = motifs_R[1] if C == 0 else motifs_R[2]
    else:
        ri = motifs_R[0]

    if diverse:
        n_range = range(N, N + 6)
    else:
        n_range = range(N, N + 1)

    for N_i in n_range:
        ni = str(SPORTS_N[N_i])

        if diverse:
            out_prefix = f"scenarios/gemini-diverse-{domain}-P-{P}-C-{C}-R-{R}-N-{N_i}" if domain == 'healthcare' \
                else f"scenarios/gemini-diverse-P-{P}-C-{C}-R-{R}-N-{N_i}"
        else:
            out_prefix = f"scenarios/gemini-{domain}-P-{P}-C-{C}-R-{R}-N-{N_i}-Nind-{N_ind}-Nepi-{N_episodic}"

        out_path_seed = f"{out_prefix}-{seed}.txt"
        if os.path.isfile(out_path_seed):
            print(f"Skipping P={P} C={C} R={R} N_i={N_i} seed={seed} (file exists)")
            continue

        async with semaphore:
            print(f"Generating P={P} C={C} R={R} N_i={N_i} seed={seed} ...")
            scenarios = await generate_motif_scenarios_async(pi, ci, ri, ni, N_ind, N_episodic, domain=domain, diverse=diverse)
            print(f"  -> Got {len(scenarios)} scenario(s) for P={P} C={C} R={R} N_i={N_i}")

        for j, sce in enumerate(scenarios):
            if diverse or domain == 'general':
                out_path = f"{out_prefix}-{seed}.txt"
            else:
                out_path = f"{out_prefix}-{j}.txt"
            with open(out_path, "w") as f:
                f.write(sce)
            print(f"  Saved {out_path}")


async def main():
    parser = argparse.ArgumentParser(description='Async scenario generation')
    parser.add_argument('--diverse', action='store_true', default=False)
    parser.add_argument('--domain', default='sports', type=str)
    parser.add_argument('--sweep_file', default='msa_get_program_part1_sweep.json', type=str)
    args = parser.parse_args()

    with open(args.sweep_file) as f:
        sweep = json.load(f)

    keys = list(sweep.keys())          # e.g. ['P','C','R','N','seed']
    values = [sweep[k] for k in keys]
    combinations = list(itertools.product(*values))
    print(f"Total combinations: {len(combinations)}, domain={args.domain}, diverse={args.diverse}")

    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    N_ind_default = sweep.get('N_ind', [1])[0] if 'N_ind' in sweep else 1
    N_epi_default = sweep.get('N_epi', [1])[0] if 'N_epi' in sweep else 1

    tasks = [
        generate_scenario_async(
            *[combo[keys.index(k)] for k in ['P', 'C', 'R', 'N']],
            combo[keys.index('N_ind')] if 'N_ind' in keys else N_ind_default,
            combo[keys.index('N_epi')] if 'N_epi' in keys else N_epi_default,
            combo[keys.index('seed')],
            semaphore,
            domain=args.domain,
            diverse=args.diverse,
        )
        for combo in combinations
    ]

    await asyncio.gather(*tasks)
    print("All done.")


if __name__ == '__main__':
    asyncio.run(main())
