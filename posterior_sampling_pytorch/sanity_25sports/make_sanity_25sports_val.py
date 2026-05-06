"""Combine sanity_25sports_queries.json + per-scenario inference_results
into the standard pytorch_*_val.json eval format.

Output schema matches data_evaluation/evaluate_healthcare.py:
    [{"input": <full prompt>, "output": <int str>, "bins": [[101 floats]]}, ...]
"""
import json
import os

ROOT = './posterior_sampling_pytorch'
QUERIES_JSON = f'{ROOT}/sanity_25sports/sanity_25sports_queries.json'
RESULTS_DIR = f'{ROOT}/sanity_25sports/inference_results'
OUT_PATH = ('./torchtune/data/'
            'pyro-rej/pytorch_sanity_25sports_val.json')

# The instruction prefix used by the original training data (we match this so
# the eval pipeline behaves identically to its existing healthcare counterpart;
# downstream `--bracket_prompt` swaps it for BRACKET_INSTRUCTION at eval time).
OLD_INSTRUCTION = (
    "Answer the query in the scenario and return only an integer. Use 0-100 "
    "scale. For a query on individual rank or performance, a higher number "
    "means more strength (e.g. 100 is stronger than 1). For a query on which "
    "team wins, a smaller number means the first team more likely wins."
)


def build_prompt(scenario_text, query_nl):
    """Stitch instruction + scenario_text (BACKGROUND, CONDITIONS) + ONE query.
    The scenario_text from the .txt file already contains <START_SCENARIO>...
    so we end with the new QUERIES line and <END_SCENARIO>."""
    body = scenario_text.strip()
    # scenario_text from parse_scenario_txt is the prefix BEFORE 'QUERIES\n'
    # — i.e. ends with the CONDITIONS block. Append a single fresh QUERY.
    return (
        f'{OLD_INSTRUCTION}\n\nHere is the scenario:\n\n{body}\n\nQUERIES\n'
        f'Query: {query_nl}\n<END_SCENARIO>'
    )


def main():
    spec = json.load(open(QUERIES_JSON))
    out = []
    n_scenarios_in = 0
    for s in spec:
        sid = s['scenario_id']
        ipath = f'{RESULTS_DIR}/{sid}.json'
        if not os.path.isfile(ipath):
            print(f'  miss: {sid}')
            continue
        infer = json.load(open(ipath))
        # Match each new_query to its inference result by (helper, args).
        spec_qs = s['new_queries']
        infer_qs = infer['queries']
        if len(spec_qs) != len(infer_qs):
            print(f'  len mismatch: {sid}')
            continue
        n_scenarios_in += 1
        for sq, iq in zip(spec_qs, infer_qs):
            assert sq['helper'] == iq['helper']
            prompt = build_prompt(s['scenario_text'], sq['nl'])
            out.append({
                'input': prompt,
                'output': str(int(round(iq['mean']))),
                'bins': [iq['bins']],
                'metadata': {
                    'scenario_id': sid,
                    'query_type': sq['type'],
                    'helper': sq['helper'],
                    'args': sq['args'],
                },
            })
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, 'w') as f:
        json.dump(out, f)
    print(f'Wrote {OUT_PATH}: {len(out)} items from {n_scenarios_in} scenarios.')


if __name__ == '__main__':
    main()
