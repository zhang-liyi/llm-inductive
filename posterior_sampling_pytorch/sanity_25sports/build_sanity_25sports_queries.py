"""Sample 25 sports training scenarios at random and design 4 new
strict-different queries per scenario for sanity-check evaluation.

Strict-different: each new query's args differ from the corresponding
original query's args (Q1' athlete != original Q1 athlete; Q2' (a,m) !=
original; Q3'/Q4' team rosters as multisets differ from originals).

Outputs:
  sanity_25sports/sanity_25sports_queries.json
"""
import argparse
import ast
import json
import os
import random
import re

ROOT = './posterior_sampling_pytorch'
OLD_ROOT = './posterior_sampling'
PROGRAMS_DIR = f'{ROOT}/programs'
SCENARIOS_DIR = f'{OLD_ROOT}/scenarios'
OUT_DIR = f'{ROOT}/sanity_25sports'

PG_RE = re.compile(
    r'^pg-gemini-P-(\d+)-C-(\d+)-R-(\d+)-N-(\d+)-Nind-(\d+)-Nepi-(\d+)-(\d+)\.py$'
)


def list_sports_program_ids():
    """Return scenario_id strings for all sports training programs (the
    plain `pg-gemini-P-{...}-{seed}.py` family, NOT diverse / general / REJ)."""
    ids = []
    for fname in os.listdir(PROGRAMS_DIR):
        if PG_RE.match(fname):
            ids.append(fname[len('pg-'):-len('.py')])  # gemini-P-...-{seed}
    return sorted(ids)


def parse_program(scenario_id):
    """AST-parse the program. Returns dict with athletes, player_matches,
    helper names, and original main-block call args."""
    src = open(f'{PROGRAMS_DIR}/pg-{scenario_id}.py').read()
    tree = ast.parse(src)

    athletes = None
    player_matches = None
    helpers = []
    main_calls = {}  # fname -> list of [arg, ...]

    for node in tree.body:
        if (isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)):
            name = node.targets[0].id
            try:
                if name == 'ATHLETES':
                    athletes = ast.literal_eval(node.value)
                elif name == 'PLAYER_MATCHES':
                    player_matches = ast.literal_eval(node.value)
            except (ValueError, SyntaxError):
                pass
        elif isinstance(node, ast.FunctionDef):
            helpers.append(node.name)
        elif isinstance(node, ast.If):
            for sub in ast.walk(node):
                if (isinstance(sub, ast.Call)
                        and isinstance(sub.func, ast.Name)):
                    fname = sub.func.id
                    if (fname == 'intrinsic_strength_rank'
                            or fname.startswith('query_')
                            or fname == 'who_would_win_by_how_much'):
                        try:
                            args = [ast.literal_eval(a) for a in sub.args[1:]]
                        except (ValueError, SyntaxError):
                            args = None
                        if args is not None:
                            main_calls.setdefault(fname, []).append(args)

    # Identify the episodic helper (the `query_*_in_match` one).
    episodic = next(
        (h for h in helpers if h.startswith('query_') and h != 'query_'),
        None,
    )
    return {
        'athletes': athletes,
        'player_matches': [tuple(pm) for pm in (player_matches or [])],
        'episodic_helper': episodic,
        'main_calls': main_calls,
    }


def parse_scenario_txt(scenario_id):
    """Extract BACKGROUND+CONDITIONS prefix and the 4 raw query NL strings."""
    txt_path = f'{SCENARIOS_DIR}/{scenario_id}.txt'
    if not os.path.isfile(txt_path):
        raise FileNotFoundError(txt_path)
    txt = open(txt_path).read()
    pre, _, queries_block = txt.partition('QUERIES\n')
    queries_block = queries_block.replace('<END_SCENARIO>', '').strip()
    queries = []
    for line in queries_block.splitlines():
        line = line.strip()
        m = re.match(r'^Query\s*\d+\s*:\s*(.+)$', line)
        if m:
            queries.append(m.group(1).strip())
    return pre.strip(), queries


def cap(name):
    return name.capitalize()


def render_team(names):
    """Render a list of names with Oxford comma + 'and'."""
    names = [cap(n) for n in names]
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f'{names[0]} and {names[1]}'
    return ', '.join(names[:-1]) + ', and ' + names[-1]


def substitute_q1(template, old_athlete, new_athlete):
    """Replace the old athlete name (capitalized) with the new one."""
    return re.sub(rf'\b{cap(old_athlete)}\b', cap(new_athlete), template)


def substitute_q2(template, old_athlete, new_athlete, old_match, new_match):
    """Replace athlete name and the ordinal match number."""
    ord_words = {1: 'first', 2: 'second', 3: 'third', 4: 'fourth',
                 5: 'fifth', 6: 'sixth', 7: 'seventh', 8: 'eighth',
                 9: 'ninth', 10: 'tenth'}
    out = re.sub(rf'\b{cap(old_athlete)}\b', cap(new_athlete), template)
    if old_match in ord_words and new_match in ord_words:
        out = re.sub(rf'\b{ord_words[old_match]}\b',
                     ord_words[new_match], out)
    return out


def substitute_q34(template, old_team1, old_team2, new_team1, new_team2):
    """Replace 'X and Y (Team 1) and W and Z (Team 2)' rosters."""
    old_t1 = render_team(old_team1)
    old_t2 = render_team(old_team2)
    new_t1 = render_team(new_team1)
    new_t2 = render_team(new_team2)
    out = template.replace(old_t1, new_t1, 1)
    out = out.replace(old_t2, new_t2, 1)
    return out


def design_new_queries(parsed, raw_queries, rng):
    """Pick strict-different args + render the 4 new query NL strings."""
    athletes = parsed['athletes']
    player_matches = parsed['player_matches']
    main_calls = parsed['main_calls']

    # Original args — assume one call each for Q1/Q2 and two for Q3/Q4 (in
    # the standard 4-query main-block layout).
    if 'intrinsic_strength_rank' not in main_calls:
        return None
    if parsed['episodic_helper'] is None:
        return None
    if 'who_would_win_by_how_much' not in main_calls:
        return None

    q1_args = main_calls['intrinsic_strength_rank'][0]   # [athlete, n_pool]
    q2_args = main_calls[parsed['episodic_helper']][0]   # [athlete, match_id]
    q34 = main_calls['who_would_win_by_how_much']
    if len(q34) < 2:
        return None
    q3_args, q4_args = q34[0], q34[1]                     # [team1, team2]

    if len(raw_queries) != 4:
        return None
    raw_q1, raw_q2, raw_q3, raw_q4 = raw_queries

    # Q1' — different athlete.
    q1_pool = [a for a in athletes if a != q1_args[0]]
    if not q1_pool:
        return None
    new_q1_athlete = rng.choice(q1_pool)

    # Q2' — different (athlete, match) from PLAYER_MATCHES.
    q2_pool = [pm for pm in player_matches
               if (pm[0], pm[1]) != (q2_args[0], q2_args[1])]
    if not q2_pool:
        return None
    new_q2_athlete, new_q2_match = rng.choice(q2_pool)

    # Q3'/Q4' — keep same team sizes; team multisets differ from original.
    def gen_pair(orig_t1, orig_t2, exclude_pair=None):
        n1, n2 = len(orig_t1), len(orig_t2)
        for _ in range(40):
            cand1 = rng.sample(athletes, n1)
            remaining = [a for a in athletes if a not in cand1]
            if len(remaining) < n2:
                continue
            cand2 = rng.sample(remaining, n2)
            t1_set, t2_set = sorted(cand1), sorted(cand2)
            if t1_set == sorted(orig_t1) and t2_set == sorted(orig_t2):
                continue
            if exclude_pair and (t1_set, t2_set) == exclude_pair:
                continue
            return cand1, cand2
        return None

    q3_pair = gen_pair(q3_args[0], q3_args[1])
    if q3_pair is None:
        return None
    excl = (sorted(q3_pair[0]), sorted(q3_pair[1]))
    q4_pair = gen_pair(q4_args[0], q4_args[1], exclude_pair=excl)
    if q4_pair is None:
        return None

    new_q1_nl = substitute_q1(raw_q1, q1_args[0], new_q1_athlete)
    new_q2_nl = substitute_q2(raw_q2, q2_args[0], new_q2_athlete,
                              q2_args[1], new_q2_match)
    new_q3_nl = substitute_q34(raw_q3, q3_args[0], q3_args[1], *q3_pair)
    new_q4_nl = substitute_q34(raw_q4, q4_args[0], q4_args[1], *q4_pair)

    return [
        {'type': 'intrinsic',  'helper': 'intrinsic_strength_rank',
         'args': [new_q1_athlete, q1_args[1]], 'nl': new_q1_nl},
        {'type': 'episodic',   'helper': parsed['episodic_helper'],
         'args': [new_q2_athlete, new_q2_match], 'nl': new_q2_nl},
        {'type': 'future',     'helper': 'who_would_win_by_how_much',
         'args': [list(q3_pair[0]), list(q3_pair[1])], 'nl': new_q3_nl},
        {'type': 'future',     'helper': 'who_would_win_by_how_much',
         'args': [list(q4_pair[0]), list(q4_pair[1])], 'nl': new_q4_nl},
    ]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=25, help='Number of scenarios.')
    p.add_argument('--seed', type=int, default=42, help='RNG seed.')
    args = p.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    all_ids = list_sports_program_ids()
    print(f'Found {len(all_ids)} sports programs.')

    rng = random.Random(args.seed)
    rng.shuffle(all_ids)

    out = []
    n_tried = 0
    for sid in all_ids:
        if len(out) >= args.n:
            break
        n_tried += 1
        try:
            parsed = parse_program(sid)
            scn_text, raw_queries = parse_scenario_txt(sid)
        except (FileNotFoundError, SyntaxError) as e:
            continue
        if not parsed['athletes'] or not parsed['player_matches']:
            continue
        new_queries = design_new_queries(parsed, raw_queries, rng)
        if new_queries is None:
            continue
        out.append({
            'scenario_id': sid,
            'scenario_text': scn_text,
            'athletes': parsed['athletes'],
            'episodic_helper': parsed['episodic_helper'],
            'original_queries_raw': raw_queries,
            'original_args': {
                'q1': parsed['main_calls']['intrinsic_strength_rank'][0],
                'q2': parsed['main_calls'][parsed['episodic_helper']][0],
                'q3': parsed['main_calls']['who_would_win_by_how_much'][0],
                'q4': parsed['main_calls']['who_would_win_by_how_much'][1],
            },
            'new_queries': new_queries,
        })

    print(f'Picked {len(out)} scenarios after trying {n_tried}.')
    out_path = f'{OUT_DIR}/sanity_25sports_queries.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'Wrote {out_path}')


if __name__ == '__main__':
    main()
