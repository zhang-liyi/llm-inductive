"""
Forward sampling script for generating sports scenario training data.

Samples from the joint generative model (mirroring the WebPPL structure in
pg-gemini-canoe.wppl) to produce scenario-answer pairs without MCMC inference.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import logging
import math
import os
import random
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

_LOG_FMT = "%(asctime)s [%(levelname)s] %(message)s"
_LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"

import torch
import torch.distributions as dist

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
os.environ['GEMINI_API_KEY'] = '<GEMINI_API_KEY_PLACEHOLDER>'

client = genai.Client()  # reads API key from env

def run_gemini(prompt, temperature=0, max_tokens=1024, system_prompt='You are a helpful assistant.'):
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

# ── Generate background and query templates via Gemini ────────────────────────

_DIR = os.path.dirname(__file__)
PROMPT_FILE = os.path.join(_DIR, "forward-1.txt")
SUBDOMAINS_FILE = os.path.join(_DIR, "sports-subdomains.txt")


def _parse_query_template(query_raw: str) -> str:
    """Turn a raw query string into a template with {name} and {round} placeholders."""
    # Remove "Query 2: " prefix if present
    query_raw = re.sub(r"^Query\s*\d*:\s*", "", query_raw)

    # Replace <NAME> literal (Gemini is instructed to use it) or detect name
    if "<NAME>" in query_raw:
        template = query_raw.replace("<NAME>", "{name}")
    else:
        name_match = re.search(r"think\s+(\w+)\s+was", query_raw)
        if not name_match:
            raise ValueError(f"Could not find player name in query: {query_raw}")
        template = query_raw.replace(name_match.group(1), "{name}")

    # Replace the round ordinal with {round} placeholder
    round_match = re.search(r"in the (\w+) round", template, re.IGNORECASE)
    if round_match:
        template = template.replace(round_match.group(1), "{round}")

    return template


def generate_scenario_templates(
    N: int, available_subdomains: list[str]
) -> tuple[list[tuple[str, str]], list[str]] | None:
    """Call Gemini to generate 5 (background, query2_template) pairs.

    Passes the full available_subdomains list so Gemini picks 5 itself.
    Returns (templates, chosen_subdomains), or None if the response does not
    contain exactly 5 BACKGROUND and 5 QUERY blocks.
    """
    prompt_text = read_file(PROMPT_FILE)

    prompt_text = prompt_text.replace("<SUBDOMAINS>", ", ".join(available_subdomains))
    prompt_text = prompt_text.replace("<N>", str(N))

    response = run_gemini(prompt_text, temperature=1.0, max_tokens=4096)

    subdomains = re.findall(
        r"<SUBDOMAIN>\s*(.*?)\s*</SUBDOMAIN>", response, re.DOTALL
    )
    backgrounds = re.findall(
        r"<BACKGROUND>\s*(.*?)\s*</BACKGROUND>", response, re.DOTALL
    )
    queries = re.findall(
        r"<QUERY>\s*(.*?)\s*</QUERY>", response, re.DOTALL
    )

    if len(backgrounds) != 5 or len(queries) != 5:
        logging.warning(
            "Round FAILED: expected 5 BACKGROUND and 5 QUERY blocks from Gemini, "
            "got %d and %d. Skipping this round.\nResponse:\n%s",
            len(backgrounds), len(queries), response,
        )
        return None

    # If Gemini didn't return subdomain tags, fall back to empty strings
    if len(subdomains) != 5:
        subdomains = [""] * 5

    templates = []
    for bg, q in zip(backgrounds, queries):
        templates.append((bg.strip(), _parse_query_template(q.strip())))
    return templates, subdomains

NAMES = [
    "alice", "blake", "casey", "drew", "emma", "finn",
    "harper", "iris", "jamie", "kai", "logan", "max",
    "olive", "pat", "quinn", "riley", "sam", "taylor",
    "avery", "jordan", "cameron", "dakota", "joe", "willow",
    "alex", "morgan", "reese", "sage", "skylar", "robin",
    "lane", "kendall", "hayden", "parker", "peyton", "rowan", "elliot",
    "charlie", "lee", "ash", "remy", "brett", "corey", "dana",
    "eden", "jade", "kit", "nova", "uma",
    "august", "clem", "eve", "fern", "glen", "hale", "ira", "june", "kent",
    "nell", "shay", "tate", "ula", "wren", "yael", "zane", "arlo",
    "bryn", "cruz", "emi", "faye", "ines", "joss", "kade", "lars",
    "yuki", "hana", "jin", "mei", "ravi",
    "priya", "suki", "nori", "arjun",
    "amara", "kofi", "zara", "jomo", "nia",
    "kwame", "ada", "seun", "amina",
    "lena", "nora", "vera", "sofia", "aria", "mia",
    "leo", "felix", "theo", "eli", "ivan", "nate",
    "cole", "dean", "seth", "omar", "luca", "marc",
    "reed", "miles", "beau", "demi", "tess", "ruth",
    "gwen", "lyra", "mira", "nadia", "petra", "rosa",
    "tara", "alba", "bea", "dani", "ezra", "flo",
    "hal", "ida", "jude", "kim", "liv", "ned",
    "opal", "rex", "sid", "tom", "val", "bram",
    "cora", "greta", "pia",
]


NUM_CONDITION_MATCHES = 3
NUM_FUTURE_SAMPLES = 100
DEFAULT_TEAM_SIZE = 2


# ── Generative model ──────────────────────────────────────────────────────────

STRENGTH_CATEGORIES = torch.tensor([80.0, 100.0, 120.0])
STRENGTH_PRIORS = torch.tensor([1.0 / 3, 1.0 / 3, 1.0 / 3])
STRENGTH_STD = 10.0

EFFORT_MEANS = torch.tensor([30.0, 60.0, 90.0])
EFFORT_STD = 10.0


def sample_strength() -> float:
    """Sample intrinsic strength for one athlete."""
    cat = dist.Categorical(probs=STRENGTH_PRIORS).sample()
    mean = STRENGTH_CATEGORIES[cat]
    strength = dist.Normal(mean, STRENGTH_STD).sample().item()
    return max(strength, 0.0)


def effort_priors(strength: float) -> torch.Tensor:
    """Strength-dependent effort category priors (low/med/high)."""
    if strength > 110:
        return torch.tensor([0.05, 0.15, 0.80])
    elif strength < 90:
        return torch.tensor([0.60, 0.30, 0.10])
    else:
        return torch.tensor([0.20, 0.50, 0.30])


def sample_effort(strength: float) -> float:
    """Sample effort percentage given intrinsic strength."""
    priors = effort_priors(strength)
    cat = dist.Categorical(probs=priors).sample()
    mean = EFFORT_MEANS[cat]
    effort = dist.Normal(mean, EFFORT_STD).sample().item()
    return max(0.0, min(100.0, effort))


def individual_speed(strength: float, effort: float) -> float:
    return strength * (effort / 100.0)


def team_speed(strengths: list[float], efforts: list[float]) -> float:
    speeds = [individual_speed(s, e) for s, e in zip(strengths, efforts)]
    return sum(speeds) / len(speeds)


def intrinsic_strength_rank(athlete_strength: float, out_of: int = 100) -> int:
    """Rank athlete among `out_of` random athletes (higher = stronger)."""
    count_beaten = 0
    for _ in range(out_of - 1):
        other = sample_strength()
        if athlete_strength > other:
            count_beaten += 1
    return count_beaten


# ── Match & team assignment ───────────────────────────────────────────────────

@dataclass
class Match:
    team1: list[str]
    team2: list[str]
    match_idx: int
    team1_speeds: list[float] = field(default_factory=list)
    team2_speeds: list[float] = field(default_factory=list)
    team1_speed: float = 0.0
    team2_speed: float = 0.0
    team1_wins: bool = False


def pick_teams(athletes: list[str], team_size: int = DEFAULT_TEAM_SIZE) -> tuple[list[str], list[str]]:
    """Pick two non-overlapping teams from the pool."""
    chosen = random.sample(athletes, team_size * 2)
    return chosen[:team_size], chosen[team_size:]


def assign_condition_teams(
    athletes: list[str],
    team_size: int,
    C: int,
    R: int,
    num_matches: int = NUM_CONDITION_MATCHES,
) -> list[tuple[list[str], list[str]]]:
    """Assign teams for condition matches based on C and R motifs.

    C=0: team1 has the same composition across all matches (fixed).
    C=1: team1 composition changes across matches (rotating).
    R=0: team2 (opponents) are fixed across matches.
    R=1: team2 (opponents) rotate each match.
    C=1,R=0: team1 changes each match, but team2 stays fixed.
    C=1,R=1: each team rotates within its own fixed pool (first/second half of athletes).
    """
    if C == 0 and R == 0:
        # Both teams fixed across all matches
        t1, t2 = pick_teams(athletes, team_size)
        return [(list(t1), list(t2))] * num_matches

    elif C == 0 and R == 1:
        # Fixed team1, rotating opponents from remaining pool
        shuffled = list(athletes)
        random.shuffle(shuffled)
        fixed_team = shuffled[:team_size]
        remaining = shuffled[team_size:]
        matches = []
        for _ in range(num_matches):
            opponents = random.sample(remaining, team_size)
            matches.append((list(fixed_team), opponents))
        return matches

    elif C == 1 and R == 1:
        # Each team rotates among its own fixed pool (~1.5N athletes each)
        shuffled = list(athletes)
        random.shuffle(shuffled)
        mid = len(shuffled) // 2
        pool1 = shuffled[:mid]
        pool2 = shuffled[mid:]
        matches = []
        for _ in range(num_matches):
            t1 = random.sample(pool1, team_size)
            t2 = random.sample(pool2, team_size)
            matches.append((t1, t2))
        return matches

    elif C == 1 and R == 0:
        # team1 changes each match, but team2 stays fixed
        shuffled = list(athletes)
        random.shuffle(shuffled)
        fixed_opponents = shuffled[:team_size]
        remaining = shuffled[team_size:]
        matches = []
        for _ in range(num_matches):
            team1 = random.sample(remaining, team_size)
            matches.append((team1, list(fixed_opponents)))
        return matches

    else:
        raise ValueError(f"Invalid motif combination C={C}, R={R}")



def simulate_match(
    team1: list[str],
    team2: list[str],
    match_idx: int,
    strengths: dict[str, float],
    efforts: dict[tuple[str, int], float],
) -> Match:
    """Simulate a match, sampling effort for each athlete."""
    m = Match(team1=team1, team2=team2, match_idx=match_idx)

    for athlete in team1 + team2:
        eff = sample_effort(strengths[athlete])
        efforts[(athlete, match_idx)] = eff

    t1_strengths = [strengths[a] for a in team1]
    t1_efforts = [efforts[(a, match_idx)] for a in team1]
    t2_strengths = [strengths[a] for a in team2]
    t2_efforts = [efforts[(a, match_idx)] for a in team2]

    m.team1_speed = team_speed(t1_strengths, t1_efforts)
    m.team2_speed = team_speed(t2_strengths, t2_efforts)
    m.team1_wins = m.team1_speed > m.team2_speed
    return m


# ── Scenario text construction ────────────────────────────────────────────────

def ordinal(n: int) -> str:
    """1 -> 'first', 2 -> 'second', etc."""
    words = {1: "first", 2: "second", 3: "third", 4: "fourth",
             5: "fifth", 6: "sixth", 7: "seventh", 8: "eighth"}
    return words.get(n, f"{n}th")


def names_str(names: list[str]) -> str:
    """['alice', 'bob'] -> 'Alice and Bob'; ['a','b','c'] -> 'A, B, and C'"""
    titled = [n.title() for n in names]
    if len(titled) == 1:
        return titled[0]
    if len(titled) == 2:
        return " and ".join(titled)
    return ", ".join(titled[:-1]) + ", and " + titled[-1]


def condition_text(match: Match) -> str:
    """Generate a condition sentence for an observed match."""
    rd = ordinal(match.match_idx)
    if match.team1_wins:
        return (f"In the {rd} round, {names_str(match.team1)} beat "
                f"{names_str(match.team2)}.")
    else:
        return (f"In the {rd} round, {names_str(match.team1)} lost to "
                f"{names_str(match.team2)}.")


def build_scenario_text(
    condition_matches: list[Match],
    query1_athlete: str,
    query2_athlete: str,
    query2_match_idx: int,
    query3_match: Match,
    query4_match: Match,
    background_text: str,
    query2_template: str,
) -> str:
    conditions = "\n".join(condition_text(m) for m in condition_matches)

    q1 = (f"Query 1: Out of 100 random athletes, where do you think "
          f"{query1_athlete.title()} ranks in terms of intrinsic skill?")
    q2 = "Query 2: " + query2_template.format(
        name=query2_athlete.title(), round=ordinal(query2_match_idx)
    )
    q3 = (f"Query 3: In a new round later this same day between "
          f"{names_str(query3_match.team1)} (Team 1) and "
          f"{names_str(query3_match.team2)} (Team 2), who would win and by "
          f"how much?")
    q4 = (f"Query 4: In a new round later this same day between "
          f"{names_str(query4_match.team1)} (Team 1) and "
          f"{names_str(query4_match.team2)} (Team 2), who would win and by "
          f"how much?")

    return (
        f"<START_SCENARIO>\n"
        f"BACKGROUND\n{background_text}\n\n"
        f"CONDITIONS\n{conditions}\n\n"
        f"QUERIES\n{q1}\n{q2}\n{q3}\n{q4}\n"
        f"<END_SCENARIO>"
    )


# ── Single forward sample ────────────────────────────────────────────────────


def forward_sample_one(
    C: int,
    R: int,
    N: int,
    background_text: str,
    query2_template: str,
    num_athletes: int = None,
) -> dict:
    """Generate one complete sample from the generative model.

    Team assignments for condition matches are controlled by motifs:
      C: Confounded teammates (0=team1 fixed, 1=team1 rotating).
      R: Round-robin (0=team2 fixed, 1=team2 rotating).
      N: Team size (players per team).
    """
    team_size = N
    min_athletes = team_size * 2
    if num_athletes is None:
        if C == 0 and R == 0:
            num_athletes = 2 * team_size
        elif C + R == 1:
            num_athletes = math.ceil(2.5 * team_size)
        else:  # C == 1 and R == 1
            num_athletes = 3 * team_size
    if num_athletes < min_athletes:
        raise ValueError(
            f"num_athletes ({num_athletes}) must be >= 2 * team_size ({min_athletes})"
        )
    if num_athletes > len(NAMES):
        raise ValueError(
            f"num_athletes ({num_athletes}) exceeds available names ({len(NAMES)})"
        )

    athletes = random.sample(NAMES, num_athletes)

    # Sample intrinsic strengths
    strengths = {a: sample_strength() for a in athletes}
    efforts: dict[tuple[str, int], float] = {}

    # Assign condition match teams based on C, R motifs
    team_assignments = assign_condition_teams(athletes, team_size, C, R)

    # Simulate condition matches
    condition_matches: list[Match] = []
    for i, (t1, t2) in enumerate(team_assignments, start=1):
        m = simulate_match(t1, t2, i, strengths, efforts)
        condition_matches.append(m)

    # Count appearances per athlete across all condition matches
    appearance_counts = Counter(
        a for m in condition_matches for a in m.team1 + m.team2
    )
    max_appearances = max(appearance_counts.values())
    most_frequent = [a for a, c in appearance_counts.items() if c == max_appearances]

    # Query 1: skill rank of the most-frequent condition athlete (random tiebreak)
    query1_athlete = random.choice(most_frequent)
    query1_answer = intrinsic_strength_rank(strengths[query1_athlete])

    # Query 2: effort of the most-frequent condition athlete (random tiebreak),
    # in a random match they appeared in
    q2_athlete = random.choice(most_frequent)
    q2_match = random.choice([m for m in condition_matches
                               if q2_athlete in m.team1 + m.team2])
    query2_answer = efforts[(q2_athlete, q2_match.match_idx)]

    # Query 3 & 4: pick two future matchups, simulate each
    # NUM_FUTURE_SAMPLES times, average outcomes for ground truth.
    q3_t1, q3_t2 = pick_teams(athletes, team_size)
    q4_t1, q4_t2 = pick_teams(athletes, team_size)

    future_efforts: dict[tuple[str, int], float] = {}
    q3_wins = 0
    q4_wins = 0
    for k in range(NUM_FUTURE_SAMPLES):
        midx3 = NUM_CONDITION_MATCHES + 1 + 2 * k
        midx4 = NUM_CONDITION_MATCHES + 2 + 2 * k
        m3 = simulate_match(q3_t1, q3_t2, midx3, strengths, future_efforts)
        m4 = simulate_match(q4_t1, q4_t2, midx4, strengths, future_efforts)
        if not m3.team1_wins:
            q3_wins += 1
        if not m4.team1_wins:
            q4_wins += 1

    query3_match = Match(team1=q3_t1, team2=q3_t2, match_idx=0)
    query4_match = Match(team1=q4_t1, team2=q4_t2, match_idx=0)
    query3_answer = q3_wins / NUM_FUTURE_SAMPLES * 100
    query4_answer = q4_wins / NUM_FUTURE_SAMPLES * 100

    scenario = build_scenario_text(
        condition_matches,
        query1_athlete,
        q2_athlete,
        q2_match.match_idx,
        query3_match,
        query4_match,
        background_text,
        query2_template,
    )

    # Build metadata
    condition_match_metadata = []
    for m in condition_matches:
        condition_match_metadata.append({
            "match_idx": m.match_idx,
            "team1": m.team1,
            "team2": m.team2,
            "team1_speed": round(m.team1_speed, 2),
            "team2_speed": round(m.team2_speed, 2),
            "team1_wins": m.team1_wins,
        })

    effort_metadata = {
        f"{a}_match{mi}": round(e, 2)
        for (a, mi), e in efforts.items()
    }

    return {
        "scenario": scenario,
        "answers": {
            "query1": query1_answer,
            "query2": round(query2_answer, 1),
            "query3": round(query3_answer, 2),
            "query4": round(query4_answer, 2),
        },
        "metadata": {
            "motifs": {"C": C, "R": R, "N": N},
            "athletes": {a: {"strength": round(s, 2)} for a, s in strengths.items()},
            "condition_matches": condition_match_metadata,
            "efforts": effort_metadata,
            "query1_athlete": query1_athlete,
            "query2_athlete": q2_athlete,
            "query2_match_idx": q2_match.match_idx,
            "query3_teams": {"team1": q3_t1, "team2": q3_t2},
            "query4_teams": {"team1": q4_t1, "team2": q4_t2},
            "num_future_samples": NUM_FUTURE_SAMPLES,
            "team_size": team_size,
        },
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Forward sampling for sports scenarios")
    parser.add_argument("--num_samples", type=int, default=1000,
                        help="Number of samples per subdomain")
    parser.add_argument("--C", type=int, required=True, choices=[0, 1],
                        help="Team1 composition: 0=fixed, 1=rotating")
    parser.add_argument("--R", type=int, required=True, choices=[0, 1],
                        help="Team2 composition: 0=fixed, 1=rotating")
    parser.add_argument("--N", type=int, required=True,
                        help="Team size (players per team)")
    parser.add_argument("--num_athletes", type=int, default=None,
                        help="Size of the athlete pool (default: 2N if C=0,R=0; ceil(2.5N) if C+R=1; 3N if C=1,R=1)")
    parser.add_argument("--output", type=str, default=None, help="Output JSONL path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # ── Logging setup ─────────────────────────────────────────────────────────
    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(
        log_dir, f"forward_sample-C{args.C}-R{args.R}-N{args.N}-seed{args.seed}.log"
    )
    formatter = logging.Formatter(_LOG_FMT, datefmt=_LOG_DATEFMT)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setFormatter(formatter)
    root.addHandler(console_handler)
    root.addHandler(file_handler)
    logging.info("Logging to %s", log_path)
    # ──────────────────────────────────────────────────────────────────────────

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # Load all available sports subdomains
    subdomains_raw = read_file(SUBDOMAINS_FILE)
    all_sports = [s.strip() for s in subdomains_raw.split(",") if s.strip()]
    random.shuffle(all_sports)

    NUM_ROUNDS = 4
    SUBDOMAINS_PER_ROUND = 5
    total_subdomains = NUM_ROUNDS * SUBDOMAINS_PER_ROUND  # 20

    if len(all_sports) < total_subdomains:
        raise ValueError(
            f"Need at least {total_subdomains} sports in {SUBDOMAINS_FILE}, "
            f"but only found {len(all_sports)}."
        )

    output_path = args.output or os.path.join(
        os.path.dirname(__file__), "samples",
        f"samples-C{args.C}-R{args.R}-N{args.N}.jsonl",
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Generate templates in 4 rounds of 5 subdomains each (no repeats)
    used_sports: set[str] = set()
    all_templates: list[tuple[str, str]] = []
    failed_rounds = 0

    for round_idx in range(NUM_ROUNDS):
        available = [s for s in all_sports if s not in used_sports]

        logging.info(
            "Round %d/%d: calling Gemini with %d available subdomains...",
            round_idx + 1, NUM_ROUNDS, len(available),
        )
        result = generate_scenario_templates(args.N, available)
        if result is None:
            failed_rounds += 1
            logging.warning(
                "Round %d/%d skipped due to bad Gemini response format "
                "(%d round(s) failed so far).",
                round_idx + 1, NUM_ROUNDS, failed_rounds,
            )
            continue
        templates, chosen = result
        used_sports.update(s for s in chosen if s)
        logging.info(
            "Round %d/%d: got %d templates. Gemini chose: %s",
            round_idx + 1, NUM_ROUNDS, len(templates),
            ", ".join(c for c in chosen if c),
        )
        all_templates.extend(templates)

    actual_subdomains = len(all_templates)
    logging.info(
        "Template generation complete: %d subdomains across %d successful round(s) "
        "(%d round(s) failed).",
        actual_subdomains, NUM_ROUNDS - failed_rounds, failed_rounds,
    )

    total = 0
    with open(output_path, "w") as f:
        for t_idx, (bg, q2_tmpl) in enumerate(all_templates):
            logging.info(
                "Subdomain %d/%d: generating %d samples...",
                t_idx + 1, actual_subdomains, args.num_samples,
            )
            for i in range(args.num_samples):
                sample = forward_sample_one(
                    C=args.C, R=args.R, N=args.N,
                    background_text=bg,
                    query2_template=q2_tmpl,
                    num_athletes=args.num_athletes,
                )
                sample["metadata"]["subdomain_idx"] = t_idx
                f.write(json.dumps(sample) + "\n")
                total += 1
                if total % 100 == 0:
                    logging.info("  Generated %d samples total", total)

    logging.info(
        "Wrote %d samples (%d x %d subdomains) to %s",
        total, args.num_samples, actual_subdomains, output_path,
    )


if __name__ == "__main__":
    main()
