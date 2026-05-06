import argparse
import json
import datetime
import os
import torch
import torch.nn.functional as F
import pyro
import pyro.distributions as dist
import math as _math
import random as _random
from pyro.poutine import trace as _pt_trace
from pyro.distributions import Unit as _Unit

torch.set_default_dtype(torch.float64)

# ── Athletes that appear in the conditions ────────────────────────────────────
ATHLETES = ['elena', 'sofia', 'chloe', 'maya', 'zoe', 'harper', 'lily']

# ── (team1, team2, match_id) pairs from the conditions, in original order ─────
MATCHES = [
    (['elena', 'sofia'], ['chloe', 'maya'], 1),
    (['zoe', 'harper'],  ['chloe', 'lily'], 2),
    (['elena', 'sofia'], ['zoe', 'harper'], 3),
    (['chloe', 'harper'],['maya', 'lily'],  4),
    (['elena', 'sofia'], ['chloe', 'harper'], 5),
]

# Temperature for soft beat likelihoods.
# Performance differences are typically O(~10-50); BEAT_TEMP=10.0 makes the soft boundary
# sharp but everywhere-finite and smooth.
BEAT_TEMP = 10.0

# ── Shared skill prior (used in model and query helpers) ──────────────────────
_SKILL_PRIOR = dist.MixtureSameFamily(
    dist.Categorical(probs=torch.tensor([0.33, 0.33, 0.34])),
    dist.Normal(torch.tensor([80., 100., 140.]), torch.tensor([10., 10., 10.])),
)

# ── Pyro model ────────────────────────────────────────────────────────────────

def model():
    # ── Latents (mem'd per athlete) ───────────────────────────────────────────
    intrinsic_skill = {}
    intrinsic_endurance = {}
    player_wind_adaptability = {}
    
    for athlete in ATHLETES:
        # Intrinsic skill: continuous value drawn from a normal mixture distribution
        s = pyro.sample(f"intrinsic_skill_{athlete}", _SKILL_PRIOR)
        intrinsic_skill[athlete] = s.clamp(min=0.)
        
        # Intrinsic endurance: 0 to 100 scale (higher is better)
        e = pyro.sample(f"intrinsic_endurance_{athlete}", dist.Normal(50.0, 15.0))
        intrinsic_endurance[athlete] = e.clamp(0., 100.)
        
        # Wind adaptability: multiplier for wind intensity (can be positive or negative)
        w = pyro.sample(f"player_wind_adaptability_{athlete}", dist.Normal(0.0, 10.0))
        player_wind_adaptability[athlete] = w

    # ── Derived quantities (deterministic given samples) ──────────────────────

    def matches_played_before(player, match_id):
        count = 0
        for (t1, t2, m_id) in MATCHES:
            if m_id < match_id:
                if player in t1 or player in t2:
                    count += 1
        return count

    def wind_intensity_in_match(match_id):
        # Wind intensity scale: 0.0 (calm), 1.0 (moderate), 2.0 (heavy/howling)
        if match_id in [1, 2]:
            return 0.0
        elif match_id == 3:
            return 1.0
        elif match_id in [4, 5]:
            return 2.0
        return 0.0

    def player_fatigue_in_match(player, match_id):
        played = matches_played_before(player, match_id)
        endurance = intrinsic_endurance[player]
        # Fatigue scale 0 to 100
        # Base fatigue per match is mitigated by endurance
        fatigue = played * (100.0 - endurance) * 0.5
        return fatigue.clamp(0., 100.)

    def player_performance_in_match(player, match_id):
        skill = intrinsic_skill[player]
        fatigue = player_fatigue_in_match(player, match_id)
        wind = wind_intensity_in_match(match_id)
        adaptability = player_wind_adaptability[player]
        
        return skill - fatigue + wind * adaptability

    def team_performance_in_match(team, match_id):
        return sum(player_performance_in_match(p, match_id) for p in team)

    # ── Conditions (soft likelihoods) ─────────────────────────────────────────
    # beat(team1, team2, match)  →  score_diff > 0  →  factor(logsigmoid(+diff/T))

    def soft_beat(team1, team2, match_id, name):
        score_diff = team_performance_in_match(team1, match_id) - team_performance_in_match(team2, match_id)
        pyro.factor(name, F.logsigmoid(score_diff / BEAT_TEMP))

    # 9:00 AM: In perfect, calm conditions, the established duo of Elena and Sofia opened the day by comfortably dispatching Chloe and Maya.
    soft_beat(['elena', 'sofia'], ['chloe', 'maya'], 1, 'cond_1')
    # 11:00 AM: As the sand began to heat up, Zoe and Harper teamed up to narrowly edge out Chloe and Lily in a marathon three-setter.
    soft_beat(['zoe', 'harper'], ['chloe', 'lily'], 2, 'cond_2')
    # 1:00 PM: With the coastal winds starting to pick up, Elena and Sofia returned to the court and comfortably swept Zoe and Harper.
    soft_beat(['elena', 'sofia'], ['zoe', 'harper'], 3, 'cond_3')
    # 3:00 PM: With the wind now howling, Chloe and Harper formed a new pair and pulled off a shocking blowout, defeating Maya and Lily by a massive margin. Many spectators wondered if the heavy wind perfectly favored Chloe's signature low-trajectory serves, or if Maya and Lily simply lacked the communication to handle the chaotic gusts.
    soft_beat(['chloe', 'harper'], ['maya', 'lily'], 4, 'cond_4')
    # 5:00 PM: An exhausted Elena and Sofia faced the red-hot Chloe and Harper. Despite visibly lagging and struggling to chase down deep shots in the heavy wind, Elena and Sofia managed to come from behind and win narrowly in the final few points.
    soft_beat(['elena', 'sofia'], ['chloe', 'harper'], 5, 'cond_5')


# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference(num_samples=1000, max_attempts=10_000_000):
    """Importance-weighted rejection sampling.

    Target posterior = prior * prod_i sigmoid(diff_i / BEAT_TEMP), exactly the
    distribution the REJ + pyro.factor(logsigmoid(...)) formulation targets.
    Procedure: draw from the prior (via a model trace), sum the log-factor
    values at every Unit site (= sum logsigmoid(diff/T) <= 0), and accept with
    probability exp(sum). Accepted samples are i.i.d. from the soft posterior.
    Single chain, 1000 accepted samples.
    """
    kept = {}
    accepted = 0
    attempts = 0
    while accepted < num_samples and attempts < max_attempts:
        attempts += 1
        tr = _pt_trace(model).get_trace()
        log_accept = 0.0
        for _name, _node in tr.nodes.items():
            if _node.get('type') != 'sample':
                continue
            _fn = _node.get('fn')
            if isinstance(_fn, _Unit):
                _lf = _fn.log_factor
                if hasattr(_lf, 'item'):
                    _lf = _lf.item()
                log_accept += _lf
        if _math.log(_random.random()) >= log_accept:
            continue
        for _name, _node in tr.nodes.items():
            if _node.get('type') != 'sample':
                continue
            if isinstance(_node.get('fn'), _Unit):
                continue
            kept.setdefault(_name, []).append(_node['value'])
        accepted += 1
    if accepted < num_samples:
        print(f"[WARN] rejection sampler: only {accepted}/{num_samples} "
              f"accepted after {attempts} attempts.")
    return {k: torch.stack(v) for k, v in kept.items()}
# ── Query helpers (post-hoc, matching WebPPL queries) ────────────────────────

def intrinsic_skill_rank(samples, athlete='sofia', out_of_n_athletes=100):
    """
    Query 1: Out of 100 random competitive beach volleyball players, where do you think Sofia ranks in terms of intrinsic skill?
    """
    s_post = samples[f'intrinsic_skill_{athlete}'].double()
    others = _SKILL_PRIOR.sample((s_post.shape[0], out_of_n_athletes - 1)).clamp(min=0.)
    rank   = (s_post.unsqueeze(1) > others).double().sum(dim=1)
    return dict(mean=rank.mean().item(), std=rank.std().item(),
                p10=rank.quantile(0.10).item(), p90=rank.quantile(0.90).item(),
                raw=rank.tolist())


def query_player_fatigue_in_match(samples, player='elena', match_id=5):
    """
    Query 2: On a scale from 0 to 100, what was Elena's likely fatigue level during her 5:00 PM match?
    """
    played = 0
    for (t1, t2, m_id) in MATCHES:
        if m_id < match_id:
            if player in t1 or player in t2:
                played += 1
    
    endurance = samples[f'intrinsic_endurance_{player}'].double()
    fatigue = (played * (100.0 - endurance) * 0.5).clamp(0., 100.)
    
    return dict(mean=fatigue.mean().item(), std=fatigue.std().item(),
                p10=fatigue.quantile(0.10).item(), p90=fatigue.quantile(0.90).item(),
                raw=fatigue.tolist())


def probability_team_wins(samples, team1, team2):
    """
    Queries 3 & 4: Probability (from 0 to 100) that Team 1 wins in a hypothetical match.
    Both queries specify calm weather/no wind and fully rested players (fatigue=0).
    """
    all_players = list(set(team1 + team2))
    n_post      = samples[f'intrinsic_skill_{ATHLETES[0]}'].shape[0]
    p_team1_wins = []

    for i in range(n_post):
        # Use posterior skill for known athletes; prior for unknowns
        skill = {}
        for a in all_players:
            key = f'intrinsic_skill_{a}'
            skill[a] = (samples[key][i].double() if key in samples
                        else _SKILL_PRIOR.sample().clamp(min=0.))

        # With fatigue=0 and wind=0, performance is just the sum of intrinsic skills
        perf1 = sum(skill[a] for a in team1)
        perf2 = sum(skill[a] for a in team2)
        
        if perf1 > perf2:
            p_team1_wins.append(1.0)
        else:
            p_team1_wins.append(0.0)

    p1 = torch.tensor(p_team1_wins)
    return dict(p_team1_wins=p1.mean().item() * 100.0,
                raw=(p1 * 100.0).tolist())


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    args = parser.parse_args()
    pyro.set_rng_seed(args.seed)

    print("Running importance-weighted rejection on beach volleyball model (1000 accepted samples) ...")
    samples = run_inference(num_samples=1000)
    print(f"  num accepted = {next(iter(samples.values())).shape[0]}")

    print("\n=== Query 1: Sofia's skill rank out of 100 players ===")
    q1 = intrinsic_skill_rank(samples, 'sofia', 100)
    print(f"  mean={q1['mean']:.1f}  std={q1['std']:.1f}  "
          f"[p10={q1['p10']:.1f}, p90={q1['p90']:.1f}]")

    print("\n=== Query 2: Elena's fatigue level in match 5 (0-100) ===")
    q2 = query_player_fatigue_in_match(samples, 'elena', 5)
    print(f"  mean={q2['mean']:.1f}  std={q2['std']:.1f}  "
          f"[p10={q2['p10']:.1f}, p90={q2['p90']:.1f}]")

    print("\n=== Query 3: P(Chloe+Maya beat Zoe+Lily) in calm weather ===")
    q3 = probability_team_wins(samples, ['chloe', 'maya'], ['zoe', 'lily'])
    print(f"  P(Team 1 wins) = {q3['p_team1_wins']:.1f}%")

    print("\n=== Query 4: P(Elena+Sofia beat Chloe+Harper) indoors fully rested ===")
    q4 = probability_team_wins(samples, ['elena', 'sofia'], ['chloe', 'harper'])
    print(f"  P(Team 1 wins) = {q4['p_team1_wins']:.1f}%")

    # ── Save full posterior distributions ─────────────────────────────────────
    tm = datetime.datetime.now()
    result = {
        'query1': {'description': "Sofia's rank out of 100 players",
                   'samples': q1['raw'], 'mean': q1['mean'], 'std': q1['std']},
        'query2': {'description': "Elena's fatigue in match 5",
                   'samples': q2['raw'], 'mean': q2['mean'], 'std': q2['std']},
        'query3': {'description': 'P(Chloe+Maya beat Zoe+Lily)',
                   'samples': q3['raw'], 'mean': q3['p_team1_wins']},
        'query4': {'description': 'P(Elena+Sofia beat Chloe+Harper)',
                   'samples': q4['raw'], 'mean': q4['p_team1_wins']},
    }
    os.makedirs('inference_results', exist_ok=True)
    out_path = "inference_results/result-gemini-REJ-diverse-P-0-C-0-R-0-N-0-0-2026-04-18-165011.json"
    with open(out_path, 'w') as f:
        json.dump(result, f)