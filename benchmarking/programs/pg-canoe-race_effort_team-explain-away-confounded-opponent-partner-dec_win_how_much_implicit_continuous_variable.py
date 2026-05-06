import argparse
import json
import datetime
import os
import torch
import torch.nn.functional as F
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS

torch.set_default_dtype(torch.float64)

# ── Athletes that appear in the conditions ────────────────────────────────────
ATHLETES = ['gale', 'robin', 'blake', 'kay', 'willow', 'harper', 'drew']

# ── (athlete, race) pairs from the conditions ─────────────────────────────────
ATHLETE_RACES = [
    ('gale', 1), ('robin', 1), ('blake', 1), ('kay', 1),
    ('gale', 2), ('robin', 2), ('blake', 2), ('kay', 2),
    ('gale', 3), ('robin', 3), ('blake', 3), ('willow', 3),
    ('gale', 4), ('robin', 4), ('blake', 4), ('harper', 4),
    ('gale', 5), ('robin', 5), ('blake', 5), ('drew', 5),
]

# Temperature for soft beat/lost likelihoods.
# team_speed_in_race is O(~37.5); BEAT_TEMP=1.0 makes the soft boundary
# sharp (sigmoid(±10) ≈ 0/1) but everywhere-finite and smooth.
BEAT_TEMP = 1.0

# ── Shared priors (used in model and query helpers) ───────────────────────────
# Intrinsic strength is normally distributed with mean 50 and std 15.
_STRENGTH_PRIOR = dist.Normal(50., 15.)
# Effort is normally distributed around 75% with std 15%.
_EFFORT_PRIOR = dist.Normal(75., 15.)

# ── Pyro model ────────────────────────────────────────────────────────────────

def model():
    # ── intrinsic_strength (mem'd per athlete) ───────────────────────────────
    intrinsic_strength = {}
    for athlete in ATHLETES:
        raw = pyro.sample(f"intrinsic_strength_{athlete}", _STRENGTH_PRIOR)
        intrinsic_strength[athlete] = raw.clamp(min=0.)

    # ── athlete_effort_in_race (mem'd per athlete × race) ────────────────────
    athlete_effort_in_race = {}
    for (athlete, race_id) in ATHLETE_RACES:
        raw = pyro.sample(f"athlete_effort_{athlete}_r{race_id}", _EFFORT_PRIOR)
        athlete_effort_in_race[(athlete, race_id)] = raw.clamp(0., 100.)

    # ── Derived quantities (deterministic given samples) ──────────────────────
    def athlete_speed_in_race(athlete, race_id):
        strength = intrinsic_strength[athlete]
        effort = athlete_effort_in_race[(athlete, race_id)]
        return strength * (effort / 100.)

    def team_speed_in_race(team, race_id):
        return sum(athlete_speed_in_race(a, race_id) for a in team) / len(team)

    # ── Conditions (soft likelihoods) ─────────────────────────────────────────
    def soft_beat(team1, team2, race_id, name):
        speed_diff = team_speed_in_race(team1, race_id) - team_speed_in_race(team2, race_id)
        pyro.factor(name, F.logsigmoid(speed_diff / BEAT_TEMP))

    def soft_lost(team1, team2, race_id, name):
        speed_diff = team_speed_in_race(team1, race_id) - team_speed_in_race(team2, race_id)
        pyro.factor(name, F.logsigmoid(-speed_diff / BEAT_TEMP))

    # In the first race, Gale and Robin beat Blake and Kay.
    soft_beat(['gale', 'robin'], ['blake', 'kay'], 1, 'cond_1')
    # In the second race, Gale and Robin lost to Blake and Kay.
    soft_lost(['gale', 'robin'], ['blake', 'kay'], 2, 'cond_2')
    # In the third race, Gale and Robin beat Blake and Willow.
    soft_beat(['gale', 'robin'], ['blake', 'willow'], 3, 'cond_3')
    # In the fourth race, Gale and Robin beat Blake and Harper.
    soft_beat(['gale', 'robin'], ['blake', 'harper'], 4, 'cond_4')
    # In the fifth race, Gale and Robin beat Blake and Drew.
    soft_beat(['gale', 'robin'], ['blake', 'drew'], 5, 'cond_5')

# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference(num_samples=500, warmup_steps=200):
    kernel = NUTS(model, adapt_step_size=True, target_accept_prob=0.8)
    mcmc   = MCMC(kernel, num_samples=num_samples, warmup_steps=warmup_steps, num_chains=4)
    mcmc.run()
    return mcmc

# ── Query helpers (post-hoc) ──────────────────────────────────────────────────

def intrinsic_strength_rank(samples, athlete='gale', out_of_n_athletes=100):
    """
    Queries 1-3: Out of 100 random athletes, where does athlete rank?
    """
    s_post = samples[f'intrinsic_strength_{athlete}'].double()
    others = _STRENGTH_PRIOR.sample((s_post.shape[0], out_of_n_athletes - 1)).clamp(min=0.)
    rank   = (s_post.unsqueeze(1) > others).double().sum(dim=1)
    return dict(mean=rank.mean().item(), std=rank.std().item(),
                p10=rank.quantile(0.10).item(), p90=rank.quantile(0.90).item(),
                raw=rank.tolist())

def query_athlete_effort_in_race(samples, athlete='gale', race_id=2):
    """
    Queries 4-6: Effort for an athlete in a given race (0–100 scale).
    """
    key = f"athlete_effort_{athlete}_r{race_id}"
    e = samples[key].double().clamp(0., 100.)
    return dict(mean=e.mean().item(), std=e.std().item(),
                p10=e.quantile(0.10).item(), p90=e.quantile(0.90).item(),
                raw=e.tolist())

def who_would_win_by_how_much(samples, team1, team2, n_future=100):
    """
    Queries 7 & 8: P(team2 wins) over simulated future races.
    New athletes are drawn fresh from the strength prior.
    """
    all_players = list(set(team1 + team2))
    n_post      = samples[f'intrinsic_strength_{ATHLETES[0]}'].shape[0]
    p_team2_wins = []

    for i in range(n_post):
        # Use posterior strength for known athletes; prior for unknowns
        strength = {}
        for a in all_players:
            key = f'intrinsic_strength_{a}'
            strength[a] = (samples[key][i].double() if key in samples
                           else _STRENGTH_PRIOR.sample().clamp(min=0.))

        wins = 0
        for _ in range(n_future):
            def sim_speed(team):
                speeds = []
                for a in team:
                    eff = _EFFORT_PRIOR.sample().clamp(0., 100.)
                    speeds.append(strength[a] * (eff / 100.))
                return sum(speeds) / len(team)

            if sim_speed(team2) > sim_speed(team1):
                wins += 1

        p_team2_wins.append(wins / n_future)

    p2 = torch.tensor(p_team2_wins)
    return dict(p_team2_wins=p2.mean().item(),
                p10=p2.quantile(0.10).item(), p90=p2.quantile(0.90).item(),
                raw=p2.tolist())

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    args = parser.parse_args()
    pyro.set_rng_seed(args.seed)

    print("Running NUTS inference on canoe racing model …")
    mcmc    = run_inference(num_samples=500, warmup_steps=200)
    samples = mcmc.get_samples()
    mcmc.summary()

    print("\n=== Query 1: Gale's strength rank out of 100 athletes ===")
    q1 = intrinsic_strength_rank(samples, 'gale', 100)
    print(f"  mean={q1['mean']:.1f}  std={q1['std']:.1f}  "
          f"[p10={q1['p10']:.1f}, p90={q1['p90']:.1f}]")

    print("\n=== Query 2: Robin's strength rank out of 100 athletes ===")
    q2 = intrinsic_strength_rank(samples, 'robin', 100)
    print(f"  mean={q2['mean']:.1f}  std={q2['std']:.1f}  "
          f"[p10={q2['p10']:.1f}, p90={q2['p90']:.1f}]")

    print("\n=== Query 3: Blake's strength rank out of 100 athletes ===")
    q3 = intrinsic_strength_rank(samples, 'blake', 100)
    print(f"  mean={q3['mean']:.1f}  std={q3['std']:.1f}  "
          f"[p10={q3['p10']:.1f}, p90={q3['p90']:.1f}]")

    print("\n=== Query 4: Gale's effort in race 2 (0-100) ===")
    q4 = query_athlete_effort_in_race(samples, 'gale', 2)
    print(f"  mean={q4['mean']:.1f}  std={q4['std']:.1f}  "
          f"[p10={q4['p10']:.1f}, p90={q4['p90']:.1f}]")

    print("\n=== Query 5: Robin's effort in race 2 (0-100) ===")
    q5 = query_athlete_effort_in_race(samples, 'robin', 2)
    print(f"  mean={q5['mean']:.1f}  std={q5['std']:.1f}  "
          f"[p10={q5['p10']:.1f}, p90={q5['p90']:.1f}]")

    print("\n=== Query 6: Blake's effort in race 2 (0-100) ===")
    q6 = query_athlete_effort_in_race(samples, 'blake', 2)
    print(f"  mean={q6['mean']:.1f}  std={q6['std']:.1f}  "
          f"[p10={q6['p10']:.1f}, p90={q6['p90']:.1f}]")

    print("\n=== Query 7: P(Blake+Kay beat Gale+Robin) in a future race ===")
    q7 = who_would_win_by_how_much(samples, ['gale', 'robin'], ['blake', 'kay'])
    print(f"  P(team2 wins) = {q7['p_team2_wins']:.3f}  "
          f"[p10={q7['p10']:.3f}, p90={q7['p90']:.3f}]")

    print("\n=== Query 8: P(Kay+Willow beat Gale+Robin) in a future race ===")
    q8 = who_would_win_by_how_much(samples, ['gale', 'robin'], ['kay', 'willow'])
    print(f"  P(team2 wins) = {q8['p_team2_wins']:.3f}  "
          f"[p10={q8['p10']:.3f}, p90={q8['p90']:.3f}]")

    # ── Save full posterior distributions ─────────────────────────────────────
    tm = datetime.datetime.now()
    result = {
        'query1': {'description': "Gale's strength rank out of 100 athletes",
                   'samples': q1['raw'], 'mean': q1['mean'], 'std': q1['std']},
        'query2': {'description': "Robin's strength rank out of 100 athletes",
                   'samples': q2['raw'], 'mean': q2['mean'], 'std': q2['std']},
        'query3': {'description': "Blake's strength rank out of 100 athletes",
                   'samples': q3['raw'], 'mean': q3['mean'], 'std': q3['std']},
        'query4': {'description': "Gale's effort in race 2 (0-100)",
                   'samples': q4['raw'], 'mean': q4['mean'], 'std': q4['std']},
        'query5': {'description': "Robin's effort in race 2 (0-100)",
                   'samples': q5['raw'], 'mean': q5['mean'], 'std': q5['std']},
        'query6': {'description': "Blake's effort in race 2 (0-100)",
                   'samples': q6['raw'], 'mean': q6['mean'], 'std': q6['std']},
        'query7': {'description': 'P(Blake+Kay beat Gale+Robin)',
                   'samples': q7['raw'], 'mean': q7['p_team2_wins']},
        'query8': {'description': 'P(Kay+Willow beat Gale+Robin)',
                   'samples': q8['raw'], 'mean': q8['p_team2_wins']},
    }
    out_path = f'inference_results/result-canoe-pytorch.json'
    os.makedirs('inference_results', exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(result, f)