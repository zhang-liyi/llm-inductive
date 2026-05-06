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
ATHLETES = ['kay', 'ollie', 'willow', 'max']

# Temperature for soft lost likelihoods.
# Team pulling force is O(~100); BEAT_TEMP=10.0 makes the soft boundary
# sharp but smooth.
BEAT_TEMP = 10.0

# ── Shared strength prior (used in model and query helpers) ───────────────────
# The scratchpad suggests a normal distribution with mean 100 and std 15 units.
_STRENGTH_PRIOR = dist.Normal(torch.tensor(100.), torch.tensor(15.))

# ── Pyro model ────────────────────────────────────────────────────────────────

def model():
    # ── intrinsic_strength (mem'd per athlete) ───────────────────────────────
    intrinsic_strength = {}
    for athlete in ATHLETES:
        raw = pyro.sample(f"intrinsic_strength_{athlete}", _STRENGTH_PRIOR)
        # Clamp strength to be non-negative
        intrinsic_strength[athlete] = raw.clamp(min=0.)

    # ── athlete_effort_in_match (mem'd per athlete × match) ──────────────────
    # The scratchpad specifies effort is drawn from a uniform distribution between 0 and 1.
    athlete_effort_in_match = {}
    for athlete in ATHLETES:
        for match_id in [1, 2, 3]:
            effort = pyro.sample(f"effort_{athlete}_m{match_id}", dist.Uniform(0., 1.))
            athlete_effort_in_match[(athlete, match_id)] = effort

    # ── Derived quantities (deterministic given samples) ──────────────────────

    def athlete_pulling_force_in_match(athlete, match_id):
        # Pulling force = intrinsic strength * effort
        return intrinsic_strength[athlete] * athlete_effort_in_match[(athlete, match_id)]

    def team_pulling_force_in_match(team, match_id):
        # Team's collective pulling force is the sum of individual pulling forces
        return sum(athlete_pulling_force_in_match(a, match_id) for a in team)

    # ── Conditions (soft likelihoods) ─────────────────────────────────────────
    # lost(team1, team2, match) → score_diff < 0 → factor(logsigmoid(-diff/T))

    def soft_lost(team1, team2, match_id, name):
        score_diff = team_pulling_force_in_match(team1, match_id) - team_pulling_force_in_match(team2, match_id)
        pyro.factor(name, F.logsigmoid(-score_diff / BEAT_TEMP))

    # In the first match, Kay and Ollie lost to Willow and Max.
    soft_lost(['kay', 'ollie'], ['willow', 'max'], 1, 'cond_1')
    # In the second match, Kay and Willow lost to Ollie and Max.
    soft_lost(['kay', 'willow'], ['ollie', 'max'], 2, 'cond_2')
    # In the third match, Kay and Max lost to Ollie and Willow.
    soft_lost(['kay', 'max'], ['ollie', 'willow'], 3, 'cond_3')

# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference(num_samples=500, warmup_steps=200):
    kernel = NUTS(model, adapt_step_size=True, target_accept_prob=0.8)
    mcmc   = MCMC(kernel, num_samples=num_samples, warmup_steps=warmup_steps, num_chains=4)
    mcmc.run()
    return mcmc

# ── Query helpers ─────────────────────────────────────────────────────────────

def intrinsic_strength_rank(samples, athlete, out_of_n_athletes=100):
    """
    Queries 1-3: Out of 100 random athletes, where does athlete rank?
    """
    s_post = samples[f'intrinsic_strength_{athlete}'].double()
    others = _STRENGTH_PRIOR.sample((s_post.shape[0], out_of_n_athletes - 1)).clamp(min=0.)
    rank   = (s_post.unsqueeze(1) > others).double().sum(dim=1)
    return dict(mean=rank.mean().item(), std=rank.std().item(),
                p10=rank.quantile(0.10).item(), p90=rank.quantile(0.90).item(),
                raw=rank.tolist())

def query_athlete_effort_in_match(samples, athlete, match_id):
    """
    Queries 4-6: Effort put into a match on a 0-100% scale.
    """
    key = f"effort_{athlete}_m{match_id}"
    effort_pct = samples[key].double() * 100.
    return dict(mean=effort_pct.mean().item(), std=effort_pct.std().item(),
                p10=effort_pct.quantile(0.10).item(), p90=effort_pct.quantile(0.90).item(),
                raw=effort_pct.tolist())

def who_would_win_by_how_much(samples, team1, team2, n_future=100):
    """
    Queries 7-8: P(team2 wins) over simulated future matches.
    New athletes (e.g. emery) are drawn fresh from the strength prior.
    """
    all_players = list(set(team1 + team2))
    n_post      = samples[f'intrinsic_strength_{ATHLETES[0]}'].shape[0]
    p_team2_wins = []

    for i in range(n_post):
        strength = {}
        for a in all_players:
            key = f'intrinsic_strength_{a}'
            strength[a] = (samples[key][i].double() if key in samples
                           else _STRENGTH_PRIOR.sample().clamp(min=0.))

        wins = 0
        for _ in range(n_future):
            def sim_force(team):
                force = 0.
                for a in team:
                    effort = dist.Uniform(0., 1.).sample()
                    force += strength[a] * effort
                return force

            if sim_force(team2) > sim_force(team1):
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

    print("Running NUTS inference on tug-of-war model …")
    mcmc    = run_inference(num_samples=500, warmup_steps=200)
    samples = mcmc.get_samples()
    mcmc.summary()

    print("\n=== Query 1: Kay's strength rank out of 100 athletes ===")
    q1 = intrinsic_strength_rank(samples, 'kay', 100)
    print(f"  mean={q1['mean']:.1f}  std={q1['std']:.1f}  [p10={q1['p10']:.1f}, p90={q1['p90']:.1f}]")

    print("\n=== Query 2: Ollie's strength rank out of 100 athletes ===")
    q2 = intrinsic_strength_rank(samples, 'ollie', 100)
    print(f"  mean={q2['mean']:.1f}  std={q2['std']:.1f}  [p10={q2['p10']:.1f}, p90={q2['p90']:.1f}]")

    print("\n=== Query 3: Willow's strength rank out of 100 athletes ===")
    q3 = intrinsic_strength_rank(samples, 'willow', 100)
    print(f"  mean={q3['mean']:.1f}  std={q3['std']:.1f}  [p10={q3['p10']:.1f}, p90={q3['p90']:.1f}]")

    print("\n=== Query 4: Kay's effort in match 2 (0-100%) ===")
    q4 = query_athlete_effort_in_match(samples, 'kay', 2)
    print(f"  mean={q4['mean']:.1f}%  std={q4['std']:.1f}%  [p10={q4['p10']:.1f}%, p90={q4['p90']:.1f}%]")

    print("\n=== Query 5: Ollie's effort in match 2 (0-100%) ===")
    q5 = query_athlete_effort_in_match(samples, 'ollie', 2)
    print(f"  mean={q5['mean']:.1f}%  std={q5['std']:.1f}%  [p10={q5['p10']:.1f}%, p90={q5['p90']:.1f}%]")

    print("\n=== Query 6: Willow's effort in match 2 (0-100%) ===")
    q6 = query_athlete_effort_in_match(samples, 'willow', 2)
    print(f"  mean={q6['mean']:.1f}%  std={q6['std']:.1f}%  [p10={q6['p10']:.1f}%, p90={q6['p90']:.1f}%]")

    print("\n=== Query 7: P(Willow+Emery beat Kay+Ollie) in a future match ===")
    q7 = who_would_win_by_how_much(samples, ['kay', 'ollie'], ['willow', 'emery'])
    print(f"  P(team2 wins) = {q7['p_team2_wins']:.3f}  [p10={q7['p10']:.3f}, p90={q7['p90']:.3f}]")

    print("\n=== Query 8: P(Ollie+Emery beat Kay+Willow) in a future match ===")
    q8 = who_would_win_by_how_much(samples, ['kay', 'willow'], ['ollie', 'emery'])
    print(f"  P(team2 wins) = {q8['p_team2_wins']:.3f}  [p10={q8['p10']:.3f}, p90={q8['p90']:.3f}]")

    # ── Save full posterior distributions ─────────────────────────────────────
    tm = datetime.datetime.now()
    result = {
        'query1': {'description': "Kay's strength rank out of 100 athletes",
                   'samples': q1['raw'], 'mean': q1['mean'], 'std': q1['std']},
        'query2': {'description': "Ollie's strength rank out of 100 athletes",
                   'samples': q2['raw'], 'mean': q2['mean'], 'std': q2['std']},
        'query3': {'description': "Willow's strength rank out of 100 athletes",
                   'samples': q3['raw'], 'mean': q3['mean'], 'std': q3['std']},
        'query4': {'description': "Kay's effort in match 2 (0-100%)",
                   'samples': q4['raw'], 'mean': q4['mean'], 'std': q4['std']},
        'query5': {'description': "Ollie's effort in match 2 (0-100%)",
                   'samples': q5['raw'], 'mean': q5['mean'], 'std': q5['std']},
        'query6': {'description': "Willow's effort in match 2 (0-100%)",
                   'samples': q6['raw'], 'mean': q6['mean'], 'std': q6['std']},
        'query7': {'description': 'P(Willow+Emery beat Kay+Ollie)',
                   'samples': q7['raw'], 'mean': q7['p_team2_wins']},
        'query8': {'description': 'P(Ollie+Emery beat Kay+Willow)',
                   'samples': q8['raw'], 'mean': q8['p_team2_wins']},
    }
    out_path = f'inference_results/result-tug-of-war-pytorch.json'
    os.makedirs('inference_results', exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(result, f)