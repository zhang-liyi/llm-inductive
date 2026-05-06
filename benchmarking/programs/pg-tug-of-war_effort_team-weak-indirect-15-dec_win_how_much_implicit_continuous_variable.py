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
ATHLETES = ['drew', 'lane', 'casey', 'robin', 'indiana', 'avery']

# ── (athlete, match) pairs from the conditions ────────────────────────────────
ATHLETE_MATCHES = [
    ('drew', 1), ('lane', 1), ('casey', 1), ('robin', 1),
    ('lane', 2), ('indiana', 2), ('casey', 2), ('robin', 2),
    ('lane', 3), ('avery', 3), ('casey', 3), ('robin', 3)
]

# Temperature for soft beat/lost likelihoods.
# team_pulling_force is O(~160); BEAT_TEMP=10.0 makes the soft boundary
# sharp but everywhere-finite and smooth.
BEAT_TEMP = 10.0

# ── Shared priors (used in model and query helpers) ───────────────────────────
# Intrinsic strength: normally distributed with mean 100, std 15
_STRENGTH_PRIOR = dist.Normal(torch.tensor(100.), torch.tensor(15.))
# Effort: normally distributed centered around 80% with std 10%
_EFFORT_PRIOR = dist.Normal(torch.tensor(80.), torch.tensor(10.))

# ── Pyro model ────────────────────────────────────────────────────────────────

def model():
    # ── intrinsic_strength (mem'd per athlete) ────────────────────────────────
    intrinsic_strength = {}
    for athlete in ATHLETES:
        raw = pyro.sample(f"intrinsic_strength_{athlete}", _STRENGTH_PRIOR)
        intrinsic_strength[athlete] = raw.clamp(min=0.)

    # ── effort_in_match (mem'd per athlete × match) ───────────────────────────
    effort_in_match = {}
    for (athlete, match_id) in ATHLETE_MATCHES:
        raw = pyro.sample(f"effort_{athlete}_m{match_id}", _EFFORT_PRIOR)
        effort_in_match[(athlete, match_id)] = raw.clamp(0., 100.)

    # ── Derived quantities (deterministic given samples) ──────────────────────

    def athlete_pulling_force(athlete, match_id):
        # Pulling force = intrinsic strength * (effort / 100)
        return intrinsic_strength[athlete] * (effort_in_match[(athlete, match_id)] / 100.)

    def team_pulling_force(team, match_id):
        return sum(athlete_pulling_force(a, match_id) for a in team)

    # ── Conditions (soft likelihoods) ─────────────────────────────────────────
    # beat(team1, team2, match)  →  force_diff > 0  →  factor(logsigmoid(+diff/T))

    def soft_beat(team1, team2, match_id, name):
        force_diff = team_pulling_force(team1, match_id) - team_pulling_force(team2, match_id)
        pyro.factor(name, F.logsigmoid(force_diff / BEAT_TEMP))

    # In the first match, Drew and Lane beat Casey and Robin.
    soft_beat(['drew', 'lane'], ['casey', 'robin'], 1, 'cond_1')
    # In the second match, Lane and Indiana beat Casey and Robin.
    soft_beat(['lane', 'indiana'], ['casey', 'robin'], 2, 'cond_2')
    # In the third match, Lane and Avery beat Casey and Robin.
    soft_beat(['lane', 'avery'], ['casey', 'robin'], 3, 'cond_3')


# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference(num_samples=500, warmup_steps=200):
    kernel = NUTS(model, adapt_step_size=True, target_accept_prob=0.8)
    mcmc   = MCMC(kernel, num_samples=num_samples, warmup_steps=warmup_steps, num_chains=4)
    mcmc.run()
    return mcmc


# ── Query helpers (post-hoc) ──────────────────────────────────────────────────

def intrinsic_strength_rank(samples, athlete, out_of_n_athletes=100):
    """
    Out of 100 random athletes, where does athlete rank in terms of intrinsic strength?
    """
    s_post = samples[f'intrinsic_strength_{athlete}'].double()
    others = _STRENGTH_PRIOR.sample((s_post.shape[0], out_of_n_athletes - 1)).clamp(min=0.)
    rank   = (s_post.unsqueeze(1) > others).double().sum(dim=1)
    return dict(mean=rank.mean().item(), std=rank.std().item(),
                p10=rank.quantile(0.10).item(), p90=rank.quantile(0.90).item(),
                raw=rank.tolist())


def query_effort_in_match(samples, athlete, match_id):
    """
    On a percentage scale from 0 to 100%, how much effort did athlete put into the match?
    """
    key = f"effort_{athlete}_m{match_id}"
    e = samples[key].double().clamp(0., 100.)
    return dict(mean=e.mean().item(), std=e.std().item(),
                p10=e.quantile(0.10).item(), p90=e.quantile(0.90).item(),
                raw=e.tolist())


def who_would_win_by_how_much(samples, team1, team2, n_future=100):
    """
    P(team2 wins) over simulated future matches.
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
            def sim_force(team):
                force = 0.
                for a in team:
                    effort = _EFFORT_PRIOR.sample().clamp(0., 100.)
                    force += strength[a] * (effort / 100.)
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

    print("\n=== Query 1: Drew's strength rank out of 100 athletes ===")
    q1 = intrinsic_strength_rank(samples, 'drew', 100)
    print(f"  mean={q1['mean']:.1f}  std={q1['std']:.1f}  "
          f"[p10={q1['p10']:.1f}, p90={q1['p90']:.1f}]")

    print("\n=== Query 2: Lane's strength rank out of 100 athletes ===")
    q2 = intrinsic_strength_rank(samples, 'lane', 100)
    print(f"  mean={q2['mean']:.1f}  std={q2['std']:.1f}  "
          f"[p10={q2['p10']:.1f}, p90={q2['p90']:.1f}]")

    print("\n=== Query 3: Casey's strength rank out of 100 athletes ===")
    q3 = intrinsic_strength_rank(samples, 'casey', 100)
    print(f"  mean={q3['mean']:.1f}  std={q3['std']:.1f}  "
          f"[p10={q3['p10']:.1f}, p90={q3['p90']:.1f}]")

    print("\n=== Query 4: Drew's effort in match 1 (0-100%) ===")
    q4 = query_effort_in_match(samples, 'drew', 1)
    print(f"  mean={q4['mean']:.1f}  std={q4['std']:.1f}  "
          f"[p10={q4['p10']:.1f}, p90={q4['p90']:.1f}]")

    print("\n=== Query 5: Lane's effort in match 1 (0-100%) ===")
    q5 = query_effort_in_match(samples, 'lane', 1)
    print(f"  mean={q5['mean']:.1f}  std={q5['std']:.1f}  "
          f"[p10={q5['p10']:.1f}, p90={q5['p90']:.1f}]")

    print("\n=== Query 6: Casey's effort in match 1 (0-100%) ===")
    q6 = query_effort_in_match(samples, 'casey', 1)
    print(f"  mean={q6['mean']:.1f}  std={q6['std']:.1f}  "
          f"[p10={q6['p10']:.1f}, p90={q6['p90']:.1f}]")

    print("\n=== Query 7: P(Indiana+Casey beat Drew+Lane) in a future match ===")
    q7 = who_would_win_by_how_much(samples, ['drew', 'lane'], ['indiana', 'casey'])
    print(f"  P(team2 wins) = {q7['p_team2_wins']:.3f}  "
          f"[p10={q7['p10']:.3f}, p90={q7['p90']:.3f}]")

    print("\n=== Query 8: P(Lane+Robin beat Drew+Casey) in a future match ===")
    q8 = who_would_win_by_how_much(samples, ['drew', 'casey'], ['lane', 'robin'])
    print(f"  P(team2 wins) = {q8['p_team2_wins']:.3f}  "
          f"[p10={q8['p10']:.3f}, p90={q8['p90']:.3f}]")

    # ── Save full posterior distributions ─────────────────────────────────────
    tm = datetime.datetime.now()
    result = {
        'query1': {'description': "Drew's strength rank out of 100 athletes",
                   'samples': q1['raw'], 'mean': q1['mean'], 'std': q1['std']},
        'query2': {'description': "Lane's strength rank out of 100 athletes",
                   'samples': q2['raw'], 'mean': q2['mean'], 'std': q2['std']},
        'query3': {'description': "Casey's strength rank out of 100 athletes",
                   'samples': q3['raw'], 'mean': q3['mean'], 'std': q3['std']},
        'query4': {'description': "Drew's effort in match 1 (0-100%)",
                   'samples': q4['raw'], 'mean': q4['mean'], 'std': q4['std']},
        'query5': {'description': "Lane's effort in match 1 (0-100%)",
                   'samples': q5['raw'], 'mean': q5['mean'], 'std': q5['std']},
        'query6': {'description': "Casey's effort in match 1 (0-100%)",
                   'samples': q6['raw'], 'mean': q6['mean'], 'std': q6['std']},
        'query7': {'description': 'P(Indiana+Casey beat Drew+Lane)',
                   'samples': q7['raw'], 'mean': q7['p_team2_wins']},
        'query8': {'description': 'P(Lane+Robin beat Drew+Casey)',
                   'samples': q8['raw'], 'mean': q8['p_team2_wins']},
    }
    
    os.makedirs('inference_results', exist_ok=True)
    out_path = f'inference_results/result-tug-of-war-pytorch.json'
    with open(out_path, 'w') as f:
        json.dump(result, f)