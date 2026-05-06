import argparse
import json
import datetime
import torch
import torch.nn.functional as F
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS

torch.set_default_dtype(torch.float64)

# ── Athletes that appear in the conditions ────────────────────────────────────
ATHLETES = ['lane', 'ollie', 'robin', 'quinn', 'drew', 'casey']

# ── Matches from the conditions, in original order ────────────────────────────
MATCHES = [
    (['lane', 'ollie'], ['robin', 'quinn'], 1),
    (['lane', 'ollie'], ['drew', 'casey'],  2),
    (['lane', 'ollie'], ['drew', 'quinn'],  3),
    (['lane', 'ollie'], ['robin', 'drew'],  4),
    (['lane', 'ollie'], ['casey', 'quinn'], 5),
]

# Temperature for soft beat/lost likelihoods.
# Team pulling forces are typically around 100-200. BEAT_TEMP=10.0 makes the 
# soft boundary sharp but smooth.
BEAT_TEMP = 10.0

# ── Shared priors ─────────────────────────────────────────────────────────────

# Intrinsic strength prior: Normal distribution with mean 100 units and std 15 units.
_STRENGTH_PRIOR = dist.Normal(torch.tensor(100.), torch.tensor(15.))

def _effort_prior(strength):
    """
    3-component Gaussian mixture for effort (0-100% scale).
    Prior weights depend on intrinsic strength, mirroring the example's 
    difficulty prior depending on skill.
    """
    s = strength
    p_low    = torch.where(s > 110, torch.tensor(0.05),
               torch.where(s <  90, torch.tensor(0.80), torch.tensor(0.20)))
    p_medium = torch.where(s > 110, torch.tensor(0.15),
               torch.where(s <  90, torch.tensor(0.15), torch.tensor(0.60)))
    p_high   = torch.where(s > 110, torch.tensor(0.80),
               torch.where(s <  90, torch.tensor(0.05), torch.tensor(0.20)))
    return dist.MixtureSameFamily(
        dist.Categorical(probs=torch.stack([p_low, p_medium, p_high])),
        dist.Normal(torch.tensor([30., 60., 90.]), torch.tensor([10., 10., 10.])),
    )


# ── Pyro model ────────────────────────────────────────────────────────────────

def model():
    # ── intrinsic_strength (mem'd per athlete) ───────────────────────────────
    intrinsic_strength = {}
    for athlete in ATHLETES:
        raw = pyro.sample(f"intrinsic_strength_{athlete}", _STRENGTH_PRIOR)
        intrinsic_strength[athlete] = raw.clamp(min=0.)

    # ── effort_in_match (mem'd per athlete × match) ──────────────────────────
    effort_in_match = {}
    for team1, team2, match_id in MATCHES:
        for athlete in team1 + team2:
            if (athlete, match_id) not in effort_in_match:
                raw = pyro.sample(
                    f"effort_{athlete}_m{match_id}",
                    _effort_prior(intrinsic_strength[athlete])
                )
                effort_in_match[(athlete, match_id)] = raw.clamp(0., 100.)

    # ── Derived quantities (deterministic given samples) ──────────────────────

    def athlete_pulling_force(athlete, match_id):
        strength = intrinsic_strength[athlete]
        effort = effort_in_match[(athlete, match_id)]
        # Pulling force = intrinsic strength * effort (as a multiplier 0-1)
        return strength * (effort / 100.)

    def team_pulling_force(team, match_id):
        return sum(athlete_pulling_force(a, match_id) for a in team)

    # ── Conditions (soft likelihoods) ─────────────────────────────────────────

    def soft_beat(team1, team2, match_id, name):
        score_diff = team_pulling_force(team1, match_id) - team_pulling_force(team2, match_id)
        pyro.factor(name, F.logsigmoid(score_diff / BEAT_TEMP))

    def soft_lost(team1, team2, match_id, name):
        score_diff = team_pulling_force(team1, match_id) - team_pulling_force(team2, match_id)
        pyro.factor(name, F.logsigmoid(-score_diff / BEAT_TEMP))

    # In the first match, Lane and Ollie beat Robin and Quinn.
    soft_beat(['lane', 'ollie'], ['robin', 'quinn'], 1, 'cond_1')
    # In the second match, Lane and Ollie beat Drew and Casey.
    soft_beat(['lane', 'ollie'], ['drew', 'casey'], 2, 'cond_2')
    # In the third match, Lane and Ollie lost to Drew and Quinn.
    soft_lost(['lane', 'ollie'], ['drew', 'quinn'], 3, 'cond_3')
    # In the fourth match, Lane and Ollie beat Robin and Drew.
    soft_beat(['lane', 'ollie'], ['robin', 'drew'], 4, 'cond_4')
    # In the fifth match, Lane and Ollie beat Casey and Quinn.
    soft_beat(['lane', 'ollie'], ['casey', 'quinn'], 5, 'cond_5')


# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference(num_samples=500, warmup_steps=200):
    kernel = NUTS(model, adapt_step_size=True, target_accept_prob=0.8)
    mcmc   = MCMC(kernel, num_samples=num_samples, warmup_steps=warmup_steps, num_chains=4)
    mcmc.run()
    return mcmc


# ── Query helpers (post-hoc) ──────────────────────────────────────────────────

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


def query_effort_in_match(samples, athlete, match_id):
    """
    Queries 4-6: Effort for an athlete in a given match (0–100 scale).
    """
    key = f"effort_{athlete}_m{match_id}"
    e = samples[key].double().clamp(0., 100.)
    return dict(mean=e.mean().item(), std=e.std().item(),
                p10=e.quantile(0.10).item(), p90=e.quantile(0.90).item(),
                raw=e.tolist())


def who_would_win_by_how_much(samples, team1, team2, n_future=100):
    """
    Queries 7 & 8: P(team2 wins) over simulated future matches.
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
                    eff = _effort_prior(strength[a]).sample().clamp(0., 100.)
                    force += strength[a] * (eff / 100.)
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

    print("\n=== Query 1: Lane's strength rank out of 100 athletes ===")
    q1 = intrinsic_strength_rank(samples, 'lane', 100)
    print(f"  mean={q1['mean']:.1f}  std={q1['std']:.1f}  [p10={q1['p10']:.1f}, p90={q1['p90']:.1f}]")

    print("\n=== Query 2: Ollie's strength rank out of 100 athletes ===")
    q2 = intrinsic_strength_rank(samples, 'ollie', 100)
    print(f"  mean={q2['mean']:.1f}  std={q2['std']:.1f}  [p10={q2['p10']:.1f}, p90={q2['p90']:.1f}]")

    print("\n=== Query 3: Drew's strength rank out of 100 athletes ===")
    q3 = intrinsic_strength_rank(samples, 'drew', 100)
    print(f"  mean={q3['mean']:.1f}  std={q3['std']:.1f}  [p10={q3['p10']:.1f}, p90={q3['p90']:.1f}]")

    print("\n=== Query 4: Lane's effort in match 3 (0-100) ===")
    q4 = query_effort_in_match(samples, 'lane', 3)
    print(f"  mean={q4['mean']:.1f}  std={q4['std']:.1f}  [p10={q4['p10']:.1f}, p90={q4['p90']:.1f}]")

    print("\n=== Query 5: Ollie's effort in match 3 (0-100) ===")
    q5 = query_effort_in_match(samples, 'ollie', 3)
    print(f"  mean={q5['mean']:.1f}  std={q5['std']:.1f}  [p10={q5['p10']:.1f}, p90={q5['p90']:.1f}]")

    print("\n=== Query 6: Drew's effort in match 3 (0-100) ===")
    q6 = query_effort_in_match(samples, 'drew', 3)
    print(f"  mean={q6['mean']:.1f}  std={q6['std']:.1f}  [p10={q6['p10']:.1f}, p90={q6['p90']:.1f}]")

    print("\n=== Query 7: P(Drew+Quinn beat Lane+Ollie) in a future match ===")
    q7 = who_would_win_by_how_much(samples, ['lane', 'ollie'], ['drew', 'quinn'])
    print(f"  P(team2 wins) = {q7['p_team2_wins']:.3f}  [p10={q7['p10']:.3f}, p90={q7['p90']:.3f}]")

    print("\n=== Query 8: P(Ollie+Casey beat Lane+Robin) in a future match ===")
    q8 = who_would_win_by_how_much(samples, ['lane', 'robin'], ['ollie', 'casey'])
    print(f"  P(team2 wins) = {q8['p_team2_wins']:.3f}  [p10={q8['p10']:.3f}, p90={q8['p90']:.3f}]")

    # ── Save full posterior distributions ─────────────────────────────────────
    tm = datetime.datetime.now()
    result = {
        'query1': {'description': "Lane's strength rank out of 100 athletes", 'samples': q1['raw'], 'mean': q1['mean'], 'std': q1['std']},
        'query2': {'description': "Ollie's strength rank out of 100 athletes", 'samples': q2['raw'], 'mean': q2['mean'], 'std': q2['std']},
        'query3': {'description': "Drew's strength rank out of 100 athletes", 'samples': q3['raw'], 'mean': q3['mean'], 'std': q3['std']},
        'query4': {'description': "Lane's effort in match 3", 'samples': q4['raw'], 'mean': q4['mean'], 'std': q4['std']},
        'query5': {'description': "Ollie's effort in match 3", 'samples': q5['raw'], 'mean': q5['mean'], 'std': q5['std']},
        'query6': {'description': "Drew's effort in match 3", 'samples': q6['raw'], 'mean': q6['mean'], 'std': q6['std']},
        'query7': {'description': 'P(Drew+Quinn beat Lane+Ollie)', 'samples': q7['raw'], 'mean': q7['p_team2_wins']},
        'query8': {'description': 'P(Ollie+Casey beat Lane+Robin)', 'samples': q8['raw'], 'mean': q8['p_team2_wins']},
    }
    import os
    os.makedirs('inference_results', exist_ok=True)
    out_path = f'inference_results/result-tug-of-war-pytorch.json'
    with open(out_path, 'w') as f:
        json.dump(result, f)