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
ATHLETES = ['emery', 'peyton', 'blake', 'gale', 'lane', 'avery']

# ── (athlete, match) pairs from the conditions ────────────────────────────────
MATCH_ATHLETES = [
    ('emery', 1), ('peyton', 1), ('blake', 1), ('gale', 1),
    ('peyton', 2), ('lane', 2),   ('blake', 2), ('gale', 2),
    ('peyton', 3), ('avery', 3),  ('blake', 3), ('gale', 3),
]

# Temperature for soft lost likelihoods.
# Matches the ~5 std dev noise mentioned in the scratchpad.
BEAT_TEMP = 5.0

def _athlete_match_key(athlete, match_id):
    """Canonical sample-site name for an (athlete, match) pair."""
    return f"{athlete}_m{match_id}"

# ── Shared strength prior (used in model and query helpers) ───────────────────
# Intrinsic strength is in arbitrary units (e.g., ~100).
_STRENGTH_PRIOR = dist.MixtureSameFamily(
    dist.Categorical(probs=torch.tensor([0.33, 0.33, 0.34])),
    dist.Normal(torch.tensor([80., 100., 140.]), torch.tensor([10., 10., 10.])),
)

def _effort_prior(strength):
    """
    3-component Gaussian mixture for effort (0-100%).
    Prior weights depend on intrinsic strength.
    """
    s = strength
    p_low  = torch.where(s > 120, torch.tensor(0.05),
             torch.where(s <  80, torch.tensor(0.80), torch.tensor(0.20)))
    p_med  = torch.where(s > 120, torch.tensor(0.15),
             torch.where(s <  80, torch.tensor(0.15), torch.tensor(0.60)))
    p_high = torch.where(s > 120, torch.tensor(0.80),
             torch.where(s <  80, torch.tensor(0.05), torch.tensor(0.20)))
    return dist.MixtureSameFamily(
        dist.Categorical(probs=torch.stack([p_low, p_med, p_high])),
        dist.Normal(torch.tensor([30., 60., 90.]), torch.tensor([10., 10., 10.])),
    )

# ── Pyro model ────────────────────────────────────────────────────────────────

def model():
    # ── intrinsic_strength (mem'd per athlete) ───────────────────────────────
    intrinsic_strength = {}
    for athlete in ATHLETES:
        raw = pyro.sample(f"intrinsic_strength_{athlete}", _STRENGTH_PRIOR)
        intrinsic_strength[athlete] = raw.clamp(min=0.)

    # ── athlete_effort_in_match (mem'd per athlete × match) ──────────────────
    athlete_effort_in_match = {}
    for (athlete, match_id) in MATCH_ATHLETES:
        strength = intrinsic_strength[athlete]
        raw = pyro.sample(
            f"athlete_effort_{_athlete_match_key(athlete, match_id)}",
            _effort_prior(strength)
        )
        athlete_effort_in_match[(athlete, match_id)] = raw.clamp(0., 100.)

    # ── Derived quantities (deterministic given samples) ──────────────────────

    def athlete_pulling_force_in_match(athlete, match_id):
        effort = athlete_effort_in_match[(athlete, match_id)]
        return intrinsic_strength[athlete] * (effort / 100.)

    def team_pulling_force_in_match(team, match_id):
        return sum(athlete_pulling_force_in_match(a, match_id) for a in team)

    # ── Conditions (soft likelihoods) ─────────────────────────────────────────
    # lost(team1, team2, match)  →  score_diff < 0  →  factor(logsigmoid(-diff/T))

    def soft_lost(team1, team2, match_id, name):
        score_diff = team_pulling_force_in_match(team1, match_id) - team_pulling_force_in_match(team2, match_id)
        pyro.factor(name, F.logsigmoid(-score_diff / BEAT_TEMP))

    # In the first match, Emery and Peyton lost to Blake and Gale.
    soft_lost(['emery', 'peyton'], ['blake', 'gale'], 1, 'cond_1')
    # In the second match, Peyton and Lane lost to Blake and Gale.
    soft_lost(['peyton', 'lane'], ['blake', 'gale'], 2, 'cond_2')
    # In the third match, Peyton and Avery lost to Blake and Gale.
    soft_lost(['peyton', 'avery'], ['blake', 'gale'], 3, 'cond_3')

# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference(num_samples=500, warmup_steps=200):
    kernel = NUTS(model, adapt_step_size=True, target_accept_prob=0.8)
    mcmc   = MCMC(kernel, num_samples=num_samples, warmup_steps=warmup_steps, num_chains=4)
    mcmc.run()
    return mcmc

# ── Query helpers (post-hoc) ──────────────────────────────────────────────────

def intrinsic_strength_rank(samples, athlete='emery', out_of_n_athletes=100):
    """
    Queries 1-3: Out of 100 random athletes, where does athlete rank?
    """
    s_post = samples[f'intrinsic_strength_{athlete}'].double()
    others = _STRENGTH_PRIOR.sample((s_post.shape[0], out_of_n_athletes - 1)).clamp(min=0.)
    rank   = (s_post.unsqueeze(1) > others).double().sum(dim=1)
    return dict(mean=rank.mean().item(), std=rank.std().item(),
                p10=rank.quantile(0.10).item(), p90=rank.quantile(0.90).item(),
                raw=rank.tolist())

def query_athlete_effort_in_match(samples, athlete='emery', match_id=1):
    """
    Queries 4-6: Effort for an athlete in a given match (0–100 scale).
    """
    key = f"athlete_effort_{_athlete_match_key(athlete, match_id)}"
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
                    s = strength[a]
                    e = _effort_prior(s).sample().clamp(0., 100.)
                    force += s * (e / 100.)
                return force

            force1 = sim_force(team1)
            force2 = sim_force(team2)
            # Add noise to the margin of victory
            noise = dist.Normal(0., 5.).sample()
            if force2 > force1 + noise:
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

    print("\n=== Query 1: Emery's intrinsic strength rank out of 100 ===")
    q1 = intrinsic_strength_rank(samples, 'emery', 100)
    print(f"  mean={q1['mean']:.1f}  std={q1['std']:.1f}  [p10={q1['p10']:.1f}, p90={q1['p90']:.1f}]")

    print("\n=== Query 2: Peyton's intrinsic strength rank out of 100 ===")
    q2 = intrinsic_strength_rank(samples, 'peyton', 100)
    print(f"  mean={q2['mean']:.1f}  std={q2['std']:.1f}  [p10={q2['p10']:.1f}, p90={q2['p90']:.1f}]")

    print("\n=== Query 3: Blake's intrinsic strength rank out of 100 ===")
    q3 = intrinsic_strength_rank(samples, 'blake', 100)
    print(f"  mean={q3['mean']:.1f}  std={q3['std']:.1f}  [p10={q3['p10']:.1f}, p90={q3['p90']:.1f}]")

    print("\n=== Query 4: Emery's effort in match 1 (0-100) ===")
    q4 = query_athlete_effort_in_match(samples, 'emery', 1)
    print(f"  mean={q4['mean']:.1f}  std={q4['std']:.1f}  [p10={q4['p10']:.1f}, p90={q4['p90']:.1f}]")

    print("\n=== Query 5: Peyton's effort in match 1 (0-100) ===")
    q5 = query_athlete_effort_in_match(samples, 'peyton', 1)
    print(f"  mean={q5['mean']:.1f}  std={q5['std']:.1f}  [p10={q5['p10']:.1f}, p90={q5['p90']:.1f}]")

    print("\n=== Query 6: Blake's effort in match 1 (0-100) ===")
    q6 = query_athlete_effort_in_match(samples, 'blake', 1)
    print(f"  mean={q6['mean']:.1f}  std={q6['std']:.1f}  [p10={q6['p10']:.1f}, p90={q6['p90']:.1f}]")

    print("\n=== Query 7: P(Lane+Blake beat Emery+Peyton) in a future match ===")
    q7 = who_would_win_by_how_much(samples, ['emery', 'peyton'], ['lane', 'blake'])
    print(f"  P(team2 wins) = {q7['p_team2_wins']:.3f}  [p10={q7['p10']:.3f}, p90={q7['p90']:.3f}]")

    print("\n=== Query 8: P(Peyton+Gale beat Emery+Blake) in a future match ===")
    q8 = who_would_win_by_how_much(samples, ['emery', 'blake'], ['peyton', 'gale'])
    print(f"  P(team2 wins) = {q8['p_team2_wins']:.3f}  [p10={q8['p10']:.3f}, p90={q8['p90']:.3f}]")

    # ── Save full posterior distributions ─────────────────────────────────────
    result = {
        'query1': {'description': "Emery's rank out of 100 athletes", 'samples': q1['raw'], 'mean': q1['mean'], 'std': q1['std']},
        'query2': {'description': "Peyton's rank out of 100 athletes", 'samples': q2['raw'], 'mean': q2['mean'], 'std': q2['std']},
        'query3': {'description': "Blake's rank out of 100 athletes", 'samples': q3['raw'], 'mean': q3['mean'], 'std': q3['std']},
        'query4': {'description': "Emery's effort in match 1", 'samples': q4['raw'], 'mean': q4['mean'], 'std': q4['std']},
        'query5': {'description': "Peyton's effort in match 1", 'samples': q5['raw'], 'mean': q5['mean'], 'std': q5['std']},
        'query6': {'description': "Blake's effort in match 1", 'samples': q6['raw'], 'mean': q6['mean'], 'std': q6['std']},
        'query7': {'description': 'P(Lane+Blake beat Emery+Peyton)', 'samples': q7['raw'], 'mean': q7['p_team2_wins']},
        'query8': {'description': 'P(Peyton+Gale beat Emery+Blake)', 'samples': q8['raw'], 'mean': q8['p_team2_wins']},
    }
    os.makedirs('inference_results', exist_ok=True)
    out_path = f'inference_results/result-tug-of-war-pytorch.json'
    with open(out_path, 'w') as f:
        json.dump(result, f)