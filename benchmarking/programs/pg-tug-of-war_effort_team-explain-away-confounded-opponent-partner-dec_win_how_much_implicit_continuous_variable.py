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
ATHLETES = ['peyton', 'max', 'ness', 'robin', 'blake', 'casey', 'kay']

# ── (athlete, match) pairs from the conditions, in original order ─────────────
ATHLETE_MATCHES = [
    ('peyton', 1), ('max', 1), ('ness', 1), ('robin', 1),
    ('peyton', 2), ('max', 2), ('ness', 2), ('robin', 2),
    ('peyton', 3), ('max', 3), ('ness', 3), ('blake', 3),
    ('peyton', 4), ('max', 4), ('ness', 4), ('casey', 4),
    ('peyton', 5), ('max', 5), ('ness', 5), ('kay', 5),
]

# Temperature for soft beat/lost likelihoods.
# Team pull force is O(~120); BEAT_TEMP=10.0 makes the soft boundary
# sharp but everywhere-finite and smooth.
BEAT_TEMP = 10.0

# ── Shared strength prior (used in model and query helpers) ───────────────────
# Intrinsic strength is normally distributed with mean 100 and std 15.
_STRENGTH_PRIOR = dist.Normal(torch.tensor(100.), torch.tensor(15.))


def _effort_prior(strength):
    """
    3-component Gaussian mixture for effort (0-100 scale).
    Prior weights depend on intrinsic strength, mirroring the WebPPL if/elif.
    Stronger athletes might exert less effort, weaker athletes might exert more.
    """
    s = strength
    p_low      = torch.where(s > 115, torch.tensor(0.80),
                 torch.where(s <  85, torch.tensor(0.05), torch.tensor(0.20)))
    p_medium   = torch.where(s > 115, torch.tensor(0.15),
                 torch.where(s <  85, torch.tensor(0.15), torch.tensor(0.60)))
    p_high     = torch.where(s > 115, torch.tensor(0.05),
                 torch.where(s <  85, torch.tensor(0.80), torch.tensor(0.20)))
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
    for (athlete, match_id) in ATHLETE_MATCHES:
        strength = intrinsic_strength[athlete]
        raw = pyro.sample(
            f"effort_{athlete}_m{match_id}",
            _effort_prior(strength),
        )
        effort_in_match[(athlete, match_id)] = raw.clamp(0., 100.)

    # ── Derived quantities (deterministic given samples) ──────────────────────

    def athlete_pull_force(athlete, match_id):
        effort = effort_in_match[(athlete, match_id)]
        return intrinsic_strength[athlete] * (effort / 100.)

    def team_pull_force(team, match_id):
        return sum(athlete_pull_force(a, match_id) for a in team)

    # ── Conditions (soft likelihoods) ─────────────────────────────────────────
    # beat(team1, team2, match)  →  score_diff > 0  →  factor(logsigmoid(+diff/T))
    # lost(team1, team2, match)  →  score_diff < 0  →  factor(logsigmoid(-diff/T))

    def soft_beat(team1, team2, match_id, name):
        score_diff = team_pull_force(team1, match_id) - team_pull_force(team2, match_id)
        pyro.factor(name, F.logsigmoid(score_diff / BEAT_TEMP))

    def soft_lost(team1, team2, match_id, name):
        score_diff = team_pull_force(team1, match_id) - team_pull_force(team2, match_id)
        pyro.factor(name, F.logsigmoid(-score_diff / BEAT_TEMP))

    # In the first match, Peyton and Max beat Ness and Robin.
    soft_beat(['peyton', 'max'], ['ness', 'robin'], 1, 'cond_1')
    # In the second match, Peyton and Max lost to Ness and Robin.
    soft_lost(['peyton', 'max'], ['ness', 'robin'], 2, 'cond_2')
    # In the third match, Peyton and Max beat Ness and Blake.
    soft_beat(['peyton', 'max'], ['ness', 'blake'], 3, 'cond_3')
    # In the fourth match, Peyton and Max beat Ness and Casey.
    soft_beat(['peyton', 'max'], ['ness', 'casey'], 4, 'cond_4')
    # In the fifth match, Peyton and Max beat Ness and Kay.
    soft_beat(['peyton', 'max'], ['ness', 'kay'], 5, 'cond_5')


# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference(num_samples=500, warmup_steps=200):
    kernel = NUTS(model, adapt_step_size=True, target_accept_prob=0.8)
    mcmc   = MCMC(kernel, num_samples=num_samples, warmup_steps=warmup_steps, num_chains=4)
    mcmc.run()
    return mcmc


# ── Query helpers (post-hoc, matching WebPPL queries) ────────────────────────

def intrinsic_strength_rank(samples, athlete='peyton', out_of_n_athletes=100):
    """
    Queries 1-3: Out of 100 random athletes, where does athlete rank?
    Mirrors: intrinsic_strength_rank({athlete, out_of_n_athletes: 100})
    """
    s_post = samples[f'intrinsic_strength_{athlete}'].double()
    others = _STRENGTH_PRIOR.sample((s_post.shape[0], out_of_n_athletes - 1)).clamp(min=0.)
    rank   = (s_post.unsqueeze(1) > others).double().sum(dim=1)
    return dict(mean=rank.mean().item(), std=rank.std().item(),
                p10=rank.quantile(0.10).item(), p90=rank.quantile(0.90).item(),
                raw=rank.tolist())


def query_effort_in_match(samples, athlete='peyton', match_id=2):
    """
    Queries 4-6: Effort for an athlete in a given match (0–100 scale).
    Mirrors: query_effort_in_match({athlete, match})
    """
    key = f"effort_{athlete}_m{match_id}"
    e = samples[key].double().clamp(0., 100.)
    return dict(mean=e.mean().item(), std=e.std().item(),
                p10=e.quantile(0.10).item(), p90=e.quantile(0.90).item(),
                raw=e.tolist())


def who_would_win_by_how_much(samples, team1, team2, n_future=100):
    """
    Queries 7 & 8: P(team2 wins) over simulated future matches.
    Mirrors: who_would_win_by_how_much({team1, team2})
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

    print("\n=== Query 1: Peyton's intrinsic strength rank out of 100 ===")
    q1 = intrinsic_strength_rank(samples, 'peyton', 100)
    print(f"  mean={q1['mean']:.1f}  std={q1['std']:.1f}  "
          f"[p10={q1['p10']:.1f}, p90={q1['p90']:.1f}]")

    print("\n=== Query 2: Max's intrinsic strength rank out of 100 ===")
    q2 = intrinsic_strength_rank(samples, 'max', 100)
    print(f"  mean={q2['mean']:.1f}  std={q2['std']:.1f}  "
          f"[p10={q2['p10']:.1f}, p90={q2['p90']:.1f}]")

    print("\n=== Query 3: Ness's intrinsic strength rank out of 100 ===")
    q3 = intrinsic_strength_rank(samples, 'ness', 100)
    print(f"  mean={q3['mean']:.1f}  std={q3['std']:.1f}  "
          f"[p10={q3['p10']:.1f}, p90={q3['p90']:.1f}]")

    print("\n=== Query 4: Peyton's effort in match 2 (0-100) ===")
    q4 = query_effort_in_match(samples, 'peyton', 2)
    print(f"  mean={q4['mean']:.1f}  std={q4['std']:.1f}  "
          f"[p10={q4['p10']:.1f}, p90={q4['p90']:.1f}]")

    print("\n=== Query 5: Max's effort in match 2 (0-100) ===")
    q5 = query_effort_in_match(samples, 'max', 2)
    print(f"  mean={q5['mean']:.1f}  std={q5['std']:.1f}  "
          f"[p10={q5['p10']:.1f}, p90={q5['p90']:.1f}]")

    print("\n=== Query 6: Ness's effort in match 2 (0-100) ===")
    q6 = query_effort_in_match(samples, 'ness', 2)
    print(f"  mean={q6['mean']:.1f}  std={q6['std']:.1f}  "
          f"[p10={q6['p10']:.1f}, p90={q6['p90']:.1f}]")

    print("\n=== Query 7: P(Ness+Robin beat Peyton+Max) in a future match ===")
    q7 = who_would_win_by_how_much(samples, ['peyton', 'max'], ['ness', 'robin'])
    print(f"  P(team2 wins) = {q7['p_team2_wins']:.3f}  "
          f"[p10={q7['p10']:.3f}, p90={q7['p90']:.3f}]")

    print("\n=== Query 8: P(Robin+Blake beat Peyton+Max) in a future match ===")
    q8 = who_would_win_by_how_much(samples, ['peyton', 'max'], ['robin', 'blake'])
    print(f"  P(team2 wins) = {q8['p_team2_wins']:.3f}  "
          f"[p10={q8['p10']:.3f}, p90={q8['p90']:.3f}]")

    # ── Save full posterior distributions ─────────────────────────────────────
    tm = datetime.datetime.now()
    result = {
        'query1': {'description': "Peyton's rank out of 100 athletes",
                   'samples': q1['raw'], 'mean': q1['mean'], 'std': q1['std']},
        'query2': {'description': "Max's rank out of 100 athletes",
                   'samples': q2['raw'], 'mean': q2['mean'], 'std': q2['std']},
        'query3': {'description': "Ness's rank out of 100 athletes",
                   'samples': q3['raw'], 'mean': q3['mean'], 'std': q3['std']},
        'query4': {'description': "Peyton's effort in match 2 (0-100)",
                   'samples': q4['raw'], 'mean': q4['mean'], 'std': q4['std']},
        'query5': {'description': "Max's effort in match 2 (0-100)",
                   'samples': q5['raw'], 'mean': q5['mean'], 'std': q5['std']},
        'query6': {'description': "Ness's effort in match 2 (0-100)",
                   'samples': q6['raw'], 'mean': q6['mean'], 'std': q6['std']},
        'query7': {'description': 'P(Ness+Robin beat Peyton+Max)',
                   'samples': q7['raw'], 'mean': q7['p_team2_wins']},
        'query8': {'description': 'P(Robin+Blake beat Peyton+Max)',
                   'samples': q8['raw'], 'mean': q8['p_team2_wins']},
    }
    out_path = f'inference_results/result-tug-of-war-pytorch.json'
    os.makedirs('inference_results', exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(result, f)