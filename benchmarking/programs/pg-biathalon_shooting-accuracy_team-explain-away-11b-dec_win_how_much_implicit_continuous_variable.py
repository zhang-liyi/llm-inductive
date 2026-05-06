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
ATHLETES = ['robin', 'sam', 'willow', 'gale', 'peyton', 'emery', 'casey', 'avery']

# ── (team, round) pairs from the conditions, in original order ────────────────
TEAM_ROUNDS = [
    (['robin', 'sam'], 1),
    (['willow', 'gale'], 1),
    (['robin', 'peyton'], 2),
    (['willow', 'emery'], 2),
    (['robin', 'casey'], 3),
    (['willow', 'avery'], 3),
    (['robin', 'sam'], 4),
    (['willow', 'gale'], 4),
]

# ── (athlete, round) pairs to track shooting accuracy ─────────────────────────
ATHLETE_ROUNDS = [
    ('robin', 1), ('sam', 1), ('willow', 1), ('gale', 1),
    ('robin', 2), ('peyton', 2), ('willow', 2), ('emery', 2),
    ('robin', 3), ('casey', 3), ('willow', 3), ('avery', 3),
    ('robin', 4), ('sam', 4), ('willow', 4), ('gale', 4),
]

# Temperature for soft beat/lost likelihoods.
# Since we explicitly model noise with std=5, the score difference has std ≈ 7.07.
# BEAT_TEMP=1.0 makes the soft boundary sharp but smooth enough for NUTS.
BEAT_TEMP = 1.0


def _team_key(team, round_id):
    """Canonical sample-site name for a (team, round) pair."""
    return '_'.join(team) + f'_r{round_id}'


# ── Shared priors (used in model and query helpers) ───────────────────────────
def _strength_prior():
    """Intrinsic strength prior: Normal(50, 15)"""
    return dist.Normal(50., 15.)

def _accuracy_prior():
    """Shooting accuracy prior: Normal(50, 15)"""
    return dist.Normal(50., 15.)


# ── Pyro model ────────────────────────────────────────────────────────────────

def model():
    # ── intrinsic_strength (mem'd per athlete) ────────────────────────────────
    intrinsic_strength = {}
    for athlete in ATHLETES:
        raw = pyro.sample(f"intrinsic_strength_{athlete}", _strength_prior())
        intrinsic_strength[athlete] = raw.clamp(min=0.)

    # ── shooting_accuracy_in_round (mem'd per athlete × round) ────────────────
    shooting_accuracy_in_round = {}
    for athlete, round_id in ATHLETE_ROUNDS:
        raw = pyro.sample(f"shooting_accuracy_{athlete}_r{round_id}", _accuracy_prior())
        shooting_accuracy_in_round[(athlete, round_id)] = raw.clamp(0., 100.)

    # ── team noise (mem'd per team × round) ───────────────────────────────────
    # Random performance noise to account for unpredictable environmental factors
    team_noise = {}
    for team, round_id in TEAM_ROUNDS:
        team_noise[(tuple(team), round_id)] = pyro.sample(
            f"noise_{_team_key(team, round_id)}", dist.Normal(0., 5.)
        )

    # ── Derived quantities (deterministic given samples) ──────────────────────

    def team_skiing_speed(team):
        return sum(intrinsic_strength[a] for a in team) / len(team)

    def team_shooting_accuracy_in_round(team, round_id):
        return sum(shooting_accuracy_in_round[(a, round_id)] for a in team) / len(team)

    def overall_team_score(team, round_id):
        speed = team_skiing_speed(team)
        accuracy = team_shooting_accuracy_in_round(team, round_id)
        noise = team_noise[(tuple(team), round_id)]
        return speed + accuracy + noise

    # ── Conditions (soft likelihoods) ─────────────────────────────────────────
    # beat(team1, team2, round)  →  score_diff > 0  →  factor(logsigmoid(+diff/T))
    # lost(team1, team2, round)  →  score_diff < 0  →  factor(logsigmoid(-diff/T))

    def soft_beat(team1, team2, round_id, name):
        score_diff = overall_team_score(team1, round_id) - overall_team_score(team2, round_id)
        pyro.factor(name, F.logsigmoid(score_diff / BEAT_TEMP))

    def soft_lost(team1, team2, round_id, name):
        score_diff = overall_team_score(team1, round_id) - overall_team_score(team2, round_id)
        pyro.factor(name, F.logsigmoid(-score_diff / BEAT_TEMP))

    # In the first round, Robin and Sam lost to Willow and Gale.
    soft_lost(['robin', 'sam'], ['willow', 'gale'], 1, 'cond_1')
    # In the second round, Robin and Peyton lost to Willow and Emery.
    soft_lost(['robin', 'peyton'], ['willow', 'emery'], 2, 'cond_2')
    # In the third round, Robin and Casey lost to Willow and Avery.
    soft_lost(['robin', 'casey'], ['willow', 'avery'], 3, 'cond_3')
    # In the fourth round, Robin and Sam beat Willow and Gale.
    soft_beat(['robin', 'sam'], ['willow', 'gale'], 4, 'cond_4')


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
    others = _strength_prior().sample((s_post.shape[0], out_of_n_athletes - 1)).clamp(min=0.)
    rank   = (s_post.unsqueeze(1) > others).double().sum(dim=1)
    return dict(mean=rank.mean().item(), std=rank.std().item(),
                p10=rank.quantile(0.10).item(), p90=rank.quantile(0.90).item(),
                raw=rank.tolist())


def query_shooting_accuracy_in_round(samples, athlete, round_id):
    """
    Queries 4-6: Shooting accuracy for an athlete in a given round (0–100 scale).
    """
    key = f"shooting_accuracy_{athlete}_r{round_id}"
    d = samples[key].double().clamp(0., 100.)
    return dict(mean=d.mean().item(), std=d.std().item(),
                p10=d.quantile(0.10).item(), p90=d.quantile(0.90).item(),
                raw=d.tolist())


def who_would_win_by_how_much(samples, team1, team2, n_future=100):
    """
    Queries 7 & 8: P(team2 wins) over simulated future rounds.
    New athletes are drawn fresh from the strength prior if they don't exist.
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
                           else _strength_prior().sample().clamp(min=0.))

        wins = 0
        for _ in range(n_future):
            def sim_score(team):
                speed = sum(strength[a] for a in team) / len(team)
                accuracy = sum(_accuracy_prior().sample().clamp(0., 100.) for _ in team) / len(team)
                noise = dist.Normal(0., 5.).sample()
                return speed + accuracy + noise

            if sim_score(team2) > sim_score(team1):
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

    print("Running NUTS inference on biathlon model …")
    mcmc    = run_inference(num_samples=500, warmup_steps=200)
    samples = mcmc.get_samples()
    mcmc.summary()

    print("\n=== Query 1: Robin's strength rank out of 100 athletes ===")
    q1 = intrinsic_strength_rank(samples, 'robin', 100)
    print(f"  mean={q1['mean']:.1f}  std={q1['std']:.1f}  [p10={q1['p10']:.1f}, p90={q1['p90']:.1f}]")

    print("\n=== Query 2: Sam's strength rank out of 100 athletes ===")
    q2 = intrinsic_strength_rank(samples, 'sam', 100)
    print(f"  mean={q2['mean']:.1f}  std={q2['std']:.1f}  [p10={q2['p10']:.1f}, p90={q2['p90']:.1f}]")

    print("\n=== Query 3: Willow's strength rank out of 100 athletes ===")
    q3 = intrinsic_strength_rank(samples, 'willow', 100)
    print(f"  mean={q3['mean']:.1f}  std={q3['std']:.1f}  [p10={q3['p10']:.1f}, p90={q3['p90']:.1f}]")

    print("\n=== Query 4: Robin's shooting accuracy in round 4 (0-100) ===")
    q4 = query_shooting_accuracy_in_round(samples, 'robin', 4)
    print(f"  mean={q4['mean']:.1f}  std={q4['std']:.1f}  [p10={q4['p10']:.1f}, p90={q4['p90']:.1f}]")

    print("\n=== Query 5: Sam's shooting accuracy in round 4 (0-100) ===")
    q5 = query_shooting_accuracy_in_round(samples, 'sam', 4)
    print(f"  mean={q5['mean']:.1f}  std={q5['std']:.1f}  [p10={q5['p10']:.1f}, p90={q5['p90']:.1f}]")

    print("\n=== Query 6: Willow's shooting accuracy in round 4 (0-100) ===")
    q6 = query_shooting_accuracy_in_round(samples, 'willow', 4)
    print(f"  mean={q6['mean']:.1f}  std={q6['std']:.1f}  [p10={q6['p10']:.1f}, p90={q6['p90']:.1f}]")

    print("\n=== Query 7: P(Peyton+Avery beat Robin+Sam) in a future round ===")
    q7 = who_would_win_by_how_much(samples, ['robin', 'sam'], ['peyton', 'avery'])
    print(f"  P(team2 wins) = {q7['p_team2_wins']:.3f}  [p10={q7['p10']:.3f}, p90={q7['p90']:.3f}]")

    print("\n=== Query 8: P(Sam+Gale beat Robin+Willow) in a future round ===")
    q8 = who_would_win_by_how_much(samples, ['robin', 'willow'], ['sam', 'gale'])
    print(f"  P(team2 wins) = {q8['p_team2_wins']:.3f}  [p10={q8['p10']:.3f}, p90={q8['p90']:.3f}]")

    # ── Save full posterior distributions ─────────────────────────────────────
    import os
    os.makedirs('inference_results', exist_ok=True)
    result = {
        'query1': {'description': "Robin's strength rank out of 100 athletes",
                   'samples': q1['raw'], 'mean': q1['mean'], 'std': q1['std']},
        'query2': {'description': "Sam's strength rank out of 100 athletes",
                   'samples': q2['raw'], 'mean': q2['mean'], 'std': q2['std']},
        'query3': {'description': "Willow's strength rank out of 100 athletes",
                   'samples': q3['raw'], 'mean': q3['mean'], 'std': q3['std']},
        'query4': {'description': "Robin's shooting accuracy in round 4",
                   'samples': q4['raw'], 'mean': q4['mean'], 'std': q4['std']},
        'query5': {'description': "Sam's shooting accuracy in round 4",
                   'samples': q5['raw'], 'mean': q5['mean'], 'std': q5['std']},
        'query6': {'description': "Willow's shooting accuracy in round 4",
                   'samples': q6['raw'], 'mean': q6['mean'], 'std': q6['std']},
        'query7': {'description': 'P(Peyton+Avery beat Robin+Sam)',
                   'samples': q7['raw'], 'mean': q7['p_team2_wins']},
        'query8': {'description': 'P(Sam+Gale beat Robin+Willow)',
                   'samples': q8['raw'], 'mean': q8['p_team2_wins']},
    }
    out_path = f'inference_results/result-biathlon-pytorch.json'
    with open(out_path, 'w') as f:
        json.dump(result, f)