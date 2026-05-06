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
ATHLETES = ['kay', 'ness', 'blake', 'fey', 'taylor', 'robin']

# ── (athlete, round) pairs from the conditions ────────────────────────────────
ATHLETE_ROUNDS = [
    ('kay', 1), ('ness', 1), ('blake', 1), ('fey', 1),
    ('ness', 2), ('taylor', 2), ('blake', 2), ('fey', 2),
    ('ness', 3), ('robin', 3), ('blake', 3), ('fey', 3),
]

# Temperature for soft beat/lost likelihoods.
BEAT_TEMP = 100.0

# ── Pyro model ────────────────────────────────────────────────────────────────

def model():
    # ── intrinsic_strength (mem'd per athlete) ───────────────────────────────
    # Units: arbitrary continuous scale (mean 50, std 15)
    intrinsic_strength = {}
    for athlete in ATHLETES:
        raw = pyro.sample(f"intrinsic_strength_{athlete}", dist.Normal(50., 15.))
        intrinsic_strength[athlete] = raw.clamp(min=0.)

    # ── skiing_speed (mem'd per athlete) ─────────────────────────────────────
    # Units: arbitrary continuous scale, dependent on intrinsic strength
    skiing_speed = {}
    for athlete in ATHLETES:
        raw = pyro.sample(f"skiing_speed_{athlete}", dist.Normal(intrinsic_strength[athlete], 5.))
        skiing_speed[athlete] = raw.clamp(min=0.)

    # ── shooting_accuracy_in_round (mem'd per athlete × round) ───────────────
    # Units: percentage (0 to 100%)
    shooting_accuracy_in_round = {}
    for (athlete, round_id) in ATHLETE_ROUNDS:
        raw = pyro.sample(f"shooting_accuracy_{athlete}_r{round_id}", dist.Normal(50., 15.))
        shooting_accuracy_in_round[(athlete, round_id)] = raw.clamp(0., 100.)

    # ── Derived quantities (deterministic given samples) ──────────────────────
    def overall_team_score(team, round_id):
        avg_skiing = sum(skiing_speed[a] for a in team) / len(team)
        avg_shooting = sum(shooting_accuracy_in_round[(a, round_id)] for a in team) / len(team)
        return avg_skiing + avg_shooting

    # ── Conditions (soft likelihoods) ─────────────────────────────────────────
    def soft_beat(team1, team2, round_id, name):
        score_diff = overall_team_score(team1, round_id) - overall_team_score(team2, round_id)
        pyro.factor(name, F.logsigmoid(score_diff / BEAT_TEMP))

    def soft_lost(team1, team2, round_id, name):
        score_diff = overall_team_score(team1, round_id) - overall_team_score(team2, round_id)
        pyro.factor(name, F.logsigmoid(-score_diff / BEAT_TEMP))

    # In the first round, Kay and Ness beat Blake and Fey.
    soft_beat(['kay', 'ness'], ['blake', 'fey'], 1, 'cond_1')
    # In the second round, Ness and Taylor beat Blake and Fey.
    soft_beat(['ness', 'taylor'], ['blake', 'fey'], 2, 'cond_2')
    # In the third round, Ness and Robin beat Blake and Fey.
    soft_beat(['ness', 'robin'], ['blake', 'fey'], 3, 'cond_3')


# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference(num_samples=500, warmup_steps=200):
    kernel = NUTS(model, adapt_step_size=True, target_accept_prob=0.8)
    mcmc   = MCMC(kernel, num_samples=num_samples, warmup_steps=warmup_steps, num_chains=4)
    mcmc.run()
    return mcmc


# ── Query helpers (post-hoc) ──────────────────────────────────────────────────

def intrinsic_strength_rank(samples, athlete='kay', out_of_n_athletes=100):
    """
    Queries 1-3: Out of 100 random athletes, where does athlete rank?
    """
    s_post = samples[f'intrinsic_strength_{athlete}'].double()
    others = dist.Normal(50., 15.).sample((s_post.shape[0], out_of_n_athletes - 1)).clamp(min=0.)
    rank = (s_post.unsqueeze(1) > others).double().sum(dim=1)
    return dict(mean=rank.mean().item(), std=rank.std().item(),
                p10=rank.quantile(0.10).item(), p90=rank.quantile(0.90).item(),
                raw=rank.tolist())

def query_shooting_accuracy_in_round(samples, athlete='kay', round_id=1):
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
    """
    all_players = list(set(team1 + team2))
    n_post = samples[f'intrinsic_strength_{ATHLETES[0]}'].shape[0]
    p_team2_wins = []

    for i in range(n_post):
        strength = {}
        speed = {}
        for a in all_players:
            key_str = f'intrinsic_strength_{a}'
            key_spd = f'skiing_speed_{a}'
            if key_str in samples:
                strength[a] = samples[key_str][i].double()
                speed[a] = samples[key_spd][i].double()
            else:
                # Unknown athletes are drawn fresh from the prior
                strength[a] = dist.Normal(50., 15.).sample().clamp(min=0.)
                speed[a] = dist.Normal(strength[a], 5.).sample().clamp(min=0.)

        wins = 0
        for _ in range(n_future):
            def sim_score(team):
                avg_speed = sum(speed[a] for a in team) / len(team)
                # Shooting accuracy varies per round
                avg_shooting = sum(dist.Normal(50., 15.).sample().clamp(0., 100.) for a in team) / len(team)
                return avg_speed + avg_shooting

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
    mcmc = run_inference(num_samples=500, warmup_steps=200)
    samples = mcmc.get_samples()
    mcmc.summary()

    print("\n=== Query 1: Kay's strength rank out of 100 athletes ===")
    q1 = intrinsic_strength_rank(samples, 'kay', 100)
    print(f"  mean={q1['mean']:.1f}  std={q1['std']:.1f}  [p10={q1['p10']:.1f}, p90={q1['p90']:.1f}]")

    print("\n=== Query 2: Ness's strength rank out of 100 athletes ===")
    q2 = intrinsic_strength_rank(samples, 'ness', 100)
    print(f"  mean={q2['mean']:.1f}  std={q2['std']:.1f}  [p10={q2['p10']:.1f}, p90={q2['p90']:.1f}]")

    print("\n=== Query 3: Blake's strength rank out of 100 athletes ===")
    q3 = intrinsic_strength_rank(samples, 'blake', 100)
    print(f"  mean={q3['mean']:.1f}  std={q3['std']:.1f}  [p10={q3['p10']:.1f}, p90={q3['p90']:.1f}]")

    print("\n=== Query 4: Kay's shooting accuracy in round 1 ===")
    q4 = query_shooting_accuracy_in_round(samples, 'kay', 1)
    print(f"  mean={q4['mean']:.1f}  std={q4['std']:.1f}  [p10={q4['p10']:.1f}, p90={q4['p90']:.1f}]")

    print("\n=== Query 5: Ness's shooting accuracy in round 1 ===")
    q5 = query_shooting_accuracy_in_round(samples, 'ness', 1)
    print(f"  mean={q5['mean']:.1f}  std={q5['std']:.1f}  [p10={q5['p10']:.1f}, p90={q5['p90']:.1f}]")

    print("\n=== Query 6: Blake's shooting accuracy in round 1 ===")
    q6 = query_shooting_accuracy_in_round(samples, 'blake', 1)
    print(f"  mean={q6['mean']:.1f}  std={q6['std']:.1f}  [p10={q6['p10']:.1f}, p90={q6['p90']:.1f}]")

    print("\n=== Query 7: P(Taylor+Blake beat Kay+Ness) in a future round ===")
    q7 = who_would_win_by_how_much(samples, ['kay', 'ness'], ['taylor', 'blake'])
    print(f"  P(team2 wins) = {q7['p_team2_wins']:.3f}  [p10={q7['p10']:.3f}, p90={q7['p90']:.3f}]")

    print("\n=== Query 8: P(Ness+Fey beat Kay+Blake) in a future round ===")
    q8 = who_would_win_by_how_much(samples, ['kay', 'blake'], ['ness', 'fey'])
    print(f"  P(team2 wins) = {q8['p_team2_wins']:.3f}  [p10={q8['p10']:.3f}, p90={q8['p90']:.3f}]")

    # ── Save full posterior distributions ─────────────────────────────────────
    tm = datetime.datetime.now()
    result = {
        'query1': {'description': "Kay's strength rank out of 100 athletes", 'samples': q1['raw'], 'mean': q1['mean'], 'std': q1['std']},
        'query2': {'description': "Ness's strength rank out of 100 athletes", 'samples': q2['raw'], 'mean': q2['mean'], 'std': q2['std']},
        'query3': {'description': "Blake's strength rank out of 100 athletes", 'samples': q3['raw'], 'mean': q3['mean'], 'std': q3['std']},
        'query4': {'description': "Kay's shooting accuracy in round 1", 'samples': q4['raw'], 'mean': q4['mean'], 'std': q4['std']},
        'query5': {'description': "Ness's shooting accuracy in round 1", 'samples': q5['raw'], 'mean': q5['mean'], 'std': q5['std']},
        'query6': {'description': "Blake's shooting accuracy in round 1", 'samples': q6['raw'], 'mean': q6['mean'], 'std': q6['std']},
        'query7': {'description': 'P(Taylor+Blake beat Kay+Ness)', 'samples': q7['raw'], 'mean': q7['p_team2_wins']},
        'query8': {'description': 'P(Ness+Fey beat Kay+Blake)', 'samples': q8['raw'], 'mean': q8['p_team2_wins']},
    }
    os.makedirs('inference_results', exist_ok=True)
    out_path = f'inference_results/result-biathlon-pytorch.json'
    with open(out_path, 'w') as f:
        json.dump(result, f)