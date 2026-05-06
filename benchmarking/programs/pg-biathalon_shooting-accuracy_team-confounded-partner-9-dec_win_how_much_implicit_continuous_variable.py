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
ATHLETES = ['robin', 'ollie', 'lane', 'ness', 'sam', 'blake', 'val', 'taylor']

# ── (team, round) pairs from the conditions, in original order ────────────────
TEAM_ROUNDS = [
    (['robin', 'ollie'], 1),
    (['lane',  'ness'],  1),
    (['robin', 'ollie'], 2),
    (['sam',   'blake'], 2),
    (['robin', 'ollie'], 3),
    (['val',   'taylor'],3),
]

# Temperature for soft beat/lost likelihoods.
# overall_team_score is O(~100); BEAT_TEMP=5.0 makes the soft boundary
# sharp but everywhere-finite and smooth.
BEAT_TEMP = 5.0

# ── Shared priors (used in model and query helpers) ───────────────────────────
# The scratchpad explicitly specifies Normal distributions for strength and shooting.
_STRENGTH_PRIOR = dist.Normal(50., 15.)
_SHOOTING_PRIOR = dist.Normal(50., 15.)

def _team_key(team, round_id):
    """Canonical sample-site name for a (team, round) pair."""
    return '_'.join(team) + f'_r{round_id}'

# ── Pyro model ────────────────────────────────────────────────────────────────
def model():
    # ── intrinsic_strength (mem'd per athlete) ───────────────────────────────
    intrinsic_strength = {}
    for athlete in ATHLETES:
        raw = pyro.sample(f"intrinsic_strength_{athlete}", _STRENGTH_PRIOR)
        intrinsic_strength[athlete] = raw.clamp(min=0.)

    # ── skiing_speed and shooting_accuracy (mem'd per athlete × round) ───────
    skiing_speed = {}
    shooting_accuracy = {}
    
    for team, round_id in TEAM_ROUNDS:
        for athlete in team:
            key = f"{athlete}_r{round_id}"
            if key not in skiing_speed:
                # Skiing speed is normally distributed around intrinsic strength
                speed = pyro.sample(f"skiing_speed_{key}", dist.Normal(intrinsic_strength[athlete], 10.0))
                skiing_speed[key] = speed.clamp(min=0.)
                
                # Shooting accuracy is independent of intrinsic strength
                acc = pyro.sample(f"shooting_accuracy_{key}", _SHOOTING_PRIOR)
                shooting_accuracy[key] = acc.clamp(0., 100.)

    # ── Derived quantities (deterministic given samples) ──────────────────────
    def team_skiing_speed(team, round_id):
        return sum(skiing_speed[f"{a}_r{round_id}"] for a in team) / len(team)

    def team_shooting_accuracy_in_round(team, round_id):
        return sum(shooting_accuracy[f"{a}_r{round_id}"] for a in team) / len(team)

    def overall_team_score(team, round_id):
        return team_skiing_speed(team, round_id) + team_shooting_accuracy_in_round(team, round_id)

    # ── Conditions (soft likelihoods) ─────────────────────────────────────────
    def soft_beat(team1, team2, round_id, name):
        score_diff = overall_team_score(team1, round_id) - overall_team_score(team2, round_id)
        pyro.factor(name, F.logsigmoid(score_diff / BEAT_TEMP))

    # In the first round, Robin and Ollie beat Lane and Ness.
    soft_beat(['robin', 'ollie'], ['lane', 'ness'], 1, 'cond_1')
    # In the second round, Robin and Ollie beat Sam and Blake.
    soft_beat(['robin', 'ollie'], ['sam', 'blake'], 2, 'cond_2')
    # In the third round, Robin and Ollie beat Val and Taylor.
    soft_beat(['robin', 'ollie'], ['val', 'taylor'], 3, 'cond_3')

# ── Inference ─────────────────────────────────────────────────────────────────
def run_inference(num_samples=500, warmup_steps=200):
    kernel = NUTS(model, adapt_step_size=True, target_accept_prob=0.8)
    mcmc   = MCMC(kernel, num_samples=num_samples, warmup_steps=warmup_steps, num_chains=4)
    mcmc.run()
    return mcmc

# ── Query helpers ─────────────────────────────────────────────────────────────
def intrinsic_strength_rank(samples, athlete, out_of_n_athletes=100):
    s_post = samples[f'intrinsic_strength_{athlete}'].double()
    others = _STRENGTH_PRIOR.sample((s_post.shape[0], out_of_n_athletes - 1)).clamp(min=0.)
    rank   = (s_post.unsqueeze(1) > others).double().sum(dim=1)
    return dict(mean=rank.mean().item(), std=rank.std().item(),
                p10=rank.quantile(0.10).item(), p90=rank.quantile(0.90).item(),
                raw=rank.tolist())

def query_shooting_accuracy_in_round(samples, athlete, round_id):
    key = f"shooting_accuracy_{athlete}_r{round_id}"
    d = samples[key].double().clamp(0., 100.)
    return dict(mean=d.mean().item(), std=d.std().item(),
                p10=d.quantile(0.10).item(), p90=d.quantile(0.90).item(),
                raw=d.tolist())

def who_would_win_by_how_much(samples, team1, team2, n_future=100):
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
            def sim_score(team):
                team_speed = sum(dist.Normal(strength[a], 10.0).sample().clamp(min=0.) for a in team) / len(team)
                team_acc = sum(_SHOOTING_PRIOR.sample().clamp(0., 100.) for a in team) / len(team)
                return team_speed + team_acc

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

    print("\n=== Query 2: Ollie's strength rank out of 100 athletes ===")
    q2 = intrinsic_strength_rank(samples, 'ollie', 100)
    print(f"  mean={q2['mean']:.1f}  std={q2['std']:.1f}  [p10={q2['p10']:.1f}, p90={q2['p90']:.1f}]")

    print("\n=== Query 3: Sam's strength rank out of 100 athletes ===")
    q3 = intrinsic_strength_rank(samples, 'sam', 100)
    print(f"  mean={q3['mean']:.1f}  std={q3['std']:.1f}  [p10={q3['p10']:.1f}, p90={q3['p90']:.1f}]")

    print("\n=== Query 4: Robin's shooting accuracy in round 2 ===")
    q4 = query_shooting_accuracy_in_round(samples, 'robin', 2)
    print(f"  mean={q4['mean']:.1f}  std={q4['std']:.1f}  [p10={q4['p10']:.1f}, p90={q4['p90']:.1f}]")

    print("\n=== Query 5: Ollie's shooting accuracy in round 2 ===")
    q5 = query_shooting_accuracy_in_round(samples, 'ollie', 2)
    print(f"  mean={q5['mean']:.1f}  std={q5['std']:.1f}  [p10={q5['p10']:.1f}, p90={q5['p90']:.1f}]")

    print("\n=== Query 6: Sam's shooting accuracy in round 2 ===")
    q6 = query_shooting_accuracy_in_round(samples, 'sam', 2)
    print(f"  mean={q6['mean']:.1f}  std={q6['std']:.1f}  [p10={q6['p10']:.1f}, p90={q6['p90']:.1f}]")

    print("\n=== Query 7: P(Lane+Taylor beat Robin+Ollie) in a future round ===")
    q7 = who_would_win_by_how_much(samples, ['robin', 'ollie'], ['lane', 'taylor'])
    print(f"  P(team2 wins) = {q7['p_team2_wins']:.3f}  [p10={q7['p10']:.3f}, p90={q7['p90']:.3f}]")

    print("\n=== Query 8: P(Ollie+Ness beat Robin+Lane) in a future round ===")
    q8 = who_would_win_by_how_much(samples, ['robin', 'lane'], ['ollie', 'ness'])
    print(f"  P(team2 wins) = {q8['p_team2_wins']:.3f}  [p10={q8['p10']:.3f}, p90={q8['p90']:.3f}]")

    # ── Save full posterior distributions ─────────────────────────────────────
    tm = datetime.datetime.now()
    result = {
        'query1': {'description': "Robin's strength rank out of 100 athletes", 'samples': q1['raw'], 'mean': q1['mean'], 'std': q1['std']},
        'query2': {'description': "Ollie's strength rank out of 100 athletes", 'samples': q2['raw'], 'mean': q2['mean'], 'std': q2['std']},
        'query3': {'description': "Sam's strength rank out of 100 athletes", 'samples': q3['raw'], 'mean': q3['mean'], 'std': q3['std']},
        'query4': {'description': "Robin's shooting accuracy in round 2", 'samples': q4['raw'], 'mean': q4['mean'], 'std': q4['std']},
        'query5': {'description': "Ollie's shooting accuracy in round 2", 'samples': q5['raw'], 'mean': q5['mean'], 'std': q5['std']},
        'query6': {'description': "Sam's shooting accuracy in round 2", 'samples': q6['raw'], 'mean': q6['mean'], 'std': q6['std']},
        'query7': {'description': "P(Lane+Taylor beat Robin+Ollie)", 'samples': q7['raw'], 'mean': q7['p_team2_wins']},
        'query8': {'description': "P(Ollie+Ness beat Robin+Lane)", 'samples': q8['raw'], 'mean': q8['p_team2_wins']},
    }
    os.makedirs('inference_results', exist_ok=True)
    out_path = f'inference_results/result-biathlon-pytorch.json'
    with open(out_path, 'w') as f:
        json.dump(result, f)