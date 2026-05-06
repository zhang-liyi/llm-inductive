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
ATHLETES = ['peyton', 'fey', 'indiana', 'max', 'casey', 'kay']

# ── (athlete, round) pairs from the conditions ────────────────────────────────
ATHLETE_ROUNDS = [
    ('peyton', 1),
    ('fey', 1),
    ('indiana', 1),
    ('max', 1),
    ('fey', 2),
    ('casey', 2),
    ('indiana', 2),
    ('max', 2),
    ('fey', 3),
    ('kay', 3),
    ('indiana', 3),
    ('max', 3),
]

# Temperature for soft beat/lost likelihoods.
BEAT_TEMP = 100.0

# ── Shared priors (used in model and query helpers) ───────────────────────────
# Intrinsic strength prior (similar to skill in the example)
_STRENGTH_PRIOR = dist.MixtureSameFamily(
    dist.Categorical(probs=torch.tensor([0.33, 0.33, 0.34])),
    dist.Normal(torch.tensor([80., 100., 140.]), torch.tensor([10., 10., 10.])),
)

# Shooting accuracy prior (independent of strength, as per scratchpad)
_SHOOTING_PRIOR = dist.MixtureSameFamily(
    dist.Categorical(probs=torch.tensor([0.33, 0.33, 0.34])),
    dist.Normal(torch.tensor([30., 60., 90.]), torch.tensor([10., 10., 10.])),
)


# ── Pyro model ────────────────────────────────────────────────────────────────

def model():
    # ── intrinsic_strength (mem'd per athlete) ────────────────────────────────
    intrinsic_strength = {}
    for athlete in ATHLETES:
        raw = pyro.sample(f"intrinsic_strength_{athlete}", _STRENGTH_PRIOR)
        intrinsic_strength[athlete] = raw.clamp(min=0.)

    # ── shooting_accuracy_in_round (mem'd per athlete × round) ────────────────
    shooting_accuracy_in_round = {}
    for (athlete, round_id) in ATHLETE_ROUNDS:
        raw = pyro.sample(
            f"shooting_accuracy_{athlete}_r{round_id}",
            _SHOOTING_PRIOR,
        )
        shooting_accuracy_in_round[(athlete, round_id)] = raw.clamp(0., 100.)

    # ── Derived quantities (deterministic given samples) ──────────────────────

    def skiing_speed(athlete):
        # Skiing speed is directly causally dependent on intrinsic strength
        return intrinsic_strength[athlete]

    def team_skiing_speed(team):
        return sum(skiing_speed(a) for a in team) / len(team)

    def team_shooting_accuracy_in_round(team, round_id):
        return sum(shooting_accuracy_in_round[(a, round_id)] for a in team) / len(team)

    def overall_team_score(team, round_id):
        # Overall score combines average skiing speed and average shooting accuracy
        return team_skiing_speed(team) + team_shooting_accuracy_in_round(team, round_id)

    # ── Conditions (soft likelihoods) ─────────────────────────────────────────
    def soft_beat(team1, team2, round_id, name):
        score_diff = overall_team_score(team1, round_id) - overall_team_score(team2, round_id)
        pyro.factor(name, F.logsigmoid(score_diff / BEAT_TEMP))

    def soft_lost(team1, team2, round_id, name):
        score_diff = overall_team_score(team1, round_id) - overall_team_score(team2, round_id)
        pyro.factor(name, F.logsigmoid(-score_diff / BEAT_TEMP))

    # In the first round, Peyton and Fey lost to Indiana and Max.
    soft_lost(['peyton', 'fey'], ['indiana', 'max'], 1, 'cond_1')
    # In the second round, Fey and Casey beat Indiana and Max.
    soft_beat(['fey', 'casey'], ['indiana', 'max'], 2, 'cond_2')
    # In the third round, Fey and Kay beat Indiana and Max.
    soft_beat(['fey', 'kay'], ['indiana', 'max'], 3, 'cond_3')


# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference(num_samples=500, warmup_steps=200):
    kernel = NUTS(model, adapt_step_size=True, target_accept_prob=0.8)
    mcmc   = MCMC(kernel, num_samples=num_samples, warmup_steps=warmup_steps, num_chains=4)
    mcmc.run()
    return mcmc


# ── Query helpers (post-hoc) ──────────────────────────────────────────────────

def intrinsic_strength_rank(samples, athlete='peyton', out_of_n_athletes=100):
    """
    Queries 1-3: Out of 100 random athletes, where does athlete rank?
    """
    s_post = samples[f'intrinsic_strength_{athlete}'].double()
    others = _STRENGTH_PRIOR.sample((s_post.shape[0], out_of_n_athletes - 1)).clamp(min=0.)
    rank   = (s_post.unsqueeze(1) > others).double().sum(dim=1)
    return dict(mean=rank.mean().item(), std=rank.std().item(),
                p10=rank.quantile(0.10).item(), p90=rank.quantile(0.90).item(),
                raw=rank.tolist())


def query_shooting_accuracy_in_round(samples, athlete='peyton', round_id=1):
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
            def sim_score(team):
                team_speed = sum(strength[a] for a in team) / len(team)
                team_shooting = sum(_SHOOTING_PRIOR.sample().clamp(0., 100.) for a in team) / len(team)
                return team_speed + team_shooting

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

    print("\n=== Query 1: Peyton's strength rank out of 100 athletes ===")
    q1 = intrinsic_strength_rank(samples, 'peyton', 100)
    print(f"  mean={q1['mean']:.1f}  std={q1['std']:.1f}  [p10={q1['p10']:.1f}, p90={q1['p90']:.1f}]")

    print("\n=== Query 2: Fey's strength rank out of 100 athletes ===")
    q2 = intrinsic_strength_rank(samples, 'fey', 100)
    print(f"  mean={q2['mean']:.1f}  std={q2['std']:.1f}  [p10={q2['p10']:.1f}, p90={q2['p90']:.1f}]")

    print("\n=== Query 3: Indiana's strength rank out of 100 athletes ===")
    q3 = intrinsic_strength_rank(samples, 'indiana', 100)
    print(f"  mean={q3['mean']:.1f}  std={q3['std']:.1f}  [p10={q3['p10']:.1f}, p90={q3['p90']:.1f}]")

    print("\n=== Query 4: Peyton's shooting accuracy in round 1 (0-100) ===")
    q4 = query_shooting_accuracy_in_round(samples, 'peyton', 1)
    print(f"  mean={q4['mean']:.1f}  std={q4['std']:.1f}  [p10={q4['p10']:.1f}, p90={q4['p90']:.1f}]")

    print("\n=== Query 5: Fey's shooting accuracy in round 1 (0-100) ===")
    q5 = query_shooting_accuracy_in_round(samples, 'fey', 1)
    print(f"  mean={q5['mean']:.1f}  std={q5['std']:.1f}  [p10={q5['p10']:.1f}, p90={q5['p90']:.1f}]")

    print("\n=== Query 6: Indiana's shooting accuracy in round 1 (0-100) ===")
    q6 = query_shooting_accuracy_in_round(samples, 'indiana', 1)
    print(f"  mean={q6['mean']:.1f}  std={q6['std']:.1f}  [p10={q6['p10']:.1f}, p90={q6['p90']:.1f}]")

    print("\n=== Query 7: P(Casey+Indiana beat Peyton+Fey) in a future round ===")
    q7 = who_would_win_by_how_much(samples, ['peyton', 'fey'], ['casey', 'indiana'])
    print(f"  P(team2 wins) = {q7['p_team2_wins']:.3f}  [p10={q7['p10']:.3f}, p90={q7['p90']:.3f}]")

    print("\n=== Query 8: P(Fey+Max beat Peyton+Indiana) in a future round ===")
    q8 = who_would_win_by_how_much(samples, ['peyton', 'indiana'], ['fey', 'max'])
    print(f"  P(team2 wins) = {q8['p_team2_wins']:.3f}  [p10={q8['p10']:.3f}, p90={q8['p90']:.3f}]")

    # ── Save full posterior distributions ─────────────────────────────────────
    tm = datetime.datetime.now()
    result = {
        'query1': {'description': "Peyton's strength rank out of 100 athletes",
                   'samples': q1['raw'], 'mean': q1['mean'], 'std': q1['std']},
        'query2': {'description': "Fey's strength rank out of 100 athletes",
                   'samples': q2['raw'], 'mean': q2['mean'], 'std': q2['std']},
        'query3': {'description': "Indiana's strength rank out of 100 athletes",
                   'samples': q3['raw'], 'mean': q3['mean'], 'std': q3['std']},
        'query4': {'description': 'Peyton shooting accuracy round 1 (0-100)',
                   'samples': q4['raw'], 'mean': q4['mean'], 'std': q4['std']},
        'query5': {'description': 'Fey shooting accuracy round 1 (0-100)',
                   'samples': q5['raw'], 'mean': q5['mean'], 'std': q5['std']},
        'query6': {'description': 'Indiana shooting accuracy round 1 (0-100)',
                   'samples': q6['raw'], 'mean': q6['mean'], 'std': q6['std']},
        'query7': {'description': 'P(Casey+Indiana beat Peyton+Fey)',
                   'samples': q7['raw'], 'mean': q7['p_team2_wins']},
        'query8': {'description': 'P(Fey+Max beat Peyton+Indiana)',
                   'samples': q8['raw'], 'mean': q8['p_team2_wins']},
    }
    
    os.makedirs('inference_results', exist_ok=True)
    out_path = f'inference_results/result-biathlon-pytorch.json'
    with open(out_path, 'w') as f:
        json.dump(result, f)