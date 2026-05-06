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
ATHLETES = ['indiana', 'ness', 'casey', 'lane', 'kay', 'max', 'drew', 'peyton']

# ── (team, round) pairs from the conditions, in original order ────────────────
TEAM_ROUNDS = [
    (['indiana', 'ness'], 1),
    (['casey',   'lane'], 1),
    (['indiana', 'ness'], 2),
    (['kay',     'max'],  2),
    (['indiana', 'ness'], 3),
    (['drew',    'peyton'], 3),
]

# Temperature for soft beat/lost likelihoods.
BEAT_TEMP = 100.0

# ── Shared strength prior (used in model and query helpers) ───────────────────
_STRENGTH_PRIOR = dist.MixtureSameFamily(
    dist.Categorical(probs=torch.tensor([0.33, 0.33, 0.34])),
    dist.Normal(torch.tensor([80., 100., 140.]), torch.tensor([10., 10., 10.])),
)

def _shooting_accuracy_prior(strength):
    """
    3-component Gaussian mixture for shooting accuracy.
    Prior weights depend on intrinsic strength, mirroring the WebPPL if/elif.
    """
    s = strength
    p_low      = torch.where(s > 120, torch.tensor(0.05),
                 torch.where(s <  80, torch.tensor(0.80), torch.tensor(0.20)))
    p_medium   = torch.where(s > 120, torch.tensor(0.15),
                 torch.where(s <  80, torch.tensor(0.15), torch.tensor(0.60)))
    p_high     = torch.where(s > 120, torch.tensor(0.80),
                 torch.where(s <  80, torch.tensor(0.05), torch.tensor(0.20)))
    return dist.MixtureSameFamily(
        dist.Categorical(probs=torch.stack([p_low, p_medium, p_high])),
        dist.Normal(torch.tensor([30., 60., 90.]), torch.tensor([10., 10., 10.])),
    )


# ── Pyro model ────────────────────────────────────────────────────────────────

def model():
    # ── intrinsic_strength (mem'd per athlete) ────────────────────────────────
    intrinsic_strength = {}
    for athlete in ATHLETES:
        raw = pyro.sample(f"intrinsic_strength_{athlete}", _STRENGTH_PRIOR)
        intrinsic_strength[athlete] = raw.clamp(min=0.)

    # ── shooting_accuracy_in_round (mem'd per athlete × round) ───────────────
    shooting_accuracy_in_round = {}
    for (team, round_id) in TEAM_ROUNDS:
        for athlete in team:
            if (athlete, round_id) not in shooting_accuracy_in_round:
                raw = pyro.sample(
                    f"shooting_accuracy_{athlete}_r{round_id}",
                    _shooting_accuracy_prior(intrinsic_strength[athlete]),
                )
                shooting_accuracy_in_round[(athlete, round_id)] = raw.clamp(0., 100.)

    # ── Derived quantities (deterministic given samples) ──────────────────────

    def team_skiing_speed(team):
        return sum(intrinsic_strength[a] for a in team) / len(team)

    def team_shooting_accuracy_in_round(team, round_id):
        return sum(shooting_accuracy_in_round[(a, round_id)] for a in team) / len(team)

    def overall_team_score(team, round_id):
        return team_skiing_speed(team) + team_shooting_accuracy_in_round(team, round_id)

    # ── Conditions (soft likelihoods) ─────────────────────────────────────────
    # lost(team1, team2, round)  →  score_diff < 0  →  factor(logsigmoid(-diff/T))

    def soft_lost(team1, team2, round_id, name):
        score_diff = overall_team_score(team1, round_id) - overall_team_score(team2, round_id)
        pyro.factor(name, F.logsigmoid(-score_diff / BEAT_TEMP))

    # In the first round, Indiana and Ness lost to Casey and Lane.
    soft_lost(['indiana', 'ness'], ['casey', 'lane'], 1, 'cond_1')
    # In the second round, Indiana and Ness lost to Kay and Max.
    soft_lost(['indiana', 'ness'], ['kay', 'max'], 2, 'cond_2')
    # In the third round, Indiana and Ness lost to Drew and Peyton.
    soft_lost(['indiana', 'ness'], ['drew', 'peyton'], 3, 'cond_3')


# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference(num_samples=500, warmup_steps=200):
    kernel = NUTS(model, adapt_step_size=True, target_accept_prob=0.8)
    mcmc   = MCMC(kernel, num_samples=num_samples, warmup_steps=warmup_steps, num_chains=4)
    mcmc.run()
    return mcmc


# ── Query helpers (post-hoc, matching WebPPL queries) ────────────────────────

def intrinsic_strength_rank(samples, athlete, out_of_n_athletes=100):
    """
    Queries 1, 2, 3: Out of 100 random athletes, where does athlete rank?
    Mirrors: intrinsic_strength_rank(samples, athlete, 100)
    """
    s_post = samples[f'intrinsic_strength_{athlete}'].double()
    others = _STRENGTH_PRIOR.sample((s_post.shape[0], out_of_n_athletes - 1)).clamp(min=0.)
    rank   = (s_post.unsqueeze(1) > others).double().sum(dim=1)
    return dict(mean=rank.mean().item(), std=rank.std().item(),
                p10=rank.quantile(0.10).item(), p90=rank.quantile(0.90).item(),
                raw=rank.tolist())


def query_shooting_accuracy_in_round(samples, athlete, round_id):
    """
    Queries 4, 5, 6: Shooting accuracy for an athlete in a given round (0–100 scale).
    Mirrors: query_shooting_accuracy_in_round(samples, athlete, round_id)
    """
    key = f"shooting_accuracy_{athlete}_r{round_id}"
    d = samples[key].double().clamp(0., 100.)
    return dict(mean=d.mean().item(), std=d.std().item(),
                p10=d.quantile(0.10).item(), p90=d.quantile(0.90).item(),
                raw=d.tolist())


def who_would_win_by_how_much(samples, team1, team2, n_future=100):
    """
    Queries 7 & 8: P(team2 wins) over simulated future rounds.
    Mirrors: who_would_win_by_how_much(samples, team1, team2)
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
            def sim_score(team):
                speed = sum(strength[a] for a in team) / len(team)
                acc = sum(_shooting_accuracy_prior(strength[a]).sample().clamp(0., 100.) for a in team) / len(team)
                return speed + acc

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

    print("\n=== Query 1: Indiana's strength rank out of 100 athletes ===")
    q1 = intrinsic_strength_rank(samples, 'indiana', 100)
    print(f"  mean={q1['mean']:.1f}  std={q1['std']:.1f}  "
          f"[p10={q1['p10']:.1f}, p90={q1['p90']:.1f}]")

    print("\n=== Query 2: Ness's strength rank out of 100 athletes ===")
    q2 = intrinsic_strength_rank(samples, 'ness', 100)
    print(f"  mean={q2['mean']:.1f}  std={q2['std']:.1f}  "
          f"[p10={q2['p10']:.1f}, p90={q2['p90']:.1f}]")

    print("\n=== Query 3: Drew's strength rank out of 100 athletes ===")
    q3 = intrinsic_strength_rank(samples, 'drew', 100)
    print(f"  mean={q3['mean']:.1f}  std={q3['std']:.1f}  "
          f"[p10={q3['p10']:.1f}, p90={q3['p90']:.1f}]")

    print("\n=== Query 4: Indiana's shooting accuracy in round 3 (0-100) ===")
    q4 = query_shooting_accuracy_in_round(samples, 'indiana', 3)
    print(f"  mean={q4['mean']:.1f}  std={q4['std']:.1f}  "
          f"[p10={q4['p10']:.1f}, p90={q4['p90']:.1f}]")

    print("\n=== Query 5: Ness's shooting accuracy in round 3 (0-100) ===")
    q5 = query_shooting_accuracy_in_round(samples, 'ness', 3)
    print(f"  mean={q5['mean']:.1f}  std={q5['std']:.1f}  "
          f"[p10={q5['p10']:.1f}, p90={q5['p90']:.1f}]")

    print("\n=== Query 6: Drew's shooting accuracy in round 3 (0-100) ===")
    q6 = query_shooting_accuracy_in_round(samples, 'drew', 3)
    print(f"  mean={q6['mean']:.1f}  std={q6['std']:.1f}  "
          f"[p10={q6['p10']:.1f}, p90={q6['p90']:.1f}]")

    print("\n=== Query 7: P(Casey+Peyton beat Indiana+Ness) in a future round ===")
    q7 = who_would_win_by_how_much(samples, ['indiana', 'ness'], ['casey', 'peyton'])
    print(f"  P(team2 wins) = {q7['p_team2_wins']:.3f}  "
          f"[p10={q7['p10']:.3f}, p90={q7['p90']:.3f}]")

    print("\n=== Query 8: P(Ness+Lane beat Indiana+Casey) in a future round ===")
    q8 = who_would_win_by_how_much(samples, ['indiana', 'casey'], ['ness', 'lane'])
    print(f"  P(team2 wins) = {q8['p_team2_wins']:.3f}  "
          f"[p10={q8['p10']:.3f}, p90={q8['p90']:.3f}]")

    # ── Save full posterior distributions ─────────────────────────────────────
    tm = datetime.datetime.now()
    result = {
        'query1': {'description': "Indiana's strength rank out of 100 athletes",
                   'samples': q1['raw'], 'mean': q1['mean'], 'std': q1['std']},
        'query2': {'description': "Ness's strength rank out of 100 athletes",
                   'samples': q2['raw'], 'mean': q2['mean'], 'std': q2['std']},
        'query3': {'description': "Drew's strength rank out of 100 athletes",
                   'samples': q3['raw'], 'mean': q3['mean'], 'std': q3['std']},
        'query4': {'description': "Indiana's shooting accuracy in round 3",
                   'samples': q4['raw'], 'mean': q4['mean'], 'std': q4['std']},
        'query5': {'description': "Ness's shooting accuracy in round 3",
                   'samples': q5['raw'], 'mean': q5['mean'], 'std': q5['std']},
        'query6': {'description': "Drew's shooting accuracy in round 3",
                   'samples': q6['raw'], 'mean': q6['mean'], 'std': q6['std']},
        'query7': {'description': 'P(Casey+Peyton beat Indiana+Ness)',
                   'samples': q7['raw'], 'mean': q7['p_team2_wins']},
        'query8': {'description': 'P(Ness+Lane beat Indiana+Casey)',
                   'samples': q8['raw'], 'mean': q8['p_team2_wins']},
    }
    out_path = f'inference_results/result-biathlon-pytorch.json'
    with open(out_path, 'w') as f:
        json.dump(result, f)