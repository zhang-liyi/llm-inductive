"""Create pg-part3-rej.txt: rejection-sampling variant of pg-part3.txt.

Keeps `soft_beat` / `soft_lost` (and their `pyro.factor(F.logsigmoid(diff / BEAT_TEMP))`
implementation) intact — the model and its target posterior are unchanged.
Only the inference method is swapped: NUTS → importance-weighted rejection.

Target posterior:   prior × ∏ᵢ sigmoid(diffᵢ / BEAT_TEMP).
Envelope:           the prior (so the acceptance probability is ∏ sigmoid ≤ 1).
Accepted samples are i.i.d. draws from the same posterior NUTS targets.
"""
import re
from pathlib import Path

SRC = Path('./posterior_sampling_pytorch/pg-part3.txt')
DST = Path('./posterior_sampling_pytorch/pg-part3-rej.txt')

text = SRC.read_text()

# Swap NUTS import for the imports the importance-weighted rejection sampler
# needs (math, random, the trace messenger, and pyro's Unit distribution).
text = text.replace(
    'from pyro.infer import MCMC, NUTS\n',
    'import math as _math\n'
    'import random as _random\n'
    'from pyro.poutine import trace as _pt_trace\n'
    'from pyro.distributions import Unit as _Unit\n',
    1,
)

# Replace the NUTS run_inference body with an importance-weighted rejection
# sampler targeting 1000 accepted samples from the same soft posterior.
text = re.sub(
    r'def run_inference\(num_samples=500, warmup_steps=200\):\n'
    r'    kernel = NUTS\(model, adapt_step_size=True, target_accept_prob=0\.8\)\n'
    r'    mcmc   = MCMC\(kernel, num_samples=num_samples, warmup_steps=warmup_steps, num_chains=4\)\n'
    r'    mcmc\.run\(\)\n'
    r'    return mcmc\n',
    'def run_inference(num_samples=1000, max_attempts=10_000_000):\n'
    '    """Importance-weighted rejection sampling.\n'
    '\n'
    '    Target posterior = prior × ∏ᵢ sigmoid(diffᵢ / BEAT_TEMP), exactly the\n'
    '    distribution the NUTS + pyro.factor(logsigmoid(...)) formulation targets.\n'
    '    Procedure: draw from the prior (via a model trace), sum the log-factor\n'
    '    values at every Unit site (= Σ logsigmoid(diff/T) ≤ 0), and accept with\n'
    '    probability exp(sum). Accepted samples are i.i.d. from the soft posterior.\n'
    '    """\n'
    '    kept = {}\n'
    '    accepted = 0\n'
    '    attempts = 0\n'
    '    while accepted < num_samples and attempts < max_attempts:\n'
    '        attempts += 1\n'
    '        tr = _pt_trace(model).get_trace()\n'
    '        log_accept = 0.0\n'
    '        for _name, _node in tr.nodes.items():\n'
    '            if _node.get(\'type\') != \'sample\':\n'
    '                continue\n'
    '            _fn = _node.get(\'fn\')\n'
    '            if isinstance(_fn, _Unit):\n'
    '                _lf = _fn.log_factor\n'
    '                if hasattr(_lf, \'item\'):\n'
    '                    _lf = _lf.item()\n'
    '                log_accept += _lf\n'
    '        if _math.log(_random.random()) >= log_accept:\n'
    '            continue\n'
    '        for _name, _node in tr.nodes.items():\n'
    '            if _node.get(\'type\') != \'sample\':\n'
    '                continue\n'
    '            if isinstance(_node.get(\'fn\'), _Unit):\n'
    '                continue\n'
    '            kept.setdefault(_name, []).append(_node[\'value\'])\n'
    '        accepted += 1\n'
    '    if accepted < num_samples:\n'
    '        print(f"[WARN] rejection sampler: only {accepted}/{num_samples} "\n'
    '              f"accepted after {attempts} attempts.")\n'
    '    return {k: torch.stack(v) for k, v in kept.items()}\n',
    text,
    count=1,
)

# __main__ NUTS invocation → rejection call.
text = re.sub(
    r'    print\("Running NUTS inference on diving model …"\)\n'
    r'    mcmc    = run_inference\(num_samples=500, warmup_steps=200\)\n'
    r'    samples = mcmc\.get_samples\(\)\n'
    r'    mcmc\.summary\(\)\n',
    '    print("Running importance-weighted rejection on diving model (1000 accepted samples) …")\n'
    '    samples = run_inference(num_samples=1000)\n'
    '    print(f"  num accepted = {next(iter(samples.values())).shape[0]}")\n',
    text,
    count=1,
)

# Update the single instruction-paragraph reference to mcmc.get_samples —
# everything else about pyro.factor / soft_beat / soft_lost / logsigmoid stays
# because the model is unchanged.
text = text.replace(
    'Query functions take `samples` (the dict returned by mcmc.get_samples()) as their first argument.',
    'Query functions take `samples` (the dict returned by run_inference()) as their first argument.',
)

DST.write_text(text)
print(f'wrote {DST} ({len(text.splitlines())} lines)')
