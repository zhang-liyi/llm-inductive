"""Mechanically convert pg-gemini-{domain}-...py (NUTS) into pg-gemini-REJ-{domain}-...py.

The model body and queries are unchanged; only the inference scaffolding swaps
(NUTS imports, `run_inference` definition, `__main__` invocation, out_path tag).
This matches the existing REJ programs already on disk and mirrors the
template-level conversion done by make_part3_rej.py.
"""
import argparse
import os
import re
from pathlib import Path

ROOT = Path('./posterior_sampling_pytorch')
PROG = ROOT / 'programs'

OLD_IMPORT = 'from pyro.infer import MCMC, NUTS\n'
NEW_IMPORT = (
    'import math as _math\n'
    'import random as _random\n'
    'from pyro.poutine import trace as _pt_trace\n'
    'from pyro.distributions import Unit as _Unit\n'
)

OLD_RUN_RE = re.compile(
    r'def run_inference\(num_samples=500, warmup_steps=200\):\n'
    r'    kernel\s*=\s*NUTS\(model, adapt_step_size=True, target_accept_prob=0\.8\)\n'
    r'    mcmc\s*=\s*MCMC\(kernel, num_samples=num_samples, warmup_steps=warmup_steps, num_chains=4\)\n'
    r'    mcmc\.run\(\)\n'
    r'    return mcmc\n'
)
NEW_RUN = (
    'def run_inference(num_samples=1000, max_attempts=10_000_000):\n'
    '    """Importance-weighted rejection sampling.\n'
    '\n'
    '    Target posterior = prior * prod_i sigmoid(diff_i / BEAT_TEMP), exactly the\n'
    '    distribution the REJ + pyro.factor(logsigmoid(...)) formulation targets.\n'
    '    Procedure: draw from the prior (via a model trace), sum the log-factor\n'
    '    values at every Unit site (= sum logsigmoid(diff/T) <= 0), and accept with\n'
    '    probability exp(sum). Accepted samples are i.i.d. from the soft posterior.\n'
    '    Single chain, 1000 accepted samples.\n'
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
    '    return {k: torch.stack(v) for k, v in kept.items()}\n'
)

OLD_MAIN_RE = re.compile(
    r'    print\("Running NUTS inference on (.+?) model ?(?:…|\.\.\.)"\)\n'
    r'    mcmc\s*=\s*run_inference\(num_samples=500, warmup_steps=200\)\n'
    r'    samples\s*=\s*mcmc\.get_samples\(\)\n'
    r'    mcmc\.summary\(\)\n'
)


def convert(text):
    if OLD_IMPORT not in text:
        raise ValueError('NUTS import not found')
    text = text.replace(OLD_IMPORT, NEW_IMPORT, 1)

    text, n = OLD_RUN_RE.subn(NEW_RUN, text, count=1)
    if n != 1:
        raise ValueError('NUTS run_inference block not found')

    def _main_repl(m):
        model_name = m.group(1)
        return (
            f'    print("Running importance-weighted rejection on {model_name} model '
            f'(1000 accepted samples) ...")\n'
            f'    samples = run_inference(num_samples=1000)\n'
            f'    print(f"  num accepted = {{next(iter(samples.values())).shape[0]}}")\n'
        )

    text, n = OLD_MAIN_RE.subn(_main_repl, text, count=1)
    if n != 1:
        raise ValueError('__main__ NUTS invocation not found')

    text = text.replace('result-gemini-NUTS-', 'result-gemini-REJ-')
    return text


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()

    targets = []
    for N in (5, 6, 7):
        targets.append(('general', N))
    for N in (6, 7):
        targets.append(('healthcare', N))

    converted = skipped = failed = 0
    for domain, N in targets:
        prefix = f'pg-gemini-{domain}-'
        suffix_n = f'-N-{N}-'
        for fname in sorted(os.listdir(PROG)):
            if not fname.startswith(prefix):
                continue
            if 'REJ' in fname:
                continue
            if suffix_n not in fname:
                continue
            src = PROG / fname
            dst_name = fname.replace(prefix, f'pg-gemini-REJ-{domain}-', 1)
            dst = PROG / dst_name
            if dst.exists():
                skipped += 1
                continue
            try:
                new_text = convert(src.read_text())
            except ValueError as e:
                print(f'FAIL {fname}: {e}')
                failed += 1
                continue
            if args.dry_run:
                print(f'[dry-run] would write {dst_name}')
            else:
                dst.write_text(new_text)
            converted += 1

    print(f'converted={converted}  skipped_existing={skipped}  failed={failed}')


if __name__ == '__main__':
    main()
