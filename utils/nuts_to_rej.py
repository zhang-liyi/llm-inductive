"""Convert a Pyro NUTS program to a rejection-sampling program.

Two modes are supported:

  * ``mode='soft'``   – importance-weighted rejection that targets the *exact*
    soft posterior (prior * prod_i sigmoid(diff_i / BEAT_TEMP)) used by the
    NUTS + ``pyro.factor(F.logsigmoid(...))`` formulation. Accepted samples
    are i.i.d. from that posterior.
  * ``mode='hard'``   – prior rejection with the *hard* condition that every
    soft factor exceeds log(2) (equivalent to ``diff > 0`` in the original
    soft formulation). Useful when you want exact-constraint posteriors.

Usage:
    # as a library:
    from utils.nuts_to_rej import convert_text, convert_file, rej_filename
    new_text = convert_text(open(path).read(), mode='soft', num_samples=4000)

    # CLI (single file or directory):
    python -m utils.nuts_to_rej SRC DST [--mode soft|hard] [--num-samples N]
                                        [--wrap-json]
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

# ── shared imports injected into the rewritten programs ────────────────────-

REJECTION_IMPORTS_SOFT = (
    "import math as _math\n"
    "import random as _random\n"
    "from pyro.poutine import trace as _pt_trace\n"
    "from pyro.distributions import Unit as _Unit\n"
)

REJECTION_IMPORTS_HARD = (
    "import math as _math\n"
    "from pyro.poutine import trace as _pt_trace\n"
    "from pyro.distributions import Unit as _Unit\n"
)


# ── run_inference body templates (num_samples is filled in at format time) ─-

_SOFT_RUN_INFERENCE_TMPL = '''\
def run_inference(num_samples={num_samples}, max_attempts=10_000_000):
    """Importance-weighted rejection sampling.

    Target posterior = prior * prod_i sigmoid(diff_i / BEAT_TEMP), exactly the
    distribution the NUTS + pyro.factor(logsigmoid(...)) formulation targets.
    Procedure: draw from the prior (via a model trace), sum the log-factor
    values at every Unit site (= sum logsigmoid(diff/T) <= 0), and accept with
    probability exp(sum). Accepted samples are i.i.d. from the soft posterior.
    """
    kept = {{}}
    accepted = 0
    attempts = 0
    while accepted < num_samples and attempts < max_attempts:
        attempts += 1
        tr = _pt_trace(model).get_trace()
        log_accept = 0.0
        for _name, _node in tr.nodes.items():
            if _node.get('type') != 'sample':
                continue
            _fn = _node.get('fn')
            if isinstance(_fn, _Unit):
                _lf = _fn.log_factor
                if hasattr(_lf, 'item'):
                    _lf = _lf.item()
                log_accept += _lf
        if _math.log(_random.random()) >= log_accept:
            continue
        for _name, _node in tr.nodes.items():
            if _node.get('type') != 'sample':
                continue
            if isinstance(_node.get('fn'), _Unit):
                continue
            kept.setdefault(_name, []).append(_node['value'])
        accepted += 1
    if accepted < num_samples:
        print(f"[WARN] rejection sampler: only {{accepted}}/{{num_samples}} "
              f"accepted after {{attempts}} attempts.")
    return {{k: torch.stack(v) for k, v in kept.items()}}
'''


_HARD_RUN_INFERENCE_TMPL = '''\
def run_inference(num_samples={num_samples}, max_attempts=10_000_000):
    """Prior rejection sampling on the hard version of each soft_beat/soft_lost factor.

    For every factor site `pyro.factor(name, F.logsigmoid(diff / T))`, the soft
    constraint `logsigmoid(diff/T) > -log(2)` is equivalent to the hard
    condition `diff > 0`. We trace the model (which draws priors and records
    factor log-values), accept iff every factor log-value > -log(2).
    """
    _THRESHOLD = -_math.log(2.0)
    kept = {{}}
    accepted = 0
    attempts = 0
    while accepted < num_samples and attempts < max_attempts:
        attempts += 1
        tr = _pt_trace(model).get_trace()
        ok = True
        for _name, _node in tr.nodes.items():
            if _node.get('type') != 'sample':
                continue
            _fn = _node.get('fn')
            if not isinstance(_fn, _Unit):
                continue
            _lf = _fn.log_factor
            if hasattr(_lf, 'item'):
                _lf = _lf.item()
            if _lf <= _THRESHOLD:
                ok = False
                break
        if not ok:
            continue
        for _name, _node in tr.nodes.items():
            if _node.get('type') != 'sample':
                continue
            _fn = _node.get('fn')
            if isinstance(_fn, _Unit):
                continue
            kept.setdefault(_name, []).append(_node.get('value'))
        accepted += 1
    if accepted < num_samples:
        print(f"[WARN] rejection sampler: only {{accepted}}/{{num_samples}} "
              f"accepted after {{attempts}} attempts.")
    return {{k: torch.stack(v) for k, v in kept.items()}}
'''


# ── helpers ────────────────────────────────────────────────────────────────-

def _replace_imports(text: str, mode: str) -> str:
    """Swap `from pyro.infer import MCMC, NUTS[, ...]` for rejection imports."""
    pattern = re.compile(
        r'^from pyro\.infer import[^\n]*\b(?:MCMC|NUTS)\b[^\n]*\n',
        re.MULTILINE,
    )
    if not pattern.search(text):
        return text
    imports = REJECTION_IMPORTS_SOFT if mode == 'soft' else REJECTION_IMPORTS_HARD
    return pattern.sub(imports, text, count=1)


_DEF_RUN_INF = re.compile(
    r'^def run_inference\([^)]*\):\n'
    r'(?:(?:    [^\n]*|\s*)\n)+?'
    r'    return mcmc\s*\n',
    re.MULTILINE,
)


def _replace_run_inference(text: str, mode: str, num_samples: int) -> tuple[str, bool]:
    """Replace an existing `def run_inference(...): ... return mcmc` block."""
    tmpl = _SOFT_RUN_INFERENCE_TMPL if mode == 'soft' else _HARD_RUN_INFERENCE_TMPL
    body = tmpl.format(num_samples=num_samples)
    if _DEF_RUN_INF.search(text):
        return _DEF_RUN_INF.sub(body, text, count=1), True
    return text, False


_MAIN_BLOCK = re.compile(
    r'(?P<indent>[ \t]+)print\("Running NUTS inference on (?P<name>[^"]+?) model ?[.…]{1,3}"\)\n'
    r'(?P<mid>(?:[ \t]+[^\n]*\n)*?)'
    r'[ \t]+mcmc\.summary\(\)\n',
)


def _replace_main_block(text: str, mode: str, num_samples: int) -> tuple[str, bool]:
    """Replace the NUTS invocation block in `__main__` with a rejection call."""
    match = _MAIN_BLOCK.search(text)
    if not match:
        return text, False
    indent = match.group('indent')
    name = match.group('name')
    if mode == 'soft':
        intro = (f'importance-weighted rejection on {name} model '
                 f'({num_samples} accepted samples)')
    else:
        intro = (f'rejection sampling (hard beat/lost) on {name} model '
                 f'({num_samples} accepted samples)')
    replacement = (
        f'{indent}print("Running {intro} ...")\n'
        f'{indent}samples = run_inference(num_samples={num_samples})\n'
        f'{indent}print(f"  num accepted = '
        f'{{next(iter(samples.values())).shape[0]}}")\n'
    )
    return text[: match.start()] + replacement + text[match.end():], True


def _inject_run_inference(text: str, mode: str, num_samples: int) -> str:
    """Insert the rejection `run_inference` right before `if __name__`."""
    tmpl = _SOFT_RUN_INFERENCE_TMPL if mode == 'soft' else _HARD_RUN_INFERENCE_TMPL
    body = tmpl.format(num_samples=num_samples)
    marker = "\nif __name__ == '__main__':"
    idx = text.find(marker)
    if idx < 0:
        marker = '\nif __name__ == "__main__":'
        idx = text.find(marker)
    if idx < 0:
        return text + "\n\n" + body + "\n"
    return text[:idx] + "\n" + body + text[idx:]


def _swap_nuts_token(text: str) -> str:
    """Replace the token NUTS with REJ globally (for embedded output paths)."""
    return re.sub(r'\bNUTS\b', 'REJ', text)


def _wrap_json_dump(text: str) -> str:
    """Wrap `json.dump(result, ...)` in `{'seeds': [0], 'queries': result}`.

    Matches both `json.dump(result, f)` and any single-arg-handle form. Only
    rewrites the first occurrence that targets `result` directly.
    """
    return re.sub(
        r"json\.dump\(\s*result\s*,",
        "json.dump({'seeds': [0], 'queries': result},",
        text,
        count=1,
    )


def convert_text(text: str,
                 mode: str = 'soft',
                 num_samples: int = 1000,
                 wrap_json: bool = False) -> str:
    """Return the rejection-sampling version of a Pyro NUTS program.

    Parameters
    ----------
    text         : source text of a NUTS-style Pyro program (as written by
                   the program-generation pipeline).
    mode         : 'soft' (importance-weighted; same target as NUTS) or
                   'hard' (prior rejection on diff>0).
    num_samples  : number of accepted samples to draw in the rewritten
                   program. Embedded as the default in the generated
                   ``def run_inference(num_samples=N, ...)``.
    wrap_json    : if True, rewrite the program's ``json.dump(result, ...)``
                   call so the on-disk JSON is ``{'seeds': [0],
                   'queries': result}`` (matches the combined-posterior layout
                   that ``compare_to_humans.py`` reads).
    """
    if mode not in ('soft', 'hard'):
        raise ValueError(f"mode must be 'soft' or 'hard'; got {mode!r}")

    text = _replace_imports(text, mode)
    text, replaced_def = _replace_run_inference(text, mode, num_samples)
    text, _ = _replace_main_block(text, mode, num_samples)
    if not replaced_def:
        # Outlier programs inline MCMC+NUTS in __main__ without a dedicated
        # run_inference function. Inject one so the rewritten main block
        # can still call it.
        text = _inject_run_inference(text, mode, num_samples)
    text = _swap_nuts_token(text)
    if wrap_json:
        text = _wrap_json_dump(text)
    return text


# ── filename helper ────────────────────────────────────────────────────────-

_PG_GEMINI = re.compile(r'^(pg-gemini-)')
_PG_OTHER = re.compile(r'^(pg-[^-]+-)')


def rej_filename(name: str) -> str:
    """Insert ``REJ-`` after the first two hyphen-separated tokens of a
    program filename.

    e.g. ``pg-gemini-...`` -> ``pg-gemini-REJ-...``,
         ``pg-diving-pytorch.py`` -> ``pg-diving-REJ-pytorch.py``.
    """
    m = _PG_GEMINI.match(name) or _PG_OTHER.match(name)
    if m is None:
        raise ValueError(f"Cannot compute REJ filename for {name!r}")
    prefix = m.group(1)
    return prefix + 'REJ-' + name[len(prefix):]


def convert_file(src: Path, dst: Path,
                 mode: str = 'soft',
                 num_samples: int = 1000,
                 wrap_json: bool = False) -> None:
    new_text = convert_text(src.read_text(), mode=mode,
                            num_samples=num_samples, wrap_json=wrap_json)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(new_text)


def _main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('src', type=Path,
                    help='Source file or directory containing pg-*.py.')
    ap.add_argument('dst', type=Path,
                    help='Destination file or directory.')
    ap.add_argument('--mode', choices=('soft', 'hard'), default='soft')
    ap.add_argument('--num-samples', type=int, default=1000)
    ap.add_argument('--wrap-json', action='store_true',
                    help='Wrap json.dump(result, ...) so the file contents are '
                         "{'seeds': [0], 'queries': result}.")
    args = ap.parse_args()

    if args.src.is_file():
        dst = args.dst
        if dst.is_dir() or str(dst).endswith('/'):
            dst = dst / rej_filename(args.src.name)
        convert_file(args.src, dst, mode=args.mode,
                     num_samples=args.num_samples, wrap_json=args.wrap_json)
        print(f"wrote {dst}")
        return

    if not args.src.is_dir():
        raise SystemExit(f"src must be a file or directory: {args.src}")
    args.dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for src in sorted(args.src.glob('pg-*.py')):
        dst = args.dst / rej_filename(src.name)
        convert_file(src, dst, mode=args.mode,
                     num_samples=args.num_samples, wrap_json=args.wrap_json)
        n += 1
    print(f"wrote {n} rejection-sampling programs to {args.dst}")


if __name__ == '__main__':
    _main()
