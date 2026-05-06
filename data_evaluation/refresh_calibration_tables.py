"""Refresh calibration_tables.tex:
  1. Drop the Fusion (Dist) row from each subtable.
  2. Replace the legacy single-shot Forward (Mean) rows with the new
     3-seed mean ± SE values.
  3. Round all NLL cells (values + SEs) from 3 decimals to 2.
  4. Re-run the rebold pass so per-block winners reflect the new state.
"""
import re
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rebold_calibration_tables as rebold

TEX = './data_evaluation/results/calibration_tables.tex'

# Forward (Mean) 3-seed values from the
# forward_sampling_mean_s{1,2,3}_bracket aggregator.
FWD_ACC = [
    ('$27.7$',     '$\\pm 0.0$'),
    ('$38.0\\%$',  '$\\pm 0.5\\%$'),
    ('$38.3\\%$',  '$\\pm 0.6\\%$'),
    ('$60.2\\%$',  '$\\pm 0.2\\%$'),
    ('$42.5\\%$',  '$\\pm 0.7\\%$'),
    ('$64.4\\%$',  '$\\pm 0.6\\%$'),
    ('$77.3\\%$',  '$\\pm 0.4\\%$'),
    ('$55.4\\%$',  '$\\pm 0.3\\%$'),
]
# NLL values written 3-decimal here; the global NLL pass rounds them to 2.
FWD_NLL = [
    ('$4.788$', '$\\pm 0.007$'),
    ('$1.111$', '$\\pm 0.009$'),
    ('$1.084$', '$\\pm 0.004$'),
    ('$0.951$', '$\\pm 0.002$'),
    ('$1.518$', '$\\pm 0.029$'),
    ('$1.054$', '$\\pm 0.012$'),
    ('$0.672$', '$\\pm 0.012$'),
    ('$0.892$', '$\\pm 0.018$'),
]
FWD_ECE = [
    ('$0.091$', '$\\pm 0.015$'),
    ('$0.040$', '$\\pm 0.009$'),
    ('$0.043$', '$\\pm 0.001$'),
    ('$0.105$', '$\\pm 0.003$'),
    ('$0.222$', '$\\pm 0.012$'),
    ('$0.126$', '$\\pm 0.006$'),
    ('$0.072$', '$\\pm 0.014$'),
]


def fmt_dist_row(label, mu_se_pairs):
    """Multi-line ±SE row (matches the existing Pyro (Dist) format)."""
    lines = [f'{label:<15s}']
    for i, (mu, se) in enumerate(mu_se_pairs):
        sep = '&' if i == 0 else '                &'
        lines[-1] += f' {sep} {mu}'
        lines.append(f'                  \\scriptsize{{{se}}}')
    return '\n'.join(lines).rstrip() + r' \\'


def main():
    with open(TEX) as f:
        text = f.read()

    # 1) Drop Fusion (Dist) rows (single-line, 1 per subtable).
    text, n_drop = re.subn(r'^Fusion \(Dist\)[^\n]*?\\\\\n', '', text, flags=re.MULTILINE)
    print(f'Dropped {n_drop} Fusion (Dist) rows')

    # 2) Replace each Forward (Mean) row with multi-line ±SE.
    new_rows = [
        fmt_dist_row('Forward (Mean)', FWD_ACC) + '\n',
        fmt_dist_row('Forward (Mean)', FWD_NLL) + '\n',
        fmt_dist_row('Forward (Mean)', FWD_ECE) + '\n',
    ]
    fwd_iter = list(re.finditer(r'^Forward \(Mean\)[^\n]*?\\\\\n', text, re.MULTILINE))
    if len(fwd_iter) != 3:
        raise SystemExit(f'expected 3 Forward (Mean) rows; found {len(fwd_iter)}')
    for m, row in zip(reversed(fwd_iter), reversed(new_rows)):
        text = text[:m.start()] + row + text[m.end():]
    print(f'Replaced 3 Forward (Mean) rows with 3-seed ±SE form')

    # 3) Round 3-decimal cells inside the NLL subtable to 2 decimals.
    nll_start = text.index('\\subcaption{NLL')
    ece_start = text.index('\\subcaption{Expected')
    nll = text[nll_start:ece_start]
    nll = re.sub(r'\$(-?\d+\.\d{3})\$',
                 lambda m: f'${float(m.group(1)):.2f}$', nll)
    nll = re.sub(r'\$\\pm (\d+\.\d{3})\$',
                 lambda m: f'$\\pm {float(m.group(1)):.2f}$', nll)
    text = text[:nll_start] + nll + text[ece_start:]
    print('Rounded NLL cells to 2 decimals')

    with open(TEX, 'w') as f:
        f.write(text)

    # 4) Rebold per-block winners.
    rebold.TEX = TEX
    rebold.main()


if __name__ == '__main__':
    main()
