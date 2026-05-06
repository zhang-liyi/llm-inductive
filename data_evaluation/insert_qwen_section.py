"""Insert a Qwen section into each subtable of calibration_tables.tex.

For each of the 3 subtables (Accuracy / NLL / ECE):
  - Rename "Method" -> "Method: Llama 3" in the top header.
  - Before \\bottomrule, append:
        \\midrule\\midrule
        Method: Qwen 2 & ...same column headers... \\\\
        \\midrule
        Qwen 2 (Base) & ...values... \\\\
        Qwen (Dist)   & ...values (with SE)... \\\\

Values pulled from calibration_tables_all.tex (Qwen2 (Base) single-shot,
Qwen2 (Dist+Term) → renamed "Qwen (Dist)" 3-seed mean ± SE).

Bolding within the Qwen mini-block is handled by a subsequent run of
rebold_calibration_tables.py, which processes each \\midrule-separated
block independently.
"""
import re

TEX = './data_evaluation/results/calibration_tables.tex'

# Per subtable: list of (Qwen2 (Base) cells, Qwen (Dist) cells with SE).
# Cell content for plain row is just the inner $X$ or $X\%$ literal (no wraps).
# For the multi-line ±SE row, the cell is rendered manually below.

ACC_COLS = (
    'OE (MAE) $\\downarrow$', 'BT-nonG', 'BT-guided', 'MMLU', 'TruthfulQA',
    'HellaSwag', 'ARC-C', 'Winogrande')
NLL_COLS = (
    'OE', 'BT-nonG', 'BT-guided', 'MMLU', 'TruthfulQA', 'HellaSwag', 'ARC-C',
    'Winogrande')
ECE_COLS = (
    'BT-nonG', 'BT-guided', 'MMLU', 'TruthfulQA', 'HellaSwag', 'ARC-C',
    'Winogrande')

# Qwen 2 (Base) — single-shot. Values from calibration_tables_all.tex.
QWEN_BASE_ACC = ['$14.2$', '$39.6\\%$', '$37.6\\%$', '$65.1\\%$', '$41.1\\%$',
                 '$80.1\\%$', '$83.6\\%$', '$61.3\\%$']
QWEN_BASE_NLL = ['$6.906$', '$2.561$', '$2.207$', '$1.960$', '$4.807$',
                 '$0.842$', '$0.896$', '$2.178$']
QWEN_BASE_ECE = ['$0.464$', '$0.445$', '$0.273$', '$0.475$', '$0.134$',
                 '$0.135$', '$0.337$']

# Qwen (Dist) — renamed from Qwen2 (Dist+Term). 3-seed mean ± SE.
# Format: (mu_str, se_str) per cell.
QWEN_DIST_ACC = [
    ('$13.4$',     '$\\pm 0.1$'),
    ('$38.9\\%$',  '$\\pm 0.3\\%$'),
    ('$38.4\\%$',  '$\\pm 0.5\\%$'),
    ('$65.5\\%$',  '$\\pm 0.1\\%$'),
    ('$40.4\\%$',  '$\\pm 0.1\\%$'),
    ('$79.4\\%$',  '$\\pm 0.1\\%$'),
    ('$84.1\\%$',  '$\\pm 0.1\\%$'),
    ('$63.8\\%$',  '$\\pm 0.4\\%$'),
]
QWEN_DIST_NLL = [
    ('$3.863$', '$\\pm 0.003$'),
    ('$1.395$', '$\\pm 0.011$'),
    ('$1.284$', '$\\pm 0.007$'),
    ('$1.119$', '$\\pm 0.008$'),
    ('$3.045$', '$\\pm 0.027$'),
    ('$0.602$', '$\\pm 0.003$'),
    ('$0.509$', '$\\pm 0.005$'),
    ('$1.232$', '$\\pm 0.006$'),
]
QWEN_DIST_ECE = [
    ('$0.278$', '$\\pm 0.005$'),
    ('$0.228$', '$\\pm 0.005$'),
    ('$0.180$', '$\\pm 0.001$'),
    ('$0.381$', '$\\pm 0.002$'),
    ('$0.060$', '$\\pm 0.001$'),
    ('$0.092$', '$\\pm 0.001$'),
    ('$0.246$', '$\\pm 0.003$'),
]


def fmt_base_row(label, cells):
    return f'{label:<15s} & ' + ' & '.join(cells) + r' \\'


def fmt_dist_row(label, mu_se_pairs):
    """Multi-line ±SE format mirroring the existing Pyro (Dist) rows."""
    lines = [f'{label:<15s}']
    for i, (mu, se) in enumerate(mu_se_pairs):
        sep = '&' if i == 0 else '                &'
        lines[-1] += f' {sep} {mu}'
        lines.append(f'                  \\scriptsize{{{se}}}')
    out = '\n'.join(lines).rstrip()
    return out + r' \\'


def build_qwen_block(cols, base_cells, dist_pairs):
    header = 'Method: Qwen 2 & ' + ' & '.join(cols) + r' \\'
    base_row = fmt_base_row('Base', base_cells)
    dist_row = fmt_dist_row('Pyro (Dist)', dist_pairs)
    return (
        '\\midrule\\midrule\n'
        f'{header}\n'
        '\\midrule\n'
        f'{base_row}\n'
        f'{dist_row}\n'
    )


def main():
    with open(TEX) as f:
        text = f.read()

    # 1. Rename "Method" -> "Method: Llama 3" in the header rows of each
    # tabular. The only "Method &" occurrences are in those headers, so
    # global replace is safe.
    text = text.replace('Method &', 'Method: Llama 3 &')

    # 2. Build the 3 Qwen sections, indexed by their target subtable.
    qwen_sections = [
        build_qwen_block(ACC_COLS, QWEN_BASE_ACC, QWEN_DIST_ACC),
        build_qwen_block(NLL_COLS, QWEN_BASE_NLL, QWEN_DIST_NLL),
        build_qwen_block(ECE_COLS, QWEN_BASE_ECE, QWEN_DIST_ECE),
    ]

    # 3. Insert before each \bottomrule in document order. Patch in reverse
    # so prior offsets stay valid.
    bot_iter = list(re.finditer(r'\\bottomrule', text))
    if len(bot_iter) != 3:
        raise SystemExit(f'expected 3 \\bottomrule, found {len(bot_iter)}')
    for m, qwen in zip(reversed(bot_iter), reversed(qwen_sections)):
        text = text[:m.start()] + qwen + text[m.start():]

    with open(TEX, 'w') as f:
        f.write(text)
    print(f'Inserted 3 Qwen sections into {TEX}')


if __name__ == '__main__':
    main()
