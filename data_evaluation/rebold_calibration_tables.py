"""Re-bold every column in calibration_tables_all.tex.

For each tabular block (separated by \\midrule), find the best value per
numeric column and wrap it with \\underline{\\bm{...}} (bold + underline).
Strips any existing \\bm{...} or \\underline{\\bm{...}} wraps first, so the
script is idempotent.

Direction:
  Subtable 1 (Accuracy): col 1 = OE-MAE  → min;  cols 2..8 = % accuracy → max
  Subtable 2 (NLL):      all 8 cols      → min
  Subtable 3 (ECE):      all 7 cols      → min
"""
import re

TEX = './data_evaluation/results/calibration_tables_all.tex'


def parse_value(cell):
    """Numeric value of the leading $...$ in cell, else None.
    Skips $\\pm ...$ (SE) — only matches digit-leading values."""
    m = re.search(r'\$([0-9]+(?:\.[0-9]+)?)(?:\\%)?\$', cell)
    return float(m.group(1)) if m else None


def strip_wrap(cell):
    """Remove \\underline{\\bm{$X$}} or \\bm{$X$} wraps around the leading value."""
    cell = re.sub(r'\\underline\{\\bm\{(\$[0-9.]+(?:\\%)?\$)\}\}', r'\1', cell, count=1)
    cell = re.sub(r'\\bm\{(\$[0-9.]+(?:\\%)?\$)\}', r'\1', cell, count=1)
    return cell


def add_wrap(cell):
    """Wrap the leading $X$ (or $X\\%$) with \\underline{\\bm{...}}."""
    return re.sub(r'(\$[0-9.]+(?:\\%)?\$)', r'\\underline{\\bm{\1}}', cell, count=1)


def process_block(block, dirs):
    rows = re.split(r'(\\\\\s*\n)', block)
    n_cols = len(dirs)
    parsed = [None] * len(rows)
    for i in range(0, len(rows), 2):
        content = rows[i]
        if '&' not in content:
            continue
        cells = content.split('&')
        cells = [strip_wrap(c) for c in cells]
        vals = [parse_value(c) for c in cells]
        parsed[i] = (cells, vals)

    winners = [None] * (n_cols + 1)
    best = [None] * (n_cols + 1)
    for i, p in enumerate(parsed):
        if p is None:
            continue
        cells, vals = p
        for j in range(1, min(n_cols + 1, len(vals))):
            v = vals[j]
            if v is None:
                continue
            cur = best[j]
            d = dirs[j - 1]
            if cur is None or (d == 'min' and v < cur) or (d == 'max' and v > cur):
                best[j] = v
                winners[j] = i

    for i, p in enumerate(parsed):
        if p is None:
            continue
        cells, _ = p
        for j in range(1, min(n_cols + 1, len(cells))):
            if winners[j] == i:
                cells[j] = add_wrap(cells[j])
        rows[i] = '&'.join(cells)

    return ''.join(rows)


def process_tabular(tab, dirs):
    midrules = list(re.finditer(r'\\midrule', tab))
    bot = re.search(r'\\bottomrule', tab)
    if not midrules or not bot:
        return tab
    ranges = []
    for i, mr in enumerate(midrules):
        start = mr.end()
        end = midrules[i + 1].start() if i + 1 < len(midrules) else bot.start()
        ranges.append((start, end))
    out = tab
    for start, end in reversed(ranges):
        block = out[start:end]
        out = out[:start] + process_block(block, dirs) + out[end:]
    return out


def main():
    with open(TEX) as f:
        text = f.read()

    sub_dirs = [
        ['min'] + ['max'] * 7,  # Accuracy: OE↓ + 7 % cols
        ['min'] * 8,            # NLL
        ['min'] * 7,            # ECE
    ]

    tabulars = list(re.finditer(r'\\begin\{tabular\}.*?\\end\{tabular\}', text, re.DOTALL))
    if len(tabulars) != 3:
        print(f'WARN: expected 3 tabular blocks, found {len(tabulars)}')
        return

    for tab, dirs in reversed(list(zip(tabulars, sub_dirs))):
        new_tab = process_tabular(text[tab.start():tab.end()], dirs)
        text = text[:tab.start()] + new_tab + text[tab.end():]

    with open(TEX, 'w') as f:
        f.write(text)
    print(f'Re-bolded + underlined column winners in {TEX}')


if __name__ == '__main__':
    main()
