"""Front/back segmented attention report from existing training logs.

Re-parses the per-position attention matrices already appended to training
logs (the "i=.. -> j=.." lines of the Attention Analysis section) and
re-aggregates them into FRONT/BACK segments. The split point is computed from
the log's per-position accuracy lines with the same rule as
analyze_attention.compute_front_split:

  skip epoch 0 (init noise); at the first epoch where a leading run of
  positions is < 0.1 while the position right after is >= 0.15, k = length
  of that leading run. Log-line value j corresponds to predicting
  x_{start_x+j}, i.e. query position start_x + j - 1 in the analysis input.

Torch-free: only parses log text, so it runs anywhere.

Usage:
    python src/report_front_back_attention.py <logs_dir> [<logs_dir> ...] [-o out.txt]
"""
import argparse
import glob
import os
import re
from collections import defaultdict

MIN_PRINT_VAL = 0.05   # overview table cells below this are omitted

EPOCH_RE = re.compile(r'Epoch\s+(\d+)\s+\|')
POS_RE = re.compile(r'from x(\d+):\s+(.+)')
LAYER_RE = re.compile(r'---\s*Layer\s+(\d+)\s*---')
HEAD_RE = re.compile(r'Head\s+(\d+):')
IROW_RE = re.compile(r'i=\s*(\d+)\s*->\s*(.+)')
JPAIR_RE = re.compile(r'j=(\d+):([\d.]+)')
ATTN_HEADER = 'Attention Analysis'


def parse_log(path):
    """Return (start_x, series, attn) where series = [(epoch, per_pos_accs)],
    attn[layer][head] = {i: {j: value}}."""
    start_x = None
    series = []
    cur_ep = None
    in_attn = False
    attn = defaultdict(lambda: defaultdict(dict))
    cur_layer = cur_head = None

    with open(path, encoding='utf-8') as f:
        for line in f:
            if ATTN_HEADER in line:
                in_attn = True
                continue
            if not in_attn:
                m = EPOCH_RE.search(line)
                if m:
                    cur_ep = int(m.group(1))
                    continue
                m = POS_RE.search(line)
                if m and cur_ep is not None:
                    if start_x is None:
                        start_x = int(m.group(1))
                    series.append((cur_ep, [float(x) for x in m.group(2).split()]))
            else:
                m = LAYER_RE.search(line)
                if m:
                    cur_layer = int(m.group(1))
                    continue
                m = HEAD_RE.search(line)
                if m:
                    cur_head = int(m.group(1))
                    continue
                m = IROW_RE.search(line)
                if m and cur_layer is not None and cur_head is not None:
                    i = int(m.group(1))
                    attn[cur_layer][cur_head][i] = {
                        int(j): float(v) for j, v in JPAIR_RE.findall(m.group(2))}
    return start_x, series, attn


def split_k(series):
    """Front-segment length k; 0 = no shortcut, None = never separated."""
    for ep, accs in series:
        if ep == 0:
            continue
        k = 0
        for a in accs:
            if a < 0.1:
                k += 1
            else:
                break
        if k < len(accs) and accs[k] >= 0.15:
            return k
    return None


def attention_signature(attn, pos, threshold=MIN_PRINT_VAL):
    """Significant attention distances at one query position (union over
    layers/heads): {i-j : attn[pos][j] >= threshold}."""
    sig = set()
    for layer in attn.values():
        for mat in layer.values():
            row = mat.get(pos, {})
            for j, v in row.items():
                if v >= threshold and pos - j >= 0:
                    sig.add(pos - j)
    return frozenset(sig)


def segment_by_attention(attn, start_pos, threshold=MIN_PRINT_VAL):
    """Segment consecutive query positions (from start_pos) by identical
    attention signatures. Returns [(seg_start, seg_end, signature), ...]
    with inclusive ends; positions with no matrix rows are skipped."""
    T = max((i for layer in attn.values() for head in layer.values() for i in head),
            default=-1) + 1
    segments = []
    cur_sig, cur_start = None, None
    for pos in range(start_pos, T):
        sig = attention_signature(attn, pos, threshold)
        if cur_sig is None:
            cur_sig, cur_start = sig, pos
        elif sig != cur_sig:
            segments.append((cur_start, pos - 1, cur_sig))
            cur_sig, cur_start = sig, pos
    if cur_sig is not None:
        segments.append((cur_start, T - 1, cur_sig))
    return segments


def focus_by_distance(attn_mat, query_positions):
    """Mean attention per distance over the given query positions."""
    out = {}
    if not attn_mat:
        return out
    T = max(attn_mat.keys()) + 1
    for d in range(T):
        vals = [row[i - d] for i in query_positions
                for row in [attn_mat.get(i)]
                if row is not None and i - d in row]
        if vals:
            out[d] = sum(vals) / len(vals)
    return out


def fmt_cell(fbd):
    parts = [f"d{d}:{v:.3f}" for d, v in sorted(fbd.items()) if v >= MIN_PRINT_VAL]
    return ' '.join(parts) if parts else '-'


# Symbol scale for the attention matrix dump (ALL full-width, 2 cells each,
# so the grid aligns in Notepad with a CJK font like SimSun/NSimSun):
#   <0.05 -> '　', 0.05~0.1 -> '・', 0.1~0.25 -> '○', 0.25~0.5 -> '×', 0.5~1 -> '※'
SYMBOL_BREAKS = (0.05, 0.1, 0.25, 0.5)
SYMBOLS = ('　', '·', '○', '×', '※')  # all full-width and GBK-encodable
FW_DIGITS = '０１２３４５６７８９'  # full-width digits for the column ruler


def attn_symbol(v):
    for i, thr in enumerate(SYMBOL_BREAKS):
        if v < thr:
            return SYMBOLS[i]
    return SYMBOLS[-1]


def dump_matrices(attn, emit, title):
    """Print the T x T attention matrix of every layer/head as a symbol grid.
    Row i, column j = attention from query position i to key position j.
    One row per line, no spaces between columns."""
    for layer in sorted(attn):
        for head in sorted(attn[layer]):
            mat = attn[layer][head]
            T = max(mat.keys()) + 1
            emit(f'\n{title} | Layer {layer} Head {head}  ({T}x{T})')
            emit('j\\i ' + ''.join(FW_DIGITS[j % 10] for j in range(T)))
            for i in range(T):
                row = mat.get(i, {})
                emit(f'{i:>3} ' + ''.join(attn_symbol(row.get(j, 0.0)) for j in range(T)).rstrip())


def main():
    ap = argparse.ArgumentParser(description='Front/back segmented attention report from logs.')
    ap.add_argument('logs_dirs', nargs='+', help='one or more logs directories')
    ap.add_argument('-o', '--out', default='front_back_attention_report.txt',
                    help='output report path')
    ap.add_argument('--matrices', action='store_true',
                    help='dump per-layer/head attention matrices as symbol grids '
                         '(space, ·, ○, ×, ※ for <0.05/0.1/0.25/0.5/1)')
    ap.add_argument('--match', default=None,
                    help='only process logs whose filename contains this substring')
    ap.add_argument('--seed', default=None,
                    help='only process the log with this seed number')
    args = ap.parse_args()

    lines = []
    def emit(s=''):
        print(s)
        lines.append(s)

    if args.matrices:
        for logs_dir in args.logs_dirs:
            for path in sorted(glob.glob(os.path.join(logs_dir, '*.log'))):
                name = os.path.splitext(os.path.basename(path))[0]
                if args.match and args.match not in name:
                    continue
                if args.seed and not name.endswith(f'_seed{args.seed}'):
                    continue
                start_x, series, attn = parse_log(path)
                k = split_k(series)
                emit('=' * 78)
                emit(f'{name}  (front k={k})')
                if not attn:
                    emit('  no attention data found, skipped')
                    continue
                dump_matrices(attn, emit, name)
        with open(args.out, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines) + '\n')
        print(f'\n[report saved to {args.out}]')
        return

    for logs_dir in args.logs_dirs:
        emit('=' * 78)
        emit(f'LOGS: {logs_dir}')
        emit('=' * 78)
        # group by config (name minus _seedN)
        per_cfg = defaultdict(list)
        for path in sorted(glob.glob(os.path.join(logs_dir, '*.log'))):
            name = os.path.splitext(os.path.basename(path))[0]
            cfg = name.rsplit('_seed', 1)[0]
            per_cfg[cfg].append(path)

        for cfg in sorted(per_cfg):
            paths = per_cfg[cfg]
            # per-seed split + segmented stats
            seed_stats = []
            ks = []
            start_x = None
            for path in paths:
                sx, series, attn = parse_log(path)
                k = split_k(series)
                if sx is None or not attn:
                    continue
                start_x = sx
                ks.append(k)
                seed_stats.append((k, attn))
            if not seed_stats:
                emit(f'\n## {cfg}: no attention data found, skipped')
                continue
            k_vals = {k for k, _ in seed_stats}
            k = ks[0]
            note = f'front k={k}' if k is not None else 'no separation'
            if len(k_vals) > 1:
                note += f' (WARNING: seeds disagree: {sorted(k_vals, key=str)})'
            emit(f'\n## {cfg}  [{note}, {len(seed_stats)} seeds]')
            emit(f'   (start_x={start_x}; front queries = input positions '
                 f'{start_x - 1}..{start_x - 2 + k if k else start_x - 2})'
                 if k else '')

            layers = sorted({l for _, attn in seed_stats for l in attn})
            heads = sorted({h for _, attn in seed_stats for l in attn.values() for h in l})

            if not k:
                segments = [('ALL', None)]
            else:
                segments = [('FRONT', 'front'), ('BACK', 'back')]

            for seg_name, seg in segments:
                emit(f'\n  [{seg_name}]  rows=head, cols=layer, cells: d<distance>:<mean attn> >= {MIN_PRINT_VAL}')
                emit('  ' + 'head \\ layer'.ljust(14) + ''.join(f'layer {l}'.rjust(30) for l in layers))
                for h in heads:
                    cells = []
                    for l in layers:
                        # average focus_by_distance over seeds
                        acc = defaultdict(list)
                        for k_s, attn in seed_stats:
                            T = max((i for hd in attn.get(l, {}).values() for i in hd), default=0) + 1
                            if seg == 'front':
                                qpos = list(range(start_x - 1, start_x - 1 + k_s))
                            elif seg == 'back':
                                qpos = list(range(start_x - 1 + k_s, T - 1))
                            else:
                                qpos = list(range(0, T - 1))
                            if l in attn and h in attn[l]:
                                fbd = focus_by_distance(attn[l][h], qpos)
                                for d, v in fbd.items():
                                    acc[d].append(v)
                        fbd_mean = {d: sum(vs) / len(vs) for d, vs in acc.items() if vs}
                        cells.append(fmt_cell(fbd_mean))
                    emit('  ' + f'head {h}'.ljust(14) + ''.join(c.rjust(30) for c in cells))
        emit()

    with open(args.out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'\n[report saved to {args.out}]')


if __name__ == '__main__':
    main()
