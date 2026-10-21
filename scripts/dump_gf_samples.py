#!/usr/bin/env python3
"""Dump single-rule recurrence windows in polynomial notation.

Internal storage remains ints (bit i = coeff of x^i). This script is the
human-facing view Kay asked for: always print GF(2^n) elements as
polynomials like ``x^2 + 1``.

Examples:
  PYTHONPATH=src python scripts/dump_gf_samples.py \\
      --coeffs 70,124 --length 16 --inits 1,1 3,5
  PYTHONPATH=src python scripts/dump_gf_samples.py \\
      --coeffs 70,124 --also-fp --p-prime 127
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from finite_field import F128, GF2n, generate_recurrence_sequence, poly_str
from rules import LinearRecurrenceRule, format_recurrence_name


def parse_coeffs(s: str):
    return tuple(int(x.strip()) for x in s.split(',') if x.strip())


def parse_init(s: str):
    parts = [int(x.strip()) for x in s.split(',') if x.strip()]
    return tuple(parts)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--coeffs', required=True,
                    help='Canonical coeffs c1,c2,... (ints; shown as polys)')
    ap.add_argument('--length', type=int, default=16)
    ap.add_argument('--inits', nargs='+', default=['1,1', '3,5', '10,20'],
                    help='Initial states as comma-separated ints')
    ap.add_argument('--field', default='f128',
                    help="'f128' or path unused; currently only F128 baked in")
    ap.add_argument('--also-fp', action='store_true',
                    help='Also print Z/pZ windows with the same numeric coeffs')
    ap.add_argument('--p-prime', type=int, default=127)
    ap.add_argument('--json-out', default=None,
                    help='Optional JSON dump (polys + ints)')
    ap.add_argument('--md-out', default=None,
                    help='Optional markdown path')
    args = ap.parse_args()

    coeffs = parse_coeffs(args.coeffs)
    field = F128 if args.field == 'f128' else GF2n(n=7)
    inits = [parse_init(s) for s in args.inits]
    for init in inits:
        if len(init) != len(coeffs):
            raise SystemExit(
                f"init {init} length {len(init)} != order {len(coeffs)}")

    lines = []
    lines.append(f"# Dump GF(2^{field.n}) samples (polynomial view)")
    lines.append("")
    lines.append(f"Irreducible: `{poly_str(field.irreducible)}`")
    lines.append(f"RULES ints: `{list(coeffs)}`")
    lines.append("RULES polys: " + ", ".join(f"`{poly_str(c)}`" for c in coeffs))
    lines.append("")
    lines.append(format_recurrence_name(coeffs, field.order, field=field))
    lines.append("")
    lines.append("## GF windows")
    records = []
    for init in inits:
        seq = generate_recurrence_sequence(field, coeffs, init=init,
                                           length=args.length)
        init_poly = "(" + ", ".join(poly_str(x) for x in init) + ")"
        seq_poly = "[" + ", ".join(poly_str(x) for x in seq) + "]"
        lines.append(f"init {init_poly}:")
        lines.append(f"  polys: {seq_poly}")
        lines.append(f"  ints:  {list(seq)}")
        lines.append("")
        records.append({
            'init_int': list(init),
            'init_poly': [poly_str(x) for x in init],
            'seq_int': list(seq),
            'seq_poly': [poly_str(x) for x in seq],
            'field': f'GF(2^{field.n})',
        })

    if args.also_fp:
        lines.append(f"## F_{{{args.p_prime}}} windows (residues, not polynomials)")
        rule = LinearRecurrenceRule(coeffs=coeffs, p=args.p_prime)
        fn = rule.next_fn()
        for init in inits:
            if any(x >= args.p_prime for x in init):
                lines.append(f"init {init}: skipped (element >= {args.p_prime})")
                continue
            seq = list(init)
            while len(seq) < args.length:
                seq.append(fn(seq, args.p_prime))
            lines.append(f"init {init}: {seq}")
            records.append({
                'init_int': list(init),
                'seq_int': seq,
                'field': f'F_{args.p_prime}',
            })
        lines.append("")

    text = "\n".join(lines)
    print(text)

    if args.md_out:
        Path(args.md_out).write_text(text)
        print(f"WROTE {args.md_out}", file=sys.stderr)
    if args.json_out:
        payload = {
            'irreducible_poly': poly_str(field.irreducible),
            'coeffs_int': list(coeffs),
            'coeffs_poly': [poly_str(c) for c in coeffs],
            'recurrence': format_recurrence_name(coeffs, field.order, field=field),
            'samples': records,
        }
        Path(args.json_out).write_text(json.dumps(payload, indent=2))
        print(f"WROTE {args.json_out}", file=sys.stderr)


if __name__ == '__main__':
    main()
