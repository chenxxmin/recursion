#!/usr/bin/env python3
"""Chain runner helper: regenerate a resume-aware config before each round.

For each experiment in the config:
  - skip (drop) if its log in this batch's log dir already contains
    'Best test accuracy' (a completed run);
  - else if a resume checkpoint exists in this batch's model dir, set
    RESUME_FROM to it (latest round's state);
  - otherwise keep as-is (fresh run, or first resume from a pre-existing
    RESUME_FROM path carried in the config).

Usage: prepare_chain_round.py <config.json>
Prints the number of remaining experiments; exits 3 when none remain.
"""
import json
import os
import sys

DATA_BASE = '/data/cxm/recursion'
MODEL_BASE = '/data/cxm/models'


def main():
    path = sys.argv[1]
    cfg = json.load(open(path))
    stem = os.path.splitext(os.path.basename(path))[0]
    log_dir = os.path.join(DATA_BASE, stem, 'logs')
    model_dir = os.path.join(MODEL_BASE, stem)

    remaining = []
    for e in cfg['experiments']:
        name = e['name']
        log_path = os.path.join(log_dir, name + '.log')
        if os.path.exists(log_path):
            txt = open(log_path, errors='replace').read()
            if 'Best test accuracy' in txt:
                continue  # completed in a previous round
        ckpt = os.path.join(model_dir, name + '_resume.pth')
        if os.path.exists(ckpt):
            e['config']['RESUME_FROM'] = ckpt
        remaining.append(e)

    cfg['experiments'] = remaining
    json.dump(cfg, open(path, 'w'), indent=2, ensure_ascii=False)
    print(len(remaining))
    if not remaining:
        sys.exit(3)


if __name__ == '__main__':
    main()
