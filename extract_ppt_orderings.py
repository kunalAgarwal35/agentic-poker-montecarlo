"""Extract ProPokerTools' exact percentile orderings from ppt_base.jar (propokertools/
core/orderings/*.txt) and write our class-order JSON artifacts. Provenance: the orderings
are data derived from the project's reference oracle (the user's licensed ProPokerTools
jar); the committed artifacts are the parsed orderings in our notation.

Usage: python extract_ppt_orderings.py --game holdem|plo4|plo5|all
"""
import argparse
import json
import os
import zipfile

from pql.ranges.ppt_order import parse_ppt_ordering_line, class_key_to_string

_ROOT = os.path.dirname(os.path.abspath(__file__))
_JAR = os.path.join(_ROOT, 'java_files', 'ppt_base.jar')

# game -> (resource path, output json, expected class count, holdem?)
SPEC = {
    'holdem': ('propokertools/core/orderings/heordering.txt', 'holdem_class_order.json', 169, True),
    'plo4':   ('propokertools/core/orderings/ohordering.txt', 'plo4_class_order.json', 16432, False),
    'plo5':   ('propokertools/core/orderings/oh5ordering.txt', 'plo5_class_order.json', 134459, False),
}


def extract(game: str):
    res, out, n, holdem = SPEC[game]
    with zipfile.ZipFile(_JAR) as z:
        text = z.read(res).decode()
    order, seen = [], set()
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        key = parse_ppt_ordering_line(line)
        assert len(set(key)) == len(key), f"{game}: duplicate card in class {line!r} -> {key}"
        seen.add(key)
        order.append(class_key_to_string(key, holdem=holdem))
    assert len(order) == n, f"{game}: parsed {len(order)} classes, expected {n}"
    assert len(seen) == n, f"{game}: {len(seen)} unique classes, expected {n}"
    assert len(set(order)) == n, f"{game}: rendered strings not unique"
    with open(os.path.join(_ROOT, out), 'w') as f:
        json.dump(order, f)
    print(f"wrote {out}: {len(order)} classes, top: {order[:6]}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--game', required=True, choices=list(SPEC) + ['all'])
    args = ap.parse_args()
    games = list(SPEC) if args.game == 'all' else [args.game]
    for g in games:
        extract(g)
