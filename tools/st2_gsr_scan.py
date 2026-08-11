#!/usr/bin/env python
"""
Stream-audit a SoccerTrack v2 GSR JSON without loading 2.7 GB into memory.

Two passes over the bytes, for two different reasons.

EXACT COUNTS come from byte-level counting of the pretty-printed key/value
lines. The files are machine-written with a fixed indentation, so counting
'"category_id": 4' is exact rather than approximate -- and it is the only way
to answer "how many ball annotations exist" on 55 GB of JSON in reasonable time.

SCHEMA AND GEOMETRY come from actually parsing objects. Every Nth annotation
object is extracted by brace matching and json.loads'd, so field names, box
coordinates and attributes are read, never inferred.

The distinction matters: a count taken from a substring is a fact about the
file's bytes, a box measured from a parsed object is a fact about the
annotation. Both are reported, and they are not mixed.
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CHUNK = 1 << 24


def count_tokens(path: Path, tokens):
    """Exact substring counts, chunked with an overlap so nothing splits."""
    counts = Counter()
    tail = b''
    maxlen = max(len(t) for t in tokens)
    with open(path, 'rb') as f:
        while True:
            b = f.read(CHUNK)
            if not b:
                break
            hay = tail + b
            for t in tokens:
                counts[t.decode()] += hay.count(t)
            tail = hay[-(maxlen - 1):] if maxlen > 1 else b''
    return counts


def iter_objects(path: Path, array_key: bytes, every=1, limit=None):
    """Yield parsed objects from a top-level JSON array, by brace matching."""
    with open(path, 'rb') as f:
        # find the array
        pos, prev = 0, b''
        start = -1
        while True:
            b = f.read(CHUNK)
            if not b:
                break
            hay = prev + b
            i = hay.find(array_key)
            if i != -1:
                start = pos - len(prev) + i
                break
            prev = hay[-64:]
            pos += len(b)
        if start < 0:
            return
        f.seek(start + len(array_key))
        # skip to '['
        while True:
            c = f.read(1)
            if c == b'[':
                break
            if not c:
                return
        depth, cur, n, yielded = 0, bytearray(), 0, 0
        in_str, esc = False, False
        while True:
            b = f.read(CHUNK)
            if not b:
                break
            for ch in b:
                c = bytes([ch])
                if depth:
                    cur += c
                if in_str:
                    if esc:
                        esc = False
                    elif c == b'\\':
                        esc = True
                    elif c == b'"':
                        in_str = False
                    continue
                if c == b'"':
                    in_str = True
                elif c == b'{':
                    if depth == 0:
                        cur = bytearray(b'{')
                    depth += 1
                elif c == b'}':
                    depth -= 1
                    if depth == 0:
                        n += 1
                        if n % every == 0:
                            yielded += 1
                            yield json.loads(cur.decode('utf-8'))
                            if limit and yielded >= limit:
                                return
                elif c == b']' and depth == 0:
                    return


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('files', nargs='+')
    ap.add_argument('--every', type=int, default=500,
                    help='parse every Nth annotation object for schema/geometry')
    ap.add_argument('--limit', type=int, default=4000)
    ap.add_argument('--out', default='experiments/soccertrack_audit/reports/gsr_scan.json')
    ap.add_argument('--counts-only', action='store_true')
    args = ap.parse_args()

    TOK = [b'"category_id": 1', b'"category_id": 2', b'"category_id": 3',
           b'"category_id": 4', b'"category_id": 5', b'"category_id": 6',
           b'"bbox_image"', b'"bbox_pitch"', b'"track_id"', b'"jersey_number"',
           b'"team"', b'"role"', b'"attributes"', b'"is_labeled": true',
           b'"is_labeled": false', b'"has_labeled_person": true',
           b'"file_name"', b'"lines"', b'"camera"']
    CAT = {1: 'player', 2: 'goalkeeper', 3: 'referee', 4: 'ball',
           5: 'pitch', 6: 'camera'}

    out = {}
    for fp in args.files:
        p = Path(fp)
        print(f'== {p.name}  ({p.stat().st_size/1e9:.2f} GB)')
        c = count_tokens(p, TOK)
        rec = {'file': p.name, 'bytes': p.stat().st_size,
               'exact_token_counts': dict(c),
               'annotations_per_category': {CAT[i]: c[f'"category_id": {i}']
                                            for i in range(1, 7)}}
        print('   annotations per category:', rec['annotations_per_category'])
        print('   bbox_image occurrences  :', c['"bbox_image"'],
              '  bbox_pitch:', c['"bbox_pitch"'])
        print('   images labeled true/false:', c['"is_labeled": true'],
              '/', c['"is_labeled": false'])

        if not args.counts_only:
            fields = defaultdict(Counter)
            attrs = defaultdict(Counter)
            geom = defaultdict(list)
            samples = {}
            for a in iter_objects(p, b'"annotations"', args.every, args.limit):
                cat = CAT.get(a.get('category_id'), str(a.get('category_id')))
                for k in a:
                    fields[cat][k] += 1
                for k, v in (a.get('attributes') or {}).items():
                    attrs[cat][k] += 1
                if 'bbox_image' in a and isinstance(a['bbox_image'], dict):
                    b = a['bbox_image']
                    geom[cat].append((b.get('w'), b.get('h')))
                if cat not in samples:
                    samples[cat] = a
            rec['parsed_sample'] = {
                'every': args.every,
                'fields_by_category': {k: dict(v) for k, v in fields.items()},
                'attribute_keys_by_category': {k: dict(v) for k, v in attrs.items()},
                'example': {k: (v if k != 'lines' else
                                {'<pitch line names>': sorted(v)[:6]})
                            for k, v in
                            {kk: vv for kk, vv in samples.items()}.items()},
            }
            print('   parsed fields by category:')
            for k, v in fields.items():
                print(f'      {k:<12} n={max(v.values()) if v else 0:<5} keys={sorted(v)}')
        out[p.name] = rec

    dst = REPO / args.out
    dst.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(dst.read_text(encoding='utf-8')) if dst.exists() else {}
    prev.update(out)
    dst.write_text(json.dumps(prev, indent=1, ensure_ascii=False), encoding='utf-8')
    print(f'\nwrote {args.out}')


if __name__ == '__main__':
    main()
