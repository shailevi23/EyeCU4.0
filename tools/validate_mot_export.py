#!/usr/bin/env python
"""
Round-trip the MOT GT export against the canonical JSON. Read-only.

The MOT files are derived, so the only interesting question is whether anything
was lost or invented on the way. A benchmark that silently drops a box, filters
a class, or renumbers an identity produces metrics that look completely normal
and mean nothing.

Checked per sequence:

    frames          exactly 300, 1-based, none missing
    count           one MOT row per canonical box, no more
    identities      the same set, sequence-local, unchanged
    mapping         every (frame, id) pair present exactly once, both ways
    geometry        x,y,w,h reconstructs the canonical [x1,y1,x2,y2] within
                    the two decimals the format serialises
    columns         conf == 1, class == 1, visibility == 1 on every row
    no filtering    no role was dropped; all three target roles evaluate as one
                    tracking class, which is the point
    exclusion       the rejected Austin window appears nowhere

`occluded` deliberately does NOT appear in MOT. It stays a boolean in the
canonical JSON. Turning it into a visibility fraction would fabricate a
measurement nobody made; TrackEval's MotChallenge2DBox loader never reads that
column anyway.
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

EXCLUDED = 'austin_fc_vs__club_tijuana_284'
TOL = 0.011          # the export writes two decimals


def check(rows, name, ok, detail=''):
    rows.append({'check': name, 'ok': bool(ok), 'detail': detail})
    return ok


def audit(root: Path, mot: Path, s: dict, split: str):
    tag = s['sequence']
    rows = []
    ann = json.loads((root / s['annotation_file_expected']).read_text(encoding='utf-8'))
    canon = ann['boxes']

    gt = mot / split / tag / 'gt' / 'gt.txt'
    check(rows, 'gt.txt exists', gt.exists(), str(gt))
    if not gt.exists():
        return rows, {}
    lines = [l for l in gt.read_text(encoding='utf-8').splitlines() if l.strip()]
    parsed = []
    for l in lines:
        p = l.split(',')
        parsed.append({'frame': int(p[0]), 'id': int(p[1]),
                       'x': float(p[2]), 'y': float(p[3]),
                       'w': float(p[4]), 'h': float(p[5]),
                       'conf': p[6], 'cls': p[7], 'vis': p[8]})

    check(rows, 'one MOT row per canonical box',
          len(parsed) == len(canon), f'{len(parsed)} vs {len(canon)}')
    frames = sorted({r['frame'] for r in parsed})
    check(rows, 'exactly 300 frames, 1..300',
          frames == list(range(1, s['frame_count'] + 1)),
          f'{len(frames)} frames, {frames[:1]}..{frames[-1:]}')
    check(rows, 'identities unchanged',
          {r['id'] for r in parsed} == {b['id'] for b in canon},
          str(sorted({r['id'] for r in parsed} ^ {b['id'] for b in canon})))
    check(rows, 'identities are sequence-local positive ints',
          all(r['id'] > 0 for r in parsed))

    key_mot = Counter((r['frame'], r['id']) for r in parsed)
    key_can = Counter((b['frame'], b['id']) for b in canon)
    check(rows, 'exact (frame, id) mapping both ways', key_mot == key_can,
          str(list((key_mot - key_can).items())[:3]
              + list((key_can - key_mot).items())[:3]))
    check(rows, 'no dropped boxes', not (key_can - key_mot),
          str(list((key_can - key_mot).items())[:3]))
    check(rows, 'no extra boxes', not (key_mot - key_can),
          str(list((key_mot - key_can).items())[:3]))

    by_key = {(b['frame'], b['id']): b['bbox'] for b in canon}
    worst, bad = 0.0, []
    for r in parsed:
        x1, y1, x2, y2 = by_key[(r['frame'], r['id'])]
        d = max(abs(r['x'] - x1), abs(r['y'] - y1),
                abs(r['w'] - (x2 - x1)), abs(r['h'] - (y2 - y1)))
        worst = max(worst, d)
        if d > TOL:
            bad.append((r['frame'], r['id'], round(d, 4)))
    check(rows, f'bbox round-trips within {TOL} px', not bad,
          f'worst {worst:.4f} px' if not bad else str(bad[:3]))

    check(rows, 'conf == 1 on every row', all(r['conf'] == '1' for r in parsed))
    check(rows, 'class == 1 on every row', all(r['cls'] == '1' for r in parsed))
    check(rows, 'visibility == 1 on every row', all(r['vis'] == '1' for r in parsed))
    check(rows, 'no occluded column leaked into MOT',
          all(len(l.split(',')) == 9 for l in lines))

    roles = Counter(b['role'] for b in canon)
    check(rows, 'no role filtered: every canonical role reaches MOT',
          sum(roles.values()) == len(parsed), str(dict(roles)))

    ini = mot / split / tag / 'seqinfo.ini'
    check(rows, 'seqinfo.ini copied', ini.exists())
    if ini.exists():
        txt = ini.read_text(encoding='utf-8')
        check(rows, 'seqinfo seqLength is 300', 'seqLength=300' in txt.replace(' ', ''))
    return rows, {'rows': len(parsed), 'identities': len({r['id'] for r in parsed}),
                  'roles': dict(roles), 'worst_bbox_delta_px': round(worst, 4)}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', default='data/tracking_val_gt')
    ap.add_argument('--mot', default='data/tracking_val_gt/mot')
    ap.add_argument('--split', default='EyeCU-val')
    ap.add_argument('--json-out', default=None)
    args = ap.parse_args()

    root, mot = Path(args.root), Path(args.mot)
    man = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    all_rows, report = [], {}

    for s in man['sequences']:
        rows, stats = audit(root, mot, s, args.split)
        report[s['sequence']] = {'checks': rows, 'stats': stats}
        all_rows += rows
        failed = [r for r in rows if not r['ok']]
        print(f'{s["sequence"]:<32}{len(rows) - len(failed)}/{len(rows)} checks  '
              f'rows {stats.get("rows")}  ids {stats.get("identities")}  '
              f'worst bbox delta {stats.get("worst_bbox_delta_px")} px')
        for r in failed:
            print(f'   [FAIL] {r["check"]}   -- {r["detail"]}')

    bench = []
    present = sorted(p.name for p in (mot / args.split).iterdir() if p.is_dir())
    check(bench, 'exactly the three clean sequences exported',
          present == sorted(s['sequence'] for s in man['sequences']), str(present))
    check(bench, f'{EXCLUDED} absent from the export',
          EXCLUDED not in present and not list(mot.rglob(f'*{EXCLUDED}*')))
    seqmap = (mot / 'seqmaps' / f'{args.split}.txt').read_text(encoding='utf-8')
    check(bench, 'seqmap lists exactly those three',
          sorted(seqmap.split()[1:]) == present, seqmap.split())
    check(bench, f'{EXCLUDED} absent from the seqmap', EXCLUDED not in seqmap)
    print('\nbenchmark-level')
    for r in bench:
        print(f'   [{"ok  " if r["ok"] else "FAIL"}] {r["check"]}'
              + (f'   -- {r["detail"]}' if r['detail'] and not r['ok'] else ''))
    all_rows += bench

    failed = [r for r in all_rows if not r['ok']]
    print(f'\n{len(all_rows) - len(failed)}/{len(all_rows)} round-trip checks passed')
    print('MOT ROUND TRIP: ' + ('PASS' if not failed else 'FAIL'))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(
            {'split': args.split, 'sequences': report, 'benchmark_checks': bench,
             'passed': not failed}, indent=1), encoding='utf-8')
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
