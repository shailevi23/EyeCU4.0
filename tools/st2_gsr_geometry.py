#!/usr/bin/env python
"""
Measure GSR box geometry, identity persistence and role stability.

The first sampled annotations came back with boxes WIDER than tall for standing
humans (a player 47x40, a goalkeeper 65x37). That is not what a tight person box
looks like, and it decides whether this data can train a detector at all -- so
it is measured over a large sample rather than accepted from two examples.

Also answered here, because both bear on tracking value:
  * do track_ids persist, or restart?
  * does one track_id keep one role, jersey and team for the whole half?
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'tools'))
from st2_gsr_scan import iter_objects           # noqa: E402

CAT = {1: 'player', 2: 'goalkeeper', 3: 'referee', 4: 'ball'}


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('file')
    ap.add_argument('--every', type=int, default=37,
                    help='parse every Nth annotation; 37 is coprime with the '
                         '22 boxes per frame so it does not sample one track')
    ap.add_argument('--limit', type=int, default=120000)
    ap.add_argument('--out', default='experiments/soccertrack_audit/reports/gsr_geometry.json')
    args = ap.parse_args()

    p = Path(args.file)
    wh = defaultdict(list)
    tracks = defaultdict(lambda: {'frames': [], 'roles': Counter(),
                                  'jerseys': Counter(), 'teams': Counter(),
                                  'player_ids': Counter()})
    frames = set()
    n = 0
    for a in iter_objects(p, b'"annotations"', args.every, args.limit):
        cat = CAT.get(a.get('category_id'))
        if cat is None:
            continue
        n += 1
        b = a.get('bbox_image')
        if isinstance(b, dict) and b.get('w') and b.get('h'):
            wh[cat].append((b['w'], b['h'], b['x'], b['y']))
        t = a.get('track_id')
        at = a.get('attributes') or {}
        fid = int(str(a['image_id'])[-6:]) if a.get('image_id') else None
        if t is not None:
            r = tracks[t]
            r['frames'].append(fid)
            r['roles'][at.get('role')] += 1
            r['jerseys'][str(at.get('jersey'))] += 1
            r['teams'][at.get('team')] += 1
            r['player_ids'][at.get('player_id')] += 1
        if fid:
            frames.add(fid)

    rec = {'file': p.name, 'parsed_annotations': n, 'sample_every': args.every,
           'distinct_frames_touched': len(frames), 'geometry': {}, 'tracks': {}}

    print(f'{p.name}: parsed {n} object annotations, {len(frames)} frames touched')
    for cat, v in wh.items():
        a = np.array(v, float)
        w, h = a[:, 0], a[:, 1]
        ar = h / np.maximum(w, 1e-6)
        rec['geometry'][cat] = {
            'n': len(a),
            'w': {'median': float(np.median(w)), 'p10': float(np.percentile(w, 10)),
                  'p90': float(np.percentile(w, 90)), 'min': float(w.min()),
                  'max': float(w.max())},
            'h': {'median': float(np.median(h)), 'p10': float(np.percentile(h, 10)),
                  'p90': float(np.percentile(h, 90)), 'min': float(h.min()),
                  'max': float(h.max())},
            'aspect_h_over_w': {'median': float(np.median(ar)),
                                'p10': float(np.percentile(ar, 10)),
                                'p90': float(np.percentile(ar, 90))},
            'fraction_wider_than_tall': float((ar < 1).mean()),
            'height_le_20px': int((h <= 20).sum()),
            'height_le_40px': int((h <= 40).sum()),
        }
        g = rec['geometry'][cat]
        print(f'  {cat:<11} n={len(a):>7}  w med {g["w"]["median"]:5.1f}  '
              f'h med {g["h"]["median"]:5.1f}  aspect h/w med '
              f'{g["aspect_h_over_w"]["median"]:5.2f}  '
              f'wider-than-tall {100*g["fraction_wider_than_tall"]:5.1f}%')

    spans, multi_role, multi_jersey, multi_team = [], 0, 0, 0
    for t, r in tracks.items():
        f = [x for x in r['frames'] if x]
        if f:
            spans.append((min(f), max(f), len(f)))
        multi_role += len(r['roles']) > 1
        multi_jersey += len(r['jerseys']) > 1
        multi_team += len(r['teams']) > 1
    rec['tracks'] = {
        'distinct_track_ids': len(tracks),
        'track_ids': sorted(tracks)[:40],
        'tracks_with_more_than_one_role': multi_role,
        'tracks_with_more_than_one_jersey': multi_jersey,
        'tracks_with_more_than_one_team': multi_team,
        'span_first_to_last_frame': [[int(a), int(b), int(c)] for a, b, c in spans[:30]],
        'role_histogram': {str(t): dict(r['roles']) for t, r in
                           list(tracks.items())[:30]},
    }
    print(f'  distinct track_ids: {len(tracks)}   '
          f'tracks whose role changes: {multi_role}   '
          f'jersey changes: {multi_jersey}   team changes: {multi_team}')

    dst = REPO / args.out
    dst.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(dst.read_text(encoding='utf-8')) if dst.exists() else {}
    prev[p.name] = rec
    dst.write_text(json.dumps(prev, indent=1, ensure_ascii=False), encoding='utf-8')
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
