#!/usr/bin/env python
"""
Audit the small SoccerTrack v2 components: BAS, RAW metadata, diversity.

Everything here is small enough to read whole, so it is read whole. The point
is to establish, from the files rather than from the project's documentation:

  * what BAS events actually contain -- and specifically whether an event ever
    carries a ball position, which decides whether BAS can serve ball detection
    or only event detection
  * what roles exist in the squad metadata, since GSR itself has zero referee
    annotations and the role vocabulary is the only other place a referee could
    be declared
  * how many INDEPENDENT matches, venues, teams and camera geometries the ten
    downloaded halves actually represent
"""

import csv
import json
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
AUDIT = REPO / 'experiments' / 'soccertrack_audit'
EX = AUDIT / 'extracted'


def audit_bas():
    out = {'files': {}, 'event_classes': Counter(), 'schema_keys': Counter(),
           'events_with_ball_location': 0, 'events_with_player': 0,
           'total_events': 0}
    for p in sorted((EX / 'bas').rglob('*_events.json')):
        d = json.loads(p.read_text(encoding='utf-8'))
        acts = d.get('actions', [])
        out['files'][p.parent.name] = {
            'file': p.name, 'fps': d.get('fps'), 'match_id': d.get('match_id'),
            'events': len(acts), 'top_keys': sorted(d)}
        for a in acts:
            out['total_events'] += 1
            out['event_classes'][a.get('label')] += 1
            for k in a:
                out['schema_keys'][k] += 1
            if a.get('player_id'):
                out['events_with_player'] += 1
            # a ball location would have to be a coordinate; 'position' is a
            # timestamp in milliseconds, not a place. Check for any 2D field.
            if any(k in a for k in ('x', 'y', 'ball_x', 'ball_y', 'location',
                                    'loc', 'bbox', 'position_x')):
                out['events_with_ball_location'] += 1
        # confirm what 'position' means
        if acts:
            pos = [int(a['position']) for a in acts[:200] if str(a.get('position', '')).isdigit()]
            gt = [a.get('gameTime') for a in acts[:200]]
            out['files'][p.parent.name]['position_semantics'] = (
                f'monotonic={all(b >= a for a, b in zip(pos, pos[1:]))}, '
                f'first={pos[:3]}, gameTime={gt[:3]}, '
                f'position/1000 vs gameTime seconds consistent='
                f'{abs(pos[1]/1000 - _gt_seconds(gt[1])) < 3 if len(pos) > 1 else None}')
    out['event_classes'] = dict(out['event_classes'].most_common())
    out['schema_keys'] = dict(out['schema_keys'])
    return out


def _gt_seconds(gt):
    m = re.match(r'(\d+)\s*-\s*(\d+):(\d+)', gt or '')
    return int(m.group(2)) * 60 + int(m.group(3)) if m else -999


def audit_metadata():
    out = {'matches': {}, 'position_vocabulary': Counter(),
           'role_like_entries': Counter()}
    for p in sorted(EX.glob('raw/*/*_tracker_box_metadata.xml')):
        root = ET.parse(p).getroot()
        m = root.find('match')
        pitch = root.find('pitch')
        teams = [t.attrib for t in root.findall('.//team')]
        players = [pl.attrib for pl in root.findall('.//player')]
        periods = {pr.get('period'): dict(pr.attrib) for pr in root.findall('.//period')}
        for pl in players:
            out['position_vocabulary'][pl.get('position')] += 1
        out['matches'][m.get('matchId')] = {
            'title': m.get('matchTitle'),
            'datetime_local': m.get('matchDatetimeLocal'),
            'timezone': m.get('timezone'),
            'pitch_m': [pitch.get('width'), pitch.get('height')] if pitch is not None else None,
            'teams': [{'id': t.get('id'), 'nameEn': t.get('nameEn'),
                       'side': t.get('side')} for t in teams],
            'players': len(players),
            'periods': {k: {'frameStart': v.get('frameStart'),
                            'frameEnd': v.get('frameEnd'), 'fps': v.get('fps')}
                        for k, v in periods.items()
                        if v.get('frameStart') not in (None, 'nan')},
        }
    out['position_vocabulary'] = dict(out['position_vocabulary'].most_common())
    return out


def audit_geometry():
    """Camera geometry per match -- the only evidence of venue/rig diversity."""
    out = {}
    for d in sorted((EX / 'raw').iterdir()):
        if not d.is_dir():
            continue
        mid = d.name
        rec = {}
        hp = d / f'{mid}_homography.npy'
        if hp.exists():
            rec['homography'] = np.load(hp).round(6).tolist()
        mp = d / f'{mid}_mapx.npy'
        if mp.exists():
            a = np.load(mp, mmap_mode='r')
            rec['undistort_canvas'] = [int(a.shape[1]), int(a.shape[0])]
        ip = d / f'{mid}_camera_intrinsics.npz'
        if ip.exists():
            z = np.load(ip, allow_pickle=True)
            rec['K'] = z['K'].round(3).tolist()
            rec['D'] = z['D'].ravel().round(6).tolist()
            rec['calibration_rms'] = float(z['rms'])
        kp = d / f'{mid}_keypoints.json'
        if kp.exists():
            rec['pitch_keypoints'] = len(json.loads(kp.read_text(encoding='utf-8')))
        pp = d / f'{mid}_padding_info.csv'
        if pp.exists():
            rec['padding_info'] = list(csv.DictReader(pp.read_text(encoding='utf-8')
                                                      .splitlines()))
        out[mid] = rec
    # group by canvas + intrinsics: identical rigs are not independent geometry
    groups = defaultdict(list)
    for mid, r in out.items():
        groups[(tuple(r.get('undistort_canvas', [])), str(r.get('K')))].append(mid)
    return out, {f'rig_{i+1}': v for i, v in enumerate(groups.values())}


def audit_player_nodes():
    out = {}
    for p in sorted((EX / 'raw').glob('*/*_player_nodes.csv')):
        rows = list(csv.DictReader(p.read_text(encoding='utf-8').splitlines()))
        out[p.parent.name] = {'rows': len(rows), 'columns': list(rows[0]) if rows else [],
                              'distinct_players': len({r['player_id'] for r in rows}),
                              'event_periods': sorted({r['event_period'] for r in rows})}
    return out


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    bas = audit_bas()
    meta = audit_metadata()
    geom, rigs = audit_geometry()
    nodes = audit_player_nodes()

    print(f'BAS: {bas["total_events"]} events across {len(bas["files"])} matches')
    print(f'  event keys: {bas["schema_keys"]}')
    print(f'  events carrying a ball LOCATION: {bas["events_with_ball_location"]}')
    print(f'  events carrying a player id    : {bas["events_with_player"]}')
    print(f'  classes ({len(bas["event_classes"])}): {bas["event_classes"]}')

    print(f'\nSquad metadata: {len(meta["matches"])} matches')
    print(f'  position vocabulary: {meta["position_vocabulary"]}')
    teams = {t['nameEn'] for m in meta['matches'].values() for t in m['teams']}
    print(f'  distinct teams: {len(teams)} -> {sorted(teams)}')

    print(f'\nCamera geometry groups (identical canvas + intrinsics):')
    for k, v in rigs.items():
        print(f'  {k}: {v}')

    out = {'bas': bas, 'metadata': meta, 'geometry': geom,
           'camera_geometry_groups': rigs, 'player_nodes': nodes}
    (AUDIT / 'reports').mkdir(parents=True, exist_ok=True)
    (AUDIT / 'reports' / 'components.json').write_text(
        json.dumps(out, indent=1, ensure_ascii=False), encoding='utf-8')
    print('\nwrote reports/components.json')


if __name__ == '__main__':
    main()
