#!/usr/bin/env python
"""
Import manually annotated identity GT from CVAT into EyeCU canonical form.

CANONICAL EXPORT FORMAT: **CVAT for video 1.1** (XML, one file per sequence).

Chosen over MOT because MOT 1.1 carries no label, so exporting through it would
lose the semantic role and force a second manual sidecar. CVAT for video keeps
everything the benchmark needs in one file:

    <track id="0" label="player">
      <box frame="0" outside="0" occluded="0" keyframe="1"
           xtl=".." ytl=".." xbr=".." ybr=".."/>
      ...
    </track>

  persistent identity   <track id>
  semantic role         <track label>
  frame index           <box frame>       0-BASED in CVAT
  visibility            outside="1" ends the visible interval
  keyframes             keyframe="1"; gaps between keyframes are interpolated

Frame conversion is owned here: CVAT frame 0 is EyeCU package frame 1. The
annotator never converts anything by hand.

Interpolation follows CVAT semantics: between two consecutive shapes of a track
the box moves linearly, and the interval ends at a shape with outside="1". A
track is therefore absent exactly where the annotator marked it outside, which
is what an occlusion should produce -- no box is invented while a person is
hidden.

Running this successfully moves the benchmark from UNANNOTATED to
ANNOTATED_PENDING_QC. It never reaches VERIFIED: that requires human QC
confirmation, which is a separate tool.
"""

import argparse
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trackers.detector import HUMAN_CLASSES  # noqa: E402

ROLES = set(HUMAN_CLASSES)
STATUS_PENDING = 'ANNOTATED_PENDING_QC'


def parse_cvat_video(xml_path: Path, n_frames: int):
    """
    -> (boxes, roles, warnings)

    boxes: [{'frame': 1-based, 'id': positive int, 'bbox': [x1,y1,x2,y2],
             'role': str}]
    """
    root = ET.parse(xml_path).getroot()
    tracks = root.findall('.//track')
    if not tracks:
        raise SystemExit(
            f'{xml_path.name}: no <track> elements. This looks like a shape-only '
            f'export ("CVAT for images"). Identity lives in tracks -- re-export '
            f'as "CVAT for video 1.1".')

    boxes, roles, warn = [], {}, []
    for tr in tracks:
        raw_id = int(tr.get('id'))
        ident = raw_id + 1                     # CVAT ids are 0-based
        label = (tr.get('label') or '').strip()
        if label not in ROLES:
            raise SystemExit(f'{xml_path.name}: track {raw_id} has label '
                             f'{label!r}; expected one of {sorted(ROLES)}')
        roles[ident] = label

        shapes = []
        for b in tr.findall('box'):
            shapes.append({
                'frame': int(b.get('frame')),
                'outside': b.get('outside') == '1',
                'xtl': float(b.get('xtl')), 'ytl': float(b.get('ytl')),
                'xbr': float(b.get('xbr')), 'ybr': float(b.get('ybr')),
            })
        shapes.sort(key=lambda s: s['frame'])
        if not shapes:
            warn.append(f'track {raw_id} has no boxes')
            continue

        emitted = {}
        for a, b in zip(shapes, shapes[1:]):
            if a['outside']:
                continue                        # invisible interval: emit nothing
            span = b['frame'] - a['frame']
            # interpolate up to, but not including, the next shape's frame
            for k in range(span):
                f = a['frame'] + k
                t = (k / span) if span else 0.0
                emitted[f] = [a['xtl'] + (b['xtl'] - a['xtl']) * t,
                              a['ytl'] + (b['ytl'] - a['ytl']) * t,
                              a['xbr'] + (b['xbr'] - a['xbr']) * t,
                              a['ybr'] + (b['ybr'] - a['ybr']) * t]
        last = shapes[-1]
        if not last['outside']:
            emitted[last['frame']] = [last['xtl'], last['ytl'],
                                      last['xbr'], last['ybr']]

        for f, bb in emitted.items():
            pf = f + 1                          # -> 1-based package frame
            if not (1 <= pf <= n_frames):
                warn.append(f'track {raw_id}: frame {pf} outside 1..{n_frames}, dropped')
                continue
            boxes.append({'frame': pf, 'id': ident,
                          'bbox': [round(v, 2) for v in bb], 'role': label})

    boxes.sort(key=lambda r: (r['frame'], r['id']))
    return boxes, roles, warn


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', default='data/tracking_val_gt')
    ap.add_argument('--exports', default='data/tracking_val_gt/cvat_exports',
                    help='directory of <sequence>.xml CVAT-for-video exports')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    root, exports = Path(args.root), Path(args.exports)
    mp = root / 'manifest.json'
    man = json.loads(mp.read_text(encoding='utf-8'))

    print(f'{"sequence":<32}{"tracks":>8}{"boxes":>8}{"frames":>8}{"roles":>28}')
    imported = 0
    for s in man['sequences']:
        seq = s['sequence']
        xml = exports / f'{seq}.xml'
        if not xml.exists():
            print(f'  {seq[:30]:<32}{"-- no export found --":>52}')
            continue
        boxes, roles, warn = parse_cvat_video(xml, s['frame_count'])

        # one role per identity, straight from the reviewed CVAT track label
        per_id = defaultdict(set)
        for b in boxes:
            per_id[b['id']].add(b['role'])
        bad = {i: sorted(v) for i, v in per_id.items() if len(v) > 1}
        if bad:
            raise SystemExit(f'{seq}: identities with inconsistent roles {bad}. '
                             f'Split them into separate tracks or correct the '
                             f'label in CVAT; roles are not editable here.')

        # no duplicate identity within a frame
        seen = defaultdict(list)
        for b in boxes:
            seen[b['frame']].append(b['id'])
        dup = {f: [i for i, c in Counter(v).items() if c > 1]
               for f, v in seen.items() if len(v) != len(set(v))}
        if dup:
            raise SystemExit(f'{seq}: duplicate identity within a frame {dict(list(dup.items())[:5])}')

        counts = Counter(roles.values())
        print(f'  {seq[:30]:<32}{len(roles):>8}{len(boxes):>8}'
              f'{len(seen):>8}{str(dict(counts)):>28}')
        for w in warn[:3]:
            print(f'      warning: {w}')

        if not args.dry_run:
            (root / 'annotations').mkdir(exist_ok=True)
            (root / 'roles').mkdir(exist_ok=True)
            (root / s['annotation_file_expected']).write_text(
                json.dumps({'sequence': seq, 'frame_numbering': '1-based',
                            'source': 'CVAT for video 1.1',
                            'boxes': boxes}, indent=1), encoding='utf-8')
            # role sidecar is GENERATED, never hand-written
            (root / s['roles_expected']).write_text(json.dumps({
                'sequence': seq,
                'generated_from': 'CVAT track labels via import_tracking_gt_cvat.py',
                'identity_roles': {str(i): r for i, r in sorted(roles.items())},
            }, indent=1), encoding='utf-8')
        imported += 1

    if args.dry_run:
        print('\n(dry run -- nothing written)')
        return
    if imported == len(man['sequences']):
        man['identity_gt_status'] = STATUS_PENDING
        man['identity_import'] = {
            'canonical_format': 'CVAT for video 1.1',
            'importer': 'tools/import_tracking_gt_cvat.py',
            'frame_conversion': 'CVAT frame 0 -> package frame 1',
            'roles_generated_from': 'CVAT track labels (single source of truth)',
        }
        mp.write_text(json.dumps(man, indent=2, ensure_ascii=False), encoding='utf-8')
        print(f'\nstatus -> {STATUS_PENDING}')
        print('QC confirmation is still required before this GT can be evaluated:')
        print('  python tools/confirm_tracking_gt_qc.py --reviewer "<name>" --confirm')
    else:
        print(f'\n{imported}/{len(man["sequences"])} sequences imported; status '
              f'unchanged until all four are present')


if __name__ == '__main__':
    main()
