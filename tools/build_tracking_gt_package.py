#!/usr/bin/env python
"""
Build the manual identity-annotation package for EyeCU-Tracking-Val-v1.

This creates the ANSWER KEY's raw materials. It does not create the answer.

A tracker may never define identity ground truth, so this tool emits box
geometry only: preannotations carry frame, bbox and a detector class HINT, and
carry no tracker id, no temporal link and no identity of any kind. The frozen
accepted detection view is reused as draft geometry purely to spare the
annotator drawing ~18k boxes by hand; the annotator must still add missed
people, delete false boxes, fix geometry and roles, and assign every identity.

Layout follows MOTChallenge so the same frames serve annotation, TrackEval and
QC without a second copy:

    sequences/<seq>/img1/000001.jpg ...   1-based, as TrackEval expects
    sequences/<seq>/seqinfo.ini
    preannotations/<seq>.cvat.xml         boxes only, no identity
    preannotations/<seq>.det.txt          MOT detection rows, id column = -1

Frame numbering: the package is 1-based internally (img1/000001.jpg is source
frame `start_frame`). The manifest records both numberings so a GT row can
always be traced back to an absolute source frame.

Verified against TrackEval 1.3.0 source, not assumed:
  - frames are 1-based           (mot_challenge_2d_box.py L215, L227)
  - row is frame,id,x,y,w,h,conf,class,visibility  (L240/241/252/261)
  - GT conf must be non-zero, else the row is dropped  (L394)
  - GT class must be 1 = pedestrian; other values raise (L78, L365-373)
"""

import argparse
import configparser
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trackers.detector import HUMAN_CLASSES  # noqa: E402
from tools.build_derived_train import TEST_MATCHES, VAL_MATCHES  # noqa: E402

BENCHMARK = 'EyeCU-Tracking-Val-v1'
SCHEMA = '1.0'
FROZEN = Path('data/tracking_val_v1')
ROLES = list(HUMAN_CLASSES)
MOT_PEDESTRIAN_CLASS = 1        # TrackEval: only class 1 is evaluable
MOT_GT_CONF = 1                 # TrackEval: zero-marked GT rows are discarded


def sha256_text(t):
    return hashlib.sha256(t.encode('utf-8')).hexdigest()


def cvat_xml(seq, w, h, n_frames, per_frame):
    """CVAT-for-images 1.1: shapes only. No track element, so no identity."""
    out = ['<?xml version="1.0" encoding="utf-8"?>', '<annotations>',
           '  <version>1.1</version>', '  <meta>', '    <task>',
           f'      <name>{escape(seq)}</name>',
           f'      <size>{n_frames}</size>',
           '      <mode>interpolation</mode>',
           '      <labels>']
    for r in ROLES:
        out += ['        <label>', f'          <name>{r}</name>',
                '          <attributes/>', '        </label>']
    out += ['      </labels>', '    </task>', '  </meta>']
    for i, dets in enumerate(per_frame):
        out.append(f'  <image id="{i}" name="{i+1:06d}.jpg" '
                   f'width="{w}" height="{h}">')
        for d in dets:
            x1, y1, x2, y2 = d['bbox']
            out.append(f'    <box label="{escape(d["class"])}" occluded="0" source="auto" '
                       f'xtl="{x1:.2f}" ytl="{y1:.2f}" xbr="{x2:.2f}" ybr="{y2:.2f}" '
                       f'z_order="0"/>')
        out.append('  </image>')
    out.append('</annotations>')
    return '\n'.join(out) + '\n'


def det_txt(per_frame):
    """MOT detection rows. id column is -1: detections carry no identity."""
    rows = []
    for i, dets in enumerate(per_frame, start=1):     # 1-based
        for d in dets:
            x1, y1, x2, y2 = d['bbox']
            rows.append(f'{i},-1,{x1:.2f},{y1:.2f},{x2-x1:.2f},{y2-y1:.2f},'
                        f'{d["confidence"]:.4f},-1,-1,-1')
    return '\n'.join(rows) + '\n'


def seqinfo(seq, n, w, h, fps):
    c = configparser.ConfigParser()
    c.optionxform = str
    c['Sequence'] = {'name': seq, 'imDir': 'img1', 'frameRate': f'{fps:.6g}',
                     'seqLength': str(n), 'imWidth': str(w), 'imHeight': str(h),
                     'imExt': '.jpg'}
    return c


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--frozen', default=str(FROZEN))
    ap.add_argument('--out', default='data/tracking_val_gt')
    ap.add_argument('--quality', type=int, default=95)
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    import cv2
    frozen = Path(args.frozen)
    fman = json.loads((frozen / 'manifest.json').read_text(encoding='utf-8'))
    out = Path(args.out)

    names = {w['match'] for w in fman['windows']}
    if names & TEST_MATCHES:
        raise SystemExit(f'REFUSING: TEST source {sorted(names & TEST_MATCHES)}')
    if not names <= VAL_MATCHES:
        raise SystemExit(f'REFUSING: non-VAL source {sorted(names - VAL_MATCHES)}')

    commit = subprocess.run(['git', 'rev-parse', 'HEAD'],
                            capture_output=True, text=True).stdout.strip()
    tool_sha = hashlib.sha256(
        Path(__file__).read_text(encoding='utf-8').encode('utf-8')).hexdigest()

    man = {
        'benchmark': BENCHMARK,
        'annotation_schema_version': SCHEMA,
        'created': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'code_commit': commit,
        'build_tool_sha256': tool_sha,
        'identity_gt_status': 'UNANNOTATED',
        'identity_provenance': (
            'Identity ground truth MUST be assigned manually by a human and was '
            'NOT generated from tracker IDs. Preannotations contain box geometry '
            'and a detector class hint only -- no tracker id, no temporal link, '
            'no identity.'),
        'target': {
            'classes': ROLES,
            'ball_excluded': True,
            'association_is_class_agnostic_human': True,
            'role_evaluated_separately': True,
            'not_targets': ['coach', 'substitute', 'medical staff', 'ball boy',
                            'spectator', 'ball'],
        },
        'conventions': {
            'bbox': 'visible extent of one person, xyxy in pixels, positive '
                    'width/height, inside the frame; never infer a fully '
                    'occluded person',
            'identity': 'one stable positive integer per physical person per '
                        'sequence; preserved across occlusion, detector misses, '
                        'blur and role jitter; never renumbered because a '
                        'tracker would',
            'occlusion': 'omit the box while fully hidden; resume the SAME id '
                         'when visible again; mark for QC if genuinely ambiguous',
            'entry_exit': 'new identity on first appearance; identity ends on '
                          'exit unless the same person is confidently '
                          'recognised again within the same sequence',
            'role': 'one canonical role per identity; detector role jitter must '
                    'not create a new identity',
            'ignored_objects': 'non-target people are left unlabelled',
        },
        'frame_numbering': {
            'package': '1-based; img1/000001.jpg is source frame start_frame',
            'trackeval': '1-based, verified in trackeval 1.3.0 '
                         'mot_challenge_2d_box.py L215/L227',
        },
        'mot_export': {
            'row': 'frame,id,x,y,w,h,conf,class,visibility',
            'gt_conf': MOT_GT_CONF,
            'gt_class': MOT_PEDESTRIAN_CLASS,
            'note': 'TrackEval discards GT rows whose conf column is 0 (L394) '
                    'and raises on any class outside its valid set (L365-373). '
                    'A trailing -1,-1,-1 is the TRACKER row convention and is '
                    'invalid for GT.',
        },
        'frozen_detection_benchmark': {
            'path': str(frozen),
            'relationship': 'preannotation geometry is the frozen ACCEPTED '
                            'view (>=0.25); the candidate view is not used here',
            'accepted_human_threshold': fman['accepted_human_threshold'],
        },
        'sequences': [],
    }

    print(f'{"sequence":<30}{"frames":>8}{"preann boxes":>14}{"WxH":>12}{"fps":>8}')
    for w in fman['windows']:
        seq, start, n = w['sequence'], w['start_frame'], w['frame_count']
        cap = cv2.VideoCapture(w['source_video'])
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        frames, h_ = [], hashlib.sha256()
        for _ in range(n):
            ok, f = cap.read()
            if not ok:
                break
            frames.append(f)
            h_.update(np.ascontiguousarray(f).tobytes())
        cap.release()
        if len(frames) != n:
            raise SystemExit(f'{seq}: decoded {len(frames)} of {n}')
        if h_.hexdigest() != w['decoded_frames_sha256']:
            raise SystemExit(f'{seq}: decoded pixels differ from the frozen '
                             f'detection benchmark -- refusing to annotate '
                             f'different frames')

        rows = [json.loads(l) for l in
                (frozen / w['detections_file']).read_text(encoding='utf-8').splitlines()
                if l.strip()]
        per_frame = []
        for r in rows:
            keep = []
            for d in r['detections']:
                # geometry + class hint only; strip anything identity-shaped
                keep.append({'bbox': [float(v) for v in d['bbox']],
                             'class': d['class'],
                             'confidence': float(d['confidence'])})
            per_frame.append(keep)
        n_boxes = sum(len(p) for p in per_frame)
        H, W = frames[0].shape[:2]
        print(f'{seq[:28]:<30}{len(frames):>8}{n_boxes:>14}{f"{W}x{H}":>12}'
              f'{w["native_fps"]:>8.2f}')

        xml = cvat_xml(seq, W, H, len(frames), per_frame)
        det = det_txt(per_frame)
        si = seqinfo(seq, len(frames), W, H, w['native_fps'])

        if not args.dry_run:
            sdir = out / 'sequences' / seq
            (sdir / 'img1').mkdir(parents=True, exist_ok=True)
            for i, f in enumerate(frames, start=1):
                ok, buf = cv2.imencode('.jpg', f,
                                       [cv2.IMWRITE_JPEG_QUALITY, args.quality])
                if not ok:
                    raise SystemExit(f'{seq}: could not encode frame {i}')
                (sdir / 'img1' / f'{i:06d}.jpg').write_bytes(buf.tobytes())
            with open(sdir / 'seqinfo.ini', 'w', encoding='utf-8') as fh:
                si.write(fh)
            for sub in ('preannotations', 'annotations', 'mot', 'roles', 'qc'):
                (out / sub).mkdir(parents=True, exist_ok=True)
            (out / 'preannotations' / f'{seq}.cvat.xml').write_text(xml, encoding='utf-8')
            (out / 'preannotations' / f'{seq}.det.txt').write_text(det, encoding='utf-8')

        man['sequences'].append({
            'sequence': seq, 'match': w['match'],
            'source_video': w['source_video'],
            'source_video_sha256': w['source_video_sha256'],
            'decoded_frames_sha256': w['decoded_frames_sha256'],
            'source_frame_range': [start, start + n - 1],
            'package_frame_range': [1, n],
            'frame_count': n,
            'native_fps': w['native_fps'],
            'frame_width': W, 'frame_height': H,
            'preannotation_boxes': n_boxes,
            'preannotation_cvat': f'preannotations/{seq}.cvat.xml',
            'preannotation_det': f'preannotations/{seq}.det.txt',
            'preannotation_cvat_sha256': sha256_text(xml),
            'preannotation_det_sha256': sha256_text(det),
            'annotation_file_expected': f'annotations/{seq}.json',
            'mot_gt_expected': f'mot/{seq}/gt/gt.txt',
            'roles_expected': f'roles/{seq}.json',
        })

    if args.dry_run:
        print('\n(dry run -- nothing written)')
        return man
    out.mkdir(parents=True, exist_ok=True)
    (out / 'manifest.json').write_text(json.dumps(man, indent=2, ensure_ascii=False),
                                       encoding='utf-8')
    print(f'\nwritten: {out}  (identity_gt_status = UNANNOTATED)')
    return man


if __name__ == '__main__':
    main()
