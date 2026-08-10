#!/usr/bin/env python
"""
Read-only QC renderer for identity GT. Never writes to annotations.

Draws each box with its GT identity and role, colouring by identity so an ID
change on the same person is visible as a colour change. Also flags, per frame,
the mistakes that are hardest to catch by scrubbing:

    duplicate id      the same identity on two boxes in one frame
    id reuse jump     an identity reappears far from where it vanished
    role disagreement an identity whose role is not constant
    overlap           two GT boxes at IoU >= 0.9, likely a duplicated person

Outputs annotated frames and a text report. It opens annotation files in read
mode only; a failed QC pass must be fixed in the annotation tool, not here.
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROLE_TAG = {'player': 'P', 'goalkeeper': 'G', 'referee': 'R'}


def colour(i):
    rng = np.random.default_rng(i * 9973)
    return tuple(int(x) for x in rng.integers(60, 255, size=3))


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def qc(root: Path, seq: str, out_dir: Path, stride: int, render: bool):
    import cv2
    man = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    s = next(x for x in man['sequences'] if x['sequence'] == seq)
    ann_path = root / s['annotation_file_expected']
    if not ann_path.exists():
        raise SystemExit(f'no annotation for {seq}: {ann_path} (annotate first)')
    data = json.loads(ann_path.read_text(encoding='utf-8'))     # read-only

    per_frame = defaultdict(list)
    for b in data['boxes']:
        per_frame[b['frame']].append(b)

    issues = []
    roles = defaultdict(Counter)
    last_seen = {}
    for f in range(1, s['frame_count'] + 1):
        rows = per_frame.get(f, [])
        ids = [r['id'] for r in rows]
        for i, c in Counter(ids).items():
            if c > 1:
                issues.append(f'frame {f}: identity {i} appears {c} times')
        for a in range(len(rows)):
            roles[rows[a]['id']][rows[a].get('role')] += 1
            for b in range(a + 1, len(rows)):
                if iou(rows[a]['bbox'], rows[b]['bbox']) >= 0.9:
                    issues.append(f'frame {f}: ids {rows[a]["id"]}/{rows[b]["id"]} '
                                  f'overlap at IoU>=0.9')
        for r in rows:
            cx = ((r['bbox'][0] + r['bbox'][2]) / 2, (r['bbox'][1] + r['bbox'][3]) / 2)
            prev = last_seen.get(r['id'])
            if prev:
                pf, pc = prev
                gap = f - pf
                dist = float(np.hypot(cx[0] - pc[0], cx[1] - pc[1]))
                w = r['bbox'][2] - r['bbox'][0]
                if gap > 1 and dist > 3 * max(w, 1) :
                    issues.append(f'frame {f}: identity {r["id"]} reappears after '
                                  f'{gap} frames {dist:.0f}px away -- verify it is '
                                  f'the same person')
            last_seen[r['id']] = (f, cx)

    for i, c in roles.items():
        if len(c) > 1:
            issues.append(f'identity {i}: role is not constant {dict(c)}')

    if render:
        out_dir.mkdir(parents=True, exist_ok=True)
        for f in range(1, s['frame_count'] + 1, stride):
            img = cv2.imread(str(root / 'sequences' / seq / 'img1' / f'{f:06d}.jpg'))
            if img is None:
                continue
            for r in per_frame.get(f, []):
                x1, y1, x2, y2 = (int(v) for v in r['bbox'])
                col = colour(r['id'])
                cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
                tag = f"{r['id']}{ROLE_TAG.get(r.get('role'), '?')}"
                cv2.putText(img, tag, (x1, max(10, y1 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
            cv2.putText(img, f'{seq}  f{f}', (5, 15), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (255, 255, 255), 1, cv2.LINE_AA)
            ok, buf = cv2.imencode('.jpg', img)
            if ok:
                (out_dir / f'{f:06d}.jpg').write_bytes(buf.tobytes())

    ids = {b['id'] for b in data['boxes']}
    print(f'\n{seq}')
    print(f'  boxes {len(data["boxes"])}   identities {len(ids)}   '
          f'frames with annotation {len(per_frame)}/{s["frame_count"]}')
    print(f'  QC issues: {len(issues)}')
    for msg in issues[:30]:
        print(f'    - {msg}')
    if len(issues) > 30:
        print(f'    ... and {len(issues)-30} more')
    return issues


def render_video(frames_dir: Path, out: Path, fps: str):
    """
    Encode the QC frames into one scrubbing video.

    An identity error is a property of a person over time, so it is spotted by
    watching, not by opening 300 stills. The video is a review aid built from
    the QC frames; nothing reads it back.
    """
    import shutil
    import subprocess
    if not shutil.which('ffmpeg'):
        print('  (ffmpeg not found; skipping video)')
        return None
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
           '-framerate', fps, '-start_number', '1',
           '-i', str(frames_dir / '%06d.jpg'),
           '-c:v', 'libx264', '-preset', 'slow', '-crf', '16',
           '-pix_fmt', 'yuv420p', '-fps_mode', 'passthrough',
           '-movflags', '+faststart', str(out)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        print(f'  (ffmpeg failed: {p.stderr[:300]})')
        return None
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', default='data/tracking_val_gt')
    ap.add_argument('--sequence', default=None, help='default: all')
    ap.add_argument('--stride', type=int, default=1)
    ap.add_argument('--no-render', action='store_true',
                    help='report only; draw nothing')
    ap.add_argument('--video', action='store_true',
                    help='also encode the QC frames into an MP4 for scrubbing')
    args = ap.parse_args()
    root = Path(args.root)
    man = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    seqs = [args.sequence] if args.sequence else [s['sequence'] for s in man['sequences']]
    total = 0
    for s in seqs:
        out_dir = root / 'qc' / s
        total += len(qc(root, s, out_dir, args.stride, not args.no_render))
        if args.video and not args.no_render:
            if args.stride != 1:
                print('  (video needs --stride 1; skipping)')
                continue
            from fractions import Fraction
            spec = next(x for x in man['sequences'] if x['sequence'] == s)
            f = Fraction(float(spec['native_fps'])).limit_denominator(1000)
            vid = render_video(out_dir, root / 'qc' / f'{s}_qc.mp4',
                               f'{f.numerator}/{f.denominator}')
            if vid:
                print(f'  QC video: {vid}')
    print(f'\ntotal QC issues: {total}')


if __name__ == '__main__':
    main()
