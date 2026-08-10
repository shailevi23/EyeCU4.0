#!/usr/bin/env python
"""
Read-only QC clips around a disappearance and reappearance.

A long gap is the one annotation error that no structural check can catch: the
GT is perfectly well-formed whether or not the person who came back is the
person who left. Only a human looking at the footage can say. This tool puts
the two moments side by side and gets out of the way.

For each identity it renders roughly a second before the person vanishes and a
second after they return:

    target      thick, bright, labelled
    others      thin grey, so the scene reads without competing for attention
    last seen   on the AFTER frames, a marker where the target was last
                observed, plus the pixel distance -- the jump is the thing
                being judged

Outputs an MP4 for scrubbing and a contact sheet for a side-by-side look. No
tracker output is involved, and nothing is corrected: the tool cannot write to
an annotation file.

Every event is recorded as HUMAN_REVIEW_REQUIRED and stays there until a person
decides. A reconnect is not accepted by default -- silently keeping an identity
across a long absence would hand the tracker bake-off an answer key that
already assumes the answer.

Decisions a human has already made live in identity_gap_decisions.json, which
this tool READS and never writes. Regenerating the report must not make a
settled question look open again -- a reviewer who sees HUMAN_REVIEW_REQUIRED
on something they already decided will either redo the work or, worse, assume
their decision was rejected. Where no authoritative decision exists the event
stays HUMAN_REVIEW_REQUIRED, and this tool has no path that creates one.
"""

import argparse
import json
import subprocess
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

TARGET = (70, 240, 255)
OTHER = (150, 150, 150)
GHOST = (90, 90, 255)
CELL_W, CELL_H, COLS = 300, 190, 6


def imread(p: Path):
    if not p.exists():
        return None
    return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)


def imwrite(p: Path, img, q=95):
    ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, q])
    if ok:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(buf.tobytes())
    return ok


DECISIONS_FILE = 'identity_gap_decisions.json'


def authoritative_decisions(out_dir: Path):
    """{id: decision} from the hand-recorded file. Read-only, never created."""
    p = out_dir / DECISIONS_FILE
    if not p.exists():
        return {}
    rec = json.loads(p.read_text(encoding='utf-8'))
    return {int(d['id']): d for d in rec.get('decisions', [])}


def gaps_of(frames):
    fr = sorted(frames)
    missing = sorted(set(range(fr[0], fr[-1] + 1)) - set(fr))
    spans, run = [], []
    for x in missing:
        if run and x == run[-1] + 1:
            run.append(x)
        else:
            if run:
                spans.append((run[0], run[-1]))
            run = [x]
    if run:
        spans.append((run[0], run[-1]))
    return spans


def draw(img, rows, ident, phase, note, last_seen=None, band=True):
    out = img.copy()
    for r in rows:
        x1, y1, x2, y2 = (int(round(v)) for v in r['bbox'])
        if r['id'] == ident:
            continue
        cv2.rectangle(out, (x1, y1), (x2, y2), OTHER, 1)
    if last_seen is not None:
        x1, y1, x2, y2 = (int(round(v)) for v in last_seen)
        cv2.rectangle(out, (x1, y1), (x2, y2), GHOST, 1)
        cv2.putText(out, 'last seen', (x1, max(9, y1 - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, GHOST, 1, cv2.LINE_AA)
    tgt = next((r for r in rows if r['id'] == ident), None)
    if tgt:
        x1, y1, x2, y2 = (int(round(v)) for v in tgt['bbox'])
        cv2.rectangle(out, (x1, y1), (x2, y2), TARGET, 2)
        cv2.putText(out, f'id{ident} {tgt["role"]}', (x1, max(10, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, TARGET, 1, cv2.LINE_AA)
    if not band:
        return out
    bar = np.full((18, out.shape[1], 3), 20, np.uint8)
    cv2.putText(bar, note, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (240, 240, 240) if phase == 'BEFORE' else (140, 240, 140),
                1, cv2.LINE_AA)
    return np.vstack([bar, out])


def contact(frames, cols=COLS):
    cells = []
    for img, cap in frames:
        c = cv2.resize(img, (CELL_W, CELL_H), interpolation=cv2.INTER_AREA)
        bar = np.full((18, CELL_W, 3), 20, np.uint8)
        cv2.putText(bar, cap, (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.34,
                    (235, 235, 235), 1, cv2.LINE_AA)
        cells.append(np.vstack([c, bar]))
    rows = []
    for i in range(0, len(cells), cols):
        row = cells[i:i + cols]
        while len(row) < cols:
            row.append(np.full_like(cells[0], 20))
        rows.append(np.hstack(row))
    return np.vstack(rows)


def clip_for(root: Path, s: dict, ident: int, out_dir: Path, lead: int,
             decided=None):
    seq = s['sequence']
    ann = json.loads((root / s['annotation_file_expected']).read_text(encoding='utf-8'))
    per_frame = defaultdict(list)
    for b in ann['boxes']:
        per_frame[b['frame']].append(b)
    mine = sorted(b['frame'] for b in ann['boxes'] if b['id'] == ident)
    if not mine:
        print(f'  id {ident}: not present')
        return None
    spans = gaps_of(mine)
    if not spans:
        print(f'  id {ident}: no gap; nothing to review')
        return None
    a, b = max(spans, key=lambda t: t[1] - t[0])
    role = next(x['role'] for x in ann['boxes'] if x['id'] == ident)
    last_box = next(x['bbox'] for x in ann['boxes']
                    if x['id'] == ident and x['frame'] == a - 1)
    first_box = next(x['bbox'] for x in ann['boxes']
                     if x['id'] == ident and x['frame'] == b + 1)
    jump = float(np.hypot((first_box[0] + first_box[2]) / 2 - (last_box[0] + last_box[2]) / 2,
                          (first_box[1] + first_box[3]) / 2 - (last_box[1] + last_box[3]) / 2))

    n = s['frame_count']
    before = [f for f in range(max(1, a - lead), a)]
    after = [f for f in range(b + 1, min(n, b + lead) + 1)]
    img1 = root / 'sequences' / seq / 'img1'
    frames_dir = out_dir / f'id{ident}'
    frames_dir.mkdir(parents=True, exist_ok=True)

    seq_imgs, k = [], 0
    for phase, flist in (('BEFORE', before), ('AFTER', after)):
        for f in flist:
            img = imread(img1 / f'{f:06d}.jpg')
            if img is None:
                continue
            if phase == 'BEFORE':
                note = (f'{seq}  id{ident} {role}  BEFORE  f{f}  '
                        f'(vanishes after f{a-1})')
                ghost = None
            else:
                note = (f'{seq}  id{ident} {role}  AFTER  f{f}  '
                        f'(absent f{a}-{b}, {b-a+1} frames, jump {jump:.0f}px)')
                ghost = last_box
            k += 1
            drawn = draw(img, per_frame.get(f, []), ident, phase, note, ghost)
            imwrite(frames_dir / f'{k:06d}.jpg', drawn)
            # the sheet adds its own caption; keeping the in-frame band too
            # puts one cell's label directly above the next cell's picture
            bare = draw(img, per_frame.get(f, []), ident, phase, note, ghost,
                        band=False)
            seq_imgs.append((bare, f'{phase} f{f}'))

    imwrite(out_dir / f'{seq}_id{ident}_contact.jpg', contact(seq_imgs), q=92)

    mp4 = out_dir / f'{seq}_id{ident}_gap.mp4'
    if shutil.which('ffmpeg'):
        p = subprocess.run(
            ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
             '-framerate', '8', '-start_number', '1',
             '-i', str(frames_dir / '%06d.jpg'),
             '-c:v', 'libx264', '-preset', 'slow', '-crf', '16',
             '-pix_fmt', 'yuv420p', '-fps_mode', 'passthrough',
             '-movflags', '+faststart', str(mp4)],
            capture_output=True, text=True)
        if p.returncode != 0:
            print(f'  (ffmpeg failed for id {ident}: {p.stderr[:200]})')
            mp4 = None
    else:
        mp4 = None

    print(f'  id {ident:>3} ({role:<10}) absent f{a}-{b}  {b-a+1} frames, '
          f'jump {jump:.0f}px   {len(before)} before + {len(after)} after')
    # an existing human decision is displayed; its absence is not a decision
    d = (decided or {}).get(ident)
    return {'id': ident, 'role': role,
            'status': d['status'] if d else 'HUMAN_REVIEW_REQUIRED',
            'decision': d.get('decision') if d else None,
            'decided_by': d.get('decided_by', 'human annotator') if d else None,
            'decision_source': DECISIONS_FILE if d else None,
            'gap': [a, b], 'gap_frames': b - a + 1,
            'jump_px': round(jump, 1),
            'last_seen_frame': a - 1, 'reappears_frame': b + 1,
            'last_bbox': [round(v, 2) for v in last_box],
            'first_bbox_after': [round(v, 2) for v in first_box],
            'jump_px_caveat': 'image-space distance between the last and first '
                              'observation. The camera moves during the gap, so '
                              'this is NOT a measure of how far the person '
                              'walked and must not be read as one.',
            'contact_sheet': str(out_dir / f'{seq}_id{ident}_contact.jpg'),
            'video': str(mp4) if mp4 else None,
            'frames_dir': str(frames_dir)}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', default='data/tracking_val_gt')
    ap.add_argument('--sequence', default='women_1_239')
    ap.add_argument('--ids', default='16,14,12')
    ap.add_argument('--lead', type=int, default=None,
                    help='frames before/after; default ~1 second at native fps')
    args = ap.parse_args()

    root = Path(args.root)
    man = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    s = next(x for x in man['sequences'] if x['sequence'] == args.sequence)
    lead = args.lead or int(round(float(s['native_fps'])))
    out_dir = root / 'qc_identity' / args.sequence
    out_dir.mkdir(parents=True, exist_ok=True)

    decided = authoritative_decisions(out_dir)
    print(f'{args.sequence}: ~{lead} frames each side (native '
          f'{s["native_fps"]} fps)')
    if decided:
        print(f'{len(decided)} authoritative decision(s) found in '
              f'{DECISIONS_FILE}; they are displayed, never recomputed')
    recs = []
    for tok in args.ids.split(','):
        r = clip_for(root, s, int(tok.strip()), out_dir, lead, decided)
        if r:
            recs.append(r)
    (out_dir / 'identity_gap_review.json').write_text(json.dumps({
        'sequence': args.sequence,
        'purpose': 'human review of disappearance/reappearance identity',
        'policy': {
            'default': 'A long-gap reconnect is never accepted automatically. '
                       'Every event starts at HUMAN_REVIEW_REQUIRED.',
            'if_confident': 'The same physical person established across the '
                            'disappearance keeps the same identity.',
            'if_uncertain': 'Do not guess. Start a NEW GT identity and record '
                            'an uncertain-reentry QC event.',
            'insufficient_evidence': 'Role or team-kit similarity alone does '
                                     'not establish physical identity.',
            'forbidden': 'Tracker output and appearance embeddings must not '
                         'inform this decision -- the benchmark exists to '
                         'judge trackers, and GT built from one would be '
                         'marking its own homework.',
        },
        'decided_by': 'human, from the footage',
        'authoritative_record': DECISIONS_FILE,
        'this_file_is_generated': True,
        'tracker_output_used': False,
        'embeddings_used': False,
        'alters_annotations': False,
        'lead_frames': lead,
        'open_events': sum(1 for r in recs
                           if r['status'] == 'HUMAN_REVIEW_REQUIRED'),
        'events': recs,
    }, indent=1), encoding='utf-8')
    print(f'\nwritten to {out_dir}')
    print('Nothing was corrected. Each event is a human decision.')


if __name__ == '__main__':
    main()
