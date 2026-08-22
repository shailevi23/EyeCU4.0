#!/usr/bin/env python
"""
P0 -- score the current production chain against POSSESSION_VAL_V1.

Runs, unchanged:
    SN3D_BASE ball branch  ->  BallTemporalSelector v1  ->  CBIoU human tracking
    ->  PlayerBallAssigner(max_distance=70)

No parameter is modified. The model's assigned player is mapped to a GT identity
by IoU against the frozen tracking GT boxes (>= 0.50), not by assuming model
track ids equal GT ids.

AMBIGUOUS frames are excluded from primary accuracy, as the protocol requires.

VALIDATION ONLY.
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compare_models import iou_matrix                                   # noqa: E402
from trackers.ball_temporal import (BallTemporalSelector, FrameInput,    # noqa: E402
                                    detect_cuts)
from trackers.detector import BALL_ACCEPT_CONF                          # noqa: E402
from trackers.football_tracker import FootballTracker                   # noqa: E402
from trackers.player_ball_assigner import PlayerBallAssigner            # noqa: E402

SEQ = Path('data/tracking_val_gt/sequences')
MOT = Path('data/tracking_val_gt/mot/EyeCU-val')
PV = Path('data/possession_val_v1')
HUMAN_MODEL = 'best_A_960.pt'
HUMAN_IMGSZ = 960
MAP_IOU = 0.50


def load_gt(seq):
    out = defaultdict(list)
    for line in (MOT / seq / 'gt' / 'gt.txt').read_text(encoding='utf-8').splitlines():
        q = line.split(',')
        if len(q) >= 6:
            x, y, w, h = (float(v) for v in q[2:6])
            out[int(q[0])].append((int(q[1]), [x, y, x + w, y + h]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--annotations', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    import cv2

    frozen = json.loads((PV / 'POSSESSION_VAL_V1_FROZEN.json').read_text(encoding='utf-8'))
    ann = json.loads(Path(args.annotations).read_text(encoding='utf-8'))
    labelled = {r['frame_id']: r for r in ann['annotations']}

    assigner = PlayerBallAssigner(max_distance=70)

    rows = []
    # One pass per SEQUENCE from frame 1, not per 10-frame window. Two reasons,
    # both found the hard way: the detector caches by frame_id, so reusing a
    # tracker across clips whose indices restart at 0 replays the first clip's
    # detections; and CBIoU needs more than 10 frames to confirm a track, so a
    # window-sized run emits no players at all and nothing can be assigned.
    by_seq = defaultdict(list)
    for w in frozen['windows']:
        if any(f"{w['window_id']}:{f}" in labelled for f in w['frames_1based']):
            by_seq[w['sequence']].append(w)

    for seq, wins in sorted(by_seq.items()):
        # M3: a fresh tracker per sequence -- see the identical note in
        # tools/eval_possession_val_p1.py. The detector-cache fix above
        # (comment right above) covered the detector's own per-frame-id
        # cache; it did not cover CBIoUTracker's own persistent identity
        # state (lost_track_buffer=30 frames), which was still being shared
        # across unrelated sequences until this fix.
        tracker = FootballTracker(model_path=HUMAN_MODEL, imgsz=HUMAN_IMGSZ,
                                  confidence=BALL_ACCEPT_CONF, persist_cache=False,
                                  ball_candidate_pool=True,
                                  ball_detector_backend='sn3d')
        gt = load_gt(seq)
        last = max(max(w['frames_1based']) for w in wins)
        frame_ids = list(range(1, last + 1))
        imgs = [cv2.imdecode(np.fromfile(str(SEQ / seq / 'img1' / f'{f:06d}.jpg'),
                                         dtype=np.uint8), cv2.IMREAD_COLOR)
                for f in frame_ids]
        tracker.detector.clear_cache()
        print(f'{seq}: running {len(imgs)} contiguous frames', flush=True)

        tracks = tracker.get_object_tracks(imgs)
        cands = tracker.ball_candidates
        thumbs = [cv2.cvtColor(cv2.resize(im, (64, 36)), cv2.COLOR_BGR2GRAY)
                  for im in imgs]
        cuts = detect_cuts(thumbs)
        sel = BallTemporalSelector(frame_width=imgs[0].shape[1])
        fin = [FrameInput(candidates=[dict(c) for c in cands[i]],
                          timestamp=i * 0.04, dt=0.04, cut=cuts[i])
               for i in range(len(imgs))]
        outs = sel.run(fin)

      # score only the frozen, labelled frames
        want = {f: w['window_id'] for w in wins for f in w['frames_1based']}
        for f, wid in sorted(want.items()):
            i = f - 1
            key = f'{wid}:{f}'
            if key not in labelled:
                continue
            lab = labelled[key]
            ball = outs[i]
            players = tracks['players'][i]
            assigned = -1
            if ball.bbox is not None and players:
                assigned = assigner.assign_ball_to_player(players, ball.bbox)

            pred_gt_id = None
            if assigned != -1:
                pb = np.array(players[assigned]['bbox']).reshape(1, 4)
                gtb = gt.get(f, [])
                if gtb:
                    m = iou_matrix(np.array([b for _, b in gtb]), pb)
                    j = int(m[:, 0].argmax())
                    if float(m[j, 0]) >= MAP_IOU:
                        pred_gt_id = gtb[j][0]

            rows.append({'frame_id': key, 'label': lab['label_state'],
                         'gt_player': lab.get('gt_player_track_id'),
                         'ball_state': ball.state,
                         'ball_present': ball.bbox is not None,
                         'assigned_model_track': assigned,
                         'assigned_gt_id': pred_gt_id})

    # ---- metrics
    prim = [r for r in rows if r['label'] != 'AMBIGUOUS']
    pl = [r for r in prim if r['label'] == 'PLAYER']
    nc = [r for r in prim if r['label'] == 'NO_CONTROL']
    nb = [r for r in prim if r['label'] == 'NO_BALL']

    correct = sum(1 for r in pl if r['assigned_gt_id'] == r['gt_player'])
    wrong = sum(1 for r in pl if r['assigned_model_track'] != -1
                and r['assigned_gt_id'] != r['gt_player'])
    unass = sum(1 for r in pl if r['assigned_model_track'] == -1)
    false_nc = sum(1 for r in nc if r['assigned_model_track'] != -1)
    false_nb = sum(1 for r in nb if r['assigned_model_track'] != -1)
    neg = nc + nb

    rep = {
        'benchmark': 'POSSESSION_VAL_V1 (annotated subset)',
        'frozen_list_sha256': ann['frozen_list_sha256'],
        'parameters_modified': 'NONE',
        'chain': 'SN3D_BASE -> BallTemporalSelector v1 -> CBIoU humans -> PlayerBallAssigner(70)',
        'mapping': f'model player box -> GT id by IoU >= {MAP_IOU}',
        'n_labelled': len(rows),
        'distribution': dict(Counter(r['label'] for r in rows)),
        'player_frames': {
            'n': len(pl), 'correct': correct, 'wrong_player': wrong,
            'unassigned': unass,
            'exact_player_accuracy': round(correct / len(pl), 4) if pl else None,
            'assignment_coverage': round((len(pl) - unass) / len(pl), 4) if pl else None},
        'no_control_frames': {'n': len(nc), 'correctly_unknown': len(nc) - false_nc,
                              'false_assignments': false_nc},
        'no_ball_frames': {'n': len(nb), 'correctly_unknown': len(nb) - false_nb,
                           'false_assignments': false_nb},
        'false_assignment_rate_on_negatives': (
            round((false_nc + false_nb) / len(neg), 4) if neg else None),
        'ball_state_breakdown': {
            s: {'n': sum(1 for r in rows if r['ball_state'] == s),
                'assigned': sum(1 for r in rows if r['ball_state'] == s
                                and r['assigned_model_track'] != -1)}
            for s in sorted({r['ball_state'] for r in rows})},
        'rows': rows,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rep, indent=1), encoding='utf-8')
    print(json.dumps({k: v for k, v in rep.items() if k != 'rows'}, indent=1))
    for r in rows:
        print(f"  {r['frame_id']:<40} lab {r['label']:<11} gt {str(r['gt_player']):<5} "
              f"state {r['ball_state']:<24} assigned_gt {r['assigned_gt_id']}")
    print('written:', args.out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
