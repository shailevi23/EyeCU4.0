#!/usr/bin/env python
"""
P1.1 -- read-only diagnostic re-run to attribute P1 errors.

This does NOT modify SN3D, BallTemporalSelector, CBIoU, the detector, or
PlayerBallAssigner. It re-runs the identical, unchanged production chain
(same as tools/eval_possession_val_p1.py) once per sequence and, for the
frames named below, dumps evidence that P1's scoring run did not persist:

  - the raw ball candidate list BallTemporalSelector chose from that frame
    (tracker.ball_candidates), so we can tell whether a plausible match-ball
    candidate existed and was passed over, or never existed at all
  - for every predicted player track alive in the frame: its frozen P1
    identity-correspondence mapping (reused from P1_POSSESSION_RESULT.json,
    NOT recomputed), its bbox, and four foot-distance metrics to the selected
    ball (bottom-left corner, bottom-right corner, bottom-centre, nearest
    point on the bottom edge) -- PlayerBallAssigner itself only ever looks at
    the first two.

VALIDATION / DIAGNOSTIC ONLY. No parameter of any production component is
changed by this script.
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trackers.ball_temporal import (BallTemporalSelector, FrameInput,    # noqa: E402
                                    detect_cuts)
from trackers.detector import BALL_ACCEPT_CONF                          # noqa: E402
from trackers.football_tracker import FootballTracker                   # noqa: E402

SEQ = Path('data/tracking_val_gt/sequences')
PV = Path('data/possession_val_v1')
HUMAN_MODEL = 'best_A_960.pt'
HUMAN_IMGSZ = 960

# frames (1-based) needed per sequence, and the last frame each sequence must
# be run to (same contiguous-from-1 requirement as P0/P1)
NEEDED = {
    'bayern_munich_3-1_chelsea_228': {75, 76, 77, 78, 79, 80, 81, 82, 83, 84,
                                      195, 196, 197, 198, 199, 200, 201, 202, 203, 204},
    'women_1_239': {75, 76, 77, 78, 79, 80, 81, 82, 83, 84},
    'youth_premier_league_1133': {75, 76, 77, 78, 79, 80, 81, 82, 83, 84,
                                  195, 196, 197, 198, 199, 200, 201, 202, 203, 204},
}


def nearest_point_on_bottom_edge(bbox, point):
    x1, y1, x2, y2 = bbox
    px, py = point
    cx = min(max(px, x1), x2)
    return (cx, y2)


def dist(p, q):
    return float(np.hypot(p[0] - q[0], p[1] - q[1]))


def main():
    import cv2

    p1 = json.loads(Path('experiments/records/experiment_P1/P1_POSSESSION_RESULT.json')
                    .read_text(encoding='utf-8'))
    corr = p1['identity_correspondence']
    row_by_key = {r['frame_id']: r for r in p1['rows']}

    out = {}
    for seq, frames_needed in NEEDED.items():
        # M3: a fresh tracker per sequence -- see the identical note in
        # tools/eval_possession_val_p1.py. This is the exact script whose
        # rerun P1.1 found nondeterministic for youth_premier_league_1133;
        # the shared tracker across bayern/women/youth in this loop was the
        # isolated root cause (see experiments/records/experiment_M3/).
        tracker = FootballTracker(model_path=HUMAN_MODEL, imgsz=HUMAN_IMGSZ,
                                  confidence=BALL_ACCEPT_CONF, persist_cache=False,
                                  ball_candidate_pool=True,
                                  ball_detector_backend='sn3d')
        last = max(frames_needed)
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

        seq_corr = corr.get(seq, {})
        seq_out = {}
        for f in sorted(frames_needed):
            i = f - 1
            sel_out = outs[i]
            ball_c = (None if sel_out.bbox is None else
                     ((sel_out.bbox[0] + sel_out.bbox[2]) / 2,
                      (sel_out.bbox[1] + sel_out.bbox[3]) / 2))
            raw_cands = [{'centre': [round((c['bbox'][0] + c['bbox'][2]) / 2, 1),
                                    round((c['bbox'][1] + c['bbox'][3]) / 2, 1)],
                         'conf': round(float(c.get('confidence', c.get('conf', -1))), 3),
                         'bbox': [round(v, 1) for v in c['bbox']],
                         'dist_to_selected': (None if ball_c is None else
                                              round(dist(((c['bbox'][0] + c['bbox'][2]) / 2,
                                                         (c['bbox'][1] + c['bbox'][3]) / 2),
                                                        ball_c), 1))}
                        for c in cands[i]]

            players = tracks['players'][i]
            player_geom = []
            for tid, p in players.items():
                bbox = p['bbox']
                x1, y1, x2, y2 = bbox
                rec = seq_corr.get(str(tid), {})
                entry = {'track_id': tid, 'mapped_gt_id': rec.get('gt_id'),
                        'reason': rec.get('reason'), 'purity': rec.get('purity'),
                        'bbox': [round(v, 1) for v in bbox]}
                if ball_c is not None:
                    bl, br = (x1, y2), (x2, y2)
                    bc = ((x1 + x2) / 2, y2)
                    edge = nearest_point_on_bottom_edge(bbox, ball_c)
                    entry.update(
                        d_bottom_left=round(dist(bl, ball_c), 1),
                        d_bottom_right=round(dist(br, ball_c), 1),
                        d_bottom_centre=round(dist(bc, ball_c), 1),
                        d_nearest_bottom_edge=round(dist(edge, ball_c), 1),
                        assigner_metric=round(min(dist(bl, ball_c), dist(br, ball_c)), 1))
                player_geom.append(entry)
            player_geom.sort(key=lambda e: e.get('assigner_metric', 1e9))

            seq_out[f] = {
                'ball_state': sel_out.state,
                'ball_selected_centre': (None if ball_c is None else
                                         [round(ball_c[0], 1), round(ball_c[1], 1)]),
                'raw_candidates': raw_cands,
                'n_raw_candidates': len(raw_cands),
                'players': player_geom,
            }
        out[seq] = seq_out

    Path('experiments/records/experiment_P1').mkdir(parents=True, exist_ok=True)
    dest = Path('experiments/records/experiment_P1/P1_1_ATTRIBUTION_DIAGNOSTICS.json')
    dest.write_text(json.dumps(out, indent=1), encoding='utf-8')
    print('written:', dest)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
