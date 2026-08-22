#!/usr/bin/env python
"""
P1 -- score the unchanged production chain against POSSESSION_VAL_V1 using the
sequence-level identity correspondence contract, and report the physical
NO_CONTROL/NO_BALL contract's outcome.

Production chain, UNCHANGED from P0:
    SN3D_BASE -> BallTemporalSelector v1 -> CBIoU humans -> PlayerBallAssigner(70)

What changed since P0 is the SCORING, not the system:
  - the model player box -> GT identity mapping is now
    tools.possession_identity.build_correspondence() (fragment-tolerant,
    sequence-level, UNMAPPABLE-capable) instead of P0's single-frame IoU>=0.50
  - NO_CONTROL is now annotated physically (protocol v2), not via the 70px
    distance PlayerBallAssigner itself is evaluated against

Also recomputes the OLD (P0) mapper on the same rows for the side-by-side
delta required by the P1 milestone. VALIDATION ONLY. No parameter is modified.
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
from possession_identity import (UNMAPPABLE, build_correspondence,      # noqa: E402
                                 tracks_from_frames)
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
OLD_MAP_IOU = 0.50


def load_gt(seq):
    out = defaultdict(list)
    for line in (MOT / seq / 'gt' / 'gt.txt').read_text(encoding='utf-8').splitlines():
        q = line.split(',')
        if len(q) >= 6:
            x, y, w, h = (float(v) for v in q[2:6])
            out[int(q[0])].append((int(q[1]), [x, y, x + w, y + h]))
    return out


def foot_points(bbox):
    x1, y1, x2, y2 = bbox
    return (x1, y2), (x2, y2)


def dist(p, q):
    return float(np.hypot(p[0] - q[0], p[1] - q[1]))


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
    by_seq = defaultdict(list)
    for w in frozen['windows']:
        if any(f"{w['window_id']}:{f}" in labelled for f in w['frames_1based']):
            by_seq[w['sequence']].append(w)

    correspondence_records = {}   # seq -> {track_id: record}

    for seq, wins in sorted(by_seq.items()):
        # M3: a fresh tracker per sequence, not one shared across all of
        # them. Sequences are unrelated videos; a CBIoUTracker instance
        # carries mutable identity state (lost_track_buffer=30 frames of
        # "still might come back") between calls to .update(), so reusing one
        # instance across sequences let one video's leftover tracks compete
        # for matches against a completely different video's opening frames.
        # This is the isolated root cause of the youth_premier_league_1133
        # reproducibility gap P1.1 recorded: both the detector and CBIoU are
        # independently deterministic given fixed input (see
        # experiments/records/experiment_M3/*_determinism_probe_result.json,
        # 204/204 frames identical each), so the only way two runs of this
        # script could diverge is if what carried into a sequence differed --
        # which cross-sequence state leakage makes possible and a fresh
        # instance per sequence makes impossible by construction.
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

        # ---- sequence-level identity correspondence, built ONCE per sequence
        # over the full contiguous run, per the frozen contract.
        pred_tracks = tracks_from_frames(tracks['players'], frame_ids)
        gt_by_frame = {f: gt.get(f, []) for f in frame_ids}
        corr = build_correspondence(pred_tracks, gt_by_frame)
        correspondence_records[seq] = corr

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

            # ---- NEW sequence-level mapping
            new_gt_id, new_reason = None, None
            if assigned != -1:
                rec = corr.get(assigned)
                if rec is not None:
                    new_gt_id = rec['gt_id']
                    new_reason = rec['reason']

            # ---- OLD single-frame P0 mapping, recomputed for the delta
            old_gt_id = None
            if assigned != -1:
                pb = np.array(players[assigned]['bbox']).reshape(1, 4)
                gtb = gt.get(f, [])
                if gtb:
                    m = iou_matrix(np.array([b for _, b in gtb]), pb)
                    j = int(m[:, 0].argmax())
                    if float(m[j, 0]) >= OLD_MAP_IOU:
                        old_gt_id = gtb[j][0]

            # ---- error-mining geometry, recorded automatically
            geom = {}
            if ball.bbox is not None:
                bc = ((ball.bbox[0] + ball.bbox[2]) / 2, (ball.bbox[1] + ball.bbox[3]) / 2)
                geom['ball_centre'] = [round(v, 1) for v in bc]
                if lab.get('gt_player_track_id') is not None:
                    gtb = {tid: b for tid, b in gt.get(f, [])}
                    gtbox = gtb.get(lab['gt_player_track_id'])
                    if gtbox is not None:
                        lf, rf = foot_points(gtbox)
                        geom['gt_player_foot'] = [round(v, 1) for v in
                                                  (lf if dist(lf, bc) < dist(rf, bc) else rf)]
                        geom['ball_to_gt_player_dist'] = round(
                            min(dist(lf, bc), dist(rf, bc)), 1)
                if assigned != -1:
                    pbb = players[assigned]['bbox']
                    lf, rf = foot_points(pbb)
                    geom['picked_player_foot'] = [round(v, 1) for v in
                                                  (lf if dist(lf, bc) < dist(rf, bc) else rf)]
                    geom['ball_to_picked_player_dist'] = round(
                        min(dist(lf, bc), dist(rf, bc)), 1)

            rows.append({'frame_id': key, 'window_id': wid, 'sequence': seq,
                        'label': lab['label_state'],
                        'gt_player': lab.get('gt_player_track_id'),
                        'ball_state': ball.state, 'ball_present': ball.bbox is not None,
                        'assigned_model_track': assigned,
                        'new_gt_id': new_gt_id, 'new_reason': new_reason,
                        'old_gt_id': old_gt_id, **geom})

    # ---------------------------------------------------------------- metrics
    def is_unmappable(r):
        return r['assigned_model_track'] != -1 and r['new_gt_id'] == UNMAPPABLE

    prim = [r for r in rows if r['label'] != 'AMBIGUOUS']
    pl = [r for r in prim if r['label'] == 'PLAYER']
    nc = [r for r in prim if r['label'] == 'NO_CONTROL']
    nb = [r for r in prim if r['label'] == 'NO_BALL']
    ambiguous_n = sum(1 for r in rows if r['label'] == 'AMBIGUOUS')

    unmap_pl = [r for r in pl if is_unmappable(r)]
    unass_pl = [r for r in pl if r['assigned_model_track'] == -1]
    mappable_pl = [r for r in pl if r['assigned_model_track'] != -1 and not is_unmappable(r)]
    correct_pl = [r for r in mappable_pl if r['new_gt_id'] == r['gt_player']]
    wrong_pl = [r for r in mappable_pl if r['new_gt_id'] != r['gt_player']]

    def false_assign(neg_rows):
        return sum(1 for r in neg_rows if r['assigned_model_track'] != -1)

    false_nc = false_assign(nc)
    false_nb = false_assign(nb)
    neg = nc + nb

    # ---- per-window
    per_window = {}
    for wid in sorted({r['window_id'] for r in rows}):
        wr = [r for r in rows if r['window_id'] == wid]
        wprim = [r for r in wr if r['label'] != 'AMBIGUOUS']
        wpl = [r for r in wprim if r['label'] == 'PLAYER']
        wnc = [r for r in wprim if r['label'] == 'NO_CONTROL']
        wnb = [r for r in wprim if r['label'] == 'NO_BALL']
        wmap = [r for r in wpl if r['assigned_model_track'] != -1 and not is_unmappable(r)]
        wcorrect = sum(1 for r in wmap if r['new_gt_id'] == r['gt_player'])
        per_window[wid] = {
            'n_frames': len(wr),
            'PLAYER': {'n': len(wpl), 'mappable': len(wmap), 'correct': wcorrect,
                      'wrong': len(wmap) - wcorrect,
                      'unassigned': sum(1 for r in wpl if r['assigned_model_track'] == -1),
                      'unmappable': sum(1 for r in wpl if is_unmappable(r))},
            'NO_CONTROL': {'n': len(wnc), 'false_assignment': false_assign(wnc)},
            'NO_BALL': {'n': len(wnb), 'false_assignment': false_assign(wnb)},
            'AMBIGUOUS_n': sum(1 for r in wr if r['label'] == 'AMBIGUOUS'),
        }

    # ---- selector-state breakdown
    states = sorted({r['ball_state'] for r in rows})
    by_state = {}
    for s in states:
        sr = [r for r in rows if r['ball_state'] == s]
        spl = [r for r in sr if r['label'] == 'PLAYER']
        smap = [r for r in spl if r['assigned_model_track'] != -1 and not is_unmappable(r)]
        scorrect = sum(1 for r in smap if r['new_gt_id'] == r['gt_player'])
        sneg = [r for r in sr if r['label'] in ('NO_CONTROL', 'NO_BALL')]
        by_state[s] = {
            'n': len(sr),
            'PLAYER_evaluated': len(spl),
            'correct': scorrect, 'wrong': len(smap) - scorrect,
            'unassigned': sum(1 for r in spl if r['assigned_model_track'] == -1),
            'unmappable': sum(1 for r in spl if is_unmappable(r)),
            'false_assignments_on_negatives': false_assign(sneg),
        }

    # ---- error decomposition
    err = Counter()
    err_examples = []
    for r in pl:
        if is_unmappable(r):
            err['UNMAPPABLE'] += 1
        elif r['assigned_model_track'] == -1:
            err['UNASSIGNED_PLAYER'] += 1
        elif r['new_gt_id'] != r['gt_player']:
            if not r.get('ball_present'):
                err['BALL_MISSING'] += 1
            else:
                err['WRONG_PLAYER'] += 1
            err_examples.append(r)
    for r in nc + nb:
        if r['assigned_model_track'] != -1:
            err['SHOULD_BE_UNKNOWN'] += 1
            err_examples.append(r)

    # ---- P0 contract delta: rows that change classification old vs new
    def old_bucket(r):
        if r['label'] != 'PLAYER':
            return None
        if r['assigned_model_track'] == -1:
            return 'unassigned'
        return 'correct' if r['old_gt_id'] == r['gt_player'] else 'wrong_player'

    def new_bucket(r):
        if r['label'] != 'PLAYER':
            return None
        if r['assigned_model_track'] == -1:
            return 'unassigned'
        if is_unmappable(r):
            return 'unmappable'
        return 'correct' if r['new_gt_id'] == r['gt_player'] else 'wrong_player'

    delta_rows = []
    for r in pl:
        ob, nbk = old_bucket(r), new_bucket(r)
        if ob != nbk:
            delta_rows.append({'frame_id': r['frame_id'], 'old': ob, 'new': nbk,
                              'old_gt_id': r['old_gt_id'], 'new_gt_id': r['new_gt_id'],
                              'assigned_model_track': r['assigned_model_track']})

    rep = {
        'benchmark': 'POSSESSION_VAL_V1 (full 60-frame frozen population)',
        'frozen_list_sha256': ann['frozen_list_sha256'],
        'annotation_sha256': None,
        'parameters_modified': 'NONE',
        'chain': 'SN3D_BASE -> BallTemporalSelector v1 -> CBIoU humans -> PlayerBallAssigner(70)',
        'mapping_new': 'sequence-level fragment-tolerant correspondence (experiments/records/experiment_P1/IDENTITY_CORRESPONDENCE_CONTRACT.md)',
        'mapping_old': f'P0 single-frame IoU >= {OLD_MAP_IOU}',
        'n_labelled': len(rows),
        'distribution': dict(Counter(r['label'] for r in rows)),
        'player_frames': {
            'n': len(pl), 'mappable': len(mappable_pl),
            'correct': len(correct_pl), 'wrong_player': len(wrong_pl),
            'unassigned': len(unass_pl), 'unmappable': len(unmap_pl),
            'exact_player_accuracy_among_mappable': (
                round(len(correct_pl) / len(mappable_pl), 4) if mappable_pl else None),
            'overall_player_coverage': (
                round(len(mappable_pl) / len(pl), 4) if pl else None),
            'unmappable_rate': round(len(unmap_pl) / len(pl), 4) if pl else None},
        'no_control_frames': {'n': len(nc), 'correctly_unknown': len(nc) - false_nc,
                              'false_assignments': false_nc},
        'no_ball_frames': {'n': len(nb), 'correctly_unknown': len(nb) - false_nb,
                           'false_assignments': false_nb},
        'false_assignment_rate_on_negatives': (
            round((false_nc + false_nb) / len(neg), 4) if neg else None),
        'ambiguous_n': ambiguous_n,
        'per_window': per_window,
        'selector_state_breakdown': by_state,
        'error_categories': dict(err),
        'p0_contract_delta': {'n_changed': len(delta_rows), 'rows': delta_rows},
        'identity_correspondence': {
            seq: {str(t): {k: v for k, v in rec.items()} for t, rec in tracks_.items()}
            for seq, tracks_ in correspondence_records.items()},
        'rows': rows,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rep, indent=1), encoding='utf-8')
    print(json.dumps({k: v for k, v in rep.items()
                      if k not in ('rows', 'identity_correspondence')}, indent=1))
    print('written:', args.out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
