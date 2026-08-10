#!/usr/bin/env python
"""
Diagnostic: what does supervision's update_with_detections() wrapper do to
EyeCU's accepted detections after internal ByteTrack has already associated?

sv.ByteTrack.update_with_detections() is not a thin pass-through. After calling
update_with_tensors() it computes IoU between the ORIGINAL detection boxes and
the returned track boxes, runs a second linear assignment at cost <= 0.5
(IoU >= 0.5), writes tracker_id only for pairs matched in that step, and then
returns detections[tracker_id != -1] -- discarding the rest. If no track is
returned at all it returns Detections.empty(), dropping every detection in the
frame.

This measures that, and nothing else. It does not tune, does not modify
supervision, and does not touch production. No identity ground truth exists, so
nothing here may be expressed as an identity-switch or association-accuracy
rate.

Method. Two independently constructed trackers per sequence receive identical
frozen accepted detections in identical order:

    Path A   update_with_tensors()      internal view
    Path B   update_with_detections()   public view

Never both on one instance -- that would advance its state twice. Because both
paths are deterministic and receive identical input, their internal state
trajectories coincide, which lets Path A's returned tracks explain Path B's
output. That assumption is verified every frame by replaying supervision's own
wrapper on Path A's tracks and requiring it to reproduce Path B exactly.

Production-exact configuration: EyeCU constructs sv.ByteTrack() with no
arguments (trackers/football_tracker.py), so this audit does the same --
including frame_rate=30 even where the source is 25 fps. Changing that here
would alter tracker lifecycle at the same time as measuring the wrapper.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# supervision's own primitives, imported read-only so the replication is exact
from supervision.detection.utils.iou_and_nms import box_iou_batch  # noqa: E402
from supervision.tracker.byte_tracker import matching  # noqa: E402
import supervision as sv  # noqa: E402

WRAPPER_COST_THRESH = 0.5      # linear_assignment(iou_costs, 0.5) -> IoU >= 0.5
ROOT = Path('data/tracking_val_v1')


def load(seq_file):
    return [json.loads(l) for l in
            Path(seq_file).read_text(encoding='utf-8').splitlines() if l.strip()]


def snapshot(tracker):
    """Read-only internal state counts. A track sitting in lost_tracks is NOT
    the same thing as a detection missing from public output."""
    return {
        'tracked': len(tracker.tracked_tracks),
        'lost': len(tracker.lost_tracks),
        'removed': len(tracker.removed_tracks),
        'tracked_active': sum(1 for t in tracker.tracked_tracks
                              if getattr(t, 'state', None) == 1),
    }


def replay_wrapper(det_boxes, tracks):
    """
    Exact replay of supervision 0.26.1 update_with_detections mapping:
        ious = box_iou_batch(detection_boxes, track_boxes)
        iou_costs = 1 - ious
        matches, _, _ = linear_assignment(iou_costs, 0.5)
    Returns (tracker_id per detection, iou matrix, matched pairs).
    """
    n = len(det_boxes)
    ids = np.full(n, -1, dtype=int)
    if len(tracks) == 0 or n == 0:
        return ids, np.zeros((n, len(tracks))), []
    tb = np.asarray([t.tlbr for t in tracks], dtype=float)
    ious = box_iou_batch(np.asarray(det_boxes, dtype=float), tb)
    matches, _, _ = matching.linear_assignment(1 - ious, WRAPPER_COST_THRESH)
    for i_det, i_trk in matches:
        ids[i_det] = int(tracks[i_trk].external_track_id)
    return ids, ious, [tuple(m) for m in matches]


def audit_sequence(rows, label):
    # two independent instances, production-exact (no constructor arguments)
    ta, tb = sv.ByteTrack(), sv.ByteTrack()
    cat = Counter()
    matched_ious, plausible_unmatched_ious, near_miss = [], [], 0
    internal_returned = public_returned = n_input = 0
    empty_track_frames = empty_track_dropped = 0
    replay_mismatch = 0
    gate_band_total = gate_band_no_id = 0
    snaps = []

    for r in rows:
        dets = r['detections']
        n_input += len(dets)
        if not dets:
            continue
        boxes = np.array([d['bbox'] for d in dets], dtype=float)
        confs = np.array([d['confidence'] for d in dets], dtype=float)

        # ---- Path A: internal
        tensors = np.hstack((boxes, confs[:, None]))
        tracks = ta.update_with_tensors(tensors=tensors)
        internal_returned += len(tracks)
        snaps.append(('A', snapshot(ta)))

        # ---- Path B: public wrapper, independent instance
        d = sv.Detections(xyxy=boxes.copy(), confidence=confs.copy(),
                          class_id=np.zeros(len(dets), dtype=int))
        out = tb.update_with_detections(d)
        public_returned += len(out)
        snaps.append(('B', snapshot(tb)))

        # ---- replay supervision's mapping on Path A's tracks
        ids, ious, pairs = replay_wrapper(boxes, tracks)
        if sorted(int(x) for x in ids if x != -1) != sorted(int(x) for x in out.tracker_id):
            replay_mismatch += 1

        if len(tracks) == 0:
            empty_track_frames += 1
            empty_track_dropped += len(dets)
            cat['E other: wrapper returned Detections.empty()'] += len(dets)
            continue

        matched_det = {int(i) for i, _ in pairs}
        for i in range(len(dets)):
            row = ious[i] if ious.size else np.zeros(0)
            plausible = int((row >= WRAPPER_COST_THRESH).sum())
            if i in matched_det:
                cat['A wrapper-preserved'] += 1
                matched_ious.append(float(row.max()))
            elif plausible >= 2:
                cat['C wrapper-ambiguous'] += 1
                plausible_unmatched_ious.append(float(row.max()))
            elif plausible == 1:
                cat['B wrapper-dropped'] += 1
                plausible_unmatched_ious.append(float(row.max()))
            else:
                cat['D internal-no-match'] += 1
                if row.size and 0.4 <= row.max() < WRAPPER_COST_THRESH:
                    near_miss += 1

            # hidden new-track gate band, observable on the accepted view
            if 0.25 < confs[i] < 0.35:
                gate_band_total += 1
                if ids[i] == -1:
                    gate_band_no_id += 1

    return {
        'sequence': label,
        'input_accepted_detections': n_input,
        'internal_returned_track_observations': internal_returned,
        'public_returned_observations': public_returned,
        'categories': dict(cat),
        'frames_with_no_internal_track': empty_track_frames,
        'detections_dropped_by_empty_return': empty_track_dropped,
        'replay_mismatched_frames': replay_mismatch,
        'matched_iou': {
            'n': len(matched_ious),
            'mean': round(float(np.mean(matched_ious)), 4) if matched_ious else None,
            'p10': round(float(np.percentile(matched_ious, 10)), 4) if matched_ious else None,
            'min': round(float(np.min(matched_ious)), 4) if matched_ious else None,
        },
        'unmatched_plausible_iou': {
            'n': len(plausible_unmatched_ious),
            'mean': round(float(np.mean(plausible_unmatched_ious)), 4)
            if plausible_unmatched_ious else None,
        },
        'pairs_just_below_threshold_0p40_0p50': near_miss,
        'new_track_gate_band_0p25_0p35': gate_band_total,
        'new_track_gate_band_without_id': gate_band_no_id,
        'final_internal_state_A': snaps[-1][1] if snaps else None,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', default=str(ROOT))
    ap.add_argument('--view', choices=['accepted', 'candidate'], default='accepted')
    ap.add_argument('--out', default='experiments/tracking_v2/wrapper_audit/result.json')
    args = ap.parse_args()

    root = Path(args.root)
    man = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    key = 'detections_file' if args.view == 'accepted' else 'candidate_file'

    t = sv.ByteTrack()
    cfg = {'constructor': 'sv.ByteTrack()  # production-exact, no arguments',
           'track_activation_threshold': t.track_activation_threshold,
           'det_thresh': t.det_thresh,
           'minimum_matching_threshold': t.minimum_matching_threshold,
           'minimum_consecutive_frames': t.minimum_consecutive_frames,
           'max_time_lost': t.max_time_lost,
           'frame_rate_used': 30,
           'wrapper_cost_threshold': WRAPPER_COST_THRESH,
           'view': args.view}
    print('production-exact tracker config:')
    for k, v in cfg.items():
        print(f'  {k:<32}{v}')

    results = []
    print(f'\n{"sequence":<30}{"input":>7}{"internal":>10}{"public":>8}'
          f'{"A":>6}{"B":>5}{"C":>5}{"D":>6}{"E":>5}{"replayΔ":>9}')
    for w in man['windows']:
        rows = load(root / w[key])
        r = audit_sequence(rows, w['sequence'])
        results.append(r)
        c = r['categories']
        print(f'  {w["sequence"][:28]:<30}{r["input_accepted_detections"]:>7}'
              f'{r["internal_returned_track_observations"]:>10}'
              f'{r["public_returned_observations"]:>8}'
              f'{c.get("A wrapper-preserved", 0):>6}{c.get("B wrapper-dropped", 0):>5}'
              f'{c.get("C wrapper-ambiguous", 0):>5}{c.get("D internal-no-match", 0):>6}'
              f'{c.get("E other: wrapper returned Detections.empty()", 0):>5}'
              f'{r["replay_mismatched_frames"]:>9}')

    tot = Counter()
    for r in results:
        tot.update(r['categories'])
    n_in = sum(r['input_accepted_detections'] for r in results)
    print(f'\nCOMBINED  input {n_in}   public '
          f'{sum(r["public_returned_observations"] for r in results)}')
    for k in sorted(tot):
        print(f'  {k:<48}{tot[k]:>7}  {tot[k]/n_in:>6.2%}')

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({'config': cfg, 'sequences': results,
                               'combined': dict(tot), 'input_total': n_in},
                              indent=2), encoding='utf-8')
    print(f'\nwritten: {out}')


if __name__ == '__main__':
    main()
