#!/usr/bin/env python
"""
EyeCU sequence-level identity correspondence -- possession scoring only.

Implements experiments/records/experiment_P1/IDENTITY_CORRESPONDENCE_CONTRACT.md
(sha256 recorded alongside it). Read that document first; this module is the
executable form of it and adds no rule of its own.

This is NOT HOTA and NOT IDF1. It does not score tracking. It answers only:
when the assigner hands possession to predicted track T, which physical player
did it hand it to -- or is that genuinely undecidable (UNMAPPABLE)?

The mapper P0 used decided that per frame at IoU >= 0.50, so a stable track
that dipped below the bar on a single frame lost its identity for that frame
and the frame scored as a possession error. Here a frame contributes a vote,
not a verdict, and the verdict is taken over the frames a fragment exists on.
"""

from collections import defaultdict

import numpy as np

# --- declared constants; see the contract. Not tunable on possession outcome.
IOU_VOTE = 0.30
MIN_SUPPORT_FRAMES = 2
MIN_SUPPORT_RATE = 0.50
DOMINANCE_RATIO = 2.0
DOMINANCE_MARGIN = 2

UNMAPPABLE = 'UNMAPPABLE'


def iou_1_to_n(box, boxes):
    """box: (4,) xyxy, boxes: (N,4) xyxy -> (N,) IoU."""
    if len(boxes) == 0:
        return np.zeros(0)
    b = np.asarray(box, dtype=float)
    o = np.asarray(boxes, dtype=float)
    x1 = np.maximum(b[0], o[:, 0])
    y1 = np.maximum(b[1], o[:, 1])
    x2 = np.minimum(b[2], o[:, 2])
    y2 = np.minimum(b[3], o[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    ba = max(b[2] - b[0], 0) * max(b[3] - b[1], 0)
    oa = np.clip(o[:, 2] - o[:, 0], 0, None) * np.clip(o[:, 3] - o[:, 1], 0, None)
    union = ba + oa - inter
    return np.where(union > 0, inter / union, 0.0)


def build_correspondence(pred_tracks, gt_by_frame, iou_vote=IOU_VOTE):
    """Contract steps 1-4.

    pred_tracks: {track_id: {frame: bbox_xyxy}}
    gt_by_frame: {frame: [(gt_id, bbox_xyxy), ...]}

    Returns {track_id: record}, record carrying the decision and every number
    it was made from, so the decision can be audited without re-running.
    """
    votes = {}
    evaluable = {}

    # ---- step 1: per-frame votes, argmax GT with ties broken by lowest GT id
    for tid, frames in pred_tracks.items():
        v = defaultdict(int)
        n_eval = 0
        for f, box in frames.items():
            gts = gt_by_frame.get(f) or []
            if not gts:
                continue                      # frame carries no GT -> not evaluable
            n_eval += 1
            ids = [g for g, _ in gts]
            ious = iou_1_to_n(box, [b for _, b in gts])
            best = float(ious.max())
            if best < iou_vote:
                continue                      # overlapped nobody -> no vote
            # deterministic tie-break: among maxima, the lowest GT id
            cand = [ids[i] for i in range(len(ids)) if float(ious[i]) == best]
            v[min(cand)] += 1
        votes[tid] = dict(v)
        evaluable[tid] = n_eval

    # ---- step 2: provisional decision per fragment
    prov = {}
    for tid in pred_tracks:
        v = votes[tid]
        n_eval = evaluable[tid]
        support = sum(v.values())
        ranked = sorted(v.items(), key=lambda kv: (-kv[1], kv[0]))
        top_id, top_n = (ranked[0] if ranked else (None, 0))
        runner_n = ranked[1][1] if len(ranked) > 1 else 0

        rec = {'track_id': tid, 'n_frames': len(pred_tracks[tid]),
               'n_evaluable': n_eval, 'support': support,
               'votes': dict(sorted(v.items())), 'top_gt_id': top_id,
               'top_votes': top_n, 'runner_votes': runner_n,
               'purity': round(top_n / support, 4) if support else None}

        if n_eval < MIN_SUPPORT_FRAMES:
            rec.update(gt_id=UNMAPPABLE, reason='too_short')
        elif support < MIN_SUPPORT_FRAMES:
            rec.update(gt_id=UNMAPPABLE, reason='no_support')
        elif support < MIN_SUPPORT_RATE * n_eval:
            rec.update(gt_id=UNMAPPABLE, reason='low_support_rate')
        elif not (top_n >= DOMINANCE_RATIO * runner_n
                  and top_n - runner_n >= DOMINANCE_MARGIN):
            rec.update(gt_id=UNMAPPABLE, reason='not_dominant')
        else:
            rec.update(gt_id=top_id, reason='mapped')
        prov[tid] = rec

    # ---- step 3: simultaneous conflict. Disjoint fragments of one player are
    # legal and stay; co-existing claimants are not.
    by_gt = defaultdict(list)
    for tid, rec in prov.items():
        if rec['gt_id'] != UNMAPPABLE:
            by_gt[rec['gt_id']].append(tid)

    for gid, tids in by_gt.items():
        if len(tids) < 2:
            continue
        # group co-existing claimants transitively: any shared frame conflicts
        spans = {t: set(pred_tracks[t]) for t in tids}
        groups = []
        for t in sorted(tids):
            hit = [g for g in groups if any(spans[t] & spans[o] for o in g)]
            if not hit:
                groups.append({t})
            else:
                merged = {t}
                for g in hit:
                    merged |= g
                    groups.remove(g)
                groups.append(merged)
        for g in groups:
            if len(g) < 2:
                continue
            strength = {t: prov[t]['votes'].get(gid, 0) for t in g}
            best = max(strength.values())
            winners = [t for t in g if strength[t] == best]
            if len(winners) > 1:
                for t in g:                        # tied -> nobody claims it
                    prov[t].update(gt_id=UNMAPPABLE, reason='conflict_tied')
            else:
                for t in g:
                    if t != winners[0]:
                        prov[t].update(gt_id=UNMAPPABLE, reason='conflict_lost')

    return prov


def tracks_from_frames(player_frames, frame_ids):
    """[{tid: {'bbox': [...]}}, ...] aligned to frame_ids -> {tid: {frame: bbox}}."""
    out = defaultdict(dict)
    for f, per_frame in zip(frame_ids, player_frames):
        for tid, p in (per_frame or {}).items():
            out[tid][f] = list(p['bbox'])
    return dict(out)
