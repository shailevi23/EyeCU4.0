#!/usr/bin/env python
"""
Run the LEGACY production baseline over the frozen detections.

supervision==0.26.1 `sv.ByteTrack()` driven exactly as
trackers/football_tracker.py drives it in production:

    detections fed   every human detection the detector emits. Production runs
                     with human_candidate_pool=False and confidence=0.25, so
                     that is the ACCEPTED store (>= 0.25), not the candidate one.
    tracker          sv.ByteTrack() with no arguments -- its real semantics,
                     including the effective ~0.35 new-track gate
                     (det_thresh = track_activation_threshold + 0.1) and the
                     (0.10, 0.25) second-stage pool that this configuration
                     never fills.
    call             update_with_detections(), the wrapper production uses,
                     with its IoU >= 0.5 re-map and its dropping of unmatched
                     tracks.
    output           production withholds anything below human_accept_conf
                     (0.25) from the output; with this feed nothing is below it,
                     so the filter is a no-op and is applied anyway rather than
                     skipped.

Nothing here is normalised toward the modern profile. The baseline has to be
the thing production actually is, or the comparison measures a baseline nobody
runs.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trackers.detector import HUMAN_ACCEPT_CONF, HUMAN_CLASSES  # noqa: E402

CLASS_IDS = {'player': 0, 'goalkeeper': 1, 'referee': 2}


def run(candidates: Path, out: Path, meta_out: Path, min_conf: float):
    import supervision as sv
    rows = [json.loads(l) for l in candidates.read_text(encoding='utf-8').splitlines()
            if l.strip()]
    tracker = sv.ByteTrack()

    lines, n_in, n_out, held, update_s = [], 0, 0, 0, 0.0
    ids = set()
    for rec in rows:
        # frozen store is 0-based; package/MOT are 1-based (verified against
        # the preannotation det.txt to 0.0099 px)
        f = rec['frame'] + 1
        dets = [d for d in rec['detections']
                if d['class'] in HUMAN_CLASSES and d['confidence'] > min_conf]
        n_in += len(dets)
        if not dets:
            continue
        sd = sv.Detections(
            xyxy=np.array([d['bbox'] for d in dets], dtype=float),
            class_id=np.array([CLASS_IDS[d['class']] for d in dets]),
            confidence=np.array([d['confidence'] for d in dets], dtype=float))
        t0 = time.perf_counter()
        tracked = tracker.update_with_detections(sd)
        update_s += time.perf_counter() - t0
        for i in range(len(tracked.xyxy)):
            conf = float(tracked.confidence[i])
            if conf < HUMAN_ACCEPT_CONF:
                held += 1          # association evidence only, as in production
                continue
            tid = tracked.tracker_id[i]
            if tid is None or int(tid) <= 0:
                continue
            x1, y1, x2, y2 = (float(v) for v in tracked.xyxy[i])
            lines.append(f'{f},{int(tid)},{x1:.2f},{y1:.2f},'
                         f'{x2-x1:.2f},{y2-y1:.2f},{conf:.4f},1,1')
            ids.add(int(tid))
            n_out += 1

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('\n'.join(lines) + ('\n' if lines else ''), encoding='utf-8')
    meta = {
        'tracker': 'LEGACY_SUPERVISION_BYTETRACK',
        'profile': 'PRODUCTION_HISTORICAL',
        'implementation': 'supervision.ByteTrack() via update_with_detections',
        'supervision_version': sv.__version__,
        'arguments_passed': 'none -- real production semantics preserved',
        'new_track_gate': 'effective ~0.35 (det_thresh = activation + 0.1), NOT normalised',
        'detections_fed': 'accepted store, confidence > %.2f' % min_conf,
        'output_filter': f'conf >= {HUMAN_ACCEPT_CONF} (production behaviour)',
        'frames': len(rows),
        'detections_supplied': n_in,
        'output_rows': n_out,
        'withheld_below_accept_conf': held,
        'distinct_track_ids': len(ids),
        'tracker_update_seconds': round(update_s, 4),
        'tracker_ms_per_frame': round(1000 * update_s / max(1, len(rows)), 4),
        'python': sys.version.split()[0],
    }
    meta_out.parent.mkdir(parents=True, exist_ok=True)
    meta_out.write_text(json.dumps(meta, indent=1), encoding='utf-8')
    print(f'LEGACY  in {n_in:>6}  out {n_out:>6}  ids {len(ids):>4}  '
          f'{meta["tracker_ms_per_frame"]:.2f} ms/frame')
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--candidates', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--meta-out', required=True)
    ap.add_argument('--min-conf', type=float, default=0.0)
    args = ap.parse_args()
    run(Path(args.candidates), Path(args.out), Path(args.meta_out), args.min_conf)


if __name__ == '__main__':
    main()
