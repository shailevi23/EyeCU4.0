#!/usr/bin/env python
"""
EXPERIMENTAL -- T1: Temporal Candidate Recovery.

One question: can weak per-frame ball evidence in the 0.01-0.10 band be
exploited temporally without increasing hallucinations?

The failure analysis found that during player contact the detector's evidence
oscillates frame to frame at constant ball size -- a ball is accepted, then
weakly proposed, then absent, then accepted again, all within one duel. The
frozen selector already rescues from the 0.10-0.25 band and already interpolates
short bidirectional gaps. T1 changes exactly one thing: it widens the rescue
pool's floor from 0.10 to 0.01.

What is FROZEN and reused unchanged:
  - the detector, its weights, imgsz, and the 0.25 accept threshold
  - suppress_ball_duplicates at BALL_DEDUPE_IOU
  - the pass-1 gate: 60px base + 40px growth, scaled by frame_width/640,
    velocity-predicted from >=2 history points
  - selection within the gate: closest to prediction, then most confident
  - cut handling (detect_cuts), which clears history
  - the entire pass-2 interpolation (BallTemporalSelector._interpolate),
    called directly rather than reimplemented

What is EXPERIMENTAL:
  - the rescue pool floor, 0.10 -> 0.01
  - applying the same frozen 60px gate symmetrically, anchoring on t+1 as well
    as on history. The NUMBER is frozen; only its symmetric use is new.

T1 cannot reach a frame that has no proposal at all at >=0.01. Its addressable
population is the low-confidence band only; any coverage of a truly
proposal-less frame comes from the frozen interpolation, not from T1.

VALIDATION ONLY. The benchmark is VAL_ONLY and there is no split argument.

    python tools/experiment_temporal_candidate_recovery.py --model <last.pt> --out-dir <dir>
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compare_models import iou_matrix                                  # noqa: E402
from trackers.ball_temporal import (INTERPOLATED, OBSERVED, RECOVERED,  # noqa: E402
                                    UNKNOWN, BallOutput,
                                    BallTemporalSelector, FrameInput,
                                    _centre, detect_cuts)
from trackers.detector import (BALL_ACCEPT_CONF, BALL_CANDIDATE_CONF,   # noqa: E402
                               BALL_DEDUPE_IOU, LocalDetector,
                               suppress_ball_duplicates)

MATCH_IOU = 0.5
T1_FLOOR = 0.01
TV = Path('data/temporal_val')


def load_gt(path: Path, w: int, h: int) -> np.ndarray:
    if not path.exists():
        return np.empty((0, 4))
    out = []
    for line in path.read_text(encoding='utf-8').splitlines():
        p = line.split()
        if len(p) == 5:
            cx, cy, bw, bh = (float(v) for v in p[1:5])
            out.append([(cx-bw/2)*w, (cy-bh/2)*h, (cx+bw/2)*w, (cy+bh/2)*h])
    return np.array(out).reshape(-1, 4)


def hit(gt: np.ndarray, box) -> bool:
    if not len(gt) or box is None:
        return False
    return float(iou_matrix(gt, np.array(box).reshape(1, 4)).max()) >= MATCH_IOU


def t1_pass1(frames, sel, use_next_anchor: bool):
    """Frozen pass-1 logic, with the pool floor taken from `sel.candidate_conf`
    and an optional symmetric t+1 anchor. With use_next_anchor=False and the
    default floor this reproduces BallTemporalSelector.run's first pass exactly
    -- asserted by the validity check in main()."""
    out, history = [], []
    accepted_at = {}
    for i, fr in enumerate(frames):
        obs = [c for c in fr.candidates if c['confidence'] >= sel.accept_conf]
        accepted_at[i] = max(obs, key=lambda c: (c['confidence'], -c['bbox'][0])) if obs else None

    decisions = []
    for i, fr in enumerate(frames):
        if fr.cut:
            history = []
        if accepted_at[i] is not None:
            best = accepted_at[i]
            out.append(BallOutput(OBSERVED, list(best['bbox']), float(best['confidence'])))
            history.append((i, _centre(best['bbox']), fr.timestamp))
            continue

        pool = [c for c in fr.candidates
                if sel.candidate_conf <= c['confidence'] < sel.accept_conf]
        pred, radius = sel._predict(history, fr.timestamp, fr.dt)

        scored = []
        for c in pool:
            cx, cy = _centre(c['bbox'])
            if pred is not None:
                d = ((cx-pred[0])**2 + (cy-pred[1])**2) ** 0.5
                if d <= radius:
                    scored.append((d, -c['confidence'], c, 'history'))
            if use_next_anchor and i+1 < len(frames) and not frames[i+1].cut:
                nxt = accepted_at[i+1]
                if nxt is not None:
                    ncx, ncy = _centre(nxt['bbox'])
                    d = ((cx-ncx)**2 + (cy-ncy)**2) ** 0.5
                    # same frozen number, one adjacent frame of motion
                    if d <= sel.gate_base_px:
                        scored.append((d, -c['confidence'], c, 'next_frame'))
        if scored:
            scored.sort(key=lambda s: (s[0], s[1]))
            d, _, best, src = scored[0]
            out.append(BallOutput(RECOVERED, list(best['bbox']), float(best['confidence'])))
            history.append((i, _centre(best['bbox']), fr.timestamp))
            decisions.append({'frame_index': i, 'source': f'promoted_current_proposal[{src}]',
                              'conf': round(float(best['confidence']), 4),
                              'distance_px': round(float(d), 2),
                              'bbox': [round(v, 2) for v in best['bbox']]})
            continue
        out.append(BallOutput(UNKNOWN))
    return out, decisions


def metrics(frames_meta, gts, outputs):
    raw = orr = cov = 0
    gt_frames = empty = false_rec = 0
    st = Counter()
    for f in frames_meta:
        o = outputs[f['file']]
        g = gts[f['file']]
        st[o.state] += 1
        box = o.bbox
        if len(g):
            gt_frames += 1
            raw += (o.state == OBSERVED and hit(g, box))
            orr += (o.state in (OBSERVED, RECOVERED) and hit(g, box))
            cov += hit(g, box)
        else:
            empty += 1
            if box is not None:
                false_rec += 1
    return {'gt_frames': gt_frames, 'empty_frames': empty,
            'raw_hits': raw, 'observed_plus_recovered_hits': orr, 'coverage_hits': cov,
            'raw_recall': round(raw/gt_frames, 4) if gt_frames else None,
            'observed_plus_recovered_recall': round(orr/gt_frames, 4) if gt_frames else None,
            'trajectory_coverage': round(cov/gt_frames, 4) if gt_frames else None,
            'hallucinated_empty_frames': false_rec,
            'false_recovery_rate_on_empty': round(false_rec/empty, 4) if empty else None,
            'states': dict(st)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', required=True)
    ap.add_argument('--imgsz', type=int, default=960)
    ap.add_argument('--out-dir', required=True, type=Path)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    import cv2
    from PIL import Image

    man = json.loads((TV / 'manifest.json').read_text(encoding='utf-8'))
    frames_meta = man['frames']
    by_window = defaultdict(list)
    for f in frames_meta:
        by_window[(f['match'], f['window'])].append(f)
    for v in by_window.values():
        v.sort(key=lambda f: f['order_in_window'])

    ctrl_det = LocalDetector(args.model, confidence=BALL_ACCEPT_CONF,
                             imgsz=args.imgsz, ball_candidate_pool=True)
    t1_det = LocalDetector(args.model, confidence=T1_FLOOR,
                           imgsz=args.imgsz, ball_candidate_pool=False)

    ctrl_cand, t1_cand, gts, thumbs, sizes = {}, {}, {}, {}, {}
    for key, fl in sorted(by_window.items()):
        for f in fl:
            p = TV / 'images' / f['file']
            img = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
            w, h = Image.open(p).size
            sizes[f['file']] = (w, h)
            ctrl_cand[f['file']] = [d for d in ctrl_det.detect(img) if d['class'] == 'ball']
            raw_balls = [d for d in t1_det.detect(img) if d['class'] == 'ball']
            # same frozen suppression rule, applied at the experimental floor
            t1_cand[f['file']] = suppress_ball_duplicates(raw_balls, BALL_DEDUPE_IOU)
            gts[f['file']] = load_gt(TV/'labels'/f'{Path(f["file"]).stem}.txt', w, h)
            thumbs[f['file']] = cv2.cvtColor(cv2.resize(img, (64, 36)), cv2.COLOR_BGR2GRAY)
        print(f'  {key[0]} w{key[1]}: {len(fl)} frames', flush=True)

    ctrl_out, t1_out, all_decisions = {}, {}, []
    parity_ok = True
    for key, fl in sorted(by_window.items()):
        w, _ = sizes[fl[0]['file']]
        cuts = detect_cuts([thumbs[f['file']] for f in fl])
        dt = 1.0 / fl[0]['effective_fps']   # uniform within a window

        def build(cand_map, floor):
            sel = BallTemporalSelector(frame_width=w, candidate_conf=floor)
            # Uniform synthetic timestamps, exactly as tools/eval_temporal_val.py
            # builds them. Using the real timestamp_seconds instead shifts the
            # pass-2 interpolation gate and the control stops being the frozen
            # control.
            fin = [FrameInput(candidates=cand_map[f['file']],
                              timestamp=f['order_in_window'] * dt,
                              dt=dt, cut=cuts[i])
                   for i, f in enumerate(fl)]
            return sel, fin

        # CONTROL -- the frozen class, called exactly as tools/eval_temporal_val.py does
        sel_c, fin_c = build(ctrl_cand, BALL_CANDIDATE_CONF)
        res_c = sel_c.run(fin_c)
        for f, o in zip(fl, res_c):
            ctrl_out[f['file']] = o

        # parity: our pass-1 with the frozen floor and no t+1 anchor must equal
        # the frozen run's pass-1. Guards against silent divergence.
        chk, _ = t1_pass1(fin_c, sel_c, use_next_anchor=False)
        sel_c._interpolate(fin_c, chk)
        if [(o.state, o.bbox) for o in chk] != [(o.state, o.bbox) for o in res_c]:
            parity_ok = False

        # T1
        sel_t, fin_t = build(t1_cand, T1_FLOOR)
        res_t, decisions = t1_pass1(fin_t, sel_t, use_next_anchor=True)
        sel_t._interpolate(fin_t, res_t)
        for f, o in zip(fl, res_t):
            t1_out[f['file']] = o
        for d in decisions:
            f = fl[d['frame_index']]
            g = gts[f['file']]
            d.update({'file': f['file'], 'window': f"{f['match']} w{f['window']}",
                      'gt_present': bool(len(g)),
                      'gt_matched': bool(hit(g, d['bbox'])),
                      'true_recovery': bool(len(g) and hit(g, d['bbox'])),
                      'false_recovery': bool(not len(g))})
            all_decisions.append(d)

    ctrl_m = metrics(frames_meta, gts, ctrl_out)
    t1_m = metrics(frames_meta, gts, t1_out)

    # ---- per-GT-ball subgroups, defined by the CONTROL outcome
    subgroup = []
    for f in frames_meta:
        g = gts[f['file']]
        if not len(g):
            continue
        acc = [c for c in ctrl_cand[f['file']] if c['confidence'] >= BALL_ACCEPT_CONF]
        raw01 = t1_cand[f['file']]
        for i, gb in enumerate(g):
            det = float(iou_matrix(gb.reshape(1, 4),
                                   np.array([c['bbox'] for c in acc]))[0].max()) >= MATCH_IOU if acc else False
            r01 = float(iou_matrix(gb.reshape(1, 4),
                                   np.array([c['bbox'] for c in raw01]))[0].max()) >= MATCH_IOU if raw01 else False
            base = 'DETECTED' if det else ('LOW_CONF_RECOVERABLE' if r01 else 'NO_USABLE_PROPOSAL')
            co, to = ctrl_out[f['file']], t1_out[f['file']]
            subgroup.append({
                'file': f['file'], 'window': f"{f['match']} w{f['window']}", 'gt_index': i,
                'baseline_class': base,
                'control_or_hit': bool(co.state in (OBSERVED, RECOVERED) and hit(gb.reshape(1, 4), co.bbox)),
                'control_cov_hit': bool(hit(gb.reshape(1, 4), co.bbox)),
                't1_or_hit': bool(to.state in (OBSERVED, RECOVERED) and hit(gb.reshape(1, 4), to.bbox)),
                't1_cov_hit': bool(hit(gb.reshape(1, 4), to.bbox)),
                't1_state': to.state,
            })

    report = {
        'EXPERIMENTAL': 'T1 -- Temporal Candidate Recovery',
        'not_production': True,
        'model': args.model, 'imgsz': args.imgsz,
        'frozen': {'accept_conf': BALL_ACCEPT_CONF, 'control_pool_floor': BALL_CANDIDATE_CONF,
                   'dedupe_iou': BALL_DEDUPE_IOU, 'match_iou': MATCH_IOU,
                   'gate_base_px_at_640': 60.0, 'gate_growth_px_at_640': 40.0,
                   'interpolation': 'BallTemporalSelector._interpolate, unchanged, both arms'},
        'experimental': {'t1_pool_floor': T1_FLOOR,
                         'symmetric_next_frame_anchor': True,
                         'note': 'gate NUMBER frozen; only its symmetric use is experimental'},
        'pass1_parity_with_frozen_run': parity_ok,
        'control': ctrl_m, 't1': t1_m,
        'subgroup_records': subgroup,
        'recovery_decisions': all_decisions,
    }
    (args.out_dir / 't1_summary.json').write_text(json.dumps(report, indent=1), encoding='utf-8')
    (args.out_dir / 't1_per_frame_decisions.json').write_text(
        json.dumps(all_decisions, indent=1), encoding='utf-8')

    print('\n' + '=' * 78)
    print('EXPERIMENTAL -- T1: Temporal Candidate Recovery')
    print('=' * 78)
    print(f"pass-1 parity with frozen selector: {'OK' if parity_ok else 'DIVERGED'}")
    for name, m in (('CONTROL', ctrl_m), ('T1', t1_m)):
        print(f"\n{name}")
        print(f"  observed+recovered : {m['observed_plus_recovered_hits']}/{m['gt_frames']} = {m['observed_plus_recovered_recall']}")
        print(f"  trajectory coverage: {m['coverage_hits']}/{m['gt_frames']} = {m['trajectory_coverage']}")
        print(f"  hallucinated empty : {m['hallucinated_empty_frames']}/{m['empty_frames']} = {m['false_recovery_rate_on_empty']}")
        print(f"  states             : {m['states']}")
    print(f"\n  report -> {args.out_dir/'t1_summary.json'}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
