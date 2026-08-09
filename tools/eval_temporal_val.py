#!/usr/bin/env python
"""
Evaluate ball detection and temporal recovery on the continuous benchmark.

Reports the frozen baseline (detector only, temporal recovery off) and, with
--temporal, the BallTemporalSelector result, keeping the two strictly apart.
A recovered or interpolated ball is never counted as a raw detector hit -- that
conflation is exactly what makes a smoothed demo video look like a better
detector than it is.

Three views are reported throughout, because close-up and non-gameplay frames
answer a different question from gameplay frames: on a close-up there is
usually no identifiable ball at all, so the only correct behaviour is
`unknown`, and any carried or interpolated box is a false recovery.

VALIDATION ONLY. --split test is refused.

Example:
    python tools/eval_temporal_val.py --model best_A_960.pt
    python tools/eval_temporal_val.py --model best_A_960.pt --temporal
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compare_models import iou_matrix                       # noqa: E402
from trackers.ball_temporal import (INTERPOLATED, OBSERVED,  # noqa: E402
                                    RECOVERED, UNKNOWN,
                                    BallTemporalSelector, FrameInput,
                                    detect_cuts)
from trackers.detector import (BALL_ACCEPT_CONF, BALL_CANDIDATE_CONF,  # noqa: E402
                               BALL_DEDUPE_IOU, LocalDetector)

MATCH_IOU = 0.5
TV = Path('data/temporal_val')


def load_gt_ball(path: Path, w: int, h: int):
    """Benchmark labels are single-class (0 = ball)."""
    if not path.exists():
        return np.empty((0, 4))
    out = []
    for line in path.read_text(encoding='utf-8').splitlines():
        p = line.split()
        if len(p) == 5:
            cx, cy, bw, bh = (float(v) for v in p[1:5])
            out.append([(cx - bw / 2) * w, (cy - bh / 2) * h,
                        (cx + bw / 2) * w, (cy + bh / 2) * h])
    return np.array(out).reshape(-1, 4)


def hit(gt, box) -> bool:
    if not len(gt) or box is None:
        return False
    return float(iou_matrix(gt, np.array(box).reshape(1, 4)).max()) >= MATCH_IOU


def block(title, rows):
    print(f'\n{title}')
    w = max(len(k) for k, _ in rows) + 2
    for k, v in rows:
        print(f'  {k:<{w}}{v}')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', default='best_A_960.pt')
    ap.add_argument('--imgsz', type=int, default=960)
    ap.add_argument('--split', default='val')
    ap.add_argument('--temporal', action='store_true',
                    help='Also run BallTemporalSelector (predeclared settings).')
    ap.add_argument('--out', default='temporal_val_results.json')
    args = ap.parse_args()

    if args.split == 'test':
        sys.exit('Refusing: the frozen test split must not be used.')

    man = json.loads((TV / 'manifest.json').read_text(encoding='utf-8'))
    frames_meta = man['frames']
    by_window = defaultdict(list)
    for f in frames_meta:
        by_window[(f['match'], f['window'])].append(f)
    for v in by_window.values():
        v.sort(key=lambda f: f['order_in_window'])

    import cv2
    from PIL import Image

    det = LocalDetector(args.model, confidence=BALL_ACCEPT_CONF,
                        imgsz=args.imgsz, ball_candidate_pool=True)

    # ---- inference, once, per window (order matters for the selector)
    cand, gts, sizes, thumbs = {}, {}, {}, {}
    for key, fl in sorted(by_window.items()):
        for f in fl:
            p = TV / 'images' / f['file']
            img = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
            dets = det.detect(img)
            cand[f['file']] = [d for d in dets if d['class'] == 'ball']
            w, h = Image.open(p).size
            sizes[f['file']] = (w, h)
            gts[f['file']] = load_gt_ball(
                TV / 'labels' / f'{Path(f["file"]).stem}.txt', w, h)
            thumbs[f['file']] = cv2.cvtColor(cv2.resize(img, (64, 36)),
                                             cv2.COLOR_BGR2GRAY)
        print(f'  {key[0]} w{key[1]}: {len(fl)} frames', flush=True)

    views = {'GAMEPLAY ONLY': lambda f: f['shot_type'] == 'gameplay',
             'CLOSEUP / NON-GAMEPLAY ONLY':
                 lambda f: f['shot_type'] == 'closeup_or_non_gameplay',
             'ALL FRAMES': lambda f: True}
    report = {'model': args.model,
              'thresholds': {'candidate_conf': BALL_CANDIDATE_CONF,
                             'accept_conf': BALL_ACCEPT_CONF,
                             'dedupe_iou': BALL_DEDUPE_IOU,
                             'match_iou': MATCH_IOU}}

    # =================================================== frozen baseline
    print('\n' + '=' * 78)
    print('FROZEN BASELINE -- detector only, temporal recovery OFF')
    print(f'A@{args.imgsz}, candidate floor {BALL_CANDIDATE_CONF}, '
          f'dedupe IoU {BALL_DEDUPE_IOU}, accept {BALL_ACCEPT_CONF}, '
          f'match IoU {MATCH_IOU}')
    print('=' * 78)

    base = {}
    for f in frames_meta:
        g = gts[f['file']]
        acc = [c for c in cand[f['file']] if c['confidence'] >= BALL_ACCEPT_CONF]
        acc.sort(key=lambda c: -c['confidence'])
        tp = sum(1 for c in acc if hit(g, c['bbox']))
        tp = min(tp, len(g))
        base[f['file']] = {'n_gt': len(g), 'n_pred': len(acc), 'tp': tp,
                           'fp': len(acc) - tp, 'fn': len(g) - tp}

    def summarise(pred_fn, keep):
        agg = Counter()
        for f in frames_meta:
            if not keep(f):
                continue
            d = pred_fn(f)
            for k, v in d.items():
                agg[k] += v
            agg['frames'] += 1
            agg['empty_frames'] += (d['n_gt'] == 0)
        return agg

    report['baseline'] = {}
    for vname, keep in views.items():
        a = summarise(lambda f: base[f['file']], keep)
        prec = a['tp'] / (a['tp'] + a['fp']) if (a['tp'] + a['fp']) else None
        rec = a['tp'] / a['n_gt'] if a['n_gt'] else None
        block(vname, [
            ('frames', a['frames']),
            ('visible GT ball instances', a['n_gt']),
            ('empty / no-visible-ball frames', a['empty_frames']),
            ('TP / FP / FN', f"{a['tp']} / {a['fp']} / {a['fn']}"),
            ('precision', f'{prec:.4f}' if prec is not None else '-'),
            ('recall', f'{rec:.4f}' if rec is not None else '-'),
            ('FP per frame', f"{a['fp'] / a['frames']:.4f}" if a['frames'] else '-'),
        ])
        report['baseline'][vname] = {
            'frames': a['frames'], 'gt': a['n_gt'], 'empty': a['empty_frames'],
            'tp': a['tp'], 'fp': a['fp'], 'fn': a['fn'],
            'precision': None if prec is None else round(prec, 4),
            'recall': None if rec is None else round(rec, 4),
            'fp_per_frame': round(a['fp'] / a['frames'], 4) if a['frames'] else None}

    for label, grouper in (('PER MATCH', lambda f: f['match']),
                           ('PER WINDOW', lambda f: f"{f['match']} w{f['window']}")):
        print(f'\n{label} (baseline)')
        hdr = f'  {"key":<36}{"frames":>7}{"GT":>5}{"empty":>7}{"TP":>5}{"FP":>5}{"FN":>5}{"recall":>9}{"FP/fr":>8}'
        print(hdr); print('  ' + '-' * (len(hdr) - 2))
        groups = defaultdict(Counter)
        for f in frames_meta:
            g = groups[grouper(f)]
            for k, v in base[f['file']].items():
                g[k] += v
            g['frames'] += 1
            g['empty_frames'] += (base[f['file']]['n_gt'] == 0)
        for k in sorted(groups):
            a = groups[k]
            r = a['tp'] / a['n_gt'] if a['n_gt'] else 0.0
            print(f'  {k:<36}{a["frames"]:>7}{a["n_gt"]:>5}{a["empty_frames"]:>7}'
                  f'{a["tp"]:>5}{a["fp"]:>5}{a["fn"]:>5}{r:>9.4f}'
                  f'{a["fp"]/a["frames"]:>8.3f}')
        report[f'baseline_{label.lower().replace(" ", "_")}'] = {
            k: dict(v) for k, v in groups.items()}

    multi = [f['file'] for f in frames_meta if base[f['file']]['n_gt'] > 1]
    print(f'\nframes containing >1 GT ball: {len(multi)}')
    report['frames_with_multiple_gt_balls'] = multi

    if not args.temporal:
        Path(args.out).write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(f'\nwritten: {args.out}')
        return

    # =================================================== temporal selector
    print('\n' + '=' * 78)
    print('BALL TEMPORAL SELECTOR -- predeclared v1 settings, first run, untuned')
    print('=' * 78)

    states, outputs = {}, {}
    cut_count = 0
    for key, fl in sorted(by_window.items()):
        cuts = detect_cuts([thumbs[f['file']] for f in fl])
        cut_count += sum(cuts)
        w, _ = sizes[fl[0]['file']]
        sel = BallTemporalSelector(frame_width=w)
        fin = []
        for i, f in enumerate(fl):
            dt = 1.0 / f['effective_fps']
            fin.append(FrameInput(candidates=cand[f['file']],
                                  timestamp=f['order_in_window'] * dt,
                                  dt=dt, cut=cuts[i]))
        for f, o, c in zip(fl, sel.run(fin), cuts):
            states[f['file']] = o.state
            outputs[f['file']] = o.bbox
            f['_cut'] = c

    trans = Counter()
    runs = defaultdict(list)
    report['temporal'] = {}
    for vname, keep in views.items():
        n = gt_frames = empty = 0
        raw_hits = or_hits = cov = 0
        false_rec = halluc = unk = 0
        st = Counter()
        for f in frames_meta:
            if not keep(f):
                continue
            n += 1
            g = gts[f['file']]
            s = states[f['file']]
            box = outputs[f['file']]
            st[s] += 1
            unk += (s == UNKNOWN)
            # Transitions are tallied ONLY on the ALL FRAMES pass. Every frame
            # belongs to its shot-type view *and* to ALL, so counting on every
            # pass double-counts each frame.
            count_transitions = (vname == 'ALL FRAMES')
            if len(g):
                gt_frames += 1
                rh = (s == OBSERVED and hit(g, box))
                raw_hits += rh
                or_hits += (s in (OBSERVED, RECOVERED) and hit(g, box))
                cov += hit(g, box)
                if not rh and count_transitions:
                    trans[f'raw miss -> {s}'] += 1
            else:
                empty += 1
                if box is not None:
                    false_rec += 1
                    halluc += 1
                    if count_transitions:
                        trans[f'GT empty -> false recovery ({s})'] += 1
                elif count_transitions:
                    trans['GT empty -> correctly unknown'] += 1

        block(vname, [
            ('frames', n),
            ('GT-ball frames / empty frames', f'{gt_frames} / {empty}'),
            ('raw detector recall (observed only)',
             f'{raw_hits / gt_frames:.4f}' if gt_frames else '-'),
            ('observed + recovered recall',
             f'{or_hits / gt_frames:.4f}' if gt_frames else '-'),
            ('temporal trajectory coverage (any state)',
             f'{cov / gt_frames:.4f}' if gt_frames else '-'),
            ('false recovery rate on GT-empty frames',
             f'{false_rec / empty:.4f}' if empty else '-'),
            ('hallucinated-ball frames', halluc),
            ('unknown-frame count', unk),
            ('states', dict(st)),
        ])
        report['temporal'][vname] = {
            'frames': n, 'gt_frames': gt_frames, 'empty_frames': empty,
            'raw_recall': round(raw_hits / gt_frames, 4) if gt_frames else None,
            'observed_plus_recovered_recall':
                round(or_hits / gt_frames, 4) if gt_frames else None,
            'trajectory_coverage': round(cov / gt_frames, 4) if gt_frames else None,
            'false_recovery_rate_on_empty':
                round(false_rec / empty, 4) if empty else None,
            'hallucinated_frames': halluc, 'unknown_frames': unk,
            'states': dict(st)}

    # longest run of consecutive hallucinated frames, within a window
    longest, longest_key = 0, None
    for key, fl in sorted(by_window.items()):
        cur = 0
        for f in fl:
            bad = len(gts[f['file']]) == 0 and outputs[f['file']] is not None
            cur = cur + 1 if bad else 0
            if cur > longest:
                longest, longest_key = cur, f"{key[0]} w{key[1]}"
    print(f'\nlongest hallucinated run: {longest} consecutive frames'
          + (f'  ({longest_key})' if longest_key else ''))
    print(f'camera cuts detected: {cut_count}')
    report['longest_hallucinated_run'] = longest
    report['longest_hallucinated_run_window'] = longest_key
    report['camera_cuts_detected'] = cut_count

    print('\nTRANSITION TABLE (all frames)')
    for k in ['raw miss -> recovered_low_conf', 'raw miss -> interpolated_short_gap',
              'raw miss -> unknown', 'raw miss -> observed']:
        print(f'  {k:<52}{trans.get(k, 0)}')
    for k in sorted(trans):
        if k.startswith('GT empty'):
            print(f'  {k:<52}{trans[k]}')
    report['transitions'] = dict(trans)

    print('\nPER WINDOW (temporal)')
    hdr = (f'  {"window":<30}{"GT":>4}{"empty":>7}{"obs":>5}{"rec":>5}{"interp":>8}'
           f'{"unk":>5}{"cov":>7}{"falseRec":>10}{"cuts":>6}')
    print(hdr); print('  ' + '-' * (len(hdr) - 2))
    per_window = {}
    for key, fl in sorted(by_window.items()):
        st = Counter(states[f['file']] for f in fl)
        gtf = sum(1 for f in fl if len(gts[f['file']]))
        emp = len(fl) - gtf
        cov = sum(1 for f in fl if len(gts[f['file']]) and hit(gts[f['file']], outputs[f['file']]))
        fr = sum(1 for f in fl if not len(gts[f['file']]) and outputs[f['file']] is not None)
        cts = sum(1 for f in fl if f.get('_cut'))
        name = f'{key[0][:22]} w{key[1]}'
        print(f'  {name:<30}{gtf:>4}{emp:>7}{st[OBSERVED]:>5}{st[RECOVERED]:>5}'
              f'{st[INTERPOLATED]:>8}{st[UNKNOWN]:>5}'
              f'{(cov/gtf if gtf else 0):>7.3f}{(fr/emp if emp else 0):>10.3f}{cts:>6}')
        per_window[name] = {'gt': gtf, 'empty': emp, 'observed': st[OBSERVED],
                            'recovered': st[RECOVERED], 'interpolated': st[INTERPOLATED],
                            'unknown': st[UNKNOWN], 'coverage': round(cov/gtf, 4) if gtf else None,
                            'false_recovery': round(fr/emp, 4) if emp else None,
                            'cuts': cts}
    report['temporal_per_window'] = per_window

    print('\nPER MATCH (temporal)')
    pm = defaultdict(lambda: Counter())
    for f in frames_meta:
        m = pm[f['match']]
        m[states[f['file']]] += 1
        m['gt'] += bool(len(gts[f['file']]))
        m['empty'] += not len(gts[f['file']])
        m['cov'] += bool(len(gts[f['file']]) and hit(gts[f['file']], outputs[f['file']]))
        m['false'] += bool(not len(gts[f['file']]) and outputs[f['file']] is not None)
    hdr = (f'  {"match":<32}{"GT":>4}{"empty":>7}{"obs":>5}{"rec":>5}{"interp":>8}'
           f'{"unk":>5}{"cov":>7}{"falseRec":>10}')
    print(hdr); print('  ' + '-' * (len(hdr) - 2))
    for k in sorted(pm):
        a = pm[k]
        print(f'  {k:<32}{a["gt"]:>4}{a["empty"]:>7}{a[OBSERVED]:>5}{a[RECOVERED]:>5}'
              f'{a[INTERPOLATED]:>8}{a[UNKNOWN]:>5}'
              f'{(a["cov"]/a["gt"] if a["gt"] else 0):>7.3f}'
              f'{(a["false"]/a["empty"] if a["empty"] else 0):>10.3f}')
    report['temporal_per_match'] = {k: dict(v) for k, v in pm.items()}

    Path(args.out).write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'\nwritten: {args.out}')


if __name__ == '__main__':
    main()
