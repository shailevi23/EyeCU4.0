#!/usr/bin/env python
"""
Calibrate the ball proposal generator against human-confirmed additions.

Choosing a confidence threshold by eye is guessing with extra steps. There is a
better reference available: 132 of the reviewed PP images hold 248 footballs a
human personally drew. Those are HUMAN_CONFIRMED_POSITIVES -- the strongest
evidence in this project -- so running the same generator over the same images
answers the only question that matters before 488 candidates go in front of a
reviewer: at each threshold, how many balls a human would have found does the
generator still propose?

WHAT THIS IS NOT. It is not detector recall. The reviewed images are
HUMAN_REVIEWED_PARTIAL, not gold: the existing blue annotations were never
exhaustively validated, and one of them turned out to be a player. Measuring
against them would be measuring against unverified data. So the denominator is
exactly the 248 human-drawn additions and nothing else:

    PROPOSAL RECALL ON HUMAN_CONFIRMED_ADDITIONS

A ball that neither the human nor the model found is invisible to this metric,
which is precisely why the residual full-frame QA stays separate and stays
mandatory.

    python tools/kb_ball_calibrate.py --run

Inference only. Nothing is written to the dataset, no verdict is recorded.
"""

import argparse
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'tools'))
import kb_ball_candidates as CAND                                 # noqa: E402
import kb_ball_pp_sweep_server as PP                              # noqa: E402
import kb_ball_qa_sample as kb_sample                             # noqa: E402
import kb_images                                                  # noqa: E402

PKG = REPO / 'experiments' / 'external_sources' / 'keremberke_review'
REPORT = PKG / 'BALL_CANDIDATE_CALIBRATION.json'
TIERS = PKG / 'BALL_CANDIDATE_TIERS.json'

THRESHOLDS = (0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25)
BUCKETS = ((5.0, '<=5px'), (8.0, '>5-8px'), (12.0, '>8-12px'),
           (float('inf'), '>12px'))
TARGET_RECALL = 0.95


def bucket(w):
    for lim, name in BUCKETS:
        if w <= lim:
            return name
    return '>12px'


def confirmed_additions():
    """The 248 human-drawn footballs, with the image they belong to.

    Read from the EFFECTIVE fold, so a re-answered image contributes only the
    boxes that currently stand. Superseded drafts are not confirmed anything.
    """
    q = json.loads(PP.QUEUE.read_text(encoding='utf-8'))
    rows = {r['IMAGE']: r for r in q['images']}
    out = []
    for im, v in PP.answers().items():
        if im not in rows or v.get('answer') != 'MISSING_BALL':
            continue
        for k, m in enumerate(v['missing']):
            out.append({'IMAGE': im, 'index': k, 'bbox_xywh': m['bbox_xywh'],
                        'width': m['bbox_xywh'][2],
                        'size_bucket': bucket(m['bbox_xywh'][2]),
                        'split': rows[im]['split'], 'run': rows[im]['run'],
                        'coco_image_id': rows[im]['coco_image_id'],
                        'img_w': rows[im]['img_w'], 'img_h': rows[im]['img_h']})
    return out


def reviewed_images():
    """Every reviewed PP image, positive or not -- the calibration frame."""
    q = json.loads(PP.QUEUE.read_text(encoding='utf-8'))
    rows = {r['IMAGE']: r for r in q['images']}
    ans = PP.answers()
    return [rows[im] for im in sorted(rows)
            if ans.get(im, {}).get('answer')]


def propose(rows, conf=min(THRESHOLDS)):
    """Run the generator once at the LOWEST threshold and keep every score.

    Running seven times would be seven times the compute for the same answer:
    a threshold is a filter on scores that are already known, so one pass at
    0.03 contains every higher threshold as a subset.
    """
    from ultralytics import YOLO

    gts = CAND.existing_ball_gt(rows)
    model = YOLO(str(CAND.WEIGHTS))
    per_image = defaultdict(list)
    stats = Counter()
    t0 = time.time()
    for n, r in enumerate(rows, 1):
        path = kb_images.resolve(r['IMAGE'])
        res = model.predict(source=str(path), imgsz=CAND.IMGSZ, conf=conf,
                            classes=[CAND.BALL_CLASS], verbose=False)[0]
        raw = []
        for box in res.boxes:
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
            raw.append({'bbox_xywh': [round(x1, 2), round(y1, 2),
                                      round(x2 - x1, 2), round(y2 - y1, 2)],
                        'conf': round(float(box.conf[0]), 4)})
        stats['raw'] += len(raw)
        for d in raw:
            ok, _ = CAND.plausible(d['bbox_xywh'])
            if not ok:
                stats['implausible'] += 1
                continue
            d['matched_gt'] = CAND.matches_gt(d['bbox_xywh'], gts[r['IMAGE']])
            if d['matched_gt']:
                stats['matched_existing_gt'] += 1
            per_image[r['IMAGE']].append(d)
        if n % 50 == 0:
            print(f'  {n}/{len(rows)} images, {time.time()-t0:.0f}s', flush=True)
    return per_image, stats


def evaluate(per_image, adds, threshold):
    """Coverage of the confirmed additions at one threshold.

    Matching uses CAND.matches_gt, the same rule the review tooling uses, with
    the human box in the GT position -- so "covered" here means exactly what
    "already covered" means in the queue builder.
    """
    covered, missed = [], []
    for a in adds:
        dets = [d for d in per_image.get(a['IMAGE'], [])
                if d['conf'] >= threshold]
        ref = [{'BOX_ID': 'human', 'bbox_xywh': a['bbox_xywh']}]
        hit = next((d for d in dets
                    if CAND.matches_gt(d['bbox_xywh'], ref)), None)
        (covered if hit else missed).append(
            {**a, 'matched_conf': hit['conf'] if hit else None})
    total = sum(1 for v in per_image.values() for d in v
                if d['conf'] >= threshold)
    unmatched = sum(1 for v in per_image.values() for d in v
                    if d['conf'] >= threshold and not d['matched_gt'])
    return {
        'threshold': threshold,
        'proposals_total': total,
        'proposals_unmatched_to_existing_gt': unmatched,
        'confirmed_covered': len(covered),
        'confirmed_missed': len(missed),
        'confirmed_recall': (len(covered) / len(adds)) if adds else 0.0,
        'recall_by_size': {
            name: {
                'n': sum(1 for a in adds if a['size_bucket'] == name),
                'covered': sum(1 for c in covered if c['size_bucket'] == name),
            } for _, name in BUCKETS},
        'missed_examples': [{'IMAGE': m['IMAGE'], 'bbox_xywh': m['bbox_xywh'],
                             'width': m['width']} for m in missed[:8]],
    }


def choose_tiers(rows, adds):
    """Tier boundaries from the measured curve, not from intuition.

    T1/T2 boundary: the lowest threshold that still holds the target recall,
    so tier 1 is 'everything the calibration says is load-bearing'.
    T2/T3 boundary: the floor, so tier 3 is the tail below which the generator
    contributed nothing measurable to confirmed-addition recall.
    """
    curve = rows
    at_target = [r for r in curve if r['confirmed_recall'] >= TARGET_RECALL]
    if at_target:
        t1 = max(r['threshold'] for r in at_target)
        feasible = True
    else:
        t1 = min(r['threshold'] for r in curve)
        feasible = False
    # the tail: highest threshold whose recall equals the floor's recall, i.e.
    # the region where lowering the bar bought nothing
    floor_recall = curve[0]['confirmed_recall']
    same = [r['threshold'] for r in curve if r['confirmed_recall'] >= floor_recall]
    t2 = max(same) if same else curve[0]['threshold']
    return t1, min(t2, t1), feasible


def apply_tiers(queue, t1, t2):
    out = {'TIER_1': [], 'TIER_2': [], 'TIER_3': []}
    for c in queue['candidates']:
        if c['conf'] >= t1:
            out['TIER_1'].append(c)
        elif c['conf'] >= t2:
            out['TIER_2'].append(c)
        else:
            out['TIER_3'].append(c)
    return out


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', action='store_true')
    ap.add_argument('--limit', type=int)
    args = ap.parse_args()
    if not args.run:
        if REPORT.is_file():
            print(REPORT.read_text(encoding='utf-8')[:4000])
        else:
            print('run with --run')
        return

    rows = reviewed_images()
    if args.limit:
        rows = rows[:args.limit]
    adds = [a for a in confirmed_additions()
            if a['IMAGE'] in {r['IMAGE'] for r in rows}]
    print(f'calibration frame: {len(rows)} reviewed PP images')
    print(f'human-confirmed additions: {len(adds)}')
    print(f'size mix: {dict(Counter(a["size_bucket"] for a in adds))}')
    print(f'\nweights {CAND.WEIGHTS.name}  imgsz {CAND.IMGSZ}  '
          f'floor {min(THRESHOLDS)}')
    print('INFERENCE ONLY. No training, no annotation modified.\n')

    per_image, stats = propose(rows)
    curve = [evaluate(per_image, adds, t) for t in THRESHOLDS]

    print(f'\n{"thr":>6}{"proposals":>11}{"unmatched":>11}{"covered":>9}'
          f'{"missed":>8}{"recall":>9}')
    for r in curve:
        print(f'{r["threshold"]:>6.3f}{r["proposals_total"]:>11}'
              f'{r["proposals_unmatched_to_existing_gt"]:>11}'
              f'{r["confirmed_covered"]:>9}{r["confirmed_missed"]:>8}'
              f'{100*r["confirmed_recall"]:>8.1f}%')

    print(f'\nrecall by size bucket')
    print(f'{"thr":>6}' + ''.join(f'{n:>12}' for _, n in BUCKETS))
    for r in curve:
        cells = []
        for _, n in BUCKETS:
            b = r['recall_by_size'][n]
            cells.append(f'{b["covered"]}/{b["n"]}' if b['n'] else '-')
        print(f'{r["threshold"]:>6.3f}' + ''.join(f'{c:>12}' for c in cells))

    t1, t2, feasible = choose_tiers(curve, adds)
    q = json.loads(CAND.CANDIDATES.read_text(encoding='utf-8'))
    tiers = apply_tiers(q, t1, t2)
    at_t1 = next(r for r in curve if r['threshold'] == t1)

    print(f'\ntier boundaries from the curve: T1 >= {t1}, T2 >= {t2}')
    print(f'target {100*TARGET_RECALL:.0f}% confirmed recall '
          f'{"reachable" if feasible else "NOT reachable above the floor"}')
    for k in ('TIER_1', 'TIER_2', 'TIER_3'):
        print(f'  {k}: {len(tiers[k])}')

    rep = {
        'calibration': 'PROPOSAL RECALL ON HUMAN_CONFIRMED_ADDITIONS',
        'not_detector_recall': (
            'the reviewed images are HUMAN_REVIEWED_PARTIAL, not gold. Existing '
            'blue GT was never exhaustively validated and one annotation proved '
            'to be a player, so the denominator is the 248 human-drawn '
            'additions only.'),
        'frame': {'reviewed_pp_images': len(rows),
                  'human_confirmed_additions': len(adds),
                  'size_mix': dict(Counter(a['size_bucket'] for a in adds))},
        'model': {'weights': CAND.WEIGHTS.name,
                  'weights_sha256': hashlib.sha256(
                      CAND.WEIGHTS.read_bytes()).hexdigest(),
                  'imgsz': CAND.IMGSZ, 'floor': min(THRESHOLDS)},
        'matching_rule': 'centre distance <= max(1.5 * box width, 8 px), the '
                         'same rule the review tooling uses',
        'stats': dict(stats),
        'curve': curve,
        'tiers': {'T1_min_conf': t1, 'T2_min_conf': t2,
                  'target_recall': TARGET_RECALL,
                  'target_reachable': feasible,
                  'expected_confirmed_recall_at_T1': at_t1['confirmed_recall'],
                  'counts': {k: len(v) for k, v in tiers.items()}},
        'residual_risk': (
            'candidates left unreviewed are UNRESOLVED, not negatives. This '
            'calibration says nothing about footballs neither the human nor '
            'the model found -- that is what the residual full-frame QA is '
            'for, and it stays mandatory.'),
        'generated_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    }
    REPORT.write_text(json.dumps(rep, indent=1) + '\n', encoding='utf-8')
    TIERS.write_text(json.dumps(
        {'T1_min_conf': t1, 'T2_min_conf': t2,
         'counts': {k: len(v) for k, v in tiers.items()},
         'tiers': {k: [c['candidate_id'] for c in v]
                   for k, v in tiers.items()}}, indent=1) + '\n',
        encoding='utf-8')
    print(f'\nwritten: {REPORT.relative_to(REPO)}')
    print(f'         {TIERS.relative_to(REPO)}')
    print('\nNo verdict was written and no annotation was modified.')


if __name__ == '__main__':
    main()
