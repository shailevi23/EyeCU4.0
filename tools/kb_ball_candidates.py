#!/usr/bin/env python
"""
High-recall ball proposals for the PP images no human has swept yet.

The exhaustive census was the right instrument while the missing-ball rate was
unknown; 256 of 357 images in, it is known well enough that spending the
remaining human hours on frames the model can pre-filter is the worse trade. So
the model proposes and the human disposes -- on candidates only.

WHAT A PROPOSAL IS AND IS NOT. Nothing here becomes an annotation. The output is
a queue of unmatched detections for a human to answer Y/N/U, and only a Y
produces a HUMAN_APPROVED_ADDITION. That is the same rule the role repair ran
under, and the reason it is not negotiable is written in this project's own
history: the role triage answered 4,153 of 4,153 candidates and a QA of what it
REJECTED still found 6.40% missed. A completed candidate queue is evidence about
the candidates, never about the images.

THE CIRCULARITY IS REAL AND IS HANDLED SEPARATELY. This detector is weakest on
small balls, which is why keremberke was wanted in the first place. A ball it
misses never enters the queue, so candidate review alone cannot certify the
remaining images. That is what the 40-image residual QA sample exists for --
drawn from images where this generator found NOTHING, which is precisely where
its blind spot lives.

MATCHING. Centre distance, not IoU. At 4-8 px a one-pixel offset moves IoU by
tens of points, so IoU would report "unmatched" for detections that are plainly
the same object as an existing annotation, and the queue would fill with
duplicates of GT the reviewer already trusts.

    python tools/kb_ball_candidates.py --generate     # runs the detector
    python tools/kb_ball_candidates.py --summary      # read the stored queue

The immutable export is read-only here. Nothing in this module writes to it.
"""

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'tools'))
import kb_ball_pp_sweep_server as PP                              # noqa: E402
import kb_ball_qa_sample as kb_sample                             # noqa: E402
import kb_images                                                  # noqa: E402

PKG = REPO / 'experiments' / 'external_sources' / 'keremberke_review'
CANDIDATES = PKG / 'BALL_CANDIDATE_QUEUE.json'
RESIDUAL = PKG / 'BALL_RESIDUAL_QA_SAMPLE.json'

WEIGHTS = REPO / 'best_A_960.pt'
BALL_CLASS = 3
CONF = 0.03                 # deliberately far below any operating threshold
IMGSZ = 1280
# A football on a 1280x720 broadcast frame measured 3-20 px across the 1,267
# annotations already in the dataset (p50 9.1, p95 48.4). The wide bound is
# generous rather than tight: rejecting a real ball costs recall, which is the
# one thing this pass exists to protect.
MIN_W, MAX_W = 2.0, 60.0
MIN_AR, MAX_AR = 0.35, 2.8
RESIDUAL_N = 40
RESIDUAL_SEED = 20260814


def unresolved_images(decisions: Path = None):
    """PP queue images with no effective human answer yet.

    Anything already answered stays answered: this pass must not put a frame
    back in front of the reviewer, and must not let a model proposal compete
    with a human answer already on record.
    """
    q = json.loads(PP.QUEUE.read_text(encoding='utf-8'))
    ans = PP.answers(decisions or PP.DECISIONS)
    return [r for r in q['images']
            if not ans.get(r['IMAGE'], {}).get('answer')]


def existing_ball_gt(images):
    """{IMAGE: [{BOX_ID, bbox}]} for the given queue rows."""
    cache, out = {}, {}
    for r in images:
        split = r['split']
        if split not in cache:
            doc = json.loads((kb_sample.EXPORT /
                              f'{split}_annotations.coco.json')
                             .read_text(encoding='utf-8'))
            bc = kb_sample._ball_category(doc['categories'])
            per = {}
            for a in doc['annotations']:
                if a['category_id'] == bc:
                    per.setdefault(a['image_id'], []).append(a)
            cache[split] = per
        out[r['IMAGE']] = [
            {'BOX_ID': f'{split}:{a["id"]}', 'annotation_id': a['id'],
             'bbox_xywh': [float(v) for v in a['bbox']]}
            for a in cache[split].get(r['coco_image_id'], [])]
    return out


def _centre(b):
    return b[0] + b[2] / 2.0, b[1] + b[3] / 2.0


def matches_gt(cand, gts):
    """Is this detection the same object as an annotation already present?

    Centre distance within max(1.5 x GT width, 8 px). The floor matters: for a
    4 px annotation, 1.5x width is 6 px, and a detection 7 px away is far more
    likely to be the same ball seen slightly differently than a second ball
    that close to the first.
    """
    cx, cy = _centre(cand)
    for g in gts:
        gx, gy = _centre(g['bbox_xywh'])
        tol = max(1.5 * g['bbox_xywh'][2], 8.0)
        if ((cx - gx) ** 2 + (cy - gy) ** 2) ** 0.5 <= tol:
            return g['BOX_ID']
    return None


def plausible(b):
    w, h = b[2], b[3]
    if not (MIN_W <= w <= MAX_W):
        return False, f'width {w:.1f} outside {MIN_W}-{MAX_W}'
    if h <= 0:
        return False, 'non-positive height'
    ar = w / h
    if not (MIN_AR <= ar <= MAX_AR):
        return False, f'aspect {ar:.2f} outside {MIN_AR}-{MAX_AR}'
    return True, None


def generate(limit=None):
    """Run the detector over unresolved PP images. Returns the queue dict."""
    from ultralytics import YOLO                       # imported only here

    rows = unresolved_images()
    if limit:
        rows = rows[:limit]
    gts = existing_ball_gt(rows)
    model = YOLO(str(WEIGHTS))

    cands, per_image, stats = [], {}, Counter()
    t0 = time.time()
    for n, r in enumerate(rows, 1):
        path = kb_images.resolve(r['IMAGE'])
        res = model.predict(source=str(path), imgsz=IMGSZ, conf=CONF,
                            classes=[BALL_CLASS], verbose=False)[0]
        raw = []
        for box in res.boxes:
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
            raw.append({'bbox_xywh': [round(x1, 2), round(y1, 2),
                                      round(x2 - x1, 2), round(y2 - y1, 2)],
                        'conf': round(float(box.conf[0]), 4)})
        stats['raw_detections'] += len(raw)
        kept = []
        for d in raw:
            ok, why = plausible(d['bbox_xywh'])
            if not ok:
                stats['rejected_implausible'] += 1
                continue
            m = matches_gt(d['bbox_xywh'], gts[r['IMAGE']])
            if m:
                stats['matched_existing_gt'] += 1
                continue
            kept.append(d)
        # de-duplicate near-identical detections of the same object
        kept.sort(key=lambda d: -d['conf'])
        uniq = []
        for d in kept:
            cx, cy = _centre(d['bbox_xywh'])
            if any(((cx - _centre(u['bbox_xywh'])[0]) ** 2 +
                    (cy - _centre(u['bbox_xywh'])[1]) ** 2) ** 0.5
                   <= max(1.5 * u['bbox_xywh'][2], 8.0) for u in uniq):
                stats['deduplicated'] += 1
                continue
            uniq.append(d)
        per_image[r['IMAGE']] = len(uniq)
        for k, d in enumerate(uniq):
            cands.append({
                'candidate_id': candidate_id(r['IMAGE'], d['bbox_xywh']),
                'IMAGE': r['IMAGE'], 'split': r['split'], 'run': r['run'],
                'bbox_xywh': d['bbox_xywh'], 'conf': d['conf'],
                'index_in_image': k,
                'img_w': r['img_w'], 'img_h': r['img_h'],
                'existing_ball_gt': len(gts[r['IMAGE']]),
                'geometry_author': 'MODEL PROPOSAL -- not an annotation',
            })
        if n % 25 == 0:
            print(f'  {n}/{len(rows)} images, {len(cands)} candidates, '
                  f'{time.time()-t0:.0f}s', flush=True)

    cands.sort(key=lambda c: -c['conf'])
    zero = [im for im, k in per_image.items() if k == 0]
    return {
        'queue': 'ball_candidates',
        'purpose': ('model proposals for the unresolved PP images; a proposal '
                    'is never an annotation'),
        'model': {
            'weights': WEIGHTS.name,
            'weights_sha256': hashlib.sha256(WEIGHTS.read_bytes()).hexdigest(),
            'class': 'ball', 'conf_floor': CONF, 'imgsz': IMGSZ,
            'note': ('high recall on purpose. This detector is weakest on small '
                     'balls, which is why this dataset was acquired, so a ball '
                     'it misses never appears here -- see the residual QA'),
        },
        'matching': {
            'rule': 'centre distance <= max(1.5 * GT width, 8 px)',
            'why_not_iou': ('at 4-8 px a one-pixel offset moves IoU by tens of '
                            'points, so IoU would call true duplicates unmatched'),
        },
        'plausibility': {'width_px': [MIN_W, MAX_W],
                         'aspect_ratio': [MIN_AR, MAX_AR]},
        'population': {
            'unresolved_pp_images': len(rows),
            **kb_sample.population_fingerprint(),
        },
        'stats': dict(stats),
        'images_with_candidates': sum(1 for v in per_image.values() if v),
        'images_with_zero_candidates': len(zero),
        'zero_candidate_images': sorted(zero),
        'n_candidates': len(cands),
        'candidates': cands,
        'no_candidate_is_ground_truth': True,
        'generated_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    }


def candidate_id(image, bbox):
    key = f'{image}|' + ','.join(f'{float(v):.2f}' for v in bbox)
    return 'BALLCAND:' + hashlib.sha256(key.encode('utf-8')).hexdigest()[:16]


def residual_sample(queue, n=RESIDUAL_N, seed=RESIDUAL_SEED):
    """Deterministic sample for the residual QA.

    The intended frame was images the generator proposed NOTHING in, which is
    exactly where its blind spot lives. At conf 0.03 that population is 2 images
    out of 101 -- the floor is permissive enough that almost every frame gets
    some proposal, so there is nearly nothing left to sample.

    Raising the threshold would manufacture a bigger pool, and that would be the
    wrong move twice over: it would discard real recall from the correction
    queue purely to create something to measure, and the resulting "zero
    candidate" images would be zero-at-a-threshold-chosen-for-this-purpose
    rather than genuinely unproposed. The measurement would be of an artefact.

    So the frame widens honestly instead. Priority order:

      1. every zero-candidate image (the true blind-spot population)
      2. topped up at random from images whose candidates are ALL weak
         (max conf < 0.15), where the generator is closest to saying nothing

    Stratum membership is recorded per image so the analysis can report them
    separately and never average a weak claim into a strong one.
    """
    import random
    zero = sorted(queue['zero_candidate_images'])
    by_img = {}
    for c in queue['candidates']:
        by_img.setdefault(c['IMAGE'], []).append(c['conf'])
    weak = sorted(im for im, cs in by_img.items() if max(cs) < 0.15)
    rnd = random.Random(seed)
    take = [{'IMAGE': im, 'stratum': 'ZERO_CANDIDATE'} for im in zero]
    need = max(0, n - len(take))
    for im in rnd.sample(weak, min(need, len(weak))):
        take.append({'IMAGE': im, 'stratum': 'WEAK_CANDIDATES_ONLY'})
    take.sort(key=lambda r: r['IMAGE'])
    return {
        'queue': 'ball_residual_qa',
        'purpose': ('measure what the candidate generator missed, by human '
                    'full-frame review of images it had little or nothing to '
                    'say about'),
        'design': ('census of the zero-candidate stratum, plus a seeded random '
                   'sample of the weak-candidate stratum'),
        'seed': seed,
        'requested_n': n,
        'strata': {
            'ZERO_CANDIDATE': {
                'N': len(zero), 'n': len(zero),
                'meaning': 'the generator proposed nothing at all here'},
            'WEAK_CANDIDATES_ONLY': {
                'N': len(weak),
                'n': sum(1 for r in take
                         if r['stratum'] == 'WEAK_CANDIDATES_ONLY'),
                'meaning': 'every proposal in this image scored below 0.15'},
        },
        'why_not_pure_zero_candidate': (
            f'only {len(zero)} of {queue["population"]["unresolved_pp_images"]} '
            'images have zero candidates at conf 0.03. Raising the floor to '
            'enlarge that pool would discard recall from the correction queue '
            'to manufacture something to measure, so the frame was widened to '
            'the weakest-evidence images instead. Report the two strata '
            'separately.'),
        'n': len(take),
        'images': take,
        'note': ('review as a full-frame sweep, exactly as Round 0 did. A '
                 'positive here is a football the MODEL missed.'),
    }


def _summary(q):
    print(f'model      {q["model"]["weights"]} @ conf {q["model"]["conf_floor"]}'
          f'  imgsz {q["model"]["imgsz"]}')
    print(f'images     {q["population"]["unresolved_pp_images"]} unresolved PP')
    s = q['stats']
    print(f'\nraw detections            {s.get("raw_detections", 0):>6}')
    print(f'  rejected implausible    {s.get("rejected_implausible", 0):>6}')
    print(f'  matched existing GT     {s.get("matched_existing_gt", 0):>6}')
    print(f'  deduplicated            {s.get("deduplicated", 0):>6}')
    print(f'  UNMATCHED -> queue      {q["n_candidates"]:>6}')
    print(f'\nimages with candidates    {q["images_with_candidates"]:>6}')
    print(f'images with none          {q["images_with_zero_candidates"]:>6}'
          f'   <- residual QA is drawn from here')
    if q['candidates']:
        c = [x['conf'] for x in q['candidates']]
        w = [x['bbox_xywh'][2] for x in q['candidates']]
        import statistics as st
        print(f'\ncandidate conf   median {st.median(c):.3f}  max {max(c):.3f}')
        print(f'candidate width  median {st.median(w):.1f} px  '
              f'<=8px {sum(1 for x in w if x <= 8)}')
        print(f'per run: {dict(Counter(x["run"] for x in q["candidates"]))}')


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--generate', action='store_true',
                    help='run the detector and write the candidate queue')
    ap.add_argument('--summary', action='store_true')
    ap.add_argument('--limit', type=int, help='first N images only (smoke test)')
    args = ap.parse_args()

    if args.generate:
        rows = unresolved_images()
        print(f'unresolved PP images: {len(rows)}')
        print(f'weights: {WEIGHTS.name}  conf {CONF}  imgsz {IMGSZ}')
        print('INFERENCE ONLY. No training, no weight is written, no '
              'annotation is modified.\n')
        q = generate(args.limit)
        CANDIDATES.write_text(json.dumps(q, indent=1) + '\n', encoding='utf-8')
        r = residual_sample(q)
        RESIDUAL.write_text(json.dumps(r, indent=1) + '\n', encoding='utf-8')
        print()
        _summary(q)
        print(f'\nresidual QA sample: {r["n"]} of {r["N_zero_candidate_images"]}'
              f' zero-candidate images, seed {r["seed"]}')
        print(f'\nwritten: {CANDIDATES.relative_to(REPO)}')
        print(f'         {RESIDUAL.relative_to(REPO)}')
        print('\nNo candidate is ground truth. Only a human YES creates an '
              'annotation.')
        return

    if not CANDIDATES.is_file():
        print(f'no candidate queue at {CANDIDATES}\n'
              f'run: python tools/kb_ball_candidates.py --generate')
        sys.exit(1)
    _summary(json.loads(CANDIDATES.read_text(encoding='utf-8')))


if __name__ == '__main__':
    main()
