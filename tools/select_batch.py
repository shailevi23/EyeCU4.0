#!/usr/bin/env python
"""
Select the first annotation batch: ~400-500 diverse frames from TRAIN sources.

Val and test frames are deliberately excluded. Model-assisted iteration means
each round's model has seen the frames it helped label; letting that touch val
or test would quietly contaminate the only honest measurement available.

The split is recomputed with build_dataset.assign_splits using the same seed,
so "train" here means exactly what it will mean at build time.

Selection balances two things:
  * per-source quota, so no single match dominates the batch
  * within a source, spread over the clip plus image-statistic diversity
    (brightness for day/night, sharpness, and a dHash spread for camera angle
    and framing)

If pseudo-label drafts already exist, their content is used to prioritise the
frames that actually matter: crowded scenes, goalkeepers, referees and small
balls. Without drafts the tool falls back to diversity alone and says so.

Examples:
    python tools/select_batch.py --size 450
    python tools/select_batch.py --size 450 --out data/batches/batch_01.json
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_dataset import DEFAULT_EXCLUDE, assign_splits, infer_domain  # noqa: E402

IMG_EXTS = {'.jpg', '.jpeg', '.png'}

# Weights for the priority score when drafts are available. Rare classes are
# worth far more than another midfield frame with twelve players in it.
W_GOALKEEPER = 6.0
W_REFEREE = 6.0
W_SMALL_BALL = 5.0
W_BALL = 2.0
W_CROWD = 0.35          # per detected person, capped below
CROWD_CAP = 6.0
SMALL_BALL_AREA = 3e-4  # normalised area under which a ball counts as small


def imread_unicode(path: Path):
    try:
        return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        return None


def dhash(gray, size=8):
    small = cv2.resize(gray, (size + 1, size), interpolation=cv2.INTER_AREA)
    bits = 0
    for b in (small[:, 1:] > small[:, :-1]).flatten():
        bits = (bits << 1) | int(b)
    return bits


def hamming(a, b):
    return bin(a ^ b).count('1')


def image_stats(path: Path):
    img = imread_unicode(path)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (320, 180), interpolation=cv2.INTER_AREA)
    return {
        'brightness': float(small.mean()),
        'sharpness': float(cv2.Laplacian(small, cv2.CV_64F).var()),
        'hash': dhash(gray),
    }


def draft_priority(meta_path: Path):
    """Content-based score from an existing pseudo-label draft, or None."""
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
    except (json.JSONDecodeError, OSError):
        return None

    counts = Counter()
    score = 0.0
    for box in meta.get('boxes', []):
        cls = box.get('class')
        counts[cls] += 1
        if cls == 'goalkeeper':
            score += W_GOALKEEPER
        elif cls == 'referee':
            score += W_REFEREE
        elif cls == 'ball':
            area = float(box.get('w', 0)) * float(box.get('h', 0))
            score += W_SMALL_BALL if area <= SMALL_BALL_AREA else W_BALL

    people = counts['player'] + counts['goalkeeper'] + counts['referee']
    score += min(people * W_CROWD, CROWD_CAP)
    return {'score': score, 'counts': dict(counts), 'people': people}


def pick_diverse(candidates, quota):
    """
    Greedy: take the highest-priority frame, then repeatedly take the frame that
    is most different from everything already taken (perceptual hash distance),
    tie-broken by priority. Keeps the batch from being N views of one moment.
    """
    if quota >= len(candidates):
        return list(candidates)

    remaining = sorted(candidates, key=lambda c: -c['priority'])
    chosen = [remaining.pop(0)]
    while remaining and len(chosen) < quota:
        best_i, best_val = 0, None
        for i, cand in enumerate(remaining):
            if cand['hash'] is None:
                dist = 32
            else:
                dist = min((hamming(cand['hash'], c['hash'])
                            for c in chosen if c['hash'] is not None), default=64)
            val = dist + cand['priority']
            if best_val is None or val > best_val:
                best_i, best_val = i, val
        chosen.append(remaining.pop(best_i))
    return chosen


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--frames', default='data/frames')
    p.add_argument('--meta', default='data/pseudo_meta',
                   help='Pseudo-label metadata; used to prioritise if present.')
    p.add_argument('--out', default='data/batches/batch_01.json')
    p.add_argument('--size', type=int, default=450, help='Target batch size.')
    p.add_argument('--ratios', default='0.70,0.15,0.15')
    p.add_argument('--seed', type=int, default=42,
                   help='Must match build_dataset.py or the split will differ.')
    p.add_argument('--exclude', nargs='*', default=list(DEFAULT_EXCLUDE))
    p.add_argument('--max-per-source', type=int,
                   help='Hard cap per source (default: proportional quota).')
    args = p.parse_args()

    frames_root = Path(args.frames)
    if not frames_root.exists():
        sys.exit(f'Frames not found: {frames_root}')

    sources = {}
    for d in sorted(p for p in frames_root.iterdir() if p.is_dir()):
        imgs = sorted(q for q in d.iterdir() if q.suffix.lower() in IMG_EXTS)
        if imgs and d.name not in args.exclude:
            sources[d.name] = imgs
    if not sources:
        sys.exit('No usable sources.')

    ratios = tuple(float(x) for x in args.ratios.split(','))
    sizes = {k: len(v) for k, v in sources.items()}
    domains = {k: infer_domain(k) for k in sources}
    splits = assign_splits(sizes, domains, ratios, args.seed)
    train_sources = sorted(splits['train'])

    print(f'TRAIN sources ({len(train_sources)}): '
          f'{sum(sizes[s] for s in train_sources)} frames available')
    print(f'held out (not eligible): val={sorted(splits["val"])} '
          f'test={sorted(splits["test"])}\n')

    # Proportional quota per source, so the batch mirrors the training pool.
    train_total = sum(sizes[s] for s in train_sources)
    target = min(args.size, train_total)
    quotas = {s: max(1, round(target * sizes[s] / train_total)) for s in train_sources}
    if args.max_per_source:
        quotas = {s: min(q, args.max_per_source) for s, q in quotas.items()}

    meta_root = Path(args.meta)
    used_drafts = False
    selected, per_source_counts, class_totals = [], Counter(), Counter()

    for src in train_sources:
        cands = []
        for img in sources[src]:
            rel = img.relative_to(frames_root).as_posix()
            stats = image_stats(img)
            prio = draft_priority((meta_root / rel).with_suffix('.json'))
            if prio:
                used_drafts = True
                score = prio['score']
            else:
                # No drafts: prefer sharper, mid-exposure frames, and let the
                # diversity step do the real work.
                score = 0.0
                if stats:
                    score += min(stats['sharpness'] / 200.0, 2.0)
                    score += 1.0 - abs(stats['brightness'] - 128) / 128.0
            cands.append({
                'rel': rel, 'source': src, 'priority': score,
                'hash': stats['hash'] if stats else None,
                'brightness': round(stats['brightness'], 1) if stats else None,
                'counts': prio['counts'] if prio else None,
            })

        chosen = pick_diverse(cands, quotas[src])
        selected.extend(chosen)
        per_source_counts[src] = len(chosen)
        for c in chosen:
            if c['counts']:
                class_totals.update(c['counts'])

    selected.sort(key=lambda c: c['rel'])

    print(f'{"source":<40}{"picked":>8}{"of":>7}')
    print('-' * 55)
    for src in train_sources:
        print(f'{src:<40}{per_source_counts[src]:>8}{sizes[src]:>7}')
    print('-' * 55)
    print(f'{"TOTAL":<40}{len(selected):>8}{train_total:>7}')

    if used_drafts:
        print('\nprioritised using existing pseudo-label drafts')
        print('  draft instances in batch: ' + '  '.join(
            f'{k}={v}' for k, v in sorted(class_totals.items())))
    else:
        print('\nNOTE: no pseudo-label drafts found, so the batch was selected on '
              'image diversity alone\n(spread over each clip, brightness, '
              'sharpness, perceptual-hash distance).\nRun pseudo_label.py first '
              'and re-run this to prioritise goalkeepers, referees and small balls.')

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        'created_from': str(frames_root, encoding='utf-8'),
        'target_size': args.size,
        'seed': args.seed,
        'split_ratios': list(ratios),
        'train_sources': train_sources,
        'held_out': {'val': sorted(splits['val']), 'test': sorted(splits['test'])},
        'prioritised_by_drafts': used_drafts,
        'frames_per_source': dict(per_source_counts),
        'images': [c['rel'] for c in selected],
    }, indent=2), encoding='utf-8')
    print(f'\nbatch: {out}  ({len(selected)} images)')
    print(f'\nNext:\n'
          f'  python tools/pseudo_label.py --batch {out}')


if __name__ == '__main__':
    main()
