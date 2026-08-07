#!/usr/bin/env python
"""
Assemble the Ultralytics YOLO dataset, splitting by MATCH (never by frame).

Adjacent video frames are near-identical, so a random frame split leaks the
training set into validation and produces meaningless scores. Every frame of a
match therefore lands in exactly one split (ROADMAP.md section 2).

Input layout (from extract_frames.py + your corrected labels):
    data/frames/<match_id>/<match_id>_000123.jpg
    data/labels/<match_id>/<match_id>_000123.txt

Output:
    data/dataset/{train,val,test}/{images,labels}/...
    data/dataset/football.yaml
    data/dataset/split_report.json

Examples:
    python tools/build_dataset.py --frames data/frames --labels data/labels
    python tools/build_dataset.py --zip          # also produce football_dataset.zip for Colab
"""

import argparse
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

CLASSES = ['player', 'goalkeeper', 'referee', 'ball']
IMG_EXTS = {'.jpg', '.jpeg', '.png'}


def collect(frames_root: Path, labels_root: Path):
    """Return {match_id: [(image_path, label_path), ...]} plus unlabelled images."""
    matches = defaultdict(list)
    unlabelled = []
    for img in sorted(p for p in frames_root.rglob('*') if p.suffix.lower() in IMG_EXTS):
        rel = img.relative_to(frames_root)
        match_id = rel.parts[0] if len(rel.parts) > 1 else img.stem.rsplit('_', 1)[0]
        label = (labels_root / rel).with_suffix('.txt')
        if not label.exists():
            unlabelled.append(img)
            continue
        matches[match_id].append((img, label))
    return matches, unlabelled


def read_counts(label_path: Path) -> Counter:
    counts = Counter()
    for line in label_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        cid = int(line.split()[0])
        if 0 <= cid < len(CLASSES):
            counts[CLASSES[cid]] += 1
        else:
            counts[f'INVALID_{cid}'] += 1
    return counts


def assign_splits(match_sizes: dict, ratios, seed: int):
    """Greedy: place the largest match into whichever split is furthest below quota."""
    total = sum(match_sizes.values())
    targets = {'train': total * ratios[0], 'val': total * ratios[1], 'test': total * ratios[2]}
    order = sorted(match_sizes, key=lambda m: (-match_sizes[m], m))

    rng = random.Random(seed)
    rng.shuffle(order)
    order.sort(key=lambda m: -match_sizes[m])

    assigned = {'train': [], 'val': [], 'test': []}
    current = {'train': 0, 'val': 0, 'test': 0}
    for match in order:
        split = max(targets, key=lambda s: targets[s] - current[s])
        assigned[split].append(match)
        current[split] += match_sizes[match]
    return assigned


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--frames', default='data/frames')
    p.add_argument('--labels', default='data/labels')
    p.add_argument('--out', default='data/dataset')
    p.add_argument('--ratios', default='0.70,0.15,0.15',
                   help='train,val,test fractions (default 0.70,0.15,0.15).')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--move', action='store_true', help='Move files instead of copying.')
    p.add_argument('--zip', action='store_true',
                   help='Also write football_dataset.zip next to --out (for Colab upload).')
    p.add_argument('--allow-unlabelled', action='store_true',
                   help='Skip unlabelled frames instead of aborting.')
    args = p.parse_args()

    frames_root, labels_root, out_root = Path(args.frames), Path(args.labels), Path(args.out)
    if not frames_root.exists():
        sys.exit(f'Frames not found: {frames_root}')
    if not labels_root.exists():
        sys.exit(f'Labels not found: {labels_root}')

    ratios = tuple(float(x) for x in args.ratios.split(','))
    if len(ratios) != 3 or abs(sum(ratios) - 1.0) > 1e-6:
        sys.exit('--ratios must be three fractions summing to 1.0')

    matches, unlabelled = collect(frames_root, labels_root)
    if unlabelled and not args.allow_unlabelled:
        sys.exit(f'{len(unlabelled)} frames have no label file '
                 f'(first: {unlabelled[0]}). Run pseudo_label.py, or pass '
                 f'--allow-unlabelled to drop them.')
    if not matches:
        sys.exit('No labelled frames found.')
    if len(matches) < 3:
        sys.exit(f'Only {len(matches)} match(es) found. A match-disjoint '
                 f'train/val/test split needs at least 3 (ROADMAP.md targets 4-6).')

    sizes = {m: len(v) for m, v in matches.items()}
    splits = assign_splits(sizes, ratios, args.seed)

    if out_root.exists():
        shutil.rmtree(out_root)
    transfer = shutil.move if args.move else shutil.copy2

    report = {'classes': CLASSES, 'splits': {}, 'match_sizes': sizes, 'totals': {}}
    grand = Counter()

    for split, match_ids in splits.items():
        img_dir = out_root / split / 'images'
        lbl_dir = out_root / split / 'labels'
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        counts = Counter()
        empty = 0
        n_images = 0
        for match_id in match_ids:
            for img, lbl in matches[match_id]:
                transfer(str(img), str(img_dir / img.name))
                transfer(str(lbl), str(lbl_dir / lbl.name))
                c = read_counts(lbl)
                if not c:
                    empty += 1
                counts.update(c)
                n_images += 1
        grand.update(counts)
        report['splits'][split] = {
            'matches': sorted(match_ids),
            'images': n_images,
            'share': round(n_images / sum(sizes.values()), 3),
            'empty_frames': empty,
            'instances': {c: counts.get(c, 0) for c in CLASSES},
        }

    report['totals'] = {c: grand.get(c, 0) for c in CLASSES}
    report['total_images'] = sum(sizes.values())

    # Match-disjointness is the whole point of this script -- verify it.
    seen = Counter(m for ids in splits.values() for m in ids)
    leaked = [m for m, n in seen.items() if n > 1]
    if leaked:
        sys.exit(f'BUG: matches in more than one split: {leaked}')

    yaml_path = out_root / 'football.yaml'
    yaml_path.write_text(
        '# EyeCU football detector dataset\n'
        '# Split by match -- no match appears in more than one split.\n'
        f'path: {out_root.resolve().as_posix()}\n'
        'train: train/images\n'
        'val: val/images\n'
        'test: test/images\n\n'
        'names:\n' + ''.join(f'  {i}: {c}\n' for i, c in enumerate(CLASSES))
    )
    (out_root / 'split_report.json').write_text(json.dumps(report, indent=2))

    print(f'\nDataset: {report["total_images"]} images, {len(matches)} matches -> {out_root}')
    for split in ('train', 'val', 'test'):
        s = report['splits'][split]
        print(f'  {split:<5} {s["images"]:>5} imgs ({s["share"]:.0%})  '
              f'matches={",".join(s["matches"])}')
        print(f'        instances: ' +
              '  '.join(f'{c}={s["instances"][c]}' for c in CLASSES))
    print(f'\n  totals: ' + '  '.join(f'{c}={report["totals"][c]}' for c in CLASSES))
    print(f'  config: {yaml_path}')

    for split in ('val', 'test'):
        for c in CLASSES:
            if report['splits'][split]['instances'][c] == 0:
                print(f'! {split} split has no `{c}` instances -- '
                      f'per-class metrics for it will be meaningless.')

    if args.zip:
        archive = out_root.parent / 'football_dataset'
        shutil.make_archive(str(archive), 'zip', root_dir=out_root)
        print(f'\n  zip: {archive}.zip  (upload this to Google Drive for Colab)')
        print('  NOTE: football.yaml `path:` is a local path -- the Colab '
              'notebook rewrites it after unzipping.')


if __name__ == '__main__':
    main()
