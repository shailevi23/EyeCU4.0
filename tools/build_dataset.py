#!/usr/bin/env python
"""
Assemble the Ultralytics YOLO dataset, splitting by MATCH (never by frame).

Adjacent video frames are near-identical, so a random frame split leaks the
training set into validation and produces meaningless scores. Every frame of a
match therefore lands in exactly one split.

The split is also stratified by DOMAIN (pro / women / youth / amateur), so each
of train/val/test sees a spread of footage types. Without that, a 15% test set
drawn by size alone can easily end up all-youth or all-pro, and the reported
accuracy then measures something other than intended use.

Input layout (from extract_frames.py + your corrected labels):
    data/frames/<match_id>/<match_id>_000123.jpg
    data/labels/<match_id>/<match_id>_000123.txt

Output (Ultralytics layout):
    data/dataset/images/{train,val,test}/...
    data/dataset/labels/{train,val,test}/...
    data/dataset/football.yaml
    data/dataset/split_report.json

Examples:
    python tools/build_dataset.py --plan-only          # show the split, write nothing
    python tools/build_dataset.py --zip
    python tools/build_dataset.py --check              # validate an existing dataset
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
SPLITS = ('train', 'val', 'test')

# Repo sample clips: also the fixtures the regression tests run against, so they
# are kept out of training to keep those tests an independent check.
DEFAULT_EXCLUDE = ('short', '08fd33_4')


def infer_domain(match_id: str) -> str:
    """
    Coarse footage type from the source name. Heuristic and overridable with
    --domains; the chosen classification is always printed so it can be checked.
    """
    name = match_id.lower()
    if name.startswith('women') or '_women' in name:
        return 'women'
    if name.startswith('youth') or 'youth' in name:
        return 'youth'
    if 'sunday_league' in name or 'amateur' in name:
        return 'amateur'
    return 'pro'


def collect(frames_root: Path, labels_root: Path, require_labels: bool):
    """{match_id: [(image, label_or_None)]} plus a list of unlabelled images."""
    matches = defaultdict(list)
    unlabelled = []
    for img in sorted(p for p in frames_root.rglob('*') if p.suffix.lower() in IMG_EXTS):
        rel = img.relative_to(frames_root)
        match_id = rel.parts[0] if len(rel.parts) > 1 else img.stem.rsplit('_', 1)[0]
        label = (labels_root / rel).with_suffix('.txt')
        if not label.exists():
            unlabelled.append(img)
            if require_labels:
                continue
            label = None
        matches[match_id].append((img, label))
    return matches, unlabelled


def read_counts(label_path: Path) -> Counter:
    counts = Counter()
    if label_path is None or not label_path.exists():
        return counts
    for line in label_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        cid = int(float(line.split()[0]))
        counts[CLASSES[cid] if 0 <= cid < len(CLASSES) else f'INVALID_{cid}'] += 1
    return counts


def assign_splits(match_sizes: dict, domains: dict, ratios, seed: int):
    """
    Greedy, domain-stratified, match-disjoint.

    Each domain is distributed across train/val/test independently, so every
    split gets a mix. Within a domain the largest match goes to whichever split
    is furthest below its quota.
    """
    assigned = {s: [] for s in SPLITS}
    by_domain = defaultdict(list)
    for match_id, size in match_sizes.items():
        by_domain[domains[match_id]].append(match_id)

    rng = random.Random(seed)
    for domain in sorted(by_domain):
        members = by_domain[domain]
        rng.shuffle(members)
        members.sort(key=lambda m: -match_sizes[m])

        total = sum(match_sizes[m] for m in members)
        target = dict(zip(SPLITS, (total * r for r in ratios)))
        current = {s: 0 for s in SPLITS}

        for match_id in members:
            # Only consider splits that can still be reached; with few matches
            # per domain this keeps val/test from being starved entirely.
            split = max(SPLITS, key=lambda s: target[s] - current[s])
            assigned[split].append(match_id)
            current[split] += match_sizes[match_id]

    return assigned


def check_leakage(dataset_root: Path) -> list:
    """Fail loudly if any image or match appears in more than one split."""
    problems = []
    stems, sources = {}, {}
    for split in SPLITS:
        img_dir = dataset_root / 'images' / split
        if not img_dir.is_dir():
            problems.append(f'MISSING SPLIT DIRECTORY: {img_dir}')
            continue
        for img in img_dir.iterdir():
            if img.suffix.lower() not in IMG_EXTS:
                continue
            if img.stem in stems:
                problems.append(
                    f'IMAGE LEAK: {img.name} in both {stems[img.stem]} and {split}')
            stems[img.stem] = split

            source = img.stem.rsplit('_', 1)[0]
            if source in sources and sources[source] != split:
                problems.append(
                    f'MATCH LEAK: source {source!r} in both '
                    f'{sources[source]} and {split}')
            sources[source] = split

            label = dataset_root / 'labels' / split / f'{img.stem}.txt'
            if not label.exists():
                problems.append(f'MISSING LABEL: {split}/{img.name}')
    return problems


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--frames', default='data/frames')
    p.add_argument('--labels', default='data/labels')
    p.add_argument('--out', default='data/dataset')
    p.add_argument('--ratios', default='0.70,0.15,0.15')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--exclude', nargs='*', default=list(DEFAULT_EXCLUDE),
                   help=f'Match ids to leave out entirely (default: {" ".join(DEFAULT_EXCLUDE)}).')
    p.add_argument('--domains', help='JSON file mapping match_id -> domain, '
                                     'overriding the built-in heuristic.')
    p.add_argument('--plan-only', action='store_true',
                   help='Print the split plan and exit without writing files.')
    p.add_argument('--check', action='store_true',
                   help='Validate an existing --out dataset for leakage; '
                        'exits non-zero if any is found.')
    p.add_argument('--move', action='store_true', help='Move instead of copy.')
    p.add_argument('--zip', action='store_true', help='Also write football_dataset.zip.')
    p.add_argument('--allow-unlabelled', action='store_true',
                   help='Skip unlabelled frames instead of aborting.')
    args = p.parse_args()

    out_root = Path(args.out)

    # --- validation-only mode ------------------------------------------------
    if args.check:
        if not out_root.exists():
            sys.exit(f'Nothing to check: {out_root} does not exist.')
        problems = check_leakage(out_root)
        if problems:
            print(f'{len(problems)} problem(s):')
            for prob in problems[:40]:
                print(f'  ! {prob}')
            if len(problems) > 40:
                print(f'  ... and {len(problems) - 40} more')
            sys.exit(1)
        print(f'{out_root}: no leakage. Every image and match is in exactly one split.')
        return

    frames_root, labels_root = Path(args.frames), Path(args.labels)
    if not frames_root.exists():
        sys.exit(f'Frames not found: {frames_root}')

    ratios = tuple(float(x) for x in args.ratios.split(','))
    if len(ratios) != 3 or abs(sum(ratios) - 1.0) > 1e-6:
        sys.exit('--ratios must be three fractions summing to 1.0')

    require_labels = not (args.plan_only or args.allow_unlabelled)
    matches, unlabelled = collect(frames_root, labels_root, require_labels)

    for match_id in args.exclude:
        if matches.pop(match_id, None) is not None:
            print(f'excluded: {match_id}')

    if not matches:
        sys.exit('No frames found (after exclusions).')

    if unlabelled and not args.plan_only and not args.allow_unlabelled:
        sys.exit(f'{len(unlabelled)} frames have no label file '
                 f'(first: {unlabelled[0]}). Label them, or use --plan-only to '
                 f'preview the split, or --allow-unlabelled to drop them.')

    domains = {m: infer_domain(m) for m in matches}
    if args.domains:
        domains.update(json.loads(Path(args.domains).read_text()))

    sizes = {m: len(v) for m, v in matches.items()}
    if len(matches) < 3:
        sys.exit(f'Only {len(matches)} match(es); a match-disjoint split needs 3+.')

    splits = assign_splits(sizes, domains, ratios, args.seed)

    # --- report --------------------------------------------------------------
    total = sum(sizes.values())
    print(f'\n{total} frames, {len(matches)} matches, '
          f'{len(set(domains.values()))} domains\n')
    domain_matrix = defaultdict(Counter)
    for split, ids in splits.items():
        for m in ids:
            domain_matrix[split][domains[m]] += sizes[m]

    for split in SPLITS:
        n = sum(sizes[m] for m in splits[split])
        print(f'{split:<6} {n:>5} frames ({n / total:>5.1%})  '
              f'{len(splits[split])} matches')
        print(f'       domains: ' + ', '.join(
            f'{d}={c}' for d, c in sorted(domain_matrix[split].items())))
        for m in sorted(splits[split]):
            print(f'         - {m} ({sizes[m]}, {domains[m]})')

    missing_domains = {
        s: sorted(set(domains.values()) - set(domain_matrix[s]))
        for s in SPLITS
    }
    for split, missing in missing_domains.items():
        if missing:
            print(f'! {split} contains no {missing} footage')

    if args.plan_only:
        print('\n--plan-only: nothing written.')
        kept = {img for pairs in matches.values() for img, _ in pairs}
        pending = sum(1 for img in unlabelled if img in kept)
        if pending:
            print(f'{pending} of {total} frames in this plan are still unlabelled.')
        return

    # --- write ---------------------------------------------------------------
    if out_root.exists():
        shutil.rmtree(out_root)
    transfer = shutil.move if args.move else shutil.copy2

    report = {'classes': CLASSES, 'ratios': list(ratios), 'seed': args.seed,
              'excluded': args.exclude, 'domains': domains,
              'match_sizes': sizes, 'splits': {}}
    grand = Counter()

    for split in SPLITS:
        img_dir = out_root / 'images' / split
        lbl_dir = out_root / 'labels' / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        counts, empty, n_images = Counter(), 0, 0
        for match_id in splits[split]:
            for img, lbl in matches[match_id]:
                transfer(str(img), str(img_dir / img.name))
                if lbl is not None:
                    transfer(str(lbl), str(lbl_dir / lbl.name))
                    c = read_counts(lbl)
                    if not c:
                        empty += 1
                    counts.update(c)
                n_images += 1
        grand.update(counts)
        report['splits'][split] = {
            'matches': sorted(splits[split]),
            'domains': dict(domain_matrix[split]),
            'images': n_images,
            'share': round(n_images / total, 3),
            'empty_label_files': empty,
            'instances': {c: counts.get(c, 0) for c in CLASSES},
        }

    report['totals'] = {c: grand.get(c, 0) for c in CLASSES}
    report['total_images'] = total

    yaml_path = out_root / 'football.yaml'
    yaml_path.write_text(
        '# EyeCU football detector dataset\n'
        '# Split by match and stratified by domain -- no match appears in\n'
        '# more than one split.\n'
        f'path: {out_root.resolve().as_posix()}\n'
        'train: images/train\n'
        'val: images/val\n'
        'test: images/test\n\n'
        'names:\n' + ''.join(f'  {i}: {c}\n' for i, c in enumerate(CLASSES))
    )
    (out_root / 'split_report.json').write_text(json.dumps(report, indent=2))

    problems = check_leakage(out_root)
    if problems:
        print('\nLEAKAGE DETECTED after writing:')
        for prob in problems[:20]:
            print(f'  ! {prob}')
        sys.exit(1)
    print('\nLeakage check: passed.')

    print(f'\n  totals: ' + '  '.join(f'{c}={report["totals"][c]}' for c in CLASSES))
    print(f'  config: {yaml_path}')

    for split in ('val', 'test'):
        for c in CLASSES:
            if report['splits'][split]['instances'][c] == 0:
                print(f'! {split} has no `{c}` instances -- '
                      f'its per-class metrics will be meaningless.')

    if args.zip:
        archive = out_root.parent / 'football_dataset'
        shutil.make_archive(str(archive), 'zip', root_dir=out_root)
        print(f'\n  zip: {archive}.zip  (upload to Google Drive for Colab)')
        print('  NOTE: football.yaml `path:` is local; the Colab notebook '
              'rewrites it after unzipping.')


if __name__ == '__main__':
    main()
