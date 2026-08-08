#!/usr/bin/env python
"""
Import a corrected Roboflow (or CVAT) YOLO export back into the repo layout.

Two things make this necessary:

1. Roboflow renames files on export, e.g.
       youth_3_000615.jpg  ->  youth_3_000615_jpg.rf.9f2c....jpg
   The original stem has to be recovered or the frame cannot be traced to its
   source match.

2. Roboflow assigns its own random train/valid/test split. Frames from the same
   match land in different splits, which leaks near-identical images across the
   boundary and inflates every score. This tool deliberately DISCARDS their
   split and restores the flat per-source layout, so tools/build_dataset.py can
   redo the match-disjoint split.

It also verifies the exported class order still matches this project's, because
a reordered data.yaml silently relabels the whole dataset.

Examples:
    python tools/import_roboflow.py --export ~/Downloads/football.v1i.yolov8.zip
    python tools/import_roboflow.py --export path/to/unzipped/folder --dry-run
"""

import argparse
import json
import shutil
import sys
import tempfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

CLASSES = ['player', 'goalkeeper', 'referee', 'ball']
IMG_EXTS = {'.jpg', '.jpeg', '.png'}


def unpack(export: Path) -> tuple[Path, tempfile.TemporaryDirectory | None]:
    if export.is_dir():
        return export, None
    tmp = tempfile.TemporaryDirectory()
    with zipfile.ZipFile(export) as z:
        z.extractall(tmp.name)
    return Path(tmp.name), tmp


def check_classes(root: Path, problems: list):
    """A reordered names list would silently relabel everything."""
    for yaml_path in list(root.rglob('data.yaml')) + list(root.rglob('*.yaml')):
        text = yaml_path.read_text(encoding='utf-8', errors='replace')
        if 'names' not in text:
            continue
        found = []
        for line in text.splitlines():
            line = line.strip()
            if line.startswith('names:') and '[' in line:      # names: ['a','b']
                inner = line.split('[', 1)[1].rsplit(']', 1)[0]
                found = [x.strip().strip('\'"') for x in inner.split(',') if x.strip()]
                break
            if line[:1].isdigit() and ':' in line:             # "0: player"
                found.append(line.split(':', 1)[1].strip().strip('\'"'))
            elif line.startswith('- '):                        # "- player"
                found.append(line[2:].strip().strip('\'"'))
        if found:
            if [c.lower() for c in found] != CLASSES:
                problems.append(
                    f'CLASS ORDER MISMATCH in {yaml_path.name}: export has '
                    f'{found}, project requires {CLASSES}. Importing would '
                    f'relabel every box.')
            return found
    problems.append('No data.yaml with a names list found; class order unverified.')
    return None


def recover_stem(name: str, known_sources: set) -> tuple[str | None, str | None]:
    """
    Map an exported filename back to (source, original_stem).

    Roboflow turns  <stem>.jpg  into  <stem>_jpg.rf.<hash>.jpg
    """
    stem = Path(name).stem
    if '.rf.' in stem:
        stem = stem.split('.rf.')[0]
    for suffix in ('_jpg', '_jpeg', '_png', '_JPG'):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break

    # Longest matching source prefix wins (source names contain underscores).
    best = None
    for src in known_sources:
        if stem.startswith(src + '_') and (best is None or len(src) > len(best)):
            best = src
    return best, stem


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--export', required=True,
                   help='Roboflow/CVAT export: .zip or an unzipped folder.')
    p.add_argument('--frames', default='data/frames',
                   help='Used to learn the known source names.')
    p.add_argument('--labels', default='data/labels',
                   help='Destination for corrected labels.')
    p.add_argument('--backup', action='store_true', default=True,
                   help='Back up the existing labels first (default on).')
    p.add_argument('--no-backup', dest='backup', action='store_false')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()

    export = Path(args.export)
    if not export.exists():
        sys.exit(f'Export not found: {export}')

    frames_root = Path(args.frames)
    known = {d.name for d in frames_root.iterdir() if d.is_dir()}
    if not known:
        sys.exit(f'No source directories under {frames_root}')

    root, tmp = unpack(export)
    problems = []
    exported_classes = check_classes(root, problems)

    # Collect every label in the export, whatever split folder it sits in.
    labels = [p for p in root.rglob('*.txt')
              if p.name not in ('classes.txt', 'obj.names', 'README.txt')]
    images = [p for p in root.rglob('*') if p.suffix.lower() in IMG_EXTS]
    by_stem_img = {}
    for img in images:
        src, stem = recover_stem(img.name, known)
        if stem:
            by_stem_img[stem] = img

    mapped, unmapped = defaultdict(list), []
    split_counts = Counter()
    for lbl in labels:
        src, stem = recover_stem(lbl.name, known)
        parts = lbl.relative_to(root).parts
        split_counts[next((s for s in ('train', 'valid', 'val', 'test') if s in parts),
                          'unknown')] += 1
        if src is None:
            unmapped.append(lbl.name)
            continue
        mapped[src].append((stem, lbl))

    total = sum(len(v) for v in mapped.values())
    print(f'export: {export}')
    if exported_classes:
        print(f'classes in export: {exported_classes}')
    print(f'label files found : {len(labels)}')
    print(f'  mapped to a source: {total}')
    print(f'  unmapped          : {len(unmapped)}')
    print(f"roboflow's split (discarded): "
          + ', '.join(f'{k}={v}' for k, v in sorted(split_counts.items())))
    print()
    print(f'{"source":<40}{"labels":>8}')
    print('-' * 50)
    for src in sorted(mapped):
        print(f'{src:<40}{len(mapped[src]):>8}')
    print('-' * 50)
    print(f'{"TOTAL":<40}{total:>8}')

    if unmapped:
        problems.append(f'{len(unmapped)} file(s) could not be traced to a source '
                        f'(first: {unmapped[:3]}). Filenames were probably renamed '
                        f'beyond the Roboflow pattern.')

    # Warn about frames that exist locally but came back with no label file.
    local = {p.stem for p in frames_root.rglob('*.jpg')}
    returned = {stem for v in mapped.values() for stem, _ in v}
    sent = json.loads(Path('data/batches/batch_01.json').read_text(encoding='utf-8')
                      )['images'] if Path('data/batches/batch_01.json').exists() else []
    sent_stems = {Path(s).stem for s in sent}
    missing = sorted(sent_stems - returned)
    if missing:
        problems.append(
            f'{len(missing)} frame(s) were sent for annotation but have no label in '
            f'the export (first: {missing[:3]}). Empty/hard-negative frames are '
            f'often dropped by annotation tools -- they must come back as empty '
            f'.txt files, not vanish.')

    if problems:
        print('\nproblems:')
        for prob in problems:
            print(f'  ! {prob}')

    if args.dry_run:
        print('\n--dry-run: nothing written.')
        if tmp:
            tmp.cleanup()
        return

    if any('CLASS ORDER MISMATCH' in p for p in problems):
        sys.exit('\nRefusing to import: fix the class order in Roboflow first.')

    labels_root = Path(args.labels)
    if args.backup and labels_root.exists():
        from datetime import datetime
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        dest = Path('data/backups') / f'labels_before_import_{stamp}'
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(labels_root, dest)
        print(f'\nbacked up existing labels -> {dest}')

    written = 0
    for src, items in mapped.items():
        out_dir = labels_root / src
        out_dir.mkdir(parents=True, exist_ok=True)
        for stem, lbl in items:
            shutil.copy2(lbl, out_dir / f'{stem}.txt')
            written += 1

    print(f'\nimported {written} label file(s) -> {labels_root}')
    print('\nNext:')
    print('  python tools/validate_annotations.py --strict')
    print('  python tools/build_dataset.py --plan-only')

    if tmp:
        tmp.cleanup()


if __name__ == '__main__':
    main()
