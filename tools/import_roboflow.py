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
            return [c.lower() for c in found]
    problems.append('No data.yaml with a names list found; class order unverified.')
    return None


def build_remap(exported: list, problems: list):
    """
    Map exported class ids onto this project's ids, matching by NAME.

    Roboflow alphabetises class names on export, so the ids almost never line
    up with ours. Remapping by name is correct and safe; trusting the ids is
    what would silently turn every player into a referee.

    Returns {exported_id: project_id}, or None if the names cannot be matched.
    """
    if exported is None:
        return None
    if exported == CLASSES:
        return {i: i for i in range(len(CLASSES))}

    unknown = [n for n in exported if n not in CLASSES]
    if unknown:
        problems.append(
            f'UNKNOWN CLASSES in export: {unknown}. Expected exactly {CLASSES}. '
            f'A class was renamed or added in the annotation tool.')
        return None
    missing = [n for n in CLASSES if n not in exported]
    if missing:
        problems.append(
            f'Export is missing class(es) {missing}; ids for the rest will still '
            f'be remapped by name.')
    return {i: CLASSES.index(name) for i, name in enumerate(exported)}


def polygon_to_box(coords: list) -> tuple:
    """
    Segmentation polygon -> YOLO detection box.

    Roboflow's Smart Polygon tool emits `class x1 y1 x2 y2 ...` instead of
    `class cx cy w h`. Ultralytics detection training rejects those lines, so
    the enclosing axis-aligned box is taken. Lossless for detection: the box is
    exactly what a detector would have been asked to predict anyway.
    """
    xs, ys = coords[0::2], coords[1::2]
    x1, x2 = max(0.0, min(xs)), min(1.0, max(xs))
    y1, y2 = max(0.0, min(ys)), min(1.0, max(ys))
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return None
    return (x1 + w / 2, y1 + h / 2, w, h)


def remap_label_text(text: str, remap: dict, stats: Counter) -> str:
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        try:
            old = int(float(parts[0]))
            values = [float(v) for v in parts[1:]]
        except ValueError:
            stats['dropped_malformed'] += 1
            continue

        new = remap.get(old)
        if new is None:
            stats['dropped_unknown_id'] += 1
            continue

        if len(values) == 4:
            cx, cy, w, h = values
        elif len(values) >= 6 and len(values) % 2 == 0:
            box = polygon_to_box(values)
            if box is None:
                stats['dropped_degenerate_polygon'] += 1
                continue
            cx, cy, w, h = box
            stats['polygons_converted'] += 1
        else:
            stats['dropped_malformed'] += 1
            continue

        stats[CLASSES[new]] += 1
        out.append(f'{new} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}')
    return '\n'.join(out) + ('\n' if out else '')


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
    remap = build_remap(exported_classes, problems)

    # Collect every label in the export, whatever split folder it sits in.
    # Exports ship documentation and class-name files as .txt too.
    NON_LABEL = {'classes.txt', 'obj.names', 'obj.data'}
    labels = [p for p in root.rglob('*.txt')
              if p.name not in NON_LABEL and not p.name.lower().startswith('readme')]
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
        print(f'classes in export : {exported_classes}')
        print(f'classes in project: {CLASSES}')
        if remap and any(k != v for k, v in remap.items()):
            print('CLASS ID REMAP (by name):')
            for old, new in sorted(remap.items()):
                arrow = '  (unchanged)' if old == new else ''
                print(f'  {old} {exported_classes[old]:<11} -> {new} '
                      f'{CLASSES[new]}{arrow}')
        elif remap:
            print('class ids already match; no remap needed')
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

    if remap is None:
        sys.exit('\nRefusing to import: class names could not be matched, so box '
                 'classes cannot be trusted.')

    labels_root = Path(args.labels)
    if args.backup and labels_root.exists():
        from datetime import datetime
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        dest = Path('data/backups') / f'labels_before_import_{stamp}'
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(labels_root, dest)
        print(f'\nbacked up existing labels -> {dest}')

    written = 0
    stats = Counter()
    for src, items in mapped.items():
        out_dir = labels_root / src
        out_dir.mkdir(parents=True, exist_ok=True)
        for stem, lbl in items:
            text = lbl.read_text(encoding='utf-8', errors='replace')
            (out_dir / f'{stem}.txt').write_text(
                remap_label_text(text, remap, stats), encoding='utf-8')
            written += 1

    print(f'\nimported {written} label file(s) -> {labels_root}')
    print('instances by class after remap:')
    for c in CLASSES:
        print(f'  {c:<11}{stats.get(c, 0):>7}')
    if stats.get('polygons_converted'):
        print(f'  converted {stats["polygons_converted"]} segmentation '
              f'polygon(s) to bounding boxes')
    for key, label in (('dropped_unknown_id', 'unmappable class id'),
                       ('dropped_malformed', 'malformed line'),
                       ('dropped_degenerate_polygon', 'zero-area polygon')):
        if stats.get(key):
            print(f'  ! dropped {stats[key]} annotation(s): {label}')
    print('\nNext:')
    print('  python tools/validate_annotations.py --strict')
    print('  python tools/build_dataset.py --plan-only')

    if tmp:
        tmp.cleanup()


if __name__ == '__main__':
    main()
