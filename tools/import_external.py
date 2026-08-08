#!/usr/bin/env python
"""
Import an external YOLO dataset as EXTRA TRAINING DATA ONLY.

External frames may never reach validation or test: those splits are the only
honest measurement this project has, and they must stay entirely EyeCU footage
from matches the model has never seen.

What this handles:

  * Class remapping by NAME. Roboflow alphabetises class lists on export, so
    the ids almost never line up with ours. Trusting the ids would relabel
    every box.

  * Aspect-ratio restoration. Roboflow exports are commonly resized to a
    square, which horizontally squashes 16:9 broadcast footage and teaches the
    model that people are ~1.8x narrower than they are. Images are stretched
    back to --aspect. YOLO coordinates are normalised, so the boxes follow the
    content automatically and no relabelling is needed.

  * Provenance. Each source clip becomes its own directory `<prefix><clip>`,
    so every imported frame is traceable and the split tool can keep clips
    match-disjoint.

Examples:
    python tools/import_external.py --export football.zip --dry-run
    python tools/import_external.py --export football.zip
    python tools/import_external.py --export football.zip --aspect 1.0   # no restore
"""

import argparse
import json
import re
import shutil
import sys
import tempfile
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

CLASSES = ['player', 'goalkeeper', 'referee', 'ball']
IMG_EXTS = {'.jpg', '.jpeg', '.png'}
NON_LABEL = {'classes.txt', 'obj.names', 'obj.data'}


def unpack(export: Path):
    if export.is_dir():
        return export, None
    tmp = tempfile.TemporaryDirectory()
    with zipfile.ZipFile(export) as z:
        z.extractall(tmp.name)
    return Path(tmp.name), tmp


def read_names(root: Path):
    """Class names declared by the export, in their id order."""
    for yaml_path in sorted(root.rglob('*.yaml')):
        text = yaml_path.read_text(encoding='utf-8', errors='replace')
        if 'names' not in text:
            continue
        found = []
        for line in text.splitlines():
            line = line.strip()
            if line.startswith('names:') and '[' in line:
                inner = line.split('[', 1)[1].rsplit(']', 1)[0]
                found = [x.strip().strip('\'"') for x in inner.split(',') if x.strip()]
                break
            if line[:1].isdigit() and ':' in line:
                found.append(line.split(':', 1)[1].strip().strip('\'"'))
            elif line.startswith('- '):
                found.append(line[2:].strip().strip('\'"'))
        if found:
            return [c.lower() for c in found], yaml_path
    return None, None


def clip_id(filename: str) -> str:
    """
    Group frames by source clip so provenance survives and the splitter can
    keep clips together. Roboflow names look like `<clip>_<n>_<m>_png.rf.<hash>`.
    """
    stem = Path(filename).stem
    if '.rf.' in stem:
        stem = stem.split('.rf.')[0]
    for suffix in ('_png', '_jpg', '_jpeg'):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    m = re.match(r'^([0-9a-zA-Z]+?)_\d+(?:_\d+)?$', stem)
    return (m.group(1) if m else stem.split('_')[0]).lower()


def imread_unicode(path: Path):
    try:
        return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        return None


def imwrite_unicode(path: Path, image, quality=95) -> bool:
    ok, buf = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return False
    path.write_bytes(buf.tobytes())
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--export', required=True, help='Zip or unzipped folder.')
    p.add_argument('--frames', default='data/frames')
    p.add_argument('--labels', default='data/labels')
    p.add_argument('--prefix', default='rfext_',
                   help='Prefix for imported source directories.')
    p.add_argument('--aspect', type=float, default=16 / 9,
                   help='Target width/height to restore. 0 disables resizing.')
    p.add_argument('--provenance', default='data/external_provenance.json')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()

    export = Path(args.export)
    if not export.exists():
        sys.exit(f'Export not found: {export}')

    root, tmp = unpack(export)
    names, yaml_path = read_names(root)
    if not names:
        sys.exit('No class names found in the export; cannot remap safely.')

    unknown = [n for n in names if n not in CLASSES]
    if unknown:
        sys.exit(f'Unknown classes in export: {unknown}. Expected {CLASSES}.')
    remap = {i: CLASSES.index(n) for i, n in enumerate(names)}

    print(f'export : {export}')
    print(f'classes in export : {names}')
    print(f'classes in project: {CLASSES}')
    print('remap by name:')
    for old, new in sorted(remap.items()):
        mark = '' if old == new else '   <-- changed'
        print(f'  {old} {names[old]:<11} -> {new} {CLASSES[new]}{mark}')

    images = [q for q in root.rglob('*')
              if q.suffix.lower() in IMG_EXTS and q.is_file()]
    pairs, missing = [], []
    for img in sorted(images):
        lbl = img.parent.parent / 'labels' / f'{img.stem}.txt'
        if not lbl.exists():
            cands = [c for c in root.rglob(f'{img.stem}.txt')
                     if c.name not in NON_LABEL]
            lbl = cands[0] if cands else None
        if lbl is None:
            missing.append(img.name)
            continue
        pairs.append((img, lbl))

    by_clip = defaultdict(list)
    for img, lbl in pairs:
        by_clip[clip_id(img.name)].append((img, lbl))

    print(f'\n{len(pairs)} image/label pair(s), {len(by_clip)} source clip(s)')
    if missing:
        print(f'! {len(missing)} image(s) with no label, skipped: {missing[:3]}')

    counts = Counter()
    for _, lbl in pairs:
        for line in lbl.read_text(encoding='utf-8').splitlines():
            if line.strip():
                counts[CLASSES[remap[int(float(line.split()[0]))]]] += 1
    print('instances after remap: ' + '  '.join(f'{c}={counts[c]}' for c in CLASSES))

    print(f'\n{"source dir":<24}{"frames":>8}')
    print('-' * 34)
    for c in sorted(by_clip):
        print(f'{args.prefix + c:<24}{len(by_clip[c]):>8}')

    if args.aspect:
        sample = imread_unicode(pairs[0][0])
        if sample is not None:
            h, w = sample.shape[:2]
            print(f'\naspect: source {w}x{h} ({w / h:.3f}) -> '
                  f'target {args.aspect:.3f}'
                  f'{"  (no change)" if abs(w / h - args.aspect) < 0.01 else ""}')

    if args.dry_run:
        print('\n--dry-run: nothing written.')
        if tmp:
            tmp.cleanup()
        return

    frames_root, labels_root = Path(args.frames), Path(args.labels)
    clash = [args.prefix + c for c in by_clip if (frames_root / (args.prefix + c)).exists()]
    if clash:
        sys.exit(f'Refusing to overwrite existing sources: {clash[:5]}. '
                 f'Remove them or use a different --prefix.')

    written = 0
    resized = 0
    for c, items in sorted(by_clip.items()):
        src = args.prefix + c
        (frames_root / src).mkdir(parents=True, exist_ok=True)
        (labels_root / src).mkdir(parents=True, exist_ok=True)
        for i, (img, lbl) in enumerate(sorted(items, key=lambda t: t[0].name)):
            image = imread_unicode(img)
            if image is None:
                print(f'  ! unreadable, skipped: {img.name}')
                continue
            h, w = image.shape[:2]
            if args.aspect and abs(w / h - args.aspect) > 0.01:
                # Stretch back to the true aspect. Normalised YOLO coordinates
                # are scale-invariant, so the boxes stretch with the content
                # and the label file needs no edit.
                image = cv2.resize(image, (int(round(h * args.aspect)), h),
                                   interpolation=cv2.INTER_CUBIC)
                resized += 1

            stem = f'{src}_{i:06d}'
            imwrite_unicode(frames_root / src / f'{stem}.jpg', image)

            out = []
            for line in lbl.read_text(encoding='utf-8').splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                new = remap[int(float(parts[0]))]
                out.append(' '.join([str(new)] + parts[1:]))
            (labels_root / src / f'{stem}.txt').write_text(
                '\n'.join(out) + ('\n' if out else ''), encoding='utf-8')
            written += 1

    prov = Path(args.provenance)
    record = {}
    if prov.exists():
        record = json.loads(prov.read_text(encoding='utf-8'))
    record[str(export.name)] = {
        'imported_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'source_yaml': str(yaml_path.relative_to(root)) if yaml_path else None,
        'declared_classes': names,
        'class_remap': {str(k): CLASSES[v] for k, v in remap.items()},
        'aspect_restored_to': args.aspect or None,
        'images_written': written,
        'images_resized': resized,
        'instances': {c: counts[c] for c in CLASSES},
        'source_dirs': sorted(args.prefix + c for c in by_clip),
        'training_only': True,
        'note': 'External data. Training only -- must never enter val or test.',
    }
    prov.parent.mkdir(parents=True, exist_ok=True)
    prov.write_text(json.dumps(record, indent=2), encoding='utf-8')

    print(f'\nimported {written} frame(s), {resized} resized')
    print(f'provenance: {prov}')
    print('\nThese sources are TRAINING ONLY. Build with:')
    print('  python tools/build_dataset.py --force-train ' +
          ' '.join(sorted(args.prefix + c for c in by_clip)[:3]) + ' ...')

    if tmp:
        tmp.cleanup()


if __name__ == '__main__':
    main()
