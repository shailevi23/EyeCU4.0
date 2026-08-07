#!/usr/bin/env python
"""
Build a per-image manifest of data/frames and validate its integrity.

Reports, per image: path, source match, frame index, and timestamp derived from
the source video's real FPS. Checks for filename collisions, orphan or empty
source directories, and disagreement with the extractor's own manifest.

Read-only: never modifies or deletes source images.

Examples:
    python tools/dataset_manifest.py
    python tools/dataset_manifest.py --check-images     # also decode every file
    python tools/dataset_manifest.py --out data/manifest.json
"""

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

IMG_EXTS = {'.jpg', '.jpeg', '.png'}


def load_extract_manifest(frames_root: Path) -> dict:
    """FPS per match, as recorded by extract_frames.py. Optional."""
    path = frames_root / 'manifest.json'
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except (json.JSONDecodeError, OSError):
        return {}
    return {m['match_id']: m for m in data.get('matches', [])}


def fps_from_videos(videos_dir: Path) -> dict:
    """
    Read FPS straight from the source videos, keyed by the same sanitized id
    extract_frames.py uses for directory names. Authoritative: it does not
    depend on the extractor's manifest surviving intact.
    """
    if not videos_dir or not videos_dir.is_dir():
        return {}

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from extract_frames import VIDEO_EXTS, sanitize

    import cv2
    out = {}
    for path in sorted(videos_dir.iterdir()):
        if path.suffix.lower() not in VIDEO_EXTS:
            continue
        cap = cv2.VideoCapture(str(path))
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps and fps > 0:
                meta = {'fps': float(fps), 'video': path.name}
                # Register under both the current id and a loose form, so
                # directories created by an older sanitize() (which left
                # double underscores) still resolve.
                out[sanitize(path.stem)] = meta
                out.setdefault(_loose(path.stem), meta)
        cap.release()
    return out


def _loose(name: str) -> str:
    """Lowercase alphanumerics only -- ignores punctuation and underscore runs."""
    import unicodedata
    folded = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode()
    return ''.join(ch for ch in folded.lower() if ch.isalnum())


def resolve_meta(source_name: str, info: dict) -> dict:
    """Exact id first, then the loose form, then an explicit alias file."""
    if source_name in info:
        return info[source_name]
    return info.get(_loose(source_name), {})


def frame_index_from(stem: str, match_id: str):
    """Files are named <match_id>_<frame_index>; recover the index."""
    tail = stem[len(match_id):].lstrip('_') if stem.startswith(match_id) else stem
    digits = ''.join(ch for ch in tail if ch.isdigit())
    return int(digits) if digits else None


def collect(frames_root: Path, extract_info: dict):
    rows, problems = [], []
    per_source = Counter()
    basenames = defaultdict(list)

    sources = sorted(p for p in frames_root.iterdir() if p.is_dir())
    for src in sources:
        images = sorted(p for p in src.iterdir() if p.suffix.lower() in IMG_EXTS)
        if not images:
            problems.append(f'EMPTY SOURCE: {src.name} contains no images')
            continue

        meta = resolve_meta(src.name, extract_info)
        fps = float(meta.get('fps') or 0)
        video = meta.get('video')
        for img in images:
            idx = frame_index_from(img.stem, src.name)
            if idx is None:
                problems.append(f'UNPARSEABLE FRAME INDEX: {img}')
            rows.append({
                'image_path': img.relative_to(frames_root).as_posix(),
                'source': src.name,
                'source_video': video,
                'frame_index': idx,
                'timestamp_sec': round(idx / fps, 3) if (idx is not None and fps) else None,
                'source_fps': fps or None,
                'bytes': img.stat().st_size,
            })
            per_source[src.name] += 1
            basenames[img.name].append(img.relative_to(frames_root).as_posix())

    for name, paths in basenames.items():
        if len(paths) > 1:
            problems.append(f'FILENAME COLLISION: {name} -> {paths}')

    return rows, per_source, problems


def cross_check(extract_info: dict, per_source: Counter, problems: list):
    """The extractor's manifest must not claim more frames than exist."""
    for match_id, info in extract_info.items():
        claimed = info.get('saved', 0)
        actual = per_source.get(match_id, 0)
        if claimed != actual:
            problems.append(
                f'MANIFEST MISMATCH: {match_id} claims {claimed} frames, '
                f'{actual} on disk (difference {actual - claimed})')


def check_images(rows, frames_root: Path, problems: list):
    """Decode every image. Slow, so opt-in."""
    import cv2
    import numpy as np
    for i, row in enumerate(rows, 1):
        path = frames_root / row['image_path']
        try:
            data = np.fromfile(path, dtype=np.uint8)  # unicode-safe read
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        except Exception as e:
            problems.append(f'UNREADABLE: {row["image_path"]} ({e})')
            continue
        if img is None or img.size == 0:
            problems.append(f'CORRUPT: {row["image_path"]}')
        else:
            row['width'], row['height'] = int(img.shape[1]), int(img.shape[0])
        if i % 250 == 0:
            print(f'  decoded {i}/{len(rows)}')


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--frames', default='data/frames')
    p.add_argument('--videos-dir', default='input-videos',
                   help='Source videos, read for authoritative FPS.')
    p.add_argument('--out', default='data/manifest.json',
                   help='Per-image manifest output (JSON).')
    p.add_argument('--csv', help='Optional CSV copy of the manifest.')
    p.add_argument('--check-images', action='store_true',
                   help='Decode every image to catch corruption (slow).')
    p.add_argument('--strict', action='store_true',
                   help='Exit non-zero if any problem is found.')
    args = p.parse_args()

    frames_root = Path(args.frames)
    if not frames_root.exists():
        sys.exit(f'Frames directory not found: {frames_root}')

    # Prefer FPS read from the videos themselves; fall back to the extractor's
    # manifest for any source whose video is no longer present.
    extract_info = load_extract_manifest(frames_root)
    video_info = fps_from_videos(Path(args.videos_dir))
    merged_info = {**extract_info}
    for match_id, meta in video_info.items():
        merged_info.setdefault(match_id, {}).update(meta)

    rows, per_source, problems = collect(frames_root, merged_info)
    cross_check(extract_info, per_source, problems)

    missing_fps = sorted({r['source'] for r in rows if not r['source_fps']})
    if missing_fps:
        problems.append(f'NO FPS (timestamps unavailable): {missing_fps}')
    if args.check_images:
        print('Decoding images...')
        check_images(rows, frames_root, problems)

    if not rows:
        sys.exit(f'No images found under {frames_root}')

    print(f'\n{"source":<40}{"frames":>8}{"share":>8}')
    print('-' * 56)
    total = sum(per_source.values())
    for src, n in sorted(per_source.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f'{src:<40}{n:>8}{n / total:>7.1%}')
    print('-' * 56)
    print(f'{"TOTAL":<40}{total:>8}   {len(per_source)} sources')

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        'total_frames': total,
        'sources': len(per_source),
        'frames_per_source': dict(sorted(per_source.items())),
        'problems': problems,
        'images': rows,
    }, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'\nManifest: {out}  ({len(rows)} entries)')

    if args.csv:
        csv_path = Path(args.csv)
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f'CSV     : {csv_path}')

    if problems:
        print(f'\n{len(problems)} problem(s):')
        for prob in problems[:40]:
            print(f'  ! {prob}')
        if len(problems) > 40:
            print(f'  ... and {len(problems) - 40} more')
    else:
        print('\nNo collisions, no empty sources, manifest agrees with disk.')

    if total < 1000:
        print(f'\n! {total} frames is below the 1,000-1,500 target.')
    if len(per_source) < 4:
        print(f'\n! only {len(per_source)} sources; a match-disjoint split needs more.')

    if args.strict and problems:
        sys.exit(1)


if __name__ == '__main__':
    main()
