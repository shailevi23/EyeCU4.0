#!/usr/bin/env python
"""
Build a small CONTINUOUS validation benchmark for temporal ball logic.

Why this exists. The 208-image validation split is excellent for per-frame
detector evaluation and useless for anything temporal: its frames were interval
sampled, so consecutive images are seconds apart. A gap-recovery rule, a
velocity gate or a track-continuity metric evaluated on them would be measuring
nothing -- there is no motion to model between two frames taken 4 seconds
apart. Temporal logic needs frames that are actually adjacent in time.

So this extracts short continuous windows from the *same four frozen
validation matches*, at a fixed diagnostic frame rate, preserving the source
frame number and timestamp of every frame.

Window selection is deterministic and content-neutral: a fixed fraction into
each video, a fixed duration. Choosing windows by looking at where the detector
struggles would bias the benchmark toward whatever rescue logic gets built
next, which is precisely the mistake this benchmark exists to avoid.

VALIDATION ONLY. These frames come from matches pinned to val, so they may
never enter train or test. `assert_val_only()` enforces that and is covered by
tests; the output directory sits outside data/dataset_baseline/ so the dataset
builder cannot pick it up by globbing.

Example:
    python tools/build_temporal_val.py --dry-run
    python tools/build_temporal_val.py --zip
"""

import argparse
import json
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# The four matches pinned to validation. Any change here is a split change and
# must go through tools/build_dataset.py --force-val, not this file.
VAL_MATCHES = {
    'austin_fc_vs__club_tijuana': 'Austin FC vs. Club Tijuana.mp4',
    'bayern_munich_3-1_chelsea': 'Bayern Munich 3-1 Chelsea.mp4',
    'women_1': 'women 1.mp4',
    'youth_premier_league': 'ליגת העל לנוער.mp4',
}

# Deterministic, content-neutral window rule.
#
# Two shorter windows rather than one long one. A single 5s window at 40% put
# the austin sample entirely inside a goalkeeper close-up that also contained a
# camera cut -- a quarter of the benchmark with almost no visible ball. Two
# windows at different points in the match make that far less likely.
#
# This rule was revised after looking at the extracted frames, but *not* after
# running any detector over them: the criterion was coverage of shot types,
# which is independent of how the model performs. Windows are still fixed
# fractions of duration, never chosen by inspecting detector behaviour.
WINDOW_FRACTIONS = (0.40, 0.70)   # of total duration
WINDOW_SECONDS = 2.5              # each
DIAGNOSTIC_FPS = 5.0

SPLIT_MARKER = 'VAL_ONLY'


def assert_val_only(paths) -> None:
    """
    Refuse to let temporal-val frames reach train or test.

    Called by the dataset tooling and by tests. Kept deliberately blunt: a
    frame that leaks from here into training would silently invalidate every
    validation number the project reports.
    """
    for p in paths:
        parts = {q.lower() for q in Path(p).parts}
        if 'train' in parts or 'test' in parts:
            raise ValueError(
                f'temporal-val frame routed to a non-val split: {p}. '
                f'These frames come from matches pinned to validation and may '
                f'only ever be used for validation.')


def imwrite_unicode(path: Path, image, quality: int = 95) -> bool:
    """cv2.imwrite silently fails on non-ASCII paths on Windows; this repo has
    Hebrew in its path. Encode in memory, write the bytes ourselves."""
    import cv2
    ok, buf = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return False
    try:
        path.write_bytes(buf.tobytes())
    except OSError:
        return False
    return True


def plan_windows(fps: float, n_frames: int):
    """
    Return (stride, [(window_index, start_frame, [frame_indices]), ...]).

    None when the video is unusable. Windows never overlap and never run past
    the end of the video.
    """
    if fps <= 0 or n_frames <= 0:
        return None
    stride = max(1, int(round(fps / DIAGNOSTIC_FPS)))
    span = int(round(fps * WINDOW_SECONDS))

    windows, used_to = [], -1
    for w, frac in enumerate(WINDOW_FRACTIONS):
        start = int(round(n_frames * frac))
        if start <= used_to:                 # keep windows disjoint
            continue
        idx = list(range(start, min(start + span, n_frames), stride))
        if not idx:
            continue
        windows.append((w, start, idx))
        used_to = idx[-1]
    return stride, windows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--videos', default='input-videos')
    ap.add_argument('--out', default='data/temporal_val')
    ap.add_argument('--dry-run', action='store_true',
                    help='Plan and report only; write nothing.')
    ap.add_argument('--zip', action='store_true',
                    help='Also write an annotation ZIP next to the frames.')
    args = ap.parse_args()

    import cv2

    videos_root = Path(args.videos)
    out_root = Path(args.out)
    img_dir = out_root / 'images'

    manifest = {
        'split': SPLIT_MARKER,
        'purpose': 'continuous temporal validation benchmark (ball)',
        'created': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'window_rule': {
            'start_fractions_of_duration': list(WINDOW_FRACTIONS),
            'window_seconds_each': WINDOW_SECONDS,
            'diagnostic_fps': DIAGNOSTIC_FPS,
            'note': 'Deterministic and content-neutral. Windows were NOT chosen '
                    'by inspecting detector behaviour.',
        },
        'frames': [],
    }

    print(f'{"match":<32}{"fps":>7}{"frames":>8}{"start":>7}{"stride":>7}'
          f'{"sampled":>8}{"window":>16}')
    print('-' * 85)

    total = 0
    for match, filename in VAL_MATCHES.items():
        path = videos_root / filename
        if not path.exists():
            print(f'{match:<32}  MISSING: {path}')
            continue

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            print(f'{match:<32}  COULD NOT OPEN: {path}')
            cap.release()
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        plan = plan_windows(fps, n_frames)
        if plan is None:
            print(f'{match:<32}  UNUSABLE fps/frame count')
            cap.release()
            continue
        stride, windows = plan

        for w, start, indices in windows:
            t0, t1 = start / fps, indices[-1] / fps
            label = f'{match} [w{w}]'
            print(f'{label:<32}{fps:>7.2f}{n_frames:>8}{start:>7}{stride:>7}'
                  f'{len(indices):>8}{f"{t0:.1f}-{t1:.1f}s":>16}')
            total += len(indices)

            if args.dry_run:
                continue

            img_dir.mkdir(parents=True, exist_ok=True)
            written = 0
            for order, fi in enumerate(indices):
                cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
                ok, frame = cap.read()
                if not ok:
                    print(f'  warning: could not read frame {fi} of {match}')
                    continue
                # 'tv' namespaces these away from the interval-sampled val
                # frames, which use <match>_<index>.jpg and could collide.
                name = f'{match}_tv{fi:06d}.jpg'
                if not imwrite_unicode(img_dir / name, frame):
                    print(f'  warning: could not write {name}')
                    continue
                written += 1
                manifest['frames'].append({
                    'file': name,
                    'match': match,
                    'window': w,
                    'source_video': str(path),
                    'source_frame_index': fi,
                    'timestamp_seconds': round(fi / fps, 4),
                    'order_in_window': order,
                    'source_fps': round(fps, 4),
                    'stride_frames': stride,
                    'effective_fps': round(fps / stride, 4),
                    'split': SPLIT_MARKER,
                })
            if written != len(indices):
                print(f'  {match} w{w}: wrote {written} of {len(indices)}')
        cap.release()

    print('-' * 85)
    print(f'{"TOTAL":<32}{"":>7}{"":>8}{"":>7}{"":>7}{total:>8}')

    if args.dry_run:
        print('\n(dry run -- nothing written)')
        return

    if not manifest['frames']:
        sys.exit('No frames extracted; refusing to write an empty benchmark.')

    assert_val_only(f['file'] for f in manifest['frames'])

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / 'manifest.json').write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    (out_root / 'SPLIT').write_text(
        f'{SPLIT_MARKER}\n\nFrames here come from the four matches pinned to '
        f'validation.\nThey must never be added to train or test.\n',
        encoding='utf-8')

    readme = out_root / 'ANNOTATION.md'
    readme.write_text(
        '# Temporal validation benchmark — ball annotation\n\n'
        f'{len(manifest["frames"])} frames: two continuous windows per match, '
        f'{DIAGNOSTIC_FPS:.0f} FPS diagnostic sampling, '
        f'{WINDOW_SECONDS:.0f}s per match.\n\n'
        '## What to label\n\n'
        'The **ball only**. Players, goalkeepers and referees are not needed '
        'here — this benchmark exists to measure temporal ball recovery.\n\n'
        '- One box on the ball when it is **visually identifiable**.\n'
        '- If you cannot see it, leave the frame empty. An empty label file is '
        'a real signal: it is the ground truth for "no visible ball", which is '
        'what the selector must learn to report as `unknown` rather than '
        'inventing a position.\n'
        '- **Do not** infer position from player gaze, from the previous frame, '
        'or from where the ball must be. A guessed ball makes the benchmark '
        'reward hallucination.\n'
        '- If a second football is genuinely visible (spare ball, sideline), '
        'label it too and note the frame. The main val set never labels a '
        'second ball, so this benchmark is the only place that ambiguity can '
        'be resolved.\n\n'
        '## Class ids\n\n```\n0 ball\n```\n\n'
        'Note this is a **benchmark-local** id. The main dataset uses '
        '`3 ball`; conversion happens at evaluation time, not here.\n\n'
        '## Split\n\nVALIDATION ONLY. Never train on these frames.\n',
        encoding='utf-8')

    print(f'\nwritten: {img_dir} ({len(manifest["frames"])} frames)')
    print(f'         {out_root / "manifest.json"}')
    print(f'         {readme}')

    if args.zip:
        zpath = out_root / 'temporal_val_for_annotation.zip'
        with zipfile.ZipFile(zpath, 'w', zipfile.ZIP_DEFLATED) as z:
            for f in sorted(img_dir.glob('*.jpg')):
                z.write(f, f'images/{f.name}')
            z.write(out_root / 'manifest.json', 'manifest.json')
            z.write(readme, 'ANNOTATION.md')
        print(f'         {zpath} ({zpath.stat().st_size / 1e6:.1f} MB)')


if __name__ == '__main__':
    main()
