#!/usr/bin/env python
"""
Extract a diverse set of training frames from match videos.

Frames are named `<match_id>_<frame_index>.jpg` so that build_dataset.py can
split by match (never by random frame -- adjacent frames are near-duplicates
and would leak the validation set, see ROADMAP.md section 2).

Sampling strategy per video:
  * one frame every --interval-sec seconds (ordinary gameplay coverage)
  * short bursts of consecutive frames around high-motion moments
    (corners, tackles, fast pans -- the hard cases)
  * blurry frames below --min-sharpness are dropped
  * near-duplicate frames are dropped via a dHash comparison

Examples:
    python tools/extract_frames.py --videos-dir input-videos --out data/frames
    python tools/extract_frames.py --video match1.mp4 --match-id ars_liv --max-frames 300
"""

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

VIDEO_EXTS = {'.mp4', '.mkv', '.avi', '.mov', '.m4v', '.ts', '.webm'}


def sanitize(name: str) -> str:
    keep = [c if (c.isalnum() or c in '-_') else '_' for c in name]
    return ''.join(keep).strip('_').lower() or 'match'


def sharpness(gray: np.ndarray) -> float:
    """Variance of Laplacian -- low values mean motion blur / out of focus."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def dhash(gray: np.ndarray, size: int = 8) -> int:
    small = cv2.resize(gray, (size + 1, size), interpolation=cv2.INTER_AREA)
    diff = small[:, 1:] > small[:, :-1]
    bits = 0
    for b in diff.flatten():
        bits = (bits << 1) | int(b)
    return bits


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count('1')


def motion_score(prev_gray: np.ndarray, gray: np.ndarray) -> float:
    """Mean absolute difference between consecutive downscaled frames."""
    a = cv2.resize(prev_gray, (160, 90), interpolation=cv2.INTER_AREA)
    b = cv2.resize(gray, (160, 90), interpolation=cv2.INTER_AREA)
    return float(np.mean(cv2.absdiff(a, b)))


def extract_one(video_path: Path, out_dir: Path, match_id: str, args) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open video: {video_path}')

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    interval = max(1, int(round(fps * args.interval_sec)))

    out_dir.mkdir(parents=True, exist_ok=True)

    kept_hashes: list[int] = []
    saved = 0
    dropped_blur = 0
    dropped_dupe = 0
    burst_left = 0
    prev_gray = None
    motion_history: list[float] = []
    idx = -1

    # Blur is judged relative to THIS video: broadcast sources differ hugely in
    # intrinsic sharpness, so an absolute threshold would empty out a soft video
    # and filter nothing from a crisp one. Candidates are buffered until the
    # warmup sample is large enough to pick a percentile threshold.
    warmup: list[tuple[str, np.ndarray, np.ndarray, float]] = []
    blur_threshold = None

    def commit(name, frame, gray):
        """Dedupe against recent frames, then write. Returns True if written."""
        nonlocal saved, dropped_dupe
        h = dhash(gray)
        if any(hamming(h, k) <= args.dupe_distance for k in kept_hashes[-args.dupe_window:]):
            dropped_dupe += 1
            return False
        kept_hashes.append(h)
        cv2.imwrite(str(out_dir / name), frame,
                    [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
        saved += 1
        return True

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Track motion so we can trigger bursts on the busy moments.
        motion = motion_score(prev_gray, gray) if prev_gray is not None else 0.0
        prev_gray = gray
        motion_history.append(motion)
        if len(motion_history) > 300:
            motion_history.pop(0)

        want = idx % interval == 0
        if not want and burst_left > 0 and idx % args.burst_stride == 0:
            want = True
            burst_left -= 1

        # Trigger a burst when this frame is a clear motion outlier.
        if args.burst_frames and len(motion_history) > 30 and burst_left == 0:
            med = float(np.median(motion_history))
            if med > 0 and motion > med * args.burst_motion_ratio:
                burst_left = args.burst_frames
                want = True

        if not want:
            continue

        sharp = sharpness(gray)
        if sharp < args.min_sharpness:  # absolute floor, off by default
            dropped_blur += 1
            continue

        name = f'{match_id}_{idx:06d}.jpg'

        if blur_threshold is None:
            warmup.append((name, frame, gray, sharp))
            if len(warmup) < args.blur_warmup:
                continue
            # Enough samples: fix the threshold and flush the buffer in order.
            blur_threshold = float(np.percentile([w[3] for w in warmup],
                                                 args.blur_percentile))
            for wname, wframe, wgray, wsharp in warmup:
                if wsharp < blur_threshold:
                    dropped_blur += 1
                    continue
                commit(wname, wframe, wgray)
                if args.max_frames and saved >= args.max_frames:
                    break
            warmup.clear()
        else:
            if sharp < blur_threshold:
                dropped_blur += 1
                continue
            commit(name, frame, gray)

        if args.max_frames and saved >= args.max_frames:
            break

    # Short video that never reached the warmup size: keep everything buffered.
    if blur_threshold is None:
        for wname, wframe, wgray, _ in warmup:
            if args.max_frames and saved >= args.max_frames:
                break
            commit(wname, wframe, wgray)

    cap.release()
    return {
        'match_id': match_id,
        'video': str(video_path),
        'fps': round(fps, 3),
        'total_frames': total,
        'duration_sec': round(total / fps, 1) if fps else None,
        'saved': saved,
        'dropped_blur': dropped_blur,
        'dropped_duplicate': dropped_dupe,
        'blur_threshold': round(blur_threshold, 2) if blur_threshold else None,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_argument_group('input')
    src.add_argument('--video', action='append', default=[],
                     help='Path to a video (repeatable).')
    src.add_argument('--videos-dir', help='Directory of videos; each file is one match.')
    src.add_argument('--match-id', help='Match id for a single --video (default: filename stem).')

    p.add_argument('--out', default='data/frames', help='Output root (one subdir per match).')
    p.add_argument('--interval-sec', type=float, default=3.0,
                   help='Seconds between routine samples (default 3).')
    p.add_argument('--max-frames', type=int, default=400,
                   help='Max frames saved per video (0 = unlimited).')
    p.add_argument('--blur-percentile', type=float, default=15.0,
                   help='Drop the blurriest N%% of candidates, measured against '
                        'this video only (default 15).')
    p.add_argument('--blur-warmup', type=int, default=40,
                   help='Candidates sampled before fixing the blur threshold.')
    p.add_argument('--min-sharpness', type=float, default=0.0,
                   help='Absolute Laplacian-variance floor, applied on top of '
                        '--blur-percentile (default 0 = disabled).')
    p.add_argument('--burst-frames', type=int, default=6,
                   help='Extra frames captured around a motion spike (0 disables bursts).')
    p.add_argument('--burst-stride', type=int, default=5,
                   help='Frame stride inside a burst.')
    p.add_argument('--burst-motion-ratio', type=float, default=1.8,
                   help='Motion must exceed median * this ratio to trigger a burst.')
    p.add_argument('--dupe-distance', type=int, default=6,
                   help='dHash Hamming distance below which frames count as duplicates.')
    p.add_argument('--dupe-window', type=int, default=60,
                   help='How many recent hashes to compare against.')
    p.add_argument('--jpeg-quality', type=int, default=92)
    p.add_argument('--clean', action='store_true', help='Delete the output root first.')

    args = p.parse_args()

    videos: list[tuple[Path, str]] = []
    for v in args.video:
        path = Path(v)
        videos.append((path, sanitize(args.match_id or path.stem)))
    if args.videos_dir:
        for path in sorted(Path(args.videos_dir).iterdir()):
            if path.suffix.lower() in VIDEO_EXTS:
                videos.append((path, sanitize(path.stem)))

    if not videos:
        p.error('No videos found. Pass --video and/or --videos-dir.')

    out_root = Path(args.out)
    if args.clean and out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    manifest = []
    for path, match_id in videos:
        if not path.exists():
            print(f'! skipping missing video: {path}')
            continue
        print(f'-> {path.name}  (match_id={match_id})')
        info = extract_one(path, out_root / match_id, match_id, args)
        manifest.append(info)
        print(f'   saved {info["saved"]}  '
              f'(blur {info["dropped_blur"]}, dupes {info["dropped_duplicate"]})')

    total = sum(m['saved'] for m in manifest)
    (out_root / 'manifest.json').write_text(
        json.dumps({'total_frames': total, 'matches': manifest}, indent=2))

    print(f'\nTotal frames: {total} across {len(manifest)} match(es) -> {out_root}')
    if len(manifest) < 4:
        print('! ROADMAP.md asks for 4-6 different matches. '
              'One or two matches will not generalise.')
    if total < 1000:
        print(f'! Target is 1000-1500 frames; you have {total}. '
              'Add more matches or lower --interval-sec.')


if __name__ == '__main__':
    main()
