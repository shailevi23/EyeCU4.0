#!/usr/bin/env python
"""
Freeze A@960 human detections for the four VAL continuity windows.

Every tracker comparison downstream must receive byte-identical input, or the
comparison measures detector nondeterminism and timing as much as it measures
association. This writes that input once, with enough provenance to prove later
that a given result came from these exact detections.

DETECTIONS ONLY. No tracker is constructed, no tracker id is written, no
temporal or role logic runs. The ball is excluded from the human tracking
stream by construction -- ball settings are recorded in the manifest for
provenance only.

Determinism notes:
  - the detector's in-memory cache is bypassed (frame_id=None), and no disk
    cache is consulted, so nothing stale can be frozen;
  - detections keep the detector's native ordering, not a re-sorted one, so a
    later equivalence check can compare exactly rather than up to permutation;
  - floats are serialised with Python's repr, which round-trips exactly, so no
    tolerance is required to reload them.

Example:
    python tools/freeze_tracking_detections.py --dry-run
    python tools/freeze_tracking_detections.py
"""

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trackers.detector import (BALL_ACCEPT_CONF, BALL_CANDIDATE_CONF,  # noqa: E402
                               BALL_DEDUPE_IOU, CLASSES, HUMAN_ACCEPT_CONF,
                               HUMAN_CANDIDATE_CONF, HUMAN_CLASSES,
                               LocalDetector)
from tools.build_derived_train import TEST_MATCHES, VAL_MATCHES  # noqa: E402

BENCHMARK = 'EyeCU-Tracking-Val-v1'
N_FRAMES = 300
MODEL = 'best_A_960.pt'
IMGSZ = 960
CONF = 0.25

# The four frozen continuity windows. Changing any of these invalidates every
# downstream comparison, so they live here as data, not as CLI arguments.
WINDOWS = [
    ('austin_fc_vs__club_tijuana', 'input-videos/Austin FC vs. Club Tijuana.mp4', 284),
    ('bayern_munich_3-1_chelsea', 'input-videos/Bayern Munich 3-1 Chelsea.mp4', 228),
    ('women_1', 'input-videos/women 1.mp4', 239),
    ('youth_premier_league', 'input-videos/ליגת העל לנוער.mp4', 1133),
]


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def read_window(path: str, start: int, n: int):
    """Decode n consecutive frames and hash the decoded pixels."""
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f'cannot open {path}')
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames, h = [], hashlib.sha256()
    for _ in range(n):
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
        h.update(np.ascontiguousarray(f).tobytes())
    cap.release()
    return frames, fps, total, h.hexdigest()


def serialise(frame_idx: int, dets) -> str:
    """One JSON object per frame. Sorted keys; native detection order kept."""
    rows = []
    for d in dets:
        rows.append({
            'bbox': [float(v) for v in d['bbox']],
            'class': d['class'],
            'confidence': float(d['confidence']),
            'state': d.get('state'),
        })
    return json.dumps({'frame': frame_idx, 'detections': rows}, sort_keys=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default='data/tracking_val_v1')
    ap.add_argument('--model', default=MODEL)
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    names = {m for m, _, _ in WINDOWS}
    leaked = names & TEST_MATCHES
    if leaked:
        raise SystemExit(f'REFUSING: TEST source in the freeze list: {sorted(leaked)}')
    if not names <= VAL_MATCHES:
        raise SystemExit(f'REFUSING: non-VAL source: {sorted(names - VAL_MATCHES)}')

    out = Path(args.out)
    det_dir = out / 'detections'
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'],
                            capture_output=True, text=True).stdout.strip()

    import cv2
    manifest = {
        'benchmark': BENCHMARK,
        'created': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'code_commit': commit,
        'contains': 'human detections only; no tracker ids, no ball, no temporal logic',
        'detector': {
            'checkpoint': args.model,
            'checkpoint_sha256': sha256_file(Path(args.model)),
            'imgsz': IMGSZ,
            'confidence': CONF,
            'accepted_human_confidence': HUMAN_ACCEPT_CONF,
            'accepted_human_behaviour':
                'detector emits humans at >= confidence; every serialised '
                'detection is therefore an accepted human detection',
            'human_candidate_pool': False,
            'human_candidate_conf_if_enabled': HUMAN_CANDIDATE_CONF,
            'ball_candidate_pool': False,
            'classes': CLASSES,
            'human_classes': list(HUMAN_CLASSES),
            'cache': 'bypassed (frame_id=None); no disk cache consulted',
        },
        'ball_settings_for_provenance_only': {
            'ball_candidate_conf': BALL_CANDIDATE_CONF,
            'ball_accept_conf': BALL_ACCEPT_CONF,
            'ball_dedupe_iou': BALL_DEDUPE_IOU,
            'note': 'recorded for provenance; no ball detection is serialised',
        },
        'environment': {
            'python': sys.version.split()[0],
            'numpy': np.__version__,
            'cv2_resolved': cv2.__version__,
        },
        'windows': [],
    }
    try:
        import supervision, scipy, ultralytics, torch
        manifest['environment'].update({
            'supervision': supervision.__version__, 'scipy': scipy.__version__,
            'ultralytics': ultralytics.__version__, 'torch': torch.__version__})
    except Exception:
        pass

    print(f'{"window":<30}{"start":>7}{"frames":>8}{"fps":>7}{"WxH":>12}'
          f'{"humans":>9}{"empty":>7}')
    for match, video, start in WINDOWS:
        vp = Path(video)
        if not vp.exists():
            raise SystemExit(f'missing source video: {video}')
        frames, fps, total, frames_hash = read_window(video, start, N_FRAMES)
        if len(frames) != N_FRAMES:
            raise SystemExit(f'{match}: got {len(frames)} frames, need {N_FRAMES}')
        h, w = frames[0].shape[:2]

        det = LocalDetector(args.model, confidence=CONF, imgsz=IMGSZ,
                            ball_candidate_pool=False, human_candidate_pool=False)
        lines, n_h, n_empty = [], 0, 0
        cls_count = {c: 0 for c in HUMAN_CLASSES}
        for i, f in enumerate(frames):
            # frame_id=None -> the detector's in-memory cache is not consulted
            # or populated, so nothing stale can enter the freeze.
            humans = [d for d in det.detect(f, None) if d['class'] in HUMAN_CLASSES]
            for d in humans:
                cls_count[d['class']] += 1
            n_h += len(humans)
            n_empty += (len(humans) == 0)
            lines.append(serialise(i, humans))

        text = '\n'.join(lines) + '\n'
        name = f'{match}_{start}.jsonl'
        print(f'{match[:28]:<30}{start:>7}{len(frames):>8}{fps:>7.2f}'
              f'{f"{w}x{h}":>12}{n_h:>9}{n_empty:>7}')
        manifest['windows'].append({
            'match': match, 'sequence': f'{match}_{start}',
            'source_video': video,
            'source_video_sha256': sha256_file(vp),
            'source_total_frames': total,
            'start_frame': start, 'frame_count': len(frames),
            'native_fps': round(float(fps), 6),
            'frame_width': w, 'frame_height': h,
            'decoded_frames_sha256': frames_hash,
            'human_detections': n_h,
            'frames_with_no_human': n_empty,
            'class_counts': cls_count,
            'detections_file': f'detections/{name}',
            'detections_sha256': sha256_text(text),
        })
        if not args.dry_run:
            det_dir.mkdir(parents=True, exist_ok=True)
            (det_dir / name).write_text(text, encoding='utf-8')

    if args.dry_run:
        print('\n(dry run -- nothing written)')
        return manifest

    (out / 'manifest.json').write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'\nwritten: {out}/manifest.json and {len(WINDOWS)} detection files')
    return manifest


if __name__ == '__main__':
    main()
