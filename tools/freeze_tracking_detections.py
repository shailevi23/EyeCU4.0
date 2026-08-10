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
CONF = 0.25                    # accepted / public human threshold

# Frozen detector evidence floor for the candidate view. This is a benchmark
# evidence-store floor -- NOT a production acceptance threshold, NOT a ByteTrack
# threshold, and NOT a tuned tracking parameter. Modern ByteTrackTracker could
# theoretically consume detections below 0.01; EyeCU-Tracking-Val-v1 defines
# 0.01 as its frozen detector evidence floor.
CANDIDATE_FLOOR = 0.01

# YOLO26 is end-to-end: the Detect head applies a hard top-k
# (get_topk_index(scores, self.max_det), max_det=300) across ALL classes before
# confidence filtering. If a frame returns this many boxes the candidate
# universe may be truncated, so every frame's raw box count is recorded.
MAX_DET = 300

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



def canonical(dets):
    """Order-independent comparison key: lowering the inference floor may change
    the detector's native ordering, so the invariance check must not depend on
    it."""
    return sorted((d['class'], round(float(d['confidence']), 12),
                   tuple(round(float(v), 12) for v in d['bbox'])) for d in dets)


def freeze_candidates(args, out, det_dir):
    """
    Freeze the >= CANDIDATE_FLOOR evidence view and prove its >= CONF subset
    reproduces the already-versioned accepted freeze exactly.
    """
    import cv2
    mpath = out / 'manifest.json'
    if not mpath.exists():
        raise SystemExit('accepted freeze missing; run --view accepted first')
    man = json.loads(mpath.read_text(encoding='utf-8'))
    cand_dir = out / 'candidates'

    # Record the tool's own content hash, not just git HEAD. HEAD names the last
    # commit, which is wrong whenever the tool itself is still uncommitted -- a
    # defect this manifest hit once already. The content hash cannot drift from
    # the code that actually ran.
    freeze_tool_commit = subprocess.run(['git', 'rev-parse', 'HEAD'],
                                        capture_output=True, text=True).stdout.strip()
    freeze_tool_sha256 = hashlib.sha256(
        Path(__file__).read_text(encoding='utf-8').encode('utf-8')).hexdigest()
    dirty = subprocess.run(['git', 'status', '--porcelain', '--', __file__],
                           capture_output=True, text=True).stdout.strip()
    if dirty:
        print('WARNING: this tool has uncommitted changes; freeze_tool_commit will '
              'name the previous commit. freeze_tool_sha256 records what actually ran.')
    print(f'{"window":<30}{"cand":>8}{"acc":>7}{"maxRaw":>8}{"@cap":>6}{"90%cap":>8}'
          f'{"mismatch":>10}')

    total_mismatch, saturated, near = 0, 0, 0
    for w in man['windows']:
        match, start = w['match'], w['start_frame']
        video = w['source_video']
        frames, fps, total, frames_hash = read_window(video, start, N_FRAMES)
        if len(frames) != N_FRAMES:
            raise SystemExit(f'{match}: got {len(frames)} frames')
        if frames_hash != w['decoded_frames_sha256']:
            raise SystemExit(f'{match}: decoded pixels differ from the accepted freeze '
                             f'-- refusing to build a candidate view on different frames')

        det = LocalDetector(args.model, confidence=CANDIDATE_FLOOR, imgsz=IMGSZ,
                            ball_candidate_pool=False, human_candidate_pool=False)
        accepted_rows = [json.loads(l) for l in
                         (out / w['detections_file']).read_text(encoding='utf-8').splitlines()
                         if l.strip()]
        lines, n_c, raw_counts, mism = [], 0, [], 0
        for i, f in enumerate(frames):
            alld = det.detect(f, None)          # every class; == raw model boxes
            raw_counts.append(len(alld))
            humans = [d for d in alld if d['class'] in HUMAN_CLASSES]
            n_c += len(humans)
            lines.append(serialise(i, humans))
            # accepted-subset invariance, canonical (order-independent)
            subset = [d for d in humans if d['confidence'] >= CONF]
            if canonical(subset) != canonical(accepted_rows[i]['detections']):
                mism += 1
        raw = np.array(raw_counts)
        at_cap = int((raw >= MAX_DET).sum())
        near_cap = int((raw >= 0.9 * MAX_DET).sum())
        saturated += at_cap; near += near_cap; total_mismatch += mism

        text = '\n'.join(lines) + '\n'
        name = f'{match}_{start}.jsonl'
        print(f'{match[:28]:<30}{n_c:>8}{w["human_detections"]:>7}{int(raw.max()):>8}'
              f'{at_cap:>6}{near_cap:>8}{mism:>10}')
        w.update({
            'candidate_file': f'candidates/{name}',
            'candidate_detections': n_c,
            'candidate_sha256': sha256_text(text),
            'candidate_raw_boxes_max': int(raw.max()),
            'candidate_raw_boxes_mean': round(float(raw.mean()), 2),
            'candidate_frames_at_max_det': at_cap,
            'candidate_frames_within_90pct_max_det': near_cap,
            'accepted_subset_mismatched_frames': mism,
        })
        if not args.dry_run:
            cand_dir.mkdir(parents=True, exist_ok=True)
            (cand_dir / name).write_text(text, encoding='utf-8')

    print(f'\nframes at max_det ({MAX_DET}): {saturated}   within 90%: {near}')
    if saturated or near:
        raise SystemExit('STOP: detector top-k cap reached; the candidate universe '
                         'may be truncated. Do not raise max_det silently.')
    print(f'accepted-subset invariance mismatched frames: {total_mismatch}')
    if total_mismatch:
        raise SystemExit('STOP: lowering the inference floor changed the >=0.25 '
                         'subset. The accepted freeze was NOT overwritten.')

    man['version'] = '1.1'
    man['detector_source_commit'] = man.pop('code_commit', man.get('detector_source_commit'))
    man['freeze_tool_commit'] = freeze_tool_commit
    man['freeze_tool_sha256'] = freeze_tool_sha256
    man['accepted_human_threshold'] = CONF
    man['candidate_human_floor'] = CANDIDATE_FLOOR
    man['views'] = {
        'accepted': 'confidence >= %s -- EyeCU public/reporting boundary; the '
                    'detector output production actually exposes' % CONF,
        'candidate': 'confidence >= %s -- frozen detector EVIDENCE STORE for '
                     'tracker association, before any tracker-specific '
                     'confidence filtering. NOT a production threshold.' % CANDIDATE_FLOOR,
        'relationship': 'accepted is exactly the >= %s subset of candidate, '
                        'verified frame by frame on all %d frames' % (CONF, N_FRAMES * 4),
    }
    man['evidence_floor_note'] = (
        'Modern ByteTrackTracker could theoretically consume detections below '
        '0.01; EyeCU-Tracking-Val-v1 defines 0.01 as its frozen detector '
        'evidence floor. 0.01 is not claimed to be algorithmically optimal.')
    man['max_det_mechanism'] = (
        'YOLO26 is end-to-end (no NMS). trackers head applies a hard top-k: '
        'get_topk_index(scores, self.max_det) with max_det=%d, across all four '
        'classes before confidence filtering.' % MAX_DET)
    if not args.dry_run:
        mpath.write_text(json.dumps(man, indent=2, ensure_ascii=False), encoding='utf-8')
        print(f'manifest updated to v1.1')
    return man


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default='data/tracking_val_v1')
    ap.add_argument('--model', default=MODEL)
    ap.add_argument('--view', choices=['accepted', 'candidate'], default='accepted',
                    help="'accepted' freezes the >=0.25 public view; 'candidate' "
                         "freezes the >=0.01 evidence view and verifies that its "
                         ">=0.25 subset reproduces the accepted freeze exactly.")
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
    if args.view == 'candidate':
        return freeze_candidates(args, out, det_dir)
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
