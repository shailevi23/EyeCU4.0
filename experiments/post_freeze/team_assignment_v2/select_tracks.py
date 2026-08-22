"""
Phase 3 deterministic track selection + crop extraction for the team-
assignment development benchmark.

Selection rule (prediction-independent, deterministic):
  1. keep player tracks with appearance_count >= MIN_LIFETIME
  2. sort by appearance_count DESC, tie-break by track_id ASC
  3. take up to MAX_TRACKS_PER_MATCH

Goalkeeper/referee tracks are never considered (only tracks['players']).
Track #4 in the Bayern match is not special-cased -- it is included only if
this rule would select it on its own merits.

For each selected track, 5 temporally-uniform "central player crops" (bbox +
10% padding, clipped to frame) are saved for the label UI. No detector/model
call -- crops come from the ORIGINAL source video decoded by frame index,
paired with the already-cached bbox.
"""
import os
import sys
import json
import pickle
import hashlib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import cv2
import numpy as np

MIN_LIFETIME = 15
MAX_TRACKS_PER_MATCH = 30
HERE = os.path.dirname(__file__)
CROPS_DIR = os.path.join(HERE, 'label_ui', 'crops')

MATCHES = [
    {
        'match_id': 'bayern_munich_3-1_chelsea',
        'video': os.path.join('input-videos', 'Bayern Munich 3-1 Chelsea.mp4'),
        'cache': os.path.join('demo_outputs', 'final_e2e_demo', 'cache',
                              'tracks_b3bacf1707645184.pkl'),
        'skip_frames': 2,
    },
    {
        'match_id': 'chelsea_v_leeds_united',
        'video': os.path.join('input-videos', 'Chelsea v Leeds United.mp4'),
        'cache': os.path.join('experiments', 'post_freeze', 'team_assignment_v2',
                              'match2_cache', 'cache', 'tracks_7562185f8dd2d6c6.pkl'),
        'skip_frames': 2,
    },
]


def imwrite_unicode(path, img, ext='.jpg'):
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        return False
    with open(path, 'wb') as f:
        f.write(buf.tobytes())
    return True


def padded_crop(frame, bbox, pad_frac=0.10):
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    x1p = max(0, int(x1 - w * pad_frac))
    y1p = max(0, int(y1 - h * pad_frac))
    x2p = min(frame.shape[1], int(x2 + w * pad_frac))
    y2p = min(frame.shape[0], int(y2 + h * pad_frac))
    return frame[y1p:y2p, x1p:x2p]


def select_for_match(match):
    with open(match['cache'], 'rb') as f:
        blob = pickle.load(f)
    tracks = blob['tracks']
    n = len(tracks['players'])

    appearances = {}
    for idx in range(n):
        for tid in tracks['players'][idx].keys():
            appearances.setdefault(tid, []).append(idx)

    candidates = [(tid, idxs) for tid, idxs in appearances.items() if len(idxs) >= MIN_LIFETIME]
    candidates.sort(key=lambda t: (-len(t[1]), t[0]))
    selected = candidates[:MAX_TRACKS_PER_MATCH]

    print(f"[{match['match_id']}] {len(appearances)} player track ids total, "
          f"{len(candidates)} with >= {MIN_LIFETIME} appearances, "
          f"selecting {len(selected)}")

    cap = cv2.VideoCapture(match['video'])
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {match['video']}")

    match_dir = os.path.join(CROPS_DIR, match['match_id'])
    os.makedirs(match_dir, exist_ok=True)

    track_entries = []
    for tid, idxs in selected:
        n_samples = min(5, len(idxs))
        positions = np.linspace(0, len(idxs) - 1, n_samples).round().astype(int)
        positions = sorted(set(positions.tolist()))
        track_dir = os.path.join(match_dir, str(tid))
        os.makedirs(track_dir, exist_ok=True)
        crop_paths = []
        for i, pos in enumerate(positions):
            idx = idxs[pos]
            raw_idx = idx * match['skip_frames']
            cap.set(cv2.CAP_PROP_POS_FRAMES, raw_idx)
            ok, frame = cap.read()
            if not ok:
                continue
            bbox = tracks['players'][idx][tid]['bbox']
            crop = padded_crop(frame, bbox)
            if crop.size == 0:
                continue
            rel_path = f"crops/{match['match_id']}/{tid}/crop_{i}.jpg"
            abs_path = os.path.join(HERE, 'label_ui', rel_path.replace('/', os.sep))
            if imwrite_unicode(abs_path, crop):
                crop_paths.append({'rel_path': rel_path, 'processed_idx': idx, 'raw_idx': raw_idx})
        track_entries.append({
            'track_id': tid,
            'appearance_count': len(idxs),
            'first_processed_idx': idxs[0],
            'last_processed_idx': idxs[-1],
            'crops': crop_paths,
        })
    cap.release()
    return track_entries


def main():
    manifest = {'min_lifetime': MIN_LIFETIME, 'max_tracks_per_match': MAX_TRACKS_PER_MATCH,
                'selection_rule': ('player tracks only (goalkeepers/referees excluded); '
                                   'keep appearance_count >= min_lifetime; sort by '
                                   'appearance_count DESC then track_id ASC; take top '
                                   'max_tracks_per_match. Selection uses ONLY bbox '
                                   'presence counts -- no color/prediction/team info.'),
                'matches': []}
    for match in MATCHES:
        entries = select_for_match(match)
        manifest['matches'].append({
            'match_id': match['match_id'],
            'video': match['video'],
            'cache': match['cache'],
            'skip_frames': match['skip_frames'],
            'n_tracks_selected': len(entries),
            'tracks': entries,
        })

    manifest_path = os.path.join(HERE, 'label_ui', 'selection_manifest.json')
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)

    with open(manifest_path, 'rb') as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    with open(manifest_path + '.sha256', 'w') as f:
        f.write(sha + '\n')

    total_tracks = sum(m['n_tracks_selected'] for m in manifest['matches'])
    print(f"\nWrote {manifest_path}")
    print(f"Total tracks selected across {len(manifest['matches'])} matches: {total_tracks}")
    print(f"SHA256: {sha}")


if __name__ == '__main__':
    main()
