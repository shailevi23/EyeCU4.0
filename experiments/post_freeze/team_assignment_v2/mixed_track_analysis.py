"""
Phase 6 -- for every human-labeled MIXED_TRACK, run the same bimodality test
used for track #4 in rootcause_track4.py (full per-frame chest color trace,
KMeans k=2 on the track's own colors, check separation + contiguity), over
ALL its appearances (not just the label UI's 5 crops). Zero detector
inference -- cached tracks + plain video reads only.
"""
import os
import sys
import json
import pickle

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import cv2
import numpy as np
from sklearn.cluster import KMeans

from trackers.team_assigner import TeamAssigner

HERE = os.path.dirname(os.path.abspath(__file__))

MATCHES = [
    {'match_id': 'bayern_munich_3-1_chelsea',
     'video': os.path.join('input-videos', 'Bayern Munich 3-1 Chelsea.mp4'),
     'cache': os.path.join('demo_outputs', 'final_e2e_demo', 'cache', 'tracks_b3bacf1707645184.pkl'),
     'skip_frames': 2},
    {'match_id': 'chelsea_v_leeds_united',
     'video': os.path.join('input-videos', 'Chelsea v Leeds United.mp4'),
     'cache': os.path.join('experiments', 'post_freeze', 'team_assignment_v2',
                           'match2_cache', 'cache', 'tracks_7562185f8dd2d6c6.pkl'),
     'skip_frames': 2},
]


def analyze_track(video_path, cache_path, skip_frames, track_id):
    with open(cache_path, 'rb') as f:
        blob = pickle.load(f)
    tracks = blob['tracks']
    appearances = [i for i, fr in enumerate(tracks['players']) if track_id in fr]
    ta = TeamAssigner(num_teams=2)
    cap = cv2.VideoCapture(video_path)
    colors = []
    for idx in appearances:
        raw_idx = idx * skip_frames
        cap.set(cv2.CAP_PROP_POS_FRAMES, raw_idx)
        ok, frame = cap.read()
        if not ok:
            continue
        bbox = tracks['players'][idx][track_id]['bbox']
        colors.append(ta.get_player_color(frame, bbox))
    cap.release()
    colors = np.array(colors)
    if len(colors) < 4:
        return {'n_appearances': len(appearances), 'n_usable': len(colors),
                'classification': 'TOO_SHORT_TO_ANALYZE'}

    km = KMeans(n_clusters=2, init='k-means++', n_init=10, random_state=0)
    labels_ = km.fit_predict(colors)
    c0, c1 = km.cluster_centers_
    center_dist = float(np.linalg.norm(c0 - c1))
    n0, n1 = int((labels_ == 0).sum()), int((labels_ == 1).sum())
    minority_frac = min(n0, n1) / len(labels_)
    spread0 = float(np.linalg.norm(colors[labels_ == 0] - c0, axis=1).std()) if n0 > 1 else 0.0
    spread1 = float(np.linalg.norm(colors[labels_ == 1] - c1, axis=1).std()) if n1 > 1 else 0.0
    avg_spread = (spread0 + spread1) / 2 if (n0 > 1 and n1 > 1) else max(spread0, spread1, 1e-6)
    separation_ratio = center_dist / max(avg_spread, 1e-6)

    minority_label = 0 if n0 < n1 else 1
    minority_idxs = [appearances[i] for i in range(len(labels_)) if labels_[i] == minority_label]
    runs = []
    if minority_idxs:
        run_start = minority_idxs[0]
        prev = minority_idxs[0]
        for v in minority_idxs[1:]:
            if v - prev > 3:
                runs.append((run_start, prev))
                run_start = v
            prev = v
        runs.append((run_start, prev))
    longest_run_len = max((b - a + 1 for a, b in runs), default=0)
    contiguous_switch_like = bool(runs) and (longest_run_len / max(len(minority_idxs), 1) >= 0.7)
    is_bimodal = (minority_frac >= 0.15) and (separation_ratio >= 2.0)

    if is_bimodal and contiguous_switch_like:
        classification = 'ABRUPT_SUSTAINED_TRANSITION'
    elif is_bimodal:
        classification = 'INTERMITTENT_CONTAMINATION'
    else:
        classification = 'UNIMODAL_NO_CLEAR_SPLIT'

    return {
        'n_appearances': len(appearances), 'n_usable': len(colors),
        'center_distance_bgr': center_dist, 'separation_ratio': separation_ratio,
        'minority_fraction': minority_frac, 'minority_runs': runs,
        'longest_minority_run_len': longest_run_len,
        'contiguous_switch_like': contiguous_switch_like,
        'classification': classification,
    }


def main():
    labels = json.load(open(os.path.join(HERE, 'label_ui', 'labels.json')))
    manifest = json.load(open(os.path.join(HERE, 'label_ui', 'selection_manifest.json')))

    results = {}
    mixed_by_match = {}
    for match in manifest['matches']:
        mid = match['match_id']
        mixed_ids = [t['track_id'] for t in match['tracks']
                     if labels[f"{mid}:{t['track_id']}"] == 'MIXED_TRACK']
        mixed_by_match[mid] = len(mixed_ids)
        for tid in mixed_ids:
            print(f"analyzing {mid}:{tid} ...")
            r = analyze_track(match['video'], match['cache'], match['skip_frames'], tid)
            results[f"{mid}:{tid}"] = r
            print(f"  -> {r['classification']} (sep_ratio={r.get('separation_ratio', 'n/a')}, "
                  f"minority_frac={r.get('minority_fraction', 'n/a')})")

    track4_key = 'bayern_munich_3-1_chelsea:4'
    track4_label = labels.get(track4_key)
    out = {
        'mixed_track_count_total': sum(mixed_by_match.values()),
        'mixed_track_count_by_match': mixed_by_match,
        'per_track_analysis': results,
        'track4_human_label': track4_label,
        'track4_conclusion': (
            'TRACK_ID_CONTAMINATION VISUALLY CONFIRMED on this development example '
            '(blind human label = MIXED_TRACK, matching the computational hypothesis).'
            if track4_label == 'MIXED_TRACK' else
            f'Human label for track #4 was {track4_label}, NOT MIXED_TRACK -- the '
            'computational hypothesis (TRACK_ID_CONTAMINATION) is not visually '
            'confirmed by this blind label; see discrepancy discussion in FINAL_RESULTS.md.'
        ),
    }
    out_path = os.path.join(HERE, 'MIXED_TRACK_ANALYSIS.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {out_path}")
    print(f"track4_conclusion: {out['track4_conclusion']}")


if __name__ == '__main__':
    main()
