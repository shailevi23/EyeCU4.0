"""
Candidate B -- robust color tracklet, exactly per the FROZEN definition in
CANDIDATE_DEFINITIONS.md / candidate_config.json (written and hashed before
this was run). Do not change any parameter here after seeing scores.
"""
import os
import sys
import time
import pickle
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

HERE = os.path.dirname(os.path.abspath(__file__))

CFG = json.load(open(os.path.join(HERE, 'candidate_config.json')))['candidate_B']
ROI = CFG['roi_fractions']
QR = CFG['quality_rejection']
MAX_OBS = CFG['max_observations_per_track']

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


def chest_roi(frame, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    h, w = crop.shape[:2]
    ys, ye = int(h * ROI['y_start']), int(h * ROI['y_end'])
    xs, xe = int(w * ROI['x_start']), int(w * ROI['x_end'])
    torso = crop[ys:ye, xs:xe]
    return torso if torso.size else None


def descriptor_or_none(torso_bgr):
    """Returns the frozen 6-D [L, a, b, S, sin2H, cos2H] descriptor, or None
    if this observation fails the frozen quality-rejection rule."""
    if torso_bgr is None:
        return None
    n_pixels = torso_bgr.shape[0] * torso_bgr.shape[1]  # h*w, NOT .size (which also counts the 3 channels)
    if n_pixels < QR['min_pixels']:
        return None

    hsv = cv2.cvtColor(torso_bgr, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0].astype(np.float64), hsv[..., 1].astype(np.float64), hsv[..., 2].astype(np.float64)
    lo, hi = QR['green_hue_range_opencv_0_179']
    is_green = (H >= lo) & (H <= hi) & (S >= QR['green_min_saturation']) & (V >= QR['green_min_value'])
    keep = ~is_green

    total_px = H.size
    if keep.sum() / total_px < QR['min_non_green_fraction']:
        return None

    lab = cv2.cvtColor(torso_bgr, cv2.COLOR_BGR2LAB).astype(np.float64)
    L = lab[..., 0][keep]
    a = lab[..., 1][keep]
    b = lab[..., 2][keep]
    S_keep = S[keep]
    H_keep = H[keep]
    if L.size == 0:
        return None

    H_rad = H_keep * (np.pi / 90.0)  # OpenCV hue 0-179 spans 360 deg -> *2 before radians
    sin2H = np.median(np.sin(H_rad))
    cos2H = np.median(np.cos(H_rad))

    return np.array([np.median(L), np.median(a), np.median(b), np.median(S_keep), sin2H, cos2H])


def main():
    predictions = {}
    runtimes = {}
    for match in MATCHES:
        t0 = time.time()
        with open(match['cache'], 'rb') as f:
            blob = pickle.load(f)
        tracks = blob['tracks']
        n = len(tracks['players'])

        appearances = {}
        for idx in range(n):
            for tid in tracks['players'][idx].keys():
                appearances.setdefault(tid, []).append(idx)

        cap = cv2.VideoCapture(match['video'])
        track_descriptors = {}
        for tid, idxs in appearances.items():
            # gather usable observations, temporally spread, up to MAX_OBS
            usable = []
            sample_positions = np.linspace(0, len(idxs) - 1, min(len(idxs), MAX_OBS * 3)).round().astype(int)
            sample_positions = sorted(set(sample_positions.tolist()))
            for pos in sample_positions:
                if len(usable) >= MAX_OBS:
                    break
                idx = idxs[pos]
                raw_idx = idx * match['skip_frames']
                cap.set(cv2.CAP_PROP_POS_FRAMES, raw_idx)
                ok, frame = cap.read()
                if not ok:
                    continue
                bbox = tracks['players'][idx][tid]['bbox']
                torso = chest_roi(frame, bbox)
                desc = descriptor_or_none(torso)
                if desc is not None:
                    usable.append(desc)
            if usable:
                track_descriptors[tid] = np.median(np.stack(usable, axis=0), axis=0)
        cap.release()

        tids = sorted(track_descriptors.keys())
        if len(tids) < 2:
            # Not enough tracks with a usable descriptor to even fit k=2 --
            # this is a genuine data-availability limitation (see disclosure
            # in FINAL_RESULTS.md), not something patched here by loosening
            # the frozen min_pixels threshold after seeing it fail.
            match_preds = {}
            print(f"[{match['match_id']}] WARNING: only {len(tids)} tracks had a usable "
                  f"descriptor (need >=2 to cluster) -- emitting 0 predictions for this match.")
        else:
            X = np.stack([track_descriptors[tid] for tid in tids], axis=0)
            Xs = StandardScaler().fit_transform(X)
            km = KMeans(n_clusters=2, n_init=20, random_state=0)
            cluster_labels = km.fit_predict(Xs)
            match_preds = {tid: int(cluster_labels[i]) + 1 for i, tid in enumerate(tids)}
        predictions[match['match_id']] = match_preds
        runtimes[match['match_id']] = time.time() - t0
        print(f"[{match['match_id']}] {len(tids)}/{len(appearances)} tracks got a usable descriptor, "
              f"runtime {runtimes[match['match_id']]:.2f}s")

    out = {'predictions': predictions, 'runtimes': runtimes}
    out_path = os.path.join(HERE, 'candidate_B_predictions.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print(f"wrote {out_path}")


if __name__ == '__main__':
    main()
