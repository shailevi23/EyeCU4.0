"""
Candidate B -- robust jersey change-point guard, exactly per the FROZEN
definition in CANDIDATE_DEFINITIONS.md / candidate_config.json.
"""
import os
import sys
import time
import json
import pickle

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import cv2
import numpy as np

from changepoint import detect_change_point

HERE = os.path.dirname(os.path.abspath(__file__))
MANIFEST_PATH = os.path.join(HERE, '..', 'team_assignment_v2', 'label_ui', 'selection_manifest.json')
CFG = json.load(open(os.path.join(HERE, 'candidate_config.json')))['candidate_B']
ROI = CFG['roi_fractions']
QR = CFG['quality_rejection']
MAX_OBS = CFG['max_observations_per_track']

MATCHES = {
    'bayern_munich_3-1_chelsea': {
        'video': os.path.join('input-videos', 'Bayern Munich 3-1 Chelsea.mp4'),
        'cache': os.path.join('demo_outputs', 'final_e2e_demo', 'cache', 'tracks_b3bacf1707645184.pkl'),
        'skip_frames': 2},
    'chelsea_v_leeds_united': {
        'video': os.path.join('input-videos', 'Chelsea v Leeds United.mp4'),
        'cache': os.path.join('experiments', 'post_freeze', 'team_assignment_v2',
                              'match2_cache', 'cache', 'tracks_7562185f8dd2d6c6.pkl'),
        'skip_frames': 2},
}


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
    if torso_bgr is None:
        return None
    n_pixels = torso_bgr.shape[0] * torso_bgr.shape[1]
    if n_pixels < QR['min_pixels']:
        return None
    hsv = cv2.cvtColor(torso_bgr, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0].astype(np.float64), hsv[..., 1].astype(np.float64), hsv[..., 2].astype(np.float64)
    lo, hi = QR['green_hue_range_opencv_0_179']
    is_green = (H >= lo) & (H <= hi) & (S >= QR['green_min_saturation']) & (V >= QR['green_min_value'])
    keep = ~is_green
    if keep.sum() / H.size < QR['min_non_green_fraction']:
        return None
    lab = cv2.cvtColor(torso_bgr, cv2.COLOR_BGR2LAB).astype(np.float64)
    L, a, b = lab[..., 0][keep], lab[..., 1][keep], lab[..., 2][keep]
    if L.size == 0:
        return None
    S_keep, H_keep = S[keep], H[keep]
    H_rad = H_keep * (np.pi / 90.0)
    return np.array([np.median(L), np.median(a), np.median(b),
                      np.median(S_keep), np.median(np.sin(H_rad)), np.median(np.cos(H_rad))])


def main():
    manifest = json.load(open(MANIFEST_PATH, encoding='utf-8'))
    predictions = {}
    runtimes = {}
    diagnostics = {}

    for match in manifest['matches']:
        mid = match['match_id']
        cfg = MATCHES[mid]
        t0 = time.time()
        with open(cfg['cache'], 'rb') as f:
            blob = pickle.load(f)
        tracks = blob['tracks']
        n = len(tracks['players'])

        appearances = {}
        for idx in range(n):
            for tid in tracks['players'][idx].keys():
                appearances.setdefault(tid, []).append(idx)

        selected_tids = [t['track_id'] for t in match['tracks']]
        cap = cv2.VideoCapture(cfg['video'])
        match_preds = {}
        match_diag = {}
        for tid in selected_tids:
            idxs = appearances.get(tid, [])
            sample_positions = np.linspace(0, len(idxs) - 1,
                                           min(len(idxs), MAX_OBS * 3)).round().astype(int) if idxs else []
            sample_positions = sorted(set(sample_positions.tolist()))
            usable_descs, usable_idxs = [], []
            for pos in sample_positions:
                if len(usable_descs) >= MAX_OBS:
                    break
                idx = idxs[pos]
                raw_idx = idx * cfg['skip_frames']
                cap.set(cv2.CAP_PROP_POS_FRAMES, raw_idx)
                ok, frame = cap.read()
                if not ok:
                    continue
                bbox = tracks['players'][idx][tid]['bbox']
                desc = descriptor_or_none(chest_roi(frame, bbox))
                if desc is not None:
                    usable_descs.append(desc)
                    usable_idxs.append(idx)
            result = detect_change_point(usable_descs, usable_idxs)
            match_preds[tid] = bool(result['contaminated'])
            match_diag[tid] = {**result, 'n_usable_observations': len(usable_descs)}
        cap.release()

        predictions[mid] = match_preds
        diagnostics[mid] = match_diag
        runtimes[mid] = time.time() - t0
        n_flagged = sum(match_preds.values())
        n_zero_obs = sum(1 for d in match_diag.values() if d['n_usable_observations'] == 0)
        print(f"[{mid}] {n_flagged}/{len(selected_tids)} flagged CONTAMINATED, "
              f"{n_zero_obs}/{len(selected_tids)} had ZERO usable observations, "
              f"runtime {runtimes[mid]:.1f}s")

    out = {'predictions': predictions, 'runtimes': runtimes, 'diagnostics': diagnostics}
    out_path = os.path.join(HERE, 'candidate_B_predictions.json')
    json.dump(out, open(out_path, 'w', encoding='utf-8'), indent=2)
    print(f"wrote {out_path}")


if __name__ == '__main__':
    main()
