"""
Phase 2 root-cause trace for track #4 (Bayern demo, NON-TEST).

Reads ONLY the existing tracks cache (demo_outputs/final_e2e_demo/cache/
tracks_b3bacf1707645184.pkl) and decodes the ORIGINAL source video frames by
index (plain video read, not detection) -- zero model inference. Reuses
TeamAssigner.get_player_color() unmodified so the color extraction here is
provably identical to production's own chest-ROI/median logic.

Outputs (this directory):
  - TRACK4_TRACE.json       per-appearance bbox/color/size for up to 15
                            uniformly spaced samples + full per-frame stats
  - track4_contact_sheet.jpg  one local contact sheet, chest crops only
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

CACHE_PATH = os.path.join('demo_outputs', 'final_e2e_demo', 'cache',
                           'tracks_b3bacf1707645184.pkl')
VIDEO_PATH = os.path.join('input-videos', 'Bayern Munich 3-1 Chelsea.mp4')
SKIP_FRAMES = 2
TRACK_ID = 4
OUT_DIR = os.path.dirname(__file__)


def imwrite_unicode(path, img, ext='.jpg'):
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        return False
    with open(path, 'wb') as f:
        f.write(buf.tobytes())
    return True


def chest_crop(frame, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    player_img = frame[y1:y2, x1:x2]
    if player_img.size == 0:
        return None
    h, w = player_img.shape[:2]
    y_start, y_end = int(h * 0.15), int(h * 0.50)
    x_start, x_end = int(w * 0.25), int(w * 0.75)
    torso = player_img[y_start:y_end, x_start:x_end]
    return torso if torso.size else player_img


def main():
    with open(CACHE_PATH, 'rb') as f:
        blob = pickle.load(f)
    tracks = blob['tracks']
    n = len(tracks['players'])

    appearances = [i for i, fr in enumerate(tracks['players']) if TRACK_ID in fr]
    print(f"track {TRACK_ID}: {len(appearances)} appearances out of {n} processed frames "
          f"(first={appearances[0]}, last={appearances[-1]})")

    ta = TeamAssigner(num_teams=2)  # only used for get_player_color(), no fitting here

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"STOP: could not open {VIDEO_PATH}")
        sys.exit(1)

    # Full per-frame trace (used for the bimodality test), not just the 15 samples.
    full_trace = []
    for idx in appearances:
        raw_idx = idx * SKIP_FRAMES
        cap.set(cv2.CAP_PROP_POS_FRAMES, raw_idx)
        ok, frame = cap.read()
        if not ok:
            continue
        bbox = tracks['players'][idx][TRACK_ID]['bbox']
        color = ta.get_player_color(frame, bbox)  # identical production chest-ROI/median call
        x1, y1, x2, y2 = bbox
        full_trace.append({
            'processed_idx': idx, 'raw_idx': raw_idx,
            'bbox': [float(v) for v in bbox],
            'bbox_w': float(x2 - x1), 'bbox_h': float(y2 - y1),
            'color_bgr_median': [float(v) for v in color],
        })

    colors = np.array([t['color_bgr_median'] for t in full_trace])
    brightness = colors.max(axis=1)

    # --- bimodality test: does track 4's OWN color history split into two
    # well-separated, both-substantial clusters, or is it one consistent mode?
    km = KMeans(n_clusters=2, init='k-means++', n_init=10, random_state=0)
    labels = km.fit_predict(colors)
    c0, c1 = km.cluster_centers_
    center_dist = float(np.linalg.norm(c0 - c1))
    n0, n1 = int((labels == 0).sum()), int((labels == 1).sum())
    minority_frac = min(n0, n1) / len(labels)
    # intra-cluster spread, to judge whether center_dist is "large" relative to noise
    spread0 = float(np.linalg.norm(colors[labels == 0] - c0, axis=1).std()) if n0 > 1 else 0.0
    spread1 = float(np.linalg.norm(colors[labels == 1] - c1, axis=1).std()) if n1 > 1 else 0.0
    avg_spread = (spread0 + spread1) / 2 if (n0 > 1 and n1 > 1) else max(spread0, spread1, 1e-6)
    separation_ratio = center_dist / max(avg_spread, 1e-6)

    is_bimodal = (minority_frac >= 0.15) and (separation_ratio >= 2.0)

    # contiguity of the minority mode: one contiguous block => plausible single
    # ID-switch event; scattered singletons => plausible transient noise, not a switch
    minority_label = 0 if n0 < n1 else 1
    minority_idxs = [full_trace[i]['processed_idx'] for i in range(len(labels)) if labels[i] == minority_label]
    runs = []
    if minority_idxs:
        run_start = minority_idxs[0]
        prev = minority_idxs[0]
        for v in minority_idxs[1:]:
            if v - prev > 3:  # allow small tracker gaps within one "run"
                runs.append((run_start, prev))
                run_start = v
            prev = v
        runs.append((run_start, prev))
    longest_run_len = max((b - a + 1 for a, b in runs), default=0)
    contiguous_switch_like = bool(runs) and (longest_run_len / max(len(minority_idxs), 1) >= 0.7)

    if is_bimodal and contiguous_switch_like:
        hypothesis = 'TRACK_ID_CONTAMINATION'
        evidence = (f"Track {TRACK_ID}'s own chest-color history splits into two clusters "
                    f"{center_dist:.1f} BGR-units apart ({separation_ratio:.1f}x the average "
                    f"intra-cluster spread), with the minority mode ({n0 if minority_label==0 else n1}"
                    f"/{len(labels)} frames, {minority_frac*100:.0f}%) concentrated in "
                    f"{len(runs)} contiguous run(s), longest {longest_run_len} frames "
                    f"({minority_idxs[0]}-{minority_idxs[-1]} span) -- consistent with a single "
                    f"sustained identity swap partway through the track, not scattered noise.")
    elif is_bimodal:
        hypothesis = 'BOTH_POSSIBLE'
        evidence = (f"Track {TRACK_ID}'s chest colors do split into two separated clusters "
                    f"({center_dist:.1f} BGR-units, {separation_ratio:.1f}x spread, minority "
                    f"{minority_frac*100:.0f}%), but the minority-mode frames are scattered across "
                    f"{len(runs)} run(s) (longest {longest_run_len}/{len(minority_idxs)} frames) rather "
                    f"than one sustained block -- consistent with either a brief/occlusion-driven "
                    f"contamination OR classifier noise on individual frames; evidence does not "
                    f"cleanly separate the two hypotheses.")
    else:
        hypothesis = 'TEAM_CLASSIFIER_ERROR'
        evidence = (f"Track {TRACK_ID}'s chest-color history is effectively unimodal: the best 2-way "
                    f"split only isolates a minority of {minority_frac*100:.0f}% of frames at "
                    f"{separation_ratio:.1f}x the intra-cluster spread (below the "
                    f"15%/2.0x bimodality bar). The same jersey appearance persists across the whole "
                    f"track, so a wrong team label is attributable to the CLASSIFIER/cluster-fit "
                    f"stage (e.g. an early-frame color-model fit, or an arbitrary KMeans team-index "
                    f"permutation across runs -- see CURRENT_IMPLEMENTATION.md D/E), not to tracking.")

    print(f"\nBimodality test: center_dist={center_dist:.1f}, separation_ratio={separation_ratio:.2f}, "
          f"minority_frac={minority_frac:.2f}, contiguous_switch_like={contiguous_switch_like}")
    print(f"HYPOTHESIS: {hypothesis}")
    print(evidence)

    # --- up to 15 uniformly distributed samples for the report/contact sheet
    n_samples = min(15, len(appearances))
    sample_positions = np.linspace(0, len(appearances) - 1, n_samples).round().astype(int)
    sample_positions = sorted(set(sample_positions.tolist()))
    samples = []
    crops = []
    for pos in sample_positions:
        idx = appearances[pos]
        raw_idx = idx * SKIP_FRAMES
        cap.set(cv2.CAP_PROP_POS_FRAMES, raw_idx)
        ok, frame = cap.read()
        if not ok:
            continue
        bbox = tracks['players'][idx][TRACK_ID]['bbox']
        crop = chest_crop(frame, bbox)
        color = ta.get_player_color(frame, bbox)
        x1, y1, x2, y2 = bbox
        samples.append({
            'raw_frame_index': raw_idx,
            'processed_frame_index': idx,
            'bbox': [float(v) for v in bbox],
            'bbox_size_px': [float(x2 - x1), float(y2 - y1)],
            'chest_color_bgr_median': [float(v) for v in color],
            'brightness_max_channel': float(max(color)),
        })
        crops.append((idx, crop))
    cap.release()

    # contact sheet: crops in a single row, resized to a common height, labeled with processed idx
    tile_h = 96
    tiles = []
    for idx, crop in crops:
        if crop is None or crop.size == 0:
            tile = np.zeros((tile_h, 48, 3), dtype=np.uint8)
        else:
            scale = tile_h / crop.shape[0]
            tile = cv2.resize(crop, (max(1, int(crop.shape[1] * scale)), tile_h))
        tile = cv2.copyMakeBorder(tile, 20, 4, 2, 2, cv2.BORDER_CONSTANT, value=(30, 30, 30))
        cv2.putText(tile, f"f{idx}", (2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(tile)
    max_h = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, max_h - t.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(30, 30, 30)) for t in tiles]
    sheet = np.concatenate(tiles, axis=1)
    sheet_path = os.path.join(OUT_DIR, 'track4_contact_sheet.jpg')
    written = imwrite_unicode(sheet_path, sheet)
    print(f"\ncontact sheet {'written' if written else 'FAILED'}: {sheet_path}")

    report = {
        'track_id': TRACK_ID,
        'video': VIDEO_PATH,
        'cache': CACHE_PATH,
        'n_appearances_total': len(appearances),
        'n_processed_frames': n,
        'first_appearance_processed_idx': appearances[0],
        'last_appearance_processed_idx': appearances[-1],
        'reported_team_in_last_production_run': 2,  # from reports/player_statistics.json
        'bimodality_test': {
            'center_distance_bgr': center_dist,
            'separation_ratio': separation_ratio,
            'minority_fraction': minority_frac,
            'minority_runs': runs,
            'longest_minority_run_len': longest_run_len,
            'contiguous_switch_like': contiguous_switch_like,
        },
        'hypothesis': hypothesis,
        'evidence': evidence,
        'samples': samples,
    }
    out_path = os.path.join(OUT_DIR, 'TRACK4_TRACE.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    print(f"trace written: {out_path}")


if __name__ == '__main__':
    main()
