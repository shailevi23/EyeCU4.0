"""
Candidate C -- SigLIP change-point guard, exactly per the FROZEN definition.
Run inside the existing isolated siglip_env (no redownload). Extracts up to
15 temporally-distributed central-player crops across each track's FULL
cached lifetime (fresh from the original source video -- not the 5 label
crops, which are a coarser sample), embeds them, and applies the SAME frozen
change-point rule as Candidate B.
"""
import os
import sys
import time
import json
import pickle

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from changepoint import detect_change_point

import cv2
import numpy as np
from PIL import Image
import torch
from transformers import AutoProcessor, SiglipVisionModel

HERE = os.path.dirname(os.path.abspath(__file__))
MANIFEST_PATH = os.path.join(HERE, '..', 'team_assignment_v2', 'label_ui', 'selection_manifest.json')
CFG = json.load(open(os.path.join(HERE, 'candidate_config.json')))['candidate_C']
MAX_OBS = CFG['max_observations_per_track']
PAD = CFG['crop_fraction_padding']
MODEL_ID = CFG['model_id']

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


def padded_crop(frame, bbox):
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    x1p = max(0, int(x1 - w * PAD)); y1p = max(0, int(y1 - h * PAD))
    x2p = min(frame.shape[1], int(x2 + w * PAD)); y2p = min(frame.shape[0], int(y2 + h * PAD))
    crop = frame[y1p:y2p, x1p:x2p]
    return crop if crop.size else None


def main():
    manifest = json.load(open(MANIFEST_PATH, encoding='utf-8'))

    t_setup0 = time.time()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = SiglipVisionModel.from_pretrained(MODEL_ID)
    model.eval()
    setup_time = time.time() - t_setup0
    print(f"model loaded in {setup_time:.1f}s")

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
            sample_positions = np.linspace(0, len(idxs) - 1, min(len(idxs), MAX_OBS)).round().astype(int) if idxs else []
            sample_positions = sorted(set(sample_positions.tolist()))
            imgs, used_idxs = [], []
            for pos in sample_positions:
                idx = idxs[pos]
                raw_idx = idx * cfg['skip_frames']
                cap.set(cv2.CAP_PROP_POS_FRAMES, raw_idx)
                ok, frame = cap.read()
                if not ok:
                    continue
                bbox = tracks['players'][idx][tid]['bbox']
                crop = padded_crop(frame, bbox)
                if crop is None:
                    continue
                imgs.append(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
                used_idxs.append(idx)

            if imgs:
                with torch.no_grad():
                    inputs = processor(images=imgs, return_tensors='pt')
                    out = model(**inputs)
                    feats = out.pooler_output.numpy()
                feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)
            else:
                feats = np.zeros((0, 1))

            result = detect_change_point(feats, used_idxs)
            match_preds[tid] = bool(result['contaminated'])
            match_diag[tid] = {**result, 'n_usable_observations': len(used_idxs)}
        cap.release()

        predictions[mid] = match_preds
        diagnostics[mid] = match_diag
        runtimes[mid] = time.time() - t0
        n_flagged = sum(match_preds.values())
        print(f"[{mid}] {n_flagged}/{len(selected_tids)} flagged CONTAMINATED, runtime {runtimes[mid]:.1f}s")

    out = {'predictions': predictions, 'runtimes': runtimes, 'diagnostics': diagnostics,
           'model_load_time_s': setup_time, 'model_id': MODEL_ID}
    out_path = os.path.join(HERE, 'candidate_C_predictions.json')
    json.dump(out, open(out_path, 'w', encoding='utf-8'), indent=2)
    print(f"wrote {out_path}")


if __name__ == '__main__':
    main()
