"""
Candidate C -- SigLIP appearance embeddings, exactly per the FROZEN
definition in CANDIDATE_DEFINITIONS.md / candidate_config.json. Feature
extraction only (no fine-tuning, no text prompting). Runs inside the
isolated C:/eyecu_siglip_env (kept outside the deep Hebrew-path project
directory only because Windows MAX_PATH broke torch's install there --
purely a filesystem workaround, not a change to what gets run or how).

Detector/tracker/selector inference is NOT used here -- this script only
reads the crop JPGs already extracted for human labeling.
"""
import os
import sys
import time
import json

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
from PIL import Image
import torch
from transformers import AutoProcessor, SiglipVisionModel
from sklearn.cluster import KMeans

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_ID = 'google/siglip-base-patch16-224'
MAX_OBS = 9  # frozen cap; the label crops only ever provide 5 anyway


def load_manifest():
    with open(os.path.join(HERE, 'label_ui', 'selection_manifest.json'), 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    t_setup0 = time.time()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = SiglipVisionModel.from_pretrained(MODEL_ID)
    model.eval()
    setup_time = time.time() - t_setup0
    print(f"model loaded in {setup_time:.1f}s")

    manifest = load_manifest()
    predictions = {}
    runtimes = {}

    for match in manifest['matches']:
        mid = match['match_id']
        t0 = time.time()
        track_descriptors = {}
        for t in match['tracks']:
            tid = t['track_id']
            crop_paths = [os.path.join(HERE, 'label_ui', c['rel_path'].replace('/', os.sep))
                          for c in t['crops'][:MAX_OBS]]
            imgs = []
            for p in crop_paths:
                try:
                    imgs.append(Image.open(p).convert('RGB'))
                except Exception:
                    continue
            if not imgs:
                continue
            with torch.no_grad():
                inputs = processor(images=imgs, return_tensors='pt')
                out = model(**inputs)
                # pooled_output: (n_crops, hidden_dim)
                feats = out.pooler_output.numpy()
            feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)
            track_desc = np.median(feats, axis=0)
            track_desc = track_desc / (np.linalg.norm(track_desc) + 1e-8)
            track_descriptors[tid] = track_desc

        tids = sorted(track_descriptors.keys())
        X = np.stack([track_descriptors[tid] for tid in tids], axis=0)
        km = KMeans(n_clusters=2, n_init=20, random_state=0)
        cluster_labels = km.fit_predict(X)
        match_preds = {tid: int(cluster_labels[i]) + 1 for i, tid in enumerate(tids)}
        predictions[mid] = match_preds
        runtimes[mid] = time.time() - t0
        print(f"[{mid}] {len(tids)}/{len(match['tracks'])} tracks embedded, "
              f"runtime {runtimes[mid]:.2f}s (excl. one-time model load)")

    out = {
        'predictions': predictions,
        'runtimes': runtimes,
        'model_load_time_s': setup_time,
        'model_id': MODEL_ID,
        'embedding_dim': int(X.shape[1]),
    }
    out_path = os.path.join(HERE, 'candidate_C_predictions.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print(f"wrote {out_path}")


if __name__ == '__main__':
    main()
