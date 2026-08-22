#!/usr/bin/env python
"""
M4 -- automated leakage screen for the 60 initial-candidate TEST frames
against TRAIN and VAL, before any human visual TEST access.

FROZEN METHOD (declared here, before results are seen; not tuned against
this run):

  Layer 1 -- exact duplicate: sha256 of the raw file bytes. Any exact match
             against a TRAIN/VAL file is an automatic leak.

  Layer 2 -- perceptual candidate search: REUSES, unmodified, the exact
             functions already frozen in tools/extdata_leakage.py
             (canonical(), dihedral(), dhash(), ncc(), POP) -- imported, not
             reimplemented or retuned. 64-bit dhash of a letterbox-stripped
             64x64 canonical signature, all 8 dihedral orientations (4
             rotations x mirror), Hamming distance <= 10 as the candidate
             gate. Identical constants (SIG=64, HAM=10) to the already-frozen
             tool.

  Layer 3 -- deterministic algorithmic confirmation: NCC >= 0.95 on the
             matched orientation, exactly the frozen tool's own verification
             threshold. A candidate below this is not counted as a leak.

This step reads pixels computationally (required to hash/compare them) but
performs NO human visual inspection, NO production-model inference (SN3D /
best_A_960 / CBIoU / BallTemporalSelector are never invoked here), and NO
annotation. machine_leakage_accessed=true; human_annotation_accessed
remains false until this script and the replacement pass (if needed) are
both done.

Reference pool (TRAIN + VAL, exhaustive, not sampled):
  - data/dataset_baseline/images/train
  - data/dataset_baseline/images/val
  - data/tracking_val_gt/sequences/*/img1 (the 4 pinned VAL match sequences,
    tracking-format frames not present in the YOLO-format dataset folder)
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'tools'))
from extdata_leakage import canonical, dihedral, dhash, ncc, POP, HAM  # noqa: E402  -- frozen, reused unmodified

NCC_THRESHOLD = 0.95  # identical to tools/extdata_leakage.py's own verification gate
IMG_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def reference_pool():
    pool = []
    for d in [REPO / 'data/dataset_baseline/images/train',
             REPO / 'data/dataset_baseline/images/val']:
        if d.exists():
            pool += [p for p in sorted(d.rglob('*')) if p.suffix.lower() in IMG_EXT]
    for seq_dir in sorted((REPO / 'data/tracking_val_gt/sequences').iterdir()):
        img1 = seq_dir / 'img1'
        if img1.exists():
            pool += sorted(img1.glob('*.jpg'))
    return pool


def load_gray(path):
    import cv2
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


def main():
    candidates = json.loads(Path('experiments/records/experiment_M4/candidates_manifest.json')
                            .read_text(encoding='utf-8'))
    ref_paths = reference_pool()
    print(f'{len(candidates)} TEST candidates vs {len(ref_paths)} TRAIN/VAL reference images')

    # ---- Layer 1: exact file hash
    ref_sha = {}
    for p in ref_paths:
        ref_sha.setdefault(hashlib.sha256(p.read_bytes()).hexdigest(), []).append(str(p))

    # ---- precompute reference dhash bank (Layer 2 candidate gate)
    ref_sigs = []
    ref_valid = []
    for p in ref_paths:
        img = load_gray(p)
        ref_sigs.append(None if img is None else canonical(img))
    ref_valid = [i for i, s in enumerate(ref_sigs) if s is not None]
    RH = np.stack([dhash(ref_sigs[i]) for i in ref_valid])

    results = []
    for c in candidates:
        fpath = Path(c['file'])
        raw = fpath.read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        entry = {**c, 'sha256': sha, 'exact_duplicate_of': ref_sha.get(sha, []),
                 'perceptual_hits': []}

        img = load_gray(fpath)
        sig = canonical(img)
        for k, t in enumerate(dihedral(sig)):
            h = dhash(t)
            dist = POP[np.bitwise_xor(RH, h)].sum(axis=1)
            for w in np.where(dist <= HAM)[0]:
                j = ref_valid[int(w)]
                v = ncc(t, ref_sigs[j])
                if v >= NCC_THRESHOLD:
                    entry['perceptual_hits'].append({
                        'reference': str(ref_paths[j]).replace('\\', '/'),
                        'orientation': k, 'hamming': int(dist[w]), 'ncc': round(v, 4),
                    })
        entry['is_leak'] = bool(entry['exact_duplicate_of']) or bool(entry['perceptual_hits'])
        results.append(entry)

    n_leaks = sum(1 for r in results if r['is_leak'])
    report = {
        'method': {
            'layer1_exact_sha256': True,
            'layer2_perceptual_candidate': 'dhash, 8 dihedral orientations, Hamming <= 10 (reused unmodified from tools/extdata_leakage.py)',
            'layer3_verification': f'NCC >= {NCC_THRESHOLD} (reused unmodified threshold)',
            'reference_pool_size': len(ref_paths),
            'reference_pool_sources': ['data/dataset_baseline/images/train',
                                       'data/dataset_baseline/images/val',
                                       'data/tracking_val_gt/sequences/*/img1'],
        },
        'n_candidates': len(candidates),
        'n_leaks_found': n_leaks,
        'leaking_frames': [r['file'] for r in results if r['is_leak']],
        'results': results,
    }
    out = Path('experiments/records/experiment_M4/LEAKAGE_SCREEN_RESULT.json')
    out.write_text(json.dumps(report, indent=1), encoding='utf-8')
    print('n_leaks_found:', n_leaks)
    if n_leaks:
        print('leaking frames:', report['leaking_frames'])
    print('written:', out)


if __name__ == '__main__':
    main()
