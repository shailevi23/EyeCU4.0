"""
Shared, frozen single-change-point test used by both Candidate B (color) and
Candidate C (SigLIP). Implements exactly the rule in CANDIDATE_DEFINITIONS.md
Sec B.5 -- do not alter after any score has been viewed.
"""
import numpy as np

MIN_REGIME_SIZE = 3
SEPARATION_RATIO_THRESHOLD = 3.0


def detect_change_point(ordered_descriptors, ordered_frame_idxs):
    """
    ordered_descriptors: list/array of per-observation descriptor vectors,
        already sorted by processed-frame index (ascending).
    ordered_frame_idxs:  matching processed-frame indices (for reporting).

    Returns a dict:
        {'contaminated': bool, 'split_pos': int or None (index into the
         ordered list, i.e. first observation of the post-split regime),
         'split_at_processed_frame': int or None,
         'separation_ratio': float or None}
    """
    n = len(ordered_descriptors)
    if n < 2 * MIN_REGIME_SIZE:
        return {'contaminated': False, 'split_pos': None,
                'split_at_processed_frame': None, 'separation_ratio': None}

    X = np.asarray(ordered_descriptors, dtype=np.float64)
    best = None
    for split_pos in range(MIN_REGIME_SIZE, n - MIN_REGIME_SIZE + 1):
        pre, post = X[:split_pos], X[split_pos:]
        pre_centroid, post_centroid = np.median(pre, axis=0), np.median(post, axis=0)
        pre_spread = float(np.linalg.norm(pre - pre_centroid, axis=1).mean())
        post_spread = float(np.linalg.norm(post - post_centroid, axis=1).mean())
        avg_spread = (pre_spread + post_spread) / 2
        center_dist = float(np.linalg.norm(pre_centroid - post_centroid))
        ratio = center_dist / max(avg_spread, 1e-6)
        if best is None or ratio > best['separation_ratio']:
            best = {'split_pos': split_pos, 'separation_ratio': ratio}

    contaminated = best['separation_ratio'] >= SEPARATION_RATIO_THRESHOLD
    return {
        'contaminated': contaminated,
        'split_pos': best['split_pos'] if contaminated else None,
        'split_at_processed_frame': (ordered_frame_idxs[best['split_pos']]
                                     if contaminated else None),
        'separation_ratio': best['separation_ratio'],
    }
