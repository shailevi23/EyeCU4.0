"""Binary CONTAMINATED/CLEAN scoring, per EVALUATION_CONTRACT.md."""
import json


def load_labels(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_manifest(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def score_candidate(predictions, labels, manifest, candidate_name):
    """
    predictions: {match_id: {track_id: bool (True=CONTAMINATED, False=CLEAN)}}
    Returns pooled + per-match TP/FP/FN/TN and derived metrics.
    """
    pooled = {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0}
    per_match = {}

    for match in manifest['matches']:
        mid = match['match_id']
        m = {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0, 'detected_track_ids_real': []}
        for t in match['tracks']:
            tid = t['track_id']
            lbl = labels[f"{mid}:{tid}"]
            if lbl == 'AMBIGUOUS':
                continue
            is_mixed = (lbl == 'MIXED_TRACK')
            pred = predictions.get(mid, {}).get(tid, predictions.get(mid, {}).get(str(tid), False))
            if is_mixed and pred:
                m['tp'] += 1
                m['detected_track_ids_real'].append(tid)
            elif is_mixed and not pred:
                m['fn'] += 1
            elif not is_mixed and pred:
                m['fp'] += 1
                m['detected_track_ids_real'].append(tid)
            else:
                m['tn'] += 1
        per_match[mid] = m
        for k in ('tp', 'fp', 'fn', 'tn'):
            pooled[k] += m[k]

    def derive(d):
        tp, fp, fn, tn = d['tp'], d['fp'], d['fn'], d['tn']
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        balanced_acc = (recall + specificity) / 2
        return {'recall': recall, 'specificity': specificity, 'precision': precision,
                'f1': f1, 'balanced_accuracy': balanced_acc,
                'fp_rate': fp / (tn + fp) if (tn + fp) else 0.0}

    result = {'candidate': candidate_name, **pooled, **derive(pooled), 'per_match': {}}
    for mid, m in per_match.items():
        result['per_match'][mid] = {**m, **derive(m)}
    return result
