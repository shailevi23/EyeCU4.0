"""
Shared scoring harness, implementing EVALUATION_CONTRACT.md exactly (incl.
its abstention fix). Used identically by all three candidates so no
candidate gets a bespoke metric.

predictions: {match_id: {track_id(str or int): predicted_team (1, 2, or None)}}
labels:      {"<match_id>:<track_id>": "TEAM_A"|"TEAM_B"|"MIXED_TRACK"|"AMBIGUOUS"}
manifest:    selection_manifest.json (for appearance_count, frame weighting)
"""
import json
import itertools


def load_labels(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_manifest(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def score_candidate(predictions, labels, manifest, candidate_name):
    """
    Returns a dict with all metrics EVALUATION_CONTRACT.md requires:
    pooled + per-match primary track accuracy, primary frame-weighted
    accuracy, coverage, selective accuracy, mixed/ambiguous counts,
    and the list of incorrect track_ids (real ids) per match.
    """
    per_match_results = {}
    pooled_correct = 0
    pooled_total = 0
    pooled_weighted_correct = 0.0
    pooled_weighted_total = 0.0
    pooled_selective_correct = 0
    pooled_selective_total = 0

    for match in manifest['matches']:
        mid = match['match_id']
        appearance_by_tid = {t['track_id']: t['appearance_count'] for t in match['tracks']}
        ab_tracks = []  # (track_id, human_label 'TEAM_A'/'TEAM_B', predicted or None, appearance_count)
        mixed_count = 0
        ambiguous_count = 0
        for t in match['tracks']:
            tid = t['track_id']
            key = f"{mid}:{tid}"
            lbl = labels[key]
            if lbl == 'MIXED_TRACK':
                mixed_count += 1
                continue
            if lbl == 'AMBIGUOUS':
                ambiguous_count += 1
                continue
            pred = predictions.get(mid, {}).get(tid, predictions.get(mid, {}).get(str(tid)))
            ab_tracks.append((tid, lbl, pred, appearance_by_tid[tid]))

        best = None
        for mapping in ({1: 'TEAM_A', 2: 'TEAM_B'}, {1: 'TEAM_B', 2: 'TEAM_A'}):
            correct = 0
            weighted_correct = 0.0
            weighted_total = 0.0
            wrong_ids = []
            selective_correct = 0
            selective_total = 0
            for tid, lbl, pred, n_frames in ab_tracks:
                weighted_total += n_frames
                is_correct = (pred is not None) and (mapping.get(pred) == lbl)
                if is_correct:
                    correct += 1
                    weighted_correct += n_frames
                else:
                    wrong_ids.append(tid)
                if pred is not None:
                    selective_total += 1
                    if mapping.get(pred) == lbl:
                        selective_correct += 1
            total = len(ab_tracks)
            acc = correct / total if total else 0.0
            if best is None or acc > best['accuracy']:
                best = {
                    'mapping': mapping, 'correct': correct, 'total': total,
                    'accuracy': acc, 'wrong_track_ids': wrong_ids,
                    'weighted_correct': weighted_correct, 'weighted_total': weighted_total,
                    'selective_correct': selective_correct, 'selective_total': selective_total,
                }

        coverage = sum(1 for _, _, pred, _ in ab_tracks if pred is not None) / len(ab_tracks) if ab_tracks else 0.0
        selective_acc = (best['selective_correct'] / best['selective_total']) if best['selective_total'] else None

        per_match_results[mid] = {
            'ab_track_count': best['total'],
            'correct': best['correct'],
            'accuracy': best['accuracy'],
            'chosen_permutation': best['mapping'],
            'wrong_track_ids_real': best['wrong_track_ids'],
            'frame_weighted_accuracy': (best['weighted_correct'] / best['weighted_total']) if best['weighted_total'] else 0.0,
            'coverage': coverage,
            'selective_accuracy': selective_acc,
            'mixed_track_count': mixed_count,
            'ambiguous_count': ambiguous_count,
        }
        pooled_correct += best['correct']
        pooled_total += best['total']
        pooled_weighted_correct += best['weighted_correct']
        pooled_weighted_total += best['weighted_total']
        pooled_selective_correct += best['selective_correct']
        pooled_selective_total += best['selective_total']

    return {
        'candidate': candidate_name,
        'pooled_track_accuracy': pooled_correct / pooled_total if pooled_total else 0.0,
        'pooled_correct': pooled_correct,
        'pooled_total': pooled_total,
        'pooled_frame_weighted_accuracy': (pooled_weighted_correct / pooled_weighted_total) if pooled_weighted_total else 0.0,
        'pooled_coverage': sum(r['coverage'] * r['ab_track_count'] for r in per_match_results.values()) / pooled_total if pooled_total else 0.0,
        'pooled_selective_accuracy': (pooled_selective_correct / pooled_selective_total) if pooled_selective_total else None,
        'per_match': per_match_results,
    }
