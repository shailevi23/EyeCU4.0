#!/usr/bin/env python
"""
M5.1 -- ONE metric evaluation against the corrected GT, reusing the exact
same metric code (iou/ap_101point/evaluate) from experiments/records/
experiment_M5/m5_metrics.py -- imported, not reimplemented, so this is the
same metric definition as the original M5, not a new one. Only the GT
input changes (TEST_DETECTION_ANNOTATIONS_CORRECTED.json instead of the
original); RAW_PREDICTIONS.json is reused verbatim, unmodified, not
regenerated.
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'experiments/records/experiment_M5'))
from m5_metrics import evaluate, iou, CLASSES, IOU_THRESHOLDS  # noqa: E402


def main():
    m5_1 = REPO / 'experiments/records/experiment_M5_1'
    gt_records = json.loads((m5_1 / 'TEST_DETECTION_ANNOTATIONS_CORRECTED.json')
                            .read_text(encoding='utf-8'))
    pred_data = json.loads((REPO / 'experiments/records/experiment_M5/RAW_PREDICTIONS.json')
                           .read_text(encoding='utf-8'))

    def key(seq, n):
        return f'{seq}#{n}'

    gt_by_image = {key(r['sequence'], r['frame_number_1based']): r['objects'] for r in gt_records}
    pred_by_image = {key(r['sequence'], r['frame_number_1based']): r['detections']
                     for r in pred_data['records']}
    all_keys = list(gt_by_image.keys())
    sequences = sorted(set(r['sequence'] for r in gt_records))

    overall = evaluate(gt_by_image, pred_by_image, CLASSES, all_keys)
    per_sequence = {}
    for seq in sequences:
        keys = [key(seq, r['frame_number_1based']) for r in gt_records if r['sequence'] == seq]
        per_sequence[seq] = evaluate(gt_by_image, pred_by_image, CLASSES, keys)

    mAP50 = sum(v['AP50'] for v in overall.values()) / len(CLASSES)
    mAP50_95 = sum(v['AP50_95'] for v in overall.values()) / len(CLASSES)

    multi_ball_keys = [key(r['sequence'], r['frame_number_1based']) for r in gt_records
                       if sum(1 for o in r['objects'] if o['class'] == 'ball') > 1]
    mb_gt = mb_tp = mb_fp = 0
    for k in multi_ball_keys:
        gts = [o['bbox'] for o in gt_by_image[k] if o['class'] == 'ball']
        preds = sorted([p for p in pred_by_image.get(k, []) if p['class'] == 'ball'],
                       key=lambda p: -p['confidence'])
        used = [False] * len(gts)
        mb_gt += len(gts)
        for p in preds:
            best_iou, best_j = 0.0, -1
            for j, g in enumerate(gts):
                if used[j]:
                    continue
                v = iou(p['bbox'], g)
                if v > best_iou:
                    best_iou, best_j = v, j
            if best_iou >= 0.50 and best_j >= 0:
                used[best_j] = True
                mb_tp += 1
            else:
                mb_fp += 1

    result = {
        'gt_source': 'TEST_DETECTION_ANNOTATIONS_CORRECTED.json (M5.1: 20 blind-reannotated como_2-0_sassuolo + 40 unchanged manchester/youth_2)',
        'predictions_source': 'experiments/records/experiment_M5/RAW_PREDICTIONS.json (reused verbatim, not regenerated)',
        'iou_match_threshold': 0.50,
        'ap_iou_thresholds': IOU_THRESHOLDS,
        'overall': overall,
        'mAP50': round(mAP50, 4),
        'mAP50_95': round(mAP50_95, 4),
        'per_sequence': per_sequence,
        'multi_ball_descriptive': {
            'n_multi_ball_frames': len(multi_ball_keys),
            'gt_balls_on_multi_ball_frames': mb_gt,
            'matched_balls': mb_tp,
            'missed_balls': mb_gt - mb_tp,
            'false_positive_balls': mb_fp,
        },
    }
    out = m5_1 / 'M5_1_DETECTION_METRICS.json'
    out.write_text(json.dumps(result, indent=1), encoding='utf-8')
    print(json.dumps(result, indent=1))
    print('\nwritten:', out)


if __name__ == '__main__':
    main()
