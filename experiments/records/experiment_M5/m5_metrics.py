#!/usr/bin/env python
"""
M5 Section 5-6 -- ONE metric evaluation from the saved raw predictions.
Standard COCO-style detection metrics (the M5 contract's own definition,
Section 3 of experiments/records/experiment_M4/M5_EVALUATION_CONTRACT.md):
IoU>=0.50 for a match, AP50 at IoU=0.50, AP50-95 averaged over
IoU=0.50:0.05:0.95 (10 thresholds), 101-point interpolated precision-recall
area (standard COCO definition). No new metric, no threshold sweep, no
confidence substitution -- every prediction already carries the frozen
accept_confidence>=0.25 from RAW_PREDICTIONS.json.
"""
import json
from pathlib import Path

CLASSES = ['player', 'goalkeeper', 'referee', 'ball']
IOU_THRESHOLDS = [round(0.5 + 0.05 * i, 2) for i in range(10)]  # 0.50..0.95


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def ap_101point(recalls, precisions):
    """Standard COCO 101-point interpolated AP."""
    if not recalls:
        return 0.0
    # precision envelope: precision(r) = max precision for recall' >= r
    pts = sorted(zip(recalls, precisions))
    rs = [p[0] for p in pts]
    ps = [p[1] for p in pts]
    for i in range(len(ps) - 2, -1, -1):
        ps[i] = max(ps[i], ps[i + 1])
    ap = 0.0
    for t in [i / 100 for i in range(101)]:
        # max precision at recall >= t
        p = 0.0
        for r, pr in zip(rs, ps):
            if r >= t:
                p = max(p, pr)
        ap += p
    return ap / 101


def evaluate(gt_by_image, pred_by_image, classes, group_keys):
    """group_keys: list of image keys to restrict to (e.g. one sequence, or all)."""
    result = {}
    for cls in classes:
        # gather all preds across group, sorted by confidence desc, with per-image GT pools
        all_preds = []  # (confidence, key, bbox)
        n_gt = 0
        gt_pools = {}
        for k in group_keys:
            g = [o['bbox'] for o in gt_by_image.get(k, []) if o['class'] == cls]
            gt_pools[k] = g
            n_gt += len(g)
            for p in pred_by_image.get(k, []):
                if p['class'] == cls:
                    all_preds.append((p['confidence'], k, p['bbox']))
        all_preds.sort(key=lambda x: -x[0])

        # AP per IoU threshold
        ap_per_thr = {}
        for thr in IOU_THRESHOLDS:
            used = {k: [False] * len(v) for k, v in gt_pools.items()}
            tp_cum, fp_cum = 0, 0
            recalls, precisions = [], []
            for conf, k, bbox in all_preds:
                gts = gt_pools[k]
                best_iou, best_j = 0.0, -1
                for j, g in enumerate(gts):
                    if used[k][j]:
                        continue
                    v = iou(bbox, g)
                    if v > best_iou:
                        best_iou, best_j = v, j
                if best_iou >= thr and best_j >= 0:
                    used[k][best_j] = True
                    tp_cum += 1
                else:
                    fp_cum += 1
                precisions.append(tp_cum / (tp_cum + fp_cum))
                recalls.append(tp_cum / n_gt if n_gt else 0.0)
            ap_per_thr[thr] = ap_101point(recalls, precisions) if n_gt else (1.0 if not all_preds else 0.0)

        # single-point P/R at IoU=0.50 using every raw prediction (no threshold sweep),
        # matched per-image (each image's GT pool is independent)
        used = {k: [False] * len(v) for k, v in gt_pools.items()}
        TP = FP = 0
        for conf, k, bbox in all_preds:
            gts = gt_pools[k]
            best_iou, best_j = 0.0, -1
            for j, g in enumerate(gts):
                if used[k][j]:
                    continue
                v = iou(bbox, g)
                if v > best_iou:
                    best_iou, best_j = v, j
            if best_iou >= 0.50 and best_j >= 0:
                used[k][best_j] = True
                TP += 1
            else:
                FP += 1
        FN = n_gt - TP
        precision = TP / (TP + FP) if (TP + FP) else 0.0
        recall = TP / (TP + FN) if (TP + FN) else 0.0

        result[cls] = {
            'gt_count': n_gt, 'pred_count': len(all_preds),
            'TP': TP, 'FP': FP, 'FN': FN,
            'precision': round(precision, 4), 'recall': round(recall, 4),
            'AP50': round(ap_per_thr[0.5], 4),
            'AP50_95': round(sum(ap_per_thr.values()) / len(ap_per_thr), 4),
        }
    return result


def main():
    m4 = Path('experiments/records/experiment_M4')
    gt_records = json.loads((m4 / 'TEST_DETECTION_ANNOTATIONS.json').read_text(encoding='utf-8'))
    pred_data = json.loads(Path('experiments/records/experiment_M5/RAW_PREDICTIONS.json')
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

    # multi-ball descriptive (Section 6)
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
    multi_ball_descriptive = {
        'n_multi_ball_frames': len(multi_ball_keys),
        'gt_balls_on_multi_ball_frames': mb_gt,
        'matched_balls': mb_tp,
        'missed_balls': mb_gt - mb_tp,
        'false_positive_balls': mb_fp,
    }

    result = {
        'iou_match_threshold': 0.50,
        'ap_iou_thresholds': IOU_THRESHOLDS,
        'overall': overall,
        'mAP50': round(mAP50, 4),
        'mAP50_95': round(mAP50_95, 4),
        'per_sequence': per_sequence,
        'multi_ball_descriptive': multi_ball_descriptive,
    }
    out = Path('experiments/records/experiment_M5/DETECTION_METRICS.json')
    out.write_text(json.dumps(result, indent=1), encoding='utf-8')
    print(json.dumps(result, indent=1))
    print('\nwritten:', out)


if __name__ == '__main__':
    main()
