import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from scoring import load_labels, load_manifest, score_candidate

HERE = os.path.dirname(os.path.abspath(__file__))
LABELS_PATH = os.path.join(HERE, '..', 'team_assignment_v2', 'label_ui', 'labels.json')
MANIFEST_PATH = os.path.join(HERE, '..', 'team_assignment_v2', 'label_ui', 'selection_manifest.json')


def load_preds(name):
    path = os.path.join(HERE, f'{name}_predictions.json')
    if not os.path.exists(path):
        return None
    blob = json.load(open(path, encoding='utf-8'))
    preds = {mid: {int(k): v for k, v in d.items()} for mid, d in blob['predictions'].items()}
    return preds, blob.get('runtimes', {})


def main():
    labels = load_labels(LABELS_PATH)
    manifest = load_manifest(MANIFEST_PATH)

    results = {}
    for name in ['candidate_A', 'candidate_B', 'candidate_C']:
        loaded = load_preds(name)
        if loaded is None:
            print(f"{name}: SKIPPED (no predictions file)")
            results[name] = {'status': 'SKIPPED'}
            continue
        preds, runtimes = loaded
        score = score_candidate(preds, labels, manifest, name)
        score['runtimes'] = runtimes
        score['status'] = 'RUN'
        results[name] = score
        print(f"\n=== {name} ===")
        print(f"TP={score['tp']} FN={score['fn']} FP={score['fp']} TN={score['tn']}")
        print(f"recall={score['recall']:.3f} specificity={score['specificity']:.3f} "
              f"precision={score['precision']:.3f} f1={score['f1']:.3f} "
              f"balanced_acc={score['balanced_accuracy']:.3f}")
        for mid, m in score['per_match'].items():
            print(f"  [{mid}] TP={m['tp']} FN={m['fn']} FP={m['fp']} TN={m['tn']} "
                  f"recall={m['recall']:.3f} specificity={m['specificity']:.3f}")
            print(f"    detected (real ids): {m['detected_track_ids_real']}")

    out_path = os.path.join(HERE, 'RESULTS.json')
    json.dump(results, open(out_path, 'w', encoding='utf-8'), indent=2)
    print(f"\nwrote {out_path}")


if __name__ == '__main__':
    main()
