import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from scoring import load_labels, load_manifest, score_candidate

HERE = os.path.dirname(os.path.abspath(__file__))


def load_preds(name):
    path = os.path.join(HERE, f'{name}_predictions.json')
    if not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf-8') as f:
        blob = json.load(f)
    # normalize keys to int track ids
    preds = {}
    for mid, d in blob['predictions'].items():
        preds[mid] = {int(k): v for k, v in d.items()}
    return preds, blob.get('runtimes', {})


def main():
    labels = load_labels(os.path.join(HERE, 'label_ui', 'labels.json'))
    manifest = load_manifest(os.path.join(HERE, 'label_ui', 'selection_manifest.json'))

    results = {}
    for name in ['candidate_A', 'candidate_B', 'candidate_C']:
        loaded = load_preds(name)
        if loaded is None:
            print(f"{name}: no predictions file found, skipping")
            continue
        preds, runtimes = loaded
        score = score_candidate(preds, labels, manifest, name)
        score['runtimes'] = runtimes
        results[name] = score
        print(f"\n=== {name} ===")
        print(f"pooled track accuracy: {score['pooled_correct']}/{score['pooled_total']} "
              f"= {score['pooled_track_accuracy']:.3f}")
        print(f"pooled frame-weighted accuracy: {score['pooled_frame_weighted_accuracy']:.3f}")
        print(f"pooled coverage: {score['pooled_coverage']:.3f}")
        print(f"pooled selective accuracy: {score['pooled_selective_accuracy']}")
        for mid, r in score['per_match'].items():
            print(f"  [{mid}] {r['correct']}/{r['ab_track_count']} = {r['accuracy']:.3f}  "
                  f"(frame-weighted {r['frame_weighted_accuracy']:.3f}, coverage {r['coverage']:.3f}, "
                  f"mixed={r['mixed_track_count']}, ambiguous={r['ambiguous_count']})")
            print(f"    wrong track_ids (real): {r['wrong_track_ids_real']}")

    out_path = os.path.join(HERE, 'RESULTS.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == '__main__':
    main()
