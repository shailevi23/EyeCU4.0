"""Candidate A -- no guard. Every track is assumed CLEAN, always."""
import os
import sys
import json

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

HERE = os.path.dirname(os.path.abspath(__file__))
MANIFEST_PATH = os.path.join(HERE, '..', 'team_assignment_v2', 'label_ui', 'selection_manifest.json')


def main():
    manifest = json.load(open(MANIFEST_PATH, encoding='utf-8'))
    predictions = {}
    for match in manifest['matches']:
        predictions[match['match_id']] = {t['track_id']: False for t in match['tracks']}
    out_path = os.path.join(HERE, 'candidate_A_predictions.json')
    json.dump({'predictions': predictions, 'runtimes': {m['match_id']: 0.0 for m in manifest['matches']}},
              open(out_path, 'w', encoding='utf-8'), indent=2)
    print(f"wrote {out_path}")


if __name__ == '__main__':
    main()
