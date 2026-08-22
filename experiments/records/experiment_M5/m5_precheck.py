#!/usr/bin/env python
"""M5 Section 1 -- minimum blocking identity checks before any TEST inference."""
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def sha256(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    checks = []

    manifest = REPO / 'experiments/records/experiment_M3/SYSTEM_FREEZE_MANIFEST.json'
    sidecar = REPO / 'experiments/records/experiment_M3/SYSTEM_FREEZE_MANIFEST.sha256'
    checks.append(('manifest hash matches sidecar', sha256(manifest) == sidecar.read_text().strip()))

    final_list = REPO / 'experiments/records/experiment_M4/FINAL_TEST_FRAME_LIST.json'
    final_sha = REPO / 'experiments/records/experiment_M4/FINAL_TEST_FRAME_LIST.sha256'
    checks.append(('FINAL_TEST_FRAME_LIST.json hash matches sidecar',
                  sha256(final_list) == final_sha.read_text().strip()))

    gt = REPO / 'experiments/records/experiment_M4/TEST_DETECTION_ANNOTATIONS.json'
    gt_sha = REPO / 'experiments/records/experiment_M4/TEST_DETECTION_ANNOTATIONS.sha256'
    checks.append(('TEST_DETECTION_ANNOTATIONS.json hash matches sidecar',
                  sha256(gt) == gt_sha.read_text().strip()))

    contract = REPO / 'experiments/records/experiment_M4/M5_EVALUATION_CONTRACT.md'
    contract_sha = REPO / 'experiments/records/experiment_M4/M5_EVALUATION_CONTRACT.sha256'
    checks.append(('M5_EVALUATION_CONTRACT.md hash matches sidecar',
                  sha256(contract) == contract_sha.read_text().strip()))

    access = json.loads((REPO / 'experiments/records/experiment_M4/TEST_ACCESS_STATE.json')
                        .read_text(encoding='utf-8'))
    checks.append(('labels_frozen == true', access['access_state']['labels_frozen'] is True))
    checks.append(('production_predictions_run == false (pre-M5)',
                  access['access_state']['production_predictions_run'] is False))

    human_w = REPO / 'best_A_960.pt'
    checks.append(('best_A_960.pt sha256 matches manifest',
                  sha256(human_w) == '5eaf2e81d7f6b28fd0c665e769a5fb66ec71dc6be7f0d51576300a8370768e4a'))

    ball_w = REPO / 'models/third_party/soccernet_v3d/yolo-sn-ball.pt'
    checks.append(('yolo-sn-ball.pt sha256 matches manifest',
                  sha256(ball_w) == 'e8c1a900300893c34bf36c964c5854ed93603470e04a4a8eba73f70e4eea148b'))

    all_pass = all(v for _, v in checks)
    result = {'checks': [{'name': n, 'pass': v} for n, v in checks], 'all_pass': all_pass}
    out = REPO / 'experiments/records/experiment_M5/PRECHECK_RESULT.json'
    out.write_text(json.dumps(result, indent=1), encoding='utf-8')
    for n, v in checks:
        print(('PASS' if v else 'FAIL'), '-', n)
    print('\nALL_PASS:', all_pass)
    return 0 if all_pass else 1


if __name__ == '__main__':
    raise SystemExit(main())
