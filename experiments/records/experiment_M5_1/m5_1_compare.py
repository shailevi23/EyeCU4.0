#!/usr/bin/env python
"""M5.1 -- numeric comparison of original M5 vs corrected M5.1 detection metrics."""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
orig = json.loads((REPO / 'experiments/records/experiment_M5/DETECTION_METRICS.json').read_text(encoding='utf-8'))
corr = json.loads((REPO / 'experiments/records/experiment_M5_1/M5_1_DETECTION_METRICS.json').read_text(encoding='utf-8'))
CLASSES = ['player', 'goalkeeper', 'referee', 'ball']


def row(o, c):
    return {'orig_AP50': o['AP50'], 'corrected_AP50': c['AP50'], 'delta_AP50': round(c['AP50'] - o['AP50'], 4),
           'orig_AP50_95': o['AP50_95'], 'corrected_AP50_95': c['AP50_95'], 'delta_AP50_95': round(c['AP50_95'] - o['AP50_95'], 4),
           'orig_P': o['precision'], 'corrected_P': c['precision'],
           'orig_R': o['recall'], 'corrected_R': c['recall']}


out = {'overall': {cls: row(orig['overall'][cls], corr['overall'][cls]) for cls in CLASSES},
      'mAP50': {'orig': orig['mAP50'], 'corrected': corr['mAP50'], 'delta': round(corr['mAP50'] - orig['mAP50'], 4)},
      'mAP50_95': {'orig': orig['mAP50_95'], 'corrected': corr['mAP50_95'], 'delta': round(corr['mAP50_95'] - orig['mAP50_95'], 4)},
      'per_sequence': {}}

for seq in ['como_2-0_sassuolo', 'manchester_city_v_liverpool', 'youth_2']:
    out['per_sequence'][seq] = {cls: row(orig['per_sequence'][seq][cls], corr['per_sequence'][seq][cls])
                                for cls in CLASSES}

outp = Path('experiments/records/experiment_M5_1/M5_1_VS_M5_COMPARISON.json')
outp.write_text(json.dumps(out, indent=1), encoding='utf-8')
print(json.dumps(out, indent=1))
print('written', outp)
