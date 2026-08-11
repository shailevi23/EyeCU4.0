#!/usr/bin/env python
"""
Score the bake-off outputs with the pinned TrackEval, under the frozen protocol.

DO_PREPROC comes from experiments/tracking_v2/trackeval_protocol.json and is
asserted, not passed in: a flag that can be typed differently on the command
line is a flag that can be changed after seeing results.

Reports the three declared views: per sequence, the official COMBINED_SEQ, and
an unweighted macro mean that is descriptive only.
"""

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
VENDOR = REPO / 'third_party' / 'trackeval-1.3.0'
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(VENDOR))

SPLIT_BENCH, SPLIT = 'EyeCU', 'val'
FIELDS = ['HOTA', 'DetA', 'AssA', 'MOTA', 'IDF1']
EXTRA = ['IDSW', 'Frag', 'DetRe', 'DetPr', 'AssRe', 'AssPr', 'LocA']


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bakeoff', default='experiments/tracking_v2/bakeoff')
    ap.add_argument('--profile', required=True)
    args = ap.parse_args()

    import numpy as np
    import trackeval

    proto = json.loads((REPO / 'experiments' / 'tracking_v2'
                        / 'trackeval_protocol.json').read_text(encoding='utf-8'))
    do_preproc = proto['DO_PREPROC']
    assert do_preproc is False, 'frozen protocol says DO_PREPROC False'

    base = REPO / args.bakeoff
    gt_folder = REPO / 'data' / 'tracking_val_gt' / 'mot'
    trackers_folder = base / 'outputs' / args.profile

    eval_cfg = trackeval.Evaluator.get_default_eval_config()
    eval_cfg.update({'USE_PARALLEL': False, 'PRINT_RESULTS': False,
                     'PRINT_CONFIG': False, 'TIME_PROGRESS': False,
                     'OUTPUT_SUMMARY': False, 'OUTPUT_EMPTY_CLASSES': False,
                     'OUTPUT_DETAILED': False, 'PLOT_CURVES': False,
                     'DISPLAY_LESS_PROGRESS': True})
    ds_cfg = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    ds_cfg.update({'GT_FOLDER': str(gt_folder),
                   'TRACKERS_FOLDER': str(trackers_folder),
                   'BENCHMARK': SPLIT_BENCH, 'SPLIT_TO_EVAL': SPLIT,
                   'DO_PREPROC': do_preproc, 'CLASSES_TO_EVAL': ['pedestrian'],
                   'PRINT_CONFIG': False, 'SKIP_SPLIT_FOL': False})
    metrics = [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR(),
               trackeval.metrics.Identity()]

    ev = trackeval.Evaluator(eval_cfg)
    ds = trackeval.datasets.MotChallenge2DBox(ds_cfg)
    raw, _ = ev.evaluate([ds], metrics)

    key = 'MotChallenge2DBox'
    out = {}
    for tracker, res in raw[key].items():
        per_seq = {}
        for seq, r in res.items():
            c = r['pedestrian']
            row = {}
            for f in FIELDS + EXTRA:
                for m in ('HOTA', 'CLEAR', 'Identity'):
                    if f in c.get(m, {}):
                        v = c[m][f]
                        row[f] = float(np.mean(v)) * (100 if f in
                                                      ('HOTA', 'DetA', 'AssA',
                                                       'DetRe', 'DetPr', 'AssRe',
                                                       'AssPr', 'LocA') else 1)
                        if f in ('MOTA', 'IDF1'):
                            row[f] = float(v) * 100
                        break
            per_seq[seq] = row
        out[tracker] = per_seq

    (base / 'trackeval_raw').mkdir(parents=True, exist_ok=True)
    (base / 'trackeval_raw' / f'{args.profile}.json').write_text(
        json.dumps(out, indent=1), encoding='utf-8')

    seqs = sorted(s for s in next(iter(out.values())) if s != 'COMBINED_SEQ')
    print(f'\n=== {args.profile}   (DO_PREPROC={do_preproc}) ===')
    hdr = f'{"tracker":<32}{"sequence":<32}' + ''.join(f'{f:>8}' for f in FIELDS)
    print(hdr)
    for t in sorted(out):
        for s in seqs:
            r = out[t][s]
            print(f'  {t[:30]:<32}{s[:30]:<32}'
                  + ''.join(f'{r.get(f, float("nan")):>8.2f}' for f in FIELDS))
    print(f'\n{"tracker":<32}{"COMBINED_SEQ":<32}' + ''.join(f'{f:>8}' for f in FIELDS))
    for t in sorted(out):
        r = out[t]['COMBINED_SEQ']
        print(f'  {t[:30]:<32}{"":<32}'
              + ''.join(f'{r.get(f, float("nan")):>8.2f}' for f in FIELDS))
    print(f'\n{"tracker":<32}{"MACRO MEAN (descriptive)":<32}'
          + ''.join(f'{f:>8}' for f in ('HOTA', 'AssA', 'IDF1')))
    macro = {}
    for t in sorted(out):
        macro[t] = {f: float(np.mean([out[t][s][f] for s in seqs]))
                    for f in ('HOTA', 'AssA', 'IDF1')}
        print(f'  {t[:30]:<32}{"":<32}'
              + ''.join(f'{macro[t][f]:>8.2f}' for f in ('HOTA', 'AssA', 'IDF1')))
    (base / 'trackeval_raw' / f'{args.profile}_macro.json').write_text(
        json.dumps(macro, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
