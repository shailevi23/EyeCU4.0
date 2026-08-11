#!/usr/bin/env python
"""
Load the exported GT with the pinned TrackEval MotChallenge2DBox reader.

No metric is computed. The question is only whether the official loader accepts
our files and returns exactly what we wrote -- if it silently drops rows, every
later number is wrong in a way no metric would reveal.

TrackEval is VENDORED, not installed: third_party/trackeval-1.3.0, extracted
from the trackeval 1.3.0 wheel and added to sys.path only here. The active
environment is unchanged, and the exact bytes being reasoned about are in the
repository rather than in whatever happens to be on a machine later.
"""

import argparse
import configparser
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
VENDOR = REPO / 'third_party' / 'trackeval-1.3.0'
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(VENDOR))

BENCHMARK = 'EyeCU'
SPLIT = 'val'
DO_PREPROC = False        # frozen; see experiments/tracking_v2/trackeval_protocol.json


def check(rows, name, ok, detail=''):
    rows.append({'check': name, 'ok': bool(ok), 'detail': detail})
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mot', default='data/tracking_val_gt/mot')
    ap.add_argument('--root', default='data/tracking_val_gt')
    ap.add_argument('--json-out', default=None)
    args = ap.parse_args()

    import numpy as np
    import trackeval
    from trackeval.datasets.mot_challenge_2d_box import MotChallenge2DBox

    rows = []
    print(f'trackeval from {VENDOR}')
    print(f'version attr  : {getattr(trackeval, "__version__", "not exposed")}')

    mot = (REPO / args.mot).resolve()
    cfg = MotChallenge2DBox.get_default_dataset_config()
    cfg.update({
        'GT_FOLDER': str(mot),
        'TRACKERS_FOLDER': str(mot / '_no_trackers'),
        'BENCHMARK': BENCHMARK,
        'SPLIT_TO_EVAL': SPLIT,
        'DO_PREPROC': DO_PREPROC,
        'CLASSES_TO_EVAL': ['pedestrian'],
        'PRINT_CONFIG': False,
        'SKIP_SPLIT_FOL': False,
    })
    (mot / '_no_trackers' / f'{BENCHMARK}-{SPLIT}').mkdir(parents=True, exist_ok=True)

    ds = MotChallenge2DBox(cfg)
    check(rows, 'dataset class instantiates', True, type(ds).__name__)
    check(rows, 'DO_PREPROC as frozen', ds.do_preproc is DO_PREPROC, str(ds.do_preproc))
    check(rows, 'class list is pedestrian only', ds.class_list == ['pedestrian'],
          str(ds.class_list))

    seqs = sorted(ds.seq_list)
    expected = sorted(p.name for p in (mot / f'{BENCHMARK}-{SPLIT}').iterdir()
                      if p.is_dir())
    check(rows, 'loader discovers exactly the three sequences', seqs == expected,
          str(seqs))
    check(rows, 'austin absent', not any('austin' in s for s in seqs))

    canon = {}
    manifest = json.loads((REPO / args.root / 'manifest.json').read_text(encoding='utf-8'))
    for s in manifest['sequences']:
        ann = json.loads((REPO / args.root / s['annotation_file_expected']
                          ).read_text(encoding='utf-8'))
        canon[s['sequence']] = ann['boxes']

    stats = {}
    for seq in seqs:
        # GT keys are remapped to gt_* at the end of _load_raw_file (L277-286)
        raw = ds._load_raw_file(None, seq, is_gt=True)
        n = raw['num_timesteps']
        loaded = sum(len(raw['gt_ids'][t]) for t in range(n))
        ids = set()
        for t in range(n):
            ids |= set(int(i) for i in raw['gt_ids'][t])
        classes = set()
        for t in range(n):
            classes |= set(int(c) for c in raw['gt_classes'][t])
        zero = set()
        for t in range(n):
            zero |= set(int(z) for z in raw['gt_extras'][t]['zero_marked'])
        want = canon[seq]
        check(rows, f'{seq}: 300 timesteps', n == 300, str(n))
        check(rows, f'{seq}: every GT row loaded', loaded == len(want),
              f'{loaded} vs {len(want)}')
        check(rows, f'{seq}: identities unchanged',
              ids == {b['id'] for b in want}, str(sorted(ids ^ {b['id'] for b in want})))
        check(rows, f'{seq}: every row class 1 (pedestrian)', classes == {1}, str(classes))
        check(rows, f'{seq}: every row zero_marked 1', zero == {1}, str(zero))

        by_key = {(b['frame'], b['id']): b['bbox'] for b in want}
        worst = 0.0
        for t in range(n):
            for i, d in zip(raw['gt_ids'][t], raw['gt_dets'][t]):
                x1, y1, x2, y2 = by_key[(t + 1, int(i))]
                worst = max(worst, float(np.max(np.abs(
                    np.array(d) - np.array([x1, y1, x2 - x1, y2 - y1])))))
        check(rows, f'{seq}: boxes match canonical within 0.011 px', worst <= 0.011,
              f'worst {worst:.4f}')

        ini = configparser.ConfigParser()
        ini.read(mot / f'{BENCHMARK}-{SPLIT}' / seq / 'seqinfo.ini', encoding='utf-8')
        stats[seq] = {'timesteps': n, 'gt_rows': loaded, 'identities': len(ids),
                      'classes': sorted(classes), 'zero_marked': sorted(zero),
                      'worst_box_delta_px': round(worst, 4),
                      'seq_length': int(ini['Sequence']['seqLength'])}
        print(f'  {seq:<32}{n} timesteps  {loaded} rows  {len(ids)} ids  '
              f'classes {sorted(classes)}  worst delta {worst:.4f} px')

    failed = [r for r in rows if not r['ok']]
    print(f'\n{len(rows) - len(failed)}/{len(rows)} loader checks passed')
    for r in failed:
        print(f'   [FAIL] {r["check"]}   -- {r["detail"]}')
    print('TRACKEVAL LOADER: ' + ('PASS' if not failed else 'FAIL'))
    print('No metric was computed.')

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(
            {'vendored_trackeval': str(VENDOR.relative_to(REPO)),
             'benchmark': f'{BENCHMARK}-{SPLIT}', 'do_preproc': DO_PREPROC,
             'metrics_computed': False, 'checks': rows, 'sequences': stats,
             'passed': not failed}, indent=1), encoding='utf-8')
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
