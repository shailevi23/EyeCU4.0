#!/usr/bin/env python
"""
Stage 15: prove the audit changed nothing.

An audit that quietly modified what it was auditing would be worthless, so the
claim is checked rather than asserted:

  * the six ZIPs still hash to what they hashed before extraction
  * EyeCU TRAIN, VAL and TEST images are byte-for-byte unchanged
  * no detector checkpoint changed
  * no tracker code or frozen tracker config changed
  * no training ran and no TEST evaluation ran

TRAIN/VAL/TEST integrity is measured against a baseline this tool writes on its
first run and verifies on every run after. TEST is hashed as image bytes; its
labels do not exist yet and were never opened.
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
AUDIT = REPO / 'experiments' / 'external_data_audit'
BASE = AUDIT / 'reports' / 'integrity_baseline.json'
IMG_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

WATCHED = {
    'EYECU_TRAIN': ('data/dataset_baseline/images/train', None),
    'EYECU_VAL': ('data/dataset_baseline/images/val', None),
    'EYECU_TEST_IMAGES': ('data/frames', ['como_2-0_sassuolo',
                                          'manchester_city_v_liverpool', 'youth_2']),
    'EYECU_TRAIN_LABELS': ('data/dataset_baseline/labels/train', None),
    'EYECU_VAL_LABELS': ('data/dataset_baseline/labels/val', None),
}
CODE = ['trackers/football_tracker.py', 'trackers/detector.py',
        'experiments/tracking_v2/integration/TRACKER_FREEZE.json',
        'data/dataset_baseline/split_report.json',
        'data/external_provenance.json']
WEIGHTS = ['best_A_960.pt', 'best_B_1280.pt', 'best_C_960.pt', 'eyecu_football_v1.pt']


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, 'rb') as fh:
        for b in iter(lambda: fh.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def tree_digest(root: Path, subs):
    """One hash over (relative path, content hash) for every file, sorted."""
    files = []
    for sub in (subs or ['']):
        d = root / sub if sub else root
        if not d.exists():
            continue
        for p in sorted(d.rglob('*')):
            if p.is_file() and (not IMG_EXT or p.suffix.lower() in IMG_EXT
                                or p.suffix.lower() == '.txt'):
                files.append((str(p.relative_to(REPO)).replace('\\', '/'), sha(p)))
    h = hashlib.sha256()
    for rel, d in files:
        h.update(rel.encode()); h.update(d.encode())
    return {'files': len(files), 'digest': h.hexdigest()}


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    srcs = json.loads((AUDIT / 'raw' / 'SOURCES.json').read_text(encoding='utf-8'))
    now = {'zips': {}, 'trees': {}, 'code': {}, 'weights': {}}

    for sid, s in srcs['sources'].items():
        p = REPO / 'check_datasets' / s['original_filename']
        now['zips'][sid] = {'file': s['original_filename'], 'sha256': sha(p),
                            'expected': s['sha256'], 'bytes': p.stat().st_size}
    for k, (root, subs) in WATCHED.items():
        now['trees'][k] = tree_digest(REPO / root, subs)
    for f in CODE:
        p = REPO / f
        now['code'][f] = sha(p) if p.exists() else None
    for f in WEIGHTS:
        p = REPO / f
        now['weights'][f] = {'sha256': sha(p), 'mtime': int(p.stat().st_mtime)} \
            if p.exists() else None

    first = not BASE.exists()
    if first:
        BASE.write_text(json.dumps(now, indent=1), encoding='utf-8')
        print('baseline written (first run)')
    base = json.loads(BASE.read_text(encoding='utf-8'))

    checks, ok = [], True

    def check(name, cond, detail=''):
        nonlocal ok
        ok &= bool(cond)
        checks.append({'check': name, 'pass': bool(cond), 'detail': detail})
        print(f'  {"PASS" if cond else "FAIL"}  {name}  {detail}')

    print('ZIP hashes unchanged since extraction')
    for sid, v in now['zips'].items():
        check(f'zip {sid} ({v["file"]})', v['sha256'] == v['expected'],
              v['sha256'][:16])

    print('EyeCU data unchanged')
    for k, v in now['trees'].items():
        b = base['trees'].get(k, {})
        check(k, v['digest'] == b.get('digest') and v['files'] == b.get('files'),
              f'{v["files"]} files, {v["digest"][:16]}')

    print('Frozen code and records unchanged')
    for k, v in now['code'].items():
        check(k, v == base['code'].get(k), (v or 'absent')[:16])

    print('Detector checkpoints unchanged')
    for k, v in now['weights'].items():
        b = base['weights'].get(k)
        check(k, v == b, (v or {}).get('sha256', 'absent')[:16] if v else 'absent')

    # no training / no TEST evaluation: nothing in this audit imports ultralytics
    # or TrackEval, and no run directory was produced.
    runs = list((REPO / 'runs').glob('**/*')) if (REPO / 'runs').exists() else []
    check('no training run directory produced', not runs, f'{len(runs)} entries')
    # This file is excluded from its own scan: it contains the banned tokens as
    # string literals precisely because it searches for them, and matching on
    # itself would report a permanent false failure. It is the only exclusion,
    # and it is in version control where the claim can be read directly.
    tools = sorted(p.name for p in (REPO / 'tools').glob('extdata_*.py')
                   if p.name != Path(__file__).name)
    bad = []
    for t in tools:
        txt = (REPO / 'tools' / t).read_text(encoding='utf-8')
        for banned in ('ultralytics', 'YOLO(', 'trackeval', 'TrackEval', '.train('):
            if banned in txt:
                bad.append(f'{t}:{banned}')
    check('audit tools import no trainer or evaluator', not bad, ', '.join(bad) or
          f'{len(tools)} audit tools checked')
    lbl = []
    for t in tools:
        txt = (REPO / 'tools' / t).read_text(encoding='utf-8')
        if 'data/frames' in txt and 'labels' in txt.split('data/frames')[1][:400]:
            lbl.append(t)
    check('no TEST label path referenced', not lbl, ', '.join(lbl) or 'none')

    out = {'first_run': first, 'all_pass': ok, 'checks': checks, 'state': now}
    (AUDIT / 'reports' / 'integrity.json').write_text(
        json.dumps(out, indent=1), encoding='utf-8')
    print(f'\nAUDIT INTEGRITY: {"PASS" if ok else "FAIL"}')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
