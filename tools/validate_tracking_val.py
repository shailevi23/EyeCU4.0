#!/usr/bin/env python
"""
Validate the frozen tracking-input package. Read-only; safe to rerun.

Checks the properties a downstream tracker comparison silently depends on: the
right windows, every frame present exactly once, no ball in the human stream,
no TEST source, and hashes matching the manifest. A freeze that drifts from its
manifest is worse than no freeze, because results would still look reproducible.

Exit code 0 = valid, 1 = invalid.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trackers.detector import CLASSES, HUMAN_CLASSES  # noqa: E402
from tools.build_derived_train import TEST_MATCHES, VAL_MATCHES  # noqa: E402

EXPECTED = {'austin_fc_vs__club_tijuana_284', 'bayern_munich_3-1_chelsea_228',
            'women_1_239', 'youth_premier_league_1133'}
N_FRAMES = 300


def validate(root: Path):
    errors, checks = [], 0

    def check(ok, msg):
        nonlocal checks
        checks += 1
        if not ok:
            errors.append(msg)

    mpath = root / 'manifest.json'
    if not mpath.exists():
        return ['manifest.json missing'], 0
    man = json.loads(mpath.read_text(encoding='utf-8'))

    seqs = {w['sequence'] for w in man['windows']}
    check(seqs == EXPECTED, f'window set mismatch: {sorted(seqs ^ EXPECTED)}')
    check(len(man['windows']) == 4, f'expected 4 windows, got {len(man["windows"])}')

    for w in man['windows']:
        tag = w['sequence']
        check(w['match'] in VAL_MATCHES, f'{tag}: source not in VAL: {w["match"]}')
        check(w['match'] not in TEST_MATCHES, f'{tag}: TEST source: {w["match"]}')
        check(w['frame_count'] == N_FRAMES, f'{tag}: {w["frame_count"]} frames, need {N_FRAMES}')
        check(w['native_fps'] > 0, f'{tag}: non-positive fps')
        check(w['frame_width'] > 0 and w['frame_height'] > 0, f'{tag}: bad frame size')

        f = root / w['detections_file']
        if not f.exists():
            errors.append(f'{tag}: detections file missing'); checks += 1; continue
        text = f.read_text(encoding='utf-8')
        check(hashlib.sha256(text.encode('utf-8')).hexdigest() == w['detections_sha256'],
              f'{tag}: SHA256 mismatch -- file changed since the freeze')

        rows = [json.loads(l) for l in text.splitlines() if l.strip()]
        check(len(rows) == N_FRAMES, f'{tag}: {len(rows)} rows, need {N_FRAMES}')
        idx = [r['frame'] for r in rows]
        check(idx == list(range(N_FRAMES)),
              f'{tag}: frame indices not 0..{N_FRAMES-1} exactly once '
              f'(duplicate or missing frame)')

        n_det = 0
        for r in rows:
            for d in r['detections']:
                n_det += 1
                c = d['class']
                check(c in CLASSES, f'{tag} f{r["frame"]}: unknown class {c!r}')
                check(c in HUMAN_CLASSES,
                      f'{tag} f{r["frame"]}: non-human class {c!r} in the human '
                      f'tracking stream')
                check(c != 'ball', f'{tag} f{r["frame"]}: BALL present in human stream')
                check('tracker_id' not in d and 'id' not in d,
                      f'{tag} f{r["frame"]}: tracker id present in a frozen detection')
                b = d['bbox']
                check(isinstance(b, list) and len(b) == 4, f'{tag} f{r["frame"]}: malformed bbox')
                if isinstance(b, list) and len(b) == 4:
                    check(b[2] > b[0] and b[3] > b[1],
                          f'{tag} f{r["frame"]}: non-positive bbox extent {b}')
                    check(b[0] >= -1 and b[1] >= -1
                          and b[2] <= w['frame_width'] + 1
                          and b[3] <= w['frame_height'] + 1,
                          f'{tag} f{r["frame"]}: bbox outside frame {b}')
                cf = d['confidence']
                check(isinstance(cf, float) and 0.0 <= cf <= 1.0,
                      f'{tag} f{r["frame"]}: invalid confidence {cf}')
                check(cf >= man['detector']['confidence'] - 1e-9,
                      f'{tag} f{r["frame"]}: confidence {cf} below the frozen threshold')
        check(n_det == w['human_detections'],
              f'{tag}: manifest says {w["human_detections"]} detections, file has {n_det}')

    ck = Path(man['detector']['checkpoint'])
    if ck.exists():
        h = hashlib.sha256()
        with open(ck, 'rb') as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b''):
                h.update(chunk)
        check(h.hexdigest() == man['detector']['checkpoint_sha256'],
              'checkpoint SHA256 mismatch -- weights differ from the freeze')
    else:
        errors.append(f'checkpoint missing: {ck}'); checks += 1

    return errors, checks


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', default='data/tracking_val_v1')
    args = ap.parse_args()
    errors, checks = validate(Path(args.root))
    print(f'{checks} checks run')
    if errors:
        print(f'\nFAILED ({len(errors)}):')
        for e in errors[:40]:
            print(f'  - {e}')
        if len(errors) > 40:
            print(f'  ... and {len(errors)-40} more')
        sys.exit(1)
    print('VALID: frozen tracking inputs match their manifest')


if __name__ == '__main__':
    main()
