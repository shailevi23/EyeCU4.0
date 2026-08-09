#!/usr/bin/env python
"""
Run a REAL CVAT export through the whole import chain and check every property
the benchmark depends on.

The synthetic fixture in tests/ proves the parser handles the XML we believe
CVAT writes. It cannot prove that the annotator's actual CVAT version, task
settings, and export dialog produce that XML. A version that numbers frames
differently, omits `outside`, or writes tracks as shapes would pass every unit
test and silently corrupt 1,200 frames of annotation. Ten minutes of real
annotation buys certainty before the expensive work starts.

The test task is deliberately tiny and its expected content is declared here in
advance, so the check is against a specification rather than against whatever
came out.

    frames        first 10 of one sequence
    tracks        2, different roles
    identity      each track keeps one id throughout
    interpolation at least one gap between keyframes
    outside       one short invisible interval on track A
    reappearance  track A returns afterwards under the SAME id

Everything runs on a scratch copy. The real package is never written to.

    python tools/smoke_test_cvat_export.py --export path/to/smoke.xml
"""

import argparse
import json
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.confirm_tracking_gt_qc import promote_to_verified  # noqa: E402
from tools.export_tracking_gt_mot import export  # noqa: E402
from tools.import_tracking_gt_cvat import parse_cvat_video  # noqa: E402
from tools.render_tracking_gt_qc import qc  # noqa: E402
from tools.validate_tracking_gt import validate_gt_content  # noqa: E402
from trackers.detector import HUMAN_CLASSES  # noqa: E402

SMOKE_FRAMES = 10


def check(results, name, ok, detail=''):
    results.append((ok, name, detail))
    return ok


def run(export_xml: Path, root: Path, seq: str, keep: Path = None):
    results = []
    tman = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    s = next((x for x in tman['sequences'] if x['sequence'] == seq), None)
    if s is None:
        raise SystemExit(f'unknown sequence {seq!r}')
    W, H = s['frame_width'], s['frame_height']

    # ---- 1. parse the real export -------------------------------------------
    boxes, roles, warn = parse_cvat_video(export_xml, n_frames=s['frame_count'])
    check(results, 'export parses as CVAT for video 1.1 (has <track>)',
          bool(boxes), f'{len(boxes)} boxes, {len(roles)} tracks')
    check(results, 'exactly 2 tracks', len(roles) == 2, f'got {len(roles)}')
    check(results, 'two different roles', len(set(roles.values())) == 2,
          str(dict(roles)))
    check(results, 'roles are EyeCU human classes',
          set(roles.values()) <= set(HUMAN_CLASSES), str(sorted(set(roles.values()))))

    # ---- 2. frame numbering -------------------------------------------------
    frames = sorted({b['frame'] for b in boxes})
    check(results, 'CVAT frame 0 maps to package frame 1', min(frames) == 1,
          f'min package frame = {min(frames)}')
    check(results, f'stays within the first {SMOKE_FRAMES} frames',
          max(frames) <= SMOKE_FRAMES, f'max package frame = {max(frames)}')
    check(results, 'no frame 0 leaked through', 0 not in frames)

    # ---- 3. identity --------------------------------------------------------
    check(results, 'identities are positive ints',
          all(isinstance(i, int) and i > 0 for i in roles), str(sorted(roles)))
    per_id = defaultdict(list)
    for b in boxes:
        per_id[b['id']].append(b['frame'])
    dupes = [(b['frame'], b['id']) for b in boxes]
    check(results, 'no duplicate identity within a frame',
          len(dupes) == len(set(dupes)))
    const_role = {i: {b['role'] for b in boxes if b['id'] == i} for i in per_id}
    check(results, 'each identity keeps one role',
          all(len(v) == 1 for v in const_role.values()),
          str({k: sorted(v) for k, v in const_role.items()}))

    # ---- 4. outside interval and reappearance -------------------------------
    gapped = {i: sorted(set(range(min(f), max(f) + 1)) - set(f))
              for i, f in per_id.items()}
    with_gap = {i: g for i, g in gapped.items() if g}
    check(results, 'one track has an outside interval (no boxes emitted there)',
          len(with_gap) >= 1, str(with_gap))
    check(results, 'that track reappears under the SAME identity',
          any(max(per_id[i]) > max(g) for i, g in with_gap.items()),
          str({i: (min(per_id[i]), g, max(per_id[i])) for i, g in with_gap.items()}))

    # ---- 5. interpolation ---------------------------------------------------
    moved = {i: len({tuple(b['bbox']) for b in boxes if b['id'] == i})
             for i in per_id}
    check(results, 'boxes vary across frames (interpolation decoded)',
          any(v > 1 for v in moved.values()), str(moved))

    # ---- 6. geometry --------------------------------------------------------
    bad = [b for b in boxes
           if not (b['bbox'][2] > b['bbox'][0] and b['bbox'][3] > b['bbox'][1])]
    check(results, 'all boxes have positive extent', not bad, str(bad[:2]))
    oob = [b for b in boxes
           if b['bbox'][0] < -1 or b['bbox'][1] < -1
           or b['bbox'][2] > W + 1 or b['bbox'][3] > H + 1]
    check(results, f'all boxes lie within the {W}x{H} frame', not oob,
          str(oob[:2]))

    # ---- 7. full chain on a scratch copy ------------------------------------
    tmp = Path(tempfile.mkdtemp(prefix='cvat_smoke_'))
    try:
        dst = tmp / 'gt'
        shutil.copytree(root, dst, ignore=shutil.ignore_patterns('img1', 'qc', 'mot'))
        man = json.loads((dst / 'manifest.json').read_text(encoding='utf-8'))
        man['sequences'] = [x for x in man['sequences'] if x['sequence'] == seq]
        (dst / 'manifest.json').write_text(json.dumps(man), encoding='utf-8')
        exports = tmp / 'cvat'
        exports.mkdir()
        shutil.copy2(export_xml, exports / f'{seq}.xml')

        from tools.import_tracking_gt_cvat import main as import_main
        argv = sys.argv
        sys.argv = ['import', '--root', str(dst), '--exports', str(exports)]
        try:
            import_main()
        finally:
            sys.argv = argv
        man = json.loads((dst / 'manifest.json').read_text(encoding='utf-8'))
        check(results, 'importer writes canonical JSON and lands in PENDING_QC',
              man['identity_gt_status'] == 'ANNOTATED_PENDING_QC',
              man['identity_gt_status'])

        rj = json.loads((dst / man['sequences'][0]['roles_expected']
                         ).read_text(encoding='utf-8'))
        check(results, 'role sidecar generated from track labels',
              {int(k): v for k, v in rj['identity_roles'].items()} == roles,
              str(rj['identity_roles']))

        errors, _ = validate_gt_content(dst)
        check(results, 'validator accepts the imported GT', not errors,
              str(errors[:3]))

        issues = qc(dst, seq, tmp / 'qcout', stride=1, render=False)
        check(results, 'QC renderer runs and reports no structural issue',
              not issues, str(issues[:3]))

        promote_to_verified(dst, man, reviewer='smoke-test')
        export(dst, tmp / 'mot')
        gt = tmp / 'mot' / 'EyeCU-val' / seq / 'gt' / 'gt.txt'
        rows = [r.split(',') for r in gt.read_text(encoding='utf-8').splitlines()]
        check(results, 'MOT export row count equals canonical box count',
              len(rows) == len(boxes), f'{len(rows)} vs {len(boxes)}')
        check(results, 'MOT frames are 1-based',
              min(int(r[0]) for r in rows) == 1)
        check(results, 'MOT conf == 1 (TrackEval drops conf 0 GT)',
              all(int(r[6]) == 1 for r in rows))
        check(results, 'MOT class == 1 (pedestrian)',
              all(int(r[7]) == 1 for r in rows))
        check(results, 'MOT identities equal the CVAT identities',
              {int(r[1]) for r in rows} == set(roles))
        first = rows[0]
        b0 = next(b for b in boxes
                  if b['frame'] == int(first[0]) and b['id'] == int(first[1]))
        x1, y1, x2, y2 = b0['bbox']
        got = [round(float(v), 2) for v in first[2:6]]
        want = [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)]
        check(results, 'MOT geometry is x,y,w,h of the canonical box',
              got == want, f'{got} vs {want}')
        if keep:
            shutil.copytree(dst, keep, dirs_exist_ok=True)
    finally:
        if not keep:
            shutil.rmtree(tmp, ignore_errors=True)

    for w in warn:
        print(f'warning: {w}')
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--export', required=True, help='the real CVAT XML export')
    ap.add_argument('--root', default='data/tracking_val_gt')
    ap.add_argument('--sequence', default='women_1_239')
    ap.add_argument('--keep', default=None,
                    help='copy the scratch package here for inspection')
    args = ap.parse_args()

    results = run(Path(args.export), Path(args.root), args.sequence,
                  Path(args.keep) if args.keep else None)
    print()
    for ok, name, detail in results:
        mark = 'PASS' if ok else 'FAIL'
        print(f'  [{mark}] {name}' + (f'   -- {detail}' if detail else ''))
    failed = [r for r in results if not r[0]]
    print(f'\n{len(results) - len(failed)}/{len(results)} checks passed')
    if failed:
        print('\nSTOP. The real CVAT export does not match what the importer '
              'expects.\nFix the importer (or the export settings) before any '
              'full annotation begins.')
        sys.exit(1)
    print('Real CVAT export matches the importer contract. Full annotation may '
          'begin.')


if __name__ == '__main__':
    main()
