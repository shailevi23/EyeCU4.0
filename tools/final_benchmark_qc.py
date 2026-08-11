#!/usr/bin/env python
"""
Whole-benchmark final QC for EyeCU-Tracking-Val-v1.1. Read-only.

The last gate before identity GT may be confirmed. It checks STRUCTURE across
all three sequences at once -- the things that are wrong regardless of what
anyone intended -- and reports. It does not promote anything: only
tools/confirm_tracking_gt_qc.py does that, and only with a human reviewer.

WHAT IT WILL NOT DO
-------------------
It never reinterprets a human-confirmed semantic label. Bayern contains exactly
one goalkeeper because one goalkeeper is visible in that window, and the
annotator has confirmed it. A check that flagged that as suspicious would be
substituting a football expectation for an observation, which is precisely the
failure the whole benchmark exists to avoid. So there is no rule here about how
many goalkeepers a match "should" have, no positional inference, and no
detector or tracker input of any kind.

Role composition is REPORTED, never judged.

WHAT IT CHECKS
--------------
    role sidecar consistency   every annotated identity declared, and the
                               declared role matches every box of that identity
    role changes               an identity whose role is not constant is
                               structurally impossible, whatever the labels are
    duplicate identities       the same id twice in one frame
    malformed boxes            non-positive extent, outside the frame, NaN
    malformed tracks           empty tracks, non-integer or non-positive ids,
                               frames outside 1..N, unordered rows
    coverage                   frames with no target, and per-sequence totals
    acceptance                 every sequence carries a human acceptance record
                               and no sequence has an open review event
    evidence                   the raw CVAT export still hashes to what the
                               provenance record says it did
"""

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.validate_tracking_gt import (EXPECTED, N_FRAMES,  # noqa: E402
                                        validate_gt_content, validate_pre)
from trackers.detector import HUMAN_CLASSES  # noqa: E402

ACCEPTED_STATES = {'IDENTITY_ACCEPTED_PENDING_FINAL_QC'}


def check(rows, name, ok, detail=''):
    rows.append({'check': name, 'ok': bool(ok), 'detail': detail})
    return ok


def audit_sequence(root: Path, s: dict):
    tag = s['sequence']
    rows = []
    ann = json.loads((root / s['annotation_file_expected']).read_text(encoding='utf-8'))
    sidecar = json.loads((root / s['roles_expected']).read_text(encoding='utf-8'))
    declared = {int(k): v for k, v in sidecar['identity_roles'].items()}
    boxes = ann['boxes']

    per_id, per_frame = defaultdict(list), defaultdict(list)
    for b in boxes:
        per_id[b['id']].append(b)
        per_frame[b['frame']].append(b)

    check(rows, 'role sidecar covers exactly the annotated identities',
          set(declared) == set(per_id),
          str(sorted(set(declared) ^ set(per_id))))
    mismatched = {i: (declared.get(i), sorted({b['role'] for b in v}))
                  for i, v in per_id.items()
                  if {b['role'] for b in v} != {declared.get(i)}}
    check(rows, 'declared role matches every box of that identity',
          not mismatched, str(dict(list(mismatched.items())[:3])))
    unstable = {i: sorted({b['role'] for b in v}) for i, v in per_id.items()
                if len({b['role'] for b in v}) > 1}
    check(rows, 'no identity changes role mid-sequence', not unstable,
          str(unstable))
    check(rows, 'every role is one of the three target classes',
          set(declared.values()) <= set(HUMAN_CLASSES),
          str(sorted(set(declared.values()))))

    dup = {f: [i for i, c in Counter(b['id'] for b in v).items() if c > 1]
           for f, v in per_frame.items()
           if len({b['id'] for b in v}) != len(v)}
    check(rows, 'no duplicate identity within a frame', not dup,
          str(dict(list(dup.items())[:3])))

    bad_extent = [b for b in boxes
                  if not (b['bbox'][2] > b['bbox'][0] and b['bbox'][3] > b['bbox'][1])]
    oob = [b for b in boxes
           if b['bbox'][0] < -1 or b['bbox'][1] < -1
           or b['bbox'][2] > s['frame_width'] + 1
           or b['bbox'][3] > s['frame_height'] + 1]
    nan = [b for b in boxes if any(v != v for v in b['bbox'])]
    check(rows, 'all boxes have positive extent', not bad_extent, str(bad_extent[:2]))
    check(rows, f'all boxes within {s["frame_width"]}x{s["frame_height"]}',
          not oob, str(oob[:2]))
    check(rows, 'no NaN coordinates', not nan, str(nan[:2]))

    check(rows, 'identities are positive integers',
          all(isinstance(i, int) and i > 0 for i in per_id), str(sorted(per_id)[:5]))
    check(rows, 'no empty track', all(per_id.values()))
    check(rows, f'frames within 1..{N_FRAMES}',
          all(1 <= b['frame'] <= N_FRAMES for b in boxes))
    check(rows, 'rows ordered by (frame, id)',
          boxes == sorted(boxes, key=lambda r: (r['frame'], r['id'])))
    check(rows, 'occluded is boolean on every box',
          all(isinstance(b.get('occluded', False), bool) for b in boxes))

    counts = [len(per_frame.get(f, [])) for f in range(1, N_FRAMES + 1)]
    check(rows, 'no frame without a target', all(c > 0 for c in counts),
          f'{sum(1 for c in counts if c == 0)} empty frames')

    dec_path = root / 'qc_identity' / tag / 'identity_gap_decisions.json'
    dec = json.loads(dec_path.read_text(encoding='utf-8')) if dec_path.exists() else None
    check(rows, 'human acceptance record present', dec is not None, str(dec_path))
    if dec:
        check(rows, 'no open review event', dec.get('open_events') == 0,
              str(dec.get('open_events')))
        check(rows, 'decisions used no tracker output, embeddings or ReID',
              not any(dec.get(k) for k in
                      ('tracker_output_used', 'embeddings_used', 'reid_used')))
        check(rows, 'annotations were not altered by review',
              dec.get('annotations_altered') is False)

    prov = root / 'cvat_exports' / f'{tag}.provenance.json'
    if prov.exists():
        p = json.loads(prov.read_text(encoding='utf-8'))
        xml = root / 'cvat_exports' / f'{tag}.xml'
        check(rows, 'raw CVAT export still matches its recorded hash',
              hashlib.sha256(xml.read_bytes()).hexdigest()
              == p['annotations_xml_sha256'])
        check(rows, 'provenance records the export as unmodified',
              p.get('raw_export_modified') is False)

    stats = {
        'identities': len(per_id),
        'role_composition': dict(Counter(declared.values())),
        'boxes': len(boxes),
        'occluded': sum(b.get('occluded', False) for b in boxes),
        'targets_per_frame': {
            'min': min(counts), 'median': sorted(counts)[len(counts) // 2],
            'max': max(counts), 'mean': round(sum(counts) / len(counts), 2)},
        'frames_covered': len(per_frame),
    }
    return rows, stats


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', default='data/tracking_val_gt')
    ap.add_argument('--json-out', default=None)
    args = ap.parse_args()

    root = Path(args.root)
    man = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    report, all_rows = {}, []

    print('EyeCU-Tracking-Val-v1.1  WHOLE-BENCHMARK FINAL QC')
    print('structure only. Human-confirmed semantic labels are reported, '
          'never judged.\n')

    pre_errors, pre_n = validate_pre(root)
    content_errors, content_n = validate_gt_content(root)
    print(f'package structure : {pre_n} checks, {len(pre_errors)} errors')
    print(f'GT content        : {content_n} checks, {len(content_errors)} errors')
    for e in (pre_errors + content_errors)[:10]:
        print(f'   - {e}')

    got = {s['sequence'] for s in man['sequences']}
    bench_rows = []
    check(bench_rows, 'exactly the three v1.1 sequences', got == EXPECTED, str(sorted(got)))
    check(bench_rows, 'every sequence is a distinct match',
          len({s['match'] for s in man['sequences']}) == len(man['sequences']))
    check(bench_rows, 'benchmark is v1.1',
          man.get('benchmark') == 'EyeCU-Tracking-Val-v1.1', str(man.get('benchmark')))
    check(bench_rows, 'not yet claiming VERIFIED without a QC record',
          man['identity_gt_status'] != 'VERIFIED'
          or (root / 'qc' / 'qc_confirmation.json').exists(),
          man['identity_gt_status'])
    check(bench_rows, 'every sequence accepted by a human',
          all(man.get('sequence_review', {}).get(t, {}).get('identity_status')
              in ACCEPTED_STATES for t in EXPECTED),
          str({t: man.get('sequence_review', {}).get(t, {}).get('identity_status')
               for t in sorted(EXPECTED)}))

    print('\nbenchmark-level')
    for r in bench_rows:
        print(f'   [{"ok  " if r["ok"] else "FAIL"}] {r["check"]}'
              + (f'   -- {r["detail"]}' if r['detail'] and not r['ok'] else ''))
    all_rows += bench_rows

    totals = Counter()
    for s in man['sequences']:
        rows, stats = audit_sequence(root, s)
        report[s['sequence']] = {'checks': rows, 'stats': stats}
        all_rows += rows
        totals['identities'] += stats['identities']
        totals['boxes'] += stats['boxes']
        totals['occluded'] += stats['occluded']
        for k, v in stats['role_composition'].items():
            totals[k] += v
        failed = [r for r in rows if not r['ok']]
        print(f'\n{s["sequence"]}  ({len(rows) - len(failed)}/{len(rows)} checks)')
        print(f'   identities {stats["identities"]}  '
              f'roles {stats["role_composition"]}  boxes {stats["boxes"]}  '
              f'occluded {stats["occluded"]}')
        t = stats['targets_per_frame']
        print(f'   targets/frame  min {t["min"]}  median {t["median"]}  '
              f'max {t["max"]}  mean {t["mean"]}')
        for r in failed:
            print(f'   [FAIL] {r["check"]}   -- {r["detail"]}')

    print(f'\nTOTALS  identities {totals["identities"]}   boxes {totals["boxes"]}   '
          f'occluded {totals["occluded"]}')
    print(f'        roles  player {totals["player"]}  '
          f'goalkeeper {totals["goalkeeper"]}  referee {totals["referee"]}'
          f'   (REPORTED, not judged)')

    failed = [r for r in all_rows if not r['ok']]
    n_fail = len(failed) + len(pre_errors) + len(content_errors)
    print(f'\n{len(all_rows) - len(failed)}/{len(all_rows)} structural checks passed'
          f'   validators: {pre_n + content_n} checks, '
          f'{len(pre_errors) + len(content_errors)} errors')
    if n_fail:
        print('\nFINAL QC: FAIL')
    else:
        print('\nFINAL QC: PASS -- structure is sound across all three sequences.')
        print('This is a report, not a promotion. identity GT becomes VERIFIED '
              'only when a human accepts this QC via '
              'tools/confirm_tracking_gt_qc.py.')

    if args.json_out:
        Path(args.json_out).write_text(json.dumps({
            'benchmark': man.get('benchmark'),
            'generated': '2026-08-11',
            'scope': 'structural invariants only',
            'does_not': ('reinterpret human-confirmed semantic labels; role '
                         'composition is reported, never judged'),
            'tracker_output_used': False,
            'benchmark_checks': bench_rows,
            'sequences': report,
            'totals': dict(totals),
            'validator_errors': pre_errors + content_errors,
            'passed': n_fail == 0,
        }, indent=1), encoding='utf-8')
    sys.exit(1 if n_fail else 0)


if __name__ == '__main__':
    main()
