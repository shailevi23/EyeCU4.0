#!/usr/bin/env python
"""
Validate the identity-GT benchmark. Read-only; safe to rerun.

Two distinct modes, because the dangerous failure is an empty or half-finished
benchmark being mistaken for a finished answer key:

    --stage pre    package structure is ready for a human to annotate
    --stage sequence  one finished sequence, checked on its own
    --stage post   annotations exist and are self-consistent
    --stage final  GT is VERIFIED: post passed AND human QC was confirmed, with
                   the confirmation still matching the artifacts on disk

Status is a three-state machine -- UNANNOTATED, ANNOTATED_PENDING_QC, VERIFIED
-- because a two-state one lets a manifest string edit turn an unreviewed
import into an answer key. `final` re-checks the QC record's hashes, so editing
the status by hand does not make GT evaluable.

Exit 0 = valid, 1 = invalid.
"""

import argparse
import configparser
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trackers.detector import HUMAN_CLASSES  # noqa: E402
from tools.build_derived_train import TEST_MATCHES, VAL_MATCHES  # noqa: E402

EXPECTED = {'austin_fc_vs__club_tijuana_284', 'bayern_munich_3-1_chelsea_228',
            'women_1_239', 'youth_premier_league_1133'}
N_FRAMES = 300
ROLES = set(HUMAN_CLASSES)
MOT_PEDESTRIAN_CLASS = 1


def _check(errors, ok, msg):
    if not ok:
        errors.append(msg)
    return 1


def validate_pre(root: Path):
    """Structure only. Says nothing about identity quality."""
    errors, n = [], 0
    mp = root / 'manifest.json'
    if not mp.exists():
        return ['manifest.json missing'], 1
    man = json.loads(mp.read_text(encoding='utf-8'))

    n += _check(errors, {s['sequence'] for s in man['sequences']} == EXPECTED,
                'sequence set mismatch')
    n += _check(errors, man.get('annotation_schema_version'), 'schema version missing')
    n += _check(errors, 'NOT generated from tracker' in man.get('identity_provenance', ''),
                'manifest does not record identity provenance')
    n += _check(errors, man['target']['ball_excluded'] is True, 'ball not excluded')
    n += _check(errors, set(man['target']['classes']) == ROLES, 'class mapping wrong')

    for s in man['sequences']:
        tag = s['sequence']
        n += _check(errors, s['match'] in VAL_MATCHES, f'{tag}: not a VAL source')
        n += _check(errors, s['match'] not in TEST_MATCHES, f'{tag}: TEST source')
        n += _check(errors, s['frame_count'] == N_FRAMES, f'{tag}: frame count')
        lo, hi = s['source_frame_range']
        n += _check(errors, hi - lo + 1 == N_FRAMES, f'{tag}: source frame range')
        n += _check(errors, s['package_frame_range'] == [1, N_FRAMES],
                    f'{tag}: package numbering must be 1-based 1..{N_FRAMES}')

        sdir = root / 'sequences' / tag
        n += _check(errors, (sdir / 'seqinfo.ini').exists(), f'{tag}: seqinfo.ini missing')
        if (sdir / 'seqinfo.ini').exists():
            c = configparser.ConfigParser(); c.optionxform = str
            c.read(sdir / 'seqinfo.ini', encoding='utf-8')
            n += _check(errors, int(c['Sequence']['seqLength']) == N_FRAMES,
                        f'{tag}: seqinfo seqLength')
            n += _check(errors, int(c['Sequence']['imWidth']) == s['frame_width'],
                        f'{tag}: seqinfo imWidth')
            n += _check(errors, int(c['Sequence']['imHeight']) == s['frame_height'],
                        f'{tag}: seqinfo imHeight')
        imgs = sorted((sdir / 'img1').glob('*.jpg')) if (sdir / 'img1').exists() else []
        n += _check(errors, len(imgs) == N_FRAMES, f'{tag}: {len(imgs)} frames on disk')
        if imgs:
            n += _check(errors, imgs[0].name == '000001.jpg',
                        f'{tag}: frames must start at 000001.jpg (1-based)')

        det = root / s['preannotation_det']
        n += _check(errors, det.exists(), f'{tag}: preannotation det missing')
        if det.exists():
            n += _check(errors,
                        hashlib.sha256(det.read_text(encoding='utf-8').encode()).hexdigest()
                        == s['preannotation_det_sha256'], f'{tag}: preannotation hash')
            for line in det.read_text(encoding='utf-8').splitlines():
                if not line.strip():
                    continue
                p = line.split(',')
                n += _check(errors, p[1].strip() == '-1',
                            f'{tag}: preannotation carries an identity in the id column')
                break
        xml = root / s['preannotation_cvat']
        n += _check(errors, xml.exists(), f'{tag}: preannotation cvat missing')
        if xml.exists():
            t = xml.read_text(encoding='utf-8')
            n += _check(errors, '<track' not in t,
                        f'{tag}: CVAT preannotation contains <track>, i.e. identity')
    return errors, n


def validate_gt_content(root: Path, only=None):
    """
    GT content only: identities, boxes, roles, and the annotation-is-not-the-
    preannotation check. Kept separate from validate_pre so content rules can be
    exercised without a full 1,200-frame package on disk.

    `only` restricts the check to one sequence. Sequences arrive one at a time
    -- annotating 300 frames is a session, not a moment -- and a finished one
    should be checkable immediately, while the mistakes are still fresh. It
    narrows what is checked, never what is required: the benchmark still
    reaches VERIFIED only when every sequence passes.
    """
    errors, n = [], 0
    man = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    for s in man['sequences']:
        tag = s['sequence']
        if only and tag != only:
            continue
        ann = root / s['annotation_file_expected']
        n += _check(errors, ann.exists(), f'{tag}: annotation file missing')
        if not ann.exists():
            continue
        data = json.loads(ann.read_text(encoding='utf-8'))

        ids_seen, per_frame = set(), defaultdict(list)
        for row in data['boxes']:
            f_, i_ = row['frame'], row['id']
            per_frame[f_].append(row)
            ids_seen.add(i_)
            n += _check(errors, isinstance(i_, int) and i_ > 0,
                        f'{tag} f{f_}: identity must be a positive integer, got {i_!r}')
            n += _check(errors, 1 <= f_ <= N_FRAMES, f'{tag}: frame {f_} out of range')
            x1, y1, x2, y2 = row['bbox']
            n += _check(errors, x2 > x1 and y2 > y1, f'{tag} f{f_} id{i_}: bbox extent')
            n += _check(errors,
                        -1 <= x1 and -1 <= y1
                        and x2 <= s['frame_width'] + 1 and y2 <= s['frame_height'] + 1,
                        f'{tag} f{f_} id{i_}: bbox outside frame')
            n += _check(errors, row.get('role') in ROLES,
                        f'{tag} f{f_} id{i_}: role {row.get("role")!r} not in {sorted(ROLES)}')
            n += _check(errors, row.get('class') != 'ball',
                        f'{tag} f{f_}: ball present in human GT')
            # a boolean, never a fabricated visibility fraction
            n += _check(errors, isinstance(row.get('occluded', False), bool),
                        f'{tag} f{f_} id{i_}: occluded must be a boolean, got '
                        f'{row.get("occluded")!r}')

        for f_, rows in per_frame.items():
            ids = [r['id'] for r in rows]
            n += _check(errors, len(ids) == len(set(ids)),
                        f'{tag} f{f_}: duplicate GT id in one frame')
            for a in range(len(rows)):
                for b in range(a + 1, len(rows)):
                    n += _check(errors, rows[a]['bbox'] != rows[b]['bbox'],
                                f'{tag} f{f_}: duplicate identical boxes')

        roles = root / s['roles_expected']
        n += _check(errors, roles.exists(), f'{tag}: role sidecar missing')
        if roles.exists():
            rj = json.loads(roles.read_text(encoding='utf-8'))
            try:
                declared = {int(k) for k in rj.get('identity_roles', {})}
            except (TypeError, ValueError):
                declared = None
                errors.append(f'{tag}: role sidecar has a non-integer identity key')
                n += 1
            if declared is not None:
                n += _check(errors, declared == ids_seen,
                            f'{tag}: role sidecar identities '
                            f'{sorted(declared ^ ids_seen)} do not match '
                            f'annotated identities')
            for v in rj.get('identity_roles', {}).values():
                n += _check(errors, v in ROLES, f'{tag}: role {v!r} invalid')

        # the annotation must not simply be the preannotation
        pre = (root / s['preannotation_det']).read_text(encoding='utf-8')
        n += _check(errors,
                    hashlib.sha256(json.dumps(data['boxes'], sort_keys=True).encode()
                                   ).hexdigest() != hashlib.sha256(pre.encode()).hexdigest(),
                    f'{tag}: annotation is byte-identical to the preannotation')
    return errors, n


def validate_post(root: Path):
    """A complete, manually verified GT. Never passes on an unannotated package."""
    errors, n = validate_pre(root)
    man = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    status = man.get('identity_gt_status')
    n += _check(errors, status in ('ANNOTATED_PENDING_QC', 'VERIFIED'),
                f'identity_gt_status is {status!r}; nothing has been imported')
    if status not in ('ANNOTATED_PENDING_QC', 'VERIFIED'):
        return errors, n
    e2, n2 = validate_gt_content(root)
    return errors + e2, n + n2


def validate_verified(root: Path):
    """
    The identity-safety gate: status is VERIFIED, the QC confirmation still
    matches the annotations on disk, and the content rules hold.

    Separate from validate_pre so that the gate can be applied to GT content
    alone. Structure is a build-time property; this is the part that decides
    whether numbers computed against this GT mean anything.
    """
    from tools.confirm_tracking_gt_qc import qc_record_valid
    errors, n = [], 0
    man = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    status = man.get('identity_gt_status')
    n += _check(errors, status == 'VERIFIED',
                f'identity_gt_status is {status!r}; only VERIFIED GT may be '
                f'evaluated')
    ok, why = qc_record_valid(root, man)
    n += _check(errors, ok, f'QC confirmation invalid: {why}')
    if status == 'VERIFIED':
        e2, n2 = validate_gt_content(root)
        errors += e2
        n += n2
    return errors, n


def validate_final(root: Path):
    """Everything: package structure plus the verified-identity gate."""
    errors, n = validate_pre(root)
    e2, n2 = validate_verified(root)
    return errors + e2, n + n2


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', default='data/tracking_val_gt')
    ap.add_argument('--stage', choices=['pre', 'post', 'final', 'sequence'],
                    default='pre')
    ap.add_argument('--sequence', default=None,
                    help='with --stage sequence: check this sequence alone')
    args = ap.parse_args()
    if args.stage == 'sequence':
        if not args.sequence:
            sys.exit('--stage sequence requires --sequence <tag>')
        errors, n = validate_gt_content(Path(args.root), only=args.sequence)
    else:
        fn = {'pre': validate_pre, 'post': validate_post,
              'final': validate_final}[args.stage]
        errors, n = fn(Path(args.root))
    print(f'{n} checks run  (stage={args.stage})')
    if errors:
        print(f'\nFAILED ({len(errors)}):')
        for e in errors[:40]:
            print(f'  - {e}')
        sys.exit(1)
    if args.stage == 'sequence':
        print(f'VALID: {args.sequence} content is self-consistent')
        print('NOTE: one sequence only. The benchmark is not VERIFIED, and no '
              'other sequence was checked.')
    elif args.stage == 'pre':
        print('VALID: package structure is ready for manual annotation')
        print('NOTE: this says nothing about identity quality. Identity GT does '
              'not exist until --stage post passes.')
    elif args.stage == 'post':
        print('VALID: annotations are self-consistent')
        print('NOTE: this is not yet VERIFIED GT. Run '
              'tools/confirm_tracking_gt_qc.py, then --stage final.')
    else:
        print('VALID: complete, human-verified identity GT')


if __name__ == '__main__':
    main()
