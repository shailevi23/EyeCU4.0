#!/usr/bin/env python
"""
Record human QC confirmation and promote the benchmark to VERIFIED.

The GT status is a three-state machine:

    UNANNOTATED            package built, no annotation
    ANNOTATED_PENDING_QC   CVAT import succeeded
    VERIFIED               post validation passed AND a human confirmed QC

Editing the manifest string is deliberately not enough. Promotion writes a QC
record containing the SHA-256 of every annotation and role file, and downstream
tools re-check those hashes: if anyone edits the manifest by hand, or changes an
annotation after confirming, the record no longer matches and the benchmark
falls back to unusable. The point is that a benchmark cannot become evaluable
by assertion, only by evidence.

Refuses to promote while post validation fails or QC issues are outstanding,
unless the reviewer explicitly accepts them with --accept-qc-issues and a
reason, which is recorded.
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.render_tracking_gt_qc import qc  # noqa: E402
from tools.validate_tracking_gt import validate_post  # noqa: E402

STATUS_PENDING = 'ANNOTATED_PENDING_QC'
STATUS_VERIFIED = 'VERIFIED'
QC_RECORD = 'qc/qc_confirmation.json'


def artifact_hashes(root: Path, man):
    h = {}
    for s in man['sequences']:
        for key in ('annotation_file_expected', 'roles_expected'):
            p = root / s[key]
            if p.exists():
                h[s[key]] = hashlib.sha256(p.read_bytes()).hexdigest()
    return h


def qc_record_valid(root: Path, man):
    """True when a QC record exists and still matches the artifacts on disk."""
    rec_path = root / QC_RECORD
    if not rec_path.exists():
        return False, 'no QC confirmation record'
    rec = json.loads(rec_path.read_text(encoding='utf-8'))
    now = artifact_hashes(root, man)
    if rec.get('artifact_sha256') != now:
        changed = [k for k in now if rec.get('artifact_sha256', {}).get(k) != now[k]]
        return False, f'annotations changed since QC confirmation: {changed}'
    if not rec.get('reviewer') or not rec.get('confirmed'):
        return False, 'QC record is not a confirmation'
    return True, 'ok'


def promote_to_verified(root: Path, man, reviewer, qc_issue_count=0,
                        accepted_reason=None):
    """Write the QC record and flip the status. The only writer of VERIFIED."""
    rec = {
        'confirmed': True,
        'reviewer': reviewer,
        'confirmed_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'qc_issue_count': qc_issue_count,
        'accepted_qc_issues_reason': accepted_reason,
        'artifact_sha256': artifact_hashes(root, man),
    }
    (root / 'qc').mkdir(exist_ok=True)
    (root / QC_RECORD).write_text(json.dumps(rec, indent=2), encoding='utf-8')
    man['identity_gt_status'] = STATUS_VERIFIED
    man['qc_confirmation'] = QC_RECORD
    (root / 'manifest.json').write_text(
        json.dumps(man, indent=2, ensure_ascii=False), encoding='utf-8')
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', default='data/tracking_val_gt')
    ap.add_argument('--reviewer', required=True, help='who reviewed the QC output')
    ap.add_argument('--confirm', action='store_true',
                    help='required; without it this only reports')
    ap.add_argument('--accept-qc-issues', default=None,
                    help='reason for accepting outstanding QC issues')
    args = ap.parse_args()

    root = Path(args.root)
    mp = root / 'manifest.json'
    man = json.loads(mp.read_text(encoding='utf-8'))

    if man.get('identity_gt_status') == 'UNANNOTATED':
        sys.exit('REFUSING: nothing has been imported yet.')

    errors, n = validate_post(root)
    print(f'post validation: {n} checks, {len(errors)} errors')
    for e in errors[:10]:
        print(f'  - {e}')
    # validate_post requires ANNOTATED; while pending, ignore only that one error
    blocking = [e for e in errors if 'identity_gt_status' not in e]
    if blocking:
        sys.exit('REFUSING: post validation failed. Fix the annotation first.')

    issues = []
    for s in man['sequences']:
        issues += qc(root, s['sequence'], root / 'qc' / s['sequence'],
                     stride=1, render=False)
    print(f'\nQC issues: {len(issues)}')
    if issues and not args.accept_qc_issues:
        for m in issues[:15]:
            print(f'  - {m}')
        sys.exit('REFUSING: QC issues outstanding. Fix them, or re-run with '
                 '--accept-qc-issues "<reason>" to record an explicit decision.')

    if not args.confirm:
        print('\nDry report only. Re-run with --confirm to promote to VERIFIED.')
        return

    promote_to_verified(root, man, args.reviewer, len(issues),
                        args.accept_qc_issues)
    print(f'\nstatus -> {STATUS_VERIFIED}  (reviewer: {args.reviewer})')
    print('MOT export is now permitted: python tools/export_tracking_gt_mot.py')


if __name__ == '__main__':
    main()
