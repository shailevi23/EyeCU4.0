#!/usr/bin/env python
"""
Drive the tracker bake-off end to end, from frozen inputs only.

Everything the modern trackers need runs in an ISOLATED interpreter outside
this repository, because EyeCU has a local package called `trackers/` and the
external Roboflow package imports under the same name. No sys.path hack, no
change to the active environment.

    LEGACY_SUPERVISION_BYTETRACK   active env, supervision 0.26.1, accepted
                                   store, real production semantics
    modern trackers                isolated venv, trackers 2.6.0, candidate
                                   store, per profile

The detector is never run: every tracker consumes the same frozen evidence.
Outputs land in the TrackEval tracker layout, are validated before any metric
is computed, and a tracker that emits invalid rows has the failure recorded
rather than repaired.
"""

import argparse
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

MODERN = ['ByteTrackTracker', 'BoTSORTTracker', 'CBIoUTracker', 'OCSORTTracker']
PROFILES = {
    # profile -> (store, min_conf EXCLUSIVE)
    'LIBRARY_DEFAULTS': ('candidates', 0.01),
    'EYECU_SCORE_POLICY_V1': ('candidates', 0.10),
}
LEGACY = 'LEGACY_SUPERVISION_BYTETRACK'
SPLIT = 'EyeCU-val'


def validate_output(path: Path, n_frames: int, seq: str):
    """Structural gate. Never repairs; only reports."""
    issues = []
    lines = [l for l in path.read_text(encoding='utf-8').splitlines() if l.strip()]
    per_frame = defaultdict(list)
    for ln, l in enumerate(lines, 1):
        p = l.split(',')
        if len(p) != 9:
            issues.append(f'{seq} line {ln}: {len(p)} columns'); continue
        try:
            f, i = int(p[0]), int(p[1])
            x, y, w, h = (float(v) for v in p[2:6])
        except ValueError:
            issues.append(f'{seq} line {ln}: unparseable'); continue
        if not (1 <= f <= n_frames):
            issues.append(f'{seq} line {ln}: frame {f} outside 1..{n_frames}')
        if w <= 0 or h <= 0:
            issues.append(f'{seq} line {ln}: non-positive box {w}x{h}')
        if any(v != v or abs(v) == float('inf') for v in (x, y, w, h)):
            issues.append(f'{seq} line {ln}: non-finite coordinate')
        if p[7] != '1':
            issues.append(f'{seq} line {ln}: class {p[7]} (must be 1)')
        if p[8] != '1':
            issues.append(f'{seq} line {ln}: visibility {p[8]} (must be 1)')
        per_frame[f].append(i)
    for f, ids in per_frame.items():
        dup = [i for i, c in Counter(ids).items() if c > 1]
        if dup:
            issues.append(f'{seq} frame {f}: duplicate tracker id {dup[:3]}')
    stats = {
        'rows': len(lines),
        'frames_with_output': len(per_frame),
        'distinct_ids': len({i for v in per_frame.values() for i in v}),
        'issues': issues[:20],
        'issue_count': len(issues),
        'valid': not issues,
    }
    return stats


def track_lengths(path: Path):
    per_id = Counter()
    for l in path.read_text(encoding='utf-8').splitlines():
        if l.strip():
            per_id[int(l.split(',')[1])] += 1
    return sorted(per_id.values())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--iso', required=True, help='isolated bake-off env dir')
    ap.add_argument('--out', default='experiments/tracking_v2/bakeoff')
    args = ap.parse_args()

    iso = Path(args.iso)
    py = iso / 'venv' / 'Scripts' / 'python.exe'
    runner = iso / 'bakeoff_runner.py'
    out = REPO / args.out
    gt_root = REPO / 'data' / 'tracking_val_gt'
    v1 = REPO / 'data' / 'tracking_val_v1'
    man = json.loads((gt_root / 'manifest.json').read_text(encoding='utf-8'))
    seqs = sorted(man['sequences'], key=lambda s: s['sequence'])

    summary = {'runs': {}, 'validation': {}, 'diagnostics': {}}

    for profile, (store, min_conf) in PROFILES.items():
        for tracker in MODERN:
            for s in seqs:
                seq = s['sequence']
                dst = (out / 'outputs' / profile / SPLIT / tracker / 'data'
                       / f'{seq}.txt')
                meta = (out / 'meta' / profile / tracker / f'{seq}.json')
                dst.parent.mkdir(parents=True, exist_ok=True)
                meta.parent.mkdir(parents=True, exist_ok=True)
                cmd = [str(py), str(runner), '--tracker', tracker,
                       '--profile', profile,
                       '--candidates', str(v1 / store / f'{seq}.jsonl'),
                       '--frames', str(gt_root / 'sequences' / seq / 'img1'),
                       '--out', str(dst), '--meta-out', str(meta),
                       '--fps', str(s['native_fps']), '--min-conf', str(min_conf)]
                r = subprocess.run(cmd, cwd=str(iso), capture_output=True, text=True)
                if r.returncode != 0:
                    print(f'FAILED {tracker} {profile} {seq}\n{r.stderr[-800:]}')
                    summary['runs'][f'{profile}/{tracker}/{seq}'] = {
                        'ok': False, 'stderr': r.stderr[-2000:]}
                    continue
                st = validate_output(dst, s['frame_count'], seq)
                summary['runs'][f'{profile}/{tracker}/{seq}'] = {
                    'ok': True, 'meta': json.loads(meta.read_text(encoding='utf-8'))}
                summary['validation'][f'{profile}/{tracker}/{seq}'] = st
                tl = track_lengths(dst)
                summary['diagnostics'][f'{profile}/{tracker}/{seq}'] = {
                    'distinct_ids': st['distinct_ids'],
                    'median_track_length': tl[len(tl)//2] if tl else 0,
                    'track_length_min': tl[0] if tl else 0,
                    'track_length_max': tl[-1] if tl else 0,
                    'tracks_len_1': sum(1 for x in tl if x == 1),
                    'tracks_len_ge_100': sum(1 for x in tl if x >= 100),
                }
                print(f'  {profile:<22}{tracker:<18}{seq:<32}'
                      f'{st["rows"]:>6} rows  {st["distinct_ids"]:>4} ids  '
                      f'{"OK" if st["valid"] else "INVALID"}')

    # legacy, in the active environment, on the accepted store
    from tools.bakeoff_legacy_runner import run as run_legacy
    for s in seqs:
        seq = s['sequence']
        for profile in PROFILES:
            dst = out / 'outputs' / profile / SPLIT / LEGACY / 'data' / f'{seq}.txt'
            meta = out / 'meta' / profile / LEGACY / f'{seq}.json'
            if profile == 'LIBRARY_DEFAULTS':
                # the baseline has one set of semantics; it is copied into both
                # profile folders so TrackEval can score it alongside each, and
                # is NOT re-parameterised per profile
                pass
            run_legacy(v1 / 'detections' / f'{seq}.jsonl', dst, meta, 0.0)
            st = validate_output(dst, s['frame_count'], seq)
            summary['validation'][f'{profile}/{LEGACY}/{seq}'] = st
            tl = track_lengths(dst)
            summary['diagnostics'][f'{profile}/{LEGACY}/{seq}'] = {
                'distinct_ids': st['distinct_ids'],
                'median_track_length': tl[len(tl)//2] if tl else 0,
                'track_length_min': tl[0] if tl else 0,
                'track_length_max': tl[-1] if tl else 0,
                'tracks_len_1': sum(1 for x in tl if x == 1),
                'tracks_len_ge_100': sum(1 for x in tl if x >= 100),
            }

    (out / 'run_summary.json').write_text(json.dumps(summary, indent=1),
                                          encoding='utf-8')
    bad = [k for k, v in summary['validation'].items() if not v['valid']]
    print(f'\noutputs written under {out / "outputs"}')
    print(f'invalid outputs: {len(bad)}')
    for k in bad[:10]:
        print(f'   {k}: {summary["validation"][k]["issues"][:2]}')


if __name__ == '__main__':
    main()
