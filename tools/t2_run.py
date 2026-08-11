#!/usr/bin/env python
"""
Execute Tracking Experiment T2 exactly as frozen. Read-only on every input.

Runs the three candidates from the frozen detections, in a freshly created
isolated environment, and checks configuration equivalence against the
instantiated-object configs preserved by the closed bake-off -- not against
documentation, and not against this file's idea of what the defaults are.

Nothing here is tunable. There is no parameter argument, because a parameter
that can be passed is a parameter that can be changed after seeing a result.
"""

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
EXP = REPO / 'experiments' / 'tracking_v2'
BAKE = EXP / 'bakeoff'
SPEC_SHA = '5f8d2a752c57aef1fd4fd82719d2b6bfd9bc8c7cb9e28b9433b114761a756f60'
LEGACY = 'LEGACY_SUPERVISION_BYTETRACK'
CANDIDATES = ['CBIoUTracker', 'BoTSORTTracker']
PROFILE = 'LIBRARY_DEFAULTS'          # the exact configuration T2 qualifies
MIN_CONF = 0.01                       # as in the recorded LIBRARY_DEFAULTS runs
SPLIT = 'EyeCU-val'

# parameters that are not part of the tracker's configuration identity
IGNORED_CONFIG_KEYS = {'frame_rate'}


def config_equivalent(new: dict, old: dict):
    """Key-by-key comparison of the instantiated-object configs."""
    keys = (set(new) | set(old)) - IGNORED_CONFIG_KEYS
    diffs = {k: (old.get(k, '<missing>'), new.get(k, '<missing>'))
             for k in sorted(keys) if new.get(k) != old.get(k)}
    return not diffs, diffs


def validate(path: Path, n_frames: int):
    issues, per_frame = [], defaultdict(list)
    for ln, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        p = line.split(',')
        if len(p) != 9:
            issues.append(f'line {ln}: {len(p)} columns'); continue
        f, i = int(p[0]), int(p[1])
        x, y, w, h = (float(v) for v in p[2:6])
        if not (1 <= f <= n_frames):
            issues.append(f'line {ln}: frame {f}')
        if i <= 0:
            issues.append(f'line {ln}: id {i}')
        if w <= 0 or h <= 0:
            issues.append(f'line {ln}: box {w}x{h}')
        if any(v != v or abs(v) == float('inf') for v in (x, y, w, h)):
            issues.append(f'line {ln}: non-finite')
        if p[7] != '1':
            issues.append(f'line {ln}: class {p[7]}')
        if p[8] != '1':
            issues.append(f'line {ln}: visibility {p[8]}')
        per_frame[f].append(i)
    for f, ids in per_frame.items():
        if len(ids) != len(set(ids)):
            issues.append(f'frame {f}: duplicate tracker id')
    lens = Counter()
    for ids in per_frame.values():
        for i in ids:
            lens[i] += 1
    tl = sorted(lens.values())
    return issues, {
        'rows': sum(len(v) for v in per_frame.values()),
        'distinct_ids': len(lens),
        'median_track_length': tl[len(tl) // 2] if tl else 0,
        'track_length_min': tl[0] if tl else 0,
        'track_length_max': tl[-1] if tl else 0,
        'tracks_len_1': sum(1 for x in tl if x == 1),
        'tracks_len_ge_100': sum(1 for x in tl if x >= 100),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--iso', required=True, help='FRESH isolated env for T2')
    ap.add_argument('--out', default='experiments/tracking_v2/t2')
    args = ap.parse_args()

    got = hashlib.sha256((EXP / 'T2_modern_default_policy_spec.json'
                          ).read_bytes()).hexdigest()
    if got != SPEC_SHA:
        raise SystemExit(f'REFUSING: T2 spec hash {got} != frozen {SPEC_SHA}')

    iso = Path(args.iso)
    py, runner = iso / 'venv' / 'Scripts' / 'python.exe', iso / 't2_runner.py'
    out = REPO / args.out
    gt = REPO / 'data' / 'tracking_val_gt'
    v1 = REPO / 'data' / 'tracking_val_v1'
    man = json.loads((gt / 'manifest.json').read_text(encoding='utf-8'))
    seqs = sorted(man['sequences'], key=lambda s: s['sequence'])

    report = {'spec_sha256': got, 'profile': PROFILE, 'config_equivalence': {},
              'validation': {}, 'diagnostics': {}, 'meta': {}}

    for tracker in CANDIDATES:
        for s in seqs:
            seq = s['sequence']
            dst = out / 'outputs' / SPLIT / tracker / 'data' / f'{seq}.txt'
            meta = out / 'meta' / tracker / f'{seq}.json'
            dst.parent.mkdir(parents=True, exist_ok=True)
            meta.parent.mkdir(parents=True, exist_ok=True)
            cmd = [str(py), str(runner), '--tracker', tracker,
                   '--profile', PROFILE,
                   '--candidates', str(v1 / 'candidates' / f'{seq}.jsonl'),
                   '--frames', str(gt / 'sequences' / seq / 'img1'),
                   '--out', str(dst), '--meta-out', str(meta),
                   '--fps', str(s['native_fps']), '--min-conf', str(MIN_CONF)]
            r = subprocess.run(cmd, cwd=str(iso), capture_output=True, text=True)
            if r.returncode != 0:
                raise SystemExit(f'{tracker} {seq} failed:\n{r.stderr[-1500:]}')
            m = json.loads(meta.read_text(encoding='utf-8'))
            old_p = BAKE / 'meta' / PROFILE / tracker / f'{seq}.json'
            old = json.loads(old_p.read_text(encoding='utf-8'))
            same, diffs = config_equivalent(m['effective_config'],
                                            old['effective_config'])
            report['config_equivalence'][f'{tracker}/{seq}'] = {
                'equivalent': same, 'diffs': diffs,
                'compared_against': str(old_p.relative_to(REPO))}
            issues, stats = validate(dst, s['frame_count'])
            report['validation'][f'{tracker}/{seq}'] = {
                'valid': not issues, 'issues': issues[:10]}
            report['diagnostics'][f'{tracker}/{seq}'] = stats
            report['meta'][f'{tracker}/{seq}'] = m
            print(f'  {tracker:<18}{seq:<32}{stats["rows"]:>6} rows  '
                  f'{stats["distinct_ids"]:>3} ids  '
                  f'config {"MATCH" if same else "DIFFERS"}  '
                  f'{"OK" if not issues else "INVALID"}')

    from tools.bakeoff_legacy_runner import run as run_legacy
    for s in seqs:
        seq = s['sequence']
        dst = out / 'outputs' / SPLIT / LEGACY / 'data' / f'{seq}.txt'
        meta = out / 'meta' / LEGACY / f'{seq}.json'
        run_legacy(v1 / 'detections' / f'{seq}.jsonl', dst, meta, 0.0)
        issues, stats = validate(dst, s['frame_count'])
        report['validation'][f'{LEGACY}/{seq}'] = {'valid': not issues,
                                                   'issues': issues[:10]}
        report['diagnostics'][f'{LEGACY}/{seq}'] = stats
        report['meta'][f'{LEGACY}/{seq}'] = json.loads(meta.read_text(encoding='utf-8'))

    out.mkdir(parents=True, exist_ok=True)
    (out / 'run_report.json').write_text(json.dumps(report, indent=1),
                                         encoding='utf-8')
    bad_cfg = [k for k, v in report['config_equivalence'].items() if not v['equivalent']]
    bad_out = [k for k, v in report['validation'].items() if not v['valid']]
    print(f'\nconfig mismatches: {len(bad_cfg)}   invalid outputs: {len(bad_out)}')
    for k in bad_cfg:
        print(f'   {k}: {report["config_equivalence"][k]["diffs"]}')


if __name__ == '__main__':
    main()
