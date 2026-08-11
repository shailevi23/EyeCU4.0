#!/usr/bin/env python
"""
Controlled tracker-side runtime, on the frozen detections.

Design, because the earlier +37% figure in this project was contaminated by
running all OFF windows before all ON windows on a machine that slowed down:

    warmup      one discarded pass per tracker per sequence
    repeats     3 measured passes
    order       ROTATED between repeats, so a machine drifting over the run
                cannot be mistaken for a tracker being slow
    isolation   detector time is excluded entirely -- detections are read from
                the frozen store, and frame decode is excluded from the timer
    statistic   median of repeats, to blunt a single scheduling hiccup

Only the tracker update call is timed.
"""

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

MODERN = ['ByteTrackTracker', 'BoTSORTTracker', 'CBIoUTracker', 'OCSORTTracker']
LEGACY = 'LEGACY_SUPERVISION_BYTETRACK'
PROFILE = 'EYECU_SCORE_POLICY_V1'
MIN_CONF = 0.10
REPEATS = 3


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--iso', required=True)
    ap.add_argument('--out', default='experiments/tracking_v2/bakeoff/runtime.json')
    args = ap.parse_args()

    iso = Path(args.iso)
    py = iso / 'venv' / 'Scripts' / 'python.exe'
    runner = iso / 'bakeoff_runner.py'
    gt = REPO / 'data' / 'tracking_val_gt'
    v1 = REPO / 'data' / 'tracking_val_v1'
    man = json.loads((gt / 'manifest.json').read_text(encoding='utf-8'))
    seqs = sorted(man['sequences'], key=lambda s: s['sequence'])
    tmp = iso / 'runtime_tmp'
    tmp.mkdir(exist_ok=True)

    from tools.bakeoff_legacy_runner import run as run_legacy
    samples = {t: {s['sequence']: [] for s in seqs} for t in MODERN + [LEGACY]}

    def one(tracker, s):
        seq = s['sequence']
        if tracker == LEGACY:
            m = run_legacy(v1 / 'detections' / f'{seq}.jsonl',
                           tmp / 'out.txt', tmp / 'meta.json', 0.0)
            return m['tracker_ms_per_frame']
        cmd = [str(py), str(runner), '--tracker', tracker, '--profile', PROFILE,
               '--candidates', str(v1 / 'candidates' / f'{seq}.jsonl'),
               '--frames', str(gt / 'sequences' / seq / 'img1'),
               '--out', str(tmp / 'out.txt'), '--meta-out', str(tmp / 'meta.json'),
               '--fps', str(s['native_fps']), '--min-conf', str(MIN_CONF)]
        r = subprocess.run(cmd, cwd=str(iso), capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit(r.stderr[-800:])
        return json.loads((tmp / 'meta.json').read_text(encoding='utf-8'))[
            'tracker_ms_per_frame']

    order = MODERN + [LEGACY]
    print('warmup (discarded)')
    for t in order:
        for s in seqs:
            one(t, s)
    for rep in range(REPEATS):
        rot = order[rep % len(order):] + order[:rep % len(order)]
        print(f'repeat {rep + 1}  order {[x[:12] for x in rot]}')
        for t in rot:
            for s in seqs:
                samples[t][s['sequence']].append(one(t, s))

    legacy_total = statistics.median(
        [statistics.median(samples[LEGACY][s['sequence']]) for s in seqs])
    out = {'profile': PROFILE, 'repeats': REPEATS, 'warmup': 1,
           'order': 'rotated between repeats',
           'detector_excluded': True,
           'frame_decode_excluded_from_timer': True,
           'statistic': 'median of repeats, then median across sequences',
           'trackers': {}}
    print(f'\n{"tracker":<32}' + ''.join(f'{s["sequence"][:12]:>14}' for s in seqs)
          + f'{"median":>10}{"vs legacy":>11}')
    for t in order:
        per = {s['sequence']: round(statistics.median(samples[t][s['sequence']]), 3)
               for s in seqs}
        med = statistics.median(per.values())
        pct = 100 * (med - legacy_total) / legacy_total
        out['trackers'][t] = {
            'per_sequence_ms_per_frame': per,
            'all_samples': {k: [round(x, 3) for x in v] for k, v in samples[t].items()},
            'median_ms_per_frame': round(med, 3),
            'tracker_ms_per_frame_pct_vs_legacy': round(pct, 1),
            'total_ms_per_frame_pct_vs_legacy': None,
        }
        print(f'  {t[:30]:<32}' + ''.join(f'{per[s["sequence"]]:>14.3f}' for s in seqs)
              + f'{med:>10.3f}{pct:>10.1f}%')
    out['end_to_end_note'] = (
        'Full-pipeline detector+tracker runtime was NOT measured. Criterion 8 '
        'applies to shortlisted production candidates, and no candidate reached '
        'the shortlist: every modern tracker failed the accuracy criteria on '
        'the primary profile. Measuring end-to-end runtime for a candidate that '
        'cannot be adopted would not change any decision. The tracker-side '
        'figures above are reported as evidence, not as criterion 8.')
    Path(REPO / args.out).write_text(json.dumps(out, indent=1), encoding='utf-8')
    print(f'\nwritten {args.out}')


if __name__ == '__main__':
    main()
