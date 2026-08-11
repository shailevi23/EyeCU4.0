#!/usr/bin/env python
"""
Evaluate the frozen adoption criteria mechanically, and gather diagnostics.

The criteria come from experiments/tracking_v2/adoption_criteria.json, whose
sha256 is pinned in the evaluation contract and re-checked here before anything
is judged. Each criterion is applied as written. No tie-breaker is added, no
threshold is softened, and a tracker that fails is reported as failing.

Runtime (criterion 8) is filled in from a separate controlled measurement; if
that measurement is absent the criterion is reported as NOT MEASURED rather
than assumed.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
EXP = REPO / 'experiments' / 'tracking_v2'
LEGACY = 'LEGACY_SUPERVISION_BYTETRACK'
PRIMARY = 'EYECU_SCORE_POLICY_V1'


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bakeoff', default='experiments/tracking_v2/bakeoff')
    ap.add_argument('--runtime', default=None)
    args = ap.parse_args()

    contract = json.loads((EXP / 'evaluation_contract.json').read_text(encoding='utf-8'))
    got = hashlib.sha256((EXP / 'adoption_criteria.json').read_bytes()).hexdigest()
    if got != contract['adoption_criteria']['sha256']:
        raise SystemExit('REFUSING: adoption criteria changed since the contract '
                         'was frozen.')
    print(f'adoption criteria sha256 verified: {got[:16]}...  spec '
          f'{contract["adoption_criteria"]["spec_version"]}')

    base = REPO / args.bakeoff
    res = json.loads((base / 'trackeval_raw' / f'{PRIMARY}.json').read_text(encoding='utf-8'))
    runs = json.loads((base / 'run_summary.json').read_text(encoding='utf-8'))
    runtime = json.loads(Path(args.runtime).read_text(encoding='utf-8')) if args.runtime else None

    seqs = sorted(s for s in res[LEGACY] if s != 'COMBINED_SEQ')
    base_comb = res[LEGACY]['COMBINED_SEQ']

    verdict = {'profile': PRIMARY, 'baseline': LEGACY,
               'criteria_sha256': got, 'trackers': {}}

    for t in sorted(res):
        if t == LEGACY:
            continue
        comb = res[t]['COMBINED_SEQ']
        d_hota = comb['HOTA'] - base_comb['HOTA']
        d_assa = comb['AssA'] - base_comb['AssA']
        d_idf1 = comb['IDF1'] - base_comb['IDF1']
        per_seq_d = {s: res[t][s]['HOTA'] - res[LEGACY][s]['HOTA'] for s in seqs}
        improved = [s for s, v in per_seq_d.items() if v > 0]
        worst_regress = min(per_seq_d.values())

        c = {}
        c['1_combined_HOTA_ge_+2.0'] = {
            'value': round(d_hota, 2), 'pass': d_hota >= 2.0}
        c['2_combined_AssA_ge_+3.0'] = {
            'value': round(d_assa, 2), 'pass': d_assa >= 3.0}
        c['3_combined_IDF1_regression_le_0.5'] = {
            'value': round(d_idf1, 2), 'pass': d_idf1 >= -0.5}
        c['4_HOTA_improves_in_ge_2_of_3'] = {
            'value': f'{len(improved)}/3 {sorted(improved)}',
            'pass': len(improved) >= 2}
        c['5_remaining_match_not_worse_than_-2.0_HOTA'] = {
            'value': round(worst_regress, 2), 'pass': worst_regress >= -2.0}
        qc_bad = [k for k, v in runs['validation'].items()
                  if k.startswith(f'{PRIMARY}/{t}/') and not v['valid']]
        c['6_no_catastrophic_identity_failure'] = {
            'value': f'{len(qc_bad)} invalid outputs', 'pass': not qc_bad}
        inv = all(
            runs['validation'][f'{PRIMARY}/{t}/{s}']['valid'] for s in seqs)
        c['7_eyecu_invariants_preserved'] = {
            'value': 'class=1, visibility=1, no ball, no duplicate id/frame'
                     if inv else 'violated', 'pass': inv}
        # Criterion 8 is about END-TO-END runtime. The tracker-side figure is
        # evidence, not a substitute: reporting it as if it were end-to-end
        # would quietly change what the criterion means.
        rt = (runtime or {}).get('trackers', {}).get(t, {})
        end_to_end = rt.get('total_ms_per_frame_pct_vs_legacy')
        if end_to_end is not None:
            c['8_end_to_end_runtime_regression_le_10pct'] = {
                'value': f'{end_to_end:+.1f}%', 'pass': end_to_end <= 10.0}
        else:
            side = rt.get('tracker_ms_per_frame_pct_vs_legacy')
            c['8_end_to_end_runtime_regression_le_10pct'] = {
                'value': ('NOT MEASURED (end-to-end). Tracker-side '
                          f'{side:+.1f}% vs legacy.' if side is not None
                          else 'NOT MEASURED'),
                'pass': None}

        overall = all(v['pass'] for v in c.values() if v['pass'] is not None) \
            and all(v['pass'] is not None for v in c.values())
        verdict['trackers'][t] = {
            'combined': {k: round(comb[k], 2) for k in
                         ('HOTA', 'DetA', 'AssA', 'MOTA', 'IDF1')},
            'deltas_vs_legacy': {'HOTA': round(d_hota, 2), 'AssA': round(d_assa, 2),
                                 'IDF1': round(d_idf1, 2)},
            'per_sequence_HOTA_delta': {s: round(v, 2) for s, v in per_seq_d.items()},
            'criteria': c, 'overall_pass': overall}

    passing = [t for t, v in verdict['trackers'].items() if v['overall_pass']]
    verdict['baseline_combined'] = {k: round(base_comb[k], 2) for k in
                                    ('HOTA', 'DetA', 'AssA', 'MOTA', 'IDF1')}
    verdict['passing_trackers'] = passing
    verdict['recommendation'] = (passing[0] if len(passing) == 1 else
                                 'KEEP LEGACY SUPERVISION BYTETRACK'
                                 if not passing else
                                 f'MULTIPLE PASS: {passing}')
    (base / 'adoption_verdict.json').write_text(json.dumps(verdict, indent=1),
                                                encoding='utf-8')

    print(f'\nbaseline {LEGACY} COMBINED_SEQ: '
          + '  '.join(f'{k} {v}' for k, v in verdict['baseline_combined'].items()))
    for t, v in verdict['trackers'].items():
        print(f'\n{t}   HOTA {v["combined"]["HOTA"]}  '
              f'(delta {v["deltas_vs_legacy"]["HOTA"]:+.2f})')
        for k, r in v['criteria'].items():
            mark = 'PASS' if r['pass'] else ('n/a ' if r['pass'] is None else 'FAIL')
            print(f'   [{mark}] {k:<46}{r["value"]}')
        print(f'   OVERALL: {"PASS" if v["overall_pass"] else "FAIL"}')
    print(f'\nRECOMMENDATION: {verdict["recommendation"]}')


if __name__ == '__main__':
    main()
