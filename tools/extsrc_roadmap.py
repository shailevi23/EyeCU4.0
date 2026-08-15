#!/usr/bin/env python
"""
Build the research priority matrix and the Experiment D decision gate.

The matrix is deliberately NOT collapsed into one ranking. A source that is
useless for the ball can be the best thing available for calibration, and a
single "score" would hide exactly that. Five separate rankings are produced,
one per research priority, and each is justified by the numbers in the registry
and the audits rather than by an impression.

Experiment D is a GATE, not a plan. It answers whether the data now in hand
justifies running it, using the rule the previous audits established: original
unique images matter, augmentation is not diversity, same-match frames are
correlated, and a ball that only looks small because the export was downscaled
is not tiny-ball evidence.
"""

import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EXT = REPO / 'EyeCU_external_data'
XS = REPO / 'experiments' / 'external_sources'

RATE = ['HIGH', 'MEDIUM', 'LOW', 'NONE', 'UNKNOWN']


def load(p, d=None):
    p = Path(p)
    return json.loads(p.read_text(encoding='utf-8')) if p.exists() else d


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    reg = load(EXT / 'MASTER_EXTERNAL_SOURCE_REGISTRY.json')
    kb = load(EXT / 'huggingface/keremberke_football_object_detection/manifests/audit.json', {})
    sn_gate = load(EXT / 'huggingface/soccernet_v3/manifests/metadata_gate.json', {})
    sn_dims = load(EXT / 'huggingface/soccernet_v3/manifests/image_dimensions.json', {})
    sn_leak = load(XS / 'reports/soccernet_leakage.json', {})
    sn_qual = load(XS / 'reports/soccernet_label_quality.json', {})
    rb_sum = load(REPO / 'experiments/external_data_audit/reports/AUDIT_SUMMARY.json', {})

    # ---- priority matrix ----------------------------------------------------
    M = {
        'RB_S1': dict(ball='LOW', humans='LOW', tracking='NONE', calibration='NONE',
                      events='NONE', identity='NONE', domain='MEDIUM',
                      quality='MEDIUM', diversity='LOW', effort='MEDIUM'),
        'RB_S2': dict(ball='NONE', humans='NONE', tracking='NONE', calibration='NONE',
                      events='NONE', identity='NONE', domain='LOW',
                      quality='LOW', diversity='LOW', effort='HIGH'),
        'RB_S3': dict(ball='LOW', humans='LOW', tracking='NONE', calibration='NONE',
                      events='NONE', identity='NONE', domain='HIGH',
                      quality='MEDIUM', diversity='MEDIUM', effort='MEDIUM'),
        'RB_S4': dict(ball='NONE', humans='NONE', tracking='NONE', calibration='NONE',
                      events='NONE', identity='NONE', domain='HIGH',
                      quality='MEDIUM', diversity='NONE', effort='NONE'),
        'RB_S5': dict(ball='LOW', humans='NONE', tracking='NONE', calibration='NONE',
                      events='NONE', identity='NONE', domain='MEDIUM',
                      quality='MEDIUM', diversity='MEDIUM', effort='HIGH'),
        'RB_S6': dict(ball='MEDIUM', humans='MEDIUM', tracking='NONE', calibration='NONE',
                      events='NONE', identity='NONE', domain='HIGH',
                      quality='HIGH', diversity='LOW', effort='LOW'),
        'HF_KEREMBERKE': dict(ball='HIGH', humans='LOW', tracking='NONE',
                              calibration='NONE', events='NONE', identity='NONE',
                              domain='HIGH', quality='HIGH', diversity='MEDIUM',
                              effort='MEDIUM'),
        # quality raised from UNKNOWN to HIGH only after the payload arrived and
        # the boxes were looked at: 24/24 sampled ball boxes on the ball, and
        # goalkeeper/referee correctly separated in the rendered frames
        'HF_SOCCERNET_V3': dict(ball='MEDIUM', humans='HIGH', tracking='LOW',
                                calibration='NONE', events='MEDIUM', identity='HIGH',
                                domain='HIGH', quality='HIGH', diversity='HIGH',
                                effort='MEDIUM'),
        'SOCCERTRACK_V2': dict(ball='NONE', humans='LOW', tracking='MEDIUM',
                               calibration='HIGH', events='MEDIUM', identity='MEDIUM',
                               domain='LOW', quality='MEDIUM', diversity='LOW',
                               effort='HIGH'),
        'SOCCERTRACK_MOT': dict(ball='NONE', humans='LOW', tracking='LOW',
                                calibration='NONE', events='NONE', identity='NONE',
                                domain='LOW', quality='UNKNOWN', diversity='UNKNOWN',
                                effort='HIGH'),
        'HF_MARTINJOLIF': dict(ball='UNKNOWN', humans='UNKNOWN', tracking='NONE',
                               calibration='NONE', events='NONE', identity='NONE',
                               domain='UNKNOWN', quality='UNKNOWN', diversity='NONE',
                               effort='LOW'),
        'HF_SOCCANA': dict(ball='UNKNOWN', humans='UNKNOWN', tracking='NONE',
                           calibration='NONE', events='NONE', identity='NONE',
                           domain='UNKNOWN', quality='UNKNOWN', diversity='UNKNOWN',
                           effort='HIGH'),
        'RS_TEAMTRACK': dict(ball='UNKNOWN', humans='UNKNOWN', tracking='MEDIUM',
                             calibration='UNKNOWN', events='NONE', identity='UNKNOWN',
                             domain='LOW', quality='UNKNOWN', diversity='UNKNOWN',
                             effort='HIGH'),
        'RS_SPORTSLABKIT': dict(ball='NONE', humans='NONE', tracking='MEDIUM',
                                calibration='MEDIUM', events='LOW', identity='LOW',
                                domain='NONE', quality='UNKNOWN', diversity='NONE',
                                effort='HIGH'),
    }
    for k, v in M.items():
        for r in v.values():
            assert r in RATE, (k, r)

    rankings = {
        '1_CURRENT_BALL_IMPROVEMENT': [
            ('HF_KEREMBERKE', 'The only source whose export applied NO resize, so its '
                              'stored pixels are original pixels. 1,263 ball boxes, '
                              'median 9.1 px at native 1280x720 -- SMALLER than '
                              "EyeCU's own ball median. 474 balls <=8 px as stored."),
            ('HF_SOCCERNET_V3', '3,830 ball boxes across 55 matches and 6 leagues, '
                                'real broadcast. Balls are LARGER than EyeCU\'s own '
                                '(35 px vs 18 px at 1920-equivalent), so this is ball '
                                'DIVERSITY, not tiny-ball supply.'),
            ('RB_S6', '1,251 keep-candidate images with the best ball boxes of the six, '
                      'but one La Liga match only.'),
            ('RB_S5', 'Ball-only with unlabelled humans; large balls; reference at best.'),
            ('RB_S3', '141 padded ball boxes across a few Premier League matches.'),
            ('RB_S1', '498 ball boxes, one match, boxes distorted by a 640x640 stretch.'),
        ],
        '2_FUTURE_CALIBRATION': [
            ('SOCCERTRACK_V2', 'Complete and verified: homography, fisheye intrinsics '
                               'and extrinsics, undistortion maps, 65 pitch '
                               'correspondences, for all 10 matches. Two independent '
                               'routes agree to 9.8 px; 0.83 m median positional '
                               'accuracy. Nothing else audited comes close.'),
            ('RS_SPORTSLABKIT', 'Calibration and pitch-coordinate methodology, as '
                                'reading material only -- GPL-3.0.'),
            ('HF_SOCCERNET_V3', 'No calibration in this export.'),
        ],
        '3_FUTURE_EVENTS': [
            ('SOCCERTRACK_V2', '23,663 BAS events, 12 classes, ms timestamps, player_id '
                               'on 99.5%, alignable to video.'),
            ('HF_SOCCERNET_V3', 'Action-centred sampling with replay groups; the export '
                                'is organised around events even though it ships no '
                                'event labels of its own.'),
        ],
        '4_FUTURE_IDENTITY_JERSEY': [
            ('HF_SOCCERNET_V3', '15,041 readable jersey numbers on detections, plus '
                                'team-left/team-right role split and 57 teams.'),
            ('SOCCERTRACK_V2', 'Full squad metadata, jersey numbers, persistent '
                               'track_id -> player_id, but only 10 amateur matches and '
                               'unusable boxes.'),
        ],
        '5_FUTURE_TRACKING_RESEARCH': [
            ('SOCCERTRACK_V2', 'Persistent identities, though with no occlusion gaps, '
                               'so it does not exercise what a tracker must solve.'),
            ('RS_TEAMTRACK', 'Designed as an external MOT benchmark; access currently '
                             'BLOCKED_ACCESS (401).'),
            ('RS_SPORTSLABKIT', 'Tracking pipeline ideas, GPL-3.0, reference only.'),
        ],
    }

    # ---- Experiment D gate --------------------------------------------------
    kball = (kb.get('ball') or {})
    sn_ball = (sn_dims.get('ball_box_pixels_1920x1080_equivalent') or {})
    gate = {
        'question': 'SHOULD WE CREATE EXPERIMENT D?',
        'answer': 'YES',
        'previous_answer_on_the_six_roboflow_sources_alone': 'WEAK',
        'what_changed': [
            'keremberke supplies 1,263 ball boxes at NATIVE resolution with no resize '
            'in the export chain -- the first source in any EyeCU audit where a small '
            'stored ball is a genuinely small ball. Its balls are smaller than '
            "EyeCU's own: 13.7 px vs 18 px at 1920-equivalent.",
            'SoccerNet-V3 supplies 40,139 player, 2,948 goalkeeper and 3,402 referee '
            'boxes with goalkeeper and referee as DISTINCT classes, across 55 matches '
            'in 6 leagues -- the class distinction EyeCU needs and the diversity the '
            'six Roboflow sources could not provide.',
            'Neither source overlaps EyeCU VAL or TEST.',
        ],
        'what_has_NOT_changed': [
            'Still no large supply of genuinely tiny broadcast balls. Across everything '
            'audited, balls <=8 px in original terms remain rare.',
            'A large correlated block from one match is still not diversity: RB_S6 '
            'remains 1,251 frames of one La Liga match.',
        ],
        'blocking_conditions_before_D_can_be_designed': [
            'keremberke collapses goalkeeper AND referee into "player" (verified '
            'visually on yellow-kit officials). Mixing its human boxes as-is would '
            'damage the two classes EyeCU is weakest on. Either use it ball-only with '
            'its humans handled as ignore-regions, or relabel -- and neither is '
            'authorised yet.',
            'SoccerNet-V3 label quality has not been visually verified on the '
            'downloaded payload yet, and its Staff members / Wall of players / Referee '
            'flag / card classes must be dropped rather than mapped.',
            'SoccerNet-V3 ships an action frame plus up to 8 replay frames of the SAME '
            'action; replays are correlated by construction and must not be counted as '
            'independent images.',
        ],
        'design_constraints_for_a_future_D': {
            'keep_fixed': ['YOLO26s', '960 input', 'same training policy',
                           'same frozen VAL', 'same evaluation protocol'],
            'change_only': 'audited external real data added to TRAIN',
            'never': ['external data in VAL or TEST',
                      'goalkeeper or referee collapsed into player',
                      'unlabelled humans mixed in without ignore-region handling'],
        },
        'not_executed': True,
    }

    report = {
        'generated_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'rating_scale': RATE,
        'priority_matrix': M,
        'rankings': rankings,
        'experiment_d_gate': gate,
    }
    XS.mkdir(parents=True, exist_ok=True)
    (XS / 'reports').mkdir(parents=True, exist_ok=True)
    (XS / 'reports' / 'priority_matrix.json').write_text(
        json.dumps(report, indent=1, ensure_ascii=False), encoding='utf-8')

    cols = ['ball', 'humans', 'tracking', 'calibration', 'events', 'identity',
            'domain', 'quality', 'diversity', 'effort']
    print(f'{"SOURCE":<18}' + ''.join(f'{c[:9]:<11}' for c in cols))
    for k, v in M.items():
        print(f'{k:<18}' + ''.join(f'{v[c]:<11}' for c in cols))
    print(f'\nEXPERIMENT D: {gate["answer"]} (was {gate["previous_answer_on_the_six_roboflow_sources_alone"]})')
    print(f'wrote {(XS / "reports" / "priority_matrix.json").relative_to(REPO)}')


if __name__ == '__main__':
    main()
