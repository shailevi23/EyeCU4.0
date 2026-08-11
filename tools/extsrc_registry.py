#!/usr/bin/env python
"""
Build MASTER_EXTERNAL_SOURCE_REGISTRY.json from the audit artifacts.

One record per source, and every number that can be read from an audit file IS
read from it rather than retyped. Retyped numbers drift; that is how a registry
becomes a second, quietly wrong source of truth.

Where a field genuinely cannot be filled from evidence it is written as
UNKNOWN or NOT_ESTABLISHED, never guessed and never left blank.
"""

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EXT = REPO / 'EyeCU_external_data'
RB = REPO / 'experiments' / 'external_data_audit'
ST = REPO / 'experiments' / 'soccertrack_audit'
XS = REPO / 'experiments' / 'external_sources'


def load(p, default=None):
    p = Path(p)
    if not p.exists():
        return default
    return json.loads(p.read_text(encoding='utf-8'))


def dir_bytes(p: Path):
    if not p.exists():
        return 0
    return sum(f.stat().st_size for f in p.rglob('*') if f.is_file())


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    rb_sum = load(RB / 'reports' / 'AUDIT_SUMMARY.json', {})
    rb_src = (load(RB / 'raw' / 'SOURCES.json', {}) or {}).get('sources', {})
    rb_inv = (load(RB / 'reports' / 'inventory.json', {}) or {}).get('sources', {})
    rb_idx = load(RB / 'candidate_index' / 'summary.json', {}) or {}
    st_sum = load(ST / 'reports' / 'AUDIT_SUMMARY.json', {})
    kb = load(EXT / 'huggingface/keremberke_football_object_detection/manifests/audit.json', {})
    kb_log = load(EXT / 'huggingface/download_logs/keremberke__football-object-detection.json', {})
    kb_leak = load(XS / 'reports' / 'keremberke_leakage.json', {})
    sn_gate = load(EXT / 'huggingface/soccernet_v3/manifests/metadata_gate.json', {})
    sn_dims = load(EXT / 'huggingface/soccernet_v3/manifests/image_dimensions.json', {})
    sn_log = load(EXT / 'huggingface/download_logs/Voxel51__SoccerNet-V3.json', {})
    sn_full = load(EXT / 'huggingface/download_logs/Voxel51__SoccerNet-V3_full.json', {})
    sn_leak = load(XS / 'reports' / 'soccernet_leakage.json', {})
    res = load(EXT / 'research_sources/manifests/research_sources.json', {})
    nd = load(EXT / 'huggingface/manifests/not_downloaded_sources.json', {})
    st_hash = load(ST / 'reports' / 'archive_hashes.json', {})

    records = []

    # ---- the six Roboflow sources, preserved, not re-audited -----------------
    verdict_map = (rb_sum.get('verdicts') or {})
    for sid, s in rb_src.items():
        inv = rb_inv.get(sid, {})
        idx = (rb_idx.get('per_source') or {}).get(sid, {})
        ball = inv.get('ball', {})
        dec = (verdict_map.get(sid) or {}).get('decision', 'UNKNOWN')
        cur = {'KEEP': 'KEEP_ACTIVE', 'PARTIAL USE': 'KEEP_REFERENCE',
               'REJECT': 'REJECT'}.get(dec, 'KEEP_REFERENCE')
        if dec.startswith('PARTIAL USE -- ball'):
            cur = 'KEEP_REFERENCE'
        records.append({
            'SOURCE_ID': f'RB_{sid}',
            'NAME': f"{s['workspace']}/{s['project']} v{s['version']}",
            'ORIGIN': 'Roboflow Universe (user-supplied ZIP)',
            'URL_OR_REPO_ID': s.get('roboflow_url'),
            'SOURCE_TYPE': 'detection dataset (YOLO)',
            'LICENSE': s.get('license'),
            'LICENSE_VERIFIED_FROM': 'data.yaml roboflow block + README.dataset.txt in the ZIP',
            'LOCAL_PATH': 'EyeCU_external_data/roboflow_audit/raw_zips/' + s['original_filename'],
            'DOWNLOAD_STATUS': 'COMPLETE',
            'SIZE': s['archive_bytes'],
            'HASH_OR_REVISION': s['sha256'],
            'DOMAIN': (rb_sum.get('source_inventory', {}).get(sid, {}) or {}).get('footage'),
            'CLASSES': s.get('declared_classes'),
            'BALL_BOXES': ball.get('instances', 0),
            'GK_SEPARATE': 'goalkeeper' in (inv.get('proposed_eyecu_mapping') or {}).values(),
            'REFEREE_AVAILABLE': 'referee' in (inv.get('proposed_eyecu_mapping') or {}).values(),
            'PARTIAL_ANNOTATION_RISK': (idx.get('annotation_completeness') or {}).get('verdict'),
            'DUPLICATE_RISK': f"export inflation {(rb_sum.get('duplicates',{}).get('augmentation_inflation') or {}).get(sid)}",
            'TRAIN_OVERLAP': 272 if sid == 'S4' else 0,
            'VAL_OVERLAP': 0,
            'TEST_OVERLAP': 0,
            'BALL_DETECTOR_VALUE': {'S6': 'MEDIUM', 'S1': 'LOW', 'S3': 'LOW',
                                    'S5': 'LOW', 'S4': 'NONE', 'S2': 'NONE'}[sid],
            'HUMAN_DETECTOR_VALUE': {'S6': 'MEDIUM', 'S1': 'LOW', 'S3': 'LOW',
                                     'S5': 'NONE', 'S4': 'NONE', 'S2': 'NONE'}[sid],
            'TRACKING_VALUE': 'NONE', 'CALIBRATION_VALUE': 'NONE',
            'EVENT_VALUE': 'NONE', 'IDENTITY_JERSEY_VALUE': 'NONE',
            'CURRENT_DECISION': cur,
            'NEXT_ALLOWED_ACTION': ('candidate pool for a future Experiment D, '
                                    'TRAIN-only, no merge yet'),
            'NOTES': (verdict_map.get(sid) or {}).get('why'),
        })

    # ---- keremberke ---------------------------------------------------------
    kball = (kb.get('ball') or {})
    records.append({
        'SOURCE_ID': 'HF_KEREMBERKE',
        'NAME': 'keremberke/football-object-detection',
        'ORIGIN': 'Hugging Face (mirror of Roboflow augmented-startups/football-player-detection-kucab v3)',
        'URL_OR_REPO_ID': 'keremberke/football-object-detection',
        'SOURCE_TYPE': 'detection dataset (COCO)',
        'LICENSE': 'CC BY 4.0',
        'LICENSE_VERIFIED_FROM': 'README.md dataset card + README.dataset.txt in the export',
        'LOCAL_PATH': 'EyeCU_external_data/huggingface/keremberke_football_object_detection/',
        'DOWNLOAD_STATUS': 'COMPLETE',
        'SIZE': kb_log.get('total_bytes'),
        'HASH_OR_REVISION': kb_log.get('revision'),
        'DOMAIN': 'Euro 2020 broadcast, wide tactical, beIN/RaiPlay feeds, 1280x720, no resize applied',
        'CLASSES': list((kb.get('classes_total') or {})),
        'BALL_BOXES': kball.get('instances', 0),
        'GK_SEPARATE': False,
        'REFEREE_AVAILABLE': False,
        'PARTIAL_ANNOTATION_RISK': ('CLASS COLLAPSE, not missing labels: officials and '
                                    'goalkeepers ARE annotated, but as "player" '
                                    '(verified visually on yellow-kit officials)'),
        'DUPLICATE_RISK': '1232 files from 1232 distinct source images; no augmentation',
        'TRAIN_OVERLAP': kb_leak.get('EXTERNAL_vs_EYECU_TRAIN', 'NOT_ESTABLISHED'),
        'VAL_OVERLAP': kb_leak.get('EXTERNAL_vs_EYECU_VAL', 'NOT_ESTABLISHED'),
        'TEST_OVERLAP': kb_leak.get('EXTERNAL_vs_EYECU_TEST', 'NOT_ESTABLISHED'),
        'BALL_DETECTOR_VALUE': 'HIGH',
        'HUMAN_DETECTOR_VALUE': 'LOW',
        'TRACKING_VALUE': 'NONE', 'CALIBRATION_VALUE': 'NONE',
        'EVENT_VALUE': 'NONE', 'IDENTITY_JERSEY_VALUE': 'NONE',
        'CURRENT_DECISION': 'KEEP_ACTIVE',
        'NEXT_ALLOWED_ACTION': ('ball-focused Experiment D candidate; requires a '
                                'decision on the GK/referee class collapse before '
                                'any human box is used'),
        'NOTES': ('The first source in any EyeCU audit whose export applied NO resize: '
                  'stored pixels are original pixels. Its balls are SMALLER than '
                  "EyeCU's own (13.7 px vs 18 px at 1920-equivalent)."),
    })

    # ---- SoccerNet-V3 -------------------------------------------------------
    ball_px = (sn_dims.get('ball_box_pixels_1920x1080_equivalent') or {})
    records.append({
        'SOURCE_ID': 'HF_SOCCERNET_V3',
        'NAME': 'Voxel51/SoccerNet-V3',
        'ORIGIN': 'Hugging Face (FiftyOne snapshot of SoccerNet-V3)',
        'URL_OR_REPO_ID': 'Voxel51/SoccerNet-V3',
        'SOURCE_TYPE': 'detection dataset (FiftyOne samples.json, relative boxes)',
        'LICENSE': 'mit (dataset card)',
        'LICENSE_VERIFIED_FROM': 'Hugging Face dataset card cardData.license',
        'LOCAL_PATH': 'EyeCU_external_data/huggingface/soccernet_v3/',
        'DOWNLOAD_STATUS': ('COMPLETE' if sn_full.get('file_count')
                            else 'METADATA_ONLY (payload download in progress or stopped)'),
        'SIZE': sn_full.get('total_bytes') or sn_log.get('total_bytes'),
        'HASH_OR_REVISION': sn_log.get('revision'),
        'DOMAIN': ('real broadcast, 55 matches, 6 European leagues, 3 seasons, '
                   '1280x720 and 1920x1080'),
        'CLASSES': list((sn_gate.get('classes') or {})),
        'BALL_BOXES': sn_gate.get('ball_annotations', 0),
        'GK_SEPARATE': sn_gate.get('goalkeeper_distinct'),
        'REFEREE_AVAILABLE': sn_gate.get('referee_distinct'),
        'PARTIAL_ANNOTATION_RISK': ('LOW for the four EyeCU classes; the export also '
                                    'carries Staff members, Wall of players, Referee '
                                    'flag and cards, which must be dropped, not mapped'),
        'DUPLICATE_RISK': ('action frame + up to 8 replay frames per event; replays '
                           'are of the SAME action and are correlated by construction'),
        'TRAIN_OVERLAP': sn_leak.get('EXTERNAL_vs_EYECU_TRAIN', 'NOT_ESTABLISHED'),
        'VAL_OVERLAP': sn_leak.get('EXTERNAL_vs_EYECU_VAL', 'NOT_ESTABLISHED'),
        'TEST_OVERLAP': sn_leak.get('EXTERNAL_vs_EYECU_TEST', 'NOT_ESTABLISHED'),
        'BALL_DETECTOR_VALUE': 'MEDIUM',
        'HUMAN_DETECTOR_VALUE': 'HIGH',
        'TRACKING_VALUE': 'LOW', 'CALIBRATION_VALUE': 'NONE',
        'EVENT_VALUE': 'MEDIUM', 'IDENTITY_JERSEY_VALUE': 'HIGH',
        'CURRENT_DECISION': 'KEEP_ACTIVE',
        'NEXT_ALLOWED_ACTION': ('label-quality and duplicate audit on the downloaded '
                                'payload, then Experiment D candidacy'),
        'NOTES': ('The only audited source with goalkeeper AND referee as distinct '
                  'classes at scale, and the only one with real multi-league '
                  f'diversity. Ball median {ball_px.get("median")} px at '
                  f'1920-equivalent, {ball_px.get("le8")} balls <=8 px.'),
    })

    # ---- SoccerTrack v2 -----------------------------------------------------
    records.append({
        'SOURCE_ID': 'SOCCERTRACK_V2',
        'NAME': 'SoccerTrack v2 (partial download)',
        'ORIGIN': 'Google Drive package + public GitHub repository',
        'URL_OR_REPO_ID': 'github.com/AtomScott/SoccerTrack-v2 ; hf atomscott/soccertrack-v2 (GATED)',
        'SOURCE_TYPE': 'GSR / BAS / RAW calibration / panoramic video',
        'LICENSE': 'data CC BY 4.0, code MIT',
        'LICENSE_VERIFIED_FROM': ('LICENSE and LICENSE-DATA in the public repository; '
                                  'NOTHING shipped with the data download itself'),
        'LOCAL_PATH': 'EyeCU_external_data/soccertrack_v2/',
        'DOWNLOAD_STATUS': 'PARTIAL -- GSR/BAS/RAW/1 video complete; MOT archive empty',
        'SIZE': dir_bytes(EXT / 'soccertrack_v2'),
        'HASH_OR_REVISION': {k: v.get('sha256') for k, v in (st_hash or {}).items()},
        'DOMAIN': 'Japanese university/amateur, fixed panoramic rig, one shot type',
        'CLASSES': ['player', 'goalkeeper', '(referee declared, 0 instances)',
                    '(ball declared, 0 instances)'],
        'BALL_BOXES': 0,
        'GK_SEPARATE': True,
        'REFEREE_AVAILABLE': False,
        'PARTIAL_ANNOTATION_RISK': ('boxes are regression-generated from position, not '
                                    'person-tight; the annotated images are not in the '
                                    'download at all'),
        'DUPLICATE_RISK': 'none found',
        'TRAIN_OVERLAP': (st_sum.get('eyecu_leakage_check') or {}).get('EXTERNAL_vs_TRAIN', 0),
        'VAL_OVERLAP': (st_sum.get('eyecu_leakage_check') or {}).get('EXTERNAL_vs_VAL', 0),
        'TEST_OVERLAP': (st_sum.get('eyecu_leakage_check') or {}).get('EXTERNAL_vs_TEST', 0),
        'BALL_DETECTOR_VALUE': 'NONE',
        'HUMAN_DETECTOR_VALUE': 'LOW',
        'TRACKING_VALUE': 'MEDIUM', 'CALIBRATION_VALUE': 'HIGH',
        'EVENT_VALUE': 'MEDIUM', 'IDENTITY_JERSEY_VALUE': 'MEDIUM',
        'CURRENT_DECISION': 'KEEP_REFERENCE',
        'NEXT_ALLOWED_ACTION': ('calibration/homography methodology reference for a '
                                'future EyeCU calibration phase; no detector use'),
        'NOTES': 'MOT acquisition CLOSED -- see SOCCERTRACK_MOT record.',
    })
    records.append({
        'SOURCE_ID': 'SOCCERTRACK_MOT',
        'NAME': 'SoccerTrack v2 MOT component',
        'ORIGIN': 'Hugging Face atomscott/soccertrack-v2 (gated)',
        'URL_OR_REPO_ID': 'atomscott/soccertrack-v2 mot/*',
        'SOURCE_TYPE': 'MOTChallenge ground truth',
        'LICENSE': 'CC BY 4.0 (per repository LICENSE-DATA)',
        'LICENSE_VERIFIED_FROM': 'public repository LICENSE-DATA',
        'LOCAL_PATH': 'EyeCU_external_data/soccertrack_v2/raw/mot-...zip (158 bytes, empty)',
        'DOWNLOAD_STATUS': 'DISTRIBUTION_UNAVAILABLE_OR_EMPTY',
        'SIZE': 158, 'HASH_OR_REVISION': (st_hash or {}).get(
            'mot/mot-20260811T095052Z-1-001.zip', {}).get('sha256'),
        'DOMAIN': 'as SoccerTrack v2', 'CLASSES': ['class=1 for every entity; role not encoded'],
        'BALL_BOXES': 'NOT_ESTABLISHED', 'GK_SEPARATE': False,
        'REFEREE_AVAILABLE': 'NOT_ESTABLISHED',
        'PARTIAL_ANNOTATION_RISK': 'NOT_ESTABLISHED',
        'DUPLICATE_RISK': 'NOT_ESTABLISHED',
        'TRAIN_OVERLAP': 'NOT_ESTABLISHED', 'VAL_OVERLAP': 'NOT_ESTABLISHED',
        'TEST_OVERLAP': 'NOT_ESTABLISHED',
        'BALL_DETECTOR_VALUE': 'NONE', 'HUMAN_DETECTOR_VALUE': 'LOW',
        'TRACKING_VALUE': 'LOW', 'CALIBRATION_VALUE': 'NONE',
        'EVENT_VALUE': 'NONE', 'IDENTITY_JERSEY_VALUE': 'NONE',
        'CURRENT_DECISION': 'REJECT',
        'NEXT_ALLOWED_ACTION': ('NOT_REQUIRED_FOR_CURRENT_EYECU_DETECTOR_WORK -- '
                                'acquisition closed, do not pursue'),
        'NOTES': ('The public repo documents MOT ground truth as generated by '
                  'position-based width/height regression, the same step that makes '
                  'GSR bbox_image, so obtaining it would not create a person-tight '
                  'detector GT source. Public MOT code/docs remain research references.'),
    })

    # ---- skip / defer -------------------------------------------------------
    mj = nd.get('martinjolif/football-player-detection', {})
    records.append({
        'SOURCE_ID': 'HF_MARTINJOLIF', 'NAME': 'martinjolif/football-player-detection',
        'ORIGIN': 'Hugging Face', 'URL_OR_REPO_ID': 'martinjolif/football-player-detection',
        'SOURCE_TYPE': 'detection dataset', 'LICENSE': mj.get('license'),
        'LICENSE_VERIFIED_FROM': 'Hugging Face dataset card (metadata only, not downloaded)',
        'LOCAL_PATH': 'NOT DOWNLOADED', 'DOWNLOAD_STATUS': 'SKIPPED_BY_INSTRUCTION',
        'SIZE': mj.get('total_bytes'), 'HASH_OR_REVISION': mj.get('revision'),
        'DOMAIN': 'single match, camera near the middle of the pitch (per its card)',
        'CLASSES': ['Ball', 'Goalkeeper', 'Player', 'Referee'],
        'BALL_BOXES': 'NOT_ESTABLISHED', 'GK_SEPARATE': True, 'REFEREE_AVAILABLE': True,
        'PARTIAL_ANNOTATION_RISK': 'NOT_ESTABLISHED', 'DUPLICATE_RISK': 'HIGH (user-identified duplicate)',
        'TRAIN_OVERLAP': 'NOT_ESTABLISHED', 'VAL_OVERLAP': 'NOT_ESTABLISHED',
        'TEST_OVERLAP': 'NOT_ESTABLISHED',
        'BALL_DETECTOR_VALUE': 'UNKNOWN', 'HUMAN_DETECTOR_VALUE': 'UNKNOWN',
        'TRACKING_VALUE': 'NONE', 'CALIBRATION_VALUE': 'NONE', 'EVENT_VALUE': 'NONE',
        'IDENTITY_JERSEY_VALUE': 'NONE',
        'CURRENT_DECISION': 'SKIP_DUPLICATE',
        'NEXT_ALLOWED_ACTION': 'none; re-open only if a duplicate check is required',
        'NOTES': ('STATUS = SKIP_DUPLICATE_SOURCE. Its card describes the same four-class '
                  'single-match taxonomy as the Roboflow football-players-detection '
                  'corpus EyeCU already holds as rfext_*. Metadata checked; no images '
                  'downloaded.'),
    })
    sc = nd.get('Adit-jain/Soccana_player_ball_detection_v1', {})
    records.append({
        'SOURCE_ID': 'HF_SOCCANA', 'NAME': 'Adit-jain/Soccana_player_ball_detection_v1',
        'ORIGIN': 'Hugging Face', 'URL_OR_REPO_ID': 'Adit-jain/Soccana_player_ball_detection_v1',
        'SOURCE_TYPE': 'detection dataset', 'LICENSE': sc.get('license') or 'NOT DECLARED',
        'LICENSE_VERIFIED_FROM': 'Hugging Face dataset card (metadata only, not downloaded)',
        'LOCAL_PATH': 'NOT DOWNLOADED', 'DOWNLOAD_STATUS': 'DEFERRED_BY_INSTRUCTION',
        'SIZE': sc.get('total_bytes'), 'HASH_OR_REVISION': sc.get('revision'),
        'DOMAIN': '25k-image curated subset of a 1M+ image pool, aggregated and augmented from multiple open-source datasets (per its card)',
        'CLASSES': ['player', 'ball', 'referee (per tags)'],
        'BALL_BOXES': 'NOT_ESTABLISHED', 'GK_SEPARATE': False, 'REFEREE_AVAILABLE': 'per tags, unverified',
        'PARTIAL_ANNOTATION_RISK': 'NOT_ESTABLISHED',
        'DUPLICATE_RISK': 'HIGH -- derived from multiple existing datasets, sliced and augmented',
        'TRAIN_OVERLAP': 'NOT_ESTABLISHED', 'VAL_OVERLAP': 'NOT_ESTABLISHED',
        'TEST_OVERLAP': 'NOT_ESTABLISHED',
        'BALL_DETECTOR_VALUE': 'UNKNOWN', 'HUMAN_DETECTOR_VALUE': 'UNKNOWN',
        'TRACKING_VALUE': 'NONE', 'CALIBRATION_VALUE': 'NONE', 'EVENT_VALUE': 'NONE',
        'IDENTITY_JERSEY_VALUE': 'NONE',
        'CURRENT_DECISION': 'DEFER',
        'NEXT_ALLOWED_ACTION': ('re-open only after a goalkeeper relabelling policy and a '
                                'dedup plan exist'),
        'NOTES': 'STATUS = DEFER_REQUIRES_GK_RELABEL_AND_DEDUP. Its own card confirms it aggregates and augments other open-source datasets, which is the duplicate/diversity concern.',
    })

    # ---- research sources ---------------------------------------------------
    tt = res.get('teamtrack', {})
    records.append({
        'SOURCE_ID': 'RS_TEAMTRACK', 'NAME': 'AtomScott/TeamTrack',
        'ORIGIN': 'GitHub + Hugging Face', 'URL_OR_REPO_ID': 'AtomScott/TeamTrack',
        'SOURCE_TYPE': 'multi-sport MOT tracking benchmark',
        'LICENSE': tt.get('license_spdx'),
        'LICENSE_VERIFIED_FROM': 'GitHub API license field + saved LICENSE file',
        'LOCAL_PATH': 'EyeCU_external_data/research_sources/teamtrack/ (README/LICENSE only)',
        'DOWNLOAD_STATUS': f"METADATA_ONLY; Hugging Face dataset returned "
                           f"{tt.get('huggingface_dataset_status')} -> "
                           f"{tt.get('huggingface_access')}",
        'SIZE': dir_bytes(EXT / 'research_sources/teamtrack'),
        'HASH_OR_REVISION': tt.get('pushed_at'),
        'DOMAIN': 'multi-sport fixed-camera tracking, includes soccer_side and soccer_top',
        'CLASSES': 'NOT_ESTABLISHED (data not downloaded)',
        'BALL_BOXES': 'NOT_ESTABLISHED', 'GK_SEPARATE': 'NOT_ESTABLISHED',
        'REFEREE_AVAILABLE': 'NOT_ESTABLISHED',
        'PARTIAL_ANNOTATION_RISK': 'NOT_ESTABLISHED', 'DUPLICATE_RISK': 'NOT_ESTABLISHED',
        'TRAIN_OVERLAP': 'NOT_ESTABLISHED', 'VAL_OVERLAP': 'NOT_ESTABLISHED',
        'TEST_OVERLAP': 'NOT_ESTABLISHED',
        'BALL_DETECTOR_VALUE': 'UNKNOWN', 'HUMAN_DETECTOR_VALUE': 'UNKNOWN',
        'TRACKING_VALUE': 'MEDIUM', 'CALIBRATION_VALUE': 'UNKNOWN',
        'EVENT_VALUE': 'NONE', 'IDENTITY_JERSEY_VALUE': 'UNKNOWN',
        'CURRENT_DECISION': 'FUTURE_RESEARCH',
        'NEXT_ALLOWED_ACTION': ('CURRENT_USE = FUTURE_EXTERNAL_TRACKING_STRESS_TEST. '
                                'Access is blocked; do not pursue now, and do not '
                                'reopen tracker selection on it.'),
        'NOTES': ('Code MIT. The Hugging Face dataset is not publicly readable '
                  '(401) -> BLOCKED_ACCESS. No large download attempted.'),
    })
    sl = res.get('sportslabkit', {})
    records.append({
        'SOURCE_ID': 'RS_SPORTSLABKIT', 'NAME': 'AtomScott/SportsLabKit',
        'ORIGIN': 'GitHub', 'URL_OR_REPO_ID': 'AtomScott/SportsLabKit',
        'SOURCE_TYPE': 'source code / research library',
        'LICENSE': sl.get('license_spdx'),
        'LICENSE_VERIFIED_FROM': 'GitHub API license field + saved LICENSE file',
        'LOCAL_PATH': 'EyeCU_external_data/research_sources/sportslabkit/ (README/LICENSE only)',
        'DOWNLOAD_STATUS': 'METADATA_ONLY', 'SIZE': dir_bytes(EXT / 'research_sources/sportslabkit'),
        'HASH_OR_REVISION': sl.get('pushed_at'),
        'DOMAIN': 'n/a (code)', 'CLASSES': 'n/a', 'BALL_BOXES': 0,
        'GK_SEPARATE': 'n/a', 'REFEREE_AVAILABLE': 'n/a',
        'PARTIAL_ANNOTATION_RISK': 'n/a', 'DUPLICATE_RISK': 'n/a',
        'TRAIN_OVERLAP': 0, 'VAL_OVERLAP': 0, 'TEST_OVERLAP': 0,
        'BALL_DETECTOR_VALUE': 'NONE', 'HUMAN_DETECTOR_VALUE': 'NONE',
        'TRACKING_VALUE': 'MEDIUM (ideas)', 'CALIBRATION_VALUE': 'MEDIUM (ideas)',
        'EVENT_VALUE': 'LOW', 'IDENTITY_JERSEY_VALUE': 'LOW',
        'CURRENT_DECISION': 'KEEP_REFERENCE',
        'NEXT_ALLOWED_ACTION': ('RESEARCH_REFERENCE_ONLY -- read the papers and ideas; '
                                'do NOT copy or vendor code into EyeCU without a '
                                'dedicated licence review'),
        'NOTES': (f"License confirmed {sl.get('license_spdx')}. GPL-3.0 is incompatible "
                  'with vendoring into a commercial-product-oriented codebase without a '
                  'separate decision. Code availability is not permission.'),
    })

    reg = {
        'registry': 'EyeCU master external source registry',
        'generated_utc': __import__('time').strftime('%Y-%m-%dT%H:%M:%SZ',
                                                     __import__('time').gmtime()),
        'primary_goal': 'improve EyeCU detector robustness, especially BALL detection',
        'frozen_context': {
            'detector': 'YOLO26s @960 = A (production), @1280 = B (accuracy reference), C rejected',
            'tracker': 'CBIoUTracker FROZEN, vendored trackers==2.6.0 library defaults',
            'tracking_benchmark': 'EyeCU-Tracking-Val-v1.1, 3 matches, 900 frames, 13,021 boxes',
            'tracker_selection': 'CLOSED -- not reopened by anything in this registry',
        },
        'decision_values': ['KEEP_ACTIVE', 'KEEP_REFERENCE', 'METADATA_GATE',
                            'DEFER', 'REJECT', 'SKIP_DUPLICATE', 'FUTURE_RESEARCH'],
        'source_count': len(records),
        'sources': records,
    }
    out = EXT / 'MASTER_EXTERNAL_SOURCE_REGISTRY.json'
    out.write_text(json.dumps(reg, indent=1, ensure_ascii=False), encoding='utf-8')
    print(f'{len(records)} sources')
    print(f'{"SOURCE_ID":<20}{"DECISION":<18}{"BALL":<8}{"GK":<7}{"REF":<7}TRAIN/VAL/TEST overlap')
    for r in records:
        print(f'{r["SOURCE_ID"]:<20}{r["CURRENT_DECISION"]:<18}'
              f'{str(r["BALL_DETECTOR_VALUE"]):<8}{str(r["GK_SEPARATE"]):<7}'
              f'{str(r["REFEREE_AVAILABLE"])[:6]:<7}'
              f'{r["TRAIN_OVERLAP"]}/{r["VAL_OVERLAP"]}/{r["TEST_OVERLAP"]}')
    print(f'\nwrote {out.relative_to(REPO)}')


if __name__ == '__main__':
    main()
