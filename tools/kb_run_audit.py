#!/usr/bin/env python
"""
Per-broadcast-run value and risk audit, and the RARE_WIDE_ANGLE question.

Every run is measured separately and on the same terms, because the decision
that matters is per run: a run can be large and still be a liability, and a run
can be unusual and still be the only source of something EyeCU needs.

Value here means ball supply, especially small balls on the audit's stored-pixel
convention, plus human scale and view diversity. Risk means unresolved roles,
missed officials, and annotation noise. Both are reported for every run; neither
is collapsed into a single score, because that would hide the trade the decision
actually turns on.

Camera type is inferred from measured geometry -- how many humans are in frame
and how tall they are relative to the frame -- not from an impression of the
footage.
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
PKG = REPO / 'experiments' / 'external_sources' / 'keremberke_review'
SRC = REPO / 'EyeCU_external_data/huggingface/keremberke_football_object_detection'


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ledger = json.loads((PKG / 'ledger.json').read_text(encoding='utf-8'))
    by_id = {r['BOX_ID']: r for r in ledger}
    last = {}
    for line in (PKG / 'decisions.json').read_text(encoding='utf-8').splitlines():
        if line.strip():
            d = json.loads(line)
            last[(d.get('mode', 'candidates'), d['BOX_ID'])] = d['HUMAN_FINAL_CLASS']
    cand = {b: v for (m, b), v in last.items() if m == 'candidates'}
    qa = {b: v for (m, b), v in last.items() if m == 'qa_player'}
    noc = {b: v for (m, b), v in last.items() if m == 'qa_nocand'}
    mrq = json.loads((PKG / 'missed_role_queue.json').read_text(encoding='utf-8'))

    ball = json.loads((SRC / 'manifests' / 'ball_instances.json').read_text(encoding='utf-8'))
    ball_by_file = defaultdict(list)
    for b in ball:
        ball_by_file[b['file']].append(b)

    # Ball rows carry no run: the triage only ran on human boxes. Attribute every
    # row through its IMAGE, or ball boxes land in a phantom 'other' run, images
    # get counted twice, and the per-run ball figures that drive the keep/exclude
    # decision come out wrong.
    img_run = {}
    for r in ledger:
        if r['run']:
            img_run[r['IMAGE']] = r['run']
    for r in ledger:
        if not r['run']:
            r['run'] = img_run.get(r['IMAGE'], 'other')

    runs = defaultdict(lambda: {'images': set(), 'human_boxes': 0, 'heights': [],
                                'per_image_humans': Counter(),
                                'P': 0, 'G': 0, 'R': 0, 'U': 0,
                                'qa_n': 0, 'qa_missed': 0,
                                'noc_imgs': set(), 'noc_bad': set(),
                                'ball': [], 'mr_queue': 0})
    for r in ledger:
        run = r['run'] or 'other'
        d = runs[run]
        d['images'].add(r['IMAGE'])
        if r['eyecu_original_class'] == 'player':
            d['human_boxes'] += 1
            d['heights'].append(r['bbox_xywh'][3])
            d['per_image_humans'][r['IMAGE']] += 1
        v = cand.get(r['BOX_ID'])
        if v:
            d[{'player': 'P', 'goalkeeper': 'G', 'referee': 'R',
               'uncertain': 'U'}[v]] += 1
        if r['BOX_ID'] in qa:
            d['qa_n'] += 1
            if qa[r['BOX_ID']] in ('goalkeeper', 'referee'):
                d['qa_missed'] += 1
        if r['BOX_ID'] in noc:
            d['noc_imgs'].add(r['IMAGE'])
            if noc[r['BOX_ID']] in ('goalkeeper', 'referee'):
                d['noc_bad'].add(r['IMAGE'])
    for r in ledger:
        if r['eyecu_original_class'] == 'ball':
            runs[r['run'] or 'other']['ball'] += ball_by_file.get(r['file'], [])[:1]
    for z in mrq['rows']:
        runs[z['run']]['mr_queue'] += 1

    # ball rows are keyed by file, so attribute them via the image's run
    run_of_img = {r['IMAGE']: (r['run'] or 'other') for r in ledger}
    # One entry per BALL INSTANCE, attributed by image. Taking the first ball per
    # file dropped every extra ball in a multi-ball image and lost 49 of the 474
    # balls at <=8 px -- the exact number the keep/exclude decision turns on.
    bl = defaultdict(list)
    seen_files = set()
    for r in ledger:
        if r['eyecu_original_class'] == 'ball' and r['file'] not in seen_files:
            seen_files.add(r['file'])
            for b in ball_by_file.get(r['file'], []):
                bl[run_of_img[r['IMAGE']]].append(b['w'])
    total_imgs = len({r['IMAGE'] for r in ledger})

    out = {}
    print(f'{"run":<9}{"imgs":>6}{"%ds":>6}{"humans":>8}{"balls":>7}'
          f'{"<=5":>5}{"<=8":>5}{"<=12":>6}{"U%":>6}{"miss%":>7}'
          f'{"h/img":>7}{"medH":>6}  domain')
    for run in sorted(runs):
        d = runs[run]
        imgs = len(d['images'])
        heights = np.array(d['heights']) if d['heights'] else np.array([0.0])
        hpi = (np.mean(list(d['per_image_humans'].values()))
               if d['per_image_humans'] else 0)
        w = np.array(bl.get(run, [])) if bl.get(run) else np.array([])
        reviewed = d['P'] + d['G'] + d['R'] + d['U']
        u_rate = d['U'] / reviewed if reviewed else 0
        miss_rate = d['qa_missed'] / d['qa_n'] if d['qa_n'] else None
        # camera type from measured geometry
        medh = float(np.median(heights))
        frac = medh / 720.0
        domain = ('HIGH_WIDE_TACTICAL' if hpi >= 16 and frac < 0.075
                  else 'STANDARD_BROADCAST' if frac >= 0.075 or hpi < 16
                  else 'OTHER')
        out[run] = {
            'images': imgs, 'pct_of_dataset': round(100 * imgs / total_imgs, 1),
            'human_boxes': d['human_boxes'],
            'balls': int(len(w)),
            'ball_le5': int((w <= 5).sum()) if len(w) else 0,
            'ball_le8': int((w <= 8).sum()) if len(w) else 0,
            'ball_le12': int((w <= 12).sum()) if len(w) else 0,
            'ball_median_px': round(float(np.median(w)), 2) if len(w) else None,
            'decisions': {'P': d['P'], 'G': d['G'], 'R': d['R'], 'U': d['U']},
            'U_rate': round(u_rate, 4),
            'qa_player_sampled': d['qa_n'], 'qa_player_missed_officials': d['qa_missed'],
            'measured_missed_role_rate': (round(miss_rate, 4)
                                          if miss_rate is not None else None),
            'qa_nocand_images': len(d['noc_imgs']),
            'qa_nocand_images_with_missed_official': len(d['noc_bad']),
            'retrospective_queue_boxes': d['mr_queue'],
            'humans_per_image': round(float(hpi), 1),
            'median_human_height_px': round(medh, 1),
            'median_human_height_frac_of_frame': round(frac, 4),
            'domain_type': domain,
        }
        print(f'{run:<9}{imgs:>6}{out[run]["pct_of_dataset"]:>6.1f}'
              f'{d["human_boxes"]:>8}{len(w):>7}'
              f'{out[run]["ball_le5"]:>5}{out[run]["ball_le8"]:>5}'
              f'{out[run]["ball_le12"]:>6}{100*u_rate:>6.1f}'
              f'{(100*miss_rate if miss_rate is not None else float("nan")):>7.1f}'
              f'{hpi:>7.1f}{medh:>6.0f}  {domain}')

    rare = [r for r, v in out.items() if v['domain_type'] == 'HIGH_WIDE_TACTICAL']
    tot_ball = sum(v['balls'] for v in out.values())
    rep = {'runs': out,
           'dataset_totals': {
               'images': total_imgs,
               'balls': tot_ball,
               'ball_le5': sum(v['ball_le5'] for v in out.values()),
               'ball_le8': sum(v['ball_le8'] for v in out.values()),
               'ball_le12': sum(v['ball_le12'] for v in out.values())},
           'rare_wide_angle_runs': rare,
           'camera_type_rule': ('HIGH_WIDE_TACTICAL when a frame carries >=16 '
                                'annotated humans AND the median human is under '
                                '7.5% of frame height; measured, not eyeballed')}
    if rare:
        rr = {'runs': rare,
              'images': sum(out[r]['images'] for r in rare),
              'pct_of_dataset': round(sum(out[r]['pct_of_dataset'] for r in rare), 1),
              'balls': sum(out[r]['balls'] for r in rare),
              'ball_le5': sum(out[r]['ball_le5'] for r in rare),
              'ball_le8': sum(out[r]['ball_le8'] for r in rare),
              'ball_le12': sum(out[r]['ball_le12'] for r in rare),
              'share_of_all_balls_pct': round(
                  100 * sum(out[r]['balls'] for r in rare) / max(tot_ball, 1), 1),
              'share_of_le8_balls_pct': round(
                  100 * sum(out[r]['ball_le8'] for r in rare)
                  / max(sum(v['ball_le8'] for v in out.values()), 1), 1),
              'U_boxes': sum(out[r]['decisions']['U'] for r in rare),
              'share_of_all_U_pct': round(
                  100 * sum(out[r]['decisions']['U'] for r in rare)
                  / max(sum(v['decisions']['U'] for v in out.values()), 1), 1),
              'retrospective_queue_boxes': sum(out[r]['retrospective_queue_boxes']
                                               for r in rare)}
        rep['rare_wide_angle'] = rr
        print(f'\nRARE_WIDE_ANGLE runs {rare}: {rr["images"]} images '
              f'({rr["pct_of_dataset"]}%), {rr["balls"]} balls '
              f'({rr["share_of_all_balls_pct"]}% of all), '
              f'{rr["ball_le8"]} at <=8px ({rr["share_of_le8_balls_pct"]}% of all), '
              f'{rr["U_boxes"]} U ({rr["share_of_all_U_pct"]}% of all)')
    (PKG / 'RUN_AUDIT.json').write_text(json.dumps(rep, indent=1), encoding='utf-8')
    print('\nwrote RUN_AUDIT.json')


if __name__ == '__main__':
    main()
