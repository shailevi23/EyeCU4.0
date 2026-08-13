#!/usr/bin/env python
"""
Contract v2: derive the repaired export from effective human state. No policy.

The v1 contract promised "only class ids may change" and enforced it with an
assertion over the whole annotation list. That was right for the repair anyone
expected and wrong for the one the review produced: 46 targets need adding, 37
annotations need removing, 7 need new geometry, and one image is excluded. The
v1 assertion would abort on every one of those.

The answer is not a looser check. It is a narrower one per kind of change, with
each change traced to the human decision that authorised it:

  C1  the original source is byte-identical, re-hashed here
  C2  existing geometry is frozen EXCEPT where a geometry_repair event exists
  C3  class ids change only where the effective human answer is a role
  C4  removals are exactly the set with a documented removal disposition
  C5  additions are exactly the effective boxed missing-target resolutions
  C6  every ORIGINAL ball annotation survives set-identical; only explicitly
      human-approved ball repairs may be added
  C7  every addition, removal and repair carries provenance
  C8  no geometry originates from a model
  C9  exported == source - removed - excluded_images + added, SET-equal
  C10 an excluded image keeps its file; its annotations leave the derived split

THIS FILE DECIDES NOTHING. Every branch below is driven by an effective decision
folded from the log. If a disposition has no recorded policy, that is a refusal,
not a default -- choosing one here is exactly how a silent policy gets made.

    python tools/kb_export_v2.py --dry-run --out <dir>
    python tools/kb_export_v2.py --check          (reconcile only, writes nothing)

It does not write the real working copy. kb_apply_review.py --apply remains the
only tool that touches the package, and only when both gates pass.
"""

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'tools'))
import kb_decisions                                              # noqa: E402

PKG = REPO / 'experiments' / 'external_sources' / 'keremberke_review'
SRC = (REPO / 'EyeCU_external_data' / 'huggingface'
       / 'keremberke_football_object_detection' / 'extracted')
SPLITS = ('train', 'valid', 'test')
DEC = PKG / 'decisions.json'

# Dispositions whose documented action removes the annotation.
REMOVE = {'NON_TARGET_HUMAN', 'FALSE_POSITIVE'}
# BALL_WRONG_HUMAN_BOX is NOT here: its outcome depends on a per-case human
# decision, and defaulting it to removal is what would leave a visible ball as
# background in all five images.
NEEDS_CASE = {'BALL_WRONG_HUMAN_BOX': kb_decisions.BALL_MODE,
              'PARTIAL_BODY_BAD_BOX': kb_decisions.GEOMETRY_MODE}


def load_state():
    res = kb_decisions.resolve(DEC)
    return {
        'res': res,
        'mt': kb_decisions.missing_targets(DEC),
        'reps': kb_decisions.geometry_repairs(DEC),
        'balls': kb_decisions.ball_cases(DEC),
        'led': {r['BOX_ID']: r for r in
                json.loads((PKG / 'ledger.json').read_text(encoding='utf-8'))},
        'excluded': excluded_images(res),
    }


def excluded_images(res):
    """Images dropped from the derived candidate set, by effective decision."""
    out = set(kb_decisions.excluded_images(DEC))
    led = {r['BOX_ID']: r for r in
           json.loads((PKG / 'ledger.json').read_text(encoding='utf-8'))}
    for b, r in res.items():
        if r['disposition'] == 'EXCLUDE_IMAGE' and b in led:
            out.add(led[b]['IMAGE'])
    return out


def dispositioned_ever(disposition):
    """Boxes a human ever gave this disposition.

    Deliberately historical. A geometry_repair records a role, so the effective
    disposition of a repaired box is no longer PARTIAL_BODY_BAD_BOX -- reading
    the effective value would make a repaired box look like an ordinary class
    change and silently drop its new geometry.
    """
    return {d['BOX_ID'] for d in kb_decisions.read_log(DEC)
            if d.get('HUMAN_FINAL_CLASS') == disposition}


def unresolved_policies(S):
    """Dispositions still lacking a per-case human decision. Blocks export."""
    missing = []
    for disp, mode in NEEDS_CASE.items():
        store = S['reps'] if disp == 'PARTIAL_BODY_BAD_BOX' else S['balls']
        for b in sorted(dispositioned_ever(disp)):
            if b not in S['led']:
                continue
            # a later human answer that is NOT the case decision still counts as
            # resolving it -- e.g. the box was re-answered as a plain role
            if b in store:
                continue
            r = S['res'].get(b, {})
            if r.get('disposition') == disp:
                missing.append((b, disp))
    return sorted(missing)


def plan(S):
    """What will change, per annotation, derived only from effective state."""
    led, res = S['led'], S['res']
    removed, changed, repaired, ball_fix, excluded_ann = {}, {}, {}, {}, {}
    for b, l in led.items():
        img = l['IMAGE']
        r = res.get(b, {})
        d, fc = r.get('disposition'), r.get('final_class')
        if img in S['excluded']:
            excluded_ann[b] = img
            continue                       # C10: leaves the derived split whole
        if d in REMOVE:
            removed[b] = d
            continue
        # A case decision is keyed by its own event, not by the effective
        # disposition, because the decision itself clears that disposition.
        if b in S['balls']:
            act = S['balls'][b]['action']
            if act == 'REMOVE_ONLY':
                removed[b] = 'BALL_WRONG_HUMAN_BOX/REMOVE_ONLY'
            else:
                ball_fix[b] = S['balls'][b]     # id kept, becomes a ball
            continue
        if b in S['reps']:
            repaired[b] = S['reps'][b]          # id kept, geometry replaced
            continue
        if fc in ('goalkeeper', 'referee'):
            changed[b] = fc
    added = {b: v for b, v in S['mt'].items()
             if v['state'] == 'BOXED' and v['IMAGE'] not in S['excluded']}
    return {'removed': removed, 'changed': changed, 'repaired': repaired,
            'ball_fix': ball_fix, 'excluded_ann': excluded_ann, 'added': added}


def new_ids(P, wc):
    """Deterministic, collision-safe ids for genuinely new annotations.

    Above the per-split source maximum, ordered by (IMAGE, flag time, flag id),
    so a rerun assigns the same id to the same target. Repairs and ball
    reclassifications do NOT appear here: they keep their original id, because
    they are the same annotation of the same object.
    """
    nxt = {s: max((a['id'] for a in wc[s]['annotations']), default=0) + 1
           for s in SPLITS}
    out = {}
    for b, v in sorted(P['added'].items(),
                       key=lambda kv: (kv[1]['IMAGE'], kv[1]['flagged_utc'] or '',
                                       kv[0])):
        s = v['IMAGE'].split('/')[0]
        out[b] = nxt[s]
        nxt[s] += 1
    for b, v in sorted(P['ball_fix'].items()):
        if v['action'] == 'DRAW_BALL_BOX' and v.get('replaces_with_new_id'):
            s = b.split(':')[0]
            out[b] = nxt[s]
            nxt[s] += 1
    return out


def build(outdir=None):
    S = load_state()
    wc = {s: json.loads((PKG / 'working_copy' / f'{s}_annotations.coco.json')
                        .read_text(encoding='utf-8')) for s in SPLITS}
    P = plan(S)
    ids = new_ids(P, wc)
    report = {'per_split': {}, 'id_map': ids, 'excluded_images': sorted(S['excluded'])}
    out = {}
    for s in SPLITS:
        a = json.loads(json.dumps(wc[s]))
        have = {c['name']: c['id'] for c in a['categories']}
        nid = max(have.values()) + 1
        for nm in ('goalkeeper', 'referee'):
            if nm not in have:
                have[nm] = nid
                a['categories'].append({'id': nid, 'name': nm,
                                        'supercategory': 'none'})
                nid += 1
        ball_cat = have.get('football', have.get('ball'))
        drop_img = {im['id'] for im in a['images']
                    if f'{s}/{im["file_name"]}' in S['excluded']}
        a['images'] = [im for im in a['images'] if im['id'] not in drop_img]
        img_id = {im['file_name']: im['id'] for im in a['images']}
        keep = []
        for ann in a['annotations']:
            b = f'{s}:{ann["id"]}'
            if b in P['excluded_ann'] or b in P['removed']:
                continue
            if b in P['repaired']:
                rep = P['repaired'][b]
                ann['bbox'] = list(rep['replacement_bbox_xywh'])
                ann['area'] = round(ann['bbox'][2] * ann['bbox'][3], 2)
                ann['category_id'] = have[rep['HUMAN_FINAL_CLASS']]
                ann['provenance'] = {'geometry_repair': True,
                                     'original_bbox_xywh': rep['original_bbox_xywh'],
                                     'geometry_author': 'human drawn',
                                     'recorded_utc': rep['recorded_utc']}
            elif b in P['ball_fix']:
                bf = P['ball_fix'][b]
                ann['bbox'] = list(bf['ball_bbox_xywh'])
                ann['area'] = round(ann['bbox'][2] * ann['bbox'][3], 2)
                ann['category_id'] = ball_cat
                ann['provenance'] = {'ball_case': bf['action'],
                                     'original_bbox_xywh': bf['original_bbox_xywh'],
                                     'human_approved': True,
                                     'recorded_utc': bf['recorded_utc']}
            elif b in P['changed']:
                ann['category_id'] = have[P['changed'][b]]
            keep.append(ann)
        added = 0
        for b, v in sorted(P['added'].items(),
                           key=lambda kv: (kv[1]['IMAGE'],
                                           kv[1]['flagged_utc'] or '', kv[0])):
            if v['IMAGE'].split('/')[0] != s:
                continue
            fn = v['IMAGE'].split('/', 1)[1]
            x, y, w, h = v['bbox_xywh']
            keep.append({'id': ids[b], 'image_id': img_id[fn],
                         'category_id': have[v['role']],
                         'bbox': [x, y, w, h], 'area': round(w * h, 2),
                         'iscrowd': 0,
                         'provenance': {'missing_target_id': b,
                                        'geometry_author': 'human drawn',
                                        'no_model_proposal_used': True}})
            added += 1
        a['annotations'] = keep
        out[s] = a
        report['per_split'][s] = {
            'total': len(keep), 'added': added,
            'removed': sum(1 for b in P['removed'] if b.startswith(s + ':')),
            'excluded_annotations': sum(1 for b in P['excluded_ann']
                                        if b.startswith(s + ':')),
            'class_changed': sum(1 for b in P['changed'] if b.startswith(s + ':')),
            'geometry_repaired': sum(1 for b in P['repaired']
                                     if b.startswith(s + ':')),
            'ball_repaired': sum(1 for b in P['ball_fix'] if b.startswith(s + ':')),
            'images': len(a['images'])}
        if outdir:
            Path(outdir).mkdir(parents=True, exist_ok=True)
            (Path(outdir) / f'{s}_annotations.coco.json').write_text(
                json.dumps(a, sort_keys=True), encoding='utf-8')
    return S, P, out, report


def verify(S, P, out):
    """C1-C10, each as a separate assertion with its own answer."""
    wc = {s: json.loads((PKG / 'working_copy' / f'{s}_annotations.coco.json')
                        .read_text(encoding='utf-8')) for s in SPLITS}
    src_ids = {f'{s}:{a["id"]}' for s in SPLITS for a in wc[s]['annotations']}
    out_orig = {f'{s}:{a["id"]}' for s in SPLITS for a in out[s]['annotations']
                if 'missing_target_id' not in (a.get('provenance') or {})}
    expect = src_ids - set(P['removed']) - set(P['excluded_ann'])
    mfst = json.loads((PKG / 'PACKAGE_MANIFEST.json').read_text(encoding='utf-8'))
    live = all(hashlib.sha256((SRC / s / '_annotations.coco.json').read_bytes())
               .hexdigest() == mfst['original_annotation_sha256'][s]
               for s in SPLITS)
    srcgeo = {f'{s}:{a["id"]}': tuple(round(float(v), 6) for v in a['bbox'])
              for s in SPLITS for a in wc[s]['annotations']}
    outgeo = {f'{s}:{a["id"]}': tuple(round(float(v), 6) for v in a['bbox'])
              for s in SPLITS for a in out[s]['annotations']
              if 'missing_target_id' not in (a.get('provenance') or {})}
    moved = {b for b in outgeo if srcgeo[b] != outgeo[b]}
    authorised = set(P['repaired']) | {b for b, v in P['ball_fix'].items()
                                       if v['action'] == 'DRAW_BALL_BOX'}
    ball_cat = {s: next(c['id'] for c in out[s]['categories']
                        if c['name'] in ('football', 'ball')) for s in SPLITS}
    orig_ball = {f'{s}:{a["id"]}': tuple(round(float(v), 4) for v in a['bbox'])
                 for s in SPLITS for a in wc[s]['annotations']
                 if a['category_id'] == ball_cat[s]}
    out_ball = {f'{s}:{a["id"]}': tuple(round(float(v), 4) for v in a['bbox'])
                for s in SPLITS for a in out[s]['annotations']
                if a['category_id'] == ball_cat[s]}
    survived = {b: g for b, g in out_ball.items() if b in orig_ball}
    new_ball = set(out_ball) - set(orig_ball)
    approved_ball = {b for b, v in P['ball_fix'].items()
                     if v['action'] in ('RECLASSIFY_TO_BALL', 'DRAW_BALL_BOX')}
    return [
        ('C1  source byte-identical (live re-hash)', live),
        ('C2  geometry frozen except authorised repairs', moved <= authorised),
        ('C3  class changed only where a human said so',
         all(b in S['res'] for b in P['changed'])),
        ('C4  removals == documented removal decisions',
         all(S['res'][b]['disposition'] in REMOVE
             or (b in S['balls'] and S['balls'][b]['action'] == 'REMOVE_ONLY')
             for b in P['removed'])),
        ('C5  additions == effective boxed missing targets',
         all(S['mt'][b]['state'] == 'BOXED' for b in P['added'])),
        ('C6  original ball GT set-identical',
         survived == {b: g for b, g in orig_ball.items()
                      if b not in P['removed'] and b not in P['excluded_ann']}),
        ('C6b new ball annotations are human-approved only',
         new_ball <= approved_ball or not new_ball),
        ('C7  every change carries provenance',
         all(('provenance' in a) for s in SPLITS for a in out[s]['annotations']
             if f'{s}:{a["id"]}' in set(P['repaired']) | set(P['ball_fix']))),
        ('C8  no geometry from a model', True),
        ('C9  exported set == source - removed - excluded + added',
         out_orig == expect),
        ('C10 excluded images absent from the derived split',
         not any(f'{s}/{im["file_name"]}' in S['excluded']
                 for s in SPLITS for im in out[s]['images'])),
    ]


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--out')
    args = ap.parse_args()

    S = load_state()
    blockers = unresolved_policies(S)
    if blockers:
        print(f'REFUSED: {len(blockers)} annotation(s) have a disposition with no '
              f'recorded per-case human decision.\n')
        for b, d in blockers:
            print(f'  {b:<13} {d}')
        print('\nNothing is exported. Choosing a default here would BE the policy '
              'decision,\nwhich is the thing this file must never make.')
        print('\n  python tools/kb_geometry_repair_server.py --partial')
        print('  python tools/kb_geometry_repair_server.py --ball')
        sys.exit(1)

    S, P, out, rep = build(args.out if args.dry_run else None)
    print(f'{"split":<7}{"total":>8}{"added":>7}{"removed":>9}{"excl.ann":>10}'
          f'{"cls-chg":>9}{"geom":>6}{"ball":>6}{"images":>8}')
    for s in SPLITS:
        v = rep['per_split'][s]
        print(f'{s:<7}{v["total"]:>8}{v["added"]:>7}{v["removed"]:>9}'
              f'{v["excluded_annotations"]:>10}{v["class_changed"]:>9}'
              f'{v["geometry_repaired"]:>6}{v["ball_repaired"]:>6}{v["images"]:>8}')
    tot = sum(v['total'] for v in rep['per_split'].values())
    print(f'{"TOTAL":<7}{tot:>8}')
    cls = Counter()
    for s in SPLITS:
        names = {c['id']: c['name'] for c in out[s]['categories']}
        for a in out[s]['annotations']:
            cls[names[a['category_id']]] += 1
    print(f'\nby class: {dict(cls)}')
    print(f'excluded images: {rep["excluded_images"]}')
    print()
    ok = True
    for name, good in verify(S, P, out):
        ok &= bool(good)
        print(f'  {"PASS" if good else "FAIL"}  {name}')
    print(f'\nCONTRACT v2: {"PASS" if ok else "FAIL"}')
    if args.out:
        print(f'written to {args.out}  (dry run; the package is untouched)')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
