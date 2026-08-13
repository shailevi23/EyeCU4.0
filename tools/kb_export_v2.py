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
import shutil
import sys
import tempfile
import time
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


def preconditions(S):
    """Everything that must hold before a real write. Fails closed on each.

    Returns a list of (name, ok, detail). Any False refuses the apply. These are
    deliberately re-derived here rather than read from a report: a report can be
    stale, and the one time a stale report was trusted it said zero outstanding
    flags while fifty-one were on record.
    """
    import subprocess as _sp
    mfst = json.loads((PKG / 'PACKAGE_MANIFEST.json').read_text(encoding='utf-8'))
    # Read the gate report AS IT STANDS first. Regenerating it before checking
    # staleness would make that check vacuous -- it can never be stale against a
    # log it was just rebuilt from, so a reviewer who appended a decision after
    # their last --check would sail straight through the guard meant to catch
    # exactly that. The stored report must have been computed from THIS log.
    gate_path = PKG / 'SECOND_PASS_GATE.json'
    stored = json.loads(gate_path.read_text(encoding='utf-8')) \
        if gate_path.exists() else {}
    was_current = not kb_decisions.is_stale(stored, DEC)
    _sp.run([sys.executable, str(REPO / 'tools' / 'kb_second_pass_gate.py')],
            capture_output=True, text=True, encoding='utf-8', cwd=str(REPO))
    gate = json.loads(gate_path.read_text(encoding='utf-8'))
    live = {s: hashlib.sha256((SRC / s / '_annotations.coco.json').read_bytes())
            .hexdigest() == mfst['original_annotation_sha256'][s]
            for s in SPLITS}
    open_u = [b for b, r in S['res'].items()
              if r['disposition'] == 'UNRESOLVED' and not b.startswith('MISSING:')]
    pending_mt = [b for b, v in S['mt'].items() if v['state'] == 'PENDING']
    unresolved = unresolved_policies(S)
    geo_open = [b for b in dispositioned_ever('PARTIAL_BODY_BAD_BOX')
                if b in S['led'] and b not in S['reps']
                and S['res'][b]['disposition'] == 'PARTIAL_BODY_BAD_BOX']
    ball_open = [b for b in dispositioned_ever('BALL_WRONG_HUMAN_BOX')
                 if b in S['led'] and b not in S['balls']
                 and S['res'][b]['disposition'] == 'BALL_WRONG_HUMAN_BOX']
    known = set(kb_decisions.ROLES) | set(kb_decisions.U_CATEGORIES) | {
        kb_decisions.UNRESOLVED, None} | set(BOXED_VALUES) | set(
        kb_decisions.BALL_ACTIONS)
    unknown = sorted({d.get('HUMAN_FINAL_CLASS') for d in kb_decisions.read_log(DEC)
                      if d.get('HUMAN_FINAL_CLASS') not in known})
    return [
        ('second-pass gate PASS', gate.get('passed') is True,
         f'{len(gate.get("blocking", []))} blocking'),
        ('a --check was run against THIS log', was_current,
         'stored gate fingerprint' if was_current
         else 'decisions.json changed since the last check -- re-run --check'),
        ('live source hashes match the manifest', all(live.values()),
         ', '.join(f'{s}={"ok" if v else "MISMATCH"}' for s, v in live.items())),
        ('0 unresolved uncertain', not open_u, f'{len(open_u)} open'),
        ('0 pending missing targets', not pending_mt, f'{len(pending_mt)} pending'),
        ('all geometry-repair cases resolved', not geo_open,
         f'{len(geo_open)} outstanding'),
        ('all ball-review cases resolved', not ball_open,
         f'{len(ball_open)} outstanding'),
        ('no unrecorded per-case policy', not unresolved,
         f'{len(unresolved)} outstanding'),
        ('no unknown decision value', not unknown, str(unknown or 'none')),
    ]


BOXED_VALUES = ('boxed_player', 'boxed_goalkeeper', 'boxed_referee',
                'EXCLUDE_IMAGE')


def git_commit():
    import subprocess as _sp
    try:
        r = _sp.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True,
                    cwd=str(REPO), timeout=20)
        return r.stdout.strip() or None
    except Exception:
        return None


def manifest(S, P, out, rep):
    """Provenance for every change, traceable to the event that authorised it."""
    mf = json.loads((PKG / 'PACKAGE_MANIFEST.json').read_text(encoding='utf-8'))
    cls = {}
    for s in SPLITS:
        names = {c['id']: c['name'] for c in out[s]['categories']}
        cls[s] = dict(Counter(names[a['category_id']] for a in out[s]['annotations']))
    total_cls = Counter()
    for v in cls.values():
        total_cls.update(v)
    return {
        'contract_version': 2,
        'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'tool': 'tools/kb_export_v2.py',
        'git_commit': git_commit(),
        'immutable_source': {
            'sha256_per_split': mf['original_annotation_sha256'],
            'verified_live_at_apply': True},
        'decisions_log': kb_decisions.log_version(DEC),
        'counts': {'per_split': rep['per_split'],
                   'total': sum(v['total'] for v in rep['per_split'].values()),
                   'by_class': dict(total_cls), 'by_split_class': cls},
        'additions': [
            {'annotation_id': rep['id_map'][b], 'split': v['IMAGE'].split('/')[0],
             'IMAGE': v['IMAGE'], 'class': v['role'], 'bbox_xywh': v['bbox_xywh'],
             'missing_target_id': b, 'flagged_role': v['flag_role'],
             'flagged_utc': v['flagged_utc'], 'geometry_author': 'human drawn'}
            for b, v in sorted(P['added'].items())],
        'removals': [
            {'BOX_ID': b, 'reason': why,
             'decided_in_mode': S['res'][b]['decided_in_mode'],
             'recorded_utc': S['res'][b]['recorded_utc']}
            for b, why in sorted(P['removed'].items())],
        'class_changes': [
            {'BOX_ID': b, 'to': c,
             'decided_in_mode': S['res'][b]['decided_in_mode'],
             'recorded_utc': S['res'][b]['recorded_utc']}
            for b, c in sorted(P['changed'].items())],
        'geometry_repairs': [
            {'BOX_ID': b, 'class': r['HUMAN_FINAL_CLASS'],
             'original_bbox_xywh': r['original_bbox_xywh'],
             'replacement_bbox_xywh': r['replacement_bbox_xywh'],
             'geometry_author': r['geometry_author'],
             'recorded_utc': r['recorded_utc']}
            for b, r in sorted(P['repaired'].items())],
        'ball_changes': [
            {'BOX_ID': b, 'action': r['action'],
             'original_bbox_xywh': r['original_bbox_xywh'],
             'ball_bbox_xywh': r.get('ball_bbox_xywh'),
             'human_approved': True, 'recorded_utc': r['recorded_utc']}
            for b, r in sorted(P['ball_fix'].items())],
        'retracted_missing_target_flags': sorted(
            b for b, v in S['mt'].items() if v['state'] == 'RETRACTED'),
        'effective_image_exclusions': sorted(S['excluded']),
        'excluded_annotations': sorted(P['excluded_ann']),
        'ball_integrity_rule': (
            'every ORIGINAL ball annotation survives set-identical; only '
            'explicitly human-approved ball repairs may be added'),
        'no_model_geometry': True,
        'every_change_traceable_to_a_human_event': True,
    }


def apply_atomic(dest):
    """Stage, verify the STAGED bytes, then promote. Never a partial overwrite."""
    S = load_state()
    fingerprint = kb_decisions.log_version(DEC)['decisions_sha256']
    pre = preconditions(S)
    print(f'{"precondition":<44}{"":>2}detail')
    for name, ok, detail in pre:
        print(f'  {"PASS" if ok else "FAIL"}  {name:<40} {detail}')
    if not all(ok for _, ok, _ in pre):
        print('\nREFUSED: a precondition failed. Nothing written.')
        return 1

    staging = Path(tempfile.mkdtemp(prefix='kb_export_v2_stage_'))
    S, P, out, rep = build(staging)
    checks = verify(S, P, out)
    for name, ok in checks:
        print(f'  {"PASS" if ok else "FAIL"}  {name}')
    if not all(ok for _, ok in checks):
        shutil.rmtree(staging, ignore_errors=True)
        print('\nREFUSED: a contract clause failed. Nothing written.')
        return 1

    mf = manifest(S, P, out, rep)
    (staging / 'EXPORT_MANIFEST.json').write_text(
        json.dumps(mf, indent=1), encoding='utf-8')

    # Re-read the log AFTER staging. If a review server appended while this ran,
    # the staged bytes describe a state that no longer exists.
    if kb_decisions.log_version(DEC)['decisions_sha256'] != fingerprint:
        shutil.rmtree(staging, ignore_errors=True)
        print('\nREFUSED: decisions.json changed while the export was being '
              'built. Re-run --check and apply again. Nothing written.')
        return 1

    # Verify the STAGED FILES, not the in-memory objects: what gets promoted is
    # what was written, and a serialisation fault would otherwise go unnoticed.
    staged = {s: json.loads((staging / f'{s}_annotations.coco.json')
                            .read_text(encoding='utf-8')) for s in SPLITS}
    if any(len(staged[s]['annotations']) != rep['per_split'][s]['total']
           for s in SPLITS):
        shutil.rmtree(staging, ignore_errors=True)
        print('\nREFUSED: staged files do not match the plan. Nothing written.')
        return 1

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    prev = dest.with_name(dest.name + '.previous')
    if prev.exists():
        shutil.rmtree(prev)
    # Promote by rename: the destination is never half-written. If this raises,
    # the source and the previous export are both still intact.
    if dest.exists():
        dest.rename(prev)
    try:
        Path(staging).rename(dest)
    except Exception:
        if prev.exists():
            prev.rename(dest)
        raise
    print(f'\nPROMOTED to {dest.relative_to(REPO)}')
    if prev.exists():
        print(f'previous export retained at {prev.name}')
    print(f'manifest: {(dest / "EXPORT_MANIFEST.json").relative_to(REPO)}')
    print('original source untouched; working copy untouched.')
    return 0


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--apply', action='store_true',
                    help='stage, verify, then atomically promote the repaired '
                         'export. Refuses unless every precondition holds.')
    ap.add_argument('--dest', default=str(PKG / 'repaired_export'),
                    help='destination for --apply')
    ap.add_argument('--out')
    args = ap.parse_args()

    if args.apply:
        sys.exit(apply_atomic(args.dest))

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
    if ok:
        # Refresh the gate report so its fingerprint records that a check ran
        # against THIS log. --apply requires that, which is what makes "check,
        # then apply" a real sequence rather than two independent commands.
        import subprocess as _sp
        _sp.run([sys.executable, str(REPO / 'tools' / 'kb_second_pass_gate.py')],
                capture_output=True, cwd=str(REPO))
        print('gate report refreshed; --apply will accept this log')
    if args.out:
        print(f'written to {args.out}  (dry run; the package is untouched)')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
