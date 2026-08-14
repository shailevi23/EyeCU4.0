"""BALL ONTOLOGY REVISIT: classifying the 128 Round-0 findings by ball kind.

The frozen Round-0 result answers "is any visible football missing". This pass
answers a different question about the same 128 boxes -- was it the ACTIVE match
ball, or a spare on the touchline. The two must not contaminate each other, so
a good half of these tests are about what must NOT change: the frozen result
file, the export, the Round-0 events, and the geometry a human already drew.
"""

import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'tools'))

import kb_ball_ontology_revisit_server as ONTO                    # noqa: E402
import kb_ball_qa_sample as SAMPLE                                # noqa: E402
import kb_ball_qa_server as QA                                    # noqa: E402
import kb_decisions                                               # noqa: E402

PKG = REPO / 'experiments' / 'external_sources' / 'keremberke_review'
EXPORT = PKG / 'repaired_export'
RESULT = PKG / 'BALL_QA_ROUND0_RESULT.json'

pytestmark = pytest.mark.skipif(
    not RESULT.is_file(),
    reason='frozen Round-0 result not present in this checkout')


def _frozen():
    return json.loads(RESULT.read_text(encoding='utf-8'))


# ------------------------------------------------------------------- queue


def test_queue_holds_exactly_the_frozen_object_count():
    objs = ONTO.round0_objects()
    assert len(objs) == _frozen()['secondary']['total_missing_objects'] == 128


def test_queue_covers_exactly_the_positive_images():
    objs = ONTO.round0_objects()
    assert len({o['IMAGE'] for o in objs}) == \
           _frozen()['primary']['positive_images'] == 84


def test_queue_ignores_superseded_round0_drafts():
    """Six images were re-answered. Reading the raw log gives 135, not 128.

    The seven extra boxes belong to answers a human replaced; they are not
    findings any more. Putting them in the queue would invent work and inflate
    the denominator of every ontology proportion derived from it.
    """
    raw = [d for d in kb_decisions.read_log(ONTO.DECISIONS)
           if d.get('mode') == QA.QA_MODE]
    raw_objects = sum(len(d.get('missing_balls') or []) for d in raw)
    assert raw_objects > 128, 'this dataset should contain superseded drafts'
    assert len(ONTO.round0_objects()) == 128


def test_every_queued_object_maps_to_a_real_round0_bbox():
    effective = {}
    for im, a in QA.answers().items():
        if a['answer'] == 'MISSING_BALL':
            effective[im] = [tuple(m['bbox_xywh']) for m in a['missing']]
    for o in ONTO.round0_objects():
        assert o['IMAGE'] in effective
        assert tuple(o['bbox_xywh']) in effective[o['IMAGE']]


def test_object_ids_are_unique_and_stable():
    objs = ONTO.round0_objects()
    ids = [o['object_id'] for o in objs]
    assert len(set(ids)) == len(ids)
    assert ids == [o['object_id'] for o in ONTO.round0_objects()]


def test_object_id_is_geometry_based_not_positional():
    """An index would reattach an answer to a different ball if an earlier
    image were ever re-answered. Identity follows what the human pointed at."""
    a = ONTO.object_id('train/x.jpg', [10, 20, 4, 4])
    assert a == ONTO.object_id('train/x.jpg', [10.0, 20.0, 4.0, 4.0])
    assert a != ONTO.object_id('train/x.jpg', [10, 20, 4, 5])
    assert a != ONTO.object_id('train/y.jpg', [10, 20, 4, 4])
    assert a.startswith('BALLOBJ:')


def test_no_new_geometry_is_created():
    """Every queued box is byte-equal to the box Round 0 recorded."""
    st = ONTO.build_state()
    eff = {im: [m['bbox_xywh'] for m in a['missing']]
           for im, a in QA.answers().items() if a['answer'] == 'MISSING_BALL'}
    for it in st['items']:
        assert it['bbox_xywh'] in eff[it['IMAGE']]
    drawn = [it['bbox_xywh'] for it in st['items']]
    assert sorted(map(tuple, drawn)) == \
           sorted(tuple(b) for v in eff.values() for b in v)


# ------------------------------------------------------------------- state


def test_state_shows_full_image_context_and_existing_gt():
    st = ONTO.build_state()
    pop = {r['IMAGE']: r for r in SAMPLE.population()}
    for it in st['items'][:40]:
        assert len(it['ball_gt']) == pop[it['IMAGE']]['ball_gt_count']
        assert it['img_w'] and it['img_h'], 'full image dimensions available'
        assert 'context' in it


def test_siblings_are_the_other_findings_in_the_same_image():
    st = ONTO.build_state()
    by_img = {}
    for it in st['items']:
        by_img.setdefault(it['IMAGE'], []).append(it)
    multi = [v for v in by_img.values() if len(v) > 1]
    assert multi, 'some images hold several findings'
    for group in multi:
        for it in group:
            assert len(it['siblings']) == len(group) - 1
            assert it['object_id'] not in {s['object_id'] for s in it['siblings']}


def test_state_carries_secondary_metadata_only():
    st = ONTO.build_state()
    for it in st['items'][:20]:
        for k in ('split', 'run', 'view_proxy', 'gt_state'):
            assert k in it


def test_state_exposes_the_three_role_names():
    st = ONTO.build_state()
    assert (st['ACTIVE'], st['NON_ACTIVE'], st['UNSURE']) == ONTO.ROLES
    assert ONTO.ROLES == ('ACTIVE_MATCH_BALL', 'NON_ACTIVE_EXTRA_BALL', 'UNSURE')


# ------------------------------------------------------------- fold / restart


def test_ontology_fold_latest_wins_and_keeps_history(tmp_path):
    log = tmp_path / 'd.json'
    oid = 'BALLOBJ:abc123'
    rows = [
        {'mode': ONTO.ONTOLOGY_MODE, 'BOX_ID': oid, 'missing_object_id': oid,
         'IMAGE': 'train/a.jpg', 'HUMAN_BALL_ROLE': ONTO.UNSURE,
         'recorded_utc': '2026-08-14T10:00:00Z'},
        {'mode': ONTO.ONTOLOGY_MODE, 'BOX_ID': oid, 'missing_object_id': oid,
         'IMAGE': 'train/a.jpg', 'HUMAN_BALL_ROLE': ONTO.NON_ACTIVE,
         'recorded_utc': '2026-08-14T11:00:00Z'},
    ]
    log.write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
    o = ONTO.ontology(log)
    assert o[oid]['role'] == ONTO.NON_ACTIVE
    assert [h['role'] for h in o[oid]['history']] == [ONTO.UNSURE,
                                                      ONTO.NON_ACTIVE]
    assert len(kb_decisions.read_log(log)) == 2, 'append-only'


def test_ontology_fold_ignores_other_modes(tmp_path):
    log = tmp_path / 'd.json'
    rows = [
        {'mode': QA.QA_MODE, 'BOX_ID': 'BALLQA0:train/a.jpg',
         'IMAGE': 'train/a.jpg', 'answer': 'MISSING_BALL',
         'missing_balls': [{'bbox_xywh': [1, 1, 4, 4]}],
         'recorded_utc': '2026-08-14T10:00:00Z'},
        {'mode': 'missed_role_manual', 'BOX_ID': 'train:1',
         'HUMAN_FINAL_CLASS': 'referee', 'recorded_utc': '2026-08-14T10:00:00Z'},
    ]
    log.write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
    assert ONTO.ontology(log) == {}


def test_restart_reconstructs_progress_from_the_log(tmp_path):
    """Nothing is held in memory: state is rebuilt from the event log."""
    log = tmp_path / 'd.json'
    src = ONTO.DECISIONS.read_text(encoding='utf-8')
    objs = ONTO.round0_objects()
    extra = [{'mode': ONTO.ONTOLOGY_MODE, 'BOX_ID': o['object_id'],
              'missing_object_id': o['object_id'], 'IMAGE': o['IMAGE'],
              'round0_bbox_xywh': o['bbox_xywh'],
              'HUMAN_BALL_ROLE': ONTO.NON_ACTIVE,
              'recorded_utc': '2026-08-14T12:00:00Z'} for o in objs[:5]]
    log.write_text(src + ''.join(json.dumps(r) + '\n' for r in extra),
                   encoding='utf-8')
    st = ONTO.build_state(decisions=log)
    answered = [it for it in st['items'] if it['role']]
    assert len(answered) == 5
    assert {it['object_id'] for it in answered} == \
           {o['object_id'] for o in objs[:5]}
    assert all(it['role'] == ONTO.NON_ACTIVE for it in answered)
    assert len([it for it in st['items'] if not it['role']]) == 123


def test_resolve_semantics_for_role_annotations_are_untouched(tmp_path):
    log = tmp_path / 'd.json'
    rows = [
        {'mode': 'missed_role_manual', 'BOX_ID': 'train:1',
         'HUMAN_FINAL_CLASS': 'goalkeeper',
         'recorded_utc': '2026-08-14T10:00:00Z'},
        {'mode': ONTO.ONTOLOGY_MODE, 'BOX_ID': 'BALLOBJ:zz',
         'missing_object_id': 'BALLOBJ:zz', 'IMAGE': 'train/a.jpg',
         'HUMAN_BALL_ROLE': ONTO.ACTIVE, 'HUMAN_FINAL_CLASS': None,
         'recorded_utc': '2026-08-14T11:00:00Z'},
    ]
    log.write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
    res = kb_decisions.resolve(log)
    assert res['train:1']['final_class'] == 'goalkeeper'
    assert res['BALLOBJ:zz']['final_class'] is None
    assert res['BALLOBJ:zz']['disposition'] is None
    assert kb_decisions.geometry_repairs(log) == {}
    assert kb_decisions.ball_cases(log) == {}
    assert kb_decisions.missing_targets(log) == {}


# --------------------------------------------------- image-level derivation


def _image_roles(objs, onto):
    """Fold objects up to images -- the shape the secondary analysis needs."""
    per = {}
    for o in objs:
        per.setdefault(o['IMAGE'], []).append(onto.get(o['object_id']))
    return per


def test_image_level_active_count_deduplicates_multiple_active_objects():
    """Three ACTIVE objects in one image is still ONE image with an active miss.

    This is the same rule Round 0 used for its own endpoint, and getting it
    wrong would push the derived active-ball rate above the measured 28%.
    """
    objs = [{'IMAGE': 'train/a.jpg', 'object_id': 'o1'},
            {'IMAGE': 'train/a.jpg', 'object_id': 'o2'},
            {'IMAGE': 'train/a.jpg', 'object_id': 'o3'},
            {'IMAGE': 'train/b.jpg', 'object_id': 'o4'},
            {'IMAGE': 'train/c.jpg', 'object_id': 'o5'}]
    onto = {'o1': ONTO.ACTIVE, 'o2': ONTO.ACTIVE, 'o3': ONTO.NON_ACTIVE,
            'o4': ONTO.NON_ACTIVE, 'o5': ONTO.UNSURE}
    per = _image_roles(objs, onto)
    active_images = {im for im, rs in per.items() if ONTO.ACTIVE in rs}
    non_only = {im for im, rs in per.items()
                if rs and all(r == ONTO.NON_ACTIVE for r in rs)}
    assert active_images == {'train/a.jpg'}
    assert len(active_images) == 1, 'two ACTIVE objects, one image'
    assert non_only == {'train/b.jpg'}
    assert sum(1 for r in onto.values() if r == ONTO.ACTIVE) == 2, 'objects'


def test_derived_active_image_rate_cannot_exceed_the_frozen_rate():
    """Every ACTIVE image is a Round-0 positive image, so the derived
    secondary rate is bounded above by the measured 84/300."""
    objs = ONTO.round0_objects()
    images = {o['IMAGE'] for o in objs}
    fr = _frozen()['primary']
    assert len(images) == fr['positive_images']
    assert len(images) / fr['n'] == fr['rate']


# ------------------------------------------------- the server, over HTTP


@pytest.fixture(scope='module')
def live(tmp_path_factory):
    import threading
    from http.server import ThreadingHTTPServer

    d = tmp_path_factory.mktemp('onto')
    log = d / 'decisions.json'
    log.write_text(ONTO.DECISIONS.read_text(encoding='utf-8'), encoding='utf-8')
    old = ONTO.DECISIONS
    ONTO.DECISIONS = log
    objs = ONTO.round0_objects(log)
    ONTO.H.OBJECTS = {o['object_id']: o for o in objs}
    ONTO.H.BALL_GT = ONTO.ball_gt_index(log)
    srv = ThreadingHTTPServer(('127.0.0.1', 0), ONTO.H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f'http://127.0.0.1:{srv.server_port}', log, objs
    srv.shutdown()
    ONTO.DECISIONS = old


def _route(base, route, payload):
    import urllib.error
    import urllib.request
    req = urllib.request.Request(
        base + route, data=json.dumps(payload).encode(),
        headers={'Content-Type': 'application/json'}, method='POST')
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def _post(base, payload):
    return _route(base, '/api/ontology', payload)


def _flagpost(base, payload):
    return _route(base, '/api/flag', payload)


def _events(log, mode, box):
    return [json.loads(l) for l in log.read_text(encoding='utf-8').splitlines()
            if l.strip() and json.loads(l).get('mode') == mode
            and json.loads(l).get('BOX_ID') == box]


def test_server_rejects_an_object_outside_the_128(live):
    base, log, objs = live
    before = log.read_text(encoding='utf-8')
    code, body = _post(base, {'object_id': 'BALLOBJ:not_a_real_object',
                              'HUMAN_BALL_ROLE': ONTO.ACTIVE})
    assert code == 400 and 'not one of the Round-0 missing objects' in body['error']
    assert log.read_text(encoding='utf-8') == before, 'nothing appended'


def test_server_rejects_a_role_outside_the_three(live):
    base, log, objs = live
    code, body = _post(base, {'object_id': objs[0]['object_id'],
                              'HUMAN_BALL_ROLE': 'PROBABLY_THE_MATCH_BALL'})
    assert code == 400 and 'HUMAN_BALL_ROLE must be one of' in body['error']


def test_server_records_geometry_unchanged(live):
    base, log, objs = live
    o = objs[0]
    code, _ = _post(base, {'object_id': o['object_id'],
                           'HUMAN_BALL_ROLE': ONTO.NON_ACTIVE})
    assert code == 200
    rows = [json.loads(l) for l in log.read_text(encoding='utf-8').splitlines()
            if l.strip()]
    ev = [r for r in rows if r.get('missing_object_id') == o['object_id']]
    assert len(ev) == 1
    assert ev[0]['round0_bbox_xywh'] == o['bbox_xywh'], 'original box carried'
    assert ev[0]['geometry_unchanged'] is True
    assert ev[0]['no_new_geometry_created'] is True
    assert ev[0]['no_model_proposal_used'] is True
    assert ev[0]['HUMAN_FINAL_CLASS'] is None
    assert ev[0]['mode'] == ONTO.ONTOLOGY_MODE


def test_server_re_answer_appends_and_latest_wins(live):
    base, log, objs = live
    o = objs[1]
    assert _post(base, {'object_id': o['object_id'],
                        'HUMAN_BALL_ROLE': ONTO.UNSURE})[0] == 200
    assert _post(base, {'object_id': o['object_id'],
                        'HUMAN_BALL_ROLE': ONTO.ACTIVE})[0] == 200
    rec = ONTO.ontology(log)[o['object_id']]
    assert rec['role'] == ONTO.ACTIVE
    assert [h['role'] for h in rec['history']] == [ONTO.UNSURE, ONTO.ACTIVE]


def test_server_two_objects_in_one_image_are_independent(live):
    base, log, objs = live
    by_img = {}
    for o in objs:
        by_img.setdefault(o['IMAGE'], []).append(o)
    pair = next(v for v in by_img.values() if len(v) > 1)
    assert _post(base, {'object_id': pair[0]['object_id'],
                        'HUMAN_BALL_ROLE': ONTO.ACTIVE})[0] == 200
    assert _post(base, {'object_id': pair[1]['object_id'],
                        'HUMAN_BALL_ROLE': ONTO.NON_ACTIVE})[0] == 200
    o = ONTO.ontology(log)
    assert o[pair[0]['object_id']]['role'] == ONTO.ACTIVE
    assert o[pair[1]['object_id']]['role'] == ONTO.NON_ACTIVE


def test_server_serves_only_images_holding_a_finding(live):
    import urllib.error
    import urllib.request
    base, log, objs = live
    outside = next(r['IMAGE'] for r in
                   json.loads(SAMPLE.MANIFEST.read_text(encoding='utf-8'))['sample']
                   if r['IMAGE'] not in {o['IMAGE'] for o in objs})
    for path, want in ((objs[0]['IMAGE'], 200), (outside, 403)):
        try:
            with urllib.request.urlopen(base + '/img/' + path, timeout=20) as r:
                got = r.status
        except urllib.error.HTTPError as e:
            got = e.code
        assert got == want, f'{path} returned {got}'


# ------------------------------------------------------- ball GT flags


def test_ball_gt_index_covers_only_existing_annotations():
    idx = ONTO.ball_gt_index()
    st = ONTO.build_state()
    shown = {g['BOX_ID'] for it in st['items'] for g in it['ball_gt']}
    assert shown == set(idx), 'what the UI offers is what the server accepts'
    assert idx, 'these images do hold existing ball GT'
    for box, gt in idx.items():
        split, _, aid = box.partition(':')
        assert split in SAMPLE.SPLITS and aid.isdigit()
        assert gt['annotation_id'] == int(aid)


def test_flag_keys_are_split_qualified():
    """Raw COCO ids collide across splits -- train and valid share 4,508."""
    ids = {}
    for sp in SAMPLE.SPLITS:
        doc = json.loads((EXPORT / f'{sp}_annotations.coco.json')
                         .read_text(encoding='utf-8'))
        ids[sp] = {a['id'] for a in doc['annotations']}
    assert ids['train'] & ids['valid'], 'bare ids are genuinely ambiguous'
    assert all(':' in b for b in ONTO.ball_gt_index())


def test_no_round0_object_can_appear_in_the_flag_index():
    """The two populations must never merge: one is what the dataset claims,
    the other is what it missed."""
    idx = ONTO.ball_gt_index()
    for o in ONTO.round0_objects():
        assert o['object_id'] not in idx
    assert not any(b.startswith('BALLOBJ:') for b in idx)


def test_flag_fold_latest_wins_and_retraction_is_append_only(tmp_path):
    log = tmp_path / 'd.json'
    box = 'train:1234'
    rows = [
        {'mode': ONTO.FLAG_MODE, 'BOX_ID': box, 'flag_type': ONTO.FALSE_BALL,
         'IMAGE': 'train/a.jpg', 'annotation_id': 1234,
         'bbox_xywh': [1, 2, 3, 4], 'recorded_utc': '2026-08-14T10:00:00Z'},
        {'mode': ONTO.FLAG_MODE, 'BOX_ID': box, 'flag_type': ONTO.BAD_BOX,
         'IMAGE': 'train/a.jpg', 'annotation_id': 1234,
         'bbox_xywh': [1, 2, 3, 4], 'recorded_utc': '2026-08-14T11:00:00Z'},
        {'mode': ONTO.FLAG_RETRACT_MODE, 'BOX_ID': box,
         'target_flag_event': '2026-08-14T11:00:00Z',
         'reason': 'human correction',
         'recorded_utc': '2026-08-14T12:00:00Z'},
    ]
    log.write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
    eff = ONTO.gt_flags(log)[box]
    assert eff['flag_type'] is None and eff['retracted'] is True
    assert len(eff['history']) == 3, 'nothing was deleted'
    assert len(kb_decisions.read_log(log)) == 3
    assert ONTO.flag_counts(log) == {'false': 0, 'bad_box': 0, 'retracted': 1}


def test_flag_can_be_raised_again_after_retraction(tmp_path):
    log = tmp_path / 'd.json'
    box = 'train:9'
    rows = [
        {'mode': ONTO.FLAG_MODE, 'BOX_ID': box, 'flag_type': ONTO.FALSE_BALL,
         'IMAGE': 'train/a.jpg', 'recorded_utc': '2026-08-14T10:00:00Z'},
        {'mode': ONTO.FLAG_RETRACT_MODE, 'BOX_ID': box,
         'recorded_utc': '2026-08-14T11:00:00Z'},
        {'mode': ONTO.FLAG_MODE, 'BOX_ID': box, 'flag_type': ONTO.BAD_BOX,
         'IMAGE': 'train/a.jpg', 'recorded_utc': '2026-08-14T12:00:00Z'},
    ]
    log.write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
    eff = ONTO.gt_flags(log)[box]
    assert eff['flag_type'] == ONTO.BAD_BOX and eff['retracted'] is False


def test_duplicate_identical_flags_are_unambiguous(tmp_path):
    """Two identical flags mean the same thing as one. No ambiguity to resolve."""
    log = tmp_path / 'd.json'
    box = 'train:5'
    row = {'mode': ONTO.FLAG_MODE, 'BOX_ID': box, 'flag_type': ONTO.FALSE_BALL,
           'IMAGE': 'train/a.jpg'}
    rows = [dict(row, recorded_utc='2026-08-14T10:00:00Z'),
            dict(row, recorded_utc='2026-08-14T10:00:05Z')]
    log.write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
    eff = ONTO.gt_flags(log)[box]
    assert eff['flag_type'] == ONTO.FALSE_BALL
    assert len(eff['history']) == 2
    assert ONTO.flag_counts(log)['false'] == 1, 'counted once, per annotation'


def test_flags_and_ontology_answers_never_read_each_other(tmp_path):
    log = tmp_path / 'd.json'
    rows = [
        {'mode': ONTO.ONTOLOGY_MODE, 'BOX_ID': 'BALLOBJ:aa',
         'missing_object_id': 'BALLOBJ:aa', 'IMAGE': 'train/a.jpg',
         'HUMAN_BALL_ROLE': ONTO.ACTIVE, 'recorded_utc': '2026-08-14T10:00:00Z'},
        {'mode': ONTO.FLAG_MODE, 'BOX_ID': 'train:7',
         'flag_type': ONTO.FALSE_BALL, 'IMAGE': 'train/a.jpg',
         'recorded_utc': '2026-08-14T11:00:00Z'},
    ]
    log.write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
    assert set(ONTO.ontology(log)) == {'BALLOBJ:aa'}
    assert set(ONTO.gt_flags(log)) == {'train:7'}
    res = kb_decisions.resolve(log)
    for k in ('BALLOBJ:aa', 'train:7'):
        assert res[k]['final_class'] is None and res[k]['disposition'] is None


def test_shared_readers_survive_a_row_without_a_classification(tmp_path):
    """An observation event has no HUMAN_FINAL_CLASS, and resolve() read it
    with [] rather than .get(). One such row would raise KeyError and take down
    every consumer of the WHOLE log, not just its own mode -- the flag pass
    found this the first time it wrote one.
    """
    log = tmp_path / 'd.json'
    rows = [
        {'mode': 'missed_role_manual', 'BOX_ID': 'train:1',
         'HUMAN_FINAL_CLASS': 'referee', 'recorded_utc': '2026-08-14T10:00:00Z'},
        # no HUMAN_FINAL_CLASS key at all
        {'mode': ONTO.FLAG_MODE, 'BOX_ID': 'train:2',
         'flag_type': ONTO.FALSE_BALL, 'IMAGE': 'train/a.jpg',
         'recorded_utc': '2026-08-14T11:00:00Z'},
    ]
    log.write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
    res = kb_decisions.resolve(log)
    assert res['train:1']['final_class'] == 'referee'
    assert res['train:2']['final_class'] is None
    assert res['train:2']['disposition'] is None
    assert kb_decisions.by_mode(log)[(ONTO.FLAG_MODE, 'train:2')] is None
    assert kb_decisions.prior_non_manual(log, 'train:2') == (None, ONTO.FLAG_MODE)
    # the manual click is still classified; the flag row simply does not appear
    assert set(kb_decisions.classify_manual(log)) == {'train:1'}
    assert kb_decisions.conflicts(log) == []


def test_flag_events_carry_the_field_shared_readers_expect():
    """Belt and braces: the tool writes it even though resolve() now tolerates
    its absence, so an older checkout of kb_decisions cannot be broken by a
    log this tool produced."""
    src = (REPO / 'tools' / 'kb_ball_ontology_revisit_server.py').read_text(
        encoding='utf-8')
    body = src[src.index('def _flag('):]
    assert body.count("'HUMAN_FINAL_CLASS': None") == 2, \
        'both the flag and its retraction must carry the field'


def test_flag_server_rejects_a_non_gt_target(live):
    base, log, objs = live
    before = log.read_text(encoding='utf-8')
    for bad in (objs[0]['object_id'], 'train:999999', 'nonsense'):
        code, body = _flagpost(base, {'BOX_ID': bad,
                                      'flag_type': ONTO.FALSE_BALL})
        assert code == 400, f'{bad} was accepted'
        assert 'not an existing ball annotation' in body['error']
    assert log.read_text(encoding='utf-8') == before, 'nothing appended'


def test_flag_server_rejects_an_unknown_flag_type(live):
    base, log, objs = live
    box = next(iter(ONTO.H.BALL_GT))
    code, body = _flagpost(base, {'BOX_ID': box, 'flag_type': 'LOOKS_ODD'})
    assert code == 400 and 'flag_type must be one of' in body['error']


def test_flag_server_records_false_ball(live):
    base, log, objs = live
    box = sorted(ONTO.H.BALL_GT)[0]
    gt = ONTO.H.BALL_GT[box]
    code, _ = _flagpost(base, {'BOX_ID': box, 'flag_type': ONTO.FALSE_BALL})
    assert code == 200
    ev = _events(log, ONTO.FLAG_MODE, box)[-1]
    assert ev['flag_type'] == ONTO.FALSE_BALL
    assert ev['annotation_id'] == gt['annotation_id']
    assert ev['IMAGE'] == gt['IMAGE']
    assert ev['bbox_xywh'] == gt['bbox_xywh']
    assert ev['current_class'] == 'football'
    assert ev['reason'] == 'human visual review'
    assert ev['annotation_unchanged'] is True
    assert ev['no_annotation_modified'] is True
    assert ev['author'] == 'human reviewer'


def test_flag_server_records_bad_box(live):
    base, log, objs = live
    box = sorted(ONTO.H.BALL_GT)[1]
    assert _flagpost(base, {'BOX_ID': box,
                            'flag_type': ONTO.BAD_BOX})[0] == 200
    assert _events(log, ONTO.FLAG_MODE, box)[-1]['flag_type'] == ONTO.BAD_BOX


def test_flag_server_retraction_names_what_it_withdraws(live):
    base, log, objs = live
    box = sorted(ONTO.H.BALL_GT)[2]
    assert _flagpost(base, {'BOX_ID': box,
                            'flag_type': ONTO.FALSE_BALL})[0] == 200
    raised = ONTO.gt_flags(log)[box]['recorded_utc']
    assert _flagpost(base, {'BOX_ID': box, 'retract': True})[0] == 200
    ev = _events(log, ONTO.FLAG_RETRACT_MODE, box)[-1]
    assert ev['target_flag_event'] == raised
    assert ev['retracts_flag_type'] == ONTO.FALSE_BALL
    assert ev['reason'] == 'human correction'
    assert ONTO.gt_flags(log)[box]['flag_type'] is None


def test_flag_server_refuses_to_retract_nothing(live):
    base, log, objs = live
    box = sorted(ONTO.H.BALL_GT)[3]
    code, body = _flagpost(base, {'BOX_ID': box, 'retract': True})
    assert code == 400 and 'no effective flag' in body['error']


def test_flagging_does_not_answer_or_advance_the_ontology_queue(live):
    base, log, objs = live
    before_onto = ONTO.ontology(log)
    before_objs = ONTO.round0_objects(log)
    box = sorted(ONTO.H.BALL_GT)[4]
    assert _flagpost(base, {'BOX_ID': box,
                            'flag_type': ONTO.FALSE_BALL})[0] == 200
    assert ONTO.ontology(log) == before_onto, 'no ontology answer changed'
    after = ONTO.round0_objects(log)
    assert len(after) == len(before_objs) == 128, 'denominator untouched'
    assert [o['object_id'] for o in after] == \
           [o['object_id'] for o in before_objs]


def test_flagging_does_not_alter_round0_answers(live):
    base, log, objs = live
    before = QA.answers(log)
    box = sorted(ONTO.H.BALL_GT)[5]
    assert _flagpost(base, {'BOX_ID': box, 'flag_type': ONTO.BAD_BOX})[0] == 200
    assert QA.answers(log) == before


def test_flag_restart_reconstructs_effective_state(live):
    base, log, objs = live
    box = sorted(ONTO.H.BALL_GT)[6]
    assert _flagpost(base, {'BOX_ID': box,
                            'flag_type': ONTO.FALSE_BALL})[0] == 200
    st = ONTO.build_state(decisions=log)
    shown = [g for it in st['items'] for g in it['ball_gt']
             if g['BOX_ID'] == box]
    assert shown and all(g['flag'] == ONTO.FALSE_BALL for g in shown)
    assert st['flag_counts']['false'] >= 1


def test_flag_counts_ignore_retracted(live):
    base, log, objs = live
    counts = ONTO.flag_counts(log)
    eff = ONTO.gt_flags(log)
    assert counts['false'] == sum(1 for v in eff.values()
                                  if v['flag_type'] == ONTO.FALSE_BALL)
    assert counts['bad_box'] == sum(1 for v in eff.values()
                                    if v['flag_type'] == ONTO.BAD_BOX)
    for v in eff.values():
        assert not (v['retracted'] and v['flag_type'])


# ------------------------------------------------- the page, actually driven


HARNESS = REPO / 'tests' / 'js' / 'ball_ontology.js'


@pytest.fixture(scope='module')
def driven(tmp_path_factory):
    node = shutil.which('node')
    if not node:
        pytest.skip('node not installed')
    d = tmp_path_factory.mktemp('driven')
    (d / 'page.html').write_text(ONTO.PAGE, encoding='utf-8')
    st = ONTO.build_state()
    # Drive from an UNCLASSIFIED state, so these assertions keep testing the
    # page rather than however far the human has got. The Round-0 page tests
    # had to learn this the moment that round was completed.
    for it in st['items']:
        it['role'] = None
        it['history'] = []
        for s in it['siblings']:
            s['role'] = None
            s['short'] = ''
    (d / 'state.json').write_text(json.dumps(st), encoding='utf-8')
    r = subprocess.run([node, str(HARNESS), str(d / 'page.html'),
                        str(d / 'state.json')],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)


def test_page_script_parses():
    import kb_review_server2
    assert kb_review_server2.page_script_defects(ONTO.PAGE) == []


def test_page_keyboard_mapping_is_what_the_ui_advertises(driven):
    assert driven['keys']['a'] == ONTO.ACTIVE
    assert driven['keys']['x'] == ONTO.NON_ACTIVE
    assert driven['keys']['u'] == ONTO.UNSURE


def test_page_n_key_is_unbound(driven):
    """'n' reads as both NEXT and NON-ACTIVE. Either binding mislabels silently."""
    assert driven['keys']['n'] is None, "'n' must classify nothing"


def test_page_navigation_never_classifies(driven):
    assert driven['navigation']['posted_anything'] is False
    assert driven['navigation']['moved_on_j'] is True


def test_page_classifies_the_highlighted_object_only(driven):
    m = driven['multi_object_image']
    assert m['siblings'] >= 1 and m['sibling_ids_differ'] is True
    assert m['posted'] == [m['object_id']], 'only the target was classified'
    assert m['overlay_boxes'] >= 2, 'target and siblings are both drawn'


def test_page_objects_in_one_image_take_different_roles(driven):
    ind = driven['independent']
    assert ind['same_image'] is True and ind['distinct_from_first'] is True
    assert ind['roles'] == [ONTO.NON_ACTIVE, ONTO.ACTIVE]


def test_page_can_jump_to_the_object_and_back(driven):
    z = driven['zoom']
    assert z['fit'] == 1 and z['on_object'] > 1 and z['back_to_fit'] == 1


def test_page_only_ever_posts_queued_objects(driven):
    assert driven['all_posted_in_queue'] is True
    assert driven['roles_posted'] == sorted(ONTO.ROLES)


def test_page_flag_keys_need_a_selection_first(driven):
    f = driven['flag_without_selection']
    assert f['posted'] == 0, 'F/V/C with nothing selected must post nothing'
    assert f['alerted'] is True, 'and must say why'


def test_page_click_selects_ball_gt_by_geometry(driven):
    s = driven['selection']
    assert s['matched'] is True, f'{s["selGT"]} != {s["expected"]}'


def test_page_flag_targets_existing_gt_never_the_round0_object(driven):
    """The magenta box is a human drawing with no annotation id. F must not
    reach it -- the whole point is that flags describe what the dataset claims."""
    f = driven['flag_false']
    assert f['posted'] == 1
    assert f['flag_type'] == ONTO.FALSE_BALL
    assert f['targets_existing_gt'] is True
    assert f['is_not_the_round0_object'] is True
    assert f['carries_no_object_id'] is True


def test_page_flagging_is_inert_for_the_ontology_pass(driven):
    i = driven['flag_is_inert']
    assert i['index_unchanged'] is True, 'a flag must not advance the queue'
    assert i['ontology_posts_added'] == 0, 'a flag is not an ontology answer'


def test_page_v_and_c_keys(driven):
    assert driven['flag_bad_box']['flag_type'] == ONTO.BAD_BOX
    assert driven['flag_clear']['retract'] is True


def test_page_duplicate_flags_stay_unambiguous(driven):
    d = driven['duplicate_flag']
    assert d['count'] == 2 and d['same_target'] and d['same_type']


def test_page_overlapping_gt_selection_is_deterministic(driven):
    o = driven.get('overlap')
    if not o:
        pytest.skip('no image with multiple ball GT in this sample')
    assert o['deterministic'] is True, 'the same click must select the same box'
    assert o['cycles'] is True, 'repeat clicks cycle only when boxes overlap'


def test_page_clicking_empty_space_clears_selection(driven):
    c = driven['click_empty_clears']
    assert c['had'] is True and c['now'] is None


def test_page_flag_posts_only_ever_name_shown_gt(driven):
    assert driven['flag_posts_all_target_existing_gt'] is True


# ------------------------------------------------------------ immutability


FROZEN_SHA = 'abeda1ff52011ece2b62d3b3df226f157bdcf41789336a774b1228bb911100ec'


def test_frozen_round0_result_is_byte_identical():
    assert hashlib.sha256(RESULT.read_bytes()).hexdigest() == FROZEN_SHA
    fr = _frozen()
    assert fr['frozen'] is True
    assert fr['primary']['positive_images'] == 84
    assert fr['primary']['n'] == 300
    assert fr['secondary']['total_missing_objects'] == 128


def test_repaired_export_is_byte_identical():
    fp = SAMPLE.population_fingerprint()
    man = json.loads(SAMPLE.MANIFEST.read_text(encoding='utf-8'))
    assert fp['population_sha256'] == man['population']['population_sha256']
    ONTO.build_state()
    assert SAMPLE.population_fingerprint()['population_sha256'] == \
           fp['population_sha256']


def test_round0_events_are_unchanged_by_the_ontology_pass():
    """Reading and classifying must leave every Round-0 answer exactly as it was."""
    before = QA.answers()
    ONTO.build_state()
    ONTO.ontology()
    after = QA.answers()
    assert before == after
    assert sum(len(v['missing']) for v in after.values()
               if v['answer'] == 'MISSING_BALL') == 128


def test_ontology_tool_writes_only_append_only_events():
    src = (REPO / 'tools' / 'kb_ball_ontology_revisit_server.py').read_text(
        encoding='utf-8')
    assert "open(DECISIONS, 'a'" in src, 'the log is opened for append only'
    for bad in ("open(DECISIONS, 'w'", 'DECISIONS.write_text',
                'RESULT.write_text', 'ROUND0_RESULT.write_text'):
        assert bad not in src, f'{bad} would rewrite an authoritative file'
    # flags and their retraction are the only two writes the flag path makes,
    # and both go through the same append()
    assert src.count('def append(') == 1
    assert 'repaired_export' not in src.replace(
        'kb_ball_qa_sample.EXPORT', ''), 'the export is only read via the sampler'


def test_ontology_tool_never_touches_a_detector():
    import ast
    tree = ast.parse((REPO / 'tools' / 'kb_ball_ontology_revisit_server.py')
                     .read_text(encoding='utf-8'))
    banned = {'torch', 'ultralytics', 'cv2', 'yolo', 'onnxruntime',
              'tensorflow', 'keras'}
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split('.')[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split('.')[0])
    assert not (imported & banned)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            assert not node.value.endswith('.pt')


def test_tool_refuses_if_the_queue_disagrees_with_the_frozen_result():
    """A queue that is not exactly the frozen findings must not be reviewable."""
    src = (REPO / 'tools' / 'kb_ball_ontology_revisit_server.py').read_text(
        encoding='utf-8')
    assert 'total_missing_objects' in src
    assert re.search(r'if len\(objs\) != want', src), \
        'main() must compare the queue against the frozen object count'
