"""Model-assisted completion of the PP sweep: proposals, verdicts, provenance.

The rule this file exists to defend is that a model proposal is a question, not
an annotation. Everything else here follows from it: the queue only covers
images no human answered, a YES is the sole path to an addition, and completing
the queue is never evidence that the images are clean.
"""

import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'tools'))

import kb_ball_candidate_server as CS                             # noqa: E402
import kb_ball_candidates as CAND                                 # noqa: E402
import kb_ball_pp_sweep_server as PP                              # noqa: E402
import kb_ball_qa_sample as SAMPLE                                # noqa: E402
import kb_ball_qa_server as QA                                    # noqa: E402
import kb_decisions                                               # noqa: E402

PKG = REPO / 'experiments' / 'external_sources' / 'keremberke_review'
RESULT = PKG / 'BALL_QA_ROUND0_RESULT.json'
POLICY = PKG / 'BALL_ONTOLOGY_POLICY.json'
FROZEN_SHA = 'abeda1ff52011ece2b62d3b3df226f157bdcf41789336a774b1228bb911100ec'

pytestmark = pytest.mark.skipif(
    not CAND.CANDIDATES.is_file(), reason='candidate queue not generated')


@pytest.fixture(scope='module')
def q():
    return json.loads(CAND.CANDIDATES.read_text(encoding='utf-8'))


# ------------------------------------------- completed human work is kept


def test_completed_pp_answers_are_preserved():
    a = PP.answers()
    queue = {r['IMAGE'] for r in
             json.loads(PP.QUEUE.read_text(encoding='utf-8'))['images']}
    done = {im: v for im, v in a.items() if im in queue}
    assert len(done) >= 256, 'human answers must never be dropped'
    drawn = sum(len(v['missing']) for v in done.values()
                if v['answer'] == 'MISSING_BALL')
    assert drawn >= 248
    for v in done.values():
        assert v['answer'] in PP.ANSWERS


def test_candidates_cover_only_unanswered_images(q):
    """A reviewed image must never re-enter the workflow: a model proposal
    must not get to compete with a human answer already on record."""
    answered = {im for im, v in PP.answers().items() if v.get('answer')}
    cand_images = {c['IMAGE'] for c in q['candidates']}
    assert not (cand_images & answered)
    unresolved = {r['IMAGE'] for r in CAND.unresolved_images()}
    assert cand_images <= unresolved


def test_unresolved_set_excludes_round0_and_plain():
    rows = CAND.unresolved_images()
    r0 = PP.round0_images()
    for r in rows:
        assert r['IMAGE'] not in r0
        assert PP.is_pp(r['IMAGE'])


def test_pp_totals_reconcile(q):
    queue = json.loads(PP.QUEUE.read_text(encoding='utf-8'))
    answered = {im for im, v in PP.answers().items()
                if v.get('answer')} & {r['IMAGE'] for r in queue['images']}
    assert len(answered) + q['population']['unresolved_pp_images'] == \
           queue['n'] == 357


# ------------------------------------------------------- provenance classes


def test_three_evidence_classes_are_distinguishable():
    """HUMAN_ADDITION, EXISTING_SOURCE_GT and HUMAN_IMAGE_REVIEW must not be
    collapsed: a human-drawn ball is stronger evidence than inherited GT, and
    NO_MISSING_BALL certifies neither."""
    src = (REPO / 'tools' / 'kb_ball_candidate_server.py').read_text(
        encoding='utf-8')
    assert 'HUMAN_APPROVED_ADDITION' in src
    assert 'HUMAN_REJECTED_PROPOSAL' in src
    assert 'UNRESOLVED_PROPOSAL' in src


def test_no_missing_ball_does_not_certify_existing_gt():
    """The sweep answer is about ADDITIONS. Nothing in its record claims the
    existing annotations in that image were validated."""
    a = PP.answers()
    clean = [v for v in a.values() if v['answer'] == 'NO_MISSING_BALL']
    assert clean
    for v in clean[:20]:
        assert 'existing_gt_verified' not in v
        assert 'clean' not in json.dumps(v).lower()


# ------------------------------------------------------------- the queue


def test_candidate_queue_is_proposal_only(q):
    assert q['no_candidate_is_ground_truth'] is True
    for c in q['candidates'][:50]:
        assert 'MODEL PROPOSAL' in c['geometry_author']
        assert 'not an annotation' in c['geometry_author']


def test_candidate_ids_are_unique_and_geometry_based(q):
    ids = [c['candidate_id'] for c in q['candidates']]
    assert len(set(ids)) == len(ids)
    a = CAND.candidate_id('train/x.jpg', [1, 2, 3, 4])
    assert a == CAND.candidate_id('train/x.jpg', [1.0, 2.0, 3.0, 4.0])
    assert a != CAND.candidate_id('train/x.jpg', [1, 2, 3, 5])
    assert a.startswith('BALLCAND:')


def test_matching_uses_centre_distance_not_iou():
    gts = [{'BOX_ID': 'train:1', 'bbox_xywh': [100.0, 100.0, 5.0, 5.0]}]
    # a 3 px offset on a 5 px ball: IoU would be tiny, but it is the same object
    assert CAND.matches_gt([103.0, 100.0, 5.0, 5.0], gts) == 'train:1'
    assert CAND.matches_gt([400.0, 400.0, 5.0, 5.0], gts) is None
    # the 8 px floor protects very small annotations
    assert CAND.matches_gt([107.0, 100.0, 4.0, 4.0], gts) == 'train:1'


def test_plausibility_filter_keeps_real_ball_sizes():
    assert CAND.plausible([0, 0, 4.0, 4.0])[0]
    assert CAND.plausible([0, 0, 9.0, 8.0])[0]
    assert not CAND.plausible([0, 0, 1.0, 1.0])[0]
    assert not CAND.plausible([0, 0, 400.0, 200.0])[0]
    assert not CAND.plausible([0, 0, 20.0, 2.0])[0], 'aspect ratio'


def test_queue_is_bound_to_the_export_fingerprint(q):
    assert q['population']['population_sha256'] == \
           SAMPLE.population_fingerprint()['population_sha256']


def test_queue_records_the_weights_it_used(q):
    assert q['model']['weights'] == 'best_A_960.pt'
    assert len(q['model']['weights_sha256']) == 64
    assert q['model']['conf_floor'] == 0.03


def test_queue_states_its_own_blind_spot(q):
    note = q['model']['note'].lower()
    assert 'weakest on small balls' in note
    assert 'residual' in note


# ---------------------------------------------------------- residual QA


def test_residual_sample_is_deterministic(q):
    a = CAND.residual_sample(q)
    b = CAND.residual_sample(q)
    assert [r['IMAGE'] for r in a['images']] == [r['IMAGE'] for r in b['images']]


def test_residual_sample_prefers_zero_candidate_images(q):
    r = CAND.residual_sample(q)
    zero = set(q['zero_candidate_images'])
    got = {x['IMAGE'] for x in r['images'] if x['stratum'] == 'ZERO_CANDIDATE'}
    assert got == zero, 'every zero-candidate image must be included'


def test_residual_sample_is_honest_about_the_shortfall(q):
    """The requested 40 is not attainable, and the file says why rather than
    padding the number or raising the threshold to manufacture a pool."""
    r = CAND.residual_sample(q)
    assert r['requested_n'] == 40
    assert r['n'] <= 40
    if r['n'] < 40:
        assert 'why_not_pure_zero_candidate' in r
        assert 'manufacture' in r['why_not_pure_zero_candidate']
    for s in r['strata'].values():
        assert s['n'] <= s['N']


def test_residual_images_are_all_unresolved_pp(q):
    r = CAND.residual_sample(q)
    unresolved = {x['IMAGE'] for x in CAND.unresolved_images()}
    for x in r['images']:
        assert x['IMAGE'] in unresolved


# --------------------------------------------------------------- verdicts


def test_verdict_fold_latest_wins(tmp_path):
    log = tmp_path / 'd.json'
    cid = 'BALLCAND:abc'
    rows = [
        {'mode': CS.CAND_MODE, 'BOX_ID': cid, 'candidate_id': cid,
         'IMAGE': 'train/a.jpg', 'verdict': CS.NO,
         'recorded_utc': '2026-08-14T10:00:00Z'},
        {'mode': CS.CAND_MODE, 'BOX_ID': cid, 'candidate_id': cid,
         'IMAGE': 'train/a.jpg', 'verdict': CS.YES,
         'recorded_utc': '2026-08-14T11:00:00Z'},
    ]
    log.write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
    v = CS.verdicts(log)[cid]
    assert v['verdict'] == CS.YES
    assert [h['verdict'] for h in v['history']] == [CS.NO, CS.YES]
    assert len(kb_decisions.read_log(log)) == 2


def test_candidate_events_do_not_disturb_other_folds(tmp_path):
    log = tmp_path / 'd.json'
    rows = [
        {'mode': 'missed_role_manual', 'BOX_ID': 'train:1',
         'HUMAN_FINAL_CLASS': 'referee', 'recorded_utc': '2026-08-14T10:00:00Z'},
        {'mode': CS.CAND_MODE, 'BOX_ID': 'BALLCAND:z', 'candidate_id': 'BALLCAND:z',
         'IMAGE': 'train/a.jpg', 'verdict': CS.YES, 'HUMAN_FINAL_CLASS': None,
         'recorded_utc': '2026-08-14T11:00:00Z'},
    ]
    log.write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
    assert kb_decisions.resolve(log)['train:1']['final_class'] == 'referee'
    assert QA.answers(log) == {}
    assert PP.answers(log) == {}


# ------------------------------------------------- server over real HTTP


@pytest.fixture(scope='module')
def live(tmp_path_factory, q):
    import threading
    from http.server import ThreadingHTTPServer
    d = tmp_path_factory.mktemp('cand')
    log = d / 'decisions.json'
    log.write_text(CS.DECISIONS.read_text(encoding='utf-8'), encoding='utf-8')
    old = CS.DECISIONS
    CS.DECISIONS = log
    CS.H.CANDS = {c['candidate_id']: c for c in q['candidates']}
    srv = ThreadingHTTPServer(('127.0.0.1', 0), CS.H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f'http://127.0.0.1:{srv.server_port}', log, q
    srv.shutdown()
    CS.DECISIONS = old


def _post(base, payload):
    import urllib.error
    import urllib.request
    req = urllib.request.Request(
        base + '/api/verdict', data=json.dumps(payload).encode(),
        headers={'Content-Type': 'application/json'}, method='POST')
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def _events(log, cid):
    return [json.loads(l) for l in log.read_text(encoding='utf-8').splitlines()
            if l.strip() and json.loads(l).get('candidate_id') == cid]


def test_server_rejects_an_unknown_candidate(live):
    base, log, q = live
    before = log.read_text(encoding='utf-8')
    code, body = _post(base, {'candidate_id': 'BALLCAND:nope',
                              'verdict': CS.YES})
    assert code == 400 and 'not a generated candidate' in body['error']
    assert log.read_text(encoding='utf-8') == before


def test_server_rejects_an_unknown_verdict(live):
    base, log, q = live
    cid = q['candidates'][0]['candidate_id']
    code, body = _post(base, {'candidate_id': cid, 'verdict': 'PROBABLY'})
    assert code == 400 and 'verdict must be one of' in body['error']


def test_yes_records_a_human_approved_addition(live):
    base, log, q = live
    c = q['candidates'][1]
    assert _post(base, {'candidate_id': c['candidate_id'],
                        'verdict': CS.YES})[0] == 200
    ev = _events(log, c['candidate_id'])[-1]
    assert ev['evidence_class'] == 'HUMAN_APPROVED_ADDITION'
    assert ev['approved_bbox_xywh'] == c['bbox_xywh']
    assert ev['geometry_author'] == 'model proposal, human approved'
    assert ev['proposal_source'].startswith('best_A_960.pt')
    assert ev['ontology'] == 'ALL_VISIBLE_PHYSICAL_FOOTBALLS'
    assert ev['no_annotation_modified'] is True
    assert 'ACTIVE' not in json.dumps(ev).replace('NON_ACTIVE', '')


def test_no_and_unsure_create_no_addition(live):
    base, log, q = live
    for c, v, want in ((q['candidates'][2], CS.NO, 'HUMAN_REJECTED_PROPOSAL'),
                       (q['candidates'][3], CS.UNSURE, 'UNRESOLVED_PROPOSAL')):
        assert _post(base, {'candidate_id': c['candidate_id'],
                            'verdict': v})[0] == 200
        ev = _events(log, c['candidate_id'])[-1]
        assert ev['evidence_class'] == want
        assert 'approved_bbox_xywh' not in ev


def test_only_yes_candidates_are_approved_additions(live):
    base, log, q = live
    v = CS.verdicts(log)
    approved = {c for c, r in v.items() if r['verdict'] == CS.YES}
    for cid, r in v.items():
        if r['verdict'] != CS.YES:
            assert cid not in approved
    ids = {c['candidate_id'] for c in q['candidates']}
    assert approved <= ids


def test_multiple_candidates_per_image_are_independent(live):
    base, log, q = live
    by_img = {}
    for c in q['candidates']:
        by_img.setdefault(c['IMAGE'], []).append(c)
    pair = next(v for v in by_img.values() if len(v) > 1)
    assert _post(base, {'candidate_id': pair[0]['candidate_id'],
                        'verdict': CS.YES})[0] == 200
    assert _post(base, {'candidate_id': pair[1]['candidate_id'],
                        'verdict': CS.NO})[0] == 200
    v = CS.verdicts(log)
    assert v[pair[0]['candidate_id']]['verdict'] == CS.YES
    assert v[pair[1]['candidate_id']]['verdict'] == CS.NO


def test_restart_resumes_from_the_log(live):
    base, log, q = live
    c = q['candidates'][5]
    assert _post(base, {'candidate_id': c['candidate_id'],
                        'verdict': CS.UNSURE})[0] == 200
    assert _post(base, {'candidate_id': c['candidate_id'],
                        'verdict': CS.YES})[0] == 200
    r = CS.verdicts(log)[c['candidate_id']]
    assert r['verdict'] == CS.YES and len(r['history']) == 2


# ------------------------------------------------------------- the page


def test_page_parses():
    import kb_review_server2
    assert kb_review_server2.page_script_defects(CS.PAGE) == []


def test_page_asks_one_question_and_never_active_vs_non_active():
    p = CS.PAGE
    assert 'IS THIS A REAL MISSING FOOTBALL?' in p
    for w in ('ACTIVE MATCH BALL', 'NON-ACTIVE', 'ACTIVE_MATCH_BALL'):
        assert w not in p
    for k in ('>Y<', '>N<', '>U<'):
        assert k in p.replace('<span class="k">', '>').replace('</span>', '<')


def test_page_shows_existing_gt_and_sibling_proposals():
    st = CS.build_state()
    assert len(st['items']) > 0
    with_gt = [i for i in st['items'] if i['ball_gt']]
    with_sib = [i for i in st['items'] if i['siblings']]
    assert with_gt and with_sib
    for it in st['items'][:20]:
        assert 'context' in it


# ------------------------------------------------ no training, no writes


def test_candidate_generation_is_inference_only():
    src = (REPO / 'tools' / 'kb_ball_candidates.py').read_text(encoding='utf-8')
    for bad in ('.train(', 'model.fit', 'save(', 'export('):
        assert bad not in src, f'{bad} would not be inference'
    assert 'YOLO(' in src and 'predict(' in src


def test_review_server_never_loads_a_detector():
    tree = ast.parse((REPO / 'tools' / 'kb_ball_candidate_server.py')
                     .read_text(encoding='utf-8'))
    banned = {'torch', 'ultralytics', 'cv2'}
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split('.')[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split('.')[0])
    assert not (imported & banned), 'the reviewer must not see a model run'


def test_tools_only_append_to_the_log():
    for name in ('kb_ball_candidates.py', 'kb_ball_candidate_server.py'):
        src = (REPO / 'tools' / name).read_text(encoding='utf-8')
        for bad in ("open(DECISIONS, 'w'", 'DECISIONS.write_text',
                    'RESULT.write_text'):
            assert bad not in src


# ------------------------------------------------------------ immutability


def test_frozen_round0_result_unchanged():
    assert hashlib.sha256(RESULT.read_bytes()).hexdigest() == FROZEN_SHA


def test_round0_findings_unchanged():
    a = QA.answers()
    pos = {im: v for im, v in a.items() if v['answer'] == 'MISSING_BALL'}
    assert len(pos) == 84
    assert sum(len(v['missing']) for v in pos.values()) == 128


def test_export_unchanged(q):
    assert SAMPLE.population_fingerprint()['population_sha256'] == \
           q['population']['population_sha256']


def test_ontology_policy_unchanged():
    p = json.loads(POLICY.read_text(encoding='utf-8'))
    assert p['BALL_DETECTOR_ONTOLOGY'] == 'ALL_VISIBLE_PHYSICAL_FOOTBALLS'
    assert p['ACTIVE_MATCH_BALL_SELECTION'] == 'DOWNSTREAM_TEMPORAL_SELECTOR'


def test_existing_gt_observations_are_preserved():
    """E/F/V observations already recorded stay as provenance."""
    import kb_ball_ontology_revisit_server as ONTO
    c = ONTO.flag_counts()
    assert c['non_active'] == 22
    assert c['false'] == 1
