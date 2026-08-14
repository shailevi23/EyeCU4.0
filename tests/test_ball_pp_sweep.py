"""PP BALL SWEEP: the census of pp images Round 0 never reviewed.

The load-bearing property is the queue's boundary. Round 0's 300 images are
frozen with 128 findings; if one of them leaked into this queue a second answer
could silently supersede a frozen one, and the two populations could no longer
be combined without double-counting. So most of these tests are about what is
NOT in the queue.
"""

import ast
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'tools'))

import kb_ball_ontology_revisit_server as ONTO                    # noqa: E402
import kb_ball_pp_sweep_server as PP                              # noqa: E402
import kb_ball_qa_sample as SAMPLE                                # noqa: E402
import kb_ball_qa_server as QA                                    # noqa: E402
import kb_decisions                                               # noqa: E402

PKG = REPO / 'experiments' / 'external_sources' / 'keremberke_review'
EXPORT = PKG / 'repaired_export'
RESULT = PKG / 'BALL_QA_ROUND0_RESULT.json'
POLICY = PKG / 'BALL_ONTOLOGY_POLICY.json'
FROZEN_SHA = 'abeda1ff52011ece2b62d3b3df226f157bdcf41789336a774b1228bb911100ec'

pytestmark = pytest.mark.skipif(
    not RESULT.is_file(), reason='frozen Round-0 result not in this checkout')


# ------------------------------------------------------------------ policy


def test_ontology_policy_is_recorded_and_binding():
    p = json.loads(POLICY.read_text(encoding='utf-8'))
    assert p['BALL_DETECTOR_ONTOLOGY'] == 'ALL_VISIBLE_PHYSICAL_FOOTBALLS'
    assert p['ACTIVE_MATCH_BALL_SELECTION'] == 'DOWNSTREAM_TEMPORAL_SELECTOR'
    assert p['status'] == 'BINDING'
    assert p['active_vs_non_active']['is_a_detector_class_distinction'] is False


def test_policy_forbids_removing_non_active_gt():
    """The 22 EXISTING_NON_ACTIVE annotations are correct under this policy.
    A later tool must not be able to read that flag as a deletion instruction."""
    p = json.loads(POLICY.read_text(encoding='utf-8'))
    forbidden = ' '.join(p['forbidden_without_a_new_recorded_decision']).lower()
    assert 'removal instruction' in forbidden
    assert 'active_only' in forbidden
    assert p['consequences']['existing_non_active_ball_gt_must_not_be_removed'] == 22
    assert ONTO.FALSE_BALL in p['flags_that_remain_defect_reports']
    assert ONTO.BAD_BOX in p['flags_that_remain_defect_reports']
    assert ONTO.EXISTING_NON_ACTIVE not in p['flags_that_remain_defect_reports']


def test_sweep_events_record_the_ontology_they_were_answered_under():
    src = (REPO / 'tools' / 'kb_ball_pp_sweep_server.py').read_text(
        encoding='utf-8')
    assert "'ontology': 'ALL_VISIBLE_PHYSICAL_FOOTBALLS'" in src


# ------------------------------------------------------------------- queue


@pytest.fixture(scope='module')
def queue():
    return json.loads(PP.QUEUE.read_text(encoding='utf-8'))


def test_queue_is_all_pp_images_minus_round0(queue):
    pop = SAMPLE.population()
    pp = {r['IMAGE'] for r in pop if PP.is_pp(r['IMAGE'])}
    r0 = PP.round0_images()
    want = pp - r0
    got = {r['IMAGE'] for r in queue['images']}
    assert got == want
    assert len(got) == queue['n']
    assert queue['population']['N_pp_images'] == len(pp)
    assert queue['round0_pp_already_reviewed'] == len(pp & r0)
    assert len(pp) == len(got) + len(pp & r0)


def test_queue_has_zero_overlap_with_round0(queue):
    got = {r['IMAGE'] for r in queue['images']}
    assert not (got & PP.round0_images())


def test_queue_contains_no_plain_images(queue):
    for r in queue['images']:
        assert PP.is_pp(r['IMAGE']), r['IMAGE']
        assert '_pp_' in r['IMAGE'].rsplit('/', 1)[-1]


def test_pp_membership_comes_from_the_filename_not_the_run_label():
    """One pp frame has no ledger run. A run-based population would be 485 and
    would silently drop it, which is the wrong answer to "every pp image"."""
    pop = SAMPLE.population()
    by_name = {r['IMAGE'] for r in pop if PP.is_pp(r['IMAGE'])}
    by_run = {r['IMAGE'] for r in pop if str(r['run']).startswith('pp')}
    assert by_name > by_run, 'the filename definition is the wider one'
    missed = by_name - by_run
    assert missed, 'this dataset does contain such an image'
    for im in missed:
        assert '_pp_' in im


def test_every_queue_image_exists_in_the_export(queue):
    pop = {r['IMAGE'] for r in SAMPLE.population()}
    for r in queue['images']:
        assert r['IMAGE'] in pop


def test_queue_is_deterministic(queue):
    a = [r['IMAGE'] for r in PP.build_queue()['images']]
    b = [r['IMAGE'] for r in PP.build_queue()['images']]
    assert a == b
    assert a == [r['IMAGE'] for r in queue['images']]


def test_queue_is_bound_to_the_export_fingerprint(queue):
    assert queue['population']['population_sha256'] == \
           SAMPLE.population_fingerprint()['population_sha256']


def test_queue_is_a_census_not_a_sample(queue):
    """No inclusion probability applies to a complete enumeration, so none is
    claimed -- quoting a CI off a census would be a category error."""
    assert 'census' in queue['design']
    assert 'inclusion_probability' not in queue
    assert 'seed' not in queue


# ------------------------------------------------------------------- state


def test_state_exposes_only_queue_images(queue):
    st = PP.build_state()
    assert [i['IMAGE'] for i in st['items']] == \
           [r['IMAGE'] for r in queue['images']]
    assert not ({i['IMAGE'] for i in st['items']} & PP.round0_images())


def test_state_carries_existing_ball_gt():
    st = PP.build_state()
    pop = {r['IMAGE']: r for r in SAMPLE.population()}
    for it in st['items'][:60]:
        assert len(it['ball_gt']) == pop[it['IMAGE']]['ball_gt_count']
    assert st['ontology'] == 'ALL_VISIBLE_PHYSICAL_FOOTBALLS'


def test_answers_fold_latest_wins(tmp_path):
    log = tmp_path / 'd.json'
    rows = [
        {'mode': PP.SWEEP_MODE, 'BOX_ID': 'PPSWEEP:train/a.jpg',
         'IMAGE': 'train/a.jpg', 'answer': 'NO_MISSING_BALL',
         'missing_balls': [], 'recorded_utc': '2026-08-14T10:00:00Z'},
        {'mode': PP.SWEEP_MODE, 'BOX_ID': 'PPSWEEP:train/a.jpg',
         'IMAGE': 'train/a.jpg', 'answer': 'MISSING_BALL',
         'missing_balls': [{'bbox_xywh': [1, 2, 4, 4]},
                           {'bbox_xywh': [9, 9, 5, 5]}],
         'recorded_utc': '2026-08-14T11:00:00Z'},
        {'mode': QA.QA_MODE, 'BOX_ID': 'BALLQA0:train/b.jpg',
         'IMAGE': 'train/b.jpg', 'answer': 'MISSING_BALL',
         'missing_balls': [{'bbox_xywh': [3, 3, 4, 4]}],
         'recorded_utc': '2026-08-14T12:00:00Z'},
    ]
    log.write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
    a = PP.answers(log)
    assert set(a) == {'train/a.jpg'}, 'Round-0 events are a different mode'
    assert a['train/a.jpg']['answer'] == 'MISSING_BALL'
    assert len(a['train/a.jpg']['missing']) == 2
    assert len(a['train/a.jpg']['history']) == 2


def test_sweep_events_do_not_disturb_round0_or_box_resolution(tmp_path):
    log = tmp_path / 'd.json'
    rows = [
        {'mode': 'missed_role_manual', 'BOX_ID': 'train:1',
         'HUMAN_FINAL_CLASS': 'referee', 'recorded_utc': '2026-08-14T10:00:00Z'},
        {'mode': PP.SWEEP_MODE, 'BOX_ID': 'PPSWEEP:train/a.jpg',
         'IMAGE': 'train/a.jpg', 'answer': 'MISSING_BALL',
         'HUMAN_FINAL_CLASS': None,
         'missing_balls': [{'bbox_xywh': [1, 1, 4, 4]}],
         'recorded_utc': '2026-08-14T11:00:00Z'},
    ]
    log.write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
    res = kb_decisions.resolve(log)
    assert res['train:1']['final_class'] == 'referee'
    assert res['PPSWEEP:train/a.jpg']['final_class'] is None
    assert QA.answers(log) == {}, 'Round-0 fold sees nothing'
    assert ONTO.ontology(log) == {}


# ------------------------------------------------------------ no detector


def test_sweep_path_never_touches_a_detector():
    tree = ast.parse((REPO / 'tools' / 'kb_ball_pp_sweep_server.py')
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


def test_no_detector_module_imported_at_runtime():
    out = subprocess.run(
        [sys.executable, '-c',
         'import sys; sys.path.insert(0, r"%s");'
         'import kb_ball_pp_sweep_server;'
         'bad=[m for m in sys.modules if m.split(".")[0] in '
         '("torch","ultralytics","cv2")]; print(",".join(bad))'
         % (REPO / 'tools')],
        capture_output=True, text=True, timeout=180)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == ''


# ------------------------------------------------------ server over HTTP


@pytest.fixture(scope='module')
def live(tmp_path_factory, queue):
    import threading
    from http.server import ThreadingHTTPServer

    d = tmp_path_factory.mktemp('ppsweep')
    log = d / 'decisions.json'
    log.write_text(PP.DECISIONS.read_text(encoding='utf-8'), encoding='utf-8')
    old = PP.DECISIONS
    PP.DECISIONS = log
    PP.PAGE = PP._page()
    PP.H.IMAGES = frozenset(r['IMAGE'] for r in queue['images'])
    PP.H.DIMS = {r['IMAGE']: (r['img_w'], r['img_h']) for r in queue['images']}
    srv = ThreadingHTTPServer(('127.0.0.1', 0), PP.H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f'http://127.0.0.1:{srv.server_port}', log, queue
    srv.shutdown()
    PP.DECISIONS = old


def _post(base, payload):
    import urllib.error
    import urllib.request
    req = urllib.request.Request(
        base + '/api/answer', data=json.dumps(payload).encode(),
        headers={'Content-Type': 'application/json'}, method='POST')
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_server_rejects_a_round0_image(live):
    """The frozen 129 must be unanswerable here, or a second answer could
    supersede a finding the frozen result already counted."""
    base, log, q = live
    before = log.read_text(encoding='utf-8')
    r0 = sorted(PP.round0_images())[0]
    code, body = _post(base, {'IMAGE': r0, 'answer': 'NO_MISSING_BALL'})
    assert code == 400 and 'not in the pp sweep queue' in body['error']
    assert log.read_text(encoding='utf-8') == before


def test_server_rejects_an_image_outside_the_queue(live):
    base, log, q = live
    code, body = _post(base, {'IMAGE': 'train/NOT_REAL.jpg',
                              'answer': 'NO_MISSING_BALL'})
    assert code == 400 and 'not in the pp sweep queue' in body['error']


def test_server_rejects_unknown_answers_and_empty_positives(live):
    base, log, q = live
    im = q['images'][0]['IMAGE']
    assert _post(base, {'IMAGE': im, 'answer': 'MAYBE'})[0] == 400
    code, body = _post(base, {'IMAGE': im, 'answer': 'MISSING_BALL',
                              'missing_balls_xywh': []})
    assert code == 400 and 'at least one drawn ball' in body['error']


def test_server_records_multiple_missing_balls_as_one_event(live):
    base, log, q = live
    im = q['images'][1]['IMAGE']
    code, body = _post(base, {'IMAGE': im, 'answer': 'MISSING_BALL',
                              'missing_balls_xywh': [[10, 10, 4.2, 3.9],
                                                     [700, 400, 9, 9],
                                                     [200, 300, 6, 6]]})
    assert code == 200 and len(body['missing']) == 3
    rows = [json.loads(l) for l in log.read_text(encoding='utf-8').splitlines()
            if l.strip()]
    ev = [r for r in rows if r.get('mode') == PP.SWEEP_MODE
          and r['IMAGE'] == im]
    assert len(ev) == 1, 'one image-level event, not one per ball'
    assert ev[0]['n_missing_objects'] == 3
    assert ev[0]['ontology'] == 'ALL_VISIBLE_PHYSICAL_FOOTBALLS'
    assert all(b['geometry_author'] == 'human drawn'
               for b in ev[0]['missing_balls'])


def test_tiny_four_px_ball_survives_the_round_trip(live):
    base, log, q = live
    im = q['images'][2]['IMAGE']
    code, body = _post(base, {'IMAGE': im, 'answer': 'MISSING_BALL',
                              'missing_balls_xywh': [[640.37, 360.62,
                                                      4.24, 3.81]]})
    assert code == 200
    got = body['missing'][0]['bbox_xywh']
    assert got == [640.37, 360.62, 4.24, 3.81]
    assert PP.answers(log)[im]['missing'][0]['bbox_xywh'] == got


def test_server_restart_resumes_from_the_log(live):
    base, log, q = live
    im = q['images'][3]['IMAGE']
    assert _post(base, {'IMAGE': im, 'answer': 'UNSURE'})[0] == 200
    assert PP.answers(log)[im]['answer'] == 'UNSURE'
    assert _post(base, {'IMAGE': im, 'answer': 'NO_MISSING_BALL'})[0] == 200
    a = PP.answers(log)[im]
    assert a['answer'] == 'NO_MISSING_BALL', 'latest wins'
    assert len(a['history']) == 2, 'the UNSURE event is still on record'


def test_server_serves_only_queue_images(live):
    import urllib.error
    import urllib.request
    base, log, q = live
    r0 = sorted(PP.round0_images())[0]
    for path, want in ((q['images'][0]['IMAGE'], 200), (r0, 403)):
        try:
            with urllib.request.urlopen(base + '/img/' + path, timeout=20) as r:
                got = r.status
        except urllib.error.HTTPError as e:
            got = e.code
        assert got == want, f'{path} returned {got}'


# --------------------------------------------------------------- the page


def test_page_parses_and_is_relabelled():
    import kb_review_server2
    p = PP._page()
    assert kb_review_server2.page_script_defects(p) == []
    assert 'PP BALL SWEEP' in p and 'BALL QA ROUND 0' not in p
    assert 'REAL PHYSICAL FOOTBALL' in p


def test_page_does_not_ask_active_versus_non_active():
    """Under ALL_VISIBLE the answer would change nothing, so asking it would
    be wasted human effort and an invitation to inconsistency."""
    p = PP._page()
    for word in ('ACTIVE MATCH BALL', 'NON-ACTIVE EXTRA', 'ACTIVE_MATCH_BALL'):
        assert word not in p


def test_page_keeps_the_three_answers():
    p = PP._page()
    for a in ('NO MISSING BALL', 'MISSING BALL', 'UNSURE'):
        assert a in p


@pytest.mark.skipif(not shutil.which('node'), reason='node not installed')
def test_page_multi_ball_drawing_still_works(tmp_path):
    """Driven against the real relabelled page, via the Round-0 harness."""
    d = tmp_path
    (d / 'page.html').write_text(PP._page(), encoding='utf-8')
    st = PP.build_state()
    st['items'] = st['items'][:5]
    for it in st['items']:
        it['answer'] = None
        it['missing'] = []
        it['history'] = []
    (d / 'state.json').write_text(json.dumps(st), encoding='utf-8')
    r = subprocess.run(
        [shutil.which('node'), str(REPO / 'tests' / 'js' / 'ball_qa_round0.js'),
         str(d / 'page.html'), str(d / 'state.json')],
        capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    assert out['multi_object']['objects_posted'] == 3
    assert out['multi_object']['answer'] == 'MISSING_BALL'
    assert out['tiny_at_zoom']['width_ok'] and out['tiny_at_zoom']['height_ok']
    assert out['navigation']['stayed_in_sample'] is True


# ------------------------------------------------------------ immutability


def test_frozen_round0_result_is_byte_identical():
    assert hashlib.sha256(RESULT.read_bytes()).hexdigest() == FROZEN_SHA


def test_round0_findings_are_untouched():
    a = QA.answers()
    pos = {im: v for im, v in a.items() if v['answer'] == 'MISSING_BALL'}
    assert len(pos) == 84
    assert sum(len(v['missing']) for v in pos.values()) == 128
    assert len(ONTO.round0_objects()) == 128


def test_repaired_export_is_byte_identical():
    fp = SAMPLE.population_fingerprint()['population_sha256']
    PP.build_queue()
    PP.build_state()
    assert SAMPLE.population_fingerprint()['population_sha256'] == fp


def test_sweep_tool_only_appends():
    src = (REPO / 'tools' / 'kb_ball_pp_sweep_server.py').read_text(
        encoding='utf-8')
    assert "open(DECISIONS, 'a'" in src
    for bad in ("open(DECISIONS, 'w'", 'DECISIONS.write_text',
                'RESULT.write_text'):
        assert bad not in src
