"""BALL QA ROUND 0: the sample, the review server, and the frozen report.

These are invariants, not snapshots. An earlier round of this project pinned
counters and mode allowlists to whatever the code happened to produce, and every
one of those tests had to be rewritten the first time the code legitimately
changed. So nothing here asserts "300 images and these exact ids" as a golden
value; it asserts the properties that make the measurement mean something --
equal inclusion probability, determinism given a seed, invalidation when the
population changes, and a report that refuses rather than guesses.
"""

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'tools'))

import kb_ball_qa_sample as SAMPLE                                # noqa: E402
import kb_ball_qa_server as SERVER                                # noqa: E402
import kb_ball_round0_report as REPORT                            # noqa: E402
import kb_decisions                                               # noqa: E402

PKG = REPO / 'experiments' / 'external_sources' / 'keremberke_review'
EXPORT = PKG / 'repaired_export'
pytestmark = pytest.mark.skipif(
    not (EXPORT / 'train_annotations.coco.json').is_file(),
    reason='promoted export not present in this checkout')


# --------------------------------------------------------------- population


def test_population_is_every_image_in_the_export():
    pop = SAMPLE.population()
    n = 0
    for s in SAMPLE.SPLITS:
        doc = json.loads((EXPORT / f'{s}_annotations.coco.json')
                         .read_text(encoding='utf-8'))
        n += len(doc['images'])
    assert len(pop) == n
    assert len({r['IMAGE'] for r in pop}) == n, 'IMAGE must identify an image'


def test_population_order_is_deterministic_before_the_seed():
    # if enumeration order varied, the seed would select different images from
    # run to run and reproducibility would be an illusion
    assert [r['IMAGE'] for r in SAMPLE.population()] == \
           [r['IMAGE'] for r in SAMPLE.population()]


def test_ball_gt_count_matches_the_export():
    pop = {r['IMAGE']: r for r in SAMPLE.population()}
    for s in SAMPLE.SPLITS:
        doc = json.loads((EXPORT / f'{s}_annotations.coco.json')
                         .read_text(encoding='utf-8'))
        bc = SAMPLE._ball_category(doc['categories'])
        by_id = {im['id']: im for im in doc['images']}
        counted = {}
        for a in doc['annotations']:
            if a['category_id'] == bc:
                counted[a['image_id']] = counted.get(a['image_id'], 0) + 1
        for im_id, im in by_id.items():
            key = f'{s}/{im["file_name"]}'
            assert pop[key]['ball_gt_count'] == counted.get(im_id, 0)
            assert pop[key]['gt_state'] == ('GT0' if not counted.get(im_id)
                                            else 'GT1+')


# ------------------------------------------------------------------ sampling


def test_sample_is_exactly_n_unique_images():
    man = SAMPLE.build()
    ids = [r['IMAGE'] for r in man['sample']]
    assert len(ids) == SAMPLE.N_SAMPLE
    assert len(set(ids)) == SAMPLE.N_SAMPLE, 'SRSWOR draws without replacement'
    pop = {r['IMAGE'] for r in SAMPLE.population()}
    assert set(ids) <= pop


def test_inclusion_probability_is_equal_and_exact():
    man = SAMPLE.build()
    assert man['inclusion_probability'] == man['n'] / man['population']['N']
    assert man['inclusion_probability_is_equal_for_every_image'] is True
    assert man['design'].startswith('simple random sample')


def test_equal_inclusion_probability_holds_empirically():
    """Over many seeds, every image must be reachable at the same rate.

    The point of SRSWOR here is that no image is structurally advantaged. A
    design bug -- sorting, slicing, or a stratum that quietly excluded someone --
    would show up as an image that never appears, or one that always does.
    """
    pop = SAMPLE.population()
    seen = {}
    trials = 60
    for s in range(trials):
        for r in SAMPLE.draw(pop, 300, seed=s):
            seen[r['IMAGE']] = seen.get(r['IMAGE'], 0) + 1
    assert len(seen) > 0.98 * len(pop), 'some images are never selectable'
    f = 300 / len(pop)
    hits = [seen.get(r['IMAGE'], 0) / trials for r in pop]
    assert abs(sum(hits) / len(hits) - f) < 0.01
    assert max(hits) < 0.75 and min(hits) < f + 0.5


def test_same_source_and_seed_give_the_same_images_in_the_same_order():
    a = SAMPLE.build()
    b = SAMPLE.build()
    assert [r['IMAGE'] for r in a['sample']] == [r['IMAGE'] for r in b['sample']]
    assert a['population']['population_sha256'] == b['population']['population_sha256']


def test_a_different_seed_gives_a_different_sample():
    a = [r['IMAGE'] for r in SAMPLE.build(seed=SAMPLE.SEED)['sample']]
    b = [r['IMAGE'] for r in SAMPLE.build(seed=SAMPLE.SEED + 1)['sample']]
    assert a != b


def test_changed_population_invalidates_the_sample(tmp_path):
    """A changed export must break the binding, not silently re-point it."""
    fake = tmp_path / 'repaired_export'
    fake.mkdir()
    for s in SAMPLE.SPLITS:
        src = EXPORT / f'{s}_annotations.coco.json'
        doc = json.loads(src.read_text(encoding='utf-8'))
        if s == 'train':                       # one annotation removed
            doc['annotations'] = doc['annotations'][1:]
        (fake / f'{s}_annotations.coco.json').write_text(
            json.dumps(doc), encoding='utf-8')
    before = SAMPLE.population_fingerprint(EXPORT)['population_sha256']
    after = SAMPLE.population_fingerprint(fake)['population_sha256']
    assert before != after

    stored = json.loads(SAMPLE.MANIFEST.read_text(encoding='utf-8'))
    assert stored['population']['population_sha256'] == before
    assert SAMPLE.build(fake)['population']['population_sha256'] == after


def test_stored_manifest_matches_a_fresh_draw():
    stored = json.loads(SAMPLE.MANIFEST.read_text(encoding='utf-8'))
    fresh = SAMPLE.build()
    assert [r['IMAGE'] for r in stored['sample']] == \
           [r['IMAGE'] for r in fresh['sample']]
    assert stored['seed'] == SAMPLE.SEED
    assert stored['population']['population_sha256'] == \
           fresh['population']['population_sha256']


def test_manifest_records_what_the_analysis_needs_to_read_back():
    man = json.loads(SAMPLE.MANIFEST.read_text(encoding='utf-8'))
    for k in ('seed', 'n', 'inclusion_probability', 'design', 'sample'):
        assert k in man
    for k in ('N', 'population_sha256', 'sha256_per_split'):
        assert k in man['population']
    for r in man['sample']:
        for k in ('IMAGE', 'split', 'run', 'ball_gt_count', 'gt_state',
                  'img_w', 'img_h'):
            assert k in r


def test_descriptive_variables_are_not_used_for_inclusion():
    """Metadata is described as descriptive AND behaves that way.

    Drawing from a population whose descriptive fields are blanked must produce
    the identical sample -- proof the fields play no part in selection.
    """
    pop = SAMPLE.population()
    blanked = [dict(r, run=None, gt_state='?', ball_gt_count=-1,
                    view_proxy=None) for r in pop]
    assert [r['IMAGE'] for r in SAMPLE.draw(pop, 300, SAMPLE.SEED)] == \
           [r['IMAGE'] for r in SAMPLE.draw(blanked, 300, SAMPLE.SEED)]


def test_presentation_order_is_not_population_order():
    """Draw order must interleave splits, or fatigue aligns with grouping."""
    man = json.loads(SAMPLE.MANIFEST.read_text(encoding='utf-8'))
    splits = [r['split'] for r in man['sample']]
    runs = [r['run'] for r in man['sample'] if r['run']]
    assert splits != sorted(splits, key=SAMPLE.SPLITS.index)
    switches = sum(1 for a, b in zip(runs, runs[1:]) if a != b)
    assert switches > len(runs) / 4, 'runs appear in long blocks'


# ------------------------------------------------------- no detector anywhere


@pytest.mark.parametrize('mod', ['kb_ball_qa_sample', 'kb_ball_qa_server',
                                 'kb_ball_round0_report'])
def test_round0_path_never_touches_a_detector(mod):
    """Round 0's independence must be structural, not a promise in prose.

    Checked against the parsed AST rather than the text, so that the docstrings
    explaining WHY no detector is used do not themselves trip the check -- and,
    more importantly, so a real import cannot hide inside a string.
    """
    import ast
    tree = ast.parse((REPO / 'tools' / f'{mod}.py').read_text(encoding='utf-8'))
    banned = {'torch', 'ultralytics', 'cv2', 'yolo', 'onnxruntime',
              'tensorflow', 'keras'}
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split('.')[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split('.')[0])
    assert not (imported & banned), f'{mod} imports {imported & banned}'
    # and no weights file is named anywhere in executable code
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            assert not node.value.endswith('.pt'), f'{mod} names weights'


def test_no_detector_module_is_imported_at_runtime():
    out = subprocess.run(
        [sys.executable, '-c',
         'import sys; sys.path.insert(0, r"%s");'
         'import kb_ball_qa_sample, kb_ball_qa_server, kb_ball_round0_report;'
         'bad=[m for m in sys.modules if m.split(".")[0] in '
         '("torch","ultralytics","cv2")];'
         'print(",".join(bad))' % (REPO / 'tools')],
        capture_output=True, text=True, timeout=180)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == '', f'detector stack imported: {out.stdout}'


# ------------------------------------------------------------------- server


def test_served_page_script_parses():
    """A raw newline in a JS literal blanks the whole page while HTTP says 200."""
    import kb_review_server2
    assert kb_review_server2.page_script_defects(SERVER.PAGE) == []


def test_state_exposes_only_sampled_images():
    st = SERVER.build_state()
    man = json.loads(SAMPLE.MANIFEST.read_text(encoding='utf-8'))
    assert [i['IMAGE'] for i in st['items']] == [r['IMAGE'] for r in man['sample']]
    assert len(st['items']) == man['n']


def test_state_carries_the_real_ball_gt_for_each_image():
    st = SERVER.build_state()
    pop = {r['IMAGE']: r for r in SAMPLE.population()}
    for it in st['items'][:40]:
        assert len(it['ball_gt']) == pop[it['IMAGE']]['ball_gt_count']


def test_answers_fold_latest_wins_and_keeps_history(tmp_path):
    log = tmp_path / 'd.json'
    rows = [
        {'mode': SERVER.QA_MODE, 'BOX_ID': 'BALLQA0:train/a.jpg',
         'IMAGE': 'train/a.jpg', 'answer': 'NO_MISSING_BALL',
         'missing_balls': [], 'recorded_utc': '2026-08-14T10:00:00Z'},
        {'mode': SERVER.QA_MODE, 'BOX_ID': 'BALLQA0:train/a.jpg',
         'IMAGE': 'train/a.jpg', 'answer': 'MISSING_BALL',
         'missing_balls': [{'bbox_xywh': [1, 2, 4, 4]}],
         'recorded_utc': '2026-08-14T11:00:00Z'},
        {'mode': 'missed_role_manual', 'BOX_ID': 'train:1',
         'IMAGE': 'train/a.jpg', 'HUMAN_FINAL_CLASS': 'player',
         'recorded_utc': '2026-08-14T12:00:00Z'},
    ]
    log.write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
    a = SERVER.answers(log)
    assert set(a) == {'train/a.jpg'}, 'only ball_qa_r0 events are folded'
    assert a['train/a.jpg']['answer'] == 'MISSING_BALL'
    assert len(a['train/a.jpg']['history']) == 2, 'history is not rewritten'
    assert a['train/a.jpg']['history'][0]['answer'] == 'NO_MISSING_BALL'


def test_re_answering_appends_rather_than_replacing(tmp_path):
    log = tmp_path / 'd.json'
    base = {'mode': SERVER.QA_MODE, 'BOX_ID': 'BALLQA0:train/a.jpg',
            'IMAGE': 'train/a.jpg', 'missing_balls': []}
    lines = [dict(base, answer='UNSURE', recorded_utc='2026-08-14T10:00:00Z'),
             dict(base, answer='NO_MISSING_BALL',
                  recorded_utc='2026-08-14T10:05:00Z')]
    log.write_text(''.join(json.dumps(r) + '\n' for r in lines), encoding='utf-8')
    assert len(kb_decisions.read_log(log)) == 2
    a = SERVER.answers(log)['train/a.jpg']
    assert a['answer'] == 'NO_MISSING_BALL'
    assert [h['answer'] for h in a['history']] == ['UNSURE', 'NO_MISSING_BALL']


def test_answer_validation_rules():
    from kb_missing_target_server import validate_bbox
    assert validate_bbox([10, 10, 4, 4], 1280, 720)[1] is None, '4px ball is valid'
    assert validate_bbox([10, 10, 0, 4], 1280, 720)[1] is not None
    assert validate_bbox([1279, 719, 40, 40], 1280, 720)[1] is not None
    assert SERVER.ANSWERS == ('NO_MISSING_BALL', 'MISSING_BALL', 'UNSURE')


def test_tiny_box_survives_the_json_round_trip():
    from kb_missing_target_server import validate_bbox
    box, err = validate_bbox([640.37, 360.62, 4.24, 3.81], 1280, 720)
    assert err is None
    assert json.loads(json.dumps({'b': box}))['b'] == box
    assert box[2] >= 4 and box[3] >= 3.8


# ------------------------------------------------------------------- report


def _fake(tmp_path, answers, n=10):
    """A self-contained sample + log, so the real ones are never written to."""
    man = SAMPLE.build()
    sub = dict(man, n=n, sample=man['sample'][:n],
               inclusion_probability=n / man['population']['N'])
    sp = tmp_path / 'sample.json'
    sp.write_text(json.dumps(sub), encoding='utf-8')
    rows = []
    for im, (ans, boxes) in answers.items():
        rows.append({'mode': SERVER.QA_MODE, 'BOX_ID': f'BALLQA0:{im}',
                     'IMAGE': im, 'answer': ans,
                     'missing_balls': [{'bbox_xywh': b} for b in boxes],
                     'recorded_utc': '2026-08-14T10:00:00Z'})
    log = tmp_path / 'd.json'
    log.write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
    return sp, log, [r['IMAGE'] for r in sub['sample']]


def test_report_refuses_while_images_are_unanswered(tmp_path):
    sp, log, ids = _fake(tmp_path, {}, n=10)
    data, blocking = REPORT.collect(sp, log)
    assert not blocking
    rep = REPORT.build(data)
    assert rep['frozen'] is False
    assert rep['unresolved']['unanswered_images'] == 10


def test_report_refuses_on_a_changed_population(tmp_path):
    sp, log, ids = _fake(tmp_path, {}, n=5)
    man = json.loads(sp.read_text(encoding='utf-8'))
    man['population']['population_sha256'] = 'deadbeef' * 8
    sp.write_text(json.dumps(man), encoding='utf-8')
    data, blocking = REPORT.collect(sp, log)
    assert data is None or blocking
    assert any('changed' in b for b in blocking)


def test_report_refuses_when_an_unsampled_image_was_answered(tmp_path):
    sp, log, ids = _fake(tmp_path, {}, n=5)
    rows = [{'mode': SERVER.QA_MODE, 'BOX_ID': 'BALLQA0:train/NOTSAMPLED.jpg',
             'IMAGE': 'train/NOTSAMPLED.jpg', 'answer': 'NO_MISSING_BALL',
             'missing_balls': [], 'recorded_utc': '2026-08-14T10:00:00Z'}]
    log.write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
    data, blocking = REPORT.collect(sp, log)
    assert any('outside the sample' in b for b in blocking)


def test_unsure_is_never_counted_as_clean(tmp_path):
    ids = [r['IMAGE'] for r in SAMPLE.build()['sample'][:6]]
    ans = {i: ('NO_MISSING_BALL', []) for i in ids[:5]}
    ans[ids[5]] = ('UNSURE', [])
    sp, log, _ = _fake(tmp_path, ans, n=6)
    data, blocking = REPORT.collect(sp, log)
    rep = REPORT.build(data)
    assert rep['frozen'] is False, 'an UNSURE image leaves the round unfrozen'
    assert rep['unresolved']['unsure_images'] == 1
    assert 'LOWER BOUND' in rep['unresolved']['interpretation']
    assert rep['primary']['positive_images'] == 0


def test_primary_endpoint_counts_images_not_objects(tmp_path):
    man = SAMPLE.build()
    ids = [r['IMAGE'] for r in man['sample'][:10]]
    ans = {i: ('NO_MISSING_BALL', []) for i in ids}
    ans[ids[0]] = ('MISSING_BALL', [[10, 10, 4, 4], [50, 50, 9, 9],
                                    [90, 90, 30, 30]])
    ans[ids[1]] = ('MISSING_BALL', [[20, 20, 6, 6]])
    sp, log, _ = _fake(tmp_path, ans, n=10)
    data, blocking = REPORT.collect(sp, log)
    assert not blocking
    rep = REPORT.build(data)
    assert rep['primary']['positive_images'] == 2, 'two positive IMAGES'
    assert rep['secondary']['total_missing_objects'] == 4, 'four OBJECTS'
    assert rep['primary']['rate'] == 2 / 10
    assert rep['secondary']['objects_per_positive_image'] == {1: 1, 3: 1}
    assert 'no_object_rate_reported' in rep['secondary']
    assert rep['secondary']['size_buckets'] == {'<=5px': 1, '<=8px': 1,
                                               '<=12px': 1, '>12px': 1}


def test_clopper_pearson_matches_known_values():
    lo, hi = REPORT.clopper_pearson(0, 300)
    assert lo == 0.0 and abs(hi - 0.01220) < 5e-4
    lo, hi = REPORT.clopper_pearson(3, 300)
    assert abs(lo - 0.00207) < 5e-4 and abs(hi - 0.02893) < 5e-4
    lo, hi = REPORT.clopper_pearson(10, 300)
    assert abs(lo - 0.01614) < 5e-4 and abs(hi - 0.06035) < 5e-4


# ------------------------------- the exact finite-population interval


N_POP, N_SAMP = 1232, 300


@pytest.mark.parametrize('x,want_lo,want_hi', [
    (0, 0, 13),
    (1, 1, 20),
    (3, 4, 32),
    (10, 23, 70),
])
def test_hypergeometric_ci_endpoints(x, want_lo, want_hi):
    assert REPORT.hypergeometric_ci(x, N_POP, N_SAMP) == (want_lo, want_hi)


@pytest.mark.parametrize('x', [0, 1, 3, 10])
def test_hypergeometric_ci_satisfies_its_defining_inequalities(x):
    """The endpoints are exactly where the 2.5% tails begin, not near them.

    This is the property the interval IS, so it is checked directly rather than
    against remembered numbers: one step beyond either endpoint must tip the
    relevant tail probability below alpha/2.
    """
    from scipy.stats import hypergeom
    lo, hi = REPORT.hypergeometric_ci(x, N_POP, N_SAMP)
    a = 0.025
    assert hypergeom.cdf(x, N_POP, hi, N_SAMP) >= a
    if hi < N_POP:
        assert hypergeom.cdf(x, N_POP, hi + 1, N_SAMP) < a, 'hi is not maximal'
    assert hypergeom.sf(x - 1, N_POP, lo, N_SAMP) >= a
    if lo > 0:
        assert hypergeom.sf(x - 1, N_POP, lo - 1, N_SAMP) < a, 'lo is not minimal'


@pytest.mark.parametrize('x', [0, 1, 3, 10])
def test_interval_endpoints_are_valid_integer_population_counts(x):
    lo, hi = REPORT.hypergeometric_ci(x, N_POP, N_SAMP)
    assert isinstance(lo, int) and isinstance(hi, int)
    assert 0 <= lo <= hi <= N_POP
    for v in (lo, hi):
        assert v == int(v), 'a count of images cannot be fractional'
        assert 0.0 <= v / N_POP <= 1.0


@pytest.mark.parametrize('x', [0, 1, 3, 10])
def test_finite_population_interval_is_deterministic(x):
    a = REPORT.hypergeometric_ci(x, N_POP, N_SAMP)
    for _ in range(4):
        assert REPORT.hypergeometric_ci(x, N_POP, N_SAMP) == a


@pytest.mark.parametrize('x', [0, 1, 3, 10])
def test_finite_population_interval_is_narrower_than_binomial(x):
    """The FPC is the information Clopper-Pearson throws away."""
    lo, hi = REPORT.hypergeometric_ci(x, N_POP, N_SAMP)
    b_lo, b_hi = REPORT.clopper_pearson(x, N_SAMP)
    assert hi / N_POP < b_hi, 'the finite-population upper bound must be tighter'
    assert lo / N_POP >= b_lo - 1e-9


def test_hypergeometric_ci_covers_the_point_estimate_region():
    """x=0 must admit M=0; a sample of everything must pin M exactly."""
    assert REPORT.hypergeometric_ci(0, N_POP, N_SAMP)[0] == 0
    assert REPORT.hypergeometric_ci(7, 50, 50) == (7, 7), 'census leaves no doubt'


def test_hypergeometric_ci_rejects_impossible_inputs():
    with pytest.raises(ValueError):
        REPORT.hypergeometric_ci(301, N_POP, N_SAMP)
    with pytest.raises(ValueError):
        REPORT.hypergeometric_ci(-1, N_POP, N_SAMP)


def test_frozen_report_uses_the_finite_population_interval(tmp_path):
    ids = [r['IMAGE'] for r in SAMPLE.build()['sample'][:10]]
    ans = {i: ('NO_MISSING_BALL', []) for i in ids}
    sp, log, _ = _fake(tmp_path, ans, n=10)
    data, blocking = REPORT.collect(sp, log)
    assert not blocking
    p = REPORT.build(data)['primary']
    assert 'hypergeometric' in p['ci_method']
    lo, hi = p['ci95_population_counts']
    assert (lo, hi) == REPORT.hypergeometric_ci(0, p['N'], 10)
    assert p['ci95_finite_population'] == [lo / p['N'], hi / p['N']]
    ref = p['conservative_binomial_reference']
    assert 'NOT the exact interval' in ref['note']
    assert 'ci95_clopper_pearson' in ref
    # the old top-level key must be gone, so nothing keeps reading it as primary
    assert 'ci95_clopper_pearson' not in p


# -------------------------------------------- interim reports no interval


def _interim_text(tmp_path, answers, n):
    sp, log, _ = _fake(tmp_path, answers, n=n)
    data, blocking = REPORT.collect(sp, log)
    assert not blocking
    return REPORT.build(data)


def test_interim_emits_no_confidence_interval(tmp_path, capsys):
    ids = [r['IMAGE'] for r in SAMPLE.build()['sample'][:20]]
    ans = {i: ('NO_MISSING_BALL', []) for i in ids[:8]}
    ans[ids[8]] = ('MISSING_BALL', [[10, 10, 4, 4], [20, 20, 5, 5]])
    rep = _interim_text(tmp_path, ans, n=20)
    REPORT._print_interim(rep)
    out = capsys.readouterr().out
    for banned in ('95%', 'Clopper', 'hypergeometric', 'binomial'):
        assert banned not in out, f'interim printed {banned!r}'
    # no percentage anywhere, and no bracketed numeric pair that could be read
    # as an interval -- the caption is not what stops a number being quoted
    assert '%' not in out, 'interim printed a rate'
    assert not re.search(r'\[\s*[\d.]+\s*,\s*[\d.]+\s*\]', out), \
        'interim printed something shaped like an interval'
    assert 'NO CONFIDENCE INTERVAL IS REPORTED HERE' in out
    for shown in ('answered', 'outstanding', 'positive images',
                  'missing objects', 'UNSURE'):
        assert shown in out


def test_interim_reports_the_five_required_counts(tmp_path):
    ids = [r['IMAGE'] for r in SAMPLE.build()['sample'][:20]]
    ans = {i: ('NO_MISSING_BALL', []) for i in ids[:8]}
    ans[ids[8]] = ('MISSING_BALL', [[10, 10, 4, 4], [20, 20, 5, 5]])
    ans[ids[9]] = ('UNSURE', [])
    rep = _interim_text(tmp_path, ans, n=20)
    assert rep['unresolved']['unanswered_images'] == 10
    assert rep['primary']['positive_images'] == 1
    assert rep['secondary']['total_missing_objects'] == 2
    assert rep['unresolved']['unsure_images'] == 1


def test_interim_logical_bounds_are_arithmetic_not_inference(tmp_path):
    ids = [r['IMAGE'] for r in SAMPLE.build()['sample'][:20]]
    ans = {i: ('NO_MISSING_BALL', []) for i in ids[:8]}
    ans[ids[8]] = ('MISSING_BALL', [[10, 10, 4, 4]])
    ans[ids[9]] = ('UNSURE', [])
    b = _interim_text(tmp_path, ans, n=20)['interim_bounds']
    assert b['min_possible_positives'] == 1
    assert b['max_possible_positives'] == 1 + 10 + 1
    assert 'not a confidence interval' in b['note']


def test_interim_writes_nothing(tmp_path):
    """--interim never prints a rate and never touches the result file.

    Originally this also asserted the result file did not exist, which was true
    only while the round was unfinished. That was a snapshot: the round has
    since been completed and frozen. The invariant is that --interim leaves
    whatever is on disk exactly as it found it.
    """
    before = (REPORT.REPORT.read_bytes() if REPORT.REPORT.exists() else None)
    out = subprocess.run(
        [sys.executable, str(REPO / 'tools' / 'kb_ball_round0_report.py'),
         '--interim'], capture_output=True, text=True, timeout=180,
        cwd=str(REPO))
    assert out.returncode == 0, out.stderr
    assert '95%' not in out.stdout and 'Clopper' not in out.stdout
    assert '%' not in out.stdout, 'interim must not print a rate'
    assert not re.search(r'\[\s*[\d.]+\s*,\s*[\d.]+\s*\]', out.stdout)
    after = (REPORT.REPORT.read_bytes() if REPORT.REPORT.exists() else None)
    assert after == before, 'interim must not write or alter the result file'


def test_frozen_report_refuses_while_any_image_is_outstanding(tmp_path):
    """Refusal is driven by outstanding work, not by the calendar.

    This used to run the real tool and demand exit 3, which stopped being true
    the moment the reviewer finished all 300. The behaviour worth protecting is
    conditional, so it is now exercised against a deliberately incomplete
    sample rather than against whatever the live log happens to hold.
    """
    ids = [r['IMAGE'] for r in SAMPLE.build()['sample'][:6]]
    sp, log, _ = _fake(tmp_path, {i: ('NO_MISSING_BALL', []) for i in ids[:4]},
                       n=6)
    data, blocking = REPORT.collect(sp, log)
    assert not blocking
    rep = REPORT.build(data)
    assert rep['frozen'] is False
    assert rep['unresolved']['unanswered_images'] == 2
    # and the real CLI refuses on the same condition
    out = subprocess.run(
        [sys.executable, '-c',
         'import sys; sys.path.insert(0, r"%s");'
         'import kb_ball_round0_report as R;'
         'd, b = R.collect(r"%s", r"%s");'
         'print("INCOMPLETE" if (d["unanswered"] or d["unsure"]) else "COMPLETE")'
         % (REPO / 'tools', sp, log)],
        capture_output=True, text=True, timeout=180)
    assert out.stdout.strip() == 'INCOMPLETE', out.stderr


def test_frozen_result_on_disk_is_internally_consistent():
    """The round is complete, so the written result must hold together."""
    if not REPORT.REPORT.exists():
        pytest.skip('round not yet frozen in this checkout')
    rep = json.loads(REPORT.REPORT.read_text(encoding='utf-8'))
    p = rep['primary']
    assert rep['frozen'] is True
    assert rep['unresolved']['unanswered_images'] == 0
    assert rep['unresolved']['unsure_images'] == 0
    assert p['rate'] == p['positive_images'] / p['n']
    lo, hi = p['ci95_population_counts']
    assert (lo, hi) == REPORT.hypergeometric_ci(p['positive_images'],
                                                p['N'], p['n'])
    assert lo <= p['rate'] * p['N'] <= hi, 'the point estimate sits inside'
    assert sum(int(k) * v for k, v in
               rep['secondary']['objects_per_positive_image'].items()) == \
           rep['secondary']['total_missing_objects']
    assert sum(rep['secondary']['size_buckets'].values()) == \
           rep['secondary']['total_missing_objects']


@pytest.mark.parametrize('k,boxes,unsure,band', [
    (0, [], 0, '0 positives'),
    (2, [[0, 0, 30, 30]], 0, '1-3 positives, all found balls >12px'),
    (2, [[0, 0, 5, 5]], 0, '1-3 positives with at least one found ball <=8px'),
    (5, [[0, 0, 30, 30]], 0, '4-9 positives'),
    (12, [[0, 0, 30, 30]], 0, '>=10 positives'),
    (0, [], 16, 'UNSURE>15'),
])
def test_escalation_bands_are_pre_declared(k, boxes, unsure, band):
    objects = [{'width': b[2]} for b in boxes] * max(k, 1) if k else []
    out = REPORT._escalate(k, objects, unsure)
    assert out['band'] == band
    assert 'NOT started automatically' in out['note']


def test_escalation_large_ball_band_does_not_claim_a_cause():
    out = REPORT._escalate(2, [{'width': 30}], 0)
    text = out['pre_declared_action'].lower()
    assert 'no evidence' in text and 'tiny-ball' in text
    assert 'do not claim a cause' in text
    for word in ('fatigue', 'because', 'caused by'):
        assert word not in text, f'the band asserts a cause: {word!r}'


def test_report_fingerprints_bind_it_to_what_it_measured(tmp_path):
    man = SAMPLE.build()
    ids = [r['IMAGE'] for r in man['sample'][:4]]
    sp, log, _ = _fake(tmp_path, {i: ('NO_MISSING_BALL', []) for i in ids}, n=4)
    data, _ = REPORT.collect(sp, log)
    rep = REPORT.build(data)
    f = rep['fingerprints']
    assert f['population_sha256'] == man['population']['population_sha256']
    assert f['population_matches_live_export'] is True
    assert len(f['sample_manifest_sha256']) == 64
    assert 'decisions_sha256' in f['decisions_log']
    assert rep['frozen'] is True


# ------------------------------------------- the server, over real HTTP


@pytest.fixture(scope='module')
def live(tmp_path_factory):
    """The real handler on a real socket, writing to a throwaway log."""
    import threading
    from http.server import ThreadingHTTPServer

    man = json.loads(SAMPLE.MANIFEST.read_text(encoding='utf-8'))
    log = tmp_path_factory.mktemp('live') / 'decisions.json'
    log.write_text('', encoding='utf-8')
    old = SERVER.DECISIONS
    SERVER.DECISIONS = log
    SERVER.H.SAMPLED = frozenset(r['IMAGE'] for r in man['sample'])
    SERVER.H.DIMS = {r['IMAGE']: (r['img_w'], r['img_h']) for r in man['sample']}
    srv = ThreadingHTTPServer(('127.0.0.1', 0), SERVER.H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f'http://127.0.0.1:{srv.server_port}', log, man
    srv.shutdown()
    SERVER.DECISIONS = old


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


def test_server_rejects_an_image_outside_the_sample(live):
    base, log, man = live
    code, body = _post(base, {'IMAGE': 'train/NOT_SAMPLED.jpg',
                              'answer': 'NO_MISSING_BALL'})
    assert code == 400
    assert 'not in the Round-0 sample' in body['error']
    assert log.read_text(encoding='utf-8') == '', 'nothing was appended'


def test_server_rejects_an_unknown_answer(live):
    base, log, man = live
    code, body = _post(base, {'IMAGE': man['sample'][0]['IMAGE'],
                              'answer': 'PROBABLY_FINE'})
    assert code == 400 and 'answer must be one of' in body['error']


def test_server_rejects_a_positive_with_no_geometry(live):
    """A positive with nothing drawn would be an unusable finding."""
    base, log, man = live
    code, body = _post(base, {'IMAGE': man['sample'][0]['IMAGE'],
                              'answer': 'MISSING_BALL',
                              'missing_balls_xywh': []})
    assert code == 400 and 'at least one drawn ball' in body['error']


def test_server_rejects_an_out_of_bounds_box(live):
    base, log, man = live
    code, body = _post(base, {'IMAGE': man['sample'][0]['IMAGE'],
                              'answer': 'MISSING_BALL',
                              'missing_balls_xywh': [[5000, 5000, 4, 4]]})
    assert code == 400 and 'outside' in body['error']


def test_server_records_a_multi_object_positive_as_one_event(live):
    base, log, man = live
    im = man['sample'][1]['IMAGE']
    code, body = _post(base, {'IMAGE': im, 'answer': 'MISSING_BALL',
                              'missing_balls_xywh': [[10, 10, 4.2, 3.9],
                                                     [700, 400, 9, 9]]})
    assert code == 200 and len(body['missing']) == 2
    rows = [json.loads(l) for l in
            log.read_text(encoding='utf-8').splitlines() if l.strip()]
    ev = [r for r in rows if r['IMAGE'] == im]
    assert len(ev) == 1, 'one image-level event, not one per object'
    assert ev[0]['image_is_positive'] is True
    assert ev[0]['n_missing_objects'] == 2
    assert ev[0]['no_detector_consulted'] is True
    assert all(b['geometry_author'] == 'human drawn'
               for b in ev[0]['missing_balls'])
    assert ev[0]['missing_balls'][0]['bbox_xywh'][2] == 4.2, 'tiny box kept'


def test_server_survives_restart_and_resumes(live):
    """State is rebuilt from the append-only log, so a restart loses nothing."""
    base, log, man = live
    im = man['sample'][2]['IMAGE']
    assert _post(base, {'IMAGE': im, 'answer': 'UNSURE'})[0] == 200
    again = SERVER.answers(log)
    assert again[im]['answer'] == 'UNSURE'
    assert _post(base, {'IMAGE': im, 'answer': 'NO_MISSING_BALL'})[0] == 200
    third = SERVER.answers(log)
    assert third[im]['answer'] == 'NO_MISSING_BALL', 'latest wins'
    assert len(third[im]['history']) == 2, 'the UNSURE event is still on record'


def test_server_serves_only_sampled_images(live):
    import urllib.error
    import urllib.request
    base, log, man = live
    for path, want in ((man['sample'][0]['IMAGE'], 200),
                       ('train/NOT_SAMPLED.jpg', 403)):
        try:
            with urllib.request.urlopen(base + '/img/' + path, timeout=20) as r:
                got = r.status
        except urllib.error.HTTPError as e:
            got = e.code
        assert got == want, f'{path} returned {got}'


# ------------------------------------------------- the page, actually driven


HARNESS = REPO / 'tests' / 'js' / 'ball_qa_round0.js'


@pytest.fixture(scope='module')
def driven(tmp_path_factory):
    """Execute the REAL served script and fire REAL events against it.

    Reading the Python cannot show that Enter finalises a positive, that three
    drawn balls post as three objects under one answer, or that a 4 px box
    drawn at 4x zoom survives the coordinate conversion. Those live in the page.
    """
    node = shutil.which('node')
    if not node:
        pytest.skip('node not installed')
    d = tmp_path_factory.mktemp('driven')
    (d / 'page.html').write_text(SERVER.PAGE, encoding='utf-8')
    st = SERVER.build_state()
    st['items'] = st['items'][:5]
    # Drive the page from an UNANSWERED state. The real log is now complete, so
    # feeding it back would start boot() at an already-answered image and the
    # navigation assertions would be measuring the fixture, not the page.
    for it in st['items']:
        it['answer'] = None
        it['missing'] = []
        it['history'] = []
    (d / 'state.json').write_text(json.dumps(st), encoding='utf-8')
    r = subprocess.run([node, str(HARNESS), str(d / 'page.html'),
                        str(d / 'state.json')],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)


def test_page_negative_answer_posts_and_advances(driven):
    assert driven['negative']['post']['answer'] == 'NO_MISSING_BALL'
    assert driven['negative']['advanced'] is True


def test_page_missing_ball_draws_before_it_posts(driven):
    """Pressing 2 must open drawing, not record a positive with no geometry."""
    assert driven['enters_draw_mode']['mode'] == 'draw'
    assert driven['enters_draw_mode']['posted_prematurely'] is False


def test_page_one_image_many_balls_is_one_positive(driven):
    m = driven['multi_object']
    assert m['answer'] == 'MISSING_BALL', 'a single image-level answer'
    assert m['objects_posted'] == 3, 'three independent objects'
    assert driven['drawn_count'] == 3


def test_page_tiny_box_survives_being_drawn_at_zoom(driven):
    t = driven['tiny_at_zoom']
    assert t['zoom'] == 4, 'the tile sweep magnifies'
    assert t['width_ok'] and t['height_ok'], (
        f'a ~4px ball drawn at {t["zoom"]}x came back as {t["drawn_image_px"]}; '
        f'geometry must be stored in image pixels, not screen pixels')


def test_page_undo_removes_only_the_last_ball(driven):
    assert driven['undo']['after'] == driven['undo']['before'] - 1


def test_page_tile_navigation_changes_nothing_but_the_view(driven):
    t = driven['tiles_are_inert']
    assert t['drawn_unchanged'] and t['image_unchanged']
    assert t['tiles_visited'] >= 2 and t['zoom'] > 1
    assert driven['fit_returns'] == {'zoom': 1, 'tile': -1}


def test_page_unsure_posts_its_own_answer(driven):
    assert driven['unsure']['post']['answer'] == 'UNSURE'


def test_page_navigation_cannot_leave_the_sample(driven):
    n = driven['navigation']
    assert n['stayed_in_sample'] is True
    assert 0 <= n['index'] < n['sample_size']
    assert 0 <= n['after_many_back'] < n['sample_size']
    assert driven['all_posted_images_in_sample'] is True


# ---------------------------------------------------------------- safety


def test_nothing_in_round0_writes_to_the_export():
    for mod in ('kb_ball_qa_sample', 'kb_ball_qa_server',
                'kb_ball_round0_report'):
        src = (REPO / 'tools' / f'{mod}.py').read_text(encoding='utf-8')
        for line in src.splitlines():
            if 'write_text' in line or 'open(' in line and "'a'" in line:
                assert 'repaired_export' not in line and 'EXPORT /' not in line


def test_export_files_are_only_ever_read():
    before = SAMPLE.population_fingerprint()
    SAMPLE.build()
    SERVER.build_state()
    assert SAMPLE.population_fingerprint() == before


def test_qa_events_do_not_disturb_box_level_resolution(tmp_path):
    """A Round-0 event must not be mistaken for a decision about an annotation."""
    log = tmp_path / 'd.json'
    rows = [
        {'mode': 'missed_role_manual', 'BOX_ID': 'train:1',
         'HUMAN_FINAL_CLASS': 'referee', 'recorded_utc': '2026-08-14T10:00:00Z'},
        {'mode': SERVER.QA_MODE, 'BOX_ID': 'BALLQA0:train/a.jpg',
         'IMAGE': 'train/a.jpg', 'answer': 'MISSING_BALL',
         'HUMAN_FINAL_CLASS': None, 'missing_balls': [{'bbox_xywh': [1, 1, 4, 4]}],
         'recorded_utc': '2026-08-14T11:00:00Z'},
    ]
    log.write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
    res = kb_decisions.resolve(log)
    assert res['train:1']['final_class'] == 'referee'
    qa = res['BALLQA0:train/a.jpg']
    assert qa['final_class'] is None and qa['disposition'] is None
    assert kb_decisions.missing_targets(log) == {}
    assert kb_decisions.geometry_repairs(log) == {}
    assert kb_decisions.ball_cases(log) == {}
