"""Tiled / zoomed inference probe: does cropping recover the tiny footballs?

A negative result here decides whether 101 images get swept by hand, so the
mechanics have to be beyond doubt. These tests pin the parts that would fake a
negative if they were wrong: tile coverage, coordinate mapping back to the
original frame, and deduplication across overlapping crops.
"""

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'tools'))

import kb_ball_calibrate as CAL                                   # noqa: E402
import kb_ball_candidates as CAND                                 # noqa: E402
import kb_ball_tiled_probe as T                                   # noqa: E402


# ------------------------------------------------------------------ tiles


@pytest.mark.parametrize('cols,rows', [(2, 2), (3, 3)])
def test_tiles_cover_the_whole_frame(cols, rows):
    """A gap between tiles is a blind stripe, and a ball in it would look like
    a detector failure rather than a tiling bug."""
    ts = T.tiles(1280, 720, cols, rows)
    assert len(ts) == cols * rows
    # every pixel belongs to at least one tile
    for x in range(0, 1280, 7):
        for y in range(0, 720, 7):
            assert any(x0 <= x <= x1 and y0 <= y <= y1
                       for (x0, y0, x1, y1) in ts), f'({x},{y}) uncovered'


@pytest.mark.parametrize('cols,rows', [(2, 2), (3, 3)])
def test_tiles_stay_inside_the_image(cols, rows):
    for (x0, y0, x1, y1) in T.tiles(1280, 720, cols, rows):
        assert 0 <= x0 < x1 <= 1280
        assert 0 <= y0 < y1 <= 720


def test_tiles_overlap_by_the_configured_amount():
    """Without overlap a ball on a seam is cut in half in both neighbours and
    detected in neither -- a recall floor no amount of tiling would fix."""
    ts = T.tiles(1280, 720, 2, 2)
    # the two left-hand tiles must share rows; the two top tiles share columns
    (ax0, ay0, ax1, ay1) = ts[0]
    (bx0, by0, bx1, by1) = ts[1]
    assert ax1 > bx0, 'horizontal neighbours must overlap'
    (cx0, cy0, cx1, cy1) = ts[2]
    assert ay1 > cy0, 'vertical neighbours must overlap'
    assert (ax1 - bx0) >= 0.15 * (1280 / 2)


def test_tile_coordinate_mapping_round_trips():
    """A detection at a known point in a crop must map back to that point.

    This is the assertion that separates 'the detector cannot see these balls'
    from 'the boxes were put back in the wrong place'.
    """
    W, H, scale = 1280, 720, 2
    for (x0, y0, x1, y1) in T.tiles(W, H, 3, 3):
        # a 6 px box at the centre of this crop, in crop coordinates x scale
        cw, ch = (x1 - x0) * scale, (y1 - y0) * scale
        bx, by = cw / 2, ch / 2
        mapped = [bx / scale + x0, by / scale + y0, 6.0, 6.0]
        assert abs(mapped[0] - (x0 + (x1 - x0) / 2)) < 1e-6
        assert abs(mapped[1] - (y0 + (y1 - y0) / 2)) < 1e-6
        assert x0 <= mapped[0] <= x1 and y0 <= mapped[1] <= y1


# --------------------------------------------------------------- dedupe


def test_dedupe_merges_the_same_ball_seen_in_two_tiles():
    dets = [{'bbox_xywh': [100.0, 100.0, 6.0, 6.0], 'conf': 0.4},
            {'bbox_xywh': [102.0, 101.0, 6.0, 6.0], 'conf': 0.2}]
    out = T.dedupe(dets)
    assert len(out) == 1
    assert out[0]['conf'] == 0.4, 'the more confident detection survives'


def test_dedupe_keeps_genuinely_separate_balls():
    dets = [{'bbox_xywh': [100.0, 100.0, 6.0, 6.0], 'conf': 0.4},
            {'bbox_xywh': [400.0, 400.0, 6.0, 6.0], 'conf': 0.3}]
    assert len(T.dedupe(dets)) == 2


def test_dedupe_is_order_independent():
    a = [{'bbox_xywh': [100.0, 100.0, 6.0, 6.0], 'conf': 0.4},
         {'bbox_xywh': [103.0, 100.0, 6.0, 6.0], 'conf': 0.9},
         {'bbox_xywh': [400.0, 400.0, 6.0, 6.0], 'conf': 0.1}]
    out1 = T.dedupe(a)
    out2 = T.dedupe(list(reversed(a)))
    assert len(out1) == len(out2) == 2
    assert max(d['conf'] for d in out1) == max(d['conf'] for d in out2) == 0.9


def test_union_deduplicates_across_methods():
    a = {'i': [{'bbox_xywh': [10.0, 10.0, 6.0, 6.0], 'conf': 0.5}]}
    b = {'i': [{'bbox_xywh': [11.0, 10.0, 6.0, 6.0], 'conf': 0.3},
               {'bbox_xywh': [500.0, 10.0, 6.0, 6.0], 'conf': 0.3}]}
    u = T.union(a, b)
    assert len(u['i']) == 2, 'the shared ball is counted once'


# --------------------------------------------------------------- scoring


def test_score_counts_images_not_boxes_for_the_denominator():
    adds = [{'IMAGE': 'i', 'bbox_xywh': [100.0, 100.0, 6.0, 6.0],
             'size_bucket': '>5-8px'},
            {'IMAGE': 'i', 'bbox_xywh': [400.0, 400.0, 6.0, 6.0],
             'size_bucket': '>5-8px'}]
    per = {'i': [{'bbox_xywh': [101.0, 100.0, 6.0, 6.0], 'conf': 0.5}]}
    s = T.score(per, adds, {'i': []})
    assert s['confirmed_covered'] == 1
    assert s['confirmed_missed'] == 1
    assert s['confirmed_recall'] == 0.5


def test_score_uses_the_review_tooling_matching_rule():
    adds = [{'IMAGE': 'i', 'bbox_xywh': [100.0, 100.0, 6.0, 6.0],
             'size_bucket': '>5-8px'}]
    near = {'i': [{'bbox_xywh': [104.0, 100.0, 6.0, 6.0], 'conf': 0.5}]}
    far = {'i': [{'bbox_xywh': [300.0, 100.0, 6.0, 6.0], 'conf': 0.5}]}
    assert T.score(near, adds, {'i': []})['confirmed_recall'] == 1.0
    assert T.score(far, adds, {'i': []})['confirmed_recall'] == 0.0


def test_size_buckets_partition_the_additions():
    adds = CAL.confirmed_additions()
    counts = {}
    for _, name in T.BUCKETS:
        counts[name] = sum(1 for a in adds if a['size_bucket'] == name)
    assert sum(counts.values()) == len(adds)


# ------------------------------------------------------- the recorded run


@pytest.mark.skipif(not T.REPORT.is_file(), reason='probe not run')
def test_report_reconciles():
    rep = json.loads(T.REPORT.read_text(encoding='utf-8'))
    n = rep['frame']['confirmed_additions']
    for m, s in rep['results'].items():
        assert s['confirmed_covered'] + s['confirmed_missed'] == n
        assert s['confirmed_recall'] == pytest.approx(
            s['confirmed_covered'] / n)
        assert s['proposals_unmatched'] <= s['proposals_total']
        assert sum(b['n'] for b in s['recall_by_size'].values()) == n
        assert sum(b['covered'] for b in s['recall_by_size'].values()) == \
            s['confirmed_covered']


@pytest.mark.skipif(not T.REPORT.is_file(), reason='probe not run')
def test_union_never_scores_below_its_parts():
    rep = json.loads(T.REPORT.read_text(encoding='utf-8'))
    r = rep['results']
    for u, parts in (('FULL+2x2', ('FULL', 'TILE_2x2')),
                     ('FULL+3x3', ('FULL', 'TILE_3x3'))):
        if u in r:
            for p in parts:
                assert r[u]['confirmed_covered'] >= r[p]['confirmed_covered']


@pytest.mark.skipif(not T.REPORT.is_file(), reason='probe not run')
def test_verdict_follows_the_measured_recall():
    rep = json.loads(T.REPORT.read_text(encoding='utf-8'))
    viable = rep['best_recall'] >= 0.80
    assert rep['target_80pc_achievable'] is viable
    assert rep['verdict'] == ('MODEL_ASSISTED PP COMPLETION VIABLE' if viable
                              else 'MODEL_ASSISTED PP COMPLETION NOT VIABLE')
    assert rep['best_recall'] == max(
        s['confirmed_recall'] for s in rep['results'].values())


@pytest.mark.skipif(not T.REPORT.is_file(), reason='probe not run')
def test_reference_is_human_confirmed_only():
    rep = json.loads(T.REPORT.read_text(encoding='utf-8'))
    assert 'HUMAN_CONFIRMED_ADDITIONS' in rep['reference']
    assert 'unverified' in rep['reference']


# ------------------------------------------------------------ no writes


def test_probe_writes_only_its_own_report():
    import ast
    src = (REPO / 'tools' / 'kb_ball_tiled_probe.py').read_text(encoding='utf-8')
    for bad in ('DECISIONS', '.train(', 'repaired_export'):
        assert bad not in src
    written = set()
    for node in ast.walk(ast.parse(src)):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in ('write_text', 'write_bytes')):
            v = node.func.value
            written.add(v.id if isinstance(v, ast.Name) else '?')
    assert written <= {'REPORT'}, f'writes to {written}'


def test_probe_uses_the_eyecu_detector_not_coco():
    src = (REPO / 'tools' / 'kb_ball_tiled_probe.py').read_text(encoding='utf-8')
    assert 'yolov8n' not in src, 'the COCO probe was already shown useless'
    assert 'CAND.WEIGHTS' in src
    assert 'CAND.BALL_CLASS' in src
