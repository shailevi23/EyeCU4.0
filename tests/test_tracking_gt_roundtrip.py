"""
CVAT -> canonical EyeCU GT -> MOT round trip, on a synthetic fixture.

This runs before any human annotates anything, because the expensive failure is
not a crash: it is a silent off-by-one or an identity remap that nobody notices
until the tracker bake-off produces numbers that mean nothing. The fixture is
deliberately small and hand-checkable -- two humans, different roles, a short
outside interval, and one reappearance -- so every assertion below can be
verified by reading the XML.

The fixture is synthetic geometry with invented identities. It is a test of the
conversion path only, and is never GT for any sequence.
"""

import json
import shutil
from collections import defaultdict
from pathlib import Path

import pytest

from tools.import_tracking_gt_cvat import parse_cvat_video
from tools.validate_tracking_gt import validate_gt_content

ROOT = Path(__file__).resolve().parents[1] / 'data' / 'tracking_val_gt'

pytestmark = pytest.mark.skipif(not (ROOT / 'manifest.json').exists(),
                                reason='identity GT package not built')

# CVAT frames are 0-based. Track 0 is visible 0-4, outside for 5-6, and
# reappears 7-9 -- the same identity, as a real occlusion must be annotated.
# Track 1 is visible throughout with a different role.
FIXTURE = """<?xml version="1.0" encoding="utf-8"?>
<annotations>
  <version>1.1</version>
  <meta><task><name>synthetic</name><size>10</size></task></meta>
  <track id="0" label="player">
    <box frame="0" outside="0" occluded="0" keyframe="1" xtl="100" ytl="200" xbr="140" ybr="300"/>
    <box frame="4" outside="0" occluded="0" keyframe="1" xtl="140" ytl="200" xbr="180" ybr="300"/>
    <box frame="5" outside="1" occluded="0" keyframe="1" xtl="150" ytl="200" xbr="190" ybr="300"/>
    <box frame="7" outside="0" occluded="0" keyframe="1" xtl="170" ytl="200" xbr="210" ybr="300"/>
    <box frame="9" outside="0" occluded="0" keyframe="1" xtl="190" ytl="200" xbr="230" ybr="300"/>
  </track>
  <track id="1" label="goalkeeper">
    <box frame="0" outside="0" occluded="0" keyframe="1" xtl="500" ytl="210" xbr="545" ybr="320"/>
    <box frame="9" outside="0" occluded="1" keyframe="1" xtl="545" ytl="210" xbr="590" ybr="320"/>
  </track>
</annotations>
"""

# Verbatim tracks from a REAL app.cvat.ai export (job 4344203, women_1_239
# frames 0-9), the one the smoke test passed 25/25 on.
# It differs from the fixture above in ways worth pinning down:
#   * every frame is written out, including interpolated ones (keyframe="0"),
#     not just the keyframes
#   * tracks carry source="manual", boxes carry z_order, and <box> is an open
#     element with whitespace inside rather than self-closing
#   * the frame after an outside marker is simply absent, not marked
# All of that must decode to the same thing as the sparse fixture.
REAL_CVAT = """<?xml version="1.0" encoding="utf-8"?>
<annotations>
  <version>1.1</version>
  <meta>
    <job><id>4344203</id><size>10</size><mode>interpolation</mode>
      <start_frame>0</start_frame><stop_frame>9</stop_frame>
    </job>
    <original_size><width>640</width><height>360</height></original_size>
  </meta>
  <track id="0" label="player" source="manual">
    <box frame="0" keyframe="1" outside="0" occluded="0" xtl="206.22" ytl="148.71" xbr="233.82" ybr="206.68" z_order="0">
    </box>
    <box frame="1" keyframe="0" outside="0" occluded="0" xtl="206.22" ytl="148.71" xbr="233.82" ybr="206.68" z_order="0">
    </box>
    <box frame="2" keyframe="0" outside="0" occluded="0" xtl="206.22" ytl="148.71" xbr="233.82" ybr="206.68" z_order="0">
    </box>
    <box frame="3" keyframe="0" outside="0" occluded="0" xtl="206.22" ytl="148.71" xbr="233.82" ybr="206.68" z_order="0">
    </box>
    <box frame="4" keyframe="1" outside="0" occluded="0" xtl="206.22" ytl="148.71" xbr="233.82" ybr="206.68" z_order="0">
    </box>
    <box frame="5" keyframe="1" outside="1" occluded="0" xtl="206.22" ytl="148.71" xbr="233.82" ybr="206.68" z_order="0">
    </box>
    <box frame="7" keyframe="1" outside="0" occluded="0" xtl="206.90" ytl="148.63" xbr="232.30" ybr="208.84" z_order="0">
    </box>
    <box frame="8" keyframe="1" outside="0" occluded="0" xtl="204.84" ytl="148.63" xbr="230.24" ybr="208.84" z_order="0">
    </box>
    <box frame="9" keyframe="0" outside="0" occluded="0" xtl="204.84" ytl="148.63" xbr="230.24" ybr="208.84" z_order="0">
    </box>
  </track>
  <track id="1" label="referee" source="manual">
    <box frame="0" keyframe="1" outside="0" occluded="0" xtl="542.26" ytl="142.97" xbr="571.48" ybr="200.37" z_order="0">
    </box>
    <box frame="1" keyframe="1" outside="0" occluded="0" xtl="543.83" ytl="142.97" xbr="573.04" ybr="200.37" z_order="0">
    </box>
    <box frame="2" keyframe="1" outside="0" occluded="0" xtl="546.44" ytl="145.06" xbr="575.65" ybr="202.45" z_order="0">
    </box>
    <box frame="3" keyframe="1" outside="0" occluded="0" xtl="548.00" ytl="145.58" xbr="577.22" ybr="202.98" z_order="0">
    </box>
    <box frame="4" keyframe="1" outside="0" occluded="0" xtl="552.18" ytl="146.63" xbr="581.39" ybr="204.02" z_order="0">
    </box>
    <box frame="5" keyframe="1" outside="0" occluded="0" xtl="554.26" ytl="146.63" xbr="583.48" ybr="204.02" z_order="0">
    </box>
    <box frame="6" keyframe="1" outside="0" occluded="0" xtl="554.26" ytl="145.58" xbr="583.48" ybr="202.98" z_order="0">
    </box>
    <box frame="7" keyframe="1" outside="0" occluded="0" xtl="555.83" ytl="144.02" xbr="585.05" ybr="201.41" z_order="0">
    </box>
    <box frame="8" keyframe="1" outside="0" occluded="0" xtl="557.39" ytl="143.50" xbr="586.61" ybr="200.89" z_order="0">
    </box>
    <box frame="9" keyframe="1" outside="0" occluded="0" xtl="558.96" ytl="142.97" xbr="588.18" ybr="200.36" z_order="0">
    </box>
  </track>
</annotations>
"""

SHAPES_ONLY = """<?xml version="1.0" encoding="utf-8"?>
<annotations>
  <version>1.1</version>
  <image id="0" name="000001.jpg" width="1280" height="720">
    <box label="player" xtl="100" ytl="200" xbr="140" ybr="300"/>
  </image>
</annotations>
"""


@pytest.fixture
def parsed(tmp_path):
    p = tmp_path / 'seq.xml'
    p.write_text(FIXTURE, encoding='utf-8')
    boxes, roles, warn = parse_cvat_video(p, n_frames=300)
    return boxes, roles, warn


class TestCvatImport:
    def test_identities_are_persistent_and_positive(self, parsed):
        boxes, roles, _ = parsed
        assert set(roles) == {1, 2}, 'CVAT id 0 must become identity 1'
        assert {b['id'] for b in boxes} == {1, 2}

    def test_frames_are_one_based(self, parsed):
        boxes, _, _ = parsed
        assert min(b['frame'] for b in boxes) == 1, 'CVAT frame 0 -> package 1'
        assert max(b['frame'] for b in boxes) == 10

    def test_outside_interval_produces_no_boxes(self, parsed):
        boxes, _, _ = parsed
        f1 = sorted(b['frame'] for b in boxes if b['id'] == 1)
        assert f1 == [1, 2, 3, 4, 5, 8, 9, 10], f1
        assert 6 not in f1 and 7 not in f1, 'no box may be invented while hidden'

    def test_reappearance_keeps_the_same_identity(self, parsed):
        boxes, _, _ = parsed
        after = [b for b in boxes if b['id'] == 1 and b['frame'] >= 8]
        assert len(after) == 3
        assert {b['role'] for b in after} == {'player'}

    def test_roles_come_from_track_labels(self, parsed):
        _, roles, _ = parsed
        assert roles == {1: 'player', 2: 'goalkeeper'}

    def test_interpolation_between_keyframes(self, parsed):
        """CVAT interpolates linearly; frame 2 (CVAT 1) is a quarter of the way."""
        boxes, _, _ = parsed
        b = next(b for b in boxes if b['id'] == 1 and b['frame'] == 2)
        assert b['bbox'] == [110.0, 200.0, 150.0, 300.0]

    def test_last_keyframe_is_emitted(self, parsed):
        boxes, _, _ = parsed
        b = next(b for b in boxes if b['id'] == 1 and b['frame'] == 10)
        assert b['bbox'] == [190.0, 200.0, 230.0, 300.0]

    def test_no_duplicate_identity_within_a_frame(self, parsed):
        boxes, _, _ = parsed
        seen = set()
        for b in boxes:
            key = (b['frame'], b['id'])
            assert key not in seen, key
            seen.add(key)

    def test_shape_only_export_is_rejected(self, tmp_path):
        p = tmp_path / 'seq.xml'
        p.write_text(SHAPES_ONLY, encoding='utf-8')
        with pytest.raises(SystemExit, match='shape-only'):
            parse_cvat_video(p, n_frames=300)

    def test_unknown_label_is_rejected(self, tmp_path):
        p = tmp_path / 'seq.xml'
        p.write_text(FIXTURE.replace('label="player"', 'label="coach"'),
                     encoding='utf-8')
        with pytest.raises(SystemExit, match='coach'):
            parse_cvat_video(p, n_frames=300)

    def test_frames_beyond_the_sequence_are_dropped_with_a_warning(self, tmp_path):
        p = tmp_path / 'seq.xml'
        p.write_text(FIXTURE, encoding='utf-8')
        boxes, _, warn = parse_cvat_video(p, n_frames=5)
        assert warn and any('outside 1..5' in w for w in warn)
        assert max(b['frame'] for b in boxes) <= 5


class TestRealCvatExportShape:
    """
    Pinned against an actual app.cvat.ai export, not against what we assumed.

    The synthetic fixture writes only keyframes; real CVAT writes every frame.
    Both must decode identically, or the parser is agreeing with our
    imagination rather than with CVAT.
    """

    @pytest.fixture
    def real(self, tmp_path):
        p = tmp_path / 'real.xml'
        p.write_text(REAL_CVAT, encoding='utf-8')
        return parse_cvat_video(p, n_frames=300)

    def test_dense_per_frame_boxes_decode(self, real):
        boxes, roles, warn = real
        assert roles == {1: 'player', 2: 'referee'}
        assert warn == []
        assert len(boxes) == 18

    def test_outside_marker_and_absent_frame_both_yield_no_box(self, real):
        """Frame 5 is marked outside; frame 6 is simply missing. Both drop."""
        boxes, _, _ = real
        frames = sorted(b['frame'] for b in boxes if b['id'] == 1)
        assert frames == [1, 2, 3, 4, 5, 8, 9, 10], frames

    def test_the_other_track_is_unaffected_by_the_gap(self, real):
        """One track going outside must not disturb the other."""
        boxes, _, _ = real
        assert sorted(b['frame'] for b in boxes if b['id'] == 2) == list(range(1, 11))

    def test_two_identities_stay_separate(self, real):
        boxes, _, _ = real
        per_frame = defaultdict(list)
        for b in boxes:
            per_frame[b['frame']].append(b['id'])
        for f, ids in per_frame.items():
            assert len(ids) == len(set(ids)), f
        roles_by_id = defaultdict(set)
        for b in boxes:
            roles_by_id[b['id']].add(b['role'])
        assert roles_by_id == {1: {'player'}, 2: {'referee'}}

    def test_reappearance_keeps_the_identity(self, real):
        boxes, _, _ = real
        late = [b for b in boxes if b['frame'] >= 8 and b['role'] == 'player']
        assert {b['id'] for b in late} == {1}

    def test_geometry_survives_verbatim(self, real):
        boxes, _, _ = real
        first = next(b for b in boxes if b['frame'] == 1 and b['id'] == 1)
        assert first['bbox'] == [206.22, 148.71, 233.82, 206.68]
        last = next(b for b in boxes if b['frame'] == 10 and b['id'] == 1)
        assert last['bbox'] == [204.84, 148.63, 230.24, 208.84]
        ref = next(b for b in boxes if b['frame'] == 1 and b['id'] == 2)
        assert ref['bbox'] == [542.26, 142.97, 571.48, 200.37]

    def test_matches_the_synthetic_fixture_frame_pattern(self, tmp_path, real):
        """The two fixtures disagree on encoding, agree on meaning."""
        p = tmp_path / 'syn.xml'
        p.write_text(FIXTURE, encoding='utf-8')
        syn, syn_roles, _ = parse_cvat_video(p, n_frames=300)
        real_boxes, real_roles, _ = real
        assert set(real_roles) == set(syn_roles)
        for ident in (1, 2):
            assert sorted(b['frame'] for b in real_boxes if b['id'] == ident) == \
                sorted(b['frame'] for b in syn if b['id'] == ident)


class TestOcclusionIsCarriedAndInert:
    """
    The annotator's occluded marks must survive into canonical GT, and must
    change nothing else.

    Recovering metadata from a re-import is only safe if the re-import is
    otherwise a no-op. The test flips every occlusion flag in the same XML and
    asserts that frame, identity, bbox, role and ORDER are byte-identical --
    so a later re-import to recover metadata cannot quietly move a box.
    """

    def _parse(self, tmp_path, xml, name):
        p = tmp_path / name
        p.write_text(xml, encoding='utf-8')
        return parse_cvat_video(p, n_frames=300)

    def test_every_box_carries_a_boolean_occluded(self, tmp_path):
        boxes, _, _ = self._parse(tmp_path, REAL_CVAT, 'a.xml')
        assert boxes
        for b in boxes:
            assert isinstance(b['occluded'], bool), b

    def test_occluded_marks_are_recovered(self, tmp_path):
        marked = REAL_CVAT.replace(
            'frame="2" keyframe="0" outside="0" occluded="0"',
            'frame="2" keyframe="0" outside="0" occluded="1"')
        boxes, _, _ = self._parse(tmp_path, marked, 'b.xml')
        occ = [b for b in boxes if b['occluded']]
        assert len(occ) == 1 and occ[0]['frame'] == 3, occ

    def test_flipping_occlusion_changes_nothing_else(self, tmp_path):
        plain, _, _ = self._parse(tmp_path, REAL_CVAT, 'c.xml')
        flipped, _, _ = self._parse(
            tmp_path, REAL_CVAT.replace('occluded="0"', 'occluded="1"'), 'd.xml')
        assert len(plain) == len(flipped)
        strip = lambda bs: [{k: v for k, v in b.items() if k != 'occluded'}
                            for b in bs]
        assert strip(plain) == strip(flipped), 'occlusion must be inert'
        assert all(b['occluded'] for b in flipped)
        assert not any(b['occluded'] for b in plain)

    def test_occlusion_holds_across_interpolated_frames(self, tmp_path):
        """An interpolated frame inherits occlusion from the shape before it."""
        marked = FIXTURE.replace(
            '<box frame="0" outside="0" occluded="0" keyframe="1" xtl="100"',
            '<box frame="0" outside="0" occluded="1" keyframe="1" xtl="100"')
        boxes, _, _ = self._parse(tmp_path, marked, 'e.xml')
        got = {b['frame']: b['occluded'] for b in boxes if b['id'] == 1}
        assert got[1] is True and got[2] is True and got[4] is True
        assert got[8] is False, 'the next keyframe ends the occluded run'


def _import_into_package(tmp_path):
    """Run the real importer against a one-sequence copy of the real package."""
    from tools.import_tracking_gt_cvat import main as import_main

    dst = tmp_path / 'gt'
    shutil.copytree(ROOT, dst, ignore=shutil.ignore_patterns('img1', 'qc'))
    man = json.loads((dst / 'manifest.json').read_text(encoding='utf-8'))
    man['sequences'] = man['sequences'][:1]
    (dst / 'manifest.json').write_text(json.dumps(man), encoding='utf-8')
    seq = man['sequences'][0]['sequence']

    exports = tmp_path / 'cvat'
    exports.mkdir()
    (exports / f'{seq}.xml').write_text(FIXTURE, encoding='utf-8')

    import sys
    argv = sys.argv
    sys.argv = ['import_tracking_gt_cvat.py', '--root', str(dst),
                '--exports', str(exports)]
    try:
        import_main()
    finally:
        sys.argv = argv
    return dst, json.loads((dst / 'manifest.json').read_text(encoding='utf-8'))


class TestRoundTrip:
    def test_import_writes_canonical_json_and_role_sidecar(self, tmp_path):
        dst, man = _import_into_package(tmp_path)
        s = man['sequences'][0]
        ann = json.loads((dst / s['annotation_file_expected']).read_text(encoding='utf-8'))
        assert ann['frame_numbering'] == '1-based'
        assert ann['source'] == 'CVAT for video 1.1'
        roles = json.loads((dst / s['roles_expected']).read_text(encoding='utf-8'))
        assert roles['identity_roles'] == {'1': 'player', '2': 'goalkeeper'}
        assert 'CVAT track labels' in roles['generated_from']

    def test_import_lands_in_pending_qc_not_verified(self, tmp_path):
        _, man = _import_into_package(tmp_path)
        assert man['identity_gt_status'] == 'ANNOTATED_PENDING_QC'

    def test_imported_gt_satisfies_the_content_rules(self, tmp_path):
        dst, _ = _import_into_package(tmp_path)
        errors, n = validate_gt_content(dst)
        assert n > 0 and errors == [], errors[:5]

    def test_mot_export_after_qc_confirmation(self, tmp_path):
        from tools.confirm_tracking_gt_qc import promote_to_verified
        from tools.export_tracking_gt_mot import export

        dst, man = _import_into_package(tmp_path)
        promote_to_verified(dst, man, reviewer='roundtrip-test')
        export(dst, tmp_path / 'mot')

        seq = man['sequences'][0]['sequence']
        gt = tmp_path / 'mot' / 'EyeCU-val' / seq / 'gt' / 'gt.txt'
        rows = [r.split(',') for r in gt.read_text(encoding='utf-8').splitlines()]

        assert len(rows) == 8 + 10, 'track 1 is hidden for two frames'
        frames = sorted({int(r[0]) for r in rows})
        assert frames == list(range(1, 11)), frames
        assert {int(r[1]) for r in rows} == {1, 2}, 'identities must survive'
        assert all(int(r[6]) == 1 for r in rows), 'TrackEval drops conf == 0 GT'
        assert all(int(r[7]) == 1 for r in rows), 'class must be 1 (pedestrian)'

        # geometry: CVAT xtl/ytl/xbr/ybr -> MOT x,y,w,h
        first = next(r for r in rows if int(r[0]) == 1 and int(r[1]) == 1)
        assert [float(v) for v in first[2:6]] == [100.0, 200.0, 40.0, 100.0]

        # no duplicate identity within a frame, no ball
        keys = [(r[0], r[1]) for r in rows]
        assert len(keys) == len(set(keys))
        assert 'ball' not in gt.read_text(encoding='utf-8')
