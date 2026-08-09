"""
Experiment C derived-data contract.

The dangerous failures here are silent: a mistransformed box teaches the model a
wrong location, a dropped ball in a "hard negative" teaches it that a real ball
is background, and a leaked validation frame invalidates every number the
project reports. All three are cheap to test and expensive to discover later.
"""

import json
from pathlib import Path

import pytest

from tools.build_derived_train import (BALL_ID, CLASSES, DERIVED_H, DERIVED_W,
                                       IMGSZ, MAX_ZOOM, MIN_RETAINED_AREA,
                                       TARGET_BAND, TEST_MATCHES, VAL_MATCHES,
                                       assert_train_only, plan_crop, to_yolo,
                                       transform)

ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / 'data' / 'derived_train'


class TestLeakageGuard:
    @pytest.mark.parametrize('match', sorted(VAL_MATCHES))
    def test_rejects_val_sources(self, match):
        with pytest.raises(SystemExit, match='held-out'):
            assert_train_only([{'match': match}])

    @pytest.mark.parametrize('match', sorted(TEST_MATCHES))
    def test_rejects_test_sources(self, match):
        with pytest.raises(SystemExit, match='held-out'):
            assert_train_only([{'match': match}])

    def test_allows_train_sources(self):
        assert_train_only([{'match': 'youth_3'}, {'match': 'chelsea_v_leeds_united'}])

    def test_rejects_when_one_of_many_leaks(self):
        with pytest.raises(SystemExit):
            assert_train_only([{'match': 'youth_3'}, {'match': 'women_1'}])

    def test_val_and_test_sets_are_disjoint(self):
        assert not (VAL_MATCHES & TEST_MATCHES)


class TestCropPlanning:
    def test_crop_width_hits_the_requested_ball_scale(self):
        box = [100, 100, 108, 108]              # 8 px native
        crop = plan_crop(box, 1920, 1080, 18.0, __import__('random').Random(0))
        assert crop is not None
        _, _, cw, _ = crop
        derived_w = 8 * (DERIVED_W / cw)         # px in the derived frame
        assert abs(derived_w * IMGSZ / DERIVED_W - 18.0) < 0.5

    def test_refuses_crops_needing_excessive_upscale(self):
        rng = __import__('random').Random(0)
        # a 2 px ball would need a tiny crop blown up far past the ceiling
        assert plan_crop([10, 10, 12, 12], 640, 360, 22.0, rng) is None

    def test_refuses_crop_larger_than_the_source(self):
        rng = __import__('random').Random(0)
        assert plan_crop([0, 0, 60, 60], 640, 360, 12.0, rng) is None

    def test_crop_stays_inside_the_image(self):
        rng = __import__('random').Random(1)
        for _ in range(50):
            crop = plan_crop([600, 330, 610, 340], 640, 360, 18.0, rng)
            if crop is None:
                continue
            x0, y0, cw, ch = crop
            assert x0 >= 0 and y0 >= 0
            assert x0 + cw <= 640 + 1e-6 and y0 + ch <= 360 + 1e-6

    def test_ball_is_not_always_centred(self):
        rng = __import__('random').Random(3)
        offs = set()
        for _ in range(30):
            c = plan_crop([500, 280, 510, 290], 1920, 1080, 18.0, rng)
            if c:
                offs.add(round((505 - c[0]) / c[2], 2))
        assert len(offs) > 5, 'ball position within the crop is not varying'

    def test_zoom_never_exceeds_the_ceiling(self):
        rng = __import__('random').Random(0)
        for _ in range(50):
            c = plan_crop([300, 200, 312, 212], 1024, 576, 20.0, rng)
            if c:
                assert DERIVED_W / c[2] <= MAX_ZOOM + 1e-9


class TestLabelTransform:
    CROP = (100.0, 50.0, 320.0, 180.0)          # 16:9, scale factor 2.0

    def test_ball_box_maps_correctly(self):
        out = transform([(BALL_ID, [200.0, 100.0, 210.0, 110.0])], self.CROP)
        assert len(out) == 1
        c, b = out[0]
        assert c == BALL_ID
        assert b == pytest.approx([200.0, 100.0, 220.0, 120.0])

    def test_human_box_maps_correctly(self):
        out = transform([(0, [110.0, 60.0, 150.0, 160.0])], self.CROP)
        c, b = out[0]
        assert c == 0
        assert b == pytest.approx([20.0, 20.0, 100.0, 220.0])

    def test_classes_are_never_remapped(self):
        boxes = [(i, [150.0 + i, 80.0, 190.0 + i, 160.0]) for i in range(4)]
        out = transform(boxes, self.CROP)
        assert sorted(c for c, _ in out) == [0, 1, 2, 3]

    def test_goalkeeper_stays_goalkeeper(self):
        out = transform([(1, [150.0, 80.0, 190.0, 160.0])], self.CROP)
        assert out[0][0] == 1
        assert CLASSES[out[0][0]] == 'goalkeeper'

    def test_boxes_outside_the_crop_are_dropped(self):
        assert transform([(0, [0.0, 0.0, 20.0, 20.0])], self.CROP) == []

    def test_heavily_truncated_boxes_are_dropped(self):
        # only a sliver survives the left edge -> below MIN_RETAINED_AREA
        out = transform([(0, [60.0, 60.0, 105.0, 160.0])], self.CROP)
        assert out == []

    def test_mildly_clipped_box_is_kept_and_clipped(self):
        out = transform([(0, [95.0, 60.0, 175.0, 160.0])], self.CROP)
        assert len(out) == 1
        b = out[0][1]
        assert b[0] == 0.0 and b[2] <= DERIVED_W

    def test_no_box_escapes_the_frame(self):
        boxes = [(0, [90.0, 40.0, 340.0, 260.0]), (3, [300.0, 170.0, 330.0, 200.0])]
        for _, b in transform(boxes, self.CROP):
            assert 0 <= b[0] < b[2] <= DERIVED_W
            assert 0 <= b[1] < b[3] <= DERIVED_H

    def test_keep_ball_false_drops_only_the_ball(self):
        boxes = [(0, [150.0, 80.0, 190.0, 160.0]), (BALL_ID, [200.0, 100.0, 210.0, 110.0])]
        out = transform(boxes, self.CROP, keep_ball=False)
        assert [c for c, _ in out] == [0]

    def test_yolo_lines_are_normalised_and_valid(self):
        txt = to_yolo(transform([(BALL_ID, [200.0, 100.0, 210.0, 110.0])], self.CROP))
        parts = txt.split()
        assert int(parts[0]) == BALL_ID
        assert all(0.0 <= float(v) <= 1.0 for v in parts[1:])


@pytest.mark.skipif(not (DERIVED / 'manifest.json').exists(),
                    reason='derived package not built')
class TestBuiltPackage:
    @staticmethod
    def _man():
        return json.loads((DERIVED / 'manifest.json').read_text(encoding='utf-8'))

    def test_marked_train_only(self):
        assert self._man()['split'] == 'TRAIN_ONLY'
        assert (DERIVED / 'SPLIT').exists()

    def test_no_held_out_source(self):
        srcs = {i['source_match'] for i in self._man()['items']}
        assert not (srcs & (VAL_MATCHES | TEST_MATCHES))

    def test_no_duplicate_filenames(self):
        files = [i['file'] for i in self._man()['items']]
        assert len(files) == len(set(files))

    def test_provenance_always_present(self):
        for i in self._man()['items']:
            assert i['source_image'] and i['source_match'] and i['crop']
            assert i['kind'] in ('positive_scale_context', 'hard_negative')

    def test_positives_land_in_the_target_band(self):
        w = [i['derived_ball_width_960'] for i in self._man()['items']
             if i['kind'] == 'positive_scale_context']
        assert w, 'no positives'
        assert all(TARGET_BAND[0] - 1 <= v <= TARGET_BAND[1] + 1 for v in w)

    def test_every_label_file_is_valid(self):
        for i in self._man()['items']:
            lp = DERIVED / 'labels' / f"{Path(i['file']).stem}.txt"
            assert lp.exists()
            for line in lp.read_text(encoding='utf-8').splitlines():
                if not line.strip():
                    continue
                q = line.split()
                assert len(q) == 5
                assert int(q[0]) in range(len(CLASSES))
                assert all(0.0 <= float(v) <= 1.0 for v in q[1:])
                assert float(q[3]) > 0 and float(q[4]) > 0

    def test_images_match_derived_geometry(self):
        from PIL import Image
        for i in self._man()['items'][::5]:
            assert Image.open(DERIVED / 'images' / i['file']).size == (DERIVED_W, DERIVED_H)

    def test_hard_negatives_declare_whether_a_real_ball_survived(self):
        """A hard negative that silently drops a real ball is a mislabelled
        image. Every negative must state this, and any that does contain a ball
        must actually carry the ball label."""
        for i in self._man()['items']:
            if i['kind'] != 'hard_negative':
                continue
            assert 'contains_real_ball' in i
            lp = DERIVED / 'labels' / f"{Path(i['file']).stem}.txt"
            has = any(line.split()[0] == str(BALL_ID)
                      for line in lp.read_text(encoding='utf-8').splitlines() if line.strip())
            assert has == i['contains_real_ball']

    def test_source_diversity_is_capped(self):
        from collections import Counter
        for kind, cap in (('positive_scale_context', 6), ('hard_negative', 4)):
            c = Counter(i['source_match'] for i in self._man()['items']
                        if i['kind'] == kind)
            assert not c or max(c.values()) <= cap

    def test_generation_is_deterministic(self):
        """Same seed, same plan -- otherwise the audit describes a package that
        cannot be regenerated."""
        from tools.build_derived_train import main
        import sys as _s
        argv = _s.argv
        _s.argv = ['x', '--dry-run', '--skip-negatives']
        try:
            a = main(); b = main()
        finally:
            _s.argv = argv
        assert [(i['rec']['path'].name, tuple(round(v, 4) for v in i['crop'])) for i in a] == \
               [(i['rec']['path'].name, tuple(round(v, 4) for v in i['crop'])) for i in b]
