"""
M2 -- the pitch calibration contract, held in place by test.

The whole point of this milestone is that a metric claim is bad by default
and only becomes acceptable inside a specific, frozen, validated camera
segment. Every test here is either proving the geometry is right on a
synthetic, ground-truth-known case, or proving the module refuses to answer
outside that guarantee.
"""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from trackers.pitch_calibration import (CalibratedSegment, CalibrationStore,
                                        ground_point, load_segment,
                                        short_window_displacement_and_speed)

REPO = Path(__file__).resolve().parents[1]
CAL_DIR = REPO / 'experiments' / 'records' / 'experiment_M2' / 'calibration'
BAYERN_ARTIFACT = CAL_DIR / 'bayern_calibration.json'
WOMEN_ARTIFACT = CAL_DIR / 'women_calibration.json'
BAYERN_RAW = CAL_DIR / 'bayern_correspondences_raw.json'
M2_1_STATUS = CAL_DIR / 'M2_1_CALIBRATION_STATUS.json'


def perpendicular_distance(point, line_a, line_b):
    """Distance from `point` to the infinite line through `line_a`/`line_b`,
    all as (x, y) pairs. Zero iff the three are exactly collinear."""
    point, line_a, line_b = (np.asarray(p, dtype=np.float64) for p in (point, line_a, line_b))
    d = line_b - line_a
    d = d / np.linalg.norm(d)
    n = np.array([d[1], -d[0]])
    return float(abs(n @ (point - line_a)))

pytestmark = pytest.mark.skipif(not BAYERN_ARTIFACT.exists(),
                                reason='M2 calibration artifacts not built')


# --------------------------------------------------------------------------
# synthetic homography: exact ground truth, independent of any real footage
# --------------------------------------------------------------------------

def make_synthetic_segment(sequence='synthetic_seq', frame_range=(1, 10)):
    """A known planar homography: pitch (X,Y) metres -> image pixels, built
    from a hand-specified projective transform, then inverted to get the
    image->pitch H this module expects. Because we construct BOTH sides
    ourselves, every projected point has an exactly known correct answer.
    """
    pitch_pts = np.array([[9.15, 0], [-9.15, 0], [0, 9.15], [0, -9.15]], dtype=np.float64)
    # a plausible perspective mapping pitch metres -> synthetic pixels
    image_pts = np.array([[467.4, 138.5], [466.1, 209.1],
                          [309.0, 172.3], [575.9, 174.8]], dtype=np.float64)
    H_pitch_to_image, _ = cv2.findHomography(pitch_pts, image_pts, method=0)
    H_image_to_pitch = np.linalg.inv(H_pitch_to_image)
    start, end = frame_range
    seg = CalibratedSegment(sequence=sequence, frame_start_1based=start,
                            frame_end_1based=end, H_image_to_pitch=H_image_to_pitch,
                            artifact_sha256='synthetic', source_path='<memory>')
    return seg, H_pitch_to_image, pitch_pts, image_pts


def project_pitch_to_image(H_pitch_to_image, pitch_xy):
    p = H_pitch_to_image @ np.array([pitch_xy[0], pitch_xy[1], 1.0])
    return (p[0] / p[2], p[1] / p[2])


class TestSyntheticHomography:
    def test_known_points_map_back_correctly(self):
        seg, H_p2i, pitch_pts, image_pts = make_synthetic_segment()
        for pitch_xy, image_xy in zip(pitch_pts, image_pts):
            got = seg.image_to_pitch(tuple(image_xy))
            assert got[0] == pytest.approx(pitch_xy[0], abs=1e-3)
            assert got[1] == pytest.approx(pitch_xy[1], abs=1e-3)

    def test_arbitrary_point_round_trips(self):
        seg, H_p2i, *_ = make_synthetic_segment()
        for pitch_xy in [(3.0, -4.0), (-7.5, 2.2), (0.0, 0.0), (9.0, 9.0)]:
            image_xy = project_pitch_to_image(H_p2i, pitch_xy)
            got = seg.image_to_pitch(image_xy)
            assert got[0] == pytest.approx(pitch_xy[0], abs=1e-3)
            assert got[1] == pytest.approx(pitch_xy[1], abs=1e-3)

    def test_held_out_validation_computation(self):
        """The exact validation arithmetic used by tools/m2_fit_calibration.py:
        a held-out point's reprojection error, in metres, against its known
        true pitch coordinate."""
        seg, H_p2i, *_ = make_synthetic_segment()
        true_centre = (0.0, 0.0)
        image_xy = project_pitch_to_image(H_p2i, true_centre)
        got = seg.image_to_pitch(image_xy)
        error_m = float(np.hypot(got[0] - true_centre[0], got[1] - true_centre[1]))
        assert error_m < 1e-3


# --------------------------------------------------------------------------
# segment boundary / fail-closed behaviour
# --------------------------------------------------------------------------

class TestFailClosed:
    def test_inside_segment_returns_a_value(self):
        seg, *_ = make_synthetic_segment(frame_range=(10, 20))
        store = CalibrationStore([seg])
        assert store.find('synthetic_seq', 15) is seg

    def test_before_segment_is_unknown(self):
        seg, *_ = make_synthetic_segment(frame_range=(10, 20))
        store = CalibrationStore([seg])
        assert store.find('synthetic_seq', 9) is None
        assert store.image_to_pitch('synthetic_seq', 9, (100, 100)) is None

    def test_after_segment_is_unknown(self):
        seg, *_ = make_synthetic_segment(frame_range=(10, 20))
        store = CalibrationStore([seg])
        assert store.find('synthetic_seq', 21) is None

    def test_boundary_frames_are_inclusive(self):
        seg, *_ = make_synthetic_segment(frame_range=(10, 20))
        store = CalibrationStore([seg])
        assert store.find('synthetic_seq', 10) is seg
        assert store.find('synthetic_seq', 20) is seg

    def test_wrong_sequence_is_unknown(self):
        seg, *_ = make_synthetic_segment(frame_range=(10, 20))
        store = CalibrationStore([seg])
        assert store.find('a_different_sequence', 15) is None

    def test_no_calibration_at_all_is_unknown(self):
        store = CalibrationStore([])
        assert store.image_to_pitch('any_seq', 1, (0, 0)) is None

    def test_camera_invalidation_is_modelled_as_segment_boundary(self):
        """M2 uses manually frozen segment boundaries as the invalidation
        mechanism (section 9): a camera cut/pan a frame past the boundary is
        exactly a frame the segment does not cover."""
        seg, *_ = make_synthetic_segment(frame_range=(10, 20))
        store = CalibrationStore([seg])
        # frame 21 stands in for "one frame after the camera moved"
        assert store.image_to_pitch('synthetic_seq', 21, (100, 100)) is None

    def test_two_segments_do_not_bleed_into_each_other(self):
        seg_a, *_ = make_synthetic_segment('seq_a', (1, 10))
        seg_b, *_ = make_synthetic_segment('seq_b', (1, 10))
        store = CalibrationStore([seg_a, seg_b])
        assert store.find('seq_a', 5) is seg_a
        assert store.find('seq_b', 5) is seg_b
        assert store.find('seq_c', 5) is None


# --------------------------------------------------------------------------
# no guessed-scale fallback, ever
# --------------------------------------------------------------------------

class TestNoGuessedScaleFallback:
    def test_module_does_not_import_the_guessed_pixels_per_meter_estimator(self):
        import trackers.pitch_calibration as mod
        assert not hasattr(mod, 'SpeedDistanceEstimator')
        import_lines = [l for l in Path(mod.__file__).read_text(encoding='utf-8').splitlines()
                        if l.strip().startswith(('import ', 'from '))]
        assert not any('speed_distance' in l.lower() for l in import_lines)

    def test_short_window_speed_refuses_without_calibration(self):
        store = CalibrationStore([])
        out = short_window_displacement_and_speed(
            store, 'any_seq', 1, (0, 0), 26, (10, 10), native_fps=25.0)
        assert out is None

    def test_short_window_speed_refuses_across_two_different_segments(self):
        import dataclasses
        seg_a, *_ = make_synthetic_segment('same_seq', (1, 30))
        seg_b, H_p2i, *_ = make_synthetic_segment('same_seq', (31, 60))
        seg_a = dataclasses.replace(seg_a, artifact_sha256='segment_a_hash')
        seg_b = dataclasses.replace(seg_b, artifact_sha256='segment_b_hash')
        # two DIFFERENT calibrations covering the same sequence at different
        # times must not be silently chained together
        store = CalibrationStore([seg_a, seg_b])
        out = short_window_displacement_and_speed(
            store, 'same_seq', 20, (467.4, 138.5), 40, (467.4, 138.5), native_fps=25.0)
        assert out is None

    def test_short_window_speed_refuses_without_a_real_fps(self):
        seg, *_ = make_synthetic_segment(frame_range=(1, 60))
        store = CalibrationStore([seg])
        for bad_fps in (0, -1, None):
            out = short_window_displacement_and_speed(
                store, 'synthetic_seq', 1, (467.4, 138.5), 26, (466.1, 209.1),
                native_fps=bad_fps)
            assert out is None


# --------------------------------------------------------------------------
# speed uses world coordinates + real dt
# --------------------------------------------------------------------------

class TestWorldSpeed:
    def test_speed_uses_metric_displacement_and_real_dt_not_pixels(self):
        seg, H_p2i, *_ = make_synthetic_segment(frame_range=(1, 60))
        store = CalibrationStore([seg])
        p_a, p_b = (0.0, 0.0), (5.0, 0.0)          # exactly 5 m apart
        img_a = project_pitch_to_image(H_p2i, p_a)
        img_b = project_pitch_to_image(H_p2i, p_b)
        frame_a, frame_b, fps = 1, 26, 25.0        # 25 frames @ 25fps = 1.0 s
        out = short_window_displacement_and_speed(
            store, 'synthetic_seq', frame_a, img_a, frame_b, img_b, native_fps=fps)
        assert out is not None
        assert out['displacement_m'] == pytest.approx(5.0, abs=1e-4)
        assert out['dt_s'] == pytest.approx(1.0, abs=1e-9)
        assert out['speed_mps'] == pytest.approx(5.0, abs=1e-4)

    def test_ground_point_is_bbox_bottom_centre_not_centre(self):
        bbox = [100.0, 50.0, 140.0, 210.0]
        gp = ground_point(bbox)
        assert gp == (120.0, 210.0)
        centre = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
        assert gp != centre


# --------------------------------------------------------------------------
# artifact integrity
# --------------------------------------------------------------------------

class TestArtifactIntegrity:
    @pytest.mark.parametrize('path', [BAYERN_ARTIFACT, WOMEN_ARTIFACT])
    def test_frozen_artifact_hash_matches_its_own_content(self, path):
        seg = load_segment(str(path))     # raises on mismatch
        assert seg.artifact_sha256

    @pytest.mark.parametrize('path', [BAYERN_ARTIFACT, WOMEN_ARTIFACT])
    def test_tampered_artifact_is_rejected(self, path, tmp_path):
        d = json.loads(path.read_text(encoding='utf-8'))
        d['homography_image_to_pitch'][0][0] += 0.5   # tamper after freezing
        tampered = tmp_path / 'tampered.json'
        tampered.write_text(json.dumps(d, indent=1), encoding='utf-8')
        with pytest.raises(ValueError):
            load_segment(str(tampered))

    @pytest.mark.parametrize('path', [BAYERN_ARTIFACT, WOMEN_ARTIFACT])
    def test_artifact_validation_is_geometric_not_speed_based(self, path):
        d = json.loads(path.read_text(encoding='utf-8'))
        v = d['validation']
        assert 'held_out_landmark' in v
        assert 'known_geometry_reconstruction' in v
        assert 'numerical_stability' in v
        assert not v['numerical_stability']['degenerate']
        # nothing in the validation block may be a speed/velocity figure
        blob = json.dumps(v).lower()
        assert 'speed' not in blob and 'velocity' not in blob and 'km/h' not in blob

    def test_bayern_held_out_landmark_is_independent_of_its_fit_points(self):
        raw = json.loads((CAL_DIR / 'bayern_correspondences_raw.json').read_text(encoding='utf-8'))
        assert raw['held_out_landmark_independent'] is True

    def test_women_held_out_landmark_independence_is_disclosed(self):
        raw = json.loads((CAL_DIR / 'women_correspondences_raw.json').read_text(encoding='utf-8'))
        # touchline-direction mode constructs the fit points FROM the centre,
        # so the centre check there is not a blind independent landmark --
        # this must be recorded, not silently presented as if it were.
        assert raw['held_out_landmark_independent'] is False


# --------------------------------------------------------------------------
# M2.1 -- the withdrawn centre-computation defect, held in place so it can
# never silently come back. World points axis_a(+9.15), (0,0), axis_a(-9.15)
# are one diameter of the pitch circle and are collinear BY DEFINITION; a
# homography preserves collinearity, so their images must be collinear too.
# The withdrawn method violated this by 24.3px (34% of the line's own
# on-screen span) -- large, not noise. See experiment_M2_1_record.md.
# --------------------------------------------------------------------------

class TestM2_1CentreDefect:
    def test_collinearity_metric_is_zero_for_truly_collinear_points(self):
        d = perpendicular_distance((5.0, 5.0), (0.0, 0.0), (10.0, 10.0))
        assert d == pytest.approx(0.0, abs=1e-9)

    def test_collinearity_metric_is_nonzero_for_an_offset_point(self):
        d = perpendicular_distance((5.0, 6.0), (0.0, 0.0), (10.0, 0.0))
        assert d == pytest.approx(6.0, abs=1e-9)

    @pytest.mark.skipif(not BAYERN_RAW.exists(), reason='M2 raw correspondences not built')
    def test_withdrawn_bayern_centre_violates_collinearity(self):
        """Documents the confirmed defect: this MUST stay failing-if-inverted --
        i.e. the historical artifact's claimed centre is NOT on the halfway
        line, by a wide, unambiguous margin. If a future fix makes this
        assertion fail, the fix must be validated by a NEW, independent
        method (M2.1 section 6/7), not by silently tightening this test."""
        raw = json.loads(BAYERN_RAW.read_text(encoding='utf-8'))
        fp = raw['fit_points_image']
        centre = raw['held_out_landmark_image']['(0,0)']
        dist = perpendicular_distance(centre, fp['axis_a(+9.15)'], fp['axis_a(-9.15)'])
        span = np.hypot(*(np.array(fp['axis_a(+9.15)']) - np.array(fp['axis_a(-9.15)'])))
        assert dist / span > 0.30, (
            'the withdrawn defect no longer reproduces -- if the underlying '
            'artifact changed, re-verify M2.1 by hand before trusting this')

    def test_m2_1_status_marks_both_segments_not_validated(self):
        status = json.loads(M2_1_STATUS.read_text(encoding='utf-8'))
        assert status['segments']['bayern_calibration.json']['validated'] is False
        assert status['segments']['women_calibration.json']['validated'] is False
        assert status['verdict'].startswith('B')
        assert status['supported_after_m2_1']['metric_speed'] == 'unsupported'
        assert status['supported_after_m2_1']['metric_distance'] == 'unsupported'
