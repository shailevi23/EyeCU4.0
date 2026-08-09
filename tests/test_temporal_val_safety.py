"""
Split safety for the continuous temporal validation benchmark.

These frames come from the four matches pinned to validation. If one ever
reaches training, every validation number the project reports becomes
meaningless and nothing in the metrics would reveal it. So the guard is tested
harder than the feature.
"""

import json
from pathlib import Path

import pytest

from tools.build_temporal_val import (DIAGNOSTIC_FPS, SPLIT_MARKER,
                                      VAL_MATCHES, WINDOW_FRACTIONS,
                                      WINDOW_SECONDS, assert_val_only,
                                      plan_windows)

ROOT = Path(__file__).resolve().parents[1]
TV = ROOT / 'data' / 'temporal_val'


class TestLeakageGuard:
    @pytest.mark.parametrize('path', [
        'data/dataset_baseline/images/train/women_1_tv000639.jpg',
        'data/dataset_baseline/images/test/women_1_tv000639.jpg',
        'train/x.jpg', 'test/x.jpg',
        'data/TRAIN/x.jpg',                      # case-insensitive
        Path('a') / 'Test' / 'b.jpg',
    ])
    def test_rejects_train_and_test_destinations(self, path):
        with pytest.raises(ValueError, match='non-val split'):
            assert_val_only([path])

    @pytest.mark.parametrize('path', [
        'women_1_tv000639.jpg',
        'data/temporal_val/images/women_1_tv000639.jpg',
        'data/dataset_baseline/images/val/women_1_tv000639.jpg',
    ])
    def test_allows_val_destinations(self, path):
        assert_val_only([path])          # must not raise

    def test_rejects_when_any_one_path_leaks(self):
        paths = ['data/temporal_val/images/a.jpg',
                 'data/dataset_baseline/images/train/b.jpg']
        with pytest.raises(ValueError):
            assert_val_only(paths)

    def test_substring_does_not_false_positive(self):
        """'train' inside a longer name is not the train split."""
        assert_val_only(['data/temporal_val/images/training_ground_tv01.jpg'])


class TestWindowPlan:
    def test_deterministic(self):
        assert plan_windows(25.0, 1522) == plan_windows(25.0, 1522)

    def test_stride_targets_the_diagnostic_rate(self):
        for fps in (25.0, 29.97, 30.0, 50.0):
            stride, _ = plan_windows(fps, 5000)
            assert abs(fps / stride - DIAGNOSTIC_FPS) < 1.0

    def test_one_window_per_configured_fraction(self):
        _, windows = plan_windows(25.0, 5000)
        assert len(windows) == len(WINDOW_FRACTIONS)

    def test_window_length_respects_the_rule(self):
        fps = 25.0
        _, windows = plan_windows(fps, 5000)
        for _, _, idx in windows:
            assert (idx[-1] - idx[0]) / fps <= WINDOW_SECONDS

    def test_windows_are_disjoint(self):
        _, windows = plan_windows(25.0, 5000)
        spans = sorted((idx[0], idx[-1]) for _, _, idx in windows)
        for (_, end), (nxt_start, _) in zip(spans, spans[1:]):
            assert nxt_start > end

    def test_short_video_does_not_produce_overlapping_windows(self):
        """A video short enough that both fractions collide must not emit the
        same frames twice."""
        _, windows = plan_windows(25.0, 60)
        seen = [i for _, _, idx in windows for i in idx]
        assert len(seen) == len(set(seen))

    def test_never_runs_past_the_end(self):
        _, windows = plan_windows(25.0, 100)
        for _, _, idx in windows:
            assert all(i < 100 for i in idx)

    def test_rejects_unusable_video(self):
        assert plan_windows(0, 100) is None
        assert plan_windows(25.0, 0) is None


@pytest.fixture(scope='module')
def manifest():
    return json.loads((TV / 'manifest.json').read_text(encoding='utf-8'))


@pytest.mark.skipif(not (TV / 'manifest.json').exists(),
                    reason='temporal val not built')
class TestBuiltBenchmark:
    def test_marked_val_only(self, manifest):
        assert manifest['split'] == SPLIT_MARKER
        assert (TV / 'SPLIT').exists()

    def test_only_pinned_val_matches(self, manifest):
        assert {f['match'] for f in manifest['frames']} <= set(VAL_MATCHES)

    def test_provenance_is_complete(self, manifest):
        for f in manifest['frames']:
            assert f['source_frame_index'] >= 0
            assert f['timestamp_seconds'] >= 0
            assert f['source_fps'] > 0
            assert f['stride_frames'] >= 1

    def test_frames_are_continuous_within_a_window(self, manifest):
        """Adjacent samples must be one stride apart -- that is the whole point
        of this benchmark, as opposed to the interval-sampled val split.
        Grouped per window: the jump *between* windows is expected."""
        by_window = {}
        for f in manifest['frames']:
            by_window.setdefault((f['match'], f['window']), []).append(f)
        assert by_window, 'no windows found'
        for key, frames in by_window.items():
            frames.sort(key=lambda f: f['order_in_window'])
            stride = frames[0]['stride_frames']
            gaps = {b['source_frame_index'] - a['source_frame_index']
                    for a, b in zip(frames, frames[1:])}
            assert gaps == {stride}, f'{key}: irregular gaps {gaps}'

    def test_every_match_has_all_configured_windows(self, manifest):
        by_match = {}
        for f in manifest['frames']:
            by_match.setdefault(f['match'], set()).add(f['window'])
        for match, windows in by_match.items():
            assert windows == set(range(len(WINDOW_FRACTIONS))), \
                f'{match} has windows {windows}'

    def test_gap_between_samples_is_sub_second(self, manifest):
        """Interval-sampled val frames are seconds apart and cannot support a
        motion model; these must be far closer."""
        for f in manifest['frames']:
            assert f['stride_frames'] / f['source_fps'] < 0.5

    def test_filenames_cannot_collide_with_the_val_split(self, manifest):
        existing = {p.name for p in
                    (ROOT / 'data/dataset_baseline/images/val').glob('*.jpg')}
        assert not existing & {f['file'] for f in manifest['frames']}

    def test_match_recoverable_from_filename(self, manifest):
        """Per-match grouping elsewhere uses stem.rsplit('_', 1)[0]."""
        for f in manifest['frames']:
            assert Path(f['file']).stem.rsplit('_', 1)[0] == f['match']

    def test_images_exist_and_are_readable(self, manifest):
        from PIL import Image
        for f in manifest['frames'][::10]:
            im = Image.open(TV / 'images' / f['file'])
            assert im.size[0] > 0 and im.size[1] > 0

    def test_no_image_leaked_into_the_dataset_tree(self, manifest):
        names = {f['file'] for f in manifest['frames']}
        for split in ('train', 'val', 'test'):
            d = ROOT / 'data/dataset_baseline/images' / split
            if d.exists():
                assert not names & {p.name for p in d.glob('*.jpg')}
