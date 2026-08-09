"""
docs/archive/TODO_legacy.md section 6 / section 8: incompatible runs must not share a cache file.

A stale cache is silent — the run completes and reports numbers that belong to
a different video, model or setting. Each test changes exactly one input and
asserts the key moves.
"""

import pytest

from trackers.cache_utils import cache_path_for, compute_cache_key, file_fingerprint

BASE = dict(
    model_path=None,
    detector_settings={'imgsz': 960, 'confidence': 0.25, 'use_roboflow': False},
    tracker_settings={'max_ball_gap': 15, 'tracker': 'bytetrack'},
    skip_frames=2,
    max_frames=600,
)


@pytest.fixture
def video(tmp_path):
    p = tmp_path / 'clip.mp4'
    p.write_bytes(b'\x00' * 4096)
    return str(p)


def key(video, **overrides):
    kwargs = {**BASE, **overrides}
    return compute_cache_key(video_path=video, **kwargs)


def test_identical_config_reuses_cache(video):
    assert key(video) == key(video)


def test_different_video_content_changes_key(video, tmp_path):
    other = tmp_path / 'other.mp4'
    other.write_bytes(b'\x01' * 4096)
    assert key(video) != key(str(other))


def test_different_model_changes_key(video):
    assert key(video, model_path='yolov8n.pt') != key(video, model_path='yolov8s.pt')


@pytest.mark.parametrize('setting,value', [
    ('imgsz', 640),
    ('confidence', 0.4),
    ('use_roboflow', True),
])
def test_detector_settings_change_key(video, setting, value):
    changed = {**BASE['detector_settings'], setting: value}
    assert key(video) != key(video, detector_settings=changed)


def test_tracker_settings_change_key(video):
    changed = {**BASE['tracker_settings'], 'max_ball_gap': 3}
    assert key(video) != key(video, tracker_settings=changed)


def test_skip_frames_changes_key(video):
    assert key(video, skip_frames=1) != key(video, skip_frames=2)


def test_max_frames_changes_key(video):
    assert key(video, max_frames=300) != key(video, max_frames=600)


def test_cache_paths_do_not_collide(video, tmp_path):
    """Different configs must land in different files, not overwrite each other."""
    a = cache_path_for(tmp_path, 'tracks', key(video, skip_frames=1))
    b = cache_path_for(tmp_path, 'tracks', key(video, skip_frames=2))
    assert a != b


def test_fingerprint_detects_replaced_file(tmp_path):
    p = tmp_path / 'v.mp4'
    p.write_bytes(b'A' * 8192)
    before = file_fingerprint(str(p))
    p.write_bytes(b'B' * 8192)
    assert file_fingerprint(str(p)) != before


def test_fingerprint_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        file_fingerprint(str(tmp_path / 'nope.mp4'))


class TestHumanCandidatePoolKeying:
    """
    The pool changes which detections reach association, so a cache produced
    with it on must never be reused by a run with it off. This is the narrow
    regression test for that: the flag and both confidence boundaries have to
    reach the key, not merely the pipeline.
    """

    def _key(self, video, **overrides):
        settings = {'imgsz': 960, 'confidence': 0.25, 'use_roboflow': False,
                    'human_candidate_pool': False, 'human_candidate_conf': None,
                    'human_accept_conf': 0.25}
        settings.update(overrides)
        return compute_cache_key(
            video_path=str(video), model_path=None,
            detector_settings=settings,
            tracker_settings={'max_ball_gap': 15, 'tracker': 'bytetrack'},
            skip_frames=1, max_frames=None)

    def test_pool_on_and_off_do_not_share_a_cache(self, tmp_path):
        v = tmp_path / 'v.mp4'; v.write_bytes(b'x' * 4096)
        off = self._key(v)
        on = self._key(v, human_candidate_pool=True, human_candidate_conf=0.10)
        assert off != on, 'a run with the pool off could reuse a cache built with it on'

    def test_candidate_floor_is_part_of_the_key(self, tmp_path):
        v = tmp_path / 'v.mp4'; v.write_bytes(b'x' * 4096)
        a = self._key(v, human_candidate_pool=True, human_candidate_conf=0.10)
        b = self._key(v, human_candidate_pool=True, human_candidate_conf=0.15)
        assert a != b

    def test_accept_boundary_is_part_of_the_key(self, tmp_path):
        v = tmp_path / 'v.mp4'; v.write_bytes(b'x' * 4096)
        a = self._key(v, human_accept_conf=0.25)
        b = self._key(v, human_accept_conf=0.30)
        assert a != b

    def test_identical_settings_still_share_a_cache(self, tmp_path):
        v = tmp_path / 'v.mp4'; v.write_bytes(b'x' * 4096)
        a = self._key(v, human_candidate_pool=True, human_candidate_conf=0.10)
        b = self._key(v, human_candidate_pool=True, human_candidate_conf=0.10)
        assert a == b

    def test_pipeline_actually_puts_the_flag_in_the_key(self):
        """Guards against the flag existing but never reaching the key."""
        import inspect
        from full_pipeline import FootballAnalysisPipeline
        src = inspect.getsource(FootballAnalysisPipeline._process_video_advanced)
        assert 'human_candidate_pool' in src
        assert 'human_accept_conf' in src
        sig = inspect.signature(FootballAnalysisPipeline.__init__)
        assert sig.parameters['human_candidate_pool'].default is False
