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


class TestBallIdentityCacheKeying:
    """M3: the ball branch's identity (backend/checkpoint/imgsz/conf) must be
    part of the cache key, or a cache built with one ball detector could be
    silently reused by a run with a different one -- exactly the
    eyecu-vs-SN3D scenario the frozen contract in full_pipeline.py's
    ball_identity dict exists to prevent. Does not instantiate
    FootballAnalysisPipeline (it eagerly loads a YOLO model in __init__),
    only checks the wiring is still present and exercises the real
    ball_identity shape through compute_cache_key directly.
    """

    def _key(self, video, ball):
        return key(video, detector_settings={**BASE['detector_settings'], 'ball': ball})

    def test_eyecu_and_sn3d_backends_produce_different_keys(self, video):
        eyecu = self._key(video, {'backend': 'eyecu'})
        sn3d = self._key(video, {'backend': 'sn3d', 'sha256': 'a' * 64,
                                 'imgsz': 1280, 'accept_conf': 0.25, 'candidate_conf': 0.10})
        assert eyecu != sn3d

    def test_sn3d_checkpoint_sha_change_moves_the_key(self, video):
        base = {'backend': 'sn3d', 'sha256': 'a' * 64, 'imgsz': 1280,
                'accept_conf': 0.25, 'candidate_conf': 0.10}
        changed = {**base, 'sha256': 'b' * 64}
        assert self._key(video, base) != self._key(video, changed)

    def test_sn3d_imgsz_change_moves_the_key(self, video):
        base = {'backend': 'sn3d', 'sha256': 'a' * 64, 'imgsz': 1280,
                'accept_conf': 0.25, 'candidate_conf': 0.10}
        changed = {**base, 'imgsz': 640}
        assert self._key(video, base) != self._key(video, changed)

    def test_sn3d_candidate_pool_off_moves_the_key(self, video):
        """candidate_conf is None when ball_candidate_pool is off (see
        full_pipeline.py ball_identity construction) -- must still change
        the key relative to the pool being on."""
        with_pool = {'backend': 'sn3d', 'sha256': 'a' * 64, 'imgsz': 1280,
                    'accept_conf': 0.25, 'candidate_conf': 0.10}
        without_pool = {**with_pool, 'candidate_conf': None}
        assert self._key(video, with_pool) != self._key(video, without_pool)

    def test_pipeline_still_builds_ball_identity_with_the_frozen_fields(self):
        """Guards against ball_identity existing but drifting in shape, or the
        cache key construction forgetting to pass it."""
        import inspect
        from full_pipeline import FootballAnalysisPipeline
        init_src = inspect.getsource(FootballAnalysisPipeline.__init__)
        assert "self.ball_identity" in init_src
        for field in ('backend', 'sha256', 'imgsz', 'accept_conf', 'candidate_conf'):
            assert field in init_src
        process_src = inspect.getsource(FootballAnalysisPipeline._process_video_advanced)
        assert "'ball': self.ball_identity" in process_src
