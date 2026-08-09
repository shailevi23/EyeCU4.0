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
