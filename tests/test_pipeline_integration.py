"""
End-to-end regression over a handful of real frames.

Slower than the rest of the suite because it loads real YOLO weights and
decodes video. Deselect with `-m "not slow"`.

Covers TODO.md section 8: reports are produced, the four classes survive a real
run, the source FPS is read rather than assumed, and caches from different
configurations do not collide.
"""

import json
from pathlib import Path

import pytest

from full_pipeline import FootballAnalysisPipeline
from trackers.football_tracker import TRACK_KEYS
from trackers.video_utils import get_video_info

pytestmark = pytest.mark.slow

MODEL = 'yolov8n.pt'
MAX_FRAMES = 12
IMGSZ = 320  # small on purpose: this asserts behaviour, not accuracy


def run(tmp_path, video, skip_frames=2, use_cache=False, max_frames=MAX_FRAMES):
    pipeline = FootballAnalysisPipeline(
        yolo_model=MODEL,
        output_dir=str(tmp_path),
        use_roboflow=False,
        use_cache=use_cache,
        imgsz=IMGSZ,
    )
    pipeline.process_video(video_path=video, skip_frames=skip_frames,
                           max_frames=max_frames, display_results=False)
    report = pipeline.generate_final_report()
    pipeline.cleanup()
    return pipeline, report


@pytest.fixture(scope='module')
def completed_run(tmp_path_factory, sample_video):
    tmp = tmp_path_factory.mktemp('run')
    pipeline, report = run(tmp, sample_video)
    return tmp, pipeline, report


def test_final_report_is_created(completed_run):
    tmp, _, report = completed_run
    path = Path(tmp) / 'final_report.json'
    assert path.exists()
    assert json.loads(path.read_text())['frames_processed'] == report['frames_processed']


def test_player_statistics_is_created(completed_run):
    tmp, _, _ = completed_run
    path = Path(tmp) / 'reports' / 'player_statistics.json'
    assert path.exists()
    assert isinstance(json.loads(path.read_text()), dict)


def test_report_does_not_claim_a_player_count(completed_run):
    """
    Track ids are not people. Until tracking/ReID is stable the report must not
    present the id count as a number of players.
    """
    _, _, report = completed_run
    assert 'unique_track_ids' in report
    assert isinstance(report['unique_track_ids'], int)
    assert 'player_count' not in report
    assert 'num_players' not in report
    assert 'players_detected' not in report


def test_source_fps_is_read_not_assumed(completed_run, sample_video):
    _, pipeline, report = completed_run
    actual = get_video_info(sample_video)['fps']
    assert report['source_fps'] == pytest.approx(actual, rel=1e-3)
    assert pipeline.source_fps == pytest.approx(actual, rel=1e-3)


def test_effective_fps_accounts_for_skip_frames(completed_run):
    _, pipeline, report = completed_run
    assert report['effective_fps'] == pytest.approx(report['source_fps'] / 2, rel=1e-6)
    assert pipeline.effective_fps == pytest.approx(pipeline.source_fps / 2, rel=1e-6)


def test_four_classes_survive_a_real_run(completed_run):
    _, pipeline, _ = completed_run
    assert set(pipeline.adv_tracker.tracks) == set(TRACK_KEYS)


def test_runs_without_network(completed_run):
    """use_roboflow=False must mean a purely local detector."""
    _, pipeline, _ = completed_run
    assert type(pipeline.adv_tracker.detector).__name__ == 'LocalDetector'


def test_different_skip_frames_do_not_share_a_cache(tmp_path, sample_video):
    """
    The real wiring, not just the key helper: two runs differing only in
    skip_frames must write two distinct cache files.
    """
    p1, _ = run(tmp_path, sample_video, skip_frames=1, use_cache=True, max_frames=8)
    p2, _ = run(tmp_path, sample_video, skip_frames=2, use_cache=True, max_frames=8)

    assert p1.cache_key != p2.cache_key
    caches = list((Path(tmp_path) / 'cache').glob('tracks_*.pkl'))
    assert len(caches) == 2, f'expected 2 distinct cache files, found {caches}'


def test_different_imgsz_does_not_share_a_cache(tmp_path, sample_video):
    def go(imgsz):
        p = FootballAnalysisPipeline(yolo_model=MODEL, output_dir=str(tmp_path),
                                     use_roboflow=False, use_cache=True, imgsz=imgsz)
        p.process_video(video_path=sample_video, skip_frames=2,
                        max_frames=8, display_results=False)
        return p.cache_key

    assert go(320) != go(416)
