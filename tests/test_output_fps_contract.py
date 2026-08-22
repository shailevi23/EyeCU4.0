"""
The output-video FPS bug: --fps used to default to a fixed 15, which was
passed straight to save_output_video() even though the pipeline always
computes effective_fps = source_fps / skip_frames. At 25fps source with
skip_frames=2 (effective 12.5), that made 375 processed frames play back at
15fps instead of their real ~30s timing -- visibly sped up.

--fps now defaults to None, meaning "use pipeline.effective_fps"; an
explicit --fps still overrides it. This test checks both halves of that
contract without running the full CLI/pipeline (no model load).
"""
from run_pipeline import build_arg_parser, resolve_output_fps


def test_fps_defaults_to_none_not_fifteen():
    parser = build_arg_parser()
    args = parser.parse_args(['--input', 'video.mp4'])
    assert args.fps is None


def test_explicit_fps_override_is_respected():
    parser = build_arg_parser()
    args = parser.parse_args(['--input', 'video.mp4', '--fps', '24'])
    assert args.fps == 24.0


def test_resolve_output_fps_uses_effective_fps_when_no_override():
    """At 25fps source / skip_frames=2, effective_fps is 12.5 -- the real
    resolve_output_fps() must return that, never a fixed 15."""
    effective_fps = 25.0 / 2
    assert resolve_output_fps(None, effective_fps) == 12.5


def test_resolve_output_fps_explicit_value_wins():
    effective_fps = 25.0 / 2
    assert resolve_output_fps(30.0, effective_fps) == 30.0
