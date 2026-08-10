"""
MP4 clips for CVAT video tasks.

The clip is only a UI vehicle, but its frame mapping is load-bearing: if
decoded frame k is not package frame k+1, every annotation lands on the wrong
frame and nothing downstream notices. So the tests are about the mapping and
about the file never being mistaken for evidence.
"""

import json
import shutil
from fractions import Fraction
from pathlib import Path

import pytest

from tools.build_cvat_video_clips import (NEIGHBOUR_MARGIN, encode, fps_fraction,
                                          verify)

ROOT = Path(__file__).resolve().parents[1] / 'data' / 'tracking_val_gt'
CLIPS = ROOT / 'cvat_video'
has_ffmpeg = shutil.which('ffmpeg') is not None


class TestFrameRate:
    def test_integer_rate_stays_exact(self):
        assert fps_fraction(25.0) == '25/1'

    def test_rate_is_rational_not_decimal(self):
        """A decimal fps drifts; ffmpeg wants a ratio."""
        for v in (25.0, 29.97, 30.0, 50.0):
            r = fps_fraction(v)
            assert '/' in r
            assert abs(float(Fraction(r)) - v) < 0.001, r

    def test_exact_fps_refuses_to_guess_on_disagreement(self, tmp_path):
        from tools.build_cvat_video_clips import exact_fps
        man = json.loads((ROOT / 'manifest.json').read_text(encoding='utf-8'))
        s = man['sequences'][0]
        with pytest.raises(SystemExit, match='Refusing to guess'):
            exact_fps(s['source_video'], declared=5.0)


@pytest.mark.skipif(not has_ffmpeg, reason='ffmpeg not installed')
class TestRoundTripThroughMp4:
    """Encode a handful of real package frames and decode them back."""

    @pytest.fixture(scope='class')
    def built(self, tmp_path_factory):
        man = json.loads((ROOT / 'manifest.json').read_text(encoding='utf-8'))
        s = next(x for x in man['sequences'] if x['sequence'] == 'women_1_239')
        img1 = ROOT / 'sequences' / s['sequence'] / 'img1'
        if not (img1 / '000001.jpg').exists():
            pytest.skip('package frames not on disk')
        out = tmp_path_factory.mktemp('clip') / 'c.mp4'
        encode(img1, out, 6, '25/1')
        return out, img1, s

    def test_frame_count_is_exact(self, built):
        clip, img1, s = built
        ok, checks, mapping, _ = verify(clip, img1, 6, 25.0,
                                        s['frame_width'], s['frame_height'])
        assert ok, [c for c in checks if not c['ok']]
        assert len(mapping) == 6

    def test_decoded_frame_zero_is_package_frame_one(self, built):
        clip, img1, s = built
        _, _, mapping, _ = verify(clip, img1, 6, 25.0,
                                  s['frame_width'], s['frame_height'])
        assert mapping[0]['decoded_frame_0based'] == 0
        assert mapping[0]['package_frame_1based'] == 1

    def test_every_frame_beats_its_neighbours(self, built):
        """Ordering is checked against the frames, not assumed from the count."""
        clip, img1, s = built
        _, _, _, worst = verify(clip, img1, 6, 25.0,
                                s['frame_width'], s['frame_height'])
        assert worst > NEIGHBOUR_MARGIN, worst

    def test_wrong_frame_count_is_detected(self, built):
        clip, img1, s = built
        ok, checks, _, _ = verify(clip, img1, 7, 25.0,
                                  s['frame_width'], s['frame_height'])
        assert not ok
        assert any(c['check'] == 'exact frame count' and not c['ok']
                   for c in checks)

    def test_wrong_fps_is_detected(self, built):
        clip, img1, s = built
        ok, checks, _, _ = verify(clip, img1, 6, 30.0,
                                  s['frame_width'], s['frame_height'])
        assert not ok
        assert any('fps' in c['check'] and not c['ok'] for c in checks)


@pytest.mark.skipif(not (CLIPS / 'women_1_239_smoke10.provenance.json').exists(),
                    reason='smoke clip not built')
class TestShippedSmokeClip:
    @pytest.fixture(scope='class')
    def rec(self):
        return json.loads((CLIPS / 'women_1_239_smoke10.provenance.json'
                           ).read_text(encoding='utf-8'))

    def test_clip_shipped_only_because_it_verified(self, rec):
        assert rec['verified'] is True
        assert all(c['ok'] for c in rec['verification'])
        assert (CLIPS / rec['clip']).exists()

    def test_clip_is_not_authoritative(self, rec):
        assert rec['authoritative'] is False
        assert 'UI vehicle' in rec['canonical_provenance']

    def test_mapping_is_contiguous_and_one_based(self, rec):
        m = rec['frame_mapping']
        assert [x['decoded_frame_0based'] for x in m] == list(range(len(m)))
        assert [x['package_frame_1based'] for x in m] == list(range(1, len(m) + 1))
        srcs = [x['source_frame'] for x in m]
        assert srcs == list(range(srcs[0], srcs[0] + len(m)))

    def test_source_frames_match_the_frozen_range(self, rec):
        man = json.loads((ROOT / 'manifest.json').read_text(encoding='utf-8'))
        s = next(x for x in man['sequences'] if x['sequence'] == rec['sequence'])
        assert rec['frame_mapping'][0]['source_frame'] == s['source_frame_range'][0]
        assert rec['width'] == s['frame_width'] == 640
        assert rec['height'] == s['frame_height'] == 360
