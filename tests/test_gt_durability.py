"""
Durability of hand-made GT: version control, and human decisions surviving
regeneration.

Both failures these guard against are silent. A `.gitignore` that quietly
excludes the annotation looks fine until a disk dies, and a QC report that
resets a settled event to HUMAN_REVIEW_REQUIRED looks fine until a reviewer
either redoes the work or concludes their decision was thrown away.
"""

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / 'data' / 'tracking_val_gt'
SEQ = 'women_1_239'
NEXT = 'youth_premier_league_1133'          # not yet annotated: the real test


def ignored(rel: str) -> bool:
    """True when git would refuse to add this path without -f."""
    p = subprocess.run(['git', 'check-ignore', '-q', rel],
                       cwd=REPO, capture_output=True)
    return p.returncode == 0


pytestmark = pytest.mark.skipif(
    subprocess.run(['git', 'rev-parse', '--git-dir'], cwd=REPO,
                   capture_output=True).returncode != 0,
    reason='not a git repository')


class TestFutureAnnotationIsTrackable:
    """
    Paths for a sequence nobody has annotated yet.

    Asserting on women_1 would prove nothing: those files are already in the
    index, and git ignores ignore-rules for tracked files. The question is
    whether the NEXT sequence lands without `git add -f`.
    """

    @pytest.mark.parametrize('rel', [
        f'data/tracking_val_gt/cvat_exports/{NEXT}.xml',
        f'data/tracking_val_gt/annotations/{NEXT}.json',
        f'data/tracking_val_gt/roles/{NEXT}.json',
        f'data/tracking_val_gt/qc_identity/{NEXT}/identity_gap_decisions.json',
        f'data/tracking_val_gt/human_seed/{NEXT}.json',
        f'data/tracking_val_gt/audit/{NEXT}/{NEXT}_box_audit_sample.json',
        'data/tracking_val_gt/manifest.json',
    ])
    def test_annotation_artifact_needs_no_force_add(self, rel):
        assert not ignored(rel), f'{rel} would need `git add -f`'

    @pytest.mark.parametrize('rel', [
        'data/tracking_val_gt/mot/EyeCU-val/women_1_239/gt/gt.txt',
        'data/tracking_val_gt/mot/EyeCU-val/women_1_239/seqinfo.ini',
        'data/tracking_val_gt/mot/seqmaps/EyeCU-val.txt',
    ])
    def test_mot_export_is_trackable_at_full_depth(self, rel):
        """The MOT layout nests four deep; counting levels got this wrong once."""
        assert not ignored(rel), rel


class TestBulkStaysIgnored:
    @pytest.mark.parametrize('rel', [
        f'data/tracking_val_gt/cvat_video/{NEXT}.mp4',
        f'data/tracking_val_gt/sequences/{NEXT}/img1/000001.jpg',
        'data/tracking_val_gt/qc/women_1_239_qc.mp4',
        'data/tracking_val_gt/qc/women_1_239/000001.jpg',
        f'data/tracking_val_gt/qc_identity/{SEQ}/{SEQ}_id16_gap.mp4',
        f'data/tracking_val_gt/qc_identity/{SEQ}/id16/000001.jpg',
        f'data/tracking_val_gt/audit/{SEQ}/{SEQ}_box_audit_gt_only.jpg',
        'data/tracking_val_gt/cvat_smoke/women_1_239_smoke_2tracks.xml',
        'data/frames/women_1/women_1_000250.jpg',
        'data/labels/women_1/women_1_000250.txt',
        'data/temporal_val/temporal_val_for_annotation.zip',
        'best_A_960.pt',
        'input-videos/anything.mp4',
    ])
    def test_reproducible_or_bulk_asset_stays_ignored(self, rel):
        assert ignored(rel), f'{rel} should not be version controlled'

    def test_data_was_not_broadly_unignored(self):
        """The fix is an allow-list, not `!data/`."""
        gi = (REPO / '.gitignore').read_text(encoding='utf-8')
        assert 'data/*' in gi
        assert not any(l.strip() == '!data/' for l in gi.splitlines())
        assert ignored('data/some_new_bulk_dir/whatever.bin')


class TestHumanDecisionsSurviveRegeneration:
    OUT = ROOT / 'qc_identity' / SEQ
    DEC = OUT / 'identity_gap_decisions.json'

    def test_authoritative_file_is_read_not_written(self, tmp_path):
        from tools.render_identity_qc_clips import authoritative_decisions
        assert authoritative_decisions(tmp_path) == {}, 'must not invent one'
        assert not (tmp_path / 'identity_gap_decisions.json').exists()

    @pytest.mark.skipif(not DEC.exists(), reason='no decisions recorded')
    def test_recorded_decisions_are_displayed(self):
        from tools.render_identity_qc_clips import authoritative_decisions
        d = authoritative_decisions(self.OUT)
        assert set(d) == {12, 14, 16}
        for i, v in d.items():
            assert v['status'] == 'HUMAN_CONFIRMED'
            assert v['decision'] == 'SAME'

    @pytest.mark.skipif(not DEC.exists(), reason='no decisions recorded')
    def test_generated_report_shows_them_as_settled(self):
        rec = json.loads((self.OUT / 'identity_gap_review.json'
                          ).read_text(encoding='utf-8'))
        assert rec['open_events'] == 0
        assert rec['authoritative_record'] == 'identity_gap_decisions.json'
        for e in rec['events']:
            assert e['status'] == 'HUMAN_CONFIRMED'
            assert e['decision'] == 'SAME'
            assert e['decision_source'] == 'identity_gap_decisions.json'

    @pytest.mark.skipif(not DEC.exists(), reason='no decisions recorded')
    def test_decision_record_disclaims_tracker_and_embeddings(self):
        rec = json.loads(self.DEC.read_text(encoding='utf-8'))
        assert rec['tracker_output_used'] is False
        assert rec['embeddings_used'] is False
        assert rec['review_source'] == 'human annotator'
        assert rec['annotations_altered'] is False

    def test_no_decision_means_review_required(self, tmp_path):
        """Absence of a decision is not a decision."""
        from tools.render_identity_qc_clips import authoritative_decisions
        (tmp_path / 'identity_gap_decisions.json').write_text(
            json.dumps({'decisions': [{'id': 7, 'status': 'HUMAN_CONFIRMED',
                                       'decision': 'SAME'}]}), encoding='utf-8')
        d = authoritative_decisions(tmp_path)
        assert 7 in d and 99 not in d

    def test_tool_source_never_writes_the_decisions_file(self):
        src = (REPO / 'tools' / 'render_identity_qc_clips.py').read_text(
            encoding='utf-8')
        for line in src.splitlines():
            if 'DECISIONS_FILE' in line or 'identity_gap_decisions' in line:
                assert 'write_text' not in line, line


@pytest.mark.skipif(not (ROOT / 'annotations' / f'{SEQ}.json').exists(),
                    reason='women_1 not imported')
class TestWomen1AnnotationIsUntouched:
    """Hashes pinned when the human accepted the sequence."""

    ANNOTATION_SHA256 = '12feeaf12091b90132692b5bbf49ea39a8dddc2306a70e8c14a98105a7d03408'
    ROLES_SHA256 = 'd66f3d93ebe79cac51eb0549a16128d32161936be4d5c59653e39643b0e904a4'

    def test_box_count_identity_count_and_occlusion_are_unchanged(self):
        ann = json.loads((ROOT / 'annotations' / f'{SEQ}.json'
                          ).read_text(encoding='utf-8'))
        boxes = ann['boxes']
        assert len(boxes) == 3391
        assert len({b['id'] for b in boxes}) == 19
        assert sum(b['occluded'] for b in boxes) == 158

    def test_roles_are_unchanged(self):
        roles = json.loads((ROOT / 'roles' / f'{SEQ}.json'
                            ).read_text(encoding='utf-8'))['identity_roles']
        from collections import Counter
        assert Counter(roles.values()) == {'player': 16, 'referee': 2,
                                           'goalkeeper': 1}

    def test_hashes_match_the_accepted_annotation(self):
        got_ann = hashlib.sha256(
            (ROOT / 'annotations' / f'{SEQ}.json').read_bytes()).hexdigest()
        got_roles = hashlib.sha256(
            (ROOT / 'roles' / f'{SEQ}.json').read_bytes()).hexdigest()
        assert got_ann == self.ANNOTATION_SHA256, got_ann
        assert got_roles == self.ROLES_SHA256, got_roles
