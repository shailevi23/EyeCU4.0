# Candidate Definitions — frozen BEFORE any candidate is scored

Written and hashed before Candidate A/B/C is run against the frozen 57-track
labels. No parameter below may change after the first score is computed
without recording it as a disclosed bug-fix rerun (see the task's anti-
leakage rule).

## Candidate A — exact current baseline

The existing production `TeamAssigner.assign_teams_to_tracks(frames, tracks)`
(`trackers/team_assigner.py`), invoked exactly as `full_pipeline.py` invokes
it — no fixes, no smoothing, no confidence threshold, no ROI change. Full
behavior already documented in `CURRENT_IMPLEMENTATION.md`. Run once per
match on that match's own cached tracks + decoded frames, `num_teams=2`.

## Candidate B — robust color tracklet

1. **ROI**: same chest sub-region already audited in `CURRENT_IMPLEMENTATION.md`
   (15-50% height, 25-75% width of the player's own bbox crop) — reused, not
   redesigned, since the audit found no defect in the ROI itself.
2. **Quality rejection** (applied per observation, before any aggregation):
   - reject if the chest crop has fewer than 150 pixels total (`h*w < 150`
     after the ROI slice) — too small to trust any statistic.
   - pitch-green suppression: convert the chest crop to HSV; a pixel is
     "green" if `35 <= H <= 85` (OpenCV 0-179 hue scale) and `S >= 60` and
     `V >= 60`; those pixels are excluded from all statistics for that
     observation.
   - reject the whole observation if fewer than 40% of its pixels remain
     after green suppression (heavily contaminated / mostly pitch).
3. **Color representation** (per surviving observation, in Lab + HSV,
   computed only over the non-green pixel set):
   - Lab median: `L, a, b` (3 values)
   - HSV median saturation: `S` (1 value)
   - circular hue, since raw hue wraps at 180: `median(sin(2*H_rad))`,
     `median(cos(2*H_rad))` where `H_rad = H * pi / 90` (OpenCV hue is 0-179
     for a 360-degree wheel, so `*2` before converting to radians) — 2
     values.
   - **descriptor = 6-dimensional**: `[L, a, b, S, sin2H, cos2H]`.
   - No raw BGR mean, no x/y pitch position anywhere in this descriptor.
4. **Observation selection**: up to **9** temporally-uniform *usable*
   (post-quality-rejection) observations per track — frozen fixed number,
   not swept. If a track has fewer than 9 usable observations, all of them
   are used.
5. **Track descriptor**: component-wise **median** across the selected
   observations' 6-D descriptors.
6. **Clustering**: `StandardScaler` fit within each match on that match's
   own track descriptors only, then `KMeans(n_clusters=2, n_init=20,
   random_state=0)` fit within that match.
7. **Abstention**: no principled pre-existing confidence threshold exists in
   the codebase for this descriptor, so per the task's own instruction,
   Candidate B **always emits one of the two teams** for this benchmark (no
   invented/GT-tuned threshold). Coverage = 100% by construction.

## Candidate C — SigLIP appearance embeddings (isolated experiment)

1. **Model**: `google/siglip-base-patch16-224` (a standard, small, publicly
   pretrained SigLIP checkpoint — not fine-tuned, no text prompting used at
   all). Loaded via HuggingFace `transformers`
   (`SiglipVisionModel`/`AutoProcessor`).
2. **Isolation**: installed only into a throwaway virtualenv
   (`experiments/post_freeze/team_assignment_v2/siglip_env/`, created with
   `--system-site-packages` to reuse the already-installed CPU `torch`
   without a second heavy download), never into the production environment
   requirements. If this venv/download cannot be created, Candidate C is
   reported as infeasible with the reason, and Candidate B stands as the
   only lightweight comparison — no forced substitute.
3. **Crops**: the SAME central-player crops already extracted for labeling
   (`label_ui/crops/<match_id>/<track_id>/crop_*.jpg`, bbox + 10% padding) —
   no new crop fractions, no GT-informed selection. Up to the same **9**
   observations-per-track cap as Candidate B is applied on top of the
   already-selected 5 label crops (i.e., at most the 5 that already exist
   per track, since only 5 were extracted for labeling — this is fewer than
   9 and that is fine, the cap is a ceiling not a target).
4. **Embedding**: each crop resized/preprocessed by the model's own
   `AutoProcessor` defaults (no custom cropping/fractions), encoded through
   the SigLIP vision tower, L2-normalized.
5. **Track aggregation**: component-wise median across a track's
   L2-normalized crop embeddings, then the resulting track descriptor is
   L2-normalized again.
6. **Dimensionality**: **no PCA / no UMAP** — KMeans runs directly on the
   L2-normalized embeddings (768-D for the base model), since track counts
   per match (27, 30) do not require reduction for a k=2 fit. This is
   decided here, before any score, specifically to avoid a GT-tuned
   dimensionality choice.
7. **Clustering**: `KMeans(n_clusters=2, n_init=20, random_state=0)` fit
   within each match on that match's track descriptors only (no
   StandardScaler — embeddings are already L2-normalized and roughly
   unit-scale per-dimension by construction of the encoder).
8. **Abstention**: none — always emits one of the two teams (same
   coverage=100% rule as Candidate B, for a fair comparison).
9. This arm is authorized to run SigLIP inference (feature extraction only,
   never fine-tuned, never given the frozen labels). **YOLO/SN3D detector
   inference remains forbidden and is not touched by this arm** — Candidate
   C reuses the exact same cached bboxes/crops as everything else in this
   milestone.

## What is common to all three

- Scored only on the 57 frozen, human-labeled tracks
  (`label_ui/labels.json`, SHA256
  `24b6d4963d32a44df48fdadd599c2936835e73fd77093c81de10ee98dd5a7bf8`).
- `EVALUATION_CONTRACT.md` (already frozen, including its abstention fix)
  governs scoring for all three — no candidate-specific scoring rule.
- None of the three touches the detector, CBIoU, SN3D, or BallTemporalSelector.

See `candidate_config.json` for the machine-readable version of the frozen
numeric parameters above.
