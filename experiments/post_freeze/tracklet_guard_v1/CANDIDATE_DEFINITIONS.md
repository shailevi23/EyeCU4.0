# Tracklet Consistency Guard V1 — Candidate Definitions (frozen BEFORE scoring)

## Candidate A — no guard (baseline)

Every CBIoU track is assumed clean; no split, no contamination flag is ever
raised. Expected by construction: 0 false positives, 0 recall. Still scored
through the identical evaluator as B/C for a like-for-like table.

## Candidate B — robust jersey change-point guard

**Not TeamAssigner V2.** A track-consistency detector, not a team classifier.

1. **ROI**: identical chest sub-region already frozen for the team-
   assignment benchmark (15-50% height, 25-75% width of the player's own
   bbox crop) — reused, not redesigned.
2. **Quality rejection** (identical to the team-assignment benchmark's
   Candidate B, reused verbatim): reject an observation if its chest crop
   has fewer than 150 pixels (`h*w`), or if fewer than 40% of its pixels
   remain after pitch-green suppression (HSV hue 35-85, S≥60, V≥60).
3. **Descriptor** (identical 6-D, per surviving observation): Lab median
   `[L, a, b]` + HSV median saturation `S` + circular hue
   `[median(sin(2H)), median(cos(2H))]`. No pitch x/y position anywhere.
4. **Observation sampling**: up to **15** temporally-uniform *usable*
   (post-quality-rejection) observations across the track's **entire**
   cached lifetime (not just the 5 label crops) — frozen fixed cap, not
   swept.
5. **Change-point test** (deterministic, single frozen rule):
   - Order the up-to-15 surviving observations by processed-frame index.
   - For every candidate split index `i` such that both the pre-split
     prefix and post-split suffix contain **≥ 3** observations (frozen
     `min_regime_size = 3` — "both regimes must contain multiple
     observations"), compute:
     - `pre_centroid`, `post_centroid` = component-wise median of each
       side's descriptors.
     - `pre_spread`, `post_spread` = mean Euclidean distance of each side's
       observations to their own centroid (0 if a side has exactly the
       minimum 3 and all are identical).
     - `separation_ratio(i) = ||pre_centroid - post_centroid|| /
       max(mean(pre_spread, post_spread), 1e-6)`.
   - Take the split `i*` maximizing `separation_ratio` (the single best,
     most-defensible change point — not an arbitrary one).
   - **Accept as CONTAMINATED** only if `separation_ratio(i*) >= 3.0`
     (frozen threshold — stricter than the 2.0 informal bar used for
     exploratory bimodality checks earlier in this project, because this
     result gates an actual track split, not just a descriptive note).
   - A track with fewer than `2 * min_regime_size = 6` usable observations
     has no valid split candidate and is reported CLEAN (insufficient
     evidence to justify a split, not evidence of cleanliness).
   - Exactly one candidate split point per track (never more) — matches the
     "maximum one proposed split per original track in V1" requirement.
6. **No sweep**: `min_regime_size` (3) and the acceptance threshold (3.0)
   are fixed here, before any score is computed, and are not tuned
   afterward. Track #4, its frame ~88 transition, or any color identity is
   never hard-coded — the search is generic over every track's own ordered
   observations.

**Known a priori risk (disclosed before running, not after):** this exact
ROI + 150-pixel quality floor was measured, in the team-assignment
benchmark, to find **zero** usable observations on the Bayern match (median
chest-ROI area ≈ 27px, far below the 150px floor — tiny tactical-camera
bounding boxes). Candidate B is therefore expected to be unable to flag
*any* Bayern track, clean or mixed, purely from a data-availability
limitation of this ROI/threshold on this footage — not a defect in the
change-point logic itself. This is reported as a real result, not tuned
away.

## Candidate C — SigLIP change-point (optional, isolated experiment)

Run only if the isolated environment from the team-assignment benchmark
(`experiments/post_freeze/team_assignment_v2/siglip_env` or its short-path
fallback `C:/eyecu_siglip_env`, model `google/siglip-base-patch16-224`
already downloaded) is still available. If not, **skip and report SKIPPED**
— no large setup effort is spent solely for this arm.

1. **Crops**: central player crop (bbox + 10% padding, same fraction as the
   team-assignment benchmark), extracted fresh from the ORIGINAL source
   video at up to **15** temporally-uniform points across the track's
   **entire** cached lifetime (not the 5 label crops, which only span a
   coarser sample) — same cap as Candidate B, for a fair comparison.
2. **Embedding**: `SiglipVisionModel` pooled output, L2-normalized per crop.
   No fine-tuning, no text prompts, no GT-informed crop choice.
3. **Change-point test**: identical rule to Candidate B §5 (order by
   processed-frame index, search every valid split with `min_regime_size =
   3`, take the max-separation split, accept if `separation_ratio >= 3.0`),
   applied to the L2-normalized embedding vectors directly (768-D) instead
   of the 6-D color descriptor. No PCA/UMAP (same reasoning as the team-
   assignment benchmark: track-level sample counts do not need it, and
   avoiding it avoids a GT-tuned dimensionality choice).
4. No parameter sweep; the 3.0 threshold and `min_regime_size=3` are
   inherited unchanged from Candidate B's frozen rule, not re-tuned for
   embeddings.

## Common to all three

- Scored on the 56-track population in `EVALUATION_CONTRACT.md` (10
  MIXED_TRACK positive, 46 TEAM_A/B negative; 1 AMBIGUOUS excluded).
- None of the three touches the detector, CBIoU, SN3D, or
  BallTemporalSelector.
- None of the three is TeamAssigner V2 — a passing guard only decides
  whether to SPLIT a track before team assignment runs; it never assigns a
  team itself.
