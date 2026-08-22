# Team Assignment V2 — Final Results

**Status: POST-FREEZE DEVELOPMENT BENCHMARK.** Measured on two NON-TEST
matches (`bayern_munich_3-1_chelsea`, `chelsea_v_leeds_united`), 57 human-
labeled tracks, 46 of which are TEAM_A/TEAM_B (the classifier-accuracy
population; 10 MIXED_TRACK + 1 AMBIGUOUS are reported separately, never in
the accuracy denominator). **This is NOT held-out TEST validation** and does
not change any M4/M5/M5.1 result.

Labels frozen at `label_ui/labels.json`, SHA256
`24b6d4963d32a44df48fdadd599c2936835e73fd77093c81de10ee98dd5a7bf8`.
Candidate definitions frozen at `CANDIDATE_DEFINITIONS.md` /
`candidate_config.json` before any candidate was scored.

## Results

| candidate | pooled track acc. | pooled frame-weighted | Bayern | Chelsea/Leeds | coverage | runtime (both matches) |
|---|---|---|---|---|---|---|
| **A — legacy baseline** | **46/46 = 100%** | **100%** | 22/22 = 100% | 24/24 = 100% | 100% | 2.1s |
| B — robust color | 24/46 = 52.2% | 29.1% | **0/22 = 0%** | 24/24 = 100% | 52.2% | 25.2s |
| C — SigLIP | 36/46 = 78.3% | 89.4% | 22/22 = 100% | 14/24 = 58.3% | 100% | 41.6s + 99.4s model load |

## Candidate A — legacy production `TeamAssigner`

Perfect on this benchmark: 100% on both matches. This is a genuinely useful
finding, not just "nothing to fix" — it means the *classifier* is not the
source of the visible team-assignment failures users observe (like the
Bayern #4 case). See the mixed-track section below for where those failures
actually originate.

## Candidate B — robust color tracklet

A real implementation bug was found and fixed before final scoring:
`torso_bgr.size` (which counts all 3 color channels) was used where the
frozen definition specified a plain pixel count (`h*w`), making the min-
pixel quality floor ~3x more lenient than intended. Fixed, and rerun (see
`candidate_B_predictions.PRE_BUGFIX.json` for the pre-fix run, disclosed per
the anti-leakage rule). **After the fix, Bayern coverage is 0/22 (not the
pre-fix 8/30)** — every single chest-ROI observation on the Bayern match
(median ROI area ≈ 27px, max ≈ 115px) falls below the frozen 150-pixel
quality floor, because Bayern's tactical-camera player boxes are simply too
small for this ROI approach at all. This is a genuine data-limitation
finding about the frozen ROI/threshold choice on this footage, not
something loosened after seeing it fail (that would be tuning against the
benchmark) — it is reported as-is. Chelsea/Leeds crops are large enough and
Candidate B is perfect there (100%).

## Candidate C — SigLIP appearance embeddings (isolated experiment)

`google/siglip-base-patch16-224`, feature extraction only (no fine-tuning,
no text prompting), run in an isolated throwaway environment
(`C:/eyecu_siglip_env`, kept outside the repo/production requirements —
note it had to live at a short filesystem path because Windows' MAX_PATH
broke installation under this repo's long Hebrew-containing path; purely a
local filesystem workaround, not a change to the frozen model/config).
Scores 100% on Bayern (where color-based Candidate B fails completely) but
only 58.3% on Chelsea/Leeds — worse than both the free baseline and
Candidate B there. Net: does not show the "clear, repeatable improvement"
the adoption rule requires to justify a ~500MB model and a new heavy
dependency (`torch`+`transformers`).

## Mixed tracks — where the real, visible failures come from

10/57 tracks (4 Bayern, 6 Chelsea/Leeds) were human-labeled `MIXED_TRACK`.
Per-track descriptor-trajectory analysis (`MIXED_TRACK_ANALYSIS.json`)
classifies each as `ABRUPT_SUSTAINED_TRANSITION` (6 tracks — one long,
contiguous run of a second color mode, consistent with a single sustained
ID swap), `INTERMITTENT_CONTAMINATION` (3 tracks — scattered contamination,
not one clean switch), or `UNIMODAL_NO_CLEAR_SPLIT` (1 track — the color
evidence doesn't actually show a clear second mode despite the human MIXED
label, a case worth noting but not overridden: the human label stands).

**Track #4 (Bayern):** blind human label = `MIXED_TRACK`. This **confirms**
the earlier computational hypothesis from `TRACK4_TRACE.json`
(`STRONG COMPUTATIONAL EVIDENCE CONSISTENT WITH TRACK_ID_CONTAMINATION;
HUMAN CONFIRMATION PENDING`) — the wording is now upgraded to:

> **TRACK_ID_CONTAMINATION VISUALLY CONFIRMED on this development example.**

This was never a team-classifier defect. No candidate here was selected or
rejected on the basis of track #4 specifically, and none of A/B/C can fix
it — a team classifier can only ever affect the *presentation* of jersey
affiliation for a track that is, in fact, one consistent identity. It
cannot repair a genuine CBIoU tracking ID switch. **CBIoU was not modified
in this milestone.**

## Decision

**Winner: Candidate A (legacy `TeamAssigner`).** Adopted as the default
(`team_assignment_backend="legacy_color"`). Candidate B is implemented and
available (`team_assignment_backend="v2"`, `trackers/team_assigner_v2.py`)
for rollback/experimentation only — it is not adopted, given its Bayern
collapse. Candidate C (SigLIP) is not integrated into production at all
(isolated-experiment only, per its own frozen definition) — it did not show
enough improvement to justify its dependency/runtime cost, and in fact
underperformed the free baseline on one match.

`trackers/team_assigner_v2.py` supports `team=None` (unknown) when a track
has no usable observation anywhere in its lifetime, rather than guessing —
downstream consumers (`compute_team_ball_control`, viewer/debug drawing)
already tolerate a missing `team`/`team_color` key unchanged.

## Viewer fixes applied (presentation-only, separate from the above)

- Possessor lookup in the HUD now includes any eligible human with
  `has_ball=True` (not only `role == 'player'`), so a goalkeeper possession
  would display correctly if the underlying possession logic ever produces
  one.
- `"Ball: #ID"` → `"Possessor #ID"`.
- Team-possession percentages removed from VIEWER mode (possession remains
  CLOSED-LIMITATION, not a display-worthy accuracy claim); still shown in
  DEBUG mode.
- Processed-frame index removed from VIEWER mode.
- Possessor halo changed from a yellow-ish ring to a neutral white/grey
  ring — yellow is now reserved for the ball marker only.
- Label placement remains capped to a few pill-heights of displacement
  (3 bounded, clamped tries; no long leader lines) and already suppresses
  the pill (keeping the marker) when no in-frame, non-colliding placement
  is found.

## Final cached demo

`demo_outputs/archive/tracked_output_team_v2_final.mp4` — rendered
from the ORIGINAL Bayern source frames + the existing tracks cache (zero
YOLO/SN3D inference; cache-hit confirmed), 640×360, 375 frames, 12.5 fps,
real H.264 (`avc1`), using the winning `legacy_color` backend and the
viewer fixes above.

## Scientific boundary

No TEST data was accessed. No detector, CBIoU, SN3D, or BallTemporalSelector
code was modified. M4/M5/M5.1 artifacts and metrics are unchanged. This
milestone's results are development-measured evidence on two NON-TEST
matches, not a held-out TEST claim, and do not retroactively upgrade the
project's frozen scientific status.
