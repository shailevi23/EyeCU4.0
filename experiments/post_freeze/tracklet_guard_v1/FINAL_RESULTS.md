# Tracklet Consistency Guard V1 — Final Results

**Status: POST-FREEZE DEVELOPMENT BENCHMARK. NOT held-out TEST validation.**
Does not change M4/M5/M5.1. Reuses the same frozen human labels as the
team-assignment benchmark (SHA256
`24b6d4963d32a44df48fdadd599c2936835e73fd77093c81de10ee98dd5a7bf8`, verified
unchanged before this benchmark started).

**Scope note (do not overstate):** this measures whether an automatic guard
detects the same kind of contamination already found by human labeling on
57 selected long-lived tracks from two NON-TEST matches. It is **not** a
global tracking error rate, not per-frame accuracy, and not a same-team
ID-switch measurement.

## Adoption gate (frozen before any result)

1. Mixed recall ≥ 6/10
2. Clean false positives ≤ 1/46
3. ≥ 1 detection in **both** matches
4. No downstream TeamAssigner degradation (checked only if 1-3 pass)

## Results

| candidate | TP/10 | FP/46 | recall | specificity | precision | F1 | Bayern detections | Chelsea/Leeds detections |
|---|---|---|---|---|---|---|---|---|
| A — no guard | 0 | 0 | 0.000 | 1.000 | — | 0.000 | 0 | 0 |
| B — color change-point | 2 | 7 | 0.200 | 0.848 | 0.222 | 0.211 | **0** | 2 |
| C — SigLIP change-point | 0 | 0 | 0.000 | 1.000 | — | 0.000 | 0 | 0 |

**None passes the gate.** See `RESULTS.json` for full per-match TP/FP/FN/TN
and real track-id diagnostics.

- **Candidate A** trivially has 0 recall (never flags anything) — included
  only for a like-for-like table.
- **Candidate B** reaches 2/10 recall with 7/46 false positives (fails both
  the recall and FP gates on its own), and — as flagged as an a priori risk
  in `CANDIDATE_DEFINITIONS.md` before it was run — found **zero usable
  chest-ROI observations on all 27 Bayern tracks**, the same tiny-bbox
  data-availability limitation already measured in the team-assignment
  benchmark. It cannot even attempt Bayern, so it fails gate criterion 3
  outright regardless of its Chelsea/Leeds numbers.
- **Candidate C** (SigLIP) processed every track on both matches (no
  data-availability failure) but its embedding-trajectory separation ratio
  never crossed the frozen 3.0 threshold on a single track — 0 recall, 0
  false positives. The frozen threshold, carried over unchanged from
  Candidate B's color descriptor, may simply not be calibrated for a 768-D
  embedding space's typical distances; this is reported as a finding, not
  adjusted after the fact (that would be tuning against the benchmark).

## Track #4 (diagnostic only — did not determine the winner)

- Human label: `MIXED_TRACK`
- Candidate B auto-detected: **NO** (zero usable observations for this
  track, same as every other Bayern track)
- Candidate C auto-detected: **NO** (processed, but separation ratio never
  reached 3.0)
- No candidate was tuned toward this specific example at any point; the
  frozen rule is generic over every track's own ordered observations.

## Decision

**No guard is adopted.** Per the frozen gate, since no candidate passes,
`DO NOT ADOPT`. The raw CBIoU pipeline is retained completely unchanged —
`trackers/tracklet_consistency.py` was **not created**, and no downstream
split/regression-check work (task §§10-12) was performed, since those steps
are explicitly conditional on a candidate passing the gate.

Cross-team identity contamination (evidenced by 10/57 MIXED_TRACK human
labels across two NON-TEST matches, §6 of the team-assignment benchmark) is
documented as a **measured development limitation** of the current system:
the automatic appearance-based guards tried here could not reach a usable
recall/false-split trade-off. This is an honest negative result, not a
reason to ship an unreliable automatic correction on top of CBIoU.

## What is retained from this milestone regardless

- **Output FPS default fix** (`run_pipeline.py --fps` now defaults to
  `None` → `pipeline.effective_fps`, not a fixed 15).
- **Goalkeeper possession fix** (`trackers/player_ball_assigner.py` now
  searches players AND goalkeepers for ball possession; team-control credit
  stays UNKNOWN for a goalkeeper possessor, never fabricated).
- Both are independent of the guard decision and are unconditionally kept.

## Scientific boundary

No TEST data accessed. CBIoU association algorithm/parameters unchanged.
Detector, SN3D, and BallTemporalSelector unchanged. M4/M5/M5.1 unchanged.
Legacy TeamAssigner unchanged (still the adopted default, 46/46 on its own
frozen benchmark — see `experiments/post_freeze/team_assignment_v2/`).
