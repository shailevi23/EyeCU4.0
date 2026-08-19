# M3 — final supported-output contract

Every user-visible/statistics output of the production pipeline
(`full_pipeline.py`, `trackers/*`), classified before sealed TEST. This
governs what may be presented as a real claim versus a labelled
diagnostic/limitation from this point forward.

## SUPPORTED

| output | source | notes |
|---|---|---|
| Human detections (player/goalkeeper/referee boxes + confidence) | `trackers/detector.py`, human branch of `TwoBranchDetector` | closed component |
| Tracked human boxes + track ids | `trackers/football_tracker.py` (CBIoU, default backend) | closed component; track ids are TRACK identities, not physical-player counts — `report['unique_track_ids']` is explicitly labelled an upper bound, not a player count (`full_pipeline.py:481-486`), and stays that way |
| Ball detections and selected ball position/state (`observed` / `recovered_low_conf` / `interpolated_short_gap` / `unknown`) | SN3D_BASE + `BallTemporalSelector` v1 | closed component; `unknown` is a legitimate output, not an error |
| `processing_stats` (frame count, fps, timing) | `full_pipeline.py:_generate_statistics` | ordinary runtime counters, no geometry/team/possession dependency |
| `source_fps` / `effective_fps` | `full_pipeline.py` | read from the real video, never guessed |
| `cache_key` / provenance fields | `trackers/cache_utils.py` | identity/audit metadata |

## DEVELOPMENT-ONLY / LIMITATION (may be reported, must carry its status)

| output | status | must be labelled |
|---|---|---|
| Possession / team ball control (`compute_team_ball_control`, `team_ball_control`) | CLOSED-LIMITATION (P1/P1.1) | 46.43% exact-player correctness among mappable frames; attributed failure mechanisms in `experiments/records/experiment_P1/`. Never presented as a validated accuracy claim. |
| Team assignment (`TeamAssigner`) | IMPLEMENTED BUT UNVALIDATED | no benchmark exists; do not imply measured accuracy. No new team-assignment benchmark was created in M3 (out of scope, per instruction). |

## UNSUPPORTED (must fail closed; must never appear as a validated number)

| output | status | mechanism |
|---|---|---|
| Calibrated metric speed (km/h or m/s from real-world scale) | UNSUPPORTED | `trackers/speed_distance.py` fields are suffixed `_UNCALIBRATED` (`max_speed_kmh_UNCALIBRATED`, `total_distance_m_UNCALIBRATED`) and `report['speed_distance_calibrated'] = False` is set at every call site (`full_pipeline.py:479`, and the report-generation path). `trackers/pitch_calibration.py`'s `CalibrationStore` returns `None` outside an explicitly loaded, valid segment, and per M2.1 **no currently-loadable artifact is valid** (`M2_1_CALIBRATION_STATUS.json`), so the calibrated path also cannot currently produce a number. |
| Calibrated metric distance | UNSUPPORTED | same as above |
| Validated metric world coordinates | EXPERIMENTAL / NOT VALIDATED | M2.1 verdict B; `trackers/pitch_calibration.py` still functions as tested infrastructure but has no validated segment to apply |
| Full-match per-player total distance | UNSUPPORTED | never implemented; would additionally require solved track-identity continuity, itself unresolved (`docs/archive/TODO_legacy.md` section 6) |
| Production-ready events | UNSUPPORTED / DEFERRED | not built; out of M3 scope by instruction |

## Enforcement already in code (verified, not newly added for M3 except where noted)

- `trackers/speed_distance.py` module docstring: "⚠️ UNCALIBRATED — every km/h and metre figure produced here is unvalidated."
- `full_pipeline.py:479`: `report['speed_distance_calibrated'] = False` — unconditional, not computed from any check.
- `trackers/pitch_calibration.py`: `CalibrationStore.image_to_pitch` / `short_window_displacement_and_speed` return `None` (Python UNKNOWN) whenever no calibration covers the query; never a guessed fallback (tested, `tests/test_pitch_calibration.py::TestNoGuessedScaleFallback`).
- `M2_1_CALIBRATION_STATUS.json`: authoritative record that both existing calibration artifacts are `"validated": false`; enforced by
  `tests/test_pitch_calibration.py::TestM2_1CentreDefect::test_m2_1_status_marks_both_segments_not_validated`.
- `full_pipeline.py:481-486`: `unique_track_ids` is explicitly commented as a track-id count, not a player count.

No change was made to any of the above for M3 beyond verifying they still hold and adding the one M3-specific status test named above. Backward-compatible `_UNCALIBRATED` fields are retained as diagnostic, per instruction, and are not turned into a supported claim.
