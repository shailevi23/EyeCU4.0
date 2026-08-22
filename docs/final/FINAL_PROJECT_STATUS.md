FINAL PROJECT STATUS:
COMPLETE

FINAL EVALUATION:
M5.1 corrected held-out detection evaluation,
using blind post-hoc GT repair and the original frozen M5 predictions.

VERDICT:
B — DEFENSIBLE FINAL PROJECT RESULT WITH MATERIAL HELD-OUT
GENERALIZATION LIMITATION.

REPORTING RULE:
- Original M5 pooled metric remains preserved but INVALID/superseded for
  reporting because of the original Como GT defect.
- M5.1 corrected metrics are the authoritative detection results.
- M5.1 is NOT a pristine one-shot sealed evaluation; it is a corrected-GT
  evaluation after blinded post-hoc GT repair, and is reported as such.
- Human detector held-out result: supported.
- Raw SN3D ball detector held-out result: supported but materially variable /
  weakest class.
- CBIoU and BallTemporalSelector: development evaluation only.
- Possession: CLOSED-LIMITATION.
- Team assignment: IMPLEMENTED; 46/46 (100%) on a POST-FREEZE NON-TEST
  development benchmark; NOT held-out TEST validated.
- Automatic tracklet consistency guard: evaluated POST-FREEZE; NOT ADOPTED
  (raw CBIoU tracks retained, unchanged).
- Goalkeeper possession: goalkeeper may be the recorded ball possessor;
  team control stays UNKNOWN for that frame, never fabricated.
- Metric calibration: NOT VALIDATED.
- Speed/distance: UNSUPPORTED.
- Events: UNSUPPORTED / DEFERRED.

NO M6. Post-freeze NON-TEST development work is documented in
../provenance/POST_FREEZE_SYSTEM_PATCH.md and does not reopen M4/M5/M5.1.
