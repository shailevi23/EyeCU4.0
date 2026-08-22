# EyeCU 4.0 — Final Summary

**Project status: COMPLETE. No M6.**

EyeCU detects and tracks players, goalkeepers, referees and the ball in
football match video, assigns teams from jersey colour, and produces
per-track statistics, running fully offline on a locally trained detector.

## Final result

The authoritative final evaluation is **M5.1** — a corrected held-out
detection evaluation. It is reported transparently as a corrected-GT result
(blind post-hoc GT repair applied to the Como sequence, reusing the
original frozen M5 predictions with zero new inference), not as a pristine
one-shot sealed test.

**Verdict: B — Defensible final project result with material held-out
generalization limitation.**

| metric | value |
|---|---|
| mAP50 (pooled) | 0.6175 |
| mAP50-95 (pooled) | 0.2697 |
| Player AP50 / AP50-95 | 0.9220 / 0.3861 |
| Goalkeeper AP50 / AP50-95 | 0.5218 / 0.2056 |
| Referee AP50 / AP50-95 | 0.6906 / 0.3145 |
| Ball AP50 / AP50-95 | 0.3357 / 0.1726 |

The original M5 pooled number (mAP50 0.2616) is preserved for history but is
invalid/superseded due to a since-repaired Como GT annotation defect.

## What is supported vs not

| component | status |
|---|---|
| Human detector (best_A_960.pt: player/goalkeeper/referee) | **Supported** — held-out TEST |
| Ball detector (production: separate SN3D_BASE YOLO11l ball branch) | Supported, but weakest / materially variable class |
| CBIoU tracker (human-only; ball does not enter CBIoU) | Development evaluation only; unchanged by post-freeze work |
| BallTemporalSelector (on the SN3D ball branch) | Development evaluation only |
| Team assignment (legacy, production default) | Implemented; **46/46 (100%)** on a post-freeze NON-TEST development benchmark; not held-out TEST validated |
| Automatic tracklet consistency guard | Evaluated post-freeze; **NOT ADOPTED** — raw CBIoU tracks retained |
| Possession (incl. goalkeeper) | Closed-limitation — goalkeeper may be the recorded possessor, but never given a fabricated team |
| Calibration (speed/distance units) | Not validated |
| Speed / distance / events | Unsupported / deferred |

## Post-freeze development (NON-TEST, after project closure)

10/57 human-labelled tracks in a post-freeze development benchmark showed
`MIXED_TRACK` contamination (a track's visual identity changed mid-life) —
development evidence only, not a global tracking error rate. An automatic
guard to detect and split such tracks was designed and evaluated but did
not pass its own frozen adoption gate, so it was **not adopted**; CBIoU's
raw output is unchanged. Goalkeepers can now be recorded as the ball
possessor (previously impossible), without ever being given a fabricated
team. See [POST_FREEZE_SYSTEM_PATCH.md](../provenance/POST_FREEZE_SYSTEM_PATCH.md) for
the full patch and links to the underlying benchmarks.

## Final demo

`demo_outputs/final_e2e_demo/tracked_output_final_system.mp4` — Bayern
Munich 3-1 Chelsea (NON-TEST), 640×360, 375 frames, 12.5 fps, 30.0s, H.264.
Rendered from the existing cache — zero YOLO/SN3D inference. On-screen IDs
are tracking IDs, not jersey numbers; remaining visual softness is
source-resolution-limited (source is itself 640×360).

## Where to look

- [FINAL_PROJECT_REPORT.md](FINAL_PROJECT_REPORT.md) — full narrative report
- [FINAL_PRESENTATION_OUTLINE.md](FINAL_PRESENTATION_OUTLINE.md) — slide-by-slide outline
- [PIPELINE_DIAGRAM.md](PIPELINE_DIAGRAM.md) — architecture diagram
- [FINAL_PROJECT_STATUS.md](FINAL_PROJECT_STATUS.md) — one-line status/verdict record
- [POST_FREEZE_SYSTEM_PATCH.md](../provenance/POST_FREEZE_SYSTEM_PATCH.md) — post-freeze NON-TEST development patch (FPS, goalkeeper possession, team-assignment benchmark, tracklet guard)
- `experiments/records/experiment_M5_1/` — authoritative final evaluation artifacts
- `experiments/records/experiment_M5/` — historical first evaluation (superseded for reporting)
- `experiments/post_freeze/team_assignment_v2/` — team-assignment development benchmark
- `experiments/post_freeze/tracklet_guard_v1/` — tracklet-guard development benchmark (not adopted)
- `docs/results/RESULTS.md` — full measured-results history including negatives

No further experiments, model runs, GT changes, or tuning are planned or
authorized. This closes the project; the post-freeze work above is
explicitly-labelled NON-TEST development, not a reopening.
