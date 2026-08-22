# Post-Freeze System Patch — FPS, Goalkeeper Possession, Tracklet Guard Evaluation

Post-freeze development work. Does not change M4/M5/M5.1, which remain
historical and unchanged. No TEST data was accessed at any point.

## Team assignment

Legacy `TeamAssigner` retained as the production default
(`team_assignment_backend="legacy_color"`). Not redesigned or replaced in
this milestone. **46/46 (100%) on the frozen post-freeze NON-TEST
development team-assignment benchmark**, both matches — see
`experiments/post_freeze/team_assignment_v2/FINAL_RESULTS.md`. `v2`
(robust color tracklet) remains available for rollback/experimentation only
and is not the default.

## Tracking (CBIoU)

**Scientific status unchanged.** CBIoU association algorithm and parameters
were not touched in this milestone. A tracklet-consistency guard — a
post-CBIoU development layer that would flag and split sustained cross-team
identity contamination inside a single track ID — was evaluated against a
frozen gate (mixed recall ≥6/10, clean false positives ≤1/46, ≥1 detection
on both matches) using three candidates (no-guard baseline, a robust-color
change-point detector, and an optional SigLIP change-point detector).
**None passed the gate**, so **no guard was adopted**; the raw CBIoU
pipeline and its output tracks are unchanged. Full results:
`experiments/post_freeze/tracklet_guard_v1/FINAL_RESULTS.md`.

## Mixed track / contamination evidence

**10/57** selected long-lived player tracks across two NON-TEST development
matches were human-labeled `MIXED_TRACK`. This is evidence from a specific,
deliberately-selected development benchmark of long-lived tracks — **it is
not a global tracking error rate, not a per-frame error rate, and does not
measure same-team ID switches.** Bayern track #4 — the originally observed,
visibly-wrong-team example — was blindly human-labeled `MIXED_TRACK`,
independently confirming the earlier hypothesis
(`TRACK_ID_CONTAMINATION VISUALLY CONFIRMED on this development example`).
No automatic guard tried here could detect it (see above); the underlying
tracking identity for that track remains scientifically as CBIoU produced
it, undisturbed.

## Possession

- Goalkeepers may now become the recorded human possessor (`has_ball=True`)
  when the ball is nearest to them, under the exact same geometry and
  `max_distance` threshold already used for field players (not tuned).
  Previously `PlayerBallAssigner` only ever searched `tracks['players']`,
  so a goalkeeper could never receive possession at all.
- Goalkeeper team-control credit remains **UNKNOWN** for any frame a
  goalkeeper is the possessor — no team is fabricated from goalkeeper kit
  color, since goalkeepers are deliberately excluded from jersey
  `TeamAssigner`. Team-possession statistics remain **CLOSED-LIMITATION**,
  unchanged from prior status.
- The viewer's "Possessor #ID" HUD line already worked for any eligible
  human (fixed in the previous milestone); it now has real goalkeeper-
  possession events to actually display.

## Output FPS

`run_pipeline.py --fps` now defaults to `None`, meaning "use the pipeline's
own `effective_fps` (`source_fps / skip_frames`)" instead of a fixed `15`.
At the demo's 25fps source with `skip_frames=2`, that is `12.5` — the same
number the pipeline already computed and used internally, now also used for
the output video's actual playback timing by default. An explicit `--fps`
value still overrides it. The one remaining internal hardcoded-15 call site
(`full_pipeline.py`'s own demo `__main__` block) was fixed the same way.

## Viewer

Unchanged in this milestone (kept exactly as previously approved): compact
foot markers, current ID pills, "Possessor #ID" wording, no possession
percentages or frame counter in VIEWER mode, yellow reserved for the ball,
neutral possessor emphasis, bounded/suppress-on-failure label placement.

## Tests added

- `tests/test_output_fps_contract.py` — `--fps` defaults to `None`, an
  explicit value still overrides, and the resolution formula never guesses
  a fixed number.
- `tests/test_goalkeeper_possession.py` — a field player nearest the ball
  gets possession and team credit; a goalkeeper nearest the ball gets
  possession but never a fabricated team; no ball / unresolved ball stays
  unknown for both; missing `'goalkeepers'` key (as several existing
  possession tests use) still works.

## Final NON-TEST cached demo

`demo_outputs/final_e2e_demo/tracked_output_final_system.mp4` — rendered
from the existing Bayern tracks cache (zero YOLO/SN3D inference; cache-hit
confirmed), using the unchanged raw CBIoU tracks (no guard applied, per the
decision above), legacy `TeamAssigner`, the corrected goalkeeper-possession
logic, and the current viewer. Verified on readback: 640x360, 375 frames,
12.5 fps, 30.0s, H.264 (`avc1`).

## Scientific boundary

TEST accessed: NO. Detector weights/config, YOLO, SN3D, and
BallTemporalSelector: unchanged. CBIoU association algorithm/parameters:
unchanged. M4/M5/M5.1 and the frozen team-assignment labels: unchanged.
Legacy TeamAssigner algorithm: unchanged. This patch's new evidence status
is **POST-FREEZE NON-TEST DEVELOPMENT** throughout.
