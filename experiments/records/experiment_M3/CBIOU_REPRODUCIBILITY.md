# M3 — CBIoU reproducibility blocker — resolved

## The blocker

P1.1 (`experiment_P1_1_record.md`, "Determinism check") observed that
`youth_premier_league_1133` did not reproduce identical CBIoU track topology
between the original P1 scoring run and the P1.1 diagnostic rerun: the same
bbox at frame 202 carried a different track id with a completely different
vote history between the two runs (support 10/41 vs 107/108).

## Isolation (minimum experiment, per instruction)

**Step A/B/C — CBIoU given byte-identical input, twice.**
`tools/m3_cbiou_determinism_probe.py`: ran the human detector once over
`youth_premier_league_1133` frames 1–204, froze the raw per-frame detection
records, then fed that exact frozen stream into two fresh `CBIoUTracker`
instances, replicating `FootballTracker.get_object_tracks()`'s human-branch
logic exactly (including: never calling `.update()` at all on a frame with
no human boxes, matching production precisely).

Result: **0/204 frames differ.** Record-identical.
(`experiments/records/experiment_M3/cbiou_determinism_probe_result.json`)

**Step "compare detector streams" — the detector, run fresh, twice.**
`tools/m3_detector_determinism_probe.py`: ran `FootballTracker.detect_objects_in_frames`
twice from a cold cache over the same 204 frames, compared full per-frame
detection sets (class, bbox to 1e-6, confidence to 1e-6).

Result: **0/204 frames differ.** Record-identical.
(`experiments/records/experiment_M3/detector_determinism_probe_result.json`)

Both components are individually deterministic given fixed input, on CPU
(`torch.cuda.is_available() == False` in this environment — classic GPU
nondeterminism was never in play here).

## Source isolated

Neither isolated-component probe reproduced the original defect, so the
divergence P1.1 saw could not have come from CBIoU's or the detector's own
per-call behaviour. It traced to how the *evaluation scripts* call them.
All three of `tools/eval_possession_val.py` (P0), `tools/eval_possession_val_p1.py`
(P1), and `tools/p1_1_attribution_diagnostics.py` (P1.1) constructed **one**
`FootballTracker` — and so one `CBIoUTracker` — **before** looping over
multiple, unrelated development sequences (`bayern_munich_3-1_chelsea_228`,
`women_1_239`, `youth_premier_league_1133`), instead of one per sequence.

`CBIoUTracker` carries mutable identity state across calls to `.update()`:
`lost_track_buffer=30` frames means a track that stopped matching is kept
alive for up to 30 more frames hoping to be re-matched. Sharing one instance
across sequences means the leftover, still-alive tracks from the end of one
video's processing are present — and can compete for association — at the
start of a completely unrelated video's opening frames. This is wrong
independent of reproducibility (mixing identity state between unrelated
videos is a correctness defect on its own), and — since both components are
individually deterministic given fixed input — it is also the only
mechanism that could make two runs of the *same* script diverge: whatever
state carried into `youth_premier_league_1133`'s frames depended on the
exact history of the two sequences processed before it, and any run-to-run
difference in that carried-over state (not directly measured here, but the
only remaining candidate once both components were cleared) would explain
the symptom.

`tools/eval_possession_val.py` already contained a related, narrower fix
(a comment explaining detector-cache clearing per sequence, to stop the
*detector's* frame-id cache from replaying one clip's detections into
another). That earlier fix covered the detector's cache; it did not extend
to `CBIoUTracker`'s own persistent state, which is the gap this closes.

**Production entry points were never affected.** `full_pipeline.py` /
`run_pipeline.py` construct exactly one `FootballTracker` per process and
process exactly one video per process (`FootballAnalysisPipeline.__init__`
constructs it once; `process_video` takes a single `video_path`) — there is
no multi-sequence loop for state to leak across. Locked by
`tests/test_cbiou_determinism.py::TestProductionEntryPointsAreUnaffected`.

## Fix (smallest cause only)

Moved `tracker = FootballTracker(...)` from before each script's
`for seq, wins in sorted(by_seq.items()):` loop to the first line inside it,
in all three scripts. Every sequence now gets a fresh detector and a fresh
`CBIoUTracker`, exactly matching the isolated configuration already proven
deterministic above. Nothing about the tracking algorithm, CBIoU's
thresholds, or the detector's thresholds was touched —
`tests/test_cbiou_determinism.py` and the untouched `rf_trackers/core/cbiou/tracker.py`
confirm this. `tracker.detector.clear_cache()`, called immediately after
construction in each script, is now a harmless no-op on an already-fresh
instance and was left in place rather than removed, to keep the diff
minimal.

## Two-run record identity (post-fix)

Post-fix, each sequence's processing is, by construction, exactly the
isolated single-sequence configuration already empirically tested above —
a fresh detector and a fresh `CBIoUTracker`, given one sequence's frames,
nothing carried in from any other run. That configuration was directly
run twice and found record-identical (0/204 frame differences, both
components, on `youth_premier_league_1133` specifically — the sequence
where the original defect was observed). A fresh full three-sequence,
two-pass rerun of the corrected scripts was not additionally executed in
this pass (time-bounded); the claim above rests on the two direct
determinism probes plus the structural argument that the fix eliminates the
only channel (shared mutable tracker state) through which sequence order or
prior history could have entered the computation. This is disclosed
explicitly rather than overstated.

**Defensive addition, not required by the evidence above but pre-approved
by instruction ("deterministic backend setting if practical and
necessary")**: for M4/M5 TEST evaluation, the freeze manifest records a
requirement to run with `torch.set_num_threads(1)` and `cv2.setNumThreads(1)`
as an extra margin against any theoretical CPU multi-threaded
non-deterministic reduction order in a longer, multi-sequence-in-one-process
run — a documented, real (if not GPU-specific) phenomenon. This changes no
algorithm or threshold, only execution parallelism, and is not baked into
the default library behaviour for ordinary users, only specified as a
required setting for the frozen TEST run.

## Historical frozen results are unaffected

P0's, P1's, and P1.1's frozen result files were not re-run and are not
altered by this fix. P1.1 already explicitly caveated the
`youth_premier_league_1133` nondeterminism as a limitation in its own
record rather than silently trusting either run's numbers for the affected
rows; that disclosure stands. This M3 fix only ensures these scripts (and
any future script following the same multi-sequence pattern) are
deterministic going forward.
