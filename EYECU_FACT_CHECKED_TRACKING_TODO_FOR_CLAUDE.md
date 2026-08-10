# EyeCU 4.0 — Fact-Checked Tracking Validation, Wrapper Audit & Tracker Migration TODO

**Purpose:** This is an execution prompt for Claude Code.  
**Scope:** Human multi-object tracking only, plus the minimum tooling needed to measure it correctly.  
**Date:** 2026-08-09.  
**Status:** Detector research is frozen. TEST remains untouched.

---

# 0. YOUR ROLE

Act as a **senior computer-vision / multi-object-tracking research engineer** working on EyeCU 4.0.

Your job is **not** to make tracking numbers look better.

Your job is to determine, with the least ambiguous experiment possible:

1. what part of the current human-tracking stack is actually causing fragmentation;
2. whether the old `supervision==0.26.1` ByteTrack wrapper is altering or dropping otherwise-valid tracker output;
3. how good the current tracker is when measured against **real identity ground truth**, not proxy metrics;
4. whether a modern tracker implementation provides a real improvement on EyeCU's frozen detector output;
5. when to STOP optimizing tracking and move the project forward.

Treat this as a research/engineering investigation, not a hyperparameter contest.

---

# 1. UNDERSTAND THE SYSTEM ROLE BEFORE CHANGING ANYTHING

EyeCU is a **video-analysis system**, not just a detector.

The relevant pipeline is conceptually:

```text
VIDEO FRAME
    |
    v
DETECTOR
    |
    |-- player
    |-- goalkeeper
    |-- referee
    `-- ball
    |
    +------------------------------+
    |                              |
    v                              v
HUMAN TRACKING                 BALL SYSTEM
player/GK/referee              separate path
class-agnostic association     candidate/dedupe/temporal logic
    |                              |
    v                              |
TRACK IDENTITY                     |
    |                              |
    v                              |
SEMANTIC ROLE / TEAM                |
    |                              |
    +--------------+---------------+
                   |
                   v
          REPORTS / VISUALIZATION
                   |
                   v
          CALIBRATION / ANALYTICS
```

The detector answers:

> **What do I see in this frame?**

The tracker answers:

> **Which detection belongs to the same physical person across frames?**

Those are different problems.

A person may be detected correctly in every frame while the tracker keeps assigning new IDs.

Conversely, a tracker may preserve an ID through a difficult sequence while the detector temporarily assigns the wrong semantic role.

Do not mix:

- detection recall;
- tracking association;
- role classification;
- track-level role stability;
- ball recovery;
- pitch calibration.

They require different metrics.

---

# 2. WHY THIS PHASE EXISTS

Earlier EyeCU work used proxy tracking diagnostics because there was no identity ground truth.

Those diagnostics were useful for finding problems, but they cannot answer the final association question.

For example:

```text
unique IDs / simultaneous humans
median track length
interior gaps
implausible bbox jumps
```

can reveal fragmentation.

They **cannot prove identity correctness**.

A track can remain visually smooth while silently switching from one nearby player to another.

Therefore:

> **Do not call proxy continuity metrics IDF1, HOTA, AssA, identity switches, or false associations.**

The project now needs to move from:

```text
"this looks like association failure"
```

to:

```text
"this is measured association failure against identity GT"
```

before any tracker is tuned or replaced in production.

---

# 3. FACT-CHECKED EXTERNAL FINDINGS — TREAT THESE AS CONSTRAINTS

These findings were re-checked against primary/official sources immediately before this TODO was written.

## 3.1 `supervision==0.26.1` does more than internal ByteTrack association

In Supervision 0.26.1:

```python
tracks = self.update_with_tensors(tensors=tensors)
```

is followed by another mapping step inside `update_with_detections()`.

The wrapper:

1. takes the original detection boxes;
2. takes the boxes returned by internal ByteTrack;
3. calculates pairwise IoU;
4. converts IoU to cost;
5. runs another linear assignment with threshold `0.5`;
6. writes `tracker_id` back only for detections matched in this wrapper step;
7. returns only detections with `tracker_id != -1`.

Therefore the public EyeCU result from:

```python
sv.ByteTrack().update_with_detections(...)
```

is **not simply the raw internal `update_with_tensors()` output**.

This wrapper behavior must be measured before blaming all observed fragmentation on the ByteTrack algorithm itself.

Do not assume the wrapper is harmful.

Measure it.

---

## 3.2 Supervision 0.26.1 confidence boundaries are strict

The verified internal code uses:

```python
remain_inds = scores > self.track_activation_threshold
inds_low    = scores > 0.1
inds_high   = scores < self.track_activation_threshold
inds_second = np.logical_and(inds_low, inds_high)
```

With the default activation threshold `0.25`:

```text
high-confidence association: confidence > 0.25
second-stage pool:           0.10 < confidence < 0.25
```

Exact `0.10` is not in the low pool.

Exact `0.25` is not in either pool.

Do not silently rewrite this as inclusive thresholds.

---

## 3.3 The old implementation also has a separate new-track gate

Supervision 0.26.1 sets:

```python
self.det_thresh = self.track_activation_threshold + 0.1
```

and unmatched detections are prevented from creating a new track when:

```python
track.score < self.det_thresh
```

With the default activation threshold `0.25`, new-track initiation therefore has an effective default gate of approximately:

```text
0.35
```

This is a different concept from high-confidence association.

Do not describe the old implementation as if one `0.25` threshold controls everything.

---

## 3.4 `sv.ByteTrack` is now deprecated in favor of a dedicated tracker package

Current Supervision documentation deprecates `sv.ByteTrack` in favor of Roboflow's dedicated `trackers` package and `ByteTrackTracker`.

The dedicated package exposes tracker implementations including:

- ByteTrack
- OC-SORT
- BoT-SORT
- C-BIoU
- SORT

and uses a shared tracking interface.

This does **not** mean EyeCU should migrate automatically.

It means the legacy wrapper deserves scrutiny, and modern implementations are legitimate comparison candidates.

---

## 3.5 CRITICAL EyeCU-specific integration risk: Python namespace collision

Roboflow's external package is imported as:

```python
from trackers import ByteTrackTracker
```

EyeCU itself already has a top-level local package:

```text
trackers/
```

Therefore a direct install/import inside the current EyeCU repo may resolve:

```python
import trackers
```

to **EyeCU's own package**, not Roboflow's external distribution.

This is a real namespace collision risk.

Before installing or integrating the external package, verify:

```bash
python -c "import trackers; print(trackers.__file__)"
```

Do **not** rename EyeCU's entire package or manipulate `sys.path` just to make an experiment work.

For the first tracker bake-off, prefer an **isolated evaluation environment outside the repo import root**, consuming serialized frozen detections.

Production integration is a later decision.

---

## 3.6 SoccerNet explicitly separates association from complete tracking

SoccerNet Tracking defines two useful experimental tasks:

### Pure association

Use **ground-truth detections** and evaluate only whether identities are associated correctly.

### Complete tracking

Start from video/detector output and evaluate detection + association together.

SoccerNet uses HOTA and explicitly separates:

```text
DetA = detection accuracy component
AssA = association accuracy component
```

This is directly relevant to EyeCU.

EyeCU should reproduce this conceptual separation on its own VAL footage.

---

## 3.7 TrackEval is the correct kind of evaluation tool

TrackEval supports:

- HOTA
- DetA
- AssA
- IDF1 / IDP / IDR
- CLEAR MOT metrics
- fragmentation metrics

For custom tracking benchmarks it recommends converting data to an implemented format, with MOTChallenge format as the default recommendation.

Do not invent an EyeCU-only formula for identity quality if the standard metrics are available.

---

## 3.8 Sports tracking is association-heavy

SportsMOT identifies sports tracking as difficult because of:

- fast motion;
- variable-speed motion;
- similar appearances;
- dense interactions.

Its authors identify object association as a key challenge.

That matches the domain EyeCU operates in, but it does not prove any specific EyeCU tracker is wrong.

---

## 3.9 BoT-SORT and C-BIoU are scientifically relevant alternatives

BoT-SORT explicitly adds camera-motion compensation and stronger association machinery.

C-BIoU expands/buffers matching space to address irregular motion and similar appearances, and its paper reports it as a major component of a second-place SoccerNet MOT solution.

These are legitimate comparison candidates.

Do not assume they will beat ByteTrack on EyeCU.

Measure them on EyeCU identity GT.

---

## 3.10 External benchmark rankings are context, not EyeCU evidence

Roboflow's tracker comparison reports results on SportsMOT and SoccerNet.

Important methodology difference:

- SportsMOT uses detector outputs.
- Their SoccerNet comparison uses oracle / ground-truth detections.

Therefore:

> **Never copy a tuned SoccerNet configuration and call it optimized for EyeCU.**

The external numbers justify which algorithms are worth testing.

They do not select the EyeCU winner.

---

# 4. CURRENT EYECU STATE — VERIFY LOCALLY BEFORE TRUSTING THIS SECTION

The following is the current project state reported during this work session.

Verify every item against the local repository.

Do not assume the public GitHub `main` is current.

## Detector

Frozen detector decision:

```text
A = YOLO26s @ 960
production/speed baseline

B = YOLO26s @ 1280
accuracy reference

C = rejected
contextual crop/zoom + hard-negative experiment
```

Do not retrain A/B/C in this task.

Do not implement P2 in this task.

---

## Ball

Ball has been removed from the human ByteTrack association pool.

The old double-write was removed.

Ball must remain fully separate from human multi-object tracking.

Do not change ball thresholds, dedupe, TemporalSelector, or ball recovery in this task.

---

## Human low-confidence candidate experiment

A feature exists for low-confidence human association evidence.

It is **OFF by default** and was rejected for production.

Mechanism test succeeded:

```text
0.90 -> 0.15 -> 0.90
```

could preserve identity internally without exposing the 0.15 box publicly.

But the real four-window benchmark failed the predeclared production criteria:

```text
macro IDs/human: 5.34 -> 5.18
pooled IDs/human: 4.66 -> 4.53
runtime: ~+37%
association-break proxy share did not improve
```

Therefore:

> Keep the feature for reproducibility/testing, default OFF.

Do not retune its threshold.

---

## Frozen continuity windows

Use these exact windows unless this TODO explicitly says otherwise:

```text
austin      start 284
bayern      start 228
women_1     start 239
youth       start 1133
```

Each is:

```text
300 native-FPS frames
continuous camera shot
>=10 mean humans/frame over the full 300-frame window
```

Do not replace a window after seeing tracker results.

---

## Current proxy baseline

Reported baseline on those windows:

```text
macro-average IDs/human = 5.34
pooled IDs/human        = 4.66
median of window median track lengths = 21 frames
```

These are diagnostic proxies only.

Do not use them as the final tracker-selection metric after identity GT exists.

---

## Role instability

A large share of observed raw semantic role flips occur on otherwise-continuous tracks, especially:

```text
referee <-> player
```

This appears to be a real semantic-jitter problem.

Do not implement role smoothing yet.

First fix/choose tracking, then evaluate role smoothing separately.

---

## TEST

The frozen TEST split remains untouched.

Do not:

- inspect TEST labels;
- label TEST;
- run trackers on TEST for selection;
- choose a tracker from TEST;
- tune anything against TEST.

TEST is still reserved for the final frozen-system evaluation.

---

## Repository / remote state

The public GitHub repository visible at the time this TODO was created still showed an old legacy state.

The local working repository is therefore the source of truth for this task.

Do not push just to make GitHub match local.

First:

- verify local history;
- complete repo/doc cleanup;
- verify no secrets remain in the working tree;
- decide intentionally when to push.

---

# 5. GLOBAL SCIENTIFIC RULES

These apply to every phase below.

## Rule 1 — one question per experiment

Do not combine:

```text
tracker migration
+
threshold tuning
+
role smoothing
+
detector changes
```

in one result.

---

## Rule 2 — frozen detections for tracker comparisons

When comparing trackers on the complete-tracking task, every tracker must receive the **same serialized A@960 detections**.

Do not rerun the detector separately for each tracker unless validating serialization once.

This prevents detector nondeterminism/timing from contaminating the tracker comparison.

---

## Rule 3 — distinguish oracle association from end-to-end tracking

Always report:

```text
ORACLE ASSOCIATION
GT boxes -> tracker -> identity metrics
```

separately from:

```text
END-TO-END TRACKING
frozen A detections -> tracker -> identity metrics
```

If oracle association is strong but end-to-end performance is weak, the detector/support is limiting the system.

If oracle association itself is weak, the tracker/association model is limiting.

---

## Rule 4 — do not tune before a default bake-off

First compare default/frozen configurations.

No Optuna.

No grid search.

No copying SoccerNet tuned parameters.

No manual threshold fishing.

---

## Rule 5 — proxy metrics remain secondary

Continue reporting:

- IDs/human;
- track length;
- gaps;
- role flips;

because they are useful for interpretation.

But after identity GT exists, primary tracking selection uses standard identity metrics.

---

## Rule 6 — preserve semantic classes

EyeCU semantics remain:

```text
0 player
1 goalkeeper
2 referee
3 ball
```

For human association it is allowed to treat:

```text
player/GK/referee -> association class HUMAN
```

but raw semantic class/confidence must survive.

Never permanently normalize goalkeeper to player.

Ball never enters human association.

---

## Rule 7 — no fabricated GT

Do not use tracker-generated IDs as identity ground truth.

Do not auto-copy ByteTrack IDs into the annotation package.

The tracker being evaluated cannot define the answer key.

---

## Rule 8 — no TEST access

Repeated because it matters.

All work below is TRAIN/VAL only.

---

# 6. PHASE 0 — REPOSITORY + ENVIRONMENT TRUTH AUDIT

Do this before changing code.

Return the exact outputs/summary for:

```bash
git status
git branch --show-current
git rev-parse HEAD
git log --oneline -12
git remote -v
```

Record:

- current branch;
- current HEAD;
- clean/dirty status;
- whether the latest human-candidate/cache work is committed;
- latest test count.

Also record exact environment versions:

```bash
python --version
python -c "import supervision; print(supervision.__version__)"
python -c "import numpy; print(numpy.__version__)"
python -c "import scipy; print(scipy.__version__)"
python -c "import cv2; print(cv2.__version__)"
```

Verify current tracker import resolution:

```bash
python -c "import trackers; print(trackers.__file__)"
```

Expected concern:

EyeCU's local `trackers/` package may shadow the external Roboflow `trackers` distribution.

Do not install the external tracker package into the active project environment until you have reported:

```text
IMPORT RESOLUTION
DEPENDENCY COMPATIBILITY
SAFE ISOLATION PLAN
```

Also inspect:

```text
requirements.txt
pyproject.toml
lock files if any
```

and report whether adding modern `trackers` would change:

- NumPy;
- Supervision;
- SciPy;
- OpenCV;
- Python version requirements.

Do not upgrade dependencies yet.

---

# 7. PHASE 1 — FREEZE TRACKING INPUTS

Create a reproducible frozen-detection package for the exact four VAL continuity windows.

Suggested path:

```text
data/tracking_val_v1/
    manifest.json
    detections/
        austin_284.jsonl
        bayern_228.jsonl
        women_1_239.jsonl
        youth_1133.jsonl
```

If a better existing project convention already exists, reuse it.

The manifest must contain:

```text
source match
source video path / stable source identifier
start frame
frame count
native FPS
frame width/height
A checkpoint hash
detector config
confidence behavior
human candidate feature flag
ball settings
code commit
file hashes for frozen detection outputs
creation date
```

For every detection preserve:

```text
frame index
bbox xyxy
confidence
raw semantic class
state/provenance if present
```

Do not include tracker IDs.

Add a validator that fails if:

- wrong window;
- wrong frame count;
- TEST source;
- ball enters human-detection file intended for human tracking;
- missing frame;
- malformed bbox;
- class outside the four-class mapping;
- manifest/hash mismatch.

Once frozen, all tracker comparisons below reuse this package.

---

# 8. PHASE 2 — AUDIT THE OLD SUPERVISION WRAPPER BEFORE REPLACING IT

This is a diagnostic experiment.

Do not modify Supervision source.

Do not tune ByteTrack.

Do not change production behavior.

## 8.1 Verify installed source against the known 0.26.1 semantics

Inspect the installed package, not only online docs.

Confirm and report exact source locations for:

```text
track_activation_threshold
low-confidence floor
second-stage definition
det_thresh
new-track initialization
update_with_detections wrapper remapping
wrapper IoU assignment threshold
lost-track lifecycle
frame-rate handling
minimum_consecutive_frames
```

If installed code differs from the fact-checked assumptions above, STOP and report the difference before continuing.

---

## 8.2 Build a diagnostic runner

Create a diagnostic-only tool, e.g.:

```text
tools/audit_supervision_bytetrack_wrapper.py
```

Do not route production through it.

For each frozen frame, run two independent tracker instances from identical initial state:

### Path A — raw internal behavior

Use the exact internal tensor update path:

```text
update_with_tensors()
```

Record returned internal tracks.

### Path B — normal EyeCU/Supervision behavior

Use:

```text
update_with_detections()
```

Record public returned detections + tracker IDs.

Important:

Do NOT call both update methods sequentially on one tracker instance.

That would advance tracker state twice.

Use independent instances or an equivalent safe instrumentation method.

---

## 8.3 Measure wrapper visibility effects

For each frame record:

```text
input accepted human detections
internal output track count
wrapper output detection count
internal external_track_ids
wrapper tracker_ids
internal track tlbr
original detection xyxy
wrapper remap IoU
wrapper-remap unmatched detections
wrapper-remap unmatched internal tracks
```

Specifically count:

### WRAPPER-DROPPED ACCEPTED DETECTION

An accepted human detection existed, internal ByteTrack returned active track state, but the wrapper failed to return that detection with an ID after its remapping step.

### INTERNAL TRACK PRESENT / PUBLIC DETECTION ABSENT

Internal active track exists but no corresponding public detection survives wrapper output.

Do not call this an identity error yet.

It is a wrapper-output discrepancy.

### MAPPING AMBIGUITY

Multiple nearby detections/tracks make wrapper remapping ambiguous.

Record the full cost/IoU matrix for these cases.

---

## 8.4 Measure the hidden new-track confidence region

Count detections in:

```text
<=0.10
(0.10, 0.25)
exactly 0.25 if any
(0.25, 0.35)
>=0.35
```

For unmatched detections in `(0.25, 0.35)` record how often they cannot initialize a track because of `det_thresh`.

Do not change the thresholds.

We are measuring semantics, not fixing them.

---

## 8.5 Profile the previous +37% low-confidence experiment properly

The earlier low-confidence human experiment increased total runtime substantially.

Do not attribute that cost to ByteTrack without profiling.

On one representative frozen window, separately time:

```text
frame decode / preparation
detector inference
detector postprocessing
number of detector boxes emitted
construction of sv.Detections
tracker internal update
wrapper remapping
EyeCU conversion/writeback
other
```

Compare feature OFF vs ON.

Report:

```text
ms/frame
box counts
percentage of total runtime
```

The goal is to identify whether the added cost came mainly from:

- detector/NMS/postprocessing;
- more candidate boxes;
- tracker association;
- wrapper mapping;
- EyeCU Python handling.

Do not optimize yet.

---

## 8.6 Phase-2 conclusion rules

Allowed conclusions:

```text
wrapper contributes materially to public output loss
wrapper effect is small
new-track gate affects a meaningful number of cases
new-track gate is negligible
low-confidence runtime cost is detector/postprocess dominated
low-confidence runtime cost is tracker/wrapper dominated
```

Not allowed without identity GT:

```text
wrapper causes X% identity switches
ByteTrack is bad
new ByteTrack is better
BoT-SORT will solve it
```

Return:

```text
OLD SUPERVISION SOURCE AUDIT
WRAPPER VS INTERNAL COUNTS
WRAPPER-DROPPED CASES
NEW-TRACK GATE CASES
RUNTIME PROFILE
WHAT IS PROVEN
WHAT IS STILL UNKNOWN
```

---

# 9. PHASE 3 — BUILD A REAL IDENTITY-GT TRACKING VAL BENCHMARK

This is the most important phase.

Without this, do not select or tune trackers.

Use the same four frozen continuity windows:

```text
austin      284..583
bayern      228..527
women_1     239..538
youth       1133..1432
```

Use native frame rate.

Name the benchmark:

```text
EyeCU-Tracking-Val-v1
```

---

## 9.1 Benchmark target

Human tracking only:

```text
player
goalkeeper
referee
```

Ball excluded.

Coaches, substitutes, staff, spectators, ball boys, medical staff are not target tracks unless EyeCU's existing annotation rules explicitly include them.

Keep the benchmark consistent with current EyeCU human semantics.

Tracking evaluation itself is class-agnostic HUMAN identity.

Role is preserved separately as metadata.

---

## 9.2 Annotation package

Create:

```text
data/tracking_val_gt/
    manifest.json
    sequences/
        austin_284/
        bayern_228/
        women_1_239/
        youth_1133/
    mot/
    roles/
    qc/
```

Derived frame images may be gitignored if reproducible from source video.

The valuable annotation files/manifests must be protected and backed up.

---

## 9.3 Annotation assistance rule

It is acceptable to use **untracked detector boxes as drafts** to reduce manual box drawing.

It is NOT acceptable to use:

- ByteTrack IDs;
- BoT-SORT IDs;
- any tracker-generated identity;
- model-generated track links;

as the GT identity answer.

If detector boxes are used as draft geometry:

- remove all tracker IDs first;
- manually correct missed/false boxes;
- manually assign identity;
- manually verify occlusions/reentries.

The annotation must be treated as human-corrected GT, not detector output.

If this distinction cannot be maintained, export raw frames only.

---

## 9.4 Identity annotation rules

Each physical target receives one stable positive integer ID within a sequence.

Same physical person:

```text
same GT ID
```

across:

- temporary occlusion;
- detector confidence changes;
- semantic role prediction changes;
- short disappearance behind another player;
- reasonable camera motion within the continuous shot.

Do not assign a new GT ID simply because the current tracker would.

If a person's identity becomes genuinely impossible to determine:

- mark the event for QC;
- do not invent certainty;
- document how the final annotation resolves it.

No single frame may contain two boxes with the same GT ID.

---

## 9.5 Bounding-box rules

Use the project's existing visible-extent convention unless a benchmark-specific reason requires otherwise.

Every GT bbox must:

```text
be within image bounds
have positive width/height
represent one visible target
not duplicate another GT box
```

Do not infer a fully hidden person's bbox.

A temporary absence in GT is acceptable when the person is not visually identifiable.

The identity can resume when they become identifiable again.

---

## 9.6 Role metadata

Tracking metrics are class-agnostic HUMAN.

Still store GT role metadata separately:

```text
player
goalkeeper
referee
```

Suggested sidecar:

```text
roles/<sequence>.json
```

with:

```text
GT identity -> role
```

or per-frame role only if role genuinely varies.

Do not embed role classification into HOTA unless implementing a separately-defined semantic metric later.

---

## 9.7 MOTChallenge-format export

Create standard MOT-style GT rows:

```text
frame,id,x,y,w,h,conf,-1,-1,-1
```

Use:

```text
conf = 1
```

for GT.

Be explicit about:

- 1-based vs 0-based frame indexing;
- xywh convention;
- integer/float handling.

Create any required `seqinfo.ini` metadata.

Do not guess.

Follow TrackEval's expected format exactly.

---

## 9.8 GT validators

Add tests/tooling for:

```text
all four expected sequences present
exact expected frame ranges
no TEST sources
no duplicate ID per frame
positive bbox size
bbox inside frame
valid MOT row shape
GT confidence == 1
valid identity ids
no ball in human GT
role metadata valid
manifest hashes valid
```

Also create a visual QC renderer:

```text
GT bbox + GT ID + role
```

for fast manual inspection.

---

## 9.9 Human annotation checkpoint

Claude cannot create trustworthy identity ground truth by inference.

If identity GT does not already exist, prepare the complete annotation package and STOP at:

```text
IDENTITY GT READY FOR MANUAL ANNOTATION
```

Return exact user instructions for CVAT/annotation import/export.

Do not proceed to final tracker selection using fake or auto-tracked GT.

This is an allowed and required human checkpoint.

---

# 10. PHASE 4 — TRACKING EVALUATION HARNESS

After identity GT is completed and validated, build the evaluation harness.

Prefer standard TrackEval behavior over custom formulas.

Suggested:

```text
tools/evaluate_tracking.py
```

It may wrap TrackEval or produce correctly formatted files and invoke it.

---

## 10.1 Required primary metrics

Report:

```text
HOTA
DetA
AssA
IDF1
IDP
IDR
```

Also useful:

```text
MOTA
ID switches / CLEAR identity events
Frag
Mostly Tracked / Mostly Lost if available
```

Do not make MOTA the sole headline metric.

For EyeCU's current question, HOTA/AssA/IDF1 are more informative about identity continuity.

---

## 10.2 Sanity tests before trusting the evaluator

Create tiny synthetic sequences.

### PERFECT

GT and tracker output identical.

Expected:

```text
HOTA ~ perfect
DetA ~ perfect
AssA ~ perfect
IDF1 ~ perfect
```

### ID SWAP

Boxes are perfect but two identities swap.

Expected:

```text
DetA stays strong
AssA / IDF1 worsen
```

### DETECTION MISS

Identity is otherwise correct but boxes disappear.

Expected:

```text
DetA worsens
```

### FRAGMENT

One true identity is split into two tracker IDs.

Expected association/identity degradation.

Do not proceed until these tests behave as expected.

---

# 11. PHASE 5 — RUN TWO DIFFERENT TRACKING TASKS

This is mandatory.

## TASK A — PURE ASSOCIATION / ORACLE DETECTIONS

Input to tracker:

```text
GT human boxes
confidence = 1
```

The tracker must associate identities.

Purpose:

> Measure tracker association independently of EyeCU detector misses.

This mirrors the logic of SoccerNet's pure-association task.

Report HOTA/AssA/IDF1 per sequence and combined.

DetA should be near the ceiling because geometry is oracle; if not, investigate output filtering/lifecycle semantics.

---

## TASK B — COMPLETE EYECU TRACKING

Input:

```text
frozen A@960 human detections
```

No tracker gets a different detector run.

Evaluate against the same identity GT.

Purpose:

> Measure the real EyeCU detection + association system.

Report:

```text
HOTA
DetA
AssA
IDF1
```

per sequence and combined.

---

## 11.1 Interpretation matrix

Use this logic:

### Oracle good, end-to-end poor

```text
association algorithm can work
detector support / detector-to-tracker interface is limiting
```

### Oracle poor, end-to-end poor

```text
tracker association itself is a major limitation
```

### Oracle good and end-to-end good

```text
tracking is sufficient
```

### Oracle differs strongly between old wrapper and raw/new tracker

```text
implementation/interface layer matters materially
```

Do not jump to ReID before this decomposition exists.

---

# 12. PHASE 6 — MODERN TRACKER BAKE-OFF

Only after identity GT and evaluation sanity tests pass.

Do not tune.

Use defaults/frozen configs first.

Compare:

```text
1. current EyeCU sv.ByteTrack 0.26.1 path
2. modern ByteTrackTracker
3. BoT-SORT with its default CMC behavior/config
4. OC-SORT
5. C-BIoU, only if present in the selected stable package version
```

If a tracker is not available in the pinned stable release, report it.

Do not silently install a development branch just to include it.

---

## 12.1 ISOLATE THE EXTERNAL `trackers` PACKAGE

Because EyeCU already owns a local `trackers/` package, do not perform the first comparison by importing the external package from the EyeCU repo root.

Preferred design:

```text
EyeCU repo
    |
    | exports frozen detections + frames + manifest
    v
isolated evaluation environment / working directory
    |
    | imports external Roboflow `trackers`
    v
tracker outputs in MOT format
    |
    v
EyeCU evaluation harness / TrackEval
```

Pin:

```text
Python version
external trackers version
supervision version
numpy/scipy/opencv versions
```

Record environment in a machine-readable file.

Do not alter EyeCU production dependencies during the bake-off.

---

## 12.2 Exact same inputs

All modern trackers receive:

```text
same frames
same detections
same confidence
same image coordinates
same source FPS
same frame order
```

For BoT-SORT / CMC, pass actual frame pixels only if required by the documented API.

Do not give one tracker more information than another except information intrinsically required by that algorithm, and document it.

---

## 12.3 Frame rate

Use the actual source FPS.

Do not assume 30 FPS.

Audit how each implementation scales:

```text
lost track buffer
motion prediction
time-related lifecycle
```

when source video is 25 FPS.

Report exact effective values.

---

## 12.4 No copied tuned parameters

Do not copy:

```text
SoccerNet tuned config
SportsMOT tuned config
MOT17 tuned config
DanceTrack tuned config
```

Those are different detection/domain conditions.

The first EyeCU bake-off uses library defaults, with only unavoidable input metadata such as actual FPS set correctly.

---

## 12.5 Selection metrics

Primary:

```text
HOTA
AssA
IDF1
```

Secondary:

```text
DetA
Frag
ID-switch-related metrics
runtime
memory
proxy IDs/human
track length
```

Report both:

```text
oracle-detection task
end-to-end A-detection task
```

---

## 12.6 Consistency requirement

Do not choose a tracker because of one lucky sequence.

A production replacement must show:

- better combined identity performance;
- improvement or at least non-material regression across most sequences;
- no catastrophic sequence-specific failure;
- acceptable runtime;
- preserved EyeCU semantics.

Before running the bake-off, write the exact numeric adoption criteria to:

```text
experiments/tracking_v2/adoption_criteria.json
```

Freeze that file before results are generated.

Suggested starting criteria to evaluate and freeze BEFORE results:

```text
combined HOTA improvement >= 2 absolute points
combined AssA improvement >= 3 absolute points
IDF1 does not regress
improves HOTA in >=3 of 4 sequences
no sequence HOTA regression >3 absolute points
accepted human detection semantics preserved
ball remains isolated
end-to-end runtime regression <=10% unless accuracy gain is compelling and explicitly accepted
```

If you believe a suggested threshold is methodologically inappropriate, explain why and replace it BEFORE running competitor results.

Never alter criteria after seeing the bake-off.

---

# 13. PHASE 7 — DECISION TREE AFTER THE BAKE-OFF

## CASE A — old wrapper is the main implementation problem

If internal old ByteTrack behaves materially better than `update_with_detections()` public output, and identity GT confirms the wrapper path damages metrics:

Do not monkeypatch site-packages.

Prefer:

- migrate to a cleaner tracker API; or
- write a minimal project-owned adapter with explicit semantics.

Any production change must be feature-flagged initially and tested.

---

## CASE B — modern ByteTrack wins

If modern `ByteTrackTracker` clearly beats the old path:

Production migration is justified.

But first solve the namespace collision cleanly.

Do not use fragile `sys.path` insertion.

Propose the smallest maintainable integration plan and wait for approval before renaming large packages.

---

## CASE C — BoT-SORT wins

If BoT-SORT wins, verify whether the gain depends on camera-motion compensation.

Run at most one explanatory ablation:

```text
BoT-SORT CMC ON
vs
BoT-SORT CMC OFF
```

Same defaults otherwise.

This is an interpretation ablation, not hyperparameter tuning.

If CMC is the gain, record that explicitly.

---

## CASE D — C-BIoU wins

If C-BIoU wins, record whether the gain is concentrated in:

```text
fast motion
low overlap
crowded contact
camera movement
```

Do not immediately tune buffer ratios.

Default win first.

---

## CASE E — OC-SORT wins

If OC-SORT wins, record whether improvement is concentrated in non-linear motion/reacquisition cases.

Do not tune direction-consistency parameters yet.

---

## CASE F — all trackers are similar

STOP tracker architecture work.

Record:

```text
association algorithm choice was not the binding limitation under the measured benchmark
```

Then move to the next system bottleneck.

---

## CASE G — oracle tracking is strong, complete tracking remains weak

Do not keep swapping trackers.

The detector/support interface is limiting.

Document this and return to detector-continuity work only if the coursework/system goals require it.

Do not reopen broad detector architecture research automatically.

---

## CASE H — even oracle association is weak across trackers

Only then consider:

```text
appearance/ReID
tracklet merging
offline/global association
```

Do not implement them in this TODO.

Return a proposal first.

---

# 14. ROLE SMOOTHING — DEFER UNTIL TRACKER IS CHOSEN

Current evidence suggests raw semantic role jitter is real.

But role smoothing is a separate experiment.

After the tracker is frozen:

1. map tracker outputs to GT identities;
2. measure raw role accuracy;
3. measure raw role flips per GT identity;
4. only then test one conservative smoothing method.

Allowed candidate later:

```text
confidence-weighted rolling vote
```

Do not implement it during tracker bake-off.

Otherwise two variables change at once.

---

# 15. CACHE + REPRODUCIBILITY REQUIREMENTS

Any production tracker backend/config must enter the cache key.

At minimum:

```text
tracker backend name
tracker implementation/package version
tracker config
source FPS / effective FPS if relevant
detector checkpoint hash
detector confidence settings
human candidate pool flag
human candidate floor
accepted human threshold
skip_frames
video identity
```

If camera-motion compensation depends on frame pixels, include any setting that changes it.

A cache from one tracker must never be reused by another tracker.

Add tests.

---

# 16. PRODUCTION OUTPUT INVARIANTS

Whatever tracker eventually wins:

- player remains player semantically;
- goalkeeper remains goalkeeper semantically;
- referee remains referee semantically;
- ball remains separate;
- low-confidence association evidence does not become public detection merely because the tracker used it;
- accepted output thresholds remain explicit;
- team assignment does not receive referees/GKs incorrectly;
- reports label track IDs as track IDs, not physical-player counts;
- speed/distance remains marked uncalibrated until homography/calibration is solved.

---

# 17. SECURITY + PUBLIC REPO SAFETY

Do not push during this task unless explicitly asked.

Before any eventual push:

```text
git status
git diff
git log
secret scan / grep for key-like literals
```

There is historical project context involving an exposed API credential.

Do not print any secret value in terminal summaries or responses.

If found:

```text
[REDACTED SECRET]
```

and report the file/path only.

Make sure:

```text
.env
virtualenv
model caches
temporary evaluation environments
generated frames
large reproducible exports
```

are ignored appropriately.

Do not rewrite git history in this TODO unless explicitly approved.

---

# 18. REQUIRED TESTS

Keep existing tests green.

Add narrowly-scoped tests as phases require.

## Wrapper audit

- independent tracker instances do not double-advance state;
- diagnostic instrumentation does not alter behavior;
- exact 0.10 / 0.25 / 0.35 boundary semantics are captured;
- wrapper remap accounting sums correctly.

## Frozen detection package

- four exact windows;
- no TEST;
- manifest hashes;
- correct classes;
- no ball in human stream;
- correct FPS/frame count.

## Tracking GT

- MOT format validity;
- no duplicate ID in frame;
- valid bboxes;
- correct sequence/frame ranges;
- no TEST;
- role sidecar validity.

## Evaluator

- perfect synthetic case;
- identity swap case;
- missed-detection case;
- fragmentation case.

## Tracker adapter / bake-off

- identical serialized inputs;
- tracker reset between sequences;
- actual FPS passed;
- ball excluded;
- raw semantic metadata preserved outside association;
- cache key differs by backend/config/version.

## Production migration, only if later approved

- feature flag OFF reproduces baseline;
- accepted-output volume semantics preserved;
- all four detector classes preserved;
- goalkeeper not normalized;
- ball isolated;
- CLI smoke test;
- full fast regression suite.

---

# 19. FILE / FOLDER ORGANIZATION FOR THIS WORK

Do not dump new diagnostics in repo root.

Suggested:

```text
experiments/
    tracking_v2/
        README.md
        adoption_criteria.json
        wrapper_audit/
        runtime_profile/
        bakeoff/
        figures/

data/
    tracking_val_v1/
    tracking_val_gt/

tools/
    freeze_tracking_detections.py
    validate_tracking_val.py
    audit_supervision_bytetrack_wrapper.py
    build_tracking_gt_package.py
    validate_tracking_gt.py
    evaluate_tracking.py

tests/
    test_tracking_val_freeze.py
    test_tracking_gt.py
    test_tracking_eval.py
```

Reuse existing equivalent tools if present.

Do not create duplicate scripts just because these names are suggested.

---

# 20. DOCUMENTATION OWNERSHIP

Keep documentation canonical.

After each completed phase, update the appropriate document instead of creating another random root `.md`.

Target ownership:

```text
README.md
    project entry point

docs/results/RESULTS.md
    measured final/experimental results

docs/coursework/COURSEWORK_PLAN.md
    what remains

docs/guides/
    operational guides

docs/research/
    research rationale

experiments/tracking_v2/
    tracking experiment specification + records

docs/archive/
    superseded plans
```

Do not create:

```text
check3.md
new_todo.md
final_final.md
tracking2.md
```

---

# 21. WHAT NOT TO DO

Do NOT:

- access TEST;
- retrain detector A/B/C;
- implement P2;
- tune detector thresholds;
- tune ByteTrack before identity GT;
- run Optuna before the default tracker bake-off;
- copy tuned SoccerNet parameters;
- copy tuned SportsMOT parameters;
- add ReID yet;
- add appearance embeddings yet;
- add tracklet merging yet;
- implement role smoothing yet;
- merge goalkeeper into player;
- send ball through human MOT;
- call IDs/human HOTA;
- call bbox-jump proxies identity switches;
- use tracker outputs as GT identity;
- patch installed Supervision source in site-packages;
- use `sys.path` hacks to defeat the `trackers` namespace collision;
- upgrade the active environment just to test the new package;
- push to GitHub during the investigation;
- delete the rejected low-confidence experiment/tests needed for reproducibility.

---

# 22. CHECKPOINT / EXECUTION ORDER

Follow this order.

```text
PHASE 0
repo/environment audit
        |
        v
PHASE 1
freeze A detections
        |
        v
PHASE 2
old Supervision wrapper/core audit + runtime profile
        |
        v
PHASE 3
build identity-GT annotation package
        |
        v
MANUAL ANNOTATION CHECKPOINT
        |
        v
validate identity GT
        |
        v
PHASE 4
TrackEval harness + synthetic tests
        |
        v
PHASE 5
oracle-association baseline
+
complete-tracking baseline
        |
        v
PHASE 6
isolated default tracker bake-off
        |
        v
PHASE 7
choose / stop / propose next architecture
        |
        v
freeze tracking
        |
        v
role smoothing only if still needed
        |
        v
calibration
        |
        v
FINAL TEST once
```

Do not skip the GT checkpoint.

---

# 23. STOP RULES

## Stop immediately and report if:

- installed Supervision behavior differs from the assumed 0.26.1 code;
- a frozen window resolves to TEST;
- frozen detection hashes do not match;
- source FPS is uncertain;
- external `trackers` import resolves to EyeCU's own package;
- dependency installation would mutate/break the active environment;
- GT identity is not manually verified;
- TrackEval sanity tests fail.

---

## Freeze tracking when:

A tracker/system configuration has:

- valid identity-GT evaluation;
- acceptable HOTA/AssA/IDF1;
- no catastrophic match-specific regression;
- acceptable runtime;
- correct ball isolation;
- preserved raw role semantics;
- reproducible configuration;
- no unresolved wrapper ambiguity that changes the conclusion.

Do not keep optimizing for marginal validation gains after this point.

---

# 24. REQUIRED CLAUDE RESPONSE STYLE

Do not narrate every shell command.

Do not paste giant source files.

At each phase return:

```text
PHASE
STATUS

FILES CHANGED
- ...

MEASUREMENTS
- ...

TESTS
- ...

WHAT IS PROVEN
- ...

WHAT IS NOT PROVEN
- ...

RISKS / BLOCKERS
- ...

NEXT ACTION
- ...
```

If a human annotation checkpoint is reached, stop and give concise exact instructions.

---

# 25. FINAL DELIVERABLE AFTER THE TRACKER BAKE-OFF

Return one final report with exactly these sections:

```text
REPOSITORY / ENVIRONMENT TRUTH

OLD SUPERVISION WRAPPER AUDIT

WRAPPER VS INTERNAL EFFECT

LOW-CONFIDENCE RUNTIME PROFILE

IDENTITY-GT BENCHMARK

TRACKING EVALUATOR VALIDATION

CURRENT SV.BYTETRACK — ORACLE

CURRENT SV.BYTETRACK — END TO END

MODERN TRACKER BAKE-OFF

PER-SEQUENCE HOTA / DETA / ASSA / IDF1

RUNTIME

ROLE-JITTER OBSERVATION

INTERPRETATION

SUPPORTED CONCLUSIONS

UNSUPPORTED CONCLUSIONS

PRODUCTION DECISION

TRACKING FREEZE / DO NOT FREEZE

NEXT SINGLE SYSTEM ACTION
```

The production decision must be one of:

```text
KEEP CURRENT TRACKER
MIGRATE TO MODERN BYTETRACK
MIGRATE TO BOT-SORT
MIGRATE TO OC-SORT
MIGRATE TO C-BIOU
NO TRACKER CHANGE — DETECTOR/INTERFACE LIMITED
NO TRACKER CHANGE — NEED LONG-TERM APPEARANCE/REID STUDY
INCONCLUSIVE — STATE EXACT BLOCKER
```

Do not invent a middle answer.

---

# 26. FACT-CHECKED REFERENCE SET

Use these as the external technical references for this phase.

## Supervision 0.26.1 ByteTrack source documentation

https://supervision.roboflow.com/0.26.1/trackers/

Relevant verified behaviors:

- strict high/low confidence split;
- `det_thresh = track_activation_threshold + 0.1`;
- `update_with_detections()` calls internal tracking and then performs an additional IoU-based remapping;
- unmatched wrapper detections are removed from returned output.

---

## Supervision current deprecation / migration documentation

https://supervision.roboflow.com/develop/trackers/

https://github.com/roboflow/supervision/releases

Relevant fact:

- `sv.ByteTrack` is deprecated in favor of `ByteTrackTracker` from the external `trackers` package.

Do not depend on an exact future removal version unless re-checking the current release notes, because documentation can change.

---

## Roboflow Trackers

https://github.com/roboflow/trackers

https://trackers.roboflow.com/

https://trackers.roboflow.com/latest/trackers/comparison/

Relevant facts:

- external import namespace is `trackers`;
- detector-agnostic;
- common `supervision.Detections` interface;
- modern ByteTrack / OC-SORT / BoT-SORT / C-BIoU implementations;
- benchmarks are available across sports/general MOT datasets;
- benchmark methodology differs by dataset.

---

## SoccerNet Tracking

https://github.com/SoccerNet/sn-tracking

https://www.soccer-net.org/tasks/tracking

Relevant facts:

- pure association task with GT detections;
- complete detection + association task;
- HOTA primary benchmark concept;
- DetA / AssA decomposition.

---

## TrackEval

https://github.com/JonathonLuiten/TrackEval

https://github.com/SoccerNet/sn-trackeval

Relevant facts:

- standard HOTA/DetA/AssA/IDF1/CLEAR metrics;
- supports custom benchmarks;
- MOTChallenge format is recommended for custom 2D box tracking evaluation.

---

## SportsMOT

https://openaccess.thecvf.com/content/ICCV2023/html/Cui_SportsMOT_A_Large_Multi-Object_Tracking_Dataset_in_Multiple_Sports_Scenes_ICCV_2023_paper.html

Relevant facts:

- sports MOT includes fast/variable motion and similar appearances;
- association is a key sports-tracking challenge.

---

## BoT-SORT

https://arxiv.org/abs/2206.14651

Relevant fact:

- combines stronger motion/association with camera-motion compensation and appearance capability.

For the first EyeCU bake-off, do not add external ReID unless the chosen implementation's default tested mode requires it and it is explicitly documented.

Prefer CMC-only interpretation before adding appearance complexity.

---

## C-BIoU

https://openaccess.thecvf.com/content/WACV2023/html/Yang_Hard_To_Track_Objects_With_Irregular_Motions_and_Similar_Appearances_WACV_2023_paper.html

Relevant facts:

- buffered/cascaded geometric matching;
- designed for irregular motion / similar appearances;
- reported as a dominant component in a second-place SoccerNet MOT solution.

---

# 27. THE CORE QUESTION

Do not lose sight of this:

> **EyeCU does not need the fanciest tracker. It needs the simplest tracker whose identity quality is demonstrably good enough on EyeCU footage.**

The correct sequence is:

```text
understand wrapper
        ->
build identity GT
        ->
measure current tracker correctly
        ->
compare modern alternatives on identical inputs
        ->
choose one
        ->
stop
```

Not:

```text
change threshold
-> inspect proxy
-> change threshold again
-> change tracker
-> inspect proxy
-> add ReID
-> hope
```

The goal of this TODO is to remove that ambiguity permanently.
