# Identity GT Annotation Guide

How to annotate the three 300-frame tracking sequences of
**EyeCU-Tracking-Val-v1.1** so that HOTA/IDF1 numbers computed against them mean
something.

> A fourth window (austin) was proposed in v1.0 and removed before any tracker
> ran: its window straddles a broadcast dissolve and its source turned out to be
> a highlights montage. It is preserved as a transition-stress case in
> `data/tracking_val_gt/rejected/` and never enters the clean aggregate.
>
> Order: women_1 (accepted) -> youth_premier_league_1133 -> bayern_munich_3-1_chelsea_228.

This benchmark answers one question: **does a tracker keep the same person under
the same identity?** Box geometry is secondary — a slightly loose box costs a
little localisation accuracy, but a swapped identity corrupts the association
metrics the entire investigation depends on.

## What exists before you start

```
data/tracking_val_gt/
  sequences/<seq>/img1/000001.jpg .. 000300.jpg   frames, 1-based  (canonical)
  sequences/<seq>/seqinfo.ini                     MOT sequence header
  cvat_video/<seq>.mp4                            what you upload to CVAT
  cvat_video/<seq>.provenance.json                its verified frame mapping
  preannotations/<seq>.cvat.xml                   detector boxes, NO identity
  preannotations/<seq>.det.txt                    same, MOT det format (id = -1)
  manifest.json                                   identity_gt_status: UNANNOTATED
```

The preannotations are detector output. They carry **no identity** by
construction — the validator refuses the package if a `<track>` element or a
non-`-1` id column appears in them, because identity that came from a tracker
cannot be used to score trackers.

## Upload the MP4, never the JPEG folder

CVAT picks the annotation mode from what you upload:

| Upload | CVAT mode | Objects you can create |
|---|---|---|
| **video** | interpolation | **tracks** — what this benchmark needs |
| images | annotation | shapes — no identity, ever |

Our importer reads `<track>` semantics. A task made from `img1/` is an image
task, and no amount of care during annotation recovers identity from it. So
upload `cvat_video/<seq>.mp4`.

The MP4 is a **UI vehicle only**. Canonical provenance stays with the frozen
package: `img1/000001.jpg` is package frame 1, which is source frame
`source_frame_range[0]`. Nothing measured comes from the MP4's pixels — it
exists so CVAT offers track mode.

Each clip is built by [build_cvat_video_clips.py](../../tools/build_cvat_video_clips.py)
from the frozen frames and then verified by decoding it back: exact frame
count, decoded frame 0 = package frame 1, every decoded frame matching its own
package frame better than its neighbours, no duplicated frames, 640×360, and
the container frame rate equal to the source's exact rational rate (taken from
`ffprobe r_frame_rate`, not reconstructed from a rounded decimal). The record
lands in `<seq>.provenance.json`; a clip that fails verification is deleted
rather than shipped.

## Preannotations are reference only

Do **not** import `preannotations/<seq>.cvat.xml` into the annotation task.

Two reasons:

1. The file is a shape-mode export — 300 `<image>` blocks, 18,269 boxes in
   total, no `<track>` elements. Loading shapes into a track-mode task gives
   you thousands of identity-less shapes that must each be converted by hand;
   the cleanup costs more than drawing tracks from scratch. (CVAT's own import
   behaviour here has not been verified in this environment, which is itself a
   reason not to bet the annotation session on it.)
2. Seeding tracks from detector geometry makes the GT partly an echo of the
   detector being evaluated. `validate_tracking_gt.py` already refuses GT that
   is byte-identical to the preannotation; keeping the two independent avoids
   the softer version of the same problem.

Use them as a **cross-check**: open the XML if you want to see where the
detector fired, or compare counts afterwards. Nothing in the pipeline reads
them as annotation input.

## 42 frames have existing labels — reference only, provenance unconfirmed

42 of the 1,200 frames were already labelled during detector annotation: 560
boxes (481 player, 22 goalkeeper, 57 referee; 28 ball boxes dropped), in
`human_seed/<seq>.json` as absolute pixels on the 640×360 package frames.

**These are not ground truth.** They began as output of one Roboflow model and
were later modified — 90 boxes added, 32 classes changed — by a process this
repository does not record. The label pipeline has no reviewed flag, only
`status: draft`. Until someone can say how those edits were reviewed, the file
is marked `provenance_status: UNCONFIRMED` and must not be adopted as tracking
GT.

Overlap itself *is* confirmed: source-frame provenance, then a pixel check in
which each frame beat its neighbours by at least 2.3×. It is the labels, not
the frame identification, that are uncertain.

Use them as a visual reference while annotating, and afterwards as a
cross-check:

```bash
python tools/check_human_seed_agreement.py
```

It reports where the two passes differ. Report-only, and your annotation is the
authority — a disagreement is a prompt to look, not a correction to apply.

The seed carries **no identity**. Nothing from the detector era can.

## Before the real work: a 10-minute CVAT smoke test

The synthetic fixture in `tests/` proves the parser handles the XML we believe
CVAT writes. It cannot prove your CVAT version and export dialog produce that
XML. A version that numbers frames differently or writes tracks as shapes would
pass every unit test and silently corrupt 1,200 frames of annotation.

1. **Create task** → name `smoke` → labels `player`, `goalkeeper`, `referee` →
   **Select files** → upload the single file
   `data/tracking_val_gt/cvat_video/women_1_239_smoke10.mp4` → Submit.
   It is 10 frames of the least crowded sequence (~11 people per frame).
   Because it is a video, CVAT opens it in interpolation mode.
2. Open the job. Annotate **2 tracks** with **different roles** — press **N**
   with a track label selected, and make sure the object is a *Track*, not a
   Shape.
3. Track A: box it on frame 0, move to frame 4 and adjust the box (that makes a
   keyframe and CVAT interpolates 1–3); on frame 5 mark it **outside**; on
   frame 7 make it visible again — **same track**, do not create a new one —
   and adjust it once more on frame 9.
4. Track B: box it on frame 0, adjust on frame 9, visible throughout.
5. **Menu → Export job dataset → CVAT for video 1.1**, then:

```bash
python tools/smoke_test_cvat_export.py --export path/to/smoke.xml
```

It runs the real export through import → canonical JSON → validator → QC
renderer → MOT export on a scratch copy, and checks 25 properties: CVAT frame 0
becomes package frame 1, identities stay stable, roles survive, outside frames
emit no box, the reappearance keeps the same identity, interpolation decodes,
boxes are in-frame, no duplicate ids, MOT `conf=1` and `class=1`. The real
package is never written to.

If it fails, **stop** — the importer or the export settings need fixing before
any full annotation.

## Box convention

**Tight visible-person extent with a small practical tolerance at this
resolution; no deliberate horizontal safety margin; no shadow; do not infer
fully hidden body extent; a partially visible person is annotated
consistently.**

The authority is the image. GT is never tuned for IoU agreement with the
detector under evaluation, in either direction — a benchmark adjusted to
flatter the thing it measures has stopped measuring it.

A QA audit of women_1 found GT boxes about 1.39x wider and 1.15x taller than
the detector's, with centres coincident. At these target sizes that is roughly
**4.6 px of total width** on 13–18 px people, so read the ratio with the
absolute number beside it. That is a recorded observation, not a reason to
resize anything: automatic resizing is forbidden, and women_1 stands unless
human visual QC finds clearly excessive background margin. Full record in
[experiments/tracking_v2/gt_conventions.json](../../experiments/tracking_v2/gt_conventions.json).

## When someone disappears and comes back

**A long-gap reconnect is never accepted automatically.** Every such event is
recorded `HUMAN_REVIEW_REQUIRED` and stays there until a person decides.

- Same physical person confidently established across the gap → **keep the same
  identity**.
- Genuinely uncertain → **do not guess**. Start a new GT identity and record an
  uncertain-reentry QC event.
- Role or team kit matching is **not** proof of physical identity. Twenty-two
  people wear two kits.
- Tracker output and appearance embeddings **must not** inform the decision.
  The benchmark exists to judge trackers; GT built from one would be marking
  its own homework.

Review aids, per identity:

```bash
python tools/render_identity_qc_clips.py --sequence <seq> --ids 16,14,12
```

~1 s before the disappearance and after the reappearance, target highlighted,
others dimmed. The reported jump is **image-space** displacement, and the
camera moves during these gaps — it is not how far the person walked.

## The workflow

0. **Build the annotation clips** (after the smoke test passes):

   ```bash
   python tools/build_cvat_video_clips.py
   ```

   Four verified 300-frame MP4s in `data/tracking_val_gt/cvat_video/`.

1. **Create one CVAT task per sequence.** Name it exactly the sequence tag
   (e.g. `bayern_munich_3-1_chelsea_228`) and upload the single file
   `cvat_video/<seq>.mp4`. **Not** the `img1/` folder — that makes an image
   task, which cannot hold identity.
2. **Create exactly three labels:** `player`, `goalkeeper`, `referee`.
   No `ball`, no other labels — the importer rejects anything else.
3. **Annotate in track mode**, not shape mode. Every person is one track that
   persists across the whole sequence. The track id *is* the identity.
4. **One track per person.** If the same person leaves and returns, reuse their
   existing track — do not start a new one. A new track id means "a different
   person" to every metric downstream.
5. **Mark disappearances with `outside`**, not by deleting boxes. When someone
   leaves frame or is fully hidden, set `outside` on that frame; set it back
   when they return. No box is emitted while a track is outside.
6. **Occluded but visible stays annotated.** `occluded=1` with a box is the
   right annotation for a partly hidden player; `outside` is for *not visible
   at all*. Guessing a box for someone you cannot see invents GT. The
   `occluded` flag is preserved into the canonical JSON as a boolean — mark it
   honestly, it is kept.
7. **Keyframe sparsely, check densely.** Set keyframes where motion changes;
   CVAT interpolates linearly between them and the importer reproduces that
   exactly. Then scrub the whole sequence and fix drift.
8. **Set the role on the track, once.** The role comes from the CVAT track
   label and is the single source of truth — there is no separate role file to
   fill in, and the importer refuses a track whose role is inconsistent.
9. **Export as `CVAT for video 1.1`** — not "CVAT for images", not MOT. Save to
   `data/tracking_val_gt/cvat_exports/<seq>.xml` (create that directory; like
   the rest of `data/`, it is not tracked in git). Video format is the only one
   that carries identity *and* label in one file.
10. **Import the finished sequences:**

    ```bash
    python tools/import_tracking_gt_cvat.py --dry-run    # inspect first
    python tools/import_tracking_gt_cvat.py
    python tools/validate_tracking_gt.py --stage post
    ```

    Frame conversion (CVAT 0-based → package 1-based) and identity conversion
    (CVAT id 0 → identity 1) happen here. Never convert anything by hand.
11. **Review the QC render, then confirm:**

    ```bash
    python tools/render_tracking_gt_qc.py --root data/tracking_val_gt
    python tools/confirm_tracking_gt_qc.py --reviewer "<your name>"           # report
    python tools/confirm_tracking_gt_qc.py --reviewer "<your name>" --confirm # promote
    python tools/validate_tracking_gt.py --stage final
    ```

## Status is a three-state machine

```
UNANNOTATED  --import-->  ANNOTATED_PENDING_QC  --human QC-->  VERIFIED
```

TrackEval and the MOT export refuse to run on anything but `VERIFIED`.

Promotion writes `qc/qc_confirmation.json` containing the SHA-256 of every
annotation and role file, and the export re-checks those hashes. So editing
`identity_gt_status` in the manifest by hand does not make the benchmark
evaluable, and editing an annotation after confirming silently invalidates the
confirmation rather than silently changing the answer key. A benchmark becomes
usable by evidence, not by assertion.

## When it is VERIFIED

```bash
python tools/export_tracking_gt_mot.py
```

writes `data/tracking_val_gt/mot/` in the layout TrackEval 1.3.0 reads:
`frame,id,x,y,w,h,conf,class,visibility`, 1-based frames, `conf = 1`,
`class = 1` (pedestrian). The commonly quoted `...,conf,-1,-1,-1` row is the
*tracker prediction* convention — used for GT it sets class to -1 and TrackEval
rejects it.

## Common mistakes

| Mistake | What it costs |
|---|---|
| New track id after an occlusion | Fabricates an ID switch the tracker never made |
| Deleting boxes instead of `outside` | GT claims the person was there; recall looks worse than it is |
| Uploading the `img1/` folder | Makes an image task; shapes only, no identity |
| Annotating in shape mode | No identity at all; the importer refuses the export |
| Exporting "CVAT for images" | Same — the importer detects it and tells you to re-export |
| Importing the preannotations as a starting point | GT becomes an echo of the detector under test |
| Treating `human_seed/` as GT | Its provenance is unconfirmed; it is reference only |
| Editing the manifest status by hand | Blocked; the QC record hashes will not match |
