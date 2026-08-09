# Identity GT Annotation Guide

How to annotate the four 300-frame tracking sequences so that HOTA/IDF1 numbers
computed against them mean something.

This benchmark answers one question: **does a tracker keep the same person under
the same identity?** Box geometry is secondary — a slightly loose box costs a
little localisation accuracy, but a swapped identity corrupts the association
metrics the entire investigation depends on.

## What exists before you start

```
data/tracking_val_gt/
  sequences/<seq>/img1/000001.jpg .. 000300.jpg   frames, 1-based
  sequences/<seq>/seqinfo.ini                     MOT sequence header
  preannotations/<seq>.cvat.xml                   detector boxes, NO identity
  preannotations/<seq>.det.txt                    same, MOT det format (id = -1)
  manifest.json                                   identity_gt_status: UNANNOTATED
```

The preannotations are detector output. They carry **no identity** by
construction — the validator refuses the package if a `<track>` element or a
non-`-1` id column appears in them, because identity that came from a tracker
cannot be used to score trackers.

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

## The workflow

1. **Create one CVAT task per sequence.** Name it exactly the sequence tag
   (e.g. `bayern_munich_3-1_chelsea_228`). Upload
   `data/tracking_val_gt/sequences/<seq>/img1/` as an image sequence, ordered by
   filename.
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
   at all*. Guessing a box for someone you cannot see invents GT.
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
10. **Import all four sequences:**

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
| Annotating in shape mode | No identity at all; the importer refuses the export |
| Exporting "CVAT for images" | Same — the importer detects it and tells you to re-export |
| Importing the preannotations as a starting point | GT becomes an echo of the detector under test |
| Editing the manifest status by hand | Blocked; the QC record hashes will not match |
