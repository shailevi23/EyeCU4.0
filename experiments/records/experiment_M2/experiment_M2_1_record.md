# Experiment M2.1 — calibration validity repair — final disposition

```
STATUS      CLOSED
VERDICT     B — M2 METRIC CALIBRATION NOT VALIDATED
```

The original M2 record (`experiment_M2_record.md`) is **not overwritten**.
This is an explicit erratum superseding its validation claims for both
segments. There is no M2.2.

## 1. Defect confirmed, numerically, before anything else

`bayern_correspondences_raw.json`:

```
ellipse_fit centre                = (442.4499816894531, 173.5367889404297)
held_out_landmark_image "(0,0)"   = (442.44998168945295, 173.53678894042966)
```

Identical to floating-point precision. Root cause: `conic_centre()` in
`tools/m2_circle_correspondences.py` computed the pole of the line
`(0,0,1)` — the IMAGE's own coordinate line at infinity — with respect to
the fitted conic. That formula is correct for finding the Euclidean centroid
of an ellipse **as literally drawn in the image**, which is exactly what
`cv2.fitEllipse` already returns directly; it is not, in general, the image
of a circle's true metric centre under a projective homography, because a
general homography does not fix the image's own line at infinity — it maps
the pitch plane's true line at infinity to some other, generally finite,
line (the plane's vanishing line), which was never computed or used.

Independent confirmation, geometric rather than algebraic: the world points
`axis_a(+9.15)`, `(0,0)`, `axis_a(-9.15)` are one diameter of the centre
circle and are collinear by definition; a homography preserves collinearity,
so their images must be too. Measured perpendicular distance of the claimed
`(0,0)` from the line through the two `axis_a` points: **24.3 px**, against
a line whose own visible span is 70.6 px — a **34% deviation**, large and
unambiguous, not floating-point noise. (`tests/test_pitch_calibration.py::
TestM2_1CentreDefect::test_withdrawn_bayern_centre_violates_collinearity`
locks this in as a permanent regression check.)

**Withdrawn, effective immediately:**
- `bayern`: the 1.6657 m held-out-landmark error (it was never independent —
  see below — and is now known to be measuring a defect, not calibration
  precision).
- The full speed-error budget table derived from it (11.8 / 2.4 / 1.0 m/s
  at 5/25/60-frame windows). Do not reuse any of those numbers.
- `women`: its 0.0 m held-out error was already flagged
  `held_out_landmark_independent: false` in the original M2 record — it is
  now additionally known that **all 4** of its fit points (not just the
  centre check) rest on the same defective centre, since
  `reference_line_mode: direction` builds both axes through it. Fully
  compromised, more so than bayern's (bayern's `axis_a` pair came from a
  genuine, unrelated line-conic intersection and remains geometrically
  sound on its own).

Full machine-readable status:
`experiments/records/experiment_M2/calibration/M2_1_CALIBRATION_STATUS.json`.
Neither original artifact file was edited — both are preserved verbatim as
evidence.

## 2. What was retained unchanged

Per instruction, not rewritten: `CalibrationStore`, artifact SHA256 hashing
and tamper-detection, segment frame-range boundaries and their inclusive
semantics, the bottom-centre player ground-point convention, fail-closed
`None` behaviour outside a segment, the refusal to chain two different
segments in one speed query, the refusal to import/use
`pixels_per_meter`/`SpeedDistanceEstimator`, and real-FPS handling. None of
this depended on the specific (defective) homography values — it is
infrastructure, not a claim about this footage — and all 25 pre-existing
tests plus 4 new ones (29 total) still pass unmodified in their assertions
about that infrastructure.

## 3. Attempt to repair with independent geometry (bayern only, frames 1–60)

Per scope, only `bayern_munich_3-1_chelsea_228` was attempted; `women_1_239`
was correctly left alone since bayern did not succeed (the conditional for
extending to women — "exactly the same corrected method... essentially for
free" — was never reached).

**M3 correction to this section:** the paragraph originally here claimed that
a single well-fit image line's own 2D slope directly gives its vanishing
point, as the image-affine point at infinity `(dir_x, dir_y, 0)` in that
direction, with no second line needed. **That claim was itself wrong, and
wrong in the same way the withdrawn centre construction was wrong.** A
single observed image line constrains the vanishing point of the
corresponding world-line direction to lie somewhere ON that image line — it
does not, by itself, determine WHERE on that line the true perspective
vanishing point sits. The image-affine point at infinity associated with a
line's on-screen orientation is not generally the finite vanishing point a
general (non-affine) homography produces for that world direction; treating
the two as the same thing is exactly the "image's own coordinate infinity
vs. the plane's true vanishing line" confusion section 1 of this record
already identified for the ellipse centroid. Finding an actual vanishing
point still requires either two or more lines from the same real-world
direction family (their intersection), or some other independent metric
constraint — there is no shortcut through a single line's slope. This does
**not** change the practical finding: no second reliable line (of either
family) was available in frames 1–60 regardless of which correct method
would have been used to combine it with the first (see the evidence table
below), so the identifiability gate answer and the verdict are unchanged.

**Evidence actually gathered, all measured, none assumed:**

| candidate line | family | result |
|---|---|---|
| halfway line | F1 | excellent — already in use, high pixel count, tight robust fit |
| centre circle | (scale) | excellent — already in use |
| near touchline (bottom of frame, x<330) | F2 | measured: 266 pixels, robust-fit residual median **2.5 px**, p90 **13.1 px**, max **13.7 px** against a line whose own visible span is only ~330 px. An order of magnitude noisier than the halfway-line/circle fits. Not trustworthy as an independent constraint at this resolution. |
| near touchline (x>330) | F2 | no usable signal at all — per-column brightness scores went negative (below grass baseline) |
| far goal-line (upper-left box, native ~y 50-62) | F1 | contaminated: its detection ROI merges with the adjacent advertising-board brightness into one blob (component fill 0.61, indistinguishable from real line pixels by colour/shape alone at this scale) |
| far box/corner edge (diagonal segment near the far box) | F1 or F2 (ambiguous) | too small (order of 50-70 px total extent) and too low-contrast for a confident family assignment or a precise direction fit; visually plausible as a box side-line (F2) by topological reasoning (it meets the goal-line at roughly a right angle heading into the pitch) but not independently confirmable at this resolution |

Manual visual reads used: 4 (the cap), covering: (1) full reference-frame
grid overview, (2) bottom-left corner zoom, (3) bottom-strip grid for the
touchline/box hint, (4) far-box zoom. All subsequent pixel work was
algorithmic verification of what those reads had already located, not new
exploration.

## 4. Identifiability gate (section 7) — answered NO

**Do the independent line constraints + centre circle supply enough
information to determine the intended metric homography up to only an
irrelevant Euclidean rotation/translation?**

**NO.** Reasoning: the circle reliably supplies two finite point
correspondences with known pitch coordinates (`axis_a(+9.15)`,
`axis_a(-9.15)`, from a genuine line–conic intersection, independent of the
withdrawn centre construction) — a real, valid contribution, but the exact
count of homography degrees of freedom this and the halfway line jointly
constrain is **withdrawn** (M3 instruction): the original "6 of 8 DOF"
figure was computed on top of the flawed single-line vanishing-point claim
just corrected above, so it is not a supported number and is not replaced
with a new speculative one here — establishing that precisely would need its
own careful derivation, not attempted in this pass. What the evidence does
settle, without needing that count: a second, independently-reliable world
direction (F2) is required to fully identify the homography beyond the one
axis the circle already anchors, and every candidate F2 source measured in
frames 1–60 was either too noisy (near touchline: 13px residuals on a 330px
baseline), had no signal at all (touchline beyond x=330), or was too small
and ambiguous to confidently assign to a direction family at all (far
box/corner edge). Per instruction, this was not patched by inventing a
missing DOF, assuming pitch length/width, or reusing the same conic
construction a third way to manufacture a fourth point — the gate is
answered NO and the attempt stops here.

## 5. Corrected centre / axis collinearity / independent validation

Not applicable — no corrected homography was produced, so there is nothing
to report a corrected centre, temporal-validity check, or independent
validation for. Sections 6, 8, and 9 of the M2.1 brief are consequently
short-circuited by the NO at section 4, exactly as the brief specifies
("If NO: STOP").

## 6. Verdict

**B — M2 METRIC CALIBRATION NOT VALIDATED.** The bounded geometry available
in the existing development footage (frames 1–60 of
`bayern_munich_3-1_chelsea_228`, at native 640×360 resolution) does not
support a defensible metric homography beyond one reliable axis. No
automatic-calibration research, no new dataset search, and no further M2
experiment was started, per explicit instruction.

## Supported after M2.1

| | status |
|---|---|
| metric world coordinates | experimental / not validated |
| metric speed | unsupported |
| metric distance | unsupported |

Production remains fail-closed by construction (`CalibrationStore` returns
`None` for anything outside an explicitly loaded, valid segment); no
currently-loadable artifact should be treated as validated per
`M2_1_CALIBRATION_STATUS.json`.

## Explicit non-actions

Possession: not touched. TEST: not accessed. Training: none. New dataset:
none. Automatic/learned calibration: not researched. `women_1_239`: not
attempted (correctly, per the conditional not being met). M3: not started.
