# M4 FINAL REPORT -- sealed TEST GT

## 0. Verdict

**CLOSED.** All 14 sections complete. 60/60 TEST frames exhaustively
annotated, QC-clean, frozen with SHA256, M3 freeze identity re-verified
unchanged. No EyeCU production prediction was ever run on TEST. M5 is
prepared but **not executed**.

## 1-2. Freeze precheck + TEST scope (done earlier this pass)

- M3 freeze verified MATCH before any TEST access (manifest hash, source
  tree via clean-checkout/git-tree-hash methodology, both weight hashes).
- TEST split pinned per `docs/guides/LABELING.md`: `como_2-0_sassuolo`,
  `manchester_city_v_liverpool`, `youth_2`.
- 60 frames selected deterministically, `floor((k+0.5)*N/20)` for k=0..19,
  20 per match, computed from true sequential-decode frame counts (the
  container header was unreliable for all three videos, corrected --
  a mechanical fix, not a content-dependent one).
- `INITIAL_TEST_FRAME_LIST.json` frozen before any pixel was viewed
  (sha256 `8a4abe76b4a4d4b22fe9eda4647675f6ed290defde6f75a7b9fd200e7b50e34d`).

## 3-4. Leakage screening + final frame list

- Reused the already-frozen `tools/extdata_leakage.py` method unmodified
  (64x64 canonical grayscale signature, 64-bit dhash, 8 dihedral
  orientations, Hamming<=10 candidate gate, NCC>=0.95 verification),
  imported (not edited) into a new M4-only wrapper, plus an added exact-
  SHA256 pre-check layer.
- Screened against the full TRAIN+VAL pool (2231 images: 823 train + 208
  val from `data/dataset_baseline`, 1200 from `data/tracking_val_gt`
  sequences).
- **Result: 0/60 leaks found.** The frozen replacement rule (nearest
  ordinal frame, search order +-1,+-2,...) was therefore never exercised.
- `FINAL_TEST_FRAME_LIST.json` == `INITIAL_TEST_FRAME_LIST.json` in
  content, frozen separately as its own artifact
  (sha256 `2ea02e1839726a6b4ad2fc2dbd979b58fa65f038e828d556eb0b290e8a47a62f`).

## 5. Annotation protocol

Frozen before any visual TEST access
(`TEST_ANNOTATION_PROTOCOL.md`, sha256
`2dfaed2eb0057f9c4a0e5ae94baf66a80e5334853ef4970f2ba23ed9a94ebd6d`):
classes `player`/`goalkeeper`/`referee`/`ball`, ball ontology
`ALL_VISIBLE_PHYSICAL_FOOTBALLS` (every real physical football, not just
the active one), pixel `[x1,y1,x2,y2]` boxes, occlusion/goalkeeper/referee
rules reused verbatim from the existing TRAIN/VAL convention.

## 6-7. Manual annotation -- three phases, no model prelabels at any point

1. **In-session (assistant, visual reads)**: all 20 `como_2-0_sassuolo`
   frames, via grid-overlay crops read directly, no EyeCU model involved.
2. **Stopped for cost** after 20/60 (not 3/60 as first assumed -- corrected
   at the time). A CVAT/Roboflow handoff package was built for the
   remaining 40, then superseded before use.
3. **Local tool**, built after confirming no existing repo tool fit (the
   `kb_*` servers in `tools/` are all narrow, fixed-queue *review* tools
   tied to historical decision logs, not from-scratch multi-object
   annotators): `experiments/records/experiment_M4/local_annotator/`, a
   stdlib-only browser bbox editor, used to hand-label all 40 remaining
   frames (`manchester_city_v_liverpool`, `youth_2`). No model inference
   anywhere in that tool.

Merged via `local_annotator/merge_into_draft.py` into a single 60-frame
`ANNOTATIONS_DRAFT.json`.

## 8. Completeness/validity QC -- computational only, no visual re-review

`m4_qc.py`: population match against `FINAL_TEST_FRAME_LIST.json` (60/60,
no duplicates, no missing/extra frames), per-object class validity, bbox
positivity (`x2>x1`, `y2>y1`), in-bounds check against each sequence's
known fixed frame dimensions, exact-duplicate-box detection.

First pass found **21 boxes (of 924) out-of-bounds by 1-25px** -- mouse-
drag overshoot at the frame edge in the local tool. This is a pure
geometric property (a box coordinate outside `[0,w]x[0,h]`), checkable and
fixable without opening the image, so it was corrected mechanically:
`m4_clamp_boxes.py` clamped each offending coordinate to the frame
boundary, full before/after log in `CLAMP_LOG.json`. Re-run: **0 errors, 0
warnings, `qc_pass: true`.**

## 9. GT-only descriptive statistics (no model comparison)

| | frames | objects | player | goalkeeper | referee | ball |
|---|---|---|---|---|---|---|
| como_2-0_sassuolo | 20 | 376 | 322 | 14 | 25 | 15 |
| manchester_city_v_liverpool | 20 | 337 | 286 | 8 | 27 | 16 |
| youth_2 | 20 | 211 | 174 | 9 | 11 | 17 |
| **total** | **60** | **924** | **782** | **31** | **63** | **48** |

Objects/frame: min 1, max 23, mean 15.4, median 17. Zero-object frames: 0.
Frames missing a goalkeeper: 29/60. Frames missing a referee: 14/60.
Frames missing a ball: 15/60. **Multi-ball frames: 3** (2 balls each) --
`manchester_city_v_liverpool` frames 38 and 113, `youth_2` frame 1522 --
confirming the `ALL_VISIBLE_PHYSICAL_FOOTBALLS` ontology was actually
exercised, not vacuous. Full detail in `GT_DESCRIPTIVE_STATS.json`.

## 10. Freeze

| artifact | sha256 |
|---|---|
| `TEST_DETECTION_ANNOTATIONS.json` | `4702b60fbdb173773e6bc7246d45587c1446093559c96792d3ae864e4d6896cb` |
| `QC_RESULT.json` | `4c1c4987dd79f68fd9ed9f854b6801234e7844b5b9d4d8f51278e7088542fcef` |
| `GT_DESCRIPTIVE_STATS.json` | `c19dc50dc32f0c4eea7606e0c94f4376a61e4bcfa789dbae0b0d82b35439b9ec` |
| `TEST_ACCESS_STATE.json` | `558c291ad6bf266cb9225c4848d1874984d11aa421cd36cfaec1b2f16f2e7552` |

Access state (final): `machine_leakage_accessed=true`,
`human_annotation_accessed=true`, `labels_frozen=true`,
`production_predictions_run=false`, `evaluation_results_viewed=false`.
`TEST_DETECTION_ANNOTATIONS.json`'s hash is computed **after** the Section 8
clamp correction -- it is the authoritative, QC-clean artifact. Any future
correction must supersede explicitly, never silently overwrite.

## 11. M5 contract -- prepared, not executed

`M5_EVALUATION_CONTRACT.md` (sha256
`7af73ee61c82096818f37a2723134ea7bffee7fd63f715674c0563c68fc0e265`):
detection-only scope (P/R/mAP50/mAP50-95, per-class and per-match, fixed
IoU>=0.50 + the frozen production confidence thresholds, no invented CIs,
no accuracy claim beyond detection, no VAL-vs-TEST "improvement" framing,
all 60 frames/4 classes always reported). Gated on a fresh M3
freeze-identity re-check at the start of M5 itself.

## 12. M3 freeze integrity re-verification -- PASS

`M3_INTEGRITY_REVERIFICATION.json` (sha256
`6d1d554c26bd7d449728ef7c5cb85d36aee544e8e1607c65b3df2b8e2685909b`):
manifest file hash MATCH; source-tree identity confirmed via git-native
tree-object hash comparison for every hashed path (`trackers`, `tools`,
`rf_trackers`, `full_pipeline.py`, `run_pipeline.py`) between the commit
the manifest's recorded hash was computed from and current HEAD -- all
MATCH, working tree clean. Both weight hashes (`best_A_960.pt`,
`yolo-sn-ball.pt`) MATCH. One transparency note: an independent attempt to
re-derive the manifest's bespoke `deterministic_source_tree_sha256` from a
fresh clone produced a different number than recorded, most likely because
the original hashing script's exact path/byte-concatenation order wasn't
fully specified in the manifest's prose and wasn't reproduced exactly --
not evidence of a content change, since the git tree-hash comparison
(content-addressed, unambiguous) independently proves identity. Recorded
rather than silently reconciled.

## 13-14. Hard-stop compliance

- No EyeCU production model (`best_A_960`, SN3D, `BallTemporalSelector`,
  CBIoU, full pipeline) was ever run on TEST. **Confirmed: `NO`.**
- No TEST detection/tracking/possession metric was computed.
  **Confirmed: `NO`.**
- No production code/config/model file was modified.
  **Confirmed: `NO`** (all M4 work lives under `experiments/records/experiment_M4/`).
- No tuning from TEST content occurred at any point.
- M5 was **not** executed -- contract prepared only.

**If CLOSED: M5 ONE-SHOT SEALED TEST EVALUATION.**
M5 is not executed here. STOP.
