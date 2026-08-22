# M5.1 FINAL REPORT -- corrected-GT evaluation (Como blind re-annotation)

`experiments/records/experiment_M5/` is preserved unedited as the
historical first evaluation. This milestone is a GT-methodology repair and
re-measurement, not a new model test: **zero model inference occurred**;
`RAW_PREDICTIONS.json` from M5 was reused byte-for-byte verbatim.

## What happened

1. **Erratum** (`M5_1_ERRATUM.md`): clarified that the Como GT-quality
   finding's primary evidence is the documented annotation-process
   mismatch (grid-read vs mouse-drawn), with the GT/prediction area
   comparison as supporting evidence only, not proof-by-detector; and that
   M5 in fact ran two distinct TEST-touching executions per sequence (the
   scored raw pass, and a separate E2E structural pipeline run), not one --
   which does not affect the frozen `RAW_PREDICTIONS.json`.
2. **Blind re-annotation**: the 20 frozen `como_2-0_sassuolo` TEST frames
   were re-annotated from scratch in a purpose-built local tool
   (`experiments/records/experiment_M5_1/local_annotator/`) that never
   loaded or displayed the original GT, predictions, confidences, or IoU
   results -- confirmed by the tool's own code (never opens those files)
   and by an empty `/api/annotations` response at boot.
3. **QC**: 2 of 480 new boxes were 1px over the frame boundary (mouse
   overshoot, same mechanical class of defect as M4's original clamp
   pass) -- clamped to `[0,640]x[0,360]`, log in `COMO_BLIND_CLAMP_LOG.json`.
   Re-run: 0 errors.
4. **Merge**: `TEST_DETECTION_ANNOTATIONS_CORRECTED.json` = 20 corrected
   Como records + the 40 unchanged manchester/youth_2 records from the
   original, untouched `TEST_DETECTION_ANNOTATIONS.json` (re-verified
   byte-identical to its frozen hash throughout).
5. **Metrics**: one evaluation pass, reusing the exact same metric code
   (`iou`, `ap_101point`, `evaluate`) imported from
   `experiments/records/experiment_M5/m5_metrics.py`, against the corrected
   GT and the unchanged `RAW_PREDICTIONS.json`.
6. **Comparison**: `M5_1_VS_M5_COMPARISON.json`, computed numerically from
   the two metrics files.

## Result -- the process-mismatch hypothesis is confirmed, decisively

| | orig M5 (grid-read Como GT) | M5.1 (blind mouse-drawn Como GT) | delta |
|---|---|---|---|
| **mAP50** (pooled) | 0.2616 | **0.6175** | **+0.3559** |
| **mAP50-95** (pooled) | 0.1351 | **0.2697** | **+0.1346** |

Como-only, by class (AP50 / AP50-95):

| class | orig AP50 | corrected AP50 | orig AP50-95 | corrected AP50-95 |
|---|---|---|---|---|
| player | 0.003 | **0.941** | 0.000 | 0.314 |
| goalkeeper | 0.000 | **0.772** | 0.000 | 0.280 |
| referee | 0.017 | **0.691** | 0.007 | 0.301 |
| ball | 0.000 | 0.064 | 0.000 | 0.008 |

`manchester_city_v_liverpool` and `youth_2` are **exactly unchanged**
(zero delta on every field) -- expected, since their GT was untouched, and
confirms the comparison script and metric reuse are correct (no
accidental cross-contamination between sequences).

Como's player/goalkeeper/referee numbers, once re-annotated by direct
mouse-drag rather than grid-reading, land in the same range as
manchester_city_v_liverpool and youth_2 (AP50 0.69-0.97 vs their 0.44-0.97),
strongly supporting the erratum's primary claim: the original near-zero
Como numbers were an artifact of the assistant's grid-crop annotation
method, not a property of the frozen detector. Ball remains weak on Como
(AP50 0.064) even under the corrected GT -- with only 16 GT ball instances
across 20 frames, this is a small-sample, genuinely weak result, not a GT
artifact (its GT/prediction area ratio was never flagged as anomalous).

## Corrected pooled read

With the Como GT defect repaired, the pooled mAP50=0.6175 / mAP50-95=0.2697
is a substantially more trustworthy read of the frozen detector's
held-out generalization than M5's invalid 0.2616 -- though ball detection
(pooled AP50 0.336) and goalkeeper (pooled AP50 0.522) remain the weakest
classes, and per-sequence variance (player AP50 ranges 0.79-0.97 across
the three matches) shows real match-to-match variability, not a single
clean number.

## Compliance

- Zero TEST image reads by the assistant this milestone.
- Zero model inference; `RAW_PREDICTIONS.json` reused verbatim (hash
  unchanged from M5: `c3df0edb...`).
- Zero threshold/model/frame-population changes.
- Zero error-analysis galleries or further investigation after the
  comparison above.
- Original M4 GT (`4702b60f...`) and M5 metrics/report untouched.
- No new milestone started.

## Freeze

| artifact | sha256 |
|---|---|
| `M5_1_COMO_QC_RESULT.json` | `0516c08099a116cccd4bade1440ec44892017d2654c1aac79f43261abb4a7c9b` |
| `TEST_DETECTION_ANNOTATIONS_CORRECTED.json` | `f9f1fa9d9b090df0056335cf232ff6f8de2e974ceab3ef002f1d099bb0c49cde` |
| `M5_1_DETECTION_METRICS.json` | `9989f7883288ca6d28808c136021d6f4b80ab92488ff50493def493cd42635b3` |
| `M5_1_VS_M5_COMPARISON.json` | `abec36b5a8e19d3c973dec56abe5f44e9127ee5bceab4f3d22f81e565d8d1252` |
| `COMO_BLIND_CLAMP_LOG.json` | `33115f41602fbb81164ee56fa97eb2eddfbdc036dfed178cadf738e4a3abf9c3` |

STOP.
