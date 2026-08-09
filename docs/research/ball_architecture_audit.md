# Ball detection architecture audit

_2026-08-09. Audit only — nothing here was implemented or trained._

Records the measured architecture of the production detector, why the ball is
hard for it, and the one architecture experiment worth running if detector work
resumes. **Detector development is frozen** as of this document; see the freeze
record at the end.

---

## 1. Current detection feature levels

Measured on `best_A_960.pt`: head class `Detect`, `end2end=True`, `reg_max=1`,
`nc=4`, 3 levels, 934,960 head parameters of 9,950,960 total.

| level | source layer | channels | stride | 960×960 | 640×360 letterboxed to 960 |
|---|---|---|---|---|---|
| P3 | 16 | 128 | **8** | 120×120 (14,400 cells) | 68×120 = **8,160 cells** |
| P4 | 19 | 256 | 16 | 60×60 (3,600) | 34×60 = 2,040 |
| P5 | 22 | 512 | 32 | 30×30 (900) | 17×30 = 510 |

**There is no stride-4 / P2 detection path.** The finest stride available is 8.
Total prediction cells at EyeCU's real input shape: 10,710.

GFLOPs: 22.8 @640, **51.9 @960**, 94.0 @1280.

## 2. Ball footprint by feature level

Footprint in cells = ball width ÷ stride, at 960 inference geometry.

| ball width | context | P3 (s8) | P4 (s16) | P5 (s32) | *hypothetical P2 (s4)* |
|---|---|---|---|---|---|
| 4.0 px | TRAIN p10 | 0.50 | 0.25 | 0.13 | *1.00* |
| 6.7 px | **TRAIN median** | **0.84** | 0.42 | 0.21 | *1.68* |
| 12.0 px | TRAIN p90 | 1.50 | 0.75 | 0.38 | *3.00* |
| 20.0 px | youth w1 lower | 2.50 | 1.25 | 0.63 | *5.00* |
| 24.0 px | youth w1 upper | 3.00 | 1.50 | 0.75 | *6.00* |

At the finest level the network has, the median training ball is sub-cell.
**425 of 673 TRAIN balls (63%) are under 8 px** and therefore sub-cell at every
available stride. Only the youth-w1 regime (20–24 px) has a multi-cell
footprint at P3, which is consistent with that regime failing for reasons other
than spatial resolution — contact, occlusion and clutter.

**Cell count alone does not determine detectability.** Effective receptive
field, feature semantics, the end-to-end one-to-one assignment and the
object-to-background ratio within a cell all matter. Bounding-box regression is
continuous, so stride 8 does **not** impose 8-pixel localisation quantisation.
The defensible statement is narrower:

> The lack of a stride-4 feature level may limit preservation and assignment of
> fine spatial information for very small balls; a P2 path is a plausible
> future architecture ablation.

## 3. Options considered

| | A. current | B1. stock p2 | **B2. width-matched p2** | C. dedicated ball branch | D. ROI 2nd stage |
|---|---|---|---|---|---|
| strides | 8/16/32 | 4/8/16/32 | 4/8/16/32 | 4 + context | unchanged |
| params | 9,950,960 | 9,665,024 | 10,389,376 (1.04×) | +small | unchanged |
| GFLOPs @960 | 51.9 | 60.6 (1.17×) | **88.2 (1.70×)** | est. 1.2–1.5× | 1.6× at 34.6% invocation |
| prediction cells | 10,710 | 43,350 | 43,350 (4.05×) | ball-only | unchanged |
| head cls/box width | 128 / 32 | **64 / 16** | 128 / 32 | independent | n/a |
| complexity | — | trivial | one yaml line | high | moderate |
| human-class risk | — | elevated | low | minimal | none |

**The key architectural finding.** Ultralytics derives the Detect head's branch
widths from `ch[0]`. The stock `yolo26-p2.yaml` places a 64-channel P2 branch
first, which halves the classification branch (128→64) and box branch (32→16)
at *every* level, including the P3/P4/P5 levels that serve players, goalkeepers
and referees. The stock config therefore does not "add P2" — it adds P2 **and**
narrows the whole head. Widening the P2 branch to 128 channels restores the
original widths at 60.6 → 88.2 GFLOPs.

Option C (FootAndBall / DeepBall-style dedicated high-resolution ball pathway)
is the strongest in principle but needs a custom module, loss and trainer
integration. Option D is largely foreclosed by measurement: A@1280 is *worse*
than A@960, so an ROI pass with A's weights has no headroom.

## 4. Weight transfer audit (B2)

Donor: the A-topology checkpoint. Matching is by state-dict key **and** shape.

| | tensors | parameters |
|---|---|---|
| baseline (A topology) | 708/708 = 100% | 9,989,058/9,989,058 = **100%** |
| width-matched P2 | 360/902 = **39.9%** | 6,075,238/10,431,985 = **58.2%** |

Module-wise, for the P2 topology:

| module | tensors | parameters | transfer |
|---|---|---|---|
| **backbone** (layers 0–10) | 240/240 | 5,458,760 / 5,458,760 | **100%** |
| neck, shared layers | 111/288 | 616,469 / 3,756,592 | 16.4% |
| new P2 path (layers 17–19) | 9/54 | 9 / 104,457 | ~0% |
| Detect head | 0/320 | 0 / 1,112,176 | 0% |

**This is not a half-random initialisation.** The entire feature extractor
transfers intact; what re-initialises is the neck above it, the new P2 branch
and the detection head — an ordinary fine-tuning situation. The tensor
percentage (39.9%) understates transfer badly because the head contributes many
small tensors while the backbone holds the large convolutions; the parameter
figure (58.2%) is the meaningful one, and the module breakdown is more
informative than either.

## 5. If detector work resumes — recommended experiment

**B2: width-matched P2 head, YOLO26s @960, original 823-frame TRAIN.**

- **Architecture delta:** `yolo26-p2.yaml` with `head[8]` C3k2 args `[128, True]` → `[256, True]`
- **Starting weights:** `yolo26s.pt` COCO, partial (backbone 100%)
- **Data:** original TRAIN only — *not* the Experiment C derived set
- **Resolution:** 960, unchanged
- **Cost:** +438,416 params (1.04×), 88.2 GFLOPs (1.70×) — within 6% of B@1280's 94.0, enabling a matched-cost comparison of finer feature stride versus input upsampling
- **Memory:** the P2 activation is 136×240×128 per image; batch may need to drop from 3 to 2, which would be a deviation from A's recipe and must be reported

**Predeclared success** — all four: temporal-val raw ball recall > 0.5714 with
the gain concentrated in the small-ball windows; coverage with the frozen
selector ≥ 0.7143 (B); 208-VAL player recall ≥ 0.90 and referee recall ≥ 0.75;
measured ms/frame ≤ B's 263.3.

**Predeclared failure:** ball no better than A, or the human guardrail
breached, or cost ≥ B without matching B's accuracy.

---

## Detector freeze record — 2026-08-09

Detector development is **paused**. The unresolved system problem is tracking
and ID continuity, not detection.

| model | status |
|---|---|
| **A@960** (`best_A_960.pt`) | **production / speed baseline** — mAP50 0.739, ~58 FPS on T4 |
| **B@1280** (`best_B_1280.pt`) | **accuracy reference** — 0.7143 temporal coverage, 1.70× compute |
| **C@960** (`best_C_960.pt`) | **rejected** — contextual crop/zoom + hard-negative ablation, net −10 ball |
| **B2 / P2** | **documented future experiment, not executed** |

B2 would cost roughly B-level compute and therefore faces a high bar to change
the system decision. It is preserved here as the leading candidate should
detector research resume.
