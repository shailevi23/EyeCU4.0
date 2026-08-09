> **Experiment C specification — CLOSED.**
> Predeclared plan for `C_yolo26s_960_scale_context_hardneg`, written before the
> run. C was **rejected**; results in
> [../../../docs/results/RESULTS.md](../../../docs/results/RESULTS.md).
> Kept verbatim so the criteria can be checked against the outcome.

---

# EyeCU 4.0 — EXPERIMENT C TODO
## Fact-checked scale/context + hard-negative training ablation

Act as a senior computer-vision engineer working on a football-analysis research
pipeline.

The objective is NOT to keep optimizing detector mAP indefinitely.

The objective is:

> Run exactly one low-cost, scientifically defensible training-side ablation
> that addresses the ball failure modes we have actually measured, and determine
> whether we can materially improve A@960 without paying the B@1280 compute cost.

Do not implement unrelated improvements.

======================================================================
0. CURRENT FROZEN BASELINES
======================================================================

Keep these checkpoints/configurations frozen:

A:
YOLO26s trained @960

B:
YOLO26s trained @1280

Measured temporal-validation raw detector results:

A@960:
TP 44
FP 4
FN 33
recall 0.5714

A@1280:
TP 41
FP 3
FN 36
recall 0.5325

B@960:
TP 44
FP 4
FN 33
recall 0.5714

B@1280:
TP 50
FP 7
FN 27
recall 0.6494

With the SAME untuned TemporalSelector:

A@960 + temporal:
coverage 48/77 = 0.6234
hallucinated 1/27

B@1280 + temporal:
coverage 55/77 = 0.7143
hallucinated 3/27

A remains the speed baseline.

B@1280 remains the current accuracy baseline.

Do not modify either.

======================================================================
1. CORRECT THE SCIENTIFIC WORDING
======================================================================

Do NOT write:

"weights alone do nothing"

because A@960 and B@960 have equal aggregate TP/FP but materially different
per-window detection behavior.

B@960 gains detections in some windows and loses them in others.

Use:

> At 960, A and B have equal aggregate recall but materially different
> per-window detection behavior. B's aggregate advantage appears at 1280,
> indicating a strong interaction between training configuration, learned
> weights and inference resolution.

Also do NOT write:

"models must be evaluated at the resolution they were trained at"

or:

"resolution matching is the cause."

The experiment only establishes:

> For these specific checkpoints on this benchmark, A performs worse when run
> at 1280 than at 960, while B performs better at 1280 than at 960.

This is a checkpoint-specific interaction, not a universal YOLO rule.

======================================================================
2. EXTERNAL RESEARCH CONSTRAINTS
======================================================================

The next experiment should reflect findings from football-specific computer
vision literature, not only generic YOLO intuition.

Relevant findings:

### FootAndBall
Football ball detection is not only a tiny-object problem.

The literature identifies difficulties including:

- large apparent-scale variation
- partial/total occlusion by players
- balls at players' feet
- motion distortion
- socks/body parts
- advertisements
- pitch markings
- background clutter

FootAndBall also uses high-resolution + semantic multi-scale features rather
than relying only on larger full-frame input resolution.

Reference:
https://www.scitepress.org/Papers/2020/89160/pdf/index.html

### DeepBall
Ball detection benefits from combining local spatial detail with broader
semantic context.

Reference:
https://arxiv.org/abs/1902.07304

### Hard-negative mining
FootAndBall explicitly uses hard-negative mining because easy background
examples massively outnumber difficult ball-like negatives.

This is especially relevant because EyeCU has already produced false ball
candidates on objects such as a goalkeeper-shirt crest.

### Temporal reasoning
Football-specific literature supports temporal reasoning for ball recovery,
especially under occlusion, but our current measurements show that temporal
logic cannot rescue frames where the detector produces no useful proposal.

Therefore TemporalSelector stays frozen during Experiment C.

======================================================================
3. WHAT OUR DATA ACTUALLY SHOWS
======================================================================

TRAIN:

673 ball instances
823 train images

Ball width distribution normalized to 960 geometry:

p10     4.2 px
p25     5.7 px
median  6.7 px
p75     9.0 px
p90     12.0 px

Counts:

<5 px       115
5–8 px      310
8–12 px     170
12–16 px     43
>16 px       35

Only ~40 training balls are >=15 px.

Human proximity:

ball center inside human bbox:
201 / 673 = 29.9%

ball overlaps human bbox:
253 / 673 = 37.6%

ball <30 px from nearest human:
523 / 673 = 77.7%

Therefore:

"ball near a player" is NOT a rare training condition.

Do not create Hard-100 around generic proximity.

======================================================================
4. MEASURED FAILURE REGIME A — YOUTH W1
======================================================================

Youth w1:

12 GT balls
A@960 raw accepted detections: 0/12

Apparent ball size at 960 geometry:
approximately 18.9–24.2 px

This is well above:

TRAIN median = 6.7 px
TRAIN p90    = 12.0 px

Visual failure characteristics:

- close-camera shot
- ball at/around players' feet
- partial occlusion by boots/legs
- close-contact duel
- ball is LARGE, not tiny

Most A misses remain unrecoverable even with confidence reduced to 0.01.

Therefore this failure cannot be described simply as:

"small-object detection failure."

Supported hypothesis:

> Large-ball / close-camera contexts are underrepresented relative to the
> normal broadcast-scale ball distribution.

However:

do NOT claim that scale alone is proven to be causal.

Experiment C is an ablation designed to test this hypothesis.

======================================================================
5. MEASURED FAILURE REGIME B — WOMEN_1 W1
======================================================================

women_1 w1:

small/normal-scale balls
approximately 5–7 px

Failure context includes:

- upper-center image region
- stadium/stone/dark background clutter
- non-clean-grass appearance around ball

TRAIN does contain some similar non-grass/background-clutter contexts.

Therefore we do NOT currently have strong evidence that women_w1 represents a
pure dataset-coverage hole.

Do not build an entire augmentation branch only for women_w1.

However, preserve background context in Experiment C so that contextual
augmentation may also help this case.

======================================================================
6. EXPERIMENT C — CORE HYPOTHESIS
======================================================================

Experiment name:

C_yolo26s_960_scale_context_hardneg

Question:

> Can targeted TRAIN-only scale/context augmentation plus a small amount of
> hard-negative mining improve the measured ball failure regimes while
> preserving A-like inference speed?

Architecture remains:

YOLO26s

Training/inference resolution remains:

960

Do NOT change model size.

Do NOT move to 1280.

Do NOT change architecture.

Do NOT add P2 yet.

Do NOT tune TemporalSelector.

======================================================================
7. POSITIVE DERIVED TRAINING VIEWS
======================================================================

Create a small derived TRAIN-only augmentation package from EXISTING GT labels.

Do not manually annotate new positive frames yet.

Target approximately:

50–80 derived positive training images.

Goal:

increase representation of ball appearances around:

12–25 px at 960 geometry

with emphasis around:

15–22 px

This targets the measured large-ball scale gap.

But these must be CONTEXTUAL crops.

Do NOT make isolated ball crops.

Each derived image should preserve useful surroundings such as:

- player feet
- boots
- legs
- nearby players
- grass
- stadium/background
- advertising/clutter where naturally present

======================================================================
8. CROP GENERATION RULES
======================================================================

For each selected TRAIN ball:

create contextual crop(s) around the existing annotation.

Requirements:

- random spatial offset
- ball must NOT always be centered
- retain surrounding humans/context
- crop sizes should vary
- resize derived image to the normal training geometry
- transform ALL retained labels correctly
- preserve player/GK/referee/ball classes
- do not normalize GK to player
- clip boxes conservatively
- discard objects that become pathologically truncated if necessary
- do not generate unrealistic giant balls
- no validation/test images

Prefer examples where resulting ball width lands around:

15–22 px

while allowing a broader:

12–25 px

distribution.

======================================================================
9. SOURCE DIVERSITY
======================================================================

Do not let one source dominate.

Record:

- original image
- original match/source
- ball bbox before transform
- resulting ball bbox after transform
- crop parameters
- derived image filename

Cap derived views per original source/match.

Goal:

increase SCALE diversity without replacing match diversity with duplicate views.

======================================================================
10. HARD-NEGATIVE MINING
======================================================================

Add approximately:

20–40 TRAIN-only hard-negative views.

Purpose:

Teach the ball classifier what visually similar NON-BALL objects look like.

Mine from TRAIN sources only.

Use A@960 predictions to find confident/plausible ball predictions where no GT
ball matches.

Potential hard negatives include:

- shirt crests/logos
- white socks
- boots
- heads
- white pitch markings
- advertisement elements
- circular stadium objects
- other recurrent ball-like shapes

Do NOT simply add thousands of background negatives.

Prefer difficult negatives that the current detector actually confuses with a
ball.

======================================================================
11. IMPORTANT HARD-NEGATIVE DATA REPRESENTATION
======================================================================

Do not create incorrect annotations.

If the hard-negative view contains no real labelled ball:

the image may legitimately contain NO ball annotation.

But retain valid player/GK/referee labels if they remain in the crop and belong
to the training taxonomy.

Document which derived images are:

positive-context crops

vs

hard-negative crops

in provenance metadata.

======================================================================
12. BEFORE TRAINING — STOP AND RETURN AN AUDIT
======================================================================

Do NOT start training immediately.

First generate the proposed derived-data package and return:

DERIVED POSITIVE AUDIT

- number of derived positives
- per-source counts
- original ball-scale distribution
- derived ball-scale distribution
- combined TRAIN ball-scale distribution
- count >=15 px before
- count >=15 px after
- count >=20 px before
- count >=20 px after

HARD NEGATIVE AUDIT

- number mined
- confidence distribution
- false-positive categories
- per-source distribution

CONTEXT AUDIT

- examples containing nearby humans
- examples with ball at/near feet
- background-context diversity

LEAKAGE AUDIT

Explicitly verify:

- no frozen 208 VAL images
- no temporal 104 VAL images
- no TEST images
- no external validation/test sources

CONTACT SHEET

Create a representative contact sheet containing roughly:

- 12 positive derived crops
- 8 hard-negative examples

This must be reviewed before training.

======================================================================
13. TRAINING — EXACTLY ONE RUN
======================================================================

Only after the derived-data audit is accepted:

Train:

C_yolo26s_960_scale_context_hardneg

Use:

model = YOLO26s
imgsz = 960

Base data:

current original TRAIN
+
approved derived TRAIN-only views

Keep all remaining training settings as close to Experiment A as practical.

Do NOT simultaneously modify:

- optimizer strategy
- LR strategy
- architecture
- resolution
- P2
- loss functions
- tracker
- TemporalSelector
- confidence thresholds

The experiment should isolate:

targeted data distribution change.

Do NOT run multiple C variants.

======================================================================
14. WHY NOT JUST INCREASE GLOBAL SCALE AUGMENTATION?
======================================================================

Do not automatically replace this experiment with something like:

scale=0.9

Generic random scaling is not equivalent to the measured hypothesis.

The problem is specifically:

underrepresentation of contextual large-ball appearances.

Experiment C therefore uses object-aware contextual derived views while keeping
the remainder of the recipe stable.

A generic scale-augmentation experiment can be considered later only if C fails
and there is a strong reason.

======================================================================
15. EVALUATION — SPARSE 208 VAL
======================================================================

Evaluate A vs C using identical settings.

Do not tune C thresholds separately.

Use the same:

- confidence handling
- candidate floor
- ball dedupe IoU
- matching IoU

Report:

overall:
- P
- R
- mAP50
- mAP50-95

per class:
- player
- goalkeeper
- referee
- ball

For BALL additionally:

- A-only detections
- C-only detections
- both
- neither
- paired comparison
- per-match results
- FP change
- duplicate rate

Important:

Do not improve ball by substantially damaging humans.

======================================================================
16. EVALUATION — CONTINUOUS 104 TEMPORAL VAL
======================================================================

First evaluate detector only.

Temporal OFF.

Report:

A@960 vs C@960:

- TP
- FP
- FN
- precision
- recall
- per-match
- per-window

Especially:

youth w1
women_1 w1

Also compare failure membership:

For every A miss:

- A miss / C hit
- A miss / C miss

For every C regression:

- A hit / C miss

======================================================================
17. TARGETED YOUTH-W1 QUESTION
======================================================================

Explicitly answer:

Did C improve the large-ball/close-duel failure regime?

Baseline:

A youth w1:
0 / 12

Report C youth w1 exactly.

Also inspect whether C detects balls:

- around boots
- between legs
- partially occluded
- at large apparent scale

Do not summarize only aggregate recall.

======================================================================
18. TEMPORAL EVALUATION
======================================================================

After raw C results are recorded:

Run the EXACT SAME existing TemporalSelector.

Do not tune it.

Report:

C raw recall
C temporal coverage
recovered_low_conf
interpolated_short_gap
unknown
hallucinated frames
false recovery rate
per-window results

Compare:

A@960 + temporal

C@960 + temporal

B@1280 + temporal

B remains the accuracy reference:

B@1280 + temporal:
55 / 77 = 0.7143 coverage

Do not require C to beat B at all costs.

The important question is whether C materially closes the accuracy gap while
retaining A-like compute.

======================================================================
19. SPEED
======================================================================

Measure C@960 inference speed with the same environment/method used for A@960.

Expected hypothesis:

C uses the same architecture and resolution, therefore inference cost should be
approximately A-like.

But measure it.

Do not assume.

Report:

ms/frame
FPS
relative cost vs A
relative cost vs B

======================================================================
20. PREDECLARED SUCCESS CRITERIA
======================================================================

Do not invent success criteria after seeing results.

Strong success:

- material ball-recall improvement on BOTH validation datasets
- targeted improvement in youth w1
- no major precision collapse
- no material human-class regression
- approximately A-like inference speed

A particularly interesting result would be C approaching:

~49–50 / 77 raw temporal-val TP

while staying near A@960 speed.

That would put C near B@1280 raw accuracy without B's ~1.7x compute cost.

This is an aspirational reference, NOT a hard p-hacking threshold.

======================================================================
21. FAILURE CRITERIA
======================================================================

If C:

- does not improve youth w1 materially
- does not improve overall ball performance
- causes substantial FP growth
- harms human classes
- or gains only on the derived failure window while regressing broadly

then Experiment C fails.

Do not generate C2/C3/C4 immediately.

Stop and analyze.

======================================================================
22. IF C FAILS — NEXT ARCHITECTURAL QUESTION
======================================================================

If the data-side ablation fails, do NOT immediately:

- annotate hundreds more images
- switch to YOLO26l/x
- increase full-frame resolution to 1600
- tune TemporalSelector endlessly

Football-specific literature suggests another plausible direction:

a dedicated high-spatial-resolution ball feature pathway.

Examples from literature include:

- FootAndBall high-resolution/FPN-style ball features
- DeepBall multi-level/hypercolumn context

Therefore a later experiment may investigate:

- P2/high-resolution detection head
- dedicated ball branch
- specialized ball detector

BUT:

Do not implement this as part of Experiment C.

Only return it as a recommendation if C fails.

======================================================================
23. HARD-100 RULE
======================================================================

Do not start generic Hard-100 now.

If C fails, re-characterize residual failures first.

Hard-100 should only be created if there is evidence for genuinely missing
appearances/domains.

Do NOT mine generic:

"ball near player"

because ~77.7% of current TRAIN balls already satisfy close human proximity.

Any future Hard-100 must target residual measured failure regimes, not broad
categories that are already common.

======================================================================
24. KEEP HUMAN PIPELINE WORK SEPARATE
======================================================================

Do not mix Experiment C with the human-role work.

Existing diagnostics indicate that human localisation and semantic role errors
are distinct problems.

Later work may use:

human localisation
-> class-agnostic human association
-> track-level role reasoning

This is consistent with modern football-CV pipelines and SoccerNet-style system
decomposition.

But do not modify this during C.

======================================================================
25. CALIBRATION REMAINS OUT OF SCOPE
======================================================================

Do not attempt to improve speed/distance reporting during Experiment C.

Current metric conversion remains uncalibrated.

Football tracking literature and FIFA EPTS validation make clear that reliable
metric position/velocity requires proper camera/pitch calibration.

Keep speed/distance marked experimental/uncalibrated.

======================================================================
26. TEST SPLIT — ABSOLUTE RULE
======================================================================

DO NOT ACCESS TEST.

No:

- inference
- threshold testing
- visual inspection for model selection
- training
- mining
- hard negatives
- augmentation
- qualitative comparison

Experiment C is selected exclusively with:

TRAIN
+
208 VAL
+
104 temporal VAL

TEST remains frozen until the final system is selected.

======================================================================
27. REQUIRED TESTS
======================================================================

Add tests for derived dataset generation:

- transformed ball bbox correctness
- transformed human bbox correctness
- no class remapping
- GK remains GK
- crop clipping correctness
- no invalid boxes
- provenance always present
- source split preserved
- validation/test source rejection
- deterministic generation with fixed seed
- no duplicate output filenames

Hard-negative tests:

- no unmatched GT ball accidentally cropped out and represented as a true
  negative without explicit handling
- source remains TRAIN
- valid retained human labels remain valid

======================================================================
28. REQUIRED RESPONSE BEFORE TRAINING
======================================================================

Return ONLY:

EXPERIMENT C DATA PLAN
- ...

POSITIVE SCALE-CONTEXT AUDIT
- ...

HARD-NEGATIVE AUDIT
- ...

BALL SCALE BEFORE/AFTER
- ...

SOURCE DIVERSITY
- ...

LEAKAGE CHECK
- ...

CONTACT SHEET
- path

FILES TO ADD/MODIFY
- ...

TESTS
- ...

GO / NO-GO TO TRAIN C
- ...

Do NOT train until this audit is returned.

======================================================================
29. AFTER APPROVAL
======================================================================

After approval:

1. train exactly ONE C model
2. evaluate sparse 208 VAL
3. evaluate continuous 104 VAL
4. run frozen TemporalSelector
5. compare against A and B
6. make a production recommendation
7. stop detector experimentation if C fails to provide meaningful value

The final scientific question is:

> Can correcting the measured scale/context imbalance and teaching the model
> difficult ball-like negatives materially improve EyeCU's ball detection at
> 960px, without increasing production inference cost?