EYECU 4.0 — EXTERNAL DATA / RESEARCH SOURCES CONSOLIDATION
MASTER ACQUISITION + AUDIT PLAN

We have accumulated several external datasets, Hugging Face sources,
research repositories, and already-completed audits.

The current problem is that the information is fragmented across multiple
experiments and conversations.

Your task is to CONSOLIDATE ALL OF IT into one clean, reproducible external
data/research workspace and produce a prioritized EyeCU source roadmap.

THIS IS NOT A TRAINING TASK.

DO NOT:
- train a detector
- merge anything into EyeCU TRAIN
- modify EyeCU VAL or TEST
- evaluate TEST performance
- change the frozen detector
- change the frozen CBIoU production tracker
- start threshold tuning
- start a new tracking bake-off
- download large datasets without the gates below
- ask me to paste credentials/tokens into the transcript

==================================================
A. CURRENT EYECU FROZEN CONTEXT
==================================================

Treat these as fixed context.

Detector:
- YOLO26s @ 960 remains production/speed baseline A.
- YOLO26s @ 1280 remains accuracy reference B.
- Experiment C was rejected.
- P2 remains future architecture research, not active.

Tracker:
- CBIoUTracker is FROZEN production tracker.
- vendored trackers==2.6.0 exact library defaults.
- public IDs are positive integers.
- modern raw_id >= 0 maps to public_id = raw_id + 1.
- raw_id < 0 is unconfirmed and not emitted.
- legacy ByteTrack remains rollback only.
- tracking selection is CLOSED.

Tracking development benchmark:
- EyeCU-Tracking-Val-v1.1
- 3 clean VAL matches
- 900 frames
- 60 sequence-local identities
- 13,021 boxes
- TEST remains sealed.

Do not reopen tracker selection because of any external dataset discovered here.

==================================================
B. PRIMARY GOAL
==================================================

The immediate research priority is:

    improve EyeCU detector robustness,
    especially BALL detection.

Secondary/future priorities:

1. pitch calibration / homography
2. physical speed / distance validation
3. player identity / jersey / team
4. event detection
5. external tracking stress tests

Every external source must be categorized according to which of these it can
actually help.

==================================================
C. CREATE ONE MASTER EXTERNAL-SOURCES STRUCTURE
==================================================

Use the existing EyeCU_external_data root and organize it cleanly.

Target structure:

EyeCU_external_data/
│
├── roboflow_audit/
│   ├── raw_zips/
│   ├── reports/
│   └── manifests/
│
├── huggingface/
│   ├── keremberke_football_object_detection/
│   │   ├── raw/
│   │   ├── metadata/
│   │   └── manifests/
│   │
│   ├── soccernet_v3/
│   │   ├── metadata_only/
│   │   ├── full_dataset/
│   │   └── manifests/
│   │
│   ├── manifests/
│   └── download_logs/
│
├── soccertrack_v2/
│   ├── gsr/
│   ├── bas/
│   ├── raw/
│   ├── videos/
│   ├── public_repo/
│   ├── reports/
│   └── manifests/
│
├── research_sources/
│   ├── teamtrack/
│   ├── sportslabkit/
│   └── manifests/
│
└── MASTER_EXTERNAL_SOURCE_REGISTRY.json

Do not duplicate large files unnecessarily.

If files already exist elsewhere:
- hash first
- identify byte-identical duplicates
- move into canonical location safely
- verify hashes after move
- preserve previous audit evidence

Dataset binaries must not be committed to Git.

==================================================
D. EXISTING SIX ROBOFLOW DATASETS — DO NOT RE-AUDIT
==================================================

The six-source audit is already complete.

Preserve it.

Current findings:

S1 modulov/soccer-3ezqd
- PARTIAL USE
- only 148 distinct originals
- GK effectively labelled as player
- augmentation-heavy

S2 va-sah7v/football-eitpt
- REJECT
- poor/damaged augmented pixels
- despite having four classes

S3 roboflow-100/soccer-players
- PARTIAL USE
- useful diversity
- no separate goalkeeper class
- ball boxes somewhat padded

S4 simon6048/soccer
- REJECT
- effectively already represented in EyeCU TRAIN
- external↔TRAIN overlap found

S5 mishahal/soccer
- BALL REFERENCE ONLY
- ball-only annotations
- humans visible but unlabelled

S6 old-storm/soccer
- strongest Roboflow source
- 1,251 KEEP_CANDIDATE images
- good ball annotations
- but mostly one La Liga match
- 342 partial/unlabelled images excluded

Overall previous conclusion:

    Experiment D based on these six alone = WEAK

because they do not provide a meaningful new supply of genuinely tiny,
diverse broadcast balls.

Preserve:
- source ZIP hashes
- reports
- contact sheets
- candidate index

Do not rerun this audit unless required to validate a new duplicate source.

==================================================
E. MARTINJOLIF — SKIP
==================================================

Do NOT download:

    martinjolif/football-player-detection

User identified it as a copy/duplicate of Roboflow data already present in the
completed audit.

Record:

    STATUS = SKIP_DUPLICATE_SOURCE

If needed, verify provenance/duplication cheaply from metadata, but do not
download another full redundant copy.

==================================================
F. SOCCANA — DEFER
==================================================

Do NOT download now:

    Adit-jain/Soccana_player_ball_detection_v1

Reason:

- it combines/derives from multiple existing datasets
- slicing/derived images create duplicate/diversity concerns
- goalkeeper is merged into player
- EyeCU requires goalkeeper to remain a distinct class
- we are NOT starting a goalkeeper relabelling campaign now

Record:

    STATUS = DEFER_REQUIRES_GK_RELABEL_AND_DEDUP

Potential future use:
- ball candidate mining
- research/reference only

Do not use it for current Experiment D planning.

==================================================
G. KEREMBERKE — DOWNLOAD AND ORGANIZE
==================================================

Target Hugging Face source:

    keremberke/football-object-detection

Purpose:

    BALL DATA AUDIT

Download the full public Hugging Face dataset into:

    EyeCU_external_data/huggingface/
    keremberke_football_object_detection/raw/

Preserve:
- README/dataset card
- class metadata
- splits
- license
- Hugging Face repo revision/commit
- file list
- SHA256s

Validate:
- real files, not HTML/error responses
- image count
- annotation count
- image dimensions
- integrity

IMPORTANT:

This source has a partial class ontology relative to EyeCU.

Do NOT assume it can be directly mixed into EyeCU's four-class detector.

Audit specifically:

- exact class list
- total ball boxes
- images containing ball
- bbox width/height in stored/original pixels
- <=5 px
- <=8 px
- <=12 px
- median / p10 / p25 / p75 / p90
- image/source diversity
- broadcast vs non-broadcast
- partial-annotation risk for referee/GK
- exact and perceptual duplicates against:
    current EyeCU TRAIN
    EyeCU VAL
    EyeCU TEST
    previous six Roboflow sources

TEST use is leakage checking only.
No TEST performance.

End with:

    KEREMBERKE_USE =
        STRONG_BALL_SOURCE
        USEFUL_BALL_REFERENCE
        WEAK
        REJECT

Do not merge it into training.

==================================================
H. SOCCERNET-V3 — TOP-PRIORITY METADATA GATE
==================================================

Target:

    Voxel51/SoccerNet-V3

This is currently the most important external source to inspect because it may
contain diverse real broadcast football data.

However:

DO NOT assume that the full SoccerNet project and this specific Voxel51
Hugging Face representation contain identical annotations.

Use the ACTUAL Voxel51 files.

--------------------------------------------------
STEP H1 — METADATA ONLY
--------------------------------------------------

First download ONLY lightweight metadata/schema files where available:

- README.md
- metadata.json
- samples.json
- dataset card/config
- small indexes/schema files

Store under:

    EyeCU_external_data/huggingface/
    soccernet_v3/metadata_only/

Do NOT immediately download the large payload.

--------------------------------------------------
STEP H2 — VERIFY ACTUAL CONTENT
--------------------------------------------------

From actual metadata determine:

- exact number of samples
- classes
- whether ball exists in THIS export
- number of ball bounding boxes
- number of images containing ball
- player classes
- goalkeeper classes
- referee classes
- unknown/staff classes
- bbox coordinates
- image dimensions
- game/source identifiers
- number of independent matches if recoverable

Critically verify whether the data could map to EyeCU:

    player
    goalkeeper
    referee
    ball

without destroying the GK distinction.

--------------------------------------------------
STEP H3 — BALL VALUE GATE
--------------------------------------------------

If bbox dimensions + image dimensions exist in metadata, calculate ball-size
distribution BEFORE downloading images.

Report:

<3 px
3–5 px
>5–8 px
>8–12 px
>12–20 px
>20–40 px
>40 px

And totals:

<=5
<=8
<=12

Use actual source dimensions where available.

Do not call resized dimensions "native" unless verified.

--------------------------------------------------
STEP H4 — FULL DOWNLOAD DECISION
--------------------------------------------------

Download the Voxel51 Hugging Face image payload ONLY IF metadata demonstrates
meaningful value for EyeCU, especially one or more of:

- substantial ball annotation count
- genuinely useful tiny-ball examples
- strong multi-match broadcast diversity
- useful separate GK/referee annotations

If gate passes:

download into:

    EyeCU_external_data/huggingface/
    soccernet_v3/full_dataset/

Then perform visual label-quality verification and duplicate/leakage audit.

If gate fails:

STOP before the large download.

Do NOT download the original ~60 GB SoccerNet frame package during this task.

==================================================
I. SOCCERTRACK-V2 — CLOSE DATA ACQUISITION
==================================================

Existing locally:

- GSR: 20 JSON, ~55 GB
- BAS
- RAW
- one representative game/video
- public GitHub repository

Preserve these.

Current SoccerTrack audit already found:

GSR:
- player and goalkeeper identities/metadata
- no referee instances in delivered GSR
- no ball instances
- bbox_image geometry is not usable manual person GT
- boxes show position/size-regression behaviour

BALL:
- no ball bounding boxes
- BAS has no ball image coordinates
- RAW tracker data contains pitch-plane ball locations, not detector boxes

BAS:
- useful for future event/action research

RAW:
- very useful for calibration/homography methodology
- complete calibration material for the downloaded set
- strongest SoccerTrack asset for EyeCU

VIDEO:
- fixed panoramic domain
- not representative of EyeCU broadcast cuts/zoom

--------------------------------------------------
SOCCERTRACK MOT
--------------------------------------------------

STOP trying to acquire SoccerTrack-v2 MOT.

Do NOT:
- request Hugging Face credentials
- request gated access
- chase private mirrors
- reconstruct the whole dataset

Reason:

The public SoccerTrack-v2 repository documents that MOT ground truth is
generated using:

    position-based width/height estimation / regression

rather than independent person-tight manual bounding boxes.

Therefore obtaining the missing MOT archive is not expected to create a new
high-quality detector GT source.

Preserve this conclusion in the registry:

    SOCCERTRACK_MOT =
        DISTRIBUTION_UNAVAILABLE_OR_EMPTY
        NOT_REQUIRED_FOR_CURRENT_EYECU_DETECTOR_WORK

The public MOT code/docs may remain as research references:

- baselines/mot/
- src/evaluation/mot_hota.py
- docs/leaderboards/mot.json
- docs/format-mot.md
- docs/task-mot.html
- scripts/coordinate_conversion/convert_raw_to_pitch_plane_mot.py
- GT creation documentation

Do not spend additional acquisition time on MOT.

==================================================
J. SOCCERTRACK CALIBRATION — KEEP FOR FUTURE PHASE
==================================================

SoccerTrack RAW/calibration is HIGH VALUE as methodology/reference.

Create a future-research record for:

    EYECU CALIBRATION / HOMOGRAPHY PHASE

Potential use:

- image pixel -> pitch plane
- pitch plane -> metres
- validating speed/distance
- replacing current uncalibrated pixels_per_meter assumptions
- understanding distortion / fisheye / mapx/mapy
- calibration QA

Important transfer caveat:

SoccerTrack uses a fixed panoramic rig.

EyeCU has broadcast footage with:
- cuts
- pans
- zooms
- changing cameras

Therefore SoccerTrack's static calibration cannot simply be copied into
production EyeCU.

Use it as:
- methodological reference
- calibration ground-truth testbed
- evaluation asset

not as a direct production homography.

==================================================
K. TEAMTRACK — FUTURE TRACKING BENCHMARK ONLY
==================================================

Research source:

    AtomScott/TeamTrack

Relevant soccer components include:
- soccer_side
- soccer_top
- MOT-style tracking annotations
- trajectory/pitch-coordinate research

Do NOT download the whole TeamTrack dataset now.

Do NOT reopen CBIoU selection.

Record:

    CURRENT_USE = FUTURE_EXTERNAL_TRACKING_STRESS_TEST

Potential future uses:
- external MOT sanity/stress benchmark
- long fixed-camera identities
- calibration/trajectory research

Not a current ball-detector intervention.

If lightweight metadata / README / license can be saved without large download,
preserve it under:

    research_sources/teamtrack/

but no large data acquisition in this task.

==================================================
L. SPORTSLABKIT — CODE/RESEARCH REFERENCE
==================================================

Research source:

    AtomScott/SportsLabKit

Potentially useful concepts:

- sports tracking pipelines
- camera calibration
- pitch coordinates
- detection/tracking integration
- analytics representations

IMPORTANT LICENSE RULE:

Do not copy or vendor SportsLabKit code into EyeCU during this task.

Record the actual current license from the repository.

If GPL-3.0 is confirmed:

    RESEARCH_REFERENCE_ONLY

for the current commercial-product-oriented EyeCU codebase unless a later
dedicated license review decides otherwise.

Ideas/algorithms/papers can be studied independently.

Do not treat code availability as automatic permission for code integration.

==================================================
M. MASTER SOURCE REGISTRY
==================================================

Create:

    EyeCU_external_data/
    MASTER_EXTERNAL_SOURCE_REGISTRY.json

One record per source.

Fields:

SOURCE_ID
NAME
ORIGIN
URL_OR_REPO_ID
SOURCE_TYPE
LICENSE
LICENSE_VERIFIED_FROM
LOCAL_PATH
DOWNLOAD_STATUS
SIZE
HASH_OR_REVISION
DOMAIN
CLASSES
BALL_BOXES
GK_SEPARATE
REFEREE_AVAILABLE
PARTIAL_ANNOTATION_RISK
DUPLICATE_RISK
TRAIN_OVERLAP
VAL_OVERLAP
TEST_OVERLAP
BALL_DETECTOR_VALUE
HUMAN_DETECTOR_VALUE
TRACKING_VALUE
CALIBRATION_VALUE
EVENT_VALUE
IDENTITY_JERSEY_VALUE
CURRENT_DECISION
NEXT_ALLOWED_ACTION
NOTES

Possible CURRENT_DECISION values:

KEEP_ACTIVE
KEEP_REFERENCE
METADATA_GATE
DEFER
REJECT
SKIP_DUPLICATE
FUTURE_RESEARCH

==================================================
N. RESEARCH PRIORITY MATRIX
==================================================

Produce one table with rows for every source and columns:

BALL DETECTION
PLAYER/GK/REF DETECTION
TRACKING
CALIBRATION
EVENTS
PLAYER ID/JERSEY
DOMAIN MATCH TO EYECU
DATA QUALITY
DIVERSITY
EFFORT TO USE

Rate:

HIGH
MEDIUM
LOW
NONE
UNKNOWN

Then rank sources separately for:

1. CURRENT BALL IMPROVEMENT
2. FUTURE CALIBRATION
3. FUTURE EVENTS
4. FUTURE IDENTITY/JERSEY
5. FUTURE TRACKING RESEARCH

Do not collapse these into one generic ranking.

==================================================
O. EXPERIMENT D DECISION GATE
==================================================

Experiment D must NOT run yet.

After Keremberke and SoccerNet metadata/full audit, answer:

    SHOULD WE CREATE EXPERIMENT D?

Possible answer:

STRONG YES
YES
WEAK
NO

A future D should ideally preserve:

- YOLO26s
- 960 input
- same training policy
- same frozen VAL
- same evaluation protocol

and change mainly:

    audited external real data added to TRAIN

Do not execute D.

Important:

A large correlated block from one match is NOT automatically useful diversity.

Thousands of resized tiny-looking balls are NOT tiny-ball evidence if the
original source scale was larger.

==================================================
P. EXTERNAL DATA QUALITY RULES
==================================================

Apply these rules consistently:

1. Original unique images matter more than export count.
2. Augmentations are not independent diversity.
3. Same-match sequential frames are highly correlated.
4. Partial annotation can actively damage a multi-class detector.
5. GK must remain distinct from player.
6. Referee must remain distinct.
7. Ball-only data with visible unlabeled humans cannot be blindly mixed.
8. Never use detector predictions as GT authority during source audit.
9. Never use VAL/TEST performance to choose external training data.
10. TEST may be used only for image/source leakage checks while sealed.
11. Preserve rejected sources and reasons instead of silently deleting evidence.

==================================================
Q. HUGGING FACE DOWNLOAD SECURITY
==================================================

For public datasets:
- use explicit local destinations
- record repo revision
- record hashes
- save dataset card/license metadata

Do not store:
- HF tokens
- Authorization headers
- cookies
- credentials
- private signed URLs

Do not ask me to paste a token into the transcript.

If a source unexpectedly becomes gated/private:

    STATUS = BLOCKED_ACCESS

and continue independent public-source work.

Do not spend the task bypassing access restrictions.

==================================================
R. GIT / STORAGE POLICY
==================================================

Do not Git-commit:
- dataset images
- videos
- giant JSON archives
- raw downloaded ZIPs
- Hugging Face cache blobs

Small reproducibility artifacts may be committed:
- scripts
- manifests
- source registry
- reports
- tests
- README documentation

Nothing pushed.

Preserve existing:
- Roboflow raw ZIP hashes
- SoccerTrack GSR/BAS/RAW/video
- previous audits
- frozen EyeCU artifacts

==================================================
S. FINAL INTEGRITY
==================================================

Verify at the end:

- EyeCU TRAIN unchanged
- EyeCU VAL unchanged
- EyeCU TEST unchanged
- TEST performance not accessed
- detector checkpoints unchanged
- frozen CBIoU production tracker unchanged
- no training occurred
- no credentials stored
- previous raw source hashes unchanged
- no large unintended duplicate downloads

==================================================
RETURN
==================================================

1. MASTER SOURCE REGISTRY SUMMARY

2. FINAL FOLDER TREE

3. EXISTING ROBOFLOW STATUS
   - no re-audit
   - preserved

4. KEREMBERKE
   - download status
   - audit summary
   - ball counts/sizes
   - duplicate/leakage status
   - value verdict

5. SOCCERNET-V3
   - metadata status
   - actual classes
   - ball present YES/NO
   - ball count
   - ball size distribution if possible
   - number of games/sources
   - full-download gate PASS/FAIL
   - full download status
   - value verdict

6. MARTINJOLIF
   - SKIP_DUPLICATE

7. SOCCANA
   - DEFER
   - exact reason

8. SOCCERTRACK-V2
   - detector value
   - ball value
   - event value
   - calibration value
   - MOT acquisition CLOSED
   - assets retained

9. TEAMTRACK
   - future-use classification
   - no large download

10. SPORTSLABKIT
    - research value
    - license classification
    - no code integration

11. PRIORITY MATRIX

12. BALL-DATA RANKING

13. CALIBRATION-SOURCE RANKING

14. EVENTS / IDENTITY / TRACKING FUTURE SOURCES

15. EXPERIMENT D FEASIBILITY
    STRONG YES / YES / WEAK / NO

16. EXACT NEXT RECOMMENDED ACTION

17. FILES / MANIFESTS / REPORTS CREATED

18. SECURITY CHECK
    PASS / FAIL

19. AUDIT INTEGRITY
    PASS / FAIL

20. TEST ACCESSED FOR PERFORMANCE
    YES / NO

Then STOP.

Do NOT train.
Do NOT merge external data.
Do NOT start Experiment D.
Do NOT reopen tracker selection.
Do NOT download more SoccerTrack video.
Do NOT chase SoccerTrack MOT.
Do NOT download full TeamTrack.
Do NOT download Soccana.
Do NOT download Martinjolif.
Do NOT download the original ~60 GB SoccerNet package.