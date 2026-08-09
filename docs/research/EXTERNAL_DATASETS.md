# External Datasets — Assessment

_Last updated: 2026-08-08. Nothing here has been downloaded; this is a paper
study to decide what is worth downloading later._

EyeCU's detector classes are fixed and non-negotiable:

```
0 player   1 goalkeeper   2 referee   3 ball
```

Every external dataset is judged on one question: **can it produce those four
classes, and may we legally use it?**

---

## Summary

| Dataset | Boxes | Roles | Ball boxes | Licence | Verdict |
|---|---|---|---|---|---|
| Roboflow `3zvbc` v20 | ✅ | ✅ 4 classes | ✅ | CC BY 4.0 | **Merged** (372 frames) |
| Roboflow `2frwp` v1 | ✅ | ✅ 4 classes | ✅ | CC BY 4.0 | **Rejected** — same 15 clips, 4.4× augmented |
| Roboflow `xrbge` v6 | ✅ | ⚠️ no `player` class | ✅ | CC BY 4.0 | **Rejected** — leaky splits, wrong taxonomy |
| **SoccerNet GSR v1.3** | ✅ | ✅ player/GK/ref/other | ⚠️ unverified | **NDA** | **Recommended, gated on NDA** |
| SoccerNet Tracking | ✅ | ❌ **none** | ❌ | NDA | Not for the detector; useful in Phase 5 |
| **SoccerTrack v2** | ✅ | ✅ player/GK/ref/other | ❌ **none** | **CC BY 4.0** | **Recommended for roles + tracking** |

Two datasets are worth pursuing, for different reasons. SoccerNet GSR has the
larger volume; SoccerTrack v2 has the cleaner licence and a documented schema.

---

## 1. SoccerNet Game State Reconstruction v1.3

**What it is.** 200 broadcast clips of 30 s, ~2.36 M annotated athlete
positions, 9.37 M pitch keypoints. Derived from SoccerNet-Tracking's 12
complete games.

### Annotations

`Labels-GameState.json`, one per clip. Image-space bounding boxes are provided —
the paper states each frame includes *"bounding box annotations for the
localization of players, referees, and balls tracked over time with extra role,
team, and jersey number attributes."*

Roles, quoted from the official README:

> "Their role (player, goalkeeper, referee, other)"

### Mapping to EyeCU

| SoccerNet | EyeCU | Action |
|---|---|---|
| `player` | `0 player` | direct |
| `goalkeeper` | `1 goalkeeper` | direct |
| `referee` | `2 referee` | direct |
| `other` | — | **drop** — coaches, staff, substitutes. Our labelling rules exclude them, so importing them would contradict our own training data. |
| ball | `3 ball` | ⚠️ **unverified** |

### The ball problem

Contradictory evidence. The paper says ball boxes exist; the role list has no
ball; and the authors state they *"remove the ball as it spends significant time
in the air"* because it breaks the pitch-plane projection GSR depends on.

**Most likely:** ball boxes exist in the raw annotation but are excluded from
the GSR task. **This must be verified on a single downloaded clip before any
plan depends on class 3.** Assume GSR supplies classes 0/1/2 only.

### Licensing — the blocker

- **NDA required** for the broadcast videos; downloads are password-protected.
- `sn-gamestate` **code** is GPL-3.0. That covers the code, not the data.
- **The NDA text could not be retrieved from public sources.** Commercial use
  and derived-model terms are unknown.

**Read the NDA before downloading.** Two questions it must answer: may we train
on it, and may the resulting model be used commercially? For coursework this is
likely fine. If EyeCU ever ships, a model trained on NDA'd broadcast footage
inherits that restriction, and that is not a problem to discover late.

### Estimated gain

Annotations are per-frame at 25 fps — 750 frames per clip, 150 k frames total.
Sampling 1 frame per 3 s (matching our own extraction) gives ~2,000 useful
frames. Applying the class rates observed in comparable broadcast footage:

| Class | EyeCU now | + SoccerNet (est.) | multiple |
|---|---|---|---|
| goalkeeper | 398 | ~1,500 | **~4×** |
| referee | 1,350 | ~4,600 | ~4× |
| player | 11,908 | ~30,000 | ~3× |

Estimates, not measurements. The important part is the order of magnitude — and
that it costs **zero annotation labour**.

SoccerNet is 12 complete games, a different corpus from the DFL clips already
merged (`08fd33`, `4b770a`, …). That is ~24 new kits: real diversity, which is
exactly what the referee-confusion failure needs.

---

## 2. SoccerNet Tracking — not usable for the detector

MOT format `gt.txt`, 10 columns:
`frame, track_id, x, y, w, h, conf, -1, -1, -1`

The specification is explicit:

> "object classes are not taken into account in this challenge or the evaluation"

Tracked objects include "players, goalkeepers, referees, balls and any other
human entering the field" — **but which is which is never recorded.** There is
no way to separate goalkeeper from player, or to find the ball.

Splits: 57 train / 49 test / 58 challenge clips.

**Verdict:** useless for our four classes. Genuinely valuable in **Phase 5**,
where it is the standard benchmark with persistent track IDs for HOTA / IDF1.

---

## 3. SoccerTrack v2

**What it is.** 10 full-length university-level amateur matches, ~900 minutes,
4K panoramic full-pitch video (BePro Cerberus ×2, 3-camera panoramic ×8).

**Licence: CC BY 4.0 for the data, MIT for the code — commercial use permitted
with attribution.** No NDA. This is the cleanest licence of anything assessed.

### Components

| Component | Contains | Useful to EyeCU |
|---|---|---|
| **GSR** | `bbox_image`, `bbox_pitch`, `track_id`, `role`, `jersey_number`, `team_side`, pitch `x,y` in metres | ✅ **the one that matters** |
| **MOT** | MOTChallenge boxes + persistent IDs, "player tracking" | tracking eval only |
| **BAS** | 12 action classes, timestamps only | ❌ nothing spatial |
| **Calibration** | keypoints, camera matrices | pitch calibration work |

### GSR record schema (from `docs/format-gsr.md`)

```json
{
  "image_id": 12480,
  "track_id": 7,
  "player_id": "117092_L_9",
  "role": "player",
  "jersey_number": 9,
  "team_side": "left",
  "x": 48.21, "y": 34.07,
  "bbox_image": [1840, 710, 108, 242],
  "bbox_pitch": [46.8, 33.1, 1.4, 2.0]
}
```

One JSON array per half per match. 25 fps. `image_id = 0` is kickoff.

### Answers to the specific questions asked

**Q: Can GSR roles be joined to MOT boxes by frame + track ID?**
**No join is needed.** `bbox_image` lives inside the GSR record alongside
`role`. Roles and image boxes are already in the same row.

Caveat: `bbox_image` is marked *not required* — *"Omitted for entities derived
from pitch-plane-only annotations."* So some records carry a role and pitch
position but no pixel box. Those must be skipped. **Measure the fraction of
records with `bbox_image` on one match before planning volume.**

**Q: Do ball bounding boxes exist anywhere?**
**No. Nowhere in SoccerTrack v2.**

- GSR `role` is one of `player`, `goalkeeper`, `referee`, `other` — no ball.
- MOT is documented as *"Persistent **player** tracking"*.
- BAS was checked directly: its schema is `gameTime`, `position`, `label`,
  `team`, `player_id`, `visibility` — **no spatial fields at all.** Action
  labels tell you a pass happened, never where the ball was.

SoccerTrack v2 cannot contribute a single `3 ball` instance.

**Q: Track IDs.** `track_id` is per-half and persistent within a half, **not
guaranteed across halves**. Re-link via `player_id`, or via jersey + team side.
Note `team_side` may flip between halves for the same physical team.

### Domain caveat

Panoramic full-pitch from a fixed rig, not broadcast. Players are small, there
are no close-ups, and the camera never zooms or cuts. Excellent for roles and
tracking; it will **not** fix EyeCU's close-up blindness.

### Smallest useful subset

| Purpose | Download | Why |
|---|---|---|
| Goalkeeper / referee roles | GSR JSON for **2–3 matches**, plus the matching video only if pixels are needed | JSON is small; video is 4K and large |
| Tracking evaluation | MOT ground truth for **1–2 matches** | Enough for HOTA/IDF1 on a fixed detector |
| Pitch calibration | Calibration data for **1 match** | Enough to prototype homography |

Start with **annotations only**. The schemas are documented well enough to
build and unit-test a converter before committing to any 4K download.

---

## 4. Safe import architecture

The pattern already proven with the Roboflow merge:

```
data/frames/<prefix><clip>/     one directory per source clip
data/labels/<prefix><clip>/
data/external_provenance.json   licence, remap, date, training-only flag
```

Prefixes: `sngs_` for SoccerNet GSR, `st2_` for SoccerTrack v2.

**Non-negotiable rules:**

1. **Training data only.** External frames may never enter val or test.
   `build_dataset.py --force-train "sngs_*" "st2_*"` enforces this — pinned
   sources bypass the split entirely.
2. **Our frozen EyeCU validation set does not change.** It is the only honest
   measurement in the project. A model measured on someone else's footage is
   not measured on ours.
3. **Subsample to ~1 frame per 3 s.** Both datasets annotate at 25 fps. Importing
   every frame would give 25 near-identical images per second and drown the
   EyeCU footage hundreds to one.
4. **Record provenance per source**, including licence and NDA status, so any
   frame can be traced and removed if terms change.
5. **Drop the `other` role.** Our own rules exclude coaches, staff and
   substitutes; importing them would contradict our training data.

### Preserving track IDs separately from detector class IDs

Track IDs and class IDs must never share a file. YOLO label files stay strictly
`class cx cy w h` — adding a fifth column silently corrupts every loader.

Keep tracking ground truth in a **parallel sidecar**, exactly as pseudo-label
confidence is kept out of the label files today:

```
data/labels/st2_117092/st2_117092_000123.txt      # YOLO, detector only
data/tracking_gt/st2_117092/st2_117092_000123.json # frame, track_id, bbox, role
```

The detector never sees the sidecar. Phase 5 tracker evaluation reads it and
ignores the YOLO files. One source of truth, two consumers, no coupling.

---

## 5. Recommendation

**Both, in this order — and SoccerTrack v2 first.**

1. **SoccerTrack v2 first**, despite lower volume:
   - CC BY 4.0, no NDA, no legal question to resolve
   - schema documented publicly and precisely, so the converter can be written
     and tested before downloading anything large
   - supplies roles **and** tracking ground truth for Phase 5

2. **SoccerNet GSR second**, gated on reading the NDA. Larger volume and real
   broadcast footage, which matches EyeCU's actual domain better than
   SoccerTrack's panoramic rig.

3. **SoccerNet Tracking**: not now. Revisit in Phase 5 alongside SoccerTrack MOT.

**Neither solves the ball.** SoccerTrack has no ball boxes at all, SoccerNet's
are unverified and probably excluded. Ball remains an EyeCU-only problem, fixed
by labelling more of our own footage at higher resolution.

**Neither solves close-ups.** Both are wide-angle. The frames where our detector
returns nothing are broadcast close-ups, and only our own footage has those.

---

## Sources

- <https://www.soccer-net.org/data>
- <https://github.com/SoccerNet/sn-gamestate>
- <https://github.com/SoccerNet/sn-tracking>
- SoccerNet GSR paper: <https://arxiv.org/abs/2404.11335>
- <https://atomscott.github.io/SoccerTrack-v2/>
- <https://github.com/AtomScott/SoccerTrack-v2>
- SoccerTrack v2 GSR schema: `docs/format-gsr.md` in that repository
- SoccerTrack v2 paper: <https://arxiv.org/abs/2508.01802>
