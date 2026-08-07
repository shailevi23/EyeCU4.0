# Labeling Guide

How to turn 1,483 extracted frames into a training set — starting with one
batch of ~450, not all of them.

## Why batches

Labeling all 1,483 frames before knowing whether the approach works would be
25–70 hours spent before the first signal. Instead:

1. Correct ~450 **train** frames → train a baseline.
2. Use that baseline to draft the next batch — it will be far better at this
   footage than a generic model, so corrections get faster each round.
3. Repeat, prioritising frames the current model gets wrong.

**Val and test are never part of this loop.** Each round's model has seen the
frames it helped label, so letting it touch val or test would contaminate the
only honest measurement available. Those splits are corrected by hand, once.

---

## Step 1 — Set your API key

The key is read from the environment only. It is never written to disk, never
committed, and never passed on the command line.

```powershell
$env:ROBOFLOW_API_KEY = "your-key"      # PowerShell
```
```bash
export ROBOFLOW_API_KEY="your-key"      # bash
```

Get one at <https://app.roboflow.com/settings/api>. If the key is missing the
tool stops with an explicit message rather than silently producing nothing.

> The old key `bzHGvvsL4gjqNOHIPR5J` was committed to this repo and must be
> treated as compromised. Revoke it and issue a new one.

## Step 2 — Draft the batch

The first batch is already selected: `data/batches/batch_01.json`
(451 frames, train sources only).

```bash
python tools/pseudo_label.py --batch data/batches/batch_01.json --dry-run   # preview
python tools/pseudo_label.py --batch data/batches/batch_01.json             # run
```

Useful flags:

| Flag | Effect |
|---|---|
| `--dry-run` | Show what would be sent. No API calls, no writes. |
| `--limit N` | Stop after N images — use this to sanity-check output first. |
| `--source X` | Restrict to one source match (repeatable). |
| `--confidence` | Detection threshold, default `0.30`. Lower catches more balls and more junk. |
| `--refresh-drafts` | Regenerate untouched drafts. Never touches corrected labels. |

Output:

- `data/labels/<source>/<frame>.txt` — YOLO drafts, clean format
- `data/pseudo_meta/<source>/<frame>.json` — per-box confidence and provenance

Confidence is kept out of the label files on purpose, so the labels stay valid
YOLO and the metadata can be regenerated or discarded independently.

**Your corrections are safe.** The tool records a hash of each draft it writes.
On any later run it classifies every label as `draft`, `edited`, `foreign` or
`absent`, and only ever writes to `absent` (or `draft` with `--refresh-drafts`).
A label you have edited, or one it did not create, is never overwritten.

## Step 3 — Correct in CVAT or Roboflow Annotate

Import `data/frames/<source>/` plus `data/labels/<source>/` as **Ultralytics
YOLO** format. Classes must be declared in this exact order:

```
0 player
1 goalkeeper
2 referee
3 ball
```

Then fix the drafts. Expect roughly 1–3 minutes per frame.

### Labeling rules

**Occlusion** — label a partially occluded player whenever you can identify
them as a person. Box only the visible extent; do not guess where hidden limbs
are. If you cannot tell it is a person, do not label it.

**Goalkeeper stays goalkeeper.** Do not relabel a goalkeeper as a player just
because the model did. This is the single most common draft error, and it is
the reason the class exists: a keeper's kit deliberately differs from their own
team's, which is exactly what breaks jersey-colour team assignment.

**Referee stays referee** — including assistant referees on the touchline, and
referees at long range. Never assign a referee to a team.

**Ball only when visually identifiable.** If you cannot see it, leave it out.
Do not infer position from player gaze or from the previous frame. A missing
ball is honest; a guessed ball teaches the model to hallucinate.

**Remove duplicate boxes.** Two boxes on one player is the specific failure
this project is chasing. If two boxes cover the same person, delete one.

**Keep boxes tight and consistent.** Enclose the visible player, no generous
margin. Consistency across frames matters more than any single perfect box.

**Not players:** coaches, substitutes, medical staff, ball boys, spectators.
Leave them unlabelled — they are useful hard negatives.

**Empty frames are valid.** A crowd shot, bench or advertising board with no
labels is a real training signal. Keep the empty `.txt` file.

## Step 4 — Export and validate

Export YOLO format back into `data/labels/`, preserving `<source>/` folders.

```bash
python tools/validate_annotations.py --strict
```

This fails on corrupt images, missing or orphaned labels, malformed lines,
class ids outside 0–3, coordinates outside `[0,1]`, and boxes past the image
edge. It must pass before training.

## Step 5 — Build and train

```bash
python tools/build_dataset.py --plan-only          # preview the split
python tools/build_dataset.py --zip                # build it
python tools/build_dataset.py --check              # verify no leakage
```

Then upload `data/football_dataset.zip` to Drive and open
`notebooks/EyeCU_Train_Colab.ipynb`.

---

## First-training gate

Do not start training until all of these hold:

- [ ] ~400–500 **train** frames corrected (batch 1 is 451)
- [ ] Enough **val** frames corrected to measure all four classes — val must
      contain a non-zero count of `goalkeeper`, `referee` and `ball`, or those
      per-class scores are meaningless
- [ ] `python tools/validate_annotations.py --strict` exits 0
- [ ] `python tools/build_dataset.py --check` reports no leakage

Test stays untouched until a model has been selected on val.

## Splits (seed 42, do not change without rebuilding)

| Split | Sources |
|---|---|
| **train** | betis_3_vs_5_fc_barcelona, chelsea_v_leeds_united, chelsea_v_leicester_city, clemson_vs__notre_dame, croatia_1-1_czechia, fc_barcelona_3_vs_1_atletico_de_mad, full_match_chelsea_vs_arsenal, sunday_league_full_match, women_2, youth_1, youth_3, youth_4, youth_5, youth_7 |
| **val** | austin_fc_vs__club_tijuana, bayern_munich_3-1_chelsea, women_1, youth_premier_league |
| **test** | como_2-0_sassuolo, manchester_city_v_liverpool, youth_2 |

`short` and `08fd33_4` are excluded entirely — they are the fixtures the
regression tests run against.
