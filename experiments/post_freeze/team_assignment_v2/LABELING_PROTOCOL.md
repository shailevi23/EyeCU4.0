# Team Assignment V2 — Labeling Protocol (frozen)

## What is being labeled

57 player tracks total, selected deterministically (see `select_tracks.py` /
`SELECTION_MANIFEST` below) from two NON-TEST matches:

- `bayern_munich_3-1_chelsea` — 27 tracks (from the existing
  `demo_outputs/final_e2e_demo/cache/tracks_b3bacf1707645184.pkl`)
- `chelsea_v_leeds_united` — 30 tracks (from the one additional authorized
  cached run, `experiments/post_freeze/team_assignment_v2/match2_cache/`,
  300 processed frames, `cache_key=7562185f8dd2d6c6`)

Selection rule (prediction-independent): player tracks only (goalkeepers/
referees excluded) with `appearance_count >= 15`, sorted by appearance count
descending then track_id ascending, top 30 per match. Track #4 (Bayern) is
included because it qualifies on this rule alone — it was not hand-picked
for its known team-assignment failure (see `TRACK4_TRACE.json`).

## How to label

1. `cd experiments/post_freeze/team_assignment_v2/label_ui`
2. `python server.py` (default port 8765)
3. Open `http://127.0.0.1:8765/index.html`
4. The UI is **blind**: it shows only anonymous "Match N" / "Track N of
   TOTAL" indices — never the real match name, track_id, appearance count,
   or frame numbers, and the roster shows only an anonymous position number
   and label. This matters specifically because Bayern track #4 is a
   previously-observed failure the labeler already knows about; hiding the
   real id prevents that prior knowledge from leaking into the label. The
   real `match_id`/`track_id` are still sent to the server internally on
   save (the `labels.json` schema is unchanged), just never displayed.
5. For each track, 5 temporally-spaced central-player crops are shown (bbox
   + 10% padding, no chest-ROI-only crop, no prediction overlay, no team
   color, no confidence). TEAM_A and TEAM_B buttons use the **same neutral
   styling** — text is the only difference, so button color cannot leak a
   "meaning" for A or B. Click exactly one:

   - **TEAM_A** — one of the two match-local jersey identities (arbitrary
     which one is "A" per match; see scoring note below)
   - **TEAM_B** — the other jersey identity
   - **MIXED_TRACK** — the 5 crops appear to show more than one team
     identity for this single track ID (likely ID-switch/tracklet
     contamination)
   - **AMBIGUOUS** — insufficient visual evidence to decide (motion blur,
     too small, occluded in all 5 crops, kit not visible)

5. Each click **autosaves immediately** via `POST /save` into
   `label_ui/labels.json` (keyed `"<match_id>:<track_id>"`); re-clicking a
   track overwrites its previous label. No manual export step.

## TEAM_A / TEAM_B are match-local and arbitrary

The labeler is not told which real jersey color is "A" vs "B", and does not
need to be consistent about it across the two matches — "A" in the Bayern
match and "A" in the Chelsea/Leeds match are independent, arbitrary choices.
This mirrors the fact that the underlying clustering algorithms (KMeans) also
produce arbitrary, unstable cluster indices (confirmed in
`CURRENT_IMPLEMENTATION.md` D/E: two separate production runs on the same
Bayern cache assigned the *same two jersey colors* to *swapped* team-index
numbers). The scoring contract (Phase 4, frozen separately in
`EVALUATION_CONTRACT.md`) accounts for this: each match is scored under
whichever of the two possible global A/B<->1/2 permutations gives the higher
accuracy, so the labeler's arbitrary A/B choice never penalizes a
correct-but-oppositely-numbered prediction.

## MIXED_TRACK handling (frozen in advance, before any candidate is run)

`MIXED_TRACK` is never scored as a normal classifier error. It is evidence
that a single track ID does not correspond to one consistent identity for
its full lifetime — a tracking/tracklet-contamination signal, not a team-
assignment signal. See `EVALUATION_CONTRACT.md` for exactly how it is
reported (as a separate count, not folded into denominator/numerator of
"track accuracy").

## Freeze

- `label_ui/selection_manifest.json` (+ `.sha256`) — the 57 selected tracks
  and their crop paths, frozen before any label was collected.
- This file (`LABELING_PROTOCOL.md`) — frozen before any label was
  collected.
- `label_ui/labels.json` — the actual human labels, produced by autosave;
  frozen (hashed) once labeling is reported complete, before Phase 4
  scoring begins.

No candidate (baseline, robust-color, SigLIP) result may be viewed before
`labels.json` is frozen, so no result can retroactively shape the labels.
