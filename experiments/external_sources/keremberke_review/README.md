# Keremberke human review package

Turns keremberke's two classes (`player`, `football`) into EyeCU's four
(`player`, `goalkeeper`, `referee`, `ball`) **by changing class labels only**.
No box is redrawn, moved or resized. The original export is immutable; all work
happens on `working_copy/`.

## Status

`PACKAGE_BUILT_AWAITING_HUMAN` — **0 human decisions recorded.** Every
`HUMAN_FINAL_CLASS` in the ledger is empty. The detector's proposals are
candidate filters and are never ground truth.

## Run the review

```bash
python tools/kb_review_server.py          # opens http://127.0.0.1:8733/
```

Read-only, localhost, standard library only. Decisions append to
`decisions.json` as you make them, so closing the tab loses nothing.

| key | action |
|---|---|
| `P` `G` `R` | player / goalkeeper / referee for the highlighted box |
| `U` | uncertain — revisit later |
| `Tab` | next candidate box **in the same image** |
| `N` / space | next image · `B` previous |
| `A` | accept every proposal in this image (one keypress, image on screen) |

Three modes in the dropdown, and they must not be mixed:

- **candidates** — 1,170 images / 4,153 proposed boxes. Ordered by broadcast
  run, then referee → goalkeeper → ambiguous, then by signal agreement, so each
  run's kit rule is settled once.
- **qa_player** — 250 stratified `LIKELY_PLAYER` boxes. Measures triage
  **recall**. `P` = TRUE_PLAYER, `G`/`R` = a missed official, `U` = uncertain.
- **qa_nocand** — all 57 images where nothing was flagged. Catches an official
  the triage missed entirely. `A` marks the image clean.

Before starting a run, open `reference_sheets/<run>_kits.jpg`: it shows that
run's referee, goalkeeper, team and ambiguous kits side by side.

## Then

```bash
python tools/kb_apply_review.py            # gate report only, writes nothing
python tools/kb_apply_review.py --apply    # refused unless the gate passes
```

The gate requires all candidates reviewed, no unresolved uncertain, both QA
samples complete, and no systematic missed officials (>2% in the player QA, or
any official found in a no-candidate image, blocks it). `--apply` changes class
ids only and aborts if any box geometry or the box count differs.

## Files

| file | what |
|---|---|
| `PACKAGE_MANIFEST.json` | source hashes, counts, review status |
| `ledger.json` | one row per box: BOX_ID, IMAGE, ORIGINAL_CLASS, PROPOSED_CLASS, HUMAN_FINAL_CLASS, REVIEW_STATUS, REASON_OR_GROUP *(regenerated, not tracked)* |
| `review_queue.json` | per-image review units in review order *(regenerated)* |
| `qa_likely_player.json` | the 250-box recall sample and its strata |
| `qa_no_candidate_images.json` | the 57-image census |
| `decisions.json` | **the human work — tracked in git** |
| `REVIEW_STATUS.json` | gate result, counts, ball preservation |
| `working_copy/` | the annotations being repaired *(regenerated)* |
