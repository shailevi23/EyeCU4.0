# Pre-label Methodology Erratum

Applied **before any human label existed** — verified `label_ui/labels.json`
was absent (zero labels) at the time of this repair, so this is a
methodology fix, not a mid-labeling protocol change. No TEST access, no
detector/SN3D inference, no candidate A/B/C execution, no team-assignment
scoring, and no production pipeline change occurred while making it.

## 1. Selection preserved (unchanged)

`label_ui/selection_manifest.json` SHA256 before and after this erratum:

```
c368951fef59bf3c38ea4ebfe7a20a17cec9c68760477d2344cb0a2f2bba720c
```

Unchanged — the 57 selected tracks, their crop sampling, and the crop files
themselves were not touched. Track #4 remains selected only because it
qualifies under the deterministic selection rule (appearance_count >= 15,
longest-first, tie-break by track_id).

## 2. Label UI made actually blind

**Reason:** the frozen protocol already said TEAM_A/TEAM_B are arbitrary
match-local identities, but the UI contradicted that by (a) coloring the
buttons blue/red — a semantic-looking cue — and (b) displaying the real
match name, track_id, appearance count, and frame numbers, including in a
roster table. This matters specifically because the labeler already knows
Bayern track #4 is a previously-observed failure; an unblinded UI could let
that prior knowledge (rather than the crops) drive the label.

**Change (`label_ui/index.html`):**
- TEAM_A / TEAM_B buttons now share one neutral style (`.team-neutral`); the
  only difference between them is the button text.
- The page now shows only anonymous `Match N` (dropdown) and `Track N of
  TOTAL` (progress line) — never the real match_id, track_id, appearance
  count, or raw/processed frame numbers. The roster table shows an anonymous
  position number and label only.
- The instruction block from the frozen protocol is now shown directly in
  the UI.
- Internally, `label()` still POSTs the real `match_id`/`track_id` to
  `/save`, and `labels.json`'s schema (`"<match_id>:<track_id>": "<LABEL>"`)
  is unchanged. Verified by smoke test: saving a label from the anonymized
  UI still produces the correct real-id key in `labels.json`, and existing
  crop image paths still resolve (HTTP 200).
- The five crop images and their sampling are unchanged.

## 3. Abstention loophole closed in the evaluation contract

**Reason:** Phase 10 allows a future candidate to emit `team=None` when
uncertain, but the original contract did not say how that scores, which
would have let a candidate raise its apparent accuracy simply by abstaining
on hard tracks.

**Change (`EVALUATION_CONTRACT.md`):** added an explicit "Abstention" section
before any candidate exists:
- PRIMARY track accuracy and PRIMARY frame-weighted accuracy both use ALL
  TEAM_A/TEAM_B tracks as the denominator; an abstention counts as incorrect
  (0 credit / 0 weighted frames), identically to a wrong guess.
- Added SELECTIVE ACCURACY (accuracy over only the tracks a candidate
  actually predicted a team for) and COVERAGE (fraction of TEAM_A/TEAM_B
  tracks predicted at all) as explicit diagnostics — never primary metrics.
- Restated the adoption order as 5 steps, with coverage/selective accuracy
  demoted to diagnostics that cannot by themselves win an adoption decision
  over a candidate with better PRIMARY accuracy.

## 4. Track #4 interpretation wording corrected (data untouched)

**Reason:** `TRACK4_TRACE.json`'s `hypothesis` field
(`"TRACK_ID_CONTAMINATION"`) and the prior chat report stated the root cause
more definitively than color statistics alone can support. BGR bimodality
plus a long sustained mode transition is strong computational evidence
consistent with an ID switch, but it does not by itself prove a physical
identity change — that requires the (still-pending, still-blind) human
label for that same track to come back `MIXED_TRACK`.

**Change:** `TRACK4_TRACE.json` is **not modified** — its `hypothesis` and
`evidence` fields, and all underlying trace data, are preserved exactly as
computed. Going forward, this finding must be described in prose as:

> STRONG COMPUTATIONAL EVIDENCE CONSISTENT WITH TRACK_ID_CONTAMINATION;
> HUMAN CONFIRMATION PENDING.

If track #4's blind human label later returns `MIXED_TRACK`, the final
report may then describe the contamination as visually confirmed (not
before).

## Hashes

| file | before | after |
|---|---|---|
| `LABELING_PROTOCOL.md` | `0f702d8d61e403ecca03cb27ebe91c5467eb98656a8efd45eced1b656842743c` | `11c7e8b19de6db8b3c0a8560968516fc511727ee6e7ae2d4a4ba55cd57afe22c` |
| `EVALUATION_CONTRACT.md` | `3ee6c0fce7319ad5e2cf43ce591bbcd4299aa27fc26c3c653f17c5f71827ca6e` | `10b882e8c16233c42f0a481cf7b02e35a2cae8c84ad4c137588a3a2d1fe980ce` |
| `label_ui/selection_manifest.json` | `c368951fef59bf3c38ea4ebfe7a20a17cec9c68760477d2344cb0a2f2bba720c` | `c368951fef59bf3c38ea4ebfe7a20a17cec9c68760477d2344cb0a2f2bba720c` (unchanged) |

No labels or candidate results existed at any point during this repair.
