# Team Assignment V2 — Evaluation Contract (frozen BEFORE any candidate is run)

This file is written and frozen before `labels.json` is collected and before
any of Candidate A (baseline), B (robust color), or C (SigLIP) is scored, so
no result can shape the metric definitions after the fact.

## Population

Scored population = tracks labeled **TEAM_A** or **TEAM_B** only.
`MIXED_TRACK` and `AMBIGUOUS` tracks are excluded from the accuracy
denominator entirely (see below for how they ARE reported).

## Primary metric: track-level team accuracy, permutation-corrected

For each match independently:

1. Take every TEAM_A/TEAM_B-labeled track in that match and the candidate's
   predicted team id (an arbitrary 1/2, produced by unsupervised clustering).
2. There are exactly two possible global mappings from the candidate's
   {1,2} to the human's {TEAM_A,TEAM_B} for that match: `(1->A,2->B)` or
   `(1->B,2->A)`.
3. Score the match under **both** mappings; report track accuracy under
   whichever mapping is higher for that match. This is the "best one global
   2-label permutation" rule — it is applied once per match (not per track),
   so a candidate cannot cherry-pick per-track: every track in a match uses
   the same chosen mapping.
4. **Track-level accuracy** (headline number) = correct tracks / total
   TEAM_A+TEAM_B tracks, pooled across both matches (each match's
   already-permutation-corrected predictions are pooled before dividing).

## Abstention (`team=None`/unknown) — closed BEFORE any label/result exists

Phase 10 allows a candidate to emit `team=None` when uncertain instead of
guessing. Without a rule this creates a scoring loophole: a candidate could
raise its apparent accuracy simply by refusing the hard tracks. Closed as
follows:

- **PRIMARY track accuracy and PRIMARY frame-weighted accuracy** both use
  **all** TEAM_A/TEAM_B-labeled tracks as the denominator, always — an
  abstention on a labeled track counts as **incorrect** (0 credit, 0
  weighted frames), exactly like a wrong team guess. A candidate cannot
  improve either PRIMARY number by abstaining.
- **SELECTIVE ACCURACY** (diagnostic, not primary) = accuracy computed only
  over the subset of TEAM_A/TEAM_B tracks where the candidate actually
  predicted team 1 or 2 (abstentions excluded from this one denominator).
- **COVERAGE** (diagnostic, not primary) = (TEAM_A/TEAM_B tracks the
  candidate predicted a team for) / (all TEAM_A/TEAM_B tracks).
- A **low-coverage candidate cannot win on selective accuracy alone** — see
  the restated adoption order below. Selective accuracy and coverage are
  reported for diagnosis only.

## Secondary metrics (always reported alongside the primary number)

- **Correct tracks / total** — the raw fraction behind the headline number.
- **Errors by match** — per-match correct/total and the list of
  incorrectly-predicted track_ids, so a collapse on one match cannot hide
  behind a good average.
- **Frame-weighted team accuracy** — instead of one vote per track, each
  track's correctness is weighted by its `appearance_count` (frames), then
  summed and divided by total appearance_count across TEAM_A/TEAM_B tracks.
  This shows whether errors concentrate in short-lived or long-lived tracks.
- **Coverage** — see the abstention section above; a diagnostic, not part of
  the primary accuracy calculation.
- **MIXED_TRACK count** — reported separately, per match and pooled. Never
  added to the accuracy numerator or denominator, and never scored right or
  wrong against any candidate's prediction — it is tracking-quality
  evidence, not a team-assignment test case.
- **AMBIGUOUS count** — reported separately, per match and pooled, excluded
  from accuracy for the same reason (insufficient evidence, not a defect).

## Candidate A (baseline) specifics

- **Team flips**: report whether/how often the current implementation's
  cluster-index-to-color mapping differs between independent runs on the
  same input (already observed once in `CURRENT_IMPLEMENTATION.md`: same
  two jersey colors, swapped 1/2 labels across two runs). This is exactly
  what the permutation-correction step above is designed to neutralize for
  scoring, but it is still reported as a qualitative robustness note.
- **Runtime**: wall-clock time to assign teams for all tracks in a match,
  reusing the pipeline's own team-assignment call (not re-implemented).

## Adoption rule (frozen here, restated with the abstention fix folded in)

1. PRIMARY closed-set track accuracy (all TEAM_A/B tracks, abstention = incorrect)
2. PRIMARY frame-weighted accuracy (same denominator/abstention rule)
3. Robustness across BOTH matches (no material collapse on either)
4. Coverage / selective accuracy — diagnostics only, never a deciding metric
5. Simplicity/runtime/dependencies as the tie-breaker

A candidate with lower coverage cannot win on selective accuracy alone — it
must still win on (1) and (2) above, computed over the full labeled set. If
B and C are close on (1)-(3), B wins for being lighter. C is adopted only for
a clear, repeatable improvement large enough to justify its dependency and
runtime cost. No result from this contract may be used to retroactively
change the M4/M5/M5.1 held-out claims — this is POST-FREEZE DEVELOPMENT
EVIDENCE only.
