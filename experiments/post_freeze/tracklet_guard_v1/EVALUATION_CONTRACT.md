# Tracklet Consistency Guard V1 — Evaluation Contract (frozen BEFORE any candidate is run)

Reuses the same frozen human labels as the team-assignment benchmark
(`experiments/post_freeze/team_assignment_v2/label_ui/labels.json`, SHA256
`24b6d4963d32a44df48fdadd599c2936835e73fd77093c81de10ee98dd5a7bf8`,
verified unchanged before this contract was written). Those labels are used
here for evaluation only — never modified.

## What this benchmark measures (and does not)

The measured fact is: **10 of 57 selected long-lived player tracks from two
NON-TEST development matches were human-labeled MIXED_TRACK from 5
temporally spaced crops.** This benchmark tests whether an automatic guard
can detect *that specific, already-observed kind of contamination*
(sustained cross-team identity mixing within one track ID). It does **not**
measure: all tracking errors, a per-frame tracking error rate, same-team ID
switches, or general tracker reliability. Do not describe any result from
this contract as a global "17.5% tracking error rate" or similar.

## Population

- **POSITIVE** (10 tracks): human label `MIXED_TRACK`.
- **NEGATIVE** (46 tracks): human label `TEAM_A` or `TEAM_B` (pooled — the
  A/B distinction itself is irrelevant here, only "not mixed" matters).
- **EXCLUDED** (1 track): human label `AMBIGUOUS` — insufficient evidence,
  not scored either way.

## Task

Binary, per track: **CONTAMINATED** or **CLEAN**.

## Primary safety metric

**Clean false-positive rate** (FP / 46) is primary, because splitting a
correct identity is costly (it fabricates two track IDs out of one correct
one, corrupting downstream statistics for a track that was never wrong).

## Always reported

- Mixed recall = TP / 10
- Clean specificity = TN / 46
- Clean false-positive count and rate = FP, FP / 46
- Precision = TP / (TP + FP)
- F1 = 2 · precision · recall / (precision + recall)
- Balanced accuracy = (recall + specificity) / 2
- Per-match breakdown (Bayern: 4 positive / 22 negative; Chelsea/Leeds: 6
  positive / 24 negative)
- Runtime

## Adoption gate — frozen BEFORE any candidate result is viewed

A candidate may be integrated **only if ALL of the following hold**:

1. Mixed recall ≥ 6/10
2. Clean false positives ≤ 1/46
3. At least one MIXED_TRACK detected in **both** matches (not just one)
4. No evidence that the downstream legacy TeamAssigner degrades on
   previously-clean tracks after the guard is applied (checked in the
   regression pass, §12 of the task, only if a candidate reaches this gate)

**If no candidate passes, no guard is adopted.** This gate is not weakened
after seeing results, and is not the deciding factor for whether track #4
specifically is caught — track #4 is diagnostic only (§9).

## Anti-leakage

Candidate definitions (`CANDIDATE_DEFINITIONS.md` / `candidate_config.json`
in this directory) are frozen and hashed before any candidate is scored. No
parameter may change after the first score is computed, except a disclosed
bug fix (same rule as the team-assignment benchmark) — fixed, both pre/post
behavior recorded, all affected candidates rerun, disclosed in
`FINAL_RESULTS.md`.
