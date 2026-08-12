# Repair contract — audit and revision

Written 2026-08-12, before the missing-target repair pass. Documentation and
design only: nothing here has been applied, and `kb_apply_review.py --apply` has
not been run.

## Why this document exists

The package was built when the only repair anyone expected was **changing class
ids on boxes that already existed**. The manifest says so, the applier enforces
it with an assertion, and both are correct for that job.

The completed human review then discovered work the contract cannot express:

- **48 live MISSING_TARGET_BOX targets** across 39 images — real targets with no
  annotation at all. Resolving them means **adding geometry**.
- **38 dispositions in force** whose documented actions are *removals*:
  `NON_TARGET_HUMAN 22 · PARTIAL_BODY_BAD_BOX 7 · BALL_WRONG_HUMAN_BOX 5 ·
  FALSE_POSITIVE 3 · EXCLUDE_IMAGE 1`.

Both conflict with `only_class_ids_may_change`. The right response is not to
loosen the checks until they stop complaining — it is to state a **new contract
that is still narrow**, and enforce that one just as strictly.

---

## Audit table

| # | Current contract | What current human decisions require | Conflict? | Required safe change |
|---|---|---|---|---|
| 1 | `PACKAGE_MANIFEST.json`: `geometry_may_change: false` | 48 human-drawn boxes must be added | **YES** | Replace with an explicit allow-list: geometry may be *added* only for a live missing-target flag, and no existing box's geometry may change |
| 2 | `PACKAGE_MANIFEST.json`: `only_class_ids_may_change: true` | Class changes **plus** additions **plus** removals | **YES** | Split into four named permissions, each with its own count reconciled against the log |
| 3 | `kb_apply_review.py`: `assert before == after` on `(id, bbox)` for the whole annotation list | Additions and removals | **YES** | Keep the assertion for every **pre-existing** id; verify additions/removals against an expected set derived from decisions, not against "no change" |
| 4 | Applier writes `category_id` only, for `goalkeeper`/`referee` only | Same, unchanged | No | None |
| 5 | Applier ignores dispositions entirely — `NON_TARGET_HUMAN` etc. change `REVIEW_STATUS` but the annotation is left in the working copy | `REMOVE_ANNOTATION_KEEP_IMAGE`, `REMOVE_ANNOTATION`, … | **YES — silent** | Implement the documented `DISPOSITION_ACTION` map, or record an explicit decision to defer it. Today the gate passes H/I/J/K on "categorised", while the export would still contain every one of those boxes |
| 6 | First-pass gate A–E (review/QA completion) | Unchanged | No | None |
| 7 | First-pass structural checks F/G/H (`class counts`, `ball geometry identical`, `no boxes added or deleted`) | H is now false by design | **YES** | Restate as: no box added or deleted **except** those reconciling exactly to human decisions |
| 8 | Second-pass gate N2 | Boxed or excluded | No — already correct | Now also rejects unrecognised resolution values |
| 9 | Second-pass gate M: ball counts must equal `{1263, 90, 474, 969}` exactly | 5 `BALL_WRONG_HUMAN_BOX` boxes may need a ball check | **POTENTIAL** | Keep the equality. If any of the 5 changes ball GT, the expected tuple must be updated **in the same commit** as the change, with the derivation shown |
| 10 | `working_copy/*.coco.json` is the mutable target; original export untouched | Unchanged | No | None — this is the part that already works |
| 11 | Source immutability: `original_annotation_sha256` per split, gate O | Unchanged | No | Add a live re-hash at apply time rather than relying on a test |
| 12 | Image exclusion | 1 `EXCLUDE_IMAGE` today, plus any from the new passes | **YES** | No mechanism exists to drop an image from the export. Needs an explicit exclusion list applied at export |
| 13 | Ledger `HUMAN_FINAL_CLASS` is the applier's input | Missing targets have **no ledger row** | **YES** | New annotations come from the decision log, not the ledger; they must get ids that cannot collide with existing ones |
| 14 | Derived reports read as state | `missing_target_queue.json` was stale by 51 flags | **YES — fixed** | Reports now carry `source_log`; the gate folds from the log; `--apply` refuses a stale gate report |

---

## The revised contract

Narrow, explicit, and each clause independently checkable.

### C1 — the original source is byte-identical
`original_annotation_sha256` per split, re-hashed at apply time, not merely
asserted by a test. Any mismatch is a hard failure.

### C2 — existing geometry is frozen
For every annotation id present in the source, `bbox` in the export equals `bbox`
in the source, exactly. The only exception is an explicit human geometry-repair
event, which does not exist yet and must not be invented silently — until such an
event type is introduced, this clause has **no exceptions**.

### C3 — class ids change only where a human said so
`category_id` may differ only for ids whose effective human answer is a role, and
the new value must equal that role. Reconciles 1:1 against `resolve()`.

### C4 — removals are enumerated, not inferred
An annotation may be removed only if its effective disposition has a documented
removal action:

| Disposition | Action | Count in force |
|---|---|---|
| `NON_TARGET_HUMAN` | `REMOVE_ANNOTATION_KEEP_IMAGE` | 22 |
| `FALSE_POSITIVE` | `REMOVE_ANNOTATION` | 3 |
| `BALL_WRONG_HUMAN_BOX` | `REMOVE_HUMAN_BOX_AND_CHECK_EXISTING_BALL_GT` | 5 |
| `PARTIAL_BODY_BAD_BOX` | `QUANTIFY_THEN_REPAIR_OR_EXCLUDE` | 7 |

The removed set must equal the set derived from the log — not a subset, not a
superset. `PARTIAL_BODY_BAD_BOX` is **not** a removal until its policy is chosen;
until then it must be counted and reported, and the gate must say so rather than
letting it drift into either bucket.

### C5 — additions are enumerated, not inferred
An annotation may be added only if it is the effective
`missing_target_resolution` of a live, non-retracted `missing_target_box` flag,
with human-drawn geometry. One flag yields at most one annotation. The added set
must equal the set derived from the log.

### C6 — ball GT is unchanged
`{instances 1263, ≤5px 90, ≤8px 474, ≤12px 969}` must hold after apply. If a
`BALL_WRONG_HUMAN_BOX` resolution changes ball GT, the expected values change in
the same commit, with the derivation recorded.

### C7 — every addition and removal carries provenance
Each carries the decision id, mode, timestamp, author, and for additions
`geometry_author: human drawn` and `no_model_proposal_used: true`.

### C8 — no model-generated geometry becomes GT
No proposal path exists in the drawing tool. If one is added later, an accepted
proposal must be recorded as a distinct event and remain distinguishable forever.

### C9 — counts reconcile exactly
```
exported = source − removed + added        per split
```
with `removed` and `added` derived from the decision log and compared
element-wise. A count that matches by coincidence while the sets differ is a
failure.

### C10 — excluded images are dropped from the candidate set, not deleted
An excluded image keeps its file and its original annotations. It is recorded in
an exclusion list and omitted from the repaired training candidate set. The gate
treats exclusion as a valid resolution; it is never a silent shortcut, and the
reason is required.

---

## Proposed manifest replacement

```jsonc
{
  "contract_version": 2,
  "immutable_source": { "sha256_per_split": { }, "verified_at_apply": true },
  "permitted_changes": {
    "class_id_change":      { "allowed": true,  "only_where": "effective human role" },
    "geometry_change":      { "allowed": false, "note": "no geometry-repair event type exists" },
    "annotation_addition":  { "allowed": true,  "only_where": "effective missing_target_resolution of a live flag" },
    "annotation_removal":   { "allowed": true,  "only_where": "effective disposition with a documented removal action" },
    "image_exclusion":      { "allowed": true,  "only_where": "explicit EXCLUDE_IMAGE with a reason" },
    "ball_gt_change":       { "allowed": false, "note": "unless a BALL_WRONG_HUMAN_BOX case documents one" }
  },
  "reconciliation": "exported = source - removed + added, set-equal to the decision log",
  "provenance_required": true,
  "no_model_geometry": true
}
```

## What is deliberately still open

- **`PARTIAL_BODY_BAD_BOX` policy (7 boxes).** Repair the geometry, or exclude?
  Repair needs a geometry-repair event type that does not exist. Not decided.
- **The disposition removals are not implemented in the applier.** The gate
  currently passes H/I/J/K on *categorised*, which is honest about the review but
  not about the export. This must be built or explicitly deferred before apply.
- **Image exclusion is not implemented in the applier.**
- **New annotation ids.** Proposal: allocate above `max(existing id)` per split,
  deterministically ordered by `(IMAGE, flag recorded_utc)`, each carrying its
  `missing_target_id`. Not implemented.
