# M4 workflow — documented in M3, NOT executed

Sealed TEST was not accessed at any point during M3. This records the
sequence M4 must follow; it is preparation only.

```
SYSTEM FREEZE complete (M3, this record)
    -> choose all-177 or a deterministic predeclared subset
    -> automated leakage screening against TRAIN/VAL
    -> apply the predeclared deterministic exclusion/replacement rule if needed
    -> freeze/hash the FINAL frame list
    -> annotation_accessed = true
    -> exhaustive manual detection GT
    -> labels frozen
    -> M5: production predictions, ONCE
```

TEST detection ontology (for M4/M5, not used anywhere in M3):

```
player
goalkeeper
referee
ALL_VISIBLE_PHYSICAL_FOOTBALLS
```

Preconditions this M3 freeze establishes for M4 to begin:

- `SYSTEM_FREEZE_MANIFEST.json` exists and is hashed
  (`SYSTEM_FREEZE_MANIFEST.sha256`).
- No production algorithm/config/model change is permitted from this point
  until TEST is annotated and evaluated. Any change that would affect
  predictions invalidates this freeze and requires a new one, produced
  before TEST annotation/evaluation resumes.
- CBIoU reproducibility is resolved (`CBIOU_REPRODUCIBILITY.md`); M4/M5 must
  run with the deterministic settings recorded in
  `SYSTEM_FREEZE_MANIFEST.json:cbiou_reproducibility.required_deterministic_settings_for_TEST`.
- Every TEST-facing script must construct one tracker per sequence (never
  share a tracker instance across multiple sequences) — the exact defect
  this milestone fixed in the development evaluation scripts. Any new M4/M5
  script must not reintroduce the shared-tracker-across-sequences pattern.

Nothing above was run in M3. No TEST image, label, or leakage screen was
touched.
