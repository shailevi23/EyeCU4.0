# P1 erratum — provenance metadata defect (metadata-only, no score change)

`experiments/records/experiment_P1/P1_POSSESSION_RESULT.json` was written by
`tools/eval_possession_val_p1.py` with `annotation_sha256` hardcoded to `null`
— the field was declared in the report schema but never populated from the
annotation file actually loaded at run time. Every other field in the file
(rows, metrics, correspondence records) was computed from the correctly loaded
`POSSESSION_VAL_V1_ANNOTATIONS.json` and is unaffected; this was a metadata
omission only, not a scoring defect.

Fix applied: the single field `annotation_sha256` was set to
`df7d142c72963703d1848adc0d00906125e60b5279495830e71aff9e850e36d1`, the sha256
of `data/possession_val_v1/POSSESSION_VAL_V1_ANNOTATIONS.json` as frozen at the
close of M1 (re-verified identical before this edit). No other field was
touched. No row, count, or metric was recomputed or altered.

| | sha256 |
|---|---|
| `P1_POSSESSION_RESULT.json` before this edit | `b69306c290f2b5a80bdbbcf3e602f3ea46b96450513c9edf392fe415835abcf7` |
| `P1_POSSESSION_RESULT.json` after this edit | `c302a7d22b9ad2b0113ad27e0db2a1fb87e994764d4a4de3b4267edaea952672` |
| `POSSESSION_VAL_V1_ANNOTATIONS.json` (unchanged, referenced) | `df7d142c72963703d1848adc0d00906125e60b5279495830e71aff9e850e36d1` |

This erratum is recorded rather than silently applied so the file's history
stays auditable: anyone diffing the two hashes above can confirm the only
change was the one field named here.
