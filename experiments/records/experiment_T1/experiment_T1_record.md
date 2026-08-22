# Experiment T1 — temporal candidate recovery — final disposition

```
STATUS    CLOSED
VERDICT   FAIL
ADOPTED   NO
```

## Provenance of this record — read first

**This record was reconstructed after the fact.** The original T1 result artifact
was never persisted: at the time this file was written the repository contained
no T1 spec, no T1 result JSON, no T1 entry in `experiments/records/`, and no T1
section in `docs/results/RESULTS.md`. The tool
(`tools/experiment_temporal_candidate_recovery.py`) and its containment test
(`tests/test_experiment_temporal_candidate_recovery.py`) were both untracked and
had never been committed.

The figures below are the **project owner's supplied measured result**, recorded
here as the formal disposition that was missing from the repository. They were
**not** re-measured when this file was created, and T1 was **not** re-run.

No historical file or timestamp has been fabricated. This document is dated by
its own commit, not by the original experiment.

## What T1 tested

T1 changed exactly one thing against the frozen production path: the floor of the
`BallTemporalSelector` rescue pool, from `BALL_CANDIDATE_CONF` 0.10 down to 0.01.
Everything else — the detector, its weights, imgsz, the 0.25 accept threshold,
`suppress_ball_duplicates` at `BALL_DEDUPE_IOU`, the 60 px base + 40 px growth
gate, cut handling, and the whole pass-2 interpolation — was reused unchanged.

Benchmark: the frozen 104-frame temporal validation set, 77 GT-ball frames and
27 GT-empty frames. VALIDATION ONLY.

## Measured result

| | control (floor 0.10) | T1 (floor 0.01) | delta |
|---|---|---|---|
| observed + recovered | 54/77 = 0.7013 | **59/77 = 0.7662** | **+0.0649** |
| trajectory coverage | 56/77 = 0.7273 | **59/77 = 0.7662** | **+0.0390** |
| hallucinated empty frames | 2/27 = 0.0741 | **6/27 = 0.2222** | **+0.1481** |

## Verdict

**T1 FAILED. The 0.01 candidate floor was NOT adopted.**

Recall improved on both measures, and that gain is real. It was bought by
tripling the hallucination rate on GT-empty frames, 2/27 → 6/27. A ball asserted
on a frame that has none is not a cheaper error than a missed ball: it feeds
possession and team-control statistics as though it were an observation. The
recall gain does not justify that cost, so the floor stays where it was.

## Binding consequences for production

These are the invariants T1's closure leaves behind, and the containment test
`tests/test_experiment_temporal_candidate_recovery.py` exists to enforce them:

- production `BALL_CANDIDATE_CONF` remains **0.10**
- production `BALL_ACCEPT_CONF` remains **0.25**
- production `BALL_DEDUPE_IOU` remains **0.70**
- the experimental **0.01 floor must remain confined to T1 experiment code** and
  must never be written back into `trackers/detector.py` or any production path
- `trackers/ball_temporal.py` frozen gate constants remain unchanged
- `tools/eval_temporal_val.py`, the frozen evaluator, remains unchanged

## Note on the containment guard

The guard originally also asserted that `trackers/detector.py` was byte-identical
to git HEAD. That clause was correct while T1 was open and is obsolete now: D2
made an authorised architecture change to that file (the selectable SN3D ball
branch). The clause was replaced with assertions that protect what T1 actually
requires — the production thresholds, the confinement of the 0.01 floor, and the
frozen selector/evaluator — rather than freezing an unrelated file forever. The
guard still fails if the 0.01 floor reaches production.
