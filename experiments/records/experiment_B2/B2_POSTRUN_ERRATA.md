# B2 post-run errata

Factual corrections and execution repairs for the B2 experiment. **The governing
spec, `experiment_B2_spec.md`, is deliberately not edited** — it must remain as
written before any B2 result was seen. This file records what was found to be
inaccurate or repaired, without rewriting the predeclaration.

**No result, criterion, threshold or adoption decision was changed by anything
below.**

_Written 2026-08-16, after both arms completed._

---

## 1. Runtime criterion 5 — what 263.3 ms/frame actually is

The spec discloses `263.3` as a derived rather than measured figure, and states
it was computed as A's *pipeline* ms/frame scaled by the compute ratio. **The
"pipeline" part is wrong.**

[docs/coursework/COURSEWORK_PLAN.md:19](../../../docs/coursework/COURSEWORK_PLAN.md)
records the source figure explicitly:

> "~58 FPS on T4; **154.7 ms/frame on the dev CPU**"

Corrected chain of facts:

- `154.7 ms` is **dev-CPU detector-inference timing**, not full-pipeline timing.
- It was produced by `tools/compare_models.py`, which reports `ms/img` for one
  detector forward pass — the same tool and convention that produced the
  `154.7 / 164.4` pair in the Experiment C table.
- `263.3 = 154.7 x 1.702` — A's CPU detector-inference time scaled by the B2
  compute ratio.
- `263.3` is therefore a **PREDECLARED ESTIMATED CPU DETECTOR-INFERENCE
  CEILING**.

It is **not**, and must not be described as, any of:

- full-pipeline timing
- T4 timing
- a measured B@1280 runtime

**The number itself remains frozen and unchanged at 263.3.** It was predeclared,
so it stands as predeclared.

Consequence for how B2 is judged: comparing a GPU-measured B2 figure against a
CPU-derived ceiling would have been meaningless. The **primary runtime evidence
is the matched baseline-versus-B2 ratio**, both measured with
`tools/compare_models.py` on the same local CPU, in one session, at imgsz 960.
The ceiling is retained as a weak guard against gross regression only.

---

## 2. Preflight — the memory probe was defective and was repaired before training

The first version of the Colab memory-preflight cell raised before producing any
measurement:

```
ValueError: outputs must be a Tensor or an iterable of Tensors
```

**Cause.** YOLO26's training-mode forward output is nested. The probe scanned
only the top level for tensors, so the synthetic loss collapsed to a plain `int`
and `GradScaler.scale()` rejected it.

**Repair.** The probe now recurses through nested `dict`/`list`/`tuple` outputs,
keeps floating-point tensors requiring grad, builds a scalar from their squared
means, runs an AMP forward and backward, and records peak CUDA allocation. It
raises rather than continuing if no differentiable tensor is found.

**Scope.** Only the probe implementation changed. Architecture, dataset,
initialization, hyperparameters, evaluation, thresholds, arms and success
criteria were untouched.

**Timing.** The repair was made **before either arm was trained**. The defect
prevented the probe from producing a number at all, so no experimental result
was known — or knowable — when it was fixed. This is a preflight implementation
repair, not a post-result experimental change.

**Corrected measurement**, batch 3, imgsz 960, AMP, forward + backward:

| arm | batch 3 | indicative peak |
|---|---|---|
| matched baseline | FITS | **3.79 GiB** |
| B2 width-matched P2 | FITS | **6.01 GiB** |

Both well inside the T4's 14.56 GiB. The predeclared rule (*both fit -> 3 for
both*) therefore resolved to:

```
FROZEN_BATCH = 3   for both arms
accumulate   = 21
effective wd = 0.0004921875
```

Confirmed in the completed runs: `args.yaml` records `batch: 3` for both arms and
each `RUN_MANIFEST.json` records `oom_fallback_detected: false`.

The probe remains an **indicative activation-memory preflight**. It is not the
real YOLO detection loss, not a training result, and not a statement about
detector performance.

---

## 3. Diagnostic tooling added after training, and why that is not a criterion change

`tools/diagnose_temporal_ball_proposals.py` was written after both arms
finished, because no existing tool could answer the spec's layer-C question
(raw ball proposals at confidence >= 0.01 on the temporal windows).
`tools/eval_temporal_val.py` builds its pool at `BALL_CANDIDATE_CONF = 0.10`
with no override, and `tools/diagnose_detector.py` has a 0.05 floor on the
208-image VAL rather than the temporal benchmark.

The **diagnostic itself was predeclared** in the spec; only its implementation
was missing. It carries no threshold, defines no success criterion, and cannot
adopt or reject anything — the spec already states that layer C is diagnostic
and that the "gain concentrated in the small-ball windows" qualifier is NOT
PREDECLARED.

`tools/eval_temporal_val.py` was not modified; a test asserts it is
byte-identical to HEAD.

---

## 4. What did not change

- Success criteria: unchanged, all five as predeclared.
- Guardrails: unchanged (player recall >= 0.90, referee recall >= 0.75).
- Production thresholds: unchanged (candidate 0.10, accept 0.25, dedupe IoU
  0.70, match IoU 0.50) — asserted by test.
- Primary readout: unchanged, FINAL EPOCH.
- Dataset, initialization donor, training recipe, arms: unchanged.
- EyeCU sealed TEST: not accessed at any point.
