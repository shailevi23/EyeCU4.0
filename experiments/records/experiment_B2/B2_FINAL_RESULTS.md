# Experiment B2 — final results

Permanent record of the width-matched P2 architecture experiment. Predeclaration
is `experiment_B2_spec.md` (unchanged since before training); execution repairs
and the runtime-criterion correction are in `B2_POSTRUN_ERRATA.md`.

```
run          b2_20260816T092152Z
git HEAD     c5ce3e23ac382937c0cd36fb39775fa7c4cb699e
GPU          Tesla T4 (14.56 GiB)
donor        yolo26s.pt  sha256 646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b
data         823 EyeCU TRAIN / 208 EyeCU VAL, original baseline, no Stage-A data
```

**Verdict: NOT ADOPT.**

> **B2 DID NOT JUSTIFY ADOPTION UNDER THE FROZEN 80-EPOCH TRAINING BUDGET.**

---

## A. Run validity

| | epochs | batch | OOM fallback | parameter transfer |
|---|---|---|---|---|
| matched baseline | 80/80 | 3 | false | 99.969% (backbone 100%) |
| B2 | 80/80 | 3 | false | 58.2366% (backbone 100%) |

Both arms: imgsz 960, AdamW, lr0 0.00125, momentum 0.9, weight_decay 0.0005,
warmup_bias_lr 0.0, seed 0, deterministic, patience 0, accumulate 21, effective
weight decay 0.0004921875. Both initialized independently from the same
`yolo26s.pt`; neither was chained from the other.

## B. Primary readout — final epoch (`last.pt`), 208-image VAL

| metric | baseline | B2 | delta |
|---|---|---|---|
| player recall | 0.9338 | 0.9245 | −0.0093 |
| goalkeeper recall | 0.3652 | **0.4981** | **+0.1329** |
| referee recall | 0.7160 | **0.6654** | **−0.0506** |
| ball precision | 0.7079 | **0.8320** | **+0.1241** |
| ball recall | 0.4685 | 0.4414 | −0.0271 |
| ball mAP50 | 0.5428 | 0.5346 | −0.0082 |
| ball mAP50-95 | 0.2490 | 0.2362 | −0.0128 |
| all mAP50 | 0.6777 | **0.7134** | **+0.0357** |
| all mAP50-95 | 0.4308 | **0.4437** | **+0.0129** |

`results.csv` final rows corroborate the all-class figures to ±0.0005.

## C. Hard guardrails

```
player recall  >= 0.90 :  0.9245   PASS
referee recall >= 0.75 :  0.6654   FAIL
```

**The referee guardrail failure alone prevents adoption under the frozen spec**,
independently of every other result. The spec states it directly: *"If a human
guardrail fails, B2 fails even if ball recall improves."* The width match existed
specifically to protect the human classes from the reinitialised head, and on
referee it did not.

## D. Temporal benchmark (104 frames, frozen `tools/eval_temporal_val.py`)

| | baseline | B2 | delta |
|---|---|---|---|
| raw ball recall | 52/77 = **0.6753** | 48/77 = **0.6234** | −0.0519 |
| selector trajectory coverage | **0.7273** | **0.6883** | −0.0390 |
| ball precision | 0.8966 | 0.9231 | +0.0265 |
| FP per frame | 0.0577 | 0.0385 | −0.0192 |

```
Criterion 1  raw ball recall > 0.5714  :  0.6234   PASS
Criterion 2  coverage >= 0.7143        :  0.6883   FAIL
```

Both arms beat historical A (0.5714), so the pinned 80-epoch recipe is an
improvement over the historical run. B2 loses that gain relative to the matched
control on both measures. Note the matched baseline **passes** criterion 2 at
0.7273 while B2 fails it — the regression is attributable to the architecture,
not to the recipe.

## E. Predeclared small-ball target windows

| window | baseline | B2 | delta |
|---|---|---|---|
| `women_1 w1` (7–11 px, the P2 target regime) | 3/6 = 0.5000 | 3/6 = 0.5000 | **0.0000** |
| `youth_premier_league w1` (18.9–24.2 px) | 3/12 = 0.2500 | 3/12 = 0.2500 | **0.0000** |

**The stride-4 P2 branch produced no measurable gain in either predeclared
diagnostic window.**

Every B2 temporal loss came from outside the target regime:
`bayern_munich w0` −1, `bayern_munich w1` −1, `youth_premier_league w0` −2.

## F. Zero-proposal diagnostic

Raw detector proposals at confidence >= 0.01, IoU >= 0.50, selector not
involved. Tool: `tools/diagnose_temporal_ball_proposals.py` (DIAGNOSTIC ONLY,
carries no success criterion).

**Historical population reproduced exactly.** Running the tool on the historical
`best_A_960.pt` returns **17 misses / 12 proposal-less** in the two target
windows, matching the recorded figures and validating the implementation against
the historical record.

| all windows | missed | no usable proposal @0.01 |
|---|---|---|
| historical A | 33 | 24 |
| matched baseline | **25** | **14** |
| B2 | **29** | **19** |

| two target windows | missed | no usable proposal |
|---|---|---|
| historical A | 17 | 12 |
| matched baseline | 12 | 7 |
| B2 | 12 | **9** |

Of the **14** balls for which the baseline produced no box at any confidence,
B2 gained a usable >= 0.01 proposal for **1**, and fully detected **1** other
via the frozen path. Meanwhile the aggregate proposal-less count **rose from 14
to 19**.

This is the most direct evidence in the experiment. If insufficient spatial
resolution were the binding constraint on small-ball detection, the
proposal-less count had to fall. It rose.

## G. Runtime

Matched measurement: same machine, same session, same tool
(`tools/compare_models.py`), imgsz 960 both, conf 0.25, 208-image VAL.

```
baseline : 233.42 CPU detector ms/img  (4.3 FPS)
B2       : 443.23 CPU detector ms/img  (2.3 FPS)
ratio    : 1.899x

263.3 predeclared estimated CPU detector-inference ceiling : FAIL
```

**The matched same-machine ratio is the primary runtime evidence.** The measured
1.90× exceeds the 1.70× GFLOPs ratio — P2 costs more in practice than FLOPs
predict.

The 263.3 ceiling is retained as predeclared but is a weak guard: it derives
from `154.7 × 1.702` where 154.7 was CPU detector-inference timing on an earlier
dev machine, and this machine measures the baseline at 233.4 ms. See
`B2_POSTRUN_ERRATA.md` §1.

## H. Learning curves — longer schedule NOT justified

```
B2 best mAP50     0.7324 @ epoch 61   ->  final 0.7129
B2 best mAP50-95  0.4683 @ epoch 62   ->  final 0.4439
```

Final-20-epoch slopes are negative for every metric (mAP50 −0.00141/epoch,
mAP50-95 −0.00154/epoch, recall −0.00321/epoch), and mean(71–80) is below
mean(61–70) throughout (−0.0212, −0.0207, −0.0459). Last-10 mAP slopes are
marginally positive (+0.0019, +0.0014) but sit inside the run-to-run noise
(σ ≈ 0.006–0.007).

The baseline behaves the same way from an earlier peak (best mAP50 0.7330
@ epoch 36 → 0.6775 final). Both arms are past peak at epoch 80, not still
climbing.

The spec permits a longer-schedule follow-up **only** on concrete evidence of
material improvement at epoch 80. That evidence does not exist, so the branch
closes.

## I. Interpretation

**What improved.** All-class mAP50 (+0.0357) and mAP50-95 (+0.0129). Goalkeeper
recall substantially (+0.1329). Ball precision (+0.1241), with temporal FP per
frame falling 0.0577 → 0.0385.

**What did not.** Ball recall (−0.0271). Both predeclared small-ball windows
(0.0000 change). Selector trajectory coverage (−0.0390). The zero-proposal
mechanism (14 → 19 proposal-less). Referee recall (−0.0506, breaching a hard
guardrail). Runtime (1.90× cost).

B2 did not fail because the extra detection level was broadly harmful — several
metrics improved, and the improvement in goalkeeper and ball precision is real.
It failed because the stride-4 level **delivered nothing in the regime it was
designed for**, while the reinitialised neck and head cost accuracy elsewhere,
at nearly twice the compute.

> **B2 DID NOT JUSTIFY ADOPTION UNDER THE FROZEN 80-EPOCH TRAINING BUDGET.**

### What this experiment does not establish

**"P2 cannot work" is NOT supported by this experiment.** B2 inherited only
58.24% of its parameters against the baseline's 99.97%, because adding a
detection level necessarily reinitialises the neck and the whole head. The spec
recorded that asymmetry before training and fixed the scope accordingly: this is
an adoption test under a fixed budget, not a test of whether stride-4 features
can help in principle.

Equally not established: that resolution is irrelevant to small-ball detection,
or that the remaining misses have a single cause.

### Secondary diagnostic — fixed-threshold comparison

At a **fixed matched confidence of 0.25** applied identically to both arms
(`tools/compare_models.py`, paired on the 208-image VAL):

| class | n | baseline R | B2 R | delta |
|---|---|---|---|---|
| player | 2490 | 0.9382 | 0.9321 | −0.0060 |
| goalkeeper | 115 | 0.3478 | 0.5043 | +0.1565 |
| referee | 257 | 0.6693 | 0.6693 | 0.0000 |
| ball | 111 | 0.4685 | 0.4775 | +0.0090 |

At a matched threshold, referee recall is **identical** and ball recall is
marginally **better** for B2. The primary figures come from Ultralytics `val()`,
which selects one shared operating point per model by maximising class-mean F1 —
a per-model threshold, documented for the earlier A-vs-B comparison in
`docs/results/RESULTS.md`. Part of the headline referee gap is therefore
operating-point selection rather than pure capability.

**This is secondary diagnostic information and does not replace or alter the
predeclared primary readout.** The spec fixes the primary readout as the final
epoch under the standard evaluation, that readout fails the referee guardrail,
and the verdict stands.

---

## Evaluation artifacts

Outside the repository, at
`C:\Users\shail\Desktop\EyeCU_B2_results\eval_b2_20260816T092152Z\`:
`temporal_baseline.json`, `temporal_b2.json`, `zeroprop_historical_A.json`,
`zeroprop_baseline.json`, `zeroprop_b2.json`, `runtime_and_paired_val.json`,
plus logs.

**EyeCU sealed TEST was not accessed at any point in this experiment.**
