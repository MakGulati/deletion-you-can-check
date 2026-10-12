# Pre-flight criterion — base-model calibration on BookMIA + Qwen2.5-1.5B

**Written before the Qwen check was run.** The reference measurement below was
computed first, from already-saved ArxivMIA scores, because it defines what
"clean" looks like; the criterion is expressed relative to chance, not to
whatever BookMIA turns out to give.

## What is being checked, and why it matters

The membership axis scores `loss_ratio = L_unlearned(z) / L_base(z)`. That
calibration **assumes the pretrained base model never saw the data**: dividing
by `L_base` is supposed to remove intrinsic sample difficulty, leaving
membership.

BookMIA's `label=1` marks snippets from books that were in *pretraining*
corpora. If Qwen2.5-1.5B's pretraining included them, `L_base` is already low on
our "members" and the denominator carries membership rather than difficulty —
the axis would then partly measure Qwen's pretraining, not our fine-tuning.

A second mechanism produces the same failure: label=1 and label=0 books may
simply differ in distribution (era, genre, register). This check cannot separate
memorisation from distribution shift, and it does not need to — **either would
degrade the calibration in the same way**, and the criterion is about the net
effect.

## The measurement

Score D_forget, D_retain and D_test under the **pretrained base only** — no
fine-tuning, no adapters — and compute

```
AUC( member vs unseen ),   score = -L_base(z)
```

members = forget ∪ retain, unseen = test. Lower base loss ranks as more
member-like, so a base model that has memorised the members gives AUC > 0.5.

## Reference: ArxivMIA + Llama-3.2-1B

Computed from `ds_confirm/controls/exact_retrain/seed*/per_sample_scores.json`,
which already store `loss_base` per row:

| seed | base member_vs_unseen AUC | mean base NLL, members | unseen |
|---:|---:|---:|---:|
| 42 | 0.4568 | 2.470 | 2.403 |
| 43 | 0.4871 | 2.457 | 2.429 |
| 44 | 0.4440 | 2.466 | 2.379 |

**Mean 0.4626 ± 0.0221; deviation from chance |AUC − 0.5| = 0.037.**

The base model marginally *dis*favours members — they are slightly harder, not
easier. The calibration is clean on ArxivMIA.

## The criterion, graded and fixed in advance

Let `d = |AUC_base − 0.5|`, averaged over seeds 42/43/44.

| outcome | condition | consequence |
|---|---|---|
| **PASS** | `d ≤ 0.10` | Calibration usable. Proceed with the full BookMIA matrix as planned; report the base AUC in the paper as evidence the calibration holds. |
| **CONDITIONAL** | `0.10 < d ≤ 0.20` | Proceed, but the membership axis on BookMIA carries a **declared calibration caveat**, and the base AUC is reported beside every membership number so a reader can discount it. |
| **FAIL** | `d > 0.20` | The base calibration is degenerate. BookMIA results are reported on the other three axes only (selective forgetting, utility, detector comparison), and the membership axis is declared uninterpretable on this dataset/model pair rather than reported with a caveat. |

`0.10` is not arbitrary: it is the **full width of the membership
chance-equivalence region**, [0.45, 0.55], which the study already uses to
decide when membership evidence has been removed. A base model that deviates by
more than that region is wide has moved the axis by more than the axis's own
resolution.

## What this check does NOT do

It does not license using BookMIA if it passes. It only establishes that the
membership axis means on BookMIA what it means on ArxivMIA. Every other
limitation — n=3, truncation, the method set — is unaffected.
