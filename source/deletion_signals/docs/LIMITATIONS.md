# Limitations

Written to be usable as a manuscript section. Ordered by how much each one
constrains the conclusions, not by how comfortable it is to state.

## 1. One dataset, one model, one adapter regime

Everything is ArxivMIA + Llama-3.2-1B with QLoRA r=16 on the attention
projections. Nothing here licenses a claim about unlearning in general, about
other datasets, or about full fine-tuning.

The specific quantity most likely to be dataset-dependent is the one the whole
selectivity axis is scaled by: the **member/non-member loss gap**, +0.421 ± 0.006
nats under θ_finetuned. A dataset where fine-tuning separates members from
non-members more sharply would give a larger denominator, and RNFF values — and
therefore the position of the 0.50 bar relative to what a method achieves —
would move. The *structure* of the result (specificity degrading with strength)
does not obviously depend on that scale, but this study cannot demonstrate it.

## 2. n = 3 seeds

Every seed is reported separately and no claim rests on a p-value. Across-seed
SDs are small for the quantities that carry the argument (the RNFF denominator
varies by 1.5%, the collateral ratio ordering is identical on all three seeds),
but three points cannot support a distributional assumption, and Wilcoxon
signed-rank cannot reach p < 0.05 below six pairs at all.

Where the SDs are *not* small — RNFF at 3.5e-5 is 5.38 ± 2.76 — the runs have
passed the collapse boundary and are no longer reproducible with each other.
That instability is itself reported rather than smoothed.

## 3. Two methods

GA+GD and SCRUB. Both are first-order and both apply an undifferentiated update
to the whole adapter. The conclusion "collapse is directional" is a statement
about *these* update rules. A method whose direction is selective by
construction — influence-function-based removal, second-order or Fisher-weighted
updates, or task-arithmetic style projections — could occupy the quadrant that is
empty here, and **nothing in this study rules that out**. The paper should say
this in the discussion, not only here.

## 4. The thresholds are defensible choices, not facts

The three criteria were fixed before any confirmatory run and derived from
measured scales — the equivalence region from the observed across-seed SD, the
utility bar from a regime separated by eight orders of magnitude, the RNFF bar
from the data's own member/non-member gap. But a different reasonable analyst
could have set RNFF_forget ≥ 0.40, and **2.5e-5 would then have passed all three
axes on utility and specificity while sitting at 0.450**.

This is the single most important caveat in the study and it should be stated in
the results, not buried. The negative conclusion is: *no setting reaches half of
exact retraining while remaining specific and useful.* It is not: *GA+GD forgets
nothing.* At 2.5e-5 GA+GD achieves 45% of exact retraining's effect with retain
damage at 13% — genuinely selective, genuinely insufficient by the declared bar,
and close enough to it that the bar is doing real work in the verdict.

## 5. The decisive interval — resolved by a labelled post-hoc refinement

The magnitude and specificity thresholds are crossed at ≈2.6e-5, inside the
(2.5e-5, 3.0e-5) gap between preregistered grid points. The preregistered grid
could locate the crossing but not resolve whether a sliver of viability existed
inside it.

No grid points were added *during* the confirmatory sweep. Afterwards, a
**separately labelled refinement** (`ds_confirm_refine/`, 2.6–2.9e-5 × 3 seeds)
resolved the interval. Its points were chosen with the answer's location already
visible, so it is reported apart from the preregistered evidence and never
merged into Block B's tables.

Result: the viable region has essentially **zero width**, and is unreliable even
at its centre.

| ascent_lr | RNFF_f | RNFF_r | retain ppl | seeds passing all three |
|---|---|---|---|---|
| 2.5e-5 | 0.450 | 0.134 | 1.058 | 0/3 |
| **2.6e-5** | 0.517 | 0.193 | 1.085 | **1/3** |
| 2.7e-5 | 0.610 | 0.278 | 1.124 | 0/3 |

**The finding that outlives the grid question.** At 2.6e-5 the across-seed *mean*
of every criterion passes — RNFF_f 0.517 ≥ 0.50, RNFF_r 0.193 ≤ 0.20, retain ppl
1.085 ≤ 1.10 — while only **one seed of three** satisfies all three
simultaneously. Seed 42 fails on specificity and utility, seed 44 on magnitude;
they fail on *different* axes, so no single model is ever actually good. An
evaluation reporting means would have certified this setting.

That result is about seed variance at a boundary, not about where the boundary
sits, so it does not depend on the post-hoc choice of grid points. It is the
second demonstration in this study of an ordinary evaluation practice certifying
something that is not there — the first being membership-only evaluation passing
four destroyed settings.

## 6. The controls are not perfect matches

Exact retraining matches the original procedure in architecture, optimiser,
schedule, epochs, LoRA initialisation and split. It necessarily differs in the
one way that defines it — 800 training samples instead of 1000 — so it takes
slightly fewer gradient steps per epoch. Its retain perplexity ratio of 1.037
plausibly includes a small undertraining component alongside the genuine effect
of the missing data, which would make it a marginally conservative gold standard.

The smoke gate demonstrated how large this effect can be when training is
mismatched: at 1 epoch the null refit showed RNFF_forget 0.964, essentially all
of it undertraining. At the matched 6 epochs it falls to 0.001, which is the
evidence that the effect is negligible in the runs actually used.

## 7. Bootstrap intervals are within-run

They describe how precisely one model's AUC is estimated, not how it varies
between independently trained models. They appear only where the preregistered
rule names one, and no across-model claim rests on them.

## 8. Detector coverage is uneven by design

FUMA and the Fisher SIF were disabled for the transition sweep and the
stabilisation block — an approved scoping decision, since those blocks ask about
the three axes rather than about detector ranking, and the shadow model roughly
doubles per-sample cost. The detector comparison therefore rests on the controls
and the detector-comparison family, not on all 63 runs. Where a detector is
absent it is reported as absent, never as zero.

## 9. What "membership removed" is measured with

A single calibrated loss ratio against a pretrained-base reference, not a LiRA-
style shadow-model attack. A stronger membership attack could plausibly still
distinguish forgotten records at settings this audit scores as chance, which
would make the membership axis *optimistic* — that is, the study may
under-report how much membership evidence survives. This cuts against the
methods, not in their favour, so it does not weaken the negative conclusion, but
it does weaken any positive claim that membership was genuinely removed at the
collapsed settings.

## 10. Exploratory contamination, declared

Two analysis choices were made after seeing exploratory results: naming
`loss_ratio_vs_finetuned` the primary detector, and replacing the forget/retain
loss ratio with RNFF. Both are recorded with their reasons in
`configs/preregistration_v2.yaml` under `declared_conflicts`. The confirmatory
runs test them on data that did not select them, but the choices themselves were
not blind and the manuscript should say so.
