# Confirmatory preregistration v2 — three-axis unlearning audit on ArxivMIA

The machine-readable version is
[`../configs/preregistration_v2.yaml`](../configs/preregistration_v2.yaml) and
it is authoritative: `deletion_signals.audit` reads its thresholds from that
file rather than from constants in code, so a number cannot drift from the
document that declares it. This page explains why each number is what it is.

**Scope.** ArxivMIA + Llama-3.2-1B QLoRA only. Nothing here licenses a
cross-dataset claim; that is future work and the manuscript must say so.

## Status of the earlier work

The runs in `results/` and `ds_sweep/` are **exploratory**. They generated the
hypotheses tested here and cannot confirm them. Confirmatory runs go to
`ds_confirm/`, are never mixed into a headline table with exploratory ones, and
are analysed under the rules below — which were fixed before any of them
existed.

What the exploratory phase established, and which of it survives as a
hypothesis rather than a finding:

| Observation | Status |
|---|---|
| `forget_vs_test` is confounded by ordinary membership | Established — AUC 0.001 under the declared direction, i.e. forgotten records still read as members |
| GA+GD reaches chance membership only after utility collapses | Established across 5 strengths × 3 seeds |
| The transition sits between 2e-5 and 5e-5 | Established coarsely; its shape is the subject of Block B |
| Two-checkpoint calibrated loss matches signed cosine | **Hypothesis** — must be retested where it was not selected |
| Signed cosine's advantage over cos² nearly vanishes at 2e-5 | **Hypothesis** |
| SIF is inert | Established — ρ(sif, grad_norm_sq) = 0.9999 |
| SCRUB shows no directional advantage | **Untested at adequate strength** — `final_forget_kl` was 0.098, so that setting barely moved |

## The central question

> Can an unlearning method make ArxivMIA forgotten records cease looking like
> training members, preserve retained/test utility, and nevertheless leave a
> detectable deletion-set fingerprint?

Three axes, evaluated **jointly**:

| Axis | Comparison | Question |
|---|---|---|
| Membership removal | forget vs test | Do forgotten records still look like members? |
| Deletion specificity | forget vs retain | Can an auditor identify which members were targeted? |
| Utility | retain/test, before vs after | Does the model remain useful? |

No setting is called successful on membership AUC alone. That failure mode is
not hypothetical: in the exploratory sweep, three of five settings reached the
chance region, and all three had destroyed the model.

## Where each threshold comes from

Every number below was set by inspecting the *scale* of the exploratory
measurements, before any confirmatory run. The reasoning is recorded here so a
reader can check that each is a real bar rather than one drawn around a known
answer.

### Membership removal — AUC ∈ [0.45, 0.55]

A two-sided **equivalence region**, not an upper bound. Under the preregistered
higher-is-forget direction the exploratory runs give 0.001–0.004 when the model
remembers and 0.475–0.479 after collapse; the across-seed SD is 0.001–0.004. A
one-sided rule ("AUC ≤ 0.55") would be satisfied by the runs carrying the
*strongest possible* membership evidence, since 0.001 is as informative as
0.999 — it is the same signal read in the other direction. ±0.05 is more than
ten times the observed SD, so the region is not merely absorbing noise.

### Utility — retain and test perplexity ratio ≤ 1.10

Observed: 0.970 (retain) and 1.036 (test) at the strongest viable strength;
10⁷–10⁹ after collapse. The threshold does no delicate work — the two regimes
are separated by eight orders of magnitude — which is exactly what makes it a
safe number to fix in advance.

For the **controls** the comparison is between two independently trained
models, where run-to-run variation is genuinely nonzero. The null refit measures
that variation; its mean + 2 SD is reported beside the 1.10 threshold rather
than replacing it.

### Selective forgetting — RNFF ≥ 0.50 on forget, ≤ 0.20 on retain

This replaces the forget/retain loss ratio used in the exploratory phase, and
the reason is a failure of that instrument:

| | forget/retain ratio | forget Δ NLL | retain Δ NLL |
|---|---|---|---|
| 2e-5, viable | 1.049 | **+0.111** | +0.003 |
| 5e-5, destroyed | 1.005 | +16.9 | +16.9 |

The ratio gives nearly the same number for "barely forgot" and "destroyed
everything", because both losses sit near 2.0 in the first case and both are
huge in the second. It also hides the real result: GA+GD at 2e-5 is **highly
selective**, moving the forget set 37× more than the retain set. Its problem is
magnitude, not specificity.

The **Retraining-Normalised Forgetting Fraction** measures magnitude against the
only meaningful yardstick — what exact retraining does:

    D       = L_retrain(forget) − L_finetuned(forget)      (per seed)
    RNFF(P) = [ L_post(P) − L_finetuned(P) ] / D

token-weighted over population P, using that seed's own retraining control.
RNFF_forget = 1 means "this method moved the forget set exactly as far as
deleting it would have"; RNFF_retain is the specificity check, and global
destruction fails it by construction because destruction moves retain as far as
forget.

**The same denominator is used for both populations**, and that is the only
well-posed choice rather than a convenience. Normalising D_retain by *its own*
retraining change does not work: exact retraining trains on D_retain, so that
denominator is ≈0 by construction, the ratio is undefined in the ordinary case,
and a specificity clause built on it would never fire — including for a
destroyed model, which is precisely the case it exists to catch. (This was found
during implementation: the first version of the criterion had exactly that flaw,
and a synthetic destroyed model passed the specificity clause. Corrected before
any confirmatory run existed; the unit test
`test_destruction_fails_the_selectivity_clause_by_construction` pins it.)

With the shared denominator both numbers read on one scale — *fraction of the
intended deletion effect* — and both must be **measured**. An absent RNFF means
the denominator was unavailable, so specificity is unknown, and unknown is not
the same as satisfied.

**Why 0.50 and 0.20.** The member/non-member gap under θ_finetuned — the loss
rise a forget sample must undergo to stop reading as a member — is **+0.388 /
+0.453 / +0.343** nats across the three seeds. GA+GD at its best viable strength
achieves +0.111, i.e. **0.30 / 0.24 / 0.31** of it. So 0.50 is a bar the current
method demonstrably fails, set at "at least halfway to real deletion", and 0.20
requires the retain set to move less than a fifth as far as the forget set's
gold-standard move. Neither was chosen to be passable.

The **form** of this criterion is fixed now. Its denominator is a measurement
supplied by the exact-retraining control, which runs first, before any method it
scores.

### Deletion detectable — AUC ≥ 0.70, CI lower > 0.50, **and** above the null floor

The first two clauses carry over from the exploratory rule. The third is new and
is the one that matters: a detector scoring 0.65 on an exactly retrained model is
reading run-to-run training randomness, not deletion. Every detector must clear
its own false-positive floor, measured on the null refit, by two across-seed SDs.

## The exact-retraining displacement problem

This is the easiest thing in the study to get wrong, so it is stated explicitly.

For an unlearning method, ĝ = θ_post − θ_finetuned is a **small displacement**:
a short optimisation that started at θ_finetuned. Every directional detector was
designed around that fact.

For a control, θ_post is an **independent fine-tune**. The subtraction is still
exactly what the auditor holds — the pre-deletion model and the published model,
which is the regulator's situation — but the result is not small, and it
contains the deletion effect *plus* all the randomness of a separate training
run: batch order, dropout, optimiser state. A nonzero forget-vs-retain AUC on
the exact-retraining control is therefore **uninterpretable on its own**.

Hence two controls, differing in exactly one thing:

| Control | ĝ | What was deleted |
|---|---|---|
| **Exact retrain** | θ_retrain(retain only) − θ_finetuned(full) | the forget set |
| **Null refit** | θ_refit(full, new training RNG) − θ_finetuned(full) | nothing |

The null refit has the same two-independent-models structure with nothing
removed, so any signal it produces is a pure false positive. The **contrast**
between the two isolates deletion from training randomness. Without the null
refit the gold-standard control has no scale, and I would not be able to say
whether a retraining AUC of 0.6 meant "deletion leaves a trace" or "two training
runs differ".

Both controls reuse each seed's own split manifest, so forget, retain, curvature
and test populations are identical to every other run, and the curvature holdout
stays disjoint from the scored samples.

## Classifications

Mechanical, so the verdict is not a judgement made while looking at numbers.
Full definitions are in the YAML; in brief:

- **Persistent fingerprint** — all three axes pass *and* deletion is detectable.
- **Incomplete forgetting** — useful model, but forgotten records still read as members.
- **Global destruction** — membership gone, utility gone. Never reinterpreted as success.
- **Successful deletion without fingerprint** — all three axes pass, detectors at chance.
- **Inconclusive** — nothing satisfied the joint criteria. A statement about what was tested.

## Statistics

n = 3 seeds. Every seed is reported separately. Means, paired within-seed
differences and across-seed SDs lead; p-values are secondary and are the basis
of no claim. At three seeds distributional assumptions cannot be validated, and
Wilcoxon signed-rank cannot reach p < 0.05 below six pairs at all. Bootstrap
intervals are over samples within one run and describe that run's precision, not
between-model variation.

## Threat model

Every two-checkpoint detector — which includes the primary one — requires the
**pre-deletion checkpoint**. These are **white-box deletion-audit tools** for a
model owner, a regulator or an authorised verifier, and are not black-box
membership-inference attacks. A party holding only the released model cannot
mount them. Every headline result states the auditor's exact access, and the
detector table in the YAML records the access class of each score.

## Declared conflicts of interest

Two choices in this document were made after seeing exploratory results, and
recording them is what keeps this a preregistration rather than a description:

1. **`loss_ratio_vs_finetuned` is named primary detector** after it matched or
   beat `raw_cosine` at both viable exploratory settings while requiring
   strictly weaker access. Naming it primary is the conservative reading of the
   matched-access rule — it sets the bar at the cheapest adequate detector — but
   the choice was still made with the answer visible, and the exploratory runs
   cannot confirm it.
2. **`raw_cosine` was promoted to headline** in the exploratory phase after
   reaching 0.982 ± 0.006. It enters this study as a hypothesis under test.

The RNFF replacement was also made after seeing results, but on different
grounds: the old instrument was shown to be incapable of distinguishing two
opposite outcomes, and RNFF's thresholds come from a property of the data (the
member/non-member gap) rather than from any method's performance.
