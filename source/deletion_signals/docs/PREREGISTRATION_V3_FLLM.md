# Preregistration v3 — the FLLM audit

Machine-readable authority: [`configs/preregistration_v3_fllm.yaml`](../configs/preregistration_v3_fllm.yaml).
This document is its rationale. Where the two differ, the YAML is what the code
reads and what a verdict rests on.

**v2 is not modified, superseded or reinterpreted.** Every threshold it froze is
restated in v3 by value and unchanged. v3 only adds.

| carried over from v2 | value |
|---|---|
| membership equivalence region | [0.45, 0.55] |
| utility, max perplexity ratio | 1.10 |
| RNFF min forget / max retain | 0.50 / 0.20 |
| deletion detectable | AUC ≥ 0.70, CI lower > 0.50, above the null-refit floor by 2 SD |
| primary detector | `loss_ratio_vs_finetuned` |
| replication unit | seed (n = 3) |

## What this extension adds, and why

The revised central claim is that **membership removal, selective forgetting and
utility preservation are distinct properties**; that membership-only evaluation
can falsely certify a globally damaged model; that exact retraining and null
refitting calibrate what a real deletion looks like; and that under matched
access, two-checkpoint loss statistics may do as well as substantially more
expensive gradient detectors.

Four things were missing to support that claim.

**A — NPO.** GA+GD and SCRUB both failed the conjunction, but both are known to
be blunt. NPO exists specifically to avoid the catastrophic collapse GA+GD
exhibits, so "no method succeeded" is a much weaker statement without it. If NPO
also fails, the claim is about the difficulty of the problem rather than about
two poorly chosen methods.

**B — low prevalence.** Every number so far is an AUC at a deletion prevalence of
1 in 4. No audit looks like that. Precision at 1:100 is not a relabelling of AUC
— AUC is invariant to the base rate and precision is not — so a detector
certified on AUC alone can be unusable in the setting it is proposed for.

**C — matched access and cost.** A gradient detector reading two checkpoints is
not competing with a loss statistic reading two checkpoints unless both are
charged for what they need. Without an explicit access classification and a cost
measurement, "gradient alignment works" is not a comparable claim.

**D — routine update.** The null refit measures training randomness. It does not
measure what an *ordinary, legitimate* checkpoint change does. A deletion
detector that fires on every maintenance pass has no operational value, and
nothing in the study so far would have revealed that.

## Joint success criteria

A setting passes only if **all four** hold **jointly on an independently trained
seed**:

1. membership evidence inside the chance-equivalence region;
2. RNFF forget ≥ 0.50 **and** |RNFF retain| ≤ 0.20;
3. retain **and** test perplexity ratio ≤ 1.10;
4. the collapse guard did not fire.

Two counts are reported for every setting and neither is inferred from the
other: how many seeds passed each criterion **alone**, and how many passed all
four **together**. Three seeds each passing a different criterion is not a pass.

A criterion that could not be **measured** is not satisfied. That is what stops a
destroyed model slipping through on a missing RNFF denominator — destruction and
success both leave the criterion unevaluated.

## The six classifications

A total function of the four criteria plus the stopping reason, evaluated in a
fixed precedence so the first match wins. Implemented in
[`fllm_report.classify_run`](../fllm_report.py); the partition is enumerated
exhaustively in `test_fllm_report.py`.

| label | fires when |
|---|---|
| `failed_run` | collapse guard tripped, run did not complete, or losses non-finite |
| `successful_selective_deletion` | membership ∧ selective ∧ utility |
| `deletion_without_preserved_utility` | selective ∧ ¬utility |
| `global_destruction` | ¬utility ∧ ¬selective |
| `incomplete_forgetting` | utility ∧ ¬membership |
| `inconclusive` | anything else, including any unmeasurable criterion |

**The label names a pattern of criteria, not a magnitude.** `global_destruction`
fires whenever utility is out of tolerance *and* the collateral movement exceeds
the specificity budget — which covers a model whose perplexity rose 18% as well
as one destroyed outright. Magnitude is read off the utility and RNFF columns,
never off the label.

A setting takes a classification only when its seeds agree; otherwise it is
reported as `inconclusive` with every seed's own label shown.

## Component A — NPO

Formulation, sign convention, β, the reference model and every departure are
documented separately in [NPO.md](NPO.md). In brief: **NPO-RT**, Eq. 3 of
arXiv:2404.05868, β = 0.1 fixed, λ = 1.0, sequence-**sum** log-ratio, reference =
each seed's own `θ_finetuned`, 100 steps, batch 4, clip 1.0.

**The grid**, swept on the learning rate:

| learning rate | intended regime |
|---|---|
| 1.0e-5 | weak |
| 2.0e-5 | moderate |
| 2.8e-5 | expected viable region |
| 5.0e-5 | aggressive |
| 1.0e-4 | near-collapse / over-aggressive |

Justified by a **gradient identity**, not by an NPO outcome: at initialisation
`θ = θ_ref`, so the NPO weight is `2σ(0) = 1` and NPO's first step is GA's first
step — identical after clipping, which `test_npo.py` asserts. The grid is
therefore anchored on GA+GD's *measured* transition and extended upward, because
NPO's weight decays monotonically and is weaker per step at equal learning rate.

Every cell records seed, configuration hash, steps completed, stopping reason,
collapse status, losses and perplexities under all three models, membership
AUCs, RNFF, utility change, every detector, runtime, peak memory, null and
non-finite counts, truncation and failed-sample counts. **Collapsed and failed
runs appear in every table.**

## Component B — low-prevalence metrics

Computed offline from saved score files; no model is re-run.

Prevalence adjustment is **analytic**:

```
precision(π) = π·TPR / ( π·TPR + (1−π)·FPR )
recall(π)    = TPR
```

Deterministic, reproducible, and it does not discard measured negatives to
simulate a base rate. A subsampling cross-check is reported beside it as a
diagnostic and is asserted to agree in the unit tests.

Two resolution rules are enforced rather than left to the reader:

- **An FPR target below `1/n_negative` is not estimable** and is reported as
  null with its reason. With 600 scored retain samples the floor is 1/600 =
  0.00167, so **TPR@1%FPR is estimable and TPR@0.1%FPR is not**. Declared in
  advance, so its absence is an outcome rather than a gap.
- **An observed FPR of exactly zero is floored at `1/n_negative`.** Otherwise
  precision reads 1.0 at every prevalence on the strength of having seen no
  false positive in 600 draws.

Seed is the replication unit; per-seed values, mean and SD are all reported.
Sample-level bootstrap intervals are labelled as within-run precision and are
never presented as replication.

## Component C — matched access and cost

Every detector is classified on twelve access dimensions. A challenger may be
claimed advantageous over a baseline **only if the baseline's access is a subset
of the challenger's**.

One modelling decision is worth stating because it decides which comparisons run
at all: a **gradient detector also possesses the per-example loss**, since a
gradient cannot be computed without the loss it differentiates. Access is what
the holder possesses, not what the formula consumes. Under this rule
`raw_cosine`, `alignment_cos_sq`, `grad_ratio` and `combined_log_score` subsume
the two-checkpoint loss baseline and are comparable to it; `grad_norm_sq`, `sif`
and `fuma_lora_ratio` never read the pre-deletion checkpoint and are **not**, so
no advantage over the baseline may be claimed for them.

Costs are **measured as primitives and composed**, because in the real collector
four detectors come out of one backward pass. Both numbers are reported:
`standalone` (what it costs alone) and `marginal` (what it adds to a run already
computing the shared pass). Reporting only one would misstate the comparison in
whichever direction the reporter preferred.

## Component D — routine update

Four conditions, two of them new:

| # | condition | base | deletion |
|---|---|---|---|
| 1 | null refit | pretrained base, full member set, new RNG | none |
| 2 | **routine update** | `θ_finetuned` | none |
| 3 | exact retraining | pretrained base, retain only | exact |
| 4 | **exact deletion + routine update** | `θ_exact_retrain` | exact |

Conditions 2 and 4 receive the **identical** update — same data, objective,
learning rate, step count and RNG stream — so the only difference between them is
whether a deletion happened underneath it.

**The update**: continued LoRA fine-tuning with the ordinary causal-LM
objective, lr 5.0e-5, 3 epochs (~18 optimizer steps), on 200 records. Frozen; it
may not be tuned to make a detector succeed or fail.

**The data, and the confound this dataset forces.** ArxivMIA has exactly 1000
members and 1000 non-members, and the split manifests already consume **all 1000
members**. There are no unused member records. The only records that are both
fresh and disjoint from every scored population are the non-members beyond the
scoring cap — `test[300:500]`, which no configuration in this study scores.

Those records are **non-members**, so the update nudges the model toward the
non-member distribution and can move the membership axis for a reason unrelated
to deletion. This cannot be resolved within this dataset. It is handled by
reporting the membership axis for both update conditions beside the same axis
without the update, so the shift is a visible measurement. The alternative —
continuing on retain data — was rejected because retain is the negative class of
the primary task.

Disjointness is enforced by text fingerprint at run time, not assumed:
`routine_update.py` refuses to train if the update overlaps forget, retain
(scored tail or curvature head), or the scored test prefix.

## Setting selection, fixed in advance

- **GA+GD best viable** — largest RNFF forget among settings also passing utility
  on the most seeds; ties to smaller RNFF retain. Resolves to `ascent_lr=2.6e-5`.
- **GA+GD destructive** — largest retain perplexity ratio among completed
  confirmatory settings. Resolves to `ascent_lr=5.0e-5`.
- **SCRUB best** — most marginal criteria passed; ties to largest RNFF forget.
  Resolved mechanically by the report generator.

Both resolutions were made before any Component B or C metric was computed. The
rules are stated so a reader can check that each names the setting the rule
selects rather than the one producing a convenient number.

## Re-scoring, and what is immutable

`ds_results/`, `ds_sweep/`, `ds_confirm/`, `ds_confirm_refine/`,
`configs/preregistration_v2.yaml` and `configs/orientations.yaml` are **not
modified**.

The confirmatory GA+GD runs were scored with FUMA and Fisher-SIF disabled, so
they carry 10 of 13 score columns and 800 rather than 600 scored retain samples.
Two settings × three seeds are **re-scored** — scoring only, from the existing
checkpoints, no retraining — into `fllm_audit/rescore/`, so every row of
the central detector table carries all 13 columns at a uniform 600 negatives.

## Claim discipline

Not claimed, anywhere:

- that membership AUC near chance proves forgetting;
- that a collapsed model has unlearned;
- that gradient direction is uniquely informative unless it beats matched-access
  loss;
- that deletion-history detection is automatically a privacy attack;
- that anything generalises beyond ArxivMIA, Llama-3.2-1B, LoRA and the methods
  tested;
- that n = 3 gives population-level statistical certainty.

**Threat model.** Three bands, stated on every headline result: *authorised
audit* (owner, regulator, verifier — holds both checkpoints), *insider or
checkpoint compromise* (same access, obtained illegitimately), and *ordinary
black-box user* (released model's outputs only, which supports `loss_unlearned`
alone and only in degraded form). A two-checkpoint result is framed as an
**audit capability** unless a realistic adversary can obtain both checkpoints,
and that precondition is stated in the same sentence.

## Known risks, declared before the runs

1. **AdamW partially defeats NPO's mechanism.** The collapse-avoidance weight is
   a gradient *magnitude* gate; Adam's second-moment normalisation undoes part of
   it. The official implementation also uses AdamW, so this is faithful — but an
   NPO collapse here would not refute the paper's mechanism, only its behaviour
   under this optimiser.
2. **NPO may reproduce GA+GD's null.** GA+GD's membership AUC sat at 0.013–0.072
   against a [0.45, 0.55] region while utility was already breaking. A null
   result across the whole NPO grid is a plausible and publishable outcome, and
   nothing will be tuned to avoid it.
3. **The routine-update confound** described above.
4. **TPR@0.1%FPR is not estimable** at this sample size and will be absent by
   preregistration.
5. **Benchmark contention.** The GPU is shared. Concurrent processes are recorded
   alongside every timing; this documents contention rather than correcting it.
6. **n = 3, one dataset, one model, one adapter configuration.**
