# Preregistered score orientations

The machine-readable table is
[`../configs/orientations.yaml`](../configs/orientations.yaml); it is the single
source of truth and each entry carries its own rationale. This document states
the policy and summarises the directions.

## The policy

1. Every score's direction on every task is written down **before** any metric
   is computed, together with the reasoning that produced it.
2. All reported AUCs, average precisions and TPR-at-FPR values use those
   directions.
3. **`max(AUC, 1 − AUC)` is never used.** A score that lands below 0.5 under its
   declared direction is reported below 0.5.
4. A score with no declared orientation for a task cannot be evaluated on it —
   `OrientationTable.orientation` raises rather than defaulting. Adding a score
   forces a written-down direction first.
5. A declared direction with no rationale is rejected at load time. A direction
   without a stated reason is not a preregistration.

Why (3) matters: post-hoc flipping scores every method on its best-case reading.
An anti-correlated score becomes indistinguishable from a correlated one, and
the ranking *between* scores — which is the entire output of this study — stops
carrying information. The sibling pipeline's comments endorse `max(auc, 1-auc)`
for legacy component comparison; that path is untouched over there and is not
used here.

## Tasks

| task | positive | negative | role |
|---|---|---|---|
| `forget_vs_retain` | forget | retain | **primary** — deletion-set inference |
| `member_vs_unseen` | forget + retain | test | secondary — ordinary membership |
| `forget_vs_unseen` | forget | test | diagnostic |
| `forget_vs_rest` | forget | retain + test | diagnostic, one-vs-rest |
| `retain_vs_unseen` | retain | test | diagnostic, the three-way third axis |

`forget_vs_unseen` is diagnostic only: a high value there is consistent with
either a deletion signal or a plain membership signal, so it cannot separate the
two hypotheses by itself.

## Directions

`↑` = higher is positive, `↓` = lower is positive.

| score | fvr | mvu | fvu | fvrest | rvu | headline |
|---|:--:|:--:|:--:|:--:|:--:|:--:|
| `raw_cosine` | ↑ | ↑ | ↑ | ↑ | ↑ | yes |
| `alignment_cos_sq` | ↑ | ↑ | ↑ | ↑ | ↑ | yes |
| `loss_unlearned` | ↑ | ↓ | ↑ | ↑ | ↓ | yes |
| `loss_ratio` | ↑ | ↓ | ↑ | ↑ | ↓ | yes |
| `loss_delta` | ↑ | ↓ | ↑ | ↑ | ↓ | yes |
| `loss_ratio_vs_finetuned` | ↑ | ↓ | ↑ | ↑ | ↓ | yes |
| `loss_delta_vs_finetuned` | ↑ | ↓ | ↑ | ↑ | ↓ | yes |
| `grad_norm_sq` | ↑ | ↓ | ↑ | ↑ | ↓ | yes |
| `grad_ratio` | ↑ | ↓ | ↑ | ↑ | ↓ | yes |
| `log_grad_ratio` | ↑ | ↓ | ↑ | ↑ | ↓ | yes |
| `fuma_lora_ratio` | ↑ | ↑ | ↑ | ↑ | ↑ | yes |
| `sif` | ↑ | ↓ | ↑ | ↑ | ↓ | yes |
| `combined_log_score` | ↑ | ↑ | ↑ | ↑ | ↑ | yes |

Columns: fvr = `forget_vs_retain`, mvu = `member_vs_unseen`,
fvu = `forget_vs_unseen`, fvrest = `forget_vs_rest`, rvu = `retain_vs_unseen`.

## The entries that need explaining

**`loss_*` flips between the two primary tasks.** Membership lowers loss;
unlearning raises it. `↓` on `member_vs_unseen` and `↑` on `forget_vs_retain`
are not an inconsistency — they are the reason one threshold cannot serve both
questions.

**`grad_norm_sq` on `forget_vs_retain` is method-dependent, and the assumption
is declared rather than chosen afterwards.** Under GA+GD the ascent on
`D_forget` is halted by a loss-ratio criterion, not by convergence, so forget
samples end in a raised-loss region with large residual gradients while retain
samples sit near a descent optimum — hence `↑`. Under SCRUB the forget objective
is `−KL − entropy` rather than `+NLL`, so the NLL gradient magnitude is not
directly driven up; the same direction is assumed but expected to be weaker. If
SCRUB lands below 0.5, it is reported below 0.5.

**`raw_cosine` is the preregistered primary score, and it was promoted after it
was seen to win.** Both halves of that sentence matter.

Its direction does depend on the ĝ sign convention (θ_unlearned − θ_finetuned
rather than its negation), which is why it was originally `headline: false`. The
convention is now stated exactly and the direction it implies is argued from the
mechanism rather than assumed: under GA+GD the forget set receives gradient
*ascent*, so its contribution to ĝ is −∇L and its gradients point along ĝ; the
retain set receives *descent*, so its gradients oppose ĝ. The two classes sit on
opposite sides of zero, which is precisely the information `cos²` throws away by
squaring. The argument is specific to gradient-ascent unlearning; under SCRUB
the forget objective is −KL − entropy and no such argument is available.

The promotion happened *after* the score reached 0.982 ± 0.006 on the primary
task in the first three seeds. That is recorded, with its date, in the change
log at the top of [`../configs/orientations.yaml`](../configs/orientations.yaml).
The consequence is that those three seeds cannot serve as the score's
confirmation — the unlearning-strength sweep, which was preregistered
afterwards, is what tests it on data that did not select it. See
[PREREGISTRATION.md](PREREGISTRATION.md).

One direction is declared for all five tasks, so no task-by-task choice was
available after the fact. On `retain_vs_unseen` the mechanism *predicts* an AUC
below 0.5 — retain gradients oppose ĝ while unseen gradients sit near zero — and
that is reported below 0.5.

**The two loss families answer different questions.** `loss_ratio` divides by
the pretrained base's loss and measures **membership**: was this sample fitted.
`loss_ratio_vs_finetuned` divides by the pre-unlearning model's loss and
measures **what unlearning did**: ascent pushed the forget set up, descent
pulled the retain set down, and unseen data barely moved. The sweep needs both
at once, because its question is whether a deletion signal survives after the
membership signal is gone.

**`sif` is declared `↓` for members.** Members were fitted, so their
self-influence is lower. Under this direction `retain_vs_unseen` — read with
retain as the positive class — is *predicted* to land below 0.5. That is the
prediction holding, not signal absent, and it is declared here precisely so it
cannot be reinterpreted afterwards.

**`combined_log_score` on `member_vs_unseen` and `retain_vs_unseen` partially
cancels.** Its two factors pull in opposite directions there: `cos²` is higher
for members, `sif` is lower. The declared direction follows `cos²`, which spans
several orders of magnitude and so dominates the log-space ranking. Near-chance
is the expected outcome and is reported as such.

**`grad_ratio` on `member_vs_unseen` is expected near chance.** With the
fine-tuned model as reference, the membership effect largely cancels — the
reference fitted the members too.

## Adding a score

1. Add the column to `schema.SCORE_COLUMNS` and to `ScoreRow`.
2. Add an entry to `configs/orientations.yaml` with a direction **and a
   rationale** for every task. The loader rejects a partial entry.
3. `tests/test_orientations.py::test_every_score_column_has_a_declared_orientation_on_every_task`
   will fail until step 2 is complete.

## Amending the table

A preregistration is only honest if its amendments are visible. Any change to a
direction, or to a score's headline status, goes in the **change log at the top
of `configs/orientations.yaml`** with its date and, where it applies, an
explicit statement that the change was made after seeing results. Silently
editing a direction and re-running the evaluation is the failure mode this whole
file exists to prevent.
