# Design notes

## Training membership is not deletion-set membership

Two questions get conflated in this literature, and the baselines that are
meaningful for one are not meaningful for the other.

**Training membership.** *Was z in the training data at all?* The negative class
is data the model never saw. Loss thresholds work because fitting a sample
lowers its loss; every calibrated-loss and gradient-magnitude attack in the MIA
canon is an answer to this question.

**Deletion-set membership.** *Given that z was a training member, was it in the
set targeted for unlearning?* Both classes are members. Both were fitted. The
negative class is `D_retain` — data that was trained on and deliberately kept.

The second question is the one this package evaluates, as
`forget_vs_retain`. It is strictly harder, and a score that excels at the first
can be at chance on the second. That is not a defect in the score; the two tasks
have different sufficient statistics.

The consequence for orientation is concrete. Under `member_vs_unseen`, low loss
means member. Under `forget_vs_retain`, *high* loss means forget, because
raising loss on the forget set is what unlearning does. A single loss threshold
cannot serve both, and the preregistered table declares opposite directions for
`loss_unlearned` on the two tasks
([ORIENTATIONS.md](ORIENTATIONS.md)).

## Why calibrated loss is a baseline, not a replacement

`loss_ratio(z) = L_unlearned(z) / max(L_base(z), eps)` divides out how
intrinsically hard the sample is. That removes a genuine nuisance variable: a
rare, long, or unusual sequence has high loss under every model, and an
uncalibrated threshold spends most of its discriminative power on that rather
than on membership.

It is a **difficulty-calibrated membership baseline**. It has no access to the
unlearning direction. It can only see the forget set through whatever the
intervention did to the forget set's *loss* — which for a KL-based method like
SCRUB may be very little, even when the parameter displacement is large. Its
role in the comparison is to answer "is the alignment score doing anything a
much cheaper forward-pass-only statistic could not?".

It is deliberately **not called LiRA**. LiRA fits per-sample Gaussians over
many shadow models and performs a likelihood-ratio test. This is one reference
model and a ratio. Labelling it LiRA would claim a calibration strength the
implementation does not have.

## Two loss references, not one setting with two values

`loss_calibrated.reference` picks the model that divides out *difficulty*.
`loss_vs_finetuned` is a separate family with a separate reference, and the
distinction is not bookkeeping:

| pair | reference | what it measures |
|---|---|---|
| `loss_ratio`, `loss_delta` | pretrained base | **membership** — was this sample fitted at all |
| `loss_ratio_vs_finetuned`, `loss_delta_vs_finetuned` | fine-tuned pre-unlearning | **what unlearning did** — ascent raised the forget set, descent lowered the retain set, unseen data barely moved |

Collapsing them into one configurable reference would force a run to choose, and
the question that motivates the unlearning-strength sweep needs both at once:
*after* the membership signal has gone to chance, is there still a deletion
signal? `loss_finetuned` costs one extra forward pass on a model that is already
resident for ĝ, and none at all when the calibrated loss already uses it.

Both raw measurements — `loss_base` and `loss_finetuned` — stay out of
`SCORE_COLUMNS`. They are inputs to scores, not rankings.

## The access assumption for ĝ

The unlearning direction is

    ĝ = θ_unlearned − θ_finetuned

which requires **white-box access to both checkpoints**: the fine-tuned model as
it stood before unlearning, and the unlearned model. That is a strong
assumption. It is the right one for the question being asked — whether the
directional signature exists at all and how much it buys over cheaper signals —
but it is not an attack a party holding only the released model could mount.

Note the practical consequence, visible in this repository: the sibling
pipeline's `results_seed4*` directories retain their `unlearned_model/` adapters
but their fine-tuned checkpoints were deleted, so ĝ cannot be recomputed for
those runs. Any pipeline that intends to support this analysis must keep
θ_finetuned.

`cos²` is sign-invariant, so the choice of ĝ over −ĝ affects only `raw_cosine`.
`raw_cosine` was originally excluded from the headline comparison for exactly
that reason. It is now the preregistered primary score for the
unlearning-strength sweep, because the sign is not incidental under GA+GD: the
forget set receives ascent and the retain set descent, so their gradients sit on
opposite sides of ĝ rather than merely at different magnitudes, and squaring
discards precisely that. The convention is fixed in writing, one direction is
declared for every task, and the promotion — which happened after the score was
seen to win on three seeds — is recorded in the change log at the top of
`configs/orientations.yaml`. See [PREREGISTRATION.md](PREREGISTRATION.md).

Framing follows from the access assumption: this is a **white-box deletion
audit** a model owner, regulator or authorised verifier could run, not a
black-box membership-inference attack.

## Reference-model choice, per score family

| family | reference | why |
|---|---|---|
| calibrated loss | **pretrained base** | Forward passes only, so a 4-bit backbone is fine. The base model is the one that never saw any of this data, which is what "intrinsic difficulty" means. |
| calibrated gradient, LoRA mode | **fine-tuned pre-unlearning** | Gradients live on adapter parameters. The pretrained base has no adapters and a 4-bit backbone, so ‖∇L_base‖² over the same parameter space does not exist. The fine-tuned model shares the space exactly and isolates the unlearning effect. |
| calibrated gradient, full fine-tuning | **either** | The parameter spaces match, so `pretrained_base` is also computable. It answers a different question — it includes the effect of fine-tuning, where the fine-tuned reference does not. |

`config.validate_config` rejects `grad_calibrated.reference: pretrained_base` in
LoRA mode rather than letting it fail hours into a run, and every row records
which reference it actually used in `grad_reference_model`.

## One backward pass per model

Per sample:

| model | passes | yields |
|---|---|---|
| unlearned | 1 backward | `loss_unlearned`, `grad_norm_sq`, `raw_cosine`, `alignment_cos_sq`, `sif` |
| pretrained base | 1 forward | `loss_base` |
| fine-tuned pre-unlearning | 1 backward | `grad_norm_sq_ref` |
| fine-tuned pre-unlearning | 1 forward | `loss_finetuned` (only when `loss_vs_finetuned` is on) |
| bf16 shadow (LoRA mode, FUMA on) | 1 backward | both halves of the FUMA ratio |

Every unlearned-model gradient statistic is a reduction over the *same* backward
pass; nothing is recomputed. When the FUMA denominator comes from the scored
model itself (`denominator_source: same_model`, which is the case on an
unquantised CPU backbone), even that pass is reused —
`tests/test_collect_tiny.py::test_fuma_reuses_the_single_backward_on_the_scored_model`
pins that.

## Numerics

Alignment is a ratio of sums over millions of terms. Gradients arrive in
bfloat16 or float32 and accumulating a 3.4M-term dot product at that precision
loses low-order bits that move the cosine — which is what the headline AUC ranks
by. `gradients.accumulate_gradient_stats` casts each parameter tensor to float64
and accumulates per tensor, so the running totals stay exact to ~15 digits while
peak extra memory is one tensor rather than one model.

`concatenated_gradient_stats` is the obvious-but-costly formulation, kept solely
so the layerwise path can be pinned against it
(`tests/test_gradients.py::test_layerwise_accumulation_matches_concatenated_vector`,
and again on real model gradients in the smoke test).

Explicit handling, all covered by tests:

- **zero gradient or zero direction** → `raw_cosine = 0.0`, `zero_grad_norm` /
  `zero_direction_norm` flagged. Zero is the neutral point of the cos² ranking,
  and the flag distinguishes "no alignment" from "no gradient".
- **missing `.grad`** → contributes zeros, counted in `missing_grad_params`,
  and its element count still enters the FUMA denominator via `numel_hint` so a
  frozen tensor cannot inflate the ratio by disappearing from it.
- **NaN / Inf** → counted in `n_nonfinite_tensors` and allowed to propagate, so
  the row's scores come out non-finite and the evaluator drops them with a
  count rather than silently rounding them into something plausible.
- **eps** → `max(denominator, eps)` for the ratio forms, additive `+ eps` inside
  the logs. The two differ exactly in the underflow regime: the quotient form
  returns 1.0 when both norms vanish, implying "no change" where the truth is
  "no signal", and the log form returns 0.0 while staying finite.
- **truncation** → flagged per row when the tokenised length hits `max_length`.

## Two units of analysis

Within a run, the bootstrap resamples **samples** and answers "how precisely is
this AUC estimated, given this trained model". It cannot support a claim about
the method, because every sample shares one model.

Across runs, `stats.across_seed_comparison` treats each **seed** — an
independently fine-tuned and unlearned model — as one observation. That is the
only output in this package from which "score A beats score B for this method"
follows.

At three seeds the honest reporting is the mean difference and its across-seed
SD. Wilcoxon signed-rank cannot reach p < 0.05 below six pairs (its smallest
attainable two-sided p at n = 3 is 0.25), so below six the code uses a paired
t-test and states in the output that its normality assumption is unverifiable at
that n. A difference that is constant across seeds gets no test at all: there is
no between-seed variation to test against, and a paired test fed that data
returns p ≈ 0 out of nothing.

## Suppressed operating points

TPR at 0.1% FPR is requested by default and almost always suppressed. With `n`
negatives the finest non-zero FPR an empirical ROC can express is `1/n`; at a
few hundred negatives, "TPR at 0.1% FPR" collapses to "TPR at zero false
positives", a statistic set by the single highest-scoring negative. Those
entries are reported as `resolvable: false` with `tpr: null` and the reason
attached, rather than as a number that would read as a measurement.
