# Three-axis unlearning audit — ArxivMIA

Generated 2026-08-08T23:07:02.553696+00:00 from 27 score file(s).
Preregistration: `evidence/deletion_signals/configs/preregistration_v2.yaml` (schema v2).

Thresholds below were fixed before any confirmatory run and are read from
that file, not from constants in code.

| axis | criterion |
|---|---|
| membership removed | `loss_ratio` AUC on `forget_vs_unseen` within [0.45, 0.55] |
| utility preserved | retain and test perplexity ratio ≤ 1.1 |
| selective forgetting | RNFF_forget ≥ 0.5 and \|RNFF_retain\| ≤ 0.2 |
| deletion detectable | `loss_ratio_vs_finetuned` AUC ≥ 0.7, CI lower > 0.5, above the null-refit floor |

A setting is a candidate operating point only if the first three hold on
**every** seed. Membership AUC alone never certifies anything.

## Verdict

**remember-or-collapse: every setting either left forgotten records looking like training members, or removed the membership evidence only by destroying the model. No setting did both. Membership evidence disappeared, but retained or test utility exceeded the damage threshold. Membership-only evaluation would have falsely certified this as successful unlearning.**

No setting satisfied the three joint criteria on every seed.

| method | setting | membership | utility | selective | detectable | candidate |
|---|---|---|---|---|---|---|
| exact_retrain | exact_retrain | 3/3 | 3/3 | 3/3 | 3/3 | yes |
| ga_gd | ascent_lr=2.5e-05 | 0/3 | 3/3 | 0/3 | 3/3 | no |
| ga_gd | ascent_lr=2e-05 | 0/3 | 3/3 | 0/3 | 3/3 | no |
| ga_gd | ascent_lr=3.5e-05 | 2/3 | 0/3 | 0/3 | 0/3 | no |
| ga_gd | ascent_lr=3e-05 | 0/3 | 0/3 | 0/3 | 3/3 | no |
| ga_gd | ascent_lr=4.5e-05 | 3/3 | 0/3 | 0/3 | 0/3 | no |
| ga_gd | ascent_lr=4e-05 | 3/3 | 0/3 | 0/3 | 0/3 | no |
| ga_gd | ascent_lr=5e-05 | 3/3 | 0/3 | 0/3 | 0/3 | no |
| null_refit | null_refit | 0/3 | 3/3 | 0/3 | 0/3 | no |

## Controls

`exact_retrain` is the gold standard — the model you would have had if
the forget set had never been in the training data. `null_refit` deletes
**nothing** and differs from the pre-deletion model only in training
randomness, so whatever a detector scores there is a false positive.

Read them together. For both, ĝ = θ_post − θ_finetuned is a difference
between two *independently trained* models, not the small displacement an
unlearning method produces, so a retraining signal is uninterpretable
without the null floor beside it.

| control | seeds | retain ppl ratio | test ppl ratio | RNFF forget | membership AUC |
|---|---|---|---|---|---|
| exact_retrain | 3 | 1.037 ± 0.005 | 0.989 ± 0.007 | 1.000 ± 0.000 | 0.474 ± 0.003 |
| null_refit | 3 | 1.002 ± 0.007 | 1.000 ± 0.003 | 0.001 ± 0.014 | 0.001 ± 0.000 |

### Is the RNFF denominator measuring deletion?

RNFF divides by the loss change exact retraining produced on the forget
set, *attributing that change to deletion*. The null refit deleted
nothing, so its RNFF_forget is how much of that change two independently
trained models would have produced anyway.

| null-refit RNFF_forget | 0.001 ± 0.014 |
|---|---|
| null-refit RNFF_retain | 0.005 ± 0.016 |

> 
> the null refit, which deleted nothing, shows RNFF_forget near zero, so the denominator is attributable to deletion rather than to training variance.

### False-positive floors on `forget_vs_retain` (null refit)

Nothing was deleted in these runs. A detector scoring above chance here
is reading run-to-run training variation.

| detector | AUC on the null refit |
|---|---|
| `alignment_cos_sq` | 0.523 ± 0.026 |
| `combined_log_score` | 0.516 ± 0.016 |
| `loss_delta` | 0.511 ± 0.017 |
| `grad_ratio` | 0.510 ± 0.021 |
| `log_grad_ratio` | 0.510 ± 0.021 |
| `fuma_lora_ratio` | 0.505 ± 0.020 |
| `loss_ratio` | 0.502 ± 0.014 |
| `loss_unlearned` | 0.498 ± 0.018 |
| `loss_delta_vs_finetuned` | 0.497 ± 0.023 |
| `loss_ratio_vs_finetuned` | 0.495 ± 0.025 |
| `sif` | 0.484 ± 0.024 |
| `grad_norm_sq` | 0.484 ± 0.023 |
| `raw_cosine` | 0.477 ± 0.026 |

## Axis 1 — membership removal (forget vs test)

Under the preregistered higher-is-forget direction, **0.5 is chance and
0.0 is as much evidence as 1.0** — a forgotten record reading far below 0.5
still looks exactly like a training member. Hence the two-sided region.

| method | setting | forget_vs_unseen | member_vs_unseen | retain_vs_unseen |
|---|---|---|---|---|
| exact_retrain | exact_retrain | 0.474 ± 0.003 | 0.881 ± 0.000 | 0.999 ± 0.000 |
| ga_gd | ascent_lr=2.5e-05 | 0.010 ± 0.001 | 0.998 ± 0.000 | 0.999 ± 0.000 |
| ga_gd | ascent_lr=2e-05 | 0.004 ± 0.000 | 0.999 ± 0.000 | 1.000 ± 7.65e-05 |
| ga_gd | ascent_lr=3.5e-05 | 0.480 ± 0.033 | 0.601 ± 0.047 | 0.621 ± 0.050 |
| ga_gd | ascent_lr=3e-05 | 0.136 ± 0.053 | 0.957 ± 0.019 | 0.980 ± 0.010 |
| ga_gd | ascent_lr=4.5e-05 | 0.479 ± 0.001 | 0.526 ± 0.014 | 0.527 ± 0.018 |
| ga_gd | ascent_lr=4e-05 | 0.489 ± 0.018 | 0.530 ± 0.004 | 0.535 ± 0.005 |
| ga_gd | ascent_lr=5e-05 | 0.479 ± 0.002 | 0.527 ± 0.009 | 0.529 ± 0.012 |
| null_refit | null_refit | 0.001 ± 0.000 | 0.999 ± 0.001 | 0.999 ± 0.001 |

## Axis 2 — utility, and Axis 3 — selective forgetting

RNFF is the fraction of the *exact-retraining* loss change achieved:
1.0 means the forget set moved exactly as far as deleting it would have.
Global destruction fails the retain clause by construction, because it
moves retain as far as it moves forget.

| method | setting | retain ppl | test ppl | RNFF forget | RNFF retain | stop reason(s) |
|---|---|---|---|---|---|---|
| exact_retrain | exact_retrain | 1.037 ± 0.005 | 0.989 ± 0.007 | 1.000 ± 0.000 | 0.087 ± 0.013 | -- |
| ga_gd | ascent_lr=2.5e-05 | 1.058 ± 0.017 | 1.049 ± 0.008 | 0.450 ± 0.032 | 0.134 ± 0.037 | completed |
| ga_gd | ascent_lr=2e-05 | 0.997 ± 0.005 | 1.035 ± 0.008 | 0.262 ± 0.009 | -0.008 ± 0.013 | completed |
| ga_gd | ascent_lr=3.5e-05 | 12.740 ± 14.473 | 9.640 ± 11.181 | 5.405 ± 2.851 | 5.034 ± 2.714 | completed |
| ga_gd | ascent_lr=3e-05 | 1.399 ± 0.043 | 1.251 ± 0.022 | 1.147 ± 0.099 | 0.797 ± 0.077 | completed |
| ga_gd | ascent_lr=4.5e-05 | 2.65e+07 ± 4.56e+07 | 1.75e+07 ± 3.02e+07 | 33.231 ± 9.402 | 33.068 ± 9.334 | collapse_guard, completed |
| ga_gd | ascent_lr=4e-05 | 14667.708 ± 24349.976 | 10334.846 ± 17248.445 | 17.680 ± 7.777 | 17.504 ± 7.843 | completed |
| ga_gd | ascent_lr=5e-05 | 4.93e+07 ± 1.53e+07 | 3.00e+07 ± 1.19e+07 | 42.280 ± 1.285 | 42.001 ± 1.393 | collapse_guard |
| null_refit | null_refit | 1.002 ± 0.007 | 1.000 ± 0.003 | 0.001 ± 0.014 | 0.005 ± 0.016 | -- |

## Matched-access detector comparison

Primary detector: **`loss_ratio_vs_finetuned`** — access `unlearned, finetuned`.

A detector may be claimed as advantageous only over detectors whose access
it already subsumes. Rows where that does not hold are marked, and no claim
may rest on them.

| method | setting | vs detector | its access | weaker/equal | mean diff | SD | n |
|---|---|---|---|---|---|---|---|
| exact_retrain | exact_retrain | `alignment_cos_sq` | unlearned, finetuned, gradients | no | 0.001 | 0.001 | 3 |
| exact_retrain | exact_retrain | `combined_log_score` | unlearned, finetuned, gradients, curvature | no | -0.001 | 0.000 | 3 |
| exact_retrain | exact_retrain | `fuma_lora_ratio` | unlearned, gradients, lora_structure | no | 0.399 | 0.038 | 3 |
| exact_retrain | exact_retrain | `grad_norm_sq` | unlearned, gradients | no | 0.276 | 0.011 | 3 |
| exact_retrain | exact_retrain | `grad_ratio` | unlearned, finetuned, gradients | no | 0.010 | 0.005 | 3 |
| exact_retrain | exact_retrain | `log_grad_ratio` | unlearned, finetuned, gradients | no | 0.010 | 0.005 | 3 |
| exact_retrain | exact_retrain | `loss_delta` | unlearned, base | no | -0.000 | 0.000 | 3 |
| exact_retrain | exact_retrain | `loss_delta_vs_finetuned` | unlearned, finetuned | yes | -0.001 | 0.000 | 3 |
| exact_retrain | exact_retrain | `loss_ratio` | unlearned, base | no | 0.000 | 0.001 | 3 |
| exact_retrain | exact_retrain | `loss_unlearned` | unlearned | yes | 0.251 | 0.016 | 3 |
| exact_retrain | exact_retrain | `raw_cosine` | unlearned, finetuned, gradients | no | 0.001 | 0.001 | 3 |
| exact_retrain | exact_retrain | `sif` | unlearned, gradients, curvature | no | 0.276 | 0.011 | 3 |
| ga_gd | ascent_lr=2.5e-05 | `alignment_cos_sq` | unlearned, finetuned, gradients | no | 0.030 | 0.022 | 3 |
| ga_gd | ascent_lr=2.5e-05 | `grad_norm_sq` | unlearned, gradients | no | 0.388 | 0.016 | 3 |
| ga_gd | ascent_lr=2.5e-05 | `grad_ratio` | unlearned, finetuned, gradients | no | 0.175 | 0.011 | 3 |
| ga_gd | ascent_lr=2.5e-05 | `log_grad_ratio` | unlearned, finetuned, gradients | no | 0.175 | 0.011 | 3 |
| ga_gd | ascent_lr=2.5e-05 | `loss_delta` | unlearned, base | no | 0.108 | 0.022 | 3 |
| ga_gd | ascent_lr=2.5e-05 | `loss_delta_vs_finetuned` | unlearned, finetuned | yes | -0.016 | 0.007 | 3 |
| ga_gd | ascent_lr=2.5e-05 | `loss_ratio` | unlearned, base | no | 0.115 | 0.029 | 3 |
| ga_gd | ascent_lr=2.5e-05 | `loss_unlearned` | unlearned | yes | 0.369 | 0.029 | 3 |
| ga_gd | ascent_lr=2.5e-05 | `raw_cosine` | unlearned, finetuned, gradients | no | 0.030 | 0.022 | 3 |
| ga_gd | ascent_lr=2e-05 | `alignment_cos_sq` | unlearned, finetuned, gradients | no | 0.006 | 0.004 | 3 |
| ga_gd | ascent_lr=2e-05 | `grad_norm_sq` | unlearned, gradients | no | 0.420 | 0.018 | 3 |
| ga_gd | ascent_lr=2e-05 | `grad_ratio` | unlearned, finetuned, gradients | no | 0.105 | 0.019 | 3 |
| ga_gd | ascent_lr=2e-05 | `log_grad_ratio` | unlearned, finetuned, gradients | no | 0.105 | 0.019 | 3 |
| ga_gd | ascent_lr=2e-05 | `loss_delta` | unlearned, base | no | 0.185 | 0.015 | 3 |
| ga_gd | ascent_lr=2e-05 | `loss_delta_vs_finetuned` | unlearned, finetuned | yes | -0.001 | 0.000 | 3 |
| ga_gd | ascent_lr=2e-05 | `loss_ratio` | unlearned, base | no | 0.201 | 0.019 | 3 |
| ga_gd | ascent_lr=2e-05 | `loss_unlearned` | unlearned | yes | 0.414 | 0.019 | 3 |
| ga_gd | ascent_lr=2e-05 | `raw_cosine` | unlearned, finetuned, gradients | no | 0.005 | 0.004 | 3 |
| ga_gd | ascent_lr=3.5e-05 | `alignment_cos_sq` | unlearned, finetuned, gradients | no | 0.059 | 0.047 | 3 |
| ga_gd | ascent_lr=3.5e-05 | `grad_norm_sq` | unlearned, gradients | no | 0.093 | 0.015 | 3 |
| ga_gd | ascent_lr=3.5e-05 | `grad_ratio` | unlearned, finetuned, gradients | no | 0.077 | 0.007 | 3 |
| ga_gd | ascent_lr=3.5e-05 | `log_grad_ratio` | unlearned, finetuned, gradients | no | 0.077 | 0.007 | 3 |
| ga_gd | ascent_lr=3.5e-05 | `loss_delta` | unlearned, base | no | -0.020 | 0.031 | 3 |
| ga_gd | ascent_lr=3.5e-05 | `loss_delta_vs_finetuned` | unlearned, finetuned | yes | -0.016 | 0.030 | 3 |
| ga_gd | ascent_lr=3.5e-05 | `loss_ratio` | unlearned, base | no | -0.015 | 0.009 | 3 |
| ga_gd | ascent_lr=3.5e-05 | `loss_unlearned` | unlearned | yes | 0.012 | 0.027 | 3 |
| ga_gd | ascent_lr=3.5e-05 | `raw_cosine` | unlearned, finetuned, gradients | no | 0.059 | 0.047 | 3 |
| ga_gd | ascent_lr=3e-05 | `alignment_cos_sq` | unlearned, finetuned, gradients | no | 0.081 | 0.009 | 3 |
| ga_gd | ascent_lr=3e-05 | `grad_norm_sq` | unlearned, gradients | no | 0.259 | 0.030 | 3 |
| ga_gd | ascent_lr=3e-05 | `grad_ratio` | unlearned, finetuned, gradients | no | 0.222 | 0.028 | 3 |
| ga_gd | ascent_lr=3e-05 | `log_grad_ratio` | unlearned, finetuned, gradients | no | 0.222 | 0.028 | 3 |
| ga_gd | ascent_lr=3e-05 | `loss_delta` | unlearned, base | no | -0.034 | 0.024 | 3 |
| ga_gd | ascent_lr=3e-05 | `loss_delta_vs_finetuned` | unlearned, finetuned | yes | -0.035 | 0.024 | 3 |
| ga_gd | ascent_lr=3e-05 | `loss_ratio` | unlearned, base | no | -0.032 | 0.028 | 3 |
| ga_gd | ascent_lr=3e-05 | `loss_unlearned` | unlearned | yes | 0.179 | 0.038 | 3 |
| ga_gd | ascent_lr=3e-05 | `raw_cosine` | unlearned, finetuned, gradients | no | 0.081 | 0.009 | 3 |
| ga_gd | ascent_lr=4.5e-05 | `alignment_cos_sq` | unlearned, finetuned, gradients | no | 0.006 | 0.036 | 3 |
| ga_gd | ascent_lr=4.5e-05 | `grad_norm_sq` | unlearned, gradients | no | -0.018 | 0.052 | 3 |
| ga_gd | ascent_lr=4.5e-05 | `grad_ratio` | unlearned, finetuned, gradients | no | -0.021 | 0.036 | 3 |
| ga_gd | ascent_lr=4.5e-05 | `log_grad_ratio` | unlearned, finetuned, gradients | no | -0.021 | 0.036 | 3 |
| ga_gd | ascent_lr=4.5e-05 | `loss_delta` | unlearned, base | no | -0.024 | 0.018 | 3 |
| ga_gd | ascent_lr=4.5e-05 | `loss_delta_vs_finetuned` | unlearned, finetuned | yes | -0.021 | 0.018 | 3 |
| ga_gd | ascent_lr=4.5e-05 | `loss_ratio` | unlearned, base | no | -0.002 | 0.005 | 3 |
| ga_gd | ascent_lr=4.5e-05 | `loss_unlearned` | unlearned | yes | -0.018 | 0.026 | 3 |
| ga_gd | ascent_lr=4.5e-05 | `raw_cosine` | unlearned, finetuned, gradients | no | 0.018 | 0.023 | 3 |
| ga_gd | ascent_lr=4e-05 | `alignment_cos_sq` | unlearned, finetuned, gradients | no | 0.029 | 0.036 | 3 |
| ga_gd | ascent_lr=4e-05 | `grad_norm_sq` | unlearned, gradients | no | 0.016 | 0.037 | 3 |
| ga_gd | ascent_lr=4e-05 | `grad_ratio` | unlearned, finetuned, gradients | no | 0.005 | 0.030 | 3 |
| ga_gd | ascent_lr=4e-05 | `log_grad_ratio` | unlearned, finetuned, gradients | no | 0.005 | 0.030 | 3 |
| ga_gd | ascent_lr=4e-05 | `loss_delta` | unlearned, base | no | -0.024 | 0.024 | 3 |
| ga_gd | ascent_lr=4e-05 | `loss_delta_vs_finetuned` | unlearned, finetuned | yes | -0.020 | 0.023 | 3 |
| ga_gd | ascent_lr=4e-05 | `loss_ratio` | unlearned, base | no | -0.006 | 0.003 | 3 |
| ga_gd | ascent_lr=4e-05 | `loss_unlearned` | unlearned | yes | -0.018 | 0.034 | 3 |
| ga_gd | ascent_lr=4e-05 | `raw_cosine` | unlearned, finetuned, gradients | no | 0.029 | 0.035 | 3 |
| ga_gd | ascent_lr=5e-05 | `alignment_cos_sq` | unlearned, finetuned, gradients | no | 0.051 | 0.043 | 3 |
| ga_gd | ascent_lr=5e-05 | `grad_norm_sq` | unlearned, gradients | no | -0.018 | 0.040 | 3 |
| ga_gd | ascent_lr=5e-05 | `grad_ratio` | unlearned, finetuned, gradients | no | -0.014 | 0.026 | 3 |
| ga_gd | ascent_lr=5e-05 | `log_grad_ratio` | unlearned, finetuned, gradients | no | -0.014 | 0.026 | 3 |
| ga_gd | ascent_lr=5e-05 | `loss_delta` | unlearned, base | no | -0.030 | 0.033 | 3 |
| ga_gd | ascent_lr=5e-05 | `loss_delta_vs_finetuned` | unlearned, finetuned | yes | -0.028 | 0.034 | 3 |
| ga_gd | ascent_lr=5e-05 | `loss_ratio` | unlearned, base | no | -0.003 | 0.004 | 3 |
| ga_gd | ascent_lr=5e-05 | `loss_unlearned` | unlearned | yes | -0.025 | 0.042 | 3 |
| ga_gd | ascent_lr=5e-05 | `raw_cosine` | unlearned, finetuned, gradients | no | 0.052 | 0.043 | 3 |
| null_refit | null_refit | `alignment_cos_sq` | unlearned, finetuned, gradients | no | -0.029 | 0.044 | 3 |
| null_refit | null_refit | `combined_log_score` | unlearned, finetuned, gradients, curvature | no | -0.021 | 0.037 | 3 |
| null_refit | null_refit | `fuma_lora_ratio` | unlearned, gradients, lora_structure | no | -0.010 | 0.012 | 3 |
| null_refit | null_refit | `grad_norm_sq` | unlearned, gradients | no | 0.011 | 0.028 | 3 |
| null_refit | null_refit | `grad_ratio` | unlearned, finetuned, gradients | no | -0.015 | 0.024 | 3 |
| null_refit | null_refit | `log_grad_ratio` | unlearned, finetuned, gradients | no | -0.015 | 0.024 | 3 |
| null_refit | null_refit | `loss_delta` | unlearned, base | no | -0.016 | 0.013 | 3 |
| null_refit | null_refit | `loss_delta_vs_finetuned` | unlearned, finetuned | yes | -0.002 | 0.003 | 3 |
| null_refit | null_refit | `loss_ratio` | unlearned, base | no | -0.007 | 0.020 | 3 |
| null_refit | null_refit | `loss_unlearned` | unlearned | yes | -0.003 | 0.038 | 3 |
| null_refit | null_refit | `raw_cosine` | unlearned, finetuned, gradients | no | 0.018 | 0.024 | 3 |
| null_refit | null_refit | `sif` | unlearned, gradients, curvature | no | 0.011 | 0.029 | 3 |

Positive `mean diff` means the primary detector scored higher. Paired within
seed, so the between-seed variation both share cancels.

## Per seed

The replication unit is the seed, so every one is shown. A setting whose
seeds disagree is reported as disagreeing, never averaged into a verdict.

| method | setting | seed | classification | membership | retain ppl | RNFF forget | deletion AUC |
|---|---|---:|---|---|---|---|---|
| exact_retrain | exact_retrain | 42 | persistent_fingerprint | 0.476 | 1.034 | 1.000 | 0.999 |
| exact_retrain | exact_retrain | 43 | persistent_fingerprint | 0.474 | 1.043 | 1.000 | 0.999 |
| exact_retrain | exact_retrain | 44 | persistent_fingerprint | 0.471 | 1.034 | 1.000 | 1.000 |
| ga_gd | ascent_lr=2.5e-05 | 42 | incomplete_forgetting | 0.009 | 1.075 | 0.478 | 0.959 |
| ga_gd | ascent_lr=2.5e-05 | 43 | incomplete_forgetting | 0.011 | 1.057 | 0.456 | 0.960 |
| ga_gd | ascent_lr=2.5e-05 | 44 | incomplete_forgetting | 0.009 | 1.041 | 0.415 | 0.976 |
| ga_gd | ascent_lr=2e-05 | 42 | incomplete_forgetting | 0.003 | 1.002 | 0.270 | 0.994 |
| ga_gd | ascent_lr=2e-05 | 43 | incomplete_forgetting | 0.004 | 0.996 | 0.263 | 0.997 |
| ga_gd | ascent_lr=2e-05 | 44 | incomplete_forgetting | 0.004 | 0.991 | 0.253 | 0.995 |
| ga_gd | ascent_lr=3.5e-05 | 42 | inconclusive | 0.441 | 3.910 | 3.495 | 0.584 |
| ga_gd | ascent_lr=3.5e-05 | 43 | global_destruction | 0.502 | 29.443 | 8.683 | 0.575 |
| ga_gd | ascent_lr=3.5e-05 | 44 | global_destruction | 0.496 | 4.867 | 4.037 | 0.585 |
| ga_gd | ascent_lr=3e-05 | 42 | inconclusive | 0.103 | 1.410 | 1.135 | 0.782 |
| ga_gd | ascent_lr=3e-05 | 43 | inconclusive | 0.197 | 1.435 | 1.252 | 0.765 |
| ga_gd | ascent_lr=3e-05 | 44 | inconclusive | 0.108 | 1.352 | 1.054 | 0.799 |
| ga_gd | ascent_lr=4.5e-05 | 42 | global_destruction | 0.478 | 1.80e+05 | 28.424 | 0.490 |
| ga_gd | ascent_lr=4.5e-05 | 43 | global_destruction | 0.479 | 7.92e+07 | 44.065 | 0.495 |
| ga_gd | ascent_lr=4.5e-05 | 44 | global_destruction | 0.479 | 88310.154 | 27.204 | 0.527 |
| ga_gd | ascent_lr=4e-05 | 42 | global_destruction | 0.491 | 73.743 | 10.315 | 0.518 |
| ga_gd | ascent_lr=4e-05 | 43 | global_destruction | 0.469 | 42777.750 | 25.813 | 0.496 |
| ga_gd | ascent_lr=4e-05 | 44 | global_destruction | 0.506 | 1151.631 | 16.912 | 0.546 |
| ga_gd | ascent_lr=5e-05 | 42 | global_destruction | 0.478 | 3.35e+07 | 40.988 | 0.500 |
| ga_gd | ascent_lr=5e-05 | 43 | global_destruction | 0.479 | 6.41e+07 | 43.557 | 0.494 |
| ga_gd | ascent_lr=5e-05 | 44 | global_destruction | 0.481 | 5.02e+07 | 42.295 | 0.525 |
| null_refit | null_refit | 42 | incomplete_forgetting | 0.000 | 1.005 | -0.004 | 0.469 |
| null_refit | null_refit | 43 | incomplete_forgetting | 0.001 | 1.007 | 0.017 | 0.497 |
| null_refit | null_refit | 44 | incomplete_forgetting | 0.001 | 0.994 | -0.010 | 0.518 |

## Data quality

Reported for every run, including failed and collapsed ones. Nothing is
excluded from these tables.

| method | setting | seed | rows | populations disjoint | truncated | non-finite |
|---|---|---:|---:|---|---:|---:|
| exact_retrain | exact_retrain | 42 | 1100 | yes | 4 | 0 |
| exact_retrain | exact_retrain | 43 | 1100 | yes | 2 | 0 |
| exact_retrain | exact_retrain | 44 | 1100 | yes | 5 | 0 |
| ga_gd | ascent_lr=2.5e-05 | 42 | 1300 | yes | 5 | 0 |
| ga_gd | ascent_lr=2.5e-05 | 43 | 1300 | yes | 3 | 0 |
| ga_gd | ascent_lr=2.5e-05 | 44 | 1300 | yes | 5 | 0 |
| ga_gd | ascent_lr=2e-05 | 42 | 1300 | yes | 5 | 0 |
| ga_gd | ascent_lr=2e-05 | 43 | 1300 | yes | 3 | 0 |
| ga_gd | ascent_lr=2e-05 | 44 | 1300 | yes | 5 | 0 |
| ga_gd | ascent_lr=3.5e-05 | 42 | 1300 | yes | 5 | 0 |
| ga_gd | ascent_lr=3.5e-05 | 43 | 1300 | yes | 3 | 0 |
| ga_gd | ascent_lr=3.5e-05 | 44 | 1300 | yes | 5 | 0 |
| ga_gd | ascent_lr=3e-05 | 42 | 1300 | yes | 5 | 0 |
| ga_gd | ascent_lr=3e-05 | 43 | 1300 | yes | 3 | 0 |
| ga_gd | ascent_lr=3e-05 | 44 | 1300 | yes | 5 | 0 |
| ga_gd | ascent_lr=4.5e-05 | 42 | 1300 | yes | 5 | 0 |
| ga_gd | ascent_lr=4.5e-05 | 43 | 1300 | yes | 3 | 0 |
| ga_gd | ascent_lr=4.5e-05 | 44 | 1300 | yes | 5 | 0 |
| ga_gd | ascent_lr=4e-05 | 42 | 1300 | yes | 5 | 0 |
| ga_gd | ascent_lr=4e-05 | 43 | 1300 | yes | 3 | 0 |
| ga_gd | ascent_lr=4e-05 | 44 | 1300 | yes | 5 | 0 |
| ga_gd | ascent_lr=5e-05 | 42 | 1300 | yes | 5 | 0 |
| ga_gd | ascent_lr=5e-05 | 43 | 1300 | yes | 3 | 0 |
| ga_gd | ascent_lr=5e-05 | 44 | 1300 | yes | 5 | 0 |
| null_refit | null_refit | 42 | 1100 | yes | 4 | 0 |
| null_refit | null_refit | 43 | 1100 | yes | 2 | 0 |
| null_refit | null_refit | 44 | 1100 | yes | 5 | 0 |

## Caveats that apply to every number above

- **ArxivMIA only.** No cross-dataset claim is licensed by anything here.
- **n = 3 seeds.** Every seed is shown. Means, paired within-seed differences
  and across-seed SDs are the evidence; at this n, distributional assumptions
  cannot be validated and no p-value would carry weight.
- **Access.** Every two-checkpoint detector, including the primary one,
  requires the pre-deletion checkpoint. These are white-box deletion-audit
  tools for a model owner, regulator or authorised verifier — not black-box
  membership-inference attacks.
- **Control displacement.** For both controls ĝ compares two independently
  trained models, so their detector scores carry training randomness that an
  unlearning run's small displacement does not. That is what the null refit
  measures, and why no retraining number is read without it.
- **Bootstrap intervals** are over samples within one run and describe that
  run's precision, not variation between models.
