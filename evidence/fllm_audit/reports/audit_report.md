# Three-axis unlearning audit — ArxivMIA

Generated 2026-08-09T17:33:02.864157+00:00 from 45 score file(s).
Preregistration: `deletion_signals/configs/preregistration_v3_fllm.yaml` (schema v2).

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

**No setting satisfied the jointly preregistered criteria, or uncertainty prevents classification. A statement about what was tested, not about what is true.**

No setting satisfied the three joint criteria on every seed.

| method | setting | membership | utility | selective | detectable | candidate |
|---|---|---|---|---|---|---|
| exact_retrain | exact_retrain | 3/3 | 3/3 | 3/3 | 3/3 | yes |
| exact_retrain_routine_update | exact_retrain_routine_update | 3/3 | 3/3 | 2/3 | 3/3 | no |
| ga_gd | ascent_lr=2.6e-05 | 0/3 | 2/3 | 1/3 | 3/3 | no |
| ga_gd | ascent_lr=5e-05 | 3/3 | 0/3 | 0/3 | 0/3 | no |
| npo | lr=0.0001, beta=0.1 | 0/3 | 0/3 | 0/3 | 3/3 | no |
| npo | lr=1e-05, beta=0.1 | 0/3 | 0/3 | 0/3 | 0/3 | no |
| npo | lr=2.8e-05, beta=0.1 | 0/3 | 0/3 | 0/3 | 0/3 | no |
| npo | lr=2e-05, beta=0.1 | 0/3 | 0/3 | 0/3 | 0/3 | no |
| npo | lr=5e-05, beta=0.1 | 0/3 | 0/3 | 0/3 | 3/3 | no |
| null_refit | null_refit | 0/3 | 3/3 | 0/3 | 0/3 | no |
| routine_update | routine_update | 0/3 | 3/3 | 0/3 | 0/3 | no |
| scrub | forget_kl=1, lr=1e-05 | 0/3 | 0/3 | 0/3 | 0/3 | no |
| scrub | forget_kl=1, lr=5e-05 | 0/3 | 0/3 | 0/3 | 0/3 | no |
| scrub | forget_kl=4, lr=1e-05 | 1/3 | 0/3 | 0/3 | 0/3 | no |
| scrub | forget_kl=4, lr=5e-05 | 2/3 | 0/3 | 0/3 | 0/3 | no |

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
| exact_retrain_routine_update | 3 | 1.088 ± 0.005 | 0.943 ± 0.006 | 0.911 ± 0.007 | 0.527 ± 0.015 |
| null_refit | 3 | 1.002 ± 0.007 | 1.000 ± 0.003 | 0.001 ± 0.014 | 0.001 ± 0.000 |
| routine_update | 3 | 1.057 ± 0.002 | 0.951 ± 0.003 | 0.132 ± 0.009 | 0.001 ± 0.001 |

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
| exact_retrain_routine_update | exact_retrain_routine_update | 0.527 ± 0.015 | 0.867 ± 0.003 | 0.999 ± 0.001 |
| ga_gd | ascent_lr=2.6e-05 | 0.013 ± 0.003 | 0.996 ± 0.001 | 0.999 ± 0.000 |
| ga_gd | ascent_lr=5e-05 | 0.479 ± 0.002 | 0.527 ± 0.013 | 0.529 ± 0.018 |
| npo | lr=0.0001, beta=0.1 | 0.307 ± 0.040 | 0.912 ± 0.006 | 0.985 ± 0.006 |
| npo | lr=1e-05, beta=0.1 | 0.018 ± 0.014 | 0.985 ± 0.015 | 0.986 ± 0.016 |
| npo | lr=2.8e-05, beta=0.1 | 0.101 ± 0.017 | 0.947 ± 0.013 | 0.963 ± 0.017 |
| npo | lr=2e-05, beta=0.1 | 0.079 ± 0.020 | 0.952 ± 0.009 | 0.962 ± 0.009 |
| npo | lr=5e-05, beta=0.1 | 0.160 ± 0.016 | 0.940 ± 0.007 | 0.973 ± 0.014 |
| null_refit | null_refit | 0.001 ± 0.000 | 0.999 ± 0.001 | 0.999 ± 0.001 |
| routine_update | routine_update | 0.001 ± 0.001 | 0.999 ± 0.000 | 0.999 ± 0.000 |
| scrub | forget_kl=1, lr=1e-05 | 0.007 ± 0.003 | 0.995 ± 0.002 | 0.996 ± 0.002 |
| scrub | forget_kl=1, lr=5e-05 | 0.365 ± 0.031 | 0.748 ± 0.031 | 0.786 ± 0.034 |
| scrub | forget_kl=4, lr=1e-05 | 0.374 ± 0.078 | 0.649 ± 0.077 | 0.656 ± 0.077 |
| scrub | forget_kl=4, lr=5e-05 | 0.463 ± 0.020 | 0.542 ± 0.019 | 0.543 ± 0.021 |

## Axis 2 — utility, and Axis 3 — selective forgetting

RNFF is the fraction of the *exact-retraining* loss change achieved:
1.0 means the forget set moved exactly as far as deleting it would have.
Global destruction fails the retain clause by construction, because it
moves retain as far as it moves forget.

| method | setting | retain ppl | test ppl | RNFF forget | RNFF retain | stop reason(s) |
|---|---|---|---|---|---|---|
| exact_retrain | exact_retrain | 1.037 ± 0.005 | 0.989 ± 0.007 | 1.000 ± 0.000 | 0.087 ± 0.013 | -- |
| exact_retrain_routine_update | exact_retrain_routine_update | 1.088 ± 0.005 | 0.943 ± 0.006 | 0.911 ± 0.007 | 0.200 ± 0.014 | -- |
| ga_gd | ascent_lr=2.6e-05 | 1.085 ± 0.020 | 1.060 ± 0.008 | 0.517 ± 0.037 | 0.193 ± 0.042 | completed |
| ga_gd | ascent_lr=5e-05 | 4.95e+07 ± 1.58e+07 | 3.00e+07 ± 1.19e+07 | 42.280 ± 1.285 | 42.009 ± 1.398 | collapse_guard |
| npo | lr=0.0001, beta=0.1 | 1.221 ± 0.020 | 1.125 ± 0.033 | 1.007 ± 0.038 | 0.474 ± 0.034 | completed |
| npo | lr=1e-05, beta=0.1 | 1.226 ± 0.065 | 1.088 ± 0.046 | 0.549 ± 0.102 | 0.481 ± 0.123 | completed |
| npo | lr=2.8e-05, beta=0.1 | 1.308 ± 0.014 | 1.141 ± 0.011 | 0.803 ± 0.065 | 0.638 ± 0.033 | completed |
| npo | lr=2e-05, beta=0.1 | 1.323 ± 0.032 | 1.148 ± 0.013 | 0.788 ± 0.100 | 0.665 ± 0.066 | completed |
| npo | lr=5e-05, beta=0.1 | 1.260 ± 0.018 | 1.127 ± 0.026 | 0.840 ± 0.041 | 0.548 ± 0.034 | completed |
| null_refit | null_refit | 1.002 ± 0.007 | 1.000 ± 0.003 | 0.001 ± 0.014 | 0.005 ± 0.016 | -- |
| routine_update | routine_update | 1.057 ± 0.002 | 0.951 ± 0.003 | 0.132 ± 0.009 | 0.131 ± 0.005 | -- |
| scrub | forget_kl=1, lr=1e-05 | 1.330 ± 0.015 | 1.219 ± 0.013 | 0.730 ± 0.013 | 0.677 ± 0.022 | completed |
| scrub | forget_kl=1, lr=5e-05 | 4.077 ± 0.547 | 3.728 ± 0.379 | 3.826 ± 0.373 | 3.327 ± 0.366 | completed |
| scrub | forget_kl=4, lr=1e-05 | 5.464 ± 2.914 | 4.003 ± 1.848 | 3.861 ± 1.198 | 3.814 ± 1.216 | completed |
| scrub | forget_kl=4, lr=5e-05 | 1.85e+18 ± 3.20e+18 | 9.02e+17 ± 1.56e+18 | 58.440 ± 38.272 | 58.427 ± 38.252 | completed |

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
| exact_retrain_routine_update | exact_retrain_routine_update | `alignment_cos_sq` | unlearned, finetuned, gradients | no | 0.007 | 0.003 | 3 |
| exact_retrain_routine_update | exact_retrain_routine_update | `combined_log_score` | unlearned, finetuned, gradients, curvature | no | -0.003 | 0.002 | 3 |
| exact_retrain_routine_update | exact_retrain_routine_update | `fuma_lora_ratio` | unlearned, gradients, lora_structure | no | 0.451 | 0.026 | 3 |
| exact_retrain_routine_update | exact_retrain_routine_update | `grad_norm_sq` | unlearned, gradients | no | 0.344 | 0.014 | 3 |
| exact_retrain_routine_update | exact_retrain_routine_update | `grad_ratio` | unlearned, finetuned, gradients | no | 0.048 | 0.007 | 3 |
| exact_retrain_routine_update | exact_retrain_routine_update | `log_grad_ratio` | unlearned, finetuned, gradients | no | 0.048 | 0.007 | 3 |
| exact_retrain_routine_update | exact_retrain_routine_update | `loss_delta` | unlearned, base | no | -0.005 | 0.001 | 3 |
| exact_retrain_routine_update | exact_retrain_routine_update | `loss_delta_vs_finetuned` | unlearned, finetuned | yes | -0.004 | 0.002 | 3 |
| exact_retrain_routine_update | exact_retrain_routine_update | `loss_ratio` | unlearned, base | no | -0.005 | 0.001 | 3 |
| exact_retrain_routine_update | exact_retrain_routine_update | `loss_unlearned` | unlearned | yes | 0.292 | 0.021 | 3 |
| exact_retrain_routine_update | exact_retrain_routine_update | `raw_cosine` | unlearned, finetuned, gradients | no | 0.007 | 0.003 | 3 |
| exact_retrain_routine_update | exact_retrain_routine_update | `sif` | unlearned, gradients, curvature | no | 0.335 | 0.016 | 3 |
| ga_gd | ascent_lr=2.6e-05 | `alignment_cos_sq` | unlearned, finetuned, gradients | no | 0.043 | 0.027 | 3 |
| ga_gd | ascent_lr=2.6e-05 | `combined_log_score` | unlearned, finetuned, gradients, curvature | no | 0.026 | 0.007 | 3 |
| ga_gd | ascent_lr=2.6e-05 | `fuma_lora_ratio` | unlearned, gradients, lora_structure | no | 0.391 | 0.006 | 3 |
| ga_gd | ascent_lr=2.6e-05 | `grad_norm_sq` | unlearned, gradients | no | 0.371 | 0.025 | 3 |
| ga_gd | ascent_lr=2.6e-05 | `grad_ratio` | unlearned, finetuned, gradients | no | 0.201 | 0.006 | 3 |
| ga_gd | ascent_lr=2.6e-05 | `log_grad_ratio` | unlearned, finetuned, gradients | no | 0.201 | 0.006 | 3 |
| ga_gd | ascent_lr=2.6e-05 | `loss_delta` | unlearned, base | no | 0.083 | 0.022 | 3 |
| ga_gd | ascent_lr=2.6e-05 | `loss_delta_vs_finetuned` | unlearned, finetuned | yes | -0.022 | 0.009 | 3 |
| ga_gd | ascent_lr=2.6e-05 | `loss_ratio` | unlearned, base | no | 0.088 | 0.029 | 3 |
| ga_gd | ascent_lr=2.6e-05 | `loss_unlearned` | unlearned | yes | 0.345 | 0.035 | 3 |
| ga_gd | ascent_lr=2.6e-05 | `raw_cosine` | unlearned, finetuned, gradients | no | 0.043 | 0.027 | 3 |
| ga_gd | ascent_lr=2.6e-05 | `sif` | unlearned, gradients, curvature | no | 0.372 | 0.027 | 3 |
| ga_gd | ascent_lr=5e-05 | `alignment_cos_sq` | unlearned, finetuned, gradients | no | 0.047 | 0.054 | 3 |
| ga_gd | ascent_lr=5e-05 | `combined_log_score` | unlearned, finetuned, gradients, curvature | no | 0.037 | 0.054 | 3 |
| ga_gd | ascent_lr=5e-05 | `fuma_lora_ratio` | unlearned, gradients, lora_structure | no | -0.001 | 0.018 | 3 |
| ga_gd | ascent_lr=5e-05 | `grad_norm_sq` | unlearned, gradients | no | -0.020 | 0.040 | 3 |
| ga_gd | ascent_lr=5e-05 | `grad_ratio` | unlearned, finetuned, gradients | no | -0.013 | 0.023 | 3 |
| ga_gd | ascent_lr=5e-05 | `log_grad_ratio` | unlearned, finetuned, gradients | no | -0.013 | 0.023 | 3 |
| ga_gd | ascent_lr=5e-05 | `loss_delta` | unlearned, base | no | -0.031 | 0.024 | 3 |
| ga_gd | ascent_lr=5e-05 | `loss_delta_vs_finetuned` | unlearned, finetuned | yes | -0.029 | 0.025 | 3 |
| ga_gd | ascent_lr=5e-05 | `loss_ratio` | unlearned, base | no | -0.003 | 0.002 | 3 |
| ga_gd | ascent_lr=5e-05 | `loss_unlearned` | unlearned | yes | -0.026 | 0.035 | 3 |
| ga_gd | ascent_lr=5e-05 | `raw_cosine` | unlearned, finetuned, gradients | no | 0.048 | 0.055 | 3 |
| ga_gd | ascent_lr=5e-05 | `sif` | unlearned, gradients, curvature | no | 0.003 | 0.044 | 3 |
| npo | lr=0.0001, beta=0.1 | `alignment_cos_sq` | unlearned, finetuned, gradients | no | -0.103 | 0.022 | 3 |
| npo | lr=0.0001, beta=0.1 | `combined_log_score` | unlearned, finetuned, gradients, curvature | no | -0.020 | 0.035 | 3 |
| npo | lr=0.0001, beta=0.1 | `fuma_lora_ratio` | unlearned, gradients, lora_structure | no | 0.271 | 0.020 | 3 |
| npo | lr=0.0001, beta=0.1 | `grad_norm_sq` | unlearned, gradients | no | 0.258 | 0.012 | 3 |
| npo | lr=0.0001, beta=0.1 | `grad_ratio` | unlearned, finetuned, gradients | no | 0.049 | 0.025 | 3 |
| npo | lr=0.0001, beta=0.1 | `log_grad_ratio` | unlearned, finetuned, gradients | no | 0.049 | 0.025 | 3 |
| npo | lr=0.0001, beta=0.1 | `loss_delta` | unlearned, base | no | -0.023 | 0.016 | 3 |
| npo | lr=0.0001, beta=0.1 | `loss_delta_vs_finetuned` | unlearned, finetuned | yes | -0.019 | 0.007 | 3 |
| npo | lr=0.0001, beta=0.1 | `loss_ratio` | unlearned, base | no | -0.019 | 0.017 | 3 |
| npo | lr=0.0001, beta=0.1 | `loss_unlearned` | unlearned | yes | 0.199 | 0.026 | 3 |
| npo | lr=0.0001, beta=0.1 | `raw_cosine` | unlearned, finetuned, gradients | no | -0.103 | 0.022 | 3 |
| npo | lr=0.0001, beta=0.1 | `sif` | unlearned, gradients, curvature | no | 0.258 | 0.011 | 3 |
| npo | lr=1e-05, beta=0.1 | `alignment_cos_sq` | unlearned, finetuned, gradients | no | -0.027 | 0.029 | 3 |
| npo | lr=1e-05, beta=0.1 | `combined_log_score` | unlearned, finetuned, gradients, curvature | no | 0.035 | 0.012 | 3 |
| npo | lr=1e-05, beta=0.1 | `fuma_lora_ratio` | unlearned, gradients, lora_structure | no | 0.103 | 0.044 | 3 |
| npo | lr=1e-05, beta=0.1 | `grad_norm_sq` | unlearned, gradients | no | 0.114 | 0.020 | 3 |
| npo | lr=1e-05, beta=0.1 | `grad_ratio` | unlearned, finetuned, gradients | no | 0.091 | 0.011 | 3 |
| npo | lr=1e-05, beta=0.1 | `log_grad_ratio` | unlearned, finetuned, gradients | no | 0.091 | 0.011 | 3 |
| npo | lr=1e-05, beta=0.1 | `loss_delta` | unlearned, base | no | -0.010 | 0.011 | 3 |
| npo | lr=1e-05, beta=0.1 | `loss_delta_vs_finetuned` | unlearned, finetuned | yes | -0.021 | 0.020 | 3 |
| npo | lr=1e-05, beta=0.1 | `loss_ratio` | unlearned, base | no | -0.011 | 0.014 | 3 |
| npo | lr=1e-05, beta=0.1 | `loss_unlearned` | unlearned | yes | 0.077 | 0.038 | 3 |
| npo | lr=1e-05, beta=0.1 | `raw_cosine` | unlearned, finetuned, gradients | no | -0.027 | 0.029 | 3 |
| npo | lr=1e-05, beta=0.1 | `sif` | unlearned, gradients, curvature | no | 0.109 | 0.021 | 3 |
| npo | lr=2.8e-05, beta=0.1 | `alignment_cos_sq` | unlearned, finetuned, gradients | no | -0.117 | 0.054 | 3 |
| npo | lr=2.8e-05, beta=0.1 | `combined_log_score` | unlearned, finetuned, gradients, curvature | no | 0.022 | 0.018 | 3 |
| npo | lr=2.8e-05, beta=0.1 | `fuma_lora_ratio` | unlearned, gradients, lora_structure | no | 0.121 | 0.017 | 3 |
| npo | lr=2.8e-05, beta=0.1 | `grad_norm_sq` | unlearned, gradients | no | 0.136 | 0.006 | 3 |
| npo | lr=2.8e-05, beta=0.1 | `grad_ratio` | unlearned, finetuned, gradients | no | 0.075 | 0.010 | 3 |
| npo | lr=2.8e-05, beta=0.1 | `log_grad_ratio` | unlearned, finetuned, gradients | no | 0.075 | 0.010 | 3 |
| npo | lr=2.8e-05, beta=0.1 | `loss_delta` | unlearned, base | no | -0.054 | 0.022 | 3 |
| npo | lr=2.8e-05, beta=0.1 | `loss_delta_vs_finetuned` | unlearned, finetuned | yes | -0.027 | 0.014 | 3 |
| npo | lr=2.8e-05, beta=0.1 | `loss_ratio` | unlearned, base | no | -0.056 | 0.028 | 3 |
| npo | lr=2.8e-05, beta=0.1 | `loss_unlearned` | unlearned | yes | 0.090 | 0.026 | 3 |
| npo | lr=2.8e-05, beta=0.1 | `raw_cosine` | unlearned, finetuned, gradients | no | -0.117 | 0.054 | 3 |
| npo | lr=2.8e-05, beta=0.1 | `sif` | unlearned, gradients, curvature | no | 0.131 | 0.005 | 3 |
| npo | lr=2e-05, beta=0.1 | `alignment_cos_sq` | unlearned, finetuned, gradients | no | -0.077 | 0.057 | 3 |
| npo | lr=2e-05, beta=0.1 | `combined_log_score` | unlearned, finetuned, gradients, curvature | no | 0.026 | 0.012 | 3 |
| npo | lr=2e-05, beta=0.1 | `fuma_lora_ratio` | unlearned, gradients, lora_structure | no | 0.100 | 0.010 | 3 |
| npo | lr=2e-05, beta=0.1 | `grad_norm_sq` | unlearned, gradients | no | 0.115 | 0.010 | 3 |
| npo | lr=2e-05, beta=0.1 | `grad_ratio` | unlearned, finetuned, gradients | no | 0.078 | 0.013 | 3 |
| npo | lr=2e-05, beta=0.1 | `log_grad_ratio` | unlearned, finetuned, gradients | no | 0.078 | 0.013 | 3 |
| npo | lr=2e-05, beta=0.1 | `loss_delta` | unlearned, base | no | -0.047 | 0.020 | 3 |
| npo | lr=2e-05, beta=0.1 | `loss_delta_vs_finetuned` | unlearned, finetuned | yes | -0.023 | 0.012 | 3 |
| npo | lr=2e-05, beta=0.1 | `loss_ratio` | unlearned, base | no | -0.050 | 0.025 | 3 |
| npo | lr=2e-05, beta=0.1 | `loss_unlearned` | unlearned | yes | 0.070 | 0.028 | 3 |
| npo | lr=2e-05, beta=0.1 | `raw_cosine` | unlearned, finetuned, gradients | no | -0.077 | 0.057 | 3 |
| npo | lr=2e-05, beta=0.1 | `sif` | unlearned, gradients, curvature | no | 0.110 | 0.010 | 3 |
| npo | lr=5e-05, beta=0.1 | `alignment_cos_sq` | unlearned, finetuned, gradients | no | -0.153 | 0.043 | 3 |
| npo | lr=5e-05, beta=0.1 | `combined_log_score` | unlearned, finetuned, gradients, curvature | no | 0.006 | 0.038 | 3 |
| npo | lr=5e-05, beta=0.1 | `fuma_lora_ratio` | unlearned, gradients, lora_structure | no | 0.190 | 0.010 | 3 |
| npo | lr=5e-05, beta=0.1 | `grad_norm_sq` | unlearned, gradients | no | 0.193 | 0.010 | 3 |
| npo | lr=5e-05, beta=0.1 | `grad_ratio` | unlearned, finetuned, gradients | no | 0.055 | 0.018 | 3 |
| npo | lr=5e-05, beta=0.1 | `log_grad_ratio` | unlearned, finetuned, gradients | no | 0.055 | 0.018 | 3 |
| npo | lr=5e-05, beta=0.1 | `loss_delta` | unlearned, base | no | -0.042 | 0.012 | 3 |
| npo | lr=5e-05, beta=0.1 | `loss_delta_vs_finetuned` | unlearned, finetuned | yes | -0.024 | 0.010 | 3 |
| npo | lr=5e-05, beta=0.1 | `loss_ratio` | unlearned, base | no | -0.037 | 0.019 | 3 |
| npo | lr=5e-05, beta=0.1 | `loss_unlearned` | unlearned | yes | 0.149 | 0.022 | 3 |
| npo | lr=5e-05, beta=0.1 | `raw_cosine` | unlearned, finetuned, gradients | no | -0.153 | 0.043 | 3 |
| npo | lr=5e-05, beta=0.1 | `sif` | unlearned, gradients, curvature | no | 0.190 | 0.007 | 3 |
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
| routine_update | routine_update | `alignment_cos_sq` | unlearned, finetuned, gradients | no | 0.005 | 0.015 | 3 |
| routine_update | routine_update | `combined_log_score` | unlearned, finetuned, gradients, curvature | no | 0.017 | 0.003 | 3 |
| routine_update | routine_update | `fuma_lora_ratio` | unlearned, gradients, lora_structure | no | 0.008 | 0.017 | 3 |
| routine_update | routine_update | `grad_norm_sq` | unlearned, gradients | no | 0.030 | 0.016 | 3 |
| routine_update | routine_update | `grad_ratio` | unlearned, finetuned, gradients | no | 0.018 | 0.032 | 3 |
| routine_update | routine_update | `log_grad_ratio` | unlearned, finetuned, gradients | no | 0.018 | 0.032 | 3 |
| routine_update | routine_update | `loss_delta` | unlearned, base | no | -0.005 | 0.010 | 3 |
| routine_update | routine_update | `loss_delta_vs_finetuned` | unlearned, finetuned | yes | 0.001 | 0.006 | 3 |
| routine_update | routine_update | `loss_ratio` | unlearned, base | no | 0.004 | 0.021 | 3 |
| routine_update | routine_update | `loss_unlearned` | unlearned | yes | 0.014 | 0.025 | 3 |
| routine_update | routine_update | `raw_cosine` | unlearned, finetuned, gradients | no | 0.005 | 0.015 | 3 |
| routine_update | routine_update | `sif` | unlearned, gradients, curvature | no | 0.030 | 0.017 | 3 |
| scrub | forget_kl=1, lr=1e-05 | `alignment_cos_sq` | unlearned, finetuned, gradients | no | 0.021 | 0.011 | 3 |
| scrub | forget_kl=1, lr=1e-05 | `combined_log_score` | unlearned, finetuned, gradients, curvature | no | -0.002 | 0.018 | 3 |
| scrub | forget_kl=1, lr=1e-05 | `fuma_lora_ratio` | unlearned, gradients, lora_structure | no | 0.056 | 0.022 | 3 |
| scrub | forget_kl=1, lr=1e-05 | `grad_norm_sq` | unlearned, gradients | no | 0.071 | 0.021 | 3 |
| scrub | forget_kl=1, lr=1e-05 | `grad_ratio` | unlearned, finetuned, gradients | no | 0.022 | 0.011 | 3 |
| scrub | forget_kl=1, lr=1e-05 | `log_grad_ratio` | unlearned, finetuned, gradients | no | 0.022 | 0.011 | 3 |
| scrub | forget_kl=1, lr=1e-05 | `loss_delta` | unlearned, base | no | 0.013 | 0.009 | 3 |
| scrub | forget_kl=1, lr=1e-05 | `loss_delta_vs_finetuned` | unlearned, finetuned | yes | -0.011 | 0.016 | 3 |
| scrub | forget_kl=1, lr=1e-05 | `loss_ratio` | unlearned, base | no | 0.014 | 0.014 | 3 |
| scrub | forget_kl=1, lr=1e-05 | `loss_unlearned` | unlearned | yes | 0.068 | 0.018 | 3 |
| scrub | forget_kl=1, lr=1e-05 | `raw_cosine` | unlearned, finetuned, gradients | no | 0.021 | 0.011 | 3 |
| scrub | forget_kl=1, lr=1e-05 | `sif` | unlearned, gradients, curvature | no | 0.076 | 0.020 | 3 |
| scrub | forget_kl=1, lr=5e-05 | `alignment_cos_sq` | unlearned, finetuned, gradients | no | 0.025 | 0.043 | 3 |
| scrub | forget_kl=1, lr=5e-05 | `combined_log_score` | unlearned, finetuned, gradients, curvature | no | -0.013 | 0.018 | 3 |
| scrub | forget_kl=1, lr=5e-05 | `fuma_lora_ratio` | unlearned, gradients, lora_structure | no | 0.072 | 0.027 | 3 |
| scrub | forget_kl=1, lr=5e-05 | `grad_norm_sq` | unlearned, gradients | no | 0.037 | 0.013 | 3 |
| scrub | forget_kl=1, lr=5e-05 | `grad_ratio` | unlearned, finetuned, gradients | no | 0.072 | 0.028 | 3 |
| scrub | forget_kl=1, lr=5e-05 | `log_grad_ratio` | unlearned, finetuned, gradients | no | 0.072 | 0.028 | 3 |
| scrub | forget_kl=1, lr=5e-05 | `loss_delta` | unlearned, base | no | -0.055 | 0.021 | 3 |
| scrub | forget_kl=1, lr=5e-05 | `loss_delta_vs_finetuned` | unlearned, finetuned | yes | -0.063 | 0.012 | 3 |
| scrub | forget_kl=1, lr=5e-05 | `loss_ratio` | unlearned, base | no | -0.018 | 0.013 | 3 |
| scrub | forget_kl=1, lr=5e-05 | `loss_unlearned` | unlearned | yes | 0.025 | 0.009 | 3 |
| scrub | forget_kl=1, lr=5e-05 | `raw_cosine` | unlearned, finetuned, gradients | no | 0.025 | 0.043 | 3 |
| scrub | forget_kl=1, lr=5e-05 | `sif` | unlearned, gradients, curvature | no | 0.037 | 0.024 | 3 |
| scrub | forget_kl=4, lr=1e-05 | `alignment_cos_sq` | unlearned, finetuned, gradients | no | 0.020 | 0.017 | 3 |
| scrub | forget_kl=4, lr=1e-05 | `combined_log_score` | unlearned, finetuned, gradients, curvature | no | 0.029 | 0.008 | 3 |
| scrub | forget_kl=4, lr=1e-05 | `fuma_lora_ratio` | unlearned, gradients, lora_structure | no | 0.007 | 0.024 | 3 |
| scrub | forget_kl=4, lr=1e-05 | `grad_norm_sq` | unlearned, gradients | no | 0.024 | 0.007 | 3 |
| scrub | forget_kl=4, lr=1e-05 | `grad_ratio` | unlearned, finetuned, gradients | no | 0.005 | 0.011 | 3 |
| scrub | forget_kl=4, lr=1e-05 | `log_grad_ratio` | unlearned, finetuned, gradients | no | 0.005 | 0.011 | 3 |
| scrub | forget_kl=4, lr=1e-05 | `loss_delta` | unlearned, base | no | -0.011 | 0.022 | 3 |
| scrub | forget_kl=4, lr=1e-05 | `loss_delta_vs_finetuned` | unlearned, finetuned | yes | -0.002 | 0.016 | 3 |
| scrub | forget_kl=4, lr=1e-05 | `loss_ratio` | unlearned, base | no | -0.009 | 0.010 | 3 |
| scrub | forget_kl=4, lr=1e-05 | `loss_unlearned` | unlearned | yes | 0.008 | 0.032 | 3 |
| scrub | forget_kl=4, lr=1e-05 | `raw_cosine` | unlearned, finetuned, gradients | no | 0.020 | 0.017 | 3 |
| scrub | forget_kl=4, lr=1e-05 | `sif` | unlearned, gradients, curvature | no | 0.030 | 0.015 | 3 |
| scrub | forget_kl=4, lr=5e-05 | `alignment_cos_sq` | unlearned, finetuned, gradients | no | 0.063 | 0.008 | 3 |
| scrub | forget_kl=4, lr=5e-05 | `combined_log_score` | unlearned, finetuned, gradients, curvature | no | 0.061 | 0.016 | 3 |
| scrub | forget_kl=4, lr=5e-05 | `fuma_lora_ratio` | unlearned, gradients, lora_structure | no | 0.022 | 0.034 | 3 |
| scrub | forget_kl=4, lr=5e-05 | `grad_norm_sq` | unlearned, gradients | no | -0.001 | 0.027 | 3 |
| scrub | forget_kl=4, lr=5e-05 | `grad_ratio` | unlearned, finetuned, gradients | no | -0.013 | 0.028 | 3 |
| scrub | forget_kl=4, lr=5e-05 | `log_grad_ratio` | unlearned, finetuned, gradients | no | -0.013 | 0.028 | 3 |
| scrub | forget_kl=4, lr=5e-05 | `loss_delta` | unlearned, base | no | 0.005 | 0.023 | 3 |
| scrub | forget_kl=4, lr=5e-05 | `loss_delta_vs_finetuned` | unlearned, finetuned | yes | 0.007 | 0.022 | 3 |
| scrub | forget_kl=4, lr=5e-05 | `loss_ratio` | unlearned, base | no | -0.003 | 0.004 | 3 |
| scrub | forget_kl=4, lr=5e-05 | `loss_unlearned` | unlearned | yes | 0.010 | 0.028 | 3 |
| scrub | forget_kl=4, lr=5e-05 | `raw_cosine` | unlearned, finetuned, gradients | no | 0.063 | 0.008 | 3 |
| scrub | forget_kl=4, lr=5e-05 | `sif` | unlearned, gradients, curvature | no | 0.017 | 0.012 | 3 |

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
| exact_retrain_routine_update | exact_retrain_routine_update | 42 | persistent_fingerprint | 0.539 | 1.083 | 0.907 | 0.993 |
| exact_retrain_routine_update | exact_retrain_routine_update | 43 | inconclusive | 0.533 | 1.093 | 0.919 | 0.993 |
| exact_retrain_routine_update | exact_retrain_routine_update | 44 | persistent_fingerprint | 0.510 | 1.087 | 0.906 | 0.996 |
| ga_gd | ascent_lr=2.6e-05 | 42 | inconclusive | 0.011 | 1.105 | 0.546 | 0.940 |
| ga_gd | ascent_lr=2.6e-05 | 43 | incomplete_forgetting | 0.017 | 1.085 | 0.529 | 0.935 |
| ga_gd | ascent_lr=2.6e-05 | 44 | incomplete_forgetting | 0.012 | 1.065 | 0.476 | 0.958 |
| ga_gd | ascent_lr=5e-05 | 42 | global_destruction | 0.478 | 3.40e+07 | 40.988 | 0.503 |
| ga_gd | ascent_lr=5e-05 | 43 | global_destruction | 0.479 | 6.55e+07 | 43.557 | 0.487 |
| ga_gd | ascent_lr=5e-05 | 44 | global_destruction | 0.481 | 4.90e+07 | 42.295 | 0.527 |
| npo | lr=0.0001, beta=0.1 | 42 | inconclusive | 0.285 | 1.231 | 0.996 | 0.846 |
| npo | lr=0.0001, beta=0.1 | 43 | inconclusive | 0.353 | 1.198 | 0.975 | 0.883 |
| npo | lr=0.0001, beta=0.1 | 44 | inconclusive | 0.282 | 1.234 | 1.049 | 0.865 |
| npo | lr=1e-05, beta=0.1 | 42 | inconclusive | 0.011 | 1.259 | 0.592 | 0.564 |
| npo | lr=1e-05, beta=0.1 | 43 | inconclusive | 0.010 | 1.151 | 0.433 | 0.658 |
| npo | lr=1e-05, beta=0.1 | 44 | inconclusive | 0.034 | 1.268 | 0.624 | 0.580 |
| npo | lr=2.8e-05, beta=0.1 | 42 | inconclusive | 0.083 | 1.293 | 0.737 | 0.625 |
| npo | lr=2.8e-05, beta=0.1 | 43 | inconclusive | 0.116 | 1.315 | 0.866 | 0.674 |
| npo | lr=2.8e-05, beta=0.1 | 44 | inconclusive | 0.104 | 1.317 | 0.808 | 0.644 |
| npo | lr=2e-05, beta=0.1 | 42 | inconclusive | 0.059 | 1.298 | 0.707 | 0.593 |
| npo | lr=2e-05, beta=0.1 | 43 | inconclusive | 0.100 | 1.359 | 0.901 | 0.632 |
| npo | lr=2e-05, beta=0.1 | 44 | inconclusive | 0.078 | 1.312 | 0.756 | 0.616 |
| npo | lr=5e-05, beta=0.1 | 42 | inconclusive | 0.148 | 1.252 | 0.797 | 0.729 |
| npo | lr=5e-05, beta=0.1 | 43 | inconclusive | 0.179 | 1.246 | 0.846 | 0.766 |
| npo | lr=5e-05, beta=0.1 | 44 | inconclusive | 0.154 | 1.281 | 0.878 | 0.743 |
| null_refit | null_refit | 42 | incomplete_forgetting | 0.000 | 1.005 | -0.004 | 0.469 |
| null_refit | null_refit | 43 | incomplete_forgetting | 0.001 | 1.007 | 0.017 | 0.497 |
| null_refit | null_refit | 44 | incomplete_forgetting | 0.001 | 0.994 | -0.010 | 0.518 |
| routine_update | routine_update | 42 | incomplete_forgetting | 0.000 | 1.055 | 0.125 | 0.505 |
| routine_update | routine_update | 43 | incomplete_forgetting | 0.001 | 1.058 | 0.142 | 0.517 |
| routine_update | routine_update | 44 | incomplete_forgetting | 0.000 | 1.056 | 0.128 | 0.520 |
| scrub | forget_kl=1, lr=1e-05 | 42 | inconclusive | 0.004 | 1.332 | 0.723 | 0.587 |
| scrub | forget_kl=1, lr=1e-05 | 43 | inconclusive | 0.009 | 1.315 | 0.722 | 0.587 |
| scrub | forget_kl=1, lr=1e-05 | 44 | inconclusive | 0.009 | 1.344 | 0.746 | 0.580 |
| scrub | forget_kl=1, lr=5e-05 | 42 | inconclusive | 0.330 | 3.560 | 3.481 | 0.641 |
| scrub | forget_kl=1, lr=5e-05 | 43 | inconclusive | 0.390 | 4.649 | 4.223 | 0.638 |
| scrub | forget_kl=1, lr=5e-05 | 44 | inconclusive | 0.375 | 4.022 | 3.776 | 0.621 |
| scrub | forget_kl=4, lr=1e-05 | 42 | inconclusive | 0.323 | 4.477 | 3.518 | 0.500 |
| scrub | forget_kl=4, lr=1e-05 | 43 | inconclusive | 0.336 | 3.171 | 2.872 | 0.532 |
| scrub | forget_kl=4, lr=1e-05 | 44 | global_destruction | 0.464 | 8.743 | 5.193 | 0.528 |
| scrub | forget_kl=4, lr=5e-05 | 42 | inconclusive | 0.441 | 4.02e+06 | 35.677 | 0.498 |
| scrub | forget_kl=4, lr=5e-05 | 43 | global_destruction | 0.476 | 4.89e+06 | 37.018 | 0.488 |
| scrub | forget_kl=4, lr=5e-05 | 44 | global_destruction | 0.473 | 5.54e+18 | 102.626 | 0.524 |

## Data quality

Reported for every run, including failed and collapsed ones. Nothing is
excluded from these tables.

| method | setting | seed | rows | populations disjoint | truncated | non-finite |
|---|---|---:|---:|---|---:|---:|
| exact_retrain | exact_retrain | 42 | 1100 | yes | 4 | 0 |
| exact_retrain | exact_retrain | 43 | 1100 | yes | 2 | 0 |
| exact_retrain | exact_retrain | 44 | 1100 | yes | 5 | 0 |
| exact_retrain_routine_update | exact_retrain_routine_update | 42 | 1100 | yes | 4 | 0 |
| exact_retrain_routine_update | exact_retrain_routine_update | 43 | 1100 | yes | 2 | 0 |
| exact_retrain_routine_update | exact_retrain_routine_update | 44 | 1100 | yes | 5 | 0 |
| ga_gd | ascent_lr=2.6e-05 | 42 | 1100 | yes | 4 | 0 |
| ga_gd | ascent_lr=2.6e-05 | 43 | 1100 | yes | 2 | 0 |
| ga_gd | ascent_lr=2.6e-05 | 44 | 1100 | yes | 5 | 0 |
| ga_gd | ascent_lr=5e-05 | 42 | 1100 | yes | 4 | 0 |
| ga_gd | ascent_lr=5e-05 | 43 | 1100 | yes | 2 | 0 |
| ga_gd | ascent_lr=5e-05 | 44 | 1100 | yes | 5 | 0 |
| npo | lr=0.0001, beta=0.1 | 42 | 1100 | yes | 4 | 0 |
| npo | lr=0.0001, beta=0.1 | 43 | 1100 | yes | 2 | 0 |
| npo | lr=0.0001, beta=0.1 | 44 | 1100 | yes | 5 | 0 |
| npo | lr=1e-05, beta=0.1 | 42 | 1100 | yes | 4 | 0 |
| npo | lr=1e-05, beta=0.1 | 43 | 1100 | yes | 2 | 0 |
| npo | lr=1e-05, beta=0.1 | 44 | 1100 | yes | 5 | 0 |
| npo | lr=2.8e-05, beta=0.1 | 42 | 1100 | yes | 4 | 0 |
| npo | lr=2.8e-05, beta=0.1 | 43 | 1100 | yes | 2 | 0 |
| npo | lr=2.8e-05, beta=0.1 | 44 | 1100 | yes | 5 | 0 |
| npo | lr=2e-05, beta=0.1 | 42 | 1100 | yes | 4 | 0 |
| npo | lr=2e-05, beta=0.1 | 43 | 1100 | yes | 2 | 0 |
| npo | lr=2e-05, beta=0.1 | 44 | 1100 | yes | 5 | 0 |
| npo | lr=5e-05, beta=0.1 | 42 | 1100 | yes | 4 | 0 |
| npo | lr=5e-05, beta=0.1 | 43 | 1100 | yes | 2 | 0 |
| npo | lr=5e-05, beta=0.1 | 44 | 1100 | yes | 5 | 0 |
| null_refit | null_refit | 42 | 1100 | yes | 4 | 0 |
| null_refit | null_refit | 43 | 1100 | yes | 2 | 0 |
| null_refit | null_refit | 44 | 1100 | yes | 5 | 0 |
| routine_update | routine_update | 42 | 1100 | yes | 4 | 0 |
| routine_update | routine_update | 43 | 1100 | yes | 2 | 0 |
| routine_update | routine_update | 44 | 1100 | yes | 5 | 0 |
| scrub | forget_kl=1, lr=1e-05 | 42 | 1100 | yes | 4 | 0 |
| scrub | forget_kl=1, lr=1e-05 | 43 | 1100 | yes | 2 | 0 |
| scrub | forget_kl=1, lr=1e-05 | 44 | 1100 | yes | 5 | 0 |
| scrub | forget_kl=1, lr=5e-05 | 42 | 1100 | yes | 4 | 0 |
| scrub | forget_kl=1, lr=5e-05 | 43 | 1100 | yes | 2 | 0 |
| scrub | forget_kl=1, lr=5e-05 | 44 | 1100 | yes | 5 | 0 |
| scrub | forget_kl=4, lr=1e-05 | 42 | 1100 | yes | 4 | 0 |
| scrub | forget_kl=4, lr=1e-05 | 43 | 1100 | yes | 2 | 0 |
| scrub | forget_kl=4, lr=1e-05 | 44 | 1100 | yes | 5 | 0 |
| scrub | forget_kl=4, lr=5e-05 | 42 | 1100 | yes | 4 | 0 |
| scrub | forget_kl=4, lr=5e-05 | 43 | 1100 | yes | 2 | 0 |
| scrub | forget_kl=4, lr=5e-05 | 44 | 1100 | yes | 5 | 0 |

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
