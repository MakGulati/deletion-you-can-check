# Three-axis unlearning audit — ArxivMIA

Generated 2026-08-10T02:18:17.344048+00:00 from 24 score file(s).
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
| exact_retrain | exact_retrain | 0/3 | 3/3 | 3/3 | 3/3 | no |
| exact_retrain_routine_update | exact_retrain_routine_update | 0/3 | 3/3 | 0/3 | 3/3 | no |
| ga_gd | ascent_lr=2.6e-05 | 0/3 | 3/3 | 1/3 | 3/3 | no |
| ga_gd | ascent_lr=5e-05 | 1/3 | 0/3 | 0/3 | 0/3 | no |
| npo | lr=0.0001, beta=0.1 | 0/3 | 2/3 | 0/3 | 3/3 | no |
| null_refit | null_refit | 0/3 | 3/3 | 0/3 | 0/3 | no |
| routine_update | routine_update | 0/3 | 3/3 | 0/3 | 0/3 | no |
| scrub | forget_kl=0.1, lr=5e-05 | 1/3 | 0/3 | 0/3 | 0/3 | no |

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
| exact_retrain | 3 | 1.032 ± 0.002 | 0.993 ± 0.003 | 1.000 ± 0.000 | 0.257 ± 0.023 |
| exact_retrain_routine_update | 3 | 1.041 ± 0.003 | 0.987 ± 0.002 | 0.947 ± 0.006 | 0.233 ± 0.021 |
| null_refit | 3 | 0.998 ± 0.003 | 1.001 ± 0.002 | -0.008 ± 0.018 | 0.032 ± 0.004 |
| routine_update | 3 | 1.012 ± 0.003 | 0.991 ± 0.002 | 0.063 ± 0.014 | 0.039 ± 0.005 |

### Is the RNFF denominator measuring deletion?

RNFF divides by the loss change exact retraining produced on the forget
set, *attributing that change to deletion*. The null refit deleted
nothing, so its RNFF_forget is how much of that change two independently
trained models would have produced anyway.

| null-refit RNFF_forget | -0.008 ± 0.018 |
|---|---|
| null-refit RNFF_retain | -0.011 ± 0.019 |

> 
> the null refit, which deleted nothing, shows RNFF_forget near zero, so the denominator is attributable to deletion rather than to training variance.

### False-positive floors on `forget_vs_retain` (null refit)

Nothing was deleted in these runs. A detector scoring above chance here
is reading run-to-run training variation.

| detector | AUC on the null refit |
|---|---|
| `grad_norm_sq` | 0.521 ± 0.025 |
| `loss_ratio` | 0.508 ± 0.031 |
| `loss_delta` | 0.506 ± 0.035 |
| `alignment_cos_sq` | 0.505 ± 0.037 |
| `loss_unlearned` | 0.503 ± 0.009 |
| `grad_ratio` | 0.501 ± 0.019 |
| `log_grad_ratio` | 0.501 ± 0.019 |
| `loss_delta_vs_finetuned` | 0.498 ± 0.001 |
| `loss_ratio_vs_finetuned` | 0.498 ± 0.006 |
| `raw_cosine` | 0.495 ± 0.037 |

## Axis 1 — membership removal (forget vs test)

Under the preregistered higher-is-forget direction, **0.5 is chance and
0.0 is as much evidence as 1.0** — a forgotten record reading far below 0.5
still looks exactly like a training member. Hence the two-sided region.

| method | setting | forget_vs_unseen | member_vs_unseen | retain_vs_unseen |
|---|---|---|---|---|
| exact_retrain | exact_retrain | 0.257 ± 0.023 | 0.914 ± 0.010 | 0.956 ± 0.007 |
| exact_retrain_routine_update | exact_retrain_routine_update | 0.233 ± 0.021 | 0.914 ± 0.009 | 0.951 ± 0.006 |
| ga_gd | ascent_lr=2.6e-05 | 0.110 ± 0.071 | 0.935 ± 0.036 | 0.946 ± 0.027 |
| ga_gd | ascent_lr=5e-05 | 0.572 ± 0.089 | 0.433 ± 0.092 | 0.435 ± 0.092 |
| npo | lr=0.0001, beta=0.1 | 0.252 ± 0.022 | 0.902 ± 0.013 | 0.941 ± 0.011 |
| null_refit | null_refit | 0.032 ± 0.004 | 0.969 ± 0.004 | 0.969 ± 0.004 |
| routine_update | routine_update | 0.039 ± 0.005 | 0.962 ± 0.005 | 0.962 ± 0.005 |
| scrub | forget_kl=0.1, lr=5e-05 | 0.445 ± 0.090 | 0.671 ± 0.073 | 0.700 ± 0.069 |

## Axis 2 — utility, and Axis 3 — selective forgetting

RNFF is the fraction of the *exact-retraining* loss change achieved:
1.0 means the forget set moved exactly as far as deleting it would have.
Global destruction fails the retain clause by construction, because it
moves retain as far as it moves forget.

| method | setting | retain ppl | test ppl | RNFF forget | RNFF retain | stop reason(s) |
|---|---|---|---|---|---|---|
| exact_retrain | exact_retrain | 1.032 ± 0.002 | 0.993 ± 0.003 | 1.000 ± 0.000 | 0.174 ± 0.013 | -- |
| exact_retrain_routine_update | exact_retrain_routine_update | 1.041 ± 0.003 | 0.987 ± 0.002 | 0.947 ± 0.006 | 0.222 ± 0.019 | -- |
| ga_gd | ascent_lr=2.6e-05 | 1.053 ± 0.030 | 1.012 ± 0.003 | 0.615 ± 0.152 | 0.286 ± 0.166 | completed |
| ga_gd | ascent_lr=5e-05 | 1.70e+07 ± 1.82e+07 | 8.42e+06 ± 7.38e+06 | 82.557 ± 23.784 | 82.300 ± 24.154 | collapse_guard, completed |
| npo | lr=0.0001, beta=0.1 | 1.100 ± 0.014 | 1.036 ± 0.011 | 1.079 ± 0.084 | 0.525 ± 0.088 | completed |
| null_refit | null_refit | 0.998 ± 0.003 | 1.001 ± 0.002 | -0.008 ± 0.018 | -0.011 ± 0.019 | -- |
| routine_update | routine_update | 1.012 ± 0.003 | 0.991 ± 0.002 | 0.063 ± 0.014 | 0.063 ± 0.015 | -- |
| scrub | forget_kl=0.1, lr=5e-05 | 1.599 ± 0.041 | 1.389 ± 0.023 | 2.965 ± 0.146 | 2.579 ± 0.144 | completed |

## Matched-access detector comparison

Primary detector: **`loss_ratio_vs_finetuned`** — access `unlearned, finetuned`.

A detector may be claimed as advantageous only over detectors whose access
it already subsumes. Rows where that does not hold are marked, and no claim
may rest on them.

| method | setting | vs detector | its access | weaker/equal | mean diff | SD | n |
|---|---|---|---|---|---|---|---|
| exact_retrain | exact_retrain | `alignment_cos_sq` | unlearned, finetuned, gradients | no | -0.005 | 0.002 | 3 |
| exact_retrain | exact_retrain | `grad_norm_sq` | unlearned, gradients | no | 0.234 | 0.022 | 3 |
| exact_retrain | exact_retrain | `grad_ratio` | unlearned, finetuned, gradients | no | 0.062 | 0.006 | 3 |
| exact_retrain | exact_retrain | `log_grad_ratio` | unlearned, finetuned, gradients | no | 0.062 | 0.006 | 3 |
| exact_retrain | exact_retrain | `loss_delta` | unlearned, base | no | 0.048 | 0.006 | 3 |
| exact_retrain | exact_retrain | `loss_delta_vs_finetuned` | unlearned, finetuned | yes | -0.004 | 0.001 | 3 |
| exact_retrain | exact_retrain | `loss_ratio` | unlearned, base | no | 0.065 | 0.005 | 3 |
| exact_retrain | exact_retrain | `loss_unlearned` | unlearned | yes | 0.368 | 0.014 | 3 |
| exact_retrain | exact_retrain | `raw_cosine` | unlearned, finetuned, gradients | no | -0.005 | 0.002 | 3 |
| exact_retrain_routine_update | exact_retrain_routine_update | `alignment_cos_sq` | unlearned, finetuned, gradients | no | -0.008 | 0.004 | 3 |
| exact_retrain_routine_update | exact_retrain_routine_update | `grad_norm_sq` | unlearned, gradients | no | 0.271 | 0.019 | 3 |
| exact_retrain_routine_update | exact_retrain_routine_update | `grad_ratio` | unlearned, finetuned, gradients | no | 0.113 | 0.012 | 3 |
| exact_retrain_routine_update | exact_retrain_routine_update | `log_grad_ratio` | unlearned, finetuned, gradients | no | 0.113 | 0.012 | 3 |
| exact_retrain_routine_update | exact_retrain_routine_update | `loss_delta` | unlearned, base | no | 0.047 | 0.006 | 3 |
| exact_retrain_routine_update | exact_retrain_routine_update | `loss_delta_vs_finetuned` | unlearned, finetuned | yes | -0.007 | 0.004 | 3 |
| exact_retrain_routine_update | exact_retrain_routine_update | `loss_ratio` | unlearned, base | no | 0.065 | 0.002 | 3 |
| exact_retrain_routine_update | exact_retrain_routine_update | `loss_unlearned` | unlearned | yes | 0.374 | 0.012 | 3 |
| exact_retrain_routine_update | exact_retrain_routine_update | `raw_cosine` | unlearned, finetuned, gradients | no | -0.008 | 0.004 | 3 |
| ga_gd | ascent_lr=2.6e-05 | `alignment_cos_sq` | unlearned, finetuned, gradients | no | 0.022 | 0.031 | 3 |
| ga_gd | ascent_lr=2.6e-05 | `grad_norm_sq` | unlearned, gradients | no | 0.307 | 0.030 | 3 |
| ga_gd | ascent_lr=2.6e-05 | `grad_ratio` | unlearned, finetuned, gradients | no | 0.256 | 0.024 | 3 |
| ga_gd | ascent_lr=2.6e-05 | `log_grad_ratio` | unlearned, finetuned, gradients | no | 0.256 | 0.024 | 3 |
| ga_gd | ascent_lr=2.6e-05 | `loss_delta` | unlearned, base | no | 0.168 | 0.055 | 3 |
| ga_gd | ascent_lr=2.6e-05 | `loss_delta_vs_finetuned` | unlearned, finetuned | yes | 0.005 | 0.008 | 3 |
| ga_gd | ascent_lr=2.6e-05 | `loss_ratio` | unlearned, base | no | 0.178 | 0.057 | 3 |
| ga_gd | ascent_lr=2.6e-05 | `loss_unlearned` | unlearned | yes | 0.342 | 0.089 | 3 |
| ga_gd | ascent_lr=2.6e-05 | `raw_cosine` | unlearned, finetuned, gradients | no | 0.022 | 0.031 | 3 |
| ga_gd | ascent_lr=5e-05 | `alignment_cos_sq` | unlearned, finetuned, gradients | no | 0.030 | 0.024 | 3 |
| ga_gd | ascent_lr=5e-05 | `grad_norm_sq` | unlearned, gradients | no | -0.004 | 0.028 | 3 |
| ga_gd | ascent_lr=5e-05 | `grad_ratio` | unlearned, finetuned, gradients | no | 0.013 | 0.036 | 3 |
| ga_gd | ascent_lr=5e-05 | `log_grad_ratio` | unlearned, finetuned, gradients | no | 0.013 | 0.036 | 3 |
| ga_gd | ascent_lr=5e-05 | `loss_delta` | unlearned, base | no | -0.011 | 0.010 | 3 |
| ga_gd | ascent_lr=5e-05 | `loss_delta_vs_finetuned` | unlearned, finetuned | yes | -0.009 | 0.011 | 3 |
| ga_gd | ascent_lr=5e-05 | `loss_ratio` | unlearned, base | no | -0.004 | 0.007 | 3 |
| ga_gd | ascent_lr=5e-05 | `loss_unlearned` | unlearned | yes | -0.012 | 0.020 | 3 |
| ga_gd | ascent_lr=5e-05 | `raw_cosine` | unlearned, finetuned, gradients | no | 0.030 | 0.024 | 3 |
| npo | lr=0.0001, beta=0.1 | `alignment_cos_sq` | unlearned, finetuned, gradients | no | -0.005 | 0.011 | 3 |
| npo | lr=0.0001, beta=0.1 | `grad_norm_sq` | unlearned, gradients | no | 0.269 | 0.025 | 3 |
| npo | lr=0.0001, beta=0.1 | `grad_ratio` | unlearned, finetuned, gradients | no | 0.206 | 0.011 | 3 |
| npo | lr=0.0001, beta=0.1 | `log_grad_ratio` | unlearned, finetuned, gradients | no | 0.206 | 0.011 | 3 |
| npo | lr=0.0001, beta=0.1 | `loss_delta` | unlearned, base | no | 0.111 | 0.022 | 3 |
| npo | lr=0.0001, beta=0.1 | `loss_delta_vs_finetuned` | unlearned, finetuned | yes | -0.029 | 0.008 | 3 |
| npo | lr=0.0001, beta=0.1 | `loss_ratio` | unlearned, base | no | 0.113 | 0.020 | 3 |
| npo | lr=0.0001, beta=0.1 | `loss_unlearned` | unlearned | yes | 0.339 | 0.009 | 3 |
| npo | lr=0.0001, beta=0.1 | `raw_cosine` | unlearned, finetuned, gradients | no | -0.005 | 0.011 | 3 |
| null_refit | null_refit | `alignment_cos_sq` | unlearned, finetuned, gradients | no | -0.007 | 0.041 | 3 |
| null_refit | null_refit | `grad_norm_sq` | unlearned, gradients | no | -0.023 | 0.021 | 3 |
| null_refit | null_refit | `grad_ratio` | unlearned, finetuned, gradients | no | -0.003 | 0.025 | 3 |
| null_refit | null_refit | `log_grad_ratio` | unlearned, finetuned, gradients | no | -0.003 | 0.025 | 3 |
| null_refit | null_refit | `loss_delta` | unlearned, base | no | -0.007 | 0.032 | 3 |
| null_refit | null_refit | `loss_delta_vs_finetuned` | unlearned, finetuned | yes | -0.000 | 0.005 | 3 |
| null_refit | null_refit | `loss_ratio` | unlearned, base | no | -0.010 | 0.027 | 3 |
| null_refit | null_refit | `loss_unlearned` | unlearned | yes | -0.004 | 0.015 | 3 |
| null_refit | null_refit | `raw_cosine` | unlearned, finetuned, gradients | no | 0.003 | 0.032 | 3 |
| routine_update | routine_update | `alignment_cos_sq` | unlearned, finetuned, gradients | no | -0.004 | 0.013 | 3 |
| routine_update | routine_update | `grad_norm_sq` | unlearned, gradients | no | -0.017 | 0.032 | 3 |
| routine_update | routine_update | `grad_ratio` | unlearned, finetuned, gradients | no | 0.017 | 0.018 | 3 |
| routine_update | routine_update | `log_grad_ratio` | unlearned, finetuned, gradients | no | 0.017 | 0.018 | 3 |
| routine_update | routine_update | `loss_delta` | unlearned, base | no | -0.006 | 0.062 | 3 |
| routine_update | routine_update | `loss_delta_vs_finetuned` | unlearned, finetuned | yes | -0.002 | 0.006 | 3 |
| routine_update | routine_update | `loss_ratio` | unlearned, base | no | -0.009 | 0.054 | 3 |
| routine_update | routine_update | `loss_unlearned` | unlearned | yes | -0.004 | 0.028 | 3 |
| routine_update | routine_update | `raw_cosine` | unlearned, finetuned, gradients | no | -0.003 | 0.013 | 3 |
| scrub | forget_kl=0.1, lr=5e-05 | `alignment_cos_sq` | unlearned, finetuned, gradients | no | -0.008 | 0.026 | 3 |
| scrub | forget_kl=0.1, lr=5e-05 | `grad_norm_sq` | unlearned, gradients | no | 0.027 | 0.011 | 3 |
| scrub | forget_kl=0.1, lr=5e-05 | `grad_ratio` | unlearned, finetuned, gradients | no | 0.021 | 0.013 | 3 |
| scrub | forget_kl=0.1, lr=5e-05 | `log_grad_ratio` | unlearned, finetuned, gradients | no | 0.021 | 0.013 | 3 |
| scrub | forget_kl=0.1, lr=5e-05 | `loss_delta` | unlearned, base | no | 0.004 | 0.020 | 3 |
| scrub | forget_kl=0.1, lr=5e-05 | `loss_delta_vs_finetuned` | unlearned, finetuned | yes | -0.011 | 0.004 | 3 |
| scrub | forget_kl=0.1, lr=5e-05 | `loss_ratio` | unlearned, base | no | 0.011 | 0.019 | 3 |
| scrub | forget_kl=0.1, lr=5e-05 | `loss_unlearned` | unlearned | yes | 0.089 | 0.013 | 3 |
| scrub | forget_kl=0.1, lr=5e-05 | `raw_cosine` | unlearned, finetuned, gradients | no | -0.008 | 0.026 | 3 |

Positive `mean diff` means the primary detector scored higher. Paired within
seed, so the between-seed variation both share cancels.

## Per seed

The replication unit is the seed, so every one is shown. A setting whose
seeds disagree is reported as disagreeing, never averaged into a verdict.

| method | setting | seed | classification | membership | retain ppl | RNFF forget | deletion AUC |
|---|---|---:|---|---|---|---|---|
| exact_retrain | exact_retrain | 42 | incomplete_forgetting | 0.254 | 1.032 | 1.000 | 0.990 |
| exact_retrain | exact_retrain | 43 | incomplete_forgetting | 0.236 | 1.034 | 1.000 | 0.989 |
| exact_retrain | exact_retrain | 44 | incomplete_forgetting | 0.281 | 1.031 | 1.000 | 0.989 |
| exact_retrain_routine_update | exact_retrain_routine_update | 42 | incomplete_forgetting | 0.224 | 1.041 | 0.954 | 0.985 |
| exact_retrain_routine_update | exact_retrain_routine_update | 43 | incomplete_forgetting | 0.218 | 1.044 | 0.944 | 0.977 |
| exact_retrain_routine_update | exact_retrain_routine_update | 44 | incomplete_forgetting | 0.256 | 1.038 | 0.944 | 0.983 |
| ga_gd | ascent_lr=2.6e-05 | 42 | incomplete_forgetting | 0.079 | 1.040 | 0.544 | 0.930 |
| ga_gd | ascent_lr=2.6e-05 | 43 | incomplete_forgetting | 0.059 | 1.032 | 0.512 | 0.954 |
| ga_gd | ascent_lr=2.6e-05 | 44 | incomplete_forgetting | 0.191 | 1.087 | 0.789 | 0.793 |
| ga_gd | ascent_lr=5e-05 | 42 | inconclusive | 0.650 | 40257.144 | 55.438 | 0.509 |
| ga_gd | ascent_lr=5e-05 | 43 | global_destruction | 0.475 | 1.46e+07 | 92.357 | 0.507 |
| ga_gd | ascent_lr=5e-05 | 44 | inconclusive | 0.592 | 3.62e+07 | 99.876 | 0.500 |
| npo | lr=0.0001, beta=0.1 | 42 | incomplete_forgetting | 0.237 | 1.098 | 1.038 | 0.937 |
| npo | lr=0.0001, beta=0.1 | 43 | incomplete_forgetting | 0.242 | 1.087 | 1.023 | 0.923 |
| npo | lr=0.0001, beta=0.1 | 44 | inconclusive | 0.278 | 1.115 | 1.176 | 0.912 |
| null_refit | null_refit | 42 | incomplete_forgetting | 0.032 | 1.001 | 0.003 | 0.495 |
| null_refit | null_refit | 43 | incomplete_forgetting | 0.028 | 0.999 | 0.002 | 0.505 |
| null_refit | null_refit | 44 | incomplete_forgetting | 0.036 | 0.994 | -0.029 | 0.494 |
| routine_update | routine_update | 42 | incomplete_forgetting | 0.036 | 1.011 | 0.065 | 0.534 |
| routine_update | routine_update | 43 | incomplete_forgetting | 0.036 | 1.014 | 0.077 | 0.478 |
| routine_update | routine_update | 44 | incomplete_forgetting | 0.045 | 1.009 | 0.049 | 0.484 |
| scrub | forget_kl=0.1, lr=5e-05 | 42 | inconclusive | 0.446 | 1.613 | 2.915 | 0.652 |
| scrub | forget_kl=0.1, lr=5e-05 | 43 | global_destruction | 0.534 | 1.632 | 3.130 | 0.624 |
| scrub | forget_kl=0.1, lr=5e-05 | 44 | inconclusive | 0.354 | 1.553 | 2.851 | 0.654 |

## Data quality

Reported for every run, including failed and collapsed ones. Nothing is
excluded from these tables.

| method | setting | seed | rows | populations disjoint | truncated | non-finite |
|---|---|---:|---:|---|---:|---:|
| exact_retrain | exact_retrain | 42 | 1300 | yes | 1300 | 0 |
| exact_retrain | exact_retrain | 43 | 1300 | yes | 1300 | 0 |
| exact_retrain | exact_retrain | 44 | 1300 | yes | 1300 | 0 |
| exact_retrain_routine_update | exact_retrain_routine_update | 42 | 1300 | yes | 1300 | 0 |
| exact_retrain_routine_update | exact_retrain_routine_update | 43 | 1300 | yes | 1300 | 0 |
| exact_retrain_routine_update | exact_retrain_routine_update | 44 | 1300 | yes | 1300 | 0 |
| ga_gd | ascent_lr=2.6e-05 | 42 | 1300 | yes | 1300 | 0 |
| ga_gd | ascent_lr=2.6e-05 | 43 | 1300 | yes | 1300 | 0 |
| ga_gd | ascent_lr=2.6e-05 | 44 | 1300 | yes | 1300 | 0 |
| ga_gd | ascent_lr=5e-05 | 42 | 1300 | yes | 1300 | 0 |
| ga_gd | ascent_lr=5e-05 | 43 | 1300 | yes | 1300 | 0 |
| ga_gd | ascent_lr=5e-05 | 44 | 1300 | yes | 1300 | 0 |
| npo | lr=0.0001, beta=0.1 | 42 | 1300 | yes | 1300 | 0 |
| npo | lr=0.0001, beta=0.1 | 43 | 1300 | yes | 1300 | 0 |
| npo | lr=0.0001, beta=0.1 | 44 | 1300 | yes | 1300 | 0 |
| null_refit | null_refit | 42 | 1300 | yes | 1300 | 0 |
| null_refit | null_refit | 43 | 1300 | yes | 1300 | 0 |
| null_refit | null_refit | 44 | 1300 | yes | 1300 | 0 |
| routine_update | routine_update | 42 | 1300 | yes | 1300 | 0 |
| routine_update | routine_update | 43 | 1300 | yes | 1300 | 0 |
| routine_update | routine_update | 44 | 1300 | yes | 1300 | 0 |
| scrub | forget_kl=0.1, lr=5e-05 | 42 | 1300 | yes | 1300 | 0 |
| scrub | forget_kl=0.1, lr=5e-05 | 43 | 1300 | yes | 1300 | 0 |
| scrub | forget_kl=0.1, lr=5e-05 | 44 | 1300 | yes | 1300 | 0 |

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
