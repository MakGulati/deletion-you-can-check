# Deletion-set inference under matched access — FLLM audit

Generated 2026-08-09T17:48:45.932154+00:00.
Preregistration: `deletion_signals/configs/preregistration_v3_fllm.yaml`.

Thresholds were frozen before any run in this extension existed and are
read from that file, not from constants in code.

## Table 2 — Method verdicts

`membership`, `RNFF` and `utility` count how many seeds passed each
criterion **alone**. `conjunction` counts how many passed **all of them
together on the same seed**. The second is never inferred from the first:
three seeds each failing a different criterion is not a pass.

The classification names a **pattern of criteria**, not a magnitude.
`global_destruction` fires whenever utility is out of tolerance *and* the
collateral movement is over the specificity budget — which covers a model
whose perplexity rose 18% as well as one that was destroyed outright. Read
the magnitude off the utility and RNFF columns in Tables 1 and 4, never
off the label.

| method | setting | seeds | membership | RNFF | utility | conjunction | collapsed | classification |
|---|---|---:|---:|---:|---:|---:|---:|---|
| exact_retrain | exact_retrain | 3 | 3/3 | 3/3 | 3/3 | **3/3** | 0 | successful_selective_deletion |
| ga_gd | ascent_lr=2.6e-05 | 3 | 0/3 | 1/3 | 2/3 | **0/3** | 0 | inconclusive |
| ga_gd | ascent_lr=5e-05 | 3 | 3/3 | 0/3 | 0/3 | **0/3** | 3 | failed_run |
| npo | lr=0.0001, beta=0.1 | 3 | 0/3 | 0/3 | 0/3 | **0/3** | 0 | global_destruction |
| npo | lr=1e-05, beta=0.1 | 3 | 0/3 | 0/3 | 0/3 | **0/3** | 0 | global_destruction |
| npo | lr=2.8e-05, beta=0.1 | 3 | 0/3 | 0/3 | 0/3 | **0/3** | 0 | global_destruction |
| npo | lr=2e-05, beta=0.1 | 3 | 0/3 | 0/3 | 0/3 | **0/3** | 0 | global_destruction |
| npo | lr=5e-05, beta=0.1 | 3 | 0/3 | 0/3 | 0/3 | **0/3** | 0 | global_destruction |
| scrub | forget_kl=1, lr=1e-05 | 3 | 0/3 | 0/3 | 0/3 | **0/3** | 0 | global_destruction |
| scrub | forget_kl=1, lr=5e-05 | 3 | 0/3 | 0/3 | 0/3 | **0/3** | 0 | global_destruction |
| scrub | forget_kl=4, lr=1e-05 | 3 | 1/3 | 0/3 | 0/3 | **0/3** | 0 | global_destruction |
| scrub | forget_kl=4, lr=5e-05 | 3 | 2/3 | 0/3 | 0/3 | **0/3** | 0 | global_destruction |

Settings whose seeds disagreed, shown per seed rather than averaged:

- `ga_gd / ascent_lr=2.6e-05` — seed 42: global_destruction, seed 43: incomplete_forgetting, seed 44: incomplete_forgetting

## Table 1 — Control calibration

What the axes and the detectors read when the answer is known in advance.
`null_refit` and `routine_update` deleted **nothing**, so anything they
produce is a false positive; the first measures training randomness and
the second measures ordinary maintenance.

| condition | seeds | deletion | retain ppl | test ppl | RNFF forget | RNFF retain |
|---|---:|---|---|---|---|---|
| finetuned_reference | 3 | none (this is the origin) | 1.000 | 1.000 | 0.000 | 0.000 |
| null_refit | 3 | none; only the training RNG changed | 1.002 ± 0.007 | 1.000 ± 0.003 | 0.001 ± 0.014 | 0.005 ± 0.016 |
| routine_update | 3 | none; an ordinary maintenance update was applied | 1.057 ± 0.002 | 0.951 ± 0.003 | 0.132 ± 0.009 | 0.131 ± 0.005 |
| exact_retrain | 3 | exact -- the forget set was never trained on | 1.037 ± 0.005 | 0.989 ± 0.007 | 1.000 ± 0.000 | 0.087 ± 0.013 |
| exact_retrain_routine_update | 3 | exact, followed by the identical maintenance update | 1.088 ± 0.005 | 0.943 ± 0.006 | 0.911 ± 0.007 | 0.200 ± 0.014 |

## Table 3 — Matched-access detector comparison

Capability is quoted on `exact_retrain`, so every number in a row describes the same experiment.

**`comparable`** is the matched-access rule: a detector may be claimed
advantageous over the two-checkpoint loss baseline only if it already
holds everything that baseline holds. A `no` means the row is reported
for completeness and supports no comparative claim.

| detector | threat band | fwd | bwd | comparable | s/sample | peak MiB | ROC-AUC | PR-AUC | TPR@1%FPR | top-k recall |
|---|---|---:|---:|:--:|---:|---:|---|---|---|---|
| `loss_ratio_vs_finetuned` | authorized_audit | 2 | 0 | yes | 0.0671 | 3494 | 0.999 ± 0.000 | 0.998 ± 0.001 | 0.988 ± 0.016 | 0.982 ± 0.010 |
| `raw_cosine` | authorized_audit | 1 | 1 | yes | 0.1084 | 3494 | 0.999 ± 0.000 | 0.996 ± 0.001 | 0.968 ± 0.008 | 0.968 ± 0.008 |
| `alignment_cos_sq` | authorized_audit | 1 | 1 | yes | 0.1084 | 3494 | 0.999 ± 0.000 | 0.996 ± 0.001 | 0.968 ± 0.008 | 0.968 ± 0.008 |
| `grad_ratio` | authorized_audit | 2 | 2 | yes | 0.2203 | 3494 | 0.990 ± 0.005 | 0.967 ± 0.015 | 0.853 ± 0.040 | 0.922 ± 0.021 |
| `sif` | insider_or_checkpoint_compromise | 1 | 1 | no | 0.1084 | 3429 | 0.724 ± 0.011 | 0.474 ± 0.008 | 0.090 ± 0.010 | 0.447 ± 0.012 |
| `combined_log_score` | authorized_audit | 1 | 1 | yes | 0.1084 | 3494 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.993 ± 0.006 |

## Table 4 — NPO strength sweep

Every strength and every seed, including collapsed and failed runs.

| setting | seed | steps | stop | membership | retain ppl | RNFF forget | RNFF retain | classification |
|---|---:|---:|---|---|---|---|---|---|
| lr=0.0001, beta=0.1 | 42 | 100 | completed | 0.285 | 1.231 | 0.996 | 0.487 | global_destruction |
| lr=0.0001, beta=0.1 | 43 | 100 | completed | 0.353 | 1.198 | 0.975 | 0.436 | global_destruction |
| lr=0.0001, beta=0.1 | 44 | 100 | completed | 0.282 | 1.234 | 1.049 | 0.500 | global_destruction |
| lr=1e-05, beta=0.1 | 42 | 100 | completed | 0.011 | 1.259 | 0.592 | 0.539 | global_destruction |
| lr=1e-05, beta=0.1 | 43 | 100 | completed | 0.010 | 1.151 | 0.433 | 0.339 | global_destruction |
| lr=1e-05, beta=0.1 | 44 | 100 | completed | 0.034 | 1.268 | 0.624 | 0.564 | global_destruction |
| lr=2.8e-05, beta=0.1 | 42 | 100 | completed | 0.083 | 1.293 | 0.737 | 0.600 | global_destruction |
| lr=2.8e-05, beta=0.1 | 43 | 100 | completed | 0.116 | 1.315 | 0.866 | 0.659 | global_destruction |
| lr=2.8e-05, beta=0.1 | 44 | 100 | completed | 0.104 | 1.317 | 0.808 | 0.655 | global_destruction |
| lr=2e-05, beta=0.1 | 42 | 100 | completed | 0.059 | 1.298 | 0.707 | 0.611 | global_destruction |
| lr=2e-05, beta=0.1 | 43 | 100 | completed | 0.100 | 1.359 | 0.901 | 0.738 | global_destruction |
| lr=2e-05, beta=0.1 | 44 | 100 | completed | 0.078 | 1.312 | 0.756 | 0.645 | global_destruction |
| lr=5e-05, beta=0.1 | 42 | 100 | completed | 0.148 | 1.252 | 0.797 | 0.527 | global_destruction |
| lr=5e-05, beta=0.1 | 43 | 100 | completed | 0.179 | 1.246 | 0.846 | 0.530 | global_destruction |
| lr=5e-05, beta=0.1 | 44 | 100 | completed | 0.154 | 1.281 | 0.878 | 0.588 | global_destruction |

## Failure log

Failed or collapsed runs: **3**. All of them appear in the tables above.

## Caveats that apply to every number above

- **ArxivMIA, Llama-3.2-1B, LoRA, and these methods only.** Nothing here
  licenses a claim beyond that scope.
- **n = 3 seeds.** Every seed is shown. Means, paired within-seed
  differences and across-seed SDs are the evidence; n=3 does not provide
  population-level statistical certainty.
- **Membership AUC near chance is not proof of forgetting.** It is one of
  four criteria and never certifies anything alone; a destroyed model
  reaches chance by destroying the signal along with the model.
- **Access.** Every two-checkpoint detector requires the pre-deletion
  checkpoint. These are audit capabilities for a model owner, regulator
  or authorised verifier — not black-box membership-inference attacks,
  and detecting a deletion is not by itself a privacy harm.
- **Bootstrap intervals** are over samples within one run. They describe
  that run's precision, never variation between independently trained
  models.
