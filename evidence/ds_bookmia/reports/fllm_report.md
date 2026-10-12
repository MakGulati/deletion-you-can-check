# Deletion-set inference under matched access — FLLM audit

Generated 2026-08-10T02:24:44.323402+00:00.
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
| exact_retrain | exact_retrain | 3 | 0/3 | 3/3 | 3/3 | **0/3** | 0 | incomplete_forgetting |
| ga_gd | ascent_lr=2.6e-05 | 3 | 0/3 | 1/3 | 3/3 | **0/3** | 0 | incomplete_forgetting |
| ga_gd | ascent_lr=5e-05 | 3 | 1/3 | 0/3 | 0/3 | **0/3** | 2 | inconclusive |
| npo | lr=0.0001, beta=0.1 | 3 | 0/3 | 0/3 | 2/3 | **0/3** | 0 | inconclusive |
| scrub | forget_kl=0.1, lr=5e-05 | 3 | 1/3 | 0/3 | 0/3 | **0/3** | 0 | global_destruction |

Settings whose seeds disagreed, shown per seed rather than averaged:

- `ga_gd / ascent_lr=5e-05` — seed 42: global_destruction, seed 43: failed_run, seed 44: failed_run
- `npo / lr=0.0001, beta=0.1` — seed 42: incomplete_forgetting, seed 43: incomplete_forgetting, seed 44: global_destruction

## Table 1 — Control calibration

What the axes and the detectors read when the answer is known in advance.
`null_refit` and `routine_update` deleted **nothing**, so anything they
produce is a false positive; the first measures training randomness and
the second measures ordinary maintenance.

| condition | seeds | deletion | retain ppl | test ppl | RNFF forget | RNFF retain |
|---|---:|---|---|---|---|---|
| finetuned_reference | 3 | none (this is the origin) | 1.000 | 1.000 | 0.000 | 0.000 |
| null_refit | 3 | none; only the training RNG changed | 0.998 ± 0.003 | 1.001 ± 0.002 | -0.008 ± 0.018 | -0.011 ± 0.019 |
| routine_update | 3 | none; an ordinary maintenance update was applied | 1.012 ± 0.003 | 0.991 ± 0.002 | 0.063 ± 0.014 | 0.063 ± 0.015 |
| exact_retrain | 3 | exact -- the forget set was never trained on | 1.032 ± 0.002 | 0.993 ± 0.003 | 1.000 ± 0.000 | 0.174 ± 0.013 |
| exact_retrain_routine_update | 3 | exact, followed by the identical maintenance update | 1.041 ± 0.003 | 0.987 ± 0.002 | 0.947 ± 0.006 | 0.222 ± 0.019 |

## Table 3 — Matched-access detector comparison

Capability is quoted on `exact_retrain`, so every number in a row describes the same experiment.

**`comparable`** is the matched-access rule: a detector may be claimed
advantageous over the two-checkpoint loss baseline only if it already
holds everything that baseline holds. A `no` means the row is reported
for completeness and supports no comparative claim.

| detector | threat band | fwd | bwd | comparable | s/sample | peak MiB | ROC-AUC | PR-AUC | TPR@1%FPR | top-k recall |
|---|---|---:|---:|:--:|---:|---:|---|---|---|---|
| `loss_ratio_vs_finetuned` | None | None | None | no | -- | -- | 0.990 ± 0.001 | 0.918 ± 0.005 | 0.603 ± 0.127 | 0.918 ± 0.003 |
| `raw_cosine` | None | None | None | no | -- | -- | 0.995 ± 0.001 | 0.967 ± 0.017 | 0.903 ± 0.044 | 0.948 ± 0.006 |
| `alignment_cos_sq` | None | None | None | no | -- | -- | 0.995 ± 0.001 | 0.967 ± 0.017 | 0.903 ± 0.044 | 0.948 ± 0.006 |
| `grad_ratio` | None | None | None | no | -- | -- | 0.927 ± 0.006 | 0.781 ± 0.008 | 0.288 ± 0.093 | 0.762 ± 0.013 |
| `sif` | None | None | None | no | -- | -- | -- | -- | -- | -- |
| `combined_log_score` | None | None | None | no | -- | -- | -- | -- | -- | -- |

## Table 4 — NPO strength sweep

Every strength and every seed, including collapsed and failed runs.

| setting | seed | steps | stop | membership | retain ppl | RNFF forget | RNFF retain | classification |
|---|---:|---:|---|---|---|---|---|---|
| lr=0.0001, beta=0.1 | 42 | 100 | completed | 0.237 | 1.098 | 1.038 | 0.482 | incomplete_forgetting |
| lr=0.0001, beta=0.1 | 43 | 100 | completed | 0.242 | 1.087 | 1.023 | 0.466 | incomplete_forgetting |
| lr=0.0001, beta=0.1 | 44 | 100 | completed | 0.278 | 1.115 | 1.176 | 0.626 | global_destruction |

## Failure log

Failed or collapsed runs: **2**. All of them appear in the tables above.

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
