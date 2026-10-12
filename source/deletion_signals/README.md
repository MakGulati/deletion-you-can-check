# deletion_signals — head-to-head evaluation of deletion-set inference signals

Does directional gradient alignment with the unlearning direction carry
information beyond ordinary gradient magnitude, calibrated difficulty, and the
FUMA LoRA ratio?

The central task is **forget vs retain**: both classes are known training
members, and the attack must determine which of them were targeted for
unlearning. That is *deletion-set inference*, not membership inference — see
[docs/DESIGN.md](docs/DESIGN.md) for why the distinction changes which
baselines are meaningful.

This package is **self-contained**. It does not import from any sibling project
in this workspace. It does *read* the older
`unlearning_paradox_sanity/results_seed*/…/per_sample_components.json` files,
in a degraded mode, so their three seeds can be re-analysed without a GPU.

## Environment

```bash
PY=python3
$PY -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# expected: 2.11.0+cu128 True
```

`py311` has a `cu130` torch build that cannot initialise against this machine's
driver, and bare `python` resolves to it. Use the absolute path.

## Quick start

```bash
# unit tests (CPU, ~90 s)
$PY -m pytest deletion_signals -c deletion_signals/pytest.ini

# end-to-end smoke test (CPU, ~1 min, pythia-160m)
$PY deletion_signals/scripts/smoke_test.py

# re-analyse the existing three seeds with no GPU (degraded mode)
$PY -m deletion_signals.offline_eval --legacy \
    --inputs 'unlearning_paradox_sanity/results_seed4*/*/per_sample_components.json' \
    --output ds_results/legacy_evaluation
```

## The full evaluation

```bash
# 1. fine-tune, unlearn and score -- three seeds, two methods
$PY -m deletion_signals.run_pipeline \
    --config llama_lora.yaml --seeds 42 43 44 --methods ga_gd scrub \
    --output ds_results

# 2. recompute every metric offline (seconds, CPU)
$PY -m deletion_signals.offline_eval \
    --inputs 'ds_results/seed*/*/per_sample_scores.json' \
    --output ds_results/evaluation
```

Stage 1 is the expensive one. Stage 2 can be re-run freely: a different FPR
target, a new task, a corrected test are all offline recomputes over the saved
per-sample table.

## Scores

| name | definition | reference model |
|---|---|---|
| `raw_cosine` | signed cos(∇L(z), ĝ), ĝ = θ_unlearned − θ_finetuned | — |
| `alignment_cos_sq` | cos²(∇L(z), ĝ) — the sign-invariant form | — |
| `loss_unlearned` | mean token NLL under the unlearned model | — |
| `loss_ratio` | `loss_unlearned / max(loss_base, eps)` — membership | pretrained base |
| `loss_delta` | `loss_unlearned − loss_base` — membership | pretrained base |
| `loss_ratio_vs_finetuned` | `loss_unlearned / max(loss_finetuned, eps)` — what unlearning did | fine-tuned pre-unlearning |
| `loss_delta_vs_finetuned` | `loss_unlearned − loss_finetuned` — what unlearning did | fine-tuned pre-unlearning |
| `grad_norm_sq` | ‖∇L_unlearned(z)‖² | — |
| `grad_ratio` | `grad_norm_sq / max(grad_norm_sq_ref, eps)` | fine-tuned pre-unlearning |
| `log_grad_ratio` | `log(g + eps) − log(g_ref + eps)` | fine-tuned pre-unlearning |
| `fuma_lora_ratio` | mean \|∇L\| on LoRA ÷ mean \|∇L\| off LoRA | bf16 shadow (LoRA mode) |
| `sif` | ∇Lᵀ(C + λI)⁻¹∇L — ablation, off by default | — |
| `combined_log_score` | β·log(SIF) + log(cos²) — the existing combined score | — |

Reference-model choices and the reasoning behind them are in
[docs/DESIGN.md](docs/DESIGN.md); the FUMA conventions and their departures from
the published setting are in [docs/FUMA.md](docs/FUMA.md).

## Preregistered orientations

Every score's direction on every task is fixed **before** measurement in
[`configs/orientations.yaml`](configs/orientations.yaml), each with a written
rationale. All reported AUCs use those directions.

`max(AUC, 1 − AUC)` is not used anywhere. A score that lands below 0.5 under its
declared direction is reported below 0.5. The evaluator refuses to score a
column that has no declared orientation. See
[docs/ORIENTATIONS.md](docs/ORIENTATIONS.md).

## The unlearning-strength sweep

The three committed seeds answer the original question and undermine its
framing: on `forget_vs_unseen` the calibrated loss ratio reaches AUC ≈ 0.999, so
there it measures membership rather than deletion — and no seed reached its
forget/retain loss-ratio target, so the directional signal on `forget_vs_retain`
is also consistent with unlearning not having finished.

The sweep varies `ascent_lr` over five settings × three seeds and asks whether
the signed cosine still identifies the deletion set once the membership evidence
is gone and retained utility survives. Its decision rule, thresholds and the
three outcomes that all count as results are fixed in advance in
[docs/PREREGISTRATION.md](docs/PREREGISTRATION.md).

```bash
# gate first: two strengths, seed 42, 20 steps, ~5 min on GPU
$PY deletion_signals/scripts/sweep_smoke_gate.py

for LR in 1e-5 2e-5 5e-5 1e-4 2e-4; do
  for S in 42 43 44; do
    $PY -m deletion_signals.run_pipeline \
        --config arxivmia_sweep.yaml --seeds $S --methods ga_gd \
        --set unlearning.ga_gd.ascent_lr=$LR \
        --skip-finetune --manifest results/seed{seed}/splits.json \
        --output ds_sweep/ascent_$LR
  done
done

$PY -m deletion_signals.sweep_report \
    --inputs 'ds_sweep/ascent_*/seed*/ga_gd/per_sample_scores.json' \
    --output ds_sweep/analysis
```

`--manifest` makes each cell reuse that seed's committed split, so every run in
the sweep is comparable to the baseline sample by sample.

## Layout

```
config.py           defaults, deep merge, validation, effective-config capture
data.py             deterministic splits and the split manifest
models.py           QLoRA / full-precision loading, and the bf16 shadow model
gradients.py        parameter selection, layerwise float64 accumulation
scores.py           every score definition (torch-free, unit-tested)
sif.py              self-influence ablation (empirical Fisher via Woodbury)
train.py            fine-tuning, GA+GD, SCRUB, and the method registry
collect.py          CLI: one row per sample -> per_sample_scores.json
offline_eval.py     CLI: metrics, CIs, paired tests, correlations
sweep_report.py     CLI: utility, forgetting, membership and deletion per strength
orientations.py     the preregistered orientation table
stats.py            bootstrap CIs, permutation tests, across-seed comparisons
schema.py           the row format, plus the legacy reader
run_pipeline.py     end-to-end driver
configs/            orientations.yaml + per-setting configs
docs/               DESIGN, ORIENTATIONS, PREREGISTRATION, FUMA, USAGE
scripts/smoke_test       CPU end-to-end check
scripts/sweep_smoke_gate GPU gate the sweep must pass before launching
tests/              unit tests
```

## What each run produces

```
ds_results/
  seed42/
    splits.json                     # the split manifest, read by every later stage
    ga_gd/
      unlearned_model/
      unlearning_run.json           # steps taken, stop reason, final losses
      per_sample_scores.json        # one row per sample + full metadata
  pipeline_summary.json
  evaluation/
    results.json
    report.md
```

`per_sample_scores.json` carries the complete effective config, git commit,
package versions, model descriptions, the orientation table and the FUMA
discrepancy list alongside the rows, so any row can be traced to the settings
that produced it. Full gradient vectors are never stored.
