# Running the FLLM audit

Exact commands, in order, with what each costs. Governed by
[`configs/preregistration_v3_fllm.yaml`](../configs/preregistration_v3_fllm.yaml);
rationale in [PREREGISTRATION_V3_FLLM.md](PREREGISTRATION_V3_FLLM.md).

```bash
PY=python3
```

Nothing below writes to `ds_results/`, `ds_sweep/`, `ds_confirm/` or
`ds_confirm_refine/`. All output goes to `fllm_audit/`.

## 0. Tests and the smoke gate

```bash
$PY -m pytest deletion_signals/tests -q                       # ~2 min, CPU

$PY deletion_signals/scripts/fllm_smoke_gate.py                # ~10 min, GPU
```

The smoke gate runs one 6-step NPO cell and one routine-update condition at 40
scored samples per population, then puts both through the real audit, the real
low-prevalence analysis and the real report generator. It asserts numerical
stability, that the NPO weight starts at 1 and falls, that the forget loss
rises, that only adapters moved, and that the update data are disjoint from
every scored population.

**Nothing it prints is a measurement**, and under the preregistration it may not
be used to choose the grid, select β, or move a threshold.

## 1. NPO sweep — Component A

Five preregistered strengths × three seeds. Each seed uses **its own**
fine-tuned checkpoint; `--manifest` with `{seed}` gives each seed its own split.

```bash
for LR in 1.0e-5 2.0e-5 2.8e-5 5.0e-5 1.0e-4; do
  $PY -m deletion_signals.run_pipeline \
      --config arxivmia_npo.yaml \
      --set unlearning.npo.learning_rate=$LR \
      --seeds 42 43 44 --methods npo \
      --skip-finetune \
      --manifest 'results/seed{seed}/splits.json' \
      --output "fllm_audit/npo/lr_${LR}" \
      2>&1 | tee "fllm_audit/logs/npo_lr_${LR}.log"
done
```

≈ 3.8 h GPU (15 cells × ~15 min: ~2 min unlearning, ~13 min scoring with FUMA
and Fisher-SIF on).

## 2. Routine-update controls — Component D

Conditions 2 and 4. Both receive the identical update; only the base checkpoint
differs.

```bash
for COND in routine_update exact_retrain_routine_update; do
  $PY -m deletion_signals.routine_update \
      --config arxivmia_routine_update.yaml \
      --condition $COND --seeds 42 43 44 \
      --manifest 'results/seed{seed}/splits.json' \
      --exact-retrain-checkpoint 'ds_confirm/controls/exact_retrain/seed{seed}/model' \
      --output fllm_audit/routine_update \
      2>&1 | tee "fllm_audit/logs/${COND}.log"
done
```

≈ 1.5 h GPU. The driver refuses to run if the update records are not disjoint
from forget, retain (scored tail or curvature head), or the scored test prefix.

## 3. Re-score for full detector coverage

Scoring only — no retraining. Gives the central table all 13 score columns at a
uniform 600 scored retain samples.

```bash
$PY deletion_signals/scripts/rescore_full_detectors.py \
    --targets ds_confirm_refine/ascent_2.6e-5 ds_confirm/transition/ascent_5.0e-5 \
    --seeds 42 43 44 --method ga_gd \
    --config arxivmia_confirm.yaml \
    --manifest 'results/seed{seed}/splits.json' \
    --output fllm_audit/rescore \
    2>&1 | tee fllm_audit/logs/rescore.log
```

≈ 1.3 h GPU.

## 4. Detector cost benchmark — Component C

Run this with the sweep **idle**: it is a timing measurement on a shared card.

```bash
# access table only -- no GPU, no models loaded
$PY -m deletion_signals.benchmark --access-only --output fllm_audit/benchmark

# the full benchmark
$PY -m deletion_signals.benchmark \
    --config arxivmia_confirm.yaml \
    --unlearned-checkpoint ds_confirm/controls/exact_retrain/seed42/model \
    --finetuned-checkpoint ds_checkpoints/finetuned_seed42 \
    --manifest results/seed42/splits.json \
    --n-samples 32 --repeats 5 --warmup 2 --scale-to 800 \
    --output fllm_audit/benchmark \
    2>&1 | tee fllm_audit/logs/benchmark.log
```

≈ 0.7 h GPU. Concurrent GPU processes are recorded in the output.

## 5. Audit every run under the frozen criteria

```bash
$PY -m deletion_signals.audit \
    --inputs 'ds_confirm/controls/*/seed*/per_sample_scores.json' \
             'fllm_audit/npo/*/seed*/npo/per_sample_scores.json' \
             'fllm_audit/routine_update/*/seed*/per_sample_scores.json' \
             'fllm_audit/rescore/*/seed*/*/per_sample_scores.json' \
             'ds_confirm/scrub/*/seed*/scrub/per_sample_scores.json' \
    --preregistration deletion_signals/configs/preregistration_v3_fllm.yaml \
    --output fllm_audit/reports
```

The exact-retraining controls must be in `--inputs`: they supply the RNFF
denominator, and without them the selectivity axis is unevaluable for every
method — which counts as **not satisfied**, not as satisfied.

## 6. Low-prevalence analysis — Component B

Offline, CPU only, no model loaded.

```bash
$PY -m deletion_signals.lowprev \
    --inputs 'ds_confirm/controls/*/seed*/per_sample_scores.json' \
             'fllm_audit/npo/*/seed*/npo/per_sample_scores.json' \
             'fllm_audit/routine_update/*/seed*/per_sample_scores.json' \
             'fllm_audit/rescore/*/seed*/*/per_sample_scores.json' \
             'ds_confirm/scrub/*/seed*/scrub/per_sample_scores.json' \
    --n-bootstrap 1000 --resample-check \
    --output fllm_audit/lowprev
```

## 7. Tables, report, and current FLLM figures

```bash
$PY -m deletion_signals.fllm_report \
    --audit fllm_audit/reports/audit_results.json \
    --lowprev fllm_audit/lowprev/lowprev_results.json \
    --benchmark fllm_audit/benchmark/benchmark.json \
    --table3-method exact_retrain \
    --output fllm_audit/reports

# From the repository root; reads artifacts/figure_data.json.
$PY scripts/make_figures.py
```

Produces `fllm_report.md`, `fllm_tables.json` (strict JSON), `failure_log.json`,
and four CSVs under `fllm_audit/reports/`. The second command regenerates the
four current manuscript figures as vector PDF plus PNG under `figures/`.

## Output layout

```
fllm_audit/
  preregistration/   a copy of the frozen YAML and its hash
  smoke/             smoke-gate artifacts (never a measurement)
  npo/lr_*/seed*/npo/
  routine_update/<condition>/seed*/
  rescore/<setting>/seed*/ga_gd/
  benchmark/         benchmark.json, detector_costs.csv
  lowprev/           lowprev_results.json, lowprev_summary.csv
  reports/           audit_results.json, fllm_tables.json, fllm_report.md,
                     failure_log.json, table1..table4 CSVs
  logs/              one log per stage
```

The paper-facing figure data and rendered figures are kept separately at the
repository root under `artifacts/` and `figures/`.

## Reproducibility record

Every output carries `provenance`: git revision and dirty state, the effective
configuration, the Python executable, and the versions of torch, transformers,
peft, datasets, numpy, scipy and scikit-learn. Environment used for the runs:

| | |
|---|---|
| Python | 3.12.13, `python3` |
| torch | 2.11.0+cu128 |
| transformers | 5.14.1 |
| peft | 0.20.0 |
| datasets | 5.0.1 |
| numpy / scipy / scikit-learn | 2.5.1 / 1.18.0 / 1.9.0 |
| GPU | NVIDIA RTX A5000, 24 GB |

## Total cost

| stage | GPU |
|---|---|
| smoke gate | ~0.2 h |
| NPO sweep | ~3.8 h |
| routine-update controls | ~1.5 h |
| re-score | ~1.3 h |
| benchmark | ~0.7 h |
| audit, low-prevalence, tables, figures | CPU only |
| **total** | **~7.5 h** |

Storage: ~1.5 GB (21 LoRA adapters, ~27 score files, figures and reports).
