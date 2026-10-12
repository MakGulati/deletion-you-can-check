# Deletion-set inference signals

Does directional gradient alignment with the unlearning direction carry
information beyond ordinary gradient magnitude, difficulty-calibrated loss, and
the FUMA LoRA ratio?

The task is **forget vs retain**: both classes are known training members, and
the attack must determine which of them were targeted for unlearning. That is
deletion-set inference, not membership inference, and the distinction changes
which baselines are meaningful.

Every score's direction on every task is fixed **before** measurement in
[`deletion_signals/configs/orientations.yaml`](deletion_signals/configs/orientations.yaml),
each with a written rationale. `max(AUC, 1 - AUC)` is not used anywhere: a score
that lands below 0.5 under its declared direction is reported below 0.5.

Full documentation is in
[`deletion_signals/README.md`](deletion_signals/README.md) and
[`deletion_signals/docs/`](deletion_signals/docs/).

## Status: preliminary

Three seeds, one model (Llama-3.2-1B, QLoRA), one dataset (ArxivMIA), two
unlearning methods. **The unlearning in these runs is weak** -- no GA+GD seed
reached its 2.0 forget/retain loss-ratio target, and SCRUB's forget KL stayed
near 0.09. Conclusions below are conditional on that regime; an
unlearning-strength sweep is the next experiment and could change them.

## What the three seeds show

Forget vs retain, mean ± SD of the per-seed AUC, preregistered orientations:

| score | GA+GD | SCRUB |
|---|---|---|
| `raw_cosine` (signed) | **0.982 ± 0.006** | 0.586 ± 0.014 |
| `combined_log_score` | 0.857 ± 0.020 | 0.615 ± 0.017 |
| `grad_ratio` (calibrated magnitude) | 0.851 ± 0.006 | 0.583 ± 0.009 |
| `alignment_cos_sq` | 0.841 ± 0.023 | 0.586 ± 0.014 |
| `loss_ratio` (calibrated loss) | 0.676 ± 0.017 | 0.560 ± 0.018 |
| `loss_unlearned` | 0.548 ± 0.022 | 0.514 ± 0.021 |
| `grad_norm_sq` | 0.542 ± 0.025 | 0.513 ± 0.018 |
| `fuma_lora_ratio` | 0.528 ± 0.024 | 0.526 ± 0.023 |
| `sif` | 0.541 ± 0.025 | 0.508 ± 0.018 |

Four findings, all with the caveat above:

1. **Squared alignment does not beat calibrated gradient magnitude.** Paired
   across matched seeds, `alignment_cos_sq - grad_ratio = -0.010 ± 0.018`
   (p = 0.44). Dividing a sample's gradient norm by its norm under the
   pre-unlearning checkpoint is as informative as the cosine, needs the same
   access, and is cheaper.

2. **The sign is where the deletion-specific information lives.** Signed cosine
   reaches 0.982 ± 0.006 under GA+GD and beats `grad_ratio` by
   +0.131 ± 0.006. Under GA+GD only 36.7% of retain gradients have positive
   cosine with ĝ against 100% of forget gradients -- retain received gradient
   *descent*, so its gradients point against the update. Squaring discards that.
   Under SCRUB every cosine is positive, so `raw_cosine` and `alignment_cos_sq`
   are identical and neither beats its baselines.

3. **Self-influence is inert.** Spearman ρ(`sif`, `grad_norm_sq`) = 0.9999
   (GA+GD) / 0.9888 (SCRUB). At rank 200 against d = 3.4M the empirical Fisher
   resolves nothing, so SIF is a rescaled gradient norm.

4. **Forget-vs-unseen is a membership measurement, not a deletion one.**
   Difficulty-calibrated loss separates forget from unseen at ~0.999 when read
   in the membership direction, and separates *retain* from unseen equally well
   (0.9998). Forget samples in these runs remain full-strength training members.

`raw_cosine` is deliberately marked `headline: false` in the orientation table:
its direction depends on the ĝ sign convention rather than on the score's
semantics. Its declared direction held, but it was promoted to prominence after
the results were seen, so it needs confirmation on a setting that did not select
it.

## Layout

```
deletion_signals/     the package: collection, scoring, offline evaluation, tests
results/
  seed4{2,3,4}/
    splits.json                       the split manifest for that seed
    {ga_gd,scrub}/
      per_sample_scores.json          1800 rows, every score + full metadata
      unlearning_run.json             steps taken, stop reason, final losses
  evaluation/{report.md,results.json} the three-seed offline evaluation
  pipeline_summary.json
```

Model weights are not committed. The score tables are what make the numbers
checkable: every metric in `results/evaluation/` is an offline recompute over
them, no GPU required.

## Reproducing

```bash
PY=/path/to/python   # needs torch, transformers, peft, datasets, scikit-learn, scipy

# checks, no GPU
$PY -m pytest deletion_signals -c deletion_signals/pytest.ini    # 198 tests, ~90 s
$PY deletion_signals/scripts/smoke_test.py                       # end to end, ~1 min

# recompute every published metric from the committed tables, no GPU
$PY -m deletion_signals.offline_eval \
    --inputs 'results/seed*/*/per_sample_scores.json' \
    --output /tmp/recomputed

# regenerate the tables from scratch (~2.2 h on an RTX A5000)
$PY -m deletion_signals.run_pipeline \
    --config arxivmia_lora.yaml --seeds 42 43 44 --methods ga_gd scrub \
    --output ds_results
```

Each `per_sample_scores.json` embeds the complete effective config, the git
commit, package versions, model descriptions, the orientation table and the FUMA
discrepancy list, so any row can be traced to the settings that produced it.

## Known limitations

- Weak unlearning in every run (see Status).
- One model, one dataset, three seeds. The unseen population is the dataset's
  full non-member set, so it is identical across seeds; across-seed SDs for
  tasks involving it understate uncertainty. Forget-vs-retain is unaffected --
  forget sets overlap only ~18% between seeds.
- The published FUMA formula could not be verified from the working
  environment. The convention used is stated explicitly and is configurable;
  see [`deletion_signals/docs/FUMA.md`](deletion_signals/docs/FUMA.md).
- The alignment score assumes white-box access to both the fine-tuned
  pre-unlearning and the unlearned checkpoints.
- n = 3 seeds cannot support a distribution-free paired test; the code uses a
  paired t-test below six seeds and says so in its output.
