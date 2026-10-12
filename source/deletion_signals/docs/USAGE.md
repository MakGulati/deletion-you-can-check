# Usage

```bash
PY=python3
```

`py311` has a `cu130` torch build that cannot initialise against this machine's
driver, and bare `python` resolves to it. Verify before any long run:

```bash
$PY -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# expected: 2.11.0+cu128 True
```

## Checks that need no GPU

```bash
$PY -m pytest deletion_signals -c deletion_signals/pytest.ini   # ~90 s
$PY deletion_signals/scripts/smoke_test.py                      # ~1 min
```

The smoke test runs the whole machinery on pythia-160m: LoRA attachment, both
unlearning methods, the collector with every score enabled, the score file, and
the offline evaluator. Its numbers are meaningless by construction (3 vs 3
samples); it checks that each stage executes and that the invariants hold.

## Collecting scores

### End to end

```bash
$PY -m deletion_signals.run_pipeline \
    --config llama_lora.yaml --seeds 42 43 44 --methods ga_gd scrub \
    --output ds_results
```

Stages are individually skippable and each writes a checkpoint:

```bash
--skip-finetune    # reuse ds_checkpoints/finetuned_seed<N> if present
--skip-unlearn     # reuse <output>/seed<N>/<method>/unlearned_model if present
--skip-collect     # train only
--max-samples 32   # cap scored samples per population
--manifest results/seed{seed}/splits.json   # reuse an existing split
```

`--manifest` makes the run share an earlier run's exact forget/retain/test
partition, so the two are comparable sample by sample rather than merely in
distribution. Unlearning and scoring both read it, and a copy is written into
the run directory so the run stays self-describing. `{seed}` is substituted per
seed; without the placeholder the same file is used for every seed, which
`load_or_build_splits` rejects for all but the seed it was written for — scoring
one seed's checkpoints against another seed's split mislabels every row.

### Capping populations independently

```yaml
scoring:
  max_samples_per_population: {test: 300}   # int and null still work
```

Shrinks one population and leaves the others whole. **Do not use
`data.test_samples` for this**: `build_splits` draws the test set from the same
RNG stream *before* it shuffles the members, so changing the test count also
reshuffles which members land in forget and retain, and the run stops sharing a
split with its siblings.

Checkpoint paths are seed-scoped on purpose. A seed-independent path makes every
seed reuse the first one's fine-tuned model: the split still varies, so the runs
are not identical, but fine-tuning randomness is never resampled and across-seed
intervals come out too narrow.

### Scoring checkpoints produced elsewhere

The collector does not care how the unlearned checkpoint was made. Give it the
two checkpoints and a split manifest:

```bash
$PY -m deletion_signals.collect \
    --config llama_lora.yaml --seed 42 --method my_method \
    --unlearned-checkpoint /path/to/unlearned_model \
    --finetuned-checkpoint /path/to/finetuned_seed42 \
    --manifest ds_results/seed42/splits.json \
    --output ds_results/seed42/my_method/per_sample_scores.json
```

Both checkpoints are required. Without θ_finetuned there is no
ĝ = θ_unlearned − θ_finetuned and therefore no alignment score, which is the
point of the comparison; the collector refuses rather than emitting rows with
that column empty.

### Full fine-tuning

```bash
$PY -m deletion_signals.run_pipeline \
    --config llama_full_ft.yaml --seeds 42 --methods ga_gd --output ds_results_full
```

FUMA is not applicable there and rows record the reason; see
[FUMA.md](FUMA.md).

### Turning the self-influence ablation on

```bash
$PY -m deletion_signals.run_pipeline --config llama_lora.yaml \
    --set scores.sif.enabled=true --set scores.sif.method=fisher \
    --seeds 42 --methods ga_gd --output ds_results
```

The Fisher estimator holds out `scores.sif.fisher_rank` retain samples to build
the curvature estimate and **excludes them from scoring**. Overlap between the
curvature set and the scored set inflates the retain-vs-unseen AUC — those
samples lie in the row space of G and are damped hardest —
so `FisherSelfInfluence.assert_disjoint` raises on it rather than letting it
through.

## Recomputing metrics offline

```bash
$PY -m deletion_signals.offline_eval \
    --inputs 'ds_results/seed*/*/per_sample_scores.json' \
    --output ds_results/evaluation
```

Writes `results.json` (everything) and `report.md` (readable). No model is
loaded; changing an FPR target, adding a task or correcting a test is seconds of
CPU.

Useful flags:

```bash
--n-bootstrap 5000              # more bootstrap replicates
--orientations /path/to.yaml    # an alternative orientation table
--quiet                         # do not print the report
```

## Reporting an unlearning-strength sweep

```bash
$PY -m deletion_signals.sweep_report \
    --inputs 'ds_sweep/ascent_*/seed*/ga_gd/per_sample_scores.json' \
    --output ds_sweep/analysis
```

Writes `sweep_results.json`, `sweep_report.md` and `sweep.png`. It answers a
different question from `offline_eval`: not "which score separates best" but
"how much utility survived, how much forgetting happened, is the membership
evidence gone, and is the deletion signal still there" — at every strength, so a
deletion result can be placed against the other three rather than read alone.

The thresholds are preregistered in [PREREGISTRATION.md](PREREGISTRATION.md) and
mirrored as module constants; the report applies them and states which criteria
each setting met. The unit of replication is the seed, and a setting qualifies
only if every seed qualifies.

Each run's strength is read from its recorded hyperparameters, never from its
directory name — a path is a label someone typed, and a mislabelled cell would
turn a transposition into a monotone curve.

Useful flags: `--n-bootstrap`, `--no-plot`, `--quiet`.

### Legacy files from the sibling pipeline

```bash
$PY -m deletion_signals.offline_eval --legacy \
    --inputs 'unlearning_paradox_sanity/results_seed4*/*/per_sample_components.json' \
    --output ds_results/legacy_evaluation \
    --legacy-dataset swj0419/WikiMIA --legacy-model meta-llama/Llama-3.2-1B
```

Degraded mode. Those files carry only `alignment_cos_sq`, `grad_norm_sq`, `sif`
and `combined_log_score`; every other score is reported **absent**, not zero.
They have no per-sample identifiers, so rows are matched positionally within a
population and cross-seed pairing is by population only. Seed and method are
inferred from the directory path; pass `--legacy-seed` when the path does not
encode one.

## Adding an unlearning method

```python
from deletion_signals.train import register_unlearning_method

def my_method(model, tokenizer, splits, config, save_path) -> dict:
    ...
    model.save_pretrained(save_path)
    return {
        "method": "my_method",
        "hyperparameters": dict(config["unlearning"]["my_method"]),
        "steps_taken": steps,
        "stop_reason": "completed",
        "save_path": save_path,
    }

register_unlearning_method("my_method", my_method)
```

`--method my_method` then works throughout. The returned dict is stored verbatim
in the collection metadata as the run's hyperparameters and outcome, so put in
it whatever a reader needs to know about how the run actually ended — the
configured step count is not always what the model received.

Nothing downstream changes. The orientation table is defined per score, not per
method, and the collector only ever sees a checkpoint directory. Where a
score's direction is method-dependent — `grad_norm_sq` is the real case — the
assumption and its reasoning are already stated in
[ORIENTATIONS.md](ORIENTATIONS.md) and apply as written.

## Adding a score

See the checklist at the end of [ORIENTATIONS.md](ORIENTATIONS.md). In short:
add the column to `schema.SCORE_COLUMNS` and `ScoreRow`, compute it in
`collect.ScoreCollector.score_sample`, and declare a direction plus a rationale
for every task. The orientation-table test fails until the last step is done.

## Configuration

Every knob has a default in `config.DEFAULTS`; a config file states only what it
changes, and the merged *effective* config is recorded in every output file.

Override anything from the command line:

```bash
--set scoring.param_scope=lora
--set scoring.param_name_patterns='["*layers.0.*", "*layers.1.*"]'
--set scores.fuma.norm=l2_sq
--set evaluation.n_bootstrap=5000
```

`config.validate_config` rejects combinations that could not be computed as
written — `grad_calibrated.reference: pretrained_base` in LoRA mode, `fisher`
self-influence under full fine-tuning, `param_scope: lora` without adapters —
rather than letting them fail hours into a run or, worse, produce a
plausible-looking number from the wrong thing.
