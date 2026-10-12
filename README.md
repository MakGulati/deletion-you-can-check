# Deletion You Can Check: Auditing LLM Unlearning

## Public research artifact

This repository contains the implementation, saved evidence, report generators,
and figure code for the study by Mayank Gulati and Gerhard Wunder at Freie
Universität Berlin. It is a public artifact snapshot with its own Git history;
the manuscript sources are maintained separately.

The title is intentionally operational: a deletion claim is *checkable* only
when the evidence supporting it can be inspected against controls with known
ground truth. In this paper, that means checking the following sequence:

1. **Validate the reference.** Exact retraining must read as chance under the
   proposed membership comparison before that comparison can support a deletion
   verdict.
2. **Calibrate ordinary change.** Null refits and routine updates measure how
   much the selected statistics move without a deletion request.
3. **Apply a conjunctive verdict.** Membership, targeted shift, and retained
   utility must hold on the same checkpoint and seed.
4. **Match detector access and cost.** Detectors are compared on the same
   checkpoints, with low-prevalence precision and measured evaluation cost
   reported alongside ROC-AUC.

The artifact makes this audit trail inspectable and reproducible within the
paper's evaluated scope. It does not turn an empirical audit into a formal
certificate of deletion.

## What is included

- `source/`: experiment implementation, configurations, report generators, and
  tests.
- `evidence/`: curated outputs used by the paper, including per-example scores,
  control runs, audit summaries, low-prevalence results, report tables, and
  detector-cost measurements.
- `scripts/make_figures.py`: generator for the four figures used by the FLLM
  manuscript.
- `figures/`: the four current PDF/PNG figures used by the FLLM manuscript.
- `artifacts/figure_data.json`: machine-readable values read by the current
  figure generator.
- `artifacts/PROVENANCE_AND_IMPLEMENTATION.md`: analysis status, configuration
  counts, and implementation details moved from the paper appendix.
- `ARTIFACT_GUIDE.md`: reviewer-oriented map from each audit stage to the
  corresponding code and evidence.
- `ARTIFACT_MANIFEST.md`: inclusion and exclusion details.

## Reproduce the paper figures

From the repository root:

```bash
python scripts/make_figures.py
```

The command regenerates these four figures:

- `fig1_pipeline`: reference-gated audit workflow;
- `fig2_reference`: exact-retraining reference validation and the post-hoc
  same-pool diagnostic;
- `fig3_conjunctive`: per-criterion seed counts and low-prevalence precision;
- `fig4_operational`: matched-access detector comparison and paired leading-pair
  differences.

The script reads only `artifacts/figure_data.json`; it does not download a model,
retrain one, or invent measurements. It requires Python, NumPy, and Matplotlib.

## Recompute and inspect the evidence

Full training, scoring, offline evaluation, and report-generation instructions
are documented under `source/deletion_signals/`. The committed per-example
tables permit CPU-only recomputation of reported metrics. Model checkpoints and
raw dataset records are intentionally excluded; see `ARTIFACT_MANIFEST.md` for
the scope and rationale.
