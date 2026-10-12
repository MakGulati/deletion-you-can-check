# Artifact Manifest

## Artifact identity

This snapshot supports **Deletion You Can Check: Auditing LLM Unlearning**. It
repackages the same audited experiments and verdicts in the FLLM paper's
reference-gated, three-stage presentation. No measurement, threshold, seed, or
method verdict is changed by this presentation update.

## Included

- Experiment source, configurations, report generators, and tests from the code
  revision audited for the manuscript.
- All 126 curated `per_sample_scores.json` files from the reported factorial,
  control, method, transfer, and refinement runs.
- Audit results, low-prevalence outputs, control summaries, method-verdict
  tables, detector-cost measurements, failure logs, and preregistration records.
- Four FLLM paper figures and the script that regenerates them from the
  machine-readable figure record.

Per-example score rows contain sample IDs, truncated record fingerprints,
population labels, token counts, metrics, and diagnostic flags. They do not
contain raw record text.

## Excluded

- Model checkpoints, adapter weights, and duplicated tokenizer assets.
- Raw dataset records and split manifests containing source text.
- Local logs, caches, editor files, build intermediates, and private planning or
  reviewer-response notes.
- Original development Git history, remotes, machine paths, credentials, and
  source commit hashes not needed for reproducing the reported results.
- Manuscript sources, bibliography, compiled paper, and LaTeX build files.

These exclusions do not affect offline reproduction of the reported tables and
figures. Re-running model training requires obtaining the public datasets and
base models identified in the source configurations.

## Public release preparation

This public release has a fresh history under the authors' identity. Its source
and evidence match the reviewed artifact, with the corrected Figure 3 gate
presentation. Raw records, model weights, local paths, and credentials remain
excluded.
