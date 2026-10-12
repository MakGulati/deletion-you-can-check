# Provenance and implementation details

This document holds the details moved out of the eight-page FLLM manuscript.
The reported figure and table values come from the audited experiment revision
identified in `figure_data.json`. The public artifact also includes source
configurations, per-sample scores, reports, and preregistration records.

## Analysis breadth and status

- ArxivMIA/Llama uses seeds 42–44 for seven GA+GD, five NPO-RT, and four
  SCRUB configurations. GA+GD and SCRUB are confirmatory; NPO-RT is
  preregistered.
- BookMIA/Qwen tests two GA+GD, one NPO-RT, and one exploratory SCRUB
  configuration. The narrow GA+GD refinement and the matched-population
  diagnostic are post-hoc.
- The audit criteria and thresholds were fixed before confirmatory runs and
  reused across datasets. Full configuration grids and checkpoint results
  are in the artifact's `source/`, `evidence/`, and report files.

## Training and evaluation settings

- Seeds 42–44 fix shuffled data splits.
- QLoRA uses NF4 quantization, rank 16, scaling 32, dropout 0.05, and
  query/key/value/output projections. Fine-tuning and exact retraining use
  six epochs, learning rate 2 × 10⁻⁴, and effective batch size 32.
- Exact retraining omits the 200 forget records and trains on the 800 retain
  records. Logged full fine-tuning and exact-retraining runs take 186 and
  150 optimizer steps, respectively; they are epoch-matched, not step-matched.
- NPO-RT uses β = 0.1, retain weight 1, and sequence-sum log ratios.
- Runs use one RTX A5000 24 GB GPU, Python 3.12.13, PyTorch 2.11.0+cu128,
  Transformers 5.14.1, PEFT 0.20.0, and Datasets 5.0.1.

The source configurations and scripts, rather than this summary, are the
operative specification for rerunning an experiment.
