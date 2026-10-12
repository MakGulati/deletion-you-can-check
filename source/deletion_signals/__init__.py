"""Head-to-head evaluation of deletion-set inference signals.

This package is self-contained: it does not import from any sibling project in
this workspace. It answers one question -- does directional gradient alignment
with the unlearning direction carry information beyond ordinary gradient
magnitude, calibrated difficulty, and the FUMA LoRA ratio?

Layout
------
``config``          effective-configuration loading and recording
``data``            deterministic forget/retain/test splits and split manifests
``models``          QLoRA / full-precision loading, plus the bf16 shadow model
``gradients``       parameter selection and layerwise float64 accumulation
``scores``          every per-sample score definition (pure numerics, no torch)
``sif``             the self-influence ablation (empirical Fisher via Woodbury)
``train``           fine-tuning and the unlearning method registry (GA+GD, SCRUB)
``collect``         CLI: one row per sample -> per_sample_scores.json
``offline_eval``    CLI: metrics, CIs, paired tests, correlations
``orientations``    the preregistered score-orientation table
``stats``           bootstrap CIs, permutation tests, across-seed comparisons
"""

__version__ = "1.0.0"

SCHEMA_VERSION = "deletion-signals/1"
