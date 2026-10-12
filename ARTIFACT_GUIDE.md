# Reviewer Guide: What Makes Deletion Checkable?

The artifact follows the paper's audit order. Reviewers can stop at any failed
stage: later statistics cannot repair an invalid reference comparison.

## 1. Validate the reference gate

**Question.** Does exact retraining, whose deletion ground truth is known,
produce chance-level membership separation under the proposed comparison?

**Inspect.**

- `evidence/ds_factorial/FACTORIAL_RESULTS.md`
- `evidence/ds_factorial/*/controls/exact_retrain/`
- `evidence/ds_bookmia/membership_null/membership_null_check.json`
- `source/deletion_signals/scripts/membership_null_check.py`
- `figures/fig2_reference.pdf`

The BookMIA/Qwen same-pool comparison is a post-hoc diagnostic of the rejected
axis, not a retroactive confirmatory result.

## 2. Calibrate non-deletion change

**Question.** How far can loss-based statistics move because of refitting
randomness or ordinary maintenance rather than deletion?

**Inspect.**

- `evidence/ds_confirm/controls/null_refit/`
- `evidence/fllm_audit/routine_update/`
- `source/deletion_signals/controls.py`
- `source/deletion_signals/routine_update.py`
- `evidence/fllm_audit/reports/table1_control_calibration.csv`

These controls define an empirical non-deletion scale; they do not establish a
universal bound for every future training pipeline.

## 3. Apply the conjunctive checkpoint verdict

**Question.** On the same seed and checkpoint, do membership removal, targeted
shift toward exact retraining, and utility preservation all hold?

**Inspect.**

- `evidence/fllm_audit/reports/audit_results.json`
- `evidence/fllm_audit/reports/table2_method_verdicts.csv`
- `evidence/fllm_audit/reports/failure_log.json`
- `source/deletion_signals/audit.py`
- `source/deletion_signals/audit_report.py`
- `figures/fig3_conjunctive.pdf`

The scoped finding is that none of the tested GA+GD, NPO-RT, or SCRUB
configurations passes all criteria. It is not a universal failure claim about
these method families.

The original BookMIA membership comparison fails the reference gate. Raw
pre-gate pass counts remain in saved JSON and CSV files for inspection, but
Figure 3 marks BookMIA membership and conjunctive verdicts invalid. Those raw
counts must not be interpreted as accepted deletion verdicts.

## 4. Compare detectors under matched conditions

**Question.** When checkpoint access is held fixed, which detector group leads,
and what operational cost accompanies it?

**Inspect.**

- `evidence/fllm_audit/benchmark/benchmark.json`
- `evidence/fllm_audit/benchmark/detector_costs.csv`
- `evidence/fllm_audit/lowprev/lowprev_results.json`
- `source/deletion_signals/benchmark.py`
- `source/deletion_signals/lowprev.py`
- `figures/fig4_operational.pdf`

Two-checkpoint loss change and directional alignment form the leading group in
the tested cells. Their small, sign-changing difference does not establish a
universal winner or statistical equivalence; operational cost therefore informs
the representative choice.

## 5. Regenerate the presentation

```bash
python scripts/make_figures.py
```

This step verifies the deterministic mapping from
`artifacts/figure_data.json` to the four manuscript figures. For recomputation
from per-example scores, use the offline evaluation and report commands under
`source/deletion_signals/`.
