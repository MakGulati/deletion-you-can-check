# Amendment 1 — 2026-08-09, during stage 1 of the matrix

Two defects in `preregistration_v3_fllm.yaml` were found and fixed **after the
NPO sweep had started and before any result was examined**:

1. **YAML syntax error.** `interpretation: positives : negatives in the audited
   pool` — a bare scalar containing ` : ` parses as a mapping, so the whole
   document failed to load. Now quoted.
2. **Missing top-level keys.** The document declared the axes but not
   `primary_detector`, `primary_task`, `detectors` or `decision_rule`, which
   `deletion_signals.audit` reads directly. Added, restating
   `preregistration_v2.yaml` **by value and unchanged**.

**No threshold, criterion, grid or definition was altered.** Both fixes were
mechanical: one syntactic, one restoring keys that had been documented in prose
but not in the machine-readable section. `tests/test_preregistration_v3.py`
now asserts, for every value, that v3 equals v2 — so this amendment is
verifiable rather than asserted.

The NPO runs in progress at the time were unaffected: they read their
hyperparameters from `configs/arxivmia_npo.yaml`, not from this file, and that
config was unchanged (hash unchanged in `config_hashes.txt` history).

No NPO result had been inspected when the fix was made. The interim audit that
surfaced the error had not yet produced a number.
