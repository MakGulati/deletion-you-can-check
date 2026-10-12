"""The v3 preregistration parses, loads, and restates v2 without loosening it.

Two failures this file exists to prevent, both of which happened:

* **A YAML syntax error.** A bare scalar containing `` : `` -- "positives :
  negatives in the audited pool" -- reads as a mapping, and the whole document
  fails to parse. Nothing catches that until the audit runs, which in a long
  matrix is hours after the runs it is supposed to judge.
* **A missing top-level key.** ``deletion_signals.audit`` reads
  ``primary_detector``, ``primary_task``, ``detectors`` and ``decision_rule``
  directly. A preregistration that declares the axes but omits those either
  crashes the audit or, worse, leaves a run judged by a different document than
  the one that governs it.

The second test is the one that matters most: **every threshold v3 restates must
equal v2's**. v3 is allowed to add and forbidden to loosen, and an accidental
edit to a restated number would silently change a verdict.
"""

from __future__ import annotations

import os

import pytest
import yaml

from deletion_signals.audit import Preregistration

CONFIG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs"
)
V2_PATH = os.path.join(CONFIG_DIR, "preregistration_v2.yaml")
V3_PATH = os.path.join(CONFIG_DIR, "preregistration_v3_fllm.yaml")


@pytest.fixture(scope="module")
def v2() -> Preregistration:
    return Preregistration.load(V2_PATH)


@pytest.fixture(scope="module")
def v3() -> Preregistration:
    return Preregistration.load(V3_PATH)


def test_the_v3_preregistration_is_valid_yaml():
    with open(V3_PATH) as handle:
        document = yaml.safe_load(handle)
    assert isinstance(document, dict)
    assert document["document_version"] == "3-fllm"


def test_v3_loads_through_the_audits_own_loader(v3):
    """Not just parseable -- loadable by the code that will read it."""
    assert v3.path == V3_PATH


def test_v3_declares_every_key_the_audit_reads(v3):
    """Each of these is dereferenced directly by audit.py; a missing one is a
    KeyError hours into a run, or a silently different set of thresholds."""
    for key in ("primary_detector", "primary_task", "detectors", "decision_rule",
                "axes", "seeds", "replication_unit"):
        assert key in v3.raw, f"v3 does not declare {key!r}"

    # Exercise the accessors themselves, not just the raw keys.
    assert v3.primary_detector
    assert v3.primary_task
    assert v3.membership["equivalence_region"]
    assert v3.utility["max_perplexity_ratio"]
    assert v3.selectivity["min_forget"] is not None
    assert v3.deletion["min_auc"]


def test_v3_restates_every_v2_threshold_unchanged(v2, v3):
    """v3 may ADD. It may not loosen, tighten or drift."""
    assert v3.primary_detector == v2.primary_detector
    assert v3.primary_task == v2.primary_task
    assert v3.membership["equivalence_region"] == v2.membership["equivalence_region"]
    assert v3.membership["score"] == v2.membership["score"]
    assert v3.membership["task"] == v2.membership["task"]
    assert v3.utility["max_perplexity_ratio"] == v2.utility["max_perplexity_ratio"]
    assert v3.utility["populations"] == v2.utility["populations"]
    assert v3.selectivity["min_forget"] == v2.selectivity["min_forget"]
    assert v3.selectivity["max_retain"] == v2.selectivity["max_retain"]
    assert v3.deletion == v2.deletion


def test_v3_declares_the_same_detector_access_classes_as_v2(v2, v3):
    assert set(v3.detectors) == set(v2.detectors)
    for detector in v2.detectors:
        assert v3.access_of(detector) == v2.access_of(detector), detector


def test_v3_does_not_modify_v2(v2):
    """v2 is immutable under this extension. If someone edited it to make v3
    agree, this is the test that says so."""
    assert v2.raw["schema_version"] == 2
    assert v2.raw["output_root"] == "ds_confirm"
    assert "document_version" not in v2.raw


def test_the_matched_access_rule_still_holds_for_the_primary_detector(v3):
    """The primary detector must be beatable only by something that already
    holds what it holds -- the premise of the whole comparison."""
    assert v3.weaker_or_equal_access("raw_cosine", v3.primary_detector)
    assert v3.weaker_or_equal_access("grad_ratio", v3.primary_detector)
    # grad_norm_sq never reads the pre-deletion checkpoint.
    assert not v3.weaker_or_equal_access("grad_norm_sq", v3.primary_detector)


def test_the_extension_sections_are_present_and_populated(v3):
    """The four components' frozen parameters. A run governed by v3 reads its
    NPO grid and routine-update schedule from here."""
    npo = v3.raw["component_a_npo"]
    assert npo["fixed_hyperparameters"]["beta"] == 0.1
    assert npo["fixed_hyperparameters"]["log_ratio_reduction"] == "sum"
    assert npo["strength_grid"]["values"] == [1.0e-5, 2.0e-5, 2.8e-5, 5.0e-5, 1.0e-4]
    assert npo["variant_selected"] == "NPO-RT"

    update = v3.raw["component_d_routine_update"]["update_definition"]
    assert update["learning_rate"] == 5.0e-5
    assert update["epochs"] == 3
    assert v3.raw["component_d_routine_update"]["update_data"]["source"] == (
        "the scored-test prefix complement -- test[300:500], 200 records"
    )

    lowprev = v3.raw["component_b_low_prevalence"]
    assert lowprev["prevalence_simulation"]["levels"] == ["1:1", "1:10", "1:100"]
    assert lowprev["prevalence_simulation"]["method"] == "analytic"


def test_the_frozen_defaults_in_code_match_the_preregistration(v3):
    """config.DEFAULTS and the preregistration must not drift apart: the code
    reads the former and a verdict rests on the latter."""
    from deletion_signals.config import DEFAULTS

    npo = v3.raw["component_a_npo"]["fixed_hyperparameters"]
    assert DEFAULTS["unlearning"]["npo"]["beta"] == npo["beta"]
    assert DEFAULTS["unlearning"]["npo"]["retain_weight"] == npo["retain_weight"]
    assert DEFAULTS["unlearning"]["npo"]["log_ratio_reduction"] == npo["log_ratio_reduction"]
    assert DEFAULTS["unlearning"]["npo"]["steps"] == npo["steps"]

    update = v3.raw["component_d_routine_update"]["update_definition"]
    assert DEFAULTS["routine_update"]["learning_rate"] == update["learning_rate"]
    assert DEFAULTS["routine_update"]["epochs"] == update["epochs"]


def test_the_six_fllm_classifications_are_declared_in_evaluation_order(v3):
    from deletion_signals.fllm_report import CLASSIFICATION_ORDER

    declared = v3.raw["fllm_classifications"]["order"]
    assert tuple(declared) == CLASSIFICATION_ORDER
    for label in declared:
        assert label in v3.raw["fllm_classifications"], f"{label} has no definition"


def test_v3_keeps_v2s_classification_vocabulary_for_the_shared_audit(v2, v3):
    """`deletion_signals.audit` dereferences ``classifications[label]`` for the
    label IT computes, which is v2's vocabulary. Replacing that block with the
    extension's six labels does not rename anything -- it makes the shared audit
    raise KeyError on every run this document governs."""
    assert set(v3.raw["classifications"]) == set(v2.raw["classifications"])
    for label, spec in v2.raw["classifications"].items():
        assert v3.raw["classifications"][label]["statement"].split() == (
            spec["statement"].split()
        ), f"the statement for {label!r} drifted from v2"


def test_every_label_the_audit_can_emit_has_a_statement_in_v3(v3):
    """Enumerated from audit.classify's own branches, so a new label there
    without a definition here fails at test time rather than mid-matrix."""
    for label in ("persistent_fingerprint", "successful_deletion_no_fingerprint",
                  "global_destruction", "incomplete_forgetting", "inconclusive"):
        assert label in v3.raw["classifications"], label
        assert v3.raw["classifications"][label]["statement"].strip()


def test_the_two_vocabularies_are_kept_separate(v3):
    """They answer different questions -- the audit's includes detectability,
    the extension's includes the collapse guard -- so neither may be silently
    substituted for the other."""
    assert "failed_run" not in v3.raw["classifications"]
    assert "persistent_fingerprint" not in v3.raw["fllm_classifications"]
