"""The preregistered orientation table, and the rule that it is never bypassed."""

from __future__ import annotations

import os
import textwrap

import pytest

from deletion_signals.orientations import (
    HIGHER_IS_POSITIVE,
    LOWER_IS_POSITIVE,
    OrientationTable,
    build_task_arrays,
)
from deletion_signals.schema import SCORE_COLUMNS
from deletion_signals.stats import roc_auc


@pytest.fixture(scope="module")
def table() -> OrientationTable:
    return OrientationTable.load()


# ------------------------------------------------------- the shipped table


def test_shipped_table_loads(table):
    assert table.task_names
    assert "forget_vs_retain" in table.task_names


def test_every_score_column_has_a_declared_orientation_on_every_task(table):
    """A score with no declared direction cannot be evaluated at all, so the
    shipped table must cover the whole schema."""
    missing = [
        (score, task)
        for score in SCORE_COLUMNS
        for task in table.task_names
        if (score, task) not in table._orientations  # noqa: SLF001 - the point of the test
    ]
    assert missing == []


def test_every_declared_orientation_carries_a_rationale(table):
    for score in table.score_names:
        for task in table.task_names:
            assert table.orientation(score, task).rationale


def test_primary_task_is_forget_versus_retain(table):
    task = table.task("forget_vs_retain")
    assert task.positive == ("forget",)
    assert task.negative == ("retain",)
    assert task.role == "primary"


def test_member_task_pools_forget_and_retain_against_unseen(table):
    task = table.task("member_vs_unseen")
    assert set(task.positive) == {"forget", "retain"}
    assert task.negative == ("test",)


def test_raw_cosine_is_preregistered_as_a_headline_score(table):
    """Inverted deliberately: this test asserted the opposite until the signed
    cosine was promoted for the GA+GD unlearning-strength sweep.

    The promotion happened AFTER the score was seen to win on three seeds, which
    is why the promotion, its date and that conflict are written into the change
    log at the top of the table -- and why the sweep, not those three seeds, is
    what confirms it."""
    assert table.orientation("raw_cosine", "forget_vs_retain").headline is True
    assert "raw_cosine" in table.headline_scores("forget_vs_retain")


def test_the_raw_cosine_rationale_states_the_sign_convention_and_its_scope(table):
    """A promoted score whose direction rests on a convention has to say which
    convention, and where the argument stops applying."""
    rationale = table.orientation("raw_cosine", "forget_vs_retain").rationale
    assert "theta_unlearned - theta_finetuned" in rationale
    assert "ASCENT" in rationale and "DESCENT" in rationale
    assert "SCRUB" in rationale


def test_raw_cosine_declares_one_direction_on_every_task(table):
    """No task-by-task choice was available after the fact."""
    directions = {
        table.orientation("raw_cosine", task).direction for task in table.task_names
    }
    assert directions == {HIGHER_IS_POSITIVE}


def test_the_change_log_records_that_raw_cosine_was_promoted_after_winning(table):
    """The preregistration is only honest if its amendments are visible."""
    with open(table.source_path) as handle:
        text = handle.read()
    assert "CHANGE LOG" in text
    assert "PROMOTED AFTER BEING SEEN TO WIN" in text


@pytest.mark.parametrize(
    "score", ["loss_ratio_vs_finetuned", "loss_delta_vs_finetuned"]
)
def test_the_unlearning_attributed_loss_scores_are_fully_declared(table, score):
    """Added for the sweep; the loader rejects a partial entry, so this pins the
    directions themselves rather than merely their presence."""
    assert table.orientation(score, "forget_vs_retain").direction == HIGHER_IS_POSITIVE
    assert table.orientation(score, "forget_vs_unseen").direction == HIGHER_IS_POSITIVE
    assert table.orientation(score, "retain_vs_unseen").direction == LOWER_IS_POSITIVE
    assert table.orientation(score, "member_vs_unseen").direction == LOWER_IS_POSITIVE


def test_the_two_loss_families_are_separate_entries(table):
    """loss_ratio measures membership and loss_ratio_vs_finetuned measures what
    unlearning did. Their retain_vs_unseen rationales must not be the same
    argument wearing two names."""
    membership = table.orientation("loss_ratio", "retain_vs_unseen").rationale
    unlearning = table.orientation("loss_ratio_vs_finetuned", "retain_vs_unseen").rationale
    assert membership != unlearning
    assert "member" in membership.lower()
    assert "descent" in unlearning.lower()


def test_loss_flips_direction_between_the_two_primary_tasks(table):
    """Deletion-set inference and membership inference want opposite thresholds;
    a table that declared one direction for both would be wrong on one of them."""
    assert table.orientation("loss_unlearned", "forget_vs_retain").direction == HIGHER_IS_POSITIVE
    assert table.orientation("loss_unlearned", "member_vs_unseen").direction == LOWER_IS_POSITIVE


def test_sif_is_declared_lower_for_members(table):
    """The repository's own measured direction, declared rather than discovered."""
    assert table.orientation("sif", "retain_vs_unseen").direction == LOWER_IS_POSITIVE


def test_method_dependent_directions_say_so_in_their_rationale(table):
    rationale = table.orientation("grad_norm_sq", "forget_vs_retain").rationale
    assert "METHOD-DEPENDENT" in rationale
    assert "SCRUB" in rationale


# ----------------------------------------------------------- orientation logic


def test_sign_is_plus_one_for_higher_is_positive(table):
    assert table.sign("alignment_cos_sq", "forget_vs_retain") == 1.0


def test_sign_is_minus_one_for_lower_is_positive(table):
    assert table.sign("loss_unlearned", "member_vs_unseen") == -1.0


def test_orient_negates_a_lower_is_positive_score(table):
    assert table.orient([1.0, 2.0], "loss_unlearned", "member_vs_unseen") == [-1.0, -2.0]


def test_orientation_makes_an_inverted_score_score_below_half(table):
    """The declared direction is applied even when it produces a bad AUC. This
    is the behaviour post-hoc max(AUC, 1-AUC) would destroy."""
    # A score that is LOWER for the positive class, declared higher_is_positive.
    values = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    labels = [1, 1, 1, 0, 0, 0]
    oriented = table.orient(values, "alignment_cos_sq", "forget_vs_retain")
    auc = roc_auc(oriented, labels)
    assert auc == pytest.approx(0.0)
    assert auc < 0.5  # reported as-is, never flipped to 1.0


def test_a_correctly_oriented_lower_is_positive_score_scores_above_half(table):
    values = [0.1, 0.2, 0.3, 0.8, 0.9, 1.0]   # members have LOWER loss
    labels = [1, 1, 1, 0, 0, 0]
    auc = roc_auc(table.orient(values, "loss_unlearned", "member_vs_unseen"), labels)
    assert auc == pytest.approx(1.0)


def test_unknown_score_raises_instead_of_defaulting(table):
    with pytest.raises(KeyError, match="preregistered orientation"):
        table.orientation("a_score_nobody_declared", "forget_vs_retain")


def test_unknown_task_raises(table):
    with pytest.raises(KeyError, match="Unknown task"):
        table.task("no_such_task")


def test_orientation_table_exposes_no_flipping_helper():
    """Guard against a max(AUC, 1-AUC) convenience creeping back in.

    The behavioural proof is the test above; this one stops the API from
    growing an affordance for the thing that test forbids.
    """
    from deletion_signals import orientations as module

    public = [name for name in dir(OrientationTable) if not name.startswith("_")]
    assert not any("flip" in name or "best" in name for name in public)
    assert not any(
        "flip" in name or "oriented_best" in name
        for name in dir(module)
        if not name.startswith("_")
    )


# ------------------------------------------------------------- array building


def test_build_task_arrays_selects_only_the_task_populations(table):
    rows = [
        {"population": "forget", "alignment_cos_sq": 0.9},
        {"population": "retain", "alignment_cos_sq": 0.1},
        {"population": "test", "alignment_cos_sq": 0.5},
    ]
    values, labels, dropped = build_task_arrays(rows, "alignment_cos_sq", table.task("forget_vs_retain"))
    assert values == [0.9, 0.1]
    assert labels == [1, 0]
    assert dropped == 0


def test_build_task_arrays_drops_missing_values_and_counts_them(table):
    rows = [
        {"population": "forget", "fuma_lora_ratio": None},
        {"population": "forget", "fuma_lora_ratio": 2.0},
        {"population": "retain", "fuma_lora_ratio": 1.0},
    ]
    values, labels, dropped = build_task_arrays(rows, "fuma_lora_ratio", table.task("forget_vs_retain"))
    assert values == [2.0, 1.0]
    assert labels == [1, 0]
    assert dropped == 1


def test_build_task_arrays_keeps_negative_infinity_as_a_ranking_position(table):
    rows = [
        {"population": "forget", "combined_log_score": float("-inf")},
        {"population": "retain", "combined_log_score": 1.0},
    ]
    values, labels, dropped = build_task_arrays(rows, "combined_log_score", table.task("forget_vs_retain"))
    assert values == [float("-inf"), 1.0]
    assert dropped == 0


def test_build_task_arrays_drops_nan(table):
    rows = [
        {"population": "forget", "grad_norm_sq": float("nan")},
        {"population": "retain", "grad_norm_sq": 1.0},
    ]
    values, _, dropped = build_task_arrays(rows, "grad_norm_sq", table.task("forget_vs_retain"))
    assert values == [1.0]
    assert dropped == 1


# --------------------------------------------------------------- table parsing


def _write(tmp_path, body: str) -> str:
    path = os.path.join(tmp_path, "orientations.yaml")
    with open(path, "w") as handle:
        handle.write(textwrap.dedent(body))
    return path


def test_missing_rationale_is_rejected(tmp_path):
    path = _write(
        tmp_path,
        """
        tasks:
          t: {positive: [forget], negative: [retain]}
        scores:
          s:
            t: {direction: higher_is_positive}
        """,
    )
    with pytest.raises(ValueError, match="no rationale"):
        OrientationTable.load(path)


def test_a_score_missing_a_task_is_rejected(tmp_path):
    path = _write(
        tmp_path,
        """
        tasks:
          t1: {positive: [forget], negative: [retain]}
          t2: {positive: [forget], negative: [test]}
        scores:
          s:
            t1: {direction: higher_is_positive, rationale: because}
        """,
    )
    with pytest.raises(ValueError, match="no declared orientation"):
        OrientationTable.load(path)


def test_an_invalid_direction_is_rejected(tmp_path):
    path = _write(
        tmp_path,
        """
        tasks:
          t: {positive: [forget], negative: [retain]}
        scores:
          s:
            t: {direction: whichever_wins, rationale: because}
        """,
    )
    with pytest.raises(ValueError, match="not one of"):
        OrientationTable.load(path)


def test_a_population_in_both_classes_is_rejected(tmp_path):
    path = _write(
        tmp_path,
        """
        tasks:
          t: {positive: [forget, retain], negative: [retain]}
        scores: {}
        """,
    )
    with pytest.raises(ValueError, match="both"):
        OrientationTable.load(path)
