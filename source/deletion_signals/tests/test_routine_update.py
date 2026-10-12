"""The routine-update control's data selection and disjointness guarantee.

One failure here would invalidate the whole component: if the update trains on
records the detector is then evaluated on, every Component D number becomes a
measurement of leakage. The disjointness check is therefore enforced at run time
and tested here on splits built to violate it in each available way.
"""

from __future__ import annotations

import pytest

from deletion_signals.config import DEFAULTS, deep_merge, validate_config
from deletion_signals.data import Sample
from deletion_signals.routine_update import (
    CONDITIONS,
    assert_disjoint_from_scored,
    scored_test_cap,
    update_samples,
    update_training_config,
    update_training_seed,
)


def _splits(n_forget=10, n_retain=30, n_test=40):
    """Populations with distinct text, so fingerprints separate them."""
    return {
        "forget": [Sample.create(i, f"forget text {i}", "forget") for i in range(n_forget)],
        "retain": [Sample.create(i, f"retain text {i}", "retain") for i in range(n_retain)],
        "test": [Sample.create(i, f"test text {i}", "test") for i in range(n_test)],
    }


def _config(**overrides):
    base = deep_merge(
        DEFAULTS,
        {
            "scoring": {"max_samples_per_population": {"test": 20}},
            "routine_update": {"n_samples": 10},
        },
    )
    return deep_merge(base, overrides)


# ------------------------------------------------------------- the safe tail


def test_the_update_slice_starts_where_scoring_stops():
    config = _config()
    splits = _splits()

    selected = update_samples(config, splits)

    assert len(selected) == 10
    assert [s.sample_id for s in selected] == [f"test:{i}" for i in range(20, 30)]


def test_a_uniform_sample_cap_also_defines_the_tail():
    config = _config(scoring={"max_samples_per_population": 15})
    selected = update_samples(config, _splits())
    assert [s.sample_id for s in selected] == [f"test:{i}" for i in range(15, 25)]


def test_no_scoring_cap_is_refused_rather_than_guessed():
    """With every test record scored there is no slice that is provably never
    scored, and picking one anyway would put training data into a population the
    detector is evaluated on."""
    config = _config(scoring={"max_samples_per_population": None})
    with pytest.raises(ValueError, match="never scored"):
        scored_test_cap(config)
    with pytest.raises(ValueError, match="never scored"):
        update_samples(config, _splits())


def test_too_few_records_fails_instead_of_silently_shrinking():
    """A shorter update on one seed would make the seeds incomparable."""
    config = _config(routine_update={"n_samples": 50})
    with pytest.raises(ValueError, match="taking fewer would silently change"):
        update_samples(config, _splits(n_test=40))


def test_the_slice_is_deterministic():
    config = _config()
    splits = _splits()
    first = [s.sample_id for s in update_samples(config, splits)]
    second = [s.sample_id for s in update_samples(config, splits)]
    assert first == second


# ------------------------------------------------------------- disjointness


def test_a_correctly_sliced_update_passes_the_check():
    config = _config()
    splits = _splits()
    report = assert_disjoint_from_scored(update_samples(config, splits), splits, config)

    assert report["disjoint"] is True
    assert report["n_update"] == 10
    assert report["overlap_with_scored"] == {"forget": 0, "retain": 0, "test_scored": 0}
    assert report["scored_test_cap"] == 20


def test_an_overlap_with_the_scored_test_prefix_is_caught():
    config = _config()
    splits = _splits()
    offending = list(splits["test"])[:10]        # inside the scored prefix
    with pytest.raises(ValueError, match="test_scored"):
        assert_disjoint_from_scored(offending, splits, config)


def test_an_overlap_with_the_forget_set_is_caught():
    config = _config()
    splits = _splits()
    with pytest.raises(ValueError, match="forget"):
        assert_disjoint_from_scored(list(splits["forget"])[:5], splits, config)


def test_an_overlap_with_retain_is_caught_including_the_curvature_head():
    """The whole retain population is off limits: its head is the Fisher
    curvature holdout and its tail is scored."""
    config = _config()
    splits = _splits()
    with pytest.raises(ValueError, match="retain"):
        assert_disjoint_from_scored(list(splits["retain"])[:3], splits, config)
    with pytest.raises(ValueError, match="retain"):
        assert_disjoint_from_scored(list(splits["retain"])[-3:], splits, config)


def test_disjointness_is_checked_by_fingerprint_not_by_index():
    """Identical text in two populations is an overlap however it is indexed."""
    config = _config()
    splits = _splits()
    duplicate = Sample.create(99, splits["forget"][0].text, "test")
    with pytest.raises(ValueError, match="forget"):
        assert_disjoint_from_scored([duplicate], splits, config)


# ------------------------------------------------------- the update schedule


def test_the_update_schedule_replaces_the_fine_tuning_schedule():
    config = _config(
        routine_update={
            "learning_rate": 5.0e-5, "epochs": 3, "batch_size": 8,
            "gradient_accumulation_steps": 4, "n_samples": 10,
        }
    )
    merged = update_training_config(config)

    assert merged["training"]["learning_rate"] == 5.0e-5
    assert merged["training"]["epochs"] == 3
    assert merged["training"]["batch_size"] == 8
    # Not the fine-tune's 2.0e-4 / 6 epochs.
    assert merged["training"]["learning_rate"] != DEFAULTS["training"]["learning_rate"]


def test_the_update_uses_no_warmup():
    """~18 optimiser steps under a 100-step warmup would never reach the
    declared learning rate, so the declared rate would not be the applied one."""
    assert DEFAULTS["routine_update"]["warmup_steps"] == 0
    assert update_training_config(_config())["training"]["warmup_steps"] == 0


def test_substituting_the_schedule_does_not_mutate_the_caller_config():
    config = _config()
    before = config["training"]["learning_rate"]
    update_training_config(config)
    assert config["training"]["learning_rate"] == before


# ------------------------------------------------------------ the conditions


def test_both_conditions_share_one_update_seed():
    """Conditions 2 and 4 must receive the IDENTICAL update, so the only thing
    that differs between them is whether a deletion happened underneath it."""
    assert update_training_seed(42) == update_training_seed(42)
    assert update_training_seed(42) != update_training_seed(43)


def test_the_update_seed_is_distinct_from_the_other_controls():
    """Offsets: exact_retrain 0, null_refit 1000, routine update 2000."""
    from deletion_signals.controls import training_seed_for

    for seed in (42, 43, 44):
        assert update_training_seed(seed) != training_seed_for("exact_retrain", seed)
        assert update_training_seed(seed) != training_seed_for("null_refit", seed)


def test_the_two_conditions_differ_only_in_their_base_checkpoint():
    assert CONDITIONS["routine_update"]["base"] == "finetuned"
    assert CONDITIONS["routine_update"]["deletion"] == "none"
    assert CONDITIONS["exact_retrain_routine_update"]["base"] == "exact_retrain"
    assert CONDITIONS["exact_retrain_routine_update"]["deletion"] == "exact"


def test_both_conditions_are_treated_as_controls_by_the_audit():
    from deletion_signals.audit import CONTROL_METHODS

    for condition in CONDITIONS:
        assert condition in CONTROL_METHODS


def test_only_exact_retraining_supplies_the_rnff_denominator():
    """Adding conditions to CONTROL_METHODS must not change what normalises RNFF."""
    from deletion_signals.audit import EXACT_RETRAIN, retrain_losses_by_seed
    from deletion_signals.schema import ScoreFile

    files = [
        ScoreFile(metadata={"method": "routine_update", "seed": 42},
                  rows=[{"population": "forget", "loss_unlearned": 9.0,
                         "loss_finetuned": 1.0, "n_tokens": 10}]),
        ScoreFile(metadata={"method": EXACT_RETRAIN, "seed": 42},
                  rows=[{"population": "forget", "loss_unlearned": 2.0,
                         "loss_finetuned": 1.0, "n_tokens": 10}]),
    ]
    losses = retrain_losses_by_seed(files)
    assert losses[42]["forget"] == pytest.approx(2.0)


# ------------------------------------------------------------- configuration


def test_an_unsupported_update_source_is_rejected():
    """'retain' in particular: it is the negative class of the primary task, so
    an update on it would train on exactly what the detector is scored against."""
    config = deep_merge(DEFAULTS, {"routine_update": {"source": "retain"}})
    with pytest.raises(ValueError, match="valid sources are"):
        validate_config(config)


def test_a_nonpositive_sample_count_is_rejected():
    config = deep_merge(DEFAULTS, {"routine_update": {"n_samples": 0}})
    with pytest.raises(ValueError, match="n_samples must be positive"):
        validate_config(config)


def test_the_frozen_defaults_match_the_preregistration():
    """These numbers are frozen by preregistration_v3_fllm.yaml and may not be
    tuned to make a detector succeed or fail."""
    update = DEFAULTS["routine_update"]
    assert update["learning_rate"] == 5.0e-5
    assert update["epochs"] == 3
    assert update["n_samples"] == 200
    assert update["source"] == "test_tail"


# ------------------------------------------------------- the member_tail source


def test_member_tail_is_an_accepted_source():
    """BookMIA has 4935 members against a draw of 1000, so the clean source --
    fresh records from the SAME population as the training data -- is available
    there even though ArxivMIA's exhausted member pool forbade it."""
    config = deep_merge(DEFAULTS, {"routine_update": {"source": "member_tail"}})
    validate_config(config)          # must not raise


def test_member_tail_requires_a_seed():
    """The draw it excludes is a function of the seed; without one it could
    silently overlap the split."""
    config = _config(routine_update={"source": "member_tail", "n_samples": 5})
    with pytest.raises(ValueError, match="needs the seed"):
        update_samples(config, _splits(), seed=None)


def test_member_tail_excludes_everything_the_split_drew(monkeypatch):
    """The whole point: the update records must be members the split never saw.

    A fake 20-member corpus stands in for BookMIA, so the exclusion logic is
    tested without a download.
    """
    import random as _random

    from deletion_signals import routine_update as module

    corpus = [{"snippet": f"member {i}", "label": 1} for i in range(20)]
    corpus += [{"snippet": f"nonmember {i}", "label": 0} for i in range(20)]

    class FakeDataset(list):
        column_names = ["snippet", "label"]

    monkeypatch.setattr(
        module, "load_dataset", lambda *a, **k: FakeDataset(corpus), raising=False
    )
    import datasets
    monkeypatch.setattr(datasets, "load_dataset", lambda *a, **k: FakeDataset(corpus))

    config = deep_merge(
        DEFAULTS,
        {
            "data": {"name": "swj0419/BookMIA", "split": "train",
                     "text_field": "snippet", "train_samples": 12,
                     "test_samples": 10, "forget_ratio": 0.2},
            "routine_update": {"source": "member_tail", "n_samples": 5},
            "scoring": {"max_samples_per_population": {"test": 5}},
        },
    )

    selected = module.member_tail_samples(config, seed=42, n_samples=5)
    assert len(selected) == 5
    assert all(s.population == "update" for s in selected)

    # Rebuild the split's own draw and confirm zero overlap.
    rng = _random.Random(42)
    drawn = {row["snippet"] for row in rng.sample(
        [r for r in corpus if r["label"] == 1], 12)}
    assert not ({s.text for s in selected} & drawn)
    # ...and that they are members, not the test tail.
    assert all(s.text.startswith("member ") for s in selected)


def test_member_tail_fails_loudly_when_the_pool_is_exhausted(monkeypatch):
    """ArxivMIA's case: the split consumes every member, so this source cannot
    be used and must say so rather than silently returning fewer records."""
    from deletion_signals import routine_update as module

    corpus = [{"text": f"m{i}", "label": 1} for i in range(10)]
    corpus += [{"text": f"n{i}", "label": 0} for i in range(10)]

    class FakeDataset(list):
        column_names = ["text", "label"]

    import datasets
    monkeypatch.setattr(datasets, "load_dataset", lambda *a, **k: FakeDataset(corpus))

    config = deep_merge(
        DEFAULTS,
        {
            "data": {"name": "zhliu/ArxivMIA", "split": "train",
                     "text_field": "text", "train_samples": 10},
            "routine_update": {"source": "member_tail", "n_samples": 5},
        },
    )
    with pytest.raises(ValueError, match="beyond the split's draw"):
        module.member_tail_samples(config, seed=42, n_samples=5)


def test_the_disjointness_report_names_the_population_it_drew_from():
    """test_tail carries a confound and member_tail does not; the run record has
    to say which, because the two support different claims."""
    config = _config()
    splits = _splits()
    report = assert_disjoint_from_scored(update_samples(config, splits), splits, config)
    assert report["source"] == "test_tail"
    assert "NON-MEMBERS" in report["population_of_update_records"]
    assert "confound" in report["population_of_update_records"]
