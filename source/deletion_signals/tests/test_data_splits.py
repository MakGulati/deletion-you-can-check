"""Split handling: the manifest round-trip and per-population subsampling."""

from __future__ import annotations

import pytest

from deletion_signals.data import Sample, subsample
from deletion_signals.schema import text_fingerprint


def _population(name: str, size: int):
    return [
        Sample(
            sample_id=f"{name}:{index}",
            text=f"{name} sample {index}",
            population=name,
            text_sha1=text_fingerprint(f"{name} sample {index}"),
        )
        for index in range(size)
    ]


@pytest.fixture
def splits():
    return {"forget": _population("forget", 5), "retain": _population("retain", 9),
            "test": _population("test", 7)}


# ------------------------------------------------------------------ no cap


def test_none_keeps_every_population_whole(splits):
    result = subsample(splits, None)
    assert {k: len(v) for k, v in result.items()} == {"forget": 5, "retain": 9, "test": 7}


def test_the_result_is_a_copy_not_a_view(splits):
    result = subsample(splits, None)
    result["forget"].clear()
    assert len(splits["forget"]) == 5


# ------------------------------------------------------------------ int cap


def test_an_int_caps_every_population_alike(splits):
    result = subsample(splits, 3)
    assert {k: len(v) for k, v in result.items()} == {"forget": 3, "retain": 3, "test": 3}


def test_a_cap_above_the_population_size_is_not_an_error(splits):
    assert len(subsample(splits, 100)["forget"]) == 5


# ------------------------------------------------------------------ dict cap


def test_a_mapping_caps_only_the_named_populations(splits):
    """The sweep's actual setting: shrink the membership control, leave the two
    populations that carry the primary comparison whole."""
    result = subsample(splits, {"test": 4})
    assert {k: len(v) for k, v in result.items()} == {"forget": 5, "retain": 9, "test": 4}


def test_an_explicit_none_in_the_mapping_means_keep_all(splits):
    result = subsample(splits, {"test": None, "forget": 2})
    assert len(result["test"]) == 7
    assert len(result["forget"]) == 2


def test_an_unknown_population_in_the_mapping_is_rejected(splits):
    """A typo would otherwise leave that population silently uncapped, and the
    run would cost hours more than intended without saying so."""
    with pytest.raises(ValueError, match="not in the split"):
        subsample(splits, {"tests": 4})


# -------------------------------------------------------------- reproducibility


def test_subsampling_takes_a_prefix_so_smaller_runs_share_samples(splits):
    """Two runs at different caps must score overlapping samples, which a random
    subset would not guarantee without a second RNG to record."""
    small = subsample(splits, {"test": 3})["test"]
    large = subsample(splits, {"test": 6})["test"]
    assert [s.sample_id for s in small] == [s.sample_id for s in large][:3]


def test_a_capped_population_is_a_subset_of_the_full_one(splits):
    capped = {s.text_sha1 for s in subsample(splits, 2)["retain"]}
    full = {s.text_sha1 for s in splits["retain"]}
    assert capped < full
