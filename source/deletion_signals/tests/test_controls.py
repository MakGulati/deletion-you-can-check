"""The two reference points, and the trajectory instrumentation.

No model is loaded here: what can go wrong in the controls is *which samples
they train on* and *which RNG stream they use*, and both are decided before any
tensor is allocated.
"""

from __future__ import annotations

import json
import os

import pytest
import torch

from deletion_signals.config import DEFAULTS, deep_merge
from deletion_signals.controls import CONTROLS, control_training_samples, training_seed_for
from deletion_signals.data import Sample
from deletion_signals.schema import text_fingerprint
from deletion_signals.train import build_probes, probe_loss


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
    return {
        "forget": _population("forget", 20),
        "retain": _population("retain", 80),
        "test": _population("test", 30),
    }


# ------------------------------------------------------------ what they train on


def test_exact_retraining_never_sees_the_forget_set(splits):
    """The omission IS the experiment. A single forget sample leaking into this
    training set would make the gold standard not a gold standard."""
    samples = control_training_samples(splits, "exact_retrain")
    assert len(samples) == 80
    assert {s.population for s in samples} == {"retain"}

    forget_ids = {s.text_sha1 for s in splits["forget"]}
    assert not forget_ids & {s.text_sha1 for s in samples}


def test_the_null_refit_sees_exactly_what_the_original_finetune_saw(splits):
    """It must differ from theta_finetuned in training randomness and nothing
    else, or it stops being a null."""
    samples = control_training_samples(splits, "null_refit")
    assert len(samples) == 100
    assert {s.population for s in samples} == {"forget", "retain"}


def test_neither_control_trains_on_the_test_population(splits):
    for control in CONTROLS:
        samples = control_training_samples(splits, control)
        assert "test" not in {s.population for s in samples}


def test_an_unknown_control_is_rejected(splits):
    with pytest.raises(KeyError, match="Unknown control"):
        control_training_samples(splits, "wishful_thinking")


def test_a_control_with_no_training_samples_is_rejected():
    with pytest.raises(ValueError, match="no training samples"):
        control_training_samples({"forget": [], "retain": [], "test": []}, "exact_retrain")


# ----------------------------------------------------------------- RNG streams


def test_the_null_refit_uses_a_different_training_seed():
    """Same data, new randomness -- that is the entire difference between the
    null refit and the original fine-tune."""
    assert training_seed_for("null_refit", 42) != 42
    assert training_seed_for("null_refit", 42) == 1042


def test_exact_retraining_keeps_the_original_training_seed():
    """Its difference from theta_finetuned should come from the missing forget
    set, not from a different RNG stream layered on top of it."""
    assert training_seed_for("exact_retrain", 42) == 42


def test_the_two_controls_never_collide_across_the_studied_seeds():
    used = [
        training_seed_for(control, seed)
        for control in CONTROLS
        for seed in (42, 43, 44)
    ]
    assert len(set(used)) == len(used)


# ------------------------------------------------------------------ trajectory


class TinyOutput:
    def __init__(self, loss):
        self.loss = loss


class TinyModel(torch.nn.Module):
    """Loss depends on a parameter, so a probe can detect the model moving."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.0]))
        self.training_calls = []

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        return TinyOutput(self.weight.pow(2).sum() * input_ids.shape[1])

    def train(self, mode: bool = True):
        self.training_calls.append(mode)
        return super().train(mode)


class TinyTokenizer:
    def __call__(self, texts, padding=None, truncation=None, max_length=None,
                 return_tensors=None):
        if isinstance(texts, str):
            texts = [texts]
        length = 4
        ids = torch.ones((len(texts), length), dtype=torch.long)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


def _probe_config():
    return deep_merge(DEFAULTS, {"device": "cpu", "data": {"max_length": 8}})


def test_the_probe_restores_the_models_training_mode(splits):
    """Dropout would otherwise make every probe a different random function, and
    a rising trajectory could be noise rather than the model moving."""
    model = TinyModel()
    model.train()
    model.training_calls.clear()

    probe_loss(model, TinyTokenizer(), splits["retain"][:4], _probe_config(), batch_size=2)

    assert model.training is True          # restored
    assert False in model.training_calls   # and eval() was actually entered


def test_the_probe_is_token_weighted(splits):
    model = TinyModel()
    value = probe_loss(model, TinyTokenizer(), splits["retain"][:4], _probe_config(), 2)
    # Every sample has the same length here, so the token-weighted mean equals
    # the per-batch loss: weight^2 * n_tokens = 1 * 4.
    assert value == pytest.approx(4.0)


def test_probes_are_a_fixed_prefix_of_each_population(splits):
    """The same samples at every step and across runs -- a resampled probe would
    move for reasons unrelated to the model."""
    first = build_probes(splits, 8)
    second = build_probes(splits, 8)
    assert [s.sample_id for s in first["forget"]] == [s.sample_id for s in second["forget"]]
    assert len(first["forget"]) == 8
    assert len(first["retain"]) == 8


def test_probes_cover_every_population_including_test(splits):
    """Utility collapse shows up on test as well as retain, and the trajectory
    has to be able to see it."""
    probes = build_probes(splits, 4)
    assert set(probes) == {"forget", "retain", "test"}


def test_a_probe_larger_than_the_population_is_not_an_error(splits):
    probes = build_probes(splits, 10_000)
    assert len(probes["forget"]) == 20
