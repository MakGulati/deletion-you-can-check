"""The collector's scoring path, on a tiny hand-built model.

No downloads, no GPU, no ``transformers`` model hub. A three-parameter linear
model and a two-token "tokenizer" are enough to exercise the parts that can go
quietly wrong: which parameters are selected, that one backward feeds every
statistic, that the FUMA groups partition correctly, and that a not-applicable
score comes out as ``None`` with a reason rather than as a number.
"""

from __future__ import annotations

import copy
import math

import pytest
import torch

from deletion_signals.collect import ModelBundle, ScoreCollector
from deletion_signals.config import DEFAULTS, deep_merge
from deletion_signals.data import Sample
from deletion_signals.schema import text_fingerprint


class TinyOutput:
    def __init__(self, loss):
        self.loss = loss


class TinyModel(torch.nn.Module):
    """A stand-in for a PEFT-wrapped causal LM.

    Parameter names deliberately mimic PEFT's, so the LoRA/non-LoRA split is
    exercised by the same substring rule the real models go through.
    """

    def __init__(self, freeze_backbone: bool = True):
        super().__init__()
        self.layer_lora_A = torch.nn.Parameter(torch.tensor([[0.5, -0.25]]))
        self.layer_lora_B = torch.nn.Parameter(torch.tensor([[0.75], [0.1]]))
        self.layer_base_layer_weight = torch.nn.Parameter(
            torch.tensor([2.0, -1.0, 0.5]), requires_grad=not freeze_backbone
        )
        self.config = type("Cfg", (), {"use_cache": False, "pad_token_id": 0})()

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        x = input_ids.float().sum()
        loss = (
            (self.layer_lora_A.sum() * x) ** 2
            + (self.layer_lora_B.sum() * x)
            + (self.layer_base_layer_weight.pow(2).sum() * 0.01 * x)
        )
        return TinyOutput(loss)

    def zero_grad(self, set_to_none: bool = True):  # noqa: D102 - torch signature
        super().zero_grad(set_to_none=set_to_none)


class TinyTokenizer:
    """Returns a fixed-length token sequence derived from the text length."""

    def __call__(self, text, return_tensors=None, truncation=None, max_length=None,
                 padding=None):
        length = min(max(len(text.split()), 2), max_length or 8)
        ids = torch.arange(1, length + 1).unsqueeze(0)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


def _config(**overrides):
    config = deep_merge(
        DEFAULTS,
        {
            "device": "cpu",
            "training_mode": "lora",
            "scoring": {"param_scope": "trainable", "max_length": 8, "eps": 1e-12},
            "scores": {
                "loss_calibrated": {"enabled": True, "reference": "pretrained_base"},
                "grad_calibrated": {"enabled": True, "reference": "finetuned_pre_unlearning"},
                "fuma": {"enabled": True, "denominator_source": "same_model"},
                "sif": {"enabled": True, "method": "gradient", "damping": 1e-3, "beta": 1.0},
            },
        },
    )
    return deep_merge(config, overrides)


def _bundle(freeze_backbone: bool = True, with_base: bool = True):
    unlearned = TinyModel(freeze_backbone=freeze_backbone)
    finetuned = copy.deepcopy(unlearned)
    with torch.no_grad():
        # Make ghat non-zero: unlearning moved lora_B.
        unlearned.layer_lora_B.add_(0.3)
    base = TinyModel(freeze_backbone=True) if with_base else None
    return ModelBundle(
        unlearned=unlearned,
        tokenizer=TinyTokenizer(),
        finetuned=finetuned,
        base=base,
        shadow=None,
    )


def _sample(text: str = "one two three four") -> Sample:
    return Sample(
        sample_id="forget:0",
        text=text,
        population="forget",
        text_sha1=text_fingerprint(text),
    )


META = {
    "seed": 42,
    "method": "ga_gd",
    "dataset": "tiny",
    "model_id": "tiny",
    "unlearned_checkpoint": "/tmp/unlearned",
    "finetuned_checkpoint": "/tmp/finetuned",
}


# ------------------------------------------------------------------- selection


def test_only_trainable_parameters_are_scored():
    collector = ScoreCollector(_config(), _bundle(freeze_backbone=True), torch.device("cpu"))
    assert collector.param_names == ["layer_lora_A", "layer_lora_B"]


def test_the_unlearning_direction_spans_the_scored_parameters():
    collector = ScoreCollector(_config(), _bundle(), torch.device("cpu"))
    assert set(collector.direction) == set(collector.param_names)
    assert collector.direction_norm > 0.0


def test_no_finetuned_model_means_no_direction():
    bundle = _bundle()
    bundle.finetuned = None
    config = _config(scores={"grad_calibrated": {"enabled": False}})
    collector = ScoreCollector(config, bundle, torch.device("cpu"))
    assert collector.direction is None


# ------------------------------------------------------------------- scoring


def test_a_row_carries_every_required_column():
    collector = ScoreCollector(_config(), _bundle(), torch.device("cpu"))
    row = collector.score_sample(_sample(), META).as_dict()

    for column in (
        "seed", "method", "dataset", "model_id", "sample_id", "text_sha1",
        "population", "loss_unlearned", "loss_base", "loss_ratio", "loss_delta",
        "grad_norm_sq", "grad_norm_sq_ref", "grad_reference_model", "grad_ratio",
        "log_grad_ratio", "fuma_lora_ratio", "fuma_na_reason", "raw_cosine",
        "alignment_cos_sq", "sif", "combined_log_score", "flags",
    ):
        assert column in row, f"missing required column {column}"


def test_scores_are_internally_consistent():
    collector = ScoreCollector(_config(), _bundle(), torch.device("cpu"))
    row = collector.score_sample(_sample(), META)

    assert row.grad_norm_sq > 0
    assert row.loss_ratio == pytest.approx(row.loss_unlearned / row.loss_base)
    assert row.loss_delta == pytest.approx(row.loss_unlearned - row.loss_base)
    assert row.grad_ratio == pytest.approx(row.grad_norm_sq / row.grad_norm_sq_ref)
    assert row.log_grad_ratio == pytest.approx(
        math.log(row.grad_norm_sq + 1e-12) - math.log(row.grad_norm_sq_ref + 1e-12)
    )
    assert row.alignment_cos_sq == pytest.approx(row.raw_cosine**2)
    assert 0.0 <= row.alignment_cos_sq <= 1.0
    # SIF in the gradient limit is a rescaled gradient norm, by construction.
    assert row.sif == pytest.approx(row.grad_norm_sq / 1e-3)


def test_scoring_is_deterministic():
    collector = ScoreCollector(_config(), _bundle(), torch.device("cpu"))
    first = collector.score_sample(_sample(), META)
    second = collector.score_sample(_sample(), META)
    assert first.grad_norm_sq == second.grad_norm_sq
    assert first.alignment_cos_sq == second.alignment_cos_sq


def test_flags_are_clean_on_a_healthy_sample():
    collector = ScoreCollector(_config(), _bundle(), torch.device("cpu"))
    flags = collector.score_sample(_sample(), META).flags
    assert flags["zero_grad_norm"] is False
    assert flags["nonfinite_grad"] is False
    assert flags["nonfinite_loss"] is False


def test_the_reference_model_used_is_recorded_on_the_row():
    collector = ScoreCollector(_config(), _bundle(), torch.device("cpu"))
    row = collector.score_sample(_sample(), META)
    assert row.loss_reference_model.startswith("pretrained_base:")
    assert row.grad_reference_model.startswith("finetuned:")


def test_all_three_loss_references_are_collected_and_finite():
    """The sweep needs L_base (membership) and L_finetuned (what unlearning did)
    at the same time, which is why they are separate score families rather than
    two settings of one reference."""
    config = _config(scores={"loss_vs_finetuned": {"enabled": True}})
    collector = ScoreCollector(config, _bundle(), torch.device("cpu"))
    row = collector.score_sample(_sample(), META)

    for value in (row.loss_unlearned, row.loss_base, row.loss_finetuned):
        assert value is not None and math.isfinite(value)
    assert row.loss_reference_model.startswith("pretrained_base:")
    assert row.loss_finetuned_model.startswith("finetuned:")


def test_the_unlearning_attributed_loss_scores_match_their_definitions():
    config = _config(scores={"loss_vs_finetuned": {"enabled": True}})
    bundle = _bundle()
    # The fixture's base and fine-tuned models start identical, which would make
    # the two loss families numerically equal for reasons that have nothing to
    # do with the code under test. Move the base so the two references are
    # genuinely different models, as they are in a real run.
    with torch.no_grad():
        bundle.base.layer_lora_A.add_(0.4)
    collector = ScoreCollector(config, bundle, torch.device("cpu"))
    row = collector.score_sample(_sample(), META)

    assert row.loss_ratio_vs_finetuned == pytest.approx(
        row.loss_unlearned / row.loss_finetuned
    )
    assert row.loss_delta_vs_finetuned == pytest.approx(
        row.loss_unlearned - row.loss_finetuned
    )
    # The two families must not collapse into one another: different reference
    # models, therefore different numbers.
    assert row.loss_finetuned != row.loss_base
    assert row.loss_ratio != row.loss_ratio_vs_finetuned


def test_the_second_loss_reference_is_off_by_default():
    collector = ScoreCollector(_config(), _bundle(), torch.device("cpu"))
    row = collector.score_sample(_sample(), META)
    assert row.loss_finetuned is None
    assert row.loss_ratio_vs_finetuned is None
    assert row.loss_delta_vs_finetuned is None


def test_the_finetuned_forward_pass_is_not_repeated_when_it_is_already_the_reference():
    """With loss_calibrated pointed at the fine-tuned model, loss_finetuned is
    the number that block already computed; a second forward would return it
    again at full cost."""
    config = _config(
        scores={
            "loss_calibrated": {"enabled": True, "reference": "finetuned_pre_unlearning"},
            "loss_vs_finetuned": {"enabled": True},
        }
    )
    collector = ScoreCollector(config, _bundle(with_base=False), torch.device("cpu"))

    calls = {"n": 0}
    original = collector._loss_only

    def counting(model, inputs):
        calls["n"] += 1
        return original(model, inputs)

    collector._loss_only = counting  # type: ignore[method-assign]
    row = collector.score_sample(_sample(), META)

    assert calls["n"] == 1
    assert row.loss_finetuned == row.loss_base
    assert row.loss_ratio_vs_finetuned == pytest.approx(row.loss_ratio)


def test_no_finetuned_model_leaves_the_new_columns_absent_rather_than_zero():
    bundle = _bundle()
    bundle.finetuned = None
    config = _config(
        scores={
            "grad_calibrated": {"enabled": False},
            "loss_vs_finetuned": {"enabled": True},
        }
    )
    collector = ScoreCollector(config, bundle, torch.device("cpu"))
    row = collector.score_sample(_sample(), META)
    assert row.loss_finetuned is None
    assert row.loss_ratio_vs_finetuned is None


def test_disabling_the_loss_reference_leaves_the_calibrated_columns_absent():
    config = _config(scores={"loss_calibrated": {"enabled": False}})
    bundle = _bundle(with_base=False)
    collector = ScoreCollector(config, bundle, torch.device("cpu"))
    row = collector.score_sample(_sample(), META)
    assert row.loss_base is None
    assert row.loss_ratio is None
    assert row.loss_delta is None


# ---------------------------------------------------------------------- FUMA


def test_fuma_is_computed_when_the_backbone_is_unfrozen_for_it():
    collector = ScoreCollector(_config(), _bundle(freeze_backbone=True), torch.device("cpu"))
    assert collector.fuma_unfroze_backbone is True
    row = collector.score_sample(_sample(), META)
    assert row.fuma_na_reason is None
    assert row.fuma_lora_ratio is not None and row.fuma_lora_ratio > 0


def test_fuma_reuses_the_single_backward_on_the_scored_model():
    bundle = _bundle()
    collector = ScoreCollector(_config(), bundle, torch.device("cpu"))
    assert collector.fuma_model is bundle.unlearned

    calls = {"n": 0}
    original = collector._forward_backward

    def counting(model, names, inputs):
        calls["n"] += 1
        return original(model, names, inputs)

    collector._forward_backward = counting  # type: ignore[method-assign]
    collector.score_sample(_sample(), META)
    # One for the unlearned model, one for the gradient reference. FUMA adds
    # none, because it reads the gradients the first pass already produced.
    assert calls["n"] == 2


def test_fuma_is_not_applicable_when_disabled_and_says_so():
    config = _config(scores={"fuma": {"enabled": False}})
    collector = ScoreCollector(config, _bundle(), torch.device("cpu"))
    row = collector.score_sample(_sample(), META)
    assert row.fuma_lora_ratio is None
    assert "disabled" in row.fuma_na_reason


def test_fuma_source_none_records_the_reason_rather_than_a_zero():
    config = _config(scores={"fuma": {"enabled": True, "denominator_source": "none"}})
    collector = ScoreCollector(config, _bundle(), torch.device("cpu"))
    row = collector.score_sample(_sample(), META)
    assert row.fuma_lora_ratio is None
    assert "not computed" in row.fuma_na_reason


def test_fuma_groups_partition_the_parameters():
    """The LoRA and non-LoRA element counts must add up to the whole model."""
    from deletion_signals.gradients import ParamScope, select_param_names

    bundle = _bundle()
    collector = ScoreCollector(_config(), bundle, torch.device("cpu"))
    all_names = select_param_names(bundle.unlearned, ParamScope.FULL)
    assert set(collector.fuma_names) == set(all_names)
    total = sum(p.numel() for p in bundle.unlearned.parameters())
    assert sum(collector.fuma_numels.values()) == total


# --------------------------------------------------------------- truncation


def test_a_truncated_sample_is_flagged():
    config = _config(scoring={"max_length": 3})
    collector = ScoreCollector(config, _bundle(), torch.device("cpu"))
    row = collector.score_sample(_sample("a b c d e f g h i j"), META)
    assert row.flags["truncated"] is True
    assert row.n_tokens == 2
