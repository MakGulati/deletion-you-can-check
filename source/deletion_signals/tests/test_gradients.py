"""Gradient statistics: cosine, layerwise accumulation, grouping, edge cases."""

from __future__ import annotations

import math

import pytest
import torch

from deletion_signals.gradients import (
    ParamScope,
    accumulate_gradient_stats,
    concatenated_gradient_stats,
    direction_norm,
    is_lora_param,
    parameter_difference,
    select_param_names,
)


# --------------------------------------------------------------- known vectors


def test_cosine_of_identical_vectors_is_one():
    grad = torch.tensor([3.0, 4.0])
    stats = accumulate_gradient_stats([("w", grad)], direction={"w": grad})
    assert stats.raw_cosine() == pytest.approx(1.0)
    assert stats.norm_sq == pytest.approx(25.0)


def test_cosine_of_antiparallel_vectors_is_minus_one_but_cos_sq_is_one():
    grad = torch.tensor([1.0, 2.0, 2.0])
    stats = accumulate_gradient_stats([("w", grad)], direction={"w": -grad})
    assert stats.raw_cosine() == pytest.approx(-1.0)
    # cos^2 is what the evaluation ranks by, and it is sign-invariant -- which
    # is exactly why the ghat sign convention does not matter.
    assert stats.raw_cosine() ** 2 == pytest.approx(1.0)


def test_cosine_of_orthogonal_vectors_is_zero():
    stats = accumulate_gradient_stats(
        [("w", torch.tensor([1.0, 0.0]))], direction={"w": torch.tensor([0.0, 1.0])}
    )
    assert stats.raw_cosine() == pytest.approx(0.0)


def test_cosine_matches_hand_computation():
    # cos = (1*4 + 2*5 + 3*6) / (sqrt(14) * sqrt(77)) = 32 / sqrt(1078)
    g = torch.tensor([1.0, 2.0, 3.0])
    d = torch.tensor([4.0, 5.0, 6.0])
    stats = accumulate_gradient_stats([("w", g)], direction={"w": d})
    assert stats.raw_cosine() == pytest.approx(32.0 / math.sqrt(1078.0), rel=1e-12)


def test_cosine_is_clamped_into_range():
    g = torch.randn(1000, dtype=torch.float64)
    stats = accumulate_gradient_stats([("w", g)], direction={"w": g})
    assert -1.0 <= stats.raw_cosine() <= 1.0


# ------------------------------------------------- layerwise vs concatenated


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_layerwise_accumulation_matches_concatenated_vector(dtype):
    """The whole point of the layerwise path: identical numbers, no d-vector."""
    torch.manual_seed(0)
    shapes = [(16, 64), (64, 16), (7,), (3, 3, 3), (1024,)]
    named_grads = [(f"layer{i}.lora_A.weight", torch.randn(*s).to(dtype))
                   for i, s in enumerate(shapes)]
    direction = {
        name: torch.randn_like(grad).to(dtype) for name, grad in named_grads
    }

    stats = accumulate_gradient_stats(named_grads, direction=direction)
    norm_sq, dot, dir_norm_sq = concatenated_gradient_stats(named_grads, direction)

    assert stats.norm_sq == pytest.approx(norm_sq, rel=1e-12, abs=1e-12)
    assert stats.dot_direction == pytest.approx(dot, rel=1e-12, abs=1e-12)
    assert stats.direction_norm_sq == pytest.approx(dir_norm_sq, rel=1e-12, abs=1e-12)


def test_layerwise_matches_concatenated_with_a_missing_gradient():
    named_grads = [
        ("a.lora_A.weight", torch.tensor([1.0, 2.0])),
        ("b.weight", None),
        ("c.lora_B.weight", torch.tensor([3.0])),
    ]
    direction = {
        "a.lora_A.weight": torch.tensor([1.0, 1.0]),
        "b.weight": torch.tensor([5.0, 5.0]),
        "c.lora_B.weight": torch.tensor([2.0]),
    }
    stats = accumulate_gradient_stats(named_grads, direction=direction)
    norm_sq, dot, dir_norm_sq = concatenated_gradient_stats(named_grads, direction)

    assert stats.norm_sq == pytest.approx(norm_sq)
    assert stats.dot_direction == pytest.approx(dot)
    assert stats.direction_norm_sq == pytest.approx(dir_norm_sq)
    assert stats.n_missing_grad == 1


def test_float64_accumulation_beats_float32_on_a_long_vector():
    """A float32 running sum drifts where the float64 one does not.

    The reference is the exact sum of the *stored float32 values*, not of the
    decimal literal -- 1e-3 is not representable in binary32, and comparing
    against the literal would measure the cast rather than the accumulation.
    """
    n = 500_000
    grad = torch.full((n,), 1e-3, dtype=torch.float32)
    stored = float(grad[0])
    exact = n * stored * stored

    stats = accumulate_gradient_stats([("w", grad)])
    assert stats.norm_sq == pytest.approx(exact, rel=1e-12)

    # The naive float32 reduction over the same input is no better.
    float32_sum = float(torch.dot(grad, grad))
    assert abs(stats.norm_sq - exact) <= abs(float32_sum - exact)


# ------------------------------------------------------------------ edge cases


def test_zero_gradient_gives_zero_cosine_and_sets_the_flag():
    stats = accumulate_gradient_stats(
        [("w", torch.zeros(10))], direction={"w": torch.ones(10)}
    )
    assert stats.norm_sq == 0.0
    assert stats.raw_cosine() == 0.0
    assert stats.zero_norm() is True


def test_zero_direction_gives_zero_cosine():
    stats = accumulate_gradient_stats(
        [("w", torch.ones(10))], direction={"w": torch.zeros(10)}
    )
    assert stats.raw_cosine() == 0.0
    assert stats.zero_norm() is True


def test_no_direction_leaves_cosine_at_zero_without_claiming_a_measurement():
    stats = accumulate_gradient_stats([("w", torch.ones(4))])
    assert stats.direction_norm_sq == 0.0
    assert stats.raw_cosine() == 0.0


def test_nonfinite_gradient_is_counted_and_propagates():
    grad = torch.tensor([1.0, float("nan"), 3.0])
    stats = accumulate_gradient_stats([("w", grad)])
    assert stats.n_nonfinite_tensors == 1
    assert math.isnan(stats.norm_sq)
    assert stats.is_finite is False


def test_infinite_gradient_is_counted():
    stats = accumulate_gradient_stats([("w", torch.tensor([float("inf"), 1.0]))])
    assert stats.n_nonfinite_tensors == 1
    assert stats.is_finite is False


def test_shape_mismatch_between_gradient_and_direction_raises():
    with pytest.raises(ValueError, match="Shape mismatch"):
        accumulate_gradient_stats(
            [("w", torch.ones(4))], direction={"w": torch.ones(5)}
        )


def test_empty_input_is_all_zeros_not_a_crash():
    stats = accumulate_gradient_stats([])
    assert stats.norm_sq == 0.0
    assert stats.n_params == 0
    assert stats.raw_cosine() == 0.0


# ------------------------------------------------------------ LoRA / non-LoRA


@pytest.mark.parametrize(
    "name,expected",
    [
        ("base_model.model.layers.0.self_attn.q_proj.lora_A.default.weight", True),
        ("base_model.model.layers.0.self_attn.q_proj.lora_B.default.weight", True),
        ("base_model.model.layers.0.self_attn.q_proj.base_layer.weight", False),
        ("model.embed_tokens.weight", False),
        ("LORA_A.weight", True),  # matching is case-insensitive
        ("model.lm_head.weight", False),
    ],
)
def test_lora_name_classification(name, expected):
    assert is_lora_param(name) is expected


def test_lora_and_non_lora_accumulators_partition_the_gradient():
    named_grads = [
        ("l.lora_A.weight", torch.tensor([1.0, -1.0])),        # abs sum 2, sq 2, numel 2
        ("l.lora_B.weight", torch.tensor([2.0])),              # abs sum 2, sq 4, numel 1
        ("l.base_layer.weight", torch.tensor([-3.0, 3.0, 0.0])),  # abs 6, sq 18, numel 3
    ]
    stats = accumulate_gradient_stats(named_grads)

    assert stats.numel_lora == 3
    assert stats.numel_non_lora == 3
    assert stats.abs_sum_lora == pytest.approx(4.0)
    assert stats.abs_sum_non_lora == pytest.approx(6.0)
    assert stats.sq_sum_lora == pytest.approx(6.0)
    assert stats.sq_sum_non_lora == pytest.approx(18.0)
    assert stats.sq_sum_lora + stats.sq_sum_non_lora == pytest.approx(stats.norm_sq)
    assert stats.n_tensors_lora == 2
    assert stats.n_tensors_non_lora == 1


def test_tensor_weighted_accumulator_averages_per_tensor_means():
    named_grads = [
        ("a.lora_A.weight", torch.tensor([1.0, 1.0, 1.0, 1.0])),  # mean |g| = 1
        ("b.lora_B.weight", torch.tensor([3.0])),                 # mean |g| = 3
    ]
    stats = accumulate_gradient_stats(named_grads)
    assert stats.tensor_abs_mean_sum_lora == pytest.approx(4.0)
    assert stats.n_tensors_lora == 2
    # Parameter-weighted mean is 7/5 = 1.4; tensor-weighted is 4/2 = 2.0.
    assert stats.abs_sum_lora / stats.numel_lora == pytest.approx(1.4)


# --------------------------------------------------------- parameter selection


class _FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q_lora_A = torch.nn.Parameter(torch.zeros(2, 3))
        self.q_lora_B = torch.nn.Parameter(torch.zeros(3, 2))
        self.backbone = torch.nn.Parameter(torch.zeros(4), requires_grad=False)
        self.head = torch.nn.Parameter(torch.zeros(5), requires_grad=False)


def test_scope_trainable_selects_only_requires_grad():
    names = select_param_names(_FakeModel(), ParamScope.TRAINABLE)
    assert names == ["q_lora_A", "q_lora_B"]


def test_scope_lora_selects_by_name():
    names = select_param_names(_FakeModel(), ParamScope.LORA)
    assert names == ["q_lora_A", "q_lora_B"]


def test_scope_full_selects_everything_including_frozen():
    names = select_param_names(_FakeModel(), ParamScope.FULL)
    assert names == ["q_lora_A", "q_lora_B", "backbone", "head"]


def test_scope_named_subset_uses_globs():
    names = select_param_names(_FakeModel(), ParamScope.NAMED_SUBSET, ["*lora_B*", "head"])
    assert names == ["q_lora_B", "head"]


def test_named_subset_without_patterns_raises():
    with pytest.raises(ValueError, match="name_patterns"):
        select_param_names(_FakeModel(), ParamScope.NAMED_SUBSET)


def test_empty_selection_raises_with_an_actionable_message():
    model = _FakeModel()
    for param in model.parameters():
        param.requires_grad = False
    with pytest.raises(ValueError, match="is_trainable=True"):
        select_param_names(model, ParamScope.TRAINABLE)


# ------------------------------------------------------- unlearning direction


def test_parameter_difference_is_unlearned_minus_finetuned():
    unlearned, finetuned = _FakeModel(), _FakeModel()
    with torch.no_grad():
        unlearned.q_lora_A.fill_(3.0)
        finetuned.q_lora_A.fill_(1.0)
    direction = parameter_difference(unlearned, finetuned, ["q_lora_A"])
    assert torch.allclose(direction["q_lora_A"], torch.full((6,), 2.0))


def test_parameter_difference_reports_the_missing_name():
    unlearned = _FakeModel()

    class _Other(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.something_else = torch.nn.Parameter(torch.zeros(2, 3))

    with pytest.raises(KeyError, match="q_lora_A"):
        parameter_difference(unlearned, _Other(), ["q_lora_A"])


def test_direction_norm_accumulates_across_parameters():
    direction = {"a": torch.tensor([3.0, 0.0]), "b": torch.tensor([4.0])}
    assert direction_norm(direction) == pytest.approx(5.0)
