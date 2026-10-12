"""Score definitions: ratios, eps behaviour, FUMA, and not-applicable handling."""

from __future__ import annotations

import math

import pytest

from deletion_signals.scores import (
    DEFAULT_EPS,
    FumaNorm,
    FumaWeighting,
    combined_log_score,
    cos_sq,
    fuma_lora_ratio,
    grad_ratio,
    log_grad_ratio,
    loss_delta,
    loss_ratio,
    updated_parameter_ratio,
)


# ------------------------------------------------------------- calibrated loss


def test_loss_ratio_is_the_plain_quotient():
    assert loss_ratio(4.0, 2.0) == pytest.approx(2.0)


def test_loss_ratio_floors_the_denominator_at_eps():
    assert loss_ratio(1.0, 0.0, eps=1e-6) == pytest.approx(1e6)


def test_loss_ratio_is_none_without_a_reference():
    assert loss_ratio(1.0, None) is None


def test_loss_ratio_rejects_a_nonfinite_reference():
    assert loss_ratio(1.0, float("nan")) is None
    assert loss_ratio(float("inf"), 2.0) is None


def test_loss_delta_is_the_difference():
    assert loss_delta(4.0, 2.5) == pytest.approx(1.5)
    assert loss_delta(4.0, None) is None


# --------------------------------------------------- calibrated gradient norm


def test_grad_ratio_is_the_plain_quotient():
    assert grad_ratio(9.0, 3.0) == pytest.approx(3.0)


def test_grad_ratio_floors_the_denominator_at_eps():
    assert grad_ratio(1.0, 0.0, eps=1e-8) == pytest.approx(1e8)


def test_log_grad_ratio_matches_log_of_the_ratio_away_from_eps():
    a, b = 12.5, 2.5
    assert log_grad_ratio(a, b) == pytest.approx(math.log(a / b), rel=1e-9)


def test_log_grad_ratio_stays_finite_when_both_norms_underflow():
    """The additive eps is the point: the quotient form would return 1.0 here,
    implying 'no change' where the truth is 'no signal'."""
    value = log_grad_ratio(0.0, 0.0, eps=1e-12)
    assert value == pytest.approx(0.0)
    assert math.isfinite(value)


def test_log_grad_ratio_is_none_without_a_reference():
    assert log_grad_ratio(1.0, None) is None


# ------------------------------------------------------------------ alignment


def test_cos_sq_squares_and_clamps():
    assert cos_sq(0.5) == pytest.approx(0.25)
    assert cos_sq(-0.5) == pytest.approx(0.25)
    assert cos_sq(1.0) == pytest.approx(1.0)
    assert cos_sq(None) is None
    assert cos_sq(float("nan")) is None


def test_cos_sq_is_sign_invariant():
    assert cos_sq(0.3) == cos_sq(-0.3)


# ------------------------------------------------------------ combined score


def test_combined_log_score_is_the_log_space_product():
    assert combined_log_score(math.e, 0.25, beta=1.0) == pytest.approx(
        1.0 + math.log(0.25)
    )


def test_combined_log_score_with_beta_zero_is_alignment_alone():
    assert combined_log_score(1234.0, 0.25, beta=0.0) == pytest.approx(math.log(0.25))


def test_combined_log_score_sends_zero_alignment_to_minus_inf():
    assert combined_log_score(10.0, 0.0) == float("-inf")


def test_combined_log_score_sends_nonpositive_sif_to_minus_inf():
    assert combined_log_score(-1.0, 0.5) == float("-inf")


# ----------------------------------------------------------------------- FUMA


def _fuma(**overrides):
    kwargs = dict(
        abs_sum_lora=4.0,
        abs_sum_non_lora=6.0,
        sq_sum_lora=6.0,
        sq_sum_non_lora=18.0,
        numel_lora=2,
        numel_non_lora=3,
    )
    kwargs.update(overrides)
    return fuma_lora_ratio(**kwargs)


def test_fuma_parameter_weighted_mean_abs_is_the_default():
    value, reason = _fuma()
    # (4/2) / (6/3) = 2/2 = 1
    assert reason is None
    assert value == pytest.approx(1.0)


def test_fuma_l2_uses_the_root_mean_square():
    value, reason = _fuma(norm=FumaNorm.L2)
    # sqrt(6/2) / sqrt(18/3) = sqrt(3)/sqrt(6)
    assert reason is None
    assert value == pytest.approx(math.sqrt(3.0) / math.sqrt(6.0))


def test_fuma_l2_sq_uses_the_mean_square():
    value, reason = _fuma(norm=FumaNorm.L2_SQ)
    assert reason is None
    assert value == pytest.approx((6.0 / 2.0) / (18.0 / 3.0))


def test_fuma_tensor_weighting_averages_per_tensor_means():
    value, reason = _fuma(
        weighting=FumaWeighting.TENSOR,
        tensor_abs_mean_sum_lora=4.0,
        n_tensors_lora=2,
        tensor_abs_mean_sum_non_lora=6.0,
        n_tensors_non_lora=3,
    )
    assert reason is None
    assert value == pytest.approx((4.0 / 2.0) / (6.0 / 3.0))


def test_fuma_tensor_weighting_rejects_a_norm_it_cannot_express():
    with pytest.raises(ValueError, match="TENSOR"):
        _fuma(
            weighting=FumaWeighting.TENSOR,
            norm=FumaNorm.L2,
            tensor_abs_mean_sum_lora=1.0,
            n_tensors_lora=1,
        )


def test_fuma_is_not_applicable_without_lora_parameters():
    value, reason = _fuma(numel_lora=0, abs_sum_lora=0.0, sq_sum_lora=0.0)
    assert value is None
    assert "full fine-tuning" in reason


def test_fuma_is_not_applicable_without_non_lora_gradients():
    value, reason = _fuma(numel_non_lora=0, abs_sum_non_lora=0.0, sq_sum_non_lora=0.0)
    assert value is None
    assert "4-bit" in reason


def test_fuma_never_reports_a_misleading_zero_for_a_missing_group():
    """A not-applicable ratio must be None, not 0.0 -- 0.0 is a real value."""
    for overrides in ({"numel_lora": 0}, {"numel_non_lora": 0}):
        value, reason = _fuma(**overrides)
        assert value is None and reason


def test_fuma_floors_a_zero_denominator_rather_than_dividing_by_zero():
    value, reason = _fuma(abs_sum_non_lora=0.0, eps=1e-9)
    assert reason is None
    assert value == pytest.approx(2.0 / 1e-9)


def test_fuma_rejects_nonfinite_inputs():
    value, reason = _fuma(abs_sum_lora=float("nan"))
    assert value is None
    assert "not finite" in reason


# ---------------------------------------------- the deliberately-renamed variant


def test_updated_parameter_ratio_is_not_applicable_without_a_partition():
    value, reason = updated_parameter_ratio(None, None)
    assert value is None
    assert "partition" in reason


def test_updated_parameter_ratio_computes_when_given_one():
    value, reason = updated_parameter_ratio(6.0, 2.0)
    assert reason is None
    assert value == pytest.approx(3.0)


def test_default_eps_is_far_below_any_real_measurement():
    assert DEFAULT_EPS < 1e-9
