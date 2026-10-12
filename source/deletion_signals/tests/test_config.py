"""Configuration: merging, overrides, and the checks that stop impossible runs."""

from __future__ import annotations

import copy

import pytest

from deletion_signals.config import (
    CONFIG_DIR,
    DEFAULTS,
    REFERENCE_BASE,
    REFERENCE_FINETUNED,
    apply_cli_overrides,
    deep_merge,
    effective_config,
    enabled_scores,
    is_full_finetuning,
    load_config,
    validate_config,
)


def _config(**overrides):
    config = copy.deepcopy(DEFAULTS)
    return deep_merge(config, overrides)


# -------------------------------------------------------------------- merging


def test_deep_merge_replaces_scalars_and_recurses_into_mappings():
    merged = deep_merge({"a": {"b": 1, "c": 2}}, {"a": {"b": 9}})
    assert merged == {"a": {"b": 9, "c": 2}}


def test_deep_merge_replaces_lists_outright():
    """A config that names target_modules means exactly those, not those plus
    the defaults."""
    merged = deep_merge({"a": [1, 2, 3]}, {"a": [9]})
    assert merged["a"] == [9]


def test_deep_merge_does_not_mutate_its_inputs():
    base = {"a": {"b": 1}}
    deep_merge(base, {"a": {"b": 2}})
    assert base == {"a": {"b": 1}}


def test_effective_config_is_a_copy():
    config = _config()
    snapshot = effective_config(config)
    snapshot["seed"] = 999
    assert config["seed"] != 999


# ------------------------------------------------------------------ shipped configs


@pytest.mark.parametrize(
    "name",
    [
        "llama_lora.yaml",
        "llama_full_ft.yaml",
        "smoke_cpu.yaml",
        "arxivmia_lora.yaml",
        "arxivmia_sweep.yaml",
        "arxivmia_sweep_smoke.yaml",
    ],
)
def test_shipped_configs_load_and_validate(name):
    config = load_config(name)
    assert config["_config_path"].startswith(CONFIG_DIR)
    validate_config(config)


def test_lora_config_calibrates_gradients_against_the_finetuned_model():
    config = load_config("llama_lora.yaml")
    assert config["scores"]["grad_calibrated"]["reference"] == REFERENCE_FINETUNED
    assert config["scores"]["loss_calibrated"]["reference"] == REFERENCE_BASE


def test_full_ft_config_marks_fuma_unavailable():
    config = load_config("llama_full_ft.yaml")
    assert is_full_finetuning(config)
    assert config["scores"]["fuma"]["denominator_source"] == "none"


def test_the_sweep_config_disables_early_stopping_and_matches_the_baseline():
    """Early stopping fires on the forget/retain loss ratio -- the very quantity
    the sweep varies -- so leaving it on would clamp every strong cell back to
    the same stopping point. And the 1e-5 cell has to reproduce the baseline, so
    everything outside the marked SWEEP blocks must match arxivmia_lora.yaml."""
    sweep = load_config("arxivmia_sweep.yaml")
    baseline = load_config("arxivmia_lora.yaml")

    ga_gd = sweep["unlearning"]["ga_gd"]
    assert ga_gd["min_steps"] > ga_gd["steps"]
    assert ga_gd["collapse_loss"] == 20.0
    assert ga_gd["steps"] == baseline["unlearning"]["ga_gd"]["steps"]
    assert ga_gd["descent_lr"] == baseline["unlearning"]["ga_gd"]["descent_lr"]

    for block in ("data", "model", "lora", "training"):
        assert sweep[block] == baseline[block], f"{block} drifted from the baseline"

    assert sweep["scores"]["loss_vs_finetuned"]["enabled"] is True
    assert sweep["scoring"]["max_samples_per_population"] == {"test": 300}


def test_missing_config_file_names_where_it_looked():
    with pytest.raises(FileNotFoundError, match="configs"):
        load_config("no_such_config.yaml")


# ----------------------------------------------------------------- validation


def test_gradient_calibration_against_the_base_is_rejected_in_lora_mode():
    """The pretrained base has no adapters and a 4-bit backbone, so the
    reference gradient does not exist over the scored parameter space."""
    config = _config(scores={"grad_calibrated": {"enabled": True, "reference": REFERENCE_BASE}})
    with pytest.raises(ValueError, match="not computable in LoRA mode"):
        validate_config(config)


def test_gradient_calibration_against_the_base_is_allowed_in_full_ft_mode():
    config = _config(
        training_mode="full",
        scores={
            "grad_calibrated": {"enabled": True, "reference": REFERENCE_BASE},
            "fuma": {"enabled": True, "denominator_source": "none"},
        },
    )
    validate_config(config)


def test_lora_param_scope_is_rejected_under_full_finetuning():
    config = _config(
        training_mode="full",
        scoring={"param_scope": "lora"},
        scores={"fuma": {"denominator_source": "none"}},
    )
    with pytest.raises(ValueError, match="no adapter parameters"):
        validate_config(config)


def test_named_subset_without_patterns_is_rejected():
    config = _config(scoring={"param_scope": "named_subset", "param_name_patterns": None})
    with pytest.raises(ValueError, match="param_name_patterns"):
        validate_config(config)


def test_a_nonpositive_eps_is_rejected():
    with pytest.raises(ValueError, match="eps must be positive"):
        validate_config(_config(scoring={"eps": 0.0}))


def test_an_unsupported_direction_source_is_rejected():
    config = _config(scoring={"unlearning_direction": "forget_grad"})
    with pytest.raises(ValueError, match="white-box"):
        validate_config(config)


def test_the_shadow_model_is_rejected_under_full_finetuning():
    config = _config(training_mode="full", scores={"fuma": {"denominator_source": "shadow_bf16"}})
    with pytest.raises(ValueError, match="pointless under"):
        validate_config(config)


def test_tensor_weighting_with_an_incompatible_norm_is_rejected():
    config = _config(scores={"fuma": {"weighting": "tensor", "norm": "l2"}})
    with pytest.raises(ValueError, match="l1_mean_abs"):
        validate_config(config)


def test_fisher_sif_is_rejected_under_full_finetuning():
    config = _config(
        training_mode="full",
        scores={
            "sif": {"enabled": True, "method": "fisher"},
            "fuma": {"denominator_source": "none"},
        },
    )
    with pytest.raises(ValueError, match="not usable under full"):
        validate_config(config)


def test_nonpositive_fisher_damping_is_rejected():
    config = _config(scores={"sif": {"enabled": True, "method": "fisher", "damping": 0.0}})
    with pytest.raises(ValueError, match="damping must be positive"):
        validate_config(config)


def test_a_per_population_sample_cap_is_accepted():
    validate_config(_config(scoring={"max_samples_per_population": {"test": 300}}))


def test_an_unknown_population_in_the_sample_cap_is_rejected():
    config = _config(scoring={"max_samples_per_population": {"tset": 300}})
    with pytest.raises(ValueError, match="unknown population"):
        validate_config(config)


def test_a_negative_sample_cap_is_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        validate_config(_config(scoring={"max_samples_per_population": -1}))


def test_a_non_numeric_sample_cap_is_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        validate_config(_config(scoring={"max_samples_per_population": {"test": "all"}}))


def test_an_out_of_range_fpr_target_is_rejected():
    with pytest.raises(ValueError, match="fpr_targets"):
        validate_config(_config(evaluation={"fpr_targets": [1.5]}))


def test_a_primary_task_that_is_not_evaluated_is_rejected():
    config = _config(evaluation={"tasks": ["member_vs_unseen"], "primary_task": "forget_vs_retain"})
    with pytest.raises(ValueError, match="primary_task"):
        validate_config(config)


# ------------------------------------------------------------- CLI overrides


def test_cli_override_sets_a_nested_key():
    config = apply_cli_overrides(_config(), ["scores.sif.enabled=true"])
    assert config["scores"]["sif"]["enabled"] is True


def test_cli_override_parses_numbers_and_lists():
    config = apply_cli_overrides(
        _config(), ["evaluation.n_bootstrap=250", "evaluation.fpr_targets=[0.01, 0.05]"]
    )
    assert config["evaluation"]["n_bootstrap"] == 250
    assert config["evaluation"]["fpr_targets"] == [0.01, 0.05]


def test_cli_override_reads_a_bare_exponent_as_a_number():
    """YAML 1.1 calls `1e-5` a string -- it wants a decimal point and a signed
    exponent. A learning rate written the way every paper writes it would then
    reach AdamW as a string and fail deep inside the training loop."""
    config = apply_cli_overrides(_config(), ["unlearning.ga_gd.ascent_lr=1e-5"])
    value = config["unlearning"]["ga_gd"]["ascent_lr"]
    assert isinstance(value, float)
    assert value == pytest.approx(1e-5)


def test_cli_override_leaves_genuine_strings_alone():
    config = apply_cli_overrides(_config(), ["data.text_field=abstract"])
    assert config["data"]["text_field"] == "abstract"


def test_cli_override_revalidates_and_rejects_an_impossible_combination():
    with pytest.raises(ValueError, match="not computable in LoRA mode"):
        apply_cli_overrides(_config(), ["scores.grad_calibrated.reference=pretrained_base"])


def test_cli_override_without_an_equals_sign_is_rejected():
    with pytest.raises(ValueError, match="key=value"):
        apply_cli_overrides(_config(), ["nonsense"])


def test_enabled_scores_lists_what_is_switched_on():
    names = enabled_scores(_config())
    assert "loss_unlearned" in names
    assert "fuma" in names
    assert "sif" not in names  # off by default: it is an ablation
