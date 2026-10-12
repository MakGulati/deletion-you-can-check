"""Configuration: defaults, deep merge, validation, and effective-config capture.

Every knob the evaluation exposes lives here with a default, so a config file
only has to state what it changes. The merged result -- the *effective* config --
is what gets recorded in each output file's metadata, so a row can always be
traced to the exact settings that produced it, not to whatever the config file
on disk says today.
"""

from __future__ import annotations

import copy
import os
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .gradients import ParamScope
from .scores import DEFAULT_EPS, FumaNorm, FumaWeighting

CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")

# Where the routine-update control draws its training records.
#
# `test_tail`   the slice of D_test beyond the scoring cap. The only disjoint
#               slice ArxivMIA has -- its 1000 members are fully consumed by the
#               split -- but the records are NON-MEMBERS, so the update also
#               nudges the model toward the non-member distribution. Declared as
#               a confound wherever it is used.
# `member_tail` members the split never drew. Needs a corpus with more members
#               than data.train_samples; BookMIA has 4935 against a draw of
#               1000. Same population as the training data, so no distribution
#               shift, which is what makes it the clean version of the control.
ROUTINE_UPDATE_SOURCES = ("test_tail", "member_tail")

# Which model supplies the reference measurement for each calibrated family.
REFERENCE_BASE = "pretrained_base"
REFERENCE_FINETUNED = "finetuned_pre_unlearning"
REFERENCE_NONE = "none"

DEFAULTS: Dict[str, Any] = {
    "seed": 42,
    "device": "cuda",
    # "lora" -> QLoRA 4-bit backbone with PEFT adapters; "full" -> full fine-tune.
    "training_mode": "lora",
    "model": {
        "name": "meta-llama/Llama-3.2-1B",
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": "bfloat16",
        "torch_dtype": "bfloat16",
        "trust_remote_code": True,
    },
    "lora": {
        "r": 16,
        "alpha": 32,
        "dropout": 0.05,
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
        "bias": "none",
        "task_type": "CAUSAL_LM",
    },
    "data": {
        "name": "swj0419/WikiMIA",
        "split": None,          # None -> inferred from the dataset name
        "text_field": None,     # None -> inferred from the dataset's columns
        "label_field": "label",
        "train_samples": 2000,
        "test_samples": 1000,
        "forget_ratio": 0.2,
        "max_length": 512,
        "manifest": None,       # path to a split manifest; overrides the loader
    },
    "training": {
        "epochs": 6,
        "batch_size": 8,
        "gradient_accumulation_steps": 4,
        "learning_rate": 2.0e-4,
        "warmup_steps": 100,
        "max_grad_norm": 1.0,
        "logging_steps": 10,
        "gradient_checkpointing": False,
    },
    "unlearning": {
        "ga_gd": {
            "steps": 100,
            "ascent_lr": 1.0e-5,
            "descent_lr": 2.0e-5,
            "batch_size": 4,
            "max_grad_norm": 1.0,
            "min_steps": 20,
            "target_loss_ratio": 2.0,
            "early_stop_patience": 5,
            "collapse_loss": 5.0,
            # Trajectory instrumentation. The endpoint cannot say whether
            # membership evidence disappeared before, with, or after utility
            # collapsed; a probe every `probe_every` steps on a fixed slice of
            # each population can. 0 disables it.
            "probe_every": 10,
            "probe_samples": 32,
        },
        "scrub": {
            "steps": 100,
            "learning_rate": 1.0e-5,
            "batch_size": 4,
            "kl_temperature": 4.0,
            "forget_kl_weight": 0.1,
            "forget_entropy_weight": 0.5,
            "retain_ce_weight": 1.0,
            "retain_kl_weight": 1.0,
            "max_grad_norm": 1.0,
        },
        # NPO-RT (Zhang et al. 2024, arXiv:2404.05868 Eq. 3). See npo.py for the
        # equation, the sign convention and the reference model.
        "npo": {
            "steps": 100,
            "learning_rate": 2.8e-5,
            "batch_size": 4,
            # Temperature on the sequence log-ratio. The published default;
            # beta -> 0 recovers plain gradient ascent.
            "beta": 0.1,
            # lambda on the retain cross-entropy term, making this NPO-RT
            # rather than bare NPO.
            "retain_weight": 1.0,
            # sum -> the paper's sequence-level derivation and the official
            # implementation. mean -> this repository's usual token-mean
            # convention. They differ by up to max_length, which moves beta's
            # effective scale by two orders of magnitude, so the choice is
            # explicit and is recorded in every output row.
            "log_ratio_reduction": "sum",
            "max_grad_norm": 1.0,
            "collapse_loss": 20.0,
            "probe_every": 10,
            "probe_samples": 32,
        },
    },
    # A legitimate, non-deletion checkpoint change: the nuisance condition the
    # null refit cannot supply. Consumed by routine_update.py, which asserts
    # that its data are disjoint from every scored population before training.
    "routine_update": {
        "learning_rate": 5.0e-5,
        "epochs": 3,
        "batch_size": 8,
        "gradient_accumulation_steps": 4,
        "max_grad_norm": 1.0,
        # Zero, unlike the fine-tune's 100: an update of ~18 optimiser steps
        # under a 100-step warmup would never reach the learning rate it
        # declares.
        "warmup_steps": 0,
        # Which records the update trains on. "test_tail" takes the slice of
        # D_test beyond the scoring cap -- the only records in ArxivMIA that no
        # model in this study has seen and that no run scores.
        "source": "test_tail",
        "n_samples": 200,
    },
    "scoring": {
        # Which parameters the gradient statistics range over.
        # trainable | lora | full | named_subset
        "param_scope": "trainable",
        "param_name_patterns": None,   # fnmatch globs, for named_subset
        "eps": DEFAULT_EPS,
        "max_length": 512,
        # Cap on evaluated samples per population. None means all of them, an
        # int caps every population alike, and a mapping caps them
        # independently ({test: 300} leaves forget and retain untouched).
        "max_samples_per_population": None,
        # How the unlearning direction is obtained. param_diff is the only
        # source this evaluation uses; the field exists so a run that used
        # something else cannot be mistaken for one that did not.
        "unlearning_direction": "param_diff",
        "output_path": None,
    },
    "scores": {
        "loss_unlearned": {"enabled": True},
        "loss_calibrated": {
            "enabled": True,
            # Difficulty calibration needs forward passes only, so the 4-bit
            # pretrained base is fine and is the right reference: it is the
            # model that never saw any of this data.
            "reference": REFERENCE_BASE,
        },
        # An independent second reference, not an alternative setting of
        # loss_calibrated.reference: it compares the unlearned model against the
        # pre-unlearning fine-tuned model, which isolates what unlearning did
        # rather than what membership did. Off by default -- it costs one extra
        # forward pass per sample, and only runs that hold theta_finetuned can
        # produce it.
        "loss_vs_finetuned": {"enabled": False},
        "grad_norm_sq": {"enabled": True},
        "grad_calibrated": {
            "enabled": True,
            # LoRA mode: the pretrained base has no adapters and a 4-bit
            # backbone, so ||grad L_base||^2 is not computable over the same
            # parameter space. The fine-tuned pre-unlearning model is, and it
            # isolates the unlearning effect. Full fine-tuning may use either.
            "reference": REFERENCE_FINETUNED,
        },
        "fuma": {
            "enabled": True,
            # shadow_bf16 -> load an unquantised copy carrying the same
            #                adapters and take BOTH the numerator and the
            #                denominator from it, so the ratio lives in one
            #                parameter space rather than straddling a
            #                4-bit/bf16 boundary.
            # same_model   -> take both from the scored model. Only valid when
            #                non-LoRA parameters actually receive gradients.
            # none         -> do not compute it; rows record the reason.
            "denominator_source": "shadow_bf16",
            "norm": FumaNorm.L1_MEAN_ABS.value,
            "weighting": FumaWeighting.PARAMETER.value,
        },
        "sif": {
            # Off by default: it is an ablation, and the Fisher path needs the
            # concatenated gradient vector that every other score avoids.
            "enabled": False,
            "method": "gradient",   # fisher | gradient
            "damping": 1.0e-3,
            "fisher_rank": 200,
            "fisher_device": "cpu",
            "beta": 1.0,
        },
    },
    "evaluation": {
        "tasks": [
            "forget_vs_retain",
            "member_vs_unseen",
            "forget_vs_unseen",
            "forget_vs_rest",
            "retain_vs_unseen",
        ],
        "primary_task": "forget_vs_retain",
        "fpr_targets": [0.001, 0.01, 0.05],
        "n_bootstrap": 1000,
        "confidence": 0.95,
        "n_permutations": 10000,
        "orientations_path": None,   # None -> configs/orientations.yaml
        "seeds": [42, 43, 44],
        "methods": ["ga_gd", "scrub"],
        "correlations": [
            ["sif", "grad_norm_sq"],
            ["alignment_cos_sq", "grad_norm_sq"],
            ["fuma_lora_ratio", "alignment_cos_sq"],
            ["loss_ratio", "loss_unlearned"],
            ["grad_ratio", "grad_norm_sq"],
            # Diagnostic for the two loss families: a near-1 correlation would
            # mean the pre-unlearning reference is telling us nothing the
            # difficulty reference did not already say.
            ["loss_ratio", "loss_ratio_vs_finetuned"],
        ],
    },
    "paths": {
        "checkpoints": "./ds_checkpoints",
        "results": "./ds_results",
    },
}


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` into ``base``, returning a new dict.

    Mappings merge key by key; every other type, lists included, is replaced
    outright. A config that sets ``lora.target_modules`` means "use exactly
    these", not "append to the defaults".
    """
    merged = dict(copy.deepcopy(dict(base)))
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(
    path: Optional[str] = None, overrides: Optional[Mapping[str, Any]] = None
) -> Dict[str, Any]:
    """Load a YAML config on top of :data:`DEFAULTS` and validate the result."""
    config = copy.deepcopy(DEFAULTS)

    if path:
        import yaml

        resolved = path
        if not os.path.exists(resolved):
            candidate = os.path.join(CONFIG_DIR, path)
            if os.path.exists(candidate):
                resolved = candidate
            else:
                raise FileNotFoundError(
                    f"Config {path!r} not found (also looked in {CONFIG_DIR})"
                )
        with open(resolved, "r") as handle:
            config = deep_merge(config, yaml.safe_load(handle) or {})
        config["_config_path"] = os.path.abspath(resolved)

    if overrides:
        config = deep_merge(config, overrides)

    validate_config(config)
    return config


def is_full_finetuning(config: Mapping[str, Any]) -> bool:
    return str(config.get("training_mode", "lora")).lower() == "full"


def validate_config(config: Mapping[str, Any]) -> None:
    """Reject configurations whose scores could not be computed as written.

    These checks exist because every one of them otherwise surfaces hours into
    a GPU run, or -- worse -- as a plausible-looking number computed from the
    wrong thing.
    """
    scoring = config.get("scoring", {})
    scores = config.get("scores", {})
    full_ft = is_full_finetuning(config)

    scope = ParamScope(scoring.get("param_scope", "trainable"))
    if scope is ParamScope.NAMED_SUBSET and not scoring.get("param_name_patterns"):
        raise ValueError(
            "scoring.param_scope='named_subset' requires "
            "scoring.param_name_patterns"
        )
    if scope is ParamScope.LORA and full_ft:
        raise ValueError(
            "scoring.param_scope='lora' is impossible under training_mode='full': "
            "a fully fine-tuned checkpoint has no adapter parameters."
        )

    eps = float(scoring.get("eps", DEFAULT_EPS))
    if eps <= 0:
        raise ValueError(f"scoring.eps must be positive, got {eps}")

    validate_sample_cap(scoring.get("max_samples_per_population"))

    direction = scoring.get("unlearning_direction", "param_diff")
    if direction != "param_diff":
        raise ValueError(
            f"scoring.unlearning_direction={direction!r} is not supported; this "
            f"evaluation defines ghat = theta_unlearned - theta_finetuned "
            f"('param_diff'), which requires white-box access to both "
            f"checkpoints."
        )

    grad_ref = scores.get("grad_calibrated", {}).get("reference", REFERENCE_FINETUNED)
    if scores.get("grad_calibrated", {}).get("enabled", True):
        if grad_ref not in (REFERENCE_BASE, REFERENCE_FINETUNED, REFERENCE_NONE):
            raise ValueError(f"Unknown grad_calibrated.reference {grad_ref!r}")
        if grad_ref == REFERENCE_BASE and not full_ft:
            raise ValueError(
                "scores.grad_calibrated.reference='pretrained_base' is not "
                "computable in LoRA mode: the pretrained base has no adapter "
                "parameters and its backbone is 4-bit quantised, so "
                "||grad L_base||^2 does not exist over the scored parameter "
                "space. Use 'finetuned_pre_unlearning'."
            )

    loss_ref = scores.get("loss_calibrated", {}).get("reference", REFERENCE_BASE)
    if scores.get("loss_calibrated", {}).get("enabled", True) and loss_ref not in (
        REFERENCE_BASE,
        REFERENCE_FINETUNED,
        REFERENCE_NONE,
    ):
        raise ValueError(f"Unknown loss_calibrated.reference {loss_ref!r}")

    fuma = scores.get("fuma", {})
    if fuma.get("enabled", True):
        source = fuma.get("denominator_source", "shadow_bf16")
        if source not in ("shadow_bf16", "same_model", "none"):
            raise ValueError(f"Unknown scores.fuma.denominator_source {source!r}")
        FumaNorm(fuma.get("norm", FumaNorm.L1_MEAN_ABS.value))
        weighting = FumaWeighting(fuma.get("weighting", FumaWeighting.PARAMETER.value))
        if weighting is FumaWeighting.TENSOR and FumaNorm(
            fuma.get("norm", FumaNorm.L1_MEAN_ABS.value)
        ) is not FumaNorm.L1_MEAN_ABS:
            raise ValueError(
                "scores.fuma.weighting='tensor' is only defined together with "
                "norm='l1_mean_abs'"
            )
        if source == "shadow_bf16" and full_ft:
            raise ValueError(
                "scores.fuma.denominator_source='shadow_bf16' is pointless under "
                "training_mode='full': there is no quantised backbone to shadow, "
                "and no LoRA parameters to form the numerator. Set it to 'none'."
            )

    sif = scores.get("sif", {})
    if sif.get("enabled", False):
        if sif.get("method") not in ("fisher", "gradient"):
            raise ValueError(
                f"scores.sif.method must be 'fisher' or 'gradient', got "
                f"{sif.get('method')!r}"
            )
        if sif.get("method") == "fisher":
            if full_ft:
                raise ValueError(
                    "scores.sif.method='fisher' is not usable under full "
                    "fine-tuning: G is n x d with d the full parameter count. "
                    "Use 'gradient'."
                )
            if float(sif.get("damping", 0.0)) <= 0:
                raise ValueError(
                    "scores.sif.damping must be positive to keep (F + lambda I) "
                    "invertible"
                )

    # NPO. Every one of these otherwise surfaces as a plausible-looking number
    # computed from a different objective than the one the paper defines.
    npo = config.get("unlearning", {}).get("npo", {})
    if npo:
        if float(npo.get("beta", 0.1)) <= 0:
            raise ValueError(
                f"unlearning.npo.beta must be positive -- it is a temperature on "
                f"the sequence log-ratio, and the objective's 2/beta prefactor "
                f"diverges at 0. Got {npo.get('beta')!r}."
            )
        if float(npo.get("retain_weight", 1.0)) < 0:
            raise ValueError(
                f"unlearning.npo.retain_weight must be non-negative; a negative "
                f"lambda would perform gradient ASCENT on the retain set, which "
                f"is not NPO-RT. Got {npo.get('retain_weight')!r}."
            )
        reduction = str(npo.get("log_ratio_reduction", "sum"))
        if reduction not in ("sum", "mean"):
            raise ValueError(
                f"unlearning.npo.log_ratio_reduction must be 'sum' (the paper's "
                f"sequence-level form) or 'mean' (this repository's token-mean "
                f"convention), got {reduction!r}"
            )

    routine = config.get("routine_update", {})
    if routine:
        source = str(routine.get("source", "test_tail"))
        if source not in ROUTINE_UPDATE_SOURCES:
            raise ValueError(
                f"routine_update.source={source!r} is not supported; valid "
                f"sources are {sorted(ROUTINE_UPDATE_SOURCES)}. 'test_tail' "
                f"takes the unscored tail of D_test -- the only disjoint slice "
                f"ArxivMIA has, since its member pool is fully consumed by the "
                f"split. 'member_tail' takes members the split never drew, "
                f"which is cleaner but needs a corpus with more members than "
                f"data.train_samples (BookMIA has 4935)."
            )
        if int(routine.get("n_samples", 0)) <= 0:
            raise ValueError(
                f"routine_update.n_samples must be positive, got "
                f"{routine.get('n_samples')!r}"
            )

    evaluation = config.get("evaluation", {})
    for target in evaluation.get("fpr_targets", []):
        if not 0 < float(target) < 1:
            raise ValueError(f"evaluation.fpr_targets must lie in (0, 1), got {target}")
    if evaluation.get("primary_task") not in evaluation.get("tasks", []):
        raise ValueError(
            f"evaluation.primary_task={evaluation.get('primary_task')!r} is not "
            f"listed in evaluation.tasks"
        )


def validate_sample_cap(cap: Any) -> None:
    """Check ``scoring.max_samples_per_population``: None, an int, or a mapping.

    A mapping key that is not a population is rejected rather than ignored: a
    typo would otherwise silently leave that population uncapped, and the run
    would cost hours more than intended without saying so.
    """
    from .schema import POPULATIONS

    if cap is None:
        return
    if isinstance(cap, Mapping):
        unknown = sorted(set(cap) - set(POPULATIONS))
        if unknown:
            raise ValueError(
                f"scoring.max_samples_per_population has unknown population(s) "
                f"{unknown}; valid keys are {list(POPULATIONS)}"
            )
        for population, limit in cap.items():
            if limit is None:
                continue
            if not isinstance(limit, int) or isinstance(limit, bool) or limit < 0:
                raise ValueError(
                    f"scoring.max_samples_per_population[{population!r}] must be a "
                    f"non-negative integer or null, got {limit!r}"
                )
        return
    if not isinstance(cap, int) or isinstance(cap, bool) or cap < 0:
        raise ValueError(
            f"scoring.max_samples_per_population must be null, a non-negative "
            f"integer, or a per-population mapping, got {cap!r}"
        )


def enabled_scores(config: Mapping[str, Any]) -> List[str]:
    """Names of the score families switched on in ``config``."""
    scores = config.get("scores", {})
    return sorted(name for name, spec in scores.items() if spec.get("enabled", False))


def effective_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    """A deep copy suitable for embedding in output metadata."""
    return copy.deepcopy(dict(config))


def _parse_override_value(raw_value: str) -> Any:
    """Parse one ``--set`` value as a YAML scalar, rescuing bare exponents.

    PyYAML implements YAML 1.1, whose float pattern requires a decimal point and
    a signed exponent: ``1.0e-5`` parses as a float but ``1e-5`` parses as the
    *string* ``"1e-5"``. A learning rate written the way every paper writes it
    would then reach the optimiser as a string and fail deep inside a training
    loop, so a scalar that YAML calls a string but ``float`` accepts is taken as
    the number it plainly is.
    """
    import yaml

    value = yaml.safe_load(raw_value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return value
    return value


def apply_cli_overrides(
    config: Dict[str, Any], pairs: Optional[Sequence[str]]
) -> Dict[str, Any]:
    """Apply ``--set a.b.c=value`` overrides, parsing values as YAML scalars."""
    if not pairs:
        return config

    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"--set expects key=value, got {pair!r}")
        key, raw_value = pair.split("=", 1)
        value = _parse_override_value(raw_value)
        cursor: Dict[str, Any] = config
        parts = key.split(".")
        for part in parts[:-1]:
            if part not in cursor or not isinstance(cursor[part], dict):
                cursor[part] = {}
            cursor = cursor[part]
        cursor[parts[-1]] = value

    validate_config(config)
    return config
