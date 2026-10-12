"""The stabilisation constraints, on a tiny model with no downloads.

What can go wrong in a stabilisation study is that the "stabilised" arm is not
actually constrained, or that the unconstrained arm is not actually the same
loop. Both are checked here, because either one silently turns the comparison
into a comparison of two different methods.
"""

from __future__ import annotations

import math

import pytest
import torch

from deletion_signals.stabilize import project_to_trust_region
from deletion_signals.train import _displacement_norm, _parameter_snapshot


class Tiny(torch.nn.Module):
    def __init__(self, value=0.0):
        super().__init__()
        self.a = torch.nn.Parameter(torch.tensor([value, value]))
        self.b = torch.nn.Parameter(torch.tensor([value]))


def _move(model, delta):
    with torch.no_grad():
        for param in model.parameters():
            param.add_(delta)


# --------------------------------------------------------------- trust region


def test_no_projection_happens_inside_the_ball():
    model = Tiny()
    reference = _parameter_snapshot(model)
    _move(model, 0.1)                       # ||delta|| = sqrt(3)*0.1 = 0.173
    assert project_to_trust_region(model, reference, radius=1.0) is None
    assert _displacement_norm(model, reference) == pytest.approx(0.1 * math.sqrt(3))


def test_a_run_outside_the_ball_is_pulled_back_to_its_surface():
    model = Tiny()
    reference = _parameter_snapshot(model)
    _move(model, 1.0)                       # ||delta|| = sqrt(3) = 1.732
    before = project_to_trust_region(model, reference, radius=0.5)

    assert before == pytest.approx(math.sqrt(3))
    assert _displacement_norm(model, reference) == pytest.approx(0.5)


def test_projection_preserves_the_direction_of_travel():
    """A trust region constrains how far the run goes, not where it is going.
    Rotating the update would make the constrained arm optimise something else."""
    model = Tiny()
    reference = _parameter_snapshot(model)
    with torch.no_grad():
        model.a.add_(torch.tensor([3.0, 4.0]))   # ||.|| = 5 in the a-block

    project_to_trust_region(model, reference, radius=1.0)

    direction = model.a.detach() - reference["a"]
    assert direction[1] / direction[0] == pytest.approx(4.0 / 3.0)
    assert float(direction.norm()) == pytest.approx(1.0)


def test_projection_is_idempotent():
    model = Tiny()
    reference = _parameter_snapshot(model)
    _move(model, 1.0)
    project_to_trust_region(model, reference, radius=0.5)
    assert project_to_trust_region(model, reference, radius=0.5) is None


def test_a_model_that_has_not_moved_is_left_alone():
    model = Tiny()
    reference = _parameter_snapshot(model)
    assert project_to_trust_region(model, reference, radius=0.5) is None


# ------------------------------------------------------------- the knobs

def _settings(**overrides):
    from deletion_signals.config import load_config

    config = load_config("arxivmia_confirm_stabilize.yaml")
    settings = config["unlearning"]["ga_gd_stabilized"]
    settings.update(overrides)
    return settings


def test_every_stabiliser_is_off_by_default_so_the_control_arm_is_plain_ga_gd():
    """With all four knobs at zero the stabilised loop must be ordinary GA+GD.
    If the default arm were already constrained, every comparison in Block C
    would be against the wrong baseline."""
    settings = _settings()
    assert float(settings["retain_kl_weight"]) == 0.0
    assert float(settings["trust_region_radius"]) == 0.0
    assert float(settings["utility_guard_ratio"]) == 0.0
    assert float(settings["max_grad_norm"]) == 1.0     # the base GA+GD value


def test_the_stabilisation_config_runs_at_a_strength_that_collapses():
    """Block C asks whether a constraint rescues a collapsing setting. Applying
    it at a setting that was already fine would answer nothing."""
    settings = _settings()
    assert float(settings["ascent_lr"]) == 5.0e-5
    assert float(settings["descent_lr"]) == 2.0e-5


def test_the_guard_probes_often_enough_to_stop_before_the_collapse_completes():
    settings = _settings()
    assert 0 < int(settings["probe_every"]) <= 10


def test_the_method_is_registered_under_its_own_name():
    import deletion_signals.stabilize  # noqa: F401 - registers on import
    from deletion_signals.train import available_methods, get_unlearning_method

    assert "ga_gd_stabilized" in available_methods()
    assert get_unlearning_method("ga_gd_stabilized") is not None
    # And it has NOT replaced the unconstrained method it is compared against.
    assert "ga_gd" in available_methods()


# ------------------------------------------------------------------ Block D


def test_the_scrub_grid_starts_above_the_setting_that_did_nothing():
    """The exploratory SCRUB run ended with final_forget_kl = 0.098: the forget
    objective barely moved, so no conclusion about SCRUB was ever licensed."""
    from deletion_signals.config import load_config

    settings = load_config("arxivmia_confirm_scrub.yaml")["unlearning"]["scrub"]
    assert float(settings["forget_kl_weight"]) >= 1.0    # was 0.1
    assert float(settings["retain_ce_weight"]) == 1.0    # published relative weight
    assert float(settings["retain_kl_weight"]) == 1.0
