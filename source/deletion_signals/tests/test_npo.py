"""NPO: the objective, its gradient, and the loop that applies it.

The numerics are tested against hand-computed values rather than against
themselves. An unlearning objective that is subtly wrong -- a flipped sign, a
mean where the paper says sum, a reference model that is silently the current
model -- does not crash. It produces a plausible curve and a wrong conclusion,
and the only thing standing between that and the paper is this file.

Nothing here downloads a model or needs a GPU: a 7-token vocabulary and a
two-layer network exercise every path that can go quietly wrong.
"""

from __future__ import annotations

import copy
import math

import pytest
import torch
import torch.nn.functional as F

from deletion_signals.config import DEFAULTS, deep_merge, validate_config
from deletion_signals.npo import MEAN, SUM, npo_loss, sequence_log_probs


# --------------------------------------------------------------- tiny fixtures


class TinyOutput:
    def __init__(self, logits, loss=None):
        self.logits = logits
        self.loss = loss


class TinyCausalLM(torch.nn.Module):
    """A real (if minute) causal LM: embeddings, one projection, honest logits.

    Named to mimic PEFT so the LoRA-substring rule that decides trainability in
    the real pipeline is the same rule exercised here.
    """

    VOCAB = 7
    HIDDEN = 4

    def __init__(self, seed: int = 0):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.embed = torch.nn.Embedding(self.VOCAB, self.HIDDEN)
        self.lora_head = torch.nn.Parameter(
            torch.randn(self.HIDDEN, self.VOCAB, generator=generator) * 0.3
        )
        self.frozen_bias = torch.nn.Parameter(
            torch.randn(self.VOCAB, generator=generator) * 0.1, requires_grad=False
        )
        with torch.no_grad():
            self.embed.weight.copy_(
                torch.randn(self.VOCAB, self.HIDDEN, generator=generator) * 0.3
            )
        self.embed.weight.requires_grad = False
        self.config = type("Cfg", (), {"use_cache": False, "pad_token_id": 0})()

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        hidden = self.embed(input_ids)
        logits = hidden @ self.lora_head + self.frozen_bias
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1, :].reshape(-1, self.VOCAB),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        return TinyOutput(logits, loss)


def _batch(rows, pad_to=None):
    """``rows`` are token-id lists; shorter ones are padded and label-masked."""
    width = pad_to or max(len(r) for r in rows)
    input_ids, labels, mask = [], [], []
    for row in rows:
        padding = width - len(row)
        input_ids.append(row + [0] * padding)
        labels.append(row + [-100] * padding)
        mask.append([1] * len(row) + [0] * padding)
    return (
        torch.tensor(input_ids),
        torch.tensor(labels),
        torch.tensor(mask),
    )


# ------------------------------------------------------- the objective's value


def test_npo_loss_matches_the_published_equation_on_a_hand_computed_example():
    """L = -(2/beta) * mean log sigmoid(-beta * (log pi_theta - log pi_ref))."""
    beta = 0.25
    policy = torch.tensor([-3.0, -7.5])
    reference = torch.tensor([-2.0, -4.0])

    loss, _ = npo_loss(policy, reference, beta)

    # Computed by hand from the paper's Eq. 3, not from the implementation.
    expected_terms = []
    for p, r in zip([-3.0, -7.5], [-2.0, -4.0]):
        log_ratio = p - r
        expected_terms.append(math.log(1.0 / (1.0 + math.exp(beta * log_ratio))))
    expected = -(2.0 / beta) * (sum(expected_terms) / len(expected_terms))

    assert loss.item() == pytest.approx(expected, rel=1e-6)


def test_npo_loss_equals_the_log_one_plus_ratio_form():
    """The two forms in the paper are the same function; both are implemented."""
    beta = 0.4
    policy = torch.tensor([-2.5, -9.0, -1.0])
    reference = torch.tensor([-2.0, -3.0, -4.0])

    loss, _ = npo_loss(policy, reference, beta)

    log_ratio = policy - reference
    alternative = (2.0 / beta) * torch.log1p(torch.exp(beta * log_ratio)).mean()
    assert loss.item() == pytest.approx(alternative.item(), rel=1e-6)


def test_the_gradient_weight_is_two_sigmoid_beta_log_ratio():
    beta = 0.3
    policy = torch.tensor([-1.0, -6.0])
    reference = torch.tensor([-2.0, -2.0])

    _, weight = npo_loss(policy, reference, beta)

    expected = 2.0 * torch.sigmoid(beta * (policy - reference))
    assert torch.allclose(weight, expected, atol=1e-7)


def test_the_weight_is_exactly_one_at_initialisation():
    """theta = theta_ref means r = 0, so 2*sigmoid(0) = 1: NPO's step is GA's step.

    This is the anchor the whole preregistered learning-rate grid rests on. If it
    ever stops holding, the grid's justification goes with it.
    """
    values = torch.tensor([-4.0, -11.0, -2.5])
    _, weight = npo_loss(values, values.clone(), beta=0.1)
    assert torch.allclose(weight, torch.ones(3), atol=1e-9)


def test_the_weight_decays_to_zero_once_a_record_is_forgotten():
    """The collapse-avoidance mechanism, stated as a monotonicity property."""
    reference = torch.zeros(1)
    weights = [
        npo_loss(torch.tensor([r]), reference, beta=0.5)[1].item()
        for r in (2.0, 0.0, -2.0, -10.0, -50.0)
    ]
    assert weights == sorted(weights, reverse=True)
    assert weights[0] > 1.0          # not yet forgotten: pushed harder than GA
    assert weights[2] < 1.0          # forgetting: already backing off
    assert weights[-1] == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------- the sign


def test_minimising_the_objective_reduces_the_forget_likelihood():
    """The sign convention, tested as behaviour rather than asserted in a docstring.

    A flipped sign here would make the method a forget-set *reinforcer* while
    every loss curve still looked reasonable.
    """
    model = TinyCausalLM(seed=1)
    reference_model = copy.deepcopy(model)
    input_ids, labels, _ = _batch([[1, 2, 3, 4], [5, 6, 1, 2]])

    def forget_log_prob(m):
        with torch.no_grad():
            lp, _ = sequence_log_probs(m(input_ids=input_ids).logits, labels, SUM)
        return lp.sum().item()

    before = forget_log_prob(model)
    optimizer = torch.optim.SGD([model.lora_head], lr=0.5)
    for _ in range(5):
        optimizer.zero_grad()
        policy_lp, _ = sequence_log_probs(model(input_ids=input_ids).logits, labels, SUM)
        with torch.no_grad():
            ref_lp, _ = sequence_log_probs(
                reference_model(input_ids=input_ids).logits, labels, SUM
            )
        loss, _ = npo_loss(policy_lp, ref_lp, beta=0.1)
        loss.backward()
        optimizer.step()

    assert forget_log_prob(model) < before


def test_npo_first_step_matches_gradient_ascent_up_to_a_positive_scale():
    """At theta = theta_ref the two methods take the SAME direction.

    GA backpropagates ``-mean_token_nll``; NPO backpropagates its objective,
    whose gradient is ``2*sigmoid(0) * grad log pi``. The two differ only by a
    positive scalar, so their cosine is exactly 1 -- and because both are then
    clipped to the same norm, the first *step* is identical, not merely parallel.
    That identity is what licenses anchoring NPO's grid on GA's measured
    transition.
    """
    model = TinyCausalLM(seed=2)
    reference_model = copy.deepcopy(model)
    input_ids, labels, _ = _batch([[1, 2, 3, 4], [4, 3, 2, 1], [5, 6, 2, 3]])

    # --- GA's ascent gradient: d/dtheta of (-mean token NLL) ---------------
    model.zero_grad()
    ga_loss = model(input_ids=input_ids, labels=labels).loss
    (-ga_loss).backward()
    ga_grad = model.lora_head.grad.detach().clone().flatten()

    # --- NPO's forget gradient at theta = theta_ref -----------------------
    model.zero_grad()
    policy_lp, _ = sequence_log_probs(model(input_ids=input_ids).logits, labels, SUM)
    with torch.no_grad():
        ref_lp, _ = sequence_log_probs(
            reference_model(input_ids=input_ids).logits, labels, SUM
        )
    loss, weight = npo_loss(policy_lp, ref_lp, beta=0.1)
    loss.backward()
    npo_grad = model.lora_head.grad.detach().clone().flatten()

    assert torch.allclose(weight, torch.ones_like(weight), atol=1e-9)
    cosine = F.cosine_similarity(ga_grad, npo_grad, dim=0)
    assert cosine.item() == pytest.approx(1.0, abs=1e-5)

    # ...and identical once both are clipped to the same norm.
    assert torch.allclose(
        ga_grad / ga_grad.norm(), npo_grad / npo_grad.norm(), atol=1e-5
    )


# ------------------------------------------------------------------ beta


def test_small_beta_approaches_plain_gradient_ascent():
    """beta -> 0 freezes the weight at 1, which is exactly what GA does."""
    policy = torch.tensor([-3.0, -20.0])
    reference = torch.tensor([-2.0, -2.0])

    _, tiny_beta = npo_loss(policy, reference, beta=1e-6)
    assert torch.allclose(tiny_beta, torch.ones(2), atol=1e-4)


def test_larger_beta_cuts_a_forgotten_record_off_sooner():
    policy = torch.tensor([-8.0])
    reference = torch.tensor([-2.0])

    _, gentle = npo_loss(policy, reference, beta=0.05)
    _, sharp = npo_loss(policy, reference, beta=1.0)
    assert sharp.item() < gentle.item()


def test_a_nonpositive_beta_is_rejected():
    with pytest.raises(ValueError, match="beta must be positive"):
        npo_loss(torch.tensor([-1.0]), torch.tensor([-1.0]), beta=0.0)


def test_config_validation_rejects_a_nonpositive_beta():
    config = deep_merge(DEFAULTS, {"unlearning": {"npo": {"beta": -0.1}}})
    with pytest.raises(ValueError, match="npo.beta must be positive"):
        validate_config(config)


def test_config_validation_rejects_a_negative_retain_weight():
    config = deep_merge(DEFAULTS, {"unlearning": {"npo": {"retain_weight": -1.0}}})
    with pytest.raises(ValueError, match="gradient ASCENT on the retain set"):
        validate_config(config)


def test_config_validation_rejects_an_unknown_reduction():
    config = deep_merge(
        DEFAULTS, {"unlearning": {"npo": {"log_ratio_reduction": "median"}}}
    )
    with pytest.raises(ValueError, match="log_ratio_reduction"):
        validate_config(config)


# ------------------------------------------------------- sequence log-probs


def test_sequence_log_probs_sums_only_over_real_tokens():
    """Padding must not enter the sum, or r becomes a function of batch shape."""
    model = TinyCausalLM(seed=3)
    short_alone, _, _ = _batch([[1, 2, 3]])
    labels_alone = torch.tensor([[1, 2, 3]])

    # The same sequence, now padded because it shares a batch with a longer one.
    padded_ids, padded_labels, _ = _batch([[1, 2, 3], [1, 2, 3, 4, 5]])

    with torch.no_grad():
        alone, n_alone = sequence_log_probs(
            model(input_ids=short_alone).logits, labels_alone, SUM
        )
        padded, n_padded = sequence_log_probs(
            model(input_ids=padded_ids).logits, padded_labels, SUM
        )

    assert n_alone[0].item() == n_padded[0].item() == 2  # 3 tokens -> 2 predictions
    assert alone[0].item() == pytest.approx(padded[0].item(), rel=1e-6)


def test_sum_and_mean_reductions_differ_by_the_token_count():
    """The factor that moves beta's effective scale by two orders of magnitude."""
    model = TinyCausalLM(seed=4)
    input_ids, labels, _ = _batch([[1, 2, 3, 4, 5, 6]])

    with torch.no_grad():
        summed, n_tokens = sequence_log_probs(
            model(input_ids=input_ids).logits, labels, SUM
        )
        meaned, _ = sequence_log_probs(model(input_ids=input_ids).logits, labels, MEAN)

    assert summed[0].item() == pytest.approx(meaned[0].item() * n_tokens[0].item(), rel=1e-6)


def test_sequence_log_probs_rejects_an_unknown_reduction():
    model = TinyCausalLM(seed=5)
    input_ids, labels, _ = _batch([[1, 2, 3]])
    with pytest.raises(ValueError, match="log_ratio_reduction"):
        sequence_log_probs(model(input_ids=input_ids).logits, labels, "median")


def test_log_probs_are_negative_and_finite():
    model = TinyCausalLM(seed=6)
    input_ids, labels, _ = _batch([[1, 2, 3, 4], [5, 6, 1, 2]])
    with torch.no_grad():
        values, _ = sequence_log_probs(model(input_ids=input_ids).logits, labels, SUM)
    assert torch.all(torch.isfinite(values))
    assert torch.all(values < 0)


# ------------------------------------------------------- numerical stability


def test_the_loss_stays_finite_at_extreme_log_ratios():
    """logsigmoid, not log(sigmoid): the latter underflows to -inf here.

    A sequence-summed log-ratio of several hundred nats is ordinary once
    unlearning gets going, and an -inf loss would take the whole run with it.
    """
    beta = 0.1
    for offset in (-5000.0, -500.0, 500.0, 5000.0):
        policy = torch.tensor([offset], requires_grad=True)
        reference = torch.tensor([0.0])
        loss, weight = npo_loss(policy, reference, beta)
        assert math.isfinite(loss.item()), offset
        loss.backward()
        assert math.isfinite(policy.grad.item()), offset
        assert 0.0 <= weight.item() <= 2.0


def test_the_gradient_vanishes_rather_than_exploding_when_fully_forgotten():
    policy = torch.tensor([-5000.0], requires_grad=True)
    loss, _ = npo_loss(policy, torch.tensor([0.0]), beta=0.1)
    loss.backward()
    assert policy.grad.item() == pytest.approx(0.0, abs=1e-9)


# ------------------------------------------------------------ the reference


def test_the_reference_model_receives_no_gradient():
    model = TinyCausalLM(seed=7)
    reference_model = copy.deepcopy(model)
    for param in reference_model.parameters():
        param.requires_grad = False

    input_ids, labels, _ = _batch([[1, 2, 3, 4]])
    policy_lp, _ = sequence_log_probs(model(input_ids=input_ids).logits, labels, SUM)
    with torch.no_grad():
        ref_lp, _ = sequence_log_probs(
            reference_model(input_ids=input_ids).logits, labels, SUM
        )
    loss, _ = npo_loss(policy_lp, ref_lp, beta=0.1)
    loss.backward()

    assert all(p.grad is None for p in reference_model.parameters())
    assert model.lora_head.grad is not None


def test_the_reference_stays_fixed_while_the_policy_moves():
    """If the reference drifted with the policy, r would be 0 forever and NPO
    would silently become plain gradient ascent with no gate at all."""
    model = TinyCausalLM(seed=8)
    reference_model = copy.deepcopy(model)
    for param in reference_model.parameters():
        param.requires_grad = False
    input_ids, labels, _ = _batch([[1, 2, 3, 4]])

    with torch.no_grad():
        before, _ = sequence_log_probs(
            reference_model(input_ids=input_ids).logits, labels, SUM
        )

    optimizer = torch.optim.SGD([model.lora_head], lr=1.0)
    for _ in range(3):
        optimizer.zero_grad()
        policy_lp, _ = sequence_log_probs(model(input_ids=input_ids).logits, labels, SUM)
        with torch.no_grad():
            ref_lp, _ = sequence_log_probs(
                reference_model(input_ids=input_ids).logits, labels, SUM
            )
        npo_loss(policy_lp, ref_lp, beta=0.1)[0].backward()
        optimizer.step()

    with torch.no_grad():
        after, _ = sequence_log_probs(
            reference_model(input_ids=input_ids).logits, labels, SUM
        )
        moved, _ = sequence_log_probs(model(input_ids=input_ids).logits, labels, SUM)

    assert after.item() == pytest.approx(before.item(), rel=1e-9)
    assert moved.item() != pytest.approx(before.item(), rel=1e-6)


def test_frozen_parameters_are_untouched_by_an_npo_update():
    model = TinyCausalLM(seed=9)
    reference_model = copy.deepcopy(model)
    frozen_before = model.frozen_bias.detach().clone()
    embed_before = model.embed.weight.detach().clone()

    input_ids, labels, _ = _batch([[1, 2, 3, 4]])
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=0.1)
    for _ in range(3):
        optimizer.zero_grad()
        policy_lp, _ = sequence_log_probs(model(input_ids=input_ids).logits, labels, SUM)
        with torch.no_grad():
            ref_lp, _ = sequence_log_probs(
                reference_model(input_ids=input_ids).logits, labels, SUM
            )
        npo_loss(policy_lp, ref_lp, beta=0.1)[0].backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()

    assert torch.equal(model.frozen_bias, frozen_before)
    assert torch.equal(model.embed.weight, embed_before)
    assert not torch.equal(model.lora_head.detach(), reference_model.lora_head.detach())


# ----------------------------------------------------------- reproducibility


def _run_short_npo(seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    model = TinyCausalLM(seed=11)
    reference_model = copy.deepcopy(model)
    input_ids, labels, _ = _batch([[1, 2, 3, 4], [5, 6, 1, 2]])
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=0.05)
    for _ in range(4):
        optimizer.zero_grad()
        policy_lp, _ = sequence_log_probs(model(input_ids=input_ids).logits, labels, SUM)
        with torch.no_grad():
            ref_lp, _ = sequence_log_probs(
                reference_model(input_ids=input_ids).logits, labels, SUM
            )
        npo_loss(policy_lp, ref_lp, beta=0.1)[0].backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
    return model.lora_head.detach().clone()


def test_a_fixed_seed_reproduces_the_run_exactly():
    assert torch.equal(_run_short_npo(1234), _run_short_npo(1234))


# --------------------------------------------------------------- registration


def test_npo_is_registered_and_reachable_from_the_pipeline():
    from deletion_signals import run_pipeline  # noqa: F401  (registers npo)
    from deletion_signals.train import available_methods, get_unlearning_method

    assert "npo" in available_methods()
    assert get_unlearning_method("npo") is not None


def test_the_default_configuration_declares_the_published_defaults():
    """The grid is preregistered; a drifting default would silently redefine it."""
    npo = DEFAULTS["unlearning"]["npo"]
    assert npo["beta"] == 0.1
    assert npo["retain_weight"] == 1.0
    assert npo["log_ratio_reduction"] == "sum"
    assert npo["max_grad_norm"] == 1.0     # what makes the GA identity exact
