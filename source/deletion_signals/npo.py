"""Negative Preference Optimization (NPO), implemented from the published form.

The claim NPO makes, and why it is worth testing here
-----------------------------------------------------
GA+GD passes from "forgot almost nothing" to "destroyed the model" with no
usable setting in between: the ascent term has no idea when to stop, so it keeps
pushing forget records that have already been pushed. NPO's entire contribution
is a gradient weight that decays to zero once a record is already forgotten.
Whether that weight actually prevents collapse *here* -- under AdamW, on QLoRA
adapters, on ArxivMIA -- is Component A's question.

The equation
------------
Zhang, Lin, Chen, Sun, Wu and Wu (2024), *Negative Preference Optimization:
From Catastrophic Collapse to Effective Unlearning*, arXiv:2404.05868, Eq. 3 --
the DPO objective with the preferred-response term removed::

    L_NPO(θ) = (2/β) · E_{(x,y)~D_forget} [ log( 1 + (π_θ(y|x) / π_ref(y|x))^β ) ]
             = -(2/β) · E [ log σ( -β · r(x,y) ) ]

    r(x,y) = log π_θ(y|x) − log π_ref(y|x)

**Sign convention.** ``r`` is *current minus reference*, on the forget sequence.
Minimising ``L_NPO`` drives ``π_θ`` DOWN on forget records. A run whose forget
loss falls rather than rises has the sign inverted and is a bug, not a result;
:func:`npo_loss` is unit-tested against a hand-computed example for exactly this
reason.

**The gradient, and the mechanism.** Differentiating::

    ∇_θ L_NPO = 2 · σ(β · r) · ∇_θ log π_θ(y|x)

which is gradient ascent on the forget log-likelihood, weighted by a factor in
(0, 2). At initialisation θ = θ_ref, so r = 0 and the weight is 2σ(0) = **1.0**:
NPO's first step is GA's first step. As forgetting proceeds r becomes negative,
σ(βr) → 0, and the weight vanishes. That decay is the whole method.

**β** is the temperature on the sequence log-ratio. β → 0 recovers plain
gradient ascent (the weight freezes at 1); larger β sharpens the gate, cutting a
record off sooner. The published default is 0.1 and this study fixes it there,
sweeping the learning rate instead -- see ``configs/preregistration_v3_fllm.yaml``.

**The reference model** is θ_finetuned: this seed's own pre-unlearning
checkpoint, which is also the initialisation. It is deep-copied, put in eval
mode and has ``requires_grad`` cleared on every parameter, exactly as SCRUB
holds its teacher. Reusing another seed's checkpoint is prohibited by the
preregistration and would make the seeds dependent.

Sum or mean over tokens -- a choice that had to be declared
-----------------------------------------------------------
The paper derives ``r`` over sequences and the official implementation
(``licong-lin/negative-preference-optimization``, following TOFU's
``get_batch_loss``) uses the **sum** of token log-probabilities. Every other
loss in this repository is a token *mean*. The two differ by a factor of up to
``max_length`` = 512, which moves β's effective scale by about two orders of
magnitude, so the choice is a config field (``log_ratio_reduction``), defaults
to the faithful ``sum``, and is written into every output row's metadata.

It does not affect the learning-rate grid: both GA+GD and NPO clip gradients to
norm 1.0, which normalises the scale difference away. That is also why the
step-0 identity above is exact after clipping rather than merely proportional --
:func:`test_npo.py` asserts it.

NPO-RT, and why that variant
----------------------------
The paper defines NPO (forget term alone), NPO-RT (plus retain cross-entropy)
and NPO-KL (plus KL to the reference on retain). This module implements
**NPO-RT**::

    L = L_NPO(D_forget) + λ · CE(D_retain)

because it is the variant structurally comparable to this repository's GA+GD,
which is ascent on forget plus cross-entropy descent on retain. Comparability,
not outcome, selected it: no NPO number existed when the choice was made.

One deviation from GA+GD is recorded rather than engineered away: GA+GD uses two
AdamW optimizers with independently swept learning rates, whereas NPO-RT as
published is a single objective under a single optimizer, so sweeping its
learning rate scales the retain term too.

Registered as ``npo``. Importing this module registers it.
"""

from __future__ import annotations

import copy
import json
import math
import os
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .train import (
    _displacement_norm,
    _loader,
    _parameter_snapshot,
    _trainable,
    build_probes,
    enable_training_gradients,
    probe_loss,
    register_unlearning_method,
)

SUM = "sum"
MEAN = "mean"
VALID_REDUCTIONS = (SUM, MEAN)


# ----------------------------------------------------------------- the numerics


def sequence_log_probs(
    logits: torch.Tensor,
    labels: torch.Tensor,
    reduction: str = SUM,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-sequence log π(y|x), and the token count it was computed over.

    Returns ``(log_probs, n_tokens)`` with one entry per sequence in the batch.

    Shifted the standard causal way -- position *t* predicts token *t+1* -- and
    masked on ``labels == -100``, so padding contributes nothing. Without that
    mask a sequence's log-probability would depend on how much padding the other
    members of its batch happened to need, which would make ``r`` a function of
    batch composition rather than of the sample.

    ``F.cross_entropy`` with ``reduction="none"`` is used rather than an explicit
    ``log_softmax``: it is the fused path, it handles ``ignore_index`` natively,
    and it does not materialise a [B, L, V] float32 tensor -- which at
    B=4, L=511, V=128k is about a gigabyte, and is what exhausts a 24 GB card.
    """
    if reduction not in VALID_REDUCTIONS:
        raise ValueError(
            f"log_ratio_reduction must be one of {VALID_REDUCTIONS}, got "
            f"{reduction!r}"
        )

    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]

    token_nll = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).view(shift_labels.shape)

    mask = (shift_labels != -100).to(token_nll.dtype)
    n_tokens = mask.sum(dim=-1)
    total_nll = (token_nll * mask).sum(dim=-1)

    if reduction == MEAN:
        # clamp, not add-eps: a sequence with zero scored tokens has no
        # log-probability, and dividing by an epsilon would invent a huge one.
        total_nll = total_nll / n_tokens.clamp(min=1.0)

    return -total_nll, n_tokens


def npo_loss(
    policy_log_probs: torch.Tensor,
    reference_log_probs: torch.Tensor,
    beta: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """The NPO forget objective, and the per-sample gradient weight.

    Returns ``(loss, weight)`` where::

        loss   = -(2/β) · mean over the batch of log σ( -β · r )
        weight = 2 · σ(β · r)          -- the coefficient on ∇ log π_θ

    ``weight`` is returned because it is the mechanism under test, not a
    diagnostic afterthought: it starts at 1.0 and must decay for NPO to be doing
    anything GA does not. The trajectory records its batch mean at every probe.

    ``F.logsigmoid`` rather than ``log(sigmoid(·))``: the latter underflows to
    ``-inf`` once β·r drops below about -100, which it does routinely once a
    sequence-summed log-ratio gets going. The fused form is exact there.
    """
    if beta <= 0:
        raise ValueError(f"NPO beta must be positive, got {beta}")

    log_ratio = policy_log_probs - reference_log_probs
    loss = -(2.0 / beta) * F.logsigmoid(-beta * log_ratio).mean()
    weight = 2.0 * torch.sigmoid(beta * log_ratio.detach())
    return loss, weight


# ------------------------------------------------------------------- the method


def npo_unlearning(
    model,
    tokenizer,
    splits: Mapping[str, Sequence],
    config: Mapping[str, Any],
    save_path: str,
) -> Dict[str, Any]:
    """NPO-RT against a frozen copy of the fine-tuned model.

    Forget and retain are backwarded separately rather than summed. Gradients
    accumulate into ``.grad`` either way so the update is identical, but summing
    first keeps both autograd graphs alive at once -- the same constraint that
    shapes SCRUB's loop, for the same reason.
    """
    device = torch.device(config["device"])
    settings = config["unlearning"]["npo"]

    beta = float(settings["beta"])
    retain_weight = float(settings.get("retain_weight", 1.0))
    reduction = str(settings.get("log_ratio_reduction", SUM))
    learning_rate = float(settings["learning_rate"])
    steps = int(settings["steps"])
    batch_size = int(settings["batch_size"])
    max_grad_norm = float(settings.get("max_grad_norm", 1.0))
    collapse_loss = float(settings.get("collapse_loss", 20.0))
    probe_every = int(settings.get("probe_every", 10))
    n_probe = int(settings.get("probe_samples", 32))

    if reduction not in VALID_REDUCTIONS:
        raise ValueError(
            f"unlearning.npo.log_ratio_reduction must be one of "
            f"{VALID_REDUCTIONS}, got {reduction!r}"
        )

    # The reference model IS the initialisation: θ_ref = θ_finetuned. Copied
    # before any optimiser touches the model, so r = 0 exactly at step 0 and the
    # published weight 2σ(0) = 1 holds by construction rather than by luck.
    reference_model = copy.deepcopy(model)
    reference_model.eval()
    for param in reference_model.parameters():
        param.requires_grad = False

    enable_training_gradients(model, config)
    model.train()

    forget_loader = _loader(
        splits["forget"], tokenizer, batch_size, config["data"]["max_length"], True
    )
    retain_loader = _loader(
        splits["retain"], tokenizer, batch_size, config["data"]["max_length"], True
    )
    optimizer = torch.optim.AdamW(_trainable(model), lr=learning_rate)

    snapshot = _parameter_snapshot(model)
    probes = build_probes(splits, n_probe) if probe_every > 0 else {}
    trajectory: list = []

    def record_probe(step_index: int, event: str) -> Dict[str, Any]:
        entry: Dict[str, Any] = {"step": step_index, "event": event}
        for population, samples in probes.items():
            entry[f"loss_{population}"] = probe_loss(
                model, tokenizer, samples, config, batch_size
            )
        entry["displacement_norm"] = _displacement_norm(model, snapshot)
        trajectory.append(entry)
        return entry

    if probe_every > 0:
        record_probe(0, "start")

    def next_batch(iterator, loader):
        try:
            return next(iterator), iterator
        except StopIteration:
            iterator = iter(loader)
            return next(iterator), iterator

    forget_iter, retain_iter = iter(forget_loader), iter(retain_loader)
    steps_taken = 0
    stop_reason = "completed"
    npo_value = retain_ce_value = weight_value = float("nan")
    forget_nll_value = float("nan")
    nonfinite_steps = 0

    for step in tqdm(range(steps), desc="NPO unlearning"):
        optimizer.zero_grad()

        # ---- the NPO term on D_forget ---------------------------------
        batch, forget_iter = next_batch(forget_iter, forget_loader)
        batch = {k: v.to(device) for k, v in batch.items()}
        policy_logits = model(**batch).logits
        with torch.no_grad():
            reference_logits = reference_model(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
            ).logits

        policy_lp, n_tokens = sequence_log_probs(
            policy_logits, batch["labels"], reduction
        )
        reference_lp, _ = sequence_log_probs(
            reference_logits, batch["labels"], reduction
        )
        forget_loss, weight = npo_loss(policy_lp, reference_lp.detach(), beta)
        forget_loss.backward()

        npo_value = float(forget_loss.item())
        weight_value = float(weight.mean().item())
        # Token-mean NLL on the same batch, on the scale every other loss in
        # this study is reported on, so the collapse guard and the trajectory
        # are comparable with GA+GD's.
        forget_nll_value = float(
            (-policy_lp.detach().sum() / n_tokens.sum().clamp(min=1)).item()
            if reduction == SUM
            else (-policy_lp.detach().mean()).item()
        )
        del policy_logits, reference_logits, forget_loss

        # ---- the retain term ------------------------------------------
        batch, retain_iter = next_batch(retain_iter, retain_loader)
        batch = {k: v.to(device) for k, v in batch.items()}
        retain_ce = model(**batch).loss
        (retain_weight * retain_ce).backward()
        retain_ce_value = float(retain_ce.item())
        del retain_ce

        if not (math.isfinite(npo_value) and math.isfinite(retain_ce_value)):
            nonfinite_steps += 1
            stop_reason = "non_finite_loss"
            steps_taken = step + 1
            break

        torch.nn.utils.clip_grad_norm_(_trainable(model), max_grad_norm)
        optimizer.step()
        steps_taken = step + 1

        if probe_every > 0 and steps_taken % probe_every == 0:
            entry = record_probe(steps_taken, "probe")
            entry["npo_loss"] = npo_value
            entry["retain_ce"] = retain_ce_value
            entry["batch_forget_nll"] = forget_nll_value
            # The mechanism under test. Starts at 1.0; if it does not fall, NPO
            # is behaving as plain gradient ascent on this problem.
            entry["npo_weight_mean"] = weight_value

        if forget_nll_value > collapse_loss or retain_ce_value > collapse_loss:
            stop_reason = "collapse_guard"
            break

    if probe_every > 0:
        final = record_probe(steps_taken, stop_reason)
        final["npo_loss"] = npo_value
        final["retain_ce"] = retain_ce_value
        final["batch_forget_nll"] = forget_nll_value
        final["npo_weight_mean"] = weight_value

    del reference_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)

    trajectory_path = None
    if trajectory:
        trajectory_path = os.path.join(
            os.path.dirname(os.path.abspath(save_path)), "trajectory.json"
        )
        with open(trajectory_path, "w") as handle:
            json.dump(
                {
                    "method": "npo",
                    "probe_every": probe_every,
                    "probe_samples": {p: len(s) for p, s in probes.items()},
                    "stop_reason": stop_reason,
                    "entries": trajectory,
                },
                handle,
                indent=2,
            )

    return {
        "method": "npo",
        "variant": "NPO-RT",
        "equation": (
            "L_NPO = -(2/beta) * E[ log sigmoid( -beta * (log pi_theta - "
            "log pi_ref) ) ]; total = L_NPO(forget) + retain_weight * CE(retain)"
        ),
        "reference": "Zhang et al. 2024, arXiv:2404.05868, Eq. 3",
        "reference_model": "finetuned_pre_unlearning (this seed's own checkpoint)",
        "sign_convention": (
            "log-ratio is current minus reference on the forget sequence; "
            "minimising drives forget likelihood down"
        ),
        "log_ratio_reduction": reduction,
        "hyperparameters": {
            **dict(settings),
            "beta": beta,
            "learning_rate": learning_rate,
            "retain_weight": retain_weight,
            "log_ratio_reduction": reduction,
        },
        "steps_configured": steps,
        "steps_taken": steps_taken,
        "stop_reason": stop_reason,
        "nonfinite_steps": nonfinite_steps,
        "final_npo_loss": npo_value,
        "final_forget_loss": forget_nll_value,
        "final_retain_loss": retain_ce_value,
        "final_npo_weight_mean": weight_value,
        "trajectory_path": trajectory_path,
        "save_path": os.path.abspath(save_path),
    }


register_unlearning_method("npo", npo_unlearning)
