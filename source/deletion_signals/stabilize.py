"""Stabilised GA+GD: is collapse avoidable, or is it the method?

The exploratory sweep showed GA+GD passing straight from "forgot almost nothing"
to "destroyed the model" with no usable setting in between. A learning-rate
sweep alone cannot distinguish two explanations of that:

* **optimisation instability** -- the ascent term runs away and takes the whole
  model with it, and a suitable constraint would keep it on the manifold; or
* **a real trade-off** -- on this model and data, moving the forget set far
  enough necessarily moves everything else.

This module implements four *theoretically motivated* constraints, each aimed at
one mechanism by which unconstrained ascent could be destroying the model. They
are knobs on one loop rather than four separate methods, so a run differs from
plain GA+GD in exactly the term being tested.

======  ==============================  ====================================
knob    constraint                      the mechanism it targets
======  ==============================  ====================================
S1      retain_kl_weight                function-space drift: the model stops
                                        matching theta_finetuned where it
                                        should still agree with it
S2      trust_region_radius             weight-space runaway: the iterate
                                        leaves the neighbourhood in which the
                                        fine-tuned solution lives
S3      max_grad_norm (already in       per-step magnitude: one bad batch
        base GA+GD, tightened here)     moves the model further than any
                                        batch should
S4      utility_guard_ratio             none -- it does not prevent collapse,
                                        it *detects* it and stops, which
                                        answers "was there a good point on
                                        this trajectory that the endpoint
                                        hid?"
======  ==============================  ====================================

S4 deserves its own note. The other three try to make the trajectory better; S4
assumes the trajectory already passes through a good region and simply stops
there. If S4 alone finds an operating point, the finding is not "GA+GD can be
stabilised" but "GA+GD's *stopping rule* was the problem", which is a different
and cheaper claim.

Registered as ``ga_gd_stabilized``. Importing this module registers it; nothing
in :mod:`train` changes.
"""

from __future__ import annotations

import copy
import json
import math
import os
from typing import Any, Dict, Mapping, Optional, Sequence

import torch
from tqdm import tqdm

from .train import (
    _displacement_norm,
    _forward_batch,
    _loader,
    _parameter_snapshot,
    _trainable,
    build_probes,
    enable_training_gradients,
    masked_kl_and_entropy,
    probe_loss,
    register_unlearning_method,
)


def project_to_trust_region(
    model, reference: Mapping[str, torch.Tensor], radius: float
) -> Optional[float]:
    """Pull theta back onto the ball ||theta - theta_ft|| <= radius.

    A hard constraint rather than a penalty, because a penalty needs a
    coefficient whose right value depends on the loss scale at the point where
    collapse begins -- which is not known before the run and would have to be
    tuned on the data the run produces. A radius is interpretable without
    tuning: "never travel further from the fine-tuned model than the strongest
    setting that preserved utility did".

    Returns the displacement norm before projection, or None if no projection
    was needed, so the caller can record how hard the constraint was binding.
    """
    with torch.no_grad():
        total = 0.0
        for name, param in model.named_parameters():
            if name in reference:
                total += float((param.detach() - reference[name]).pow(2).sum().item())
        norm = math.sqrt(max(total, 0.0))
        if norm <= radius or norm == 0.0:
            return None
        scale = radius / norm
        for name, param in model.named_parameters():
            if name in reference:
                param.copy_(reference[name] + (param - reference[name]) * scale)
        return norm


def stabilized_gradient_ascent_descent(
    model,
    tokenizer,
    splits: Mapping[str, Sequence],
    config: Mapping[str, Any],
    save_path: str,
) -> Dict[str, Any]:
    """GA+GD with any combination of the four constraints switched on.

    With every knob off this is plain GA+GD, which is deliberate: the comparison
    a stabilisation study rests on is between runs that differ in one term, so
    the unconstrained arm must go through the same code.
    """
    device = torch.device(config["device"])
    settings = config["unlearning"]["ga_gd_stabilized"]
    enable_training_gradients(model, config)
    model.train()

    ascent_lr = float(settings["ascent_lr"])
    descent_lr = float(settings["descent_lr"])
    steps = int(settings["steps"])
    batch_size = int(settings["batch_size"])
    max_grad_norm = float(settings.get("max_grad_norm", 1.0))
    collapse_loss = float(settings.get("collapse_loss", 20.0))

    # --- the four knobs -------------------------------------------------
    retain_kl_weight = float(settings.get("retain_kl_weight", 0.0))
    kl_temperature = float(settings.get("kl_temperature", 2.0))
    trust_region_radius = settings.get("trust_region_radius")
    trust_region_radius = None if trust_region_radius in (None, 0) else float(trust_region_radius)
    utility_guard_ratio = settings.get("utility_guard_ratio")
    utility_guard_ratio = None if utility_guard_ratio in (None, 0) else float(utility_guard_ratio)
    probe_every = int(settings.get("probe_every", 5))
    n_probe = int(settings.get("probe_samples", 32))

    forget_loader = _loader(
        splits["forget"], tokenizer, batch_size, config["data"]["max_length"], True
    )
    retain_loader = _loader(
        splits["retain"], tokenizer, batch_size, config["data"]["max_length"], True
    )
    ascent_optimizer = torch.optim.AdamW(_trainable(model), lr=ascent_lr)
    descent_optimizer = torch.optim.AdamW(_trainable(model), lr=descent_lr)

    # The frozen pre-unlearning model, needed by S1 only. Skipped otherwise --
    # it is a second full copy of the model in memory.
    teacher = None
    if retain_kl_weight > 0.0:
        teacher = copy.deepcopy(model)
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad = False

    reference = _parameter_snapshot(model)
    probes = build_probes(splits, n_probe) if probe_every > 0 else {}
    trajectory: list = []

    def record_probe(step_index: int, event: str) -> Dict[str, Any]:
        entry: Dict[str, Any] = {"step": step_index, "event": event}
        for population, samples in probes.items():
            entry[f"loss_{population}"] = probe_loss(
                model, tokenizer, samples, config, batch_size
            )
        entry["displacement_norm"] = _displacement_norm(model, reference)
        trajectory.append(entry)
        return entry

    baseline = record_probe(0, "start") if probe_every > 0 else {}
    retain_baseline = baseline.get("loss_retain")

    forget_iter, retain_iter = iter(forget_loader), iter(retain_loader)
    steps_taken = 0
    stop_reason = "completed"
    n_projections = 0
    forget_loss_value = retain_loss_value = float("nan")

    def next_batch(iterator, loader):
        try:
            return next(iterator), iterator
        except StopIteration:
            iterator = iter(loader)
            return next(iterator), iterator

    for step in tqdm(range(steps), desc="GA+GD (stabilised)"):
        # ---- ascent on D_forget ----------------------------------------
        batch, forget_iter = next_batch(forget_iter, forget_loader)
        outputs, _ = _forward_batch(model, batch, device)
        forget_loss = outputs.loss
        (-forget_loss).backward()
        ascent_grad_norm = float(
            torch.nn.utils.clip_grad_norm_(_trainable(model), max_grad_norm)
        )
        ascent_optimizer.step()
        ascent_optimizer.zero_grad()

        # ---- descent on D_retain, plus the KL anchor -------------------
        batch, retain_iter = next_batch(retain_iter, retain_loader)
        batch = {k: v.to(device) for k, v in batch.items()}
        student_outputs = model(**batch)
        retain_loss = student_outputs.loss
        total = retain_loss
        retain_kl_value = float("nan")

        if teacher is not None:
            student_logits = student_outputs.logits[:, :-1, :]
            with torch.no_grad():
                teacher_logits = teacher(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                ).logits[:, :-1, :]
            mask = batch["attention_mask"][:, 1:]
            n_tok = min(student_logits.shape[1], mask.shape[1])
            retain_kl, _ = masked_kl_and_entropy(
                student_logits[:, :n_tok],
                teacher_logits[:, :n_tok],
                mask[:, :n_tok],
                kl_temperature,
            )
            retain_kl_value = float(retain_kl.item())
            total = retain_loss + retain_kl_weight * retain_kl
            del teacher_logits, student_logits

        total.backward()
        descent_grad_norm = float(
            torch.nn.utils.clip_grad_norm_(_trainable(model), max_grad_norm)
        )
        descent_optimizer.step()
        descent_optimizer.zero_grad()
        del student_outputs

        # ---- trust region ----------------------------------------------
        if trust_region_radius is not None:
            before = project_to_trust_region(model, reference, trust_region_radius)
            if before is not None:
                n_projections += 1

        forget_loss_value = float(forget_loss.item())
        retain_loss_value = float(retain_loss.item())
        steps_taken = step + 1

        # ---- probes, and the utility guard ------------------------------
        if probe_every > 0 and steps_taken % probe_every == 0:
            entry = record_probe(steps_taken, "probe")
            entry["batch_forget_loss"] = forget_loss_value
            entry["batch_retain_loss"] = retain_loss_value
            entry["ascent_grad_norm"] = ascent_grad_norm
            entry["descent_grad_norm"] = descent_grad_norm
            entry["retain_kl"] = retain_kl_value
            entry["n_projections"] = n_projections

            if (
                utility_guard_ratio is not None
                and retain_baseline
                and entry.get("loss_retain") is not None
                and entry["loss_retain"] > retain_baseline * utility_guard_ratio
            ):
                # Not a failure: the guard is the intervention. It stops at the
                # last point where retained utility was still intact, which is
                # the best this trajectory had to offer.
                stop_reason = "utility_guard"
                break

        if forget_loss_value > collapse_loss or retain_loss_value > collapse_loss:
            stop_reason = "collapse_guard"
            break

    if probe_every > 0:
        final = record_probe(steps_taken, stop_reason)
        final["batch_forget_loss"] = forget_loss_value
        final["batch_retain_loss"] = retain_loss_value
        final["n_projections"] = n_projections

    if teacher is not None:
        del teacher
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
                    "method": "ga_gd_stabilized",
                    "probe_every": probe_every,
                    "stop_reason": stop_reason,
                    "entries": trajectory,
                },
                handle,
                indent=2,
            )

    return {
        "method": "ga_gd_stabilized",
        "hyperparameters": {
            **dict(settings),
            "ascent_lr": ascent_lr,
            "descent_lr": descent_lr,
        },
        "stabilizers": {
            "retain_kl_weight": retain_kl_weight,
            "trust_region_radius": trust_region_radius,
            "max_grad_norm": max_grad_norm,
            "utility_guard_ratio": utility_guard_ratio,
        },
        "steps_configured": steps,
        "steps_taken": steps_taken,
        "stop_reason": stop_reason,
        "n_trust_region_projections": n_projections,
        "final_forget_loss": forget_loss_value,
        "final_retain_loss": retain_loss_value,
        "trajectory_path": trajectory_path,
        "save_path": os.path.abspath(save_path),
    }


register_unlearning_method("ga_gd_stabilized", stabilized_gradient_ascent_descent)
