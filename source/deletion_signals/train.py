"""Fine-tuning and unlearning.

The evaluation itself is method-agnostic -- :mod:`collect` consumes a fine-tuned
checkpoint and an unlearned checkpoint and never asks how either was produced.
This module supplies two methods so the package can run end to end on its own,
and a registry so adding a third does not touch anything downstream.

Adding an unlearning method
---------------------------
Write ``def my_method(model, tokenizer, splits, config, save_path) -> dict`` and
register it::

    from deletion_signals.train import register_unlearning_method
    register_unlearning_method("my_method", my_method)

The returned dict is stored verbatim in the collection metadata as the run's
unlearning hyperparameters and outcome, so put in it whatever a reader would
need to know about how the run actually ended (steps taken, early stopping,
final losses). Nothing else needs to change: ``--method my_method`` then works
throughout, and the orientation table is defined per score, not per method.
"""

from __future__ import annotations

import copy
import json
import math
import os
from functools import partial
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import is_full_finetuning
from .data import SampleDataset, collate_fn

UnlearningMethod = Callable[..., Dict[str, Any]]
_REGISTRY: Dict[str, UnlearningMethod] = {}


def register_unlearning_method(name: str, fn: UnlearningMethod) -> None:
    """Register an unlearning method under ``name``."""
    if name in _REGISTRY:
        raise ValueError(f"Unlearning method {name!r} is already registered")
    _REGISTRY[name] = fn


def get_unlearning_method(name: str) -> UnlearningMethod:
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown unlearning method {name!r}; registered methods are "
            f"{sorted(_REGISTRY)}. Register a new one with "
            f"register_unlearning_method()."
        )
    return _REGISTRY[name]


def available_methods() -> Sequence[str]:
    return sorted(_REGISTRY)


def _loader(samples, tokenizer, batch_size: int, max_length: int, shuffle: bool):
    return DataLoader(
        SampleDataset(samples),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=partial(collate_fn, tokenizer=tokenizer, max_length=max_length),
    )


def _trainable(model) -> list:
    return [p for p in model.parameters() if p.requires_grad]


def enable_training_gradients(model, config: Mapping[str, Any]) -> int:
    """Turn on gradients for whatever this training mode considers trainable."""
    if is_full_finetuning(config):
        for param in model.parameters():
            param.requires_grad = True
    else:
        for name, param in model.named_parameters():
            if "lora" in name.lower() or "adapter" in name.lower():
                param.requires_grad = True
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _forward_batch(model, batch, device):
    batch = {k: v.to(device) for k, v in batch.items()}
    return model(**batch), batch


# ------------------------------------------------------------------ trajectory


@torch.no_grad()
def probe_loss(model, tokenizer, samples, config, batch_size: int = 8) -> float:
    """Mean token-weighted NLL over a fixed probe set.

    Switches to ``eval()`` for the measurement and restores the previous mode.
    Without that, dropout makes the probe a different random function at every
    call and a rising trajectory could be noise rather than the model moving.

    Token-weighted rather than a mean over batch losses, so the number is
    comparable with the perplexities the audit computes from per-sample rows.
    """
    was_training = model.training
    model.eval()
    device = torch.device(config["device"])
    max_length = config["data"]["max_length"]

    total_nll = 0.0
    total_tokens = 0
    try:
        for start in range(0, len(samples), batch_size):
            chunk = samples[start : start + batch_size]
            batch = collate_fn(
                [{"text": s.text, "sample_id": s.sample_id} for s in chunk],
                tokenizer=tokenizer,
                max_length=max_length,
            )
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            # HF averages over non-masked label positions; recover the sum.
            n_tokens = int((batch["labels"] != -100).sum().item()) - len(chunk)
            n_tokens = max(n_tokens, 1)
            total_nll += float(outputs.loss.item()) * n_tokens
            total_tokens += n_tokens
    finally:
        model.train(was_training)

    return total_nll / max(total_tokens, 1)


def build_probes(
    splits: Mapping[str, Sequence], n_probe: int
) -> Dict[str, list]:
    """A fixed slice of each population, used for every probe in a run.

    A prefix, so the probe is the same samples at every step and across runs at
    different strengths -- a trajectory measured on a resampled probe would move
    for reasons that have nothing to do with the model.
    """
    return {
        population: list(samples)[:n_probe]
        for population, samples in splits.items()
        if samples
    }


def _parameter_snapshot(model) -> Dict[str, torch.Tensor]:
    """Detached copy of the trainable parameters, for the displacement norm."""
    return {
        name: param.detach().clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }


def _displacement_norm(model, reference: Mapping[str, torch.Tensor]) -> float:
    """||theta - theta_reference||, over the trainable parameters.

    The same quantity whose direction becomes ghat at scoring time, so the
    trajectory records how far the run travelled as well as what it cost.
    """
    total = 0.0
    for name, param in model.named_parameters():
        if name in reference:
            total += float((param.detach() - reference[name]).pow(2).sum().item())
    return math.sqrt(max(total, 0.0))


# --------------------------------------------------------------------- fine-tune


def finetune(
    model,
    tokenizer,
    train_samples,
    config: Mapping[str, Any],
    save_path: str,
) -> Dict[str, Any]:
    """Fine-tune on the full member set (D_forget ∪ D_retain), then save.

    This is the θ_finetuned the unlearning direction is measured against, so it
    must be saved: ĝ = θ_unlearned − θ_finetuned cannot be recovered from the
    unlearned checkpoint alone.
    """
    device = torch.device(config["device"])
    training = config["training"]
    enable_training_gradients(model, config)
    model.train()

    loader = _loader(
        train_samples,
        tokenizer,
        training["batch_size"],
        config["data"]["max_length"],
        shuffle=True,
    )
    optimizer = torch.optim.AdamW(_trainable(model), lr=training["learning_rate"])
    accumulation = max(1, int(training.get("gradient_accumulation_steps", 1)))
    warmup = int(training.get("warmup_steps", 0))
    total_optimizer_steps = max(1, (len(loader) * int(training["epochs"])) // accumulation)

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min(1.0, (step + 1) / warmup) if warmup else 1.0,
    )

    step = 0
    last_loss = float("nan")
    for epoch in range(int(training["epochs"])):
        progress = tqdm(loader, desc=f"Fine-tuning epoch {epoch + 1}/{training['epochs']}")
        optimizer.zero_grad()
        for index, batch in enumerate(progress):
            outputs, _ = _forward_batch(model, batch, device)
            loss = outputs.loss / accumulation
            loss.backward()
            last_loss = float(outputs.loss.item())

            if (index + 1) % accumulation == 0:
                torch.nn.utils.clip_grad_norm_(
                    _trainable(model), training.get("max_grad_norm", 1.0)
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1
            progress.set_postfix({"loss": f"{last_loss:.4f}"})

    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    return {
        "epochs": int(training["epochs"]),
        "optimizer_steps": step,
        "planned_optimizer_steps": total_optimizer_steps,
        "final_loss": last_loss,
        "save_path": os.path.abspath(save_path),
    }


# ------------------------------------------------------------------------ GA+GD


def gradient_ascent_descent(
    model,
    tokenizer,
    splits: Mapping[str, Sequence],
    config: Mapping[str, Any],
    save_path: str,
) -> Dict[str, Any]:
    """Gradient ascent on D_forget, gradient descent on D_retain.

    Two separate optimizers so the ascent and descent learning rates are
    genuinely independent -- sharing one AdamW would couple them through the
    shared second-moment estimate.

    The loop stops early on either of two conditions, and the returned dict
    records which. Both matter downstream: the configured step count is not what
    the model received, and a run that hit the collapse guard is a different
    experiment from one that ran to completion.
    """
    device = torch.device(config["device"])
    settings = config["unlearning"]["ga_gd"]
    enable_training_gradients(model, config)
    model.train()

    forget_loader = _loader(
        splits["forget"], tokenizer, settings["batch_size"], config["data"]["max_length"], True
    )
    retain_loader = _loader(
        splits["retain"], tokenizer, settings["batch_size"], config["data"]["max_length"], True
    )

    # float() rather than trusting the config: a learning rate arriving from
    # `--set unlearning.ga_gd.ascent_lr=1e-4` can be a string, because YAML 1.1
    # does not read a bare exponent as a float. AdamW would accept it and fail
    # obscurely later.
    ascent_lr = float(settings["ascent_lr"])
    descent_lr = float(settings["descent_lr"])
    ascent_optimizer = torch.optim.AdamW(_trainable(model), lr=ascent_lr)
    descent_optimizer = torch.optim.AdamW(_trainable(model), lr=descent_lr)

    steps = int(settings["steps"])
    max_grad_norm = settings.get("max_grad_norm", 1.0)
    min_steps = int(settings.get("min_steps", 20))
    target_ratio = float(settings.get("target_loss_ratio", 2.0))
    patience = int(settings.get("early_stop_patience", 5))
    collapse_loss = float(settings.get("collapse_loss", 5.0))

    forget_iter, retain_iter = iter(forget_loader), iter(retain_loader)
    consecutive_above = 0
    steps_taken = 0
    stop_reason = "completed"
    forget_loss_value = retain_loss_value = float("nan")

    # ---- trajectory instrumentation -------------------------------------
    # The endpoint alone cannot say whether membership evidence disappeared
    # before, with, or after utility collapsed -- by the end both have already
    # happened. The probe is a fixed slice of each population, evaluated every
    # probe_every steps under eval(), so the run can be read as a path rather
    # than a destination.
    probe_every = int(settings.get("probe_every", 10))
    n_probe = int(settings.get("probe_samples", 32))
    probes = build_probes(splits, n_probe) if probe_every > 0 else {}
    reference = _parameter_snapshot(model) if probe_every > 0 else {}
    trajectory: list = []

    def record_probe(step_index: int, event: str) -> Dict[str, Any]:
        entry: Dict[str, Any] = {"step": step_index, "event": event}
        for population, samples in probes.items():
            entry[f"loss_{population}"] = probe_loss(
                model, tokenizer, samples, config, settings["batch_size"]
            )
        entry["displacement_norm"] = _displacement_norm(model, reference)
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

    for step in tqdm(range(steps), desc="GA+GD unlearning"):
        batch, forget_iter = next_batch(forget_iter, forget_loader)
        outputs, _ = _forward_batch(model, batch, device)
        forget_loss = outputs.loss
        (-forget_loss).backward()
        # clip_grad_norm_ returns the total norm BEFORE clipping, so the
        # stability diagnostic is free: a run that is about to diverge shows it
        # here first, several steps before the loss does.
        ascent_grad_norm = float(
            torch.nn.utils.clip_grad_norm_(_trainable(model), max_grad_norm)
        )
        ascent_optimizer.step()
        ascent_optimizer.zero_grad()

        batch, retain_iter = next_batch(retain_iter, retain_loader)
        outputs, _ = _forward_batch(model, batch, device)
        retain_loss = outputs.loss
        retain_loss.backward()
        descent_grad_norm = float(
            torch.nn.utils.clip_grad_norm_(_trainable(model), max_grad_norm)
        )
        descent_optimizer.step()
        descent_optimizer.zero_grad()

        forget_loss_value = float(forget_loss.item())
        retain_loss_value = float(retain_loss.item())
        ratio = forget_loss_value / (retain_loss_value + 1e-8)
        steps_taken = step + 1

        if probe_every > 0 and steps_taken % probe_every == 0:
            entry = record_probe(steps_taken, "probe")
            entry["batch_forget_loss"] = forget_loss_value
            entry["batch_retain_loss"] = retain_loss_value
            entry["ascent_grad_norm"] = ascent_grad_norm
            entry["descent_grad_norm"] = descent_grad_norm

        if step >= min_steps and ratio >= target_ratio:
            consecutive_above += 1
            if consecutive_above >= patience:
                stop_reason = "early_stop_loss_ratio"
                break
        else:
            consecutive_above = 0

        if forget_loss_value > collapse_loss or retain_loss_value > collapse_loss:
            stop_reason = "collapse_guard"
            break

    if probe_every > 0:
        final = record_probe(steps_taken, stop_reason)
        final["batch_forget_loss"] = forget_loss_value
        final["batch_retain_loss"] = retain_loss_value

    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)

    # Written beside the checkpoint rather than into the returned record: the
    # record is embedded verbatim in every scored row's metadata, and a 100-step
    # trajectory there would be carried a thousand times over.
    trajectory_path = None
    if trajectory:
        trajectory_path = os.path.join(
            os.path.dirname(os.path.abspath(save_path)), "trajectory.json"
        )
        with open(trajectory_path, "w") as handle:
            json.dump(
                {
                    "method": "ga_gd",
                    "probe_every": probe_every,
                    "probe_samples": {p: len(s) for p, s in probes.items()},
                    "stop_reason": stop_reason,
                    "entries": trajectory,
                },
                handle,
                indent=2,
            )

    return {
        "method": "ga_gd",
        # The coerced learning rates, not the raw config values: this block is
        # what a sweep report reads back to label each run's strength.
        "hyperparameters": {**dict(settings), "ascent_lr": ascent_lr,
                            "descent_lr": descent_lr},
        "steps_configured": steps,
        "steps_taken": steps_taken,
        "stop_reason": stop_reason,
        "final_forget_loss": forget_loss_value,
        "final_retain_loss": retain_loss_value,
        "trajectory_path": trajectory_path,
        "save_path": os.path.abspath(save_path),
    }


# ------------------------------------------------------------------------ SCRUB


def masked_kl_and_entropy(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    attention_mask: torch.Tensor,
    temperature: float,
    chunk_tokens: int = 4096,
) -> tuple:
    """Mean KL(teacher ‖ student) and mean student entropy over real tokens.

    Two constraints shape this function. **Precision:** both quantities sum over
    a 128k-entry vocabulary, and accumulating that many bfloat16 terms loses
    enough precision to make the KL come out negative -- impossible for a
    divergence -- so the arithmetic is float32. **Memory:** a [B, L, V] float32
    tensor is ~1 GB at B=4, L=511, V=128k, so tokens are processed in chunks and
    only the masked running sums are kept.
    """
    _, _, vocab = student_logits.shape
    flat_student = student_logits.reshape(-1, vocab)
    flat_teacher = teacher_logits.reshape(-1, vocab)
    flat_mask = attention_mask.reshape(-1).to(torch.float32)

    denom = flat_mask.sum() + 1e-8
    kl_total = student_logits.new_zeros((), dtype=torch.float32)
    entropy_total = student_logits.new_zeros((), dtype=torch.float32)

    for start in range(0, flat_student.shape[0], chunk_tokens):
        end = min(start + chunk_tokens, flat_student.shape[0])
        mask = flat_mask[start:end]
        if float(mask.sum()) == 0.0:
            continue

        student = flat_student[start:end].to(torch.float32)
        teacher = flat_teacher[start:end].to(torch.float32)

        student_log_probs = F.log_softmax(student / temperature, dim=-1)
        teacher_log_probs = F.log_softmax(teacher / temperature, dim=-1)
        teacher_probs = teacher_log_probs.exp()
        kl_token = (teacher_probs * (teacher_log_probs - student_log_probs)).sum(dim=-1)
        kl_total = kl_total + (kl_token * mask).sum()

        # Entropy at T=1: this term is about the student's own uncertainty, not
        # about matching the teacher, so it must not be temperature-scaled.
        log_probs_t1 = F.log_softmax(student, dim=-1)
        entropy_token = -(log_probs_t1.exp() * log_probs_t1).sum(dim=-1)
        entropy_total = entropy_total + (entropy_token * mask).sum()

    # T^2 keeps the KL gradient magnitude from shrinking as T grows.
    kl = (kl_total / denom) * (temperature**2)
    entropy = entropy_total / denom

    if float(kl.item()) < -1e-6:
        raise RuntimeError(
            f"KL is negative ({float(kl.item()):.3e}), which is impossible for a "
            f"divergence. Check that logits reach log_softmax in float32."
        )
    return kl, entropy


def scrub_unlearning(
    model,
    tokenizer,
    splits: Mapping[str, Sequence],
    config: Mapping[str, Any],
    save_path: str,
) -> Dict[str, Any]:
    """SCRUB-style unlearning against a frozen copy of the fine-tuned model.

    On D_retain: match the teacher (KL) and preserve utility (cross-entropy).
    On D_forget: diverge from the teacher (negative KL) and become uncertain
    (negative entropy).

    Retain and forget are backwarded separately rather than summed. Gradients
    accumulate into ``.grad`` either way so the update is identical, but summing
    first keeps BOTH autograd graphs alive at once, each holding float32
    activations over the full vocabulary -- which is what exhausts a 24 GB card.
    """
    device = torch.device(config["device"])
    settings = config["unlearning"]["scrub"]

    teacher = copy.deepcopy(model)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    enable_training_gradients(model, config)
    model.train()

    forget_loader = _loader(
        splits["forget"], tokenizer, settings["batch_size"], config["data"]["max_length"], True
    )
    retain_loader = _loader(
        splits["retain"], tokenizer, settings["batch_size"], config["data"]["max_length"], True
    )
    optimizer = torch.optim.AdamW(_trainable(model), lr=settings["learning_rate"])

    steps = int(settings["steps"])
    temperature = float(settings["kl_temperature"])
    forget_kl_weight = float(settings.get("forget_kl_weight", 0.1))
    forget_entropy_weight = float(settings.get("forget_entropy_weight", 0.5))
    retain_ce_weight = float(settings.get("retain_ce_weight", 1.0))
    retain_kl_weight = float(settings.get("retain_kl_weight", 1.0))
    max_grad_norm = float(settings.get("max_grad_norm", 1.0))

    forget_iter, retain_iter = iter(forget_loader), iter(retain_loader)
    metrics: Dict[str, float] = {}

    def next_batch(iterator, loader):
        try:
            return next(iterator), iterator
        except StopIteration:
            iterator = iter(loader)
            return next(iterator), iterator

    for _ in tqdm(range(steps), desc="SCRUB unlearning"):
        optimizer.zero_grad()

        batch, retain_iter = next_batch(retain_iter, retain_loader)
        batch = {k: v.to(device) for k, v in batch.items()}
        student_outputs = model(**batch)
        retain_ce = student_outputs.loss
        student_logits = student_outputs.logits[:, :-1, :]
        with torch.no_grad():
            teacher_logits = teacher(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
            ).logits[:, :-1, :]
        mask = batch["attention_mask"][:, 1:]
        n_tok = min(student_logits.shape[1], mask.shape[1])
        retain_kl, _ = masked_kl_and_entropy(
            student_logits[:, :n_tok], teacher_logits[:, :n_tok], mask[:, :n_tok], temperature
        )
        retain_total = retain_ce_weight * retain_ce + retain_kl_weight * retain_kl
        metrics["retain_ce"] = float(retain_ce.item())
        metrics["retain_kl"] = float(retain_kl.item())
        retain_total.backward()
        del student_outputs, student_logits, teacher_logits, retain_total

        batch, forget_iter = next_batch(forget_iter, forget_loader)
        batch = {k: v.to(device) for k, v in batch.items()}
        student_outputs = model(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        student_logits = student_outputs.logits[:, :-1, :]
        with torch.no_grad():
            teacher_logits = teacher(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
            ).logits[:, :-1, :]
        mask = batch["attention_mask"][:, 1:]
        n_tok = min(student_logits.shape[1], mask.shape[1])
        forget_kl, forget_entropy = masked_kl_and_entropy(
            student_logits[:, :n_tok], teacher_logits[:, :n_tok], mask[:, :n_tok], temperature
        )
        forget_total = -forget_kl_weight * forget_kl - forget_entropy_weight * forget_entropy
        metrics["forget_kl"] = float(forget_kl.item())
        metrics["forget_entropy"] = float(forget_entropy.item())
        forget_total.backward()
        del student_outputs, student_logits, teacher_logits, forget_total

        torch.nn.utils.clip_grad_norm_(_trainable(model), max_grad_norm)
        optimizer.step()

    del teacher
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)

    return {
        "method": "scrub",
        "hyperparameters": dict(settings),
        "steps_configured": steps,
        "steps_taken": steps,
        "stop_reason": "completed",
        "final_retain_ce": metrics.get("retain_ce", float("nan")),
        "final_forget_kl": metrics.get("forget_kl", float("nan")),
        "final_forget_entropy": metrics.get("forget_entropy", float("nan")),
        "save_path": os.path.abspath(save_path),
    }


register_unlearning_method("ga_gd", gradient_ascent_descent)
register_unlearning_method("scrub", scrub_unlearning)
