"""Model loading: QLoRA, full precision, and the bf16 shadow model.

Three kinds of model appear in a collection run:

* the **unlearned** model, which every score is computed on;
* the **references** -- the pretrained base (loss calibration) and the
  fine-tuned pre-unlearning checkpoint (gradient calibration and the unlearning
  direction ĝ);
* the **bf16 shadow**, an unquantised copy of the unlearned model that exists
  only so the FUMA denominator has somewhere to come from. See
  :func:`load_shadow_model` and docs/FUMA.md.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Mapping, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import is_full_finetuning

_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def resolve_dtype(name: Optional[str], default: torch.dtype = torch.bfloat16) -> torch.dtype:
    if not name:
        return default
    if name not in _DTYPES:
        raise ValueError(f"Unknown dtype {name!r}; expected one of {sorted(_DTYPES)}")
    return _DTYPES[name]


def checkpoint_exists(path: str, config: Mapping[str, Any]) -> bool:
    """Whether ``path`` holds a usable checkpoint for this training mode.

    PEFT writes ``adapter_model.safetensors`` on current versions and
    ``adapter_model.bin`` on older ones; accepting only the first name makes
    caching silently never trigger, so every run re-trains.
    """
    if not os.path.isdir(path):
        return False
    if is_full_finetuning(config):
        return os.path.exists(os.path.join(path, "config.json"))
    return any(
        os.path.exists(os.path.join(path, name))
        for name in ("adapter_model.safetensors", "adapter_model.bin")
    )


def load_tokenizer(config: Mapping[str, Any], path: Optional[str] = None) -> AutoTokenizer:
    """Tokenizer for the configured model, with a pad token guaranteed."""
    source = path or config["model"]["name"]
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            source, trust_remote_code=config["model"].get("trust_remote_code", True)
        )
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(
            config["model"]["name"],
            trust_remote_code=config["model"].get("trust_remote_code", True),
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_base_model(
    config: Mapping[str, Any], quantized: Optional[bool] = None, device_map: str = "auto"
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load the pretrained backbone.

    Args:
        quantized: Force 4-bit on or off. ``None`` follows the training mode:
            quantised in LoRA mode, unquantised for full fine-tuning. The
            shadow model uses ``quantized=False`` in LoRA mode, which is the
            whole reason this is an argument.
    """
    model_cfg = config["model"]
    if quantized is None:
        quantized = not is_full_finetuning(config) and bool(model_cfg.get("load_in_4bit", True))

    kwargs: Dict[str, Any] = {
        "device_map": device_map,
        "trust_remote_code": model_cfg.get("trust_remote_code", True),
        "dtype": resolve_dtype(model_cfg.get("torch_dtype")),
    }

    if quantized:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=resolve_dtype(model_cfg.get("bnb_4bit_compute_dtype")),
            bnb_4bit_quant_type=model_cfg.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(model_cfg["name"], **kwargs)
    tokenizer = load_tokenizer(config)
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    # The KV cache is useless for training and scoring (no generation happens)
    # and its [B, L, H] tensors are a real fraction of the memory budget on a
    # 24 GB card.
    model.config.use_cache = False
    return model, tokenizer


def attach_lora(model: AutoModelForCausalLM, config: Mapping[str, Any]):
    """Wrap ``model`` in freshly initialised PEFT LoRA adapters."""
    from peft import LoraConfig, get_peft_model

    lora_cfg = config["lora"]
    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        target_modules=list(lora_cfg["target_modules"]),
        lora_dropout=lora_cfg["dropout"],
        bias=lora_cfg["bias"],
        task_type=lora_cfg["task_type"],
    )
    return get_peft_model(model, peft_config)


def load_adapter(base_model, checkpoint_path: str, is_trainable: bool = True):
    """Load a PEFT adapter checkpoint onto ``base_model``.

    ``is_trainable=True`` is required whenever the model will be differentiated:
    parameter selection keys off ``requires_grad``, and PEFT loads adapters
    frozen by default, which would leave the gradient statistics ranging over an
    empty parameter set.
    """
    from peft import PeftModel

    model = PeftModel.from_pretrained(base_model, checkpoint_path, is_trainable=is_trainable)
    if is_trainable:
        for name, param in model.named_parameters():
            if "lora" in name.lower():
                param.requires_grad = True
    else:
        model.eval()
    return model


def load_full_checkpoint(
    config: Mapping[str, Any], checkpoint_path: str, is_trainable: bool = True
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load a fully fine-tuned checkpoint."""
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        device_map="auto",
        trust_remote_code=config["model"].get("trust_remote_code", True),
        dtype=resolve_dtype(config["model"].get("torch_dtype")),
    )
    tokenizer = load_tokenizer(config, checkpoint_path)
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False

    for param in model.parameters():
        param.requires_grad = bool(is_trainable)
    model.train() if is_trainable else model.eval()
    return model, tokenizer


def load_checkpoint_auto(
    config: Mapping[str, Any],
    checkpoint_path: str,
    is_trainable: bool = True,
    base_model=None,
) -> Tuple[Any, AutoTokenizer]:
    """Load a checkpoint according to the configured training mode."""
    if is_full_finetuning(config):
        return load_full_checkpoint(config, checkpoint_path, is_trainable)
    if base_model is None:
        base_model, _ = load_base_model(config)
    tokenizer = load_tokenizer(config)
    return load_adapter(base_model, checkpoint_path, is_trainable), tokenizer


def load_shadow_model(config: Mapping[str, Any], adapter_path: str):
    """An unquantised bf16 copy of the unlearned model with EVERY parameter trainable.

    Why this exists
    ---------------
    The FUMA ratio divides the gradient magnitude on LoRA parameters by the
    gradient magnitude on the non-LoRA backbone. Under QLoRA the backbone is
    both frozen and 4-bit quantised, and bitsandbytes does not backpropagate
    into 4-bit weights, so the denominator simply does not exist on the scored
    model.

    This loads the backbone unquantised in bf16, applies the same adapters, and
    unfreezes everything, so one backward pass yields both halves of the ratio
    in a single consistent parameter space.

    Documented discrepancies from the published FUMA setting -- both recorded in
    every output file's metadata:

    1. The denominator (and, for consistency, the numerator) come from an
       unquantised copy of the model, not from the quantised model that was
       actually unlearned. Dequantisation is not exact, so these are gradients
       of a nearby model rather than of the scored one.
    2. The backbone is unfrozen purely to obtain gradients. It was frozen during
       training, so these gradients describe a direction the optimiser never
       took.

    Raises:
        RuntimeError: if the unquantised copy does not fit. FUMA is then marked
            not-applicable with the reason recorded, rather than being computed
            from an invented definition.
    """
    if is_full_finetuning(config):
        raise ValueError(
            "A shadow model is meaningless under full fine-tuning: nothing is "
            "quantised and there are no adapters to form the FUMA numerator."
        )

    try:
        base, _ = load_base_model(config, quantized=False)
        model = load_adapter(base, adapter_path, is_trainable=True)
    except torch.cuda.OutOfMemoryError as exc:  # pragma: no cover - hardware dependent
        torch.cuda.empty_cache()
        raise RuntimeError(
            f"The unquantised bf16 shadow model does not fit alongside the models "
            f"already loaded ({exc}). Set scores.fuma.denominator_source='none' "
            f"to record FUMA as not-applicable with this reason."
        ) from exc

    # Unfreeze the backbone: without this only the adapters carry gradients and
    # the denominator is still missing, which is the exact problem this model
    # was loaded to solve.
    for param in model.parameters():
        param.requires_grad = True
    model.config.use_cache = False
    return model


def describe_model(model, identifier: str) -> Dict[str, Any]:
    """A compact record of what was actually loaded, for output metadata."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    lora = sum(
        p.numel() for n, p in model.named_parameters() if "lora" in n.lower()
    )
    dtypes = sorted({str(p.dtype) for p in model.parameters()})
    return {
        "identifier": identifier,
        "n_parameters": total,
        "n_trainable": trainable,
        "n_lora": lora,
        "quantized": any("4bit" in d or "uint8" in d for d in dtypes),
        "dtypes": dtypes,
    }


def free(*models) -> None:
    """Drop references and release the CUDA caching allocator's blocks."""
    for model in models:
        del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
