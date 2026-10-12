#!/usr/bin/env python3
"""CPU end-to-end smoke test.

Runs the whole machinery -- LoRA attachment, both unlearning methods, the
per-sample collector with every score enabled, the score file, and the offline
evaluator -- on a small model and a handful of synthetic texts. Nothing it
prints is a measurement; it is a check that each stage executes and that the
numbers it produces satisfy the properties the evaluation depends on.

    $PY deletion_signals/scripts/smoke_test.py
    $PY deletion_signals/scripts/smoke_test.py --model /path/to/snapshot --device cuda

Checks, in order:

  1. LoRA parameters are selected and the unlearning direction is non-zero
  2. layerwise float64 accumulation equals the concatenated-vector computation
     on real model gradients
  3. every enabled score is produced, finite and internally consistent
  4. the FUMA ratio is computed with both halves from one parameter space
  5. a zero unlearning direction yields cos = 0 rather than a crash
  6. GA+GD and SCRUB both run and change the parameters
  7. the score file round-trips and the offline evaluator recomputes from it
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from deletion_signals.collect import ModelBundle, ScoreCollector  # noqa: E402
from deletion_signals.config import DEFAULTS, deep_merge, validate_config  # noqa: E402
from deletion_signals.data import Sample  # noqa: E402
from deletion_signals.gradients import (  # noqa: E402
    ParamScope,
    accumulate_gradient_stats,
    collect_named_grads,
    concatenated_gradient_stats,
    select_param_names,
)
from deletion_signals.offline_eval import evaluate, format_report  # noqa: E402
from deletion_signals.orientations import OrientationTable  # noqa: E402
from deletion_signals.reproducibility import set_seed  # noqa: E402
from deletion_signals.schema import load_any, text_fingerprint, write_score_file  # noqa: E402
from deletion_signals.train import gradient_ascent_descent, scrub_unlearning  # noqa: E402

TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine unlearning aims to remove the influence of specific training "
    "examples from a trained model, which is a much longer sentence than the "
    "others in this list so that truncation and padding both get exercised.",
    "Short.",
    "Membership inference attacks exploit differences in model behaviour "
    "between members and non-members.",
    "Gradient ascent increases the loss on the forget set.",
    "Tiny.",
    "Deletion-set inference asks which known members were targeted.",
    "Calibrated loss divides out intrinsic sample difficulty.",
    "A frozen four-bit backbone yields no gradients at all.",
]

POPULATION_SIZES = {"forget": 3, "retain": 3, "test": 3}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="EleutherAI/pythia-160m-deduped",
                        help="Model id or local snapshot (small models only)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--lora-target", default=None,
                        help="LoRA target module (default: guessed from the architecture)")
    parser.add_argument("--keep", action="store_true",
                        help="Keep the temporary output directory")
    return parser.parse_args()


def guess_lora_target(model) -> str:
    names = [n for n, _ in model.named_modules()]
    for candidate in ("query_key_value", "q_proj", "c_attn"):
        if any(n.endswith(candidate) for n in names):
            return candidate
    raise SystemExit("Could not guess a LoRA target module; pass --lora-target")


def build_splits():
    splits = {}
    cursor = 0
    for population, size in POPULATION_SIZES.items():
        samples = []
        for index in range(size):
            text = TEXTS[cursor % len(TEXTS)]
            cursor += 1
            samples.append(
                Sample(
                    sample_id=f"{population}:{index}",
                    text=text,
                    population=population,
                    text_sha1=text_fingerprint(f"{population}-{index}-{text}"),
                )
            )
        splits[population] = samples
    return splits


def make_config(model_name: str, device: str, lora_target: str):
    config = deep_merge(
        DEFAULTS,
        {
            "device": device,
            "training_mode": "lora",
            "seed": 42,
            "model": {"name": model_name, "load_in_4bit": False, "torch_dtype": "float32"},
            "lora": {"r": 4, "alpha": 8, "dropout": 0.0, "target_modules": [lora_target]},
            "data": {"max_length": 64},
            "unlearning": {
                "ga_gd": {
                    "steps": 2, "batch_size": 2, "ascent_lr": 1e-4, "descent_lr": 1e-4,
                    "min_steps": 1000, "collapse_loss": 1e9,
                },
                "scrub": {"steps": 1, "batch_size": 2, "learning_rate": 1e-4,
                          "kl_temperature": 2.0},
            },
            "scoring": {"param_scope": "trainable", "max_length": 64, "eps": 1e-12},
            "scores": {
                "loss_calibrated": {"enabled": True, "reference": "pretrained_base"},
                # The second, independent loss reference: L_unlearned /
                # L_finetuned. On here so the smoke test exercises all three
                # references on a real model in one pass.
                "loss_vs_finetuned": {"enabled": True},
                "grad_calibrated": {"enabled": True, "reference": "finetuned_pre_unlearning"},
                # The CPU backbone is unquantised, so both halves of the ratio
                # come from the scored model; no bf16 shadow copy is needed.
                "fuma": {"enabled": True, "denominator_source": "same_model"},
                "sif": {"enabled": True, "method": "gradient", "damping": 1e-3, "beta": 1.0},
            },
            "evaluation": {"n_bootstrap": 50, "fpr_targets": [0.05], "seeds": [42],
                           "methods": ["ga_gd"]},
        },
    )
    validate_config(config)
    return config


def main() -> int:
    args = parse_args()
    set_seed(42)
    device = torch.device(args.device)

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("=" * 72)
    print(f"Loading {args.model} on {device}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def fresh():
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
        model.config.use_cache = False
        return model.to(device)

    target = args.lora_target or guess_lora_target(fresh())
    config = make_config(args.model, args.device, target)
    lora_cfg = LoraConfig(r=4, lora_alpha=8, target_modules=[target],
                          lora_dropout=0.0, bias="none", task_type="CAUSAL_LM")

    finetuned = get_peft_model(fresh(), lora_cfg)   # lora_B is zero-initialised
    unlearned = copy.deepcopy(finetuned)
    with torch.no_grad():                            # make ghat non-zero
        for name, param in unlearned.named_parameters():
            if "lora_B" in name:
                param.add_(torch.randn_like(param) * 0.01)
    for model in (finetuned, unlearned):
        for name, param in model.named_parameters():
            if "lora" in name.lower():
                param.requires_grad = True
        model.eval()

    base = fresh().eval()
    for param in base.parameters():
        param.requires_grad = False

    splits = build_splits()
    bundle = ModelBundle(unlearned=unlearned, tokenizer=tokenizer,
                         finetuned=finetuned, base=base, shadow=None)

    # ---------------------------------------------------------------- [1]
    print("\n[1] parameter selection and the unlearning direction")
    collector = ScoreCollector(config, bundle, device)
    assert collector.param_names, "no parameters selected"
    assert all("lora" in n.lower() for n in collector.param_names)
    assert collector.direction is not None and collector.direction_norm > 0
    print(f"    OK -- {len(collector.param_names)} LoRA tensors, "
          f"dim={sum(collector.param_numels.values())}, "
          f"||ghat||={collector.direction_norm:.4e}")

    # ---------------------------------------------------------------- [2]
    print("\n[2] layerwise float64 accumulation == concatenated computation")
    inputs, _ = collector._tokenize(TEXTS[1])
    _, named_grads = collector._forward_backward(unlearned, collector.param_names, inputs)
    stats = accumulate_gradient_stats(named_grads, direction=collector.direction)
    norm_sq, dot, dir_norm_sq = concatenated_gradient_stats(named_grads, collector.direction)
    for label, layered, concat in (
        ("||g||^2", stats.norm_sq, norm_sq),
        ("<g,ghat>", stats.dot_direction, dot),
        ("||ghat||^2", stats.direction_norm_sq, dir_norm_sq),
    ):
        rel = abs(layered - concat) / max(abs(concat), 1e-300)
        assert rel < 1e-12, f"{label}: layered={layered} concat={concat} rel={rel}"
    print(f"    OK -- ||g||^2={stats.norm_sq:.6e}, cos={stats.raw_cosine():+.6e}, "
          f"max rel. difference < 1e-12")

    # ---------------------------------------------------------------- [3]
    print("\n[3] every enabled score is produced and internally consistent")
    meta = {"seed": 42, "method": "ga_gd", "dataset": "smoke", "model_id": args.model,
            "unlearned_checkpoint": "smoke://unlearned",
            "finetuned_checkpoint": "smoke://finetuned"}
    rows = [collector.score_sample(s, meta)
            for population in splits for s in splits[population]]
    assert len(rows) == sum(POPULATION_SIZES.values())

    for row in rows:
        for field in ("loss_unlearned", "loss_base", "loss_ratio", "loss_delta",
                      "loss_finetuned", "loss_ratio_vs_finetuned",
                      "loss_delta_vs_finetuned", "grad_norm_sq", "grad_norm_sq_ref",
                      "grad_ratio", "log_grad_ratio", "raw_cosine",
                      "alignment_cos_sq", "sif"):
            value = getattr(row, field)
            assert value is not None, f"{field} missing on {row.sample_id}"
            assert math.isfinite(value), f"{field} not finite on {row.sample_id}"
        assert 0.0 <= row.alignment_cos_sq <= 1.0
        assert row.alignment_cos_sq == round(row.raw_cosine**2, 15) or \
            abs(row.alignment_cos_sq - row.raw_cosine**2) < 1e-12
        assert abs(row.loss_ratio - row.loss_unlearned / row.loss_base) < 1e-9
        assert abs(
            row.loss_ratio_vs_finetuned - row.loss_unlearned / row.loss_finetuned
        ) < 1e-9
        assert abs(row.grad_ratio - row.grad_norm_sq / row.grad_norm_sq_ref) < 1e-9
        assert abs(row.sif - row.grad_norm_sq / 1e-3) < 1e-6 * max(1.0, abs(row.sif))
        assert not row.flags["nonfinite_grad"] and not row.flags["nonfinite_loss"]
    print(f"    OK -- {len(rows)} rows, all scores finite and consistent")
    print(f"         cos^2 range [{min(r.alignment_cos_sq for r in rows):.3e}, "
          f"{max(r.alignment_cos_sq for r in rows):.3e}]")

    # ---------------------------------------------------------------- [4]
    print("\n[4] FUMA: both halves from one parameter space")
    assert collector.fuma_model is not None, collector.fuma_na_reason
    all_names = select_param_names(unlearned, ParamScope.FULL)
    assert set(collector.fuma_names) == set(all_names)
    fuma_values = [r.fuma_lora_ratio for r in rows]
    assert all(v is not None and math.isfinite(v) and v > 0 for v in fuma_values)
    fuma_grads = collect_named_grads(unlearned, collector.fuma_names)
    fuma_stats = accumulate_gradient_stats(fuma_grads, numel_hint=collector.fuma_numels)
    assert fuma_stats.numel_lora > 0 and fuma_stats.numel_non_lora > 0
    print(f"    OK -- LoRA {fuma_stats.numel_lora:,} params / non-LoRA "
          f"{fuma_stats.numel_non_lora:,} params; "
          f"ratio range [{min(fuma_values):.3e}, {max(fuma_values):.3e}]")
    if collector.fuma_unfroze_backbone:
        print("         (backbone unfrozen for the denominator -- recorded as a "
              "discrepancy)")

    # ---------------------------------------------------------------- [5]
    print("\n[5] a zero unlearning direction gives cos = 0, not a crash")
    zero_bundle = ModelBundle(unlearned=unlearned, tokenizer=tokenizer,
                              finetuned=copy.deepcopy(unlearned), base=None, shadow=None)
    zero_config = deep_merge(config, {"scores": {
        "loss_calibrated": {"enabled": False},
        "fuma": {"enabled": False},
    }})
    zero_collector = ScoreCollector(zero_config, zero_bundle, device)
    assert zero_collector.direction_norm == 0.0
    zero_row = zero_collector.score_sample(splits["forget"][0], meta)
    assert zero_row.raw_cosine == 0.0 and zero_row.alignment_cos_sq == 0.0
    assert zero_row.flags["zero_direction_norm"] is True
    print("    OK -- cos = 0 and zero_direction_norm flagged")

    # ---------------------------------------------------------------- [6]
    print("\n[6] GA+GD and SCRUB both run and move the parameters")
    workdir = tempfile.mkdtemp(prefix="deletion_signals_smoke_")
    for name, method in (("ga_gd", gradient_ascent_descent), ("scrub", scrub_unlearning)):
        model = copy.deepcopy(finetuned)
        before = torch.cat([p.detach().reshape(-1).clone()
                            for n, p in model.named_parameters() if "lora" in n.lower()])
        record = method(model, tokenizer, splits, config, os.path.join(workdir, name))
        after = torch.cat([p.detach().reshape(-1)
                           for n, p in model.named_parameters() if "lora" in n.lower()])
        moved = float((after - before).norm())
        assert moved > 0, f"{name} left the adapters unchanged"
        assert os.path.isdir(record["save_path"])
        print(f"    OK -- {name}: {record['steps_taken']} step(s), "
              f"stop={record['stop_reason']}, ||delta theta||={moved:.4e}")

    # ---------------------------------------------------------------- [7]
    print("\n[7] score file round-trip and offline recomputation")
    scores_path = os.path.join(workdir, "per_sample_scores.json")
    write_score_file(scores_path, rows, {"seed": 42, "method": "ga_gd", "smoke": True})
    parsed = load_any(scores_path)
    assert len(parsed.rows) == len(rows)
    expected = {"loss_unlearned", "loss_ratio", "loss_delta", "grad_norm_sq",
                "grad_ratio", "log_grad_ratio", "fuma_lora_ratio", "raw_cosine",
                "alignment_cos_sq", "sif", "combined_log_score"}
    assert expected.issubset(set(parsed.available_scores)), \
        f"missing: {expected - set(parsed.available_scores)}"

    table = OrientationTable.load()
    results = evaluate([parsed], config, table)
    primary = results["per_run"]["ga_gd/seed42"]["tasks"]["forget_vs_retain"]["scores"]
    reported = {k for k, v in primary.items() if v.get("available")}
    assert "alignment_cos_sq" in reported
    for score, entry in sorted(primary.items()):
        if entry.get("available"):
            assert 0.0 <= entry["auc"] <= 1.0
            assert entry["ci_lower"] <= entry["auc"] <= entry["ci_upper"]
    report = format_report(results, table)
    assert "max(AUC, 1-AUC)` is not used" in report
    print(f"    OK -- {len(parsed.available_scores)} scores round-tripped, "
          f"{len(reported)} evaluated on the primary task")
    print("         (AUCs from 3 vs 3 synthetic samples are meaningless by "
          "construction; only the plumbing is under test)")

    if not args.keep:
        import shutil

        shutil.rmtree(workdir, ignore_errors=True)
    else:
        print(f"\n    outputs kept in {workdir}")

    print("\n" + "=" * 72)
    print("ALL SMOKE CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
