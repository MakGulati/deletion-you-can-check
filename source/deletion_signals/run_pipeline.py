"""End-to-end driver: split -> fine-tune -> unlearn -> collect scores.

Each stage is skippable and each writes a checkpoint, so a run that fails in
unlearning does not re-do the fine-tuning. The split manifest is written once
per seed and read by every later stage, which is what guarantees that the model
that was unlearned and the samples that get scored refer to the same data.

Checkpoint paths are seed-scoped. With a seed-independent path every seed
silently reuses the first one's fine-tuned model: the forget/retain split still
varies, so the runs are not identical, but fine-tuning randomness is never
resampled and across-seed intervals come out too narrow. Those would be
forget-set draws from one model, not independent seeds.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

from .collect import collect_scores
from .config import apply_cli_overrides, effective_config, is_full_finetuning, load_config
from .data import build_splits, load_or_build_splits, write_manifest
from .models import (
    attach_lora,
    checkpoint_exists,
    free,
    load_base_model,
    load_checkpoint_auto,
)
from .reproducibility import run_provenance, set_seed
from .train import available_methods, finetune, get_unlearning_method

# Importing a method module is what registers it. NPO lives in its own file for
# the same reason the stabilised GA+GD does -- the equation and its provenance
# need room to be stated -- but --methods npo has to resolve without the caller
# knowing that, so the import happens here.
from . import npo as _npo  # noqa: F401,E402  (import for its registration)


def seed_dir(root: str, seed: int) -> str:
    return os.path.join(root, f"seed{seed}")


def run_seed(
    config: Dict[str, Any],
    seed: int,
    methods: Sequence[str],
    output_root: str,
    skip_finetune: bool = False,
    skip_unlearn: bool = False,
    skip_collect: bool = False,
    manifest: Optional[str] = None,
) -> Dict[str, Any]:
    """Run every stage for one seed, returning a summary of what happened.

    ``manifest`` points at an existing split manifest -- typically an earlier
    run's ``splits.json`` -- and makes this run share that run's exact
    forget/retain/test partition. Unlearning and scoring both read it, so the
    two runs are comparable sample by sample rather than merely in distribution.
    Without it the split is rebuilt from ``seed``, which is deterministic but
    depends on every field of ``data`` that feeds the draw.
    """
    set_seed(seed)
    config = dict(config)
    config["seed"] = seed

    run_root = seed_dir(output_root, seed)
    os.makedirs(run_root, exist_ok=True)

    # ---- split ------------------------------------------------------------
    local_manifest = os.path.join(run_root, "splits.json")
    manifest_source = None
    if manifest:
        print(f"[seed {seed}] using supplied split manifest {manifest}")
        # load_or_build_splits refuses a manifest written for another seed.
        splits = load_or_build_splits(config, seed, manifest)["splits"]
        manifest_source = os.path.abspath(manifest)
        manifest_path = manifest_source
        if os.path.abspath(local_manifest) != manifest_source:
            # A copy inside the run directory, so the run stays self-describing
            # and downstream globs over seed*/splits.json still find it. The
            # supplied file remains the one that was actually read.
            shutil.copyfile(manifest_source, local_manifest)
    elif os.path.exists(local_manifest):
        manifest_path = local_manifest
        print(f"[seed {seed}] reusing split manifest {manifest_path}")
        splits = load_or_build_splits(config, seed, manifest_path)["splits"]
    else:
        manifest_path = local_manifest
        print(f"[seed {seed}] building splits")
        splits = build_splits(config, seed)
        write_manifest(manifest_path, splits, config, seed)
    counts = {k: len(v) for k, v in splits.items()}
    print(f"[seed {seed}] forget={counts['forget']} retain={counts['retain']} test={counts['test']}")

    summary: Dict[str, Any] = {
        "seed": seed,
        "counts": counts,
        "manifest": manifest_path,
        "manifest_source": manifest_source,
        "methods": {},
    }

    # ---- fine-tune --------------------------------------------------------
    finetuned_path = os.path.join(
        config["paths"]["checkpoints"], f"finetuned_seed{seed}"
    )
    if skip_finetune and checkpoint_exists(finetuned_path, config):
        print(f"[seed {seed}] reusing fine-tuned checkpoint {finetuned_path}")
        summary["finetune"] = {"reused": True, "save_path": os.path.abspath(finetuned_path)}
    else:
        print(f"[seed {seed}] fine-tuning -> {finetuned_path}")
        model, tokenizer = load_base_model(config)
        if not is_full_finetuning(config):
            model = attach_lora(model, config)
        else:
            for param in model.parameters():
                param.requires_grad = True
        train_samples = list(splits["forget"]) + list(splits["retain"])
        summary["finetune"] = finetune(
            model, tokenizer, train_samples, config, finetuned_path
        )
        free(model)

    # ---- unlearn and collect, per method ----------------------------------
    for method in methods:
        method_dir = os.path.join(run_root, method)
        os.makedirs(method_dir, exist_ok=True)
        unlearned_path = os.path.join(method_dir, "unlearned_model")
        method_summary: Dict[str, Any] = {}

        if skip_unlearn and checkpoint_exists(unlearned_path, config):
            print(f"[seed {seed}] {method}: reusing unlearned checkpoint")
            method_summary["unlearning"] = {"reused": True}
            run_record_path = os.path.join(method_dir, "unlearning_run.json")
            if os.path.exists(run_record_path):
                with open(run_record_path) as handle:
                    method_summary["unlearning"] = json.load(handle)
        else:
            print(f"[seed {seed}] {method}: unlearning -> {unlearned_path}")
            # A full fine-tuning checkpoint carries its own weights, so loading
            # the pretrained backbone first would allocate a second copy of the
            # model for nothing. Only the LoRA path needs a backbone to attach
            # the adapters to.
            base = None
            if is_full_finetuning(config):
                model, tokenizer = load_checkpoint_auto(
                    config, finetuned_path, is_trainable=True
                )
            else:
                base, tokenizer = load_base_model(config)
                model, _ = load_checkpoint_auto(
                    config, finetuned_path, is_trainable=True, base_model=base
                )
            record = get_unlearning_method(method)(
                model, tokenizer, splits, config, unlearned_path
            )
            with open(os.path.join(method_dir, "unlearning_run.json"), "w") as handle:
                json.dump(record, handle, indent=2)
            method_summary["unlearning"] = record
            free(model, base)

        if not skip_collect:
            scores_path = os.path.join(method_dir, "per_sample_scores.json")
            print(f"[seed {seed}] {method}: collecting scores -> {scores_path}")
            collect_scores(
                config=config,
                seed=seed,
                method=method,
                unlearned_checkpoint=unlearned_path,
                finetuned_checkpoint=finetuned_path,
                output_path=scores_path,
                unlearning_run=method_summary.get("unlearning"),
                manifest_path=manifest_path,
            )
            method_summary["scores"] = os.path.abspath(scores_path)

        summary["methods"][method] = method_summary

    return summary


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune, unlearn and collect deletion-set inference scores.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # one seed, both methods\n"
            "  $PY -m deletion_signals.run_pipeline --config llama_lora.yaml \\\n"
            "      --seeds 42 --methods ga_gd scrub --output ds_results\n\n"
            "  # the full three-seed evaluation\n"
            "  $PY -m deletion_signals.run_pipeline --config llama_lora.yaml \\\n"
            "      --seeds 42 43 44 --methods ga_gd scrub --output ds_results\n\n"
            "  # then\n"
            "  $PY -m deletion_signals.offline_eval \\\n"
            "      --inputs 'ds_results/seed*/*/per_sample_scores.json' \\\n"
            "      --output ds_results/evaluation\n"
        ),
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--methods", nargs="+", default=["ga_gd"],
                        help=f"Registered methods: {', '.join(available_methods())}")
    parser.add_argument("--output", default="./ds_results")
    parser.add_argument("--device", default=None)
    parser.add_argument("--skip-finetune", action="store_true",
                        help="Reuse a cached fine-tuned checkpoint if present")
    parser.add_argument("--skip-unlearn", action="store_true",
                        help="Reuse a cached unlearned checkpoint if present")
    parser.add_argument("--skip-collect", action="store_true",
                        help="Train only; do not score")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Cap scored samples per population")
    parser.add_argument("--manifest", default=None,
                        help="Existing split manifest to reuse for every seed, so "
                             "this run shares its forget/retain/test partition. "
                             "With several --seeds, {seed} in the path is "
                             "substituted per seed.")
    return parser.parse_args(argv)


def manifest_for_seed(pattern: Optional[str], seed: int) -> Optional[str]:
    """Resolve ``--manifest`` for one seed.

    ``{seed}`` in the pattern is substituted, so a multi-seed run can point at
    ``results/seed{seed}/splits.json`` and give each seed its own split. Without
    the placeholder the same file is used for every seed, which
    ``load_or_build_splits`` rejects for all but the seed it was written for --
    scoring one seed's checkpoints against another seed's split mislabels every
    row, so failing is the correct outcome.
    """
    if pattern is None:
        return None
    return pattern.replace("{seed}", str(seed))


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    if args.device:
        config["device"] = args.device
    if args.max_samples is not None:
        config["scoring"]["max_samples_per_population"] = args.max_samples
    config = apply_cli_overrides(config, args.overrides)

    unknown = [m for m in args.methods if m not in available_methods()]
    if unknown:
        raise SystemExit(
            f"Unknown method(s) {unknown}; registered: {list(available_methods())}"
        )

    os.makedirs(args.output, exist_ok=True)
    summaries: List[Dict[str, Any]] = []
    for seed in args.seeds:
        summaries.append(
            run_seed(
                config=config,
                seed=seed,
                methods=args.methods,
                output_root=args.output,
                skip_finetune=args.skip_finetune,
                skip_unlearn=args.skip_unlearn,
                skip_collect=args.skip_collect,
                manifest=manifest_for_seed(args.manifest, seed),
            )
        )

    summary_path = os.path.join(args.output, "pipeline_summary.json")
    with open(summary_path, "w") as handle:
        json.dump(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "seeds": args.seeds,
                "methods": args.methods,
                "runs": summaries,
                "provenance": run_provenance(effective_config(config)),
            },
            handle,
            indent=2,
            default=str,
        )
    print(f"\nWrote {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
