"""The two reference points every unlearning claim is measured against.

Approximate unlearning has no meaning without a definition of success, and the
only unambiguous one is: *the model you would have had if the forget set had
never been in the training data*. That is the *exact retraining* control. It
sets the scale for how far a forget sample's loss should move, and it shows what
the detectors report when deletion is genuinely complete.

But a second control is needed to read the first one at all.

Why the null refit exists
-------------------------
Every directional detector is built on

    ghat = theta_post - theta_finetuned

For an unlearning method that is a **small displacement**: a short optimisation
that started at ``theta_finetuned``. For exact retraining ``theta_post`` is an
**independent fine-tune**, so the same subtraction is not small and contains the
deletion effect *plus* everything that differs between two separate training
runs -- batch order, dropout, optimiser state. A forget-vs-retain AUC of 0.6 on
the retraining control could mean "deletion leaves a trace" or "two training runs
differ", and nothing in that number distinguishes them.

The **null refit** removes the ambiguity: fine-tune again on the *full* member
set, forget included, changing only the training RNG. Same
two-independent-models structure, nothing deleted. Any signal it produces is a
false positive by construction, and it is the floor the retraining control must
clear before any of its detectors can be said to have found deletion.

Both controls reuse the seed's own split manifest, so the populations -- and the
Fisher curvature holdout's disjointness from them -- are identical to every
other run in the study.

Usage
-----
::

    $PY -m deletion_signals.controls --config arxivmia_confirm.yaml \\
        --control exact_retrain --seeds 42 43 44 \\
        --manifest 'results/seed{seed}/splits.json' --output ds_confirm/controls
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .collect import collect_scores
from .config import apply_cli_overrides, effective_config, is_full_finetuning, load_config
from .data import load_or_build_splits
from .models import attach_lora, checkpoint_exists, free, load_base_model
from .reproducibility import run_provenance, set_seed
from .train import finetune

# What each control trains on, and how its training randomness is offset from
# the original fine-tune. The offset is what makes the null refit a *different*
# draw rather than a bit-identical repeat: same data, same split, new RNG.
CONTROLS: Dict[str, Dict[str, Any]] = {
    "exact_retrain": {
        "populations": ("retain",),
        "training_seed_offset": 0,
        "description": (
            "gold standard: fine-tuned from the pretrained base on D_retain "
            "only, so the forget set was never seen. This is what deletion "
            "looks like when it is real."
        ),
    },
    "null_refit": {
        "populations": ("forget", "retain"),
        "training_seed_offset": 1000,
        "description": (
            "false-positive floor: fine-tuned from the pretrained base on the "
            "FULL member set, differing from theta_finetuned only in training "
            "randomness. Nothing was deleted, so any deletion signal here is "
            "produced by run-to-run variation between independently trained "
            "models."
        ),
    },
}


def control_training_samples(
    splits: Mapping[str, Sequence], control: str
) -> List[Any]:
    """The training set for a control, in the pipeline's own population order.

    ``exact_retrain`` omits D_forget entirely -- that omission is the whole
    experiment. ``null_refit`` keeps it, so the only difference from the
    original fine-tune is the RNG.
    """
    if control not in CONTROLS:
        raise KeyError(
            f"Unknown control {control!r}; registered controls are "
            f"{sorted(CONTROLS)}"
        )
    populations = CONTROLS[control]["populations"]
    samples: List[Any] = []
    for population in populations:
        samples.extend(splits[population])
    if not samples:
        raise ValueError(
            f"Control {control!r} has no training samples; the split supplied "
            f"populations {sorted(splits)}"
        )
    return samples


def training_seed_for(control: str, seed: int) -> int:
    """The RNG seed the control's fine-tune runs under.

    Offset, not shared: ``null_refit`` must differ from the original fine-tune
    somewhere, and training randomness is the only thing it is allowed to differ
    in. ``exact_retrain`` keeps the offset at 0 so that it matches the original
    procedure as closely as the removed data permits -- its difference from
    theta_finetuned should come from the missing forget set, not from a
    different RNG stream on top of it.
    """
    return int(seed) + int(CONTROLS[control]["training_seed_offset"])


def run_control(
    config: Dict[str, Any],
    control: str,
    seed: int,
    output_root: str,
    manifest: Optional[str] = None,
    finetuned_checkpoint: Optional[str] = None,
    skip_train: bool = False,
    skip_collect: bool = False,
) -> Dict[str, Any]:
    """Train one control model and score it against theta_finetuned.

    The scoring call is the ordinary one: ``theta_post`` takes the place of the
    unlearned checkpoint and ``theta_finetuned`` stays the reference, so ghat,
    the calibrated losses and every detector are computed by exactly the same
    code that scores an unlearning method. That identity is the point -- a
    control scored by a different path would not be a control.
    """
    if control not in CONTROLS:
        raise KeyError(f"Unknown control {control!r}; expected one of {sorted(CONTROLS)}")

    spec = CONTROLS[control]
    config = dict(config)
    config["seed"] = seed

    run_root = os.path.join(output_root, control, f"seed{seed}")
    os.makedirs(run_root, exist_ok=True)

    split_info = load_or_build_splits(config, seed, manifest)
    splits = split_info["splits"]
    train_samples = control_training_samples(splits, control)

    finetuned_checkpoint = finetuned_checkpoint or os.path.join(
        config["paths"]["checkpoints"], f"finetuned_seed{seed}"
    )
    if not checkpoint_exists(finetuned_checkpoint, config):
        raise FileNotFoundError(
            f"No pre-deletion checkpoint at {finetuned_checkpoint}. Both controls "
            f"are defined as a displacement FROM theta_finetuned, so it must "
            f"exist before either can be scored."
        )

    checkpoint = os.path.join(run_root, "model")
    training_seed = training_seed_for(control, seed)

    record: Dict[str, Any] = {
        "control": control,
        "description": spec["description"],
        "seed": seed,
        "training_seed": training_seed,
        "training_populations": list(spec["populations"]),
        "n_training_samples": len(train_samples),
        "counts": {k: len(v) for k, v in splits.items()},
        "finetuned_checkpoint": os.path.abspath(finetuned_checkpoint),
        "split_source": split_info["source"],
        "split_manifest": split_info.get("manifest_path"),
    }

    if skip_train and checkpoint_exists(checkpoint, config):
        print(f"[{control} seed {seed}] reusing {checkpoint}")
        record["training"] = {"reused": True, "save_path": os.path.abspath(checkpoint)}
    else:
        print(
            f"[{control} seed {seed}] training on "
            f"{len(train_samples)} samples from {list(spec['populations'])} "
            f"(training_seed={training_seed}) -> {checkpoint}"
        )
        # Seed immediately before training so the control's RNG stream is a
        # function of training_seed alone, not of whatever ran before it.
        set_seed(training_seed)
        model, tokenizer = load_base_model(config)
        if not is_full_finetuning(config):
            model = attach_lora(model, config)
        else:
            for param in model.parameters():
                param.requires_grad = True
        record["training"] = finetune(
            model, tokenizer, train_samples, config, checkpoint
        )
        free(model)

    with open(os.path.join(run_root, "control_run.json"), "w") as handle:
        json.dump(record, handle, indent=2, default=str)

    if not skip_collect:
        scores_path = os.path.join(run_root, "per_sample_scores.json")
        print(f"[{control} seed {seed}] scoring -> {scores_path}")
        collect_scores(
            config=config,
            seed=seed,
            method=control,
            unlearned_checkpoint=checkpoint,
            finetuned_checkpoint=finetuned_checkpoint,
            output_path=scores_path,
            unlearning_run=record,
            manifest_path=split_info.get("manifest_path") or manifest,
        )
        record["scores"] = os.path.abspath(scores_path)

    return record


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and score the exact-retraining and null-refit controls.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # the gold standard: what deletion looks like when it is real\n"
            "  $PY -m deletion_signals.controls --config arxivmia_confirm.yaml \\\n"
            "      --control exact_retrain --seeds 42 43 44 \\\n"
            "      --manifest 'results/seed{seed}/splits.json' \\\n"
            "      --output ds_confirm/controls\n\n"
            "  # the false-positive floor: nothing deleted, only the RNG changed\n"
            "  $PY -m deletion_signals.controls --config arxivmia_confirm.yaml \\\n"
            "      --control null_refit --seeds 42 43 44 \\\n"
            "      --manifest 'results/seed{seed}/splits.json' \\\n"
            "      --output ds_confirm/controls\n"
        ),
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--control", required=True, choices=sorted(CONTROLS),
                        help="Which control to run")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--output", default="./ds_confirm/controls")
    parser.add_argument("--manifest", default=None,
                        help="Split manifest; {seed} is substituted per seed")
    parser.add_argument("--finetuned-checkpoint", default=None,
                        help="Override the pre-deletion checkpoint path "
                             "({seed} is substituted)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--skip-train", action="store_true",
                        help="Reuse a cached control checkpoint if present")
    parser.add_argument("--skip-collect", action="store_true",
                        help="Train only; do not score")
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args(argv)


def _per_seed(pattern: Optional[str], seed: int) -> Optional[str]:
    return None if pattern is None else pattern.replace("{seed}", str(seed))


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    if args.device:
        config["device"] = args.device
    if args.max_samples is not None:
        config["scoring"]["max_samples_per_population"] = args.max_samples
    config = apply_cli_overrides(config, args.overrides)

    os.makedirs(args.output, exist_ok=True)
    records = []
    for seed in args.seeds:
        records.append(
            run_control(
                config=config,
                control=args.control,
                seed=seed,
                output_root=args.output,
                manifest=_per_seed(args.manifest, seed),
                finetuned_checkpoint=_per_seed(args.finetuned_checkpoint, seed),
                skip_train=args.skip_train,
                skip_collect=args.skip_collect,
            )
        )

    summary_path = os.path.join(args.output, f"{args.control}_summary.json")
    with open(summary_path, "w") as handle:
        json.dump(
            {
                "control": args.control,
                "description": CONTROLS[args.control]["description"],
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "seeds": args.seeds,
                "runs": records,
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
