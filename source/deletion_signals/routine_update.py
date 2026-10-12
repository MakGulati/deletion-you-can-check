"""The routine-update control: a checkpoint that changed for a legitimate reason.

What the existing controls do and do not cover
-----------------------------------------------
The **null refit** answers "what does a detector report when two models were
trained independently on the same data and nothing was deleted?" -- it measures
training randomness. The **exact-retraining** control answers "what does deletion
look like when it is real?"

Neither answers the question a deployed audit actually faces. In production a
checkpoint changes for many reasons, almost none of them deletion: a maintenance
pass, a top-up on new records, a routine refresh. A deletion detector that fires
on every one of those has no operational value, however well it separates forget
from retain in a study where the only thing that ever changed was deletion.

This module supplies that missing floor. Four conditions, of which two are new::

    1  no deletion + null refit      (existing -- v2 control, reused unchanged)
    2  no deletion + routine update  (NEW)
    3  exact deletion only           (existing -- v2 control, reused unchanged)
    4  exact deletion + routine update (NEW)

2 versus 1 asks whether an ordinary update raises the false-positive floor above
what training randomness alone produces. 4 versus 3 asks whether genuine deletion
is still recoverable once it is mixed with ordinary maintenance. Conditions 2 and
4 receive the **identical** update -- same data, same objective, same learning
rate, same step count, same RNG stream -- so the only difference between them is
whether a deletion happened underneath it.

Which records the update trains on, and the confound that forces
-----------------------------------------------------------------
ArxivMIA supplies exactly 1000 members and 1000 non-members, and the split
manifests already consume ALL 1000 members (200 forget + 800 retain). There are
no unused member records. The only records in the corpus that are genuinely
fresh *and* disjoint from every scored population are the non-members beyond the
scoring cap: scoring takes ``test`` by deterministic prefix and caps it at 300,
so ``test[300:500]`` is never scored under any configuration in this study.

That is what the update trains on, and it carries a confound this dataset cannot
remove: those records are ArxivMIA **non-members**, so the update nudges the
model toward the non-member distribution and can move the membership axis for a
reason that has nothing to do with deletion. The membership axis is therefore
reported for both update conditions beside the same axis without the update, so
the shift is visible as a measurement rather than inferred. The alternative --
continuing on retain data -- was rejected because retain is the negative class
of the primary task, so the update would not be disjoint from what it is scored
against.

:func:`assert_disjoint_from_scored` enforces the disjointness by fingerprint and
refuses to train if it does not hold. A silent overlap here would put training
data into the population the detector is evaluated on, which is the one failure
mode that would make every number in Component D meaningless.

The reference checkpoint is always theta_finetuned
--------------------------------------------------
For every condition, including condition 4, scoring compares against
``theta_finetuned`` -- ghat = theta_post - theta_finetuned, and
``loss_ratio_vs_finetuned`` divides by the same model. That is deliberate: the
auditor holds the pre-deletion checkpoint and the published one, whatever
happened in between. Scoring condition 4 against its own exact-retraining
intermediate would answer a question no auditor is in a position to ask.

Usage
-----
::

    $PY -m deletion_signals.routine_update --config arxivmia_routine_update.yaml \\
        --condition routine_update --seeds 42 43 44 \\
        --manifest 'results/seed{seed}/splits.json' \\
        --output fllm_audit/routine_update
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .collect import collect_scores
from .config import apply_cli_overrides, effective_config, is_full_finetuning, load_config
from .data import Sample, load_or_build_splits
from .models import checkpoint_exists, free, load_base_model, load_checkpoint_auto
from .reproducibility import run_provenance, set_seed
from .train import finetune

# Where each condition starts from. The update applied on top is identical in
# both cases -- that identity is what makes the pair a controlled comparison.
CONDITIONS: Dict[str, Dict[str, Any]] = {
    "routine_update": {
        "base": "finetuned",
        "deletion": "none",
        "description": (
            "no deletion + routine update: theta_finetuned continued on a "
            "disjoint batch of fresh ArxivMIA records. NOTHING was deleted and "
            "no forget record was targeted, so any deletion signal here is a "
            "false positive produced by an ordinary maintenance operation."
        ),
    },
    "exact_retrain_routine_update": {
        "base": "exact_retrain",
        "deletion": "exact",
        "description": (
            "exact deletion + routine update: the exact-retraining control, "
            "then the identical routine update. Deletion is real and complete; "
            "the question is whether it is still recoverable once ordinary "
            "maintenance has been applied on top of it."
        ),
    },
}


# ------------------------------------------------------------------- the slice


def scored_test_cap(config: Mapping[str, Any]) -> int:
    """How many test samples scoring evaluates, i.e. where the safe tail begins.

    Raises when no cap is configured. Without one, scoring evaluates the whole
    test population and there is no tail that is provably never scored -- so
    there is no honest place to draw update data from, and guessing would put
    training data into a scored population.
    """
    cap = config.get("scoring", {}).get("max_samples_per_population")
    if isinstance(cap, Mapping):
        cap = cap.get("test")
    if cap is None:
        raise ValueError(
            "routine_update needs scoring.max_samples_per_population to cap the "
            "'test' population. Its update data are the test records BEYOND that "
            "cap, which are never scored; with no cap, every test record is "
            "scored and there is no disjoint slice to update on."
        )
    return int(cap)


def member_tail_samples(
    config: Mapping[str, Any], seed: int, n_samples: int
) -> List[Any]:
    """Members the split never drew -- the clean update source.

    ``build_splits`` draws ``data.train_samples`` members from the corpus and
    partitions those into forget and retain. On a corpus with more members than
    that draw, everything left over is fresh: never trained on, never scored,
    and -- unlike the test tail -- **from the same population as the training
    data**, so an update on it carries no distribution shift.

    That is what makes this the clean version of the control. The test-tail
    source has to be used when the member pool is exhausted (ArxivMIA draws all
    1000 of its members), and it brings a declared confound with it: the update
    also moves the model toward the non-member distribution, which can shift the
    membership axis for a reason unrelated to deletion.

    Selection is deterministic: the same corpus order and the same seed rebuild
    the same draw, the drawn texts are removed by fingerprint, and the remainder
    is taken by prefix. No second RNG is introduced.
    """
    import random

    from datasets import load_dataset

    from .data import _infer_split, _infer_text_field
    from .schema import text_fingerprint

    data_cfg = config["data"]
    dataset_name = data_cfg["name"]
    split = _infer_split(dataset_name, data_cfg.get("split"))
    label_field = data_cfg.get("label_field", "label")

    dataset = load_dataset(dataset_name, split=split)
    text_field = data_cfg.get("text_field") or _infer_text_field(
        dataset_name, dataset.column_names
    )
    members = [row for row in dataset if row[label_field] == 1]

    # Reproduce build_splits' draw exactly, then take what it left behind.
    rng = random.Random(seed)
    n_train = min(int(data_cfg["train_samples"]), len(members))
    drawn = rng.sample(members, n_train)
    drawn_prints = {text_fingerprint(str(row[text_field])) for row in drawn}

    remainder = [
        row for row in members
        if text_fingerprint(str(row[text_field])) not in drawn_prints
    ]
    if len(remainder) < n_samples:
        raise ValueError(
            f"routine_update.source='member_tail' needs {n_samples} members "
            f"beyond the split's draw of {n_train}, but only {len(remainder)} "
            f"remain in {dataset_name!r}. Use 'test_tail' on a corpus whose "
            f"member pool the split consumes (ArxivMIA), and declare the "
            f"non-member confound that brings."
        )

    return [
        Sample.create(index, str(row[text_field]), "update")
        for index, row in enumerate(remainder[:n_samples])
    ]


def update_samples(
    config: Mapping[str, Any], splits: Mapping[str, Sequence], seed: Optional[int] = None
) -> List[Any]:
    """The routine update's training records, from the configured source.

    Both sources are deterministic functions of the configuration and the seed,
    reproducible without a second RNG whose state would have to be recorded, and
    identical for every condition at the same configuration -- which is what
    makes conditions 2 and 4 receive the same update.
    """
    settings = config.get("routine_update", {})
    n_samples = int(settings.get("n_samples", 200))
    source = str(settings.get("source", "test_tail"))

    if source == "member_tail":
        if seed is None:
            raise ValueError(
                "routine_update.source='member_tail' needs the seed: the draw it "
                "excludes is a function of it."
            )
        return member_tail_samples(config, seed, n_samples)

    offset = scored_test_cap(config)
    test = list(splits["test"])

    selected = test[offset : offset + n_samples]
    if len(selected) < n_samples:
        raise ValueError(
            f"routine_update.n_samples={n_samples} needs test records "
            f"[{offset}:{offset + n_samples}] but the split has only "
            f"{len(test)}. Either lower n_samples or widen data.test_samples; "
            f"taking fewer would silently change the update between seeds."
        )
    return selected


def assert_disjoint_from_scored(
    update: Sequence[Any], splits: Mapping[str, Sequence], config: Mapping[str, Any]
) -> Dict[str, Any]:
    """Fail unless the update records are disjoint from everything scored.

    Checked by text fingerprint rather than by index: the two are only the same
    thing while the slicing above is correct, and this function exists precisely
    to catch the case where it is not. Returns the overlap counts so a run record
    can show the check was performed and what it found.
    """
    from .schema import text_fingerprint

    update_prints = {text_fingerprint(s.text) for s in update}
    cap = scored_test_cap(config)

    scored = {
        "forget": {text_fingerprint(s.text) for s in splits["forget"]},
        # The whole retain population: its head is the Fisher curvature holdout
        # and its tail is scored, so every retain record is off limits either
        # way.
        "retain": {text_fingerprint(s.text) for s in splits["retain"]},
        "test_scored": {text_fingerprint(s.text) for s in list(splits["test"])[:cap]},
    }

    # The unscored test tail is also off limits to a member_tail update and vice
    # versa, but neither can reach the other, so the three populations above are
    # the complete set of things an update must not touch.
    overlaps = {name: len(update_prints & prints) for name, prints in scored.items()}
    offenders = {name: n for name, n in overlaps.items() if n}
    if offenders:
        raise ValueError(
            f"The routine-update records overlap scored population(s) "
            f"{offenders}. Training on data the detector is then evaluated on "
            f"would make every Component D number meaningless. Check "
            f"routine_update.source, scoring.max_samples_per_population and "
            f"routine_update.n_samples."
        )

    source = str(config.get("routine_update", {}).get("source", "test_tail"))
    return {
        "n_update": len(update),
        "n_unique_update": len(update_prints),
        "overlap_with_scored": overlaps,
        "disjoint": True,
        "scored_test_cap": cap,
        "source": source,
        "population_of_update_records": (
            "members the split never drew -- same population as the training "
            "data, so the update carries no distribution shift"
            if source == "member_tail"
            else "NON-MEMBERS beyond the scoring cap -- declared confound: the "
                 "update also moves the model toward the non-member distribution"
        ),
    }


def update_training_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    """A config whose ``training`` block is the routine update's own schedule.

    ``finetune`` reads its schedule from ``config["training"]``; substituting the
    block rather than adding a parallel code path means the update goes through
    exactly the same optimiser, clipping and checkpoint-writing as every other
    fine-tune in this study.
    """
    settings = config.get("routine_update", {})
    merged = copy.deepcopy(dict(config))
    merged["training"] = {
        "epochs": int(settings.get("epochs", 3)),
        "batch_size": int(settings.get("batch_size", 8)),
        "gradient_accumulation_steps": int(
            settings.get("gradient_accumulation_steps", 4)
        ),
        "learning_rate": float(settings.get("learning_rate", 5.0e-5)),
        # Zero, not the fine-tune's 100: an update of ~18 optimiser steps under
        # a 100-step warmup would never reach its configured learning rate, so
        # the declared rate would not be the rate applied.
        "warmup_steps": int(settings.get("warmup_steps", 0)),
        "max_grad_norm": float(settings.get("max_grad_norm", 1.0)),
        "logging_steps": 10,
        "gradient_checkpointing": False,
    }
    return merged


def update_training_seed(seed: int) -> int:
    """The RNG seed the routine update runs under.

    Offset by a constant and shared by BOTH conditions, so the update is the
    same operation in each: same batch order, same dropout draws. A per-condition
    seed would make conditions 2 and 4 differ in their update as well as in the
    deletion underneath it, and the pair would no longer isolate anything.
    """
    return int(seed) + 2000


# --------------------------------------------------------------------- the run


def run_condition(
    config: Dict[str, Any],
    condition: str,
    seed: int,
    output_root: str,
    manifest: Optional[str] = None,
    finetuned_checkpoint: Optional[str] = None,
    exact_retrain_checkpoint: Optional[str] = None,
    skip_train: bool = False,
    skip_collect: bool = False,
) -> Dict[str, Any]:
    """Apply the routine update to one condition's base checkpoint, then score it."""
    if condition not in CONDITIONS:
        raise KeyError(
            f"Unknown condition {condition!r}; registered conditions are "
            f"{sorted(CONDITIONS)}"
        )

    spec = CONDITIONS[condition]
    config = dict(config)
    config["seed"] = seed

    run_root = os.path.join(output_root, condition, f"seed{seed}")
    os.makedirs(run_root, exist_ok=True)

    split_info = load_or_build_splits(config, seed, manifest)
    splits = split_info["splits"]

    update = update_samples(config, splits, seed=seed)
    disjointness = assert_disjoint_from_scored(update, splits, config)

    finetuned_checkpoint = finetuned_checkpoint or os.path.join(
        config["paths"]["checkpoints"], f"finetuned_seed{seed}"
    )
    if not checkpoint_exists(finetuned_checkpoint, config):
        raise FileNotFoundError(
            f"No pre-deletion checkpoint at {finetuned_checkpoint}. Every "
            f"condition is scored as a displacement FROM theta_finetuned, so it "
            f"must exist before any of them can be scored."
        )

    if spec["base"] == "finetuned":
        base_checkpoint = finetuned_checkpoint
    else:
        base_checkpoint = exact_retrain_checkpoint or os.path.join(
            "ds_confirm", "controls", "exact_retrain", f"seed{seed}", "model"
        )
        if not checkpoint_exists(base_checkpoint, config):
            raise FileNotFoundError(
                f"Condition {condition!r} starts from the exact-retraining "
                f"control, but no checkpoint exists at {base_checkpoint}. Run "
                f"deletion_signals.controls --control exact_retrain first, or "
                f"pass --exact-retrain-checkpoint."
            )

    checkpoint = os.path.join(run_root, "model")
    training_seed = update_training_seed(seed)
    settings = config.get("routine_update", {})

    record: Dict[str, Any] = {
        "control": condition,
        "condition": condition,
        "description": spec["description"],
        "deletion": spec["deletion"],
        "seed": seed,
        "training_seed": training_seed,
        "base_checkpoint": os.path.abspath(base_checkpoint),
        "base_role": spec["base"],
        "finetuned_checkpoint": os.path.abspath(finetuned_checkpoint),
        "update": {
            "objective": "causal language-model cross-entropy",
            "parameter_scope": "lora" if not is_full_finetuning(config) else "full",
            "learning_rate": float(settings.get("learning_rate", 5.0e-5)),
            "epochs": int(settings.get("epochs", 3)),
            "batch_size": int(settings.get("batch_size", 8)),
            "gradient_accumulation_steps": int(
                settings.get("gradient_accumulation_steps", 4)
            ),
            "n_samples": len(update),
            "source": "test_tail",
            "data_composition": (
                "ArxivMIA non-member records beyond the test scoring cap; "
                "disjoint from forget, retain (scored and curvature holdout) "
                "and the scored test prefix"
            ),
            "sample_ids": [s.sample_id for s in update],
            "disjointness_check": disjointness,
        },
        "counts": {k: len(v) for k, v in splits.items()},
        "split_source": split_info["source"],
        "split_manifest": split_info.get("manifest_path"),
    }

    if skip_train and checkpoint_exists(checkpoint, config):
        print(f"[{condition} seed {seed}] reusing {checkpoint}")
        record["training"] = {"reused": True, "save_path": os.path.abspath(checkpoint)}
    else:
        print(
            f"[{condition} seed {seed}] routine update on {len(update)} disjoint "
            f"records from {os.path.basename(base_checkpoint)} "
            f"(training_seed={training_seed}) -> {checkpoint}"
        )
        # Seeded immediately before training so the update's RNG stream is a
        # function of training_seed alone -- identical in both conditions.
        set_seed(training_seed)
        base_model = None
        if is_full_finetuning(config):
            model, tokenizer = load_checkpoint_auto(
                config, base_checkpoint, is_trainable=True
            )
        else:
            base_model, tokenizer = load_base_model(config)
            model, _ = load_checkpoint_auto(
                config, base_checkpoint, is_trainable=True, base_model=base_model
            )
        record["training"] = finetune(
            model, tokenizer, update, update_training_config(config), checkpoint
        )
        free(model, base_model)

    with open(os.path.join(run_root, "control_run.json"), "w") as handle:
        json.dump(record, handle, indent=2, default=str)

    if not skip_collect:
        scores_path = os.path.join(run_root, "per_sample_scores.json")
        print(f"[{condition} seed {seed}] scoring -> {scores_path}")
        collect_scores(
            config=config,
            seed=seed,
            method=condition,
            unlearned_checkpoint=checkpoint,
            # Always theta_finetuned, including for condition 4: the auditor
            # holds the pre-deletion model and the published one, never the
            # intermediate.
            finetuned_checkpoint=finetuned_checkpoint,
            output_path=scores_path,
            unlearning_run=record,
            manifest_path=split_info.get("manifest_path") or manifest,
        )
        record["scores"] = os.path.abspath(scores_path)

    return record


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply a legitimate, non-deletion update to a checkpoint and score "
            "it, to measure detector false-positive floors under ordinary "
            "model maintenance."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # condition 2: nothing deleted, ordinary maintenance applied\n"
            "  $PY -m deletion_signals.routine_update \\\n"
            "      --config arxivmia_routine_update.yaml \\\n"
            "      --condition routine_update --seeds 42 43 44 \\\n"
            "      --manifest 'results/seed{seed}/splits.json' \\\n"
            "      --output fllm_audit/routine_update\n\n"
            "  # condition 4: real deletion, then the identical maintenance\n"
            "  $PY -m deletion_signals.routine_update \\\n"
            "      --config arxivmia_routine_update.yaml \\\n"
            "      --condition exact_retrain_routine_update --seeds 42 43 44 \\\n"
            "      --manifest 'results/seed{seed}/splits.json' \\\n"
            "      --output fllm_audit/routine_update\n"
        ),
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--condition", required=True, choices=sorted(CONDITIONS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--output", default="./fllm_audit/routine_update")
    parser.add_argument("--manifest", default=None,
                        help="Split manifest; {seed} is substituted per seed")
    parser.add_argument("--finetuned-checkpoint", default=None,
                        help="Override theta_finetuned ({seed} is substituted)")
    parser.add_argument("--exact-retrain-checkpoint", default=None,
                        help="Override the exact-retraining control checkpoint "
                             "({seed} is substituted)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-collect", action="store_true")
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
            run_condition(
                config=config,
                condition=args.condition,
                seed=seed,
                output_root=args.output,
                manifest=_per_seed(args.manifest, seed),
                finetuned_checkpoint=_per_seed(args.finetuned_checkpoint, seed),
                exact_retrain_checkpoint=_per_seed(
                    args.exact_retrain_checkpoint, seed
                ),
                skip_train=args.skip_train,
                skip_collect=args.skip_collect,
            )
        )

    summary_path = os.path.join(args.output, f"{args.condition}_summary.json")
    with open(summary_path, "w") as handle:
        json.dump(
            {
                "condition": args.condition,
                "description": CONDITIONS[args.condition]["description"],
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
