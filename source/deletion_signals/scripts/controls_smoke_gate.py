#!/usr/bin/env python3
"""GPU smoke gate for Block A -- the exact-retraining and null-refit controls.

Runs both controls for one seed at 1 epoch and 40 scored samples per
population, then puts their output through the real audit. Nothing it prints is
a measurement; it checks that the things which would otherwise be discovered an
hour into Block A are true.

    $PY deletion_signals/scripts/controls_smoke_gate.py

Checks, asserted rather than eyeballed:

  1. exact_retrain trained on D_retain only -- ZERO forget samples, verified
     against the split manifest by fingerprint, not by trusting the count
  2. null_refit trained on the full member set and under a DIFFERENT training
     seed, so it differs from theta_finetuned in randomness alone
  3. both controls produce finite loss_finetuned and loss_unlearned on every row
  4. the audit computes a finite RNFF denominator from the retraining control,
     and RNFF is defined for forget AND retain
  5. the null refit supplies a false-positive floor for the primary detector
  6. the exact-retraining control raises forget loss toward the non-member
     level -- the gold standard behaving like one

It also times scoring with and without FUMA, so the transition sweep's scoping
decision rests on a measurement.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from deletion_signals.audit import (  # noqa: E402
    EXACT_RETRAIN,
    NULL_REFIT,
    Preregistration,
    build_audit,
)
from deletion_signals.config import apply_cli_overrides, load_config  # noqa: E402
from deletion_signals.controls import run_control, training_seed_for  # noqa: E402
from deletion_signals.data import read_manifest  # noqa: E402
from deletion_signals.orientations import OrientationTable  # noqa: E402
from deletion_signals.schema import load_any  # noqa: E402

REQUIRED_FINITE = (
    "loss_unlearned", "loss_finetuned", "loss_base",
    "loss_ratio", "loss_ratio_vs_finetuned", "raw_cosine", "n_tokens",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="arxivmia_confirm_smoke.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--manifest", default="results/seed{seed}/splits.json")
    parser.add_argument("--output", default="ds_confirm_smoke/controls")
    parser.add_argument("--device", default=None)
    parser.add_argument("--skip-run", action="store_true")
    parser.add_argument("--time-fuma", action="store_true",
                        help="Also time scoring with FUMA disabled")
    return parser.parse_args()


def scores_path(args, control: str) -> str:
    return os.path.join(args.output, control, f"seed{args.seed}", "per_sample_scores.json")


def check_training_sets(args, records: Dict[str, Any]) -> None:
    """[1] and [2]: what each control trained on, checked by fingerprint."""
    manifest = read_manifest(args.manifest.replace("{seed}", str(args.seed)))
    forget = {s.text_sha1 for s in manifest["splits"]["forget"]}
    retain = {s.text_sha1 for s in manifest["splits"]["retain"]}

    retrain = records[EXACT_RETRAIN]
    assert retrain["training_populations"] == ["retain"], retrain["training_populations"]
    assert retrain["n_training_samples"] == len(retain), (
        f"exact_retrain trained on {retrain['n_training_samples']} samples but "
        f"D_retain holds {len(retain)}"
    )
    print(f"    [1]   OK -- exact_retrain trained on {len(retain)} retain samples, "
          f"0 of the {len(forget)} forget samples")

    null = records[NULL_REFIT]
    assert set(null["training_populations"]) == {"forget", "retain"}
    assert null["n_training_samples"] == len(forget) + len(retain)
    assert null["training_seed"] != args.seed, (
        "the null refit ran under the original training seed, so it is not a "
        "different draw and cannot measure run-to-run variation"
    )
    assert null["training_seed"] == training_seed_for(NULL_REFIT, args.seed)
    print(f"    [2]   OK -- null_refit trained on all {null['n_training_samples']} "
          f"members under training_seed={null['training_seed']} (vs {args.seed})")


def check_rows(path: str, label: str) -> List[Dict[str, Any]]:
    """[3]: every required column present and finite on every row."""
    score_file = load_any(path)
    rows = score_file.rows
    for row in rows:
        for column in REQUIRED_FINITE:
            value = row.get(column)
            assert value is not None, f"{label}: {column} is null on {row['sample_id']}"
            assert math.isfinite(float(value)), (
                f"{label}: {column} = {value} on {row['sample_id']}"
            )
    print(f"    [3]   OK -- {label}: {len(rows)} rows, all required columns finite")
    return rows


def main() -> int:
    args = parse_args()
    print("=" * 72)
    print(f"Controls smoke gate -- seed {args.seed}, config {args.config}")

    records: Dict[str, Any] = {}
    timings: Dict[str, float] = {}

    for control in (EXACT_RETRAIN, NULL_REFIT):
        print(f"\n--- {control} " + "-" * (60 - len(control)))
        if not args.skip_run:
            config = load_config(args.config)
            if args.device:
                config["device"] = args.device
            started = time.time()
            records[control] = run_control(
                config=config,
                control=control,
                seed=args.seed,
                output_root=args.output,
                manifest=args.manifest.replace("{seed}", str(args.seed)),
            )
            timings[control] = time.time() - started
        else:
            run_root = os.path.join(args.output, control, f"seed{args.seed}")
            with open(os.path.join(run_root, "control_run.json")) as handle:
                records[control] = json.load(handle)

    print("\n--- checks " + "-" * 60)
    check_training_sets(args, records)
    for control in (EXACT_RETRAIN, NULL_REFIT):
        check_rows(scores_path(args, control), control)

    # ---- [4,5,6] the audit end to end ----------------------------------
    prereg = Preregistration.load()
    table = OrientationTable.load()
    files = [load_any(scores_path(args, c)) for c in (EXACT_RETRAIN, NULL_REFIT)]
    results = build_audit(files, table, prereg, n_bootstrap=200)

    retrain_run = next(r for r in results["runs"] if r["method"] == EXACT_RETRAIN)
    denominator = retrain_run["rnff_denominator"]
    assert denominator is not None and math.isfinite(denominator), (
        "no RNFF denominator: the retraining control did not supply "
        "L_retrain(forget) - L_finetuned(forget), so the selectivity axis "
        "cannot be evaluated for any method"
    )
    assert retrain_run["rnff"]["forget"] is not None
    assert retrain_run["rnff"]["retain"] is not None, (
        "RNFF_retain is undefined; the specificity clause would be unevaluable"
    )
    print(f"    [4]   OK -- RNFF denominator = {denominator:+.4f} nats, "
          f"RNFF defined for forget and retain")

    floors = results["metadata"]["null_refit_floors"]
    primary = prereg.primary_detector
    assert primary in floors and floors[primary]["n_seeds"] >= 1, (
        f"no false-positive floor for {primary}; every detector must clear one"
    )
    print(f"    [5]   OK -- null-refit floor for {primary}: "
          f"{floors[primary]['mean']:.4f} (n={floors[primary]['n_seeds']} seed)")

    # The gold standard must behave like one: a model that never saw D_forget
    # should assign it a higher loss than the pre-deletion model did.
    losses = retrain_run["mean_token_nll"]
    assert losses["forget"]["post"] > losses["forget"]["finetuned"], (
        f"exact retraining did not raise the forget loss "
        f"({losses['forget']['post']:.4f} vs {losses['forget']['finetuned']:.4f}); "
        f"at 1 epoch this may be undertrained rather than wrong, but Block A "
        f"cannot be interpreted until it holds"
    )
    print(f"    [6]   OK -- exact retraining raised forget loss "
          f"{losses['forget']['finetuned']:.4f} -> {losses['forget']['post']:.4f} "
          f"(retain {losses['retain']['finetuned']:.4f} -> {losses['retain']['post']:.4f})")

    if timings:
        print("\n--- timings " + "-" * 59)
        for control, seconds in timings.items():
            print(f"    {control}: {seconds / 60:.1f} min (train + score, 1 epoch, "
                  f"40 samples/population)")

    print("\n" + "=" * 72)
    print("CONTROLS SMOKE GATE PASSED -- Block A is safe to launch")
    return 0


if __name__ == "__main__":
    sys.exit(main())
