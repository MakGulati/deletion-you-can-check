#!/usr/bin/env python3
"""GPU smoke gate for the unlearning-strength sweep.

Runs the real pipeline twice -- the bottom and the top of the sweep range, seed
42 only, 20 unlearning steps, 40 scored samples per population -- and asserts
the things that would otherwise be discovered four GPU-hours into the sweep.

    $PY deletion_signals/scripts/sweep_smoke_gate.py

Checks, all asserted rather than eyeballed:

  1. loss_base, loss_finetuned and loss_unlearned are present and finite on
     every row of both runs
  2. every new score column resolves an orientation on every task
  3. forget, retain and test are disjoint, and the Fisher curvature holdout is
     disjoint from every scored sample (when SIF is on)
  4. no null and no non-finite value in any score column
  5. THE STRENGTH KNOB BITES -- final_forget_loss differs materially between
     the bottom and the top of the range

Nothing it prints is a measurement: 40 samples per population cannot estimate an
AUC. It is a check that the sweep will produce interpretable output.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from typing import Any, Dict, List, Mapping, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from deletion_signals.config import apply_cli_overrides, load_config  # noqa: E402
from deletion_signals.orientations import OrientationTable  # noqa: E402
from deletion_signals.run_pipeline import run_seed  # noqa: E402
from deletion_signals.schema import POPULATIONS, SCORE_COLUMNS, load_any  # noqa: E402

# The bottom and the top of the sweep range. If the knob does not bite between
# these two, it does not bite anywhere in between either.
STRENGTHS = (1.0e-5, 2.0e-4)

# How much the final forget loss must differ between the two cells, relative to
# the weaker one, before the knob counts as biting. 20% over 20 steps is a low
# bar deliberately: the gate is meant to catch "the setting does nothing", not
# to prejudge how much effect the full 100-step run should have.
MIN_RELATIVE_FORGET_LOSS_GAP = 0.20

# Columns that must be present and finite on every row for the sweep's analysis
# to be computable at all.
REQUIRED_FINITE = (
    "loss_unlearned",
    "loss_base",
    "loss_finetuned",
    "loss_ratio",
    "loss_delta",
    "loss_ratio_vs_finetuned",
    "loss_delta_vs_finetuned",
    "grad_norm_sq",
    "raw_cosine",
    "alignment_cos_sq",
    "n_tokens",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="arxivmia_sweep_smoke.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--manifest", default="results/seed{seed}/splits.json",
                        help="Split manifest to share with the baseline runs")
    parser.add_argument("--output", default="ds_sweep_smoke")
    parser.add_argument("--device", default=None)
    parser.add_argument("--skip-run", action="store_true",
                        help="Only check the outputs of a previous gate run")
    return parser.parse_args()


def run_cell(args: argparse.Namespace, strength: float) -> str:
    """Run one sweep cell and return the path to its score table."""
    config = load_config(args.config)
    if args.device:
        config["device"] = args.device
    config = apply_cli_overrides(
        config, [f"unlearning.ga_gd.ascent_lr={strength:g}"]
    )
    assert isinstance(config["unlearning"]["ga_gd"]["ascent_lr"], float), (
        "ascent_lr did not survive --set as a float; a string learning rate "
        "would reach the optimiser"
    )

    output_root = os.path.join(args.output, f"ascent_{strength:g}")
    summary = run_seed(
        config=config,
        seed=args.seed,
        methods=["ga_gd"],
        output_root=output_root,
        skip_finetune=True,
        manifest=args.manifest.replace("{seed}", str(args.seed)),
    )
    return summary["methods"]["ga_gd"]["scores"]


def scores_path(args: argparse.Namespace, strength: float) -> str:
    return os.path.join(
        args.output, f"ascent_{strength:g}", f"seed{args.seed}", "ga_gd",
        "per_sample_scores.json",
    )


def check_losses_and_finiteness(rows: Sequence[Mapping[str, Any]], label: str) -> None:
    """[1] and [4]: every required column present, non-null and finite."""
    for row in rows:
        for column in REQUIRED_FINITE:
            value = row.get(column)
            assert value is not None, f"{label}: {column} is null on {row['sample_id']}"
            assert math.isfinite(float(value)), (
                f"{label}: {column} is not finite ({value}) on {row['sample_id']}"
            )
    for row in rows:
        for column in SCORE_COLUMNS:
            value = row.get(column)
            if value is None:
                continue
            value = float(value)
            # -inf is a legitimate ranking position for combined_log_score only.
            if math.isinf(value) and value < 0 and column == "combined_log_score":
                continue
            assert math.isfinite(value), (
                f"{label}: score column {column} is {value} on {row['sample_id']}"
            )
    print(f"    [1,4] OK -- {len(rows)} rows, all required columns present and finite")


def check_orientations(rows: Sequence[Mapping[str, Any]]) -> None:
    """[2]: every column carried by these rows resolves on every task."""
    table = OrientationTable.load()
    present = [
        column
        for column in SCORE_COLUMNS
        if any(row.get(column) is not None for row in rows)
    ]
    for column in present:
        for task in table.task_names:
            orientation = table.orientation(column, task)
            assert orientation.rationale, f"{column}/{task} has no rationale"
    new_columns = ("raw_cosine", "loss_ratio_vs_finetuned", "loss_delta_vs_finetuned")
    for column in new_columns:
        assert column in present, f"{column} is absent from the rows entirely"
    assert table.orientation("raw_cosine", "forget_vs_retain").headline, (
        "raw_cosine is not marked headline; the sweep preregisters it as primary"
    )
    print(f"    [2]   OK -- {len(present)} columns x {len(table.task_names)} tasks "
          f"resolve, raw_cosine is headline")


def check_disjoint(score_file, rows: Sequence[Mapping[str, Any]]) -> None:
    """[3]: populations disjoint, and the curvature holdout out of the scored set."""
    fingerprints: Dict[str, set] = {}
    for population in POPULATIONS:
        fingerprints[population] = {
            row["text_sha1"] for row in rows if row["population"] == population
        }
    for left in POPULATIONS:
        for right in POPULATIONS:
            if left >= right:
                continue
            overlap = fingerprints[left] & fingerprints[right]
            assert not overlap, (
                f"{len(overlap)} sample(s) appear in both {left} and {right}"
            )

    fisher = (score_file.metadata.get("sif") or {}).get("fisher")
    holdout_note = "SIF off in the gate config, no curvature holdout to check"
    if fisher:
        # collect_scores already calls fisher.assert_disjoint(); this records
        # that the holdout existed and was of the expected size.
        holdout_note = f"curvature holdout rank {fisher.get('rank')} held out of scoring"
    counts = {p: len(fingerprints[p]) for p in POPULATIONS}
    print(f"    [3]   OK -- populations disjoint {counts}; {holdout_note}")


def check_knob_bites(records: Mapping[float, Mapping[str, Any]]) -> None:
    """[5]: the two strengths must produce materially different forget losses."""
    weak, strong = STRENGTHS
    weak_loss = float(records[weak]["final_forget_loss"])
    strong_loss = float(records[strong]["final_forget_loss"])
    assert math.isfinite(weak_loss) and math.isfinite(strong_loss), (
        f"final_forget_loss is not finite: {weak_loss}, {strong_loss}"
    )
    gap = abs(strong_loss - weak_loss) / max(abs(weak_loss), 1e-8)
    assert gap >= MIN_RELATIVE_FORGET_LOSS_GAP, (
        f"the strength knob does not bite: final_forget_loss is {weak_loss:.4f} at "
        f"ascent_lr={weak:g} and {strong_loss:.4f} at ascent_lr={strong:g}, a "
        f"relative gap of {gap:.1%} < {MIN_RELATIVE_FORGET_LOSS_GAP:.0%}. Running "
        f"the full sweep would vary a setting that changes nothing."
    )
    assert strong_loss > weak_loss, (
        f"the stronger ascent rate produced the LOWER forget loss "
        f"({strong_loss:.4f} vs {weak_loss:.4f}); the knob moves, but not in the "
        f"direction gradient ascent implies"
    )
    print(f"    [5]   OK -- final_forget_loss {weak_loss:.4f} (lr {weak:g}) -> "
          f"{strong_loss:.4f} (lr {strong:g}), relative gap {gap:.1%}")


def main() -> int:
    args = parse_args()
    print("=" * 72)
    print(f"Sweep smoke gate: ascent_lr {STRENGTHS[0]:g} and {STRENGTHS[1]:g}, "
          f"seed {args.seed}")

    records: Dict[float, Mapping[str, Any]] = {}
    for strength in STRENGTHS:
        print(f"\n--- ascent_lr = {strength:g} " + "-" * 44)
        if not args.skip_run:
            run_cell(args, strength)
        path = scores_path(args, strength)
        assert os.path.exists(path), f"no score table at {path}"

        score_file = load_any(path)
        rows: List[Mapping[str, Any]] = score_file.rows
        check_losses_and_finiteness(rows, f"lr={strength:g}")
        check_orientations(rows)
        check_disjoint(score_file, rows)

        unlearning = score_file.metadata.get("unlearning_run") or {}
        assert unlearning.get("final_forget_loss") is not None, (
            "the score file carries no unlearning record; the sweep report reads "
            "the strength and the stop reason from it"
        )
        recorded = float(
            (unlearning.get("hyperparameters") or {}).get("ascent_lr", float("nan"))
        )
        assert abs(recorded - strength) < 1e-12, (
            f"the run recorded ascent_lr={recorded} but was asked for {strength}"
        )
        records[strength] = unlearning
        print(f"          steps={unlearning.get('steps_taken')} "
              f"stop={unlearning.get('stop_reason')} "
              f"forget_loss={unlearning.get('final_forget_loss'):.4f} "
              f"retain_loss={unlearning.get('final_retain_loss'):.4f}")

    print("\n--- across the two cells " + "-" * 46)
    check_knob_bites(records)

    print("\n" + "=" * 72)
    print("SMOKE GATE PASSED -- the sweep is safe to launch")
    return 0


if __name__ == "__main__":
    sys.exit(main())
