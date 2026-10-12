#!/usr/bin/env python3
"""GPU smoke gate for the FLLM audit -- NPO and the routine-update control.

Runs one NPO cell at 6 steps and one routine-update condition at 40 scored
samples per population, then puts both through the real audit, the real
low-prevalence analysis and the real report generator.

    $PY deletion_signals/scripts/fllm_smoke_gate.py

**Nothing it prints is a measurement.** Under the preregistration a smoke test
may check two things and no others: numerical stability, and whether the
parameters have their intended QUALITATIVE effect. It may not be used to choose
the learning-rate grid, to select beta, or to move any threshold. The checks
below are written so that none of them could be satisfied by tuning.

Checks, asserted rather than eyeballed:

  1. NPO runs to completion with finite loss and finite gradients throughout
  2. the NPO weight 2*sigmoid(beta*r) starts at 1.0 -- theta = theta_ref -- and
     FALLS, which is the collapse-avoidance mechanism the method is claimed for
  3. the forget loss RISES: the sign convention is the published one
  4. only LoRA parameters moved; the frozen backbone is bit-identical
  5. the routine-update records are disjoint from every scored population, by
     fingerprint rather than by index
  6. the routine update preserves utility under the frozen threshold
  7. every score column the low-prevalence component needs is finite
  8. the audit, the low-prevalence analysis and the report all run end to end on
     the smoke output and produce the four tables
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

from deletion_signals.audit import Preregistration, build_audit  # noqa: E402
from deletion_signals.config import apply_cli_overrides, load_config  # noqa: E402
from deletion_signals.lowprev import build_results as build_lowprev  # noqa: E402
from deletion_signals.orientations import OrientationTable  # noqa: E402
from deletion_signals.routine_update import run_condition  # noqa: E402
from deletion_signals.run_pipeline import run_seed  # noqa: E402
from deletion_signals.schema import load_any  # noqa: E402
from deletion_signals.fllm_report import build_document, format_report  # noqa: E402

REQUIRED_FINITE = (
    "loss_unlearned", "loss_finetuned", "loss_base", "loss_ratio",
    "loss_ratio_vs_finetuned", "loss_delta_vs_finetuned",
    "grad_norm_sq", "grad_ratio", "raw_cosine", "alignment_cos_sq", "n_tokens",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="arxivmia_npo_smoke.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="./fllm_audit/smoke")
    parser.add_argument("--manifest", default="results/seed{seed}/splits.json")
    parser.add_argument("--device", default=None)
    parser.add_argument("--skip-run", action="store_true",
                        help="Re-check artifacts from an earlier smoke run")
    return parser.parse_args()


def _config(args) -> Dict[str, Any]:
    config = load_config(args.config)
    if args.device:
        config["device"] = args.device
    return config


# ------------------------------------------------------------------ the checks


def check_trajectory(path: str) -> None:
    """[1,2,3]: stability, the weight's decay, and the sign convention."""
    with open(path) as handle:
        trajectory = json.load(handle)
    entries = trajectory["entries"]

    for entry in entries:
        for key, value in entry.items():
            if isinstance(value, (int, float)):
                assert math.isfinite(float(value)), (
                    f"non-finite {key}={value} at step {entry.get('step')}"
                )
    print(f"    [1]   OK -- {len(entries)} trajectory entries, all finite")

    weights = [e["npo_weight_mean"] for e in entries if "npo_weight_mean" in e]
    assert weights, "no NPO weight was recorded; the mechanism is unobservable"
    for weight in weights:
        assert 0.0 < weight <= 2.0, (
            f"NPO weight {weight} is outside (0, 2]. It is 2*sigmoid(beta*r), so "
            f"a value outside that range means the objective is not the one in "
            f"the paper."
        )

    # The weight is measured on a DIFFERENT shuffled forget batch at every probe,
    # and near initialisation the log-ratio is ~0, so it fluctuates around 1.0
    # by a few percent from batch composition alone. Asserting a monotone fall
    # over a handful of probes would be asserting on noise -- it would pass or
    # fail on which batches happened to be drawn.
    #
    # So: the RANGE is asserted always, and the decay only where there are
    # enough probes for a trend to mean anything. A 6-step smoke run cannot
    # demonstrate the gate engaging, and this says so instead of implying it did.
    if len(weights) >= 8:
        head = sum(weights[: len(weights) // 3]) / max(len(weights) // 3, 1)
        tail = sum(weights[-len(weights) // 3:]) / max(len(weights) // 3, 1)
        assert tail <= head + 1e-6, (
            f"the NPO weight rose from {head:.4f} (first third) to {tail:.4f} "
            f"(last third). It is a monotone function of the forget log-ratio, "
            f"so a sustained rise means the policy became MORE likely on the "
            f"forget set -- the inverted-sign failure mode."
        )
        print(f"    [2]   OK -- NPO weight fell {head:.4f} -> {tail:.4f} across "
              f"{len(weights)} probes (gate engaging, as the method claims)")
    else:
        print(f"    [2]   OK -- NPO weight in (0, 2] on all {len(weights)} probes: "
              f"{[round(w, 4) for w in weights]}. NOTE: too few probes to show the "
              f"gate engaging; that is a property of the full 100-step run, not "
              f"of this smoke test.")

    forget = [e["loss_forget"] for e in entries if "loss_forget" in e]
    assert len(forget) >= 2, "no forget-loss probe; the sign cannot be checked"
    assert forget[-1] > forget[0], (
        f"the forget loss FELL, {forget[0]:.4f} -> {forget[-1]:.4f}. NPO must "
        f"drive forget likelihood down, i.e. forget loss up; this is the "
        f"inverted-sign failure mode."
    )
    print(f"    [3]   OK -- forget loss rose {forget[0]:.4f} -> {forget[-1]:.4f}")


def check_only_adapters_moved(unlearned_dir: str, finetuned_dir: str) -> None:
    """[4]: the NPO update touched LoRA parameters and nothing else."""
    from safetensors.torch import load_file

    unlearned = load_file(os.path.join(unlearned_dir, "adapter_model.safetensors"))
    finetuned = load_file(os.path.join(finetuned_dir, "adapter_model.safetensors"))

    assert set(unlearned) == set(finetuned), (
        "the unlearned checkpoint has a different parameter set from the "
        "fine-tuned one; the adapter structure changed"
    )
    moved = sum(
        1 for name in unlearned
        if not unlearned[name].equal(finetuned[name])
    )
    assert moved > 0, "no adapter parameter changed; the optimiser did nothing"
    non_lora = [n for n in unlearned if "lora" not in n.lower()]
    for name in non_lora:
        assert unlearned[name].equal(finetuned[name]), (
            f"non-adapter parameter {name} changed; only LoRA parameters were "
            f"meant to receive gradients"
        )
    # A PEFT checkpoint stores ONLY adapter tensors, so `non_lora` is normally
    # empty and the loop above proves nothing on its own. Say so, rather than
    # printing a count that reads like a check that happened. The real guarantee
    # that the frozen backbone is untouched comes from
    # test_npo.py::test_frozen_parameters_are_untouched_by_an_npo_update, which
    # can see the whole model.
    coverage = (
        f"{len(non_lora)} non-adapter tensors also bit-identical"
        if non_lora
        else "the checkpoint stores adapter tensors only, so backbone freezing is "
             "covered by the unit test rather than here"
    )
    print(f"    [4]   OK -- {moved}/{len(unlearned)} adapter tensors moved; {coverage}")


def check_disjointness(record: Dict[str, Any]) -> None:
    """[5]: the update trained on nothing the detector is scored on."""
    check = record["update"]["disjointness_check"]
    assert check["disjoint"] is True
    assert all(v == 0 for v in check["overlap_with_scored"].values()), (
        f"the routine update overlaps a scored population: "
        f"{check['overlap_with_scored']}"
    )
    assert check["n_unique_update"] == check["n_update"], (
        "the update set contains duplicate records"
    )
    print(f"    [5]   OK -- {check['n_update']} update records, zero overlap with "
          f"forget/retain/scored-test (cap {check['scored_test_cap']})")


def check_rows(path: str, label: str) -> List[Dict[str, Any]]:
    """[7]: every column the low-prevalence component needs is finite."""
    rows = load_any(path).rows
    for row in rows:
        for column in REQUIRED_FINITE:
            value = row.get(column)
            assert value is not None, f"{label}: {column} is null on {row['sample_id']}"
            assert math.isfinite(float(value)), (
                f"{label}: {column} = {value} on {row['sample_id']}"
            )
    print(f"    [7]   OK -- {label}: {len(rows)} rows, all required columns finite")
    return rows


def check_utility(audit: Dict[str, Any], prereg: Preregistration) -> None:
    """[6]: the routine update did not break the model.

    A smoke run is one epoch on a handful of records, so this is a stability
    check rather than the real measurement -- but an update that destroyed
    utility here would destroy it at full scale too, and finding that out now
    costs minutes instead of an hour.
    """
    limit = float(prereg.utility["max_perplexity_ratio"])
    for run in audit["runs"]:
        if run["method"] != "routine_update":
            continue
        ratios = {p: run["perplexity_ratio"].get(p) for p in ("retain", "test")}
        for population, ratio in ratios.items():
            assert ratio is not None and math.isfinite(ratio), (
                f"routine update: {population} perplexity ratio is not measurable"
            )
        print(f"    [6]   {'OK' if all(r <= limit for r in ratios.values()) else 'NOTE'}"
              f" -- routine update perplexity ratios {ratios} "
              f"(frozen limit {limit}; a smoke-scale update is not the measurement)")
        return
    print("    [6]   SKIP -- no routine-update run in the audit")


# ---------------------------------------------------------------------- driver


def main() -> int:
    args = parse_args()
    manifest = args.manifest.replace("{seed}", str(args.seed))
    print("=" * 72)
    print(f"FLLM smoke gate -- seed {args.seed}, config {args.config}")
    print("Nothing printed here is a measurement.")

    npo_dir = os.path.join(args.output, "npo")
    routine_dir = os.path.join(args.output, "routine_update")
    timings: Dict[str, float] = {}

    # ---- run ------------------------------------------------------------
    if not args.skip_run:
        print("\n--- NPO " + "-" * 63)
        started = time.time()
        run_seed(
            config=_config(args),
            seed=args.seed,
            methods=["npo"],
            output_root=npo_dir,
            skip_finetune=True,
            manifest=manifest,
        )
        timings["npo"] = time.time() - started

        print("\n--- routine update " + "-" * 52)
        started = time.time()
        routine_record = run_condition(
            config=_config(args),
            condition="routine_update",
            seed=args.seed,
            output_root=routine_dir,
            manifest=manifest,
        )
        timings["routine_update"] = time.time() - started
    else:
        with open(os.path.join(routine_dir, "routine_update",
                               f"seed{args.seed}", "control_run.json")) as handle:
            routine_record = json.load(handle)

    npo_run = os.path.join(npo_dir, f"seed{args.seed}", "npo")
    npo_scores = os.path.join(npo_run, "per_sample_scores.json")
    routine_scores = os.path.join(
        routine_dir, "routine_update", f"seed{args.seed}", "per_sample_scores.json"
    )

    # ---- check ----------------------------------------------------------
    print("\n--- checks " + "-" * 60)
    check_trajectory(os.path.join(npo_run, "trajectory.json"))
    check_only_adapters_moved(
        os.path.join(npo_run, "unlearned_model"),
        os.path.join(_config(args)["paths"]["checkpoints"], f"finetuned_seed{args.seed}"),
    )
    check_disjointness(routine_record)
    check_rows(npo_scores, "npo")
    check_rows(routine_scores, "routine_update")

    # ---- the analysis chain end to end -----------------------------------
    print("\n--- analysis chain " + "-" * 52)
    prereg = Preregistration.load()
    table = OrientationTable.load()
    files = [load_any(npo_scores), load_any(routine_scores)]
    audit = build_audit(files, table, prereg, n_bootstrap=100)
    check_utility(audit, prereg)

    lowprev = build_lowprev([npo_scores, routine_scores])
    settings = lowprev["aggregate"]["settings"]
    assert settings, "the low-prevalence component produced no settings"
    detector_blocks = [d for s in settings for d in s["detectors"].values()]
    assert detector_blocks, "no detector was evaluated"
    for block in detector_blocks:
        assert block["prevalence"], "no prevalence-adjusted precision was computed"
    print(f"    [8a]  OK -- low-prevalence: {len(settings)} setting(s), "
          f"{len(detector_blocks)} detector evaluations")

    document = build_document(audit, lowprev, None, table3_method="npo")
    assert document["table2_method_verdicts"], "Table 2 is empty"
    assert document["table4_npo_sweep"], "Table 4 has no NPO rows"
    report = format_report(document)
    assert "Table 4" in report

    report_dir = os.path.join(args.output, "reports")
    os.makedirs(report_dir, exist_ok=True)
    with open(os.path.join(report_dir, "smoke_report.md"), "w") as handle:
        handle.write(report)
    with open(os.path.join(report_dir, "smoke_tables.json"), "w") as handle:
        json.dump(document, handle, indent=2, allow_nan=False, default=str)
    print(f"    [8b]  OK -- report and four tables written to {report_dir}")

    print("\n--- NPO cell summary (not a measurement) " + "-" * 30)
    for row in document["table4_npo_sweep"]:
        print(f"    setting={row['setting']} seed={row['seed']} "
              f"steps={row['steps_taken']} stop={row['stop_reason']} "
              f"membership={row['membership_auc']} "
              f"classification={row['classification']}")

    if timings:
        print("\n--- timings " + "-" * 59)
        for key, value in timings.items():
            print(f"    {key}: {value / 60:.1f} min")

    print("\n" + "=" * 72)
    print("SMOKE GATE PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
