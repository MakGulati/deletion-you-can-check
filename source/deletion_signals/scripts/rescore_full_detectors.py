#!/usr/bin/env python3
"""Re-score existing unlearned checkpoints with FUMA and Fisher-SIF switched on.

Why this exists
---------------
The confirmatory GA+GD runs were scored under ``arxivmia_confirm_transition.yaml``,
which disables FUMA and the Fisher SIF. That was a deliberate scoping decision
for a dense strength sweep, and it left those runs carrying 10 of the 13 score
columns -- no ``sif``, no ``fuma_lora_ratio``, no ``combined_log_score`` -- and
scoring 800 retain samples rather than 600, because with the Fisher estimator
off there is no 200-sample curvature holdout.

The FLLM audit's central detector table needs all 13 columns on the same
populations for every row, or the comparison straddles two different negative
counts and three missing detectors.

**This re-scores; it does not retrain.** The unlearned checkpoints already
exist, so the cost is the scoring pass alone. Output goes to a NEW directory;
the original score files are never touched, and the runs they describe are not
re-run, so nothing about the existing results changes.

    $PY deletion_signals/scripts/rescore_full_detectors.py \\
        --targets ds_confirm_refine/ascent_2.6e-5 ds_confirm/transition/ascent_5.0e-5 \\
        --seeds 42 43 44 --output fllm_audit/rescore

The ``unlearning_run`` block is read from each checkpoint's own
``unlearning_run.json`` and passed through, so the rescored rows keep the
setting label the audit groups them by. Without it every rescored GA+GD run
would collapse into one row called "ga_gd".
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from deletion_signals.collect import collect_scores  # noqa: E402
from deletion_signals.config import (  # noqa: E402
    apply_cli_overrides,
    effective_config,
    load_config,
)
from deletion_signals.reproducibility import run_provenance  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--targets", nargs="+", required=True,
                        help="Run roots, each containing seed<N>/<method>/unlearned_model")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--method", default="ga_gd",
                        help="Method subdirectory inside each seed directory")
    parser.add_argument("--config", default="arxivmia_confirm.yaml",
                        help="Config with FUMA and Fisher-SIF ENABLED")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--checkpoints", default="./ds_checkpoints")
    parser.add_argument("--manifest", default="results/seed{seed}/splits.json")
    parser.add_argument("--output", default="./fllm_audit/rescore")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def _read_unlearning_run(method_dir: str) -> Optional[Dict[str, Any]]:
    """The original run record, so the setting label survives the re-score."""
    path = os.path.join(method_dir, "unlearning_run.json")
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        return json.load(handle)


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    if args.device:
        config["device"] = args.device
    config = apply_cli_overrides(config, args.overrides)

    scores = config["scores"]
    if not (scores["fuma"]["enabled"] and scores["sif"]["enabled"]):
        raise SystemExit(
            f"{args.config} does not enable both FUMA and SIF, which is the only "
            f"reason to re-score. Use arxivmia_confirm.yaml or enable them with "
            f"--set scores.fuma.enabled=true --set scores.sif.enabled=true"
        )

    os.makedirs(args.output, exist_ok=True)
    records: List[Dict[str, Any]] = []

    for target in args.targets:
        label = os.path.basename(os.path.normpath(target))
        for seed in args.seeds:
            method_dir = os.path.join(target, f"seed{seed}", args.method)
            checkpoint = os.path.join(method_dir, "unlearned_model")
            if not os.path.isdir(checkpoint):
                print(f"[skip] no checkpoint at {checkpoint}")
                records.append({"target": target, "seed": seed, "status": "missing",
                                "checkpoint": checkpoint})
                continue

            destination = os.path.join(args.output, label, f"seed{seed}", args.method)
            os.makedirs(destination, exist_ok=True)
            output_path = os.path.join(destination, "per_sample_scores.json")

            print(f"\n[{label} seed {seed}] re-scoring {checkpoint}")
            started = time.time()
            collect_scores(
                config=config,
                seed=seed,
                method=args.method,
                unlearned_checkpoint=checkpoint,
                finetuned_checkpoint=os.path.join(
                    args.checkpoints, f"finetuned_seed{seed}"
                ),
                output_path=output_path,
                unlearning_run=_read_unlearning_run(method_dir),
                manifest_path=args.manifest.replace("{seed}", str(seed)),
            )
            elapsed = time.time() - started
            print(f"[{label} seed {seed}] done in {elapsed / 60:.1f} min")
            records.append(
                {
                    "target": target,
                    "label": label,
                    "seed": seed,
                    "status": "rescored",
                    "source_checkpoint": os.path.abspath(checkpoint),
                    "output": os.path.abspath(output_path),
                    "seconds": elapsed,
                }
            )

    summary_path = os.path.join(args.output, "rescore_summary.json")
    with open(summary_path, "w") as handle:
        json.dump(
            {
                "purpose": (
                    "scoring only, from existing unlearned checkpoints, with FUMA "
                    "and Fisher-SIF enabled so every row of the central detector "
                    "table carries all 13 score columns at a uniform 600 scored "
                    "retain samples. No model was retrained and no existing score "
                    "file was modified."
                ),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "config": args.config,
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
