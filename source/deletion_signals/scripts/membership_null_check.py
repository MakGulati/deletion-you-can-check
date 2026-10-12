#!/usr/bin/env python3
"""Where does the membership metric's null actually sit?

The finding this exists to diagnose
-----------------------------------
Exact retraining is a **perfect deletion**: the forget set was never trained on.
A correctly calibrated membership axis must therefore read chance on it. It
does on ArxivMIA (0.474) and does not on BookMIA (0.257).

The hypothesis is that BookMIA's "unseen" population is the wrong counterfactual.
D_forget and D_retain are both ``label=1``; D_test is ``label=0`` -- a different
set of books. "Unseen" then differs from "forgotten" in two ways at once, never
trained **and** differently distributed, so fine-tuning on D_retain lowers
D_forget's loss by domain transfer alone and the axis reads it as membership.

This script tests that directly. It rebuilds the membership axis against an
unseen population of **held-out members**: ``label=1`` records the split never
drew, which are never-trained *and* identically distributed. If the hypothesis
is right, exact retraining moves to chance.

Why it is cheap
---------------
The membership axis needs only losses, not gradients. ``L_base`` on the new
population is computed once; each checkpoint then needs a single forward pass
over it. The forget side is already stored -- ``loss_unlearned`` and
``loss_base`` sit on every saved row -- so nothing is recomputed there.

    $PY deletion_signals/scripts/membership_null_check.py \\
        --config bookmia_qwen.yaml \\
        --inputs 'ds_bookmia/**/per_sample_scores.json' \\
        --output ds_bookmia/membership_null

**This is a post-hoc diagnostic.** It was designed after seeing the BookMIA
controls. It does not move a threshold and does not replace the preregistered
axis; it explains it. Both are reported.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import statistics
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402

from deletion_signals.config import apply_cli_overrides, effective_config, load_config  # noqa: E402
from deletion_signals.data import collate_fn  # noqa: E402
from deletion_signals.models import free, load_base_model, load_checkpoint_auto  # noqa: E402
from deletion_signals.reproducibility import run_provenance, set_seed  # noqa: E402
from deletion_signals.schema import load_any  # noqa: E402
from deletion_signals.stats import roc_auc  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="bookmia_qwen.yaml")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--inputs", nargs="+",
                        default=["ds_bookmia/**/per_sample_scores.json"])
    parser.add_argument("--n-unseen", type=int, default=300,
                        help="Held-out members to use as the unseen population")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default="./ds_bookmia/membership_null")
    return parser.parse_args()


def held_out_member_population(config, seed: int, n: int, skip: int) -> List[Any]:
    """``label=1`` records the split never drew, past the routine-update slice.

    ``skip`` steps over whatever ``routine_update`` already claimed, so the
    unseen population is disjoint from the split, from the forget and retain
    sets, and from the update data -- never trained on by anything in the study.
    """
    from deletion_signals.routine_update import member_tail_samples

    pool = member_tail_samples(config, seed, skip + n)
    return pool[skip: skip + n]


@torch.no_grad()
def population_losses(model, tokenizer, samples, device, max_length: int) -> List[float]:
    losses = []
    for sample in samples:
        batch = collate_fn(
            [{"text": sample.text, "sample_id": sample.sample_id}],
            tokenizer=tokenizer, max_length=max_length,
        )
        batch = {k: v.to(device) for k, v in batch.items()}
        losses.append(float(model(**batch).loss.item()))
    return losses


def forget_loss_ratios(path: str) -> List[float]:
    """The forget side of the axis, taken from the saved rows -- not recomputed."""
    rows = load_any(path).rows
    out = []
    for row in rows:
        if row.get("population") != "forget":
            continue
        value = row.get("loss_ratio")
        if value is not None and math.isfinite(float(value)):
            out.append(float(value))
    return out


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    if args.device:
        config["device"] = args.device
    config = apply_cli_overrides(config, args.overrides)

    device = torch.device(config["device"])
    max_length = int(config["scoring"].get("max_length", config["data"]["max_length"]))
    skip = int(config.get("routine_update", {}).get("n_samples", 200))

    paths: List[str] = []
    for pattern in args.inputs:
        paths.extend(sorted(glob.glob(pattern, recursive=True)))
    paths = sorted(dict.fromkeys(os.path.abspath(p) for p in paths))
    if not paths:
        raise SystemExit(f"No score files matched {args.inputs}")

    # Group by seed: the held-out population is a function of the seed, because
    # the draw it excludes is.
    by_seed: Dict[int, List[str]] = {}
    for path in paths:
        meta = load_any(path).metadata
        by_seed.setdefault(int(meta.get("seed", -1)), []).append(path)

    print("=" * 76)
    print("MEMBERSHIP NULL CHECK -- held-out members as the unseen population")
    print(f"  {len(paths)} scored checkpoints across seeds {sorted(by_seed)}")
    print("  POST-HOC DIAGNOSTIC: explains the preregistered axis, does not replace it")
    print("=" * 76)

    results: List[Dict[str, Any]] = []

    for seed in sorted(by_seed):
        set_seed(seed)
        unseen = held_out_member_population(config, seed, args.n_unseen, skip)
        print(f"\n--- seed {seed}: {len(unseen)} held-out members as unseen ---")

        base_model, tokenizer = load_base_model(config)
        base_model.eval()
        base_losses = population_losses(base_model, tokenizer, unseen, device, max_length)

        for path in tqdm(by_seed[seed], desc=f"seed {seed} checkpoints"):
            score_file = load_any(path)
            meta = score_file.metadata
            checkpoint = meta.get("unlearned_checkpoint")
            if not checkpoint or not os.path.isdir(checkpoint):
                results.append({"path": path, "seed": seed, "error": "checkpoint missing",
                                "checkpoint": checkpoint})
                continue

            model, _ = load_checkpoint_auto(
                config, checkpoint, is_trainable=False, base_model=base_model
            )
            model.eval()
            post = population_losses(model, tokenizer, unseen, device, max_length)
            # Unload the adapter, keeping the shared 4-bit backbone resident.
            try:
                model.unload()
            except Exception:  # pragma: no cover - PEFT version differences
                pass

            unseen_ratios = [
                p / max(b, float(config["scoring"]["eps"]))
                for p, b in zip(post, base_losses)
            ]
            forget_ratios = forget_loss_ratios(path)
            if not forget_ratios:
                results.append({"path": path, "seed": seed, "error": "no forget rows"})
                continue

            # Same orientation as the preregistered axis: loss_ratio is
            # higher_is_positive on forget_vs_unseen, positive = forget.
            values = forget_ratios + unseen_ratios
            labels = [1] * len(forget_ratios) + [0] * len(unseen_ratios)

            from deletion_signals.audit import setting_label

            results.append(
                {
                    "path": path,
                    "seed": seed,
                    "method": meta.get("method"),
                    "setting": setting_label(meta, str(meta.get("method"))),
                    "auc_vs_heldout_members": roc_auc(values, labels),
                    "n_forget": len(forget_ratios),
                    "n_unseen": len(unseen_ratios),
                    "mean_loss_ratio_forget": statistics.mean(forget_ratios),
                    "mean_loss_ratio_unseen": statistics.mean(unseen_ratios),
                }
            )

        free(base_model)

    # ---- pair each result with the preregistered axis it explains -----------
    preregistered: Dict[str, Dict[int, float]] = {}
    for path in paths:
        score_file = load_any(path)
        from deletion_signals.audit import setting_label
        from deletion_signals.orientations import OrientationTable, build_task_arrays

        table = OrientationTable.load()
        task = table.task("forget_vs_unseen")
        values, labels, _ = build_task_arrays(score_file.rows, "loss_ratio", task)
        if values and len(set(labels)) > 1:
            key = f"{score_file.metadata.get('method')} / {setting_label(score_file.metadata, str(score_file.metadata.get('method')))}"
            preregistered.setdefault(key, {})[int(score_file.metadata.get("seed", -1))] = (
                roc_auc(table.orient(values, "loss_ratio", "forget_vs_unseen"), labels)
            )

    grouped: Dict[str, List[float]] = {}
    for entry in results:
        if "auc_vs_heldout_members" not in entry:
            continue
        grouped.setdefault(f"{entry['method']} / {entry['setting']}", []).append(
            entry["auc_vs_heldout_members"]
        )

    print("\n" + "=" * 76)
    print(f"{'condition':<40}{'label=0 unseen':>17}{'held-out members':>19}")
    print("-" * 76)
    summary = []
    for key in sorted(grouped):
        new = grouped[key]
        old = list(preregistered.get(key, {}).values())
        row = {
            "condition": key,
            "auc_label0_unseen_mean": statistics.mean(old) if old else None,
            "auc_label0_unseen_sd": statistics.stdev(old) if len(old) > 1 else None,
            "auc_heldout_members_mean": statistics.mean(new),
            "auc_heldout_members_sd": statistics.stdev(new) if len(new) > 1 else None,
            "n_seeds": len(new),
        }
        summary.append(row)
        o = f"{row['auc_label0_unseen_mean']:.3f}" if old else "  --  "
        osd = f" ± {row['auc_label0_unseen_sd']:.3f}" if row["auc_label0_unseen_sd"] else "        "
        nsd = f" ± {row['auc_heldout_members_sd']:.3f}" if row["auc_heldout_members_sd"] else ""
        print(f"{key:<40}{o}{osd}{row['auc_heldout_members_mean']:>12.3f}{nsd}")

    gold = next((r for r in summary if r["condition"].startswith("exact_retrain /")), None)
    print("=" * 76)
    if gold:
        moved = gold["auc_heldout_members_mean"]
        print(f"\nGOLD STANDARD (exact retraining -- a perfect deletion):")
        print(f"  against label=0 unseen      : {gold['auc_label0_unseen_mean']:.4f}")
        print(f"  against held-out members    : {moved:.4f}")
        if abs(moved - 0.5) <= 0.10:
            print("\n  -> The axis moves to CHANCE once the unseen population is drawn from")
            print("     the same pool. The domain-transfer account is SUPPORTED: the 0.257")
            print("     was the counterfactual, not the deletion.")
        else:
            print("\n  -> The axis does NOT move to chance. The domain-transfer account is")
            print("     NOT supported; something else is producing the deviation, and the")
            print("     paper must say so rather than repeating the hypothesis.")

    os.makedirs(args.output, exist_ok=True)
    path = os.path.join(args.output, "membership_null_check.json")
    with open(path, "w") as handle:
        json.dump(
            {
                "check": "membership axis against held-out members vs label=0 non-members",
                "status": "POST-HOC DIAGNOSTIC -- designed after seeing the BookMIA controls",
                "does_not": "move any threshold or replace the preregistered axis",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "dataset": config["data"]["name"],
                "model": config["model"]["name"],
                "n_unseen": args.n_unseen,
                "unseen_population": (
                    "label=1 records the split never drew, past the routine-update "
                    "slice: never trained on by anything in the study, and drawn "
                    "from the same pool as forget and retain"
                ),
                "per_checkpoint": results,
                "summary": summary,
                "provenance": run_provenance(effective_config(config)),
            },
            handle,
            indent=2,
            default=str,
        )
    print(f"\nWrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
