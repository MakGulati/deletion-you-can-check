#!/usr/bin/env python3
"""Pre-flight: does the PRETRAINED BASE already separate members from unseen?

The membership axis scores ``loss_ratio = L_unlearned(z) / L_base(z)``. Dividing
by ``L_base`` is meant to remove intrinsic sample difficulty, leaving membership
-- and that only works if **the base model never saw the data**.

On BookMIA that assumption is not free. Its ``label=1`` marks snippets from
books that were in pretraining corpora, so a base model may already assign them
lower loss, and the denominator would then carry membership rather than
difficulty. Distribution shift between the label=1 and label=0 books would
produce the same failure by a different route; this check cannot separate the
two causes and does not need to, because either degrades the calibration
identically.

    $PY deletion_signals/scripts/base_calibration_check.py \\
        --config bookmia_qwen.yaml --seeds 42 43 44

Reports ``AUC(member vs unseen)`` under ``-L_base``, so a base that memorised
the members scores above 0.5. The graded pass criterion was fixed in advance in
``ds_bookmia/preregistration/BASE_CALIBRATION_CRITERION.md``; this script
applies it and prints the verdict rather than leaving it to judgement.

No fine-tuning, no adapters, no unlearning: the base model is loaded once and
each population is scored under it. About 20 minutes.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402

from deletion_signals.config import apply_cli_overrides, effective_config, load_config  # noqa: E402
from deletion_signals.data import collate_fn, load_or_build_splits, subsample  # noqa: E402
from deletion_signals.models import free, load_base_model  # noqa: E402
from deletion_signals.reproducibility import run_provenance, set_seed  # noqa: E402
from deletion_signals.stats import roc_auc  # noqa: E402

# Fixed in BASE_CALIBRATION_CRITERION.md before any Qwen number existed.
PASS_THRESHOLD = 0.10
CONDITIONAL_THRESHOLD = 0.20
# ArxivMIA + Llama-3.2-1B, computed from the saved confirmatory scores.
ARXIVMIA_REFERENCE = {"auc": 0.4626, "sd": 0.0221, "deviation": 0.0374}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="bookmia_qwen.yaml")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Cap per population; the default uses the config's own cap")
    parser.add_argument("--output", default="./ds_bookmia/preregistration")
    return parser.parse_args()


@torch.no_grad()
def sample_losses(model, tokenizer, samples, device, max_length: int) -> List[Dict[str, Any]]:
    """Token-mean NLL for each sample, one forward pass at a time.

    One sample per batch, matching how the collector scores: a per-sample loss
    computed inside a padded batch depends on what else was in the batch, and
    this number has to be comparable with the ``loss_base`` column the real
    pipeline writes.
    """
    out: List[Dict[str, Any]] = []
    for sample in tqdm(samples, desc="scoring under the base model", leave=False):
        batch = collate_fn(
            [{"text": sample.text, "sample_id": sample.sample_id}],
            tokenizer=tokenizer, max_length=max_length,
        )
        batch = {k: v.to(device) for k, v in batch.items()}
        loss = float(model(**batch).loss.item())
        n_tokens = int((batch["labels"] != -100).sum().item()) - 1
        out.append(
            {
                "sample_id": sample.sample_id,
                "population": sample.population,
                "loss_base": loss,
                "n_tokens": max(n_tokens, 0),
                "truncated": bool(n_tokens + 1 >= max_length),
            }
        )
    return out


def evaluate_seed(config: Dict[str, Any], seed: int) -> Dict[str, Any]:
    set_seed(seed)
    device = torch.device(config["device"])
    max_length = int(config["scoring"].get("max_length", config["data"]["max_length"]))

    split_info = load_or_build_splits(config, seed, None)
    splits = subsample(
        split_info["splits"], config["scoring"].get("max_samples_per_population")
    )

    model, tokenizer = load_base_model(config)
    model.eval()
    rows: List[Dict[str, Any]] = []
    for population in ("forget", "retain", "test"):
        rows.extend(
            sample_losses(model, tokenizer, splits.get(population, []), device, max_length)
        )
    free(model)

    values, labels = [], []
    for row in rows:
        if row["population"] in ("forget", "retain"):
            labels.append(1)
        elif row["population"] == "test":
            labels.append(0)
        else:
            continue
        # Lower base loss ranks as more member-like.
        values.append(-row["loss_base"])

    auc = roc_auc(values, labels)
    members = [r["loss_base"] for r in rows if r["population"] in ("forget", "retain")]
    unseen = [r["loss_base"] for r in rows if r["population"] == "test"]

    return {
        "seed": seed,
        "auc_member_vs_unseen": auc,
        "deviation_from_chance": abs(auc - 0.5),
        "mean_base_nll_members": statistics.mean(members) if members else None,
        "mean_base_nll_unseen": statistics.mean(unseen) if unseen else None,
        "counts": {p: sum(1 for r in rows if r["population"] == p)
                   for p in ("forget", "retain", "test")},
        "n_truncated": sum(1 for r in rows if r["truncated"]),
        "n_rows": len(rows),
        "mean_tokens": statistics.mean(r["n_tokens"] for r in rows) if rows else None,
    }


def verdict_for(deviation: float) -> Dict[str, str]:
    if deviation <= PASS_THRESHOLD:
        return {
            "verdict": "PASS",
            "consequence": (
                "the base calibration is usable; the membership axis means on "
                "BookMIA what it means on ArxivMIA. Proceed with the full matrix."
            ),
        }
    if deviation <= CONDITIONAL_THRESHOLD:
        return {
            "verdict": "CONDITIONAL",
            "consequence": (
                "proceed, but the membership axis carries a declared calibration "
                "caveat and the base AUC is reported beside every membership "
                "number so a reader can discount it."
            ),
        }
    return {
        "verdict": "FAIL",
        "consequence": (
            "the base calibration is degenerate. Report BookMIA on the other "
            "three axes only and declare the membership axis uninterpretable "
            "for this dataset/model pair -- do not report it with a caveat."
        ),
    }


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    if args.device:
        config["device"] = args.device
    if args.max_samples is not None:
        config["scoring"]["max_samples_per_population"] = args.max_samples
    config = apply_cli_overrides(config, args.overrides)

    print("=" * 74)
    print("BASE-MODEL CALIBRATION CHECK")
    print(f"  dataset : {config['data']['name']}")
    print(f"  model   : {config['model']['name']}  (pretrained base, no adapters)")
    print(f"  criterion fixed in advance: PASS if |AUC-0.5| <= {PASS_THRESHOLD},"
          f" CONDITIONAL <= {CONDITIONAL_THRESHOLD}")
    print(f"  ArxivMIA/Llama reference  : AUC {ARXIVMIA_REFERENCE['auc']:.4f}, "
          f"deviation {ARXIVMIA_REFERENCE['deviation']:.4f}")
    print("=" * 74)

    results = []
    for seed in args.seeds:
        print(f"\n--- seed {seed} ---")
        entry = evaluate_seed(config, seed)
        results.append(entry)
        print(f"  AUC(member vs unseen) = {entry['auc_member_vs_unseen']:.4f}  "
              f"(deviation {entry['deviation_from_chance']:.4f})")
        print(f"  mean base NLL: members {entry['mean_base_nll_members']:.3f}  "
              f"unseen {entry['mean_base_nll_unseen']:.3f}")
        print(f"  truncated {entry['n_truncated']}/{entry['n_rows']} rows, "
              f"mean {entry['mean_tokens']:.0f} scored tokens")

    aucs = [r["auc_member_vs_unseen"] for r in results if math.isfinite(r["auc_member_vs_unseen"])]
    mean_auc = statistics.mean(aucs)
    deviation = abs(mean_auc - 0.5)
    outcome = verdict_for(deviation)

    print("\n" + "=" * 74)
    print(f"  mean AUC over {len(aucs)} seeds : {mean_auc:.4f}"
          f" ± {statistics.stdev(aucs):.4f}" if len(aucs) > 1 else f"  mean AUC: {mean_auc:.4f}")
    print(f"  deviation from chance    : {deviation:.4f}")
    print(f"  ArxivMIA/Llama reference : {ARXIVMIA_REFERENCE['deviation']:.4f}")
    print(f"\n  VERDICT: {outcome['verdict']}")
    print(f"  {outcome['consequence']}")
    print("=" * 74)

    os.makedirs(args.output, exist_ok=True)
    path = os.path.join(args.output, "base_calibration_check.json")
    with open(path, "w") as handle:
        json.dump(
            {
                "check": "does the pretrained base already separate members from unseen?",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "dataset": config["data"]["name"],
                "model": config["model"]["name"],
                "criterion": {
                    "pass_threshold": PASS_THRESHOLD,
                    "conditional_threshold": CONDITIONAL_THRESHOLD,
                    "fixed_in": "ds_bookmia/preregistration/BASE_CALIBRATION_CRITERION.md",
                    "fixed_before_measurement": True,
                },
                "arxivmia_reference": ARXIVMIA_REFERENCE,
                "per_seed": results,
                "mean_auc": mean_auc,
                "deviation_from_chance": deviation,
                **outcome,
                "provenance": run_provenance(effective_config(config)),
            },
            handle,
            indent=2,
            default=str,
        )
    print(f"\nWrote {path}")
    return 0 if outcome["verdict"] != "FAIL" else 2


if __name__ == "__main__":
    sys.exit(main())
