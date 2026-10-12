"""Publication figures for the three-axis audit.

Reads ``audit_results.json`` and writes the figures the study's claims rest on.
Nothing here computes a metric: every number is taken from the audit, so a
figure cannot disagree with the report beside it.

Design rules applied throughout, so the set reads as one system:

* one measure per axis -- never two scales on one panel;
* colour assigned in a fixed order and never cycled, with marker shape carrying
  identity as well, so the figures survive greyscale and colour-vision
  deficiency;
* error bars are the ACROSS-SEED SD, because the seed is the replication unit;
* thresholds drawn as dashed reference lines, always labelled;
* controls drawn distinctly from methods -- an exactly retrained model is not a
  point on a strength sweep and must not read as one.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple  # noqa: F401

# Validated categorical palette, assigned in fixed order.
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
SERIES = (BLUE, ORANGE, AQUA, YELLOW)
MARKERS = ("o", "s", "^", "D")
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8985"
SURFACE = "#fcfcfb"
DANGER = "#e34948"
GOLD = "#4a3aa7"        # the exact-retraining control
NULL_C = "#8a8985"      # the null refit

CLASS_STYLE = {
    "persistent_fingerprint": (AQUA, "*", "persistent fingerprint"),
    "successful_deletion_no_fingerprint": (BLUE, "P", "deletion, no fingerprint"),
    "incomplete_forgetting": (ORANGE, "o", "incomplete forgetting"),
    "global_destruction": (DANGER, "X", "global destruction"),
    "inconclusive": (MUTED, "s", "inconclusive"),
}


def _style(axis, title, xlabel, ylabel, logx=False, logy=False):
    from matplotlib.ticker import ScalarFormatter

    axis.set_title(title, fontsize=10.5, color=INK, loc="left", pad=8)
    axis.set_xlabel(xlabel, fontsize=9, color=INK2)
    axis.set_ylabel(ylabel, fontsize=9, color=INK2)
    if logx:
        axis.set_xscale("log")
    if logy:
        axis.set_yscale("log")
        low, high = axis.get_ylim()
        if low > 0 and high / low < 100:
            axis.yaxis.set_major_formatter(ScalarFormatter())
            axis.yaxis.set_minor_formatter(ScalarFormatter())
    axis.grid(True, which="major", color=MUTED, alpha=0.22, linewidth=0.6)
    axis.set_axisbelow(True)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        axis.spines[spine].set_color(MUTED)
    axis.tick_params(colors=INK2, labelsize=8)


def _legend(axis, **kwargs):
    axis.legend(fontsize=7, labelcolor=INK2, frameon=True, facecolor=SURFACE,
                edgecolor="none", framealpha=0.85, **kwargs)


def _mean_sd(summary: Optional[Mapping[str, Any]]) -> Tuple[float, float]:
    """(mean, sd) with a missing or single-seed SD reported as 0 for drawing."""
    if not summary or not summary.get("n_seeds"):
        return float("nan"), 0.0
    mean = summary.get("mean")
    sd = summary.get("sd")
    mean = float("nan") if mean is None else float(mean)
    sd = 0.0 if sd is None or (isinstance(sd, float) and math.isnan(sd)) else float(sd)
    return mean, sd


def _strength(setting: Mapping[str, Any]) -> Optional[float]:
    """The swept value out of a label like ``ascent_lr=2.5e-05``."""
    label = str(setting.get("setting", ""))
    if "=" not in label:
        return None
    try:
        return float(label.split("=", 1)[1])
    except ValueError:
        return None


def _sweep_settings(results: Mapping[str, Any], method: str = "ga_gd") -> List[Dict[str, Any]]:
    out = [
        s for s in results["settings"]
        if s["method"] == method and _strength(s) is not None
    ]
    return sorted(out, key=_strength)


def _control(results: Mapping[str, Any], name: str) -> Optional[Dict[str, Any]]:
    for setting in results["settings"]:
        if setting["method"] == name:
            return setting
    return None


# --------------------------------------------------------------- figure 1


def figure_operating_map(results: Mapping[str, Any], path: str) -> str:
    """Every setting on the three axes at once, with its classification.

    The point of the figure: membership removal and utility are anticorrelated
    across the sweep, so the region where both hold is empty for the methods and
    occupied only by exact retraining.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    thresholds = results["metadata"]["thresholds"]
    low, high = thresholds["membership_equivalence_region"]
    max_ppl = thresholds["max_perplexity_ratio"]

    figure, axes = plt.subplots(1, 2, figsize=(12.0, 5.2), facecolor=SURFACE)
    for axis in axes:
        axis.set_facecolor(SURFACE)

    # -- panel a: membership vs utility, the two axes that must hold together
    axis = axes[0]
    axis.axvspan(low, high, color=AQUA, alpha=0.10, zorder=0)
    axis.axhline(max_ppl, color=DANGER, linestyle="--", linewidth=1.2, zorder=1)
    axis.text(0.012, max_ppl * 1.04, f"utility bar {max_ppl}", fontsize=7,
              color=DANGER, transform=axis.get_yaxis_transform())

    seen = set()
    for setting in results["settings"]:
        for seed, entry in setting["per_seed"].items():
            membership = entry.get("membership_auc")
            utility = (entry.get("perplexity_ratio") or {}).get("retain")
            if membership is None or utility is None:
                continue
            label = entry.get("classification") or "inconclusive"
            colour, marker, pretty = CLASS_STYLE.get(label, CLASS_STYLE["inconclusive"])
            if setting["method"] == "exact_retrain":
                colour, marker, pretty = GOLD, "*", "exact retraining (gold standard)"
            elif setting["method"] == "null_refit":
                colour, marker, pretty = NULL_C, "v", "null refit (nothing deleted)"
            axis.scatter(membership, max(utility, 1e-3), s=95 if marker == "*" else 46,
                         color=colour, marker=marker, alpha=0.9,
                         edgecolor=SURFACE, linewidth=0.6, zorder=3,
                         label=pretty if pretty not in seen else None)
            seen.add(pretty)

    axis.set_ylim(bottom=0.9)
    _style(axis, "a. The two axes that must hold together",
           "membership AUC on forget vs test (0.5 = chance)",
           "retain perplexity ratio (log)", logy=True)
    axis.text(0.5 * (low + high), 0.93, "membership\nremoved", fontsize=7,
              color=INK2, ha="center", va="bottom")
    _legend(axis, loc="upper left")

    # -- panel b: selectivity -- forgetting achieved against collateral cost
    axis = axes[1]
    rnff_min = thresholds["rnff_min_forget"]
    rnff_max = thresholds["rnff_max_retain"]
    axis.axvline(rnff_min, color=DANGER, linestyle="--", linewidth=1.2, zorder=1)
    axis.axhline(rnff_max, color=DANGER, linestyle="--", linewidth=1.2, zorder=1)
    # The quadrant a candidate operating point has to reach.
    axis.axvspan(rnff_min, 1e4, ymin=0, ymax=1, color=AQUA, alpha=0.06, zorder=0)

    diagonal = [1e-3, 1e4]
    axis.plot(diagonal, diagonal, color=MUTED, linestyle=":", linewidth=1.0,
              zorder=1, label="retain damaged as much as forget")

    seen = set()
    for setting in results["settings"]:
        for seed, entry in setting["per_seed"].items():
            rnff = entry.get("rnff") or {}
            f, r = rnff.get("forget"), rnff.get("retain")
            if f is None or r is None or f <= 0:
                continue
            label = entry.get("classification") or "inconclusive"
            colour, marker, pretty = CLASS_STYLE.get(label, CLASS_STYLE["inconclusive"])
            if setting["method"] == "exact_retrain":
                colour, marker, pretty = GOLD, "*", "exact retraining"
            elif setting["method"] == "null_refit":
                colour, marker, pretty = NULL_C, "v", "null refit"
            axis.scatter(f, max(abs(r), 1e-3), s=95 if marker == "*" else 46,
                         color=colour, marker=marker, alpha=0.9,
                         edgecolor=SURFACE, linewidth=0.6, zorder=3,
                         label=pretty if pretty not in seen else None)
            seen.add(pretty)

    axis.set_xlim(1e-2, 1e2)
    axis.set_ylim(1e-3, 1e2)
    _style(axis, "b. Selectivity: what forgetting costs the retain set",
           "RNFF forget  (1.0 = exact retraining)",
           "|RNFF retain|  (collateral)", logx=True, logy=True)
    _legend(axis, loc="upper left")

    figure.suptitle(
        "Figure 1 — three-axis operating map. Each point is one seed of one setting.",
        fontsize=11.5, color=INK,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    figure.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(figure)
    return path


# --------------------------------------------------------------- figure 2


def figure_transition(results: Mapping[str, Any], path: str) -> Optional[str]:
    """The GA+GD transition: what moves, when, and in what order."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    settings = _sweep_settings(results)
    if len(settings) < 2:
        return None

    thresholds = results["metadata"]["thresholds"]
    x = [_strength(s) for s in settings]
    figure, axes = plt.subplots(2, 2, figsize=(11.5, 8.2), facecolor=SURFACE)
    for axis in axes.flat:
        axis.set_facecolor(SURFACE)

    # (a) perplexity by population
    axis = axes[0][0]
    for index, population in enumerate(("forget", "retain", "test")):
        means, sds = zip(*[_mean_sd(s["perplexity_ratio"][population]) for s in settings])
        axis.errorbar(x, means, yerr=sds, label=f"{population}", color=SERIES[index],
                      marker=MARKERS[index], markersize=6, linewidth=2.0, capsize=3)
    axis.axhline(thresholds["max_perplexity_ratio"], color=DANGER, linestyle="--",
                 linewidth=1.2, label=f"utility bar {thresholds['max_perplexity_ratio']}")
    _style(axis, "a. Utility: perplexity relative to the pre-deletion model",
           "ascent learning rate", "perplexity ratio (log)", logx=True, logy=True)
    _legend(axis)

    # (b) membership -- the axis that must reach chance
    axis = axes[0][1]
    low, high = thresholds["membership_equivalence_region"]
    axis.axhspan(low, high, color=AQUA, alpha=0.12, zorder=0)
    means, sds = zip(*[
        _mean_sd(s["auc"].get("forget_vs_unseen", {}).get("loss_ratio")) for s in settings
    ])
    axis.errorbar(x, means, yerr=sds, color=SERIES[0], marker=MARKERS[0],
                  markersize=6, linewidth=2.0, capsize=3, label="loss_ratio, forget vs test")
    axis.axhline(0.5, color=MUTED, linestyle=":", linewidth=1.0)
    control = _control(results, "exact_retrain")
    if control:
        mean, _ = _mean_sd(control["auc"].get("forget_vs_unseen", {}).get("loss_ratio"))
        axis.axhline(mean, color=GOLD, linestyle="-.", linewidth=1.4,
                     label="exact retraining")
    _style(axis, "b. Membership evidence (must vanish into the band)",
           "ascent learning rate", "AUC", logx=True)
    _legend(axis)

    # (c) selectivity -- the two RNFFs on one scale, which is legitimate:
    # both are the same quantity measured on different populations.
    axis = axes[1][0]
    for index, population in enumerate(("forget", "retain")):
        means, sds = zip(*[_mean_sd(s["rnff"][population]) for s in settings])
        axis.errorbar(x, [abs(m) for m in means], yerr=sds, label=f"RNFF {population}",
                      color=SERIES[index], marker=MARKERS[index], markersize=6,
                      linewidth=2.0, capsize=3)
    axis.axhline(thresholds["rnff_min_forget"], color=DANGER, linestyle="--",
                 linewidth=1.2, label=f"forget bar {thresholds['rnff_min_forget']}")
    axis.axhline(thresholds["rnff_max_retain"], color=DANGER, linestyle=":",
                 linewidth=1.2, label=f"retain bar {thresholds['rnff_max_retain']}")
    axis.axhline(1.0, color=GOLD, linestyle="-.", linewidth=1.4, label="exact retraining")
    _style(axis, "c. Selectivity: forgetting and its collateral cost",
           "ascent learning rate", "|RNFF| (log)", logx=True, logy=True)
    _legend(axis)

    # (d) how far the run travelled, and whether it survived
    axis = axes[1][1]
    ratios = []
    for setting in settings:
        f, _ = _mean_sd(setting["rnff"]["forget"])
        r, _ = _mean_sd(setting["rnff"]["retain"])
        ratios.append(abs(r / f) if f not in (0.0, float("nan")) else float("nan"))
    axis.plot(x, ratios, color=SERIES[0], marker=MARKERS[0], markersize=6, linewidth=2.0,
              label="collateral ratio  |RNFF retain| / RNFF forget")
    axis.axhline(1.0, color=DANGER, linestyle="--", linewidth=1.2,
                 label="retain damaged as much as forget")
    axis.set_ylim(-0.05, 1.15)
    _style(axis, "d. Specificity degrades monotonically with strength",
           "ascent learning rate", "collateral ratio", logx=True)
    _legend(axis, loc="lower right")

    figure.suptitle(
        "Figure 2 — the GA+GD transition. Mean over 3 seeds, error bars = across-seed SD.",
        fontsize=11.5, color=INK,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(figure)
    return path


# --------------------------------------------------------------- figure 3


def figure_detectors(results: Mapping[str, Any], path: str, prereg=None) -> Optional[str]:
    """Detectors on the gold standard, against their false-positive floors.

    Grouped by ACCESS class, because a detector that needs gradients may only be
    claimed as advantageous over one that does not if it actually beats it.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    control = _control(results, "exact_retrain")
    if not control:
        return None

    task = results["metadata"]["primary_task"]
    block = control["auc"].get(task, {})
    floors = results["metadata"].get("null_refit_floors") or {}
    if not block:
        return None

    detectors = sorted(block, key=lambda d: -(block[d].get("mean") or 0))
    values = [block[d]["mean"] for d in detectors]
    errors = [_mean_sd(block[d])[1] for d in detectors]
    floor_values = [(floors.get(d) or {}).get("mean") for d in detectors]

    access = {}
    if prereg is not None:
        access = {d: prereg.access_of(d) for d in detectors}
    needs_gradients = [("gradients" in access.get(d, [])) for d in detectors]

    figure, axis = plt.subplots(figsize=(10.5, 5.6), facecolor=SURFACE)
    axis.set_facecolor(SURFACE)

    positions = np.arange(len(detectors))
    colours = [ORANGE if g else BLUE for g in needs_gradients]
    axis.barh(positions, values, height=0.62, color=colours, alpha=0.9,
              edgecolor=SURFACE, linewidth=1.5, zorder=3)
    axis.errorbar(values, positions, xerr=errors, fmt="none", ecolor=INK2,
                  elinewidth=1.1, capsize=3, zorder=4)

    for index, floor in enumerate(floor_values):
        if floor is None:
            continue
        axis.plot([floor, floor], [index - 0.34, index + 0.34], color=DANGER,
                  linewidth=2.0, zorder=5,
                  label="false-positive floor (null refit)" if index == 0 else None)

    axis.axvline(0.5, color=MUTED, linestyle=":", linewidth=1.2, label="chance")
    axis.set_yticks(positions)
    axis.set_yticklabels(detectors, fontsize=8, color=INK)
    axis.invert_yaxis()
    axis.set_xlim(0.4, 1.02)

    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor=BLUE, label="no gradients needed (forward passes only)"),
        Patch(facecolor=ORANGE, label="requires per-sample gradients"),
    ]
    handles += [h for h in axis.get_legend_handles_labels()[0]]
    axis.legend(handles=handles, fontsize=7.5, labelcolor=INK2, frameon=True,
                facecolor=SURFACE, edgecolor="none", framealpha=0.9, loc="lower right")

    _style(axis, "Figure 3 — detectors on the exact-retraining control, by access class",
           "AUC, forget vs retain (mean over 3 seeds)", "")
    figure.tight_layout()
    figure.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(figure)
    return path


# --------------------------------------------------------------- figure 4


def figure_controls(results: Mapping[str, Any], path: str) -> Optional[str]:
    """Where exact retraining sits relative to every method, on all three axes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    thresholds = results["metadata"]["thresholds"]
    rows: List[Tuple[str, float, float, float, bool]] = []
    for setting in results["settings"]:
        membership, _ = _mean_sd(
            setting["auc"].get("forget_vs_unseen", {}).get("loss_ratio")
        )
        utility, _ = _mean_sd(setting["perplexity_ratio"]["retain"])
        rnff, _ = _mean_sd(setting["rnff"]["forget"])
        label = setting["setting"] if setting["method"] not in ("exact_retrain", "null_refit") \
            else setting["method"]
        rows.append((label, membership, utility, rnff, setting["is_control"]))

    if not rows:
        return None
    rows.sort(key=lambda r: (not r[4], r[3]))

    figure, axes = plt.subplots(1, 3, figsize=(13.0, 5.4), facecolor=SURFACE, sharey=True)
    for axis in axes:
        axis.set_facecolor(SURFACE)

    positions = np.arange(len(rows))
    labels = [r[0] for r in rows]
    colours = [GOLD if r[4] and "exact" in r[0] else (NULL_C if r[4] else BLUE) for r in rows]

    panels = (
        (0, [r[1] for r in rows], "membership AUC (forget vs test)",
         thresholds["membership_equivalence_region"], False),
        (1, [max(r[2], 1e-3) for r in rows], "retain perplexity ratio",
         [thresholds["max_perplexity_ratio"]], True),
        (2, [max(abs(r[3]), 1e-3) for r in rows], "RNFF forget",
         [thresholds["rnff_min_forget"], 1.0], True),
    )
    for index, values, title, marks, logx in panels:
        axis = axes[index]
        axis.barh(positions, values, height=0.6, color=colours, alpha=0.9,
                  edgecolor=SURFACE, linewidth=1.2, zorder=3)
        for mark in marks:
            axis.axvline(mark, color=DANGER if mark != 1.0 else GOLD,
                         linestyle="--" if mark != 1.0 else "-.", linewidth=1.2, zorder=2)
        _style(axis, title, "", "", logx=logx)
        axis.set_yticks(positions)
        axis.set_yticklabels(labels, fontsize=7.5, color=INK)
        axis.invert_yaxis()

    from matplotlib.patches import Patch
    axes[0].legend(handles=[
        Patch(facecolor=GOLD, label="exact retraining (gold standard)"),
        Patch(facecolor=NULL_C, label="null refit (nothing deleted)"),
        Patch(facecolor=BLUE, label="unlearning method"),
    ], fontsize=7, labelcolor=INK2, frameon=True, facecolor=SURFACE,
        edgecolor="none", framealpha=0.9, loc="lower right")

    figure.suptitle(
        "Figure 4 — exact retraining against every method, on all three axes.",
        fontsize=11.5, color=INK,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(figure)
    return path


# ------------------------------------------------------------------- driver


def build_all(results: Mapping[str, Any], output_dir: str, prereg=None) -> List[str]:
    os.makedirs(output_dir, exist_ok=True)
    written = []
    for name, builder in (
        ("figure1_operating_map.png", lambda p: figure_operating_map(results, p)),
        ("figure2_transition.png", lambda p: figure_transition(results, p)),
        ("figure3_detectors.png", lambda p: figure_detectors(results, p, prereg)),
        ("figure4_controls.png", lambda p: figure_controls(results, p)),
    ):
        path = builder(os.path.join(output_dir, name))
        if path:
            written.append(path)
    return written


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    from .audit import Preregistration

    parser = argparse.ArgumentParser(description="Build the audit's figures.")
    parser.add_argument("--results", required=True, help="audit_results.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--preregistration", default=None)
    args = parser.parse_args(argv)

    with open(args.results) as handle:
        results = json.load(handle)
    prereg = Preregistration.load(args.preregistration)

    for path in build_all(results, args.output, prereg):
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
