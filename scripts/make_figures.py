#!/usr/bin/env python3
"""Build the FLLM 2026 figures from the recorded experiment values.

All plotted numbers are read from ``artifacts/figure_data.json``, which was
recorded by the original analysis pipeline from the saved experiment outputs
in this repository. This script does not load a model, retrain anything, or
invent measurements.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
PURPLE = "#CC79A7"
RED = "#B22222"
GRAY = "#666666"
LIGHT = "#E6E6E6"
SEEDS = (42, 43, 44)

plt.rcParams.update({
    "font.size": 7.4,
    "axes.titlesize": 7.8,
    "axes.labelsize": 7.4,
    "xtick.labelsize": 7.0,
    "ytick.labelsize": 7.0,
    "legend.fontsize": 6.6,
    "axes.linewidth": 0.7,
    "lines.linewidth": 1.1,
    "figure.dpi": 300,
})


def save(fig: plt.Figure, out: Path, name: str) -> None:
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.png", dpi=300)
    plt.close(fig)


def box(ax, x, y, w, h, text, fc, ec="black", size=6.6, weight="normal"):
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008",
                           fc=fc, ec=ec, lw=0.8, mutation_aspect=0.5)
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=size, weight=weight)
    return patch


def arrow(ax, x0, y0, x1, y1, color="black", lw=1.0):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 shrinkA=0, shrinkB=0, mutation_scale=8,
                                 linewidth=lw, color=color, zorder=5))


def figure_pipeline(out: Path, record: dict) -> None:
    ov = record["overview"]
    original = ov["bookmia_qwen_original_auc"]
    matched = ov["bookmia_qwen_matched_auc"]

    fig, ax = plt.subplots(figsize=(7.15, 2.50))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    stages = [(0.012, 0.305, "Stage 1: Validate\nthe reference"),
              (0.340, 0.305, "Stage 2: Calibrate\nnon-deletion change"),
              (0.668, 0.320, "Stage 3: Audit\nthe released checkpoint")]
    for x, w, title in stages:
        ax.add_patch(FancyBboxPatch((x, 0.075), w, 0.865,
                     boxstyle="round,pad=0.006", fc="#FAFAFA", ec=GRAY, lw=0.7))
        ax.text(x + w / 2, 0.885, title, ha="center", va="center",
                fontsize=7.0, weight="bold", linespacing=0.95)

    # Stage 1
    box(ax, 0.027, 0.645, 0.130, 0.17,
        "Fine-tune\n$\\theta_{\\mathbf{ft}}\\leftarrow D_{\\mathbf{f}}\\cup D_{\\mathbf{r}}$", "#DCEAF4")
    box(ax, 0.167, 0.645, 0.130, 0.17,
        "Exact retrain\n$\\theta_{\\mathbf{rt}}\\leftarrow D_{\\mathbf{r}}$", "#D8EEE7")
    box(ax, 0.027, 0.335, 0.270, 0.19,
        "Gate: $\\mathbf{AUC}(D_{\\mathbf{f}}\\;\\mathrm{vs}\\;Q)$ on\n"
        "$\\theta_{\\mathbf{rt}}$ in $\\mathbf{[0.45,\\,0.55]}$?",
        "#FFFFFF", ec=GRAY)
    arrow(ax, 0.092, 0.645, 0.092, 0.525)
    arrow(ax, 0.232, 0.645, 0.232, 0.525)
    box(ax, 0.027, 0.115, 0.270, 0.15,
        f"Reject axis (BookMIA: {original:.3f})", "#F2C6C2", ec=RED, weight="bold")
    arrow(ax, 0.112, 0.335, 0.112, 0.265, RED)
    ax.text(0.125, 0.295, "fail", color=RED, fontsize=6.6, weight="bold")
    arrow(ax, 0.297, 0.43, 0.340, 0.43, GREEN)
    ax.text(0.292, 0.462, "pass", color=GREEN, fontsize=5.9, weight="bold",
            ha="right", va="center")

    # Stage 2
    box(ax, 0.355, 0.645, 0.130, 0.17,
        "Null refit $\\theta_{\\mathbf{nr}}$\nunchanged data\nfresh seed", "#FFFFFF",
        size=6.2)
    box(ax, 0.495, 0.645, 0.130, 0.17,
        "Routine update $\\theta_{\\mathbf{ru}}$\n200 fresh records", "#FFFFFF")
    box(ax, 0.355, 0.335, 0.270, 0.19,
        "Non-deletion scale of loss change\n(randomness / maintenance)", "#F5E1A4")
    arrow(ax, 0.420, 0.645, 0.420, 0.525)
    arrow(ax, 0.560, 0.645, 0.560, 0.525)
    arrow(ax, 0.625, 0.43, 0.668, 0.43, GREEN)

    # Stage 3
    box(ax, 0.683, 0.645, 0.115, 0.17,
        "Released\n$\\theta_{\\mathbf{un}}$", "#DCEAF4")
    box(ax, 0.828, 0.645, 0.143, 0.17,
        "Membership $\\wedge$\ntargeted shift $\\wedge$\nutility (per seed)",
        "#FFFFFF", size=6.2)
    arrow(ax, 0.798, 0.73, 0.828, 0.73)
    box(ax, 0.683, 0.335, 0.293, 0.19,
        "Conjunctive pass only if all three\nhold on the same checkpoint", "#F5E1A4")
    arrow(ax, 0.899, 0.645, 0.899, 0.525)
    box(ax, 0.683, 0.115, 0.293, 0.15,
        f"Scoped verdict (not a certificate)\nBookMIA repair: {original:.3f} $\\to$ {matched:.3f}",
        "#D8EEE7", weight="bold")
    arrow(ax, 0.830, 0.335, 0.830, 0.265)

    fig.tight_layout(pad=0.15)
    save(fig, out, "fig1_pipeline")
    record["figure1_pipeline"] = {"role": "conceptual summary of the audit protocol"}


def figure_reference(out: Path, record: dict) -> None:
    f1 = record["figure1"]
    cells = ["ArxivMIA / Llama", "ArxivMIA / Qwen", "BookMIA / Llama", "BookMIA / Qwen"]
    values = {c: f1["exact_retraining_auc"][c] for c in cells}
    matched = f1["bookmia_qwen_matched"]
    matched_seeds = [matched[str(s)] for s in SEEDS]
    original_seeds = values["BookMIA / Qwen"]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.15, 2.12),
                                  gridspec_kw={"width_ratios": [1.5, 1]})
    ax.axhspan(0.45, 0.55, color="#D8EEE7", alpha=0.55, zorder=0)
    ax.axhline(0.5, color=GREEN, lw=0.7, ls=":", zorder=1)
    positions = [0, 0.34, 1.0, 1.34]
    colors = [BLUE, BLUE, ORANGE, ORANGE]
    markers = ["o", "s", "o", "s"]
    rng = np.random.default_rng(8)
    for pos, cell, color, marker in zip(positions, cells, colors, markers):
        ys = np.asarray(values[cell])
        xs = pos + rng.uniform(-0.035, 0.035, len(ys))
        ax.scatter(xs, ys, s=24, marker=marker, facecolor="white",
                   edgecolor=color, linewidth=1.1, zorder=3)
        mean = ys.mean()
        ax.plot([pos - 0.09, pos + 0.09], [mean, mean], color=color, lw=2.2, zorder=4)
    ax.set_xticks(positions, ["Arxiv/\nLlama", "Arxiv/\nQwen", "Book/\nLlama", "Book/\nQwen"])
    ax.set_ylabel("Exact-retraining membership AUC")
    ax.set_ylim(0.05, 0.60)
    ax.set_xlim(-0.22, 1.6)
    ax.set_title("(a) The reference must read chance", weight="bold")
    ax.grid(axis="y", color=LIGHT, lw=0.6)
    ax.text(1.6, 0.5, "admission band", ha="right", va="bottom",
            fontsize=6.2, color=GREEN)

    xs = np.arange(3)
    ax2.axhspan(0.45, 0.55, color="#D8EEE7", alpha=0.55, zorder=0)
    ax2.axhline(0.5, color=GREEN, lw=0.7, ls=":", zorder=1)
    for x, y0, y1 in zip(xs, original_seeds, matched_seeds):
        ax2.plot([x, x], [y0, y1], color=GRAY, lw=0.9, ls="--", zorder=2)
    ax2.scatter(xs, original_seeds, s=26, marker="o", facecolor="white",
                edgecolor=ORANGE, linewidth=1.2, zorder=3, label="label-0 comparison")
    ax2.scatter(xs, matched_seeds, s=26, marker="s", facecolor="white",
                edgecolor=GREEN, linewidth=1.2, zorder=3, label="same-pool comparison")
    ax2.set_xticks(xs, [f"seed {s}" for s in SEEDS])
    ax2.set_ylim(0.15, 0.62)
    ax2.set_title("(b) Same-pool repair (BookMIA/Qwen)", weight="bold")
    ax2.legend(frameon=False, loc="upper center",
               bbox_to_anchor=(0.5, -0.24), ncol=2)
    ax2.grid(axis="y", color=LIGHT, lw=0.6)
    fig.tight_layout()
    save(fig, out, "fig2_reference")
    record["figure2_reference"] = {
        "cells": {c: values[c] for c in cells},
        "matched_seeds": matched_seeds,
        "original_seeds": original_seeds,
    }


def figure_conjunctive(out: Path, record: dict) -> None:
    rows = record["figure2"]
    criteria = ["membership_pass", "rnff_pass", "utility_pass", "conjunction_pass"]
    headers = ["Membership", "Targeted shift", "Utility", "Conjunctive"]
    data = np.array([[r[key] for key in criteria] for r in rows], dtype=float)
    # The rejected BookMIA reference makes membership and the joint verdict
    # inapplicable; retain the raw counts in figure_data.json for diagnostics.
    for i, row in enumerate(rows):
        if row["dataset"] == "BookMIA":
            data[i, [0, 3]] = np.nan

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.15, 2.62),
                                  gridspec_kw={"width_ratios": [1.15, 1]})
    cmap = mpl.colors.ListedColormap(["#F4F4F4", "#F2C6C2", "#F5E1A4", "#B8DCCE"])
    cmap.set_bad("#DDDDDD")
    ax.imshow(data, cmap=cmap, vmin=0, vmax=3, aspect="auto")
    for i, r in enumerate(rows):
        for j, key in enumerate(criteria):
            text = f"{int(r[key])}/3"
            if r["dataset"] == "BookMIA" and key in ("membership_pass", "conjunction_pass"):
                text = "invalid*"
            ax.text(j, i, text, ha="center", va="center", fontsize=6.8,
                    weight="bold" if j == 3 else "normal")
    ax.set_yticks(range(len(rows)), [f'{r["dataset"].split("MIA")[0]}MIA: {r["label"]}' for r in rows])
    ax.set_xticks(range(4), headers)
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False, length=0)
    for edge in np.arange(-0.5, len(rows), 1):
        ax.axhline(edge, color="white", lw=1.2)
    for edge in np.arange(-0.5, 4, 1):
        ax.axvline(edge, color="white", lw=1.2)
    ax.set_title("(a) Seed-pass counts per criterion", weight="bold")
    ax.text(0, len(rows) + 0.14,
            "*axis rejected by the reference gate", fontsize=6.0, ha="left")

    detectors = ["loss_delta_vs_finetuned", "raw_cosine", "grad_ratio", "loss_unlearned"]
    labels = ["two-checkpoint loss change", "directional alignment",
              "calibrated gradient magnitude", "released-model loss"]
    colors = [BLUE, ORANGE, GREEN, PURPLE]
    markers = ["o", "s", "^", "D"]
    prevs = ["1:1", "1:10", "1:100"]
    x = np.arange(len(prevs))
    bookmia_dash = (0, (3.0, 2.0))
    for study, ls in (("ArxivMIA / Llama", "-"),
                      ("BookMIA / Qwen", bookmia_dash)):
        for detector, label, color, marker in zip(detectors, labels, colors, markers):
            block = record["figure3"][study][detector]
            means = [block["precision_by_prevalence"][p]["mean"] for p in prevs]
            sds = [block["precision_by_prevalence"][p]["sd"] for p in prevs]
            ax2.errorbar(x, means, yerr=sds, color=color, marker=marker, ms=4,
                         lw=1.2, ls=ls, capsize=2)
    ax2.set_xticks(x, prevs)
    ax2.set_xlabel("Deleted-to-non-deleted ratio")
    ax2.set_ylabel("Precision at top-$k$")
    ax2.set_ylim(0, 1.05)
    ax2.set_title("(b) Precision under class imbalance", weight="bold")
    ax2.grid(axis="y", color=LIGHT, lw=0.6)
    compact_labels = ["loss change", "directional alignment",
                      "gradient magnitude", "released-model loss"]
    detector_handles = [
        Line2D([], [], color=color, marker=marker, linestyle="none",
               markersize=4.0, label=label)
        for label, color, marker in zip(compact_labels, colors, markers)
    ]
    setting_handles = [
        Line2D([], [], color="#333333", linewidth=1.45,
               linestyle="-", label="ArxivMIA/Llama"),
        Line2D([], [], color="#333333", linewidth=1.45,
               linestyle=bookmia_dash, label="BookMIA/Qwen"),
    ]
    fig.tight_layout(pad=0.38, w_pad=0.8)
    fig.subplots_adjust(bottom=0.185)
    fig.legend(handles=detector_handles + setting_handles,
               loc="lower center", bbox_to_anchor=(0.5, 0.002), ncol=6,
               fontsize=5.65, frameon=False, borderpad=0,
               columnspacing=0.62, handlelength=1.65, handletextpad=0.3)
    save(fig, out, "fig3_conjunctive")


def figure_operational(out: Path, record: dict) -> None:
    f4 = record["figure4"]
    cells = ["ArxivMIA / Llama", "ArxivMIA / Qwen", "BookMIA / Llama", "BookMIA / Qwen"]
    detectors = ["loss_unlearned", "grad_ratio", "raw_cosine",
                 "loss_delta_vs_finetuned"]
    detector_labels = ["Released-model\nloss", "Calibrated\ngradient\nmagnitude",
                       "Directional\nalignment", "Two-checkpoint\nloss change"]

    means = np.asarray([
        [np.mean(f4["auc_by_detector"][cell][detector])
         for detector in detectors]
        for cell in cells
    ])
    sds = np.asarray([
        [np.std(f4["auc_by_detector"][cell][detector], ddof=1)
         for detector in detectors]
        for cell in cells
    ])
    auc_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "auc", ["#FAFAFA", "#DDEFE8", "#74BFA5"])

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.15, 2.32),
                                  gridspec_kw={"width_ratios": [1.45, 1]})
    ax.imshow(means, cmap=auc_cmap, vmin=0.5, vmax=1.0,
              aspect="auto", interpolation="nearest")
    for i in range(len(cells)):
        for j in range(len(detectors)):
            ax.text(j, i - 0.09, f"{means[i, j]:.4f}",
                    ha="center", va="center", fontsize=6.4,
                    color="#17332A", weight="bold")
            ax.text(j, i + 0.13, f"$\\pm$ {sds[i, j]:.4f}",
                    ha="center", va="center", fontsize=5.5,
                    color="#25473C")
    ax.set_yticks(range(len(cells)),
                  [cell.replace("MIA / ", "/") for cell in cells])
    ax.set_xticks(range(len(detectors)), detector_labels)
    ax.tick_params(length=0, pad=2)
    for label in ax.get_xticklabels()[2:]:
        label.set_fontweight("bold")
    for edge in np.arange(-0.5, len(cells), 1):
        ax.axhline(edge, color="white", lw=1.4)
    for edge in np.arange(-0.5, len(detectors), 1):
        ax.axvline(edge, color="white", lw=1.4)
    ax.set_title("(a) Matched-access detector AUC", weight="bold")

    rng = np.random.default_rng(12)
    ax2.axhspan(0, 0.0022, color=BLUE, alpha=0.045, zorder=0)
    ax2.axhspan(-0.0032, 0, color=ORANGE, alpha=0.045, zorder=0)
    for i, cell in enumerate(cells):
        ys = np.asarray(f4["paired_difference"][cell])
        ax2.scatter(i + rng.uniform(-0.05, 0.05, len(ys)), ys, s=24,
                    facecolor="white", edgecolor=GRAY, linewidth=1.1, zorder=3)
        ax2.plot([i - 0.13, i + 0.13], [ys.mean(), ys.mean()], color=BLUE, lw=2.2)
    ax2.axhline(0, color="black", lw=0.9)
    ax2.set_xticks(range(4), ["Arxiv/\nLlama", "Arxiv/\nQwen", "Book/\nLlama", "Book/\nQwen"])
    ax2.set_ylabel("Paired AUC difference")
    ax2.set_title("(b) Leading-pair difference (zoom)", weight="bold")
    ax2.set_ylim(-0.0032, 0.0022)
    ax2.grid(axis="y", color=LIGHT, lw=0.6)
    annotation_box = {"facecolor": "white", "edgecolor": "none",
                      "alpha": 0.82, "pad": 1.2}
    ax2.text(0.98, 0.96, "loss change higher", transform=ax2.transAxes,
             ha="right", va="top", fontsize=6.2, weight="semibold",
             color="#004B76", bbox=annotation_box)
    ax2.text(0.98, 0.04, "alignment higher", transform=ax2.transAxes,
             ha="right", va="bottom", fontsize=6.2, weight="semibold",
             color="#8F3B00", bbox=annotation_box)
    fig.tight_layout(pad=0.45, w_pad=1.0)
    save(fig, out, "fig4_operational")


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    data = json.loads((root / "artifacts" / "figure_data.json").read_text())
    out = root / "figures"
    out.mkdir(exist_ok=True)
    # Fig. 1 uses bold math glyphs so checkpoint and dataset symbols remain
    # visually distinct from the surrounding explanatory text at IEEE size.
    with mpl.rc_context({"mathtext.default": "bf"}):
        figure_pipeline(out, data)
    figure_reference(out, data)
    figure_conjunctive(out, data)
    figure_operational(out, data)
    (root / "artifacts" / "figure_data.json").write_text(
        json.dumps(data, indent=2) + "\n")
    print("figures written to", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
