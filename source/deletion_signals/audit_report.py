"""Human-readable rendering of the three-axis audit.

Kept apart from :mod:`audit` so the computation can be tested without going
through string formatting, and so the report's ordering -- which is an editorial
decision about what a reader must see before what -- is visible in one place.

The order is deliberate: what was run, then data quality, then the three axes
together, then the verdict, then the detector comparison. A reader reaches the
detector table only after seeing whether the models it was measured on were
still functioning.
"""

from __future__ import annotations

import math
from typing import Any, List, Mapping, Optional, Sequence

AXIS_LABELS = {
    "membership_removed": "membership removed",
    "utility_preserved": "utility preserved",
    "selective_forgetting": "selective forgetting",
    "deletion_detectable": "deletion detectable",
}


def _fmt(value: Optional[float], digits: int = 3) -> str:
    if value is None:
        return "--"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf"
    if value != 0.0 and (abs(value) >= 1e5 or abs(value) < 1e-4):
        return f"{value:.{max(digits - 1, 1)}e}"
    return f"{value:.{digits}f}"


def _mean_sd(summary: Optional[Mapping[str, Any]], digits: int = 3) -> str:
    if not summary or not summary.get("n_seeds"):
        return "--"
    sd = summary.get("sd")
    if sd is None or (isinstance(sd, float) and math.isnan(sd)):
        return f"{_fmt(summary['mean'], digits)} (n=1)"
    return f"{_fmt(summary['mean'], digits)} ± {_fmt(sd, digits)}"


def _yn(value: Optional[bool]) -> str:
    if value is None:
        return "--"
    return "yes" if value else "no"


def format_report(results: Mapping[str, Any], prereg) -> str:
    metadata = results["metadata"]
    settings = results["settings"]
    thresholds = metadata["thresholds"]

    lines: List[str] = []
    lines += [
        "# Three-axis unlearning audit — ArxivMIA",
        "",
        f"Generated {metadata['timestamp']} from {len(metadata['inputs'])} score file(s).",
        f"Preregistration: `{metadata['preregistration']}` (schema v{metadata['preregistration_schema']}).",
        "",
        "Thresholds below were fixed before any confirmatory run and are read from",
        "that file, not from constants in code.",
        "",
        "| axis | criterion |",
        "|---|---|",
        f"| membership removed | `{prereg.membership['score']}` AUC on "
        f"`{prereg.membership['task']}` within {thresholds['membership_equivalence_region']} |",
        f"| utility preserved | retain and test perplexity ratio ≤ {thresholds['max_perplexity_ratio']} |",
        f"| selective forgetting | RNFF_forget ≥ {thresholds['rnff_min_forget']} and "
        f"\\|RNFF_retain\\| ≤ {thresholds['rnff_max_retain']} |",
        f"| deletion detectable | `{metadata['primary_detector']}` AUC ≥ "
        f"{thresholds['deletion_min_auc']}, CI lower > 0.5, above the null-refit floor |",
        "",
        "A setting is a candidate operating point only if the first three hold on",
        "**every** seed. Membership AUC alone never certifies anything.",
        "",
    ]

    # ---- verdict -----------------------------------------------------------
    lines += ["## Verdict", "", f"**{results['verdict']['outcome']}**", ""]
    candidates = results["verdict"]["candidate_operating_points"]
    if candidates:
        listed = ", ".join(f"{c['method']} @ {c['setting']}" for c in candidates)
        lines += [f"Candidate operating point(s): {listed}.", ""]
    else:
        lines += ["No setting satisfied the three joint criteria on every seed.", ""]

    lines += [
        "| method | setting | membership | utility | selective | detectable | candidate |",
        "|---|---|---|---|---|---|---|",
    ]
    for setting in settings:
        met = setting["axes_met_seed_counts"]
        n = setting["n_seeds"]
        lines.append(
            f"| {setting['method']} | {setting['setting']} | "
            f"{met['membership_removed']}/{n} | {met['utility_preserved']}/{n} | "
            f"{met['selective_forgetting']}/{n} | {met['deletion_detectable']}/{n} | "
            f"{_yn(setting['candidate_operating_point'])} |"
        )
    lines.append("")

    # ---- controls first ----------------------------------------------------
    controls = [s for s in settings if s["is_control"]]
    if controls:
        lines += [
            "## Controls",
            "",
            "`exact_retrain` is the gold standard — the model you would have had if",
            "the forget set had never been in the training data. `null_refit` deletes",
            "**nothing** and differs from the pre-deletion model only in training",
            "randomness, so whatever a detector scores there is a false positive.",
            "",
            "Read them together. For both, ĝ = θ_post − θ_finetuned is a difference",
            "between two *independently trained* models, not the small displacement an",
            "unlearning method produces, so a retraining signal is uninterpretable",
            "without the null floor beside it.",
            "",
            "| control | seeds | retain ppl ratio | test ppl ratio | RNFF forget | membership AUC |",
            "|---|---|---|---|---|---|",
        ]
        for setting in controls:
            lines.append(
                f"| {setting['method']} | {setting['n_seeds']} | "
                f"{_mean_sd(setting['perplexity_ratio']['retain'])} | "
                f"{_mean_sd(setting['perplexity_ratio']['test'])} | "
                f"{_mean_sd(setting['rnff']['forget'])} | "
                f"{_mean_sd(setting['auc'].get(prereg.membership['task'], {}).get(prereg.membership['score']))} |"
            )
        lines.append("")

        validity = metadata.get("control_validity") or {}
        if validity.get("evaluable"):
            lines += [
                "### Is the RNFF denominator measuring deletion?",
                "",
                "RNFF divides by the loss change exact retraining produced on the forget",
                "set, *attributing that change to deletion*. The null refit deleted",
                "nothing, so its RNFF_forget is how much of that change two independently",
                "trained models would have produced anyway.",
                "",
                f"| null-refit RNFF_forget | {_mean_sd(validity['null_refit_rnff_forget'])} |",
                "|---|---|",
                f"| null-refit RNFF_retain | {_mean_sd(validity['null_refit_rnff_retain'])} |",
                "",
                ("> **Denominator contaminated.** " if validity.get("denominator_contaminated")
                 else "> "),
                f"> {validity['interpretation']}",
                "",
            ]

        floors = metadata.get("null_refit_floors") or {}
        if floors:
            lines += [
                "### False-positive floors on `forget_vs_retain` (null refit)",
                "",
                "Nothing was deleted in these runs. A detector scoring above chance here",
                "is reading run-to-run training variation.",
                "",
                "| detector | AUC on the null refit |",
                "|---|---|",
            ]
            for detector, summary in sorted(
                floors.items(), key=lambda kv: -(kv[1].get("mean") or 0)
            ):
                lines.append(f"| `{detector}` | {_mean_sd(summary)} |")
            lines.append("")

    # ---- three axes --------------------------------------------------------
    lines += [
        "## Axis 1 — membership removal (forget vs test)",
        "",
        "Under the preregistered higher-is-forget direction, **0.5 is chance and",
        "0.0 is as much evidence as 1.0** — a forgotten record reading far below 0.5",
        "still looks exactly like a training member. Hence the two-sided region.",
        "",
        "| method | setting | forget_vs_unseen | member_vs_unseen | retain_vs_unseen |",
        "|---|---|---|---|---|",
    ]
    membership_score = prereg.membership["score"]
    for setting in settings:
        aucs = setting["auc"]
        lines.append(
            f"| {setting['method']} | {setting['setting']} | "
            f"{_mean_sd(aucs.get('forget_vs_unseen', {}).get(membership_score))} | "
            f"{_mean_sd(aucs.get('member_vs_unseen', {}).get(membership_score))} | "
            f"{_mean_sd(aucs.get('retain_vs_unseen', {}).get(membership_score))} |"
        )
    lines.append("")

    lines += [
        "## Axis 2 — utility, and Axis 3 — selective forgetting",
        "",
        "RNFF is the fraction of the *exact-retraining* loss change achieved:",
        "1.0 means the forget set moved exactly as far as deleting it would have.",
        "Global destruction fails the retain clause by construction, because it",
        "moves retain as far as it moves forget.",
        "",
        "| method | setting | retain ppl | test ppl | RNFF forget | RNFF retain | stop reason(s) |",
        "|---|---|---|---|---|---|---|",
    ]
    for setting in settings:
        stops = sorted({
            str(v.get("stop_reason")) for v in setting["per_seed"].values()
            if v.get("stop_reason")
        })
        lines.append(
            f"| {setting['method']} | {setting['setting']} | "
            f"{_mean_sd(setting['perplexity_ratio']['retain'])} | "
            f"{_mean_sd(setting['perplexity_ratio']['test'])} | "
            f"{_mean_sd(setting['rnff']['forget'])} | "
            f"{_mean_sd(setting['rnff']['retain'])} | "
            f"{', '.join(stops) if stops else '--'} |"
        )
    lines.append("")

    # ---- detector comparison ----------------------------------------------
    primary = metadata["primary_detector"]
    lines += [
        "## Matched-access detector comparison",
        "",
        f"Primary detector: **`{primary}`** — access "
        f"`{', '.join(prereg.access_of(primary))}`.",
        "",
        "A detector may be claimed as advantageous only over detectors whose access",
        "it already subsumes. Rows where that does not hold are marked, and no claim",
        "may rest on them.",
        "",
        "| method | setting | vs detector | its access | weaker/equal | mean diff | SD | n |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for setting in settings:
        for comparison in setting["paired_differences"]:
            lines.append(
                f"| {setting['method']} | {setting['setting']} | "
                f"`{comparison['other']}` | {', '.join(comparison['other_access'])} | "
                f"{_yn(comparison['other_access_is_weaker_or_equal'])} | "
                f"{_fmt(comparison['mean'])} | {_fmt(comparison['sd'])} | "
                f"{comparison['n_seeds']} |"
            )
    lines += [
        "",
        "Positive `mean diff` means the primary detector scored higher. Paired within",
        "seed, so the between-seed variation both share cancels.",
        "",
    ]

    # ---- per seed ----------------------------------------------------------
    lines += [
        "## Per seed",
        "",
        "The replication unit is the seed, so every one is shown. A setting whose",
        "seeds disagree is reported as disagreeing, never averaged into a verdict.",
        "",
        "| method | setting | seed | classification | membership | retain ppl | RNFF forget | deletion AUC |",
        "|---|---|---:|---|---|---|---|---|",
    ]
    for setting in settings:
        for seed, entry in sorted(setting["per_seed"].items(), key=lambda kv: int(kv[0])):
            lines.append(
                f"| {setting['method']} | {setting['setting']} | {seed} | "
                f"{entry.get('classification') or '--'} | "
                f"{_fmt(entry.get('membership_auc'))} | "
                f"{_fmt((entry.get('perplexity_ratio') or {}).get('retain'))} | "
                f"{_fmt((entry.get('rnff') or {}).get('forget'))} | "
                f"{_fmt(entry.get('deletion_auc'))} |"
            )
    lines.append("")

    # ---- data quality ------------------------------------------------------
    lines += [
        "## Data quality",
        "",
        "Reported for every run, including failed and collapsed ones. Nothing is",
        "excluded from these tables.",
        "",
        "| method | setting | seed | rows | populations disjoint | truncated | non-finite |",
        "|---|---|---:|---:|---|---:|---:|",
    ]
    for run in results["runs"]:
        quality = run["data_quality"]
        nonfinite = sum(quality.get("nonfinite_counts", {}).values())
        lines.append(
            f"| {run['method']} | {run['setting']} | {run['seed']} | "
            f"{quality['n_rows']} | {_yn(quality['populations_disjoint'])} | "
            f"{quality['truncated']} | {nonfinite} |"
        )
    lines.append("")

    # ---- standing caveats --------------------------------------------------
    lines += [
        "## Caveats that apply to every number above",
        "",
        "- **ArxivMIA only.** No cross-dataset claim is licensed by anything here.",
        "- **n = 3 seeds.** Every seed is shown. Means, paired within-seed differences",
        "  and across-seed SDs are the evidence; at this n, distributional assumptions",
        "  cannot be validated and no p-value would carry weight.",
        "- **Access.** Every two-checkpoint detector, including the primary one,",
        "  requires the pre-deletion checkpoint. These are white-box deletion-audit",
        "  tools for a model owner, regulator or authorised verifier — not black-box",
        "  membership-inference attacks.",
        "- **Control displacement.** For both controls ĝ compares two independently",
        "  trained models, so their detector scores carry training randomness that an",
        "  unlearning run's small displacement does not. That is what the null refit",
        "  measures, and why no retraining number is read without it.",
        "- **Bootstrap intervals** are over samples within one run and describe that",
        "  run's precision, not variation between models.",
        "",
    ]

    return "\n".join(lines)
