"""The score-table format: round-trip, atomic write, and the legacy reader."""

from __future__ import annotations

import json
import os

import pytest

from deletion_signals.schema import (
    SCHEMA_VERSION,
    SCORE_COLUMNS,
    ScoreRow,
    infer_legacy_identity,
    load_any,
    make_flags,
    read_legacy_components,
    read_score_file,
    text_fingerprint,
    write_score_file,
)


def _row(**overrides) -> ScoreRow:
    base = dict(
        seed=42,
        method="ga_gd",
        dataset="swj0419/WikiMIA",
        model_id="meta-llama/Llama-3.2-1B",
        sample_id="forget:0",
        text_sha1=text_fingerprint("hello"),
        population="forget",
        loss_unlearned=2.5,
        grad_norm_sq=1.5,
        alignment_cos_sq=0.001,
        flags=make_flags(),
    )
    base.update(overrides)
    return ScoreRow(**base)


# ------------------------------------------------------------- fingerprinting


def test_fingerprint_is_stable_and_short():
    assert text_fingerprint("hello") == text_fingerprint("hello")
    assert len(text_fingerprint("hello")) == 16


def test_fingerprint_distinguishes_texts():
    assert text_fingerprint("a") != text_fingerprint("b")


# ------------------------------------------------------------------ round-trip


def test_round_trip_preserves_rows_and_metadata(tmp_path):
    path = os.path.join(tmp_path, "scores.json")
    write_score_file(path, [_row(), _row(sample_id="retain:0", population="retain")],
                     {"seed": 42, "method": "ga_gd"})
    parsed = read_score_file(path)
    assert parsed.schema_version == SCHEMA_VERSION
    assert len(parsed.rows) == 2
    assert parsed.rows[0]["sample_id"] == "forget:0"
    assert parsed.metadata["seed"] == 42


def test_write_is_atomic_and_leaves_no_temp_file(tmp_path):
    path = os.path.join(tmp_path, "scores.json")
    write_score_file(path, [_row()], {})
    assert os.path.exists(path)
    assert not os.path.exists(path + ".tmp")


def test_every_row_carries_the_full_flag_block(tmp_path):
    path = os.path.join(tmp_path, "scores.json")
    write_score_file(path, [_row()], {})
    flags = read_score_file(path).rows[0]["flags"]
    for key in ("zero_grad_norm", "zero_direction_norm", "nonfinite_grad",
                "nonfinite_loss", "missing_grad_params", "truncated"):
        assert key in flags


def test_available_scores_reports_only_columns_with_finite_values(tmp_path):
    path = os.path.join(tmp_path, "scores.json")
    write_score_file(path, [_row(fuma_lora_ratio=None, sif=None)], {})
    available = read_score_file(path).available_scores
    assert "alignment_cos_sq" in available
    assert "grad_norm_sq" in available
    assert "fuma_lora_ratio" not in available
    assert "sif" not in available


def test_a_not_applicable_fuma_ratio_stays_none_rather_than_becoming_zero(tmp_path):
    path = os.path.join(tmp_path, "scores.json")
    write_score_file(path, [_row(fuma_lora_ratio=None, fuma_na_reason="no adapters")], {})
    row = read_score_file(path).rows[0]
    assert row["fuma_lora_ratio"] is None
    assert row["fuma_na_reason"] == "no adapters"


def test_reading_a_foreign_file_says_what_to_do(tmp_path):
    path = os.path.join(tmp_path, "other.json")
    with open(path, "w") as handle:
        json.dump({"something": "else"}, handle)
    with pytest.raises(ValueError, match="--legacy"):
        read_score_file(path)


def test_reading_a_future_schema_version_is_refused(tmp_path):
    path = os.path.join(tmp_path, "scores.json")
    with open(path, "w") as handle:
        json.dump({"schema_version": "deletion-signals/99", "metadata": {}, "rows": []}, handle)
    with pytest.raises(ValueError, match="schema_version"):
        read_score_file(path)


def test_score_row_from_dict_ignores_unknown_columns():
    row = ScoreRow.from_dict({**_row().as_dict(), "a_future_column": 1})
    assert row.sample_id == "forget:0"


# ---------------------------------------------------------------- legacy files


LEGACY_PAYLOAD = {
    "forget": [
        {"score": 12.5, "log_score": 2.5, "sif_value": 14673.3,
         "grad_norm_sq": 15.3, "cos_sq": 8.5e-4, "fisher_explained": 0.04},
        {"score": 9.0, "log_score": 2.2, "sif_value": 10000.0,
         "grad_norm_sq": 11.0, "cos_sq": 7.0e-4, "fisher_explained": 0.03},
    ],
    "retain": [
        {"score": 3.0, "log_score": 1.1, "sif_value": 9000.0,
         "grad_norm_sq": 9.5, "cos_sq": 1.5e-4, "fisher_explained": 0.02},
    ],
    "test": [
        {"score": 2.0, "log_score": 0.7, "sif_value": 8000.0,
         "grad_norm_sq": 8.0, "cos_sq": 1.2e-4, "fisher_explained": 0.02},
    ],
}


def _write_legacy(tmp_path, seed: int = 42, method: str = "ga_gd") -> str:
    directory = os.path.join(tmp_path, f"results_seed{seed}", method)
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, "per_sample_components.json")
    with open(path, "w") as handle:
        json.dump(LEGACY_PAYLOAD, handle)
    return path


def test_legacy_seed_and_method_come_from_the_directory_path(tmp_path):
    path = _write_legacy(tmp_path, seed=43, method="scrub")
    identity = infer_legacy_identity(path)
    assert identity == {"seed": 43, "method": "scrub"}


def test_legacy_rows_map_onto_the_new_column_names(tmp_path):
    parsed = read_legacy_components(_write_legacy(tmp_path))
    assert len(parsed.rows) == 4
    forget = parsed.rows[0]
    assert forget["population"] == "forget"
    assert forget["alignment_cos_sq"] == pytest.approx(8.5e-4)
    assert forget["grad_norm_sq"] == pytest.approx(15.3)
    assert forget["sif"] == pytest.approx(14673.3)
    assert forget["combined_log_score"] == pytest.approx(2.5)


def test_legacy_absent_scores_are_none_never_zero(tmp_path):
    parsed = read_legacy_components(_write_legacy(tmp_path))
    for row in parsed.rows:
        for column in ("loss_unlearned", "loss_ratio", "loss_delta", "grad_ratio",
                       "log_grad_ratio", "fuma_lora_ratio"):
            assert row[column] is None, f"{column} must stay absent, not become 0.0"


def test_legacy_available_scores_lists_only_what_the_format_carries(tmp_path):
    parsed = read_legacy_components(_write_legacy(tmp_path))
    assert set(parsed.available_scores) == {
        "grad_norm_sq", "alignment_cos_sq", "sif", "combined_log_score"
    }
    assert set(SCORE_COLUMNS) - set(parsed.available_scores)


def test_legacy_metadata_flags_the_degradation_and_its_caveats(tmp_path):
    parsed = read_legacy_components(_write_legacy(tmp_path))
    assert parsed.legacy is True
    assert parsed.metadata["degraded"] is True
    assert any("positionally" in caveat for caveat in parsed.metadata["caveats"])
    assert "loss_ratio" in parsed.metadata["absent_scores"]


def test_legacy_sample_ids_are_positional_within_a_population(tmp_path):
    parsed = read_legacy_components(_write_legacy(tmp_path))
    assert [r["sample_id"] for r in parsed.rows] == [
        "forget:0", "forget:1", "retain:0", "test:0"
    ]
    assert all(r["text_sha1"] == "" for r in parsed.rows)


def test_legacy_without_a_seed_in_the_path_demands_one(tmp_path):
    directory = os.path.join(tmp_path, "somewhere", "ga_gd")
    os.makedirs(directory)
    path = os.path.join(directory, "per_sample_components.json")
    with open(path, "w") as handle:
        json.dump(LEGACY_PAYLOAD, handle)
    with pytest.raises(ValueError, match="--legacy-seed"):
        read_legacy_components(path)


def test_legacy_seed_can_be_supplied_explicitly(tmp_path):
    directory = os.path.join(tmp_path, "somewhere", "ga_gd")
    os.makedirs(directory)
    path = os.path.join(directory, "per_sample_components.json")
    with open(path, "w") as handle:
        json.dump(LEGACY_PAYLOAD, handle)
    parsed = read_legacy_components(path, seed=7)
    assert all(row["seed"] == 7 for row in parsed.rows)


def test_load_any_dispatches_on_content_not_on_filename(tmp_path):
    legacy = load_any(_write_legacy(tmp_path))
    assert legacy.legacy is True

    modern_path = os.path.join(tmp_path, "per_sample_components.json")
    write_score_file(modern_path, [_row()], {})
    modern = load_any(modern_path)
    assert modern.legacy is False


def test_a_legacy_file_with_no_known_populations_is_rejected(tmp_path):
    path = os.path.join(tmp_path, "bad.json")
    with open(path, "w") as handle:
        json.dump({"nope": []}, handle)
    with pytest.raises(ValueError, match="legacy"):
        read_legacy_components(path)
