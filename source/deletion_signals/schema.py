"""The per-sample score table: row definition, file format, and readers.

Format
------
One JSON object per output file::

    {
      "schema_version": "deletion-signals/1",
      "metadata": { ... everything needed to reproduce the rows ... },
      "rows": [ { ... one per evaluated sample ... } ]
    }

Full per-sample gradient vectors are never stored -- only the scalar reductions
of them. A row is a few hundred bytes, so three seeds by two methods by ~1800
samples is a handful of megabytes.

Score files may contain JSON's non-strict ``-Infinity`` literal, which Python's
``json`` reads and writes. That is deliberate: ``-inf`` is a meaningful ranking
position for ``combined_log_score`` (a sample with zero alignment sorts below
every sample that has some), and collapsing it to ``null`` would drop the sample
from the evaluation instead of ranking it last. The offline evaluator's own
``results.json`` is strict JSON -- a non-finite value there always means "not
computed", which ``null`` expresses exactly.

Reading files back
------------------
:func:`read_score_file` handles this format. :func:`read_legacy_components`
handles the older ``per_sample_components.json`` written by the sibling
``unlearning_paradox_sanity`` pipeline, which predates this schema: it carries
only the alignment, self-influence and gradient-magnitude columns, has no sample
identifiers, and records seed and method only in its directory path. Those files
load in a degraded mode where the absent scores stay absent -- they are never
filled with zeros, which would rank as real measurements.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

SCHEMA_VERSION = "deletion-signals/1"

POPULATIONS = ("forget", "retain", "test")

# Columns that are scores (something the evaluator can rank samples by), as
# opposed to identifiers, raw measurements or bookkeeping.
SCORE_COLUMNS = (
    "loss_unlearned",
    "loss_ratio",
    "loss_delta",
    "loss_ratio_vs_finetuned",
    "loss_delta_vs_finetuned",
    "grad_norm_sq",
    "grad_ratio",
    "log_grad_ratio",
    "fuma_lora_ratio",
    "raw_cosine",
    "alignment_cos_sq",
    "sif",
    "combined_log_score",
)


def text_fingerprint(text: str) -> str:
    """Stable 16-hex-char digest of a sample's text.

    Lets rows be matched across runs and across seeds without storing the text
    itself in every output file. Two runs that drew the same sample produce the
    same fingerprint, which is what the paired comparisons key on when sample
    ordering differs.
    """
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


@dataclass
class ScoreRow:
    """One evaluated sample.

    Every field the action plan requires is present. Optional numeric fields are
    ``None`` when the score was not computed for this run (a disabled reference
    model, a not-applicable FUMA ratio) -- never 0.0, which is a legitimate value
    for several of these and would silently enter the rankings.
    """

    # --- identity and provenance --------------------------------------------
    seed: int
    method: str
    dataset: str
    model_id: str
    sample_id: str
    text_sha1: str
    population: str
    unlearned_checkpoint: Optional[str] = None
    finetuned_checkpoint: Optional[str] = None
    n_tokens: Optional[int] = None

    # --- loss family ---------------------------------------------------------
    # Two independent references, because they answer different questions.
    # ``loss_base`` is the difficulty/membership reference (the model that never
    # saw any of this data); ``loss_finetuned`` is the pre-unlearning model, so
    # the second pair isolates the change unlearning itself caused. Both raw
    # measurements stay out of SCORE_COLUMNS -- only the derived comparisons are
    # rankable scores.
    loss_unlearned: Optional[float] = None
    loss_base: Optional[float] = None
    loss_reference_model: Optional[str] = None
    loss_ratio: Optional[float] = None
    loss_delta: Optional[float] = None
    loss_finetuned: Optional[float] = None
    loss_finetuned_model: Optional[str] = None
    loss_ratio_vs_finetuned: Optional[float] = None
    loss_delta_vs_finetuned: Optional[float] = None

    # --- gradient magnitude family -------------------------------------------
    grad_norm_sq: Optional[float] = None
    grad_norm_sq_ref: Optional[float] = None
    grad_reference_model: Optional[str] = None
    grad_ratio: Optional[float] = None
    log_grad_ratio: Optional[float] = None

    # --- FUMA ----------------------------------------------------------------
    fuma_lora_ratio: Optional[float] = None
    fuma_na_reason: Optional[str] = None
    fuma_source_model: Optional[str] = None

    # --- direction family ----------------------------------------------------
    raw_cosine: Optional[float] = None
    alignment_cos_sq: Optional[float] = None

    # --- self-influence ablation ---------------------------------------------
    sif: Optional[float] = None
    fisher_explained: Optional[float] = None
    combined_log_score: Optional[float] = None

    # --- numerical-validity flags --------------------------------------------
    # Present on every row so a downstream filter never has to guess whether a
    # missing flag means "clean" or "not checked".
    flags: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScoreRow":
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


def make_flags(
    *,
    zero_grad_norm: bool = False,
    zero_direction_norm: bool = False,
    nonfinite_grad: bool = False,
    nonfinite_loss: bool = False,
    missing_grad_params: int = 0,
    truncated: bool = False,
) -> Dict[str, Any]:
    """The numerical-validity flag block, with every key always present."""
    return {
        "zero_grad_norm": bool(zero_grad_norm),
        "zero_direction_norm": bool(zero_direction_norm),
        "nonfinite_grad": bool(nonfinite_grad),
        "nonfinite_loss": bool(nonfinite_loss),
        "missing_grad_params": int(missing_grad_params),
        "truncated": bool(truncated),
    }


@dataclass
class ScoreFile:
    """A parsed score file: metadata plus rows."""

    metadata: Dict[str, Any]
    rows: List[Dict[str, Any]]
    path: Optional[str] = None
    schema_version: str = SCHEMA_VERSION
    legacy: bool = False

    @property
    def available_scores(self) -> List[str]:
        """Score columns for which at least one row carries a finite value."""
        present: List[str] = []
        for column in SCORE_COLUMNS:
            for row in self.rows:
                value = row.get(column)
                if value is None:
                    continue
                try:
                    if math.isnan(float(value)):
                        continue
                except (TypeError, ValueError):
                    continue
                present.append(column)
                break
        return present


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:  # pragma: no cover - defensive
            pass
    if isinstance(value, set):
        return sorted(value)
    return str(value)


def write_score_file(
    path: str,
    rows: Sequence[ScoreRow],
    metadata: Dict[str, Any],
    indent: Optional[int] = None,
) -> str:
    """Write rows and metadata atomically (temp file plus rename).

    A collection run costs GPU hours; a half-written file from an interrupted
    write would be indistinguishable from a complete one at read time.
    """
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "metadata": metadata,
        "rows": [row.as_dict() if isinstance(row, ScoreRow) else dict(row) for row in rows],
    }
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as handle:
        json.dump(payload, handle, indent=indent, default=_json_default)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)
    return path


def read_score_file(path: str) -> ScoreFile:
    """Read a schema-v1 score file."""
    with open(path, "r") as handle:
        payload = json.load(handle)

    if not isinstance(payload, dict) or "rows" not in payload:
        raise ValueError(
            f"{path} is not a deletion-signals score file (no 'rows' key). If it "
            f"is a legacy per_sample_components.json, load it with "
            f"read_legacy_components() or pass --legacy to offline_eval."
        )

    version = payload.get("schema_version", "unknown")
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"{path} declares schema_version={version!r}; this build reads "
            f"{SCHEMA_VERSION!r}."
        )

    return ScoreFile(
        metadata=payload.get("metadata", {}),
        rows=list(payload.get("rows", [])),
        path=path,
        schema_version=version,
    )


# Legacy files live at <something>/results_seed<N>/<method>/per_sample_components.json
_LEGACY_SEED_RE = re.compile(r"results?_?seed[_-]?(\d+)", re.IGNORECASE)

# Legacy column -> schema-v1 column. Everything not listed here has no legacy
# equivalent and stays absent.
_LEGACY_COLUMN_MAP = {
    "cos_sq": "alignment_cos_sq",
    "grad_norm_sq": "grad_norm_sq",
    "sif_value": "sif",
    "log_score": "combined_log_score",
    "fisher_explained": "fisher_explained",
}


def infer_legacy_identity(path: str) -> Dict[str, Any]:
    """Recover seed and method from a legacy file's directory path.

    Legacy files record neither, so the path is the only source. A path that
    does not encode a seed yields ``seed=None``; the caller must then supply one
    explicitly, because rows without a seed cannot enter a paired across-seed
    comparison.
    """
    absolute = os.path.abspath(path)
    method = os.path.basename(os.path.dirname(absolute))
    seed: Optional[int] = None
    for part in absolute.split(os.sep):
        match = _LEGACY_SEED_RE.search(part)
        if match:
            seed = int(match.group(1))
    return {"seed": seed, "method": method}


def read_legacy_components(
    path: str,
    seed: Optional[int] = None,
    method: Optional[str] = None,
    dataset: str = "unknown",
    model_id: str = "unknown",
) -> ScoreFile:
    """Read a legacy ``per_sample_components.json`` in degraded mode.

    The legacy format is ``{population: [ {score, log_score, sif_value,
    grad_norm_sq, cos_sq, fisher_explained}, ... ]}`` -- positional lists with
    no sample identity. Rows therefore get ``sample_id = "<population>:<index>"``
    and an empty fingerprint: they can be compared across scores within a file,
    and across seeds only by population, never sample-by-sample.

    Scores absent from the legacy format (every loss, every calibrated
    magnitude, the FUMA ratio) are left as ``None``. The evaluator reports them
    as unavailable rather than computing a metric from filler values.
    """
    with open(path, "r") as handle:
        payload = json.load(handle)

    if not isinstance(payload, dict) or not set(payload) & set(POPULATIONS):
        raise ValueError(
            f"{path} does not look like a legacy per_sample_components.json "
            f"(expected top-level keys among {POPULATIONS}, got "
            f"{sorted(payload)[:5] if isinstance(payload, dict) else type(payload)})"
        )

    identity = infer_legacy_identity(path)
    seed = seed if seed is not None else identity["seed"]
    method = method or identity["method"]

    if seed is None:
        raise ValueError(
            f"Could not infer a seed from {path!r} and none was supplied. Pass "
            f"--legacy-seed; rows without a seed cannot take part in the paired "
            f"across-seed comparison."
        )

    rows: List[Dict[str, Any]] = []
    for population in POPULATIONS:
        for index, entry in enumerate(payload.get(population, [])):
            row = ScoreRow(
                seed=int(seed),
                method=str(method),
                dataset=dataset,
                model_id=model_id,
                sample_id=f"{population}:{index}",
                text_sha1="",
                population=population,
                flags=make_flags(),
            ).as_dict()
            for legacy_key, column in _LEGACY_COLUMN_MAP.items():
                if legacy_key in entry:
                    row[column] = entry[legacy_key]
            rows.append(row)

    metadata = {
        "source": "legacy per_sample_components.json",
        "source_path": os.path.abspath(path),
        "seed": int(seed),
        "method": method,
        "dataset": dataset,
        "model_id": model_id,
        "degraded": True,
        "absent_scores": sorted(set(SCORE_COLUMNS) - set(_LEGACY_COLUMN_MAP.values())),
        "caveats": [
            "no per-sample identifiers: rows are matched positionally within a "
            "population, so cross-seed pairing is by population only",
            "seed and method were inferred from the directory path",
            "loss, calibrated-loss, calibrated-gradient and FUMA columns do not "
            "exist in this format and are absent, not zero",
        ],
    }
    return ScoreFile(metadata=metadata, rows=rows, path=path, legacy=True)


def load_any(path: str, **legacy_kwargs: Any) -> ScoreFile:
    """Read either format, deciding from the file's own contents."""
    with open(path, "r") as handle:
        head = json.load(handle)
    if isinstance(head, dict) and "rows" in head:
        return read_score_file(path)
    return read_legacy_components(path, **legacy_kwargs)


def concat_rows(files: Iterable[ScoreFile]) -> List[Dict[str, Any]]:
    """Flatten several score files into one row list."""
    rows: List[Dict[str, Any]] = []
    for score_file in files:
        rows.extend(score_file.rows)
    return rows
