"""The preregistered score-orientation table.

Loads ``configs/orientations.yaml`` and applies it. There is deliberately no
function in this module that computes ``max(auc, 1 - auc)``: the orientation of
every score on every task is fixed before measurement, and a score that lands
below 0.5 under its declared direction is reported below 0.5.

A lookup for a score/task pair that the table does not declare raises rather
than defaulting, so adding a new score forces a written-down direction and a
rationale before any number can be produced from it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

HIGHER_IS_POSITIVE = "higher_is_positive"
LOWER_IS_POSITIVE = "lower_is_positive"
_VALID_DIRECTIONS = (HIGHER_IS_POSITIVE, LOWER_IS_POSITIVE)

DEFAULT_TABLE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "configs", "orientations.yaml"
)


@dataclass(frozen=True)
class TaskSpec:
    """A binary evaluation task defined over sample populations."""

    name: str
    positive: Tuple[str, ...]
    negative: Tuple[str, ...]
    role: str = "diagnostic"
    description: str = ""

    def label_of(self, population: str) -> Optional[int]:
        """1 for a positive population, 0 for a negative one, None if excluded."""
        if population in self.positive:
            return 1
        if population in self.negative:
            return 0
        return None


@dataclass(frozen=True)
class Orientation:
    """The declared direction of one score on one task."""

    score: str
    task: str
    direction: str
    rationale: str
    headline: bool

    @property
    def sign(self) -> float:
        """+1 if higher means positive, -1 if lower does.

        Multiplying the raw score by this makes "larger is more positive" true
        for every score, which is what ``roc_auc_score`` assumes.
        """
        return 1.0 if self.direction == HIGHER_IS_POSITIVE else -1.0


class OrientationTable:
    """Parsed ``orientations.yaml``, with lookups that fail loudly."""

    def __init__(
        self,
        tasks: Mapping[str, TaskSpec],
        orientations: Mapping[Tuple[str, str], Orientation],
        labels: Mapping[str, str],
        source_path: Optional[str] = None,
    ):
        self._tasks = dict(tasks)
        self._orientations = dict(orientations)
        self._labels = dict(labels)
        self.source_path = source_path

    # ---------------------------------------------------------------- loading

    @classmethod
    def load(cls, path: Optional[str] = None) -> "OrientationTable":
        import yaml

        path = path or DEFAULT_TABLE_PATH
        with open(path, "r") as handle:
            raw = yaml.safe_load(handle)

        tasks: Dict[str, TaskSpec] = {}
        for name, spec in (raw.get("tasks") or {}).items():
            positive = tuple(spec["positive"])
            negative = tuple(spec["negative"])
            overlap = set(positive) & set(negative)
            if overlap:
                raise ValueError(
                    f"Task {name!r} lists population(s) {sorted(overlap)} as both "
                    f"positive and negative."
                )
            tasks[name] = TaskSpec(
                name=name,
                positive=positive,
                negative=negative,
                role=spec.get("role", "diagnostic"),
                description=(spec.get("description") or "").strip(),
            )
        if not tasks:
            raise ValueError(f"{path} declares no tasks")

        orientations: Dict[Tuple[str, str], Orientation] = {}
        labels: Dict[str, str] = {}
        for score, spec in (raw.get("scores") or {}).items():
            labels[score] = spec.get("label", score)
            headline = bool(spec.get("headline", True))
            declared = [k for k in spec if k in tasks]
            missing = sorted(set(tasks) - set(declared))
            if missing:
                raise ValueError(
                    f"Score {score!r} in {path} has no declared orientation for "
                    f"task(s) {missing}. Every score must declare a direction "
                    f"for every task before it can be evaluated."
                )
            for task_name in declared:
                entry = spec[task_name]
                direction = entry["direction"]
                if direction not in _VALID_DIRECTIONS:
                    raise ValueError(
                        f"Score {score!r}, task {task_name!r}: direction "
                        f"{direction!r} is not one of {_VALID_DIRECTIONS}"
                    )
                rationale = (entry.get("rationale") or "").strip()
                if not rationale:
                    raise ValueError(
                        f"Score {score!r}, task {task_name!r} has no rationale. "
                        f"A direction without a stated reason is not a "
                        f"preregistration."
                    )
                orientations[(score, task_name)] = Orientation(
                    score=score,
                    task=task_name,
                    direction=direction,
                    rationale=rationale,
                    headline=headline,
                )

        return cls(tasks, orientations, labels, source_path=path)

    # ---------------------------------------------------------------- lookups

    @property
    def task_names(self) -> List[str]:
        return list(self._tasks)

    @property
    def score_names(self) -> List[str]:
        return sorted({score for score, _ in self._orientations})

    def task(self, name: str) -> TaskSpec:
        if name not in self._tasks:
            raise KeyError(
                f"Unknown task {name!r}; declared tasks are {sorted(self._tasks)}"
            )
        return self._tasks[name]

    def label(self, score: str) -> str:
        return self._labels.get(score, score)

    def orientation(self, score: str, task: str) -> Orientation:
        key = (score, task)
        if key not in self._orientations:
            raise KeyError(
                f"No preregistered orientation for score {score!r} on task "
                f"{task!r}. Add it to {self.source_path or 'orientations.yaml'} "
                f"with a rationale before evaluating this score -- the direction "
                f"must be fixed before the measurement, not after."
            )
        return self._orientations[key]

    def sign(self, score: str, task: str) -> float:
        return self.orientation(score, task).sign

    def headline_scores(self, task: str) -> List[str]:
        """Scores that belong in the primary comparison table for ``task``."""
        return sorted(
            score
            for (score, task_name), orientation in self._orientations.items()
            if task_name == task and orientation.headline
        )

    def orient(self, values: Iterable[float], score: str, task: str) -> List[float]:
        """Multiply by the declared sign so that larger always means positive."""
        sign = self.sign(score, task)
        return [sign * float(v) for v in values]

    def as_dict(self) -> Dict[str, object]:
        """Serialisable snapshot, embedded in every output file's metadata."""
        return {
            "source_path": self.source_path,
            "tasks": {
                name: {
                    "role": spec.role,
                    "positive": list(spec.positive),
                    "negative": list(spec.negative),
                    "description": spec.description,
                }
                for name, spec in self._tasks.items()
            },
            "scores": {
                score: {
                    "label": self._labels.get(score, score),
                    "headline": any(
                        o.headline
                        for (s, _), o in self._orientations.items()
                        if s == score
                    ),
                    "orientations": {
                        task: {
                            "direction": o.direction,
                            "rationale": o.rationale,
                        }
                        for (s, task), o in sorted(self._orientations.items())
                        if s == score
                    },
                }
                for score in self.score_names
            },
        }


def build_task_arrays(
    rows: Sequence[Mapping[str, object]],
    score: str,
    task: TaskSpec,
) -> Tuple[List[float], List[int], int]:
    """Extract ``(values, labels, n_dropped)`` for one score on one task.

    Rows whose population is in neither class are excluded. Rows whose value for
    this score is missing (``None``) or non-finite are dropped and counted --
    a not-applicable FUMA ratio must not become a 0.0 that quietly ranks last.
    """
    import math

    values: List[float] = []
    labels: List[int] = []
    dropped = 0

    for row in rows:
        label = task.label_of(str(row.get("population")))
        if label is None:
            continue
        raw = row.get(score, None)
        if raw is None:
            dropped += 1
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            dropped += 1
            continue
        # -inf is a meaningful ranking position for combined_log_score (zero
        # alignment) and is kept; NaN and +inf are not and are dropped.
        if math.isnan(value) or value == float("inf"):
            dropped += 1
            continue
        values.append(value)
        labels.append(label)

    return values, labels, dropped
