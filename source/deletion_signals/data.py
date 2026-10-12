"""Deterministic forget/retain/test splits, and the split manifest format.

A sample's identity has to survive between the training run and the scoring run,
and between seeds. That is what the manifest is for: ``build_splits`` produces
one, training and scoring both consume it, and every output row carries the
sample id and text fingerprint it came from. Regenerating the split from the same
config and seed reproduces the manifest exactly.

The manifest is also the seam that makes the collector method-agnostic. Any
pipeline that can write one -- including one outside this package -- can have its
checkpoints scored here.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Mapping, Optional, Sequence

from torch.utils.data import Dataset

from .schema import POPULATIONS, text_fingerprint

# Datasets in this family put the text under different column names. Inferring
# it is a convenience; ``data.text_field`` in the config overrides.
_TEXT_FIELD_BY_DATASET = {
    "wikimia": "input",
    "bookmia": "snippet",
    "arxivmia": "text",
    "xsum": "text",
    "mimir": "member",
}
_TEXT_FIELD_FALLBACKS = ("text", "input", "snippet", "sentence", "content")


@dataclass(frozen=True)
class Sample:
    """One text with a stable identity."""

    sample_id: str
    text: str
    population: str
    text_sha1: str

    @classmethod
    def create(cls, index: int, text: str, population: str) -> "Sample":
        return cls(
            sample_id=f"{population}:{index}",
            text=text,
            population=population,
            text_sha1=text_fingerprint(text),
        )


class SampleDataset(Dataset):
    """Minimal ``torch`` dataset over :class:`Sample`, yielding ``{"text": ...}``."""

    def __init__(self, samples: Sequence[Sample]):
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.samples[index]
        return {"text": sample.text, "sample_id": sample.sample_id}

    @property
    def texts(self) -> List[str]:
        return [s.text for s in self.samples]


def collate_fn(batch: Sequence[Mapping[str, Any]], tokenizer, max_length: int) -> Dict:
    """Tokenise a batch and mask padding out of the labels.

    Padding positions get label ``-100``. Without that the model is trained, and
    scored, on predicting pad tokens, and a sample's loss would depend on how
    much padding the other members of its batch happened to need -- which makes
    per-sample scores a function of batch composition.
    """
    texts = [item["text"] for item in batch]
    encodings = tokenizer(
        texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
    )
    labels = encodings["input_ids"].clone()
    labels[encodings["attention_mask"] == 0] = -100
    encodings["labels"] = labels
    return encodings


def _infer_text_field(dataset_name: str, columns: Sequence[str]) -> str:
    lowered = dataset_name.lower()
    for marker, field in _TEXT_FIELD_BY_DATASET.items():
        if marker in lowered and field in columns:
            return field
    for candidate in _TEXT_FIELD_FALLBACKS:
        if candidate in columns:
            return candidate
    remaining = [c for c in columns if c not in ("label", "id", "index")]
    if not remaining:
        raise ValueError(
            f"Cannot infer the text column of {dataset_name!r} from {list(columns)}; "
            f"set data.text_field in the config."
        )
    return remaining[0]


def _infer_split(dataset_name: str, configured: Optional[str]) -> str:
    if configured:
        return configured
    lowered = dataset_name.lower()
    if "wikimia" in lowered:
        # WikiMIA ships length-bucketed splits rather than a "train" split.
        return "WikiMIA_length32"
    return "train"


def build_splits(config: Mapping[str, Any], seed: int) -> Dict[str, List[Sample]]:
    """Build D_forget, D_retain and D_test deterministically from ``seed``.

    Uses a private ``random.Random(seed)`` rather than the global RNG, so the
    split does not depend on how much other code happened to draw before it.

    Returns a mapping with keys ``forget``, ``retain`` and ``test``. Members
    (label 1) are sampled and then partitioned by ``data.forget_ratio``;
    non-members (label 0) become the unseen test population.
    """
    from datasets import load_dataset

    data_cfg = config["data"]
    dataset_name = data_cfg["name"]
    split = _infer_split(dataset_name, data_cfg.get("split"))
    label_field = data_cfg.get("label_field", "label")

    dataset = load_dataset(dataset_name, split=split)
    if label_field not in dataset.column_names:
        raise ValueError(
            f"Dataset {dataset_name!r} has no {label_field!r} column (columns: "
            f"{dataset.column_names}); members and non-members cannot be "
            f"separated. Defaulting the label would place every sample in both "
            f"populations and contaminate D_test with training data."
        )

    text_field = data_cfg.get("text_field") or _infer_text_field(
        dataset_name, dataset.column_names
    )

    members = [row for row in dataset if row[label_field] == 1]
    non_members = [row for row in dataset if row[label_field] == 0]
    if not members or not non_members:
        raise ValueError(
            f"{dataset_name}[{split}] yielded {len(members)} members and "
            f"{len(non_members)} non-members; both populations are required."
        )

    rng = random.Random(seed)
    n_train = min(int(data_cfg["train_samples"]), len(members))
    n_test = min(int(data_cfg["test_samples"]), len(non_members))
    member_rows = rng.sample(members, n_train)
    non_member_rows = rng.sample(non_members, n_test)

    train_texts = [str(row[text_field]) for row in member_rows]
    test_texts = [str(row[text_field]) for row in non_member_rows]

    order = list(range(len(train_texts)))
    rng.shuffle(order)
    n_forget = int(len(train_texts) * float(data_cfg["forget_ratio"]))
    forget_order, retain_order = order[:n_forget], order[n_forget:]

    return {
        "forget": [
            Sample.create(i, train_texts[j], "forget")
            for i, j in enumerate(forget_order)
        ],
        "retain": [
            Sample.create(i, train_texts[j], "retain")
            for i, j in enumerate(retain_order)
        ],
        "test": [Sample.create(i, text, "test") for i, text in enumerate(test_texts)],
    }


def write_manifest(
    path: str,
    splits: Mapping[str, Sequence[Sample]],
    config: Mapping[str, Any],
    seed: int,
) -> str:
    """Persist a split so training and scoring provably see the same samples."""
    payload = {
        "manifest_version": 1,
        "seed": int(seed),
        "dataset": config["data"]["name"],
        "split": _infer_split(config["data"]["name"], config["data"].get("split")),
        "forget_ratio": config["data"]["forget_ratio"],
        "counts": {population: len(splits.get(population, [])) for population in POPULATIONS},
        "populations": {
            population: [asdict(sample) for sample in splits.get(population, [])]
            for population in POPULATIONS
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as handle:
        json.dump(payload, handle)
    os.replace(tmp_path, path)
    return path


def read_manifest(path: str) -> Dict[str, Any]:
    """Read a split manifest, returning ``{"meta": ..., "splits": ...}``.

    Fingerprints are recomputed from the stored text and checked. A manifest
    whose text was edited after the fact would otherwise let the collector
    attribute rows to samples the model never saw.
    """
    with open(path, "r") as handle:
        payload = json.load(handle)

    splits: Dict[str, List[Sample]] = {}
    for population, entries in (payload.get("populations") or {}).items():
        samples: List[Sample] = []
        for entry in entries:
            sample = Sample(
                sample_id=entry["sample_id"],
                text=entry["text"],
                population=entry.get("population", population),
                text_sha1=entry.get("text_sha1", ""),
            )
            expected = text_fingerprint(sample.text)
            if sample.text_sha1 and sample.text_sha1 != expected:
                raise ValueError(
                    f"{path}: sample {sample.sample_id!r} has fingerprint "
                    f"{sample.text_sha1} but its text hashes to {expected}. The "
                    f"manifest has been modified since it was written."
                )
            samples.append(
                sample if sample.text_sha1 else Sample(
                    sample.sample_id, sample.text, sample.population, expected
                )
            )
        splits[population] = samples

    meta = {k: v for k, v in payload.items() if k != "populations"}
    return {"meta": meta, "splits": splits}


def load_or_build_splits(
    config: Mapping[str, Any], seed: int, manifest_path: Optional[str] = None
) -> Dict[str, Any]:
    """Read the manifest if one is configured, otherwise build the split.

    Returns ``{"splits": ..., "manifest_path": ..., "source": ...}``.
    """
    manifest_path = manifest_path or config["data"].get("manifest")
    if manifest_path and os.path.exists(manifest_path):
        parsed = read_manifest(manifest_path)
        manifest_seed = parsed["meta"].get("seed")
        if manifest_seed is not None and int(manifest_seed) != int(seed):
            raise ValueError(
                f"{manifest_path} was written for seed {manifest_seed} but this "
                f"run uses seed {seed}. Scoring one seed's checkpoints against "
                f"another seed's split silently mislabels every row."
            )
        return {
            "splits": parsed["splits"],
            "manifest_path": os.path.abspath(manifest_path),
            "source": "manifest",
            "meta": parsed["meta"],
        }

    splits = build_splits(config, seed)
    return {
        "splits": splits,
        "manifest_path": manifest_path,
        "source": "built",
        "meta": {"seed": seed, "dataset": config["data"]["name"]},
    }


def subsample(
    splits: Mapping[str, Sequence[Sample]],
    limit: Optional[Any] = None,
) -> Dict[str, List[Sample]]:
    """Cap each population, either uniformly or one population at a time.

    ``limit`` is ``None`` (keep everything), an ``int`` (the same cap on every
    population), or a mapping from population name to a cap, where an absent key
    or a ``None`` value means "keep all of this one". The mapping form exists so
    a run can shrink only ``test`` -- the population that costs the most and
    carries the least -- without touching ``forget`` or ``retain``.

    Takes a prefix rather than a random subset: the split order is already
    seeded, so a prefix is reproducible without introducing a second RNG whose
    state would have to be recorded too. It also keeps a shrunken population a
    subset of the full one, so runs at different caps still share samples.

    Note that this cannot be achieved by lowering ``data.test_samples`` instead:
    :func:`build_splits` draws the test set from the same RNG stream *before* it
    shuffles the members, so changing the test count also reshuffles which
    members land in forget and retain, and the run would no longer share a split
    with its siblings.
    """
    if limit is None:
        return {k: list(v) for k, v in splits.items()}

    if isinstance(limit, Mapping):
        unknown = sorted(set(limit) - set(splits))
        if unknown:
            raise ValueError(
                f"subsample() was given caps for population(s) {unknown}, which "
                f"are not in the split ({sorted(splits)}). An unrecognised key "
                f"would silently leave that population uncapped."
            )
        capped: Dict[str, List[Sample]] = {}
        for population, samples in splits.items():
            cap = limit.get(population)
            capped[population] = (
                list(samples) if cap is None else list(samples)[: int(cap)]
            )
        return capped

    return {k: list(v)[: int(limit)] for k, v in splits.items()}
