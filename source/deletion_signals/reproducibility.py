"""Deterministic seeding and run provenance.

Every RNG the pipeline draws from is seeded here. Data splits use ``random``,
bootstrap and permutation tests use ``numpy``, and training uses ``torch``;
seeding one of them and not the others leaves the run irreproducible in a way
that only shows up as unexplained across-seed variance.
"""

from __future__ import annotations

import os
import platform
import random
import subprocess
import sys
from typing import Any, Dict, Optional

_REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def set_seed(seed: int, deterministic_algorithms: bool = False) -> None:
    """Seed python, numpy and torch (including all CUDA devices).

    Args:
        seed: The seed.
        deterministic_algorithms: Ask torch to refuse non-deterministic kernels.
            Off by default because it makes some attention kernels unavailable
            and slows training measurably; the per-sample scoring path does not
            need it, since scoring is a pure function of the checkpoints.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # numpy is required in practice; do not hard-fail here
        pass

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic_algorithms:
            torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass


def new_rng(seed: int):
    """A private numpy Generator, so callers cannot perturb each other.

    Bootstrap and permutation tests take one of these rather than touching the
    global numpy RNG: two resampling routines sharing global state produce
    results that depend on the order they happened to be called in.
    """
    import numpy as np

    return np.random.default_rng(seed)


def git_revision(repo_dir: Optional[str] = None) -> str:
    """Current git commit with a ``-dirty`` suffix, or ``"unknown"``."""
    repo_dir = repo_dir or _REPO_DIR
    try:
        sha = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=repo_dir, stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
        dirty = (
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=repo_dir,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
        return f"{sha}-dirty" if dirty else sha
    except Exception:
        return "unknown"


def run_provenance(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Everything needed to reproduce a row, for the output metadata block."""
    provenance: Dict[str, Any] = {
        "git_commit": git_revision(),
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "argv": list(sys.argv),
    }

    try:
        import torch

        provenance["torch_version"] = torch.__version__
        provenance["cuda_available"] = torch.cuda.is_available()
        provenance["gpu"] = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        )
    except ImportError:
        provenance["torch_version"] = None
        provenance["cuda_available"] = False
        provenance["gpu"] = None

    for package in ("transformers", "peft", "datasets", "numpy", "scipy", "sklearn"):
        try:
            module = __import__(package)
            provenance[f"{package}_version"] = getattr(module, "__version__", None)
        except ImportError:
            provenance[f"{package}_version"] = None

    if config is not None:
        provenance["effective_config"] = config

    return provenance
