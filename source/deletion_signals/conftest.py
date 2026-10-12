"""Put the repository root on ``sys.path`` so ``import deletion_signals`` works.

The test suite lives inside the package, so pytest's rootdir is the package
directory rather than its parent. Without this, ``pytest`` run from anywhere
other than the repository root cannot import the package it is testing.
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
