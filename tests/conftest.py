"""Pytest configuration hooks."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the src directory is importable when running tests without installing the package.
SRC_PATH = Path(__file__).resolve().parents[1] / "src"
SYS_PATH = str(SRC_PATH)
if SYS_PATH not in sys.path:
    sys.path.insert(0, SYS_PATH)
