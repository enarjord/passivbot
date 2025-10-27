"""Risk management utilities for automated trading portfolios.

This package exposes a lightweight command line dashboard that demonstrates
how monitoring and alerting can work without requiring access to live
exchange credentials. The implementation is intentionally pure Python so it
can run inside the sandboxed execution environment used for the exercises.
"""

from __future__ import annotations

from pathlib import Path
import sys

__all__ = ["__version__"]

__version__ = "0.1.0"


def _ensure_src_on_path() -> None:
    """Expose the repository ``src`` directory for local executions."""

    package_root = Path(__file__).resolve().parent
    src_dir = package_root.parent / "src"
    if not src_dir.is_dir():
        return

    src_path = str(src_dir)
    if src_path not in sys.path:
        # Prepend so that an editable install still takes precedence when present.
        sys.path.insert(0, src_path)


_ensure_src_on_path()

