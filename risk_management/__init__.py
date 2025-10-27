"""Risk management utilities for automated trading portfolios.

This package exposes a lightweight command line dashboard that demonstrates
how monitoring and alerting can work without requiring access to live
exchange credentials. The implementation is intentionally pure Python so it
can run inside the sandboxed execution environment used for the exercises.
"""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
import sys

__all__ = ["__version__"]

__version__ = "0.1.0"


def _ensure_passivbot_modules_available() -> None:
    """Ensure the Passivbot source tree is importable when running from the repo.

    The risk management tools live alongside the main Passivbot sources.  When
    executed from an isolated virtual environment (for example via
    ``python -m risk_management.web_server``) the interpreter may not know about
    the sibling ``src`` directory that exposes modules such as
    :mod:`custom_endpoint_overrides`.  Historically the installer dropped a
    ``.pth`` file into the environment to bridge the gap, but direct execution
    outside of that flow regressed after refactoring the custom endpoint
    loading.

    To keep the developer experience intact we lazily add ``../src`` to
    ``sys.path`` only when the import fails.  The behaviour mirrors what the
    installer performs and avoids interfering when Passivbot is installed as a
    proper package.
    """

    try:
        import_module("custom_endpoint_overrides")
    except ModuleNotFoundError:
        repo_root = Path(__file__).resolve().parent.parent
        src_path = repo_root / "src"
        if src_path.is_dir():
            sys.path.insert(0, str(src_path))
            try:
                import_module("custom_endpoint_overrides")
            except ModuleNotFoundError as exc:  # pragma: no cover - sanity guard
                raise ModuleNotFoundError(
                    "Passivbot modules are unavailable. Install Passivbot or run "
                    "commands from the repository root so ../src is importable."
                ) from exc


_ensure_passivbot_modules_available()

