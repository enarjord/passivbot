from pathlib import Path

from tools.generate_live_event_registry import render_registry


REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_DOC = REPO_ROOT / "docs" / "ai" / "generated" / "live_event_registry.md"


def test_generated_live_event_registry_is_current():
    assert REGISTRY_DOC.read_text(encoding="utf-8") == render_registry()
