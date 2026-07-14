from pathlib import Path

from tools.check_ai_docs import check_ai_docs


AI_DOCS_DIR = Path(__file__).resolve().parents[1] / "docs" / "ai"


def test_ai_documentation_has_no_structural_errors():
    errors = [issue for issue in check_ai_docs() if issue.level == "error"]
    assert errors == []


def test_commands_compatibility_route_points_to_canonical_runbook():
    compatibility_route = AI_DOCS_DIR / "commands.md"

    assert compatibility_route.is_file()
    assert "`runbooks/commands.md`" in compatibility_route.read_text(encoding="utf-8")
