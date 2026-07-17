from pathlib import Path

from tools.check_ai_docs import check_ai_docs


AI_DOCS_DIR = Path(__file__).resolve().parents[1] / "docs" / "ai"


def test_ai_documentation_has_no_structural_errors():
    errors = [issue for issue in check_ai_docs() if issue.level == "error"]
    assert errors == []


def test_compatibility_routes_point_to_canonical_documents():
    expected_routes = {
        "commands.md": "`runbooks/commands.md`",
        "pr_auto_review_loop.md": "`runbooks/pr_review.md`",
        "code_review_prompt.md": "`validation.md`",
        "principles.yaml": "canonical_document: docs/ai/principles.md",
    }

    for route_name, canonical_reference in expected_routes.items():
        compatibility_route = AI_DOCS_DIR / route_name
        assert compatibility_route.is_file()
        assert canonical_reference in compatibility_route.read_text(encoding="utf-8")


def test_pr_review_contract_preserves_scheduler_and_verdict_semantics():
    contract = " ".join(
        (AI_DOCS_DIR / "runbooks" / "pr_review.md").read_text(encoding="utf-8").split()
    )

    required_contracts = [
        "digests of CI and review/comment metadata",
        "exact base, head, and effective merge-base identities",
        "Scope completed-review records by reviewer and those identities",
        "The target-relative production, test, configuration, and contract diff is unchanged",
        "records the old and new heads, target SHA, inspected delta, validation",
        (
            "Re-fetch the exact base and head and recompute the effective merge base immediately "
            "before posting"
        ),
        (
            "Every completed review records the reviewer identity, exact base, head, and effective "
            "merge-base SHAs"
        ),
        "This marker records completion by that reviewer, not approval",
        (
            "A requested draft review remains advisory and uses `COMMENT` unless formal approval "
            "of the draft was explicitly requested"
        ),
    ]

    for required_contract in required_contracts:
        assert required_contract in contract
