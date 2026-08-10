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
        "Distinguish a valid empty decision from malformed producer output and unavailable input",
        "Reject proposals that preserve, synthesize, or reinterpret strategy intent outside Rust",
    ]

    for required_contract in required_contracts:
        assert required_contract in contract


def test_trading_contract_boundaries_remain_explicit():
    principles = " ".join((AI_DOCS_DIR / "principles.md").read_text(encoding="utf-8").split())
    architecture = " ".join((AI_DOCS_DIR / "architecture.md").read_text(encoding="utf-8").split())
    error_contract = " ".join(
        (AI_DOCS_DIR / "error_contract.md").read_text(encoding="utf-8").split()
    )

    assert "A Rust ideal-order result is atomic current intent" in principles
    assert "not identical raw-data availability" in principles
    assert "An absent ideal authorizes cancellation only within such a batch" in architecture
    assert "A malformed Rust ideal-order batch is fatal before reconciliation" in error_contract


def test_release_hygiene_trigger_is_always_routed():
    principles = " ".join((AI_DOCS_DIR / "principles.md").read_text(encoding="utf-8").split())
    router = " ".join((AI_DOCS_DIR / "README.md").read_text(encoding="utf-8").split())
    release_runbook = AI_DOCS_DIR / "runbooks" / "release.md"

    assert "50 top-level user-facing entries" in principles
    assert "14 days since the latest stable tag with at least 10" in principles
    assert "Ask for explicit permission" in principles
    assert "Version selection, release trigger, release preparation, or publication" in router
    assert release_runbook.is_file()
