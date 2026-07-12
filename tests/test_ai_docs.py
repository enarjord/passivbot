from tools.check_ai_docs import check_ai_docs


def test_ai_documentation_has_no_structural_errors():
    errors = [issue for issue in check_ai_docs() if issue.level == "error"]
    assert errors == []
