import ast
from pathlib import Path


def _passivbot_class() -> ast.ClassDef:
    module = ast.parse(Path("src/passivbot.py").read_text(encoding="utf-8"))
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == "Passivbot":
            return node
    raise AssertionError("Passivbot class not found")


def test_passivbot_uses_passivbot_hsl_module_for_hsl_methods():
    cls = _passivbot_class()
    duplicate_defs = [
        node.name
        for node in cls.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and (
            node.name.startswith("_equity_hard_stop")
            or node.name
            in {
                "_calc_upnl_sum_strict",
                "_apply_equity_hard_stop_orange_overlay",
            }
        )
    ]
    assert duplicate_defs == []

    assigned_names = {
        target.id
        for node in cls.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    assert {
        "_equity_hard_stop_check",
        "_equity_hard_stop_check_coin",
        "_equity_hard_stop_run_red_supervisor",
        "_equity_hard_stop_run_coin_red_supervisor",
        "_apply_equity_hard_stop_orange_overlay",
        "_equity_hard_stop_set_red_paused_runtime_forced_modes",
        "_equity_hard_stop_latest_flatten_fill_timestamp_optional_ms",
        "_equity_hard_stop_defer_missing_flatten_fill",
        "_equity_hard_stop_flatten_fill_timestamp_with_refresh",
        "_equity_hard_stop_coverage_allow_incomplete",
        "_equity_hard_stop_required_fill_history_start_ms",
    } <= assigned_names
