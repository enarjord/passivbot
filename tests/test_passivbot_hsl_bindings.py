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
        "_hsl_replay_cache_config_digest",
        "_hsl_replay_cache_expected_metadata",
        "_hsl_replay_cache_dir",
        "_equity_hard_stop_persist_replay_matrices",
    } <= assigned_names
