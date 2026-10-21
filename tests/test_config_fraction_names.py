"""Public scale-one names preserve native IDs, values and migration authority."""

from copy import deepcopy
import json

import pytest

from config import prepare_config
from config.field_registry import get_field_registry
from config.optimize_bounds import flatten_optimize_bounds, set_flat_optimize_bound
from config.param_paths import resolve_optimizer_key_path
from config.public_names import PUBLIC_PATH_RENAMES, SHARED_SIDE_PATH_RENAMES
from config.schema import get_template_config
from config.strategy_paths import STRATEGY_NATIVE_TO_PUBLIC_PATHS
from released_v8_fixtures import released_v8_policy
from tools.migrate_config_v9 import migrate


def assign(tree, path, value):
    for name in path[:-1]:
        tree = tree.setdefault(name, {})
    tree[path[-1]] = value


PUBLIC_PAIRS = list(PUBLIC_PATH_RENAMES.items()) + [
    (("bot", "long", *old), ("bot", "long", *new))
    for old, new in SHARED_SIDE_PATH_RENAMES.items()
] + [
    (("bot", "long", "strategy", kind, *old), ("bot", "long", "strategy", kind, *new))
    for kind, fields in STRATEGY_NATIVE_TO_PUBLIC_PATHS.items() for old, new in fields.items()
]


@pytest.mark.parametrize("old,new", PUBLIC_PAIRS, ids=lambda path: ".".join(path))
def test_old_development_spellings_are_not_normal_loader_aliases(old, new):
    source = get_template_config()
    target = source
    for name in new[:-1]:
        target = target[name]
    target[old[-1]] = target.pop(new[-1])
    with pytest.raises(ValueError, match="unknown configuration field"):
        prepare_config(source, verbose=False)


@pytest.mark.parametrize("kind", ["trailing_martingale", "ema_anchor"])
def test_nondefault_fraction_bounds_keep_stable_native_optimizer_ids(kind):
    config = get_template_config()
    config["bot"]["global"]["strategy_kind"] = kind
    keys = ["long_unstuck_close_pct", "long_unstuck_loss_allowance_pct", "long_risk_we_excess_allowance_pct"]
    keys += (["long_entry_initial_qty_pct", "long_entry_initial_ema_dist", "long_close_qty_pct"]
             if kind == "trailing_martingale" else ["long_base_qty_pct", "long_offset"])
    bounds = {}
    for index, key in enumerate(keys):
        value = 0.012345678901234567 + index / 1000
        set_flat_optimize_bound(bounds, kind, key, [value, value])
        path = resolve_optimizer_key_path(config, key)
        assert path is not None and path[-1].endswith("fraction")
        assign(config, path, value)
    flat = flatten_optimize_bounds(bounds, strategy_kind=kind)
    assert set(flat) == set(keys)
    for index, key in enumerate(keys):
        assert flat[key] == [0.012345678901234567 + index / 1000] * 2


def test_fraction_metadata_names_actual_denominators_and_score_gap():
    registry = get_field_registry()
    fields = {field.path: field for field in registry}
    for _, path in PUBLIC_PAIRS:
        field = fields[path]
        if path[-1] == "score_hysteresis_margin":
            assert field.unit == "normalized score gap"
            assert field.denominator is None
        else:
            assert field.unit == "fraction" and field.denominator
    close = fields[("bot", "long", "strategy", "trailing_martingale", "close", "qty_fraction")]
    assert "budget-derived" in close.denominator and "over-budget" in close.description


def test_nondefault_live_fee_and_backtest_policy_lower_without_rescaling():
    from backtest import prep_backtest_args
    from fill_events_manager import fee_policy_kwargs_from_config, _normalize_fee_paid_from_payload

    source = get_template_config()
    source["live"]["fills"].update(fallback_fee_fraction=0.00037, max_abs_fee_fraction=0.0013)
    source["backtest"].update(market_order_slippage_fraction=0.00073, limit_order_fill_buffer_fraction=0.00051)
    source["bot"]["global"]["forager"]["score_hysteresis_margin"] = 0.037
    original = deepcopy(source)
    policy = fee_policy_kwargs_from_config(source)
    assert policy["fee_pct_fallback"] == 0.00037
    assert policy["fee_pct_sanity_abs_max"] == 0.0013
    fee, _ = _normalize_fee_paid_from_payload(
        {"qty": 2.0, "price": 123.0}, fee_pct_fallback=policy["fee_pct_fallback"],
        fee_pct_sanity_abs_max=policy["fee_pct_sanity_abs_max"],
    )
    assert fee == pytest.approx(-246.0 * 0.00037)
    config = prepare_config(source, verbose=False)
    config["backtest"]["coins"] = {"binance": ["BTC"]}
    market = dict(maker=0.0002, taker=0.0005, qty_step=0.001, price_step=0.01,
                  min_qty=0.001, min_cost=1.0, c_mult=1.0)
    *_, payload = prep_backtest_args(config, {"BTC": market}, "binance")
    assert payload["market_order_slippage_pct"] == 0.00073
    assert payload["limit_order_fill_buffer_pct"] == 0.00051
    assert payload["forager_score_hysteresis_pct"] == 0.037
    assert source == original


@pytest.mark.parametrize("selector,expected", [
    ("close_pct", {"long_unstuck_close_pct", "short_unstuck_close_pct"}),
    ("long.unstuck.close_pct", {"long_unstuck_close_pct"}),
    ("bot.*.unstuck.close_pct", {"long_unstuck_close_pct", "short_unstuck_close_pct"}),
    ("initial_qty_pct", {"long_entry_initial_qty_pct", "short_entry_initial_qty_pct"}),
    ("strategy.trailing_martingale.close.qty_pct", {"long_close_qty_pct", "short_close_qty_pct"}),
])
def test_released_fixed_selector_shorthand_retains_exact_native_gene_membership(selector, expected):
    from config.param_paths import resolve_bound_selectors

    source = released_v8_policy()
    source["optimize"]["fixed_params"] = [selector]
    result = migrate(source, memory_span_minutes=720.0)
    flat = flatten_optimize_bounds(result.config["optimize"]["bounds"], strategy_kind="trailing_martingale")
    assert set(resolve_bound_selectors(result.config, result.config["optimize"]["fixed_params"], flat)) == expected


@pytest.mark.parametrize("field", ["market_order_slippage_fraction", "limit_order_fill_buffer_fraction"])
def test_resume_detects_changed_canonical_execution_assumptions(field):
    from optimization.evaluation_contract import build_evaluation_contract
    from optimize import _resume_config_mismatches

    config = get_template_config()
    recorded = deepcopy(config)
    recorded["optimizer_evaluation_contract"] = build_evaluation_contract(config)
    assert _resume_config_mismatches(recorded, config) == []
    config["backtest"][field] = 0.001234567
    assert any(field in mismatch for mismatch in _resume_config_mismatches(recorded, config))


@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("close,allowance", [(None, None), (0.0, 0.0), (0.117, 0.0031)])
def test_trailing_inspection_reads_authored_unstuck_fractions(side, close, allowance):
    from tools.trailing_inspect import _extract_side_context

    source = get_template_config()
    for name, value in (("close_fraction", close), ("loss_allowance_fraction", allowance)):
        if value is None:
            source["bot"][side]["unstuck"].pop(name)
        else:
            source["bot"][side]["unstuck"][name] = value
    original = deepcopy(source)
    report = _extract_side_context(source, side)
    assert report["unstuck_close_pct"] == (close or 0.0)
    assert report["unstuck_loss_allowance_pct"] == (allowance or 0.0)
    assert source == original


def test_preflight_hysteresis_reports_the_authored_fraction_and_canonical_path():
    from tools.live_config_preflight import _balance_hysteresis_report

    issues = []
    assert _balance_hysteresis_report({}, issues)["present"] is False
    report = _balance_hysteresis_report({"account": {"balance_hysteresis_snap_fraction": 0.07}}, issues)
    assert report["present"] is True and report["value"] == 0.07
    assert issues[-1]["path"] == "live.account.balance_hysteresis_snap_fraction"


@pytest.mark.parametrize("stem", ["entry-threshold", "entry-retracement", "close-threshold", "close-retracement"])
def test_trailing_cli_uses_fraction_flags_at_the_native_calculation_boundary(stem):
    from tools.trailing_inspect import build_parser

    parser = build_parser()
    args = parser.parse_args([f"--{stem}-base-fraction", "0.0123456789"])
    assert getattr(args, stem.replace("-", "_") + "_base_pct") == 0.0123456789
    with pytest.raises(SystemExit):
        parser.parse_args([f"--{stem}-base-pct", "0.0123456789"])


@pytest.mark.parametrize("kind", ["trailing_martingale", "ema_anchor"])
def test_released_conversion_projects_root_bounds_pins_scenarios_and_file_precedence(tmp_path, kind):
    source = released_v8_policy()
    source["live"]["strategy_kind"] = kind
    source["bot"]["long"]["unstuck"]["close_pct"] = 0.071
    source["optimize"]["fixed_runtime_overrides"]["bot.long.unstuck.close_pct"] = 0.081
    source["backtest"]["scenarios"] = [{"label": "one", "overrides": {
        "bot.long.unstuck.close_pct": 0.091,
        "coin_overrides.BTC.bot.long.unstuck.close_pct": 0.101,
    }}]
    filename = tmp_path / "coin.json"
    filename.write_text(json.dumps({"bot": {"long": {"unstuck": {"close_pct": 0.111}}}}))
    source["coin_overrides"] = {"BTC": {"override_config_path": filename.name,
        "bot": {"long": {"unstuck": {"close_pct": 0.121}}}}}
    original = deepcopy(source)
    result = migrate(source, base_config_path=str(tmp_path / "input.json"), memory_span_minutes=720.0)
    config = result.config
    assert config["bot"]["long"]["unstuck"]["close_fraction"] == 0.071
    assert config["coin_overrides"]["BTC"]["bot"]["long"]["unstuck"]["close_fraction"] == 0.121
    assert config["optimize"]["fixed_runtime_overrides"]["bot.long.unstuck.close_fraction"] == 0.081
    assert config["backtest"]["scenarios"][0]["overrides"]["bot.long.unstuck.close_fraction"] == 0.091
    flat = flatten_optimize_bounds(config["optimize"]["bounds"], strategy_kind=kind)
    assert "long_unstuck_close_pct" in flat
    assert any("close_pct" in change["path"] and "close_fraction" in str(change["new"])
               for change in result.changes)
    prepare_config(config, verbose=False)
    assert source == original
