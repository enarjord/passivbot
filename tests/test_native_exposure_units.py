"""Exposure unit names are coordinated across native boundaries and released inputs."""

from copy import deepcopy

import pytest

from config import prepare_config
from config.optimize_bounds import flatten_optimize_bounds
from config.param_paths import resolve_optimizer_key_path
from config.shared_bot import flatten_shared_bot_side
from released_v8_fixtures import released_v8_policy
from test_orchestrator_json_api import compute, make_input, make_symbol
from tools.migrate_config_v9 import migrate


RENAMES = [
    ("wallet_exposure_limit", "wallet_exposure_limit_ratio"),
    ("total_wallet_exposure_limit", "total_wallet_exposure_limit_ratio"),
    ("risk_we_excess_allowance_pct", "risk_we_excess_allowance_fraction"),
    ("risk_wel_enforcer_threshold", "risk_wel_enforcer_threshold_ratio"),
    ("risk_twel_enforcer_threshold", "risk_twel_enforcer_threshold_ratio"),
]


@pytest.fixture(autouse=True)
def require_real_native_extension():
    import passivbot_rust as pbr

    assert not getattr(pbr, "__is_stub__", False)


@pytest.mark.parametrize("old,new", RENAMES)
@pytest.mark.parametrize("location", ["global", "symbol"])
@pytest.mark.parametrize("side", ["long", "short"])
def test_native_bot_packet_rejects_old_names_even_with_complete_current_values(
    old, new, location, side,
):
    import passivbot_rust as pbr

    packet = make_input(balance=1000.0, symbols=[make_symbol(0, bid=100.0, ask=100.0)])
    params = (packet["global"]["global_bot_params"][side] if location == "global"
              else packet["symbols"][0][side]["bot_params"])
    assert new in params
    params[old] = params[new]
    with pytest.raises(ValueError, match=rf"unknown field `{old}`"):
        compute(pbr, packet)


def test_twel_binding_threshold_is_exposure_ratio_and_not_hsl_red_threshold():
    import passivbot_rust as pbr

    args = dict(side="long", exposure_threshold_ratio=0.8,
                total_wallet_exposure_limit_ratio=1.5, effective_n_positions=1,
                balance=1000.0, positions=[])
    assert pbr.calc_twel_enforcer_orders_py(**args) == []
    old = dict(args)
    old["red_threshold"] = old.pop("exposure_threshold_ratio")
    with pytest.raises(TypeError, match="red_threshold"):
        pbr.calc_twel_enforcer_orders_py(**old)


def test_inspector_requires_complete_exposure_ratio_option_name():
    from tools.trailing_inspect import build_parser

    parser = build_parser()
    assert parser.parse_args([
        "--effective-wallet-exposure-limit-ratio", "0.625",
    ]).effective_wallet_exposure_limit_ratio == 0.625
    with pytest.raises(SystemExit):
        parser.parse_args(["--effective-wallet-exposure-limit", "0.625"])


@pytest.mark.parametrize("old,new", RENAMES)
@pytest.mark.parametrize("kind", ["trailing_martingale", "ema_anchor"])
def test_cpu_backtest_packet_requires_current_exposure_names_without_fallback(old, new, kind):
    import passivbot_rust as pbr
    from backtest import build_backtest_payload
    from tools.gpu_parity import build_parser, fixture_inputs

    args = build_parser().parse_args([
        "--fixture", kind, "--sides", "both", "--hsl", "disabled",
        "--coins", "2", "--bars", "65",
    ])
    config, candles, markets, btc, timestamps = fixture_inputs(args)
    payload = build_backtest_payload(candles, markets, config, args.exchange, btc, timestamps)
    pair = deepcopy(payload.bot_params_list)
    pair[0]["long"][old] = pair[0]["long"].pop(new)
    with pytest.raises((KeyError, ValueError), match=new):
        pbr.run_backtest_bundle(payload.bundle, pair, payload.strategy_params_list,
                               payload.exchange_params, payload.backtest_params)


RELEASED = [
    ("total_wallet_exposure_limit", "total_wallet_exposure_limit",
     ("total_wallet_exposure_limit_ratio",), "total_wallet_exposure_limit_ratio", 2.375),
    ("risk_we_excess_allowance_pct", "we_excess_allowance_pct",
     ("we_excess_allowance_fraction",), "risk_we_excess_allowance_fraction", 0.375),
    ("risk_wel_enforcer_threshold", "position_exposure_enforcer_threshold",
     ("position_exposure_enforcer", "threshold_ratio"), "risk_wel_enforcer_threshold_ratio", 0.8125),
    ("risk_twel_enforcer_threshold", "total_exposure_enforcer_threshold",
     ("total_exposure_enforcer", "threshold_ratio"), "risk_twel_enforcer_threshold_ratio", 0.9375),
]


@pytest.mark.parametrize("old,leaf,path,native,value", RELEASED)
@pytest.mark.parametrize("layout", ["grouped", "flat", "mixed"])
@pytest.mark.parametrize("fixed", [False, True])
def test_released_exposure_inputs_and_selectors_emit_final_public_and_native_names(
    old, leaf, path, native, value, layout, fixed,
):
    source = released_v8_policy()
    risk = source["bot"]["long"]["risk"]
    bounds = [0.125, 3.0, 0.125]
    if layout == "flat":
        risk.pop(leaf)
        source["bot"]["long"][old] = value
    else:
        risk[leaf] = value
    source["optimize"]["bounds"] = {}
    if layout == "grouped":
        source["optimize"]["bounds"] = {"long": {"risk": {leaf: bounds}}}
    else:
        source["optimize"]["bounds"][f"long_{old}"] = bounds
    selector = f"bot.long.{old}" if layout != "grouped" else f"bot.long.risk.{leaf}"
    source["optimize"]["fixed_params"] = [selector] if fixed else []
    source["optimize"]["fixed_runtime_overrides"] = {selector: value}
    source["backtest"]["scenarios"] = [{"label": "one", "overrides": {selector: value}}]
    original = deepcopy(source)

    result = migrate(source, memory_span_minutes=123.5)
    config = prepare_config(result.config, verbose=False)
    target = "bot.long.risk." + ".".join(path)
    assert source == original
    assert flatten_shared_bot_side(config["bot"]["long"])[native] == value
    assert old not in flatten_shared_bot_side(config["bot"]["long"])
    assert config["optimize"]["fixed_params"] == ([target] if fixed else [])
    assert config["optimize"]["fixed_runtime_overrides"] == {target: value}
    assert config["backtest"]["scenarios"][0]["overrides"] == {target: value}
    assert flatten_optimize_bounds(
        config["optimize"]["bounds"], strategy_kind="trailing_martingale",
    )[f"long_{native}"] == ([value, value, bounds[2]] if fixed else bounds)
    assert resolve_optimizer_key_path(config, f"long_{native}") == tuple(target.split("."))
    reports = [row for row in result.changes if row["path"] == "optimize.fixed_params"]
    if reports:
        assert reports[-1]["new"] == [target]


@pytest.mark.parametrize("owner,old", [
    ("position_exposure_enforcer", "risk_wel_enforcer_threshold"),
    ("total_exposure_enforcer", "risk_twel_enforcer_threshold"),
])
@pytest.mark.parametrize("equal", [False, True])
def test_released_grouped_and_flat_enforcer_aliases_share_one_default_and_conflict_check(
    owner, old, equal,
):
    source = released_v8_policy()
    source["bot"]["long"]["risk"][f"{owner}_threshold"] = 0.8125
    source["bot"]["long"][old] = 0.8125 if equal else 0.9375
    original = deepcopy(source)
    if equal:
        config = migrate(source, memory_span_minutes=123.5).config
        assert config["bot"]["long"]["risk"][owner]["threshold_ratio"] == 0.8125
    else:
        with pytest.raises(ValueError, match="conflicts"):
            migrate(source, memory_span_minutes=123.5)
    assert source == original
