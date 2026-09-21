"""Canonical policy transport into the real native simulator; no network I/O."""
from copy import deepcopy

import numpy as np
import pytest

from backtest import prep_backtest_args, build_backtest_payload
from config import prepare_config
from config.hsl_revised import generated_template
from config.schema import get_template_config


@pytest.fixture(autouse=True)
def real_extension():
    import passivbot_rust
    assert not getattr(passivbot_rust, "__is_stub__", False)


def inputs(mode="coin"):
    cfg = generated_template(get_template_config(), mode)
    cfg["live"].update(approved_coins={"long": ["AAA"], "short": []},
                       ignored_coins={"long": [], "short": []}, pnls_max_lookback_days=1)
    cfg["backtest"]["coins"] = {"binance": ["AAA"]}
    cfg["backtest"]["starting_balance"] = 1000.0
    cfg["backtest"]["btc_collateral_cap"] = 0.0
    for side in ("long", "short"):
        cfg["bot"][side]["hsl"].update(enabled=False, restart_after_red_policy=None)
        cfg["bot"][side]["risk"].update(n_positions=1, total_wallet_exposure_limit=1)
    active = cfg["bot"]["hsl"] if mode == "unified" else cfg["bot"]["long"]["hsl"]
    active.update(enabled=True, red_threshold=.01, ema_span_minutes=1.5,
                  cooldown_minutes_after_red=10, restart_after_red_policy="always",
                  panic_close_order_type="market")
    cfg = prepare_config(cfg, verbose=False, target="canonical", runtime=None)
    cfg["backtest"]["coins"] = {"binance": ["AAA"]}
    n = 160
    marks = np.concatenate([np.full(40,100.), np.linspace(100,30,n-40)])
    candles = np.array([[[p*1.01,p*.99,p,1000.]] for p in marks])
    mss = {"AAA": dict(qty_step=.001, price_step=.01, min_qty=.001, min_cost=1.,
                       c_mult=1., maker=.0002, taker=.0005, exchange="binance")}
    return cfg, mss, candles


def payload(mode="coin", cfg=None, mss=None):
    base, markets, candles = inputs(mode)
    bot, strategy, exchange, params = prep_backtest_args(cfg or base, mss or markets, "binance")
    params.update(first_timestamp_ms=1704067200000, requested_start_timestamp_ms=1704067200000,
                  first_valid_indices=[0], last_valid_indices=[len(candles)-1],
                  warmup_minutes=[1], trade_start_indices=[1], global_warmup_bars=1,
                  candle_interval_minutes=1)
    return candles, np.full(len(candles), 50000.), bot, strategy, exchange, params


def run(args):
    import passivbot_rust
    return passivbot_rust.run_backtest(*args)


@pytest.mark.parametrize("mode", ["coin", "pside", "unified"])
def test_native_revised_transport_and_report(mode):
    args = payload(mode)
    hsl = args[-1]["equity_hard_stop_loss"]
    selected = hsl["portfolio"] if mode == "unified" else hsl["sides"][0]
    assert selected["ema_span_minutes"] == 1.5
    assert hsl["sides"][1]["restart_after_red_policy"] is None
    assert not any(key.startswith("hsl_") for pair in args[2] for side in pair.values() for key in side)
    full = run(args)
    report = full[4]["revised"]
    assert report["mode"] == mode
    assert report["summary"]["triggers"] > 0
    assert report["summary"]["panic_close_fills"] > 0
    assert any("panic" in str(fill[13]) for fill in full[0])
    assert "hard_stop_time_in_yellow_pct" not in full[2]
    assert "hard_stop_time_in_orange_pct" not in full[2]
    assert full[2]["hard_stop_triggers"] == report["summary"]["triggers"]
    assert full[2]["hard_stop_panic_close_loss_sum"] == report["summary"]["panic_close_loss"]
    assert full[2]["drawdown_worst_strategy_eq"] > 0.0
    assert full[2]["drawdown_worst_ema_strategy_eq"] > 0.0
    assert "drawdown_worst" in full[2]
    compact_args = deepcopy(args)
    compact_args[-1]["metrics_only"] = True
    compact = run(compact_args)
    assert compact[4]["revised"]["summary"] == report["summary"]
    assert compact[4]["revised"]["samples"] == []
    assert compact[4]["revised"]["events"] == []
    assert compact[2] == full[2]


def test_effective_coin_override_has_single_authority():
    cfg, mss, _ = inputs()
    cfg["coin_overrides"] = {"AAA": {"bot": {"long": {"hsl": {
        "ema_span_minutes": 2.75, "red_threshold": .4, "panic_close_order_type": "limit"}}}}}
    args = payload(cfg=cfg, mss=mss)
    hsl = args[-1]["equity_hard_stop_loss"]
    assert hsl["sides"][0]["ema_span_minutes"] == 1.5
    assert hsl["coins"]["AAA"][0]["ema_span_minutes"] == 2.75
    assert hsl["coins"]["AAA"][0]["panic_close_order_type"] == "limit"


@pytest.mark.parametrize("change,match", [
    (lambda p: p.update(engine="unknown"), "engine"),
    (lambda p: p.update(mode="unknown"), "signal mode"),
    (lambda p: p.update(intervention="manual"), "intervention"),
    (lambda p: p.update(tier_ratios={}), "unknown revised HSL"),
    (lambda p: p.update(sides=[]), "exactly long and short"),
    (lambda p: p["coins"]["AAA"][0].update(restart_after_red_policy=None), "explicit"),
    (lambda p: p["coins"]["AAA"][0].update(ema_span_minutes=float("nan")), "numeric"),
    (lambda p: p["coins"]["AAA"][0].update(panic_close_order_type="bad"), "order type"),
    (lambda p: p["coins"].update(UNKNOWN=deepcopy(p["sides"])), "outside the dataset"),
])
def test_native_rejects_invalid_config_before_simulation(change, match):
    args = payload()
    change(args[-1]["equity_hard_stop_loss"])
    with pytest.raises(ValueError, match=match):
        run(args)


@pytest.mark.parametrize("key,value,match", [("pnls_max_lookback_days", 0, "lookback"),
                                             ("candle_interval_minutes", 15, "1m")])
def test_native_requires_enabled_scope_temporal_contract(key, value, match):
    args = payload()
    args[-1][key] = value
    with pytest.raises(ValueError, match=match):
        run(args)


def test_unified_never_infers_policy_from_sides():
    args = payload("unified")
    args[-1]["equity_hard_stop_loss"]["portfolio"] = None
    with pytest.raises(ValueError, match="explicit revised portfolio"):
        run(args)


def test_duplicate_bot_hsl_transport_is_rejected():
    args = payload()
    args[2][0]["long"]["hsl_enabled"] = False
    with pytest.raises(ValueError, match="belong only"):
        run(args)


def test_public_activation_stays_gated():
    cfg, mss, candles = inputs()
    with pytest.raises(ValueError, match="runtime integration is not available"):
        build_backtest_payload(candles, mss, cfg, "binance", np.full(len(candles), 50000.))


def test_coin_override_controls_actual_execution():
    args = payload()
    args[-1]["equity_hard_stop_loss"]["coins"]["AAA"][0]["enabled"] = False
    result = run(args)
    assert result[4]["revised"]["summary"]["triggers"] == 0
    assert result[4]["revised"]["summary"]["panic_close_fills"] == 0
    assert not any("panic" in str(fill[13]) for fill in result[0])


def test_unified_ignores_conflicting_inactive_side_policies():
    args = payload("unified")
    expected = run(args)
    for policy in args[-1]["equity_hard_stop_loss"]["sides"]:
        policy.update(enabled=True, restart_after_red_policy="threshold", red_threshold=.9)
    actual = run(args)
    np.testing.assert_array_equal(actual[0], expected[0])
    assert actual[4]["revised"] == expected[4]["revised"]


def test_disabled_scopes_keep_null_restart_without_policy_hydration():
    args = payload("pside")
    for policy in args[-1]["equity_hard_stop_loss"]["sides"]:
        policy.update(enabled=False, restart_after_red_policy=None)
    args[-1]["pnls_max_lookback_days"] = 0
    result = run(args)
    assert result[4]["revised"]["summary"]["triggers"] == 0
    assert result[4]["revised"]["samples"] == []


def test_cached_payload_cannot_change_selected_engine_or_policy():
    cfg, mss, _ = inputs()
    args = payload()
    cached = deepcopy(args[-1])
    cached["equity_hard_stop_loss"]["coins"]["AAA"][0]["red_threshold"] = .9
    with pytest.raises(ValueError, match="effective config"):
        prep_backtest_args(cfg, mss, "binance", backtest_params=cached)
    cached["equity_hard_stop_loss"]["engine"] = "legacy"
    with pytest.raises(ValueError, match="engine differs"):
        prep_backtest_args(cfg, mss, "binance", backtest_params=cached)


def test_revised_artifact_never_substitutes_btc_collateral_equity_for_strategy():
    from backtest import process_forager_fills

    args = list(payload("unified"))
    args[-1]["btc_collateral_cap"] = 0.5
    args[1] = np.linspace(50000.0, 90000.0, len(args[0]))
    result = run(args)
    assert result[1].shape[1] == 4
    assert np.ptp(result[1][:, 1]) > 1.0
    _, _, frame = process_forager_fills(
        result[0], args[-1]["coins"], args[0], result[1], balance_sample_divider=1
    )
    expected = []
    for timestamp, *_ in result[1]:
        k = int((timestamp - args[-1]["first_timestamp_ms"]) / 60000)
        fills = [row for row in result[0] if row[1] <= timestamp]
        realized = sum(float(row[3]) + float(row[4]) for row in fills)
        size, basis = (float(fills[-1][11]), float(fills[-1][12])) if fills else (0., 0.)
        expected.append(1000. + realized + size * (args[0][k, 0, 2] - basis))
    np.testing.assert_allclose(result[1][:, 3], expected, rtol=1e-12, atol=1e-9)
    assert not np.allclose(result[1][:, 1], result[1][:, 3])
    assert frame["strategy_equity"].notna().all()


def test_strategy_equity_statistics_exclude_btc_collateral_even_without_fills():
    args = list(payload("unified"))
    args[-1]["btc_collateral_cap"] = .5
    args[1] = np.linspace(50000., 90000., len(args[0]))
    args[0][:] = [100., 100., 100., 1000.]
    result = run(args)
    assert len(result[0]) == 0
    assert result[1][-1, 1] > 1300.
    np.testing.assert_allclose(result[1][:, 3], 1000.)
    assert result[2]["gain_strategy_eq"] == 0.
    assert result[2]["drawdown_worst_strategy_eq"] == 0.
    assert result[2]["drawdown_worst_strategy_eq_long"] == 0.
    assert result[2]["drawdown_worst_strategy_eq_short"] == 0.


def test_general_strategy_metrics_survive_disabled_hsl():
    args = payload()
    for policy in args[-1]["equity_hard_stop_loss"]["coins"]["AAA"]:
        policy.update(enabled=False, restart_after_red_policy=None)
    result = run(args)
    assert result[2]["hard_stop_triggers"] == 0
    assert result[2]["drawdown_worst_strategy_eq"] > 0
    assert result[2]["drawdown_worst_strategy_eq_long"] > 0
    assert result[2]["drawdown_worst_strategy_eq_short"] == 0
    assert result[1].shape[1] == 4


def prepared_payload(args):
    from backtest import BacktestPayload, _build_hlcvs_bundle
    params = args[-1]
    timestamps = params["first_timestamp_ms"] + np.arange(len(args[0]), dtype=np.int64) * 60000
    mss = {coin: dict(qty_step=.001,price_step=.01,min_qty=.001,min_cost=1.,c_mult=1.)
           for coin in params["coins"]}
    bundle = _build_hlcvs_bundle(args[0], args[1], timestamps, "binance", mss,
        params["coins"], params["first_valid_indices"], params["last_valid_indices"],
        params["warmup_minutes"], params["trade_start_indices"], {
            "requested_start_timestamp_ms": params["first_timestamp_ms"],
            "effective_start_timestamp_ms": params["first_timestamp_ms"],
            "warmup_minutes_requested": 1, "warmup_minutes_provided": 1})
    return BacktestPayload(bundle, args[2], args[3], args[4], params)


@pytest.mark.parametrize("compact", [False, True])
def test_execute_backtest_preserves_revised_report_and_shared_metrics(compact):
    from backtest import execute_backtest
    cfg, _, _ = inputs("unified")
    args = payload("unified")
    args[-1]["metrics_only"] = compact
    prepared = prepared_payload(args)
    _, _, analysis = execute_backtest(prepared, cfg)
    assert analysis["hard_stop_triggers"] > 0
    assert "hard_stop_triggers_usd" not in analysis
    assert not any("time_in_orange" in key or "time_in_yellow" in key for key in analysis)
    assert prepared.hard_stop_plot_data["revised"]["summary"]["triggers"] == analysis["hard_stop_triggers"]


def test_dataset_subset_preserves_only_selected_coin_policies():
    from backtest import subset_backtest_payload, execute_backtest
    args = list(payload())
    args[0] = np.repeat(args[0], 2, axis=1)
    for index in (2,3,4):
        args[index] = [deepcopy(args[index][0]), deepcopy(args[index][0])]
    params = args[-1]
    params["coins"] = ["AAA", "BBB"]
    for key in ("first_valid_indices", "last_valid_indices", "warmup_minutes", "trade_start_indices"):
        params[key] *= 2
    hsl = params["equity_hard_stop_loss"]
    hsl["coins"]["BBB"] = deepcopy(hsl["coins"]["AAA"])
    prepared = prepared_payload(args)
    subset = subset_backtest_payload(prepared, coin_indices=[1])
    assert set(subset.backtest_params["equity_hard_stop_loss"]["coins"]) == {"BBB"}
    assert set(prepared.backtest_params["equity_hard_stop_loss"]["coins"]) == {"AAA", "BBB"}
    cfg, _, _ = inputs()
    _, _, analysis = execute_backtest(subset, cfg)
    assert analysis["hard_stop_triggers"] > 0


@pytest.mark.parametrize("btc_cap", [0.0, 0.5])
def test_liquidation_keeps_strategy_observations_aligned_with_equities(btc_cap):
    args = list(payload())
    args[-1]["btc_collateral_cap"] = btc_cap
    args[1] = np.full(len(args[0]), 50000.)
    args[1][70:] = 25000.
    # Gap through the floor rather than landing exactly on it.
    args[0][70:] *= np.array([.5, .5, .5, 1.])
    for policy in args[-1]["equity_hard_stop_loss"]["coins"]["AAA"]:
        policy["enabled"] = False
    args[-1]["liquidation_threshold"] = .99
    result = run(args)
    assert result[2]["liquidated"]
    assert 0 < len(result[1]) < len(args[0]) - 3
    assert result[1].shape[1] == 4
    assert np.isfinite(result[1][:, 3]).all()
    assert result[2]["drawdown_worst_strategy_eq"] > 0
    timestamp, account_equity, _, strategy_equity = result[1][-1]
    fills = [row for row in result[0] if row[1] <= timestamp]
    assert fills
    k = int((timestamp - args[-1]["first_timestamp_ms"]) / 60000)
    realized = sum(float(row[3]) + float(row[4]) for row in fills)
    size, basis = float(fills[-1][11]), float(fills[-1][12])
    expected = 1000. + realized + size * (args[0][k, 0, 2] - basis)
    assert account_equity == pytest.approx(990.)
    assert strategy_equity == pytest.approx(expected, abs=1e-9)
    assert strategy_equity != pytest.approx(account_equity)
