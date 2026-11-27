"""The fills report groups saved facts; it never invents prices, positions or PnL."""

import json
import os

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
import pytest

from backtest_reports import (
    ReportOptions,
    add_report_arguments,
    options_from_args,
    parse_plots,
    render_reports,
)
from backtest_results import ResultStore, write_equity
from fills_report import (
    INTERVALS_MS,
    MINUTE_MS,
    CandleSource,
    build_payload,
    candles_cover,
    choose_interval_ms,
    classify_fill_type,
    encode_bars,
    ohlc_bars,
    position_episodes,
    prepare_fills,
    render_fills_html,
    select_detail_episodes,
)

T0 = int(pd.Timestamp("2026-01-01").value // 1_000_000)


def test_every_native_order_type_has_an_intent():
    import passivbot_rust as pbr

    if getattr(pbr, "__is_stub__", False) or not hasattr(pbr, "all_order_types_ids"):
        pytest.skip("requires the compiled extension's order type registry")
    names = [pbr.order_type_id_to_snake(i) for i in pbr.all_order_types_ids()]
    kinds = {name: classify_fill_type(name) for name in names if name != "empty"}
    assert {k.intent for k in kinds.values()} == {"entry", "take_profit", "risk"}
    for name, kind in kinds.items():
        assert kind.action == name.split("_")[0]
        assert kind.pside == name.split("_")[-1]


@pytest.mark.parametrize(
    ("name", "family", "mechanism", "intent"),
    [
        ("entry_initial_partial_long", "initial", "initial_partial", "entry"),
        ("entry_grid_inflated_short", "grid", "grid_inflated", "entry"),
        ("entry_ema_anchor_long", "ema_anchor", "ema_anchor", "entry"),
        ("close_ema_anchor_short", "ema_anchor", "ema_anchor", "take_profit"),
        ("close_trailing_long", "trailing", "trailing", "take_profit"),
        ("close_unstuck_long", "unstuck", "unstuck", "risk"),
        ("close_auto_reduce_wel_short", "auto_reduce", "auto_reduce_wel", "risk"),
        ("close_auto_reduce_twel_long", "auto_reduce", "auto_reduce_twel", "risk"),
        ("close_panic_short", "panic", "panic", "risk"),
    ],
)
def test_order_types_split_into_family_mechanism_and_intent(name, family, mechanism, intent):
    kind = classify_fill_type(name)
    assert (kind.family, kind.mechanism, kind.intent) == (family, mechanism, intent)


@pytest.mark.parametrize("name", ["long_entry", "entry_long", "open_grid_long", "close_grid_both"])
def test_unrecognized_order_types_are_rejected(name):
    with pytest.raises(ValueError, match="Unrecognized fill order type"):
        classify_fill_type(name)


def test_bars_open_at_the_previous_close_and_skip_unpriced_rows():
    ts = T0 + np.arange(8) * MINUTE_MS
    close = np.array([0.0, 10.0, 11.0, 9.0, 12.0, np.nan, 13.0, 14.0])
    high = close + 0.5
    low = close - 0.5
    bars = ohlc_bars(ts, high, low, close, T0, T0 + 7 * MINUTE_MS, 4 * MINUTE_MS)
    # First bucket: rows 1-3 (row 0 is unpriced padding). Its open falls back
    # to its own first close because the preceding close is not positive.
    np.testing.assert_allclose(bars[0], [T0, 10.0, 11.5, 8.5, 9.0])
    # Second bucket: rows 4, 6, 7 (NaN row 5 dropped), opening at row 3's close.
    np.testing.assert_allclose(bars[1], [T0 + 4 * MINUTE_MS, 9.0, 14.5, 11.5, 14.0])


def test_bars_respect_the_requested_window_and_previous_close_outside_it():
    ts = T0 + np.arange(6) * MINUTE_MS
    close = np.array([5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    bars = ohlc_bars(ts, close, close, close, T0 + 2 * MINUTE_MS, T0 + 3 * MINUTE_MS, MINUTE_MS)
    np.testing.assert_allclose(bars[:, 1], [6.0, 7.0])  # opens from the row before the window
    assert ohlc_bars(
        ts, close, close, close, T0 + 99 * MINUTE_MS, T0 + 100 * MINUTE_MS, MINUTE_MS
    ).shape == (0, 5)


def test_encoded_bars_round_trip_clocks_and_keep_gaps():
    bars = np.array(
        [[T0, 1, 2, 0.5, 1.5], [T0 + 3600e3, 1.5, 2, 1, 1.2], [T0 + 3 * 3600e3, 1.2, 1.3, 1.1, 1.25]]
    )
    encoded = encode_bars(bars, 3_600_000)
    slots = np.cumsum(encoded["dt"]) + encoded["t0"] // 3_600_000
    assert (slots * 3_600_000).tolist() == bars[:, 0].astype(int).tolist()
    assert encoded["dt"] == [0, 1, 2]
    assert encode_bars(np.empty((0, 5)), MINUTE_MS)["dt"] == []


def test_interval_choice_bounds_bars_and_never_undercuts_native_candles():
    assert choose_interval_ms(100 * MINUTE_MS, 720) == MINUTE_MS
    assert choose_interval_ms(100 * MINUTE_MS, 720, 5 * MINUTE_MS) == 5 * MINUTE_MS
    assert choose_interval_ms(30 * 86_400_000, 720) == 60 * MINUTE_MS
    assert choose_interval_ms(10**15, 10) == INTERVALS_MS[-1]


def _fill(
    minute, coin, type_, *, price, qty, psize, pprice=None, pnl=0.0, fee=-0.1, we=0.1, balance=1000.0
):
    return {
        "timestamp": pd.Timestamp(T0 + minute * MINUTE_MS, unit="ms"),
        "coin": coin,
        "type": type_,
        "price": price,
        "qty": qty,
        "pnl": pnl,
        "fee_paid": fee,
        "psize": psize,
        "pprice": price if pprice is None else pprice,
        "wallet_exposure": we,
        "usd_total_balance": balance + pnl + fee,
    }


def _ledger():
    rows = [
        _fill(0, "AAA", "close_trailing_long", price=10, qty=-1, psize=0, pnl=5.0),  # prior position
        _fill(1, "AAA", "entry_initial_normal_long", price=10, qty=2, psize=2, we=0.02),
        _fill(1, "AAA", "entry_initial_normal_short", price=10, qty=-3, psize=-3, we=-0.03),
        _fill(2, "AAA", "entry_trailing_normal_long", price=9, qty=2, psize=4, pprice=9.5, we=0.04),
        _fill(
            3, "AAA", "close_unstuck_long", price=9.2, qty=-1, psize=3, pprice=9.5, pnl=-0.3, we=0.03
        ),
        _fill(
            4, "AAA", "close_trailing_long", price=11, qty=-3, psize=0, pprice=9.5, pnl=4.5, we=0.0
        ),
        _fill(
            5,
            "AAA",
            "close_panic_short",
            price=12,
            qty=3,
            psize=0.0005,
            pprice=10,
            pnl=-6.0,
            we=-0.00001,
        ),
        _fill(6, "BBB", "entry_ema_anchor_long", price=50, qty=1, psize=1, we=0.05),
        _fill(7, "BBB", "close_ema_anchor_long", price=51, qty=-0.5, psize=0.5, pnl=0.5, we=0.025),
    ]
    return pd.DataFrame(rows)


def test_episodes_follow_each_side_until_flat_and_keep_open_positions():
    fills = prepare_fills(_ledger())
    episodes = position_episodes(fills, {"AAA": 0.001})
    long_aaa, short_aaa, long_bbb = (
        episodes.loc[(episodes.coin == c) & (episodes.pside == s)].iloc[0]
        for c, s in (("AAA", "long"), ("AAA", "short"), ("BBB", "long"))
    )
    assert len(episodes) == 3  # The leading close of an earlier position is not an episode.
    assert (long_aaa.entries, long_aaa.closes, long_aaa.exit_family) == (2, 2, "trailing")
    assert long_aaa.pnl == pytest.approx(4.2)
    assert long_aaa.max_wallet_exposure == pytest.approx(0.04)
    # Dust below min_qty counts as flat; BBB without settings needs exactly zero.
    assert short_aaa.finished and short_aaa.exit_family == "panic"
    assert not long_bbb.finished and long_bbb.end_ms == T0 + 7 * MINUTE_MS


def test_a_position_of_exactly_min_qty_stays_open():
    fills = prepare_fills(
        pd.DataFrame(
            [
                _fill(0, "AAA", "entry_initial_normal_long", price=10, qty=1, psize=1),
                _fill(1, "AAA", "entry_grid_normal_long", price=9, qty=2, psize=3, pprice=9.33),
                _fill(2, "AAA", "close_grid_long", price=11, qty=-2, psize=1, pnl=3.0),
                _fill(3, "AAA", "close_grid_long", price=12, qty=-1, psize=0, pnl=2.0),
            ]
        )
    )
    episodes = position_episodes(fills, {"AAA": 1.0})
    assert len(episodes) == 1
    ep = episodes.iloc[0]
    assert (ep.entries, ep.closes, ep.finished) == (2, 2, True)
    assert ep.pnl == pytest.approx(5.0)


def test_open_positions_end_at_the_final_observation_clock():
    fills = prepare_fills(_ledger())
    final = T0 + 60 * MINUTE_MS
    episodes = position_episodes(fills, {"AAA": 0.001}, final_ms=final)
    long_bbb = episodes.loc[episodes.coin == "BBB"].iloc[0]
    assert not long_bbb.finished and long_bbb.end_ms == final
    assert (episodes.loc[episodes.finished, "end_ms"] < final).all()


def test_pnl_fraction_uses_the_balance_before_each_fill_and_keeps_saved_order():
    raw = _ledger()
    fills = prepare_fills(raw)
    assert fills["type"].tolist()[1:3] == ["entry_initial_normal_long", "entry_initial_normal_short"]
    panic = fills.loc[fills.type == "close_panic_short"].iloc[0]
    assert panic.pnl_fraction == pytest.approx(-6.0 / 1000.0)
    episodes = position_episodes(fills)
    short = episodes.loc[episodes.pside == "short"].iloc[0]
    assert short.pnl_fraction == pytest.approx(-6.0 / 1000.0)
    assert not short.finished  # without min_qty, 0.0005 dust is still a position


def test_fills_report_requires_position_columns():
    with pytest.raises(ValueError, match="pprice"):
        prepare_fills(_ledger().drop(columns=["pprice"]))


def test_detail_selection_is_bounded_and_unique():
    episodes = pd.DataFrame(
        {
            "pnl_fraction": np.linspace(-0.1, 0.1, 50),
            "start_ms": np.zeros(50, dtype=int),
            "end_ms": np.arange(50) * 1000,
            "max_wallet_exposure": np.linspace(1, 0, 50),
        }
    )
    chosen = select_detail_episodes(episodes, limit=10)
    assert len(chosen) == len(set(chosen)) == 10
    assert {0, 49} <= set(chosen)  # largest |PnL %| on both sides
    assert select_detail_episodes(episodes.iloc[:0]) == []


def test_detail_candles_are_skipped_only_when_a_tier_covers_the_window():
    hour = 60 * MINUTE_MS
    entry = {
        "overview": {"interval_ms": 720 * MINUTE_MS},
        "details": [{"interval_ms": MINUTE_MS, "start_ms": 0, "end_ms": 10 * hour}],
    }
    assert candles_cover(entry, MINUTE_MS, 2 * hour, 3 * hour)
    assert candles_cover(entry, 5 * MINUTE_MS, 0, 10 * hour)
    # A distant episode of the same coin still gets its own fine window.
    assert not candles_cover(entry, MINUTE_MS, 1000 * hour, 1001 * hour)
    assert not candles_cover(entry, MINUTE_MS, 9 * hour, 11 * hour)
    assert candles_cover(entry, 720 * MINUTE_MS, 1000 * hour, 1001 * hour)


def _result(tmp_path, *, coin_label="AAA"):
    ledger = _ledger()
    if coin_label != "AAA":
        ledger["coin"] = ledger["coin"].replace({"AAA": coin_label})
    ledger.to_csv(tmp_path / "fills.csv", index=False)
    clocks = pd.date_range(pd.Timestamp(T0, unit="ms"), periods=10, freq="min")
    write_equity(
        tmp_path,
        pd.DataFrame(
            {
                "strategy_equity": np.linspace(1000, 1010, 10),
                "usd_total_equity": np.linspace(1000, 1010, 10),
                "btc_total_equity": np.linspace(10, 10.1, 10),
                "usd_total_balance": np.full(10, 1000.0),
            },
            index=clocks,
        ),
    )
    (tmp_path / "analysis.json").write_text(
        json.dumps({"drawdown_max_fraction_strategy_equity": 0.1})
    )
    (tmp_path / "market_specific_settings.json").write_text(
        json.dumps({coin_label: {"min_qty": 0.001, "c_mult": 1.0}, "BBB": {"c_mult": 0.01}})
    )
    (tmp_path / "dataset.json").write_text(
        json.dumps(
            {
                "coins": [coin_label, "BBB"],
                "market_specific_settings_file": "market_specific_settings.json",
            }
        )
    )
    ts = T0 - 5 * MINUTE_MS + np.arange(20) * MINUTE_MS
    hlcvs = np.zeros((20, 2, 4))
    hlcvs[:, 0, :3] = np.column_stack([np.full(20, 13.0), np.full(20, 8.0), np.linspace(9, 12, 20)])
    hlcvs[:, 1, :3] = np.column_stack([np.full(20, 52.0), np.full(20, 49.0), np.linspace(50, 51, 20)])
    return ResultStore(tmp_path), CandleSource(ts, hlcvs, {coin_label: 0, "BBB": 1})


def _embedded(html):
    start = html.index("const DATA = ") + len("const DATA = ")
    end = html.index(";\n", start)
    return json.loads(html[start:end])


def test_payload_round_trips_through_html_and_cannot_close_the_script(tmp_path):
    store, candles = _result(tmp_path, coin_label="A</script><b>")
    payload = build_payload(store, candles=candles)
    html = render_fills_html(payload)
    assert html.count("</script>") == 1
    assert _embedded(html) == json.loads(json.dumps(payload))
    assert payload["coins"] == ["A</script><b>", "BBB"]
    assert len(payload["fills"]["t"]) == 9
    assert sorted(payload["episodes"]["pside"]) == ["long", "long", "short"]
    # Flow notionals need the contract multiplier; BBB is still open at the last
    # equity observation.
    assert payload["c_mult"] == {"A</script><b>": 1.0, "BBB": 0.01}
    bbb = payload["coins"].index("BBB")
    open_end = [
        end
        for coin, end, done in zip(
            payload["episodes"]["coin"],
            payload["episodes"]["end_ms"],
            payload["episodes"]["finished"],
        )
        if coin == bbb and not done
    ]
    assert open_end == [T0 + 9 * MINUTE_MS]
    assert payload["seed"] == {}
    overview = payload["candles"]["BBB"]["overview"]
    assert overview["interval_ms"] == MINUTE_MS and overview["bars"]["c"][0] is not None


def test_focused_payload_filters_coins_and_window(tmp_path):
    store, candles = _result(tmp_path)
    payload = build_payload(
        store, candles=candles, coins=("AAA",), start_ms=T0 + 2 * MINUTE_MS, end_ms=T0 + 4 * MINUTE_MS
    )
    assert payload["coins"] == ["AAA"]
    assert payload["fills"]["t"] == [T0 + m * MINUTE_MS for m in (2, 3, 4)]
    assert payload["filters"] == {
        "coins": ["AAA"],
        "start_ms": T0 + 2 * MINUTE_MS,
        "end_ms": T0 + 4 * MINUTE_MS,
    }
    # The long opened before the window: it keeps its full history, and its rows
    # point at the window's fills. The short has no fill inside the window.
    episodes = payload["episodes"]
    assert episodes["pside"] == ["long"] and episodes["entries"] == [2]
    assert episodes["start_ms"] == [T0 + MINUTE_MS]
    assert (episodes["first_row"], episodes["last_row"]) == ([0], [2])
    # Positions already open at the window start seed the price and exposure lines.
    assert payload["seed"] == {"AAA": {"long": [2.0, 10.0, 0.02], "short": [-3.0, 10.0, -0.03]}}
    with pytest.raises(ValueError, match="Unknown coins"):
        build_payload(store, candles=candles, coins=("ZZZ",))


def test_fills_plot_group_writes_html_and_follows_coin_fill_disabling(tmp_path):
    store, candles = _result(tmp_path)
    status = render_reports(store, options=ReportOptions(("fills",)), candles=candles)
    assert status["status"] == "complete" and status["artifacts"][0] == "fills.html"
    html = (tmp_path / "reports" / "fills.html").read_text(encoding="utf-8")
    assert _embedded(html)["coins"] == ["AAA", "BBB"]
    assert "/*__APP__*/" not in html and "renderCoin" in html

    (tmp_path / "reports" / "fills.html").unlink()
    status = render_reports(
        store, options=ReportOptions(("fills",)), candles=candles, disabled={"coin_fills"}
    )
    assert status["artifacts"] == [] and not (tmp_path / "reports" / "fills.html").exists()


def test_report_arguments_parse_fills_focus_options():
    import argparse

    parser = argparse.ArgumentParser()
    add_report_arguments(parser)
    args = parser.parse_args(
        [
            "--plots",
            "equity,fills",
            "--fills-coins",
            "BTC, ETH,BTC",
            "--fills-start",
            "2026-01-01",
            "--fills-end",
            "2026-01-02T12:00",
        ]
    )
    options = options_from_args(args)
    assert options.plots == ("equity", "fills")
    assert options.fills_coins == ("BTC", "ETH")
    assert options.fills_start_ms == T0
    assert options.fills_end_ms == T0 + 36 * 3_600_000
    assert "fills" in parse_plots("all")
    with pytest.raises(ValueError, match="start"):
        ReportOptions(("fills",), fills_start_ms=2, fills_end_ms=1)


def test_candle_source_rejects_misaligned_clocks():
    with pytest.raises(ValueError, match="different lengths"):
        CandleSource(np.arange(3), np.zeros((4, 1, 4)), {"AAA": 0})
    with pytest.raises(ValueError, match="strictly increasing"):
        CandleSource(np.array([0, 2, 1]), np.zeros((3, 1, 4)), {"AAA": 0})
