from types import SimpleNamespace

import numpy as np
import pytest

from optimization.gpu.model import (
    EMA_ANCHOR_MULTICOIN_PARAM_KEYS,
    EMA_ANCHOR_PARAM_KEYS,
    TRAILING_MARTINGALE_COIN_OVERRIDE_PATHS,
    TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS,
    TRAILING_MARTINGALE_PARAM_KEYS,
    flatten_trailing_martingale_params,
    gpu_side_enabled,
)
from optimization.gpu.service import (
    CORE_OUTPUT_KEYS,
    DIRECTIONAL_HSL_OUTPUT_KEYS,
    MpsEmaAnchorProxy,
    MpsSingleCoinProxy,
    MpsMulticoinEmaProxy,
    _build_multicoin_ema_coin_overrides,
    _build_multicoin_tm_coin_overrides,
    _candidate_wallet_exposure_limit_outputs,
    _candidate_position_slot_outputs,
    _combine_hedged_multicoin_outputs,
    _directional_entry_initial_metrics,
    _hsl_params,
    _position_exposure_enforcer_params,
    _prepared_single_coin_side_enabled,
    _require_multicoin_metric_topology,
    _require_complete_valid_tail,
    _require_no_internal_invalid_hsl_candles,
    _single_coin_exposure_params,
    _total_exposure_enforcer_params,
    _unstuck_params,
)


def test_core_output_contract_retains_gross_pnl_aggregates():
    assert {
        "profit_sum",
        "loss_sum",
        "position_unchanged_max_ms",
        "entry_initial_balance_pct",
        "entry_initial_balance_pct_long",
        "entry_initial_balance_pct_short",
        "total_wallet_exposure_max",
        "total_wallet_exposure_mean",
        "day_fill_count",
        "fill_count",
        "fill_count_entry",
        "fill_count_long",
        "fills_active_days_count",
        "coin_fill_counts",
        "pnl_recovery_max_ms",
        "account_recovery_max_ms",
    } <= CORE_OUTPUT_KEYS


@pytest.mark.parametrize(
    "metric",
    [
        "adg_pnl",
        "adg_pnl_w",
        "fills_analysis_duration_days",
        "fills_active_days_count",
        "fills_active_days_ratio",
        "fills_active_symbols_count",
        "fills_count",
        "fills_count_close",
        "fills_count_entry",
        "fills_count_long",
        "fills_count_short",
        "fills_entry_per_close",
        "fills_per_day",
        "fills_per_day_close",
        "fills_per_day_entry",
        "fills_per_day_long",
        "fills_per_day_per_position_slot",
        "fills_per_day_per_position_slot_long",
        "fills_per_day_per_position_slot_short",
        "fills_per_day_short",
        "fills_top_symbol_share",
        "mdg_pnl",
        "mdg_pnl_w",
        "peak_recovery_days_equity_usd",
        "peak_recovery_hours_equity_usd",
        "peak_recovery_days_pnl",
        "peak_recovery_hours_pnl",
        "sharpe_ratio_pnl",
        "sharpe_ratio_pnl_w",
        "sortino_ratio_pnl",
        "sortino_ratio_pnl_w",
    ],
)
def test_dual_side_multicoin_intraday_cutoff_metrics_fail_closed(metric):
    with pytest.raises(ValueError, match="shared-liquidation cutoff"):
        _require_multicoin_metric_topology(["long", "short"], {metric})

    _require_multicoin_metric_topology(["long"], {metric})
    _require_multicoin_metric_topology(["long", "short"], {"adg_strategy_eq"})


@pytest.mark.parametrize("side", ["long", "short"])
def test_directional_entry_initial_metrics_preserve_candidate_batch_shape(side):
    torch = pytest.importorskip("torch")
    values = torch.tensor([0.1, 0.2, 0.3])

    metrics = _directional_entry_initial_metrics(side, values)

    assert metrics[f"entry_initial_balance_pct_{side}"].tolist() == values.tolist()
    other = "short" if side == "long" else "long"
    assert metrics[f"entry_initial_balance_pct_{other}"].shape == values.shape
    assert metrics[f"entry_initial_balance_pct_{other}"].tolist() == [0.0, 0.0, 0.0]


def test_candidate_wallet_exposure_limits_preserve_sides_and_base_fallback():
    torch = pytest.importorskip("torch")

    outputs = _candidate_wallet_exposure_limit_outputs(
        [
            {"long_total_wallet_exposure_limit": 1.25},
            {
                "long_total_wallet_exposure_limit": 1.5,
                "short_total_wallet_exposure_limit": 0.75,
            },
        ],
        {"long": 1.0, "short": 0.5},
        torch=torch,
    )

    assert outputs["candidate_total_wallet_exposure_limit_long"].tolist() == [
        1.25,
        1.5,
    ]
    assert outputs["candidate_total_wallet_exposure_limit_short"].tolist() == [
        0.5,
        0.75,
    ]


def test_candidate_position_slots_follow_candidate_positions_and_enabledness():
    torch = pytest.importorskip("torch")

    outputs = _candidate_position_slot_outputs(
        [
            {"long_n_positions": 3.0, "short_total_wallet_exposure_limit": 0.0},
            {"short_n_positions": 2.0},
        ],
        {"long": 2.0, "short": 1.0},
        {"long": 1.0, "short": 0.5},
        torch=torch,
    )

    assert outputs["position_slots_long"].tolist() == [3.0, 2.0]
    assert outputs["position_slots_short"].tolist() == [0.0, 2.0]


def test_single_coin_side_eligibility_uses_prepared_coin_payload():
    config = {
        "bot": {
            "long": {"risk": {"total_wallet_exposure_limit": 1.0, "n_positions": 1}}
        },
        "live": {"approved_coins": {"long": ["BTC"]}},
    }

    assert _prepared_single_coin_side_enabled(
        config, "long", {"entry_eligible": True}
    )
    assert not _prepared_single_coin_side_enabled(
        config, "long", {"entry_eligible": False}
    )


def test_single_coin_side_eligibility_requires_canonical_payload_flag():
    with pytest.raises(ValueError, match="entry_eligible"):
        _prepared_single_coin_side_enabled({}, "short", {})


def test_directional_hsl_output_contract_retains_lifecycle_and_panic_scalars():
    assert {
        "hsl_triggers_long",
        "hsl_duration_sum_steps",
        "hsl_restart_retrigger_count",
        "hsl_halt_to_restart_equity_loss",
        "hsl_panic_close_loss_sum",
        "hsl_panic_loss_drawdown_count",
    } <= DIRECTIONAL_HSL_OUTPUT_KEYS
    assert len(DIRECTIONAL_HSL_OUTPUT_KEYS) == 25


def test_single_coin_proxy_preserves_directional_hsl_outputs_for_reduction():
    torch = pytest.importorskip("torch")
    proxy = MpsSingleCoinProxy.__new__(MpsSingleCoinProxy)
    proxy.batch_size = 1
    proxy._torch = torch
    proxy.profile_enabled = False
    proxy.metrics_data = {"ts0": 0.0}
    proxy.run = SimpleNamespace()
    proxy.needed_metrics = {"hard_stop_panic_close_loss_sum"}
    proxy._parameter_matrix = lambda candidates: np.zeros((len(candidates), 0))
    raw = {
        key: torch.zeros(1)
        for key in (
            "first_fill_ts",
            "last_fill_ts",
            "last_high_ts",
            "first_eq_ts",
            "last_eq_ts",
        )
    }
    raw["hsl_triggers_long"] = torch.tensor([2.0])
    raw["hsl_panic_close_loss_sum"] = torch.tensor([37.5])
    proxy.runner = SimpleNamespace(run=lambda *args, **kwargs: raw)

    def reduce(output, *args, **kwargs):
        assert output["hsl_triggers_long"].item() == 2.0
        assert output["hsl_panic_close_loss_sum"].item() == 37.5
        return {
            "hard_stop_panic_close_loss_sum": output[
                "hsl_panic_close_loss_sum"
            ]
        }

    proxy._compute_objectives = reduce

    assert proxy.evaluate([{}]) == [{"hard_stop_panic_close_loss_sum": 37.5}]


def test_gpu_proxy_requires_complete_valid_tail():
    _require_complete_valid_tail(99, 100)

    with pytest.raises(ValueError, match="force-realizes open positions"):
        _require_complete_valid_tail(98, 100)


def test_gpu_hsl_requires_contiguous_valid_candles():
    high = np.array([100.0, np.nan, 100.0])
    low = np.array([99.0, np.nan, 99.0])
    close = np.array([99.5, np.nan, 99.5])

    with pytest.raises(ValueError, match="invalid candle at 1"):
        _require_no_internal_invalid_hsl_candles(
            high, low, close, first_valid_idx=0, last_valid_idx=2
        )

    _require_no_internal_invalid_hsl_candles(
        high, low, close, first_valid_idx=2, last_valid_idx=2
    )


@pytest.mark.parametrize(
    ("mode", "legacy_raw"), [("bounded", 0.0), ("legacy_raw", 1.0)]
)
def test_single_coin_exposure_policy_packs_rust_inputs(mode, legacy_raw):
    packed = _single_coin_exposure_params(
        {
            "we_excess_allowance_pct": 0.25,
            "we_excess_allowance_mode": mode,
            "total_exposure_entry_gate_enabled": False,
            "total_exposure_enforcer_threshold": 0.8,
        },
        side="long",
    )

    assert packed == {
        "we_excess_allowance_pct": 0.25,
        "we_excess_allowance_legacy_raw": legacy_raw,
        "twel_entry_gate_enabled": 0.0,
        "twel_enforcer_threshold": 0.8,
    }


def test_single_coin_exposure_policy_rejects_unknown_allowance_mode():
    with pytest.raises(ValueError, match="we_excess_allowance_mode"):
        _single_coin_exposure_params(
            {"we_excess_allowance_mode": "raw"}, side="short"
        )


def test_tm_position_exposure_repair_packs_exact_rust_inputs():
    assert _position_exposure_enforcer_params(
        {
            "position_exposure_enforcer_enabled": True,
            "position_exposure_enforcer_threshold": 0.8,
        },
        side="short",
    ) == {
        "wel_enforcer_enabled": 1.0,
        "wel_enforcer_threshold": 0.8,
    }

    with pytest.raises(ValueError, match="finite positive"):
        _position_exposure_enforcer_params(
            {
                "position_exposure_enforcer_enabled": True,
                "position_exposure_enforcer_threshold": 0.0,
            },
            side="long",
        )


@pytest.mark.parametrize(
    ("policy", "reduce_portfolio"),
    [("reduce_overweight", 0.0), ("reduce_portfolio", 1.0)],
)
def test_tm_total_exposure_repair_packs_exact_rust_inputs(
    policy, reduce_portfolio
):
    assert _total_exposure_enforcer_params(
        {
            "total_exposure_enforcer_enabled": True,
            "total_exposure_enforcer_policy": policy,
        },
        side="long",
    ) == {
        "twel_enforcer_enabled": 1.0,
        "twel_enforcer_reduce_portfolio": reduce_portfolio,
    }

    with pytest.raises(ValueError, match="total_exposure_enforcer_policy"):
        _total_exposure_enforcer_params(
            {"total_exposure_enforcer_policy": "largest_loss"},
            side="short",
        )


def test_single_coin_unstuck_packs_exact_rust_inputs():
    assert _unstuck_params(
        {
            "unstuck_enabled": True,
            "unstuck_ema_gating_enabled": False,
            "unstuck_close_pct": 0.125,
            "unstuck_ema_dist": -0.01,
            "unstuck_loss_allowance_pct": 0.02,
            "unstuck_threshold": 0.85,
        }
    ) == {
        "unstuck_enabled": 1.0,
        "unstuck_ema_gating_enabled": 0.0,
        "unstuck_close_pct": 0.125,
        "unstuck_ema_dist": -0.01,
        "unstuck_loss_allowance_pct": 0.02,
        "unstuck_threshold": 0.85,
    }


def test_single_coin_hsl_packs_state_machine_inputs():
    packed = _hsl_params(
        {
            "hsl_enabled": True,
            "hsl_red_threshold": 0.2,
            "hsl_ema_span_minutes": 60.0,
            "hsl_cooldown_minutes_after_red": 120.0,
            "hsl_no_restart_drawdown_threshold": 0.8,
            "hsl_restart_after_red_policy": "threshold",
            "hsl_tier_ratio_yellow": 0.5,
            "hsl_tier_ratio_orange": 0.75,
            "hsl_orange_tier_mode": "graceful_stop",
            "n_positions": 4,
        },
        signal_mode="coin",
    )

    assert packed == {
        "hsl_enabled": 1.0,
        "hsl_red_threshold": 0.2,
        "hsl_ema_span_minutes": 60.0,
        "hsl_cooldown_minutes_after_red": 120.0,
        "hsl_no_restart_drawdown_threshold": 0.8,
        "hsl_restart_policy": 1.0,
        "hsl_tier_ratio_yellow": 0.5,
        "hsl_tier_ratio_orange": 0.75,
        "hsl_orange_graceful_stop": 1.0,
        "hsl_signal_coin": 1.0,
        "hsl_slot_count": 1.0,
    }

    with pytest.raises(ValueError, match="cannot represent"):
        _hsl_params(
            {
                "hsl_enabled": True,
                "hsl_no_restart_drawdown_threshold": 0.99999999,
            },
            signal_mode="coin",
        )


def test_directional_parameter_matrix_keeps_side_values_separate():
    proxy = MpsEmaAnchorProxy.__new__(MpsEmaAnchorProxy)
    proxy.base_params = {
        "long": {key: 1.0 for key in EMA_ANCHOR_PARAM_KEYS},
        "short": {key: 2.0 for key in EMA_ANCHOR_PARAM_KEYS},
    }
    proxy.param_keys = EMA_ANCHOR_PARAM_KEYS

    matrix = proxy._parameter_matrix(
        [{"long_offset": 0.125, "short_offset": 0.25}]
    )

    assert matrix.shape == (1, 2 * len(EMA_ANCHOR_PARAM_KEYS))
    offset_index = EMA_ANCHOR_PARAM_KEYS.index("offset")
    assert matrix[0, offset_index] == 0.125
    assert matrix[0, len(EMA_ANCHOR_PARAM_KEYS) + offset_index] == 0.25


@pytest.mark.parametrize(("side", "base"), [("long", 1.0), ("short", 2.0)])
def test_multicoin_parameter_matrix_uses_only_enabled_side(side, base):
    proxy = MpsMulticoinEmaProxy.__new__(MpsMulticoinEmaProxy)
    proxy.sides = [side]
    proxy.base_params = {
        side: {key: base for key in EMA_ANCHOR_MULTICOIN_PARAM_KEYS}
    }

    other_side = "short" if side == "long" else "long"
    matrix = proxy._parameter_matrix(
        [
            {
                f"{side}_offset": 0.125,
                f"{side}_we_excess_allowance_pct": 0.25,
                f"{side}_twel_enforcer_threshold": 0.8,
                f"{other_side}_offset": 9.0,
            }
        ]
    )

    assert matrix.shape == (1, len(EMA_ANCHOR_MULTICOIN_PARAM_KEYS))
    offset_index = EMA_ANCHOR_MULTICOIN_PARAM_KEYS.index("offset")
    assert matrix[0, offset_index] == 0.125
    assert matrix[
        0, EMA_ANCHOR_MULTICOIN_PARAM_KEYS.index("we_excess_allowance_pct")
    ] == pytest.approx(0.25)
    assert matrix[
        0, EMA_ANCHOR_MULTICOIN_PARAM_KEYS.index("twel_enforcer_threshold")
    ] == pytest.approx(0.8)


def test_multicoin_parameter_matrix_keeps_dual_side_values_separate():
    proxy = MpsMulticoinEmaProxy.__new__(MpsMulticoinEmaProxy)
    proxy.sides = ["long", "short"]
    proxy.base_params = {
        "long": {key: 1.0 for key in EMA_ANCHOR_MULTICOIN_PARAM_KEYS},
        "short": {key: 2.0 for key in EMA_ANCHOR_MULTICOIN_PARAM_KEYS},
    }
    candidate = {"long_offset": 0.125, "short_offset": 0.25}

    long_matrix = proxy._parameter_matrix([candidate], "long")
    short_matrix = proxy._parameter_matrix([candidate], "short")

    offset_index = EMA_ANCHOR_MULTICOIN_PARAM_KEYS.index("offset")
    assert long_matrix[0, offset_index] == 0.125
    assert short_matrix[0, offset_index] == 0.25


@pytest.mark.parametrize("side", ["long", "short"])
def test_multicoin_tm_parameter_matrix_keeps_forager_and_strategy_values(side):
    proxy = MpsMulticoinEmaProxy.__new__(MpsMulticoinEmaProxy)
    proxy.sides = [side]
    proxy.param_keys = TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS
    proxy.base_params = {
        side: {key: 1.0 for key in TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS}
    }

    matrix = proxy._parameter_matrix(
        [
            {
                f"{side}_entry_threshold_base_pct": 0.125,
                f"{side}_forager_volume_drop_pct": 0.25,
                f"{side}_we_excess_allowance_pct": 0.4,
                f"{side}_twel_enforcer_threshold": 0.75,
            }
        ]
    )

    assert matrix.shape == (1, len(TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS))
    assert matrix[
        0,
        TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index(
            "entry_threshold_base_pct"
        ),
    ] == 0.125
    assert matrix[
        0,
        TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index(
            "forager_volume_drop_pct"
        ),
    ] == 0.25
    assert matrix[
        0,
        TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index(
            "we_excess_allowance_pct"
        ),
    ] == pytest.approx(0.4)
    assert matrix[
        0,
        TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index(
            "twel_enforcer_threshold"
        ),
    ] == pytest.approx(0.75)


def test_multicoin_tm_parameter_matrix_keeps_dual_side_values_separate():
    proxy = MpsMulticoinEmaProxy.__new__(MpsMulticoinEmaProxy)
    proxy.sides = ["long", "short"]
    proxy.param_keys = TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS
    proxy.base_params = {
        "long": {
            key: 1.0 for key in TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS
        },
        "short": {
            key: 2.0 for key in TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS
        },
    }
    candidate = {
        "long_entry_threshold_base_pct": 0.125,
        "short_entry_threshold_base_pct": 0.25,
    }

    long_matrix = proxy._parameter_matrix([candidate], "long")
    short_matrix = proxy._parameter_matrix([candidate], "short")

    index = TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS.index(
        "entry_threshold_base_pct"
    )
    assert long_matrix[0, index] == 0.125
    assert short_matrix[0, index] == 0.25


def test_combine_hedged_multicoin_outputs_uses_conservative_surface():
    torch = pytest.importorskip("torch")

    def side_output(*, end, minimum, fill, first_fill, last_fill, liq):
        return {
            "day_end_eq": torch.tensor([end], dtype=torch.float64),
            "day_min_eq": torch.tensor([minimum], dtype=torch.float64),
            "day_max_dd": torch.tensor([[0.10, 0.20]]),
            "day_volume": torch.tensor([[0.4, 0.5]]),
            "day_has_fill": torch.tensor([fill]),
            "day_min_balance": torch.tensor([[1_000.0, 1_000.0]]),
            "day_net_pnl": torch.tensor(
                [[end[0] - 1_000.0, end[1] - end[0]]]
            ),
            "day_last_fill_balance": torch.tensor([end]),
            "day_fill_count": torch.tensor(
                [[float(fill[0]), float(fill[1])]]
            ),
            "max_dd": torch.tensor([0.20]),
            "held_max_ms": torch.tensor([100.0]),
            "held_sum_ms": torch.tensor([300.0]),
            "held_count": torch.tensor([2.0]),
            "position_unchanged_max_ms": torch.tensor([150.0]),
            "gap_hist": torch.tensor([[1, 2]]),
            "gap_max_ms": torch.tensor([300.0]),
            "first_fill_ts": torch.tensor([first_fill]),
            "last_fill_ts": torch.tensor([last_fill]),
            "recovery_max_ms": torch.tensor([400.0]),
            "pnl_recovery_max_ms": torch.tensor([350.0]),
            "last_high_ts": torch.tensor([900.0]),
            "first_eq_ts": torch.tensor([100.0]),
            "last_eq_ts": torch.tensor([1_000.0]),
            "liq_step": torch.tensor([liq]),
            "profit_sum": torch.tensor([20.0]),
            "loss_sum": torch.tensor([5.0]),
            "fill_count": torch.tensor([float(sum(fill))]),
            "fill_count_entry": torch.tensor([float(sum(fill))]),
            "fill_count_long": torch.tensor([float(sum(fill))]),
            "fills_active_days_count": torch.tensor([float(any(fill))]),
        }

    long = side_output(
        end=[1_100.0, 1_200.0],
        minimum=[1_050.0, 1_100.0],
        fill=[True, False],
        first_fill=float("nan"),
        last_fill=700.0,
        liq=-1,
    )
    short = side_output(
        end=[950.0, 900.0],
        minimum=[925.0, 850.0],
        fill=[False, True],
        first_fill=300.0,
        last_fill=float("nan"),
        liq=-1,
    )
    short["day_max_dd"] = torch.tensor([[0.05, 0.30]])
    short["day_volume"] = torch.tensor([[0.1, 0.2]])
    short["max_dd"] = torch.tensor([0.30])
    short["held_max_ms"] = torch.tensor([200.0])
    short["held_sum_ms"] = torch.tensor([500.0])
    short["held_count"] = torch.tensor([3.0])
    short["position_unchanged_max_ms"] = torch.tensor([250.0])
    short["gap_max_ms"] = torch.tensor([250.0])
    short["recovery_max_ms"] = torch.tensor([500.0])
    short["pnl_recovery_max_ms"] = torch.tensor([450.0])
    short["last_high_ts"] = torch.tensor([800.0])

    combined = _combine_hedged_multicoin_outputs(
        long, short, 1_000.0, 0.05, 0, 60_000
    )

    assert combined["day_end_eq"].tolist() == [[1_050.0, 1_100.0]]
    assert combined["day_min_eq"].tolist() == [[975.0, 950.0]]
    np.testing.assert_allclose(combined["day_max_dd"].numpy(), [[0.15, 0.50]])
    assert combined["max_dd"].item() == pytest.approx(0.50)
    np.testing.assert_allclose(combined["day_volume"].numpy(), [[0.5, 0.7]])
    assert combined["day_has_fill"].tolist() == [[True, True]]
    assert combined["day_net_pnl"].tolist() == [[50.0, 50.0]]
    assert combined["day_last_fill_balance"].tolist() == [[1_050.0, 1_100.0]]
    assert combined["day_fill_count"].tolist() == [[1.0, 1.0]]
    assert combined["first_fill_ts"].item() == 300.0
    assert combined["last_fill_ts"].item() == 700.0
    assert combined["last_high_ts"].item() == 800.0
    assert combined["liq_step"].item() == -1
    assert combined["profit_sum"].item() == 40.0
    assert combined["loss_sum"].item() == 10.0
    assert combined["fill_count"].item() == 2.0
    assert combined["fill_count_entry"].item() == 2.0
    assert combined["fill_count_long"].item() == 2.0
    assert combined["fills_active_days_count"].item() == 1.0
    assert combined["held_sum_ms"].item() == 800.0
    assert combined["held_count"].item() == 5.0
    assert combined["position_unchanged_max_ms"].item() == 250.0
    assert combined["pnl_recovery_max_ms"].item() == 450.0

    short["day_min_eq"][0, 1] = float("inf")
    short["last_eq_ts"] = torch.tensor([800.0])
    truncated = _combine_hedged_multicoin_outputs(
        long, short, 1_000.0, 0.05, 0, 60_000
    )
    assert truncated["day_end_eq"][0, 1].item() == 0.0
    assert torch.isinf(truncated["day_min_eq"][0, 1])
    assert truncated["day_volume"][0, 1].item() == 0.0
    assert not truncated["day_has_fill"][0, 1].item()
    assert truncated["last_eq_ts"].item() == 800.0

    short["day_min_eq"][0, 1] = 850.0
    long["liq_step"] = torch.tensor([1])
    liquidated = _combine_hedged_multicoin_outputs(
        long, short, 1_000.0, 0.05, 0, 60_000
    )
    assert torch.isfinite(liquidated["day_min_eq"][0, 0])
    assert torch.isinf(liquidated["day_min_eq"][0, 1])
    assert liquidated["day_end_eq"][0, 1].item() == 0.0
    assert liquidated["liq_step"].item() == 1


def test_combine_hedged_multicoin_outputs_detects_shared_equity_liquidation():
    torch = pytest.importorskip("torch")

    def side_output():
        return {
            "day_end_eq": torch.tensor([[1_000.0, 520.0, 900.0]]),
            "day_min_eq": torch.tensor([[900.0, 520.0, 800.0]]),
            "day_max_dd": torch.tensor([[0.10, 0.48, 0.20]]),
            "day_volume": torch.tensor([[0.1, 0.1, 0.1]]),
            "day_has_fill": torch.tensor([[True, True, True]]),
            "day_min_balance": torch.tensor([[900.0, 900.0, 900.0]]),
            "day_net_pnl": torch.zeros((1, 3)),
            "day_last_fill_balance": torch.full((1, 3), 1_000.0),
            "day_fill_count": torch.ones((1, 3)),
            "max_dd": torch.tensor([0.48]),
            "held_max_ms": torch.tensor([100.0]),
            "held_sum_ms": torch.tensor([100.0]),
            "held_count": torch.tensor([1.0]),
            "position_unchanged_max_ms": torch.tensor([150.0]),
            "gap_hist": torch.tensor([[1, 2]]),
            "gap_max_ms": torch.tensor([300.0]),
            "first_fill_ts": torch.tensor([100.0]),
            "last_fill_ts": torch.tensor([200_000_000.0]),
            "recovery_max_ms": torch.tensor([400.0]),
            "pnl_recovery_max_ms": torch.tensor([350.0]),
            "last_high_ts": torch.tensor([1_000.0]),
            "first_eq_ts": torch.tensor([0.0]),
            "last_eq_ts": torch.tensor([200_000_000.0]),
            "liq_step": torch.tensor([-1]),
            "profit_sum": torch.tensor([0.0]),
            "loss_sum": torch.tensor([0.0]),
            "fill_count": torch.tensor([3.0]),
            "fill_count_entry": torch.tensor([2.0]),
            "fill_count_long": torch.tensor([2.0]),
            "fills_active_days_count": torch.tensor([1.0]),
        }

    combined = _combine_hedged_multicoin_outputs(
        side_output(), side_output(), 1_000.0, 0.05, 0, 60_000
    )

    assert combined["liq_step"].item() == 1
    assert torch.isfinite(combined["day_min_eq"][0, 0])
    assert torch.isinf(combined["day_min_eq"][0, 1])
    assert torch.isinf(combined["day_min_eq"][0, 2])
    assert combined["day_end_eq"][0, 1].item() == 0.0
    assert combined["last_eq_ts"].item() == 86_340_000.0


def test_combine_hedged_multicoin_outputs_detects_shared_balance_depletion():
    torch = pytest.importorskip("torch")

    def side_output():
        return {
            "day_end_eq": torch.tensor([[900.0, 600.0, 800.0]]),
            "day_min_eq": torch.tensor([[800.0, 600.0, 700.0]]),
            "day_max_dd": torch.tensor([[0.10, 0.40, 0.30]]),
            "day_volume": torch.tensor([[0.1, 0.1, 0.1]]),
            "day_has_fill": torch.tensor([[True, True, True]]),
            "day_min_balance": torch.tensor([[900.0, 450.0, 700.0]]),
            "day_net_pnl": torch.zeros((1, 3)),
            "day_last_fill_balance": torch.full((1, 3), 1_000.0),
            "day_fill_count": torch.ones((1, 3)),
            "max_dd": torch.tensor([0.40]),
            "held_max_ms": torch.tensor([100.0]),
            "held_sum_ms": torch.tensor([100.0]),
            "held_count": torch.tensor([1.0]),
            "position_unchanged_max_ms": torch.tensor([150.0]),
            "gap_hist": torch.tensor([[1, 2]]),
            "gap_max_ms": torch.tensor([300.0]),
            "first_fill_ts": torch.tensor([100.0]),
            "last_fill_ts": torch.tensor([200_000_000.0]),
            "recovery_max_ms": torch.tensor([400.0]),
            "pnl_recovery_max_ms": torch.tensor([350.0]),
            "last_high_ts": torch.tensor([1_000.0]),
            "first_eq_ts": torch.tensor([0.0]),
            "last_eq_ts": torch.tensor([200_000_000.0]),
            "liq_step": torch.tensor([-1]),
            "profit_sum": torch.tensor([0.0]),
            "loss_sum": torch.tensor([0.0]),
            "fill_count": torch.tensor([3.0]),
            "fill_count_entry": torch.tensor([2.0]),
            "fill_count_long": torch.tensor([2.0]),
            "fills_active_days_count": torch.tensor([1.0]),
        }

    combined = _combine_hedged_multicoin_outputs(
        side_output(), side_output(), 1_000.0, 0.05, 0, 60_000
    )

    # Combined equity remains 200, above the 50 floor, while conservative
    # combined realized balance is -100 and must terminate the screen.
    assert combined["liq_step"].item() == 1
    assert torch.isfinite(combined["day_min_eq"][0, 0])
    assert torch.isinf(combined["day_min_eq"][0, 1])
    assert combined["last_eq_ts"].item() == 86_340_000.0


def test_multicoin_coin_overrides_pack_only_explicit_exact_values():
    strategy_base = {key: 1.0 for key in EMA_ANCHOR_PARAM_KEYS[:-2]}
    strategy_override = dict(strategy_base, offset=0.25, ema_span_0=90.0)
    payload = SimpleNamespace(
        strategy_params_list=[
            {"long": strategy_base},
            {"long": strategy_override},
        ],
        bot_params_list=[
            {"long": {"entry_cooldown_minutes": 0.0, "wallet_exposure_limit": -1.0}},
            {
                "long": {
                    "entry_cooldown_minutes": 15.0,
                    "wallet_exposure_limit": 0.4,
                    "risk_we_excess_allowance_pct": 0.25,
                    "unstuck_enabled": True,
                    "unstuck_ema_gating_enabled": False,
                    "unstuck_close_pct": 0.125,
                    "unstuck_ema_dist": -0.01,
                    "unstuck_loss_allowance_pct": 0.02,
                    "unstuck_threshold": 0.85,
                }
            },
        ],
    )
    config = {
        "coin_overrides": {
            "ETH": {
                "bot": {
                    "long": {
                        "strategy": {
                            "ema_anchor": {"offset": 0.25, "ema_span_0": 90.0}
                        },
                        "risk": {
                            "entry_cooldown_minutes": 15.0,
                            "we_excess_allowance_pct": 0.25,
                        },
                        "wallet_exposure_limit": 0.4,
                        "unstuck": {
                            "enabled": True,
                            "ema_gating_enabled": False,
                            "close_pct": 0.125,
                            "ema_dist": -0.01,
                            "loss_allowance_pct": 0.02,
                            "threshold": 0.85,
                        },
                    }
                }
            }
        }
    }

    matrix, contract = _build_multicoin_ema_coin_overrides(
        config=config,
        mss={"BTC": {}, "ETH": {}},
        exchange="bybit",
        coins=["BTC", "ETH"],
        payload=payload,
        side="long",
        resolve_override=lambda config, _mss, _exchange, coin: config[
            "coin_overrides"
        ].get(coin, {}),
    )

    assert matrix.shape == (2, 19)
    assert np.isnan(matrix[0]).all()
    assert matrix[1, EMA_ANCHOR_PARAM_KEYS.index("offset")] == pytest.approx(0.25)
    assert matrix[1, EMA_ANCHOR_PARAM_KEYS.index("ema_span_0")] == pytest.approx(90.0)
    assert matrix[1, 10] == pytest.approx(15.0)
    assert matrix[1, 11] == pytest.approx(0.4)
    assert matrix[1, 12] == pytest.approx(0.25)
    assert matrix[1, 13:].tolist() == pytest.approx(
        [1.0, 0.0, 0.125, -0.01, 0.02, 0.85]
    )
    assert contract["coins"] == ["BTC", "ETH"]
    assert contract["values"][0] == [None] * 19


def test_multicoin_coin_overrides_pack_dual_sides_independently():
    strategy_base = {key: 1.0 for key in EMA_ANCHOR_PARAM_KEYS[:-2]}
    payload = SimpleNamespace(
        strategy_params_list=[
            {"long": strategy_base, "short": strategy_base},
            {
                "long": dict(strategy_base, offset=0.25),
                "short": dict(strategy_base, offset=0.5),
            },
        ],
        bot_params_list=[
            {
                "long": {"entry_cooldown_minutes": 0.0, "wallet_exposure_limit": -1.0},
                "short": {"entry_cooldown_minutes": 0.0, "wallet_exposure_limit": -1.0},
            },
            {
                "long": {"entry_cooldown_minutes": 0.0, "wallet_exposure_limit": 0.4},
                "short": {"entry_cooldown_minutes": 30.0, "wallet_exposure_limit": -1.0},
            },
        ],
    )
    config = {
        "coin_overrides": {
            "ETH": {
                "bot": {
                    "long": {
                        "strategy": {"ema_anchor": {"offset": 0.25}},
                        "wallet_exposure_limit": 0.4,
                    },
                    "short": {
                        "strategy": {"ema_anchor": {"offset": 0.5}},
                        "risk": {"entry_cooldown_minutes": 30.0},
                    },
                }
            }
        }
    }
    def resolver(config, _mss, _exchange, coin):
        return config["coin_overrides"].get(coin, {})

    long_matrix, _ = _build_multicoin_ema_coin_overrides(
        config=config,
        mss={"BTC": {}, "ETH": {}},
        exchange="bybit",
        coins=["BTC", "ETH"],
        payload=payload,
        side="long",
        resolve_override=resolver,
    )
    short_matrix, _ = _build_multicoin_ema_coin_overrides(
        config=config,
        mss={"BTC": {}, "ETH": {}},
        exchange="bybit",
        coins=["BTC", "ETH"],
        payload=payload,
        side="short",
        resolve_override=resolver,
    )

    offset_index = EMA_ANCHOR_PARAM_KEYS.index("offset")
    assert long_matrix[1, offset_index] == pytest.approx(0.25)
    assert long_matrix[1, 11] == pytest.approx(0.4)
    assert np.isnan(long_matrix[1, 10])
    assert short_matrix[1, offset_index] == pytest.approx(0.5)
    assert short_matrix[1, 10] == pytest.approx(30.0)
    assert np.isnan(short_matrix[1, 11])


def test_multicoin_tm_coin_overrides_pack_only_explicit_exact_values():
    assert tuple(
        key for key, _path in TRAILING_MARTINGALE_COIN_OVERRIDE_PATHS
    ) == TRAILING_MARTINGALE_PARAM_KEYS[:23]
    strategy_base = {
        "ema_span_0": 10.0,
        "ema_span_1": 20.0,
        "volatility_ema_span_1h": 30.0,
        "volatility_ema_span_1m": 40.0,
        "entry": {
            "ema_gate_mode": "all",
            "double_down_factor": 1.1,
            "initial_ema_dist": 0.01,
            "initial_qty_pct": 0.02,
            "threshold_base_pct": 0.03,
            "threshold_we_weight": 0.04,
            "threshold_volatility_1h_weight": 0.05,
            "threshold_volatility_1m_weight": 0.06,
            "retracement_base_pct": 0.007,
            "retracement_we_weight": 0.08,
            "retracement_volatility_1h_weight": 0.09,
            "retracement_volatility_1m_weight": 0.1,
        },
        "close": {
            "qty_pct": 0.2,
            "threshold_base_pct": 0.03,
            "threshold_we_weight": 0.04,
            "threshold_volatility_1h_weight": 0.05,
            "threshold_volatility_1m_weight": 0.06,
            "retracement_base_pct": 0.007,
            "retracement_volatility_1h_weight": 0.09,
            "retracement_volatility_1m_weight": 0.1,
        },
    }
    strategy_override = {
        **strategy_base,
        "entry": {**strategy_base["entry"], "threshold_base_pct": 0.25},
        "close": {**strategy_base["close"], "qty_pct": 0.5},
    }
    payload = SimpleNamespace(
        strategy_params_list=[
            {"long": strategy_base},
            {"long": strategy_override},
        ],
        bot_params_list=[
            {
                "long": {
                    "entry_cooldown_minutes": 0.0,
                    "total_wallet_exposure_limit": 1.0,
                    "wallet_exposure_limit": -1.0,
                }
            },
            {
                "long": {
                    "entry_cooldown_minutes": 15.0,
                    "total_wallet_exposure_limit": 1.0,
                    "wallet_exposure_limit": 0.4,
                    "risk_we_excess_allowance_pct": 0.25,
                    "risk_wel_enforcer_enabled": True,
                    "risk_wel_enforcer_threshold": 0.8,
                    "unstuck_enabled": True,
                    "unstuck_ema_gating_enabled": False,
                    "unstuck_close_pct": 0.125,
                    "unstuck_ema_dist": -0.01,
                    "unstuck_loss_allowance_pct": 0.02,
                    "unstuck_threshold": 0.85,
                }
            },
        ],
    )
    config = {
        "coin_overrides": {
            "ETH": {
                "bot": {
                    "long": {
                        "strategy": {
                            "trailing_martingale": {
                                "entry": {"threshold_base_pct": 0.25},
                                "close": {"qty_pct": 0.5},
                            }
                        },
                        "risk": {
                            "entry_cooldown_minutes": 15.0,
                            "we_excess_allowance_pct": 0.25,
                            "position_exposure_enforcer_enabled": True,
                            "position_exposure_enforcer_threshold": 0.8,
                        },
                        "wallet_exposure_limit": 0.4,
                        "unstuck": {
                            "enabled": True,
                            "ema_gating_enabled": False,
                            "close_pct": 0.125,
                            "ema_dist": -0.01,
                            "loss_allowance_pct": 0.02,
                            "threshold": 0.85,
                        },
                    }
                }
            }
        }
    }

    matrix, contract = _build_multicoin_tm_coin_overrides(
        config=config,
        mss={"BTC": {}, "ETH": {}},
        exchange="bybit",
        coins=["BTC", "ETH"],
        payload=payload,
        side="long",
        resolve_override=lambda config, _mss, _exchange, coin: config[
            "coin_overrides"
        ].get(coin, {}),
    )

    assert matrix.shape == (2, 34)
    assert np.isnan(matrix[0]).all()
    assert matrix[1, 7] == pytest.approx(0.25)
    assert matrix[1, 15] == pytest.approx(0.5)
    assert matrix[1, 23] == pytest.approx(15.0)
    assert matrix[1, 24] == pytest.approx(0.4)
    assert matrix[1, 25] == pytest.approx(0.25)
    assert matrix[1, 26] == pytest.approx(1.0)
    assert matrix[1, 27] == pytest.approx(0.8)
    assert matrix[1, 28:].tolist() == pytest.approx(
        [1.0, 0.0, 0.125, -0.01, 0.02, 0.85]
    )
    assert contract["values"][0] == [None] * 34


def test_trailing_parameter_matrix_keeps_nested_flattened_sides_separate():
    proxy = MpsEmaAnchorProxy.__new__(MpsEmaAnchorProxy)
    proxy.param_keys = TRAILING_MARTINGALE_PARAM_KEYS
    proxy.base_params = {
        "long": {key: 1.0 for key in TRAILING_MARTINGALE_PARAM_KEYS},
        "short": {key: 2.0 for key in TRAILING_MARTINGALE_PARAM_KEYS},
    }

    matrix = proxy._parameter_matrix(
        [
            {
                "long_entry_threshold_base_pct": 0.125,
                "short_close_qty_pct": 0.25,
            }
        ]
    )

    assert matrix.shape == (1, 2 * len(TRAILING_MARTINGALE_PARAM_KEYS))
    entry_index = TRAILING_MARTINGALE_PARAM_KEYS.index(
        "entry_threshold_base_pct"
    )
    close_index = TRAILING_MARTINGALE_PARAM_KEYS.index("close_qty_pct")
    assert matrix[0, entry_index] == 0.125
    assert (
        matrix[0, len(TRAILING_MARTINGALE_PARAM_KEYS) + close_index] == 0.25
    )


def test_gpu_side_enablement_uses_config_risk_not_per_coin_sentinel():
    config = {
        "bot": {
            "long": {
                "risk": {"total_wallet_exposure_limit": 1.0, "n_positions": 1}
            },
            "short": {
                "risk": {"total_wallet_exposure_limit": 0.0, "n_positions": 0}
            },
        },
        "live": {"approved_coins": {"long": ["BTC"], "short": ["BTC"]}},
    }

    assert gpu_side_enabled(config, "long")
    assert not gpu_side_enabled(config, "short")


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("disabled", (0.0, 0.0)),
        ("all", (1.0, 1.0)),
        ("initial", (1.0, 0.0)),
        ("reentry", (0.0, 1.0)),
    ],
)
def test_trailing_martingale_flattening_preserves_nested_params_and_gates(
    mode, expected
):
    strategy = {
        "ema_span_0": 10.0,
        "ema_span_1": 20.0,
        "volatility_ema_span_1h": 30.0,
        "volatility_ema_span_1m": 40.0,
        "entry": {
            "ema_gate_mode": mode,
            "double_down_factor": 1.1,
            "initial_ema_dist": 0.01,
            "initial_qty_pct": 0.02,
            "threshold_base_pct": 0.03,
            "threshold_we_weight": 0.04,
            "threshold_volatility_1h_weight": 0.05,
            "threshold_volatility_1m_weight": 0.06,
            "retracement_base_pct": 0.007,
            "retracement_we_weight": 0.08,
            "retracement_volatility_1h_weight": 0.09,
            "retracement_volatility_1m_weight": 0.1,
        },
        "close": {
            "qty_pct": 0.2,
            "threshold_base_pct": 0.03,
            "threshold_we_weight": 0.04,
            "threshold_volatility_1h_weight": 0.05,
            "threshold_volatility_1m_weight": 0.06,
            "retracement_base_pct": 0.007,
            "retracement_volatility_1h_weight": 0.09,
            "retracement_volatility_1m_weight": 0.1,
        },
    }
    flattened = flatten_trailing_martingale_params(
        strategy,
        {"entry_cooldown_minutes": 7.0, "total_wallet_exposure_limit": 1.5},
    )

    assert tuple(flattened) == TRAILING_MARTINGALE_PARAM_KEYS
    assert flattened["entry_double_down_factor"] == 1.1
    assert flattened["close_qty_pct"] == 0.2
    assert (flattened["gate_initial"], flattened["gate_reentry"]) == expected


def test_trailing_martingale_flattening_rejects_unknown_gate_mode():
    with pytest.raises(ValueError, match="ema_gate_mode"):
        flatten_trailing_martingale_params(
            {
                "entry": {"ema_gate_mode": "mystery"},
                "close": {},
            },
            {"total_wallet_exposure_limit": 1.0},
        )
