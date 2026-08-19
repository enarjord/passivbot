from types import SimpleNamespace

import numpy as np
import pytest

from optimization.gpu.model import (
    EMA_ANCHOR_MULTICOIN_PARAM_KEYS,
    EMA_ANCHOR_PARAM_KEYS,
    TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS,
    TRAILING_MARTINGALE_PARAM_KEYS,
    flatten_trailing_martingale_params,
    gpu_side_enabled,
)
from optimization.gpu.service import (
    MpsEmaAnchorProxy,
    MpsMulticoinEmaProxy,
    _build_multicoin_ema_coin_overrides,
    _combine_hedged_multicoin_outputs,
    _require_complete_valid_tail,
)


def test_gpu_proxy_requires_complete_valid_tail():
    _require_complete_valid_tail(99, 100)

    with pytest.raises(ValueError, match="force-realizes open positions"):
        _require_complete_valid_tail(98, 100)


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
        [{f"{side}_offset": 0.125, f"{other_side}_offset": 9.0}]
    )

    assert matrix.shape == (1, len(EMA_ANCHOR_MULTICOIN_PARAM_KEYS))
    offset_index = EMA_ANCHOR_MULTICOIN_PARAM_KEYS.index("offset")
    assert matrix[0, offset_index] == 0.125


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
            "max_dd": torch.tensor([0.20]),
            "held_max_ms": torch.tensor([100.0]),
            "gap_hist": torch.tensor([[1, 2]]),
            "gap_max_ms": torch.tensor([300.0]),
            "first_fill_ts": torch.tensor([first_fill]),
            "last_fill_ts": torch.tensor([last_fill]),
            "recovery_max_ms": torch.tensor([400.0]),
            "last_high_ts": torch.tensor([900.0]),
            "first_eq_ts": torch.tensor([100.0]),
            "last_eq_ts": torch.tensor([1_000.0]),
            "liq_step": torch.tensor([liq]),
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
    short["gap_max_ms"] = torch.tensor([250.0])
    short["recovery_max_ms"] = torch.tensor([500.0])
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
    assert combined["first_fill_ts"].item() == 300.0
    assert combined["last_fill_ts"].item() == 700.0
    assert combined["last_high_ts"].item() == 800.0
    assert combined["liq_step"].item() == -1

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
            "max_dd": torch.tensor([0.48]),
            "held_max_ms": torch.tensor([100.0]),
            "gap_hist": torch.tensor([[1, 2]]),
            "gap_max_ms": torch.tensor([300.0]),
            "first_fill_ts": torch.tensor([100.0]),
            "last_fill_ts": torch.tensor([200_000_000.0]),
            "recovery_max_ms": torch.tensor([400.0]),
            "last_high_ts": torch.tensor([1_000.0]),
            "first_eq_ts": torch.tensor([0.0]),
            "last_eq_ts": torch.tensor([200_000_000.0]),
            "liq_step": torch.tensor([-1]),
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
            "max_dd": torch.tensor([0.40]),
            "held_max_ms": torch.tensor([100.0]),
            "gap_hist": torch.tensor([[1, 2]]),
            "gap_max_ms": torch.tensor([300.0]),
            "first_fill_ts": torch.tensor([100.0]),
            "last_fill_ts": torch.tensor([200_000_000.0]),
            "recovery_max_ms": torch.tensor([400.0]),
            "last_high_ts": torch.tensor([1_000.0]),
            "first_eq_ts": torch.tensor([0.0]),
            "last_eq_ts": torch.tensor([200_000_000.0]),
            "liq_step": torch.tensor([-1]),
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
            {"long": {"entry_cooldown_minutes": 15.0, "wallet_exposure_limit": 0.4}},
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
                        "risk": {"entry_cooldown_minutes": 15.0},
                        "wallet_exposure_limit": 0.4,
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

    assert matrix.shape == (2, 12)
    assert np.isnan(matrix[0]).all()
    assert matrix[1, EMA_ANCHOR_PARAM_KEYS.index("offset")] == pytest.approx(0.25)
    assert matrix[1, EMA_ANCHOR_PARAM_KEYS.index("ema_span_0")] == pytest.approx(90.0)
    assert matrix[1, 10] == pytest.approx(15.0)
    assert matrix[1, 11] == pytest.approx(0.4)
    assert contract["coins"] == ["BTC", "ETH"]
    assert contract["values"][0] == [None] * 12


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
