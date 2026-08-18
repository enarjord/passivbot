from types import SimpleNamespace

import numpy as np
import pytest

from optimization.gpu.model import (
    EMA_ANCHOR_MULTICOIN_PARAM_KEYS,
    EMA_ANCHOR_PARAM_KEYS,
    TRAILING_MARTINGALE_PARAM_KEYS,
    flatten_trailing_martingale_params,
    gpu_side_enabled,
)
from optimization.gpu.service import (
    MpsEmaAnchorProxy,
    MpsMulticoinEmaProxy,
    _build_multicoin_ema_coin_overrides,
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
    proxy.side = side
    proxy.base_params = {
        key: base for key in EMA_ANCHOR_MULTICOIN_PARAM_KEYS
    }

    other_side = "short" if side == "long" else "long"
    matrix = proxy._parameter_matrix(
        [{f"{side}_offset": 0.125, f"{other_side}_offset": 9.0}]
    )

    assert matrix.shape == (1, len(EMA_ANCHOR_MULTICOIN_PARAM_KEYS))
    offset_index = EMA_ANCHOR_MULTICOIN_PARAM_KEYS.index("offset")
    assert matrix[0, offset_index] == 0.125


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
