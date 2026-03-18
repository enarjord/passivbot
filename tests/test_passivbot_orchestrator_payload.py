import pytest


def test_bot_params_to_rust_dict_preserves_forager_score_weights_dict():
    import passivbot as pb_mod

    class FakeBot:
        def __init__(self):
            self._global = {
                "hsl_enabled": True,
                "hsl_red_threshold": 0.25,
                "hsl_ema_span_minutes": 60.0,
                "hsl_cooldown_minutes_after_red": 0.0,
                "hsl_no_restart_drawdown_threshold": 1.0,
                "hsl_tier_ratios.yellow": 0.5,
                "hsl_tier_ratios.orange": 0.75,
                "hsl_orange_tier_mode": "tp_only_with_active_entry_cancellation",
                "hsl_panic_close_order_type": "market",
            }
            self._bot = {
                "close_grid_markup_end": 0.01,
                "close_grid_markup_start": 0.02,
                "close_grid_qty_pct": 0.1,
                "close_trailing_retracement_pct": 0.0,
                "close_trailing_grid_ratio": 0.0,
                "close_trailing_qty_pct": 0.0,
                "close_trailing_threshold_pct": 0.0,
                "entry_grid_double_down_factor": 1.0,
                "entry_grid_spacing_volatility_weight": 0.0,
                "entry_grid_spacing_we_weight": 0.0,
                "entry_grid_spacing_pct": 0.02,
                "entry_volatility_ema_span_hours": 72.0,
                "entry_initial_ema_dist": -0.01,
                "entry_initial_qty_pct": 0.1,
                "entry_trailing_double_down_factor": 1.0,
                "entry_trailing_retracement_pct": 0.0,
                "entry_trailing_retracement_we_weight": 0.0,
                "entry_trailing_retracement_volatility_weight": 0.0,
                "entry_trailing_grid_ratio": 0.0,
                "entry_trailing_threshold_pct": 0.0,
                "entry_trailing_threshold_we_weight": 0.0,
                "entry_trailing_threshold_volatility_weight": 0.0,
                "filter_volatility_ema_span": 10.0,
                "filter_volume_ema_span": 10.0,
                "forager_volume_drop_pct": 0.95,
                "forager_score_weights": {
                    "volume": 0.0,
                    "ema_readiness": 0.25,
                    "volatility": 0.75,
                },
                "ema_span_0": 10.0,
                "ema_span_1": 20.0,
                "n_positions": 1.0,
                "total_wallet_exposure_limit": 1.0,
                "wallet_exposure_limit": 1.0,
                "risk_wel_enforcer_threshold": 1.0,
                "risk_twel_enforcer_threshold": 1.0,
                "risk_we_excess_allowance_pct": 0.0,
                "unstuck_close_pct": 0.0,
                "unstuck_ema_dist": 0.0,
                "unstuck_loss_allowance_pct": 0.0,
                "unstuck_threshold": 0.0,
            }

        def bot_value(self, pside, key):
            if key in self._global:
                return self._global[key]
            return self._bot[key]

        def bp(self, pside, key, symbol=None):
            return self._bot[key]

    out = pb_mod.Passivbot._bot_params_to_rust_dict(FakeBot(), "long", None)

    assert out["n_positions"] == 1
    assert out["forager_score_weights"] == {
        "volume": pytest.approx(0.0),
        "ema_readiness": pytest.approx(0.25),
        "volatility": pytest.approx(0.75),
    }
