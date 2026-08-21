from __future__ import annotations

import copy
from dataclasses import replace
import os

import numpy as np

from optimization.gpu.model import (
    EMA_ANCHOR_PARAM_KEYS,
    EMA_ANCHOR_MULTICOIN_PARAM_KEYS,
    GPU_STRATEGY_PARAM_KEYS,
    MPS_MULTICOIN_MAX_COINS,
    ProxyMarket,
    ProxyRun,
    TRAILING_MARTINGALE_COIN_OVERRIDE_PATHS,
    TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS,
    build_mps_data,
    build_mps_multicoin_data,
    flatten_trailing_martingale_params,
    gpu_side_enabled,
)


CORE_OUTPUT_KEYS = {
    "day_end_eq",
    "day_min_eq",
    "day_max_dd",
    "day_volume",
    "day_has_fill",
    "day_net_pnl",
    "day_last_fill_balance",
    "day_fill_count",
    "day_min_balance",
    "max_dd",
    "held_max_ms",
    "position_unchanged_max_ms",
    "gap_hist",
    "gap_max_ms",
    "first_fill_ts",
    "last_fill_ts",
    "recovery_max_ms",
    "last_high_ts",
    "first_eq_ts",
    "last_eq_ts",
    "liq_step",
    "profit_sum",
    "loss_sum",
    "entry_initial_balance_pct",
    "entry_initial_balance_pct_long",
    "entry_initial_balance_pct_short",
    "total_wallet_exposure_max",
    "total_wallet_exposure_mean",
}

DIRECTIONAL_HSL_OUTPUT_KEYS = {
    "hsl_long_enabled",
    "hsl_short_enabled",
    "hsl_triggers_long",
    "hsl_triggers_short",
    "hsl_restarts_long",
    "hsl_restarts_short",
    "hsl_tier_samples_total",
    "hsl_tier_samples_yellow",
    "hsl_tier_samples_orange",
    "hsl_tier_samples_red",
    "hsl_duration_sum_steps",
    "hsl_duration_max_steps",
    "hsl_duration_count",
    "hsl_trigger_drawdown_sum",
    "hsl_trigger_drawdown_count",
    "hsl_flatten_time_sum_steps",
    "hsl_flatten_time_count",
    "hsl_restart_retrigger_count",
    "hsl_halt_to_restart_equity_loss",
    "hsl_panic_close_loss_sum",
    "hsl_panic_close_loss_max",
    "hsl_panic_loss_drawdown_min",
    "hsl_panic_loss_drawdown_sum",
    "hsl_panic_loss_drawdown_max",
    "hsl_panic_loss_drawdown_count",
}

_DUAL_SIDE_MULTICOIN_INTRADAY_CUTOFF_METRICS = {
    "adg_pnl",
    "adg_pnl_w",
    "fills_analysis_duration_days",
    "fills_count",
    "fills_per_day",
    "mdg_pnl",
    "mdg_pnl_w",
    "sharpe_ratio_pnl",
    "sharpe_ratio_pnl_w",
    "sortino_ratio_pnl",
    "sortino_ratio_pnl_w",
}


def _require_multicoin_metric_topology(sides, needed_metrics) -> None:
    unsupported = (
        set(needed_metrics) & _DUAL_SIDE_MULTICOIN_INTRADAY_CUTOFF_METRICS
        if len(sides) == 2
        else set()
    )
    if unsupported:
        raise ValueError(
            "MPS dual-side multicoin proxy cannot reconstruct the intraday "
            "shared-liquidation cutoff required by these metrics: "
            + ", ".join(sorted(unsupported))
        )


def _candidate_wallet_exposure_limit_outputs(
    candidates: list[dict],
    base_limits: dict[str, float],
    *,
    torch,
) -> dict:
    """Expose each candidate's configured side TWEL to proxy metric reducers."""

    outputs = {}
    for side in ("long", "short"):
        if side not in base_limits:
            raise ValueError(f"GPU candidate TWEL context is missing {side}")
        key = f"{side}_total_wallet_exposure_limit"
        values = np.asarray(
            [float(candidate.get(key, base_limits[side])) for candidate in candidates],
            dtype=np.float64,
        )
        outputs[f"candidate_total_wallet_exposure_limit_{side}"] = torch.from_numpy(
            values
        )
    return outputs


def _single_coin_exposure_params(risk: dict, *, side: str) -> dict[str, float]:
    allowance_mode = str(
        risk.get("we_excess_allowance_mode", "bounded")
    ).strip().lower()
    if allowance_mode not in {"bounded", "legacy_raw"}:
        raise ValueError(
            "MPS proxy requires "
            f"bot.{side}.risk.we_excess_allowance_mode to be bounded or "
            f"legacy_raw, got {allowance_mode!r}"
        )
    return {
        "we_excess_allowance_pct": float(
            risk.get("we_excess_allowance_pct", 0.0) or 0.0
        ),
        "we_excess_allowance_legacy_raw": float(
            allowance_mode == "legacy_raw"
        ),
        "twel_entry_gate_enabled": float(
            bool(risk.get("total_exposure_entry_gate_enabled", True))
        ),
        "twel_enforcer_threshold": float(
            risk.get("total_exposure_enforcer_threshold", 1.0) or 0.0
        ),
    }


def _position_exposure_enforcer_params(
    risk: dict, *, side: str
) -> dict[str, float]:
    enabled = bool(
        risk.get(
            "position_exposure_enforcer_enabled",
            risk.get("risk_wel_enforcer_enabled", False),
        )
    )
    threshold = float(
        risk.get(
            "position_exposure_enforcer_threshold",
            risk.get("risk_wel_enforcer_threshold", 0.0),
        )
        or 0.0
    )
    if enabled and (not np.isfinite(threshold) or threshold <= 0.0):
        raise ValueError(
            "MPS proxy requires a finite positive "
            f"bot.{side}.risk.position_exposure_enforcer_threshold when the "
            "position exposure enforcer is enabled"
        )
    return {
        "wel_enforcer_enabled": float(enabled),
        "wel_enforcer_threshold": threshold,
    }


def _total_exposure_enforcer_params(
    risk: dict, *, side: str
) -> dict[str, float]:
    policy = str(
        risk.get("total_exposure_enforcer_policy", "reduce_overweight")
    ).strip().lower()
    if policy not in {"reduce_overweight", "reduce_portfolio"}:
        raise ValueError(
            "MPS proxy requires "
            f"bot.{side}.risk.total_exposure_enforcer_policy to be "
            f"reduce_overweight or reduce_portfolio, got {policy!r}"
        )
    return {
        "twel_enforcer_enabled": float(
            bool(risk.get("total_exposure_enforcer_enabled", False))
        ),
        "twel_enforcer_reduce_portfolio": float(policy == "reduce_portfolio"),
    }


def _unstuck_params(bot: dict) -> dict[str, float]:
    return {
        "unstuck_enabled": float(bool(bot["unstuck_enabled"])),
        "unstuck_ema_gating_enabled": float(
            bool(bot["unstuck_ema_gating_enabled"])
        ),
        "unstuck_close_pct": float(bot["unstuck_close_pct"]),
        "unstuck_ema_dist": float(bot["unstuck_ema_dist"]),
        "unstuck_loss_allowance_pct": float(bot["unstuck_loss_allowance_pct"]),
        "unstuck_threshold": float(bot["unstuck_threshold"]),
    }


def _hsl_params(bot: dict, *, signal_mode: str) -> dict[str, float]:
    restart_policy = str(
        bot.get("hsl_restart_after_red_policy", "threshold")
    ).strip().lower()
    restart_policy_ids = {"always": 0.0, "threshold": 1.0, "never": 2.0}
    if restart_policy not in restart_policy_ids:
        raise ValueError(
            "MPS HSL requires restart_after_red_policy to be always, threshold, "
            f"or never, got {restart_policy!r}"
        )
    signal_mode = str(signal_mode).strip().lower()
    if signal_mode not in {"coin", "pside", "unified"}:
        raise ValueError(
            "MPS HSL requires live.hsl_signal_mode to be coin, pside, or "
            f"unified, got {signal_mode!r}"
        )
    orange_mode = str(
        bot.get("hsl_orange_tier_mode", "tp_only_with_active_entry_cancellation")
    ).strip().lower()
    if orange_mode not in {
        "graceful_stop",
        "tp_only",
        "tp_only_with_active_entry_cancellation",
    }:
        raise ValueError(
            "MPS HSL requires orange_tier_mode to be graceful_stop or tp_only, "
            f"got {orange_mode!r}"
        )
    float32_below_one = float(
        np.nextafter(np.float32(1.0), np.float32(0.0))
    )
    for key, default in (
        ("hsl_red_threshold", 0.15),
        ("hsl_no_restart_drawdown_threshold", 1.0),
    ):
        value = float(bot.get(key, default))
        if float32_below_one < value < 1.0:
            raise ValueError(
                f"MPS HSL cannot represent {key}={value} distinctly from 1.0; "
                f"use <= {float32_below_one} or exactly 1.0"
            )
    return {
        "hsl_enabled": float(bool(bot.get("hsl_enabled", False))),
        "hsl_red_threshold": float(bot.get("hsl_red_threshold", 0.15)),
        "hsl_ema_span_minutes": float(bot.get("hsl_ema_span_minutes", 720.0)),
        "hsl_cooldown_minutes_after_red": float(
            bot.get("hsl_cooldown_minutes_after_red", 0.0)
        ),
        "hsl_no_restart_drawdown_threshold": float(
            bot.get("hsl_no_restart_drawdown_threshold", 1.0)
        ),
        "hsl_restart_policy": restart_policy_ids[restart_policy],
        "hsl_tier_ratio_yellow": float(
            bot.get("hsl_tier_ratio_yellow", 0.5)
        ),
        "hsl_tier_ratio_orange": float(
            bot.get("hsl_tier_ratio_orange", 0.75)
        ),
        "hsl_orange_graceful_stop": float(orange_mode == "graceful_stop"),
        "hsl_signal_coin": float(signal_mode == "coin"),
        # The single-coin GPU search contract pins authoritative candidate
        # n_positions to one even if the source config is outside its bounds.
        "hsl_slot_count": 1.0,
    }


def _require_complete_valid_tail(last_valid_idx: int, candle_count: int) -> None:
    if int(last_valid_idx) != int(candle_count) - 1:
        raise ValueError(
            "GPU foundation requires the final prepared candle to be valid because "
            "the exact Rust backtest force-realizes open positions at its valid tail; "
            f"last_valid_idx={last_valid_idx}, candle_count={candle_count}"
        )


def _require_no_internal_invalid_hsl_candles(
    high, low, close, *, first_valid_idx: int, last_valid_idx: int
) -> None:
    first = max(0, int(first_valid_idx))
    last = min(int(last_valid_idx), len(close) - 1)
    valid = (
        np.isfinite(high[first : last + 1])
        & np.isfinite(low[first : last + 1])
        & np.isfinite(close[first : last + 1])
        & (np.asarray(close[first : last + 1]) > 0.0)
    )
    if not bool(np.all(valid)):
        first_invalid = first + int(np.flatnonzero(~valid)[0])
        raise ValueError(
            "MPS single-coin HSL currently requires contiguous valid candles "
            f"between first and last valid indices; invalid candle at {first_invalid}"
        )


def _nan_min(left, right):
    """Elementwise minimum which preserves the finite operand when only one exists."""

    left_finite = left.isfinite()
    right_finite = right.isfinite()
    return left.where(
        left_finite & ~right_finite,
        right.where(right_finite & ~left_finite, left.minimum(right)),
    )


def _nan_max(left, right):
    """Elementwise maximum which preserves the finite operand when only one exists."""

    left_finite = left.isfinite()
    right_finite = right.isfinite()
    return left.where(
        left_finite & ~right_finite,
        right.where(right_finite & ~left_finite, left.maximum(right)),
    )


def _directional_entry_initial_metrics(side: str, entry_pct):
    """Expand one directional Metal value into the shared two-side surface."""

    if side not in {"long", "short"}:
        raise ValueError(f"expected long or short entry metric side, got {side!r}")
    zeros = entry_pct.new_zeros(entry_pct.shape)
    return {
        "entry_initial_balance_pct_long": entry_pct if side == "long" else zeros,
        "entry_initial_balance_pct_short": entry_pct if side == "short" else zeros,
    }


def _prepared_single_coin_side_enabled(config: dict, side: str, bot: dict) -> bool:
    """Match exact Rust eligibility for the one coin prepared by the payload."""

    if "entry_eligible" not in bot:
        raise ValueError(
            f"GPU single-coin payload for {side} is missing entry_eligible"
        )
    return gpu_side_enabled(config, side) and bool(bot["entry_eligible"])


def _combine_hedged_multicoin_outputs(
    long: dict,
    short: dict,
    starting_balance: float,
    liquidation_threshold: float,
    start_minute_of_day: int,
    interval_ms: int,
):
    """Build a conservative portfolio surface from independent directional screens.

    This is deliberately only a ranking proxy. The unchanged Rust backtest remains
    authoritative for every accepted result and the optimizer's drift gates halt on
    material disagreement.
    """

    active = long["day_min_eq"].isfinite() & short["day_min_eq"].isfinite()
    day_count = int(active.shape[1])
    day_ids = active.new_tensor(range(day_count), dtype=long["liq_step"].dtype)
    no_liquidation = long["liq_step"].new_full(long["liq_step"].shape, day_count)
    directional_liquidation_day = long["liq_step"].where(
        long["liq_step"] >= 0, no_liquidation
    ).minimum(
        short["liq_step"].where(short["liq_step"] >= 0, no_liquidation)
    )
    raw_combined_min = (
        long["day_min_eq"] + short["day_min_eq"] - float(starting_balance)
    )
    raw_combined_min_balance = (
        long["day_min_balance"]
        + short["day_min_balance"]
        - float(starting_balance)
    )
    portfolio_floor = max(0.0, float(starting_balance)) * max(
        0.0, float(liquidation_threshold)
    )
    portfolio_breach = active & (
        (raw_combined_min <= portfolio_floor) | (raw_combined_min_balance <= 0.0)
    )
    portfolio_liquidation_day = portfolio_breach.to(
        dtype=long["liq_step"].dtype
    ).argmax(dim=1).where(
        portfolio_breach.any(dim=1), no_liquidation
    )
    terminal_day = directional_liquidation_day.minimum(portfolio_liquidation_day)
    liquidated = terminal_day < day_count
    liquidation_day = terminal_day.where(
        liquidated, -no_liquidation.new_ones(())
    )
    active &= (~liquidated).unsqueeze(1) | (
        day_ids.unsqueeze(0) < terminal_day.unsqueeze(1)
    )

    combined = {}
    for key in ("day_end_eq", "day_min_eq"):
        values = raw_combined_min if key == "day_min_eq" else (
            long[key] + short[key] - float(starting_balance)
        )
        if key == "day_min_eq":
            values = values.where(active, values.new_full((), float("inf")))
        else:
            values = values.where(active, values.new_zeros(()))
        combined[key] = values

    combined["day_max_dd"] = (
        long["day_max_dd"] + short["day_max_dd"]
    ).clamp(max=1.0).where(active, long["day_max_dd"].new_zeros(()))
    combined["day_volume"] = (long["day_volume"] + short["day_volume"]).where(
        active, long["day_volume"].new_zeros(())
    )
    combined["day_has_fill"] = (
        long["day_has_fill"] | short["day_has_fill"]
    ) & active
    combined["day_net_pnl"] = (
        long["day_net_pnl"] + short["day_net_pnl"]
    ).where(active, long["day_net_pnl"].new_zeros(()))
    combined["day_last_fill_balance"] = (
        long["day_last_fill_balance"]
        + short["day_last_fill_balance"]
        - float(starting_balance)
    ).where(active, long["day_last_fill_balance"].new_zeros(()))
    combined["day_fill_count"] = (
        long["day_fill_count"] + short["day_fill_count"]
    ).where(active, long["day_fill_count"].new_zeros(()))
    combined["max_dd"] = (long["max_dd"] + short["max_dd"]).clamp(max=1.0)
    combined["held_max_ms"] = long["held_max_ms"].maximum(short["held_max_ms"])
    combined["position_unchanged_max_ms"] = long[
        "position_unchanged_max_ms"
    ].maximum(short["position_unchanged_max_ms"])
    combined["gap_hist"] = long["gap_hist"] + short["gap_hist"]
    combined["gap_max_ms"] = long["gap_max_ms"].maximum(short["gap_max_ms"])
    combined["first_fill_ts"] = _nan_min(
        long["first_fill_ts"], short["first_fill_ts"]
    )
    combined["last_fill_ts"] = _nan_max(
        long["last_fill_ts"], short["last_fill_ts"]
    )
    combined["recovery_max_ms"] = long["recovery_max_ms"].maximum(
        short["recovery_max_ms"]
    )
    # The earlier directional high produces the longer, safer final-recovery estimate.
    combined["last_high_ts"] = _nan_min(
        long["last_high_ts"], short["last_high_ts"]
    )
    combined["first_eq_ts"] = _nan_max(
        long["first_eq_ts"], short["first_eq_ts"]
    )
    last_eq_ts = _nan_min(long["last_eq_ts"], short["last_eq_ts"])
    # Daily summaries do not reveal the exact intra-day portfolio breach. Stop at
    # the final complete candle before that UTC day so completion cannot imply
    # coverage beyond the conservative combined-equity liquidation surface.
    terminal_day_start_ms = (
        terminal_day.to(last_eq_ts.dtype) * 86_400_000.0
        - float(start_minute_of_day) * 60_000.0
    ).clamp(min=0.0)
    complete_tail_ms = (terminal_day_start_ms - float(interval_ms)).clamp(min=0.0)
    first_eq_ts = combined["first_eq_ts"]
    complete_tail_ms = complete_tail_ms.maximum(first_eq_ts).where(
        first_eq_ts.isfinite(), complete_tail_ms
    )
    combined["last_eq_ts"] = last_eq_ts.minimum(complete_tail_ms).where(
        liquidated, last_eq_ts
    )

    combined["liq_step"] = liquidation_day
    combined["profit_sum"] = long["profit_sum"] + short["profit_sum"]
    combined["loss_sum"] = long["loss_sum"] + short["loss_sum"]
    return combined


class MpsSingleCoinProxy:
    """Batched directional screening proxy for supported single-coin strategies."""

    def __init__(
        self,
        *,
        config: dict,
        hlcvs: np.ndarray,
        mss: dict,
        btc: np.ndarray,
        timestamps: np.ndarray,
        exchange: str,
        batch_size: int,
        needed_metrics,
    ):
        try:
            import torch
        except (
            ModuleNotFoundError
        ) as exc:  # pragma: no cover - optional dependency path
            raise ModuleNotFoundError(
                "GPU optimization requires the optional 'gpu-mps' dependencies; "
                "install Passivbot with `pip install -e '.[full,gpu-mps]'`"
            ) from exc
        if not torch.backends.mps.is_available():
            raise RuntimeError(
                "GPU optimization requested but Apple MPS is unavailable in this process"
            )

        from backtest import build_backtest_payload
        from optimization.gpu.metrics import compute_objectives
        from optimization.gpu.mps_kernel import (
            MpsEmaAnchorRunner,
            MpsTrailingMartingaleRunner,
        )

        self._torch = torch
        self._compute_objectives = compute_objectives
        self.needed_metrics = set(needed_metrics)
        self.batch_size = max(1, int(batch_size))
        self.profile_enabled = os.environ.get(
            "PASSIVBOT_GPU_PROFILE", ""
        ).strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
        }
        self.strategy_kind = str(
            config.get("live", {}).get("strategy_kind", "")
        ).strip().lower()
        if self.strategy_kind not in GPU_STRATEGY_PARAM_KEYS:
            raise ValueError(
                "MPS single-coin proxy supports ema_anchor or "
                f"trailing_martingale, got {self.strategy_kind!r}"
            )
        self.param_keys = GPU_STRATEGY_PARAM_KEYS[self.strategy_kind]

        payload = build_backtest_payload(
            np.ascontiguousarray(hlcvs),
            mss,
            copy.deepcopy(config),
            exchange,
            np.ascontiguousarray(btc),
            timestamps,
            metrics_only=True,
            skip_btc_analysis=True,
        )
        if len(payload.bot_params_list) != 1:
            raise ValueError(
                "GPU foundation supports exactly one backtest coin; "
                f"prepared {len(payload.bot_params_list)}"
            )
        backtest_params = payload.backtest_params
        if int(backtest_params.get("candle_interval_minutes", 1)) != 1:
            raise ValueError(
                "GPU foundation currently supports one-minute candles only"
            )
        _require_complete_valid_tail(
            int(backtest_params["last_valid_indices"][0]), len(hlcvs)
        )

        long_bot = payload.bot_params_list[0]["long"]
        short_bot = payload.bot_params_list[0]["short"]
        self.enabled = {
            side: _prepared_single_coin_side_enabled(config, side, bot)
            for side, bot in (("long", long_bot), ("short", short_bot))
        }
        if not any(self.enabled.values()):
            raise ValueError("GPU foundation requires at least one enabled side")
        hsl_enabled_sides = [
            side
            for side, bot in (("long", long_bot), ("short", short_bot))
            if self.enabled[side] and bool(bot.get("hsl_enabled"))
        ]
        if hsl_enabled_sides and sum(self.enabled.values()) != 1:
            raise ValueError(
                "MPS single-coin HSL currently requires exactly one enabled side"
            )
        signal_mode = (
            backtest_params.get("equity_hard_stop_loss", {})
            .get("signal_mode", "unified")
        )
        hsl_panic_market = {}
        self.base_params = {}
        for side, bot in (("long", long_bot), ("short", short_bot)):
            panic_close_order_type = str(
                bot.get("hsl_panic_close_order_type", "limit")
            ).strip().lower()
            if panic_close_order_type not in {"limit", "market"}:
                raise ValueError(
                    f"MPS single-coin HSL requires bot.{side}.hsl."
                    "panic_close_order_type to be limit or market, got "
                    f"{panic_close_order_type!r}"
                )
            hsl_panic_market[side] = (
                self.enabled[side]
                and bool(bot.get("hsl_enabled"))
                and panic_close_order_type == "market"
            )
            strategy = dict(payload.strategy_params_list[0][side])
            risk = config["bot"][side]["risk"]
            if self.strategy_kind == "trailing_martingale":
                strategy = flatten_trailing_martingale_params(strategy, risk)
            else:
                strategy["entry_cooldown_minutes"] = float(
                    risk.get("entry_cooldown_minutes", 0.0) or 0.0
                )
                strategy["total_wallet_exposure_limit"] = float(
                    risk["total_wallet_exposure_limit"]
                )
            strategy.update(_single_coin_exposure_params(risk, side=side))
            if self.strategy_kind == "trailing_martingale":
                strategy.update(
                    _position_exposure_enforcer_params(risk, side=side)
                )
            if self.strategy_kind in {"ema_anchor", "trailing_martingale"}:
                strategy.update(
                    _total_exposure_enforcer_params(risk, side=side)
                )
                strategy.update(_unstuck_params(bot))
                strategy.update(_hsl_params(bot, signal_mode=signal_mode))
            missing = [key for key in self.param_keys if key not in strategy]
            if missing:
                raise ValueError(
                    f"GPU {self.strategy_kind} payload for {side} is missing "
                    f"parameters: {missing}"
                )
            self.base_params[side] = strategy

        self.base_total_wallet_exposure_limits = {
            side: float(self.base_params[side]["total_wallet_exposure_limit"])
            for side in ("long", "short")
        }

        market_params = payload.exchange_params[0]
        self.market = ProxyMarket(
            qty_step=float(market_params["qty_step"]),
            price_step=float(market_params["price_step"]),
            min_qty=float(market_params["min_qty"]),
            min_cost=float(market_params["min_cost"]),
            c_mult=float(market_params["c_mult"]),
            maker_fee=float(market_params["maker_fee"]),
        )
        interval_ms = int(backtest_params["candle_interval_minutes"]) * 60_000
        self.run = ProxyRun(
            starting_balance=float(backtest_params["starting_balance"]),
            warmup_bars=max(1, int(backtest_params.get("global_warmup_bars", 0) or 1)),
            trade_start_idx=int(backtest_params["trade_start_indices"][0]),
            requested_start_ts_ms=int(
                backtest_params["requested_start_timestamp_ms"]
            ),
            guard_ts_ms=int(
                max(
                    backtest_params["requested_start_timestamp_ms"],
                    backtest_params["first_timestamp_ms"],
                )
            ),
            first_ts_ms=int(backtest_params["first_timestamp_ms"]),
            interval_ms=interval_ms,
            liquidation_threshold=float(
                backtest_params.get("liquidation_threshold", 0.05)
            ),
            first_valid_idx=int(backtest_params["first_valid_indices"][0]),
            last_valid_idx=int(backtest_params["last_valid_indices"][0]),
        )

        high = hlcvs[:, 0, 0].astype(np.float64)
        low = hlcvs[:, 0, 1].astype(np.float64)
        close = hlcvs[:, 0, 2].astype(np.float64)
        if hsl_enabled_sides:
            _require_no_internal_invalid_hsl_candles(
                high,
                low,
                close,
                first_valid_idx=self.run.first_valid_idx,
                last_valid_idx=self.run.last_valid_idx,
            )
        self.data = build_mps_data(high, low, close, timestamps, self.run, self.market)
        self.metrics_data = {"ts0": self.data["ts0"], "n": self.data["n"]}
        runner_cls = (
            MpsTrailingMartingaleRunner
            if self.strategy_kind == "trailing_martingale"
            else MpsEmaAnchorRunner
        )
        self.runner = runner_cls(
            self.market,
            self.run,
            self.data,
            long_enabled=self.enabled["long"],
            short_enabled=self.enabled["short"],
            hedge_mode=bool(backtest_params["hedge_mode"]),
            filter_by_min_effective_cost=bool(
                backtest_params["filter_by_min_effective_cost"]
            ),
            max_realized_loss_pct=float(
                backtest_params.get("max_realized_loss_pct", 1.0)
            ),
            taker_fee=float(market_params["taker_fee"]),
            market_order_slippage_pct=float(
                backtest_params.get("market_order_slippage_pct", 0.0)
            ),
            hsl_panic_market_long=hsl_panic_market["long"],
            hsl_panic_market_short=hsl_panic_market["short"],
        )

    def _parameter_matrix(self, candidates: list[dict]) -> np.ndarray:
        rows = []
        for candidate in candidates:
            row = []
            for side in ("long", "short"):
                merged = dict(self.base_params[side])
                merged.update(
                    {
                        key.removeprefix(f"{side}_"): value
                        for key, value in candidate.items()
                        if key.startswith(f"{side}_")
                    }
                )
                row.extend(float(merged[key]) for key in self.param_keys)
            rows.append(row)
        return np.asarray(rows, dtype=np.float64)

    def evaluate(self, candidates: list[dict]) -> list[dict]:
        results: list[dict] = []
        torch = self._torch
        for start in range(0, len(candidates), self.batch_size):
            chunk = candidates[start : start + self.batch_size]
            parameter_matrix = self._parameter_matrix(chunk)
            output = self.runner.run(
                parameter_matrix,
                profile=self.profile_enabled,
            )
            output = {
                key: value.cpu()
                for key, value in output.items()
                if key in CORE_OUTPUT_KEYS | DIRECTIONAL_HSL_OUTPUT_KEYS
            }
            timestamp_origin = float(self.metrics_data["ts0"])
            for key in (
                "first_fill_ts",
                "last_fill_ts",
                "last_high_ts",
                "first_eq_ts",
                "last_eq_ts",
            ):
                values = output[key].to(torch.float64)
                output[key] = torch.where(
                    torch.isfinite(values), values + timestamp_origin, values
                )
            if any("_per_exposure_" in name for name in self.needed_metrics):
                output.update(
                    _candidate_wallet_exposure_limit_outputs(
                        chunk,
                        self.base_total_wallet_exposure_limits,
                        torch=torch,
                    )
                )
            objectives = self._compute_objectives(
                output,
                self.run,
                self.metrics_data,
                needed=self.needed_metrics,
            )
            arrays = {
                name: value.detach().cpu().numpy() for name, value in objectives.items()
            }
            results.extend(
                {name: float(values[index]) for name, values in arrays.items()}
                for index in range(len(chunk))
            )
        return results


# Compatibility name retained for downstream imports from the EMA foundation PR.
MpsEmaAnchorProxy = MpsSingleCoinProxy


def _build_multicoin_ema_coin_overrides(
    *,
    config: dict,
    mss: dict,
    exchange: str,
    coins: list[str],
    payload,
    side: str,
    resolve_override=None,
) -> tuple[np.ndarray, dict]:
    """Pack exact-last static coin overrides for the Metal EMA proxy."""

    if resolve_override is None:
        from backtest import _get_backtest_coin_override

        resolve_override = _get_backtest_coin_override

    override_keys = tuple(EMA_ANCHOR_PARAM_KEYS[:-2])
    matrix = np.full((len(coins), 19), np.nan, dtype=np.float32)
    for coin_index, coin in enumerate(coins):
        patch = resolve_override(config, mss, exchange, coin) or {}
        side_patch = patch.get("bot", {}).get(side, {})
        strategy_patch = side_patch.get("strategy", {}).get("ema_anchor", {}) or {}
        effective_strategy = payload.strategy_params_list[coin_index][side]
        effective_bot = payload.bot_params_list[coin_index][side]
        for column, key in enumerate(override_keys):
            if key in strategy_patch:
                matrix[coin_index, column] = float(effective_strategy[key])
        risk_patch = side_patch.get("risk", {}) or {}
        if "entry_cooldown_minutes" in risk_patch:
            matrix[coin_index, 10] = float(
                effective_bot.get("entry_cooldown_minutes", 0.0) or 0.0
            )
        if "wallet_exposure_limit" in side_patch:
            matrix[coin_index, 11] = float(effective_bot["wallet_exposure_limit"])
        if "we_excess_allowance_pct" in risk_patch:
            matrix[coin_index, 12] = float(
                effective_bot.get("risk_we_excess_allowance_pct", 0.0) or 0.0
            )
        unstuck_patch = side_patch.get("unstuck", {}) or {}
        for offset, (patch_key, bot_key) in enumerate(
            (
                ("enabled", "unstuck_enabled"),
                ("ema_gating_enabled", "unstuck_ema_gating_enabled"),
                ("close_pct", "unstuck_close_pct"),
                ("ema_dist", "unstuck_ema_dist"),
                ("loss_allowance_pct", "unstuck_loss_allowance_pct"),
                ("threshold", "unstuck_threshold"),
            ),
            start=13,
        ):
            if patch_key in unstuck_patch:
                matrix[coin_index, offset] = float(effective_bot[bot_key])
    contract = {
        "exchange": exchange,
        "coins": coins,
        "side": side,
        "values": [
            [None if not np.isfinite(value) else float(value) for value in row]
            for row in matrix
        ],
    }
    return matrix, contract


def _build_multicoin_tm_coin_overrides(
    *,
    config: dict,
    mss: dict,
    exchange: str,
    coins: list[str],
    payload,
    side: str,
    resolve_override=None,
) -> tuple[np.ndarray, dict]:
    """Pack exact-last static coin overrides for the Metal TM proxy."""

    if resolve_override is None:
        from backtest import _get_backtest_coin_override

        resolve_override = _get_backtest_coin_override

    cooldown_column = len(TRAILING_MARTINGALE_COIN_OVERRIDE_PATHS)
    wallet_exposure_column = cooldown_column + 1
    allowance_pct_column = wallet_exposure_column + 1
    wel_enforcer_enabled_column = allowance_pct_column + 1
    wel_enforcer_threshold_column = wel_enforcer_enabled_column + 1
    unstuck_start_column = wel_enforcer_threshold_column + 1
    matrix = np.full(
        (len(coins), unstuck_start_column + 6),
        np.nan,
        dtype=np.float32,
    )
    missing = object()
    for coin_index, coin in enumerate(coins):
        patch = resolve_override(config, mss, exchange, coin) or {}
        side_patch = patch.get("bot", {}).get(side, {})
        strategy_patch = (
            side_patch.get("strategy", {}).get("trailing_martingale", {}) or {}
        )
        effective_strategy = flatten_trailing_martingale_params(
            payload.strategy_params_list[coin_index][side],
            payload.bot_params_list[coin_index][side],
        )
        effective_bot = payload.bot_params_list[coin_index][side]
        for column, (key, path) in enumerate(
            TRAILING_MARTINGALE_COIN_OVERRIDE_PATHS
        ):
            value = strategy_patch
            for part in path:
                value = (
                    value.get(part, missing) if isinstance(value, dict) else missing
                )
                if value is missing:
                    break
            if value is not missing:
                matrix[coin_index, column] = float(effective_strategy[key])
        risk_patch = side_patch.get("risk", {}) or {}
        if "entry_cooldown_minutes" in risk_patch:
            matrix[coin_index, cooldown_column] = float(
                effective_bot.get("entry_cooldown_minutes", 0.0) or 0.0
            )
        if "wallet_exposure_limit" in side_patch:
            matrix[coin_index, wallet_exposure_column] = float(
                effective_bot["wallet_exposure_limit"]
            )
        if "we_excess_allowance_pct" in risk_patch:
            matrix[coin_index, allowance_pct_column] = float(
                effective_bot.get("risk_we_excess_allowance_pct", 0.0) or 0.0
            )
        if "position_exposure_enforcer_enabled" in risk_patch:
            matrix[coin_index, wel_enforcer_enabled_column] = float(
                bool(effective_bot.get("risk_wel_enforcer_enabled", False))
            )
        if "position_exposure_enforcer_threshold" in risk_patch:
            matrix[coin_index, wel_enforcer_threshold_column] = float(
                effective_bot.get("risk_wel_enforcer_threshold", 0.0) or 0.0
            )
        unstuck_patch = side_patch.get("unstuck", {}) or {}
        for offset, (patch_key, bot_key) in enumerate(
            (
                ("enabled", "unstuck_enabled"),
                ("ema_gating_enabled", "unstuck_ema_gating_enabled"),
                ("close_pct", "unstuck_close_pct"),
                ("ema_dist", "unstuck_ema_dist"),
                ("loss_allowance_pct", "unstuck_loss_allowance_pct"),
                ("threshold", "unstuck_threshold"),
            ),
            start=unstuck_start_column,
        ):
            if patch_key in unstuck_patch:
                matrix[coin_index, offset] = float(effective_bot[bot_key])
    contract = {
        "exchange": exchange,
        "coins": coins,
        "side": side,
        "values": [
            [None if not np.isfinite(value) else float(value) for value in row]
            for row in matrix
        ],
    }
    return matrix, contract


class MpsMulticoinProxy:
    """Batched multi-coin MPS proxy for the supported strategy topology."""

    def __init__(
        self,
        *,
        config: dict,
        hlcvs: np.ndarray,
        mss: dict,
        btc: np.ndarray,
        timestamps: np.ndarray,
        exchange: str,
        batch_size: int,
        needed_metrics,
    ):
        try:
            import torch
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
            raise ModuleNotFoundError(
                "GPU optimization requires the optional 'gpu-mps' dependencies; "
                "install Passivbot with `pip install -e '.[full,gpu-mps]'`"
            ) from exc
        if not torch.backends.mps.is_available():
            raise RuntimeError(
                "GPU optimization requested but Apple MPS is unavailable in this process"
            )

        from backtest import build_backtest_payload
        from optimization.gpu.metrics import compute_objectives
        from optimization.gpu.mps_kernel import (
            MpsEmaAnchorMulticoinRunner,
            MpsTrailingMartingaleMulticoinRunner,
        )

        self._torch = torch
        self._compute_objectives = compute_objectives
        self.needed_metrics = set(needed_metrics)
        self.batch_size = max(1, int(batch_size))
        self.profile_enabled = os.environ.get(
            "PASSIVBOT_GPU_PROFILE", ""
        ).strip().lower() in {"1", "true", "yes", "y"}

        values = np.asarray(hlcvs)
        if values.ndim != 3:
            raise ValueError(
                "expected multicoin HLCVs with three dimensions, "
                f"got {values.shape}"
            )
        coin_count = int(values.shape[1])
        if not (2 <= coin_count <= MPS_MULTICOIN_MAX_COINS):
            raise ValueError(
                f"MPS multicoin proxy supports 2..{MPS_MULTICOIN_MAX_COINS} coins; "
                f"got {coin_count}"
            )
        self.strategy_kind = (
            str(config.get("live", {}).get("strategy_kind", "")).strip().lower()
        )
        if self.strategy_kind not in {"ema_anchor", "trailing_martingale"}:
            raise ValueError(
                "MPS multicoin proxy supports ema_anchor or "
                f"trailing_martingale, got {self.strategy_kind!r}"
            )
        self.param_keys = (
            TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS
            if self.strategy_kind == "trailing_martingale"
            else EMA_ANCHOR_MULTICOIN_PARAM_KEYS
        )
        enabled_sides = [
            side for side in ("long", "short") if gpu_side_enabled(config, side)
        ]
        if len(enabled_sides) not in (1, 2):
            raise ValueError(
                "MPS multicoin proxy requires one or two enabled sides"
            )
        self.sides = enabled_sides
        _require_multicoin_metric_topology(self.sides, self.needed_metrics)
        if len(enabled_sides) == 2 and not bool(
            config.get("live", {}).get("hedge_mode")
        ):
            raise ValueError(
                "MPS dual-side multicoin proxy currently requires live.hedge_mode=true; "
                "one-way arbitration is not modeled"
            )
        if len(enabled_sides) == 2:
            approved = config.get("live", {}).get("approved_coins", {}) or {}
            ignored = config.get("live", {}).get("ignored_coins", {}) or {}
            for label, values_by_side in (
                ("approved", approved),
                ("ignored", ignored),
            ):
                if set(values_by_side.get("long", []) or []) != set(
                    values_by_side.get("short", []) or []
                ):
                    raise ValueError(
                        "MPS dual-side multicoin proxy requires matching "
                        f"long/short {label}_coins"
                    )

        payload = build_backtest_payload(
            np.ascontiguousarray(values),
            mss,
            copy.deepcopy(config),
            exchange,
            np.ascontiguousarray(btc),
            timestamps,
            metrics_only=True,
            skip_btc_analysis=True,
        )
        if not (
            len(payload.bot_params_list)
            == len(payload.strategy_params_list)
            == len(payload.exchange_params)
            == coin_count
        ):
            raise ValueError(
                "MPS multicoin payload length disagrees with prepared coin count: "
                f"coins={coin_count}, bots={len(payload.bot_params_list)}, "
                f"strategies={len(payload.strategy_params_list)}, "
                f"markets={len(payload.exchange_params)}"
            )
        backtest_params = payload.backtest_params
        if int(backtest_params.get("candle_interval_minutes", 1)) != 1:
            raise ValueError("MPS multicoin proxy currently supports one-minute candles only")
        if not bool(backtest_params.get("dynamic_wel_by_tradability")):
            raise ValueError(
                "MPS multicoin proxy requires backtest.dynamic_wel_by_tradability=true"
            )
        self.forager_score_hysteresis_pct = float(
            backtest_params.get("forager_score_hysteresis_pct", 0.0) or 0.0
        )
        if not np.isfinite(self.forager_score_hysteresis_pct) or (
            self.forager_score_hysteresis_pct < 0.0
        ):
            raise ValueError(
                "MPS multicoin proxy requires a finite non-negative "
                "live.forager_score_hysteresis_pct"
            )
        for last_valid_idx in backtest_params["last_valid_indices"]:
            _require_complete_valid_tail(int(last_valid_idx), len(values))

        comparable_bot_keys = (
            "total_wallet_exposure_limit",
            "filter_volume_ema_span_1m",
            "filter_volatility_ema_span_1m",
            "filter_volume_drop_pct",
            "forager_score_weights",
            "n_positions",
            "risk_twel_entry_gate_enabled",
            "risk_twel_enforcer_enabled",
            "risk_twel_enforcer_policy",
            "risk_twel_enforcer_threshold",
            "hsl_enabled",
        )
        self.base_total_wallet_exposure_limits = {
            side: float(
                payload.bot_params_list[0][side]["total_wallet_exposure_limit"]
            )
            for side in ("long", "short")
        }
        self.base_params = {}
        for side in self.sides:
            first_bot = payload.bot_params_list[0][side]
            first_strategy = dict(payload.strategy_params_list[0][side])
            if bool(first_bot.get("hsl_enabled")):
                raise ValueError(
                    f"MPS multicoin proxy requires {side} HSL disabled"
                )
            if len(self.sides) == 2 and any(
                bool(item[side].get("unstuck_enabled"))
                for item in payload.bot_params_list
            ):
                raise ValueError(
                    "MPS dual-side multicoin auto-unstuck requires a shared-balance "
                    "portfolio kernel"
                )
            if self.strategy_kind == "trailing_martingale":
                first_strategy = flatten_trailing_martingale_params(
                    first_strategy, first_bot
                )
            weights = first_bot.get("forager_score_weights", {}) or {}
            first_strategy.update(
                {
                    "entry_cooldown_minutes": float(
                        first_bot.get("entry_cooldown_minutes", 0.0) or 0.0
                    ),
                    "total_wallet_exposure_limit": float(
                        first_bot["total_wallet_exposure_limit"]
                    ),
                    "forager_volume_ema_span_1m": float(
                        first_bot.get("filter_volume_ema_span_1m", 0.0) or 0.0
                    ),
                    "forager_volatility_ema_span_1m": float(
                        first_bot.get("filter_volatility_ema_span_1m", 0.0) or 0.0
                    ),
                    "forager_volume_drop_pct": float(
                        first_bot.get("filter_volume_drop_pct", 0.0) or 0.0
                    ),
                    "forager_score_weights_volume": float(
                        weights.get("volume", 0.0)
                    ),
                    "forager_score_weights_ema_readiness": float(
                        weights.get("ema_readiness", 0.0)
                    ),
                    "forager_score_weights_volatility": float(
                        weights.get("volatility", 0.0)
                    ),
                    "n_positions": float(first_bot["n_positions"]),
                }
            )
            first_strategy.update(
                _single_coin_exposure_params(
                    config["bot"][side].get("risk", {}), side=side
                )
            )
            if self.strategy_kind == "trailing_martingale":
                first_strategy.update(
                    _position_exposure_enforcer_params(
                        config["bot"][side].get("risk", {}), side=side
                    )
                )
            if self.strategy_kind in {"ema_anchor", "trailing_martingale"}:
                first_strategy.update(
                    _total_exposure_enforcer_params(
                        config["bot"][side].get("risk", {}), side=side
                    )
                )
            first_strategy.update(_unstuck_params(config["bot"][side]))
            missing = [
                key for key in self.param_keys if key not in first_strategy
            ]
            if missing:
                raise ValueError(
                    f"MPS multicoin {self.strategy_kind} {side} payload is "
                    f"missing parameters: {missing}"
                )
            self.base_params[side] = first_strategy

            for coin in range(1, coin_count):
                bot = payload.bot_params_list[coin][side]
                if any(
                    bot.get(key) != first_bot.get(key)
                    for key in comparable_bot_keys
                ):
                    raise ValueError(
                        "MPS multicoin proxy requires identical global "
                        f"{side} forager/risk settings across coins"
                    )

        coins = list(backtest_params.get("coins") or [])
        if len(coins) != coin_count:
            raise ValueError(
                "MPS multicoin payload coin identity disagrees with prepared data: "
                f"coins={coins}, prepared={coin_count}"
            )
        per_side_coin_overrides = {}
        per_side_override_contracts = {}
        for side in self.sides:
            if self.strategy_kind == "ema_anchor":
                overrides, contract = _build_multicoin_ema_coin_overrides(
                    config=config,
                    mss=mss,
                    exchange=exchange,
                    coins=coins,
                    payload=payload,
                    side=side,
                )
            else:
                overrides, contract = _build_multicoin_tm_coin_overrides(
                    config=config,
                    mss=mss,
                    exchange=exchange,
                    coins=coins,
                    payload=payload,
                    side=side,
                )
            per_side_coin_overrides[side] = overrides
            per_side_override_contracts[side] = contract
        if len(self.sides) == 1:
            self.coin_override_contract = per_side_override_contracts[self.sides[0]]
        else:
            self.coin_override_contract = {
                "exchange": exchange,
                "coins": coins,
                "sides": list(self.sides),
                "values_by_side": {
                    side: per_side_override_contracts[side]["values"]
                    for side in self.sides
                },
                "proxy_mode": "independent-side-hedge-v1",
            }
        self.coin_override_contract["forager_score_hysteresis_pct"] = (
            self.forager_score_hysteresis_pct
        )

        markets = [
            ProxyMarket(
                qty_step=float(item["qty_step"]),
                price_step=float(item["price_step"]),
                min_qty=float(item["min_qty"]),
                min_cost=float(item["min_cost"]),
                c_mult=float(item["c_mult"]),
                maker_fee=float(item["maker_fee"]),
            )
            for item in payload.exchange_params
        ]
        interval_ms = int(backtest_params["candle_interval_minutes"]) * 60_000
        runs = [
            ProxyRun(
                starting_balance=float(backtest_params["starting_balance"]),
                warmup_bars=max(
                    1, int(backtest_params.get("global_warmup_bars", 0) or 1)
                ),
                trade_start_idx=int(backtest_params["trade_start_indices"][coin]),
                requested_start_ts_ms=int(
                    backtest_params["requested_start_timestamp_ms"]
                ),
                guard_ts_ms=int(
                    max(
                        backtest_params["requested_start_timestamp_ms"],
                        backtest_params["first_timestamp_ms"],
                    )
                ),
                first_ts_ms=int(backtest_params["first_timestamp_ms"]),
                interval_ms=interval_ms,
                liquidation_threshold=float(
                    backtest_params.get("liquidation_threshold", 0.05)
                ),
                first_valid_idx=int(backtest_params["first_valid_indices"][coin]),
                last_valid_idx=int(backtest_params["last_valid_indices"][coin]),
            )
            for coin in range(coin_count)
        ]
        self.run = replace(
            runs[0],
            first_valid_idx=min(run.first_valid_idx for run in runs),
            last_valid_idx=max(run.last_valid_idx for run in runs),
            trade_start_idx=min(run.trade_start_idx for run in runs),
        )
        self.data = build_mps_multicoin_data(
            values, timestamps, runs=runs, markets=markets
        )
        self.metrics_data = {"ts0": self.data["ts0"], "n": self.data["n"]}
        runner_cls = (
            MpsTrailingMartingaleMulticoinRunner
            if self.strategy_kind == "trailing_martingale"
            else MpsEmaAnchorMulticoinRunner
        )
        self.runners = {}
        for side in self.sides:
            runner_kwargs = {
                "side": side,
                "forager_score_hysteresis_pct": self.forager_score_hysteresis_pct,
                "max_realized_loss_pct": float(
                    backtest_params.get("max_realized_loss_pct", 1.0)
                ),
            }
            runner_kwargs["coin_overrides"] = per_side_coin_overrides[side]
            self.runners[side] = runner_cls(
                self.run,
                self.data,
                **runner_kwargs,
            )

    def _parameter_matrix(
        self, candidates: list[dict], side: str | None = None
    ) -> np.ndarray:
        if side is None:
            if len(self.sides) != 1:
                raise ValueError("side is required for dual-side multicoin parameters")
            side = self.sides[0]
        param_keys = getattr(self, "param_keys", EMA_ANCHOR_MULTICOIN_PARAM_KEYS)
        rows = []
        for candidate in candidates:
            merged = dict(self.base_params[side])
            merged.update(
                {
                    key.removeprefix(f"{side}_"): value
                    for key, value in candidate.items()
                    if key.startswith(f"{side}_")
                }
            )
            rows.append(
                [float(merged[key]) for key in param_keys]
            )
        return np.asarray(rows, dtype=np.float64)

    def evaluate(self, candidates: list[dict]) -> list[dict]:
        results: list[dict] = []
        torch = self._torch
        for start in range(0, len(candidates), self.batch_size):
            chunk = candidates[start : start + self.batch_size]
            parameter_matrices = {
                side: self._parameter_matrix(chunk, side) for side in self.sides
            }
            raw_side_outputs = {
                side: self.runners[side].run(
                    parameter_matrices[side],
                    profile=self.profile_enabled,
                )
                for side in self.sides
            }
            side_outputs = {
                side: {
                    key: value.cpu()
                    for key, value in raw_side_outputs[side].items()
                    if key in CORE_OUTPUT_KEYS
                }
                for side in self.sides
            }
            if len(self.sides) == 1:
                side = self.sides[0]
                output = side_outputs[side]
                output.update(
                    _directional_entry_initial_metrics(
                        side, output["entry_initial_balance_pct"]
                    )
                )
            else:
                output = _combine_hedged_multicoin_outputs(
                    side_outputs["long"],
                    side_outputs["short"],
                    self.run.starting_balance,
                    self.run.liquidation_threshold,
                    self.runners["long"].start_minute_of_day,
                    self.run.interval_ms,
                )
                output["entry_initial_balance_pct_long"] = side_outputs[
                    "long"
                ]["entry_initial_balance_pct"]
                output["entry_initial_balance_pct_short"] = side_outputs[
                    "short"
                ]["entry_initial_balance_pct"]
            timestamp_origin = float(self.metrics_data["ts0"])
            for key in (
                "first_fill_ts",
                "last_fill_ts",
                "last_high_ts",
                "first_eq_ts",
                "last_eq_ts",
            ):
                values = output[key].to(torch.float64)
                output[key] = torch.where(
                    torch.isfinite(values), values + timestamp_origin, values
                )
            if any("_per_exposure_" in name for name in self.needed_metrics):
                output.update(
                    _candidate_wallet_exposure_limit_outputs(
                        chunk,
                        self.base_total_wallet_exposure_limits,
                        torch=torch,
                    )
                )
            objectives = self._compute_objectives(
                output, self.run, self.metrics_data, needed=self.needed_metrics
            )
            arrays = {
                name: value.detach().cpu().numpy()
                for name, value in objectives.items()
            }
            results.extend(
                {name: float(values[index]) for name, values in arrays.items()}
                for index in range(len(chunk))
            )
        return results


# Compatibility alias retained for callers and tests from the EMA-only slices.
MpsMulticoinEmaProxy = MpsMulticoinProxy
