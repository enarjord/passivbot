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
    "max_dd",
    "held_max_ms",
    "gap_hist",
    "gap_max_ms",
    "first_fill_ts",
    "last_fill_ts",
    "recovery_max_ms",
    "last_high_ts",
    "first_eq_ts",
    "last_eq_ts",
    "liq_step",
}


def _require_complete_valid_tail(last_valid_idx: int, candle_count: int) -> None:
    if int(last_valid_idx) != int(candle_count) - 1:
        raise ValueError(
            "GPU foundation requires the final prepared candle to be valid because "
            "the exact Rust backtest force-realizes open positions at its valid tail; "
            f"last_valid_idx={last_valid_idx}, candle_count={candle_count}"
        )


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
            side: gpu_side_enabled(config, side) for side in ("long", "short")
        }
        if not any(self.enabled.values()):
            raise ValueError("GPU foundation requires at least one enabled side")
        self.base_params = {}
        for side, bot in (("long", long_bot), ("short", short_bot)):
            if self.enabled[side] and bool(bot.get("unstuck_enabled")):
                raise ValueError(
                    f"GPU foundation requires bot.{side}.unstuck.enabled=false"
                )
            if self.enabled[side] and bool(bot.get("hsl_enabled")):
                raise ValueError(f"GPU foundation requires bot.{side}.hsl.enabled=false")
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
            missing = [key for key in self.param_keys if key not in strategy]
            if missing:
                raise ValueError(
                    f"GPU {self.strategy_kind} payload for {side} is missing "
                    f"parameters: {missing}"
                )
            self.base_params[side] = strategy

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
            output = self.runner.run(
                self._parameter_matrix(chunk),
                profile=self.profile_enabled,
            )
            output = {
                key: value.cpu()
                for key, value in output.items()
                if key in CORE_OUTPUT_KEYS
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
    matrix = np.full((len(coins), 12), np.nan, dtype=np.float32)
    for coin_index, coin in enumerate(coins):
        patch = resolve_override(config, mss, exchange, coin) or {}
        side_patch = patch.get("bot", {}).get(side, {})
        strategy_patch = side_patch.get("strategy", {}).get("ema_anchor", {}) or {}
        effective_strategy = payload.strategy_params_list[coin_index][side]
        effective_bot = payload.bot_params_list[coin_index][side]
        for column, key in enumerate(override_keys):
            if key in strategy_patch:
                matrix[coin_index, column] = float(effective_strategy[key])
        if "entry_cooldown_minutes" in (side_patch.get("risk", {}) or {}):
            matrix[coin_index, 10] = float(
                effective_bot.get("entry_cooldown_minutes", 0.0) or 0.0
            )
        if "wallet_exposure_limit" in side_patch:
            matrix[coin_index, 11] = float(effective_bot["wallet_exposure_limit"])
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


class MpsMulticoinEmaProxy:
    """Batched single-side multi-coin EMA Anchor screening proxy."""

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
        from optimization.gpu.mps_kernel import MpsEmaAnchorMulticoinRunner

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
        if (
            str(config.get("live", {}).get("strategy_kind", "")).lower()
            != "ema_anchor"
        ):
            raise ValueError("MPS multicoin proxy currently supports ema_anchor only")
        enabled_sides = [
            side for side in ("long", "short") if gpu_side_enabled(config, side)
        ]
        if len(enabled_sides) != 1:
            raise ValueError(
                "MPS multicoin proxy currently requires exactly one enabled side"
            )
        self.side = enabled_sides[0]

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
        if (
            float(backtest_params.get("forager_score_hysteresis_pct", 0.0) or 0.0)
            != 0.0
        ):
            raise ValueError(
                "MPS multicoin proxy requires live.forager_score_hysteresis_pct=0"
            )
        for last_valid_idx in backtest_params["last_valid_indices"]:
            _require_complete_valid_tail(int(last_valid_idx), len(values))

        first_bot = payload.bot_params_list[0][self.side]
        first_strategy = dict(payload.strategy_params_list[0][self.side])
        if bool(first_bot.get("unstuck_enabled")) or bool(first_bot.get("hsl_enabled")):
            raise ValueError(
                f"MPS multicoin proxy requires {self.side} HSL and unstuck disabled"
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
                "forager_score_weights_volume": float(weights.get("volume", 0.0)),
                "forager_score_weights_ema_readiness": float(
                    weights.get("ema_readiness", 0.0)
                ),
                "forager_score_weights_volatility": float(
                    weights.get("volatility", 0.0)
                ),
                "n_positions": float(first_bot["n_positions"]),
            }
        )
        missing = [
            key for key in EMA_ANCHOR_MULTICOIN_PARAM_KEYS if key not in first_strategy
        ]
        if missing:
            raise ValueError(f"MPS multicoin EMA payload is missing parameters: {missing}")
        self.base_params = first_strategy

        comparable_bot_keys = (
            "total_wallet_exposure_limit",
            "filter_volume_ema_span_1m",
            "filter_volatility_ema_span_1m",
            "filter_volume_drop_pct",
            "forager_score_weights",
            "n_positions",
            "unstuck_enabled",
            "hsl_enabled",
        )
        for coin in range(1, coin_count):
            strategy = payload.strategy_params_list[coin][self.side]
            bot = payload.bot_params_list[coin][self.side]
            if any(
                bot.get(key) != first_bot.get(key) for key in comparable_bot_keys
            ):
                raise ValueError(
                    "MPS multicoin proxy requires identical global "
                    f"{self.side} forager/risk settings across coins"
                )

        coins = list(backtest_params.get("coins") or [])
        if len(coins) != coin_count:
            raise ValueError(
                "MPS multicoin payload coin identity disagrees with prepared data: "
                f"coins={coins}, prepared={coin_count}"
            )
        per_coin_overrides, self.coin_override_contract = (
            _build_multicoin_ema_coin_overrides(
                config=config,
                mss=mss,
                exchange=exchange,
                coins=coins,
                payload=payload,
                side=self.side,
            )
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
        self.runner = MpsEmaAnchorMulticoinRunner(
            self.run,
            self.data,
            side=self.side,
            coin_overrides=per_coin_overrides,
        )

    def _parameter_matrix(self, candidates: list[dict]) -> np.ndarray:
        rows = []
        for candidate in candidates:
            merged = dict(self.base_params)
            merged.update(
                {
                    key.removeprefix(f"{self.side}_"): value
                    for key, value in candidate.items()
                    if key.startswith(f"{self.side}_")
                }
            )
            rows.append(
                [float(merged[key]) for key in EMA_ANCHOR_MULTICOIN_PARAM_KEYS]
            )
        return np.asarray(rows, dtype=np.float64)

    def evaluate(self, candidates: list[dict]) -> list[dict]:
        results: list[dict] = []
        torch = self._torch
        for start in range(0, len(candidates), self.batch_size):
            chunk = candidates[start : start + self.batch_size]
            output = self.runner.run(
                self._parameter_matrix(chunk), profile=self.profile_enabled
            )
            output = {
                key: value.cpu()
                for key, value in output.items()
                if key in CORE_OUTPUT_KEYS
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
