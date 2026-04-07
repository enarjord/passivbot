import gzip
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import passivbot_rust as pbr
from config.access import require_config_value
from config.strategy import EMA_ANCHOR_STRATEGY_KIND, get_active_strategy_side, normalize_strategy_kind


EMA_ANCHOR_BID_FILL_TYPES = {"entry_ema_anchor_long", "close_ema_anchor_short"}
EMA_ANCHOR_ASK_FILL_TYPES = {"close_ema_anchor_long", "entry_ema_anchor_short"}


def _load_json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)


def _load_npy(path: Path) -> np.ndarray:
    if path.suffix == ".gz":
        with gzip.open(path, "rb") as f:
            return np.load(f)
    return np.load(path)


def _ensure_ema_anchor_config(config: dict) -> dict:
    normalized = deepcopy(config)
    strategy_kind = normalize_strategy_kind(normalized.get("live", {}).get("strategy_kind"))
    if strategy_kind != EMA_ANCHOR_STRATEGY_KIND:
        raise ValueError(
            f"ema_anchor analysis helpers require live.strategy_kind = {EMA_ANCHOR_STRATEGY_KIND!r}; got {strategy_kind!r}"
        )
    return normalized


def _require_price_step(*, market_settings: dict | None = None, price_step: float | None = None) -> float:
    if price_step is None:
        if not isinstance(market_settings, dict):
            raise ValueError("price_step or market_settings.price_step is required")
        if "price_step" not in market_settings:
            raise KeyError("market_settings missing required key 'price_step'")
        price_step = float(market_settings["price_step"])
    price_step = float(price_step)
    if not np.isfinite(price_step) or price_step <= 0.0:
        raise ValueError(f"price_step must be finite and > 0; got {price_step}")
    return price_step


def _require_side(side: str) -> str:
    normalized = str(side).strip().lower()
    if normalized not in {"long", "short"}:
        raise ValueError(f"side must be 'long' or 'short'; got {side!r}")
    return normalized


def _require_side_mode(side_mode: str) -> str:
    normalized = str(side_mode).strip().lower()
    if normalized not in {"active", "long", "short"}:
        raise ValueError(f"side_mode must be one of {{'active', 'long', 'short'}}; got {side_mode!r}")
    return normalized


def _prepare_candles_df(candles_df: pd.DataFrame) -> pd.DataFrame:
    required = ["timestamp", "high", "low", "close"]
    missing = [col for col in required if col not in candles_df.columns]
    if missing:
        raise KeyError(f"candles_df missing required columns: {missing}")
    out = candles_df.loc[:, required].copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=False)
    for col in ("high", "low", "close"):
        out[col] = pd.to_numeric(out[col], errors="raise")
    out = out.sort_values("timestamp").reset_index(drop=True)
    if out["timestamp"].duplicated().any():
        raise ValueError("candles_df timestamps must be unique")
    return out


def _timestamp_ms(series: pd.Series) -> np.ndarray:
    ts = pd.to_datetime(series, utc=False)
    return (ts.astype("int64") // 1_000_000).astype(np.int64)


def _get_ema_anchor_side_params(config: dict, side: str) -> dict:
    side_cfg = get_active_strategy_side(
        require_config_value(config, f"bot.{side}"),
        strategy_kind=EMA_ANCHOR_STRATEGY_KIND,
        pside=side,
    )
    if not side_cfg:
        raise KeyError(f"config.bot.{side}.strategy.{EMA_ANCHOR_STRATEGY_KIND} is required")
    return side_cfg


def _calc_quote_series(
    candles_df: pd.DataFrame,
    *,
    side: str,
    price_step: float,
    params: dict,
    balances: np.ndarray,
    position_sizes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    timestamps_ms = _timestamp_ms(candles_df["timestamp"])
    bids, asks = pbr.calc_ema_anchor_quote_series_py(
        side,
        np.ascontiguousarray(timestamps_ms),
        np.ascontiguousarray(candles_df["high"].to_numpy(dtype=float)),
        np.ascontiguousarray(candles_df["low"].to_numpy(dtype=float)),
        np.ascontiguousarray(candles_df["close"].to_numpy(dtype=float)),
        np.ascontiguousarray(np.asarray(balances, dtype=float)),
        np.ascontiguousarray(np.asarray(position_sizes, dtype=float)),
        price_step=price_step,
        ema_span_0=float(params["ema_span_0"]),
        ema_span_1=float(params["ema_span_1"]),
        offset=float(params["offset"]),
        offset_volatility_ema_span_minutes=float(params["offset_volatility_ema_span_minutes"]),
        offset_volatility_1m_weight=float(params["offset_volatility_1m_weight"]),
        entry_volatility_ema_span_hours=float(params["entry_volatility_ema_span_hours"]),
        offset_volatility_1h_weight=float(params["offset_volatility_1h_weight"]),
        offset_psize_weight=float(params["offset_psize_weight"]),
    )
    return np.asarray(bids, dtype=float), np.asarray(asks, dtype=float)


def calc_ema_anchor_neutral_bands(
    config: dict,
    candles_df: pd.DataFrame,
    *,
    side: str,
    market_settings: dict | None = None,
    price_step: float | None = None,
    neutral_balance: float = 1.0,
) -> pd.DataFrame:
    config = _ensure_ema_anchor_config(config)
    side = _require_side(side)
    price_step = _require_price_step(market_settings=market_settings, price_step=price_step)
    candles_df = _prepare_candles_df(candles_df)
    params = _get_ema_anchor_side_params(config, side)

    balances = np.full(len(candles_df), float(neutral_balance), dtype=float)
    position_sizes = np.zeros(len(candles_df), dtype=float)
    bids, asks = _calc_quote_series(
        candles_df,
        side=side,
        price_step=price_step,
        params=params,
        balances=balances,
        position_sizes=position_sizes,
    )
    out = candles_df.copy()
    out["bid"] = bids
    out["ask"] = asks
    return out.loc[:, ["timestamp", "high", "low", "close", "bid", "ask"]]


def _prepare_fills_df(fills_df: pd.DataFrame, coin: str | None = None) -> pd.DataFrame:
    if fills_df.empty:
        return pd.DataFrame(
            columns=["timestamp", "coin", "type", "qty", "price", "psize", "pprice", "usd_total_balance"]
        )
    out = fills_df.copy()
    if "coin" in out.columns and coin is not None:
        out = out[out["coin"] == coin].copy()
    required = ["timestamp", "type", "qty", "price", "psize", "pprice", "usd_total_balance"]
    missing = [col for col in required if col not in out.columns]
    if missing:
        raise KeyError(f"fills_df missing required columns: {missing}")
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=False)
    for col in ("qty", "price", "psize", "pprice", "usd_total_balance"):
        out[col] = pd.to_numeric(out[col], errors="raise")
    return out.sort_values("timestamp").reset_index(drop=True)


def _prepare_balance_df(balance_and_equity_df: pd.DataFrame) -> pd.DataFrame:
    if balance_and_equity_df.empty:
        raise ValueError("balance_and_equity_df must not be empty")
    out = balance_and_equity_df.copy()
    ts_col = "timestamp" if "timestamp" in out.columns else out.columns[0]
    out = out.rename(columns={ts_col: "timestamp"})
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=False)
    for col in ("usd_total_balance", "usd_total_equity"):
        if col not in out.columns:
            raise KeyError(f"balance_and_equity_df missing required column {col!r}")
        out[col] = pd.to_numeric(out[col], errors="raise")
    return out.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)


def _build_fill_state_series(
    candles_df: pd.DataFrame,
    fills_df: pd.DataFrame,
    initial_balance: float,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    candle_index = pd.DatetimeIndex(candles_df["timestamp"])
    if fills_df.empty:
        zeros = pd.Series(0.0, index=candle_index)
        balances = pd.Series(float(initial_balance), index=candle_index)
        return balances, zeros.copy(), zeros.copy(), zeros.copy(), zeros.copy()

    grouped = fills_df.groupby("timestamp", sort=True).agg(
        psize=("psize", "last"),
        pprice=("pprice", "last"),
        usd_total_balance=("usd_total_balance", "last"),
    )
    state = grouped.reindex(candle_index).ffill()
    state = state.shift(1)
    state["psize"] = state["psize"].fillna(0.0)
    state["pprice"] = state["pprice"].fillna(0.0)
    state["usd_total_balance"] = state["usd_total_balance"].fillna(float(initial_balance))

    bid_fill_qty = (
        fills_df[fills_df["type"].isin(EMA_ANCHOR_BID_FILL_TYPES)]
        .groupby("timestamp")["qty"]
        .sum()
        .reindex(candle_index, fill_value=0.0)
    )
    ask_fill_qty = (
        fills_df[fills_df["type"].isin(EMA_ANCHOR_ASK_FILL_TYPES)]
        .groupby("timestamp")["qty"]
        .sum()
        .reindex(candle_index, fill_value=0.0)
    )
    return (
        state["usd_total_balance"],
        state["psize"],
        state["pprice"],
        bid_fill_qty,
        ask_fill_qty,
    )


def calc_ema_anchor_inventory_bands(
    config: dict,
    candles_df: pd.DataFrame,
    fills_df: pd.DataFrame,
    balance_and_equity_df: pd.DataFrame,
    *,
    market_settings: dict | None = None,
    price_step: float | None = None,
    coin: str | None = None,
    side_mode: str = "active",
) -> pd.DataFrame:
    config = _ensure_ema_anchor_config(config)
    side_mode = _require_side_mode(side_mode)
    price_step = _require_price_step(market_settings=market_settings, price_step=price_step)
    candles_df = _prepare_candles_df(candles_df)
    fills_df = _prepare_fills_df(fills_df, coin=coin)
    balance_df = _prepare_balance_df(balance_and_equity_df)

    candle_index = pd.DatetimeIndex(candles_df["timestamp"])
    equity_state = (
        balance_df.set_index("timestamp")
        .reindex(candle_index)
        .ffill()
        .rename_axis("timestamp")
        .reset_index()
    )
    initial_balance = float(balance_df["usd_total_balance"].iloc[0])
    balances, psize, pprice, fill_qty_bid, fill_qty_ask = _build_fill_state_series(
        candles_df, fills_df, initial_balance
    )

    long_params = _get_ema_anchor_side_params(config, "long")
    short_params = _get_ema_anchor_side_params(config, "short")
    psize_long = psize.clip(lower=0.0).to_numpy(dtype=float)
    psize_short = psize.clip(upper=0.0).to_numpy(dtype=float)
    balances_np = balances.to_numpy(dtype=float)

    bid_long, ask_long = _calc_quote_series(
        candles_df,
        side="long",
        price_step=price_step,
        params=long_params,
        balances=balances_np,
        position_sizes=psize_long,
    )
    bid_short, ask_short = _calc_quote_series(
        candles_df,
        side="short",
        price_step=price_step,
        params=short_params,
        balances=balances_np,
        position_sizes=psize_short,
    )

    out = candles_df.copy()
    out["balance"] = balances.to_numpy(dtype=float)
    out["equity"] = equity_state["usd_total_equity"].to_numpy(dtype=float)
    out["psize"] = psize.to_numpy(dtype=float)
    out["pprice"] = pprice.to_numpy(dtype=float)
    out["bid_long"] = bid_long
    out["ask_long"] = ask_long
    out["bid_short"] = bid_short
    out["ask_short"] = ask_short
    out["fill_qty_bid"] = fill_qty_bid.to_numpy(dtype=float)
    out["fill_qty_ask"] = fill_qty_ask.to_numpy(dtype=float)

    if side_mode == "long":
        out["bid"] = out["bid_long"]
        out["ask"] = out["ask_long"]
    elif side_mode == "short":
        out["bid"] = out["bid_short"]
        out["ask"] = out["ask_short"]
    else:
        psize_values = out["psize"].to_numpy(dtype=float)
        out["bid"] = np.where(psize_values < 0.0, out["bid_short"], out["bid_long"])
        out["ask"] = np.where(psize_values > 0.0, out["ask_long"], out["ask_short"])

    return out.loc[
        :,
        [
            "timestamp",
            "high",
            "low",
            "close",
            "balance",
            "equity",
            "psize",
            "pprice",
            "bid_long",
            "ask_long",
            "bid_short",
            "ask_short",
            "bid",
            "ask",
            "fill_qty_bid",
            "fill_qty_ask",
        ],
    ]


def load_backtest_artifact_dataset(artifact_dir: str | Path, *, coin: str | None = None) -> dict:
    artifact_dir = Path(artifact_dir)
    dataset_meta = _load_json(artifact_dir / "dataset.json")
    config = _load_json(artifact_dir / "config.json")
    fills_df = pd.read_csv(artifact_dir / "fills.csv")
    balance_df = pd.read_csv(artifact_dir / "balance_and_equity.csv.gz")

    coins = list(dataset_meta["coins"])
    if not coins:
        raise ValueError("dataset metadata contains no coins")
    if coin is None:
        if len(coins) != 1:
            raise ValueError(
                f"artifact contains multiple coins {coins}; coin=... is required to load candles"
            )
        coin = coins[0]
    if coin not in dataset_meta["coin_index"]:
        raise KeyError(f"coin {coin!r} not present in artifact dataset coins {coins}")

    hlcvs = _load_npy(Path(dataset_meta["hlcvs_file"]))
    timestamps = _load_npy(Path(dataset_meta["timestamps_file"]))
    if hlcvs.ndim != 3 or hlcvs.shape[2] < 3:
        raise ValueError(f"expected hlcvs array shape (T, N, 4+), got {hlcvs.shape}")
    idx = int(dataset_meta["coin_index"][coin])
    coin_slice = hlcvs[:, idx, :]
    candles_df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(timestamps.astype(np.int64), unit="ms"),
            "high": coin_slice[:, 0],
            "low": coin_slice[:, 1],
            "close": coin_slice[:, 2],
        }
    )
    market_settings = _load_json(Path(dataset_meta["market_specific_settings_file"]))
    if coin not in market_settings:
        raise KeyError(f"market_specific_settings missing coin {coin!r}")
    return {
        "artifact_dir": artifact_dir,
        "dataset": dataset_meta,
        "config": config,
        "coin": coin,
        "candles": candles_df,
        "fills": fills_df,
        "balance_and_equity": balance_df,
        "market_settings": market_settings[coin],
    }


def calc_ema_anchor_neutral_bands_from_artifact(
    artifact_dir: str | Path,
    *,
    coin: str | None = None,
    side: str = "long",
) -> pd.DataFrame:
    loaded = load_backtest_artifact_dataset(artifact_dir, coin=coin)
    return calc_ema_anchor_neutral_bands(
        loaded["config"],
        loaded["candles"],
        side=side,
        market_settings=loaded["market_settings"],
    )


def calc_ema_anchor_inventory_bands_from_artifact(
    artifact_dir: str | Path,
    *,
    coin: str | None = None,
    side_mode: str = "active",
) -> pd.DataFrame:
    loaded = load_backtest_artifact_dataset(artifact_dir, coin=coin)
    return calc_ema_anchor_inventory_bands(
        loaded["config"],
        loaded["candles"],
        loaded["fills"],
        loaded["balance_and_equity"],
        market_settings=loaded["market_settings"],
        coin=loaded["coin"],
        side_mode=side_mode,
    )
