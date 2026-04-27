import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


HLCV_COLUMNS = ("high", "low", "close", "volume")


def _load_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_npy(path: Path) -> np.ndarray:
    if path.suffix == ".gz":
        with gzip.open(path, "rb") as f:
            return np.load(f)
    return np.load(path)


def _resolve_artifact_path(
    dataset: dict,
    key: str,
    *,
    artifact_dir: Path,
    required: bool = True,
) -> Path | None:
    raw = dataset.get(key)
    if raw in (None, ""):
        if required:
            raise KeyError(f"dataset.json missing required path key {key!r}")
        return None
    path = Path(str(raw)).expanduser()
    candidates = [path]
    if not path.is_absolute():
        candidates.append(artifact_dir / path)
        cache_dir = dataset.get("hlcv_cache_dir")
        if cache_dir:
            candidates.append(Path(str(cache_dir)).expanduser() / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    if required:
        raise FileNotFoundError(f"dataset path {key!r} does not exist: {raw}")
    return None


def _read_csv_if_exists(path: Path, *, compression: str | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, compression=compression)


def _normalize_timestamp_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "timestamp" not in out.columns:
        first_col = out.columns[0] if len(out.columns) else None
        if first_col is not None and str(first_col).startswith("Unnamed"):
            out = out.rename(columns={first_col: "timestamp"})
    else:
        generated_index_cols = [col for col in out.columns if str(col).startswith("Unnamed")]
        if generated_index_cols:
            out = out.drop(columns=generated_index_cols)
    if "timestamp" in out.columns:
        out["timestamp"] = pd.to_datetime(out["timestamp"], errors="raise")
    return out


@dataclass(frozen=True)
class BacktestArtifact:
    artifact_dir: Path
    dataset: dict
    config: dict
    analysis: dict
    fills: pd.DataFrame
    balance_and_equity: pd.DataFrame
    hlcvs: np.ndarray
    timestamps: np.ndarray
    btc_usd_prices: np.ndarray | None
    coins: list[str]
    coin_index: dict[str, int]
    market_settings: dict

    def candles_for_coin(self, coin: str) -> pd.DataFrame:
        if coin not in self.coin_index:
            raise KeyError(f"coin {coin!r} not present in artifact coins {self.coins}")
        idx = int(self.coin_index[coin])
        if self.hlcvs.ndim != 3:
            raise ValueError(f"expected hlcvs shape (T, N, C), got {self.hlcvs.shape}")
        if idx < 0 or idx >= self.hlcvs.shape[1]:
            raise IndexError(f"coin index {idx} for {coin!r} outside hlcvs shape {self.hlcvs.shape}")
        coin_hlcvs = self.hlcvs[:, idx, :]
        if coin_hlcvs.shape[1] < 3:
            raise ValueError(f"expected at least high/low/close columns, got {coin_hlcvs.shape}")
        columns = list(HLCV_COLUMNS[: min(len(HLCV_COLUMNS), coin_hlcvs.shape[1])])
        df = pd.DataFrame(coin_hlcvs[:, : len(columns)], columns=columns)
        df.insert(0, "timestamp", pd.to_datetime(self.timestamps.astype(np.int64), unit="ms"))
        return df

    def workspace(self) -> dict[str, Any]:
        return {
            "artifact": self,
            "artifact_dir": self.artifact_dir,
            "dataset": self.dataset,
            "config": self.config,
            "cfg": self.config,
            "analysis": self.analysis,
            "fills": self.fills,
            "fdf": self.fills,
            "balance_and_equity": self.balance_and_equity,
            "bdf": self.balance_and_equity,
            "hlcvs": self.hlcvs,
            "timestamps": self.timestamps,
            "btc_usd_prices": self.btc_usd_prices,
            "coins": self.coins,
            "coin_index": self.coin_index,
            "market_settings": self.market_settings,
            "candles_for_coin": self.candles_for_coin,
            "plot_fills_for_coin": lambda coin, **kwargs: plot_fills_for_coin(
                self, coin=coin, **kwargs
            ),
        }


def load_backtest_artifact(artifact_dir: str | Path) -> BacktestArtifact:
    artifact_dir = Path(artifact_dir).expanduser().resolve()
    if not artifact_dir.exists():
        raise FileNotFoundError(artifact_dir)
    dataset = _load_json(artifact_dir / "dataset.json")
    config = _load_json(artifact_dir / "config.json")
    analysis = _load_json(artifact_dir / "analysis.json")

    fills = _normalize_timestamp_column(_read_csv_if_exists(artifact_dir / "fills.csv"))
    balance_and_equity = _normalize_timestamp_column(
        _read_csv_if_exists(artifact_dir / "balance_and_equity.csv.gz", compression="gzip")
    )

    hlcvs_path = _resolve_artifact_path(dataset, "hlcvs_file", artifact_dir=artifact_dir)
    timestamps_path = _resolve_artifact_path(dataset, "timestamps_file", artifact_dir=artifact_dir)
    btc_path = _resolve_artifact_path(
        dataset, "btc_usd_prices_file", artifact_dir=artifact_dir, required=False
    )
    market_settings_path = _resolve_artifact_path(
        dataset, "market_specific_settings_file", artifact_dir=artifact_dir
    )

    hlcvs = _load_npy(hlcvs_path)
    timestamps = _load_npy(timestamps_path)
    btc_usd_prices = _load_npy(btc_path) if btc_path is not None else None
    market_settings = _load_json(market_settings_path)
    coins = list(dataset.get("coins") or [])
    coin_index = {str(k): int(v) for k, v in (dataset.get("coin_index") or {}).items()}
    if not coin_index and coins:
        coin_index = {coin: idx for idx, coin in enumerate(coins)}
    if not coins and coin_index:
        coins = [coin for coin, _idx in sorted(coin_index.items(), key=lambda item: item[1])]

    return BacktestArtifact(
        artifact_dir=artifact_dir,
        dataset=dataset,
        config=config,
        analysis=analysis,
        fills=fills,
        balance_and_equity=balance_and_equity,
        hlcvs=hlcvs,
        timestamps=timestamps,
        btc_usd_prices=btc_usd_prices,
        coins=coins,
        coin_index=coin_index,
        market_settings=market_settings,
    )


def load_backtest_artifact_workspace(artifact_dir: str | Path) -> dict[str, Any]:
    """
    Return a Jupyter-friendly dict for `globals().update(...)`.

    Example:
        workspace = load_backtest_artifact_workspace("backtests/combined/latest_run")
        globals().update(workspace)
        candles = candles_for_coin("BTC")
    """
    return load_backtest_artifact(artifact_dir).workspace()


def candles_for_coin(artifact: BacktestArtifact | dict[str, Any], coin: str) -> pd.DataFrame:
    artifact = _coerce_artifact(artifact)
    return artifact.candles_for_coin(coin)


def _coerce_artifact(artifact: BacktestArtifact | dict[str, Any]) -> BacktestArtifact:
    if isinstance(artifact, BacktestArtifact):
        return artifact
    if isinstance(artifact, dict):
        loaded = artifact.get("artifact")
        if isinstance(loaded, BacktestArtifact):
            return loaded
    raise TypeError(
        "artifact must be BacktestArtifact or workspace dict from load_backtest_artifact_workspace"
    )


def _parse_optional_date(value: str | pd.Timestamp | None, *, name: str) -> pd.Timestamp | None:
    if value in (None, ""):
        return None
    try:
        return pd.to_datetime(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be parseable as a datetime; got {value!r}") from exc


def _filter_time_window(
    df: pd.DataFrame,
    *,
    start_date: str | pd.Timestamp | None,
    end_date: str | pd.Timestamp | None,
) -> pd.DataFrame:
    start = _parse_optional_date(start_date, name="start_date")
    end = _parse_optional_date(end_date, name="end_date")
    if start is not None and end is not None and start > end:
        raise ValueError(f"start_date must be <= end_date; got {start} > {end}")
    out = df
    if start is not None:
        out = out[out["timestamp"] >= start]
    if end is not None:
        out = out[out["timestamp"] <= end]
    return out


def _select_coin_fills(
    fills: pd.DataFrame,
    coin: str,
    *,
    start_date: str | pd.Timestamp | None,
    end_date: str | pd.Timestamp | None,
) -> pd.DataFrame:
    if fills.empty:
        return fills.copy()
    if "timestamp" not in fills.columns:
        raise KeyError("fills missing required column 'timestamp'")
    if "coin" not in fills.columns:
        raise KeyError("fills missing required column 'coin'")
    out = fills[fills["coin"] == coin].copy()
    out = _filter_time_window(out, start_date=start_date, end_date=end_date)
    return out.sort_values("timestamp").reset_index(drop=True)


def _plot_fill_markers(ax, fills: pd.DataFrame) -> None:
    if fills.empty:
        return
    required = {"timestamp", "type", "price"}
    missing = required.difference(fills.columns)
    if missing:
        raise KeyError(f"fills missing required columns: {sorted(missing)}")
    fills = fills.copy()
    fills["price"] = pd.to_numeric(fills["price"], errors="raise")
    type_series = fills["type"].astype(str)

    marker_specs = [
        (
            "long_entry",
            type_series.str.contains("long", regex=False)
            & type_series.str.contains("entry", regex=False),
            "b",
            ".",
            "long entries",
        ),
        (
            "long_close",
            type_series.str.contains("long", regex=False)
            & type_series.str.contains("close", regex=False),
            "r",
            ".",
            "long closes",
        ),
        (
            "short_entry",
            type_series.str.contains("short", regex=False)
            & type_series.str.contains("entry", regex=False),
            "m",
            "x",
            "short entries",
        ),
        (
            "short_close",
            type_series.str.contains("short", regex=False)
            & type_series.str.contains("close", regex=False),
            "c",
            "x",
            "short closes",
        ),
    ]
    for _name, mask, color, marker, label in marker_specs:
        rows = fills[mask]
        if not rows.empty:
            ax.scatter(
                rows["timestamp"],
                rows["price"],
                c=color,
                marker=marker,
                label=label,
                zorder=3.0,
            )


def _plot_position_prices(ax, candles: pd.DataFrame, fills: pd.DataFrame) -> None:
    if fills.empty or not {"timestamp", "type", "pprice", "psize"}.issubset(fills.columns):
        return
    candle_index = pd.DatetimeIndex(candles["timestamp"])
    type_series = fills["type"].astype(str)
    for side, color in (("long", "b"), ("short", "r")):
        side_fills = fills[type_series.str.contains(side, regex=False)].copy()
        if side_fills.empty:
            continue
        side_fills["pprice"] = pd.to_numeric(side_fills["pprice"], errors="coerce")
        side_fills["psize"] = pd.to_numeric(side_fills["psize"], errors="coerce")
        state = side_fills.groupby("timestamp", sort=True)[["pprice", "psize"]].last()
        aligned = state.reindex(candle_index).ffill()
        pprice_change = aligned["pprice"].pct_change().fillna(0.0)
        aligned.loc[pprice_change != 0.0, "pprice"] = np.nan
        pprices = aligned.loc[aligned["psize"].abs() > 0.0, "pprice"]
        if not pprices.empty:
            ax.plot(
                pprices.index,
                pprices.to_numpy(dtype=float),
                color=color,
                linestyle="--",
                alpha=0.65,
                label=f"{side} pprice",
            )


def plot_fills_for_coin(
    artifact: BacktestArtifact | dict[str, Any],
    coin: str,
    *,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
    figsize: tuple[float, float] = (21, 13),
    include_high_low: bool = True,
):
    """
    Plot cached candles and fill markers for one coin from a loaded backtest artifact.

    `artifact` should be a `BacktestArtifact` or workspace dict returned by
    `load_backtest_artifact_workspace()`. This avoids reloading large HLCV arrays for repeated
    notebook plots.
    """
    try:
        from plotting import plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError("matplotlib/plotting helpers are required for plot_fills_for_coin") from exc

    artifact = _coerce_artifact(artifact)
    candles = artifact.candles_for_coin(coin)
    candles = _filter_time_window(candles, start_date=start_date, end_date=end_date)
    if candles.empty:
        raise ValueError(f"no candle rows for {coin!r} in requested time window")
    fills = _select_coin_fills(
        artifact.fills,
        coin,
        start_date=start_date,
        end_date=end_date,
    )

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(candles["timestamp"], candles["close"], "y-", label="close", zorder=1.0)
    if include_high_low:
        ax.plot(candles["timestamp"], candles["low"], "g--", alpha=0.75, label="low", zorder=0.9)
        ax.plot(candles["timestamp"], candles["high"], "g-.", alpha=0.55, label="high", zorder=0.8)
    _plot_fill_markers(ax, fills)
    _plot_position_prices(ax, candles, fills)
    ax.set_title(f"Fills {coin}")
    ax.set_xlabel("datetime")
    ax.set_ylabel("Price")
    ax.legend()
    fig.tight_layout()
    plt.close(fig)
    return fig
