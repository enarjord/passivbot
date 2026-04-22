import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

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
    if isinstance(artifact, BacktestArtifact):
        return artifact.candles_for_coin(coin)
    helper: Callable[[str], pd.DataFrame] | None = artifact.get("candles_for_coin")
    if helper is not None:
        return helper(coin)
    raise TypeError("artifact must be BacktestArtifact or workspace dict from load_backtest_artifact_workspace")
