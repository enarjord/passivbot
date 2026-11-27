"""Self-contained HTML view of saved backtest fills, positions and candles.

This module only reads saved facts (fills, equity, cached candles). It never
replays strategy logic. Every quantity shown is either a saved fill field, a
candle aggregation, or a grouping of consecutive fills of one position side.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

MINUTE_MS = 60_000
INTERVALS_MS = tuple(minutes * MINUTE_MS for minutes in (1, 5, 15, 30, 60, 240, 720, 1440))
OVERVIEW_MAX_BARS = 4000
DETAIL_MAX_BARS = 720
# Full-span tier finer than the overview, bounded across all coins (~40 B/bar).
MEDIUM_TOTAL_BARS = 240_000
DETAIL_MAX_WINDOWS = 40
EQUITY_MAX_POINTS = 2000
STATIC = Path(__file__).resolve().parent / "tools" / "fills_report_static"

# Longest prefixes first so multi-word mechanisms are not split.
FAMILIES = (
    "ema_anchor",
    "auto_reduce",
    "initial",
    "grid",
    "trailing",
    "unstuck",
    "panic",
)
TAKE_PROFIT_FAMILIES = frozenset({"grid", "trailing", "ema_anchor"})
RISK_FAMILIES = frozenset({"unstuck", "auto_reduce", "panic"})
REQUIRED_FILL_COLUMNS = (
    "timestamp",
    "coin",
    "type",
    "price",
    "qty",
    "pnl",
    "fee_paid",
    "psize",
    "pprice",
    "wallet_exposure",
    "usd_total_balance",
)


@dataclass(frozen=True)
class FillKind:
    action: str  # entry | close
    family: str  # initial, grid, trailing, ema_anchor, unstuck, auto_reduce, panic
    mechanism: str  # full middle part, e.g. trailing_cropped, auto_reduce_wel
    pside: str  # long | short

    @property
    def intent(self) -> str:
        if self.action == "entry":
            return "entry"
        if self.family in RISK_FAMILIES:
            return "risk"
        if self.family in TAKE_PROFIT_FAMILIES:
            return "take_profit"
        return "other"


def classify_fill_type(order_type: str) -> FillKind:
    """Split a snake-case order type such as ``close_auto_reduce_wel_long``."""
    parts = str(order_type).split("_")
    if len(parts) < 3 or parts[0] not in {"entry", "close"} or parts[-1] not in {"long", "short"}:
        raise ValueError(f"Unrecognized fill order type: {order_type!r}")
    mechanism = "_".join(parts[1:-1])
    family = next(
        (name for name in FAMILIES if mechanism == name or mechanism.startswith(name + "_")),
        mechanism,
    )
    return FillKind(parts[0], family, mechanism, parts[-1])


def choose_interval_ms(span_ms: int, max_bars: int, min_interval_ms: int = MINUTE_MS) -> int:
    for interval in INTERVALS_MS:
        if interval >= min_interval_ms and span_ms / interval <= max_bars:
            return interval
    return INTERVALS_MS[-1]


def ohlc_bars(
    timestamps: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    start_ms: int,
    end_ms: int,
    interval_ms: int,
) -> np.ndarray:
    """Aggregate 1m high/low/close rows into ``[[open_ms, o, h, l, c], ...]``.

    Saved candles carry no open, so a bar opens at the previous minute's close
    (or its own first close when none precedes it). Rows without a positive
    finite close, such as pre-inception padding, are excluded.
    """
    if interval_ms % MINUTE_MS:
        raise ValueError("Candle intervals must be whole minutes")
    first = int(np.searchsorted(timestamps, start_ms, side="left"))
    last = int(np.searchsorted(timestamps, end_ms, side="right"))
    if last <= first:
        return np.empty((0, 5))
    ts = np.asarray(timestamps[first:last], dtype=np.int64)
    hi = np.asarray(high[first:last], dtype=np.float64)
    lo = np.asarray(low[first:last], dtype=np.float64)
    cl = np.asarray(close[first:last], dtype=np.float64)
    previous = np.empty_like(cl)
    previous[1:] = cl[:-1]
    previous[0] = close[first - 1] if first > 0 else cl[0]
    valid = np.isfinite(cl) & (cl > 0) & np.isfinite(hi) & np.isfinite(lo)
    if not valid.any():
        return np.empty((0, 5))
    ts, hi, lo, cl, previous = ts[valid], hi[valid], lo[valid], cl[valid], previous[valid]
    bad_open = ~np.isfinite(previous) | (previous <= 0)
    previous[bad_open] = cl[bad_open]
    buckets = ts // interval_ms
    starts = np.flatnonzero(np.r_[True, buckets[1:] != buckets[:-1]])
    out = np.empty((len(starts), 5))
    out[:, 0] = buckets[starts] * interval_ms
    out[:, 1] = previous[starts]
    out[:, 2] = np.maximum.reduceat(np.maximum(hi, cl), starts)
    out[:, 3] = np.minimum.reduceat(np.minimum(lo, cl), starts)
    out[:, 4] = cl[np.r_[starts[1:], len(cl)] - 1]
    return out


@dataclass(frozen=True)
class CandleSource:
    """1m high/low/close rows aligned to millisecond clocks."""

    timestamps: np.ndarray
    hlcvs: np.ndarray
    coin_index: dict[str, int]

    def __post_init__(self):
        if self.hlcvs.ndim != 3 or self.hlcvs.shape[2] < 3:
            raise ValueError(f"Expected hlcvs shape (T, N, >=3), got {self.hlcvs.shape}")
        if len(self.timestamps) != self.hlcvs.shape[0]:
            raise ValueError("Candle clocks and hlcvs rows have different lengths")
        if len(self.timestamps) > 1 and not (np.diff(self.timestamps) > 0).all():
            raise ValueError("Candle clocks must be strictly increasing")

    @property
    def native_interval_ms(self) -> int:
        if len(self.timestamps) < 2:
            return MINUTE_MS
        return max(MINUTE_MS, int(np.median(np.diff(self.timestamps[:10_000]))))

    def bars(self, coin: str, start_ms: int, end_ms: int, interval_ms: int) -> np.ndarray:
        idx = self.coin_index[coin]
        data = self.hlcvs[:, idx, :]
        return ohlc_bars(
            self.timestamps, data[:, 0], data[:, 1], data[:, 2], start_ms, end_ms, interval_ms
        )

    @classmethod
    def from_result(cls, directory: Path, dataset: dict) -> "CandleSource":
        from backtest_artifacts import _load_npy, _resolve_artifact_path

        hlcvs = _load_npy(_resolve_artifact_path(dataset, "hlcvs_file", artifact_dir=directory))
        timestamps = _load_npy(
            _resolve_artifact_path(dataset, "timestamps_file", artifact_dir=directory)
        ).astype(np.int64)
        return cls(timestamps, hlcvs, coin_index_from_dataset(dataset))


def coin_index_from_dataset(dataset: dict) -> dict[str, int]:
    index = {str(k): int(v) for k, v in (dataset.get("coin_index") or {}).items()}
    if not index:
        index = {str(coin): i for i, coin in enumerate(dataset.get("coins") or [])}
    return index


def prepare_fills(fills: pd.DataFrame) -> pd.DataFrame:
    """Validate the ledger and add classification columns in saved row order."""
    missing = [key for key in REQUIRED_FILL_COLUMNS if key not in fills]
    if missing:
        raise ValueError(f"Fill ledger lacks columns required by the fills report: {missing}")
    frame = fills.reset_index(drop=True).copy()
    kinds = {value: classify_fill_type(value) for value in frame["type"].unique()}
    frame["action"] = frame["type"].map(lambda v: kinds[v].action)
    frame["family"] = frame["type"].map(lambda v: kinds[v].family)
    frame["intent"] = frame["type"].map(lambda v: kinds[v].intent)
    frame["pside"] = frame["type"].map(lambda v: kinds[v].pside)
    frame["timestamp_ms"] = frame["timestamp"].dt.as_unit("ms").astype("int64")
    # Saved balances are post-fill; PnL % is relative to the balance before the fill.
    balance = (
        frame["usd_total_balance"].to_numpy(dtype=np.float64)
        - frame["pnl"].to_numpy(dtype=np.float64)
        - frame["fee_paid"].to_numpy(dtype=np.float64)
    )
    frame["balance_before"] = balance
    frame["pnl_fraction"] = np.divide(
        frame["pnl"].to_numpy(dtype=np.float64),
        balance,
        out=np.zeros(len(frame)),
        where=balance > 0,
    )
    # Stable: same-clock fills keep their saved execution order.
    return frame.sort_values("timestamp_ms", kind="mergesort").reset_index(drop=True)


def position_episodes(
    fills: pd.DataFrame,
    min_qty: dict[str, float] | None = None,
    final_ms: int | None = None,
) -> pd.DataFrame:
    """Group each coin/side's fills from the first entry until the position is flat.

    Flat means ``|psize|`` strictly below the coin's minimum quantity (exactly zero
    when unknown); a position of exactly the minimum is still tradable and open. A
    position still open at the end has ``finished`` False and ends at ``final_ms``,
    the run's last observation clock, when given.
    """
    min_qty = min_qty or {}
    columns = [
        "coin",
        "pside",
        "start_ms",
        "end_ms",
        "entries",
        "closes",
        "pnl",
        "pnl_fraction",
        "max_wallet_exposure",
        "exit_family",
        "finished",
        "first_row",
        "last_row",
    ]
    records = []
    for (coin, pside), group in fills.groupby(["coin", "pside"], sort=True):
        threshold = float(min_qty.get(coin, 0.0) or 0.0)
        current = None
        for row in group.itertuples():
            if current is None:
                if row.action != "entry":
                    continue  # Closes of a position opened before the ledger window.
                current = {
                    "coin": coin,
                    "pside": pside,
                    "start_ms": row.timestamp_ms,
                    "entries": 0,
                    "closes": 0,
                    "pnl": 0.0,
                    "balance": float(row.balance_before),
                    "max_we": 0.0,
                    "exits": {},
                    "first_row": row.Index,
                }
            current["pnl"] += float(row.pnl)
            current["max_we"] = max(current["max_we"], abs(float(row.wallet_exposure)))
            if row.action == "entry":
                current["entries"] += 1
            else:
                current["closes"] += 1
                current["exits"][row.family] = current["exits"].get(row.family, 0) + 1
            flat = abs(float(row.psize)) < threshold if threshold > 0 else float(row.psize) == 0.0
            if flat:
                records.append(_finish_episode(current, row, True))
                current = None
        if current is not None:
            record = _finish_episode(current, row, False)
            if final_ms is not None:
                record["end_ms"] = max(int(record["end_ms"]), int(final_ms))
            records.append(record)
    frame = pd.DataFrame.from_records(records, columns=columns)
    return frame.sort_values(["start_ms", "coin", "pside"], kind="mergesort").reset_index(drop=True)


def _finish_episode(current, row, finished):
    exits = current["exits"]
    # Most closes wins; a tie goes to the family of the final close.
    final = row.family if row.action == "close" else None
    return {
        "coin": current["coin"],
        "pside": current["pside"],
        "start_ms": current["start_ms"],
        "end_ms": row.timestamp_ms,
        "entries": current["entries"],
        "closes": current["closes"],
        "pnl": current["pnl"],
        "pnl_fraction": current["pnl"] / current["balance"] if current["balance"] > 0 else 0.0,
        "max_wallet_exposure": current["max_we"],
        "exit_family": (max(exits, key=lambda f: (exits[f], f == final)) if exits else ""),
        "finished": finished,
        "first_row": current["first_row"],
        "last_row": row.Index,
    }


def select_detail_episodes(episodes: pd.DataFrame, limit: int = DETAIL_MAX_WINDOWS) -> list[int]:
    """Episodes worth fine candles: largest |PnL %|, longest, highest exposure."""
    if episodes.empty:
        return []
    ranked = [
        episodes["pnl_fraction"].abs().sort_values(ascending=False, kind="mergesort").index,
        (episodes["end_ms"] - episodes["start_ms"])
        .sort_values(ascending=False, kind="mergesort")
        .index,
        episodes["max_wallet_exposure"].sort_values(ascending=False, kind="mergesort").index,
    ]
    quotas = (limit - 2 * (limit // 5), limit // 5, limit // 5)
    chosen: list[int] = []
    for order, quota in zip(ranked, quotas):
        taken = 0
        for idx in order:
            if taken >= quota or len(chosen) >= limit:
                break
            if idx not in chosen:
                chosen.append(int(idx))
                taken += 1
    return chosen


def episode_window(start_ms: int, end_ms: int) -> tuple[int, int]:
    pad = max(int((end_ms - start_ms) * 0.08), 3 * 3_600_000)
    return start_ms - pad, end_ms + pad


def _rounded(values, digits):
    return [
        None if not math.isfinite(v) else round(v, digits)
        for v in np.asarray(values, dtype=np.float64)
    ]


def _significant(values, digits=8):
    out = []
    for v in np.asarray(values, dtype=np.float64):
        if not math.isfinite(v):
            out.append(None)
        elif v == 0:
            out.append(0.0)
        else:
            out.append(float(f"{v:.{digits}g}"))
    return out


def encode_bars(bars: np.ndarray, interval_ms: int) -> dict[str, Any]:
    """Columnar bars: open clocks as interval-unit deltas, prices at 6 significant digits."""
    if not len(bars):
        return {"t0": None, "dt": [], "o": [], "h": [], "l": [], "c": []}
    slots = (bars[:, 0].astype(np.int64) // interval_ms).astype(np.int64)
    return {
        "t0": int(bars[0, 0]),
        "dt": [0, *np.diff(slots).astype(int).tolist()],
        "o": _significant(bars[:, 1], 6),
        "h": _significant(bars[:, 2], 6),
        "l": _significant(bars[:, 3], 6),
        "c": _significant(bars[:, 4], 6),
    }


def candles_cover(entry: dict[str, Any], interval_ms: int, start_ms: int, end_ms: int) -> bool:
    """Whether a coin's tiers already hold candles this fine for the whole window.

    The overview spans the entire view; each detail tier only its own window.
    """
    if entry["overview"]["interval_ms"] <= interval_ms:
        return True
    return any(
        d["interval_ms"] <= interval_ms and d["start_ms"] <= start_ms and d["end_ms"] >= end_ms
        for d in entry["details"]
    )


def build_payload(
    store,
    *,
    candles: CandleSource | None = None,
    coins: tuple[str, ...] = (),
    start_ms: int | None = None,
    end_ms: int | None = None,
) -> dict[str, Any]:
    from backtest_artifacts import _resolve_artifact_path
    from backtest_reports import equity_view

    dataset = store.json("dataset.json")
    analysis = store.json("analysis.json", optional=True) or {}
    fills = prepare_fills(store.fills())
    if coins:
        unknown = sorted(
            set(coins) - set(fills["coin"].unique()) - set(coin_index_from_dataset(dataset))
        )
        if unknown:
            raise ValueError(f"Unknown coins for the fills report: {unknown}")
        fills = fills.loc[fills["coin"].isin(coins)].reset_index(drop=True)

    mss_path = _resolve_artifact_path(
        dataset, "market_specific_settings_file", artifact_dir=store.directory, required=False
    )
    market = json.loads(Path(mss_path).read_text()) if mss_path is not None else {}
    if mss_path is None:
        logging.warning(
            "[fills report] market settings unavailable; positions count as flat only at exactly zero size"
        )
    coin_settings = {
        coin: settings
        for coin, settings in market.items()
        if isinstance(settings, dict) and not coin.startswith("__")
    }
    min_qty = {coin: float(s.get("min_qty") or 0.0) for coin, s in coin_settings.items()}
    c_mult = {coin: float(s.get("c_mult") or 1.0) for coin, s in coin_settings.items()}
    equity_clock = store.equity(columns=["strategy_equity"]).index
    final_ms = int(equity_clock.as_unit("ms").asi8[-1]) if len(equity_clock) else None
    # Episodes come from the whole ledger so a focused window never truncates the
    # history of a position that opened before it.
    episodes = position_episodes(fills, min_qty, final_ms)
    in_window = np.ones(len(fills), dtype=bool)
    if start_ms is not None:
        in_window &= fills["timestamp_ms"].to_numpy() >= start_ms
        episodes = episodes.loc[episodes["end_ms"] >= start_ms]
    if end_ms is not None:
        in_window &= fills["timestamp_ms"].to_numpy() <= end_ms
        episodes = episodes.loc[episodes["start_ms"] <= end_ms]
    # Keep episodes with at least one of their own fills inside the window.
    keep = np.zeros(len(episodes), dtype=bool)
    for (coin, pside), rows in fills.loc[in_window].groupby(["coin", "pside"]).groups.items():
        rows = np.asarray(rows)
        mask = ((episodes["coin"] == coin) & (episodes["pside"] == pside)).to_numpy()
        first = episodes["first_row"].to_numpy()[mask]
        last = episodes["last_row"].to_numpy()[mask]
        keep[mask] = np.searchsorted(rows, last, "right") > np.searchsorted(rows, first, "left")
    episodes = episodes.loc[keep]
    # Position state just before the window, so the page can draw positions that
    # were already open when the focused window starts.
    seed: dict[str, dict[str, list[float]]] = {}
    if start_ms is not None:
        before = fills.loc[fills["timestamp_ms"].to_numpy() < start_ms]
        for (coin, pside), group in before.groupby(["coin", "pside"], sort=True):
            last = group.iloc[-1]
            if float(last["psize"]) != 0.0:
                seed.setdefault(coin, {})[pside] = _significant(
                    [last["psize"], last["pprice"], last["wallet_exposure"]], 10
                )
    kept_rows = np.flatnonzero(in_window)
    fills = fills.loc[in_window].reset_index(drop=True)
    # Re-point episode rows at the windowed ledger (the nearest rows inside it).
    episodes = episodes.assign(
        first_row=np.searchsorted(kept_rows, episodes["first_row"].to_numpy(), side="left"),
        last_row=np.searchsorted(kept_rows, episodes["last_row"].to_numpy(), side="right") - 1,
    ).reset_index(drop=True)

    if candles is None and not fills.empty:
        candles = CandleSource.from_result(store.directory, dataset)

    traded = list(dict.fromkeys(fills["coin"]))
    types = sorted(fills["type"].unique())
    type_index = {t: i for i, t in enumerate(types)}
    coin_order = {c: i for i, c in enumerate(traded)}
    span_start = int(fills["timestamp_ms"].min()) if len(fills) else start_ms or 0
    span_end = int(fills["timestamp_ms"].max()) if len(fills) else end_ms or 0
    view_start = start_ms if start_ms is not None else span_start
    view_end = end_ms if end_ms is not None else span_end

    candle_payload: dict[str, Any] = {}
    if candles is not None and traded:
        native = candles.native_interval_ms
        overview_interval = choose_interval_ms(
            max(view_end - view_start, MINUTE_MS), OVERVIEW_MAX_BARS, native
        )
        pad = overview_interval * 2
        for coin in traded:
            if coin not in candles.coin_index:
                raise ValueError(f"Cached candles lack traded coin {coin!r}")
            candle_payload[coin] = {
                "overview": {
                    "interval_ms": overview_interval,
                    "bars": encode_bars(
                        candles.bars(coin, view_start - pad, view_end + pad, overview_interval),
                        overview_interval,
                    ),
                },
                "details": [],
            }
        medium = choose_interval_ms(
            max(view_end - view_start, MINUTE_MS) * len(traded), MEDIUM_TOTAL_BARS, native
        )
        if medium < overview_interval:
            for coin in traded:
                candle_payload[coin]["details"].append(
                    {
                        "interval_ms": medium,
                        "start_ms": view_start - pad,
                        "end_ms": view_end + pad,
                        "bars": encode_bars(
                            candles.bars(coin, view_start - pad, view_end + pad, medium), medium
                        ),
                    }
                )
        for idx in select_detail_episodes(episodes):
            ep = episodes.loc[idx]
            w0, w1 = episode_window(int(ep.start_ms), int(ep.end_ms))
            interval = choose_interval_ms(w1 - w0, DETAIL_MAX_BARS, native)
            entry = candle_payload[ep.coin]
            if candles_cover(entry, interval, w0, w1):
                continue
            entry["details"].append(
                {
                    "interval_ms": interval,
                    "start_ms": w0,
                    "end_ms": w1,
                    "bars": encode_bars(candles.bars(ep.coin, w0, w1, interval), interval),
                }
            )

    equity = equity_view(store, max_points=EQUITY_MAX_POINTS, start=start_ms, end=end_ms)
    hsl_enabled = bool((analysis.get("metric_metadata") or {}).get("hsl_enabled"))
    return {
        "schema_version": 1,
        "title": store.directory.name,
        "coins": traded,
        "types": types,
        "start_ms": view_start,
        "end_ms": view_end,
        "filters": {"coins": list(coins), "start_ms": start_ms, "end_ms": end_ms},
        "hsl_enabled": hsl_enabled,
        "c_mult": {coin: c_mult.get(coin, 1.0) for coin in traded},
        "seed": {coin: seed[coin] for coin in traded if coin in seed},
        "analysis": {
            key: analysis.get(key)
            for key in (
                "drawdown_max_fraction_strategy_equity",
                "sharpe_ratio_strategy_equity",
                "position_held_time_weighted_mean_hours",
                "gross_exposure_max_ratio_strategy_equity",
            )
            if isinstance(analysis.get(key), (int, float)) and math.isfinite(analysis.get(key))
        },
        "equity": {
            "t": [int(v) for v in equity["timestamp"]],
            "equity": _significant(equity["strategy_equity"], 10),
            "balance": _significant(equity["usd_total_balance"], 10),
        },
        "fills": {
            "t": fills["timestamp_ms"].astype(int).tolist(),
            "coin": [coin_order[c] for c in fills["coin"]],
            "type": [type_index[t] for t in fills["type"]],
            "price": _significant(fills["price"], 10),
            "qty": _significant(fills["qty"], 10),
            "pnl": _rounded(fills["pnl"], 6),
            "pnl_fraction": _significant(fills["pnl_fraction"], 6),
            "psize": _significant(fills["psize"], 10),
            "pprice": _significant(fills["pprice"], 8),
            "we": _significant(fills["wallet_exposure"], 6),
        },
        "episodes": {
            "coin": [coin_order[c] for c in episodes["coin"]],
            "pside": episodes["pside"].tolist(),
            "start_ms": episodes["start_ms"].astype(int).tolist(),
            "end_ms": episodes["end_ms"].astype(int).tolist(),
            "entries": episodes["entries"].astype(int).tolist(),
            "closes": episodes["closes"].astype(int).tolist(),
            "pnl": _rounded(episodes["pnl"], 4),
            "pnl_fraction": _significant(episodes["pnl_fraction"], 6),
            "max_we": _significant(episodes["max_wallet_exposure"], 5),
            "exit_family": episodes["exit_family"].tolist(),
            "finished": episodes["finished"].astype(bool).tolist(),
            "first_row": episodes["first_row"].astype(int).tolist(),
            "last_row": episodes["last_row"].astype(int).tolist(),
        },
        "candles": candle_payload,
    }


def render_fills_html(payload: dict[str, Any]) -> str:
    template = (STATIC / "template.html").read_text(encoding="utf-8")
    script = (STATIC / "app.js").read_text(encoding="utf-8")
    data = json.dumps(payload, separators=(",", ":"), allow_nan=False)
    # Keep the embedded JSON from closing the script element.
    data = data.replace("</", "<\\/").replace("<!--", "<\\!--")
    title = str(payload.get("title", "Backtest fills")).replace("&", "&amp;").replace("<", "&lt;")
    return (
        template.replace("__TITLE__", title)
        .replace("/*__DATA__*/", f"const DATA = {data};")
        .replace("/*__APP__*/", script)
    )


def write_fills_report(store, path: Path, **kwargs) -> Path:
    from backtest_results import atomic_output

    html = render_fills_html(build_payload(store, **kwargs))
    with atomic_output(Path(path), "wb") as handle:
        handle.write(html.encode("utf-8"))
    return Path(path)
