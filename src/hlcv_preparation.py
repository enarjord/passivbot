import asyncio
import json
import logging
import os
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from uuid import uuid4

import numpy as np
import pandas as pd

from candlestick_manager import CandlestickManager


def _pct_log(level: str, pct: int, msg: str, *args, **kwargs) -> None:
    """Log with percentage prefix: '12% | message here'."""
    pct_str = f"{pct:3d}%" if pct < 100 else "100%"
    full_msg = f"{pct_str} | {msg}"
    getattr(logging, level)(full_msg, *args, **kwargs)


class ProgressTracker:
    """Track progress and emit percentage-prefixed logs periodically."""

    def __init__(self, total: int, context: str, log_interval_seconds: float = 10.0):
        self.total = max(1, total)
        self.context = context
        self.log_interval = log_interval_seconds
        self.processed = 0
        self.started = time.monotonic()
        self.last_logged = self.started

    def update(self, n: int = 1) -> None:
        self.processed += n

    def pct(self) -> int:
        return min(100, int(100 * self.processed / self.total))

    def maybe_log(self, current_item: str = "", force: bool = False) -> None:
        now = time.monotonic()
        if not force and (now - self.last_logged) < self.log_interval:
            return
        self.last_logged = now
        elapsed = max(0.1, now - self.started)
        rate = self.processed / elapsed
        remaining = max(0, self.total - self.processed)
        eta_s = int(remaining / rate) if rate > 0 else 0
        eta_str = f"ETA {eta_s}s" if eta_s > 0 else ""
        item_str = f"current={current_item}" if current_item else ""
        parts = [f"{self.context} {self.processed}/{self.total}"]
        if item_str:
            parts.append(item_str)
        if eta_str:
            parts.append(eta_str)
        _pct_log("info", self.pct(), " ".join(parts))

    def log_done(self) -> None:
        elapsed = round(time.monotonic() - self.started, 1)
        _pct_log("info", 100, f"{self.context} done {self.processed}/{self.total} in {elapsed}s")


from config_utils import require_config_value, require_live_value
from ohlcv_utils import dump_ohlcv_data, get_days_in_between, load_ohlcv_data
from procedures import get_first_timestamps_unified
from utils import (
    coin_to_symbol,
    to_standard_exchange_name,
    format_end_date,
    get_quote,
    load_ccxt_instance,
    load_markets,
    make_get_filepath,
    to_ccxt_exchange_id,
    symbol_to_coin,
    ts_to_date,
    utc_ms,
    date_to_ts,
)
from warmup_utils import compute_backtest_warmup_minutes, compute_per_coin_warmup_minutes


class HLCVManager:
    """Backtest-oriented OHLCV manager using CandlestickManager for fetching/caching."""

    def __init__(
        self,
        exchange: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        cc=None,
        gap_tolerance_ohlcvs_minutes: float = 120.0,
        verbose: bool = True,
        cm_debug_level: int = 0,
        cm_progress_log_interval_seconds: float = 10.0,  # Log progress every 10s by default
        force_refetch_gaps: bool = False,
        ohlcv_source_dir: Optional[str] = None,
    ):
        self.exchange = to_ccxt_exchange_id(exchange)
        self.quote = get_quote(self.exchange)
        self.start_date = "2020-01-01" if start_date is None else format_end_date(start_date)
        self.end_date = format_end_date("now" if end_date is None else end_date)
        self.start_ts = int(date_to_ts(self.start_date))
        self.end_ts = int(date_to_ts(self.end_date))
        self.cc = cc
        # Use standard exchange name for cache paths (e.g., "binance" not "binanceusdm")
        cache_exchange = to_standard_exchange_name(self.exchange)
        self.cache_filepaths = {
            "markets": os.path.join("caches", cache_exchange, "markets.json"),
            "first_timestamps": os.path.join("caches", cache_exchange, "first_timestamps.json"),
        }
        self.markets = None
        self.verbose = bool(verbose)
        self.gap_tolerance_ohlcvs_minutes = float(gap_tolerance_ohlcvs_minutes)
        self.cm_debug_level = int(cm_debug_level)
        try:
            self.cm_progress_log_interval_seconds = max(0.0, float(cm_progress_log_interval_seconds))
        except Exception:
            self.cm_progress_log_interval_seconds = 0.0
        self.force_refetch_gaps = bool(force_refetch_gaps)
        self.cm: Optional[CandlestickManager] = None
        self.ohlcv_source_dir = str(ohlcv_source_dir) if ohlcv_source_dir else None

    def update_date_range(self, new_start_date=None, new_end_date=None):
        if new_start_date is not None:
            if isinstance(new_start_date, (float, int)):
                self.start_date = ts_to_date(new_start_date)
            elif isinstance(new_start_date, str):
                self.start_date = new_start_date
            else:
                raise ValueError(f"invalid start date {new_start_date}")
            self.start_ts = int(date_to_ts(self.start_date))
        if new_end_date is not None:
            if isinstance(new_end_date, (float, int)):
                self.end_date = ts_to_date(new_end_date)
            elif isinstance(new_end_date, str):
                self.end_date = new_end_date
            else:
                raise ValueError(f"invalid end date {new_end_date}")
            self.end_date = format_end_date(self.end_date)
            self.end_ts = int(date_to_ts(self.end_date))

    def load_cc(self):
        if self.cc is None:
            self.cc = load_ccxt_instance(self.exchange, enable_rate_limit=True)
        if self.cm is None:
            # Limit concurrent ccxt requests per exchange to reduce timeouts under heavy parallelism.
            # Bybit tends to be more sensitive.
            max_concurrent_requests = 3 if self.exchange == "bybit" else 5
            # Use standard name for cache paths (e.g., "binance" not "binanceusdm")
            self.cm = CandlestickManager(
                exchange=self.cc,
                exchange_name=to_standard_exchange_name(self.exchange),
                debug=int(self.cm_debug_level),
                progress_log_interval_seconds=float(self.cm_progress_log_interval_seconds),
                max_concurrent_requests=max_concurrent_requests,
                remote_fetch_callback=self._remote_fetch_log,
                # Backtest HLCV preparation may share the same cache directory as live trading.
                # Force-disable disk retention here to avoid deleting shards created/needed by other processes.
                max_disk_candles_per_symbol_per_tf=0,
            )

    async def aclose(self) -> None:
        """Close CandlestickManager resources (not the ccxt exchange)."""
        if self.cm is None:
            return
        try:
            await self.cm.aclose()
        except Exception:
            pass
        self.cm = None

    def close(self) -> None:
        """Best-effort sync close for CandlestickManager resources."""
        if self.cm is None:
            return
        try:
            self.cm.close()
        except Exception:
            pass
        self.cm = None

    def _remote_fetch_log(self, payload: dict) -> None:
        """Log a concise view of download progress (CCXT + archive).

        CandlestickManager uses its own logger which is often set to WARNING when
        cm_debug_level=0. This callback lets hlcv_preparation surface progress at
        INFO level without changing CandlestickManager verbosity globally.
        """

        # Throttle per (kind, symbol, tf/stage) to avoid log spam.
        if not hasattr(self, "_download_log_last"):
            self._download_log_last = {}
        now = time.monotonic()

        kind = payload.get("kind")
        stage = payload.get("stage")
        symbol = payload.get("symbol")
        tf = payload.get("tf")

        if kind == "ccxt_fetch_ohlcv":
            attempt = int(payload.get("attempt", 1) or 1)
            if stage == "start":
                # Only show attempt 1, and at most once per 10 seconds per symbol/tf.
                key = ("ccxt", symbol, tf, "start")
                if (now - float(self._download_log_last.get(key, 0.0))) < 10.0:
                    return
                self._download_log_last[key] = now

                since_ms = payload.get("since_ts")
                since_iso = None
                try:
                    if since_ms is not None:
                        since_iso = datetime.fromtimestamp(
                            int(since_ms) / 1000, tz=timezone.utc
                        ).strftime("%Y-%m-%dT%H:%M:%SZ")
                except Exception:
                    since_iso = None

                logging.info(
                    "[%s] download ccxt start symbol=%s tf=%s since=%s since_ms=%s limit=%s params=%s",
                    self.exchange,
                    symbol,
                    payload.get("tf"),
                    since_iso if since_iso is not None else since_ms,
                    since_ms,
                    payload.get("limit"),
                    payload.get("params"),
                )
                return

            if stage == "ok":
                # Throttle OK logs to once per 10 seconds per symbol/tf.
                key = ("ccxt", symbol, tf, "ok")
                if (now - float(self._download_log_last.get(key, 0.0))) < 10.0:
                    return
                self._download_log_last[key] = now
                logging.info(
                    "[%s] download ccxt ok symbol=%s tf=%s rows=%s elapsed_ms=%s",
                    self.exchange,
                    symbol,
                    payload.get("tf"),
                    payload.get("rows"),
                    payload.get("elapsed_ms"),
                )
                return

            if stage == "error":
                # Let CandlestickManager warnings show the details; keep this concise.
                if attempt == 1:
                    logging.warning(
                        "[%s] download ccxt error symbol=%s tf=%s error_type=%s error=%s",
                        self.exchange,
                        symbol,
                        tf,
                        payload.get("error_type"),
                        payload.get("error"),
                    )
                return

        if kind == "archive_prefetch":
            if stage == "start":
                logging.info(
                    "[%s] download archive start symbol=%s days=%s parallel=%s range=%s",
                    self.exchange,
                    symbol,
                    payload.get("days_to_fetch"),
                    payload.get("parallel"),
                    payload.get("date_range"),
                )
                return
            if stage == "skip":
                logging.info(
                    "[%s] download archive skip symbol=%s reasons=%s",
                    self.exchange,
                    symbol,
                    payload.get("reasons"),
                )
                return
            if stage == "progress":
                key = ("archive", symbol, "progress")
                if (now - float(self._download_log_last.get(key, 0.0))) < 10.0:
                    return
                self._download_log_last[key] = now
                logging.info(
                    "[%s] download archive progress symbol=%s %s/%s (%s%%) batch=%s",
                    self.exchange,
                    symbol,
                    payload.get("completed"),
                    payload.get("total"),
                    payload.get("pct"),
                    payload.get("batch"),
                )
                return
            if stage == "done":
                logging.info(
                    "[%s] download archive done symbol=%s fetched=%s skipped=%s total=%s elapsed_s=%s",
                    self.exchange,
                    symbol,
                    payload.get("fetched"),
                    payload.get("skipped"),
                    payload.get("total"),
                    payload.get("elapsed_s"),
                )
                return

    async def load_markets(self):
        self.load_cc()
        self.markets = await load_markets(self.exchange, verbose=False)
        try:
            if hasattr(self.cc, "set_markets"):
                self.cc.set_markets(self.markets)
        except Exception:
            pass

    def get_symbol(self, coin: str) -> str:
        assert self.markets, "needs to call load_markets() first"
        return coin_to_symbol(coin, self.exchange)

    def has_coin(self, coin: str) -> bool:
        symbol = self.get_symbol(coin)
        # Also verify symbol exists in markets (fallback symbols won't be present)
        return symbol and symbol in self.markets

    def get_market_specific_settings(self, coin: str) -> dict:
        mss = dict(self.markets[self.get_symbol(coin)])
        mss["hedge_mode"] = True
        mss["maker_fee"] = mss.get("maker")
        mss["taker_fee"] = mss.get("taker")
        mss["c_mult"] = mss.get("contractSize")
        mss["min_cost"] = (
            mc if (mc := mss.get("limits", {}).get("cost", {}).get("min")) is not None else 0.01
        )
        mss["price_step"] = mss.get("precision", {}).get("price")
        mss["min_qty"] = max(
            lm if (lm := mss.get("limits", {}).get("amount", {}).get("min")) is not None else 0.0,
            pm if (pm := mss.get("precision", {}).get("amount")) is not None else 0.0,
        )
        mss["qty_step"] = mss.get("precision", {}).get("amount")
        if self.exchange == "bybit":
            # ccxt reports incorrect fees for bybit perps
            mss["maker"] = mss["maker_fee"] = 0.0002
            mss["taker"] = mss["taker_fee"] = 0.00055
        elif self.exchange in ("kucoin", "kucoinfutures"):
            # ccxt reports incorrect fees for kucoin futures. Assume VIP0
            mss["maker"] = mss["maker_fee"] = 0.0002
            mss["taker"] = mss["taker_fee"] = 0.0006
        elif self.exchange == "gateio":
            # ccxt reports incorrect fees for gateio perps. Assume VIP0
            mss["maker"] = mss["maker_fee"] = 0.0002
            mss["taker"] = mss["taker_fee"] = 0.0005
        return mss

    def load_first_timestamp(self, coin: str):
        fpath = self.cache_filepaths["first_timestamps"]
        if os.path.exists(fpath):
            try:
                ftss = json.load(open(fpath))
                if coin in ftss:
                    return ftss[coin]
            except Exception as e:
                logging.error(f"Error loading {fpath} {e}")

    def dump_first_timestamp(self, coin: str, fts: float):
        fpath = self.cache_filepaths["first_timestamps"]
        try:
            if os.path.exists(fpath):
                try:
                    ftss = json.load(open(fpath))
                except Exception:
                    ftss = {}
            else:
                make_get_filepath(fpath)
                ftss = {}
            ftss[coin] = fts
            json.dump(ftss, open(fpath, "w"), indent=True, sort_keys=True)
        except Exception as e:
            logging.error(f"Error dumping {fpath}: {e}")

    async def get_first_timestamp(self, coin: str) -> float:
        if fts := self.load_first_timestamp(coin):
            return float(fts)
        if not self.markets:
            await self.load_markets()
        self.load_cc()
        try:
            ohlcvs = await self.cc.fetch_ohlcv(
                self.get_symbol(coin),
                since=int(date_to_ts("2018-01-01")),
                timeframe="1d",
            )
            if not ohlcvs:
                ohlcvs = await self.cc.fetch_ohlcv(
                    self.get_symbol(coin),
                    since=int(date_to_ts("2020-01-01")),
                    timeframe="1d",
                )
            fts = float(ohlcvs[0][0]) if ohlcvs else 0.0
        except Exception:
            fts = 0.0
        self.dump_first_timestamp(coin, fts)
        return float(fts)

    def _try_load_ohlcvs_from_source_dir(
        self, coin: str, symbol: str, start_ts: int, end_ts: int
    ) -> Optional[pd.DataFrame]:
        if not self.ohlcv_source_dir:
            return None
        source_root = Path(self.ohlcv_source_dir)
        if not source_root.exists():
            return None

        exchange_dir = to_standard_exchange_name(self.exchange)
        exchange_root = source_root / exchange_dir / "1m"

        start_day = ts_to_date(start_ts)
        end_day = ts_to_date(end_ts)
        days = get_days_in_between(start_day, end_day)

        frames = []
        candidates: list[str] = []
        
        # Mirror CandlestickManager's symbol sanitization for Windows compatibility
        windows_compat = os.name == "nt" or os.getenv("WINDOWS_COMPATIBILITY") == "1"
        
        for name in (coin, symbol):
            if not name:
                continue
            # Original name
            candidates.append(name)
            # Variant with "/" replaced by "_"
            if "/" in name:
                candidates.append(name.replace("/", "_"))
            # In Windows compatibility mode, also mirror CandlestickManager's ":" -> "_"
            if windows_compat and ("/" in name or ":" in name):
                candidates.append(name.replace("/", "_").replace(":", "_"))
        
        # De-duplicate while preserving order
        seen = set()
        candidates = [c for c in candidates if not (c in seen or seen.add(c))]
        for day in days:
            loaded = False
            for name in candidates:
                base_dir = exchange_root / name
                if not base_dir.exists():
                    continue
                for ext in (".npz", ".npy"):
                    filepath = base_dir / f"{day}{ext}"
                    if not filepath.exists():
                        continue
                    try:
                        df = load_ohlcv_data(str(filepath))
                    except Exception as exc:
                        logging.warning(
                            "[%s] source dir load failed for %s %s: %s",
                            self.exchange,
                            coin,
                            filepath,
                            exc,
                        )
                        continue
                    if not df.empty:
                        frames.append(df)
                        loaded = True
                        break
                if loaded:
                    break

        if not frames:
            return None

        df = pd.concat(frames, ignore_index=True)
        df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
        df = df[(df["timestamp"] >= start_ts) & (df["timestamp"] <= end_ts)]
        if df.empty:
            return None

        ts = df["timestamp"].astype(np.int64, copy=False).values
        if ts.size > 1:
            intervals = np.diff(ts)
            if not np.all(intervals == 60_000):
                greatest_gap_ms = int(intervals.max(initial=60_000))
                logging.warning(
                    "[%s] source dir non-contiguous data for %s; greatest gap %.1f minutes. Falling back.",
                    self.exchange,
                    coin,
                    greatest_gap_ms / 60_000.0,
                )
                return None

        return df.reset_index(drop=True)

    async def get_ohlcvs(self, coin: str, start_date=None, end_date=None) -> pd.DataFrame:
        empty_df = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        if start_date is not None or end_date is not None:
            self.update_date_range(start_date, end_date)
        if not self.markets:
            await self.load_markets()
        if not self.has_coin(coin):
            logging.debug("[%s] get_ohlcvs: coin %s not found in markets", self.exchange, coin)
            return empty_df
        symbol = self.get_symbol(coin)
        start_ts = int(self.start_ts)
        end_ts = int(self.end_ts)
        if start_ts > end_ts:
            logging.debug(
                "[%s] get_ohlcvs: invalid range start_ts=%s > end_ts=%s",
                self.exchange,
                start_ts,
                end_ts,
            )
            return empty_df

        if self.ohlcv_source_dir:
            df = self._try_load_ohlcvs_from_source_dir(coin, symbol, start_ts, end_ts)
            if df is not None and not df.empty:
                logging.info(
                    "[%s] get_ohlcvs: using source dir for %s",
                    self.exchange,
                    coin,
                )
                return df
            logging.debug(
                "[%s] get_ohlcvs: source dir had no data for %s; falling back to candlestick manager",
                self.exchange,
                coin,
            )

        self.load_cc()
        assert self.cm is not None

        # Fetch strict (real) candles first to detect large gaps.
        real = await self.cm.get_candles(
            symbol,
            start_ts=start_ts,
            end_ts=end_ts,
            max_age_ms=0,
            strict=True,
            timeframe="1m",
            force_refetch_gaps=self.force_refetch_gaps,
        )
        if real.size == 0:
            logging.warning(
                "[%s] get_ohlcvs: cm.get_candles returned empty for %s range %s to %s",
                self.exchange,
                symbol,
                ts_to_date(start_ts),
                ts_to_date(end_ts),
            )
            return empty_df

        ts = real["ts"].astype(np.int64, copy=False)
        if ts.size > 1:
            intervals = np.diff(ts)
            greatest_gap_ms = int(intervals.max(initial=60_000))
            if greatest_gap_ms > int(self.gap_tolerance_ohlcvs_minutes * 60_000):
                # Helpful diagnostics: locate the exact gap boundaries
                gap_start_ts = None
                gap_end_ts = None
                try:
                    imax = int(np.argmax(intervals))
                    if 0 <= imax < ts.size - 1:
                        gap_start_ts = int(ts[imax])
                        gap_end_ts = int(ts[imax + 1])
                except Exception:
                    gap_start_ts = None
                    gap_end_ts = None

                # Give CandlestickManager one chance to self-heal (legacy merge / refetch gaps)
                # before returning empty.
                if not self.force_refetch_gaps:
                    real = await self.cm.get_candles(
                        symbol,
                        start_ts=start_ts,
                        end_ts=end_ts,
                        max_age_ms=0,
                        strict=True,
                        timeframe="1m",
                        force_refetch_gaps=True,
                    )
                    if real.size == 0:
                        return empty_df
                    ts = real["ts"].astype(np.int64, copy=False)
                    if ts.size > 1:
                        intervals = np.diff(ts)
                        greatest_gap_ms = int(intervals.max(initial=60_000))
                    else:
                        # Single or zero timestamp implies no measurable gaps;
                        # don't reuse stale greatest_gap_ms from the initial fetch.
                        greatest_gap_ms = 0
                if greatest_gap_ms > int(self.gap_tolerance_ohlcvs_minutes * 60_000):
                    if self.verbose:
                        logging.warning(
                            "[%s] gaps detected in %s OHLCV data; greatest gap: %.1f minutes. Returning empty.",
                            self.exchange,
                            coin,
                            greatest_gap_ms / 60_000.0,
                        )
                        if gap_start_ts is not None and gap_end_ts is not None:
                            gap_start_iso = datetime.fromtimestamp(
                                int(gap_start_ts) / 1000, tz=timezone.utc
                            ).strftime("%Y-%m-%dT%H:%M:%SZ")
                            gap_end_iso = datetime.fromtimestamp(
                                int(gap_end_ts) / 1000, tz=timezone.utc
                            ).strftime("%Y-%m-%dT%H:%M:%SZ")
                            logging.warning(
                                "[%s] largest_gap_window %s -> %s (%.1f minutes)",
                                self.exchange,
                                gap_start_iso,
                                gap_end_iso,
                                (gap_end_ts - gap_start_ts) / 60_000.0,
                            )
                    return empty_df

        # Fill to the full requested window (bfill/ffill inside CM).
        # Data from get_candles is already sorted, so skip redundant sort
        filled = self.cm.standardize_gaps(
            real, start_ts=start_ts, end_ts=end_ts, strict=False, assume_sorted=True
        )
        if filled.size == 0:
            return empty_df
        df = pd.DataFrame(
            {
                "timestamp": filled["ts"].astype(np.int64),
                "open": filled["o"].astype(float),
                "high": filled["h"].astype(float),
                "low": filled["l"].astype(float),
                "close": filled["c"].astype(float),
                "volume": filled["bv"].astype(float),
            }
        )
        return df.reset_index(drop=True)


async def prepare_hlcvs(config: dict, exchange: str, *, force_refetch_gaps: bool = False):
    approved = require_live_value(config, "approved_coins")
    coins = sorted(
        set(symbol_to_coin(c) for c in approved["long"])
        | set(symbol_to_coin(c) for c in approved["short"])
    )
    orig_coins = list(coins)
    exchange = to_ccxt_exchange_id(exchange)
    requested_start_date = require_config_value(config, "backtest.start_date")
    requested_start_ts = int(date_to_ts(requested_start_date))
    end_date = format_end_date(require_config_value(config, "backtest.end_date"))
    end_ts = int(date_to_ts(end_date))

    warmup_minutes = compute_backtest_warmup_minutes(config)
    minute_ms = 60_000
    warmup_ms = warmup_minutes * minute_ms
    effective_start_ts = max(0, requested_start_ts - warmup_ms)
    effective_start_ts = (effective_start_ts // minute_ms) * minute_ms
    effective_start_date = ts_to_date(effective_start_ts)

    if warmup_minutes > 0:
        logging.info(
            f"{exchange} applying warmup: {warmup_minutes} minutes -> fetch start {effective_start_date}, "
            f"requested start {requested_start_date}"
        )

    om = HLCVManager(
        exchange,
        effective_start_date,
        end_date,
        gap_tolerance_ohlcvs_minutes=require_config_value(
            config, "backtest.gap_tolerance_ohlcvs_minutes"
        ),
        cm_debug_level=int(config.get("backtest", {}).get("cm_debug_level", 0) or 0),
        cm_progress_log_interval_seconds=float(
            config.get("backtest", {}).get("cm_progress_log_interval_seconds", 10.0) or 10.0
        ),
        force_refetch_gaps=force_refetch_gaps,
        ohlcv_source_dir=config.get("backtest", {}).get("ohlcv_source_dir"),
    )

    try:
        mss, timestamps, hlcvs = await prepare_hlcvs_internal(
            config,
            coins,
            exchange,
            effective_start_ts,
            requested_start_ts,
            end_ts,
            om,
        )

        om.update_date_range(int(timestamps[0]), int(timestamps[-1]))
        btc_df = await om.get_ohlcvs("BTC")
        if btc_df.empty:
            raise ValueError(f"Failed to fetch BTC/USD prices from {exchange}")

        btc_df = (
            btc_df.set_index("timestamp")
            .reindex(timestamps, method="ffill")
            .ffill()
            .bfill()
            .reset_index()
        )
        btc_usd_prices = btc_df["close"].values

        warmup_provided = max(0, int(max(0, requested_start_ts - int(timestamps[0])) // minute_ms))
        mss["__meta__"] = {
            "requested_start_ts": int(requested_start_ts),
            "requested_start_date": ts_to_date(requested_start_ts),
            "effective_start_ts": int(timestamps[0]),
            "effective_start_date": ts_to_date(int(timestamps[0])),
            "warmup_minutes_requested": int(warmup_minutes),
            "warmup_minutes_provided": int(warmup_provided),
        }

        return mss, timestamps, hlcvs, btc_usd_prices
    finally:
        await om.aclose()
        if om.cc:
            await om.cc.close()


async def prepare_hlcvs_internal(
    config,
    coins,
    exchange,
    effective_start_ts,
    requested_start_ts,
    end_ts,
    om: HLCVManager,
):
    minimum_coin_age_days = float(require_live_value(config, "minimum_coin_age_days"))
    interval_ms = 60_000

    first_timestamps_unified = await get_first_timestamps_unified(coins)
    per_coin_warmups = compute_per_coin_warmup_minutes(config)
    default_warm = int(per_coin_warmups.get("__default__", 0))

    cache_dir = Path(f"./caches/hlcvs_data/{uuid4().hex[:16]}")
    cache_dir.mkdir(parents=True, exist_ok=True)

    coin_metadata = {}
    valid_coins = {}
    global_start_time = float("inf")
    global_end_time = float("-inf")
    await om.load_markets()
    min_coin_age_ms = 1000 * 60 * 60 * 24 * minimum_coin_age_days

    progress = ProgressTracker(len(coins), f"{exchange} fetching candles")
    progress.maybe_log(force=True)

    # Async helper to fetch a single coin's data with all validation
    async def fetch_coin_data(coin: str, sem: asyncio.Semaphore):
        """Fetch and validate data for a single coin. Returns (coin, metadata, file_path, data_bounds) or None."""
        async with sem:  # Rate limiting
            try:
                adjusted_start_ts = int(effective_start_ts)

                # Validation: check if coin exists
                if not om.has_coin(coin):
                    _pct_log("info", progress.pct(), f"{exchange} coin {coin} missing, skipping")
                    return None

                if coin not in first_timestamps_unified:
                    _pct_log(
                        "info",
                        progress.pct(),
                        f"coin {coin} missing from first_timestamps_unified, skipping",
                    )
                    return None

                # Minimum coin age validation
                if minimum_coin_age_days > 0.0:
                    try:
                        first_ts = await om.get_first_timestamp(coin)
                    except Exception as e:
                        _pct_log(
                            "error",
                            progress.pct(),
                            f"error with get_first_timestamp for {coin} {e}. Skipping",
                        )
                        traceback.print_exc()
                        return None

                    if first_ts >= end_ts:
                        _pct_log(
                            "info",
                            progress.pct(),
                            f"{exchange} Coin {coin} too young, start date {ts_to_date(first_ts)}. Skipping",
                        )
                        return None

                    coin_age_days = int(
                        round(utc_ms() - first_timestamps_unified[coin]) / (1000 * 60 * 60 * 24)
                    )
                    if coin_age_days < minimum_coin_age_days:
                        _pct_log(
                            "info",
                            progress.pct(),
                            f"{exchange} Coin {coin}: Not traded due to min_coin_age {int(minimum_coin_age_days)} days. "
                            f"{coin} is {coin_age_days} days old. Skipping",
                        )
                        return None

                    new_adjusted_start_ts = max(
                        first_timestamps_unified[coin] + min_coin_age_ms, first_ts
                    )
                    if new_adjusted_start_ts > adjusted_start_ts:
                        _pct_log(
                            "info",
                            progress.pct(),
                            f"{exchange} Coin {coin}: Adjusting start date from {ts_to_date(adjusted_start_ts)} "
                            f"to {ts_to_date(new_adjusted_start_ts)}",
                        )
                        adjusted_start_ts = int(new_adjusted_start_ts)

                # Fetch OHLCV data
                om.update_date_range(adjusted_start_ts)
                df = await om.get_ohlcvs(coin)
                data = df[["timestamp", "high", "low", "close", "volume"]].values

                if len(data) == 0:
                    return None

                # Validate no gaps
                assert (np.diff(data[:, 0]) == interval_ms).all(), f"gaps in hlcv data {coin}"

                # Save to cache
                file_path = cache_dir / f"{coin}.npy"
                dump_ohlcv_data(data, str(file_path))

                # Prepare metadata
                metadata = {
                    "start_time": int(data[0, 0]),
                    "end_time": int(data[-1, 0]),
                    "length": len(data),
                }

                data_bounds = (data[0, 0], data[-1, 0])

                return (coin, metadata, file_path, data_bounds)

            except Exception as e:
                _pct_log("error", progress.pct(), f"error with get_ohlcvs for {coin} {e}. Skipping")
                traceback.print_exc()
                return None

    # Parallelize coin fetching with rate limiting.
    # Bybit is more sensitive to bursts; keep concurrency lower to reduce timeouts.
    COIN_CONCURRENCY = 3 if str(exchange).lower() == "bybit" else 6
    sem = asyncio.Semaphore(COIN_CONCURRENCY)

    tasks = [fetch_coin_data(coin, sem) for coin in coins]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Process results
    for result in results:
        progress.maybe_log()

        if result is None or isinstance(result, Exception):
            progress.update()
            continue

        coin, metadata, file_path, data_bounds = result
        coin_metadata[coin] = metadata
        valid_coins[coin] = file_path
        global_start_time = min(global_start_time, data_bounds[0])
        global_end_time = max(global_end_time, data_bounds[1])
        progress.update()

    progress.log_done()

    if not valid_coins:
        logging.error(
            "[%s] no valid coins found with data for %s -> %s (coins=%s)",
            exchange,
            ts_to_date(effective_start_ts),
            ts_to_date(end_ts),
            ",".join(coins),
        )
        raise ValueError("No valid coins found with data")

    n_timesteps = int((global_end_time - global_start_time) / interval_ms) + 1
    n_coins = len(valid_coins)
    timestamps = np.arange(global_start_time, global_end_time + interval_ms, interval_ms)
    unified_array = np.full((n_timesteps, n_coins, 4), np.nan, dtype=np.float64)
    valid_index_ranges = {}

    logging.info(
        f"{exchange} Unifying data for {len(valid_coins)} coin{'s' if len(valid_coins) > 1 else ''} into single numpy array..."
    )
    for i, coin in enumerate(valid_coins):
        file_path = valid_coins[coin]
        ohlcv = np.load(file_path)
        start_idx = int((ohlcv[0, 0] - global_start_time) / interval_ms)
        end_idx = start_idx + len(ohlcv)
        coin_data = ohlcv[:, 1:]
        unified_array[start_idx:end_idx, i, :] = coin_data
        first_idx = int(start_idx)
        last_idx = int(end_idx - 1)
        warm_minutes = int(per_coin_warmups.get(coin, default_warm))
        trade_start_idx = first_idx + warm_minutes
        if trade_start_idx > last_idx:
            trade_start_idx = last_idx
        valid_index_ranges[coin] = (first_idx, last_idx)

        os.remove(file_path)

    try:
        os.rmdir(cache_dir)
    except OSError:
        pass
    mss = {}
    for coin in sorted(valid_coins):
        meta = om.get_market_specific_settings(coin)
        first_idx, last_idx = valid_index_ranges.get(
            coin, (unified_array.shape[0], unified_array.shape[0])
        )
        meta["first_valid_index"] = first_idx
        meta["last_valid_index"] = last_idx
        warm_minutes = int(per_coin_warmups.get(coin, default_warm))
        meta["warmup_minutes"] = warm_minutes
        trade_start_idx = first_idx + warm_minutes
        if trade_start_idx > last_idx:
            trade_start_idx = last_idx
        meta["trade_start_index"] = trade_start_idx
        mss[coin] = meta
    return mss, timestamps, unified_array


async def prepare_hlcvs_combined(
    config,
    forced_sources=None,
    market_settings_sources=None,
    *,
    force_refetch_gaps: bool = False,
):
    backtest_exchanges = require_config_value(config, "backtest.exchanges")
    exchanges_to_consider = [to_ccxt_exchange_id(e) for e in backtest_exchanges]
    forced_sources = forced_sources or {}
    normalized_forced_sources = {
        str(coin): to_ccxt_exchange_id(exchange)
        for coin, exchange in forced_sources.items()
        if exchange
    }
    market_settings_sources = market_settings_sources or {}
    normalized_mss_sources = {
        str(coin): to_ccxt_exchange_id(exchange)
        for coin, exchange in market_settings_sources.items()
        if exchange
    }
    ohlcv_exchanges = sorted(
        set(exchanges_to_consider) | set(normalized_forced_sources.values())
    )

    requested_start_date = require_config_value(config, "backtest.start_date")
    requested_start_ts = int(date_to_ts(requested_start_date))
    end_date = format_end_date(require_config_value(config, "backtest.end_date"))
    end_ts = int(date_to_ts(end_date))

    warmup_minutes = compute_backtest_warmup_minutes(config)
    minute_ms = 60_000
    warmup_ms = warmup_minutes * minute_ms
    effective_start_ts = max(0, requested_start_ts - warmup_ms)
    effective_start_ts = (effective_start_ts // minute_ms) * minute_ms
    effective_start_date = ts_to_date(effective_start_ts)

    if warmup_minutes > 0:
        logging.info(
            f"combined applying warmup: {warmup_minutes} minutes -> fetch start {effective_start_date}, "
            f"requested start {requested_start_date}"
        )

    om_dict: Dict[str, HLCVManager] = {}
    for ex in exchanges_to_consider:
        om_dict[ex] = HLCVManager(
            ex,
            effective_start_date,
            end_date,
            gap_tolerance_ohlcvs_minutes=require_config_value(
                config, "backtest.gap_tolerance_ohlcvs_minutes"
            ),
            cm_debug_level=int(config.get("backtest", {}).get("cm_debug_level", 0) or 0),
            cm_progress_log_interval_seconds=float(
                config.get("backtest", {}).get("cm_progress_log_interval_seconds", 10.0) or 10.0
            ),
            force_refetch_gaps=force_refetch_gaps,
            ohlcv_source_dir=config.get("backtest", {}).get("ohlcv_source_dir"),
        )
    extra_forced = set(normalized_forced_sources.values()) - set(exchanges_to_consider)
    extra_mss = set(normalized_mss_sources.values()) - set(exchanges_to_consider) - extra_forced
    for ex in extra_forced | extra_mss:
        om_dict[ex] = HLCVManager(
            ex,
            effective_start_date,
            end_date,
            gap_tolerance_ohlcvs_minutes=require_config_value(
                config, "backtest.gap_tolerance_ohlcvs_minutes"
            ),
            cm_debug_level=int(config.get("backtest", {}).get("cm_debug_level", 0) or 0),
            cm_progress_log_interval_seconds=float(
                config.get("backtest", {}).get("cm_progress_log_interval_seconds", 10.0) or 10.0
            ),
            force_refetch_gaps=force_refetch_gaps,
            ohlcv_source_dir=config.get("backtest", {}).get("ohlcv_source_dir"),
        )
    btc_om: Optional[HLCVManager] = None

    try:
        mss, timestamps, unified_array = await _prepare_hlcvs_combined_impl(
            config,
            om_dict,
            effective_start_ts,
            requested_start_ts,
            end_ts,
            normalized_forced_sources,
            normalized_mss_sources,
            ohlcv_exchanges=ohlcv_exchanges,
        )

        btc_exchange = exchanges_to_consider[0] if len(exchanges_to_consider) == 1 else "binanceusdm"
        btc_om = HLCVManager(
            btc_exchange,
            effective_start_date,
            end_date,
            gap_tolerance_ohlcvs_minutes=require_config_value(
                config, "backtest.gap_tolerance_ohlcvs_minutes"
            ),
        )
        # Align BTC date range to actual timestamps (mirrors single-exchange case)
        btc_om.update_date_range(int(timestamps[0]), int(timestamps[-1]))
        logging.info(
            "fetching BTC/USD prices from %s for range %s to %s",
            btc_exchange,
            ts_to_date(int(timestamps[0])),
            ts_to_date(int(timestamps[-1])),
        )
        btc_df = await btc_om.get_ohlcvs("BTC")
        if btc_df.empty:
            logging.error(
                "BTC/USD fetch returned empty for %s (start=%s end=%s)",
                btc_exchange,
                btc_om.start_date,
                btc_om.end_date,
            )
            raise ValueError(f"Failed to fetch BTC/USD prices from {btc_exchange}")

        btc_df = (
            btc_df.set_index("timestamp")
            .reindex(timestamps, method="ffill")
            .ffill()
            .bfill()
            .reset_index()
        )
        btc_usd_prices = btc_df["close"].values

        warmup_provided = max(0, int(max(0, requested_start_ts - int(timestamps[0])) // minute_ms))
        mss["__meta__"] = {
            "requested_start_ts": int(requested_start_ts),
            "requested_start_date": ts_to_date(requested_start_ts),
            "effective_start_ts": int(timestamps[0]),
            "effective_start_date": ts_to_date(int(timestamps[0])),
            "warmup_minutes_requested": int(warmup_minutes),
            "warmup_minutes_provided": int(warmup_provided),
        }

        return mss, timestamps, unified_array, btc_usd_prices
    finally:
        for om in om_dict.values():
            await om.aclose()
            if om.cc:
                await om.cc.close()
        if btc_om:
            await btc_om.aclose()
        if btc_om and btc_om.cc:
            await btc_om.cc.close()


async def _prepare_hlcvs_combined_impl(
    config,
    om_dict: Dict[str, HLCVManager],
    base_start_ts: int,
    _requested_start_ts: int,
    end_ts: int,
    forced_sources: Dict[str, str],
    market_settings_sources: Optional[Dict[str, str]] = None,
    *,
    ohlcv_exchanges: Optional[Sequence[str]] = None,
):
    market_settings_sources = market_settings_sources or {}
    approved = require_live_value(config, "approved_coins")
    coins = sorted(
        set(symbol_to_coin(c) for c in approved["long"])
        | set(symbol_to_coin(c) for c in approved["short"])
    )
    orig_coins = list(coins)
    if ohlcv_exchanges is not None:
        exchanges_to_consider = [ex for ex in ohlcv_exchanges if ex in om_dict]
    else:
        exchanges_to_consider = sorted(list(om_dict.keys()))
    minimum_coin_age_days = float(require_live_value(config, "minimum_coin_age_days"))
    interval_ms = 60_000
    min_coin_age_ms = 1000 * 60 * 60 * 24 * minimum_coin_age_days

    first_timestamps_unified = await get_first_timestamps_unified(coins)

    per_coin_warmups = compute_per_coin_warmup_minutes(config)
    default_warm = int(per_coin_warmups.get("__default__", 0))

    chosen_data_per_coin = {}
    chosen_mss_per_coin = {}

    # Preload markets
    await asyncio.gather(*[om.load_markets() for om in om_dict.values()])

    # Normalize stock perp coins: convert plain tickers to xyz: prefixed for hyperliquid
    def normalize_stock_perp_coin(coin: str, forced_exchange: Optional[str]) -> str:
        """Convert plain ticker to xyz: prefix if it's a stock perp on hyperliquid."""
        if forced_exchange != "hyperliquid":
            return coin
        if coin.startswith("xyz:"):
            return coin  # Already prefixed
        # Check if xyz:TICKER exists on hyperliquid
        hl_om = om_dict.get("hyperliquid")
        if hl_om and hl_om.markets:
            xyz_coin = f"xyz:{coin}"
            # Check if the xyz: prefixed version maps to a valid symbol
            xyz_symbol = coin_to_symbol(xyz_coin, "hyperliquid")
            if xyz_symbol and xyz_symbol in hl_om.markets:
                logging.info(f"Normalizing stock perp coin: {coin} -> {xyz_coin}")
                return xyz_coin
        return coin

    # Apply normalization to coins
    orig_coins = list(coins)
    normalized_coins = []
    for coin in coins:
        # Get the forced exchange for this coin (checking both with and without xyz: prefix)
        forced_ex = forced_sources.get(coin)
        if forced_ex is None and coin.startswith("xyz:"):
            forced_ex = forced_sources.get(coin[4:])  # Check without prefix
        elif forced_ex is None and not coin.startswith("xyz:"):
            forced_ex = forced_sources.get(f"xyz:{coin}")  # Check with prefix
        normalized_coin = normalize_stock_perp_coin(coin, forced_ex)
        normalized_coins.append(normalized_coin)

    # Deduplicate and update coins list
    coins = sorted(set(normalized_coins))

    # Update first_timestamps for normalized coins
    for orig, norm in zip(orig_coins, normalized_coins):
        if orig != norm and orig in first_timestamps_unified and norm not in first_timestamps_unified:
            first_timestamps_unified[norm] = first_timestamps_unified[orig]

    progress = ProgressTracker(len(coins), "combined fetching candles")
    progress.maybe_log(force=True)

    # Async helper to process a single coin across all candidate exchanges
    async def process_coin(coin: str, sem: asyncio.Semaphore):
        """Fetch coin data from all candidate exchanges and select the best one."""
        async with sem:  # Rate limiting
            try:
                if coin not in first_timestamps_unified:
                    return None

                coin_fts = int(first_timestamps_unified[coin])
                effective_start_ts = max(int(base_start_ts), coin_fts + int(min_coin_age_ms))
                if effective_start_ts >= end_ts:
                    logging.info(
                        "%s: skipping - effective start %s >= end %s "
                        "(coin first available: %s, min_age=%d days)",
                        coin,
                        ts_to_date(effective_start_ts),
                        ts_to_date(end_ts),
                        ts_to_date(coin_fts),
                        int(minimum_coin_age_days),
                    )
                    return None
                if effective_start_ts > base_start_ts:
                    logging.info(
                        "%s: adjusting start from %s to %s (coin first available: %s, min_age=%d days)",
                        coin,
                        ts_to_date(base_start_ts),
                        ts_to_date(effective_start_ts),
                        ts_to_date(coin_fts),
                        int(minimum_coin_age_days),
                    )

                # Try matching coin directly, then try base coin without xyz: prefix
                forced_exchange = forced_sources.get(coin)
                if forced_exchange is None and coin.startswith("xyz:"):
                    base_coin = coin[4:]  # Remove "xyz:" prefix
                    forced_exchange = forced_sources.get(base_coin)
                candidate_exchanges = [forced_exchange] if forced_exchange else exchanges_to_consider
                for ex in candidate_exchanges:
                    if ex not in om_dict:
                        raise ValueError(f"Unknown exchange '{ex}' requested for coin {coin}")

                # Fetch from all candidate exchanges in parallel
                tasks = []
                position_map = {ex0: (1 + i) for i, ex0 in enumerate(exchanges_to_consider)}
                for ex in candidate_exchanges:
                    tasks.append(
                        asyncio.create_task(
                            fetch_data_for_coin_and_exchange(
                                coin,
                                ex,
                                om_dict[ex],
                                effective_start_ts,
                                end_ts,
                                progress_position=position_map.get(ex, 1),
                            )
                        )
                    )
                results = await asyncio.gather(*tasks, return_exceptions=True)

                # Filter successful results
                exchange_candidates = []
                for r in results:
                    if r is None or isinstance(r, Exception):
                        continue
                    ex, df, coverage_count, gap_count, total_volume = r
                    exchange_candidates.append((ex, df, coverage_count, gap_count, total_volume))

                if not exchange_candidates:
                    if forced_exchange:
                        raise ValueError(
                            f"No exchange data found for coin {coin} on forced exchange {forced_exchange}."
                        )
                    logging.info(f"No exchange data found at all for coin {coin}. Skipping.")
                    return None

                # Select best exchange
                if forced_exchange:
                    chosen = [c for c in exchange_candidates if c[0] == forced_exchange]
                    if not chosen:
                        raise ValueError(
                            f"Forced exchange {forced_exchange} returned no usable data for coin {coin}."
                        )
                    best_exchange, best_df, best_cov, best_gaps, best_vol = chosen[0]
                elif len(exchange_candidates) == 1:
                    best_exchange, best_df, best_cov, best_gaps, best_vol = exchange_candidates[0]
                else:
                    exchange_candidates.sort(key=lambda x: (x[2], -x[3], x[4]), reverse=True)
                    best_exchange, best_df, best_cov, best_gaps, best_vol = exchange_candidates[0]

                logging.info(f"{coin} exchange preference: {[x[0] for x in exchange_candidates]}")

                # Determine market settings source (may differ from OHLCV source)
                settings_exchange = market_settings_sources.get(coin, best_exchange)
                if settings_exchange != best_exchange:
                    settings_om = om_dict.get(settings_exchange)
                    if settings_om is None:
                        logging.warning(
                            f"{coin}: market_settings_sources exchange '{settings_exchange}' "
                            f"not available, falling back to OHLCV source '{best_exchange}'"
                        )
                        settings_exchange = best_exchange
                    else:
                        try:
                            settings_om.get_symbol(coin)
                        except Exception:
                            logging.warning(
                                f"{coin}: not listed on market_settings_sources exchange "
                                f"'{settings_exchange}', falling back to OHLCV source '{best_exchange}'"
                            )
                            settings_exchange = best_exchange

                # Prepare market settings from (possibly overridden) exchange
                mss = om_dict[settings_exchange].get_market_specific_settings(coin)
                mss["exchange"] = to_standard_exchange_name(settings_exchange)
                if settings_exchange != best_exchange:
                    mss["ohlcv_source"] = to_standard_exchange_name(best_exchange)
                    logging.info(
                        f"{coin}: OHLCV from {best_exchange}, market settings from {settings_exchange}"
                    )
                warm_minutes = int(per_coin_warmups.get(coin, default_warm))
                mss["warmup_minutes"] = warm_minutes

                return (coin, best_df, mss)

            except Exception as e:
                logging.error(f"Error processing coin {coin}: {e}")
                traceback.print_exc()
                return None

    # Parallelize coin processing with rate limiting
    # Use semaphore of 6 to respect API rate limits across all exchanges
    COIN_CONCURRENCY = 6
    sem = asyncio.Semaphore(COIN_CONCURRENCY)

    tasks = [process_coin(coin, sem) for coin in coins]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Process results
    for result in results:
        progress.maybe_log()

        if result is None or isinstance(result, Exception):
            progress.update()
            continue

        coin, df, mss = result
        chosen_data_per_coin[coin] = df
        chosen_mss_per_coin[coin] = mss
        progress.update()

    progress.log_done()

    if not chosen_data_per_coin:
        raise ValueError("No coin data found on any exchange for the requested date range.")

    global_start_time = min(df.timestamp.iloc[0] for df in chosen_data_per_coin.values())
    global_end_time = max(df.timestamp.iloc[-1] for df in chosen_data_per_coin.values())

    timestamps = np.arange(global_start_time, global_end_time + interval_ms, interval_ms)
    n_timesteps = len(timestamps)
    valid_coins = sorted(chosen_data_per_coin.keys())
    n_coins = len(valid_coins)

    start_date_for_volume_ratios = ts_to_date(
        max(global_start_time, global_end_time - 1000 * 60 * 60 * 24 * 60)
    )
    end_date_for_volume_ratios = ts_to_date(global_end_time)

    # Use OHLCV sources for volume ratio calculation, not market settings sources
    exchanges_with_data = sorted(set([
        chosen_mss_per_coin[coin].get("ohlcv_source", chosen_mss_per_coin[coin]["exchange"]) 
        for coin in valid_coins
    ]))
    exchange_volume_ratios = await compute_exchange_volume_ratios(
        exchanges_with_data,
        valid_coins,
        start_date_for_volume_ratios,
        end_date_for_volume_ratios,
        # om_dict keys are ccxt IDs (e.g. "binanceusdm"), but exchanges_with_data use standard names
        {ex: om_dict[to_ccxt_exchange_id(ex)] for ex in exchanges_with_data},
    )
    exchanges_counts = defaultdict(int)
    for coin in chosen_mss_per_coin:
        # Use OHLCV source for volume normalization, not market settings source
        ohlcv_exchange = chosen_mss_per_coin[coin].get("ohlcv_source", chosen_mss_per_coin[coin]["exchange"])
        exchanges_counts[ohlcv_exchange] += 1
    reference_exchange = sorted(exchanges_counts.items(), key=lambda x: x[1])[-1][0]
    exchange_volume_ratios_mapped = defaultdict(dict)
    if len(exchanges_counts) == 1:
        exchange_volume_ratios_mapped[reference_exchange][reference_exchange] = 1.0
    else:
        for ex0, ex1 in exchange_volume_ratios:
            exchange_volume_ratios_mapped[ex0][ex1] = 1 / exchange_volume_ratios[(ex0, ex1)]
            exchange_volume_ratios_mapped[ex1][ex0] = exchange_volume_ratios[(ex0, ex1)]
            exchange_volume_ratios_mapped[ex1][ex1] = 1.0
            exchange_volume_ratios_mapped[ex0][ex0] = 1.0

    # Log volume normalization ratios (used to scale volumes when combining multi-exchange data)
    if len(exchanges_counts) > 1:
        ratio_summary = ", ".join(
            f"{ex}={exchange_volume_ratios_mapped[ex][reference_exchange]:.3f}"
            for ex in sorted(exchanges_counts.keys())
            if ex != reference_exchange and ex in exchange_volume_ratios_mapped
        )
        logging.info(
            "volume normalization: reference=%s ratios=[%s] (coins per exchange: %s)",
            reference_exchange,
            ratio_summary,
            ", ".join(f"{ex}={cnt}" for ex, cnt in sorted(exchanges_counts.items())),
        )

    # Log exchange assignment summary (which exchange was chosen for which coins)
    if len(exchanges_counts) > 1:
        coins_by_exchange = defaultdict(list)
        for coin in valid_coins:
            # Use OHLCV source for grouping, not market settings source
            ohlcv_ex = chosen_mss_per_coin[coin].get("ohlcv_source", chosen_mss_per_coin[coin]["exchange"])
            coins_by_exchange[ohlcv_ex].append(symbol_to_coin(coin))
        for ex in sorted(coins_by_exchange.keys()):
            coins_list = coins_by_exchange[ex]
            logging.info(
                "[combined] chose %s for %d coins: %s",
                ex,
                len(coins_list),
                ", ".join(sorted(coins_list)),
            )

    unified_array = np.full((n_timesteps, n_coins, 4), np.nan, dtype=np.float64)

    for i, coin in enumerate(valid_coins):
        df = chosen_data_per_coin[coin].copy()
        df = df.set_index("timestamp").reindex(timestamps)
        # Use OHLCV source for volume normalization, not market settings source
        exchange_for_this_coin = chosen_mss_per_coin[coin].get("ohlcv_source", chosen_mss_per_coin[coin]["exchange"])
        scaling_factor = exchange_volume_ratios_mapped[exchange_for_this_coin][reference_exchange]
        df["volume"] *= scaling_factor

        coin_data = df[["high", "low", "close", "volume"]].values
        unified_array[:, i, :] = coin_data
        start_idx = int(
            (chosen_data_per_coin[coin].timestamp.iloc[0] - global_start_time) / interval_ms
        )
        end_idx = start_idx + len(chosen_data_per_coin[coin]) - 1
        chosen_mss_per_coin[coin]["first_valid_index"] = start_idx
        chosen_mss_per_coin[coin]["last_valid_index"] = end_idx
        first_idx = int(start_idx)
        last_idx = int(end_idx)
        warm_minutes = int(chosen_mss_per_coin[coin].get("warmup_minutes", 0))
        trade_start_idx = first_idx + warm_minutes
        if trade_start_idx > last_idx:
            trade_start_idx = last_idx
        chosen_mss_per_coin[coin]["trade_start_index"] = trade_start_idx

    return chosen_mss_per_coin, timestamps, unified_array


async def fetch_data_for_coin_and_exchange(
    coin: str,
    ex: str,
    om: HLCVManager,
    effective_start_ts: int,
    end_ts: int,
    *,
    progress_position: int = 1,
):
    t0 = time.monotonic()
    # Calculate approximate number of days for better visibility
    days_approx = max(1, (end_ts - effective_start_ts) // (24 * 60 * 60 * 1000))

    # Stock perps (xyz:*) are only available on Hyperliquid
    if coin.startswith("xyz:") and ex != "hyperliquid":
        logging.debug(
            "%s candles fetch skip coin=%s reason=stock_perp_only_on_hyperliquid",
            ex,
            coin,
        )
        return None

    logging.info(
        "%s candles fetch start coin=%s range=%s..%s (~%d days)",
        ex,
        coin,
        ts_to_date(effective_start_ts),
        ts_to_date(end_ts),
        days_approx,
    )

    if not om.has_coin(coin):
        logging.info(
            "%s candles fetch skip coin=%s reason=missing_market elapsed_s=%.1f",
            ex,
            coin,
            time.monotonic() - t0,
        )
        return None
    om.update_date_range(effective_start_ts, end_ts)
    try:
        df = await om.get_ohlcvs(coin)
    except Exception as e:
        logging.warning(f"Error retrieving {coin} from {ex}: {e}")
        traceback.print_exc()
        logging.info(
            "%s candles fetch failed coin=%s elapsed_s=%.1f",
            ex,
            coin,
            time.monotonic() - t0,
        )
        return None
    if df.empty:
        logging.info(
            "%s candles fetch empty coin=%s elapsed_s=%.1f",
            ex,
            coin,
            time.monotonic() - t0,
        )
        return None
    df = df[(df.timestamp >= effective_start_ts) & (df.timestamp <= end_ts)].reset_index(drop=True)
    if df.empty:
        logging.info(
            "%s candles fetch empty_after_clip coin=%s elapsed_s=%.1f",
            ex,
            coin,
            time.monotonic() - t0,
        )
        return None
    coverage_count = len(df)
    intervals = np.diff(df["timestamp"].values)
    gap_count = sum((gap // 60_000) - 1 for gap in intervals if gap > 60_000)
    total_volume = df["volume"].sum()
    logging.info(
        "%s candles fetch ok coin=%s rows=%d gaps=%d elapsed_s=%.1f",
        ex,
        coin,
        coverage_count,
        gap_count,
        time.monotonic() - t0,
    )
    return (ex, df, coverage_count, gap_count, total_volume)


async def compute_exchange_volume_ratios(
    exchanges: List[str],
    coins: List[str],
    start_date: str,
    end_date: str,
    om_dict: Dict[str, HLCVManager] = None,
) -> Dict[Tuple[str, str], float]:
    if om_dict is None:
        om_dict = {ex: HLCVManager(ex, start_date, end_date) for ex in exchanges}
        await asyncio.gather(*[om_dict[ex].load_markets() for ex in om_dict])
    assert all([ex in om_dict for ex in exchanges])
    exchange_pairs = []
    for i, ex0 in enumerate(sorted(exchanges)):
        for ex1 in exchanges[i + 1 :]:
            exchange_pairs.append((ex0, ex1))

    all_data = {}

    for coin in coins:
        if not all(om_dict[ex].has_coin(coin) for ex in exchanges):
            continue

        tasks = []
        for ex in exchanges:
            om = om_dict[ex]
            om.update_date_range(start_date, end_date)
            tasks.append(om.get_ohlcvs(coin))

        dfs = await asyncio.gather(*tasks, return_exceptions=True)
        for i, df in enumerate(dfs):
            if isinstance(df, Exception) or df is None or df.empty:
                dfs[i] = pd.DataFrame()

        if any(df.empty for df in dfs):
            continue

        daily_volumes = []
        for df in dfs:
            df["day"] = df["timestamp"] // 86_400_000
            grouped = df.groupby("day", as_index=False)["volume"].sum()
            daily_dict = dict(zip(grouped["day"], grouped["volume"]))
            daily_volumes.append(daily_dict)

        sets_of_days = [set(dv.keys()) for dv in daily_volumes]
        common_days = set.intersection(*sets_of_days)
        if not common_days:
            continue

        coin_data = {}
        for ex0, ex1 in exchange_pairs:
            i0 = exchanges.index(ex0)
            i1 = exchanges.index(ex1)
            sum0 = sum(daily_volumes[i0][day] for day in common_days)
            sum1 = sum(daily_volumes[i1][day] for day in common_days)
            ratio = (sum0 / sum1) if sum1 > 0 else 0.0
            coin_data[(ex0, ex1)] = ratio

        if coin_data:
            all_data[coin] = coin_data

    averages = {}
    if not all_data:
        return averages

    used_pairs = set()
    for coin in all_data:
        for pair in all_data[coin]:
            used_pairs.add(pair)

    for pair in used_pairs:
        ratios_for_pair = []
        for coin in all_data:
            if pair in all_data[coin]:
                ratios_for_pair.append(all_data[coin][pair])
        averages[pair] = float(np.mean(ratios_for_pair)) if ratios_for_pair else 0.0

    return averages


__all__ = [
    "HLCVManager",
    "prepare_hlcvs",
    "prepare_hlcvs_combined",
    "compute_exchange_volume_ratios",
]
