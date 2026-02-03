"""
TradFi Data Provider

Fetches traditional finance OHLCV data from external APIs (Finnhub, Alpha Vantage).
Used for historical backtesting of stock perpetuals when native perp data is unavailable.

Symbol Mapping:
- xyz:TSLA (Hyperliquid HIP-3) -> TSLA (TradFi)
- xyz:NVDA (Hyperliquid HIP-3) -> NVDA (TradFi)

Note: TradFi data represents actual stock prices without:
- Perpetual funding rates
- Oracle-driven pricing during market closure
- Weekend/after-hours trading

Use this data for pre-HIP3 historical backtesting only.
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta, UTC
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import numpy as np

# OHLCV dtype matching CandlestickManager
CANDLE_DTYPE = np.dtype(
    [
        ("ts", "int64"),  # UTC milliseconds
        ("o", "float32"),  # open
        ("h", "float32"),  # high
        ("l", "float32"),  # low
        ("c", "float32"),  # close
        ("bv", "float32"),  # base volume
    ]
)

ONE_MIN_MS = 60_000
ONE_HOUR_MS = 3_600_000
ONE_DAY_MS = 86_400_000

logger = logging.getLogger(__name__)


def hip3_to_tradfi_symbol(hip3_symbol: str) -> str:
    """Convert HIP-3 symbol to TradFi ticker.

    Args:
        hip3_symbol: HIP-3 symbol like "xyz:TSLA", "XYZ-TSLA/USDC:USDC", etc.

    Returns:
        TradFi ticker like "TSLA"
    """
    # Extract base from CCXT-style symbol
    if "/" in hip3_symbol:
        base = hip3_symbol.split("/")[0]
    else:
        base = hip3_symbol

    # Handle various HIP-3 prefixes:
    # - xyz:TSLA (lowercase prefix with colon)
    # - XYZ-TSLA (CCXT format with hyphen)
    # - XYZ:TSLA (uppercase with colon)
    if base.startswith("xyz:"):
        return base[4:]
    if base.startswith("XYZ-"):
        return base[4:]
    if base.startswith("XYZ:"):
        return base[4:]

    return base


def tradfi_to_hip3_symbol(tradfi_symbol: str, quote: str = "USDC") -> str:
    """Convert TradFi ticker to HIP-3 symbol.

    Args:
        tradfi_symbol: TradFi ticker like "TSLA"
        quote: Quote currency (default: USDC)

    Returns:
        HIP-3 symbol like "xyz:TSLA/USDC:USDC"
    """
    return f"xyz:{tradfi_symbol}/{quote}:{quote}"


@dataclass
class TradFiCandle:
    """Single OHLCV candle from TradFi source."""

    timestamp_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float


class TradFiProvider(ABC):
    """Abstract base class for TradFi data providers."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self._session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        timeout = aiohttp.ClientTimeout(total=60, connect=15)
        self._session = aiohttp.ClientSession(timeout=timeout)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._session is not None:
            await self._session.close()
            self._session = None

    @abstractmethod
    async def fetch_1m_candles(
        self,
        symbol: str,
        start_ts: int,
        end_ts: int,
    ) -> List[TradFiCandle]:
        """Fetch 1-minute candles for a symbol.

        Args:
            symbol: TradFi ticker (e.g., "TSLA")
            start_ts: Start timestamp (ms)
            end_ts: End timestamp (ms)

        Returns:
            List of TradFiCandle objects
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name for logging."""
        pass

    @property
    @abstractmethod
    def rate_limit_delay(self) -> float:
        """Minimum delay between API calls (seconds)."""
        pass


class FinnhubProvider(TradFiProvider):
    """Finnhub API provider for TradFi data.

    Free tier: 60 API calls/minute
    Docs: https://finnhub.io/docs/api/stock-candles
    """

    BASE_URL = "https://finnhub.io/api/v1"

    @property
    def name(self) -> str:
        return "finnhub"

    @property
    def rate_limit_delay(self) -> float:
        return 1.1  # ~54 calls/min to stay safe

    async def fetch_1m_candles(
        self,
        symbol: str,
        start_ts: int,
        end_ts: int,
    ) -> List[TradFiCandle]:
        if not self.api_key:
            raise ValueError("Finnhub API key required")
        if self._session is None:
            raise RuntimeError("Session not initialized. Use 'async with' context.")

        candles = []
        # Finnhub uses seconds, not milliseconds
        from_ts = start_ts // 1000
        to_ts = end_ts // 1000

        url = f"{self.BASE_URL}/stock/candle"
        params = {
            "symbol": symbol,
            "resolution": "1",  # 1 minute
            "from": from_ts,
            "to": to_ts,
            "token": self.api_key,
        }

        try:
            async with self._session.get(url, params=params) as resp:
                if resp.status == 429:
                    logger.warning("Finnhub rate limit hit, backing off")
                    await asyncio.sleep(60)
                    return []
                resp.raise_for_status()
                data = await resp.json()

            if data.get("s") != "ok":
                logger.debug("Finnhub no data for %s: %s", symbol, data.get("s"))
                return []

            timestamps = data.get("t", [])
            opens = data.get("o", [])
            highs = data.get("h", [])
            lows = data.get("l", [])
            closes = data.get("c", [])
            volumes = data.get("v", [])

            for i in range(len(timestamps)):
                candles.append(
                    TradFiCandle(
                        timestamp_ms=timestamps[i] * 1000,
                        open=opens[i],
                        high=highs[i],
                        low=lows[i],
                        close=closes[i],
                        volume=volumes[i],
                    )
                )

            logger.debug(
                "Finnhub fetched %d candles for %s (%s - %s)",
                len(candles),
                symbol,
                datetime.fromtimestamp(from_ts, tz=UTC).isoformat(),
                datetime.fromtimestamp(to_ts, tz=UTC).isoformat(),
            )

        except aiohttp.ClientError as e:
            logger.warning("Finnhub API error for %s: %s", symbol, e)

        return candles


class AlphaVantageProvider(TradFiProvider):
    """Alpha Vantage API provider for TradFi data.

    Free tier: 25 API calls/day (very limited)
    Docs: https://www.alphavantage.co/documentation/
    """

    BASE_URL = "https://www.alphavantage.co/query"

    @property
    def name(self) -> str:
        return "alphavantage"

    @property
    def rate_limit_delay(self) -> float:
        return 12.0  # Very conservative for free tier

    async def fetch_1m_candles(
        self,
        symbol: str,
        start_ts: int,
        end_ts: int,
    ) -> List[TradFiCandle]:
        if not self.api_key:
            raise ValueError("Alpha Vantage API key required")
        if self._session is None:
            raise RuntimeError("Session not initialized. Use 'async with' context.")

        candles = []
        # Alpha Vantage returns data by month, request current month first
        params = {
            "function": "TIME_SERIES_INTRADAY",
            "symbol": symbol,
            "interval": "1min",
            "outputsize": "full",
            "apikey": self.api_key,
        }

        try:
            async with self._session.get(self.BASE_URL, params=params) as resp:
                if resp.status == 429:
                    logger.warning("Alpha Vantage rate limit hit")
                    return []
                resp.raise_for_status()
                data = await resp.json()

            # Check for rate limit message
            if "Note" in data or "Information" in data:
                logger.warning(
                    "Alpha Vantage rate limit: %s",
                    data.get("Note", data.get("Information")),
                )
                return []

            time_series = data.get("Time Series (1min)", {})
            if not time_series:
                logger.debug("Alpha Vantage no data for %s", symbol)
                return []

            for timestamp_str, values in time_series.items():
                # Parse timestamp (format: "2025-01-15 16:00:00")
                dt = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                # Alpha Vantage returns US Eastern time, convert to UTC
                # This is a simplification - proper timezone handling would need pytz
                ts_ms = int(dt.timestamp() * 1000)

                if start_ts <= ts_ms <= end_ts:
                    candles.append(
                        TradFiCandle(
                            timestamp_ms=ts_ms,
                            open=float(values["1. open"]),
                            high=float(values["2. high"]),
                            low=float(values["3. low"]),
                            close=float(values["4. close"]),
                            volume=float(values["5. volume"]),
                        )
                    )

            logger.debug(
                "Alpha Vantage fetched %d candles for %s", len(candles), symbol
            )

        except aiohttp.ClientError as e:
            logger.warning("Alpha Vantage API error for %s: %s", symbol, e)

        return sorted(candles, key=lambda c: c.timestamp_ms)


class PolygonProvider(TradFiProvider):
    """Polygon.io (Massive) API provider for TradFi data.

    Free tier: 2 years of 1m historical data
    Rate limit: 5 API calls/minute
    Max 50,000 results per query (~35 days of 1m bars)

    Docs: https://polygon.readthedocs.io/en/latest/Stocks.html
    """

    BASE_URL = "https://api.polygon.io/v2/aggs/ticker"

    @property
    def name(self) -> str:
        return "polygon"

    @property
    def rate_limit_delay(self) -> float:
        return 12.5  # 5 calls/min = 1 per 12 seconds, add buffer

    async def fetch_1m_candles(
        self,
        symbol: str,
        start_ts: int,
        end_ts: int,
    ) -> List[TradFiCandle]:
        if not self.api_key:
            raise ValueError("Polygon API key required")
        if self._session is None:
            raise RuntimeError("Session not initialized. Use 'async with' context.")

        candles = []

        # Polygon API uses timestamps in milliseconds
        # Build URL for aggregates endpoint
        url = f"{self.BASE_URL}/{symbol}/range/1/minute/{start_ts}/{end_ts}"
        params = {
            "adjusted": "true",
            "sort": "asc",
            "limit": 50000,  # Max allowed
            "apiKey": self.api_key,
        }

        try:
            async with self._session.get(url, params=params) as resp:
                if resp.status == 429:
                    logger.warning("Polygon rate limit hit, backing off")
                    await asyncio.sleep(60)
                    return []
                if resp.status == 403:
                    logger.warning("Polygon API key invalid or unauthorized")
                    return []
                resp.raise_for_status()
                data = await resp.json()

            if data.get("status") != "OK":
                logger.debug(
                    "Polygon no data for %s: status=%s", symbol, data.get("status")
                )
                return []

            results = data.get("results", [])
            if not results:
                logger.debug("Polygon no results for %s in range", symbol)
                return []

            for bar in results:
                # Polygon returns: t (timestamp ms), o, h, l, c, v
                candles.append(
                    TradFiCandle(
                        timestamp_ms=bar["t"],
                        open=bar["o"],
                        high=bar["h"],
                        low=bar["l"],
                        close=bar["c"],
                        volume=bar.get("v", 0),
                    )
                )

            logger.debug(
                "Polygon fetched %d candles for %s (%s - %s)",
                len(candles),
                symbol,
                datetime.fromtimestamp(start_ts / 1000, tz=UTC).isoformat(),
                datetime.fromtimestamp(end_ts / 1000, tz=UTC).isoformat(),
            )

        except aiohttp.ClientError as e:
            logger.warning("Polygon API error for %s: %s", symbol, e)

        return candles


class AlpacaProvider(TradFiProvider):
    """Alpaca Markets API provider for TradFi data.

    FREE - No payment required, just free API keys!
    - 5+ years of historical 1m data
    - Free tier uses IEX data feed
    - 15-minute delay (doesn't matter for backtesting)
    - Rate limit: 200 requests/minute

    Docs: https://docs.alpaca.markets/docs/about-market-data-api
    Sign up: https://alpaca.markets/
    """

    BASE_URL = "https://data.alpaca.markets/v2/stocks"

    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        super().__init__(api_key)
        self.api_secret = api_secret

    @property
    def name(self) -> str:
        return "alpaca"

    @property
    def rate_limit_delay(self) -> float:
        return 0.5  # 200 req/min = 3.3 req/sec, be conservative

    async def fetch_1m_candles(
        self,
        symbol: str,
        start_ts: int,
        end_ts: int,
    ) -> List[TradFiCandle]:
        if not self.api_key or not self.api_secret:
            raise ValueError("Alpaca API key and secret required")
        if self._session is None:
            raise RuntimeError("Session not initialized. Use 'async with' context.")

        candles = []

        # Alpaca uses RFC3339 timestamps
        start_dt = datetime.fromtimestamp(start_ts / 1000, tz=UTC)
        end_dt = datetime.fromtimestamp(end_ts / 1000, tz=UTC)

        url = f"{self.BASE_URL}/{symbol}/bars"
        params = {
            "timeframe": "1Min",
            "start": start_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end": end_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "limit": 10000,  # Max per request
            "adjustment": "split",
            "feed": "iex",  # Free tier uses IEX
        }
        headers = {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.api_secret,
        }

        try:
            next_page_token = None
            while True:
                if next_page_token:
                    params["page_token"] = next_page_token

                async with self._session.get(url, params=params, headers=headers) as resp:
                    if resp.status == 429:
                        logger.warning("Alpaca rate limit hit, backing off")
                        await asyncio.sleep(60)
                        return candles
                    if resp.status == 403:
                        logger.warning("Alpaca API key invalid or unauthorized")
                        return []
                    if resp.status == 422:
                        # Unprocessable entity - often means no data for range
                        logger.debug("Alpaca no data for %s in range", symbol)
                        return []
                    resp.raise_for_status()
                    data = await resp.json()

                bars = data.get("bars", [])
                if not bars:
                    break

                for bar in bars:
                    # Parse ISO timestamp to milliseconds
                    ts_str = bar["t"]
                    dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                    ts_ms = int(dt.timestamp() * 1000)

                    candles.append(
                        TradFiCandle(
                            timestamp_ms=ts_ms,
                            open=bar["o"],
                            high=bar["h"],
                            low=bar["l"],
                            close=bar["c"],
                            volume=bar.get("v", 0),
                        )
                    )

                # Check for pagination
                next_page_token = data.get("next_page_token")
                if not next_page_token:
                    break

            logger.debug(
                "Alpaca fetched %d candles for %s (%s - %s)",
                len(candles),
                symbol,
                start_dt.isoformat(),
                end_dt.isoformat(),
            )

        except aiohttp.ClientError as e:
            logger.warning("Alpaca API error for %s: %s", symbol, e)

        return candles


class YFinanceProvider(TradFiProvider):
    """Yahoo Finance provider for TradFi data.

    FREE - No API key required!
    Limitations:
    - 1m data: last 7 days only
    - 5m data: last 60 days
    - 1h data: last 730 days (2 years)
    - 1d data: full history

    Docs: https://github.com/ranaroussi/yfinance
    """

    @property
    def name(self) -> str:
        return "yfinance"

    @property
    def rate_limit_delay(self) -> float:
        return 0.5  # Be nice to Yahoo

    async def fetch_1m_candles(
        self,
        symbol: str,
        start_ts: int,
        end_ts: int,
    ) -> List[TradFiCandle]:
        """Fetch 1m candles from Yahoo Finance.

        Note: yfinance only provides 1m data for the last 7 days.
        """
        try:
            import yfinance as yf
        except ImportError:
            logger.warning("yfinance not installed. Install with: pip install yfinance")
            return []

        candles = []

        try:
            # yfinance uses synchronous API, run in executor
            import asyncio

            loop = asyncio.get_event_loop()

            def fetch_sync():
                ticker = yf.Ticker(symbol)
                # Convert timestamps to datetime
                start_dt = datetime.fromtimestamp(start_ts / 1000, tz=UTC)
                end_dt = datetime.fromtimestamp(end_ts / 1000, tz=UTC)

                # yfinance 1m data is limited to last 7 days
                seven_days_ago = datetime.now(UTC) - timedelta(days=7)
                if start_dt < seven_days_ago:
                    start_dt = seven_days_ago
                    logger.debug(
                        "yfinance 1m data limited to 7 days, adjusted start to %s",
                        start_dt,
                    )

                # Fetch data
                df = ticker.history(
                    interval="1m",
                    start=start_dt,
                    end=end_dt,
                )
                return df

            df = await loop.run_in_executor(None, fetch_sync)

            if df is None or df.empty:
                logger.debug("yfinance no data for %s", symbol)
                return []

            # Convert DataFrame to TradFiCandle list
            for idx, row in df.iterrows():
                # idx is a timezone-aware datetime
                ts_ms = int(idx.timestamp() * 1000)

                if start_ts <= ts_ms <= end_ts:
                    candles.append(
                        TradFiCandle(
                            timestamp_ms=ts_ms,
                            open=float(row["Open"]),
                            high=float(row["High"]),
                            low=float(row["Low"]),
                            close=float(row["Close"]),
                            volume=float(row["Volume"]),
                        )
                    )

            logger.debug("yfinance fetched %d candles for %s", len(candles), symbol)

        except Exception as e:
            logger.warning("yfinance API error for %s: %s", symbol, e)

        return sorted(candles, key=lambda c: c.timestamp_ms)


def get_provider(
    name: str, api_key: Optional[str] = None, api_secret: Optional[str] = None
) -> TradFiProvider:
    """Factory function to get a TradFi data provider.

    Args:
        name: Provider name ("alpaca", "polygon", "yfinance", "finnhub", "alphavantage")
        api_key: API key for the provider (not needed for yfinance)
        api_secret: API secret for Alpaca (required for alpaca provider)

    Returns:
        TradFiProvider instance
    """
    if name == "alpaca":
        return AlpacaProvider(api_key=api_key, api_secret=api_secret)

    providers = {
        "polygon": PolygonProvider,
        "yfinance": YFinanceProvider,
        "finnhub": FinnhubProvider,
        "alphavantage": AlphaVantageProvider,
    }

    if name not in providers:
        raise ValueError(f"Unknown provider: {name}. Available: {list(providers.keys())}")

    return providers[name](api_key=api_key)


def candles_to_array(candles: List[TradFiCandle]) -> np.ndarray:
    """Convert TradFiCandle list to numpy structured array.

    Args:
        candles: List of TradFiCandle objects

    Returns:
        Structured array with CANDLE_DTYPE
    """
    if not candles:
        return np.empty((0,), dtype=CANDLE_DTYPE)

    arr = np.empty((len(candles),), dtype=CANDLE_DTYPE)
    for i, c in enumerate(candles):
        arr[i] = (c.timestamp_ms, c.open, c.high, c.low, c.close, c.volume)

    return arr


class TradFiDataFetcher:
    """High-level fetcher for TradFi data with caching and rate limiting."""

    def __init__(
        self,
        provider: TradFiProvider,
        cache_dir: Optional[str] = None,
    ):
        self.provider = provider
        self.cache_dir = cache_dir
        self._last_request_time = 0.0

    async def __aenter__(self):
        await self.provider.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.provider.__aexit__(exc_type, exc_val, exc_tb)

    async def _rate_limit_wait(self):
        """Wait to respect rate limits."""
        elapsed = time.monotonic() - self._last_request_time
        delay = self.provider.rate_limit_delay
        if elapsed < delay:
            await asyncio.sleep(delay - elapsed)
        self._last_request_time = time.monotonic()

    async def fetch_day(
        self,
        hip3_symbol: str,
        day_key: str,
    ) -> np.ndarray:
        """Fetch a full day of 1m candles for a HIP-3 symbol.

        Args:
            hip3_symbol: HIP-3 symbol (e.g., "xyz:TSLA/USDC:USDC")
            day_key: Date string (YYYY-MM-DD)

        Returns:
            Structured array with CANDLE_DTYPE (may be sparse for market hours only)
        """
        tradfi_symbol = hip3_to_tradfi_symbol(hip3_symbol)

        # Calculate day boundaries (UTC)
        day_start = datetime.strptime(day_key, "%Y-%m-%d").replace(tzinfo=UTC)
        start_ts = int(day_start.timestamp() * 1000)
        end_ts = start_ts + ONE_DAY_MS - ONE_MIN_MS

        await self._rate_limit_wait()

        candles = await self.provider.fetch_1m_candles(tradfi_symbol, start_ts, end_ts)
        if not candles:
            logger.info(
                "No TradFi data for %s on %s (possibly non-trading day)",
                tradfi_symbol,
                day_key,
            )
            return np.empty((0,), dtype=CANDLE_DTYPE)

        arr = candles_to_array(candles)
        logger.info(
            "Fetched %d TradFi candles for %s on %s",
            len(arr),
            tradfi_symbol,
            day_key,
        )
        return arr

    async def fetch_range(
        self,
        hip3_symbol: str,
        start_date: str,
        end_date: str,
    ) -> Dict[str, np.ndarray]:
        """Fetch candles for a date range.

        Args:
            hip3_symbol: HIP-3 symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            Dict mapping day keys to candle arrays
        """
        results = {}
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        current = start
        while current <= end:
            day_key = current.strftime("%Y-%m-%d")
            arr = await self.fetch_day(hip3_symbol, day_key)
            if arr.size > 0:
                results[day_key] = arr
            current += timedelta(days=1)

        return results


# Known stock tickers available as HIP-3 perps on Hyperliquid/TradeXYZ
# These can be used with or without the xyz: prefix
KNOWN_STOCK_TICKERS = {
    "TSLA",
    "NVDA",
    "AAPL",
    "MSFT",
    "META",
    "AMZN",
    "GOOGL",
    "PLTR",
    "COIN",
    "AMD",
    "NFLX",
    "HOOD",
    "CRCL",
    "SBET",
    "XYZ100",  # Nasdaq-like index
}

# Available stock perps with xyz: prefix format
AVAILABLE_STOCK_PERPS = [f"xyz:{ticker}" for ticker in KNOWN_STOCK_TICKERS]


def is_stock_ticker(coin: str) -> bool:
    """Check if a coin name is a known stock ticker.

    This allows users to simply add "TSLA" to approved_coins without
    needing to know the xyz: prefix.

    Args:
        coin: Coin name (e.g., "TSLA", "xyz:TSLA", "XYZ-TSLA", "BTC")

    Returns:
        True if this is a known stock ticker
    """
    # Remove any HIP-3 prefix
    if coin.startswith("xyz:"):
        coin = coin[4:]
    elif coin.startswith("XYZ-"):
        coin = coin[4:]
    elif coin.startswith("XYZ:"):
        coin = coin[4:]

    # Remove any quote suffix (e.g., from CCXT symbols)
    if "/" in coin:
        coin = coin.split("/")[0]

    return coin.upper() in KNOWN_STOCK_TICKERS


def is_stock_perp_symbol(symbol: str) -> bool:
    """Check if a symbol is a stock perp.

    Detects stock perps by:
    1. xyz: or XYZ- prefix (HIP-3 format)
    2. Known stock ticker name (TSLA, NVDA, etc.)

    Args:
        symbol: CCXT-style symbol or coin name

    Returns:
        True if this is a stock perp symbol
    """
    # Check for HIP-3 prefixes
    if symbol.startswith("xyz:") or symbol.startswith("XYZ-") or symbol.startswith("XYZ:"):
        return True
    base = symbol.split("/")[0] if "/" in symbol else symbol
    if base.startswith("xyz:") or base.startswith("XYZ-") or base.startswith("XYZ:"):
        return True

    # Check if it's a known stock ticker
    return is_stock_ticker(symbol)
