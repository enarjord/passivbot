import asyncio
import json
import random
import traceback
from copy import deepcopy

import ccxt.pro as ccxt_pro
import ccxt.async_support as ccxt_async
import passivbot_rust as pbr
from ccxt.base.errors import RateLimitExceeded

from exchanges.ccxt_bot import CCXTBot, format_exchange_config_response
from passivbot import logging
from passivbot_exceptions import FatalBotException
from utils import ts_to_date, utc_ms
from config.access import require_live_value
from pure_funcs import calc_hash
from procedures import print_async_exception, assert_correct_ccxt_version

round_ = pbr.round_
round_dynamic = pbr.round_dynamic
round_dynamic_up = pbr.round_dynamic_up
round_dynamic_dn = pbr.round_dynamic_dn

assert_correct_ccxt_version(ccxt=ccxt_async)


class HyperliquidBot(CCXTBot):
    # HIP-3 stock perps have a max leverage of 10x
    HIP3_MAX_LEVERAGE = 10
    # HIP-3 symbols use "xyz:" prefix (TradeXYZ builder)
    HIP3_PREFIX = "xyz:"
    HIP3_ALT_PREFIXES = ("XYZ-", "XYZ:")
    HIP3_ISOLATED_SUPPORTED = False
    HIP3_FULL_DEX_SWEEP_INTERVAL_MS = 300_000

    def __init__(self, config: dict):
        super().__init__(config)
        self.quote = "USDC"
        self.hedge_mode = False
        self.significant_digits = {}
        self._hl_live_margin_modes = {}
        self._hl_force_full_dex_sweep_surfaces = set()
        self._hl_last_full_dex_sweep_ms_by_surface = {}
        if "is_vault" not in self.user_info or self.user_info["is_vault"] == "":
            logging.info(
                f"parameter 'is_vault' missing from api-keys.json for user {self.user}. Setting to false"
            )
            self.user_info["is_vault"] = False
        self.max_n_concurrent_ohlcvs_1m_updates = 2
        self.custom_id_max_length = 34
        self._hl_fetch_lock = asyncio.Lock()
        self._hl_cache_generation = 0
        self._hl_user_abstraction = "unknown"
        self._hl_unified_enabled = False

    def _hl_state_fetch_concurrency(self) -> int:
        """Bound internal Hyperliquid account-state fanout to avoid rate-limit spikes."""
        return 4

    async def _hl_gather_limited(self, coros: list):
        """Run coroutines with bounded concurrency, preserving input order."""
        if not coros:
            return []
        semaphore = asyncio.Semaphore(max(1, int(self._hl_state_fetch_concurrency())))

        async def _run(coro):
            async with semaphore:
                return await coro

        return await asyncio.gather(*[_run(coro) for coro in coros])

    def _log_hl_fetch_breakdown(
        self,
        label: str,
        *,
        wall_ms: int,
        timings_ms: dict[str, int],
        extra_parts: list[str] | None = None,
    ) -> None:
        """Emit throttled INFO diagnostics for slow Hyperliquid account-state fetches."""
        if wall_ms < 5_000:
            return
        now_ms = utc_ms()
        if not hasattr(self, "_hl_fetch_breakdown_last_log_ms"):
            self._hl_fetch_breakdown_last_log_ms = {}
        last_ms = int(self._hl_fetch_breakdown_last_log_ms.get(label, 0) or 0)
        if now_ms - last_ms < 30_000:
            return
        self._hl_fetch_breakdown_last_log_ms[label] = now_ms
        parts = list(extra_parts or [])
        parts.extend(f"{key}={int(timings_ms[key])}ms" for key in sorted(timings_ms))
        log_level = logging.INFO if wall_ms >= 10_000 else logging.DEBUG
        logging.log(
            log_level,
            "[state] hyperliquid %s timings | wall=%dms | %s",
            label,
            wall_ms,
            " ".join(parts),
        )

    def _hl_info_url(self) -> str:
        """Derive the Hyperliquid /info endpoint from the CCXT session URL config."""
        base = self.cca.urls.get("api", {}).get("public", "https://api.hyperliquid.xyz")
        hostname = getattr(self.cca, "hostname", "hyperliquid.xyz")
        return base.replace("{hostname}", hostname).rstrip("/") + "/info"

    def _normalize_hl_user_abstraction(self, raw) -> str:
        """Normalize Hyperliquid userAbstraction response into a stable string."""
        if raw is None:
            return "unknown"
        text = str(raw).strip()
        if len(text) >= 2 and text[0] == text[-1] == '"':
            text = text[1:-1]
        return text or "unknown"

    async def fetch_user_abstraction_state(self) -> str:
        """Fetch and cache the Hyperliquid account abstraction mode."""
        wallet_address = str(self.user_info.get("wallet_address") or "")
        if not wallet_address:
            raise ValueError(f"user {self.user!r} missing wallet_address for Hyperliquid abstraction")
        raw = await self.cca.publicPostInfo({"type": "userAbstraction", "user": wallet_address})
        abstraction = self._normalize_hl_user_abstraction(raw)
        self._hl_user_abstraction = abstraction
        self._hl_unified_enabled = abstraction == "unifiedAccount"
        if hasattr(self, "cca") and getattr(self, "cca", None) is not None:
            self.cca.options["enableUnifiedMargin"] = bool(self._hl_unified_enabled)
        if hasattr(self, "ccp") and getattr(self, "ccp", None) is not None:
            self.ccp.options["enableUnifiedMargin"] = bool(self._hl_unified_enabled)
        return abstraction

    async def refresh_and_log_user_abstraction_state(self) -> str:
        """Refresh Hyperliquid account abstraction mode and log first sighting or changes."""
        abstraction = await self.fetch_user_abstraction_state()
        previous = getattr(self, "_hl_last_logged_user_abstraction", None)
        if previous is None:
            logging.info(
                "[account] Hyperliquid abstraction=%s | unified=%s",
                abstraction,
                "yes" if abstraction == "unifiedAccount" else "no",
            )
        elif previous != abstraction:
            logging.warning(
                "[account] Hyperliquid abstraction changed %s -> %s | unified=%s",
                previous,
                abstraction,
                "yes" if abstraction == "unifiedAccount" else "no",
            )
        self._hl_last_logged_user_abstraction = abstraction
        return abstraction

    def create_ccxt_sessions(self):
        creds = {
            "walletAddress": self.user_info["wallet_address"],
            "privateKey": self.user_info["private_key"],
        }
        # Configure fetchMarkets to include HIP-3 stock perps from TradeXYZ
        fetch_markets_config = {
            "types": ["swap", "hip3"],  # Include HIP-3 markets
            "hip3": {
                "dex": ["xyz"],  # TradeXYZ DEX for stock perps (TSLA, NVDA, etc.)
            },
        }
        if self.ws_enabled:
            self.ccp = getattr(ccxt_pro, self.exchange)(creds)
            self.ccp.options.update(self._build_ccxt_options())
            self.ccp.options["defaultType"] = "swap"
            self.ccp.options["fetchMarkets"] = fetch_markets_config
            self._apply_endpoint_override(self.ccp)
        elif self.endpoint_override:
            logging.info("Skipping Hyperliquid websocket session due to custom endpoint override.")
        self.cca = getattr(ccxt_async, self.exchange)(creds)
        self.cca.options.update(self._build_ccxt_options())
        self.cca.options["defaultType"] = "swap"
        self.cca.options["fetchMarkets"] = fetch_markets_config
        self._apply_endpoint_override(self.cca)

    def set_market_specific_settings(self):
        super().set_market_specific_settings()
        isolated_count = 0
        for symbol in self.markets_dict:
            elm = self.markets_dict[symbol]
            self.symbol_ids[symbol] = elm["id"]
            self.min_costs[symbol] = (
                10.0 if elm["limits"]["cost"]["min"] is None else elm["limits"]["cost"]["min"]
            )
            self.min_costs[symbol] = pbr.round_(self.min_costs[symbol] * 1.01, 0.01)
            self.qty_steps[symbol] = elm["precision"]["amount"]
            self.min_qtys[symbol] = (
                self.qty_steps[symbol]
                if elm["limits"]["amount"]["min"] is None
                else elm["limits"]["amount"]["min"]
            )
            self.price_steps[symbol] = elm["precision"]["price"]
            self.c_mults[symbol] = elm["contractSize"]

            # For isolated-only markets (HIP-3), cap at 10x leverage
            if self._requires_isolated_margin(symbol):
                isolated_count += 1
                self.max_leverage[symbol] = min(
                    self.HIP3_MAX_LEVERAGE,
                    (
                        int(elm["info"]["maxLeverage"])
                        if "maxLeverage" in elm["info"]
                        else self.HIP3_MAX_LEVERAGE
                    ),
                )
            else:
                self.max_leverage[symbol] = (
                    int(elm["info"]["maxLeverage"]) if "maxLeverage" in elm["info"] else 0
                )
        self.n_decimal_places = 6
        self.n_significant_figures = 5
        if isolated_count:
            logging.debug(
                f"Detected {isolated_count} isolated-margin-only symbols (HIP-3/stock perps)"
            )

    def _hip3_margin_metadata(self, symbol: str) -> dict:
        market = getattr(self, "markets_dict", {}).get(symbol, {})
        info = market.get("info", {})
        margin_modes = market.get("marginModes", {})
        raw_mode = str(info.get("marginMode") or "").strip()
        raw_mode_l = raw_mode.lower()
        only_isolated = bool(info.get("onlyIsolated") or info.get("isolatedOnly"))
        cross_capable = not only_isolated and raw_mode_l not in {"strictisolated", "nocross"}
        if isinstance(margin_modes, dict) and margin_modes.get("cross") is False:
            cross_capable = False
        return {
            "cross_capable": cross_capable,
            "only_isolated": only_isolated,
        }

    def _requires_isolated_margin(self, symbol: str) -> bool:
        """Check if a symbol requires isolated margin mode.

        On Hyperliquid, this includes:
        1. HIP-3 markets that are actually isolated-only by metadata
        2. Other markets with onlyIsolated=True flag

        Args:
            symbol: CCXT-style symbol (e.g., "xyz:TSLA/USDC:USDC")

        Returns:
            True if this symbol requires isolated margin mode
        """
        prefixes = (self.HIP3_PREFIX,) + tuple(self.HIP3_ALT_PREFIXES)
        base = symbol.split("/")[0] if "/" in symbol else symbol
        if (
            self._get_hl_dex_for_symbol(symbol)
            or symbol.startswith(prefixes)
            or base.startswith(prefixes)
        ):
            return not self._hip3_margin_metadata(symbol)["cross_capable"]

        # Fall back to base class check (onlyIsolated flag, etc.)
        return super()._requires_isolated_margin(symbol)

    def _record_hl_live_margin_mode(self, symbol: str, margin_mode: str | None) -> None:
        if not symbol or not margin_mode:
            return
        normalized = str(margin_mode).lower()
        if normalized in {"cross", "isolated"}:
            self._hl_live_margin_modes[symbol] = normalized

    def _get_hl_dex_for_symbol(self, symbol: str) -> str | None:
        """Return HIP-3 dex name for a symbol if available."""
        market = getattr(self, "markets_dict", {}).get(symbol, {})
        base_name = market.get("baseName") or market.get("info", {}).get("baseName", "")
        if isinstance(base_name, str) and ":" in base_name:
            dex_name = base_name.split(":", 1)[0]
            if dex_name:
                return dex_name
        return None

    def _get_hl_hip3_state_symbols(self) -> list[str]:
        """Return tracked HIP-3 symbols that need dex-scoped state queries."""
        tracked = set(getattr(self, "active_symbols", []) or [])
        tracked.update(getattr(self, "open_orders", {}).keys())
        tracked.update(getattr(self, "positions", {}).keys())
        return sorted(
            symbol
            for symbol in tracked
            if symbol in getattr(self, "markets_dict", {}) and self._get_hl_dex_for_symbol(symbol)
        )

    def _get_hl_hip3_dex_names(self) -> list[str]:
        dexes = set()
        for symbol in getattr(self, "markets_dict", {}) or {}:
            dex_name = self._get_hl_dex_for_symbol(symbol)
            if dex_name:
                dexes.add(dex_name)
        return sorted(dexes)

    def _get_hl_active_dex_names(self) -> list[str]:
        """Return HIP-3 dexes currently relevant to tracked live state."""
        dexes = set()
        for symbol in self._get_hl_hip3_state_symbols():
            dex_name = self._get_hl_dex_for_symbol(symbol)
            if dex_name:
                dexes.add(dex_name)
        return sorted(dexes)

    def _hl_should_force_full_dex_sweep(self, surface: str) -> bool:
        """Return True when the next HIP-3 refresh for a surface should sweep every dex."""
        if bool(getattr(self, "_hl_force_full_dex_sweep", False)):
            return True
        if surface in set(getattr(self, "_hl_force_full_dex_sweep_surfaces", set()) or set()):
            return True
        last_full_map = getattr(self, "_hl_last_full_dex_sweep_ms_by_surface", {}) or {}
        last_full = int(last_full_map.get(surface, 0) or 0)
        if last_full <= 0:
            return True
        return utc_ms() - last_full >= int(self.HIP3_FULL_DEX_SWEEP_INTERVAL_MS)

    def _hl_select_dex_names_for_state(self, surface: str) -> tuple[list[str], bool]:
        """Choose HIP-3 dexes for the next authoritative state query for one surface."""
        full_sweep = self._hl_should_force_full_dex_sweep(surface)
        dexes = self._get_hl_hip3_dex_names() if full_sweep else self._get_hl_active_dex_names()
        if not dexes and full_sweep:
            if not hasattr(self, "_hl_last_full_dex_sweep_ms_by_surface"):
                self._hl_last_full_dex_sweep_ms_by_surface = {}
            if not hasattr(self, "_hl_force_full_dex_sweep_surfaces"):
                self._hl_force_full_dex_sweep_surfaces = set()
            self._hl_last_full_dex_sweep_ms_by_surface[surface] = utc_ms()
            self._hl_force_full_dex_sweep = False
            self._hl_force_full_dex_sweep_surfaces.discard(surface)
        return dexes, full_sweep

    def _hl_mark_dex_scope_consumed(self, surface: str, *, full_sweep: bool) -> None:
        """Update dex-sweep bookkeeping after a successful scoped/full HIP-3 query."""
        if full_sweep:
            if not hasattr(self, "_hl_last_full_dex_sweep_ms_by_surface"):
                self._hl_last_full_dex_sweep_ms_by_surface = {}
            if not hasattr(self, "_hl_force_full_dex_sweep_surfaces"):
                self._hl_force_full_dex_sweep_surfaces = set()
            self._hl_last_full_dex_sweep_ms_by_surface[surface] = utc_ms()
            self._hl_force_full_dex_sweep = False
            self._hl_force_full_dex_sweep_surfaces.discard(surface)

    def _hl_note_ws_symbols_for_dex_scope(self, upd_list: list[dict]) -> None:
        """Force a full HIP-3 sweep if WS mentions a dex outside the active tracked scope."""
        if not hasattr(self, "_hl_force_full_dex_sweep_surfaces"):
            self._hl_force_full_dex_sweep_surfaces = set()
        active_dexes = set(self._get_hl_active_dex_names())
        unknown = set()
        for order in upd_list or []:
            if not isinstance(order, dict):
                continue
            symbol = str(order.get("symbol") or "")
            dex_name = self._get_hl_dex_for_symbol(symbol)
            if dex_name and dex_name not in active_dexes:
                unknown.add(dex_name)
        if unknown:
            self._hl_force_full_dex_sweep_surfaces.update({"open_orders", "positions"})
            logging.info(
                "[ws] unknown hip3 dex activity detected | dexes=%s | forcing full hip3 sweep",
                ",".join(sorted(unknown)),
            )

    def _normalize_ccxt_position(self, position: dict) -> dict:
        side = position.get("side")
        contracts = float(position.get("contracts") or 0.0)
        if side == "short":
            contracts = -contracts
        margin_mode = position.get("marginMode")
        if margin_mode is None and isinstance(position.get("info"), dict):
            leverage = position["info"].get("position", {}).get("leverage", {})
            if isinstance(leverage, dict):
                margin_mode = leverage.get("type")
        if margin_mode is None and position.get("isolated") is not None:
            margin_mode = "isolated" if position.get("isolated") else "cross"
        info_position = {}
        if isinstance(position.get("info"), dict):
            info_position = position["info"].get("position", {}) or {}
        return {
            "symbol": position["symbol"],
            "position_side": side,
            "size": contracts,
            "price": float(position.get("entryPrice") or 0.0),
            "margin_mode": str(margin_mode).lower() if margin_mode else None,
            "margin_used": float(
                position.get("initialMargin")
                or position.get("margin")
                or info_position.get("marginUsed")
                or 0.0
            ),
        }

    async def _fetch_hip3_positions(self, *, include_raw: bool = False):
        """Fetch HIP-3 positions via dex-scoped CCXT routes."""
        positions_by_key = {}
        raw_payloads = []
        dex_names, full_sweep = self._hl_select_dex_names_for_state("positions")
        fetch_specs = [{"params": {"dex": dex_name}} for dex_name in dex_names]
        started = utc_ms()
        coros = [self.cca.fetch_positions(**fetch_spec) for fetch_spec in fetch_specs]
        fetched_batches = await self._hl_gather_limited(coros)
        wall_ms = int(max(0, utc_ms() - started))
        if fetch_specs:
            self._log_hl_fetch_breakdown(
                "hip3_positions",
                wall_ms=wall_ms,
                timings_ms={},
                extra_parts=[
                    f"dex_queries={len(fetch_specs)}",
                    f"scope={'full' if full_sweep else 'active'}",
                    f"concurrency={self._hl_state_fetch_concurrency()}",
                ],
            )
        self._hl_mark_dex_scope_consumed("positions", full_sweep=full_sweep)
        for fetch_spec, fetched in zip(fetch_specs, fetched_batches):
            if include_raw:
                raw_payloads.append({"fetch_spec": deepcopy(fetch_spec), "response": deepcopy(fetched)})
            for position in fetched:
                normalized = self._normalize_ccxt_position(position)
                if not self._get_hl_dex_for_symbol(normalized["symbol"]):
                    continue
                self._record_hl_live_margin_mode(
                    normalized["symbol"], normalized.get("margin_mode")
                )
                key = (normalized["symbol"], normalized["position_side"])
                positions_by_key[key] = normalized
        normalized_positions = list(positions_by_key.values())
        if include_raw:
            return raw_payloads, normalized_positions
        return normalized_positions

    def _filter_approved_symbols(self, pside: str, symbols: set[str]) -> set[str]:
        del pside
        return symbols

    def _hl_supports_hip3_live_trading(self) -> bool:
        return bool(getattr(self, "_hl_unified_enabled", False))

    def _assert_supported_live_state(self) -> None:
        if self.HIP3_ISOLATED_SUPPORTED or self._hl_supports_hip3_live_trading():
            return
        unsupported = []
        approved = set()
        for syms in getattr(self, "approved_coins_minus_ignored_coins", {}).values():
            approved.update(syms)
        approved_hip3 = sorted(symbol for symbol in approved if self._get_hl_dex_for_symbol(symbol))
        if approved_hip3:
            unsupported.append(
                "approved_coins="
                + ",".join(sorted({symbol.split("/")[0] if "/" in symbol else symbol for symbol in approved_hip3}))
            )
        for symbol in sorted(
            set(getattr(self, "positions", {})) | set(getattr(self, "open_orders", {}))
        ):
            if not self._get_hl_dex_for_symbol(symbol):
                continue
            has_pos = False
            pos = getattr(self, "positions", {}).get(symbol, {})
            for pside in ("long", "short"):
                if abs(float(pos.get(pside, {}).get("size", 0.0) or 0.0)) > 0.0:
                    has_pos = True
                    break
            has_orders = bool(getattr(self, "open_orders", {}).get(symbol))
            if not (has_pos or has_orders):
                continue
            isolated_live_mode = getattr(self, "_hl_live_margin_modes", {}).get(symbol) == "isolated"
            isolated_only = self._requires_isolated_margin(symbol)
            reasons = []
            if isolated_only:
                reasons.append("isolated-only market")
            if isolated_live_mode:
                reasons.append("live isolated margin state")
            if not reasons:
                reasons.append("hip3 live state")
            state_bits = []
            if has_pos:
                state_bits.append("position")
            if has_orders:
                state_bits.append("open_orders")
            unsupported.append(f"{symbol} ({'/'.join(state_bits)}; {', '.join(reasons)})")
        if unsupported:
            raise FatalBotException(
                "Hyperliquid HIP-3/non-standard perps require unifiedAccount mode in Passivbot. "
                f"Current abstraction={getattr(self, '_hl_user_abstraction', 'unknown')}. "
                f"Unsupported HIP-3 state detected: {'; '.join(unsupported)}. "
                "Upgrade the Hyperliquid account to unifiedAccount or remove all HIP-3 "
                "symbols, positions, and open orders before running the bot."
            )

    async def watch_orders(self):
        res = None
        _ws_consecutive_rate_limits = 0
        while True:
            try:
                if self.stop_websocket:
                    break
                res = await self.ccp.watch_orders()
                _ws_consecutive_rate_limits = 0  # reset on success
                for i in range(len(res)):
                    res[i]["position_side"] = self.determine_pos_side(res[i])
                    res[i]["qty"] = res[i]["amount"]
                self._hl_note_ws_symbols_for_dex_scope(res)
                self.handle_order_update(res)
            except asyncio.CancelledError:
                break
            except RateLimitExceeded:
                self._health_ws_reconnects += 1
                self._health_rate_limits += 1
                _ws_consecutive_rate_limits += 1
                backoff = min(30, 2 ** _ws_consecutive_rate_limits) + random.uniform(0, 1)
                logging.warning(
                    "[ws] %s: rate limited (reconnect #%d), backing off %.0fs...",
                    self.exchange,
                    self._health_ws_reconnects,
                    backoff,
                )
                await asyncio.sleep(backoff)
                logging.info("[ws] %s: reconnecting after rate limit...", self.exchange)
            except Exception as e:
                self._health_ws_reconnects += 1
                _ws_consecutive_rate_limits = 0
                logging.warning(
                    "[ws] %s: connection lost (reconnect #%d), retrying in 1s: %s",
                    self.exchange,
                    self._health_ws_reconnects,
                    type(e).__name__,
                )
                logging.debug("[ws] %s: full exception: %s", self.exchange, e)
                logging.debug("".join(traceback.format_exc()))
                await asyncio.sleep(1)
                logging.info("[ws] %s: reconnecting...", self.exchange)

    def determine_pos_side(self, order):
        # hyperliquid is not hedge mode
        if order["symbol"] in self.positions:
            if self.positions[order["symbol"]]["long"]["size"] != 0.0:
                return "long"
            elif self.positions[order["symbol"]]["short"]["size"] != 0.0:
                return "short"
            else:
                return "long" if order["side"] == "buy" else "short"
        else:
            if "reduceOnly" in order:
                if order["side"] == "buy":
                    return "short" if order["reduceOnly"] else "long"
                if order["side"] == "sell":
                    return "long" if order["reduceOnly"] else "short"
            return "long" if order["side"] == "buy" else "short"

    def _get_position_side_for_order(self, order: dict) -> str:
        """Hook: Derive position_side from order data for Hyperliquid (one-way mode)."""
        return self.determine_pos_side(order)

    async def _do_fetch_open_orders(self, symbol: str = None):
        fetched = []
        seen_ids = set()
        query_symbols = [symbol] if symbol is not None else []
        if symbol is None:
            query_dexes, full_sweep = self._hl_select_dex_names_for_state("open_orders")
        else:
            query_dexes, full_sweep = ([], False)
        fetch_specs = []

        # Default route covers core perps; HIP-3 symbols need dex-scoped queries.
        if symbol is None or not self._get_hl_dex_for_symbol(symbol):
            fetch_specs.append(("core", {"symbol": symbol}))

        if symbol is not None and self._get_hl_dex_for_symbol(symbol):
            hip3_symbols = query_symbols
        else:
            hip3_symbols = []

        for hip3_symbol in hip3_symbols:
            fetch_specs.append((f"symbol:{hip3_symbol}", {"symbol": hip3_symbol}))

        for dex_name in query_dexes:
            fetch_specs.append((f"dex:{dex_name}", {"params": {"dex": dex_name}}))

        started = utc_ms()
        fetched_batches = await self._hl_gather_limited(
            [self.cca.fetch_open_orders(**kwargs) for _label, kwargs in fetch_specs]
        )
        wall_ms = int(max(0, utc_ms() - started))
        if fetch_specs:
            self._log_hl_fetch_breakdown(
                "open_orders",
                wall_ms=wall_ms,
                timings_ms={},
                extra_parts=[
                    f"queries={len(fetch_specs)}",
                    f"scope={'full' if full_sweep else 'active'}",
                    f"concurrency={self._hl_state_fetch_concurrency()}",
                ],
            )
        if symbol is None:
            self._hl_mark_dex_scope_consumed("open_orders", full_sweep=full_sweep)

        for (_label, _kwargs), batch in zip(fetch_specs, fetched_batches):
            for order in batch:
                if order["id"] in seen_ids:
                    continue
                seen_ids.add(order["id"])
                fetched.append(order)
        return fetched

    def _normalize_open_orders(self, fetched: list) -> list:
        for elm in fetched:
            elm["position_side"] = self.determine_pos_side(elm)
            elm["qty"] = elm["amount"]
        return sorted(fetched, key=lambda x: x["timestamp"])

    async def fetch_open_orders(self, symbol: str = None):
        fetched = await self._do_fetch_open_orders(symbol=symbol)
        return self._normalize_open_orders(fetched)

    def _hl_balance_payload_is_unified(self, balance_payload: dict) -> bool:
        info = balance_payload.get("info", {}) if isinstance(balance_payload, dict) else {}
        return isinstance(info, dict) and isinstance(info.get("balances"), list)

    def _hl_extract_unified_total(self, balance_payload: dict) -> float:
        total = balance_payload.get("total", {}) if isinstance(balance_payload, dict) else {}
        if isinstance(total, dict) and total.get(self.quote) is not None:
            return float(total[self.quote])
        info = balance_payload.get("info", {}) if isinstance(balance_payload, dict) else {}
        balances = info.get("balances", []) if isinstance(info, dict) else []
        for row in balances or []:
            if str(row.get("coin") or "") == self.quote:
                return float(row.get("total") or 0.0)
        raise KeyError(f"unified Hyperliquid balance payload missing total for {self.quote}")

    async def _fetch_positions_and_balance(self):
        timings_ms = {}
        started = utc_ms()
        balance_task = asyncio.create_task(
            self._timed_authoritative_fetch("balance", self.cca.fetch_balance(), timings_ms)
        )
        hip3_positions_task = asyncio.create_task(
            self._timed_authoritative_fetch(
                "hip3_positions", self._fetch_hip3_positions(include_raw=True), timings_ms
            )
        )
        speculative_core_positions_task = None
        if bool(getattr(self, "_hl_unified_enabled", False)):
            speculative_core_positions_task = asyncio.create_task(
                self._timed_authoritative_fetch(
                    "core_positions", self.cca.fetch_positions(), timings_ms
                )
            )
        try:
            info = await balance_task
            positions = {}
            raw_core_positions = None
            if self._hl_balance_payload_is_unified(info):
                self._hl_balance_payload_mode = "unified_total"
                if speculative_core_positions_task is not None:
                    raw_core_positions = await speculative_core_positions_task
                else:
                    raw_core_positions = await self._timed_authoritative_fetch(
                        "core_positions", self.cca.fetch_positions(), timings_ms
                    )
                for position in raw_core_positions:
                    normalized = self._normalize_ccxt_position(position)
                    if self._get_hl_dex_for_symbol(normalized["symbol"]):
                        continue
                    self._record_hl_live_margin_mode(
                        normalized["symbol"], normalized.get("margin_mode")
                    )
                    positions[(normalized["symbol"], normalized["position_side"])] = normalized
                balance = self._hl_extract_unified_total(info)
            else:
                self._hl_balance_payload_mode = "perp_account_value"
                raw_core_positions = deepcopy(info["info"].get("assetPositions", []))
                for x in raw_core_positions:
                    symbol = self.coin_to_symbol(x["position"]["coin"])
                    leverage = x["position"].get("leverage", {})
                    if isinstance(leverage, dict):
                        self._record_hl_live_margin_mode(symbol, leverage.get("type"))
                    size = float(x["position"]["szi"])
                    elm = {
                        "symbol": symbol,
                        "position_side": ("long" if size > 0.0 else "short"),
                        "size": size,
                        "price": float(x["position"]["entryPx"]),
                        "margin_mode": (
                            str(leverage.get("type")).lower()
                            if isinstance(leverage, dict) and leverage.get("type")
                            else None
                        ),
                        "margin_used": float(x["position"].get("marginUsed") or 0.0),
                    }
                    positions[(elm["symbol"], elm["position_side"])] = elm
                balance = float(info["info"]["marginSummary"]["accountValue"]) - sum(
                    [float(x["position"]["unrealizedPnl"]) for x in raw_core_positions]
                )
                if speculative_core_positions_task is not None:
                    if not speculative_core_positions_task.done():
                        speculative_core_positions_task.cancel()
                    await asyncio.gather(speculative_core_positions_task, return_exceptions=True)
            hip3_raw, hip3_positions = await hip3_positions_task
            for position in hip3_positions:
                positions[(position["symbol"], position["position_side"])] = position
            raw_snapshot = {
                "balance": deepcopy(info),
                "positions": {
                    "core": deepcopy(raw_core_positions),
                    "hip3": hip3_raw,
                },
                "balance_mode": str(getattr(self, "_hl_balance_payload_mode", "")),
            }
            wall_ms = int(max(0, utc_ms() - started))
            self._log_hl_fetch_breakdown(
                "positions_balance",
                wall_ms=wall_ms,
                timings_ms=timings_ms,
                extra_parts=[
                    f"unified={'yes' if self._hl_balance_payload_mode == 'unified_total' else 'no'}",
                    f"hip3_dexes={len(self._get_hl_hip3_dex_names())}",
                ],
            )
            return raw_snapshot, list(positions.values()), balance
        except Exception:
            for task in (speculative_core_positions_task, hip3_positions_task, balance_task):
                if task is not None and not task.done():
                    task.cancel()
            await asyncio.gather(
                *[task for task in (speculative_core_positions_task, hip3_positions_task, balance_task) if task is not None],
                return_exceptions=True,
            )
            raise

    async def _get_positions_and_balance_cached(self, my_gen: int = 0):
        """Fetch positions+balance with dedup: concurrent callers share one API call.

        my_gen is the caller's snapshot of _hl_cache_generation taken *before*
        acquiring the lock.  If another caller completed a fetch in the
        meantime (cache_generation advanced), we return the cached result
        (or re-raise the cached exception if the fetch failed).
        """
        async with self._hl_fetch_lock:
            cached_gen = self._hl_cache_generation
            if cached_gen > my_gen and hasattr(self, "_hl_cached_result"):
                if isinstance(self._hl_cached_result, Exception):
                    raise self._hl_cached_result
                return self._hl_cached_result
            try:
                result = await self._fetch_positions_and_balance()
            except Exception as e:
                self._hl_cached_result = e
                self._hl_cache_generation = cached_gen + 1
                raise
            self._hl_cached_result = result
            self._hl_cache_generation = cached_gen + 1
            return result

    async def fetch_positions(self):
        # Snapshot generation *before* lock so each caller tracks its own view.
        my_gen = self._hl_cache_generation
        _, positions, balance = await self._get_positions_and_balance_cached(my_gen)
        self._last_hl_balance = balance
        self._hl_balance_consumed = False
        return positions

    async def capture_positions_snapshot(self) -> tuple[list, list]:
        my_gen = self._hl_cache_generation
        raw_snapshot, positions, balance = await self._get_positions_and_balance_cached(my_gen)
        self._last_hl_balance = balance
        self._hl_balance_consumed = False
        return deepcopy(raw_snapshot["positions"]), deepcopy(positions)

    async def fetch_balance(self):
        # Check if fetch_positions already got us a fresh balance
        if getattr(self, "_last_hl_balance", None) is not None and not getattr(
            self, "_hl_balance_consumed", True
        ):
            self._hl_balance_consumed = True
            return self._last_hl_balance
        # Snapshot generation *before* lock so each caller tracks its own view.
        my_gen = self._hl_cache_generation
        _, positions, balance = await self._get_positions_and_balance_cached(my_gen)
        return balance

    async def capture_balance_snapshot(self) -> tuple[dict, float]:
        my_gen = self._hl_cache_generation
        raw_snapshot, positions, balance = await self._get_positions_and_balance_cached(my_gen)
        return deepcopy(raw_snapshot["balance"]), float(balance)

    async def _capture_positions_balance_staged_snapshot(self) -> tuple[dict, list, float]:
        """Fetch Hyperliquid positions+balance once for staged authoritative refresh."""
        my_gen = self._hl_cache_generation
        raw_snapshot, positions, balance = await self._get_positions_and_balance_cached(my_gen)
        self._last_hl_balance = balance
        self._hl_balance_consumed = False
        return deepcopy(raw_snapshot), deepcopy(positions), float(balance)

    async def capture_authoritative_state_staged_snapshot(
        self, plan: set[str], timings_ms: dict[str, int]
    ) -> dict | None:
        """Fetch Hyperliquid authoritative staged surfaces using coherent account cohorts."""
        out = {"plan": set(plan), "pnls_ok": True}
        tasks = {}
        if "balance" in plan or "positions" in plan:
            tasks["positions_balance"] = asyncio.create_task(
                self._timed_authoritative_fetch(
                    "positions_balance",
                    self._capture_positions_balance_staged_snapshot(),
                    timings_ms,
                )
            )
        if "open_orders" in plan:
            tasks["open_orders"] = asyncio.create_task(
                self._timed_authoritative_fetch("open_orders", self.fetch_open_orders(), timings_ms)
            )
        if "fills" in plan:
            tasks["fills"] = asyncio.create_task(
                self._timed_authoritative_fetch("fills", self.update_pnls(), timings_ms)
            )
        try:
            keys = list(tasks)
            results = await asyncio.gather(*[tasks[key] for key in keys])
        except Exception:
            for task in tasks.values():
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks.values(), return_exceptions=True)
            raise
        for key, result in zip(keys, results):
            if key == "positions_balance":
                _raw_snapshot, positions, balance = result
                if "positions" in plan:
                    out["positions"] = positions
                if "balance" in plan:
                    out["balance"] = balance
            elif key == "open_orders":
                out["open_orders"] = result
            elif key == "fills":
                out["pnls_ok"] = result
        return out

    async def fetch_tickers(self):
        fetched = await self.cca.fetch(
            self._hl_info_url(),
            method="POST",
            headers={"Content-Type": "application/json"},
            body=json.dumps({"type": "allMids"}),
        )
        tickers = {}
        for coin, price_raw in fetched.items():
            symbol = self.symbol_ids_inv.get(coin)
            if symbol is None:
                symbol = self.coin_to_symbol(coin, verbose=False)
            if symbol not in self.markets_dict:
                continue
            price = float(price_raw)
            tickers[symbol] = {"bid": price, "ask": price, "last": price}
        return tickers

    async def fetch_tickers_for_symbols(self, symbols: list[str]) -> dict:
        """Fetch current tickers for specific symbols, including HIP-3 markets.

        Hyperliquid's cheap allMids endpoint does not reliably expose builder
        deployed HIP-3 markets with CCXT unified symbols. For those, use the
        per-dex metaAndAssetCtxs endpoint and derive bid/ask/last from the
        current asset context.
        """
        requested = [s for s in dict.fromkeys(symbols or []) if s in self.markets_dict]
        if not requested:
            return {}
        out = {}
        vanilla_symbols = []
        hip3_by_dex = {}
        for symbol in requested:
            info = self.markets_dict.get(symbol, {}).get("info", {})
            if info.get("hip3"):
                dex = str(info.get("dex") or "").strip()
                if not dex:
                    raise ValueError(f"Hyperliquid HIP-3 symbol {symbol} missing dex metadata")
                hip3_by_dex.setdefault(dex, []).append(symbol)
            else:
                vanilla_symbols.append(symbol)
        if vanilla_symbols:
            bulk = await self.fetch_tickers()
            out.update({symbol: bulk[symbol] for symbol in vanilla_symbols if symbol in bulk})
        for dex, dex_symbols in hip3_by_dex.items():
            out.update(await self._fetch_hip3_tickers_for_symbols(dex, dex_symbols))
        return out

    async def _fetch_hip3_tickers_for_symbols(self, dex: str, symbols: list[str]) -> dict:
        response = await self.cca.fetch(
            self._hl_info_url(),
            method="POST",
            headers={"Content-Type": "application/json"},
            body=json.dumps({"type": "metaAndAssetCtxs", "dex": dex}),
        )
        if not isinstance(response, list) or len(response) < 2:
            raise ValueError(f"unexpected Hyperliquid HIP-3 meta response for dex={dex}")
        universe = response[0].get("universe", []) if isinstance(response[0], dict) else []
        asset_ctxs = response[1] if isinstance(response[1], list) else []
        name_to_symbol = {}
        for symbol in symbols:
            market = self.markets_dict.get(symbol, {})
            info = market.get("info", {})
            for name in (info.get("name"), market.get("baseName")):
                if name:
                    name_to_symbol[str(name)] = symbol
        out = {}
        for idx, asset in enumerate(universe):
            if not isinstance(asset, dict):
                continue
            symbol = name_to_symbol.get(str(asset.get("name") or ""))
            if symbol is None:
                continue
            ctx = asset_ctxs[idx] if idx < len(asset_ctxs) and isinstance(asset_ctxs[idx], dict) else {}
            ticker = self._hip3_ticker_from_asset_ctx(symbol, ctx)
            if ticker is not None:
                out[symbol] = ticker
        return out

    @staticmethod
    def _hip3_ticker_from_asset_ctx(symbol: str, ctx: dict) -> dict | None:
        def _positive(value):
            try:
                out = float(value)
            except (TypeError, ValueError):
                return None
            return out if out > 0.0 else None

        impact = ctx.get("impactPxs") if isinstance(ctx, dict) else None
        bid = ask = None
        if isinstance(impact, list) and len(impact) >= 2:
            bid = _positive(impact[0])
            ask = _positive(impact[1])
        last = _positive(ctx.get("midPx")) or _positive(ctx.get("markPx")) or _positive(
            ctx.get("oraclePx")
        )
        if bid is None:
            bid = last
        if ask is None:
            ask = last
        if last is None or bid is None or ask is None:
            logging.debug(
                "[market] hyperliquid HIP-3 ticker missing usable price | symbol=%s", symbol
            )
            return None
        return {"bid": bid, "ask": ask, "last": last}

    async def fetch_ohlcv(self, symbol: str, timeframe="1m"):
        # intervals: 1,3,5,15,30,60,120,240,360,720,D,M,W
        # fetches latest ohlcvs
        str2int = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 60 * 4}
        n_candles = 480
        since = int(utc_ms() - 1000 * 60 * str2int[timeframe] * n_candles)
        return await self.cca.fetch_ohlcv(symbol, timeframe=timeframe, since=since)

    async def fetch_ohlcvs_1m(self, symbol: str, since: float = None, limit=None):
        n_candles_limit = 5000 if limit is None else limit
        result = await self.cca.fetch_ohlcv(
            symbol,
            timeframe="1m",
            limit=n_candles_limit,
            since=int(self.get_exchange_time() - 1000 * 60 * n_candles_limit * 0.95),
        )
        return result

    async def fetch_pnls(
        self,
        start_time: int = None,
        end_time: int = None,
        limit=None,
    ):
        # hyperliquid fetches from past to future
        if limit is None:
            limit = 2000
        if start_time is None:
            # hyperliquid returns latest trades if no time frame is passed
            return await self.fetch_pnl(limit=limit)
        all_fetched = {}
        prev_hash = ""
        while True:
            fetched = await self.fetch_pnl(start_time=start_time, limit=limit)
            if fetched == []:
                break
            for elm in fetched:
                all_fetched[elm["id"]] = elm
            if len(fetched) < limit:
                break
            if end_time and fetched[-1]["timestamp"] >= end_time:
                break
            new_hash = calc_hash(fetched)
            if prev_hash == new_hash:
                logging.debug(f"pnls hash unchanged: {prev_hash}")
                break
            prev_hash = new_hash
            logging.info(
                f"debug fetching pnls {ts_to_date(fetched[-1]['timestamp'])} len {len(fetched)}"
            )
            start_time = fetched[-1]["timestamp"] - 1000
            limit = 2000
        return sorted(all_fetched.values(), key=lambda x: x["timestamp"])

    async def gather_fill_events(self, start_time=None, end_time=None, limit=None):
        """Return canonical fill events for Hyperliquid (draft placeholder)."""
        events = []
        fills = await self.fetch_pnls(start_time=start_time, end_time=end_time, limit=limit)
        for fill in fills:
            events.append(
                {
                    "id": fill.get("id"),
                    "timestamp": fill.get("timestamp"),
                    "symbol": fill.get("symbol"),
                    "side": fill.get("side"),
                    "position_side": fill.get("position_side"),
                    "qty": fill.get("amount"),
                    "price": fill.get("price"),
                    "pnl": fill.get("pnl"),
                    "fee": fill.get("fee"),
                    "info": fill.get("info"),
                }
            )
        return events

    async def fetch_pnl(
        self,
        start_time: int = None,
        limit=None,
    ):
        if start_time is None:
            fetched = await self.cca.fetch_my_trades(limit=limit)
        else:
            fetched = await self.cca.fetch_my_trades(since=max(1, int(start_time)), limit=limit)
        for elm in fetched:
            elm["pnl"] = float(elm["info"]["closedPnl"])
            elm["position_side"] = "long" if "long" in elm["info"]["dir"].lower() else "short"
        return sorted(fetched, key=lambda x: x["timestamp"])

    async def execute_cancellation(self, order: dict) -> dict:
        """Hyperliquid: Cancel order with vault support."""
        params = (
            {"vaultAddress": self.user_info["wallet_address"]} if self.user_info["is_vault"] else {}
        )

        def _is_already_gone(payload) -> bool:
            try:
                text = str(payload)
            except Exception:
                text = ""
            text_l = text.lower()
            if (
                "order was never placed" in text_l
                or "already canceled" in text_l
                or "already cancelled" in text_l
            ):
                return True
            return False

        try:
            res = await self.cca.cancel_order(order["id"], symbol=order["symbol"], params=params)
            # Sometimes hyperliquid returns an "ok" wrapper with an embedded error; treat as non-fatal.
            if _is_already_gone(res):
                logging.info("Order already canceled/filled on exchange; treating as success.")
                return {
                    "status": "success",
                    "_passivbot_cancel_requires_full_authoritative_confirmation": True,
                }
            return res
        except Exception as e:
            if _is_already_gone(e):
                logging.info("Order already canceled/filled on exchange; treating as success.")
                return {
                    "status": "success",
                    "_passivbot_cancel_requires_full_authoritative_confirmation": True,
                }
            raise

    def did_cancel_order(self, executed, order=None) -> bool:
        if isinstance(executed, list) and len(executed) == 1:
            return self.did_cancel_order(executed[0], order)
        try:
            return "status" in executed and executed["status"] == "success"
        except (TypeError, KeyError):
            return False

    def _build_order_params(self, order: dict) -> dict:
        params = {
            "reduceOnly": order["reduce_only"],
            "timeInForce": (
                "Alo" if require_live_value(self.config, "time_in_force") == "post_only" else "Gtc"
            ),
            "clientOrderId": order["custom_id"],
        }
        if self.user_info["is_vault"]:
            params["vaultAddress"] = self.user_info["wallet_address"]
        return params

    async def execute_order(self, order: dict) -> dict:
        """Hyperliquid: Execute order with min_cost auto-adjustment on specific errors."""
        try:
            return await super().execute_order(order)
        except Exception as e:
            # Try to recover from Hyperliquid's "$10 minimum" errors by adjusting min_cost
            try:
                if self.adjust_min_cost_on_error(e, order):
                    logging.info(f"Adjusted min_cost for order, will retry: {order['symbol']}")
                    return {}
            except Exception as e0:
                logging.error(f"error with adjust_min_cost_on_error {e0}")
            # Could not recover - re-raise to trigger restart_bot_on_too_many_errors
            raise

    async def execute_orders(self, orders: [dict]) -> [dict]:
        return await self.execute_multiple(orders, "execute_order")

    def did_create_order(self, executed) -> bool:
        did_create = super().did_create_order(executed)
        try:
            return did_create and (
                "info" in executed and ("filled" in executed["info"] or "resting" in executed["info"])
            )
        except (TypeError, KeyError):
            return False

    def adjust_min_cost_on_error(self, error, order=None):
        any_adjusted = False
        successful_orders = []
        str_e = str(error)
        brace_idx = str_e.find("{")
        if brace_idx == -1:
            return False
        try:
            error_json = json.loads(str_e[brace_idx:])
        except json.JSONDecodeError:
            return False
        if (
            "response" in error_json
            and "data" in error_json["response"]
            and "statuses" in error_json["response"]["data"]
        ):
            for elm in error_json["response"]["data"]["statuses"]:
                if "error" in elm:
                    if "Order must have minimum value of $10" in elm["error"]:
                        asset_id = int(elm["error"][elm["error"].find("asset=") + 6 :])
                        for symbol in self.markets_dict:
                            if (
                                "baseId" in self.markets_dict[symbol]["info"]
                                and self.markets_dict[symbol]["info"]["baseId"] == asset_id
                            ):
                                break
                        else:
                            raise Exception(f"No symbol match for asset_id={asset_id}")
                        new_min_cost = pbr.round_(self.min_costs[symbol] * 1.1, 0.1)
                        logging.info(
                            f"caught {elm['error']} {symbol}. Upping min_cost from {self.min_costs[symbol]} to {new_min_cost}. Order: {order}"
                        )
                        self.min_costs[symbol] = new_min_cost
                        any_adjusted = True
        return any_adjusted

    def symbol_is_eligible(self, symbol):
        """Check if a symbol is eligible for trading.

        HIP-3 stock perps remain discoverable, but isolated-only live trading is
        currently disabled elsewhere via symbol filtering/startup validation.
        """
        try:
            market_info = self.markets_dict[symbol]["info"]

            # Zero open interest means market is inactive
            if float(market_info.get("openInterest", 0)) == 0.0:
                return False
        except Exception as e:
            logging.error(f"error with symbol_is_eligible {e} {symbol}")
            return False
        return True

    async def update_exchange_config_by_symbols(self, symbols):
        """Set leverage and margin mode for Hyperliquid symbols.

        Uses base class methods for isolated margin detection and leverage calculation.
        Adds Hyperliquid-specific vault address handling.
        Calls are made sequentially with a small delay to avoid rate-limit bursts.
        """
        for symbol in symbols:
            to_print = ""
            try:
                leverage = self._calc_leverage_for_symbol(symbol)
                margin_mode = self._get_margin_mode_for_symbol(symbol)

                params = {"leverage": leverage}
                if self.user_info["is_vault"]:
                    params["vaultAddress"] = self.user_info["wallet_address"]

                try:
                    res = await self.cca.set_margin_mode(
                        margin_mode, symbol=symbol, params=params
                    )
                    to_print = (
                        f"margin={format_exchange_config_response(res)} ({margin_mode})"
                    )
                except Exception as e:
                    if '"code":"59107"' in str(e):
                        to_print = f"margin=ok (unchanged, {margin_mode})"
                    else:
                        logging.error(f"{symbol} error setting {margin_mode} mode {e}")
            except Exception as e:
                logging.error(f"{symbol}: error setting margin mode and leverage {e}")
            if to_print:
                logging.debug(f"{symbol}: {to_print}")
            # Small delay between margin-mode API calls to avoid rate-limit bursts
            await asyncio.sleep(0.2)

    async def update_exchange_config(self):
        pass

    async def calc_ideal_orders(self):
        # hyperliquid needs custom price rounding
        ideal_orders = await super().calc_ideal_orders()
        for sym in ideal_orders:
            for i in range(len(ideal_orders[sym])):
                if ideal_orders[sym][i]["side"] == "sell":
                    ideal_orders[sym][i]["price"] = round_dynamic_up(
                        round(ideal_orders[sym][i]["price"], self.n_decimal_places),
                        self.n_significant_figures,
                    )
                elif ideal_orders[sym][i]["side"] == "buy":
                    ideal_orders[sym][i]["price"] = round_dynamic_dn(
                        round(ideal_orders[sym][i]["price"], self.n_decimal_places),
                        self.n_significant_figures,
                    )
                else:
                    ideal_orders[sym][i]["price"] = round_dynamic(
                        round(ideal_orders[sym][i]["price"], self.n_decimal_places),
                        self.n_significant_figures,
                    )
                ideal_orders[sym][i]["price"] = round_(
                    ideal_orders[sym][i]["price"], self.price_steps[sym]
                )
        return ideal_orders

    def format_custom_id_single(self, order_type_id: int) -> str:
        formatted = super().format_custom_id_single(order_type_id)
        return (formatted)[: self.custom_id_max_length]
