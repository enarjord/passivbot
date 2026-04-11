# Exchange API Quirks

Only task-specific, high-impact quirks are listed here.

## Bybit

### Closed-PnL pagination mismatch

Problem:

1. Cursor pagination has limited historical reach.
2. Time-based pagination can skip records when windows exceed page limits.

Handling in Passivbot:

1. Use hybrid pagination (cursor for recent, time-window for older).
2. Deduplicate by `orderId`.

Primary reference: `src/fill_events_manager.py` (`BybitFetcher._fetch_positions_history`).

## KuCoin Futures

### OHLCV limit behavior + sparse-minute markets

Problem:

1. Effective page size is 200 rows.
2. Illiquid symbols legitimately have missing trade minutes.

Handling:

1. Page with `limit=200`.
2. Overlap page boundaries by 1 candle to validate inter-page gaps.

## Bitget Futures

### `since` is effectively exclusive for OHLCV paging

Problem: naive paging can miss first candle in each page.

Handling:

1. Overlap boundaries by 1 candle.
2. Back up initial `since` by one candle on pagination start.

## Lighter

### Native signer binary with pinned ABI

Problem:

1. CCXT's Lighter client signs transactions through a Go-compiled shared library
   loaded via `ctypes`. CCXT ships the Python `ctypes` wrapper
   (`ccxt.static_dependencies.lighter_client.signer`) but **not** the binary.
2. The upstream binary has had breaking ABI changes (e.g. `CreateOrderTxReq`
   field layout) since CCXT's wrapper was generated. Newer releases segfault or
   produce invalid signatures.

Handling in Passivbot:

1. Users must download `elliottech/lighter-python@b7fc10b2` / v1.0.4 binaries
   and point `api-keys.json:library_path` at the result.
2. `LighterBot.create_ccxt_sessions` calls `load_lighter_library(library_path)`
   and stores the signer on both `cca.options["signer"]` and `self._signer`.
3. Do not bump the pinned commit in user-facing docs without verifying the
   ctypes wrapper in the bundled `ccxt` version still matches.

Primary reference: `src/exchanges/lighter.py` (`LighterBot.create_ccxt_sessions`).

### Leverage/margin updates bypass CCXT

Problem: CCXT's Lighter adapter routes `set_leverage` / `set_margin_mode`
through a `load_account`-based path that is broken against the current Lighter
endpoint.

Handling in Passivbot:

1. `LighterBot.update_exchange_config_by_symbols` calls
   `self._signer.SignUpdateLeverage` directly, decodes the tx with
   `decode_tx_info`, and submits via `self.cca.publicPostSendTx`.
2. "already"/"no change" errors are swallowed as no-ops. Any other exception
   propagates so the orchestrator's per-symbol retry/backoff loop can react.

Primary reference: `src/exchanges/lighter.py` (`LighterBot.update_exchange_config_by_symbols`).

### No all-symbols `fetch_open_orders`

Problem: Lighter's REST API requires a per-symbol query for open orders.

Handling: `LighterBot.fetch_open_orders(symbol=None)` fans out across tracked
symbols (union of `open_orders` keys, symbols with positions, and
`active_symbols`) and merges the results. Scales with active symbol count.

Primary reference: `src/exchanges/lighter.py` (`LighterBot.fetch_open_orders`).

### Native WebSocket (not CCXT `watch_orders`)

`LighterBot` overrides `can_watch_orders` to `True` and implements its own
connection to `wss://mainnet.zklighter.elliot.ai/stream`, subscribing to
`account_all/{account_index}`. Order updates arrive through `_do_watch_orders`
and are normalized by `_normalize_order_update`. If changing the WebSocket
layer, remember that Lighter's channel payloads use `client_order_id`,
`reduce_only`, and `status` string set mapped in `_normalize_status`.

Primary reference: `src/exchanges/lighter.py` (`LighterBot._ws_connect`, `_ws_receive`, `_normalize_order_update`).

## General Guidance

1. Check raw exchange payloads when CCXT abstraction is insufficient.
2. Treat intra-page gaps and inter-page gaps differently.
3. For missing data incidents, verify source data before changing logic.
