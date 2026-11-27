# Exchange Integration Contracts

Shared exchange contracts live here; venue-specific contracts live under
`exchanges/`. Authenticated verification
requires explicit user approval; prefer offline request-construction tests.

## Broker Attribution

Changes to broker-code or broker-agreement attribution require explicit approval. Registry
loading must fail visibly on missing registries, invalid data, or unknown exchanges. Every
relevant order request must carry the required field, header, or tag. Test changes at the CCXT
or raw signed-request boundary.

## Supported Live-Exchange Boundary

The supported production live connectors are Binance, Bybit, Bitget, Bitunix, OKX, Gate.io,
KuCoin, Hyperliquid, WEEX, and Lighter. The fake connector is an offline deterministic test harness, not an
exchange.

Defx and Paradex are retired. Runtime routing and offline connector inspection reject
these identifiers before generic CCXT fallback; no adapter implementation remains.

The generic `CCXTBot` fallback for arbitrary exchange names is also outside the supported
production boundary. It preserves compatibility for unaudited CCXT venues, but a feature requiring
authoritative order-type, close-only, remaining-quantity, or one-way position-side normalization
must use an explicit supported-connector allowlist. The generic fallback retains its legacy basic
reconciliation/tolerance path until that venue receives a connector-specific contract audit; the
separate global retirement of the old initial-entry-only distance gate does not enable the new
churn policy there.

### Temporary exchange-side symbol suspension

Some venues reject otherwise valid authenticated writes because API trading for one symbol is
temporarily unavailable. This is a connector-owned classification boundary: the shared runtime
must never infer a suspension from exception text or a portable-looking numeric code. A supported
connector may opt in only with an exact structured exchange code and a fixture-backed regression
test.

On a proven suspension, `live.recovery.exchange_symbol_unavailable_cooldown_hours` starts a per-symbol,
RAM-only cooldown (default `6.0` hours; `0` disables it; maximum `876600` hours). Values outside
that bounded conversion domain are rejected during configuration and cannot mask the original
exchange failure. A flat affected symbol is nontradable for
new planning through `graceful_stop`. An affected symbol with a position uses `tp_only`, overriding
`normal` or `graceful_stop`, so all further entries are suppressed while ordinary closes remain
available without successful leverage refresh. Explicit `panic` and `manual` modes remain stronger.
The initial failed entry/configuration write retains the normal execution failure/restart-budget
consequences. Retry-backoff cycles with no authenticated write do not count as additional failures;
the cooldown prevents repeated futile entry writes rather than hiding the fault.
Expiry restores on-demand attempts, another classified response starts a new cooldown, and process
restart deliberately clears the state and retries immediately.

Future connectors encountering an equivalent venue rule must add their own exact classifier. Do
not generalize one exchange's code, match human-readable messages, persist cooldowns across runs, or
make held positions wholly nontradable.

## HTTP Retry Delays

Live CCXT clients and the native Bitunix transport carry a valid HTTP `Retry-After`
delay on the transport exception from that response. Parse nonnegative integer
seconds or an HTTP date, relative to the response `Date` when available and the
local wall clock otherwise. Preserve the exchange's error classification and avoid
an inline retry of the request when the response requires a positive delay. The
owner schedules current-account transport recovery. Other retry policies retain
their own input readiness and scope. Never read shared last-response headers to
associate a retry delay with a request.

## CCXT Position Snapshot Readiness

The shared CCXT position normalizer, also used by OKX and Bybit, requires a
finite contract count on every position row. Explicit zero contracts are flat.
Held rows require a symbol, a finite positive entry price, and a canonical
long/short side. Bybit validates held rows before pagination deduplication and
rejects a provided native hedge index if it contradicts the normalized side.
Bybit follows the exchange cursor even if the current page contributes no held
positions. A malformed row or incomplete pagination makes the whole positions
surface unavailable.

## Private Order Websocket Normalization

Order rows proving fill progress (`filled > 0` or `remaining < amount`) request an authoritative
account refresh even when their status remains open and they match a recent local creation.
A creation acknowledgement may be a self-echo; fill progress invalidates the account-wide plan.

Authoritative REST open-order reconciliation remains strict. Binance and KuCoin
private websocket notifications for Passivbot-owned orders may omit native
long/short metadata; only that hint path may recover `position_side`, only in
effective hedge mode, and only when a valid Passivbot client-order marker has
no conflicting native position-side field and every supplied exchange/client
identity matches the same record in this process's emitted-order registry.
Acknowledged emitted identities remain registered for as long as the
corresponding order is present in the bot's authoritative open-order state,
even beyond the normal foreign-writer lookback. Recovered rows always force an
authoritative account refresh. Sparse foreign, explicitly one-way,
identity-conflicting, or unmarked notifications remain rejected.

Bitget UTA websocket rows use the exchange-native `holdSide` field for hedge
position attribution, while classic rows use `posSide`. Hyperliquid may deliver
an order-open websocket row before the concurrent create request returns its
exchange order ID. While a create is still freshly submitted, the connector may
briefly yield for that acknowledgement and retry the row, but recovery still
requires the exact acknowledged exchange ID and all existing contradiction
checks. Sparse Hyperliquid rows for orders already resting at process startup may
instead recover side, `position_side`, and close-only intent from one exact
exchange-ID match in the current authoritative REST open-order snapshot. A
bounded five-minute in-memory copy of those exact semantics covers terminal
updates which arrive just after reconciliation removes the order; it is rebuilt
from REST after restart and does not preserve ownership or trading intent.
Every supplied exchange-ID and client-ID alias must agree with the canonical
identity retained by the snapshot or its bounded copy. A contradictory current
snapshot row invalidates any older cached semantics for its exchange-ID aliases.
Missing, duplicate, expired, or contradictory matches remain rejected and
trigger a fresh account-state read;
authoritative snapshot contradictions must not fall back to process-local
acknowledgement evidence. A unified `reduceOnly` value of `false` or `null` is
treated as a CCXT placeholder only when the native row omits that field; an
explicit native value still must agree with the recovered semantics. All
snapshot-recovered rows request authoritative refresh because an exact exchange
ID proves semantics but not local ownership. Price, quantity, side, or order
shape alone never prove ownership.

A successful private-websocket read and a valid individual order row are separate health
boundaries. When a supported CCXT connector receives a row whose mandatory side, position-side,
quantity, or close-only semantics cannot be normalized, it discards that row, requests an
authoritative account-state refresh, and emits a bounded warning. Other valid rows from the same
websocket message remain usable. A semantic row rejection must not be reported as a transport
disconnect or consume the websocket reconnect budget; actual watch/read failures retain the
bounded reconnect backoff.

## Broker Agreement Attribution

Problem:

1. Broker attribution is implemented differently per exchange: headers, CCXT options, order tags, or client order ids.
2. CCXT defaults may point attribution to CCXT, not Passivbot.
3. Removing broker code can silently break Passivbot broker agreements while trading continues normally.

Handling in Passivbot:

1. Treat broker-code handling as exchange-critical behavior.
2. Do not remove existing broker attribution without explicit user approval.
3. Broker-code registry loading must fail loudly on missing/invalid registry data and unknown exchange names.
4. For each broker-agreement exchange, verify the actual signed CCXT/raw request includes the required broker field/header/tag.
5. Add regression tests at the request-construction boundary when changing exchange sessions, signing, or order payload code.

### WEEX broker client-order IDs

WEEX attributes both Spot and Futures API volume when `newClientOrderId` starts with
`b-{brokerId}-`. Passivbot's WEEX connector is Futures-only and the V3 Futures order endpoint
limits this field to 36 characters, so every generated ID uses
`b-{brokerId}-0xTTTT{random}` within that limit. The `0xTTTT` Passivbot order-type marker must
remain intact for ownership, reconciliation, and fill-event diagnostics. Reject invalid or
unattributed IDs before submission; do not rely on CCXT's WEEX `partner` default because CCXT
ignores it when the caller supplies a client order ID.

## Exchange Hedge Mode Versus Strategy Hedge Mode

`bot.portfolio.hedge_mode=false` disables simultaneous long and short strategy exposure; it does not put an
exchange account into one-way mode. Binance and Bitget connectors keep the exchange account in
hedge mode. Their private order updates must therefore normalize `position_side` and close-only
semantics from the exchange's actual mode and explicit `positionSide`/`posSide` fields even when
the strategy setting is false. Only connectors whose `hedge_mode` capability is actually false may
use the one-way side plus `reduceOnly` attribution path.

## Venue Contracts
Open only the venue being changed:

| Venue | Contract |
|---|---|
| Binance | [`exchanges/binance.md`](exchanges/binance.md) |
| Bybit | [`exchanges/bybit.md`](exchanges/bybit.md) |
| KuCoin Futures | [`exchanges/kucoin.md`](exchanges/kucoin.md) |
| Bitget Futures | [`exchanges/bitget.md`](exchanges/bitget.md) |
| OKX Futures | [`exchanges/okx.md`](exchanges/okx.md) |
| Gate.io Futures | [`exchanges/gateio.md`](exchanges/gateio.md) |
| Hyperliquid | [`exchanges/hyperliquid.md`](exchanges/hyperliquid.md) |
| Bitunix Futures | [`exchanges/bitunix.md`](exchanges/bitunix.md) |
| WEEX Futures | [`exchanges/weex.md`](exchanges/weex.md) |
| Lighter USDC Perpetuals | [`exchanges/lighter.md`](exchanges/lighter.md) |

## Unavailable Configured Live Markets

Live coin overrides are resolved against the current exchange market snapshot before downstream
mode or sizing lookups. Skip an unavailable market (including an exact identifier classified as
`UnknownMarketIdentifier`) with a bounded notice on change, retaining the original config so the
next market refresh can reconsider it. Do not synthesize an active market or change the shared
resolver's historical/backtest behavior. Ambiguous identifiers, conflicting overrides, and other
resolution failures still propagate; a failed refresh must not publish a partial override map.

An inactive market is not an absent market: retain its overrides and existing protective-mode
handling. Never drop exchange positions or open orders because their coin is unavailable in config;
missing market metadata for such state remains an explicit failure. Approved-list filtering also
applies when the eligible market set is empty.
Connector-specific live-state hooks must run the shared base validation before any connector
early return, including Hyperliquid unified-account support.

## General Guidance

1. Check raw exchange payloads when CCXT abstraction is insufficient.
2. Treat intra-page gaps and inter-page gaps differently.
3. For missing data incidents, verify source data before changing logic.

## Validation

- Exercise actual CCXT/raw request construction for payload, header, broker, and client-ID changes.
- Use sanitized response fixtures for normalization and ambiguous-side cases.
- Test pagination overlap, deduplication, and retention boundaries with multi-page fixtures.
- Keep authenticated exchange checks outside the default suite and require explicit approval.

## Key Code And Tests

- `src/exchanges/`
- `src/fill_events_manager.py`
- `tests/exchanges/`
- `tests/ccxt_upgrade/`
