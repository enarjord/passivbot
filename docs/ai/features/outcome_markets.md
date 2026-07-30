# Binary Outcome Markets

## Scope

This contract defines Passivbot's venue-neutral model for fully collateralized binary outcome
markets. Hyperliquid HIP-4 is the first live venue. Polymarket is the second reference venue used
to keep the domain, data, simulation, and strategy contracts exchange-independent.

Binary support does not make multi-outcome, negative-risk, scalar, leveraged, or AMM markets
implicitly supported. Those require their own explicit contracts.

## Ownership

- Rust owns outcome strategy intent, canonical quote translation, inventory accounting, fill and
  fee simulation, settlement, backtest behavior, and derived metrics.
- Python owns venue discovery, public event collection, durable storage, live exchange I/O,
  account reconciliation, capability gating, and chronological backtest orchestration.
- Venue adapters translate canonical Rust intent into native token and order actions. They must not
  reimplement pricing or inventory policy.
- The existing perpetual-futures position and backtest model remains separate. Outcome code may
  share generic time-series and analysis infrastructure but must not reinterpret outcome tokens as
  leveraged long/short positions.

CCXT prediction-market connectors may be used as venue transports after their raw contracts are
validated, but they do not replace the canonical outcome adapter. In particular, a connector's
normalized fee, quote asset, precision, lifecycle, or capability defaults are not authoritative
when the venue payload exposes a different value.

## Canonical Binary Model

For a market with one collateral unit of payout, every valid settlement is represented by
`yes_fraction` in `[0, 1]`:

```text
yes_payout = payout_unit * yes_fraction
no_payout  = payout_unit * (1 - yes_fraction)
```

Ordinary binary settlement uses `yes_fraction` equal to `0` or `1`. The fractional representation
is retained for venues that report invalid, ambiguous, interpolated, or otherwise fractional
settlements.

The initial normalized venue contract supports one collateral unit of payout, so
`payout_unit == 1.0` and all native and canonical prices remain in `[0, 1]`. Rust keeps the payout
term explicit for accounting clarity; accepting non-unit payout markets would require a deliberate
extension of every normalized trade, account-fill, book, candle, adapter, and archive boundary.

Strategy pricing uses one canonical YES coordinate. Native NO price `n` maps to canonical YES price
`payout_unit - n`:

| Native action | Canonical YES price | Binary exposure change |
|---|---:|---:|
| Buy YES at `p` | `p` | `+qty` |
| Sell YES at `p` | `p` | `-qty` |
| Buy NO at `n` | `payout_unit - n` | `-qty` |
| Sell NO at `n` | `payout_unit - n` | `+qty` |

Canonical equivalence does not erase native execution. Every synthetic level and intended order
retains its venue, asset, native side, native price, and executable action. The execution planner
chooses among native actions based on venue capabilities, available token inventory, collateral,
fees, order constraints, and split/merge support.

## Inventory And Accounting

The authoritative outcome ledger tracks at least:

- free and order-reserved collateral
- YES and NO quantities and cost bases
- paired quantity `min(yes_qty, no_qty)`
- residual exposure `yes_qty - no_qty`
- realized trading, merge, settlement, fee, and rebate cash flows
- unsettled and redeemable settlement value

Paired inventory is settlement-neutral. Residual exposure is binary directional risk. For total
inventory cost `C`:

```text
YES settlement PnL = yes_qty * payout_unit - C
NO settlement PnL  = no_qty  * payout_unit - C
```

For equal quantities `q`, gross locked-in PnL is:

```text
q * (payout_unit - average_yes_cost - average_no_cost)
```

These canonical mappings are exposure equivalences, not unconditional execution equivalences.
Buying NO at `n` is economically equivalent to splitting one collateral unit into YES+NO and
selling the YES share at `1 - n`, before fees and operational costs. It is not the same operation
as selling pre-existing YES inventory unless inventory acquisition is included. Likewise, buying
equal YES and NO quantities locks pair value until merge or settlement, while buying and later
selling the same native share realizes a round-trip PnL and ends flat. The paths coincide only
when their complete collateral transformations, prices, fees, rebates, inventory constraints, and
capital lockup are identical.

Split and merge operations are inventory transformations, not trades:

```text
split: collateral -> equal YES and NO quantities
merge: equal YES and NO quantities -> collateral
```

The ledger must conserve marked equity across a fee-free split or merge. A split needs a reference
YES price solely to allocate cost basis between its two assets; that allocation must sum exactly
to the collateral consumed.

## Market Lifecycle

Do not collapse these timestamps or states:

1. discovery or market creation
2. trading open
3. order-entry cutoff or trading close
4. scheduled event/expiry
5. authoritative resolution
6. collateral settlement or token redemption

Hyperliquid may settle a recurring price outcome close to its scheduled event. Polymarket may
remain unresolved through proposal, challenge, and dispute periods. Strategy orders must expire or
be cancelled before the venue-specific cutoff. Capital is reusable only when authoritative account
state proves it available.

The Rust simulation contract retains both market-open time and the effective order-entry-open time.
The latter is the later of trading open and an authoritative venue order-acceptance timestamp.
Signal history may begin at trading open, but quotes and explicit orders cannot begin before order
entry opens. An actual trading close may occur after the scheduled event; the scheduled event,
actual close, resolution, and capital release remain independent lifecycle facts.

Settlement truth includes the immutable market terms, resolution source, final payout fraction,
resolution timestamp, and settlement/redemption evidence. A candle close is not a settlement
oracle unless the venue contract explicitly says it is.

For HIP-4, a market disappearing from `outcomeMeta` after its scheduled event proves only
`expired_awaiting_settlement`. An account fill with `dir = Settlement` and a consistent binary
YES/NO payout is authoritative settlement evidence. Settlement rows are archived separately and
must never become public trades, signal candles, or execution candles. Live reconciliation may
cancel an authoritative managed order after expiry, but new orders require an active lifecycle.

For standard Polymarket markets, the Polygon Conditional Tokens contract's
`ConditionResolution` event is authoritative. Preserve its condition ID, block timestamp,
transaction/log identity, and complete payout-numerator vector. Map that vector through the
market's retained original outcome ordering; do not assume canonical YES is always vector index
zero. Binary `[1, 0]`, `[0, 1]`, and fractional `[1, 1]` payouts respectively normalize to YES
fractions `1`, `0`, and `0.5`. Gamma `outcomePrices`, the last fill, and book state are not
settlement evidence. Resolution makes winning shares redeemable; it does not prove that a wallet
has redeemed them or that collateral is reusable. Portfolio replay and live allocation release
therefore require a redemption event or authoritative account-availability timestamp in addition
to `ConditionResolution`.

## Fees And Incentives

Fees are functions of venue, market, native action, liquidity role, price, quantity, and lifecycle
stage. Do not assume a constant perp-style notional rate.

The simulator supports pluggable fee formulas, configurable fill incidence (`every_fill` or
`inventory_reduction_only`), payout-notional settlement fees, and authoritative signed per-fill
fees. In the current fully collateralized inventory model, buys increase token inventory and sells
reduce it. A positive fee is a charge and a negative fee is a rebate. Rebate proceeds are credited
only after execution and cannot fund the gross purchase or reduce resting-order collateral
reservation. Rewards, gas sponsorship, and other incentives remain separate cash events unless
the venue reports them as fill-level fees.

Current zero-fee venue behavior is runtime metadata, not a permanent default.

Backtest results report `trading_fees_paid`, `settlement_fees_paid`, and their sum
`fees_paid` separately. Trading fees are captured before settlement and are invariant when the
same fill path is replayed across alternative settlement outcomes. Settlement and total fees may
vary with the winning inventory and payout. Settlement-scenario summaries compare only trading
fees as a path invariant and report ranges for settlement and total fees.

## Orders And Fill Simulation

The common order model preserves:

- market and asset identity
- native YES/NO asset
- buy/sell side
- canonical YES price and exposure change
- native price and quantity
- maker/taker intent, post-only, time-in-force, and expiry
- lifecycle state including partial fills, cancellation, rejection, and ambiguity

The initial outcome backtester uses trade-derived one-second candles for both strategy inputs and
fill simulation. Order-book quote changes never contribute to these candles.

A resting canonical bid fills only when the candle has positive trade volume and its low is
strictly below the bid. A resting canonical ask fills only when the candle has positive trade
volume and its high is strictly above the ask:

```text
fill_bid = candle.volume > 0 and candle.low  < bid_price
fill_ask = candle.volume > 0 and candle.high > ask_price
```

A touch does not fill. The initial candle model does not reconstruct queue position or cap
simulated fill quantity by candle volume; outputs identify this fill model explicitly. When YES
and NO have separate native books, execution candles retain their native source asset even though
their prices are transformed into the canonical YES coordinate. This prevents a trade in one
native book from falsely filling an order resting only in the other.

This initial execution model is passive-only. It accepts post-only orders and rejects non-post-only
orders until an explicit taker model defines immediate execution, liquidity role, fees, and
slippage. A GTD expiry must be later than the order's placement timestamp and no later than the
market's trading close; an already-expired order is rejected rather than rested until a later
bucket.

Some venues use a merged complementary book and report one economic trade as mirrored YES and NO
fill records. Adapters retain both native records for audit and execution reconciliation, attach a
shared economic-event identity when the venue proves one, and count that event once in the
canonical signal candle.

## Public Data Contract

Persist immutable normalized raw fills before deriving bars. A normalized fill retains:

- venue, market, and native asset identifiers
- exchange event time, local receive time, and collector-session sequence
- venue event/trade/ordered-sequence identifier where available
- native side, price, quantity, and event kind
- enough source metadata to deduplicate and detect gaps

Persist normalized market metadata at discovery and at every historical import before its fills.
The archive fingerprints immutable terms—venue and market identity, description, side assets,
payout, scheduled event, and capabilities—and fails if the same venue market ID is later observed
with conflicting contract terms. Constraint, fee, price-grid, and lifecycle observations may
change without replacing the retained contract. This is required because expired HIP-4
price-binary rows disappear from `outcomeMeta`.

Settlement source-event identity is immutable as well. Re-importing the same venue, market, and
source event may differ in observation metadata such as receive time or endpoint provenance, but
contradictory payout, winner, event time, release time, fee, quantity, or raw authoritative
evidence is archive corruption and must fail.

Trade source-event and ordered-sequence identities are immutable. Re-importing one of those
identities may differ in observation metadata such as receive time, collector session, or source
cursor, but contradictory outcome, native side, price, quantity, exchange time, economic-event
identity, or alternate sequence/source identity is archive corruption and must fail.

Replay consolidates lifecycle observations without replacing the initial trading terms. In
particular, an initial live Polymarket observation commonly has no `closedTime`; a later closed
observation supplies the authoritative close. Full-contract replay fails closed if quantity-step
or minimum-order constraints change because the initial simulator does not yet model those
constraint transitions.

Preserve raw venue payloads or lossless fixtures at the adapter boundary when practical. Derive
one-second bars reproducibly from actual fills, never from bids, asks, midpoints, marks, or reported
probabilities. During a verified-continuous collection interval, no-trade seconds carry the prior
fill close with `open = high = low = close` and zero volume. Unknown collector gaps remain
unavailable and must not be fabricated into flat candles.

Open and close require a proven chronological order within each second. If multiple fills share
the same exchange timestamp, use a unique venue-ordered sequence when available, otherwise the
collector's monotonic observation sequence, and only then distinct local receive times. A
historical endpoint that supplies only whole-second timestamps and no event sequence may still
prove high, low, and volume, but it is not valid full OHLC/EMA input; do not silently trust its
response-array order. Collector sequence is session-scoped chronology, never a trade identity.

Preferred backfill sources must preserve event chronology:

- Hyperliquid: use the public `hl-mainnet-node-data/node_fills_by_block` archive, whose
  block-batched output can retain block metadata and event order. `recentTrades` is only a bounded
  bootstrap/diagnostic source: `tid` is a trade-identity hash, not an ordered sequence. Historical
  ingestion streams chronologically ordered plain or LZ4 hourly files, verifies block continuity
  across file boundaries, and archives coverage only after the complete input manifest parses.
- Polymarket: use Polygon CTF Exchange `OrderFilled` logs ordered by block number, transaction
  index, and log index, joined to Gamma market/token metadata. Normalize both the original CTF
  Exchange event schema and CTF Exchange v2 into the same canonical trade contract. Import the
  requested condition's CTF `ConditionResolution` log separately as settlement evidence. The
  public Data API's whole-second trade timestamps are insufficient for full OHLC when multiple
  fills share a second.

The standard-market Polygon downloader queries both official CTF Exchange addresses, both
`OrderFilled` topics, and the condition-indexed CTF resolution topic over a complete block
interval. It finds the first block at or after each second-aligned time boundary, attaches
canonical block timestamps, rejects removed, duplicate, or out-of-range logs, excludes a
configurable confirmation depth from the chain head, and records coverage only after the entire
range is decoded and archived. RPC endpoints are transport configuration: public providers may
impose archive, range, traffic, or retention limits, and any such failure must leave the interval
uncovered.

Polymarket collateral identity is versioned transport metadata, not a permanent `USDC` constant.
Gamma currently omits it on some market rows, and the venue is migrating from USDC.e to pUSD.
Collectors and historical importers must therefore receive an explicit authoritative
`quote_asset` when the raw market payload does not declare one, and persist it with the market
version. Never infer the collateral era from the title, terminal prices, or current date.

CCXT `fetchOHLCV` or `fetchTrades` output is acceptable only when its retained raw source proves
the same actual-fill, volume, identity, and ordering contract. Venue-generated one-minute bars,
price-history samples without volume, bounded recent-trade windows, and whole-second trade rows
without log order are not substitutes for ordered raw fills.

When CCXT prediction connectors are used as an optional transport, construct them through
Passivbot's outcome transport factory. Builder attribution is disabled by default. A connector
must never turn its first order into an implicit builder approval or attach builder metadata unless
the caller explicitly opted into that separate capability.

Archive full public book snapshots separately when the venue exposes them. They support live
post-only checks, spread/depth diagnostics, and a possible future book-replay model, but they are
not inputs to the initial signal or candle fill model. Preserve venue capability differences such
as Hyperliquid's per-level order count and Polymarket's aggregated levels without inventing absent
fields.

For current HIP-4 markets, derive the quote asset from each live `outcomeMeta.quoteToken` rather
than hardcoding a settlement token. Reconcile side-token inventory from
`spotClearinghouseState`; outcome rows use the `+<encoding>` coin form and may omit the ordinary
spot token index. Reconcile orders and fills by their `#<encoding>` coin. Fill-level `fee` and
`feeToken` are authoritative. Fetch the account's current `userFees` rates as a required live
input. Until HIP-4-specific fee incidence is authoritative, strategy edge gating uses the larger
non-negative maker rate reported for regular and spot trading as a conservative per-share floor;
that floor does not overwrite actual fill fees or assert a settlement-fee rate.

HIP-4 lifecycle reconciliation first checks settlement rows present in the current `userFills`
snapshot. After scheduled expiry it also queries `userFillsByTime` from the event timestamp
through the observation time. This endpoint is a bounded recovery source, not durable state:
Hyperliquid documents a 2,000-row response limit and availability of only the account's 10,000
most recent fills. Persist every observed settlement record before it can age out. If neither
current account state nor retained evidence proves a payout, remain
`expired_awaiting_settlement`; never infer the winner from market disappearance or price.
A bounded settlement-history lookup failure is recorded on that unresolved lifecycle and does not
block protective cancellation of exact managed orders.

## Outcome EMA Anchor

`ema_anchor_outcome` is a derivative strategy, not a direct reuse of perpetual-futures sizing.
EMA spans are measured in seconds against the dense actual-fill-derived one-second signal series
and must be at least one one-second observation.
Quote offsets and inventory skew are absolute probability points on `[0, payout_unit]`; they are
not unbounded multiplicative percentages. Every quote remains a canonical bid or ask but resolves
to an executable native YES/NO buy or inventory-backed sell.

The initial strategy exposes three execution modes for comparative backtests:

- `accumulate_pairs` buys YES below the anchor and NO above it, targeting complete-set edge.
  Once one side leads, it suppresses further orders that would increase the residual and quotes
  only the missing complement at the most aggressive maker price consistent with the configured
  locked-pair edge
- `inventory_aware` sells owned complementary inventory before adding more shares
- `yes_only` buys and sells only the canonical YES share

All modes enforce total-inventory and residual-exposure limits. A configurable
`risk_reduction_only_ms_before_close` phase stops pair completion and new entries and quotes only
passive, inventory-backed sales of the excess YES or NO token. This reduces both settlement
residual and gross inventory instead of adding the missing complement late in the lifecycle. A later
`entry_cutoff_ms_before_close` cancels quote generation entirely. Backtest comparison must report
both settlement outcomes, pair-completion ratio, and settlement sensitivity; spread PnL alone is
not a sufficient objective.

Total token quantities remain authoritative for exposure, cost basis, and strategy skew.
Executable sell sizing uses available token quantities plus only inventory reserved by
Passivbot-managed sell orders that the same reconciliation will cancel. Inventory held by
unmanaged user orders is never treated as reclaimable.

## Backtest Composition

The Rust single-market kernel simulates one market from trading through settlement and emits fills,
ledger events, equity, exposure, capital usage, and outcome metrics.

The Python outcome orchestrator runs markets in chronological order against a shared wallet.
Independent single-market runs must not each receive the full starting balance and then be averaged
as if capital were reusable. The orchestrator accounts for overlapping markets, locked complete
sets, delayed resolution/redemption, and venue-specific settlement timing.

An archived full-contract job is admissible only when the archive contains immutable market
metadata, one consistent authoritative settlement payout, actual fills, and continuous verified
coverage for both native side assets from trading open through trading close. The archive replay
builder also requires independent full-window price-grid stream coverage on venues such as
Polymarket where tick-size changes are a separate event source. Fill coverage does not prove grid
coverage. A bounded live capture without an authoritative grid-subscription readiness boundary
archives observed changes but does not certify grid coverage. The builder supplies the
authoritative capital-release timestamp and payout to Rust; resolution-only evidence remains
archived but is not a release timestamp. The EMA-anchor job adapter invokes the EMA strategy
kernel—not the generic scripted-action simulator—before the shared-wallet orchestrator locks that
job's allocation until that release.

Required aggregate metrics include gross spread capture, fees, rebates, settlement PnL, paired and
residual inventory, worst-case settlement equity, time-weighted exposure, capital utilization,
maker ratio, fill rate, and post-fill adverse selection. `pair_completion_ratio` is based on
cumulative YES and NO buy quantities (`min(YES buys, NO buys) / max(YES buys, NO buys)`), not on
the final flatness of inventory. Buying YES and later selling YES is a round trip with zero pair
completion, even though no settlement residual remains.

The Rust result reports cumulative YES/NO buys, pair completion, peak absolute residual, and
time-weighted absolute residual and total token inventory from trading open through settlement.
With one-second aggregate execution candles, all fills eligible in a bucket are applied at that
bucket's timestamp and the resulting inventory is charged through that bucket's one-second end.
This is a deterministic bucket model, not a claim about sub-second fill order. The shared-wallet
orchestrator aggregates buy quantities before calculating portfolio pair completion and weights
inventory-time areas across overlapping contracts on the common executed-market horizon. Skipped
jobs do not extend that horizon. Portfolio peak residual is swept chronologically as the sum of
absolute per-market residuals; every release and fill-derived residual update sharing one timestamp
is applied atomically before measuring the peak.

Post-fill adverse selection is represented as a horizon-indexed collection so later experiments
may add horizons without changing its aggregation contract. The initial collection contains only
`horizon_ms = 1000`. A fill timestamp denotes the start of its one-second execution bucket, and its
one-second mark is the canonical YES close at that bucket's end. EMA-anchor runs use the canonical
signal close. Scripted-action runs use the canonicalized close from the native book that produced
the fill, or the single merged-book close when the venue proves complementary books are merged.

For canonical YES-equivalent exposure direction `d` (`+1` for buying YES or selling NO, `-1` for
selling YES or buying NO), canonical fill price `f`, mark price `m`, and quantity `q`:

```text
adverse_selection_per_share = d * (f - m)
adverse_selection_quote     = q * adverse_selection_per_share
```

Positive values are adverse and negative values are favorable. Each horizon reports total and
observed fill count and quantity, quantity coverage, total quote-currency markout, and the
quantity-weighted mean per share. An unavailable exact-horizon mark is excluded and remains
observable through coverage; settlement payout is never substituted for a missing mark. Portfolio
and mode summaries preserve the horizon and aggregate total markout before calculating the
quantity-weighted mean.

## Live Safety

Before any outcome order action, require fresh authoritative:

1. free and reserved collateral
2. YES and NO token balances
3. open orders
4. current venue capabilities and market constraints
5. market lifecycle and order-acceptance state
6. symbol-scoped book and strategy inputs

Missing outcome inventory, settlement, fee, or lifecycle inputs are unavailable and fail closed.
Do not substitute perp positions, zero balances, assumed fees, title-parsed expiry, or candle-based
settlement.

An active lifecycle may create or replace managed quotes. An expired or settled lifecycle targets
an empty managed-order set and reports the exact state and settlement evidence, if any. Protective
cancellation is allowed after expiry only for an order proven by the fresh account snapshot to
belong to the retained market, outcome side, and exact expected client-order ID. The mutation
executor independently validates every cancellation and creation against the deterministic
Passivbot outcome namespace before its first write, then verifies the complete final managed-order
set rather than trusting the previously constructed reconciliation object. Kept orders must still
match their exact expected remaining quantity and terms. If final verification fails, the executor
cancels every surviving managed quote for the market, continues through individual cancellation
errors, and verifies the managed set absent before propagating the failure.

If the verified actual-fill signal is unavailable or stale, new and replacement quotes are
unavailable. Reconciliation targets an empty managed-order set for the affected outcome market:
cancel Passivbot-namespaced quotes and preserve all unmanaged user orders. A missing fill is never
converted into a fabricated candle merely to keep existing quotes alive.

Live split, merge, redeem, and order writes are distinct authenticated mutations. Each requires an
explicitly supported adapter path, reconciliation, idempotency or authoritative confirmation, and
the normal approval boundary in `AGENTS.md`.

## Initial Non-Goals

- authenticated Polymarket trading
- multi-outcome and negative-risk markets
- cross-venue arbitrage
- liquidity-incentive optimization
- order-book replay, exact exchange queue reconstruction, or sub-second HFT simulation
- applying existing perp strategies to outcome markets without an outcome-specific contract

## Validation

- Algebraic tests cover all four native actions and complementary-price mapping.
- Conservation tests cover split, merge, fees, and both settlement outcomes.
- Fill tests cover insufficient collateral/inventory, partial quantities, price bounds, and fees.
- Venue fixtures prove HIP-4 and Polymarket metadata/book translation without credentials.
- Data tests cover deduplication, ordering, verified no-trade seconds, and unknown gaps.
- Multi-market tests prove shared-capital behavior across overlap and delayed settlement.
- Strategy tests compare live and backtest canonical intents.
- Authenticated writes remain outside default tests and require explicit approval.

## Key Code

- `passivbot-rust/src/outcome.rs`
- `src/outcome/capture.py`
- `src/outcome/`
- `tests/outcome/`
