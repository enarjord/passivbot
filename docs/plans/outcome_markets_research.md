# Outcome Markets Research And Validation Notes

Status: active foundation work, 2026-07-24.

This note records empirical findings and implementation decisions. The normative contract is
[`docs/ai/features/outcome_markets.md`](../ai/features/outcome_markets.md).

## Current Conclusions

1. Extend the existing Rust backtester with a separate outcome simulation layer. Reusing its
   one-second time axis, orchestration boundary, and compiled runtime is materially simpler than
   maintaining a second backtester, while the separate outcome ledger avoids forcing binary shares
   into the perpetual-position model.
2. Use actual fills for both signals and fill eligibility. Verified no-trade seconds carry the
   prior close with zero volume. A book move never becomes a candle, and a limit order requires
   positive volume plus a strict trade-through.
3. Use canonical YES price for strategy state while retaining native YES/NO execution books.
   Canonical pricing makes the four native actions comparable but does not make their inventory,
   fee, capital-lock, or execution paths interchangeable.
4. Treat `accumulate_pairs`, inventory-aware native round trips, and YES-only round trips as
   separate strategy modes. Buying both shares is settlement-neutral only after the complement
   actually fills. Until then it is directional inventory.
5. Keep `ema_anchor_outcome` separate from perpetual `ema_anchor`. Its offsets and inventory skew
   are absolute probability points, its spans are seconds, and its risk objective includes pair
   completion and worst-case settlement rather than only realized spread.

## CCXT Prediction Namespace

The repository now pins `ccxt==4.5.68`. Contrary to the initial assessment, direct wheel comparison
confirmed that 4.5.66 already contained the dedicated Hyperliquid and Polymarket connectors under
the [`ccxt.prediction` namespace](https://github.com/ccxt/ccxt/wiki/manual#prediction-markets).
The upgrade is therefore maintenance and forward-compatibility work, not what makes prediction
connectors available.

The upgrade followed the repository CCXT capture/diff runbook:

- Authorized read-only account snapshots before and after the upgrade had identical
  capabilities, eligible and ineligible symbol sets, market symbol sets (468/468), limits,
  contract metadata, balance/order/position/trade shapes, and trade item schemas. The normalized
  balance was exactly unchanged.
- Source comparison from 4.5.66 to 4.5.68 found no change in ordinary Hyperliquid, Bitget, Gate.io,
  KuCoin, or WEEX connectors. The relevant ordinary-connector changes were bounded Binance balance
  and leverage normalization, Bybit option classification, and OKX closed/cancelled-order
  pagination direction.
- Public unauthenticated ticker, bulk-ticker, and order-book probes passed for Binance USD-M,
  Bybit, Bitget, Gate.io, OKX, Hyperliquid, WEEX, Paradex, and KuCoin Futures. The KuCoin probe now
  requests its supported 20-level minimum instead of the generic five-level limit.
- A public-only Hyperliquid CLI probe also passed market loading, server time, ticker, concurrent
  ticker, order book, and one-minute OHLCV paths and proved that it made zero private calls.

The prediction connectors still do not own Passivbot's outcome contract:

- The isolated Hyperliquid connector normalized the live HIP-4 quote token as `USDH` while the raw
  market declared `USDC`.
- It supplied generic fixed precision and fee defaults instead of treating HIP-4 significant-figure
  pricing and fill-level fees as authoritative.
- Its smallest HIP-4 OHLCV interval is one minute.
- Polymarket discovery, order signing, and live market websocket support are useful, but its
  `fetchOHLCV` price-history buckets contain no actual volume and its public `fetchTrades`
  implementation explicitly ignores `since`; neither proves complete ordered one-second fills.

CCXT 4.5.68 also enables builder attribution by default on both prediction connectors. In
particular, Hyperliquid's connector may submit an authenticated `approveBuilderFee` action before
its first order even when its default builder fee is zero. Passivbot's prediction-client factory
therefore forces `options.builderFee = false` unless a caller explicitly opts in, with an offline
regression test proving that default initialization cannot call the approval method.

Decision: retain Passivbot's canonical adapters and use CCXT only as an optional transport. Do not
route HIP-4 constraints, fees, lifecycle, settlement, or candles through CCXT defaults. Builder
attribution is a separately authorized mutation capability, not an incidental transport default.

Polymarket's July 2026 documentation describes a USDC.e-to-pUSD collateral migration, while a
current public Gamma market row still omitted both `denominationToken` and `collateralToken`.
Passivbot therefore does not let the adapter silently choose a collateral era: capture and
backfill tools require `--quote-asset` when Gamma omits it, and archive that identity with the
market metadata.

An isolated public probe of 4.5.68 confirmed that both prediction connectors expose market discovery,
trades, one-minute-or-coarser OHLCV, balances, positions, account fills, open orders, create, and
cancel methods. Polymarket additionally exposes websocket trades and books; Hyperliquid does not.
Neither connector exposes outcome settlement or redemption. These gaps preserve several
Passivbot-owned responsibilities:

| Contract | Hyperliquid prediction | Polymarket prediction | Passivbot decision |
| --- | --- | --- | --- |
| Live trades/books | REST only | websocket supported | retain canonical streams and normalization |
| Smallest OHLCV | 1 minute | 1 minute | build 1-second candles from actual fills |
| Settlement/redeem | absent | absent | retain canonical lifecycle and payout layer |
| Quote normalization | reported `USDH` for live HIP-4 | `USDC` | raw venue payload is authoritative |
| Price/quantity rules | generic connector precision | dynamic fixed ticks | validate against raw market metadata |

The prediction namespace is therefore a useful future transport abstraction, not Passivbot's
outcome domain model.

The isolated CCXT Hyperliquid probe found the expected BTC outcome 913 and its expiry, target, and
YES/NO encodings, but reported generic fixed 0.0001 price/amount precision and quote `USDH` while
Passivbot's raw venue adapter uses significant-figure pricing and the live payload's `USDC` quote.
The Polymarket search found the expected daily Bitcoin-above family, including outcome IDs, 0.001
ticks, fee schedules, and resolved winners, but returned closed markets in a search that must still
be lifecycle-filtered by Passivbot.

### Current HIP-4 read-only account evidence

An authorized dry cycle on 2026-07-24 used a funded account and selected a current daily BTC
price-binary outcome. It performed no exchange mutations and reported:

- nonzero USDC collateral and a positive conservatively available balance;
- zero YES/NO inventory, zero open orders, and zero recent selected-market fills;
- finite regular and spot maker/taker rates from the required `userFees` response;
- no public outcome fill during the three-second collection window, so the live planner remained
  unavailable and reconciliation targeted an empty managed-order set.

The account is funded and the read-only integration works. The fee endpoint does not itself prove
which rate, incidence, or settlement charge HIP-4 applies. The live planner therefore floors its
fee-per-share estimate at 0.0004 while retaining fill-level fees as authoritative and leaving
settlement rate unassumed.

Public participant addresses embedded in the archived HIP-4 trade stream made it possible to
cross-check the same transaction hashes through Hyperliquid's unauthenticated `userFills`
endpoint. Across 27 economic transactions, the matched sample contained 60 participant fill rows:
51 `Buy` rows and 9 `Sell` rows, with both maker and taker examples. Every fill reported fee
`0.0`. Buy rows named the acquired outcome token as `feeToken`; sell rows named `USDC`, but the
amount was zero in both cases. This is authoritative evidence that both opening/mint and
inventory-reduction/burn trades in the sample were fee-free. It supersedes secondary descriptions
which suggested that closing fills are charged. It does not prove future fee policy or settlement
charges, so fee incidence and settlement rate remain configurable and the conservative live edge
floor remains in place.

### HIP-4 settlement evidence

Public unauthenticated `userFills` samples also exposed 12 explicit settlement rows across prior
HIP-4 markets. Each row used `dir = Settlement`, sell side, `crossed = true`, quantity equal to the
starting outcome-token position, a price of exactly `0` or `1`, `USDC` as the fee token, and a zero
fee. Complementary outcomes resolved consistently. This is direct evidence that Hyperliquid
records settlement as an account fill and that the observed settlements had no fee; it does not
guarantee future fee policy.

The live adapter now normalizes these rows into separate market-level settlement evidence and
excludes them from ordinary trade and candle ingestion. Market disappearance from `outcomeMeta`
after the scheduled event produces `expired_awaiting_settlement`, not an invented winner. The
adapter checks both the recent account snapshot and the documented `userFillsByTime` recovery
window. The latter is limited to 2,000 rows per response and the account's 10,000 most recent
fills, so normalized settlement evidence is also stored durably in the outcome archive.

CCXT 4.5.68 does not recover old price-binary metadata: its Hyperliquid prediction connector
groups only the current `outcomeMeta` response. The archive therefore stores normalized market
metadata before live collection and historical ingestion, rejects conflicting immutable terms for
one venue market ID, and provides a streaming importer for chronologically ordered
`node_fills_by_block` NDJSON or LZ4 files. The official bucket is requester-pays; no transfer was
started without explicit cost authorization.

## Historical Trade-Data Source Evaluation

The canonical archive should accept multiple transports, but it must retain source identity,
coverage boundaries, raw ordering keys, and enough raw payload to audit normalization. A convenient
vendor candle is not interchangeable with ordered fills.

### Hyperliquid

| Rank | Source | Suitability | Decision |
| --- | --- | --- | --- |
| 1 | [Official `node_fills_by_block` S3 archive](https://hyperliquid.gitbook.io/hyperliquid-docs/historical-data) | Native fill rows batched by block; best available first-party chronology. Requester pays transfer, file layout/coverage must be inventoried, and the official docs do not guarantee completeness or timeliness for every dataset. | Primary authoritative backfill. Keep downloads external to Passivbot, then stream NDJSON/LZ4 through the existing manifest and block-continuity importer. Do not spend requester-pays funds without approval. |
| 2 | [Dune HyperCore](https://dune.com/blog/hyperliquid-hypercore-is-live-on-dune) | Advertises every fill with price, size, side, fee, and PnL plus HIP-4 definitions/outcomes; SQL/export is operationally attractive. Access is an enterprise add-on. | Strong managed source and independent cross-check if access is justified. Require a HIP-4 sample export containing deterministic fill order and stable market IDs before integrating. |
| 3 | [0xArchive HIP-4 transport](https://docs.0xarchive.io/crypto-historical-data-api) | Explicit HIP-4 routes, bounded REST, sequenced replay, exports, and data-quality context. Commercial service and raw redistribution rights are not implicit. | Candidate managed feed. Validate one settled contract row-for-row against S3 before trusting coverage. |
| 4 | [Dwellir historical products](https://www.dwellir.com/docs/hyperliquid/historical-data) | Offers native raw archives, tick data with block sequencing, and one-second OHLCV. Published coverage begins at different dates and does not yet make HIP-4 contract coverage explicit. | Useful operational alternative only after the provider confirms HIP-4 IDs, launch-date coverage, ordering, and terms. Prefer ticks/raw fills over derived candles. |
| 5 | Hyperliquid REST/websocket | Excellent for forward capture and bounded repair, but `recentTrades` is only recent public data; account `userFillsByTime` is capped at 2,000 rows per call and the account's latest 10,000 fills. | Live collector, bootstrap, and audit cross-check only—not historical market-wide backfill. |

Recommended first acquisition is a small official S3 manifest covering several complete HIP-4
contracts and both resolutions, after explicit requester-pays approval. Compare it with a bounded
managed-feed sample before deciding whether operating the S3 downloader is cheaper than buying a
normalized feed.

### Polymarket

| Rank | Source | Suitability | Decision |
| --- | --- | --- | --- |
| 1 | Polygon CTF Exchange `OrderFilled` plus CTF `ConditionResolution` logs | First-party settlement records with `(block_number, transaction_index, log_index)` chronology. Covers the legacy and current standard CTF Exchange generations when both addresses and event layouts are decoded. Archive RPCs can reject large or old ranges. | Canonical source. The current downloader already bisects ranges, decodes v1/v2, removes the exchange-facing duplicate, imports resolution vectors, and records coverage only after the complete confirmed interval succeeds. |
| 2 | [Goldsky Polymarket datasets](https://docs.goldsky.com/chains/polymarket) | Historical and realtime `Order Filled` datasets with block-range fast scan and warehouse/webhook sinks. Goldsky explicitly notes two event rows can represent the two perspectives of one match. | Best managed ingestion candidate. Preserve its event IDs/order and reuse Passivbot's economic-fill deduplication rather than assuming one row equals one trade. |
| 3 | [Dune `polymarket_polygon.market_trades`](https://docs.dune.com/data-catalog/curated/prediction-markets/polymarket/market_trades) or [Allium prediction tables](https://docs.allium.so/historical-data/predictions/schemas) | Both expose block, transaction, and event/log ordering keys with normalized market and trade fields. They reduce RPC operations but introduce transformation/version and export-access dependencies. | Strong bulk backfill or independent validation source. Reconcile sampled rows to raw logs and pin dataset/version semantics. |
| 4 | [Public Polymarket Data API trades](https://docs.polymarket.com/api-reference/core/get-trades-for-a-user-or-markets) | Public, market-filterable, up to 10,000 rows per page/window, and time-window pagination can go deeper. Rows have whole-second timestamps and transaction hashes but no transaction/log index. `takerOnly` defaults to true. | Useful discovery, volume, and transaction-hash cross-check. Insufficient as authoritative EMA-close input when multiple fills share a second. |
| 5 | Polymarket CLOB account trades / price history | Account trades require API authentication; price history is sampled and lacks actual fill volume. | Not a public market-history backfill and not valid for the initial fill-derived candle contract. |

The preferred implementation remains raw Polygon logs behind a transport interface. Goldsky,
Dune, or Allium can later implement that interface without changing the canonical archive,
one-second candle builder, or backtester.

## Public Polymarket Evidence

All probes below were public and unauthenticated.

### Ordered Polygon history

The official standard-market CTF Exchange v1 and v2 contracts expose different `OrderFilled`
layouts. The archive reader now queries both event topics over complete block ranges, decodes their
indexed maker/taker fields and packed data, filters the requested two outcome token IDs, removes the
exchange-facing taker aggregate, and preserves `(block, transaction, log)` order.

Settlement is imported from the official Conditional Tokens contract's condition-indexed
`ConditionResolution` event. Its payout vector supports ordinary YES/NO and rare 50/50 outcomes
and is archived separately from fills. Gamma terminal prices remain discovery metadata only.

A public dRPC smoke over `2026-07-24T18:48:15Z` through `18:48:25Z`:

- proved blocks 90,809,265 through 90,809,271 as the complete time interval;
- decoded 304 standard-market `OrderFilled` logs;
- found two logs for Gamma market 3,045,251;
- normalized one economic fill after removing the taker aggregate: 10.86 shares of the second
  outcome bought at 0.999, transaction
  `0x2c8e09504d2e47e210f0f1de846ebeff71aec5d86ac0d00a122dc67c7f02d3f4`.

The same older-block request against Allnodes PublicNode failed closed because anonymous archive
requests require a personal token. Public endpoints are therefore useful sources but not an
availability guarantee; sustained backfills should accept a configurable archival RPC or indexed
raw-log transport while retaining the same decoder and coverage proof.

### Ordered-fill strategy samples

Two five-minute intervals were backfilled from complete Polygon block ranges and evaluated with
actual-fill-derived one-second signal and execution candles. Settlement at each sample end is
synthetic, so these are microstructure and inventory-path checks, not full-contract profitability
backtests.

The active tennis market Gamma 3,045,251 sample covered blocks 90,807,935 through 90,808,134:

- 13,580 standard-market logs decoded, 472 target-market logs, 282 economic maker fills;
- 96 positive-volume signal seconds and 195 carried-close zero-volume seconds after the first
  available close; canonical YES moved from roughly 0.40 to 0.15 with a 0.11 to 0.40 range;
- default pair-first accumulation made one 5-share YES fill and no complement: worst settlement
  PnL -1.50, best +3.50, settlement sensitivity 5.00;
- inventory-aware and YES-only modes each ended with 20 unpaired YES shares and 20.00 of settlement
  sensitivity, with a -4.895 worst settlement case;
- narrowing the quote and locked-edge parameters from 0.01 to 0.001 still did not complete the
  pair. After a YES fill during a falling market, a profitable NO complement requires a sufficient
  YES rebound; quote distance alone cannot guarantee it.

Adding passive residual liquidation before the synthetic close separated pair completion from
inventory flatness:

| Risk-only window | Accumulate fills | Pair completion | Final residual | Worst / best PnL |
| --- | ---: | ---: | ---: | ---: |
| 30 seconds | 1 | 0% | 5 YES | -1.500 / +3.500 |
| 60 seconds | 2 | 0% | 0 | -0.745 / -0.745 |
| 120 seconds | 2 | 0% | 0 | -0.650 / -0.650 |

The second fill in the 60- and 120-second runs was a sale of the original YES inventory, not a NO
purchase. It removed settlement sensitivity but crystallized a trading loss. The evaluator now
defines pair completion from cumulative complementary buys, so a buy-then-sell round trip can no
longer report 100% pair completion merely because final inventory is flat. The window is therefore
a configurable tail-risk tradeoff, not an assumed profit improvement; this single sample does not
justify a recommended duration.

The exact daily-threshold-family market “Will the price of Bitcoin be above $54,000 on July 24?”
(Gamma 2,960,104) covered blocks 90,772,935 through 90,773,134:

- 10,801 standard-market logs decoded, 20 target-market logs, 10 economic maker fills;
- all modes made zero fills despite continuously quoting after warmup: the interval was too sparse
  to cross the passive orders under the strict actual-fill rule.

Current decision: `ema_anchor_outcome` is mechanically suitable as a passive quote generator, and
pair-first accumulation sharply bounds repeated same-side inventory compared with the other modes.
It has not demonstrated settlement-independent spread capture. Pair completion is a path-dependent
inventory risk, not an automatic arbitrage, and sparse daily threshold markets may generate many
orders with no fills. Passive pre-close liquidation can bound the final residual, but it may simply
convert settlement uncertainty into a certain trading loss. Do not start authenticated order
mutation until multiple full-lifecycle contracts show acceptable pair-completion, worst-settlement,
fee, residual-duration, and capital-utilization metrics.

## Current HIP-4 Dry-Cycle Evidence

An authorized read-only dry cycle on a current daily BTC price-binary outcome used public trade
data plus unauthenticated address-indexed account-state reads and made no exchange mutation:

- the account remained funded with positive conservatively available collateral;
- zero selected-market YES/NO inventory, open orders, recent fills, or unknown outcome state;
- no public fill arrived during the bounded collection window;
- the integrated live cycle returned `planning_available=false`,
  `planning_unavailable_reason=no_public_fill`, zero creates, and an empty managed cancellation
  set.

Sparse-market silence is now a successful fail-closed decision rather than a traceback. In a live
mutation-enabled cycle, the same path targets an empty Passivbot-managed order set for that market,
while unmanaged orders remain untouched. It still does not create a zero-volume signal candle
without a prior actual fill.

Two markets from the exact daily threshold family were silent during bounded websocket probes:

- Gamma market `2972731`, “Bitcoin above $64,000 on July 25?”: no fill in 90 seconds.
- Gamma market `2972728`, “Bitcoin above $62,000 on July 25?”: no fill in 90 seconds.

The collector correctly recorded no verified candle coverage for these attempts. Silence is not
converted into zero-volume bars until a connected session has a prior actual fill.

A liquid binary market (`3045251`) provided a five-minute mechanics sample:

| Measure | Result |
|---|---:|
| Verified signal seconds | 300 |
| Covered actual-fill records | 123 |
| Signal seconds with one or more fills | 101 |
| Verified zero-volume seconds | 199 |
| Covered execution candles across native books | 110 |
| Canonical sample range | 0.06 to 0.25 |

The sample is not evidence of profitability and is not a full-contract backtest. It is a bounded
test of websocket chronology, YES/NO canonicalization, zero-second construction, native-book fill
eligibility, and settlement-scenario accounting.

A later 10-second capture of the same market archived two covered fills and six full public book
snapshots on separate websocket streams. Eight no-trade seconds were derived only from the fill
stream. The book snapshots did not alter OHLCV. The market's live tick had moved from `0.01` to
`0.001`, confirming that Polymarket tick changes must be archived and replayed independently.

The later ordered Polygon backfill contains 282 economic fills over the same five-minute market
window. Replaying it after the pair-first and fee-incidence changes, with 5/20-second EMAs, a 0.01
probability-point offset, 5-share clips, zero maker fees, and synthetic settlement at the sample
end, produced:

| Mode | Fills | Final YES | Final NO | Pair completion | Worst net PnL | Best net PnL |
|---|---:|---:|---:|---:|---:|---:|
| `accumulate_pairs` | 1 | 5 | 0 | 0.000 | -1.50 | 3.50 |
| `inventory_aware` | 8 | 0 | 0 | 0.333 | -1.22 | -1.22 |
| `yes_only` | 6 | 0 | 0 | 0.000 | -1.12 | -1.12 |

The paired mode avoided realized churn but never acquired a complement, leaving one clip of full
settlement sensitivity. Both round-trip modes flattened before synthetic settlement but realized a
loss; their flat ending inventory does not imply pair completion. This supports a hard residual cap
of approximately one clip for pair accumulation and keeps all three modes unproven for positive
expectancy.

## Gates Before A HIP-4 Mutation Test

- Collect ordered, verified HIP-4 windows long enough to compare modes across multiple contracts
  and both settlement outcomes.
- Demonstrate pair completion, residual duration, and worst-case settlement behavior under
  trending and mean-reverting paths.
- Require the archived portfolio report—not a hand-calculated sample—to expose cumulative
  complementary buys, pair completion, peak and time-weighted residual inventory, total
  inventory-time, shared-capital utilization, and worst-case settlement equity.
- Confirm current market-specific order minimums, quote asset, lifecycle, and fill-level fee
  behavior immediately before the test. Reconfirm the observed zero trading fees and establish
  settlement fee behavior rather than inferring either from generic spot tiers.
- Keep mutations opt-in and bounded to explicit symbols, quantities, order count, price distance,
  and maximum live duration.
- Reconcile collateral, side inventory, orders, and fills before and after every mutation.

The bounded live mutation should be a final execution-path test, not the first source of strategy
evidence.
