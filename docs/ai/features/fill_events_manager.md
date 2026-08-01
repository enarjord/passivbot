# Fill Events Manager

## Contract

1. Build a deduplicated fill-event stream per exchange/account.
2. Preserve source data needed for realized PnL reconstruction.
3. Keep fetch behavior explicit and observable during investigations.
4. Canonical event accounting uses `pnl_contract = gross_pnl_quote_fee_best_effort_v2`:
   `pnl` is gross price PnL excluding fees, `fee_paid` is signed quote-currency
   balance impact (paid fees negative, rebates positive), and net realized PnL
   is derived as `pnl + fee_paid`.
5. Fee resolution is best-effort but observable: reported quote fee, reported
   non-quote fee converted by a fresh ticker, reported fee rate, then
   `live.fee_pct_fallback`. Every fill is sanity-checked by fee/notional ratio
   against `live.fee_pct_sanity_abs_max`; outliers use the fallback percentage.
6. Do not mix legacy/missing-contract cache rows with current rows. Repair or
   rebuild legacy fill-event caches before using trading-critical accounting.
7. Newly discovered fills may carry immutable `provenance` with attribution
   `first_ingested_by_runtime`. This identifies the Passivbot runtime that first
   persisted the fill locally; it does not claim that runtime created the order
   or caused the exchange fill. Refresh and deduplication preserve an existing
   provenance record, including the absence of provenance on legacy cache rows.
   Historical rows are never retroactively attributed.
8. `last_refresh_ms` records a completed exchange fetch, not local cache loading,
   normalization, or doctor repair. Preserving that distinction ensures the first
   incremental refresh after restart covers fills which occurred while the bot was
   offline.
9. A position whose latest fill identity or reconstructed after-state does not
   match the authoritative exchange position remains nontradable. Live orchestration
   retries from the position/fill anchor with bounded in-memory backoff. Direction,
   quantity, and price alone do not prove a flat-to-position transition when truncated
   history has polluted reconstructed `psize`/`pprice`. An explicit exchange position
   opening timestamp may prove a zero-state boundary only for a singleton opening-fill
   cohort: after a successful post-snapshot fill refresh, require the exact identified
   fill timestamp to equal both the opening timestamp and the authoritative latest
   position-update timestamp, then require its zero-state replay to match the
   authoritative position. Multi-fill cohorts and later position updates remain pending
   because final size/VWAP equality cannot prove omitted round trips are absent. Generic
   and update-only position timestamps do not establish this boundary. Successful use
   emits a warning and a structured fallback diagnostic only after every confirmation
   predicate, including the required post-snapshot fill generation, has cleared.
   Ordinary recorded after-state price comparison accepts at most one effective
   executable price tick because exchanges may round or truncate a sub-tick reconstructed
   VWAP. The effective tick is connector-aware, including Hyperliquid's decimal and
   significant-digit price ladder. It is derived from both compared prices so the
   narrower, lower-side interval governs discrepancies that cross a power-of-ten
   boundary; the inclusive boundary admits only floating-point representation slack.
   Quantity and fill identity remain strict. The explicit position-opening replay retains
   the ordinary half-tick comparison because it has no reconstructed VWAP rounding
   evidence. Acceptance outside that comparison but within one full tick emits one
   warning and `fill.position_price_tolerance_used` with bounded numeric discrepancy
   context after confirmation clears; larger discrepancies remain pending.
10. A degraded synthetic PnL row inside the configured live risk lookback is not
    authoritative merely because cache metadata proves time coverage. Refetch bounded
    windows around each such row, preserving the pre-repair incremental checkpoint and
    processing at most four independently bounded execution ranges per authoritative
    cycle. Rotate remaining events across backed-off cycles so one unresolved row cannot
    starve the others. Connectors whose authoritative PnL endpoint uses a timestamp
    different from execution time must search that timestamp independently in bounded,
    advancing windows while retaining the narrow execution lookup. Preserve ordinary
    recent-fill overlap for routine ingestion. When an enabled PnL consumer requires
    authoritative history, defer that planning without consuming restart budget until
    every in-lookback row is authoritatively replaced. Report every successful
    replacement before attempting later fallible fetches. Uptime health PnL counts
    authoritative net realized PnL only: pending and synthetic values remain visible in
    fill diagnostics but do not enter the health counter, and later enrichment adds the
    full authoritative amount without retaining fill-identity accounting state.
    PnL- or fee-only enrichment changes local accounting evidence and does not request
    another account-wide confirmation. New source identities and structural fill changes
    still confirm account surfaces because they may represent a new exchange-state transition.
    Structured `cycle.degraded` diagnostics preserve bounded `pending_pnl_count` and
    `degraded_pnl_count` fields through the centralized payload sanitizer.
11. `FillEventsManager` owns the canonical fill-history coverage verdict used by
    refresh, staged readiness, HSL replay, and realized-PnL consumers. Orchestration
    may choose retry timing or whether a proven-incomplete history is explicitly
    allowed, but it must not reinterpret cache metadata or known gaps. Metadata
    claiming cached rows when no rows loaded, and malformed known-gap bounds, are
    contradictory cache evidence: they are unavailable rather than proof of
    coverage. A window with no fills remains valid when zero oldest/newest metadata
    and `covered_start_ms` prove that empty result.
12. Live coverage requirements follow the enabled consumer. Realized-PnL risk
    features require the configured PnL lookback and authoritative PnL quality.
    Entry cooldown without a PnL consumer requires structural fill coverage only
    across its maximum configured cooldown horizon. Trailing reconstruction retains
    its symbol/position-side confirmation and bounded recovery. With no historical
    consumer enabled, routine ingestion starts from a bounded recent fetch rather
    than proving an unrelated PnL window.

## Runtime Provenance

The optional fill provenance record contains the runtime run id, Passivbot
version, Python Git commit and tracked-dirty state, canonical config hash, Rust
crate version, embedded Rust source fingerprint, loaded extension artifact hash,
runtime start timestamp, and the local first-ingestion timestamp. It contains
hashes rather than raw config, paths, commands, API payloads, or credentials.

The runtime identity proves which local runtime first ingested a fill after this
contract was introduced. Determining which historical runtime submitted an
order remains a separate attribution exercise using client-order identifiers,
logs, runtime windows, and immutable manifests.

## Exchange Endpoint Map (Quick Lookup)

| Exchange | Primary Fills Source | PnL/Close Source |
|----------|----------------------|------------------|
| Binance | `fetch_my_trades` | income history |
| Bybit | `fetch_my_trades` | closed-pnl (explicit time windows + cursor pagination) |
| Bitget | `fetch_my_trades` | embedded in trade payload |
| Hyperliquid | fill events | embedded |
| OKX | `fetch_my_trades` | positions history |
| KuCoin | trades + positions history | positions history |
| Gate.io | `fetch_my_trades` | embedded |
| WEEX | `fetch_my_trades` in seven-day windows | embedded `realizedPnl` |

## Non-Obvious Details

1. Exchanges split fill/PnL data across different endpoints.
2. Bybit closed-PnL history is fetched in contiguous windows shorter than the
   endpoint's seven-day maximum. Every window is cursor-paginated to exhaustion
   before moving to the next older window; sparse pages do not determine window
   boundaries.
3. Historical retention limits can make old PnL records unavailable.
4. WEEX trade-detail queries are limited to 100 rows and seven days per request,
   with up to 365 days of retention; its client order id may require an order-detail lookup.
   Full responses are recursively split into disjoint time windows because the endpoint does not
   guarantee row ordering or expose a stable cursor. Saturation within one millisecond is unavailable
   rather than silently treated as complete.
5. Old synthetic rows remain outside ordinary recent-fill overlap to avoid repeatedly
   widening every routine refresh. Risk-blocking degraded rows use a separate bounded
   repair path when an enabled HSL, auto-unstuck, or realized-loss consumer requires
   authoritative PnL. With all such consumers disabled, covered structural fill history
   remains ready while degraded rows stay visible for later repair. Repair-only calls do
   not advance `last_refresh_ms`; the subsequent
   ordinary recent refresh must still cover downtime from the prior successful
   checkpoint. Bybit keeps the execution-time range narrow while rotating a separate
   closed-PnL `updatedTime` range toward the present; each auxiliary range spans at most
   one day.
6. KuCoin position-history PnL is authoritative for a completed position cycle. Overlapping trade
   refreshes may return the same close as pending again; they must preserve an already reconciled
   cycle value. Reapplying an unchanged cycle observation is a no-op, while a changed authoritative
   total is redistributed across that lifecycle and persisted.
7. Fills sharing one millisecond carry no execution order in exchange responses or caches, yet
   position reconstruction replays them in list order. When the exchange reports the position size
   preceding each fill (Hyperliquid `startPosition`), retain each execution boundary and reorder an
   unambiguous cohort along that chain before annotation. Expand older coalesced Hyperliquid cache
   rows back into their raw components, and propagate a basis recovered from a reduction through
   chained additions in the same nonzero timestamp cohort only when the raw close component
   explicitly reports PnL. A normalized default of zero does not prove zero realized PnL. A matching
   position size at a later timestamp does not prove continuity across a potentially incomplete
   history window. If the chain is ambiguous, keep the existing order and do not choose a different
   anchor from mutable exchange position state. Hyperliquid's incremental overlap counts timestamp
   cohorts rather than individual executions and uses the earlier of that cohort anchor and the
   time-based refresh checkpoint. A same-millisecond burst or an older cohort therefore cannot be
   clamped out of the recent-fill window and strand a late-arriving component. Expand a legacy
   coalesced row only when its composite ID, canonical `source_ids`, component identities, and
   cohort fields match, position-chain data is finite, and weighted price, signed effective
   quantity, gross PnL, and signed fees reconcile to the aggregate. An unreconciled aggregate is a
   cache-contract failure: quarantine the cache and rebuild the configured lookback before
   accepting individually fetched components.

## Failure Semantics And Risks

1. Missing records from pagination assumptions.
2. Duplicate events from multi-source merge logic.
3. PnL mismatch between trade feed and positions-history feed.

Exchange fetch methods propagate endpoint failures. The manager or caller may repair, retry,
quarantine, rebuild, or defer according to `../error_contract.md`; it must not attach neutral PnL
merely because an auxiliary endpoint failed.

Unproven required coverage is a controlled live-planning deferral. The execution loop owns its
bounded, reason-aware retry cadence, and persistent coverage gaps do not consume the generic
process-restart budget. A change between coverage and PnL block reasons restarts that reason's
backoff at its configured base. Already-latched HSL RED supervisors continue protective management
without fills while coverage repair proceeds. Manager-owned known-gap state remains evidence about
coverage, not a second orchestration timer.

## Validation

1. Deduplication correctness.
2. Pagination completeness for high-activity windows.
3. PnL attachment behavior when auxiliary endpoints fail.
4. Provenance round-trip, preservation during refresh/deduplication, and legacy
   rows remaining unattributed.
5. Old degraded synthetic PnL is refetched in bounded windows when an enabled
   authoritative-PnL consumer requires it, authoritative replacement is persisted, and
   unresolved rows defer those consumers without restarts. With every PnL consumer
   disabled, unresolved PnL does not block covered structural fill history.
6. Coverage verdicts fail closed for contradictory metadata and malformed gaps,
   while confirmed-legitimate gaps and proven empty windows retain their explicit
   semantics.

## Key Code

- `src/fill_events_manager.py`
- `exchange_integrations.md`
- `../case_studies/debugging.md`
