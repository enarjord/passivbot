# Binance Portfolio Margin Support — Design Plan

## Status

Implemented: `src/exchanges/binance_portfolio.py` (`BinancePortfolioBot(BinanceBot)`), wired
into `setup_bot()` in `src/passivbot.py` and into the `ccxt_contracts` fixture-capture tool.
Existing tests pass unmodified; a new test
(`tests/test_setup_bot_fallback.py::test_setup_bot_binance_portfolio_margin_account_type`)
covers the new dispatch branch. See `docs/exchanges/binance_portfolio_margin.md` for the
user-facing guide.

The real diff needed turned out much smaller than originally scoped here, because **CCXT's
installed `binance.py` already has built-in Portfolio Margin routing for most unified
methods**, confirmed by reading the installed package source directly — see "Why this
doesn't already work" and "Proposed Shape" below.

Validated against a funded Portfolio Margin sub-account: balance reads, hedge-mode and
leverage setting, order placement and cancellation (both market and limit, UM and margin
legs), and a full `setup_bot()`-based boot through a real `BinancePortfolioBot` instance.
See the class docstring in `src/exchanges/binance_portfolio.py` for the itemized account and
the resolved Open Questions below for specifics.

## Purpose

Add support for running a Binance account in Portfolio Margin (PM) mode, so that the
existing grid-trading engine can run on a PM sub-account unchanged — including
alongside other PM-native activity on the same account (for example a separate
spot+perp strategy outside Passivbot's own scope) that benefits from PM's unified,
cross-collateral risk pool. Binance PM is account-wide — a sub-account is either fully
on PM or fully on regular margin, there is no per-position mix — so this requires the
grid engine itself to work under PM, not a connector for a specific second strategy.

## Why this doesn't already work

`src/exchanges/ccxt_bot.py` gets its exchange connection via
`getattr(ccxt_async, ccxt_id)` — for Binance this resolves to `ccxt.binanceusdm`
(`to_ccxt_exchange_id` in `src/utils.py` hardcodes `"binance" -> "binanceusdm"`).
Portfolio Margin is a structurally separate account model on Binance's side (unified
risk across spot + margin + UM futures + CM futures + options), reachable through a
parallel `papi*` endpoint family — `setup_bot()` never set an account up to use it, and
nothing in `BinanceBot` asked for it.

**Correction from the original version of this doc:** this is *not* because ccxt lacks
PM support. Confirmed directly from the installed `ccxt/binance.py` source:
`binanceusdm` is `class binanceusdm(binance, ImplicitAPI)` — a thin subclass that
inherits `binance`'s methods unmodified, and `binance.py` already has PM routing built
into most of its unified methods (`fetch_balance`, `fetch_positions_risk`,
`create_order`, `cancel_order`, `cancel_all_orders`, `fetch_open_orders`/`fetch_orders`,
`fetch_my_trades`, `set_leverage`, `set_position_mode`) gated behind a single
`options["portfolioMargin"]` flag (`handle_option_and_params_2(params, <method>, "papi",
"portfolioMargin", False)` in each). Once set, each of those methods branches
internally to the matching `papi*` endpoint (`papiGetBalance`, `papiGetUmPositionRisk`,
`papiPostUmOrder`, `papiPostMarginOrder`, `papiPostUmLeverage`,
`papiPostUmPositionSideDual`, ...) with no other call-site changes required. ccxt's own
response-shape doc comments show the papi position-risk payload uses the **same field
names** as the regular endpoint (`symbol`, `positionSide`, `positionAmt`, `entryPrice`,
`markPrice`, `unRealizedProfit`, `leverage`, ...), so `BinanceBot`'s existing
`_normalize_positions` / `_build_order_params` / order-fill parsing are reused
unchanged in `BinancePortfolioBot`.

Two things ccxt does *not* bridge:
- `set_margin_mode` has no `papi` branch at all — PM risk is unified account-wide, so
  there's no per-symbol isolated/cross toggle to set. Confirmed by reading the method
  body directly; it unconditionally calls the regular `fapiPrivatePostMarginType`.
- `_get_balance`: regular futures reads a single `info.totalCrossWalletBalance` field.
  PM's `papiGetBalance` returns one entry *per collateral asset* instead (confirmed via
  ccxt's `parse_balance_custom`), so this hook needs real PM-aware logic — this is the
  one place the original scoping in this doc was right that real work is needed.

## Coin-universe partition: avoiding collisions with other account activity

If other activity on the same PM account trades specific coins outside Passivbot
(for example a small, fixed set of majors), that overlap is avoided entirely through
Passivbot's existing `live.ignored_coins.long` config field — no new code, an
already-supported mechanism. Passivbot's grid keeps running on every other approved
coin untouched.

This is a structural guarantee, not a trust assumption: the grid engine never holds a
position in a coin excluded this way, by config. The only thing shared on the account
is the PM margin pool itself — which is the entire point of this plan — not any
individual order or position.

## Non-Goals

- No change to trading logic, risk, unstuck, or HSL behavior. This is a connector-layer
  change only — same "Rust owns order behavior, Python owns exchange I/O" boundary as
  `exchange_connector_event_contract_plan.md`.
- No behavior change whatsoever for existing non-PM accounts on any exchange. PM support
  is additive and opt-in per config, not a modification to the default Binance path.
- v1 does not include any non-Passivbot strategy logic that might also run on a PM
  account. This plan is scoped to making the account connector work for Passivbot's
  own grid engine, not to any other strategy that could share the account.
- v1 does not include CM futures or options under PM — only UM futures (the grid
  engine's unchanged symbol universe) and margin/spot.
- v1 does not build custom PM websocket streaming. See Testing & Rollout below.

## Proposed Shape

Not a new hook-by-hook rewrite of `binance.py`. Instead, `BinancePortfolioBot(BinanceBot)`
in a new `src/exchanges/binance_portfolio.py`, dispatched from `setup_bot()` when the
account's `api-keys.json` entry has `"account_type": "portfolio_margin"` (an
account-level property, parallel to the existing `"exchange"` field — deliberately
*not* a new `live.*` schema field, since account type isn't a strategy parameter and
adding one would mean touching `config/schema.py` + `config/validate.py` +
`config/coerce.py` for something that belongs in the credentials file `load_user_info`
already passes through untouched).

| Hook | Regular futures (`BinanceBot`) | PM override (`BinancePortfolioBot`) |
|---|---|---|
| `create_ccxt_sessions()` | — | sets `self.cca.options["portfolioMargin"] = True` once; everything below rides on this single flag |
| `_do_fetch_positions()` | raw `fapiprivatev3_get_positionrisk()` call | `self.cca.fetch_positions_risk()` (unified; routes to `papiGetUmPositionRisk` via the flag) |
| `_normalize_positions()` | — | **inherited unchanged** — field names match (confirmed from ccxt's response-shape comments) |
| order creation / cancellation / `_build_order_params()` | fapi `create_order`/`cancel_order` | **inherited unchanged** — same flag routes `create_order`/`cancel_order`/`cancel_all_orders` to `papiPostUmOrder`/`papiDeleteUmOrder`/etc., same response shape |
| `fetch_open_orders` / `fetch_my_trades` / pnl fetching | fapi | **inherited unchanged** — same flag, same routing mechanism |
| `_do_fetch_balance()` / `_get_balance(fetched)` | unified `fetch_balance()`, reads `info.totalCrossWalletBalance` | **real override, not inherited.** CCXT's unified `fetch_balance()` folds PM balances into only the UM-futures sub-ledger, reading `0` against a real account with real, unallocated collateral — confirmed live (see Open Questions). `_do_fetch_balance()` calls `papiGetAccount()` directly instead; `_get_balance()` reads its `accountEquity` field, a single precomputed account-wide figure that already handles multi-asset collateral by construction. |
| `update_exchange_config_by_symbols()` | sets margin mode + leverage | **real override.** Drops the `set_margin_mode` call (no `papi` branch exists); keeps `set_leverage` (which does route via the same flag) |
| `update_exchange_config()` (hedge mode) | `set_position_mode(True)` | **inherited unchanged** — `set_position_mode` also routes via the same flag, confirmed from source |
| `can_watch_orders()` | CCXT `watchOrders` capability check | **forced `False`.** No ccxt.pro PM websocket class exists, and the regular `binanceusdm` watch stream is backed by the non-PM listen-key flow — would silently watch the wrong account state. Falls back to REST polling (first-class supported path, not a hack). |

Everything else in the grid engine (signal generation, order sizing, wallet-exposure
math, HSL/orchestrator) is unchanged — it already only talks to the hook layer, not
directly to ccxt or to Binance's REST surface.

## Testing & Rollout

Validation here was direct: run the overridden hooks against a real, funded PM
sub-account and confirm behavior (see Completed, below), rather than building offline
fixtures first. `src/ccxt_contracts.py` + `capture_ccxt_contracts` / `diff_ccxt_contracts`
(see `docs/ai/ccxt_upgrade_workflow.md`) is the right tool to add regression-test
fixtures from real PM response snapshots as a follow-up, same workflow this repo already
uses for ccxt version bumps.

**v1 explicitly skips custom PM websocket support** and relies on `can_watch_orders()`
returning `False` for PM accounts — already a first-class supported path (REST polling,
not a hack), per `watch_orders()`'s existing fallback. Reasoning: CCXT Pro's websocket
transport has a known real failure mode against Binance's private user-data stream on
the already-supported (non-PM) account type (see `docs/ai/` for exchange connector
precedent on this transport). Building new raw websocket handling for PM, which has no
CCXT Pro class to begin with, in the same change that introduces the account type
itself, stacks two unproven things on top of each other. Ship polling first; PM
streaming is a separate follow-up once the REST path has run live.

Completed:
1. Hook overrides, PM behind the explicit opt-in `account_type` field —
   `src/exchanges/binance_portfolio.py` + `setup_bot()`/`ccxt_contracts.py` wiring.
2. Real PM API response shapes (balance, positions, orders) captured against a funded
   PM sub-account, resolving the Open Questions below.
3. Live validation on the same account: balance reads, hedge-mode and leverage setting,
   order placement and cancellation (market and limit, both UM and margin legs), and a
   full `setup_bot()`-based boot through a real `BinancePortfolioBot` instance. The
   existing test suite remains green; no shared code changed in a way that affects
   `account_type != "portfolio_margin"` accounts.

Not yet done: offline fixture tests (`tests/ccxt_upgrade`-equivalent) built from capturing
real PM response shapes via `capture_contract_snapshot()` in `src/ccxt_contracts.py` —
worth adding as a follow-up for regression coverage, not required for this connector's
own correctness given the live validation above.

## Open Questions

Resolved by reading ccxt's own source rather than guessing (no longer open):

- ~~Does `papiGetUmPositionRisk`/order response schema match the regular endpoint?~~
  For positions: yes, confirmed field-name parity from ccxt's response-shape comments.
  For balance: no — PM's response shape genuinely differs (one entry per collateral
  asset), which is why `_get_balance` has a real override rather than being inherited.
- ~~Preferred shape for the opt-in flag?~~ `api-keys.json`'s per-user `account_type`
  field, read in `setup_bot()` — not a new `live.*` schema field. Account type is a
  property of the credentials, same as `exchange` already is; this avoids touching
  `config/schema.py`/`validate.py`/`coerce.py` for a non-strategy parameter.
- ~~Subclass vs. mode flag?~~ Subclass (`BinancePortfolioBot(BinanceBot)`), matching
  how `exchanges/` already handles every other exchange variant. Keeps the regular
  futures path's diff surface at zero.

Resolved with a real account (a small-capital PM sub-account was stood up
specifically for this — see Testing & Rollout below):

- ~~`load_leverage_brackets()` behavior for PM?~~ Confirmed directly via an isolated
  `reload=True` call: `papiGetUmLeverageBracket` returns real tier data (1064
  symbols, correct BTC bracket structure), composes cleanly with a following
  `fetch_positions_risk()`.
- ~~`_get_balance`'s accountEquity vs actualEquity?~~ Resolved with real floating
  PnL: opened a position, polled through a real uPNL swing, found both fields move
  together — **neither** excludes unrealized PnL (contrary to the original
  assumption). `accountEquity` is correct to keep (it matched
  `totalAvailableBalance` when flat); no code change needed. Multi-asset collateral
  itself (e.g. BTC posted as margin) still hasn't been tested against a real mixed
  account, but `accountEquity` is a single account-wide figure by construction, so
  this is expected to already be correct.
- ~~Does this execute against Binance's real `papi*` endpoints?~~ Yes — balance,
  positions, hedge-mode setting (both the one-way→hedge transition and the
  already-set idempotent case, which returns the same `-4059` code `BinanceBot`
  already handles), leverage setting, market orders, limit orders, and
  cancellation were all placed/confirmed for real. A genuine `BinancePortfolioBot`
  was also constructed via the real `setup_bot()` (not just a hook-level harness)
  against the real account, confirming the hook layer behaves identically either
  way — see `src/exchanges/binance_portfolio.py`'s docstring for the full,
  itemized account.

Still open, by design rather than oversight:

1. `execute_order()`/`execute_cancellation()`'s real HSL write-coordination gate
   (`src/live/hsl_live.py`'s `connector_write`) requires a full async bootstrap
   (candle warmup, staged balance/position/fill refresh, HSL owner init) to admit
   writes — confirmed this live, not guessed. This is deliberate, exchange-agnostic
   safety machinery identical for every exchange, not PM-specific, and it gates the
   exact papi calls already validated above — not a gap in this connector.
2. A real multi-asset-collateral account (BTC *and* USDT posted as margin
   simultaneously) hasn't been tested, as noted above.
