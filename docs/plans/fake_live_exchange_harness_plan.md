# Fake Live Exchange Harness Plan

This document is a handoff plan for building a dry-run live-bot harness using a fake exchange adapter.

## Goal

Run the normal live bot end-to-end with minimal or no logic branching in the hot live code path.

Desired properties:

1. The live bot should use the same order generation, HSL, reconciliation, RED supervisor, and restart behavior as real live trading.
2. Exchange I/O should be replaced by deterministic fake state and scripted or replay-driven market data.
3. The harness should be agent-friendly:
   - runnable from CLI
   - rich logs
   - deterministic outputs
   - saved state/artifacts for postmortem analysis
4. The fake harness should be generally reusable, not HSL-only, even if HSL is the first priority.

## Recommended Architecture

### Core shape

Implement:

1. `src/exchanges/fake.py`
2. `src/tools/run_fake_live.py`

Keep fake-specific logic isolated to those modules plus the smallest possible exchange bootstrap changes.

### Exchange integration strategy

Preferred approach:

1. Add exchange name `"fake"` to the existing exchange factory in `src/passivbot.py`
2. Implement `FakeBot` in `src/exchanges/fake.py`
3. `FakeBot` should ideally subclass `CCXTBot` and provide a fake CCXT-like client (`self.cca`) underneath

Reason:

1. `Passivbot` already expects an exchange adapter object with `create_ccxt_sessions()`, `fetch_balance()`, `fetch_positions()`, `fetch_open_orders()`, `fetch_tickers()`, `execute_order()`, etc.
2. `CCXTBot` already implements most of the template-method surface Passivbot relies on.
3. If the fake client exposes the CCXT-like async methods those template methods call, the live loop can remain essentially unchanged.

Alternative:

- Subclass `Passivbot` directly instead of `CCXTBot`

This is less desirable because it duplicates more exchange-template behavior.

## Minimal Intrusion Principle

Do not add widespread `if dry_run` or `if fake_exchange` branches in:

1. `src/passivbot.py`
2. Rust orchestrator/backtest code
3. general live execution logic

Allowed core changes:

1. Exchange factory recognizing `"fake"`
2. Possibly allowing startup to skip external `api-keys.json` dependency for fake users if needed
3. Small `fill_events_manager` registration for a fake fetcher

Everything else should live in the fake adapter/runner.

## Fake Exchange Model

### Fake client

Implement a fake async client object with CCXT-like methods:

1. `load_markets(reload=True)`
2. `fetch_balance()`
3. `fetch_positions()`
4. `fetch_open_orders(symbol=None)`
5. `fetch_tickers()`
6. `fetch_ticker(symbol)`
7. `fetch_ohlcv(symbol, timeframe="1m", since=None, limit=None)`
8. `fetch_my_trades(symbol=None, since=None, limit=None, params=None)`
9. `create_order(symbol, type, side, amount, price=None, params=None)`
10. `cancel_order(order_id, symbol=None, params=None)`
11. `fetch_time()`
12. `set_position_mode(...)`
13. `set_leverage(...)`
14. `set_margin_mode(...)`
15. `close()`

Properties/fields expected by the rest of the code:

1. `id = "fake"`
2. `has = {...}`
3. `options = {}`
4. `urls = {}`

### Internal state

Maintain one authoritative in-memory state object:

1. `now_ms`
2. `balance`
3. `positions`
4. `open_orders`
5. `fills`
6. `markets`
7. `prices`
8. optional scenario metadata

This state should be serializable to JSON.

### Market metadata

Return CCXT-like `markets` dict entries with enough fields for:

1. `load_markets()`
2. `set_market_specific_settings()`
3. `filter_markets()`
4. CandlestickManager

Include at least:

1. `id`
2. `symbol`
3. `base`
4. `quote`
5. `swap = true`
6. `active = true`
7. `precision.amount`
8. `precision.price`
9. `limits.amount.min`
10. `limits.cost.min`
11. `contractSize`

## Scenario Model

### Round 1 support

Support two sources of fake market behavior:

1. scripted timeline
2. existing 1m candle data replay

Both should be available from the start if practical, because replay from existing 1m data is valuable for:

1. reproducing known historical HSL events
2. modifying known periods to create contrived scenarios
3. running realistic dry live loops around past market events

### Suggested scenario file shape

Use a JSON or HJSON scenario file.

Suggested top-level fields:

1. `exchange`
2. `start_time`
3. `tick_interval_seconds`
4. `account`
5. `symbols`
6. `timeline`
7. `fill_model`
8. `assertions`

Example:

```json
{
  "exchange": "fake",
  "start_time": "2026-01-01T00:00:00Z",
  "tick_interval_seconds": 5,
  "account": {
    "balance": 10000.0
  },
  "symbols": {
    "XMR/USDT:USDT": {
      "price_step": 0.1,
      "qty_step": 0.01,
      "min_qty": 0.01,
      "min_cost": 5.0,
      "maker_fee": 0.0002,
      "taker_fee": 0.00055
    }
  },
  "timeline": [
    {"t": 0, "prices": {"XMR/USDT:USDT": 150.0}},
    {"t": 1, "prices": {"XMR/USDT:USDT": 143.0}}
  ]
}
```

### Replay mode

Support replay of real 1m candles from:

1. cached OHLCV files
2. or prebuilt scenario candle payloads

The fake exchange can use those candles as the source for:

1. `fetch_ohlcv`
2. current ticker/price
3. limit-fill crossing logic

This makes it easy to:

1. run dry live around historical HSL-triggering windows
2. edit/warp known candle sequences into sharper stress tests

## Fill / Order Model

### Round 1 recommended behavior

Keep the execution model simple but explicit:

1. market orders fill immediately at the current price with taker fee
2. limit orders rest until the current price crosses them
3. fill events update:
   - open orders
   - positions
   - realized PnL
   - balances
   - fill history

This is enough for:

1. normal entry/close behavior
2. HSL panic closes
3. RED flattening
4. cooldown/restart logic

### What not to build in round 1

Do not build yet:

1. websocket emulation
2. orderbook depth simulation
3. funding simulation
4. partial-fill microstructure realism
5. multi-exchange fake routing

## Time Model

The fake exchange should own a scenario clock.

The runner should advance time explicitly:

1. update current price/candle
2. process resting limit fills
3. expose the latest account/order state
4. let the live bot run one execution cycle

Avoid relying on wall clock for scenario progression.

## Runner

Implement `src/tools/run_fake_live.py`.

Responsibilities:

1. load normal Passivbot config
2. load fake scenario
3. instantiate the normal bot using `exchange: "fake"`
4. start the bot in a bounded dry mode
5. advance fake time / market state deterministically
6. collect logs and artifacts
7. exit with pass/fail assertions

The runner should ideally support:

```bash
python3 src/tools/run_fake_live.py configs/hsl.json scenarios/hsl_xmr_red.hjson --debug
```

### Recommended bot-driving approach

Leverage existing `debug_mode` behavior in `Passivbot`:

1. `start_bot()` skips the endless execution loop when `debug_mode=True`
2. `run_execution_loop()` returns after one full loop when `debug_mode=True`

So the runner can:

1. set `bot.debug_mode = True`
2. call `await bot.start_bot()`
3. advance scenario
4. call `await bot.run_execution_loop()` repeatedly

This allows reuse of the real live loop logic without a permanent loop.

## Fill Events / PnL History

### Important integration point

`FillEventsManager` currently uses exchange-specific fetcher selection in `src/fill_events_manager.py`.

Add fake support cleanly:

1. implement a small `FakeFetcher(BaseFetcher)`
2. register `"fake"` in `_build_fetcher_for_bot`

Round 1 behavior can be simple:

1. return fake fills from the fake client’s internal fill ledger
2. filter by `since_ms` / `until_ms`

This keeps:

1. startup PnL reconstruction
2. HSL history replay
3. realized-loss gate inputs

within the normal code path.

## Startup Dependencies To Satisfy

The fake harness must satisfy these live startup assumptions:

1. `load_user_info(live.user)` currently reads `api-keys.json`
2. `load_markets(exchange, cc=self.cca)` is called during bot startup
3. CandlestickManager uses `self.cca.fetch_ohlcv(...)`
4. `format_approved_ignored_coins(...)` and symbol normalization expect standard CCXT-style symbols
5. `FillEventsManager` expects a registered fetcher for the exchange

### Recommended handling of `live.user`

Keep the existing pattern:

1. add a fake user entry in `api-keys.json`
2. set `"exchange": "fake"`
3. allow any fake-specific fields to pass through in `user_info`

This is less intrusive than changing `load_user_info()` just for fake mode.

## First Test Scenarios

Build these first:

1. `hsl_long_red_restart`
   - long side hits RED
   - panic closes
   - cooldown
   - restart
   - short unaffected

2. `hsl_long_terminal_no_restart`
   - long side exceeds `hsl_no_restart_drawdown_threshold`
   - long remains halted
   - short can continue

3. `hsl_long_short_independent`
   - both `psides` enabled with different thresholds
   - long enters orange/red first
   - short stays green longer

These directly validate the recent HSL `pside` split work.

## Artifacts

Have the runner save:

1. `fake_exchange_state.json`
2. `fills.json`
3. `positions.json`
4. `hsl_trace.json`
5. normal log output

Optional:

1. snapshots after every scenario step

This is useful for agents and humans investigating failures.

## Assertions

Support scenario assertions such as:

1. expected final positions
2. expected halted state per `pside`
3. expected hard-stop trigger/restart counts
4. expected cooldown behavior
5. expected log fragments

## Phase Plan

### Phase 1

1. add `src/exchanges/fake.py`
2. add minimal fake CCXT-like client
3. add factory support for `"fake"`
4. add `FakeFetcher`
5. add `src/tools/run_fake_live.py`
6. add one minimal HSL scenario

### Phase 2

1. add replay-from-1m-candles support
2. add more realistic limit-fill timing
3. add restart/reconstruction-specific scenarios

### Phase 3

1. richer assertions
2. better artifact viewers
3. optional partial-fill and latency simulation

## Practical Notes For The Next Session

### Current codebase observations

1. `src/passivbot.py:setup_bot()` is the main exchange factory.
2. `Passivbot.__init__()` loads `live.user` from `api-keys.json`, sets `self.exchange`, then calls `create_ccxt_sessions()`.
3. `start_bot()` already supports a useful bounded mode:
   - if `debug_mode=True`, it skips entering the permanent execution loop
4. `run_execution_loop()` already supports one-shot behavior:
   - if `debug_mode=True`, it returns after one full execution cycle
5. `load_markets()` can reuse an existing exchange client if `cc=` is supplied, so the fake client should implement `load_markets()`.
6. CandlestickManager expects the exchange client to expose `fetch_ohlcv()` and `id`.

### Design decisions already made

1. Keep fake logic isolated to separate modules/scripts.
2. Do not litter the hot live code with `if dry_mode`.
3. One fake exchange adapter is enough for the first round.
4. Scripted scenarios and replay-from-existing-1m data are both worthwhile.
5. Websocket emulation is explicitly deferred.

### Recommended validation loop

For another agent:

1. implement fake exchange adapter + runner
2. run a single deterministic HSL scenario in debug mode
3. read logs and saved artifacts
4. fix live-path parity bugs
5. repeat

That loop is the main reason this harness is worth building.
