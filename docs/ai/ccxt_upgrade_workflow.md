# CCXT Upgrade Workflow

Use this workflow when changing the pinned `ccxt` version or when reviewing a branch that depends on a newer `ccxt`.

## Goal

Catch behavior drift from CCXT before it reaches live trading paths.

The main risk is not only hard import/runtime failure. The bigger risk is silent semantic drift:

1. `load_markets()` changes a field shape or sentinel value.
2. A balance/position/order/trade payload changes nesting or naming.
3. A `has[...]` capability flag changes.
4. Passivbot interprets the new CCXT output differently and changes live behavior.

## Principles

1. Keep live network access out of the default test suite.
2. Use live CCXT calls only to capture sanitized snapshots.
3. Run offline contract tests against those snapshots and against stable hand-built fixtures.
4. Treat meaningful snapshot diffs as review items, not as noise to ignore.

## Tools

Shared helper:

- `src/ccxt_contracts.py`

Live capture CLI:

- `python -m src.tools.capture_ccxt_contracts --help`

Snapshot diff CLI:

- `python -m src.tools.diff_ccxt_contracts --help`

Offline regression tests:

- `pytest tests/ccxt_upgrade`

## Safe Upgrade Procedure

1. Capture baseline snapshots on the currently pinned CCXT version.

   Example:

   ```bash
   ./venv/bin/python -m src.tools.capture_ccxt_contracts \
     --user bybit_user \
     --label bybit-old \
     --symbols BTC/USDT:USDT,ETH/USDT:USDT
   ```

2. Upgrade `ccxt` in a branch and reinstall dependencies.

3. Capture the same snapshots again on the candidate CCXT version.

   Example:

   ```bash
   ./venv/bin/python -m src.tools.capture_ccxt_contracts \
     --user bybit_user \
     --label bybit-new \
     --symbols BTC/USDT:USDT,ETH/USDT:USDT
   ```

4. Diff old vs new snapshots.

   ```bash
   ./venv/bin/python -m src.tools.diff_ccxt_contracts \
     artifacts/ccxt_contracts/bybit/bybit-old.json \
     artifacts/ccxt_contracts/bybit/bybit-new.json
   ```

5. Review the diff carefully. Do not assume changes are harmless.

6. Add or update offline tests for any newly discovered contract that matters to Passivbot.

7. Run the offline regression suite.

   ```bash
   ./venv/bin/python -m pytest tests/ccxt_upgrade
   ```

8. Run the relevant existing exchange tests for the touched surfaces.

   Minimum useful slice:

   ```bash
   ./venv/bin/python -m pytest \
     tests/exchanges/test_ccxt_bot.py \
     tests/exchanges/test_ccxt_bot_position_side.py \
     tests/test_utils_maps.py
   ```

9. Only accept the upgrade after every meaningful diff is explained:

   - harmless upstream drift
   - Passivbot test/docs update needed
   - Passivbot code fix needed
   - upgrade should be rejected for now

## What To Capture

Default useful sections:

1. `markets`
2. `capabilities`
3. `balance`
4. `positions`
5. `open_orders`

Optional section:

1. `trades`

Use `trades` when reviewing fill/PnL-sensitive exchanges or when CCXT release notes mention trade-history changes.

## High-Risk Fields

Inspect these areas first in the diff:

1. `markets.summary.contracts.*`
   This covers `min_cost`, `min_qty`, `qty_step`, `price_step`, `contract_size`, `max_leverage`, isolated-only detection, and margin capability.

2. `markets.summary.eligible_symbols`
   Symbol-universe drift can change what the bot trades.

3. `markets.summary.ineligible_reasons`
   Quote-format or market-flag changes often show up here first.

4. `capabilities`
   Especially `setPositionMode`, `setMarginMode`, `setLeverage`, `watchOrders`.

5. `balance.normalized`
   Exchange-specific balance parsing is fragile.

6. `positions.normalized`
   Watch for changed `symbol`, `position_side`, `size`, `price`, and margin-mode fields.

7. `open_orders.normalized_all`
   Watch for changed `side`, `reduceOnly`, `clientOrderId`, `position_side`, or symbol formatting.

## How To Pick Symbols

Use a small, stable set per exchange:

1. one major liquid perp
2. one secondary perp
3. one special-case symbol if the exchange has one

Examples:

1. Bybit/Binance/OKX: `BTC/USDT:USDT`, `ETH/USDT:USDT`
2. Hyperliquid: add a HIP-3 symbol when relevant
3. New exchange integrations: include the exact symbols used during bring-up

## Fixture Policy

Saved live snapshots are not required to live in the repo.

Recommended use:

1. capture snapshots into `artifacts/ccxt_contracts/...` during upgrade review
2. when a changed shape reveals a durable contract worth guarding, add a sanitized fixture under `tests/fixtures/ccxt_contracts/...`
3. let `tests/ccxt_upgrade/test_fixture_replay.py` replay those fixtures through current summarization logic

Do not check in secrets or account-identifying values.

## Scope Of Offline Tests

The offline suite should guard Passivbot behavior, not CCXT behavior in the abstract.

Prioritize tests for:

1. market metadata extraction
2. market filtering and symbol mapping
3. margin-mode extraction
4. order-side / order-normalization logic
5. capability-flag handling
6. snapshot diff and fixture replay helpers

## Red Flags

Treat these as likely blockers until explained:

1. `None` -> nonzero changes in `limits.amount.min` or `limits.cost.min`
2. symbol format changes
3. `positionSide` / `posSide` / `positionIdx` changes
4. `reduceOnly` representation changes
5. capability flags flipping from true to false or vice versa
6. `contractSize` changes
7. margin mode fields moving or disappearing
8. open orders disappearing from normalized snapshots

## Notes For New Exchanges

When a PR depends on a newly added CCXT exchange:

1. capture snapshots as early as possible
2. add at least one exchange-specific contract test for the adapter
3. do not rely only on generic `CCXTBot` tests
4. document any required raw `info` fields that the adapter depends on

## Non-Goals

This workflow does not replace:

1. exchange-specific integration review
2. live paper-trading sanity checks
3. existing unit tests for adapter logic

It is specifically a guardrail against CCXT upgrade drift.
