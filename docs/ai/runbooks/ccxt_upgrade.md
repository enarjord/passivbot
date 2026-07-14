# CCXT Upgrade Runbook

Use this when changing the pinned CCXT version or reviewing code that depends on a newer CCXT.
The primary risk is silent semantic drift in market, balance, position, order, trade, or capability
shapes—not merely import failure.

Live capture uses network endpoints and may use account credentials. Obtain the authority required
by `AGENTS.md`; keep authenticated capture low-rate and sanitized. Default tests remain offline.

## Repository Tools

- contracts: `src/ccxt_contracts.py`
- capture: `python -m src.tools.capture_ccxt_contracts --help`
- diff: `python -m src.tools.diff_ccxt_contracts --help`
- offline suite: `pytest tests/ccxt_upgrade`

## Procedure

1. On the currently pinned version, capture a small stable symbol set for each affected exchange:

   ```bash
   ./venv/bin/python -m src.tools.capture_ccxt_contracts \
     --user <user> --label <exchange>-old \
     --symbols BTC/USDT:USDT,ETH/USDT:USDT
   ```

2. Upgrade CCXT in an isolated branch and reinstall the environment.
3. Capture the same users, exchanges, symbols, and optional sections as `<exchange>-new`.
4. Diff the snapshots:

   ```bash
   ./venv/bin/python -m src.tools.diff_ccxt_contracts \
     artifacts/ccxt_contracts/<exchange>/<exchange>-old.json \
     artifacts/ccxt_contracts/<exchange>/<exchange>-new.json
   ```

5. Classify every meaningful difference as harmless upstream drift, required fixture/docs update,
   Passivbot defect, or upgrade blocker.
6. Add sanitized fixtures for durable contracts exposed by the comparison.
7. Run:

   ```bash
   ./venv/bin/python -m pytest tests/ccxt_upgrade
   ./venv/bin/python -m pytest \
     tests/exchanges/test_ccxt_bot.py \
     tests/exchanges/test_ccxt_bot_position_side.py \
     tests/test_utils_maps.py
   ```

8. Run additional exchange-adapter tests for every touched surface.

## Snapshot Scope

Default sections are `markets`, `capabilities`, `balance`, `positions`, and `open_orders`. Add
`trades` for fill/PnL-sensitive exchanges or releases mentioning trade history.

Use one major perp, one secondary perp, and a documented special case where applicable—for
example a HIP-3 symbol. The suite guards Passivbot interpretation, not CCXT behavior in the
abstract.

Inspect first:

- `markets.summary.contracts.*`: min cost/qty, steps, contract size, leverage, margin capability
- eligible symbols and ineligibility reasons
- position-mode, margin-mode, leverage, and websocket capability flags
- normalized balance
- normalized position symbol, position side, size, price, and margin mode
- normalized order side, reduce-only, client ID, position side, and symbol

Treat symbol-format changes, contract-size changes, capability flips, disappearing orders, and
`None`↔nonzero minimum-limit changes as blockers until explained.

## Fixture Policy

Live snapshots stay under `artifacts/ccxt_contracts/` and need not be committed. Commit only
sanitized minimal fixtures under `tests/fixtures/ccxt_contracts/` when they encode a durable
Passivbot contract. Never commit secrets or account-identifying values.

For a newly supported exchange, capture early, add at least one adapter-specific contract test,
and document every raw `info` field the adapter requires. Generic `CCXTBot` coverage alone is not
sufficient.

Accept the upgrade only when every meaningful diff is explained and affected offline contracts
pass. This runbook does not replace exchange-specific payload review or explicitly authorized
paper-trading validation.
