# Logging Guide (Policy)

This file defines current logging policy. It is not a historical work log.

## Levels

| Level | Audience | Use |
|-------|----------|-----|
| INFO | Operators | Durable operational events |
| DEBUG | Developers | Decision context + diagnostics |
| TRACE | Deep debugging | Full payloads/firehose |

## INFO Must Include

1. Order create/cancel actions with reason/context.
2. Fill events with symbol/side/qty/price and PnL where relevant.
3. Position/balance state changes.
4. Mode changes meaningful to operators.
5. Unexpected errors with traceback.

## Keep Out Of INFO

1. High-frequency polling or fetch loops with no state change.
2. Internal cache noise and repeated maintenance chatter.
3. Full API payloads.

## Tagging

Use stable `[tag]` prefixes, for example:

1. `[order]`
2. `[fill]`
3. `[pos]`
4. `[balance]`
5. `[health]`
6. `[candle]`
7. `[ws]`
8. `[risk]`
9. `[ema]`

## Fallback Visibility

Fallbacks in critical paths log warnings with required context from `error_contract.md`.

## Review Heuristic

1. Can INFO be tailed for long periods without noise overload?
2. Can operators explain why bot is waiting vs broken?
3. Are DEBUG logs sufficient to reconstruct decision context?
