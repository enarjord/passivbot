# Pitfalls (Passivbot-Specific)

Check this file before implementing behavior changes.

## 1) Python Patch To Rust Behavior

Avoid: fixing order behavior only in Python.

Do: implement behavior changes in `passivbot-rust/src/`.

## 2) Silent Handling In Critical Paths

Avoid:

1. `except Exception: pass` / `continue`
2. dropping exceptions from `return_exceptions=True`
3. defaulting required values to neutral values

Do: follow `error_contract.md` and hard-fail by default.

Example:

```python
# WRONG
results = await asyncio.gather(*tasks, return_exceptions=True)
for span, res in zip(spans, results):
    if isinstance(res, Exception):
        continue
    out[span] = float(res)

# CORRECT
results = await asyncio.gather(*tasks, return_exceptions=True)
for span, res in zip(spans, results):
    if isinstance(res, Exception):
        raise RuntimeError(f"missing required EMA {symbol} span={span}: {res}") from res
    out[span] = float(res)
```

## 3) Catching Exceptions In Exchange Fetch Methods

Avoid: catching in `fetch_balance`, `fetch_positions`, `fetch_open_orders`, etc.

Do: let exceptions propagate to caller policy (`restart_bot_on_too_many_errors()`).

## 4) Confusing `position_side` And `side`

Avoid: using buy/sell semantics for long/short logic.

Do:

1. `position_side` / `pside` for long/short
2. `side` / `order_side` for buy/sell

## 5) Unsigned Quantity Assumptions

Avoid: treating `qty` and `pos_size` as unsigned internally.

Do: keep signed convention; use `abs(qty)` only when exchange payload requires it.

Example:

```python
# WRONG
new_exposure = abs(position_size) + abs(entry_qty)

# CORRECT
new_exposure = position_size + entry_qty
```

## 6) Rounded EMA Span Derivations

Avoid: `int(sqrt(span0 * span1))`.

Do: keep EMA spans as float throughout calculations.

Example:

```python
# WRONG
span2 = int(sqrt(span0 * span1))

# CORRECT
span2 = sqrt(span0 * span1)
```

## 7) Restart-Dependent Runtime State

Avoid: local state that changes trading decisions and cannot be rederived.

Do: keep only performance caches that do not alter decisions.

## 8) Exchange Name Mismatch In Cache Paths

Avoid: raw CCXT IDs (`binanceusdm`, `kucoinfutures`) in cache paths.

Do: normalize with `to_standard_exchange_name()`.

## 9) Blind Pagination Assumptions

Avoid: trusting wrapper defaults for completeness.

Do: read `exchange_api_quirks.md` and use overlap/cursor strategy where needed.

## 10) Stock Perps On Non-Hyperliquid

Avoid: routing HIP-3 stock perps to non-Hyperliquid exchanges.

Do: keep stock perp routing constrained to Hyperliquid.

## 11) Rust/PyO3 Build Confusion

Avoid: debugging behavior before validating which extension binary is loaded.

Do: read `build_pitfalls.md` and verify module path + rebuild status first.

## Pre-PR Safety Scan

```bash
rg -n "except Exception|return_exceptions=True|\.get\([^\n]*,\s*(0|0\.0|None|False|\{\}|\[\])\)" src tests
```
