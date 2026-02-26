# Pitfalls Registry

Common mistakes and how to avoid them. Check here before implementing.

---

## General LLM Pitfalls

These are common failure modes when coding with LLMs. Actively counteract them.

### Unchecked Assumptions

**Don't**: Silently assume intent when requirements are ambiguous.

**Because**: Assumptions compound and lead to implementations that miss the mark.

**Instead**: Verify ambiguous requirements. Ask clarifying questions.

---

### Hiding Confusion

**Don't**: Guess when uncertain; produce plausible-looking but incorrect code.

**Because**: Wrong code that looks right is harder to debug than obviously wrong code.

**Instead**: Surface uncertainty explicitly. Say "I'm not sure about X because Y."

---

### Ignoring Inconsistencies

**Don't**: Silently pick one interpretation when specs/code contain contradictions.

**Because**: The contradiction may indicate a bug or missing requirement.

**Instead**: Flag contradictions. Ask which interpretation is correct.

---

### Concealing Tradeoffs

**Don't**: Present one approach without mentioning alternatives.

**Because**: The user may prefer a different tradeoff.

**Instead**: Present options with pros/cons when multiple valid approaches exist.

---

### Failure to Push Back

**Don't**: Implement a request that seems wrong or suboptimal without comment.

**Because**: The user may not realize the implications.

**Instead**: Disagree respectfully. Explain concerns. Suggest alternatives.

---

### Sycophancy

**Don't**: Use phrases like "Great question!" or "Excellent idea!"

**Because**: It wastes tokens and sounds hollow.

**Instead**: Be direct, factual, and professionally objective.

---

### Overengineering

**Don't**: Add abstraction layers, config options, or generality before they're needed.

**Because**: YAGNI - You Ain't Gonna Need It. Complexity has ongoing costs.

**Instead**: Prefer the simplest solution. Add complexity only when justified by actual need.

---

### Abstraction Bloat

**Don't**: Create helpers/utilities for one-time operations. DRY prematurely.

**Because**: Three similar lines are often better than a premature abstraction.

**Instead**: Tolerate some duplication. Extract only when patterns stabilize.

---

### Dead Code Accumulation

**Don't**: Leave commented-out code, unused imports, or TODO comments for "later."

**Because**: Dead code obscures the codebase and suggests incomplete work.

**Instead**: Clean up anything you obsolete. Delete fully.

---

### Scope Creep in Edits

**Don't**: "Improve" code orthogonal to the task. Add docstrings, type hints, or refactors unless requested.

**Because**: Unrelated changes make PRs harder to review and may introduce bugs.

**Instead**: Stay focused on the requested change. Propose other improvements separately.

---

### Runaway Implementation

**Don't**: Keep writing when a solution grows to 500+ lines.

**Because**: Large solutions often indicate a wrong approach.

**Instead**: Stop and reconsider. Ask "Couldn't this be simpler?"

---

## Passivbot-Specific Pitfalls

### Confusing Position Side with Order Side

**Don't**: Mix up `position_side`/`pside` (long/short) with `side`/`order_side` (buy/sell).

**Because**: A long position can have both buy entries and sell closes.

**Example**:
```python
# WRONG: Using 'side' when you mean 'position_side'
if order['side'] == 'long':  # 'side' is buy/sell, not long/short!
    ...

# CORRECT:
if order['position_side'] == 'long':
    ...
```

**Instead**: Use explicit naming. Check variable names match their semantic meaning.

---

### Unsigned Quantities in Calculations

**Don't**: Forget that `qty` and `pos_size` are signed (positive=long/buy, negative=short/sell).

**Because**: Arithmetic on unsigned values gives wrong results for short positions.

**Example**:
```python
# WRONG: Treating qty as always positive
new_exposure = position_size + entry_qty

# CORRECT: Both values are already signed
new_exposure = position_size + entry_qty  # Works because signs are consistent
```

**Exception**: Final exchange payload may need `abs(qty)` per exchange requirements.

---

### Trusting CCXT Pagination Blindly

**Don't**: Assume CCXT's pagination handles all edge cases.

**Because**: CCXT normalizes APIs but may not expose cursors or handle exchange-specific pagination limits.

**Example**: Bybit's closed-PnL endpoint has cursor vs time-based pagination with different coverage (see `debugging_case_studies.md`).

**Instead**: Verify pagination behavior with real data. Check raw API responses. Implement hybrid pagination when needed.

---

### Rounding EMA Spans

**Don't**: Round intermediate EMA span calculations to integers.

**Because**: EMA calculations use float spans; rounding introduces drift.

**Example**:
```python
# WRONG:
span2 = int(sqrt(span0 * span1))

# CORRECT:
span2 = sqrt(span0 * span1)  # Keep as float
```

---

### Relying on State from Previous Loop

**Don't**: Store state that affects future trading decisions without ensuring it survives restart.

**Because**: Stateless design principle - bot must behave identically after restart.

**Example**:
```python
# WRONG: Tracking "last entry time" in memory only
self.last_entry_time[symbol] = time.time()  # Lost on restart

# ACCEPTABLE: Performance-only caches that don't change behavior
self.ema_cache[symbol] = ema_value  # Can be recomputed on restart
```

---

### Stock Perps on Non-Hyperliquid Exchanges

**Don't**: Route stock perp symbols (TSLA, NVDA, etc.) to exchanges other than Hyperliquid.

**Because**: Only Hyperliquid supports HIP-3 stock perpetuals. Other exchanges don't have these markets.

**Example**:
```python
# WRONG: Allowing stock perps in forager mode on Binance
approved = ["BTC", "ETH", "TSLA"]  # TSLA doesn't exist on Binance

# CORRECT: Filter stock perps to Hyperliquid only
if exchange != "hyperliquid" and is_stock_perp(symbol):
    continue  # Skip this symbol
```

---

### Catching Exceptions in Fetch Methods

**Don't**: Catch exceptions in exchange fetch methods (`fetch_balance`, `fetch_positions`, etc.).

**Because**: Return type becomes unclear (`Union[list, bool]`), error context is lost, errors may be silently ignored.

**Example**:
```python
# WRONG:
async def fetch_positions(self):
    try:
        return await self.exchange.fetch_positions()
    except Exception as e:
        logging.error(f"Failed: {e}")
        return False  # Caller must check for False

# CORRECT:
async def fetch_positions(self):
    return await self.exchange.fetch_positions()  # Let caller handle
```

**Instead**: Let exceptions propagate. Caller uses `restart_bot_on_too_many_errors()`.

---

### Using Raw CCXT Exchange IDs for Cache Paths

**Don't**: Use `exchange.id` (the CCXT instance ID) directly in file paths or cache directories.

**Because**: CCXT IDs like `"binanceusdm"` and `"kucoinfutures"` don't match the standard exchange names used throughout the codebase (`"binance"`, `"kucoin"`). This creates duplicate cache directories and breaks lookups.

**Example**:
```python
# WRONG: Using ccxt ID directly
cache_path = f"caches/ohlcv/{exchange.id}/"  # "caches/ohlcv/binanceusdm/"

# CORRECT: Normalize to standard name first
from utils import to_standard_exchange_name
ex = to_standard_exchange_name(exchange.id)  # "binance"
cache_path = f"caches/ohlcv/{ex}/"           # "caches/ohlcv/binance/"
```

**Instead**: Always call `to_standard_exchange_name()` (from `utils.py`) when deriving an exchange name from a CCXT instance. The standard names are: `binance`, `bybit`, `okx`, `gateio`, `bitget`, `kucoin`, `kraken`, `hyperliquid`.

---

### Patching Order Logic in Python

**Don't**: Fix order calculation bugs in Python code.

**Because**: Rust is the source of truth. Backtester and live bot must use identical logic.

**Instead**: Port the fix to Rust (`passivbot-rust/src/`). Both live bot and backtester call the same Rust code.

---

## Rust/PyO3 Build Pitfalls

### Stale .so in src/ vs venv (the #1 Rust pitfall)

**Don't**: Assume `maturin develop` makes pytest use your new Rust code.

**Because**: There are TWO copies of the compiled extension:
1. `src/passivbot_rust.cpython-312-darwin.so` — in the source tree
2. `venv/lib/python3.12/site-packages/passivbot_rust/` — installed by maturin

`tests/conftest.py` inserts `src/` at the FRONT of `sys.path`, so **pytest always loads the `src/` copy**. `maturin develop` only updates the venv copy.

**Example**:
```bash
# WRONG: Tests silently use old code
cd passivbot-rust && maturin develop --release
python -m pytest tests/test_orchestrator_json_api.py  # FAILS — stale .so in src/

# CORRECT: Update both locations
cd passivbot-rust
cargo clean && maturin develop --release
cp ../venv/lib/python3.12/site-packages/passivbot_rust/passivbot_rust.cpython-312-darwin.so ../src/
python -m pytest tests/test_orchestrator_json_api.py  # Uses fresh code
```

**Instead**: Always copy the .so from venv to `src/` after `maturin develop`. If tests fail mysteriously after a Rust change, this is the first thing to check.

---

### Using cargo test --lib on PyO3 projects

**Don't**: Run `cargo test --lib` to verify Rust changes.

**Because**: PyO3 projects can't link test binaries without the Python runtime. You'll get hundreds of linker errors (`_PyObject_GetAttr`, `_Py_InitializeEx`, etc.). This is expected and NOT caused by your code.

```bash
# WRONG: Fails with linker errors
cargo test --lib

# CORRECT: Verifies compilation without linking
cargo check --tests
```

**Instead**: Use `cargo check --tests` to catch all type/syntax errors. For actual test execution, build with maturin and run through pytest.

---

### Trusting maturin develop cache

**Don't**: Assume `maturin develop --release` recompiled when it finishes in under 1 second.

**Because**: If cargo's incremental cache thinks nothing changed, it returns "Finished in 0.04s" without recompiling — but the installed wheel may be stale from a previous build.

```bash
# WRONG: May silently skip recompilation
maturin develop --release  # "Finished in 0.04s" — not rebuilt!

# CORRECT: Force a clean build
cargo clean && maturin develop --release  # Full rebuild guaranteed
```

**Instead**: Always `cargo clean` first when you need to be certain the build is fresh.

---

### conftest.py passivbot_rust stub ordering

**Don't**: Assume all test files use the same `passivbot_rust` module.

**Because**: `tests/conftest.py` tries the real module first and falls back to a lightweight stub (`__is_stub__ = True`). But `test_passivbot_balance_split.py` installs its own minimal stub via `sys.modules.setdefault()` at module level. Import order matters:
- conftest.py runs first, tries real module
- If real module loads, `setdefault()` in test files is a no-op
- If real module fails, conftest's stub takes over

**Instead**: If your test requires the real Rust extension, add the `require_real_passivbot_rust_module` fixture (see `test_orchestrator_json_api.py`). If your test should work with the stub, use `sys.modules.setdefault()` like `test_passivbot_balance_split.py`.

---

## Template for New Pitfalls

```markdown
### [Pitfall Title]

**Don't**: What to avoid.

**Because**: Why it's wrong.

**Example**:
\`\`\`python
# WRONG:
code_example

# CORRECT:
better_code
\`\`\`

**Instead**: What to do.
```
