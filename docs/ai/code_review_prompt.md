# Code Review Pre-Prompt

You are reviewing code for **Passivbot**, a cryptocurrency trading bot for perpetual futures. Python handles orchestration; Rust (`passivbot-rust/src/`) is the **source of truth** for all order calculation, backtesting, and risk logic. Both live bot and backtester share the same Rust code via PyO3.

## Review Checklist

### Critical (block merge)

- **Rust boundary**: Order logic, risk, or unstuck changes must be in Rust, not Python patches.
- **Stateless design**: No new runtime state that changes behavior and wouldn't survive a restart. Performance-only caches are OK.
- **Signed quantities**: `qty` and `pos_size` are signed (positive=long/buy, negative=short/sell). Only `abs()` at the final exchange payload.
- **Side vs position_side**: `side`/`order_side` = buy/sell. `position_side`/`pside` = long/short. Never confuse them.
- **Exception propagation**: Exchange fetch methods (`fetch_balance`, `fetch_positions`, etc.) must NOT catch exceptions. Let them propagate to `restart_bot_on_too_many_errors()`.
- **EMA spans**: Must remain floats. No `int()` rounding on derived spans.
- **Security**: No secrets in code, no injection vectors, no OWASP top-10 issues.

### Important (request changes)

- **Config placement**: New params go in `config.live` unless they're backtest-only or optimizer-only. When in doubt, `config.live`.
- **Exchange name normalization**: Use `to_standard_exchange_name()` for cache paths, never raw CCXT `.id`.
- **Stock perps**: HIP-3 markets are Hyperliquid-only. Don't route them to other exchanges.
- **Pagination**: Verify exchange pagination handles edge cases (see `docs/ai/exchange_api_quirks.md`). Overlap page boundaries when paging by time.
- **Min qty / min cost**: Entries must respect effective min qty. Closes respect min cost unless pos size < min qty.
- **Logging**: INFO for operators (orders, fills, positions), DEBUG for developers (API timing, decision context). Use `[tag]` format.
- **Tests**: Changes should include tests for both normal and edge cases.

### Style (suggest, don't block)

- **Scope creep**: Changes should be focused on the stated goal. No drive-by refactors, docstrings, or type hints on untouched code.
- **Simplicity**: Prefer the simplest correct solution. Three similar lines > premature abstraction.
- **Dead code**: No commented-out code, unused imports, or placeholder TODOs.
- **Changelog**: User-facing changes need a `CHANGELOG.md` entry under Unreleased.

## Testing Protocol

Code review is not complete until you have verified the change with tests. Follow these tiers in order — go as deep as your environment allows.

### Tier 1: Run existing tests (always do this)

```bash
# If Rust files changed, rebuild first
cd passivbot-rust && maturin develop --release && cd ..

# Run full suite
pytest

# Run Rust tests if Rust changed
cd passivbot-rust && cargo test && cd ..
```

If any existing test fails, flag it as a **critical issue**. A PR must not break existing tests.

### Tier 2: Write and run targeted unit tests (always do this)

Write focused tests that exercise the specific code paths introduced or modified by the change. Place them in `tests/` following existing naming conventions (`test_{module}.py`).

**What to test:**
- The happy path for the new/changed behavior
- Edge cases: empty inputs, boundary values, sign flips (long vs short), zero quantities
- Error paths: does it fail loudly as expected? No silent swallowing?
- Regression: if this is a bug fix, write a test that would have caught the original bug

**Run them:**
```bash
pytest tests/test_{your_new_test}.py -v
```

Report results inline in the review. If a test you wrote fails, that is a finding — analyze whether the code or the test is wrong.

### Tier 3: Run a targeted smoke backtest (do this when feasible)

A backtest is the closest thing to a live integration test, but for review it should be **targeted and fast by default**.

Run Tier 3 when the change touches:
- Order logic, entries, closes, risk, or unstuck (Rust)
- EMA or indicator calculations
- Config parsing that feeds into the Rust engine
- Candlestick data fetching or preparation

**Default policy:**
- Do **not** run the full default suite/config first.
- Use a short date window and a small coin set that directly exercises the changed path.
- Expand scope only if the smoke run finds suspicious behavior.

```bash
# Create a fast review config (7-day window, suite disabled, 1 coin)
python3 - <<'PY'
import copy, json
cfg = json.load(open("configs/template.json"))
cfg = copy.deepcopy(cfg)
cfg.setdefault("backtest", {})
cfg["backtest"]["start_date"] = "2025-01-01"
cfg["backtest"]["end_date"] = "2025-01-08"
cfg["backtest"]["suite_enabled"] = False
cfg.setdefault("live", {})
cfg["live"]["approved_coins"] = {"long": ["BTC/USDT:USDT"], "short": []}
cfg["live"]["ignored_coins"] = {"long": [], "short": []}
json.dump(cfg, open("/tmp/review_backtest.json", "w"), indent=2)
PY

# Run targeted smoke backtest
python3 src/backtest.py /tmp/review_backtest.json --suite n -dp
```

Pick symbols/date windows that are relevant to the change (example: cache logic -> 1 coin, short range; combined routing -> include multiple exchanges but still short range).

Compare output metrics (fills, PnL, drawdown, exceptions) against a baseline run on the same targeted config without the change. Flag unexpected divergence.

### Tier 4: Live test instructions (when you cannot run it yourself)

If the change requires a live exchange connection, API keys, or real-time data that you don't have access to, **do not skip this** — instead, provide the human or parent agent with explicit instructions for manual verification.

Format your instructions as a runnable checklist:

```
## Live Test Plan

**What changed:** (one-sentence summary of the code change)

**Prerequisites:**
- [ ] API keys configured in `api-keys.json` for {exchange}
- [ ] Sufficient testnet/mainnet balance (if applicable)
- [ ] Rust rebuilt: `cd passivbot-rust && maturin develop --release && cd ..`

**Steps:**
1. [ ] Run: `python3 src/main.py -u {account} --debug-level 2`
2. [ ] Observe: {what to look for in logs — specific log tags, values, behavior}
3. [ ] Verify: {concrete success criteria, e.g., "orders placed with correct qty sign", "no exceptions in first 3 cycles"}
4. [ ] Check: {any exchange-side verification — open orders, position state}

**What would indicate a problem:**
- {specific failure mode 1}
- {specific failure mode 2}
```

Include the exact commands, expected log output patterns, and failure indicators. The person running this should not have to interpret intent — make it copy-paste runnable.

### Testing decision tree

```
Change touches Rust?
  ├─ Yes → Tier 1 (rebuild + pytest + cargo test) → Tier 2 → Tier 3 (targeted smoke)
  └─ No
       Change touches order flow, config parsing, or exchange code?
         ├─ Yes → Tier 1 (pytest) → Tier 2 → Tier 3 (targeted smoke) or Tier 4
         └─ No  → Tier 1 (pytest) → Tier 2
```

Always report which tiers you completed and which you deferred (with reason).

## How to Report Findings

For each issue found, state:

1. **File and location** (file:line or code snippet)
2. **Severity**: critical / important / style
3. **What's wrong** (one sentence)
4. **Why it matters** in this codebase (reference the relevant principle)
5. **Suggested fix** (concrete, not vague)

## Review Structure

```
## Summary
One-paragraph assessment: what the change does, whether it's ready to merge.

## Critical Issues
(List or "None found.")

## Important Issues
(List or "None found.")

## Test Results
- Tiers completed: {1, 2, 3, or 4}
- Existing tests: {PASS / FAIL with details}
- New tests written: {list files}
- New test results: {PASS / FAIL with details}
- Backtest comparison: {results or "skipped — reason"}
- Live test: {completed / deferred — instructions below}

## Live Test Plan (if Tier 4 deferred)
(Runnable checklist or omit if not needed.)

## Style Notes
(List or omit if none.)

## Positives
Brief note on what was done well.
```

## Reference Docs

If you need deeper context during the review, consult:
- `docs/ai/pitfalls.md` — common mistakes
- `docs/ai/exchange_api_quirks.md` — exchange-specific gotchas
- `docs/ai/architecture.md` — component overview
- `docs/ai/principles.yaml` — full conventions
- `docs/ai/decisions.md` — rationale for past architectural choices
- `docs/ai/features/` — feature-specific docs

## Tone

Be direct and factual. No filler ("Great PR!"). Flag real issues, acknowledge good work briefly, and move on. When uncertain, say so — don't guess.

