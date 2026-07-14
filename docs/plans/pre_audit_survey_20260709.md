# Pre-Audit Survey — 2026-07-09

High-level, shallow pass over docs and code on branch `v8` to rank areas for deeper audits.
Five parallel surveys: docs, live/exchange layer, data pipeline, Rust core, monitor + test infra.
Deliberately excludes HSL/unstuck/risk-enforcers and the optimizer, which were covered by the
July 2026 deep audits (see `risk_unstuck_hsl_fable_audit_action_plan_20260702.md` and
`optimizer_audit_pr_tracker.md`).

## Ranked deep-audit targets

### 1. Live execution edge cases (highest value — DEFERRED until live logging overhaul lands)

Architecture is better than expected: websocket is advisory-only (`handle_order_update` marks
state dirty; REST is authoritative), and `live/executor.py` ambiguity handling is thorough
(typed `EXECUTION_AMBIGUOUS` events instead of assumed success). The fragile spots are exactly
the untested ones:

- `execute_cancellation` (`src/passivbot.py:17642-17671`) matches ~12 hardcoded error substrings
  across 9 exchanges and fabricates a synthetic cancel-success
  (`_ambiguous_cancel_success_result`, `src/passivbot.py:8718`). A missed/renamed exchange error
  string turns a real cancel failure into assumed success. No dedicated tests.
- Per-exchange `_normalize_positions` / `_get_balance` parsers trust raw exchange payloads behind
  broad `except Exception` fallbacks; only bybit/bitget/hyperliquid have dedicated tests.
  defx/gateio/paradex adapters have almost no direct coverage.
- Staged concurrent state refresh (`src/live/state_refresh.py`, `src/passivbot.py:12765`) whole-dict
  reassigns `open_orders`/`positions`/`balance` while the reconciler reads them; only ~7 lock sites
  in 18k lines. Plan-computed-against-half-updated-snapshot is plausible.
- `hyperliquid.py` reimplements base paths (own execute/watch, stateful position/balance cache with
  two locks) — divergence from the tested base path.
- Debug `print()` calls in the executor's order-failure hot path
  (`src/live/executor.py:500,571,582,689,711,727`).

Deferred because the live logging overhaul (`live_logging_overhaul_plan.md`) is actively churning
this layer; audit once it settles.

### 2. Backtest data integrity + artifact-build performance (STARTING HERE)

Everything backtest and optimizer see flows through one materialized artifact
(`hlcvs.dat` / `timestamps.dat` / `btc_usd_prices.dat` via `src/backtest_dataset_materializer.py`).
A bug here silently corrupts every backtest and optimizer eval. Leads from the survey:

- `_fill_sparse_hlcv_gaps` (`src/backtest_dataset_materializer.py:36-49`): unbounded interior
  forward-fill fabricating flat candles with `valid_mask=True`, fed to Rust price/EMA math.
- `ensure_millis_df` (`src/ohlcv_utils.py:11-59`): ms-vs-s timestamp unit guessed heuristically
  from median inter-row diff; each TradFi provider hand-rolls its own unit conversion
  (`src/tradfi_data.py`) — none tested.
- Multiple independent inclusive-end/`+1` boundary conventions across
  `ohlcv_store.py:305-357`, catalog, and materializer — off-by-one territory.
- `prepare_hlcvs` partial/short-tail return paths (`src/hlcv_preparation.py`): ~15
  `except Exception → return None/df` sites; "unconfirmed short tail" logic may pass incomplete
  data as complete coverage.
- Cross-process coherence: stale-timeout lockfiles (`candlestick_manager.py:1031`) plus in-memory
  `_verified_checksums` under concurrent optimizer workers.
- Zero test coverage: `tradfi_data.py`, `legacy_data_migrator.py`, and no dedicated test for the
  materializer's synthetic-fill behavior.

Solid (narrow the audit around these): per-chunk SHA256 with repair-on-read, atomic tmp+rename
writes, dup-timestamp `HlcvsDataIntegrityError`, large OHLCV test suite.

Performance angle: artifact building is a known slow spot. This is the Python preparation path,
complementary to `backtester_performance_optimization_plan.md` (which targets the Rust sim loop;
its Pass 7 only gestures at Python-side prep).

### 3. Rust duplicated grid math + NaN panic cluster

Backtest/live parity is structurally guaranteed (both route through
`orchestrator::compute_ideal_orders*` → `strategies::generate_orders`) — skip the classic parity
audit. Real risks:

- `strategies/trailing_grid_v7.rs` reimplements `calc_grid_entry_long` / `calc_next_entry_long` /
  close math separately from `entries.rs`/`closes.rs`; the PyO3 exports `calc_entries_long_py` /
  `calc_closes_long_py` route to the entries.rs family and can silently diverge from what the
  v7 strategy actually trades. No compiler coupling between the two families.
- NaN-panic cluster reachable from Python: `backtest.rs:287-304,1440,1449` and
  `analysis.rs:732,874,893,954,1300,1621` use `partial_cmp().unwrap()` / `.max().unwrap()` on
  candle/metric data; the same `analysis.rs` handles NaN defensively elsewhere (150/457/556/628) —
  inconsistent. Delisted-coin NaN candles are the classic trigger.
- `entries.rs` has only 4 tests for ~1.3k lines of pricing code; PyO3 export layer essentially
  untested at the Rust level.

Solid: `python.rs` boundary (zero production unwrap/expect/panic, layered input validators),
`orchestrator.rs` (zero production unwraps, 43 tests incl. reference cross-checks), shared
`trailing.rs`.

### 4. Test-infrastructure integrity

- `tests/conftest.py:13` installs a stub `passivbot_rust` when the compiled extension is missing;
  core suites (HSL coin mode, unstucking, entries sizing, fake_live, orchestrator JSON API…)
  then silently skip. Combined with no-op CI (`.github/workflows/main.yml` runs `true`), nothing
  guarantees they ever run.
- `src/risk_limits.py`: zero test coverage. Highest-risk untested module.
- `pytests/` is orphaned: not in `pytest.ini` testpaths; contains a diverged copy of
  `test_gate_entries_by_twel_complex.py` and an unported `test_unstucking_cases.py`.
- `fake_live_slow` marker declared and excluded in `pytest.ini` but no test carries it.

### 5. Monitor relay auth (quick fix, not an audit)

Relay (`src/monitor_relay.py`) serves `/snapshot`, `/dashboard`, `/ws` with no authentication and
no origin checks; `--host` is free-form and `0.0.0.0` is whitelisted in the auto-launch guards
(`monitor_web.py:67`, `monitor_dev.py:165`). Default loopback bind is safe; a token check or hard
loopback lock is cheap insurance. Rest of the stack is solid: publisher is atomic/disk-full-aware,
relay is out-of-process, no secrets found in telemetry payloads. One uncontrolled sink worth a
glance: `record_error` persists arbitrary `str(error)` + caller payload
(`monitor_publisher.py:611`).

### 6. Docs (narrow pass only)

Nav-linked docs are current — every spot-checked config key, CLI flag, and tool name exists in
code, and v7 mentions are intentional migration context. Targeted checks worth doing:

- Stated defaults in `configuration.md` vs `src/config/schema.py` (defaults drift fastest).
- `config.bot.md` claims to mirror the algebra in `entries.rs`/`closes.rs`/`risk.rs` — a direct
  code-mirror contract; verify against the source of truth (and against trailing_grid_v7.rs, see
  target 3).
- ~35 orphaned pages not in `mkdocs.yml` nav, including all four `equity_hard_stop_loss*.md`
  docs, `hyperliquid_guide.md`, `min_effective_cost.md`, `trailing_grid_ratio.md`,
  `config.bot.md` — decide inclusion.
- `mkdocs.yml:27-28` references `docs/images/logo.png`, which does not exist
  (only the two SVGs do).

## Quick wins (independent of audit choice)

- Remove debug `print()`s from `src/live/executor.py` hot paths.
- Fix or remove the `logo.png` reference in `mkdocs.yml`.
- Merge-or-delete the orphaned `pytests/` directory.
- Drop or implement the `fake_live_slow` marker tier.

## Method note

Survey was shallow by design: structure mapping, grep-level smell scans, spot checks. File:line
references were reported by survey agents and spot-checked, but individual findings above are
leads, not confirmed bugs, until the corresponding deep audit verifies them.
