# v8 Fill History Coverage Bootstrap Merge Checklist

This checklist defines what should be true before merging
`codex/v8-fill-history-coverage-bootstrap` into `v8`.

## Branch State

- Local branch: `codex/v8-fill-history-coverage-bootstrap`
- Latest behavior-fix head before docs-only merge prep: `2a879ecd`
- Do not merge from stale local state. Before merge:
  - fetch `origin`
  - verify local branch equals `origin/codex/v8-fill-history-coverage-bootstrap`
  - verify target `v8` is current with `origin/v8`.

## Required VPS Evidence

After pulling the latest hardening branch and restarting bots on vps5:

1. No `TypeError`/`NoneType` hard crashes.
2. No `MissingEma` Rust payload crashes.
3. Kucoin flat forager EMA gaps produce `mark_nontradable_until_fresh`, not
   execution-loop restart bursts.
4. Gateio transient forager-selected flat symbols with no exchange open order do
   not restart on missing close/required EMA.
5. Explicit normal symbols and held/open-order symbols still fail loudly or use
   their documented fallback path.
6. HSL startup/replay produces deterministic terminal failures for real contract
   violations, not restart storms.
7. Panic/protective actions remain possible when their own account-critical
   surfaces are fresh.
8. Console is noisy but not dominated by per-span recovery spam at INFO.
9. Resource load is understood:
   - CPU may still be high on 1 vCPU with five bots.
   - Swap use should not grow without bound.
   - Any new loop error class must be explained before merge.

## Known Acceptable Residuals

These do not block the merge by themselves:

- Kucoin REST `RequestTimeout` on `account-overview` or `positions` if the bot
  backs off and recovers without hitting the hourly restart limit.
- Websocket reconnect warnings when the bot continues via reconnect/REST
  fallback.
- EMA unavailable summaries for flat forager candidates when they mark symbols
  nontradable for the current cycle.
- Gateio public OHLCV recent-window clipping warnings during HSL replay, if
  replay completes or fails deterministically according to contract.
- High CPU on the small VPS, if no new tight restart loop is present. This is a
  logging/observability and deployment-sizing concern for the next branch.

## Blocking Residuals

These should block merge:

- Any hard loop restart caused by flat forager approved-universe symbols.
- Any hard loop restart caused by flat transient forager-selected symbols with
  no exchange open order.
- Any silent neutral default passed to Rust for required EMA/risk/order inputs.
- Any HSL startup behavior where historical RED alone triggers a current panic
  after current drawdown has recovered.
- Any uncontrolled restart burst approaching the configured hourly error cap.
- Any local-only cache/state dependency that violates stateless restart
  behavior.

## Local Validation

Run before merge:

```bash
./venv/bin/python -m pytest tests/test_missing_ema_fix.py tests/test_live_candle_budget.py -q
```

Recommended if time allows:

```bash
./venv/bin/python -m pytest tests/test_live_reconciler*.py tests/test_live_executor*.py -q
```

Run diff hygiene:

```bash
git diff --check
git diff --cached --check
```

If Rust changed after this checklist was written, rebuild/restamp and run the
relevant Rust/PyO3 tests before trusting Python tests.

## Merge Procedure

1. Fetch fresh refs:
   ```bash
   git fetch origin
   ```
2. Verify branch head:
   ```bash
   git rev-parse HEAD
   git rev-parse origin/codex/v8-fill-history-coverage-bootstrap
   ```
3. Switch to `v8` and fast-forward it to origin:
   ```bash
   git checkout v8
   git pull --ff-only origin v8
   ```
4. Merge the hardening branch:
   ```bash
   git merge --no-ff origin/codex/v8-fill-history-coverage-bootstrap
   ```
5. Run local validation.
6. Push `v8`.
7. After merge, fork the logging branch from updated `v8`:
   ```bash
   git checkout -b codex/v8-live-event-pipeline-phase1
   ```

## Post-Merge Logging Branch Start

The next branch should implement only Phase 0/1 from:

- `docs/plans/live_logging_overhaul_plan.md`
- `docs/plans/live_logging_phase0_phase1_spec.md`
- `docs/plans/live_logging_migration_audit.md`

Do not start broad log migration in the first logging branch.
