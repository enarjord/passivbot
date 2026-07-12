# Validation And Review Matrix

Validation is proportional to the contract changed, not to the number of edited lines. Record the
commands run, outcomes, environmental limitations, and any intentionally untested surface.

## Minimum Matrix

| Change class | Minimum validation |
|---|---|
| Documentation only | Link/path/router checks; verify technical claims against current code/tests |
| Python orchestration | Focused pytest for changed callers and failure paths; compile/lint if configured |
| Rust strategy/order/risk | Focused Rust tests, extension rebuild/verification, Python parity tests, bounded real backtest when behaviorally relevant |
| Exchange adapter | Offline fixtures and request-construction tests; documented exchange contract review; no authenticated calls without approval |
| Config/schema | Loader/default/roundtrip tests, affected example configs, CLI parse or bounded smoke |
| Candles/data preparation | Gap/coverage/integrity tests and a bounded local-cache or CLI smoke when performance/integration changes |
| Fill/PnL/risk readiness | Coverage, degraded/unavailable, recovery, and restart/parity tests |
| Logging/monitor/events | Routing, boundedness, redaction, sink-failure isolation, event-registry parity, fake-live when useful |
| Backtest/optimizer | Focused tests plus a bounded real CLI integration smoke |

Run broader suites when the affected dependency surface justifies them. Bare `pytest` is neither a
universal minimum nor a substitute for a contract-specific smoke.

## Rust Extension Validation

Rust-backed Python tests must use the current compiled source, not a stub or stale shadowing module.
Follow `runbooks/rust_extension.md` to rebuild, identify the loaded ABI3/package artifact, verify its
source fingerprint, and distinguish stub-compatible tests from tests requiring the real extension.

## Exchange And Network Tiers

1. **Offline deterministic:** unit tests, fixtures, cached data, Rust backtests without downloads,
   and `src/tools/run_fake_live.py` using the local `fake` exchange. Safe by default.
2. **Public unauthenticated network:** public candles, market metadata, and explicitly
   unauthenticated probes. State that network access will occur; observe rate limits.
3. **Authenticated read-only:** balances, positions, open orders, private fills, account modes, and
   any endpoint requiring keys. Requires explicit user approval.
4. **Authenticated mutating:** bot startup, order create/cancel, leverage/margin/position-mode setup,
   deployment, or remote process control. Requires explicit user approval and a clear target.

Testnet, demo, sandbox, and paper-trading labels do not lower the authority tier if credentials or
account mutations are involved.

## Review Method

- Review the complete change against its current merge base and record base/head SHAs.
- Trace changed behavior through direct callers and consumers.
- Distinguish regressions introduced by the change from failures reproduced on the target branch.
- Prefer findings about correctness, trading safety, security, operations, and maintainability over
  style-only commentary.
- Findings include exact locations, evidence, impact, and the narrowest credible fix.
