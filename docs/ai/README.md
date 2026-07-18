# AI Documentation Router

Read `../../AGENTS.md` and `principles.md` first. Then load only the documents required by the
current task.

## Canonical Contracts

| Task | Read |
|---|---|
| Trading-critical failure, fallback, readiness, or data availability | `error_contract.md` |
| Component ownership, live/backtest/optimizer flow | `architecture.md` |
| Validation selection or code review | `validation.md` |
| Logging levels, redaction, sinks, console behavior | `logging_policy.md` |
| Active architectural rationale | `decisions.md` |

## Task-Specific Routing

| Task | Read |
|---|---|
| Exchange/API integration, pagination, broker attribution | `features/exchange_integrations.md` |
| CCXT dependency upgrade | `runbooks/ccxt_upgrade.md` |
| Rust/PyO3 build, extension loading, Rust tests | `runbooks/rust_extension.md` |
| Strategy schema, order logic, risk, unstuck | `architecture.md`, `error_contract.md`, `features/strategy_runtime.md` |
| Equity Hard Stop Loss episodes, replay, flattening, cooldown | `error_contract.md`, `features/equity_hard_stop_loss.md`, `features/fill_events_manager.md` |
| Candles, OHLCV coverage, gaps, projections | `error_contract.md`, `features/candlestick_manager.md`; add `features/exchange_integrations.md` for remote fetch behavior |
| Fill/PnL ingestion, fees, coverage | `error_contract.md`, `features/fill_events_manager.md`, relevant exchange contract |
| Commands, tests, backtests, optimizer, execution safety | `runbooks/commands.md` |
| Autonomous PR review | `runbooks/pr_review.md` |
| Crash discovery and stress-suite generation | `runbooks/crash_discovery.md` |
| Structured live events | `features/live_events.md`, `generated/live_event_registry.md` |
| Monitor persistence, recovery, rotation, retention | `features/monitor_persistence.md` |
| Feature-specific change | `features/README.md` and the relevant feature contract |
| Historical incident rationale | relevant file under `case_studies/` |

Logging-overhaul implementation work should additionally read
`../plans/live_logging_overhaul_current_status.md` and
`../plans/live_logging_overhaul_pr_loop_workflow.md`. Those files are temporary project state, not
canonical engineering policy.

## Document Classes

- Canonical contracts state current normative behavior.
- Feature contracts add subsystem-specific invariants, failure semantics, validation, and code maps.
- Runbooks describe procedures and may contain commands.
- Generated references mirror code-owned registries.
- Case studies and plans explain history; they do not override current contracts.

Keep canonical documents timeless, compact, and free of progress ledgers. Put each durable rule in
one canonical location and link to it elsewhere.

Run documentation checks with:

```bash
PYTHONPATH=src python src/tools/check_ai_docs.py
PYTHONPATH=src python src/tools/generate_live_event_registry.py --check
pytest tests/test_ai_docs.py tests/test_live_event_registry_docs.py
```

The mandatory-context ceiling is 1,500 words and is warning-only. Missing routed files and stale
generated registries remain test failures.

## Temporary Compatibility Paths

`principles.yaml`, `commands.md`, `pr_auto_review_loop.md`, and `code_review_prompt.md` remain as
narrow routes for active external automations. Their canonical content lives in `principles.md`,
`runbooks/commands.md`, `runbooks/pr_review.md`, and `validation.md`. Remove the routes only after
every scheduled consumer has migrated and a changed-head wake proves the new reads succeed. Review
schedulers must also migrate base-branch filters and compact state after a default-branch cutover;
the proving wake must discover a PR against the new target rather than merely load the new path.
