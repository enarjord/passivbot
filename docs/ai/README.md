# AI Documentation Router

Read `../../AGENTS.md` and `principles.md` first. Then load only the documents required by the
current task.

## Canonical Contracts

| Task | Read |
|---|---|
| Trading-critical failure, fallback, readiness, or data availability | `error_contract.md` |
| Component ownership, live/backtest/optimizer flow | `architecture.md` |
| Configuration field types and defaults | `generated/config_fields.md` |
| Analysis quantity units, denominators and observation populations | `generated/metric_fields.md` |
| Numeric names, units, denominators, and metric spelling | `numeric_conventions.md` |
| Validation selection or code review | `validation.md` |
| Adding or changing diagnostics, event routing, console or text logging | `logging_policy.md` |
| Active architectural rationale | `decisions.md` |

## Task-Specific Routing

| Task | Read |
|---|---|
| Base branch, porting between v8 and v9, integrating `master` | `runbooks/branches.md` |
| Exchange/API integration, pagination, broker attribution | `features/exchange_integrations.md`, `features/exchanges/<venue>.md` |
| CCXT dependency upgrade | `runbooks/ccxt_upgrade.md` |
| Rust/PyO3 build, extension loading, Rust tests | `runbooks/rust_extension.md` |
| Strategy schema, order logic, risk, unstuck, forager | `architecture.md`, `error_contract.md`, `features/strategy_runtime.md` |
| Equity Hard Stop Loss episodes, replay, flattening, cooldown | `error_contract.md`, `features/equity_hard_stop_loss.md`, `features/fill_events_manager.md` |
| Candles, OHLCV coverage, gaps, projections | `error_contract.md`, `features/candlestick_manager.md`; venue file for remote fetches |
| Fill/PnL ingestion, fees, coverage | `error_contract.md`, `features/fill_events_manager.md`, relevant venue contract |
| Historical runtime/fill attribution | `features/runtime_attribution.md`, `features/fill_events_manager.md`, `features/live_events.md` |
| Commands, tests, backtests, optimizer, execution safety | `runbooks/commands.md` |
| Planning, running, monitoring, or qualifying an optimization campaign | `runbooks/optimization_campaign.md` |
| Core backtest/optimizer metric definitions, clocks and statuses | `features/backtest_metrics.md` |
| Version selection, release trigger, release preparation, or publication | `runbooks/release.md` |
| Autonomous PR review | `runbooks/pr_review.md` |
| Crash discovery and stress-suite generation | `runbooks/crash_discovery.md` |
| Structured live events | `features/live_events.md`, `features/live_events/<family>.md`, `generated/live_event_registry.md` |
| Monitor persistence, recovery, rotation, retention | `features/monitor_persistence.md` |
| Feature-specific change, or a subsystem without a contract | `features/README.md` and the relevant feature contract or guide |
| Historical incident rationale | relevant file under `case_studies/` |

Document classes, authoring rules, and checks are in
[documentation validation](validation.md#documentation-classes).
