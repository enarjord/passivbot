# Risk management architecture notes

## Overview
This document summarises how the risk management services are composed across the
terminal dashboard, FastAPI web UI, realtime data fetcher, and performance tracking
helpers. It focuses on their primary data structures, asynchronous entry points,
and how runtime configuration is consumed.

## `risk_management.dashboard`
- **Data structures**: Defines the `Position`, `Order`, `Account`, and `AlertThresholds`
dataclasses that represent snapshot payloads parsed from disk or realtime
requests. These structures expose helper methods (for example `Account.exposure_pct`
and `Position.exposure_relative_to`) used when rendering alerts and aggregations.
- **Parsing and transformation**: `parse_snapshot()` normalises inbound JSON into
the dataclasses, while `evaluate_alerts()` inspects the parsed accounts against
threshold rules, and `render_dashboard()` composes a printable terminal view.
- **Async entry points**: The CLI is orchestrated through `main()` which delegates
to the asynchronous `_run_cli()` helper. `_run_cli()` optionally instantiates a
`RealtimeDataFetcher` when a realtime configuration is supplied and performs
periodic calls to `fetch_snapshot()` before rendering the dashboard.
- **Configuration touch points**: `main()` accepts command line overrides for
snapshot paths, realtime configuration files, polling intervals, and custom
endpoint behaviour. When realtime mode is enabled it loads configuration via
`load_realtime_config()` and rewrites `CustomEndpointSettings` based on CLI
flags before building the fetcher.

## `risk_management.web`
- **Data structures**: Wraps realtime access in `RiskDashboardService`, exposing
methods for fetching snapshots, placing orders, cancelling orders, closing
positions, executing kill switches, and managing portfolio or account stop-loss
state. An `AuthManager` encapsulates credential storage and password validation.
- **Async entry points**: `create_app()` wires dependencies and defines every
FastAPI coroutine handler. Key asynchronous handlers include `/api/snapshot`,
all `/api/trading/...` order and stop-loss routes, kill switch endpoints, and
report generation or download helpers that interact with `RiskDashboardService`
and `ReportManager`.
- **Configuration touch points**: `create_app()` consumes `RealtimeConfig`
attributes for authentication, Grafana embedding, reporting directories, and
LetsEncrypt challenge serving. It also persists the instantiated
`RiskDashboardService` on `app.state` so handlers can reuse the configured
`RealtimeDataFetcher` and respect any custom endpoint overrides established in
the runtime configuration.

## `risk_management.realtime`
- **Data structures**: `RealtimeDataFetcher` maintains per-account
`AccountClientProtocol` instances, a `PerformanceTracker`, and mutable state for
portfolio and account stop-loss records (including baseline balances, drawdown
percentages, and trigger timestamps). It emits snapshot dictionaries mirroring
what the dashboard expects, enriched with messages, notifications, and
performance summaries.
- **Async entry points**: The fetcher exposes coroutine methods including
`fetch_snapshot()`, `close()`, `execute_kill_switch()`, `set_portfolio_stop_loss()`,
`clear_portfolio_stop_loss()`, `set_account_stop_loss()`, `clear_account_stop_loss()`,
`place_order()`, `cancel_order()`, `cancel_all_orders()`, `close_position()`, and
`close_all_positions()`. These power both the CLI refresh loop and FastAPI
routes.
- **Configuration touch points**: Upon initialisation the fetcher applies
`CustomEndpointSettings` discovery, instantiates `CCXTAccountClient` instances
for every `RealtimeConfig.accounts` entry, seeds notification channels, and
configures `PerformanceTracker` output directories. Runtime kill switch, order,
and stop-loss commands reference `_resolve_account_client()` to ensure the
requested account exists in the active configuration.

## `risk_management.performance`
- **Data structures**: Provides the `PerformanceSnapshot` dataclass to capture a
single day's balance and timestamp and the `PerformanceTracker` class which
stores `portfolio` history and an `accounts` mapping inside `daily_balances.json`.
- **Async entry points**: The tracker is synchronous but is invoked from async
callers; `RealtimeDataFetcher.fetch_snapshot()` awaits performance summaries by
calling the synchronous `record()` method inside the event loop thread, returning
a mapping of current balances plus daily/weekly/monthly profit deltas.
- **Configuration touch points**: `PerformanceTracker` is initialised with the
reports directory derived from realtime configuration (or a default under the
module path) and respects timezone plus cut-off hour parameters to determine
when to persist new balance snapshots.
