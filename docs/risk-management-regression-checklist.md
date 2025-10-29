# Risk Management Refactor Regression Checklist

Use this checklist during release candidate validation to confirm that risk controls, analytics, and operational tooling behave as expected. Capture evidence (screenshots, logs, ticket references) for each item prior to sign-off.

## CLI Kill-Switch
- [ ] Validate `riskctl kill-switch status` displays the orchestrator node and per-venue acknowledgement timestamps.
- [ ] Trigger `riskctl kill-switch trigger --desk <desk>` and confirm broadcast receipts for every configured venue adapter.
- [ ] Issue `riskctl kill-switch reset` and verify state rollback in Redis and UI banner update within 5 seconds.
- [ ] Confirm audit log entry is persisted to `risk_audit.kill_switch` topic with correct user context and reason code.

## Order Placement & Cancellation
- [ ] Submit staged orders via `/api/risk/v2/orders/preview` and ensure responses include applied policy guardrails.
- [ ] Place live orders through CLI automation (per exchange) and confirm orchestrator propagates hedges without latency regressions.
- [ ] Cancel orders from the dashboard overrides panel and verify CLI reflects cancellation status within one polling cycle.
- [ ] Inspect venue adapters for orphaned orders after stress-testing rapid placement/cancellation loops.

## Stop-Loss Threshold Updates
- [ ] Adjust stop-loss bands in the dashboard threshold editor; confirm validation errors display when breaching policy bounds.
- [ ] Apply `PATCH /api/risk/v2/thresholds/{desk}` with staged overrides and verify Redis state + UI view stay in sync.
- [ ] Force a simulated breach to ensure alerts trigger at updated thresholds and propagate to Slack/on-call channels.
- [ ] Confirm rollbacks through `riskctl thresholds revert --desk <desk>` restore previous policy snapshots.

## Analytics Endpoints
- [ ] Hit `/api/risk/v2/analytics/pnl` with 1m cadence and confirm stream delivers incremental updates without gaps.
- [ ] Validate `/api/risk/v2/analytics/alerts` pagination and filtering parameters (desk, severity, date range).
- [ ] Confirm Grafana dashboard widgets refresh using new analytics endpoints without errors in Prometheus scrape logs.
- [ ] Ensure alert acknowledgements from the dashboard update analytics history within 2 polling intervals.

## Sign-Off
- [ ] Operations lead approval recorded in Ops ticket.
- [ ] Portfolio manager acknowledgement of UI changes captured in release readiness notes.
- [ ] Runbook updated with any deviations observed during testing.
