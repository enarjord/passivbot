# Risk management interfaces inventory

## Command line entry points
| Command | Module | Description | Key options |
| --- | --- | --- | --- |
| `python -m risk_management.dashboard` | [`risk_management/dashboard.py`](../risk_management/dashboard.py) | Renders the terminal dashboard using a static snapshot or realtime feed. | `--config`, `--realtime-config`, `--interval`, `--iterations`, `--custom-endpoints` |
| `python -m risk_management.web_server` | [`risk_management/web_server.py`](../risk_management/web_server.py) | Launches the FastAPI risk dashboard with optional TLS and Let's Encrypt automation. | `--config`, networking flags, `--custom-endpoints`, TLS parameters, LetsEncrypt options |
| `python risk_management/scripts/hash_password.py` | [`risk_management/scripts/hash_password.py`](../risk_management/scripts/hash_password.py) | Generates bcrypt password hashes for dashboard authentication records. | Optional `password` positional argument (otherwise uses interactive prompt) |

## HTTP API surface
All endpoints require an authenticated session unless explicitly noted.

| Method & Path | Description | Backend call |
| --- | --- | --- |
| `GET /login` | Render the login form. | Template rendering only |
| `POST /login` | Authenticate a user and open a session. | `AuthManager.authenticate()` |
| `POST /logout` | Clear session cookie and redirect to login. | Session middleware |
| `GET /` | Render the dashboard view model and Grafana embeds. | `RiskDashboardService.fetch_snapshot()` + `build_presentable_snapshot()` |
| `GET /trading-panel` | Render the order management panel. | `RiskDashboardService.fetch_snapshot()` |
| `GET /api/snapshot` | Return a JSON snapshot with optional filtering, paging, and sorting. | `RiskDashboardService.fetch_snapshot()` |
| `GET /api/trading/accounts/{account}/order-types` | List order types supported by a configured account. | `RiskDashboardService.list_order_types()` |
| `POST /api/trading/accounts/{account}/orders` | Place an order (market/limit/etc.) for a symbol. | `RiskDashboardService.place_order()` |
| `DELETE /api/trading/accounts/{account}/orders/{order_id}` | Cancel a specific order (optional symbol/params payload). | `RiskDashboardService.cancel_order()` |
| `POST /api/trading/accounts/{account}/positions/{symbol}/close` | Close a single symbol position. | `RiskDashboardService.close_position()` |
| `POST /api/trading/accounts/{account}/orders/cancel-all` | Cancel all orders, optionally filtered by symbol. | `RiskDashboardService.cancel_all_orders()` |
| `POST /api/trading/accounts/{account}/positions/close-all` | Close all open positions, optionally filtered by symbol. | `RiskDashboardService.close_all_positions()` |
| `POST /api/trading/accounts/{account}/stop-loss` | Create or update an account stop-loss threshold. | `RiskDashboardService.set_account_stop_loss()` |
| `GET /api/trading/accounts/{account}/stop-loss` | Retrieve the current account stop-loss state. | `RiskDashboardService.get_account_stop_loss()` |
| `DELETE /api/trading/accounts/{account}/stop-loss` | Clear the account stop-loss configuration. | `RiskDashboardService.clear_account_stop_loss()` |
| `POST /api/trading/portfolio/stop-loss` | Create or update the portfolio-level stop-loss. | `RiskDashboardService.set_portfolio_stop_loss()` |
| `GET /api/trading/portfolio/stop-loss` | Retrieve the portfolio stop-loss state. | `RiskDashboardService.get_portfolio_stop_loss()` |
| `DELETE /api/trading/portfolio/stop-loss` | Clear the portfolio stop-loss configuration. | `RiskDashboardService.clear_portfolio_stop_loss()` |
| `POST /api/kill-switch` | Trigger the global kill switch across all accounts. | `RiskDashboardService.trigger_kill_switch()` |
| `POST /api/accounts/{account}/kill-switch` | Trigger the kill switch for a specific account (optional `symbol` query). | `RiskDashboardService.trigger_kill_switch()` |
| `POST /api/accounts/{account}/positions/{symbol}/kill-switch` | Trigger the kill switch for a specific symbol within an account. | `RiskDashboardService.trigger_kill_switch()` |
| `GET /api/accounts/{account}/reports` | List generated CSV reports for an account. | `ReportManager.list_reports()` |
| `POST /api/accounts/{account}/reports` | Generate a new CSV snapshot report for an account. | `ReportManager.create_account_report()` |
| `GET /api/accounts/{account}/reports/{report_id}` | Download a report by identifier. | `ReportManager.get_report_path()` |

## Acceptance criteria usage
The tables above enumerate the external touch points that should be exercised by
acceptance and regression tests. They cover realtime safety controls (kill
switches and stop-loss APIs), order lifecycle management, reporting, and the CLI
tools operators use to drive the system.
