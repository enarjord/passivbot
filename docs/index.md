# Passivbot Documentation

Passivbot is a Python and Rust trading bot for perpetual futures markets. Start with
[Installation](installation.md), then use [Config Workflow](config_workflow.md) and
[Configuration](configuration.md) when creating or updating configs.

These guides describe the v9 development revision. See
[Releases and development versions](releases.md) to select a checkout, or the
[v8 migration guide](v8_to_v9_migration.md) to prepare an existing configuration for v9.

Common workflows:

- [Running live](live.md)
- [Migrating v8 configs](v8_to_v9_migration.md)
- [Backtesting](backtesting.md)
- [Optimizing](optimizing.md)
- [Tools](tools.md)
- [Troubleshooting](troubleshooting.md)

Reference guides:

- [Strategy parameters](config.bot.md) and [Forager selection](forager.md)
- [Risk management](risk_management.md) and [Equity Hard Stop Loss](equity_hard_stop_loss.md)
- [V9 risk model](v9_risk_model.md) and [config editor setup](config_editor.md)
- [Coin overrides](coin_overrides.md) and [suite examples](suite_examples.md)
- [Metrics](metrics.md) and [monitor output](monitor.md)
- [Container deployment](container_deployment.md) and [stock perpetuals](stock_perps.md)
- [Offline fake-live testing](fake_live.md)

Contributors: [Contributing](CONTRIBUTING.md), [engineering contracts](ai/README.md), and
[release procedure](ai/runbooks/release.md). Files under `plans/`, case studies, and dated session
summaries are historical/design material; use the maintained guides for current runtime behavior.
