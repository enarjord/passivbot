# Agent Documentation Index

This directory contains documentation for AI coding assistants working on Passivbot.

## Quick Reference

| File | When to Read | Description |
|------|--------------|-------------|
| [principles.yaml](principles.yaml) | Always | Core conventions: terminology, error handling, design principles |
| [architecture.md](architecture.md) | New to codebase | Python/Rust division, component overview, data flows |
| [commands.md](commands.md) | Need to run something | Setup, testing, backtesting, optimization commands |
| [pitfalls.md](pitfalls.md) | Before implementing | Common mistakes and how to avoid them |

## Domain Knowledge

| File | When to Read | Description |
|------|--------------|-------------|
| [exchange_api_quirks.md](exchange_api_quirks.md) | Working on exchange code | Known API limitations and workarounds |
| [logging_guide.md](logging_guide.md) | Working on logging | Log level definitions and examples |

## Learning & History

| File | When to Read | Description |
|------|--------------|-------------|
| [debugging_case_studies.md](debugging_case_studies.md) | Complex investigation | Detailed debugging sessions as reference |
| [decisions.md](decisions.md) | Understanding "why" | Architectural decision log |

## Feature Documentation

See [features/README.md](features/README.md) for feature-specific learnings.

| File | Description |
|------|-------------|
| [features/stock_perps.md](features/stock_perps.md) | HIP-3 stock perpetuals on Hyperliquid |
| [features/candlestick_manager.md](features/candlestick_manager.md) | OHLCV data fetching and caching |
| [features/fill_events_manager.md](features/fill_events_manager.md) | Fill tracking and PnL computation |
| [features/balance_routing.md](features/balance_routing.md) | Raw vs snapped balance contract and migration rules |

## Adding New Documentation

**When to add new docs:**
- **Exchange quirks**: Add to `exchange_api_quirks.md` (or create `{exchange}_quirks.md` if extensive)
- **Complex debugging**: Add case study to `debugging_case_studies.md`
- **Architectural decision**: Add entry to `decisions.md`
- **Common pitfall discovered**: Add entry to `pitfalls.md`
- **New feature**: Create `features/{feature}.md` and update features/README.md

**Documentation principles:**
- Keep files focused on a single topic
- Use consistent formatting (see existing files as templates)
- Include concrete examples, not just abstract guidelines
- Reference code locations where applicable (`file.py:123`)
