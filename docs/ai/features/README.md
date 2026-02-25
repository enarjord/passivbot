# Feature Documentation

This directory contains learnings and implementation details for specific Passivbot features. Consult these before working on the relevant subsystem.

## Available Feature Docs

| Feature | File | Description |
|---------|------|-------------|
| Stock Perpetuals | [stock_perps.md](stock_perps.md) | HIP-3 stock perps on Hyperliquid (symbol mapping, isolated margin, data sources) |
| Candlestick Manager | [candlestick_manager.md](candlestick_manager.md) | OHLCV data fetching, caching, and synthetic candle handling |
| Fill Events Manager | [fill_events_manager.md](fill_events_manager.md) | Fill tracking, PnL computation, exchange-specific pagination |
| Balance Routing | [balance_routing.md](balance_routing.md) | Raw vs hysteresis-snapped balance semantics and migration guidance |

## When to Consult Feature Docs

- Before modifying code in a feature area
- When debugging issues in a feature
- When adding new functionality that interacts with a feature
- When onboarding to understand a subsystem

## Adding New Feature Documentation

Create a new file when:
1. A feature has non-obvious implementation details
2. Exchange-specific quirks affect the feature
3. A debugging session revealed important learnings
4. The feature has configuration nuances users need to understand

### Template

```markdown
# Feature Name

Brief description of what this feature does.

## Key Concepts

Terminology and mental model for understanding the feature.

## Implementation Details

How it works internally. Reference specific files and line numbers.

## Configuration

Relevant config options with examples.

## Known Issues / Quirks

Exchange-specific behaviors, edge cases, limitations.

## Testing

How to test changes to this feature.

## Debugging Tips

Common issues and how to investigate them.
```
