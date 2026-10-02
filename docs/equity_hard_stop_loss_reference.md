# Equity Hard Stop Loss reference

Use the canonical [HSL guide](equity_hard_stop_loss.md) for the shared formula, scopes,
best-effort evidence reconstruction and lifecycle behavior. Configuration and migration
are specified in [HSL configuration](configuration.md#hsl-configuration), and
[metrics](metrics.md) documents reported values.

The previous multi-tier controller, terminal accumulated-drawdown threshold,
position-during-cooldown policy, grace fallback and durable emergency commitments
are retired. There is one current GREEN/RED decision per configured scope. No
previous panic decision is retained after current recovery or reconstructible evidence loss.
