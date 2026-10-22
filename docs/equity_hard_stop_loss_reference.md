# Equity Hard Stop Loss reference

Use [V9 risk model](v9_risk_model.md) for precise numerical/sampling/bootstrap rules and
the canonical [HSL guide](equity_hard_stop_loss.md) for the shared formula, scopes,
bounded factual reconstruction and lifecycle behavior. Configuration and migration
are specified in [HSL configuration](configuration.md#hsl-configuration), and
[metrics](metrics.md) documents reported values.

The previous multi-tier controller, terminal accumulated-drawdown threshold,
position-during-cooldown policy, grace fallback and durable emergency commitments
are retired. There is one current GREEN/RED decision per configured scope. No
previous panic decision supplies separate authority. A factual qualifying flatten creates
a finite deadline which later GREEN does not cancel.
