# HSL cooldown contract

The sole HSL engine uses the last episode's terminal drawdown and the current flat
scope to reconstruct cooldown. Renewed exposure clears cooldown. Retained history
and cooldown both expire at the configured lookback boundary.

See [HSL episode lifecycle](equity_hard_stop_loss.md) and the
[canonical feature contract](ai/features/equity_hard_stop_loss.md). The retired
position-during-cooldown policy, tier overlays and terminal threshold do not apply.
