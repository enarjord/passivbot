# Equity Hard Stop Loss Risks

Hard stop loss (HSL) is reconstructed from exchange state plus config. Local
files may speed replay or improve diagnostics, but they are not authoritative
trading state. A fresh VPS with the same exchange account and config must be
able to reconstruct the same HSL decision.

## History Reinterpretation

HSL replay does not infer user intent behind account transfers or config
changes. The following can reinterpret historical drawdown:

- deposits and withdrawals
- balance overrides
- switching `live.hsl_signal_mode`
- enabling HSL on an account with existing fill history
- changing `bot.long/short.risk.total_wallet_exposure_limit`
- changing `bot.long/short.risk.n_positions`
- changing HSL thresholds, cooldown, or restart policy

This is intentional. The bot should not guess whether a balance change was a
transfer, a realized trading result, or an operator baseline reset. Treat HSL
config changes on live accounts as risk changes, not harmless formatting edits.

## Practical Guidance

- Run `passivbot tool live-config-preflight` before enabling or changing HSL.
- Use `passivbot tool hsl-startup-preview` where available to inspect
  reconstructed HSL state before live trading.
- Be extra cautious when enabling HSL on an account that was previously traded
  without HSL.
- If account deposits or withdrawals happened inside the HSL lookback window,
  review reconstructed drawdown before trusting startup behavior.
- Do not use `live.balance_override` with account-level HSL modes
  (`unified` or `pside`). Use `coin` mode or remove the override.

## Current Limitation

HSL replay currently models fills, positions, prices, and current balance. It
does not build a separate authoritative transfer ledger. Future diagnostics may
detect suspicious balance jumps, but trading behavior should remain derived
from exchange state plus config unless an explicit stateless contract is added.

## Incomplete fill history and the override flag

With `restart_after_red_policy=always`, HSL startup tolerates missing
pre-episode fill coverage when the current episode is provable from covered
fills, because `always` ignores historical no-restart evidence. `threshold`
and `never` require full configured lookback coverage.

`--hsl-accept-incomplete-history` is a dangerous per-run CLI flag that starts
the bot on incomplete HSL evidence for any policy. While it is active,
panic, cooldown, and no-restart decisions may be wrong. Pass it on the
command line for the specific run that needs it. Persisting it in config
files does not work: any `hsl_accept_incomplete_history: true` found in a
config file is stripped at load time with a critical log, before CLI
overrides are applied, so the waiver can never survive a restart.
