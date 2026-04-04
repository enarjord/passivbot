# Equity Hard Stop Loss Cooldown Contracts

This file defines the intended target contract for
`live.hsl_position_during_cooldown_policy`.

It is the normative spec for future implementation and regression tests. It may
be stricter or cleaner than the current implementation at any given moment.

See also:

1. [Equity Hard Stop Loss](equity_hard_stop_loss.md)
2. [Equity Hard Stop Loss Reference](equity_hard_stop_loss_reference.md)
3. [Configuration](configuration.md)

## Goal

After a valid RED panic on one `pside`, restart behavior must be derivable from:

1. current exchange state
2. fill history
3. config
4. current time

No local persistent marker may be required.

## Core Definitions

For one `pside`, define:

1. `P`
   - latest `close_panic_{pside}` fill timestamp
2. `cooldown_end`
   - `P + hsl_cooldown_minutes_after_red`
3. `E`
   - first non-panic entry fill on that `pside` after `P`
4. `flat_now`
   - current exchange-truth flatness for that `pside`
5. `pos_now`
   - current exchange-truth non-flat position for that `pside`

Important distinction:

1. unresolved panic residue
   - panic close exists
   - `pos_now` is non-flat
   - no later entry `E`
2. cooldown intervention
   - panic close exists
   - later entry `E` exists

These must not be confused.

## Global Rules

Apply in order for one `pside`:

1. If no `P` exists:
   - ordinary HSL replay
   - no reconstructed cooldown
2. If `now >= cooldown_end`:
   - cooldown is over
   - restart normal operation
   - HSL drawdown baseline is post-panic
3. If `now < cooldown_end` and no `E` exists:
   - cooldown episode is still active
   - if `flat_now`: wait
   - if `pos_now`: unresolved panic residue, not operator intervention
4. If `now < cooldown_end` and `E` exists:
   - cooldown intervention happened
   - apply `hsl_position_during_cooldown_policy`

Shared invariants:

1. after a valid panic flatten, HSL drawdown tracking resets from after the panic
2. while cooldown is active and the side is flat, the bot must not open fresh initials on that `pside`
3. unresolved panic residue must still allow the bot to continue panic/flatten handling

## Supported Policies

Use standard Passivbot mode names:

1. `normal`
2. `panic`
3. `tp_only`
4. `graceful_stop`
5. `manual`

Legacy name mapping:

1. `resume_normal_reset_drawdown` -> `normal`
2. `repanic_reset_cooldown` -> `panic`
3. `manual_quarantine` -> `manual`
4. `repanic_keep_original_cooldown` -> removed
5. `graceful_stop_keep_cooldown` -> replaced by standard `graceful_stop`

## Policy Semantics During Active Cooldown

These rules apply only when `now < cooldown_end` and a later entry `E` exists.

### `normal`

Intent:

1. explicit operator override
2. collapse cooldown immediately
3. restart HSL drawdown tracking from the intervention regime

Contract:

1. collapse cooldown at `E`
2. resume normal operation
3. reset HSL drawdown baseline from after `E`
4. replay all later fills normally from that boundary

Important:

1. while the side is still flat and no `E` exists, do not open fresh initials
2. reset boundary is the first post-panic non-panic entry `E`
3. not the latest DCA add
4. not a local marker

### `panic`

Intent:

1. any intervention position during cooldown is unacceptable
2. panic-close it again
3. start a fresh cooldown from the new panic

Contract:

1. if intervention exists, panic-close it again
2. once flat, reset drawdown baseline from after the new panic
3. restart cooldown from the new panic timestamp

Note:

1. this intentionally replaces the old `repanic_keep_original_cooldown`
2. reset-on-repanic is simpler and easier to reason about

### `manual`

Intent:

1. operator owns the side during the rest of the cooldown
2. bot does nothing on that side
3. bot resumes normal automatically after cooldown ends

Contract:

1. create no new orders on that `pside`
2. cancel no orders on that `pside`
3. place no closes on that `pside`
4. keep original `cooldown_end`
5. once `now >= cooldown_end`, resume normal operation

### `tp_only`

Intent:

1. operator controls entries during cooldown
2. bot may still manage closes
3. cooldown deadline remains original

Contract:

1. block fresh initials
2. do not create new entry orders
3. allow close management according to normal TP/close logic
4. keep original `cooldown_end`
5. once flat before cooldown expiry, continue waiting until cooldown ends
6. once cooldown ends, resume normal operation

### `graceful_stop`

Intent:

1. bot may manage an existing position conservatively during cooldown
2. but it must not open a new position while cooldown is active
3. original cooldown remains authoritative

Contract:

1. block fresh initials
2. if a position exists, manage it with normal `graceful_stop` semantics
3. keep original `cooldown_end`
4. if the position closes before cooldown ends, continue waiting flat
5. once cooldown ends, resume normal operation

## Unresolved Panic Residue

If `now < cooldown_end`, `pos_now` is non-flat, and no later entry `E` exists:

1. this is not intervention
2. do not apply `normal`, `manual`, `tp_only`, or `graceful_stop` semantics
3. continue panic/flatten handling
4. the bot must still be able to succeed in flattening

This is required so the bot is not confused by a failed or partial panic flatten.
