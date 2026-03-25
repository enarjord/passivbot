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

## Scenario Matrix For Future Tests

These are the minimum scenarios that should be tested against every supported
policy.

### S1: Clean Panic, Flat, Cooldown Active

Facts:

1. valid panic at `P`
2. replay sees flatten
3. `now < cooldown_end`
4. `flat_now`
5. no `E`

Expected for all policies:

1. reconstructed cooldown active
2. no fresh initials
3. post-panic drawdown baseline

### S2: Clean Panic, Flat, Cooldown Expired

Facts:

1. valid panic at `P`
2. replay sees flatten
3. `now >= cooldown_end`
4. `flat_now`
5. no `E`

Expected for all policies:

1. normal operation resumes
2. post-panic drawdown baseline retained

### S3: Panic Residue, No Later Entry, Position Still Exists

Facts:

1. latest panic at `P`
2. `now < cooldown_end`
3. `pos_now`
4. no `E`

Expected for all policies:

1. unresolved panic residue
2. do not treat as operator override
3. do not resume normal entries
4. continue panic/flatten handling

### S4: Later Entry During Active Cooldown, Position Still Open

Facts:

1. latest panic at `P`
2. `now < cooldown_end`
3. `E` exists
4. `pos_now`

Expected by policy:

1. `normal`
   - collapse cooldown
   - resume normal
   - reset drawdown baseline from after `E`
2. `panic`
   - panic again
   - reset cooldown from the new panic
3. `manual`
   - do nothing on that `pside`
   - keep original cooldown
4. `tp_only`
   - block entries
   - allow closes
   - keep original cooldown
5. `graceful_stop`
   - manage existing position with `graceful_stop`
   - keep original cooldown

### S5: Later Entry During Cooldown, Position Closed Again Before Restart

Facts:

1. latest panic at `P`
2. `now < cooldown_end`
3. `E` exists
4. later non-panic close fills happen
5. `flat_now` at restart

Expected by policy:

1. `normal`
   - normal restart from post-`E` drawdown regime
2. `panic`
   - latest successful panic governs cooldown state
3. `manual`
   - original cooldown still governs
4. `tp_only`
   - original cooldown still governs
5. `graceful_stop`
   - original cooldown still governs

### S6: Later Entry Happens Only After Cooldown Expired

Facts:

1. valid panic at `P`
2. `now >= cooldown_end`
3. later entry exists, but not during active cooldown

Expected for all policies:

1. ordinary post-cooldown trading
2. cooldown-intervention policy does not apply

### S7: Multiple Panic Episodes

Facts:

1. more than one panic exists
2. older panic episodes precede the latest one

Expected for all policies:

1. latest panic episode governs active cooldown reconstruction
2. older episodes do not control the current cooldown

### S8: Same-Minute Panic Flatten And Re-Entry

Facts:

1. valid panic flatten happens
2. later entry happens in the same minute bucket

Expected for all policies:

1. replay must honor fill-event ordering, not minute buckets alone
2. result must match the same scenario with the events split into separate minutes

### S9: Restart During Active Cooldown While Opposite Side Continues

Facts:

1. one `pside` is in active reconstructed cooldown
2. opposite `pside` is allowed to trade

Expected for all policies:

1. cooldown behavior is side-local
2. only the cooled-down `pside` is blocked or altered

## Policy Table

| Scenario | `normal` | `panic` | `manual` | `tp_only` | `graceful_stop` |
|---|---|---|---|---|---|
| S1 clean panic, cooldown active, flat | wait | wait | wait | wait | wait |
| S2 clean panic, cooldown expired, flat | normal from after panic | normal from after panic | normal from after panic | normal from after panic | normal from after panic |
| S3 panic residue, no later entry, pos remains | continue panic handling | continue panic handling | continue panic handling | continue panic handling | continue panic handling |
| S4 later entry during active cooldown, pos open | resume from after first later entry | repanic and reset cooldown | do nothing, keep original cooldown | closes only, keep original cooldown | graceful-stop management, keep original cooldown |
| S5 later entry during cooldown, flat by restart | normal from post-entry regime | latest successful panic governs | original cooldown governs | original cooldown governs | original cooldown governs |
| S6 later entry only after cooldown expired | normal | normal | normal | normal | normal |
| S7 multiple panics | latest panic unless later intervention during active cooldown | latest panic | latest panic | latest panic | latest panic |
| S8 same-minute panic and re-entry | follow fill ordering | follow fill ordering | follow fill ordering | follow fill ordering | follow fill ordering |
| S9 opposite side still trading | side-local only | side-local only | side-local only | side-local only | side-local only |

## Logging Note

Cooldown logs should prefer human-readable remaining time, for example:

1. `remaining_time=1d8h20s`

instead of:

1. `remaining_seconds=116392.9`

## Test Planning Notes

Future tests should include:

1. unit tests for restart-state inference from synthetic fill timelines
2. unit tests for live cooldown intervention handlers
3. fake-live replay tests for at least:
   - S1
   - S3
   - S4
   - S8
4. fresh-process restart tests with no local latch dependency
