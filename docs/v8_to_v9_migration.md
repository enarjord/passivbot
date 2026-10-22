# Migrate v8 and v9.0 configurations to current v9

V9 replaces finite rolling HSL and unstuck loss memory with exponential decay, smooths
HSL's drawdown **ratio**, and always resumes continuous trading after a qualified finite
cooldown. Config schema v9.1.0 also moves the realized-loss gate onto that shared
net-realized memory. These changes require retuning and fresh backtests. Migration writes configuration
only; it performs no exchange requests and starts no bot.

## Review the semantic choices

| V8 setting or behavior | V9 migration |
|---|---|
| V8 `live.pnls_max_lookback_days` drives several risk consumers | Explicitly choose `live.lookback_ema_span_minutes`; no rolling duration is converted into an equivalent span |
| V9.0 `live.realized_loss_gate_lookback_minutes` | Remove it and explicitly acknowledge shared realized memory when an effective gate is active; retain the authored shared span |
| Zero/minimum-resolution or all-history rolling window | No EMA equivalent; explicit adoption of shared decay is required |
| Currency/equity-peak reconstruction inside a rolling window | Forward decaying currency drawdown, with an independently sampled ratio EMA |
| First observed drawdown seeds smoothing | Proven new HSL episodes start D=M=0 before entry economics |
| `unified` scope | `portfolio`, keeping the explicit `bot.hsl` policy |
| `restart_after_red_policy="always"` | Remove the field; restart follows finite cooldown and ordinary gates |
| `restart_after_red_policy="never"` | Explicit acknowledgement is required; permanent shutdown belongs to external operation policy |
| Cooldown classified using terminal RED | Latest pre-final-fill RED **or** final-accounting RED qualifies a fixed deadline |
| Saved optimizer fitness/checkpoints, including v9.0 | Invalid for the shared-gate evaluation contract; migrated numeric seed suggestions must be freshly evaluated |

Canonical v8 disabled policies may contain a null restart field. Migration removes
that representation only when effective HSL enablement is unambiguously false.
Canonical and legacy suite scenarios inherit root enablement unless their own
override supplies it.
Referenced coin files resolve enablement in root, file, then inline precedence;
inherited enablement is not copied into the exported coin patch.
Migration also checks fixed optimizer overrides, global mirroring and scenario
overrides in their runtime order before removing a null policy.
An enabled or ambiguous null policy requires explicit correction; authored `never`
still requires acknowledgement even when disabled.
Legacy entry EMA and cooldown aliases retain their existing canonical precedence,
independently of insertion order, when migration checks effective scenarios. This
structural normalization does not discard raw HSL authorization or resolve
conflicting HSL aliases.

An EMA span is not a half-life or an equivalent finite lookback. For span S,
`alpha=2/(S+1)` and half-life is `log(0.5)/log(1-alpha)` minutes
(span `1.0` has a zero half-life as a limiting case). The example slow span
`43200.0` and fast span `360.0` describe different memories; they do not reproduce a
30-day rolling policy. Fractional spans remain floats. See the exact numerical and
sampling rules in [V9 risk model](v9_risk_model.md).

## Preview, then write a separate file

```sh
passivbot tool migrate-config-v9 input.json \
  --lookback-ema-span-minutes 43200.0 --dry-run --diff

passivbot tool migrate-config-v9 input.json output.v9.json \
  --lookback-ema-span-minutes 43200.0 --report migration-report.json
```

The first command writes no configuration or backup. The second writes a new file and
an explicit change report. Without an output argument, the destination is
`input.v9.json`. Existing output/report files are never overwritten. Review the diff and
semantic warnings before choosing spans, allowances, thresholds and optimizer bounds.
The tool accepts versioned v8 and v9.0.0 inputs; migrate older layouts to a supported
v8 schema first. Its output uses config schema `v9.1.0`; this does not change the
product/package release version. Ordinary loading rejects old schemas even when
their retired window is absent.

For any effective active realized-loss gate, add
`--acknowledge-realized-loss-gate-memory-change`. This migration-only acknowledgment
explicitly adopts the shared span for the gate; it is not a runtime policy switch.
The audit covers the root, fixed optimizer overrides and every authored stored scenario,
including disabled suites. The report records active contexts and their effective spans.
`max_realized_loss_pct` and its existing activation/close policy are retained. Its
allowance becomes `max(0, r * (raw_balance + D) - D)`, without TWEL multiplication.
An equal numeric rolling duration is still a different policy.

V9.0.0 inputs retain their existing `lookback_ema_span_minutes`; do not pass a new
conflicting value to retune HSL or unstuck implicitly. If an active old-v9 context
has no authored shared span, choose it explicitly with `--lookback-ema-span-minutes`.
Old gate-only scenario window overrides are removed under the acknowledgment and
adopt that context's effective shared span. Global gate settings in a coin-only
override file are not allowed coin modifications and do not activate the gate.

If any authored root, coin, scenario, fixed optimizer override or referenced file contains
`never`, add `--acknowledge-never-removal` after reviewing its consequences. Disabled
policies also require acknowledgement. The tool does not silently enable indefinite
shutdown under a new name. Unsupported/ambiguous old policies fail with their exact path.

Every v8 scenario/fixed override of `pnls_max_lookback_days` needs its own explicit choice:

```sh
passivbot tool migrate-config-v9 input.json \
  --lookback-ema-span-minutes 43200.0 \
  --memory-span 'backtest.scenarios[0].overrides.live.pnls_max_lookback_days=21600.0' \
  --dry-run --diff
```

Use the exact path printed by the tool. Root choice does not guess distinct scenario
policy. Old window bounds require explicitly authored replacement
`optimize.bounds.live.lookback_ema_span_minutes` bounds; units conversion alone is not an
equivalent risk search distribution. The default span is fixed unless bounds are supplied.
Coin override files are materialized with file-then-inline precedence so moving the output
cannot accidentally load a different relative file; original files remain unchanged.
Legacy `live.coin_flags` settings and file references become explicit `coin_overrides`,
including effective scenario patches; migrated output retains no legacy file reference.
Existing nonempty `coin_overrides` keep their precedence over legacy flags, while removed
`never` policies in any authored referenced file still require acknowledgement.
Legacy strategy EMA and entry-cooldown paths are normalized separately in each patch
before merging, so inline values still win across old and current spellings.

## In-place replacement and rollback

```sh
passivbot tool migrate-config-v9 input.json \
  --lookback-ema-span-minutes 43200.0 --in-place --backup input.v8-backup.json
```

The original bytes are backed up before atomic replacement. Without `--backup`, the tool
uses `input.pre-v9.1-backup.json`. Existing backups are never replaced. Symlinks remain links;
the resolved target's mode and ownership are preserved. Parse, policy, validation and
serialization failures leave the input unchanged. Reprocessing a valid v9.1.0 configuration
is idempotent; a conflicting new span must be an explicit config edit, not another migration.

Keep the original source/runtime version, original configuration bytes and new configuration
with separate identities. A rollback pairs v8 source with the original v8 config, and v9
source with a v9 config. Changing code alone does not translate risk memory, existing
exposure, exchange orders or cooldown facts. Any live rollout/rollback needs its separately
authorized account-specific procedure.

## Validate the intended configuration

Reload the output through the ordinary v9 loader, then run focused unit/fake-exchange
checks and backtest the exact proposed configuration with its intended market data, fees
and scenarios. Keep private configurations and result reports private. Compare ordinary
account return, drawdown, recovery/underwater time, exposure, fees, stops and reentries;
changed HSL signal metrics are not a common account-equity ruler.

The shared slow span defaults to `43200.0` minutes. Live reconstruction targets five slow
spans and retains the active cooldown horizon, capped by `live.risk_bootstrap_max_lookback_minutes` (default/max `129600.0`, 90 days).
Enabled cooldowns exceeding the cap fail live capability validation.
Consequently the default slow-memory target is capped, with observable unknown-prefix/truncation
information. This cap is acquisition policy, not the amount of exponential memory.
Unstuck and the realized-loss gate use one declared realized-memory initializer.
A bounded authoritative suffix with an unknown prefix starts at zero with explicit
attenuation diagnostics; it is not an exact or safe-history proof. Losing that initializer
declares a new bounded bootstrap rather than silently preserving prior derived memory.

Fresh searches may use migrated numeric strategy suggestions. They must not resume old
checkpoints or reuse their fitness. Offline tests and backtests do not establish live
execution readiness. See [HSL validation](hsl_testing.md) and the
[live validation checklist](hsl_live_validation.md) for separate evidence boundaries.
