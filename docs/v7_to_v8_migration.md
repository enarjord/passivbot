# Migrating v7 trailing-grid configs to v8

V8 is a breaking release. The migration helper preserves the shape and parameters of the v7
trailing-grid strategy through the deprecated `trailing_grid_v7` compatibility strategy. It does
not convert the strategy into `trailing_martingale`, and it does not promise identical fills or
performance across the complete v7 and v8 runtimes.

Treat a migrated config as a new deployment: preserve the exact v7 revision, inspect every
migration decision, compare backtests over identical data, and review the resulting v8 config
before live use.

## What the compatibility strategy preserves

The tool moves v7 strategy fields under:

```text
bot.<side>.strategy.trailing_grid_v7
```

and sets:

```text
live.strategy_kind = "trailing_grid_v7"
```

This preserves v7 trailing/grid parameter semantics, including pure-grid, pure-trailing, and mixed
ratio modes. V8 still uses its current shared execution, risk, forager, unstuck, HSL, and backtest
runtime. Those surrounding systems can cause a migrated config to trade differently even when the
strategy parameters are preserved exactly.

## Before migrating

1. Keep the exact v7 release or commit used by the existing deployment.
2. Back up the original config.
3. Run a v7 benchmark over a representative interval and retain the complete result directory,
   including `config.json`, `analysis.json`, `dataset.json`, `fills.csv`, and
   `balance_and_equity.csv.gz`.
4. Prefer the normalized `config.json` written by the v7 backtest artifact as migration input. This
   separates v7 loader normalization from v8 migration behavior.
5. Record the exchange set, approved coins, requested dates, fees, starting balance, and dataset
   cache identity.

## Run the migration

```bash
passivbot tool migrate-config-v7 \
  path/to/config_v7.json \
  path/to/config_v8_trailing_grid_v7.json
```

The terminal prints a concise action summary. A complete JSON report is always written beside the
requested output using the suffix `.migration-report.json`. Override that location with
`--report`, or print the complete report to stdout with `--json`. Input, output, and report paths
must be distinct; the tool refuses an in-place migration so the v7 source is preserved.

Migration exit statuses are:

- `0`: a canonical output was written. If `--allow-manual-review-output` was used, the report status
  remains `unsafe_manual_review_output_written` and the file still requires manual review.
- `1`: manual-review or unsupported fields remain and no output was written.
- `2`: the migrated result failed canonical v8 validation and no output was written.

`--allow-manual-review-output` bypasses the unresolved-field write guard only. It never permits a
schema-invalid canonical config to be written.

## Decisions that deserve particular attention

### Warmup cap

V7 could contain both `live.max_warmup_minutes` and `backtest.max_warmup_minutes`. V8 has one shared
`live.max_warmup_minutes` value:

- A backtest-only v7 value is moved automatically.
- Matching live and backtest values are collapsed automatically.
- Conflicting values require manual review; the migrated candidate keeps the v7 live value.

### Zero exposure-enforcer thresholds

When a v7 WEL or TWEL enforcer threshold is zero or negative, migration disables the corresponding
v8 enforcer. A non-positive TWEL threshold also disables
`total_exposure_entry_gate_enabled`, because the v8 entry gate requires a positive threshold.

### Inserted v8 defaults

Fields that did not exist in v7 are listed under `inserted_v8_defaults`. Review risk, HSL, unstuck,
forager, and execution defaults even when the strategy subtree migrated cleanly. Mechanical field
moves are listed separately and normally need less attention.

### WEL excess allowance

V8 defaults to bounded WEL excess allowance. If the v7 raw allowance could put one symbol above the
side TWEL, the report explains the clamp and the `legacy_raw` alternative. Do not select
`legacy_raw` merely to make one backtest look more like v7; it intentionally permits the old
unclamped exposure and must be reviewed as a risk-policy decision.

## Compare v7 and v8 artifacts

Run v8 over the same requested dates, exchanges, coins, fees, balance, and cached market data, then
compare the two completed result directories:

```bash
passivbot tool compare-backtests \
  path/to/v7/result \
  path/to/v8/result \
  --output path/to/comparison.json
```

The comparator reports evidence; it does not label a deployment safe or unsafe. It checks:

- dataset identity and mismatches
- ADG, drawdown, recovery, holding-time, and high-exposure metrics
- final-equity and equity-path differences
- fill counts and fill-type redistribution
- exact fill-row and `(timestamp, coin, fill type)` event-match ratios

Exact fill-row equality is a strict measure. Small price or quantity differences can propagate
through later position state, so the event-match ratio and fill-type changes are usually more
useful for understanding whether the strategy is taking the same kinds of actions.

## What users should expect

A controlled study compared v7.11.0 (`a922ac548623564d0036a86c60b152c7992e3347`) with the
2026-07-16 master revision (`deee460a18f1f532a4c3a4c6e89a4befd5469d2a`) using seven
configurations spanning single-coin and five-coin cases over 2025-01-01 through 2026-07-10. Every
compared pair had the same dataset identity.

- With exposure enforcement and unstuck neutralized, XMR pure-grid and pure-trailing cases had
  98% or greater matching fill events, final equity within 0.3%, and ADG within 0.6%.
- A normal mixed XMR config and a pure-trailing XMR config were also close.
- Risk-heavy pure-grid and forager cases diverged materially through WEL/TWEL auto-reduction,
  unstuck, position selection, and exposure state.
- One portfolio case changed maximum holding time from about 19 days to about 176 days while final
  equity differed by only about 1.3%.

These observations describe that bounded sample, not a universal tolerance. Portfolio and forager
configs, strong exposure enforcement, and long-held positions are the highest-priority cases for
multiple-window testing.

## Review checklist

Before live use, confirm all of the following:

1. The report's canonical validation status is `ok`.
2. Every manual-review and dropped field has an explicit disposition.
3. Inserted risk, HSL, unstuck, forager, and execution defaults are intentional.
4. V7 and v8 backtests use comparable datasets.
5. Drawdown, high-exposure duration, maximum holding time, and fill-type changes are acceptable;
   do not review only ADG or final balance.
6. Any `legacy_raw` or entry-gate choice reflects the intended risk contract rather than a search
   for historical identity.
7. The normal new-deployment review is complete before starting a live bot.
