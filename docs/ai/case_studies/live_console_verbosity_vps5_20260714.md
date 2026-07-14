# VPS5 Console Verbosity Baseline, 2026-07-14

This case study records the production evidence used to sharpen the default console contract in
`../logging_policy.md`. It is historical rationale, not a runtime target or a claim about later
versions.

## Method

The sample used read-only tmux scrollback from five live bots on VPS5. No process was signalled or
restarted, and no exchange request was initiated for the audit. ANSI control sequences were
removed. A logical record began with the normal UTC timestamp/level/exchange prefix; continuation
rows up to the next timestamp were counted as physical terminal rows.

The current-run sample covered `2026-07-14T18:04:50Z` through `19:12:29Z`. A steady-state slice
excluded the first ten minutes and covered `18:15:00Z` through `19:13:51Z`. The host was not fully
healthy throughout: KuCoin experienced real request timeouts. That makes the sample useful for
both routine-volume and incident-projection analysis.

## Results

Across the full current-run sample:

- 2,010 logical records and 3,177 displayed rows in 1.128 hours
- 1,783 records/hour and 2,818 rows/hour across five bots
- about 357 records/hour and 564 rows/hour per bot
- 675 `[warmup]`, 461 `[balance]`, and 282 `[boot]` INFO records; these three routine families were
  70.5% of all logical records
- 688 displayed rows came from only 19 error or degraded-health records because repeated timeout
  tracebacks expanded to roughly 83-86 rows each

The post-startup slice remained at 1,543 logical records/hour and 2,463 displayed rows/hour across
the five bots. Dominant recurring behavior included:

- per-symbol warmup bundles repeated after the bot was already ready
- `[boot]` candle-index rebuild start/completion lines during normal maintenance
- REST balance lines for small positive and negative mark-to-market changes while the snapshot
  anchor remained unchanged
- per-cycle forager selections and staged-refresh timing detail
- multi-row candle/EMA context lists
- repeated full KuCoin timeout tracebacks in both operation-error and degraded-health records

The tracebacks were operationally important, but printing each full stack obscured the concise
error signature, repeat count, and recovery state. Conversely, fills, position changes, order
outcomes, periodic health summaries, and safety transitions were a small fraction of the volume and
remained the most useful facts to retain.

## Conclusion

The main problem was not too many structured events. It was projection: routine maintenance and
continuous numeric jitter were admitted to an operator sink, while a small number of incidents
expanded into hundreds of terminal rows. The durable remedy is event-specific transition and
aggregation policy plus bounded one-line incident summaries. A global rate limiter would be less
safe because it could hide an unrelated fill, exchange write, or risk transition during a noisy
period.
