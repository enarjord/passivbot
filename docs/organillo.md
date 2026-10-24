# Historical coin eligibility (Organillo)

Set `backtest.organillo_csv` to an externally prepared daily eligibility CSV to use historical
coin selection in backtests, CPU optimization or native GPU optimization. Leave it `null` for
ordinary behavior. Passivbot does not compute market-cap, volume, category or ranking rules.

```csv
date,BTC,ETH,SOL
2025-01-01,1,1,0
2025-01-02,1,0,1
2025-01-03,0,0,0
```

Headers must exactly match canonical coin identities in the resolved approved universe. Dates
must be unique, increasing UTC days. Supply every operational date, including execution candles;
blank cells, missing dates, duplicate headers and values other than `0`/`1` raise an error.
A row takes effect at 00:00 UTC. Candle warmup before the requested start and the final unused
execution-lookahead candle are inert. That final candle needs no additional CSV date; every
candle actually consumed by the replay still requires coverage.

`1` permits ordinary Forager selection. `0` applies graceful stop: existing positions retain DCA,
closes, unstuck and protective management; once flat they cannot reopen. Pending initial orders
are checked against eligibility on their execution candle. Approved/ignored coins, side settings,
readiness and other risk controls still apply. Candle preparation loads the period union, subject
to the existing approved universe. Include only intended exact approved coin identities in the CSV.

With `backtest.dynamic_wel_by_tradability=true`, eligibility contributes to the existing non-shrinking
per-side slot denominator: the maximum simultaneous eligible count seen so far, capped by configured
positions. A rotating selection does not accumulate a count of distinct past coins. Retired positions
retain management budgets. With this option false, the configured denominator stays fixed.

Validate formatting without loading candles:

```bash
passivbot tool organillo-validate selection.csv
passivbot tool organillo-validate selection.csv --coins BTC ETH SOL \
  --start-date 2025-01-01 --end-date 2025-01-04
```

The optional end date is exclusive. Full preparation also validates coin mapping and coverage
against actual candle timestamps. The tool cannot verify historical correctness of the external
selection decisions. Avoid future information in the tool that creates the CSV.

Preparation records `backtest.organillo_sha256`. Reusing saved fitness with different CSV contents
is rejected. To intentionally select a new source (including a scenario override), clear the old
hash to `null` and prepare again. Inputs are immutable for the run and shared across candidates;
eligibility is not an optimizer gene. Keep a supplied file available during preparation.
Initially ineligible days still count toward equity, growth, fill-rate and HSL observation periods
once ordinary candle readiness and configured position capacity are available. An all-zero schedule
therefore produces an empty-trading replay over that period rather than dropping its equity ledger.
Set the input on `backtest` or a scenario; optimizer `fixed_runtime_overrides` cannot replace it.

Native GPU scenarios retain the existing limit of 64 distinct loaded coins. Organillo does not
expand capacity or change existing backend restrictions. [Detailed contract](ai/features/organillo.md).
