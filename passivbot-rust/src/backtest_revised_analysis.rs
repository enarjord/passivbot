//! Observed backtest performance; never supplies revised HSL trading permission.
use super::*;

impl Backtest<'_> {
    pub(super) fn record_revised_analysis(&mut self, k: usize, at_fill_boundary: bool) {
        let Some(&timestamp) = self.equities.timestamps_ms.last() else {
            return;
        };
        // Account equity may have been clamped to the liquidation floor.
        // Strategy performance retains the actual marked exposure and net fills.
        let side_upnl = [
            self.unrealized_pnl_pside(LONG, k),
            self.unrealized_pnl_pside(SHORT, k),
        ];
        let upnl = side_upnl[LONG] + side_upnl[SHORT];
        self.strategy_equity_series
            .push(self.backtest_params.starting_balance + self.pnl_cumsum_running_net + upnl);
        for side in [LONG, SHORT] {
            let value = self.backtest_params.starting_balance
                + self.pnl_cumsum_running_net_pside[side]
                + side_upnl[side];
            self.strategy_equity_series_pside[side].push(value);
            self.strategy_equity_timestamps_ms_pside[side].push(timestamp);
        }
        // EMA diagnostics describe the actual enabled signal scopes. Unified
        // contributes once to the portfolio, never to invented side controllers.
        let mut emas = [0.0_f64; 3];
        for scope in &self.revised_hsl_scopes {
            if let Some(decision) = &scope.result.decision {
                emas[0] = emas[0].max(decision.ema);
                if let Some(side) = scope.side {
                    emas[side + 1] = emas[side + 1].max(decision.ema);
                }
            }
        }
        // A fill-depleted account stops at the bar's open. There is no later
        // bar-close evaluation or elapsed minute to include in its report.
        let observed_at = timestamp
            + if at_fill_boundary {
                0
            } else {
                self.interval_ms
            };
        self.revised_hsl_report
            .record_bar_signals(observed_at as i64, emas);
    }

    pub(super) fn revised_strategy_metrics(&self) -> StrategyEquityMetricsBundle {
        StrategyEquityMetricsBundle {
            overall: self.strategy_equity_metrics_from_series(
                &self.strategy_equity_series,
                Some(&self.revised_hsl_report.signal_emas[0]),
                None,
            ),
            long: self.strategy_equity_metrics_from_series(
                &self.strategy_equity_series_pside[LONG],
                Some(&self.revised_hsl_report.signal_emas[1]),
                Some(&self.strategy_equity_timestamps_ms_pside[LONG]),
            ),
            short: self.strategy_equity_metrics_from_series(
                &self.strategy_equity_series_pside[SHORT],
                Some(&self.revised_hsl_report.signal_emas[2]),
                Some(&self.strategy_equity_timestamps_ms_pside[SHORT]),
            ),
        }
    }
}
