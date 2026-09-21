//! Observed backtest performance; never supplies revised HSL trading permission.
use super::*;

impl Backtest<'_> {
    pub(super) fn record_revised_analysis(&mut self, k: usize) {
        let Some(&equity) = self.equities.usd_total_equity.last() else {
            return;
        };
        let Some(&timestamp) = self.equities.timestamps_ms.last() else {
            return;
        };
        let upnl = equity - self.balance.usd_total_balance;
        self.strategy_equity_series
            .push(self.backtest_params.starting_balance + self.pnl_cumsum_running_net + upnl);
        for side in [LONG, SHORT] {
            let value = self.backtest_params.starting_balance
                + self.pnl_cumsum_running_net_pside[side]
                + self.unrealized_pnl_pside(side, k);
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
        self.revised_hsl_report
            .record_bar_signals((timestamp + self.interval_ms) as i64, emas);
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
