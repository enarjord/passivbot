// Rust-owned selected observations. All finishing formulas live in core_metrics.rs.
// This state observes replay and must never participate in trading decisions.
#ifndef PASSIVBOT_CORE_REQUEST_MASK
#define PASSIVBOT_CORE_REQUEST_MASK 0u
#endif
#ifndef PASSIVBOT_CORE_HALF_LIFE_DAYS
#define PASSIVBOT_CORE_HALF_LIFE_DAYS 180.0f
#endif
#define CORE_WANTS(ID) ((PASSIVBOT_CORE_REQUEST_MASK & (1u << CORE_ID_##ID)) != 0u)

struct CoreGpuState {
// CORE_OBSERVATION_FIELDS
    float fill_mark_price[MAX_COINS];
    float fill_mark_step[MAX_COINS];
    float pending_equity;
    float pending_step;
    float previous_equity;
    float previous_step;
    float peak;
    float peak_step;
    float underwater_start_step;
    float daily_previous_equity;
    float pending_gross_notional;
    float pending_utilization_long;
    float pending_utilization_short;
    float drawdown_correction;
    float squared_correction;
    float turnover_correction;
    float profit_correction;
    float loss_correction;
    float gross_correction;
    float util_long_correction;
    float util_short_correction;
    float pre_fill_balance;
    float current_fill_step;
    float realized_long;
    float realized_short;
    float realized_long_correction;
    float realized_short_correction;
    float ew_log_correction;
    float ew_days_correction;
    bool has_pending;
};

inline void core_add(float value, thread float& total, thread float& correction) {
    const float adjusted = value - correction;
    const float next = total + adjusted;
    correction = (next - total) - adjusted;
    total = next;
}

inline void core_increment(thread float& count) {
    // Never permit a float32 integer counter to silently saturate at 2^24.
    count = count < 16777216.0f ? count + 1.0f : NAN;
}

inline CoreGpuState init_core_gpu(float initial_equity, int start_step) {
    CoreGpuState s;
// CORE_INITIALIZE_FIELDS
    s.start_step = float(start_step);
    s.end_step = float(start_step);
    s.initial_equity = initial_equity;
    s.last_equity = initial_equity;
    s.pending_equity = initial_equity;
    s.pending_step = float(start_step);
    s.previous_equity = initial_equity;
    s.previous_step = -1.0f;
    s.peak = initial_equity;
    s.peak_step = float(start_step);
    s.underwater_start_step = -1.0f;
    s.daily_first_step = s.daily_last_step = -1.0f;
    s.daily_previous_equity = 0.0f;
    s.last_fill_step = float(start_step);
    s.pending_gross_notional = s.pending_utilization_long = s.pending_utilization_short = 0.0f;
    s.drawdown_correction = s.squared_correction = s.turnover_correction = 0.0f;
    s.profit_correction = s.loss_correction = s.gross_correction = 0.0f;
    s.util_long_correction = s.util_short_correction = 0.0f;
    s.pre_fill_balance = initial_equity;
    s.current_fill_step = -1.0f;
    s.realized_long = s.realized_short = 0.0f;
    s.realized_long_correction = s.realized_short_correction = 0.0f;
    s.ew_log_correction = s.ew_days_correction = 0.0f;
    s.has_pending = true;
    for (int c = 0; c < MAX_COINS; ++c) {
        s.fill_mark_price[c] = 0.0f;
        s.fill_mark_step[c] = -1.0f;
    }
    return s;
}

inline void core_daily_close(thread CoreGpuState& s, float equity, int step, int origin_minute) {
    if (!(CORE_WANTS(VOLATILITY) || CORE_WANTS(SHARPE) || CORE_WANTS(SORTINO) || CORE_WANTS(OMEGA))) return;
    if ((origin_minute + step) % 1440 != 0) return;
    if (s.daily_last_step == float(step)) return;
    if (!isfinite(equity) || equity <= 0.0f) { s.daily_invalid = 1.0f; return; }
    if (s.daily_last_step >= 0.0f) {
        if (float(step) - s.daily_last_step != 1440.0f) { s.daily_invalid = 1.0f; return; }
        const float r = (equity - s.daily_previous_equity) / s.daily_previous_equity;
        if (!isfinite(r)) { s.daily_invalid = 1.0f; return; }
        core_increment(s.daily_count);
        const float delta = r - s.daily_mean;
        s.daily_mean += delta / s.daily_count;
        s.daily_m2 += delta * (r - s.daily_mean);
        s.daily_downside_squares += fmin(r, 0.0f) * fmin(r, 0.0f);
        s.daily_gains += fmax(r, 0.0f);
        s.daily_losses += fmax(-r, 0.0f);
    } else { s.daily_first_step = float(step); }
    s.daily_last_step = float(step);
    s.daily_previous_equity = equity;
}

inline void core_commit_equity(thread CoreGpuState& s, int origin_minute) {
    if (!s.has_pending) return;
    const float equity = s.pending_equity;
    const float step = s.pending_step;
    s.has_pending = false;
    core_increment(s.minute_count);
    s.last_equity = equity;
    s.end_step = step;
    if (!isfinite(equity)) s.equity_invalid = 1.0f;
    if (equity <= 0.0f) s.equity_nonpositive = 1.0f;
    if (CORE_WANTS(DRAWDOWN) || CORE_WANTS(ULCER) || CORE_WANTS(PAIN) || CORE_WANTS(RECOVERY)) {
        if (s.equity_invalid == 0.0f && isfinite(s.peak) && s.peak > 0.0f) {
            if (equity >= s.peak) {
                if (s.underwater_start_step >= 0.0f)
                    s.recovery_max_steps = fmax(s.recovery_max_steps, step - s.underwater_start_step);
                s.underwater_start_step = -1.0f;
                s.peak = equity;
                s.peak_step = step;
            } else if (s.underwater_start_step < 0.0f) { s.underwater_start_step = s.peak_step; }
            const float dd = (s.peak - equity) / s.peak;
            core_add(dd, s.drawdown_sum, s.drawdown_correction);
            core_add(dd * dd, s.drawdown_squares, s.squared_correction);
            s.drawdown_max = fmax(s.drawdown_max, dd);
        }
    }
    if (CORE_WANTS(EW) && s.previous_step >= 0.0f && s.equity_invalid == 0.0f && s.equity_nonpositive == 0.0f) {
        const float days = (step - s.previous_step) / 1440.0f;
        const float age = 0.6931471805599453f * days / PASSIVBOT_CORE_HALF_LIFE_DAYS;
        // Compute 1-exp(-age) without rounding a near-one decay to a coarse
        // float32 multiplier. The local series has negligible truncation error.
        const float decay_loss = age < 0.001f
            ? age * (1.0f + age * (-0.5f + age * (1.0f / 6.0f - age / 24.0f)))
            : 1.0f - exp(-age);
        const float ratio = equity / s.previous_equity;
        const float change = (equity - s.previous_equity) / s.previous_equity;
        const float z = fabs(change) < 0.001f
            ? change * (1.0f + change * (-0.5f + change * (1.0f / 3.0f + change * (-0.25f + change * 0.2f))))
            : (isfinite(ratio) && ratio > 0.0f ? log(ratio) : log(equity) - log(s.previous_equity));
        core_add(z - decay_loss * s.ew_log_sum, s.ew_log_sum, s.ew_log_correction);
        core_add(days - decay_loss * s.ew_days_sum, s.ew_days_sum, s.ew_days_correction);
        if (!isfinite(s.ew_log_sum) || !isfinite(s.ew_days_sum)) s.ew_invalid = 1.0f;
    }
    core_daily_close(s, equity, int(step), origin_minute);
    s.previous_equity = equity;
    s.previous_step = step;
    core_increment(s.exposure_count);
    if (CORE_WANTS(GROSS_RMS) || CORE_WANTS(GROSS_MAX)) {
        const float notional = s.pending_gross_notional;
        if (!isfinite(equity) || equity <= 0.0f || !isfinite(notional) || notional < 0.0f) s.gross_invalid = 1.0f;
        else {
            const float g = notional / equity;
            core_add(g * g, s.gross_squares, s.gross_correction);
            s.gross_max = fmax(s.gross_max, g);
        }
    }
    if (CORE_WANTS(UTIL_LONG)) {
        const float value = s.pending_utilization_long;
        if (!isfinite(value) || value < 0.0f) s.utilization_long_invalid = 1.0f;
        else core_add(value, s.utilization_long_sum, s.util_long_correction);
    }
    if (CORE_WANTS(UTIL_SHORT)) {
        const float value = s.pending_utilization_short;
        if (!isfinite(value) || value < 0.0f) s.utilization_short_invalid = 1.0f;
        else core_add(value, s.utilization_short_sum, s.util_short_correction);
    }
}

inline void core_observe_equity(thread CoreGpuState& s, float equity, int step, int origin_minute,
    float gross_marked_notional, float utilization_long_squared, float utilization_short_squared,
    bool fill_terminal) {
    if (float(step) < s.pending_step || float(step) > s.pending_step + 1.0f) { s.clock_invalid = 1.0f; return; }
    // A completed midnight close remains the daily fact when the next candle's
    // fill replaces its same-clock minute endpoint.
    if (fill_terminal && float(step) == s.pending_step)
        core_daily_close(s, s.pending_equity, step, origin_minute);
    if (float(step) > s.pending_step) core_commit_equity(s, origin_minute);
    s.pending_step = float(step);
    s.pending_equity = equity;
    s.pending_gross_notional = gross_marked_notional;
    s.pending_utilization_long = utilization_long_squared;
    s.pending_utilization_short = utilization_short_squared;
    s.has_pending = true;
}

inline void core_record_fill(thread CoreGpuState& s, int step, float net_pnl, bool is_long) {
    s.current_fill_step = float(step);
    if (float(step) < s.start_step) return;
    if (float(step) < s.last_fill_step) { s.fill_invalid = 1.0f; return; }
    s.fill_gap_max_steps = fmax(s.fill_gap_max_steps, float(step) - s.last_fill_step);
    s.last_fill_step = float(step);
    core_increment(s.fill_count);
    if (!isfinite(net_pnl)) { s.realized_invalid = 1.0f; s.equity_invalid = 1.0f; }
    else if (is_long) core_add(net_pnl, s.realized_long, s.realized_long_correction);
    else core_add(net_pnl, s.realized_short, s.realized_short_correction);
    if (CORE_WANTS(LOSS_PROFIT)) {
        if (!isfinite(net_pnl)) s.realized_invalid = 1.0f;
        else {
            core_add(fmax(net_pnl, 0.0f), s.realized_profit, s.profit_correction);
            core_add(fmax(-net_pnl, 0.0f), s.realized_loss, s.loss_correction);
        }
    }
}

inline void core_record_notional(thread CoreGpuState& s, float gross_notional) {
    if (!CORE_WANTS(TURNOVER) || s.current_fill_step < s.start_step) return;
    if (!isfinite(gross_notional) || gross_notional < 0.0f || !isfinite(s.pre_fill_balance) || s.pre_fill_balance <= 0.0f)
        s.turnover_invalid = 1.0f;
    else core_add(gross_notional / s.pre_fill_balance, s.turnover_sum, s.turnover_correction);
}

inline void core_complete_holding(thread CoreGpuState& s, float opened_step, float end_step) {
    if (!(CORE_WANTS(HOLD_MAX) || CORE_WANTS(HOLD_MEAN))) return;
    if (!isfinite(opened_step) || opened_step < s.start_step || end_step < opened_step) {
        s.holding_invalid = 1.0f; return;
    }
    const float hours = (end_step - opened_step) / 60.0f;
    core_increment(s.holding_count);
    s.holding_sum_hours += hours;
    s.holding_sum_squares += hours * hours;
    s.holding_max_hours = fmax(s.holding_max_hours, hours);
}

// Finalization is a copy, so tails and a pending terminal mark enter once even
// when diagnostics query an accepted replay more than once.
inline void write_core_gpu(thread const CoreGpuState& input, device float* output, int offset, int origin_minute) {
    CoreGpuState s = input;
    core_commit_equity(s, origin_minute);
    if (s.underwater_start_step >= 0.0f)
        s.recovery_max_steps = fmax(s.recovery_max_steps, s.end_step - s.underwater_start_step);
    s.peak_recovery_tail_censored = s.underwater_start_step >= 0.0f ? 1.0f : 0.0f;
// CORE_WRITE_FIELDS
}
