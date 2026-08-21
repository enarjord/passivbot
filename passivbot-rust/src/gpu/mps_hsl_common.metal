// Shared Apple Metal HSL screening controller.
//
// Exact Rust backtests remain authoritative. Strategy kernels provide scoped
// realized and unrealized PnL; this module owns the common proxy lifecycle.

constant int HSL_SIGNAL_UNIFIED = 0;
constant int HSL_SIGNAL_PSIDE = 1;
constant int HSL_SIGNAL_COIN = 2;

struct HslState {
    bool enabled;
    float red_threshold;
    float alpha;
    float cooldown_minutes;
    float no_restart_threshold;
    int restart_policy;
    float yellow_ratio;
    float orange_ratio;
    bool orange_graceful_stop;
    int signal_mode;
    float slot_count;
    bool initialized;
    float drawdown_ema;
    float peak_strategy_pnl;
    float no_restart_peak_strategy_equity;
    float coin_realized_baseline;
    float coin_realized_peak;
    int tier;
    bool red_latched;
    bool red_active_now;
    bool halted;
    bool no_restart_latched;
    float cooldown_until_k;
    int flat_confirmations;
    float pending_drawdown_raw;
    float pending_drawdown_ema;
    float pending_strategy_equity;
    float pending_peak_strategy_equity;
    float pending_stop_k;
    float current_red_start_k;
    float current_halt_start_k;
    float last_restart_k;
    float triggers;
    float restarts;
    float halt_duration_sum_steps;
    float halt_duration_max_steps;
    float halt_duration_count;
    float trigger_drawdown_sum;
    float trigger_drawdown_count;
    float flatten_time_sum_steps;
    float flatten_time_count;
    float restart_retrigger_count;
    float equity_at_halt;
    float halt_to_restart_equity_loss;
    float panic_event_start_equity;
    float panic_event_loss;
    float panic_close_loss_sum;
    float panic_close_loss_max;
    float panic_loss_drawdown_min;
    float panic_loss_drawdown_sum;
    float panic_loss_drawdown_max;
    float panic_loss_drawdown_count;
};

struct HslSignal {
    float drawdown_raw;
    float strategy_equity;
    float peak_strategy_equity;
};

inline HslState load_hsl(
    constant float* params,
    int po,
    int hsl_param_offset
) {
    HslState h;
    int ho = po + hsl_param_offset;
    h.enabled = params[ho + 0] > 0.5f;
    h.red_threshold = params[ho + 1];
    h.alpha = clamp(2.0f / (fmax(params[ho + 2], 1.0f) + 1.0f), 0.0f, 1.0f);
    h.cooldown_minutes = fmax(params[ho + 3], 0.0f);
    h.no_restart_threshold = fmax(params[ho + 4], h.red_threshold);
    h.restart_policy = int(round(params[ho + 5]));
    h.yellow_ratio = params[ho + 6];
    h.orange_ratio = params[ho + 7];
    h.orange_graceful_stop = params[ho + 8] > 0.5f;
    h.signal_mode = int(round(params[ho + 9]));
    h.slot_count = fmax(round(params[ho + 10]), 1.0f);
    h.initialized = false;
    h.drawdown_ema = 0.0f;
    h.peak_strategy_pnl = -INFINITY;
    h.no_restart_peak_strategy_equity = 0.0f;
    h.coin_realized_baseline = 0.0f;
    h.coin_realized_peak = 0.0f;
    h.tier = 0;
    h.red_latched = false;
    h.red_active_now = false;
    h.halted = false;
    h.no_restart_latched = false;
    h.cooldown_until_k = -1.0f;
    h.flat_confirmations = 0;
    h.pending_drawdown_raw = 0.0f;
    h.pending_drawdown_ema = 0.0f;
    h.pending_strategy_equity = 0.0f;
    h.pending_peak_strategy_equity = 0.0f;
    h.pending_stop_k = -1.0f;
    h.current_red_start_k = -1.0f;
    h.current_halt_start_k = -1.0f;
    h.last_restart_k = -1.0f;
    h.triggers = 0.0f;
    h.restarts = 0.0f;
    h.halt_duration_sum_steps = 0.0f;
    h.halt_duration_max_steps = 0.0f;
    h.halt_duration_count = 0.0f;
    h.trigger_drawdown_sum = 0.0f;
    h.trigger_drawdown_count = 0.0f;
    h.flatten_time_sum_steps = 0.0f;
    h.flatten_time_count = 0.0f;
    h.restart_retrigger_count = 0.0f;
    h.equity_at_halt = 0.0f;
    h.halt_to_restart_equity_loss = 0.0f;
    h.panic_event_start_equity = -1.0f;
    h.panic_event_loss = 0.0f;
    h.panic_close_loss_sum = 0.0f;
    h.panic_close_loss_max = 0.0f;
    h.panic_loss_drawdown_min = 0.0f;
    h.panic_loss_drawdown_sum = 0.0f;
    h.panic_loss_drawdown_max = 0.0f;
    h.panic_loss_drawdown_count = 0.0f;
    return h;
}

inline int hsl_mode(thread HslState& h, bool has_position) {
    if (!h.enabled) return 0;
    if (h.halted) return has_position ? 3 : 1;
    if (h.tier == 3) return h.red_active_now ? 3 : 2;
    if (h.tier == 2) return h.orange_graceful_stop ? 1 : 2;
    return 0;
}

inline bool derive_hsl_signal(
    thread HslState& h,
    float balance,
    float starting_balance,
    float realized_pnl,
    float unrealized_pnl,
    thread HslSignal& signal
) {
    if (!h.enabled || h.halted || !(balance > 0.0f)) return false;
    float drawdown_raw;
    float strategy_pnl = realized_pnl + unrealized_pnl;
    float strategy_equity;
    float peak_strategy_equity;
    if (h.signal_mode == HSL_SIGNAL_COIN) {
        float coin_realized = realized_pnl - h.coin_realized_baseline;
        h.coin_realized_peak = fmax(h.coin_realized_peak, coin_realized);
        drawdown_raw = fmin(fmax(
            h.coin_realized_peak - (coin_realized + unrealized_pnl), 0.0f
        ) / (balance / h.slot_count), 0.9999999403953552f);
        strategy_equity = fmax(1.0f - drawdown_raw, 1.0e-12f);
        peak_strategy_equity = 1.0f;
    } else {
        h.peak_strategy_pnl = fmax(h.peak_strategy_pnl, strategy_pnl);
        strategy_equity = starting_balance + strategy_pnl;
        peak_strategy_equity = fmax(
            starting_balance + h.peak_strategy_pnl, strategy_equity
        );
        if (!(strategy_equity > 0.0f && peak_strategy_equity > 0.0f)) return false;
        drawdown_raw = fmin(
            fmax(1.0f - strategy_equity / peak_strategy_equity, 0.0f),
            0.9999999403953552f
        );
    }
    signal.drawdown_raw = drawdown_raw;
    signal.strategy_equity = strategy_equity;
    signal.peak_strategy_equity = peak_strategy_equity;
    return true;
}

inline void update_hsl_from_signal(
    thread HslState& h,
    thread HslSignal& signal,
    float realized_pnl,
    bool has_position,
    bool has_blocking_orders,
    float kf,
    float interval_ms
) {
    float drawdown_raw = signal.drawdown_raw;
    float strategy_equity = signal.strategy_equity;
    float peak_strategy_equity = signal.peak_strategy_equity;
    if (!h.initialized) {
        h.initialized = true;
        h.drawdown_ema = 0.0f;
        h.tier = 0;
        return;
    }
    h.drawdown_ema = fma(h.alpha, drawdown_raw - h.drawdown_ema, h.drawdown_ema);
    float score = fmin(drawdown_raw, fmax(h.drawdown_ema, 0.0f));
    const float cmp_eps = 1.0e-12f;
    h.red_active_now = score + cmp_eps >= h.red_threshold;
    int next_tier = h.red_latched ? 3
        : h.red_active_now ? 3
        : score + cmp_eps >= h.orange_ratio * h.red_threshold ? 2
        : score + cmp_eps >= h.yellow_ratio * h.red_threshold ? 1 : 0;
    if (next_tier == 3 && h.tier != 3) h.current_red_start_k = kf;
    if (next_tier == 3) h.red_latched = true;
    h.tier = h.red_latched ? 3 : next_tier;
    if (h.tier != 3) h.current_red_start_k = -1.0f;
    if (h.tier == 3) {
        if (has_position || has_blocking_orders) {
            h.flat_confirmations = 0;
        } else {
            h.flat_confirmations += 1;
            if (h.flat_confirmations == 1) {
                h.pending_drawdown_raw = drawdown_raw;
                h.pending_drawdown_ema = h.drawdown_ema;
                h.pending_strategy_equity = strategy_equity;
                h.pending_peak_strategy_equity = peak_strategy_equity;
                h.pending_stop_k = kf;
            }
            if (h.flat_confirmations >= 2) {
                h.halted = true;
                h.current_halt_start_k = h.pending_stop_k;
                h.triggers += 1.0f;
                h.equity_at_halt = strategy_equity;
                if (h.signal_mode == HSL_SIGNAL_COIN) {
                    h.coin_realized_baseline = realized_pnl;
                    h.coin_realized_peak = 0.0f;
                }
                h.no_restart_peak_strategy_equity = fmax(
                    h.no_restart_peak_strategy_equity,
                    fmax(
                        h.pending_peak_strategy_equity,
                        h.pending_strategy_equity
                    )
                );
                float no_restart_drawdown_raw = h.signal_mode == HSL_SIGNAL_COIN
                    ? h.pending_drawdown_raw
                    : fmin(
                        fmax(
                            1.0f - h.pending_strategy_equity
                                / fmax(h.no_restart_peak_strategy_equity, 1.0e-12f),
                            0.0f
                        ),
                        0.9999999403953552f
                    );
                float stop_drawdown_raw = fmax(
                    h.pending_drawdown_raw,
                    fmax(
                        1.0f - h.pending_strategy_equity
                            / fmax(h.pending_peak_strategy_equity, 1.0e-12f),
                        0.0f
                    )
                );
                h.trigger_drawdown_sum += stop_drawdown_raw;
                h.trigger_drawdown_count += 1.0f;
                if (h.current_red_start_k >= 0.0f) {
                    h.flatten_time_sum_steps += fmax(
                        h.pending_stop_k - h.current_red_start_k, 0.0f
                    );
                    h.flatten_time_count += 1.0f;
                }
                if (h.last_restart_k >= 0.0f
                    && (h.pending_stop_k - h.last_restart_k) * interval_ms
                        <= 86400000.0f) {
                    h.restart_retrigger_count += 1.0f;
                }
                h.last_restart_k = -1.0f;
                if (h.panic_event_start_equity >= 0.0f) {
                    float loss_drawdown = fmax(
                        h.panic_event_loss
                            / fmax(h.panic_event_start_equity, 1.0e-12f),
                        0.0f
                    );
                    h.panic_loss_drawdown_min = h.panic_loss_drawdown_count > 0.0f
                        ? fmin(h.panic_loss_drawdown_min, loss_drawdown)
                        : loss_drawdown;
                    h.panic_loss_drawdown_sum += loss_drawdown;
                    h.panic_loss_drawdown_max = fmax(
                        h.panic_loss_drawdown_max, loss_drawdown
                    );
                    h.panic_loss_drawdown_count += 1.0f;
                    h.panic_event_start_equity = -1.0f;
                    h.panic_event_loss = 0.0f;
                }
                bool terminal = h.restart_policy == 2
                    || (h.restart_policy == 1
                        && fmax(no_restart_drawdown_raw, h.pending_drawdown_ema)
                            >= h.no_restart_threshold);
                h.no_restart_latched = terminal;
                h.cooldown_until_k = terminal || h.cooldown_minutes <= 0.0f
                    ? -1.0f : h.pending_stop_k + h.cooldown_minutes;
            }
        }
    } else {
        h.flat_confirmations = 0;
    }
}

inline void update_hsl(
    thread HslState& h,
    float balance,
    float starting_balance,
    float realized_pnl,
    float unrealized_pnl,
    bool has_position,
    bool has_blocking_orders,
    float kf,
    float interval_ms
) {
    HslSignal signal;
    if (!derive_hsl_signal(
        h, balance, starting_balance, realized_pnl, unrealized_pnl, signal
    )) return;
    update_hsl_from_signal(
        h, signal, realized_pnl, has_position, has_blocking_orders, kf, interval_ms
    );
}

inline void try_restart_hsl(thread HslState& h, float kf, float current_equity) {
    if (!h.enabled || !h.halted || h.no_restart_latched
        || h.cooldown_until_k < 0.0f || kf < h.cooldown_until_k) return;
    if (h.current_halt_start_k >= 0.0f) {
        float duration = fmax(kf - h.current_halt_start_k, 0.0f);
        h.halt_duration_sum_steps += duration;
        h.halt_duration_max_steps = fmax(h.halt_duration_max_steps, duration);
        h.halt_duration_count += 1.0f;
        h.current_halt_start_k = -1.0f;
    }
    h.restarts += 1.0f;
    if (h.signal_mode != HSL_SIGNAL_COIN && h.equity_at_halt > 0.0f) {
        h.halt_to_restart_equity_loss += fmax(
            h.equity_at_halt - current_equity, 0.0f
        );
    }
    h.last_restart_k = kf;
    h.initialized = false;
    h.drawdown_ema = 0.0f;
    h.peak_strategy_pnl = -INFINITY;
    h.tier = 0;
    h.red_latched = false;
    h.red_active_now = false;
    h.halted = false;
    h.cooldown_until_k = -1.0f;
    h.flat_confirmations = 0;
    h.current_red_start_k = -1.0f;
}

inline void record_hsl_panic_fill(
    thread HslState& h,
    float net_pnl,
    float current_equity
) {
    if (h.panic_event_start_equity < 0.0f) {
        h.panic_event_start_equity = fmax(current_equity, 1.0e-12f);
    }
    float panic_loss = fmax(-net_pnl, 0.0f);
    h.panic_event_loss += panic_loss;
    h.panic_close_loss_sum += panic_loss;
    h.panic_close_loss_max = fmax(h.panic_close_loss_max, panic_loss);
}

inline void write_one_side_hsl_outputs(
    thread HslState& h,
    bool short_side,
    float tier_samples_total,
    float tier_samples_yellow,
    float tier_samples_orange,
    float tier_samples_red,
    float last_equity_k,
    device float* scalars,
    int scalar_offset
) {
    float terminal_count = h.halted
        && h.current_halt_start_k >= 0.0f && last_equity_k >= 0.0f
        ? 1.0f : 0.0f;
    float terminal_duration = terminal_count > 0.0f
        ? fmax(last_equity_k - h.current_halt_start_k, 0.0f) : 0.0f;
    scalars[scalar_offset + 0] = !short_side && h.enabled ? 1.0f : 0.0f;
    scalars[scalar_offset + 1] = short_side && h.enabled ? 1.0f : 0.0f;
    scalars[scalar_offset + 2] = !short_side ? h.triggers : 0.0f;
    scalars[scalar_offset + 3] = short_side ? h.triggers : 0.0f;
    scalars[scalar_offset + 4] = !short_side ? h.restarts : 0.0f;
    scalars[scalar_offset + 5] = short_side ? h.restarts : 0.0f;
    scalars[scalar_offset + 6] = tier_samples_total;
    scalars[scalar_offset + 7] = tier_samples_yellow;
    scalars[scalar_offset + 8] = tier_samples_orange;
    scalars[scalar_offset + 9] = tier_samples_red;
    scalars[scalar_offset + 10] = h.halt_duration_sum_steps + terminal_duration;
    scalars[scalar_offset + 11] = fmax(
        h.halt_duration_max_steps, terminal_duration
    );
    scalars[scalar_offset + 12] = h.halt_duration_count + terminal_count;
    scalars[scalar_offset + 13] = h.trigger_drawdown_sum;
    scalars[scalar_offset + 14] = h.trigger_drawdown_count;
    scalars[scalar_offset + 15] = h.flatten_time_sum_steps;
    scalars[scalar_offset + 16] = h.flatten_time_count;
    scalars[scalar_offset + 17] = h.restart_retrigger_count;
    scalars[scalar_offset + 18] = h.halt_to_restart_equity_loss;
    scalars[scalar_offset + 19] = h.panic_close_loss_sum;
    scalars[scalar_offset + 20] = h.panic_close_loss_max;
    scalars[scalar_offset + 21] = h.panic_loss_drawdown_min;
    scalars[scalar_offset + 22] = h.panic_loss_drawdown_sum;
    scalars[scalar_offset + 23] = h.panic_loss_drawdown_max;
    scalars[scalar_offset + 24] = h.panic_loss_drawdown_count;
}
