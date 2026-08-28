// Shared Apple Metal HSL screening controller.
//
// Exact Rust backtests remain authoritative. Strategy kernels provide scoped
// realized and unrealized PnL; this module owns the common proxy lifecycle.

#ifndef PASSIVBOT_HSL_EMA_TAIL_ENABLED
#define PASSIVBOT_HSL_EMA_TAIL_ENABLED 0
#endif

#ifndef PASSIVBOT_HSL_RAW_DRAWDOWN_ENABLED
#define PASSIVBOT_HSL_RAW_DRAWDOWN_ENABLED 0
#endif

#ifndef PASSIVBOT_HSL_RAW_TAIL_ENABLED
#define PASSIVBOT_HSL_RAW_TAIL_ENABLED 0
#endif

#ifndef PASSIVBOT_HSL_DIAGNOSTICS_ENABLED
#define PASSIVBOT_HSL_DIAGNOSTICS_ENABLED 1
#endif

#define HSL_EMA_TAIL_BINS 32

constant int HSL_SIGNAL_UNIFIED = 0;
constant int HSL_SIGNAL_PSIDE = 1;
constant int HSL_SIGNAL_COIN = 2;

// Finite coin-HSL PnL windows are candidate-local.  Their fill events live in
// device buffers owned by the directional runner; this small thread-local
// structure keeps only ring/deque cursors and the absolute realized cumsum.
// Multiple realized-PnL components may occur on one candle (for example,
// several martingale entry fees followed by a close).  They are coalesced into
// one event-candle without changing the rolling current or peak.  The two
// packed buffers use float2(base_before_candle, peak_during_candle) and
// int2(event_k, monotonic_peak_slot) respectively.
struct HslRollingPnlWindow {
    int event_head;
    int event_count;
    int peak_head;
    int peak_count;
    float absolute_cumulative;
    bool overflowed;
};

struct HslRollingPnlSignal {
    float peak;
    float current;
};

inline HslRollingPnlWindow init_hsl_rolling_pnl_window() {
    HslRollingPnlWindow window;
    window.event_head = 0;
    window.event_count = 0;
    window.peak_head = 0;
    window.peak_count = 0;
    window.absolute_cumulative = 0.0f;
    window.overflowed = false;
    return window;
}

inline void reset_hsl_rolling_pnl_window(
    thread HslRollingPnlWindow& window
) {
    window.event_head = 0;
    window.event_count = 0;
    window.peak_head = 0;
    window.peak_count = 0;
}

inline void prune_hsl_rolling_pnl_window(
    thread HslRollingPnlWindow& window,
    device float2* values,
    device int2* indices,
    int base,
    int capacity,
    int k,
    int lookback_bars
) {
    if (lookback_bars <= 0 || window.overflowed) return;
    while (window.event_count > 0) {
        int slot = window.event_head;
        if (k - indices[base + slot].x <= lookback_bars) break;
        if (window.peak_count > 0
            && indices[base + window.peak_head].y == slot) {
            window.peak_head = (window.peak_head + 1) % capacity;
            window.peak_count -= 1;
        }
        window.event_head = (window.event_head + 1) % capacity;
        window.event_count -= 1;
    }
}

inline void record_hsl_rolling_pnl(
    thread HslRollingPnlWindow& window,
    device float2* values,
    device int2* indices,
    int base,
    int capacity,
    int k,
    int lookback_bars,
    bool active,
    float pnl
) {
    if (!active || lookback_bars <= 0 || window.overflowed) return;
    window.absolute_cumulative += pnl;
    prune_hsl_rolling_pnl_window(
        window, values, indices, base, capacity, k, lookback_bars
    );
    if (window.event_count > 0) {
        int slot = (window.event_head + window.event_count - 1) % capacity;
        if (indices[base + slot].x == k) {
            values[base + slot].y = fmax(
                values[base + slot].y, window.absolute_cumulative
            );
            if (window.peak_count > 0) {
                int peak_tail = (
                    window.peak_head + window.peak_count - 1
                ) % capacity;
                if (indices[base + peak_tail].y == slot) {
                    window.peak_count -= 1;
                }
            }
            while (window.peak_count > 0) {
                int back = (
                    window.peak_head + window.peak_count - 1
                ) % capacity;
                int peak_slot = indices[base + back].y;
                if (values[base + peak_slot].y
                    > values[base + slot].y) break;
                window.peak_count -= 1;
            }
            int peak_tail = (
                window.peak_head + window.peak_count
            ) % capacity;
            indices[base + peak_tail].y = slot;
            window.peak_count += 1;
            return;
        }
    }
    if (window.event_count >= capacity || window.peak_count >= capacity) {
        window.overflowed = true;
        return;
    }
    int slot = (window.event_head + window.event_count) % capacity;
    values[base + slot] = float2(
        window.absolute_cumulative - pnl, window.absolute_cumulative
    );
    indices[base + slot].x = k;
    window.event_count += 1;

    while (window.peak_count > 0) {
        int back = (window.peak_head + window.peak_count - 1) % capacity;
        int peak_slot = indices[base + back].y;
        if (values[base + peak_slot].y > window.absolute_cumulative) break;
        window.peak_count -= 1;
    }
    int peak_tail = (window.peak_head + window.peak_count) % capacity;
    indices[base + peak_tail].y = slot;
    window.peak_count += 1;
}

inline HslRollingPnlSignal effective_hsl_rolling_pnl(
    thread HslRollingPnlWindow& window,
    device float2* values,
    device int2* indices,
    int base,
    int capacity,
    int k,
    int lookback_bars
) {
    HslRollingPnlSignal signal;
    signal.peak = 0.0f;
    signal.current = 0.0f;
    if (lookback_bars <= 0 || window.overflowed) return signal;
    prune_hsl_rolling_pnl_window(
        window, values, indices, base, capacity, k, lookback_bars
    );
    if (window.event_count == 0) return signal;
    float base_cumulative = values[base + window.event_head].x;
    int peak_slot = indices[base + window.peak_head].y;
    signal.current = window.absolute_cumulative - base_cumulative;
    signal.peak = fmax(
        values[base + peak_slot].y - base_cumulative,
        fmax(signal.current, 0.0f)
    );
    return signal;
}

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
#if PASSIVBOT_HSL_DIAGNOSTICS_ENABLED
    float drawdown_ema_max;
#endif
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
#if PASSIVBOT_HSL_DIAGNOSTICS_ENABLED
    float last_restart_k;
#endif
    float triggers;
    float restarts;
#if PASSIVBOT_HSL_DIAGNOSTICS_ENABLED
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
#endif
};

inline void prepare_coin_hsl_rolling_signal(
    thread HslState& h,
    thread HslRollingPnlWindow& window,
    device float2* values,
    device int2* indices,
    int base,
    int capacity,
    int k,
    int lookback_bars,
    float realized_pnl
) {
    if (!h.enabled || h.signal_mode != HSL_SIGNAL_COIN
        || lookback_bars <= 0 || window.overflowed) return;
    HslRollingPnlSignal signal = effective_hsl_rolling_pnl(
        window, values, indices, base, capacity, k, lookback_bars
    );
    h.coin_realized_baseline = realized_pnl - signal.current;
    h.coin_realized_peak = signal.peak;
}

struct HslSignal {
    float drawdown_raw;
    float strategy_equity;
    float peak_strategy_equity;
};

struct HslStrategyEquityStats {
    bool initialized;
    float peak;
    float peak_sample_k;
    float last_sample_k;
    float recovery_max_steps;
#if PASSIVBOT_HSL_RAW_DRAWDOWN_ENABLED
    float drawdown_max;
#endif
#if PASSIVBOT_HSL_RAW_TAIL_ENABLED
    int current_drawdown_day;
    float current_day_drawdown_worst;
    float completed_day_count;
    float daily_drawdown_counts[HSL_EMA_TAIL_BINS];
    float daily_drawdown_sums[HSL_EMA_TAIL_BINS];
#endif
};

inline HslStrategyEquityStats init_hsl_strategy_equity_stats() {
    HslStrategyEquityStats stats;
    stats.initialized = false;
    stats.peak = 0.0f;
    stats.peak_sample_k = -1.0f;
    stats.last_sample_k = -1.0f;
    stats.recovery_max_steps = 0.0f;
#if PASSIVBOT_HSL_RAW_DRAWDOWN_ENABLED
    stats.drawdown_max = 0.0f;
#endif
#if PASSIVBOT_HSL_RAW_TAIL_ENABLED
    stats.current_drawdown_day = -1;
    stats.current_day_drawdown_worst = 0.0f;
    stats.completed_day_count = 0.0f;
    for (int i = 0; i < HSL_EMA_TAIL_BINS; ++i) {
        stats.daily_drawdown_counts[i] = 0.0f;
        stats.daily_drawdown_sums[i] = 0.0f;
    }
#endif
    return stats;
}

inline int hsl_drawdown_tail_bin(float value) {
    // Cover fourteen octaves over [2^-14, 1) so low-drawdown Pareto members
    // remain rankable without increasing thread-local state. Edge bins retain
    // smaller values and overflow respectively. Actual sums are not clamped,
    // so values outside the covered range keep their magnitude.
    float scaled = (log2(fmax(value, 0.00006103515625f)) + 14.0f)
        * 2.2857142857142856f;
    return clamp(int(floor(scaled)), 0, HSL_EMA_TAIL_BINS - 1);
}

inline void flush_hsl_strategy_equity_daily_drawdown(
    thread HslStrategyEquityStats& stats
) {
#if PASSIVBOT_HSL_RAW_TAIL_ENABLED
    if (stats.current_drawdown_day < 0) return;
    int bin = hsl_drawdown_tail_bin(stats.current_day_drawdown_worst);
    stats.completed_day_count += 1.0f;
    stats.daily_drawdown_counts[bin] += 1.0f;
    stats.daily_drawdown_sums[bin] += stats.current_day_drawdown_worst;
#endif
}

inline void update_hsl_strategy_equity_stats(
    thread HslStrategyEquityStats& stats,
    float strategy_equity,
    int day_index
) {
    if (!isfinite(strategy_equity)) return;
    const float sample_k = stats.initialized
        ? stats.last_sample_k + 1.0f : 0.0f;
    if (!stats.initialized) {
        stats.initialized = true;
        stats.peak = strategy_equity;
        stats.peak_sample_k = sample_k;
        stats.last_sample_k = sample_k;
#if PASSIVBOT_HSL_RAW_TAIL_ENABLED
        stats.current_drawdown_day = day_index;
        stats.current_day_drawdown_worst = 0.0f;
#endif
        return;
    }
    stats.last_sample_k = sample_k;
    if (strategy_equity > stats.peak) {
        stats.recovery_max_steps = fmax(
            stats.recovery_max_steps, sample_k - stats.peak_sample_k
        );
        stats.peak = strategy_equity;
        stats.peak_sample_k = sample_k;
    }
#if PASSIVBOT_HSL_RAW_DRAWDOWN_ENABLED
    float drawdown = (stats.peak - strategy_equity)
        / fmax(fabs(stats.peak), 1.0e-12f);
    stats.drawdown_max = fmax(stats.drawdown_max, drawdown);
#endif
#if PASSIVBOT_HSL_RAW_TAIL_ENABLED
    float daily_drawdown = (stats.peak - strategy_equity)
        / fmax(fabs(stats.peak), 1.0e-12f);
    if (day_index > stats.current_drawdown_day) {
        flush_hsl_strategy_equity_daily_drawdown(stats);
        stats.current_drawdown_day = day_index;
        stats.current_day_drawdown_worst = daily_drawdown;
    } else {
        stats.current_day_drawdown_worst = fmax(
            stats.current_day_drawdown_worst, daily_drawdown
        );
    }
#endif
}

inline float hsl_strategy_equity_drawdown_max(
    thread HslStrategyEquityStats& stats
) {
#if PASSIVBOT_HSL_RAW_DRAWDOWN_ENABLED
    return stats.drawdown_max;
#else
    return 0.0f;
#endif
}

inline float hsl_strategy_equity_drawdown_mean_worst_1pct(
    thread HslStrategyEquityStats& stats
) {
#if PASSIVBOT_HSL_RAW_TAIL_ENABLED
    float sample_count = stats.completed_day_count
        + (stats.current_drawdown_day >= 0 ? 1.0f : 0.0f);
    if (!(sample_count > 0.0f)) return 0.0f;
    float worst_n = fmax(floor(sample_count * 0.01f), 1.0f);
    float remaining = worst_n;
    float total = 0.0f;
    int current_bin = stats.current_drawdown_day >= 0
        ? hsl_drawdown_tail_bin(stats.current_day_drawdown_worst) : -1;
    for (int i = HSL_EMA_TAIL_BINS - 1; i >= 0 && remaining > 0.0f; --i) {
        float count = stats.daily_drawdown_counts[i]
            + (i == current_bin ? 1.0f : 0.0f);
        if (!(count > 0.0f)) continue;
        float sum = stats.daily_drawdown_sums[i]
            + (i == current_bin ? stats.current_day_drawdown_worst : 0.0f);
        float take = fmin(count, remaining);
        total += sum * (take / count);
        remaining -= take;
    }
    return total / worst_n;
#else
    return 0.0f;
#endif
}

inline float hsl_strategy_equity_recovery_max_steps(
    thread HslStrategyEquityStats& stats
) {
    if (!stats.initialized) return 0.0f;
    return fmax(
        stats.recovery_max_steps,
        fmax(stats.last_sample_k - stats.peak_sample_k, 0.0f)
    );
}

// Exact Rust sorts every retained drawdown-EMA sample before averaging the
// largest floor(1%) (at least one). Keeping that unbounded series per Metal
// thread would make large optimizer populations impractical. The proxy uses a
// deterministic log histogram with exact per-bin sums/counts; only the partial
// cutoff bin is approximated. Exact validations and drift gates remain
// authoritative. The preprocessor removes this state and work unless one of
// the tail metrics is requested.
struct HslDrawdownEmaTailStats {
#if PASSIVBOT_HSL_EMA_TAIL_ENABLED
    float sample_count;
    float counts[HSL_EMA_TAIL_BINS];
    float sums[HSL_EMA_TAIL_BINS];
#else
    float unused;
#endif
};

inline HslDrawdownEmaTailStats init_hsl_drawdown_ema_tail_stats() {
    HslDrawdownEmaTailStats stats;
#if PASSIVBOT_HSL_EMA_TAIL_ENABLED
    stats.sample_count = 0.0f;
    for (int i = 0; i < HSL_EMA_TAIL_BINS; ++i) {
        stats.counts[i] = 0.0f;
        stats.sums[i] = 0.0f;
    }
#else
    stats.unused = 0.0f;
#endif
    return stats;
}

inline int hsl_drawdown_ema_tail_bin(float value) {
    return hsl_drawdown_tail_bin(value);
}

inline void update_hsl_drawdown_ema_tail_stats(
    thread HslDrawdownEmaTailStats& stats,
    float drawdown_ema
) {
#if PASSIVBOT_HSL_EMA_TAIL_ENABLED
    if (!isfinite(drawdown_ema)) return;
    float value = fabs(drawdown_ema);
    int bin = hsl_drawdown_ema_tail_bin(value);
    stats.sample_count += 1.0f;
    stats.counts[bin] += 1.0f;
    stats.sums[bin] += value;
#endif
}

inline float hsl_drawdown_ema_mean_worst_1pct(
    thread HslDrawdownEmaTailStats& stats
) {
#if PASSIVBOT_HSL_EMA_TAIL_ENABLED
    if (!(stats.sample_count > 0.0f)) return 0.0f;
    float worst_n = fmax(floor(stats.sample_count * 0.01f), 1.0f);
    float remaining = worst_n;
    float total = 0.0f;
    for (int i = HSL_EMA_TAIL_BINS - 1; i >= 0 && remaining > 0.0f; --i) {
        float count = stats.counts[i];
        if (!(count > 0.0f)) continue;
        float take = fmin(count, remaining);
        total += stats.sums[i] * (take / count);
        remaining -= take;
    }
    return total / worst_n;
#else
    return 0.0f;
#endif
}

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
#if PASSIVBOT_HSL_DIAGNOSTICS_ENABLED
    h.drawdown_ema_max = 0.0f;
#endif
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
#if PASSIVBOT_HSL_DIAGNOSTICS_ENABLED
    h.last_restart_k = -1.0f;
#endif
    h.triggers = 0.0f;
    h.restarts = 0.0f;
#if PASSIVBOT_HSL_DIAGNOSTICS_ENABLED
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
#endif
    return h;
}

inline void apply_coin_hsl_overrides(
    thread HslState& h,
    constant float* coin_overrides,
    int coin,
    int override_cols,
    int start_column
) {
    int offset = coin * override_cols + start_column;
    float value = coin_overrides[offset + 0];
    if (isfinite(value)) h.enabled = value > 0.5f;
    value = coin_overrides[offset + 1];
    if (isfinite(value)) h.red_threshold = value;
    value = coin_overrides[offset + 2];
    if (isfinite(value)) {
        h.alpha = clamp(2.0f / (fmax(value, 1.0f) + 1.0f), 0.0f, 1.0f);
    }
    value = coin_overrides[offset + 3];
    if (isfinite(value)) h.cooldown_minutes = fmax(value, 0.0f);
    value = coin_overrides[offset + 4];
    if (isfinite(value)) h.no_restart_threshold = value;
    value = coin_overrides[offset + 5];
    if (isfinite(value)) h.restart_policy = int(round(value));
    value = coin_overrides[offset + 6];
    if (isfinite(value)) h.yellow_ratio = value;
    value = coin_overrides[offset + 7];
    if (isfinite(value)) h.orange_ratio = value;
    value = coin_overrides[offset + 8];
    if (isfinite(value)) h.orange_graceful_stop = value > 0.5f;
    h.no_restart_threshold = fmax(h.no_restart_threshold, h.red_threshold);
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

inline void record_coin_hsl_realized_fill(
    thread HslState& h,
    float realized_pnl
) {
    if (!h.enabled || h.signal_mode != HSL_SIGNAL_COIN) return;
    h.coin_realized_peak = fmax(
        h.coin_realized_peak,
        realized_pnl - h.coin_realized_baseline
    );
}

inline void advance_coin_hsl_equity_after_close_fill(
    thread float& equity,
    float net_pnl,
    float qty,
    float position_price,
    float mark_price,
    float c_mult,
    bool short_side
) {
    float removed_unrealized = qty * c_mult * (
        short_side ? position_price - mark_price : mark_price - position_price
    );
    equity += net_pnl - removed_unrealized;
}

inline void advance_coin_hsl_equity_after_entry_fill(
    thread float& equity,
    float fee,
    float qty,
    float fill_price,
    float mark_price,
    float c_mult,
    bool short_side
) {
    float added_unrealized = qty * c_mult * (
        short_side ? fill_price - mark_price : mark_price - fill_price
    );
    equity += added_unrealized - fee;
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
#if PASSIVBOT_HSL_DIAGNOSTICS_ENABLED
    h.drawdown_ema_max = fmax(h.drawdown_ema_max, fabs(h.drawdown_ema));
#endif
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
#if PASSIVBOT_HSL_DIAGNOSTICS_ENABLED
                h.equity_at_halt = strategy_equity;
#endif
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
#if PASSIVBOT_HSL_DIAGNOSTICS_ENABLED
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
#endif
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

inline void update_one_side_hsl(
    thread HslState& hsl,
    float balance,
    float starting_balance,
    float realized_pnl,
    float unrealized_pnl,
    bool has_position,
    bool has_blocking_orders,
    float kf,
    float interval_ms
) {
    update_hsl(
        hsl, balance, starting_balance, realized_pnl, unrealized_pnl,
        has_position, has_blocking_orders, kf, interval_ms
    );
}

// Fused long+short kernels share account PnL and flatness in unified mode,
// while pside and single-coin coin modes retain directional scope.
inline bool update_dual_side_hsl(
    thread HslState& long_hsl,
    thread HslState& short_hsl,
    float balance,
    float starting_balance,
    float realized_pnl_total,
    float realized_pnl_long,
    float realized_pnl_short,
    float unrealized_pnl_long,
    float unrealized_pnl_short,
    bool has_position_long,
    bool has_position_short,
    bool has_blocking_orders_long,
    bool has_blocking_orders_short,
    float kf,
    float interval_ms
) {
    if (long_hsl.signal_mode != short_hsl.signal_mode) return false;
    const bool unified = long_hsl.signal_mode == HSL_SIGNAL_UNIFIED;
    const bool shared_has_position = has_position_long || has_position_short;
    const bool shared_has_blocking_orders = has_blocking_orders_long
        || has_blocking_orders_short;
    update_hsl(
        long_hsl, balance, starting_balance,
        unified ? realized_pnl_total : realized_pnl_long,
        unified ? unrealized_pnl_long + unrealized_pnl_short
            : unrealized_pnl_long,
        unified ? shared_has_position : has_position_long,
        unified ? shared_has_blocking_orders : has_blocking_orders_long,
        kf, interval_ms
    );
    update_hsl(
        short_hsl, balance, starting_balance,
        unified ? realized_pnl_total : realized_pnl_short,
        unified ? unrealized_pnl_long + unrealized_pnl_short
            : unrealized_pnl_short,
        unified ? shared_has_position : has_position_short,
        unified ? shared_has_blocking_orders : has_blocking_orders_short,
        kf, interval_ms
    );
    return true;
}

inline void try_restart_hsl(thread HslState& h, float kf, float current_equity) {
    if (!h.enabled || !h.halted || h.no_restart_latched
        || h.cooldown_until_k < 0.0f || kf < h.cooldown_until_k) return;
    if (h.current_halt_start_k >= 0.0f) {
#if PASSIVBOT_HSL_DIAGNOSTICS_ENABLED
        float duration = fmax(kf - h.current_halt_start_k, 0.0f);
        h.halt_duration_sum_steps += duration;
        h.halt_duration_max_steps = fmax(h.halt_duration_max_steps, duration);
        h.halt_duration_count += 1.0f;
#endif
        h.current_halt_start_k = -1.0f;
    }
    h.restarts += 1.0f;
#if PASSIVBOT_HSL_DIAGNOSTICS_ENABLED
    if (h.signal_mode != HSL_SIGNAL_COIN && h.equity_at_halt > 0.0f) {
        h.halt_to_restart_equity_loss += fmax(
            h.equity_at_halt - current_equity, 0.0f
        );
    }
    h.last_restart_k = kf;
#endif
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
#if PASSIVBOT_HSL_DIAGNOSTICS_ENABLED
    if (h.panic_event_start_equity < 0.0f) {
        h.panic_event_start_equity = fmax(current_equity, 1.0e-12f);
    }
    float panic_loss = fmax(-net_pnl, 0.0f);
    h.panic_event_loss += panic_loss;
    h.panic_close_loss_sum += panic_loss;
    h.panic_close_loss_max = fmax(h.panic_close_loss_max, panic_loss);
#else
    (void)h;
    (void)net_pnl;
    (void)current_equity;
#endif
}

// Keep every HSL scalar reduction in one contract. Existing one-side kernels
// and future fused dual-side kernels therefore share identical sum/max/count
// and conditional-min semantics.
struct HslOutputAggregate {
    float enabled_long;
    float enabled_short;
    float triggers_long;
    float triggers_short;
    float restarts_long;
    float restarts_short;
    float tier_samples_total;
    float tier_samples_yellow;
    float tier_samples_orange;
    float tier_samples_red;
    float duration_sum;
    float duration_max;
    float duration_count;
    float trigger_drawdown_sum;
    float trigger_drawdown_count;
    float flatten_time_sum;
    float flatten_time_count;
    float restart_retrigger_count;
    float halt_to_restart_equity_loss;
    float panic_close_loss_sum;
    float panic_close_loss_max;
    float panic_loss_drawdown_min;
    float panic_loss_drawdown_sum;
    float panic_loss_drawdown_max;
    float panic_loss_drawdown_count;
    float drawdown_ema_max_long;
    float drawdown_ema_max_short;
};

inline HslOutputAggregate init_hsl_output_aggregate(
    float tier_samples_total,
    float tier_samples_yellow,
    float tier_samples_orange,
    float tier_samples_red
) {
    HslOutputAggregate output;
    output.enabled_long = 0.0f;
    output.enabled_short = 0.0f;
    output.triggers_long = 0.0f;
    output.triggers_short = 0.0f;
    output.restarts_long = 0.0f;
    output.restarts_short = 0.0f;
    output.tier_samples_total = tier_samples_total;
    output.tier_samples_yellow = tier_samples_yellow;
    output.tier_samples_orange = tier_samples_orange;
    output.tier_samples_red = tier_samples_red;
    output.duration_sum = 0.0f;
    output.duration_max = 0.0f;
    output.duration_count = 0.0f;
    output.trigger_drawdown_sum = 0.0f;
    output.trigger_drawdown_count = 0.0f;
    output.flatten_time_sum = 0.0f;
    output.flatten_time_count = 0.0f;
    output.restart_retrigger_count = 0.0f;
    output.halt_to_restart_equity_loss = 0.0f;
    output.panic_close_loss_sum = 0.0f;
    output.panic_close_loss_max = 0.0f;
    output.panic_loss_drawdown_min = 0.0f;
    output.panic_loss_drawdown_sum = 0.0f;
    output.panic_loss_drawdown_max = 0.0f;
    output.panic_loss_drawdown_count = 0.0f;
    output.drawdown_ema_max_long = 0.0f;
    output.drawdown_ema_max_short = 0.0f;
    return output;
}

inline void accumulate_hsl_output(
    thread HslOutputAggregate& output,
    thread HslState& h,
    bool short_side,
    float last_equity_k
) {
#if PASSIVBOT_HSL_DIAGNOSTICS_ENABLED
    // Forced delist closes are panic fills even when HSL itself is disabled.
    // Exact Rust reports their loss metrics independently of controller
    // enablement, so retain those fields before filtering HSL-only telemetry.
    output.panic_close_loss_sum += h.panic_close_loss_sum;
    output.panic_close_loss_max = fmax(
        output.panic_close_loss_max, h.panic_close_loss_max
    );
    if (h.panic_loss_drawdown_count > 0.0f) {
        output.panic_loss_drawdown_min = output.panic_loss_drawdown_count > 0.0f
            ? fmin(output.panic_loss_drawdown_min, h.panic_loss_drawdown_min)
            : h.panic_loss_drawdown_min;
    }
    output.panic_loss_drawdown_sum += h.panic_loss_drawdown_sum;
    output.panic_loss_drawdown_max = fmax(
        output.panic_loss_drawdown_max, h.panic_loss_drawdown_max
    );
    output.panic_loss_drawdown_count += h.panic_loss_drawdown_count;
    if (!h.enabled) return;
    float terminal_count = h.halted
        && h.current_halt_start_k >= 0.0f && last_equity_k >= 0.0f
        ? 1.0f : 0.0f;
    float terminal_duration = terminal_count > 0.0f
        ? fmax(last_equity_k - h.current_halt_start_k, 0.0f) : 0.0f;
    if (short_side) {
        output.enabled_short = 1.0f;
        output.triggers_short += h.triggers;
        output.restarts_short += h.restarts;
        output.drawdown_ema_max_short = fmax(
            output.drawdown_ema_max_short, h.drawdown_ema_max
        );
    } else {
        output.enabled_long = 1.0f;
        output.triggers_long += h.triggers;
        output.restarts_long += h.restarts;
        output.drawdown_ema_max_long = fmax(
            output.drawdown_ema_max_long, h.drawdown_ema_max
        );
    }
    output.duration_sum += h.halt_duration_sum_steps + terminal_duration;
    output.duration_max = fmax(
        output.duration_max,
        fmax(h.halt_duration_max_steps, terminal_duration)
    );
    output.duration_count += h.halt_duration_count + terminal_count;
    output.trigger_drawdown_sum += h.trigger_drawdown_sum;
    output.trigger_drawdown_count += h.trigger_drawdown_count;
    output.flatten_time_sum += h.flatten_time_sum_steps;
    output.flatten_time_count += h.flatten_time_count;
    output.restart_retrigger_count += h.restart_retrigger_count;
    output.halt_to_restart_equity_loss += h.halt_to_restart_equity_loss;
#else
    (void)output;
    (void)h;
    (void)short_side;
    (void)last_equity_k;
#endif
}

inline void write_hsl_output_aggregate(
    thread const HslOutputAggregate& output,
    device float* scalars,
    int scalar_offset
) {
    scalars[scalar_offset + 0] = output.enabled_long;
    scalars[scalar_offset + 1] = output.enabled_short;
    scalars[scalar_offset + 2] = output.triggers_long;
    scalars[scalar_offset + 3] = output.triggers_short;
    scalars[scalar_offset + 4] = output.restarts_long;
    scalars[scalar_offset + 5] = output.restarts_short;
    scalars[scalar_offset + 6] = output.tier_samples_total;
    scalars[scalar_offset + 7] = output.tier_samples_yellow;
    scalars[scalar_offset + 8] = output.tier_samples_orange;
    scalars[scalar_offset + 9] = output.tier_samples_red;
    scalars[scalar_offset + 10] = output.duration_sum;
    scalars[scalar_offset + 11] = output.duration_max;
    scalars[scalar_offset + 12] = output.duration_count;
    scalars[scalar_offset + 13] = output.trigger_drawdown_sum;
    scalars[scalar_offset + 14] = output.trigger_drawdown_count;
    scalars[scalar_offset + 15] = output.flatten_time_sum;
    scalars[scalar_offset + 16] = output.flatten_time_count;
    scalars[scalar_offset + 17] = output.restart_retrigger_count;
    scalars[scalar_offset + 18] = output.halt_to_restart_equity_loss;
    scalars[scalar_offset + 19] = output.panic_close_loss_sum;
    scalars[scalar_offset + 20] = output.panic_close_loss_max;
    scalars[scalar_offset + 21] = output.panic_loss_drawdown_min;
    scalars[scalar_offset + 22] = output.panic_loss_drawdown_sum;
    scalars[scalar_offset + 23] = output.panic_loss_drawdown_max;
    scalars[scalar_offset + 24] = output.panic_loss_drawdown_count;
    scalars[scalar_offset + 25] = output.drawdown_ema_max_long;
    scalars[scalar_offset + 26] = output.drawdown_ema_max_short;
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
    HslOutputAggregate output = init_hsl_output_aggregate(
        tier_samples_total, tier_samples_yellow,
        tier_samples_orange, tier_samples_red
    );
    accumulate_hsl_output(output, h, short_side, last_equity_k);
    write_hsl_output_aggregate(output, scalars, scalar_offset);
}

inline void write_one_side_coin_hsl_outputs(
    thread HslState* controllers,
    int controller_count,
    bool short_side,
    float tier_samples_total,
    float tier_samples_yellow,
    float tier_samples_orange,
    float tier_samples_red,
    float last_equity_k,
    device float* scalars,
    int scalar_offset
) {
    HslOutputAggregate output = init_hsl_output_aggregate(
        tier_samples_total, tier_samples_yellow,
        tier_samples_orange, tier_samples_red
    );
    for (int c = 0; c < controller_count; ++c) {
        accumulate_hsl_output(
            output, controllers[c], short_side, last_equity_k
        );
    }
    write_hsl_output_aggregate(output, scalars, scalar_offset);
}

inline void write_dual_side_hsl_outputs(
    thread HslState& long_hsl,
    thread HslState& short_hsl,
    float tier_samples_total,
    float tier_samples_yellow,
    float tier_samples_orange,
    float tier_samples_red,
    float last_equity_k,
    device float* scalars,
    int scalar_offset
) {
    HslOutputAggregate output = init_hsl_output_aggregate(
        tier_samples_total, tier_samples_yellow,
        tier_samples_orange, tier_samples_red
    );
    accumulate_hsl_output(output, long_hsl, false, last_equity_k);
    accumulate_hsl_output(output, short_hsl, true, last_equity_k);
    write_hsl_output_aggregate(output, scalars, scalar_offset);
}

inline void write_dual_side_coin_hsl_outputs(
    thread HslState* long_controllers,
    thread HslState* short_controllers,
    int controller_count,
    float tier_samples_total,
    float tier_samples_yellow,
    float tier_samples_orange,
    float tier_samples_red,
    float last_equity_k,
    device float* scalars,
    int scalar_offset
) {
    HslOutputAggregate output = init_hsl_output_aggregate(
        tier_samples_total, tier_samples_yellow,
        tier_samples_orange, tier_samples_red
    );
    for (int c = 0; c < controller_count; ++c) {
        accumulate_hsl_output(output, long_controllers[c], false, last_equity_k);
        accumulate_hsl_output(output, short_controllers[c], true, last_equity_k);
    }
    write_hsl_output_aggregate(output, scalars, scalar_offset);
}
