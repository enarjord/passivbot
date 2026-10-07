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

struct HslState {
    HslController hsl;
    device HslNode* hsl_tree;
    device int* hsl_times;
    device float* hsl_realized;
    int hsl_lookback;
    bool hsl_owner;
    bool hsl_valid;
    bool enabled;
    float red_threshold;
    float alpha;
    float cooldown_minutes;
    int restart_policy;
    int signal_mode;
    float slot_count;
    float budget_multiplier;
    float drawdown_ema;
    float sampled_drawdown_raw;
#if PASSIVBOT_HSL_DIAGNOSTICS_ENABLED
    float drawdown_ema_max;
#endif
    int tier;
    bool red_active_now;
    bool halted;
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

inline void bind_hsl(
    thread HslState& h, device HslNode* trees, device int* rows,
    int scope, int capacity, int tree_size, int lookback, bool initialize, bool owner
) {
    h.hsl_tree = trees + scope * hsl_storage_nodes(capacity, tree_size);
    h.hsl_times = rows + scope * capacity * 2;
    h.hsl_realized = reinterpret_cast<device float*>(h.hsl_times + capacity);
    h.hsl_lookback = lookback;
    h.hsl_owner = owner;
    if (initialize) {
        h.hsl = hsl_controller_init(
            h.hsl_tree, capacity, tree_size, 2.0f / h.alpha - 1.0f);
        h.hsl_valid = true;
    }
}

#ifdef PASSIVBOT_HSL_LOOKBACK
inline void bind_hsl_multicoin_hsl(
    thread HslState& aggregate, thread HslState* coins,
    device HslNode* trees, device int* rows, int scope_base, int coin_count,
    bool initialize, bool owner
) {
    bind_hsl(aggregate, trees, rows, scope_base,
        PASSIVBOT_HSL_CAPACITY, PASSIVBOT_HSL_TREE_SIZE,
        PASSIVBOT_HSL_LOOKBACK, initialize, owner);
    for (int c = 0; c < coin_count; ++c) {
        bind_hsl(coins[c], trees, rows, scope_base + 1 + c,
            PASSIVBOT_HSL_CAPACITY, PASSIVBOT_HSL_TREE_SIZE,
            PASSIVBOT_HSL_LOOKBACK, initialize, true);
    }
}

inline bool valid_hsl_multicoin_hsl(
    thread HslState& aggregate, thread HslState* coins, int coin_count
) {
    if (aggregate.signal_mode != HSL_SIGNAL_COIN)
        return !aggregate.enabled || aggregate.hsl_valid;
    for (int c = 0; c < coin_count; ++c)
        if (coins[c].enabled && !coins[c].hsl_valid) return false;
    return true;
}

#endif

inline void mirror_hsl(thread HslState& owner, thread HslState& view) {
    view.hsl = owner.hsl;
    view.hsl_valid = owner.hsl_valid;
    view.halted = owner.halted;
    view.red_active_now = owner.red_active_now;
    view.tier = owner.tier;
    view.drawdown_ema = owner.drawdown_ema;
    view.sampled_drawdown_raw = owner.sampled_drawdown_raw;
}

// Reporting only: finish one contiguous panic loss segment. GREEN cancels
// panic intent, so a later RED starts a new loss denominator even before flat.
inline void finish_hsl_panic_loss(thread HslState& h) {
#if PASSIVBOT_HSL_DIAGNOSTICS_ENABLED
    if (h.panic_event_start_equity >= 0.0f) {
        float ratio = h.panic_event_loss / h.panic_event_start_equity;
        h.panic_loss_drawdown_min = h.panic_loss_drawdown_count > 0.0f
            ? fmin(h.panic_loss_drawdown_min, ratio) : ratio;
        h.panic_loss_drawdown_sum += ratio;
        h.panic_loss_drawdown_max = fmax(h.panic_loss_drawdown_max, ratio);
        h.panic_loss_drawdown_count += 1.0f;
        h.panic_event_start_equity = -1.0f;
        h.panic_event_loss = 0.0f;
    }
#else
    (void)h;
#endif
}

inline void observe_hsl(
    thread HslState& h, float balance, float realized, float upnl,
    bool exposed, int minute, bool terminal
) {
    if (!h.enabled || !h.hsl_valid) return;
    int prior = h.hsl.action;
    h.hsl_valid = hsl_observe(h.hsl, h.hsl_tree,
        h.hsl_times, h.hsl_realized, minute, h.hsl_lookback,
        balance / (h.signal_mode == HSL_SIGNAL_COIN ? h.slot_count : 1.0f)
            * (h.signal_mode == HSL_SIGNAL_COIN ? h.budget_multiplier : 1.0f),
        realized, upnl, exposed, terminal, h.red_threshold,
        h.cooldown_minutes, h.restart_policy == 2);
    if (!h.hsl_valid) return;
    h.sampled_drawdown_raw = h.hsl.raw;
    h.drawdown_ema = h.hsl.ema;
    h.red_active_now = h.hsl.action == 3;
    if (terminal || (prior == 3 && !h.red_active_now)) finish_hsl_panic_loss(h);
    h.tier = h.red_active_now ? 3 : 0;
    h.halted = h.hsl.action == 1;
    if (h.red_active_now && prior != 3) h.current_red_start_k = float(minute);
    if (terminal && fmin(h.hsl.raw, h.hsl.ema) > h.red_threshold) {
        h.triggers += 1.0f;
        if (h.halted) h.current_halt_start_k = float(minute);
#if PASSIVBOT_HSL_DIAGNOSTICS_ENABLED
        h.trigger_drawdown_sum += fmin(h.hsl.raw, h.hsl.ema);
        h.trigger_drawdown_count += 1.0f;
        if (h.current_red_start_k >= 0.0f) {
            h.flatten_time_sum_steps += fmax(float(minute) - h.current_red_start_k, 0.0f);
            h.flatten_time_count += 1.0f;
        }
#endif
    }
    if (prior == 1 && !h.halted) {
        h.restarts += 1.0f;
#if PASSIVBOT_HSL_DIAGNOSTICS_ENABLED
        float duration = fmax(float(minute) - h.current_halt_start_k, 0.0f);
        h.halt_duration_sum_steps += duration;
        h.halt_duration_max_steps = fmax(h.halt_duration_max_steps, duration);
        h.halt_duration_count += 1.0f;
        h.last_restart_k = float(minute);
#endif
        h.current_halt_start_k = -1.0f;
    }
    if (!h.red_active_now) h.current_red_start_k = -1.0f;
#if PASSIVBOT_HSL_DIAGNOSTICS_ENABLED
    h.drawdown_ema_max = fmax(h.drawdown_ema_max, h.drawdown_ema);
#endif
}

// Reporting marks the controller RED while panic or a terminal cooldown blocks
// ordinary trading. Keep this separate from the current panic signal's tier.
inline int hsl_report_tier(thread const HslState& h) {
    return h.enabled && (h.red_active_now || h.halted) ? 3 : 0;
}

// Reporting runs after forced closes and never advances the trading controller.
// A negative tier means this side has no enabled reporting scope.
inline int record_multicoin_hsl_report(
    thread HslState& aggregate, thread HslState* coins, int coin_count,
    int effective_n_positions, bool strategy_eq_eligible,
#if PASSIVBOT_HSL_EMA_TAIL_ENABLED
    thread HslDrawdownEmaTailStats& ema_tail,
#endif
    thread HslStrategyEquityStats& strategy_eq,
    float equity, int day_index
) {
    bool enabled = aggregate.enabled;
    int tier = hsl_report_tier(aggregate);
#if PASSIVBOT_HSL_EMA_TAIL_ENABLED
    float drawdown_ema = fabs(aggregate.drawdown_ema);
#endif
    if (aggregate.signal_mode == HSL_SIGNAL_COIN) {
        enabled = false;
        tier = 0;
#if PASSIVBOT_HSL_EMA_TAIL_ENABLED
        drawdown_ema = 0.0f;
#endif
        if (effective_n_positions > 0) {
            for (int c = 0; c < coin_count; ++c) {
                enabled = enabled || coins[c].enabled;
                tier = max(tier, hsl_report_tier(coins[c]));
#if PASSIVBOT_HSL_EMA_TAIL_ENABLED
                if (coins[c].enabled)
                    drawdown_ema = fmax(drawdown_ema, fabs(coins[c].drawdown_ema));
#endif
            }
        }
        strategy_eq_eligible = enabled;
    }
    if (strategy_eq_eligible) {
#if PASSIVBOT_HSL_EMA_TAIL_ENABLED
        update_hsl_drawdown_ema_tail_stats(ema_tail, drawdown_ema);
#endif
        update_hsl_strategy_equity_stats(strategy_eq, equity, day_index);
    }
    return enabled ? tier : -1;
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
    h.restart_policy = int(round(params[ho + 4]));
    h.signal_mode = int(round(params[ho + 5]));
    h.slot_count = fmax(round(params[ho + 6]), 1.0f);
    h.budget_multiplier = 1.0f;
    h.drawdown_ema = 0.0f;
    h.sampled_drawdown_raw = 0.0f;
#if PASSIVBOT_HSL_DIAGNOSTICS_ENABLED
    h.drawdown_ema_max = 0.0f;
#endif
    h.tier = 0;
    h.red_active_now = false;
    h.halted = false;
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
    if (isfinite(value)) h.restart_policy = int(round(value));
}

inline int hsl_mode(thread HslState& h, bool has_position) {
    return h.enabled ? h.hsl.action : 0;
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
    observe_hsl(h, balance, realized_pnl, unrealized_pnl,
        has_position, int(kf) + 1, false);
}

// A real closing fill samples its final fee-inclusive drawdown before an ordinary
// episode reset. Proven-flat RED boundaries finalize before later fills can reopen.
inline bool finish_hsl_episode_at_flat(
    thread HslState& h,
    float balance,
    float starting_balance,
    float realized_pnl,
    float kf,
    float interval_ms
) {
    // A closing fill can exhaust raw cash before the kernel's end-of-bar
    // liquidation check. There is no remaining HSL budget to observe; retain
    // the controller and let liquidation record its floor sample. Non-finite
    // budgets still reach HSL validation and remain fatal.
    if (isfinite(balance) && balance <= 0.0f) return true;
    observe_hsl(h, balance, realized_pnl, 0.0f, false, int(kf), true);
    return true;
}

inline bool finish_hsl_scoped_episode_at_flat(
    thread HslState& h,
    thread HslState* opposite_hsl,
    bool scope_has_position,
    bool opposite_has_position,
    float balance,
    float starting_balance,
    float realized_total,
    float realized_scope,
    float kf,
    float interval_ms
) {
    const bool unified = h.signal_mode == HSL_SIGNAL_UNIFIED;
    if (scope_has_position || (unified && opposite_has_position)) return false;
    if (unified && opposite_hsl != nullptr) {
        thread HslState& owner = h.hsl_owner ? h : *opposite_hsl;
        thread HslState& view = h.hsl_owner ? *opposite_hsl : h;
        finish_hsl_episode_at_flat(owner, balance, starting_balance, realized_total, kf, interval_ms);
        mirror_hsl(owner, view);
        return true;
    }
    bool reset = finish_hsl_episode_at_flat(
        h, balance, starting_balance, unified ? realized_total : realized_scope, kf, interval_ms
    );
    if (unified && opposite_hsl != nullptr) {
        finish_hsl_episode_at_flat(
            *opposite_hsl, balance, starting_balance, realized_total, kf, interval_ms
        );
    }
    return reset;
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
    if (unified) {
        thread HslState& owner = long_hsl.hsl_owner ? long_hsl : short_hsl;
        thread HslState& view = long_hsl.hsl_owner ? short_hsl : long_hsl;
        update_hsl(owner, balance, starting_balance, realized_pnl_total,
            unrealized_pnl_long + unrealized_pnl_short, shared_has_position,
            shared_has_blocking_orders, kf, interval_ms);
        mirror_hsl(owner, view);
        return owner.hsl_valid;
    }
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
    return long_hsl.hsl_valid && short_hsl.hsl_valid;
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

// CPU reporting assigns elapsed time to the preceding observation's RED state.
// An initial observation and a terminal state have no extra duration of their own.
struct HslTimeObservation {
    float observed_steps;
    float red_steps;
    float last_step;
    bool was_red;
};

inline HslTimeObservation init_hsl_time_observation() {
    HslTimeObservation observation;
    observation.observed_steps = 0.0f;
    observation.red_steps = 0.0f;
    observation.last_step = -1.0f;
    observation.was_red = false;
    return observation;
}

// A terminal accounting boundary advances time without inventing a fresh tier.
inline void advance_hsl_time_observation(
    thread HslTimeObservation& observation, float step
) {
    if (observation.last_step >= 0.0f) {
        const float elapsed = fmax(step - observation.last_step, 0.0f);
        observation.observed_steps += elapsed;
        if (observation.was_red) observation.red_steps += elapsed;
    }
    observation.last_step = fmax(observation.last_step, step);
}

inline void record_hsl_time_observation(
    thread HslTimeObservation& observation, float step, int tier
) {
    advance_hsl_time_observation(observation, step);
    observation.was_red = tier == 3;
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
    // Include a censored unfinished panic segment in the final report.
    HslState report = h;
    finish_hsl_panic_loss(report);
    // Forced delist closes are panic fills even when HSL itself is disabled.
    // Exact Rust reports their loss metrics independently of controller
    // enablement, so retain those fields before filtering HSL-only telemetry.
    output.panic_close_loss_sum += h.panic_close_loss_sum;
    output.panic_close_loss_max = fmax(
        output.panic_close_loss_max, h.panic_close_loss_max
    );
    if (report.panic_loss_drawdown_count > 0.0f) {
        output.panic_loss_drawdown_min = output.panic_loss_drawdown_count > 0.0f
            ? fmin(output.panic_loss_drawdown_min, report.panic_loss_drawdown_min)
            : report.panic_loss_drawdown_min;
    }
    output.panic_loss_drawdown_sum += report.panic_loss_drawdown_sum;
    output.panic_loss_drawdown_max = fmax(
        output.panic_loss_drawdown_max, report.panic_loss_drawdown_max
    );
    output.panic_loss_drawdown_count += report.panic_loss_drawdown_count;
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
    scalars[scalar_offset + 7] = output.tier_samples_red;
    scalars[scalar_offset + 8] = output.duration_sum;
    scalars[scalar_offset + 9] = output.duration_max;
    scalars[scalar_offset + 10] = output.duration_count;
    scalars[scalar_offset + 11] = output.trigger_drawdown_sum;
    scalars[scalar_offset + 12] = output.trigger_drawdown_count;
    scalars[scalar_offset + 13] = output.flatten_time_sum;
    scalars[scalar_offset + 14] = output.flatten_time_count;
    scalars[scalar_offset + 15] = output.restart_retrigger_count;
    scalars[scalar_offset + 16] = output.halt_to_restart_equity_loss;
    scalars[scalar_offset + 17] = output.panic_close_loss_sum;
    scalars[scalar_offset + 18] = output.panic_close_loss_max;
    scalars[scalar_offset + 19] = output.panic_loss_drawdown_min;
    scalars[scalar_offset + 20] = output.panic_loss_drawdown_sum;
    scalars[scalar_offset + 21] = output.panic_loss_drawdown_max;
    scalars[scalar_offset + 22] = output.panic_loss_drawdown_count;
    scalars[scalar_offset + 23] = output.drawdown_ema_max_long;
    scalars[scalar_offset + 24] = output.drawdown_ema_max_short;
}

inline void write_one_side_hsl_outputs(
    thread HslState& h,
    bool short_side,
    float tier_samples_total,
    float tier_samples_red,
    float last_equity_k,
    device float* scalars,
    int scalar_offset
) {
    HslOutputAggregate output = init_hsl_output_aggregate(
        tier_samples_total, tier_samples_red
    );
    accumulate_hsl_output(output, h, short_side, last_equity_k);
    write_hsl_output_aggregate(output, scalars, scalar_offset);
}

inline void write_one_side_coin_hsl_outputs(
    thread HslState* controllers,
    int controller_count,
    bool short_side,
    float tier_samples_total,
    float tier_samples_red,
    float last_equity_k,
    device float* scalars,
    int scalar_offset
) {
    HslOutputAggregate output = init_hsl_output_aggregate(
        tier_samples_total, tier_samples_red
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
    float tier_samples_red,
    float last_equity_k,
    device float* scalars,
    int scalar_offset
) {
    HslOutputAggregate output = init_hsl_output_aggregate(
        tier_samples_total, tier_samples_red
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
    float tier_samples_red,
    float last_equity_k,
    device float* scalars,
    int scalar_offset
) {
    HslOutputAggregate output = init_hsl_output_aggregate(
        tier_samples_total, tier_samples_red
    );
    for (int c = 0; c < controller_count; ++c) {
        accumulate_hsl_output(output, long_controllers[c], false, last_equity_k);
        accumulate_hsl_output(output, short_controllers[c], true, last_equity_k);
    }
    write_hsl_output_aggregate(output, scalars, scalar_offset);
}
