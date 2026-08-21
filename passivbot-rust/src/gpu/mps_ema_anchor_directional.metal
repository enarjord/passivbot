#include <metal_stdlib>
using namespace metal;

constant int DAILY_COLS = 7;
constant int SCALAR_COLS = 50;
constant int GAP_BINS = 128;
constant int SIDE_PARAMS = 34;

inline float round_step(float value, float step) {
    return floor(value / step + 0.5f) * step;
}

inline float ceil_step(float value, float step) {
    return ceil(value / step - 1.0e-6f) * step;
}

inline float floor_step(float value, float step) {
    return floor(value / step + 1.0e-6f) * step;
}

inline float min_entry_qty(
    float price, float qty_step, float min_qty, float min_cost, float c_mult
) {
    float raw_min = fmax(min_qty, min_cost / fmax(price, 1.0e-12f) / c_mult);
    float raw_steps = raw_min / qty_step;
    float nearest_count = floor(raw_steps + 0.5f);
    float nearest = nearest_count * qty_step;
    float representation_tolerance = 1.1920928955078125e-7f
        * fmax(fabs(raw_min), fabs(nearest)) * 4.0f;
    bool aligned = nearest_count > 0.0f && fabs(raw_steps - nearest_count) <= 1.0e-8f
        && (nearest >= raw_min || raw_min - nearest <= representation_tolerance);
    return aligned ? fmax(nearest, raw_min) : ceil(raw_steps) * qty_step;
}

inline bool passes_min_effective_cost(
    bool enabled, float guaranteed_balance_lower, float wel,
    float initial_qty_pct, float max_effective_min_cost
) {
    if (!enabled) return true;
    float rounded_projected_cost = guaranteed_balance_lower * wel * initial_qty_pct;
    // Discount by 16 float32 unit roundoffs. This covers upward encoding and
    // multiply rounding of all three operands before the conservative compare.
    float projected_cost_lower = rounded_projected_cost
        * (1.0f - 9.5367431640625e-7f);
    return isfinite(rounded_projected_cost) && rounded_projected_cost > 0.0f
        && projected_cost_lower >= max_effective_min_cost;
}

inline bool realized_loss_gate_allows(
    float net_pnl, float remaining_loss_budget,
    bool gate_enabled
) {
    return !gate_enabled || net_pnl >= 0.0f
        || -net_pnl <= remaining_loss_budget;
}

inline float float32_floor_nonnegative(float value) {
    if (!(value > 0.0f) || !isfinite(value)) return fmax(value, 0.0f);
    return as_type<float>(as_type<uint>(value) - 1u);
}

inline void record_realized_net(
    float net_pnl,
    thread float& realized_pnl_cumsum_last,
    thread float& realized_pnl_cumsum_max
) {
    realized_pnl_cumsum_last += net_pnl;
    realized_pnl_cumsum_max = fmax(
        realized_pnl_cumsum_max, realized_pnl_cumsum_last
    );
}

inline void record_gross_pnl(
    float pnl, thread float& profit_sum, thread float& loss_sum
) {
    if (pnl > 0.0f) profit_sum += pnl;
    else loss_sum += fabs(pnl);
}

struct EmaSide {
    float alpha0;
    float alpha1;
    float alpha2;
    float alpha1m;
    float alpha1h;
    float base_qty_pct;
    float ddf;
    float offset;
    float psize_weight;
    float w1h;
    float w1m;
    float cooldown_min;
    float twel;
    float allowed_wel;
    float entry_cap;
    float twel_enforcer_threshold;
    bool twel_enforcer_enabled;
    bool unstuck_enabled;
    bool unstuck_ema_gating_enabled;
    float unstuck_close_pct;
    float unstuck_ema_dist;
    float unstuck_loss_allowance_pct;
    float unstuck_threshold;
    float ema0;
    float ema1;
    float ema2;
    float vol1m;
    float vol1h;
    float psize;
    float pprice;
    float last_inc_k;
    float pos_open_k;
    int entry_ticks;
    float entry_qty;
    int close_ticks;
    float close_qty;
    int secondary_close_ticks;
    float secondary_close_qty;
    int close_without_reducer_ticks;
    float close_without_reducer_qty;
    bool close_is_protective_reducer;
    bool close_is_panic;
};

struct ReducerVariant {
    bool valid;
    bool is_unstuck;
    int ticks;
    float qty;
    int secondary_ticks;
    float secondary_qty;
};

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
    bool signal_coin;
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

inline HslState load_hsl(constant float* params, int po) {
    HslState h;
    h.enabled = params[po + 23] > 0.5f;
    h.red_threshold = params[po + 24];
    h.alpha = clamp(2.0f / (fmax(params[po + 25], 1.0f) + 1.0f), 0.0f, 1.0f);
    h.cooldown_minutes = fmax(params[po + 26], 0.0f);
    h.no_restart_threshold = fmax(params[po + 27], h.red_threshold);
    h.restart_policy = int(round(params[po + 28]));
    h.yellow_ratio = params[po + 29];
    h.orange_ratio = params[po + 30];
    h.orange_graceful_stop = params[po + 31] > 0.5f;
    h.signal_coin = params[po + 32] > 0.5f;
    h.slot_count = fmax(round(params[po + 33]), 1.0f);
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
    if (!h.enabled || h.halted || !(balance > 0.0f)) return;
    float drawdown_raw;
    float strategy_pnl = realized_pnl + unrealized_pnl;
    float strategy_equity;
    float peak_strategy_equity;
    if (h.signal_coin) {
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
        if (!(strategy_equity > 0.0f && peak_strategy_equity > 0.0f)) return;
        drawdown_raw = fmin(
            fmax(1.0f - strategy_equity / peak_strategy_equity, 0.0f),
            0.9999999403953552f
        );
    }
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
                if (h.signal_coin) {
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
                float no_restart_drawdown_raw = h.signal_coin
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
    if (!h.signal_coin && h.equity_at_halt > 0.0f) {
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

inline EmaSide load_side(constant float* params, int po, float seed_close) {
    EmaSide side;
    float span0 = params[po + 1];
    float span1 = params[po + 2];
    float span2 = sqrt(span0 * span1);
    float lo_span = fmin(span0, fmin(span1, span2));
    float hi_span = fmax(span0, fmax(span1, span2));
    float mid_span = span0 + span1 + span2 - lo_span - hi_span;
    side.alpha0 = clamp(2.0f / (lo_span + 1.0f), 0.0f, 1.0f);
    side.alpha1 = clamp(2.0f / (mid_span + 1.0f), 0.0f, 1.0f);
    side.alpha2 = clamp(2.0f / (hi_span + 1.0f), 0.0f, 1.0f);
    float span_h = params[po + 8];
    float span_m = params[po + 9];
    side.alpha1h = span_h > 0.0f ? 2.0f / (fmax(span_h, 1.0f) + 1.0f) : 0.0f;
    side.alpha1m = span_m > 0.0f
        ? clamp(2.0f / (span_m + 1.0f), 0.0f, 1.0f) : 0.0f;
    side.base_qty_pct = params[po + 0];
    side.ddf = params[po + 3];
    side.offset = params[po + 4];
    side.psize_weight = params[po + 5];
    side.w1h = params[po + 6];
    side.w1m = params[po + 7];
    side.cooldown_min = ceil(params[po + 10]);
    side.twel = params[po + 11];
    float allowance_pct = fmax(params[po + 12], 0.0f);
    bool legacy_raw_allowance = params[po + 13] > 0.5f;
    side.allowed_wel = side.twel * (
        1.0f + (legacy_raw_allowance ? allowance_pct : 0.0f)
    );
    bool twel_entry_gate_enabled = params[po + 14] > 0.5f;
    float twel_threshold = params[po + 15];
    float gate_cap = side.twel;
    if (isfinite(twel_threshold) && twel_threshold > 0.0f) {
        gate_cap = fmin(side.twel, side.twel * twel_threshold);
    }
    side.entry_cap = twel_entry_gate_enabled ? gate_cap : INFINITY;
    side.twel_enforcer_threshold = twel_threshold;
    side.twel_enforcer_enabled = params[po + 16] > 0.5f;
    side.unstuck_enabled = params[po + 17] > 0.5f;
    side.unstuck_ema_gating_enabled = params[po + 18] > 0.5f;
    side.unstuck_close_pct = params[po + 19];
    side.unstuck_ema_dist = params[po + 20];
    side.unstuck_loss_allowance_pct = params[po + 21];
    side.unstuck_threshold = params[po + 22];
    side.ema0 = seed_close;
    side.ema1 = seed_close;
    side.ema2 = seed_close;
    side.vol1m = 0.0f;
    side.vol1h = 0.0f;
    side.psize = 0.0f;
    side.pprice = 0.0f;
    side.last_inc_k = -1.0f;
    side.pos_open_k = -1.0f;
    side.entry_ticks = 0;
    side.entry_qty = 0.0f;
    side.close_ticks = 0;
    side.close_qty = 0.0f;
    side.secondary_close_ticks = 0;
    side.secondary_close_qty = 0.0f;
    side.close_without_reducer_ticks = 0;
    side.close_without_reducer_qty = 0.0f;
    side.close_is_protective_reducer = false;
    side.close_is_panic = false;
    return side;
}

inline ReducerVariant empty_reducer_variant() {
    ReducerVariant result;
    result.valid = false;
    result.is_unstuck = false;
    result.ticks = 0;
    result.qty = 0.0f;
    result.secondary_ticks = 0;
    result.secondary_qty = 0.0f;
    return result;
}

inline ReducerVariant finalize_reducer_variant(
    float psize,
    int ordinary_ticks,
    float ordinary_qty,
    int reducer_ticks,
    float reducer_qty,
    bool is_unstuck,
    float qty_step,
    float price_step,
    float min_qty,
    float min_cost,
    float c_mult
) {
    ReducerVariant result = empty_reducer_variant();
    if (!(psize > 0.0f && reducer_ticks > 0 && reducer_qty > 0.0f)) {
        return result;
    }
    float reducer_price = float(reducer_ticks) * price_step;
    float reducer_min = min_entry_qty(
        reducer_price, qty_step, min_qty, min_cost, c_mult
    );
    reducer_qty = fmin(psize, reducer_qty);
    if (ordinary_qty > 0.0f && ordinary_ticks > 0) {
        float ordinary_price = float(ordinary_ticks) * price_step;
        float ordinary_min = min_entry_qty(
            ordinary_price, qty_step, min_qty, min_cost, c_mult
        );
        if (ordinary_qty + reducer_qty > psize) {
            ordinary_qty = fmax(round_step(psize - reducer_qty, qty_step), 0.0f);
        }
        if (ordinary_qty >= ordinary_min) {
            float remainder = fmax(
                round_step(psize - reducer_qty - ordinary_qty, qty_step), 0.0f
            );
            float minimum_any = fmin(ordinary_min, reducer_min);
            if (remainder > 0.0f && remainder < minimum_any) {
                ordinary_qty = fmin(
                    psize - reducer_qty,
                    round_step(ordinary_qty + remainder, qty_step)
                );
            }
            result.secondary_ticks = ordinary_ticks;
            result.secondary_qty = ordinary_qty;
        }
    }
    if (result.secondary_qty <= 0.0f) {
        float remainder = fmax(round_step(psize - reducer_qty, qty_step), 0.0f);
        if (remainder > 0.0f && remainder < reducer_min) {
            reducer_qty = psize;
        }
    }
    result.valid = reducer_qty > 0.0f;
    result.is_unstuck = is_unstuck;
    result.ticks = reducer_ticks;
    result.qty = reducer_qty;
    return result;
}

inline void apply_reducer_variant(
    thread EmaSide& side,
    ReducerVariant variant
) {
    side.close_ticks = variant.ticks;
    side.close_qty = variant.qty;
    side.secondary_close_ticks = variant.secondary_ticks;
    side.secondary_close_qty = variant.secondary_qty;
    side.close_is_protective_reducer = variant.valid;
}

inline void restore_ordinary_close(thread EmaSide& side) {
    side.close_ticks = side.close_without_reducer_ticks;
    side.close_qty = side.close_without_reducer_qty;
    side.secondary_close_ticks = 0;
    side.secondary_close_qty = 0.0f;
    side.close_is_protective_reducer = false;
}

inline ReducerVariant generated_reducer_variant(thread EmaSide& side) {
    if (!side.close_is_protective_reducer || !(side.close_qty > 0.0f)) {
        return empty_reducer_variant();
    }
    ReducerVariant result;
    result.valid = true;
    result.is_unstuck = false;
    result.ticks = side.close_ticks;
    result.qty = side.close_qty;
    result.secondary_ticks = side.secondary_close_ticks;
    result.secondary_qty = side.secondary_close_qty;
    return result;
}

inline float total_exposure_reducer_qty(
    float psize, float pprice, float balance, float target_exposure,
    float reducer_price, float qty_step, float min_qty, float min_cost,
    float c_mult
) {
    if (!(balance > 0.0f && psize > 0.0f && pprice > 0.0f
        && target_exposure > 0.0f && reducer_price > 0.0f)) {
        return 0.0f;
    }
    float current_exposure = psize * pprice * c_mult / balance;
    if (!(current_exposure > target_exposure + 1.0e-9f)) return 0.0f;
    float reducer_min = min_entry_qty(
        reducer_price, qty_step, min_qty, min_cost, c_mult
    );
    float exposure_to_cut = current_exposure - target_exposure;
    float requested_qty = ceil_step(
        exposure_to_cut * balance / fmax(pprice * c_mult, 1.0e-12f),
        qty_step
    );
    float reducer_qty = fmin(
        psize, fmax(reducer_min, requested_qty)
    );
    return fmin(psize, ceil_step(reducer_qty, qty_step));
}

inline void apply_twel_reducer(
    thread EmaSide& side,
    bool is_long,
    float balance,
    float price_now,
    float qty_step,
    float price_step,
    float min_qty,
    float min_cost,
    float c_mult
) {
    side.secondary_close_ticks = 0;
    side.secondary_close_qty = 0.0f;
    side.close_is_protective_reducer = false;
    float target = side.twel * side.twel_enforcer_threshold;
    int reducer_ticks = is_long
        ? int(floor(price_now * 0.9995f / price_step + 1.0e-6f))
        : int(ceil(price_now * 1.0005f / price_step - 1.0e-6f));
    reducer_ticks = max(reducer_ticks, 1);
    float reducer_price = float(reducer_ticks) * price_step;
    float reducer_qty = side.twel_enforcer_enabled
            && side.twel_enforcer_threshold > 0.0f
        ? total_exposure_reducer_qty(
            side.psize, side.pprice, balance, target, reducer_price,
            qty_step, min_qty, min_cost, c_mult
        )
        : 0.0f;
    if (!(reducer_qty > 0.0f)) return;

    apply_reducer_variant(
        side,
        finalize_reducer_variant(
            side.psize, side.close_ticks, side.close_qty,
            reducer_ticks, reducer_qty, false, qty_step, price_step,
            min_qty, min_cost, c_mult
        )
    );
}

inline ReducerVariant unstuck_reducer_variant(
    thread EmaSide& side,
    bool is_long,
    float balance,
    float balance_peak,
    float price_now,
    int touch_down_ticks,
    int touch_up_ticks,
    float qty_step,
    float price_step,
    float min_qty,
    float min_cost,
    float c_mult
) {
    ReducerVariant none = empty_reducer_variant();
    if (!(side.unstuck_enabled
        && side.unstuck_loss_allowance_pct > 0.0f
        && side.unstuck_close_pct > 0.0f
        && side.unstuck_threshold > 0.0f
        && balance > 0.0f
        && balance_peak > 0.0f
        && side.psize > 0.0f
        && side.pprice > 0.0f
        && side.allowed_wel > 0.0f
        && price_now > 0.0f)) {
        return none;
    }
    float allowance_pct = side.unstuck_loss_allowance_pct * side.twel;
    float allowance = float32_floor_nonnegative(
        fmax(balance - balance_peak * (1.0f - allowance_pct), 0.0f)
    );
    if (!(allowance > 0.0f)) return none;
    float wallet_exposure = side.psize * side.pprice * c_mult / balance;
    if (!(wallet_exposure / side.allowed_wel > side.unstuck_threshold)) {
        return none;
    }
    if (side.unstuck_ema_gating_enabled) {
        float lower = fmin(side.ema0, fmin(side.ema1, side.ema2));
        float upper = fmax(side.ema0, fmax(side.ema1, side.ema2));
        int trigger_ticks = is_long
            ? int(ceil(
                upper * (1.0f + side.unstuck_ema_dist) / price_step
                    - 1.0e-6f
            ))
            : int(floor(
                lower * (1.0f - side.unstuck_ema_dist) / price_step
                    + 1.0e-6f
            ));
        bool triggered = is_long
            ? touch_down_ticks >= trigger_ticks
            : touch_up_ticks <= trigger_ticks;
        if (!triggered) return none;
    }

    int reducer_ticks = max(
        is_long ? touch_up_ticks : touch_down_ticks, 1
    );
    float reducer_price = float(reducer_ticks) * price_step;
    float reducer_min = min_entry_qty(
        reducer_price, qty_step, min_qty, min_cost, c_mult
    );
    float target_qty = floor_step(
        balance * side.allowed_wel * side.unstuck_close_pct
            / fmax(reducer_price * c_mult, 1.0e-12f),
        qty_step
    );
    float reducer_qty = fmin(
        side.psize, fmax(reducer_min, target_qty)
    );
    float gross_pnl = reducer_qty * c_mult * (
        is_long ? reducer_price - side.pprice : side.pprice - reducer_price
    );
    if (gross_pnl < 0.0f && -gross_pnl > allowance) {
        float scaled_qty = fmin(
            side.psize, reducer_qty * allowance / -gross_pnl
        );
        reducer_qty = fmin(
            side.psize,
            fmax(reducer_min, floor_step(scaled_qty, qty_step))
        );
    }
    return finalize_reducer_variant(
        side.psize,
        side.close_without_reducer_ticks,
        side.close_without_reducer_qty,
        reducer_ticks,
        reducer_qty,
        true,
        qty_step,
        price_step,
        min_qty,
        min_cost,
        c_mult
    );
}

inline bool reducer_variant_preferred(
    ReducerVariant left,
    ReducerVariant right,
    bool is_long
) {
    if (!left.valid) return false;
    if (!right.valid) return true;
    if (left.qty != right.qty) return left.qty > right.qty;
    if (left.ticks != right.ticks) {
        return is_long ? left.ticks < right.ticks : left.ticks > right.ticks;
    }
    if (left.is_unstuck != right.is_unstuck) return left.is_unstuck;
    return false;
}

inline void order_reducer_variants(
    ReducerVariant first,
    ReducerVariant second,
    bool is_long,
    thread ReducerVariant& preferred,
    thread ReducerVariant& fallback
) {
    if (reducer_variant_preferred(second, first, is_long)) {
        preferred = second;
        fallback = first;
    } else {
        preferred = first;
        fallback = second;
    }
}

inline ReducerVariant reducer_variant_at(
    ReducerVariant preferred,
    ReducerVariant fallback,
    int index
) {
    return index == 0 ? preferred : fallback;
}

inline bool gate_reducer_variant(
    ReducerVariant variant,
    float pprice,
    bool is_long,
    float price_step,
    float c_mult,
    float maker_fee,
    bool gate_enabled,
    thread float& remaining_loss_budget
) {
    if (!variant.valid || !(variant.qty > 0.0f && pprice > 0.0f)) {
        return false;
    }
    float price = float(variant.ticks) * price_step;
    float gross_pnl = variant.qty * c_mult * (
        is_long ? price - pprice : pprice - price
    );
    float net_pnl = gross_pnl - variant.qty * price * c_mult * maker_fee;
    if (!realized_loss_gate_allows(
            net_pnl, remaining_loss_budget, gate_enabled)) {
        return false;
    }
    if (gate_enabled && net_pnl < 0.0f) {
        remaining_loss_budget = float32_floor_nonnegative(
            fmax(remaining_loss_budget + net_pnl, 0.0f)
        );
    }
    return true;
}

inline void update_indicators(
    thread EmaSide& side,
    float close,
    float log_range,
    float hour_lr,
    bool valid,
    bool hour_valid
) {
    if (hour_valid && side.alpha1h > 0.0f) {
        side.vol1h = fma(side.alpha1h, hour_lr - side.vol1h, side.vol1h);
    }
    if (valid) {
        side.ema0 = fma(side.alpha0, close - side.ema0, side.ema0);
        side.ema1 = fma(side.alpha1, close - side.ema1, side.ema1);
        side.ema2 = fma(side.alpha2, close - side.ema2, side.ema2);
        if (side.alpha1m > 0.0f) {
            side.vol1m = fma(side.alpha1m, log_range - side.vol1m, side.vol1m);
        }
    }
}

inline void generate_long_orders(
    thread EmaSide& side,
    float balance,
    float price_now,
    int touch_down_ticks,
    int touch_up_ticks,
    float qty_step,
    float price_step,
    float min_qty,
    float min_cost,
    float c_mult,
    float kf,
    bool block_initial
) {
    float lower = fmin(side.ema0, fmin(side.ema1, side.ema2));
    float upper = fmax(side.ema0, fmax(side.ema1, side.ema2));
    float mult = fmax(1.0f + side.vol1h * side.w1h + side.vol1m * side.w1m, 1.0f);
    float eff_off = side.offset * mult;
    float current_we = side.psize > 0.0f && balance > 0.0f
        ? side.psize * price_now * c_mult / balance : 0.0f;
    float current_cost_we = side.psize > 0.0f && balance > 0.0f
        ? side.psize * side.pprice * c_mult / balance : 0.0f;
    float swer = side.psize > 0.0f && balance > 0.0f
        ? current_we / fmax(side.twel, 1.0e-12f) : 0.0f;
    float inv_shift = swer * side.psize_weight;

    int bid_ticks = min(
        int(floor(lower * (1.0f - eff_off - inv_shift) / price_step + 1.0e-6f)),
        touch_down_ticks
    );
    float bid_price = float(bid_ticks) * price_step;
    float min_q = min_entry_qty(bid_price, qty_step, min_qty, min_cost, c_mult);
    float base_q = fmax(min_q, round_step(
        balance * side.allowed_wel * side.base_qty_pct
            / fmax(bid_price, 1.0e-12f) / c_mult,
        qty_step
    ));
    float e_qty = round_step(
        base_q * fmax(1.0f + fmax(swer, 0.0f) * side.ddf, 1.0f), qty_step
    );
    bool cooldown = side.cooldown_min > 0.0f && side.last_inc_k >= 0.0f
        && kf < side.last_inc_k + side.cooldown_min;
    float cap = side.entry_cap - 1.0e-7f;
    float headroom = (cap * balance - side.psize * side.pprice * c_mult)
        / fmax(bid_price * c_mult, 1.0e-12f);
    bool over = (side.psize * side.pprice + e_qty * bid_price) * c_mult
        / fmax(balance, 1.0e-9f) >= cap;
    float capped = floor_step(headroom, qty_step);
    if (over) e_qty = capped > 0.0f && capped + 1.0e-6f >= min_q ? capped : 0.0f;
    if (current_cost_we >= cap || cooldown || bid_price <= 0.0f || balance <= 0.0f
        || side.base_qty_pct <= 0.0f
        || (block_initial && side.psize <= 0.0f)) {
        e_qty = 0.0f;
    }
    side.entry_ticks = bid_ticks;
    side.entry_qty = e_qty;

    int ask_ticks = max(
        int(ceil(upper * (1.0f + eff_off - inv_shift) / price_step - 1.0e-6f)),
        touch_up_ticks
    );
    float ask_price = float(ask_ticks) * price_step;
    float min_cq = min_entry_qty(ask_price, qty_step, min_qty, min_cost, c_mult);
    float clip = fmin(side.psize, fmax(min_cq, round_step(
        balance * side.allowed_wel * side.base_qty_pct
            / fmax(ask_price, 1.0e-12f) / c_mult,
        qty_step
    )));
    float c_qty = side.psize <= min_cq || side.psize - clip < min_cq
        ? side.psize : clip;
    if (side.psize <= 0.0f || ask_price <= 0.0f) c_qty = 0.0f;
    side.close_ticks = ask_ticks;
    side.close_qty = c_qty;
    side.close_without_reducer_ticks = ask_ticks;
    side.close_without_reducer_qty = c_qty;
    apply_twel_reducer(
        side, true, balance, price_now, qty_step, price_step,
        min_qty, min_cost, c_mult
    );
}

inline void generate_short_orders(
    thread EmaSide& side,
    float balance,
    float price_now,
    int touch_down_ticks,
    int touch_up_ticks,
    float qty_step,
    float price_step,
    float min_qty,
    float min_cost,
    float c_mult,
    float kf,
    bool block_initial
) {
    float lower = fmin(side.ema0, fmin(side.ema1, side.ema2));
    float upper = fmax(side.ema0, fmax(side.ema1, side.ema2));
    float mult = fmax(1.0f + side.vol1h * side.w1h + side.vol1m * side.w1m, 1.0f);
    float eff_off = side.offset * mult;
    float swer = side.psize > 0.0f && balance > 0.0f
        ? -side.psize * price_now * c_mult / balance / fmax(side.twel, 1.0e-12f)
        : 0.0f;
    float current_cost_we = side.psize > 0.0f && balance > 0.0f
        ? side.psize * side.pprice * c_mult / balance : 0.0f;
    float inv_shift = swer * side.psize_weight;

    int ask_ticks = max(
        int(ceil(upper * (1.0f + eff_off - inv_shift) / price_step - 1.0e-6f)),
        touch_up_ticks
    );
    float ask_price = float(ask_ticks) * price_step;
    float min_q = min_entry_qty(ask_price, qty_step, min_qty, min_cost, c_mult);
    float base_q = fmax(min_q, round_step(
        balance * side.allowed_wel * side.base_qty_pct
            / fmax(ask_price, 1.0e-12f) / c_mult,
        qty_step
    ));
    float e_qty = round_step(
        base_q * fmax(1.0f + fmax(-swer, 0.0f) * side.ddf, 1.0f), qty_step
    );
    bool cooldown = side.cooldown_min > 0.0f && side.last_inc_k >= 0.0f
        && kf < side.last_inc_k + side.cooldown_min;
    float cap = side.entry_cap - 1.0e-7f;
    float headroom = (cap * balance - side.psize * side.pprice * c_mult)
        / fmax(ask_price * c_mult, 1.0e-12f);
    bool over = (side.psize * side.pprice + e_qty * ask_price) * c_mult
        / fmax(balance, 1.0e-9f) >= cap;
    float capped = floor_step(headroom, qty_step);
    if (over) e_qty = capped > 0.0f && capped + 1.0e-6f >= min_q ? capped : 0.0f;
    if (current_cost_we >= cap || cooldown || ask_price <= 0.0f || balance <= 0.0f
        || side.base_qty_pct <= 0.0f
        || (block_initial && side.psize <= 0.0f)) {
        e_qty = 0.0f;
    }
    side.entry_ticks = ask_ticks;
    side.entry_qty = e_qty;

    int bid_ticks = min(
        int(floor(lower * (1.0f - eff_off - inv_shift) / price_step + 1.0e-6f)),
        touch_down_ticks
    );
    float bid_price = float(bid_ticks) * price_step;
    float min_cq = min_entry_qty(bid_price, qty_step, min_qty, min_cost, c_mult);
    float clip = fmin(side.psize, fmax(min_cq, round_step(
        balance * side.allowed_wel * side.base_qty_pct
            / fmax(bid_price, 1.0e-12f) / c_mult,
        qty_step
    )));
    float c_qty = side.psize <= min_cq || side.psize - clip < min_cq
        ? side.psize : clip;
    if (side.psize <= 0.0f || bid_price <= 0.0f) c_qty = 0.0f;
    side.close_ticks = bid_ticks;
    side.close_qty = c_qty;
    side.close_without_reducer_ticks = bid_ticks;
    side.close_without_reducer_qty = c_qty;
    apply_twel_reducer(
        side, false, balance, price_now, qty_step, price_step,
        min_qty, min_cost, c_mult
    );
}

inline void gate_generated_close(
    thread float& qty,
    int ticks,
    float pprice,
    bool is_long,
    float price_step,
    float c_mult,
    float maker_fee,
    bool gate_enabled,
    thread float& remaining_loss_budget
) {
    if (!(qty > 0.0f && ticks > 0 && pprice > 0.0f)) return;
    float price = float(ticks) * price_step;
    float gross_pnl = qty * c_mult
        * (is_long ? price - pprice : pprice - price);
    float net_pnl = gross_pnl - qty * price * c_mult * maker_fee;
    if (!realized_loss_gate_allows(
            net_pnl, remaining_loss_budget, gate_enabled)) {
        qty = 0.0f;
        return;
    }
    if (gate_enabled && net_pnl < 0.0f) {
        remaining_loss_budget = float32_floor_nonnegative(
            fmax(remaining_loss_budget + net_pnl, 0.0f)
        );
    }
}

inline void passivbot_single_coin_impl(
    constant float* bars,
    constant int* flags,
    constant float* params,
    constant float* settings,
    constant int* sizes,
    device float* daily,
    device float* scalars,
    device int* gap_hist,
    uint b
) {
    const int B = sizes[0];
    const int T = sizes[1];
    const int D = sizes[2];
    const int P = sizes[3];
    const int first_valid = sizes[4];
    if (b >= uint(B)) return;

    const float qty_step = settings[0];
    const float price_step = settings[1];
    const float min_qty = settings[2];
    const float min_cost = settings[3];
    const float c_mult = settings[4];
    const float maker_fee = settings[5];
    const float starting_balance = settings[6];
    const float liq_floor = settings[7];
    const float interval_ms = settings[8];
    const bool long_enabled = settings[9] > 0.5f;
    const bool short_enabled = settings[10] > 0.5f;
    const bool hedge_mode = settings[11] > 0.5f;
    const bool filter_by_min_effective_cost = settings[12] > 0.5f;
    const float max_effective_min_cost = settings[13];
    const float max_realized_loss_pct = settings[14];
    const float taker_fee = settings[15];
    const float market_order_slippage_pct = fmax(settings[16], 0.0f);
    const bool long_hsl_panic_market = settings[17] > 0.5f;
    const bool short_hsl_panic_market = settings[18] > 0.5f;
    const float log_bin_scale = 127.0f / log(4000001.0f);

    const int po = int(b) * P;
    const int seed_k = clamp(first_valid, 0, T - 1);
    const float seed_close = bars[seed_k * 5 + 2];
    EmaSide long_side = load_side(params, po, seed_close);
    EmaSide short_side = load_side(params, po + SIDE_PARAMS, seed_close);
    HslState long_hsl = load_hsl(params, po);
    HslState short_hsl = load_hsl(params, po + SIDE_PARAMS);

    float balance = starting_balance;
    float realized_pnl_cumsum_last = 0.0f;
    float realized_pnl_cumsum_max = 0.0f;
    float profit_sum = 0.0f;
    float loss_sum = 0.0f;
    bool alive = true;
    int liq_day = -1;
    float held_max_min = 0.0f;
    float position_unchanged_max_min = 0.0f;
    float long_position_last_fill_k = -1.0f;
    float short_position_last_fill_k = -1.0f;
    float last_fill_k = -1.0f;
    float first_fill_k = -1.0f;
    float gap_max_min = 0.0f;
    float run_peak = -INFINITY;
    float max_dd = 0.0f;
    float total_wallet_exposure_max = 0.0f;
    float total_wallet_exposure_mean = 0.0f;
    float total_wallet_exposure_samples = 0.0f;
    float last_high_k = -1.0f;
    float recovery_max_min = 0.0f;
    float first_eq_k = -1.0f;
    float last_eq_k = -1.0f;
    bool eq_started = false;
    float hsl_tier_samples_total = 0.0f;
    float hsl_tier_samples_yellow = 0.0f;
    float hsl_tier_samples_orange = 0.0f;
    float hsl_tier_samples_red = 0.0f;

    int cur_day = flags[2];
    bool day_touched = false;
    float day_end = 0.0f;
    float day_min = INFINITY;
    float day_dd = 0.0f;
    float day_volume = 0.0f;
    float day_has_fill = 0.0f;
    float day_start_balance = balance;

    for (int j = 0; j < GAP_BINS; ++j) {
        gap_hist[int(b) * GAP_BINS + j] = 0;
    }

    for (int k = 1; k < T - 1; ++k) {
        const int bo = k * 5;
        const int fo = k * 11;
        const float high = bars[bo + 0];
        const float low = bars[bo + 1];
        const float close = bars[bo + 2];
        const float log_range = bars[bo + 3];
        const float hour_lr = bars[bo + 4];
        const bool valid = flags[fo + 0] != 0;
        const bool can_gen = flags[fo + 1] != 0;
        const int di = flags[fo + 2];
        const bool hour_valid = flags[fo + 3] != 0;
        const int high_fill_max_tick = flags[fo + 4];
        const int low_nonfill_max_tick = flags[fo + 5];
        const int touch_down_tick = flags[fo + 6];
        const int touch_up_tick = flags[fo + 7];
        const float kf = float(k);

        if (di != cur_day) {
            if (day_touched && cur_day >= 0 && cur_day < D) {
                int o = (int(b) * D + cur_day) * DAILY_COLS;
                daily[o + 0] = day_end;
                daily[o + 1] = day_min;
                daily[o + 2] = day_dd;
                daily[o + 3] = day_volume;
                daily[o + 4] = day_has_fill;
                daily[o + 5] = balance - day_start_balance;
                daily[o + 6] = balance;
            }
            cur_day = di;
            day_touched = false;
            day_end = 0.0f;
            day_min = INFINITY;
            day_dd = 0.0f;
            day_volume = 0.0f;
            day_has_fill = 0.0f;
            day_start_balance = balance;
        }

        bool long_close_fill = false;
        bool long_primary_close_fill = valid && alive && long_enabled
            && long_side.close_qty > 0.0f && long_side.psize > 0.0f
            && ((long_side.close_is_panic && long_hsl_panic_market)
                || long_side.close_ticks <= high_fill_max_tick);
        bool long_secondary_close_fill = valid && alive && long_enabled
            && long_side.secondary_close_qty > 0.0f && long_side.psize > 0.0f
            && long_side.secondary_close_ticks <= high_fill_max_tick;
        bool long_secondary_first = long_secondary_close_fill
            && (!long_primary_close_fill
                || long_side.secondary_close_ticks < long_side.close_ticks);
        for (int rank = 0; rank < 2; ++rank) {
            bool use_secondary = long_secondary_first ? rank == 0 : rank == 1;
            bool reachable = use_secondary
                ? long_secondary_close_fill : long_primary_close_fill;
            if (!reachable || long_side.psize <= 0.0f) continue;
            int close_ticks = use_secondary
                ? long_side.secondary_close_ticks : long_side.close_ticks;
            float requested_qty = use_secondary
                ? long_side.secondary_close_qty : long_side.close_qty;
            bool market_panic = !use_secondary && long_side.close_is_panic
                && long_hsl_panic_market;
            float cp = market_panic
                ? fmax(
                    floor_step(
                        close * (1.0f - market_order_slippage_pct), price_step
                    ),
                    price_step
                )
                : float(close_ticks) * price_step;
            float adj = fmin(round_step(requested_qty, qty_step), long_side.psize);
            if (!(adj > 0.0f)) continue;
            float pnl = adj * c_mult * (cp - long_side.pprice);
            float fee = adj * cp * c_mult
                * (market_panic ? taker_fee : maker_fee);
            float net_pnl = pnl - fee;
            if (!use_secondary && long_side.close_is_panic) {
                float current_equity = balance
                    + long_side.psize * c_mult * (close - long_side.pprice)
                    + (short_side.psize > 0.0f
                        ? short_side.psize * c_mult * (short_side.pprice - close)
                        : 0.0f);
                record_hsl_panic_fill(long_hsl, net_pnl, current_equity);
            }
            record_gross_pnl(pnl, profit_sum, loss_sum);
            balance += net_pnl;
            record_realized_net(
                net_pnl, realized_pnl_cumsum_last, realized_pnl_cumsum_max
            );
            float new_psize = fmax(round_step(long_side.psize - adj, qty_step), 0.0f);
            bool went_flat = new_psize <= 0.0f;
            long_side.psize = new_psize;
            if (went_flat) {
                long_side.pprice = 0.0f;
                if (long_side.pos_open_k >= 0.0f) {
                    held_max_min = fmax(held_max_min, kf - long_side.pos_open_k);
                }
                long_side.pos_open_k = -1.0f;
            }
            day_volume += fabs(adj) * cp / balance;
            long_close_fill = true;
            if (use_secondary) {
                long_side.secondary_close_qty = 0.0f;
            } else {
                long_side.close_qty = 0.0f;
            }
        }

        bool long_entry_fill = valid && alive && long_enabled
            && long_side.entry_qty > 0.0f
            && long_side.entry_ticks > low_nonfill_max_tick;
        if (long_entry_fill) {
            float ep = float(long_side.entry_ticks) * price_step;
            float eq = round_step(long_side.entry_qty, qty_step);
            float fee = eq * ep * c_mult * maker_fee;
            balance -= fee;
            record_realized_net(
                -fee, realized_pnl_cumsum_last, realized_pnl_cumsum_max
            );
            bool was_flat = long_side.psize <= 0.0f;
            float new_psize = round_step(long_side.psize + eq, qty_step);
            float new_pprice = was_flat ? ep
                : long_side.pprice * (long_side.psize / fmax(new_psize, 1.0e-12f))
                    + ep * (eq / fmax(new_psize, 1.0e-12f));
            if (was_flat) long_side.pos_open_k = kf;
            long_side.psize = new_psize;
            long_side.pprice = new_pprice;
            long_side.last_inc_k = kf;
            day_volume += fabs(eq) * ep / balance;
            long_side.entry_qty = 0.0f;
        }

        bool short_close_fill = false;
        bool short_primary_close_fill = valid && alive && short_enabled
            && short_side.close_qty > 0.0f && short_side.psize > 0.0f
            && ((short_side.close_is_panic && short_hsl_panic_market)
                || short_side.close_ticks > low_nonfill_max_tick);
        bool short_secondary_close_fill = valid && alive && short_enabled
            && short_side.secondary_close_qty > 0.0f && short_side.psize > 0.0f
            && short_side.secondary_close_ticks > low_nonfill_max_tick;
        bool short_secondary_first = short_secondary_close_fill
            && (!short_primary_close_fill
                || short_side.secondary_close_ticks > short_side.close_ticks);
        for (int rank = 0; rank < 2; ++rank) {
            bool use_secondary = short_secondary_first ? rank == 0 : rank == 1;
            bool reachable = use_secondary
                ? short_secondary_close_fill : short_primary_close_fill;
            if (!reachable || short_side.psize <= 0.0f) continue;
            int close_ticks = use_secondary
                ? short_side.secondary_close_ticks : short_side.close_ticks;
            float requested_qty = use_secondary
                ? short_side.secondary_close_qty : short_side.close_qty;
            bool market_panic = !use_secondary && short_side.close_is_panic
                && short_hsl_panic_market;
            float cp = market_panic
                ? fmax(
                    ceil_step(
                        close * (1.0f + market_order_slippage_pct), price_step
                    ),
                    price_step
                )
                : float(close_ticks) * price_step;
            float adj = fmin(round_step(requested_qty, qty_step), short_side.psize);
            if (!(adj > 0.0f)) continue;
            float pnl = adj * c_mult * (short_side.pprice - cp);
            float fee = adj * cp * c_mult
                * (market_panic ? taker_fee : maker_fee);
            float net_pnl = pnl - fee;
            if (!use_secondary && short_side.close_is_panic) {
                float current_equity = balance
                    + (long_side.psize > 0.0f
                        ? long_side.psize * c_mult * (close - long_side.pprice)
                        : 0.0f)
                    + short_side.psize * c_mult * (short_side.pprice - close);
                record_hsl_panic_fill(short_hsl, net_pnl, current_equity);
            }
            record_gross_pnl(pnl, profit_sum, loss_sum);
            balance += net_pnl;
            record_realized_net(
                net_pnl, realized_pnl_cumsum_last, realized_pnl_cumsum_max
            );
            float new_psize = fmax(round_step(short_side.psize - adj, qty_step), 0.0f);
            bool went_flat = new_psize <= 0.0f;
            short_side.psize = new_psize;
            if (went_flat) {
                short_side.pprice = 0.0f;
                if (short_side.pos_open_k >= 0.0f) {
                    held_max_min = fmax(held_max_min, kf - short_side.pos_open_k);
                }
                short_side.pos_open_k = -1.0f;
            }
            day_volume += fabs(adj) * cp / balance;
            short_close_fill = true;
            if (use_secondary) {
                short_side.secondary_close_qty = 0.0f;
            } else {
                short_side.close_qty = 0.0f;
            }
        }

        bool short_entry_fill = valid && alive && short_enabled
            && short_side.entry_qty > 0.0f
            && short_side.entry_ticks <= high_fill_max_tick;
        if (short_entry_fill) {
            float ep = float(short_side.entry_ticks) * price_step;
            float eq = round_step(short_side.entry_qty, qty_step);
            float fee = eq * ep * c_mult * maker_fee;
            balance -= fee;
            record_realized_net(
                -fee, realized_pnl_cumsum_last, realized_pnl_cumsum_max
            );
            bool was_flat = short_side.psize <= 0.0f;
            float new_psize = round_step(short_side.psize + eq, qty_step);
            float new_pprice = was_flat ? ep
                : short_side.pprice * (short_side.psize / fmax(new_psize, 1.0e-12f))
                    + ep * (eq / fmax(new_psize, 1.0e-12f));
            if (was_flat) short_side.pos_open_k = kf;
            short_side.psize = new_psize;
            short_side.pprice = new_pprice;
            short_side.last_inc_k = kf;
            day_volume += fabs(eq) * ep / balance;
            short_side.entry_qty = 0.0f;
        }

        if (long_close_fill || long_entry_fill) {
            if (long_position_last_fill_k >= 0.0f) {
                position_unchanged_max_min = fmax(
                    position_unchanged_max_min, kf - long_position_last_fill_k
                );
            }
            long_position_last_fill_k = long_side.psize > 0.0f ? kf : -1.0f;
        }
        if (short_close_fill || short_entry_fill) {
            if (short_position_last_fill_k >= 0.0f) {
                position_unchanged_max_min = fmax(
                    position_unchanged_max_min, kf - short_position_last_fill_k
                );
            }
            short_position_last_fill_k = short_side.psize > 0.0f ? kf : -1.0f;
        }

        bool any_fill = long_close_fill || long_entry_fill
            || short_close_fill || short_entry_fill;
        if (any_fill) {
            day_has_fill = 1.0f;
            day_touched = true;
            if (last_fill_k >= 0.0f) {
                float gap = kf - last_fill_k;
                int bin = clamp(
                    int(log(fmax(gap, 0.0f) + 1.0f) * log_bin_scale), 0, 127
                );
                gap_hist[int(b) * GAP_BINS + bin] += 1;
                gap_max_min = fmax(gap_max_min, gap);
            }
            if (first_fill_k < 0.0f) first_fill_k = kf;
            last_fill_k = kf;
        }

        if (long_enabled) {
            update_indicators(long_side, close, log_range, hour_lr, valid, hour_valid);
        }
        if (short_enabled) {
            update_indicators(short_side, close, log_range, hour_lr, valid, hour_valid);
        }

        bool gen = can_gen && alive;
        eq_started = eq_started || gen;
        int long_hsl_mode = hsl_mode(long_hsl, long_side.psize > 0.0f);
        int short_hsl_mode = hsl_mode(short_hsl, short_side.psize > 0.0f);
        if (gen) {
            long_side.close_is_panic = false;
            short_side.close_is_panic = false;
            // When both sides are flat, an exact Rust path that remains alive has
            // balance above liq_floor. If either side is open, equity cannot bound
            // exact cash balance, so flat-side eligibility uses zero and fails closed.
            float guaranteed_balance_lower =
                long_side.psize <= 0.0f && short_side.psize <= 0.0f
                ? liq_floor : 0.0f;
            bool long_min_cost_eligible = passes_min_effective_cost(
                filter_by_min_effective_cost, guaranteed_balance_lower,
                long_side.allowed_wel, long_side.base_qty_pct, max_effective_min_cost
            );
            bool short_min_cost_eligible = passes_min_effective_cost(
                filter_by_min_effective_cost, guaranteed_balance_lower,
                short_side.allowed_wel, short_side.base_qty_pct, max_effective_min_cost
            );
            bool block_long_initial = !long_min_cost_eligible || long_hsl_mode != 0;
            bool block_short_initial = !short_min_cost_eligible || short_hsl_mode != 0;
            if (long_enabled && short_enabled && !hedge_mode) {
                if (long_side.psize > 0.0f) {
                    block_short_initial = true;
                } else if (short_side.psize > 0.0f) {
                    block_long_initial = true;
                } else if (long_min_cost_eligible && !short_min_cost_eligible) {
                    block_short_initial = true;
                } else if (!long_min_cost_eligible && short_min_cost_eligible) {
                    block_long_initial = true;
                } else if (long_min_cost_eligible && short_min_cost_eligible) {
                    float long_lower = fmin(
                        long_side.ema0, fmin(long_side.ema1, long_side.ema2)
                    );
                    float short_upper = fmax(
                        short_side.ema0, fmax(short_side.ema1, short_side.ema2)
                    );
                    float dist_long = long_lower * (1.0f - long_side.offset)
                        / close - 1.0f;
                    float dist_short = 1.0f
                        - short_upper * (1.0f + short_side.offset) / close;
                    if (dist_long >= dist_short) {
                        block_short_initial = true;
                    } else {
                        block_long_initial = true;
                    }
                }
            }
            if (long_enabled) {
                generate_long_orders(
                    long_side, balance, close, touch_down_tick, touch_up_tick,
                    qty_step, price_step, min_qty, min_cost, c_mult, kf,
                    block_long_initial
                );
            }
            if (short_enabled) {
                generate_short_orders(
                    short_side, balance, close, touch_down_tick, touch_up_tick,
                    qty_step, price_step, min_qty, min_cost, c_mult, kf,
                    block_short_initial
                );
            }

            bool loss_gate_enabled = max_realized_loss_pct < 1.0f;
            float balance_peak = balance
                + (realized_pnl_cumsum_max - realized_pnl_cumsum_last);
            float allowed_loss_budget = float32_floor_nonnegative(
                balance_peak * fmax(max_realized_loss_pct, 0.0f)
            );
            float current_realized_loss = fmax(
                realized_pnl_cumsum_max - realized_pnl_cumsum_last, 0.0f
            );
            float remaining_loss_budget = float32_floor_nonnegative(
                fmax(allowed_loss_budget - current_realized_loss, 0.0f)
            );

            ReducerVariant long_twel = generated_reducer_variant(long_side);
            ReducerVariant short_twel = generated_reducer_variant(short_side);
            ReducerVariant long_unstuck = long_enabled
                ? unstuck_reducer_variant(
                    long_side, true, balance, balance_peak, close,
                    touch_down_tick, touch_up_tick, qty_step, price_step,
                    min_qty, min_cost, c_mult
                )
                : empty_reducer_variant();
            ReducerVariant short_unstuck = short_enabled
                ? unstuck_reducer_variant(
                    short_side, false, balance, balance_peak, close,
                    touch_down_tick, touch_up_tick, qty_step, price_step,
                    min_qty, min_cost, c_mult
                )
                : empty_reducer_variant();

            // Exact Rust emits at most one global unstuck intent: the eligible
            // long/short position with the lowest pside-aware price distance.
            if (long_unstuck.valid && short_unstuck.valid) {
                float long_diff = 1.0f - close / long_side.pprice;
                float short_diff = close / short_side.pprice - 1.0f;
                if (long_diff <= short_diff) {
                    short_unstuck = empty_reducer_variant();
                } else {
                    long_unstuck = empty_reducer_variant();
                }
            }

            ReducerVariant long_preferred, long_fallback;
            ReducerVariant short_preferred, short_fallback;
            order_reducer_variants(
                long_twel, long_unstuck, true,
                long_preferred, long_fallback
            );
            order_reducer_variants(
                short_twel, short_unstuck, false,
                short_preferred, short_fallback
            );
            restore_ordinary_close(long_side);
            restore_ordinary_close(short_side);

            ReducerVariant selected_long = empty_reducer_variant();
            ReducerVariant selected_short = empty_reducer_variant();
            int long_candidate_index = 0;
            int short_candidate_index = 0;
            bool long_resolved = !long_preferred.valid;
            bool short_resolved = !short_preferred.valid;
            // Final reducer sizes compete globally largest-first. A blocked
            // candidate advances only its own position to the next reducer.
            for (int attempt = 0; attempt < 4; ++attempt) {
                ReducerVariant long_candidate = reducer_variant_at(
                    long_preferred, long_fallback, long_candidate_index
                );
                ReducerVariant short_candidate = reducer_variant_at(
                    short_preferred, short_fallback, short_candidate_index
                );
                bool long_available = !long_resolved && long_candidate.valid;
                bool short_available = !short_resolved && short_candidate.valid;
                if (!long_available && !short_available) break;
                bool use_long = long_available && (
                    !short_available || long_candidate.qty >= short_candidate.qty
                );
                if (use_long) {
                    if (gate_reducer_variant(
                            long_candidate, long_side.pprice, true,
                            price_step, c_mult, maker_fee, loss_gate_enabled,
                            remaining_loss_budget)) {
                        selected_long = long_candidate;
                        long_resolved = true;
                    } else {
                        long_candidate_index += 1;
                        long_resolved = long_candidate_index > 1
                            || !long_fallback.valid;
                    }
                } else {
                    if (gate_reducer_variant(
                            short_candidate, short_side.pprice, false,
                            price_step, c_mult, maker_fee, loss_gate_enabled,
                            remaining_loss_budget)) {
                        selected_short = short_candidate;
                        short_resolved = true;
                    } else {
                        short_candidate_index += 1;
                        short_resolved = short_candidate_index > 1
                            || !short_fallback.valid;
                    }
                }
            }
            if (selected_long.valid) {
                apply_reducer_variant(long_side, selected_long);
            }
            if (selected_short.valid) {
                apply_reducer_variant(short_side, selected_short);
            }

            // Ordinary closes consume any remaining allowance in canonical
            // long-then-short order after protective reducers are finalized.
            if (long_enabled) {
                if (long_side.close_is_protective_reducer) {
                    gate_generated_close(
                        long_side.secondary_close_qty,
                        long_side.secondary_close_ticks, long_side.pprice, true,
                        price_step, c_mult, maker_fee, loss_gate_enabled,
                        remaining_loss_budget
                    );
                } else {
                    gate_generated_close(
                        long_side.close_qty, long_side.close_ticks,
                        long_side.pprice, true, price_step, c_mult, maker_fee,
                        loss_gate_enabled, remaining_loss_budget
                    );
                }
            }
            if (short_enabled) {
                if (short_side.close_is_protective_reducer) {
                    gate_generated_close(
                        short_side.secondary_close_qty,
                        short_side.secondary_close_ticks, short_side.pprice,
                        false, price_step, c_mult, maker_fee, loss_gate_enabled,
                        remaining_loss_budget
                    );
                } else {
                    gate_generated_close(
                        short_side.close_qty, short_side.close_ticks,
                        short_side.pprice, false, price_step, c_mult, maker_fee,
                        loss_gate_enabled, remaining_loss_budget
                    );
                }
            }

            if (long_enabled && long_hsl_mode >= 2) {
                long_side.entry_qty = 0.0f;
            }
            if (short_enabled && short_hsl_mode >= 2) {
                short_side.entry_qty = 0.0f;
            }
            if (long_enabled && long_hsl_mode == 3) {
                long_side.close_ticks = max(touch_down_tick - 1, 1);
                long_side.close_qty = long_side.psize;
                long_side.secondary_close_qty = 0.0f;
                long_side.close_is_protective_reducer = false;
                long_side.close_is_panic = true;
            }
            if (short_enabled && short_hsl_mode == 3) {
                short_side.close_ticks = max(touch_up_tick + 1, 1);
                short_side.close_qty = short_side.psize;
                short_side.secondary_close_qty = 0.0f;
                short_side.close_is_protective_reducer = false;
                short_side.close_is_panic = true;
            }
        }

        float long_unreal = long_side.psize > 0.0f
            ? long_side.psize * c_mult * (close - long_side.pprice) : 0.0f;
        float short_unreal = short_side.psize > 0.0f
            ? short_side.psize * c_mult * (short_side.pprice - close) : 0.0f;
        float equity = balance + long_unreal + short_unreal;
        if (gen && valid && alive && balance > 0.0f && equity > liq_floor) {
            bool long_blocking_orders = long_hsl_mode != 3 && (
                long_side.entry_qty > 0.0f || long_side.close_qty > 0.0f
                    || long_side.secondary_close_qty > 0.0f
            );
            bool short_blocking_orders = short_hsl_mode != 3 && (
                short_side.entry_qty > 0.0f || short_side.close_qty > 0.0f
                    || short_side.secondary_close_qty > 0.0f
            );
            update_hsl(
                long_hsl, balance, starting_balance,
                realized_pnl_cumsum_last,
                long_unreal, long_side.psize > 0.0f,
                long_blocking_orders, kf, interval_ms
            );
            update_hsl(
                short_hsl, balance, starting_balance,
                realized_pnl_cumsum_last,
                short_unreal, short_side.psize > 0.0f,
                short_blocking_orders, kf, interval_ms
            );
            if (long_hsl.enabled || short_hsl.enabled) {
                int hsl_tier = max(long_hsl.tier, short_hsl.tier);
                hsl_tier_samples_total += 1.0f;
                hsl_tier_samples_yellow += hsl_tier == 1 ? 1.0f : 0.0f;
                hsl_tier_samples_orange += hsl_tier == 2 ? 1.0f : 0.0f;
                hsl_tier_samples_red += hsl_tier == 3 ? 1.0f : 0.0f;
            }
            try_restart_hsl(long_hsl, kf, equity);
            try_restart_hsl(short_hsl, kf, equity);
        }
        bool active = eq_started && alive && valid;
        if (active) {
            if (first_eq_k < 0.0f) first_eq_k = kf;
            last_eq_k = kf;
            bool liq = balance <= 0.0f || equity <= liq_floor;
            float eqf = liq ? liq_floor : equity;
            if (eqf > run_peak) {
                if (last_high_k >= 0.0f) {
                    recovery_max_min = fmax(recovery_max_min, kf - last_high_k);
                }
                last_high_k = kf;
                run_peak = eqf;
            }
            float dd = fmax(
                (run_peak - eqf) / fmax(fabs(run_peak), 1.0e-12f), 0.0f
            );
            max_dd = fmax(max_dd, dd);
            day_end = eqf;
            day_min = fmin(day_min, eqf);
            day_dd = fmax(day_dd, dd);
            day_touched = true;
            if (!liq) {
                float twe_net = (
                    long_side.psize * long_side.pprice
                    - short_side.psize * short_side.pprice
                ) * c_mult / balance;
                float twe_abs = fabs(twe_net);
                total_wallet_exposure_samples += 1.0f;
                total_wallet_exposure_mean += (
                    twe_abs - total_wallet_exposure_mean
                ) / total_wallet_exposure_samples;
                total_wallet_exposure_max = fmax(
                    total_wallet_exposure_max, twe_abs
                );
            }
            if (liq) {
                liq_day = di;
                alive = false;
            }
        }
    }

    if (day_touched && cur_day >= 0 && cur_day < D) {
        int o = (int(b) * D + cur_day) * DAILY_COLS;
        daily[o + 0] = day_end;
        daily[o + 1] = day_min;
        daily[o + 2] = day_dd;
        daily[o + 3] = day_volume;
        daily[o + 4] = day_has_fill;
        daily[o + 5] = balance - day_start_balance;
        daily[o + 6] = balance;
    }

    if (long_side.pos_open_k >= 0.0f && last_eq_k >= 0.0f) {
        held_max_min = fmax(held_max_min, last_eq_k - long_side.pos_open_k);
    }
    if (short_side.pos_open_k >= 0.0f && last_eq_k >= 0.0f) {
        held_max_min = fmax(held_max_min, last_eq_k - short_side.pos_open_k);
    }
    if (long_position_last_fill_k >= 0.0f && last_eq_k >= 0.0f) {
        position_unchanged_max_min = fmax(
            position_unchanged_max_min, last_eq_k - long_position_last_fill_k
        );
    }
    if (short_position_last_fill_k >= 0.0f && last_eq_k >= 0.0f) {
        position_unchanged_max_min = fmax(
            position_unchanged_max_min, last_eq_k - short_position_last_fill_k
        );
    }
    int so = int(b) * SCALAR_COLS;
    scalars[so + 0] = max_dd;
    scalars[so + 1] = held_max_min * interval_ms;
    scalars[so + 2] = gap_max_min * interval_ms;
    scalars[so + 3] = first_fill_k >= 0.0f ? first_fill_k * interval_ms : -1.0f;
    scalars[so + 4] = last_fill_k >= 0.0f ? last_fill_k * interval_ms : -1.0f;
    scalars[so + 5] = recovery_max_min * interval_ms;
    scalars[so + 6] = last_high_k >= 0.0f ? last_high_k * interval_ms : -1.0f;
    scalars[so + 7] = first_eq_k >= 0.0f ? first_eq_k * interval_ms : -1.0f;
    scalars[so + 8] = last_eq_k >= 0.0f ? last_eq_k * interval_ms : -1.0f;
    scalars[so + 9] = float(liq_day);
    scalars[so + 10] = balance;
    scalars[so + 11] = long_side.psize;
    scalars[so + 12] = long_side.pprice;
    scalars[so + 13] = alive ? 1.0f : 0.0f;
    scalars[so + 14] = long_side.psize > 0.0f || short_side.psize > 0.0f ? 1.0f : 0.0f;
    scalars[so + 15] = short_side.psize;
    scalars[so + 16] = short_side.pprice;
    scalars[so + 17] = 0.0f;
    float long_terminal_count = long_hsl.halted
        && long_hsl.current_halt_start_k >= 0.0f && last_eq_k >= 0.0f
        ? 1.0f : 0.0f;
    float short_terminal_count = short_hsl.halted
        && short_hsl.current_halt_start_k >= 0.0f && last_eq_k >= 0.0f
        ? 1.0f : 0.0f;
    float long_terminal_duration = long_terminal_count > 0.0f
        ? fmax(last_eq_k - long_hsl.current_halt_start_k, 0.0f) : 0.0f;
    float short_terminal_duration = short_terminal_count > 0.0f
        ? fmax(last_eq_k - short_hsl.current_halt_start_k, 0.0f) : 0.0f;
    float terminal_count = long_terminal_count + short_terminal_count;
    scalars[so + 18] = long_hsl.enabled ? 1.0f : 0.0f;
    scalars[so + 19] = short_hsl.enabled ? 1.0f : 0.0f;
    scalars[so + 20] = long_hsl.triggers;
    scalars[so + 21] = short_hsl.triggers;
    scalars[so + 22] = long_hsl.restarts;
    scalars[so + 23] = short_hsl.restarts;
    scalars[so + 24] = hsl_tier_samples_total;
    scalars[so + 25] = hsl_tier_samples_yellow;
    scalars[so + 26] = hsl_tier_samples_orange;
    scalars[so + 27] = hsl_tier_samples_red;
    scalars[so + 28] = long_hsl.halt_duration_sum_steps
        + short_hsl.halt_duration_sum_steps
        + long_terminal_duration + short_terminal_duration;
    scalars[so + 29] = fmax(
        fmax(long_hsl.halt_duration_max_steps, short_hsl.halt_duration_max_steps),
        fmax(long_terminal_duration, short_terminal_duration)
    );
    scalars[so + 30] = long_hsl.halt_duration_count
        + short_hsl.halt_duration_count + terminal_count;
    scalars[so + 31] = long_hsl.trigger_drawdown_sum
        + short_hsl.trigger_drawdown_sum;
    scalars[so + 32] = long_hsl.trigger_drawdown_count
        + short_hsl.trigger_drawdown_count;
    scalars[so + 33] = long_hsl.flatten_time_sum_steps
        + short_hsl.flatten_time_sum_steps;
    scalars[so + 34] = long_hsl.flatten_time_count
        + short_hsl.flatten_time_count;
    scalars[so + 35] = long_hsl.restart_retrigger_count
        + short_hsl.restart_retrigger_count;
    float panic_drawdown_count = long_hsl.panic_loss_drawdown_count
        + short_hsl.panic_loss_drawdown_count;
    float panic_drawdown_min = long_hsl.panic_loss_drawdown_count > 0.0f
        ? (short_hsl.panic_loss_drawdown_count > 0.0f
            ? fmin(
                long_hsl.panic_loss_drawdown_min,
                short_hsl.panic_loss_drawdown_min
            ) : long_hsl.panic_loss_drawdown_min)
        : (short_hsl.panic_loss_drawdown_count > 0.0f
            ? short_hsl.panic_loss_drawdown_min : 0.0f);
    scalars[so + 36] = long_hsl.halt_to_restart_equity_loss
        + short_hsl.halt_to_restart_equity_loss;
    scalars[so + 37] = long_hsl.panic_close_loss_sum
        + short_hsl.panic_close_loss_sum;
    scalars[so + 38] = fmax(
        long_hsl.panic_close_loss_max, short_hsl.panic_close_loss_max
    );
    scalars[so + 39] = panic_drawdown_min;
    scalars[so + 40] = long_hsl.panic_loss_drawdown_sum
        + short_hsl.panic_loss_drawdown_sum;
    scalars[so + 41] = fmax(
        long_hsl.panic_loss_drawdown_max,
        short_hsl.panic_loss_drawdown_max
    );
    scalars[so + 42] = panic_drawdown_count;
    scalars[so + 43] = profit_sum;
    scalars[so + 44] = loss_sum;
    scalars[so + 45] = position_unchanged_max_min * interval_ms;
    scalars[so + 46] = long_enabled
        ? long_side.allowed_wel * long_side.base_qty_pct : 0.0f;
    scalars[so + 47] = short_enabled
        ? short_side.allowed_wel * short_side.base_qty_pct : 0.0f;
    scalars[so + 48] = total_wallet_exposure_max;
    scalars[so + 49] = total_wallet_exposure_mean;
}

kernel void passivbot_ema_anchor(
    constant float* bars,
    constant int* flags,
    constant float* params,
    constant float* settings,
    constant int* sizes,
    device float* daily,
    device float* scalars,
    device int* gap_hist,
    uint b [[thread_position_in_grid]]
) {
    passivbot_single_coin_impl(
        bars, flags, params, settings, sizes, daily, scalars, gap_hist, b
    );
}
