#include <metal_stdlib>
using namespace metal;

constant int MAX_COINS = 64;
constant int PARAM_COLS = 42;
constant int OVERRIDE_COLS = 28;
constant int COIN_COLS = 11;
constant int DAILY_COLS = 6;
constant int SCALAR_COLS = 18;
constant int GAP_BINS = 128;

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

inline bool finite_positive(float value) {
    return isfinite(value) && value > 0.0f;
}

inline float coin_override_or(
    constant float* coin_overrides, int coin, int column, float fallback
) {
    float value = coin_overrides[coin * OVERRIDE_COLS + column];
    return isfinite(value) ? value : fallback;
}

inline float allowed_wallet_exposure_limit(
    float base_limit, float total_limit, float allowance_pct, bool legacy_raw
) {
    if (!(isfinite(base_limit) && base_limit > 0.0f)) return 0.0f;
    float raw = fmax(allowance_pct, 0.0f);
    float effective = raw;
    if (!legacy_raw) {
        float max_effective = (
            isfinite(total_limit) && total_limit > 0.0f
        ) ? fmax(total_limit / base_limit - 1.0f, 0.0f) : 0.0f;
        effective = fmin(raw, max_effective);
    }
    return base_limit * (1.0f + effective);
}

inline float calc_close_qty(
    float psize, float pprice, float balance, float twel,
    float minimum_close, int minimum_close_relation, float close_pct,
    float qty_step, float c_mult
) {
    float full_size = balance * twel / fmax(pprice * c_mult, 1.0e-12f);
    float quantity = fmin(
        round_step(psize, qty_step),
        fmax(
            minimum_close,
            ceil_step(
                full_size * close_pct + fmax(psize - full_size, 0.0f),
                qty_step
            )
        )
    );
    float remainder = psize - quantity;
    bool remainder_below_minimum = remainder < minimum_close
        || (remainder == minimum_close && minimum_close_relation > 0);
    if (quantity > 0.0f && quantity < psize && remainder_below_minimum) {
        quantity = psize;
    }
    if (psize < minimum_close * (1.0f - 1.0e-6f) && quantity > 0.0f) {
        quantity = psize;
    } else if (quantity > 0.0f
        && quantity * (1.0f + 1.0e-6f) < minimum_close) {
        quantity = 0.0f;
    }
    return quantity;
}

inline float exposure_reducer_qty(
    float psize, float pprice, float balance, float target_exposure,
    float reducer_price, float qty_step, float min_qty, float min_cost,
    float c_mult
) {
    if (!(balance > 0.0f && psize > 0.0f && pprice > 0.0f
        && target_exposure > 0.0f && reducer_price > 0.0f)) {
        return 0.0f;
    }
    float current_exposure = psize * pprice * c_mult / balance;
    if (!(current_exposure > target_exposure)) return 0.0f;
    float reducer_min = min_entry_qty(
        reducer_price, qty_step, min_qty, min_cost, c_mult
    );
    float target_psize = target_exposure * balance
        / fmax(pprice * c_mult, 1.0e-12f);
    float reduce_qty = fmax(psize - target_psize, 0.0f);
    if (reduce_qty <= 2.220446e-16f) reduce_qty = qty_step;
    float reducer_qty = 0.0f;
    for (int steps = 0; steps <= 10000; ++steps) {
        reducer_qty = fmin(
            psize, fmax(reducer_min, ceil_step(reduce_qty, qty_step))
        );
        float new_psize = fmax(
            round_step(psize - reducer_qty, qty_step), 0.0f
        );
        if (new_psize < target_psize || new_psize <= 2.220446e-16f) {
            break;
        }
        reduce_qty += qty_step;
    }
    return reducer_qty;
}

inline float finalized_reducer_qty(
    float psize, float reducer_qty, float reducer_price,
    float qty_step, float min_qty, float min_cost, float c_mult
) {
    if (reducer_qty <= 0.0f || reducer_price <= 0.0f) return 0.0f;
    float remainder = fmax(
        round_step(psize - reducer_qty, qty_step), 0.0f
    );
    float reducer_min = min_entry_qty(
        reducer_price, qty_step, min_qty, min_cost, c_mult
    );
    // Match finalized_reducer_candidates for the reducer-only strategy close
    // set: absorb an uncloseable residual before protective reducers compete.
    return remainder > 0.0f && remainder < reducer_min
        ? psize : reducer_qty;
}

inline float clamped_market_price(
    constant float* bars, constant float* coin_settings,
    int k, int coin, int coin_count
) {
    int coin_offset = coin * COIN_COLS;
    int first_valid = int(coin_settings[coin_offset + 6]);
    int last_valid = int(coin_settings[coin_offset + 7]);
    int market_k = clamp(k, first_valid, last_valid);
    return bars[(market_k * coin_count + coin) * 4 + 2];
}

struct CloseGroup {
    int ticks;
    float price;
    float qty;
};

// Rebuild Rust's post-reducer recursive grid and return its number of
// duplicate-merged price groups. Selecting one generated-order group at a
// time keeps GPU memory bounded; the caller reorders positive-WE grids before
// applying reachable fills, matching calc_closes_long/short.
inline int recursive_grid_close_groups_after_reducer(
    bool short_side,
    float psize,
    float pprice,
    float generation_balance,
    float allowed_wel,
    int touch_down,
    int touch_up,
    int touch_nearest,
    float touch_min_qty,
    int touch_min_qty_relation,
    float close_qty_pct,
    float close_threshold_base,
    float close_threshold_we,
    float close_threshold_v1h,
    float close_threshold_v1m,
    float volatility_1h,
    float volatility_1m,
    float qty_step,
    float price_step,
    float min_qty,
    float min_cost,
    float c_mult,
    int max_rungs,
    int wanted_group,
    thread CloseGroup& selected
) {
    selected.ticks = 0;
    selected.price = 0.0f;
    selected.qty = 0.0f;
    float sim_psize = psize;
    bool have_group = false;
    int group_count = 0;
    int group_ticks = 0;
    float group_price = 0.0f;
    float group_qty = 0.0f;
    for (int rung = 0; rung < max_rungs && sim_psize > 0.0f; ++rung) {
        float we = sim_psize * pprice * c_mult
            / fmax(generation_balance, 1.0e-9f);
        float wer = we / fmax(allowed_wel, 1.0e-12f);
        float threshold = close_threshold_base
            + wer * close_threshold_we
            + volatility_1h * close_threshold_v1h
            + volatility_1m * close_threshold_v1m;
        float target = pprice * (
            short_side ? 1.0f - threshold : 1.0f + threshold
        );
        int target_tick = short_side
            ? int(floor(target / price_step + 1.0e-6f))
            : int(ceil(target / price_step - 1.0e-6f));
        int close_touch = short_side ? touch_down : touch_up;
        bool touch_controls = short_side
            ? close_touch < target_tick
            : close_touch > target_tick;
        int order_tick = touch_controls ? touch_nearest : target_tick;
        float order_price = float(order_tick) * price_step;
        float minimum_close = touch_controls
            ? touch_min_qty
            : min_entry_qty(
                order_price, qty_step, min_qty, min_cost, c_mult
            );
        int minimum_relation = touch_controls
            ? touch_min_qty_relation : 0;
        float close_pct = close_threshold_we == 0.0f
            ? 1.0f : close_qty_pct;
        float order_qty = calc_close_qty(
            sim_psize, pprice, generation_balance, allowed_wel,
            minimum_close, minimum_relation, close_pct,
            qty_step, c_mult
        );

        order_qty = round_step(order_qty, qty_step);
        if (order_qty <= 0.0f || order_tick <= 0) break;
        order_qty = fmin(order_qty, sim_psize);

        if (!have_group) {
            have_group = true;
            group_ticks = order_tick;
            group_price = order_price;
            group_qty = order_qty;
        } else if (order_tick == group_ticks) {
            group_qty = round_step(group_qty + order_qty, qty_step);
        } else {
            if (group_count == wanted_group) {
                selected.ticks = group_ticks;
                selected.price = group_price;
                selected.qty = group_qty;
            }
            ++group_count;
            group_ticks = order_tick;
            group_price = order_price;
            group_qty = order_qty;
        }
        sim_psize = fmax(round_step(sim_psize - order_qty, qty_step), 0.0f);
    }
    if (have_group) {
        if (group_count == wanted_group) {
            selected.ticks = group_ticks;
            selected.price = group_price;
            selected.qty = group_qty;
        }
        ++group_count;
    }
    return group_count;
}

inline void passivbot_trailing_martingale_multicoin_impl(
    constant float* bars,
    constant int* fill_ticks,
    constant int* touch_ticks,
    constant int* touch_nearest_ticks,
    constant int* touch_min_qty_bits,
    constant int* touch_min_qty_relation,
    constant float* coin_settings,
    constant float* coin_overrides,
    constant float* params,
    constant float* run_settings,
    constant int* sizes,
    device float* daily,
    device float* scalars,
    device int* gap_hist,
    uint b,
    bool short_side
) {
    const int B = sizes[0];
    const int T = sizes[1];
    const int C = sizes[2];
    const int D = sizes[3];
    const int requested_start_k = sizes[4];
    const int global_warmup = sizes[5];
    const int start_day_minute = sizes[6];
    const int start_hour_minute = sizes[7];
    if (b >= uint(B)) return;

    const int po = int(b) * PARAM_COLS;
    const float span_a = params[po + 0];
    const float span_b = params[po + 1];
    const float span_1h = params[po + 2];
    const float span_1m = params[po + 3];
    const float ddf = params[po + 4];
    const float initial_ema_dist = params[po + 5];
    const float initial_qty_pct = params[po + 6];
    const float entry_threshold_base = params[po + 7];
    const float entry_threshold_we = params[po + 8];
    const float entry_threshold_v1h = params[po + 9];
    const float entry_threshold_v1m = params[po + 10];
    const float entry_retracement_base = params[po + 11];
    const float entry_retracement_we = params[po + 12];
    const float entry_retracement_v1h = params[po + 13];
    const float entry_retracement_v1m = params[po + 14];
    const float close_qty_pct = params[po + 15];
    const float close_threshold_base = params[po + 16];
    const float close_threshold_we = params[po + 17];
    const float close_threshold_v1h = params[po + 18];
    const float close_threshold_v1m = params[po + 19];
    const float close_retracement_base = params[po + 20];
    const float close_retracement_v1h = params[po + 21];
    const float close_retracement_v1m = params[po + 22];
    const float cooldown_min = ceil(params[po + 23]);
    const float twel = params[po + 24];
    const bool gate_initial = params[po + 25] > 0.5f;
    const bool gate_reentry = params[po + 26] > 0.5f;
    const float forager_volume_span = params[po + 27];
    const float forager_volatility_span = params[po + 28];
    const float volume_drop = clamp(params[po + 29], 0.0f, 1.0f);
    float w_volume = params[po + 30];
    float w_ready = params[po + 31];
    float w_volatility = params[po + 32];
    const int n_positions = max(1, int(rint(params[po + 33])));
    const float allowance_pct = params[po + 34];
    const bool legacy_raw_allowance = params[po + 35] > 0.5f;
    const bool twel_entry_gate_enabled = params[po + 36] > 0.5f;
    const float twel_threshold = params[po + 37];
    const bool wel_enforcer_enabled = params[po + 38] > 0.5f;
    const float wel_enforcer_threshold = params[po + 39];
    const bool twel_enforcer_enabled = params[po + 40] > 0.5f;
    const bool twel_enforcer_reduce_portfolio = params[po + 41] > 0.5f;
    const float weight_sum = w_volume + w_ready + w_volatility;
    if (weight_sum > 0.0f) {
        w_volume /= weight_sum;
        w_ready /= weight_sum;
        w_volatility /= weight_sum;
    } else {
        w_volume = 0.0f;
        w_ready = 1.0f;
        w_volatility = 0.0f;
    }
    const float alpha_forager_volume = forager_volume_span > 0.0f
        ? clamp(2.0f / (forager_volume_span + 1.0f), 0.0f, 1.0f) : 0.0f;
    const float alpha_forager_volatility = forager_volatility_span > 0.0f
        ? clamp(2.0f / (forager_volatility_span + 1.0f), 0.0f, 1.0f) : 0.0f;

    const float starting_balance = run_settings[0];
    const float liquidation_floor = run_settings[1];
    const float interval_ms = run_settings[2];
    const float score_hysteresis = fmax(run_settings[4], 0.0f);
    const float log_bin_scale = 127.0f / log(4000001.0f);

    float ema0[MAX_COINS];
    float ema1[MAX_COINS];
    float ema2[MAX_COINS];
    float volatility_1m[MAX_COINS];
    float volatility_1h[MAX_COINS];
    float forager_volume[MAX_COINS];
    float forager_volatility[MAX_COINS];
    float hour_high[MAX_COINS];
    float hour_low[MAX_COINS];
    float psize[MAX_COINS];
    float pprice[MAX_COINS];
    float last_increase_k[MAX_COINS];
    float entry_qty[MAX_COINS];
    float close_qty[MAX_COINS];
    float secondary_close_qty[MAX_COINS];
    float twel_close_qty[MAX_COINS];
    float close_gen_balance[MAX_COINS];
    float close_gen_allowed_wel[MAX_COINS];
    float close_grid_gen_psize[MAX_COINS];
    float position_open_k[MAX_COINS];
    float score[MAX_COINS];
    float contribution[MAX_COINS];
    float minimum_entry[MAX_COINS];
    float min_since_open[MAX_COINS];
    float max_since_min[MAX_COINS];
    float max_since_open[MAX_COINS];
    float min_since_max[MAX_COINS];
    int entry_tick[MAX_COINS];
    int close_tick[MAX_COINS];
    int secondary_close_tick[MAX_COINS];
    int twel_close_tick[MAX_COINS];
    int close_grid_max_rungs[MAX_COINS];
    bool selected[MAX_COINS];
    bool incumbent[MAX_COINS];
    bool survivor[MAX_COINS];
    bool entry_candidate[MAX_COINS];
    bool close_reconstruct_after_reducer[MAX_COINS];
    bool filled_coin[MAX_COINS];
    float alpha0_coin[MAX_COINS];
    float alpha1_coin[MAX_COINS];
    float alpha2_coin[MAX_COINS];
    float alpha_1h_coin[MAX_COINS];
    float alpha_1m_coin[MAX_COINS];

    for (int c = 0; c < MAX_COINS; ++c) {
        float seed_close = c < C ? coin_settings[c * COIN_COLS + 9] : 0.0f;
        float seed_volume = c < C ? coin_settings[c * COIN_COLS + 10] : 0.0f;
        ema0[c] = seed_close;
        ema1[c] = seed_close;
        ema2[c] = seed_close;
        volatility_1m[c] = 0.0f;
        volatility_1h[c] = 0.0f;
        forager_volume[c] = seed_volume;
        forager_volatility[c] = 0.0f;
        hour_high[c] = -INFINITY;
        hour_low[c] = INFINITY;
        if (c < C && int(coin_settings[c * COIN_COLS + 6]) == 0) {
            hour_high[c] = bars[c * 4 + 0];
            hour_low[c] = bars[c * 4 + 1];
        }
        psize[c] = 0.0f;
        pprice[c] = 0.0f;
        last_increase_k[c] = -1.0e20f;
        entry_qty[c] = 0.0f;
        close_qty[c] = 0.0f;
        secondary_close_qty[c] = 0.0f;
        twel_close_qty[c] = 0.0f;
        close_gen_balance[c] = 0.0f;
        close_gen_allowed_wel[c] = 0.0f;
        close_grid_gen_psize[c] = 0.0f;
        position_open_k[c] = -1.0f;
        score[c] = -INFINITY;
        contribution[c] = 0.0f;
        minimum_entry[c] = 0.0f;
        min_since_open[c] = INFINITY;
        max_since_min[c] = 0.0f;
        max_since_open[c] = 0.0f;
        min_since_max[c] = INFINITY;
        entry_tick[c] = 0;
        close_tick[c] = 0;
        secondary_close_tick[c] = 0;
        twel_close_tick[c] = 0;
        close_grid_max_rungs[c] = 500;
        selected[c] = false;
        incumbent[c] = false;
        survivor[c] = false;
        entry_candidate[c] = false;
        close_reconstruct_after_reducer[c] = false;
        filled_coin[c] = false;
        float coin_span_a = c < C
            ? coin_override_or(coin_overrides, c, 0, span_a) : span_a;
        float coin_span_b = c < C
            ? coin_override_or(coin_overrides, c, 1, span_b) : span_b;
        float coin_span_c = sqrt(fmax(coin_span_a * coin_span_b, 1.0f));
        float coin_span_lo = fmin(
            coin_span_a, fmin(coin_span_b, coin_span_c)
        );
        float coin_span_hi = fmax(
            coin_span_a, fmax(coin_span_b, coin_span_c)
        );
        float coin_span_mid = coin_span_a + coin_span_b + coin_span_c
            - coin_span_lo - coin_span_hi;
        alpha0_coin[c] = clamp(
            2.0f / (coin_span_lo + 1.0f), 0.0f, 1.0f
        );
        alpha1_coin[c] = clamp(
            2.0f / (coin_span_mid + 1.0f), 0.0f, 1.0f
        );
        alpha2_coin[c] = clamp(
            2.0f / (coin_span_hi + 1.0f), 0.0f, 1.0f
        );
        float coin_span_1h = c < C
            ? coin_override_or(coin_overrides, c, 2, span_1h) : span_1h;
        float coin_span_1m = c < C
            ? coin_override_or(coin_overrides, c, 3, span_1m) : span_1m;
        alpha_1h_coin[c] = coin_span_1h > 0.0f
            ? 2.0f / (fmax(coin_span_1h, 1.0f) + 1.0f) : 0.0f;
        alpha_1m_coin[c] = coin_span_1m > 0.0f
            ? clamp(2.0f / (coin_span_1m + 1.0f), 0.0f, 1.0f) : 0.0f;
    }
    for (int j = 0; j < GAP_BINS; ++j) {
        gap_hist[int(b) * GAP_BINS + j] = 0;
    }

    float balance = starting_balance;
    bool alive = true;
    bool equity_started = false;
    bool selection_initialized = false;
    int max_tradable_seen = 0;
    int previous_effective_n_positions = 0;
    float run_peak = -INFINITY;
    float max_dd = 0.0f;
    float held_max_min = 0.0f;
    float first_fill_k = -1.0f;
    float last_fill_k = -1.0f;
    float gap_max_min = 0.0f;
    float last_high_k = -1.0f;
    float recovery_max_min = 0.0f;
    float first_eq_k = -1.0f;
    float last_eq_k = -1.0f;
    int liquidation_day = -1;

    int current_day = 0;
    bool day_touched = false;
    float day_end = 0.0f;
    float day_min = INFINITY;
    float day_dd = 0.0f;
    float day_volume = 0.0f;
    float day_has_fill = 0.0f;
    float day_min_balance = INFINITY;

    for (int k = 1; k < T - 1; ++k) {
        const int day_index = (start_day_minute + k) / 1440;
        if (day_index != current_day) {
            if (day_touched && current_day >= 0 && current_day < D) {
                int output = (int(b) * D + current_day) * DAILY_COLS;
                daily[output + 0] = day_end;
                daily[output + 1] = day_min;
                daily[output + 2] = day_dd;
                daily[output + 3] = day_volume;
                daily[output + 4] = day_has_fill;
                daily[output + 5] = day_min_balance;
            }
            current_day = day_index;
            day_touched = false;
            day_end = 0.0f;
            day_min = INFINITY;
            day_dd = 0.0f;
            day_volume = 0.0f;
            day_has_fill = 0.0f;
            day_min_balance = INFINITY;
        }

        bool any_fill = false;
        for (int c = 0; c < C; ++c) filled_coin[c] = false;
        for (int c = 0; c < C; ++c) {
            const int coin_offset = c * COIN_COLS;
            const int bar_offset = (k * C + c) * 4;
            const int tick_offset = (k * C + c) * 2;
            const float high = bars[bar_offset + 0];
            const float low = bars[bar_offset + 1];
            const float close = bars[bar_offset + 2];
            const int first_valid = int(coin_settings[coin_offset + 6]);
            const int last_valid = int(coin_settings[coin_offset + 7]);
            const bool valid = k >= first_valid && k <= last_valid
                && finite_positive(high) && finite_positive(low) && finite_positive(close);
            if (!valid || !alive) continue;
            const float qty_step = coin_settings[coin_offset + 0];
            const float price_step = coin_settings[coin_offset + 1];
            const float min_qty = coin_settings[coin_offset + 2];
            const float min_cost = coin_settings[coin_offset + 3];
            const float c_mult = coin_settings[coin_offset + 4];
            const float maker_fee = coin_settings[coin_offset + 5];

            bool close_ready = close_qty[c] > 0.0f && psize[c] > 0.0f;
            bool filled_close = close_ready
                && (short_side
                    ? close_tick[c] > fill_ticks[tick_offset + 1]
                    : close_tick[c] <= fill_ticks[tick_offset + 0]);
            bool filled_secondary_close = secondary_close_qty[c] > 0.0f
                && psize[c] > 0.0f
                && (short_side
                    ? secondary_close_tick[c] > fill_ticks[tick_offset + 1]
                    : secondary_close_tick[c] <= fill_ticks[tick_offset + 0]);
            bool rebuild_grid = close_ready
                && close_reconstruct_after_reducer[c];
            if (filled_close || filled_secondary_close || rebuild_grid) {
                float fill_price = float(close_tick[c]) * price_step;
                float reducer_qty = fmin(
                    round_step(close_qty[c], qty_step), psize[c]
                );
                float grid_gen_psize = close_grid_gen_psize[c];
                int gen_tick_offset = 0;
                float coin_close_qty_pct = 0.0f;
                float coin_close_threshold_base = 0.0f;
                float coin_close_threshold_we = 0.0f;
                float coin_close_threshold_v1h = 0.0f;
                float coin_close_threshold_v1m = 0.0f;
                CloseGroup group;
                int group_count = 0;
                bool reverse = false;
                bool reducer_executed = false;
                bool executed_close = false;

                if (rebuild_grid && grid_gen_psize > 0.0f) {
                    gen_tick_offset = ((k - 1) * C + c) * 2;
                    coin_close_qty_pct = coin_override_or(
                        coin_overrides, c, 15, close_qty_pct
                    );
                    coin_close_threshold_base = coin_override_or(
                        coin_overrides, c, 16, close_threshold_base
                    );
                    coin_close_threshold_we = coin_override_or(
                        coin_overrides, c, 17, close_threshold_we
                    );
                    coin_close_threshold_v1h = coin_override_or(
                        coin_overrides, c, 18, close_threshold_v1h
                    );
                    coin_close_threshold_v1m = coin_override_or(
                        coin_overrides, c, 19, close_threshold_v1m
                    );
                    group_count = recursive_grid_close_groups_after_reducer(
                        short_side,
                        grid_gen_psize,
                        pprice[c],
                        close_gen_balance[c],
                        close_gen_allowed_wel[c],
                        touch_ticks[gen_tick_offset + 0],
                        touch_ticks[gen_tick_offset + 1],
                        touch_nearest_ticks[(k - 1) * C + c],
                        as_type<float>(touch_min_qty_bits[(k - 1) * C + c]),
                        touch_min_qty_relation[(k - 1) * C + c],
                        coin_close_qty_pct,
                        coin_close_threshold_base,
                        coin_close_threshold_we,
                        coin_close_threshold_v1h,
                        coin_close_threshold_v1m,
                        volatility_1h[c],
                        volatility_1m[c],
                        qty_step,
                        price_step,
                        min_qty,
                        min_cost,
                        c_mult,
                        close_grid_max_rungs[c],
                        -1,
                        group
                    );
                    reverse = coin_close_threshold_we > 0.0f;
                }

                float ordinary_budget = fmax(
                    round_step(psize[c] - reducer_qty, qty_step), 0.0f
                );
                float remaining_budget = ordinary_budget;
                float kept_ordinary = 0.0f;
                float minimum_any = min_entry_qty(
                    fill_price, qty_step, min_qty, min_cost, c_mult
                );
                int last_kept_rank = -1;
                for (int trim_rank = 0; trim_rank < group_count; ++trim_rank) {
                    int wanted = reverse
                        ? group_count - trim_rank - 1 : trim_rank;
                    recursive_grid_close_groups_after_reducer(
                        short_side,
                        grid_gen_psize,
                        pprice[c],
                        close_gen_balance[c],
                        close_gen_allowed_wel[c],
                        touch_ticks[gen_tick_offset + 0],
                        touch_ticks[gen_tick_offset + 1],
                        touch_nearest_ticks[(k - 1) * C + c],
                        as_type<float>(touch_min_qty_bits[(k - 1) * C + c]),
                        touch_min_qty_relation[(k - 1) * C + c],
                        coin_close_qty_pct,
                        coin_close_threshold_base,
                        coin_close_threshold_we,
                        coin_close_threshold_v1h,
                        coin_close_threshold_v1m,
                        volatility_1h[c],
                        volatility_1m[c],
                        qty_step,
                        price_step,
                        min_qty,
                        min_cost,
                        c_mult,
                        close_grid_max_rungs[c],
                        wanted,
                        group
                    );
                    float trimmed_qty = fmin(group.qty, remaining_budget);
                    float group_min = min_entry_qty(
                        group.price, qty_step, min_qty, min_cost, c_mult
                    );
                    bool partial_trim = trimmed_qty + 1.0e-6f < group.qty;
                    if (trimmed_qty + 1.0e-6f < group_min) {
                        trimmed_qty = 0.0f;
                        if (partial_trim) remaining_budget = 0.0f;
                    }
                    if (trimmed_qty > 0.0f) {
                        kept_ordinary += trimmed_qty;
                        remaining_budget = fmax(
                            round_step(
                                remaining_budget - trimmed_qty, qty_step
                            ),
                            0.0f
                        );
                        minimum_any = fmin(minimum_any, group_min);
                        last_kept_rank = trim_rank;
                    }
                }
                float dust_remainder = fmax(
                    round_step(
                        psize[c] - reducer_qty - kept_ordinary, qty_step
                    ),
                    0.0f
                );
                if (dust_remainder > 0.0f && dust_remainder < minimum_any
                    && last_kept_rank < 0) {
                    reducer_qty = fmin(
                        psize[c],
                        round_step(reducer_qty + dust_remainder, qty_step)
                    );
                    dust_remainder = 0.0f;
                }

                remaining_budget = ordinary_budget;
                for (int rank = 0; rank < group_count; ++rank) {
                    int wanted = reverse ? group_count - rank - 1 : rank;
                    recursive_grid_close_groups_after_reducer(
                        short_side,
                        grid_gen_psize,
                        pprice[c],
                        close_gen_balance[c],
                        close_gen_allowed_wel[c],
                        touch_ticks[gen_tick_offset + 0],
                        touch_ticks[gen_tick_offset + 1],
                        touch_nearest_ticks[(k - 1) * C + c],
                        as_type<float>(touch_min_qty_bits[(k - 1) * C + c]),
                        touch_min_qty_relation[(k - 1) * C + c],
                        coin_close_qty_pct,
                        coin_close_threshold_base,
                        coin_close_threshold_we,
                        coin_close_threshold_v1h,
                        coin_close_threshold_v1m,
                        volatility_1h[c],
                        volatility_1m[c],
                        qty_step,
                        price_step,
                        min_qty,
                        min_cost,
                        c_mult,
                        close_grid_max_rungs[c],
                        wanted,
                        group
                    );
                    if (group.qty <= 0.0f) break;
                    float group_min = min_entry_qty(
                        group.price, qty_step, min_qty, min_cost, c_mult
                    );
                    float trimmed_group_qty = fmin(
                        group.qty, remaining_budget
                    );
                    bool partial_trim = trimmed_group_qty + 1.0e-6f
                        < group.qty;
                    if (trimmed_group_qty + 1.0e-6f < group_min) {
                        trimmed_group_qty = 0.0f;
                        if (partial_trim) remaining_budget = 0.0f;
                    }
                    if (trimmed_group_qty > 0.0f) {
                        remaining_budget = fmax(
                            round_step(
                                remaining_budget - trimmed_group_qty,
                                qty_step
                            ),
                            0.0f
                        );
                        if (rank == last_kept_rank && dust_remainder > 0.0f
                            && dust_remainder < minimum_any) {
                            trimmed_group_qty = round_step(
                                trimmed_group_qty + dust_remainder, qty_step
                            );
                        }
                    }
                    bool reducer_before_group = filled_close
                        && !reducer_executed
                        && (short_side
                            ? close_tick[c] > group.ticks
                            : close_tick[c] < group.ticks);
                    if (reducer_before_group) {
                        float qty = fmin(reducer_qty, psize[c]);
                        float pnl = qty * c_mult * (short_side
                            ? pprice[c] - fill_price
                            : fill_price - pprice[c]);
                        balance += pnl
                            - qty * fill_price * c_mult * maker_fee;
                        psize[c] = fmax(
                            round_step(psize[c] - qty, qty_step), 0.0f
                        );
                        day_volume += qty * fill_price * c_mult / balance;
                        reducer_executed = true;
                        executed_close = true;
                    }
                    bool reachable = short_side
                        ? group.ticks > fill_ticks[tick_offset + 1]
                        : group.ticks <= fill_ticks[tick_offset + 0];
                    if (!reachable) break;
                    float group_qty = trimmed_group_qty;
                    if (group_qty <= 0.0f) continue;
                    float grid_qty = fmin(
                        round_step(group_qty, qty_step), psize[c]
                    );
                    float grid_pnl = grid_qty * c_mult * (short_side
                        ? pprice[c] - group.price
                        : group.price - pprice[c]);
                    balance += grid_pnl
                        - grid_qty * group.price * c_mult * maker_fee;
                    psize[c] = fmax(
                        round_step(psize[c] - grid_qty, qty_step), 0.0f
                    );
                    day_volume += grid_qty * group.price * c_mult / balance;
                    executed_close = true;
                    if (psize[c] <= 0.0f) break;
                }

                float secondary_price = float(secondary_close_tick[c])
                    * price_step;
                bool secondary_first = filled_secondary_close
                    && (!(filled_close && !reducer_executed)
                        || (short_side
                            ? secondary_price >= fill_price
                            : secondary_price <= fill_price));
                for (int close_rank = 0; close_rank < 2; ++close_rank) {
                    bool use_secondary = secondary_first
                        ? close_rank == 0 : close_rank == 1;
                    bool reachable = use_secondary
                        ? filled_secondary_close
                        : filled_close && !reducer_executed;
                    if (!reachable || psize[c] <= 0.0f) continue;
                    float price = use_secondary
                        ? secondary_price : fill_price;
                    float requested_qty = use_secondary
                        ? secondary_close_qty[c] : reducer_qty;
                    float qty = fmin(
                        round_step(requested_qty, qty_step), psize[c]
                    );
                    float pnl = qty * c_mult * (short_side
                        ? pprice[c] - price : price - pprice[c]);
                    balance += pnl - qty * price * c_mult * maker_fee;
                    psize[c] = fmax(
                        round_step(psize[c] - qty, qty_step), 0.0f
                    );
                    day_volume += qty * price * c_mult / balance;
                    if (!use_secondary) reducer_executed = true;
                    executed_close = true;
                }

                if (executed_close) {
                    bool went_flat = psize[c] <= 0.0f;
                    if (went_flat) {
                        pprice[c] = 0.0f;
                        if (position_open_k[c] >= 0.0f) {
                            held_max_min = fmax(
                                held_max_min, float(k) - position_open_k[c]
                            );
                        }
                        position_open_k[c] = -1.0f;
                    }
                    close_qty[c] = 0.0f;
                    secondary_close_qty[c] = 0.0f;
                    close_reconstruct_after_reducer[c] = false;
                    filled_coin[c] = true;
                    any_fill = true;
                }
            }

            bool was_flat = psize[c] <= 0.0f;
            bool filled_entry = entry_qty[c] > 0.0f
                && (short_side
                    ? entry_tick[c] <= fill_ticks[tick_offset + 0]
                    : entry_tick[c] > fill_ticks[tick_offset + 1]);
            if (filled_entry) {
                float fill_price = float(entry_tick[c]) * price_step;
                float adjusted = round_step(entry_qty[c], qty_step);
                balance -= adjusted * fill_price * c_mult * maker_fee;
                float new_size = round_step(psize[c] + adjusted, qty_step);
                float new_price = was_flat ? fill_price
                    : pprice[c] * (psize[c] / fmax(new_size, 1.0e-12f))
                        + fill_price * (adjusted / fmax(new_size, 1.0e-12f));
                if (was_flat) position_open_k[c] = float(k);
                psize[c] = new_size;
                pprice[c] = new_price;
                last_increase_k[c] = float(k);
                day_volume += fabs(adjusted) * fill_price * c_mult / balance;
                entry_qty[c] = 0.0f;
                filled_coin[c] = true;
                any_fill = true;
            }
        }
        if (any_fill) {
            day_has_fill = 1.0f;
            if (last_fill_k >= 0.0f) {
                float gap = float(k) - last_fill_k;
                int bin = clamp(
                    int(log(fmax(gap, 0.0f) + 1.0f) * log_bin_scale), 0, 127
                );
                gap_hist[int(b) * GAP_BINS + bin] += 1;
                gap_max_min = fmax(gap_max_min, gap);
            }
            if (first_fill_k < 0.0f) first_fill_k = float(k);
            last_fill_k = float(k);
        }

        const bool hour_boundary = ((start_hour_minute + k) % 60) == 0;
        for (int c = 0; c < C; ++c) {
            const int coin_offset = c * COIN_COLS;
            const int bar_offset = (k * C + c) * 4;
            const float high = bars[bar_offset + 0];
            const float low = bars[bar_offset + 1];
            const float close = bars[bar_offset + 2];
            const float volume = bars[bar_offset + 3];
            const int first_valid = int(coin_settings[coin_offset + 6]);
            const int last_valid = int(coin_settings[coin_offset + 7]);
            const bool valid = k >= first_valid && k <= last_valid
                && finite_positive(high) && finite_positive(low) && finite_positive(close);
            if (hour_boundary) {
                if (hour_high[c] > 0.0f && isfinite(hour_low[c]) && hour_low[c] > 0.0f
                    && alpha_1h_coin[c] > 0.0f) {
                    float hour_range = log(hour_high[c] / hour_low[c]);
                    volatility_1h[c] = fma(
                        alpha_1h_coin[c], hour_range - volatility_1h[c], volatility_1h[c]
                    );
                }
                hour_high[c] = -INFINITY;
                hour_low[c] = INFINITY;
            }
            if (!valid) continue;
            hour_high[c] = fmax(hour_high[c], high);
            hour_low[c] = fmin(hour_low[c], low);
            float log_range = log(high / low);
            ema0[c] = fma(alpha0_coin[c], close - ema0[c], ema0[c]);
            ema1[c] = fma(alpha1_coin[c], close - ema1[c], ema1[c]);
            ema2[c] = fma(alpha2_coin[c], close - ema2[c], ema2[c]);
            if (alpha_1m_coin[c] > 0.0f) {
                volatility_1m[c] = fma(
                    alpha_1m_coin[c], log_range - volatility_1m[c], volatility_1m[c]
                );
            }
            if (alpha_forager_volatility > 0.0f) {
                forager_volatility[c] = fma(
                    alpha_forager_volatility,
                    log_range - forager_volatility[c],
                    forager_volatility[c]
                );
            }
            if (alpha_forager_volume > 0.0f) {
                float typical = (high + low + close) / 3.0f;
                float quote_volume = fmax(volume, 0.0f) * typical;
                forager_volume[c] = fma(
                    alpha_forager_volume,
                    quote_volume - forager_volume[c],
                    forager_volume[c]
                );
            }
            if (psize[c] > 0.0f) {
                if (filled_coin[c]) {
                    min_since_open[c] = INFINITY;
                    max_since_min[c] = 0.0f;
                    max_since_open[c] = 0.0f;
                    min_since_max[c] = INFINITY;
                } else {
                    if (low < min_since_open[c]) {
                        min_since_open[c] = low;
                        max_since_min[c] = close;
                    } else {
                        max_since_min[c] = fmax(max_since_min[c], high);
                    }
                    if (high > max_since_open[c]) {
                        max_since_open[c] = high;
                        min_since_max[c] = close;
                    } else {
                        min_since_max[c] = fmin(min_since_max[c], low);
                    }
                }
            }
        }

        int tradable_count = 0;
        for (int c = 0; c < C; ++c) {
            int coin_offset = c * COIN_COLS;
            int bar_offset = (k * C + c) * 4;
            float close = bars[bar_offset + 2];
            float coin_wel = coin_override_or(
                coin_overrides, c, 24, -1.0f
            );
            if (k >= int(coin_settings[coin_offset + 8])
                && k <= int(coin_settings[coin_offset + 7])
                && finite_positive(close) && coin_wel != 0.0f) {
                tradable_count += 1;
            }
        }
        max_tradable_seen = max(max_tradable_seen, tradable_count);
        const int effective_n_positions = min(n_positions, max_tradable_seen);
        const bool can_generate = alive && effective_n_positions > 0
            && k > max(global_warmup, 1) && k >= requested_start_k;
        equity_started = equity_started || can_generate;

        if (can_generate) {
            // Exact Rust ranks flat candidates every minute. Re-ranking only
            // after state changes keeps the proxy inexpensive; independent
            // exact validations and drift gates police this approximation.
            bool reselect = !selection_initialized || any_fill
                || effective_n_positions != previous_effective_n_positions;
            if (reselect) {
                int active_count = 0;
                for (int c = 0; c < C; ++c) {
                    incumbent[c] = selected[c] && psize[c] <= 0.0f;
                    selected[c] = psize[c] > 0.0f;
                    if (selected[c]) active_count += 1;
                    survivor[c] = false;
                }
                int slots = max(effective_n_positions - active_count, 0);
                int enabled_count = 0;
                for (int c = 0; c < C; ++c) {
                    int coin_offset = c * COIN_COLS;
                    int bar_offset = (k * C + c) * 4;
                    float coin_wel = coin_override_or(
                        coin_overrides, c, 24, -1.0f
                    );
                    bool enabled = !selected[c]
                        && k >= int(coin_settings[coin_offset + 8])
                        && k <= int(coin_settings[coin_offset + 7])
                        && finite_positive(bars[bar_offset + 2])
                        && coin_wel != 0.0f;
                    survivor[c] = enabled;
                    if (enabled) enabled_count += 1;
                }
                int keep = int(floor(float(enabled_count) * (1.0f - volume_drop) + 0.5f));
                keep = min(enabled_count, max(max(keep, slots), enabled_count > 0 ? 1 : 0));
                if (keep < enabled_count) {
                    for (int c = 0; c < C; ++c) {
                        if (!survivor[c]) continue;
                        int better = 0;
                        for (int j = 0; j < C; ++j) {
                            if (!survivor[j]) continue;
                            if (forager_volume[j] > forager_volume[c]
                                || (forager_volume[j] == forager_volume[c] && j < c)) {
                                better += 1;
                            }
                        }
                        if (better >= keep) survivor[c] = false;
                    }
                }

                float volume_min = INFINITY;
                float volume_max = -INFINITY;
                float ready_min = INFINITY;
                float ready_max = -INFINITY;
                float volatility_min = INFINITY;
                float volatility_max = -INFINITY;
                for (int c = 0; c < C; ++c) {
                    if (!survivor[c]) continue;
                    int bar_offset = (k * C + c) * 4;
                    float close = bars[bar_offset + 2];
                    float lower = fmin(ema0[c], fmin(ema1[c], ema2[c]));
                    float upper = fmax(ema0[c], fmax(ema1[c], ema2[c]));
                    float coin_initial_ema_dist = coin_override_or(
                        coin_overrides, c, 5, initial_ema_dist
                    );
                    float threshold = short_side
                        ? upper * (1.0f + coin_initial_ema_dist)
                        : lower * (1.0f - coin_initial_ema_dist);
                    float readiness = threshold > 0.0f
                        ? (short_side ? 1.0f - close / threshold : close / threshold - 1.0f)
                        : INFINITY;
                    volume_min = fmin(volume_min, forager_volume[c]);
                    volume_max = fmax(volume_max, forager_volume[c]);
                    ready_min = fmin(ready_min, readiness);
                    ready_max = fmax(ready_max, readiness);
                    volatility_min = fmin(volatility_min, forager_volatility[c]);
                    volatility_max = fmax(volatility_max, forager_volatility[c]);
                }
                for (int c = 0; c < C; ++c) {
                    score[c] = -INFINITY;
                    if (!survivor[c]) continue;
                    int bar_offset = (k * C + c) * 4;
                    float close = bars[bar_offset + 2];
                    float lower = fmin(ema0[c], fmin(ema1[c], ema2[c]));
                    float upper = fmax(ema0[c], fmax(ema1[c], ema2[c]));
                    float coin_initial_ema_dist = coin_override_or(
                        coin_overrides, c, 5, initial_ema_dist
                    );
                    float threshold = short_side
                        ? upper * (1.0f + coin_initial_ema_dist)
                        : lower * (1.0f - coin_initial_ema_dist);
                    float readiness = threshold > 0.0f
                        ? (short_side ? 1.0f - close / threshold : close / threshold - 1.0f)
                        : INFINITY;
                    float volume_component = volume_max > volume_min
                        ? (forager_volume[c] - volume_min) / (volume_max - volume_min) : 1.0f;
                    float ready_component = ready_max > ready_min
                        ? (ready_max - readiness) / (ready_max - ready_min) : 1.0f;
                    float volatility_component = volatility_max > volatility_min
                        ? (forager_volatility[c] - volatility_min)
                            / (volatility_max - volatility_min) : 1.0f;
                    score[c] = w_volume * volume_component
                        + w_ready * ready_component
                        + w_volatility * volatility_component;
                }
                for (int pick = 0; pick < slots; ++pick) {
                    int best = -1;
                    for (int c = 0; c < C; ++c) {
                        if (!survivor[c] || selected[c]) continue;
                        if (best < 0 || score[c] > score[best]
                            || (score[c] == score[best] && c < best)) {
                            best = c;
                        }
                    }
                    if (best >= 0) selected[best] = true;
                }
                if (score_hysteresis > 0.0f) {
                    // Match Rust's score hysteresis: consider incumbent flat
                    // candidates from best to worst, then displace only the
                    // weakest selected non-incumbent challenger when its
                    // normalized-score lead is within the configured gap.
                    for (int rank = 0; rank < C; ++rank) {
                        int incumbent_coin = -1;
                        for (int c = 0; c < C; ++c) {
                            if (!survivor[c] || !incumbent[c] || selected[c]) continue;
                            if (incumbent_coin < 0 || score[c] > score[incumbent_coin]
                                || (score[c] == score[incumbent_coin]
                                    && c < incumbent_coin)) {
                                incumbent_coin = c;
                            }
                        }
                        if (incumbent_coin < 0) break;

                        int challenger = -1;
                        for (int c = 0; c < C; ++c) {
                            if (!selected[c] || incumbent[c] || !survivor[c]) continue;
                            if (challenger < 0 || score[c] < score[challenger]
                                || (score[c] == score[challenger] && c > challenger)) {
                                challenger = c;
                            }
                        }
                        if (challenger < 0) break;
                        if (score[challenger] - score[incumbent_coin]
                            <= score_hysteresis) {
                            selected[challenger] = false;
                            selected[incumbent_coin] = true;
                        } else {
                            // Rust continues to lower-scored incumbents, but
                            // none can satisfy the gap once the best cannot.
                            break;
                        }
                    }
                }
                selection_initialized = true;
                previous_effective_n_positions = effective_n_positions;
            }

            const float effective_wel = twel / fmax(float(effective_n_positions), 1.0f);
            float current_twe = 0.0f;
            int open_position_count = 0;
            for (int c = 0; c < C; ++c) {
                int coin_offset = c * COIN_COLS;
                if (psize[c] > 0.0f && balance > 0.0f) {
                    current_twe += psize[c] * pprice[c]
                        * coin_settings[coin_offset + 4] / balance;
                    open_position_count += 1;
                }
                twel_close_qty[c] = 0.0f;
                twel_close_tick[c] = 0;
            }

            // Match calc_twel_enforcer_actions: account for every open
            // position, rank reducible positions by least adverse projected
            // loss per exposure (then symbol index), and allocate only the
            // exposure needed to cross the side-wide target.
            float twel_repair_target = twel * twel_threshold;
            if (twel_enforcer_enabled && twel_threshold > 0.0f
                && twel_repair_target > 0.0f && balance > 0.0f
                && current_twe > twel_repair_target + 1.0e-9f
                && open_position_count > 0) {
                // TWEL reduce_overweight uses the current eligible count.
                // Keep the grow-only maximum solely for dynamic WEL sizing.
                int current_effective_n_positions = min(
                    n_positions, tradable_count
                );
                int repair_n_positions = current_effective_n_positions > 0
                    ? current_effective_n_positions : open_position_count;
                float overweight_target = twel_repair_target
                    / fmax(float(repair_n_positions), 1.0f);
                bool processed[MAX_COINS];
                for (int c = 0; c < MAX_COINS; ++c) processed[c] = false;
                float running_twe = current_twe;
                for (int rank = 0; rank < C; ++rank) {
                    if (running_twe <= twel_repair_target + 1.0e-9f) break;
                    int best = -1;
                    float best_adverse = INFINITY;
                    for (int c = 0; c < C; ++c) {
                        if (processed[c] || psize[c] <= 0.0f
                            || pprice[c] <= 0.0f) continue;
                        int coin_offset = c * COIN_COLS;
                        float c_mult = coin_settings[coin_offset + 4];
                        float exposure = psize[c] * pprice[c] * c_mult
                            / balance;
                        if (!(exposure > 1.0e-9f)) continue;
                        if (!twel_enforcer_reduce_portfolio
                            && !(exposure > overweight_target + 1.0e-9f)) {
                            continue;
                        }
                        float market_price = clamped_market_price(
                            bars, coin_settings, k, c, C
                        );
                        float projected_loss = psize[c] * c_mult * fmax(
                            short_side
                                ? market_price - pprice[c]
                                : pprice[c] - market_price,
                            0.0f
                        );
                        float adverse = projected_loss / exposure;
                        if (best < 0 || adverse < best_adverse
                            || (adverse == best_adverse && c < best)) {
                            best = c;
                            best_adverse = adverse;
                        }
                    }
                    if (best < 0) break;
                    processed[best] = true;
                    int coin_offset = best * COIN_COLS;
                    float qty_step = coin_settings[coin_offset + 0];
                    float price_step = coin_settings[coin_offset + 1];
                    float min_qty = coin_settings[coin_offset + 2];
                    float min_cost = coin_settings[coin_offset + 3];
                    float c_mult = coin_settings[coin_offset + 4];
                    float exposure = psize[best] * pprice[best] * c_mult
                        / balance;
                    float exposure_to_cut = fmin(
                        fmax(running_twe - twel_repair_target, 0.0f),
                        exposure
                    );
                    float market_price = clamped_market_price(
                        bars, coin_settings, k, best, C
                    );
                    int reducer_tick = short_side
                        ? int(ceil(
                            market_price * 1.0005f / price_step - 1.0e-6f
                        ))
                        : int(floor(
                            market_price * 0.9995f / price_step + 1.0e-6f
                        ));
                    reducer_tick = max(reducer_tick, 1);
                    float reducer_price = float(reducer_tick) * price_step;
                    float requested_qty = ceil_step(
                        exposure_to_cut * balance
                            / fmax(pprice[best] * c_mult, 1.0e-12f),
                        qty_step
                    );
                    float reducer_min = min_entry_qty(
                        reducer_price, qty_step, min_qty, min_cost, c_mult
                    );
                    float reducer_qty = fmin(
                        psize[best], fmax(reducer_min, requested_qty)
                    );
                    reducer_qty = fmin(
                        psize[best], ceil_step(reducer_qty, qty_step)
                    );
                    if (reducer_qty <= 1.0e-9f) continue;
                    twel_close_qty[best] = reducer_qty;
                    twel_close_tick[best] = reducer_tick;
                    running_twe -= fmax(
                        exposure - fmax(
                            round_step(psize[best] - reducer_qty, qty_step),
                            0.0f
                        ) * pprice[best] * c_mult / balance,
                        0.0f
                    );
                }
            }
            for (int c = 0; c < C; ++c) {
                entry_qty[c] = 0.0f;
                close_qty[c] = 0.0f;
                secondary_close_qty[c] = 0.0f;
                secondary_close_tick[c] = 0;
                close_reconstruct_after_reducer[c] = false;
                close_grid_gen_psize[c] = 0.0f;
                close_grid_max_rungs[c] = 500;
                contribution[c] = 0.0f;
                entry_candidate[c] = false;
                int coin_offset = c * COIN_COLS;
                int bar_offset = (k * C + c) * 4;
                int tick_offset = (k * C + c) * 2;
                float price_now = bars[bar_offset + 2];
                float fixed_coin_wel = coin_override_or(
                    coin_overrides, c, 24, -1.0f
                );
                float coin_wel = fixed_coin_wel >= 0.0f
                    ? fixed_coin_wel : effective_wel;
                float coin_allowance_pct = coin_override_or(
                    coin_overrides, c, 25, allowance_pct
                );
                bool coin_wel_enforcer_enabled = coin_override_or(
                    coin_overrides, c, 26,
                    wel_enforcer_enabled ? 1.0f : 0.0f
                ) > 0.5f;
                float coin_wel_enforcer_threshold = coin_override_or(
                    coin_overrides, c, 27, wel_enforcer_threshold
                );
                float allowed_coin_wel = allowed_wallet_exposure_limit(
                    coin_wel, twel, coin_allowance_pct, legacy_raw_allowance
                );
                bool tradable = k >= int(coin_settings[coin_offset + 8])
                    && k <= int(coin_settings[coin_offset + 7])
                    && finite_positive(price_now) && allowed_coin_wel > 0.0f;
                if (!tradable) continue;
                float qty_step = coin_settings[coin_offset + 0];
                float price_step = coin_settings[coin_offset + 1];
                float min_qty = coin_settings[coin_offset + 2];
                float min_cost = coin_settings[coin_offset + 3];
                float c_mult = coin_settings[coin_offset + 4];
                float lower = fmin(ema0[c], fmin(ema1[c], ema2[c]));
                float upper = fmax(ema0[c], fmax(ema1[c], ema2[c]));
                float coin_ddf = coin_override_or(
                    coin_overrides, c, 4, ddf
                );
                float coin_initial_ema_dist = coin_override_or(
                    coin_overrides, c, 5, initial_ema_dist
                );
                float coin_initial_qty_pct = coin_override_or(
                    coin_overrides, c, 6, initial_qty_pct
                );
                float coin_entry_threshold_base = coin_override_or(
                    coin_overrides, c, 7, entry_threshold_base
                );
                float coin_entry_threshold_we = coin_override_or(
                    coin_overrides, c, 8, entry_threshold_we
                );
                float coin_entry_threshold_v1h = coin_override_or(
                    coin_overrides, c, 9, entry_threshold_v1h
                );
                float coin_entry_threshold_v1m = coin_override_or(
                    coin_overrides, c, 10, entry_threshold_v1m
                );
                float coin_entry_retracement_base = coin_override_or(
                    coin_overrides, c, 11, entry_retracement_base
                );
                float coin_entry_retracement_we = coin_override_or(
                    coin_overrides, c, 12, entry_retracement_we
                );
                float coin_entry_retracement_v1h = coin_override_or(
                    coin_overrides, c, 13, entry_retracement_v1h
                );
                float coin_entry_retracement_v1m = coin_override_or(
                    coin_overrides, c, 14, entry_retracement_v1m
                );
                float coin_close_qty_pct = coin_override_or(
                    coin_overrides, c, 15, close_qty_pct
                );
                float coin_close_threshold_base = coin_override_or(
                    coin_overrides, c, 16, close_threshold_base
                );
                float coin_close_threshold_we = coin_override_or(
                    coin_overrides, c, 17, close_threshold_we
                );
                float coin_close_threshold_v1h = coin_override_or(
                    coin_overrides, c, 18, close_threshold_v1h
                );
                float coin_close_threshold_v1m = coin_override_or(
                    coin_overrides, c, 19, close_threshold_v1m
                );
                float coin_close_retracement_base = coin_override_or(
                    coin_overrides, c, 20, close_retracement_base
                );
                float coin_close_retracement_v1h = coin_override_or(
                    coin_overrides, c, 21, close_retracement_v1h
                );
                float coin_close_retracement_v1m = coin_override_or(
                    coin_overrides, c, 22, close_retracement_v1m
                );
                float coin_cooldown_min = ceil(coin_override_or(
                    coin_overrides, c, 23, cooldown_min
                ));
                int touch_down = touch_ticks[tick_offset + 0];
                int touch_up = touch_ticks[tick_offset + 1];
                int entry_touch = short_side ? touch_up : touch_down;
                int band_tick = short_side
                    ? int(ceil(upper * (1.0f + coin_initial_ema_dist)
                        / price_step - 1.0e-6f))
                    : int(floor(lower * (1.0f - coin_initial_ema_dist)
                        / price_step + 1.0e-6f));
                bool initial_touch_controls = !gate_initial || (short_side
                    ? touch_down >= band_tick : touch_up <= band_tick);
                int initial_tick = initial_touch_controls ? entry_touch : band_tick;
                float initial_price = float(initial_tick) * price_step;
                float min_iq = min_entry_qty(
                    initial_price, qty_step, min_qty, min_cost, c_mult
                );
                float iq = fmax(min_iq, round_step(
                    balance * allowed_coin_wel * coin_initial_qty_pct
                        / fmax(initial_price * c_mult, 1.0e-12f),
                    qty_step
                ));
                bool flat = psize[c] <= 0.0f;
                bool partial = !flat && psize[c] < iq * 0.8f;
                float iq_partial = fmax(
                    min_iq, floor_step(iq - psize[c], qty_step)
                );
                float iq_effective = !flat && psize[c] < iq
                    ? fmax(round_step(psize[c], qty_step), min_iq) : iq;
                float we = !flat && balance > 0.0f
                    ? psize[c] * pprice[c] * c_mult / balance : 0.0f;
                float wer = we / fmax(allowed_coin_wel, 1.0e-12f);
                float threshold_multiplier = fmax(
                    1.0f + volatility_1h[c] * coin_entry_threshold_v1h
                        + volatility_1m[c] * coin_entry_threshold_v1m
                        + wer * coin_entry_threshold_we,
                    1.0f
                );
                float retracement_multiplier = fmax(
                    1.0f + volatility_1h[c] * coin_entry_retracement_v1h
                        + volatility_1m[c] * coin_entry_retracement_v1m
                        + wer * coin_entry_retracement_we,
                    1.0f
                );
                float entry_threshold = fmax(coin_entry_threshold_base, 0.0f)
                    * threshold_multiplier;
                float entry_retracement = fmax(coin_entry_retracement_base, 0.0f)
                    * retracement_multiplier;
                bool trailing_entry = coin_entry_retracement_base > 0.0f;
                bool retraced_entry = short_side
                    ? min_since_max[c] < max_since_open[c]
                        * (1.0f - entry_retracement)
                    : max_since_min[c] > min_since_open[c]
                        * (1.0f + entry_retracement);
                bool crossed_entry = short_side
                    ? max_since_open[c] > pprice[c] * (1.0f + entry_threshold)
                    : min_since_open[c] < pprice[c] * (1.0f - entry_threshold);
                bool entry_triggered = true;
                float reentry_target = pprice[c] * (short_side
                    ? 1.0f + entry_threshold : 1.0f - entry_threshold);
                if (trailing_entry) {
                    if (entry_threshold <= 0.0f) {
                        entry_triggered = entry_retracement > 0.0f
                            && retraced_entry;
                        reentry_target = price_now;
                    } else if (entry_retracement > 0.0f) {
                        entry_triggered = crossed_entry && retraced_entry;
                        reentry_target = pprice[c] * (short_side
                            ? 1.0f + entry_threshold - entry_retracement
                            : 1.0f - entry_threshold + entry_retracement);
                    }
                }
                bool reentry_is_touch = trailing_entry
                    && entry_threshold <= 0.0f;
                int raw_reentry_tick = reentry_is_touch ? entry_touch
                    : (short_side
                        ? int(ceil(reentry_target / price_step - 1.0e-6f))
                        : int(floor(reentry_target / price_step + 1.0e-6f)));
                bool reentry_touch_controls = reentry_is_touch || (short_side
                    ? touch_down >= raw_reentry_tick
                    : touch_up <= raw_reentry_tick);
                int reentry_tick = reentry_touch_controls
                    ? entry_touch : raw_reentry_tick;
                if (gate_reentry) {
                    bool band_controls = short_side
                        ? band_tick >= reentry_tick : band_tick <= reentry_tick;
                    if (band_controls) reentry_tick = band_tick;
                }
                float reentry_price = float(reentry_tick) * price_step;
                float min_rq = min_entry_qty(
                    reentry_price, qty_step, min_qty, min_cost, c_mult
                );
                float rq = fmax(iq_effective, fmax(min_rq, round_step(
                    fmax(
                        psize[c] * coin_ddf,
                        balance * allowed_coin_wel * coin_initial_qty_pct
                            / fmax(reentry_price * c_mult, 1.0e-12f)
                    ),
                    qty_step
                )));
                float we_if = (psize[c] * pprice[c] + rq * reentry_price)
                    * c_mult / fmax(balance, 1.0e-9f);
                float crop_fraction = (allowed_coin_wel - we)
                    / fmax(we_if - we, 1.0e-12f);
                float rq_crop = fmax(
                    round_step(rq * crop_fraction, qty_step), min_rq
                );
                if (we_if > allowed_coin_wel * 1.01f && rq_crop < rq) rq = rq_crop;
                bool cap_hit = trailing_entry
                    ? we > allowed_coin_wel * 0.999f
                    : we >= allowed_coin_wel * 0.999f;
                bool reentry_ok = !flat && !partial && !cap_hit
                    && reentry_tick > 1
                    && (!trailing_entry || entry_triggered);
                float quantity = flat ? iq
                    : (partial ? iq_partial : (reentry_ok ? rq : 0.0f));
                int candidate_entry_tick = flat || partial
                    ? initial_tick : reentry_tick;
                float entry_price = float(candidate_entry_tick) * price_step;
                bool cooldown = coin_cooldown_min > 0.0f
                    && last_increase_k[c] > -1.0e19f
                    && float(k) < last_increase_k[c] + coin_cooldown_min;
                if (!selected[c] || cooldown || balance <= 0.0f
                    || coin_initial_qty_pct <= 0.0f || candidate_entry_tick <= 1) {
                    quantity = 0.0f;
                }
                float headroom = (
                    allowed_coin_wel * balance - psize[c] * pprice[c] * c_mult
                ) / fmax(entry_price * c_mult, 1.0e-12f);
                if ((psize[c] * pprice[c] + quantity * entry_price) * c_mult
                    / fmax(balance, 1.0e-9f) > allowed_coin_wel * 1.01f) {
                    quantity = fmin(
                        quantity, fmax(floor_step(headroom, qty_step), 0.0f)
                    );
                }
                if (quantity + 1.0e-6f < min_entry_qty(
                    entry_price, qty_step, min_qty, min_cost, c_mult
                )) {
                    quantity = 0.0f;
                }
                entry_qty[c] = quantity;
                entry_tick[c] = candidate_entry_tick;
                minimum_entry[c] = min_entry_qty(
                    entry_price, qty_step, min_qty, min_cost, c_mult
                );
                entry_candidate[c] = quantity > 0.0f;
                contribution[c] = quantity > 0.0f
                    ? quantity * entry_price * c_mult / balance : 0.0f;

                // Exact Rust keeps only the largest protective reducer for a
                // position before allocating its ordinary close ladder.
                float raw_twel_reducer_qty = twel_close_qty[c];
                int reducer_tick = twel_close_tick[c];
                float twel_reducer_price = float(reducer_tick) * price_step;
                float finalized_twel_reducer_qty = finalized_reducer_qty(
                    psize[c], raw_twel_reducer_qty, twel_reducer_price,
                    qty_step, min_qty, min_cost, c_mult
                );
                float reducer_qty = raw_twel_reducer_qty;
                bool use_twel = raw_twel_reducer_qty > 0.0f;
                float wel_reducer_qty = 0.0f;
                float wel_target = allowed_coin_wel
                    * coin_wel_enforcer_threshold;
                if (coin_wel_enforcer_enabled
                    && coin_wel_enforcer_threshold > 0.0f
                    && balance > 0.0f && psize[c] > 0.0f && pprice[c] > 0.0f
                    && wel_target > 0.0f && we > wel_target) {
                    int wel_reducer_tick = short_side ? touch_down : touch_up;
                    float wel_reducer_price = float(wel_reducer_tick) * price_step;
                    wel_reducer_qty = exposure_reducer_qty(
                        psize[c], pprice[c], balance, wel_target,
                        wel_reducer_price, qty_step, min_qty, min_cost, c_mult
                    );
                    float finalized_wel_reducer_qty = finalized_reducer_qty(
                        psize[c], wel_reducer_qty, wel_reducer_price,
                        qty_step, min_qty, min_cost, c_mult
                    );
                    if (finalized_wel_reducer_qty
                            >= finalized_twel_reducer_qty) {
                        reducer_qty = wel_reducer_qty;
                        reducer_tick = wel_reducer_tick;
                        use_twel = false;
                    }
                }

                float close_threshold = coin_close_threshold_base
                    + wer * coin_close_threshold_we
                    + volatility_1h[c] * coin_close_threshold_v1h
                    + volatility_1m[c] * coin_close_threshold_v1m;
                float close_retracement = fmax(
                    coin_close_retracement_base, 0.0f
                )
                    * fmax(
                        1.0f + volatility_1h[c] * coin_close_retracement_v1h
                            + volatility_1m[c] * coin_close_retracement_v1m,
                        1.0f
                    );
                bool trailing_close = coin_close_retracement_base > 0.0f;
                bool retraced_close = short_side
                    ? max_since_min[c] > min_since_open[c]
                        * (1.0f + close_retracement)
                    : min_since_max[c] < max_since_open[c]
                        * (1.0f - close_retracement);
                bool crossed_close = short_side
                    ? min_since_open[c] < pprice[c]
                        * (1.0f - close_threshold)
                    : max_since_open[c] > pprice[c]
                        * (1.0f + close_threshold);
                bool close_triggered = true;
                float close_target = pprice[c] * (short_side
                    ? 1.0f - close_threshold : 1.0f + close_threshold);
                if (trailing_close) {
                    if (close_threshold <= 0.0f) {
                        close_triggered = close_retracement > 0.0f
                            && retraced_close;
                        close_target = price_now;
                    } else if (close_retracement > 0.0f) {
                        close_triggered = crossed_close && retraced_close;
                        close_target = pprice[c] * (short_side
                            ? 1.0f - close_threshold + close_retracement
                            : 1.0f + close_threshold - close_retracement);
                    }
                }
                int close_touch = short_side ? touch_down : touch_up;
                int target_close_tick = short_side
                    ? int(floor(close_target / price_step + 1.0e-6f))
                    : int(ceil(close_target / price_step - 1.0e-6f));
                bool close_touch_controls = (trailing_close
                    && close_threshold <= 0.0f) || (short_side
                        ? close_touch < target_close_tick
                        : close_touch > target_close_tick);
                int candidate_close_tick = close_touch_controls
                    ? touch_nearest_ticks[k * C + c] : target_close_tick;
                float close_price = float(candidate_close_tick) * price_step;
                float minimum_close = close_touch_controls
                    ? as_type<float>(touch_min_qty_bits[k * C + c])
                    : min_entry_qty(
                        close_price, qty_step, min_qty, min_cost, c_mult
                    );
                int minimum_close_relation = close_touch_controls
                    ? touch_min_qty_relation[k * C + c] : 0;
                float close_pct = trailing_close ? coin_close_qty_pct
                    : (coin_close_threshold_we == 0.0f
                        ? 1.0f : coin_close_qty_pct);
                float clip = calc_close_qty(
                    psize[c], pprice[c], balance, allowed_coin_wel,
                    minimum_close, minimum_close_relation, close_pct,
                    qty_step, c_mult
                );
                close_qty[c] = psize[c] > 0.0f && close_price > 0.0f
                        && (!trailing_close || close_triggered)
                    ? clip : 0.0f;
                close_tick[c] = candidate_close_tick;
                if (reducer_qty > 0.0f && reducer_tick > 0) {
                    float reducer_price = float(reducer_tick) * price_step;
                    float reducer_min = min_entry_qty(
                        reducer_price, qty_step, min_qty, min_cost, c_mult
                    );
                    if (use_twel && wel_reducer_qty <= 0.0f
                        && trailing_close && close_qty[c] > 0.0f) {
                        float ordinary_qty = close_qty[c];
                        if (ordinary_qty + reducer_qty > psize[c]) {
                            ordinary_qty = fmax(
                                round_step(psize[c] - reducer_qty, qty_step),
                                0.0f
                            );
                        }
                        bool ordinary_below_minimum = ordinary_qty
                                < minimum_close
                            || (ordinary_qty == minimum_close
                                && minimum_close_relation > 0);
                        if (!ordinary_below_minimum) {
                            float remainder = fmax(
                                round_step(
                                    psize[c] - reducer_qty - ordinary_qty,
                                    qty_step
                                ),
                                0.0f
                            );
                            float minimum_any = fmin(
                                minimum_close, reducer_min
                            );
                            if (remainder > 0.0f
                                && remainder < minimum_any) {
                                ordinary_qty = fmin(
                                    psize[c] - reducer_qty,
                                    round_step(
                                        ordinary_qty + remainder, qty_step
                                    )
                                );
                            }
                            secondary_close_qty[c] = ordinary_qty;
                            secondary_close_tick[c] = close_tick[c];
                        }
                    } else if (!trailing_close) {
                        close_reconstruct_after_reducer[c] = true;
                        close_gen_balance[c] = balance;
                        close_gen_allowed_wel[c] = allowed_coin_wel;
                        close_grid_gen_psize[c] = fmax(
                            round_step(
                                psize[c] - wel_reducer_qty, qty_step
                            ),
                            0.0f
                        );
                        close_grid_max_rungs[c] = wel_reducer_qty > 0.0f
                            ? 499 : 500;
                    }
                    if (secondary_close_qty[c] <= 0.0f) {
                        float remainder = fmax(
                            round_step(psize[c] - reducer_qty, qty_step),
                            0.0f
                        );
                        if (remainder > 0.0f && remainder < reducer_min) {
                            reducer_qty = psize[c];
                        }
                    }
                    close_qty[c] = reducer_qty;
                    close_tick[c] = reducer_tick;
                }
            }

            float gated_twel = twel;
            if (isfinite(twel_threshold) && twel_threshold > 0.0f) {
                gated_twel = fmin(twel, twel * twel_threshold);
            }
            float total_cap = twel_entry_gate_enabled
                ? gated_twel - 1.0e-7f : INFINITY;
            float proposed_twe = current_twe;
            for (int c = 0; c < C; ++c) {
                if (entry_candidate[c]) proposed_twe += contribution[c];
            }
            if (current_twe >= total_cap) {
                for (int c = 0; c < C; ++c) entry_qty[c] = 0.0f;
            } else if (proposed_twe >= total_cap) {
                bool processed[MAX_COINS];
                for (int c = 0; c < MAX_COINS; ++c) processed[c] = false;
                float running_twe = current_twe;
                for (int rank = 0; rank < C; ++rank) {
                    int best = -1;
                    float best_distance = INFINITY;
                    for (int c = 0; c < C; ++c) {
                        if (!entry_candidate[c] || processed[c]) continue;
                        int bar_offset = (k * C + c) * 4;
                        int coin_offset = c * COIN_COLS;
                        float price_now = bars[bar_offset + 2];
                        float price_step = coin_settings[coin_offset + 1];
                        float entry_price = float(entry_tick[c]) * price_step;
                        float distance = (short_side
                            ? entry_price - price_now
                            : price_now - entry_price) / fmax(price_now, 1.0e-12f);
                        // Exact Rust removes equal-distance entries in ascending
                        // symbol order, so the retained order is descending.
                        if (best < 0 || distance < best_distance
                            || (distance == best_distance && c > best)) {
                            best = c;
                            best_distance = distance;
                        }
                    }
                    if (best < 0) break;
                    processed[best] = true;
                    if (running_twe + contribution[best] < total_cap) {
                        running_twe += contribution[best];
                        continue;
                    }
                    int coin_offset = best * COIN_COLS;
                    float qty_step = coin_settings[coin_offset + 0];
                    float price_step = coin_settings[coin_offset + 1];
                    float c_mult = coin_settings[coin_offset + 4];
                    float price = float(entry_tick[best]) * price_step;
                    float room_cost = fmax((total_cap - running_twe) * balance, 0.0f);
                    float partial = floor_step(
                        room_cost / fmax(price * c_mult, 1.0e-12f), qty_step
                    );
                    entry_qty[best] = partial + 1.0e-6f >= minimum_entry[best]
                        ? partial : 0.0f;
                    for (int c = 0; c < C; ++c) {
                        if (entry_candidate[c] && !processed[c]) entry_qty[c] = 0.0f;
                    }
                    break;
                }
            }
        }

        float unrealized = 0.0f;
        bool any_valid = false;
        for (int c = 0; c < C; ++c) {
            int coin_offset = c * COIN_COLS;
            int bar_offset = (k * C + c) * 4;
            float close = bars[bar_offset + 2];
            if (k >= int(coin_settings[coin_offset + 6])
                && k <= int(coin_settings[coin_offset + 7])
                && finite_positive(close)) {
                any_valid = true;
            }
            if (psize[c] > 0.0f) {
                unrealized += psize[c] * coin_settings[coin_offset + 4]
                    * (short_side ? pprice[c] - close : close - pprice[c]);
            }
        }
        float equity = balance + unrealized;
        bool active = equity_started && alive && any_valid;
        if (active) {
            if (first_eq_k < 0.0f) first_eq_k = float(k);
            last_eq_k = float(k);
            bool liquidated = balance <= 0.0f || equity <= liquidation_floor;
            float effective_equity = liquidated ? liquidation_floor : equity;
            if (effective_equity > run_peak) {
                if (last_high_k >= 0.0f) {
                    recovery_max_min = fmax(
                        recovery_max_min, float(k) - last_high_k
                    );
                }
                last_high_k = float(k);
                run_peak = effective_equity;
            }
            float drawdown = fmax(
                (run_peak - effective_equity) / fmax(fabs(run_peak), 1.0e-12f), 0.0f
            );
            max_dd = fmax(max_dd, drawdown);
            day_end = effective_equity;
            day_min = fmin(day_min, effective_equity);
            day_min_balance = fmin(day_min_balance, balance);
            day_dd = fmax(day_dd, drawdown);
            day_touched = true;
            if (liquidated) {
                alive = false;
                liquidation_day = day_index;
            }
        }
    }

    if (day_touched && current_day >= 0 && current_day < D) {
        int output = (int(b) * D + current_day) * DAILY_COLS;
        daily[output + 0] = day_end;
        daily[output + 1] = day_min;
        daily[output + 2] = day_dd;
        daily[output + 3] = day_volume;
        daily[output + 4] = day_has_fill;
        daily[output + 5] = day_min_balance;
    }

    float total_size = 0.0f;
    float total_cost = 0.0f;
    int open_positions = 0;
    for (int c = 0; c < C; ++c) {
        if (psize[c] <= 0.0f) continue;
        total_size += psize[c];
        total_cost += psize[c] * pprice[c] * coin_settings[c * COIN_COLS + 4];
        open_positions += 1;
        if (position_open_k[c] >= 0.0f && last_eq_k >= 0.0f) {
            held_max_min = fmax(held_max_min, last_eq_k - position_open_k[c]);
        }
    }
    int scalar_offset = int(b) * SCALAR_COLS;
    scalars[scalar_offset + 0] = max_dd;
    scalars[scalar_offset + 1] = held_max_min * interval_ms;
    scalars[scalar_offset + 2] = gap_max_min * interval_ms;
    scalars[scalar_offset + 3] = first_fill_k >= 0.0f
        ? first_fill_k * interval_ms : -1.0f;
    scalars[scalar_offset + 4] = last_fill_k >= 0.0f
        ? last_fill_k * interval_ms : -1.0f;
    scalars[scalar_offset + 5] = recovery_max_min * interval_ms;
    scalars[scalar_offset + 6] = last_high_k >= 0.0f
        ? last_high_k * interval_ms : -1.0f;
    scalars[scalar_offset + 7] = first_eq_k >= 0.0f
        ? first_eq_k * interval_ms : -1.0f;
    scalars[scalar_offset + 8] = last_eq_k >= 0.0f
        ? last_eq_k * interval_ms : -1.0f;
    scalars[scalar_offset + 9] = float(liquidation_day);
    scalars[scalar_offset + 10] = balance;
    scalars[scalar_offset + 11] = short_side ? 0.0f : total_size;
    scalars[scalar_offset + 12] = short_side ? 0.0f : total_cost;
    scalars[scalar_offset + 13] = alive ? 1.0f : 0.0f;
    scalars[scalar_offset + 14] = float(open_positions);
    scalars[scalar_offset + 15] = short_side ? total_size : 0.0f;
    scalars[scalar_offset + 16] = short_side ? total_cost : 0.0f;
}

kernel void passivbot_trailing_martingale_multicoin(
    constant float* bars,
    constant int* fill_ticks,
    constant int* touch_ticks,
    constant int* touch_nearest_ticks,
    constant int* touch_min_qty_bits,
    constant int* touch_min_qty_relation,
    constant float* coin_settings,
    constant float* coin_overrides,
    constant float* params,
    constant float* run_settings,
    constant int* sizes,
    device float* daily,
    device float* scalars,
    device int* gap_hist,
    uint b [[thread_position_in_grid]]
) {
    const bool short_side = run_settings[3] > 0.5f;
    passivbot_trailing_martingale_multicoin_impl(
        bars, fill_ticks, touch_ticks, touch_nearest_ticks,
        touch_min_qty_bits, touch_min_qty_relation, coin_settings,
        coin_overrides, params, run_settings,
        sizes, daily, scalars, gap_hist, b, short_side
    );
}

kernel void passivbot_trailing_martingale_multicoin_long(
    constant float* bars,
    constant int* fill_ticks,
    constant int* touch_ticks,
    constant int* touch_nearest_ticks,
    constant int* touch_min_qty_bits,
    constant int* touch_min_qty_relation,
    constant float* coin_settings,
    constant float* coin_overrides,
    constant float* params,
    constant float* run_settings,
    constant int* sizes,
    device float* daily,
    device float* scalars,
    device int* gap_hist,
    uint b [[thread_position_in_grid]]
) {
    passivbot_trailing_martingale_multicoin_impl(
        bars, fill_ticks, touch_ticks, touch_nearest_ticks,
        touch_min_qty_bits, touch_min_qty_relation, coin_settings,
        coin_overrides, params, run_settings,
        sizes, daily, scalars, gap_hist, b, false
    );
}
