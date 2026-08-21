#include <metal_stdlib>
using namespace metal;

// PASSIVBOT_HSL_COMMON

constant int MAX_COINS = 64;
constant int PARAM_COLS = 42;
constant int COIN_COLS = 12;
constant int OVERRIDE_COLS = 29;
constant int HSL_OVERRIDE_START = 19;
constant int DAILY_COLS = 9;
constant int SCALAR_COLS = 57;
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

inline bool realized_loss_proxy_allows_close(
    float qty, float close_price, float pprice, bool short_side,
    float c_mult, float maker_fee, bool gate_enabled
) {
    if (!gate_enabled) return true;
    if (!(qty > 0.0f && close_price > 0.0f && pprice > 0.0f)) return false;
    float gross_pnl = qty * c_mult * (short_side
        ? pprice - close_price : close_price - pprice);
    float fee = qty * close_price * c_mult * maker_fee;
    float net_pnl = gross_pnl - fee;
    // A zero-loss envelope avoids shared loss-budget reservations across
    // independently dispatched coins and sides. The float32 margin covers
    // 1024 unit roundoffs in accumulated position price and this projection.
    float arithmetic_scale = fabs(gross_pnl) + fabs(fee)
        + qty * fabs(c_mult) * (fabs(close_price) + fabs(pprice));
    float margin = 1.220703125e-4f * arithmetic_scale;
    return isfinite(net_pnl) && net_pnl > margin;
}

inline float float32_floor_nonnegative(float value) {
    if (!(value > 0.0f) || !isfinite(value)) return fmax(value, 0.0f);
    return as_type<float>(as_type<uint>(value) - 1u);
}

inline void record_realized_net(
    float net_pnl,
    thread float& realized_pnl_cumsum_last,
    thread float& realized_pnl_cumsum_max,
    thread float& day_fill_count,
    thread float& fill_count,
    thread float& fill_count_entry,
    thread float& fill_count_long,
    thread float& pnl_recovery_peak,
    thread float& pnl_recovery_peak_k,
    thread float& pnl_recovery_max_min,
    float fill_k,
    bool is_entry,
    bool is_long
) {
    day_fill_count += 1.0f;
    fill_count += 1.0f;
    if (is_entry) fill_count_entry += 1.0f;
    if (is_long) fill_count_long += 1.0f;
    realized_pnl_cumsum_last += net_pnl;
    realized_pnl_cumsum_max = fmax(
        realized_pnl_cumsum_max, realized_pnl_cumsum_last
    );
    if (realized_pnl_cumsum_last > pnl_recovery_peak) {
        if (pnl_recovery_peak_k >= 0.0f) {
            pnl_recovery_max_min = fmax(
                pnl_recovery_max_min, fill_k - pnl_recovery_peak_k
            );
        }
        pnl_recovery_peak = realized_pnl_cumsum_last;
        pnl_recovery_peak_k = fill_k;
    }
}

inline void record_gross_pnl(
    float pnl, thread float& profit_sum, thread float& loss_sum
) {
    if (pnl > 0.0f) profit_sum += pnl;
    else loss_sum += fabs(pnl);
}

inline bool realized_loss_proxy_allows_reducer(
    float qty, float close_price, float pprice, bool short_side,
    float c_mult, float maker_fee, bool is_unstuck,
    bool gate_enabled, float balance,
    float realized_pnl_cumsum_last, float realized_pnl_cumsum_max,
    float max_realized_loss_pct
) {
    if (!is_unstuck) {
        return realized_loss_proxy_allows_close(
            qty, close_price, pprice, short_side,
            c_mult, maker_fee, gate_enabled
        );
    }
    if (!(qty > 0.0f && close_price > 0.0f && pprice > 0.0f)) return false;
    if (!gate_enabled) return true;
    float gross_pnl = qty * c_mult * (short_side
        ? pprice - close_price : close_price - pprice);
    float net_pnl = gross_pnl - qty * close_price * c_mult * maker_fee;
    if (!isfinite(net_pnl)) return false;
    if (net_pnl >= 0.0f) return true;
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
    return -net_pnl <= remaining_loss_budget;
}

inline float finalized_reducer_qty_with_ordinary(
    float psize, float ordinary_qty, float ordinary_price,
    float reducer_qty, float reducer_price,
    float qty_step, float min_qty, float min_cost, float c_mult
) {
    if (!(psize > 0.0f && reducer_qty > 0.0f && reducer_price > 0.0f)) {
        return 0.0f;
    }
    reducer_qty = fmin(psize, reducer_qty);
    float reducer_min = min_entry_qty(
        reducer_price, qty_step, min_qty, min_cost, c_mult
    );
    float kept_ordinary = ordinary_qty;
    if (kept_ordinary > 0.0f && ordinary_price > 0.0f) {
        float ordinary_min = min_entry_qty(
            ordinary_price, qty_step, min_qty, min_cost, c_mult
        );
        if (kept_ordinary + reducer_qty > psize) {
            kept_ordinary = fmax(
                round_step(psize - reducer_qty, qty_step), 0.0f
            );
        }
        if (kept_ordinary < ordinary_min) kept_ordinary = 0.0f;
    } else {
        kept_ordinary = 0.0f;
    }
    if (!(kept_ordinary > 0.0f)) {
        float remainder = fmax(
            round_step(psize - reducer_qty, qty_step), 0.0f
        );
        if (remainder > 0.0f && remainder < reducer_min) reducer_qty = psize;
    }
    return reducer_qty;
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

inline void passivbot_ema_anchor_multicoin_impl(
    constant float* bars,
    constant int* fill_ticks,
    constant int* touch_ticks,
    constant float* coin_settings,
    constant float* coin_overrides,
    constant float* params,
    constant float* run_settings,
    constant int* sizes,
    device float* daily,
    device float* scalars,
    device int* gap_hist,
    device float* coin_fill_counts,
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
    const bool collect_coin_fill_counts = run_settings[6] > 0.5f;
    if (b >= uint(B)) return;
    if (collect_coin_fill_counts) {
        for (int c = 0; c < C; ++c) {
            coin_fill_counts[int(b) * C + c] = 0.0f;
        }
    }

    const int po = int(b) * PARAM_COLS;
    const float base_qty_pct = params[po + 0];
    const float span_a = params[po + 1];
    const float span_b = params[po + 2];
    const float span_c = sqrt(fmax(span_a * span_b, 1.0f));
    const float span_lo = fmin(span_a, fmin(span_b, span_c));
    const float span_hi = fmax(span_a, fmax(span_b, span_c));
    const float span_mid = span_a + span_b + span_c - span_lo - span_hi;
    const float alpha0 = clamp(2.0f / (span_lo + 1.0f), 0.0f, 1.0f);
    const float alpha1 = clamp(2.0f / (span_mid + 1.0f), 0.0f, 1.0f);
    const float alpha2 = clamp(2.0f / (span_hi + 1.0f), 0.0f, 1.0f);
    const float ddf = params[po + 3];
    const float offset = params[po + 4];
    const float psize_weight = params[po + 5];
    const float weight_1h = params[po + 6];
    const float weight_1m = params[po + 7];
    const float span_1h = params[po + 8];
    const float span_1m = params[po + 9];
    const float alpha_1h = span_1h > 0.0f
        ? 2.0f / (fmax(span_1h, 1.0f) + 1.0f) : 0.0f;
    const float alpha_1m = span_1m > 0.0f
        ? clamp(2.0f / (span_1m + 1.0f), 0.0f, 1.0f) : 0.0f;
    const float cooldown_min = ceil(params[po + 10]);
    const float twel = params[po + 11];
    const float forager_volume_span = params[po + 12];
    const float forager_volatility_span = params[po + 13];
    const float volume_drop = clamp(params[po + 14], 0.0f, 1.0f);
    float w_volume = params[po + 15];
    float w_ready = params[po + 16];
    float w_volatility = params[po + 17];
    const int n_positions = max(1, int(rint(params[po + 18])));
    const float allowance_pct = params[po + 19];
    const bool legacy_raw_allowance = params[po + 20] > 0.5f;
    const bool twel_entry_gate_enabled = params[po + 21] > 0.5f;
    const float twel_threshold = params[po + 22];
    const bool twel_enforcer_enabled = params[po + 23] > 0.5f;
    const bool twel_enforcer_reduce_portfolio = params[po + 24] > 0.5f;
    const bool unstuck_enabled = params[po + 25] > 0.5f;
    const bool unstuck_ema_gating_enabled = params[po + 26] > 0.5f;
    const float unstuck_close_pct = params[po + 27];
    const float unstuck_ema_dist = params[po + 28];
    const float unstuck_loss_allowance_pct = params[po + 29];
    const float unstuck_threshold = params[po + 30];
    HslState hsl = load_hsl(params, po, 31);
    const bool coin_hsl_mode = hsl.signal_mode == HSL_SIGNAL_COIN;
    HslState coin_hsl[MAX_COINS];
    ulong coin_hsl_entry_blocked_mask = 0ul;
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
    const bool loss_gate_enabled = run_settings[5] < 1.0f;
    const float max_realized_loss_pct = run_settings[5];
    const float market_order_slippage_pct = fmax(run_settings[7], 0.0f);
    const bool hsl_panic_market = run_settings[8] > 0.5f;
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
    float unstuck_close_qty[MAX_COINS];
    float position_open_k[MAX_COINS];
    float position_last_fill_k[MAX_COINS];
    float score[MAX_COINS];
    float contribution[MAX_COINS];
    float minimum_entry[MAX_COINS];
    int entry_tick[MAX_COINS];
    int close_tick[MAX_COINS];
    int secondary_close_tick[MAX_COINS];
    int twel_close_tick[MAX_COINS];
    int unstuck_close_tick[MAX_COINS];
    bool close_is_unstuck_reducer[MAX_COINS];
    bool close_is_hsl_panic[MAX_COINS];
    bool selected[MAX_COINS];
    bool incumbent[MAX_COINS];
    bool survivor[MAX_COINS];
    bool entry_candidate[MAX_COINS];
    float alpha0_coin[MAX_COINS];
    float alpha1_coin[MAX_COINS];
    float alpha2_coin[MAX_COINS];
    float alpha_1h_coin[MAX_COINS];
    float alpha_1m_coin[MAX_COINS];
    float coin_realized_pnl[MAX_COINS];

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
        unstuck_close_qty[c] = 0.0f;
        position_open_k[c] = -1.0f;
        position_last_fill_k[c] = -1.0f;
        score[c] = -INFINITY;
        contribution[c] = 0.0f;
        minimum_entry[c] = 0.0f;
        entry_tick[c] = 0;
        close_tick[c] = 0;
        secondary_close_tick[c] = 0;
        twel_close_tick[c] = 0;
        unstuck_close_tick[c] = 0;
        close_is_unstuck_reducer[c] = false;
        close_is_hsl_panic[c] = false;
        coin_realized_pnl[c] = 0.0f;
        coin_hsl[c] = load_hsl(params, po, 31);
        if (coin_hsl_mode && c < C) {
            apply_coin_hsl_overrides(
                coin_hsl[c], coin_overrides, c,
                OVERRIDE_COLS, HSL_OVERRIDE_START
            );
        } else {
            coin_hsl[c].enabled = false;
        }
        selected[c] = false;
        incumbent[c] = false;
        survivor[c] = false;
        entry_candidate[c] = false;
        if (c < C) {
            float coin_span_a = coin_override_or(coin_overrides, c, 1, span_a);
            float coin_span_b = coin_override_or(coin_overrides, c, 2, span_b);
            float coin_span_c = sqrt(fmax(coin_span_a * coin_span_b, 1.0f));
            float coin_span_lo = fmin(coin_span_a, fmin(coin_span_b, coin_span_c));
            float coin_span_hi = fmax(coin_span_a, fmax(coin_span_b, coin_span_c));
            float coin_span_mid = coin_span_a + coin_span_b + coin_span_c
                - coin_span_lo - coin_span_hi;
            alpha0_coin[c] = clamp(2.0f / (coin_span_lo + 1.0f), 0.0f, 1.0f);
            alpha1_coin[c] = clamp(2.0f / (coin_span_mid + 1.0f), 0.0f, 1.0f);
            alpha2_coin[c] = clamp(2.0f / (coin_span_hi + 1.0f), 0.0f, 1.0f);
            float coin_span_1h = coin_override_or(coin_overrides, c, 8, span_1h);
            float coin_span_1m = coin_override_or(coin_overrides, c, 9, span_1m);
            alpha_1h_coin[c] = coin_span_1h > 0.0f
                ? 2.0f / (fmax(coin_span_1h, 1.0f) + 1.0f) : 0.0f;
            alpha_1m_coin[c] = coin_span_1m > 0.0f
                ? clamp(2.0f / (coin_span_1m + 1.0f), 0.0f, 1.0f) : 0.0f;
        } else {
            alpha0_coin[c] = alpha0;
            alpha1_coin[c] = alpha1;
            alpha2_coin[c] = alpha2;
            alpha_1h_coin[c] = alpha_1h;
            alpha_1m_coin[c] = alpha_1m;
        }
    }
    for (int j = 0; j < GAP_BINS; ++j) {
        gap_hist[int(b) * GAP_BINS + j] = 0;
    }

    float balance = starting_balance;
    float realized_pnl_cumsum_last = 0.0f;
    float realized_pnl_cumsum_max = 0.0f;
    float pnl_recovery_peak = -INFINITY;
    float pnl_recovery_peak_k = -1.0f;
    float pnl_recovery_max_min = 0.0f;
    float profit_sum = 0.0f;
    float loss_sum = 0.0f;
    float fill_count = 0.0f;
    float fill_count_entry = 0.0f;
    float fill_count_long = 0.0f;
    float fills_active_days_count = 0.0f;
    int last_active_fill_day = -1;
    bool alive = true;
    bool equity_started = false;
    bool selection_initialized = false;
    int max_tradable_seen = 0;
    int previous_effective_n_positions = 0;
    float run_peak = -INFINITY;
    float max_dd = 0.0f;
    float total_wallet_exposure_max = 0.0f;
    float total_wallet_exposure_mean = 0.0f;
    float total_wallet_exposure_samples = 0.0f;
    float held_max_min = 0.0f;
    float held_sum_min = 0.0f;
    float held_count = 0.0f;
    float position_unchanged_max_min = 0.0f;
    float first_fill_k = -1.0f;
    float last_fill_k = -1.0f;
    float gap_max_min = 0.0f;
    float last_high_k = -1.0f;
    float recovery_max_min = 0.0f;
    float account_peak = -INFINITY;
    float account_peak_k = -1.0f;
    float account_recovery_max_min = 0.0f;
    float first_eq_k = -1.0f;
    float last_eq_k = -1.0f;
    int liquidation_day = -1;
    float hsl_tier_samples_total = 0.0f;
    float hsl_tier_samples_yellow = 0.0f;
    float hsl_tier_samples_orange = 0.0f;
    float hsl_tier_samples_red = 0.0f;

    int current_day = 0;
    bool day_touched = false;
    float day_end = 0.0f;
    float day_min = INFINITY;
    float day_dd = 0.0f;
    float day_volume = 0.0f;
    float day_has_fill = 0.0f;
    float day_min_balance = INFINITY;
    float day_start_balance = balance;
    float day_fill_count = 0.0f;

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
                daily[output + 6] = balance - day_start_balance;
                daily[output + 7] = balance;
                daily[output + 8] = day_fill_count;
            }
            current_day = day_index;
            day_touched = false;
            day_end = 0.0f;
            day_min = INFINITY;
            day_dd = 0.0f;
            day_volume = 0.0f;
            day_has_fill = 0.0f;
            day_min_balance = INFINITY;
            day_start_balance = balance;
            day_fill_count = 0.0f;
        }

        bool any_fill = false;
        float hsl_equity_before_fills = balance;
        for (int c = 0; c < C; ++c) {
            int coin_offset = c * COIN_COLS;
            int bar_offset = (k * C + c) * 4;
            float close = bars[bar_offset + 2];
            if (psize[c] > 0.0f && finite_positive(close)) {
                hsl_equity_before_fills += psize[c]
                    * coin_settings[coin_offset + 4]
                    * (short_side ? pprice[c] - close : close - pprice[c]);
            }
        }
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
            const float c_mult = coin_settings[coin_offset + 4];
            const float maker_fee = coin_settings[coin_offset + 5];
            const float taker_fee = coin_settings[coin_offset + 11];

            bool coin_hsl_panic_market = coin_hsl_mode
                ? coin_override_or(
                    coin_overrides, c, HSL_OVERRIDE_START + 9,
                    hsl_panic_market ? 1.0f : 0.0f
                ) > 0.5f
                : hsl_panic_market;
            bool primary_market_panic = close_is_hsl_panic[c]
                && coin_hsl_panic_market;
            bool filled_close = close_qty[c] > 0.0f && psize[c] > 0.0f
                && (primary_market_panic || (short_side
                    ? close_tick[c] > fill_ticks[tick_offset + 1]
                    : close_tick[c] <= fill_ticks[tick_offset + 0]));
            bool filled_secondary_close = secondary_close_qty[c] > 0.0f
                && psize[c] > 0.0f
                && (short_side
                    ? secondary_close_tick[c] > fill_ticks[tick_offset + 1]
                    : secondary_close_tick[c] <= fill_ticks[tick_offset + 0]);
            bool secondary_first = filled_secondary_close
                && (!filled_close || (short_side
                    ? secondary_close_tick[c] > close_tick[c]
                    : secondary_close_tick[c] < close_tick[c]));
            bool executed_close = false;
            for (int close_rank = 0; close_rank < 2; ++close_rank) {
                bool use_secondary = secondary_first
                    ? close_rank == 0 : close_rank == 1;
                bool reachable = use_secondary
                    ? filled_secondary_close : filled_close;
                if (!reachable || psize[c] <= 0.0f) continue;
                int executed_tick = use_secondary
                    ? secondary_close_tick[c] : close_tick[c];
                float requested_qty = use_secondary
                    ? secondary_close_qty[c] : close_qty[c];
                bool market_panic = !use_secondary && primary_market_panic;
                float fill_price = market_panic
                    ? fmax(
                        short_side
                            ? ceil_step(
                                close * (1.0f + market_order_slippage_pct),
                                price_step
                            )
                            : floor_step(
                                close * (1.0f - market_order_slippage_pct),
                                price_step
                            ),
                        price_step
                    )
                    : float(executed_tick) * price_step;
                float adjusted = fmin(
                    round_step(requested_qty, qty_step), psize[c]
                );
                if (!(adjusted > 0.0f)) continue;
                float pnl = adjusted * c_mult * (short_side
                    ? pprice[c] - fill_price
                    : fill_price - pprice[c]);
                float net_pnl = pnl
                    - adjusted * fill_price * c_mult
                        * (market_panic ? taker_fee : maker_fee);
                bool is_unstuck = !use_secondary
                    && close_is_unstuck_reducer[c];
                bool is_hsl_panic = !use_secondary
                    && close_is_hsl_panic[c];
                if (!is_hsl_panic && !realized_loss_proxy_allows_reducer(
                        adjusted, fill_price, pprice[c], short_side,
                        c_mult, maker_fee, is_unstuck, loss_gate_enabled,
                        balance, realized_pnl_cumsum_last,
                        realized_pnl_cumsum_max, max_realized_loss_pct
                    )) {
                    continue;
                }
                if (is_hsl_panic) {
                    if (coin_hsl_mode) {
                        record_hsl_panic_fill(
                            coin_hsl[c], net_pnl,
                            hsl_equity_before_fills
                        );
                    } else {
                        record_hsl_panic_fill(
                            hsl, net_pnl, hsl_equity_before_fills
                        );
                    }
                }
                record_gross_pnl(pnl, profit_sum, loss_sum);
                balance += net_pnl;
                record_realized_net(
                    net_pnl, realized_pnl_cumsum_last,
                    realized_pnl_cumsum_max,
                    day_fill_count, fill_count, fill_count_entry, fill_count_long,
                    pnl_recovery_peak, pnl_recovery_peak_k,
                    pnl_recovery_max_min, float(k),
                    false, !short_side
                );
                coin_realized_pnl[c] += net_pnl;
                if (coin_hsl_mode) {
                    record_coin_hsl_realized_fill(
                        coin_hsl[c], coin_realized_pnl[c]
                    );
                    advance_coin_hsl_equity_after_close_fill(
                        hsl_equity_before_fills,
                        net_pnl, adjusted, pprice[c], close,
                        c_mult, short_side
                    );
                }
                if (collect_coin_fill_counts) {
                    coin_fill_counts[int(b) * C + c] += 1.0f;
                }
                float new_size = fmax(round_step(psize[c] - adjusted, qty_step), 0.0f);
                bool went_flat = new_size <= 0.0f;
                psize[c] = new_size;
                if (went_flat) {
                    pprice[c] = 0.0f;
                    if (position_open_k[c] >= 0.0f) {
                        float held_min = float(k) - position_open_k[c];
                        held_max_min = fmax(held_max_min, held_min);
                        held_sum_min += held_min;
                        held_count += 1.0f;
                    }
                    position_open_k[c] = -1.0f;
                }
                day_volume += fabs(adjusted) * fill_price * c_mult / balance;
                if (use_secondary) {
                    secondary_close_qty[c] = 0.0f;
                } else {
                    close_qty[c] = 0.0f;
                    close_is_unstuck_reducer[c] = false;
                    close_is_hsl_panic[c] = false;
                }
                executed_close = true;
            }
            if (executed_close) {
                any_fill = true;
            }

            bool was_flat = psize[c] <= 0.0f;
            bool filled_entry = entry_qty[c] > 0.0f
                && (short_side
                    ? entry_tick[c] <= fill_ticks[tick_offset + 0]
                    : entry_tick[c] > fill_ticks[tick_offset + 1]);
            if (filled_entry) {
                float fill_price = float(entry_tick[c]) * price_step;
                float adjusted = round_step(entry_qty[c], qty_step);
                float fee = adjusted * fill_price * c_mult * maker_fee;
                balance -= fee;
                record_realized_net(
                    -fee, realized_pnl_cumsum_last,
                    realized_pnl_cumsum_max,
                    day_fill_count, fill_count, fill_count_entry, fill_count_long,
                    pnl_recovery_peak, pnl_recovery_peak_k,
                    pnl_recovery_max_min, float(k),
                    true, !short_side
                );
                coin_realized_pnl[c] -= fee;
                if (coin_hsl_mode) {
                    record_coin_hsl_realized_fill(
                        coin_hsl[c], coin_realized_pnl[c]
                    );
                    advance_coin_hsl_equity_after_entry_fill(
                        hsl_equity_before_fills,
                        fee, adjusted, fill_price, close,
                        c_mult, short_side
                    );
                }
                if (collect_coin_fill_counts) {
                    coin_fill_counts[int(b) * C + c] += 1.0f;
                }
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
                any_fill = true;
            }
            if (executed_close || filled_entry) {
                if (position_last_fill_k[c] >= 0.0f) {
                    position_unchanged_max_min = fmax(
                        position_unchanged_max_min,
                        float(k) - position_last_fill_k[c]
                    );
                }
                position_last_fill_k[c] = psize[c] > 0.0f ? float(k) : -1.0f;
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
        }

        int tradable_count = 0;
        for (int c = 0; c < C; ++c) {
            int coin_offset = c * COIN_COLS;
            int bar_offset = (k * C + c) * 4;
            float close = bars[bar_offset + 2];
            float coin_wel = coin_override_or(coin_overrides, c, 11, -1.0f);
            if (k >= int(coin_settings[coin_offset + 8])
                && k <= int(coin_settings[coin_offset + 7])
                && finite_positive(close) && coin_wel != 0.0f) {
                tradable_count += 1;
            }
        }
        const bool post_fill_balance_depleted = isfinite(balance) && balance <= 0.0f;
        if (alive && !post_fill_balance_depleted) {
            max_tradable_seen = max(max_tradable_seen, tradable_count);
        }
        const int effective_n_positions = min(n_positions, max_tradable_seen);
        const bool can_generate = alive && effective_n_positions > 0
            && k > max(global_warmup, 1) && k >= requested_start_k;
        equity_started = equity_started || can_generate;
        bool has_hsl_position = false;
        for (int c = 0; c < C; ++c) {
            has_hsl_position = has_hsl_position || psize[c] > 0.0f;
        }
        int current_hsl_mode = coin_hsl_mode
            ? 0 : hsl_mode(hsl, has_hsl_position);

        if (can_generate) {
            // Exact Rust ranks flat candidates every minute. Re-ranking only
            // after state changes keeps the proxy inexpensive; independent
            // exact validations and drift gates police this approximation.
            bool coin_hsl_eligibility_changed = false;
            if (coin_hsl_mode) {
                ulong blocked_mask = 0ul;
                for (int c = 0; c < C; ++c) {
                    if (hsl_mode(coin_hsl[c], false) != 0) {
                        blocked_mask |= 1ul << ulong(c);
                    }
                }
                coin_hsl_eligibility_changed =
                    blocked_mask != coin_hsl_entry_blocked_mask;
                coin_hsl_entry_blocked_mask = blocked_mask;
            }
            bool reselect = !selection_initialized || any_fill
                || coin_hsl_eligibility_changed
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
                    float coin_wel = coin_override_or(coin_overrides, c, 11, -1.0f);
                    bool enabled = !selected[c]
                        && k >= int(coin_settings[coin_offset + 8])
                        && k <= int(coin_settings[coin_offset + 7])
                        && finite_positive(bars[bar_offset + 2])
                        && coin_wel != 0.0f
                        && (!coin_hsl_mode || (
                            coin_hsl_entry_blocked_mask & (1ul << ulong(c))
                        ) == 0ul);
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
                    float coin_offset_pct = coin_override_or(
                        coin_overrides, c, 4, offset
                    );
                    float threshold = short_side
                        ? upper * (1.0f + coin_offset_pct)
                        : lower * (1.0f - coin_offset_pct);
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
                    float coin_offset_pct = coin_override_or(
                        coin_overrides, c, 4, offset
                    );
                    float threshold = short_side
                        ? upper * (1.0f + coin_offset_pct)
                        : lower * (1.0f - coin_offset_pct);
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

            // Match calc_twel_enforcer_actions: every open position contributes
            // to total exposure. Reducible positions are ranked by least
            // adverse projected loss per exposure, then symbol index.
            float twel_repair_target = twel * twel_threshold;
            if (twel_enforcer_enabled && twel_threshold > 0.0f
                && twel_repair_target > 0.0f && balance > 0.0f
                && current_twe > twel_repair_target + 1.0e-9f
                && open_position_count > 0) {
                // Reduce-overweight uses the current eligible count. The
                // grow-only maximum remains exclusive to dynamic WEL sizing.
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

            // Exact Rust emits one global auto-unstuck intent across all
            // positions. A directional multicoin thread owns that complete
            // one-side portfolio, so it can apply the same least-stuck rank.
            for (int c = 0; c < C; ++c) {
                unstuck_close_qty[c] = 0.0f;
                unstuck_close_tick[c] = 0;
                close_is_unstuck_reducer[c] = false;
            }
            float balance_peak = balance
                + (realized_pnl_cumsum_max - realized_pnl_cumsum_last);
            int unstuck_coin = -1;
            float best_unstuck_diff = INFINITY;
            float selected_unstuck_qty = 0.0f;
            int selected_unstuck_tick = 0;
            for (int c = 0; c < C; ++c) {
                int coin_offset = c * COIN_COLS;
                int bar_offset = (k * C + c) * 4;
                int tick_offset = (k * C + c) * 2;
                float price_now = bars[bar_offset + 2];
                float c_mult = coin_settings[coin_offset + 4];
                float qty_step = coin_settings[coin_offset + 0];
                float price_step = coin_settings[coin_offset + 1];
                float min_qty = coin_settings[coin_offset + 2];
                float min_cost = coin_settings[coin_offset + 3];
                bool coin_unstuck_enabled = coin_override_or(
                    coin_overrides, c, 13, unstuck_enabled ? 1.0f : 0.0f
                ) > 0.5f;
                bool coin_ema_gate = coin_override_or(
                    coin_overrides, c, 14,
                    unstuck_ema_gating_enabled ? 1.0f : 0.0f
                ) > 0.5f;
                float coin_close_pct = coin_override_or(
                    coin_overrides, c, 15, unstuck_close_pct
                );
                float coin_ema_dist = coin_override_or(
                    coin_overrides, c, 16, unstuck_ema_dist
                );
                float coin_loss_allowance_pct = coin_override_or(
                    coin_overrides, c, 17, unstuck_loss_allowance_pct
                );
                float coin_threshold = coin_override_or(
                    coin_overrides, c, 18, unstuck_threshold
                );
                float fixed_coin_wel = coin_override_or(
                    coin_overrides, c, 11, -1.0f
                );
                float coin_wel = fixed_coin_wel >= 0.0f
                    ? fixed_coin_wel : effective_wel;
                float coin_allowance_pct = coin_override_or(
                    coin_overrides, c, 12, allowance_pct
                );
                float allowed_coin_wel = allowed_wallet_exposure_limit(
                    coin_wel, twel, coin_allowance_pct, legacy_raw_allowance
                );
                if (!(coin_unstuck_enabled && coin_close_pct > 0.0f
                    && coin_loss_allowance_pct > 0.0f && coin_threshold > 0.0f
                    && balance > 0.0f && balance_peak > 0.0f
                    && psize[c] > 0.0f && pprice[c] > 0.0f
                    && allowed_coin_wel > 0.0f && price_now > 0.0f)) {
                    continue;
                }
                float allowance = float32_floor_nonnegative(fmax(
                    balance - balance_peak * (
                        1.0f - coin_loss_allowance_pct * twel
                    ),
                    0.0f
                ));
                float wallet_exposure = psize[c] * pprice[c] * c_mult / balance;
                if (!(allowance > 0.0f
                    && wallet_exposure / allowed_coin_wel > coin_threshold)) {
                    continue;
                }
                if (coin_ema_gate) {
                    float lower = fmin(ema0[c], fmin(ema1[c], ema2[c]));
                    float upper = fmax(ema0[c], fmax(ema1[c], ema2[c]));
                    int trigger_tick = short_side
                        ? int(floor(
                            lower * (1.0f - coin_ema_dist) / price_step
                                + 1.0e-6f
                        ))
                        : int(ceil(
                            upper * (1.0f + coin_ema_dist) / price_step
                                - 1.0e-6f
                        ));
                    bool triggered = short_side
                        ? touch_ticks[tick_offset + 1] <= trigger_tick
                        : touch_ticks[tick_offset + 0] >= trigger_tick;
                    if (!triggered) continue;
                }
                int reducer_tick = max(
                    short_side
                        ? touch_ticks[tick_offset + 0]
                        : touch_ticks[tick_offset + 1],
                    1
                );
                float reducer_price = float(reducer_tick) * price_step;
                float reducer_min = min_entry_qty(
                    reducer_price, qty_step, min_qty, min_cost, c_mult
                );
                float target_qty = floor_step(
                    balance * allowed_coin_wel * coin_close_pct
                        / fmax(reducer_price * c_mult, 1.0e-12f),
                    qty_step
                );
                float reducer_qty = fmin(
                    psize[c], fmax(reducer_min, target_qty)
                );
                float gross_pnl = reducer_qty * c_mult * (short_side
                    ? pprice[c] - reducer_price
                    : reducer_price - pprice[c]);
                if (gross_pnl < 0.0f && -gross_pnl > allowance) {
                    float scaled_qty = fmin(
                        psize[c], reducer_qty * allowance / -gross_pnl
                    );
                    reducer_qty = fmin(
                        psize[c], fmax(
                            reducer_min, floor_step(scaled_qty, qty_step)
                        )
                    );
                }
                float pprice_diff = short_side
                    ? price_now / pprice[c] - 1.0f
                    : 1.0f - price_now / pprice[c];
                if (unstuck_coin < 0 || pprice_diff < best_unstuck_diff
                    || (pprice_diff == best_unstuck_diff && c < unstuck_coin)) {
                    unstuck_coin = c;
                    best_unstuck_diff = pprice_diff;
                    selected_unstuck_qty = reducer_qty;
                    selected_unstuck_tick = reducer_tick;
                }
            }
            if (unstuck_coin >= 0) {
                unstuck_close_qty[unstuck_coin] = selected_unstuck_qty;
                unstuck_close_tick[unstuck_coin] = selected_unstuck_tick;
            }
            for (int c = 0; c < C; ++c) {
                entry_qty[c] = 0.0f;
                close_qty[c] = 0.0f;
                secondary_close_qty[c] = 0.0f;
                secondary_close_tick[c] = 0;
                close_is_hsl_panic[c] = false;
                contribution[c] = 0.0f;
                entry_candidate[c] = false;
                int coin_offset = c * COIN_COLS;
                int bar_offset = (k * C + c) * 4;
                int tick_offset = (k * C + c) * 2;
                float price_now = bars[bar_offset + 2];
                float fixed_coin_wel = coin_override_or(
                    coin_overrides, c, 11, -1.0f
                );
                float coin_wel = fixed_coin_wel >= 0.0f
                    ? fixed_coin_wel : effective_wel;
                float coin_allowance_pct = coin_override_or(
                    coin_overrides, c, 12, allowance_pct
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
                float coin_weight_1h = coin_override_or(
                    coin_overrides, c, 6, weight_1h
                );
                float coin_weight_1m = coin_override_or(
                    coin_overrides, c, 7, weight_1m
                );
                float multiplier = fmax(
                    1.0f + volatility_1h[c] * coin_weight_1h
                        + volatility_1m[c] * coin_weight_1m,
                    1.0f
                );
                float wallet_ratio = psize[c] > 0.0f && balance > 0.0f
                    ? (psize[c] * price_now * c_mult / balance)
                        / fmax(coin_wel, 1.0e-12f) : 0.0f;
                float signed_wallet_ratio = short_side ? -wallet_ratio : wallet_ratio;
                float coin_psize_weight = coin_override_or(
                    coin_overrides, c, 5, psize_weight
                );
                float coin_offset_pct = coin_override_or(
                    coin_overrides, c, 4, offset
                );
                float inventory_shift = signed_wallet_ratio * coin_psize_weight;
                int bid_tick = min(
                    int(floor(
                        lower * (1.0f - coin_offset_pct * multiplier - inventory_shift)
                            / price_step + 1.0e-6f
                    )),
                    touch_ticks[tick_offset + 0]
                );
                int ask_tick = max(
                    int(ceil(
                        upper * (1.0f + coin_offset_pct * multiplier - inventory_shift)
                            / price_step - 1.0e-6f
                    )),
                    touch_ticks[tick_offset + 1]
                );
                float bid_price = float(bid_tick) * price_step;
                float ask_price = float(ask_tick) * price_step;
                int candidate_entry_tick = short_side ? ask_tick : bid_tick;
                int candidate_close_tick = short_side ? bid_tick : ask_tick;
                float entry_price = short_side ? ask_price : bid_price;
                float close_price = short_side ? bid_price : ask_price;
                float minimum = min_entry_qty(
                    entry_price, qty_step, min_qty, min_cost, c_mult
                );
                minimum_entry[c] = minimum;
                float coin_cooldown_min = ceil(coin_override_or(
                    coin_overrides, c, 10, cooldown_min
                ));
                bool cooldown = coin_cooldown_min > 0.0f && last_increase_k[c] > -1.0e19f
                    && float(k) < last_increase_k[c] + coin_cooldown_min;
                float cost_we = psize[c] > 0.0f && balance > 0.0f
                    ? psize[c] * pprice[c] * c_mult / balance : 0.0f;
                float position_cap = allowed_coin_wel - 1.0e-7f;
                float coin_base_qty_pct = coin_override_or(
                    coin_overrides, c, 0, base_qty_pct
                );
                float coin_ddf = coin_override_or(coin_overrides, c, 3, ddf);
                if (selected[c] && !cooldown && entry_price > 0.0f && balance > 0.0f
                    && cost_we < position_cap && coin_base_qty_pct > 0.0f) {
                    float base_qty = fmax(minimum, round_step(
                        balance * allowed_coin_wel * coin_base_qty_pct
                            / fmax(entry_price * c_mult, 1.0e-12f),
                        qty_step
                    ));
                    float quantity = round_step(
                        base_qty * fmax(1.0f + fmax(wallet_ratio, 0.0f) * coin_ddf, 1.0f),
                        qty_step
                    );
                    float headroom = (
                        position_cap * balance - psize[c] * pprice[c] * c_mult
                    ) / fmax(entry_price * c_mult, 1.0e-12f);
                    bool over = (psize[c] * pprice[c] + quantity * entry_price) * c_mult
                        / fmax(balance, 1.0e-9f) >= position_cap;
                    if (over) {
                        float capped = floor_step(headroom, qty_step);
                        quantity = capped > 0.0f && capped + 1.0e-6f >= minimum
                            ? capped : 0.0f;
                    }
                    entry_qty[c] = quantity;
                    entry_tick[c] = candidate_entry_tick;
                    entry_candidate[c] = quantity > 0.0f;
                    contribution[c] = quantity * entry_price * c_mult / balance;
                }
                if (psize[c] > 0.0f && close_price > 0.0f) {
                    float minimum_close = min_entry_qty(
                        close_price, qty_step, min_qty, min_cost, c_mult
                    );
                    float clip = fmin(psize[c], fmax(minimum_close, round_step(
                        balance * allowed_coin_wel * coin_base_qty_pct
                            / fmax(close_price * c_mult, 1.0e-12f),
                        qty_step
                    )));
                    close_qty[c] = psize[c] <= minimum_close
                            || psize[c] - clip < minimum_close
                        ? psize[c] : clip;
                    close_tick[c] = candidate_close_tick;
                }

                // Reserve the side-wide protective reducer and trim the
                // ordinary EMA close first, matching finalized_closes_with_reducer.
                float raw_twel_qty = twel_close_qty[c];
                int raw_twel_tick = twel_close_tick[c];
                float raw_unstuck_qty = unstuck_close_qty[c];
                int raw_unstuck_tick = unstuck_close_tick[c];
                float finalized_twel_qty = finalized_reducer_qty_with_ordinary(
                    psize[c], close_qty[c], close_price,
                    raw_twel_qty, float(raw_twel_tick) * price_step,
                    qty_step, min_qty, min_cost, c_mult
                );
                float finalized_unstuck_qty = finalized_reducer_qty_with_ordinary(
                    psize[c], close_qty[c], close_price,
                    raw_unstuck_qty, float(raw_unstuck_tick) * price_step,
                    qty_step, min_qty, min_cost, c_mult
                );
                bool prefer_unstuck = finalized_unstuck_qty > 0.0f && (
                    finalized_twel_qty <= 0.0f
                    || finalized_unstuck_qty > finalized_twel_qty
                    || (finalized_unstuck_qty == finalized_twel_qty && (
                        raw_unstuck_tick != raw_twel_tick
                            ? (short_side
                                ? raw_unstuck_tick > raw_twel_tick
                                : raw_unstuck_tick < raw_twel_tick)
                            : true
                    ))
                );
                float reducer_qty = prefer_unstuck
                    ? raw_unstuck_qty : raw_twel_qty;
                int reducer_tick = prefer_unstuck
                    ? raw_unstuck_tick : raw_twel_tick;
                bool reducer_is_unstuck = prefer_unstuck;
                for (int reducer_attempt = 0; reducer_attempt < 2;
                        ++reducer_attempt) {
                    if (!(reducer_qty > 0.0f && reducer_tick > 0)) break;
                    float finalized_qty = reducer_is_unstuck
                        ? finalized_unstuck_qty : finalized_twel_qty;
                    if (realized_loss_proxy_allows_reducer(
                            finalized_qty, float(reducer_tick) * price_step,
                            pprice[c], short_side, c_mult,
                            coin_settings[coin_offset + 5],
                            reducer_is_unstuck, loss_gate_enabled, balance,
                            realized_pnl_cumsum_last,
                            realized_pnl_cumsum_max, max_realized_loss_pct
                        )) {
                        break;
                    }
                    if (reducer_attempt > 0) {
                        reducer_qty = 0.0f;
                        reducer_tick = 0;
                        reducer_is_unstuck = false;
                        break;
                    }
                    reducer_is_unstuck = !reducer_is_unstuck;
                    reducer_qty = reducer_is_unstuck
                        ? raw_unstuck_qty : raw_twel_qty;
                    reducer_tick = reducer_is_unstuck
                        ? raw_unstuck_tick : raw_twel_tick;
                }
                if (reducer_qty > 0.0f && reducer_tick > 0) {
                    float reducer_price = float(reducer_tick) * price_step;
                    float reducer_min = min_entry_qty(
                        reducer_price, qty_step, min_qty, min_cost, c_mult
                    );
                    float ordinary_qty = close_qty[c];
                    if (ordinary_qty > 0.0f) {
                        float ordinary_min = min_entry_qty(
                            close_price, qty_step, min_qty, min_cost, c_mult
                        );
                        if (ordinary_qty + reducer_qty > psize[c]) {
                            ordinary_qty = fmax(
                                round_step(
                                    psize[c] - reducer_qty, qty_step
                                ),
                                0.0f
                            );
                        }
                        if (ordinary_qty >= ordinary_min) {
                            float remainder = fmax(
                                round_step(
                                    psize[c] - reducer_qty - ordinary_qty,
                                    qty_step
                                ),
                                0.0f
                            );
                            float minimum_any = fmin(
                                ordinary_min, reducer_min
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
                    close_is_unstuck_reducer[c] = reducer_is_unstuck;
                }
                if (close_qty[c] > 0.0f && !realized_loss_proxy_allows_reducer(
                        close_qty[c], float(close_tick[c]) * price_step,
                        pprice[c], short_side, c_mult,
                        coin_settings[coin_offset + 5],
                        close_is_unstuck_reducer[c], loss_gate_enabled,
                        balance, realized_pnl_cumsum_last,
                        realized_pnl_cumsum_max, max_realized_loss_pct
                    )) {
                    close_qty[c] = 0.0f;
                    close_is_unstuck_reducer[c] = false;
                }
                if (secondary_close_qty[c] > 0.0f
                    && !realized_loss_proxy_allows_close(
                        secondary_close_qty[c],
                        float(secondary_close_tick[c]) * price_step,
                        pprice[c], short_side, c_mult,
                        coin_settings[coin_offset + 5], loss_gate_enabled
                    )) {
                    secondary_close_qty[c] = 0.0f;
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
            for (int c = 0; c < C; ++c) {
                int coin_mode = coin_hsl_mode
                    ? hsl_mode(coin_hsl[c], psize[c] > 0.0f)
                    : current_hsl_mode;
                if (coin_mode >= 2
                    || (coin_mode != 0 && psize[c] <= 0.0f)) {
                    entry_qty[c] = 0.0f;
                }
                if (coin_mode == 3 && psize[c] > 0.0f) {
                    int tick_offset = (k * C + c) * 2;
                    close_qty[c] = psize[c];
                    close_tick[c] = max(
                        short_side
                            ? touch_ticks[tick_offset + 1] + 1
                            : touch_ticks[tick_offset + 0] - 1,
                        1
                    );
                    secondary_close_qty[c] = 0.0f;
                    secondary_close_tick[c] = 0;
                    close_is_unstuck_reducer[c] = false;
                    close_is_hsl_panic[c] = true;
                }
            }
        }

        float unrealized = 0.0f;
        float position_cost = 0.0f;
        bool any_valid = false;
        bool has_open_position = false;
        bool has_blocking_orders = false;
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
                has_open_position = true;
                position_cost += psize[c] * pprice[c]
                    * coin_settings[coin_offset + 4];
                unrealized += psize[c] * coin_settings[coin_offset + 4]
                    * (short_side ? pprice[c] - close : close - pprice[c]);
            }
            int coin_mode = coin_hsl_mode
                ? hsl_mode(coin_hsl[c], psize[c] > 0.0f)
                : current_hsl_mode;
            if (coin_mode != 3 && (
                    entry_qty[c] > 0.0f || close_qty[c] > 0.0f
                        || secondary_close_qty[c] > 0.0f
                )) {
                has_blocking_orders = true;
            }
        }
        float equity = balance + unrealized;
        if (can_generate && any_valid && alive
            && balance > 0.0f && equity > liquidation_floor) {
            int sampled_hsl_tier = 0;
            bool hsl_sample_enabled = !coin_hsl_mode && hsl.enabled;
            if (coin_hsl_mode) {
                for (int c = 0; c < C; ++c) {
                    hsl_sample_enabled = hsl_sample_enabled || coin_hsl[c].enabled;
                    int coin_offset = c * COIN_COLS;
                    int bar_offset = (k * C + c) * 4;
                    float close = bars[bar_offset + 2];
                    float coin_unrealized = psize[c] > 0.0f
                        && finite_positive(close)
                        ? psize[c] * coin_settings[coin_offset + 4]
                            * (short_side ? pprice[c] - close : close - pprice[c])
                        : 0.0f;
                    int coin_mode = hsl_mode(coin_hsl[c], psize[c] > 0.0f);
                    bool coin_has_blocking_orders = coin_mode != 3 && (
                        entry_qty[c] > 0.0f || close_qty[c] > 0.0f
                            || secondary_close_qty[c] > 0.0f
                    );
                    coin_hsl[c].slot_count = float(effective_n_positions);
                    update_hsl(
                        coin_hsl[c], balance, starting_balance,
                        coin_realized_pnl[c], coin_unrealized,
                        psize[c] > 0.0f, coin_has_blocking_orders,
                        float(k), interval_ms
                    );
                    sampled_hsl_tier = max(sampled_hsl_tier, coin_hsl[c].tier);
                }
            } else {
                update_hsl(
                    hsl, balance, starting_balance,
                    realized_pnl_cumsum_last, unrealized,
                    has_open_position, has_blocking_orders,
                    float(k), interval_ms
                );
                sampled_hsl_tier = hsl.tier;
            }
            if (hsl_sample_enabled) {
                hsl_tier_samples_total += 1.0f;
                hsl_tier_samples_yellow += sampled_hsl_tier == 1 ? 1.0f : 0.0f;
                hsl_tier_samples_orange += sampled_hsl_tier == 2 ? 1.0f : 0.0f;
                hsl_tier_samples_red += sampled_hsl_tier == 3 ? 1.0f : 0.0f;
            }
            if (coin_hsl_mode) {
                for (int c = 0; c < C; ++c) {
                    try_restart_hsl(coin_hsl[c], float(k), equity);
                }
            } else {
                try_restart_hsl(hsl, float(k), equity);
            }
        }
        bool active = equity_started && alive && any_valid;
        if (active) {
            if (first_eq_k < 0.0f) first_eq_k = float(k);
            last_eq_k = float(k);
            if (any_fill) {
                int active_fill_day = int(float(k) - first_eq_k) / 1440;
                if (active_fill_day != last_active_fill_day) {
                    fills_active_days_count += 1.0f;
                    last_active_fill_day = active_fill_day;
                }
            }
            bool liquidated = balance <= 0.0f || equity <= liquidation_floor;
            float effective_equity = liquidated ? liquidation_floor : equity;
            if (effective_equity >= account_peak) {
                if (account_peak_k >= 0.0f) {
                    account_recovery_max_min = fmax(
                        account_recovery_max_min, float(k) - account_peak_k
                    );
                }
                account_peak = effective_equity;
                account_peak_k = float(k);
            }
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
            if (!liquidated) {
                float twe_abs = position_cost / balance;
                total_wallet_exposure_samples += 1.0f;
                total_wallet_exposure_mean += (
                    twe_abs - total_wallet_exposure_mean
                ) / total_wallet_exposure_samples;
                total_wallet_exposure_max = fmax(
                    total_wallet_exposure_max, twe_abs
                );
            }
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
        daily[output + 6] = balance - day_start_balance;
        daily[output + 7] = balance;
        daily[output + 8] = day_fill_count;
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
            float held_min = last_eq_k - position_open_k[c];
            held_max_min = fmax(held_max_min, held_min);
            held_sum_min += held_min;
            held_count += 1.0f;
        }
        if (position_last_fill_k[c] >= 0.0f && last_eq_k >= 0.0f) {
            position_unchanged_max_min = fmax(
                position_unchanged_max_min,
                last_eq_k - position_last_fill_k[c]
            );
        }
    }
    if (pnl_recovery_peak_k >= 0.0f && last_eq_k >= 0.0f) {
        pnl_recovery_max_min = fmax(
            pnl_recovery_max_min, last_eq_k - pnl_recovery_peak_k
        );
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
    scalars[scalar_offset + 18] = profit_sum;
    scalars[scalar_offset + 19] = loss_sum;
    scalars[scalar_offset + 20] = position_unchanged_max_min * interval_ms;
    int entry_effective_n_positions = min(n_positions, max_tradable_seen);
    float entry_base_limit = entry_effective_n_positions > 0
        ? twel / float(entry_effective_n_positions) : 0.0f;
    float entry_initial_qty_pct = coin_override_or(
        coin_overrides, 0, 0, base_qty_pct
    );
    float entry_allowance_pct = coin_override_or(
        coin_overrides, 0, 12, allowance_pct
    );
    scalars[scalar_offset + 21] = allowed_wallet_exposure_limit(
        entry_base_limit, twel, entry_allowance_pct, legacy_raw_allowance
    ) * entry_initial_qty_pct;
    scalars[scalar_offset + 22] = total_wallet_exposure_max;
    scalars[scalar_offset + 23] = total_wallet_exposure_mean;
    scalars[scalar_offset + 24] = fill_count;
    scalars[scalar_offset + 25] = fill_count_entry;
    scalars[scalar_offset + 26] = fill_count_long;
    scalars[scalar_offset + 27] = fills_active_days_count;
    scalars[scalar_offset + 28] = pnl_recovery_max_min * interval_ms;
    scalars[scalar_offset + 29] = held_sum_min * interval_ms;
    scalars[scalar_offset + 30] = held_count;
    scalars[scalar_offset + 31] = account_recovery_max_min * interval_ms;
    if (coin_hsl_mode) {
        write_one_side_coin_hsl_outputs(
            coin_hsl, C, short_side,
            hsl_tier_samples_total,
            hsl_tier_samples_yellow,
            hsl_tier_samples_orange,
            hsl_tier_samples_red,
            last_eq_k,
            scalars,
            scalar_offset + 32
        );
    } else {
        write_one_side_hsl_outputs(
            hsl, short_side,
            hsl_tier_samples_total,
            hsl_tier_samples_yellow,
            hsl_tier_samples_orange,
            hsl_tier_samples_red,
            last_eq_k,
            scalars,
            scalar_offset + 32
        );
    }
}

kernel void passivbot_ema_anchor_multicoin(
    constant float* bars,
    constant int* fill_ticks,
    constant int* touch_ticks,
    constant float* coin_settings,
    constant float* coin_overrides,
    constant float* params,
    constant float* run_settings,
    constant int* sizes,
    device float* daily,
    device float* scalars,
    device int* gap_hist,
    device float* coin_fill_counts,
    uint b [[thread_position_in_grid]]
) {
    const bool short_side = run_settings[3] > 0.5f;
    passivbot_ema_anchor_multicoin_impl(
        bars, fill_ticks, touch_ticks, coin_settings, coin_overrides, params, run_settings,
        sizes, daily, scalars, gap_hist, coin_fill_counts, b, short_side
    );
}

kernel void passivbot_ema_anchor_multicoin_long(
    constant float* bars,
    constant int* fill_ticks,
    constant int* touch_ticks,
    constant float* coin_settings,
    constant float* coin_overrides,
    constant float* params,
    constant float* run_settings,
    constant int* sizes,
    device float* daily,
    device float* scalars,
    device int* gap_hist,
    device float* coin_fill_counts,
    uint b [[thread_position_in_grid]]
) {
    passivbot_ema_anchor_multicoin_impl(
        bars, fill_ticks, touch_ticks, coin_settings, coin_overrides, params, run_settings,
        sizes, daily, scalars, gap_hist, coin_fill_counts, b, false
    );
}
