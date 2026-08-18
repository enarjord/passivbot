#include <metal_stdlib>
using namespace metal;

constant int MAX_COINS = 64;
constant int PARAM_COLS = 19;
constant int COIN_COLS = 11;
constant int OVERRIDE_COLS = 12;
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
    float position_open_k[MAX_COINS];
    float score[MAX_COINS];
    float contribution[MAX_COINS];
    float minimum_entry[MAX_COINS];
    int entry_tick[MAX_COINS];
    int close_tick[MAX_COINS];
    bool selected[MAX_COINS];
    bool survivor[MAX_COINS];
    bool entry_candidate[MAX_COINS];
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
        position_open_k[c] = -1.0f;
        score[c] = -INFINITY;
        contribution[c] = 0.0f;
        minimum_entry[c] = 0.0f;
        entry_tick[c] = 0;
        close_tick[c] = 0;
        selected[c] = false;
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

            bool filled_close = close_qty[c] > 0.0f && psize[c] > 0.0f
                && (short_side
                    ? close_tick[c] > fill_ticks[tick_offset + 1]
                    : close_tick[c] <= fill_ticks[tick_offset + 0]);
            if (filled_close) {
                float fill_price = float(close_tick[c]) * price_step;
                float adjusted = fmin(round_step(close_qty[c], qty_step), psize[c]);
                float pnl = adjusted * c_mult * (short_side
                    ? pprice[c] - fill_price
                    : fill_price - pprice[c]);
                balance += pnl - adjusted * fill_price * c_mult * maker_fee;
                float new_size = fmax(round_step(psize[c] - adjusted, qty_step), 0.0f);
                bool went_flat = new_size <= 0.0f;
                psize[c] = new_size;
                if (went_flat) {
                    pprice[c] = 0.0f;
                    if (position_open_k[c] >= 0.0f) {
                        held_max_min = fmax(held_max_min, float(k) - position_open_k[c]);
                    }
                    position_open_k[c] = -1.0f;
                }
                day_volume += fabs(adjusted) * fill_price * c_mult / balance;
                close_qty[c] = 0.0f;
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
                        && finite_positive(bars[bar_offset + 2]) && coin_wel != 0.0f;
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
                selection_initialized = true;
                previous_effective_n_positions = effective_n_positions;
            }

            const float effective_wel = twel / fmax(float(effective_n_positions), 1.0f);
            float current_twe = 0.0f;
            for (int c = 0; c < C; ++c) {
                int coin_offset = c * COIN_COLS;
                if (psize[c] > 0.0f && balance > 0.0f) {
                    current_twe += psize[c] * pprice[c]
                        * coin_settings[coin_offset + 4] / balance;
                }
            }
            for (int c = 0; c < C; ++c) {
                entry_qty[c] = 0.0f;
                close_qty[c] = 0.0f;
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
                bool tradable = k >= int(coin_settings[coin_offset + 8])
                    && k <= int(coin_settings[coin_offset + 7])
                    && finite_positive(price_now) && coin_wel > 0.0f;
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
                float position_cap = coin_wel - 1.0e-7f;
                float coin_base_qty_pct = coin_override_or(
                    coin_overrides, c, 0, base_qty_pct
                );
                float coin_ddf = coin_override_or(coin_overrides, c, 3, ddf);
                if (selected[c] && !cooldown && entry_price > 0.0f && balance > 0.0f
                    && cost_we < position_cap && coin_base_qty_pct > 0.0f) {
                    float base_qty = fmax(minimum, round_step(
                        balance * coin_wel * coin_base_qty_pct
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
                        balance * coin_wel * coin_base_qty_pct
                            / fmax(close_price * c_mult, 1.0e-12f),
                        qty_step
                    )));
                    close_qty[c] = psize[c] <= minimum_close
                            || psize[c] - clip < minimum_close
                        ? psize[c] : clip;
                    close_tick[c] = candidate_close_tick;
                }
            }

            float total_cap = twel - 1.0e-7f;
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
                        if (best < 0 || distance < best_distance
                            || (distance == best_distance && c < best)) {
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
    uint b [[thread_position_in_grid]]
) {
    const bool short_side = run_settings[3] > 0.5f;
    passivbot_ema_anchor_multicoin_impl(
        bars, fill_ticks, touch_ticks, coin_settings, coin_overrides, params, run_settings,
        sizes, daily, scalars, gap_hist, b, short_side
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
    uint b [[thread_position_in_grid]]
) {
    passivbot_ema_anchor_multicoin_impl(
        bars, fill_ticks, touch_ticks, coin_settings, coin_overrides, params, run_settings,
        sizes, daily, scalars, gap_hist, b, false
    );
}
