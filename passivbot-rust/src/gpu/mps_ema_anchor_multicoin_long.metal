#include <metal_stdlib>
using namespace metal;

constant int MAX_COINS = 64;
constant int PARAM_COLS = 42;
constant int COIN_COLS = 13;
constant int OVERRIDE_COLS = 30;
constant int HSL_OVERRIDE_START = 19;
constant int FORCED_ACTIVE_OVERRIDE_COL = 29;
#if PASSIVBOT_BTC_RISK_ENABLED
constant int DAILY_COLS = 12;
#else
constant int DAILY_COLS = 9;
#endif
#if PASSIVBOT_HSL_RAW_TAIL_ENABLED
constant int SCALAR_COLS = 67;
constant int FUSED_SCALAR_COLS = 72;
#elif PASSIVBOT_HSL_RAW_DRAWDOWN_ENABLED
constant int SCALAR_COLS = 65;
constant int FUSED_SCALAR_COLS = 70;
#elif PASSIVBOT_HSL_EMA_TAIL_ENABLED
constant int SCALAR_COLS = 63;
constant int FUSED_SCALAR_COLS = 68;
#else
constant int SCALAR_COLS = 61;
constant int FUSED_SCALAR_COLS = 66;
#endif
constant int GAP_BINS = 128;
#ifdef PASSIVBOT_STRATEGY_EQ_RECOVERY_DISTRIBUTION_ENABLED
constant float RECOVERY_FAIL_CLOSED_SENTINEL = -3.402823466e+38f;
#endif

// PASSIVBOT_HSL_COMMON

// PASSIVBOT_BTC_RISK_COMMON

// PASSIVBOT_EQUITY_BALANCE_DIFF_COMMON

// PASSIVBOT_ENTRY_INTERVAL_COMMON

// PASSIVBOT_MULTICOIN_COMMON

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

// One complete directional EMA portfolio. Keeping the mutable per-coin state
// behind one thread-local value lets a future fused kernel own long and short
// portfolios concurrently without changing the proven one-side candle loop.
struct EmaMulticoinSideState {
    HslState hsl;
    HslState coin_hsl[MAX_COINS];
    HslStrategyEquityStats hsl_strategy_eq;
#if PASSIVBOT_HSL_EMA_TAIL_ENABLED
    HslDrawdownEmaTailStats hsl_ema_tail;
#endif
    ulong coin_hsl_entry_blocked_mask;
    ulong one_way_initial_blocked_mask;
    ulong candle_eligibility_mask;
    float ema0[MAX_COINS];
    float ema1[MAX_COINS];
    float ema2[MAX_COINS];
    float volatility_1m[MAX_COINS];
    float volatility_1h[MAX_COINS];
    float forager_volume[MAX_COINS];
    float forager_volatility[MAX_COINS];
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
    bool entry_market[MAX_COINS];
    bool close_market[MAX_COINS];
    bool secondary_close_market[MAX_COINS];
    bool close_is_protective_reducer[MAX_COINS];
    bool close_is_unstuck_reducer[MAX_COINS];
    bool close_is_hsl_panic[MAX_COINS];
    bool selected[MAX_COINS];
    bool incumbent[MAX_COINS];
    bool survivor[MAX_COINS];
    bool entry_candidate[MAX_COINS];
    bool selection_initialized;
    int max_tradable_seen;
    int previous_effective_n_positions;
    float alpha0_coin[MAX_COINS];
    float alpha1_coin[MAX_COINS];
    float alpha2_coin[MAX_COINS];
    float alpha_1h_coin[MAX_COINS];
    float alpha_1m_coin[MAX_COINS];
    float coin_realized_pnl[MAX_COINS];
};

struct EmaMulticoinSideConfig {
    float base_qty_pct;
    float span_a;
    float span_b;
    float alpha0;
    float alpha1;
    float alpha2;
    float ddf;
    float offset;
    float psize_weight;
    float weight_1h;
    float weight_1m;
    float span_1h;
    float span_1m;
    float alpha_1h;
    float alpha_1m;
    float cooldown_min;
    float twel;
    float forager_volume_span;
    float forager_volatility_span;
    float volume_drop;
    float w_volume;
    float w_ready;
    float w_volatility;
    int n_positions;
    float allowance_pct;
    bool legacy_raw_allowance;
    bool twel_entry_gate_enabled;
    float twel_threshold;
    bool twel_enforcer_enabled;
    bool twel_enforcer_reduce_portfolio;
    bool unstuck_enabled;
    bool unstuck_ema_gating_enabled;
    float unstuck_close_pct;
    float unstuck_ema_dist;
    float unstuck_loss_allowance_pct;
    float unstuck_threshold;
    float alpha_forager_volume;
    float alpha_forager_volatility;
    HslState hsl_template;
    bool coin_hsl_mode;
};

// Shared fill accounting is separate from strategy-side state so a fused
// long+short kernel can apply both directional fill passes to one account and
// one chronology without reconstructing totals from directional summaries.
struct EmaMulticoinFillState {
    float pnl_recovery_peak;
    float pnl_recovery_peak_k;
    float pnl_recovery_max_min;
    float profit_sum;
    float loss_sum;
    float profit_sum_long;
    float loss_sum_long;
    float profit_sum_short;
    float loss_sum_short;
    float fill_count;
    float fill_count_entry;
    float fill_count_long;
    float held_max_min;
    float held_sum_min;
    float held_count;
    float position_unchanged_max_min;
    float day_volume;
    float day_fill_count;
};

inline EmaMulticoinFillState init_ema_multicoin_fill_state() {
    EmaMulticoinFillState fills;
    fills.pnl_recovery_peak = -INFINITY;
    fills.pnl_recovery_peak_k = -1.0f;
    fills.pnl_recovery_max_min = 0.0f;
    fills.profit_sum = 0.0f;
    fills.loss_sum = 0.0f;
    fills.profit_sum_long = 0.0f;
    fills.loss_sum_long = 0.0f;
    fills.profit_sum_short = 0.0f;
    fills.loss_sum_short = 0.0f;
    fills.fill_count = 0.0f;
    fills.fill_count_entry = 0.0f;
    fills.fill_count_long = 0.0f;
    fills.held_max_min = 0.0f;
    fills.held_sum_min = 0.0f;
    fills.held_count = 0.0f;
    fills.position_unchanged_max_min = 0.0f;
    fills.day_volume = 0.0f;
    fills.day_fill_count = 0.0f;
    return fills;
}

inline void record_ema_multicoin_gross_pnl(
    float pnl,
    thread EmaMulticoinFillState& fills,
    bool short_side
) {
    record_gross_pnl(pnl, fills.profit_sum, fills.loss_sum);
    if (short_side) {
        record_gross_pnl(
            pnl, fills.profit_sum_short, fills.loss_sum_short
        );
    } else {
        record_gross_pnl(
            pnl, fills.profit_sum_long, fills.loss_sum_long
        );
    }
}

inline void record_ema_multicoin_close_fill(
    thread EmaMulticoinSideState& side,
    thread JointPortfolioAccount& account,
    thread EmaMulticoinFillState& fills,
    device float* coin_fill_counts,
    int candidate_index,
    int coin_count,
    int coin,
    int k,
    float gross_pnl,
    float net_pnl,
    float qty,
    float position_price,
    float mark_price,
    float c_mult,
    bool short_side,
    bool is_hsl_panic,
    bool collect_coin_fill_counts,
    thread float& hsl_equity_before_fills
) {
    const bool coin_hsl_mode =
        side.hsl.signal_mode == HSL_SIGNAL_COIN;
    if (is_hsl_panic) {
        if (coin_hsl_mode) {
            record_hsl_panic_fill(
                side.coin_hsl[coin], net_pnl, hsl_equity_before_fills
            );
        } else {
            record_hsl_panic_fill(
                side.hsl, net_pnl, hsl_equity_before_fills
            );
        }
    }
    record_ema_multicoin_gross_pnl(gross_pnl, fills, short_side);
    record_realized_net(
        net_pnl, account,
        fills.day_fill_count, fills.fill_count,
        fills.fill_count_entry, fills.fill_count_long,
        fills.pnl_recovery_peak, fills.pnl_recovery_peak_k,
        fills.pnl_recovery_max_min, float(k), false, !short_side
    );
    side.coin_realized_pnl[coin] += net_pnl;
    if (coin_hsl_mode) {
        record_coin_hsl_realized_fill(
            side.coin_hsl[coin], side.coin_realized_pnl[coin]
        );
        advance_coin_hsl_equity_after_close_fill(
            hsl_equity_before_fills,
            net_pnl, qty, position_price, mark_price,
            c_mult, short_side
        );
    }
    if (collect_coin_fill_counts) {
        coin_fill_counts[candidate_index * coin_count + coin] += 1.0f;
    }
}

inline EmaMulticoinSideConfig load_ema_multicoin_side_config(
    constant float* params,
    int po
) {
    EmaMulticoinSideConfig config;
    config.base_qty_pct = params[po + 0];
    config.span_a = params[po + 1];
    config.span_b = params[po + 2];
    float span_c = sqrt(fmax(config.span_a * config.span_b, 1.0f));
    float span_lo = fmin(config.span_a, fmin(config.span_b, span_c));
    float span_hi = fmax(config.span_a, fmax(config.span_b, span_c));
    float span_mid = config.span_a + config.span_b + span_c - span_lo - span_hi;
    config.alpha0 = clamp(2.0f / (span_lo + 1.0f), 0.0f, 1.0f);
    config.alpha1 = clamp(2.0f / (span_mid + 1.0f), 0.0f, 1.0f);
    config.alpha2 = clamp(2.0f / (span_hi + 1.0f), 0.0f, 1.0f);
    config.ddf = params[po + 3];
    config.offset = params[po + 4];
    config.psize_weight = params[po + 5];
    config.weight_1h = params[po + 6];
    config.weight_1m = params[po + 7];
    config.span_1h = params[po + 8];
    config.span_1m = params[po + 9];
    config.alpha_1h = config.span_1h > 0.0f
        ? 2.0f / (fmax(config.span_1h, 1.0f) + 1.0f) : 0.0f;
    config.alpha_1m = config.span_1m > 0.0f
        ? clamp(2.0f / (config.span_1m + 1.0f), 0.0f, 1.0f) : 0.0f;
    config.cooldown_min = ceil(params[po + 10]);
    config.twel = params[po + 11];
    config.forager_volume_span = params[po + 12];
    config.forager_volatility_span = params[po + 13];
    config.volume_drop = clamp(params[po + 14], 0.0f, 1.0f);
    config.w_volume = params[po + 15];
    config.w_ready = params[po + 16];
    config.w_volatility = params[po + 17];
    float weight_sum = config.w_volume + config.w_ready + config.w_volatility;
    if (weight_sum > 0.0f) {
        config.w_volume /= weight_sum;
        config.w_ready /= weight_sum;
        config.w_volatility /= weight_sum;
    } else {
        config.w_volume = 0.0f;
        config.w_ready = 1.0f;
        config.w_volatility = 0.0f;
    }
    config.n_positions = max(1, int(rint(params[po + 18])));
    config.allowance_pct = params[po + 19];
    config.legacy_raw_allowance = params[po + 20] > 0.5f;
    config.twel_entry_gate_enabled = params[po + 21] > 0.5f;
    config.twel_threshold = params[po + 22];
    config.twel_enforcer_enabled = params[po + 23] > 0.5f;
    config.twel_enforcer_reduce_portfolio = params[po + 24] > 0.5f;
    config.unstuck_enabled = params[po + 25] > 0.5f;
    config.unstuck_ema_gating_enabled = params[po + 26] > 0.5f;
    config.unstuck_close_pct = params[po + 27];
    config.unstuck_ema_dist = params[po + 28];
    config.unstuck_loss_allowance_pct = params[po + 29];
    config.unstuck_threshold = params[po + 30];
    config.alpha_forager_volume = config.forager_volume_span > 0.0f
        ? clamp(2.0f / (config.forager_volume_span + 1.0f), 0.0f, 1.0f)
        : 0.0f;
    config.alpha_forager_volatility = config.forager_volatility_span > 0.0f
        ? clamp(2.0f / (config.forager_volatility_span + 1.0f), 0.0f, 1.0f)
        : 0.0f;
    config.hsl_template = load_hsl(params, po, 31);
    config.coin_hsl_mode = config.hsl_template.signal_mode == HSL_SIGNAL_COIN;
    return config;
}

inline void init_ema_multicoin_side_state(
    thread EmaMulticoinSideState& side,
    thread const EmaMulticoinSideConfig& config,
    constant float* coin_settings,
    constant float* coin_overrides,
    int coin_count
) {
    side.hsl = config.hsl_template;
    side.hsl_strategy_eq = init_hsl_strategy_equity_stats();
#if PASSIVBOT_HSL_EMA_TAIL_ENABLED
    side.hsl_ema_tail = init_hsl_drawdown_ema_tail_stats();
#endif
    side.coin_hsl_entry_blocked_mask = 0ul;
    side.one_way_initial_blocked_mask = 0ul;
    side.candle_eligibility_mask = 0ul;
    side.selection_initialized = false;
    side.max_tradable_seen = 0;
    side.previous_effective_n_positions = 0;
    for (int c = 0; c < MAX_COINS; ++c) {
        float seed_close = c < coin_count ? coin_settings[c * COIN_COLS + 9] : 0.0f;
        float seed_volume = c < coin_count ? coin_settings[c * COIN_COLS + 10] : 0.0f;
        side.ema0[c] = seed_close;
        side.ema1[c] = seed_close;
        side.ema2[c] = seed_close;
        side.volatility_1m[c] = 0.0f;
        side.volatility_1h[c] = 0.0f;
        side.forager_volume[c] = seed_volume;
        side.forager_volatility[c] = 0.0f;
        side.psize[c] = 0.0f;
        side.pprice[c] = 0.0f;
        side.last_increase_k[c] = -1.0e20f;
        side.entry_qty[c] = 0.0f;
        side.close_qty[c] = 0.0f;
        side.secondary_close_qty[c] = 0.0f;
        side.twel_close_qty[c] = 0.0f;
        side.unstuck_close_qty[c] = 0.0f;
        side.position_open_k[c] = -1.0f;
        side.position_last_fill_k[c] = -1.0f;
        side.score[c] = -INFINITY;
        side.contribution[c] = 0.0f;
        side.minimum_entry[c] = 0.0f;
        side.entry_tick[c] = 0;
        side.close_tick[c] = 0;
        side.secondary_close_tick[c] = 0;
        side.twel_close_tick[c] = 0;
        side.unstuck_close_tick[c] = 0;
        side.entry_market[c] = false;
        side.close_market[c] = false;
        side.secondary_close_market[c] = false;
        side.close_is_protective_reducer[c] = false;
        side.close_is_unstuck_reducer[c] = false;
        side.close_is_hsl_panic[c] = false;
        side.coin_realized_pnl[c] = 0.0f;
        side.coin_hsl[c] = config.hsl_template;
        if (config.coin_hsl_mode && c < coin_count) {
            apply_coin_hsl_overrides(
                side.coin_hsl[c], coin_overrides, c,
                OVERRIDE_COLS, HSL_OVERRIDE_START
            );
        } else {
            side.coin_hsl[c].enabled = false;
        }
        side.selected[c] = false;
        side.incumbent[c] = false;
        side.survivor[c] = false;
        side.entry_candidate[c] = false;
        if (c < coin_count) {
            float coin_span_a = coin_override_or(
                coin_overrides, c, 1, config.span_a
            );
            float coin_span_b = coin_override_or(
                coin_overrides, c, 2, config.span_b
            );
            float coin_span_c = sqrt(fmax(coin_span_a * coin_span_b, 1.0f));
            float coin_span_lo = fmin(coin_span_a, fmin(coin_span_b, coin_span_c));
            float coin_span_hi = fmax(coin_span_a, fmax(coin_span_b, coin_span_c));
            float coin_span_mid = coin_span_a + coin_span_b + coin_span_c
                - coin_span_lo - coin_span_hi;
            side.alpha0_coin[c] = clamp(
                2.0f / (coin_span_lo + 1.0f), 0.0f, 1.0f
            );
            side.alpha1_coin[c] = clamp(
                2.0f / (coin_span_mid + 1.0f), 0.0f, 1.0f
            );
            side.alpha2_coin[c] = clamp(
                2.0f / (coin_span_hi + 1.0f), 0.0f, 1.0f
            );
            float coin_span_1h = coin_override_or(
                coin_overrides, c, 8, config.span_1h
            );
            float coin_span_1m = coin_override_or(
                coin_overrides, c, 9, config.span_1m
            );
            side.alpha_1h_coin[c] = coin_span_1h > 0.0f
                ? 2.0f / (fmax(coin_span_1h, 1.0f) + 1.0f) : 0.0f;
            side.alpha_1m_coin[c] = coin_span_1m > 0.0f
                ? clamp(2.0f / (coin_span_1m + 1.0f), 0.0f, 1.0f) : 0.0f;
        } else {
            side.alpha0_coin[c] = config.alpha0;
            side.alpha1_coin[c] = config.alpha1;
            side.alpha2_coin[c] = config.alpha2;
            side.alpha_1h_coin[c] = config.alpha_1h;
            side.alpha_1m_coin[c] = config.alpha_1m;
        }
    }
}

inline float accumulate_ema_multicoin_side_unrealized_pnl(
    thread const EmaMulticoinSideState& side,
    constant float* bars,
    constant float* coin_settings,
    int k,
    int coin_count,
    bool short_side,
    float accumulator
) {
    for (int c = 0; c < coin_count; ++c) {
        int coin_offset = c * COIN_COLS;
        int bar_offset = (k * coin_count + c) * 4;
        float close = bars[bar_offset + 2];
        bool valid = k >= int(coin_settings[coin_offset + 6])
            && k <= int(coin_settings[coin_offset + 7])
            && isfinite(close);
        if (side.psize[c] > 0.0f && valid) {
            accumulator += side.psize[c]
                * coin_settings[coin_offset + 4]
                * (short_side
                    ? side.pprice[c] - close
                    : close - side.pprice[c]);
        }
    }
    return accumulator;
}

inline void update_ema_multicoin_side_indicators(
    thread EmaMulticoinSideState& side,
    thread const EmaMulticoinSideConfig& config,
    constant float* bars,
    constant float* hour_log_ranges,
    constant float* coin_settings,
    int k,
    int coin_count
) {
    for (int c = 0; c < coin_count; ++c) {
        const int coin_offset = c * COIN_COLS;
        const int bar_offset = (k * coin_count + c) * 4;
        const float high = bars[bar_offset + 0];
        const float low = bars[bar_offset + 1];
        const float close = bars[bar_offset + 2];
        const float volume = bars[bar_offset + 3];
        const int first_valid = int(coin_settings[coin_offset + 6]);
        const int last_valid = int(coin_settings[coin_offset + 7]);
        const bool valid = k >= first_valid && k <= last_valid
            && finite_positive(high) && finite_positive(low)
            && finite_positive(close);
        const float hour_range = hour_log_ranges[k * coin_count + c];
        if (hour_range >= 0.0f && side.alpha_1h_coin[c] > 0.0f) {
            side.volatility_1h[c] = fma(
                side.alpha_1h_coin[c],
                hour_range - side.volatility_1h[c],
                side.volatility_1h[c]
            );
        }
        if (!valid) continue;
        float log_range = log(high / low);
        side.ema0[c] = fma(
            side.alpha0_coin[c], close - side.ema0[c], side.ema0[c]
        );
        side.ema1[c] = fma(
            side.alpha1_coin[c], close - side.ema1[c], side.ema1[c]
        );
        side.ema2[c] = fma(
            side.alpha2_coin[c], close - side.ema2[c], side.ema2[c]
        );
        if (side.alpha_1m_coin[c] > 0.0f) {
            side.volatility_1m[c] = fma(
                side.alpha_1m_coin[c],
                log_range - side.volatility_1m[c],
                side.volatility_1m[c]
            );
        }
        if (config.alpha_forager_volatility > 0.0f) {
            side.forager_volatility[c] = fma(
                config.alpha_forager_volatility,
                log_range - side.forager_volatility[c],
                side.forager_volatility[c]
            );
        }
        if (config.alpha_forager_volume > 0.0f) {
            float typical = (high + low + close) / 3.0f;
            float quote_volume = fmax(volume, 0.0f) * typical;
            side.forager_volume[c] = fma(
                config.alpha_forager_volume,
                quote_volume - side.forager_volume[c],
                side.forager_volume[c]
            );
        }
    }
}

inline int count_ema_multicoin_tradable_coins(
    constant float* bars,
    constant float* coin_settings,
    constant float* coin_overrides,
    int k,
    int coin_count
) {
    int tradable_count = 0;
    for (int c = 0; c < coin_count; ++c) {
        int coin_offset = c * COIN_COLS;
        int bar_offset = (k * coin_count + c) * 4;
        float close = bars[bar_offset + 2];
        float coin_wel = coin_override_or(coin_overrides, c, 11, -1.0f);
        if (k >= int(coin_settings[coin_offset + 8])
            && k <= int(coin_settings[coin_offset + 7])
            && finite_positive(close) && coin_wel != 0.0f) {
            tradable_count += 1;
        }
    }
    return tradable_count;
}

inline bool ema_multicoin_side_has_position(
    thread const EmaMulticoinSideState& side,
    int coin_count
) {
    for (int c = 0; c < coin_count; ++c) {
        if (side.psize[c] > 0.0f) return true;
    }
    return false;
}

inline bool ema_multicoin_side_held_marks_are_valid(
    thread const EmaMulticoinSideState& side,
    constant float* bars,
    constant float* coin_settings,
    int k,
    int coin_count
) {
    for (int c = 0; c < coin_count; ++c) {
        if (!(side.psize[c] > 0.0f)) continue;
        int coin_offset = c * COIN_COLS;
        int bar_offset = (k * coin_count + c) * 4;
        if (!finite_positive(side.pprice[c])
            || !finite_positive(coin_settings[coin_offset + 4])) {
            return false;
        }
        if (k > int(coin_settings[coin_offset + 7])) continue;
        if (!isfinite(bars[bar_offset + 2])) return false;
    }
    return true;
}

inline bool ema_multicoin_side_has_blocking_orders(
    thread EmaMulticoinSideState& side,
    thread const EmaMulticoinSideConfig& config,
    constant float* bars,
    constant float* coin_settings,
    int k,
    int coin_count
) {
    bool side_has_position = config.coin_hsl_mode
        ? false : ema_multicoin_side_has_position(side, coin_count);
    int side_mode = config.coin_hsl_mode
        ? 0 : hsl_mode(side.hsl, side_has_position);
    for (int c = 0; c < coin_count; ++c) {
        int coin_offset = c * COIN_COLS;
        int bar_offset = (k * coin_count + c) * 4;
        bool valid = k >= int(coin_settings[coin_offset + 6])
            && k <= int(coin_settings[coin_offset + 7])
            && finite_positive(bars[bar_offset + 2]);
        if (!valid) continue;
        int mode = config.coin_hsl_mode
            ? hsl_mode(side.coin_hsl[c], side.psize[c] > 0.0f)
            : side_mode;
        if (mode != 3 && (
                side.entry_qty[c] > 0.0f
                    || side.close_qty[c] > 0.0f
                    || side.secondary_close_qty[c] > 0.0f
            )) {
            return true;
        }
    }
    return false;
}

// Advance the complete dual-side HSL topology after both directional order
// phases have run against the same account. Unified and pside modes use the
// shared account controller. Coin mode keeps one controller per coin/pside,
// while its sampled tier is the maximum across the full portfolio, matching
// exact Rust reporting. Mixed signal modes fail closed.
inline bool update_ema_multicoin_dual_side_hsl(
    thread EmaMulticoinSideState& long_side,
    thread const EmaMulticoinSideConfig& long_config,
    int long_effective_n_positions,
    thread EmaMulticoinSideState& short_side,
    thread const EmaMulticoinSideConfig& short_config,
    int short_effective_n_positions,
    thread const JointPortfolioAccount& account,
    constant float* bars,
    constant float* coin_settings,
    int k,
    int day_index,
    int coin_count,
    float starting_balance,
    float interval_ms,
    thread bool& sample_enabled,
    thread int& sampled_tier
) {
    sample_enabled = false;
    sampled_tier = 0;
    if (long_side.hsl.signal_mode != short_side.hsl.signal_mode) return false;
    if (long_config.coin_hsl_mode != short_config.coin_hsl_mode) return false;
    if (!ema_multicoin_side_held_marks_are_valid(
            long_side, bars, coin_settings, k, coin_count
        ) || !ema_multicoin_side_held_marks_are_valid(
            short_side, bars, coin_settings, k, coin_count
        )) {
        return false;
    }

    float long_unrealized = accumulate_ema_multicoin_side_unrealized_pnl(
        long_side, bars, coin_settings, k, coin_count, false, 0.0f
    );
    float short_unrealized = accumulate_ema_multicoin_side_unrealized_pnl(
        short_side, bars, coin_settings, k, coin_count, true, 0.0f
    );
    bool long_has_position = ema_multicoin_side_has_position(
        long_side, coin_count
    );
    bool short_has_position = ema_multicoin_side_has_position(
        short_side, coin_count
    );
    bool long_has_blocking_orders = ema_multicoin_side_has_blocking_orders(
        long_side, long_config, bars, coin_settings, k, coin_count
    );
    bool short_has_blocking_orders = ema_multicoin_side_has_blocking_orders(
        short_side, short_config, bars, coin_settings, k, coin_count
    );

    if (long_config.coin_hsl_mode) {
        const bool long_active = long_effective_n_positions > 0;
        const bool short_active = short_effective_n_positions > 0;
        bool long_strategy_eq_enabled = false;
        bool short_strategy_eq_enabled = false;
#if PASSIVBOT_HSL_EMA_TAIL_ENABLED
        float long_drawdown_ema_sample = 0.0f;
        float short_drawdown_ema_sample = 0.0f;
#endif
        float portfolio_equity = joint_portfolio_equity(
            account, long_unrealized, short_unrealized
        );
        for (int c = 0; c < coin_count; ++c) {
            int coin_offset = c * COIN_COLS;
            int bar_offset = (k * coin_count + c) * 4;
            float close = bars[bar_offset + 2];
            float c_mult = coin_settings[coin_offset + 4];
            bool valid = k >= int(coin_settings[coin_offset + 6])
                && k <= int(coin_settings[coin_offset + 7])
                && finite_positive(close);
            float long_coin_unrealized = long_side.psize[c] > 0.0f
                    && valid
                ? long_side.psize[c] * c_mult
                    * (close - long_side.pprice[c])
                : 0.0f;
            float short_coin_unrealized = short_side.psize[c] > 0.0f
                    && valid
                ? short_side.psize[c] * c_mult
                    * (short_side.pprice[c] - close)
                : 0.0f;
            int long_mode = hsl_mode(
                long_side.coin_hsl[c], long_side.psize[c] > 0.0f
            );
            int short_mode = hsl_mode(
                short_side.coin_hsl[c], short_side.psize[c] > 0.0f
            );
            bool long_coin_has_blocking_orders = valid && long_mode != 3 && (
                long_side.entry_qty[c] > 0.0f
                    || long_side.close_qty[c] > 0.0f
                    || long_side.secondary_close_qty[c] > 0.0f
            );
            bool short_coin_has_blocking_orders = valid && short_mode != 3 && (
                short_side.entry_qty[c] > 0.0f
                    || short_side.close_qty[c] > 0.0f
                    || short_side.secondary_close_qty[c] > 0.0f
            );
            if (long_active) {
                long_side.coin_hsl[c].slot_count = float(
                    long_effective_n_positions
                );
                update_hsl(
                    long_side.coin_hsl[c], account.balance, starting_balance,
                    long_side.coin_realized_pnl[c], long_coin_unrealized,
                    long_side.psize[c] > 0.0f,
                    long_coin_has_blocking_orders, float(k), interval_ms
                );
                sample_enabled = sample_enabled
                    || long_side.coin_hsl[c].enabled;
                long_strategy_eq_enabled = long_strategy_eq_enabled
                    || long_side.coin_hsl[c].enabled;
#if PASSIVBOT_HSL_EMA_TAIL_ENABLED
                if (long_side.coin_hsl[c].enabled) {
                    long_drawdown_ema_sample = fmax(
                        long_drawdown_ema_sample,
                        fabs(long_side.coin_hsl[c].drawdown_ema)
                    );
                }
#endif
                sampled_tier = max(
                    sampled_tier, long_side.coin_hsl[c].tier
                );
                try_restart_hsl(
                    long_side.coin_hsl[c], float(k), portfolio_equity
                );
            }
            if (short_active) {
                short_side.coin_hsl[c].slot_count = float(
                    short_effective_n_positions
                );
                update_hsl(
                    short_side.coin_hsl[c], account.balance, starting_balance,
                    short_side.coin_realized_pnl[c], short_coin_unrealized,
                    short_side.psize[c] > 0.0f,
                    short_coin_has_blocking_orders, float(k), interval_ms
                );
                sample_enabled = sample_enabled
                    || short_side.coin_hsl[c].enabled;
                short_strategy_eq_enabled = short_strategy_eq_enabled
                    || short_side.coin_hsl[c].enabled;
#if PASSIVBOT_HSL_EMA_TAIL_ENABLED
                if (short_side.coin_hsl[c].enabled) {
                    short_drawdown_ema_sample = fmax(
                        short_drawdown_ema_sample,
                        fabs(short_side.coin_hsl[c].drawdown_ema)
                    );
                }
#endif
                sampled_tier = max(
                    sampled_tier, short_side.coin_hsl[c].tier
                );
                try_restart_hsl(
                    short_side.coin_hsl[c], float(k), portfolio_equity
                );
            }
        }
        if (long_strategy_eq_enabled) {
#if PASSIVBOT_HSL_EMA_TAIL_ENABLED
            update_hsl_drawdown_ema_tail_stats(
                long_side.hsl_ema_tail, long_drawdown_ema_sample
            );
#endif
            update_hsl_strategy_equity_stats(
                long_side.hsl_strategy_eq,
                starting_balance + account.realized_pnl_long + long_unrealized,
                day_index
            );
        }
        if (short_strategy_eq_enabled) {
#if PASSIVBOT_HSL_EMA_TAIL_ENABLED
            update_hsl_drawdown_ema_tail_stats(
                short_side.hsl_ema_tail, short_drawdown_ema_sample
            );
#endif
            update_hsl_strategy_equity_stats(
                short_side.hsl_strategy_eq,
                starting_balance + account.realized_pnl_short + short_unrealized,
                day_index
            );
        }
        return true;
    }

    const bool unified = long_side.hsl.signal_mode == HSL_SIGNAL_UNIFIED;
    const bool long_strategy_eq_enabled = long_side.hsl.enabled
        && !long_side.hsl.halted;
    const bool short_strategy_eq_enabled = short_side.hsl.enabled
        && !short_side.hsl.halted;
    if (long_strategy_eq_enabled) {
        update_hsl_strategy_equity_stats(
            long_side.hsl_strategy_eq,
            starting_balance + (
                unified ? account.realized_pnl_total : account.realized_pnl_long
            ) + (unified ? long_unrealized + short_unrealized : long_unrealized),
            day_index
        );
    }
    if (short_strategy_eq_enabled) {
        update_hsl_strategy_equity_stats(
            short_side.hsl_strategy_eq,
            starting_balance + (
                unified ? account.realized_pnl_total : account.realized_pnl_short
            ) + (unified ? long_unrealized + short_unrealized : short_unrealized),
            day_index
        );
    }
    if (!update_joint_pside_hsl(
            long_side.hsl, short_side.hsl, account, starting_balance,
            long_unrealized, short_unrealized,
            long_has_position, short_has_position,
            long_has_blocking_orders, short_has_blocking_orders,
            float(k), interval_ms
        )) {
        return false;
    }
#if PASSIVBOT_HSL_EMA_TAIL_ENABLED
    if (long_strategy_eq_enabled) {
        update_hsl_drawdown_ema_tail_stats(
            long_side.hsl_ema_tail, long_side.hsl.drawdown_ema
        );
    }
    if (short_strategy_eq_enabled) {
        update_hsl_drawdown_ema_tail_stats(
            short_side.hsl_ema_tail, short_side.hsl.drawdown_ema
        );
    }
#endif
    sample_enabled = long_side.hsl.enabled || short_side.hsl.enabled;
    sampled_tier = joint_pside_hsl_global_tier(
        long_side.hsl, short_side.hsl
    );
    try_restart_joint_pside_hsl(
        long_side.hsl, short_side.hsl, account, starting_balance,
        long_unrealized, short_unrealized, float(k)
    );
    return true;
}

inline void update_ema_multicoin_side_selection(
    thread EmaMulticoinSideState& side,
    thread const EmaMulticoinSideConfig& config,
    constant float* bars,
    constant float* coin_settings,
    constant float* coin_overrides,
    int k,
    int coin_count,
    bool short_side,
    bool any_fill,
    int effective_n_positions,
    float score_hysteresis,
    ulong one_way_initial_blocked_mask,
    bool filter_by_min_effective_cost,
    float guaranteed_balance_lower
) {
    thread HslState* coin_hsl = side.coin_hsl;
    thread ulong& coin_hsl_entry_blocked_mask =
        side.coin_hsl_entry_blocked_mask;
    thread float* ema0 = side.ema0;
    thread float* ema1 = side.ema1;
    thread float* ema2 = side.ema2;
    thread float* forager_volume = side.forager_volume;
    thread float* forager_volatility = side.forager_volatility;
    thread float* psize = side.psize;
    thread float* score = side.score;
    thread bool* selected = side.selected;
    thread bool* incumbent = side.incumbent;
    thread bool* survivor = side.survivor;

    // Exact Rust ranks flat candidates every minute. Re-ranking only after
    // state changes keeps the proxy inexpensive; independent exact
    // validations and drift gates police this approximation.
    bool coin_hsl_eligibility_changed = false;
    if (config.coin_hsl_mode) {
        ulong blocked_mask = 0ul;
        for (int c = 0; c < coin_count; ++c) {
            if (hsl_mode(coin_hsl[c], false) != 0) {
                blocked_mask |= 1ul << ulong(c);
            }
        }
        coin_hsl_eligibility_changed =
            blocked_mask != coin_hsl_entry_blocked_mask;
        coin_hsl_entry_blocked_mask = blocked_mask;
    }
    bool one_way_eligibility_changed = one_way_initial_blocked_mask
        != side.one_way_initial_blocked_mask;
    side.one_way_initial_blocked_mask = one_way_initial_blocked_mask;
    ulong candle_eligibility_mask = 0ul;
    int current_tradable_count = 0;
    bool flat_selected_became_ineligible = false;
    for (int c = 0; c < coin_count; ++c) {
        int coin_offset = c * COIN_COLS;
        int bar_offset = (k * coin_count + c) * 4;
        bool eligible_now = k >= int(coin_settings[coin_offset + 8])
            && k <= int(coin_settings[coin_offset + 7])
            && finite_positive(bars[bar_offset + 2]);
        if (eligible_now) {
            candle_eligibility_mask |= 1ul << ulong(c);
            if (coin_override_or(coin_overrides, c, 11, -1.0f) != 0.0f) {
                current_tradable_count += 1;
            }
        } else if (selected[c] && psize[c] <= 0.0f) {
            flat_selected_became_ineligible = true;
        }
    }
    bool candle_eligibility_changed = side.selection_initialized
        && candle_eligibility_mask != side.candle_eligibility_mask;
    side.candle_eligibility_mask = candle_eligibility_mask;
    bool reselect = !side.selection_initialized || any_fill
        || coin_hsl_eligibility_changed
        || one_way_eligibility_changed
        || candle_eligibility_changed
        || flat_selected_became_ineligible
        || effective_n_positions != side.previous_effective_n_positions;
    if (!reselect) return;

    int active_count = 0;
    for (int c = 0; c < coin_count; ++c) {
        incumbent[c] = selected[c] && psize[c] <= 0.0f;
        selected[c] = psize[c] > 0.0f;
        if (selected[c]) active_count += 1;
        survivor[c] = false;
    }
    int enabled_count = 0;
    int forced_normal_count = 0;
    for (int c = 0; c < coin_count; ++c) {
        int coin_offset = c * COIN_COLS;
        int bar_offset = (k * coin_count + c) * 4;
        float coin_wel = coin_override_or(coin_overrides, c, 11, -1.0f);
        float base_limit = coin_wel >= 0.0f
            ? coin_wel : config.twel / fmax(float(effective_n_positions), 1.0f);
        float allowance_pct = coin_override_or(
            coin_overrides, c, 12, config.allowance_pct
        );
        float allowed_wel = allowed_wallet_exposure_limit(
            base_limit, config.twel, allowance_pct,
            config.legacy_raw_allowance
        );
        float initial_qty_pct = coin_override_or(
            coin_overrides, c, 0, config.base_qty_pct
        );
        bool min_cost_eligible = passes_multicoin_min_effective_cost(
            filter_by_min_effective_cost, guaranteed_balance_lower,
            allowed_wel, initial_qty_pct, coin_settings[coin_offset + 12]
        );
        bool base_eligible = k >= int(coin_settings[coin_offset + 8])
            && k <= int(coin_settings[coin_offset + 7])
            && finite_positive(bars[bar_offset + 2])
            && coin_wel != 0.0f
            && (psize[c] > 0.0f || min_cost_eligible)
            && (!config.coin_hsl_mode || (
                coin_hsl_entry_blocked_mask & (1ul << ulong(c))
            ) == 0ul);
        bool forced_normal = coin_override_or(
            coin_overrides, c, FORCED_ACTIVE_OVERRIDE_COL, 0.0f
        ) > 0.5f;
        if (base_eligible && forced_normal) forced_normal_count += 1;
        bool enabled = !selected[c]
            && base_eligible
            && (one_way_initial_blocked_mask & (1ul << ulong(c))) == 0ul;
        survivor[c] = enabled;
        if (enabled) enabled_count += 1;
    }
    // Exact Rust expands the active-set cap to fit eligible forced-normal
    // symbols but retains the separate dynamic-WEL denominator.
    int selection_n_positions = max(
        min(config.n_positions, current_tradable_count), forced_normal_count
    );
    int slots = max(selection_n_positions - active_count, 0);
    for (int c = 0; c < coin_count && slots > 0; ++c) {
        bool forced_normal = coin_override_or(
            coin_overrides, c, FORCED_ACTIVE_OVERRIDE_COL, 0.0f
        ) > 0.5f;
        if (!survivor[c] || !forced_normal) continue;
        selected[c] = true;
        survivor[c] = false;
        enabled_count -= 1;
        slots -= 1;
    }
    int keep = int(floor(
        float(enabled_count) * (1.0f - config.volume_drop) + 0.5f
    ));
    keep = min(
        enabled_count,
        max(max(keep, slots), enabled_count > 0 ? 1 : 0)
    );
    if (keep < enabled_count) {
        for (int c = 0; c < coin_count; ++c) {
            if (!survivor[c]) continue;
            int better = 0;
            for (int j = 0; j < coin_count; ++j) {
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
    for (int c = 0; c < coin_count; ++c) {
        if (!survivor[c]) continue;
        int bar_offset = (k * coin_count + c) * 4;
        float close = bars[bar_offset + 2];
        float lower = fmin(ema0[c], fmin(ema1[c], ema2[c]));
        float upper = fmax(ema0[c], fmax(ema1[c], ema2[c]));
        float coin_offset_pct = coin_override_or(
            coin_overrides, c, 4, config.offset
        );
        float threshold = short_side
            ? upper * (1.0f + coin_offset_pct)
            : lower * (1.0f - coin_offset_pct);
        float readiness = threshold > 0.0f
            ? (short_side
                ? 1.0f - close / threshold
                : close / threshold - 1.0f)
            : INFINITY;
        volume_min = fmin(volume_min, forager_volume[c]);
        volume_max = fmax(volume_max, forager_volume[c]);
        ready_min = fmin(ready_min, readiness);
        ready_max = fmax(ready_max, readiness);
        volatility_min = fmin(volatility_min, forager_volatility[c]);
        volatility_max = fmax(volatility_max, forager_volatility[c]);
    }
    for (int c = 0; c < coin_count; ++c) {
        score[c] = -INFINITY;
        if (!survivor[c]) continue;
        int bar_offset = (k * coin_count + c) * 4;
        float close = bars[bar_offset + 2];
        float lower = fmin(ema0[c], fmin(ema1[c], ema2[c]));
        float upper = fmax(ema0[c], fmax(ema1[c], ema2[c]));
        float coin_offset_pct = coin_override_or(
            coin_overrides, c, 4, config.offset
        );
        float threshold = short_side
            ? upper * (1.0f + coin_offset_pct)
            : lower * (1.0f - coin_offset_pct);
        float readiness = threshold > 0.0f
            ? (short_side
                ? 1.0f - close / threshold
                : close / threshold - 1.0f)
            : INFINITY;
        float volume_component = volume_max > volume_min
            ? (forager_volume[c] - volume_min) / (volume_max - volume_min)
            : 1.0f;
        float ready_component = ready_max > ready_min
            ? (ready_max - readiness) / (ready_max - ready_min)
            : 1.0f;
        float volatility_component = volatility_max > volatility_min
            ? (forager_volatility[c] - volatility_min)
                / (volatility_max - volatility_min)
            : 1.0f;
        score[c] = config.w_volume * volume_component
            + config.w_ready * ready_component
            + config.w_volatility * volatility_component;
    }
    for (int pick = 0; pick < slots; ++pick) {
        int best = -1;
        for (int c = 0; c < coin_count; ++c) {
            if (!survivor[c] || selected[c]) continue;
            if (best < 0 || score[c] > score[best]
                || (score[c] == score[best] && c < best)) {
                best = c;
            }
        }
        if (best >= 0) selected[best] = true;
    }
    if (score_hysteresis > 0.0f) {
        // Match Rust's score hysteresis: consider incumbent flat candidates
        // from best to worst, then displace only the weakest selected
        // non-incumbent challenger within the configured gap.
        for (int rank = 0; rank < coin_count; ++rank) {
            int incumbent_coin = -1;
            for (int c = 0; c < coin_count; ++c) {
                if (!survivor[c] || !incumbent[c] || selected[c]) continue;
                if (incumbent_coin < 0 || score[c] > score[incumbent_coin]
                    || (score[c] == score[incumbent_coin]
                        && c < incumbent_coin)) {
                    incumbent_coin = c;
                }
            }
            if (incumbent_coin < 0) break;

            int challenger = -1;
            for (int c = 0; c < coin_count; ++c) {
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
                break;
            }
        }
    }
    side.selection_initialized = true;
    side.previous_effective_n_positions = effective_n_positions;
}

// Preselect one side's best eligible candidate without creating an order.
// The fused caller compares both sides before allowing either generator to
// consume the shared realized-loss allowance.
inline int select_ema_multicoin_unstuck_coin(
    thread EmaMulticoinSideState& side,
    thread const EmaMulticoinSideConfig& config,
    thread const JointPortfolioAccount& account,
    constant float* bars,
    constant int* touch_ticks,
    constant float* coin_settings,
    constant float* coin_overrides,
    int k,
    int coin_count,
    bool short_side,
    int effective_n_positions,
    thread float& selected_diff
) {
    selected_diff = INFINITY;
    if (effective_n_positions <= 0 || account.balance <= 0.0f) return -1;
    const float effective_wel = config.twel
        / fmax(float(effective_n_positions), 1.0f);
    const float balance_peak = account.balance
        + (account.realized_pnl_peak - account.realized_pnl_total);
    if (!(balance_peak > 0.0f)) return -1;

    int selected_coin = -1;
    for (int c = 0; c < coin_count; ++c) {
        const int coin_offset = c * COIN_COLS;
        const int bar_offset = (k * coin_count + c) * 4;
        const int tick_offset = (k * coin_count + c) * 2;
        const float price_now = bars[bar_offset + 2];
        const bool managed_candidate =
            k >= int(coin_settings[coin_offset + 6])
            && k <= int(coin_settings[coin_offset + 7])
            && finite_positive(price_now);
        if (!managed_candidate) continue;
        const float c_mult = coin_settings[coin_offset + 4];
        const float price_step = coin_settings[coin_offset + 1];
        const bool coin_unstuck_enabled = coin_override_or(
            coin_overrides, c, 13,
            config.unstuck_enabled ? 1.0f : 0.0f
        ) > 0.5f;
        const bool coin_ema_gate = coin_override_or(
            coin_overrides, c, 14,
            config.unstuck_ema_gating_enabled ? 1.0f : 0.0f
        ) > 0.5f;
        const float coin_close_pct = coin_override_or(
            coin_overrides, c, 15, config.unstuck_close_pct
        );
        const float coin_ema_dist = coin_override_or(
            coin_overrides, c, 16, config.unstuck_ema_dist
        );
        const float coin_loss_allowance_pct = coin_override_or(
            coin_overrides, c, 17, config.unstuck_loss_allowance_pct
        );
        const float coin_threshold = coin_override_or(
            coin_overrides, c, 18, config.unstuck_threshold
        );
        const float fixed_coin_wel = coin_override_or(
            coin_overrides, c, 11, -1.0f
        );
        const float coin_wel = fixed_coin_wel >= 0.0f
            ? fixed_coin_wel : effective_wel;
        const float coin_allowance_pct = coin_override_or(
            coin_overrides, c, 12, config.allowance_pct
        );
        const float allowed_coin_wel = allowed_wallet_exposure_limit(
            coin_wel, config.twel, coin_allowance_pct,
            config.legacy_raw_allowance
        );
        if (!(coin_unstuck_enabled && coin_close_pct > 0.0f
            && coin_loss_allowance_pct > 0.0f && coin_threshold > 0.0f
            && side.psize[c] > 0.0f && side.pprice[c] > 0.0f
            && allowed_coin_wel > 0.0f && price_now > 0.0f)) {
            continue;
        }
        const float allowance = float32_floor_nonnegative(fmax(
            account.balance - balance_peak * (
                1.0f - coin_loss_allowance_pct * config.twel
            ),
            0.0f
        ));
        const float wallet_exposure = side.psize[c] * side.pprice[c]
            * c_mult / account.balance;
        if (!(allowance > 0.0f
            && wallet_exposure / allowed_coin_wel > coin_threshold)) {
            continue;
        }
        if (coin_ema_gate) {
            const float lower = fmin(
                side.ema0[c], fmin(side.ema1[c], side.ema2[c])
            );
            const float upper = fmax(
                side.ema0[c], fmax(side.ema1[c], side.ema2[c])
            );
            const int trigger_tick = short_side
                ? int(floor(
                    lower * (1.0f - coin_ema_dist) / price_step
                        + 1.0e-6f
                ))
                : int(ceil(
                    upper * (1.0f + coin_ema_dist) / price_step
                        - 1.0e-6f
                ));
            const bool triggered = short_side
                ? touch_ticks[tick_offset + 1] <= trigger_tick
                : touch_ticks[tick_offset + 0] >= trigger_tick;
            if (!triggered) continue;
        }
        const float pprice_diff = short_side
            ? price_now / side.pprice[c] - 1.0f
            : 1.0f - price_now / side.pprice[c];
        if (!isfinite(pprice_diff)) continue;
        if (selected_coin < 0 || pprice_diff < selected_diff
            || (pprice_diff == selected_diff && c < selected_coin)) {
            selected_coin = c;
            selected_diff = pprice_diff;
        }
    }
    return selected_coin;
}

inline void generate_ema_multicoin_side_orders(
    thread EmaMulticoinSideState& side,
    thread const EmaMulticoinSideConfig& config,
    thread JointPortfolioAccount& account,
    constant float* bars,
    constant int* touch_ticks,
    constant float* coin_settings,
    constant float* coin_overrides,
    int k,
    int coin_count,
    bool short_side,
    int tradable_count,
    int effective_n_positions,
    int current_hsl_mode,
    bool market_orders_allowed,
    float market_order_near_touch_threshold,
    int forced_unstuck_coin,
    ulong one_way_initial_blocked_mask
) {
    const int C = coin_count;
    const float base_qty_pct = config.base_qty_pct;
    const float ddf = config.ddf;
    const float offset = config.offset;
    const float psize_weight = config.psize_weight;
    const float weight_1h = config.weight_1h;
    const float weight_1m = config.weight_1m;
    const float cooldown_min = config.cooldown_min;
    const float twel = config.twel;
    const int n_positions = config.n_positions;
    const float allowance_pct = config.allowance_pct;
    const bool legacy_raw_allowance = config.legacy_raw_allowance;
    const bool twel_entry_gate_enabled = config.twel_entry_gate_enabled;
    const float twel_threshold = config.twel_threshold;
    const bool twel_enforcer_enabled = config.twel_enforcer_enabled;
    const bool twel_enforcer_reduce_portfolio =
        config.twel_enforcer_reduce_portfolio;
    const bool unstuck_enabled = config.unstuck_enabled;
    const bool unstuck_ema_gating_enabled =
        config.unstuck_ema_gating_enabled;
    const float unstuck_close_pct = config.unstuck_close_pct;
    const float unstuck_ema_dist = config.unstuck_ema_dist;
    const float unstuck_loss_allowance_pct =
        config.unstuck_loss_allowance_pct;
    const float unstuck_threshold = config.unstuck_threshold;
    const bool coin_hsl_mode = config.coin_hsl_mode;
    thread HslState* coin_hsl = side.coin_hsl;
    thread float* ema0 = side.ema0;
    thread float* ema1 = side.ema1;
    thread float* ema2 = side.ema2;
    thread float* volatility_1m = side.volatility_1m;
    thread float* volatility_1h = side.volatility_1h;
    thread float* psize = side.psize;
    thread float* pprice = side.pprice;
    thread float* last_increase_k = side.last_increase_k;
    thread float* entry_qty = side.entry_qty;
    thread float* close_qty = side.close_qty;
    thread float* secondary_close_qty = side.secondary_close_qty;
    thread float* twel_close_qty = side.twel_close_qty;
    thread float* unstuck_close_qty = side.unstuck_close_qty;
    thread float* contribution = side.contribution;
    thread float* minimum_entry = side.minimum_entry;
    thread int* entry_tick = side.entry_tick;
    thread int* close_tick = side.close_tick;
    thread int* secondary_close_tick = side.secondary_close_tick;
    thread int* twel_close_tick = side.twel_close_tick;
    thread int* unstuck_close_tick = side.unstuck_close_tick;
    thread bool* entry_market = side.entry_market;
    thread bool* close_market = side.close_market;
    thread bool* secondary_close_market = side.secondary_close_market;
    thread bool* close_is_unstuck_reducer =
        side.close_is_unstuck_reducer;
    thread bool* close_is_hsl_panic = side.close_is_hsl_panic;
    thread bool* selected = side.selected;
    thread bool* entry_candidate = side.entry_candidate;
    thread float& balance = account.balance;
    thread float& realized_pnl_cumsum_last = account.realized_pnl_total;
    thread float& realized_pnl_cumsum_max = account.realized_pnl_peak;

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
        // Reduce-overweight follows the configured denominator mode using
        // the current eligible count as dynamic mode's observation.
        int current_effective_n_positions =
            wallet_exposure_denominator_n_positions(
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
                int bar_offset = (k * C + c) * 4;
                bool managed_candidate =
                    k >= int(coin_settings[coin_offset + 6])
                    && k <= int(coin_settings[coin_offset + 7]);
                if (!managed_candidate) continue;
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
        side.close_is_protective_reducer[c] = false;
        close_is_unstuck_reducer[c] = false;
    }
    float balance_peak = balance
        + (realized_pnl_cumsum_max - realized_pnl_cumsum_last);
    int unstuck_coin = -1;
    float best_unstuck_diff = INFINITY;
    float selected_unstuck_qty = 0.0f;
    int selected_unstuck_tick = 0;
    for (int c = 0; c < C; ++c) {
        if (forced_unstuck_coin == -1
            || (forced_unstuck_coin >= 0 && c != forced_unstuck_coin)) {
            continue;
        }
        int coin_offset = c * COIN_COLS;
        int bar_offset = (k * C + c) * 4;
        int tick_offset = (k * C + c) * 2;
        float price_now = bars[bar_offset + 2];
        bool managed_candidate =
            k >= int(coin_settings[coin_offset + 6])
            && k <= int(coin_settings[coin_offset + 7])
            && finite_positive(price_now);
        if (!managed_candidate) continue;
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
        entry_market[c] = false;
        close_market[c] = false;
        secondary_close_market[c] = false;
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
        bool candidate_entry_market = should_use_ordinary_market_execution(
            candidate_entry_tick, !short_side, price_now, price_step,
            market_orders_allowed, market_order_near_touch_threshold
        );
        float entry_exposure_price = candidate_entry_market
            ? price_now : entry_price;
        float minimum = min_entry_qty(
            entry_price, qty_step, min_qty, min_cost, c_mult
        );
        float market_entry_minimum = candidate_entry_market
            ? min_entry_qty(
                price_now, qty_step, min_qty, min_cost, c_mult
            ) : minimum;
        float effective_entry_minimum = short_side && candidate_entry_market
            ? market_entry_minimum : minimum;
        minimum_entry[c] = effective_entry_minimum;
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
        bool initial_entry_allowed = psize[c] > 0.0f
            || (one_way_initial_blocked_mask & (1ul << ulong(c))) == 0ul;
        if (selected[c] && initial_entry_allowed
            && !cooldown && entry_price > 0.0f && balance > 0.0f
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
            if (short_side && candidate_entry_market
                && quantity < market_entry_minimum) {
                quantity = market_entry_minimum;
            }
            float headroom = (
                position_cap * balance - psize[c] * pprice[c] * c_mult
            ) / fmax(entry_exposure_price * c_mult, 1.0e-12f);
            bool over = (
                psize[c] * pprice[c] + quantity * entry_exposure_price
            ) * c_mult
                / fmax(balance, 1.0e-9f) >= position_cap;
            if (over) {
                float capped = floor_step(headroom, qty_step);
                quantity = capped > 0.0f
                        && capped + 1.0e-6f >= effective_entry_minimum
                    ? capped : 0.0f;
            }
            entry_qty[c] = quantity;
            entry_tick[c] = candidate_entry_tick;
            entry_market[c] = candidate_entry_market && quantity > 0.0f;
            entry_candidate[c] = quantity > 0.0f;
            contribution[c] = quantity * entry_exposure_price * c_mult
                / balance;
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
            close_market[c] = should_use_ordinary_market_execution(
                candidate_close_tick, short_side, price_now, price_step,
                market_orders_allowed, market_order_near_touch_threshold
            );
            if (close_market[c]) {
                close_qty[c] = resize_market_close_qty(
                    close_qty[c], psize[c], price_now,
                    qty_step, min_qty, min_cost, c_mult
                );
            }
        }

        // Protective reducers are finalized after both sides have generated.
        // That coordinator classifies market intent, compares finalized
        // quantities globally, spends one shared realized-loss allowance, and
        // only then allocates the ordinary close remainder.
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
            float price = entry_market[best]
                ? clamped_market_price(
                    bars, coin_settings, k, best, C
                )
                : float(entry_tick[best]) * price_step;
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
            close_market[c] = false;
            secondary_close_market[c] = false;
            side.close_is_protective_reducer[c] = false;
            close_is_unstuck_reducer[c] = false;
            close_is_hsl_panic[c] = true;
        }
        if (!(entry_qty[c] > 0.0f)) entry_market[c] = false;
        if (!(close_qty[c] > 0.0f)) close_market[c] = false;
        if (!(secondary_close_qty[c] > 0.0f)) {
            secondary_close_market[c] = false;
        }
    }
}

struct EmaMulticoinReducerCandidate {
    bool valid;
    float qty;
    int tick;
    bool market;
    bool is_unstuck;
};

inline EmaMulticoinReducerCandidate empty_ema_multicoin_reducer_candidate() {
    EmaMulticoinReducerCandidate candidate;
    candidate.valid = false;
    candidate.qty = 0.0f;
    candidate.tick = 0;
    candidate.market = false;
    candidate.is_unstuck = false;
    return candidate;
}

inline EmaMulticoinReducerCandidate make_ema_multicoin_reducer_candidate(
    float psize,
    float ordinary_qty,
    int ordinary_tick,
    bool ordinary_market,
    float requested_qty,
    int requested_tick,
    bool is_unstuck,
    bool short_side,
    float market_price,
    float qty_step,
    float price_step,
    float min_qty,
    float min_cost,
    float c_mult,
    bool market_orders_allowed,
    float market_order_near_touch_threshold
) {
    EmaMulticoinReducerCandidate candidate =
        empty_ema_multicoin_reducer_candidate();
    if (!(psize > 0.0f && requested_qty > 0.0f
        && requested_tick > 0 && market_price > 0.0f)) {
        return candidate;
    }
    candidate.market = should_use_ordinary_market_execution(
        requested_tick, short_side, market_price, price_step,
        market_orders_allowed, market_order_near_touch_threshold
    );
    float executable_qty = requested_qty;
    if (candidate.market) {
        executable_qty = resize_market_close_qty(
            executable_qty, psize, market_price,
            qty_step, min_qty, min_cost, c_mult
        );
    }
    float ordinary_price = ordinary_market
        ? market_price : float(ordinary_tick) * price_step;
    float reducer_price = candidate.market
        ? market_price : float(requested_tick) * price_step;
    candidate.qty = finalized_reducer_qty_with_ordinary(
        psize, ordinary_qty, ordinary_price,
        executable_qty, reducer_price,
        qty_step, min_qty, min_cost, c_mult
    );
    candidate.tick = requested_tick;
    candidate.is_unstuck = is_unstuck;
    candidate.valid = candidate.qty > 0.0f;
    return candidate;
}

inline bool ema_multicoin_reducer_candidate_preferred(
    EmaMulticoinReducerCandidate left,
    EmaMulticoinReducerCandidate right,
    bool short_side
) {
    if (!left.valid) return false;
    if (!right.valid) return true;
    if (left.qty != right.qty) return left.qty > right.qty;
    if (left.tick != right.tick) {
        return short_side ? left.tick > right.tick : left.tick < right.tick;
    }
    if (left.is_unstuck != right.is_unstuck) return left.is_unstuck;
    return false;
}

inline void prepare_ema_multicoin_reducer_candidates(
    thread EmaMulticoinSideState& side,
    constant float* bars,
    constant float* coin_settings,
    int k,
    int coin_count,
    bool short_side,
    bool market_orders_allowed,
    float market_order_near_touch_threshold,
    thread EmaMulticoinReducerCandidate* preferred,
    thread EmaMulticoinReducerCandidate* fallback
) {
    for (int c = 0; c < MAX_COINS; ++c) {
        preferred[c] = empty_ema_multicoin_reducer_candidate();
        fallback[c] = empty_ema_multicoin_reducer_candidate();
    }
    for (int c = 0; c < coin_count; ++c) {
        if (!(side.psize[c] > 0.0f) || side.close_is_hsl_panic[c]) continue;
        int coin_offset = c * COIN_COLS;
        float market_price = clamped_market_price(
            bars, coin_settings, k, c, coin_count
        );
        float qty_step = coin_settings[coin_offset + 0];
        float price_step = coin_settings[coin_offset + 1];
        float min_qty = coin_settings[coin_offset + 2];
        float min_cost = coin_settings[coin_offset + 3];
        float c_mult = coin_settings[coin_offset + 4];
        EmaMulticoinReducerCandidate twel =
            make_ema_multicoin_reducer_candidate(
                side.psize[c], side.close_qty[c], side.close_tick[c],
                side.close_market[c], side.twel_close_qty[c],
                side.twel_close_tick[c], false, short_side, market_price,
                qty_step, price_step, min_qty, min_cost, c_mult,
                market_orders_allowed, market_order_near_touch_threshold
            );
        EmaMulticoinReducerCandidate unstuck =
            make_ema_multicoin_reducer_candidate(
                side.psize[c], side.close_qty[c], side.close_tick[c],
                side.close_market[c], side.unstuck_close_qty[c],
                side.unstuck_close_tick[c], true, short_side, market_price,
                qty_step, price_step, min_qty, min_cost, c_mult,
                market_orders_allowed, market_order_near_touch_threshold
            );
        if (ema_multicoin_reducer_candidate_preferred(
                unstuck, twel, short_side)) {
            preferred[c] = unstuck;
            fallback[c] = twel;
        } else {
            preferred[c] = twel;
            fallback[c] = unstuck;
        }
    }
}

inline bool gate_ema_multicoin_close(
    float qty,
    int tick,
    bool market,
    float pprice,
    bool short_side,
    float market_price,
    float price_step,
    float c_mult,
    float maker_fee,
    float taker_fee,
    float market_order_slippage_pct,
    bool gate_enabled,
    thread float& remaining_loss_budget
) {
    if (!(qty > 0.0f && tick > 0 && pprice > 0.0f)) return false;
    float execution_price = market
        ? ordinary_market_fill_price(
            market_price, short_side,
            market_order_slippage_pct, price_step
        )
        : float(tick) * price_step;
    float fee_rate = market ? taker_fee : maker_fee;
    float gross_pnl = qty * c_mult * (short_side
        ? pprice - execution_price : execution_price - pprice);
    float net_pnl = gross_pnl
        - qty * execution_price * c_mult * fee_rate;
    if (!isfinite(net_pnl) || !realized_loss_gate_allows(
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

inline void apply_ema_multicoin_reducer_candidate(
    thread EmaMulticoinSideState& side,
    int coin,
    EmaMulticoinReducerCandidate candidate,
    constant float* bars,
    constant float* coin_settings,
    int k,
    int coin_count
) {
    if (!candidate.valid) return;
    int coin_offset = coin * COIN_COLS;
    float market_price = clamped_market_price(
        bars, coin_settings, k, coin, coin_count
    );
    float qty_step = coin_settings[coin_offset + 0];
    float price_step = coin_settings[coin_offset + 1];
    float min_qty = coin_settings[coin_offset + 2];
    float min_cost = coin_settings[coin_offset + 3];
    float c_mult = coin_settings[coin_offset + 4];
    float reducer_price = candidate.market
        ? market_price : float(candidate.tick) * price_step;
    float reducer_min = min_entry_qty(
        reducer_price, qty_step, min_qty, min_cost, c_mult
    );
    float ordinary_qty = side.close_qty[coin];
    int ordinary_tick = side.close_tick[coin];
    bool ordinary_market = side.close_market[coin];
    side.secondary_close_qty[coin] = 0.0f;
    side.secondary_close_tick[coin] = 0;
    side.secondary_close_market[coin] = false;
    if (ordinary_qty > 0.0f) {
        float ordinary_price = ordinary_market
            ? market_price : float(ordinary_tick) * price_step;
        float ordinary_min = min_entry_qty(
            ordinary_price, qty_step, min_qty, min_cost, c_mult
        );
        if (ordinary_qty + candidate.qty > side.psize[coin]) {
            ordinary_qty = fmax(
                round_step(side.psize[coin] - candidate.qty, qty_step),
                0.0f
            );
        }
        if (ordinary_qty >= ordinary_min) {
            float remainder = fmax(
                round_step(
                    side.psize[coin] - candidate.qty - ordinary_qty,
                    qty_step
                ),
                0.0f
            );
            float minimum_any = fmin(ordinary_min, reducer_min);
            if (remainder > 0.0f && remainder < minimum_any) {
                ordinary_qty = fmin(
                    side.psize[coin] - candidate.qty,
                    round_step(ordinary_qty + remainder, qty_step)
                );
            }
            side.secondary_close_qty[coin] = ordinary_qty;
            side.secondary_close_tick[coin] = ordinary_tick;
            side.secondary_close_market[coin] = ordinary_market;
        }
    }
    side.close_qty[coin] = candidate.qty;
    side.close_tick[coin] = candidate.tick;
    side.close_market[coin] = candidate.market;
    side.close_is_protective_reducer[coin] = true;
    side.close_is_unstuck_reducer[coin] = candidate.is_unstuck;
}

inline float ema_multicoin_remaining_loss_budget(
    thread const JointPortfolioAccount& account,
    float max_realized_loss_pct,
    thread bool& gate_enabled
) {
    gate_enabled = max_realized_loss_pct < 1.0f
        && isfinite(account.balance) && account.balance > 0.0f
        && isfinite(account.realized_pnl_total)
        && isfinite(account.realized_pnl_peak);
    if (!gate_enabled) return INFINITY;
    float balance_peak = account.balance
        + (account.realized_pnl_peak - account.realized_pnl_total);
    if (!(isfinite(balance_peak) && balance_peak > 0.0f)) {
        gate_enabled = false;
        return INFINITY;
    }
    float allowed_loss_budget = float32_floor_nonnegative(
        balance_peak * fmax(max_realized_loss_pct, 0.0f)
    );
    float current_realized_loss = fmax(
        account.realized_pnl_peak - account.realized_pnl_total, 0.0f
    );
    return float32_floor_nonnegative(
        fmax(allowed_loss_budget - current_realized_loss, 0.0f)
    );
}

inline void gate_ema_multicoin_side_ordinary_closes(
    thread EmaMulticoinSideState& side,
    constant float* bars,
    constant float* coin_settings,
    int k,
    int coin_count,
    bool short_side,
    float market_order_slippage_pct,
    bool gate_enabled,
    thread float& remaining_loss_budget
) {
    for (int c = 0; c < coin_count; ++c) {
        if (side.close_is_hsl_panic[c]) continue;
        int coin_offset = c * COIN_COLS;
        float market_price = clamped_market_price(
            bars, coin_settings, k, c, coin_count
        );
        float price_step = coin_settings[coin_offset + 1];
        float c_mult = coin_settings[coin_offset + 4];
        float maker_fee = coin_settings[coin_offset + 5];
        float taker_fee = coin_settings[coin_offset + 11];
        bool has_reducer = side.close_is_protective_reducer[c];
        if (has_reducer) {
            if (side.secondary_close_qty[c] > 0.0f
                && !gate_ema_multicoin_close(
                    side.secondary_close_qty[c],
                    side.secondary_close_tick[c],
                    side.secondary_close_market[c], side.pprice[c],
                    short_side, market_price, price_step, c_mult,
                    maker_fee, taker_fee, market_order_slippage_pct,
                    gate_enabled, remaining_loss_budget
                )) {
                side.secondary_close_qty[c] = 0.0f;
                side.secondary_close_market[c] = false;
            }
        } else if (side.close_qty[c] > 0.0f
            && !gate_ema_multicoin_close(
                side.close_qty[c], side.close_tick[c], side.close_market[c],
                side.pprice[c], short_side, market_price, price_step,
                c_mult, maker_fee, taker_fee, market_order_slippage_pct,
                gate_enabled, remaining_loss_budget
            )) {
            side.close_qty[c] = 0.0f;
            side.close_market[c] = false;
        }
    }
}

inline void finalize_ema_multicoin_reducers_one_side(
    thread EmaMulticoinSideState& side,
    thread const JointPortfolioAccount& account,
    constant float* bars,
    constant float* coin_settings,
    int k,
    int coin_count,
    bool short_side,
    bool market_orders_allowed,
    float market_order_near_touch_threshold,
    float market_order_slippage_pct,
    float max_realized_loss_pct
) {
    EmaMulticoinReducerCandidate preferred[MAX_COINS];
    EmaMulticoinReducerCandidate fallback[MAX_COINS];
    EmaMulticoinReducerCandidate selected[MAX_COINS];
    int candidate_index[MAX_COINS];
    bool resolved[MAX_COINS];
    prepare_ema_multicoin_reducer_candidates(
        side, bars, coin_settings, k, coin_count, short_side,
        market_orders_allowed, market_order_near_touch_threshold,
        preferred, fallback
    );
    for (int c = 0; c < MAX_COINS; ++c) {
        selected[c] = empty_ema_multicoin_reducer_candidate();
        candidate_index[c] = 0;
        resolved[c] = !preferred[c].valid;
    }
    bool loss_gate_enabled;
    float remaining_loss_budget = ema_multicoin_remaining_loss_budget(
        account, max_realized_loss_pct, loss_gate_enabled
    );
    for (int attempt = 0; attempt < MAX_COINS * 2; ++attempt) {
        int best = -1;
        EmaMulticoinReducerCandidate best_candidate =
            empty_ema_multicoin_reducer_candidate();
        for (int c = 0; c < coin_count; ++c) {
            if (resolved[c]) continue;
            EmaMulticoinReducerCandidate candidate = candidate_index[c] == 0
                ? preferred[c] : fallback[c];
            if (!candidate.valid) {
                resolved[c] = true;
                continue;
            }
            if (best < 0 || candidate.qty > best_candidate.qty) {
                best = c;
                best_candidate = candidate;
            }
        }
        if (best < 0) break;
        int coin_offset = best * COIN_COLS;
        float market_price = clamped_market_price(
            bars, coin_settings, k, best, coin_count
        );
        bool allowed = gate_ema_multicoin_close(
            best_candidate.qty, best_candidate.tick, best_candidate.market,
            side.pprice[best], short_side, market_price,
            coin_settings[coin_offset + 1],
            coin_settings[coin_offset + 4],
            coin_settings[coin_offset + 5],
            coin_settings[coin_offset + 11],
            market_order_slippage_pct, loss_gate_enabled,
            remaining_loss_budget
        );
        if (allowed) {
            selected[best] = best_candidate;
            resolved[best] = true;
        } else {
            candidate_index[best] += 1;
            resolved[best] = candidate_index[best] > 1
                || !fallback[best].valid;
        }
    }
    for (int c = 0; c < coin_count; ++c) {
        apply_ema_multicoin_reducer_candidate(
            side, c, selected[c], bars, coin_settings, k, coin_count
        );
    }
    gate_ema_multicoin_side_ordinary_closes(
        side, bars, coin_settings, k, coin_count, short_side,
        market_order_slippage_pct, loss_gate_enabled,
        remaining_loss_budget
    );
}

inline void finalize_ema_multicoin_reducers_fused(
    thread EmaMulticoinSideState& long_side,
    thread EmaMulticoinSideState& short_side,
    thread const JointPortfolioAccount& account,
    constant float* bars,
    constant float* coin_settings,
    int k,
    int coin_count,
    bool long_enabled,
    bool short_enabled,
    bool market_orders_allowed,
    float market_order_near_touch_threshold,
    float market_order_slippage_pct,
    float max_realized_loss_pct
) {
    EmaMulticoinReducerCandidate long_preferred[MAX_COINS];
    EmaMulticoinReducerCandidate long_fallback[MAX_COINS];
    EmaMulticoinReducerCandidate short_preferred[MAX_COINS];
    EmaMulticoinReducerCandidate short_fallback[MAX_COINS];
    EmaMulticoinReducerCandidate long_selected[MAX_COINS];
    EmaMulticoinReducerCandidate short_selected[MAX_COINS];
    int long_candidate_index[MAX_COINS];
    int short_candidate_index[MAX_COINS];
    bool long_resolved[MAX_COINS];
    bool short_resolved[MAX_COINS];
    prepare_ema_multicoin_reducer_candidates(
        long_side, bars, coin_settings, k, coin_count, false,
        long_enabled && market_orders_allowed,
        market_order_near_touch_threshold,
        long_preferred, long_fallback
    );
    prepare_ema_multicoin_reducer_candidates(
        short_side, bars, coin_settings, k, coin_count, true,
        short_enabled && market_orders_allowed,
        market_order_near_touch_threshold,
        short_preferred, short_fallback
    );
    for (int c = 0; c < MAX_COINS; ++c) {
        long_selected[c] = empty_ema_multicoin_reducer_candidate();
        short_selected[c] = empty_ema_multicoin_reducer_candidate();
        long_candidate_index[c] = 0;
        short_candidate_index[c] = 0;
        long_resolved[c] = !long_enabled || !long_preferred[c].valid;
        short_resolved[c] = !short_enabled || !short_preferred[c].valid;
    }
    bool loss_gate_enabled;
    float remaining_loss_budget = ema_multicoin_remaining_loss_budget(
        account, max_realized_loss_pct, loss_gate_enabled
    );
    for (int attempt = 0; attempt < MAX_COINS * 4; ++attempt) {
        int best_coin = -1;
        bool best_short = false;
        EmaMulticoinReducerCandidate best_candidate =
            empty_ema_multicoin_reducer_candidate();
        for (int c = 0; c < coin_count; ++c) {
            if (!long_resolved[c]) {
                EmaMulticoinReducerCandidate candidate =
                    long_candidate_index[c] == 0
                        ? long_preferred[c] : long_fallback[c];
                if (!candidate.valid) {
                    long_resolved[c] = true;
                } else if (best_coin < 0
                    || candidate.qty > best_candidate.qty) {
                    best_coin = c;
                    best_short = false;
                    best_candidate = candidate;
                }
            }
            if (!short_resolved[c]) {
                EmaMulticoinReducerCandidate candidate =
                    short_candidate_index[c] == 0
                        ? short_preferred[c] : short_fallback[c];
                if (!candidate.valid) {
                    short_resolved[c] = true;
                } else if (best_coin < 0
                    || candidate.qty > best_candidate.qty) {
                    best_coin = c;
                    best_short = true;
                    best_candidate = candidate;
                }
            }
        }
        if (best_coin < 0) break;
        int coin_offset = best_coin * COIN_COLS;
        float market_price = clamped_market_price(
            bars, coin_settings, k, best_coin, coin_count
        );
        float pprice = best_short
            ? short_side.pprice[best_coin] : long_side.pprice[best_coin];
        bool allowed = gate_ema_multicoin_close(
            best_candidate.qty, best_candidate.tick, best_candidate.market,
            pprice, best_short, market_price,
            coin_settings[coin_offset + 1],
            coin_settings[coin_offset + 4],
            coin_settings[coin_offset + 5],
            coin_settings[coin_offset + 11],
            market_order_slippage_pct, loss_gate_enabled,
            remaining_loss_budget
        );
        if (best_short) {
            if (allowed) {
                short_selected[best_coin] = best_candidate;
                short_resolved[best_coin] = true;
            } else {
                short_candidate_index[best_coin] += 1;
                short_resolved[best_coin] =
                    short_candidate_index[best_coin] > 1
                    || !short_fallback[best_coin].valid;
            }
        } else if (allowed) {
            long_selected[best_coin] = best_candidate;
            long_resolved[best_coin] = true;
        } else {
            long_candidate_index[best_coin] += 1;
            long_resolved[best_coin] = long_candidate_index[best_coin] > 1
                || !long_fallback[best_coin].valid;
        }
    }
    for (int c = 0; c < coin_count; ++c) {
        if (long_enabled) {
            apply_ema_multicoin_reducer_candidate(
                long_side, c, long_selected[c],
                bars, coin_settings, k, coin_count
            );
        }
        if (short_enabled) {
            apply_ema_multicoin_reducer_candidate(
                short_side, c, short_selected[c],
                bars, coin_settings, k, coin_count
            );
        }
    }
    if (long_enabled) {
        gate_ema_multicoin_side_ordinary_closes(
            long_side, bars, coin_settings, k, coin_count, false,
            market_order_slippage_pct, loss_gate_enabled,
            remaining_loss_budget
        );
    }
    if (short_enabled) {
        gate_ema_multicoin_side_ordinary_closes(
            short_side, bars, coin_settings, k, coin_count, true,
            market_order_slippage_pct, loss_gate_enabled,
            remaining_loss_budget
        );
    }
}

inline bool process_ema_multicoin_side_fills(
    thread EmaMulticoinSideState& side,
    thread JointPortfolioAccount& account,
    thread EmaMulticoinFillState& fills,
    constant float* bars,
    constant int* fill_ticks,
    constant float* coin_settings,
    constant float* coin_overrides,
    device float* coin_fill_counts,
    int candidate_index,
    int k,
    int coin_count,
    bool short_side,
    bool alive,
    bool collect_coin_fill_counts,
    float market_order_slippage_pct,
    bool hsl_panic_market,
    float hsl_equity_before_fills
) {
    thread HslState& hsl = side.hsl;
    thread HslState* coin_hsl = side.coin_hsl;
    const bool coin_hsl_mode = hsl.signal_mode == HSL_SIGNAL_COIN;
    thread float* psize = side.psize;
    thread float* pprice = side.pprice;
    thread float* last_increase_k = side.last_increase_k;
    thread float* entry_qty = side.entry_qty;
    thread float* close_qty = side.close_qty;
    thread float* secondary_close_qty = side.secondary_close_qty;
    thread float* position_open_k = side.position_open_k;
    thread float* position_last_fill_k = side.position_last_fill_k;
    thread int* entry_tick = side.entry_tick;
    thread int* close_tick = side.close_tick;
    thread int* secondary_close_tick = side.secondary_close_tick;
    thread bool* entry_market = side.entry_market;
    thread bool* close_market = side.close_market;
    thread bool* secondary_close_market = side.secondary_close_market;
    thread bool* close_is_protective_reducer =
        side.close_is_protective_reducer;
    thread bool* close_is_unstuck_reducer = side.close_is_unstuck_reducer;
    thread bool* close_is_hsl_panic = side.close_is_hsl_panic;
    thread float* coin_realized_pnl = side.coin_realized_pnl;
    thread float& balance = account.balance;

    bool any_fill = false;
    for (int c = 0; c < coin_count; ++c) {
        const int coin_offset = c * COIN_COLS;
        const int bar_offset = (k * coin_count + c) * 4;
        const int tick_offset = (k * coin_count + c) * 2;
        const float high = bars[bar_offset + 0];
        const float low = bars[bar_offset + 1];
        const float close = bars[bar_offset + 2];
        const int first_valid = int(coin_settings[coin_offset + 6]);
        const int last_valid = int(coin_settings[coin_offset + 7]);
        const bool valid = k >= first_valid && k <= last_valid
            && finite_positive(high) && finite_positive(low)
            && finite_positive(close);
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
        bool primary_ordinary_market = !close_is_hsl_panic[c]
            && close_market[c];
        bool filled_close = close_qty[c] > 0.0f && psize[c] > 0.0f
            && (primary_market_panic || primary_ordinary_market || (short_side
                ? close_tick[c] > fill_ticks[tick_offset + 1]
                : close_tick[c] <= fill_ticks[tick_offset + 0]));
        bool filled_secondary_close = secondary_close_qty[c] > 0.0f
            && psize[c] > 0.0f
            && (secondary_close_market[c] || (short_side
                ? secondary_close_tick[c] > fill_ticks[tick_offset + 1]
                : secondary_close_tick[c] <= fill_ticks[tick_offset + 0]));
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
            bool market_execution = market_panic || (use_secondary
                ? secondary_close_market[c] : primary_ordinary_market);
            float fill_price = market_execution
                ? ordinary_market_fill_price(
                    close, short_side, market_order_slippage_pct, price_step
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
                    * (market_execution ? taker_fee : maker_fee);
            bool is_hsl_panic = !use_secondary
                && close_is_hsl_panic[c];
            record_ema_multicoin_close_fill(
                side, account, fills, coin_fill_counts,
                candidate_index, coin_count, c, k, pnl, net_pnl,
                adjusted, pprice[c], close, c_mult, short_side,
                is_hsl_panic, collect_coin_fill_counts,
                hsl_equity_before_fills
            );
            float new_size = fmax(
                round_step(psize[c] - adjusted, qty_step), 0.0f
            );
            bool went_flat = new_size <= 0.0f;
            psize[c] = new_size;
            if (went_flat) {
                pprice[c] = 0.0f;
                if (position_open_k[c] >= 0.0f) {
                    float held_min = float(k) - position_open_k[c];
                    fills.held_max_min = fmax(
                        fills.held_max_min, held_min
                    );
                    fills.held_sum_min += held_min;
                    fills.held_count += 1.0f;
                }
                position_open_k[c] = -1.0f;
            }
            fills.day_volume += fabs(adjusted) * fill_price * c_mult / balance;
            if (use_secondary) {
                secondary_close_qty[c] = 0.0f;
                secondary_close_market[c] = false;
            } else {
                close_qty[c] = 0.0f;
                close_market[c] = false;
                close_is_protective_reducer[c] = false;
                close_is_unstuck_reducer[c] = false;
                close_is_hsl_panic[c] = false;
            }
            executed_close = true;
        }
        if (executed_close) any_fill = true;

        bool was_flat = psize[c] <= 0.0f;
        bool filled_entry = entry_qty[c] > 0.0f
            && (entry_market[c] || (short_side
                ? entry_tick[c] <= fill_ticks[tick_offset + 0]
                : entry_tick[c] > fill_ticks[tick_offset + 1]));
        if (filled_entry) {
            float fill_price = entry_market[c]
                ? ordinary_market_fill_price(
                    close, !short_side,
                    market_order_slippage_pct, price_step
                )
                : float(entry_tick[c]) * price_step;
            float adjusted = round_step(entry_qty[c], qty_step);
            float fee = adjusted * fill_price * c_mult
                * (entry_market[c] ? taker_fee : maker_fee);
            record_realized_net(
                -fee, account,
                fills.day_fill_count, fills.fill_count,
                fills.fill_count_entry, fills.fill_count_long,
                fills.pnl_recovery_peak, fills.pnl_recovery_peak_k,
                fills.pnl_recovery_max_min, float(k),
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
                coin_fill_counts[candidate_index * coin_count + c] += 1.0f;
            }
            float new_size = round_step(psize[c] + adjusted, qty_step);
            float new_price = was_flat ? fill_price
                : pprice[c] * (psize[c] / fmax(new_size, 1.0e-12f))
                    + fill_price * (adjusted / fmax(new_size, 1.0e-12f));
            if (was_flat) position_open_k[c] = float(k);
            psize[c] = new_size;
            pprice[c] = new_price;
            last_increase_k[c] = float(k);
            fills.day_volume += fabs(adjusted) * fill_price * c_mult / balance;
            entry_qty[c] = 0.0f;
            entry_market[c] = false;
            any_fill = true;
        }
        if (executed_close || filled_entry) {
            if (position_last_fill_k[c] >= 0.0f) {
                fills.position_unchanged_max_min = fmax(
                    fills.position_unchanged_max_min,
                    float(k) - position_last_fill_k[c]
                );
            }
            position_last_fill_k[c] = psize[c] > 0.0f ? float(k) : -1.0f;
        }
    }
    return any_fill;
}

inline void clear_ema_multicoin_coin_orders(
    thread EmaMulticoinSideState& side,
    int coin
) {
    side.entry_qty[coin] = 0.0f;
    side.close_qty[coin] = 0.0f;
    side.secondary_close_qty[coin] = 0.0f;
    side.twel_close_qty[coin] = 0.0f;
    side.unstuck_close_qty[coin] = 0.0f;
    side.entry_tick[coin] = 0;
    side.close_tick[coin] = 0;
    side.secondary_close_tick[coin] = 0;
    side.twel_close_tick[coin] = 0;
    side.unstuck_close_tick[coin] = 0;
    side.entry_market[coin] = false;
    side.close_market[coin] = false;
    side.secondary_close_market[coin] = false;
    side.close_is_protective_reducer[coin] = false;
    side.close_is_unstuck_reducer[coin] = false;
    side.close_is_hsl_panic[coin] = false;
}

inline bool force_close_ema_multicoin_delisted_position(
    thread EmaMulticoinSideState& side,
    thread JointPortfolioAccount& account,
    thread EmaMulticoinFillState& fills,
    constant float* bars,
    constant float* coin_settings,
    device float* coin_fill_counts,
    int candidate_index,
    int k,
    int coin_count,
    int coin,
    bool short_side,
    bool collect_coin_fill_counts,
    float market_order_slippage_pct,
    thread float& hsl_equity_before_close
) {
    if (!(side.psize[coin] > 0.0f && side.pprice[coin] > 0.0f)) {
        return false;
    }
    const int coin_offset = coin * COIN_COLS;
    const int bar_offset = (k * coin_count + coin) * 4;
    const float close = bars[bar_offset + 2];
    if (!finite_positive(close)) return false;
    const float price_step = coin_settings[coin_offset + 1];
    const float c_mult = coin_settings[coin_offset + 4];
    const float taker_fee = coin_settings[coin_offset + 11];
    const float close_price = ordinary_market_fill_price(
        close, short_side, market_order_slippage_pct, price_step
    );
    const float close_qty = side.psize[coin];
    const float position_price = side.pprice[coin];
    const float pnl = close_qty * c_mult * (
        short_side
            ? position_price - close_price
            : close_price - position_price
    );
    const float net_pnl = pnl
        - close_qty * close_price * c_mult * taker_fee;
    const bool coin_hsl_mode =
        side.hsl.signal_mode == HSL_SIGNAL_COIN;
    record_ema_multicoin_close_fill(
        side, account, fills, coin_fill_counts,
        candidate_index, coin_count, coin, k, pnl, net_pnl,
        close_qty, position_price, close, c_mult, short_side,
        true, collect_coin_fill_counts, hsl_equity_before_close
    );
    if (!coin_hsl_mode) {
        advance_coin_hsl_equity_after_close_fill(
            hsl_equity_before_close,
            net_pnl, close_qty, position_price, close,
            c_mult, short_side
        );
    }
    side.psize[coin] = 0.0f;
    side.pprice[coin] = 0.0f;
    if (side.position_open_k[coin] >= 0.0f) {
        const float held_min = float(k) - side.position_open_k[coin];
        fills.held_max_min = fmax(fills.held_max_min, held_min);
        fills.held_sum_min += held_min;
        fills.held_count += 1.0f;
    }
    side.position_open_k[coin] = -1.0f;
    fills.day_volume += close_qty * close_price * c_mult / account.balance;
    if (side.position_last_fill_k[coin] >= 0.0f) {
        fills.position_unchanged_max_min = fmax(
            fills.position_unchanged_max_min,
            float(k) - side.position_last_fill_k[coin]
        );
    }
    side.position_last_fill_k[coin] = -1.0f;
    return true;
}

inline bool force_close_ema_multicoin_delisted_one_side(
    thread EmaMulticoinSideState& side,
    thread JointPortfolioAccount& account,
    thread EmaMulticoinFillState& fills,
    constant float* bars,
    constant float* coin_settings,
    device float* coin_fill_counts,
    int candidate_index,
    int k,
    int timestep_count,
    int coin_count,
    bool short_side,
    bool collect_coin_fill_counts,
    float market_order_slippage_pct,
    thread float& hsl_equity_before_close
) {
    bool any_close = false;
    for (int c = 0; c < coin_count; ++c) {
        const int last_valid = int(coin_settings[c * COIN_COLS + 7]);
        if (k != last_valid || last_valid + 1400 >= timestep_count) continue;
        const bool closed = force_close_ema_multicoin_delisted_position(
            side, account, fills, bars, coin_settings, coin_fill_counts,
            candidate_index, k, coin_count, c, short_side,
            collect_coin_fill_counts, market_order_slippage_pct,
            hsl_equity_before_close
        );
        if (closed) clear_ema_multicoin_coin_orders(side, c);
        any_close = any_close || closed;
    }
    return any_close;
}

inline bool force_close_ema_multicoin_delisted_fused(
    thread EmaMulticoinSideState& long_side,
    thread EmaMulticoinSideState& short_side,
    thread JointPortfolioAccount& account,
    thread EmaMulticoinFillState& fills,
    constant float* bars,
    constant float* coin_settings,
    device float* coin_fill_counts,
    int candidate_index,
    int k,
    int timestep_count,
    int coin_count,
    bool collect_coin_fill_counts,
    float market_order_slippage_pct,
    thread float& hsl_equity_before_close
) {
    bool any_close = false;
    for (int c = 0; c < coin_count; ++c) {
        const int last_valid = int(coin_settings[c * COIN_COLS + 7]);
        if (k != last_valid || last_valid + 1400 >= timestep_count) continue;
        const bool long_closed = force_close_ema_multicoin_delisted_position(
            long_side, account, fills, bars, coin_settings, coin_fill_counts,
            candidate_index, k, coin_count, c, false,
            collect_coin_fill_counts, market_order_slippage_pct,
            hsl_equity_before_close
        );
        const bool short_closed = force_close_ema_multicoin_delisted_position(
            short_side, account, fills, bars, coin_settings, coin_fill_counts,
            candidate_index, k, coin_count, c, true,
            collect_coin_fill_counts, market_order_slippage_pct,
            hsl_equity_before_close
        );
        if (long_closed || short_closed) {
            clear_ema_multicoin_coin_orders(long_side, c);
            clear_ema_multicoin_coin_orders(short_side, c);
        }
        any_close = any_close || long_closed || short_closed;
    }
    return any_close;
}

inline void passivbot_ema_anchor_multicoin_impl(
    constant float* bars,
    constant int* fill_ticks,
    constant int* touch_ticks,
    constant float* hour_log_ranges,
    constant float* coin_settings,
    constant float* coin_overrides,
    constant float* params,
    constant float* run_settings,
    constant int* sizes,
    constant int* end_steps,
#if PASSIVBOT_BTC_PRICES_ENABLED
    constant float* btc_prices,
#endif
#if PASSIVBOT_EQUITY_BALANCE_DIFF_ENABLED
    device float* equity_balance_diff,
#endif
    device float* daily,
    device float* scalars,
    device int* gap_hist,
    device float* coin_fill_counts,
#ifdef PASSIVBOT_STRATEGY_EQ_RECOVERY_DISTRIBUTION_ENABLED
    device float* recovery_samples,
#endif
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
#ifdef PASSIVBOT_STRATEGY_EQ_RECOVERY_DISTRIBUTION_ENABLED
    const int recovery_stride = sizes[8];
    const int recovery_sample_count = sizes[9];
#endif
    if (b >= uint(B)) return;
    const int stop_k = clamp(end_steps[b], 1, T - 1);
    const bool collect_coin_fill_counts = run_settings[6] > 0.5f;
    if (collect_coin_fill_counts) {
        for (int c = 0; c < C; ++c) {
            coin_fill_counts[int(b) * C + c] = 0.0f;
        }
    }

    const int po = int(b) * PARAM_COLS;
    const EmaMulticoinSideConfig config = load_ema_multicoin_side_config(params, po);
    const float base_qty_pct = config.base_qty_pct;
    const float twel = config.twel;
    const int n_positions = config.n_positions;
    const float allowance_pct = config.allowance_pct;
    const bool legacy_raw_allowance = config.legacy_raw_allowance;
    EmaMulticoinSideState side;
    init_ema_multicoin_side_state(
        side, config, coin_settings, coin_overrides, C
    );
    thread HslState& hsl = side.hsl;
    const bool coin_hsl_mode = config.coin_hsl_mode;
    thread HslState* coin_hsl = side.coin_hsl;

    const float starting_balance = run_settings[0];
    const float liquidation_floor = run_settings[1];
    const float interval_ms = run_settings[2];
    const float score_hysteresis = fmax(run_settings[4], 0.0f);
    const float max_realized_loss_pct = run_settings[5];
    const float market_order_slippage_pct = fmax(run_settings[7], 0.0f);
    const bool hsl_panic_market = run_settings[8] > 0.5f;
    const bool market_orders_allowed = run_settings[9] > 0.5f;
    const float market_order_near_touch_threshold = fmax(
        run_settings[10], 0.0f
    );
    const bool filter_by_min_effective_cost = run_settings[11] > 0.5f;
    const float log_bin_scale = 127.0f / log(4000001.0f);

    thread float* psize = side.psize;
    thread float* pprice = side.pprice;
    thread float* entry_qty = side.entry_qty;
    thread float* close_qty = side.close_qty;
    thread float* secondary_close_qty = side.secondary_close_qty;
    thread float* position_open_k = side.position_open_k;
    thread float* position_last_fill_k = side.position_last_fill_k;
    thread float* coin_realized_pnl = side.coin_realized_pnl;
    for (int j = 0; j < GAP_BINS; ++j) {
        gap_hist[int(b) * GAP_BINS + j] = 0;
    }

    JointPortfolioAccount account = init_joint_portfolio_account(starting_balance);
    thread float& balance = account.balance;
    thread float& realized_pnl_cumsum_last = account.realized_pnl_total;
    EmaMulticoinFillState fills = init_ema_multicoin_fill_state();
    thread float& pnl_recovery_peak_k = fills.pnl_recovery_peak_k;
    thread float& pnl_recovery_max_min = fills.pnl_recovery_max_min;
    thread float& profit_sum = fills.profit_sum;
    thread float& loss_sum = fills.loss_sum;
    thread float& fill_count = fills.fill_count;
    thread float& fill_count_entry = fills.fill_count_entry;
    thread float& fill_count_long = fills.fill_count_long;
    float fills_active_days_count = 0.0f;
    int last_active_fill_day = -1;
    bool alive = true;
    bool equity_started = false;
    bool min_cost_exact_open_uncertain = false;
    float run_peak = -INFINITY;
    float max_dd = 0.0f;
    float total_wallet_exposure_max = 0.0f;
    float total_wallet_exposure_mean = 0.0f;
    float total_wallet_exposure_samples = 0.0f;
    thread float& held_max_min = fills.held_max_min;
    thread float& held_sum_min = fills.held_sum_min;
    thread float& held_count = fills.held_count;
    thread float& position_unchanged_max_min = fills.position_unchanged_max_min;
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
#if PASSIVBOT_BTC_RISK_ENABLED
    BtcRiskState btc_risk = init_btc_risk_state();
#endif
#if PASSIVBOT_EQUITY_BALANCE_DIFF_ENABLED
    EquityBalanceDiffState equity_balance_diff_state =
        init_equity_balance_diff_state();
#endif
#ifdef PASSIVBOT_STRATEGY_EQ_RECOVERY_DISTRIBUTION_ENABLED
    int recovery_start_k = -1;
#endif
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
    thread float& day_volume = fills.day_volume;
    float day_has_fill = 0.0f;
    float day_min_balance = INFINITY;
    float day_start_balance = balance;
    thread float& day_fill_count = fills.day_fill_count;

    for (int k = 1; k < stop_k; ++k) {
        const int day_index = multicoin_utc_day_index(
            start_day_minute, k, interval_ms
        );
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
#if PASSIVBOT_BTC_RISK_ENABLED
                write_btc_risk_day(btc_risk, daily, output, 9);
#endif
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
#if PASSIVBOT_BTC_RISK_ENABLED
            reset_btc_risk_day(btc_risk);
#endif
        }

        float hsl_equity_before_fills =
            accumulate_ema_multicoin_side_unrealized_pnl(
                side, bars, coin_settings, k, C, short_side, balance
            );
        bool any_fill = process_ema_multicoin_side_fills(
            side, account, fills,
            bars, fill_ticks, coin_settings, coin_overrides,
            coin_fill_counts, int(b), k, C, short_side, alive,
            collect_coin_fill_counts, market_order_slippage_pct,
            hsl_panic_market, hsl_equity_before_fills
        );
        update_ema_multicoin_side_indicators(
            side, config, bars, hour_log_ranges,
            coin_settings, k, C
        );

        int tradable_count = count_ema_multicoin_tradable_coins(
            bars, coin_settings, coin_overrides, k, C
        );
        const bool post_fill_balance_depleted = isfinite(balance) && balance <= 0.0f;
        const bool past_activation_guard =
            k > max(global_warmup, 1) && k >= requested_start_k;
        if (alive && !post_fill_balance_depleted && past_activation_guard) {
            side.max_tradable_seen = max(
                side.max_tradable_seen, tradable_count
            );
        }
        const int effective_n_positions =
            wallet_exposure_denominator_n_positions(
                n_positions, side.max_tradable_seen
            );
        const bool can_generate = alive && effective_n_positions > 0
            && side.max_tradable_seen > 0 && past_activation_guard;
        equity_started = equity_started || can_generate;
        bool has_hsl_position = ema_multicoin_side_has_position(side, C);
        int current_hsl_mode = coin_hsl_mode
            ? 0 : hsl_mode(hsl, has_hsl_position);
        // A proxy position or filter rejection can leave exact Rust in a
        // different open/cash state. Once that can happen,
        // never reuse the equity-derived liquidation floor as a cash bound,
        // even if the proxy later looks flat.
        if (has_hsl_position) min_cost_exact_open_uncertain = true;
        float min_cost_balance_lower = min_cost_exact_open_uncertain
            ? 0.0f : liquidation_floor;
        if (filter_by_min_effective_cost && can_generate
            && !min_cost_exact_open_uncertain
            && multicoin_min_cost_rejection_possible(
                side.psize, side.coin_hsl, config.coin_hsl_mode,
                current_hsl_mode, config.twel, config.allowance_pct,
                config.legacy_raw_allowance, config.base_qty_pct,
                bars, coin_settings, coin_overrides, 11, 12, 0,
                k, C, effective_n_positions,
                min_cost_balance_lower
            )) {
            min_cost_exact_open_uncertain = true;
            min_cost_balance_lower = 0.0f;
        }

        if (can_generate) {
            update_ema_multicoin_side_selection(
                side, config, bars, coin_settings, coin_overrides,
                k, C, short_side,
                any_fill || min_cost_exact_open_uncertain,
                effective_n_positions,
                score_hysteresis, 0ul,
                filter_by_min_effective_cost, min_cost_balance_lower
            );
            generate_ema_multicoin_side_orders(
                side, config, account,
                bars, touch_ticks, coin_settings, coin_overrides,
                k, C, short_side, tradable_count, effective_n_positions,
                current_hsl_mode, market_orders_allowed,
                market_order_near_touch_threshold, -2, 0ul
            );
            if (filter_by_min_effective_cost) {
                min_cost_exact_open_uncertain = true;
            }
            finalize_ema_multicoin_reducers_one_side(
                side, account, bars, coin_settings, k, C, short_side,
                market_orders_allowed, market_order_near_touch_threshold,
                market_order_slippage_pct, max_realized_loss_pct
            );
        }

        float forced_delist_equity =
            accumulate_ema_multicoin_side_unrealized_pnl(
                side, bars, coin_settings, k, C, short_side, balance
            );
        bool forced_delist_fill = false;
        if (alive && balance > 0.0f) {
            forced_delist_fill = force_close_ema_multicoin_delisted_one_side(
                side, account, fills, bars, coin_settings,
                coin_fill_counts, int(b), k, T, C, short_side,
                collect_coin_fill_counts, market_order_slippage_pct,
                forced_delist_equity
            );
        }
        any_fill = any_fill || forced_delist_fill;
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

        float unrealized = 0.0f;
        float position_cost = 0.0f;
        bool has_open_position = false;
        bool has_blocking_orders = false;
        for (int c = 0; c < C; ++c) {
            int coin_offset = c * COIN_COLS;
            int bar_offset = (k * C + c) * 4;
            float close = bars[bar_offset + 2];
            bool valid = k >= int(coin_settings[coin_offset + 6])
                && k <= int(coin_settings[coin_offset + 7])
                && finite_positive(close);
            bool mark_valid = k >= int(coin_settings[coin_offset + 6])
                && k <= int(coin_settings[coin_offset + 7])
                && isfinite(close);
            if (psize[c] > 0.0f) {
                has_open_position = true;
                position_cost += psize[c] * pprice[c]
                    * coin_settings[coin_offset + 4];
                if (mark_valid) {
                    unrealized += psize[c] * coin_settings[coin_offset + 4]
                        * (short_side ? pprice[c] - close : close - pprice[c]);
                }
            }
            int coin_mode = coin_hsl_mode
                ? hsl_mode(coin_hsl[c], psize[c] > 0.0f)
                : current_hsl_mode;
            if (valid && coin_mode != 3 && (
                    entry_qty[c] > 0.0f || close_qty[c] > 0.0f
                        || secondary_close_qty[c] > 0.0f
                )) {
                has_blocking_orders = true;
            }
        }
        float equity = balance + unrealized;
        // Exact Rust keeps advancing balance-only equity and HSL time once
        // portfolio tracking starts, including declared all-invalid gaps and
        // tails. Per-coin validity still blocks fills, orders, and unrealized
        // PnL above.
        if (can_generate && alive
            && balance > 0.0f && equity > liquidation_floor) {
            int sampled_hsl_tier = 0;
#if PASSIVBOT_HSL_EMA_TAIL_ENABLED
            float sampled_hsl_drawdown_ema = 0.0f;
#endif
            bool hsl_sample_enabled = !coin_hsl_mode && hsl.enabled;
            const bool hsl_strategy_eq_sample_enabled = coin_hsl_mode
                ? false : hsl.enabled && !hsl.halted;
            if (coin_hsl_mode) {
                for (int c = 0; c < C; ++c) {
                    hsl_sample_enabled = hsl_sample_enabled || coin_hsl[c].enabled;
                    int coin_offset = c * COIN_COLS;
                    int bar_offset = (k * C + c) * 4;
                    float close = bars[bar_offset + 2];
                    bool valid = k >= int(coin_settings[coin_offset + 6])
                        && k <= int(coin_settings[coin_offset + 7])
                        && finite_positive(close);
                    bool mark_valid = k >= int(coin_settings[coin_offset + 6])
                        && k <= int(coin_settings[coin_offset + 7])
                        && isfinite(close);
                    float coin_unrealized = psize[c] > 0.0f
                        && mark_valid
                        ? psize[c] * coin_settings[coin_offset + 4]
                            * (short_side ? pprice[c] - close : close - pprice[c])
                        : 0.0f;
                    int coin_mode = hsl_mode(coin_hsl[c], psize[c] > 0.0f);
                    bool coin_has_blocking_orders = valid && coin_mode != 3 && (
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
#if PASSIVBOT_HSL_EMA_TAIL_ENABLED
                    if (coin_hsl[c].enabled) {
                        sampled_hsl_drawdown_ema = fmax(
                            sampled_hsl_drawdown_ema,
                            fabs(coin_hsl[c].drawdown_ema)
                        );
                    }
#endif
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
#if PASSIVBOT_HSL_EMA_TAIL_ENABLED
                sampled_hsl_drawdown_ema = fabs(hsl.drawdown_ema);
#endif
            }
            if ((coin_hsl_mode && effective_n_positions > 0 && hsl_sample_enabled)
                || hsl_strategy_eq_sample_enabled) {
#if PASSIVBOT_HSL_EMA_TAIL_ENABLED
                update_hsl_drawdown_ema_tail_stats(
                    side.hsl_ema_tail, sampled_hsl_drawdown_ema
                );
#endif
                update_hsl_strategy_equity_stats(
                    side.hsl_strategy_eq,
                    starting_balance + realized_pnl_cumsum_last + unrealized,
                    day_index
                );
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
        bool active = equity_started && alive;
        if (active) {
            if (first_eq_k < 0.0f) first_eq_k = float(k);
            last_eq_k = float(k);
            if (any_fill) {
                int active_fill_day = multicoin_active_fill_day(
                    k, int(first_eq_k), interval_ms
                );
                if (active_fill_day != last_active_fill_day) {
                    fills_active_days_count += 1.0f;
                    last_active_fill_day = active_fill_day;
                }
            }
            bool liquidated = balance <= 0.0f || equity <= liquidation_floor;
            float effective_equity = liquidated ? liquidation_floor : equity;
#ifdef PASSIVBOT_STRATEGY_EQ_RECOVERY_DISTRIBUTION_ENABLED
            if (recovery_stride > 0 && recovery_start_k < 0) {
                recovery_start_k = k;
                recovery_samples[int(b) * recovery_sample_count]
                    = effective_equity;
            } else if (recovery_stride > 0) {
                const int recovery_elapsed = k - recovery_start_k;
                const bool recovery_terminal = liquidated || k == stop_k - 1;
                const bool recovery_regular =
                    recovery_elapsed % recovery_stride == 0;
                if (recovery_regular || recovery_terminal) {
                    const int sample_index = recovery_terminal
                        ? (recovery_elapsed + recovery_stride - 1)
                            / recovery_stride
                        : recovery_elapsed / recovery_stride;
                    if (sample_index < recovery_sample_count) {
                        recovery_samples[
                            int(b) * recovery_sample_count + sample_index
                        ] = effective_equity;
                    }
                }
            }
#endif
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
#if PASSIVBOT_BTC_RISK_ENABLED
            update_btc_risk_state(
                btc_risk, effective_equity, btc_prices[k]
            );
#endif
#if PASSIVBOT_EQUITY_BALANCE_DIFF_ENABLED
            update_equity_balance_diff_state(
                equity_balance_diff_state,
                balance,
                effective_equity,
                btc_prices,
                k,
                starting_balance,
                any_fill
            );
#endif
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
#if PASSIVBOT_BTC_RISK_ENABLED
        write_btc_risk_day(btc_risk, daily, output, 9);
#endif
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
    int entry_effective_n_positions =
        wallet_exposure_denominator_n_positions(
            n_positions, side.max_tradable_seen
        );
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
    scalars[scalar_offset + 59] = short_side ? 0.0f
        : hsl_strategy_equity_recovery_max_steps(side.hsl_strategy_eq) * interval_ms;
    scalars[scalar_offset + 60] = short_side
        ? hsl_strategy_equity_recovery_max_steps(side.hsl_strategy_eq) * interval_ms
        : 0.0f;
#if PASSIVBOT_HSL_EMA_TAIL_ENABLED
    scalars[scalar_offset + 61] = short_side ? 0.0f
        : hsl_drawdown_ema_mean_worst_1pct(side.hsl_ema_tail);
    scalars[scalar_offset + 62] = short_side
        ? hsl_drawdown_ema_mean_worst_1pct(side.hsl_ema_tail) : 0.0f;
#endif
#if PASSIVBOT_HSL_RAW_DRAWDOWN_ENABLED
    scalars[scalar_offset + 63] = short_side ? 0.0f
        : hsl_strategy_equity_drawdown_max(side.hsl_strategy_eq);
    scalars[scalar_offset + 64] = short_side
        ? hsl_strategy_equity_drawdown_max(side.hsl_strategy_eq) : 0.0f;
#endif
#if PASSIVBOT_HSL_RAW_TAIL_ENABLED
    scalars[scalar_offset + 65] = short_side ? 0.0f
        : hsl_strategy_equity_drawdown_mean_worst_1pct(side.hsl_strategy_eq);
    scalars[scalar_offset + 66] = short_side
        ? hsl_strategy_equity_drawdown_mean_worst_1pct(side.hsl_strategy_eq)
        : 0.0f;
#endif
#if PASSIVBOT_EQUITY_BALANCE_DIFF_ENABLED
    write_equity_balance_diff_state(
        equity_balance_diff_state, equity_balance_diff, b
    );
#endif
}

inline float ema_multicoin_entry_initial_balance_pct(
    thread const EmaMulticoinSideConfig& config,
    constant float* coin_overrides,
    int effective_n_positions
) {
    float base_limit = effective_n_positions > 0
        ? config.twel / float(effective_n_positions) : 0.0f;
    float initial_qty_pct = coin_override_or(
        coin_overrides, 0, 0, config.base_qty_pct
    );
    float allowance_pct = coin_override_or(
        coin_overrides, 0, 12, config.allowance_pct
    );
    return allowed_wallet_exposure_limit(
        base_limit, config.twel, allowance_pct, config.legacy_raw_allowance
    ) * initial_qty_pct;
}

// Match exact Rust's per-symbol one-way eligibility before Forager selection
// and side-wide entry-cap allocation. Opposite-held coins are excluded from
// selection; flat/flat arbitration losers retain their selected slot but emit
// no order. Reentries and closes remain unblocked.
inline void compute_ema_multicoin_one_way_initial_blocks(
    thread EmaMulticoinSideState& long_side,
    thread const EmaMulticoinSideConfig& long_config,
    constant float* long_coin_overrides,
    thread EmaMulticoinSideState& short_side,
    thread const EmaMulticoinSideConfig& short_config,
    constant float* short_coin_overrides,
    constant float* bars,
    constant float* coin_settings,
    int k,
    int coin_count,
    int long_hsl_mode,
    int short_hsl_mode,
    int long_effective_n_positions,
    int short_effective_n_positions,
    bool long_can_generate,
    bool short_can_generate,
    bool filter_by_min_effective_cost,
    float guaranteed_balance_lower,
    thread ulong& long_selection_blocked_mask,
    thread ulong& short_selection_blocked_mask,
    thread ulong& long_order_blocked_mask,
    thread ulong& short_order_blocked_mask
) {
    long_selection_blocked_mask = 0ul;
    short_selection_blocked_mask = 0ul;
    long_order_blocked_mask = 0ul;
    short_order_blocked_mask = 0ul;
    for (int c = 0; c < coin_count; ++c) {
        const ulong bit = 1ul << ulong(c);
        const bool has_long = long_side.psize[c] > 0.0f;
        const bool has_short = short_side.psize[c] > 0.0f;
        if (has_long && !has_short) {
            short_selection_blocked_mask |= bit;
            short_order_blocked_mask |= bit;
            continue;
        }
        if (has_short && !has_long) {
            long_selection_blocked_mask |= bit;
            long_order_blocked_mask |= bit;
            continue;
        }
        if (has_long && has_short) {
            long_selection_blocked_mask |= bit;
            short_selection_blocked_mask |= bit;
            long_order_blocked_mask |= bit;
            short_order_blocked_mask |= bit;
            continue;
        }

        const int coin_offset = c * COIN_COLS;
        const float close = bars[(k * coin_count + c) * 4 + 2];
        const float long_wel = coin_override_or(
            long_coin_overrides, c, 11, -1.0f
        );
        const float short_wel = coin_override_or(
            short_coin_overrides, c, 11, -1.0f
        );
        const float long_base_limit = long_wel >= 0.0f
            ? long_wel : long_config.twel
                / fmax(float(long_effective_n_positions), 1.0f);
        const float short_base_limit = short_wel >= 0.0f
            ? short_wel : short_config.twel
                / fmax(float(short_effective_n_positions), 1.0f);
        const float long_allowed_wel = allowed_wallet_exposure_limit(
            long_base_limit, long_config.twel,
            coin_override_or(
                long_coin_overrides, c, 12, long_config.allowance_pct
            ),
            long_config.legacy_raw_allowance
        );
        const float short_allowed_wel = allowed_wallet_exposure_limit(
            short_base_limit, short_config.twel,
            coin_override_or(
                short_coin_overrides, c, 12, short_config.allowance_pct
            ),
            short_config.legacy_raw_allowance
        );
        const bool long_min_cost_eligible =
            passes_multicoin_min_effective_cost(
                filter_by_min_effective_cost, guaranteed_balance_lower,
                long_allowed_wel,
                coin_override_or(
                    long_coin_overrides, c, 0, long_config.base_qty_pct
                ),
                coin_settings[coin_offset + 12]
            );
        const bool short_min_cost_eligible =
            passes_multicoin_min_effective_cost(
                filter_by_min_effective_cost, guaranteed_balance_lower,
                short_allowed_wel,
                coin_override_or(
                    short_coin_overrides, c, 0, short_config.base_qty_pct
                ),
                coin_settings[coin_offset + 12]
            );
        const int long_coin_mode = long_config.coin_hsl_mode
            ? hsl_mode(long_side.coin_hsl[c], false) : long_hsl_mode;
        const int short_coin_mode = short_config.coin_hsl_mode
            ? hsl_mode(short_side.coin_hsl[c], false) : short_hsl_mode;
        const bool coin_tradable = k >= int(coin_settings[coin_offset + 8])
            && k <= int(coin_settings[coin_offset + 7])
            && finite_positive(close);
        const bool long_eligible = long_can_generate && coin_tradable
            && long_wel != 0.0f
            && long_min_cost_eligible
            && long_coin_mode == 0;
        const bool short_eligible = short_can_generate && coin_tradable
            && short_wel != 0.0f
            && short_min_cost_eligible
            && short_coin_mode == 0;
        if (long_eligible && !short_eligible) {
            short_order_blocked_mask |= bit;
            continue;
        }
        if (short_eligible && !long_eligible) {
            long_order_blocked_mask |= bit;
            continue;
        }
        if (!long_eligible && !short_eligible) {
            long_order_blocked_mask |= bit;
            short_order_blocked_mask |= bit;
            continue;
        }
        const float long_lower = fmin(
            long_side.ema0[c], fmin(long_side.ema1[c], long_side.ema2[c])
        );
        const float short_upper = fmax(
            short_side.ema0[c], fmax(short_side.ema1[c], short_side.ema2[c])
        );
        const float long_offset = coin_override_or(
            long_coin_overrides, c, 4, long_config.offset
        );
        const float short_offset = coin_override_or(
            short_coin_overrides, c, 4, short_config.offset
        );
        const float dist_long = long_lower * (1.0f - long_offset)
            / close - 1.0f;
        const float dist_short = 1.0f
            - short_upper * (1.0f + short_offset) / close;
        // Exact Rust blocks the side farther from triggering; stable ties
        // favor long.
        if (dist_long >= dist_short) {
            short_order_blocked_mask |= bit;
        } else {
            long_order_blocked_mask |= bit;
        }
    }
}

inline void passivbot_ema_anchor_multicoin_fused_impl(
    constant float* bars,
    constant int* fill_ticks,
    constant int* touch_ticks,
    constant float* hour_log_ranges,
    constant float* coin_settings,
    constant float* long_coin_overrides,
    constant float* short_coin_overrides,
    constant float* params,
    constant float* run_settings,
    constant int* sizes,
    constant int* end_steps,
#if PASSIVBOT_BTC_PRICES_ENABLED
    constant float* btc_prices,
#endif
#if PASSIVBOT_EQUITY_BALANCE_DIFF_ENABLED
    device float* equity_balance_diff,
#endif
    device float* daily,
    device float* scalars,
    device int* gap_hist,
    device float* coin_fill_counts,
#ifdef PASSIVBOT_STRATEGY_EQ_RECOVERY_DISTRIBUTION_ENABLED
    device float* recovery_samples,
#endif
    uint b
) {
    const int B = sizes[0];
    const int T = sizes[1];
    const int C = sizes[2];
    const int D = sizes[3];
    const int requested_start_k = sizes[4];
    const int global_warmup = sizes[5];
    const int start_day_minute = sizes[6];
#ifdef PASSIVBOT_STRATEGY_EQ_RECOVERY_DISTRIBUTION_ENABLED
    const int recovery_stride = sizes[8];
    const int recovery_sample_count = sizes[9];
#endif
    if (b >= uint(B)) return;
    const int stop_k = clamp(end_steps[b], 1, T - 1);
    const int scalar_offset = int(b) * FUSED_SCALAR_COLS;
    for (int j = 0; j < FUSED_SCALAR_COLS; ++j) {
        scalars[scalar_offset + j] = 0.0f;
    }
    for (int j = 0; j < GAP_BINS; ++j) {
        gap_hist[int(b) * GAP_BINS + j] = 0;
    }
    if (C < 1 || C > MAX_COINS) {
        scalars[scalar_offset + 9] = 0.0f;
        scalars[scalar_offset + 13] = 0.0f;
#ifdef PASSIVBOT_STRATEGY_EQ_RECOVERY_DISTRIBUTION_ENABLED
        recovery_samples[int(b) * recovery_sample_count]
            = RECOVERY_FAIL_CLOSED_SENTINEL;
#endif
        return;
    }
    const bool collect_coin_fill_counts = run_settings[6] > 0.5f;
    if (collect_coin_fill_counts) {
        for (int c = 0; c < C; ++c) {
            coin_fill_counts[int(b) * C + c] = 0.0f;
        }
    }

    const int po = int(b) * PARAM_COLS * 2;
    const EmaMulticoinSideConfig long_config =
        load_ema_multicoin_side_config(params, po);
    const EmaMulticoinSideConfig short_config =
        load_ema_multicoin_side_config(params, po + PARAM_COLS);
    const int hsl_signal_mode = long_config.hsl_template.signal_mode;
    const bool topology_valid = hsl_signal_mode >= HSL_SIGNAL_UNIFIED
        && hsl_signal_mode <= HSL_SIGNAL_COIN
        && long_config.hsl_template.signal_mode
            == short_config.hsl_template.signal_mode
        && long_config.coin_hsl_mode == short_config.coin_hsl_mode;
    if (!topology_valid) {
        scalars[scalar_offset + 9] = 0.0f;
        scalars[scalar_offset + 13] = 0.0f;
#ifdef PASSIVBOT_STRATEGY_EQ_RECOVERY_DISTRIBUTION_ENABLED
        recovery_samples[int(b) * recovery_sample_count]
            = RECOVERY_FAIL_CLOSED_SENTINEL;
#endif
        return;
    }

    EmaMulticoinSideState long_side;
    EmaMulticoinSideState short_side;
    init_ema_multicoin_side_state(
        long_side, long_config, coin_settings,
        long_coin_overrides, C
    );
    init_ema_multicoin_side_state(
        short_side, short_config, coin_settings,
        short_coin_overrides, C
    );

    const float starting_balance = run_settings[0];
    const float liquidation_floor = run_settings[1];
    const float interval_ms = run_settings[2];
    const float score_hysteresis = fmax(run_settings[4], 0.0f);
    const float max_realized_loss_pct = run_settings[5];
    const float market_order_slippage_pct = fmax(run_settings[7], 0.0f);
    const bool long_hsl_panic_market = run_settings[8] > 0.5f;
    const bool short_hsl_panic_market = run_settings[9] > 0.5f;
    const bool hedge_mode = run_settings[10] > 0.5f;
    const bool market_orders_allowed = run_settings[11] > 0.5f;
    const float market_order_near_touch_threshold = fmax(
        run_settings[12], 0.0f
    );
    const bool filter_by_min_effective_cost = run_settings[13] > 0.5f;
    const float log_bin_scale = 127.0f / log(4000001.0f);

    JointPortfolioAccount account = init_joint_portfolio_account(
        starting_balance
    );
    EmaMulticoinFillState fills = init_ema_multicoin_fill_state();
    bool alive = true;
    bool equity_started = false;
    bool min_cost_exact_open_uncertain = false;
    float fills_active_days_count = 0.0f;
    int last_active_fill_day = -1;
    float run_peak = -INFINITY;
    float max_dd = 0.0f;
    float total_wallet_exposure_max = 0.0f;
    float total_wallet_exposure_mean = 0.0f;
    float total_wallet_exposure_samples = 0.0f;
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
#if PASSIVBOT_BTC_RISK_ENABLED
    BtcRiskState btc_risk = init_btc_risk_state();
#endif
#if PASSIVBOT_EQUITY_BALANCE_DIFF_ENABLED
    EquityBalanceDiffState equity_balance_diff_state =
        init_equity_balance_diff_state();
#endif
#ifdef PASSIVBOT_STRATEGY_EQ_RECOVERY_DISTRIBUTION_ENABLED
    int recovery_start_k = -1;
#endif
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
    float day_has_fill = 0.0f;
    float day_min_balance = INFINITY;
    float day_start_balance = account.balance;

    for (int k = 1; k < stop_k; ++k) {
        const int day_index = multicoin_utc_day_index(
            start_day_minute, k, interval_ms
        );
        if (day_index != current_day) {
            if (day_touched && current_day >= 0 && current_day < D) {
                int output = (int(b) * D + current_day) * DAILY_COLS;
                daily[output + 0] = day_end;
                daily[output + 1] = day_min;
                daily[output + 2] = day_dd;
                daily[output + 3] = fills.day_volume;
                daily[output + 4] = day_has_fill;
                daily[output + 5] = day_min_balance;
                daily[output + 6] = account.balance - day_start_balance;
                daily[output + 7] = account.balance;
                daily[output + 8] = fills.day_fill_count;
#if PASSIVBOT_BTC_RISK_ENABLED
                write_btc_risk_day(btc_risk, daily, output, 9);
#endif
            }
            current_day = day_index;
            day_touched = false;
            day_end = 0.0f;
            day_min = INFINITY;
            day_dd = 0.0f;
            fills.day_volume = 0.0f;
            day_has_fill = 0.0f;
            day_min_balance = INFINITY;
            day_start_balance = account.balance;
            fills.day_fill_count = 0.0f;
#if PASSIVBOT_BTC_RISK_ENABLED
            reset_btc_risk_day(btc_risk);
#endif
        }

        float long_hsl_equity_before_fills = account.balance;
        long_hsl_equity_before_fills =
            accumulate_ema_multicoin_side_unrealized_pnl(
                long_side, bars, coin_settings, k, C, false,
                long_hsl_equity_before_fills
            );
        long_hsl_equity_before_fills =
            accumulate_ema_multicoin_side_unrealized_pnl(
                short_side, bars, coin_settings, k, C, true,
                long_hsl_equity_before_fills
            );
        bool long_fill = process_ema_multicoin_side_fills(
            long_side, account, fills,
            bars, fill_ticks, coin_settings, long_coin_overrides,
            coin_fill_counts, int(b), k, C, false, alive,
            collect_coin_fill_counts, market_order_slippage_pct,
            long_hsl_panic_market, long_hsl_equity_before_fills
        );
        float short_hsl_equity_before_fills = account.balance;
        short_hsl_equity_before_fills =
            accumulate_ema_multicoin_side_unrealized_pnl(
                long_side, bars, coin_settings, k, C, false,
                short_hsl_equity_before_fills
            );
        short_hsl_equity_before_fills =
            accumulate_ema_multicoin_side_unrealized_pnl(
                short_side, bars, coin_settings, k, C, true,
                short_hsl_equity_before_fills
            );
        bool short_fill = process_ema_multicoin_side_fills(
            short_side, account, fills,
            bars, fill_ticks, coin_settings, short_coin_overrides,
            coin_fill_counts, int(b), k, C, true, alive,
            collect_coin_fill_counts, market_order_slippage_pct,
            short_hsl_panic_market, short_hsl_equity_before_fills
        );
        bool any_fill = long_fill || short_fill;

        update_ema_multicoin_side_indicators(
            long_side, long_config, bars, hour_log_ranges,
            coin_settings, k, C
        );
        update_ema_multicoin_side_indicators(
            short_side, short_config, bars, hour_log_ranges,
            coin_settings, k, C
        );
        int long_tradable_count = count_ema_multicoin_tradable_coins(
            bars, coin_settings, long_coin_overrides, k, C
        );
        int short_tradable_count = count_ema_multicoin_tradable_coins(
            bars, coin_settings, short_coin_overrides, k, C
        );
        const bool post_fill_balance_depleted =
            isfinite(account.balance) && account.balance <= 0.0f;
        const bool past_activation_guard =
            k > max(global_warmup, 1) && k >= requested_start_k;
        if (alive && !post_fill_balance_depleted && past_activation_guard) {
            long_side.max_tradable_seen = max(
                long_side.max_tradable_seen, long_tradable_count
            );
            short_side.max_tradable_seen = max(
                short_side.max_tradable_seen, short_tradable_count
            );
        }
        const int long_effective_n_positions =
            wallet_exposure_denominator_n_positions(
                long_config.n_positions, long_side.max_tradable_seen
            );
        const int short_effective_n_positions =
            wallet_exposure_denominator_n_positions(
                short_config.n_positions, short_side.max_tradable_seen
            );
        const bool long_can_generate = alive
            && long_effective_n_positions > 0
            && long_side.max_tradable_seen > 0 && past_activation_guard;
        const bool short_can_generate = alive
            && short_effective_n_positions > 0
            && short_side.max_tradable_seen > 0 && past_activation_guard;
        equity_started = equity_started
            || long_can_generate || short_can_generate;

        const bool long_has_position =
            ema_multicoin_side_has_position(long_side, C);
        const bool short_has_position =
            ema_multicoin_side_has_position(short_side, C);
        if (long_has_position || short_has_position) {
            min_cost_exact_open_uncertain = true;
        }
        float min_cost_balance_lower =
            min_cost_exact_open_uncertain
            ? 0.0f : liquidation_floor;
        int long_hsl_mode = long_config.coin_hsl_mode ? 0
            : hsl_mode(
                long_side.hsl,
                long_has_position
            );
        int short_hsl_mode = short_config.coin_hsl_mode ? 0
            : hsl_mode(
                short_side.hsl,
                short_has_position
            );
        if (filter_by_min_effective_cost
            && !min_cost_exact_open_uncertain
            && (
                (long_can_generate
                    && multicoin_min_cost_rejection_possible(
                        long_side.psize, long_side.coin_hsl,
                        long_config.coin_hsl_mode, long_hsl_mode,
                        long_config.twel, long_config.allowance_pct,
                        long_config.legacy_raw_allowance,
                        long_config.base_qty_pct,
                        bars, coin_settings, long_coin_overrides, 11, 12, 0,
                        k, C, long_effective_n_positions,
                        min_cost_balance_lower
                    ))
                || (short_can_generate
                    && multicoin_min_cost_rejection_possible(
                        short_side.psize, short_side.coin_hsl,
                        short_config.coin_hsl_mode, short_hsl_mode,
                        short_config.twel, short_config.allowance_pct,
                        short_config.legacy_raw_allowance,
                        short_config.base_qty_pct,
                        bars, coin_settings, short_coin_overrides, 11, 12, 0,
                        k, C, short_effective_n_positions,
                        min_cost_balance_lower
                    ))
            )) {
            min_cost_exact_open_uncertain = true;
            min_cost_balance_lower = 0.0f;
        }
        ulong long_one_way_selection_blocked_mask = 0ul;
        ulong short_one_way_selection_blocked_mask = 0ul;
        ulong long_one_way_order_blocked_mask = 0ul;
        ulong short_one_way_order_blocked_mask = 0ul;
        if (!hedge_mode) {
            compute_ema_multicoin_one_way_initial_blocks(
                long_side, long_config, long_coin_overrides,
                short_side, short_config, short_coin_overrides,
                bars, coin_settings, k, C,
                long_hsl_mode, short_hsl_mode,
                long_effective_n_positions,
                short_effective_n_positions,
                long_can_generate, short_can_generate,
                filter_by_min_effective_cost, min_cost_balance_lower,
                long_one_way_selection_blocked_mask,
                short_one_way_selection_blocked_mask,
                long_one_way_order_blocked_mask,
                short_one_way_order_blocked_mask
            );
        }
        float long_unstuck_diff = INFINITY;
        float short_unstuck_diff = INFINITY;
        const int long_unstuck_candidate = long_can_generate
            ? select_ema_multicoin_unstuck_coin(
                long_side, long_config, account,
                bars, touch_ticks, coin_settings, long_coin_overrides,
                k, C, false, long_effective_n_positions,
                long_unstuck_diff
            )
            : -1;
        const int short_unstuck_candidate = short_can_generate
            ? select_ema_multicoin_unstuck_coin(
                short_side, short_config, account,
                bars, touch_ticks, coin_settings, short_coin_overrides,
                k, C, true, short_effective_n_positions,
                short_unstuck_diff
            )
            : -1;
        // Exact Rust ranks by price difference, then symbol index. Its stable
        // long-first input order makes long win an equal-symbol tie.
        const bool long_unstuck_wins = long_unstuck_candidate >= 0
            && (
                short_unstuck_candidate < 0
                || long_unstuck_diff < short_unstuck_diff
                || (long_unstuck_diff == short_unstuck_diff
                    && long_unstuck_candidate <= short_unstuck_candidate)
            );
        const int long_unstuck_coin = long_unstuck_wins
            ? long_unstuck_candidate : -1;
        const int short_unstuck_coin = !long_unstuck_wins
            ? short_unstuck_candidate : -1;
        if (long_can_generate) {
            update_ema_multicoin_side_selection(
                long_side, long_config, bars, coin_settings,
                long_coin_overrides, k, C, false,
                any_fill || min_cost_exact_open_uncertain,
                long_effective_n_positions, score_hysteresis,
                long_one_way_selection_blocked_mask,
                filter_by_min_effective_cost, min_cost_balance_lower
            );
            generate_ema_multicoin_side_orders(
                long_side, long_config, account,
                bars, touch_ticks, coin_settings, long_coin_overrides,
                k, C, false, long_tradable_count,
                long_effective_n_positions, long_hsl_mode,
                market_orders_allowed, market_order_near_touch_threshold,
                long_unstuck_coin, long_one_way_order_blocked_mask
            );
        }
        if (short_can_generate) {
            update_ema_multicoin_side_selection(
                short_side, short_config, bars, coin_settings,
                short_coin_overrides, k, C, true,
                any_fill || min_cost_exact_open_uncertain,
                short_effective_n_positions, score_hysteresis,
                short_one_way_selection_blocked_mask,
                filter_by_min_effective_cost, min_cost_balance_lower
            );
            generate_ema_multicoin_side_orders(
                short_side, short_config, account,
                bars, touch_ticks, coin_settings, short_coin_overrides,
                k, C, true, short_tradable_count,
                short_effective_n_positions, short_hsl_mode,
                market_orders_allowed, market_order_near_touch_threshold,
                short_unstuck_coin, short_one_way_order_blocked_mask
            );
        }
        if (filter_by_min_effective_cost
            && (long_can_generate || short_can_generate)) {
            min_cost_exact_open_uncertain = true;
        }
        if (long_can_generate || short_can_generate) {
            finalize_ema_multicoin_reducers_fused(
                long_side, short_side, account,
                bars, coin_settings, k, C,
                long_can_generate, short_can_generate,
                market_orders_allowed, market_order_near_touch_threshold,
                market_order_slippage_pct, max_realized_loss_pct
            );
        }

        float forced_delist_equity = account.balance;
        forced_delist_equity =
            accumulate_ema_multicoin_side_unrealized_pnl(
                long_side, bars, coin_settings, k, C, false,
                forced_delist_equity
            );
        forced_delist_equity =
            accumulate_ema_multicoin_side_unrealized_pnl(
                short_side, bars, coin_settings, k, C, true,
                forced_delist_equity
            );
        bool forced_delist_fill = false;
        if (alive && account.balance > 0.0f) {
            forced_delist_fill = force_close_ema_multicoin_delisted_fused(
                long_side, short_side, account, fills,
                bars, coin_settings, coin_fill_counts,
                int(b), k, T, C, collect_coin_fill_counts,
                market_order_slippage_pct, forced_delist_equity
            );
        }
        any_fill = any_fill || forced_delist_fill;
        if (any_fill) {
            day_has_fill = 1.0f;
            if (last_fill_k >= 0.0f) {
                float gap = float(k) - last_fill_k;
                int bin = clamp(
                    int(log(fmax(gap, 0.0f) + 1.0f) * log_bin_scale),
                    0, 127
                );
                gap_hist[int(b) * GAP_BINS + bin] += 1;
                gap_max_min = fmax(gap_max_min, gap);
            }
            if (first_fill_k < 0.0f) first_fill_k = float(k);
            last_fill_k = float(k);
        }

        float long_unrealized = 0.0f;
        float short_unrealized = 0.0f;
        float net_position_cost = 0.0f;
        for (int c = 0; c < C; ++c) {
            int coin_offset = c * COIN_COLS;
            int bar_offset = (k * C + c) * 4;
            float close = bars[bar_offset + 2];
            bool valid = k >= int(coin_settings[coin_offset + 6])
                && k <= int(coin_settings[coin_offset + 7])
                && finite_positive(close);
            bool mark_valid = k >= int(coin_settings[coin_offset + 6])
                && k <= int(coin_settings[coin_offset + 7])
                && isfinite(close);
            float c_mult = coin_settings[coin_offset + 4];
            if (long_side.psize[c] > 0.0f) {
                net_position_cost += long_side.psize[c]
                    * long_side.pprice[c] * c_mult;
                if (mark_valid) {
                    long_unrealized += long_side.psize[c] * c_mult
                        * (close - long_side.pprice[c]);
                }
            }
            if (short_side.psize[c] > 0.0f) {
                net_position_cost -= short_side.psize[c]
                    * short_side.pprice[c] * c_mult;
                if (mark_valid) {
                    short_unrealized += short_side.psize[c] * c_mult
                        * (short_side.pprice[c] - close);
                }
            }
        }
        float equity = joint_portfolio_equity(
            account, long_unrealized, short_unrealized
        );
        // Exact Rust keeps advancing balance-only equity and HSL time once
        // portfolio tracking starts, including declared all-invalid gaps and
        // tails. Per-coin validity still blocks fills, orders, and unrealized
        // PnL above.
        bool can_sample_hsl = (long_can_generate || short_can_generate)
            && alive
            && joint_portfolio_can_generate(
                account, equity, liquidation_floor
            );
        bool hsl_validation_failed = false;
        if (can_sample_hsl) {
            bool sample_enabled = false;
            int sampled_tier = 0;
            bool hsl_valid = update_ema_multicoin_dual_side_hsl(
                long_side, long_config, long_effective_n_positions,
                short_side, short_config, short_effective_n_positions,
                account, bars, coin_settings, k, day_index, C,
                starting_balance, interval_ms,
                sample_enabled, sampled_tier
            );
            if (!hsl_valid) {
                account.balance = 0.0f;
                alive = false;
                liquidation_day = day_index;
                hsl_validation_failed = true;
            } else if (sample_enabled) {
                hsl_tier_samples_total += 1.0f;
                hsl_tier_samples_yellow +=
                    sampled_tier == 1 ? 1.0f : 0.0f;
                hsl_tier_samples_orange +=
                    sampled_tier == 2 ? 1.0f : 0.0f;
                hsl_tier_samples_red +=
                    sampled_tier == 3 ? 1.0f : 0.0f;
            }
        }

        bool active = equity_started && (alive || hsl_validation_failed);
        if (active) {
            if (first_eq_k < 0.0f) first_eq_k = float(k);
            last_eq_k = float(k);
            if (any_fill) {
                int active_fill_day = multicoin_active_fill_day(
                    k, int(first_eq_k), interval_ms
                );
                if (active_fill_day != last_active_fill_day) {
                    fills_active_days_count += 1.0f;
                    last_active_fill_day = active_fill_day;
                }
            }
            bool liquidated = account.balance <= 0.0f
                || equity <= liquidation_floor;
            float effective_equity = liquidated
                ? liquidation_floor : equity;
#ifdef PASSIVBOT_STRATEGY_EQ_RECOVERY_DISTRIBUTION_ENABLED
            if (recovery_stride > 0 && recovery_start_k < 0) {
                recovery_start_k = k;
                recovery_samples[int(b) * recovery_sample_count]
                    = effective_equity;
            } else if (recovery_stride > 0) {
                const int recovery_elapsed = k - recovery_start_k;
                const bool recovery_terminal = liquidated || k == stop_k - 1;
                const bool recovery_regular =
                    recovery_elapsed % recovery_stride == 0;
                if (recovery_regular || recovery_terminal) {
                    const int sample_index = recovery_terminal
                        ? (recovery_elapsed + recovery_stride - 1)
                            / recovery_stride
                        : recovery_elapsed / recovery_stride;
                    if (sample_index < recovery_sample_count) {
                        recovery_samples[
                            int(b) * recovery_sample_count + sample_index
                        ] = effective_equity;
                    }
                }
            }
#endif
            if (effective_equity >= account_peak) {
                if (account_peak_k >= 0.0f) {
                    account_recovery_max_min = fmax(
                        account_recovery_max_min,
                        float(k) - account_peak_k
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
                (run_peak - effective_equity)
                    / fmax(fabs(run_peak), 1.0e-12f),
                0.0f
            );
            max_dd = fmax(max_dd, drawdown);
            day_end = effective_equity;
            day_min = fmin(day_min, effective_equity);
            day_min_balance = fmin(day_min_balance, account.balance);
            day_dd = fmax(day_dd, drawdown);
#if PASSIVBOT_BTC_RISK_ENABLED
            update_btc_risk_state(
                btc_risk, effective_equity, btc_prices[k]
            );
#endif
#if PASSIVBOT_EQUITY_BALANCE_DIFF_ENABLED
            update_equity_balance_diff_state(
                equity_balance_diff_state,
                account.balance,
                effective_equity,
                btc_prices,
                k,
                starting_balance,
                any_fill
            );
#endif
            day_touched = true;
            if (!liquidated) {
                float twe_abs = fabs(net_position_cost / account.balance);
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
        daily[output + 3] = fills.day_volume;
        daily[output + 4] = day_has_fill;
        daily[output + 5] = day_min_balance;
        daily[output + 6] = account.balance - day_start_balance;
        daily[output + 7] = account.balance;
        daily[output + 8] = fills.day_fill_count;
#if PASSIVBOT_BTC_RISK_ENABLED
        write_btc_risk_day(btc_risk, daily, output, 9);
#endif
    }

    float long_total_size = 0.0f;
    float long_total_cost = 0.0f;
    float short_total_size = 0.0f;
    float short_total_cost = 0.0f;
    int open_positions = 0;
    for (int c = 0; c < C; ++c) {
        float c_mult = coin_settings[c * COIN_COLS + 4];
        if (long_side.psize[c] > 0.0f) {
            long_total_size += long_side.psize[c];
            long_total_cost += long_side.psize[c]
                * long_side.pprice[c] * c_mult;
            open_positions += 1;
            if (long_side.position_open_k[c] >= 0.0f
                && last_eq_k >= 0.0f) {
                float held_min = last_eq_k
                    - long_side.position_open_k[c];
                fills.held_max_min = fmax(fills.held_max_min, held_min);
                fills.held_sum_min += held_min;
                fills.held_count += 1.0f;
            }
            if (long_side.position_last_fill_k[c] >= 0.0f
                && last_eq_k >= 0.0f) {
                fills.position_unchanged_max_min = fmax(
                    fills.position_unchanged_max_min,
                    last_eq_k - long_side.position_last_fill_k[c]
                );
            }
        }
        if (short_side.psize[c] > 0.0f) {
            short_total_size += short_side.psize[c];
            short_total_cost += short_side.psize[c]
                * short_side.pprice[c] * c_mult;
            open_positions += 1;
            if (short_side.position_open_k[c] >= 0.0f
                && last_eq_k >= 0.0f) {
                float held_min = last_eq_k
                    - short_side.position_open_k[c];
                fills.held_max_min = fmax(fills.held_max_min, held_min);
                fills.held_sum_min += held_min;
                fills.held_count += 1.0f;
            }
            if (short_side.position_last_fill_k[c] >= 0.0f
                && last_eq_k >= 0.0f) {
                fills.position_unchanged_max_min = fmax(
                    fills.position_unchanged_max_min,
                    last_eq_k - short_side.position_last_fill_k[c]
                );
            }
        }
    }
    if (fills.pnl_recovery_peak_k >= 0.0f && last_eq_k >= 0.0f) {
        fills.pnl_recovery_max_min = fmax(
            fills.pnl_recovery_max_min,
            last_eq_k - fills.pnl_recovery_peak_k
        );
    }


    scalars[scalar_offset + 0] = max_dd;
    scalars[scalar_offset + 1] = fills.held_max_min * interval_ms;
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
    scalars[scalar_offset + 10] = account.balance;
    scalars[scalar_offset + 11] = long_total_size;
    scalars[scalar_offset + 12] = long_total_cost;
    scalars[scalar_offset + 13] = alive ? 1.0f : 0.0f;
    scalars[scalar_offset + 14] = float(open_positions);
    scalars[scalar_offset + 15] = short_total_size;
    scalars[scalar_offset + 16] = short_total_cost;
    // Derive portfolio gross-PnL outputs from their directional reductions so
    // the public total/directional metric contract stays algebraically exact
    // despite different float32 accumulation orders.
    scalars[scalar_offset + 18] =
        fills.profit_sum_long + fills.profit_sum_short;
    scalars[scalar_offset + 19] =
        fills.loss_sum_long + fills.loss_sum_short;
    scalars[scalar_offset + 20] =
        fills.position_unchanged_max_min * interval_ms;
    scalars[scalar_offset + 21] = ema_multicoin_entry_initial_balance_pct(
        long_config, long_coin_overrides,
        wallet_exposure_denominator_n_positions(
            long_config.n_positions, long_side.max_tradable_seen
        )
    );
    scalars[scalar_offset + 22] = total_wallet_exposure_max;
    scalars[scalar_offset + 23] = total_wallet_exposure_mean;
    scalars[scalar_offset + 24] = fills.fill_count;
    scalars[scalar_offset + 25] = fills.fill_count_entry;
    scalars[scalar_offset + 26] = fills.fill_count_long;
    scalars[scalar_offset + 27] = fills_active_days_count;
    scalars[scalar_offset + 28] =
        fills.pnl_recovery_max_min * interval_ms;
    scalars[scalar_offset + 29] = fills.held_sum_min * interval_ms;
    scalars[scalar_offset + 30] = fills.held_count;
    scalars[scalar_offset + 31] = account_recovery_max_min * interval_ms;
    if (long_config.coin_hsl_mode) {
        write_dual_side_coin_hsl_outputs(
            long_side.coin_hsl, short_side.coin_hsl, C,
            hsl_tier_samples_total,
            hsl_tier_samples_yellow,
            hsl_tier_samples_orange,
            hsl_tier_samples_red,
            last_eq_k, scalars, scalar_offset + 32
        );
    } else {
        write_dual_side_hsl_outputs(
            long_side.hsl, short_side.hsl,
            hsl_tier_samples_total,
            hsl_tier_samples_yellow,
            hsl_tier_samples_orange,
            hsl_tier_samples_red,
            last_eq_k, scalars, scalar_offset + 32
        );
    }
    scalars[scalar_offset + 59] = ema_multicoin_entry_initial_balance_pct(
        short_config, short_coin_overrides,
        wallet_exposure_denominator_n_positions(
            short_config.n_positions, short_side.max_tradable_seen
        )
    );
    scalars[scalar_offset + 60] = fills.profit_sum_long;
    scalars[scalar_offset + 61] = fills.loss_sum_long;
    scalars[scalar_offset + 62] = fills.profit_sum_short;
    scalars[scalar_offset + 63] = fills.loss_sum_short;
    scalars[scalar_offset + 64] = hsl_strategy_equity_recovery_max_steps(
        long_side.hsl_strategy_eq
    ) * interval_ms;
    scalars[scalar_offset + 65] = hsl_strategy_equity_recovery_max_steps(
        short_side.hsl_strategy_eq
    ) * interval_ms;
#if PASSIVBOT_HSL_EMA_TAIL_ENABLED
    scalars[scalar_offset + 66] = hsl_drawdown_ema_mean_worst_1pct(
        long_side.hsl_ema_tail
    );
    scalars[scalar_offset + 67] = hsl_drawdown_ema_mean_worst_1pct(
        short_side.hsl_ema_tail
    );
#endif
#if PASSIVBOT_HSL_RAW_DRAWDOWN_ENABLED
    scalars[scalar_offset + 68] = hsl_strategy_equity_drawdown_max(
        long_side.hsl_strategy_eq
    );
    scalars[scalar_offset + 69] = hsl_strategy_equity_drawdown_max(
        short_side.hsl_strategy_eq
    );
#endif
#if PASSIVBOT_HSL_RAW_TAIL_ENABLED
    scalars[scalar_offset + 70]
        = hsl_strategy_equity_drawdown_mean_worst_1pct(
            long_side.hsl_strategy_eq
        );
    scalars[scalar_offset + 71]
        = hsl_strategy_equity_drawdown_mean_worst_1pct(
            short_side.hsl_strategy_eq
        );
#endif
#if PASSIVBOT_EQUITY_BALANCE_DIFF_ENABLED
    write_equity_balance_diff_state(
        equity_balance_diff_state, equity_balance_diff, b
    );
#endif
}

kernel void passivbot_ema_anchor_multicoin_fused(
    constant float* bars,
    constant int* fill_ticks,
    constant int* touch_ticks,
    constant float* hour_log_ranges,
    constant float* coin_settings,
    constant float* long_coin_overrides,
    constant float* short_coin_overrides,
    constant float* params,
    constant float* run_settings,
    constant int* sizes,
    constant int* end_steps,
#if PASSIVBOT_BTC_PRICES_ENABLED
    constant float* btc_prices,
#endif
#if PASSIVBOT_EQUITY_BALANCE_DIFF_ENABLED
    device float* equity_balance_diff,
#endif
    device float* daily,
    device float* scalars,
    device int* gap_hist,
    device float* coin_fill_counts,
#ifdef PASSIVBOT_STRATEGY_EQ_RECOVERY_DISTRIBUTION_ENABLED
    device float* recovery_samples,
#endif
    uint b [[thread_position_in_grid]]
) {
    passivbot_ema_anchor_multicoin_fused_impl(
        bars, fill_ticks, touch_ticks, hour_log_ranges, coin_settings,
        long_coin_overrides, short_coin_overrides,
        params, run_settings, sizes, end_steps,
#if PASSIVBOT_BTC_PRICES_ENABLED
        btc_prices,
#endif
#if PASSIVBOT_EQUITY_BALANCE_DIFF_ENABLED
        equity_balance_diff,
#endif
        daily, scalars, gap_hist, coin_fill_counts,
#ifdef PASSIVBOT_STRATEGY_EQ_RECOVERY_DISTRIBUTION_ENABLED
        recovery_samples,
#endif
        b
    );
}

kernel void passivbot_ema_anchor_multicoin(
    constant float* bars,
    constant int* fill_ticks,
    constant int* touch_ticks,
    constant float* hour_log_ranges,
    constant float* coin_settings,
    constant float* coin_overrides,
    constant float* params,
    constant float* run_settings,
    constant int* sizes,
    constant int* end_steps,
#if PASSIVBOT_BTC_PRICES_ENABLED
    constant float* btc_prices,
#endif
#if PASSIVBOT_EQUITY_BALANCE_DIFF_ENABLED
    device float* equity_balance_diff,
#endif
    device float* daily,
    device float* scalars,
    device int* gap_hist,
    device float* coin_fill_counts,
#ifdef PASSIVBOT_STRATEGY_EQ_RECOVERY_DISTRIBUTION_ENABLED
    device float* recovery_samples,
#endif
    uint b [[thread_position_in_grid]]
) {
    const bool short_side = run_settings[3] > 0.5f;
    passivbot_ema_anchor_multicoin_impl(
        bars, fill_ticks, touch_ticks, hour_log_ranges,
        coin_settings, coin_overrides, params, run_settings,
        sizes, end_steps,
#if PASSIVBOT_BTC_PRICES_ENABLED
        btc_prices,
#endif
#if PASSIVBOT_EQUITY_BALANCE_DIFF_ENABLED
        equity_balance_diff,
#endif
        daily, scalars, gap_hist, coin_fill_counts,
#ifdef PASSIVBOT_STRATEGY_EQ_RECOVERY_DISTRIBUTION_ENABLED
        recovery_samples,
#endif
        b, short_side
    );
}

kernel void passivbot_ema_anchor_multicoin_long(
    constant float* bars,
    constant int* fill_ticks,
    constant int* touch_ticks,
    constant float* hour_log_ranges,
    constant float* coin_settings,
    constant float* coin_overrides,
    constant float* params,
    constant float* run_settings,
    constant int* sizes,
    constant int* end_steps,
#if PASSIVBOT_BTC_PRICES_ENABLED
    constant float* btc_prices,
#endif
#if PASSIVBOT_EQUITY_BALANCE_DIFF_ENABLED
    device float* equity_balance_diff,
#endif
    device float* daily,
    device float* scalars,
    device int* gap_hist,
    device float* coin_fill_counts,
#ifdef PASSIVBOT_STRATEGY_EQ_RECOVERY_DISTRIBUTION_ENABLED
    device float* recovery_samples,
#endif
    uint b [[thread_position_in_grid]]
) {
    passivbot_ema_anchor_multicoin_impl(
        bars, fill_ticks, touch_ticks, hour_log_ranges,
        coin_settings, coin_overrides, params, run_settings,
        sizes, end_steps,
#if PASSIVBOT_BTC_PRICES_ENABLED
        btc_prices,
#endif
#if PASSIVBOT_EQUITY_BALANCE_DIFF_ENABLED
        equity_balance_diff,
#endif
        daily, scalars, gap_hist, coin_fill_counts,
#ifdef PASSIVBOT_STRATEGY_EQ_RECOVERY_DISTRIBUTION_ENABLED
        recovery_samples,
#endif
        b, false
    );
}
