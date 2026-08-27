#include <metal_stdlib>
using namespace metal;

constant int MAX_COINS = 64;
constant int PARAM_COLS = 59;
constant int OVERRIDE_COLS = 47;
constant int HSL_OVERRIDE_START = 34;
constant int GATE_INITIAL_OVERRIDE_COL = 44;
constant int GATE_REENTRY_OVERRIDE_COL = 45;
constant int FORCED_ACTIVE_OVERRIDE_COL = 46;
constant int COIN_COLS = 13;
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
    // Enumerating and reserving every recursive TM close group across all
    // independently dispatched coins and sides would make screening scale
    // with the 500-rung exact bound. Use a conservative zero-loss envelope
    // whenever Rust's configured loss gate is active.
    float arithmetic_scale = fabs(gross_pnl) + fabs(fee)
        + qty * fabs(c_mult) * (fabs(close_price) + fabs(pprice));
    float margin = 1.220703125e-4f * arithmetic_scale;
    return isfinite(net_pnl) && net_pnl > margin;
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

inline float finalized_reducer_qty_with_ordinary(
    float psize,
    float reducer_qty,
    float reducer_price,
    float ordinary_qty,
    float ordinary_min,
    int ordinary_min_relation,
    float qty_step,
    float min_qty,
    float min_cost,
    float c_mult
) {
    if (reducer_qty <= 0.0f || reducer_price <= 0.0f) return 0.0f;
    reducer_qty = fmin(psize, reducer_qty);
    float reducer_min = min_entry_qty(
        reducer_price, qty_step, min_qty, min_cost, c_mult
    );
    if (ordinary_qty > 0.0f) {
        if (ordinary_qty + reducer_qty > psize) {
            ordinary_qty = fmax(
                round_step(psize - reducer_qty, qty_step), 0.0f
            );
        }
        bool ordinary_below_minimum = ordinary_qty < ordinary_min
            || (ordinary_qty == ordinary_min && ordinary_min_relation > 0);
        if (!ordinary_below_minimum) {
            float remainder = fmax(
                round_step(psize - reducer_qty - ordinary_qty, qty_step),
                0.0f
            );
            float minimum_any = fmin(ordinary_min, reducer_min);
            if (remainder > 0.0f && remainder < minimum_any) {
                ordinary_qty = fmin(
                    psize - reducer_qty,
                    round_step(ordinary_qty + remainder, qty_step)
                );
            }
            if (ordinary_qty > 0.0f) return reducer_qty;
        }
    }
    float remainder = fmax(
        round_step(psize - reducer_qty, qty_step), 0.0f
    );
    return remainder > 0.0f && remainder < reducer_min
        ? psize : reducer_qty;
}

inline bool reducer_candidate_preferred(
    float left_qty, int left_ticks, int left_order_type_id,
    float right_qty, int right_ticks, int right_order_type_id,
    bool is_long
) {
    if (!(left_qty > 0.0f)) return false;
    if (!(right_qty > 0.0f)) return true;
    if (left_qty != right_qty) return left_qty > right_qty;
    if (left_ticks != right_ticks) {
        return is_long ? left_ticks < right_ticks : left_ticks > right_ticks;
    }
    return left_order_type_id < right_order_type_id;
}

struct CloseGroup {
    int ticks;
    float price;
    float qty;
    bool market;
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
    int prefix_merge_tick,
    float prefix_merge_qty,
    int max_rungs,
    int wanted_group,
    float generation_market_price,
    bool market_orders_allowed,
    float market_order_near_touch_threshold,
    float market_resize_psize,
    thread CloseGroup& selected
) {
    selected.ticks = 0;
    selected.price = 0.0f;
    selected.qty = 0.0f;
    selected.market = false;
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
                selected.qty = group_count == 0
                        && group_ticks == prefix_merge_tick
                    ? round_step(group_qty + prefix_merge_qty, qty_step)
                    : group_qty;
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
            selected.qty = group_count == 0
                    && group_ticks == prefix_merge_tick
                ? round_step(group_qty + prefix_merge_qty, qty_step)
                : group_qty;
        }
        ++group_count;
    }
    if (selected.qty > 0.0f && selected.ticks > 0) {
        selected.market = should_use_ordinary_market_execution(
            selected.ticks, short_side, generation_market_price,
            price_step, market_orders_allowed,
            market_order_near_touch_threshold
        );
        if (selected.market) {
            selected.qty = resize_market_close_qty(
                selected.qty, market_resize_psize,
                generation_market_price,
                qty_step, min_qty, min_cost, c_mult
            );
        }
    }
    return group_count;
}

// Exact Rust decides whether to expose a recursive close suffix from the
// immutable passive strategy ladder. Market promotion is intentionally absent
// here: a market-only next close must fill by itself and must not reveal later
// recursive rungs.
inline bool recursive_grid_close_would_expand(
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
    int high_fill_max_tick,
    int low_nonfill_max_tick
) {
    float sim_psize = psize;
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
            ? close_touch < target_tick : close_touch > target_tick;
        int order_tick = touch_controls ? touch_nearest : target_tick;
        float order_price = float(order_tick) * price_step;
        float minimum_close = touch_controls
            ? touch_min_qty
            : min_entry_qty(
                order_price, qty_step, min_qty, min_cost, c_mult
            );
        int minimum_relation = touch_controls ? touch_min_qty_relation : 0;
        float close_pct = close_threshold_we == 0.0f
            ? 1.0f : close_qty_pct;
        float order_qty = round_step(
            calc_close_qty(
                sim_psize, pprice, generation_balance, allowed_wel,
                minimum_close, minimum_relation, close_pct,
                qty_step, c_mult
            ),
            qty_step
        );
        if (order_qty <= 0.0f || order_tick <= 0) break;
        bool reachable = short_side
            ? order_tick > low_nonfill_max_tick
            : order_tick <= high_fill_max_tick;
        if (reachable) return true;
        sim_psize = fmax(
            round_step(sim_psize - fmin(order_qty, sim_psize), qty_step),
            0.0f
        );
    }
    return false;
}

// One complete directional Trailing Martingale portfolio. Keeping the mutable
// per-coin state behind one thread-local value lets a future fused kernel own
// long and short portfolios concurrently without changing the proven one-side
// candle loop.
struct TrailingMartingaleMulticoinSideState {
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
    float entry_strategy_qty[MAX_COINS];
    float entry_gen_balance[MAX_COINS];
    float entry_gen_allowed_wel[MAX_COINS];
    float entry_gen_market_price[MAX_COINS];
    float entry_gen_psize[MAX_COINS];
    float entry_gen_pprice[MAX_COINS];
    float entry_gate_suffix_partial_qty[MAX_COINS];
    float close_qty[MAX_COINS];
    float secondary_close_qty[MAX_COINS];
    float twel_close_qty[MAX_COINS];
    float unstuck_close_qty[MAX_COINS];
    float close_gen_balance[MAX_COINS];
    float close_gen_allowed_wel[MAX_COINS];
    float close_gen_market_price[MAX_COINS];
    float close_grid_gen_psize[MAX_COINS];
    float close_grid_prefix_qty[MAX_COINS];
    float position_open_k[MAX_COINS];
    float position_last_fill_k[MAX_COINS];
#if PASSIVBOT_ENTRY_INTERVAL_ENABLED
    float last_initial_entry_k[MAX_COINS];
#endif
    float score[MAX_COINS];
    float contribution[MAX_COINS];
    float minimum_entry[MAX_COINS];
    float min_since_open[MAX_COINS];
    float max_since_min[MAX_COINS];
    float max_since_open[MAX_COINS];
    float min_since_max[MAX_COINS];
    int entry_tick[MAX_COINS];
    int entry_gen_initial_tick[MAX_COINS];
    int entry_gen_touch_tick[MAX_COINS];
    int entry_order_type[MAX_COINS];
    int entry_gate_suffix_keep_count[MAX_COINS];
    int entry_gate_suffix_partial_rank[MAX_COINS];
    int close_tick[MAX_COINS];
    int secondary_close_tick[MAX_COINS];
    int twel_close_tick[MAX_COINS];
    int unstuck_close_tick[MAX_COINS];
    int close_grid_max_rungs[MAX_COINS];
    int close_grid_prefix_tick[MAX_COINS];
    bool selected[MAX_COINS];
    bool incumbent[MAX_COINS];
    bool survivor[MAX_COINS];
    bool entry_candidate[MAX_COINS];
    bool entry_recursive_market_mode[MAX_COINS];
    bool close_reconstruct_after_reducer[MAX_COINS];
    bool close_recursive_market_mode[MAX_COINS];
    bool filled_coin[MAX_COINS];
    bool entry_market[MAX_COINS];
    bool close_market[MAX_COINS];
    bool secondary_close_market[MAX_COINS];
    bool close_is_exposure_reducer[MAX_COINS];
    bool close_is_unstuck_reducer[MAX_COINS];
    bool close_is_hsl_panic[MAX_COINS];
    bool entry_deferred_twel_gate;
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

// Immutable decoded parameters for one directional TM portfolio. A fused
// kernel can load this twice from adjacent parameter rows while sharing only
// the explicit account state.
struct TrailingMartingaleMulticoinSideConfig {
    float span_a;
    float span_b;
    float span_1h;
    float span_1m;
    float ddf;
    float initial_ema_dist;
    float initial_qty_pct;
    float entry_threshold_base;
    float entry_threshold_we;
    float entry_threshold_v1h;
    float entry_threshold_v1m;
    float entry_retracement_base;
    float entry_retracement_we;
    float entry_retracement_v1h;
    float entry_retracement_v1m;
    float close_qty_pct;
    float close_threshold_base;
    float close_threshold_we;
    float close_threshold_v1h;
    float close_threshold_v1m;
    float close_retracement_base;
    float close_retracement_v1h;
    float close_retracement_v1m;
    float cooldown_min;
    float twel;
    bool gate_initial;
    bool gate_reentry;
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
    bool wel_enforcer_enabled;
    float wel_enforcer_threshold;
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

struct RecursiveEntryCandidate {
    int ticks;
    int order_type;
    float price;
    float strategy_qty;
    float executable_qty;
    bool market;
};

// Rebuild one recursive grid reentry from an immutable generation snapshot.
// Portfolio TWEL gating is intentionally outside this helper; the baseline
// multi-coin recursive-entry slice fails closed while that gate is enabled.
inline RecursiveEntryCandidate next_recursive_grid_entry(
    thread const TrailingMartingaleMulticoinSideState& side,
    thread const TrailingMartingaleMulticoinSideConfig& config,
    constant float* coin_overrides,
    int coin,
    bool short_side,
    float sim_psize,
    float sim_pprice,
    float generation_balance,
    float allowed_wel,
    int generation_initial_tick,
    int generation_touch_tick,
    float generation_market_price,
    float qty_step,
    float price_step,
    float min_qty,
    float min_cost,
    float c_mult,
    bool market_orders_allowed,
    float market_order_near_touch_threshold
) {
    RecursiveEntryCandidate out;
    out.ticks = 0;
    out.order_type = short_side ? 15 : 4;
    out.price = 0.0f;
    out.strategy_qty = 0.0f;
    out.executable_qty = 0.0f;
    out.market = false;
    if (!(sim_psize > 0.0f && sim_pprice > 0.0f
        && generation_balance > 0.0f && allowed_wel > 0.0f)) return out;

    float initial_ema_dist = coin_override_or(
        coin_overrides, coin, 5, config.initial_ema_dist
    );
    float initial_qty_pct = coin_override_or(
        coin_overrides, coin, 6, config.initial_qty_pct
    );
    float threshold_base = coin_override_or(
        coin_overrides, coin, 7, config.entry_threshold_base
    );
    float threshold_we = coin_override_or(
        coin_overrides, coin, 8, config.entry_threshold_we
    );
    float threshold_v1h = coin_override_or(
        coin_overrides, coin, 9, config.entry_threshold_v1h
    );
    float threshold_v1m = coin_override_or(
        coin_overrides, coin, 10, config.entry_threshold_v1m
    );
    float ddf = coin_override_or(coin_overrides, coin, 4, config.ddf);
    float band = short_side
        ? fmax(side.ema0[coin], fmax(side.ema1[coin], side.ema2[coin]))
        : fmin(side.ema0[coin], fmin(side.ema1[coin], side.ema2[coin]));
    int band_tick = short_side
        ? int(ceil(
            band * (1.0f + initial_ema_dist) / price_step - 1.0e-6f
        ))
        : int(floor(
            band * (1.0f - initial_ema_dist) / price_step + 1.0e-6f
        ));
    int initial_tick = generation_initial_tick;
    float initial_price = float(initial_tick) * price_step;
    float min_iq = min_entry_qty(
        initial_price, qty_step, min_qty, min_cost, c_mult
    );
    float iq = fmax(min_iq, round_step(
        generation_balance * allowed_wel * initial_qty_pct
            / fmax(initial_price * c_mult, 1.0e-12f),
        qty_step
    ));
    float iq_effective = sim_psize < iq
        ? fmax(round_step(sim_psize, qty_step), min_iq) : iq;
    float we = sim_psize * sim_pprice * c_mult / generation_balance;
    if (we >= allowed_wel * 0.999f) return out;
    float wer = we / fmax(allowed_wel, 1.0e-12f);
    float multiplier = fmax(
        1.0f + side.volatility_1h[coin] * threshold_v1h
            + side.volatility_1m[coin] * threshold_v1m
            + wer * threshold_we,
        1.0f
    );
    float threshold = fmax(threshold_base, 0.0f) * multiplier;
    float target = sim_pprice * (
        short_side ? 1.0f + threshold : 1.0f - threshold
    );
    int raw_tick = short_side
        ? int(ceil(target / price_step - 1.0e-6f))
        : int(floor(target / price_step + 1.0e-6f));
    bool touch_controls = short_side
        ? generation_touch_tick >= raw_tick
        : generation_touch_tick <= raw_tick;
    int entry_tick = touch_controls ? generation_touch_tick : raw_tick;
    bool coin_gate_reentry = coin_override_or(
        coin_overrides, coin, GATE_REENTRY_OVERRIDE_COL,
        config.gate_reentry ? 1.0f : 0.0f
    ) > 0.5f;
    if (coin_gate_reentry) {
        bool band_controls = short_side
            ? band_tick >= entry_tick : band_tick <= entry_tick;
        if (band_controls) entry_tick = band_tick;
    }
    if (entry_tick <= 1) return out;
    float entry_price = float(entry_tick) * price_step;
    float min_rq = min_entry_qty(
        entry_price, qty_step, min_qty, min_cost, c_mult
    );
    float rq = fmax(iq_effective, fmax(min_rq, round_step(
        fmax(
            sim_psize * ddf,
            generation_balance * allowed_wel * initial_qty_pct
                / fmax(entry_price * c_mult, 1.0e-12f)
        ),
        qty_step
    )));
    float uncropped_rq = rq;
    float we_if = (sim_psize * sim_pprice + rq * entry_price)
        * c_mult / fmax(generation_balance, 1.0e-9f);
    float crop_fraction = (allowed_wel - we)
        / fmax(we_if - we, 1.0e-12f);
    float rq_crop = fmax(
        round_step(rq * crop_fraction, qty_step), min_rq
    );
    if (we_if > allowed_wel * 1.01f && rq_crop < rq) rq = rq_crop;
    float headroom = (
        allowed_wel * generation_balance - sim_psize * sim_pprice * c_mult
    ) / fmax(entry_price * c_mult, 1.0e-12f);
    if ((sim_psize * sim_pprice + rq * entry_price) * c_mult
        / fmax(generation_balance, 1.0e-9f) > allowed_wel * 1.01f) {
        rq = fmin(rq, fmax(floor_step(headroom, qty_step), 0.0f));
    }
    if (rq * (1.0f + 1.0e-6f) < min_rq) return out;

    out.ticks = entry_tick;
    bool cropped = rq < uncropped_rq;
    out.order_type = short_side
        ? (cropped ? 16 : 15) : (cropped ? 5 : 4);
    out.price = entry_price;
    out.strategy_qty = rq;
    out.executable_qty = rq;
    out.market = should_use_ordinary_market_execution(
        entry_tick, !short_side, generation_market_price, price_step,
        market_orders_allowed, market_order_near_touch_threshold
    );
    if (short_side && out.market) {
        out.executable_qty = fmax(
            out.executable_qty,
            min_entry_qty(
                generation_market_price, qty_step, min_qty, min_cost, c_mult
            )
        );
    }
    return out;
}

inline bool recursive_entry_gate_candidate_preferred(
    thread const RecursiveEntryCandidate& candidate,
    int coin,
    float market_price,
    thread const RecursiveEntryCandidate& incumbent,
    int incumbent_coin,
    float incumbent_market_price,
    bool short_side
) {
    if (incumbent_coin < 0) return true;
    float distance = short_side
        ? candidate.price / fmax(market_price, 1.0e-12f) - 1.0f
        : 1.0f - candidate.price / fmax(market_price, 1.0e-12f);
    float incumbent_distance = short_side
        ? incumbent.price / fmax(incumbent_market_price, 1.0e-12f) - 1.0f
        : 1.0f - incumbent.price
            / fmax(incumbent_market_price, 1.0e-12f);
    if (distance != incumbent_distance) return distance < incumbent_distance;
    if (coin != incumbent_coin) return coin > incumbent_coin;
    if (candidate.order_type != incumbent.order_type) {
        return candidate.order_type > incumbent.order_type;
    }
    if (candidate.price != incumbent.price) {
        return candidate.price > incumbent.price;
    }
    return candidate.executable_qty > incumbent.executable_qty;
}

// Exact Rust flattens every per-symbol entry ladder, removes globally farthest
// entries first, and may retain one partial boundary.  Rung zero may be out of
// distance order relative to its recursive suffix, so model it as a singleton
// alongside one monotonic suffix stream per coin.  A closest-first k-way merge
// is the reverse of Rust's deterministic removal order and needs only bounded
// per-coin private state instead of a 500 * MAX_COINS candidate array.
inline void apply_tm_multicoin_recursive_entry_twel_gate(
    thread TrailingMartingaleMulticoinSideState& side,
    thread const TrailingMartingaleMulticoinSideConfig& config,
    constant float* bars,
    constant int* fill_ticks,
    constant float* coin_settings,
    constant float* coin_overrides,
    int k,
    int C,
    bool short_side,
    float market_order_near_touch_threshold
) {
    if (!side.entry_deferred_twel_gate) return;

    RecursiveEntryCandidate first[MAX_COINS];
    RecursiveEntryCandidate suffix[MAX_COINS];
    float sim_psize[MAX_COINS];
    float sim_pprice[MAX_COINS];
    int sim_touch_tick[MAX_COINS];
    int previous_tick[MAX_COINS];
    int suffix_rank[MAX_COINS];
    bool first_pending[MAX_COINS];
    bool suffix_valid[MAX_COINS];
    float generation_balance = 0.0f;
    float current_cost = 0.0f;

    for (int c = 0; c < C; ++c) {
        int coin_offset = c * COIN_COLS;
        float c_mult = coin_settings[coin_offset + 4];
        current_cost += side.psize[c] * side.pprice[c] * c_mult;
        if (!(generation_balance > 0.0f)
            && side.entry_gen_balance[c] > 0.0f) {
            generation_balance = side.entry_gen_balance[c];
        }
        first[c].ticks = side.entry_tick[c];
        first[c].order_type = side.entry_order_type[c];
        first[c].price = float(side.entry_tick[c])
            * coin_settings[coin_offset + 1];
        first[c].strategy_qty = side.entry_strategy_qty[c];
        first[c].executable_qty = side.entry_qty[c];
        first[c].market = side.entry_market[c];
        first_pending[c] = first[c].ticks > 1
            && first[c].strategy_qty > 0.0f
            && first[c].executable_qty > 0.0f;
        side.entry_qty[c] = 0.0f;
        side.entry_gate_suffix_keep_count[c] = 0;
        side.entry_gate_suffix_partial_rank[c] = -1;
        side.entry_gate_suffix_partial_qty[c] = 0.0f;
        suffix_valid[c] = false;
        suffix_rank[c] = 0;
        sim_psize[c] = side.entry_gen_psize[c];
        sim_pprice[c] = side.entry_gen_pprice[c];
        sim_touch_tick[c] = side.entry_gen_touch_tick[c];
        previous_tick[c] = first[c].ticks;
        if (!first_pending[c] || !side.entry_recursive_market_mode[c]) {
            continue;
        }
        // Exact Rust retains the already generated next entry in the global
        // TWEL gate, but NextCandle.tradable=false prevents recursively
        // expanding its market-fill suffix outside this coin's valid window.
        const int first_valid = int(coin_settings[coin_offset + 6]);
        const int last_valid = int(coin_settings[coin_offset + 7]);
        if (k < first_valid || k > last_valid) continue;
        const int bar_offset = (k * C + c) * 4;
        if (!finite_positive(bars[bar_offset + 0])
            || !finite_positive(bars[bar_offset + 1])
            || !finite_positive(bars[bar_offset + 2])) continue;
        int tick_offset = (k * C + c) * 2;
        bool first_passive_reachable = short_side
            ? first[c].ticks <= fill_ticks[tick_offset + 0]
            : first[c].ticks > fill_ticks[tick_offset + 1];
        float cooldown = coin_override_or(
            coin_overrides, c, 23, config.cooldown_min
        );
        if (!first_passive_reachable || cooldown != 0.0f) continue;

        float new_psize = round_step(
            sim_psize[c] + first[c].strategy_qty,
            coin_settings[coin_offset + 0]
        );
        sim_pprice[c] = sim_psize[c] <= 0.0f
            ? first[c].price
            : sim_pprice[c] * (
                sim_psize[c] / fmax(new_psize, 1.0e-12f)
            ) + first[c].price * (
                first[c].strategy_qty / fmax(new_psize, 1.0e-12f)
            );
        sim_psize[c] = new_psize;
        sim_touch_tick[c] = short_side
            ? max(sim_touch_tick[c], first[c].ticks)
            : min(sim_touch_tick[c], first[c].ticks);
        suffix[c] = next_recursive_grid_entry(
            side, config, coin_overrides, c, short_side,
            sim_psize[c], sim_pprice[c], side.entry_gen_balance[c],
            side.entry_gen_allowed_wel[c], side.entry_gen_initial_tick[c],
            sim_touch_tick[c], side.entry_gen_market_price[c],
            coin_settings[coin_offset + 0],
            coin_settings[coin_offset + 1],
            coin_settings[coin_offset + 2],
            coin_settings[coin_offset + 3], c_mult, true,
            market_order_near_touch_threshold
        );
        suffix_valid[c] = suffix[c].ticks > 1
            && suffix[c].ticks != previous_tick[c]
            && suffix[c].strategy_qty > 0.0f
            && suffix[c].executable_qty > 0.0f;
    }

    float gated_twel = config.twel;
    if (isfinite(config.twel_threshold) && config.twel_threshold > 0.0f) {
        gated_twel = fmin(gated_twel, gated_twel * config.twel_threshold);
    }
    float strict_cap = fmax(gated_twel - 1.0e-7f, 0.0f);
    float cap_cost = strict_cap * generation_balance;
    if (!(generation_balance > 0.0f && cap_cost > current_cost)) return;
    float retained_cost = current_cost;

    for (int selection_rank = 0;
         selection_rank < MAX_COINS * 500;
         ++selection_rank) {
        int best_coin = -1;
        int best_kind = -1;
        RecursiveEntryCandidate best;
        best.ticks = 0;
        best.order_type = 0;
        best.price = 0.0f;
        best.strategy_qty = 0.0f;
        best.executable_qty = 0.0f;
        best.market = false;
        for (int c = 0; c < C; ++c) {
            float market_price = side.entry_gen_market_price[c];
            if (first_pending[c]
                && recursive_entry_gate_candidate_preferred(
                    first[c], c, market_price, best, best_coin,
                    best_coin >= 0
                        ? side.entry_gen_market_price[best_coin] : 0.0f,
                    short_side
                )) {
                best = first[c];
                best_coin = c;
                best_kind = 0;
            }
            if (suffix_valid[c]
                && recursive_entry_gate_candidate_preferred(
                    suffix[c], c, market_price, best, best_coin,
                    best_coin >= 0
                        ? side.entry_gen_market_price[best_coin] : 0.0f,
                    short_side
                )) {
                best = suffix[c];
                best_coin = c;
                best_kind = 1;
            }
        }
        if (best_coin < 0) break;

        int coin_offset = best_coin * COIN_COLS;
        float qty_step = coin_settings[coin_offset + 0];
        float min_qty = coin_settings[coin_offset + 2];
        float min_cost = coin_settings[coin_offset + 3];
        float c_mult = coin_settings[coin_offset + 4];
        float gate_price = best.market
            ? side.entry_gen_market_price[best_coin] : best.price;
        float full_cost = best.executable_qty * gate_price * c_mult;
        bool keep_full = retained_cost + full_cost < cap_cost;
        float kept_qty = best.executable_qty;
        bool boundary = !keep_full;
        if (boundary) {
            float room_cost = fmax(cap_cost - retained_cost, 0.0f);
            kept_qty = floor_step(
                room_cost / fmax(gate_price * c_mult, 1.0e-12f),
                qty_step
            );
            if (retained_cost + kept_qty * gate_price * c_mult
                >= cap_cost) {
                kept_qty = floor_step(kept_qty - qty_step, qty_step);
            }
            float executable_min = min_entry_qty(
                gate_price, qty_step, min_qty, min_cost, c_mult
            );
            if (kept_qty * (1.0f + 1.0e-6f) < executable_min) {
                kept_qty = 0.0f;
            }
        }

        if (best_kind == 0) {
            side.entry_qty[best_coin] = kept_qty;
            first_pending[best_coin] = false;
        } else if (kept_qty > 0.0f) {
            int kept_rank = side.entry_gate_suffix_keep_count[best_coin];
            if (boundary) {
                side.entry_gate_suffix_partial_rank[best_coin] = kept_rank;
                side.entry_gate_suffix_partial_qty[best_coin] = kept_qty;
            }
            side.entry_gate_suffix_keep_count[best_coin] = kept_rank + 1;
        }
        if (kept_qty > 0.0f) {
            retained_cost += kept_qty * gate_price * c_mult;
        }
        if (boundary) break;
        if (best_kind == 0) continue;

        float new_psize = round_step(
            sim_psize[best_coin] + best.strategy_qty, qty_step
        );
        sim_pprice[best_coin] = sim_psize[best_coin] <= 0.0f
            ? best.price
            : sim_pprice[best_coin] * (
                sim_psize[best_coin] / fmax(new_psize, 1.0e-12f)
            ) + best.price * (
                best.strategy_qty / fmax(new_psize, 1.0e-12f)
            );
        sim_psize[best_coin] = new_psize;
        sim_touch_tick[best_coin] = short_side
            ? max(sim_touch_tick[best_coin], best.ticks)
            : min(sim_touch_tick[best_coin], best.ticks);
        previous_tick[best_coin] = best.ticks;
        ++suffix_rank[best_coin];
        if (suffix_rank[best_coin] >= 499) {
            suffix_valid[best_coin] = false;
            continue;
        }
        suffix[best_coin] = next_recursive_grid_entry(
            side, config, coin_overrides, best_coin, short_side,
            sim_psize[best_coin], sim_pprice[best_coin],
            side.entry_gen_balance[best_coin],
            side.entry_gen_allowed_wel[best_coin],
            side.entry_gen_initial_tick[best_coin],
            sim_touch_tick[best_coin],
            side.entry_gen_market_price[best_coin],
            qty_step, coin_settings[coin_offset + 1], min_qty, min_cost,
            c_mult, true, market_order_near_touch_threshold
        );
        suffix_valid[best_coin] = suffix[best_coin].ticks > 1
            && suffix[best_coin].ticks != previous_tick[best_coin]
            && suffix[best_coin].strategy_qty > 0.0f
            && suffix[best_coin].executable_qty > 0.0f;
    }
}

// Shared fill accounting is separate from strategy-side state so a fused
// long+short kernel can apply both directional fill passes to one account and
// one chronology without reconstructing totals from directional summaries.
struct TrailingMartingaleMulticoinFillState {
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

inline TrailingMartingaleMulticoinFillState
init_trailing_martingale_multicoin_fill_state() {
    TrailingMartingaleMulticoinFillState fills;
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

inline void record_tm_multicoin_gross_pnl(
    float pnl,
    thread TrailingMartingaleMulticoinFillState& fills,
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

inline void record_tm_multicoin_close_fill(
    thread TrailingMartingaleMulticoinSideState& side,
    thread JointPortfolioAccount& account,
    thread TrailingMartingaleMulticoinFillState& fills,
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
    record_tm_multicoin_gross_pnl(gross_pnl, fills, short_side);
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

inline void record_tm_multicoin_entry_fill(
    thread TrailingMartingaleMulticoinSideState& side,
    thread JointPortfolioAccount& account,
    thread TrailingMartingaleMulticoinFillState& fills,
    device float* coin_fill_counts,
    int candidate_index,
    int coin_count,
    int coin,
    int k,
    float fee,
    float qty,
    float fill_price,
    float mark_price,
    float c_mult,
    bool short_side,
    bool collect_coin_fill_counts,
    thread float& hsl_equity_before_fills
) {
    const bool coin_hsl_mode =
        side.hsl.signal_mode == HSL_SIGNAL_COIN;
    record_realized_net(
        -fee, account,
        fills.day_fill_count, fills.fill_count,
        fills.fill_count_entry, fills.fill_count_long,
        fills.pnl_recovery_peak, fills.pnl_recovery_peak_k,
        fills.pnl_recovery_max_min, float(k), true, !short_side
    );
    side.coin_realized_pnl[coin] -= fee;
    if (coin_hsl_mode) {
        record_coin_hsl_realized_fill(
            side.coin_hsl[coin], side.coin_realized_pnl[coin]
        );
        advance_coin_hsl_equity_after_entry_fill(
            hsl_equity_before_fills,
            fee, qty, fill_price, mark_price,
            c_mult, short_side
        );
    }
    if (collect_coin_fill_counts) {
        coin_fill_counts[candidate_index * coin_count + coin] += 1.0f;
    }
}

inline void finalize_tm_multicoin_close_position(
    thread TrailingMartingaleMulticoinSideState& side,
    thread TrailingMartingaleMulticoinFillState& fills,
    int coin,
    int k
) {
    if (side.psize[coin] <= 0.0f) {
        side.pprice[coin] = 0.0f;
        if (side.position_open_k[coin] >= 0.0f) {
            float held_min = float(k) - side.position_open_k[coin];
            fills.held_max_min = fmax(fills.held_max_min, held_min);
            fills.held_sum_min += held_min;
            fills.held_count += 1.0f;
        }
        side.position_open_k[coin] = -1.0f;
    }
    side.close_qty[coin] = 0.0f;
    side.secondary_close_qty[coin] = 0.0f;
    side.close_market[coin] = false;
    side.secondary_close_market[coin] = false;
    side.close_reconstruct_after_reducer[coin] = false;
    side.close_recursive_market_mode[coin] = false;
    side.close_grid_prefix_qty[coin] = 0.0f;
    side.close_grid_prefix_tick[coin] = 0;
    side.close_is_exposure_reducer[coin] = false;
    side.close_is_unstuck_reducer[coin] = false;
    side.close_is_hsl_panic[coin] = false;
    side.filled_coin[coin] = true;
}

inline void apply_tm_multicoin_entry_position(
    thread TrailingMartingaleMulticoinSideState& side,
    thread TrailingMartingaleMulticoinFillState& fills,
    int coin,
    int k,
    float qty,
    float fill_price,
    float qty_step,
    float c_mult,
    float balance
) {
    bool was_flat = side.psize[coin] <= 0.0f;
    float new_size = round_step(side.psize[coin] + qty, qty_step);
    float new_price = was_flat ? fill_price
        : side.pprice[coin]
            * (side.psize[coin] / fmax(new_size, 1.0e-12f))
            + fill_price * (qty / fmax(new_size, 1.0e-12f));
    if (was_flat) side.position_open_k[coin] = float(k);
    side.psize[coin] = new_size;
    side.pprice[coin] = new_price;
    side.last_increase_k[coin] = float(k);
    fills.day_volume += fabs(qty) * fill_price * c_mult / balance;
    side.entry_qty[coin] = 0.0f;
    side.entry_market[coin] = false;
    side.filled_coin[coin] = true;
}

inline void update_tm_multicoin_position_fill_timestamp(
    thread TrailingMartingaleMulticoinSideState& side,
    thread TrailingMartingaleMulticoinFillState& fills,
    int coin,
    int k
) {
    if (side.position_last_fill_k[coin] >= 0.0f) {
        fills.position_unchanged_max_min = fmax(
            fills.position_unchanged_max_min,
            float(k) - side.position_last_fill_k[coin]
        );
    }
    side.position_last_fill_k[coin] =
        side.psize[coin] > 0.0f ? float(k) : -1.0f;
}

inline void clear_tm_multicoin_coin_orders(
    thread TrailingMartingaleMulticoinSideState& side,
    int coin
) {
    side.entry_qty[coin] = 0.0f;
    side.entry_strategy_qty[coin] = 0.0f;
    side.entry_gen_balance[coin] = 0.0f;
    side.entry_gen_allowed_wel[coin] = 0.0f;
    side.entry_gen_market_price[coin] = 0.0f;
    side.entry_gen_psize[coin] = 0.0f;
    side.entry_gen_pprice[coin] = 0.0f;
    side.entry_gate_suffix_partial_qty[coin] = 0.0f;
    side.entry_gate_suffix_keep_count[coin] = 0;
    side.entry_gate_suffix_partial_rank[coin] = -1;
    side.entry_tick[coin] = 0;
    side.entry_gen_initial_tick[coin] = 0;
    side.entry_gen_touch_tick[coin] = 0;
    side.entry_order_type[coin] = 0;
    side.entry_recursive_market_mode[coin] = false;
    side.entry_market[coin] = false;
    side.close_qty[coin] = 0.0f;
    side.secondary_close_qty[coin] = 0.0f;
    side.twel_close_qty[coin] = 0.0f;
    side.unstuck_close_qty[coin] = 0.0f;
    side.close_gen_balance[coin] = 0.0f;
    side.close_gen_allowed_wel[coin] = 0.0f;
    side.close_gen_market_price[coin] = 0.0f;
    side.close_grid_gen_psize[coin] = 0.0f;
    side.close_grid_prefix_qty[coin] = 0.0f;
    side.close_tick[coin] = 0;
    side.secondary_close_tick[coin] = 0;
    side.twel_close_tick[coin] = 0;
    side.unstuck_close_tick[coin] = 0;
    side.close_grid_max_rungs[coin] = 500;
    side.close_grid_prefix_tick[coin] = 0;
    side.close_reconstruct_after_reducer[coin] = false;
    side.close_recursive_market_mode[coin] = false;
    side.close_market[coin] = false;
    side.secondary_close_market[coin] = false;
    side.close_is_exposure_reducer[coin] = false;
    side.close_is_unstuck_reducer[coin] = false;
    side.close_is_hsl_panic[coin] = false;
}

inline bool force_close_tm_multicoin_delisted_position(
    thread TrailingMartingaleMulticoinSideState& side,
    thread JointPortfolioAccount& account,
    thread TrailingMartingaleMulticoinFillState& fills,
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
    record_tm_multicoin_close_fill(
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
    fills.day_volume += close_qty * close_price * c_mult / account.balance;
    finalize_tm_multicoin_close_position(side, fills, coin, k);
    update_tm_multicoin_position_fill_timestamp(side, fills, coin, k);
    return true;
}

inline bool force_close_tm_multicoin_delisted_one_side(
    thread TrailingMartingaleMulticoinSideState& side,
    thread JointPortfolioAccount& account,
    thread TrailingMartingaleMulticoinFillState& fills,
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
        const bool closed = force_close_tm_multicoin_delisted_position(
            side, account, fills, bars, coin_settings, coin_fill_counts,
            candidate_index, k, coin_count, c, short_side,
            collect_coin_fill_counts, market_order_slippage_pct,
            hsl_equity_before_close
        );
        if (closed) clear_tm_multicoin_coin_orders(side, c);
        any_close = any_close || closed;
    }
    return any_close;
}

inline bool force_close_tm_multicoin_delisted_fused(
    thread TrailingMartingaleMulticoinSideState& long_side,
    thread TrailingMartingaleMulticoinSideState& short_side,
    thread JointPortfolioAccount& account,
    thread TrailingMartingaleMulticoinFillState& fills,
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
        const bool long_closed = force_close_tm_multicoin_delisted_position(
            long_side, account, fills, bars, coin_settings, coin_fill_counts,
            candidate_index, k, coin_count, c, false,
            collect_coin_fill_counts, market_order_slippage_pct,
            hsl_equity_before_close
        );
        const bool short_closed = force_close_tm_multicoin_delisted_position(
            short_side, account, fills, bars, coin_settings, coin_fill_counts,
            candidate_index, k, coin_count, c, true,
            collect_coin_fill_counts, market_order_slippage_pct,
            hsl_equity_before_close
        );
        if (long_closed || short_closed) {
            clear_tm_multicoin_coin_orders(long_side, c);
            clear_tm_multicoin_coin_orders(short_side, c);
        }
        any_close = any_close || long_closed || short_closed;
    }
    return any_close;
}

inline bool process_tm_multicoin_side_fills(
    thread TrailingMartingaleMulticoinSideState& side,
    thread const TrailingMartingaleMulticoinSideConfig& config,
    thread JointPortfolioAccount& account,
    thread TrailingMartingaleMulticoinFillState& fills,
    constant float* bars,
    constant int* fill_ticks,
    constant int* touch_ticks,
    constant int* touch_nearest_ticks,
    constant int* touch_min_qty_bits,
    constant int* touch_min_qty_relation,
    constant float* coin_settings,
    constant float* coin_overrides,
    device float* coin_fill_counts,
#if PASSIVBOT_ENTRY_INTERVAL_ENABLED
    device float* entry_interval_stats,
    device int* entry_interval_counts,
#endif
    int b,
    int k,
    int C,
    bool short_side,
    bool alive,
    bool collect_coin_fill_counts,
    bool loss_gate_enabled,
    float max_realized_loss_pct,
    float market_order_slippage_pct,
    float market_order_near_touch_threshold,
    bool hsl_panic_market,
    thread float& hsl_equity_before_fills
) {
    const bool coin_hsl_mode = config.coin_hsl_mode;
    const float close_qty_pct = config.close_qty_pct;
    const float close_threshold_base = config.close_threshold_base;
    const float close_threshold_we = config.close_threshold_we;
    const float close_threshold_v1h = config.close_threshold_v1h;
    const float close_threshold_v1m = config.close_threshold_v1m;
    thread float* volatility_1m = side.volatility_1m;
    thread float* volatility_1h = side.volatility_1h;
    thread float* psize = side.psize;
    thread float* pprice = side.pprice;
    thread float* entry_qty = side.entry_qty;
    thread float* entry_strategy_qty = side.entry_strategy_qty;
    thread float* entry_gen_balance = side.entry_gen_balance;
    thread float* entry_gen_allowed_wel = side.entry_gen_allowed_wel;
    thread float* entry_gen_market_price = side.entry_gen_market_price;
    thread float* entry_gen_psize = side.entry_gen_psize;
    thread float* entry_gen_pprice = side.entry_gen_pprice;
    thread float* close_qty = side.close_qty;
    thread float* secondary_close_qty = side.secondary_close_qty;
    thread float* close_gen_balance = side.close_gen_balance;
    thread float* close_gen_allowed_wel = side.close_gen_allowed_wel;
    thread float* close_gen_market_price = side.close_gen_market_price;
    thread float* close_grid_gen_psize = side.close_grid_gen_psize;
    thread float* close_grid_prefix_qty = side.close_grid_prefix_qty;
    thread int* entry_tick = side.entry_tick;
    thread int* entry_gen_initial_tick = side.entry_gen_initial_tick;
    thread int* entry_gen_touch_tick = side.entry_gen_touch_tick;
    thread int* close_tick = side.close_tick;
    thread int* secondary_close_tick = side.secondary_close_tick;
    thread int* close_grid_max_rungs = side.close_grid_max_rungs;
    thread int* close_grid_prefix_tick = side.close_grid_prefix_tick;
    thread bool* close_reconstruct_after_reducer =
        side.close_reconstruct_after_reducer;
    thread bool* close_recursive_market_mode =
        side.close_recursive_market_mode;
    thread bool* filled_coin = side.filled_coin;
    thread bool* entry_recursive_market_mode =
        side.entry_recursive_market_mode;
    thread bool* entry_market = side.entry_market;
    thread bool* close_market = side.close_market;
    thread bool* secondary_close_market = side.secondary_close_market;
    thread bool* close_is_exposure_reducer =
        side.close_is_exposure_reducer;
    thread bool* close_is_unstuck_reducer =
        side.close_is_unstuck_reducer;
    thread bool* close_is_hsl_panic = side.close_is_hsl_panic;
    thread float& balance = account.balance;
    thread float& realized_pnl_cumsum_last = account.realized_pnl_total;
    thread float& realized_pnl_cumsum_max = account.realized_pnl_peak;
    thread float& day_volume = fills.day_volume;
    bool any_fill = false;
    for (int c = 0; c < C; ++c) filled_coin[c] = false;
    apply_tm_multicoin_recursive_entry_twel_gate(
        side, config, bars, fill_ticks, coin_settings, coin_overrides,
        k, C, short_side, market_order_near_touch_threshold
    );
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
        const float taker_fee = coin_settings[coin_offset + 11];

        bool close_ready = close_qty[c] > 0.0f && psize[c] > 0.0f;
        bool coin_hsl_panic_market = coin_hsl_mode
            ? coin_override_or(
                coin_overrides, c, HSL_OVERRIDE_START + 9,
                hsl_panic_market ? 1.0f : 0.0f
            ) > 0.5f
            : hsl_panic_market;
        bool primary_market_panic = close_is_hsl_panic[c]
            && coin_hsl_panic_market;
        bool primary_market = primary_market_panic || close_market[c];
        bool filled_close = close_ready
            && (primary_market || (short_side
                ? close_tick[c] > fill_ticks[tick_offset + 1]
                : close_tick[c] <= fill_ticks[tick_offset + 0]));
        bool filled_secondary_close = secondary_close_qty[c] > 0.0f
            && psize[c] > 0.0f
            && (secondary_close_market[c] || (short_side
                ? secondary_close_tick[c] > fill_ticks[tick_offset + 1]
                : secondary_close_tick[c] <= fill_ticks[tick_offset + 0]));
        bool recursive_market_expand = psize[c] > 0.0f
            && close_recursive_market_mode[c]
            && !close_is_hsl_panic[c]
            && recursive_grid_close_would_expand(
                short_side,
                close_grid_gen_psize[c],
                pprice[c],
                close_gen_balance[c],
                close_gen_allowed_wel[c],
                touch_ticks[((k - 1) * C + c) * 2 + 0],
                touch_ticks[((k - 1) * C + c) * 2 + 1],
                touch_nearest_ticks[(k - 1) * C + c],
                as_type<float>(touch_min_qty_bits[(k - 1) * C + c]),
                touch_min_qty_relation[(k - 1) * C + c],
                coin_override_or(coin_overrides, c, 15, close_qty_pct),
                coin_override_or(
                    coin_overrides, c, 16, close_threshold_base
                ),
                coin_override_or(coin_overrides, c, 17, close_threshold_we),
                coin_override_or(
                    coin_overrides, c, 18, close_threshold_v1h
                ),
                coin_override_or(
                    coin_overrides, c, 19, close_threshold_v1m
                ),
                volatility_1h[c],
                volatility_1m[c],
                qty_step,
                price_step,
                min_qty,
                min_cost,
                c_mult,
                close_grid_max_rungs[c],
                fill_ticks[tick_offset + 0],
                fill_ticks[tick_offset + 1]
            );
        bool rebuild_grid = psize[c] > 0.0f
            && close_reconstruct_after_reducer[c]
            && (!close_recursive_market_mode[c] || recursive_market_expand)
            && !close_is_hsl_panic[c];
        if (recursive_market_expand && !close_is_exposure_reducer[c]) {
            filled_close = false;
        }
        if (filled_close || filled_secondary_close || rebuild_grid) {
            float fill_price = primary_market
                ? ordinary_market_fill_price(
                    close, short_side,
                    market_order_slippage_pct, price_step
                )
                : float(close_tick[c]) * price_step;
            float reducer_qty = rebuild_grid
                    && close_recursive_market_mode[c]
                    && !close_is_exposure_reducer[c]
                ? 0.0f
                : fmin(round_step(close_qty[c], qty_step), psize[c]);
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
                    close_grid_prefix_tick[c],
                    close_grid_prefix_qty[c],
                    close_grid_max_rungs[c],
                    -1,
                    close_gen_market_price[c],
                    close_recursive_market_mode[c],
                    market_order_near_touch_threshold,
                    grid_gen_psize,
                    group
                );
                reverse = coin_close_threshold_we > 0.0f;
            }

            float ordinary_budget = fmax(
                round_step(psize[c] - reducer_qty, qty_step), 0.0f
            );
            float remaining_budget = ordinary_budget;
            float kept_ordinary = 0.0f;
            // A market reducer was sized and admitted against the executable
            // touch captured when the order was generated.  Keep that same
            // minimum for recursive-ladder dust allocation; the next
            // candle's slipped fill price may change realized PnL, but must
            // not rewrite an already emitted quantity.
            float reducer_allocation_price = primary_market
                    && close_is_exposure_reducer[c]
                ? close_gen_market_price[c] : fill_price;
            float minimum_any = min_entry_qty(
                reducer_allocation_price,
                qty_step, min_qty, min_cost, c_mult
            );
            int last_kept_rank = -1;
            bool all_groups_below_min = close_recursive_market_mode[c]
                && group_count > 0;
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
                    close_grid_prefix_tick[c],
                    close_grid_prefix_qty[c],
                    close_grid_max_rungs[c],
                    wanted,
                    close_gen_market_price[c],
                    close_recursive_market_mode[c],
                    market_order_near_touch_threshold,
                    grid_gen_psize,
                    group
                );
                float trimmed_qty = fmin(group.qty, remaining_budget);
                float group_min = min_entry_qty(
                    group.market ? close_gen_market_price[c] : group.price,
                    qty_step, min_qty, min_cost, c_mult
                );
                all_groups_below_min = all_groups_below_min
                    && psize[c] * (1.0f + 1.0e-6f) < group_min;
                bool partial_trim = trimmed_qty * (1.0f + 1.0e-6f)
                    < group.qty;
                if (trimmed_qty * (1.0f + 1.0e-6f) < group_min) {
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
            int collapse_ordinary_rank = -1;
            if (all_groups_below_min && reducer_qty <= 0.0f) {
                // Rust preserves one closest-to-fill close at the full
                // remaining position when every recursive group is below
                // the executable minimum. This includes market-promoted
                // groups whose resize already retained that exception.
                collapse_ordinary_rank = 0;
                kept_ordinary = psize[c];
                last_kept_rank = 0;
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
                    close_grid_prefix_tick[c],
                    close_grid_prefix_qty[c],
                    close_grid_max_rungs[c],
                    wanted,
                    close_gen_market_price[c],
                    close_recursive_market_mode[c],
                    market_order_near_touch_threshold,
                    grid_gen_psize,
                    group
                );
                if (group.qty <= 0.0f) break;
                float group_min = min_entry_qty(
                    group.market ? close_gen_market_price[c] : group.price,
                    qty_step, min_qty, min_cost, c_mult
                );
                float trimmed_group_qty = 0.0f;
                if (collapse_ordinary_rank >= 0) {
                    trimmed_group_qty = rank == collapse_ordinary_rank
                        ? ordinary_budget : 0.0f;
                } else {
                    trimmed_group_qty = fmin(group.qty, remaining_budget);
                    bool partial_trim = trimmed_group_qty
                        * (1.0f + 1.0e-6f) < group.qty;
                    if (trimmed_group_qty * (1.0f + 1.0e-6f)
                        < group_min) {
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
                        if (rank == last_kept_rank
                            && dust_remainder > 0.0f
                            && dust_remainder < minimum_any) {
                            trimmed_group_qty = round_step(
                                trimmed_group_qty + dust_remainder, qty_step
                            );
                        }
                    }
                }
                bool reducer_before_group = filled_close
                    && !reducer_executed
                    && (short_side
                        ? close_tick[c] > group.ticks
                        : close_tick[c] < group.ticks);
                if (reducer_before_group) {
                    float qty = fmin(reducer_qty, psize[c]);
                    float reducer_fee_rate = primary_market
                        ? taker_fee : maker_fee;
                    float pnl = qty * c_mult * (short_side
                        ? pprice[c] - fill_price
                        : fill_price - pprice[c]);
                    if (realized_loss_proxy_allows_reducer(
                            qty, fill_price, pprice[c], short_side,
                            c_mult, reducer_fee_rate,
                            close_is_unstuck_reducer[c], loss_gate_enabled,
                            balance, realized_pnl_cumsum_last,
                            realized_pnl_cumsum_max, max_realized_loss_pct
                        )) {
                        float net_pnl = pnl
                            - qty * fill_price * c_mult * reducer_fee_rate;
                        record_tm_multicoin_close_fill(
                            side, account, fills, coin_fill_counts,
                            int(b), C, c, k, pnl, net_pnl, qty,
                            pprice[c], close, c_mult, short_side,
                            false, collect_coin_fill_counts,
                            hsl_equity_before_fills
                        );
                        psize[c] = fmax(
                            round_step(psize[c] - qty, qty_step), 0.0f
                        );
                        day_volume += qty * fill_price * c_mult / balance;
                        reducer_executed = true;
                        executed_close = true;
                    }
                }
                bool reachable = group.market || (short_side
                    ? group.ticks > fill_ticks[tick_offset + 1]
                    : group.ticks <= fill_ticks[tick_offset + 0]);
                if (!reachable) break;
                float group_qty = trimmed_group_qty;
                if (group_qty <= 0.0f) continue;
                float grid_qty = fmin(
                    round_step(group_qty, qty_step), psize[c]
                );
                float group_fill_price = group.market
                    ? ordinary_market_fill_price(
                        close, short_side,
                        market_order_slippage_pct, price_step
                    )
                    : group.price;
                float group_fee_rate = group.market ? taker_fee : maker_fee;
                float grid_pnl = grid_qty * c_mult * (short_side
                    ? pprice[c] - group_fill_price
                    : group_fill_price - pprice[c]);
                if (!realized_loss_proxy_allows_close(
                        grid_qty, group_fill_price, pprice[c], short_side,
                        c_mult, group_fee_rate, loss_gate_enabled
                    )) {
                    continue;
                }
                float grid_net_pnl = grid_pnl
                    - grid_qty * group_fill_price * c_mult * group_fee_rate;
                record_tm_multicoin_close_fill(
                    side, account, fills, coin_fill_counts,
                    int(b), C, c, k, grid_pnl, grid_net_pnl,
                    grid_qty, pprice[c], close, c_mult, short_side,
                    false, collect_coin_fill_counts,
                    hsl_equity_before_fills
                );
                psize[c] = fmax(
                    round_step(psize[c] - grid_qty, qty_step), 0.0f
                );
                day_volume += grid_qty * group_fill_price * c_mult / balance;
                executed_close = true;
                if (psize[c] <= 0.0f) break;
            }

            float secondary_price = secondary_close_market[c]
                ? ordinary_market_fill_price(
                    close, short_side,
                    market_order_slippage_pct, price_step
                )
                : float(secondary_close_tick[c]) * price_step;
            bool secondary_first = filled_secondary_close
                && (!(filled_close && !reducer_executed)
                    || (short_side
                        ? secondary_close_tick[c] >= close_tick[c]
                        : secondary_close_tick[c] <= close_tick[c]));
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
                bool is_unstuck = !use_secondary
                    && close_is_unstuck_reducer[c];
                bool is_hsl_panic = !use_secondary
                    && close_is_hsl_panic[c];
                bool market_execution = use_secondary
                    ? secondary_close_market[c] : primary_market;
                float fee_rate = market_execution ? taker_fee : maker_fee;
                if (!is_hsl_panic && !realized_loss_proxy_allows_reducer(
                        qty, price, pprice[c], short_side,
                        c_mult, fee_rate, is_unstuck, loss_gate_enabled,
                        balance, realized_pnl_cumsum_last,
                        realized_pnl_cumsum_max, max_realized_loss_pct
                    )) {
                    continue;
                }
                float net_pnl = pnl - qty * price * c_mult * fee_rate;
                record_tm_multicoin_close_fill(
                    side, account, fills, coin_fill_counts,
                    int(b), C, c, k, pnl, net_pnl, qty,
                    pprice[c], close, c_mult, short_side,
                    is_hsl_panic, collect_coin_fill_counts,
                    hsl_equity_before_fills
                );
                psize[c] = fmax(
                    round_step(psize[c] - qty, qty_step), 0.0f
                );
                day_volume += qty * price * c_mult / balance;
                if (!use_secondary) reducer_executed = true;
                executed_close = true;
            }

            if (executed_close) {
                finalize_tm_multicoin_close_position(
                    side, fills, c, k
                );
                any_fill = true;
            }
        }

        bool first_entry_passive_reachable = entry_qty[c] > 0.0f
            && (short_side
                ? entry_tick[c] <= fill_ticks[tick_offset + 0]
                : entry_tick[c] > fill_ticks[tick_offset + 1]);
        bool filled_entry = entry_qty[c] > 0.0f
            && (entry_market[c] || first_entry_passive_reachable);
        bool gated_recursive_plan = side.entry_deferred_twel_gate
            && entry_recursive_market_mode[c]
            && (entry_qty[c] > 0.0f
                || side.entry_gate_suffix_keep_count[c] > 0);
        if (gated_recursive_plan) {
            float sim_psize = entry_gen_psize[c];
            float sim_pprice = entry_gen_pprice[c];
            int sim_touch_tick = entry_gen_touch_tick[c];
            int previous_ticks = 0;
            int suffix_keep_count = side.entry_gate_suffix_keep_count[c];
            int suffix_partial_rank =
                side.entry_gate_suffix_partial_rank[c];
            float suffix_partial_qty =
                side.entry_gate_suffix_partial_qty[c];
            for (int rung = 0; rung <= suffix_keep_count; ++rung) {
                RecursiveEntryCandidate candidate;
                if (rung == 0) {
                    candidate.ticks = entry_tick[c];
                    candidate.order_type = side.entry_order_type[c];
                    candidate.price = float(entry_tick[c]) * price_step;
                    candidate.strategy_qty = entry_strategy_qty[c];
                    candidate.executable_qty = entry_qty[c];
                    candidate.market = entry_market[c];
                } else {
                    candidate = next_recursive_grid_entry(
                        side, config, coin_overrides, c, short_side,
                        sim_psize, sim_pprice, entry_gen_balance[c],
                        entry_gen_allowed_wel[c], entry_gen_initial_tick[c],
                        sim_touch_tick, entry_gen_market_price[c],
                        qty_step, price_step, min_qty, min_cost, c_mult,
                        true, market_order_near_touch_threshold
                    );
                    int suffix_index = rung - 1;
                    if (suffix_index == suffix_partial_rank) {
                        candidate.executable_qty = suffix_partial_qty;
                    }
                }
                if (!(candidate.strategy_qty > 0.0f
                    && candidate.ticks > 1)) break;
                if (rung > 0 && candidate.ticks == previous_ticks) break;
                bool passive_reachable = short_side
                    ? candidate.ticks <= fill_ticks[tick_offset + 0]
                    : candidate.ticks > fill_ticks[tick_offset + 1];
                bool selected = candidate.executable_qty > 0.0f;
                if (selected && (candidate.market || passive_reachable)) {
                    float fill_price = candidate.market
                        ? ordinary_market_fill_price(
                            close, !short_side,
                            market_order_slippage_pct, price_step
                        )
                        : candidate.price;
                    float adjusted = round_step(
                        candidate.executable_qty, qty_step
                    );
                    float fee = adjusted * fill_price * c_mult
                        * (candidate.market ? taker_fee : maker_fee);
#if PASSIVBOT_ENTRY_INTERVAL_ENABLED
                    if (candidate.order_type == 0
                        || candidate.order_type == 11) {
                        record_initial_entry_interval(
                            entry_interval_stats, entry_interval_counts, b,
                            side.last_initial_entry_k[c], float(k)
                        );
                    }
#endif
                    record_tm_multicoin_entry_fill(
                        side, account, fills, coin_fill_counts,
                        int(b), C, c, k, fee, adjusted, fill_price,
                        close, c_mult, short_side,
                        collect_coin_fill_counts,
                        hsl_equity_before_fills
                    );
                    apply_tm_multicoin_entry_position(
                        side, fills, c, k, adjusted, fill_price,
                        qty_step, c_mult, balance
                    );
                    any_fill = true;
                } else if (rung > 0 && selected) {
                    break;
                }

                float new_sim_psize = round_step(
                    sim_psize + candidate.strategy_qty, qty_step
                );
                sim_pprice = sim_psize <= 0.0f
                    ? candidate.price
                    : sim_pprice * (
                        sim_psize / fmax(new_sim_psize, 1.0e-12f)
                    ) + candidate.price * (
                        candidate.strategy_qty
                            / fmax(new_sim_psize, 1.0e-12f)
                    );
                sim_psize = new_sim_psize;
                previous_ticks = candidate.ticks;
                sim_touch_tick = short_side
                    ? max(sim_touch_tick, candidate.ticks)
                    : min(sim_touch_tick, candidate.ticks);
            }
            entry_qty[c] = 0.0f;
        } else if (filled_entry && entry_recursive_market_mode[c]) {
            float sim_psize = entry_gen_psize[c];
            float sim_pprice = entry_gen_pprice[c];
            int sim_touch_tick = entry_gen_touch_tick[c];
            int previous_ticks = 0;
            for (int rung = 0; rung < 500; ++rung) {
                RecursiveEntryCandidate candidate;
                if (rung == 0) {
                    candidate.ticks = entry_tick[c];
                    candidate.order_type = side.entry_order_type[c];
                    candidate.price = float(entry_tick[c]) * price_step;
                    candidate.strategy_qty = entry_strategy_qty[c];
                    candidate.executable_qty = entry_qty[c];
                    candidate.market = entry_market[c];
                } else {
                    candidate = next_recursive_grid_entry(
                        side, config, coin_overrides, c, short_side,
                        sim_psize, sim_pprice, entry_gen_balance[c],
                        entry_gen_allowed_wel[c], entry_gen_initial_tick[c],
                        sim_touch_tick, entry_gen_market_price[c],
                        qty_step, price_step, min_qty, min_cost, c_mult,
                        true, market_order_near_touch_threshold
                    );
                }
                if (!(candidate.strategy_qty > 0.0f
                    && candidate.executable_qty > 0.0f
                    && candidate.ticks > 1)) break;
                if (rung > 0 && candidate.ticks == previous_ticks) break;
                bool passive_reachable = short_side
                    ? candidate.ticks <= fill_ticks[tick_offset + 0]
                    : candidate.ticks > fill_ticks[tick_offset + 1];
                if (!candidate.market && !passive_reachable) break;
                float fill_price = candidate.market
                    ? ordinary_market_fill_price(
                        close, !short_side,
                        market_order_slippage_pct, price_step
                    )
                    : candidate.price;
                float adjusted = round_step(
                    candidate.executable_qty, qty_step
                );
                float fee = adjusted * fill_price * c_mult
                    * (candidate.market ? taker_fee : maker_fee);
#if PASSIVBOT_ENTRY_INTERVAL_ENABLED
                if (candidate.order_type == 0
                    || candidate.order_type == 11) {
                    record_initial_entry_interval(
                        entry_interval_stats, entry_interval_counts, b,
                        side.last_initial_entry_k[c], float(k)
                    );
                }
#endif
                record_tm_multicoin_entry_fill(
                    side, account, fills, coin_fill_counts,
                    int(b), C, c, k, fee, adjusted, fill_price,
                    close, c_mult, short_side, collect_coin_fill_counts,
                    hsl_equity_before_fills
                );
                apply_tm_multicoin_entry_position(
                    side, fills, c, k, adjusted, fill_price,
                    qty_step, c_mult, balance
                );
                any_fill = true;

                float new_sim_psize = round_step(
                    sim_psize + candidate.strategy_qty, qty_step
                );
                sim_pprice = sim_psize <= 0.0f
                    ? candidate.price
                    : sim_pprice * (
                        sim_psize / fmax(new_sim_psize, 1.0e-12f)
                    ) + candidate.price * (
                        candidate.strategy_qty
                            / fmax(new_sim_psize, 1.0e-12f)
                    );
                sim_psize = new_sim_psize;
                previous_ticks = candidate.ticks;
                sim_touch_tick = short_side
                    ? max(sim_touch_tick, candidate.ticks)
                    : min(sim_touch_tick, candidate.ticks);
                if (rung == 0 && !first_entry_passive_reachable) break;
                if (coin_override_or(
                        coin_overrides, c, 23, config.cooldown_min
                    ) != 0.0f) break;
            }
            entry_qty[c] = 0.0f;
        } else if (filled_entry) {
            float fill_price = entry_market[c]
                ? ordinary_market_fill_price(
                    close, !short_side,
                    market_order_slippage_pct, price_step
                )
                : float(entry_tick[c]) * price_step;
            float adjusted = round_step(entry_qty[c], qty_step);
            float fee = adjusted * fill_price * c_mult
                * (entry_market[c] ? taker_fee : maker_fee);
#if PASSIVBOT_ENTRY_INTERVAL_ENABLED
            if (side.entry_order_type[c] == 0
                || side.entry_order_type[c] == 11) {
                record_initial_entry_interval(
                    entry_interval_stats, entry_interval_counts, b,
                    side.last_initial_entry_k[c], float(k)
                );
            }
#endif
            record_tm_multicoin_entry_fill(
                side, account, fills, coin_fill_counts,
                int(b), C, c, k, fee, adjusted, fill_price,
                close, c_mult, short_side, collect_coin_fill_counts,
                hsl_equity_before_fills
            );
            apply_tm_multicoin_entry_position(
                side, fills, c, k, adjusted, fill_price,
                qty_step, c_mult, balance
            );
            any_fill = true;
        }
        if (filled_coin[c]) {
            update_tm_multicoin_position_fill_timestamp(
                side, fills, c, k
            );
        }
    }
    return any_fill;
}

inline TrailingMartingaleMulticoinSideConfig
load_trailing_martingale_multicoin_side_config(
    constant float* params,
    int po
) {
    TrailingMartingaleMulticoinSideConfig config;
    config.span_a = params[po + 0];
    config.span_b = params[po + 1];
    config.span_1h = params[po + 2];
    config.span_1m = params[po + 3];
    config.ddf = params[po + 4];
    config.initial_ema_dist = params[po + 5];
    config.initial_qty_pct = params[po + 6];
    config.entry_threshold_base = params[po + 7];
    config.entry_threshold_we = params[po + 8];
    config.entry_threshold_v1h = params[po + 9];
    config.entry_threshold_v1m = params[po + 10];
    config.entry_retracement_base = params[po + 11];
    config.entry_retracement_we = params[po + 12];
    config.entry_retracement_v1h = params[po + 13];
    config.entry_retracement_v1m = params[po + 14];
    config.close_qty_pct = params[po + 15];
    config.close_threshold_base = params[po + 16];
    config.close_threshold_we = params[po + 17];
    config.close_threshold_v1h = params[po + 18];
    config.close_threshold_v1m = params[po + 19];
    config.close_retracement_base = params[po + 20];
    config.close_retracement_v1h = params[po + 21];
    config.close_retracement_v1m = params[po + 22];
    config.cooldown_min = ceil(params[po + 23]);
    config.twel = params[po + 24];
    config.gate_initial = params[po + 25] > 0.5f;
    config.gate_reentry = params[po + 26] > 0.5f;
    config.forager_volume_span = params[po + 27];
    config.forager_volatility_span = params[po + 28];
    config.volume_drop = clamp(params[po + 29], 0.0f, 1.0f);
    config.w_volume = params[po + 30];
    config.w_ready = params[po + 31];
    config.w_volatility = params[po + 32];
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
    config.n_positions = max(1, int(rint(params[po + 33])));
    config.allowance_pct = params[po + 34];
    config.legacy_raw_allowance = params[po + 35] > 0.5f;
    config.twel_entry_gate_enabled = params[po + 36] > 0.5f;
    config.twel_threshold = params[po + 37];
    config.wel_enforcer_enabled = params[po + 38] > 0.5f;
    config.wel_enforcer_threshold = params[po + 39];
    config.twel_enforcer_enabled = params[po + 40] > 0.5f;
    config.twel_enforcer_reduce_portfolio = params[po + 41] > 0.5f;
    config.unstuck_enabled = params[po + 42] > 0.5f;
    config.unstuck_ema_gating_enabled = params[po + 43] > 0.5f;
    config.unstuck_close_pct = params[po + 44];
    config.unstuck_ema_dist = params[po + 45];
    config.unstuck_loss_allowance_pct = params[po + 46];
    config.unstuck_threshold = params[po + 47];
    config.alpha_forager_volume = config.forager_volume_span > 0.0f
        ? clamp(2.0f / (config.forager_volume_span + 1.0f), 0.0f, 1.0f)
        : 0.0f;
    config.alpha_forager_volatility = config.forager_volatility_span > 0.0f
        ? clamp(2.0f / (config.forager_volatility_span + 1.0f), 0.0f, 1.0f)
        : 0.0f;
    config.hsl_template = load_hsl(params, po, 48);
    config.coin_hsl_mode = config.hsl_template.signal_mode == HSL_SIGNAL_COIN;
    return config;
}

inline void init_trailing_martingale_multicoin_side_state(
    thread TrailingMartingaleMulticoinSideState& side,
    thread const TrailingMartingaleMulticoinSideConfig& config,
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
    side.entry_deferred_twel_gate = false;
    side.selection_initialized = false;
    side.max_tradable_seen = 0;
    side.previous_effective_n_positions = 0;
    for (int c = 0; c < MAX_COINS; ++c) {
        float seed_close = c < coin_count
            ? coin_settings[c * COIN_COLS + 9] : 0.0f;
        float seed_volume = c < coin_count
            ? coin_settings[c * COIN_COLS + 10] : 0.0f;
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
        side.entry_strategy_qty[c] = 0.0f;
        side.entry_gen_balance[c] = 0.0f;
        side.entry_gen_allowed_wel[c] = 0.0f;
        side.entry_gen_market_price[c] = 0.0f;
        side.entry_gen_psize[c] = 0.0f;
        side.entry_gen_pprice[c] = 0.0f;
        side.entry_gate_suffix_partial_qty[c] = 0.0f;
        side.close_qty[c] = 0.0f;
        side.secondary_close_qty[c] = 0.0f;
        side.twel_close_qty[c] = 0.0f;
        side.unstuck_close_qty[c] = 0.0f;
        side.close_gen_balance[c] = 0.0f;
        side.close_gen_allowed_wel[c] = 0.0f;
        side.close_gen_market_price[c] = 0.0f;
        side.close_grid_gen_psize[c] = 0.0f;
        side.close_grid_prefix_qty[c] = 0.0f;
        side.position_open_k[c] = -1.0f;
        side.position_last_fill_k[c] = -1.0f;
#if PASSIVBOT_ENTRY_INTERVAL_ENABLED
        side.last_initial_entry_k[c] = -1.0f;
#endif
        side.score[c] = -INFINITY;
        side.contribution[c] = 0.0f;
        side.minimum_entry[c] = 0.0f;
        side.min_since_open[c] = INFINITY;
        side.max_since_min[c] = 0.0f;
        side.max_since_open[c] = 0.0f;
        side.min_since_max[c] = INFINITY;
        side.entry_tick[c] = 0;
        side.entry_gen_initial_tick[c] = 0;
        side.entry_gen_touch_tick[c] = 0;
        side.entry_order_type[c] = 0;
        side.entry_gate_suffix_keep_count[c] = 0;
        side.entry_gate_suffix_partial_rank[c] = -1;
        side.close_tick[c] = 0;
        side.secondary_close_tick[c] = 0;
        side.twel_close_tick[c] = 0;
        side.unstuck_close_tick[c] = 0;
        side.close_grid_max_rungs[c] = 500;
        side.close_grid_prefix_tick[c] = 0;
        side.selected[c] = false;
        side.incumbent[c] = false;
        side.survivor[c] = false;
        side.entry_candidate[c] = false;
        side.entry_recursive_market_mode[c] = false;
        side.close_reconstruct_after_reducer[c] = false;
        side.close_recursive_market_mode[c] = false;
        side.filled_coin[c] = false;
        side.entry_market[c] = false;
        side.close_market[c] = false;
        side.secondary_close_market[c] = false;
        side.close_is_exposure_reducer[c] = false;
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
        float coin_span_a = c < coin_count
            ? coin_override_or(coin_overrides, c, 0, config.span_a)
            : config.span_a;
        float coin_span_b = c < coin_count
            ? coin_override_or(coin_overrides, c, 1, config.span_b)
            : config.span_b;
        float coin_span_c = sqrt(fmax(coin_span_a * coin_span_b, 1.0f));
        float coin_span_lo = fmin(
            coin_span_a, fmin(coin_span_b, coin_span_c)
        );
        float coin_span_hi = fmax(
            coin_span_a, fmax(coin_span_b, coin_span_c)
        );
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
        float coin_span_1h = c < coin_count
            ? coin_override_or(coin_overrides, c, 2, config.span_1h)
            : config.span_1h;
        float coin_span_1m = c < coin_count
            ? coin_override_or(coin_overrides, c, 3, config.span_1m)
            : config.span_1m;
        side.alpha_1h_coin[c] = coin_span_1h > 0.0f
            ? 2.0f / (fmax(coin_span_1h, 1.0f) + 1.0f) : 0.0f;
        side.alpha_1m_coin[c] = coin_span_1m > 0.0f
            ? clamp(2.0f / (coin_span_1m + 1.0f), 0.0f, 1.0f) : 0.0f;
    }
}

inline float accumulate_tm_multicoin_side_unrealized_pnl(
    thread const TrailingMartingaleMulticoinSideState& side,
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

inline void update_tm_multicoin_side_indicators(
    thread TrailingMartingaleMulticoinSideState& side,
    thread const TrailingMartingaleMulticoinSideConfig& config,
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
        if (side.psize[c] > 0.0f) {
            if (side.filled_coin[c]) {
                side.min_since_open[c] = INFINITY;
                side.max_since_min[c] = 0.0f;
                side.max_since_open[c] = 0.0f;
                side.min_since_max[c] = INFINITY;
            } else {
                if (low < side.min_since_open[c]) {
                    side.min_since_open[c] = low;
                    side.max_since_min[c] = close;
                } else {
                    side.max_since_min[c] = fmax(
                        side.max_since_min[c], high
                    );
                }
                if (high > side.max_since_open[c]) {
                    side.max_since_open[c] = high;
                    side.min_since_max[c] = close;
                } else {
                    side.min_since_max[c] = fmin(
                        side.min_since_max[c], low
                    );
                }
            }
        }
    }
}

inline int count_tm_multicoin_tradable_coins(
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
        float coin_wel = coin_override_or(coin_overrides, c, 24, -1.0f);
        if (k >= int(coin_settings[coin_offset + 8])
            && k <= int(coin_settings[coin_offset + 7])
            && finite_positive(close) && coin_wel != 0.0f) {
            tradable_count += 1;
        }
    }
    return tradable_count;
}

inline bool tm_multicoin_side_has_position(
    thread const TrailingMartingaleMulticoinSideState& side,
    int coin_count
) {
    for (int c = 0; c < coin_count; ++c) {
        if (side.psize[c] > 0.0f) return true;
    }
    return false;
}

inline bool tm_multicoin_side_held_marks_are_valid(
    thread const TrailingMartingaleMulticoinSideState& side,
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

inline bool tm_multicoin_side_has_blocking_orders(
    thread TrailingMartingaleMulticoinSideState& side,
    thread const TrailingMartingaleMulticoinSideConfig& config,
    constant float* bars,
    constant float* coin_settings,
    int k,
    int coin_count
) {
    bool side_has_position = config.coin_hsl_mode
        ? false : tm_multicoin_side_has_position(side, coin_count);
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
inline bool update_tm_multicoin_dual_side_hsl(
    thread TrailingMartingaleMulticoinSideState& long_side,
    thread const TrailingMartingaleMulticoinSideConfig& long_config,
    int long_effective_n_positions,
    thread TrailingMartingaleMulticoinSideState& short_side,
    thread const TrailingMartingaleMulticoinSideConfig& short_config,
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
    if (!tm_multicoin_side_held_marks_are_valid(
            long_side, bars, coin_settings, k, coin_count
        ) || !tm_multicoin_side_held_marks_are_valid(
            short_side, bars, coin_settings, k, coin_count
        )) {
        return false;
    }

    float long_unrealized = accumulate_tm_multicoin_side_unrealized_pnl(
        long_side, bars, coin_settings, k, coin_count, false, 0.0f
    );
    float short_unrealized = accumulate_tm_multicoin_side_unrealized_pnl(
        short_side, bars, coin_settings, k, coin_count, true, 0.0f
    );
    bool long_has_position = tm_multicoin_side_has_position(
        long_side, coin_count
    );
    bool short_has_position = tm_multicoin_side_has_position(
        short_side, coin_count
    );
    bool long_has_blocking_orders = tm_multicoin_side_has_blocking_orders(
        long_side, long_config, bars, coin_settings, k, coin_count
    );
    bool short_has_blocking_orders = tm_multicoin_side_has_blocking_orders(
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

inline void update_tm_multicoin_side_selection(
    thread TrailingMartingaleMulticoinSideState& side,
    thread const TrailingMartingaleMulticoinSideConfig& config,
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
            if (coin_override_or(coin_overrides, c, 24, -1.0f) != 0.0f) {
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
        float coin_wel = coin_override_or(coin_overrides, c, 24, -1.0f);
        float base_limit = coin_wel >= 0.0f
            ? coin_wel : config.twel / fmax(float(effective_n_positions), 1.0f);
        float allowance_pct = coin_override_or(
            coin_overrides, c, 25, config.allowance_pct
        );
        float allowed_wel = allowed_wallet_exposure_limit(
            base_limit, config.twel, allowance_pct,
            config.legacy_raw_allowance
        );
        float initial_qty_pct = coin_override_or(
            coin_overrides, c, 6, config.initial_qty_pct
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
        float coin_initial_ema_dist = coin_override_or(
            coin_overrides, c, 5, config.initial_ema_dist
        );
        float threshold = short_side
            ? upper * (1.0f + coin_initial_ema_dist)
            : lower * (1.0f - coin_initial_ema_dist);
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
        float coin_initial_ema_dist = coin_override_or(
            coin_overrides, c, 5, config.initial_ema_dist
        );
        float threshold = short_side
            ? upper * (1.0f + coin_initial_ema_dist)
            : lower * (1.0f - coin_initial_ema_dist);
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
inline int select_tm_multicoin_unstuck_coin(
    thread TrailingMartingaleMulticoinSideState& side,
    thread const TrailingMartingaleMulticoinSideConfig& config,
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
            coin_overrides, c, 28,
            config.unstuck_enabled ? 1.0f : 0.0f
        ) > 0.5f;
        const bool coin_ema_gate = coin_override_or(
            coin_overrides, c, 29,
            config.unstuck_ema_gating_enabled ? 1.0f : 0.0f
        ) > 0.5f;
        const float coin_close_pct = coin_override_or(
            coin_overrides, c, 30, config.unstuck_close_pct
        );
        const float coin_ema_dist = coin_override_or(
            coin_overrides, c, 31, config.unstuck_ema_dist
        );
        const float coin_loss_allowance_pct = coin_override_or(
            coin_overrides, c, 32, config.unstuck_loss_allowance_pct
        );
        const float coin_threshold = coin_override_or(
            coin_overrides, c, 33, config.unstuck_threshold
        );
        const float fixed_coin_wel = coin_override_or(
            coin_overrides, c, 24, -1.0f
        );
        const float coin_wel = fixed_coin_wel >= 0.0f
            ? fixed_coin_wel : effective_wel;
        const float coin_allowance_pct = coin_override_or(
            coin_overrides, c, 25, config.allowance_pct
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

// Match exact Rust's per-symbol one-way eligibility before Forager selection
// and side-wide entry-cap allocation. Opposite-held coins are excluded from
// selection; flat/flat arbitration losers retain their selected slot but emit
// no order. Reentries and closes remain unblocked.
inline void compute_tm_multicoin_one_way_initial_blocks(
    thread TrailingMartingaleMulticoinSideState& long_side,
    thread const TrailingMartingaleMulticoinSideConfig& long_config,
    constant float* long_coin_overrides,
    thread TrailingMartingaleMulticoinSideState& short_side,
    thread const TrailingMartingaleMulticoinSideConfig& short_config,
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
            long_coin_overrides, c, 24, -1.0f
        );
        const float short_wel = coin_override_or(
            short_coin_overrides, c, 24, -1.0f
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
                long_coin_overrides, c, 25, long_config.allowance_pct
            ),
            long_config.legacy_raw_allowance
        );
        const float short_allowed_wel = allowed_wallet_exposure_limit(
            short_base_limit, short_config.twel,
            coin_override_or(
                short_coin_overrides, c, 25, short_config.allowance_pct
            ),
            short_config.legacy_raw_allowance
        );
        const bool long_min_cost_eligible =
            passes_multicoin_min_effective_cost(
                filter_by_min_effective_cost, guaranteed_balance_lower,
                long_allowed_wel,
                coin_override_or(
                    long_coin_overrides, c, 6, long_config.initial_qty_pct
                ),
                coin_settings[coin_offset + 12]
            );
        const bool short_min_cost_eligible =
            passes_multicoin_min_effective_cost(
                filter_by_min_effective_cost, guaranteed_balance_lower,
                short_allowed_wel,
                coin_override_or(
                    short_coin_overrides, c, 6, short_config.initial_qty_pct
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
        const float long_dist = coin_override_or(
            long_coin_overrides, c, 5, long_config.initial_ema_dist
        );
        const float short_dist = coin_override_or(
            short_coin_overrides, c, 5, short_config.initial_ema_dist
        );
        const float dist_long = long_lower * (1.0f - long_dist)
            / close - 1.0f;
        const float dist_short = 1.0f
            - short_upper * (1.0f + short_dist) / close;
        if (dist_long >= dist_short) {
            short_order_blocked_mask |= bit;
        } else {
            long_order_blocked_mask |= bit;
        }
    }
}

inline void generate_tm_multicoin_side_orders(
    thread TrailingMartingaleMulticoinSideState& side,
    thread const TrailingMartingaleMulticoinSideConfig& config,
    thread JointPortfolioAccount& account,
    constant float* bars,
    constant int* touch_ticks,
    constant int* touch_nearest_ticks,
    constant int* touch_min_qty_bits,
    constant int* touch_min_qty_relation,
    constant float* coin_settings,
    constant float* coin_overrides,
    int k,
    int coin_count,
    bool short_side,
    int tradable_count,
    int effective_n_positions,
    int current_hsl_mode,
    bool loss_gate_enabled,
    float max_realized_loss_pct,
    bool market_orders_allowed,
    float market_order_near_touch_threshold,
    float market_order_slippage_pct,
    int forced_unstuck_coin,
    ulong one_way_initial_blocked_mask
) {
    const int C = coin_count;
    const float ddf = config.ddf;
    const float initial_ema_dist = config.initial_ema_dist;
    const float initial_qty_pct = config.initial_qty_pct;
    const float entry_threshold_base = config.entry_threshold_base;
    const float entry_threshold_we = config.entry_threshold_we;
    const float entry_threshold_v1h = config.entry_threshold_v1h;
    const float entry_threshold_v1m = config.entry_threshold_v1m;
    const float entry_retracement_base = config.entry_retracement_base;
    const float entry_retracement_we = config.entry_retracement_we;
    const float entry_retracement_v1h = config.entry_retracement_v1h;
    const float entry_retracement_v1m = config.entry_retracement_v1m;
    const float close_qty_pct = config.close_qty_pct;
    const float close_threshold_base = config.close_threshold_base;
    const float close_threshold_we = config.close_threshold_we;
    const float close_threshold_v1h = config.close_threshold_v1h;
    const float close_threshold_v1m = config.close_threshold_v1m;
    const float close_retracement_base = config.close_retracement_base;
    const float close_retracement_v1h = config.close_retracement_v1h;
    const float close_retracement_v1m = config.close_retracement_v1m;
    const float cooldown_min = config.cooldown_min;
    const float twel = config.twel;
    const bool gate_initial = config.gate_initial;
    const bool gate_reentry = config.gate_reentry;
    const int n_positions = config.n_positions;
    const float allowance_pct = config.allowance_pct;
    const bool legacy_raw_allowance = config.legacy_raw_allowance;
    const bool twel_entry_gate_enabled = config.twel_entry_gate_enabled;
    const float twel_threshold = config.twel_threshold;
    const bool wel_enforcer_enabled = config.wel_enforcer_enabled;
    const float wel_enforcer_threshold = config.wel_enforcer_threshold;
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
    thread float* entry_strategy_qty = side.entry_strategy_qty;
    thread float* entry_gen_balance = side.entry_gen_balance;
    thread float* entry_gen_allowed_wel = side.entry_gen_allowed_wel;
    thread float* entry_gen_market_price = side.entry_gen_market_price;
    thread float* entry_gen_psize = side.entry_gen_psize;
    thread float* entry_gen_pprice = side.entry_gen_pprice;
    thread float* close_qty = side.close_qty;
    thread float* secondary_close_qty = side.secondary_close_qty;
    thread float* twel_close_qty = side.twel_close_qty;
    thread float* unstuck_close_qty = side.unstuck_close_qty;
    thread float* close_gen_balance = side.close_gen_balance;
    thread float* close_gen_allowed_wel = side.close_gen_allowed_wel;
    thread float* close_gen_market_price = side.close_gen_market_price;
    thread float* close_grid_gen_psize = side.close_grid_gen_psize;
    thread float* close_grid_prefix_qty = side.close_grid_prefix_qty;
    thread float* position_open_k = side.position_open_k;
    thread float* position_last_fill_k = side.position_last_fill_k;
    thread float* contribution = side.contribution;
    thread float* minimum_entry = side.minimum_entry;
    thread float* min_since_open = side.min_since_open;
    thread float* max_since_min = side.max_since_min;
    thread float* max_since_open = side.max_since_open;
    thread float* min_since_max = side.min_since_max;
    thread int* entry_tick = side.entry_tick;
    thread int* entry_gen_initial_tick = side.entry_gen_initial_tick;
    thread int* entry_gen_touch_tick = side.entry_gen_touch_tick;
    thread int* entry_order_type = side.entry_order_type;
    thread int* close_tick = side.close_tick;
    thread int* secondary_close_tick = side.secondary_close_tick;
    thread int* twel_close_tick = side.twel_close_tick;
    thread int* unstuck_close_tick = side.unstuck_close_tick;
    thread int* close_grid_max_rungs = side.close_grid_max_rungs;
    thread int* close_grid_prefix_tick = side.close_grid_prefix_tick;
    thread bool* selected = side.selected;
    thread bool* entry_candidate = side.entry_candidate;
    thread bool* entry_recursive_market_mode =
        side.entry_recursive_market_mode;
    thread bool* close_reconstruct_after_reducer =
        side.close_reconstruct_after_reducer;
    thread bool* close_recursive_market_mode =
        side.close_recursive_market_mode;
    thread bool* entry_market = side.entry_market;
    thread bool* close_market = side.close_market;
    thread bool* secondary_close_market = side.secondary_close_market;
    thread bool* close_is_exposure_reducer =
        side.close_is_exposure_reducer;
    thread bool* close_is_unstuck_reducer =
        side.close_is_unstuck_reducer;
    thread bool* close_is_hsl_panic = side.close_is_hsl_panic;
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

    // Match calc_twel_enforcer_actions: account for every open
    // position, rank reducible positions by least adverse projected
    // loss per exposure (then symbol index), and allocate only the
    // exposure needed to cross the side-wide target.
    float twel_repair_target = twel * twel_threshold;
    if (twel_enforcer_enabled && twel_threshold > 0.0f
        && twel_repair_target > 0.0f && balance > 0.0f
        && current_twe > twel_repair_target + 1.0e-9f
        && open_position_count > 0) {
        // TWEL reduce_overweight follows the configured denominator mode
        // using the current eligible count as dynamic mode's observation.
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
            // Match exact Rust's two-stage contract: TWEL chooses and
            // accounts for its action set before the realized-loss
            // gate filters those actions.  A filtered reducer must
            // not be reallocated to another symbol.
            running_twe -= fmax(
                exposure - fmax(
                    round_step(psize[best] - reducer_qty, qty_step),
                    0.0f
                ) * pprice[best] * c_mult / balance,
                0.0f
            );
        }
    }

    // One directional thread owns every coin on this side, allowing
    // the exact one-global-intent least-stuck selector across coins.
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
            coin_overrides, c, 28, unstuck_enabled ? 1.0f : 0.0f
        ) > 0.5f;
        bool coin_ema_gate = coin_override_or(
            coin_overrides, c, 29,
            unstuck_ema_gating_enabled ? 1.0f : 0.0f
        ) > 0.5f;
        float coin_close_pct = coin_override_or(
            coin_overrides, c, 30, unstuck_close_pct
        );
        float coin_ema_dist = coin_override_or(
            coin_overrides, c, 31, unstuck_ema_dist
        );
        float coin_loss_allowance_pct = coin_override_or(
            coin_overrides, c, 32, unstuck_loss_allowance_pct
        );
        float coin_threshold = coin_override_or(
            coin_overrides, c, 33, unstuck_threshold
        );
        float fixed_coin_wel = coin_override_or(
            coin_overrides, c, 24, -1.0f
        );
        float coin_wel = fixed_coin_wel >= 0.0f
            ? fixed_coin_wel : effective_wel;
        float coin_allowance_pct = coin_override_or(
            coin_overrides, c, 25, allowance_pct
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
        entry_strategy_qty[c] = 0.0f;
        entry_recursive_market_mode[c] = false;
        entry_gen_balance[c] = 0.0f;
        entry_gen_allowed_wel[c] = 0.0f;
        entry_gen_market_price[c] = 0.0f;
        entry_gen_psize[c] = 0.0f;
        entry_gen_pprice[c] = 0.0f;
        entry_gen_initial_tick[c] = 0;
        entry_gen_touch_tick[c] = 0;
        entry_order_type[c] = 0;
        side.entry_gate_suffix_keep_count[c] = 0;
        side.entry_gate_suffix_partial_rank[c] = -1;
        side.entry_gate_suffix_partial_qty[c] = 0.0f;
        close_qty[c] = 0.0f;
        secondary_close_qty[c] = 0.0f;
        entry_market[c] = false;
        close_market[c] = false;
        secondary_close_market[c] = false;
        close_is_exposure_reducer[c] = false;
        secondary_close_tick[c] = 0;
        close_reconstruct_after_reducer[c] = false;
        close_recursive_market_mode[c] = false;
        close_is_hsl_panic[c] = false;
        close_grid_gen_psize[c] = 0.0f;
        close_grid_prefix_qty[c] = 0.0f;
        close_gen_market_price[c] = 0.0f;
        close_grid_max_rungs[c] = 500;
        close_grid_prefix_tick[c] = 0;
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
        bool coin_gate_initial = coin_override_or(
            coin_overrides, c, GATE_INITIAL_OVERRIDE_COL,
            gate_initial ? 1.0f : 0.0f
        ) > 0.5f;
        bool coin_gate_reentry = coin_override_or(
            coin_overrides, c, GATE_REENTRY_OVERRIDE_COL,
            gate_reentry ? 1.0f : 0.0f
        ) > 0.5f;
        bool initial_touch_controls = !coin_gate_initial || (short_side
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
        if (coin_gate_reentry) {
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
        float uncropped_rq = rq;
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
        bool initial_entry_allowed = !flat
            || (one_way_initial_blocked_mask & (1ul << ulong(c))) == 0ul;
        if (!selected[c] || !initial_entry_allowed
            || cooldown || balance <= 0.0f
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
        float strategy_quantity = quantity;
        bool reentry_cropped = strategy_quantity < uncropped_rq;
        int candidate_order_type = flat
            ? (short_side ? 11 : 0)
            : (partial
                ? (short_side ? 12 : 1)
                : (trailing_entry
                    ? (short_side
                        ? (reentry_cropped ? 14 : 13)
                        : (reentry_cropped ? 3 : 2))
                    : (short_side
                        ? (reentry_cropped ? 16 : 15)
                        : (reentry_cropped ? 5 : 4))));
        bool candidate_entry_market = quantity > 0.0f
            && should_use_ordinary_market_execution(
                candidate_entry_tick, !short_side, price_now, price_step,
                market_orders_allowed, market_order_near_touch_threshold
            );
        float executable_entry_price = candidate_entry_market
            ? price_now : entry_price;
        float executable_entry_min = min_entry_qty(
            executable_entry_price,
            qty_step, min_qty, min_cost, c_mult
        );
        // Exact Rust sizes the immutable strategy order first. Promotion to a
        // short market entry may require a larger executable-touch minimum.
        if (short_side && candidate_entry_market && quantity > 0.0f
            && quantity < executable_entry_min) {
            quantity = executable_entry_min;
        }
        entry_qty[c] = quantity;
        entry_strategy_qty[c] = strategy_quantity;
        entry_tick[c] = candidate_entry_tick;
        entry_order_type[c] = candidate_order_type;
        entry_market[c] = candidate_entry_market && quantity > 0.0f;
        if (quantity > 0.0f) {
            entry_gen_balance[c] = balance;
            entry_gen_allowed_wel[c] = allowed_coin_wel;
            entry_gen_market_price[c] = price_now;
            entry_gen_psize[c] = psize[c];
            entry_gen_pprice[c] = pprice[c];
            entry_gen_initial_tick[c] = initial_tick;
            entry_gen_touch_tick[c] = entry_touch;
        }
        if (coin_entry_retracement_base <= 0.0f && quantity > 0.0f
            && market_orders_allowed) {
            entry_recursive_market_mode[c] = true;
        }
        minimum_entry[c] = executable_entry_min;
        entry_candidate[c] = quantity > 0.0f;
        contribution[c] = quantity > 0.0f
            ? quantity * executable_entry_price * c_mult / balance : 0.0f;

        // Exact Rust keeps only the largest protective reducer for a
        // position before allocating its ordinary close ladder.
        float raw_twel_reducer_qty = twel_close_qty[c];
        int twel_reducer_tick = twel_close_tick[c];
        float twel_reducer_price = float(twel_reducer_tick) * price_step;
        bool twel_reducer_market = raw_twel_reducer_qty > 0.0f
            && should_use_ordinary_market_execution(
                twel_reducer_tick, short_side, price_now, price_step,
                market_orders_allowed, market_order_near_touch_threshold
            );
        if (twel_reducer_market) {
            raw_twel_reducer_qty = resize_market_close_qty(
                raw_twel_reducer_qty, psize[c], price_now,
                qty_step, min_qty, min_cost, c_mult
            );
        }
        float twel_reducer_exec_price = twel_reducer_market
            ? price_now : twel_reducer_price;
        float wel_reducer_qty = 0.0f;
        int wel_reducer_tick = 0;
        float wel_target = allowed_coin_wel
            * coin_wel_enforcer_threshold;
        if (coin_wel_enforcer_enabled
            && coin_wel_enforcer_threshold > 0.0f
            && balance > 0.0f && psize[c] > 0.0f && pprice[c] > 0.0f
            && wel_target > 0.0f && we > wel_target) {
            wel_reducer_tick = short_side ? touch_down : touch_up;
            float wel_reducer_price = float(wel_reducer_tick) * price_step;
            wel_reducer_qty = exposure_reducer_qty(
                psize[c], pprice[c], balance, wel_target,
                wel_reducer_price, qty_step, min_qty, min_cost, c_mult
            );
        }
        // Recursive strategy generation reserves the original passive WEL
        // request.  Market execution may enlarge the emitted candidate to an
        // executable-touch minimum, but that policy sizing must not alter the
        // immutable strategy ladder reconstructed from the remaining size.
        float strategy_wel_reducer_qty = wel_reducer_qty;
        bool wel_reducer_market = wel_reducer_qty > 0.0f
            && should_use_ordinary_market_execution(
                wel_reducer_tick, short_side, price_now, price_step,
                market_orders_allowed, market_order_near_touch_threshold
            );
        float wel_reducer_exec_price = float(wel_reducer_tick) * price_step;
        float raw_unstuck_reducer_qty = unstuck_close_qty[c];
        int raw_unstuck_reducer_tick = unstuck_close_tick[c];
        float unstuck_reducer_price =
            float(raw_unstuck_reducer_tick) * price_step;
        bool unstuck_reducer_market = raw_unstuck_reducer_qty > 0.0f
            && should_use_ordinary_market_execution(
                raw_unstuck_reducer_tick, short_side, price_now, price_step,
                market_orders_allowed, market_order_near_touch_threshold
            );
        if (unstuck_reducer_market) {
            raw_unstuck_reducer_qty = resize_market_close_qty(
                raw_unstuck_reducer_qty, psize[c], price_now,
                qty_step, min_qty, min_cost, c_mult
            );
        }
        float unstuck_reducer_exec_price = unstuck_reducer_market
            ? price_now : unstuck_reducer_price;

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
        if (!trailing_close && close_qty[c] > 0.0f
            && market_orders_allowed) {
            // Preserve the passive recursive strategy snapshot. On the next
            // candle it alone decides whether the full suffix is emitted;
            // ordinary market promotion is applied only after that decision.
            close_reconstruct_after_reducer[c] = true;
            close_recursive_market_mode[c] = true;
            close_gen_balance[c] = balance;
            close_gen_allowed_wel[c] = allowed_coin_wel;
            close_gen_market_price[c] = price_now;
            close_grid_gen_psize[c] = psize[c];
            close_grid_max_rungs[c] = 500;
        }
        close_market[c] = close_qty[c] > 0.0f
            && should_use_ordinary_market_execution(
                candidate_close_tick, short_side, price_now, price_step,
                market_orders_allowed, market_order_near_touch_threshold
            );
        if (close_market[c]) {
            close_qty[c] = resize_market_close_qty(
                close_qty[c], psize[c], price_now,
                qty_step, min_qty, min_cost, c_mult
            );
            minimum_close = min_entry_qty(
                price_now, qty_step, min_qty, min_cost, c_mult
            );
            minimum_close_relation = 0;
        }
        float projected_close_price = close_market[c]
            ? ordinary_market_fill_price(
                price_now, short_side,
                market_order_slippage_pct, price_step
            )
            : close_price;
        float projected_close_fee = close_market[c]
            ? coin_settings[coin_offset + 11]
            : coin_settings[coin_offset + 5];
        if (!realized_loss_proxy_allows_close(
                close_qty[c], projected_close_price, pprice[c], short_side,
                c_mult, projected_close_fee,
                loss_gate_enabled
            )) {
            if (!trailing_close && close_qty[c] > 0.0f) {
                // Exact Rust builds the complete immutable recursive
                // grid before filtering each close independently.  A
                // loss-making first rung must therefore not hide later
                // profitable rungs generated from lower exposure.
                close_reconstruct_after_reducer[c] = true;
                close_gen_balance[c] = balance;
                close_gen_allowed_wel[c] = allowed_coin_wel;
                close_grid_gen_psize[c] = psize[c];
                close_grid_max_rungs[c] = 500;
            }
            close_qty[c] = 0.0f;
            close_market[c] = false;
        }

        // calc_closes_long/short generates WEL as the first strategy close,
        // then merges the following recursive group when both quantize to
        // the same tick.  Preserve that passive prefix until the completed
        // group is known; ordinary market sizing is a later policy step and
        // must see the merged quantity once, not resize WEL independently.
        float strategy_grid_psize = fmax(
            round_step(psize[c] - strategy_wel_reducer_qty, qty_step),
            0.0f
        );
        close_grid_prefix_qty[c] = strategy_wel_reducer_qty;
        close_grid_prefix_tick[c] = strategy_wel_reducer_qty > 0.0f
            ? wel_reducer_tick : 0;
        int strategy_grid_rung_limit = strategy_wel_reducer_qty > 0.0f
            ? 499 : 500;
        bool strategy_wel_merged = false;
        CloseGroup strategy_first_group;
        if (!trailing_close && strategy_wel_reducer_qty > 0.0f
            && strategy_grid_psize > 0.0f) {
            int strategy_group_count =
                recursive_grid_close_groups_after_reducer(
                    short_side,
                    strategy_grid_psize,
                    pprice[c],
                    balance,
                    allowed_coin_wel,
                    touch_down,
                    touch_up,
                    touch_nearest_ticks[k * C + c],
                    as_type<float>(touch_min_qty_bits[k * C + c]),
                    touch_min_qty_relation[k * C + c],
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
                    wel_reducer_tick,
                    strategy_wel_reducer_qty,
                    strategy_grid_rung_limit,
                    0,
                    price_now,
                    market_orders_allowed,
                    market_order_near_touch_threshold,
                    psize[c],
                    strategy_first_group
                );
            strategy_wel_merged = strategy_group_count > 0
                && strategy_first_group.ticks == wel_reducer_tick;
        }
        if (strategy_wel_merged) {
            // The merged order retains the later ordinary-grid type in Rust,
            // so it is no longer a protective WEL candidate.  Keeping it in
            // the reconstructed ordinary ladder also lets a larger TWEL
            // reducer coexist with it exactly as the orchestrator does.
            wel_reducer_qty = 0.0f;
            wel_reducer_market = false;
            close_qty[c] = strategy_first_group.qty;
            close_tick[c] = strategy_first_group.ticks;
            close_market[c] = strategy_first_group.market;
            close_reconstruct_after_reducer[c] = true;
            close_recursive_market_mode[c] = true;
            close_gen_balance[c] = balance;
            close_gen_allowed_wel[c] = allowed_coin_wel;
            close_gen_market_price[c] = price_now;
            close_grid_gen_psize[c] = strategy_grid_psize;
            close_grid_max_rungs[c] = strategy_grid_rung_limit;
        } else if (wel_reducer_market) {
            wel_reducer_qty = resize_market_close_qty(
                wel_reducer_qty, psize[c], price_now,
                qty_step, min_qty, min_cost, c_mult
            );
            wel_reducer_exec_price = price_now;
        }

        // Exact Rust finalizes protective reducers after reserving an
        // independently valid ordinary trailing close.  That can
        // change the winning reducer at min-quantity/dust boundaries.
        bool ordinary_can_accompany_reducer = wel_reducer_qty <= 0.0f
            && trailing_close && close_qty[c] > 0.0f;
        float finalized_wel_reducer_qty = finalized_reducer_qty(
            psize[c], wel_reducer_qty,
            wel_reducer_exec_price,
            qty_step, min_qty, min_cost, c_mult
        );
        float finalized_twel_reducer_qty = ordinary_can_accompany_reducer
            ? finalized_reducer_qty_with_ordinary(
                psize[c], raw_twel_reducer_qty, twel_reducer_exec_price,
                close_qty[c], minimum_close, minimum_close_relation,
                qty_step, min_qty, min_cost, c_mult
            )
            : finalized_reducer_qty(
                psize[c], raw_twel_reducer_qty, twel_reducer_exec_price,
                qty_step, min_qty, min_cost, c_mult
            );
        float finalized_unstuck_reducer_qty = ordinary_can_accompany_reducer
            ? finalized_reducer_qty_with_ordinary(
                psize[c], raw_unstuck_reducer_qty,
                unstuck_reducer_exec_price, close_qty[c], minimum_close,
                minimum_close_relation, qty_step, min_qty, min_cost,
                c_mult
            )
            : finalized_reducer_qty(
                psize[c], raw_unstuck_reducer_qty,
                unstuck_reducer_exec_price, qty_step, min_qty, min_cost,
                c_mult
            );
        float twel_gate_price = twel_reducer_market
            ? ordinary_market_fill_price(
                price_now, short_side,
                market_order_slippage_pct, price_step
            )
            : twel_reducer_price;
        float twel_gate_fee = twel_reducer_market
            ? coin_settings[coin_offset + 11]
            : coin_settings[coin_offset + 5];
        if (!realized_loss_proxy_allows_close(
                finalized_twel_reducer_qty, twel_gate_price, pprice[c],
                short_side, c_mult, twel_gate_fee, loss_gate_enabled
            )) {
            raw_twel_reducer_qty = 0.0f;
            finalized_twel_reducer_qty = 0.0f;
            twel_reducer_market = false;
        }
        float wel_gate_price = wel_reducer_market
            ? ordinary_market_fill_price(
                price_now, short_side,
                market_order_slippage_pct, price_step
            )
            : float(wel_reducer_tick) * price_step;
        float wel_gate_fee = wel_reducer_market
            ? coin_settings[coin_offset + 11]
            : coin_settings[coin_offset + 5];
        if (!realized_loss_proxy_allows_close(
                finalized_wel_reducer_qty, wel_gate_price, pprice[c],
                short_side, c_mult, wel_gate_fee, loss_gate_enabled
            )) {
            wel_reducer_qty = 0.0f;
            finalized_wel_reducer_qty = 0.0f;
            wel_reducer_market = false;
        }
        bool use_twel = finalized_twel_reducer_qty
            > finalized_wel_reducer_qty;
        float exposure_reducer_qty = use_twel
            ? raw_twel_reducer_qty : wel_reducer_qty;
        int exposure_reducer_tick = use_twel
            ? twel_reducer_tick : wel_reducer_tick;
        float finalized_exposure_reducer_qty = use_twel
            ? finalized_twel_reducer_qty : finalized_wel_reducer_qty;
        int exposure_order_type_id = use_twel
            ? (short_side ? 21 : 10) : (short_side ? 25 : 24);
        bool use_unstuck = reducer_candidate_preferred(
            finalized_unstuck_reducer_qty, raw_unstuck_reducer_tick,
            short_side ? 20 : 9,
            finalized_exposure_reducer_qty, exposure_reducer_tick,
            exposure_order_type_id, !short_side
        );
        float reducer_qty = use_unstuck
            ? raw_unstuck_reducer_qty : exposure_reducer_qty;
        int reducer_tick = use_unstuck
            ? raw_unstuck_reducer_tick : exposure_reducer_tick;
        bool reducer_market = use_unstuck
            ? unstuck_reducer_market
            : (use_twel ? twel_reducer_market : wel_reducer_market);
        float unstuck_gate_price = unstuck_reducer_market
            ? ordinary_market_fill_price(
                price_now, short_side,
                market_order_slippage_pct, price_step
            )
            : float(raw_unstuck_reducer_tick) * price_step;
        float unstuck_gate_fee = unstuck_reducer_market
            ? coin_settings[coin_offset + 11]
            : coin_settings[coin_offset + 5];
        if (use_unstuck && !realized_loss_proxy_allows_reducer(
                finalized_unstuck_reducer_qty,
                unstuck_gate_price, pprice[c], short_side,
                c_mult, unstuck_gate_fee, true,
                loss_gate_enabled, balance, realized_pnl_cumsum_last,
                realized_pnl_cumsum_max, max_realized_loss_pct
            )) {
            use_unstuck = false;
            reducer_qty = exposure_reducer_qty;
            reducer_tick = exposure_reducer_tick;
            reducer_market = use_twel
                ? twel_reducer_market : wel_reducer_market;
        }
        if (reducer_qty > 0.0f && reducer_tick > 0) {
            float reducer_price = float(reducer_tick) * price_step;
            float reducer_exec_price = reducer_market
                ? price_now : reducer_price;
            float reducer_min = min_entry_qty(
                reducer_exec_price, qty_step, min_qty, min_cost, c_mult
            );
            if ((use_twel || use_unstuck) && wel_reducer_qty <= 0.0f
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
                    secondary_close_market[c] = close_market[c];
                }
            } else if (!trailing_close) {
                // Auto-unstuck is external to strategy generation. Preserve
                // the immutable ordinary ladder and reserve only its optional
                // strategy WEL prefix; aggregate allocation below trims that
                // ladder around whichever external reducer wins.
                float reserved_grid_qty = strategy_wel_reducer_qty;
                close_reconstruct_after_reducer[c] = true;
                close_gen_balance[c] = balance;
                close_gen_allowed_wel[c] = allowed_coin_wel;
                close_grid_gen_psize[c] = fmax(
                    round_step(
                        psize[c] - reserved_grid_qty, qty_step
                    ),
                    0.0f
                );
                close_grid_max_rungs[c] = reserved_grid_qty > 0.0f
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
            close_market[c] = reducer_market;
            if (reducer_market) {
                // Generation-time executable-touch sizing remains
                // authoritative for both trailing and recursive protective
                // closes. Recursive ordinary generation already captures
                // this snapshot above; trailing reducers need it persisted
                // here before next-candle dust allocation as well.
                close_gen_market_price[c] = price_now;
            }
            // This flag identifies any protective reducer to the recursive
            // fill/reconstruction path; unstuck has its own orthogonal flag
            // for realized-loss allowance semantics.
            close_is_exposure_reducer[c] = true;
            close_is_unstuck_reducer[c] = use_unstuck;
        }
    }
    side.entry_deferred_twel_gate = false;
    if (twel_entry_gate_enabled) {
        for (int c = 0; c < C; ++c) {
            if (entry_candidate[c] && entry_recursive_market_mode[c]) {
                side.entry_deferred_twel_gate = true;
                break;
            }
        }
    }

    if (!side.entry_deferred_twel_gate) {
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
                        : price_now - entry_price)
                        / fmax(price_now, 1.0e-12f);
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
                float room_cost = fmax(
                    (total_cap - running_twe) * balance, 0.0f
                );
                float partial = floor_step(
                    room_cost / fmax(price * c_mult, 1.0e-12f), qty_step
                );
                entry_qty[best] = partial + 1.0e-6f >= minimum_entry[best]
                    ? partial : 0.0f;
                for (int c = 0; c < C; ++c) {
                    if (entry_candidate[c] && !processed[c]) {
                        entry_qty[c] = 0.0f;
                    }
                }
                break;
            }
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
            close_is_exposure_reducer[c] = false;
            close_reconstruct_after_reducer[c] = false;
            close_recursive_market_mode[c] = false;
            close_grid_gen_psize[c] = 0.0f;
            close_grid_prefix_qty[c] = 0.0f;
            close_gen_market_price[c] = 0.0f;
            close_grid_prefix_tick[c] = 0;
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

inline float tm_multicoin_entry_initial_balance_pct(
    thread const TrailingMartingaleMulticoinSideConfig& config,
    constant float* coin_overrides,
    int effective_n_positions
) {
    float base_limit = effective_n_positions > 0
        ? config.twel / float(effective_n_positions) : 0.0f;
    float initial_qty_pct = coin_override_or(
        coin_overrides, 0, 6, config.initial_qty_pct
    );
    float allowance_pct = coin_override_or(
        coin_overrides, 0, 25, config.allowance_pct
    );
    return allowed_wallet_exposure_limit(
        base_limit, config.twel, allowance_pct, config.legacy_raw_allowance
    ) * initial_qty_pct;
}

inline void passivbot_trailing_martingale_multicoin_fused_impl(
    constant float* bars,
    constant int* fill_ticks,
    constant int* touch_ticks,
    constant int* touch_nearest_ticks,
    constant int* touch_min_qty_bits,
    constant int* touch_min_qty_relation,
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
#if PASSIVBOT_ENTRY_INTERVAL_ENABLED
    device float* entry_interval_stats,
    device int* entry_interval_counts,
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
#if PASSIVBOT_ENTRY_INTERVAL_ENABLED
    init_entry_interval_output(
        entry_interval_stats, entry_interval_counts, b
    );
#endif
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
    const TrailingMartingaleMulticoinSideConfig long_config =
        load_trailing_martingale_multicoin_side_config(params, po);
    const TrailingMartingaleMulticoinSideConfig short_config =
        load_trailing_martingale_multicoin_side_config(
            params, po + PARAM_COLS
        );
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

    TrailingMartingaleMulticoinSideState long_side;
    TrailingMartingaleMulticoinSideState short_side;
    init_trailing_martingale_multicoin_side_state(
        long_side, long_config, coin_settings,
        long_coin_overrides, C
    );
    init_trailing_martingale_multicoin_side_state(
        short_side, short_config, coin_settings,
        short_coin_overrides, C
    );

    const float starting_balance = run_settings[0];
    const float liquidation_floor = run_settings[1];
    const float interval_ms = run_settings[2];
    const float score_hysteresis = fmax(run_settings[4], 0.0f);
    const bool loss_gate_enabled = run_settings[5] < 1.0f;
    const float max_realized_loss_pct = run_settings[5];
    const float market_order_slippage_pct = fmax(run_settings[7], 0.0f);
    const bool long_hsl_panic_market = run_settings[8] > 0.5f;
    const bool short_hsl_panic_market = run_settings[9] > 0.5f;
    const bool hedge_mode = run_settings[10] > 0.5f;
    const bool market_orders_allowed = run_settings[11] > 0.5f;
    const float market_order_near_touch_threshold =
        fmax(run_settings[12], 0.0f);
    const bool filter_by_min_effective_cost = run_settings[13] > 0.5f;
    const float log_bin_scale = 127.0f / log(4000001.0f);

    JointPortfolioAccount account = init_joint_portfolio_account(
        starting_balance
    );
    TrailingMartingaleMulticoinFillState fills =
        init_trailing_martingale_multicoin_fill_state();
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
            accumulate_tm_multicoin_side_unrealized_pnl(
                long_side, bars, coin_settings, k, C, false,
                long_hsl_equity_before_fills
            );
        long_hsl_equity_before_fills =
            accumulate_tm_multicoin_side_unrealized_pnl(
                short_side, bars, coin_settings, k, C, true,
                long_hsl_equity_before_fills
            );
        bool long_fill = process_tm_multicoin_side_fills(
            long_side, long_config, account, fills,
            bars, fill_ticks, touch_ticks, touch_nearest_ticks,
            touch_min_qty_bits, touch_min_qty_relation,
            coin_settings, long_coin_overrides,
            coin_fill_counts,
#if PASSIVBOT_ENTRY_INTERVAL_ENABLED
            entry_interval_stats, entry_interval_counts,
#endif
            int(b), k, C, false, alive,
            collect_coin_fill_counts, loss_gate_enabled,
            max_realized_loss_pct, market_order_slippage_pct,
            market_order_near_touch_threshold,
            long_hsl_panic_market, long_hsl_equity_before_fills
        );
        float short_hsl_equity_before_fills = account.balance;
        short_hsl_equity_before_fills =
            accumulate_tm_multicoin_side_unrealized_pnl(
                long_side, bars, coin_settings, k, C, false,
                short_hsl_equity_before_fills
            );
        short_hsl_equity_before_fills =
            accumulate_tm_multicoin_side_unrealized_pnl(
                short_side, bars, coin_settings, k, C, true,
                short_hsl_equity_before_fills
            );
        bool short_fill = process_tm_multicoin_side_fills(
            short_side, short_config, account, fills,
            bars, fill_ticks, touch_ticks, touch_nearest_ticks,
            touch_min_qty_bits, touch_min_qty_relation,
            coin_settings, short_coin_overrides,
            coin_fill_counts,
#if PASSIVBOT_ENTRY_INTERVAL_ENABLED
            entry_interval_stats, entry_interval_counts,
#endif
            int(b), k, C, true, alive,
            collect_coin_fill_counts, loss_gate_enabled,
            max_realized_loss_pct, market_order_slippage_pct,
            market_order_near_touch_threshold,
            short_hsl_panic_market, short_hsl_equity_before_fills
        );
        bool any_fill = long_fill || short_fill;

        update_tm_multicoin_side_indicators(
            long_side, long_config, bars, hour_log_ranges,
            coin_settings, k, C
        );
        update_tm_multicoin_side_indicators(
            short_side, short_config, bars, hour_log_ranges,
            coin_settings, k, C
        );
        int long_tradable_count = count_tm_multicoin_tradable_coins(
            bars, coin_settings, long_coin_overrides, k, C
        );
        int short_tradable_count = count_tm_multicoin_tradable_coins(
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
            tm_multicoin_side_has_position(long_side, C);
        const bool short_has_position =
            tm_multicoin_side_has_position(short_side, C);
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
                        long_config.initial_qty_pct,
                        bars, coin_settings, long_coin_overrides, 24, 25, 6,
                        k, C, long_effective_n_positions,
                        min_cost_balance_lower
                    ))
                || (short_can_generate
                    && multicoin_min_cost_rejection_possible(
                        short_side.psize, short_side.coin_hsl,
                        short_config.coin_hsl_mode, short_hsl_mode,
                        short_config.twel, short_config.allowance_pct,
                        short_config.legacy_raw_allowance,
                        short_config.initial_qty_pct,
                        bars, coin_settings, short_coin_overrides, 24, 25, 6,
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
            compute_tm_multicoin_one_way_initial_blocks(
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
            ? select_tm_multicoin_unstuck_coin(
                long_side, long_config, account,
                bars, touch_ticks, coin_settings, long_coin_overrides,
                k, C, false, long_effective_n_positions,
                long_unstuck_diff
            )
            : -1;
        const int short_unstuck_candidate = short_can_generate
            ? select_tm_multicoin_unstuck_coin(
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
            update_tm_multicoin_side_selection(
                long_side, long_config, bars, coin_settings,
                long_coin_overrides, k, C, false,
                any_fill || min_cost_exact_open_uncertain,
                long_effective_n_positions, score_hysteresis,
                long_one_way_selection_blocked_mask,
                filter_by_min_effective_cost, min_cost_balance_lower
            );
            generate_tm_multicoin_side_orders(
                long_side, long_config, account,
                bars, touch_ticks, touch_nearest_ticks,
                touch_min_qty_bits, touch_min_qty_relation,
                coin_settings, long_coin_overrides,
                k, C, false, long_tradable_count,
                long_effective_n_positions, long_hsl_mode,
                loss_gate_enabled, max_realized_loss_pct,
                market_orders_allowed,
                market_order_near_touch_threshold,
                market_order_slippage_pct,
                long_unstuck_coin, long_one_way_order_blocked_mask
            );
        }
        if (short_can_generate) {
            update_tm_multicoin_side_selection(
                short_side, short_config, bars, coin_settings,
                short_coin_overrides, k, C, true,
                any_fill || min_cost_exact_open_uncertain,
                short_effective_n_positions, score_hysteresis,
                short_one_way_selection_blocked_mask,
                filter_by_min_effective_cost, min_cost_balance_lower
            );
            generate_tm_multicoin_side_orders(
                short_side, short_config, account,
                bars, touch_ticks, touch_nearest_ticks,
                touch_min_qty_bits, touch_min_qty_relation,
                coin_settings, short_coin_overrides,
                k, C, true, short_tradable_count,
                short_effective_n_positions, short_hsl_mode,
                loss_gate_enabled, max_realized_loss_pct,
                market_orders_allowed,
                market_order_near_touch_threshold,
                market_order_slippage_pct,
                short_unstuck_coin, short_one_way_order_blocked_mask
            );
        }
        if (filter_by_min_effective_cost
            && (long_can_generate || short_can_generate)) {
            min_cost_exact_open_uncertain = true;
        }

        float forced_delist_equity = account.balance;
        forced_delist_equity =
            accumulate_tm_multicoin_side_unrealized_pnl(
                long_side, bars, coin_settings, k, C, false,
                forced_delist_equity
            );
        forced_delist_equity =
            accumulate_tm_multicoin_side_unrealized_pnl(
                short_side, bars, coin_settings, k, C, true,
                forced_delist_equity
            );
        bool forced_delist_fill = false;
        if (alive && account.balance > 0.0f) {
            forced_delist_fill = force_close_tm_multicoin_delisted_fused(
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
            bool hsl_valid = update_tm_multicoin_dual_side_hsl(
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
    scalars[scalar_offset + 21] = tm_multicoin_entry_initial_balance_pct(
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
    scalars[scalar_offset + 59] = tm_multicoin_entry_initial_balance_pct(
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

kernel void passivbot_trailing_martingale_multicoin_fused(
    constant float* bars,
    constant int* fill_ticks,
    constant int* touch_ticks,
    constant int* touch_nearest_ticks,
    constant int* touch_min_qty_bits,
    constant int* touch_min_qty_relation,
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
#if PASSIVBOT_ENTRY_INTERVAL_ENABLED
    device float* entry_interval_stats,
    device int* entry_interval_counts,
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
    passivbot_trailing_martingale_multicoin_fused_impl(
        bars, fill_ticks, touch_ticks, touch_nearest_ticks,
        touch_min_qty_bits, touch_min_qty_relation,
        hour_log_ranges, coin_settings,
        long_coin_overrides, short_coin_overrides,
        params, run_settings, sizes, end_steps,
#if PASSIVBOT_BTC_PRICES_ENABLED
        btc_prices,
#endif
#if PASSIVBOT_EQUITY_BALANCE_DIFF_ENABLED
        equity_balance_diff,
#endif
#if PASSIVBOT_ENTRY_INTERVAL_ENABLED
        entry_interval_stats, entry_interval_counts,
#endif
        daily, scalars, gap_hist, coin_fill_counts,
#ifdef PASSIVBOT_STRATEGY_EQ_RECOVERY_DISTRIBUTION_ENABLED
        recovery_samples,
#endif
        b
    );
}

inline void passivbot_trailing_martingale_multicoin_impl(
    constant float* bars,
    constant int* fill_ticks,
    constant int* touch_ticks,
    constant int* touch_nearest_ticks,
    constant int* touch_min_qty_bits,
    constant int* touch_min_qty_relation,
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
#if PASSIVBOT_ENTRY_INTERVAL_ENABLED
    device float* entry_interval_stats,
    device int* entry_interval_counts,
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
#if PASSIVBOT_ENTRY_INTERVAL_ENABLED
    init_entry_interval_output(
        entry_interval_stats, entry_interval_counts, b
    );
#endif
    const int stop_k = clamp(end_steps[b], 1, T - 1);
    const bool collect_coin_fill_counts = run_settings[6] > 0.5f;
    if (collect_coin_fill_counts) {
        for (int c = 0; c < C; ++c) {
            coin_fill_counts[int(b) * C + c] = 0.0f;
        }
    }

    const int po = int(b) * PARAM_COLS;
    const TrailingMartingaleMulticoinSideConfig config =
        load_trailing_martingale_multicoin_side_config(params, po);
    const float initial_qty_pct = config.initial_qty_pct;
    const float twel = config.twel;
    const int n_positions = config.n_positions;
    const float allowance_pct = config.allowance_pct;
    const bool legacy_raw_allowance = config.legacy_raw_allowance;
    TrailingMartingaleMulticoinSideState side;
    init_trailing_martingale_multicoin_side_state(
        side, config, coin_settings, coin_overrides, C
    );
    thread HslState& hsl = side.hsl;
    const bool coin_hsl_mode = config.coin_hsl_mode;
    thread HslState* coin_hsl = side.coin_hsl;
    const float starting_balance = run_settings[0];
    const float liquidation_floor = run_settings[1];
    const float interval_ms = run_settings[2];
    const float score_hysteresis = fmax(run_settings[4], 0.0f);
    const bool loss_gate_enabled = run_settings[5] < 1.0f;
    const float max_realized_loss_pct = run_settings[5];
    const float market_order_slippage_pct = fmax(run_settings[7], 0.0f);
    const bool hsl_panic_market = run_settings[8] > 0.5f;
    const bool market_orders_allowed = run_settings[9] > 0.5f;
    const float market_order_near_touch_threshold =
        fmax(run_settings[10], 0.0f);
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
    thread float& realized_pnl_cumsum_max = account.realized_pnl_peak;
    TrailingMartingaleMulticoinFillState fills =
        init_trailing_martingale_multicoin_fill_state();
    thread float& pnl_recovery_peak = fills.pnl_recovery_peak;
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
    thread int& max_tradable_seen = side.max_tradable_seen;
    float run_peak = -INFINITY;
    float max_dd = 0.0f;
    float total_wallet_exposure_max = 0.0f;
    float total_wallet_exposure_mean = 0.0f;
    float total_wallet_exposure_samples = 0.0f;
    thread float& held_max_min = fills.held_max_min;
    thread float& held_sum_min = fills.held_sum_min;
    thread float& held_count = fills.held_count;
    thread float& position_unchanged_max_min =
        fills.position_unchanged_max_min;
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
            accumulate_tm_multicoin_side_unrealized_pnl(
                side, bars, coin_settings, k, C, short_side, balance
            );
        bool any_fill = process_tm_multicoin_side_fills(
            side, config, account, fills,
            bars, fill_ticks, touch_ticks, touch_nearest_ticks,
            touch_min_qty_bits, touch_min_qty_relation,
            coin_settings, coin_overrides, coin_fill_counts,
#if PASSIVBOT_ENTRY_INTERVAL_ENABLED
            entry_interval_stats, entry_interval_counts,
#endif
            int(b), k, C, short_side, alive,
            collect_coin_fill_counts, loss_gate_enabled,
            max_realized_loss_pct, market_order_slippage_pct,
            market_order_near_touch_threshold,
            hsl_panic_market, hsl_equity_before_fills
        );
        update_tm_multicoin_side_indicators(
            side, config, bars, hour_log_ranges, coin_settings, k, C
        );

        int tradable_count = count_tm_multicoin_tradable_coins(
            bars, coin_settings, coin_overrides, k, C
        );
        const bool post_fill_balance_depleted = isfinite(balance) && balance <= 0.0f;
        const bool past_activation_guard =
            k > max(global_warmup, 1) && k >= requested_start_k;
        if (alive && !post_fill_balance_depleted && past_activation_guard) {
            max_tradable_seen = max(max_tradable_seen, tradable_count);
        }
        const int effective_n_positions =
            wallet_exposure_denominator_n_positions(
                n_positions, max_tradable_seen
            );
        const bool can_generate = alive && effective_n_positions > 0
            && max_tradable_seen > 0 && past_activation_guard;
        equity_started = equity_started || can_generate;
        bool has_hsl_position = tm_multicoin_side_has_position(side, C);
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
                config.legacy_raw_allowance, config.initial_qty_pct,
                bars, coin_settings, coin_overrides, 24, 25, 6,
                k, C, effective_n_positions,
                min_cost_balance_lower
            )) {
            min_cost_exact_open_uncertain = true;
            min_cost_balance_lower = 0.0f;
        }

        if (can_generate) {
            update_tm_multicoin_side_selection(
                side, config, bars, coin_settings, coin_overrides,
                k, C, short_side,
                any_fill || min_cost_exact_open_uncertain,
                effective_n_positions,
                score_hysteresis, 0ul,
                filter_by_min_effective_cost, min_cost_balance_lower
            );

            generate_tm_multicoin_side_orders(
                side, config, account,
                bars, touch_ticks, touch_nearest_ticks,
                touch_min_qty_bits, touch_min_qty_relation,
                coin_settings, coin_overrides,
                k, C, short_side, tradable_count,
                effective_n_positions, current_hsl_mode,
                loss_gate_enabled, max_realized_loss_pct,
                market_orders_allowed,
                market_order_near_touch_threshold,
                market_order_slippage_pct,
                -2, 0ul
            );
            if (filter_by_min_effective_cost) {
                min_cost_exact_open_uncertain = true;
            }
        }

        float forced_delist_equity =
            accumulate_tm_multicoin_side_unrealized_pnl(
                side, bars, coin_settings, k, C, short_side, balance
            );
        bool forced_delist_fill = false;
        if (alive && balance > 0.0f) {
            forced_delist_fill = force_close_tm_multicoin_delisted_one_side(
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
            n_positions, max_tradable_seen
        );
    float entry_base_limit = entry_effective_n_positions > 0
        ? twel / float(entry_effective_n_positions) : 0.0f;
    float entry_initial_qty_pct = coin_override_or(
        coin_overrides, 0, 6, initial_qty_pct
    );
    float entry_allowance_pct = coin_override_or(
        coin_overrides, 0, 25, allowance_pct
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

kernel void passivbot_trailing_martingale_multicoin(
    constant float* bars,
    constant int* fill_ticks,
    constant int* touch_ticks,
    constant int* touch_nearest_ticks,
    constant int* touch_min_qty_bits,
    constant int* touch_min_qty_relation,
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
#if PASSIVBOT_ENTRY_INTERVAL_ENABLED
    device float* entry_interval_stats,
    device int* entry_interval_counts,
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
    passivbot_trailing_martingale_multicoin_impl(
        bars, fill_ticks, touch_ticks, touch_nearest_ticks,
        touch_min_qty_bits, touch_min_qty_relation,
        hour_log_ranges, coin_settings,
        coin_overrides, params, run_settings,
        sizes, end_steps,
#if PASSIVBOT_BTC_PRICES_ENABLED
        btc_prices,
#endif
#if PASSIVBOT_EQUITY_BALANCE_DIFF_ENABLED
        equity_balance_diff,
#endif
#if PASSIVBOT_ENTRY_INTERVAL_ENABLED
        entry_interval_stats, entry_interval_counts,
#endif
        daily, scalars, gap_hist, coin_fill_counts,
#ifdef PASSIVBOT_STRATEGY_EQ_RECOVERY_DISTRIBUTION_ENABLED
        recovery_samples,
#endif
        b, short_side
    );
}

kernel void passivbot_trailing_martingale_multicoin_long(
    constant float* bars,
    constant int* fill_ticks,
    constant int* touch_ticks,
    constant int* touch_nearest_ticks,
    constant int* touch_min_qty_bits,
    constant int* touch_min_qty_relation,
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
#if PASSIVBOT_ENTRY_INTERVAL_ENABLED
    device float* entry_interval_stats,
    device int* entry_interval_counts,
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
    passivbot_trailing_martingale_multicoin_impl(
        bars, fill_ticks, touch_ticks, touch_nearest_ticks,
        touch_min_qty_bits, touch_min_qty_relation,
        hour_log_ranges, coin_settings,
        coin_overrides, params, run_settings,
        sizes, end_steps,
#if PASSIVBOT_BTC_PRICES_ENABLED
        btc_prices,
#endif
#if PASSIVBOT_EQUITY_BALANCE_DIFF_ENABLED
        equity_balance_diff,
#endif
#if PASSIVBOT_ENTRY_INTERVAL_ENABLED
        entry_interval_stats, entry_interval_counts,
#endif
        daily, scalars, gap_hist, coin_fill_counts,
#ifdef PASSIVBOT_STRATEGY_EQ_RECOVERY_DISTRIBUTION_ENABLED
        recovery_samples,
#endif
        b, false
    );
}
