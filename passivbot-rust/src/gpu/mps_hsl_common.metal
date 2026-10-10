// Shared GPU HSL adapters and reporting. Strategy kernels provide simulator
// facts. The opt-in native reconstruction keeps factual history and disposable
// event scratch resident; ordinary result payloads contain compact metrics.

#ifndef PASSIVBOT_HSL_CUTOFF_CACHE_ENABLED
#define PASSIVBOT_HSL_CUTOFF_CACHE_ENABLED 1
#endif

#ifndef PASSIVBOT_HSL_INCREMENTAL_ENABLED
#define PASSIVBOT_HSL_INCREMENTAL_ENABLED PASSIVBOT_HSL_FACTUAL_ONLY
#endif

#ifndef PASSIVBOT_HSL_EMPTY_SCOPE_ENABLED
#define PASSIVBOT_HSL_EMPTY_SCOPE_ENABLED PASSIVBOT_HSL_FACTUAL_ONLY
#endif

#ifndef PASSIVBOT_HSL_EMA_TAIL_ENABLED
#define PASSIVBOT_HSL_EMA_TAIL_ENABLED 0
#endif

#ifndef PASSIVBOT_HSL_EMA_TAIL_SAMPLES_ENABLED
#define PASSIVBOT_HSL_EMA_TAIL_SAMPLES_ENABLED (PASSIVBOT_HSL_FACTUAL_ONLY && PASSIVBOT_HSL_EMA_TAIL_ENABLED && PASSIVBOT_HSL_FACTS_ENABLED > 0)
#endif

#ifndef PASSIVBOT_HSL_RAW_DRAWDOWN_ENABLED
#define PASSIVBOT_HSL_RAW_DRAWDOWN_ENABLED 0
#endif

#ifndef PASSIVBOT_HSL_RAW_TAIL_ENABLED
#define PASSIVBOT_HSL_RAW_TAIL_ENABLED 0
#endif

#ifndef PASSIVBOT_HSL_RAW_TAIL_CAPACITY
#define PASSIVBOT_HSL_RAW_TAIL_CAPACITY 1
#endif

#if PASSIVBOT_HSL_RAW_TAIL_ENABLED && PASSIVBOT_HSL_RAW_TAIL_CAPACITY < 1
#error Raw drawdown tail capacity must be positive
#endif

#ifndef PASSIVBOT_HSL_DIAGNOSTICS_ENABLED
#define PASSIVBOT_HSL_DIAGNOSTICS_ENABLED 1
#endif

#define HSL_EMA_TAIL_BINS 32

constant int HSL_SIGNAL_UNIFIED = 0;
constant int HSL_SIGNAL_PSIDE = 1;
constant int HSL_SIGNAL_COIN = 2;

#if PASSIVBOT_HSL_FACTS_ENABLED > 0
struct HslReplayContext;
#endif
struct HslState {
#if PASSIVBOT_HSL_FACTS_ENABLED > 0
    HslPairRing facts;
#if PASSIVBOT_HSL_CUTOFF_CACHE_ENABLED
    HslScopeCutoffCache cutoff_cache;
#if PASSIVBOT_HSL_FACTUAL_ONLY && PASSIVBOT_HSL_INCREMENTAL_ENABLED
    HslScopeCursor scope_cursor;
#endif
#endif
    device HslPairEvent* fact_events;
    thread HslReplayContext* replay;
    int replay_side;
    int replay_coin;
#endif
    HslController hsl;
    device HslNode* hsl_tree;
#if !PASSIVBOT_HSL_FACTUAL_ONLY
    device int* hsl_times;
    device float* hsl_realized;
#endif
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

#if PASSIVBOT_HSL_FACTS_ENABLED > 0
struct HslReplayContext {
    thread HslState* coins[2];
    bool short_side[2];
    int side_count;
    int coin_count;
    int coin_columns;
    constant float* bars;
    constant float* settings;
    bool enabled;
};

inline HslReplayContext hsl_replay_context(
    constant float* bars, constant float* settings, int coin_count, int coin_columns,
    bool enabled
) {
    HslReplayContext context;
    context.side_count = 0; context.coin_count = coin_count;
    context.coin_columns = coin_columns; context.bars = bars; context.settings = settings;
    context.enabled = enabled;
    return context;
}

inline void attach_hsl_replay_side(
    thread HslReplayContext& context, thread HslState& aggregate,
    thread HslState* coins, int side, bool short_side
) {
    context.coins[side] = coins; context.short_side[side] = short_side;
    context.side_count = max(context.side_count, side + 1);
    aggregate.replay = &context; aggregate.replay_side = side; aggregate.replay_coin = -1;
    for (int c = 0; c < context.coin_count; ++c) {
        coins[c].replay = &context; coins[c].replay_side = side; coins[c].replay_coin = c;
    }
}

inline bool replay_factual_hsl(
    thread HslState& h, float budget, int minute, bool exposed, bool terminal
) {
    thread HslReplayContext& context = *h.replay;
    HslPairRing rings[2 * MAX_COINS];
    HslScopePair pairs[2 * MAX_COINS];
    float current_sizes[2 * MAX_COINS], quantity_steps[2 * MAX_COINS], prior_sizes[2 * MAX_COINS];
    int cursors[2 * MAX_COINS], pair_ids[2 * MAX_COINS];
    int count = 0;
    int first = max(minute - h.hsl_lookback, 0);
    for (int s = 0; s < context.side_count; ++s) {
        if (h.signal_mode != HSL_SIGNAL_UNIFIED && s != h.replay_side) continue;
        for (int c = 0; c < context.coin_count; ++c) {
            if (h.replay_coin >= 0 && (s != h.replay_side || c != h.replay_coin)) continue;
            thread HslState& source = context.coins[s][c];
            if (!source.facts.enabled || source.facts.state->failure != 0) return false;
            HslPairFacts view = hsl_pair_ring_view(source.facts, first);
            // Observed flat, without any retained activity, has no valuation
            // contribution. It does not require an invented quote or multiplier.
            if (source.facts.state->current_size == 0.0f && view.count == 0) continue;
            if (count >= 2 * MAX_COINS) return false;
            rings[count] = source.facts;
            pair_ids[count] = s * MAX_COINS + c;
            current_sizes[count] = source.facts.state->current_size;
            quantity_steps[count] = context.settings[c * context.coin_columns];
            thread HslScopePair& pair = pairs[count];
            pair.events = source.fact_events;
            pair.current_size = current_sizes[count];
            pair.current_basis = source.facts.state->current_basis;
            pair.multiplier = context.settings[c * context.coin_columns + 4];
            pair.short_side = context.short_side[s];
            pair.sequences = reinterpret_cast<device const int*>(source.facts.records) + 6;
            pair.sequence_stride = 8;
            pair.prices = nullptr; pair.price_stride = 1;
            pair.candles = context.bars + c * 4 + 2;
            pair.candle_stride = context.coin_count * 4;
            pair.candle_first = int(context.settings[c * context.coin_columns + 6]);
            pair.candle_last = int(context.settings[c * context.coin_columns + 7]);
            pair.fallback_price = view.count > 0 ? hsl_pair_fact(view, view.count - 1).price : 0.0f;
            pair.current_mark = hsl_scope_price_at(pair, minute, first, minute);
            ++count;
        }
    }
    HslScopeResult result;
    if (count == 0) {
#if PASSIVBOT_HSL_FACTUAL_ONLY && PASSIVBOT_HSL_INCREMENTAL_ENABLED && PASSIVBOT_HSL_CUTOFF_CACHE_ENABLED
        h.scope_cursor.valid = false;
#endif
        h.hsl.raw = h.hsl.ema = 0.0f; h.hsl.action = 0; h.hsl.flat_minute = -1;
        return !exposed;
    }
    const float span = 2.0f / h.alpha - 1.0f;
#if PASSIVBOT_HSL_EMPTY_SCOPE_ENABLED
    if (h.signal_mode == HSL_SIGNAL_UNIFIED && exposed && !terminal) {
        bool empty = true;
        for (int p = 0; p < count; ++p)
            empty = empty && hsl_pair_ring_view(rings[p], first).count == 0;
        if (empty) {
            for (int p = 0; p < count; ++p) {
                pairs[p].facts = hsl_pair_ring_view(rings[p], first);
                if (!hsl_reconstruct_pair(pairs[p].facts, pairs[p].current_size,
                    pairs[p].current_basis, pairs[p].short_side, quantity_steps[p],
                    pairs[p].events, pairs[p].history)) return false;
            }
            if (hsl_compose_empty_native_scope(pairs, count, first, minute,
                budget, span, h.red_threshold, h.cooldown_minutes,
                h.restart_policy == 2, result)) {
#if PASSIVBOT_HSL_FACTUAL_ONLY && PASSIVBOT_HSL_INCREMENTAL_ENABLED && PASSIVBOT_HSL_CUTOFF_CACHE_ENABLED
                h.scope_cursor.valid = false;
#endif
                h.hsl.raw = result.raw; h.hsl.ema = result.ema;
                h.hsl.action = result.action; h.hsl.flat_minute = result.flat_minute;
                h.hsl.last_observed = minute; h.hsl.exposed = exposed;
                return true;
            }
        }
    }
#endif
    HslScopeCutoff cutoff;
#if PASSIVBOT_HSL_CUTOFF_CACHE_ENABLED
    bool reused_cutoff;
    if (!hsl_scope_history_cutoff_cached(rings, current_sizes, quantity_steps,
        pair_ids, count, cursors, prior_sizes, h.cutoff_cache, cutoff, reused_cutoff)) return false;
#else
    if (!hsl_scope_history_cutoff(rings, current_sizes, quantity_steps, count,
        cursors, prior_sizes, cutoff)) return false;
#endif
    if (cutoff.found) first = max(first, cutoff.minute);
    bool advanced = false;
#if PASSIVBOT_HSL_FACTUAL_ONLY && PASSIVBOT_HSL_INCREMENTAL_ENABLED && PASSIVBOT_HSL_CUTOFF_CACHE_ENABLED
    if (reused_cutoff && exposed && !terminal) {
        HslPairSum upnl;
        hsl_pair_sum_reset(upnl, 0.0f);
        bool valid_marks = true;
        for (int p = 0; p < count; ++p) {
            thread const HslScopePair& pair = pairs[p];
            valid_marks = valid_marks && isfinite(pair.current_mark) && pair.current_mark > 0.0f
                && isfinite(pair.multiplier) && pair.multiplier > 0.0f;
            float direction = pair.short_side ? -1.0f : 1.0f;
            float value = pair.current_size == 0.0f ? 0.0f
                : direction * fabs(pair.current_size) * pair.multiplier
                    * (pair.current_mark - pair.current_basis);
            valid_marks = valid_marks && isfinite(value);
            hsl_pair_sum_add(upnl, value);
        }
        advanced = valid_marks && hsl_advance_scope(h.scope_cursor, first, minute,
            hsl_pair_sum_value(upnl), budget, 2.0f / (span + 1.0f), h.red_threshold,
            h.cooldown_minutes, h.restart_policy == 2, result);
    }
#endif
    if (!advanced) {
        for (int p = 0; p < count; ++p) {
            pairs[p].facts = hsl_pair_ring_view_after(rings[p], first, cutoff);
            if (!hsl_reconstruct_pair(pairs[p].facts, pairs[p].current_size,
                pairs[p].current_basis, pairs[p].short_side, quantity_steps[p],
                pairs[p].events, pairs[p].history)) return false;
        }
        if (!hsl_compose_scope(pairs, count, first, minute, false, budget,
            span, h.red_threshold, h.cooldown_minutes, h.restart_policy == 2, result
#if PASSIVBOT_HSL_FACTUAL_ONLY && PASSIVBOT_HSL_INCREMENTAL_ENABLED && PASSIVBOT_HSL_CUTOFF_CACHE_ENABLED
            , nullptr, 0, &h.scope_cursor
#endif
        )) return false;
    }
#if PASSIVBOT_HSL_FACTUAL_ONLY && PASSIVBOT_HSL_INCREMENTAL_ENABLED && PASSIVBOT_HSL_CUTOFF_CACHE_ENABLED
    if (terminal) h.scope_cursor.valid = false;
#endif
    h.hsl.raw = terminal && result.latest_flat_minute == minute
        ? result.latest_flat_raw : result.raw;
    h.hsl.ema = terminal && result.latest_flat_minute == minute
        ? result.latest_flat_ema : result.ema;
    h.hsl.action = result.action; h.hsl.flat_minute = result.flat_minute;
    h.hsl.last_observed = minute; h.hsl.exposed = exposed;
    return true;
}
#endif




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
    int completed_day_count;
    int daily_tail_count;
    float daily_drawdown_top[PASSIVBOT_HSL_RAW_TAIL_CAPACITY];
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
    stats.completed_day_count = 0;
    stats.daily_tail_count = 0;
    for (int i = 0; i < PASSIVBOT_HSL_RAW_TAIL_CAPACITY; ++i)
        stats.daily_drawdown_top[i] = 0.0f;
#endif
    return stats;
}

inline void flush_hsl_strategy_equity_daily_drawdown(
    thread HslStrategyEquityStats& stats
) {
#if PASSIVBOT_HSL_RAW_TAIL_ENABLED
    if (stats.current_drawdown_day < 0) return;
    stats.completed_day_count += 1;
    // The registered timeline bounds floor(observed days / 100). Keep only
    // that many largest daily maxima, with no approximate cutoff-bin mean.
    const float value = stats.current_day_drawdown_worst;
    if (stats.daily_tail_count == PASSIVBOT_HSL_RAW_TAIL_CAPACITY
        && value <= stats.daily_drawdown_top[PASSIVBOT_HSL_RAW_TAIL_CAPACITY - 1])
        return;
    int position = min(stats.daily_tail_count, PASSIVBOT_HSL_RAW_TAIL_CAPACITY - 1);
    while (position > 0 && value > stats.daily_drawdown_top[position - 1]) {
        stats.daily_drawdown_top[position] = stats.daily_drawdown_top[position - 1];
        --position;
    }
    stats.daily_drawdown_top[position] = value;
    stats.daily_tail_count = min(stats.daily_tail_count + 1, PASSIVBOT_HSL_RAW_TAIL_CAPACITY);
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
    int sample_count = stats.completed_day_count
        + (stats.current_drawdown_day >= 0 ? 1 : 0);
    if (sample_count <= 0) return 0.0f;
    int worst_n = max(sample_count / 100, 1);
#if PASSIVBOT_HSL_RAW_DRAWDOWN_ENABLED
    if (worst_n == 1) return stats.drawdown_max;
#endif
    // An undersized direct shader invocation cannot supply a valid tail.
    // Prepared runners specialize capacity from the complete input horizon.
    if (worst_n > PASSIVBOT_HSL_RAW_TAIL_CAPACITY) return NAN;
    int stored = 0;
    bool current = stats.current_drawdown_day >= 0;
    float total = 0.0f;
    // Merge the unflushed current day without mutating the retained state:
    // repeated metric queries and temporal replay snapshots remain identical.
    for (int i = 0; i < worst_n; ++i) {
        if (current && (stored >= stats.daily_tail_count
            || stats.current_day_drawdown_worst >= stats.daily_drawdown_top[stored])) {
            total += stats.current_day_drawdown_worst;
            current = false;
        } else {
            total += stats.daily_drawdown_top[stored++];
        }
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

// Native factual replay captures requested observations in bounded device
// storage and reduces the actual largest floor(1%) after an accepted replay.
// The legacy observation replay retains its approximate log histogram. Neither
// path changes the controller's signal or clock. Unrequested work is compiled out.
struct HslDrawdownEmaTailStats {
#if PASSIVBOT_HSL_EMA_TAIL_ENABLED && PASSIVBOT_HSL_FACTUAL_ONLY
    float last_sample;
#elif PASSIVBOT_HSL_EMA_TAIL_ENABLED
    float sample_count;
    float counts[HSL_EMA_TAIL_BINS];
    float sums[HSL_EMA_TAIL_BINS];
#else
    float unused;
#endif
};

inline HslDrawdownEmaTailStats init_hsl_drawdown_ema_tail_stats() {
    HslDrawdownEmaTailStats stats;
#if PASSIVBOT_HSL_EMA_TAIL_ENABLED && PASSIVBOT_HSL_FACTUAL_ONLY
    stats.last_sample = NAN;
#elif PASSIVBOT_HSL_EMA_TAIL_ENABLED
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
    // EMA tails retain their existing histogram independently of raw daily tails.
    float scaled = (log2(fmax(value, 0.00006103515625f)) + 14.0f)
        * 2.2857142857142856f;
    return clamp(int(floor(scaled)), 0, HSL_EMA_TAIL_BINS - 1);
}

inline void update_hsl_drawdown_ema_tail_stats(
    thread HslDrawdownEmaTailStats& stats,
    float drawdown_ema
) {
#if PASSIVBOT_HSL_EMA_TAIL_ENABLED
    if (!isfinite(drawdown_ema)) return;
    float value = fabs(drawdown_ema);
#if PASSIVBOT_HSL_FACTUAL_ONLY
    stats.last_sample = value;
#else
    int bin = hsl_drawdown_ema_tail_bin(value);
    stats.sample_count += 1.0f;
    stats.counts[bin] += 1.0f;
    stats.sums[bin] += value;
#endif
#endif
}

inline float hsl_drawdown_ema_mean_worst_1pct(
    thread HslDrawdownEmaTailStats& stats
) {
#if PASSIVBOT_HSL_EMA_TAIL_SAMPLES_ENABLED
    // A successful native result must replace this sentinel with the device
    // reduction. An unreduced observational placeholder is never valid fitness.
    return NAN;
#elif PASSIVBOT_HSL_EMA_TAIL_ENABLED && !PASSIVBOT_HSL_FACTUAL_ONLY
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
    int scope, int capacity, int tree_size, int lookback, bool initialize, bool owner,
    int fact_capacity = 0
) {
#if PASSIVBOT_HSL_FACTUAL_ONLY
    h.hsl_tree = fact_capacity > 0
        ? trees + scope * hsl_storage_nodes(capacity, tree_size, fact_capacity) : nullptr;
#else
    h.hsl_tree = trees + scope * hsl_storage_nodes(capacity, tree_size, fact_capacity);
    h.hsl_times = rows + scope * capacity * 2;
    h.hsl_realized = reinterpret_cast<device float*>(h.hsl_times + capacity);
#endif
    h.hsl_lookback = lookback;
    h.hsl_owner = owner;
#if PASSIVBOT_HSL_FACTS_ENABLED > 0
#if PASSIVBOT_HSL_FACTUAL_ONLY
    const int fact_offset = 0;
#else
    const int fact_offset = 2 * tree_size + (capacity + 3) / 4;
#endif
    h.facts.state = reinterpret_cast<device HslPairRingState*>(h.hsl_tree + fact_offset);
    h.facts.records = reinterpret_cast<device HslPairRecord*>(h.hsl_tree + fact_offset + 2);
    h.fact_events = reinterpret_cast<device HslPairEvent*>(h.facts.records + fact_capacity);
    h.facts.capacity = fact_capacity;
    // Preserve the opening boundary used before the next bar's fills.
    h.facts.lookback = lookback + 1;
    h.facts.enabled = h.enabled;
    if (initialize) hsl_pair_ring_reset(h.facts);
#endif
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
    bool initialize, bool owner, int fact_capacity = 0
) {
    bind_hsl(aggregate, trees, rows, scope_base,
        PASSIVBOT_HSL_CAPACITY, PASSIVBOT_HSL_TREE_SIZE,
        PASSIVBOT_HSL_LOOKBACK, initialize, owner, fact_capacity);
    for (int c = 0; c < coin_count; ++c) {
        bind_hsl(coins[c], trees, rows, scope_base + 1 + c,
            PASSIVBOT_HSL_CAPACITY, PASSIVBOT_HSL_TREE_SIZE,
            PASSIVBOT_HSL_LOOKBACK, initialize, true, fact_capacity);
#if PASSIVBOT_HSL_FACTS_ENABLED > 0
        coins[c].facts.enabled = coins[c].enabled
            || (aggregate.enabled && aggregate.signal_mode != HSL_SIGNAL_COIN);
#endif
    }
}

inline bool valid_hsl_multicoin_hsl(
    thread HslState& aggregate, thread HslState* coins, int coin_count
) {
#if PASSIVBOT_HSL_FACTS_ENABLED > 0
    for (int c = 0; c < coin_count; ++c)
        if (coins[c].facts.enabled && coins[c].facts.state->failure != 0) return false;
#endif
    if (aggregate.signal_mode != HSL_SIGNAL_COIN)
        return !aggregate.enabled || aggregate.hsl_valid;
    for (int c = 0; c < coin_count; ++c)
        if (coins[c].enabled && !coins[c].hsl_valid) return false;
    return true;
}

inline float hsl_multicoin_failure_status(
    thread HslState& aggregate, thread HslState* coins, int coin_count
) {
    float status = -4.0f;
#if PASSIVBOT_HSL_FACTS_ENABLED > 0
    for (int c = 0; c < coin_count; ++c) {
        if (!coins[c].facts.enabled) continue;
        if (coins[c].facts.state->failure == 1) status = fmin(status, -6.0f);
        if (coins[c].facts.state->failure == 2) status = -7.0f;
    }
#endif
    return status;
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

// Observational lifecycle: current RED starts reporting before a terminal fill.
// These counters never grant trading permission or retain a panic commitment.
inline void begin_hsl_report(thread HslState& h, int minute, float score) {
    h.triggers += 1.0f;
    h.current_red_start_k = float(minute);
    if (h.current_halt_start_k < 0.0f) h.current_halt_start_k = float(minute);
#if PASSIVBOT_HSL_DIAGNOSTICS_ENABLED
    h.trigger_drawdown_sum += score;
    h.trigger_drawdown_count += 1.0f;
    // A restart contributes at most one following retrigger, per scope.
    if (h.last_restart_k >= 0.0f) {
        h.restart_retrigger_count += 1.0f;
        h.last_restart_k = -1.0f;
    }
#endif
}

inline void restart_hsl_report(thread HslState& h, int minute) {
    h.restarts += 1.0f;
#if PASSIVBOT_HSL_DIAGNOSTICS_ENABLED
    float duration = fmax(float(minute) - h.current_halt_start_k, 0.0f);
    h.halt_duration_sum_steps += duration;
    h.halt_duration_max_steps = fmax(h.halt_duration_max_steps, duration);
    h.halt_duration_count += 1.0f;
    h.last_restart_k = float(minute);
#endif
    h.current_halt_start_k = -1.0f;
    h.current_red_start_k = -1.0f;
}

inline void observe_hsl(
    thread HslState& h, float balance, float realized, float upnl,
    bool exposed, int minute, bool terminal
) {
    if (!h.enabled || !h.hsl_valid) return;
    int prior = h.hsl.action;
    const float budget = balance / (h.signal_mode == HSL_SIGNAL_COIN ? h.slot_count : 1.0f)
        * (h.signal_mode == HSL_SIGNAL_COIN ? h.budget_multiplier : 1.0f);
#if PASSIVBOT_HSL_FACTS_ENABLED > 0
    if (h.replay != nullptr && h.replay->enabled) {
        h.hsl_valid = replay_factual_hsl(h, budget, minute, exposed, terminal);
    } else
#endif
    {
#if PASSIVBOT_HSL_FACTUAL_ONLY
    // An enabled native policy requires a bound factual context. Never fall
    // back to the bypassed observation evaluator or absent window buffers.
    h.hsl_valid = false;
#else
    h.hsl_valid = hsl_observe(h.hsl, h.hsl_tree,
        h.hsl_times, h.hsl_realized, minute, h.hsl_lookback,
        budget,
        realized, upnl, exposed, terminal, h.red_threshold,
        h.cooldown_minutes, h.restart_policy == 2);
#endif
    }
    if (!h.hsl_valid) return;
    h.sampled_drawdown_raw = h.hsl.raw;
    h.drawdown_ema = h.hsl.ema;
    h.red_active_now = h.hsl.action == 3;
    if (terminal || (prior == 3 && !h.red_active_now)) finish_hsl_panic_loss(h);
    h.tier = h.red_active_now ? 3 : 0;
    h.halted = h.hsl.action == 1;
    bool reporting_red = prior != 0;
    // Renewed exposure ends the preceding terminal cooldown before the new
    // episode is assessed, even when the new exposure is immediately RED.
    if (prior == 1 && (exposed || terminal)) {
        restart_hsl_report(h, minute);
        reporting_red = false;
    }
    float score = fmin(h.hsl.raw, h.hsl.ema);
    bool terminal_red = terminal && score > h.red_threshold;
    if ((h.hsl.action != 0 || terminal_red) && !reporting_red) {
        begin_hsl_report(h, minute, score);
        reporting_red = true;
    }
    if (terminal_red) {
#if PASSIVBOT_HSL_DIAGNOSTICS_ENABLED
        if (h.current_red_start_k >= 0.0f) {
            h.flatten_time_sum_steps += fmax(float(minute) - h.current_red_start_k, 0.0f);
            h.flatten_time_count += 1.0f;
        }
#endif
        h.current_red_start_k = -1.0f;
    }
    // GREEN permits restart without requiring a flat panic exit. A terminal
    // RED with zero cooldown can trigger, flatten and restart at one timestamp.
    if (h.hsl.action == 0 && reporting_red) restart_hsl_report(h, minute);
#if PASSIVBOT_HSL_DIAGNOSTICS_ENABLED
    h.drawdown_ema_max = fmax(h.drawdown_ema_max, h.drawdown_ema);
#endif
}

// Reporting marks the controller RED while panic or a terminal cooldown blocks
// ordinary trading. Keep this separate from the current panic signal's tier.
inline int hsl_report_tier(thread const HslState& h) {
    return h.enabled && (h.red_active_now || h.halted) ? 3 : 0;
}

inline bool hsl_mark_terminal_observation_eligible(
    float balance, float equity, float liquidation_floor, bool at_fill_boundary
) {
#if PASSIVBOT_HSL_FACTUAL_ONLY
    // Rust evaluates and reports the last bar-close signal on a mark-driven
    // liquidation. A liquidating fill has no fresh bar signal to repeat.
    // This admits observation only; liquidation still prevents further fills.
    return isfinite(balance) && balance > 0.0f && isfinite(equity)
        && equity <= liquidation_floor && !at_fill_boundary;
#else
    return false;
#endif
}

#if PASSIVBOT_HSL_EMA_TAIL_ENABLED || (PASSIVBOT_HSL_FACTUAL_ONLY && PASSIVBOT_HSL_DIAGNOSTICS_ENABLED)
// The public portfolio EMA series observes the maximum current signal on each
// bar. Reducing each side's tail first would discard their joint time ordering.
inline float observed_multicoin_hsl_ema(
    thread const HslState& aggregate, thread const HslState* coins,
    int coin_count, int effective_n_positions
) {
    if (effective_n_positions <= 0) return 0.0f;
    if (aggregate.signal_mode != HSL_SIGNAL_COIN) {
        return aggregate.enabled ? fabs(aggregate.drawdown_ema) : 0.0f;
    }
    float value = 0.0f;
    for (int c = 0; c < coin_count; ++c) {
        if (coins[c].enabled) value = fmax(value, fabs(coins[c].drawdown_ema));
    }
    return value;
}
#endif

// Reporting runs after forced closes and never advances the trading controller.
// A negative tier means this side has no enabled reporting scope.
inline int record_multicoin_hsl_report(
    thread HslState& aggregate, thread HslState* coins, int coin_count,
    int effective_n_positions, bool ema_eligible
#if PASSIVBOT_HSL_FACTUAL_ONLY && PASSIVBOT_HSL_DIAGNOSTICS_ENABLED
    , thread float& report_ema_max
#endif
#if PASSIVBOT_HSL_EMA_TAIL_ENABLED
    , thread HslDrawdownEmaTailStats& ema_tail
#endif
) {
    bool enabled = aggregate.enabled;
    int tier = hsl_report_tier(aggregate);
#if PASSIVBOT_HSL_EMA_TAIL_ENABLED && PASSIVBOT_HSL_FACTUAL_ONLY
    // Reset the current bar before recording its scope observation.
    ema_tail.last_sample = NAN;
#endif
#if PASSIVBOT_HSL_EMA_TAIL_ENABLED || (PASSIVBOT_HSL_FACTUAL_ONLY && PASSIVBOT_HSL_DIAGNOSTICS_ENABLED)
    float drawdown_ema = fabs(aggregate.drawdown_ema);
#if PASSIVBOT_HSL_FACTUAL_ONLY
    // Unified has one portfolio controller, not a controller for each side.
    // The portfolio observation below retains its signal; side reports are zero.
    if (aggregate.signal_mode == HSL_SIGNAL_UNIFIED || !aggregate.enabled
            || effective_n_positions <= 0) drawdown_ema = 0.0f;
#endif
#endif
    if (aggregate.signal_mode == HSL_SIGNAL_COIN) {
        enabled = false;
        tier = 0;
#if PASSIVBOT_HSL_EMA_TAIL_ENABLED || (PASSIVBOT_HSL_FACTUAL_ONLY && PASSIVBOT_HSL_DIAGNOSTICS_ENABLED)
        drawdown_ema = 0.0f;
#endif
        if (effective_n_positions > 0) {
            for (int c = 0; c < coin_count; ++c) {
                enabled = enabled || coins[c].enabled;
                tier = max(tier, hsl_report_tier(coins[c]));
#if PASSIVBOT_HSL_EMA_TAIL_ENABLED || (PASSIVBOT_HSL_FACTUAL_ONLY && PASSIVBOT_HSL_DIAGNOSTICS_ENABLED)
                if (coins[c].enabled)
                    drawdown_ema = fmax(drawdown_ema, fabs(coins[c].drawdown_ema));
#endif
            }
        }
        ema_eligible = enabled;
    }
#if PASSIVBOT_HSL_FACTUAL_ONLY
    // Rust records each bar's enabled scope signal, including cooldown bars.
    // Permission to generate orders is not the EMA reporting clock.
    ema_eligible = true;
#endif
#if PASSIVBOT_HSL_FACTUAL_ONLY && PASSIVBOT_HSL_DIAGNOSTICS_ENABLED
    report_ema_max = fmax(report_ema_max, drawdown_ema);
#endif
    if (ema_eligible) {
#if PASSIVBOT_HSL_EMA_TAIL_ENABLED
        update_hsl_drawdown_ema_tail_stats(ema_tail, drawdown_ema);
#endif
    }
    return enabled ? tier : -1;
}

inline HslState load_hsl(
    constant float* params,
    int po,
    int hsl_param_offset
) {
    HslState h;
#if PASSIVBOT_HSL_FACTS_ENABLED > 0
    h.replay = nullptr; h.replay_side = 0; h.replay_coin = -1;
#if PASSIVBOT_HSL_CUTOFF_CACHE_ENABLED
    hsl_scope_cutoff_cache_reset(h.cutoff_cache);
#if PASSIVBOT_HSL_FACTUAL_ONLY && PASSIVBOT_HSL_INCREMENTAL_ENABLED
    h.scope_cursor.valid = false;
#endif
#endif
#endif
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

// No retained fill proves only the current coin position. Rust reconstructs a
// fresh estimated opening at each observation, not exposure across old marks.
// The adapter supplies the last factual fill from the existing position state;
// the point-tape controller and aggregate reconstructions retain their contract.
inline void update_coin_hsl(
    thread HslState& h, float balance, float realized_pnl, float unrealized_pnl,
    bool has_position, float last_fill_k, int k
) {
    const int minute = k + 1;
#if !PASSIVBOT_HSL_FACTUAL_ONLY
    bool factual = false;
#if PASSIVBOT_HSL_FACTS_ENABLED > 0
    factual = h.replay != nullptr && h.replay->enabled;
#endif
    if (h.enabled && h.hsl_valid && h.signal_mode == HSL_SIGNAL_COIN
        && !factual
        && has_position && isfinite(last_fill_k) && last_fill_k >= 0.0f
        && last_fill_k < float(minute - h.hsl_lookback)) {
        thread HslController& controller = h.hsl;
        controller.window.head = controller.window.count = 0;
        controller.window.last_minute = -1;
        controller.window.block_prefix = hsl_empty_node();
        controller.origin = realized_pnl;
        // Seed the current entry-loss reference, even when UPNL is negative.
        // A one-point curve otherwise has zero drawdown against its own peak.
        controller.episode_seed = minute - h.hsl_lookback - 1;
        controller.completed = false;
        controller.completed_entry_reference = -INFINITY;
        controller.completed_reference_minute = -1;
        controller.scalar_ready = false;
        controller.exposed = true;
    }
#endif
    observe_hsl(h, balance, realized_pnl, unrealized_pnl,
        has_position, minute, false);
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

// Convert the relative reporting clock to controller minute coordinates.
// Normal observations are bar closes; terminal fills retain their earlier time.
inline float hsl_report_end_step(thread const HslTimeObservation& observation) {
    return observation.last_step >= 0.0f ? observation.last_step + 1.0f : -1.0f;
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
    float report_end_k
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
    float terminal_count = (h.red_active_now || h.halted)
        && h.current_halt_start_k >= 0.0f && report_end_k >= 0.0f
        ? 1.0f : 0.0f;
    float terminal_duration = terminal_count > 0.0f
        ? fmax(report_end_k - h.current_halt_start_k, 0.0f) : 0.0f;
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
    float open_exit_count = h.current_red_start_k >= 0.0f && report_end_k >= 0.0f
        ? 1.0f : 0.0f;
    output.flatten_time_sum += h.flatten_time_sum_steps + (open_exit_count > 0.0f
        ? fmax(report_end_k - h.current_red_start_k, 0.0f) : 0.0f);
    output.flatten_time_count += h.flatten_time_count + open_exit_count;
    output.restart_retrigger_count += h.restart_retrigger_count;
    output.halt_to_restart_equity_loss += h.halt_to_restart_equity_loss;
#else
    (void)output;
    (void)h;
    (void)short_side;
    (void)report_end_k;
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
    float report_end_k,
    device float* scalars,
    int scalar_offset
) {
    HslOutputAggregate output = init_hsl_output_aggregate(
        tier_samples_total, tier_samples_red
    );
    accumulate_hsl_output(output, h, short_side, report_end_k);
    write_hsl_output_aggregate(output, scalars, scalar_offset);
}

inline void write_one_side_coin_hsl_outputs(
    thread HslState* controllers,
    int controller_count,
    bool short_side,
    float tier_samples_total,
    float tier_samples_red,
    float report_end_k,
    device float* scalars,
    int scalar_offset
) {
    HslOutputAggregate output = init_hsl_output_aggregate(
        tier_samples_total, tier_samples_red
    );
    for (int c = 0; c < controller_count; ++c) {
        accumulate_hsl_output(
            output, controllers[c], short_side, report_end_k
        );
    }
    write_hsl_output_aggregate(output, scalars, scalar_offset);
}

inline void write_dual_side_hsl_outputs(
    thread HslState& long_hsl,
    thread HslState& short_hsl,
    float tier_samples_total,
    float tier_samples_red,
    float report_end_k,
    device float* scalars,
    int scalar_offset
) {
    HslOutputAggregate output = init_hsl_output_aggregate(
        tier_samples_total, tier_samples_red
    );
    accumulate_hsl_output(output, long_hsl, false, report_end_k);
    accumulate_hsl_output(output, short_hsl, true, report_end_k);
    write_hsl_output_aggregate(output, scalars, scalar_offset);
}

inline void write_dual_side_coin_hsl_outputs(
    thread HslState* long_controllers,
    thread HslState* short_controllers,
    int controller_count,
    float tier_samples_total,
    float tier_samples_red,
    float report_end_k,
    device float* scalars,
    int scalar_offset
) {
    HslOutputAggregate output = init_hsl_output_aggregate(
        tier_samples_total, tier_samples_red
    );
    for (int c = 0; c < controller_count; ++c) {
        accumulate_hsl_output(output, long_controllers[c], false, report_end_k);
        accumulate_hsl_output(output, short_controllers[c], true, report_end_k);
    }
    write_hsl_output_aggregate(output, scalars, scalar_offset);
}
