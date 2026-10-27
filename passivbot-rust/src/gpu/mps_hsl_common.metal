#ifndef PASSIVBOT_HSL_NATIVE_COIN_CACHE_ENABLED
#define PASSIVBOT_HSL_NATIVE_COIN_CACHE_ENABLED 0
#endif

// Shared forward-only GPU HSL adapters and reporting. Strategy kernels supply
// causal position, mark, balance and fill facts to the authored risk graph.
#pragma STDC FP_CONTRACT OFF

#ifndef PASSIVBOT_HSL_CUTOFF_CACHE_ENABLED
#define PASSIVBOT_HSL_CUTOFF_CACHE_ENABLED 1
#endif

#ifndef PASSIVBOT_HSL_INCREMENTAL_ENABLED
#define PASSIVBOT_HSL_INCREMENTAL_ENABLED PASSIVBOT_HSL_FACTUAL_ONLY
#endif

#ifndef PASSIVBOT_HSL_EMPTY_SCOPE_ENABLED
#define PASSIVBOT_HSL_EMPTY_SCOPE_ENABLED PASSIVBOT_HSL_FACTUAL_ONLY
#endif

#ifndef PASSIVBOT_HSL_RAW_TAIL_CAPACITY
#define PASSIVBOT_HSL_RAW_TAIL_CAPACITY 1
#endif

#if 0 && PASSIVBOT_HSL_RAW_TAIL_CAPACITY < 1
#error Raw drawdown tail capacity must be positive
#endif

#define HSL_EMA_TAIL_BINS 32

constant int HSL_SIGNAL_UNIFIED = 0;
constant int HSL_SIGNAL_PSIDE = 1;
constant int HSL_SIGNAL_COIN = 2;

// V9 retains current causal position/mark facts only. History replay belongs
// exclusively to live bootstrap/repair, never to an ordinary GPU observation.
struct HslReplayContext;
struct HslState {
    HslDecayPolicy decay_policy;
    HslDecayScope decay;
    thread HslReplayContext* replay;
    int replay_side;
    int replay_coin;
    float risk_size;
    float risk_basis;
    float risk_mark;
    float risk_multiplier;
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
    float drawdown_threshold_fraction;
    float alpha;
    float cooldown_minutes;
    int signal_mode;
    float slot_count;
    float budget_multiplier;
    float drawdown_ema;
    float sampled_drawdown_raw;
    int tier;
    bool red_active_now;
    bool halted;
    int report_flat_minute; // Observed accounting boundary; never trading permission.

    float triggers;
    float restarts;

    float panic_event_start_equity;
    float panic_event_loss;
    float panic_fill_loss_sum_usd;

    float panic_loss_drawdown_sum;
    float panic_loss_drawdown_max;
    float panic_loss_drawdown_count;
};

struct HslReplayContext {
    thread HslState* coins[2];
    thread HslState* aggregates[2];
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
    context.coins[side] = coins; context.aggregates[side] = &aggregate;
    context.short_side[side] = short_side;
    context.side_count = max(context.side_count, side + 1);
    aggregate.replay = &context; aggregate.replay_side = side; aggregate.replay_coin = -1;
    for (int c = 0; c < context.coin_count; ++c) {
        coins[c].replay = &context; coins[c].replay_side = side; coins[c].replay_coin = c;
        coins[c].risk_multiplier = context.settings[c * context.coin_columns + 4];
    }
}

inline void bind_hsl(
    thread HslState& h, device HslNode* trees, device int* rows,
    int scope, int capacity, int tree_size, int lookback, bool initialize, bool owner,
    int fact_capacity = 0
) {
    h.hsl_owner = owner;
    if (initialize) {
        h.decay = hsl_decay_scope_init();
        h.hsl = hsl_controller_init(nullptr, 0, 0, 1.0f);
    }
}

#ifdef PASSIVBOT_HSL_LOOKBACK
inline void bind_hsl_multicoin_hsl(
    thread HslState& aggregate, thread HslState* coins,
    device HslNode* trees, device int* rows, int scope_base, int coin_count,
    bool initialize, bool owner, int fact_capacity = 0
) {
    bind_hsl(aggregate, trees, rows, scope_base, 0, 0, 0, initialize, owner);
    for (int c = 0; c < coin_count; ++c)
        bind_hsl(coins[c], trees, rows, scope_base + 1 + c, 0, 0, 0, initialize, true);
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

inline float hsl_multicoin_failure_status(
    thread HslState& aggregate, thread HslState* coins, int coin_count
) { return -4.0f; }
#endif

inline void mirror_hsl(thread HslState& owner, thread HslState& view) {
    view.hsl = owner.hsl;
    view.decay = owner.decay;
    view.hsl_valid = owner.hsl_valid;
    view.halted = owner.halted;
    view.red_active_now = owner.red_active_now;
    view.tier = owner.tier;
    view.drawdown_ema = owner.drawdown_ema;
    view.sampled_drawdown_raw = owner.sampled_drawdown_raw;
    view.report_flat_minute = owner.report_flat_minute;
}

// Reporting only: finish one contiguous panic loss segment. GREEN cancels
// panic intent, so a later RED starts a new loss denominator even before flat.
inline void finish_hsl_panic_loss(thread HslState& h) {
    if (h.panic_event_start_equity != -1.0f) {
        float ratio = h.panic_event_loss / h.panic_event_start_equity;

        h.panic_loss_drawdown_sum += ratio;
        h.panic_loss_drawdown_max = fmax(h.panic_loss_drawdown_max, ratio);
        h.panic_loss_drawdown_count = h.panic_loss_drawdown_count < 16777216.0f
            ? h.panic_loss_drawdown_count + 1.0f : NAN;
        h.panic_event_start_equity = -1.0f;
        h.panic_event_loss = 0.0f;
    }
}

// Observational lifecycle: current RED starts reporting before a terminal fill.
// These counters never grant trading permission or retain a panic commitment.
inline void begin_hsl_report(thread HslState& h, int minute, float score) {
    h.triggers = h.triggers < 16777216.0f ? h.triggers + 1.0f : NAN;
}

inline void restart_hsl_report(thread HslState& h, int minute) {
    h.restarts = h.restarts < 16777216.0f ? h.restarts + 1.0f : NAN;
}

inline bool hsl_coin_scope_active(thread HslState& h, int slots, bool exposed) {
    if (slots > 0) return true;
    // A flat zero-slot scope is inactive. Held exposure cannot have no budget.
    if (exposed || slots < 0) h.hsl_valid = false;
    return false;
}

inline void observe_hsl_event(
    thread HslState& h, float balance, float net_realized, float upnl,
    bool exposed, int minute, bool terminal
) {
    if (!h.enabled || !h.hsl_valid) return;
    int prior = h.hsl.action;
    const float budget = balance / (h.signal_mode == HSL_SIGNAL_COIN ? h.slot_count : 1.0f)
        * (h.signal_mode == HSL_SIGNAL_COIN ? h.budget_multiplier : 1.0f);
    h.hsl_valid = hsl_decay_observe(h.decay, h.decay_policy,
        long(minute) * 60000, budget, upnl, net_realized, exposed);
    if (!h.hsl_valid) return;
    h.hsl.raw = h.decay.ratio;
    h.hsl.ema = h.decay.ema;
    h.hsl.action = hsl_decay_action(h.decay, long(minute) * 60000);
    h.hsl.last_observed = minute;
    h.hsl.exposed = exposed;
    if (terminal) h.hsl.flat_minute = minute;
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
    bool terminal_red = terminal && (prior == 3 || h.hsl.action == 1 || h.decay.red);
    if ((h.hsl.action != 0 || terminal_red) && !reporting_red) {
        begin_hsl_report(h, minute, score);
        reporting_red = true;
    }
    // GREEN permits restart without requiring a flat panic exit. A terminal
    // RED with zero cooldown can trigger, flatten and restart at one timestamp.
    if (h.hsl.action == 0 && reporting_red) restart_hsl_report(h, minute);
}

// One symbol mark is shared across its long and short inventories. An execution
// updates this mark before its accounting; no other symbol reads a later close.
inline void hsl_causal_mark(thread HslReplayContext& context, int coin, float price) {
    for (int s = 0; s < context.side_count; ++s)
        context.coins[s][coin].risk_mark = price;
}

inline void hsl_completed_marks(thread HslReplayContext& context, int bar) {
    for (int c = 0; c < context.coin_count; ++c) {
        float mark = context.bars[(bar * context.coin_count + c) * 4 + 2];
        if (isfinite(mark) && mark > 0.0f) hsl_causal_mark(context, c, mark);
    }
}

inline bool hsl_current_scope_facts(
    thread HslReplayContext& context, int selected_side, int selected_coin,
    thread float& upnl, thread bool& exposed
) {
    upnl = 0.0f; exposed = false;
    for (int s = 0; s < context.side_count; ++s) {
        if (selected_side >= 0 && s != selected_side) continue;
        for (int c = 0; c < context.coin_count; ++c) {
            if (selected_coin >= 0 && c != selected_coin) continue;
            thread const HslState& pair = context.coins[s][c];
            if (pair.risk_size == 0.0f) continue;
            if (!isfinite(pair.risk_size) || pair.risk_size < 0.0f
                || !isfinite(pair.risk_basis) || !(pair.risk_basis > 0.0f)
                || !isfinite(pair.risk_mark) || !(pair.risk_mark > 0.0f)
                || !isfinite(pair.risk_multiplier) || !(pair.risk_multiplier > 0.0f))
                return false;
            exposed = true;
            const float difference = context.short_side[s]
                ? pair.risk_basis - pair.risk_mark : pair.risk_mark - pair.risk_basis;
            upnl += pair.risk_size * pair.risk_multiplier * difference;
        }
    }
    return isfinite(upnl);
}

inline void hsl_observe_current_scope(
    thread HslState& controller, thread HslReplayContext& context,
    int selected_side, int selected_coin, float balance, float net, int minute
) {
    if (!controller.enabled || !controller.hsl_valid) return;
    float upnl; bool exposed;
    if (!hsl_current_scope_facts(context, selected_side, selected_coin, upnl, exposed)) {
        controller.hsl_valid = false;
        return;
    }
    const bool terminal = controller.decay.exposed && !exposed;
    observe_hsl_event(controller, balance, net, upnl, exposed, minute, terminal);
}

inline void hsl_observe_fill_context(
    thread HslReplayContext& context, int fill_side, int coin,
    float balance, float net, int minute
) {
    const int mode = context.aggregates[0]->signal_mode;
    if (mode == HSL_SIGNAL_UNIFIED) {
        int owner = context.aggregates[0]->hsl_owner ? 0 : context.side_count - 1;
        hsl_observe_current_scope(*context.aggregates[owner], context, -1, -1,
            balance, net, minute);
        for (int s = 0; s < context.side_count; ++s)
            if (s != owner) mirror_hsl(*context.aggregates[owner], *context.aggregates[s]);
    } else {
        for (int s = 0; s < context.side_count; ++s) {
            if (mode == HSL_SIGNAL_COIN) {
                // Every fill changes the shared balance and therefore every
                // coin's current budget. Only the traded pair consumes its
                // cashflow; other scopes replace R/M without new economics.
                for (int c = 0; c < context.coin_count; ++c)
                    hsl_observe_current_scope(context.coins[s][c], context, s, c,
                        balance, s == fill_side && c == coin ? net : 0.0f, minute);
            } else {
                hsl_observe_current_scope(*context.aggregates[s], context, s, -1,
                    balance, s == fill_side ? net : 0.0f, minute);
            }
        }
    }
}

inline void hsl_before_fill(
    thread HslState& pair, float balance, float fill_price, int minute
) {
    if (pair.replay == nullptr) return;
    thread HslReplayContext& context = *pair.replay;
    hsl_causal_mark(context, pair.replay_coin, fill_price);
    hsl_observe_fill_context(context, pair.replay_side, pair.replay_coin,
        balance, 0.0f, minute);
}

inline void hsl_after_fill(
    thread HslState& pair, float balance, float net, float size_after,
    float basis_after, float fill_price, int minute
) {
    if (pair.replay == nullptr) return;
    pair.risk_size = size_after;
    pair.risk_basis = size_after == 0.0f ? 0.0f : basis_after;
    hsl_observe_fill_context(*pair.replay, pair.replay_side, pair.replay_coin,
        balance, net, minute);
}

// A preceding fill can change permission within the same candle. Match the CPU
// check before each outstanding entry, including each recursive entry rung.
// The entry's own pre-fill mark is consumed only after this admission check.
inline bool hsl_entry_allowed(thread HslState& pair) {
    if (pair.replay == nullptr) return true;
    thread HslReplayContext& context = *pair.replay;
    thread const HslState& controller = context.aggregates[0]->signal_mode == HSL_SIGNAL_COIN
        ? pair : *context.aggregates[pair.replay_side];
    return !controller.enabled || (controller.hsl_valid && controller.hsl.action == 0);
}

inline void observe_hsl(
    thread HslState& h, float balance, float realized, float upnl,
    bool exposed, int minute, bool terminal
) {
    if (h.replay != nullptr) hsl_completed_marks(*h.replay, minute - 1);
    observe_hsl_event(h, balance, 0.0f, upnl, exposed, minute, terminal);
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

// Reporting runs after forced closes and never advances the trading controller.
// A negative tier means this side has no enabled reporting scope.
inline int record_multicoin_hsl_report(
    thread HslState& aggregate, thread HslState* coins, int coin_count,
    int effective_n_positions
) {
    if (aggregate.signal_mode != HSL_SIGNAL_COIN)
        return aggregate.enabled ? hsl_report_tier(aggregate) : -1;
    if (effective_n_positions <= 0) return -1;
    bool enabled = false;
    int tier = 0;
    for (int c = 0; c < coin_count; ++c) {
        enabled = enabled || coins[c].enabled;
        tier = max(tier, hsl_report_tier(coins[c]));
    }
    return enabled ? tier : -1;
}

inline HslState load_hsl(
    constant float* params, int po, int hsl_param_offset
) {
    HslState h;
    int ho = po + hsl_param_offset;
    h.enabled = params[ho + 0] > 0.5f;
    h.drawdown_threshold_fraction = params[ho + 1];
    h.cooldown_minutes = 0.0f;
    h.signal_mode = int(round(params[ho + 5]));
    h.slot_count = 1.0f;
    h.budget_multiplier = 1.0f;
    long duration_ms = 0;
    h.hsl_valid = hsl_decay_duration(params[ho + 3], params[ho + 6], duration_ms)
        && hsl_decay_policy(params[ho + 4], params[ho + 2],
            h.drawdown_threshold_fraction, duration_ms, h.decay_policy);
    h.alpha = h.decay_policy.fast_alpha;
    h.decay = hsl_decay_scope_init();
    h.replay = nullptr; h.replay_side = 0; h.replay_coin = -1;
    h.risk_size = h.risk_basis = h.risk_mark = 0.0f;
    h.risk_multiplier = 1.0f;
    h.drawdown_ema = 0.0f;
    h.sampled_drawdown_raw = 0.0f;
    h.tier = 0;
    h.red_active_now = false;
    h.halted = false;
    h.report_flat_minute = -1;

    h.triggers = 0.0f;
    h.restarts = 0.0f;

    h.panic_event_start_equity = -1.0f;
    h.panic_event_loss = 0.0f;
    h.panic_fill_loss_sum_usd = 0.0f;

    h.panic_loss_drawdown_sum = 0.0f;
    h.panic_loss_drawdown_max = 0.0f;
    h.panic_loss_drawdown_count = 0.0f;
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
    if (isfinite(value)) h.drawdown_threshold_fraction = h.decay_policy.drawdown_threshold_fraction = value;
    value = coin_overrides[offset + 2];
    if (isfinite(value)) {
        h.decay_policy.fast_alpha = 2.0f / (value + 1.0f);
        h.decay_policy.fast_beta = 1.0f - h.decay_policy.fast_alpha;
        h.alpha = h.decay_policy.fast_alpha;
        h.hsl_valid = h.hsl_valid && isfinite(value) && value >= 1.0f
            && h.decay_policy.fast_beta >= 0.0f && h.decay_policy.fast_beta < 1.0f;
    }
    value = coin_overrides[offset + 6];
    if (isfinite(value))
        h.hsl_valid = h.hsl_valid
            && hsl_decay_rate(value, h.decay_policy.slow_alpha, h.decay_policy.slow_beta);
    float low = coin_overrides[offset + 3], high = coin_overrides[offset + 4];
    if (isfinite(low) || isfinite(high))
        h.hsl_valid = h.hsl_valid && hsl_decay_duration(low, high, h.decay_policy.cooldown_ms);
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

inline void update_coin_hsl(
    thread HslState& h, float balance, float realized_pnl, float unrealized_pnl,
    bool has_position, float last_fill_k, int k
) {
    observe_hsl(h, balance, realized_pnl, unrealized_pnl, has_position, k + 1, false);
}

// A real closing fill samples its final fee-inclusive drawdown before an ordinary
// episode reset. Proven-flat RED boundaries finalize before later fills can reopen.
inline bool finish_hsl_episode_at_flat(
    thread HslState& h, float balance, float starting_balance,
    float realized_pnl, float kf, float interval_ms
) {
    if (h.enabled && h.hsl_valid) h.report_flat_minute = int(kf);
    return h.hsl_valid;
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
    float pre_fill_balance
) {
    // Forced delist labels alone do not establish an HSL execution episode.
    if (!h.enabled || !h.red_active_now) return;
    if (h.panic_event_start_equity < 0.0f) {
        h.panic_event_start_equity = isfinite(pre_fill_balance) && pre_fill_balance > 0.0f
            ? pre_fill_balance : NAN;
    }
    float panic_loss = fmax(-net_pnl, 0.0f);
    h.panic_event_loss += panic_loss;
    h.panic_fill_loss_sum_usd += panic_loss;

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

// Each genuine fill can change current permission, including an entry fee that
// turns GREEN into RED. Capture the final same-timestamp state before the next
// completed close integrates its interval. This never advances the risk graph.
struct HslFillTimeReport {
    int tier;
    bool observed;
};

inline HslFillTimeReport observed_multicoin_hsl_fill_time(
    thread const HslState& aggregate, thread const HslState* coins,
    int coin_count, int effective_n_positions, int minute
) {
    HslFillTimeReport result;
    result.tier = -1; result.observed = false;
    if (aggregate.signal_mode == HSL_SIGNAL_COIN) {
        if (effective_n_positions > 0) {
            for (int c = 0; c < coin_count; ++c) {
                if (!coins[c].enabled) continue;
                result.tier = max(result.tier, hsl_report_tier(coins[c]));
                result.observed = result.observed || coins[c].hsl.last_observed == minute;
            }
        }
    } else if (aggregate.enabled) {
        result.tier = hsl_report_tier(aggregate);
        result.observed = aggregate.hsl.last_observed == minute;
    }
    return result;
}

// Convert the relative reporting clock to controller minute coordinates.
// Normal observations are bar closes; terminal fills retain their earlier time.
inline float hsl_report_end_step(thread const HslTimeObservation& observation) {
    return observation.last_step >= 0.0f ? observation.last_step + 1.0f : -1.0f;
}

// Keep every HSL scalar reduction in one contract. Existing one-side kernels
// and future fused dual-side kernels therefore share identical sum/max/count
// and conditional-min semantics.
// Compact factual reporting shared by all controller topologies. Reserved
// scalar holes do not carry retired analytic quantities.
struct HslOutputAggregate {
    float enabled_long, enabled_short;
    float triggers_long, triggers_short;
    float restarts_long, restarts_short;
    float tier_samples_total, tier_samples_red;
    float panic_fill_loss_sum_usd;
    float panic_loss_drawdown_sum, panic_loss_drawdown_max, panic_loss_drawdown_count;
};

inline HslOutputAggregate init_hsl_output_aggregate(float observed_steps, float red_steps) {
    HslOutputAggregate output;
    output.enabled_long = output.enabled_short = 0.0f;
    output.triggers_long = output.triggers_short = 0.0f;
    output.restarts_long = output.restarts_short = 0.0f;
    output.tier_samples_total = observed_steps;
    output.tier_samples_red = red_steps;
    output.panic_fill_loss_sum_usd = 0.0f;
    output.panic_loss_drawdown_sum = output.panic_loss_drawdown_max = 0.0f;
    output.panic_loss_drawdown_count = 0.0f;
    return output;
}

inline void accumulate_hsl_output(thread HslOutputAggregate& output,
    thread HslState& h, bool short_side, float report_end_k) {
    HslState report = h;
    finish_hsl_panic_loss(report);
    output.panic_fill_loss_sum_usd += h.panic_fill_loss_sum_usd;
    output.panic_loss_drawdown_sum += report.panic_loss_drawdown_sum;
    output.panic_loss_drawdown_max = fmax(output.panic_loss_drawdown_max,
                                         report.panic_loss_drawdown_max);
    output.panic_loss_drawdown_count = output.panic_loss_drawdown_count <= 16777216.0f - report.panic_loss_drawdown_count
        ? output.panic_loss_drawdown_count + report.panic_loss_drawdown_count : NAN;
    if (!h.enabled) return;
    if (short_side) {
        output.enabled_short = 1.0f;
        output.triggers_short = output.triggers_short <= 16777216.0f - h.triggers
            ? output.triggers_short + h.triggers : NAN;
        output.restarts_short += h.restarts;
    } else {
        output.enabled_long = 1.0f;
        output.triggers_long = output.triggers_long <= 16777216.0f - h.triggers
            ? output.triggers_long + h.triggers : NAN;
        output.restarts_long += h.restarts;
    }
}

inline void write_hsl_output_aggregate(thread const HslOutputAggregate& output,
    device float* scalars, int scalar_offset) {
    scalars[scalar_offset + 0] = output.enabled_long;
    scalars[scalar_offset + 1] = output.enabled_short;
    scalars[scalar_offset + 2] = output.triggers_long;
    scalars[scalar_offset + 3] = output.triggers_short;
    scalars[scalar_offset + 4] = output.restarts_long;
    scalars[scalar_offset + 5] = output.restarts_short;
    scalars[scalar_offset + 6] = output.tier_samples_total;
    scalars[scalar_offset + 7] = output.tier_samples_red;
    scalars[scalar_offset + 17] = output.panic_fill_loss_sum_usd;
    scalars[scalar_offset + 20] = output.panic_loss_drawdown_sum;
    scalars[scalar_offset + 21] = output.panic_loss_drawdown_max;
    scalars[scalar_offset + 22] = output.panic_loss_drawdown_count;
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

#pragma STDC FP_CONTRACT ON
