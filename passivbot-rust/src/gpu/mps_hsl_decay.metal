// V9 risk arithmetic. CPU and device callers consume ordered economic events;
// no operation in this module reconstructs or scans historical observations.
// Keep this arithmetic fixed even when inlined into ordinary strategy callers.
// Restore their prior contraction policy after the authored risk definitions.
#pragma STDC FP_CONTRACT OFF

struct HslDecayPolicy {
    float slow_alpha;
    float slow_beta;
    float fast_alpha;
    float fast_beta;
    float threshold;
    long cooldown_ms;
};

inline bool hsl_decay_duration(float low, float high, thread long& duration_ms) {
    if (!isfinite(low) || !isfinite(high) || low < 0.0f || high < 0.0f
        || low >= 16777216.0f || high >= 16777216.0f
        || low != floor(low) || high != floor(high)) return false;
    duration_ms = long(low) + (long(high) << 24);
    return true;
}

inline bool hsl_decay_policy(
    float slow_span, float fast_span, float threshold, long cooldown_ms,
    thread HslDecayPolicy& policy
) {
    if (!isfinite(slow_span) || slow_span < 1.0f
        || !isfinite(fast_span) || fast_span < 1.0f
        || !isfinite(threshold) || !(threshold >= 0.0f && threshold <= 1.0f)
        || cooldown_ms < 0) return false;
    policy.slow_alpha = 2.0f / (slow_span + 1.0f);
    policy.slow_beta = 1.0f - policy.slow_alpha;
    policy.fast_alpha = 2.0f / (fast_span + 1.0f);
    policy.fast_beta = 1.0f - policy.fast_alpha;
    policy.threshold = threshold;
    policy.cooldown_ms = cooldown_ms;
    // A huge finite span can round beta to one in f32. Reject a memory that
    // would silently stop decaying on one supported execution path.
    return isfinite(policy.slow_alpha) && policy.slow_alpha > 0.0f
        && policy.slow_beta >= 0.0f && policy.slow_beta < 1.0f
        && isfinite(policy.fast_alpha) && policy.fast_alpha > 0.0f
        && policy.fast_beta >= 0.0f && policy.fast_beta < 1.0f;
}

struct HslDecaySample {
    float drawdown;
    float ratio;
    float ema;
    bool red;
};

inline bool hsl_decay_sample(
    thread const HslDecayPolicy& policy, float drawdown, float ema_before_minute,
    float budget, float upnl, thread HslDecaySample& sample
) {
    if (!isfinite(drawdown) || drawdown < 0.0f || !isfinite(ema_before_minute)
        || ema_before_minute < 0.0f || !isfinite(budget)
        || !isfinite(upnl)) return false;
    const float equity = budget + upnl;
    const float denominator = equity + drawdown;
    if (!isfinite(equity) || !isfinite(denominator)) return false;
    sample.drawdown = drawdown;
    const bool nonpositive = equity <= 0.0f || denominator <= 0.0f;
    sample.ratio = nonpositive ? 1.0f : drawdown / denominator;
    sample.ema = fma(policy.fast_beta, ema_before_minute,
                     policy.fast_alpha * sample.ratio);
    if (!isfinite(sample.ratio) || !isfinite(sample.ema)) return false;
    sample.red = nonpositive || fmin(sample.ratio, sample.ema) > policy.threshold;
    return true;
}

inline bool hsl_decay_economic_event(
    thread const HslDecayPolicy& policy, float previous_drawdown,
    float previous_upnl, float current_upnl, float net_realized_pnl,
    bool first_event_in_minute, thread float& drawdown
) {
    if (!isfinite(previous_drawdown) || previous_drawdown < 0.0f
        || !isfinite(previous_upnl) || !isfinite(current_upnl)
        || !isfinite(net_realized_pnl)) return false;
    const float upnl_loss = previous_upnl - current_upnl;
    const float loss = upnl_loss - net_realized_pnl;
    if (!isfinite(upnl_loss) || !isfinite(loss)) return false;
    drawdown = fmax(0.0f, first_event_in_minute
        ? fma(policy.slow_beta, previous_drawdown, loss)
        : previous_drawdown + loss);
    return isfinite(drawdown);
}

inline bool hsl_decay_realized_minute(
    thread const HslDecayPolicy& policy, thread float& drawdown
) {
    if (!isfinite(drawdown) || drawdown < 0.0f) return false;
    drawdown = fma(policy.slow_beta, drawdown, 0.0f);
    return isfinite(drawdown);
}

inline bool hsl_decay_realized_fill(float net_realized_pnl, thread float& drawdown) {
    if (!isfinite(drawdown) || drawdown < 0.0f || !isfinite(net_realized_pnl)) return false;
    const float next = drawdown - net_realized_pnl;
    if (!isfinite(next)) return false;
    drawdown = fmax(0.0f, next);
    return true;
}

inline bool hsl_decay_realized_allowance(
    float balance, float effective_allowance_pct, float drawdown, thread float& allowance
) {
    if (!isfinite(balance) || balance <= 0.0f || !isfinite(effective_allowance_pct)
        || effective_allowance_pct < 0.0f || !isfinite(drawdown) || drawdown < 0.0f)
        return false;
    const float peak = balance + drawdown;
    if (!isfinite(peak)) return false;
    const float result = fma(effective_allowance_pct, peak, -drawdown);
    if (!isfinite(result)) return false;
    allowance = fmax(0.0f, result);
    return true;
}

inline bool hsl_decay_auto_allowance(
    float balance, float drawdown, float loss_allowance_pct, float total_exposure,
    thread float& allowance
) {
    if (!isfinite(loss_allowance_pct) || loss_allowance_pct < 0.0f
        || !isfinite(total_exposure) || total_exposure < 0.0f) return false;
    const float effective_pct = loss_allowance_pct * total_exposure;
    if (!isfinite(effective_pct)) return false;
    return hsl_decay_realized_allowance(balance, effective_pct, drawdown, allowance);
}

struct HslDecayScope {
    float drawdown;
    float ratio;
    float ema;
    float ema_before_minute;
    float previous_upnl;
    float previous_budget;
    long minute;
    long last_timestamp_ms;
    long cooldown_until_ms;
    bool exposed;
    bool red;
};

inline HslDecayScope hsl_decay_scope_init() {
    HslDecayScope scope;
    scope.drawdown = scope.ratio = scope.ema = scope.ema_before_minute = 0.0f;
    scope.previous_upnl = scope.previous_budget = 0.0f;
    scope.minute = scope.last_timestamp_ms = scope.cooldown_until_ms = -1;
    scope.exposed = scope.red = false;
    return scope;
}

inline bool hsl_decay_commit_sample(
    thread HslDecayScope& scope, thread const HslDecayPolicy& policy,
    float budget, float upnl
) {
    HslDecaySample sample;
    if (!hsl_decay_sample(policy, scope.drawdown, scope.ema_before_minute,
                          budget, upnl, sample)) return false;
    scope.ratio = sample.ratio;
    scope.ema = sample.ema;
    scope.red = sample.red;
    return true;
}

inline int hsl_decay_action(thread const HslDecayScope& scope, long now_ms) {
    return scope.exposed ? (scope.red ? 3 : 0)
        : (now_ms < scope.cooldown_until_ms ? 1 : 0);
}

// Inputs are already scoped economic deltas, not a running account balance.
// A query uses hsl_decay_action and cannot consume this function again.
inline bool hsl_decay_observe_inner(
    thread HslDecayScope& scope, thread const HslDecayPolicy& policy,
    long timestamp_ms, float budget, float upnl, float net_realized_pnl, bool exposed
) {
    if (timestamp_ms < 0 || timestamp_ms < scope.last_timestamp_ms
        || !isfinite(budget) || !isfinite(upnl) || !isfinite(net_realized_pnl)
        || (!exposed && upnl != 0.0f)) return false;
    const long minute = timestamp_ms / 60000;
    const bool was_exposed = scope.exposed;
    const bool pre_final_red = was_exposed && scope.red;
    const bool opening = exposed && !was_exposed;
    if (opening) scope = hsl_decay_scope_init();
    if (!exposed && !scope.exposed) {
        if (net_realized_pnl != 0.0f) return false;
        // A completed deadline is an event result. New marks or budgets while
        // flat cannot reclassify it or decay the finished episode into GREEN.
        scope.last_timestamp_ms = timestamp_ms;
        return true;
    }
    // Missing minutes carry the last supported facts; callers retain separate
    // readiness authority and may not present stale marks as fresh live marks.
    if (scope.exposed) {
        for (long skipped = scope.minute + 1; skipped < minute; ++skipped) {
            scope.ema_before_minute = scope.ema;
            scope.drawdown = fmax(0.0f, fma(policy.slow_beta, scope.drawdown, 0.0f));
            if (!hsl_decay_commit_sample(scope, policy, scope.previous_budget,
                                         scope.previous_upnl)) return false;
            scope.minute = skipped;
        }
    }
    const bool first = minute != scope.minute;
    if (first) scope.ema_before_minute = scope.ema;
    if (!hsl_decay_economic_event(policy, scope.drawdown, scope.previous_upnl,
                                   upnl, net_realized_pnl, first, scope.drawdown)
        || !hsl_decay_commit_sample(scope, policy, budget, upnl)) return false;
    if (scope.exposed && !exposed) {
        const bool qualified = pre_final_red || scope.red;
        // Keep integer deadline addition defined on every admitted backend.
        if (qualified && policy.cooldown_ms > 0x7fffffffffffffffl - timestamp_ms)
            return false;
        scope.cooldown_until_ms = qualified ? timestamp_ms + policy.cooldown_ms : -1;
    }
    scope.previous_upnl = upnl;
    scope.previous_budget = budget;
    scope.minute = minute;
    scope.last_timestamp_ms = timestamp_ms;
    scope.exposed = exposed;
    return true;
}

inline bool hsl_decay_observe(
    thread HslDecayScope& scope, thread const HslDecayPolicy& policy,
    long timestamp_ms, float budget, float upnl, float net_realized_pnl, bool exposed
) {
    HslDecayScope next = scope;
    if (!hsl_decay_observe_inner(next, policy, timestamp_ms, budget, upnl,
                                 net_realized_pnl, exposed)) return false;
    scope = next;
    return true;
}

#pragma STDC FP_CONTRACT ON

inline bool hsl_decay_seed_unknown(
    thread HslDecayScope& scope, thread const HslDecayPolicy& policy,
    long timestamp_ms, float budget, float upnl
) {
    HslDecayScope next = hsl_decay_scope_init();
    if (!hsl_decay_observe(next, policy, timestamp_ms, budget, upnl, 0.0f, true))
        return false;
    // Unknown-prefix approximation: start the fast signal at the observed
    // ratio instead of hiding current loss behind a synthetic warmup.
    next.ema = next.ema_before_minute = next.ratio;
    const float equity = budget + upnl;
    next.red = equity <= 0.0f || equity + next.drawdown <= 0.0f
        || next.ratio > policy.threshold;
    scope = next;
    return true;
}
