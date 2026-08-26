// Shared opt-in account equity-vs-balance accumulator.
//
// USD and BTC analysis use separate balance histories. USD balance changes on
// fills, while BTC balance is the post-fill USD balance converted at that
// fill's BTC/USD price. Keep both compact accumulators in one opt-in surface.

#ifndef PASSIVBOT_EQUITY_BALANCE_DIFF_ENABLED
#define PASSIVBOT_EQUITY_BALANCE_DIFF_ENABLED 0
#endif

#if PASSIVBOT_BTC_RISK_ENABLED || PASSIVBOT_EQUITY_BALANCE_DIFF_ENABLED
#define PASSIVBOT_BTC_PRICES_ENABLED 1
#else
#define PASSIVBOT_BTC_PRICES_ENABLED 0
#endif

#if PASSIVBOT_EQUITY_BALANCE_DIFF_ENABLED
struct EquityBalanceDiffState {
    float usd_positive_max;
    float usd_positive_sum;
    float usd_positive_count;
    float usd_negative_max;
    float usd_negative_sum;
    float usd_negative_count;
    float btc_positive_max;
    float btc_positive_sum;
    float btc_positive_count;
    float btc_negative_max;
    float btc_negative_sum;
    float btc_negative_count;
    float btc_balance;
    int first_sample_k;
    bool btc_balance_initialized;
    bool valid;
};

inline EquityBalanceDiffState init_equity_balance_diff_state() {
    EquityBalanceDiffState state;
    state.usd_positive_max = 0.0f;
    state.usd_positive_sum = 0.0f;
    state.usd_positive_count = 0.0f;
    state.usd_negative_max = 0.0f;
    state.usd_negative_sum = 0.0f;
    state.usd_negative_count = 0.0f;
    state.btc_positive_max = 0.0f;
    state.btc_positive_sum = 0.0f;
    state.btc_positive_count = 0.0f;
    state.btc_negative_max = 0.0f;
    state.btc_negative_sum = 0.0f;
    state.btc_negative_count = 0.0f;
    state.btc_balance = 0.0f;
    state.first_sample_k = -1;
    state.btc_balance_initialized = false;
    state.valid = true;
    return state;
}

inline bool update_equity_balance_diff_accumulator(
    thread float& positive_max,
    thread float& positive_sum,
    thread float& positive_count,
    thread float& negative_max,
    thread float& negative_sum,
    thread float& negative_count,
    float balance,
    float equity
) {
    const float diff = (equity - balance) / balance;
    if (!isfinite(diff)) {
        return false;
    }
    if (diff > 0.0f) {
        positive_max = fmax(positive_max, diff);
        positive_sum += diff;
        positive_count += 1.0f;
    } else if (diff < 0.0f) {
        const float loss = fabs(diff);
        negative_max = fmax(negative_max, loss);
        negative_sum += loss;
        negative_count += 1.0f;
    }
    return true;
}

inline void update_equity_balance_diff_state(
    thread EquityBalanceDiffState& state,
    float balance,
    float equity,
    constant float* btc_prices,
    int sample_k,
    float starting_balance,
    bool any_fill
) {
    const float btc_price = btc_prices[sample_k];
    if (balance == 0.0f || !isfinite(balance) || !isfinite(equity)
        || !(btc_price > 0.0f) || !isfinite(btc_price)
        || !(starting_balance > 0.0f) || !isfinite(starting_balance)) {
        state.valid = false;
        return;
    }
    if (state.first_sample_k < 0) {
        state.first_sample_k = sample_k;
    }
    if (!state.btc_balance_initialized) {
        if (!any_fill) {
            return;
        }
        state.btc_balance = balance / btc_price;
        state.btc_balance_initialized = true;
        // Canonical Rust seeds analysis balance from the first fill and
        // applies that baseline to every earlier tracked equity sample. No
        // position exists before the first fill, so replaying BTC prices with
        // starting USD equity reproduces those samples without a second
        // strategy simulation or a per-candidate history buffer.
        for (int k = state.first_sample_k; k < sample_k; ++k) {
            const float prefill_btc_price = btc_prices[k];
            if (!(prefill_btc_price > 0.0f)
                || !isfinite(prefill_btc_price)) {
                state.valid = false;
                return;
            }
            state.valid = state.valid
                && update_equity_balance_diff_accumulator(
                    state.usd_positive_max,
                    state.usd_positive_sum,
                    state.usd_positive_count,
                    state.usd_negative_max,
                    state.usd_negative_sum,
                    state.usd_negative_count,
                    balance,
                    starting_balance
                );
            state.valid = state.valid
                && update_equity_balance_diff_accumulator(
                    state.btc_positive_max,
                    state.btc_positive_sum,
                    state.btc_positive_count,
                    state.btc_negative_max,
                    state.btc_negative_sum,
                    state.btc_negative_count,
                    state.btc_balance,
                    starting_balance / prefill_btc_price
                );
        }
    } else if (any_fill) {
        state.btc_balance = balance / btc_price;
    }
    state.valid = state.valid && update_equity_balance_diff_accumulator(
        state.usd_positive_max,
        state.usd_positive_sum,
        state.usd_positive_count,
        state.usd_negative_max,
        state.usd_negative_sum,
        state.usd_negative_count,
        balance,
        equity
    );
    const float btc_equity = equity / btc_price;
    if (state.btc_balance == 0.0f || !isfinite(state.btc_balance)
        || !isfinite(btc_equity)) {
        state.valid = false;
        return;
    }
    state.valid = state.valid && update_equity_balance_diff_accumulator(
        state.btc_positive_max,
        state.btc_positive_sum,
        state.btc_positive_count,
        state.btc_negative_max,
        state.btc_negative_sum,
        state.btc_negative_count,
        state.btc_balance,
        btc_equity
    );
}

inline void write_equity_balance_diff_state(
    thread const EquityBalanceDiffState& state,
    device float* output,
    uint candidate
) {
    const int offset = int(candidate) * 12;
    if (!state.valid) {
        for (int column = 0; column < 12; ++column) {
            output[offset + column] = NAN;
        }
        return;
    }
    output[offset + 0] = state.usd_positive_max;
    output[offset + 1] = state.usd_positive_sum;
    output[offset + 2] = state.usd_positive_count;
    output[offset + 3] = state.usd_negative_max;
    output[offset + 4] = state.usd_negative_sum;
    output[offset + 5] = state.usd_negative_count;
    output[offset + 6] = state.btc_positive_max;
    output[offset + 7] = state.btc_positive_sum;
    output[offset + 8] = state.btc_positive_count;
    output[offset + 9] = state.btc_negative_max;
    output[offset + 10] = state.btc_negative_sum;
    output[offset + 11] = state.btc_negative_count;
}
#endif
