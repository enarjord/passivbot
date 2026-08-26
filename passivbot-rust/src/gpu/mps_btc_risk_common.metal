#if PASSIVBOT_BTC_RISK_ENABLED

struct BtcRiskState {
    float peak;
    float day_end;
    float day_min;
    float day_max_drawdown;
};

inline BtcRiskState init_btc_risk_state() {
    BtcRiskState state;
    state.peak = -INFINITY;
    state.day_end = 0.0f;
    state.day_min = INFINITY;
    state.day_max_drawdown = 0.0f;
    return state;
}

inline void reset_btc_risk_day(thread BtcRiskState& state) {
    state.day_end = 0.0f;
    state.day_min = INFINITY;
    state.day_max_drawdown = 0.0f;
}

inline void update_btc_risk_state(
    thread BtcRiskState& state,
    float usd_equity,
    float btc_price
) {
    float btc_equity = usd_equity / btc_price;
    state.peak = fmax(state.peak, btc_equity);
    float drawdown = fmax(
        (state.peak - btc_equity) / fmax(fabs(state.peak), 1.0e-12f),
        0.0f
    );
    state.day_end = btc_equity;
    state.day_min = fmin(state.day_min, btc_equity);
    state.day_max_drawdown = fmax(state.day_max_drawdown, drawdown);
}

inline void write_btc_risk_day(
    thread const BtcRiskState& state,
    device float* daily,
    int output,
    int first_column
) {
    daily[output + first_column + 0] = state.day_end;
    daily[output + first_column + 1] = state.day_min;
    daily[output + first_column + 2] = state.day_max_drawdown;
}

#endif
