#ifndef PASSIVBOT_HIGH_EXPOSURE_ENABLED
#define PASSIVBOT_HIGH_EXPOSURE_ENABLED 0
#endif

constant int HIGH_EXPOSURE_COLS = 8;

#if PASSIVBOT_HIGH_EXPOSURE_ENABLED

struct HighExposureState {
    bool metric_pass;
    int first_fill_day;
    int last_fill_day;
    int current_fill_day;
    float current_day_twe_sum_long;
    float current_day_twe_sum_short;
    float current_day_fill_count;
    float daily_twe_mean_sum_long;
    float daily_twe_mean_sum_short;
    float observed_fill_count;
    float threshold_long;
    float threshold_short;
    float run_start_k_long;
    float run_start_k_short;
    float duration_sum_steps_long;
    float duration_sum_steps_short;
    float duration_max_steps_long;
    float duration_max_steps_short;
    float duration_count_long;
    float duration_count_short;
};

inline HighExposureState init_high_exposure_state(
    device float* output,
    int candidate
) {
    const int offset = candidate * HIGH_EXPOSURE_COLS;
    HighExposureState state;
    state.threshold_long = output[offset + 0];
    state.threshold_short = output[offset + 1];
    state.metric_pass = isfinite(state.threshold_long)
        && isfinite(state.threshold_short);
    state.first_fill_day = -1;
    state.last_fill_day = -1;
    state.current_fill_day = -1;
    state.current_day_twe_sum_long = 0.0f;
    state.current_day_twe_sum_short = 0.0f;
    state.current_day_fill_count = 0.0f;
    state.daily_twe_mean_sum_long = 0.0f;
    state.daily_twe_mean_sum_short = 0.0f;
    state.observed_fill_count = 0.0f;
    state.run_start_k_long = -1.0f;
    state.run_start_k_short = -1.0f;
    state.duration_sum_steps_long = 0.0f;
    state.duration_sum_steps_short = 0.0f;
    state.duration_max_steps_long = 0.0f;
    state.duration_max_steps_short = 0.0f;
    state.duration_count_long = 0.0f;
    state.duration_count_short = 0.0f;
    return state;
}

inline void flush_high_exposure_threshold_day(
    thread HighExposureState& state
) {
    if (state.current_fill_day < 0 || !(state.current_day_fill_count > 0.0f)) {
        return;
    }
    state.daily_twe_mean_sum_long += state.current_day_twe_sum_long
        / state.current_day_fill_count;
    state.daily_twe_mean_sum_short += state.current_day_twe_sum_short
        / state.current_day_fill_count;
    state.current_day_twe_sum_long = 0.0f;
    state.current_day_twe_sum_short = 0.0f;
    state.current_day_fill_count = 0.0f;
}

inline void update_high_exposure_duration(
    float twe,
    float threshold,
    float fill_k,
    thread float& run_start_k,
    thread float& duration_sum_steps,
    thread float& duration_max_steps,
    thread float& duration_count
) {
    if (twe > threshold) {
        if (run_start_k < 0.0f) run_start_k = fill_k;
        return;
    }
    if (run_start_k >= 0.0f) {
        const float duration = fmax(fill_k - run_start_k, 0.0f);
        duration_sum_steps += duration;
        duration_max_steps = fmax(duration_max_steps, duration);
        duration_count += 1.0f;
        run_start_k = -1.0f;
    }
}

inline void record_high_exposure_fill(
    thread HighExposureState& state,
    float fill_k,
    int fill_day,
    float cumulative_fill_count,
    float twe_long,
    float twe_short
) {
    if (state.metric_pass) {
        update_high_exposure_duration(
            twe_long, state.threshold_long, fill_k,
            state.run_start_k_long, state.duration_sum_steps_long,
            state.duration_max_steps_long, state.duration_count_long
        );
        update_high_exposure_duration(
            twe_short, state.threshold_short, fill_k,
            state.run_start_k_short, state.duration_sum_steps_short,
            state.duration_max_steps_short, state.duration_count_short
        );
        return;
    }

    if (state.current_fill_day != fill_day) {
        flush_high_exposure_threshold_day(state);
        state.current_fill_day = fill_day;
        if (state.first_fill_day < 0) state.first_fill_day = fill_day;
        state.last_fill_day = fill_day;
    }
    const float sample_count = fmax(
        cumulative_fill_count - state.observed_fill_count, 1.0f
    );
    state.observed_fill_count = cumulative_fill_count;
    state.current_day_twe_sum_long += fmax(twe_long, 0.0f) * sample_count;
    state.current_day_twe_sum_short += fmax(twe_short, 0.0f) * sample_count;
    state.current_day_fill_count += sample_count;
}

inline void close_high_exposure_tail(
    float last_fill_k,
    thread float& run_start_k,
    thread float& duration_sum_steps,
    thread float& duration_max_steps,
    thread float& duration_count
) {
    if (run_start_k < 0.0f || last_fill_k < 0.0f) return;
    const float duration = fmax(last_fill_k - run_start_k, 0.0f);
    duration_sum_steps += duration;
    duration_max_steps = fmax(duration_max_steps, duration);
    duration_count += 1.0f;
    run_start_k = -1.0f;
}

inline void write_high_exposure_output(
    thread HighExposureState& state,
    float last_fill_k,
    device float* output,
    int candidate
) {
    const int offset = candidate * HIGH_EXPOSURE_COLS;
    if (!state.metric_pass) {
        flush_high_exposure_threshold_day(state);
        const int total_days = state.first_fill_day >= 0
            ? state.last_fill_day - state.first_fill_day + 1
            : 0;
        output[offset + 0] = total_days > 0
            ? state.daily_twe_mean_sum_long / float(total_days)
            : 0.0f;
        output[offset + 1] = total_days > 0
            ? state.daily_twe_mean_sum_short / float(total_days)
            : 0.0f;
        for (int column = 2; column < HIGH_EXPOSURE_COLS; column++) {
            output[offset + column] = 0.0f;
        }
        return;
    }

    close_high_exposure_tail(
        last_fill_k, state.run_start_k_long,
        state.duration_sum_steps_long, state.duration_max_steps_long,
        state.duration_count_long
    );
    close_high_exposure_tail(
        last_fill_k, state.run_start_k_short,
        state.duration_sum_steps_short, state.duration_max_steps_short,
        state.duration_count_short
    );
    output[offset + 2] = state.duration_sum_steps_long;
    output[offset + 3] = state.duration_max_steps_long;
    output[offset + 4] = state.duration_count_long;
    output[offset + 5] = state.duration_sum_steps_short;
    output[offset + 6] = state.duration_max_steps_short;
    output[offset + 7] = state.duration_count_short;
}

#endif
