// Shared CPU-contract entry timing and finite exponentially weighted directionality.
// The departing return is read from immutable candles; no span-sized GPU ring is needed.
struct AdaptiveTiming {
    float minimum, maximum, exposure_weight, adverse_weight, span, score_weight;
    int window;
    float mean, square, score;
    int flat_returns, last_k;
};

inline AdaptiveTiming load_adaptive_timing(constant float* params, int offset) {
    AdaptiveTiming a;
    a.minimum = params[offset];
    a.maximum = params[offset + 1];
    a.exposure_weight = params[offset + 2];
    a.adverse_weight = params[offset + 3];
    a.span = params[offset + 4];
    a.score_weight = params[offset + 5];
    a.window = int(params[offset + 6]);
    a.mean = 0.0f;
    a.square = 0.0f;
    a.score = NAN; // Insufficient history, never a fabricated neutral score.
    a.flat_returns = 0;
    a.last_k = -1;
    return a;
}

inline float adaptive_log_return(float current, float previous) {
    // Subtract prices before taking logs: subtracting two float32 log prices
    // loses small but representable returns after a large isolated move.
    float x = (current - previous) / previous;
    if (fabs(x) < 0.01f) {
        return x * (1.0f + x * (-0.5f + x * (1.0f / 3.0f
            + x * (-0.25f + x * 0.2f))));
    }
    return log(current) - log(previous);
}

inline void update_adaptive_rms(
    thread AdaptiveTiming& a, constant float* bars,
    int k, int first_valid, int stride, int close_offset
) {
    if (!(a.adverse_weight > 0.0f || a.score_weight > 0.0f)) return;
    if (k <= first_valid) return;
    int n = k - first_valid;
    float alpha = 2.0f / (a.span + 1.0f);
    float decay = 1.0f - alpha;
    float value = adaptive_log_return(bars[k * stride + close_offset], bars[(k - 1) * stride + close_offset]);
    a.flat_returns = value == 0.0f ? min(a.flat_returns + 1, a.window) : 0;
    bool restart = a.last_k != k - 1;
    a.last_k = k;
    float square_before = a.square;
    a.mean = fma(decay, a.mean, alpha * value);
    a.square = fma(decay, a.square, alpha * value * value);
    if (n > a.window) {
        int old = k - a.window;
        float expired = adaptive_log_return(bars[old * stride + close_offset], bars[(old - 1) * stride + close_offset]);
        float weight = alpha * pow(decay, float(a.window));
        a.mean -= weight * expired;
        a.square -= weight * expired * expired;
    }
    if (a.flat_returns >= a.window) {
        a.mean = 0.0f;
        a.square = 0.0f;
    } else if (restart || (n > a.window && (n % a.window == 0
               || a.square <= 0.0001f * square_before))) {
        // Bound accumulated subtraction error, especially when a shock expires.
        // Periodic rebuilding is O(window) once per window: amortized O(1).
        a.mean = 0.0f;
        a.square = 0.0f;
        for (int j = max(first_valid + 1, k - a.window + 1); j <= k; ++j) {
            float r = adaptive_log_return(bars[j * stride + close_offset], bars[(j - 1) * stride + close_offset]);
            a.mean = fma(decay, a.mean, alpha * r);
            a.square = fma(decay, a.square, alpha * r * r);
        }
    }
    a.score = n < a.window ? NAN : (a.square == 0.0f
        ? 0.0f : clamp(a.mean / sqrt(a.square), -1.0f, 1.0f));
}

inline float adaptive_duration(
    thread const AdaptiveTiming& a, float base, float exposure_ratio, bool short_side
) {
    if (a.maximum >= 0.0f && fmax(base, a.minimum) >= a.maximum) {
        return ceil(a.maximum);
    }
    float minutes = base;
    if (a.exposure_weight > 0.0f) minutes += a.exposure_weight * exposure_ratio;
    if (a.adverse_weight > 0.0f) {
        if (!isfinite(a.score)) return INFINITY;
        minutes += a.adverse_weight * fmax(short_side ? a.score : -a.score, 0.0f);
    }
    minutes = fmax(minutes, a.minimum);
    if (a.maximum >= 0.0f) minutes = fmin(minutes, a.maximum);
    return ceil(minutes);
}
