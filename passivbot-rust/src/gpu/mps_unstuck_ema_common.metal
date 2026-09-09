// Rust seeds adjusted price EMA numerator/denominator with price/1, so its
// denominator stays one and the recurrence reduces to the seeded EMA below.
// Horizons arrive in candle periods after host-side minute scaling.
struct UnstuckEmaBand {
    float3 alpha;
    float3 values;
};

inline UnstuckEmaBand init_unstuck_ema_band(float span0, float span1, float seed) {
    UnstuckEmaBand band;
    float3 spans = float3(span0, span1, sqrt(span0) * sqrt(span1));
    band.alpha = clamp(2.0f / (spans + 1.0f), 0.0f, 1.0f);
    band.values = float3(seed);
    return band;
}

inline void update_unstuck_ema_band(thread UnstuckEmaBand& band, float close) {
    band.values = fma(band.alpha, float3(close) - band.values, band.values);
}

inline float unstuck_ema_lower(thread const UnstuckEmaBand& band) {
    return fmin(band.values.x, fmin(band.values.y, band.values.z));
}

inline float unstuck_ema_upper(thread const UnstuckEmaBand& band) {
    return fmax(band.values.x, fmax(band.values.y, band.values.z));
}
