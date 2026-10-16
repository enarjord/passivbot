// Canonical decimal operands arrive as finite base-2^20 limbs. They preserve
// authored distinctions lost by float32 conversion without NaN bit payloads.
struct GpuDecimal {
    long coefficient;
    int exponent;
};

inline GpuDecimal load_gpu_decimal(constant float* values, int offset) {
    int encoded = int(values[offset + 3]);
    ulong magnitude = ulong(values[offset])
        | (ulong(values[offset + 1]) << 20) | (ulong(values[offset + 2]) << 40);
    GpuDecimal result;
    bool negative = encoded >= 512 && encoded < 1536;
    result.coefficient = negative ? -long(magnitude) : long(magnitude);
    result.exponent = negative ? encoded - 1024 : encoded;
    return result;
}

inline GpuDecimal invalid_gpu_decimal() {
    GpuDecimal result;
    result.coefficient = 0;
    result.exponent = 2048;
    return result;
}

inline bool gpu_decimal_valid(GpuDecimal value) {
    return value.exponent >= -18 && value.exponent <= 18;
}

inline ulong gpu_decimal_power(int exponent) {
    ulong result = 1;
    for (int i = 0; i < exponent; ++i) result *= 10;
    return result;
}

inline GpuDecimal add_gpu_decimal(GpuDecimal a, GpuDecimal b) {
    if (!gpu_decimal_valid(a) || !gpu_decimal_valid(b)) return invalid_gpu_decimal();
    int exponent = min(a.exponent, b.exponent);
    if (a.exponent - exponent > 18 || b.exponent - exponent > 18)
        return invalid_gpu_decimal();
    ulong a_scale = gpu_decimal_power(a.exponent - exponent);
    ulong b_scale = gpu_decimal_power(b.exponent - exponent);
    // Symmetric range avoids abs(LONG_MIN) and keeps the following sum checked.
    const long limit = 0x3fffffffffffffffl;
    if (a.coefficient > limit / long(a_scale) || a.coefficient < -limit / long(a_scale)
        || b.coefficient > limit / long(b_scale) || b.coefficient < -limit / long(b_scale))
        return invalid_gpu_decimal();
    long left = a.coefficient * long(a_scale), right = b.coefficient * long(b_scale);
    if (right > 0 && left > limit - right || right < 0 && left < -limit - right)
        return invalid_gpu_decimal();
    GpuDecimal result;
    result.coefficient = left + right;
    result.exponent = exponent;
    return result;
}

// Return an integer rounded from count * decimal, or -1 if proof is outside
// the checked domain. Long multiplication/division is done as quotient and
// remainder so the exact numerator need not fit in a machine integer.
inline long round_gpu_decimal_product(uint count, GpuDecimal value, bool nearest, bool up,
                                      float directional_envelope = 0.0f) {
    if (!gpu_decimal_valid(value) || value.coefficient < 0) return -1;
    ulong coefficient = ulong(value.coefficient), denominator = 1;
    if (value.exponent >= 0) {
        ulong scale = gpu_decimal_power(value.exponent);
        if (coefficient > 0x7ffffffffffffffful / scale) return -1;
        coefficient *= scale;
    } else denominator = gpu_decimal_power(-value.exponent);
    ulong whole = coefficient / denominator, part = coefficient % denominator;
    // All consumers need resolvable quantity counts or signed price ticks.
    const ulong limit = 0x7ffffffful;
    if (whole > limit / max(ulong(count), ulong(1))) return -1;
    ulong quotient = 0, remainder = 0;
    for (int bit = 30; bit >= 0; --bit) {
        quotient *= 2;
        remainder *= 2;
        if (remainder >= denominator) { remainder -= denominator; ++quotient; }
        if ((count & (1u << bit)) != 0) {
            quotient += whole;
            remainder += part;
            if (remainder >= denominator) { remainder -= denominator; ++quotient; }
        }
        if (quotient > limit) return -1;
    }
    // Rust only overrides fused f64 price arithmetic for decimal alignment.
    // A nonzero rational remainder inside its arithmetic envelope cannot prove
    // which side the fused result will choose. Decline; never snap it to a tick.
    if (directional_envelope > 0.0f && remainder != 0
        && float(min(remainder, denominator - remainder)) / float(denominator)
            <= directional_envelope) return -1;
    if (nearest ? remainder >= denominator / 2 + denominator % 2
                : up && remainder != 0) ++quotient;
    return quotient <= limit ? long(quotient) : -1;
}

inline float rounded_gpu_reentry_product(float quantity, float factor, float step,
                                        GpuDecimal decimal_factor) {
    float units = rint(quantity / step);
    bool recoverable = quantity >= 0.0f && units >= 0.0f && units < 0x1p24f
        && units * step == quantity
        && quantity - nextafter(quantity, -INFINITY) <= step
        && nextafter(quantity, INFINITY) - quantity <= step;
    float product = units * factor;
    bool boundary = fabs((product - floor(product)) - 0.5f)
        <= 4.0f * 0x1p-23f * fabs(product);
    if (recoverable && boundary) {
        long count = round_gpu_decimal_product(uint(units), decimal_factor, true, false);
        if (count >= 0 && count < (1l << 24)) return float(count) * step;
    }
    return quantity * factor;
}

inline int canonical_gpu_relative_ticks(int base_tick, GpuDecimal threshold,
    GpuDecimal retracement, bool subtract_threshold, bool include_retracement, bool up) {
    if (base_tick < 1 || !gpu_decimal_valid(threshold)
        || include_retracement && !gpu_decimal_valid(retracement)) return -1;
    // Conservative f64 operand/FMA/venue-division envelope in tick units. It
    // includes the separate operands when their difference nearly cancels.
    float threshold_magnitude = fabs(float(threshold.coefficient));
    threshold_magnitude = threshold.exponent >= 0
        ? threshold_magnitude * float(gpu_decimal_power(threshold.exponent))
        : threshold_magnitude / float(gpu_decimal_power(-threshold.exponent));
    float retracement_magnitude = 0.0f;
    if (include_retracement) {
        retracement_magnitude = fabs(float(retracement.coefficient));
        retracement_magnitude = retracement.exponent >= 0
            ? retracement_magnitude * float(gpu_decimal_power(retracement.exponent))
            : retracement_magnitude / float(gpu_decimal_power(-retracement.exponent));
    }
    float directional_envelope = (64.0f * 0x1p-52f * float(base_tick))
        * (1.0f + threshold_magnitude + retracement_magnitude);
    GpuDecimal one;
    one.coefficient = 1;
    one.exponent = 0;
    if (subtract_threshold) threshold.coefficient = -threshold.coefficient;
    GpuDecimal factor = add_gpu_decimal(one, threshold);
    if (include_retracement) {
        if (!subtract_threshold) retracement.coefficient = -retracement.coefficient;
        factor = add_gpu_decimal(factor, retracement);
    }
    long count = round_gpu_decimal_product(uint(base_tick), factor, false, up, directional_envelope);
    return count >= 0 ? int(count) : -1;
}

// PASSIVBOT_DECIMAL_ROUNDING_END
