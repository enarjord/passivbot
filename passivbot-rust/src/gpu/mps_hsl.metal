// Constant-sized v9 controller/report transport. No observation tree, retained
// fill window, historical composer or alternate risk model is selected here.
#ifndef PASSIVBOT_HSL_FACTUAL_ONLY
#define PASSIVBOT_HSL_FACTUAL_ONLY 1
#endif
#ifndef PASSIVBOT_HSL_FACTS_ENABLED
#define PASSIVBOT_HSL_FACTS_ENABLED 0
#endif
#ifndef PASSIVBOT_HSL_NATIVE_CACHE_ENABLED
#define PASSIVBOT_HSL_NATIVE_CACHE_ENABLED 0
#endif
#ifndef PASSIVBOT_HSL_NATIVE_COIN_CACHE_ENABLED
#define PASSIVBOT_HSL_NATIVE_COIN_CACHE_ENABLED 0
#endif

// Inert positional buffer type while callers retain the stable kernel shape.
// V9 stores decision state inside the temporal replay owner, never in this tape.
struct HslNode {
    float reserved[8];
};
inline int hsl_storage_nodes(int capacity, int tree_size, int fact_capacity = 0) {
    return 0;
}

struct HslController {
    float raw;
    float ema;
    int last_observed;
    int flat_minute;
    int action;
    bool exposed;
};
inline HslController hsl_controller_init(
    device HslNode* tree, int capacity, int tree_size, float span
) {
    HslController h;
    h.raw = h.ema = 0.0f;
    h.last_observed = h.flat_minute = -1;
    h.action = 0;
    h.exposed = false;
    return h;
}
