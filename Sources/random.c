#include "policy.h"

void random_act(uint32_t env, uint32_t seed, uint32_t step, float *ctrl) {
    uint32_t actions = env_actions(env);
    if (!env_box(env)) {
        uint32_t w0 = baselineWord(seed, RNG_STREAM_RANDOM_ACT, step, 0u);
        uint32_t a = (uint32_t)(u01(w0) * (float)actions);
        if (a > actions - 1u) a = actions - 1u;
        ctrl[0] = (float)a;
        for (uint32_t d = 1u; d < actions; d++) ctrl[d] = 0.0f;
    } else {
        for (uint32_t d = 0u; d < actions; d++) ctrl[d] = 2.0f * u01(baselineWord(seed, RNG_STREAM_RANDOM_ACT, step, d)) - 1.0f;
    }
}
