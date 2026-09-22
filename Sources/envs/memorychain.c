#include "../env.h"

typedef struct { uint32_t length, cueBits; } tmazeKnobs;

static const tmazeKnobs TMAZE_KNOBS[6] = {
    { 4u, 1u },
    { 16u, 1u },
    { 32u, 1u },
    { 64u, 1u },
    { 16u, 2u },
    { 64u, 2u }
};


static void tmaze_obs(const tmazeKnobs *k, uint32_t pos, uint32_t cue, uint32_t lastAction, float *obs) {
    uint32_t end = k->length - 1u;
    obs[0] = (float)pos / (float)k->length;
    obs[1] = (pos == 0u) ? (float)(cue & 1u) : 0.0f;
    obs[2] = (pos == 0u) ? 1.0f : 0.0f;
    obs[3] = (pos == end) ? 1.0f : 0.0f;
    obs[4] = (float)lastAction;
    obs[5] = 1.0f;
    if (k->cueBits > 1u) obs[6] = (pos == 0u) ? (float)((cue >> 1u) & 1u) : 0.0f;
}

void tmaze_reset(uint32_t instance, uint32_t seed, uint32_t *words, uint32_t *extra, uint32_t steps) { (void)steps; (void)instance; (void)seed; (void)words; (void)extra; }

float tmaze_step(uint32_t instance, uint32_t laneSeed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *doneOut, const float *shared) {
    (void)shared;
    (void)step; (void)extra;
    const tmazeKnobs *k = &TMAZE_KNOBS[instance];
    uint32_t action = (uint32_t)ctrl[0];
    uint32_t pos = words[0], cue = words[1], resampleIndex = words[3];
    uint32_t end = k->length - 1u, cueMask = (1u << k->cueBits) - 1u;
    float reward = 0.0f;
    uint32_t done = 0u;
    if (pos >= end) {
        reward = (action & cueMask) == cue ? 1.0f : 0.0f;
        done = 1u;
        u4 raw = philox4x32(laneSeed, ENV_RNG_STREAM, resampleIndex, 0u, 0u, 0u);
        cue = raw.x & cueMask;
        resampleIndex = resampleIndex + 1u;
        pos = 0u;
    } else {
        pos = (action & 1u) != 0u ? pos + 1u : (pos > 0u ? pos - 1u : 0u);
    }
    uint32_t nextLastAction = action & 1u;
    words[0] = pos; words[1] = cue; words[2] = nextLastAction; words[3] = resampleIndex;
    tmaze_obs(k, pos, cue, nextLastAction, obs);
    *doneOut = done;
    return reward;
}

void tmaze_observe(uint32_t instance, uint32_t seed, const uint32_t *words, const uint32_t *extra, uint32_t step, float *obs, const float *shared) {
    (void)shared;
    (void)seed; (void)extra; (void)step;
    tmaze_obs(&TMAZE_KNOBS[instance], words[0], words[1], words[2], obs);
}

void tmaze_oracle(uint32_t instance, uint32_t seed, uint32_t step, const uint32_t *words, const uint32_t *extra, const float *obs, float *ctrl, const float *shared) {
    (void)shared;
    (void)seed; (void)step; (void)extra; (void)obs;
    const tmazeKnobs *k = &TMAZE_KNOBS[instance];
    ctrl[0] = words[0] >= k->length - 1u ? (float)words[1] : 1.0f;
}
