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

static uint32_t tmaze_cue(const tmazeKnobs *k, uint32_t laneSeed, uint32_t resampleIndex, uint32_t *extra) {
    uint32_t slot = batchSlot(extra, resampleIndex + 1u);
    if (slot < ENV_BATCH) return extra[1u + slot];
    lane4 cues = philox4(laneSeed, ENV_RNG_STREAM, resampleIndex + (lane4){ 0u, 1u, 2u, 3u }, 0u, 0u, 0u).x & ((1u << k->cueBits) - 1u);
    __builtin_memcpy(extra + 1, &cues, sizeof cues);
    return cues[0];
}

uint32_t tmaze_extraWords(uint32_t instance, uint32_t steps) {
    (void)instance; (void)steps;
    return 1u + ENV_BATCH;
}

void tmaze_reset(uint32_t instance, uint32_t seed, uint32_t *words, uint32_t *extra, uint32_t steps) { (void)steps; (void)instance; (void)seed; (void)words; batchReset(extra); }

static __attribute__((noinline)) float tmaze_resample(const tmazeKnobs *k, uint32_t laneSeed, uint32_t action, uint32_t *words, uint32_t *extra, float *obs, uint32_t *doneOut) {
    uint32_t cue = words[1], resampleIndex = words[3], cueMask = (1u << k->cueBits) - 1u;
    float reward = (action & cueMask) == cue ? 1.0f : 0.0f;
    cue = tmaze_cue(k, laneSeed, resampleIndex, extra);
    resampleIndex = resampleIndex + 1u;
    uint32_t nextLastAction = action & 1u;
    words[0] = 0u; words[1] = cue; words[2] = nextLastAction; words[3] = resampleIndex;
    tmaze_obs(k, 0u, cue, nextLastAction, obs);
    *doneOut = 1u;
    return reward;
}

static inline __attribute__((always_inline)) float tmaze_move(const tmazeKnobs *k, uint32_t laneSeed, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *doneOut) {
    uint32_t action = (uint32_t)ctrl[0];
    uint32_t pos = words[0], end = k->length - 1u;
    if (pos >= end) return tmaze_resample(k, laneSeed, action, words, extra, obs, doneOut);
    uint32_t nextLastAction = action & 1u;
    pos = pos + nextLastAction - ((nextLastAction ^ 1u) & (pos > 0u));
    words[0] = pos; words[2] = nextLastAction;
    tmaze_obs(k, pos, words[1], nextLastAction, obs);
    *doneOut = 0u;
    return 0.0f;
}

float tmaze_step(uint32_t instance, uint32_t laneSeed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *doneOut, const float *shared) {
    (void)shared;
    (void)step;
    switch (instance) {
        case 0u: return tmaze_move(&TMAZE_KNOBS[0], laneSeed, ctrl, words, extra, obs, doneOut);
        case 1u: return tmaze_move(&TMAZE_KNOBS[1], laneSeed, ctrl, words, extra, obs, doneOut);
        case 2u: return tmaze_move(&TMAZE_KNOBS[2], laneSeed, ctrl, words, extra, obs, doneOut);
        case 3u: return tmaze_move(&TMAZE_KNOBS[3], laneSeed, ctrl, words, extra, obs, doneOut);
        case 4u: return tmaze_move(&TMAZE_KNOBS[4], laneSeed, ctrl, words, extra, obs, doneOut);
        case 5u: return tmaze_move(&TMAZE_KNOBS[5], laneSeed, ctrl, words, extra, obs, doneOut);
        default: __builtin_trap();
    }
}

void tmaze_observe(uint32_t instance, uint32_t seed, const uint32_t *words, const uint32_t *extra, uint32_t step, float *obs, const float *shared) {
    (void)shared;
    (void)seed; (void)extra; (void)step;
    tmaze_obs(&TMAZE_KNOBS[instance], words[0], words[1], words[2], obs);
}

void tmaze_oracle(uint32_t instance, uint32_t seed, uint32_t step, const uint32_t *words, uint32_t *extra, const float *obs, float *ctrl, const float *shared) {
    (void)shared;
    (void)seed; (void)step; (void)extra; (void)obs;
    const tmazeKnobs *k = &TMAZE_KNOBS[instance];
    ctrl[0] = words[0] >= k->length - 1u ? (float)words[1] : 1.0f;
}
