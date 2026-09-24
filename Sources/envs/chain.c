#include "../env.h"

typedef struct { uint32_t limit; float scale; uint32_t lastActionObs; } walkKnobs;

static const walkKnobs WALK_KNOBS[2] = {
    { 15u, 16.0f, 0u },
    { 31u, 32.0f, 1u }
};


static void walk_obs(const walkKnobs *k, uint32_t pos, uint32_t lastAction, float *obs) {
    obs[0] = (float)pos / k->scale;
    if (k->lastActionObs) { obs[1] = (float)lastAction; obs[2] = 1.0f; }
    else obs[1] = 1.0f;
}

void walk_reset(uint32_t instance, uint32_t seed, uint32_t *words, uint32_t *extra, uint32_t steps) { (void)steps; (void)extra; (void)instance; (void)seed; (void)words; }

static inline __attribute__((always_inline)) float walk_move(const walkKnobs *k, const float *ctrl, uint32_t *words, float *obs, uint32_t *done) {
    uint32_t up = (uint32_t)ctrl[0] & 1u, pos = words[0], forward = pos + 1u, back = pos > 0u ? pos - 1u : 0u, goal = up & (forward >= k->limit);
    forward = forward >= k->limit ? 0u : forward;
    uint32_t next = up ? forward : back;
    *done = goal;
    words[0] = next;
    words[1] = up;
    walk_obs(k, next, up, obs);
    return (float)goal;
}

float walk_step(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done, const float *shared) {
    (void)shared;
    (void)seed; (void)step; (void)extra;
    switch (instance) {
        case 0u: return walk_move(&WALK_KNOBS[0], ctrl, words, obs, done);
        case 1u: return walk_move(&WALK_KNOBS[1], ctrl, words, obs, done);
        default: __builtin_trap();
    }
}

void walk_observe(uint32_t instance, uint32_t seed, const uint32_t *words, const uint32_t *extra, uint32_t step, float *obs, const float *shared) {
    (void)shared;
    (void)seed; (void)extra; (void)step;
    walk_obs(&WALK_KNOBS[instance], words[0], words[1], obs);
}

void walk_oracle(uint32_t instance, uint32_t seed, uint32_t step, const uint32_t *words, uint32_t *extra, const float *obs, float *ctrl, const float *shared) {
    (void)shared;
    (void)instance; (void)seed; (void)step; (void)words; (void)extra; (void)obs;
    ctrl[0] = 1.0f;
}
