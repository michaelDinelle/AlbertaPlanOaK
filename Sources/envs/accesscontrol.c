#include "../env.h"

#define ACCESS_SERVERS 10u
#define ACCESS_PRIORITIES 4u
#define ACCESS_DRIFT_PERIOD 200000u

typedef struct { uint32_t drift; } accessKnobs;

static const accessKnobs ACCESS_KNOBS[2] = {
    { 0u },
    { 1u }
};

static const float ACCESS_REWARD[ACCESS_PRIORITIES] = { 1.0f, 2.0f, 4.0f, 8.0f };
static const uint32_t ACCESS_THRESHOLD[ACCESS_PRIORITIES] = { 6u, 4u, 2u, 0u };

static float access_freeRate(const accessKnobs *k, uint32_t step) {
    if (!k->drift) return 0.06f;
    float s, c;
    oak_sincos(2.0f * OAK_PI * (float)(step % ACCESS_DRIFT_PERIOD) / (float)ACCESS_DRIFT_PERIOD, &s, &c);
    return 0.04f + 0.02f * s;
}

static void access_obs(uint32_t free, uint32_t priority, float *obs) {
    obs[0] = (float)free / (float)ACCESS_SERVERS;
    for (uint32_t p = 0u; p < ACCESS_PRIORITIES; p++) obs[1u + p] = p == priority ? 1.0f : 0.0f;
    obs[1u + ACCESS_PRIORITIES] = 1.0f;
}

void access_reset(uint32_t instance, uint32_t seed, uint32_t *words, uint32_t *extra, uint32_t steps) {
    (void)steps;
    (void)instance; (void)extra;
    words[0] = ACCESS_SERVERS;
    words[1] = philox4x32(seed, RNG_STREAM_RESET, 0u, 0u, 0u, 0u).x % ACCESS_PRIORITIES;
}

float access_step(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done, const float *shared) {
    (void)shared;
    (void)extra;
    const accessKnobs *k = &ACCESS_KNOBS[instance];
    uint32_t free = words[0], priority = words[1];
    *done = 0u;
    float reward = 0.0f;
    if (((uint32_t)ctrl[0] & 1u) != 0u && free > 0u) { reward = ACCESS_REWARD[priority]; free -= 1u; }
    u4 r = philox4x32(seed, ENV_RNG_STREAM, step, 0u, 0u, 0u);
    uint32_t first[4] = { r.x, r.y, r.z, r.w };
    float freeRate = access_freeRate(k, step);
    uint32_t busy = ACCESS_SERVERS - free;
    for (uint32_t s = 0u; s < busy; s++) {
        uint32_t word = s < 4u ? first[s] : philox4x32(seed, ENV_RNG_STREAM, step, 1u + s, 0u, 0u).x;
        if (u01(word) < freeRate) free += 1u;
    }
    priority = philox4x32(seed, ENV_RNG_STREAM, step, 32u, 0u, 0u).x % ACCESS_PRIORITIES;
    words[0] = free; words[1] = priority;
    access_obs(free, priority, obs);
    return reward;
}

void access_observe(uint32_t instance, uint32_t seed, const uint32_t *words, const uint32_t *extra, uint32_t step, float *obs, const float *shared) {
    (void)shared;
    (void)instance; (void)seed; (void)extra; (void)step;
    access_obs(words[0], words[1], obs);
}

void access_oracle(uint32_t instance, uint32_t seed, uint32_t step, const uint32_t *words, const uint32_t *extra, const float *obs, float *ctrl, const float *shared) {
    (void)shared;
    (void)instance; (void)seed; (void)step; (void)extra; (void)obs;
    ctrl[0] = words[0] > ACCESS_THRESHOLD[words[1]] ? 1.0f : 0.0f;
}
