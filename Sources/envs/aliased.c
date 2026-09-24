#include "../env.h"

#define ALIASED_CLAIM_REWARD 1.0f
#define ALIASED_FALSE_CLAIM -0.1f

typedef struct { uint32_t cells; } aliasedKnobs;

static const aliasedKnobs ALIASED_KNOBS[3] = {
    { 8u },
    { 32u },
    { 128u }
};


uint32_t aliased_extraWords(uint32_t instance, uint32_t steps) { (void)steps; return (ALIASED_KNOBS[instance].cells + 31u) / 32u; }

static uint32_t aliased_colour(const uint32_t *extra, uint32_t cell) { return (extra[cell >> 5u] >> (cell & 31u)) & 1u; }

static void aliased_obs(const uint32_t *extra, uint32_t position, float *obs) {
    obs[0] = (float)aliased_colour(extra, position);
    obs[1] = 1.0f;
}

static inline __attribute__((always_inline)) uint32_t aliased_landing(const aliasedKnobs *k, uint32_t seed, uint32_t index, uint32_t target) {
    uint32_t cell = philox4x32(seed, ENV_RNG_STREAM, index, 0u, 0u, 0u).x % k->cells;
    return cell == target ? (cell + 1u) % k->cells : cell;
}

static inline __attribute__((always_inline)) float aliased_advance(const aliasedKnobs *k, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, const uint32_t *extra, float *obs, uint32_t *done) {
    uint32_t position = words[0], claim = (uint32_t)ctrl[0] & 1u, next = (position + 1u) % k->cells;
    if (claim & (position == words[1])) {
        words[0] = aliased_landing(k, seed, step, words[1]);
        *done = 1u;
        aliased_obs(extra, words[0], obs);
        return ALIASED_CLAIM_REWARD;
    }
    *done = 0u;
    words[0] = claim ? position : next;
    aliased_obs(extra, words[0], obs);
    return claim ? ALIASED_FALSE_CLAIM : 0.0f;
}

void aliased_reset(uint32_t instance, uint32_t seed, uint32_t *words, uint32_t *extra, uint32_t steps) {
    const aliasedKnobs *k = &ALIASED_KNOBS[instance];
    words[1] = philox4x32(seed, RNG_STREAM_RESET, 0xfffeu, 0u, 0u, 0u).x % k->cells;
    words[0] = aliased_landing(k, seed, 0u, words[1]);
    for (uint32_t word = 0u; word < aliased_extraWords(instance, steps); word++) extra[word] = 0u;
    for (uint32_t cell = 0u; cell < k->cells; cell += 4u) {
        u44 r = philox4(seed, RNG_STREAM_RESET, cell + (lane4){ 0u, 1u, 2u, 3u }, 0u, 0u, 0u);
        lane4 colours = (r.x & 1u) << ((cell & 31u) + (lane4){ 0u, 1u, 2u, 3u });
        extra[cell >> 5u] |= colours.x | colours.y | colours.z | colours.w;
    }
}

float aliased_step(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done, const float *shared) {
    (void)shared;
    switch (instance) {
        case 0u: return aliased_advance(&ALIASED_KNOBS[0], seed, step, ctrl, words, extra, obs, done);
        case 1u: return aliased_advance(&ALIASED_KNOBS[1], seed, step, ctrl, words, extra, obs, done);
        default: return aliased_advance(&ALIASED_KNOBS[2], seed, step, ctrl, words, extra, obs, done);
    }
}

void aliased_observe(uint32_t instance, uint32_t seed, const uint32_t *words, const uint32_t *extra, uint32_t step, float *obs, const float *shared) {
    (void)shared;
    (void)instance; (void)seed; (void)step;
    aliased_obs(extra, words[0], obs);
}

void aliased_oracle(uint32_t instance, uint32_t seed, uint32_t step, const uint32_t *words, uint32_t *extra, const float *obs, float *ctrl, const float *shared) {
    (void)shared;
    (void)instance; (void)seed; (void)step; (void)extra; (void)obs;
    ctrl[0] = words[0] == words[1] ? 1.0f : 0.0f;
}
