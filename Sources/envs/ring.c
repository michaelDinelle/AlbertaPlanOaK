#include "../env.h"

#define RING_CELLS 64u

typedef struct { uint32_t arcLength; } ringKnobs;

static const ringKnobs RING_KNOBS[2] = {
    { RING_CELLS / 2u },
    { RING_CELLS / 8u }
};


static uint32_t ring_colour(const ringKnobs *k, uint32_t position) { return (position / k->arcLength) & 1u; }

void ring_reset(uint32_t instance, uint32_t seed, uint32_t *words, uint32_t *extra, uint32_t steps) {
    (void)steps;
    (void)instance; (void)extra;
    words[0] = philox4x32(seed, RNG_STREAM_RESET, 0u, 0u, 0u, 0u).x % RING_CELLS;
}

float ring_step(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done, const float *shared) {
    (void)shared;
    (void)seed; (void)step; (void)extra;
    const ringKnobs *k = &RING_KNOBS[instance];
    *done = 0u;
    words[0] = (words[0] + 1u) % RING_CELLS;
    uint32_t colour = ring_colour(k, words[0]);
    obs[0] = (float)colour;
    obs[1] = 1.0f;
    return ((uint32_t)ctrl[0] & 1u) == colour ? 1.0f : 0.0f;
}

void ring_observe(uint32_t instance, uint32_t seed, const uint32_t *words, const uint32_t *extra, uint32_t step, float *obs, const float *shared) {
    (void)shared;
    (void)seed; (void)extra; (void)step;
    obs[0] = (float)ring_colour(&RING_KNOBS[instance], words[0]);
    obs[1] = 1.0f;
}

void ring_oracle(uint32_t instance, uint32_t seed, uint32_t step, const uint32_t *words, uint32_t *extra, const float *obs, float *ctrl, const float *shared) {
    (void)shared;
    (void)seed; (void)step; (void)extra; (void)obs;
    ctrl[0] = (float)ring_colour(&RING_KNOBS[instance], (words[0] + 1u) % RING_CELLS);
}
