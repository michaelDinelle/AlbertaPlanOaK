#include "../env.h"

#define GARNET_SWEEPS 200u
#define GARNET_PERIOD 100000u

typedef struct { uint32_t states, actions, branching, bits, resample; } garnetKnobs;

static const garnetKnobs GARNET_KNOBS[4] = {
    { 10u, 2u, 2u, 4u, 0u },
    { 50u, 4u, 3u, 6u, 0u },
    { 10u, 2u, 2u, 4u, 1u },
    { 50u, 4u, 3u, 6u, 1u }
};


typedef struct { uint32_t *next; float *prob; float *reward; float *value; uint32_t *greedy; uint32_t epoch; } garnetModel;

static garnetModel garnet_model(const garnetKnobs *k, uint32_t *extra) {
    uint32_t pairs = k->states * k->actions, edges = pairs * k->branching;
    garnetModel m;
    uint32_t *base = extra;
    m.next = base + 1u;
    m.prob = (float *)(m.next + edges);
    m.reward = m.prob + edges;
    m.value = m.reward + pairs;
    m.greedy = (uint32_t *)(m.value + k->states);
    m.epoch = base[0];
    return m;
}

uint32_t garnet_extraWords(uint32_t instance, uint32_t steps) {
    (void)steps;
    const garnetKnobs *k = &GARNET_KNOBS[instance];
    uint32_t pairs = k->states * k->actions, edges = pairs * k->branching;
    return 1u + 2u * edges + pairs + k->states + k->states;
}

static void garnet_build(const garnetKnobs *k, uint32_t seed, uint32_t epoch, uint32_t *extra) {
    extra[0] = epoch;
    garnetModel m = garnet_model(k, extra);
    for (uint32_t s = 0u; s < k->states; s++)
        for (uint32_t a = 0u; a < k->actions; a++) {
            uint32_t pair = s * k->actions + a;
            u4 r = philox4x32(seed, RNG_STREAM_RESET, epoch, pair, 0u, 0u);
            float total = 0.0f;
            for (uint32_t b = 0u; b < k->branching; b++) {
                u4 e = philox4x32(seed, RNG_STREAM_RESET, epoch, pair, 1u + b, 0u);
                m.next[pair * k->branching + b] = e.x % k->states;
                m.prob[pair * k->branching + b] = u01open(e.y);
                total += m.prob[pair * k->branching + b];
            }
            for (uint32_t b = 0u; b < k->branching; b++) m.prob[pair * k->branching + b] /= total;
            m.reward[pair] = (r.x % 10u) == 0u ? normalFrom(r.y, r.z) : 0.0f;
        }
    for (uint32_t s = 0u; s < k->states; s++) m.value[s] = 0.0f;
    for (uint32_t sweep = 0u; sweep < GARNET_SWEEPS; sweep++) {
        float reference = m.value[0];
        for (uint32_t s = 0u; s < k->states; s++) {
            float best = -FLT_MAX;
            uint32_t bestAction = 0u;
            for (uint32_t a = 0u; a < k->actions; a++) {
                uint32_t pair = s * k->actions + a;
                float q = m.reward[pair];
                for (uint32_t b = 0u; b < k->branching; b++) q += m.prob[pair * k->branching + b] * m.value[m.next[pair * k->branching + b]];
                if (q > best) { best = q; bestAction = a; }
            }
            m.value[s] = best - reference;
            m.greedy[s] = bestAction;
        }
    }
}

static void garnet_obs(const garnetKnobs *k, uint32_t state, float *obs) {
    for (uint32_t b = 0u; b < k->bits; b++) obs[b] = (float)((state >> b) & 1u);
    obs[k->bits] = 1.0f;
}

void garnet_reset(uint32_t instance, uint32_t seed, uint32_t *words, uint32_t *extra, uint32_t steps) {
    (void)steps;
    const garnetKnobs *k = &GARNET_KNOBS[instance];
    garnet_build(k, seed, 0u, extra);
    words[0] = philox4x32(seed, RNG_STREAM_RESET, 0xffffu, 0u, 0u, 0u).x % k->states;
}

float garnet_step(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done, const float *shared) {
    (void)shared;
    const garnetKnobs *k = &GARNET_KNOBS[instance];
    *done = 0u;
    if (k->resample && step / GARNET_PERIOD != extra[0]) garnet_build(k, seed, step / GARNET_PERIOD, extra);
    garnetModel m = garnet_model(k, extra);
    uint32_t pair = words[0] * k->actions + (uint32_t)ctrl[0] % k->actions;
    float reward = m.reward[pair];
    float roll = u01(philox4x32(seed, ENV_RNG_STREAM, step, 0u, 0u, 0u).x), cumulative = 0.0f;
    uint32_t next = m.next[pair * k->branching + k->branching - 1u];
    for (uint32_t b = 0u; b < k->branching; b++) {
        cumulative += m.prob[pair * k->branching + b];
        if (roll < cumulative) { next = m.next[pair * k->branching + b]; break; }
    }
    words[0] = next;
    garnet_obs(k, next, obs);
    return reward;
}

void garnet_observe(uint32_t instance, uint32_t seed, const uint32_t *words, const uint32_t *extra, uint32_t step, float *obs, const float *shared) {
    (void)shared;
    (void)seed; (void)extra; (void)step;
    garnet_obs(&GARNET_KNOBS[instance], words[0], obs);
}

void garnet_oracle(uint32_t instance, uint32_t seed, uint32_t step, const uint32_t *words, const uint32_t *extra, const float *obs, float *ctrl, const float *shared) {
    (void)shared;
    (void)seed; (void)step; (void)obs;
    garnetModel m = garnet_model(&GARNET_KNOBS[instance], (uint32_t *)extra);
    ctrl[0] = (float)m.greedy[words[0]];
}
