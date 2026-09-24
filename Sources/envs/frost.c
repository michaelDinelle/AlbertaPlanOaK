#include "../env.h"

#define FROST_CELLS 7u
#define FROST_CENTER 3u
#define FROST_RATE 0.5f
#define FROST_CAPACITY 6.0f
#define FROST_STIMULUS 2u
#define FROST_EPISODE 1000u
#define FROST_CASCADE 15u
#define FROST_ALPHA 0.1f
#define FROST_LAMBDA 0.9f
#define FROST_THRESHOLD 3.0f

typedef struct { uint32_t calm, calmMin, calmMax, drift; } frostKnobs;

static const frostKnobs FROST_KNOBS[3] = {
    { 8u, 8u, 8u, 0u },
    { 8u, 6u, 11u, 0u },
    { 8u, 6u, 11u, 1u }
};

enum { FROST_POSITION, FROST_HEAT, FROST_PHASE, FROST_CALM, FROST_CLOCK, FROST_TOKEN, FROST_PREDICTION };
enum { FROST_FEATURES = FROST_CASCADE + 1u, FROST_TRACES = FROST_FEATURES, FROST_EXTRA = 2u * FROST_FEATURES };

uint32_t frost_extraWords(uint32_t instance, uint32_t steps) { (void)instance; (void)steps; return FROST_EXTRA; }

static inline __attribute__((always_inline)) uint32_t frost_calm(const frostKnobs *k, uint32_t seed, uint32_t step, uint32_t previous) {
    if (k->calmMin == k->calmMax) return k->calmMin;
    uint32_t draw = philox4x32(seed, ENV_RNG_STREAM, step, 0u, 0u, 0u).x;
    if (k->drift == 0u) return k->calmMin + draw % (k->calmMax - k->calmMin + 1u);
    uint32_t next = previous + draw % (2u * k->drift + 1u) - k->drift;
    return next < k->calmMin ? k->calmMin : next > k->calmMax ? k->calmMax : next;
}

static uint32_t frost_refuge(uint32_t position) { return position < FROST_CELLS - 1u - position ? position : FROST_CELLS - 1u - position; }

static void frost_obs(uint32_t position, uint32_t hazard, float heat, uint32_t token, float *obs) {
    float4 near = flag4((lane4)((lane4){ 0u, 1u, 2u, 3u } == position)), far = flag4((lane4)((lane4){ 4u, 5u, 6u, 7u } == position));
    far.w = (float)hazard;
    __builtin_memcpy(obs, &near, sizeof near);
    __builtin_memcpy(obs + 4, &far, sizeof far);
    obs[8] = heat;
    obs[9] = (float)token;
}

static float frost_learn(uint32_t *extra, uint32_t phase, uint32_t present, uint32_t next, uint32_t hazard, float prediction, uint32_t ended) {
    float *weight = (float *)extra, *trace = weight + FROST_TRACES;
    float ahead = weight[1u + next], bootstrap = hazard ? 0.0f : ahead;
    float scale = FROST_ALPHA * (1.0f + bootstrap - prediction), decay = hazard || ended ? 0.0f : FROST_LAMBDA;
    for (uint32_t j = 0u; j < FROST_FEATURES; j += 4u) {
        float4 w, e, x = flag4((lane4)(j + (lane4){ 0u, 1u, 2u, 3u } == 1u + phase));
        if (j == 0u) x.x = (float)present;
        __builtin_memcpy(&w, weight + j, sizeof w);
        __builtin_memcpy(&e, trace + j, sizeof e);
        e += x;
        w += scale * e;
        e *= decay;
        __builtin_memcpy(weight + j, &w, sizeof w);
        __builtin_memcpy(trace + j, &e, sizeof e);
    }
    float cascade = weight[1u + (ended ? 0u : next)], presence = weight[0];
    return cascade + (hazard && !ended ? presence : 0.0f);
}

void frost_reset(uint32_t instance, uint32_t seed, uint32_t *words, uint32_t *extra, uint32_t steps) {
    (void)steps;
    const frostKnobs *k = &FROST_KNOBS[instance];
    for (uint32_t i = 0u; i < FROST_EXTRA; i++) extra[i] = 0u;
    words[FROST_POSITION] = FROST_CENTER;
    words[FROST_HEAT] = floatWord(0.0f);
    words[FROST_PHASE] = 0u;
    words[FROST_CALM] = frost_calm(k, seed, 0u, k->calm);
    words[FROST_CLOCK] = 0u;
    words[FROST_TOKEN] = 0.0f <= FROST_THRESHOLD;
    words[FROST_PREDICTION] = floatWord(0.0f);
}

float frost_step(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done, const float *shared) {
    (void)shared;
    const frostKnobs *k = &FROST_KNOBS[instance];
    float pressed = ctrl[0];
    int target = (int)words[FROST_POSITION] + (pressed >= 0.0f && pressed < 3.0f ? (int)pressed : 0) - 1;
    uint32_t position = (uint32_t)(target < 0 ? 0 : target > (int)FROST_CELLS - 1 ? (int)FROST_CELLS - 1 : target);
    uint32_t phase = words[FROST_PHASE], calm = words[FROST_CALM], present = phase >= calm;
    uint32_t next = phase + 1u == calm + FROST_STIMULUS ? 0u : phase + 1u, hazard = next >= calm;
    float heat = wordFloat(words[FROST_HEAT]);
    heat = hazard && position - 1u < FROST_CELLS - 2u ? 0.0f : position == FROST_CENTER ? heat + FROST_RATE : heat;
    float reward = heat >= FROST_CAPACITY ? 1.0f : 0.0f;
    heat = heat >= FROST_CAPACITY ? 0.0f : heat;
    uint32_t clock = words[FROST_CLOCK] + 1u, ended = clock == FROST_EPISODE;
    float prediction = frost_learn(extra, phase, present, next, hazard, wordFloat(words[FROST_PREDICTION]), ended);
    uint32_t token = prediction <= FROST_THRESHOLD;
    position = ended ? FROST_CENTER : position;
    heat = ended ? 0.0f : heat;
    next = ended ? 0u : next;
    hazard = hazard && !ended;
    *done = ended;
    frost_obs(position, hazard, heat, token, obs);
    words[FROST_POSITION] = position;
    words[FROST_HEAT] = floatWord(heat);
    words[FROST_PHASE] = next;
    words[FROST_CLOCK] = ended ? 0u : clock;
    words[FROST_TOKEN] = token;
    words[FROST_PREDICTION] = floatWord(prediction);
    if (next == 0u) words[FROST_CALM] = frost_calm(k, seed, step, ended ? k->calm : calm);
    return reward;
}

void frost_observe(uint32_t instance, uint32_t seed, const uint32_t *words, const uint32_t *extra, uint32_t step, float *obs, const float *shared) {
    (void)instance; (void)seed; (void)extra; (void)step; (void)shared;
    frost_obs(words[FROST_POSITION], words[FROST_PHASE] >= words[FROST_CALM], wordFloat(words[FROST_HEAT]), words[FROST_TOKEN], obs);
}

void frost_oracle(uint32_t instance, uint32_t seed, uint32_t step, const uint32_t *words, uint32_t *extra, const float *obs, float *ctrl, const float *shared) {
    (void)instance; (void)seed; (void)step; (void)extra; (void)obs; (void)shared;
    uint32_t position = words[FROST_POSITION], phase = words[FROST_PHASE], calm = words[FROST_CALM];
    uint32_t inward = position < FROST_CENTER ? 2u : position > FROST_CENTER ? 0u : 1u, outward = position > FROST_CENTER ? 2u : 0u, code = inward;
    float heat = wordFloat(words[FROST_HEAT]), reach = (float)FROST_CENTER * FROST_RATE;
    if (phase >= calm) code = heat > 0.0f && frost_refuge(position) == 0u && phase == calm ? 1u : inward;
    else if (heat >= reach && heat <= FROST_CAPACITY - reach && calm - phase - 1u < frost_refuge(position + inward - 1u)) code = outward;
    ctrl[0] = (float)code;
}
