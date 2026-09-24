#include "../env.h"

#define GARNET_PERIOD 100000u
#define GARNET_DISCOUNT 0.9999
#define GARNET_STATES_MAX 50u
#define GARNET_IMPROVEMENTS 200u
#define GARNET_IMPROVEMENT_MARGIN 1e-9

typedef struct { uint32_t states, actions, branching, bits, resample; } garnetKnobs;

static const garnetKnobs GARNET_KNOBS[4] = {
    { 10u, 2u, 2u, 4u, 0u },
    { 50u, 4u, 3u, 6u, 0u },
    { 10u, 2u, 2u, 4u, 1u },
    { 50u, 4u, 3u, 6u, 1u }
};


typedef struct { uint32_t cut[2], next; float reward; } garnetTransition;

typedef struct { uint32_t *next; float *prob; float *reward; garnetTransition *transition; uint32_t *greedy; uint32_t epoch; } garnetModel;

enum { GARNET_EPOCH, GARNET_TAG, GARNET_LEVEL, GARNET_GREEDY_EPOCH = GARNET_LEVEL + ENV_BATCH, GARNET_HEADER };

static garnetModel garnet_model(const garnetKnobs *k, uint32_t *extra) {
    uint32_t pairs = k->states * k->actions, edges = pairs * k->branching;
    garnetModel m;
    m.next = extra + GARNET_HEADER;
    m.prob = (float *)(m.next + edges);
    m.reward = m.prob + edges;
    m.transition = (garnetTransition *)(m.reward + pairs);
    m.greedy = (uint32_t *)(m.transition + pairs);
    m.epoch = extra[GARNET_EPOCH];
    return m;
}

uint32_t garnet_extraWords(uint32_t instance, uint32_t steps) {
    (void)steps;
    const garnetKnobs *k = &GARNET_KNOBS[instance];
    uint32_t pairs = k->states * k->actions, edges = pairs * k->branching;
    return GARNET_HEADER + 2u * edges + pairs + pairs * (uint32_t)(sizeof(garnetTransition) / sizeof(uint32_t)) + k->states;
}

static double garnet_q(const garnetKnobs *k, const garnetModel *m, const double *value, uint32_t state, uint32_t action) {
    uint32_t pair = state * k->actions + action;
    double q = (double)m->reward[pair];
    for (uint32_t b = 0u; b < k->branching; b++) q += GARNET_DISCOUNT * (double)m->prob[pair * k->branching + b] * value[m->next[pair * k->branching + b]];
    return q;
}

static void garnet_evaluate(const garnetKnobs *k, const garnetModel *m, double *value) {
    uint32_t n = k->states;
    double a[GARNET_STATES_MAX * GARNET_STATES_MAX], b[GARNET_STATES_MAX];
    for (uint32_t i = 0u; i < n * n; i++) a[i] = 0.0;
    for (uint32_t s = 0u; s < n; s++) {
        uint32_t pair = s * k->actions + m->greedy[s];
        a[s * n + s] += 1.0;
        for (uint32_t br = 0u; br < k->branching; br++) a[s * n + m->next[pair * k->branching + br]] -= GARNET_DISCOUNT * (double)m->prob[pair * k->branching + br];
        b[s] = (double)m->reward[pair];
    }
    for (uint32_t col = 0u; col < n; col++) {
        uint32_t pivot = col;
        for (uint32_t row = col + 1u; row < n; row++) if (fabs(a[row * n + col]) > fabs(a[pivot * n + col])) pivot = row;
        if (pivot != col) {
            for (uint32_t c = 0u; c < n; c++) { double t = a[col * n + c]; a[col * n + c] = a[pivot * n + c]; a[pivot * n + c] = t; }
            double t = b[col]; b[col] = b[pivot]; b[pivot] = t;
        }
        for (uint32_t row = col + 1u; row < n; row++) {
            double factor = a[row * n + col] / a[col * n + col];
            if (factor == 0.0) continue;
            for (uint32_t c = col; c < n; c++) a[row * n + c] -= factor * a[col * n + c];
            b[row] -= factor * b[col];
        }
    }
    for (uint32_t back = 0u; back < n; back++) {
        uint32_t row = n - 1u - back;
        double sum = b[row];
        for (uint32_t c = row + 1u; c < n; c++) sum -= a[row * n + c] * value[c];
        value[row] = sum / a[row * n + row];
    }
}

static void garnet_solve(const garnetKnobs *k, garnetModel m) {
    double value[GARNET_STATES_MAX];
    for (uint32_t s = 0u; s < k->states; s++) {
        uint32_t best = 0u;
        for (uint32_t action = 1u; action < k->actions; action++) if (m.reward[s * k->actions + action] > m.reward[s * k->actions + best]) best = action;
        m.greedy[s] = best;
    }
    for (uint32_t improvement = 0u; improvement < GARNET_IMPROVEMENTS; improvement++) {
        garnet_evaluate(k, &m, value);
        uint32_t changed = 0u;
        for (uint32_t s = 0u; s < k->states; s++) {
            uint32_t best = m.greedy[s];
            double bestQ = garnet_q(k, &m, value, s, best);
            for (uint32_t action = 0u; action < k->actions; action++) {
                double q = garnet_q(k, &m, value, s, action);
                if (q > bestQ + GARNET_IMPROVEMENT_MARGIN) { bestQ = q; best = action; }
            }
            if (best != m.greedy[s]) { m.greedy[s] = best; changed = 1u; }
        }
        if (!changed) break;
    }
}

static void garnet_build(const garnetKnobs *k, uint32_t seed, uint32_t epoch, uint32_t *extra) {
    extra[GARNET_EPOCH] = epoch;
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
            m.reward[pair] = normalFrom(r.y, r.z);
            garnetTransition *t = &m.transition[pair];
            float cumulative = 0.0f;
            t->next = 0u;
            for (uint32_t b = 0u; b < k->branching; b++) {
                cumulative += m.prob[pair * k->branching + b];
                if (b + 1u < k->branching) t->cut[b] = (uint32_t)ceilf(cumulative * 0x1p24f);
                t->next |= m.next[pair * k->branching + b] << (8u * b);
            }
            t->reward = m.reward[pair];
        }
}

static void garnet_obs(const garnetKnobs *k, uint32_t state, float *obs) {
    float4 bit[2] = { flag4((lane4)state & (lane4){ 1u, 2u, 4u, 8u }), flag4((lane4)state & (lane4){ 16u, 32u, 64u, 128u }) };
    __builtin_memcpy(obs, bit, k->bits * sizeof(float));
    obs[k->bits] = 1.0f;
}

static void garnet_draw(uint32_t seed, uint32_t step, uint32_t *extra) {
    lane4 level = philox4(seed, ENV_RNG_STREAM, step + (lane4){ 0u, 1u, 2u, 3u }, (lane4)0u, (lane4)0u, (lane4)0u).x >> 8u;
    __builtin_memcpy(extra + GARNET_LEVEL, &level, sizeof level);
}

void garnet_reset(uint32_t instance, uint32_t seed, uint32_t *words, uint32_t *extra, uint32_t steps) {
    (void)steps;
    const garnetKnobs *k = &GARNET_KNOBS[instance];
    garnet_build(k, seed, 0u, extra);
    extra[GARNET_GREEDY_EPOCH] = 0u;
    batchReset(&extra[GARNET_TAG]);
    words[0] = philox4x32(seed, RNG_STREAM_RESET, 0xffffu, 0u, 0u, 0u).x % k->states;
}

static inline __attribute__((always_inline)) float garnet_move(const garnetKnobs *k, uint32_t level, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs) {
    const garnetTransition *row = garnet_model(k, extra).transition + (uint32_t)ctrl[0] % k->actions, *t = row + (size_t)words[0] * k->actions;
    uint32_t next = t->next >> (8u * (k->branching - 1u));
    for (uint32_t b = k->branching - 1u; b > 0u; b--) next = level >= t->cut[b - 1u] ? next : (t->next >> (8u * (b - 1u))) & 255u;
    words[0] = next;
    garnet_obs(k, next, obs);
    return t->reward;
}

float garnet_step(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done, const float *shared);

static __attribute__((noinline)) float garnet_resample(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done, const float *shared) {
    garnet_build(&GARNET_KNOBS[instance], seed, step / GARNET_PERIOD, extra);
    return garnet_step(instance, seed, step, ctrl, words, extra, obs, done, shared);
}

static __attribute__((noinline)) float garnet_refill(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done, const float *shared) {
    garnet_draw(seed, step, extra);
    return garnet_step(instance, seed, step, ctrl, words, extra, obs, done, shared);
}

static inline __attribute__((always_inline)) float garnet_play(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done, const float *shared) {
    const garnetKnobs *k = &GARNET_KNOBS[instance];
    if (k->resample && step / GARNET_PERIOD != extra[GARNET_EPOCH]) return garnet_resample(instance, seed, step, ctrl, words, extra, obs, done, shared);
    uint32_t slot = batchSlot(&extra[GARNET_TAG], step);
    if (slot == ENV_BATCH) return garnet_refill(instance, seed, step, ctrl, words, extra, obs, done, shared);
    return garnet_move(k, extra[GARNET_LEVEL + slot], ctrl, words, extra, obs);
}

float garnet_step(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done, const float *shared) {
    *done = 0u;
    switch (instance) {
        case 0u: return garnet_play(0u, seed, step, ctrl, words, extra, obs, done, shared);
        case 1u: return garnet_play(1u, seed, step, ctrl, words, extra, obs, done, shared);
        case 2u: return garnet_play(2u, seed, step, ctrl, words, extra, obs, done, shared);
        default: return garnet_play(3u, seed, step, ctrl, words, extra, obs, done, shared);
    }
}

void garnet_observe(uint32_t instance, uint32_t seed, const uint32_t *words, const uint32_t *extra, uint32_t step, float *obs, const float *shared) {
    (void)shared;
    (void)seed; (void)extra; (void)step;
    garnet_obs(&GARNET_KNOBS[instance], words[0], obs);
}

void garnet_oracle(uint32_t instance, uint32_t seed, uint32_t step, const uint32_t *words, uint32_t *extra, const float *obs, float *ctrl, const float *shared) {
    (void)shared;
    (void)seed; (void)step; (void)obs;
    const garnetKnobs *k = &GARNET_KNOBS[instance];
    garnetModel m = garnet_model(k, extra);
    if (extra[GARNET_GREEDY_EPOCH] != m.epoch + 1u) { garnet_solve(k, m); extra[GARNET_GREEDY_EPOCH] = m.epoch + 1u; }
    ctrl[0] = (float)m.greedy[words[0]];
}
