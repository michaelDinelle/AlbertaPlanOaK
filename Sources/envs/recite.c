#include "../env.h"

#define RECITE_SYMBOLS 4u

typedef struct { uint32_t length; } reciteKnobs;

static const reciteKnobs RECITE_KNOBS[2] = {
    { 4u },
    { 16u }
};


uint32_t recite_extraWords(uint32_t instance, uint32_t steps) { (void)instance; (void)steps; return 3u; }

static uint32_t recite_part(uint32_t seed, uint32_t sequence, uint32_t part) {
    lane4 indices = 4u * part + (lane4){ 0u, 1u, 2u, 3u };
    u44 r = philox4(seed, ENV_RNG_STREAM, (lane4){ sequence, sequence, sequence, sequence }, indices, 0u, 0u);
    lane4 drawn = (r.x & 3u) << (2u * indices);
    return drawn.x | drawn.y | drawn.z | drawn.w;
}

static uint32_t recite_symbols(const reciteKnobs *k, uint32_t seed, uint32_t sequence) {
    uint32_t symbols = 0u;
    for (uint32_t part = 0u; part < k->length / 4u; part++) symbols |= recite_part(seed, sequence, part);
    return symbols;
}

static uint32_t recite_symbol(const reciteKnobs *k, uint32_t symbols, uint32_t phase) { return (symbols >> (2u * (phase % k->length))) & 3u; }

static void recite_check(const uint32_t *words, const uint32_t *extra) {
    if (extra[0] != words[0]) __builtin_trap();
}

static void recite_obs(const reciteKnobs *k, uint32_t symbols, uint32_t phase, float *obs) {
    uint32_t watching = phase < k->length;
    float4 onehot = flag4((lane4)((watching ? recite_symbol(k, symbols, phase) : RECITE_SYMBOLS) == (lane4){ 0u, 1u, 2u, 3u }));
    __builtin_memcpy(obs, &onehot, sizeof onehot);
    obs[RECITE_SYMBOLS] = watching ? 0.0f : 1.0f;
    obs[RECITE_SYMBOLS + 1u] = (float)(phase % k->length) / (float)k->length;
    obs[RECITE_SYMBOLS + 2u] = 1.0f;
}

static inline __attribute__((always_inline)) float recite_advance(const reciteKnobs *k, uint32_t seed, uint32_t fill, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done) {
    uint32_t sequence = words[0], phase = words[1], symbols = extra[1];
    float reward = phase >= k->length && (uint32_t)ctrl[0] % RECITE_SYMBOLS == recite_symbol(k, symbols, phase) ? 1.0f : 0.0f;
    uint32_t next = phase + 1u >= 2u * k->length;
    phase = next ? 0u : phase + 1u;
    sequence += next;
    *done = next;
    words[0] = sequence; words[1] = phase;
    if (next) { symbols = extra[2]; extra[0] = sequence; extra[1] = symbols; }
    if (fill) extra[2] = (phase ? extra[2] : 0u) | recite_part(seed, sequence + 1u, phase);
    recite_obs(k, symbols, phase, obs);
    return reward;
}

static inline __attribute__((always_inline)) float recite_play(uint32_t instance, uint32_t seed, uint32_t fill, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done) {
    return instance == 0u ? recite_advance(&RECITE_KNOBS[0], seed, fill, ctrl, words, extra, obs, done) : recite_advance(&RECITE_KNOBS[1], seed, fill, ctrl, words, extra, obs, done);
}

static __attribute__((noinline)) float recite_refill(uint32_t instance, uint32_t seed, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done) {
    return recite_play(instance, seed, 1u, ctrl, words, extra, obs, done);
}

void recite_reset(uint32_t instance, uint32_t seed, uint32_t *words, uint32_t *extra, uint32_t steps) {
    (void)steps; (void)words;
    const reciteKnobs *k = &RECITE_KNOBS[instance];
    extra[0] = 0u;
    extra[1] = recite_symbols(k, seed, 0u);
    extra[2] = recite_symbols(k, seed, 1u);
}

float recite_step(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done, const float *shared) {
    (void)shared;
    (void)step;
    recite_check(words, extra);
    uint32_t length = RECITE_KNOBS[instance].length, phase = words[1] + 1u;
    if (phase >= 2u * length || phase < length / 4u) return recite_refill(instance, seed, ctrl, words, extra, obs, done);
    return recite_play(instance, seed, 0u, ctrl, words, extra, obs, done);
}

void recite_observe(uint32_t instance, uint32_t seed, const uint32_t *words, const uint32_t *extra, uint32_t step, float *obs, const float *shared) {
    (void)shared;
    (void)seed; (void)step;
    recite_check(words, extra);
    recite_obs(&RECITE_KNOBS[instance], extra[1], words[1], obs);
}

void recite_oracle(uint32_t instance, uint32_t seed, uint32_t step, const uint32_t *words, uint32_t *extra, const float *obs, float *ctrl, const float *shared) {
    (void)shared;
    (void)seed; (void)step; (void)obs;
    const reciteKnobs *k = &RECITE_KNOBS[instance];
    recite_check(words, extra);
    ctrl[0] = words[1] >= k->length ? (float)recite_symbol(k, extra[1], words[1]) : 0.0f;
}

uint32_t recite_view(uint32_t instance, const uint32_t *words, const uint32_t *extra, const float *shared, uint32_t *out) {
    (void)instance; (void)words; (void)shared;
    if (!out) return 1u;
    out[0] = extra[1];
    return 1u;
}
