#include "../env.h"

typedef struct { uint32_t inputs, signs, flipping, hidden, period; float scale, beta, drift; } supervisedKnobs;

static const supervisedKnobs SUPERVISED_KNOBS[3] = {
    { 20u, 5u, 0u, 0u, 20u, 16.0f, 0.0f, 0.0f },
    { 20u, 0u, 15u, 100u, 10000u, 24.0f, 0.7f, 0.0f },
    { 1u, 0u, 0u, 0u, 50000u, 5.0f, 0.0f, 3.0f }
};

enum { SUPERVISED_SOURCE, SUPERVISED_TARGET, SUPERVISED_SHOWN };
enum { RATE_LATENT = SUPERVISED_SOURCE, RATE_SIGMA = SUPERVISED_TARGET };
enum { SLOW_LANES = 112, SLOW_FAST = 0, SLOW_SLACK = SLOW_FAST + SLOW_LANES / 4, SLOW_SIGN = SLOW_SLACK + SLOW_LANES / 4, SLOW_UNIT = SLOW_SIGN + SLOW_LANES / 4,
       SLOW_BIAS = SLOW_UNIT + SLOW_LANES, SLOW_NEXT, SLOW_DRAW = SLOW_NEXT + ENV_BATCH, SLOW_TAG = SLOW_DRAW + ENV_BATCH, SLOW_EXTRA, SLOW_VIEW = 8 };
enum { RATE_BLOCKS = 0, RATE_DRAWS = RATE_BLOCKS + 4 * ENV_BATCH, RATE_TAG = RATE_DRAWS + 2 * ENV_BATCH, RATE_EXTRA };

static inline __attribute__((always_inline)) float supervised_prediction(const supervisedKnobs *k, const float *ctrl) { return k->scale * oak_clamp(ctrl[0], -1.0f, 1.0f); }

static inline __attribute__((always_inline)) uint32_t idbd_draw(uint32_t seed, uint32_t step, float *obs) {
    u44 a = philox4(seed, ENV_RNG_STREAM, (lane4)step, (lane4){ 0u, 1u, 2u, 3u }, (lane4)0u, (lane4)0u);
    u44 b = philox4(seed, ENV_RNG_STREAM, (lane4)step, (lane4){ 4u, 5u, 6u, 7u }, (lane4)0u, (lane4)0u);
    u44 c = philox4(seed, ENV_RNG_STREAM, (lane4)step, (lane4){ 8u, 9u, 10u, 11u }, (lane4)0u, (lane4)0u);
    float4 x[5] = { normal4(a.x, a.y), normal4(a.z, a.w), normal4(b.x, b.y), normal4(b.z, b.w), normal4(c.x, c.y) };
    __builtin_memcpy(obs, x, sizeof x);
    return c.z.x;
}

static inline __attribute__((always_inline)) float idbd_target(const supervisedKnobs *k, uint32_t signs, const float *obs) {
    float target = 0.0f;
    for (uint32_t i = 0u; i < k->signs; i++) target += wordFloat(floatWord(obs[i]) ^ (signs >> i & 1u) << 31);
    return target;
}

static __attribute__((noinline)) float idbd_advance(const supervisedKnobs *k, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, float *obs) {
    float miss = supervised_prediction(k, ctrl) - wordFloat(words[SUPERVISED_TARGET]);
    uint32_t spare = idbd_draw(seed, step, obs);
    if (step % k->period == 0u) words[SUPERVISED_SOURCE] ^= 1u << (spare % k->signs);
    words[SUPERVISED_SHOWN] = words[SUPERVISED_TARGET];
    words[SUPERVISED_TARGET] = floatWord(idbd_target(k, words[SUPERVISED_SOURCE], obs));
    obs[k->inputs] = wordFloat(words[SUPERVISED_SHOWN]);
    return -(miss * miss);
}

static int slow_tolerance(const supervisedKnobs *k, uint32_t positive, int bias) {
    int inputs = (int)k->inputs, negatives = inputs - oak_popcount(positive), weightSum = inputs - 2 * negatives, misses = 0;
    float threshold = k->beta * (float)(inputs + 1) - 0.5f * (float)(inputs - weightSum + bias);
    while (misses <= inputs && (float)(inputs - misses - negatives + bias) > threshold) misses++;
    return misses - 1;
}

static void slow_slack(const supervisedKnobs *k, uint32_t *extra, uint32_t source) {
    int8_t *slack = (int8_t *)(extra + SLOW_SLACK);
    for (uint32_t j = 0u; j < k->hidden; j++) {
        uint32_t unit = extra[SLOW_UNIT + j];
        slack[j] = (int8_t)((int)(unit >> 24) - oak_popcount((source ^ unit) & ((1u << k->flipping) - 1u)));
    }
}

static inline __attribute__((always_inline)) uint8x16_t slow_fired(const uint32_t *extra, uint32_t fast, uint32_t group) {
    uint8x16_t misses = vcntq_u8(veorq_u8(vdupq_n_u8((uint8_t)fast), vld1q_u8((const uint8_t *)(extra + SLOW_FAST) + 16u * group)));
    return vcleq_s8((int8x16_t)misses, vld1q_s8((const int8_t *)(extra + SLOW_SLACK) + 16u * group));
}

static inline __attribute__((always_inline)) float slow_target(const uint32_t *extra, uint32_t fast) {
    int8x16_t sum = vdupq_n_s8(0);
    for (uint32_t group = 0u; group < SLOW_LANES / 16u; group++)
        sum = vaddq_s8(sum, vandq_s8((int8x16_t)slow_fired(extra, fast, group), vld1q_s8((const int8_t *)(extra + SLOW_SIGN) + 16u * group)));
    return wordFloat(extra[SLOW_BIAS]) + (float)vaddlvq_s8(sum);
}

static void slow_obs(const supervisedKnobs *k, uint32_t source, float *obs) {
    for (uint32_t i = 0u; i < k->inputs; i += 4u) {
        float4 row = flag4(source & ((lane4){ 1u, 2u, 4u, 8u } << i));
        __builtin_memcpy(obs + i, &row, sizeof row);
    }
}

static void slow_block(uint32_t seed, uint32_t first, uint32_t *draws) {
    u4 r = philox4x32(seed, ENV_RNG_STREAM, first, 0u, 0u, 0u);
    uint32_t block[ENV_BATCH] = { r.x, r.y, r.z, r.w };
    __builtin_memcpy(draws, block, sizeof block);
}

static inline __attribute__((always_inline)) float slow_advance(const supervisedKnobs *k, const float *ctrl, uint32_t *words, const uint32_t *extra, uint32_t draw, float *obs) {
    float miss = supervised_prediction(k, ctrl) - wordFloat(words[SUPERVISED_TARGET]);
    uint32_t fast = draw >> (32u - (k->inputs - k->flipping)), source = (words[SUPERVISED_SOURCE] & ((1u << k->flipping) - 1u)) | fast << k->flipping;
    words[SUPERVISED_SHOWN] = words[SUPERVISED_TARGET];
    words[SUPERVISED_TARGET] = floatWord(slow_target(extra, fast));
    words[SUPERVISED_SOURCE] = source;
    slow_obs(k, source, obs);
    obs[k->inputs] = wordFloat(words[SUPERVISED_SHOWN]);
    return -(miss * miss);
}

static __attribute__((noinline)) float slow_rare(const supervisedKnobs *k, uint32_t seed, uint32_t step, uint32_t slot, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs) {
    if (slot == ENV_BATCH) {
        __builtin_memcpy(extra + SLOW_DRAW, extra + SLOW_NEXT, ENV_BATCH * sizeof(uint32_t));
        slow_block(seed, step + ENV_BATCH, extra + SLOW_NEXT);
        slot = 0u;
    }
    if (step % k->period == 0u) {
        words[SUPERVISED_SOURCE] ^= 1u << (philox4x32(seed, ENV_RNG_STREAM, step, 1u, 0u, 0u).x % k->flipping);
        slow_slack(k, extra, words[SUPERVISED_SOURCE]);
    }
    return slow_advance(k, ctrl, words, extra, extra[SLOW_DRAW + slot], obs);
}

static inline __attribute__((always_inline)) float slow_play(const supervisedKnobs *k, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs) {
    uint32_t slot = batchSlot(&extra[SLOW_TAG], step);
    if (slot == ENV_BATCH || step % k->period == 0u) return slow_rare(k, seed, step, slot, ctrl, words, extra, obs);
    return slow_advance(k, ctrl, words, extra, extra[SLOW_DRAW + slot], obs);
}

static void rate_block(uint32_t seed, uint32_t first, uint32_t *blocks) {
    u44 r = philox4(seed, ENV_RNG_STREAM, first + (lane4){ 0u, 1u, 2u, 3u }, 0u, 0u, 0u);
    __builtin_memcpy(blocks, &r, sizeof r);
}

static void rate_draw(const uint32_t *blocks, uint32_t *draws) {
    u44 r;
    __builtin_memcpy(&r, blocks, sizeof r);
    float4 drift = normal4(r.x, r.y), noise = normal4(r.z, r.w);
    __builtin_memcpy(draws, &drift, sizeof drift);
    __builtin_memcpy(draws + ENV_BATCH, &noise, sizeof noise);
}

static inline __attribute__((always_inline)) float rate_advance(const supervisedKnobs *k, const float *ctrl, uint32_t *words, const uint32_t *draws, float *obs) {
    float latent = wordFloat(words[RATE_LATENT]) + wordFloat(words[RATE_SIGMA]) * wordFloat(draws[0]);
    float target = latent + wordFloat(draws[ENV_BATCH]), miss = wordFloat(words[SUPERVISED_SHOWN]) + supervised_prediction(k, ctrl) - target;
    words[RATE_LATENT] = floatWord(latent);
    words[SUPERVISED_SHOWN] = floatWord(target);
    obs[0] = 1.0f;
    obs[k->inputs] = target;
    return -(miss * miss);
}

static __attribute__((noinline)) float rate_rare(const supervisedKnobs *k, uint32_t seed, uint32_t step, uint32_t slot, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs) {
    if (slot == ENV_BATCH) {
        rate_draw(extra + RATE_BLOCKS, extra + RATE_DRAWS);
        rate_block(seed, step + ENV_BATCH, extra + RATE_BLOCKS);
        slot = 0u;
    }
    float reward = rate_advance(k, ctrl, words, extra + RATE_DRAWS + slot, obs);
    if (step % k->period == 0u) words[RATE_SIGMA] = floatWord(k->drift * u01(philox4x32(seed, ENV_RNG_STREAM, step, 1u, 0u, 0u).x));
    return reward;
}

static inline __attribute__((always_inline)) float rate_play(const supervisedKnobs *k, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs) {
    uint32_t slot = batchSlot(&extra[RATE_TAG], step);
    if (slot == ENV_BATCH || step % k->period == 0u) return rate_rare(k, seed, step, slot, ctrl, words, extra, obs);
    return rate_advance(k, ctrl, words, extra + RATE_DRAWS + slot, obs);
}

uint32_t supervised_extraWords(uint32_t instance, uint32_t steps) {
    (void)steps;
    switch (instance) {
        case 0u: return 0u;
        case 1u: return SLOW_EXTRA;
        default: return RATE_EXTRA;
    }
}

void supervised_reset(uint32_t instance, uint32_t seed, uint32_t *words, uint32_t *extra, uint32_t steps) {
    (void)steps;
    const supervisedKnobs *k = &SUPERVISED_KNOBS[instance];
    u4 start = philox4x32(seed, RNG_STREAM_RESET, 0u, 0u, 0u, 0u);
    float obs[OAK_ENV_OBS_MAX];
    if (instance == 0u) {
        words[SUPERVISED_SOURCE] = start.x & ((1u << k->signs) - 1u);
        idbd_draw(seed, 0u, obs);
        words[SUPERVISED_TARGET] = floatWord(idbd_target(k, words[SUPERVISED_SOURCE], obs));
        words[SUPERVISED_SHOWN] = floatWord(0.0f);
        return;
    }
    if (instance == 1u) {
        if (k->hidden > SLOW_LANES) __builtin_trap();
        uint8_t *fast = (uint8_t *)(extra + SLOW_FAST);
        int8_t *sign = (int8_t *)(extra + SLOW_SIGN);
        for (uint32_t j = 0u; j < SLOW_LANES; j++) {
            uint32_t draw = philox4x32(seed, RNG_STREAM_RESET, j, 1u, 0u, 0u).x, positive = draw & ((1u << k->inputs) - 1u);
            int live = j < k->hidden;
            fast[j] = live ? (uint8_t)(positive >> k->flipping) : 0u;
            sign[j] = live ? (draw >> (k->inputs + 1u) & 1u ? 1 : -1) : 0;
            extra[SLOW_UNIT + j] = live ? positive | (uint32_t)slow_tolerance(k, positive, draw >> k->inputs & 1u ? 1 : -1) << 24 : 0u;
        }
        extra[SLOW_BIAS] = floatWord(start.y & 1u ? 1.0f : -1.0f);
        uint32_t source = start.x & ((1u << k->flipping) - 1u), fastBits = philox4x32(seed, ENV_RNG_STREAM, 0u, 0u, 0u, 0u).x >> (32u - (k->inputs - k->flipping));
        slow_slack(k, extra, source);
        words[SUPERVISED_SOURCE] = source | fastBits << k->flipping;
        words[SUPERVISED_TARGET] = floatWord(slow_target(extra, fastBits));
        words[SUPERVISED_SHOWN] = floatWord(0.0f);
        batchReset(&extra[SLOW_TAG]);
        slow_block(seed, extra[SLOW_TAG] + ENV_BATCH, extra + SLOW_NEXT);
        return;
    }
    uint32_t blocks[4 * ENV_BATCH], draws[2 * ENV_BATCH];
    rate_block(seed, 0u, blocks);
    rate_draw(blocks, draws);
    words[RATE_LATENT] = floatWord(0.0f);
    words[RATE_SIGMA] = floatWord(k->drift * u01(start.x));
    words[SUPERVISED_SHOWN] = floatWord(wordFloat(words[RATE_LATENT]) + wordFloat(draws[ENV_BATCH]));
    batchReset(&extra[RATE_TAG]);
    rate_block(seed, extra[RATE_TAG] + ENV_BATCH, extra + RATE_BLOCKS);
}

float supervised_step(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done, const float *shared) {
    (void)shared;
    *done = 0u;
    switch (instance) {
        case 0u: return idbd_advance(&SUPERVISED_KNOBS[0], seed, step, ctrl, words, obs);
        case 1u: return slow_play(&SUPERVISED_KNOBS[1], seed, step, ctrl, words, extra, obs);
        default: return rate_play(&SUPERVISED_KNOBS[2], seed, step, ctrl, words, extra, obs);
    }
}

void supervised_observe(uint32_t instance, uint32_t seed, const uint32_t *words, const uint32_t *extra, uint32_t step, float *obs, const float *shared) {
    (void)shared; (void)extra;
    const supervisedKnobs *k = &SUPERVISED_KNOBS[instance];
    if (instance == 0u) idbd_draw(seed, step, obs);
    else if (instance == 1u) slow_obs(k, words[SUPERVISED_SOURCE], obs);
    else obs[0] = 1.0f;
    obs[k->inputs] = wordFloat(words[SUPERVISED_SHOWN]);
}

uint32_t supervised_view(uint32_t instance, const uint32_t *words, const uint32_t *extra, const float *shared, uint32_t *out) {
    (void)shared;
    if (instance != 1u) return 0u;
    if (!out) return SLOW_VIEW;
    const supervisedKnobs *k = &SUPERVISED_KNOBS[instance];
    const int8_t *sign = (const int8_t *)(extra + SLOW_SIGN);
    uint8_t fired[SLOW_LANES];
    uint16_t field[2u * SLOW_VIEW] = { 0u };
    for (uint32_t group = 0u; group < SLOW_LANES / 16u; group++) vst1q_u8(fired + 16u * group, slow_fired(extra, words[SUPERVISED_SOURCE] >> k->flipping, group));
    for (uint32_t j = 0u; j < k->hidden; j++) {
        field[j / 16u] |= (uint16_t)((fired[j] & 1u) << (j % 16u));
        field[SLOW_LANES / 16u + j / 16u] |= (uint16_t)((sign[j] > 0) << (j % 16u));
    }
    field[2u * (SLOW_LANES / 16u)] = wordFloat(extra[SLOW_BIAS]) > 0.0f;
    for (uint32_t i = 0u; i < SLOW_VIEW; i++) out[i] = field[2u * i] | (uint32_t)field[2u * i + 1u] << 16;
    return SLOW_VIEW;
}

void supervised_oracle(uint32_t instance, uint32_t seed, uint32_t step, const uint32_t *words, uint32_t *extra, const float *obs, float *ctrl, const float *shared) {
    (void)seed; (void)step; (void)extra; (void)obs; (void)shared;
    const supervisedKnobs *k = &SUPERVISED_KNOBS[instance];
    float aim = instance == 2u ? wordFloat(words[RATE_LATENT]) - wordFloat(words[SUPERVISED_SHOWN]) : wordFloat(words[SUPERVISED_TARGET]);
    ctrl[0] = aim / k->scale;
}
