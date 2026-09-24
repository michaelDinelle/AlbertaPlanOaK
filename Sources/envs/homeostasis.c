#include "../env.h"

#define HOMEOSTASIS_SIDE 12u
#define HOMEOSTASIS_PER_TYPE 6u
#define HOMEOSTASIS_SCALE 20
#define HOMEOSTASIS_START 10
#define HOMEOSTASIS_FULL 10
#define HOMEOSTASIS_LOW 5
#define HOMEOSTASIS_HIGH 25
#define HOMEOSTASIS_LIKED 1.0f
#define HOMEOSTASIS_CRAVED 5.0f
#define HOMEOSTASIS_SPURNED -1.0f

typedef struct { uint32_t hidden; } homeostasisKnobs;

static const homeostasisKnobs HOMEOSTASIS_KNOBS[2] = {
    { 0u },
    { 1u }
};

enum { HOMEOSTASIS_TYPES = 3, HOMEOSTASIS_GROUP = 8, HOMEOSTASIS_LEVELS = 8, HOMEOSTASIS_AGENT = 12, HOMEOSTASIS_EMPTY = 0xFF };

static const uint8_t HOMEOSTASIS_WRAP[32] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7 };

static int homeostasis_taken(const void *items, int x, int y) {
    const uint8_t *code = (const uint8_t *)items;
    uint8x16_t cell = vdupq_n_u8((uint8_t)((x - 1) | (y - 1) << 4));
    return vmaxvq_u8(vorrq_u8(vceqq_u8(vld1q_u8(code), cell), vceqq_u8(vld1q_u8(code + 16), cell))) != 0u;
}

static uint8_t homeostasis_place(uint32_t seed, uint32_t index, uint32_t block, const uint32_t *words) {
    uint32_t at = words[HOMEOSTASIS_AGENT];
    int x, y;
    sampleFreeCell(seed, index, block, homeostasis_taken, words, HOMEOSTASIS_SIDE, (int)(at & 15u) + 1, (int)(at >> 4) + 1, &x, &y);
    return (uint8_t)((x - 1) | (y - 1) << 4);
}

void homeostasis_reset(uint32_t instance, uint32_t seed, uint32_t *words, uint32_t *extra, uint32_t steps) {
    (void)instance; (void)extra; (void)steps;
    uint8_t *items = (uint8_t *)words;
    __builtin_memset(items, HOMEOSTASIS_EMPTY, HOMEOSTASIS_LEVELS * sizeof(uint32_t));
    words[HOMEOSTASIS_AGENT] = HOMEOSTASIS_SIDE / 2u | HOMEOSTASIS_SIDE / 2u << 4;
    for (uint32_t slot = 0u; slot < HOMEOSTASIS_TYPES * HOMEOSTASIS_GROUP; slot++)
        if (slot % HOMEOSTASIS_GROUP < HOMEOSTASIS_PER_TYPE) items[slot] = homeostasis_place(seed, 0u, slot << 16, words);
    int64_t level[2] = { HOMEOSTASIS_START * HOMEOSTASIS_SCALE, HOMEOSTASIS_START * HOMEOSTASIS_SCALE };
    __builtin_memcpy(words + HOMEOSTASIS_LEVELS, level, sizeof level);
}

static inline __attribute__((always_inline)) void homeostasis_obs(const homeostasisKnobs *k, const uint32_t *words, float *obs) {
    static const float REACH[HOMEOSTASIS_SIDE] = { -6.0f / 6.0f, -5.0f / 6.0f, -4.0f / 6.0f, -3.0f / 6.0f, -2.0f / 6.0f, -1.0f / 6.0f, 0.0f, 1.0f / 6.0f, 2.0f / 6.0f, 3.0f / 6.0f, 4.0f / 6.0f, 5.0f / 6.0f };
    const uint8_t *code = (const uint8_t *)words;
    uint32_t at = words[HOMEOSTASIS_AGENT];
    uint8x16x2_t wrap = vld1q_u8_x2(HOMEOSTASIS_WRAP);
    uint8x16_t across = vdupq_n_u8((uint8_t)(HOMEOSTASIS_SIDE + HOMEOSTASIS_SIDE / 2u - (at & 15u))), down = vdupq_n_u8((uint8_t)(HOMEOSTASIS_SIDE + HOMEOSTASIS_SIDE / 2u - (at >> 4)));
    uint8x16_t centre = vdupq_n_u8(HOMEOSTASIS_SIDE / 2u), corner = vdupq_n_u8(HOMEOSTASIS_SIDE / 2u - 1u), edge = vdupq_n_u8(2u), outside = vdupq_n_u8(16u);
    uint32_t nearest[4], seen[4];
    for (uint32_t half = 0u; half < 2u; half++) {
        uint8x16_t codes = vld1q_u8(code + 16u * half), empty = vceqq_u8(codes, vdupq_n_u8(HOMEOSTASIS_EMPTY));
        uint8x16_t ex = vqtbl2q_u8(wrap, vaddq_u8(vandq_u8(codes, vdupq_n_u8(15u)), across)), ey = vqtbl2q_u8(wrap, vaddq_u8(vshrq_n_u8(codes, 4), down));
        uint8x16_t spread = vorrq_u8(vaddq_u8(vabdq_u8(ex, centre), vabdq_u8(ey, centre)), empty), place = vsliq_n_u8(ex, ey, 4);
        nearest[2u * half] = vminvq_u16(vreinterpretq_u16_u8(vzip1q_u8(place, spread)));
        nearest[2u * half + 1u] = vminvq_u16(vreinterpretq_u16_u8(vzip2q_u8(place, spread)));
        uint8x16_t column = vsubq_u8(ex, corner), row = vsubq_u8(ey, corner);
        uint8x16_t near = vbicq_u8(vandq_u8(vcleq_u8(column, edge), vcleq_u8(row, edge)), empty);
        uint8x16_t cell = vbslq_u8(near, vmlaq_u8(column, row, vdupq_n_u8(3u)), outside);
        seen[2u * half] = vaddvq_u16(vshlq_u16(vdupq_n_u16(1u), vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(cell)))));
        seen[2u * half + 1u] = vaddvq_u16(vshlq_u16(vdupq_n_u16(1u), vreinterpretq_s16_u16(vmovl_high_u8(cell))));
    }
    uint32_t low = seen[0] | seen[2], high = seen[1] | seen[2];
    lane4 cells = { 1u, 2u, 4u, 8u };
    float4 front = (flag4(low & cells) + 2.0f * flag4(high & cells)) * (1.0f / 3.0f);
    float4 back = (flag4(low & cells << 4u) + 2.0f * flag4(high & cells << 4u)) * (1.0f / 3.0f);
    __builtin_memcpy(obs, &front, sizeof front);
    __builtin_memcpy(obs + 4, &back, sizeof back);
    obs[8] = (float)((low >> 8 & 1u) + 2u * (high >> 8 & 1u)) * (1.0f / 3.0f);
    float *toward = obs + 9;
    if (!k->hidden) {
        int64_t level[2];
        __builtin_memcpy(level, words + HOMEOSTASIS_LEVELS, sizeof level);
        obs[9] = (float)level[0] / (float)(HOMEOSTASIS_START * HOMEOSTASIS_SCALE);
        obs[10] = (float)level[1] / (float)(HOMEOSTASIS_START * HOMEOSTASIS_SCALE);
        toward = obs + 11;
    }
    for (uint32_t type = 0u; type < HOMEOSTASIS_TYPES; type++) {
        toward[2u * type] = REACH[nearest[type] & 15u];
        toward[2u * type + 1u] = REACH[nearest[type] >> 4 & 15u];
    }
}

static float homeostasis_taste(uint32_t first, uint32_t second, const int64_t *level) {
    float liked = level[0] <= HOMEOSTASIS_FULL * HOMEOSTASIS_SCALE ? HOMEOSTASIS_LIKED : HOMEOSTASIS_SPURNED;
    float craved = level[1] > HOMEOSTASIS_LOW * HOMEOSTASIS_SCALE && level[1] < HOMEOSTASIS_HIGH * HOMEOSTASIS_SCALE ? HOMEOSTASIS_CRAVED : HOMEOSTASIS_SPURNED;
    return (first ? liked : 0.0f) + (second ? craved : 0.0f);
}

static inline __attribute__((always_inline)) uint32_t homeostasis_found(const uint32_t *words, uint32_t at) {
    const uint8_t *code = (const uint8_t *)words;
    uint8x16_t cell = vdupq_n_u8((uint8_t)at);
    uint8x16_t low = vandq_u8(vceqq_u8(vld1q_u8(code), cell), (uint8x16_t){ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 });
    uint8x16_t high = vandq_u8(vceqq_u8(vld1q_u8(code + 16), cell), (uint8x16_t){ 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32 });
    return vmaxvq_u8(vorrq_u8(low, high));
}

void homeostasis_observe(uint32_t instance, uint32_t seed, const uint32_t *words, const uint32_t *extra, uint32_t step, float *obs, const float *shared) {
    (void)seed; (void)extra; (void)step; (void)shared;
    switch (instance) {
        case 0u: homeostasis_obs(&HOMEOSTASIS_KNOBS[0], words, obs); return;
        case 1u: homeostasis_obs(&HOMEOSTASIS_KNOBS[1], words, obs); return;
        default: __builtin_trap();
    }
}

static __attribute__((noinline)) float homeostasis_eat(uint32_t instance, uint32_t seed, uint32_t step, uint32_t slot, uint32_t *words, float *obs, float reward) {
    ((uint8_t *)words)[slot] = homeostasis_place(seed, step, 0u, words);
    homeostasis_observe(instance, seed, words, 0, step, obs, 0);
    return reward;
}

static inline __attribute__((always_inline)) float homeostasis_live(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, float *obs, uint32_t *done) {
    float pressed = ctrl[0];
    int dx, dy;
    gridDelta(pressed >= 0.0f && pressed < 4.0f ? (uint32_t)pressed : 0u, &dx, &dy);
    uint32_t at = words[HOMEOSTASIS_AGENT];
    at = HOMEOSTASIS_WRAP[(at & 15u) + HOMEOSTASIS_SIDE + (uint32_t)dx] | (uint32_t)HOMEOSTASIS_WRAP[(at >> 4) + HOMEOSTASIS_SIDE + (uint32_t)dy] << 4;
    words[HOMEOSTASIS_AGENT] = at;
    uint32_t slot = homeostasis_found(words, at), type = (slot + HOMEOSTASIS_GROUP - 1u) / HOMEOSTASIS_GROUP, first = type & 1u, second = type >> 1;
    int64_t level[2];
    __builtin_memcpy(level, words + HOMEOSTASIS_LEVELS, sizeof level);
    level[0] += (int64_t)first * HOMEOSTASIS_SCALE - 1;
    level[1] += (int64_t)second * HOMEOSTASIS_SCALE - 1;
    __builtin_memcpy(words + HOMEOSTASIS_LEVELS, level, sizeof level);
    float reward = homeostasis_taste(first, second, level);
    *done = 0u;
    if (__builtin_unpredictable(slot != 0u)) return homeostasis_eat(instance, seed, step, slot - 1u, words, obs, reward);
    homeostasis_obs(&HOMEOSTASIS_KNOBS[instance], words, obs);
    return reward;
}

float homeostasis_step(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done, const float *shared) {
    (void)extra; (void)shared;
    switch (instance) {
        case 0u: return homeostasis_live(0u, seed, step, ctrl, words, obs, done);
        case 1u: return homeostasis_live(1u, seed, step, ctrl, words, obs, done);
        default: __builtin_trap();
    }
}

static int homeostasis_offset(uint32_t to, uint32_t from) { return (int)HOMEOSTASIS_WRAP[to + HOMEOSTASIS_SIDE + HOMEOSTASIS_SIDE / 2u - from] - (int)(HOMEOSTASIS_SIDE / 2u); }

static int homeostasis_span(int delta) {
    int span = abs(delta);
    return span < (int)HOMEOSTASIS_SIDE - span ? span : (int)HOMEOSTASIS_SIDE - span;
}

static uint32_t homeostasis_heading(int dx, int dy) { return dy < 0 ? 0u : dy > 0 ? 1u : dx < 0 ? 2u : 3u; }

void homeostasis_oracle(uint32_t instance, uint32_t seed, uint32_t step, const uint32_t *words, uint32_t *extra, const float *obs, float *ctrl, const float *shared) {
    (void)instance; (void)seed; (void)step; (void)extra; (void)obs; (void)shared;
    const uint8_t *code = (const uint8_t *)words;
    uint32_t at = words[HOMEOSTASIS_AGENT];
    int64_t level[2];
    __builtin_memcpy(level, words + HOMEOSTASIS_LEVELS, sizeof level);
    int64_t firstReady = level[0] - (HOMEOSTASIS_FULL - 1) * HOMEOSTASIS_SCALE, secondReady = level[1] - (HOMEOSTASIS_HIGH - 1) * HOMEOSTASIS_SCALE + 1;
    int64_t bestValue = 0, bestWhen = 1;
    int bestX = 0, bestY = 0, bestSpan = 0;
    uint32_t forbidden = 0u;
    float bite[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    for (uint32_t slot = 0u; slot < HOMEOSTASIS_TYPES * HOMEOSTASIS_GROUP; slot++) {
        if (slot % HOMEOSTASIS_GROUP >= HOMEOSTASIS_PER_TYPE) continue;
        uint32_t type = slot / HOMEOSTASIS_GROUP + 1u, first = type & 1u, second = type >> 1;
        int dx = homeostasis_offset(code[slot] & 15u, at & 15u), dy = homeostasis_offset(code[slot] >> 4, at >> 4), span = abs(dx) + abs(dy);
        int64_t when = span;
        if (first && firstReady > when) when = firstReady;
        if (second && secondReady > when) when = secondReady;
        when += (when - span) & 1;
        int64_t value = (int64_t)first * (int64_t)HOMEOSTASIS_LIKED + (int64_t)second * (int64_t)HOMEOSTASIS_CRAVED;
        if (value * bestWhen > bestValue * when || (value * bestWhen == bestValue * when && span < bestSpan)) {
            bestValue = value; bestWhen = when; bestX = dx; bestY = dy; bestSpan = span;
        }
        if (span == 1) {
            uint32_t heading = homeostasis_heading(dx, dy);
            int64_t after[2] = { level[0] - 1 + (int64_t)first * HOMEOSTASIS_SCALE, level[1] - 1 + (int64_t)second * HOMEOSTASIS_SCALE };
            bite[heading] = homeostasis_taste(first, second, after);
            forbidden |= (uint32_t)(when != 1) << heading;
        }
    }
    int want = bestSpan >= 2 || bestWhen == 1 ? bestSpan - 1 : bestSpan + 1, fallbackSpan = 4 * (int)HOMEOSTASIS_SIDE;
    uint32_t choice = 4u, fallback = 4u;
    for (uint32_t heading = 0u; heading < 4u && choice == 4u; heading++) {
        if (forbidden >> heading & 1u) continue;
        int mx, my;
        gridDelta(heading, &mx, &my);
        int span = homeostasis_span(bestX - mx) + homeostasis_span(bestY - my);
        if (span == want) choice = heading;
        else if (span < fallbackSpan) { fallback = heading; fallbackSpan = span; }
    }
    if (choice == 4u) choice = fallback;
    if (choice == 4u) {
        choice = 0u;
        for (uint32_t heading = 1u; heading < 4u; heading++) choice = bite[heading] > bite[choice] ? heading : choice;
    }
    ctrl[0] = (float)choice;
}
