#include "../env.h"

typedef struct { uint32_t ticks, dayLength, zombieHealth, cowHealth, skeletonHealth, despawn; float cowChance, zombieChance, zombieNightChance, skeletonChance, saplingChance; } craftaxKnobs;

static const craftaxKnobs CRAFTAX_KNOBS[1] = {
    { 10000u, 300u, 5u, 3u, 3u, 14u, 0.1f, 0.02f, 0.1f, 0.05f, 0.1f }
};

enum { CXB_INVALID, CXB_OUT, CXB_GRASS, CXB_WATER, CXB_STONE, CXB_TREE, CXB_WOOD, CXB_PATH, CXB_COAL, CXB_IRON, CXB_DIAMOND, CXB_TABLE, CXB_FURNACE, CXB_SAND, CXB_LAVA, CXB_PLANT, CXB_RIPE,
       CXB_TREECAND = 32, CXB_STONELO, CXB_STONEHI };
enum { CXA_NOOP, CXA_LEFT, CXA_RIGHT, CXA_UP, CXA_DOWN, CXA_DO, CXA_SLEEP, CXA_PLACE_STONE, CXA_PLACE_TABLE, CXA_PLACE_FURNACE, CXA_PLACE_PLANT,
       CXA_WOOD_PICKAXE, CXA_STONE_PICKAXE, CXA_IRON_PICKAXE, CXA_WOOD_SWORD, CXA_STONE_SWORD, CXA_IRON_SWORD };
enum { CXG_WOOD, CXG_TABLE, CXG_COW, CXG_SAPLING, CXG_DRINK, CXG_WOOD_PICKAXE, CXG_WOOD_SWORD, CXG_PLANT, CXG_ZOMBIE, CXG_STONE, CXG_PLACE_STONE,
       CXG_EAT_PLANT, CXG_SKELETON, CXG_STONE_PICKAXE, CXG_STONE_SWORD, CXG_WAKE, CXG_FURNACE, CXG_COAL, CXG_IRON, CXG_DIAMOND, CXG_IRON_PICKAXE, CXG_IRON_SWORD };
enum { CXI_WOOD, CXI_STONE, CXI_COAL, CXI_IRON, CXI_DIAMOND, CXI_SAPLING, CXI_WOOD_PICKAXE, CXI_STONE_PICKAXE, CXI_IRON_PICKAXE, CXI_WOOD_SWORD, CXI_STONE_SWORD,
       CXI_IRON_SWORD, CXI_COUNT };
enum { CXW_PLAYER, CXW_VITALS, CXW_HUNGER, CXW_THIRST, CXW_FATIGUE, CXW_RECOVER, CXW_INV, CXW_ACH = CXW_INV + 3, CXW_TIME, CXW_GAME, CXW_MENU, CXW_GEN,
       CXW_DEATHS, CXW_ZOMBIE, CXW_COW = CXW_ZOMBIE + 3, CXW_SKELETON = CXW_COW + 3, CXW_ARROW = CXW_SKELETON + 2, CXW_COUNT = CXW_ARROW + 3 };
enum { CX_SIDE = 64, CX_CELLS = 4096, CX_NONE = 4096, CX_START = 32 * 64 + 32, CX_MENU = 13, CX_SLOTS = 50, CX_REACH = 16, CX_PATCH = 5 };
enum { CXV_PATCH = 0, CXV_VITALS = CX_PATCH * CX_PATCH, CXV_LIGHT = CXV_VITALS + 4, CXV_FACING, CXV_SLEEP = CXV_FACING + 2, CXV_MENU, CXV_COUNT = CXV_MENU + 4 };
enum { CX_GRADS = 640, CX_ROWS_PER_FIELD_SLICE = 32, CX_ROWS_PER_CLASS_SLICE = 8, CX_STAGE_FIELDS = 2, CX_STAGE_TUNNELS = 10, CX_STAGE_CLASSES = 11, CX_STAGE_DIAMOND = 19, CX_READY = 20 };
enum { CX_PURPOSE_TICK, CX_PURPOSE_TIE, CX_PURPOSE_SPAWN, CX_PURPOSE_ANGLE, CX_PURPOSE_ORE, CX_PURPOSE_DIAMOND, CX_PURPOSE_REFINE, CX_PURPOSE_WANDER };
enum {
    CXE_MAP = 0,
    CXE_GRASS = CXE_MAP + 2 * CX_CELLS / 4,
    CXE_PATH = CXE_GRASS + 4 * CX_SIDE,
    CXE_MOB = CXE_PATH + 4 * CX_SIDE,
    CXE_PLANT = CXE_MOB + 2 * CX_SIDE,
    CXE_TAG = CXE_PLANT + 12,
    CXE_BATCH = CXE_TAG + 4,
    CXE_GX = CXE_BATCH + 16,
    CXE_GY = CXE_GX + CX_GRADS,
    CXE_FIELD = CXE_GY + CX_GRADS,
    CXE_RANGE = CXE_FIELD + 4 * CX_CELLS,
    CXE_TUNNEL = CXE_RANGE + 32,
    CXE_STONE = CXE_TUNNEL + 2 * CX_SIDE,
    CXE_COLS = CXE_STONE + 2 * CX_SIDE,
    CXE_COEF = CXE_COLS + 4 * 16 * 12,
    CXE_ORE = CXE_COEF + 4 * 16 * 4,
    CXE_PAINT = CXE_ORE + 8,
    CXE_RING = CXE_PAINT + 4,
    CXE_ORACLE = CXE_RING + 4 * CX_REACH
};
enum { CXO_DIST = CXE_ORACLE, CXO_QUEUE = CXO_DIST + CX_CELLS / 4, CXO_STATE = CXO_QUEUE + CX_CELLS / 2, CXO_MEMORY = CXO_STATE + 64, CXO_END = CXO_MEMORY + CX_CELLS / 4 };
enum { CXT_NONE, CXT_TREE, CXT_WATER, CXT_STONE, CXT_COAL, CXT_IRON, CXT_DIAMOND, CXT_RIPE, CXT_TABLE, CXT_FORGE, CXT_COW, CXT_UNSEEN, CXT_SOWN, CXT_DEN, CXT_TRAIL, CXT_COUNT };
enum { CXS_GAME, CXS_LIVES, CXS_PLANT, CXS_SOWN, CXS_DEN, CXS_IDLE, CXS_GOAL, CXS_AGE, CXS_ROAM, CXS_STALL, CXS_REST, CXS_QUENCH, CXS_HUNGRY, CXS_ALARM, CXS_WOUND, CXS_STRIKE, CXS_COW, CXS_TRAIL, CXS_HEALTH, CXS_BLOW, CXS_BLOCKED = CXS_BLOW + 3,
       CXS_COUNT = CXS_BLOCKED + CXT_COUNT };
enum { CXM_ALIVE = 1u << 31 };

static const uint32_t CX_WALK = 1u << CXB_GRASS | 1u << CXB_SAND | 1u << CXB_PATH | 1u << CXB_WOOD;
static const uint32_t CX_STEP_ON = 1u << CXB_GRASS | 1u << CXB_SAND | 1u << CXB_PATH | 1u << CXB_WOOD | 1u << CXB_LAVA;
static const uint32_t CX_FLY = 1u << CXB_GRASS | 1u << CXB_SAND | 1u << CXB_PATH | 1u << CXB_WOOD | 1u << CXB_LAVA | 1u << CXB_WATER;
static const uint32_t CX_BUILT = 1u << CXB_STONE | 1u << CXB_TABLE | 1u << CXB_FURNACE | 1u << CXB_PLANT | 1u << CXB_RIPE;

static const uint8_t CX_MENU_ACTION[CX_MENU] = { CXA_DO, CXA_NOOP, CXA_SLEEP, CXA_PLACE_TABLE, CXA_PLACE_STONE, CXA_PLACE_FURNACE, CXA_PLACE_PLANT,
                                                  CXA_WOOD_PICKAXE, CXA_WOOD_SWORD, CXA_STONE_PICKAXE, CXA_STONE_SWORD, CXA_IRON_PICKAXE, CXA_IRON_SWORD };
static const uint8_t CX_MOVE_ACTION[4] = { CXA_UP, CXA_DOWN, CXA_LEFT, CXA_RIGHT };
static const int8_t CX_READOUT[][2] = {
    { -3, -4 }, { -3, -3 }, { -3, -2 }, { -3, -1 }, { -3, 0 }, { -3, 1 }, { -3, 2 }, { -3, 3 }, { -3, 4 },
    { -2, -4 }, { -2, -3 }, { -2, 3 }, { -2, 4 },
    { -1, -4 }, { -1, -3 }, { -1, 3 }, { -1, 4 },
    { 0, -4 }, { 0, -3 }, { 0, 3 }, { 0, 4 },
    { 1, -4 }, { 1, -3 }, { 1, 3 }, { 1, 4 },
    { 2, -4 }, { 2, -3 }, { 2, 3 }, { 2, 4 },
    { 3, -4 }, { 3, -3 }, { 3, -2 }, { 3, -1 }, { 3, 0 }, { 3, 1 }, { 3, 2 }, { 3, 3 }, { 3, 4 }
};
_Static_assert(sizeof CX_READOUT / sizeof CX_READOUT[0] + CXI_COUNT == CX_SLOTS, "");
_Static_assert(CXV_COUNT <= OAK_ENV_OBS_MAX && CXW_COUNT <= OAK_ENV_WORDS_MAX && CXS_COUNT <= CXO_MEMORY - CXO_STATE, "");
static const uint8_t CX_LEVEL_M[32] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12, 12, 24, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36 };
static const uint8_t CX_LEVEL_W[16] = { 0, 0, 0, 0, 3, 3, 6, 9, 9, 9, 9, 9, 9, 9, 9, 9 };
static const uint8_t CX_LEVEL_T[16] = { 0, 0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2 };
static const uint8_t CX_CLASS[48] = {
    CXB_GRASS, CXB_TREECAND, CXB_TREECAND, CXB_GRASS, CXB_TREECAND, CXB_TREECAND, CXB_SAND, CXB_SAND, CXB_SAND, CXB_WATER, CXB_WATER, CXB_WATER,
    CXB_STONELO, CXB_STONELO, CXB_STONELO, CXB_STONELO, CXB_STONELO, CXB_STONELO, CXB_STONELO, CXB_STONELO, CXB_STONELO, CXB_STONELO, CXB_STONELO, CXB_STONELO,
    CXB_STONEHI, CXB_STONEHI, CXB_STONEHI, CXB_STONEHI, CXB_STONEHI, CXB_STONEHI, CXB_STONEHI, CXB_STONEHI, CXB_STONEHI, CXB_STONEHI, CXB_STONEHI, CXB_STONEHI,
    CXB_STONEHI, CXB_STONEHI, CXB_LAVA, CXB_PATH, CXB_PATH, CXB_LAVA, CXB_PATH, CXB_PATH, CXB_LAVA, CXB_PATH, CXB_PATH, CXB_LAVA
};
static const uint8_t CX_SETTLE[48] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
                                       CXB_GRASS, CXB_STONE, CXB_STONE, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47 };

typedef struct { uint32_t rows, cols, grad; } cxField;
static const cxField CX_FIELDS[4] = { { 4u, 4u, 0u }, { 4u, 4u, 25u }, { 8u, 32u, 50u }, { 16u, 16u, 347u } };
enum { CXF_WATER, CXF_MOUNTAIN, CXF_PATH, CXF_TREE };

uint32_t craftax_extraWords(uint32_t instance, uint32_t steps) {
    (void)instance; (void)steps;
    return CXO_END;
}

uint64_t craftax_shared(float *table) {
    const craftaxKnobs *k = &CRAFTAX_KNOBS[0];
    if (table)
        for (uint32_t t = 0u; t < k->dayLength; t++) {
            double progress = fmod((double)t / (double)k->dayLength, 1.0) + 0.3, c = fabs(cos(M_PI * progress));
            float light = (float)(1.0 - c * c * c), dark = 1.0f - light;
            table[t] = light;
            table[k->dayLength + t] = (float)(u01Bound(k->zombieChance + k->zombieNightChance * dark * dark) >> 8);
        }
    return 2u * k->dayLength;
}

static inline uint8_t *cx_map(uint32_t *extra, uint32_t buffer) { return (uint8_t *)(extra + CXE_MAP) + buffer * CX_CELLS; }
static inline const uint8_t *cx_mapc(const uint32_t *extra, uint32_t buffer) { return (const uint8_t *)(extra + CXE_MAP) + buffer * CX_CELLS; }
static inline uint64_t *cx_rows(uint32_t *extra, uint32_t at) { return (uint64_t *)(extra + at); }
static inline const uint64_t *cx_rowsc(const uint32_t *extra, uint32_t at) { return (const uint64_t *)(extra + at); }
static inline uint32_t cx_bit(const uint64_t *rows, uint32_t cell) { return (uint32_t)(rows[cell >> 6] >> (cell & 63u)) & 1u; }
static inline uint32_t cx_t24(float p) { return u01Bound(p) >> 8; }

static inline uint32_t cx_ahead(uint32_t cell, uint32_t facing) {
    static const int16_t DELTA[5] = { -1, 1, -CX_SIDE, CX_SIDE, 0 };
    uint32_t x = cell & 63u, y = cell >> 6;
    uint32_t wall = (uint32_t)(x == 0u) | (uint32_t)(x == 63u) << 1 | (uint32_t)(y == 0u) << 2 | (uint32_t)(y == 63u) << 3 | 16u;
    return wall >> facing & 1u ? CX_NONE : (uint32_t)((int32_t)cell + DELTA[facing]);
}

static inline uint32_t cx_distance(uint32_t a, uint32_t b) {
    int dy = (int)(a >> 6) - (int)(b >> 6), dx = (int)(a & 63u) - (int)(b & 63u);
    return (uint32_t)(abs(dy) + abs(dx));
}

static inline uint32_t cx_toward(uint32_t from, uint32_t to, uint32_t tie) {
    int dy = (int)(to >> 6) - (int)(from >> 6), dx = (int)(to & 63u) - (int)(from & 63u), ay = abs(dy), ax = abs(dx);
    if (!ay && !ax) return 4u;
    uint32_t vertical = ay > ax || (ay == ax && tie);
    return vertical ? (dy < 0 ? 2u : 3u) : (dx < 0 ? 0u : 1u);
}

static inline void cx_setBlock(uint8_t *map, uint64_t *grass, uint64_t *path, uint32_t cell, uint32_t block) {
    uint64_t bit = 1ull << (cell & 63u);
    map[cell] = (uint8_t)block;
    grass[cell >> 6] = block == CXB_GRASS ? grass[cell >> 6] | bit : grass[cell >> 6] & ~bit;
    path[cell >> 6] = block == CXB_PATH ? path[cell >> 6] | bit : path[cell >> 6] & ~bit;
}

static inline float4 cx_fade4(float4 t) { return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f); }
static inline float cx_fade(float t) { return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f); }

static inline void cx_turn4(float4 u, float4 *c, float4 *s) {
    float4 q = __builtin_elementwise_floor(u * 4.0f + 0.5f);
    float4 r = (u - q * 0.25f) * (2.0f * OAK_PI), r2 = r * r;
    float4 sp = __builtin_elementwise_fma(r2, splat4(1.0f / 362880.0f), splat4(-1.0f / 5040.0f));
    sp = __builtin_elementwise_fma(sp, r2, splat4(1.0f / 120.0f));
    sp = __builtin_elementwise_fma(sp, r2, splat4(-1.0f / 6.0f));
    float4 sr = __builtin_elementwise_fma(sp * r2, r, r);
    float4 cp = __builtin_elementwise_fma(r2, splat4(1.0f / 40320.0f), splat4(-1.0f / 720.0f));
    cp = __builtin_elementwise_fma(cp, r2, splat4(1.0f / 24.0f));
    cp = __builtin_elementwise_fma(cp, r2, splat4(-0.5f));
    float4 cr = __builtin_elementwise_fma(cp, r2, splat4(1.0f));
    lane4 quarter = (lane4)__builtin_convertvector(q, int4) & 3u;
    lane4 swap = (lane4)((quarter & 1u) != 0u);
    float4 cc = select4(swap, sr, cr), ss = select4(swap, cr, sr);
    *c = (float4)((lane4)cc ^ (((quarter + 1u) & 2u) << 30));
    *s = (float4)((lane4)ss ^ ((quarter & 2u) << 30));
}

static void cx_angles(uint32_t seed, uint32_t game, uint32_t firstBlock, uint32_t blocks, float *gx, float *gy) {
    for (uint32_t b = firstBlock; b < firstBlock + blocks; b++) {
        u44 r = philox4(seed, ENV_RNG_STREAM, (lane4)game, 4u * b + (lane4){ 0u, 1u, 2u, 3u }, (lane4)CX_PURPOSE_ANGLE, (lane4)0u);
        lane4 words[4] = { r.x, r.y, r.z, r.w };
        for (uint32_t k = 0u; k < 4u; k++) {
            float4 u = __builtin_convertvector(words[k] >> 8u, float4) * 0x1p-24f, c, s;
            cx_turn4(u, &c, &s);
            __builtin_memcpy(gx + 16u * b + 4u * k, &c, sizeof c);
            __builtin_memcpy(gy + 16u * b + 4u * k, &s, sizeof s);
        }
    }
}

static inline void cx_lattice(const float *g, uint32_t dc, float32x4_t *near, float32x4_t *far) {
    if (dc == 2u) {
        float32x2_t a = vld1_f32(g), b = vld1_f32(g + 1);
        *near = vcombine_f32(vdup_lane_f32(a, 0), vdup_lane_f32(a, 1));
        *far = vcombine_f32(vdup_lane_f32(b, 0), vdup_lane_f32(b, 1));
    } else {
        *near = vld1q_dup_f32(g);
        *far = vld1q_dup_f32(g + 1);
    }
}

static inline __attribute__((always_inline)) void cx_fieldBody(const float *gx, const float *gy, const float *cols, const float *coef, uint32_t stride, uint32_t dr, uint32_t dc, uint32_t a0, uint32_t a1, float *out, float *range) {
    float32x4_t cf[16];
    for (uint32_t s = 1u; s < dr; s++) cf[s] = vld1q_f32(coef + 4u * s);
    float32x4_t lo0 = vld1q_f32(range), hi0 = vld1q_f32(range + 4), lo1 = lo0, hi1 = hi0;
    for (uint32_t k = 0u; k < 16u; k++) {
        float32x4_t tv = vld1q_f32(cols + 12u * k), a3 = vld1q_f32(cols + 12u * k + 4u), a4 = vld1q_f32(cols + 12u * k + 8u), g0, g1, h0, h1;
        uint32_t b = dc == 2u ? 2u * k : 4u * k / dc;
        const float *xr = gx + a0 * stride + b, *yr = gy + a0 * stride + b;
        cx_lattice(xr, dc, &g0, &g1);
        cx_lattice(yr, dc, &h0, &h1);
        float32x4_t c1 = vfmaq_f32(g0, tv, vsubq_f32(g1, g0)), c3 = vfmaq_f32(vmulq_f32(a3, h0), a4, h1);
        for (uint32_t a = a0; a < a1; a++) {
            xr += stride;
            yr += stride;
            cx_lattice(xr, dc, &g0, &g1);
            cx_lattice(yr, dc, &h0, &h1);
            float32x4_t c2 = vfmaq_f32(g0, tv, vsubq_f32(g1, g0)), next = vfmaq_f32(vmulq_f32(a3, h0), a4, h1), c4 = vsubq_f32(next, c3);
            float *o = out + a * dr * CX_SIDE + 4u * k;
            vst1q_f32(o, c3);
            lo0 = vminq_f32(lo0, c3);
            hi0 = vmaxq_f32(hi0, c3);
            for (uint32_t s = 1u; s < dr; s++) {
                float32x4_t v = vfmaq_laneq_f32(vfmaq_laneq_f32(vfmaq_laneq_f32(c3, c4, cf[s], 0), c1, cf[s], 1), c2, cf[s], 2);
                vst1q_f32(o + s * CX_SIDE, v);
                if (s & 1u) {
                    lo1 = vminq_f32(lo1, v);
                    hi1 = vmaxq_f32(hi1, v);
                } else {
                    lo0 = vminq_f32(lo0, v);
                    hi0 = vmaxq_f32(hi0, v);
                }
            }
            c1 = c2;
            c3 = next;
        }
    }
    vst1q_f32(range, vminq_f32(lo0, lo1));
    vst1q_f32(range + 4, vmaxq_f32(hi0, hi1));
}

static void cx_fieldRows(const uint32_t *extra, uint32_t f, uint32_t row0, uint32_t rows, float *out, float *range) {
    const float *gx = (const float *)(extra + CXE_GX) + CX_FIELDS[f].grad, *gy = (const float *)(extra + CXE_GY) + CX_FIELDS[f].grad;
    const float *cols = (const float *)(extra + CXE_COLS) + 192u * f, *coef = (const float *)(extra + CXE_COEF) + 64u * f;
    uint32_t stride = CX_FIELDS[f].cols + 1u;
    if (f == CXF_PATH) cx_fieldBody(gx, gy, cols, coef, stride, 8u, 2u, row0 / 8u, (row0 + rows) / 8u, out, range);
    else if (f == CXF_TREE) cx_fieldBody(gx, gy, cols, coef, stride, 4u, 4u, row0 / 4u, (row0 + rows) / 4u, out, range);
    else cx_fieldBody(gx, gy, cols, coef, stride, 16u, 16u, row0 / 16u, (row0 + rows) / 16u, out, range);
}

static inline float cx_low(const float *range) { return fminf(fminf(range[0], range[1]), fminf(range[2], range[3])); }
static inline float cx_high(const float *range) { return fmaxf(fmaxf(range[4], range[5]), fmaxf(range[6], range[7])); }

static inline uint64_t cx_bits64(uint8x16_t m0, uint8x16_t m1, uint8x16_t m2, uint8x16_t m3) {
    const uint8x16_t weight = { 1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128 };
    uint8x16_t p0 = vpaddq_u8(vandq_u8(m0, weight), vandq_u8(m1, weight)), p1 = vpaddq_u8(vandq_u8(m2, weight), vandq_u8(m3, weight));
    uint8x16_t q = vpaddq_u8(p0, p1);
    return vgetq_lane_u64(vreinterpretq_u64_u8(vpaddq_u8(q, q)), 0);
}

static inline uint8x16_t cx_narrow(uint32x4_t a, uint32x4_t b, uint32x4_t c, uint32x4_t d) {
    uint16x8_t ab = vuzp1q_u16(vreinterpretq_u16_u32(a), vreinterpretq_u16_u32(b)), cd = vuzp1q_u16(vreinterpretq_u16_u32(c), vreinterpretq_u16_u32(d));
    return vuzp1q_u8(vreinterpretq_u8_u16(ab), vreinterpretq_u8_u16(cd));
}

static void cx_transpose(uint64_t *a) {
    uint64_t m = 0x00000000FFFFFFFFull;
    for (uint32_t j = 32u; j; j >>= 1, m ^= m << j)
        for (uint32_t k = 0u; k < 64u; k = ((k | j) + 1u) & ~j) {
            uint64_t t = ((a[k] >> j) ^ a[k | j]) & m;
            a[k] ^= t << j;
            a[k | j] ^= t;
        }
}

static void cx_tunnels(const float *field, const float *range, uint64_t *tunnel) {
    float low = cx_low(range), scale = 10.0f / (cx_high(range) - low);
    float32x4_t s = vdupq_n_f32(scale), o = vdupq_n_f32(-low * scale), eight = vdupq_n_f32(8.0f);
    for (uint32_t i = 0u; i < CX_SIDE; i++) {
        const float *x = field + i * CX_SIDE;
        uint8x16_t g[4];
        for (uint32_t q = 0u; q < 4u; q++) {
            const float *y = x + 16u * q;
            g[q] = cx_narrow(vcgtq_f32(vfmaq_f32(o, vld1q_f32(y), s), eight), vcgtq_f32(vfmaq_f32(o, vld1q_f32(y + 4), s), eight),
                             vcgtq_f32(vfmaq_f32(o, vld1q_f32(y + 8), s), eight), vcgtq_f32(vfmaq_f32(o, vld1q_f32(y + 12), s), eight));
        }
        tunnel[i] = cx_bits64(g[0], g[1], g[2], g[3]);
    }
    uint64_t across[CX_SIDE];
    __builtin_memcpy(across, tunnel, sizeof across);
    cx_transpose(across);
    for (uint32_t i = 0u; i < CX_SIDE; i++) tunnel[i] |= across[i];
}

static const float CX_TREE_CHANCE = 0.2f, CX_COAL_CHANCE = 0.04f, CX_IRON_CHANCE = 0.03f, CX_DIAMOND_CHANCE = 0.005f, CX_FLEE_CHANCE = 0.15f;
enum { CX_NEAR = 4, CX_SHOT = 6, CX_FAR = 10, CX_ALOOF = 17, CX_CAP = 9, CX_RIPE_AGE = 600, CX_ZOMBIE_REST = 5, CX_SKELETON_REST = 4 };

static inline uint64_t cx_span(int lo, int hi) {
    lo = lo < 0 ? 0 : lo;
    hi = hi > 63 ? 63 : hi;
    return lo > hi ? 0u : (2ull << hi) - (1ull << lo);
}

static void cx_prepare(uint32_t *extra) {
    for (uint32_t f = 0u; f < 4u; f++) {
        uint32_t dr = CX_SIDE / CX_FIELDS[f].rows, dc = CX_SIDE / CX_FIELDS[f].cols;
        float *cols = (float *)(extra + CXE_COLS) + 192u * f, *coef = (float *)(extra + CXE_COEF) + 64u * f;
        for (uint32_t k = 0u; k < 16u; k++) {
            lane4 j = 4u * k + (lane4){ 0u, 1u, 2u, 3u };
            float4 fv = __builtin_convertvector(j & (dc - 1u), float4) / (float)dc, t = cx_fade4(fv), a3 = (1.0f - t) * fv, a4 = t * (fv - 1.0f);
            __builtin_memcpy(cols + 12u * k, &t, sizeof t);
            __builtin_memcpy(cols + 12u * k + 4u, &a3, sizeof a3);
            __builtin_memcpy(cols + 12u * k + 8u, &a4, sizeof a4);
        }
        for (uint32_t s = 0u; s < 16u; s++) {
            float fu = (float)s / (float)dr, tu = cx_fade(fu), row[4] = { tu, (1.0f - tu) * fu, tu * (fu - 1.0f), 0.0f };
            __builtin_memcpy(coef + 4u * s, row, sizeof row);
        }
    }
    double coal = (double)CX_COAL_CHANCE, iron = (1.0 - coal) * (double)CX_IRON_CHANCE, diamond = (1.0 - coal) * (1.0 - (double)CX_IRON_CHANCE) * (double)CX_DIAMOND_CHANCE;
    uint64_t *ore = cx_rows(extra, CXE_ORE);
    ore[0] = (uint64_t)ceil((double)CX_TREE_CHANCE * 0x1p40);
    ore[1] = (uint64_t)ceil(coal * 0x1p40);
    ore[2] = (uint64_t)ceil((coal + iron) * 0x1p40);
    ore[3] = (uint64_t)ceil((coal + iron + diamond) * 0x1p40);
    uint8_t *paint = (uint8_t *)(extra + CXE_PAINT);
    for (uint32_t k = 0u; k < 4u; k++) paint[k] = (uint8_t)(ore[k] >> 32);
    uint64_t *ring = cx_rows(extra, CXE_RING);
    int32_t far = (int32_t)CRAFTAX_KNOBS[0].despawn - 1, near[2] = { CX_NEAR, CX_FAR };
    for (uint32_t kind = 0u; kind < 2u; kind++)
        for (int32_t dy = 0; dy < CX_REACH; dy++) {
            int32_t reach = far - dy, gap = near[kind] - dy;
            ring[CX_REACH * kind + (uint32_t)dy] = reach < 0 ? 0u : gap <= 0 ? cx_span(32 - reach, 32 + reach) : cx_span(32 - reach, 32 - gap) | cx_span(32 + gap, 32 + reach);
        }
}

static inline uint32_t cx_ore(uint32_t kind, uint32_t r, uint32_t cell, const uint64_t *ore, uint32_t seed, uint32_t game) {
    uint32_t first = kind == CXB_TREECAND ? 0u : 1u, last = kind == CXB_TREECAND ? 1u : kind == CXB_STONELO ? 3u : 4u;
    uint64_t u = (uint64_t)r << 32;
    for (uint32_t k = first; k < last; k++)
        if (r == ore[k] >> 32) {
            u |= philox4x32(seed, ENV_RNG_STREAM, game, cell, CX_PURPOSE_REFINE, 0u).x;
            break;
        }
    for (uint32_t k = first; k < last; k++)
        if (u < ore[k]) return k == 0u ? CXB_TREE : k == 1u ? CXB_COAL : k == 2u ? CXB_IRON : CXB_DIAMOND;
    return 0u;
}

static const uint8_t CX_VERDICT[48] = {
    255, 255, 255, 255, 255, 0, 1, 1, 255, 0, 254, 254, 255, 0, 2, 2, 255, 0, 254, 254, 255, 0, 255, 3, 255, 0, 255, 254, 255, 0, 255, 255,
    255, 254, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255
};
static const uint8_t CX_FOUND[16] = { CXB_TREE, CXB_COAL, CXB_IRON, CXB_DIAMOND };

typedef struct {
    uint8x16x2_t m;
    uint8x16_t w, t, found, weight;
    uint8x16x3_t classes, settle, verdict;
    uint8x16x4_t bucket;
    float32x4_t scale, om, ow, ot;
} cxSorter;

static inline __attribute__((always_inline)) uint8x16_t cx_level4(const float *x, float32x4_t scale, uint32_t lane, float32x4_t offset) {
    float32x4_t a = vld1q_f32(x), b = vld1q_f32(x + 4), c = vld1q_f32(x + 8), d = vld1q_f32(x + 12);
    uint32x4_t ia, ib, ic, id;
    if (lane == 0u) {
        ia = vcvtpq_u32_f32(vfmaq_laneq_f32(offset, a, scale, 0));
        ib = vcvtpq_u32_f32(vfmaq_laneq_f32(offset, b, scale, 0));
        ic = vcvtpq_u32_f32(vfmaq_laneq_f32(offset, c, scale, 0));
        id = vcvtpq_u32_f32(vfmaq_laneq_f32(offset, d, scale, 0));
    } else if (lane == 1u) {
        ia = vcvtpq_u32_f32(vfmaq_laneq_f32(offset, a, scale, 1));
        ib = vcvtpq_u32_f32(vfmaq_laneq_f32(offset, b, scale, 1));
        ic = vcvtpq_u32_f32(vfmaq_laneq_f32(offset, c, scale, 1));
        id = vcvtpq_u32_f32(vfmaq_laneq_f32(offset, d, scale, 1));
    } else {
        ia = vcvtpq_u32_f32(vfmaq_laneq_f32(offset, a, scale, 2));
        ib = vcvtpq_u32_f32(vfmaq_laneq_f32(offset, b, scale, 2));
        ic = vcvtpq_u32_f32(vfmaq_laneq_f32(offset, c, scale, 2));
        id = vcvtpq_u32_f32(vfmaq_laneq_f32(offset, d, scale, 2));
    }
    return vqsubq_u8(cx_narrow(ia, ib, ic, id), vdupq_n_u8(1));
}

static inline __attribute__((always_inline)) uint8x16_t cx_class16(const cxSorter *k, const float *fm, const float *fw, const float *ft, uint32_t at, uint32_t q, uint32_t dy, uint8x8_t bore) {
    const uint8x16_t lane = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 }, spread = { 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1 };
    uint8x16_t m = cx_level4(fm + at, k->scale, 0u, k->om), w = cx_level4(fw + at, k->scale, 1u, k->ow), t = cx_level4(ft + at, k->scale, 2u, k->ot);
    if (dy <= 4u && (q == 1u || q == 2u)) {
        uint8x16_t reach = vaddq_u8(vabdq_u8(vaddq_u8(lane, vdupq_n_u8((uint8_t)(16u * q))), vdupq_n_u8(32u)), vdupq_n_u8((uint8_t)dy));
        uint8x16_t inside = vcleq_u8(reach, vdupq_n_u8(4u)), core = vcltq_u8(reach, vdupq_n_u8(4u));
        m = vqsubq_u8(m, vandq_u8(inside, vbslq_u8(core, vdupq_n_u8(255u), vdupq_n_u8(4u))));
        w = vqsubq_u8(w, vandq_u8(inside, vbslq_u8(core, vdupq_n_u8(255u), vdupq_n_u8(2u))));
    }
    uint8x16_t c = vqtbl3q_u8(k->classes, vaddq_u8(vaddq_u8(vqtbl2q_u8(k->m, m), vqtbl1q_u8(k->w, w)), vqtbl1q_u8(k->t, t)));
    uint8x16_t bored = vtstq_u8(vqtbl1q_u8(vcombine_u8(bore, bore), vaddq_u8(spread, vdupq_n_u8((uint8_t)(2u * q)))), k->weight);
    return vbslq_u8(vandq_u8(bored, vcgeq_u8(c, vdupq_n_u8(CXB_STONELO))), vdupq_n_u8(CXB_PATH), c);
}

static inline __attribute__((always_inline)) uint8x16_t cx_ore16(const cxSorter *k, uint8x16_t c, uint32_t at, uint8x16_t b, uint8_t *map, uint8x16_t *doubt) {
    const uint8x16_t lane = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
    uint8x16_t group = vqsubq_u8(c, vdupq_n_u8(CXB_TREECAND - 1u));
    if (at == (CX_START & ~15u)) group = vbicq_u8(group, vandq_u8(vceqq_u8(lane, vdupq_n_u8(CX_START & 15u)), vceqq_u8(c, vdupq_n_u8(CXB_TREECAND))));
    uint8x16_t pick = vqtbl3q_u8(k->verdict, vaddq_u8(vqtbl4q_u8(k->bucket, b), group));
    uint8x16_t g = vqtbx1q_u8(vqtbl3q_u8(k->settle, c), k->found, pick);
    *doubt = vceqq_u8(pick, vdupq_n_u8(254u));
    vst1q_u8(map + at, g);
    return g;
}

static void cx_classifyRows(uint32_t seed, uint32_t game, uint32_t row0, uint32_t rows, uint32_t *extra, uint32_t buffer) {
    const float *field = (const float *)(extra + CXE_FIELD), *range = (const float *)(extra + CXE_RANGE);
    const float *fw = field + CXF_WATER * CX_CELLS, *fm = field + CXF_MOUNTAIN * CX_CELLS, *ft = field + CXF_TREE * CX_CELLS;
    float lw = cx_low(range + 8u * CXF_WATER), sw = 10.0f / (cx_high(range + 8u * CXF_WATER) - lw);
    float lm = cx_low(range + 8u * CXF_MOUNTAIN), sm = 20.0f / (cx_high(range + 8u * CXF_MOUNTAIN) - lm);
    float lt = cx_low(range + 8u * CXF_TREE), st = 10.0f / (cx_high(range + 8u * CXF_TREE) - lt);
    cxSorter k;
    k.m = vld1q_u8_x2(CX_LEVEL_M);
    k.w = vld1q_u8(CX_LEVEL_W);
    k.t = vld1q_u8(CX_LEVEL_T);
    k.classes = vld1q_u8_x3(CX_CLASS);
    k.settle = vld1q_u8_x3(CX_SETTLE);
    k.verdict = vld1q_u8_x3(CX_VERDICT);
    k.found = vld1q_u8(CX_FOUND);
    k.weight = (uint8x16_t){ 1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128 };
    k.scale = (float32x4_t){ sm, sw, st, 0.0f };
    k.om = vdupq_n_f32(-lm * sm);
    k.ow = vdupq_n_f32(-lw * sw);
    k.ot = vdupq_n_f32(-lt * st);
    const uint8_t *paint = (const uint8_t *)(extra + CXE_PAINT);
    const uint8x16_t lane = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
    for (uint32_t q = 0u; q < 4u; q++) {
        uint8x16_t b = vaddq_u8(lane, vdupq_n_u8((uint8_t)(16u * q))), n = vdupq_n_u8(1u);
        for (uint32_t j = 0u; j < 4u; j++) n = vsubq_u8(vsubq_u8(n, vcgeq_u8(b, vdupq_n_u8(paint[j]))), vcgtq_u8(b, vdupq_n_u8(paint[j])));
        k.bucket.val[q] = vshlq_n_u8(vbicq_u8(n, vceqq_u8(n, vdupq_n_u8(9u))), 2);
    }
    const uint64_t *ore = cx_rowsc(extra, CXE_ORE), *tunnel = cx_rowsc(extra, CXE_TUNNEL);
    uint8_t *map = cx_map(extra, buffer);
    uint64_t *grass = cx_rows(extra, CXE_GRASS + 2u * CX_SIDE * buffer), *path = cx_rows(extra, CXE_PATH + 2u * CX_SIDE * buffer), *stone = cx_rows(extra, CXE_STONE);
    const uint8x16_t green = vdupq_n_u8(CXB_GRASS), trail = vdupq_n_u8(CXB_PATH), rock = vdupq_n_u8(CXB_STONE);
    for (uint32_t i = row0; i < row0 + rows; i++) {
        uint8_t kind[CX_SIDE];
        uint32_t dy = i > 32u ? i - 32u : 32u - i, at = i * CX_SIDE;
        u44 r = philox4(seed, ENV_RNG_STREAM, (lane4)game, 4u * i + (lane4){ 0u, 1u, 2u, 3u }, (lane4)CX_PURPOSE_ORE, (lane4)0u);
        uint8x16_t b0 = vreinterpretq_u8_u32((uint32x4_t)r.x), b1 = vreinterpretq_u8_u32((uint32x4_t)r.y), b2 = vreinterpretq_u8_u32((uint32x4_t)r.z), b3 = vreinterpretq_u8_u32((uint32x4_t)r.w);
        uint8x8_t bore = vcreate_u8(tunnel[i]);
        uint8x16_t c0 = cx_class16(&k, fm, fw, ft, at, 0u, dy, bore), c1 = cx_class16(&k, fm, fw, ft, at + 16u, 1u, dy, bore);
        uint8x16_t c2 = cx_class16(&k, fm, fw, ft, at + 32u, 2u, dy, bore), c3 = cx_class16(&k, fm, fw, ft, at + 48u, 3u, dy, bore);
        vst1q_u8(kind, c0);
        vst1q_u8(kind + 16, c1);
        vst1q_u8(kind + 32, c2);
        vst1q_u8(kind + 48, c3);
        uint8x16_t d0, d1, d2, d3;
        uint8x16_t f0 = cx_ore16(&k, c0, at, b0, map, &d0), f1 = cx_ore16(&k, c1, at + 16u, b1, map, &d1);
        uint8x16_t f2 = cx_ore16(&k, c2, at + 32u, b2, map, &d2), f3 = cx_ore16(&k, c3, at + 48u, b3, map, &d3);
        uint64_t meadow = cx_bits64(vceqq_u8(f0, green), vceqq_u8(f1, green), vceqq_u8(f2, green), vceqq_u8(f3, green));
        uint64_t trails = cx_bits64(vceqq_u8(f0, trail), vceqq_u8(f1, trail), vceqq_u8(f2, trail), vceqq_u8(f3, trail));
        uint64_t quarry = cx_bits64(vceqq_u8(f0, rock), vceqq_u8(f1, rock), vceqq_u8(f2, rock), vceqq_u8(f3, rock));
        if (__builtin_expect(vmaxvq_u8(vorrq_u8(vorrq_u8(d0, d1), vorrq_u8(d2, d3))) != 0u, 0)) {
            uint8_t draw[CX_SIDE];
            vst1q_u8(draw, b0);
            vst1q_u8(draw + 16, b1);
            vst1q_u8(draw + 32, b2);
            vst1q_u8(draw + 48, b3);
            uint64_t unsure = cx_bits64(d0, d1, d2, d3);
            uint8_t *row = map + at;
            while (unsure) {
                uint32_t x = (uint32_t)__builtin_ctzll(unsure);
                unsure &= unsure - 1u;
                uint32_t block = cx_ore(kind[x], draw[x], at + x, ore, seed, game);
                if (!block) continue;
                row[x] = (uint8_t)block;
                meadow &= ~(1ull << x);
                quarry &= ~(1ull << x);
            }
        }
        grass[i] = meadow;
        path[i] = trails;
        stone[i] = quarry;
    }
}

static void cx_generate(uint32_t seed, uint32_t game, uint32_t stage, uint32_t *extra, uint32_t buffer) {
    static const uint8_t ORDER[4] = { CXF_PATH, CXF_WATER, CXF_MOUNTAIN, CXF_TREE };
    float *gx = (float *)(extra + CXE_GX), *gy = (float *)(extra + CXE_GY), *field = (float *)(extra + CXE_FIELD), *range = (float *)(extra + CXE_RANGE);
    if (stage < CX_STAGE_FIELDS) {
        cx_angles(seed, game, 20u * stage, 20u, gx, gy);
        return;
    }
    if (stage < CX_STAGE_TUNNELS) {
        uint32_t f = ORDER[(stage - CX_STAGE_FIELDS) >> 1], half = (stage - CX_STAGE_FIELDS) & 1u;
        float *r = range + 8u * f;
        if (!half)
            for (uint32_t k = 0u; k < 4u; k++) { r[k] = INFINITY; r[4u + k] = -INFINITY; }
        cx_fieldRows(extra, f, CX_ROWS_PER_FIELD_SLICE * half, CX_ROWS_PER_FIELD_SLICE, field + f * CX_CELLS, r);
        return;
    }
    if (stage == CX_STAGE_TUNNELS) {
        cx_tunnels(field + CXF_PATH * CX_CELLS, range + 8u * CXF_PATH, cx_rows(extra, CXE_TUNNEL));
        return;
    }
    if (stage < CX_STAGE_DIAMOND) {
        cx_classifyRows(seed, game, CX_ROWS_PER_CLASS_SLICE * (stage - CX_STAGE_CLASSES), CX_ROWS_PER_CLASS_SLICE, extra, buffer);
        return;
    }
    const uint64_t *stone = cx_rowsc(extra, CXE_STONE);
    uint32_t count = 0u;
    for (uint32_t i = 0u; i < CX_SIDE; i++) count += (uint32_t)__builtin_popcountll(stone[i]);
    if (!count) return;
    uint32_t pick = (uint32_t)(((uint64_t)philox4x32(seed, ENV_RNG_STREAM, game, 0u, CX_PURPOSE_DIAMOND, 0u).x * count) >> 32);
    for (uint32_t i = 0u; i < CX_SIDE; i++) {
        uint32_t here = (uint32_t)__builtin_popcountll(stone[i]);
        if (pick >= here) { pick -= here; continue; }
        uint64_t m = stone[i];
        for (uint32_t k = 0u; k < pick; k++) m &= m - 1u;
        cx_map(extra, buffer)[i * CX_SIDE + (uint32_t)__builtin_ctzll(m)] = CXB_DIAMOND;
        return;
    }
}

static void cx_begin(uint32_t *words, uint32_t *extra) {
    words[CXW_PLAYER] = CX_START | 2u << 12;
    words[CXW_VITALS] = 9u | 9u << 4 | 9u << 8 | 9u << 12;
    for (uint32_t k = CXW_HUNGER; k <= CXW_TIME; k++) words[k] = 0u;
    words[CXW_MENU] &= 0x7F00u;
    for (uint32_t k = CXW_ZOMBIE; k < CXW_COUNT; k++) words[k] = 0u;
    for (uint32_t i = 0u; i < 3u; i++) {
        words[CXW_ZOMBIE + i] = 1u << 12;
        words[CXW_COW + i] = CRAFTAX_KNOBS[0].cowHealth << 12;
    }
    __builtin_memset(extra + CXE_MOB, 0, 2u * CX_SIDE * sizeof(uint32_t));
    __builtin_memset(extra + CXE_PLANT, 0, 12u * sizeof(uint32_t));
}

static __attribute__((noinline, cold)) uint32_t cx_tie(uint32_t t24, uint32_t seed, uint32_t step, uint32_t which) {
    return (philox4x32(seed, ENV_RNG_STREAM, step, which, CX_PURPOSE_TIE, 0u).x & 0xFFFFu) < (t24 & 0xFFFFu);
}

static inline uint32_t cx_chance(uint32_t byte, uint32_t t24, uint32_t seed, uint32_t step, uint32_t which) {
    uint32_t top = t24 >> 16;
    return __builtin_expect(byte != top, 1) ? byte < top : cx_tie(t24, seed, step, which);
}

static uint32_t cx_spawnCell(const uint64_t *grass, const uint64_t *path, const uint64_t *mob, uint32_t kinds, uint32_t player, const uint64_t *ring, uint32_t far, uint32_t draw) {
    uint32_t py = player >> 6, y0 = py > far ? py - far : 0u, y1 = py + far < 63u ? py + far : 63u;
    int64x2_t shift = vdupq_n_s64((int64_t)(player & 63u) - 32);
    uint64x2_t pickGrass = vdupq_n_u64(kinds & 1u ? ~0ull : 0u), pickPath = vdupq_n_u64(kinds & 2u ? ~0ull : 0u);
    uint64_t open[2 * CX_REACH];
    uint16x8_t tally = vdupq_n_u16(0u);
    for (uint32_t y = y0; y <= y1; y += 2u) {
        uint32_t a = y > py ? y - py : py - y, b = y + 1u > py ? y + 1u - py : py - y - 1u;
        uint64x2_t band = { ring[a], y < y1 ? ring[b] : 0u };
        uint64x2_t free = vandq_u64(vbicq_u64(vorrq_u64(vandq_u64(vld1q_u64(grass + y), pickGrass), vandq_u64(vld1q_u64(path + y), pickPath)), vld1q_u64(mob + y)), vshlq_u64(band, shift));
        vst1q_u64(open + (y - y0), free);
        tally = vpadalq_u8(tally, vcntq_u8(vreinterpretq_u8_u64(free)));
    }
    uint32_t total = vaddvq_u16(tally);
    if (!total) return CX_NONE;
    uint32_t pick = (uint32_t)(((uint64_t)draw * total) >> 32);
    for (uint32_t y = y0;; y++) {
        uint64_t m = open[y - y0];
        uint32_t here = (uint32_t)__builtin_popcountll(m);
        if (pick >= here) {
            pick -= here;
            continue;
        }
        for (uint32_t k = 0u; k < pick; k++) m &= m - 1u;
        return y * CX_SIDE + (uint32_t)__builtin_ctzll(m);
    }
}

static inline uint32_t cx_open(const uint8_t *map, const uint64_t *mob, uint32_t target, uint32_t player) {
    uint32_t t = target & 4095u;
    return (uint32_t)(target != CX_NONE) & (CX_WALK >> map[t] & 1u) & (uint32_t)~(mob[t >> 6] >> (t & 63u)) & (uint32_t)(target != player);
}

static inline uint32_t cx_near(const uint8_t *map, uint32_t player) {
    uint32_t y = player >> 6, x = player & 63u, found = 0u;
    uint32_t y0 = y ? y - 1u : 0u, y1 = y < 63u ? y + 1u : 63u, x0 = x ? x - 1u : 0u, x1 = x < 63u ? x + 1u : 63u;
    for (uint32_t r = y0; r <= y1; r++)
        for (uint32_t c = x0; c <= x1; c++) {
            uint32_t b = map[r * CX_SIDE + c];
            found |= (uint32_t)(b == CXB_TABLE) | (uint32_t)(b == CXB_FURNACE) << 1;
        }
    return found;
}

static inline void cx_refill(uint32_t seed, uint32_t step, uint32_t *extra) {
    u44 r = philox4(seed, ENV_RNG_STREAM, (lane4)step, (lane4){ 0u, 1u, 2u, 3u }, (lane4)CX_PURPOSE_TICK, (lane4)0u);
    vst4q_u32(extra + CXE_BATCH, ((uint32x4x4_t){ { (uint32x4_t)r.x, (uint32x4_t)r.y, (uint32x4_t)r.z, (uint32x4_t)r.w } }));
}

typedef struct {
    uint8_t *map;
    uint64_t *grass, *path, *mob;
    uint8_t *inv;
    uint32_t player, facing, sleeping, food, drink, energy, ach;
    int32_t health, hunger, thirst, fatigue, recover;
} cxLife;

static __attribute__((noinline)) void cx_strike(cxLife *s, uint32_t *words, uint32_t k) {
    int32_t wood = 2 * s->inv[CXI_WOOD_SWORD], stone = 3 * s->inv[CXI_STONE_SWORD], iron = 5 * s->inv[CXI_IRON_SWORD];
    int32_t damage = wood > stone ? wood : stone;
    damage = iron > damage ? iron : damage;
    damage = damage > 1 ? damage : 1;
    uint32_t w = words[k], target = w & 4095u;
    int32_t hp = (int32_t)(w >> 12 & 15u) - damage;
    if (hp > 0) {
        words[k] = (w & ~(15u << 12)) | (uint32_t)hp << 12;
        return;
    }
    words[k] = w & ~(CXM_ALIVE | 15u << 12);
    s->mob[target >> 6] &= ~(1ull << (target & 63u));
    if (k < CXW_COW) s->ach |= 1u << CXG_ZOMBIE;
    else if (k < CXW_SKELETON) {
        s->food = s->food + 6u > CX_CAP ? CX_CAP : s->food + 6u;
        s->hunger = 0;
        s->ach |= 1u << CXG_COW;
    } else s->ach |= 1u << CXG_SKELETON;
}

static __attribute__((noinline, cold)) void cx_respawn(cxLife *s, uint32_t *words, uint32_t *extra) {
    s->player = CX_START;
    s->facing = 2u;
    s->sleeping = 0u;
    s->food = s->drink = s->energy = CX_CAP;
    s->health = CX_CAP;
    s->hunger = s->thirst = s->fatigue = s->recover = 0;
    __builtin_memset(s->inv, 0, CXI_COUNT);
    for (uint32_t m = CXW_ZOMBIE; m < CXW_COUNT; m++) {
        uint32_t w = words[m], cell = w & 4095u, hostile = m < CXW_COW || m >= CXW_SKELETON;
        uint32_t view = abs((int)(cell >> 6) - (CX_START >> 6)) <= -CX_READOUT[0][0] && abs((int)(cell & 63u) - (CX_START & 63)) <= -CX_READOUT[0][1];
        if (!(hostile ? view : (w & CXM_ALIVE) && cell == CX_START)) continue;
        words[m] = m < CXW_ARROW ? w & ~(CXM_ALIVE | 15u << 12) : 0u;
        if ((w & CXM_ALIVE) && m < CXW_ARROW) s->mob[cell >> 6] &= ~(1ull << (cell & 63u));
    }
    s->mob[CX_START >> 6] &= ~(1ull << (CX_START & 63u));
    uint32_t *plant = extra + CXE_PLANT, sown = 0u;
    while (sown < 10u && (plant[sown] & CXM_ALIVE)) sown++;
    for (uint32_t d = 0u; d < 5u; d++) {
        uint32_t cell = d < 4u ? cx_ahead(CX_START, d) : CX_START;
        if (!(CX_BUILT >> s->map[cell] & 1u)) continue;
        cx_setBlock(s->map, s->grass, s->path, cell, CXB_PATH);
        for (uint32_t p = 0u; p < sown; p++)
            if ((plant[p] & 4095u) == cell) {
                plant[p] = plant[--sown];
                plant[sown] = 0u;
                break;
            }
    }
    words[CXW_DEATHS]++;
}

static inline void cx_gain(uint8_t *inv, uint32_t item) { inv[item] = (uint8_t)(inv[item] + 1u > CX_CAP ? CX_CAP : inv[item] + 1u); }

static __attribute__((noinline)) void cx_interact(const craftaxKnobs *k, cxLife *s, uint32_t *words, uint32_t *extra, uint32_t seed, uint32_t step, uint32_t act, const uint32_t *draw) {
    uint32_t target = cx_ahead(s->player, s->facing);
    uint8_t *inv = s->inv;
    if (act == CXA_DO) {
        uint32_t victim = 0u;
        for (uint32_t m = CXW_ZOMBIE; m < CXW_ARROW && !victim && target != CX_NONE; m++)
            if ((words[m] & CXM_ALIVE) && (words[m] & 4095u) == target) victim = m;
        if (target == CX_NONE) return;
        if (victim) {
            cx_strike(s, words, victim);
            return;
        }
        switch (s->map[target]) {
            case CXB_TREE:
                cx_gain(inv, CXI_WOOD);
                cx_setBlock(s->map, s->grass, s->path, target, CXB_GRASS);
                s->ach |= 1u << CXG_WOOD;
                return;
            case CXB_STONE:
            case CXB_COAL:
            case CXB_IRON:
            case CXB_DIAMOND: {
                uint32_t b = s->map[target];
                uint32_t tool = b == CXB_IRON ? CXI_STONE_PICKAXE : b == CXB_DIAMOND ? CXI_IRON_PICKAXE : CXI_WOOD_PICKAXE;
                if (!inv[tool]) return;
                uint32_t item = b == CXB_STONE ? CXI_STONE : b == CXB_COAL ? CXI_COAL : b == CXB_IRON ? CXI_IRON : CXI_DIAMOND;
                cx_gain(inv, item);
                cx_setBlock(s->map, s->grass, s->path, target, CXB_PATH);
                s->ach |= 1u << (b == CXB_STONE ? CXG_STONE : b == CXB_COAL ? CXG_COAL : b == CXB_IRON ? CXG_IRON : CXG_DIAMOND);
                return;
            }
            case CXB_WATER:
                s->drink = s->drink + 1u > CX_CAP ? CX_CAP : s->drink + 1u;
                s->thirst = 0;
                s->ach |= 1u << CXG_DRINK;
                return;
            case CXB_GRASS:
                if (cx_chance(draw[0] >> 24, cx_t24(k->saplingChance), seed, step, 8u)) {
                    cx_gain(inv, CXI_SAPLING);
                    s->ach |= 1u << CXG_SAPLING;
                }
                return;
            case CXB_RIPE: {
                s->food = s->food + 4u > CX_CAP ? CX_CAP : s->food + 4u;
                s->hunger = 0;
                s->map[target] = CXB_PLANT;
                uint32_t *plant = extra + CXE_PLANT;
                for (uint32_t p = 0u; p < 10u; p++)
                    if ((plant[p] & CXM_ALIVE) && (plant[p] & 4095u) == target) plant[p] &= ~(1023u << 12);
                s->ach |= 1u << CXG_EAT_PLANT;
                return;
            }
            default:
                return;
        }
    }
    if (act >= CXA_PLACE_STONE && act <= CXA_PLACE_PLANT) {
        if (target == CX_NONE || cx_bit(s->mob, target)) return;
        uint32_t b = s->map[target];
        if (act == CXA_PLACE_STONE && inv[CXI_STONE] && (CX_FLY >> b & 1u)) {
            cx_setBlock(s->map, s->grass, s->path, target, CXB_STONE);
            inv[CXI_STONE]--;
            s->ach |= 1u << CXG_PLACE_STONE;
        } else if (act == CXA_PLACE_TABLE && inv[CXI_WOOD] >= 2u && (CX_STEP_ON >> b & 1u)) {
            cx_setBlock(s->map, s->grass, s->path, target, CXB_TABLE);
            inv[CXI_WOOD] = (uint8_t)(inv[CXI_WOOD] - 2u);
            s->ach |= 1u << CXG_TABLE;
        } else if (act == CXA_PLACE_FURNACE && inv[CXI_STONE] && (CX_STEP_ON >> b & 1u)) {
            cx_setBlock(s->map, s->grass, s->path, target, CXB_FURNACE);
            inv[CXI_STONE]--;
            s->ach |= 1u << CXG_FURNACE;
        } else if (act == CXA_PLACE_PLANT && inv[CXI_SAPLING] && b == CXB_GRASS) {
            cx_setBlock(s->map, s->grass, s->path, target, CXB_PLANT);
            inv[CXI_SAPLING]--;
            uint32_t *plant = extra + CXE_PLANT;
            for (uint32_t p = 0u; p < 10u; p++)
                if (!(plant[p] & CXM_ALIVE)) { plant[p] = target | CXM_ALIVE; break; }
            s->ach |= 1u << CXG_PLANT;
        }
        return;
    }
    if (act >= CXA_WOOD_PICKAXE) {
        uint32_t near = cx_near(s->map, s->player);
        if (!(near & 1u) || !inv[CXI_WOOD]) return;
        uint32_t tier = act == CXA_WOOD_PICKAXE || act == CXA_WOOD_SWORD ? 0u : act == CXA_STONE_PICKAXE || act == CXA_STONE_SWORD ? 1u : 2u;
        if (tier >= 1u && !inv[CXI_STONE]) return;
        if (tier == 2u && (!(near & 2u) || !inv[CXI_COAL] || !inv[CXI_IRON])) return;
        inv[CXI_WOOD]--;
        if (tier >= 1u) inv[CXI_STONE]--;
        if (tier == 2u) { inv[CXI_COAL]--; inv[CXI_IRON]--; }
        static const uint8_t ITEM[6] = { CXI_WOOD_PICKAXE, CXI_STONE_PICKAXE, CXI_IRON_PICKAXE, CXI_WOOD_SWORD, CXI_STONE_SWORD, CXI_IRON_SWORD };
        static const uint8_t GOAL[6] = { CXG_WOOD_PICKAXE, CXG_STONE_PICKAXE, CXG_IRON_PICKAXE, CXG_WOOD_SWORD, CXG_STONE_SWORD, CXG_IRON_SWORD };
        cx_gain(inv, ITEM[act - CXA_WOOD_PICKAXE]);
        s->ach |= 1u << GOAL[act - CXA_WOOD_PICKAXE];
    }
}

static __attribute__((noinline)) void cx_arrive(const uint64_t *grass, const uint64_t *path, uint64_t *mob, uint32_t player, uint32_t *words, const uint64_t *ring, uint32_t base, uint32_t slots, uint32_t kinds, uint32_t health, uint32_t far, uint32_t pick) {
    uint32_t cell = cx_spawnCell(grass, path, mob, kinds, player, ring, far, pick);
    if (cell == CX_NONE) return;
    for (uint32_t i = 0u; i < slots; i++) {
        uint32_t w = words[base + i];
        if (w & CXM_ALIVE) continue;
        words[base + i] = (w & (15u << 16)) | cell | health << 12 | CXM_ALIVE;
        mob[cell >> 6] |= 1ull << (cell & 63u);
        return;
    }
}

static inline __attribute__((always_inline)) void cx_spawn(const craftaxKnobs *k, cxLife *s, uint32_t *words, const uint32_t *extra, uint32_t seed, uint32_t step, const uint32_t *draw, uint32_t zombieT24) {
    const uint64_t *ring = cx_rowsc(extra, CXE_RING);
    uint32_t spawned = 0u, far = k->despawn - 1u;
    if ((words[CXW_COW] & words[CXW_COW + 1] & words[CXW_COW + 2]) >> 31) {
        uint32_t first = words[CXW_COW] & 4095u;
        s->mob[first >> 6] |= 1ull << (first & 63u);
    } else if (__builtin_expect(cx_chance(draw[0] & 255u, cx_t24(k->cowChance), seed, step, 0u), 0)) {
        cx_arrive(s->grass, s->path, s->mob, s->player, words, ring, CXW_COW, 3u, 1u, k->cowHealth, far, draw[3]);
        spawned = 1u;
    }
    if ((words[CXW_ZOMBIE] & words[CXW_ZOMBIE + 1] & words[CXW_ZOMBIE + 2]) >> 31) {
        uint32_t first = words[CXW_ZOMBIE] & 4095u;
        s->mob[first >> 6] |= 1ull << (first & 63u);
    } else if (__builtin_expect(cx_chance(draw[0] >> 8 & 255u, zombieT24, seed, step, 1u), 0)) {
        cx_arrive(s->grass, s->path, s->mob, s->player, words, ring + CX_REACH, CXW_ZOMBIE, 3u, 3u, k->zombieHealth, far, spawned ? philox4x32(seed, ENV_RNG_STREAM, step, 1u, CX_PURPOSE_SPAWN, 0u).x : draw[3]);
        spawned = 1u;
    }
    if ((words[CXW_SKELETON] & words[CXW_SKELETON + 1]) >> 31) {
        uint32_t first = words[CXW_SKELETON] & 4095u;
        s->mob[first >> 6] |= 1ull << (first & 63u);
    } else if (__builtin_expect(cx_chance(draw[0] >> 16 & 255u, cx_t24(k->skeletonChance), seed, step, 2u), 0))
        cx_arrive(s->grass, s->path, s->mob, s->player, words, ring + CX_REACH, CXW_SKELETON, 2u, 2u, k->skeletonHealth, far, spawned ? philox4x32(seed, ENV_RNG_STREAM, step, 2u, CX_PURPOSE_SPAWN, 0u).x : draw[3]);
}

static __attribute__((noinline)) uint32_t cx_arrows(uint8_t *map, uint64_t *grass, uint64_t *path, const uint64_t *mob, uint32_t player, uint32_t *words) {
    uint32_t hits = 0u;
    for (uint32_t a = 0u; a < 3u; a++) {
        uint32_t w = words[CXW_ARROW + a];
        if (!(w & CXM_ALIVE)) continue;
        uint32_t target = cx_ahead(w & 4095u, w >> 20 & 3u), alive = 0u;
        if (target == player) hits++;
        else if (target != CX_NONE && !cx_bit(mob, target)) {
            uint32_t b = map[target];
            if (b == CXB_TABLE || b == CXB_FURNACE) cx_setBlock(map, grass, path, target, CXB_PATH);
            else if (CX_FLY >> b & 1u) alive = 1u;
        }
        words[CXW_ARROW + a] = alive ? (w & ~4095u) | target : 0u;
    }
    return hits;
}

static inline void cx_shoot(uint32_t *words, uint32_t cell, uint32_t dir) {
    if (dir > 3u) return;
    for (uint32_t a = 0u; a < 3u; a++)
        if (!(words[CXW_ARROW + a] & CXM_ALIVE)) {
            words[CXW_ARROW + a] = cell | dir << 20 | CXM_ALIVE;
            return;
        }
}

static inline void cx_settle(uint64_t *mob, uint32_t from, uint32_t to) {
    mob[from >> 6] &= ~(1ull << (from & 63u));
    mob[to >> 6] |= 1ull << (to & 63u);
}

static __attribute__((noinline)) uint32_t cx_odd(uint32_t *words, uint64_t *mob, const uint8_t *map, uint32_t player, uint32_t i, uint32_t t, uint32_t dist, uint32_t aim, uint32_t seed, uint32_t step, uint32_t luck, uint32_t sleeping) {
    uint32_t w = words[CXW_ZOMBIE + i], rest = w & 15u << 16, cell = w & 4095u, alive = w >> 31;
    if (alive && dist >= CRAFTAX_KNOBS[0].despawn) {
        alive = 0u;
        w &= ~(15u << 12);
        mob[cell >> 6] &= ~(1ull << (cell & 63u));
    } else if (alive && !rest && i < 3u && dist == 1u) {
        mob[cell >> 6] |= 1ull << (cell & 63u);
        words[CXW_ZOMBIE + i] = w | (uint32_t)CX_ZOMBIE_REST << 16;
        return 2u + 5u * sleeping;
    } else if (alive && !rest && i >= 6u && dist >= CX_NEAR && dist < CX_SHOT) {
        cx_shoot(words, cell, aim);
        mob[cell >> 6] |= 1ull << (cell & 63u);
        words[CXW_ZOMBIE + i] = w | (uint32_t)CX_SKELETON_REST << 16;
        return 0u;
    }
    if (i >= 6u && (dist < CX_NEAR || dist >= CX_FAR) && (luck >> (8u * (i - 6u)) & 255u) == cx_t24(CX_FLEE_CHANCE) >> 16 &&
        cx_chance(cx_t24(CX_FLEE_CHANCE) >> 16, cx_t24(CX_FLEE_CHANCE), seed, step, i - 2u)) {
        uint32_t target = cx_ahead(cell, aim ^ (uint32_t)(dist < CX_NEAR && aim < 4u));
        t = cx_open(map, mob, target, player) ? target : CX_NONE;
    }
    uint32_t open = t != CX_NONE && !(mob[t >> 6] >> (t & 63u) & 1u);
    if (alive && !rest && i >= 6u && dist < CX_NEAR && !open) {
        cx_shoot(words, cell, aim);
        mob[cell >> 6] |= 1ull << (cell & 63u);
        words[CXW_ZOMBIE + i] = w | (uint32_t)CX_SKELETON_REST << 16;
        return 0u;
    }
    uint32_t next = open ? t : cell;
    if (alive) cx_settle(mob, cell, next);
    words[CXW_ZOMBIE + i] = ((w & ~(4095u | CXM_ALIVE)) | next | alive << 31) - (rest ? 1u << 16 : 0u);
    return 0u;
}

static __attribute__((noinline)) uint32_t cx_trek(uint32_t *words, uint64_t *mob, const uint8_t *map, uint32_t player, uint32_t sleeping, uint32_t seed, uint32_t step, uint32_t luck, uint32_t bits, uint16x8_t aim, uint16x8_t gap, uint16x8_t face) {
    const uint16_t *plan = (const uint16_t *)&aim, *gaps = (const uint16_t *)&gap, *faces = (const uint16_t *)&face;
    uint32_t hurt = 0u;
    for (uint32_t lanes = bits & 255u; lanes; lanes &= lanes - 1u) {
        uint32_t i = (uint32_t)__builtin_ctz(lanes), to = plan[i];
        if (bits >> (8u + i) & 1u) {
            uint32_t harm = cx_odd(words, mob, map, player, i, to, gaps[i], faces[i], seed, step, luck, sleeping);
            hurt += harm;
            sleeping = harm ? 0u : sleeping;
            continue;
        }
        uint32_t w = words[CXW_ZOMBIE + i], c = w & 4095u, at = to & 4095u;
        uint32_t go1 = (uint32_t)!(to & CX_NONE) & (uint32_t)~(mob[at >> 6] >> (at & 63u));
        uint32_t next = go1 & 1u ? at : c;
        if (w >> 31) cx_settle(mob, c, next);
        words[CXW_ZOMBIE + i] = ((w & ~4095u) | next) - (w & (15u << 16) ? 1u << 16 : 0u);
    }
    return hurt;
}

static const uint16_t CX_HERD[88] = {
    0xFFFF, 0xFFFF, 0xFFFF, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0xFFFF, 0xFFFF,
    0xFFFF, 1, 0, 0, 0x40, 0x40, 0x40, 0x40,
    0, 0, 0xFFFF, 1, 0, 0, 0, 0,
    0, 0, 0xFFFB, 0xFFFF, 0xFFF6, 0xFFFF, 0xFFF1, 0xFFFF,
    0xFFEE, 0xFFFF, 0xFFEB, 0xFFFF, 0xFFE8, 0xFFFF, 0xFFE5, 0xFFFF,
    16, 16, 16, 1, 1, 1, 4, 4,
    0, 0, 0, 0, 0, 0, 0, 0xFFF8,
    0xFFFE, 0xFFFE, 0xFFFE, 0, 0, 0, 0, 0,
    3, 3, 3, 7, 7, 7, 3, 3,
    1, 2, 4, 8, 16, 32, 64, 128
};

static inline __attribute__((always_inline)) uint32_t cx_herd(uint32_t *words, uint64_t *mob, const uint8_t *map, uint32_t player, uint32_t sleeping, uint32_t seed, uint32_t step, const uint32_t *draw) {
    const uint16_t *K = (const uint16_t *)opaque(CX_HERD);
    const uint16x8_t one = vdupq_n_u16(1u), three = vdupq_n_u16(3u);
    const uint16x8_t zombie = vld1q_u16(K), skeleton = vld1q_u16(K + 8);
    const int8x16_t stepX = vreinterpretq_s8_u16(vld1q_u16(K + 16)), stepY = vreinterpretq_s8_u16(vld1q_u16(K + 24));
    uint32x4_t w0 = vld1q_u32(words + CXW_ZOMBIE), w1 = vld1q_u32(words + CXW_ZOMBIE + 4);
    uint16x8_t low = vuzp1q_u16(vreinterpretq_u16_u32(w0), vreinterpretq_u16_u32(w1)), high = vuzp2q_u16(vreinterpretq_u16_u32(w0), vreinterpretq_u16_u32(w1));
    uint16x8_t cell = vandq_u16(low, vdupq_n_u16(4095u)), x = vandq_u16(cell, vdupq_n_u16(63u)), y = vshrq_n_u16(cell, 6);
    int16x8_t dx = vsubq_s16(vdupq_n_s16((int16_t)(player & 63u)), vreinterpretq_s16_u16(x)), dy = vsubq_s16(vdupq_n_s16((int16_t)(player >> 6)), vreinterpretq_s16_u16(y));
    int16x8_t ax = vabsq_s16(dx), ay = vabsq_s16(dy);
    uint16x8_t d = vreinterpretq_u16_s16(vaddq_s16(ax, ay));
    uint32x4_t m = vdupq_n_u32(draw[1]);
    uint16x8_t r = vuzp1q_u16(vreinterpretq_u16_u32(vshlq_u32(m, vreinterpretq_s32_u16(vld1q_u16(K + 32)))), vreinterpretq_u16_u32(vshlq_u32(m, vreinterpretq_s32_u16(vld1q_u16(K + 40)))));
    uint16x8_t tie = vtstq_u16(r, vld1q_u16(K + 48));
    uint16x8_t vert = vorrq_u16(vcgtq_s16(ay, ax), vandq_u16(vceqq_s16(ay, ax), tie));
    uint16x8_t zero = vceqzq_u16(d);
    uint16x8_t toward = vbslq_u16(vert, vaddq_u16(three, vreinterpretq_u16_s16(vshrq_n_s16(dy, 15))), vaddq_u16(one, vreinterpretq_u16_s16(vshrq_n_s16(dx, 15))));
    toward = vbslq_u16(zero, vdupq_n_u16(4u), toward);
    uint16x8_t near10 = vcltq_u16(d, vdupq_n_u16(CX_FAR)), close = vcltq_u16(d, vdupq_n_u16(CX_NEAR));
    uint16x8_t chase = vandq_u16(vandq_u16(zombie, near10), vmvnq_u16(vceqq_u16(vandq_u16(r, three), three)));
    uint16x8_t edge = vandq_u16(skeleton, vornq_u16(close, near10));
    uint16x8_t byte = vandq_u16(vshlq_u16(vdupq_n_u16((uint16_t)draw[2]), vreinterpretq_s16_u16(vld1q_u16(K + 56))), vdupq_n_u16(255u));
    const uint16x8_t top = vdupq_n_u16((uint16_t)(cx_t24(CX_FLEE_CHANCE) >> 16));
    uint16x8_t flee = vandq_u16(edge, vcltq_u16(byte, top)), doubt = vandq_u16(edge, vceqq_u16(byte, top));
    uint16x8_t turn = vandq_u16(vbicq_u16(vandq_u16(flee, close), zero), one);
    uint16x8_t roam = vandq_u16(vshlq_u16(r, vreinterpretq_s16_u16(vld1q_u16(K + 64))), vld1q_u16(K + 72));
    uint16x8_t dir = vbslq_u16(vorrq_u16(chase, flee), veorq_u16(toward, turn), roam);
    uint8x16_t index = vreinterpretq_u8_u16(vmlaq_n_u16(vdupq_n_u16(0x0100u), dir, 0x0202u));
    uint16x8_t tx = vaddq_u16(x, vreinterpretq_u16_s8(vqtbl1q_s8(stepX, index))), ty = vaddq_u16(y, vreinterpretq_u16_s8(vqtbl1q_s8(stepY, index)));
    uint16x8_t t = vandq_u16(vsliq_n_u16(tx, ty, 6), vdupq_n_u16(4095u));
    uint16x8_t ok = vbicq_u16(vcleq_u16(vorrq_u16(tx, ty), vdupq_n_u16(63u)), vceqq_u16(t, vdupq_n_u16((uint16_t)player)));
    uint8x8_t cover = vdup_n_u8(0u);
    cover = vld1_lane_u8(map + vgetq_lane_u16(t, 0), cover, 0);
    cover = vld1_lane_u8(map + vgetq_lane_u16(t, 1), cover, 1);
    cover = vld1_lane_u8(map + vgetq_lane_u16(t, 2), cover, 2);
    cover = vld1_lane_u8(map + vgetq_lane_u16(t, 3), cover, 3);
    cover = vld1_lane_u8(map + vgetq_lane_u16(t, 4), cover, 4);
    cover = vld1_lane_u8(map + vgetq_lane_u16(t, 5), cover, 5);
    cover = vld1_lane_u8(map + vgetq_lane_u16(t, 6), cover, 6);
    cover = vld1_lane_u8(map + vgetq_lane_u16(t, 7), cover, 7);
    uint16x8_t walk = vtstq_u16(vshlq_u16(vdupq_n_u16((uint16_t)CX_WALK), vnegq_s16(vreinterpretq_s16_u16(vmovl_u8(cover)))), one);
    uint16x8_t go = vandq_u16(ok, walk);
    uint16x8_t alive = vtstq_u16(high, vdupq_n_u16(0x8000u)), rest = vtstq_u16(high, vdupq_n_u16(15u));
    uint16x8_t active = vorrq_u16(alive, vtstq_u16(low, vdupq_n_u16(0xF000u)));
    uint16x8_t lazy = vorrq_u16(vmvnq_u16(active), vbicq_u16(vbicq_u16(vcgeq_u16(d, vdupq_n_u16(CX_ALOOF)), alive), doubt));
    uint16x8_t moved = vbslq_u16(vandq_u16(active, go), t, cell);
    uint32x4_t v0 = vsubq_u32(vorrq_u32(vbicq_u32(w0, vdupq_n_u32(4095u)), vmovl_u16(vget_low_u16(moved))), vshll_n_u16(vget_low_u16(vandq_u16(rest, one)), 16));
    uint32x4_t v1 = vsubq_u32(vorrq_u32(vbicq_u32(w1, vdupq_n_u32(4095u)), vmovl_high_u16(moved)), vshll_high_n_u16(vandq_u16(rest, one), 16));
    vst1q_u32(words + CXW_ZOMBIE, vbslq_u32(vreinterpretq_u32_s32(vmovl_s16(vget_low_s16(vreinterpretq_s16_u16(lazy)))), v0, w0));
    vst1q_u32(words + CXW_ZOMBIE + 4, vbslq_u32(vreinterpretq_u32_s32(vmovl_high_s16(vreinterpretq_s16_u16(lazy))), v1, w1));
    uint16x8_t guard = vorrq_u16(vandq_u16(zombie, vceqq_u16(d, one)), vandq_u16(skeleton, vcltq_u16(d, vdupq_n_u16(CX_SHOT))));
    uint16x8_t odd = vorrq_u16(vandq_u16(alive, vorrq_u16(vcgeq_u16(d, vdupq_n_u16((uint16_t)CRAFTAX_KNOBS[0].despawn)), vbicq_u16(guard, rest))), doubt);
    const uint16x8_t order = vld1q_u16(K + 80);
    uint32_t bits = vaddvq_u16(vbicq_u16(vorrq_u16(order, vandq_u16(odd, vshlq_n_u16(order, 8))), lazy));
    uint16x8_t aim = vbslq_u16(go, t, vdupq_n_u16(CX_NONE));
    if (__builtin_expect(bits >> 8, 0)) return cx_trek(words, mob, map, player, sleeping, seed, step, draw[2], bits, aim, d, toward);
    const uint16_t *plan = (const uint16_t *)&aim;
    for (uint32_t lanes = bits; lanes; lanes &= lanes - 1u) {
        uint32_t i = (uint32_t)__builtin_ctz(lanes), to = plan[i], w = words[CXW_ZOMBIE + i], c = w & 4095u, at = to & 4095u;
        uint32_t go1 = (uint32_t)!(to & CX_NONE) & (uint32_t)~(mob[at >> 6] >> (at & 63u));
        uint32_t next = go1 & 1u ? at : c;
        if (w >> 31) cx_settle(mob, c, next);
        words[CXW_ZOMBIE + i] = ((w & ~4095u) | next) - (w & (15u << 16) ? 1u << 16 : 0u);
    }
    return 0u;
}

static inline __attribute__((always_inline)) void cx_mobs(const craftaxKnobs *k, cxLife *s, uint32_t *words, uint32_t seed, uint32_t step, const uint32_t *draw) {
    (void)k;
    uint32_t hurt = cx_herd(words, s->mob, s->map, s->player, s->sleeping, seed, step, draw);
    uint32_t woke = s->sleeping & (hurt != 0u), sleeping = s->sleeping & (hurt == 0u);
    if (__builtin_expect((words[CXW_ARROW] | words[CXW_ARROW + 1] | words[CXW_ARROW + 2]) & CXM_ALIVE, 0)) {
        uint32_t hits = cx_arrows(s->map, s->grass, s->path, s->mob, s->player, words);
        hurt += 2u * hits;
        sleeping = hits ? 0u : sleeping;
    }
    s->health -= (int32_t)hurt;
    s->sleeping = sleeping;
    s->ach |= woke << CXG_WAKE;
}

static inline __attribute__((always_inline)) float cx_tick(const craftaxKnobs *k, uint32_t seed, uint32_t step, uint32_t act, uint32_t *words, uint32_t *extra, const uint32_t *draw, const float *shared, uint32_t *over) {
    uint32_t buffer = words[CXW_GEN] >> 8 & 1u, pw = words[CXW_PLAYER], vit = words[CXW_VITALS];
    cxLife s = { cx_map(extra, buffer), cx_rows(extra, CXE_GRASS + 2u * CX_SIDE * buffer), cx_rows(extra, CXE_PATH + 2u * CX_SIDE * buffer), cx_rows(extra, CXE_MOB),
                 (uint8_t *)(words + CXW_INV), pw & 4095u, pw >> 12 & 3u, pw >> 14 & 1u, vit >> 4 & 15u, vit >> 8 & 15u, vit >> 12 & 15u, words[CXW_ACH],
                 (int32_t)(vit & 15u), (int32_t)words[CXW_HUNGER], (int32_t)words[CXW_THIRST], (int32_t)words[CXW_FATIGUE], (int32_t)words[CXW_RECOVER] };
    uint32_t ach0 = s.ach, inv0[3] = { words[CXW_INV], words[CXW_INV + 1], words[CXW_INV + 2] }, time = words[CXW_TIME];
    int32_t health0 = s.health;
    if (s.sleeping) act = CXA_NOOP;
    uint32_t abyss = 0u;
    if (act >= CXA_LEFT && act <= CXA_DOWN) {
        s.facing = act - 1u;
        uint32_t target = cx_ahead(s.player, s.facing), at = target & 4095u;
        uint32_t free = (uint32_t)(target != CX_NONE) & (CX_STEP_ON >> s.map[at] & 1u) & (uint32_t)~(s.mob[at >> 6] >> (at & 63u));
        abyss = (uint32_t)(target == CX_NONE) & !(s.facing & 1u) & (s.map[(s.player + (s.facing ? (CX_SIDE - 1u) * CX_SIDE : CX_SIDE - 1u)) & 4095u] == CXB_LAVA);
        s.player = free ? at : s.player;
    } else if (__builtin_expect(act >= CXA_DO && act != CXA_SLEEP, 0)) {
        cxLife t = s;
        cx_interact(k, &t, words, extra, seed, step, act, draw);
        s = t;
    }
    cx_mobs(k, &s, words, seed, step, draw);
    cx_spawn(k, &s, words, extra, seed, step, draw, (uint32_t)shared[k->dayLength + time % k->dayLength]);
    uint32_t *plant = extra + CXE_PLANT;
    if (plant[0] & CXM_ALIVE)
        for (uint32_t p = 0u; p < 10u && (plant[p] & CXM_ALIVE); p++) {
            uint32_t age = plant[p] >> 12 & 1023u;
            age = age < CX_RIPE_AGE ? age + 1u : age;
            plant[p] = (plant[p] & ~(1023u << 12)) | age << 12;
            if (age >= CX_RIPE_AGE) s.map[plant[p] & 4095u] = CXB_RIPE;
        }
    if (s.sleeping && s.energy >= CX_CAP) {
        s.sleeping = 0u;
        s.ach |= 1u << CXG_WAKE;
    }
    if (act == CXA_SLEEP && s.energy < CX_CAP) s.sleeping = 1u;
    s.hunger += s.sleeping ? 1 : 2;
    if (s.hunger > 50) { s.hunger = 0; s.food = s.food ? s.food - 1u : 0u; }
    s.thirst += s.sleeping ? 1 : 2;
    if (s.thirst > 40) { s.thirst = 0; s.drink = s.drink ? s.drink - 1u : 0u; }
    s.fatigue = s.sleeping ? (s.fatigue - 1 < 0 ? s.fatigue - 1 : 0) : s.fatigue + 1;
    if (s.fatigue > 30) { s.fatigue = 0; s.energy = s.energy ? s.energy - 1u : 0u; }
    if (s.fatigue < -10) { s.fatigue = 0; s.energy = s.energy < CX_CAP ? s.energy + 1u : CX_CAP; }
    s.recover += s.food && s.drink && (s.energy || s.sleeping) ? (s.sleeping ? 4 : 2) : (s.sleeping ? -1 : -2);
    if (s.recover > 50) { s.recover = 0; s.health = s.health < (int32_t)CX_CAP ? s.health + 1 : (int32_t)CX_CAP; }
    else if (s.recover < -30) { s.recover = 0; s.health -= 1; }
    if (abyss || s.map[s.player] == CXB_LAVA) s.health = 0;
    if (s.health < 0) s.health = 0;
    float reward = fmaf((float)(s.health - health0), 0.1f, (float)__builtin_popcount(s.ach & ~ach0));
    if (!s.health) cx_respawn(&s, words, extra);
    time++;
    uint32_t event = 0u;
    if (words[CXW_INV] != inv0[0] || words[CXW_INV + 1] != inv0[1] || words[CXW_INV + 2] != inv0[2]) {
        const uint8_t *before = (const uint8_t *)inv0;
        int32_t gain = -1, loss = -1;
        for (int32_t i = 0; i < CXI_COUNT; i++) {
            if (s.inv[i] > before[i]) gain = i;
            if (s.inv[i] < before[i]) loss = i;
        }
        event = gain >= 0 ? (uint32_t)(gain + 1) : (uint32_t)(-(loss + 1)) & 255u;
    }
    words[CXW_PLAYER] = s.player | s.facing << 12 | s.sleeping << 14;
    words[CXW_VITALS] = (uint32_t)s.health | s.food << 4 | s.drink << 8 | s.energy << 12;
    words[CXW_HUNGER] = (uint32_t)s.hunger;
    words[CXW_THIRST] = (uint32_t)s.thirst;
    words[CXW_FATIGUE] = (uint32_t)s.fatigue;
    words[CXW_RECOVER] = (uint32_t)s.recover;
    words[CXW_ACH] = s.ach;
    words[CXW_TIME] = time;
    words[CXW_MENU] = (words[CXW_MENU] & 0xFFFFu) | event << 16;
    *over = time >= k->ticks;
    return reward;
}

static inline uint32_t cx_extra(const uint32_t *words, uint32_t cell) {
    const uint32x4_t key = vdupq_n_u32(cell | CXM_ALIVE), mask = vdupq_n_u32(CXM_ALIVE | 4095u);
    const uint32x4_t weight[3] = { { 1u, 2u, 4u, 8u }, { 16u, 32u, 64u, 128u }, { 0u, 256u, 512u, 1024u } };
    uint32x4_t a = vandq_u32(vceqq_u32(vandq_u32(vld1q_u32(words + CXW_ZOMBIE), mask), key), weight[0]);
    uint32x4_t b = vandq_u32(vceqq_u32(vandq_u32(vld1q_u32(words + CXW_ZOMBIE + 4), mask), key), weight[1]);
    uint32x4_t c = vandq_u32(vceqq_u32(vandq_u32(vld1q_u32(words + CXW_COUNT - 4), mask), key), weight[2]);
    uint32_t found = vaddvq_u32(vaddq_u32(vaddq_u32(a, b), c)), mobs = found & 255u;
    return (mobs ? ((uint32_t)__builtin_ctz(mobs) + 3u) / 3u : 0u) + 4u * (found > 255u);
}

static uint32_t cx_code(const uint32_t *words, const uint8_t *map, uint32_t player, int dy, int dx) {
    int y = (int)(player >> 6) + dy, x = (int)(player & 63u) + dx;
    if (y < 0 || y > 63 || x < 0 || x > 63) return CXB_OUT;
    uint32_t cell = (uint32_t)(y * CX_SIDE + x);
    return map[cell] + 17u * cx_extra(words, cell);
}

static inline __attribute__((always_inline)) void cx_observe(const uint32_t *words, const uint32_t *extra, float *obs, const float *shared) {
    const uint8_t *map = cx_mapc(extra, words[CXW_GEN] >> 8 & 1u);
    uint32_t pw = words[CXW_PLAYER], player = pw & 4095u, px = player & 63u, py = player >> 6, facing = pw >> 12 & 3u, menu = words[CXW_MENU], slot = menu >> 8 & 127u;
    uint32_t code[CX_PATCH * CX_PATCH], crowd = 0u, volley = 0u;
    for (uint32_t i = 0u; i < CX_PATCH * CX_PATCH; i++) {
        uint32_t y = py + i / CX_PATCH - CX_PATCH / 2, x = px + i % CX_PATCH - CX_PATCH / 2;
        code[i] = y < CX_SIDE && x < CX_SIDE ? map[y * CX_SIDE + x] : CXB_OUT;
    }
    for (uint32_t m = CXW_ZOMBIE; m < CXW_COUNT; m++) {
        uint32_t w = words[m], dy = (w >> 6 & 63u) - py + CX_PATCH / 2, dx = (w & 63u) - px + CX_PATCH / 2, i = dy * CX_PATCH + dx;
        if (!(w & CXM_ALIVE) || dy >= CX_PATCH || dx >= CX_PATCH) continue;
        uint32_t *seen = m < CXW_ARROW ? &crowd : &volley;
        if (*seen >> i & 1u) continue;
        *seen |= 1u << i;
        code[i] += m < CXW_ARROW ? 17u * ((m - CXW_ZOMBIE + 3u) / 3u) : 68u;
    }
    for (uint32_t i = 0u; i < CX_PATCH * CX_PATCH; i++) obs[CXV_PATCH + i] = (float)code[i] * (1.0f / 64.0f);
    uint32x4_t vit = vandq_u32(vshlq_u32(vdupq_n_u32(words[CXW_VITALS]), (int32x4_t){ 0, -4, -8, -12 }), vdupq_n_u32(15u));
    vst1q_f32(obs + CXV_VITALS, vmulq_n_f32(vcvtq_f32_u32(vit), 0.1f));
    obs[CXV_LIGHT] = shared[words[CXW_TIME] % CRAFTAX_KNOBS[0].dayLength];
    obs[CXV_FACING] = (float)((int)(facing == 1u) - (int)(facing == 0u));
    obs[CXV_FACING + 1] = (float)((int)(facing == 3u) - (int)(facing == 2u));
    obs[CXV_SLEEP] = (float)(pw >> 14 & 1u);
    uint32_t value = slot < CXI_COUNT ? ((const uint8_t *)(words + CXW_INV))[slot] : cx_code(words, map, player, CX_READOUT[slot - CXI_COUNT][0], CX_READOUT[slot - CXI_COUNT][1]);
    const float32x4_t tally = { 1.0f / (float)(CX_MENU - 1), 1.0f / (float)(CX_SLOTS - 1), 1.0f / 64.0f, 1.0f / (float)CXI_COUNT }, count = { 1.0f / (float)(CX_MENU - 1), 1.0f / (float)(CX_SLOTS - 1), 0.1f, 1.0f / (float)CXI_COUNT };
    int32x4_t tail = { (int32_t)(menu & 15u), (int32_t)slot, (int32_t)value, (int8_t)(uint8_t)(menu >> 16) };
    vst1q_f32(obs + CXV_MENU, vmulq_f32(vcvtq_f32_s32(tail), slot < CXI_COUNT ? count : tally));
}

void craftax_reset(uint32_t instance, uint32_t seed, uint32_t *words, uint32_t *extra, uint32_t steps) {
    (void)instance; (void)steps;
    for (uint32_t k = 0u; k < CXW_COUNT; k++) words[k] = 0u;
    batchReset(&extra[CXE_TAG]);
    cx_prepare(extra);
    for (uint32_t stage = 0u; stage < CX_READY; stage++) cx_generate(seed, 0u, stage, extra, 0u);
    cx_begin(words, extra);
}

static __attribute__((noinline)) void cx_advance(uint32_t seed, uint32_t *words, uint32_t *extra, uint32_t over) {
    uint32_t gen = words[CXW_GEN], stage = gen & 255u, buffer = gen >> 8 & 1u;
    if (stage < CX_READY) {
        cx_generate(seed, words[CXW_GAME] + 1u, stage, extra, buffer ^ 1u);
        stage++;
    }
    if (over) {
        if (stage != CX_READY) __builtin_trap();
        buffer ^= 1u;
        words[CXW_GAME]++;
        cx_begin(words, extra);
        stage = 0u;
    }
    words[CXW_GEN] = stage | buffer << 8;
}

float craftax_step(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done, const float *shared) {
    (void)instance;
    const craftaxKnobs *k = &CRAFTAX_KNOBS[0];
    uint32_t slot = batchSlot(&extra[CXE_TAG], step);
    if (slot == ENV_BATCH) {
        cx_refill(seed, step, extra);
        slot = 0u;
    }
    float c = ctrl[0];
    uint32_t press = c >= 0.0f && c < 6.0f ? (uint32_t)c : 0u, menu = words[CXW_MENU], over = 0u;
    uint32_t cursor = menu & 15u, readout = menu >> 8 & 127u;
    readout = readout + 1u < CX_SLOTS ? readout + 1u : 0u;
    uint32_t act = press < 4u ? CX_MOVE_ACTION[press] : press == 4u ? CX_MENU_ACTION[cursor] : CXA_NOOP;
    words[CXW_MENU] = (press == 4u ? 0u : press == 5u ? (cursor + 1u < CX_MENU ? cursor + 1u : 0u) : cursor) | readout << 8;
    float reward = cx_tick(k, seed, step, act, words, extra, extra + CXE_BATCH + 4u * slot, shared, &over);
    if (__builtin_expect((words[CXW_GEN] & 255u) < CX_READY || over, 0)) cx_advance(seed, words, extra, over);
    *done = over;
    cx_observe(words, extra, obs, shared);
    return reward;
}

void craftax_observe(uint32_t instance, uint32_t seed, const uint32_t *words, const uint32_t *extra, uint32_t step, float *obs, const float *shared) {
    (void)instance; (void)seed; (void)step;
    cx_observe(words, extra, obs, shared);
}

uint32_t craftax_view(uint32_t instance, const uint32_t *words, const uint32_t *extra, const float *shared, uint32_t *out) {
    (void)instance; (void)shared;
    if (!out) return CX_CELLS / 8u;
    const uint8_t *map = cx_mapc(extra, words[CXW_GEN] >> 8 & 1u);
    for (uint32_t i = 0u; i < CX_CELLS / 8u; i++) {
        uint32_t packed = 0u;
        for (uint32_t j = 0u; j < 8u; j++) packed |= (uint32_t)((map[8u * i + j] - 1u) & 15u) << (4u * j);
        out[i] = packed;
    }
    return CX_CELLS / 8u;
}

static const uint8_t CXO_FACE[4] = { 2, 3, 0, 1 };
static const uint8_t CXO_BLOCK[CXT_COUNT] = { 0, CXB_TREE, CXB_WATER, CXB_STONE, CXB_COAL, CXB_IRON, CXB_DIAMOND, CXB_RIPE, 0, 0, 0, CXB_INVALID, CXB_PLANT, 0, 0 };
static const uint32_t CXO_PICKAXE_NEEDS[3] = {
    1u << CXG_WOOD_PICKAXE | 1u << CXG_STONE | 1u << CXG_PLACE_STONE | 1u << CXG_STONE_PICKAXE | 1u << CXG_STONE_SWORD | 1u << CXG_FURNACE | 1u << CXG_COAL |
        1u << CXG_IRON | 1u << CXG_DIAMOND | 1u << CXG_IRON_PICKAXE | 1u << CXG_IRON_SWORD,
    1u << CXG_STONE_PICKAXE | 1u << CXG_IRON | 1u << CXG_DIAMOND | 1u << CXG_IRON_PICKAXE | 1u << CXG_IRON_SWORD,
    1u << CXG_IRON_PICKAXE | 1u << CXG_DIAMOND
};
static const uint32_t CXO_EVERY = (1u << (CXG_IRON_SWORD + 1)) - 1u;

typedef struct {
    const uint32_t *words;
    uint32_t *extra, *state;
    const uint8_t *known, *inv;
    uint32_t player, facing, cursor, step, calm;
} cxoSelf;

static inline uint32_t cxo_sees(const uint32_t *words, uint32_t cell) {
    uint32_t player = words[CXW_PLAYER] & 4095u, slot = words[CXW_MENU] >> 8 & 127u;
    int dy = (int)(cell >> 6) - (int)(player >> 6), dx = (int)(cell & 63u) - (int)(player & 63u);
    if (abs(dy) <= CX_PATCH / 2 && abs(dx) <= CX_PATCH / 2) return 1u;
    return (uint32_t)(slot >= CXI_COUNT && dy == CX_READOUT[slot - CXI_COUNT][0] && dx == CX_READOUT[slot - CXI_COUNT][1]);
}

static uint32_t cxo_note(const uint8_t *map, uint8_t *known, int y, int x) {
    if (y < 0 || y >= CX_SIDE || x < 0 || x >= CX_SIDE) return 0u;
    uint32_t c = (uint32_t)(y * CX_SIDE + x), fresh = (uint32_t)(known[c] != map[c]) << map[c];
    known[c] = map[c];
    return fresh;
}

static uint32_t cxo_look(const uint8_t *map, uint8_t *known, const uint32_t *words) {
    uint32_t player = words[CXW_PLAYER] & 4095u, slot = words[CXW_MENU] >> 8 & 127u, fresh = 0u;
    int py = (int)(player >> 6), px = (int)(player & 63u);
    for (int dy = -CX_PATCH / 2; dy <= CX_PATCH / 2; dy++)
        for (int dx = -CX_PATCH / 2; dx <= CX_PATCH / 2; dx++) fresh |= cxo_note(map, known, py + dy, px + dx);
    if (slot >= CXI_COUNT) fresh |= cxo_note(map, known, py + CX_READOUT[slot - CXI_COUNT][0], px + CX_READOUT[slot - CXI_COUNT][1]);
    return fresh;
}

static inline uint64_t cxo_row(const uint8_t *row, uint8x16_t lo, uint8x16_t hi) {
    uint8x16_t m[4];
    for (uint32_t q = 0u; q < 4u; q++) {
        uint8x16_t b = vld1q_u8(row + 16u * q);
        m[q] = vtstq_u8(vqtbl1q_u8(lo, vandq_u8(b, vdupq_n_u8(15u))), vqtbl1q_u8(hi, vshrq_n_u8(b, 4)));
    }
    return cx_bits64(m[0], m[1], m[2], m[3]);
}

static inline uint32_t cxo_wall(const uint8_t *known, uint32_t cell) {
    return cell == CX_NONE || (known[cell] != CXB_INVALID && !(CX_WALK >> known[cell] & 1u) && known[cell] != CXB_LAVA);
}

static inline uint32_t cxo_mob(const uint32_t *words, uint32_t cell) {
    uint32_t found = 0u;
    for (uint32_t m = CXW_ZOMBIE; m < CXW_ARROW; m++) found |= (words[m] & (CXM_ALIVE | 4095u)) == (CXM_ALIVE | cell);
    return found && cxo_sees(words, cell);
}

static inline uint32_t cxo_free(const cxoSelf *o, uint32_t cell) { return cell != CX_NONE && (CX_WALK >> o->known[cell] & 1u) && !cxo_mob(o->words, cell); }

static float cxo_menu(const cxoSelf *o, uint32_t entry) {
    uint32_t c = o->cursor, ahead = (entry + CX_MENU - c) % CX_MENU;
    return c == entry || ((o->calm >> c & 1u) && entry + 1u < ahead) ? 4.0f : 5.0f;
}

static inline float cxo_face(const cxoSelf *o, uint32_t dir, uint32_t entry) { return o->facing == dir ? cxo_menu(o, entry) : (float)CXO_FACE[dir]; }

static void cxo_field(const cxoSelf *o, uint32_t goal, uint32_t dig) {
    const uint8_t *map = o->known;
    uint8_t *dist = (uint8_t *)(o->extra + CXO_DIST);
    uint16_t *queue = (uint16_t *)(o->extra + CXO_QUEUE);
    __builtin_memset(dist, 255, CX_CELLS);
    uint32_t head = 0u, tail = 0u, pass = CX_WALK | (dig ? 1u << CXB_STONE : 0u), want = goal == CXT_TABLE || goal == CXT_FORGE ? CXB_TABLE : CXO_BLOCK[goal];
    uint8_t passLo[16] = { 0 }, passHi[16] = { 0 }, wantLo[16] = { 0 }, wantHi[16] = { 0 };
    for (uint32_t b = 0u; b < 32u; b++) {
        if (pass >> b & 1u) { passLo[b & 15u] |= (uint8_t)(1u << (b >> 4)); passHi[b >> 4] |= (uint8_t)(1u << (b >> 4)); }
    }
    wantLo[want & 15u] = (uint8_t)(1u << (want >> 4));
    wantHi[want >> 4] = (uint8_t)(1u << (want >> 4));
    uint8x16_t pl = vld1q_u8(passLo), ph = vld1q_u8(passHi), wl = vld1q_u8(wantLo), wh = vld1q_u8(wantHi);
    uint64_t open[CX_SIDE], mark[CX_SIDE];
    uint32_t spot = goal == CXT_COW || goal == CXT_DEN || goal == CXT_TRAIL;
    for (uint32_t y = 0u; y < CX_SIDE; y++) {
        open[y] = cxo_row(map + y * CX_SIDE, pl, ph);
        mark[y] = spot ? 0u : cxo_row(map + y * CX_SIDE, wl, wh);
    }
    if (goal == CXT_COW)
        for (uint32_t c = CXW_COW; c < CXW_SKELETON; c++)
            if ((o->words[c] & CXM_ALIVE) && cxo_sees(o->words, o->words[c] & 4095u)) mark[(o->words[c] & 4095u) >> 6] |= 1ull << (o->words[c] & 63u);
    if (goal == CXT_COW && o->state[CXS_COW] && ((o->step - (o->state[CXS_COW] >> 12)) & 0xFFFFFu) < 64u) mark[(o->state[CXS_COW] & 4095u) >> 6] |= 1ull << (o->state[CXS_COW] & 63u);
    if (goal == CXT_DEN || goal == CXT_TRAIL) {
        uint32_t cell = o->state[goal == CXT_DEN ? CXS_DEN : CXS_TRAIL] & 4095u;
        mark[cell >> 6] |= 1ull << (cell & 63u);
    }
    for (uint32_t y = 0u; y < CX_SIDE; y++) {
        uint64_t seed;
        if (goal == CXT_TABLE || goal == CXT_FORGE) {
            uint64_t band = mark[y] | (y ? mark[y - 1u] : 0u) | (y < 63u ? mark[y + 1u] : 0u);
            seed = (band | band << 1 | band >> 1) & open[y];
        } else if (goal == CXT_DEN || goal == CXT_TRAIL) seed = mark[y] & open[y];
        else seed = (mark[y] << 1 | mark[y] >> 1 | (y ? mark[y - 1u] : 0u) | (y < 63u ? mark[y + 1u] : 0u)) & open[y];
        while (seed) {
            uint32_t c = y * CX_SIDE + (uint32_t)__builtin_ctzll(seed);
            seed &= seed - 1u;
            if (goal == CXT_FORGE && cx_near(map, c) != 3u) continue;
            dist[c] = 0u;
            queue[tail++] = (uint16_t)c;
        }
    }
    while (head < tail) {
        uint32_t c = queue[head++], d = dist[c];
        if (c == o->player || d >= 254u) break;
        for (uint32_t k = 0u; k < 4u; k++) {
            uint32_t n = cx_ahead(c, k);
            if (n == CX_NONE || dist[n] != 255u || !(open[n >> 6] >> (n & 63u) & 1u)) continue;
            dist[n] = (uint8_t)(d + 1u);
            queue[tail++] = (uint16_t)n;
        }
    }
}

static uint32_t cxo_roomy(const cxoSelf *o, uint32_t block) {
    uint16_t seen[16];
    uint32_t count = 1u;
    seen[0] = (uint16_t)o->player;
    for (uint32_t head = 0u; head < count && count < 16u; head++)
        for (uint32_t d = 0u; d < 4u && count < 16u; d++) {
            uint32_t n = cx_ahead(seen[head], d), fresh = n != CX_NONE && n != block && (CX_WALK >> o->known[n] & 1u);
            for (uint32_t k = 0u; k < count && fresh; k++) fresh = seen[k] != n;
            if (fresh) seen[count++] = (uint16_t)n;
        }
    return count >= 16u;
}

static uint32_t cxo_plot(const cxoSelf *o, uint32_t cell) {
    uint32_t den = o->state[CXS_DEN], e = den & 4095u, d = den >> 12 & 3u, x = cx_ahead(e, d), y = x == CX_NONE ? CX_NONE : cx_ahead(x, d);
    if (!(den >> 16)) return 1u;
    return cx_distance(cell, e) >= 2u && cx_distance(cell, e) <= 8u && (x == CX_NONE || cx_distance(cell, x) >= 2u) && (y == CX_NONE || cx_distance(cell, y) >= 2u);
}

static uint32_t cxo_fits(const cxoSelf *o, uint32_t cell, uint32_t entry) {
    return entry == 0u || entry == 4u || (cxo_roomy(o, cell) && (entry != 6u || !(o->state[CXS_DEN] >> 16) || cxo_plot(o, cell)));
}

static float cxo_place(const cxoSelf *o, uint32_t entry, uint32_t mask) {
    const uint8_t *map = o->known;
    uint32_t ahead = cx_ahead(o->player, o->facing);
    if (ahead != CX_NONE && (mask >> map[ahead] & 1u) && !cxo_mob(o->words, ahead) && cxo_fits(o, ahead, entry)) return cxo_menu(o, entry);
    for (uint32_t d = 0u; d < 4u; d++) {
        uint32_t n = cx_ahead(o->player, d);
        if (n == CX_NONE) continue;
        uint32_t view = cxo_free(o, n) ? cx_ahead(n, d) : n;
        if (view != CX_NONE && (mask >> map[view] & 1u) && !cxo_mob(o->words, view) && view != o->player && cxo_fits(o, view, entry)) return (float)CXO_FACE[d];
    }
    return -1.0f;
}

static uint32_t cxo_crops(const uint8_t *known) {
    uint32_t count = 0u;
    for (uint32_t c = 0u; c < CX_CELLS; c++) count += known[c] == CXB_PLANT || known[c] == CXB_RIPE;
    return count;
}

static float cxo_go(const cxoSelf *o, uint32_t goal, uint32_t dig) {
    uint32_t *state = o->state, player = o->player, step = o->step;
    const uint8_t *dist = (const uint8_t *)(o->extra + CXO_DIST), *map = o->known;
    if (state[CXS_BLOCKED + goal] > step) return -1.0f;
    uint32_t fleeting = goal == CXT_COW || goal == CXT_UNSEEN;
    uint32_t stale = (state[CXS_GOAL] & 255u) != goal || state[CXS_AGE] + (fleeting ? 3u : 24u) <= step || state[CXS_STALL] > 3u;
    if (stale) {
        cxo_field(o, goal, 0u);
        uint32_t digging = 0u;
        if (dist[player] == 255u && dig) {
            cxo_field(o, goal, 1u);
            digging = 1u;
        }
        state[CXS_GOAL] = goal | digging << 8;
        state[CXS_AGE] = step;
        state[CXS_STALL] = 0u;
    }
    uint32_t here = dist[player];
    if (here == 255u) {
        state[CXS_BLOCKED + goal] = step + (goal == CXT_COW ? 3u : 32u);
        state[CXS_GOAL] = 0u;
        return -1.0f;
    }
    if (here == 0u) {
        if (goal == CXT_TABLE || goal == CXT_FORGE || goal == CXT_DEN || goal == CXT_TRAIL) return -2.0f;
        for (uint32_t d = 0u; d < 4u; d++) {
            uint32_t n = cx_ahead(player, d), hit = 0u;
            if (n == CX_NONE) continue;
            if (goal == CXT_COW)
                for (uint32_t c = CXW_COW; c < CXW_SKELETON; c++) hit |= (o->words[c] & CXM_ALIVE) && (o->words[c] & 4095u) == n;
            else hit = map[n] == CXO_BLOCK[goal];
            if (hit) return cxo_face(o, d, 0u);
        }
        state[CXS_STALL] = 4u;
        return -1.0f;
    }
    uint32_t best = 4u, bestDist = here;
    for (uint32_t k = 0u; k < 4u; k++) {
        uint32_t d = (k + step) & 3u, n = cx_ahead(player, d);
        if (n == CX_NONE || dist[n] >= bestDist || cxo_mob(o->words, n)) continue;
        best = d;
        bestDist = dist[n];
    }
    if (best == 4u) {
        state[CXS_STALL]++;
        return -1.0f;
    }
    if (map[cx_ahead(player, best)] == CXB_STONE) return cxo_face(o, best, 0u);
    return (float)CXO_FACE[best];
}

static uint32_t cxo_flee(const cxoSelf *o, const uint32_t *hostile, uint32_t count) {
    uint32_t best = 4u, bestScore = 0u;
    for (uint32_t d = 0u; d <= 4u; d++) {
        uint32_t n = d < 4u ? cx_ahead(o->player, d) : o->player, low = 255u, exits = 0u;
        if (d < 4u && !cxo_free(o, n)) continue;
        for (uint32_t h = 0u; h < count; h++) low = cx_distance(n, hostile[h]) < low ? cx_distance(n, hostile[h]) : low;
        for (uint32_t k = 0u; k < 4u; k++) exits += cxo_free(o, cx_ahead(n, k)) || cx_ahead(n, k) == o->player;
        uint32_t score = 4u * low + (exits > 3u ? 3u : exits);
        if (d == 4u) return bestScore > score ? best : 4u;
        if (score > bestScore) { best = d; bestScore = score; }
    }
    return best;
}

static uint32_t cxo_enclosed(const uint8_t *known, uint32_t player) {
    uint32_t partner = CX_NONE;
    for (uint32_t d = 0u; d < 4u; d++) {
        uint32_t n = cx_ahead(player, d);
        if (cxo_wall(known, n)) continue;
        if (partner != CX_NONE || !(CX_WALK >> known[n] & 1u)) return 0u;
        partner = n;
    }
    if (partner == CX_NONE) return 1u;
    for (uint32_t d = 0u; d < 4u; d++) {
        uint32_t n = cx_ahead(partner, d);
        if (n != player && !cxo_wall(known, n)) return 0u;
    }
    return 1u;
}

static uint32_t cxo_room(const cxoSelf *o, uint32_t den) {
    const uint8_t *known = o->known;
    uint32_t e = den & 4095u, d = den >> 12 & 3u, x = cx_ahead(e, d), y = x == CX_NONE ? CX_NONE : cx_ahead(x, d);
    if (y == CX_NONE) return 0u;
    uint32_t f = cx_ahead(y, d), stones = o->inv[CXI_STONE], need = !cxo_wall(known, e) + !cxo_wall(known, f), inside = o->player == x || o->player == y;
    uint32_t wet = f != CX_NONE && known[f] == CXB_WATER;
    if (!inside && !(CX_WALK >> known[e] & 1u)) return 0u;
    for (uint32_t k = 0u; k < 2u; k++) {
        uint32_t c = k ? y : x;
        if (known[c] == CXB_STONE) stones++;
        else if (known[c] == CXB_WATER) need += !o->inv[CXI_STONE];
        else if (!(CX_WALK >> known[c] & 1u)) return 0u;
        for (uint32_t side = d < 2u ? 2u : 0u, s = 0u; s < 2u; s++) {
            uint32_t n = cx_ahead(c, side + s);
            if (!cxo_wall(known, n)) return 0u;
            wet |= n != CX_NONE && known[n] == CXB_WATER;
        }
    }
    return stones >= need ? 1u + wet : 0u;
}

static uint32_t cxo_seek(const cxoSelf *o) {
    uint8_t *dist = (uint8_t *)(o->extra + CXO_DIST);
    uint16_t *queue = (uint16_t *)(o->extra + CXO_QUEUE);
    __builtin_memset(dist, 255, CX_CELLS);
    uint32_t head = 0u, tail = 0u, best = 0u, cost = 255u;
    dist[o->player] = 0u;
    queue[tail++] = (uint16_t)o->player;
    while (head < tail) {
        uint32_t c = queue[head++];
        if (dist[c] >= cost) break;
        for (uint32_t d = 0u; d < 4u; d++) {
            uint32_t room = cxo_room(o, c | d << 12), price = dist[c] + (room == 2u ? 0u : 32u);
            if (room && price < cost) { best = c | d << 12 | 1u << 16; cost = price; }
        }
        if (dist[c] >= 64u) continue;
        for (uint32_t d = 0u; d < 4u; d++) {
            uint32_t n = cx_ahead(c, d);
            if (n == CX_NONE || dist[n] != 255u || !cxo_free(o, n)) continue;
            dist[n] = (uint8_t)(dist[c] + 1u);
            queue[tail++] = (uint16_t)n;
        }
    }
    return best;
}

static float cxo_hole(const cxoSelf *o) {
    const uint8_t *known = o->known;
    uint32_t den = o->state[CXS_DEN], e = den & 4095u, d = den >> 12 & 3u, back = d ^ 1u, x = cx_ahead(e, d), y = cx_ahead(x, d), f = cx_ahead(y, d);
    uint32_t p = o->player, ahead = cx_ahead(p, o->facing), openE = !cxo_wall(known, e), openF = !cxo_wall(known, f);
    if (p != e && p != x && p != y) return cxo_go(o, CXT_DEN, 0u);
    if (ahead != CX_NONE && cxo_mob(o->words, ahead)) return cxo_menu(o, 0u);
    if (known[x] == CXB_STONE || known[x] == CXB_WATER) return p == e ? cxo_face(o, d, known[x] == CXB_WATER ? 4u : 0u) : -1.0f;
    if (known[y] == CXB_STONE || known[y] == CXB_WATER) return p == x ? cxo_face(o, d, known[y] == CXB_WATER ? 4u : 0u) : p == e ? (float)CXO_FACE[d] : -1.0f;
    if (!openE && !openF) return -2.0f;
    if (o->cursor != 4u) return 5.0f;
    if (p == e) return (float)CXO_FACE[d];
    if (p == x) return openE && o->facing == back ? 4.0f : openE || openF ? (float)CXO_FACE[d] : -2.0f;
    return !openE && o->facing == d ? 4.0f : (float)CXO_FACE[back];
}

static inline uint32_t cxo_mine(const cxoSelf *o, uint32_t cell) {
    if (cell == CX_NONE) return 0u;
    uint32_t b = o->known[cell];
    return b == CXB_TREE || ((b == CXB_STONE || b == CXB_COAL) && o->inv[CXI_WOOD_PICKAXE]) || (b == CXB_IRON && o->inv[CXI_STONE_PICKAXE]) || (b == CXB_DIAMOND && o->inv[CXI_IRON_PICKAXE]);
}

static float cxo_exit(const cxoSelf *o) {
    uint32_t den = o->state[CXS_DEN], e = den & 4095u, way = den >> 12 & 3u, x = cx_ahead(e, way), y = x == CX_NONE ? CX_NONE : cx_ahead(x, way);
    if ((den >> 16) && y != CX_NONE && (o->player == x || o->player == y)) {
        uint32_t f = cx_ahead(y, way), atX = o->player == x;
        if (cxo_mine(o, e)) return atX && o->facing == (way ^ 1u) ? cxo_menu(o, 0u) : (float)CXO_FACE[atX ? way : way ^ 1u];
        if (cxo_mine(o, f)) return !atX && o->facing == way ? cxo_menu(o, 0u) : (float)CXO_FACE[atX ? way : way ^ 1u];
    }
    if (cxo_mine(o, cx_ahead(o->player, o->facing))) return cxo_menu(o, 0u);
    for (uint32_t d = 0u; d < 4u; d++)
        if (cxo_mine(o, cx_ahead(o->player, d))) return (float)CXO_FACE[d];
    for (uint32_t d = 0u; d < 4u; d++)
        if (cxo_free(o, cx_ahead(o->player, d))) return (float)CXO_FACE[d];
    return -1.0f;
}

static float cxo_wander(const cxoSelf *o, uint32_t seed) {
    uint32_t pick = philox4x32(seed, ENV_RNG_STREAM, o->step >> 4, 0u, CX_PURPOSE_WANDER, 0u).x >> 30, back = o->facing ^ 1u;
    for (uint32_t k = 0u; k < 12u; k++) {
        uint32_t d = (pick + k) & 3u, n = cx_ahead(o->player, d);
        if ((k < 8u && d == back) || !cxo_free(o, n) || (k < 4u && o->known[n] != CXB_GRASS)) continue;
        return (float)CXO_FACE[d];
    }
    return -1.0f;
}

static uint32_t cxo_trail(const cxoSelf *o, uint32_t seed) {
    uint8_t *dist = (uint8_t *)(o->extra + CXO_DIST);
    uint16_t *queue = (uint16_t *)(o->extra + CXO_QUEUE);
    __builtin_memset(dist, 255, CX_CELLS);
    uint32_t head = 0u, tail = 0u, count = 0u;
    dist[o->player] = 0u;
    queue[tail++] = (uint16_t)o->player;
    while (head < tail) {
        uint32_t c = queue[head++];
        count += dist[c] >= 8u && o->known[c] == CXB_GRASS;
        if (dist[c] >= 24u) continue;
        for (uint32_t d = 0u; d < 4u; d++) {
            uint32_t n = cx_ahead(c, d);
            if (n == CX_NONE || dist[n] != 255u || !cxo_free(o, n)) continue;
            dist[n] = (uint8_t)(dist[c] + 1u);
            queue[tail++] = (uint16_t)n;
        }
    }
    if (!count) return 0u;
    uint32_t pick = (uint32_t)(((uint64_t)philox4x32(seed, ENV_RNG_STREAM, o->step, 1u, CX_PURPOSE_WANDER, 0u).x * count) >> 32);
    for (uint32_t i = 0u; i < tail; i++) {
        uint32_t c = queue[i];
        if (dist[c] < 8u || o->known[c] != CXB_GRASS) continue;
        if (!pick--) return c | 1u << 16;
    }
    return 0u;
}

static float cxo_hunt(const cxoSelf *o, uint32_t seed) {
    uint32_t *state = o->state;
    if (!(state[CXS_TRAIL] >> 16) || state[CXS_TRAIL] >> 17 <= o->step >> 6) {
        state[CXS_TRAIL] = cxo_trail(o, seed);
        if (state[CXS_TRAIL]) state[CXS_TRAIL] |= ((o->step >> 6) + 1u) << 17;
        state[CXS_GOAL] = 0u;
        state[CXS_BLOCKED + CXT_TRAIL] = 0u;
    }
    float press = state[CXS_TRAIL] ? cxo_go(o, CXT_TRAIL, 0u) : -1.0f;
    if (press < 0.0f) state[CXS_TRAIL] = 0u;
    return press < 0.0f ? cxo_wander(o, seed) : press;
}

void craftax_oracle(uint32_t instance, uint32_t seed, uint32_t step, const uint32_t *words, uint32_t *extra, const float *obs, float *ctrl, const float *shared) {
    (void)instance; (void)obs;
    uint32_t *state = extra + CXO_STATE;
    uint8_t *known = (uint8_t *)(extra + CXO_MEMORY);
    uint32_t pw = words[CXW_PLAYER], player = pw & 4095u, facing = pw >> 12 & 3u, time = words[CXW_TIME];
    if (state[CXS_GAME] != words[CXW_GAME] + 1u) {
        __builtin_memset(state, 0, CXS_COUNT * sizeof(uint32_t));
        __builtin_memset(known, 0, CX_CELLS);
        state[CXS_GAME] = words[CXW_GAME] + 1u;
        state[CXS_PLANT] = CX_NONE;
    }
    if (state[CXS_LIVES] != words[CXW_DEATHS]) {
        __builtin_memset(state + CXS_IDLE, 0, (CXS_COUNT - CXS_IDLE) * sizeof(uint32_t));
        state[CXS_LIVES] = words[CXW_DEATHS];
    }
    uint32_t sown = state[CXS_PLANT] != CX_NONE ? known[state[CXS_PLANT]] : CXB_INVALID;
    uint32_t fresh = cxo_look(cx_mapc(extra, words[CXW_GEN] >> 8 & 1u), known, words);
    for (uint32_t goal = CXT_TREE; goal < CXT_COUNT; goal++) {
        uint32_t cue = goal == CXT_UNSEEN || goal == CXT_DEN ? ~0u : goal == CXT_TABLE || goal == CXT_FORGE ? 1u << CXB_TABLE | 1u << CXB_FURNACE : 1u << CXO_BLOCK[goal];
        if (fresh & cue) state[CXS_BLOCKED + goal] = 0u;
    }
    if (state[CXS_PLANT] != CX_NONE && cxo_sees(words, state[CXS_PLANT])) {
        uint32_t block = known[state[CXS_PLANT]];
        if (block != CXB_PLANT && block != CXB_RIPE) state[CXS_PLANT] = CX_NONE;
        else if (block == CXB_PLANT && (sown != CXB_PLANT || time >= state[CXS_SOWN] + CX_RIPE_AGE)) state[CXS_SOWN] = time;
    }
    if (pw >> 14 & 1u) {
        ctrl[0] = 4.0f;
        return;
    }
    const uint8_t *inv = (const uint8_t *)(words + CXW_INV);
    uint32_t vit = words[CXW_VITALS], health = vit & 15u, food = vit >> 4 & 15u, drink = vit >> 8 & 15u, energy = vit >> 12 & 15u, ach = words[CXW_ACH], cursor = words[CXW_MENU] & 15u;
    uint32_t front = cx_ahead(player, facing), blocked = front != CX_NONE && cxo_mob(words, front);
    uint32_t calm = 1u << 0 | 1u << 1 | 0x3Fu << 7 | (inv[CXI_WOOD] < 2u || blocked) << 3 | (!inv[CXI_STONE] || blocked) * (3u << 4) | (!inv[CXI_SAPLING] || blocked) << 6;
    cxoSelf self = { words, extra, state, known, inv, player, facing, cursor, step, calm }, *o = &self;
    uint32_t near = cx_near(known, player), light = shared[time % CRAFTAX_KNOBS[0].dayLength] > 0.5f;
    uint32_t damage = inv[CXI_IRON_SWORD] ? 5u : inv[CXI_STONE_SWORD] ? 3u : inv[CXI_WOOD_SWORD] ? 2u : 1u;
    uint32_t hostile[5], hostiles = 0u, foe = 0u, foes = 0u, herd = 0u, closest = 255u, dodge = 4u, wound = state[CXS_WOUND], struck = state[CXS_STRIKE];
    for (uint32_t m = CXW_ZOMBIE; m < CXW_COUNT; m++) {
        uint32_t w = words[m], cell = w & 4095u, gap = cx_distance(cell, player), seen = (w & CXM_ALIVE) && cxo_sees(words, cell);
        if (m >= CXW_ARROW) {
            uint32_t next = cx_ahead(cell, w >> 20 & 3u);
            if (seen && (next == player || (gap == 2u && next != CX_NONE && cx_ahead(next, w >> 20 & 3u) == player))) dodge = w >> 20 & 3u;
            continue;
        }
        if (m >= CXW_COW && m < CXW_SKELETON) {
            if (seen) state[CXS_COW] = cell | (step & 0xFFFFFu) << 12;
            herd |= seen;
            continue;
        }
        uint32_t shift = 4u * (m < CXW_COW ? m - CXW_ZOMBIE : m - CXW_SKELETON + 3u), dealt = (wound >> shift & 15u) + (struck == m + 1u ? damage : 0u);
        wound = (wound & ~(15u << shift)) | (seen ? (dealt < 15u ? dealt : 15u) : 0u) << shift;
        if (!seen) continue;
        hostile[hostiles++] = cell;
        if (m < CXW_COW) closest = gap < closest ? gap : closest;
        if (gap == 1u && (!foe || (m < CXW_COW && foe >= CXW_SKELETON))) foe = m;
        foes += gap == 1u && m < CXW_COW;
    }
    state[CXS_WOUND] = wound;
    state[CXS_STRIKE] = 0u;
    uint32_t drop = state[CXS_HEALTH] > health + 1u ? state[CXS_HEALTH] - 1u - health : 0u;
    state[CXS_HEALTH] = health + 1u;
    if (drop >= 2u && (drop >= 2u * foes || foes == 1u))
        for (uint32_t k = 0u; k < 3u; k++)
            if ((words[CXW_ZOMBIE + k] & CXM_ALIVE) && cx_distance(words[CXW_ZOMBIE + k] & 4095u, player) == 1u) state[CXS_BLOW + k] = step + 1u;
    if (hostiles || dodge != 4u) state[CXS_ALARM] = step + 1u;
    uint32_t flee = hostiles ? cxo_flee(o, hostile, hostiles) : 4u;
    if (foe) {
        uint32_t cell = words[foe] & 4095u, dir = cx_toward(player, cell, 0u), full = foe < CXW_COW ? CRAFTAX_KNOBS[0].zombieHealth : CRAFTAX_KNOBS[0].skeletonHealth;
        uint32_t dealt = wound >> (4u * (foe < CXW_COW ? foe - CXW_ZOMBIE : foe - CXW_SKELETON + 3u)) & 15u, left = dealt < full ? full - dealt : 1u;
        uint32_t span = (left + damage - 1u) / damage + (facing != dir) + (cursor != 0u), harm = 0u;
        for (uint32_t k = 0u; k < 3u; k++) {
            if (!(words[CXW_ZOMBIE + k] & CXM_ALIVE) || cx_distance(words[CXW_ZOMBIE + k] & 4095u, player) != 1u) continue;
            uint32_t blow = state[CXS_BLOW + k], free = blow && blow + 5u > step ? blow + 5u - step : 0u;
            harm += span > free + 1u ? 2u * ((span - free - 2u) / 6u + 1u) : 0u;
        }
        if (health > harm || flee == 4u) {
            float press = cxo_face(o, dir, 0u);
            if (press == 4.0f && cursor == 0u) state[CXS_STRIKE] = foe + 1u;
            ctrl[0] = press;
        } else ctrl[0] = (float)CXO_FACE[flee];
        return;
    }
    if (dodge != 4u)
        for (uint32_t k = 0u; k < 2u; k++) {
            uint32_t side = (dodge < 2u ? 2u : 0u) + ((k + step) & 1u);
            if (cxo_free(o, cx_ahead(player, side))) {
                ctrl[0] = (float)CXO_FACE[side];
                return;
            }
        }
    if (health <= 2u && closest <= 2u && flee != 4u) {
        ctrl[0] = (float)CXO_FACE[flee];
        return;
    }
    float press = -1.0f;
    const uint32_t spot = CX_STEP_ON & ~(1u << CXB_LAVA);
    uint32_t pick = inv[CXI_WOOD_PICKAXE] != 0u, stonePick = inv[CXI_STONE_PICKAXE] != 0u, ironPick = inv[CXI_IRON_PICKAXE] != 0u, craft[6];
    uint32_t sword = inv[CXI_WOOD_SWORD] || inv[CXI_STONE_SWORD] || inv[CXI_IRON_SWORD];
    craft[0] = !pick;
    craft[1] = !inv[CXI_WOOD_SWORD] && (!sword || (~ach >> CXG_WOOD_SWORD & 1u));
    craft[2] = !stonePick && (~ach & CXO_PICKAXE_NEEDS[1]) != 0u;
    craft[3] = !inv[CXI_STONE_SWORD] && (!inv[CXI_IRON_SWORD] || (~ach >> CXG_STONE_SWORD & 1u));
    craft[4] = !ironPick && (~ach & CXO_PICKAXE_NEEDS[2]) != 0u;
    craft[5] = !inv[CXI_IRON_SWORD] && (~ach >> CXG_IRON_SWORD & 1u);
    uint32_t stoneTools = craft[2] + craft[3], ironTools = craft[4] + craft[5];
    uint32_t woodNeed = (ach >> CXG_TABLE & 1u ? 0u : 2u) + craft[0] + craft[1] + stoneTools + ironTools + 2u;
    uint32_t woodWant = woodNeed < CX_CAP ? woodNeed : CX_CAP;
    if (drink <= (light ? 3u : 1u) || (drink <= 6u && state[CXS_BLOCKED + CXT_WATER] > step)) state[CXS_QUENCH] = 1u;
    if (drink >= CX_CAP) state[CXS_QUENCH] = 0u;
    if (food <= (light ? 3u : 1u)) state[CXS_HUNGRY] = 1u;
    if (food >= CX_CAP - 2u) state[CXS_HUNGRY] = 0u;
    if (pick && (energy <= 3u || health <= 5u || !light)) state[CXS_REST] = 1u;
    if (energy >= CX_CAP && health >= CX_CAP - 1u && light) state[CXS_REST] = 0u;
    uint32_t enclosed = cxo_enclosed(known, player), needy = state[CXS_QUENCH] || state[CXS_HUNGRY];
    herd |= state[CXS_COW] && ((step - (state[CXS_COW] >> 12)) & 0xFFFFFu) < 64u;
    if (enclosed && drink < CX_CAP)
        for (uint32_t d = 0u; d < 4u && press < 0.0f; d++)
            if (cx_ahead(player, d) != CX_NONE && known[cx_ahead(player, d)] == CXB_WATER) press = cxo_face(o, d, 0u);
    if (press < 0.0f && enclosed && drink < CX_CAP)
        for (uint32_t d = 0u; d < 4u && press < 0.0f; d++) {
            uint32_t n = cx_ahead(player, d);
            if (!cxo_free(o, n)) continue;
            for (uint32_t k = 0u; k < 4u; k++)
                if (cx_ahead(n, k) != CX_NONE && known[cx_ahead(n, k)] == CXB_WATER) press = (float)CXO_FACE[d];
        }
    if (press < 0.0f && enclosed && !needy) {
        if (energy < CX_CAP) press = cxo_menu(o, 2u);
        else if (state[CXS_REST] || state[CXS_IDLE] > step) press = cxo_menu(o, 1u);
    }
    if (press < 0.0f && enclosed && pick) press = cxo_exit(o);
    if (press < 0.0f && state[CXS_QUENCH]) press = cxo_go(o, CXT_WATER, 0u);
    if (press < 0.0f && state[CXS_QUENCH] && drink <= 2u) press = cxo_go(o, CXT_UNSEEN, pick);
    if (press < 0.0f && (state[CXS_HUNGRY] || (food <= 5u && light))) press = cxo_go(o, CXT_RIPE, 0u);
    if (press < 0.0f && state[CXS_HUNGRY] && herd) press = cxo_go(o, CXT_COW, 0u);
    if (press < 0.0f && state[CXS_HUNGRY]) press = cxo_hunt(o, seed);
    if (press < 0.0f && needy) press = cxo_go(o, CXT_UNSEEN, pick);
    if (press < 0.0f && !(ach >> CXG_TABLE & 1u)) {
        if (inv[CXI_WOOD] >= 2u) press = cxo_place(o, 3u, spot);
        else press = cxo_go(o, CXT_TREE, 0u);
    }
    for (uint32_t k = 0u; press < 0.0f && k < 2u; k++) {
        uint32_t tool = k ^ 1u;
        if (!craft[tool]) continue;
        if (inv[CXI_WOOD] && (near & 1u)) press = cxo_menu(o, 7u + tool);
        else if (inv[CXI_WOOD] >= 3u) press = cxo_place(o, 3u, spot);
        else if (inv[CXI_WOOD]) press = cxo_go(o, CXT_TABLE, 0u);
        if (press < 0.0f) press = cxo_go(o, CXT_TREE, 0u);
        if (press < 0.0f) press = cxo_go(o, CXT_UNSEEN, pick);
    }
    if (press < 0.0f && (state[CXS_REST] || state[CXS_IDLE] > step) && pick) {
        if (!(state[CXS_DEN] >> 16) || !cxo_room(o, state[CXS_DEN])) {
            state[CXS_DEN] = state[CXS_BLOCKED + CXT_DEN] > step ? 0u : cxo_seek(o);
            state[CXS_GOAL] = 0u;
            if (!state[CXS_DEN]) state[CXS_BLOCKED + CXT_DEN] = step + 16u;
        }
        if (state[CXS_DEN]) press = cxo_hole(o);
    }
    if (press < 0.0f && energy <= 1u && (!pick || !state[CXS_DEN]) && state[CXS_ALARM] + 16u <= step) press = cxo_menu(o, 2u);
    if (press < 0.0f && inv[CXI_WOOD] < woodWant && !state[CXS_ROAM]) {
        press = cxo_go(o, CXT_TREE, 0u);
        if (inv[CXI_WOOD] + 1u >= woodWant) state[CXS_ROAM] = 1u;
    }
    if (press < 0.0f && !(ach >> CXG_SAPLING & 1u)) {
        uint32_t ahead = cx_ahead(player, facing);
        press = ahead != CX_NONE && known[ahead] == CXB_GRASS && !cxo_mob(o->words, ahead) ? cxo_menu(o, 0u) : cxo_place(o, 0u, 1u << CXB_GRASS);
    }
    if (press < 0.0f && inv[CXI_SAPLING] && !(ach >> CXG_PLANT & 1u)) {
        press = cxo_place(o, 6u, 1u << CXB_GRASS);
        if (press == 4.0f && cursor == 6u) state[CXS_PLANT] = cx_ahead(player, facing);
    }
    if (press < 0.0f && !(ach >> CXG_DRINK & 1u)) press = cxo_go(o, CXT_WATER, 0u);
    if (press < 0.0f && !(ach >> CXG_COW & 1u) && herd) press = cxo_go(o, CXT_COW, 0u);
    if (press < 0.0f && light && health >= 5u && (state[CXS_DEN] >> 16) && (ach >> CXG_PLANT & 1u) && cxo_crops(known) < 10u) {
        if (cx_distance(player, state[CXS_DEN] & 4095u) > 8u) press = cxo_go(o, CXT_DEN, 0u);
        else if (inv[CXI_SAPLING]) press = cxo_place(o, 6u, 1u << CXB_GRASS);
        else {
            uint32_t ahead = cx_ahead(player, facing);
            press = ahead != CX_NONE && known[ahead] == CXB_GRASS && !cxo_mob(o->words, ahead) ? cxo_menu(o, 0u) : cxo_place(o, 0u, 1u << CXB_GRASS);
        }
    }
    if (press < 0.0f && pick) {
        if (!(ach >> CXG_PLACE_STONE & 1u) && inv[CXI_STONE]) press = cxo_place(o, 4u, CX_FLY);
        for (uint32_t k = 0u; press < 0.0f && k < 2u; k++) {
            uint32_t tool = k ^ 1u;
            if (!craft[2u + tool] || !inv[CXI_STONE] || !inv[CXI_WOOD]) continue;
            if (near & 1u) press = cxo_menu(o, 9u + tool);
            else if (inv[CXI_WOOD] >= 3u) press = cxo_place(o, 3u, spot);
            else press = cxo_go(o, CXT_TABLE, 0u);
        }
        if (press < 0.0f && !(ach >> CXG_FURNACE & 1u) && inv[CXI_STONE] && !stoneTools) {
            if (near & 1u) press = cxo_place(o, 5u, spot);
            else if (inv[CXI_WOOD] >= 2u + ironTools) press = cxo_place(o, 3u, spot);
            else press = cxo_go(o, CXT_TABLE, 0u);
        }
        uint32_t stoneWant = !(ach >> CXG_PLACE_STONE & 1u) + stoneTools + !(ach >> CXG_FURNACE & 1u) + ironTools + (ironTools ? 1u : 0u) + 2u;
        if (press < 0.0f && inv[CXI_STONE] < stoneWant) press = cxo_go(o, CXT_STONE, 1u);
    }
    if (press < 0.0f && pick && inv[CXI_COAL] < ironTools) press = cxo_go(o, CXT_COAL, 1u);
    if (press < 0.0f && stonePick && inv[CXI_IRON] < ironTools) press = cxo_go(o, CXT_IRON, 1u);
    if (press < 0.0f && ironTools && inv[CXI_IRON] && inv[CXI_COAL] && inv[CXI_STONE] && !inv[CXI_WOOD]) press = cxo_go(o, CXT_TREE, 0u);
    if (press < 0.0f && ironTools && inv[CXI_IRON] && inv[CXI_COAL] && inv[CXI_STONE] && inv[CXI_WOOD]) {
        uint32_t tool = craft[4] ? 0u : 1u;
        if (near == 3u) press = cxo_menu(o, 11u + tool);
        else if (!(near & 1u) && inv[CXI_WOOD] >= 2u + ironTools) press = cxo_place(o, 3u, spot);
        else if ((near & 1u) && inv[CXI_STONE] >= 1u + ironTools) press = cxo_place(o, 5u, spot);
        else press = cxo_go(o, CXT_FORGE, 0u);
        if (press < 0.0f && inv[CXI_WOOD] < 2u + ironTools) press = cxo_go(o, CXT_TREE, 0u);
    }
    if (press < 0.0f && ironPick && !(ach >> CXG_DIAMOND & 1u)) press = cxo_go(o, CXT_DIAMOND, 1u);
    uint32_t plant = state[CXS_PLANT];
    if (press < 0.0f && !(ach >> CXG_EAT_PLANT & 1u) && plant != CX_NONE && known[plant] == CXB_RIPE) press = cxo_go(o, CXT_RIPE, 0u);
    if (press < 0.0f && !(ach >> CXG_EAT_PLANT & 1u) && plant != CX_NONE && known[plant] == CXB_PLANT && time >= state[CXS_SOWN] + CX_RIPE_AGE) press = cxo_go(o, CXT_SOWN, 0u);
    if (press < 0.0f && light && drink < CX_CAP - 2u) press = cxo_go(o, CXT_WATER, 0u);
    if (press < 0.0f && light && food < CX_CAP - 3u && herd) press = cxo_go(o, CXT_COW, 0u);
    if (press < 0.0f && light && health > 5u && (ach & CXO_EVERY) != CXO_EVERY) press = cxo_go(o, CXT_UNSEEN, pick);
    if (press < 0.0f && pick && state[CXS_IDLE] <= step) {
        state[CXS_IDLE] = step + 512u;
        press = cxo_menu(o, 1u);
    }
    if (press < 0.0f) press = cxo_wander(o, seed);
    ctrl[0] = press < 0.0f ? cxo_menu(o, 1u) : press;
}
