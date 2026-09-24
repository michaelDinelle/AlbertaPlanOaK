#include <string.h>
#include "../env.h"

typedef struct { uint32_t game; float sticky; } minatarKnobs;

static const minatarKnobs MINATAR_KNOBS[5] = {
    { 0u, 0.1f },
    { 1u, 0.1f },
    { 2u, 0.1f },
    { 3u, 0.1f },
    { 4u, 0.1f }
};

enum { MINATAR_BREAKOUT, MINATAR_INVADERS, MINATAR_FREEWAY, MINATAR_ASTERIX, MINATAR_SEAQUEST };
enum { MINATAR_NOOP, MINATAR_LEFT, MINATAR_UP, MINATAR_RIGHT, MINATAR_DOWN, MINATAR_FIRE };
enum { MINATAR_TAG = 0, MINATAR_DRAWS = 1, MINATAR_RNG = MINATAR_DRAWS + 4 * ENV_BATCH };
enum { MINATAR_FISH = MINATAR_RNG, MINATAR_SUBS = MINATAR_FISH + 48, MINATAR_DIVERS = MINATAR_SUBS + 48, MINATAR_SHOTS = MINATAR_DIVERS + 4,
       MINATAR_BOMBS = MINATAR_SHOTS + 4, MINATAR_SEA = MINATAR_BOMBS + 48 };
enum { MINATAR_ENEMY_CAP = 48, MINATAR_DIVER_CAP = 4, MINATAR_SHOT_CAP = 4, MINATAR_BOMB_CAP = 48 };

static const uint8_t MINATAR_ACTION_COUNT[5] = { 3u, 4u, 3u, 5u, 6u };
static const uint8_t MINATAR_WORDS[5] = { 4u, 14u, 8u, 5u, 6u };
static const uint8_t MINATAR_ACTION_MAP[5][6] = { { 0u, 1u, 3u }, { 0u, 1u, 3u, 5u }, { 0u, 2u, 4u }, { 0u, 1u, 2u, 3u, 4u }, { 0u, 1u, 2u, 3u, 4u, 5u } };

static const float MINATAR_NINTH = 1.0f / 9.0f;

static uint32_t minatar_chosen(uint32_t game, const float *ctrl) {
    float c = ctrl[0];
    return c >= 0.0f && c < (float)MINATAR_ACTION_COUNT[game] ? MINATAR_ACTION_MAP[game][(uint32_t)c] : MINATAR_NOOP;
}

static __attribute__((noinline)) void minatar_refill(uint32_t seed, uint32_t step, uint32_t *draws) {
    u44 r = philox4(seed, ENV_RNG_STREAM, step + (lane4){ 0u, 1u, 2u, 3u }, (lane4)0u, (lane4)0u, (lane4)0u);
    memcpy(draws, &r.x, sizeof r.x);
    memcpy(draws + ENV_BATCH, &r.y, sizeof r.y);
    memcpy(draws + 2 * ENV_BATCH, &r.z, sizeof r.z);
    memcpy(draws + 3 * ENV_BATCH, &r.w, sizeof r.w);
}

static inline __attribute__((always_inline)) u4 minatar_dice(uint32_t seed, uint32_t step, uint32_t *extra) {
    uint32_t slot = batchSlot(&extra[MINATAR_TAG], step), *draws = extra + MINATAR_DRAWS;
    if (slot == ENV_BATCH) {
        minatar_refill(seed, step, draws);
        slot = 0u;
    }
    return (u4){ draws[slot], draws[ENV_BATCH + slot], draws[2 * ENV_BATCH + slot], draws[3 * ENV_BATCH + slot] };
}

static uint32_t minatar_act(const minatarKnobs *k, uint32_t chosen, uint32_t draw, uint8_t *last) {
    uint32_t action = chosen ^ ((chosen ^ *last) & (0u - (uint32_t)(draw < u01Bound(k->sticky))));
    *last = (uint8_t)action;
    return action;
}

typedef struct {
    uint32_t bricks;
    uint8_t paddle, x, y, trailX, trailY, strike, last;
    int8_t dx, dy;
    uint8_t spare[3];
} minatarBreakout;

enum { BREAKOUT_WALL = 0x3FFFFFFF };

static void breakout_start(minatarBreakout *s, uint32_t right) {
    s->bricks = BREAKOUT_WALL;
    s->paddle = 4u;
    s->x = right ? 9u : 0u;
    s->y = 3u;
    s->dx = right ? -1 : 1;
    s->dy = 1;
    s->trailX = s->x;
    s->trailY = s->y;
    s->strike = 0u;
}

static void breakout_obs(const minatarBreakout *s, float *obs) {
    obs[0] = (float)s->paddle * MINATAR_NINTH;
    obs[1] = (float)s->x * MINATAR_NINTH;
    obs[2] = (float)s->y * MINATAR_NINTH;
    obs[3] = (float)s->trailX * MINATAR_NINTH;
    obs[4] = (float)s->trailY * MINATAR_NINTH;
    uint32_t b = s->bricks;
    uint16x8_t lanes = { 1u, 2u, 4u, 8u, 16u, 32u, 64u, 128u }, one = vdupq_n_u16(1u);
    uint16x8_t low = vandq_u16(vtstq_u16(vdupq_n_u16((uint16_t)b), lanes), one);
    uint16x8_t mid = vandq_u16(vtstq_u16(vdupq_n_u16((uint16_t)(b >> 10)), lanes), vdupq_n_u16(2u));
    uint16x8_t high = vandq_u16(vtstq_u16(vdupq_n_u16((uint16_t)(b >> 20)), lanes), vdupq_n_u16(4u));
    uint16x8_t code = vorrq_u16(vorrq_u16(low, mid), high);
    vst1q_f32(obs + 5, vmulq_n_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(code))), 1.0f / 7.0f));
    vst1q_f32(obs + 9, vmulq_n_f32(vcvtq_f32_u32(vmovl_high_u16(code)), 1.0f / 7.0f));
    for (uint32_t x = 8u; x < 10u; x++) obs[5u + x] = (float)((b >> x & 1u) | (b >> (x + 9u) & 2u) | (b >> (x + 18u) & 4u)) * (1.0f / 7.0f);
}

static float breakout_fly(minatarBreakout *s) {
    float reward = 0.0f;
    int x = s->x, y = s->y, nx = x + s->dx, ny = y + s->dy;
    s->trailX = (uint8_t)x;
    s->trailY = (uint8_t)y;
    if (nx < 0 || nx > 9) {
        s->dx = (int8_t)-s->dx;
        nx = x;
    }
    if (ny < 0) {
        s->dy = (int8_t)-s->dy;
        ny = y;
    }
    uint32_t row = (uint32_t)ny - 1u, bit = row < 3u ? 1u << (row * 10u + (uint32_t)nx) : 0u;
    if (s->bricks & bit) {
        if (!s->strike) {
            s->bricks &= ~bit;
            reward = 1.0f;
            s->dy = (int8_t)-s->dy;
            ny = y;
            s->strike = 1u;
        }
    } else s->strike = 0u;
    s->x = (uint8_t)nx;
    s->y = (uint8_t)ny;
    return reward;
}

static uint32_t breakout_bounce(minatarBreakout *s) {
    if (!s->bricks) s->bricks = BREAKOUT_WALL;
    if (s->paddle == s->trailX) s->dy = (int8_t)-s->dy;
    else if (s->paddle == s->x) {
        s->dy = (int8_t)-s->dy;
        s->dx = (int8_t)-s->dx;
    } else return 1u;
    s->y = s->trailY;
    return 0u;
}

static float breakout_tick(minatarBreakout *s, uint32_t action, uint32_t *done) {
    s->paddle -= action == MINATAR_LEFT && s->paddle > 0u;
    s->paddle += action == MINATAR_RIGHT && s->paddle < 9u;
    float reward = breakout_fly(s);
    if (s->y == 9u) *done = breakout_bounce(s);
    return reward;
}

typedef unsigned __int128 minatarGrid;

typedef struct {
    uint32_t aliens[4], shots[4], bombs[4];
    uint8_t cannon, right, reload, march, volley, pace, last, spare;
} minatarInvaders;

static minatarGrid invaders_load(const uint32_t *w) { minatarGrid g; memcpy(&g, w, sizeof g); return g; }
static void invaders_store(uint32_t *w, minatarGrid g) { memcpy(w, &g, sizeof g); }
static minatarGrid invaders_cell(uint32_t x, uint32_t y) { return (minatarGrid)1u << (10u * y + x); }
static minatarGrid invaders_board(void) { return ((minatarGrid)1u << 100) - 1u; }
static minatarGrid invaders_column(uint32_t x) { return ((minatarGrid)0x4010040u << 64 | 0x1004010040100401u) << x; }
static minatarGrid invaders_wave(void) { return 0x3F0FC3F0FCu; }
static uint32_t invaders_count(minatarGrid g) { return (uint32_t)__builtin_popcountll((uint64_t)g) + (uint32_t)__builtin_popcountll((uint64_t)(g >> 64)); }
static uint32_t invaders_lowest(minatarGrid g) { return (uint64_t)g ? (uint32_t)__builtin_ctzll((uint64_t)g) : 64u + (uint32_t)__builtin_ctzll((uint64_t)(g >> 64)); }
static uint32_t invaders_columns(minatarGrid g) {
    g |= g >> 50;
    g |= g >> 30;
    return (uint32_t)(g | g >> 10 | g >> 20) & 0x3FFu;
}

static void invaders_start(minatarInvaders *s) {
    invaders_store(s->aliens, invaders_wave());
    memset(s->shots, 0, sizeof s->shots);
    memset(s->bombs, 0, sizeof s->bombs);
    s->cannon = 5u;
    s->right = 0u;
    s->reload = 0u;
    s->pace = 12u;
    s->march = 12u;
    s->volley = 10u;
}

static void invaders_obs(const minatarInvaders *s, float *obs) {
    minatarGrid aliens = invaders_load(s->aliens), shots = invaders_load(s->shots), bombs = invaders_load(s->bombs);
    uint32_t top = invaders_lowest(aliens) / 10u, left = (uint32_t)__builtin_ctz(invaders_columns(aliens) | 0x400u);
    obs[0] = (float)s->cannon * MINATAR_NINTH;
    obs[1] = (float)left * MINATAR_NINTH;
    obs[2] = (float)top * MINATAR_NINTH;
    obs[3] = (float)s->right;
    for (uint32_t r = 0u; r < 4u; r++) obs[4u + r] = top + r < 10u ? (float)(((uint32_t)(aliens >> (10u * (top + r))) & 0x3FFu) >> left & 0x3Fu) * (1.0f / 63.0f) : 0.0f;
    for (uint32_t b = 0u; b < 2u; b++) {
        uint32_t at = shots ? invaders_lowest(shots) : 0u;
        obs[8u + 2u * b] = shots ? (float)(at % 10u) * MINATAR_NINTH : -1.0f;
        obs[9u + 2u * b] = shots ? (float)(at / 10u) * MINATAR_NINTH : -1.0f;
        shots &= shots - 1u;
    }
    uint32_t at = bombs ? invaders_lowest(bombs) : 0u;
    obs[12] = bombs ? (float)(at % 10u) * MINATAR_NINTH : -1.0f;
    obs[13] = bombs ? (float)(at / 10u) * MINATAR_NINTH : -1.0f;
}

static uint32_t invaders_nearest(uint32_t columns, uint32_t x) {
    for (uint32_t d = 0u; d < 10u; d++) {
        if (x >= d && columns >> (x - d) & 1u) return x - d;
        if (x + d < 10u && columns >> (x + d) & 1u) return x + d;
    }
    return x;
}

static float invaders_tick(minatarInvaders *s, uint32_t action, uint32_t *done) {
    minatarGrid aliens = invaders_load(s->aliens), shots = invaders_load(s->shots), bombs = invaders_load(s->bombs), board = invaders_board();
    uint32_t fire = (uint32_t)(action == MINATAR_FIRE) & (uint32_t)(s->reload == 0u);
    shots |= invaders_cell(s->cannon, 9u) & (0u - (minatarGrid)fire);
    s->reload = (uint8_t)(s->reload | 5u * fire);
    s->cannon = (uint8_t)(s->cannon - ((uint32_t)(action == MINATAR_LEFT) & (uint32_t)(s->cannon > 0u)) + ((uint32_t)(action == MINATAR_RIGHT) & (uint32_t)(s->cannon < 9u)));
    shots >>= 10;
    bombs = bombs << 10 & board;
    minatarGrid cannon = invaders_cell(s->cannon, 9u);
    if ((bombs | aliens) & cannon) *done = 1u;
    if (s->march == 0u) {
        uint32_t alive = invaders_count(aliens);
        s->march = (uint8_t)(alive < s->pace ? alive : s->pace);
        if (aliens & invaders_column(s->right ? 9u : 0u)) {
            s->right ^= 1u;
            if (aliens >> 90) *done = 1u;
            aliens = (aliens << 10 | aliens >> 90) & board;
        } else aliens = s->right ? aliens << 1 : aliens >> 1;
        if (aliens & cannon) *done = 1u;
    }
    if (s->volley == 0u) {
        s->volley = 10u;
        uint32_t x = invaders_nearest(invaders_columns(aliens), s->cannon), y = 9u;
        while (!(aliens & invaders_cell(x, y))) y--;
        bombs |= invaders_cell(x, y);
    }
    minatarGrid kill = aliens & shots;
    float reward = (float)invaders_count(kill);
    aliens &= ~kill;
    shots &= ~kill;
    s->reload -= s->reload > 0u;
    s->march--;
    s->volley--;
    if (!aliens) {
        s->pace -= s->pace > 6u;
        aliens = invaders_wave();
    }
    invaders_store(s->aliens, aliens);
    invaders_store(s->shots, shots);
    invaders_store(s->bombs, bombs);
    return reward;
}

typedef struct {
    uint8_t carX[8], carTimer[8];
    int8_t carSpeed[8];
    uint8_t y, cool, last, spare;
    uint32_t clock;
} minatarFreeway;

enum { FREEWAY_LIMIT = 2500 };

static void freeway_speeds(minatarFreeway *s, uint32_t seed, uint32_t step, uint32_t counter) {
    u4 a = philox4x32(seed, ENV_RNG_STREAM, step, counter, 0u, 0u), b = philox4x32(seed, ENV_RNG_STREAM, step, counter + 1u, 0u, 0u);
    uint32_t w[8] = { a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w };
    for (uint32_t i = 0u; i < 8u; i++) {
        int speed = 1 + (int)(((uint64_t)w[i] * 5u) >> 32);
        s->carSpeed[i] = (int8_t)(w[i] & 1u ? speed : -speed);
        s->carTimer[i] = (uint8_t)speed;
    }
}

static void freeway_start(minatarFreeway *s, uint32_t seed, uint32_t step) {
    memset(s->carX, 0, sizeof s->carX);
    freeway_speeds(s, seed, step, 3u);
    s->y = 9u;
    s->cool = 3u;
    s->clock = FREEWAY_LIMIT;
}

static void freeway_obs(const minatarFreeway *s, float *obs) {
    obs[0] = (float)s->y * MINATAR_NINTH;
    uint16x8_t x = vmovl_u8(vld1_u8(s->carX));
    int16x8_t v = vmovl_s8(vld1_s8(s->carSpeed));
    vst1q_f32(obs + 1, vmulq_n_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(x))), MINATAR_NINTH));
    vst1q_f32(obs + 5, vmulq_n_f32(vcvtq_f32_u32(vmovl_high_u16(x)), MINATAR_NINTH));
    vst1q_f32(obs + 9, vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(v))), 0.2f));
    vst1q_f32(obs + 13, vmulq_n_f32(vcvtq_f32_s32(vmovl_high_s16(v)), 0.2f));
}

static float freeway_cross(minatarFreeway *s, uint32_t action) {
    uint32_t up = action == MINATAR_UP, down = action == MINATAR_DOWN, press = (up | down) & (uint32_t)(s->cool == 0u);
    s->y = (uint8_t)(s->y - (press & up & (uint32_t)(s->y > 0u)) + (press & down & (uint32_t)(s->y < 9u)));
    s->cool = (uint8_t)(s->cool | 3u * press);
    if (s->y) return 0.0f;
    s->y = 9u;
    return 1.0f;
}

static void freeway_traffic(minatarFreeway *s, uint32_t *done) {
    uint32_t lane = (uint32_t)s->y - 1u, hit = lane < 8u && s->carX[lane] == 4u;
    uint8x8_t timer = vld1_u8(s->carTimer), x = vld1_u8(s->carX), ten = vdup_n_u8(10u);
    int8x8_t speed = vld1_s8(s->carSpeed);
    uint8x8_t due = vceq_u8(timer, vdup_n_u8(0u)), ahead = vadd_u8(x, vbsl_u8(vcltz_s8(speed), vdup_n_u8(9u), vdup_n_u8(1u)));
    ahead = vbsl_u8(vcge_u8(ahead, ten), vsub_u8(ahead, ten), ahead);
    vst1_u8(s->carX, vbsl_u8(due, ahead, x));
    vst1_u8(s->carTimer, vbsl_u8(due, vreinterpret_u8_s8(vabs_s8(speed)), vsub_u8(timer, vdup_n_u8(1u))));
    if (hit || (lane < 8u && s->carX[lane] == 4u)) s->y = 9u;
    s->cool -= s->cool > 0u;
    if (s->clock-- == 0u) *done = 1u;
}

typedef struct {
    uint8_t row[8];
    uint8_t x, y, spawnTimer, moveTimer, spawnSpeed, moveSpeed, ramps, last;
    uint32_t clock;
} minatarAsterix;

enum { ASTERIX_LIVE = 0x80, ASTERIX_MOVED = 0x40, ASTERIX_GOLD = 0x20, ASTERIX_EAST = 0x10, ASTERIX_RAMP = 102 };

static void asterix_start(minatarAsterix *s) {
    memset(s->row, 0, sizeof s->row);
    s->x = 5u;
    s->y = 5u;
    s->spawnSpeed = 10u;
    s->moveSpeed = 5u;
    s->spawnTimer = 10u;
    s->moveTimer = 5u;
    s->ramps = 0u;
    s->clock = 0u;
}

static void asterix_obs(const minatarAsterix *s, float *obs) {
    obs[0] = (float)s->x * MINATAR_NINTH;
    obs[1] = (float)s->y * MINATAR_NINTH;
    uint16x8_t row = vmovl_u8(vld1_u8(s->row));
    for (uint32_t half = 0u; half < 2u; half++) {
        uint32x4_t e = half ? vmovl_high_u16(row) : vmovl_u16(vget_low_u16(row));
        float32x4_t kind = vbslq_f32(vtstq_u32(e, vdupq_n_u32(ASTERIX_GOLD)), vdupq_n_f32(1.0f), vdupq_n_f32(0.5f));
        kind = vbslq_f32(vtstq_u32(e, vdupq_n_u32(ASTERIX_EAST)), kind, vnegq_f32(kind));
        float32x4x2_t pair = { { vmulq_n_f32(vcvtq_f32_u32(vandq_u32(e, vdupq_n_u32(15u))), MINATAR_NINTH), vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(kind), vtstq_u32(e, e))) } };
        vst2q_f32(obs + 2u + 8u * half, pair);
    }
}

static float asterix_touch(minatarAsterix *s, uint32_t *done) {
    uint8_t *e = &s->row[s->y - 1u];
    if (!*e || (*e & 15u) != s->x) return 0.0f;
    if (!(*e & ASTERIX_GOLD)) {
        *done = 1u;
        return 0.0f;
    }
    *e = 0u;
    return 1.0f;
}

static void asterix_spawn(minatarAsterix *s, u4 dice) {
    uint32_t free = 0u;
    for (uint32_t r = 0u; r < 8u; r++) free |= (uint32_t)(s->row[r] == 0u) << r;
    if (free) {
        uint32_t pick = (uint32_t)(((uint64_t)dice.y * (uint32_t)__builtin_popcount(free)) >> 32);
        while (pick--) free &= free - 1u;
        s->row[__builtin_ctz(free)] = (uint8_t)(ASTERIX_LIVE | (dice.w < u01Bound(1.0f / 3.0f) ? ASTERIX_GOLD : 0u) | (dice.z >> 31 ? ASTERIX_EAST : 9u));
    }
    s->spawnTimer = s->spawnSpeed;
}

static void asterix_march(minatarAsterix *s) {
    s->moveTimer = s->moveSpeed;
    uint8x8_t e = vld1_u8(s->row), low = vdup_n_u8(15u);
    uint8x8_t x = vand_u8(vadd_u8(vand_u8(e, low), vbsl_u8(vtst_u8(e, vdup_n_u8(ASTERIX_EAST)), vdup_n_u8(1u), low)), low);
    uint8x8_t moved = vorr_u8(vorr_u8(vand_u8(e, vdup_n_u8(0xF0u)), vdup_n_u8(ASTERIX_MOVED)), x);
    vst1_u8(s->row, vand_u8(moved, vand_u8(vtst_u8(e, e), vcle_u8(x, vdup_n_u8(9u)))));
}

static void asterix_clock(minatarAsterix *s) {
    s->spawnTimer--;
    s->moveTimer--;
    if (++s->clock % ASTERIX_RAMP == 0u) {
        s->spawnSpeed -= s->spawnSpeed > 1u;
        s->moveSpeed -= s->moveSpeed > 1u && (s->ramps & 1u);
        s->ramps++;
    }
}

static float asterix_tick(minatarAsterix *s, uint32_t action, u4 dice, uint32_t *done) {
    if (s->spawnTimer == 0u) asterix_spawn(s, dice);
    s->x -= action == MINATAR_LEFT && s->x > 0u;
    s->x += action == MINATAR_RIGHT && s->x < 9u;
    s->y -= action == MINATAR_UP && s->y > 1u;
    s->y += action == MINATAR_DOWN && s->y < 8u;
    float reward = asterix_touch(s, done);
    if (s->moveTimer == 0u) {
        asterix_march(s);
        reward += asterix_touch(s, done);
    }
    asterix_clock(s);
    return reward;
}

typedef struct {
    int32_t oxygen, rescued;
    uint8_t x, y, right, reload, surfaced, enemyTimer, diverTimer, enemySpeed;
    uint8_t moveSpeed, ramps, last, fish, subs, divers, shots, bombs;
} minatarSeaquest;

enum { SEA_X = 0x0F, SEA_Y = 0xF0, SEA_EAST = 0x100, SEA_MOVED = 0x200, SEA_TIMER_SHIFT = 12, SEA_VOLLEY_SHIFT = 16, SEA_OXYGEN = 200 };

static uint32_t sea_x(uint32_t e) { return e & SEA_X; }
static uint32_t sea_y(uint32_t e) { return (e & SEA_Y) >> 4; }
static uint32_t sea_at(uint32_t e, uint32_t x, uint32_t y) { return (e & (SEA_X | SEA_Y)) == (x | y << 4); }
static uint32_t sea_timer(uint32_t e) { return e >> SEA_TIMER_SHIFT & 15u; }

static void sea_push(uint32_t *list, uint8_t *count, uint32_t cap, uint32_t e) {
    if (*count >= cap) __builtin_trap();
    list[(*count)++] = e;
}

static void sea_start(minatarSeaquest *s) {
    s->oxygen = SEA_OXYGEN;
    s->rescued = 0;
    s->x = 5u;
    s->y = 0u;
    s->right = 0u;
    s->reload = 0u;
    s->surfaced = 1u;
    s->enemySpeed = 20u;
    s->enemyTimer = 20u;
    s->diverTimer = 30u;
    s->moveSpeed = 5u;
    s->ramps = 0u;
    s->fish = s->subs = s->divers = s->shots = s->bombs = 0u;
}

static void sea_obs(const minatarSeaquest *s, const uint32_t *extra, float *obs) {
    uint32_t low[12] = { 0u }, mid[12] = { 0u }, high[12] = { 0u };
    const uint32_t *fish = extra + MINATAR_FISH, *subs = extra + MINATAR_SUBS, *divers = extra + MINATAR_DIVERS, *shots = extra + MINATAR_SHOTS, *bombs = extra + MINATAR_BOMBS;
    for (uint32_t i = 0u; i < s->fish; i++) low[sea_y(fish[i])] |= 1u << sea_x(fish[i]);
    for (uint32_t i = 0u; i < s->subs; i++) low[sea_y(subs[i])] |= 1u << (sea_x(subs[i]) + 10u);
    for (uint32_t i = 0u; i < s->divers; i++) mid[sea_y(divers[i])] |= 1u << sea_x(divers[i]);
    for (uint32_t i = 0u; i < s->shots; i++) high[sea_y(shots[i])] |= 1u << sea_x(shots[i]);
    for (uint32_t i = 0u; i < s->bombs; i++) high[sea_y(bombs[i])] |= 1u << (sea_x(bombs[i]) + 10u);
    const uint32_t *lists[3] = { fish, subs, divers };
    const uint8_t counts[3] = { s->fish, s->subs, s->divers };
    for (uint32_t l = 0u; l < 3u; l++)
        for (uint32_t i = 0u; i < counts[l]; i++) {
            uint32_t e = lists[l][i], back = sea_x(e) + (e & SEA_EAST ? 15u : 1u) & 15u;
            if (e & SEA_MOVED && back < 10u) mid[sea_y(e)] |= 1u << (back + 10u);
        }
    int32_t oxygen = s->oxygen > 0 ? s->oxygen : 0, rescued = s->rescued > 0 ? s->rescued : 0;
    obs[0] = (float)s->x * MINATAR_NINTH;
    obs[1] = (float)s->y * MINATAR_NINTH;
    obs[2] = (float)s->right;
    obs[3] = (float)(oxygen * 10 / SEA_OXYGEN) * 0.1f;
    obs[4] = (float)rescued * (1.0f / 6.0f);
    for (uint32_t y = 1u; y < 9u; y += 4u) {
        float32x4x3_t rows = { { vmulq_n_f32(vcvtq_f32_u32(vld1q_u32(low + y)), 0x1p-20f), vmulq_n_f32(vcvtq_f32_u32(vld1q_u32(mid + y)), 0x1p-20f),
                                 vmulq_n_f32(vcvtq_f32_u32(vld1q_u32(high + y)), 0x1p-20f) } };
        vst3q_f32(obs + 3u * y + 2u, rows);
    }
    obs[29] = (float)high[0] * 0x1p-10f;
}

static float sea_shoot(minatarSeaquest *s, uint32_t *extra, uint32_t x, uint32_t y) {
    uint32_t *fish = extra + MINATAR_FISH, *subs = extra + MINATAR_SUBS;
    for (uint32_t i = 0u; i < s->fish; i++)
        if (sea_at(fish[i], x, y)) {
            memmove(fish + i, fish + i + 1u, (s->fish - i - 1u) * sizeof *fish);
            s->fish--;
            return 1.0f;
        }
    for (uint32_t i = 0u; i < s->subs; i++)
        if (sea_at(subs[i], x, y)) {
            memmove(subs + i, subs + i + 1u, (s->subs - i - 1u) * sizeof *subs);
            s->subs--;
            return 1.0f;
        }
    return 0.0f;
}

static uint32_t sea_step(uint32_t e) {
    uint32_t x = sea_x(e) + (e & SEA_EAST ? 1u : 15u) & 15u;
    return (e & ~(uint32_t)SEA_X) | x;
}

static float sea_enemies(minatarSeaquest *s, uint32_t *extra, uint32_t *list, uint8_t *count, uint32_t armed, uint32_t *done) {
    float reward = 0.0f;
    uint32_t kept = 0u, n = *count, *shots = extra + MINATAR_SHOTS;
    for (uint32_t i = 0u; i < n; i++) {
        uint32_t e = list[i], alive = 1u;
        if (sea_at(e, s->x, s->y)) *done = 1u;
        uint32_t timer = sea_timer(e);
        if (timer == 0u) {
            e = sea_step(e | SEA_MOVED);
            e = (e & ~(15u << SEA_TIMER_SHIFT)) | (uint32_t)s->moveSpeed << SEA_TIMER_SHIFT;
            if (sea_x(e) > 9u) continue;
            if (sea_at(e, s->x, s->y)) *done = 1u;
            for (uint32_t b = 0u; b < s->shots; b++)
                if (sea_at(shots[b], sea_x(e), sea_y(e))) {
                    memmove(shots + b, shots + b + 1u, (s->shots - b - 1u) * sizeof *shots);
                    s->shots--;
                    reward += 1.0f;
                    alive = 0u;
                    break;
                }
        } else e -= 1u << SEA_TIMER_SHIFT;
        if (armed) {
            uint32_t volley = e >> SEA_VOLLEY_SHIFT & 15u;
            if (volley == 0u) {
                e = (e & ~(15u << SEA_VOLLEY_SHIFT)) | 10u << SEA_VOLLEY_SHIFT;
                sea_push(extra + MINATAR_BOMBS, &s->bombs, MINATAR_BOMB_CAP, e & (SEA_X | SEA_Y | SEA_EAST));
            } else e -= 1u << SEA_VOLLEY_SHIFT;
        }
        if (alive) list[kept++] = e;
    }
    *count = (uint8_t)kept;
    return reward;
}

static float sea_tick(minatarSeaquest *s, uint32_t *extra, uint32_t action, u4 dice, uint32_t *done) {
    uint32_t *fish = extra + MINATAR_FISH, *subs = extra + MINATAR_SUBS, *divers = extra + MINATAR_DIVERS, *shots = extra + MINATAR_SHOTS, *bombs = extra + MINATAR_BOMBS;
    float reward = 0.0f;
    if (s->enemyTimer == 0u) {
        uint32_t y = 1u + (dice.y >> 29), east = dice.y >> 28 & 1u, clash = 0u;
        for (uint32_t i = 0u; i < s->fish; i++) clash |= sea_y(fish[i]) == y && (fish[i] >> 8 & 1u) != east;
        for (uint32_t i = 0u; i < s->subs; i++) clash |= sea_y(subs[i]) == y && (subs[i] >> 8 & 1u) != east;
        if (!clash) {
            uint32_t e = (east ? 0u : 9u) | y << 4 | (east ? SEA_EAST : 0u) | (uint32_t)s->moveSpeed << SEA_TIMER_SHIFT;
            if (dice.y << 4 < u01Bound(1.0f / 3.0f)) sea_push(subs, &s->subs, MINATAR_ENEMY_CAP, e | 10u << SEA_VOLLEY_SHIFT);
            else sea_push(fish, &s->fish, MINATAR_ENEMY_CAP, e);
        }
        s->enemyTimer = s->enemySpeed;
    }
    if (s->diverTimer == 0u) {
        uint32_t y = 1u + (dice.z >> 29), east = dice.z >> 28 & 1u;
        sea_push(divers, &s->divers, MINATAR_DIVER_CAP, (east ? 0u : 9u) | y << 4 | (east ? SEA_EAST : 0u) | 5u << SEA_TIMER_SHIFT);
        s->diverTimer = 30u;
    }
    uint32_t fire = (uint32_t)(action == MINATAR_FIRE) & (uint32_t)(s->reload == 0u), left = action == MINATAR_LEFT, right = action == MINATAR_RIGHT;
    if (s->shots >= MINATAR_SHOT_CAP - 1u) __builtin_trap();
    shots[s->shots] = s->x | (uint32_t)s->y << 4 | (uint32_t)s->right << 8;
    s->shots = (uint8_t)(s->shots + fire);
    s->reload = (uint8_t)(s->reload | 5u * fire);
    s->x = (uint8_t)(s->x - (left & (uint32_t)(s->x > 0u)) + (right & (uint32_t)(s->x < 9u)));
    s->right = (uint8_t)(right | (s->right & (left ^ 1u)));
    s->y = (uint8_t)(s->y - ((uint32_t)(action == MINATAR_UP) & (uint32_t)(s->y > 0u)) + ((uint32_t)(action == MINATAR_DOWN) & (uint32_t)(s->y < 8u)));
    uint32_t kept = 0u;
    for (uint32_t i = 0u; i < s->shots; i++) {
        uint32_t b = sea_step(shots[i]);
        if (sea_x(b) > 9u) continue;
        float hit = sea_shoot(s, extra, sea_x(b), sea_y(b));
        reward += hit;
        if (hit == 0.0f) shots[kept++] = b;
    }
    s->shots = (uint8_t)kept;
    kept = 0u;
    for (uint32_t i = 0u; i < s->divers; i++) {
        uint32_t e = divers[i];
        if (sea_at(e, s->x, s->y) && s->rescued < 6) {
            s->rescued++;
            continue;
        }
        if (sea_timer(e) == 0u) {
            e = sea_step(e | SEA_MOVED);
            e = (e & ~(15u << SEA_TIMER_SHIFT)) | 5u << SEA_TIMER_SHIFT;
            if (sea_x(e) > 9u) continue;
            if (sea_at(e, s->x, s->y) && s->rescued < 6) {
                s->rescued++;
                continue;
            }
        } else e -= 1u << SEA_TIMER_SHIFT;
        divers[kept++] = e;
    }
    s->divers = (uint8_t)kept;
    reward += sea_enemies(s, extra, subs, &s->subs, 1u, done);
    reward += sea_enemies(s, extra, fish, &s->fish, 0u, done);
    kept = 0u;
    for (uint32_t i = 0u; i < s->bombs; i++) {
        uint32_t b = bombs[i];
        if (sea_at(b, s->x, s->y)) *done = 1u;
        b = sea_step(b);
        if (sea_x(b) > 9u) continue;
        if (sea_at(b, s->x, s->y)) *done = 1u;
        bombs[kept++] = b;
    }
    s->bombs = (uint8_t)kept;
    if (s->y > 0u) {
        s->surfaced = 0u;
        if (--s->oxygen < 0) *done = 1u;
    } else if (!s->surfaced) {
        s->surfaced = 1u;
        if (s->rescued == 0) *done = 1u;
        else {
            if (s->oxygen == 0) *done = 1u;
            if (s->rescued == 6) {
                reward += (float)(s->oxygen * 10 / SEA_OXYGEN);
                s->rescued = 0;
            }
            s->rescued--;
            s->oxygen = SEA_OXYGEN;
            s->enemySpeed -= s->enemySpeed > 1u;
            s->moveSpeed -= s->moveSpeed > 2u && (s->ramps & 1u);
            s->ramps++;
        }
    }
    s->reload -= s->reload > 0u;
    s->enemyTimer--;
    s->diverTimer--;
    return reward;
}

_Static_assert(sizeof(minatarBreakout) == 4 * sizeof(uint32_t), "");
_Static_assert(sizeof(minatarInvaders) == 14 * sizeof(uint32_t), "");
_Static_assert(sizeof(minatarFreeway) == 8 * sizeof(uint32_t), "");
_Static_assert(sizeof(minatarAsterix) == 5 * sizeof(uint32_t), "");
_Static_assert(sizeof(minatarSeaquest) == 6 * sizeof(uint32_t), "");

uint32_t minatar_extraWords(uint32_t instance, uint32_t steps) {
    (void)steps;
    return MINATAR_KNOBS[instance].game == MINATAR_SEAQUEST ? 2u * MINATAR_SEA : MINATAR_RNG;
}

static void minatar_start(uint32_t game, uint32_t seed, uint32_t step, uint32_t *words, uint32_t side) {
    switch (game) {
        case MINATAR_BREAKOUT: breakout_start((minatarBreakout *)words, side); return;
        case MINATAR_INVADERS: invaders_start((minatarInvaders *)words); return;
        case MINATAR_FREEWAY: freeway_start((minatarFreeway *)words, seed, step); return;
        case MINATAR_ASTERIX: asterix_start((minatarAsterix *)words); return;
        default: sea_start((minatarSeaquest *)words); return;
    }
}

static void minatar_obs(uint32_t game, const uint32_t *words, const uint32_t *extra, float *obs) {
    switch (game) {
        case MINATAR_BREAKOUT: breakout_obs((const minatarBreakout *)words, obs); return;
        case MINATAR_INVADERS: invaders_obs((const minatarInvaders *)words, obs); return;
        case MINATAR_FREEWAY: freeway_obs((const minatarFreeway *)words, obs); return;
        case MINATAR_ASTERIX: asterix_obs((const minatarAsterix *)words, obs); return;
        default: sea_obs((const minatarSeaquest *)words, extra, obs); return;
    }
}

void minatar_reset(uint32_t instance, uint32_t seed, uint32_t *words, uint32_t *extra, uint32_t steps) {
    (void)steps;
    uint32_t game = MINATAR_KNOBS[instance].game;
    memset(words, 0, MINATAR_WORDS[game] * sizeof(uint32_t));
    batchReset(&extra[MINATAR_TAG]);
    minatar_start(game, seed, 0u, words, philox4x32(seed, ENV_RNG_STREAM, 0u, 1u, 0u, 0u).x >> 31);
}

static inline __attribute__((always_inline)) float minatar_play(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done) {
    const minatarKnobs *k = &MINATAR_KNOBS[instance];
    uint32_t chosen = minatar_chosen(k->game, ctrl);
    u4 dice = minatar_dice(seed, step, extra);
    float reward;
    *done = 0u;
    switch (k->game) {
        case MINATAR_BREAKOUT: {
            minatarBreakout *s = (minatarBreakout *)words;
            reward = breakout_tick(s, minatar_act(k, chosen, dice.x, &s->last), done);
            if (*done) breakout_start(s, dice.y >> 31);
            breakout_obs(s, obs);
            return reward;
        }
        case MINATAR_INVADERS: {
            minatarInvaders *s = (minatarInvaders *)words;
            reward = invaders_tick(s, minatar_act(k, chosen, dice.x, &s->last), done);
            if (*done) invaders_start(s);
            invaders_obs(s, obs);
            return reward;
        }
        case MINATAR_FREEWAY: {
            minatarFreeway *s = (minatarFreeway *)words;
            reward = freeway_cross(s, minatar_act(k, chosen, dice.x, &s->last));
            if (reward != 0.0f) freeway_speeds(s, seed, step, 1u);
            freeway_traffic(s, done);
            if (*done) freeway_start(s, seed, step);
            freeway_obs(s, obs);
            return reward;
        }
        case MINATAR_ASTERIX: {
            minatarAsterix *s = (minatarAsterix *)words;
            reward = asterix_tick(s, minatar_act(k, chosen, dice.x, &s->last), dice, done);
            if (*done) asterix_start(s);
            asterix_obs(s, obs);
            return reward;
        }
        default: {
            minatarSeaquest *s = (minatarSeaquest *)words;
            reward = sea_tick(s, extra, minatar_act(k, chosen, dice.x, &s->last), dice, done);
            if (*done) sea_start(s);
            sea_obs(s, extra, obs);
            return reward;
        }
    }
}

float minatar_step(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done, const float *shared) {
    (void)shared;
    switch (instance) {
        case 0u: return minatar_play(0u, seed, step, ctrl, words, extra, obs, done);
        case 1u: return minatar_play(1u, seed, step, ctrl, words, extra, obs, done);
        case 2u: return minatar_play(2u, seed, step, ctrl, words, extra, obs, done);
        case 3u: return minatar_play(3u, seed, step, ctrl, words, extra, obs, done);
        case 4u: return minatar_play(4u, seed, step, ctrl, words, extra, obs, done);
        default: __builtin_trap();
    }
}

void minatar_observe(uint32_t instance, uint32_t seed, const uint32_t *words, const uint32_t *extra, uint32_t step, float *obs, const float *shared) {
    (void)seed; (void)step; (void)shared;
    minatar_obs(MINATAR_KNOBS[instance].game, words, extra, obs);
}

static u4 minatar_future(uint32_t seed, uint32_t step) { return philox4x32(seed, ENV_RNG_STREAM, step, 0u, 0u, 0u); }

static uint32_t minatar_stuck(const minatarKnobs *k, uint32_t seed, uint32_t first, uint32_t n) {
    uint32_t mask = 0u, bound = u01Bound(k->sticky);
    for (uint32_t i = 0u; i < n && i < 32u; i += 4u) {
        u44 r = philox4(seed, ENV_RNG_STREAM, first + i + (lane4){ 0u, 1u, 2u, 3u }, (lane4)0u, (lane4)0u, (lane4)0u);
        lane4 hit = (lane4)(r.x < bound) & 1u;
        mask |= (hit.x | hit.y << 1 | hit.z << 2 | hit.w << 3) << i;
    }
    return n < 32u ? mask & ((1u << n) - 1u) : mask;
}

static uint32_t breakout_score(minatarBreakout sim) {
    uint32_t hits = 0u, first = 64u;
    for (uint32_t t = 1u; t < 64u && sim.y != 9u; t++)
        if (breakout_fly(&sim) != 0.0f) {
            hits++;
            first = first < t ? first : t;
        }
    return (64u - hits) << 6 | first;
}

static uint32_t breakout_plan(uint32_t goal, uint32_t stuck, uint32_t horizon, uint32_t paddle, uint32_t last, uint32_t *choice) {
    uint32_t m[3] = { goal, goal, goal };
    for (uint32_t j = horizon; j >= 2u; j--) {
        uint32_t still = m[0], left = (m[1] << 1 | (m[1] & 1u)) & 0x3FFu, right = m[2] >> 1 | (m[2] & 0x200u);
        if (j <= 32u && stuck >> (j - 1u) & 1u) {
            m[0] = still;
            m[1] = left;
            m[2] = right;
        } else m[0] = m[1] = m[2] = still | left | right;
    }
    uint32_t order[3] = { goal >> paddle & 1u ? 0u : (goal & ((1u << paddle) - 1u) ? 1u : 2u), 0u, 0u };
    order[1] = order[0] == 0u ? 1u : 0u;
    order[2] = 3u - order[0] - order[1];
    for (uint32_t i = 0u; i < 3u; i++) {
        uint32_t e = stuck & 1u ? last : order[i];
        uint32_t p = e == 1u ? (paddle ? paddle - 1u : 0u) : e == 2u ? (paddle < 9u ? paddle + 1u : 9u) : paddle;
        if (m[e] >> p & 1u) {
            *choice = e;
            return 1u;
        }
        if (stuck & 1u) break;
    }
    return 0u;
}

static uint32_t breakout_oracle(const minatarKnobs *k, const minatarBreakout *s, uint32_t seed, uint32_t step) {
    static const uint8_t CODE[3] = { MINATAR_NOOP, MINATAR_LEFT, MINATAR_RIGHT };
    minatarBreakout sim = *s;
    uint32_t horizon = 0u;
    while (horizon < 48u && sim.y != 9u) {
        breakout_fly(&sim);
        horizon++;
    }
    uint32_t paddle = s->paddle, last = s->last == MINATAR_LEFT ? 1u : s->last == MINATAR_RIGHT ? 2u : 0u, choice;
    if (sim.y != 9u) return s->x < paddle ? MINATAR_LEFT : s->x > paddle ? MINATAR_RIGHT : MINATAR_NOOP;
    uint32_t from = sim.trailX, to = sim.x, stuck = minatar_stuck(k, seed, step + 1u, horizon);
    minatarBreakout keep = sim, turn = sim;
    keep.paddle = (uint8_t)from;
    turn.paddle = (uint8_t)to;
    breakout_bounce(&keep);
    uint32_t prefer = from;
    if (to != from) {
        breakout_bounce(&turn);
        uint32_t a = breakout_score(keep), b = breakout_score(turn);
        prefer = a < b || (a == b && (minatar_future(seed, step).z & 1u)) ? from : to;
    }
    if (breakout_plan(1u << prefer, stuck, horizon, paddle, last, &choice)) return CODE[choice];
    if (breakout_plan(1u << from | 1u << to, stuck, horizon, paddle, last, &choice)) return CODE[choice];
    return prefer < paddle ? MINATAR_LEFT : prefer > paddle ? MINATAR_RIGHT : MINATAR_NOOP;
}

static uint32_t invaders_march(minatarGrid *aliens, minatarGrid *shots, uint8_t *march, uint8_t *right, uint32_t pace) {
    uint32_t killed;
    *shots >>= 10;
    if (*march == 0u) {
        uint32_t alive = invaders_count(*aliens);
        *march = (uint8_t)(alive < pace ? alive : pace);
        if (*aliens & invaders_column(*right ? 9u : 0u)) {
            *right ^= 1u;
            *aliens = (*aliens << 10 | *aliens >> 90) & invaders_board();
        } else *aliens = *right ? *aliens << 1 : *aliens >> 1;
    }
    minatarGrid kill = *aliens & *shots;
    killed = invaders_count(kill);
    *aliens &= ~kill;
    *shots &= ~kill;
    (*march)--;
    return killed;
}

static uint32_t invaders_safe(const minatarInvaders *s, uint32_t action, uint32_t depth, uint32_t stuck) {
    minatarInvaders sim = *s;
    uint32_t done = 0u;
    sim.last = (uint8_t)(stuck & 1u ? sim.last : action);
    invaders_tick(&sim, sim.last, &done);
    if (done) return 0u;
    if (depth == 0u) return 1u;
    stuck >>= 1;
    if (stuck & 1u) return invaders_safe(&sim, sim.last, depth - 1u, stuck);
    return invaders_safe(&sim, MINATAR_NOOP, depth - 1u, stuck) || invaders_safe(&sim, MINATAR_LEFT, depth - 1u, stuck) || invaders_safe(&sim, MINATAR_RIGHT, depth - 1u, stuck);
}

static uint32_t invaders_oracle(const minatarKnobs *k, const minatarInvaders *s, uint32_t seed, uint32_t step) {
    enum { HORIZON = 14 };
    minatarGrid forecast[HORIZON + 1], aliens = invaders_load(s->aliens), shots = invaders_load(s->shots);
    uint8_t march = s->march, right = s->right;
    for (uint32_t t = 1u; t <= HORIZON; t++) {
        invaders_march(&aliens, &shots, &march, &right, s->pace);
        if (!aliens) aliens = invaders_wave();
        forecast[t] = aliens;
    }
    uint32_t best = 99u, bestFire = 99u, target = s->cannon;
    for (uint32_t c = 0u; c < 10u; c++) {
        uint32_t move = c > s->cannon ? c - s->cannon : s->cannon - c, first = move > s->reload ? move + 1u : (uint32_t)s->reload + 1u;
        for (uint32_t fire = first; fire < first + 6u; fire++)
            for (uint32_t j = 0u; j < 9u && fire + j <= HORIZON; j++)
                if (forecast[fire + j] & invaders_cell(c, 8u - j)) {
                    if (fire + j < best) {
                        best = fire + j;
                        bestFire = fire;
                        target = c;
                    }
                    break;
                }
    }
    if (best == 99u) target = invaders_nearest(invaders_columns(invaders_load(s->aliens)), s->cannon);
    uint32_t want = target < s->cannon ? MINATAR_LEFT : target > s->cannon ? MINATAR_RIGHT : bestFire == 1u ? MINATAR_FIRE : MINATAR_NOOP;
    uint32_t stuck = minatar_stuck(k, seed, step + 1u, 4u);
    if (stuck & 1u) return s->last;
    const uint32_t options[5] = { want, MINATAR_NOOP, MINATAR_LEFT, MINATAR_RIGHT, MINATAR_FIRE };
    for (uint32_t i = 0u; i < 5u; i++)
        if (invaders_safe(s, options[i], 3u, stuck)) return options[i];
    return want;
}

static uint32_t freeway_safe(const minatarFreeway *s, uint32_t action, uint32_t steps) {
    minatarFreeway sim = *s;
    for (uint32_t t = 0u; t < steps; t++) {
        uint32_t done = 0u;
        if (freeway_cross(&sim, t ? MINATAR_NOOP : action) != 0.0f) return 1u;
        uint32_t moved = sim.y;
        freeway_traffic(&sim, &done);
        if (sim.y == 9u && moved != 9u) return 0u;
    }
    return 1u;
}

static uint32_t freeway_oracle(const minatarKnobs *k, const minatarFreeway *s, uint32_t seed, uint32_t step) {
    if (minatar_stuck(k, seed, step + 1u, 1u)) return s->last;
    if (s->cool) return MINATAR_NOOP;
    if (freeway_safe(s, MINATAR_UP, 4u)) return MINATAR_UP;
    if (freeway_safe(s, MINATAR_NOOP, 4u)) return MINATAR_NOOP;
    if (freeway_safe(s, MINATAR_DOWN, 4u)) return MINATAR_DOWN;
    return MINATAR_NOOP;
}

static void asterix_masks(const minatarAsterix *s, uint32_t *enemy, uint32_t *gold) {
    for (uint32_t r = 0u; r < 8u; r++) {
        uint32_t e = s->row[r];
        if (!e) continue;
        if (e & ASTERIX_GOLD) gold[r] |= 1u << (e & 15u);
        else enemy[r] |= 1u << (e & 15u);
    }
}

static void asterix_ahead(minatarAsterix *w, u4 dice, uint32_t *enemy, uint32_t *gold) {
    if (w->spawnTimer == 0u) asterix_spawn(w, dice);
    asterix_masks(w, enemy, gold);
    if (w->moveTimer == 0u) {
        asterix_march(w);
        asterix_masks(w, enemy, gold);
    }
    asterix_clock(w);
}

static void asterix_moveTo(uint32_t action, uint32_t *x, uint32_t *y) {
    *x -= action == MINATAR_LEFT && *x > 0u;
    *x += action == MINATAR_RIGHT && *x < 9u;
    *y -= action == MINATAR_UP && *y > 1u;
    *y += action == MINATAR_DOWN && *y < 8u;
}

static uint32_t asterix_oracle(const minatarKnobs *k, const minatarAsterix *s, uint32_t seed, uint32_t step) {
    minatarAsterix w = *s;
    uint32_t enemy1[8] = { 0u }, gold1[8] = { 0u }, enemy2[8] = { 0u }, gold2[8] = { 0u };
    asterix_ahead(&w, minatar_future(seed, step + 1u), enemy1, gold1);
    minatarAsterix after = w;
    asterix_ahead(&w, minatar_future(seed, step + 2u), enemy2, gold2);
    uint32_t stuck = minatar_stuck(k, seed, step + 1u, 2u);
    int best = -1000000;
    uint32_t choice = s->last;
    for (uint32_t a = 0u; a < 5u; a++) {
        uint32_t act = stuck & 1u ? s->last : a, x = s->x, y = s->y;
        asterix_moveTo(act, &x, &y);
        if (enemy1[y - 1u] >> x & 1u) continue;
        int score = gold1[y - 1u] >> x & 1u ? 1000 : 0, viable = 0;
        for (uint32_t b = 0u; b < 5u && !viable; b++) {
            uint32_t x2 = x, y2 = y;
            asterix_moveTo(stuck & 2u ? act : b, &x2, &y2);
            viable = !(enemy2[y2 - 1u] >> x2 & 1u);
        }
        if (!viable) score -= 100000;
        int near = 99;
        for (uint32_t r = 0u; r < 8u; r++) {
            uint32_t e = after.row[r];
            if (!e || !(e & ASTERIX_GOLD)) continue;
            int d = abs((int)(e & 15u) - (int)x) + abs((int)r + 1 - (int)y);
            near = d < near ? d : near;
        }
        score -= near < 99 ? 10 * near : 0;
        score -= x == 0u || x == 9u ? 5 : 0;
        if (score > best) {
            best = score;
            choice = act;
        }
        if (stuck & 1u) break;
    }
    return choice;
}

static void sea_ghost(minatarSeaquest *s, uint32_t *extra, u4 dice) {
    uint32_t done = 0u, x = s->x, y = s->y;
    s->x = 15u;
    s->y = 15u;
    sea_tick(s, extra, MINATAR_NOOP, dice, &done);
    s->x = (uint8_t)x;
    s->y = (uint8_t)y;
}

static void sea_masks(const minatarSeaquest *s, const uint32_t *extra, uint32_t *danger) {
    const uint32_t *lists[3] = { extra + MINATAR_FISH, extra + MINATAR_SUBS, extra + MINATAR_BOMBS };
    const uint32_t counts[3] = { s->fish, s->subs, s->bombs };
    for (uint32_t l = 0u; l < 3u; l++)
        for (uint32_t i = 0u; i < counts[l]; i++) danger[sea_y(lists[l][i])] |= 1u << sea_x(lists[l][i]);
}

static uint32_t sea_moveTo(uint32_t action, uint32_t *x, uint32_t *y, uint32_t right) {
    if (action == MINATAR_LEFT) {
        *x -= *x > 0u;
        return 0u;
    }
    if (action == MINATAR_RIGHT) {
        *x += *x < 9u;
        return 1u;
    }
    *y -= action == MINATAR_UP && *y > 0u;
    *y += action == MINATAR_DOWN && *y < 8u;
    return right;
}

static uint32_t sea_oracle(const minatarKnobs *k, const minatarSeaquest *s, uint32_t *extra, uint32_t seed, uint32_t step) {
    uint32_t *scratch = extra + MINATAR_SEA, danger1[10] = { 0u }, danger2[10] = { 0u };
    minatarSeaquest w = *s;
    memcpy(scratch, extra, MINATAR_SEA * sizeof(uint32_t));
    sea_masks(&w, scratch, danger1);
    sea_ghost(&w, scratch, minatar_future(seed, step + 1u));
    sea_masks(&w, scratch, danger1);
    memcpy(danger2, danger1, sizeof danger2);
    sea_ghost(&w, scratch, minatar_future(seed, step + 2u));
    sea_masks(&w, scratch, danger2);
    const uint32_t *divers = extra + MINATAR_DIVERS;
    uint32_t tx = s->x, ty = s->y, up = s->rescued == 6 || (s->rescued != 0 && s->oxygen < 16 + 4 * (int32_t)s->y);
    if (up) ty = 0u;
    else if (s->divers && s->rescued < 6) {
        uint32_t best = 99u;
        for (uint32_t i = 0u; i < s->divers; i++) {
            uint32_t dx = sea_x(divers[i]), dy = sea_y(divers[i]), d = (dx > s->x ? dx - s->x : s->x - dx) + (dy > s->y ? dy - s->y : s->y - dy);
            if (d < best) {
                best = d;
                tx = dx;
                ty = dy;
            }
        }
    } else if (s->y == 0u || s->y > 6u) ty = 4u;
    uint32_t stuck = minatar_stuck(k, seed, step + 1u, 2u), ahead = 0u;
    for (uint32_t i = 0u; i < s->fish + s->subs; i++) {
        uint32_t e = i < s->fish ? extra[MINATAR_FISH + i] : extra[MINATAR_SUBS + i - s->fish];
        if (sea_y(e) == s->y && (s->right ? sea_x(e) > s->x : sea_x(e) < s->x)) ahead = 1u;
    }
    int best = -1000000;
    uint32_t choice = s->last;
    for (uint32_t a = 0u; a < 6u; a++) {
        uint32_t act = stuck & 1u ? s->last : a, x = s->x, y = s->y, right = s->right;
        right = act == MINATAR_FIRE ? right : sea_moveTo(act, &x, &y, right);
        int score = 0;
        if (danger1[y] >> x & 1u) score -= 1000000;
        if (y == 0u && s->y > 0u && (s->rescued == 0 || s->oxygen == 0)) score -= 1000000;
        int viable = 0;
        for (uint32_t b = 0u; b < 5u && !viable; b++) {
            uint32_t x2 = x, y2 = y;
            sea_moveTo(stuck & 2u ? (act == MINATAR_FIRE ? MINATAR_NOOP : act) : b, &x2, &y2, right);
            viable = !(danger2[y2] >> x2 & 1u) && !(y2 == 0u && y > 0u && (s->rescued == 0 || s->oxygen == 1));
        }
        if (!viable) score -= 100000;
        int dist = abs((int)x - (int)tx) + abs((int)y - (int)ty);
        score -= 10 * dist;
        if (act == MINATAR_FIRE && s->reload == 0u && ahead) score += 25;
        if (score > best) {
            best = score;
            choice = act;
        }
        if (stuck & 1u) break;
    }
    return choice;
}

void minatar_oracle(uint32_t instance, uint32_t seed, uint32_t step, const uint32_t *words, uint32_t *extra, const float *obs, float *ctrl, const float *shared) {
    (void)obs; (void)shared;
    const minatarKnobs *k = &MINATAR_KNOBS[instance];
    uint32_t action;
    switch (k->game) {
        case MINATAR_BREAKOUT: action = breakout_oracle(k, (const minatarBreakout *)words, seed, step); break;
        case MINATAR_INVADERS: action = invaders_oracle(k, (const minatarInvaders *)words, seed, step); break;
        case MINATAR_FREEWAY: action = freeway_oracle(k, (const minatarFreeway *)words, seed, step); break;
        case MINATAR_ASTERIX: action = asterix_oracle(k, (const minatarAsterix *)words, seed, step); break;
        default: action = sea_oracle(k, (const minatarSeaquest *)words, extra, seed, step); break;
    }
    for (uint32_t i = 0u; i < MINATAR_ACTION_COUNT[k->game]; i++)
        if (MINATAR_ACTION_MAP[k->game][i] == action) ctrl[0] = (float)i;
}
