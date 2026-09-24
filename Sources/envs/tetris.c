#include <string.h>
#include "../env.h"

typedef struct { uint32_t startLevel; } tetrisKnobs;

static const tetrisKnobs TETRIS_KNOBS[2] = {
    { 0u },
    { 19u }
};

static const uint32_t TETRIS_FRAMES_PER_ROW[30] = { 48u, 43u, 38u, 33u, 28u, 23u, 18u, 13u, 8u, 6u, 5u, 5u, 5u, 4u, 4u, 4u, 3u, 3u, 3u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 1u };
static const uint32_t TETRIS_LINE_POINTS[5] = { 0u, 40u, 100u, 300u, 1200u };

enum { TETRIS_RIGHT = 1, TETRIS_LEFT = 2, TETRIS_DOWN = 4, TETRIS_A = 8, TETRIS_B = 16 };
enum { TETRIS_DAS = 16, TETRIS_REPEAT = 10, TETRIS_HANG = -96, TETRIS_SOFT = 3, TETRIS_HEIGHT = 20, TETRIS_FULL = 0x3FF, TETRIS_MAX_SCORE = 999999, TETRIS_CRASH_LEVEL = 155 };
enum { TETRIS_WORDS = 20, TETRIS_PLAN_GAME = 0, TETRIS_PLAN_DEALT, TETRIS_PLAN_TARGET, TETRIS_PLAN };

typedef struct { int8_t cell[4], left, right, bottom; uint8_t type, turn, cw, ccw; } tetrisShape;

static const tetrisShape TETRIS_SHAPE[19] = {
    { { -1, 0, 1, -10 }, -1, 1, 0, 0u, 0u, 1u, 3u },
    { { -10, 0, 1, 10 }, 0, 1, 1, 0u, 1u, 2u, 0u },
    { { -1, 0, 1, 10 }, -1, 1, 1, 0u, 2u, 3u, 1u },
    { { -10, -1, 0, 10 }, -1, 0, 1, 0u, 3u, 0u, 2u },
    { { -10, 0, 9, 10 }, -1, 0, 1, 1u, 0u, 5u, 7u },
    { { -11, -1, 0, 1 }, -1, 1, 0, 1u, 1u, 6u, 4u },
    { { -10, -9, 0, 10 }, 0, 1, 1, 1u, 2u, 7u, 5u },
    { { -1, 0, 1, 11 }, -1, 1, 1, 1u, 3u, 4u, 6u },
    { { -1, 0, 10, 11 }, -1, 1, 1, 2u, 0u, 9u, 9u },
    { { -9, 0, 1, 10 }, 0, 1, 1, 2u, 1u, 8u, 8u },
    { { -1, 0, 9, 10 }, -1, 0, 1, 3u, 0u, 10u, 10u },
    { { 0, 1, 9, 10 }, -1, 1, 1, 4u, 0u, 12u, 12u },
    { { -10, 0, 1, 11 }, 0, 1, 1, 4u, 1u, 11u, 11u },
    { { -10, 0, 10, 11 }, 0, 1, 1, 5u, 0u, 14u, 16u },
    { { -1, 0, 1, 9 }, -1, 1, 1, 5u, 1u, 15u, 13u },
    { { -11, -10, 0, 10 }, -1, 0, 1, 5u, 2u, 16u, 14u },
    { { -9, -1, 0, 1 }, -1, 1, 0, 5u, 3u, 13u, 15u },
    { { -20, -10, 0, 10 }, 0, 0, 1, 6u, 0u, 18u, 18u },
    { { -2, -1, 0, 1 }, -2, 1, 0, 6u, 1u, 17u, 17u }
};

static const uint8_t TETRIS_SPAWN[7] = { 2u, 7u, 8u, 10u, 11u, 14u, 18u };

typedef struct {
    uint32_t board[8];
    uint32_t score, lines, game, cleared, lfsr;
    uint8_t x, y, orient, type, next, delay, autoX, fall, hold, buttons, frame, dealt, spawnId, level;
    int8_t autoY;
    uint8_t heights[12], spare;
} tetrisState;

_Static_assert(sizeof(tetrisState) == TETRIS_WORDS * sizeof(uint32_t), "");

static uint32_t tetris_lfsr(uint32_t v) { return v >> 1 | ((v >> 1 ^ v >> 9) & 1u) << 15; }

static uint32_t tetris_cell(const uint32_t *board, uint32_t cell) { return board[cell >> 5] >> (cell & 31u) & 1u; }

static int tetris_fits(const uint32_t *board, uint32_t orient, int x, int y) {
    const tetrisShape *p = &TETRIS_SHAPE[orient];
    if (x + p->left < 0 || x + p->right > 9 || y + p->bottom > 19) return 0;
    uint32_t base = (uint32_t)(10 * y + x), hit = 0u;
    for (uint32_t k = 0u; k < 4u; k++) hit |= tetris_cell(board, (base + (uint32_t)(int32_t)p->cell[k]) & 255u);
    return !hit;
}

static uint32_t tetris_row(const uint32_t *board, uint32_t base) {
    uint64_t pair = board[base >> 5] | (uint64_t)(base < 224u ? board[(base >> 5) + 1u] : 0u) << 32;
    return (uint32_t)(pair >> (base & 31u)) & TETRIS_FULL;
}

static uint32_t tetris_mask(uint32_t count, uint32_t limb) {
    if (count >= 32u * limb + 32u) return 0xFFFFFFFFu;
    if (count <= 32u * limb) return 0u;
    return (1u << (count - 32u * limb)) - 1u;
}

static void tetris_collapse(uint32_t *board, uint32_t base) {
    uint32_t moved = ((base + 255u) & 255u) + 1u, carry = 0u;
    for (uint32_t i = 0u; i < 8u; i++) {
        uint32_t low = board[i] & tetris_mask(moved, i);
        board[i] = low << 10 | carry | (board[i] & ~tetris_mask(moved + 10u, i));
        carry = low >> 22;
    }
}

static void tetris_heights(tetrisState *s) {
    uint32_t seen = 0u;
    memset(s->heights, 0, sizeof s->heights);
    for (uint32_t r = 0u; r < TETRIS_HEIGHT && seen != TETRIS_FULL; r++) {
        uint32_t fresh = tetris_row(s->board, 10u * r) & ~seen;
        seen |= fresh;
        for (; fresh; fresh &= fresh - 1u) s->heights[__builtin_ctz(fresh)] = (uint8_t)(TETRIS_HEIGHT - r);
    }
}

static uint32_t tetris_pick(tetrisState *s) {
    s->dealt++;
    uint32_t index = ((s->lfsr >> 8) + s->dealt) & 7u;
    if (index == 7u || TETRIS_SPAWN[index] == s->spawnId) {
        s->lfsr = tetris_lfsr(s->lfsr);
        index = ((s->lfsr >> 8 & 7u) + s->spawnId) % 7u;
    }
    s->spawnId = TETRIS_SPAWN[index];
    return index;
}

static void tetris_start(const tetrisKnobs *k, uint32_t seed, tetrisState *s) {
    uint32_t game = s->game, dealt = s->dealt, buttons = s->buttons;
    memset(s, 0, sizeof *s);
    s->game = game;
    s->dealt = (uint8_t)dealt;
    s->buttons = (uint8_t)buttons;
    s->level = (uint8_t)k->startLevel;
    s->autoY = TETRIS_HANG;
    for (uint32_t block = 0u; !s->lfsr; block++) {
        u4 r = philox4x32(seed, ENV_RNG_STREAM, game, block, 0u, 0u);
        uint32_t draw[3] = { r.x & 0x7FFFu, r.y & 0x7FFFu, r.z & 0x7FFFu };
        if (!block) s->frame = (uint8_t)r.w;
        for (uint32_t i = 0u; i < 3u && !s->lfsr; i++) s->lfsr = draw[i] | ((draw[i] ^ draw[i] >> 8) & 1u) << 15;
    }
    s->type = (uint8_t)tetris_pick(s);
    s->next = (uint8_t)tetris_pick(s);
    s->orient = TETRIS_SPAWN[s->type];
    s->x = 5u;
}

static __attribute__((noinline)) void tetris_spawn(tetrisState *s) {
    s->type = s->next;
    s->next = (uint8_t)tetris_pick(s);
    s->orient = TETRIS_SPAWN[s->type];
    s->x = 5u;
    s->y = 0u;
    s->autoY = 0;
}

static uint32_t tetris_down(tetrisState *s) {
    if (!tetris_fits(s->board, s->orient, s->x, s->y + 1)) return 1u;
    s->y++;
    return 0u;
}

static inline __attribute__((always_inline)) uint32_t tetris_move(tetrisState *s, uint32_t held, uint32_t fresh) {
    if (!(held & TETRIS_DOWN)) {
        uint32_t shift = 0u;
        if (fresh & (TETRIS_LEFT | TETRIS_RIGHT)) {
            s->autoX = 0u;
            shift = 1u;
        } else if ((held & (TETRIS_LEFT | TETRIS_RIGHT)) && ++s->autoX >= TETRIS_DAS) {
            s->autoX = TETRIS_REPEAT;
            shift = 1u;
        }
        if (shift) {
            int x = s->x + (held & TETRIS_RIGHT ? 1 : -1);
            if (tetris_fits(s->board, s->orient, x, s->y)) s->x = (uint8_t)x;
            else s->autoX = TETRIS_DAS;
        }
    }
    if (fresh & (TETRIS_A | TETRIS_B)) {
        uint32_t turned = fresh & TETRIS_A ? TETRIS_SHAPE[s->orient].cw : TETRIS_SHAPE[s->orient].ccw;
        if (tetris_fits(s->board, turned, s->x, s->y)) s->orient = (uint8_t)turned;
    }
    if (s->autoY < 0) {
        if (fresh & TETRIS_DOWN) s->autoY = 0;
        else s->autoY++;
        return 0u;
    }
    uint32_t exact = (held & (TETRIS_LEFT | TETRIS_RIGHT | TETRIS_DOWN)) == TETRIS_DOWN;
    if (!s->autoY) {
        if (exact && (fresh & TETRIS_DOWN)) s->autoY = 1;
    } else if (!exact) {
        s->autoY = 0;
        s->hold = 0u;
    } else if (++s->autoY >= TETRIS_SOFT) {
        s->autoY = 1;
        s->hold++;
        s->fall = 0u;
        return tetris_down(s);
    }
    if (++s->fall < TETRIS_FRAMES_PER_ROW[s->level < 29u ? s->level : 29u]) return 0u;
    s->fall = 0u;
    return tetris_down(s);
}

static uint32_t tetris_entry(uint32_t y) {
    uint32_t from = y < 2u ? 0u : y - 2u;
    return 8u + 2u * ((23u - from) / 4u);
}

static uint32_t tetris_count(tetrisState *s, uint32_t lines) {
    for (uint32_t i = 0u; i < lines; i++) {
        s->lines = s->lines + 1u == 25600u ? 0u : s->lines + 1u;
        if (s->lines % 10u) continue;
        uint32_t middle = (s->lines / 100u & 15u) << 4 | s->lines / 10u % 10u;
        if ((int8_t)(uint8_t)(s->level - middle) >= 0) continue;
        if (++s->level == TETRIS_CRASH_LEVEL) return 1u;
    }
    return 0u;
}

static uint32_t tetris_pushdown(uint32_t score, uint32_t points) {
    uint32_t low = score % 100u, packed = (low / 10u << 4 | low % 10u) + points, carry = 0u;
    if ((packed & 15u) > 9u) packed += 6u;
    if (packed >> 4 > 9u) {
        packed = (packed & 0xF0u) - 0xA0u;
        carry = 100u;
    }
    return score - low + carry + (packed >> 4) * 10u + (packed & 15u);
}

static float tetris_over(const tetrisKnobs *k, uint32_t seed, tetrisState *s, uint32_t *done) {
    *done = 1u;
    s->game++;
    tetris_start(k, seed, s);
    return 0.0f;
}

static __attribute__((noinline)) float tetris_lock(const tetrisKnobs *k, uint32_t seed, tetrisState *s, uint32_t *done) {
    if (!tetris_fits(s->board, s->orient, s->x, s->y)) return tetris_over(k, seed, s, done);
    int base = 10 * s->y + s->x;
    for (uint32_t c = 0u; c < 4u; c++) {
        int cell = base + TETRIS_SHAPE[s->orient].cell[c];
        if (cell >= 0) s->board[cell >> 5] |= 1u << (cell & 31);
    }
    uint32_t lines = 0u, rows = 0u;
    for (int r = s->y - 2; r <= s->y + 1; r++) {
        uint32_t at = (uint32_t)(10 * r) & 255u;
        if (tetris_row(s->board, at) != TETRIS_FULL) continue;
        tetris_collapse(s->board, at);
        rows |= (uint32_t)(r + 2) << (3u + 5u * lines);
        lines++;
    }
    s->cleared = lines | rows;
    s->delay = (uint8_t)(tetris_entry(s->y) + (lines ? 20u - (s->frame & 3u) : 0u));
    if (tetris_count(s, lines)) return tetris_over(k, seed, s, done);
    uint32_t score = s->score;
    if ((int8_t)(uint8_t)(s->hold - 2u) >= 0) score = tetris_pushdown(score, s->hold - 1u);
    score += TETRIS_LINE_POINTS[lines] * (s->level + 1u);
    if (score > TETRIS_MAX_SCORE) score = TETRIS_MAX_SCORE;
    float reward = (float)(score - s->score);
    s->score = score;
    s->hold = 0u;
    tetris_heights(s);
    return reward;
}

static uint32_t tetris_buttons(const float *ctrl) {
    return (ctrl[0] > 1.0f / 3.0f ? TETRIS_RIGHT : 0u) | (ctrl[0] < -1.0f / 3.0f ? TETRIS_LEFT : 0u) | (ctrl[1] > 0.0f ? TETRIS_DOWN : 0u)
           | (ctrl[2] > 0.0f ? TETRIS_A : 0u) | (ctrl[3] > 0.0f ? TETRIS_B : 0u);
}

static void tetris_obs(const tetrisState *s, float *obs) {
    for (uint32_t c = 0u; c < 12u; c += 4u) {
        uint32_t four;
        memcpy(&four, s->heights + c, sizeof four);
        float4 heights = __builtin_convertvector(four >> (lane4){ 0u, 8u, 16u, 24u } & 255u, float4) * (1.0f / TETRIS_HEIGHT);
        memcpy(obs + c, &heights, sizeof heights);
    }
    obs[10] = (float)s->type * (1.0f / 6.0f);
    obs[11] = (float)s->x * (1.0f / 9.0f);
    obs[12] = (float)s->y * (1.0f / 19.0f);
    obs[13] = (float)TETRIS_SHAPE[s->orient].turn * (1.0f / 3.0f);
    obs[14] = (float)s->next * (1.0f / 6.0f);
    obs[15] = (float)s->level * (1.0f / 29.0f);
    obs[16] = s->delay ? 0.0f : 1.0f;
    obs[17] = 1.0f;
}

uint32_t tetris_extraWords(uint32_t instance, uint32_t steps) {
    (void)instance; (void)steps;
    return TETRIS_PLAN;
}

void tetris_reset(uint32_t instance, uint32_t seed, uint32_t *words, uint32_t *extra, uint32_t steps) {
    (void)steps;
    tetrisState *s = (tetrisState *)words;
    memset(s, 0, sizeof *s);
    tetris_start(&TETRIS_KNOBS[instance], seed, s);
    memset(extra, 0, TETRIS_PLAN * sizeof(uint32_t));
}

float tetris_step(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done, const float *shared) {
    (void)step; (void)extra; (void)shared;
    tetrisState *s = (tetrisState *)words;
    uint32_t held = tetris_buttons(ctrl), fresh = held & ~(uint32_t)s->buttons;
    float reward = 0.0f;
    s->buttons = (uint8_t)held;
    s->lfsr = tetris_lfsr(s->lfsr);
    s->frame++;
    *done = 0u;
    if (s->delay) {
        if (!--s->delay) tetris_spawn(s);
    } else if (tetris_move(s, held, fresh)) reward = tetris_lock(&TETRIS_KNOBS[instance], seed, s, done);
    tetris_obs(s, obs);
    return reward;
}

void tetris_observe(uint32_t instance, uint32_t seed, const uint32_t *words, const uint32_t *extra, uint32_t step, float *obs, const float *shared) {
    (void)instance; (void)seed; (void)extra; (void)step; (void)shared;
    tetris_obs((const tetrisState *)words, obs);
}

static float tetris_value(const uint32_t *board, uint32_t orient, int x, int y) {
    uint32_t rows[20], kept[20], piece[20] = { 0u };
    int base = 10 * y + x, top = 19, bottom = -2;
    for (uint32_t r = 0u; r < 20u; r++) rows[r] = tetris_row(board, 10u * r);
    for (uint32_t c = 0u; c < 4u; c++) {
        int cell = base + TETRIS_SHAPE[orient].cell[c], cy = (cell + 20) / 10 - 2;
        top = cy < top ? cy : top;
        bottom = cy > bottom ? cy : bottom;
        if (cy >= 0) piece[cy] |= 1u << (cell - 10 * cy);
    }
    uint32_t cleared = 0u, eroded = 0u, depth[10] = { 0u };
    int n = 20;
    for (int r = 19; r >= 0; r--) {
        uint32_t row = rows[r] | piece[r];
        if (row == TETRIS_FULL) {
            cleared++;
            eroded += (uint32_t)__builtin_popcount(piece[r]);
        } else kept[--n] = row;
    }
    uint32_t rowTransitions = 2u * (uint32_t)n, columnTransitions = 0u, holes = 0u, wells = 0u, covered = 0u, above = 0u, wellAbove = 0u;
    for (int r = n; r < 20; r++) {
        uint32_t row = kept[r], walled = row << 1 | 0x801u;
        uint32_t well = ~row & (row << 1 | 1u) & (row >> 1 | 0x200u) & TETRIS_FULL;
        rowTransitions += (uint32_t)__builtin_popcount((walled ^ walled >> 1) & 0x7FFu);
        columnTransitions += (uint32_t)__builtin_popcount(row ^ above);
        holes += (uint32_t)__builtin_popcount(covered & ~row & TETRIS_FULL);
        covered |= row;
        for (uint32_t w = well; w; w &= w - 1u) {
            uint32_t c = (uint32_t)__builtin_ctz(w);
            depth[c] = wellAbove >> c & 1u ? depth[c] + 1u : 1u;
            wells += depth[c];
        }
        wellAbove = well;
        above = row;
    }
    columnTransitions += (uint32_t)__builtin_popcount(~above & TETRIS_FULL);
    float landing = 20.0f - 0.5f * (float)(top + bottom);
    return -4.500158825082766f * landing + 3.4181268101392694f * (float)(cleared * eroded) - 3.2178882868487753f * (float)rowTransitions
           - 9.348695305445199f * (float)columnTransitions - 7.899265427351652f * (float)holes - 3.3855972247263626f * (float)wells;
}

static uint32_t tetris_press(const tetrisState *s, uint32_t x, uint32_t orient) {
    uint32_t want = ((uint32_t)s->x < x ? TETRIS_RIGHT : 0u) | ((uint32_t)s->x > x ? TETRIS_LEFT : 0u);
    const tetrisShape *p = &TETRIS_SHAPE[s->orient];
    if (s->orient != orient) want |= p->ccw == orient && p->cw != orient ? TETRIS_B : TETRIS_A;
    if (want) return want & ~(uint32_t)s->buttons;
    return s->autoY > 0 || !(s->buttons & TETRIS_DOWN) ? TETRIS_DOWN : 0u;
}

static __attribute__((noinline)) void tetris_plan(const tetrisState *s, uint32_t *extra) {
    extra[TETRIS_PLAN_GAME] = s->game + 1u;
    extra[TETRIS_PLAN_DEALT] = s->dealt;
    extra[TETRIS_PLAN_TARGET] = s->x | (uint32_t)s->orient << 8;
    if (s->score >= TETRIS_MAX_SCORE) return;
    const tetrisShape *p = &TETRIS_SHAPE[s->orient];
    uint32_t turns[4] = { s->orient, p->cw, TETRIS_SHAPE[p->cw].cw, p->ccw };
    float best = -INFINITY;
    for (uint32_t t = 0u; t < 4u; t++) {
        uint32_t orient = turns[t];
        if ((t && orient == s->orient) || (t == 3u && orient == p->cw)) continue;
        const tetrisShape *q = &TETRIS_SHAPE[orient];
        for (int x = -q->left; x <= 9 - q->right; x++) {
            tetrisState sim = *s;
            uint32_t locked = 0u;
            for (uint32_t frame = 0u; frame < 24u && (sim.x != x || sim.orient != orient); frame++) {
                uint32_t held = tetris_press(&sim, (uint32_t)x, orient), fresh = held & ~(uint32_t)sim.buttons;
                sim.buttons = (uint8_t)held;
                if ((locked = tetris_move(&sim, held, fresh))) break;
            }
            if (!locked) {
                if (sim.x != x || sim.orient != orient) continue;
                while (tetris_fits(sim.board, orient, x, sim.y + 1)) sim.y++;
            } else if (!tetris_fits(sim.board, sim.orient, sim.x, sim.y)) continue;
            float value = tetris_value(sim.board, sim.orient, sim.x, sim.y);
            if (value > best) {
                best = value;
                extra[TETRIS_PLAN_TARGET] = (uint32_t)x | orient << 8;
            }
        }
    }
}

void tetris_oracle(uint32_t instance, uint32_t seed, uint32_t step, const uint32_t *words, uint32_t *extra, const float *obs, float *ctrl, const float *shared) {
    (void)instance; (void)seed; (void)step; (void)obs; (void)shared;
    const tetrisState *s = (const tetrisState *)words;
    uint32_t held = 0u;
    if (!s->delay) {
        if (extra[TETRIS_PLAN_GAME] != s->game + 1u || extra[TETRIS_PLAN_DEALT] != s->dealt) tetris_plan(s, extra);
        held = tetris_press(s, extra[TETRIS_PLAN_TARGET] & 255u, extra[TETRIS_PLAN_TARGET] >> 8);
    }
    ctrl[0] = held & TETRIS_RIGHT ? 1.0f : held & TETRIS_LEFT ? -1.0f : 0.0f;
    ctrl[1] = held & TETRIS_DOWN ? 1.0f : 0.0f;
    ctrl[2] = held & TETRIS_A ? 1.0f : 0.0f;
    ctrl[3] = held & TETRIS_B ? 1.0f : 0.0f;
}
