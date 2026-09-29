#include <string.h>
#include "../env.h"

typedef struct { uint32_t opponent; } relicfogKnobs;

static const relicfogKnobs RELICFOG_KNOBS[3] = {
    { 0u },
    { 1u },
    { 2u }
};

static const uint32_t RELICFOG_MOVE_COSTS[5] = { 1u, 2u, 3u, 4u, 5u };
static const uint32_t RELICFOG_SENSOR_RANGES[4] = { 1u, 2u, 3u, 4u };
static const uint32_t RELICFOG_VISION_REDUCTIONS[8] = { 0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u };
static const uint32_t RELICFOG_ENERGY_REDUCTIONS[6] = { 0u, 1u, 2u, 3u, 5u, 25u };
static const uint32_t RELICFOG_SAP_COSTS[21] = { 30u, 31u, 32u, 33u, 34u, 35u, 36u, 37u, 38u, 39u, 40u, 41u, 42u, 43u, 44u, 45u, 46u, 47u, 48u, 49u, 50u };
static const uint32_t RELICFOG_SAP_RANGES[5] = { 3u, 4u, 5u, 6u, 7u };
static const uint32_t RELICFOG_DROPOFF_QUARTERS[3] = { 1u, 2u, 4u };
static const uint32_t RELICFOG_VOID_SIXTEENTHS[4] = { 1u, 2u, 4u, 6u };
static const int RELICFOG_NEBULA_DRIFT_FORTIETHS[8] = { -6, -4, -2, -1, 1, 2, 4, 6 };
static const uint32_t RELICFOG_NODE_DRIFT_HUNDREDTHS[5] = { 1u, 2u, 3u, 4u, 5u };
static const uint32_t RELICFOG_NODE_DRIFT_MAGNITUDES[3] = { 3u, 4u, 5u };
static const float RELICFOG_THETA[8] = { 1.0f, 10.0f, 30.0f, 4.0f, 8.0f, 20.0f, 0.5f, 1.0f };
static const float RELICFOG_THETA_LOW[8] = { 0.5f, 0.0f, 0.0f, 0.0f, 1.0f, -50.0f, 0.25f, 0.0f };
static const float RELICFOG_THETA_HIGH[8] = { 3.0f, 100.0f, 150.0f, 12.0f, 16.0f, 100.0f, 1.0f, 3.0f };

static const uint8_t RELICFOG_HEADINGS[5] = { 0, 0, 3, 1, 2 };
static const uint8_t RELICFOG_WAYPOINTS[16] = { 12, 12, 17, 5, 5, 17, 18, 18, 8, 8, 21, 11, 11, 21, 15, 15 };
static const uint8_t RELICFOG_LANE_BITS[16] = { 1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128 };

static const float RELICFOG_THRESHOLD = 0.5f;
static const float RELICFOG_AMPLITUDE = 4.0f;
static const float RELICFOG_FREQUENCY = 1.2f;
static const float RELICFOG_PHASE = 1.0f;
static const float RELICFOG_LIFT = 1.5f;
static const float RELICFOG_DENSITY = 0.2f;

enum {
    RF_SIDE = 24, RF_TILES = RF_SIDE * RF_SIDE, RF_EDGE = RF_SIDE - 1, RF_UNITS = 16, RF_RELICS = 6, RF_PAIRS = 3, RF_MATCH_STEPS = 100,
    RF_MATCH_FRAMES = RF_MATCH_STEPS + 1, RF_MATCHES = 5, RF_GAME_STEPS = RF_MATCHES * RF_MATCH_FRAMES, RF_SPAWN_RATE = 3, RF_START_ENERGY = 100,
    RF_MAX_ENERGY = 400, RF_FIELD_LIMIT = 20, RF_SCHEDULE_SPAN = 50, RF_SELF_SIGHT = 10, RF_FOCUS_CENTER = 8, RF_NONE = RF_UNITS,
    RF_CENTERED = RF_FOCUS_CENTER | RF_FOCUS_CENTER << 4, RF_AIM_SHIFT = 8
};
enum { RF_FIELDS = 4, RF_GRADIENTS = 208, RF_GRADIENT_BATCH = 32, RF_GRADIENT_STAGES = 7, RF_FINAL_STAGE = RF_GRADIENT_STAGES + RF_FIELDS * RF_SIDE, RF_SURVEY_DONE };
enum { RF_NEBULA_FIELD, RF_ASTEROID_FIELD, RF_NODE_FIELD, RF_RELIC_FIELD };
enum { RF_WAIT, RF_ORDER, RF_TARGET };
enum { RF_CENTER, RF_UP, RF_RIGHT, RF_DOWN, RF_LEFT, RF_SAP, RF_ACTIONS };
enum { RF_SCOUT, RF_SAPPER, RF_LEAGUE };
enum { RF_IDLE, RF_HOLD, RF_TEST, RF_EXPLORE, RF_HARVEST };
enum { RW_MODE, RW_CURSOR, RW_FOCUS, RW_CLOCK, RW_ROSTER, RW_GAME, RW_STEP, RW_MATCH, RW_OUR_POINTS, RW_THEIR_POINTS, RW_OUR_WINS, RW_THEIR_WINS, RW_GAIN, RW_MEMBER, RW_WORDS };
enum { RF_TAG_GRADIENT = 1 << 24, RF_TAG_MAP = 2 << 24, RF_TAG_CONFIG = 3 << 24, RF_TAG_DRIFT = 4 << 24, RF_TAG_TIE = 5 << 24, RF_TAG_BOT = 6 << 24, RF_TAG_LEAGUE = 7 << 24 };
enum { RF_SAP_RATIO, RF_RESERVE, RF_RETREAT, RF_EXPLORERS, RF_TESTERS, RF_CAUTION, RF_DROPOFF_GUESS, RF_HARVEST_WEIGHT, RF_THETA };
enum { RF_LAYERS = 64, RF_REACH = 2, RF_PATCH = 2 * RF_REACH + 1, RF_PATCH_CELLS = RF_PATCH * RF_PATCH };
enum { RF_OFF_MAP = -1, RF_FOGGED, RF_OPEN, RF_NEBULA, RF_ASTEROID, RF_RELIC };
enum {
    RO_MODE, RO_ENERGY, RO_X, RO_Y, RO_FX, RO_FY, RO_FIELD, RO_MATCH, RO_ROUND, RO_GAIN, RO_LEAD, RO_MOVE_COST, RO_SAP_COST, RO_SAP_RANGE,
    RO_TERRAIN, RO_SHIPS = RO_TERRAIN + RF_PATCH_CELLS, RO_COUNT = RO_SHIPS + RF_PATCH_CELLS
};
_Static_assert(RO_COUNT <= OAK_ENV_OBS_MAX && RO_COUNT % 4 == 0, "");
enum {
    RF_VIEW_TILES = 0, RF_VIEW_UNITS = RF_TILES / 2, RF_VIEW_RELICS = RF_VIEW_UNITS + 4 * RF_UNITS, RF_VIEW_WEIGHTS = RF_VIEW_RELICS + RF_RELICS,
    RF_VIEW_NODE = RF_VIEW_WEIGHTS + RF_TILES / 8, RF_VIEW_SIGHT = RF_VIEW_NODE + 1, RF_VIEW_PARAMS = RF_VIEW_SIGHT + RF_TILES / 8,
    RF_VIEW_LAST = RF_VIEW_PARAMS + 11, RF_VIEW_ORDERS = RF_VIEW_LAST + 2 * RF_UNITS, RF_VIEW_FIELDS = RF_VIEW_ORDERS + RF_UNITS, RF_VIEW_WORDS = RF_VIEW_FIELDS / 2
};

typedef struct {
    uint8_t moveCost, sensorRange, visionReduction, energyReduction, sapCost, sapRange, dropoff, voidFactor, nodeSpeed, nodeMagnitude;
    int8_t nebulaSpeed;
} relicfogParams;

typedef struct {
    uint32_t asteroid[RF_SIDE], nebula[RF_SIDE], weight[RF_PAIRS][RF_SIDE];
    int16_t schedule[RF_PAIRS];
    uint8_t relicX[RF_RELICS], relicY[RF_RELICS], nodeX, nodeY;
    relicfogParams params;
} relicfogMap;

typedef struct {
    relicfogMap map;
    float gradient[RF_GRADIENTS][2], top[2][3];
    uint16_t topAt[2][3];
    uint32_t rawNebula[RF_SIDE], rawAsteroid[RF_SIDE];
    uint32_t game, stage;
} relicfogSurvey;

typedef struct { int8_t type, dx, dy; } relicfogOrder;

typedef struct {
    uint32_t candidate[RF_SIDE], known[RF_SIDE], ruled[RF_SIDE], enemyRows[RF_SIDE];
    float theta[RF_THETA];
    int16_t enemyEnergy[RF_UNITS], planned[RF_UNITS];
    uint8_t enemyX[RF_UNITS], enemyY[RF_UNITS], targetX[RF_UNITS], targetY[RF_UNITS], task[RF_UNITS], jitter[RF_UNITS];
    uint32_t seen, enemies, turn;
} relicfogBot;

typedef struct {
    relicfogOrder plan[RF_UNITS];
    int16_t left[RF_UNITS];
    uint32_t seen[RF_SIDE], rock[RF_SIDE], relics[RF_SIDE], candidate[RF_SIDE], known[RF_SIDE], ruled[RF_SIDE];
    int8_t energy[RF_TILES];
    uint32_t key;
} relicfogOracle;

typedef struct {
    relicfogMap map;
    uint32_t sight[RF_SIDE], score[RF_SIDE], relicRows[RF_SIDE];
    uint8_t ux[2][RF_UNITS], uy[2][RF_UNITS], live[2][RF_UNITS];
    int16_t energy[2][RF_UNITS];
    relicfogOrder order[2][RF_UNITS], last[2][RF_UNITS];
    uint32_t alive[2], relicMask, pairs, fieldAt;
    int32_t gain[2];
    relicfogBot bot;
    float theta[LEAGUE_MEMBERS * RF_THETA], winRate[LEAGUE_MEMBERS];
    relicfogSurvey survey;
    relicfogOracle oracle;
} relicfogWorld;

static relicfogWorld *relicfog_world(uint32_t *extra) { return (relicfogWorld *)(((uintptr_t)extra + 15u) & ~(uintptr_t)15u); }
static const relicfogWorld *relicfog_read(const uint32_t *extra) { return (const relicfogWorld *)(((uintptr_t)extra + 15u) & ~(uintptr_t)15u); }

uint32_t relicfog_extraWords(uint32_t instance, uint32_t steps) {
    (void)instance; (void)steps;
    return (uint32_t)(sizeof(relicfogWorld) / sizeof(uint32_t)) + 4u;
}

static uint32_t relicfog_span(int a, int b) { return (uint32_t)(a > b ? a - b : b - a); }

uint64_t relicfog_shared(float *table) {
    if (table) {
        float wave[RF_TILES], sum[RF_TILES];
        for (uint32_t dy = 0u; dy < RF_SIDE; dy++)
            for (uint32_t dx = 0u; dx < RF_SIDE; dx++) {
                float s, c, d = sqrtf((float)(dx * dx + dy * dy));
                oak_sincos(RELICFOG_FREQUENCY * d + RELICFOG_PHASE, &s, &c);
                wave[dy * RF_SIDE + dx] = RELICFOG_AMPLITUDE * s;
            }
        int8_t *fields = (int8_t *)table;
        for (int node = 0; node < RF_TILES; node++) {
            int ax = node % RF_SIDE, ay = node / RF_SIDE, bx = RF_EDGE - ay, by = RF_EDGE - ax;
            float total = 0.0f;
            for (int x = 0; x < RF_SIDE; x++)
                for (int y = 0; y < RF_SIDE; y++) {
                    float v = wave[relicfog_span(y, ay) * RF_SIDE + relicfog_span(x, ax)] + wave[relicfog_span(y, by) * RF_SIDE + relicfog_span(x, bx)];
                    sum[y * RF_SIDE + x] = v;
                    total += v;
                }
            float mean = total / (float)RF_TILES;
            for (uint32_t i = 0u; i < RF_TILES; i++) fields[node * RF_TILES + (int)i] = (int8_t)fminf(fmaxf(rintf(sum[i] - mean + RELICFOG_LIFT), -(float)RF_FIELD_LIMIT), (float)RF_FIELD_LIMIT);
        }
    }
    return RF_TILES * RF_TILES / 4u;
}

static const int8_t *relicfog_field(const relicfogWorld *w, const float *shared) { return (const int8_t *)shared + w->fieldAt * RF_TILES; }
static void relicfog_delta(uint32_t action, int *dx, int *dy) {
    *dx = *dy = 0;
    if (action) gridDelta(RELICFOG_HEADINGS[action], dx, dy);
}

static uint32_t relicfog_below(uint32_t word, uint32_t n) { return (uint32_t)(((uint64_t)word * n) >> 32); }
static uint32_t relicfog_bit(const uint32_t *rows, uint32_t x, uint32_t y) { return rows[y] >> x & 1u; }
static uint32_t relicfog_band(int lo, int hi) {
    lo = lo < 0 ? 0 : lo;
    hi = hi > RF_EDGE ? RF_EDGE : hi;
    return lo > hi ? 0u : ((2u << hi) - 1u) & ~((1u << lo) - 1u);
}
static int32_t relicfog_offset(uint32_t word, uint32_t magnitude) { return (int32_t)((((word >> 8) * (4u * magnitude)) >> 24) + 1u) / 2 - (int32_t)magnitude; }

static void relicfog_symmetrize(uint32_t *rows) {
    uint32_t out[RF_SIDE];
    memcpy(out, rows, sizeof out);
    for (uint32_t y = 0u; y < RF_SIDE; y++)
        for (uint32_t bits = rows[y]; bits; bits &= bits - 1u) out[RF_EDGE - (uint32_t)__builtin_ctz(bits)] |= 1u << (RF_EDGE - y);
    memcpy(rows, out, sizeof out);
}

static float relicfog_fade(float t) { return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f); }

static float relicfog_noise(const float (*g)[2], uint32_t stride, uint32_t cell, uint32_t a, uint32_t b) {
    uint32_t ia = a / cell, ib = b / cell;
    float fa = (float)(a - ia * cell) / (float)cell, fb = (float)(b - ib * cell) / (float)cell;
    const float *g00 = g[ia * stride + ib], *g10 = g[(ia + 1u) * stride + ib], *g01 = g[ia * stride + ib + 1u], *g11 = g[(ia + 1u) * stride + ib + 1u];
    float n00 = g00[0] * fa + g00[1] * fb, n10 = g10[0] * (fa - 1.0f) + g10[1] * fb;
    float n01 = g01[0] * fa + g01[1] * (fb - 1.0f), n11 = g11[0] * (fa - 1.0f) + g11[1] * (fb - 1.0f);
    float u = relicfog_fade(fa), v = relicfog_fade(fb), n0 = n00 + u * (n10 - n00), n1 = n01 + u * (n11 - n01);
    return 1.41421356f * (n0 + v * (n1 - n0));
}

static void relicfog_rank(relicfogSurvey *s, uint32_t f, float value, uint32_t at) {
    float *top = s->top[f];
    uint16_t *where = s->topAt[f];
    if (!(value > top[2])) return;
    uint32_t i = 2u;
    for (; i > 0u && value > top[i - 1u]; i--) {
        top[i] = top[i - 1u];
        where[i] = where[i - 1u];
    }
    top[i] = value;
    where[i] = (uint16_t)at;
}

static void relicfog_cover(uint32_t *rows, uint32_t config, uint32_t x, uint32_t y) {
    for (uint32_t u = 0u; u < 5u; u++)
        for (uint32_t v = 0u; v < 5u; v++) {
            int tx = (int)x + (int)u - 2, ty = (int)y + (int)v - 2;
            if (config >> (u * 5u + v) & 1u && tx >= 0 && ty >= 0 && tx < RF_SIDE && ty < RF_SIDE) rows[ty] |= 1u << tx;
        }
}

static uint32_t relicfog_draw(uint32_t seed, uint32_t game, uint32_t index) {
    u4 r = philox4x32(seed, ENV_RNG_STREAM, game, RF_TAG_MAP | (index >> 2), 0u, 0u);
    uint32_t words[4] = { r.x, r.y, r.z, r.w };
    return words[index & 3u];
}

static void relicfog_finish(relicfogSurvey *s, uint32_t seed) {
    relicfogMap *m = &s->map;
    uint32_t game = s->game;
    memcpy(m->nebula, s->rawNebula, sizeof m->nebula);
    memcpy(m->asteroid, s->rawAsteroid, sizeof m->asteroid);
    relicfog_symmetrize(m->nebula);
    relicfog_symmetrize(m->asteroid);
    for (uint32_t y = 0u; y < RF_SIDE; y++) m->nebula[y] &= ~m->asteroid[y];
    m->nodeX = (uint8_t)(s->topAt[0][2] % RF_SIDE);
    m->nodeY = (uint8_t)(s->topAt[0][2] / RF_SIDE);
    for (uint32_t q = 0u; q < RF_PAIRS; q++) {
        uint32_t at = s->topAt[1][2u - q];
        m->relicX[q] = (uint8_t)(at % RF_SIDE);
        m->relicY[q] = (uint8_t)(at / RF_SIDE);
        m->relicX[q + RF_PAIRS] = (uint8_t)(RF_EDGE - m->relicY[q]);
        m->relicY[q + RF_PAIRS] = (uint8_t)(RF_EDGE - m->relicX[q]);
    }
    relicfogParams *p = &m->params;
    p->moveCost = (uint8_t)RELICFOG_MOVE_COSTS[relicfog_below(relicfog_draw(seed, game, 0u), 5u)];
    p->sensorRange = (uint8_t)RELICFOG_SENSOR_RANGES[relicfog_below(relicfog_draw(seed, game, 1u), 4u)];
    p->visionReduction = (uint8_t)RELICFOG_VISION_REDUCTIONS[relicfog_below(relicfog_draw(seed, game, 2u), 8u)];
    p->energyReduction = (uint8_t)RELICFOG_ENERGY_REDUCTIONS[relicfog_below(relicfog_draw(seed, game, 3u), 6u)];
    p->sapCost = (uint8_t)RELICFOG_SAP_COSTS[relicfog_below(relicfog_draw(seed, game, 4u), 21u)];
    p->sapRange = (uint8_t)RELICFOG_SAP_RANGES[relicfog_below(relicfog_draw(seed, game, 5u), 5u)];
    p->dropoff = (uint8_t)RELICFOG_DROPOFF_QUARTERS[relicfog_below(relicfog_draw(seed, game, 6u), 3u)];
    p->voidFactor = (uint8_t)RELICFOG_VOID_SIXTEENTHS[relicfog_below(relicfog_draw(seed, game, 7u), 4u)];
    p->nebulaSpeed = (int8_t)RELICFOG_NEBULA_DRIFT_FORTIETHS[relicfog_below(relicfog_draw(seed, game, 8u), 8u)];
    p->nodeSpeed = (uint8_t)RELICFOG_NODE_DRIFT_HUNDREDTHS[relicfog_below(relicfog_draw(seed, game, 9u), 5u)];
    p->nodeMagnitude = (uint8_t)RELICFOG_NODE_DRIFT_MAGNITUDES[relicfog_below(relicfog_draw(seed, game, 10u), 3u)];
    uint32_t pairs = 1u + relicfog_below(relicfog_draw(seed, game, 11u), RF_PAIRS), covered[RF_SIDE] = {0}, threshold = u01Bound(RELICFOG_DENSITY);
    for (uint32_t q = 0u; q < RF_PAIRS; q++) {
        m->schedule[q] = (int16_t)(q < pairs ? (int)(q * RF_MATCH_FRAMES + relicfog_below(relicfog_draw(seed, game, 12u + q), RF_SCHEDULE_SPAN)) : -1);
        uint32_t config = 0u;
        for (uint32_t c = 0u; c < 28u; c += 4u) {
            u4 r = philox4x32(seed, ENV_RNG_STREAM, game, RF_TAG_CONFIG | (q << 8) | c, 0u, 0u);
            uint32_t words[4] = { r.x, r.y, r.z, r.w };
            for (uint32_t j = 0u; j < 4u && c + j < 25u; j++) config |= (uint32_t)(words[j] < threshold) << (c + j);
        }
        uint32_t rows[RF_SIDE] = {0};
        if (q < pairs) {
            relicfog_cover(rows, config, m->relicX[q], m->relicY[q]);
            relicfog_symmetrize(rows);
        }
        for (uint32_t y = 0u; y < RF_SIDE; y++) {
            m->weight[q][y] = rows[y] & ~covered[y];
            covered[y] |= rows[y];
        }
    }
}

static void relicfog_survey(relicfogSurvey *s, uint32_t seed) {
    uint32_t stage = s->stage;
    if (stage >= RF_SURVEY_DONE) return;
    if (stage < RF_GRADIENT_STAGES) {
        for (uint32_t g = stage * RF_GRADIENT_BATCH; g < RF_GRADIENTS && g < (stage + 1u) * RF_GRADIENT_BATCH; g += 4u) {
            u4 r = philox4x32(seed, ENV_RNG_STREAM, s->game, RF_TAG_GRADIENT | g, 0u, 0u);
            uint32_t words[4] = { r.x, r.y, r.z, r.w };
            for (uint32_t j = 0u; j < 4u; j++) oak_sincos(2.0f * OAK_PI * u01(words[j]), &s->gradient[g + j][1], &s->gradient[g + j][0]);
        }
        if (stage == 0u)
            for (uint32_t f = 0u; f < 2u; f++)
                for (uint32_t i = 0u; i < 3u; i++) {
                    s->top[f][i] = -INFINITY;
                    s->topAt[f][i] = 0u;
                }
    } else if (stage < RF_FINAL_STAGE) {
        uint32_t f = (stage - RF_GRADIENT_STAGES) / RF_SIDE, y = (stage - RF_GRADIENT_STAGES) % RF_SIDE, bits = 0u;
        uint32_t cell = f == RF_ASTEROID_FIELD ? 3u : 6u, shift = f < RF_NODE_FIELD ? 1u : 0u, stride = f == RF_ASTEROID_FIELD ? 10u : 6u;
        uint32_t base = f == RF_NEBULA_FIELD ? 0u : f == RF_ASTEROID_FIELD ? 36u : f == RF_NODE_FIELD ? 136u : 172u;
        for (uint32_t x = 0u; x < RF_SIDE; x++) {
            float value = relicfog_noise((const float (*)[2])(s->gradient + base), stride, cell, x + shift, y);
            if (f < RF_NODE_FIELD) bits |= (uint32_t)(value > RELICFOG_THRESHOLD) << x;
            else relicfog_rank(s, f - RF_NODE_FIELD, value, y * RF_SIDE + x);
        }
        if (f == RF_NEBULA_FIELD) s->rawNebula[y] = bits;
        if (f == RF_ASTEROID_FIELD) s->rawAsteroid[y] = bits;
    } else relicfog_finish(s, seed);
    s->stage = stage + 1u;
}

static void relicfog_lanes(relicfogWorld *w) {
    uint8x16_t bits = vld1q_u8(RELICFOG_LANE_BITS);
    for (uint32_t t = 0u; t < 2u; t++) vst1q_u8(w->live[t], vtstq_u8(vcombine_u8(vdup_n_u8((uint8_t)w->alive[t]), vdup_n_u8((uint8_t)(w->alive[t] >> 8))), bits));
}

static void relicfog_rescore(relicfogWorld *w) {
    uint32_t pairs = (uint32_t)__builtin_popcount(w->relicMask) / 2u;
    w->pairs = pairs;
    for (uint32_t y = 0u; y < RF_SIDE; y++) {
        uint32_t rows = 0u;
        for (uint32_t q = 0u; q < pairs && q < RF_PAIRS; q++) rows |= w->map.weight[q][y];
        w->score[y] = rows;
    }
}

static uint32_t relicfog_visible(const relicfogWorld *w, uint32_t team, uint32_t x, uint32_t y) {
    const relicfogParams *p = &w->map.params;
    uint8x16_t dx = vabdq_u8(vld1q_u8(w->ux[team]), vdupq_n_u8((uint8_t)x)), dy = vabdq_u8(vld1q_u8(w->uy[team]), vdupq_n_u8((uint8_t)y));
    uint8x16_t far = vmaxq_u8(dx, dy), power = vqsubq_u8(vdupq_n_u8((uint8_t)(1u + p->sensorRange)), far);
    power = vaddq_u8(power, vandq_u8(vceqzq_u8(far), vdupq_n_u8(RF_SELF_SIGHT)));
    uint32_t total = vaddlvq_u8(vandq_u8(power, vld1q_u8(w->live[team])));
    return total > relicfog_bit(w->sight, x, y) * p->visionReduction;
}

static uint32_t relicfog_beside(const relicfogWorld *w, uint32_t team, uint32_t x, uint32_t y) { return !relicfog_bit(w->sight, x, y) || relicfog_visible(w, team, x, y); }

static uint8x16_t relicfog_here(const relicfogWorld *w, uint32_t team, uint32_t x, uint32_t y) {
    uint8x16_t same = vandq_u8(vceqq_u8(vld1q_u8(w->ux[team]), vdupq_n_u8((uint8_t)x)), vceqq_u8(vld1q_u8(w->uy[team]), vdupq_n_u8((uint8_t)y)));
    return vandq_u8(same, vld1q_u8(w->live[team]));
}

static int32_t relicfog_sum(const int16_t *energy, uint8x16_t mask) {
    int8x16_t m = vreinterpretq_s8_u8(mask);
    int16x8_t lo = vandq_s16(vld1q_s16(energy), vmovl_s8(vget_low_s8(m))), hi = vandq_s16(vld1q_s16(energy + 8), vmovl_high_s8(m));
    return vaddlvq_s16(lo) + vaddlvq_s16(hi);
}

static uint32_t relicfog_lanesSet(uint8x16_t mask) { return vaddvq_u8(vandq_u8(mask, vdupq_n_u8(1u))); }

static uint32_t relicfog_blocked(const relicfogWorld *w, int x, int y) { return x < 0 || y < 0 || x >= RF_SIDE || y >= RF_SIDE || relicfog_bit(w->map.asteroid, (uint32_t)x, (uint32_t)y); }

static void relicfog_drift(relicfogWorld *w, uint32_t seed, uint32_t game, uint32_t steps) {
    const relicfogParams *p = &w->map.params;
    uint32_t speed = (uint32_t)(p->nebulaSpeed < 0 ? -p->nebulaSpeed : p->nebulaSpeed);
    if (steps == 0u || steps * speed / 40u != (steps - 1u) * speed / 40u) {
        uint32_t nebula[RF_SIDE], asteroid[RF_SIDE];
        for (uint32_t y = 0u; y < RF_SIDE; y++) {
            uint32_t from = p->nebulaSpeed > 0 ? (y + 1u) % RF_SIDE : (y + RF_EDGE) % RF_SIDE, n = w->map.nebula[from], a = w->map.asteroid[from];
            nebula[y] = (p->nebulaSpeed > 0 ? n << 1 | n >> RF_EDGE : n >> 1 | n << RF_EDGE) & 0xFFFFFFu;
            asteroid[y] = (p->nebulaSpeed > 0 ? a << 1 | a >> RF_EDGE : a >> 1 | a << RF_EDGE) & 0xFFFFFFu;
        }
        memcpy(w->map.nebula, nebula, sizeof nebula);
        memcpy(w->map.asteroid, asteroid, sizeof asteroid);
    }
    if (steps == 0u || steps * p->nodeSpeed / 100u != (steps - 1u) * p->nodeSpeed / 100u) {
        u4 r = philox4x32(seed, ENV_RNG_STREAM, game, RF_TAG_DRIFT | steps, 0u, 0u);
        int x = (int)w->map.nodeX + relicfog_offset(r.x, p->nodeMagnitude), y = (int)w->map.nodeY + relicfog_offset(r.y, p->nodeMagnitude);
        w->map.nodeX = (uint8_t)(x < 0 ? 0 : x > RF_EDGE ? RF_EDGE : x);
        w->map.nodeY = (uint8_t)(y < 0 ? 0 : y > RF_EDGE ? RF_EDGE : y);
    }
}

static void relicfog_resolve(relicfogWorld *w, const int8_t *field, int16_t (*after)[RF_UNITS]) {
    const relicfogParams *p = &w->map.params;
    uint16x8_t hurt[2][2] = { { vdupq_n_u16(0u), vdupq_n_u16(0u) }, { vdupq_n_u16(0u), vdupq_n_u16(0u) } };
    int32_t paid[2][RF_UNITS] = { { 0 } };
    for (uint32_t t = 0u; t < 2u; t++)
        for (uint32_t bits = w->alive[t]; bits; bits &= bits - 1u) {
            uint32_t i = (uint32_t)__builtin_ctz(bits);
            relicfogOrder o = w->order[t][i];
            int energy = w->energy[t][i];
            if (o.type >= RF_UP && o.type <= RF_LEFT && energy >= p->moveCost) {
                int dx, dy;
                relicfog_delta((uint32_t)o.type, &dx, &dy);
                int x = w->ux[t][i] + dx, y = w->uy[t][i] + dy;
                if (!relicfog_blocked(w, x, y)) {
                    w->ux[t][i] = (uint8_t)x;
                    w->uy[t][i] = (uint8_t)y;
                    energy -= p->moveCost;
                }
            }
            after[t][i] = (int16_t)energy;
        }
    for (uint32_t t = 0u; t < 2u; t++)
        for (uint32_t bits = w->alive[t]; bits; bits &= bits - 1u) {
            uint32_t i = (uint32_t)__builtin_ctz(bits);
            relicfogOrder o = w->order[t][i];
            if (o.type != RF_SAP || relicfog_span(o.dx, 0) > p->sapRange || relicfog_span(o.dy, 0) > p->sapRange || after[t][i] < p->sapCost) continue;
            int x = w->ux[t][i] + o.dx, y = w->uy[t][i] + o.dy;
            if (x < 0 || y < 0 || x >= RF_SIDE || y >= RF_SIDE) continue;
            paid[t][i] = p->sapCost;
            uint32_t o2 = 1u - t;
            uint8x16_t far = vmaxq_u8(vabdq_u8(vld1q_u8(w->ux[o2]), vdupq_n_u8((uint8_t)x)), vabdq_u8(vld1q_u8(w->uy[o2]), vdupq_n_u8((uint8_t)y)));
            int8x16_t direct = vreinterpretq_s8_u8(vceqzq_u8(far)), near = vreinterpretq_s8_u8(vceqq_u8(far, vdupq_n_u8(1u)));
            uint16x8_t full = vdupq_n_u16((uint16_t)(4u * p->sapCost)), part = vdupq_n_u16((uint16_t)(p->dropoff * p->sapCost));
            uint16x8_t dlo = vreinterpretq_u16_s16(vmovl_s8(vget_low_s8(direct))), dhi = vreinterpretq_u16_s16(vmovl_high_s8(direct));
            uint16x8_t nlo = vreinterpretq_u16_s16(vmovl_s8(vget_low_s8(near))), nhi = vreinterpretq_u16_s16(vmovl_high_s8(near));
            hurt[o2][0] = vaddq_u16(hurt[o2][0], vorrq_u16(vandq_u16(dlo, full), vandq_u16(nlo, part)));
            hurt[o2][1] = vaddq_u16(hurt[o2][1], vorrq_u16(vandq_u16(dhi, full), vandq_u16(nhi, part)));
        }
    uint16_t damage[2][RF_UNITS];
    for (uint32_t t = 0u; t < 2u; t++) {
        vst1q_u16(damage[t], hurt[t][0]);
        vst1q_u16(damage[t] + 8, hurt[t][1]);
    }
    uint32_t removed[2] = { 0u, 0u };
    int32_t drained[2][RF_UNITS];
    for (uint32_t t = 0u; t < 2u; t++)
        for (uint32_t bits = w->alive[t]; bits; bits &= bits - 1u) {
            uint32_t i = (uint32_t)__builtin_ctz(bits), o2 = 1u - t, x = w->ux[t][i], y = w->uy[t][i];
            uint8x16_t gap = vqaddq_u8(vabdq_u8(vld1q_u8(w->ux[o2]), vdupq_n_u8((uint8_t)x)), vabdq_u8(vld1q_u8(w->uy[o2]), vdupq_n_u8((uint8_t)y)));
            uint8x16_t live = vld1q_u8(w->live[o2]);
            drained[t][i] = 0;
            if (!vmaxvq_u8(vandq_u8(vcleq_u8(gap, vdupq_n_u8(1u)), live))) continue;
            uint8x16_t own = relicfog_here(w, t, x, y), same = vandq_u8(vceqzq_u8(gap), live), touch = vandq_u8(vceqq_u8(gap, vdupq_n_u8(1u)), live);
            if (vmaxvq_u8(same) && relicfog_sum(after[t], own) <= relicfog_sum(after[o2], same)) removed[t] |= 1u << i;
            drained[t][i] = relicfog_sum(after[o2], touch) * (int32_t)p->voidFactor / (16 * (int32_t)relicfog_lanesSet(own));
        }
    for (uint32_t t = 0u; t < 2u; t++) {
        for (uint32_t bits = w->alive[t] & ~removed[t]; bits; bits &= bits - 1u) {
            uint32_t i = (uint32_t)__builtin_ctz(bits), x = w->ux[t][i], y = w->uy[t][i];
            int32_t energy = after[t][i] - paid[t][i] - damage[t][i] / 4 - drained[t][i];
            int32_t fed = energy + field[y * RF_SIDE + x] - (int32_t)(relicfog_bit(w->map.nebula, x, y) * p->energyReduction);
            w->energy[t][i] = (int16_t)(fed >= 0 ? (fed > RF_MAX_ENERGY ? RF_MAX_ENERGY : fed) : energy < 0 ? energy : 0);
        }
        w->alive[t] &= ~removed[t];
    }
}

static int32_t relicfog_total(const relicfogWorld *w, uint32_t t) {
    int32_t total = 0;
    for (uint32_t bits = w->alive[t]; bits; bits &= bits - 1u) total += w->energy[t][__builtin_ctz(bits)];
    return total;
}

static uint32_t relicfog_occupied(const relicfogWorld *w, uint32_t t, uint32_t *rows) {
    for (uint32_t y = 0u; y < RF_SIDE; y++) rows[y] = 0u;
    for (uint32_t bits = w->alive[t]; bits; bits &= bits - 1u) {
        uint32_t i = (uint32_t)__builtin_ctz(bits);
        if (w->energy[t][i] >= 0) rows[w->uy[t][i]] |= 1u << w->ux[t][i];
    }
    uint32_t n = 0u;
    for (uint32_t y = 0u; y < RF_SIDE; y++) n += (uint32_t)__builtin_popcount(rows[y] & w->score[y]);
    return n;
}

static void relicfog_advanceWorld(relicfogWorld *w, uint32_t *words, uint32_t seed, const float *shared) {
    uint32_t steps = words[RW_STEP], match = words[RW_MATCH], game = words[RW_GAME];
    if (match == 0u) w->alive[0] = w->alive[1] = 0u;
    for (uint32_t t = 0u; t < 2u; t++)
        for (uint32_t bits = w->alive[t]; bits; bits &= bits - 1u)
            if (w->energy[t][__builtin_ctz(bits)] < 0) w->alive[t] &= ~(1u << __builtin_ctz(bits));
    relicfog_lanes(w);
    w->fieldAt = (uint32_t)w->map.nodeY * RF_SIDE + w->map.nodeX;
    for (uint32_t q = 0u; q < RF_PAIRS; q++)
        if (w->map.schedule[q] >= 0 && steps >= (uint32_t)w->map.schedule[q] && !(w->relicMask >> q & 1u)) {
            w->relicMask |= 9u << q;
            w->relicRows[w->map.relicY[q]] |= 1u << w->map.relicX[q];
            w->relicRows[w->map.relicY[q + RF_PAIRS]] |= 1u << w->map.relicX[q + RF_PAIRS];
            relicfog_rescore(w);
        }
    int16_t after[2][RF_UNITS] = { { 0 } };
    if (w->alive[0] | w->alive[1]) relicfog_resolve(w, relicfog_field(w, shared), after);
    if (match % RF_SPAWN_RATE == 0u)
        for (uint32_t t = 0u; t < 2u; t++) {
            uint32_t open = ~w->alive[t] & 0xFFFFu;
            if (!open) continue;
            uint32_t i = (uint32_t)__builtin_ctz(open);
            w->alive[t] |= 1u << i;
            w->ux[t][i] = w->uy[t][i] = (uint8_t)(t * RF_EDGE);
            w->energy[t][i] = RF_START_ENERGY;
        }
    relicfog_lanes(w);
    memcpy(w->sight, w->map.nebula, sizeof w->sight);
    relicfog_drift(w, seed, game, steps);
    uint32_t rows[RF_SIDE];
    for (uint32_t t = 0u; t < 2u; t++) w->gain[t] = (int32_t)relicfog_occupied(w, t, rows);
    memcpy(w->last, w->order, sizeof w->last);
    words[RW_OUR_POINTS] += (uint32_t)w->gain[0];
    words[RW_THEIR_POINTS] += (uint32_t)w->gain[1];
    words[RW_GAIN] = (uint32_t)w->gain[0];
    words[RW_STEP] = steps + 1u;
    words[RW_MATCH] = match + 1u;
}

static uint32_t relicfog_verdict(const relicfogWorld *w, const uint32_t *words, uint32_t seed) {
    uint32_t ours = words[RW_OUR_POINTS], theirs = words[RW_THEIR_POINTS];
    if (ours != theirs) return ours > theirs ? 0u : 1u;
    int32_t a = relicfog_total(w, 0u), b = relicfog_total(w, 1u);
    if (a != b) return a > b ? 0u : 1u;
    return philox4x32(seed, ENV_RNG_STREAM, words[RW_GAME], RF_TAG_TIE | words[RW_STEP], 0u, 0u).x >> 31;
}

static void relicfog_suspect(uint32_t *candidate, uint32_t *ruled, uint32_t x, uint32_t y) {
    uint32_t band = relicfog_band((int)x - 2, (int)x + 2);
    for (int v = (int)y - 2; v <= (int)y + 2; v++) {
        if (v < 0 || v >= RF_SIDE) continue;
        candidate[v] |= band;
        ruled[v] &= ~band;
    }
}

static void relicfog_brief(relicfogWorld *w, uint32_t seed, uint32_t game) {
    relicfogBot *b = &w->bot;
    for (uint32_t i = 0u; i < RF_RELICS; i++) {
        uint32_t x = w->map.relicX[i], y = w->map.relicY[i];
        if (!(w->relicMask >> i & 1u) || b->seen >> i & 1u || !relicfog_visible(w, 1u, x, y)) continue;
        b->seen |= 1u << i;
        relicfog_suspect(b->candidate, b->ruled, x, y);
    }
    uint32_t n = 0u;
    memset(b->enemyRows, 0, sizeof b->enemyRows);
    memset(b->enemyEnergy, 0, sizeof b->enemyEnergy);
    memset(b->planned, 0, sizeof b->planned);
    for (uint32_t bits = w->alive[0]; bits; bits &= bits - 1u) {
        uint32_t i = (uint32_t)__builtin_ctz(bits);
        if (!relicfog_visible(w, 1u, w->ux[0][i], w->uy[0][i])) continue;
        b->enemyX[n] = w->ux[0][i];
        b->enemyY[n] = w->uy[0][i];
        b->enemyEnergy[n] = w->energy[0][i];
        b->enemyRows[w->uy[0][i]] |= 1u << w->ux[0][i];
        n++;
    }
    b->enemies = n;
    u4 r = philox4x32(seed, ENV_RNG_STREAM, game, RF_TAG_BOT | b->turn, 0u, 0u);
    uint32_t words[4] = { r.x, r.y, r.z, r.w };
    for (uint32_t i = 0u; i < RF_UNITS; i++) b->jitter[i] = (uint8_t)(((words[i >> 2] >> (8u * (i & 3u)) & 0xFFu) * 5u) >> 8);
    b->turn++;
}

static void relicfog_deduce(const uint32_t *occupied, int32_t gain, const uint32_t *candidate, uint32_t *known, uint32_t *ruled) {
    uint32_t unknown[RF_SIDE], untested = 0u;
    int32_t explained = 0;
    for (uint32_t y = 0u; y < RF_SIDE; y++) {
        uint32_t tested = occupied[y] & candidate[y] & ~ruled[y];
        explained += __builtin_popcount(tested & known[y]);
        unknown[y] = tested & ~known[y];
        untested += (uint32_t)__builtin_popcount(unknown[y]);
    }
    int32_t rest = gain - explained;
    if (!untested) return;
    for (uint32_t y = 0u; y < RF_SIDE; y++) {
        if (rest <= 0) ruled[y] |= unknown[y];
        else if (rest >= (int32_t)untested) known[y] |= unknown[y];
    }
}

static void relicfog_infer(relicfogWorld *w) {
    relicfogBot *b = &w->bot;
    uint32_t occupied[RF_SIDE];
    relicfog_occupied(w, 1u, occupied);
    relicfog_deduce(occupied, w->gain[1], b->candidate, b->known, b->ruled);
}

static void relicfog_waypoint(const relicfogBot *b, uint32_t u, uint32_t *x, uint32_t *y) {
    uint32_t k = (u + b->turn / 25u) & 7u;
    *x = RELICFOG_WAYPOINTS[2u * k];
    *y = RELICFOG_WAYPOINTS[2u * k + 1u];
}

static void relicfog_assign(relicfogWorld *w) {
    relicfogBot *b = &w->bot;
    const float *theta = b->theta;
    uint32_t free = 0u, testers = 0u, explorers = 0u, limit = (uint32_t)theta[RF_TESTERS], scouts = (uint32_t)theta[RF_EXPLORERS];
    for (uint32_t bits = w->alive[1]; bits; bits &= bits - 1u) {
        uint32_t i = (uint32_t)__builtin_ctz(bits);
        b->task[i] = RF_IDLE;
        if ((float)w->energy[1][i] < theta[RF_RETREAT]) b->task[i] = RF_HARVEST;
        else free |= 1u << i;
    }
    for (uint32_t pass = 0u; pass < 2u; pass++)
        for (uint32_t y = 0u; y < RF_SIDE && free; y++) {
            uint32_t rows = pass ? b->candidate[y] & ~b->ruled[y] & ~b->known[y] : b->known[y];
            for (; rows && free; rows &= rows - 1u) {
                if (pass && testers >= limit) break;
                uint32_t x = (uint32_t)__builtin_ctz(rows), best = RF_NONE, far = 1000u;
                for (uint32_t bits = free; bits; bits &= bits - 1u) {
                    uint32_t i = (uint32_t)__builtin_ctz(bits), d = relicfog_span(w->ux[1][i], (int)x) + relicfog_span(w->uy[1][i], (int)y);
                    if (d < far) { far = d; best = i; }
                }
                free &= ~(1u << best);
                b->task[best] = (uint8_t)(pass ? RF_TEST : RF_HOLD);
                b->targetX[best] = (uint8_t)x;
                b->targetY[best] = (uint8_t)y;
                testers += pass;
            }
        }
    uint32_t anchorX = RF_EDGE, anchorY = RF_EDGE, seen = b->seen;
    if (seen) {
        uint32_t i = (uint32_t)__builtin_ctz(seen);
        anchorX = w->map.relicX[i];
        anchorY = w->map.relicY[i];
    }
    for (uint32_t bits = free; bits; bits &= bits - 1u) {
        uint32_t i = (uint32_t)__builtin_ctz(bits);
        if (explorers < scouts || !seen) {
            b->task[i] = RF_EXPLORE;
            uint32_t x, y;
            relicfog_waypoint(b, i, &x, &y);
            b->targetX[i] = (uint8_t)x;
            b->targetY[i] = (uint8_t)y;
            explorers++;
        } else {
            b->task[i] = RF_HOLD;
            b->targetX[i] = (uint8_t)anchorX;
            b->targetY[i] = (uint8_t)anchorY;
        }
    }
}

static relicfogOrder relicfog_step1(const relicfogWorld *w, const int8_t *field, uint32_t i, uint32_t tx, uint32_t ty, float harvest, float caution, const uint32_t *enemyRows, uint32_t jitter) {
    relicfogOrder order = { RF_CENTER, 0, 0 };
    int32_t energy = w->energy[1][i], cost = w->map.params.moveCost;
    if (energy < cost) return order;
    int x = w->ux[1][i], y = w->uy[1][i];
    float best = -1e9f;
    for (uint32_t k = 0u, a = jitter; k < 5u; k++, a = a == 4u ? 0u : a + 1u) {
        int dx, dy;
        relicfog_delta(a, &dx, &dy);
        int nx = x + dx, ny = y + dy;
        if (a && relicfog_blocked(w, nx, ny)) continue;
        float score = -4.0f * (float)(relicfog_span(nx, (int)tx) + relicfog_span(ny, (int)ty));
        if (harvest != 0.0f && relicfog_beside(w, 1u, (uint32_t)nx, (uint32_t)ny)) score += harvest * 0.25f * (float)field[ny * RF_SIDE + nx];
        if (enemyRows && relicfog_bit(enemyRows, (uint32_t)nx, (uint32_t)ny) && (float)relicfog_sum(w->energy[0], relicfog_here(w, 0u, (uint32_t)nx, (uint32_t)ny)) >= (float)(energy - (a ? cost : 0)) - caution) score -= 1000.0f;
        if (score > best) {
            best = score;
            order.type = (int8_t)a;
        }
    }
    return order;
}

static relicfogOrder relicfog_sapper(relicfogWorld *w, const int8_t *field, uint32_t i) {
    relicfogBot *b = &w->bot;
    const float *theta = b->theta;
    const relicfogParams *p = &w->map.params;
    int32_t energy = w->energy[1][i], cost = p->sapCost;
    uint32_t x = w->ux[1][i], y = w->uy[1][i];
    if (b->enemies && (float)energy >= (float)cost + theta[RF_RESERVE]) {
        uint8x16_t ex = vld1q_u8(b->enemyX), ey = vld1q_u8(b->enemyY);
        int16x8_t left0 = vmaxq_s16(vsubq_s16(vld1q_s16(b->enemyEnergy), vld1q_s16(b->planned)), vdupq_n_s16(0));
        int16x8_t left1 = vmaxq_s16(vsubq_s16(vld1q_s16(b->enemyEnergy + 8), vld1q_s16(b->planned + 8)), vdupq_n_s16(0));
        int16x8_t full = vdupq_n_s16((int16_t)cost), part = vdupq_n_s16((int16_t)(theta[RF_DROPOFF_GUESS] * (float)cost));
        int32_t best = (int32_t)(theta[RF_SAP_RATIO] * (float)cost), pick = -1;
        for (uint32_t n = 0u; n < b->enemies; n++) {
            if (relicfog_span(b->enemyX[n], (int)x) > p->sapRange || relicfog_span(b->enemyY[n], (int)y) > p->sapRange) continue;
            uint8x16_t far = vmaxq_u8(vabdq_u8(ex, vdupq_n_u8(b->enemyX[n])), vabdq_u8(ey, vdupq_n_u8(b->enemyY[n])));
            int8x16_t direct = vreinterpretq_s8_u8(vceqzq_u8(far)), near = vreinterpretq_s8_u8(vceqq_u8(far, vdupq_n_u8(1u)));
            int16x8_t v0 = vorrq_s16(vandq_s16(vminq_s16(left0, full), vmovl_s8(vget_low_s8(direct))), vandq_s16(vminq_s16(left0, part), vmovl_s8(vget_low_s8(near))));
            int16x8_t v1 = vorrq_s16(vandq_s16(vminq_s16(left1, full), vmovl_high_s8(direct)), vandq_s16(vminq_s16(left1, part), vmovl_high_s8(near)));
            int32_t value = vaddlvq_s16(v0) + vaddlvq_s16(v1);
            if (value > best) {
                best = value;
                pick = (int32_t)n;
            }
        }
        if (pick >= 0) {
            for (uint32_t m = 0u; m < b->enemies; m++) {
                uint32_t dx = relicfog_span(b->enemyX[m], b->enemyX[pick]), dy = relicfog_span(b->enemyY[m], b->enemyY[pick]), far = dx > dy ? dx : dy;
                if (far <= 1u) b->planned[m] = (int16_t)(b->planned[m] + (far ? (int32_t)(theta[RF_DROPOFF_GUESS] * (float)cost) : cost));
            }
            return (relicfogOrder){ RF_SAP, (int8_t)((int)b->enemyX[pick] - (int)x), (int8_t)((int)b->enemyY[pick] - (int)y) };
        }
    }
    uint32_t tx = b->targetX[i], ty = b->targetY[i];
    float harvest = theta[RF_HARVEST_WEIGHT];
    if (b->task[i] == RF_HARVEST || b->task[i] == RF_IDLE) {
        tx = x;
        ty = y;
        harvest *= 4.0f;
    }
    if (b->task[i] == RF_HOLD && tx == x && ty == y && relicfog_bit(b->known, x, y)) return (relicfogOrder){ RF_CENTER, 0, 0 };
    return relicfog_step1(w, field, i, tx, ty, harvest, theta[RF_CAUTION], b->enemyRows, b->jitter[i]);
}

static relicfogOrder relicfog_scout(relicfogWorld *w, const int8_t *field, uint32_t i) {
    relicfogBot *b = &w->bot;
    uint32_t tx, ty, seen = b->seen;
    if (seen) {
        uint32_t count = (uint32_t)__builtin_popcount(seen), pick = i % count, r = seen;
        for (; pick; pick--) r &= r - 1u;
        uint32_t relic = (uint32_t)__builtin_ctz(r), spot = (i * 7u + b->turn / 10u) % 25u;
        int x = (int)w->map.relicX[relic] + (int)(spot % 5u) - 2, y = (int)w->map.relicY[relic] + (int)(spot / 5u) - 2;
        tx = (uint32_t)(x < 0 ? 0 : x > RF_EDGE ? RF_EDGE : x);
        ty = (uint32_t)(y < 0 ? 0 : y > RF_EDGE ? RF_EDGE : y);
    } else relicfog_waypoint(b, i, &tx, &ty);
    return relicfog_step1(w, field, i, tx, ty, 0.0f, 0.0f, 0, b->jitter[i]);
}

static void relicfog_think(const relicfogKnobs *k, relicfogWorld *w, const uint32_t *words, uint32_t seed, const float *shared) {
    if (words[RW_MATCH] == 0u) return;
    uint32_t sapper = k->opponent != RF_SCOUT;
    relicfog_brief(w, seed, words[RW_GAME]);
    if (sapper) {
        relicfog_infer(w);
        relicfog_assign(w);
    }
    const int8_t *field = relicfog_field(w, shared);
    for (uint32_t bits = w->alive[1]; bits; bits &= bits - 1u) {
        uint32_t i = (uint32_t)__builtin_ctz(bits);
        if (w->energy[1][i] >= 0) w->order[1][i] = sapper ? relicfog_sapper(w, field, i) : relicfog_scout(w, field, i);
    }
}

static void relicfog_open(relicfogWorld *w, uint32_t *words) {
    uint32_t roster = 0u;
    if (words[RW_MATCH] != 0u)
        for (uint32_t bits = w->alive[0]; bits; bits &= bits - 1u) {
            uint32_t i = (uint32_t)__builtin_ctz(bits);
            if (w->energy[0][i] >= w->map.params.moveCost) roster |= 1u << i;
        }
    words[RW_ROSTER] = roster;
    words[RW_CURSOR] = roster ? (uint32_t)__builtin_ctz(roster) : RF_NONE;
    words[RW_MODE] = roster ? RF_ORDER : RF_WAIT;
    words[RW_FOCUS] = RF_CENTERED;
    words[RW_CLOCK] = 0u;
    memset(w->order, 0, sizeof w->order);
}

static void relicfog_begin(const relicfogKnobs *k, relicfogWorld *w, uint32_t *words, uint32_t seed, uint32_t game) {
    while (w->survey.stage < RF_SURVEY_DONE) relicfog_survey(&w->survey, seed);
    w->map = w->survey.map;
    memset(&w->survey, 0, sizeof w->survey);
    w->survey.game = game + 1u;
    w->alive[0] = w->alive[1] = 0u;
    memset(w->energy, 0, sizeof w->energy);
    memset(w->ux, 0, sizeof w->ux);
    memset(w->uy, 0, sizeof w->uy);
    memset(w->relicRows, 0, sizeof w->relicRows);
    relicfog_lanes(w);
    w->relicMask = 0u;
    relicfog_rescore(w);
    w->fieldAt = (uint32_t)w->map.nodeY * RF_SIDE + w->map.nodeX;
    w->gain[0] = w->gain[1] = 0;
    memset(w->last, 0, sizeof w->last);
    memset(&w->bot, 0, sizeof w->bot);
    uint32_t member = 0u;
    if (k->opponent == RF_LEAGUE) member = leaguePick(w->winRate, philox4x32(seed, LEAGUE_RNG_STREAM, game, RF_TAG_LEAGUE, 0u, 0u).x);
    for (uint32_t d = 0u; d < RF_THETA; d++) w->bot.theta[d] = k->opponent == RF_LEAGUE ? w->theta[member * RF_THETA + d] : RELICFOG_THETA[d];
    memset(&w->oracle, 0, sizeof w->oracle);
    w->oracle.key = 0xFFFFFFFFu;
    words[RW_GAME] = game;
    words[RW_STEP] = words[RW_MATCH] = 0u;
    words[RW_OUR_POINTS] = words[RW_THEIR_POINTS] = words[RW_OUR_WINS] = words[RW_THEIR_WINS] = words[RW_GAIN] = 0u;
    words[RW_MEMBER] = member;
    relicfog_open(w, words);
}

static float relicfog_tick(const relicfogKnobs *k, relicfogWorld *w, uint32_t *words, uint32_t seed, const float *shared, uint32_t *done) {
    relicfog_advanceWorld(w, words, seed, shared);
    float reward = 0.0f;
    if (words[RW_MATCH] > RF_MATCH_STEPS) {
        uint32_t winner = relicfog_verdict(w, words, seed);
        words[winner ? RW_THEIR_WINS : RW_OUR_WINS]++;
        reward = winner ? 0.0f : 1.0f;
        words[RW_OUR_POINTS] = words[RW_THEIR_POINTS] = 0u;
        words[RW_MATCH] = 0u;
    }
    if (words[RW_STEP] >= RF_GAME_STEPS) {
        *done = 1u;
        uint32_t game = words[RW_GAME];
        if (k->opponent == RF_LEAGUE) {
            leagueScore(w->winRate, words[RW_MEMBER], words[RW_OUR_WINS] > words[RW_THEIR_WINS] ? 1.0f : 0.0f);
            leagueRenew(w->winRate, w->theta, RF_THETA, RELICFOG_THETA_LOW, RELICFOG_THETA_HIGH, seed, game);
        }
        relicfog_begin(k, w, words, seed, game + 1u);
        return reward;
    }
    relicfog_open(w, words);
    return reward;
}

static float relicfog_settle(const relicfogKnobs *k, relicfogWorld *w, uint32_t *words, uint32_t seed, const float *shared, uint32_t *done) {
    float reward = 0.0f;
    while (words[RW_MODE] == RF_WAIT) {
        relicfog_think(k, w, words, seed, shared);
        relicfog_survey(&w->survey, seed);
        reward += relicfog_tick(k, w, words, seed, shared, done);
    }
    return reward;
}

static void relicfog_press(relicfogWorld *w, uint32_t *words, uint32_t action) {
    uint32_t mode = words[RW_MODE], u = words[RW_CURSOR], aimed = words[RW_FOCUS] >> RF_AIM_SHIFT;
    int fx = (int)(words[RW_FOCUS] & 15u) - RF_FOCUS_CENTER, fy = (int)(words[RW_FOCUS] >> 4 & 15u) - RF_FOCUS_CENTER, reach = w->map.params.sapRange;
    if (mode == RF_ORDER && action == RF_SAP) {
        words[RW_MODE] = RF_TARGET;
        words[RW_FOCUS] = aimed << RF_AIM_SHIFT | RF_CENTERED;
        return;
    }
    if (mode == RF_TARGET && action != RF_CENTER && action != RF_SAP && aimed <= 2u * (uint32_t)reach) {
        int dx, dy;
        relicfog_delta(action, &dx, &dy);
        int nx = fx + dx, ny = fy + dy, x = w->ux[0][u] + nx, y = w->uy[0][u] + ny;
        uint32_t inside = nx >= -reach && nx <= reach && ny >= -reach && ny <= reach && x >= 0 && y >= 0 && x < RF_SIDE && y < RF_SIDE;
        words[RW_FOCUS] = (aimed + 1u) << RF_AIM_SHIFT | (inside ? (uint32_t)(nx + RF_FOCUS_CENTER) | (uint32_t)(ny + RF_FOCUS_CENTER) << 4 : RF_CENTERED);
        if (!inside) words[RW_MODE] = RF_ORDER;
        return;
    }
    w->order[0][u] = mode == RF_TARGET ? (relicfogOrder){ RF_SAP, (int8_t)fx, (int8_t)fy } : (relicfogOrder){ (int8_t)action, 0, 0 };
    uint32_t roster = words[RW_ROSTER] & ~(1u << u);
    words[RW_ROSTER] = roster;
    words[RW_CURSOR] = roster ? (uint32_t)__builtin_ctz(roster) : RF_NONE;
    words[RW_MODE] = roster ? RF_ORDER : RF_WAIT;
    words[RW_FOCUS] = RF_CENTERED;
}

static void relicfog_observeWorld(const relicfogWorld *w, const uint32_t *words, float *obs, const float *shared) {
    const relicfogParams *p = &w->map.params;
    uint32_t u = words[RW_CURSOR], target = words[RW_MODE] == RF_TARGET;
    int32_t v[RO_COUNT] = {0}, foe[RF_PATCH_CELLS] = {0};
    int x = w->ux[0][u], y = w->uy[0][u];
    int fx = target ? (int)(words[RW_FOCUS] & 15u) - RF_FOCUS_CENTER : 0, fy = target ? (int)(words[RW_FOCUS] >> 4 & 15u) - RF_FOCUS_CENTER : 0;
    int px = x + fx, py = y + fy;
    uint8_t seen[RF_PATCH_CELLS] = {0};
    v[RO_MODE] = (int32_t)target;
    v[RO_ENERGY] = w->energy[0][u];
    v[RO_X] = x;
    v[RO_Y] = y;
    v[RO_FX] = fx;
    v[RO_FY] = fy;
    for (uint32_t c = 0u; c < RF_PATCH_CELLS; c++) {
        int cx = px + (int)(c % RF_PATCH) - RF_REACH, cy = py + (int)(c / RF_PATCH) - RF_REACH;
        if (cx < 0 || cy < 0 || cx >= RF_SIDE || cy >= RF_SIDE) {
            v[RO_TERRAIN + c] = RF_OFF_MAP;
            continue;
        }
        seen[c] = (uint8_t)relicfog_visible(w, 0u, (uint32_t)cx, (uint32_t)cy);
        if (!seen[c]) continue;
        uint32_t kind = relicfog_bit(w->map.nebula, (uint32_t)cx, (uint32_t)cy) ? RF_NEBULA : relicfog_bit(w->map.asteroid, (uint32_t)cx, (uint32_t)cy) ? RF_ASTEROID : RF_OPEN;
        v[RO_TERRAIN + c] = (int32_t)(kind + relicfog_bit(w->relicRows, (uint32_t)cx, (uint32_t)cy) * RF_RELIC);
    }
    if (seen[RF_PATCH_CELLS / 2u]) v[RO_FIELD] = relicfog_field(w, shared)[py * RF_SIDE + px];
    for (uint32_t t = 0u; t < 2u; t++)
        for (uint32_t bits = w->alive[t]; bits; bits &= bits - 1u) {
            uint32_t i = (uint32_t)__builtin_ctz(bits);
            uint32_t dx = (uint32_t)(w->ux[t][i] - px + RF_REACH), dy = (uint32_t)(w->uy[t][i] - py + RF_REACH), c = dy * RF_PATCH + dx;
            if (w->energy[t][i] < 0 || dx >= RF_PATCH || dy >= RF_PATCH || (t && !seen[c])) continue;
            (t ? foe : v + RO_SHIPS)[c] += w->energy[t][i] + 1;
        }
    for (uint32_t c = 0u; c < RF_PATCH_CELLS; c++) v[RO_SHIPS + c] = v[RO_SHIPS + c] ? v[RO_SHIPS + c] : -foe[c];
    v[RO_MATCH] = (int32_t)words[RW_MATCH];
    v[RO_ROUND] = (int32_t)(words[RW_STEP] / RF_MATCH_FRAMES);
    v[RO_GAIN] = (int32_t)words[RW_GAIN];
    v[RO_LEAD] = (int32_t)words[RW_OUR_POINTS] - (int32_t)words[RW_THEIR_POINTS];
    v[RO_MOVE_COST] = p->moveCost;
    v[RO_SAP_COST] = p->sapCost;
    v[RO_SAP_RANGE] = p->sapRange;
    for (uint32_t i = 0u; i < RO_COUNT; i += 4u) vst1q_f32(obs + i, vcvtq_f32_s32(vld1q_s32(v + i)));
}

static uint32_t relicfog_action(const float *ctrl) {
    float a = ctrl[0];
    return a >= 0.0f && a < (float)RF_ACTIONS ? (uint32_t)a : 0u;
}

static inline __attribute__((always_inline)) float relicfog_advance(const relicfogKnobs *k, uint32_t seed, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done, const float *shared) {
    relicfogWorld *w = relicfog_world(extra);
    *done = 0u;
    relicfog_press(w, words, relicfog_action(ctrl));
    words[RW_CLOCK]++;
    float reward = relicfog_settle(k, w, words, seed, shared, done);
    relicfog_observeWorld(w, words, obs, shared);
    return reward;
}

static void relicfog_league(relicfogWorld *w, uint32_t seed) {
    for (uint32_t i = 0u; i < LEAGUE_MEMBERS; i++) w->winRate[i] = 0.5f;
    for (uint32_t d = 0u; d < RF_THETA; d++) w->theta[d] = RELICFOG_THETA[d];
    for (uint32_t i = 1u; i < LEAGUE_MEMBERS; i++)
        for (uint32_t d = 0u; d < RF_THETA; d += 4u) {
            u44 r = philox4(seed, LEAGUE_RNG_STREAM, (lane4)(0xFFFFFFF0u + i), d + (lane4){ 0u, 1u, 2u, 3u }, (lane4)0u, (lane4)0u);
            float4 n = normal4(r.x, r.y);
            for (uint32_t j = 0u; j < 4u; j++) {
                float value = RELICFOG_THETA[d + j] + 0.1f * (RELICFOG_THETA_HIGH[d + j] - RELICFOG_THETA_LOW[d + j]) * n[j];
                w->theta[RF_THETA * i + d + j] = fminf(fmaxf(value, RELICFOG_THETA_LOW[d + j]), RELICFOG_THETA_HIGH[d + j]);
            }
        }
}

void relicfog_reset(uint32_t instance, uint32_t seed, uint32_t *words, uint32_t *extra, uint32_t steps) {
    (void)steps;
    const relicfogKnobs *k = &RELICFOG_KNOBS[instance];
    relicfogWorld *w = relicfog_world(extra);
    memset(w, 0, sizeof *w);
    for (uint32_t i = 0u; i < RW_WORDS; i++) words[i] = 0u;
    if (k->opponent == RF_LEAGUE) relicfog_league(w, seed);
    relicfog_begin(k, w, words, seed, 0u);
    uint32_t done = 0u;
    relicfog_settle(k, w, words, seed, 0, &done);
}

float relicfog_step(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done, const float *shared) {
    (void)step;
    switch (instance) {
        case 0u: return relicfog_advance(&RELICFOG_KNOBS[0], seed, ctrl, words, extra, obs, done, shared);
        case 1u: return relicfog_advance(&RELICFOG_KNOBS[1], seed, ctrl, words, extra, obs, done, shared);
        default: return relicfog_advance(&RELICFOG_KNOBS[2], seed, ctrl, words, extra, obs, done, shared);
    }
}

void relicfog_observe(uint32_t instance, uint32_t seed, const uint32_t *words, const uint32_t *extra, uint32_t step, float *obs, const float *shared) {
    (void)instance; (void)seed; (void)step;
    relicfog_observeWorld(relicfog_read(extra), words, obs, shared);
}

static uint32_t relicfog_flood(const uint32_t *open, uint32_t sx, uint32_t sy, uint32_t gx, uint32_t gy, uint32_t (*layer)[RF_SIDE]) {
    uint32_t seen[RF_SIDE] = {0};
    memset(layer[0], 0, sizeof layer[0]);
    layer[0][sy] = seen[sy] = 1u << sx;
    for (uint32_t n = 1u; n < RF_LAYERS; n++) {
        const uint32_t *f = layer[n - 1u];
        uint32_t any = 0u;
        for (uint32_t y = 0u; y < RF_SIDE; y++) {
            uint32_t grow = (f[y] << 1 | f[y] >> 1 | (y ? f[y - 1u] : 0u) | (y + 1u < RF_SIDE ? f[y + 1u] : 0u)) & open[y] & ~seen[y] & 0xFFFFFFu;
            layer[n][y] = grow;
            seen[y] |= grow;
            any |= grow;
        }
        if (relicfog_bit(layer[n], gx, gy)) return n;
        if (!any) break;
    }
    return 0u;
}

static int32_t relicfog_route(const relicfogWorld *w, const int8_t *field, const uint32_t *open, const uint32_t *enemyRows, uint32_t i, uint32_t tx, uint32_t ty, uint32_t (*layer)[RF_SIDE]) {
    uint32_t x = w->ux[0][i], y = w->uy[0][i], rows[RF_SIDE];
    memcpy(rows, open, sizeof rows);
    rows[y] |= 1u << x;
    uint32_t d = relicfog_flood(rows, tx, ty, x, y, layer);
    if (!d) return -1;
    int32_t best = -1, value = -1000, energy = w->energy[0][i] - w->map.params.moveCost;
    for (uint32_t a = RF_UP; a <= RF_LEFT; a++) {
        int dx, dy;
        relicfog_delta(a, &dx, &dy);
        int nx = (int)x + dx, ny = (int)y + dy;
        if (nx < 0 || ny < 0 || nx >= RF_SIDE || ny >= RF_SIDE || !relicfog_bit(layer[d - 1u], (uint32_t)nx, (uint32_t)ny)) continue;
        int32_t score = field[ny * RF_SIDE + nx];
        if (relicfog_bit(enemyRows, (uint32_t)nx, (uint32_t)ny) && relicfog_sum(w->energy[1], relicfog_here(w, 1u, (uint32_t)nx, (uint32_t)ny)) >= energy) score -= 100;
        if (score > value) { value = score; best = (int32_t)a; }
    }
    return best;
}

static void relicfog_witness(relicfogWorld *w, const int8_t *field) {
    relicfogOracle *o = &w->oracle;
    uint32_t blank = 0u;
    for (uint32_t y = 0u; y < RF_SIDE; y++)
        for (uint32_t x = 0u; x < RF_SIDE; x++) {
            if (!relicfog_visible(w, 0u, x, y)) continue;
            uint32_t mx = RF_EDGE - y, my = RF_EDGE - x, rock = relicfog_bit(w->map.asteroid, x, y);
            o->seen[y] |= 1u << x;
            o->rock[y] = (o->rock[y] & ~(1u << x)) | rock << x;
            o->rock[my] = (o->rock[my] & ~(1u << mx)) | rock << mx;
            o->energy[y * RF_SIDE + x] = o->energy[my * RF_SIDE + mx] = field[y * RF_SIDE + x];
            if (!relicfog_bit(w->relicRows, x, y) || relicfog_bit(o->relics, x, y)) continue;
            o->relics[y] |= 1u << x;
            o->relics[my] |= 1u << mx;
            relicfog_suspect(o->candidate, o->ruled, x, y);
            relicfog_suspect(o->candidate, o->ruled, mx, my);
        }
    relicfog_symmetrize(o->seen);
    for (uint32_t y = 0u; y < RF_SIDE; y++) blank |= ~o->seen[y] & 0xFFFFFFu;
    if (!blank) memset(o->seen, 0, sizeof o->seen);
}

static uint32_t relicfog_nearest(const uint32_t *rows, uint32_t x, uint32_t y, uint32_t *cx, uint32_t *cy) {
    uint32_t far = 1000u;
    for (uint32_t v = 0u; v < RF_SIDE; v++)
        for (uint32_t bits = rows[v]; bits; bits &= bits - 1u) {
            uint32_t u = (uint32_t)__builtin_ctz(bits), d = relicfog_span((int)x, (int)u) + relicfog_span((int)y, (int)v);
            if (d < far) { far = d; *cx = u; *cy = v; }
        }
    return far < 1000u;
}

static void relicfog_claim(const relicfogWorld *w, uint32_t *free, uint32_t *open, uint32_t limit, uint8_t *tx, uint8_t *ty) {
    for (uint32_t n = 0u; n < limit && *free; n++) {
        uint32_t best = RF_NONE, bx = 0u, by = 0u, far = 1000u;
        for (uint32_t y = 0u; y < RF_SIDE; y++)
            for (uint32_t rows = open[y]; rows; rows &= rows - 1u) {
                uint32_t x = (uint32_t)__builtin_ctz(rows);
                for (uint32_t bits = *free; bits; bits &= bits - 1u) {
                    uint32_t i = (uint32_t)__builtin_ctz(bits), d = relicfog_span(w->ux[0][i], (int)x) + relicfog_span(w->uy[0][i], (int)y);
                    if (d < far) { far = d; best = i; bx = x; by = y; }
                }
            }
        if (best == RF_NONE) return;
        *free &= ~(1u << best);
        open[by] &= ~(1u << bx);
        tx[best] = (uint8_t)bx;
        ty[best] = (uint8_t)by;
    }
}

static void relicfog_plan(relicfogWorld *w, const uint32_t *words, const float *shared) {
    relicfogOracle *o = &w->oracle;
    const relicfogParams *p = &w->map.params;
    const int32_t splash = (int32_t)(p->sapCost * RELICFOG_DROPOFF_QUARTERS[0] / 4u);
    uint32_t occupied[RF_SIDE], hold[RF_SIDE], open[RF_SIDE], probe[RF_SIDE], lure[RF_SIDE], passable[RF_SIDE], good[RF_SIDE], unseen[RF_SIDE], enemyRows[RF_SIDE] = {0}, mine[RF_SIDE] = {0};
    uint32_t layer[RF_LAYERS][RF_SIDE], sighted = 0u, free = 0u, relics = 0u;
    relicfog_witness(w, relicfog_field(w, shared));
    relicfog_occupied(w, 0u, occupied);
    relicfog_deduce(occupied, (int32_t)words[RW_GAIN], o->candidate, o->known, o->ruled);
    relicfog_symmetrize(o->known);
    relicfog_symmetrize(o->ruled);
    for (uint32_t y = 0u; y < RF_SIDE; y++) {
        uint32_t drain = 0u;
        for (uint32_t x = 0u; x < RF_SIDE; x++) drain |= (uint32_t)(o->energy[y * RF_SIDE + x] < -1) << x;
        hold[y] = open[y] = o->known[y] & ~o->ruled[y];
        probe[y] = o->candidate[y] & ~o->known[y] & ~o->ruled[y];
        lure[y] = hold[y] | probe[y] | o->relics[y];
        passable[y] = ~o->rock[y] & 0xFFFFFFu;
        good[y] = (passable[y] & ~drain) | hold[y] | probe[y];
        unseen[y] = ~o->seen[y] & 0xFFFFFFu;
        relics += (uint32_t)__builtin_popcount(o->relics[y]);
    }
    for (uint32_t bits = w->alive[1]; bits; bits &= bits - 1u) {
        uint32_t j = (uint32_t)__builtin_ctz(bits);
        if (!relicfog_visible(w, 0u, w->ux[1][j], w->uy[1][j])) continue;
        sighted |= 1u << j;
        enemyRows[w->uy[1][j]] |= 1u << w->ux[1][j];
        o->left[j] = w->energy[1][j];
    }
    for (uint32_t bits = w->alive[0]; bits; bits &= bits - 1u) {
        uint32_t i = (uint32_t)__builtin_ctz(bits);
        o->plan[i] = (relicfogOrder){ RF_CENTER, 0, 0 };
        if (w->energy[0][i] >= 0) free |= 1u << i;
    }
    uint8_t tx[RF_UNITS] = {0}, ty[RF_UNITS] = {0};
    uint32_t held = free;
    relicfog_claim(w, &free, open, RF_UNITS, tx, ty);
    relicfog_claim(w, &free, probe, 1u, tx, ty);
    held &= ~free;
    for (uint32_t bits = held; bits; bits &= bits - 1u) {
        uint32_t i = (uint32_t)__builtin_ctz(bits);
        mine[ty[i]] |= 1u << tx[i];
    }
    uint32_t round = words[RW_STEP] / RF_MATCH_FRAMES, explore = !relics || (round < RF_PAIRS && relics < 2u * (round + 1u)), reach = p->sensorRange;
    for (uint32_t bits = free; bits; bits &= bits - 1u) {
        uint32_t i = (uint32_t)__builtin_ctz(bits), x = w->ux[0][i], y = w->uy[0][i], cx = RF_SIDE / 2u, cy = RF_SIDE / 2u;
        if (explore && relicfog_nearest(unseen, x, y, &cx, &cy)) {
            tx[i] = (uint8_t)cx;
            ty[i] = (uint8_t)cy;
            uint32_t band = relicfog_band((int)cx - (int)reach, (int)cx + (int)reach);
            for (int v = (int)cy - (int)reach; v <= (int)cy + (int)reach; v++)
                if (v >= 0 && v < RF_SIDE) unseen[v] &= ~band;
            continue;
        }
        relicfog_nearest(lure, x, y, &cx, &cy);
        int32_t value = -1000;
        tx[i] = (uint8_t)cx;
        ty[i] = (uint8_t)cy;
        for (int v = (int)cy - 2; v <= (int)cy + 2; v++)
            for (int u = (int)cx - 2; u <= (int)cx + 2; u++) {
                if (u < 0 || v < 0 || u >= RF_SIDE || v >= RF_SIDE || !relicfog_bit(passable, (uint32_t)u, (uint32_t)v) || relicfog_bit(mine, (uint32_t)u, (uint32_t)v) || relicfog_bit(probe, (uint32_t)u, (uint32_t)v)) continue;
                if (o->energy[v * RF_SIDE + u] > value) { value = o->energy[v * RF_SIDE + u]; tx[i] = (uint8_t)u; ty[i] = (uint8_t)v; }
            }
        mine[ty[i]] |= 1u << tx[i];
    }
    for (uint32_t bits = w->alive[0]; bits; bits &= bits - 1u) {
        uint32_t i = (uint32_t)__builtin_ctz(bits);
        int32_t energy = w->energy[0][i];
        if (energy < 0) continue;
        uint32_t x = w->ux[0][i], y = w->uy[0][i];
        if (energy >= p->sapCost + 10) {
            int32_t best = held >> i & 1u && tx[i] == x && ty[i] == y ? p->sapCost : p->sapCost - p->sapCost / 4, pick = -1;
            for (uint32_t bits1 = sighted; bits1; bits1 &= bits1 - 1u) {
                uint32_t j = (uint32_t)__builtin_ctz(bits1);
                if (relicfog_span(w->ux[1][j], (int)x) > p->sapRange || relicfog_span(w->uy[1][j], (int)y) > p->sapRange) continue;
                int32_t value = 0;
                for (uint32_t bits2 = sighted; bits2; bits2 &= bits2 - 1u) {
                    uint32_t m = (uint32_t)__builtin_ctz(bits2), dx = relicfog_span(w->ux[1][m], w->ux[1][j]), dy = relicfog_span(w->uy[1][m], w->uy[1][j]), far = dx > dy ? dx : dy;
                    if (far > 1u || o->left[m] < 0) continue;
                    int32_t hit = far ? splash : p->sapCost, left = o->left[m] + 1;
                    value += hit < left ? hit : left;
                    if (relicfog_bit(hold, w->ux[1][m], w->uy[1][m]) && hit >= left) value += p->sapCost;
                }
                if (value > best) { best = value; pick = (int32_t)j; }
            }
            if (pick >= 0) {
                for (uint32_t bits2 = sighted; bits2; bits2 &= bits2 - 1u) {
                    uint32_t m = (uint32_t)__builtin_ctz(bits2), dx = relicfog_span(w->ux[1][m], w->ux[1][pick]), dy = relicfog_span(w->uy[1][m], w->uy[1][pick]), far = dx > dy ? dx : dy;
                    if (far <= 1u) o->left[m] = (int16_t)(o->left[m] - (far ? splash : p->sapCost));
                }
                o->plan[i] = (relicfogOrder){ RF_SAP, (int8_t)((int)w->ux[1][pick] - (int)x), (int8_t)((int)w->uy[1][pick] - (int)y) };
                continue;
            }
        }
        if ((tx[i] == x && ty[i] == y) || energy < p->moveCost) continue;
        int32_t move = relicfog_route(w, o->energy, good, enemyRows, i, tx[i], ty[i], layer);
        if (move < 0) move = relicfog_route(w, o->energy, passable, enemyRows, i, tx[i], ty[i], layer);
        if (move >= 0) o->plan[i] = (relicfogOrder){ (int8_t)move, 0, 0 };
    }
}

void relicfog_oracle(uint32_t instance, uint32_t seed, uint32_t step, const uint32_t *words, uint32_t *extra, const float *obs, float *ctrl, const float *shared) {
    (void)instance; (void)seed; (void)step; (void)obs;
    relicfogWorld *w = relicfog_world(extra);
    uint32_t key = words[RW_GAME] << 10 | words[RW_STEP];
    if (w->oracle.key != key) {
        relicfog_plan(w, words, shared);
        w->oracle.key = key;
    }
    relicfogOrder plan = w->oracle.plan[words[RW_CURSOR]];
    int fx = (int)(words[RW_FOCUS] & 15u) - RF_FOCUS_CENTER, fy = (int)(words[RW_FOCUS] >> 4 & 15u) - RF_FOCUS_CENTER;
    uint32_t pressed = words[RW_MODE] == RF_ORDER ? (uint32_t)plan.type : fx < plan.dx ? RF_RIGHT : fx > plan.dx ? RF_LEFT : fy < plan.dy ? RF_DOWN : fy > plan.dy ? RF_UP : RF_CENTER;
    ctrl[0] = (float)pressed;
}

uint32_t relicfog_view(uint32_t instance, const uint32_t *words, const uint32_t *extra, const float *shared, uint32_t *out) {
    (void)instance; (void)words;
    if (!out) return RF_VIEW_WORDS;
    const relicfogWorld *w = relicfog_read(extra);
    const relicfogParams *p = &w->map.params;
    const int8_t *field = relicfog_field(w, shared);
    uint16_t f[RF_VIEW_FIELDS];
    memset(f, 0, sizeof f);
    for (uint32_t t = 0u; t < RF_TILES; t++) {
        uint32_t x = t % RF_SIDE, y = t / RF_SIDE, kind = relicfog_bit(w->map.nebula, x, y) + 2u * relicfog_bit(w->map.asteroid, x, y);
        uint32_t cell = kind | (uint32_t)(field[t] + 32) << 2, weight = 0u;
        for (uint32_t q = 0u; q < RF_PAIRS; q++) weight |= relicfog_bit(w->map.weight[q], x, y) * (q + 1u);
        f[RF_VIEW_TILES + t / 2u] |= (uint16_t)(cell << (8u * (t & 1u)));
        f[RF_VIEW_WEIGHTS + t / 8u] |= (uint16_t)(weight << (2u * (t & 7u)));
        for (uint32_t team = 0u; team < 2u; team++) f[RF_VIEW_SIGHT + team * (RF_TILES / 16u) + t / 16u] |= (uint16_t)(relicfog_visible(w, team, x, y) << (t & 15u));
    }
    for (uint32_t t = 0u; t < 2u; t++)
        for (uint32_t i = 0u; i < RF_UNITS; i++) {
            uint32_t at = RF_VIEW_UNITS + 2u * (t * RF_UNITS + i);
            f[at] = (uint16_t)(w->ux[t][i] | w->uy[t][i] << 5 | (w->alive[t] >> i & 1u) << 10);
            f[at + 1u] = (uint16_t)w->energy[t][i];
            relicfogOrder o = w->last[t][i];
            f[RF_VIEW_LAST + t * RF_UNITS + i] = (uint16_t)((uint32_t)o.type | (uint32_t)(o.dx + RF_FOCUS_CENTER) << 3 | (uint32_t)(o.dy + RF_FOCUS_CENTER) << 7);
        }
    for (uint32_t i = 0u; i < RF_UNITS; i++) {
        relicfogOrder o = w->order[0][i];
        f[RF_VIEW_ORDERS + i] = (uint16_t)((uint32_t)o.type | (uint32_t)(o.dx + RF_FOCUS_CENTER) << 3 | (uint32_t)(o.dy + RF_FOCUS_CENTER) << 7);
    }
    for (uint32_t i = 0u; i < RF_RELICS; i++) f[RF_VIEW_RELICS + i] = (uint16_t)(w->map.relicX[i] | w->map.relicY[i] << 5 | (w->relicMask >> i & 1u) << 10);
    f[RF_VIEW_NODE] = (uint16_t)(w->map.nodeX | w->map.nodeY << 5);
    uint8_t params[11] = { p->moveCost, p->sensorRange, p->visionReduction, p->energyReduction, p->sapCost, p->sapRange, p->dropoff, p->voidFactor, (uint8_t)p->nebulaSpeed, p->nodeSpeed, p->nodeMagnitude };
    for (uint32_t i = 0u; i < 11u; i++) f[RF_VIEW_PARAMS + i] = params[i];
    for (uint32_t i = 0u; i < RF_VIEW_WORDS; i++) out[i] = f[2u * i] | (uint32_t)f[2u * i + 1u] << 16;
    return RF_VIEW_WORDS;
}
