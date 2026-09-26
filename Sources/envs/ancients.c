#include "../env.h"
#include <string.h>

typedef struct { uint32_t heroes, lanes, fog, draft, tier, jungle, league, substeps, waveTicks; float creepGrowth; uint32_t tickCap; } ancientsKnobs;

static const ancientsKnobs ANCIENTS_KNOBS[5] = {
    { 1u, 1u, 0u, 0u, 0u, 0u, 0u, 4u, 120u, 0.08f, 6000u },
    { 1u, 1u, 1u, 1u, 1u, 0u, 0u, 4u, 120u, 0.08f, 6000u },
    { 3u, 3u, 1u, 2u, 1u, 0u, 0u, 12u, 120u, 0.10f, 4800u },
    { 5u, 3u, 1u, 1u, 2u, 1u, 0u, 20u, 120u, 0.12f, 4800u },
    { 5u, 3u, 1u, 1u, 2u, 1u, 1u, 20u, 120u, 0.12f, 4800u }
};

static const int ANCIENTS_LANE_X0[6] = { 2048, 2048, 2048, 14336, 14336, 14336 };
static const int ANCIENTS_LANE_Y0[6] = { 14336, 14336, 14336, 2048, 2048, 2048 };
static const int ANCIENTS_LANE_AX1[6] = { 46341, 0, 65536, -46341, -65536, 0 };
static const int ANCIENTS_LANE_AY1[6] = { -46341, -65536, 0, 46341, 0, 65536 };
static const int ANCIENTS_LANE_CORNER[6] = { 17378, 12288, 12288, 17378, 12288, 12288 };
static const int ANCIENTS_LANE_AX2[6] = { 0, 65536, 0, 0, 0, -65536 };
static const int ANCIENTS_LANE_AY2[6] = { 0, 0, -65536, 0, 65536, 0 };
static const int ANCIENTS_LANE_LENGTH[3] = { 17378, 24576, 24576 };
static const int ANCIENTS_STRUCTURE_X[14] = { 6348, 4505, 2048, 2048, 10650, 6963, 2048, 10036, 11879, 5734, 9421, 14336, 14336, 14336 };
static const int ANCIENTS_STRUCTURE_Y[14] = { 10036, 11879, 5734, 9421, 14336, 14336, 14336, 6348, 4505, 2048, 2048, 10650, 6963, 2048 };
static const int ANCIENTS_STRUCTURE_ARC[14] = { 6082, 3476, 8602, 4915, 8602, 4915, 0, 6082, 3476, 8602, 4915, 8602, 4915, 0 };
static const float ANCIENTS_STRUCTURE_HP[7] = { 1600.0f, 2000.0f, 1600.0f, 2000.0f, 1600.0f, 2000.0f, 4000.0f };
static const int ANCIENTS_CAMP_X[4] = { 4608, 7680, 11776, 8704 };
static const int ANCIENTS_CAMP_Y[4] = { 7680, 4608, 8704, 11776 };
static const int ANCIENTS_FOUNTAIN_X[2] = { 1024, 15360 };
static const int ANCIENTS_FOUNTAIN_Y[2] = { 15360, 1024 };
static const int ANCIENTS_SEAT_LANE[5] = { 0, 1, 2, 1, 2 };

enum {
    ANC_EDGE = 16383, ANC_SIDE = 5, ANC_HEROES = 10, ANC_LANES = 3, ANC_FIFO = 16, ANC_OWN = 7, ANC_ANCIENT = 6, ANC_STRUCTURES = 14,
    ANC_CAMPS = 4, ANC_EVENTS = 8, ANC_MODES = 6, ANC_FEATURES = 11, ANC_REACTION = ANC_MODES * ANC_FEATURES, ANC_SKILL, ANC_THETA,
    ANC_GAP = 256, ANC_CREEP_STEP = 64, ANC_R1 = 65536, ANC_R15 = 147456, ANC_R4 = 1048576, ANC_R5 = 1638400, ANC_R6 = 2359296,
    ANC_R7 = 3211264, ANC_R8 = 4194304, ANC_R10 = 6553600, ANC_R12 = 9437184, ANC_NEAR15 = 150544, ANC_NEAR5 = 1648656, ANC_HORIZON = 16, ANC_PERIOD = 8
};
enum { ANC_LANE, ANC_PUSH, ANC_DEFEND, ANC_FARM, ANC_FIGHT, ANC_RETREAT };
enum { ANC_CARRY, ANC_NUKER, ANC_STUNNER, ANC_TANK, ANC_SUPPORT, ANC_BLINKER };
enum { ANC_ANY, ANC_ONEHIT, ANC_EXACT };
enum { ANC_HOLD, ANC_ENGAGE, ANC_ESCAPE, ANC_OPTIONS };
enum { ANC_NONE = 0, ANC_HERO = 1 << 20, ANC_CREEP = 2 << 20, ANC_TOWER = 3 << 20, ANC_CAMP = 4 << 20, ANC_KIND = 15 << 20 };
enum { EV_LASTHIT = 1, EV_KILL, EV_DEATH, EV_TOWER, EV_RESPAWN, EV_LEVEL, EV_ACQUIRED, EV_RELEASED, EV_WAVE, EV_ALLY };
enum { JOB_LANES = 20, JOB_LANE, JOB_TOWERS, JOB_TOWERS_B, JOB_COMMIT, JOB_VISION, JOB_UPKEEP, JOB_ACT, JOB_CAMPS, JOB_IDLE };

static const uint8_t ANC_JOBS[5][20] = {
    { 5, JOB_LANES, JOB_TOWERS, JOB_COMMIT },
    { 5, JOB_LANES, JOB_TOWERS, JOB_COMMIT },
    { 1, 2, 5, 6, 7, JOB_LANES, JOB_LANE, JOB_TOWERS, JOB_TOWERS_B, JOB_VISION, JOB_UPKEEP, JOB_ACT },
    { 1, 2, 3, 4, 5, 6, 7, 8, 9, JOB_LANES, JOB_LANE, JOB_TOWERS, JOB_TOWERS_B, JOB_CAMPS, JOB_VISION, JOB_IDLE, JOB_IDLE, JOB_IDLE, JOB_IDLE, JOB_COMMIT },
    { 1, 2, 3, 4, 5, 6, 7, 8, 9, JOB_LANES, JOB_LANE, JOB_TOWERS, JOB_TOWERS_B, JOB_CAMPS, JOB_VISION, JOB_IDLE, JOB_IDLE, JOB_IDLE, JOB_IDLE, JOB_COMMIT }
};

typedef struct { uint32_t ranged, hpBonus, item, cost[2], cooldown[2]; } ancArchetype;

static const ancArchetype ANC_ARCHETYPES[6] = {
    { 0u, 0u, 0u, { 75u, 50u }, { 240u, 120u } },
    { 1u, 0u, 2u, { 100u, 120u }, { 90u, 180u } },
    { 1u, 0u, 1u, { 100u, 80u }, { 180u, 200u } },
    { 0u, 300u, 1u, { 80u, 60u }, { 240u, 300u } },
    { 1u, 0u, 1u, { 110u, 90u }, { 150u, 240u } },
    { 0u, 0u, 0u, { 60u, 90u }, { 150u, 150u } }
};

static const float ANC_PRESETS[3][ANC_THETA] = {
    { 0.3f, 0.0f, 0.0f, 0.0f, 0.0f, -0.5f, 0.0f, 0.0f, 0.0f, 0.0f, 0.5f,
      1.0f, 0.0f, 0.5f, 0.5f, -2.0f, 0.0f, 0.0f, -0.5f, 0.0f, 0.0f, 0.4f,
      0.0f, 0.0f, 0.0f, 0.0f, 0.3f, 0.0f, -1.5f, 0.0f, 0.0f, 0.0f, 1.2f,
      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
      0.4f, 0.2f, 0.2f, 0.3f, 2.5f, 0.0f, 0.0f, 0.0f, 0.0f, 0.2f, -0.3f,
      -2.0f, 0.0f, 0.0f, 0.0f, 0.5f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.6f,
      12.0f, 0.0f },
    { 0.3f, 0.0f, 0.0f, 0.0f, -0.4f, -0.5f, 0.0f, 0.0f, 0.0f, 0.0f, 0.4f,
      1.0f, 0.0f, 0.5f, 1.0f, -1.0f, 0.0f, 0.0f, -0.5f, 0.0f, 0.0f, 0.3f,
      0.2f, 0.0f, 0.0f, 0.0f, 0.5f, 0.0f, -1.5f, 0.0f, 0.0f, 0.0f, 1.3f,
      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
      0.9f, 0.3f, 0.5f, 0.6f, 2.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.3f, -0.8f,
      -2.5f, -0.3f, -0.3f, -0.3f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.3f,
      6.0f, 0.5f },
    { 0.3f, 0.0f, 0.0f, 0.0f, -0.4f, -0.5f, 0.0f, 0.0f, 0.0f, 0.0f, 0.4f,
      1.0f, 0.0f, 0.5f, 1.0f, -1.0f, 0.0f, 0.0f, -0.5f, 0.0f, 0.0f, 0.3f,
      0.2f, 0.0f, 0.0f, 0.0f, 0.5f, 0.0f, -1.5f, 0.0f, 0.0f, 0.0f, 1.3f,
      0.3f, 0.0f, 0.0f, -0.4f, -1.0f, 0.0f, 0.0f, 0.0f, 0.8f, 0.0f, 0.0f,
      0.9f, 0.3f, 0.5f, 0.6f, 2.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.3f, -0.8f,
      -2.5f, -0.3f, -0.3f, -0.3f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.3f,
      2.0f, 1.0f }
};

static const float ANC_BOUNDS[2][ANC_THETA] = {
    { -3.0f, -3.0f, -3.0f, -3.0f, -3.0f, -3.0f, -3.0f, -3.0f, -3.0f, -3.0f, -3.0f,
      -3.0f, -3.0f, -3.0f, -3.0f, -3.0f, -3.0f, -3.0f, -3.0f, -3.0f, -3.0f, -3.0f,
      -3.0f, -3.0f, -3.0f, -3.0f, -3.0f, -3.0f, -3.0f, -3.0f, -3.0f, -3.0f, -3.0f,
      -3.0f, -3.0f, -3.0f, -3.0f, -3.0f, -3.0f, -3.0f, -3.0f, -3.0f, -3.0f, -3.0f,
      -3.0f, -3.0f, -3.0f, -3.0f, -3.0f, -3.0f, -3.0f, -3.0f, -3.0f, -3.0f, -3.0f,
      -3.0f, -3.0f, -3.0f, -3.0f, -3.0f, -3.0f, -3.0f, -3.0f, -3.0f, -3.0f, -3.0f,
      2.0f, 0.0f },
    { 3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f,
      3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f,
      3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f,
      3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f,
      3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f,
      3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f,
      12.0f, 1.0f }
};

typedef struct {
    int32_t x, y, goalX, goalY;
    float rx, ry, hp, mana, maxHp, maxMana, damage, xp, strike;
    uint32_t level, arch, gold, items, dead, respawnAt, attackReady, landAt, landTarget, ready[2];
    uint32_t frenzyUntil, stunUntil, slowUntil, tauntUntil, tauntBy, fortifyUntil, wardUntil, harassTick, harassVictim;
    uint32_t mode, target, lane, retarget;
} ancHero;

typedef struct {
    _Alignas(16) int32_t arc[ANC_FIFO];
    int32_t x[ANC_FIFO], y[ANC_FIFO];
    float hp[ANC_FIFO];
    uint32_t tag[ANC_FIFO], count, nextId, volleyAt;
    int32_t box[4];
} ancFifo;

typedef struct { float hp; uint32_t target, readyAt, aggroUntil, idleUntil; } ancTower;

typedef struct { int32_t x, y; float hp[3]; uint32_t aggro, readyAt, respawnAt; } ancCamp;

typedef struct {
    int32_t hx[ANC_SIDE], hy[ANC_SIDE];
    float hhp[ANC_SIDE];
    uint32_t hseen[ANC_SIDE], hvisible[ANC_SIDE];
    float shp[ANC_OWN];
    int32_t farc[ANC_LANES];
    float fhp[ANC_LANES];
    uint32_t camp[ANC_CAMPS];
} ancMemory;

typedef struct {
    ancFifo fifo[2 * ANC_LANES];
    ancHero hero[ANC_HEROES];
    ancTower tower[ANC_STRUCTURES];
    ancCamp camp[ANC_CAMPS];
    ancMemory memory[2];
    uint32_t event[ANC_EVENTS], eventHead, eventCount, eventShown, sweep, latch, latchClass[3], member, result;
    float latchX, latchY;
} ancWorld;

typedef struct { _Alignas(16) float winRate[LEAGUE_MEMBERS]; float theta[LEAGUE_MEMBERS * ANC_THETA]; } ancLeague;

typedef struct { int32_t x, y; uint32_t chase; float ctrl[OAK_ENV_ACT_MAX]; } ancPlan;

typedef struct { ancWorld world; ancPlan plan; uint32_t words[8], option, game, tick, primed, planGame, planOption; } ancOracle;

typedef struct {
    const ancientsKnobs *k;
    const uint8_t *jobs;
    ancWorld *w;
    const float *theta[2];
    uint32_t tick, seed, game;
} ancCtx;

static ancWorld *anc_world(uint32_t *extra) { return (ancWorld *)((uintptr_t)(extra + 4) & ~(uintptr_t)15u); }
static const ancWorld *anc_worldAt(const uint32_t *extra) { return (const ancWorld *)((uintptr_t)(extra + 4) & ~(uintptr_t)15u); }
static ancLeague *anc_league(uint32_t *extra) { return (ancLeague *)(anc_world(extra) + 1); }
static ancOracle *anc_oracleOf(uint32_t *extra) { return (ancOracle *)(anc_league(extra) + 1); }

static void anc_bind(ancCtx *c, const ancientsKnobs *k, uint32_t instance, ancWorld *w, const ancLeague *g, uint32_t seed, const uint32_t *words) {
    c->k = k;
    c->jobs = ANC_JOBS[instance];
    c->w = w;
    c->tick = words[0];
    c->seed = seed;
    c->game = words[2];
    c->theta[0] = ANC_PRESETS[k->tier];
    c->theta[1] = k->league ? g->theta + w->member * ANC_THETA : ANC_PRESETS[k->tier];
}

static int32_t anc_d2(int32_t ax, int32_t ay, int32_t bx, int32_t by) {
    int32_t dx = ax - bx, dy = ay - by;
    return dx * dx + dy * dy;
}

static int32_t anc_offset(int32_t s, int32_t a) { return (s * a + (a < 0 ? 65535 : 0)) >> 16; }

static void anc_point(uint32_t f, int32_t s, int32_t *x, int32_t *y) {
    int32_t corner = ANCIENTS_LANE_CORNER[f];
    s = s < 0 ? 0 : s;
    int32_t s1 = s < corner ? s : corner, s2 = s > corner ? s - corner : 0;
    *x = ANCIENTS_LANE_X0[f] + anc_offset(s1, ANCIENTS_LANE_AX1[f]) + anc_offset(s2, ANCIENTS_LANE_AX2[f]);
    *y = ANCIENTS_LANE_Y0[f] + anc_offset(s1, ANCIENTS_LANE_AY1[f]) + anc_offset(s2, ANCIENTS_LANE_AY2[f]);
}

static void anc_place(ancFifo *F, uint32_t f, int32_t head) {
    int4 corner = ANCIENTS_LANE_CORNER[f], ax1 = ANCIENTS_LANE_AX1[f], ay1 = ANCIENTS_LANE_AY1[f], ax2 = ANCIENTS_LANE_AX2[f], ay2 = ANCIENTS_LANE_AY2[f];
    int4 bx1 = ax1 < 0 ? 65535 : 0, by1 = ay1 < 0 ? 65535 : 0, bx2 = ax2 < 0 ? 65535 : 0, by2 = ay2 < 0 ? 65535 : 0, zero = 0;
    int4 x0 = ANCIENTS_LANE_X0[f], y0 = ANCIENTS_LANE_Y0[f], lo = 1 << 20, hi = -(1 << 20), count = (int32_t)F->count, slot = { 0, 1, 2, 3 };
    int4 lox = lo, loy = lo, hix = hi, hiy = hi, lead = head, step = ANC_CREEP_STEP, spacing = slot * ANC_GAP;
    for (uint32_t k = 0u; k < F->count; k += 4u) {
        int4 arc;
        memcpy(&arc, F->arc + k, sizeof arc);
        arc = __builtin_elementwise_min(arc + step, lead - spacing - (int32_t)(k * ANC_GAP));
        memcpy(F->arc + k, &arc, sizeof arc);
        int4 s = __builtin_elementwise_max(arc, zero), on = (arc >= 0) & ((int32_t)k + slot < count);
        int4 s1 = __builtin_elementwise_min(s, corner), s2 = __builtin_elementwise_max(s - corner, zero);
        int4 x = x0 + ((s1 * ax1 + bx1) >> 16) + ((s2 * ax2 + bx2) >> 16);
        int4 y = y0 + ((s1 * ay1 + by1) >> 16) + ((s2 * ay2 + by2) >> 16);
        memcpy(F->x + k, &x, sizeof x);
        memcpy(F->y + k, &y, sizeof y);
        lox = __builtin_elementwise_min(lox, (on & x) | (~on & lo));
        loy = __builtin_elementwise_min(loy, (on & y) | (~on & lo));
        hix = __builtin_elementwise_max(hix, (on & x) | (~on & hi));
        hiy = __builtin_elementwise_max(hiy, (on & y) | (~on & hi));
    }
    F->box[0] = __builtin_reduce_min(lox);
    F->box[1] = __builtin_reduce_min(loy);
    F->box[2] = __builtin_reduce_max(hix);
    F->box[3] = __builtin_reduce_max(hiy);
}

static int anc_boxed(const ancFifo *F, int32_t x, int32_t y, int32_t r2) {
    int32_t dx = F->box[0] - x > 0 ? F->box[0] - x : x - F->box[2] > 0 ? x - F->box[2] : 0;
    int32_t dy = F->box[1] - y > 0 ? F->box[1] - y : y - F->box[3] > 0 ? y - F->box[3] : 0;
    return dx < 4096 && dy < 4096 && dx * dx + dy * dy <= r2;
}

static int32_t anc_reach(const ancHero *e) { return ANC_ARCHETYPES[e->arch].ranged ? ANC_R5 : ANC_R15; }
static uint32_t anc_lanes(const ancientsKnobs *k) { return (1u << k->lanes) - 1u; }
static float anc_scale(uint32_t level) { return 1.0f + 0.1f * (float)((level - 1u) >> 2); }
static uint32_t anc_ticks(uint32_t base, float scale) { return (uint32_t)((float)base * scale + 0.5f); }
static float anc_waveHp(const ancientsKnobs *k, uint32_t wave) { return 300.0f * (1.0f + k->creepGrowth * (float)wave); }
static uint32_t anc_reaction(const float *theta) { return (uint32_t)(theta[ANC_REACTION] + 0.5f); }
static int anc_onMap(const ancFifo *F) { return F->count > 0u && F->arc[0] >= 0; }
static uint32_t anc_creepRef(uint32_t f, const ancFifo *F, uint32_t k) { return (uint32_t)ANC_CREEP | f << 16 | (F->tag[k] & 0xFFFFu); }

static int anc_exists(const ancientsKnobs *k, uint32_t s) {
    uint32_t i = s % ANC_OWN;
    return i == ANC_ANCIENT || i / 2u < k->lanes;
}

static int anc_vulnerable(const ancCtx *c, uint32_t s) {
    const ancTower *t = c->w->tower;
    uint32_t i = s % ANC_OWN, base = s - i;
    if (!(t[s].hp > 0.0f)) return 0;
    if (i == ANC_ANCIENT) {
        for (uint32_t l = 0u; l < c->k->lanes; l++)
            if (!(t[base + 2u * l + 1u].hp > 0.0f)) return 1;
        return 0;
    }
    return (i & 1u) == 0u || !(t[s - 1u].hp > 0.0f);
}

static uint32_t anc_front(const ancWorld *w, uint32_t team, uint32_t lane) {
    uint32_t base = team * ANC_OWN;
    if (w->tower[base + 2u * lane].hp > 0.0f) return base + 2u * lane;
    if (w->tower[base + 2u * lane + 1u].hp > 0.0f) return base + 2u * lane + 1u;
    return base + ANC_ANCIENT;
}

static float anc_volley(const ancientsKnobs *k, const ancFifo *F) {
    float4 sum = 0.0f;
    int4 slot = { 0, 1, 2, 3 }, count = (int32_t)F->count, lead = F->arc[0];
    for (uint32_t i = 0u; i < F->count; i += 4u) {
        int4 arc, index = (int32_t)i + slot;
        lane4 tag;
        memcpy(&arc, F->arc + i, sizeof arc);
        memcpy(&tag, F->tag + i, sizeof tag);
        float4 damage = 20.0f * (1.0f + k->creepGrowth * __builtin_convertvector(tag >> 16u, float4));
        sum += select4((lane4)((arc == lead - index * ANC_GAP) & (arc >= 0) & (index < count)), damage, splat4(0.0f));
    }
    return (sum.x + sum.y) + (sum.z + sum.w);
}

static float anc_total(const ancFifo *F) {
    float4 sum = 0.0f;
    int4 slot = { 0, 1, 2, 3 }, count = (int32_t)F->count;
    for (uint32_t k = 0u; k < F->count; k += 4u) {
        int4 arc;
        float4 hp;
        memcpy(&arc, F->arc + k, sizeof arc);
        memcpy(&hp, F->hp + k, sizeof hp);
        sum += select4((lane4)((arc >= 0) & ((int32_t)k + slot < count)), hp, splat4(0.0f));
    }
    return (sum.x + sum.y) + (sum.z + sum.w);
}

static int anc_find(const ancFifo *F, uint32_t id) {
    for (uint32_t k = 0u; k < F->count; k++)
        if ((F->tag[k] & 0xFFFFu) == id) return (int)k;
    return -1;
}

static uint32_t anc_alive(const ancCamp *m) { return (m->hp[0] > 0.0f) + (m->hp[1] > 0.0f) + (m->hp[2] > 0.0f); }

static void anc_event(ancWorld *w, uint32_t code) {
    if (w->eventCount < ANC_EVENTS) {
        w->event[(w->eventHead + w->eventCount) & (ANC_EVENTS - 1u)] = code;
        w->eventCount++;
    }
}

static void anc_xp(ancCtx *c, uint32_t h, float xp) {
    ancHero *e = &c->w->hero[h];
    e->xp += xp;
    while (e->level < 16u && e->xp >= 100.0f * (float)e->level) {
        e->xp -= 100.0f * (float)e->level;
        e->level++;
        e->maxHp += 60.0f; e->hp += 60.0f; e->maxMana += 20.0f; e->mana += 20.0f; e->damage += 4.0f;
        if (h == 0u) anc_event(c->w, EV_LEVEL);
    }
}

static void anc_gold(ancCtx *c, uint32_t h, uint32_t gold) {
    ancHero *e = &c->w->hero[h];
    e->gold += gold;
    while (e->gold >= 450u * (e->items + 1u)) {
        e->items++;
        switch (ANC_ARCHETYPES[e->arch].item) {
            case 0u: e->damage += 8.0f; break;
            case 1u: e->maxHp += 80.0f; if (!e->dead) e->hp += 80.0f; break;
            default: e->maxMana += 40.0f; e->mana += 40.0f; e->damage += 4.0f; break;
        }
    }
}

static void anc_heroDies(ancCtx *c, uint32_t v, uint32_t by) {
    ancWorld *w = c->w;
    ancHero *e = &w->hero[v];
    uint32_t team = v >= ANC_SIDE, foe = 1u - team, level = e->level, near[ANC_SIDE], n = 0u;
    int32_t x = e->x, y = e->y;
    e->dead = 1u; e->hp = 0.0f; e->landAt = 0u; e->strike = 0.0f; e->target = ANC_NONE;
    e->respawnAt = c->tick + 60u + 12u * level;
    e->x = ANCIENTS_FOUNTAIN_X[team]; e->y = ANCIENTS_FOUNTAIN_Y[team]; e->rx = 0.0f; e->ry = 0.0f;
    if ((by & ANC_KIND) == ANC_HERO) {
        uint32_t a = by & 0xFFu;
        anc_gold(c, a, 150u + 20u * level);
        if (a == 0u) anc_event(w, EV_KILL);
    }
    for (uint32_t i = 0u; i < c->k->heroes; i++) {
        uint32_t j = foe * ANC_SIDE + i;
        if (!w->hero[j].dead && anc_d2(w->hero[j].x, w->hero[j].y, x, y) <= ANC_R12) near[n++] = j;
    }
    for (uint32_t i = 0u; i < n; i++) anc_xp(c, near[i], (100.0f + 30.0f * (float)level) / (float)n);
    if (v == 0u) anc_event(w, EV_DEATH);
    else if (team == 0u) anc_event(w, EV_ALLY);
}

static void anc_harass(ancCtx *c, uint32_t a, uint32_t v) {
    ancWorld *w = c->w;
    ancHero *e = &w->hero[a];
    uint32_t base = (v >= ANC_SIDE) * ANC_OWN;
    e->harassTick = c->tick + 1u;
    e->harassVictim = v;
    for (uint32_t i = 0u; i < 2u * c->k->lanes; i++) {
        uint32_t s = base + i;
        ancTower *t = &w->tower[s];
        if (!(t->hp > 0.0f) || anc_d2(ANCIENTS_STRUCTURE_X[s], ANCIENTS_STRUCTURE_Y[s], e->x, e->y) > ANC_R7) continue;
        if (a == 0u && t->target != (uint32_t)ANC_HERO) anc_event(w, EV_ACQUIRED);
        t->target = (uint32_t)ANC_HERO | a;
        t->aggroUntil = c->tick + 40u;
    }
}

static void anc_hitHero(ancCtx *c, uint32_t v, float amount, uint32_t by) {
    ancHero *e = &c->w->hero[v];
    if (e->dead) return;
    if (c->tick < e->fortifyUntil) amount *= 0.5f;
    if (c->tick < e->wardUntil) amount *= 0.4f;
    e->hp -= amount;
    if ((by & ANC_KIND) == ANC_HERO) anc_harass(c, by & 0xFFu, v);
    if (!(e->hp > 0.0f)) anc_heroDies(c, v, by);
}

static void anc_creepDies(ancCtx *c, uint32_t f, uint32_t k, uint32_t by) {
    ancWorld *w = c->w;
    ancFifo *F = &w->fifo[f];
    uint32_t wave = F->tag[k] >> 16, foe = f < ANC_LANES;
    int32_t x = F->x[k], y = F->y[k];
    for (uint32_t i = k; i + 1u < F->count; i++) {
        F->arc[i] = F->arc[i + 1u]; F->x[i] = F->x[i + 1u]; F->y[i] = F->y[i + 1u]; F->hp[i] = F->hp[i + 1u]; F->tag[i] = F->tag[i + 1u];
    }
    F->count--;
    if ((by & ANC_KIND) == ANC_HERO) {
        uint32_t a = by & 0xFFu;
        anc_gold(c, a, 40u + 2u * wave);
        if (a == 0u) anc_event(w, EV_LASTHIT);
    }
    for (uint32_t i = 0u; i < c->k->heroes; i++) {
        uint32_t j = (foe ? ANC_SIDE : 0u) + i;
        if (!w->hero[j].dead && anc_d2(w->hero[j].x, w->hero[j].y, x, y) <= ANC_R12) anc_xp(c, j, 40.0f);
    }
}

static void anc_hitCreep(ancCtx *c, uint32_t f, uint32_t k, float amount, uint32_t by) {
    ancFifo *F = &c->w->fifo[f];
    F->hp[k] -= amount;
    if (!(F->hp[k] > 0.0f)) anc_creepDies(c, f, k, by);
}

static void anc_hitStructure(ancCtx *c, uint32_t s, float amount) {
    ancWorld *w = c->w;
    if (!anc_vulnerable(c, s)) return;
    w->tower[s].hp -= amount;
    if (w->tower[s].hp > 0.0f) return;
    w->tower[s].hp = 0.0f;
    uint32_t team = s >= ANC_OWN;
    if (s % ANC_OWN == ANC_ANCIENT) {
        w->result = team ? 1u : 2u;
        return;
    }
    for (uint32_t i = 0u; i < c->k->heroes; i++) anc_gold(c, (1u - team) * ANC_SIDE + i, 150u);
    anc_event(w, EV_TOWER);
}

static void anc_hitCamp(ancCtx *c, uint32_t i, float amount, uint32_t by, uint32_t all) {
    ancWorld *w = c->w;
    ancCamp *m = &w->camp[i];
    uint32_t hero = (by & ANC_KIND) == ANC_HERO, a = by & 0xFFu;
    if (hero && anc_d2(w->hero[a].x, w->hero[a].y, ANCIENTS_CAMP_X[i], ANCIENTS_CAMP_Y[i]) <= ANC_R6) m->aggro = by;
    for (uint32_t j = 0u; j < 3u; j++) {
        if (!(m->hp[j] > 0.0f)) continue;
        m->hp[j] -= amount;
        if (!(m->hp[j] > 0.0f)) {
            m->hp[j] = 0.0f;
            if (hero) {
                anc_gold(c, a, 50u);
                if (!w->hero[a].dead) anc_xp(c, a, 60.0f);
                if (a == 0u) anc_event(w, EV_LASTHIT);
            }
        }
        if (!all) break;
    }
    if (anc_alive(m) == 0u) {
        m->respawnAt = c->tick + 600u;
        m->aggro = ANC_NONE;
    }
}

static int anc_locate(const ancWorld *w, uint32_t target, int32_t *x, int32_t *y) {
    uint32_t index = target & 0xFFu;
    switch (target & ANC_KIND) {
        case ANC_HERO:
            if (w->hero[index].dead) return 0;
            *x = w->hero[index].x; *y = w->hero[index].y;
            return 1;
        case ANC_CREEP: {
            const ancFifo *F = &w->fifo[(target >> 16) & 15u];
            int k = anc_find(F, target & 0xFFFFu);
            if (k < 0) return 0;
            *x = F->x[k]; *y = F->y[k];
            return 1;
        }
        case ANC_TOWER:
            if (!(w->tower[index].hp > 0.0f)) return 0;
            *x = ANCIENTS_STRUCTURE_X[index]; *y = ANCIENTS_STRUCTURE_Y[index];
            return 1;
        case ANC_CAMP:
            if (anc_alive(&w->camp[index & 3u]) == 0u) return 0;
            *x = w->camp[index & 3u].x; *y = w->camp[index & 3u].y;
            return 1;
        default: return 0;
    }
}

static int anc_apply(ancCtx *c, uint32_t target, float amount, uint32_t by, uint32_t stun) {
    ancWorld *w = c->w;
    uint32_t index = target & 0xFFu;
    switch (target & ANC_KIND) {
        case ANC_HERO: {
            ancHero *e = &w->hero[index];
            if (e->dead) return 0;
            if (stun) {
                if (e->stunUntil < c->tick + stun) e->stunUntil = c->tick + stun;
                e->landAt = 0u;
            }
            anc_hitHero(c, index, amount, by);
            return 1;
        }
        case ANC_CREEP: {
            uint32_t f = (target >> 16) & 15u;
            int k = anc_find(&w->fifo[f], target & 0xFFFFu);
            if (k < 0) return 0;
            anc_hitCreep(c, f, (uint32_t)k, amount, by);
            return 1;
        }
        case ANC_TOWER:
            if (!(w->tower[index].hp > 0.0f)) return 0;
            anc_hitStructure(c, index, amount);
            return 1;
        case ANC_CAMP:
            if (anc_alive(&w->camp[index & 3u]) == 0u) return 0;
            anc_hitCamp(c, index & 3u, amount, by, 0u);
            return 1;
        default: return 0;
    }
}

static inline __attribute__((always_inline)) uint32_t anc_lowest(const ancWorld *w, uint32_t lanes, uint32_t team, int32_t x, int32_t y, int32_t r2, float *low) {
    int4 px = x, py = y, reach = r2, slot = { 0, 1, 2, 3 }, at = -1;
    float4 m = INFINITY;
    for (uint32_t l = 0u; l < ANC_LANES; l++) {
        uint32_t f = (1u - team) * ANC_LANES + l;
        const ancFifo *F = &w->fifo[f];
        if (!(lanes >> l & 1u) || !anc_boxed(F, x, y, r2)) continue;
        int4 count = (int32_t)F->count;
        for (uint32_t k = 0u; k < F->count; k += 4u) {
            int4 cx, cy, arc, index = (int32_t)k + slot;
            float4 hp;
            memcpy(&cx, F->x + k, sizeof cx);
            memcpy(&cy, F->y + k, sizeof cy);
            memcpy(&arc, F->arc + k, sizeof arc);
            memcpy(&hp, F->hp + k, sizeof hp);
            int4 dx = cx - px, dy = cy - py, take = (dx * dx + dy * dy <= reach) & (arc >= 0) & (index < count) & (hp < m);
            m = select4((lane4)take, hp, m);
            at = (take & (index + (int32_t)(f * ANC_FIFO))) | (~take & at);
        }
    }
    float v = m.x;
    int32_t i = at.x;
    for (uint32_t j = 1u; j < 4u; j++)
        if (m[j] < v || (m[j] == v && at[j] < i)) { v = m[j]; i = at[j]; }
    *low = v;
    return i < 0 ? (uint32_t)ANC_NONE : anc_creepRef((uint32_t)i / ANC_FIFO, &w->fifo[(uint32_t)i / ANC_FIFO], (uint32_t)i % ANC_FIFO);
}

static int anc_danger(const ancCtx *c, uint32_t team, int32_t x, int32_t y) {
    const ancWorld *w = c->w;
    uint32_t base = (1u - team) * ANC_OWN, hot = 0u;
    for (uint32_t i = 0u; i < 2u * c->k->lanes; i++)
        hot |= (uint32_t)(w->tower[base + i].hp > 0.0f && anc_d2(ANCIENTS_STRUCTURE_X[base + i], ANCIENTS_STRUCTURE_Y[base + i], x, y) <= ANC_R7) << i;
    for (; hot; hot &= hot - 1u) {
        uint32_t s = base + (uint32_t)__builtin_ctz(hot), covered = 0u;
        for (uint32_t l = 0u; l < c->k->lanes; l++) {
            const ancFifo *F = &w->fifo[team * ANC_LANES + l];
            covered |= anc_onMap(F) && anc_d2(F->x[0], F->y[0], ANCIENTS_STRUCTURE_X[s], ANCIENTS_STRUCTURE_Y[s]) <= ANC_R7;
        }
        if (!covered) return 1;
    }
    return 0;
}

static uint32_t anc_lastHit(const ancCtx *c, uint32_t team, int32_t x, int32_t y, int32_t r2, uint32_t rule, float damage) {
    const ancWorld *w = c->w;
    uint32_t foe = 1u - team, best = ANC_NONE, pending[2 * ANC_SIDE + 2 * ANC_LANES], count = 0u, until = c->tick + 3u, near = 0u;
    float hits[2 * ANC_SIDE + 2 * ANC_LANES], bound = 0.0f, bestHp = INFINITY;
    for (uint32_t l = 0u; l < c->k->lanes; l++) near |= (uint32_t)anc_boxed(&w->fifo[foe * ANC_LANES + l], x, y, r2) << l;
    if (!near || anc_danger(c, team, x, y)) return ANC_NONE;
    if (rule == ANC_EXACT) {
        for (uint32_t i = 0u; i < 2u * ANC_SIDE; i++) {
            const ancHero *e = &w->hero[i];
            if (e->landAt && e->landAt <= until && (e->landTarget & ANC_KIND) == (uint32_t)ANC_CREEP) {
                pending[count] = e->landTarget;
                hits[count] = e->strike > 0.0f ? e->damage * e->strike : e->damage;
                bound += hits[count++];
            }
        }
        for (uint32_t i = 0u; i < 2u * c->k->lanes; i++) {
            const ancTower *t = &w->tower[team * ANC_OWN + i];
            if (t->hp > 0.0f && (t->target & ANC_KIND) == (uint32_t)ANC_CREEP && t->readyAt <= until) {
                pending[count] = t->target;
                hits[count] = 100.0f;
                bound += hits[count++];
            }
        }
    }
    int4 px = x, py = y, reach = r2, slot = { 0, 1, 2, 3 }, at = -1;
    float4 low = INFINITY;
    for (uint32_t l = 0u; l < c->k->lanes; l++) {
        uint32_t f = foe * ANC_LANES + l;
        const ancFifo *F = &w->fifo[f];
        if (!(near >> l & 1u)) continue;
        const ancFifo *O = &w->fifo[team * ANC_LANES + l];
        float duel = rule != ANC_ANY && anc_onMap(O) && anc_onMap(F) && (ANCIENTS_LANE_LENGTH[l] - O->arc[0]) - F->arc[0] <= ANC_GAP && O->volleyAt <= until ? anc_volley(c->k, O) : 0.0f;
        int4 count4 = (int32_t)F->count;
        float4 cap = damage + bound;
        for (uint32_t k = 0u; k < F->count; k += 4u) {
            int4 cx, cy, arc, index = (int32_t)k + slot;
            float4 hp4;
            memcpy(&cx, F->x + k, sizeof cx);
            memcpy(&cy, F->y + k, sizeof cy);
            memcpy(&arc, F->arc + k, sizeof arc);
            memcpy(&hp4, F->hp + k, sizeof hp4);
            int4 dx = cx - px, dy = cy - py, in = (dx * dx + dy * dy <= reach) & (arc >= 0) & (index < count4), take = in & (hp4 < low);
            low = select4((lane4)take, hp4, low);
            at = (take & (index + (int32_t)(f * ANC_FIFO))) | (~take & at);
            if (rule == ANC_ANY) continue;
            float4 limit = k ? cap : cap + (float4){ duel, 0.0f, 0.0f, 0.0f };
            int4 keep = in & (hp4 <= limit);
            uint32_t bits = (uint32_t)(keep.x & 1) | (uint32_t)(keep.y & 2) | (uint32_t)(keep.z & 4) | (uint32_t)(keep.w & 8);
            for (; bits; bits &= bits - 1u) {
                uint32_t j = k + (uint32_t)__builtin_ctz(bits);
                float hp = F->hp[j];
                if (rule == ANC_EXACT) {
                    uint32_t ref = anc_creepRef(f, F, j);
                    if (j == 0u) hp -= duel;
                    for (uint32_t i = 0u; i < count; i++) hp -= pending[i] == ref ? hits[i] : 0.0f;
                    if (!(hp > 0.0f)) continue;
                }
                if (hp <= damage && hp < bestHp) {
                    bestHp = hp;
                    best = anc_creepRef(f, F, j);
                }
            }
        }
    }
    if (best) return best;
    float v = low.x;
    int32_t i = at.x;
    for (uint32_t j = 1u; j < 4u; j++)
        if (low[j] < v || (low[j] == v && at[j] < i)) { v = low[j]; i = at[j]; }
    return i < 0 ? (uint32_t)ANC_NONE : anc_creepRef((uint32_t)i / ANC_FIFO, &w->fifo[(uint32_t)i / ANC_FIFO], (uint32_t)i % ANC_FIFO);
}

static uint32_t anc_nearestHero(const ancCtx *c, uint32_t team, int32_t x, int32_t y, int32_t r2, uint32_t weakest) {
    const ancWorld *w = c->w;
    uint32_t best = ANC_NONE;
    int32_t bestD = r2;
    float bestHp = INFINITY;
    for (uint32_t i = 0u; i < c->k->heroes; i++) {
        uint32_t j = (1u - team) * ANC_SIDE + i;
        const ancHero *e = &w->hero[j];
        if (e->dead) continue;
        int32_t d = anc_d2(e->x, e->y, x, y);
        if (d > r2) continue;
        if (weakest ? e->hp < bestHp : d <= bestD) {
            bestHp = e->hp;
            bestD = d;
            best = (uint32_t)ANC_HERO | j;
        }
    }
    return best;
}

static uint32_t anc_objective(const ancCtx *c, uint32_t team, int32_t x, int32_t y, int32_t r2, uint32_t spell) {
    const ancWorld *w = c->w;
    uint32_t best = ANC_NONE;
    int32_t bestD = r2;
    if (!spell)
        for (uint32_t i = 0u; i < ANC_OWN; i++) {
            uint32_t s = (1u - team) * ANC_OWN + i;
            int32_t d = anc_d2(ANCIENTS_STRUCTURE_X[s], ANCIENTS_STRUCTURE_Y[s], x, y);
            if (d <= bestD && anc_vulnerable(c, s)) {
                bestD = d;
                best = (uint32_t)ANC_TOWER | s;
            }
        }
    if (c->k->jungle)
        for (uint32_t i = 0u; i < ANC_CAMPS; i++) {
            const ancCamp *m = &w->camp[i];
            int32_t d = anc_d2(m->x, m->y, x, y);
            if (d <= bestD && anc_alive(m)) {
                bestD = d;
                best = (uint32_t)ANC_CAMP | i;
            }
        }
    return best;
}

static uint32_t anc_pick(ancCtx *c, uint32_t h, uint32_t cls, int32_t r2, uint32_t spell) {
    const ancWorld *w = c->w;
    const ancHero *e = &w->hero[h];
    uint32_t team = h >= ANC_SIDE;
    float low;
    if (!spell && c->tick < e->tauntUntil && !w->hero[e->tauntBy].dead)
        return anc_d2(w->hero[e->tauntBy].x, w->hero[e->tauntBy].y, e->x, e->y) <= r2 ? (uint32_t)ANC_HERO | e->tauntBy : ANC_NONE;
    if (cls == 0u) return anc_lowest(w, anc_lanes(c->k), team, e->x, e->y, r2, &low);
    if (cls == 1u) return anc_nearestHero(c, team, e->x, e->y, r2, 0u);
    return anc_objective(c, team, e->x, e->y, r2, spell);
}

static uint32_t anc_ally(const ancCtx *c, uint32_t h) {
    const ancWorld *w = c->w;
    const ancHero *e = &w->hero[h];
    uint32_t base = (h >= ANC_SIDE) * ANC_SIDE, best = h;
    float low = INFINITY;
    for (uint32_t i = 0u; i < c->k->heroes; i++) {
        const ancHero *a = &w->hero[base + i];
        if (base + i == h || a->dead || anc_d2(a->x, a->y, e->x, e->y) > ANC_R6) continue;
        if (a->hp / a->maxHp < low) {
            low = a->hp / a->maxHp;
            best = base + i;
        }
    }
    return best;
}

static inline __attribute__((always_inline)) void anc_shift(int32_t *p, float *r, float delta) {
    float t = *r + delta, n = floorf(t + 0.5f);
    int32_t q = *p + (int32_t)n;
    t -= n;
    if (q < 0) { q = 0; t = 0.0f; }
    if (q > ANC_EDGE) { q = ANC_EDGE; t = 0.0f; }
    *p = q;
    *r = t;
}

static void anc_walk(const ancCtx *c, ancHero *e, int32_t gx, int32_t gy) {
    float speed = c->tick < e->slowUntil ? 38.4f : 76.8f;
    float dx = (float)(gx - e->x) - e->rx, dy = (float)(gy - e->y) - e->ry, d2 = dx * dx + dy * dy;
    if (d2 <= speed * speed) {
        e->x = gx; e->y = gy; e->rx = 0.0f; e->ry = 0.0f;
        return;
    }
    float s = speed / sqrtf(d2);
    anc_shift(&e->x, &e->rx, dx * s);
    anc_shift(&e->y, &e->ry, dy * s);
}

static uint32_t anc_enemiesNear(const ancCtx *c, uint32_t h, int32_t r2) {
    const ancWorld *w = c->w;
    uint32_t n = 0u, base = (h < ANC_SIDE) * ANC_SIDE;
    for (uint32_t i = 0u; i < c->k->heroes; i++) {
        const ancHero *e = &w->hero[base + i];
        n += !e->dead && anc_d2(e->x, e->y, w->hero[h].x, w->hero[h].y) <= r2;
    }
    return n;
}

static int anc_cast(ancCtx *c, uint32_t h, uint32_t slot, uint32_t target, float vx, float vy) {
    ancWorld *w = c->w;
    ancHero *e = &w->hero[h];
    const ancArchetype *a = &ANC_ARCHETYPES[e->arch];
    if (c->tick < e->ready[slot] || e->mana < (float)a->cost[slot]) return 0;
    float m = anc_scale(e->level), level = (float)e->level;
    uint32_t foe = (h < ANC_SIDE) * ANC_SIDE;
    switch (e->arch * 2u + slot) {
        case ANC_CARRY * 2u: e->frenzyUntil = c->tick + anc_ticks(60u, m); break;
        case ANC_CARRY * 2u + 1u: e->strike = 2.5f * m; break;
        case ANC_NUKER * 2u:
            if (!anc_apply(c, target, (200.0f + 30.0f * level) * m, (uint32_t)ANC_HERO | h, 0u)) return 0;
            break;
        case ANC_NUKER * 2u + 1u: {
            float amount = (120.0f + 15.0f * level) * m;
            for (uint32_t i = 0u; i < c->k->heroes; i++) {
                const ancHero *v = &w->hero[foe + i];
                if (!v->dead && anc_d2(v->x, v->y, e->x, e->y) <= ANC_R4) anc_hitHero(c, foe + i, amount, (uint32_t)ANC_HERO | h);
            }
            for (uint32_t l = 0u; l < c->k->lanes; l++) {
                uint32_t f = (foe ? ANC_LANES : 0u) + l;
                ancFifo *F = &w->fifo[f];
                for (uint32_t k = F->count; k-- > 0u;)
                    if (F->arc[k] >= 0 && anc_d2(F->x[k], F->y[k], e->x, e->y) <= ANC_R4) anc_hitCreep(c, f, k, amount, (uint32_t)ANC_HERO | h);
            }
            if (c->k->jungle)
                for (uint32_t i = 0u; i < ANC_CAMPS; i++)
                    if (anc_alive(&w->camp[i]) && anc_d2(w->camp[i].x, w->camp[i].y, e->x, e->y) <= ANC_R4) anc_hitCamp(c, i, amount, (uint32_t)ANC_HERO | h, 1u);
            break;
        }
        case ANC_STUNNER * 2u:
            if (!anc_apply(c, target, 80.0f * m, (uint32_t)ANC_HERO | h, anc_ticks(60u, m))) return 0;
            break;
        case ANC_STUNNER * 2u + 1u:
            for (uint32_t i = 0u; i < c->k->heroes; i++) {
                ancHero *v = &w->hero[foe + i];
                if (!v->dead && anc_d2(v->x, v->y, e->x, e->y) <= ANC_R5) v->slowUntil = c->tick + anc_ticks(60u, m);
            }
            break;
        case ANC_TANK * 2u:
            for (uint32_t i = 0u; i < c->k->heroes; i++) {
                ancHero *v = &w->hero[foe + i];
                if (!v->dead && anc_d2(v->x, v->y, e->x, e->y) <= ANC_R4) {
                    v->tauntUntil = c->tick + anc_ticks(40u, m);
                    v->tauntBy = h;
                }
            }
            break;
        case ANC_TANK * 2u + 1u: e->fortifyUntil = c->tick + anc_ticks(80u, m); break;
        case ANC_SUPPORT * 2u: {
            ancHero *t = &w->hero[anc_ally(c, h)];
            t->hp = fminf(t->hp + 250.0f * m, t->maxHp);
            break;
        }
        case ANC_SUPPORT * 2u + 1u: w->hero[anc_ally(c, h)].wardUntil = c->tick + anc_ticks(50u, m); break;
        case ANC_BLINKER * 2u:
            anc_shift(&e->x, &e->rx, 2048.0f * m * vx);
            anc_shift(&e->y, &e->ry, 2048.0f * m * vy);
            for (uint32_t i = 0u; i < ANC_OWN; i++) w->tower[(h < ANC_SIDE) * ANC_OWN + i].idleUntil = 0u;
            break;
        default:
            if (!anc_apply(c, target, 160.0f * m, (uint32_t)ANC_HERO | h, anc_ticks(20u, m))) return 0;
            break;
    }
    e->mana -= (float)a->cost[slot];
    e->ready[slot] = c->tick + a->cooldown[slot];
    return 1;
}

static void anc_startAttack(ancCtx *c, uint32_t h, uint32_t target) {
    ancHero *e = &c->w->hero[h];
    e->landAt = c->tick + 3u;
    e->landTarget = target;
    e->attackReady = c->tick + (c->tick < e->frenzyUntil ? 6u : 12u);
}

static void anc_land(ancCtx *c, uint32_t h) {
    ancHero *e = &c->w->hero[h];
    float amount = e->strike > 0.0f ? e->damage * e->strike : e->damage;
    e->landAt = 0u;
    if (anc_apply(c, e->landTarget, amount, (uint32_t)ANC_HERO | h, 0u)) e->strike = 0.0f;
}

static void anc_respawn(ancCtx *c, uint32_t h) {
    ancWorld *w = c->w;
    ancHero *e = &w->hero[h];
    uint32_t team = h >= ANC_SIDE, i = h % ANC_SIDE;
    e->dead = 0u; e->hp = e->maxHp; e->mana = e->maxMana; e->retarget = 1u; e->target = ANC_NONE; e->landAt = 0u;
    if (h == 0u) anc_event(w, EV_RESPAWN);
    ancMemory *m = &w->memory[1u - team];
    m->hx[i] = e->x; m->hy[i] = e->y; m->hhp[i] = 1.0f; m->hseen[i] = c->tick + 1u; m->hvisible[i] = 0u;
}

static int anc_upkeep(ancCtx *c, uint32_t h) {
    ancHero *e = &c->w->hero[h];
    uint32_t team = h >= ANC_SIDE;
    if ((c->tick & 3u) == 0u) anc_gold(c, h, 1u);
    if (e->dead) {
        if (c->tick < e->respawnAt) return 0;
        anc_respawn(c, h);
    }
    e->hp = fminf(e->hp + 0.05f, e->maxHp);
    e->mana = fminf(e->mana + 0.1f, e->maxMana);
    if (anc_d2(e->x, e->y, ANCIENTS_FOUNTAIN_X[team], ANCIENTS_FOUNTAIN_Y[team]) <= ANC_R4) e->hp = fminf(e->hp + 0.04f * e->maxHp, e->maxHp);
    if (anc_d2(e->x, e->y, ANCIENTS_FOUNTAIN_X[1u - team], ANCIENTS_FOUNTAIN_Y[1u - team]) <= ANC_R4) {
        anc_hitHero(c, h, 300.0f, ANC_NONE);
        if (e->dead) return 0;
    }
    if (c->tick < e->stunUntil) return 0;
    if (e->landAt && c->tick >= e->landAt) anc_land(c, h);
    return 1;
}

static inline __attribute__((always_inline)) float anc_clashArc(const ancientsKnobs *k, const ancWorld *w, uint32_t team, uint32_t lane) {
    const ancFifo *F = &w->fifo[team * ANC_LANES + lane], *O = &w->fifo[(1u - team) * ANC_LANES + lane];
    int32_t own = anc_onMap(F) ? F->arc[0] : -1, foe = k->fog ? w->memory[team].farc[lane] : anc_onMap(O) ? O->arc[0] : -1;
    float length = (float)ANCIENTS_LANE_LENGTH[lane];
    if (own >= 0 && foe >= 0) return 0.5f * ((float)own + length - (float)foe);
    if (own >= 0) return (float)own;
    if (foe >= 0) return length - (float)foe;
    return 0.5f * length;
}

static void anc_idle(const ancWorld *w, uint32_t h, uint32_t lane, int32_t *x, int32_t *y) {
    const ancHero *e = &w->hero[h];
    uint32_t team = h >= ANC_SIDE, f = team * ANC_LANES + lane, s = anc_front(w, 1u - team, lane);
    const ancFifo *F = &w->fifo[f];
    int32_t head = anc_onMap(F) ? F->arc[0] : ANCIENTS_STRUCTURE_ARC[anc_front(w, team, lane)], edge = ANCIENTS_LANE_LENGTH[lane] - ANCIENTS_STRUCTURE_ARC[s] - 7 * ANC_GAP - 64;
    int32_t arc = head - (ANC_ARCHETYPES[e->arch].ranged ? 640 : 0);
    if (s % ANC_OWN != ANC_ANCIENT && head < edge + 64 && arc > edge) arc = edge;
    anc_point(f, arc, x, y);
}

static uint32_t anc_prey(const ancCtx *c, uint32_t h, uint32_t known, int32_t *px, int32_t *py) {
    const ancWorld *w = c->w;
    const ancHero *e = &w->hero[h];
    uint32_t team = h >= ANC_SIDE, foe = 1u - team, open = 0u, hidden = known && c->k->fog;
    int32_t x[ANC_SIDE], y[ANC_SIDE], d[ANC_SIDE];
    for (uint32_t i = 0u; i < c->k->heroes; i++) {
        const ancHero *v = &w->hero[foe * ANC_SIDE + i];
        x[i] = hidden ? w->memory[team].hx[i] : v->x;
        y[i] = hidden ? w->memory[team].hy[i] : v->y;
        d[i] = anc_d2(x[i], y[i], e->x, e->y);
        open |= (uint32_t)(!v->dead && anc_d2(x[i], y[i], ANCIENTS_FOUNTAIN_X[foe], ANCIENTS_FOUNTAIN_Y[foe]) > ANC_R6) << i;
    }
    while (open) {
        uint32_t best = (uint32_t)__builtin_ctz(open);
        for (uint32_t rest = open & (open - 1u); rest; rest &= rest - 1u) {
            uint32_t i = (uint32_t)__builtin_ctz(rest);
            if (d[i] < d[best]) best = i;
        }
        if (d[best] <= anc_reach(e) || !anc_danger(c, team, x[best], y[best])) {
            *px = x[best];
            *py = y[best];
            return (uint32_t)ANC_HERO | (foe * ANC_SIDE + best);
        }
        open &= ~(1u << best);
    }
    return ANC_NONE;
}

static float anc_laneFloor(const ancCtx *c, uint32_t team, uint32_t viewer, uint32_t lane) {
    const ancWorld *w = c->w;
    uint32_t base = team * ANC_OWN, hidden = team != viewer && c->k->fog;
    float low = 1.0f;
    for (uint32_t i = 2u * lane; i < 2u * lane + 3u; i++) {
        uint32_t own = i == 2u * lane + 2u ? (uint32_t)ANC_ANCIENT : i;
        float hp = hidden ? w->memory[viewer].shp[own] : w->tower[base + own].hp, share = hp / ANCIENTS_STRUCTURE_HP[own];
        low = w->tower[base + own].hp > 0.0f && share < low ? share : low;
    }
    return low;
}

static uint32_t anc_desire(const ancCtx *c, uint32_t h, const float *theta) {
    const ancWorld *w = c->w;
    const ancHero *e = &w->hero[h];
    uint32_t team = h >= ANC_SIDE, foe = 1u - team, enemies = 0u, allies = 0u, nearLevel = e->level;
    int32_t nearest = INT32_MAX, cx, cy;
    float phi[ANC_FEATURES];
    for (uint32_t i = 0u; i < c->k->heroes; i++) {
        const ancHero *v = &w->hero[foe * ANC_SIDE + i], *a = &w->hero[team * ANC_SIDE + i];
        if (!v->dead) {
            int32_t x = c->k->fog ? w->memory[team].hx[i] : v->x, y = c->k->fog ? w->memory[team].hy[i] : v->y, d = anc_d2(x, y, e->x, e->y);
            if (d < nearest) { nearest = d; nearLevel = v->level; }
            enemies += d <= ANC_R12;
        }
        allies += team * ANC_SIDE + i != h && !a->dead && anc_d2(a->x, a->y, e->x, e->y) <= ANC_R12;
    }
    anc_point(team * ANC_LANES + e->lane, (int32_t)anc_clashArc(c->k, w, team, e->lane), &cx, &cy);
    const ancArchetype *a = &ANC_ARCHETYPES[e->arch];
    phi[0] = e->hp / e->maxHp;
    phi[1] = e->mana / e->maxMana;
    phi[2] = 0.25f * ((float)e->level - (float)nearLevel);
    phi[3] = 0.25f * (float)allies;
    phi[4] = 0.2f * (float)enemies;
    phi[5] = sqrtf((float)anc_d2(cx, cy, e->x, e->y)) * (1.0f / 16384.0f);
    phi[6] = anc_laneFloor(c, team, team, e->lane);
    phi[7] = anc_laneFloor(c, foe, team, e->lane);
    phi[8] = (float)(e->gold - 450u * e->items) * (1.0f / 450.0f);
    phi[9] = 0.5f * (float)((c->tick >= e->ready[0] && e->mana >= (float)a->cost[0]) + (c->tick >= e->ready[1] && e->mana >= (float)a->cost[1]));
    phi[10] = 1.0f;
    uint32_t best = ANC_LANE, farm = c->k->jungle && theta[ANC_SKILL] >= 2.0f / 3.0f;
    float4 p0, p1, p2 = { phi[8], phi[9], phi[10], 0.0f };
    float top = -INFINITY;
    memcpy(&p0, phi, sizeof p0);
    memcpy(&p1, phi + 4, sizeof p1);
    for (uint32_t m = 0u; m < ANC_MODES; m++) {
        float4 t0, t1, t2;
        memcpy(&t0, theta + m * ANC_FEATURES, sizeof t0);
        memcpy(&t1, theta + m * ANC_FEATURES + 4, sizeof t1);
        memcpy(&t2, theta + m * ANC_FEATURES + 8, sizeof t2);
        float4 sum = t0 * p0 + t1 * p1 + t2 * p2;
        float desire = (sum.x + sum.y) + (sum.z + sum.w);
        if (desire > top && (m != ANC_FARM || farm)) { top = desire; best = m; }
    }
    return best;
}

static uint32_t anc_camp(const ancCtx *c, uint32_t team, int32_t x, int32_t y) {
    const ancWorld *w = c->w;
    uint32_t best = ANC_NONE;
    int32_t bestD = INT32_MAX;
    for (uint32_t i = 0u; i < ANC_CAMPS; i++) {
        uint32_t alive = c->k->fog ? w->memory[team].camp[i] : anc_alive(&w->camp[i]);
        int32_t d = anc_d2(ANCIENTS_CAMP_X[i], ANCIENTS_CAMP_Y[i], x, y);
        if (alive && d < bestD) { bestD = d; best = (uint32_t)ANC_CAMP | i; }
    }
    return best;
}

static uint32_t anc_weakest(const ancWorld *w, uint32_t team, uint32_t lane) {
    uint32_t pick[3] = { team * ANC_OWN + 2u * lane, team * ANC_OWN + 2u * lane + 1u, team * ANC_OWN + ANC_ANCIENT }, best = pick[2];
    float low = INFINITY;
    for (uint32_t n = 0u; n < 3u; n++) {
        float f = w->tower[pick[n]].hp / ANCIENTS_STRUCTURE_HP[pick[n] % ANC_OWN];
        if (w->tower[pick[n]].hp > 0.0f && f < low) { low = f; best = pick[n]; }
    }
    return best;
}

static uint32_t anc_creepsNear(const ancCtx *c, uint32_t team, int32_t x, int32_t y, int32_t r2) {
    const ancWorld *w = c->w;
    uint32_t n = 0u;
    for (uint32_t l = 0u; l < c->k->lanes; l++) {
        const ancFifo *F = &w->fifo[(1u - team) * ANC_LANES + l];
        if (!anc_boxed(F, x, y, r2)) continue;
        for (uint32_t k = 0u; k < F->count && F->arc[k] >= 0; k++) n += anc_d2(F->x[k], F->y[k], x, y) <= r2;
    }
    return n;
}

static void anc_botCast(ancCtx *c, uint32_t h, uint32_t mode) {
    ancWorld *w = c->w;
    ancHero *e = &w->hero[h];
    const ancArchetype *type = &ANC_ARCHETYPES[e->arch];
    if ((c->tick < e->ready[0] || e->mana < (float)type->cost[0]) && (c->tick < e->ready[1] || e->mana < (float)type->cost[1])) return;
    uint32_t team = h >= ANC_SIDE, target = e->target, kind = target & ANC_KIND, hero = kind == (uint32_t)ANC_HERO ? target : ANC_NONE;
    uint32_t unit = kind == (uint32_t)ANC_CREEP || kind == (uint32_t)ANC_CAMP ? target : ANC_NONE;
    int32_t tx = 0, ty = 0, d2 = INT32_MAX;
    if (target && anc_locate(w, target, &tx, &ty)) d2 = anc_d2(tx, ty, e->x, e->y);
    else target = hero = unit = ANC_NONE;
    uint32_t fight = mode == ANC_FIGHT, work = mode == ANC_PUSH || mode == ANC_FARM || mode == ANC_LANE;
    for (uint32_t slot = 0u; slot < 2u; slot++) {
        const ancArchetype *a = &ANC_ARCHETYPES[e->arch];
        if (c->tick < e->ready[slot] || e->mana < (float)a->cost[slot]) continue;
        uint32_t cast = 0u, aim = ANC_NONE;
        float vx = 0.0f, vy = 0.0f;
        switch (e->arch * 2u + slot) {
            case ANC_CARRY * 2u:
            case ANC_CARRY * 2u + 1u: cast = target && d2 <= anc_reach(e) && (fight || work); break;
            case ANC_NUKER * 2u:
                aim = hero ? hero : work ? unit : ANC_NONE;
                cast = aim && d2 <= ANC_R8;
                break;
            case ANC_NUKER * 2u + 1u:
                cast = ((fight || mode == ANC_DEFEND) && anc_enemiesNear(c, h, ANC_R4) > 0u) || (work && anc_creepsNear(c, team, e->x, e->y, ANC_R4) >= 3u);
                break;
            case ANC_STUNNER * 2u: cast = hero && d2 <= ANC_R6; aim = hero; break;
            case ANC_STUNNER * 2u + 1u: cast = fight && anc_enemiesNear(c, h, ANC_R5) > 0u; break;
            case ANC_TANK * 2u: cast = (fight || mode == ANC_DEFEND) && anc_enemiesNear(c, h, ANC_R4) > 0u; break;
            case ANC_TANK * 2u + 1u: cast = e->hp < 0.5f * e->maxHp && anc_enemiesNear(c, h, ANC_R12) > 0u; break;
            case ANC_SUPPORT * 2u: { const ancHero *t = &w->hero[anc_ally(c, h)]; cast = t->hp < 0.6f * t->maxHp; break; }
            case ANC_SUPPORT * 2u + 1u: { const ancHero *t = &w->hero[anc_ally(c, h)]; cast = t->hp < 0.8f * t->maxHp && anc_enemiesNear(c, h, ANC_R12) > 0u; break; }
            case ANC_BLINKER * 2u: {
                int32_t gx = fight ? tx : ANCIENTS_FOUNTAIN_X[team], gy = fight ? ty : ANCIENTS_FOUNTAIN_Y[team];
                float dx = (float)(gx - e->x), dy = (float)(gy - e->y), d = sqrtf(dx * dx + dy * dy), reach = 2048.0f * anc_scale(e->level);
                if (fight && hero && d > 768.0f) {
                    float scale = fminf(1.0f, (d - 256.0f) / reach) / d;
                    vx = dx * scale; vy = dy * scale;
                    cast = d <= reach + 384.0f;
                } else if (mode == ANC_RETREAT && d > 0.0f && anc_enemiesNear(c, h, ANC_R12) > 0u) {
                    vx = dx / d; vy = dy / d;
                    cast = 1u;
                }
                break;
            }
            default:
                aim = hero ? hero : work ? unit : ANC_NONE;
                cast = aim && d2 <= ANC_R15;
                break;
        }
        if (cast) anc_cast(c, h, slot, aim, vx, vy);
    }
}

static uint32_t anc_siege(const ancCtx *c, uint32_t team, uint32_t lane, uint32_t solo) {
    const ancWorld *w = c->w;
    uint32_t s = anc_front(w, 1u - team, lane);
    const ancFifo *F = &w->fifo[team * ANC_LANES + lane];
    if (!anc_vulnerable(c, s)) return ANC_STRUCTURES;
    if (solo || s % ANC_OWN == ANC_ANCIENT || (anc_onMap(F) && (ANCIENTS_LANE_LENGTH[lane] - ANCIENTS_STRUCTURE_ARC[s]) - F->arc[0] <= 7 * ANC_GAP)) return s;
    return ANC_STRUCTURES;
}

static uint32_t anc_weakLane(const ancCtx *c, uint32_t team, uint32_t viewer, uint32_t lane) {
    const ancWorld *w = c->w;
    uint32_t base = team * ANC_OWN, hidden = team != viewer && c->k->fog;
    uint32_t best = lane;
    float low = INFINITY;
    for (uint32_t l = 0u; l < c->k->lanes; l++) {
        uint32_t pick = (l + lane) % c->k->lanes;
        float outer = hidden ? w->memory[viewer].shp[2u * pick] : w->tower[base + 2u * pick].hp, inner = hidden ? w->memory[viewer].shp[2u * pick + 1u] : w->tower[base + 2u * pick + 1u].hp;
        if (outer + inner < low) { low = outer + inner; best = pick; }
    }
    return best;
}

static void anc_retarget(ancCtx *c, uint32_t h, const float *theta) {
    ancWorld *w = c->w;
    ancHero *e = &w->hero[h];
    uint32_t team = h >= ANC_SIDE, foe = 1u - team, lane = e->lane, mode = e->mode;
    float skill = theta[ANC_SKILL];
    uint32_t rule = skill < 1.0f / 3.0f ? ANC_ANY : skill < 2.0f / 3.0f ? ANC_ONEHIT : ANC_EXACT;
    int32_t reach = anc_reach(e);
    e->target = ANC_NONE;
    if (c->tick < e->tauntUntil && !w->hero[e->tauntBy].dead) {
        e->target = (uint32_t)ANC_HERO | e->tauntBy;
        e->goalX = w->hero[e->tauntBy].x;
        e->goalY = w->hero[e->tauntBy].y;
        return;
    }
    if (mode == ANC_FARM) {
        uint32_t camp = anc_camp(c, team, e->x, e->y);
        if (camp == ANC_NONE) mode = ANC_LANE;
        else {
            const ancCamp *m = &w->camp[camp & 3u];
            e->goalX = ANCIENTS_CAMP_X[camp & 3u];
            e->goalY = ANCIENTS_CAMP_Y[camp & 3u];
            if (anc_alive(m) && anc_d2(m->x, m->y, e->x, e->y) <= reach) e->target = camp;
        }
    }
    if (mode == ANC_FIGHT) {
        int32_t px, py;
        if (anc_prey(c, h, 1u, &px, &py) == ANC_NONE) mode = ANC_LANE;
        else {
            e->goalX = px;
            e->goalY = py;
            e->target = anc_nearestHero(c, team, e->x, e->y, reach, rule == ANC_EXACT);
        }
    }
    switch (mode) {
        case ANC_LANE:
            anc_idle(w, h, e->lane, &e->goalX, &e->goalY);
            e->target = anc_lastHit(c, team, e->x, e->y, reach, rule, e->damage);
            break;
        case ANC_PUSH: {
            lane = anc_weakLane(c, foe, team, lane);
            uint32_t s = anc_siege(c, team, lane, 0u), f = anc_front(w, foe, lane);
            if (s < ANC_STRUCTURES && w->tower[s].target != ((uint32_t)ANC_HERO | h)) {
                e->goalX = ANCIENTS_STRUCTURE_X[s];
                e->goalY = ANCIENTS_STRUCTURE_Y[s];
                if (anc_d2(e->goalX, e->goalY, e->x, e->y) <= reach) e->target = (uint32_t)ANC_TOWER | s;
                break;
            } else if (anc_onMap(&w->fifo[team * ANC_LANES + lane])) anc_idle(w, h, lane, &e->goalX, &e->goalY);
            else {
                e->goalX = ANCIENTS_STRUCTURE_X[f];
                e->goalY = ANCIENTS_STRUCTURE_Y[f];
            }
            e->target = anc_lastHit(c, team, e->x, e->y, reach, ANC_ANY, e->damage);
            if (!e->target) e->target = anc_nearestHero(c, team, e->x, e->y, reach, 0u);
            if (!e->target && anc_vulnerable(c, f) && anc_d2(ANCIENTS_STRUCTURE_X[f], ANCIENTS_STRUCTURE_Y[f], e->x, e->y) <= reach) e->target = (uint32_t)ANC_TOWER | f;
            break;
        }
        case ANC_DEFEND: {
            uint32_t s = anc_weakest(w, team, anc_weakLane(c, team, team, lane));
            e->goalX = ANCIENTS_STRUCTURE_X[s];
            e->goalY = ANCIENTS_STRUCTURE_Y[s];
            e->target = anc_nearestHero(c, team, e->x, e->y, reach, 0u);
            if (!e->target) e->target = anc_lastHit(c, team, e->x, e->y, reach, ANC_ANY, e->damage);
            break;
        }
        case ANC_RETREAT:
            e->goalX = ANCIENTS_FOUNTAIN_X[team];
            e->goalY = ANCIENTS_FOUNTAIN_Y[team];
            break;
        default: break;
    }
    anc_botCast(c, h, mode);
}

static void anc_bot(ancCtx *c, uint32_t h) {
    ancWorld *w = c->w;
    ancHero *e = &w->hero[h];
    if (!anc_upkeep(c, h)) return;
    const float *theta = c->theta[h >= ANC_SIDE];
    if (((c->tick + h) & 7u) == 0u) e->mode = anc_desire(c, h, theta);
    if (--e->retarget == 0u) {
        e->retarget = anc_reaction(theta);
        anc_retarget(c, h, theta);
    }
    if (e->landAt || e->dead) return;
    int32_t tx, ty;
    if (e->target && anc_locate(w, e->target, &tx, &ty)) {
        if (anc_d2(e->x, e->y, tx, ty) <= anc_reach(e)) {
            if (c->tick >= e->attackReady && ((e->target & ANC_KIND) != ANC_TOWER || anc_vulnerable(c, e->target & 0xFFu))) anc_startAttack(c, h, e->target);
            return;
        }
    } else e->target = ANC_NONE;
    anc_walk(c, e, e->goalX, e->goalY);
}

static uint32_t anc_aim(const ancCtx *c, uint32_t side, uint32_t lane, uint32_t duel) {
    const ancWorld *w = c->w;
    const ancFifo *F = &w->fifo[side * ANC_LANES + lane];
    if (!anc_onMap(F)) return ANC_NONE;
    if (duel) return ANC_CREEP;
    uint32_t foe = 1u - side;
    for (uint32_t i = 0u; i < c->k->heroes; i++) {
        uint32_t j = foe * ANC_SIDE + i;
        const ancHero *e = &w->hero[j];
        if (e->dead || !e->harassTick || c->tick + 1u - e->harassTick > 24u) continue;
        const ancHero *v = &w->hero[e->harassVictim];
        if (anc_d2(e->x, e->y, F->x[0], F->y[0]) <= ANC_R1 && anc_d2(v->x, v->y, F->x[0], F->y[0]) <= ANC_R4) return (uint32_t)ANC_HERO | j;
    }
    uint32_t s = anc_front(w, foe, lane);
    if ((ANCIENTS_LANE_LENGTH[lane] - ANCIENTS_STRUCTURE_ARC[s]) - F->arc[0] <= ANC_GAP && anc_vulnerable(c, s)) return (uint32_t)ANC_TOWER | s;
    return ANC_NONE;
}

static int32_t anc_limit(const ancWorld *w, uint32_t side, uint32_t lane) {
    const ancFifo *O = &w->fifo[(1u - side) * ANC_LANES + lane];
    int32_t length = ANCIENTS_LANE_LENGTH[lane], limit = length - ANCIENTS_STRUCTURE_ARC[anc_front(w, 1u - side, lane)] - ANC_GAP;
    if (anc_onMap(O) && length - O->arc[0] - ANC_GAP < limit) limit = length - O->arc[0] - ANC_GAP;
    return limit;
}

static void anc_march(ancFifo *F, uint32_t f, uint32_t blocked, int32_t limit) {
    int32_t head = F->count ? F->arc[0] : 0;
    if (F->count && !blocked && head < limit) head = head + ANC_CREEP_STEP < limit ? head + ANC_CREEP_STEP : limit;
    anc_place(F, f, head);
}

static void anc_spawn(ancFifo *F, uint32_t f, uint32_t wave, float hp, uint32_t delay) {
    for (uint32_t j = 0u; j < 4u && F->count < ANC_FIFO; j++) {
        uint32_t k = F->count;
        int32_t arc = -(int32_t)(delay * ANC_CREEP_STEP + j * ANC_GAP);
        if (k > 0u && F->arc[k - 1u] - ANC_GAP < arc) arc = F->arc[k - 1u] - ANC_GAP;
        F->arc[k] = arc;
        anc_point(f, arc, &F->x[k], &F->y[k]);
        if (arc >= 0) {
            F->box[0] = F->x[k] < F->box[0] ? F->x[k] : F->box[0];
            F->box[1] = F->y[k] < F->box[1] ? F->y[k] : F->box[1];
            F->box[2] = F->x[k] > F->box[2] ? F->x[k] : F->box[2];
            F->box[3] = F->y[k] > F->box[3] ? F->y[k] : F->box[3];
        }
        F->hp[k] = hp;
        F->tag[k] = (F->nextId++ & 0xFFFFu) | wave << 16;
        F->count = k + 1u;
    }
}

static void anc_lane(ancCtx *c, uint32_t lane) {
    ancWorld *w = c->w;
    const ancientsKnobs *k = c->k;
    ancFifo *A = &w->fifo[lane], *B = &w->fifo[ANC_LANES + lane];
    uint32_t duel = anc_onMap(A) && anc_onMap(B) && (ANCIENTS_LANE_LENGTH[lane] - B->arc[0]) - A->arc[0] <= ANC_GAP;
    uint32_t aimA = anc_aim(c, 0u, lane, duel), aimB = anc_aim(c, 1u, lane, duel), headA = A->tag[0], headB = B->tag[0];
    float toA = 0.0f, toB = 0.0f;
    if (aimA && c->tick >= A->volleyAt) {
        float amount = anc_volley(k, A);
        A->volleyAt = c->tick + 12u;
        if (aimA == (uint32_t)ANC_CREEP) toB = amount;
        else anc_apply(c, aimA, amount, ANC_NONE, 0u);
    }
    if (aimB && c->tick >= B->volleyAt) {
        float amount = anc_volley(k, B);
        B->volleyAt = c->tick + 12u;
        if (aimB == (uint32_t)ANC_CREEP) toA = amount;
        else anc_apply(c, aimB, amount, ANC_NONE, 0u);
    }
    if (toB > 0.0f) B->hp[0] -= toB;
    if (toA > 0.0f) A->hp[0] -= toA;
    if (toB > 0.0f && !(B->hp[0] > 0.0f)) anc_creepDies(c, ANC_LANES + lane, 0u, ANC_NONE);
    if (toA > 0.0f && !(A->hp[0] > 0.0f)) anc_creepDies(c, lane, 0u, ANC_NONE);
    int32_t limitA = anc_limit(w, 0u, lane), limitB = anc_limit(w, 1u, lane);
    anc_march(A, lane, aimA && A->count && A->tag[0] == headA, limitA);
    anc_march(B, ANC_LANES + lane, aimB && B->count && B->tag[0] == headB, limitB);
    if (c->tick % k->waveTicks == 0u) {
        uint32_t wave = c->tick / k->waveTicks;
        float hp = anc_waveHp(k, wave);
        u4 delay = philox4x32(c->seed, ENV_RNG_STREAM, c->game, wave, 1u + lane, 0u);
        anc_spawn(A, lane, wave, hp, delay.x >> 28);
        anc_spawn(B, ANC_LANES + lane, wave, hp, delay.y >> 28);
        if (lane == 0u) anc_event(w, EV_WAVE);
    }
}

static uint32_t anc_towerPick(const ancCtx *c, uint32_t team, int32_t x, int32_t y) {
    const ancWorld *w = c->w;
    uint32_t best = ANC_NONE;
    int32_t bestD = ANC_R7;
    for (uint32_t l = 0u; l < c->k->lanes; l++) {
        uint32_t f = (1u - team) * ANC_LANES + l;
        const ancFifo *F = &w->fifo[f];
        if (!anc_onMap(F)) continue;
        int32_t d = anc_d2(F->x[0], F->y[0], x, y);
        if (d <= bestD) { bestD = d; best = anc_creepRef(f, F, 0u); }
    }
    return best ? best : anc_nearestHero(c, team, x, y, ANC_R7, 0u);
}

static uint32_t anc_quiet(const ancCtx *c, uint32_t team, int32_t x, int32_t y) {
    const ancWorld *w = c->w;
    uint32_t foe = 1u - team;
    int32_t near = INT32_MAX;
    for (uint32_t i = 0u; i < c->k->heroes; i++) {
        int32_t d = anc_d2(w->hero[foe * ANC_SIDE + i].x, w->hero[foe * ANC_SIDE + i].y, x, y);
        near = d < near ? d : near;
    }
    for (uint32_t l = 0u; l < c->k->lanes; l++) {
        const int32_t *b = w->fifo[foe * ANC_LANES + l].box;
        int32_t dx = b[0] - x > 0 ? b[0] - x : x - b[2] > 0 ? x - b[2] : 0, dy = b[1] - y > 0 ? b[1] - y : y - b[3] > 0 ? y - b[3] : 0;
        dx = dx < 16384 ? dx : 16384;
        dy = dy < 16384 ? dy : 16384;
        near = dx * dx + dy * dy < near ? dx * dx + dy * dy : near;
    }
    float gap = sqrtf((float)near) - 1792.0f;
    return gap > 76.8f ? (uint32_t)(gap / 76.8f) : 0u;
}

static void anc_tower(ancCtx *c, uint32_t s) {
    ancWorld *w = c->w;
    ancTower *t = &w->tower[s];
    if (!(t->hp > 0.0f) || c->tick < t->readyAt || c->tick < t->idleUntil) return;
    uint32_t team = s >= ANC_OWN, old = t->target, pick = old;
    int32_t x = ANCIENTS_STRUCTURE_X[s], y = ANCIENTS_STRUCTURE_Y[s], tx, ty;
    int keep = old && anc_locate(w, old, &tx, &ty) && anc_d2(tx, ty, x, y) <= ANC_R7;
    if (t->aggroUntil && c->tick >= t->aggroUntil) {
        t->aggroUntil = 0u;
        keep = 0;
    }
    if (!keep) pick = anc_towerPick(c, team, x, y);
    if (team == 1u && old != pick) {
        if (old == (uint32_t)ANC_HERO) anc_event(w, EV_RELEASED);
        if (pick == (uint32_t)ANC_HERO) anc_event(w, EV_ACQUIRED);
    }
    t->target = pick;
    if (pick) {
        t->readyAt = c->tick + 20u;
        anc_apply(c, pick, 100.0f, ANC_NONE, 0u);
    } else t->idleUntil = c->tick + anc_quiet(c, team, x, y);
}

static void anc_towers(ancCtx *c, uint32_t team) {
    for (uint32_t i = 0u; i < 2u * c->k->lanes; i++) anc_tower(c, team * ANC_OWN + i);
}

static void anc_herd(ancCamp *m, int32_t gx, int32_t gy) {
    float dx = (float)(gx - m->x), dy = (float)(gy - m->y), d2 = dx * dx + dy * dy;
    if (d2 <= (float)(ANC_CREEP_STEP * ANC_CREEP_STEP)) { m->x = gx; m->y = gy; return; }
    float s = (float)ANC_CREEP_STEP / sqrtf(d2);
    m->x += (int32_t)floorf(dx * s + 0.5f);
    m->y += (int32_t)floorf(dy * s + 0.5f);
}

static void anc_camps(ancCtx *c) {
    ancWorld *w = c->w;
    for (uint32_t i = 0u; i < ANC_CAMPS; i++) {
        ancCamp *m = &w->camp[i];
        uint32_t alive = anc_alive(m);
        if (!alive) {
            if (c->tick >= m->respawnAt) {
                m->hp[0] = m->hp[1] = m->hp[2] = 400.0f;
                m->x = ANCIENTS_CAMP_X[i]; m->y = ANCIENTS_CAMP_Y[i]; m->aggro = ANC_NONE; m->respawnAt = UINT32_MAX;
            }
            continue;
        }
        const ancHero *e = m->aggro ? &w->hero[m->aggro & 0xFFu] : 0;
        if (!e || e->dead || anc_d2(e->x, e->y, ANCIENTS_CAMP_X[i], ANCIENTS_CAMP_Y[i]) > ANC_R6) {
            m->aggro = ANC_NONE;
            anc_herd(m, ANCIENTS_CAMP_X[i], ANCIENTS_CAMP_Y[i]);
            continue;
        }
        if (anc_d2(e->x, e->y, m->x, m->y) <= ANC_R1) {
            if (c->tick >= m->readyAt) {
                m->readyAt = c->tick + 12u;
                anc_hitHero(c, m->aggro & 0xFFu, 30.0f * (float)alive, ANC_NONE);
            }
        } else anc_herd(m, e->x, e->y);
    }
}

typedef struct { _Alignas(16) int32_t x[16], y[16], r[16]; } ancEyes;

static uint32_t anc_seen(const ancEyes *v, int32_t x, int32_t y) {
    int4 px = x, py = y, hit = 0;
    for (uint32_t i = 0u; i < 16u; i += 4u) {
        int4 ax, ay, r;
        memcpy(&ax, v->x + i, sizeof ax);
        memcpy(&ay, v->y + i, sizeof ay);
        memcpy(&r, v->r + i, sizeof r);
        int4 dx = ax - px, dy = ay - py;
        hit |= dx * dx + dy * dy <= r;
    }
    return (uint32_t)(hit.x | hit.y | hit.z | hit.w) & 1u;
}

static void anc_vision(ancCtx *c) {
    ancWorld *w = c->w;
    const ancientsKnobs *k = c->k;
    uint32_t team = c->tick & 1u, foe = 1u - team;
    ancEyes v;
    for (uint32_t i = 0u; i < ANC_SIDE; i++) {
        const ancHero *e = &w->hero[team * ANC_SIDE + i];
        v.x[i] = e->x;
        v.y[i] = e->y;
        v.r[i] = i < k->heroes && !e->dead ? ANC_R12 : -1;
    }
    for (uint32_t i = 0u; i < ANC_OWN; i++) {
        uint32_t s = team * ANC_OWN + i;
        v.x[ANC_SIDE + i] = ANCIENTS_STRUCTURE_X[s];
        v.y[ANC_SIDE + i] = ANCIENTS_STRUCTURE_Y[s];
        v.r[ANC_SIDE + i] = w->tower[s].hp > 0.0f ? ANC_R10 : -1;
    }
    for (uint32_t l = 0u; l < ANC_LANES; l++) {
        const ancFifo *F = &w->fifo[team * ANC_LANES + l];
        v.x[ANC_SIDE + ANC_OWN + l] = F->x[0];
        v.y[ANC_SIDE + ANC_OWN + l] = F->y[0];
        v.r[ANC_SIDE + ANC_OWN + l] = l < k->lanes && anc_onMap(F) ? ANC_R8 : -1;
    }
    v.x[15] = v.y[15] = 0;
    v.r[15] = -1;
    ancMemory *m = &w->memory[team];
    for (uint32_t i = 0u; i < k->heroes; i++) {
        const ancHero *e = &w->hero[foe * ANC_SIDE + i];
        m->hvisible[i] = !e->dead && anc_seen(&v, e->x, e->y);
        if (m->hvisible[i]) { m->hx[i] = e->x; m->hy[i] = e->y; m->hhp[i] = e->hp / e->maxHp; m->hseen[i] = c->tick + 1u; }
    }
    for (uint32_t i = 0u; i < ANC_OWN; i++) {
        uint32_t s = foe * ANC_OWN + i;
        if (m->shp[i] == w->tower[s].hp) continue;
        if (!(w->tower[s].hp > 0.0f) || anc_seen(&v, ANCIENTS_STRUCTURE_X[s], ANCIENTS_STRUCTURE_Y[s])) m->shp[i] = w->tower[s].hp;
    }
    for (uint32_t l = 0u; l < k->lanes; l++) {
        uint32_t f = foe * ANC_LANES + l;
        const ancFifo *F = &w->fifo[f];
        if (anc_onMap(F) && anc_seen(&v, F->x[0], F->y[0])) { m->farc[l] = F->arc[0]; m->fhp[l] = anc_total(F); }
        else if (m->farc[l] >= 0) {
            int32_t px, py;
            anc_point(f, m->farc[l], &px, &py);
            if (anc_seen(&v, px, py)) { m->farc[l] = -1; m->fhp[l] = 0.0f; }
        }
    }
    if (k->jungle)
        for (uint32_t i = 0u; i < ANC_CAMPS; i++) {
            uint32_t alive = anc_alive(&w->camp[i]);
            if (m->camp[i] != alive && anc_seen(&v, ANCIENTS_CAMP_X[i], ANCIENTS_CAMP_Y[i])) m->camp[i] = alive;
        }
}

static inline __attribute__((always_inline)) void anc_steer(ancCtx *c, const float *ctrl) {
    ancWorld *w = c->w;
    ancHero *e = &w->hero[0];
    if (e->dead) return;
    float vx = oak_clamp(ctrl[0], -1.0f, 1.0f), vy = oak_clamp(ctrl[1], -1.0f, 1.0f), n = vx * vx + vy * vy;
    if (n > 1.0f) {
        float s = 1.0f / sqrtf(n);
        vx *= s;
        vy *= s;
    }
    uint32_t cls = ctrl[3] < -1.0f / 3.0f ? 0u : ctrl[3] > 1.0f / 3.0f ? 2u : 1u;
    if (ctrl[2] > 0.0f) { w->latch |= 1u; w->latchClass[0] = cls; }
    if (ctrl[4] > 0.0f) { w->latch |= 2u; w->latchClass[1] = cls; w->latchX = vx; w->latchY = vy; }
    if (ctrl[5] > 0.0f) { w->latch |= 4u; w->latchClass[2] = cls; w->latchX = vx; w->latchY = vy; }
    if (c->tick < e->stunUntil) return;
    float step = (c->tick < e->slowUntil ? 38.4f : 76.8f) / (float)c->k->substeps;
    anc_shift(&e->x, &e->rx, vx * step);
    anc_shift(&e->y, &e->ry, vy * step);
}

static void anc_command(ancCtx *c, uint32_t slot, uint32_t cls) {
    ancWorld *w = c->w;
    uint32_t code = w->hero[0].arch * 2u + slot, target = ANC_NONE;
    int32_t r2 = code == ANC_NUKER * 2u ? ANC_R8 : code == ANC_STUNNER * 2u ? ANC_R6 : code == ANC_BLINKER * 2u + 1u ? ANC_R15 : 0;
    if (r2) {
        target = anc_pick(c, 0u, cls, r2, 1u);
        if (!target) return;
    }
    anc_cast(c, 0u, slot, target, w->latchX, w->latchY);
}

static void anc_act(ancCtx *c) {
    ancWorld *w = c->w;
    ancHero *e = &w->hero[0];
    uint32_t latch = w->latch;
    w->latch = 0u;
    if (e->dead || c->tick < e->stunUntil) return;
    if (latch & 2u) anc_command(c, 0u, w->latchClass[1]);
    if (latch & 4u) anc_command(c, 1u, w->latchClass[2]);
    if ((latch & 1u) && !e->landAt && c->tick >= e->attackReady) {
        uint32_t target = anc_pick(c, 0u, w->latchClass[0], anc_reach(e), 0u);
        if (target) anc_startAttack(c, 0u, target);
    }
}

static void anc_job(ancCtx *c, uint32_t job) {
    switch (job) {
        case JOB_LANES:
            anc_lane(c, 0u);
            if (c->k->lanes > 1u) anc_lane(c, 1u);
            return;
        case JOB_LANE: anc_lane(c, 2u); return;
        case JOB_TOWERS:
            anc_towers(c, 0u);
            if (c->k->lanes == 1u) {
                anc_towers(c, 1u);
                if (c->k->fog) anc_vision(c);
            }
            return;
        case JOB_TOWERS_B: anc_towers(c, 1u); return;
        case JOB_COMMIT:
            if (anc_upkeep(c, 0u)) anc_act(c);
            else c->w->latch = 0u;
            return;
        case JOB_VISION: anc_vision(c); return;
        case JOB_UPKEEP: anc_upkeep(c, 0u); return;
        case JOB_ACT: anc_act(c); return;
        case JOB_CAMPS: anc_camps(c); return;
        case JOB_IDLE: return;
        default: anc_bot(c, job); return;
    }
}

static inline __attribute__((always_inline)) void anc_advance(ancCtx *c, uint32_t *words, const float *ctrl) {
    uint32_t slice = words[1];
    anc_steer(c, ctrl);
    anc_job(c, c->jobs[slice]);
    if (++slice == c->k->substeps) {
        slice = 0u;
        words[0] = ++c->tick;
    }
    words[1] = slice;
}

static void anc_draw(const ancientsKnobs *k, uint32_t seed, uint32_t game, uint32_t *arch, uint32_t *seat, uint32_t *pick) {
    uint32_t r[20];
    for (uint32_t b = 0u; b < 5u; b++) {
        u4 q = philox4x32(seed, ENV_RNG_STREAM, game, b, 0u, 0u);
        r[4u * b] = q.x; r[4u * b + 1u] = q.y; r[4u * b + 2u] = q.z; r[4u * b + 3u] = q.w;
    }
    for (uint32_t team = 0u; team < 2u; team++) {
        uint32_t *a = arch + team * ANC_SIDE, *s = seat + team * ANC_SIDE, deck[6] = { 0u, 1u, 2u, 3u, 4u, 5u };
        for (uint32_t i = 5u; i > 0u; i--) {
            uint32_t j = (uint32_t)(u01(r[team * 5u + 5u - i]) * (float)(i + 1u)), t = deck[i];
            deck[i] = deck[j];
            deck[j] = t;
        }
        for (uint32_t i = 0u; i < k->heroes; i++) {
            a[i] = k->draft == 0u ? (uint32_t)ANC_CARRY : k->draft == 1u ? (uint32_t)(u01(r[team * 5u + i]) * 6.0f) : deck[i];
            s[i] = i;
        }
        for (uint32_t i = k->heroes - 1u; i > 0u; i--) {
            uint32_t j = (uint32_t)(u01(r[10u + team * 4u + k->heroes - 1u - i]) * (float)(i + 1u)), t = s[i];
            s[i] = s[j];
            s[j] = t;
        }
    }
    *pick = r[18];
}

static void anc_setup(ancCtx *c, uint32_t seed, uint32_t game, const ancLeague *g) {
    ancWorld *w = c->w;
    const ancientsKnobs *k = c->k;
    uint32_t arch[2 * ANC_SIDE], seat[2 * ANC_SIDE], pick;
    anc_draw(k, seed, game, arch, seat, &pick);
    memset(w, 0, sizeof *w);
    for (uint32_t f = 0u; f < 2u * ANC_LANES; f++) {
        w->fifo[f].box[0] = w->fifo[f].box[1] = 1 << 20;
        w->fifo[f].box[2] = w->fifo[f].box[3] = -(1 << 20);
    }
    for (uint32_t team = 0u; team < 2u; team++)
        for (uint32_t i = 0u; i < k->heroes; i++) {
            ancHero *e = &w->hero[team * ANC_SIDE + i];
            e->arch = arch[team * ANC_SIDE + i];
            e->level = 1u;
            e->maxHp = 600.0f + (float)ANC_ARCHETYPES[e->arch].hpBonus;
            e->hp = e->maxHp;
            e->maxMana = 300.0f;
            e->mana = 300.0f;
            e->damage = 50.0f;
            e->x = ANCIENTS_FOUNTAIN_X[team];
            e->y = ANCIENTS_FOUNTAIN_Y[team];
            e->lane = (uint32_t)ANCIENTS_SEAT_LANE[seat[team * ANC_SIDE + i]];
            e->retarget = 1u + (i & 1u);
        }
    for (uint32_t s = 0u; s < ANC_STRUCTURES; s++) w->tower[s].hp = anc_exists(k, s) ? ANCIENTS_STRUCTURE_HP[s % ANC_OWN] : 0.0f;
    for (uint32_t i = 0u; i < ANC_CAMPS; i++) {
        ancCamp *m = &w->camp[i];
        m->x = ANCIENTS_CAMP_X[i];
        m->y = ANCIENTS_CAMP_Y[i];
        m->hp[0] = m->hp[1] = m->hp[2] = k->jungle ? 400.0f : 0.0f;
        m->respawnAt = UINT32_MAX;
    }
    for (uint32_t team = 0u; team < 2u; team++) {
        ancMemory *m = &w->memory[team];
        uint32_t foe = 1u - team;
        for (uint32_t i = 0u; i < ANC_SIDE; i++) {
            m->hx[i] = ANCIENTS_FOUNTAIN_X[foe];
            m->hy[i] = ANCIENTS_FOUNTAIN_Y[foe];
            m->hhp[i] = 1.0f;
            m->hseen[i] = 1u;
        }
        for (uint32_t i = 0u; i < ANC_OWN; i++) m->shp[i] = w->tower[foe * ANC_OWN + i].hp;
        for (uint32_t l = 0u; l < ANC_LANES; l++) m->farc[l] = -1;
        for (uint32_t i = 0u; i < ANC_CAMPS; i++) m->camp[i] = k->jungle ? 3u : 0u;
    }
    if (k->league) w->member = leaguePick(g->winRate, pick);
    c->tick = 0u;
    c->game = game;
    c->theta[1] = k->league ? g->theta + w->member * ANC_THETA : ANC_PRESETS[k->tier];
}

static void anc_leagueInit(ancLeague *g, uint32_t seed) {
    for (uint32_t i = 0u; i < LEAGUE_MEMBERS; i++) {
        const float *base = ANC_PRESETS[i < 3u ? i : 2u];
        float *theta = g->theta + i * ANC_THETA;
        g->winRate[i] = 0.5f;
        for (uint32_t d = 0u; d < ANC_THETA; d += 4u) {
            u44 r = philox4(seed, LEAGUE_RNG_STREAM, (lane4)i, d + (lane4){ 0u, 1u, 2u, 3u }, (lane4)1u, (lane4)0u);
            float4 n = normal4(r.x, r.y);
            for (uint32_t j = 0u; j < 4u && d + j < ANC_THETA; j++) {
                float value = base[d + j] + 0.1f * (ANC_BOUNDS[1][d + j] - ANC_BOUNDS[0][d + j]) * n[j];
                theta[d + j] = i < 3u ? base[d + j] : fminf(fmaxf(value, ANC_BOUNDS[0][d + j]), ANC_BOUNDS[1][d + j]);
            }
        }
    }
}

static __attribute__((noinline)) float anc_finish(ancCtx *c, uint32_t seed, uint32_t *words, uint32_t *extra) {
    ancWorld *w = c->w;
    ancLeague *g = anc_league(extra);
    uint32_t result = w->result;
    float reward = result == 1u ? 1.0f : result == 2u ? -1.0f : 0.0f;
    words[result == 1u ? 3 : result == 2u ? 4 : 5]++;
    if (c->k->league) {
        leagueScore(g->winRate, w->member, result == 1u ? 1.0f : result == 2u ? 0.0f : 0.5f);
        leagueRenew(g->winRate, g->theta, ANC_THETA, ANC_BOUNDS[0], ANC_BOUNDS[1], seed, words[2]);
    }
    words[2]++;
    words[0] = 0u;
    words[1] = 0u;
    anc_setup(c, seed, words[2], g);
    return reward;
}

static inline __attribute__((always_inline)) float anc_along(const ancientsKnobs *k, float x, float y, float reach, uint32_t *lane, uint32_t *near) {
    float u = x - 2048.0f, v = y - 14336.0f, along = (u - v) * 0.70710678f, perp = (u + v) * 0.70710678f;
    float arc = fminf(fmaxf(along, 0.0f), 17378.0f), over = along - arc, best = perp * perp + over * over;
    *lane = 0u;
    *near = best <= reach;
    if (k->lanes == 1u) return arc;
    if (u <= 0.0f && v >= 0.0f) {
        float d = u * u + v * v;
        *near |= (uint32_t)(d <= reach) * 6u;
        *lane = d < best;
        return arc;
    }
    float a0 = fminf(fmaxf(14336.0f - y, 0.0f), 12288.0f), a1 = fminf(fmaxf(u, 0.0f), 12288.0f);
    float ex = x - (2048.0f + a1), ey = y - (14336.0f - a0), top = y - 2048.0f, right = x - 14336.0f;
    float d0 = u * u + ey * ey, d1 = ex * ex + top * top, d2 = ex * ex + v * v, d3 = right * right + ey * ey;
    *near |= (uint32_t)(fminf(d0, d1) <= reach) << 1 | (uint32_t)(fminf(d2, d3) <= reach) << 2;
    if (d0 < best) { best = d0; arc = a0; *lane = 1u; }
    if (d1 < best) { best = d1; arc = a1 + 12288.0f; *lane = 1u; }
    if (d2 < best) { best = d2; arc = a1; *lane = 2u; }
    if (d3 < best) { arc = a0 + 12288.0f; *lane = 2u; }
    return arc;
}

static const uint8_t ANC_ROSTER[3][33] = {
    { 0x05, 0x10, 0x11, 0x16, 0x17, 0x18, 0x1D, 0x20, 0x23 },
    { 0x01, 0x02, 0x05, 0x06, 0x07, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x20, 0x21, 0x22, 0x23, 0x24, 0x25 },
    { 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x30, 0x31, 0x32, 0x33 }
};

static uint32_t anc_roster(const ancientsKnobs *k) { return k->heroes == 1u ? 9u : k->heroes == 3u ? 25u : 33u; }

static inline __attribute__((always_inline)) void anc_sweep(const ancientsKnobs *k, const ancWorld *w, uint32_t tick, const ancHero *a, float *out) {
    const ancMemory *m = &w->memory[0];
    uint32_t entry = ANC_ROSTER[k->heroes >> 1][w->sweep], index = entry & 15u;
    float code = (float)(w->sweep + 1u) * (1.0f / 64.0f), value;
    int32_t x, y;
    switch (entry >> 4) {
        case 0u: {
            const ancHero *e = &w->hero[index];
            uint32_t hidden = index >= ANC_SIDE && k->fog && !e->dead;
            x = hidden ? m->hx[index - ANC_SIDE] : e->x;
            y = hidden ? m->hy[index - ANC_SIDE] : e->y;
            value = e->dead ? 0.0f : hidden ? m->hhp[index - ANC_SIDE] : e->hp / e->maxHp;
            if (index >= ANC_SIDE) code = -code;
            break;
        }
        case 1u: {
            uint32_t i = index % ANC_OWN, team = index >= ANC_OWN;
            x = ANCIENTS_STRUCTURE_X[index];
            y = ANCIENTS_STRUCTURE_Y[index];
            value = (team && k->fog ? m->shp[i] : w->tower[index].hp) / ANCIENTS_STRUCTURE_HP[i];
            if (team) code = -code;
            break;
        }
        case 2u: {
            const ancFifo *F = &w->fifo[index];
            uint32_t side = index >= ANC_LANES, l = index % ANC_LANES;
            int32_t arc = side && k->fog ? m->farc[l] : anc_onMap(F) ? F->arc[0] : -1;
            anc_point(index, arc, &x, &y);
            value = (side && k->fog ? m->fhp[l] : anc_total(F)) / (4.0f * anc_waveHp(k, tick / k->waveTicks));
            if (side) code = -code;
            break;
        }
        default:
            x = ANCIENTS_CAMP_X[index];
            y = ANCIENTS_CAMP_Y[index];
            value = (float)(k->fog ? m->camp[index] : anc_alive(&w->camp[index])) * (1.0f / 3.0f);
            code += 0.5f;
            break;
    }
    out[0] = code;
    out[1] = (float)(x - a->x) * (1.0f / 16384.0f);
    out[2] = (float)(y - a->y) * (1.0f / 16384.0f);
    out[3] = value;
}

static inline __attribute__((always_inline)) void anc_observe(const ancientsKnobs *k, const ancWorld *w, const uint32_t *words, float *obs) {
    uint32_t tick = words[0], near = ANC_SIDE, lane;
    const ancHero *a = &w->hero[0];
    const ancArchetype *type = &ANC_ARCHETYPES[a->arch];
    const ancMemory *m = &w->memory[0];
    float inv = 1.0f / 16384.0f, low;
    int32_t best = INT32_MAX, nx = 0, ny = 0;
    obs[0] = a->dead ? 0.0f : a->hp / a->maxHp;
    obs[1] = a->mana / a->maxMana;
    obs[2] = a->ready[0] > tick ? (float)(a->ready[0] - tick) / (float)type->cooldown[0] : 0.0f;
    obs[3] = a->ready[1] > tick ? (float)(a->ready[1] - tick) / (float)type->cooldown[1] : 0.0f;
    obs[4] = (float)a->x * inv;
    obs[5] = (float)a->y * inv;
    for (uint32_t i = 0u; i < k->heroes; i++) {
        const ancHero *e = &w->hero[ANC_SIDE + i];
        int32_t x = k->fog ? m->hx[i] : e->x, y = k->fog ? m->hy[i] : e->y, d = anc_d2(x, y, a->x, a->y);
        if (!e->dead && d < best) { best = d; near = i; nx = x; ny = y; }
    }
    obs[6] = near < ANC_SIDE ? (float)(nx - a->x) * inv : 0.0f;
    obs[7] = near < ANC_SIDE ? (float)(ny - a->y) * inv : 0.0f;
    obs[8] = near < ANC_SIDE ? k->fog ? m->hhp[near] : w->hero[ANC_SIDE + near].hp / w->hero[ANC_SIDE + near].maxHp : 0.0f;
    obs[9] = near < ANC_SIDE && k->fog && !m->hvisible[near] ? (float)(tick + 1u - m->hseen[near]) * (1.0f / 120.0f) : 0.0f;
    uint32_t ranged = ANC_ARCHETYPES[a->arch].ranged, lanes;
    float arc = anc_along(k, (float)a->x, (float)a->y, ranged ? (float)ANC_NEAR5 : (float)ANC_NEAR15, &lane, &lanes);
    obs[10] = (anc_clashArc(k, w, 0u, lane) - arc) * inv;
    obs[11] = !a->dead && anc_lowest(w, lanes, 0u, a->x, a->y, ranged ? ANC_R5 : ANC_R15, &low) ? low / a->damage : 0.0f;
    anc_sweep(k, w, tick, a, obs + 12);
    obs[16] = 0.1f * (float)w->eventShown;
    obs[17] = (float)(tick % k->waveTicks) / (float)k->waveTicks;
}

static inline __attribute__((always_inline)) void anc_emit(const ancientsKnobs *k, ancWorld *w, const uint32_t *words, float *obs) {
    if (w->eventCount) {
        w->eventShown = w->event[w->eventHead];
        w->eventHead = (w->eventHead + 1u) & (ANC_EVENTS - 1u);
        w->eventCount--;
    } else w->eventShown = 0u;
    w->sweep = w->sweep + 1u == anc_roster(k) ? 0u : w->sweep + 1u;
    anc_observe(k, w, words, obs);
}

uint32_t ancients_extraWords(uint32_t instance, uint32_t steps) {
    (void)instance; (void)steps;
    return 4u + (uint32_t)((sizeof(ancWorld) + sizeof(ancLeague) + sizeof(ancOracle)) / sizeof(uint32_t));
}

void ancients_reset(uint32_t instance, uint32_t seed, uint32_t *words, uint32_t *extra, uint32_t steps) {
    (void)steps;
    const ancientsKnobs *k = &ANCIENTS_KNOBS[instance];
    ancLeague *g = anc_league(extra);
    ancCtx c;
    if (k->league) anc_leagueInit(g, seed);
    anc_oracleOf(extra)->primed = 0u;
    anc_oracleOf(extra)->planGame = UINT32_MAX;
    for (uint32_t i = 0u; i < 6u; i++) words[i] = 0u;
    anc_bind(&c, k, instance, anc_world(extra), g, seed, words);
    anc_setup(&c, seed, 0u, g);
}

static inline __attribute__((always_inline)) float anc_play(const ancientsKnobs *k, uint32_t instance, uint32_t seed, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done) {
    ancWorld *w = anc_world(extra);
    ancCtx c;
    float reward = 0.0f;
    anc_bind(&c, k, instance, w, anc_league(extra), seed, words);
    anc_advance(&c, words, ctrl);
    *done = 0u;
    if (w->result || words[0] >= k->tickCap) {
        reward = anc_finish(&c, seed, words, extra);
        *done = 1u;
    }
    anc_emit(k, w, words, obs);
    return reward;
}

float ancients_step(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done, const float *shared) {
    (void)step; (void)shared;
    switch (instance) {
        case 0u: return anc_play(&ANCIENTS_KNOBS[0], 0u, seed, ctrl, words, extra, obs, done);
        case 1u: return anc_play(&ANCIENTS_KNOBS[1], 1u, seed, ctrl, words, extra, obs, done);
        case 2u: return anc_play(&ANCIENTS_KNOBS[2], 2u, seed, ctrl, words, extra, obs, done);
        case 3u: return anc_play(&ANCIENTS_KNOBS[3], 3u, seed, ctrl, words, extra, obs, done);
        default: return anc_play(&ANCIENTS_KNOBS[4], 4u, seed, ctrl, words, extra, obs, done);
    }
}

void ancients_observe(uint32_t instance, uint32_t seed, const uint32_t *words, const uint32_t *extra, uint32_t step, float *obs, const float *shared) {
    (void)seed; (void)step; (void)shared;
    anc_observe(&ANCIENTS_KNOBS[instance], anc_worldAt(extra), words, obs);
}

static void anc_chase(ancPlan *p, int32_t x, int32_t y) {
    p->x = x;
    p->y = y;
    p->chase = 1u;
}

static int anc_engage(const ancCtx *c, ancPlan *p) {
    float *ctrl = p->ctrl;
    const ancWorld *w = c->w;
    const ancHero *e = &w->hero[0];
    int32_t tx, ty;
    if (anc_prey(c, 0u, 0u, &tx, &ty) == ANC_NONE) return 0;
    int32_t d2 = anc_d2(tx, ty, e->x, e->y), reach = anc_reach(e);
    if (d2 > ANC_R10) return 0;
    if (d2 > reach - reach / 5) anc_chase(p, tx, ty);
    ctrl[2] = 1.0f;
    ctrl[3] = 0.0f;
    float m = anc_scale(e->level), d = sqrtf((float)d2);
    const ancHero *ally = &w->hero[anc_ally(c, 0u)];
    uint32_t first = 0u, second = 0u;
    switch (e->arch) {
        case ANC_CARRY: first = second = d2 <= reach; break;
        case ANC_NUKER: first = d2 <= ANC_R8; second = d2 <= ANC_R4; break;
        case ANC_STUNNER: first = d2 <= ANC_R6; second = d2 <= ANC_R5; break;
        case ANC_TANK: first = d2 <= ANC_R4; second = e->hp < 0.5f * e->maxHp; break;
        case ANC_SUPPORT: first = ally->hp < 0.6f * ally->maxHp; second = ally->hp < 0.8f * ally->maxHp; break;
        default:
            if (d > 768.0f && d <= 2048.0f * m + 384.0f) {
                float scale = fminf(1.0f, (d - 256.0f) / (2048.0f * m)) / d;
                ctrl[0] = (float)(tx - e->x) * scale;
                ctrl[1] = (float)(ty - e->y) * scale;
                p->chase = 0u;
                first = 1u;
            }
            second = d2 <= ANC_R15;
            break;
    }
    ctrl[4] = first ? 1.0f : 0.0f;
    ctrl[5] = second ? 1.0f : 0.0f;
    return 1;
}

static void anc_boost(const ancCtx *c, float *ctrl) {
    const ancHero *e = &c->w->hero[0];
    switch (e->arch) {
        case ANC_CARRY: ctrl[4] = 1.0f; ctrl[5] = 1.0f; break;
        case ANC_NUKER: ctrl[5] = anc_creepsNear(c, 0u, e->x, e->y, ANC_R4) >= 3u ? 1.0f : 0.0f; break;
        default: break;
    }
}

static uint32_t anc_threat(const ancCtx *c) {
    const ancWorld *w = c->w;
    uint32_t best = ANC_STRUCTURES;
    int32_t depth = INT32_MAX;
    for (uint32_t l = 0u; l < c->k->lanes; l++) {
        uint32_t s = anc_front(w, 0u, l), seen = 0u;
        const ancFifo *F = &w->fifo[ANC_LANES + l];
        if (!(w->tower[s].hp < 0.9f * ANCIENTS_STRUCTURE_HP[s])) continue;
        seen |= anc_onMap(F) && (ANCIENTS_LANE_LENGTH[l] - ANCIENTS_STRUCTURE_ARC[s]) - F->arc[0] <= 3 * ANC_GAP;
        for (uint32_t i = 0u; i < c->k->heroes; i++) {
            const ancHero *v = &w->hero[ANC_SIDE + i];
            seen |= !v->dead && anc_d2(v->x, v->y, ANCIENTS_STRUCTURE_X[s], ANCIENTS_STRUCTURE_Y[s]) <= ANC_R10;
        }
        if (seen && ANCIENTS_STRUCTURE_ARC[s] < depth) { depth = ANCIENTS_STRUCTURE_ARC[s]; best = s; }
    }
    return best;
}

static uint32_t anc_contested(const ancCtx *c) {
    const ancWorld *w = c->w;
    uint32_t lane = 0u;
    float worst = INFINITY;
    for (uint32_t l = 0u; l < c->k->lanes; l++) {
        float balance = anc_total(&w->fifo[l]) - anc_total(&w->fifo[ANC_LANES + l]);
        if (balance < worst) { worst = balance; lane = l; }
    }
    return lane;
}

static void anc_hold(const ancCtx *c, ancPlan *p) {
    float *ctrl = p->ctrl;
    const ancWorld *w = c->w;
    const ancHero *e = &w->hero[0];
    int32_t reach = anc_reach(e), wide = ANC_ARCHETYPES[e->arch].ranged ? ANC_R6 : 409600, tx, ty;
    float low, health = e->hp / e->maxHp;
    if (health < 0.25f || (health < 0.9f && anc_d2(e->x, e->y, ANCIENTS_FOUNTAIN_X[0], ANCIENTS_FOUNTAIN_Y[0]) <= ANC_R12)) {
        anc_chase(p, ANCIENTS_FOUNTAIN_X[0], ANCIENTS_FOUNTAIN_Y[0]);
        return;
    }
    uint32_t guard = anc_threat(c);
    if (guard < ANC_STRUCTURES) {
        uint32_t foe = anc_nearestHero(c, 0u, e->x, e->y, reach, 0u);
        if (foe) { ctrl[2] = 1.0f; ctrl[3] = 0.0f; }
        else if (anc_lowest(w, anc_lanes(c->k), 0u, e->x, e->y, reach, &low)) { ctrl[2] = 1.0f; ctrl[3] = -1.0f; }
        if (anc_d2(ANCIENTS_STRUCTURE_X[guard], ANCIENTS_STRUCTURE_Y[guard], e->x, e->y) > ANC_R4) anc_chase(p, ANCIENTS_STRUCTURE_X[guard], ANCIENTS_STRUCTURE_Y[guard]);
        return;
    }
    uint32_t lane = anc_contested(c), prey = anc_lastHit(c, 0u, e->x, e->y, wide, ANC_EXACT, e->damage), s = anc_siege(c, 0u, lane, 0u);
    if (s < ANC_STRUCTURES && w->tower[s].target != (uint32_t)ANC_HERO) {
        if (anc_d2(ANCIENTS_STRUCTURE_X[s], ANCIENTS_STRUCTURE_Y[s], e->x, e->y) <= reach) {
            ctrl[2] = 1.0f;
            ctrl[3] = 1.0f;
            anc_boost(c, ctrl);
        } else anc_chase(p, ANCIENTS_STRUCTURE_X[s], ANCIENTS_STRUCTURE_Y[s]);
        return;
    }
    if (prey && anc_locate(w, prey, &tx, &ty) && !anc_danger(c, 0u, tx, ty)) {
        if (anc_d2(tx, ty, e->x, e->y) <= reach) {
            ctrl[2] = 1.0f;
            ctrl[3] = -1.0f;
        } else anc_chase(p, tx, ty);
        return;
    }
    anc_idle(w, 0u, lane, &tx, &ty);
    anc_chase(p, tx, ty);
    if (!anc_danger(c, 0u, e->x, e->y) && anc_lowest(w, anc_lanes(c->k), 0u, e->x, e->y, reach, &low)) {
        ctrl[2] = 1.0f;
        ctrl[3] = -1.0f;
        anc_boost(c, ctrl);
    }
}

static void anc_escape(const ancCtx *c, ancPlan *p) {
    float *ctrl = p->ctrl;
    const ancWorld *w = c->w;
    const ancHero *e = &w->hero[0], *ally = &w->hero[anc_ally(c, 0u)];
    uint32_t threat = anc_enemiesNear(c, 0u, ANC_R12) > 0u;
    anc_chase(p, ANCIENTS_FOUNTAIN_X[0], ANCIENTS_FOUNTAIN_Y[0]);
    switch (e->arch) {
        case ANC_STUNNER: ctrl[3] = 0.0f; ctrl[4] = anc_enemiesNear(c, 0u, ANC_R6) ? 1.0f : 0.0f; break;
        case ANC_TANK: ctrl[5] = threat ? 1.0f : 0.0f; break;
        case ANC_SUPPORT: ctrl[4] = ally->hp < 0.9f * ally->maxHp ? 1.0f : 0.0f; ctrl[5] = threat ? 1.0f : 0.0f; break;
        case ANC_BLINKER: ctrl[4] = threat ? 1.0f : 0.0f; break;
        default: break;
    }
}

static void anc_clear(ancPlan *p) {
    p->x = p->y = 0;
    p->chase = 0u;
    memset(p->ctrl, 0, sizeof p->ctrl);
}

static void anc_plan(const ancCtx *c, uint32_t option, ancPlan *p) {
    anc_clear(p);
    if (c->w->hero[0].dead) return;
    if (option == ANC_ESCAPE) anc_escape(c, p);
    else if (option == ANC_HOLD || !anc_engage(c, p)) {
        anc_clear(p);
        anc_hold(c, p);
    }
}

static void anc_follow(const ancCtx *c, const ancPlan *p, float *ctrl) {
    const ancHero *e = &c->w->hero[0];
    memcpy(ctrl, p->ctrl, sizeof p->ctrl);
    if (!p->chase || e->dead) return;
    float step = (c->tick < e->slowUntil ? 38.4f : 76.8f) / (float)c->k->substeps;
    float dx = (float)(p->x - e->x) - e->rx, dy = (float)(p->y - e->y) - e->ry, d = sqrtf(dx * dx + dy * dy);
    if (!(d > 0.0f)) { ctrl[0] = ctrl[1] = 0.0f; return; }
    float scale = 1.0f / fmaxf(d, step);
    ctrl[0] = dx * scale;
    ctrl[1] = dy * scale;
}

static float anc_value(const ancCtx *c) {
    const ancWorld *w = c->w;
    float value = 1000.0f * (w->tower[ANC_ANCIENT].hp - w->tower[ANC_OWN + ANC_ANCIENT].hp);
    for (uint32_t i = 0u; i < ANC_ANCIENT; i++) value += 100.0f * (w->tower[i].hp - w->tower[ANC_OWN + i].hp);
    for (uint32_t i = 0u; i < c->k->heroes; i++) {
        const ancHero *a = &w->hero[i], *b = &w->hero[ANC_SIDE + i];
        float la = a->maxHp - a->hp, lb = b->maxHp - b->hp;
        value += a->hp - b->hp - 0.5f * (la * la / a->maxHp - lb * lb / b->maxHp) + (float)a->gold - (float)b->gold;
    }
    for (uint32_t l = 0u; l < c->k->lanes; l++) value += anc_total(&w->fifo[l]) - anc_total(&w->fifo[ANC_LANES + l]);
    return value;
}

static float anc_rollout(const ancientsKnobs *k, uint32_t instance, uint32_t seed, uint32_t *extra, const uint32_t *words, uint32_t option) {
    ancOracle *o = anc_oracleOf(extra);
    ancCtx c;
    memcpy(&o->world, anc_world(extra), sizeof o->world);
    memcpy(o->words, words, 6u * sizeof(uint32_t));
    anc_bind(&c, k, instance, &o->world, anc_league(extra), seed, o->words);
    ancPlan plan;
    for (uint32_t n = 0u; n < ANC_HORIZON * k->substeps; n++) {
        float ctrl[OAK_ENV_ACT_MAX];
        if (n == 0u || o->words[1] + 1u == k->substeps) anc_plan(&c, option, &plan);
        anc_follow(&c, &plan, ctrl);
        anc_advance(&c, o->words, ctrl);
        if (o->world.result) return o->world.result == 1u ? 1e9f : -1e9f;
        if (o->words[0] >= k->tickCap) break;
    }
    return anc_value(&c);
}

void ancients_oracle(uint32_t instance, uint32_t seed, uint32_t step, const uint32_t *words, uint32_t *extra, const float *obs, float *ctrl, const float *shared) {
    (void)step; (void)obs; (void)shared;
    const ancientsKnobs *k = &ANCIENTS_KNOBS[instance];
    ancOracle *o = anc_oracleOf(extra);
    ancCtx c;
    if (words[1] == 0u && words[0] % ANC_PERIOD == 0u && !(o->primed && o->game == words[2] && o->tick == words[0])) {
        float best = -INFINITY;
        uint32_t choice = ANC_HOLD;
        for (uint32_t option = 0u; option < ANC_OPTIONS; option++) {
            float value = anc_rollout(k, instance, seed, extra, words, option);
            if (value > best) { best = value; choice = option; }
        }
        o->primed = 1u;
        o->game = words[2];
        o->tick = words[0];
        o->option = choice;
    }
    anc_bind(&c, k, instance, anc_world(extra), anc_league(extra), seed, words);
    if (o->planGame != words[2] || o->planOption != o->option || words[1] + 1u == k->substeps) {
        o->planGame = words[2];
        o->planOption = o->option;
        anc_plan(&c, o->option, &o->plan);
    }
    anc_follow(&c, &o->plan, ctrl);
}

static uint32_t anc_hp16(float hp) { return hp > 0.0f ? (uint32_t)fminf(hp + 0.5f, 65535.0f) : 0u; }

static uint32_t anc_code(const ancWorld *w, uint32_t target) {
    uint32_t kind = (target & ANC_KIND) >> 20, index = target & 0xFFu;
    if (kind == 2u) {
        uint32_t f = (target >> 16) & 15u;
        int k = anc_find(&w->fifo[f], target & 0xFFFFu);
        if (k < 0) return 0u;
        index = f << 4 | (uint32_t)k;
    }
    return kind << 10 | index;
}

uint32_t ancients_view(uint32_t instance, const uint32_t *words, const uint32_t *extra, const float *shared, uint32_t *out) {
    (void)shared;
    const ancientsKnobs *k = &ANCIENTS_KNOBS[instance];
    uint32_t size = 8u * k->heroes + 34u * k->lanes + 14u + (k->jungle ? 12u : 0u) + (k->fog ? 2u * k->heroes : 0u) + 1u;
    if (!out) return size;
    const ancWorld *w = anc_worldAt(extra);
    uint32_t at = 0u, tick = words[0];
    for (uint32_t team = 0u; team < 2u; team++)
        for (uint32_t i = 0u; i < k->heroes; i++) {
            const ancHero *e = &w->hero[team * ANC_SIDE + i];
            uint32_t flags = e->dead | (uint32_t)(tick < e->stunUntil) << 1 | (uint32_t)(tick < e->slowUntil) << 2 | (uint32_t)(tick < e->frenzyUntil) << 3 |
                             (uint32_t)(tick < e->fortifyUntil) << 4 | (uint32_t)(tick < e->wardUntil) << 5 | (uint32_t)(tick < e->tauntUntil) << 6 | (uint32_t)(e->landAt != 0u) << 7;
            out[at++] = (uint32_t)e->x | (uint32_t)e->y << 16;
            out[at++] = anc_hp16(e->hp) | anc_hp16(e->maxHp) << 16;
            out[at++] = anc_hp16(e->mana) | e->level << 16 | e->arch << 21 | flags << 24;
            out[at++] = (e->gold < 65535u ? e->gold : 65535u) | e->mode << 16 | anc_code(w, e->landAt ? e->landTarget : e->target) << 19;
        }
    for (uint32_t side = 0u; side < 2u; side++)
        for (uint32_t l = 0u; l < k->lanes; l++) {
            const ancFifo *F = &w->fifo[side * ANC_LANES + l];
            out[at++] = F->count;
            for (uint32_t i = 0u; i < ANC_FIFO; i++) out[at++] = i < F->count ? ((uint32_t)F->arc[i] & 0xFFFFu) | anc_hp16(F->hp[i]) << 16 : 0u;
        }
    for (uint32_t s = 0u; s < ANC_STRUCTURES; s += 2u) out[at++] = anc_hp16(w->tower[s].hp) | anc_hp16(w->tower[s + 1u].hp) << 16;
    for (uint32_t s = 0u; s < ANC_STRUCTURES; s += 2u) out[at++] = anc_code(w, w->tower[s].target) | anc_code(w, w->tower[s + 1u].target) << 16;
    if (k->jungle)
        for (uint32_t i = 0u; i < ANC_CAMPS; i++) {
            const ancCamp *m = &w->camp[i];
            out[at++] = (uint32_t)m->x | (uint32_t)m->y << 16;
            out[at++] = anc_hp16(m->hp[0]) | anc_hp16(m->hp[1]) << 16;
            out[at++] = anc_hp16(m->hp[2]) | anc_code(w, m->aggro) << 16;
        }
    if (k->fog)
        for (uint32_t i = 0u; i < k->heroes; i++) {
            const ancMemory *m = &w->memory[0];
            uint32_t age = m->hvisible[i] ? 0u : tick + 1u - m->hseen[i];
            out[at++] = (uint32_t)m->hx[i] | (uint32_t)m->hy[i] << 16;
            out[at++] = (uint32_t)(m->hhp[i] * 65535.0f + 0.5f) | (age < 65535u ? age : 65535u) << 16;
        }
    out[at] = w->sweep | w->eventShown << 8 | w->latch << 12 | w->member << 16 | w->eventCount << 20;
    return size;
}
