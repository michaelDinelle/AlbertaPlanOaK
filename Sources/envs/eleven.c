#include "../env.h"
#include <string.h>

typedef struct { uint32_t perSide, drill, offside, slices, ticks, league; float difficulty; } elevenKnobs;

static const elevenKnobs ELEVEN_KNOBS[5] = {
    { 1u, 1u, 0u, 6u, 400u, 0u, 0.6f },
    { 2u, 0u, 0u, 6u, 3000u, 0u, 0.6f },
    { 3u, 0u, 0u, 6u, 3000u, 0u, 0.6f },
    { 11u, 0u, 1u, 6u, 3000u, 0u, 0.95f },
    { 11u, 0u, 1u, 6u, 3000u, 1u, 0.95f }
};

static const float ELEVEN_PITCH[10] = { 52.5f, 34.0f, 3.66f, 2.44f, 16.5f, 20.16f, 5.5f, 9.16f, 11.0f, 9.15f };
static const float ELEVEN_THETA[16] = { -0.6f, 3.0f, 0.0f, 1.2f, 2.0f, 0.4f, 0.8f, 1.0f, 1.2f, 30.0f, 2.0f, 0.7f, 0.12f, 0.0f, 8.0f, 0.5f };
static const float ELEVEN_THETA_LOW[16] = { -2.0f, 0.0f, -2.0f, 0.0f, 0.0f, -2.0f, 0.0f, 0.0f, 0.0f, 10.0f, 1.0f, 0.0f, 0.0f, -8.0f, 0.0f, 0.0f };
static const float ELEVEN_THETA_HIGH[16] = { 2.0f, 6.0f, 2.0f, 3.0f, 5.0f, 2.0f, 3.0f, 3.0f, 3.0f, 50.0f, 2.99f, 0.99f, 0.5f, 8.0f, 16.0f, 1.0f };

enum { PITCH_LENGTH, PITCH_WIDTH, PITCH_POST, PITCH_BAR, PITCH_BOX, PITCH_BOX_WIDTH, PITCH_AREA, PITCH_AREA_WIDTH, PITCH_MARK, PITCH_CIRCLE };
enum { TH_SHOOT, TH_SHOT_GAIN, TH_PASS, TH_PASS_GAIN, TH_PASS_RISK, TH_DRIBBLE, TH_SPACE, TH_PRESSURE, TH_EVADE, TH_PRESS_RANGE, TH_PRESSERS, TH_TACKLE, TH_SLIDE, TH_LINE, TH_PUSH, TH_MARK, ELE_THETA };
enum { ELE_SLOTS = 24, ELE_TEAM = 12, ELE_NONE = 255 };
enum { ELE_NORMAL, ELE_KICKOFF, ELE_GOALKICK, ELE_FREEKICK, ELE_CORNER, ELE_THROWIN, ELE_PENALTY };
enum { ELE_GROUND, ELE_LOFT, ELE_SHOT, ELE_THROW };
enum { ELE_QUIET, ELE_GOAL_FOR, ELE_GOAL_AGAINST, ELE_OUT, ELE_KEPT, ELE_LOST, ELE_DEAD };
enum { TAG_NOISE = 1, TAG_KICK, TAG_TOUCH, TAG_TACKLE, TAG_TOSS, TAG_AUTO };

static const float ELE_DT = 1.0f / 60.0f, ELE_RUN = 7.0f, ELE_SPRINT = 9.0f, ELE_DRIBBLE = 0.85f, ELE_ACCEL = 6.0f, ELE_DIVE_ACCEL = 25.0f, ELE_DIVE_SPEED = 8.0f;
static const float ELE_AIR = 0.99667222f, ELE_ROLL = 0.99004983f, ELE_AIR_RATE = 0.2f, ELE_ROLL_RATE = 0.6f, ELE_GRAVITY = 9.81f, ELE_BOUNCE = 0.55f, ELE_SKID = 0.85f;
static const float ELE_RADIUS = 0.11f, ELE_REACH = 0.6f, ELE_CHEST = 1.5f, ELE_HANDS = 1.1f, ELE_DIVE_REACH = 1.8f, ELE_HANDS_HIGH = 2.6f, ELE_FOOT = 0.4f;
static const float ELE_DRIVE = 3.0f, ELE_STAND = 1.2f, ELE_SLIDE = 2.5f, ELE_SHOT_SPEED = 26.0f, ELE_LOFT_ANGLE = 0.6f, ELE_THROW_ANGLE = 0.35f, ELE_FAR = 1000.0f;
static const float ELE_SPRINT_DRAIN = 1.0f / 1800.0f, ELE_RUN_DRAIN = 1.0f / 18000.0f, ELE_RECOVER = 1.0f / 3600.0f;
static const uint32_t ELE_KICK_LOCK = 18u, ELE_BEATEN = 30u, ELE_MISSED = 36u, ELE_DOWN = 72u, ELE_READY = 60u, ELE_AUTO = 180u, ELE_HOLD = 60u, ELE_DIVE = 60u;

typedef struct { float u[ELE_TEAM], v[ELE_TEAM], follow[ELE_TEAM], lift[ELE_TEAM]; uint8_t mark[ELE_TEAM], striker; } eleShape;

static const eleShape ELE_SHAPES[3] = {
    { { -50.0f, -10.0f }, { 0.0f, 0.0f }, { 0.0f, 0.8f }, { 0.0f, 6.0f }, { 11, 11, 11, 11 }, 1 },
    { { -50.0f, -30.0f, -8.0f }, { 0.0f, 6.0f, -6.0f }, { 0.0f, 0.5f, 0.8f }, { 0.0f, 0.0f, 6.0f }, { 11, 2, 11, 11 }, 2 },
    { { -50.0f, -34.0f, -36.0f, -36.0f, -34.0f, -17.0f, -20.0f, -20.0f, -17.0f, -4.0f, -4.0f },
      { 0.0f, 20.0f, 7.0f, -7.0f, -20.0f, 22.0f, 7.0f, -7.0f, -22.0f, 6.0f, -6.0f },
      { 0.0f, 0.5f, 0.5f, 0.5f, 0.5f, 0.65f, 0.65f, 0.65f, 0.65f, 0.8f, 0.8f },
      { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 6.0f, 6.0f },
      { 11, 8, 10, 9, 5, 4, 7, 6, 1, 11, 11, 11 }, 9 }
};
static const float ELEVEN_KICK_NOISE[4] = { 0.03f, 0.05f, 0.04f, 0.05f };

typedef struct {
    _Alignas(16) float x[ELE_SLOTS], y[ELE_SLOTS], vx[ELE_SLOTS], vy[ELE_SLOTS], wx[ELE_SLOTS], wy[ELE_SLOTS], acc[ELE_SLOTS], stamina[ELE_SLOTS];
    _Alignas(16) uint32_t touch[ELE_SLOTS], busy[ELE_SLOTS], tackle[ELE_SLOTS], noise[32];
    float fx[ELE_SLOTS], fy[ELE_SLOTS];
    float bx, by, bz, bvx, bvy, bvz, sx, sy;
    uint32_t clock, game, owner, held, heldAt, last, live, mode, team, taker, readyAt, autoAt, offside, active, switched, scan, dive[2], near[2][2], close, closeFor;
    uint32_t goals[2], kickoff, half, member, wins, losses, draws;
} eleWorld;

typedef struct { _Alignas(16) float winRate[LEAGUE_MEMBERS]; float theta[LEAGUE_MEMBERS * ELE_THETA]; } eleLeague;

typedef struct { eleWorld world; uint32_t pending, game; } eleOracle;

typedef struct { const elevenKnobs *k; eleWorld *w; const float *theta[2]; uint32_t seed, n, groups, shape, cycle; float sigma; } eleCtx;

static eleWorld *ele_world(uint32_t *extra) { return (eleWorld *)((uintptr_t)(extra + 4) & ~(uintptr_t)15u); }
static const eleWorld *ele_worldAt(const uint32_t *extra) { return (const eleWorld *)((uintptr_t)(extra + 4) & ~(uintptr_t)15u); }
static eleLeague *ele_league(uint32_t *extra) { return (eleLeague *)(ele_world(extra) + 1); }
static eleOracle *ele_oracleOf(uint32_t *extra) { return (eleOracle *)(ele_league(extra) + 1); }

static inline float4 ele_load(const float *p) { float4 v; memcpy(&v, p, sizeof v); return v; }
static inline void ele_store(float *p, float4 v) { memcpy(p, &v, sizeof v); }
static inline lane4 ele_loadu(const uint32_t *p) { lane4 v; memcpy(&v, p, sizeof v); return v; }
static inline uint32_t ele_teamOf(uint32_t p) { return p >= ELE_TEAM; }
static inline float ele_sign(uint32_t team) { return team ? -1.0f : 1.0f; }
static inline float ele_unit(float v) { return v > -1.0f ? (v < 1.0f ? v : 1.0f) : v <= -1.0f ? -1.0f : 0.0f; }
static inline float ele_byte(uint32_t word, uint32_t k) { return ((float)(word >> (8u * k) & 255u) + 0.5f) * (1.0f / 256.0f); }
static inline float ele_jitter(uint32_t word, uint32_t k) { return (ele_byte(word, k) - 0.5f) * 3.4641016f; }
static inline float ele_clamp(float v, float lo, float hi) { return fminf(fmaxf(v, lo), hi); }
static inline float ele_dist(float ax, float ay, float bx, float by) { return sqrtf((ax - bx) * (ax - bx) + (ay - by) * (ay - by)); }
static inline uint32_t ele_n(const eleCtx *c) { return c->n; }
static inline uint32_t ele_groups(const eleCtx *c) { return c->groups; }
static inline float ele_sigma(const eleCtx *c) { return c->sigma; }
static inline uint32_t ele_shape(const eleCtx *c) { return c->shape; }
static inline int ele_hasKeeper(const eleCtx *c, uint32_t team) { return !(c->k->drill && team == 0u); }
static inline int ele_keeper(const eleCtx *c, uint32_t p) { return p % ELE_TEAM == 0u && ele_hasKeeper(c, ele_teamOf(p)); }
static inline float4 ele_d2(const eleWorld *w, uint32_t base, float px, float py) {
    float4 dx = ele_load(w->x + base) - px, dy = ele_load(w->y + base) - py;
    return dx * dx + dy * dy;
}

static inline int ele_inBox(uint32_t team, float x, float y) {
    return ele_sign(team) * x <= ELEVEN_PITCH[PITCH_BOX] - ELEVEN_PITCH[PITCH_LENGTH] && fabsf(y) <= ELEVEN_PITCH[PITCH_BOX_WIDTH];
}

static inline __attribute__((always_inline)) void ele_bind(eleCtx *c, const elevenKnobs *k, eleWorld *w, const eleLeague *g, uint32_t seed) {
    c->k = k;
    c->w = w;
    c->seed = seed;
    c->n = k->perSide;
    c->groups = (k->perSide + 3u) / 4u;
    c->shape = k->perSide == 2u ? 0u : k->perSide == 3u ? 1u : 2u;
    c->cycle = k->slices * (uint32_t)(8.5f - 6.0f * k->difficulty);
    c->sigma = 0.3f * (1.0f - k->difficulty);
    c->theta[0] = ELEVEN_THETA;
    c->theta[1] = k->league ? g->theta + w->member * ELE_THETA : ELEVEN_THETA;
}

static u4 ele_draw(const eleCtx *c, uint32_t tag, uint32_t p) { return philox4x32(c->seed, ENV_RNG_STREAM, c->w->game, c->w->clock, tag, p); }

static inline uint32_t ele_nearest(const eleCtx *c, uint32_t team, float px, float py, int field, float *d2) {
    const eleWorld *w = c->w;
    float4 best = splat4(INFINITY);
    lane4 who = (lane4)ELE_NONE, now = (lane4)w->clock;
    for (uint32_t g = 0u; g < ele_groups(c); g++) {
        uint32_t base = team * ELE_TEAM + 4u * g;
        float4 e = select4((lane4)(4u * g + (lane4){ 0u, 1u, 2u, 3u } >= ele_n(c)), splat4(INFINITY), ele_d2(w, base, px, py));
        if (field) {
            if (g == 0u && ele_hasKeeper(c, team)) e.x = INFINITY;
            e = select4((lane4)(ele_loadu(w->busy + base) > now), splat4(INFINITY), e);
        }
        lane4 closer = (lane4)(e < best);
        best = select4(closer, e, best);
        who = (closer & (base + (lane4){ 0u, 1u, 2u, 3u })) | (~closer & who);
    }
    float low = __builtin_reduce_min(best);
    *d2 = low;
    lane4 hit = (lane4)(best == low);
    return __builtin_reduce_min((hit & who) | ~hit);
}

static float ele_offsideLine(const eleCtx *c, uint32_t team) {
    const eleWorld *w = c->w;
    float s = ele_sign(team);
    float4 first = splat4(-INFINITY), second = splat4(-INFINITY);
    for (uint32_t g = 0u; g < ele_groups(c); g++) {
        float4 u = s * ele_load(w->x + (1u - team) * ELE_TEAM + 4u * g);
        second = __builtin_elementwise_max(second, __builtin_elementwise_min(first, u));
        first = __builtin_elementwise_max(first, u);
    }
    float top = __builtin_reduce_max(first);
    lane4 hit = (lane4)(first == top);
    float runner = __builtin_reduce_add(hit & 1u) > 1u ? top : fmaxf(__builtin_reduce_max(select4(hit, splat4(-INFINITY), first)), __builtin_reduce_max(second));
    return fmaxf(runner, 0.0f);
}

static inline __attribute__((always_inline)) void ele_pair(eleCtx *c, uint32_t team) {
    eleWorld *w = c->w;
    float4 e[3];
    lane4 at[3], who = (lane4)ELE_NONE, now = (lane4)w->clock;
    float4 best = splat4(INFINITY);
    for (uint32_t g = 0u; g < ele_groups(c); g++) {
        uint32_t base = team * ELE_TEAM + 4u * g;
        e[g] = select4((lane4)(ele_loadu(w->busy + base) > now) | (lane4)(4u * g + (lane4){ 0u, 1u, 2u, 3u } >= ele_n(c)), splat4(INFINITY), ele_d2(w, base, w->bx, w->by));
        if (g == 0u && ele_hasKeeper(c, team)) e[g].x = INFINITY;
        at[g] = base + (lane4){ 0u, 1u, 2u, 3u };
        lane4 closer = (lane4)(e[g] < best);
        best = select4(closer, e[g], best);
        who = (closer & at[g]) | (~closer & who);
    }
    float low = __builtin_reduce_min(best);
    lane4 hit = (lane4)(best == low);
    uint32_t first = __builtin_reduce_min((hit & who) | ~hit);
    best = splat4(INFINITY);
    who = (lane4)ELE_NONE;
    for (uint32_t g = 0u; g < ele_groups(c); g++) {
        lane4 closer = (lane4)(e[g] < best) & (lane4)(at[g] != first);
        best = select4(closer, e[g], best);
        who = (closer & at[g]) | (~closer & who);
    }
    low = __builtin_reduce_min(best);
    hit = (lane4)(best == low);
    w->near[team][0] = first;
    w->near[team][1] = __builtin_reduce_min((hit & who) | ~hit);
}

static void ele_face(eleWorld *w, uint32_t p, float dx, float dy) {
    float d2 = dx * dx + dy * dy;
    if (!(d2 > 1e-6f)) return;
    float r = 1.0f / sqrtf(d2);
    w->fx[p] = dx * r;
    w->fy[p] = dy * r;
}

static void ele_switch(eleWorld *w, uint32_t p) {
    if (p == w->active) return;
    w->active = p;
    w->switched = 1u;
}

static void ele_restartAt(eleCtx *c, uint32_t mode, uint32_t team, float sx, float sy);

static void ele_gain(eleCtx *c, uint32_t p) {
    eleWorld *w = c->w;
    w->owner = p;
    w->held = 0u;
    w->last = p;
    ele_face(w, p, w->bx - w->x[p], w->by - w->y[p]);
    if (ele_teamOf(p) == 0u) ele_switch(w, p);
}

static void ele_flags(eleCtx *c, uint32_t kicker) {
    eleWorld *w = c->w;
    uint32_t team = ele_teamOf(kicker);
    float s = ele_sign(team), line = ele_offsideLine(c, team), ball = s * w->bx;
    w->offside = 0u;
    for (uint32_t i = 0u; i < ele_n(c); i++) {
        uint32_t q = team * ELE_TEAM + i;
        float u = s * w->x[q];
        if (q != kicker && u > 0.0f && u > ball && u > line) w->offside |= 1u << q;
    }
}

static float ele_loftTime(float reach, float slope) {
    float c = ELE_AIR_RATE, target = reach * slope, t = sqrtf(2.0f * target / ELE_GRAVITY);
    for (uint32_t i = 0u; i < 4u; i++) {
        float e = oak_exp(-c * t);
        float f = (ELE_GRAVITY / c) * (t - (1.0f - e) / c) - target, df = (ELE_GRAVITY / c) * (1.0f - e);
        t -= f / fmaxf(df, 1e-3f);
    }
    return fmaxf(t, 0.05f);
}

static __attribute__((noinline)) void ele_kick(eleCtx *c, uint32_t p, float tx, float ty, uint32_t kind, float tz) {
    eleWorld *w = c->w;
    uint32_t restart = w->mode;
    float dx = tx - w->bx, dy = ty - w->by, reach = sqrtf(dx * dx + dy * dy), ux, uy;
    if (reach > 0.5f) { ux = dx / reach; uy = dy / reach; }
    else { ux = w->fx[p]; uy = w->fy[p]; reach = 0.5f; }
    float speed, vz = 0.0f, z0 = kind == ELE_THROW ? 1.8f : w->held ? 1.0f : 0.0f;
    if (kind == ELE_GROUND) speed = fminf(4.0f + ELE_ROLL_RATE * reach, 26.0f);
    else if (kind == ELE_SHOT) {
        speed = ELE_SHOT_SPEED;
        float span = fminf(ELE_AIR_RATE * reach / speed, 0.9f), t = -oak_log(1.0f - span) / ELE_AIR_RATE;
        vz = fminf((tz - z0 + ELE_GRAVITY * t / ELE_AIR_RATE) * ELE_AIR_RATE / fmaxf(span, 1e-3f) - ELE_GRAVITY / ELE_AIR_RATE, ELE_DRIVE);
    } else {
        float angle = kind == ELE_THROW ? ELE_THROW_ANGLE : ELE_LOFT_ANGLE, slope = oak_sin(angle) / oak_cos(angle);
        float t = ele_loftTime(reach, slope);
        speed = fminf(ELE_AIR_RATE * reach / (1.0f - oak_exp(-ELE_AIR_RATE * t)), kind == ELE_THROW ? 15.0f : 32.0f);
        vz = speed * slope;
    }
    u4 r = ele_draw(c, TAG_KICK, p);
    float4 n = normalFrom4(r, r);
    float sigma = ELEVEN_KICK_NOISE[kind] * (1.5f - 0.5f * w->stamina[p]), sn, cs;
    oak_sincos(sigma * n.x, &sn, &cs);
    float scale = 1.0f + 0.04f * n.y;
    w->bvx = (ux * cs - uy * sn) * speed * scale;
    w->bvy = (ux * sn + uy * cs) * speed * scale;
    w->bvz = vz * scale;
    w->bz = z0;
    w->owner = ELE_NONE;
    w->held = 0u;
    w->live = 1u;
    w->mode = ELE_NORMAL;
    w->last = p;
    w->touch[p] = w->clock + ELE_KICK_LOCK;
    w->offside = 0u;
    if (c->k->offside && kind != ELE_THROW && restart != ELE_CORNER && restart != ELE_GOALKICK) ele_flags(c, p);
}

static void ele_foul(eleCtx *c, uint32_t victim, float x, float y) {
    uint32_t foe = 1u - ele_teamOf(victim);
    if (ele_inBox(foe, x, y)) ele_restartAt(c, ELE_PENALTY, ele_teamOf(victim), -ele_sign(foe) * (ELEVEN_PITCH[PITCH_LENGTH] - ELEVEN_PITCH[PITCH_MARK]), 0.0f);
    else ele_restartAt(c, ELE_FREEKICK, ele_teamOf(victim), x, y);
}

static float ele_front(const eleWorld *w, uint32_t o, uint32_t p) {
    float dx = w->x[o] - w->x[p], dy = w->y[o] - w->y[p], d = sqrtf(dx * dx + dy * dy);
    return fmaxf(0.0f, -(w->fx[o] * dx + w->fy[o] * dy) / fmaxf(d, 0.1f));
}

static __attribute__((noinline)) void ele_tackle(eleCtx *c, uint32_t p, uint32_t slide) {
    eleWorld *w = c->w;
    uint32_t o = w->owner;
    if (o == ELE_NONE || w->held || ele_teamOf(o) == ele_teamOf(p) || w->clock < w->tackle[p] || w->mode != ELE_NORMAL) return;
    float dx = w->x[o] - w->x[p], dy = w->y[o] - w->y[p], d = sqrtf(dx * dx + dy * dy);
    if (d > (slide ? ELE_SLIDE : ELE_STAND)) return;
    float inv = 1.0f / fmaxf(d, 0.1f), front = ele_front(w, o, p);
    u4 r = ele_draw(c, TAG_TACKLE, p);
    if (slide) {
        w->vx[p] = dx * inv * 7.0f;
        w->vy[p] = dy * inv * 7.0f;
        w->wx[p] = w->wy[p] = 0.0f;
        w->busy[p] = w->tackle[p] = w->touch[p] = w->clock + ELE_DOWN;
        if (r.x < u01Bound(0.5f + 0.35f * front)) {
            w->owner = ELE_NONE;
            w->bvx = dx * inv * 6.0f + 0.3f * w->vx[o];
            w->bvy = dy * inv * 6.0f + 0.3f * w->vy[o];
            w->bvz = 0.0f;
            w->last = p;
            w->touch[o] = w->clock + ELE_BEATEN;
            w->offside = 0u;
        } else if (r.y < u01Bound(0.6f)) ele_foul(c, o, w->x[o], w->y[o]);
    } else if (r.x < u01Bound(0.35f + 0.4f * front)) {
        w->touch[o] = w->clock + ELE_BEATEN;
        w->offside = 0u;
        ele_gain(c, p);
    } else w->tackle[p] = w->clock + ELE_MISSED;
}

static void ele_place(eleWorld *w, uint32_t p, float x, float y) {
    w->x[p] = ele_clamp(x, -ELEVEN_PITCH[PITCH_LENGTH] - 3.0f, ELEVEN_PITCH[PITCH_LENGTH] + 3.0f);
    w->y[p] = ele_clamp(y, -ELEVEN_PITCH[PITCH_WIDTH] - 3.0f, ELEVEN_PITCH[PITCH_WIDTH] + 3.0f);
    w->vx[p] = w->vy[p] = w->wx[p] = w->wy[p] = 0.0f;
}

static void ele_clear(eleCtx *c, uint32_t team, float cx, float cy, float radius) {
    eleWorld *w = c->w;
    for (uint32_t i = 0u; i < ele_n(c); i++) {
        uint32_t q = team * ELE_TEAM + i;
        float dx = w->x[q] - cx, dy = w->y[q] - cy, d = sqrtf(dx * dx + dy * dy);
        if (d >= radius) continue;
        if (d < 1e-3f) { dx = ele_sign(1u - team); dy = 0.0f; d = 1.0f; }
        ele_place(w, q, cx + dx * radius / d, cy + dy * radius / d);
    }
}

static void ele_evict(eleCtx *c, uint32_t team, uint32_t box, float edge, uint32_t spare) {
    eleWorld *w = c->w;
    float s = ele_sign(box);
    for (uint32_t i = 0u; i < ele_n(c); i++) {
        uint32_t q = team * ELE_TEAM + i;
        if (q == spare || q == w->taker) continue;
        if (s * w->x[q] < edge) ele_place(w, q, s * edge, w->y[q]);
    }
}

static uint32_t ele_closest(const eleCtx *c, uint32_t team, float x, float y) {
    float d2;
    uint32_t q = ele_nearest(c, team, x, y, 1, &d2);
    return q == ELE_NONE ? team * ELE_TEAM + (ele_hasKeeper(c, team) ? 1u : 0u) : q;
}

static __attribute__((noinline)) void ele_restartAt(eleCtx *c, uint32_t mode, uint32_t team, float sx, float sy) {
    eleWorld *w = c->w;
    float s = ele_sign(team), length = ELEVEN_PITCH[PITCH_LENGTH];
    uint32_t foe = 1u - team;
    w->mode = mode;
    w->team = team;
    w->sx = sx;
    w->sy = sy;
    w->live = 0u;
    w->held = 0u;
    w->offside = 0u;
    w->dive[0] = w->dive[1] = 0u;
    w->taker = mode == ELE_GOALKICK ? team * ELE_TEAM : mode == ELE_PENALTY ? team * ELE_TEAM + ELE_SHAPES[ele_shape(c)].striker : ele_closest(c, team, sx, sy);
    w->owner = w->taker;
    w->bx = sx;
    w->by = sy;
    w->bz = 0.0f;
    w->bvx = w->bvy = w->bvz = 0.0f;
    w->busy[w->taker] = 0u;
    ele_place(w, w->taker, mode == ELE_THROWIN ? sx : sx - s * 0.5f, sy);
    ele_face(w, w->taker, s, 0.0f);
    if (mode == ELE_THROWIN) ele_clear(c, foe, sx, sy, 2.0f);
    else if (mode == ELE_GOALKICK) ele_evict(c, foe, team, ELEVEN_PITCH[PITCH_BOX] - length + 1.0f, ELE_NONE);
    else if (mode == ELE_PENALTY) {
        uint32_t keeper = foe * ELE_TEAM;
        ele_evict(c, team, foe, ELEVEN_PITCH[PITCH_BOX] - length + 4.5f, ELE_NONE);
        ele_evict(c, foe, foe, ELEVEN_PITCH[PITCH_BOX] - length + 4.5f, keeper);
        ele_place(w, keeper, -ele_sign(foe) * (length - 0.3f), 0.0f);
    } else ele_clear(c, foe, sx, sy, ELEVEN_PITCH[PITCH_CIRCLE]);
    w->readyAt = w->clock + (mode == ELE_PENALTY ? 2u * ELE_READY : ELE_READY);
    w->autoAt = w->readyAt + ELE_AUTO;
    if (team == 0u) ele_switch(w, w->taker);
}

static __attribute__((noinline)) void ele_kickoff(eleCtx *c, uint32_t team) {
    eleWorld *w = c->w;
    const eleShape *shape = &ELE_SHAPES[ele_shape(c)];
    for (uint32_t t = 0u; t < 2u; t++) {
        float s = ele_sign(t);
        for (uint32_t i = 0u; i < ele_n(c); i++) {
            uint32_t q = t * ELE_TEAM + i;
            ele_place(w, q, s * shape->u[i], s * shape->v[i]);
            w->touch[q] = w->busy[q] = w->tackle[q] = 0u;
            w->acc[q] = ELE_ACCEL;
            ele_face(w, q, s, 0.0f);
        }
    }
    uint32_t striker = team * ELE_TEAM + shape->striker;
    ele_place(w, striker, -ele_sign(team) * 0.5f, 0.0f);
    w->mode = ELE_KICKOFF;
    w->team = team;
    w->sx = w->sy = 0.0f;
    w->live = 0u;
    w->held = 0u;
    w->offside = 0u;
    w->dive[0] = w->dive[1] = 0u;
    w->taker = w->owner = striker;
    w->bx = w->by = w->bz = 0.0f;
    w->bvx = w->bvy = w->bvz = 0.0f;
    ele_clear(c, 1u - team, 0.0f, 0.0f, ELEVEN_PITCH[PITCH_CIRCLE] + 0.35f);
    w->readyAt = w->clock + ELE_READY;
    w->autoAt = w->readyAt + ELE_AUTO;
    ele_switch(w, team == 0u ? striker : ele_closest(c, 0u, 0.0f, 0.0f));
}

static void ele_drill(eleCtx *c) {
    eleWorld *w = c->w;
    ele_place(w, 0u, 0.0f, 0.0f);
    ele_place(w, ELE_TEAM, ELEVEN_PITCH[PITCH_LENGTH] - 0.5f, 0.0f);
    ele_face(w, 0u, 1.0f, 0.0f);
    ele_face(w, ELE_TEAM, -1.0f, 0.0f);
    w->mode = ELE_NORMAL;
    w->live = 1u;
    w->owner = w->last = w->active = 0u;
    w->bx = ELE_FOOT;
}

static __attribute__((noinline)) void ele_begin(eleCtx *c, uint32_t game) {
    eleWorld *w = c->w;
    w->game = game;
    w->clock = 0u;
    w->goals[0] = w->goals[1] = 0u;
    w->half = w->switched = w->scan = w->held = w->offside = w->dive[0] = w->dive[1] = 0u;
    w->by = w->bz = w->bvx = w->bvy = w->bvz = 0.0f;
    for (uint32_t q = 0u; q < ELE_SLOTS; q++) {
        w->x[q] = ele_sign(ele_teamOf(q)) * ELE_FAR;
        w->y[q] = 0.0f;
        w->vx[q] = w->vy[q] = w->wx[q] = w->wy[q] = 0.0f;
        w->acc[q] = ELE_ACCEL;
        w->stamina[q] = 1.0f;
        w->touch[q] = w->busy[q] = w->tackle[q] = 0u;
        w->fx[q] = ele_sign(ele_teamOf(q));
        w->fy[q] = 0.0f;
    }
    if (c->k->drill) { ele_drill(c); return; }
    w->kickoff = philox4x32(c->seed, ENV_RNG_STREAM, game, 0u, TAG_TOSS, 0u).x >> 31;
    w->active = 1u;
    ele_kickoff(c, w->kickoff);
}

static inline __attribute__((always_inline)) void ele_tick(eleCtx *c) {
    eleWorld *w = c->w;
    uint32_t tick = w->clock / c->k->slices;
    if (tick % (c->cycle / c->k->slices) == 0u)
        for (uint32_t b = 0u; b < (ele_groups(c) + 1u) / 2u; b++) {
            u44 r = philox4(c->seed, ENV_RNG_STREAM, (lane4)w->game, (lane4)tick, (lane4){ 0u, 1u, 2u, 3u } + 4u * b, (lane4)TAG_NOISE);
            memcpy(w->noise + 16u * b, &r, sizeof r);
        }
    if (!c->k->drill) {
        uint32_t holder = w->owner == ELE_NONE ? 2u : ele_teamOf(w->owner);
        if (holder != 0u) ele_pair(c, 0u);
        if (holder != 1u) ele_pair(c, 1u);
        uint32_t a = w->active, q = w->near[0][0];
        if ((w->owner == ELE_NONE || ele_teamOf(w->owner) == 1u) && q != ELE_NONE && q != a) {
            float dq = (w->x[q] - w->bx) * (w->x[q] - w->bx) + (w->y[q] - w->by) * (w->y[q] - w->by);
            float da = (w->x[a] - w->bx) * (w->x[a] - w->bx) + (w->y[a] - w->by) * (w->y[a] - w->by);
            if (sqrtf(dq) + 1.5f < sqrtf(da) || ele_keeper(c, a)) ele_switch(w, q);
        }
    }
    float d2;
    w->closeFor = w->active;
    w->close = ele_nearest(c, 1u, w->x[w->active], w->y[w->active], 0, &d2);
}

static void ele_steer(eleCtx *c, uint32_t p, float tx, float ty, float top) {
    eleWorld *w = c->w;
    tx = ele_clamp(tx, -ELEVEN_PITCH[PITCH_LENGTH] - 1.0f, ELEVEN_PITCH[PITCH_LENGTH] + 1.0f);
    ty = ele_clamp(ty, -ELEVEN_PITCH[PITCH_WIDTH] - 1.0f, ELEVEN_PITCH[PITCH_WIDTH] + 1.0f);
    float dx = tx - w->x[p], dy = ty - w->y[p], d = sqrtf(dx * dx + dy * dy);
    float speed = fminf(top * (0.7f + 0.3f * w->stamina[p]) * (w->owner == p ? ELE_DRIBBLE : 1.0f), 2.0f * d);
    float scale = d > 1e-3f ? speed / d : 0.0f;
    w->wx[p] = dx * scale;
    w->wy[p] = dy * scale;
}

static void ele_lanes(const eleCtx *c, uint32_t foe, float ox, float oy, float dx, float dy, float length, float pace, float *margin, float *landing) {
    const eleWorld *w = c->w;
    float4 low = splat4(INFINITY), near = splat4(INFINITY);
    float inv = 1.0f / fmaxf(length * length, 1e-3f), run = 7.5f * length / pace;
    for (uint32_t g = 0u; g < ele_groups(c); g++) {
        uint32_t base = foe * ELE_TEAM + 4u * g;
        float4 px = ele_load(w->x + base) - ox, py = ele_load(w->y + base) - oy;
        float4 t = __builtin_elementwise_min(__builtin_elementwise_max((px * dx + py * dy) * inv, splat4(0.0f)), splat4(1.0f));
        float4 ex = px - t * dx, ey = py - t * dy;
        low = __builtin_elementwise_min(low, __builtin_elementwise_sqrt(ex * ex + ey * ey) - t * run);
        float4 rx = px - dx, ry = py - dy;
        near = __builtin_elementwise_min(near, rx * rx + ry * ry);
    }
    *margin = __builtin_reduce_min(low) - 0.6f;
    *landing = sqrtf(__builtin_reduce_min(near));
}

static __attribute__((noinline)) float ele_bestPass(const eleCtx *c, uint32_t p, float limit, uint32_t *to, float *tx, float *ty, uint32_t *kind) {
    const eleWorld *w = c->w;
    uint32_t team = ele_teamOf(p);
    const float *th = c->theta[team];
    float s = ele_sign(team), best = -INFINITY;
    float4 score[3];
    for (uint32_t g = 0u; g < ele_groups(c); g++) {
        uint32_t base = team * ELE_TEAM + 4u * g;
        float4 d = __builtin_elementwise_sqrt(ele_d2(w, base, w->bx, w->by)), v = s * (ele_load(w->x + base) - w->x[p]) - 0.25f * __builtin_elementwise_abs(d - 18.0f);
        lane4 bad = (lane4)(d < 4.0f) | (lane4)(d > limit) | (lane4)((base + (lane4){ 0u, 1u, 2u, 3u }) == p);
        if (g == 0u && ele_hasKeeper(c, team)) bad.x = ~0u;
        score[g] = select4(bad, splat4(-INFINITY), v);
    }
    *to = ELE_NONE;
    for (uint32_t k = 0u; k < 1u; k++) {
        float top = -INFINITY;
        for (uint32_t g = 0u; g < ele_groups(c); g++) top = fmaxf(top, __builtin_reduce_max(score[g]));
        if (!(top > -INFINITY)) break;
        uint32_t q = ELE_NONE;
        for (uint32_t g = 0u; g < ele_groups(c) && q == ELE_NONE; g++)
            for (uint32_t i = 0u; i < 4u; i++)
                if (score[g][i] == top) { q = team * ELE_TEAM + 4u * g + i; score[g][i] = -INFINITY; break; }
        float lead = ele_dist(w->x[q], w->y[q], w->bx, w->by) / 15.0f;
        float rx = w->x[q] + w->vx[q] * lead, ry = w->y[q] + w->vy[q] * lead, dx = rx - w->bx, dy = ry - w->by, length = sqrtf(dx * dx + dy * dy);
        float margin, landing;
        ele_lanes(c, 1u - team, w->bx, w->by, dx, dy, length, 0.5f * (8.0f + ELE_ROLL_RATE * length), &margin, &landing);
        float ground = ele_clamp(0.5f - margin / 4.0f, 0.0f, 1.0f), air = ele_clamp(0.9f - landing / 5.0f, 0.0f, 1.0f) + (length < 12.0f ? 1.0f : 0.0f);
        float risk = fminf(ground, air), value = th[TH_PASS_GAIN] * s * (rx - w->x[p]) / 20.0f - th[TH_PASS_RISK] * risk;
        if (value > best) {
            best = value;
            *to = q;
            *tx = rx;
            *ty = ry;
            *kind = ground <= air ? ELE_GROUND : ELE_LOFT;
        }
    }
    return best;
}

static void ele_shoot(eleCtx *c, uint32_t p) {
    eleWorld *w = c->w;
    uint32_t team = ele_teamOf(p), keeper = (1u - team) * ELE_TEAM;
    float s = ele_sign(team), aim = w->y[keeper] * s > 0.0f ? -2.9f : 2.9f;
    ele_kick(c, p, s * (ELEVEN_PITCH[PITCH_LENGTH] + 0.5f), s * aim, ELE_SHOT, 0.4f);
}

static __attribute__((noinline)) void ele_carrier(eleCtx *c, uint32_t p, uint32_t word) {
    eleWorld *w = c->w;
    uint32_t team = ele_teamOf(p);
    const float *th = c->theta[team];
    float s = ele_sign(team), u = s * w->x[p], v = s * w->y[p], goal = ELEVEN_PITCH[PITCH_LENGTH], sigma = ele_sigma(c);
    float gu = goal - u, reach2 = gu * gu + v * v, quality = 7.32f * gu / (reach2 + 1.0f), d2;
    uint32_t o = ele_nearest(c, 1u - team, w->x[p], w->y[p], 0, &d2), to, kind;
    float dn = sqrtf(d2), tx = 0.0f, ty = 0.0f;
    float shoot = reach2 < 35.0f * 35.0f ? th[TH_SHOOT] + th[TH_SHOT_GAIN] * quality + sigma * ele_jitter(word, 0u) : -INFINITY;
    float pass = th[TH_PASS] + ele_bestPass(c, p, 45.0f, &to, &tx, &ty, &kind) + sigma * ele_jitter(word, 1u);
    float dribble = th[TH_DRIBBLE] + th[TH_SPACE] * fminf(dn, 10.0f) * 0.1f - th[TH_PRESSURE] * (dn < 2.5f ? 1.0f : 0.0f) + sigma * ele_jitter(word, 2u);
    if (shoot >= pass && shoot >= dribble) { ele_shoot(c, p); return; }
    if (to != ELE_NONE && pass >= dribble) { ele_kick(c, p, tx, ty, kind, 0.0f); return; }
    float r = 1.0f / sqrtf(reach2 + 1e-6f), du = gu * r, dv = -v * r;
    if (o != ELE_NONE && dn < 6.0f) {
        float push = th[TH_EVADE] * (1.0f - dn / 6.0f) / fmaxf(dn, 0.1f);
        du += push * s * (w->x[p] - w->x[o]);
        dv += push * s * (w->y[p] - w->y[o]);
    }
    if (fabsf(v) > 29.0f) dv -= v > 0.0f ? 0.5f : -0.5f;
    float turn = sigma * ele_jitter(word, 3u);
    ele_steer(c, p, w->x[p] + s * 10.0f * (du - turn * dv), w->y[p] + s * 10.0f * (dv + turn * du), dn < 8.0f ? ELE_SPRINT : ELE_RUN);
}

static void ele_press(eleCtx *c, uint32_t p, uint32_t word, float db) {
    eleWorld *w = c->w;
    uint32_t team = ele_teamOf(p), o = w->owner;
    const float *th = c->theta[team];
    float tx, ty, sigma = ele_sigma(c);
    if (o != ELE_NONE) {
        float lead = db / 9.0f, d = ele_dist(w->x[p], w->y[p], w->x[o], w->y[o]);
        tx = w->x[o] + w->vx[o] * lead;
        ty = w->y[o] + w->vy[o] * lead;
        if (!w->held && w->clock >= w->tackle[p]) {
            if (d <= ELE_STAND && ele_byte(word, 2u) < th[TH_TACKLE]) { ele_tackle(c, p, 0u); return; }
            if (d <= ELE_SLIDE && ele_byte(word, 3u) < th[TH_SLIDE] && !ele_inBox(team, w->x[o], w->y[o])) { ele_tackle(c, p, 1u); return; }
        }
    } else {
        float lead = fminf(db / 8.0f, 1.5f), fade = lead * (1.0f - 0.3f * lead * ELE_ROLL_RATE);
        tx = w->bx + w->bvx * fade;
        ty = w->by + w->bvy * fade;
    }
    ele_steer(c, p, tx + sigma * 10.0f * ele_jitter(word, 0u), ty + sigma * 10.0f * ele_jitter(word, 1u), ELE_SPRINT);
}

static __attribute__((noinline)) void ele_goalie(eleCtx *c, uint32_t p, uint32_t word) {
    eleWorld *w = c->w;
    uint32_t team = ele_teamOf(p);
    const float *th = c->theta[team];
    float s = ele_sign(team), u = s * w->x[p], bu = s * w->bx, bv = s * w->by, vu = s * w->bvx, vv = s * w->bvy, goal = -ELEVEN_PITCH[PITCH_LENGTH], sigma = ele_sigma(c);
    w->acc[p] = ELE_ACCEL;
    if (w->owner == p) {
        w->wx[p] = w->wy[p] = 0.0f;
        if (w->held && w->clock < w->heldAt + ELE_HOLD) return;
        uint32_t to, kind;
        float tx, ty;
        if (ele_bestPass(c, p, 60.0f, &to, &tx, &ty, &kind) > -1.0f && to != ELE_NONE) ele_kick(c, p, tx, ty, w->held && kind == ELE_GROUND ? ELE_THROW : kind, 0.0f);
        else ele_kick(c, p, s * 5.0f, 12.0f * ele_jitter(word, 3u), ELE_LOFT, 0.0f);
        return;
    }
    if (w->mode == ELE_NORMAL && w->owner == ELE_NONE && vu < -3.0f && bu > u - 1.0f) {
        float t = (bu - u) / -vu, cross = bv + vv * t;
        if (t < 2.0f && fabsf(cross) < ELEVEN_PITCH[PITCH_POST] + 2.5f) {
            w->dive[team] = w->clock + ELE_DIVE;
            w->acc[p] = ELE_DIVE_ACCEL;
            float ty = s * ele_clamp(cross, -ELEVEN_PITCH[PITCH_POST] - 1.0f, ELEVEN_PITCH[PITCH_POST] + 1.0f);
            w->wx[p] = 0.0f;
            w->wy[p] = ele_clamp(6.0f * (ty - w->y[p]), -ELE_DIVE_SPEED, ELE_DIVE_SPEED);
            return;
        }
    }
    w->dive[team] = 0u;
    if (w->mode == ELE_NORMAL && ele_inBox(team, w->bx, w->by)) {
        float d2, mine = ele_dist(w->x[p], w->y[p], w->bx, w->by);
        uint32_t rival = ele_nearest(c, 1u - team, w->bx, w->by, 0, &d2);
        if (w->owner == ELE_NONE && (rival == ELE_NONE || mine < sqrtf(d2) + 1.0f)) {
            ele_steer(c, p, w->bx + w->bvx * 0.2f, w->by + w->bvy * 0.2f, ELE_SPRINT);
            return;
        }
        if (w->owner != ELE_NONE && ele_teamOf(w->owner) != team && mine < 5.0f) {
            if (mine <= ELE_STAND && w->clock >= w->tackle[p] && ele_byte(word, 2u) < th[TH_TACKLE]) { ele_tackle(c, p, 0u); return; }
            ele_steer(c, p, w->bx, w->by, ELE_SPRINT);
            return;
        }
    }
    float du = bu - goal, db = sqrtf(du * du + bv * bv) + 1e-3f, depth = ele_clamp(1.0f + 0.1f * db, 1.0f, 5.5f);
    float tu = goal + du / db * depth, tv = ele_clamp(bv / db * depth, -3.0f, 3.0f);
    ele_steer(c, p, s * tu + sigma * 2.0f * ele_jitter(word, 0u), s * tv + sigma * 2.0f * ele_jitter(word, 1u), ELE_RUN);
}

static __attribute__((noinline)) void ele_take(eleCtx *c, uint32_t p, uint32_t word) {
    eleWorld *w = c->w;
    uint32_t team = ele_teamOf(p), to, kind;
    float s = ele_sign(team), tx = 0.0f, ty = 0.0f, length = ELEVEN_PITCH[PITCH_LENGTH];
    if (w->mode == ELE_PENALTY) { ele_kick(c, p, s * (length + 0.5f), word >> 31 ? 2.9f : -2.9f, ELE_SHOT, 0.5f); return; }
    if (w->mode == ELE_CORNER) { ele_kick(c, p, s * (length - 9.0f), 3.0f * ele_jitter(word, 3u), ELE_LOFT, 0.0f); return; }
    if (w->mode == ELE_FREEKICK) {
        float gu = length - s * w->bx;
        if (gu * gu + w->by * w->by < 28.0f * 28.0f) { ele_shoot(c, p); return; }
    }
    float limit = w->mode == ELE_THROWIN ? 25.0f : w->mode == ELE_KICKOFF ? 20.0f : 50.0f;
    ele_bestPass(c, p, limit, &to, &tx, &ty, &kind);
    if (to == ELE_NONE) { tx = w->bx + s * 20.0f; ty = w->by * 0.5f; kind = ELE_LOFT; }
    ele_kick(c, p, tx, ty, w->mode == ELE_THROWIN ? ELE_THROW : kind, 0.0f);
}

static __attribute__((noinline)) uint32_t ele_group(eleCtx *c, uint32_t team, uint32_t g, const uint32_t *words, uint32_t shape) {
    eleWorld *w = c->w;
    const eleShape *form = &ELE_SHAPES[ele_shape(c)];
    const float *th = c->theta[team];
    uint32_t base = team * ELE_TEAM + 4u * g, first = 4u * g, foe = (1u - team) * ELE_TEAM, normal = w->mode == ELE_NORMAL;
    uint32_t own = w->owner != ELE_NONE && ele_teamOf(w->owner) == team, attack = own || (!normal && w->team == team);
    lane4 lanes = { 0u, 1u, 2u, 3u }, slot = base + lanes;
    lane4 skip = (lane4)(first + lanes >= ele_n(c)) | (lane4)(slot == w->active) | (lane4)(ele_loadu(w->busy + base) > w->clock);
    float s = ele_sign(team), bu = s * w->bx, bv = s * w->by, reach = th[TH_PRESS_RANGE];
    float4 x = ele_load(w->x + base), y = ele_load(w->y + base), dbx = x - w->bx, dby = y - w->by;
    uint32_t second = th[TH_PRESSERS] >= 1.5f ? w->near[team][1] : ELE_NONE, pressing = normal && !own ? w->near[team][0] : ELE_NONE;
    lane4 special = ((lane4)(slot == pressing) | (lane4)(slot == (normal && !own ? second : ELE_NONE))) & ((lane4)(dbx * dbx + dby * dby < reach * reach) | (lane4)(w->owner == ELE_NONE ? ~0u : 0u));
    special |= (lane4)(slot == w->owner) | (lane4)(slot == (normal ? ELE_NONE : w->taker));
    if (g == 0u && ele_hasKeeper(c, team)) special.x = ~0u;
    special &= ~skip;
    if (!shape) return __builtin_reduce_add(special & (lane4){ 1u, 2u, 4u, 8u });
    float4 ub = ele_load(form->u + first), follow = ele_load(form->follow + first), lift = ele_load(form->lift + first), vb = ele_load(form->v + first);
    float4 tu = ub + follow * bu + (attack ? lift + th[TH_PUSH] : splat4(-th[TH_LINE]));
    float4 tv = vb * (attack ? 1.15f : 0.85f) + 0.3f * bv;
    float cap = attack ? (c->k->offside ? ele_offsideLine(c, team) - 0.8f : 48.0f) : bu - 3.0f;
    tu = select4((lane4)(lift > 0.0f) & (lane4)(attack ? 0u : ~0u), tu, __builtin_elementwise_min(tu, splat4(cap)));
    float4 tx = s * __builtin_elementwise_min(__builtin_elementwise_max(tu, splat4(-48.0f)), splat4(48.0f));
    float4 ty = s * __builtin_elementwise_min(__builtin_elementwise_max(tv, splat4(-31.0f)), splat4(31.0f));
    if (!attack) {
        const uint8_t *mark = form->mark + first;
        float4 ox = { w->x[foe + mark[0]], w->x[foe + mark[1]], w->x[foe + mark[2]], w->x[foe + mark[3]] };
        float4 oy = { w->y[foe + mark[0]], w->y[foe + mark[1]], w->y[foe + mark[2]], w->y[foe + mark[3]] };
        float4 gx = ox - s * 1.5f - tx, gy = oy - ty;
        lane4 near = (lane4)(gx * gx + gy * gy < 225.0f);
        tx += select4(near, th[TH_MARK] * gx, splat4(0.0f));
        ty += select4(near, th[TH_MARK] * gy, splat4(0.0f));
    }
    if (!normal && w->team != team) {
        float keep = w->mode == ELE_THROWIN ? 2.5f : 9.5f;
        float4 kx = tx - w->bx, ky = ty - w->by, kd = __builtin_elementwise_sqrt(kx * kx + ky * ky);
        lane4 inside = (lane4)(kd < keep);
        float4 push = keep / __builtin_elementwise_max(kd, splat4(1e-3f));
        tx = select4(inside, w->bx + kx * push, tx);
        ty = select4(inside, w->by + ky * push, ty);
    }
    float scale = ele_sigma(c) * 10.0f * 3.4641016f * (1.0f / 256.0f);
    lane4 shift = lanes * 8u;
    tx += (__builtin_convertvector(((lane4)words[0] >> shift) & 255u, float4) - 127.5f) * scale;
    ty += (__builtin_convertvector(((lane4)words[1] >> shift) & 255u, float4) - 127.5f) * scale;
    float4 dx = tx - x, dy = ty - y, d = __builtin_elementwise_sqrt(dx * dx + dy * dy);
    float4 top = select4((lane4)(d > 15.0f), splat4(ELE_SPRINT), splat4(ELE_RUN)) * (0.7f + 0.3f * ele_load(w->stamina + base));
    float4 k = w->mode == ELE_KICKOFF ? splat4(0.0f) : __builtin_elementwise_min(top, 2.0f * d) / __builtin_elementwise_max(d, splat4(1e-3f));
    lane4 plain = ~(skip | special);
    ele_store(w->wx + base, select4(plain, dx * k, ele_load(w->wx + base)));
    ele_store(w->wy + base, select4(plain, dy * k, ele_load(w->wy + base)));
    return __builtin_reduce_add(special & (lane4){ 1u, 2u, 4u, 8u });
}

static __attribute__((noinline)) void ele_special(eleCtx *c, uint32_t base, uint32_t lanes, const uint32_t *words) {
    eleWorld *w = c->w;
    for (uint32_t i = 0u; i < 4u; i++) {
        if (!(lanes >> i & 1u)) continue;
        uint32_t p = base + i, word = (words[0] >> (8u * i) & 255u) | (words[1] >> (8u * i) & 255u) << 8 | (words[2] >> (8u * i) & 255u) << 16 | (words[3] >> (8u * i) & 255u) << 24;
        if (w->mode != ELE_NORMAL && p == w->taker) {
            if (w->clock >= w->readyAt) ele_take(c, p, word);
        } else if (ele_keeper(c, p)) ele_goalie(c, p, word);
        else if (p == w->owner) ele_carrier(c, p, word);
        else ele_press(c, p, word, ele_dist(w->x[p], w->y[p], w->bx, w->by));
    }
}

static inline __attribute__((always_inline)) void ele_decide(eleCtx *c, uint32_t sub) {
    uint32_t round = c->w->clock / c->cycle, shape = round & 1u;
    if (c->w->clock % c->cycle >= c->k->slices) return;
    if (ele_groups(c) > 1u) {
        uint32_t team = sub & 1u, g = sub >> 1;
        uint32_t lanes = ele_group(c, team, g, c->w->noise + 4u * sub, shape);
        if (lanes) ele_special(c, team * ELE_TEAM + 4u * g, lanes, c->w->noise + 4u * sub);
        return;
    }
    if (sub) return;
    uint32_t lead = round >> 1 & 1u, lanes[2];
    for (uint32_t i = 0u; i < 2u; i++) lanes[i] = ele_group(c, lead ^ i, 0u, c->w->noise + 4u * (lead ^ i), shape);
    for (uint32_t i = 0u; i < 2u; i++)
        if (lanes[i]) ele_special(c, (lead ^ i) * ELE_TEAM, lanes[i], c->w->noise + 4u * (lead ^ i));
}

static uint32_t ele_aim(const eleCtx *c, uint32_t a, float dx, float dy, float *tx, float *ty) {
    const eleWorld *w = c->w;
    uint32_t pick = ELE_NONE;
    float best = -INFINITY;
    for (uint32_t i = 0u; i < ele_n(c); i++) {
        if (i == a) continue;
        float ex = w->x[i] - w->x[a], ey = w->y[i] - w->y[a], e = sqrtf(ex * ex + ey * ey) + 1e-3f, cosine = (ex * dx + ey * dy) / e;
        if (cosine < 0.3f) continue;
        float score = cosine - e / 60.0f;
        if (score > best) { best = score; pick = i; }
    }
    *tx = w->bx + 20.0f * dx;
    *ty = w->by + 20.0f * dy;
    if (pick != ELE_NONE) {
        float lead = ele_dist(w->x[pick], w->y[pick], w->bx, w->by) / 15.0f;
        *tx = w->x[pick] + w->vx[pick] * lead;
        *ty = w->y[pick] + w->vy[pick] * lead;
    }
    return pick;
}

static __attribute__((noinline)) void ele_agentKick(eleCtx *c, uint32_t a, float mx, float my, const float *ctrl) {
    eleWorld *w = c->w;
    if (ctrl[4] > 0.5f && w->mode != ELE_THROWIN) {
        if (w->held) ele_kick(c, a, w->x[a] + 45.0f, w->y[a] * 0.5f + 10.0f * my, ELE_LOFT, 0.0f);
        else ele_kick(c, a, ELEVEN_PITCH[PITCH_LENGTH] + 0.5f, 2.9f * my, ELE_SHOT, 0.4f);
        return;
    }
    float d = sqrtf(mx * mx + my * my), dx = mx, dy = my, tx, ty;
    if (d < 0.1f) { dx = w->fx[a]; dy = w->fy[a]; d = 1.0f; }
    ele_aim(c, a, dx / d, dy / d, &tx, &ty);
    ele_kick(c, a, tx, ty, w->mode == ELE_THROWIN || w->held ? ELE_THROW : ctrl[3] > 2.0f / 3.0f ? ELE_LOFT : ELE_GROUND, 0.0f);
}

static inline __attribute__((always_inline)) void ele_input(eleCtx *c, const float *ctrl) {
    eleWorld *w = c->w;
    uint32_t a = w->active;
    if (w->clock < w->busy[a]) return;
    float mx = ele_unit(ctrl[0]), my = ele_unit(ctrl[1]), m2 = mx * mx + my * my;
    uint32_t kicks = ctrl[4] > 0.5f || ctrl[3] > 1.0f / 3.0f;
    if (w->mode != ELE_NORMAL && w->taker == a) {
        w->wx[a] = w->wy[a] = 0.0f;
        if (w->clock < w->readyAt) return;
        if (w->clock >= w->autoAt) ele_take(c, a, ele_draw(c, TAG_AUTO, a).x);
        else if (kicks) ele_agentKick(c, a, mx, my, ctrl);
        return;
    }
    float top = (ctrl[2] > 0.0f ? ELE_SPRINT : ELE_RUN) * (0.7f + 0.3f * w->stamina[a]) * (w->owner == a ? ELE_DRIBBLE : 1.0f);
    float scale = m2 < 0.01f ? 0.0f : top / fmaxf(1.0f, sqrtf(m2));
    w->wx[a] = mx * scale;
    w->wy[a] = my * scale;
    if (w->mode != ELE_NORMAL) return;
    if (w->owner == a) {
        if (kicks) ele_agentKick(c, a, mx, my, ctrl);
    } else if (ctrl[5] > 0.5f && w->owner != ELE_NONE && ele_teamOf(w->owner) == 1u) ele_tackle(c, a, ctrl[5] > 0.9f);
}

static __attribute__((noinline)) void ele_hold(eleCtx *c, uint32_t a) {
    eleWorld *w = c->w;
    float length = ELEVEN_PITCH[PITCH_LENGTH];
    if (w->mode == ELE_KICKOFF) {
        w->x[a] = fminf(w->x[a], -0.5f);
        if (w->team == 1u) ele_clear(c, 0u, 0.0f, 0.0f, ELEVEN_PITCH[PITCH_CIRCLE]);
        return;
    }
    if (w->team == 0u) return;
    if (w->mode == ELE_GOALKICK) {
        float edge = length - ELEVEN_PITCH[PITCH_BOX] - 1.0f;
        if (w->x[a] > edge && fabsf(w->y[a]) <= ELEVEN_PITCH[PITCH_BOX_WIDTH]) w->x[a] = edge;
        return;
    }
    if (w->mode == ELE_PENALTY) {
        float edge = ELEVEN_PITCH[PITCH_BOX] - length + 4.5f;
        if (!ele_keeper(c, a) && w->x[a] < edge) w->x[a] = edge;
        return;
    }
    float keep = w->mode == ELE_THROWIN ? 2.0f : ELEVEN_PITCH[PITCH_CIRCLE], dx = w->x[a] - w->bx, dy = w->y[a] - w->by, d = sqrtf(dx * dx + dy * dy);
    if (d >= keep) return;
    if (d < 1e-3f) { dx = -1.0f; dy = 0.0f; d = 1.0f; }
    w->x[a] = ele_clamp(w->bx + dx * keep / d, -ELEVEN_PITCH[PITCH_LENGTH] - 3.0f, ELEVEN_PITCH[PITCH_LENGTH] + 3.0f);
    w->y[a] = ele_clamp(w->by + dy * keep / d, -ELEVEN_PITCH[PITCH_WIDTH] - 3.0f, ELEVEN_PITCH[PITCH_WIDTH] + 3.0f);
}

static inline __attribute__((always_inline)) void ele_phase(eleCtx *c, uint32_t base) {
    eleWorld *w = c->w;
    float dt = ELE_DT * (float)c->k->slices, slices = (float)c->k->slices;
    float4 x = ele_load(w->x + base), y = ele_load(w->y + base), vx = ele_load(w->vx + base), vy = ele_load(w->vy + base), s = ele_load(w->stamina + base);
    float4 ax = ele_load(w->wx + base) - vx, ay = ele_load(w->wy + base) - vy, m2 = ax * ax + ay * ay;
    float4 k = __builtin_elementwise_min(splat4(1.0f), ele_load(w->acc + base) * dt / __builtin_elementwise_sqrt(m2));
    lane4 keep = (lane4)((base + (lane4){ 0u, 1u, 2u, 3u }) == w->active);
    float4 nvx = vx + ax * k, nvy = vy + ay * k, v2 = select4(keep, vx * vx + vy * vy, nvx * nvx + nvy * nvy);
    float4 drain = select4((lane4)(v2 > 49.7f), splat4(-ELE_SPRINT_DRAIN), select4((lane4)(v2 > 16.0f), splat4(-ELE_RUN_DRAIN), splat4(ELE_RECOVER)));
    ele_store(w->vx + base, select4(keep, vx, nvx));
    ele_store(w->vy + base, select4(keep, vy, nvy));
    lane4 hold = keep | (lane4)((base % ELE_TEAM) + (lane4){ 0u, 1u, 2u, 3u } >= ele_n(c));
    float xl = ELEVEN_PITCH[PITCH_LENGTH] + 3.0f, yl = ELEVEN_PITCH[PITCH_WIDTH] + 3.0f;
    ele_store(w->x + base, select4(hold, x, __builtin_elementwise_min(__builtin_elementwise_max(x + nvx * dt, splat4(-xl)), splat4(xl))));
    ele_store(w->y + base, select4(hold, y, __builtin_elementwise_min(__builtin_elementwise_max(y + nvy * dt, splat4(-yl)), splat4(yl))));
    ele_store(w->stamina + base, __builtin_elementwise_min(__builtin_elementwise_max(s + drain * slices, splat4(0.0f)), splat4(1.0f)));
}

static inline __attribute__((always_inline)) void ele_move(eleCtx *c, uint32_t sub) {
    eleWorld *w = c->w;
    uint32_t a = w->active;
    if (ele_groups(c) > 1u) ele_phase(c, (sub & 1u) * ELE_TEAM + 4u * (sub >> 1));
    else if (sub == 0u) {
        ele_phase(c, 0u);
        ele_phase(c, ELE_TEAM);
    }
    float ax = w->wx[a] - w->vx[a], ay = w->wy[a] - w->vy[a], m2 = ax * ax + ay * ay;
    float k = fminf(1.0f, w->acc[a] * ELE_DT / sqrtf(m2)), vx = w->vx[a] + ax * k, vy = w->vy[a] + ay * k;
    w->vx[a] = vx;
    w->vy[a] = vy;
    w->x[a] = ele_clamp(w->x[a] + vx * ELE_DT, -ELEVEN_PITCH[PITCH_LENGTH] - 3.0f, ELEVEN_PITCH[PITCH_LENGTH] + 3.0f);
    w->y[a] = ele_clamp(w->y[a] + vy * ELE_DT, -ELEVEN_PITCH[PITCH_WIDTH] - 3.0f, ELEVEN_PITCH[PITCH_WIDTH] + 3.0f);
    if (w->mode != ELE_NORMAL && a != w->taker) ele_hold(c, a);
}

static inline __attribute__((always_inline)) void ele_ball(eleCtx *c) {
    eleWorld *w = c->w;
    if (!w->live) return;
    uint32_t o = w->owner;
    if (o != ELE_NONE) {
        if (w->held && !ele_inBox(ele_teamOf(o), w->x[o], w->y[o])) w->held = 0u;
        if (w->held) {
            w->bx = w->x[o];
            w->by = w->y[o];
            w->bz = 1.0f;
        } else {
            ele_face(w, o, w->vx[o], w->vy[o]);
            w->bx = w->x[o] + ELE_FOOT * w->fx[o];
            w->by = w->y[o] + ELE_FOOT * w->fy[o];
            w->bz = 0.0f;
        }
        w->bvx = w->vx[o];
        w->bvy = w->vy[o];
        w->bvz = 0.0f;
        return;
    }
    if (w->bz > 0.0f || w->bvz != 0.0f) {
        w->bvx *= ELE_AIR;
        w->bvy *= ELE_AIR;
        w->bvz = w->bvz * ELE_AIR - ELE_GRAVITY * ELE_DT;
        w->bx += w->bvx * ELE_DT;
        w->by += w->bvy * ELE_DT;
        w->bz += w->bvz * ELE_DT;
        if (w->bz <= 0.0f) {
            w->bz = 0.0f;
            if (w->bvz < -1.5f) {
                w->bvz *= -ELE_BOUNCE;
                w->bvx *= ELE_SKID;
                w->bvy *= ELE_SKID;
            } else w->bvz = 0.0f;
        }
        return;
    }
    w->bvx *= ELE_ROLL;
    w->bvy *= ELE_ROLL;
    if (w->bvx * w->bvx + w->bvy * w->bvy < 0.0025f) w->bvx = w->bvy = 0.0f;
    w->bx += w->bvx * ELE_DT;
    w->by += w->bvy * ELE_DT;
}

static __attribute__((noinline)) void ele_resolve(eleCtx *c, uint32_t pick) {
    eleWorld *w = c->w;
    uint32_t team = ele_teamOf(pick), keeper = ele_keeper(c, pick) && ele_inBox(team, w->bx, w->by);
    if (w->offside >> pick & 1u) {
        w->offside = 0u;
        ele_restartAt(c, ELE_FREEKICK, 1u - team, w->x[pick], w->y[pick]);
        return;
    }
    w->offside = 0u;
    float rx = w->bvx - w->vx[pick], ry = w->bvy - w->vy[pick], rel = sqrtf(rx * rx + ry * ry + w->bvz * w->bvz);
    u4 r = ele_draw(c, TAG_TOUCH, pick);
    if (r.x < u01Bound(keeper ? ele_clamp(1.25f - rel / 28.0f, 0.3f, 0.97f) : ele_clamp(1.4f - rel / 18.0f, 0.15f, 0.99f))) {
        ele_gain(c, pick);
        if (keeper) {
            w->held = 1u;
            w->heldAt = w->clock;
            w->dive[team] = 0u;
            w->acc[pick] = ELE_ACCEL;
        }
        return;
    }
    float turn = (u01(r.y) - 0.5f) * (keeper ? 2.4f : 1.8f), sn, cs, keep = keeper ? 0.35f : 0.45f;
    oak_sincos(turn, &sn, &cs);
    float vx = keeper ? -w->bvx : w->bvx, vy = w->bvy;
    w->bvx = keep * (vx * cs - vy * sn);
    w->bvy = keep * (vx * sn + vy * cs);
    w->bvz = keeper ? 2.0f : fmaxf(w->bvz * keep, 0.0f);
    w->last = pick;
    w->touch[pick] = w->clock + 12u;
}

static __attribute__((noinline)) uint32_t ele_claim(const eleCtx *c, const float4 *d2) {
    const eleWorld *w = c->w;
    float4 best = splat4(INFINITY);
    lane4 who = (lane4)ELE_NONE, now = (lane4)w->clock;
    for (uint32_t team = 0u; team < 2u; team++)
        for (uint32_t g = 0u; g < ele_groups(c); g++) {
            uint32_t base = team * ELE_TEAM + 4u * g;
            float4 e = d2[team * 3u + g];
            lane4 ok = (lane4)(e < ELE_REACH * ELE_REACH) & (lane4)(now >= ele_loadu(w->touch + base)) & (lane4)(e < best);
            best = select4(ok, e, best);
            who = (ok & (base + (lane4){ 0u, 1u, 2u, 3u })) | (~ok & who);
        }
    float low = __builtin_reduce_min(best);
    lane4 hit = (lane4)(best == low);
    return __builtin_reduce_min((hit & who) | ~hit);
}

static inline __attribute__((always_inline)) void ele_contact(eleCtx *c) {
    eleWorld *w = c->w;
    float bx = w->bx, by = w->by, bz = w->bz, low = INFINITY;
    uint32_t pick = ELE_NONE;
    if (bz < ELE_CHEST) {
        float4 d2[6], near = splat4(INFINITY);
        for (uint32_t team = 0u; team < 2u; team++)
            for (uint32_t g = 0u; g < ele_groups(c); g++) {
                d2[team * 3u + g] = ele_d2(w, team * ELE_TEAM + 4u * g, bx, by);
                near = __builtin_elementwise_min(near, d2[team * 3u + g]);
            }
        if (__builtin_reduce_min(near) < ELE_REACH * ELE_REACH) {
            pick = ele_claim(c, d2);
            if (pick != ELE_NONE) low = (w->x[pick] - bx) * (w->x[pick] - bx) + (w->y[pick] - by) * (w->y[pick] - by);
        }
    }
    for (uint32_t team = 0u; team < 2u && fabsf(bx) >= ELEVEN_PITCH[PITCH_LENGTH] - ELEVEN_PITCH[PITCH_BOX]; team++) {
        uint32_t q = team * ELE_TEAM;
        if (!ele_hasKeeper(c, team) || w->clock < w->touch[q] || bz > ELE_HANDS_HIGH || !ele_inBox(team, w->x[q], w->y[q]) || !ele_inBox(team, bx, by)) continue;
        float reach = w->clock < w->dive[team] ? ELE_DIVE_REACH : ELE_HANDS, dx = w->x[q] - bx, dy = w->y[q] - by, lift = fmaxf(bz - 1.0f, 0.0f), d2 = dx * dx + dy * dy + lift * lift;
        if (d2 < reach * reach && d2 < low) { low = d2; pick = q; }
    }
    if (pick != ELE_NONE) ele_resolve(c, pick);
}

static inline __attribute__((always_inline)) uint32_t ele_referee(eleCtx *c) {
    eleWorld *w = c->w;
    if (!w->live) return ELE_QUIET;
    float length = ELEVEN_PITCH[PITCH_LENGTH], width = ELEVEN_PITCH[PITCH_WIDTH], ax = fabsf(w->bx), ay = fabsf(w->by);
    if (ax > length + ELE_RADIUS) {
        uint32_t attacker = w->bx > 0.0f ? 0u : 1u, defender = 1u - attacker;
        float sx = w->bx > 0.0f ? 1.0f : -1.0f, sy = w->by > 0.0f ? 1.0f : -1.0f;
        if (ay < ELEVEN_PITCH[PITCH_POST] && w->bz < ELEVEN_PITCH[PITCH_BAR]) {
            w->goals[attacker]++;
            if (!c->k->drill) ele_kickoff(c, defender);
            return attacker == 0u ? ELE_GOAL_FOR : ELE_GOAL_AGAINST;
        }
        if (!c->k->drill) {
            if (ele_teamOf(w->last) == defender) ele_restartAt(c, ELE_CORNER, attacker, sx * length, sy * width);
            else ele_restartAt(c, ELE_GOALKICK, defender, sx * (length - ELEVEN_PITCH[PITCH_AREA]), sy * 4.5f);
        }
        return ELE_OUT;
    }
    if (ay > width + ELE_RADIUS) {
        if (!c->k->drill) ele_restartAt(c, ELE_THROWIN, 1u - ele_teamOf(w->last), ele_clamp(w->bx, -length, length), (w->by > 0.0f ? 1.0f : -1.0f) * width);
        return ELE_OUT;
    }
    return ELE_QUIET;
}

static void ele_leagueInit(eleLeague *g, uint32_t seed) {
    for (uint32_t i = 0u; i < LEAGUE_MEMBERS; i++) {
        g->winRate[i] = 0.5f;
        for (uint32_t d = 0u; d < ELE_THETA; d += 4u) {
            u44 r = philox4(seed, LEAGUE_RNG_STREAM, (lane4)(0xFFFFFFF0u + i), d + (lane4){ 0u, 1u, 2u, 3u }, (lane4)0u, (lane4)0u);
            float4 n = normal4(r.x, r.y);
            for (uint32_t j = 0u; j < 4u; j++) {
                float value = ELEVEN_THETA[d + j] + (i ? 0.1f * (ELEVEN_THETA_HIGH[d + j] - ELEVEN_THETA_LOW[d + j]) * n[j] : 0.0f);
                g->theta[i * ELE_THETA + d + j] = ele_clamp(value, ELEVEN_THETA_LOW[d + j], ELEVEN_THETA_HIGH[d + j]);
            }
        }
    }
}

static void ele_pick(eleCtx *c, eleLeague *g) {
    if (!c->k->league) return;
    c->w->member = leaguePick(g->winRate, philox4x32(c->seed, LEAGUE_RNG_STREAM, c->w->game, 0x5049434Bu, 0u, 0u).x);
    c->theta[1] = g->theta + c->w->member * ELE_THETA;
}

static __attribute__((noinline)) void ele_final(eleCtx *c, eleLeague *g) {
    eleWorld *w = c->w;
    uint32_t us = w->goals[0], them = w->goals[1];
    if (us > them) w->wins++;
    else if (us < them) w->losses++;
    else w->draws++;
    if (c->k->league) {
        leagueScore(g->winRate, w->member, us > them ? 1.0f : us < them ? 0.0f : 0.5f);
        leagueRenew(g->winRate, g->theta, ELE_THETA, ELEVEN_THETA_LOW, ELEVEN_THETA_HIGH, c->seed, w->game);
    }
    ele_begin(c, w->game + 1u);
    ele_pick(c, g);
}

static __attribute__((noinline)) void ele_halftime(eleCtx *c) {
    eleWorld *w = c->w;
    w->half = 1u;
    for (uint32_t q = 0u; q < ELE_SLOTS; q++) w->stamina[q] = 1.0f;
    ele_kickoff(c, 1u - w->kickoff);
}

static inline uint32_t ele_scanSlot(const eleCtx *c, uint32_t q) { return q < ele_n(c) ? q : ELE_TEAM + q - ele_n(c); }

static inline __attribute__((always_inline)) void ele_observe(const eleCtx *c, float *obs) {
    const eleWorld *w = c->w;
    uint32_t a = w->active, o = w->owner, scan = ele_scanSlot(c, w->scan);
    float inv = 1.0f / ELEVEN_PITCH[PITCH_LENGTH], d2;
    uint32_t near = w->closeFor == a ? w->close : ele_nearest(c, 1u, w->x[a], w->y[a], 0, &d2);
    obs[0] = w->x[a] * inv;
    obs[1] = w->y[a] * inv;
    obs[2] = w->vx[a] * 0.1f;
    obs[3] = w->vy[a] * 0.1f;
    obs[4] = (w->bx - w->x[a]) * inv;
    obs[5] = (w->by - w->y[a]) * inv;
    obs[6] = w->bz * (1.0f / ELEVEN_PITCH[PITCH_BAR]);
    obs[7] = w->bvx * 0.1f;
    obs[8] = w->bvy * 0.1f;
    obs[9] = o == ELE_NONE ? 0.0f : o == a ? 1.0f : ele_teamOf(o) == 0u ? 0.5f : -1.0f;
    obs[10] = (w->x[near] - w->x[a]) * inv;
    obs[11] = (w->y[near] - w->y[a]) * inv;
    obs[12] = (scan < ELE_TEAM ? 1.0f : -1.0f) * (float)(scan % ELE_TEAM + 1u) * (1.0f / 11.0f);
    obs[13] = (w->x[scan] - w->x[a]) * inv;
    obs[14] = (w->y[scan] - w->y[a]) * inv;
    obs[15] = w->stamina[a];
    obs[16] = w->mode == ELE_NORMAL ? 0.0f : (w->team == 0u ? 1.0f : -1.0f) * (float)w->mode * (1.0f / 6.0f);
    obs[17] = (float)w->switched;
}

static inline void ele_mirror(const eleWorld *w, uint32_t *words) {
    words[0] = w->clock;
    words[1] = w->game;
    words[2] = w->goals[0] | w->goals[1] << 16;
    words[3] = w->mode | w->team << 3 | w->live << 4 | w->held << 5 | w->half << 6 | w->kickoff << 7 | w->owner << 8 | w->active << 16 | w->taker << 24;
    words[4] = floatWord(w->bx);
    words[5] = floatWord(w->by);
    words[6] = floatWord(w->bz);
}

uint32_t eleven_extraWords(uint32_t instance, uint32_t steps) {
    (void)instance; (void)steps;
    return 4u + (uint32_t)((sizeof(eleWorld) + sizeof(eleLeague) + sizeof(eleOracle)) / sizeof(uint32_t));
}

void eleven_reset(uint32_t instance, uint32_t seed, uint32_t *words, uint32_t *extra, uint32_t steps) {
    (void)steps;
    const elevenKnobs *k = &ELEVEN_KNOBS[instance];
    eleWorld *w = ele_world(extra);
    eleLeague *g = ele_league(extra);
    eleCtx c;
    memset(w, 0, sizeof *w);
    memset(ele_oracleOf(extra), 0, sizeof(eleOracle));
    if (k->league) ele_leagueInit(g, seed);
    ele_bind(&c, k, w, g, seed);
    ele_begin(&c, 0u);
    ele_pick(&c, g);
    ele_mirror(w, words);
}

static inline __attribute__((always_inline)) uint32_t ele_advance(eleCtx *c, const float *ctrl) {
    eleWorld *w = c->w;
    uint32_t sub = w->clock % c->k->slices;
    w->switched = 0u;
    if (sub == 0u) ele_tick(c);
    ele_input(c, ctrl);
    ele_decide(c, sub);
    ele_move(c, sub);
    ele_ball(c);
    if (w->live && w->owner == ELE_NONE) ele_contact(c);
    uint32_t event = ele_referee(c);
    w->clock++;
    return event;
}

static inline __attribute__((always_inline)) float ele_play(const elevenKnobs *k, uint32_t seed, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done) {
    eleWorld *w = ele_world(extra);
    eleLeague *g = ele_league(extra);
    eleCtx c;
    ele_bind(&c, k, w, g, seed);
    uint32_t event = ele_advance(&c, ctrl);
    float reward = event == ELE_GOAL_FOR ? 1.0f : event == ELE_GOAL_AGAINST ? -1.0f : 0.0f;
    *done = 0u;
    if (k->drill) {
        if (event != ELE_QUIET || w->owner == ELE_TEAM || w->clock >= k->ticks * k->slices) {
            *done = 1u;
            if (event == ELE_GOAL_FOR) w->wins++;
            else w->losses++;
            ele_begin(&c, w->game + 1u);
        }
    } else if (w->clock >= k->ticks * k->slices) {
        *done = 1u;
        ele_final(&c, g);
    } else if (w->clock == k->ticks * k->slices / 2u) ele_halftime(&c);
    uint32_t others = 2u * k->perSide, next = w->scan + 1u == others ? 0u : w->scan + 1u;
    w->scan = ele_scanSlot(&c, next) == w->active ? (next + 1u == others ? 0u : next + 1u) : next;
    ele_observe(&c, obs);
    ele_mirror(w, words);
    return reward;
}

float eleven_step(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done, const float *shared) {
    (void)step; (void)shared;
    switch (instance) {
        case 0u: return ele_play(&ELEVEN_KNOBS[0], seed, ctrl, words, extra, obs, done);
        case 1u: return ele_play(&ELEVEN_KNOBS[1], seed, ctrl, words, extra, obs, done);
        case 2u: return ele_play(&ELEVEN_KNOBS[2], seed, ctrl, words, extra, obs, done);
        case 3u: return ele_play(&ELEVEN_KNOBS[3], seed, ctrl, words, extra, obs, done);
        default: return ele_play(&ELEVEN_KNOBS[4], seed, ctrl, words, extra, obs, done);
    }
}

void eleven_observe(uint32_t instance, uint32_t seed, const uint32_t *words, const uint32_t *extra, uint32_t step, float *obs, const float *shared) {
    (void)words; (void)step; (void)shared;
    eleCtx c;
    ele_bind(&c, &ELEVEN_KNOBS[instance], (eleWorld *)ele_worldAt(extra), (const eleLeague *)(ele_worldAt(extra) + 1), seed);
    ele_observe(&c, obs);
}

static uint32_t ele_rollout(const eleCtx *real, eleWorld *scratch, const float *first, uint32_t horizon) {
    float idle[OAK_ENV_ACT_MAX] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
    uint32_t edge = real->k->ticks * real->k->slices, half = edge / 2u;
    memcpy(scratch, real->w, sizeof *scratch);
    eleCtx c = *real;
    c.w = scratch;
    for (uint32_t n = 0u; n < horizon; n++) {
        if (scratch->clock + 1u >= edge || (!real->k->drill && scratch->clock + 1u == half)) return ELE_QUIET;
        uint32_t event = ele_advance(&c, n ? idle : first);
        if (event != ELE_QUIET) return event;
        if (!scratch->live) return ELE_DEAD;
        if (scratch->owner != ELE_NONE) return ele_teamOf(scratch->owner) ? ELE_LOST : ELE_KEPT;
    }
    return ELE_QUIET;
}

static void ele_go(float *ctrl, float dx, float dy, float sprint) {
    float d = sqrtf(dx * dx + dy * dy);
    ctrl[0] = d > 1e-4f ? dx / d : 0.0f;
    ctrl[1] = d > 1e-4f ? dy / d : 0.0f;
    ctrl[2] = sprint;
}

static void ele_chase(const eleCtx *c, uint32_t a, float *ctrl) {
    const eleWorld *w = c->w;
    float x = w->bx, y = w->by, z = w->bz, vx = w->bvx, vy = w->bvy, vz = w->bvz, speed = ELE_SPRINT * (0.7f + 0.3f * w->stamina[a]);
    float px = w->x[a], py = w->y[a], tx = x, ty = y;
    for (uint32_t n = 1u; n <= 120u; n++) {
        if (z > 0.0f || vz != 0.0f) {
            vx *= ELE_AIR;
            vy *= ELE_AIR;
            vz = vz * ELE_AIR - ELE_GRAVITY * ELE_DT;
            x += vx * ELE_DT;
            y += vy * ELE_DT;
            z += vz * ELE_DT;
            if (z <= 0.0f) {
                z = 0.0f;
                if (vz < -1.5f) { vz *= -ELE_BOUNCE; vx *= ELE_SKID; vy *= ELE_SKID; }
                else vz = 0.0f;
            }
        } else {
            vx *= ELE_ROLL;
            vy *= ELE_ROLL;
            x += vx * ELE_DT;
            y += vy * ELE_DT;
        }
        tx = x;
        ty = y;
        if (z < ELE_CHEST && ele_dist(px, py, x, y) - ELE_REACH <= speed * (float)n * ELE_DT - 0.5f) break;
    }
    ele_go(ctrl, tx - px, ty - py, 1.0f);
}

static void ele_defend(const eleCtx *c, uint32_t a, float *ctrl) {
    eleWorld *w = c->w;
    uint32_t o = w->owner;
    float d = ele_dist(w->x[a], w->y[a], w->x[o], w->y[o]), lead = fminf(d / 9.0f, 1.0f);
    ele_go(ctrl, w->x[o] + w->vx[o] * lead - w->x[a], w->y[o] + w->vy[o] * lead - w->y[a], 1.0f);
    if (w->held || d > ELE_STAND || w->clock < w->tackle[a]) return;
    if (ele_draw(c, TAG_TACKLE, a).x < u01Bound(0.35f + 0.4f * ele_front(w, o, a))) ctrl[5] = 0.7f;
}

static float ele_danger(const eleWorld *w, const eleCtx *c, float x, float y, float horizon) {
    float4 low = splat4(INFINITY);
    for (uint32_t g = 0u; g < ele_groups(c); g++) {
        uint32_t base = ELE_TEAM + 4u * g;
        float4 dx = ele_load(w->x + base) + ele_load(w->vx + base) * horizon - x, dy = ele_load(w->y + base) + ele_load(w->vy + base) * horizon - y;
        low = __builtin_elementwise_min(low, dx * dx + dy * dy);
    }
    return sqrtf(__builtin_reduce_min(low));
}

static void ele_dribble(const eleCtx *c, uint32_t a, float *ctrl, float *space) {
    const eleWorld *w = c->w;
    float gx = ELEVEN_PITCH[PITCH_LENGTH] - w->x[a], gy = -w->y[a], gd = sqrtf(gx * gx + gy * gy) + 1e-3f;
    float speed = ELE_SPRINT * ELE_DRIBBLE * (0.7f + 0.3f * w->stamina[a]), best = -INFINITY, bx = gx / gd, by = gy / gd;
    *space = 0.0f;
    for (uint32_t k = 0u; k < 16u; k++) {
        float s, co;
        oak_sincos((float)k * (OAK_PI / 8.0f), &s, &co);
        float px = w->x[a] + co * speed * 0.5f, py = w->y[a] + s * speed * 0.5f;
        if (fabsf(py) > ELEVEN_PITCH[PITCH_WIDTH] - 1.0f || px > ELEVEN_PITCH[PITCH_LENGTH] || px < 1.0f - ELEVEN_PITCH[PITCH_LENGTH]) continue;
        float room = ele_danger(w, c, px, py, 0.5f), toward = (co * gx + s * gy) / gd;
        float score = toward + 0.5f * fminf(room, 6.0f) - (room < 2.8f ? 10.0f : 0.0f);
        if (score > best) { best = score; bx = co; by = s; *space = room; }
    }
    ele_go(ctrl, bx, by, w->stamina[a] > 0.2f ? 1.0f : -1.0f);
}

static int ele_passTo(const eleCtx *c, uint32_t a, float *ctrl, float floor) {
    const eleWorld *w = c->w;
    float best = floor, px = 0.0f, py = 0.0f;
    int found = 0;
    for (uint32_t q = 0u; q < ele_n(c); q++) {
        if (q == a || ele_keeper(c, q)) continue;
        float dx = w->x[q] - w->x[a], dy = w->y[q] - w->y[a], d = sqrtf(dx * dx + dy * dy);
        if (d < 5.0f || d > 38.0f) continue;
        float tx, ty;
        if (ele_aim(c, a, dx / d, dy / d, &tx, &ty) != q) continue;
        float lx = tx - w->bx, ly = ty - w->by, length = sqrtf(lx * lx + ly * ly), margin, landing;
        ele_lanes(c, 1u, w->bx, w->by, lx, ly, length, 0.5f * (8.0f + ELE_ROLL_RATE * length), &margin, &landing);
        float value = fminf(margin, 4.0f) + 0.15f * (tx - w->x[a]) + 0.3f * fminf(ele_danger(w, c, tx, ty, 0.0f), 8.0f);
        if (margin > 1.0f && value > best) { best = value; px = dx / d; py = dy / d; found = 1; }
    }
    if (!found) return 0;
    ctrl[0] = px;
    ctrl[1] = py;
    ctrl[2] = 0.0f;
    ctrl[3] = 0.5f;
    return 1;
}

static inline __attribute__((always_inline)) void ele_think(const elevenKnobs *k, uint32_t seed, uint32_t *extra, float *ctrl) {
    static const float aims[4] = { 1.0f, -1.0f, 0.55f, -0.55f };
    eleWorld *w = ele_world(extra);
    eleOracle *o = ele_oracleOf(extra);
    eleCtx c;
    ele_bind(&c, k, w, ele_league(extra), seed);
    uint32_t a = w->active;
    for (uint32_t d = 0u; d < OAK_ENV_ACT_MAX; d++) ctrl[d] = 0.0f;
    if (w->clock < w->busy[a]) return;
    if (o->pending && o->game == w->game && w->live && w->owner == ELE_NONE && w->mode == ELE_NORMAL) return;
    o->pending = 0u;
    if (w->mode != ELE_NORMAL) {
        if (w->taker != a) { ele_go(ctrl, w->bx - w->x[a], w->by - w->y[a], 0.0f); return; }
        if (w->clock < w->readyAt) return;
        if (w->mode == ELE_PENALTY || w->mode == ELE_FREEKICK) {
            for (uint32_t i = 0u; i < 4u; i++) {
                float shot[OAK_ENV_ACT_MAX] = { 0.0f, aims[i], 0.0f, 0.0f, 1.0f, 0.0f };
                if (ele_rollout(&c, &o->world, shot, 150u) == ELE_GOAL_FOR) { memcpy(ctrl, shot, sizeof shot); o->pending = 1u; o->game = w->game; return; }
            }
            if (w->mode == ELE_PENALTY) { ctrl[1] = 1.0f; ctrl[4] = 1.0f; return; }
        }
        if (!ele_passTo(&c, a, ctrl, -INFINITY)) { ctrl[0] = 1.0f; ctrl[3] = 0.9f; }
        return;
    }
    if (w->owner == a) {
        float gx = ELEVEN_PITCH[PITCH_LENGTH] - w->x[a], gy = w->y[a];
        if (!w->held && gx * gx + gy * gy < 32.0f * 32.0f && w->clock % k->slices == 0u)
            for (uint32_t i = 0u; i < 4u; i++) {
                float shot[OAK_ENV_ACT_MAX] = { 0.0f, aims[i], 0.0f, 0.0f, 1.0f, 0.0f };
                if (ele_rollout(&c, &o->world, shot, 150u) == ELE_GOAL_FOR) { memcpy(ctrl, shot, sizeof shot); o->pending = 1u; o->game = w->game; return; }
            }
        if (w->held) {
            if (!ele_passTo(&c, a, ctrl, -INFINITY)) { ctrl[0] = 1.0f; ctrl[4] = 1.0f; }
            return;
        }
        float space;
        ele_dribble(&c, a, ctrl, &space);
        if (space < 2.8f || ele_danger(w, &c, w->x[a], w->y[a], 0.0f) < 2.0f) ele_passTo(&c, a, ctrl, -INFINITY);
        return;
    }
    if (w->owner != ELE_NONE && ele_teamOf(w->owner) == 1u) { ele_defend(&c, a, ctrl); return; }
    ele_chase(&c, a, ctrl);
}

void eleven_oracle(uint32_t instance, uint32_t seed, uint32_t step, const uint32_t *words, uint32_t *extra, const float *obs, float *ctrl, const float *shared) {
    (void)step; (void)words; (void)obs; (void)shared;
    switch (instance) {
        case 0u: ele_think(&ELEVEN_KNOBS[0], seed, extra, ctrl); return;
        case 1u: ele_think(&ELEVEN_KNOBS[1], seed, extra, ctrl); return;
        case 2u: ele_think(&ELEVEN_KNOBS[2], seed, extra, ctrl); return;
        case 3u: ele_think(&ELEVEN_KNOBS[3], seed, extra, ctrl); return;
        default: ele_think(&ELEVEN_KNOBS[4], seed, extra, ctrl); return;
    }
}

static uint32_t ele_q16(float v) { return (uint32_t)ele_clamp((v + 64.0f) * 512.0f, 0.0f, 65535.0f); }
static uint32_t ele_q8(float v) { return (uint32_t)ele_clamp(v * 10.0f + 128.0f, 0.0f, 255.0f); }

uint32_t eleven_view(uint32_t instance, const uint32_t *words, const uint32_t *extra, const float *shared, uint32_t *out) {
    (void)words; (void)shared;
    const elevenKnobs *k = &ELEVEN_KNOBS[instance];
    uint32_t size = 4u * k->perSide + 7u;
    if (!out) return size;
    const eleWorld *w = ele_worldAt(extra);
    uint32_t at = 0u;
    for (uint32_t team = 0u; team < 2u; team++)
        for (uint32_t i = 0u; i < k->perSide; i++) {
            uint32_t q = team * ELE_TEAM + i;
            uint32_t flags = (uint32_t)(w->clock < w->busy[q]) | (uint32_t)(w->clock < w->dive[team] && i == 0u) << 1 | (uint32_t)(q == w->active) << 2 |
                             (uint32_t)(q == w->taker && w->mode != ELE_NORMAL) << 3 | (w->offside >> q & 1u) << 4;
            out[at++] = ele_q16(w->x[q]) | ele_q16(w->y[q]) << 16;
            out[at++] = ele_q8(w->vx[q]) | ele_q8(w->vy[q]) << 8 | (uint32_t)(w->stamina[q] * 255.0f + 0.5f) << 16 | flags << 24;
        }
    out[at++] = ele_q16(w->sx) | ele_q16(w->sy) << 16;
    out[at++] = (w->readyAt > w->clock ? w->readyAt - w->clock : 0u) | w->scan << 16 | w->switched << 24;
    out[at++] = ele_q16(w->bvx) | ele_q16(w->bvy) << 16;
    out[at++] = ele_q16(w->bvz) | w->last << 16 | w->member << 24;
    out[at++] = w->offside;
    out[at++] = (w->wins & 0xFFFFu) | (w->losses & 0xFFFFu) << 16;
    out[at] = (w->draws & 0xFFFFu) | w->close << 16 | w->near[0][0] << 24;
    return size;
}
