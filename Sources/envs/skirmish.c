#include <string.h>
#include "../env.h"

typedef struct { uint32_t allies, enemies, races, league, slices, battleTicks; float surround, tick, arena, timeoutReward; } skirmishKnobs;

static const skirmishKnobs SKIRMISH_KNOBS[5] = {
    { 5u, 5u, 1u, 0u, 8u, 320u, 0.0f, 0.25f, 32.0f, -1.0f },
    { 5u, 5u, 3u, 0u, 8u, 320u, 0.5f, 0.25f, 32.0f, -1.0f },
    { 10u, 11u, 3u, 0u, 8u, 320u, 0.5f, 0.25f, 32.0f, -1.0f },
    { 20u, 23u, 3u, 0u, 16u, 320u, 0.5f, 0.25f, 32.0f, -1.0f },
    { 10u, 10u, 3u, 1u, 8u, 320u, 0.5f, 0.25f, 32.0f, -1.0f }
};

static const float SKIRMISH_WEIGHT[3] = { 0.45f, 0.45f, 0.10f };
static const float SKIRMISH_SPAWN[5] = { 2.0f, 12.0f, 2.0f, 7.0f, 19.0f };
static const float SKIRMISH_RULES[6] = { 0.5f, 0.5f, 10.0f, 2.0f, 2.0f, 2.0f };
static const float SKIRMISH_HP[9] = { 80.0f, 100.0f, 200.0f, 45.0f, 125.0f, 150.0f, 35.0f, 90.0f, 30.0f };
static const float SKIRMISH_SHIELD[9] = { 80.0f, 50.0f, 150.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
static const float SKIRMISH_ARMOR[9] = { 1.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f };
static const float SKIRMISH_RANGE[9] = { 6.0f, 2.0f, 7.0f, 5.0f, 6.0f, 4.0f, 2.0f, 5.0f, 2.0f };
static const float SKIRMISH_SIGHT[9] = { 10.0f, 9.0f, 10.0f, 9.0f, 10.0f, 11.0f, 8.0f, 9.0f, 8.0f };
static const float SKIRMISH_DAMAGE[9] = { 13.0f, 16.0f, 20.0f, 6.0f, 10.0f, 0.0f, 5.0f, 12.0f, 16.0f };
static const float SKIRMISH_BONUS[9] = { 5.0f, 0.0f, 10.0f, 0.0f, 10.0f, 0.0f, 0.0f, 0.0f, 19.0f };
static const uint32_t SKIRMISH_BONUS_VS[9] = { 2u, 0u, 1u, 0u, 2u, 0u, 0u, 0u, 1u };
static const uint32_t SKIRMISH_ARMORED[9] = { 1u, 0u, 1u, 0u, 1u, 1u, 0u, 0u, 0u };
static const float SKIRMISH_COOLDOWN[9] = { 1.34f, 0.86f, 1.07f, 0.61f, 1.07f, 0.0f, 0.5f, 0.59f, 0.0f };
static const float SKIRMISH_SPEED[9] = { 4.13f, 3.15f, 3.15f, 3.15f, 3.15f, 3.5f, 4.13f, 3.15f, 3.5f };
static const float SKIRMISH_SPLASH[9] = { 0.0f, 0.0f, 0.5f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 2.2f };
static const float SKIRMISH_HEAL[9] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 12.6f, 0.0f, 0.0f, 0.0f };
static const float SKIRMISH_SLOW[9] = { 0.0f, 0.0f, 0.0f, 0.0f, 1.07f, 0.0f, 0.0f, 0.0f, 0.0f };
static const uint32_t SKIRMISH_SUICIDE[9] = { 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 1u };
static const float SKIRMISH_THREAT[9] = { 10.0f, 19.0f, 25.0f, 10.0f, 12.0f, 13.0f, 10.0f, 20.0f, 35.0f };
static const float SKIRMISH_THETA[11] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f };
static const float SKIRMISH_THETA_LOW[11] = { -2.0f, -2.0f, -2.0f, -2.0f, -2.0f, -2.0f, -2.0f, -2.0f, 0.0f, 0.2f, 0.0f };
static const float SKIRMISH_THETA_HIGH[11] = { 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 4.0f, 2.0f, 1.0f, 1.0f, 0.5f };

enum { SK_FOE = 20, SK_SLOTS = 64, SK_NONE = 63, SK_THETA = 11, SK_DRAWS = 96, SK_PICK = 0x5049434Bu };
enum { SK_MIN_DAMAGE, SK_SLOWDOWN, SK_REGEN_DELAY, SK_REGEN_RATE, SK_FORM, SK_ARRIVE };
enum { SK_REFLECT_NEAR, SK_REFLECT_FAR, SK_CENTRE, SK_RING_NEAR, SK_RING_FAR };
enum { SK_ATTACK = 4, SK_CYCLE = 5 };
enum { SK_TICK, SK_SLICE, SK_BATTLE, SK_WINS, SK_LOSSES, SK_DRAWN, SK_TIMEOUTS, SK_RACE, SK_START, SK_SELECT, SK_CALL, SK_ALLIES, SK_FOES, SK_MEMBER, SK_WORDS };
enum { SK_WEAK = 3, SK_CLOSE, SK_THREATENED, SK_FOCUS, SK_DANGER, SK_ENGAGE, SK_KITE, SK_RETREAT };
enum { SK_HORIZON = 20, SK_PERIOD = 8, SK_CALLS = 3, SK_CONTACT = 81, SK_PATIENCE = 60 };
enum { SK_HOLD, SK_GATHER, SK_FIGHT, SK_MODES };

typedef struct {
    _Alignas(16) float x[SK_SLOTS], y[SK_SLOTS], hp[SK_SLOTS], shield[SK_SLOTS];
    uint32_t ready[SK_SLOTS], slowUntil[SK_SLOTS], hitAt[SK_SLOTS];
    float inverse[SK_SLOTS];
    uint8_t type[SK_SLOTS], target[SK_SLOTS], hunt[SK_SLOTS];
    float goalX[2], goalY[2], pace;
    uint32_t focus;
} skWorld;

typedef struct { _Alignas(16) float winRate[LEAGUE_MEMBERS]; float theta[LEAGUE_MEMBERS * SK_THETA]; } skLeague;

typedef struct { skWorld scratch; uint32_t words[SK_WORDS], battle, since, call, mode; } skOracle;

typedef struct { const skirmishKnobs *k; skWorld *w; uint32_t *words; const float *theta; uint32_t now, seed; } skCtx;

static skWorld *sk_world(uint32_t *extra) { return (skWorld *)((uintptr_t)(extra + 4) & ~(uintptr_t)15u); }
static const skWorld *sk_worldAt(const uint32_t *extra) { return (const skWorld *)((uintptr_t)(extra + 4) & ~(uintptr_t)15u); }
static skLeague *sk_league(uint32_t *extra) { return (skLeague *)(sk_world(extra) + 1); }
static skOracle *sk_oracleOf(uint32_t *extra) { return (skOracle *)(sk_league(extra) + 1); }

static inline __attribute__((always_inline)) void sk_closer(const skWorld *w, uint32_t at, uint32_t skip, float px, float py, float4 *best, lane4 *who) {
    float4 x, y, hp;
    memcpy(&x, w->x + at, sizeof x);
    memcpy(&y, w->y + at, sizeof y);
    memcpy(&hp, w->hp + at, sizeof hp);
    float4 dx = x - px, dy = y - py, d2 = dx * dx + dy * dy;
    lane4 slot = at + (lane4){ 0u, 1u, 2u, 3u };
    lane4 closer = (lane4)(hp > 0.0f) & (lane4)(d2 < *best) & (lane4)(slot != skip);
    *best = select4(closer, d2, *best);
    *who = (slot & closer) | (*who & ~closer);
}

static inline __attribute__((always_inline)) uint32_t sk_nearest(const skWorld *w, uint32_t base, uint32_t count, uint32_t skip, float px, float py, float *reach) {
    float4 even = splat4(INFINITY), odd = splat4(INFINITY);
    lane4 first = SK_NONE, second = SK_NONE;
    uint32_t at = base;
    for (; at + 4u < base + count; at += 8u) {
        sk_closer(w, at, skip, px, py, &even, &first);
        sk_closer(w, at + 4u, skip, px, py, &odd, &second);
    }
    if (at < base + count) sk_closer(w, at, skip, px, py, &even, &first);
    lane4 swap = (lane4)(odd < even) | ((lane4)(odd == even) & (lane4)(second < first));
    float4 best = select4(swap, odd, even);
    lane4 who = (second & swap) | (first & ~swap);
    float least = vminvq_f32((float32x4_t)best);
    lane4 tied = (lane4)(best == least);
    *reach = least;
    return vminvq_u32((uint32x4_t)((who & tied) | ~tied));
}

static float sk_d2(const skWorld *w, uint32_t a, uint32_t b) {
    float dx = w->x[a] - w->x[b], dy = w->y[a] - w->y[b];
    return dx * dx + dy * dy;
}

static float sk_health(const skWorld *w, uint32_t u) { return (w->hp[u] + w->shield[u]) * w->inverse[u]; }

static float sk_reach(uint32_t t) { return SKIRMISH_RANGE[t] * SKIRMISH_RANGE[t]; }

static inline __attribute__((always_inline)) uint32_t sk_steps(const skirmishKnobs *k, float seconds) { return (uint32_t)(seconds * (float)k->slices / k->tick + 0.5f); }

static float sk_damage(uint32_t attacker, uint32_t victim) {
    float bonus = SKIRMISH_BONUS_VS[attacker] == 1u + SKIRMISH_ARMORED[victim] ? SKIRMISH_BONUS[attacker] : 0.0f;
    return fmaxf(SKIRMISH_DAMAGE[attacker] + bonus - SKIRMISH_ARMOR[victim], SKIRMISH_RULES[SK_MIN_DAMAGE]);
}

static inline __attribute__((always_inline)) float sk_stride(const skCtx *c, uint32_t u, float seconds) {
    const skWorld *w = c->w;
    return SKIRMISH_SPEED[w->type[u]] * seconds * (c->now < w->slowUntil[u] ? SKIRMISH_RULES[SK_SLOWDOWN] : 1.0f);
}

static void sk_toward(skWorld *w, uint32_t u, float tx, float ty, float step) {
    float dx = tx - w->x[u], dy = ty - w->y[u], d = sqrtf(dx * dx + dy * dy);
    if (d <= step) {
        w->x[u] = tx;
        w->y[u] = ty;
        return;
    }
    float f = step / d;
    w->x[u] += dx * f;
    w->y[u] += dy * f;
}

static void sk_away(const skirmishKnobs *k, skWorld *w, uint32_t u, float fx, float fy, float step) {
    float dx = w->x[u] - fx, dy = w->y[u] - fy, d = sqrtf(dx * dx + dy * dy);
    if (!(d > 0.0f)) {
        dx = 1.0f;
        dy = 0.0f;
        d = 1.0f;
    }
    float f = step / d;
    w->x[u] = oak_clamp(w->x[u] + dx * f, 0.0f, k->arena);
    w->y[u] = oak_clamp(w->y[u] + dy * f, 0.0f, k->arena);
}

static inline __attribute__((always_inline)) void sk_regen(skCtx *c, uint32_t u) {
    skWorld *w = c->w;
    float cap = SKIRMISH_SHIELD[w->type[u]];
    if (w->shield[u] < cap && c->now - w->hitAt[u] >= sk_steps(c->k, SKIRMISH_RULES[SK_REGEN_DELAY]))
        w->shield[u] = fminf(w->shield[u] + SKIRMISH_RULES[SK_REGEN_RATE] * c->k->tick, cap);
}

static void sk_die(skCtx *c, uint32_t v) {
    skWorld *w = c->w;
    float reach;
    w->hp[v] = 0.0f;
    w->shield[v] = 0.0f;
    c->words[v < SK_FOE ? SK_ALLIES : SK_FOES]--;
    if (v == c->words[SK_SELECT]) c->words[SK_SELECT] = sk_nearest(w, SK_FOE, c->k->enemies, SK_NONE, w->x[0], w->y[0], &reach);
    if (v == c->words[SK_CALL]) c->words[SK_CALL] = SK_NONE;
}

static void sk_hurt(skCtx *c, uint32_t v, float amount) {
    skWorld *w = c->w;
    float absorbed = fminf(w->shield[v], amount), hp = w->hp[v] - (amount - absorbed);
    w->shield[v] -= absorbed;
    w->hitAt[v] = c->now;
    if (hp > 0.0f) w->hp[v] = hp;
    else sk_die(c, v);
}

static void sk_blast(skCtx *c, uint32_t u, float px, float py, uint32_t skip) {
    skWorld *w = c->w;
    uint32_t a = w->type[u], base = u < SK_FOE ? SK_FOE : 0u, end = base + (u < SK_FOE ? c->k->enemies : c->k->allies);
    float r2 = SKIRMISH_SPLASH[a] * SKIRMISH_SPLASH[a];
    for (uint32_t v = base; v < end; v++) {
        float dx = w->x[v] - px, dy = w->y[v] - py;
        if (v != skip && w->hp[v] > 0.0f && dx * dx + dy * dy <= r2) sk_hurt(c, v, sk_damage(a, w->type[v]));
    }
}

static void sk_fire(skCtx *c, uint32_t u, uint32_t t) {
    skWorld *w = c->w;
    uint32_t a = w->type[u];
    float px = w->x[t], py = w->y[t];
    w->ready[u] = c->now + sk_steps(c->k, SKIRMISH_COOLDOWN[a]);
    if (SKIRMISH_SUICIDE[a]) {
        sk_blast(c, u, w->x[u], w->y[u], SK_NONE);
        sk_die(c, u);
        return;
    }
    if (SKIRMISH_SLOW[a] > 0.0f) w->slowUntil[t] = c->now + sk_steps(c->k, SKIRMISH_SLOW[a]);
    sk_hurt(c, t, sk_damage(a, w->type[t]));
    if (SKIRMISH_SPLASH[a] > 0.0f) sk_blast(c, u, px, py, t);
}

static void sk_mend(skCtx *c, uint32_t u, float step, float amount) {
    skWorld *w = c->w;
    uint32_t base = u < SK_FOE ? 0u : SK_FOE, count = u < SK_FOE ? c->k->allies : c->k->enemies, best = SK_NONE;
    float low = 1.0f, reach;
    for (uint32_t v = base; v < base + count; v++) {
        float health = w->hp[v] / SKIRMISH_HP[w->type[v]];
        if (v != u && w->hp[v] > 0.0f && health < low) {
            low = health;
            best = v;
        }
    }
    if (best == SK_NONE) {
        uint32_t near = sk_nearest(w, base, count, u, w->x[u], w->y[u], &reach);
        w->target[u] = (uint8_t)near;
        if (near != SK_NONE && reach > SKIRMISH_RULES[SK_FORM] * SKIRMISH_RULES[SK_FORM]) sk_toward(w, u, w->x[near], w->y[near], step);
        return;
    }
    w->target[u] = (uint8_t)best;
    if (sk_d2(w, u, best) <= sk_reach(w->type[u])) w->hp[best] = fminf(w->hp[best] + amount, SKIRMISH_HP[w->type[best]]);
    else sk_toward(w, u, w->x[best], w->y[best], step);
}

static inline __attribute__((always_inline)) uint32_t sk_pick(const skCtx *c, uint32_t u, float range2, float *reach) {
    const skWorld *w = c->w;
    uint32_t base = u < SK_FOE ? SK_FOE : 0u, count = u < SK_FOE ? c->k->enemies : c->k->allies, mask = 0u;
    float px = w->x[u], py = w->y[u];
    for (uint32_t at = 0u; at < count; at += 4u) {
        float4 x, y, hp;
        memcpy(&x, w->x + base + at, sizeof x);
        memcpy(&y, w->y + base + at, sizeof y);
        memcpy(&hp, w->hp + base + at, sizeof hp);
        float4 dx = x - px, dy = y - py, d2 = dx * dx + dy * dy;
        lane4 inside = (lane4)(hp > 0.0f) & (lane4)(d2 <= range2);
        mask |= vaddvq_u32((uint32x4_t)(inside & (lane4){ 1u, 2u, 4u, 8u })) << at;
    }
    if (!mask) return SK_NONE;
    if (mask & (mask - 1u)) {
        uint32_t skip = (uint32_t)(u01(philox4x32(c->seed, ENV_RNG_STREAM, c->words[SK_BATTLE], c->now, u, SK_PICK).x) * (float)__builtin_popcount(mask));
        for (; skip; skip--) mask &= mask - 1u;
    }
    uint32_t v = base + (uint32_t)__builtin_ctz(mask);
    *reach = sk_d2(w, u, v);
    return v;
}

static inline __attribute__((always_inline)) void sk_seek(const skCtx *c, uint32_t u, float step) {
    skWorld *w = c->w;
    uint32_t side = u >= SK_FOE;
    float gx = w->goalX[side], gy = w->goalY[side], dx = gx - w->x[u], dy = gy - w->y[u], arrive = SKIRMISH_RULES[SK_ARRIVE], reach;
    w->target[u] = SK_NONE;
    w->hunt[u] |= (uint8_t)(dx * dx + dy * dy <= arrive * arrive);
    if (w->hunt[u]) {
        uint32_t near = side ? sk_nearest(w, 0u, c->k->allies, SK_NONE, w->x[u], w->y[u], &reach) : sk_nearest(w, SK_FOE, c->k->enemies, SK_NONE, w->x[u], w->y[u], &reach);
        if (near == SK_NONE) return;
        gx = w->x[near];
        gy = w->y[near];
    }
    sk_toward(w, u, gx, gy, step);
}

static inline __attribute__((always_inline)) void sk_bot(skCtx *c, uint32_t u) {
    skWorld *w = c->w;
    const skirmishKnobs *k = c->k;
    uint32_t a = w->type[u], t = w->target[u], call = c->words[SK_CALL], squad = u < SK_FOE && w->hp[0] > 0.0f;
    float step = sk_stride(c, u, k->tick), range2 = sk_reach(a), form = SKIRMISH_RULES[SK_FORM], d2;
    sk_regen(c, u);
    if (SKIRMISH_HEAL[a] > 0.0f) {
        sk_mend(c, u, step, SKIRMISH_HEAL[a] * k->tick);
        return;
    }
    if (squad && w->hp[call] > 0.0f) {
        t = call;
        d2 = sk_d2(w, u, t);
    } else {
        d2 = sk_d2(w, u, t);
        if (!(w->hp[t] > 0.0f) || d2 > range2) {
            t = sk_pick(c, u, range2, &d2);
            if (t == SK_NONE) {
                if (!squad) {
                    sk_seek(c, u, step);
                    return;
                }
                w->target[u] = SK_NONE;
                if (sk_d2(w, u, 0u) > form * form) sk_toward(w, u, w->x[0], w->y[0], step);
                return;
            }
        }
    }
    w->target[u] = (uint8_t)t;
    if (d2 <= range2) {
        if (c->now >= w->ready[u]) sk_fire(c, u, t);
        return;
    }
    sk_toward(w, u, w->x[t], w->y[t], step);
}

static uint32_t sk_choose(const skCtx *c, uint32_t u, float limit, float *reach) {
    const skWorld *w = c->w;
    const float *theta = c->theta;
    uint32_t a = w->type[u], tied = 0u, pool[SK_FOE];
    float inverse = 1.0f / SKIRMISH_SIGHT[a], top = -INFINITY, score[SK_FOE], near;
    sk_nearest(w, 0u, c->k->allies, SK_NONE, w->x[u], w->y[u], &near);
    if (near > limit) return SK_NONE;
    for (uint32_t v = 0u; v < c->k->allies; v++) {
        float d2 = sk_d2(w, u, v);
        uint32_t b = w->type[v];
        score[v] = -INFINITY;
        if (!(w->hp[v] > 0.0f) || d2 > limit) continue;
        score[v] = theta[b % 3u] - theta[SK_WEAK] * sk_health(w, v) - theta[SK_CLOSE] * sqrtf(d2) * inverse + theta[SK_THREATENED] * (float)(w->target[v] == u) +
                   theta[SK_FOCUS] * (float)(v == w->focus) + theta[SK_DANGER] * SKIRMISH_THREAT[b] * 0.05f;
        top = fmaxf(top, score[v]);
    }
    if (top == -INFINITY) return SK_NONE;
    for (uint32_t v = 0u; v < c->k->allies; v++)
        if (score[v] == top) pool[tied++] = v;
    uint32_t pick = tied > 1u ? pool[(uint32_t)(u01(philox4x32(c->seed, ENV_RNG_STREAM, c->words[SK_BATTLE], c->now, u, SK_PICK).x) * (float)tied)] : pool[0];
    *reach = sk_d2(w, u, pick);
    return pick;
}

static void sk_rival(skCtx *c, uint32_t u) {
    skWorld *w = c->w;
    const skirmishKnobs *k = c->k;
    const float *theta = c->theta;
    uint32_t a = w->type[u], t = w->target[u];
    float step = sk_stride(c, u, k->tick), range2 = sk_reach(a), acquire = SKIRMISH_RANGE[a] + theta[SK_ENGAGE] * (SKIRMISH_SIGHT[a] - SKIRMISH_RANGE[a]), hold = acquire * acquire, d2;
    sk_regen(c, u);
    if (SKIRMISH_HEAL[a] > 0.0f) {
        sk_mend(c, u, step, SKIRMISH_HEAL[a] * k->tick);
        return;
    }
    if (sk_health(w, u) < theta[SK_RETREAT]) {
        uint32_t near = sk_nearest(w, 0u, k->allies, SK_NONE, w->x[u], w->y[u], &d2);
        if (near != SK_NONE && d2 <= SKIRMISH_SIGHT[a] * SKIRMISH_SIGHT[a]) {
            sk_away(k, w, u, w->x[near], w->y[near], step);
            return;
        }
    }
    d2 = sk_d2(w, u, t);
    if (!(w->hp[t] > 0.0f) || d2 > hold) {
        t = sk_choose(c, u, hold, &d2);
        if (t == SK_NONE) {
            sk_seek(c, u, step);
            return;
        }
    }
    w->target[u] = (uint8_t)t;
    if (d2 <= range2) {
        if (c->now >= w->ready[u]) {
            w->focus = t;
            sk_fire(c, u, t);
        } else if (SKIRMISH_RANGE[a] > 2.0f && (float)(w->ready[u] - c->now) > theta[SK_KITE] * (float)sk_steps(k, SKIRMISH_COOLDOWN[a]))
            sk_away(k, w, u, w->x[t], w->y[t], step);
        return;
    }
    sk_toward(w, u, w->x[t], w->y[t], step);
}

static uint32_t sk_next(const skirmishKnobs *k, const skWorld *w, uint32_t from) {
    float base = w->hp[from] > 0.0f ? sk_d2(w, 0u, from) : -1.0f, px = w->x[0], py = w->y[0];
    float4 later = splat4(INFINITY), least = splat4(INFINITY);
    lane4 next = SK_NONE, first = SK_NONE;
    for (uint32_t at = SK_FOE; at < SK_FOE + k->enemies; at += 4u) {
        float4 x, y, hp;
        memcpy(&x, w->x + at, sizeof x);
        memcpy(&y, w->y + at, sizeof y);
        memcpy(&hp, w->hp + at, sizeof hp);
        float4 dx = x - px, dy = y - py, d2 = dx * dx + dy * dy;
        lane4 slot = at + (lane4){ 0u, 1u, 2u, 3u }, live = (lane4)(hp > 0.0f);
        lane4 lower = live & (lane4)(d2 < least), beyond = live & ((lane4)(d2 > base) | ((lane4)(d2 == base) & (lane4)(slot > from))) & (lane4)(d2 < later);
        least = select4(lower, d2, least);
        first = (slot & lower) | (first & ~lower);
        later = select4(beyond, d2, later);
        next = (slot & beyond) | (next & ~beyond);
    }
    float bound = vminvq_f32((float32x4_t)later), floor = vminvq_f32((float32x4_t)least);
    lane4 atBound = (lane4)(later == bound), atFloor = (lane4)(least == floor);
    uint32_t after = vminvq_u32((uint32x4_t)((next & atBound) | ~atBound)), nearest = vminvq_u32((uint32x4_t)((first & atFloor) | ~atFloor));
    return after != SK_NONE ? after : nearest != SK_NONE ? nearest : from;
}

static uint32_t sk_action(const float *ctrl) {
    float c = ctrl[0];
    return c >= 0.0f && c < 6.0f ? (uint32_t)c : 0u;
}

static inline __attribute__((always_inline)) void sk_agent(skCtx *c, uint32_t action) {
    skWorld *w = c->w;
    const skirmishKnobs *k = c->k;
    if (!(w->hp[0] > 0.0f)) return;
    uint32_t a = w->type[0], t = c->words[SK_SELECT];
    float seconds = k->tick / (float)k->slices, step = sk_stride(c, 0u, seconds);
    if (c->words[SK_SLICE] == 0u) sk_regen(c, 0u);
    if (action < SK_ATTACK) {
        int dx, dy;
        gridDelta(action, &dx, &dy);
        w->x[0] = oak_clamp(w->x[0] + (float)dx * step, 0.0f, k->arena);
        w->y[0] = oak_clamp(w->y[0] + (float)dy * step, 0.0f, k->arena);
        return;
    }
    if (action == SK_CYCLE) {
        c->words[SK_SELECT] = sk_next(k, w, t);
        return;
    }
    c->words[SK_CALL] = w->hp[t] > 0.0f ? t : SK_NONE;
    if (SKIRMISH_HEAL[a] > 0.0f) {
        sk_mend(c, 0u, step, SKIRMISH_HEAL[a] * seconds);
        return;
    }
    w->target[0] = (uint8_t)t;
    if (!(w->hp[t] > 0.0f)) return;
    if (sk_d2(w, 0u, t) <= sk_reach(a)) {
        if (c->now >= w->ready[0]) sk_fire(c, 0u, t);
        return;
    }
    sk_toward(w, 0u, w->x[t], w->y[t], step);
}

static inline __attribute__((always_inline)) void sk_tick(const skirmishKnobs *k, skWorld *w, uint32_t *words, const float *theta, uint32_t seed, uint32_t action) {
    uint32_t slice = words[SK_SLICE];
    skCtx c = { k, w, words, theta, words[SK_TICK] * k->slices + slice, seed };
    sk_agent(&c, action);
    for (uint32_t u = slice; u < SK_FOE + k->enemies; u += k->slices) {
        if (u == 0u || !(w->hp[u] > 0.0f)) continue;
        if (k->league && u >= SK_FOE) sk_rival(&c, u);
        else sk_bot(&c, u);
    }
    if (++slice == k->slices) {
        slice = 0u;
        words[SK_TICK]++;
    }
    words[SK_SLICE] = slice;
}

static void sk_begin(const skirmishKnobs *k, skWorld *w, const skLeague *g, uint32_t *words, uint32_t seed) {
    uint32_t battle = words[SK_BATTLE], r[SK_DRAWS];
    for (uint32_t b = 0u; b < SK_DRAWS / 4u; b++) {
        u4 q = philox4x32(seed, ENV_RNG_STREAM, battle, b, 0u, 0u);
        r[4u * b] = q.x;
        r[4u * b + 1u] = q.y;
        r[4u * b + 2u] = q.z;
        r[4u * b + 3u] = q.w;
    }
    memset(w, 0, sizeof *w);
    memset(w->target, SK_NONE, sizeof w->target);
    uint32_t race = (uint32_t)(u01(r[0]) * (float)k->races), surround = r[1] < u01Bound(k->surround), at = 2u + k->enemies;
    uint32_t first = u01Bound(SKIRMISH_WEIGHT[0]), second = u01Bound(SKIRMISH_WEIGHT[0] + SKIRMISH_WEIGHT[1]);
    for (uint32_t j = 0u; j < k->enemies; j++) {
        uint32_t d = r[2u + j], type = race * 3u + (d < first ? 0u : d < second ? 1u : 2u);
        w->type[SK_FOE + j] = (uint8_t)type;
        if (j < k->allies) w->type[j] = (uint8_t)type;
    }
    float arena = k->arena, near = SKIRMISH_SPAWN[SK_REFLECT_NEAR], far = SKIRMISH_SPAWN[SK_REFLECT_FAR], mid = 0.5f * arena, half = SKIRMISH_SPAWN[SK_CENTRE];
    if (!surround) {
        for (uint32_t i = 0u; i < k->allies; i++) {
            w->x[i] = uniformRange(r[at++], near, far);
            w->y[i] = uniformRange(r[at++], near, arena - near);
            w->x[SK_FOE + i] = arena - w->x[i];
            w->y[SK_FOE + i] = w->y[i];
        }
        for (uint32_t j = k->allies; j < k->enemies; j++) {
            w->x[SK_FOE + j] = uniformRange(r[at++], arena - far, arena - near);
            w->y[SK_FOE + j] = uniformRange(r[at++], near, arena - near);
        }
    } else {
        for (uint32_t i = 0u; i < k->allies; i++) {
            w->x[i] = mid + uniformRange(r[at++], -half, half);
            w->y[i] = mid + uniformRange(r[at++], -half, half);
        }
        for (uint32_t j = 0u; j < k->enemies; j++) {
            uint32_t draw = r[at++];
            float s = uniformRange(draw, SKIRMISH_SPAWN[SK_RING_NEAR], SKIRMISH_SPAWN[SK_RING_FAR]) * 0.70710678f;
            w->x[SK_FOE + j] = mid + ((draw & 1u) ? s : -s);
            w->y[SK_FOE + j] = mid + ((draw & 2u) ? s : -s);
        }
    }
    for (uint32_t u = 0u; u < SK_FOE + k->enemies; u++) {
        if (u >= k->allies && u < SK_FOE) continue;
        uint32_t side = u >= SK_FOE;
        float share = 1.0f / (float)(side ? k->enemies : k->allies);
        w->hp[u] = SKIRMISH_HP[w->type[u]];
        w->shield[u] = SKIRMISH_SHIELD[w->type[u]];
        w->inverse[u] = 1.0f / (w->hp[u] + w->shield[u]);
        w->goalX[1u - side] += w->x[u] * share;
        w->goalY[1u - side] += w->y[u] * share;
    }
    float reach;
    words[SK_TICK] = 0u;
    words[SK_SLICE] = 0u;
    words[SK_RACE] = race;
    words[SK_START] = surround;
    words[SK_ALLIES] = k->allies;
    words[SK_FOES] = k->enemies;
    words[SK_SELECT] = sk_nearest(w, SK_FOE, k->enemies, SK_NONE, w->x[0], w->y[0], &reach);
    words[SK_CALL] = SK_NONE;
    w->focus = SK_NONE;
    w->pace = SKIRMISH_COOLDOWN[w->type[0]] > 0.0f ? 1.0f / (float)sk_steps(k, SKIRMISH_COOLDOWN[w->type[0]]) : 0.0f;
    if (k->league) words[SK_MEMBER] = leaguePick(g->winRate, philox4x32(seed, LEAGUE_RNG_STREAM, battle, SK_PICK, 0u, 0u).x);
}

static void sk_leagueInit(skLeague *g, uint32_t seed) {
    for (uint32_t i = 0u; i < LEAGUE_MEMBERS; i++) g->winRate[i] = 0.5f;
    for (uint32_t d = 0u; d < SK_THETA; d++) g->theta[d] = SKIRMISH_THETA[d];
    for (uint32_t i = 1u; i < LEAGUE_MEMBERS; i++)
        for (uint32_t d = 0u; d < SK_THETA; d += 4u) {
            u44 r = philox4(seed, LEAGUE_RNG_STREAM, (lane4)(0xFFFFFFF0u + i), d + (lane4){ 0u, 1u, 2u, 3u }, (lane4)0u, (lane4)0u);
            float4 n = normal4(r.x, r.y);
            for (uint32_t j = 0u; j < 4u && d + j < SK_THETA; j++) {
                float value = SKIRMISH_THETA[d + j] + 0.1f * (SKIRMISH_THETA_HIGH[d + j] - SKIRMISH_THETA_LOW[d + j]) * n[j];
                g->theta[SK_THETA * i + d + j] = fminf(fmaxf(value, SKIRMISH_THETA_LOW[d + j]), SKIRMISH_THETA_HIGH[d + j]);
            }
        }
}

static __attribute__((noinline)) float sk_finish(const skirmishKnobs *k, skWorld *w, skLeague *g, uint32_t *words, uint32_t seed) {
    uint32_t allies = words[SK_ALLIES], foes = words[SK_FOES], won = allies && !foes, lost = !allies && foes, drawn = !allies && !foes;
    words[won ? SK_WINS : lost ? SK_LOSSES : drawn ? SK_DRAWN : SK_TIMEOUTS]++;
    if (k->league) {
        leagueScore(g->winRate, words[SK_MEMBER], won ? 1.0f : drawn ? 0.5f : 0.0f);
        leagueRenew(g->winRate, g->theta, SK_THETA, SKIRMISH_THETA_LOW, SKIRMISH_THETA_HIGH, seed, words[SK_BATTLE]);
    }
    words[SK_BATTLE]++;
    sk_begin(k, w, g, words, seed);
    return won ? 1.0f : lost ? -1.0f : drawn ? 0.0f : k->timeoutReward;
}

static inline __attribute__((always_inline)) void sk_observe(const skirmishKnobs *k, const skWorld *w, const uint32_t *words, float *obs) {
    float scale = 1.0f / k->arena, x = w->x[0], y = w->y[0], reach;
    uint32_t a = w->type[0], t = words[SK_SELECT], call = words[SK_CALL], now = words[SK_TICK] * k->slices + words[SK_SLICE];
    uint32_t wait = w->ready[0] > now ? w->ready[0] - now : 0u;
    uint32_t foe = sk_nearest(w, SK_FOE, k->enemies, SK_NONE, x, y, &reach);
    float live = w->hp[t] > 0.0f ? 1.0f : 0.0f, near = w->hp[foe] > 0.0f ? scale : 0.0f, called = w->hp[call] > 0.0f ? scale : 0.0f;
    obs[0] = sk_health(w, 0u);
    obs[1] = (float)wait * w->pace;
    obs[2] = (float)a * 0.125f;
    obs[3] = x * scale;
    obs[4] = y * scale;
    obs[5] = (w->x[t] - x) * scale * live;
    obs[6] = (w->y[t] - y) * scale * live;
    obs[7] = sk_health(w, t);
    obs[8] = (float)w->type[t] * 0.125f * live;
    obs[9] = sk_d2(w, 0u, t) <= sk_reach(a) ? live : 0.0f;
    obs[10] = (w->x[foe] - x) * near;
    obs[11] = (w->y[foe] - y) * near;
    obs[12] = sk_health(w, foe);
    obs[13] = (w->x[call] - x) * called;
    obs[14] = (w->y[call] - y) * called;
    obs[15] = sk_health(w, call);
    obs[16] = (float)words[SK_ALLIES] * (1.0f / (float)k->allies);
    obs[17] = (float)words[SK_FOES] * (1.0f / (float)k->enemies);
}

uint32_t skirmish_extraWords(uint32_t instance, uint32_t steps) {
    (void)instance; (void)steps;
    return 4u + (uint32_t)((sizeof(skWorld) + sizeof(skLeague) + sizeof(skOracle)) / sizeof(uint32_t));
}

void skirmish_reset(uint32_t instance, uint32_t seed, uint32_t *words, uint32_t *extra, uint32_t steps) {
    (void)steps;
    const skirmishKnobs *k = &SKIRMISH_KNOBS[instance];
    skLeague *g = sk_league(extra);
    if (k->league) sk_leagueInit(g, seed);
    for (uint32_t i = 0u; i < SK_WORDS; i++) words[i] = 0u;
    sk_oracleOf(extra)->battle = UINT32_MAX;
    sk_begin(k, sk_world(extra), g, words, seed);
}

static const float *sk_theta(const skirmishKnobs *k, uint32_t *extra, const uint32_t *words) { return k->league ? sk_league(extra)->theta + words[SK_MEMBER] * SK_THETA : 0; }

static inline __attribute__((always_inline)) float sk_play(const skirmishKnobs *k, uint32_t seed, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done) {
    skWorld *w = sk_world(extra);
    float reward = 0.0f;
    sk_tick(k, w, words, sk_theta(k, extra, words), seed, sk_action(ctrl));
    *done = 0u;
    if (!words[SK_ALLIES] || !words[SK_FOES] || words[SK_TICK] >= k->battleTicks) {
        reward = sk_finish(k, w, sk_league(extra), words, seed);
        *done = 1u;
    }
    sk_observe(k, w, words, obs);
    return reward;
}

float skirmish_step(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done, const float *shared) {
    (void)step; (void)shared;
    switch (instance) {
        case 0u: return sk_play(&SKIRMISH_KNOBS[0], seed, ctrl, words, extra, obs, done);
        case 1u: return sk_play(&SKIRMISH_KNOBS[1], seed, ctrl, words, extra, obs, done);
        case 2u: return sk_play(&SKIRMISH_KNOBS[2], seed, ctrl, words, extra, obs, done);
        case 3u: return sk_play(&SKIRMISH_KNOBS[3], seed, ctrl, words, extra, obs, done);
        default: return sk_play(&SKIRMISH_KNOBS[4], seed, ctrl, words, extra, obs, done);
    }
}

void skirmish_observe(uint32_t instance, uint32_t seed, const uint32_t *words, const uint32_t *extra, uint32_t step, float *obs, const float *shared) {
    (void)seed; (void)step; (void)shared;
    sk_observe(&SKIRMISH_KNOBS[instance], sk_worldAt(extra), words, obs);
}

static void sk_centre(const skWorld *w, uint32_t base, uint32_t count, uint32_t skip, float *cx, float *cy) {
    float sx = 0.0f, sy = 0.0f, n = 0.0f;
    for (uint32_t u = base; u < base + count; u++)
        if (u != skip && w->hp[u] > 0.0f) {
            sx += w->x[u];
            sy += w->y[u];
            n += 1.0f;
        }
    *cx = n > 0.0f ? sx / n : w->x[0];
    *cy = n > 0.0f ? sy / n : w->y[0];
}

static uint32_t sk_heading(float dx, float dy, uint32_t now) {
    if (fabsf(dx) < 0.05f && fabsf(dy) < 0.05f) return now & 1u;
    return fabsf(dx) > fabsf(dy) ? (dx < 0.0f ? 2u : 3u) : (dy < 0.0f ? 0u : 1u);
}

static uint32_t sk_plan(const skirmishKnobs *k, const skWorld *w, const uint32_t *words, uint32_t call, uint32_t mode) {
    uint32_t now = words[SK_TICK] * k->slices + words[SK_SLICE], a = w->type[0], t = words[SK_CALL];
    float ax, ay, reach;
    if (!(w->hp[0] > 0.0f)) return SK_ATTACK;
    sk_centre(w, 0u, k->allies, 0u, &ax, &ay);
    uint32_t home = sk_heading(ax - w->x[0], ay - w->y[0], now);
    if (mode == SK_GATHER) return home;
    if (w->hp[call] > 0.0f && t != call) return words[SK_SELECT] != call ? SK_CYCLE : SK_ATTACK;
    if (mode == SK_FIGHT) return SK_ATTACK;
    if (w->hp[t] > 0.0f)
        return SKIRMISH_RANGE[a] > 2.0f && !SKIRMISH_SUICIDE[a] && sk_d2(w, 0u, t) <= sk_reach(a) && now >= w->ready[0] ? SK_ATTACK : home;
    sk_nearest(w, SK_FOE, k->enemies, SK_NONE, ax, ay, &reach);
    return reach <= (float)SK_CONTACT || words[SK_TICK] > SK_PATIENCE ? SK_ATTACK : home;
}

static float sk_rollout(const skirmishKnobs *k, skOracle *o, const skWorld *w, const uint32_t *words, const float *theta, uint32_t seed, uint32_t call, uint32_t mode) {
    skWorld *s = &o->scratch;
    uint32_t *v = o->words;
    float margin = 0.0f;
    memcpy(s, w, sizeof *s);
    memcpy(v, words, sizeof o->words);
    for (uint32_t t = 0u; t < SK_HORIZON * k->slices && v[SK_ALLIES] && v[SK_FOES] && v[SK_TICK] < k->battleTicks; t++) {
        uint32_t lead = t < SK_PERIOD * k->slices;
        sk_tick(k, s, v, theta, seed, sk_plan(k, s, v, lead ? call : SK_NONE, lead ? mode : SK_HOLD));
    }
    if (!v[SK_FOES]) return v[SK_ALLIES] ? 1000.0f : 0.0f;
    if (!v[SK_ALLIES]) return -1000.0f;
    for (uint32_t u = 0u; u < k->allies; u++) margin += s->hp[u] + s->shield[u];
    for (uint32_t u = SK_FOE; u < SK_FOE + k->enemies; u++) margin -= s->hp[u] + s->shield[u];
    return 0.01f * margin;
}

static void sk_decide(const skirmishKnobs *k, skOracle *o, const skWorld *w, const uint32_t *words, const float *theta, uint32_t seed) {
    uint32_t pool[SK_CALLS];
    float rank[SK_CALLS];
    for (uint32_t i = 0u; i < SK_CALLS; i++) {
        pool[i] = SK_NONE;
        rank[i] = -INFINITY;
    }
    for (uint32_t e = SK_FOE; e < SK_FOE + k->enemies; e++) {
        if (!(w->hp[e] > 0.0f)) continue;
        float covered = 0.0f;
        for (uint32_t u = 0u; u < k->allies; u++) covered += (float)(w->hp[u] > 0.0f && sk_d2(w, u, e) <= sk_reach(w->type[u]));
        float score = 10.0f * covered - (w->hp[e] + w->shield[e]) - 0.5f * sqrtf(sk_d2(w, 0u, e));
        for (uint32_t i = 0u; i < SK_CALLS; i++)
            if (score > rank[i]) {
                for (uint32_t j = SK_CALLS - 1u; j > i; j--) {
                    rank[j] = rank[j - 1u];
                    pool[j] = pool[j - 1u];
                }
                rank[i] = score;
                pool[i] = e;
                break;
            }
    }
    float best = -INFINITY;
    for (uint32_t mode = SK_HOLD; mode < SK_MODES; mode++) {
        float value = sk_rollout(k, o, w, words, theta, seed, SK_NONE, mode);
        if (value > best) {
            best = value;
            o->call = SK_NONE;
            o->mode = mode;
        }
    }
    for (uint32_t i = 0u; i < SK_CALLS && pool[i] != SK_NONE; i++) {
        float value = sk_rollout(k, o, w, words, theta, seed, pool[i], SK_HOLD);
        if (value > best) {
            best = value;
            o->call = pool[i];
            o->mode = SK_HOLD;
        }
    }
}

void skirmish_oracle(uint32_t instance, uint32_t seed, uint32_t step, const uint32_t *words, uint32_t *extra, const float *obs, float *ctrl, const float *shared) {
    (void)step; (void)obs; (void)shared;
    const skirmishKnobs *k = &SKIRMISH_KNOBS[instance];
    const skWorld *w = sk_worldAt(extra);
    skOracle *o = sk_oracleOf(extra);
    if (w->hp[0] > 0.0f && (o->battle != words[SK_BATTLE] || ++o->since >= SK_PERIOD * k->slices)) {
        o->battle = words[SK_BATTLE];
        o->since = 0u;
        sk_decide(k, o, w, words, sk_theta(k, extra, words), seed);
    }
    ctrl[0] = (float)sk_plan(k, w, words, o->call, o->mode);
}

static uint32_t sk_fixed(float v) { return (uint32_t)(v * 1024.0f + 0.5f); }

static uint32_t sk_amount(float v) { return (uint32_t)ceilf(v); }

uint32_t skirmish_view(uint32_t instance, const uint32_t *words, const uint32_t *extra, const float *shared, uint32_t *out) {
    (void)shared;
    const skirmishKnobs *k = &SKIRMISH_KNOBS[instance];
    uint32_t size = 3u * (k->allies + k->enemies);
    if (!out) return size;
    const skWorld *w = sk_worldAt(extra);
    uint32_t at = 0u, now = words[SK_TICK] * k->slices + words[SK_SLICE];
    for (uint32_t u = 0u; u < SK_FOE + k->enemies; u++) {
        if (u >= k->allies && u < SK_FOE) continue;
        uint32_t wait = w->ready[u] > now ? w->ready[u] - now : 0u;
        out[at++] = sk_fixed(w->x[u]) | sk_fixed(w->y[u]) << 16;
        out[at++] = sk_amount(w->hp[u]) | sk_amount(w->shield[u]) << 16;
        out[at++] = w->type[u] | (uint32_t)w->target[u] << 4 | (uint32_t)(now < w->slowUntil[u]) << 10 | (uint32_t)w->hunt[u] << 11 | (wait < 255u ? wait : 255u) << 16;
    }
    return size;
}
