#include "../env.h"

#define SLIME_WIDTH 48.0f
#define SLIME_GROUND 1.5f
#define SLIME_FENCE_WIDTH 1.0f
#define SLIME_FENCE_HEIGHT 3.5f
#define SLIME_STUB 0.5f
#define SLIME_RADIUS 1.5f
#define SLIME_BALL 0.5f
#define SLIME_RUN 17.5f
#define SLIME_JUMP 13.5f
#define SLIME_GRAVITY -29.4f
#define SLIME_RATE 30.0f
#define SLIME_MAX_BALL 22.5f
#define SLIME_NUDGE 0.1f
#define SLIME_FRICTION 1.0f
#define SLIME_HOLD 30u
#define SLIME_LIVES 5u
#define SLIME_FRAMES 3000u
#define SLIME_START 12.0f
#define SLIME_SERVE_HEIGHT 9.0f
#define SLIME_SERVE_RUN_LOW 2.0f
#define SLIME_SERVE_RUN_HIGH 14.0f
#define SLIME_SERVE_RISE_LOW 0.0f
#define SLIME_SERVE_RISE_HIGH 12.0f
#define SLIME_PRESS 0.75f
#define SLIME_GENOME 136u
#define SLIME_LEAGUE_SPREAD 0.05f

typedef struct { uint32_t league, positions; } slimeKnobs;

static const slimeKnobs SLIME_KNOBS[3] = {
    { 0u, 0u },
    { 0u, 1u },
    { 1u, 0u }
};

static const float SLIME_EVOLVED[SLIME_GENOME] = {
    -3.0f, 0.505829871f, -2.83221936f, 3.0f, 1.50626481f, 0.0309396982f, 1.42540669f, 0.0f,
    0.304669857f, -0.75759089f, 2.13260722f, 1.18425369f, -0.4802894f, 2.96658945f, -2.22196865f, 0.0f,
    -0.303514987f, 1.19690406f, 0.925290704f, 0.53750217f, 2.88177061f, -3.0f, -0.316731215f, 0.0f,
    -0.572979689f, -0.535895407f, 0.266769558f, -0.263603359f, -0.970085204f, -0.509404063f, 1.70534194f, 0.0f,
    3.0f, -3.0f, 1.70175242f, -3.0f, -1.01558661f, -2.29628587f, 1.84296989f, 0.0f,
    2.27229714f, 0.79048419f, -0.233901978f, -0.155393302f, -0.523564816f, 2.16272068f, -3.0f, 0.0f,
    -0.627031803f, -3.0f, -1.73526502f, -1.68269134f, -3.0f, 3.0f, 2.91829157f, 0.0f,
    -1.75689721f, -0.104072303f, -0.579432487f, -1.73034191f, -1.68310809f, 1.98608971f, -3.0f, 0.0f,
    1.01458871f, 0.0702034086f, -1.03348947f, 0.832310021f, 1.58372259f, -0.0126147866f, 2.04255509f, 0.0f,
    1.83854711f, 0.620455205f, -1.55832195f, 1.68952405f, 2.73556828f, 2.89417148f, -3.0f, 0.0f,
    -0.063046068f, 0.786770701f, 0.266961157f, 0.0330301821f, -0.81526047f, -2.17699575f, 1.21110654f, 0.0f,
    0.781715274f, -0.260573238f, 0.587283432f, 0.639051914f, 0.0626759231f, -1.01008868f, -2.52982736f, 0.0f,
    -0.447568268f, 0.689087868f, 2.001652f, -1.16797721f, -1.48952436f, 3.0f, 1.40036678f, 0.0f,
    -0.651513159f, 2.4335947f, 2.42502451f, -0.693590164f, -0.136282712f, -2.89872456f, 2.44975424f, 0.0f,
    -0.814346313f, 2.34675789f, 1.44369638f, -0.709731758f, -0.444619715f, -0.472236186f, 1.40130663f, 0.0f,
    -0.64405483f, 1.71056879f, 1.45447445f, 0.306433439f, 3.0f, -3.0f, 0.689626217f, 0.0f,
    -1.40949214f, -2.84060335f, 3.0f, -1.10765839f, -0.841825068f, -1.05624032f, 2.18795729f, 0.0f
};

static const float SLIME_DT = 1.0f / SLIME_RATE;
static const float SLIME_HALF = 0.5f * SLIME_WIDTH;
static const float SLIME_INNER = 0.5f * SLIME_FENCE_WIDTH + SLIME_RADIUS;
static const float SLIME_OUTER = 0.5f * SLIME_WIDTH - SLIME_RADIUS;
static const float SLIME_TOP = SLIME_GROUND + SLIME_FENCE_HEIGHT;
static const float SLIME_REACH = SLIME_RADIUS + SLIME_BALL;
static const float SLIME_POST = SLIME_STUB + SLIME_BALL;
static const float SLIME_SCALE_X = 2.0f / SLIME_WIDTH;
static const float SLIME_SCALE_V = 1.0f / SLIME_MAX_BALL;
static const float SLIME_BEHIND = 0.6f;

enum { SLIME_INPUTS = 16, SLIME_UNITS = 8, SLIME_MEMORY = 4 };
enum { SLIME_WORDS = 24 };
enum { SLIME_HOME, SLIME_HIT, SLIME_GUARD };
enum { SLIME_OFFSETS = 8, SLIME_LEAPS = 4, SLIME_CHARGES = 3, SLIME_CHECKED = 24, SLIME_REACH_FRAMES = 150 };

typedef struct {
    _Alignas(16) float ball[4];
    float body[2][4], memory[SLIME_MEMORY];
    uint32_t frame, hold, lives[2], point, game, member, last;
} slimeWorld;

_Static_assert(sizeof(slimeWorld) == SLIME_WORDS * sizeof(uint32_t), "");

typedef struct {
    float winRate[LEAGUE_MEMBERS], theta[LEAGUE_MEMBERS * SLIME_GENOME], low[SLIME_GENOME], high[SLIME_GENOME];
} slimeLeague;

typedef struct {
    float target, jump, charge;
} slimePlan;

typedef struct {
    float next[4], previous;
    slimePlan plan;
    uint32_t key, mode, crossed;
} slimeOracle;

static slimeOracle *slime_oracleOf(uint32_t *extra) { return (slimeOracle *)(((uintptr_t)extra + 15u) & ~(uintptr_t)15u); }
static slimeLeague *slime_league(uint32_t *extra) { return (slimeLeague *)(slime_oracleOf(extra) + 1); }
static const slimeLeague *slime_peek(const uint32_t *extra) { return (const slimeLeague *)((const slimeOracle *)(((uintptr_t)extra + 15u) & ~(uintptr_t)15u) + 1); }

uint32_t slime_extraWords(uint32_t instance, uint32_t steps) {
    (void)steps;
    return (uint32_t)((sizeof(slimeOracle) + (SLIME_KNOBS[instance].league ? sizeof(slimeLeague) : 0u)) / sizeof(uint32_t)) + 4u;
}

static float4 slime_load(const float *p) {
    float4 v;
    __builtin_memcpy(&v, p, sizeof v);
    return v;
}

static void slime_view(const slimeWorld *w, uint32_t player, float4 *out) {
    float4 scale = player ? (float4){ -SLIME_SCALE_X, SLIME_SCALE_X, -SLIME_SCALE_V, SLIME_SCALE_V } : (float4){ SLIME_SCALE_X, SLIME_SCALE_X, SLIME_SCALE_V, SLIME_SCALE_V };
    float4 lift = { 0.0f, SLIME_GROUND, 0.0f, 0.0f };
    out[0] = (slime_load(w->body[player]) - lift) * scale;
    out[1] = (slime_load(w->ball) - lift) * scale;
    out[2] = (slime_load(w->body[player ^ 1u]) - lift) * scale;
}

static uint32_t slime_think(const float *genome, const slimeWorld *w, uint32_t player, float *memory) {
    float4 in[4], low[4], high[4];
    slime_view(w, player, in);
    in[3] = slime_load(memory);
    for (uint32_t k = 0u; k < 4u; k++) low[k] = high[k] = splat4(0.0f);
    for (uint32_t i = 0u; i < SLIME_INPUTS; i++) {
        float4 x = splat4(in[i >> 2][i & 3u]);
        low[i & 3u] = __builtin_elementwise_fma(slime_load(genome + SLIME_UNITS * i), x, low[i & 3u]);
        high[i & 3u] = __builtin_elementwise_fma(slime_load(genome + SLIME_UNITS * i + 4u), x, high[i & 3u]);
    }
    float4 state = slime_load(genome + SLIME_UNITS * SLIME_INPUTS) + ((low[0] + low[1]) + (low[2] + low[3]));
    float4 motor = slime_load(genome + SLIME_UNITS * SLIME_INPUTS + 4u) + ((high[0] + high[1]) + (high[2] + high[3]));
    state = __builtin_elementwise_min(__builtin_elementwise_max(state, splat4(-1.0f)), splat4(1.0f));
    __builtin_memcpy(memory, &state, sizeof state);
    uint32_t forward = motor.x > SLIME_PRESS, backward = motor.y > SLIME_PRESS, jump = motor.z > SLIME_PRESS;
    uint32_t toward = player ? 2u : 1u, away = 3u - toward;
    return (forward && !backward ? toward : backward && !forward ? away : 0u) + 3u * jump;
}

static void slime_move(float *b, uint32_t action, float low, float high) {
    static const float DIRECTION[6] = { 0.0f, -1.0f, 1.0f, 0.0f, -1.0f, 1.0f };
    float vx = DIRECTION[action] * SLIME_RUN, fall = b[3] + SLIME_GRAVITY * SLIME_DT, rise = SLIME_JUMP + SLIME_GRAVITY * SLIME_DT;
    float vy = __builtin_unpredictable((action >= 3u) & (b[1] <= SLIME_GROUND)) ? rise : fall;
    float y = b[1] + vy * SLIME_DT, x = b[0] + vx * SLIME_DT;
    uint32_t landed = y <= SLIME_GROUND, blocked = x < low || x > high;
    b[0] = blocked ? fminf(fmaxf(x, low), high) : x;
    b[1] = landed ? SLIME_GROUND : y;
    b[2] = blocked ? 0.0f : vx;
    b[3] = landed ? 0.0f : vy;
}

static void slime_contact(float *ball, float nx, float ny, float vx, float vy, float x, float y) {
    float ux = ball[2] - vx, uy = ball[3] - vy, un = ux * nx + uy * ny;
    ball[0] = x;
    ball[1] = y;
    if (un < 0.0f) {
        ball[2] = vx + SLIME_FRICTION * (ux - 2.0f * un * nx);
        ball[3] = vy + SLIME_FRICTION * (uy - 2.0f * un * ny);
    }
}

static void slime_hit(float *ball, const float *b) {
    float dx = ball[0] - b[0], dy = ball[1] - b[1];
    if (dy >= 0.0f) {
        float d2 = dx * dx + dy * dy;
        if (d2 >= SLIME_REACH * SLIME_REACH) return;
        float d = sqrtf(d2), nx = d > 0.0f ? dx / d : 0.0f, ny = d > 0.0f ? dy / d : 1.0f;
        slime_contact(ball, nx, ny, b[2], b[3], b[0] + nx * (SLIME_REACH + SLIME_NUDGE), b[1] + ny * (SLIME_REACH + SLIME_NUDGE));
        return;
    }
    float cx = fminf(fmaxf(dx, -SLIME_RADIUS), SLIME_RADIUS), qx = dx - cx, d2 = qx * qx + dy * dy;
    if (d2 >= SLIME_BALL * SLIME_BALL) return;
    float d = sqrtf(d2), nx = qx / d, ny = dy / d;
    slime_contact(ball, nx, ny, b[2], b[3], b[0] + cx + nx * (SLIME_BALL + SLIME_NUDGE), b[1] + ny * (SLIME_BALL + SLIME_NUDGE));
}

static void slime_fence(float *ball) {
    float cy = fminf(fmaxf(ball[1], SLIME_GROUND), SLIME_TOP), dx = ball[0], dy = ball[1] - cy, d2 = dx * dx + dy * dy;
    if (d2 >= SLIME_POST * SLIME_POST) return;
    float d = sqrtf(d2), nx = d > 0.0f ? dx / d : ball[2] > 0.0f ? -1.0f : 1.0f, ny = d > 0.0f ? dy / d : 0.0f;
    slime_contact(ball, nx, ny, 0.0f, 0.0f, nx * (SLIME_POST + SLIME_NUDGE), cy + ny * (SLIME_POST + SLIME_NUDGE));
}

static void slime_flight(float *ball) {
    ball[3] += SLIME_GRAVITY * SLIME_DT;
    float s2 = ball[2] * ball[2] + ball[3] * ball[3];
    if (s2 > SLIME_MAX_BALL * SLIME_MAX_BALL) {
        float k = SLIME_MAX_BALL / sqrtf(s2);
        ball[2] *= k;
        ball[3] *= k;
    }
    ball[0] += ball[2] * SLIME_DT;
    ball[1] += ball[3] * SLIME_DT;
}

static uint32_t slime_bounds(float *ball) {
    slime_fence(ball);
    if (ball[0] < SLIME_BALL - SLIME_HALF) slime_contact(ball, 1.0f, 0.0f, 0.0f, 0.0f, SLIME_BALL - SLIME_HALF + SLIME_NUDGE, ball[1]);
    if (ball[0] > SLIME_HALF - SLIME_BALL) slime_contact(ball, -1.0f, 0.0f, 0.0f, 0.0f, SLIME_HALF - SLIME_BALL - SLIME_NUDGE, ball[1]);
    return ball[1] - SLIME_BALL <= SLIME_GROUND;
}

static uint32_t slime_ball(slimeWorld *w) {
    slime_flight(w->ball);
    slime_hit(w->ball, w->body[1]);
    slime_hit(w->ball, w->body[0]);
    return slime_bounds(w->ball);
}

static void slime_rally(slimeWorld *w) {
    w->ball[0] = 0.0f;
    w->ball[1] = SLIME_GROUND + SLIME_SERVE_HEIGHT;
    w->ball[2] = 0.0f;
    w->ball[3] = 0.0f;
    w->hold = SLIME_HOLD;
}

static void slime_serve(slimeWorld *w, uint32_t seed) {
    u4 r = philox4x32(seed, ENV_RNG_STREAM, w->game, w->point, 0u, 0u);
    w->ball[2] = (r.z >> 31 ? 1.0f : -1.0f) * uniformRange(r.x, SLIME_SERVE_RUN_LOW, SLIME_SERVE_RUN_HIGH);
    w->ball[3] = uniformRange(r.y, SLIME_SERVE_RISE_LOW, SLIME_SERVE_RISE_HIGH);
}

static void slime_begin(slimeWorld *w) {
    for (uint32_t p = 0u; p < 2u; p++) {
        w->body[p][0] = p ? -SLIME_START : SLIME_START;
        w->body[p][1] = SLIME_GROUND;
        w->body[p][2] = 0.0f;
        w->body[p][3] = 0.0f;
        w->lives[p] = SLIME_LIVES;
    }
    for (uint32_t u = 0u; u < SLIME_MEMORY; u++) w->memory[u] = 0.0f;
    w->frame = 0u;
    w->point = 0u;
    w->last = 0u;
    slime_rally(w);
}

static float slime_frame(slimeWorld *w, uint32_t seed, uint32_t action, uint32_t press, uint32_t *over) {
    slime_move(w->body[0], action, SLIME_INNER, SLIME_OUTER);
    slime_move(w->body[1], press, -SLIME_OUTER, -SLIME_INNER);
    float reward = 0.0f;
    if (w->hold) {
        if (--w->hold == 0u) slime_serve(w, seed);
    } else if (slime_ball(w)) {
        uint32_t loser = w->ball[0] < 0.0f;
        reward = loser ? 1.0f : -1.0f;
        w->lives[loser]--;
        w->point++;
        w->last = 2u - loser;
        if (w->lives[loser]) slime_rally(w);
        else *over = 1u;
    }
    if (++w->frame >= SLIME_FRAMES) *over = 1u;
    return reward;
}

static uint32_t slime_action(const float *ctrl) {
    float c = ctrl[0];
    return c >= 0.0f && c < 6.0f ? (uint32_t)c : 0u;
}

static const float *slime_brain(const slimeKnobs *k, const slimeWorld *w, const uint32_t *extra) {
    return k->league ? slime_peek(extra)->theta + SLIME_GENOME * w->member : SLIME_EVOLVED;
}

static void slime_pick(slimeLeague *l, slimeWorld *w, uint32_t seed) {
    w->member = leaguePick(l->winRate, philox4x32(seed, LEAGUE_RNG_STREAM, w->game, 0x5049434Bu, 0u, 0u).x);
}

static void slime_seedLeague(slimeLeague *l, uint32_t seed) {
    for (uint32_t d = 0u; d < SLIME_GENOME; d++) {
        float spread = d % SLIME_UNITS == SLIME_UNITS - 1u ? 0.0f : SLIME_LEAGUE_SPREAD;
        l->low[d] = SLIME_EVOLVED[d] - spread;
        l->high[d] = SLIME_EVOLVED[d] + spread;
        l->theta[d] = SLIME_EVOLVED[d];
    }
    for (uint32_t i = 0u; i < LEAGUE_MEMBERS; i++) l->winRate[i] = 0.5f;
    for (uint32_t i = 1u; i < LEAGUE_MEMBERS; i++)
        for (uint32_t d = 0u; d < SLIME_GENOME; d += 4u) {
            u44 r = philox4(seed, LEAGUE_RNG_STREAM, (lane4)(0xFFFFFFF0u + i), d + (lane4){ 0u, 1u, 2u, 3u }, (lane4)0u, (lane4)0u);
            float4 n = normal4(r.x, r.y);
            for (uint32_t j = 0u; j < 4u; j++) {
                float value = SLIME_EVOLVED[d + j] + 0.1f * (l->high[d + j] - l->low[d + j]) * n[j];
                l->theta[SLIME_GENOME * i + d + j] = fminf(fmaxf(value, l->low[d + j]), l->high[d + j]);
            }
        }
}

static __attribute__((noinline)) void slime_finish(const slimeKnobs *k, slimeWorld *w, uint32_t *extra, uint32_t seed) {
    if (k->league) {
        slimeLeague *l = slime_league(extra);
        float outcome = w->lives[0] > w->lives[1] ? 1.0f : w->lives[0] < w->lives[1] ? 0.0f : 0.5f;
        leagueScore(l->winRate, w->member, outcome);
        leagueRenew(l->winRate, l->theta, SLIME_GENOME, l->low, l->high, seed, w->game);
    }
    w->game++;
    slime_begin(w);
    if (k->league) slime_pick(slime_league(extra), w, seed);
}

static void slime_obs(const slimeKnobs *k, const slimeWorld *w, float *obs) {
    float4 view[3];
    slime_view(w, 0u, view);
    if (!k->positions) {
        __builtin_memcpy(obs, view, sizeof view);
        return;
    }
    obs[0] = view[0].x; obs[1] = view[0].y; obs[2] = view[1].x; obs[3] = view[1].y; obs[4] = view[2].x; obs[5] = view[2].y;
}

void slime_reset(uint32_t instance, uint32_t seed, uint32_t *words, uint32_t *extra, uint32_t steps) {
    (void)steps;
    const slimeKnobs *k = &SLIME_KNOBS[instance];
    slimeWorld w;
    __builtin_memset(&w, 0, sizeof w);
    slime_begin(&w);
    slimeOracle *o = slime_oracleOf(extra);
    o->key = 0xFFFFFFFFu;
    if (k->league) {
        slime_seedLeague(slime_league(extra), seed);
        slime_pick(slime_league(extra), &w, seed);
    }
    __builtin_memcpy(words, &w, sizeof w);
}

float slime_step(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done, const float *shared) {
    (void)step; (void)shared;
    const slimeKnobs *k = &SLIME_KNOBS[instance];
    slimeWorld w;
    __builtin_memcpy(&w, words, sizeof w);
    uint32_t press = slime_think(slime_brain(k, &w, extra), &w, 1u, w.memory), over = 0u;
    float reward = slime_frame(&w, seed, slime_action(ctrl), press, &over);
    *done = over;
    if (over) slime_finish(k, &w, extra, seed);
    __builtin_memcpy(words, &w, sizeof w);
    slime_obs(k, &w, obs);
    return reward;
}

void slime_observe(uint32_t instance, uint32_t seed, const uint32_t *words, const uint32_t *extra, uint32_t step, float *obs, const float *shared) {
    (void)seed; (void)extra; (void)step; (void)shared;
    slimeWorld w;
    __builtin_memcpy(&w, words, sizeof w);
    slime_obs(&SLIME_KNOBS[instance], &w, obs);
}

static uint32_t slime_steer(float x, float target, uint32_t jump) {
    float gap = target - x, half = 0.5f * SLIME_RUN * SLIME_DT;
    return (gap > half ? 2u : gap < -half ? 1u : 0u) + 3u * jump;
}

static uint32_t slime_rule(const slimePlan *p, float x, uint32_t frame) {
    float at = (float)frame;
    return at >= p->charge ? 1u + 3u * (at == p->jump) : slime_steer(x, p->target, at == p->jump);
}

static float slime_glide(const float *from, uint32_t *frames, uint32_t mine) {
    float ball[4] = { from[0], from[1], from[2], from[3] };
    for (uint32_t f = 1u; f <= SLIME_REACH_FRAMES; f++) {
        slime_flight(ball);
        if (slime_bounds(ball) || (mine && ball[0] >= 0.0f && ball[3] < 0.0f && ball[1] <= SLIME_GROUND + SLIME_REACH)) {
            *frames = f;
            return ball[0];
        }
    }
    *frames = SLIME_REACH_FRAMES;
    return ball[0];
}

static int slime_trial(const slimeWorld *w, const slimePlan *p, float *margin) {
    float ball[4] = { w->ball[0], w->ball[1], w->ball[2], w->ball[3] }, body[4] = { w->body[0][0], w->body[0][1], w->body[0][2], w->body[0][3] };
    for (uint32_t f = 0u; f < SLIME_REACH_FRAMES; f++) {
        slime_move(body, slime_rule(p, body[0], w->frame + f), SLIME_INNER, SLIME_OUTER);
        slime_flight(ball);
        slime_hit(ball, body);
        if (slime_bounds(ball)) return -1;
        if (ball[0] < 0.0f) {
            uint32_t frames;
            float land = slime_glide(ball, &frames, 0u);
            float need = fabsf(land - w->body[1][0]) / (SLIME_RUN * SLIME_DT);
            *margin = land < 0.0f ? need - (float)(f + 1u + frames) : -1000.0f;
            return 1;
        }
    }
    return 0;
}

static int slime_verify(const slimeWorld *from, const float *genome, uint32_t seed, const slimePlan *p) {
    slimeWorld w = *from;
    uint32_t crossed = 0u, over = 0u;
    for (uint32_t f = 0u; f < 2u * SLIME_REACH_FRAMES; f++) {
        if (w.ball[0] < 0.0f) crossed = 1u;
        uint32_t a = crossed ? slime_steer(w.body[0][0], SLIME_START, 0u) : slime_rule(p, w.body[0][0], w.frame);
        uint32_t press = slime_think(genome, &w, 1u, w.memory);
        float reward = slime_frame(&w, seed, a, press, &over);
        if (reward != 0.0f) return reward > 0.0f ? 1 : -1;
        if (over || (crossed && w.ball[0] >= 0.0f)) return 0;
    }
    return 0;
}

static void slime_plan(const slimeKnobs *k, const slimeWorld *w, const uint32_t *extra, uint32_t seed, slimeOracle *o) {
    static const float OFFSET[SLIME_OFFSETS] = { -0.3f, 0.0f, 0.3f, 0.6f, 0.9f, 1.2f, 1.5f, 1.8f };
    static const float LEAD[SLIME_LEAPS] = { -1.0f, 4.0f, 8.0f, 12.0f };
    static const float RUSH[SLIME_CHARGES] = { -1.0f, 3.0f, 6.0f };
    uint32_t frames;
    float land = slime_glide(w->ball, &frames, 1u), now = (float)w->frame, arrival = now + (float)frames;
    slimePlan kept[SLIME_CHECKED];
    float margins[SLIME_CHECKED];
    uint32_t count = 0u;
    o->mode = SLIME_HIT;
    o->crossed = 0u;
    o->plan = (slimePlan){ fminf(fmaxf(land + SLIME_BEHIND, SLIME_INNER), SLIME_OUTER), -1.0f, 1e9f };
    for (uint32_t i = 0u; i < SLIME_OFFSETS; i++)
        for (uint32_t j = 0u; j < SLIME_LEAPS; j++)
            for (uint32_t c = 0u; c < SLIME_CHARGES; c++) {
                slimePlan p = { fminf(fmaxf(land + OFFSET[i], SLIME_INNER), SLIME_OUTER), LEAD[j] < 0.0f ? -1.0f : fmaxf(arrival - LEAD[j], now), RUSH[c] < 0.0f ? 1e9f : fmaxf(arrival - RUSH[c], now) };
                float margin = 0.0f;
                if (slime_trial(w, &p, &margin) != 1) continue;
                uint32_t at = count;
                if (count < SLIME_CHECKED) count++;
                else {
                    at = 0u;
                    for (uint32_t m = 1u; m < SLIME_CHECKED; m++) if (margins[m] < margins[at]) at = m;
                    if (margin <= margins[at]) continue;
                }
                kept[at] = p;
                margins[at] = margin;
            }
    if (!count) return;
    const float *genome = slime_brain(k, w, extra);
    float best = -1e30f;
    for (uint32_t c = 0u; c < count; c++) {
        float value = 10000.0f * (float)slime_verify(w, genome, seed, &kept[c]) + margins[c];
        if (value > best) {
            best = value;
            o->plan = kept[c];
        }
    }
}

static void slime_decide(const slimeKnobs *k, const slimeWorld *w, const uint32_t *extra, uint32_t seed, slimeOracle *o) {
    uint32_t frames;
    float x = slime_glide(w->ball, &frames, 1u);
    int ours = x >= 0.0f;
    if (w->ball[0] > 0.0f || (w->ball[0] == 0.0f && w->ball[2] > 0.0f)) {
        if (ours) slime_plan(k, w, extra, seed, o);
        else o->mode = SLIME_HOME;
        return;
    }
    o->mode = ours ? SLIME_GUARD : SLIME_HOME;
    o->plan = (slimePlan){ fminf(fmaxf(x + SLIME_BEHIND, SLIME_INNER), SLIME_OUTER), -1.0f, 1e9f };
}

void slime_oracle(uint32_t instance, uint32_t seed, uint32_t step, const uint32_t *words, uint32_t *extra, const float *obs, float *ctrl, const float *shared) {
    (void)step; (void)obs; (void)shared;
    const slimeKnobs *k = &SLIME_KNOBS[instance];
    slimeOracle *o = slime_oracleOf(extra);
    slimeWorld w;
    __builtin_memcpy(&w, words, sizeof w);
    uint32_t key = w.game << 4 | w.point;
    if (o->key != key || w.hold) {
        o->key = key;
        o->mode = SLIME_HOME;
        o->previous = 0.0f;
    } else {
        uint32_t entered = o->previous < 0.0f && w.ball[0] >= 0.0f, changed = __builtin_memcmp(o->next, w.ball, sizeof o->next) != 0;
        if (entered || (changed && !(o->mode == SLIME_HIT && w.ball[0] >= 0.0f))) slime_decide(k, &w, extra, seed, o);
    }
    if (o->mode == SLIME_HIT && w.ball[0] < 0.0f) o->crossed = 1u;
    uint32_t hit = o->mode == SLIME_HIT && !o->crossed;
    ctrl[0] = (float)(hit ? slime_rule(&o->plan, w.body[0][0], w.frame) : slime_steer(w.body[0][0], o->mode == SLIME_GUARD ? o->plan.target : SLIME_START, 0u));
    o->previous = w.ball[0];
    __builtin_memcpy(o->next, w.ball, sizeof o->next);
    if (!w.hold) {
        slime_flight(o->next);
        slime_bounds(o->next);
    }
}
