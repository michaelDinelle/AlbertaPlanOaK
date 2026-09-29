#include "policy.h"
#include <string.h>

#define STREAM_H 128u
#define STREAM_IN_MAX ((OAK_ENV_OBS_MAX + 3u) & ~3u)
#define STREAM_HEADS_MAX (2u * OAK_ENV_ACT_MAX)
#define STREAM_HEADS_PAD ((STREAM_HEADS_MAX + 3u) & ~3u)
#define STREAM_W2 0u
#define STREAM_B1 (STREAM_W2 + STREAM_H * STREAM_H)
#define STREAM_B2 (STREAM_B1 + STREAM_H)
#define STREAM_B3 (STREAM_B2 + STREAM_H)
#define STREAM_W3 (STREAM_B3 + STREAM_HEADS_PAD)
#define STREAM_NET_MAX (STREAM_W3 + STREAM_HEADS_MAX * STREAM_H + STREAM_IN_MAX * STREAM_H)
#define STREAM_ALPHA 1.0f
#define STREAM_GAMMA 0.99f
#define STREAM_LAMBDA 0.8f
#define STREAM_DECAY (STREAM_GAMMA * STREAM_LAMBDA)
#define STREAM_KAPPA_CRITIC 2.0f
#define STREAM_KAPPA_ACTOR 3.0f
#define STREAM_ENTROPY 0.01f
#define STREAM_SPARSE_TENTHS 9u
#define STREAM_LEAK 0.01f
#define STREAM_LAYER_EPS 1e-5f
#define STREAM_NORM_EPS 1e-8f
#define STREAM_SOFTPLUS_LINEAR 20.0f
#define STREAM_TAG 1000u

typedef struct { float4 a, b, c, d; } StreamSum;

typedef struct {
    _Alignas(16) float n1[STREAM_H];
    _Alignas(16) float a1[STREAM_H];
    _Alignas(16) float n2[STREAM_H];
    _Alignas(16) float a2[STREAM_H];
    _Alignas(16) float out[STREAM_HEADS_PAD];
    float inv1, inv2;
} StreamPass;

typedef struct {
    _Alignas(16) float p[STREAM_NET_MAX];
    _Alignas(16) float z[STREAM_NET_MAX];
    StreamPass pass;
    uint32_t heads, count;
    float kappa;
} StreamNet;

typedef struct {
    StreamNet critic, actor;
    _Alignas(16) float x[STREAM_IN_MAX];
    _Alignas(16) float obsMean[STREAM_IN_MAX];
    _Alignas(16) float obsSquares[STREAM_IN_MAX];
    float action[OAK_ENV_ACT_MAX];
    float returnTrace, returnMean, returnSquares;
    uint32_t obsCount, returnCount, obs, in, actions, box;
} StreamAgent;

static inline float4 streamLoad(const float *p) { float4 v; memcpy(&v, p, sizeof v); return v; }
static inline void streamStore(float *p, float4 v) { memcpy(p, &v, sizeof v); }
static inline float4 streamFma(float4 a, float4 b, float4 c) { return __builtin_elementwise_fma(a, b, c); }
static inline float4 streamAbs(float4 v) { return (float4)((lane4)v & 0x7fffffffu); }
static inline float streamTotal(float4 v) { return (v.x + v.y) + (v.z + v.w); }
static inline uint32_t streamW1(uint32_t heads) { return STREAM_W3 + heads * STREAM_H; }

static inline float streamSoftplus(float x) {
    float e = oak_exp(x), w = 1.0f + e;
    return x > STREAM_SOFTPLUS_LINEAR ? x : w == 1.0f ? e : oak_log(w) * (e / (w - 1.0f));
}

static inline float streamSoftplusSlope(float x) {
    float e = oak_exp(x);
    return x > STREAM_SOFTPLUS_LINEAR ? 1.0f : e / (1.0f + e);
}

static void streamLogSoftmax(const float *logits, uint32_t count, float *logp) {
    float top = logits[0];
    for (uint32_t a = 1u; a < count; a++) top = fmaxf(top, logits[a]);
    float total = 0.0f;
    for (uint32_t a = 0u; a < count; a++) total += oak_exp(logits[a] - top);
    float spread = oak_log(total);
    for (uint32_t a = 0u; a < count; a++) logp[a] = (logits[a] - top) - spread;
}

static void streamSparse(float *w, uint32_t units, uint32_t fanIn, uint32_t unitStride, uint32_t inputStride, uint32_t seed, uint32_t tag) {
    float bound = sqrtf(1.0f / (float)fanIn);
    uint32_t zeros = (STREAM_SPARSE_TENTHS * fanIn + 9u) / 10u;
    for (uint32_t j = 0u; j < units; j++) {
        uint32_t words[2u * STREAM_H], order[STREAM_H];
        for (uint32_t b = 0u; b < fanIn + zeros; b += 4u) {
            u4 r = philox4x32(seed, RNG_STREAM_PPO_INIT, j, tag, b >> 2, 0u);
            words[b] = r.x;
            words[b + 1u] = r.y;
            words[b + 2u] = r.z;
            words[b + 3u] = r.w;
        }
        for (uint32_t i = 0u; i < fanIn; i++) {
            order[i] = i;
            w[j * unitStride + i * inputStride] = uniformRange(words[i], -bound, bound);
        }
        for (uint32_t k = 0u; k < zeros; k++) {
            uint32_t pick = k + (uint32_t)(((uint64_t)words[fanIn + k] * (fanIn - k)) >> 32);
            uint32_t held = order[k];
            order[k] = order[pick];
            order[pick] = held;
            w[j * unitStride + order[k] * inputStride] = 0.0f;
        }
    }
}

static void streamBuild(StreamNet *net, uint32_t heads, float kappa, uint32_t obs, uint32_t in, uint32_t seed, uint32_t tag) {
    net->heads = heads;
    net->count = streamW1(heads) + in * STREAM_H;
    net->kappa = kappa;
    streamSparse(net->p + streamW1(heads), STREAM_H, obs, 1u, STREAM_H, seed, tag);
    streamSparse(net->p + STREAM_W2, STREAM_H, STREAM_H, 1u, STREAM_H, seed, tag + 1u);
    streamSparse(net->p + STREAM_W3, heads, STREAM_H, STREAM_H, 1u, seed, tag + 2u);
}

static void streamAffine(const float *w, const float *bias, const float *x, uint32_t in, float *h) {
    for (uint32_t c = 0u; c < STREAM_H; c += 32u) {
        float4 acc[8];
        for (uint32_t j = 0u; j < 8u; j++) acc[j] = streamLoad(bias + c + 4u * j);
        for (uint32_t i = 0u; i < in; i++) {
            float4 xi = splat4(x[i]);
            const float *row = w + i * STREAM_H + c;
            for (uint32_t j = 0u; j < 8u; j++) acc[j] = streamFma(xi, streamLoad(row + 4u * j), acc[j]);
        }
        for (uint32_t j = 0u; j < 8u; j++) streamStore(h + c + 4u * j, acc[j]);
    }
}

static void streamNorm(const float *h, float *n, float *a, float *inv) {
    float4 total = splat4(0.0f), spread = splat4(0.0f);
    for (uint32_t c = 0u; c < STREAM_H; c += 4u) total += streamLoad(h + c);
    float mean = streamTotal(total) / (float)STREAM_H;
    for (uint32_t c = 0u; c < STREAM_H; c += 4u) {
        float4 gap = streamLoad(h + c) - mean;
        spread = streamFma(gap, gap, spread);
    }
    float scale = 1.0f / sqrtf(streamTotal(spread) / (float)STREAM_H + STREAM_LAYER_EPS);
    for (uint32_t c = 0u; c < STREAM_H; c += 4u) {
        float4 v = (streamLoad(h + c) - mean) * scale;
        streamStore(n + c, v);
        streamStore(a + c, select4((lane4)(v > 0.0f), v, v * STREAM_LEAK));
    }
    *inv = scale;
}

static float streamDot(const float *a, const float *b) {
    float4 s0 = splat4(0.0f), s1 = splat4(0.0f), s2 = splat4(0.0f), s3 = splat4(0.0f);
    for (uint32_t c = 0u; c < STREAM_H; c += 16u) {
        s0 = streamFma(streamLoad(a + c), streamLoad(b + c), s0);
        s1 = streamFma(streamLoad(a + c + 4u), streamLoad(b + c + 4u), s1);
        s2 = streamFma(streamLoad(a + c + 8u), streamLoad(b + c + 8u), s2);
        s3 = streamFma(streamLoad(a + c + 12u), streamLoad(b + c + 12u), s3);
    }
    return streamTotal((s0 + s1) + (s2 + s3));
}

static void streamForward(const StreamNet *net, uint32_t in, const float *x, StreamPass *s) {
    _Alignas(16) float h[STREAM_H];
    streamAffine(net->p + streamW1(net->heads), net->p + STREAM_B1, x, in, h);
    streamNorm(h, s->n1, s->a1, &s->inv1);
    streamAffine(net->p + STREAM_W2, net->p + STREAM_B2, s->a1, STREAM_H, h);
    streamNorm(h, s->n2, s->a2, &s->inv2);
    for (uint32_t k = 0u; k < net->heads; k++) s->out[k] = net->p[STREAM_B3 + k] + streamDot(net->p + STREAM_W3 + k * STREAM_H, s->a2);
}

static void streamNormBack(const float *n, float inv, const float *da, float *dh) {
    float4 total = splat4(0.0f), cross = splat4(0.0f);
    for (uint32_t c = 0u; c < STREAM_H; c += 4u) {
        float4 v = streamLoad(n + c);
        float4 d = streamLoad(da + c) * select4((lane4)(v > 0.0f), splat4(1.0f), splat4(STREAM_LEAK));
        streamStore(dh + c, d);
        total += d;
        cross = streamFma(d, v, cross);
    }
    float mean = streamTotal(total) / (float)STREAM_H, slope = streamTotal(cross) / (float)STREAM_H;
    for (uint32_t c = 0u; c < STREAM_H; c += 4u) streamStore(dh + c, (streamLoad(dh + c) - mean - streamLoad(n + c) * slope) * inv);
}

static inline void streamAxpy(float *y, const float *w, float g) {
    float4 gg = splat4(g);
    for (uint32_t c = 0u; c < STREAM_H; c += 4u) streamStore(y + c, streamFma(gg, streamLoad(w + c), streamLoad(y + c)));
}

static inline void streamTrace(float *z, const float *v, float g, StreamSum *m) {
    float4 gg = splat4(g), decay = splat4(STREAM_DECAY);
    for (uint32_t c = 0u; c < STREAM_H; c += 16u) {
        float4 t0 = streamFma(gg, streamLoad(v + c), decay * streamLoad(z + c));
        float4 t1 = streamFma(gg, streamLoad(v + c + 4u), decay * streamLoad(z + c + 4u));
        float4 t2 = streamFma(gg, streamLoad(v + c + 8u), decay * streamLoad(z + c + 8u));
        float4 t3 = streamFma(gg, streamLoad(v + c + 12u), decay * streamLoad(z + c + 12u));
        streamStore(z + c, t0);
        streamStore(z + c + 4u, t1);
        streamStore(z + c + 8u, t2);
        streamStore(z + c + 12u, t3);
        m->a += streamAbs(t0);
        m->b += streamAbs(t1);
        m->c += streamAbs(t2);
        m->d += streamAbs(t3);
    }
}

static inline void streamTraceBias(float *z, const float *g, uint32_t count, StreamSum *m) {
    float4 decay = splat4(STREAM_DECAY);
    for (uint32_t c = 0u; c < count; c += 4u) {
        float4 t = streamFma(decay, streamLoad(z + c), streamLoad(g + c));
        streamStore(z + c, t);
        m->a += streamAbs(t);
    }
}

static inline float streamBackRow(const float *w, float *z, const float *dh, float a, StreamSum *m) {
    float4 gg = splat4(a), decay = splat4(STREAM_DECAY);
    float4 s0 = splat4(0.0f), s1 = splat4(0.0f), s2 = splat4(0.0f), s3 = splat4(0.0f);
    for (uint32_t c = 0u; c < STREAM_H; c += 16u) {
        float4 d0 = streamLoad(dh + c), d1 = streamLoad(dh + c + 4u), d2 = streamLoad(dh + c + 8u), d3 = streamLoad(dh + c + 12u);
        s0 = streamFma(streamLoad(w + c), d0, s0);
        s1 = streamFma(streamLoad(w + c + 4u), d1, s1);
        s2 = streamFma(streamLoad(w + c + 8u), d2, s2);
        s3 = streamFma(streamLoad(w + c + 12u), d3, s3);
        float4 t0 = streamFma(gg, d0, decay * streamLoad(z + c));
        float4 t1 = streamFma(gg, d1, decay * streamLoad(z + c + 4u));
        float4 t2 = streamFma(gg, d2, decay * streamLoad(z + c + 8u));
        float4 t3 = streamFma(gg, d3, decay * streamLoad(z + c + 12u));
        streamStore(z + c, t0);
        streamStore(z + c + 4u, t1);
        streamStore(z + c + 8u, t2);
        streamStore(z + c + 12u, t3);
        m->a += streamAbs(t0);
        m->b += streamAbs(t1);
        m->c += streamAbs(t2);
        m->d += streamAbs(t3);
    }
    return streamTotal((s0 + s1) + (s2 + s3));
}

static void streamStep(float *p, float *z, uint32_t count, float scale, uint32_t done) {
    float4 k = splat4(scale);
    if (done) {
        for (uint32_t c = 0u; c < count; c += 4u) {
            streamStore(p + c, streamFma(k, streamLoad(z + c), streamLoad(p + c)));
            streamStore(z + c, splat4(0.0f));
        }
    } else {
        for (uint32_t c = 0u; c < count; c += 4u) streamStore(p + c, streamFma(k, streamLoad(z + c), streamLoad(p + c)));
    }
}

static void streamLearn(StreamNet *net, uint32_t in, const float *x, const float *grad, float delta, uint32_t done) {
    float *p = net->p, *z = net->z;
    const StreamPass *s = &net->pass;
    _Alignas(16) float da[STREAM_H], dh[STREAM_H];
    StreamSum m = { splat4(0.0f), splat4(0.0f), splat4(0.0f), splat4(0.0f) };
    for (uint32_t c = 0u; c < STREAM_H; c += 4u) streamStore(da + c, splat4(0.0f));
    for (uint32_t k = 0u; k < net->heads; k++) {
        streamAxpy(da, p + STREAM_W3 + k * STREAM_H, grad[k]);
        streamTrace(z + STREAM_W3 + k * STREAM_H, s->a2, grad[k], &m);
    }
    streamTraceBias(z + STREAM_B3, grad, STREAM_HEADS_PAD, &m);
    streamNormBack(s->n2, s->inv2, da, dh);
    streamTraceBias(z + STREAM_B2, dh, STREAM_H, &m);
    for (uint32_t i = 0u; i < STREAM_H; i++) da[i] = streamBackRow(p + STREAM_W2 + i * STREAM_H, z + STREAM_W2 + i * STREAM_H, dh, s->a1[i], &m);
    streamNormBack(s->n1, s->inv1, da, dh);
    streamTraceBias(z + STREAM_B1, dh, STREAM_H, &m);
    float *z1 = z + streamW1(net->heads);
    for (uint32_t i = 0u; i < in; i++) streamTrace(z1 + i * STREAM_H, dh, x[i], &m);
    float magnitude = streamTotal((m.a + m.b) + (m.c + m.d));
    float bound = fmaxf(fabsf(delta), 1.0f) * magnitude * STREAM_ALPHA * net->kappa;
    float stepSize = bound > 1.0f ? STREAM_ALPHA / bound : STREAM_ALPHA;
    streamStep(p, z, net->count, -stepSize * delta, done);
}

static void streamNormalize(StreamAgent *g, const float *obs, float *x) {
    g->obsCount += 1u;
    float count = (float)g->obsCount;
    for (uint32_t d = 0u; d < g->obs; d++) {
        float gap = obs[d] - g->obsMean[d];
        g->obsMean[d] += gap / count;
        g->obsSquares[d] += gap * (obs[d] - g->obsMean[d]);
        float variance = g->obsCount < 2u ? 1.0f : g->obsSquares[d] / (count - 1.0f);
        x[d] = (obs[d] - g->obsMean[d]) / sqrtf(variance + STREAM_NORM_EPS);
    }
}

static float streamScale(StreamAgent *g, float reward, uint32_t done) {
    g->returnTrace = (done ? 0.0f : g->returnTrace * STREAM_GAMMA) + reward;
    g->returnCount += 1u;
    float count = (float)g->returnCount, gap = g->returnTrace - g->returnMean;
    g->returnMean += gap / count;
    g->returnSquares += gap * (g->returnTrace - g->returnMean);
    float variance = g->returnCount < 2u ? 1.0f : g->returnSquares / (count - 1.0f);
    return reward / sqrtf(variance + STREAM_NORM_EPS);
}

static void streamPolicyGradient(const StreamAgent *g, float delta, float *grad) {
    const float *out = g->actor.pass.out;
    uint32_t actions = g->actions;
    float bonus = STREAM_ENTROPY * (delta > 0.0f ? 1.0f : delta < 0.0f ? -1.0f : 0.0f);
    if (g->box) {
        for (uint32_t d = 0u; d < actions; d++) {
            float pre = out[actions + d], inv = 1.0f / streamSoftplus(pre), gap = g->action[d] - out[d];
            grad[d] = -gap * inv * inv;
            grad[actions + d] = ((1.0f - bonus) * inv - gap * gap * inv * inv * inv) * streamSoftplusSlope(pre);
        }
        return;
    }
    float logp[OAK_ENV_ACT_MAX];
    streamLogSoftmax(out, actions, logp);
    float entropy = 0.0f;
    for (uint32_t a = 0u; a < actions; a++) entropy -= oak_exp(logp[a]) * logp[a];
    uint32_t chosen = (uint32_t)g->action[0];
    for (uint32_t a = 0u; a < actions; a++) {
        float prob = oak_exp(logp[a]);
        grad[a] = prob - (a == chosen ? 1.0f : 0.0f) + bonus * prob * (logp[a] + entropy);
    }
}

static void streamSample(StreamAgent *g, uint32_t seed, uint32_t step, float *ctrl) {
    const float *out = g->actor.pass.out;
    uint32_t actions = g->actions;
    if (g->box) {
        _Alignas(16) float noise[8];
        for (uint32_t b = 0u; b < actions; b += 4u) {
            u4 r0 = philox4x32(seed, RNG_STREAM_PPO_ACT, step, STREAM_TAG + b, 0u, 0u), r1 = philox4x32(seed, RNG_STREAM_PPO_ACT, step, STREAM_TAG + b + 1u, 0u, 0u);
            streamStore(noise + b, normalFrom4(r0, r1));
        }
        for (uint32_t d = 0u; d < actions; d++) {
            float sample = out[d] + streamSoftplus(out[actions + d]) * noise[d];
            g->action[d] = sample;
            ctrl[d] = oak_clamp(sample, -1.0f, 1.0f);
        }
        return;
    }
    float logp[OAK_ENV_ACT_MAX];
    streamLogSoftmax(out, actions, logp);
    float u = u01(philox4x32(seed, RNG_STREAM_PPO_ACT, step, STREAM_TAG, 0u, 0u).x), cdf = 0.0f;
    uint32_t chosen = actions - 1u;
    for (uint32_t a = 0u; a < actions; a++) {
        cdf += oak_exp(logp[a]);
        if (u < cdf) {
            chosen = a;
            break;
        }
    }
    g->action[0] = (float)chosen;
    ctrl[0] = (float)chosen;
    for (uint32_t d = 1u; d < actions; d++) ctrl[d] = 0.0f;
}

uint32_t stream_stateWords(void) { return (uint32_t)((sizeof(StreamAgent) + sizeof(uint32_t) - 1u) / sizeof(uint32_t)); }

void stream_init(uint32_t *state, uint32_t lane, uint32_t env, uint32_t seed, const float *obs) {
    (void)lane;
    (void)obs;
    StreamAgent *g = (StreamAgent *)state;
    memset(g, 0, sizeof *g);
    g->obs = env_obs(env);
    g->actions = env_actions(env);
    g->box = env_box(env);
    g->in = (g->obs + 3u) & ~3u;
    streamBuild(&g->critic, 1u, STREAM_KAPPA_CRITIC, g->obs, g->in, seed, STREAM_TAG);
    streamBuild(&g->actor, g->box ? 2u * g->actions : g->actions, STREAM_KAPPA_ACTOR, g->obs, g->in, seed, STREAM_TAG + 4u);
}

void stream_act(uint32_t *state, uint32_t lane, uint32_t env, uint32_t seed, uint32_t step, const float *obs, float reward, uint32_t done, float *ctrl) {
    (void)lane;
    (void)env;
    StreamAgent *g = (StreamAgent *)state;
    _Alignas(16) float x[STREAM_IN_MAX] = { 0.0f };
    streamNormalize(g, obs, x);
    if (step > 0u) {
        float scaled = streamScale(g, finiteOrZero(reward), done);
        float next = 0.0f;
        if (!done) {
            StreamPass probe;
            streamForward(&g->critic, g->in, x, &probe);
            next = probe.out[0];
        }
        float delta = scaled + STREAM_GAMMA * next - g->critic.pass.out[0];
        _Alignas(16) float criticGrad[STREAM_HEADS_PAD] = { -1.0f }, actorGrad[STREAM_HEADS_PAD] = { 0.0f };
        streamPolicyGradient(g, delta, actorGrad);
        streamLearn(&g->actor, g->in, g->x, actorGrad, delta, done);
        streamLearn(&g->critic, g->in, g->x, criticGrad, delta, done);
    }
    memcpy(g->x, x, sizeof x);
    streamForward(&g->critic, g->in, g->x, &g->critic.pass);
    streamForward(&g->actor, g->in, g->x, &g->actor.pass);
    streamSample(g, seed, step, ctrl);
}
