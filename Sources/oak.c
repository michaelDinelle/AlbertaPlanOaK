#include "geometry.h"
#include "policy.h"

_Static_assert(OAK_LANES == 1u, "");
_Static_assert(OAK_POLICY_MAX == 6u, "");
_Static_assert(OAK_UNITS % 8u == 0u, "");
_Static_assert(OAK_FANIN_BLOCKS == 8u, "");
_Static_assert(OAK_FEATURE_MAX % 8u == 0u, "");
_Static_assert(OAK_TUPLES == 23u, "");
_Static_assert(OAK_PERMS * OAK_PERM_STRIDE * sizeof(float) <= 65535u, "");

static inline float4 oak_load4(const float *p) { float4 v; __builtin_memcpy(&v, p, sizeof v); return v; }
static inline float4 oak_load4_off(const float *p, uint16_t off) { float4 v; __builtin_memcpy(&v, (const char *)p + off, sizeof v); return v; }
static inline void oak_store4(float *p, float4 v) { __builtin_memcpy(p, &v, sizeof v); }
static inline lane4 oak_finite4(float4 v) { return (lane4)vcaltq_f32((float32x4_t)v, vdupq_n_f32(INFINITY)); }
static inline float oak_sum4(float4 v) { float32x4_t p = vpaddq_f32((float32x4_t)v, (float32x4_t)v); return vget_lane_f32(vpadd_f32(vget_low_f32(p), vget_low_f32(p)), 0); }
static inline float4 oak_keepFinite4(float4 oldv, float4 newv) { return select4(oak_finite4(newv), newv, oldv); }
static inline float4 oak_finiteOrZero4(float4 v) { return select4(oak_finite4(v), v, splat4(0.0f)); }
static inline float4 oak_rsqrt4(float4 x) {
    float32x4_t v = (float32x4_t)x, r = vrsqrteq_f32(v);
    r = vmulq_f32(r, vrsqrtsq_f32(vmulq_f32(v, r), r)); r = vmulq_f32(r, vrsqrtsq_f32(vmulq_f32(v, r), r));
    return select4((lane4)(x > splat4(0.0f)), (float4)r, splat4(0.0f));
}
static inline float4 oak_rsqrt4_pos(float4 x) { float32x4_t v = (float32x4_t)x, r = vrsqrteq_f32(v); return (float4)vmulq_f32(r, vrsqrtsq_f32(vmulq_f32(v, r), r)); }
static inline float4 oak_rsqrt4_fast(float4 x) { return select4((lane4)(x > splat4(0.0f)), oak_rsqrt4_pos(x), splat4(0.0f)); }
static inline float4 oak_rcp4_fast(float4 x) { float32x4_t v = (float32x4_t)x, r = vrecpeq_f32(v); return (float4)vmulq_f32(r, vrecpsq_f32(v, r)); }
static inline float4 oak_hsum4(float4 a, float4 b, float4 c, float4 d) { return (float4)vpaddq_f32(vpaddq_f32((float32x4_t)a, (float32x4_t)b), vpaddq_f32((float32x4_t)c, (float32x4_t)d)); }
static inline float4 oak_tanh4(float4 x) {
    float4 a = __builtin_elementwise_abs(x), x2 = a * a;
    float4 num = a * __builtin_elementwise_fma(x2, __builtin_elementwise_fma(x2, __builtin_elementwise_fma(x2, splat4(1.0f), splat4(378.0f)), splat4(17325.0f)), splat4(135135.0f));
    float4 den = __builtin_elementwise_fma(x2, __builtin_elementwise_fma(x2, __builtin_elementwise_fma(x2, splat4(28.0f), splat4(3150.0f)), splat4(62370.0f)), splat4(135135.0f));
    float4 t = select4((lane4)(a > splat4(5.0f)), splat4(1.0f), num * oak_rcp4_fast(den));
    return select4((lane4)(x != x), x, (float4)(((lane4)t) | (((lane4)x) & 0x80000000u)));
}

typedef struct { float mean, second, alpha, beta, trace, energy; } Channel;
typedef struct { float q[OAK_POLICY_MAX], pad, value, policy[OAK_CELL_STRIDE]; } CellRow;
typedef struct { float q[OAK_POLICY_MAX]; uint32_t owner; uint16_t visits[OAK_POLICY_MAX]; } TableRowHead;
typedef struct { TableRowHead h; float modelReward[OAK_POLICY_MAX]; uint16_t modelNext[OAK_POLICY_MAX][2], predCell[OAK_DYNA_PREDS], predMeta; uint8_t modelProb[OAK_POLICY_MAX][3], modelTag[OAK_POLICY_MAX][2]; } ExactRow;
typedef struct { TableRowHead h; uint8_t distance[OAK_POLICY_MAX], pad[18]; } HistRow;
_Static_assert(sizeof(CellRow) == 64u && sizeof(ExactRow) == 128u && sizeof(HistRow) == 64u, "");

typedef struct {
    float feature[2u * OAK_FEATURE_MAX], permInput[2u * OAK_PERMS * OAK_PERM_STRIDE];
    float valueWeight[OAK_FEATURE_MAX], valueTrace[OAK_FEATURE_MAX];
    float policy[OAK_HEADS * OAK_POLICY_MAX * OAK_FEATURE_MAX], policyTrace[OAK_POLICY_MAX * OAK_FEATURE_MAX];
    float projection[OAK_UNITS * OAK_FANIN], unitEnergy[2u * OAK_UNITS], unitGate[OAK_UNITS];
    uint16_t inputPerm[OAK_PERMS * OAK_SOURCES_MAX], unitBlock[OAK_UNITS * OAK_FANIN_BLOCKS];
    float traceLevel[OAK_TRACE_LEVELS * OAK_OBSERVATION_LIMIT], traceMean[OAK_TRACE_LEVELS * OAK_OBSERVATION_LIMIT], traceSecond[OAK_TRACE_LEVELS * OAK_OBSERVATION_LIMIT];
    float settledMean[OAK_TRACE_LEVELS * OAK_OBSERVATION_LIMIT], settledSecond[OAK_TRACE_LEVELS * OAK_OBSERVATION_LIMIT], control[OAK_OBSERVATION_LIMIT * OAK_CELL_STRIDE];
    uint32_t cellActive[2u * OAK_CELLS];
    float chMean[OAK_OBSERVATION_LIMIT], chSecond[OAK_OBSERVATION_LIMIT], chAlpha[OAK_OBSERVATION_LIMIT], chBeta[OAK_OBSERVATION_LIMIT], chTrace[OAK_OBSERVATION_LIMIT], chEnergy[OAK_OBSERVATION_LIMIT], chCorr[OAK_OBSERVATION_LIMIT];
    Channel rewardChannel, actorMeta[OAK_HEADS];
    float headMean[OAK_ARMS], headSquare[OAK_ARMS], headWeight[OAK_ARMS], blockReward, obsCorr, prevCorr;
    uint32_t headBlocks[OAK_ARMS], activeHead, learner, blockSteps, blockCount, self;
    float actionSignal[OAK_ACTION_PARAMS], actorCredit[OAK_POLICY_MAX], actorCreditStaging[OAK_POLICY_MAX], logit[OAK_POLICY_MAX];
    float reward, td, tdNormalized, tdInv, tdSquare, advantage, advantageCenter, averageReward, actorStep, betaFloor;
    float rewardMean, rewardVariance, aspiration, rewardSquare;
    float projBound, valBound, polBound[OAK_HEADS], pTraceBound, cellBound, traceBound, featBound[2];
    OakGeometry g;
    uint32_t observationCount, actionCount, box, stepCount, slot, sources, permStride;
    uint32_t qCell[2u * OAK_KEYS], qTag[2u * OAK_KEYS], action, qCellAct, qTried[OAK_KEYS], qOpen[OAK_KEYS];
    float qSum[OAK_KEYS], qNoise[OAK_KEYS], qCellValue[OAK_POLICY_MAX], qCellTaken, qCellGain, qCellNoise, qCellStep;
    uint32_t psTag[OAK_DYNA_QUEUE], psCount, planGeneration, planLength, planIndex;
    float psPrio[OAK_DYNA_QUEUE];
    uint16_t psCell[OAK_DYNA_QUEUE];
    _Alignas(128) CellRow cell[OAK_TABLE];
    _Alignas(128) ExactRow exactTable[OAK_EXACT];
    _Alignas(128) HistRow histTable[OAK_HIST];
    uint16_t planQueue[OAK_PLAN_BUDGET], planCells[OAK_PLAN_BUDGET], planParent[OAK_EXACT];
    uint8_t planActions[OAK_PLAN_BUDGET], planParentAction[OAK_EXACT];
    uint32_t planStamp[OAK_EXACT];
    _Alignas(128) float cellPolicyBank[OAK_HEADS][OAK_TABLE][OAK_CELL_STRIDE];
} OakAgent;

enum { SCALE_TD, SCALE_RAW, SCALE_MAGNITUDE };
enum { TABLE_GREEDY, TABLE_FRONTIER, TABLE_PLAN, TABLE_CELLQ };
static const uint8_t HEAD_SCALE[OAK_HEADS] = { SCALE_TD, SCALE_RAW, SCALE_RAW }, HEAD_NATURAL[OAK_HEADS] = { 0u, 1u, 0u }, HEAD_ASPIRE[OAK_HEADS] = { 1u, 1u, 0u }, HEAD_TRUST[OAK_HEADS] = { 0u, 0u, 1u };
static const float HEAD_DECAY[OAK_HEADS] = { 0.5f, 0.75f, 0.875f };
static const uint8_t TABLE_KEY[OAK_TABLE_ARMS] = { 0u, 1u, 0u, 1u, 0u }, TABLE_MODE[OAK_TABLE_ARMS] = { TABLE_GREEDY, TABLE_GREEDY, TABLE_PLAN, TABLE_FRONTIER, TABLE_CELLQ };

static inline float *features(OakAgent *a, uint32_t slot) { return a->feature + slot * a->g.features; }
static inline float *permInputs(OakAgent *a, uint32_t slot) { return a->permInput + slot * OAK_PERMS * a->permStride; }
static inline float *headPolicy(OakAgent *a) { return a->policy + a->learner * OAK_POLICY_MAX * a->g.features; }
static inline TableRowHead *tableRow(OakAgent *a, uint32_t key, uint32_t cell) { return key == 0u ? &a->exactTable[cell].h : &a->histTable[cell].h; }
static inline const TableRowHead *tableRowConst(const OakAgent *a, uint32_t key, uint32_t cell) { return key == 0u ? &a->exactTable[cell].h : &a->histTable[cell].h; }
static inline uint8_t tag8(uint32_t tag) { return (uint8_t)(((tag ^ (tag >> 8u) ^ (tag >> 16u) ^ (tag >> 24u)) & 0xffu) | 1u); }
static inline float trackingRateFor(uint32_t step, uint32_t cap) { return fmaxf(1.0f / ((float)step + 1.0f), 1.0f / (float)(cap > 1u ? cap : 1u)); }

static inline float dotProduct(const float *restrict x, const float *restrict y, uint32_t n) {
    float4 s0 = splat4(0.0f), s1 = s0, s2 = s0, s3 = s0; uint32_t k = 0u;
    for (; k + 16u <= n; k += 16u) {
        s0 = __builtin_elementwise_fma(oak_load4(x + k), oak_load4(y + k), s0); s1 = __builtin_elementwise_fma(oak_load4(x + k + 4u), oak_load4(y + k + 4u), s1);
        s2 = __builtin_elementwise_fma(oak_load4(x + k + 8u), oak_load4(y + k + 8u), s2); s3 = __builtin_elementwise_fma(oak_load4(x + k + 12u), oak_load4(y + k + 12u), s3);
    }
    for (; k < n; k += 4u) s0 = __builtin_elementwise_fma(oak_load4(x + k), oak_load4(y + k), s0);
    return oak_sum4((s0 + s1) + (s2 + s3));
}
static inline void dotProduct2(const float *restrict w, const float *restrict x, const float *restrict y, uint32_t n, float *restrict ox, float *restrict oy) {
    float4 a0 = splat4(0.0f), a1 = a0, a2 = a0, a3 = a0, b0 = a0, b1 = a0, b2 = a0, b3 = a0; uint32_t k = 0u;
    for (; k + 16u <= n; k += 16u) {
        float4 w0 = oak_load4(w + k), w1 = oak_load4(w + k + 4u), w2 = oak_load4(w + k + 8u), w3 = oak_load4(w + k + 12u);
        a0 = __builtin_elementwise_fma(w0, oak_load4(x + k), a0); b0 = __builtin_elementwise_fma(w0, oak_load4(y + k), b0);
        a1 = __builtin_elementwise_fma(w1, oak_load4(x + k + 4u), a1); b1 = __builtin_elementwise_fma(w1, oak_load4(y + k + 4u), b1);
        a2 = __builtin_elementwise_fma(w2, oak_load4(x + k + 8u), a2); b2 = __builtin_elementwise_fma(w2, oak_load4(y + k + 8u), b2);
        a3 = __builtin_elementwise_fma(w3, oak_load4(x + k + 12u), a3); b3 = __builtin_elementwise_fma(w3, oak_load4(y + k + 12u), b3);
    }
    for (; k < n; k += 4u) { float4 w0 = oak_load4(w + k); a0 = __builtin_elementwise_fma(w0, oak_load4(x + k), a0); b0 = __builtin_elementwise_fma(w0, oak_load4(y + k), b0); }
    *ox = oak_sum4((a0 + a1) + (a2 + a3)); *oy = oak_sum4((b0 + b1) + (b2 + b3));
}
static inline void dotRows2(const float *restrict r0, const float *restrict r1, const float *restrict x, uint32_t n, float *restrict o0, float *restrict o1) {
    float4 a0 = splat4(0.0f), a1 = a0, a2 = a0, a3 = a0, b0 = a0, b1 = a0, b2 = a0, b3 = a0; uint32_t k = 0u;
    for (; k + 16u <= n; k += 16u) {
        float4 x0 = oak_load4(x + k), x1 = oak_load4(x + k + 4u), x2 = oak_load4(x + k + 8u), x3 = oak_load4(x + k + 12u);
        a0 = __builtin_elementwise_fma(oak_load4(r0 + k), x0, a0); b0 = __builtin_elementwise_fma(oak_load4(r1 + k), x0, b0);
        a1 = __builtin_elementwise_fma(oak_load4(r0 + k + 4u), x1, a1); b1 = __builtin_elementwise_fma(oak_load4(r1 + k + 4u), x1, b1);
        a2 = __builtin_elementwise_fma(oak_load4(r0 + k + 8u), x2, a2); b2 = __builtin_elementwise_fma(oak_load4(r1 + k + 8u), x2, b2);
        a3 = __builtin_elementwise_fma(oak_load4(r0 + k + 12u), x3, a3); b3 = __builtin_elementwise_fma(oak_load4(r1 + k + 12u), x3, b3);
    }
    for (; k < n; k += 4u) { float4 x0 = oak_load4(x + k); a0 = __builtin_elementwise_fma(oak_load4(r0 + k), x0, a0); b0 = __builtin_elementwise_fma(oak_load4(r1 + k), x0, b0); }
    *o0 = oak_sum4((a0 + a1) + (a2 + a3)); *o1 = oak_sum4((b0 + b1) + (b2 + b3));
}
static inline void dotRows4(const float *restrict r0, const float *restrict r1, const float *restrict r2, const float *restrict r3, const float *restrict x, uint32_t n, float *restrict o0, float *restrict o1, float *restrict o2, float *restrict o3) {
    float4 a0 = splat4(0.0f), a1 = a0, b0 = a0, b1 = a0, c0 = a0, c1 = a0, d0 = a0, d1 = a0;
    for (uint32_t k = 0u; k < n; k += 8u) {
        float4 x0 = oak_load4(x + k), x1 = oak_load4(x + k + 4u);
        a0 = __builtin_elementwise_fma(oak_load4(r0 + k), x0, a0); b0 = __builtin_elementwise_fma(oak_load4(r1 + k), x0, b0);
        c0 = __builtin_elementwise_fma(oak_load4(r2 + k), x0, c0); d0 = __builtin_elementwise_fma(oak_load4(r3 + k), x0, d0);
        a1 = __builtin_elementwise_fma(oak_load4(r0 + k + 4u), x1, a1); b1 = __builtin_elementwise_fma(oak_load4(r1 + k + 4u), x1, b1);
        c1 = __builtin_elementwise_fma(oak_load4(r2 + k + 4u), x1, c1); d1 = __builtin_elementwise_fma(oak_load4(r3 + k + 4u), x1, d1);
    }
    float4 s = oak_hsum4(a0 + a1, b0 + b1, c0 + c1, d0 + d1); *o0 = s.x; *o1 = s.y; *o2 = s.z; *o3 = s.w;
}
static inline __attribute__((always_inline)) void metaStep(Channel *c, float signal, float inverseAge, float metaRate, float decay, float betaFloor) {
    float mg = signal * c->trace, ge = keepFinite(c->energy, c->energy + inverseAge * (mg * mg - c->energy));
    float nb = keepFinite(c->beta, oak_clamp(c->beta + metaRate * (mg * oak_rsqrt(fmaxf(ge, 0.0f) + inverseAge)), betaFloor, 0.0f));
    float na = keepFinite(c->alpha, oak_exp(nb)), nh = keepFinite(c->trace, (1.0f - decay) * fmaxf(0.0f, 1.0f - c->alpha) * c->trace + c->alpha * signal);
    if (isfinite(signal)) { c->alpha = na; c->beta = nb; c->trace = nh; c->energy = ge; }
}
static inline __attribute__((always_inline)) float standardizeChannel(Channel *c, float x, float inverseAge, float floorRate, float metaRate, float decay, float betaFloor, float ridgeFloor) {
    int finite = isfinite(x); if (finite) metaStep(c, x - c->mean, inverseAge, metaRate, decay, betaFloor);
    float rate = fmaxf(floorRate, c->alpha);
    if (finite) { c->mean = keepFinite(c->mean, c->mean + rate * (x - c->mean)); c->second = keepFinite(c->second, c->second + rate * (x * x - c->second)); }
    return (x - c->mean) * oak_rsqrt(fmaxf(c->second - c->mean * c->mean, 0.0f) + ridgeFloor);
}
static inline uint32_t tableCell(uint32_t hash, uint32_t bits) { hash ^= hash >> 15u; hash *= 0x2C1B3C6Du; hash ^= hash >> 12u; return hash >> (32u - bits); }
static inline uint32_t cellCode(float value, uint32_t fine) { return fine ? (uint32_t)oak_clamp(floorf(value + 3.5f), 0.0f, 6.0f) : (uint32_t)oak_clamp(floorf(value + 1.5f), 0.0f, 2.0f); }
static inline lane4 cellCode4(float4 value, uint32_t fine) { return (lane4)vcvtq_u32_f32((float32x4_t)__builtin_elementwise_min(__builtin_elementwise_max(__builtin_elementwise_floor(value + splat4(fine ? 3.5f : 1.5f)), splat4(0.0f)), splat4(fine ? 6.0f : 2.0f))); }
static inline uint32x4_t mixHash4(uint32x4_t h) { h = veorq_u32(h, vshrq_n_u32(h, 15)); h = vmulq_n_u32(h, 0x2C1B3C6Du); return vshrq_n_u32(veorq_u32(h, vshrq_n_u32(h, 12)), 32 - OAK_TABLE_BITS); }
static inline uint32x4_t fnvStep4(uint32x4_t h, uint32x4_t a, uint32x4_t b, uint32x4_t c, uint32x4_t d) {
    uint32x4_t ab0 = vzip1q_u32(a, b), ab1 = vzip2q_u32(a, b), cd0 = vzip1q_u32(c, d), cd1 = vzip2q_u32(c, d);
    uint32x4_t t0 = vreinterpretq_u32_u64(vzip1q_u64(vreinterpretq_u64_u32(ab0), vreinterpretq_u64_u32(cd0))), t1 = vreinterpretq_u32_u64(vzip2q_u64(vreinterpretq_u64_u32(ab0), vreinterpretq_u64_u32(cd0)));
    uint32x4_t t2 = vreinterpretq_u32_u64(vzip1q_u64(vreinterpretq_u64_u32(ab1), vreinterpretq_u64_u32(cd1))), t3 = vreinterpretq_u32_u64(vzip2q_u64(vreinterpretq_u64_u32(ab1), vreinterpretq_u64_u32(cd1)));
    h = vmulq_n_u32(veorq_u32(h, t0), 0x01000193u); h = vmulq_n_u32(veorq_u32(h, t1), 0x01000193u);
    h = vmulq_n_u32(veorq_u32(h, t2), 0x01000193u); return vmulq_n_u32(veorq_u32(h, t3), 0x01000193u);
}

static __attribute__((noinline, cold)) void rescanTraceBound(OakAgent *a) {
    uint32_t obsN = a->observationCount, n = OAK_TRACE_LEVELS * obsN; float m = 0.0f;
    for (uint32_t i = 0u; i < n; i++) {
        m = fmaxf(m, fmaxf(fabsf(a->traceLevel[i]), fmaxf(fabsf(a->traceMean[i]), fabsf(a->settledMean[i]))));
        m = fmaxf(m, sqrtf(fmaxf(fabsf(a->traceSecond[i]), fabsf(a->settledSecond[i]))));
    }
    for (uint32_t o = 0u; o < obsN; o++) { m = fmaxf(m, sqrtf(fabsf(a->chCorr[o]))); for (uint32_t d = 0u; d < a->actionCount; d++) m = fmaxf(m, fabsf(a->control[o * OAK_CELL_STRIDE + d])); }
    a->traceBound = m;
}
static __attribute__((noinline, cold)) void encodeTraceSlowLoop(OakAgent *a, uint32_t obsN, uint32_t traceTotal, float momentRate, float settleRate, float ridge, uint32_t *restrict fineCode, uint32_t *restrict coarseCode, uint32_t *restrict settledCode, uint32_t *restrict histOut) {
    uint32_t history = *histOut;
    for (uint32_t c = 0u; c < traceTotal; c++) {
        float val = a->traceLevel[c], sq = val * val;
        float m = keepFinite(a->traceMean[c], fmaf(momentRate, val - a->traceMean[c], a->traceMean[c])), s = keepFinite(a->traceSecond[c], fmaf(momentRate, sq - a->traceSecond[c], a->traceSecond[c]));
        float ct = keepFinite(a->settledMean[c], fmaf(settleRate, val - a->settledMean[c], a->settledMean[c])), sp = keepFinite(a->settledSecond[c], fmaf(settleRate, sq - a->settledSecond[c], a->settledSecond[c]));
        a->traceMean[c] = m; a->traceSecond[c] = s; a->settledMean[c] = ct; a->settledSecond[c] = sp;
        uint32_t fc = cellCode(finiteOrZero((val - m) * oak_rsqrt(fmaxf(fmaf(-m, m, s), 0.0f) + ridge)), 1u);
        fineCode[obsN + c] = fc; coarseCode[obsN + c] = (fc <= 2u ? 0u : fc >= 4u ? 2u : 1u) + 8u;
        history = (history ^ fc) * 0x01000193u;
        float sd = val - ct, sv = fmaxf(fmaf(-ct, ct, sp), 0.0f) + ridge; uint32_t sc = 9u;
        if (isfinite(sd) && isfinite(sv)) { if (sd < 0.0f && sd * sd > 0.25f * sv) sc = 8u; else if (sd > 0.0f && sd * sd >= 0.25f * sv) sc = 10u; }
        settledCode[c] = sc;
    }
    *histOut = history; rescanTraceBound(a);
}
static __attribute__((noinline, cold)) void encodeCtrlSlow(OakAgent *a, const float *restrict x, const float *restrict previous, float rate, float track) {
    float maxCorr = 0.0f;
    for (uint32_t o = 0u; o < a->observationCount; o++) {
        float cc = keepFinite(a->chCorr[o], fmaf(track, x[o] * previous[o] - a->chCorr[o], a->chCorr[o]));
        a->chCorr[o] = cc; maxCorr = fmaxf(maxCorr, cc);
        float *ctrl = a->control + o * OAK_CELL_STRIDE, ch = x[o] - previous[o];
        for (uint32_t d = 0u; d < a->actionCount; d++) ctrl[d] = keepFinite(ctrl[d], fmaf(rate, fmaf(a->actionSignal[d], ch, -ctrl[d]), ctrl[d]));
    }
    a->prevCorr = a->obsCorr; a->obsCorr = maxCorr >= 1.0f ? 1.0f : maxCorr * maxCorr; rescanTraceBound(a);
}
static __attribute__((noinline, cold)) void perceiveTraceSlow(OakAgent *a, float *restrict traces, const float *restrict standardized, const float *restrict obs, uint32_t obsN) {
    static const float TRACE_RATE[OAK_TRACE_LEVELS] = { 0.5f, 0.25f, 0.125f, 0.0625f, 0.03125f, 0.015625f, 0.0078125f };
    for (uint32_t o = 0u; o < obsN; o++) {
        float st = standardized[o]; int fin = isfinite(obs[o]);
        for (uint32_t m = 0u; m < OAK_TRACE_LEVELS; m++) {
            uint32_t idx = m * obsN + o; float lv = a->traceLevel[idx];
            a->traceLevel[idx] = fin ? fmaf(TRACE_RATE[m], st - lv, lv) : lv;
            traces[idx] = finiteOrZero(a->traceLevel[idx]);
        }
    }
}

static __attribute__((noinline)) void encode(OakAgent *a, const float *restrict x, const float *restrict obs, int traceSafe) {
    OakGeometry g = a->g;
    uint32_t obsN = a->observationCount, traceTotal = OAK_TRACE_LEVELS * obsN;
    uint32_t fineCode[(1u + OAK_TRACE_LEVELS) * OAK_OBSERVATION_LIMIT], coarseCode[(1u + OAK_TRACE_LEVELS) * OAK_OBSERVATION_LIMIT], settledCode[OAK_TRACE_LEVELS * OAK_OBSERVATION_LIMIT];
    uint32x4_t vTwo = vdupq_n_u32(2u), vEight = vdupq_n_u32(8u), vNine = vdupq_n_u32(9u), vZeroU = vdupq_n_u32(0u), vStep4 = { 0u, 1u, 2u, 3u };
    uint32_t *restrict cells = a->cellActive + a->slot * OAK_CELLS;
    uint32_t selfBits; __builtin_memcpy(&selfBits, obs + a->self, sizeof selfBits);
    uint32x4_t vSelf = vdupq_n_u32(selfBits);
    uint32_t exact = 0x9E3779B9u, o = 0u;
    int doExact = !a->box;
    for (; o + 4u <= obsN; o += 4u) {
        uint32x4_t fc = (uint32x4_t)cellCode4(oak_load4(x + o), 1u);
        vst1q_u32(fineCode + o, fc); vst1q_u32(coarseCode + o, vminq_u32(vqsubq_u32(fc, vTwo), vTwo));
        uint32x4_t idx1 = vaddq_u32(vdupq_n_u32(OAK_TUPLES + 1u + o), vStep4);
        vst1q_u32(cells + OAK_TUPLES + o, mixHash4(vmulq_n_u32(veorq_u32(vmulq_n_u32(idx1, 0x9E3779B9u), fc), 0x01000193u)));
        uint32x4_t idx2 = vaddq_u32(vdupq_n_u32(OAK_TUPLES + OAK_OBSERVATION_LIMIT + 1u + o), vStep4), wb;
        __builtin_memcpy(&wb, obs + o, sizeof wb);
        uint32x4_t h = vmulq_n_u32(veorq_u32(vmulq_n_u32(idx2, 0x9E3779B9u), vSelf), 0x01000193u);
        vst1q_u32(cells + OAK_TUPLES + obsN + o, mixHash4(vmulq_n_u32(veorq_u32(h, wb), 0x01000193u)));
        if (doExact) {
            uint32x4_t b = veorq_u32(wb, vshrq_n_u32(wb, 16));
            b = vmulq_n_u32(b, 0x85EBCA6Bu); b = veorq_u32(b, vshrq_n_u32(b, 13));
            exact = (exact ^ vgetq_lane_u32(b, 0)) * 0x01000193u; exact = (exact ^ vgetq_lane_u32(b, 1)) * 0x01000193u;
            exact = (exact ^ vgetq_lane_u32(b, 2)) * 0x01000193u; exact = (exact ^ vgetq_lane_u32(b, 3)) * 0x01000193u;
        }
    }
    for (; o < obsN; o++) {
        uint32_t fc = cellCode(x[o], 1u), wb;
        fineCode[o] = fc; coarseCode[o] = fc <= 2u ? 0u : fc >= 4u ? 2u : 1u;
        cells[OAK_TUPLES + o] = tableCell(((OAK_TUPLES + 1u + o) * 0x9E3779B9u ^ fc) * 0x01000193u, OAK_TABLE_BITS);
        __builtin_memcpy(&wb, obs + o, sizeof wb);
        cells[OAK_TUPLES + obsN + o] = tableCell(((((OAK_TUPLES + OAK_OBSERVATION_LIMIT + 1u + o) * 0x9E3779B9u ^ selfBits) * 0x01000193u) ^ wb) * 0x01000193u, OAK_TABLE_BITS);
        if (doExact) { uint32_t b = wb; b ^= b >> 16u; b *= 0x85EBCA6Bu; b ^= b >> 13u; exact = (exact ^ b) * 0x01000193u; }
    }
    float momentRate = trackingRateFor(a->stepCount, OAK_FEATURE_TIMESCALE), settleRate = 1.0f / ((float)a->stepCount + 1.0f);
    float ridge = a->stepCount < 2u * OAK_CAPACITY ? 1.0f / ((float)a->stepCount + 1.0f) : g.ridgeFloor;
    uint32_t history = exact;
    if (__builtin_expect(traceSafe, 1)) {
        float4 vMoment = splat4(momentRate), vSettle = splat4(settleRate), vRidge = splat4(ridge), vZero = splat4(0.0f), vQuarter = splat4(0.25f);
        const float *restrict tLevel = a->traceLevel;
        float *restrict tMean = a->traceMean, *restrict tSec = a->traceSecond, *restrict sMean = a->settledMean, *restrict sSec = a->settledSecond;
        uint32_t c = 0u;
        for (; c + 4u <= traceTotal; c += 4u) {
            float4 val = oak_load4(tLevel + c), sq = val * val;
            float4 mean = oak_load4(tMean + c), second = oak_load4(tSec + c), center = oak_load4(sMean + c), spread = oak_load4(sSec + c);
            mean = __builtin_elementwise_fma(vMoment, val - mean, mean);
            second = __builtin_elementwise_fma(vMoment, sq - second, second);
            center = __builtin_elementwise_fma(vSettle, val - center, center);
            spread = __builtin_elementwise_fma(vSettle, sq - spread, spread);
            oak_store4(tMean + c, mean); oak_store4(tSec + c, second); oak_store4(sMean + c, center); oak_store4(sSec + c, spread);
            float4 norm = (val - mean) * oak_rsqrt4_pos(__builtin_elementwise_max(__builtin_elementwise_fma(-mean, mean, second), vZero) + vRidge);
            uint32x4_t fc = (uint32x4_t)cellCode4(norm, 1u);
            vst1q_u32(fineCode + obsN + c, fc); vst1q_u32(coarseCode + obsN + c, vaddq_u32(vminq_u32(vqsubq_u32(fc, vTwo), vTwo), vEight));
            history = (history ^ vgetq_lane_u32(fc, 0)) * 0x01000193u; history = (history ^ vgetq_lane_u32(fc, 1)) * 0x01000193u;
            history = (history ^ vgetq_lane_u32(fc, 2)) * 0x01000193u; history = (history ^ vgetq_lane_u32(fc, 3)) * 0x01000193u;
            float4 sd = val - center, sv = __builtin_elementwise_max(__builtin_elementwise_fma(-center, center, spread), vZero) + vRidge;
            float4 sd2 = sd * sd, sq4 = vQuarter * sv;
            lane4 slt = (lane4)(sd < vZero) & (lane4)(sd2 > sq4), sge = (lane4)(sd > vZero) & (lane4)(sd2 >= sq4);
            vst1q_u32(settledCode + c, vsubq_u32(vaddq_u32(vNine, (uint32x4_t)slt), (uint32x4_t)sge));
        }
        for (; c < traceTotal; c++) {
            float val = tLevel[c], mean = tMean[c] + momentRate * (val - tMean[c]), second = tSec[c] + momentRate * (val * val - tSec[c]);
            float center = sMean[c] + settleRate * (val - sMean[c]), spread = sSec[c] + settleRate * (val * val - sSec[c]);
            tMean[c] = mean; tSec[c] = second; sMean[c] = center; sSec[c] = spread;
            uint32_t fc = cellCode((val - mean) * oak_rsqrt(fmaxf(second - mean * mean, 0.0f) + ridge), 1u);
            fineCode[obsN + c] = fc; coarseCode[obsN + c] = (fc <= 2u ? 0u : fc >= 4u ? 2u : 1u) + 8u;
            history = (history ^ fc) * 0x01000193u;
            float sd = val - center, sv = fmaxf(spread - center * center, 0.0f) + ridge;
            settledCode[c] = (sd < 0.0f && sd * sd > 0.25f * sv) ? 8u : (sd > 0.0f && sd * sd >= 0.25f * sv) ? 10u : 9u;
        }
    } else encodeTraceSlowLoop(a, obsN, traceTotal, momentRate, settleRate, ridge, fineCode, coarseCode, settledCode, &history);
    if (doExact) {
        uint32_t exactCell = tableCell(exact, OAK_EXACT_BITS), histCell = tableCell(history, OAK_HIST_BITS);
        a->qCell[a->slot * OAK_KEYS] = exactCell; a->qCell[a->slot * OAK_KEYS + 1u] = histCell;
        a->qTag[a->slot * OAK_KEYS] = exact; a->qTag[a->slot * OAK_KEYS + 1u] = history;
        __builtin_prefetch(&a->exactTable[exactCell], 0, 3); __builtin_prefetch(&a->histTable[histCell], 0, 3);
    }
    static const uint32_t TUPLE_INIT[24] = { 1u*0x9E3779B9u, 2u*0x9E3779B9u, 3u*0x9E3779B9u, 4u*0x9E3779B9u, 5u*0x9E3779B9u, 6u*0x9E3779B9u, 7u*0x9E3779B9u, 8u*0x9E3779B9u, 9u*0x9E3779B9u, 10u*0x9E3779B9u, 11u*0x9E3779B9u, 12u*0x9E3779B9u, 13u*0x9E3779B9u, 14u*0x9E3779B9u, 15u*0x9E3779B9u, 16u*0x9E3779B9u, 17u*0x9E3779B9u, 18u*0x9E3779B9u, 19u*0x9E3779B9u, 20u*0x9E3779B9u, 21u*0x9E3779B9u, 22u*0x9E3779B9u, 23u*0x9E3779B9u, 0u };
    uint32x4_t h0 = vld1q_u32(TUPLE_INIT), h1 = vld1q_u32(TUPLE_INIT + 4), h2 = vld1q_u32(TUPLE_INIT + 8), h3 = vld1q_u32(TUPLE_INIT + 12), h4 = vld1q_u32(TUPLE_INIT + 16), h5 = vld1q_u32(TUPLE_INIT + 20);
    for (o = 0u; o + 4u <= obsN; o += 4u) {
        uint32x4_t co = vld1q_u32(coarseCode + o), fi = vld1q_u32(fineCode + o), cf0 = vzip1q_u32(co, fi), cf1 = vzip2q_u32(co, fi);
        uint32x4_t vcf[4] = { vreinterpretq_u32_u64(vdupq_laneq_u64(vreinterpretq_u64_u32(cf0), 0)), vreinterpretq_u32_u64(vdupq_laneq_u64(vreinterpretq_u64_u32(cf0), 1)), vreinterpretq_u32_u64(vdupq_laneq_u64(vreinterpretq_u64_u32(cf1), 0)), vreinterpretq_u32_u64(vdupq_laneq_u64(vreinterpretq_u64_u32(cf1), 1)) };
        uint32x4_t vcc[4] = { vdupq_laneq_u32(co, 0), vdupq_laneq_u32(co, 1), vdupq_laneq_u32(co, 2), vdupq_laneq_u32(co, 3) };
        #pragma clang loop unroll(full)
        for (uint32_t s = 0u; s < 4u; s++) {
            h0 = vmulq_n_u32(veorq_u32(h0, vcf[s]), 0x01000193u); h1 = vmulq_n_u32(veorq_u32(h1, vcf[s]), 0x01000193u); h2 = vmulq_n_u32(veorq_u32(h2, vcf[s]), 0x01000193u);
            h3 = vmulq_n_u32(veorq_u32(h3, vcf[s]), 0x01000193u); h4 = vmulq_n_u32(veorq_u32(h4, vcc[s]), 0x01000193u); h5 = vmulq_n_u32(veorq_u32(h5, vcc[s]), 0x01000193u);
        }
    }
    for (; o < obsN; o++) {
        uint32x4_t vcf = { coarseCode[o], fineCode[o], coarseCode[o], fineCode[o] }, vcc = vdupq_n_u32(coarseCode[o]);
        h0 = vmulq_n_u32(veorq_u32(h0, vcf), 0x01000193u); h1 = vmulq_n_u32(veorq_u32(h1, vcf), 0x01000193u); h2 = vmulq_n_u32(veorq_u32(h2, vcf), 0x01000193u);
        h3 = vmulq_n_u32(veorq_u32(h3, vcf), 0x01000193u); h4 = vmulq_n_u32(veorq_u32(h4, vcc), 0x01000193u); h5 = vmulq_n_u32(veorq_u32(h5, vcc), 0x01000193u);
    }
    uint32_t h00 = vgetq_lane_u32(h0, 0), h01 = vgetq_lane_u32(h0, 1);
    for (o = 0u; o + 4u <= obsN; o += 4u) {
        uint32_t o1 = obsN + o, o2 = 2u * obsN + o, o3 = 3u * obsN + o, o4 = 4u * obsN + o, o5 = 5u * obsN + o, o6 = 6u * obsN + o, o7 = 7u * obsN + o;
        h0 = fnvStep4(h0, vZeroU, vZeroU, vld1q_u32(coarseCode + o1), vaddq_u32(vld1q_u32(fineCode + o1), vEight));
        h1 = fnvStep4(h1, vld1q_u32(coarseCode + o2), vaddq_u32(vld1q_u32(fineCode + o2), vEight), vld1q_u32(coarseCode + o3), vaddq_u32(vld1q_u32(fineCode + o3), vEight));
        h2 = fnvStep4(h2, vld1q_u32(coarseCode + o4), vaddq_u32(vld1q_u32(fineCode + o4), vEight), vld1q_u32(coarseCode + o5), vaddq_u32(vld1q_u32(fineCode + o5), vEight));
        h3 = fnvStep4(h3, vld1q_u32(coarseCode + o6), vaddq_u32(vld1q_u32(fineCode + o6), vEight), vld1q_u32(coarseCode + o7), vaddq_u32(vld1q_u32(fineCode + o7), vEight));
        h4 = fnvStep4(h4, vld1q_u32(settledCode + o), vld1q_u32(settledCode + o1), vld1q_u32(settledCode + o2), vld1q_u32(settledCode + o3));
        h5 = fnvStep4(h5, vld1q_u32(settledCode + o4), vld1q_u32(settledCode + o5), vld1q_u32(settledCode + o6), vZeroU);
    }
    for (; o < obsN; o++) {
        uint32_t o1 = obsN + o, o2 = 2u * obsN + o, o3 = 3u * obsN + o, o4 = 4u * obsN + o, o5 = 5u * obsN + o, o6 = 6u * obsN + o, o7 = 7u * obsN + o;
        uint32x4_t v0 = { 0u, 0u, coarseCode[o1], fineCode[o1] + 8u }, v1 = { coarseCode[o2], fineCode[o2] + 8u, coarseCode[o3], fineCode[o3] + 8u };
        uint32x4_t v2 = { coarseCode[o4], fineCode[o4] + 8u, coarseCode[o5], fineCode[o5] + 8u }, v3 = { coarseCode[o6], fineCode[o6] + 8u, coarseCode[o7], fineCode[o7] + 8u };
        uint32x4_t v4 = { settledCode[o], settledCode[o1], settledCode[o2], settledCode[o3] }, v5 = { settledCode[o4], settledCode[o5], settledCode[o6], 0u };
        h0 = vmulq_n_u32(veorq_u32(h0, v0), 0x01000193u); h1 = vmulq_n_u32(veorq_u32(h1, v1), 0x01000193u);
        h2 = vmulq_n_u32(veorq_u32(h2, v2), 0x01000193u); h3 = vmulq_n_u32(veorq_u32(h3, v3), 0x01000193u);
        h4 = vmulq_n_u32(veorq_u32(h4, v4), 0x01000193u); h5 = vmulq_n_u32(veorq_u32(h5, v5), 0x01000193u);
    }
    h0 = vsetq_lane_u32(h01, vsetq_lane_u32(h00, h0, 0), 1);
    vst1q_u32(cells, mixHash4(h0)); vst1q_u32(cells + 4, mixHash4(h1)); vst1q_u32(cells + 8, mixHash4(h2));
    vst1q_u32(cells + 12, mixHash4(h3)); vst1q_u32(cells + 16, mixHash4(h4));
    uint32x4_t m5 = mixHash4(h5); cells[20] = vgetq_lane_u32(m5, 0); cells[21] = vgetq_lane_u32(m5, 1); cells[22] = vgetq_lane_u32(m5, 2);
    for (uint32_t t = 0u; t < 8u; t++) __builtin_prefetch(a->cell + cells[t], 0, 3);
    if (a->stepCount > 0u) {
        const float *restrict previous = features(a, 1u - a->slot);
        float *restrict chCorr = a->chCorr, *restrict control = a->control;
        float32x4_t vRate = vdupq_n_f32(1.0f / (float)a->stepCount), actLow = (float32x4_t)oak_load4(a->actionSignal);
        float32x4_t actHigh = { a->actionSignal[4], a->actionSignal[5], 0.0f, 0.0f };
        float4 vTrack = splat4(g.trackingRate), vMaxCorr = splat4(0.0f);
        int wideAct = a->actionCount > 4u;
        if (__builtin_expect(traceSafe, 1)) {
            #define OAK_CTRL_STEP(PTR, ACT, DIFF, LANE) do { float32x4_t c_ = (float32x4_t)oak_load4((PTR)); oak_store4((PTR), (float4)vfmaq_f32(c_, vRate, vfmaq_laneq_f32(vnegq_f32(c_), (ACT), (DIFF), (LANE)))); } while (0)
            if (!wideAct) {
                for (o = 0u; o + 4u <= obsN; o += 4u) {
                    float4 x4 = oak_load4(x + o), p4 = oak_load4(previous + o), vc = oak_load4(chCorr + o);
                    vc = __builtin_elementwise_fma(vTrack, x4 * p4 - vc, vc);
                    oak_store4(chCorr + o, vc); vMaxCorr = __builtin_elementwise_max(vMaxCorr, vc);
                    float32x4_t diff = (float32x4_t)(x4 - p4); float *ctrl = control + o * OAK_CELL_STRIDE;
                    OAK_CTRL_STEP(ctrl, actLow, diff, 0); OAK_CTRL_STEP(ctrl + OAK_CELL_STRIDE, actLow, diff, 1);
                    OAK_CTRL_STEP(ctrl + 2u * OAK_CELL_STRIDE, actLow, diff, 2); OAK_CTRL_STEP(ctrl + 3u * OAK_CELL_STRIDE, actLow, diff, 3);
                }
            } else {
                for (o = 0u; o + 4u <= obsN; o += 4u) {
                    float4 x4 = oak_load4(x + o), p4 = oak_load4(previous + o), vc = oak_load4(chCorr + o);
                    vc = __builtin_elementwise_fma(vTrack, x4 * p4 - vc, vc);
                    oak_store4(chCorr + o, vc); vMaxCorr = __builtin_elementwise_max(vMaxCorr, vc);
                    float32x4_t diff = (float32x4_t)(x4 - p4); float *ctrl = control + o * OAK_CELL_STRIDE;
                    OAK_CTRL_STEP(ctrl, actLow, diff, 0); OAK_CTRL_STEP(ctrl + 4u, actHigh, diff, 0);
                    OAK_CTRL_STEP(ctrl + OAK_CELL_STRIDE, actLow, diff, 1); OAK_CTRL_STEP(ctrl + OAK_CELL_STRIDE + 4u, actHigh, diff, 1);
                    OAK_CTRL_STEP(ctrl + 2u * OAK_CELL_STRIDE, actLow, diff, 2); OAK_CTRL_STEP(ctrl + 2u * OAK_CELL_STRIDE + 4u, actHigh, diff, 2);
                    OAK_CTRL_STEP(ctrl + 3u * OAK_CELL_STRIDE, actLow, diff, 3); OAK_CTRL_STEP(ctrl + 3u * OAK_CELL_STRIDE + 4u, actHigh, diff, 3);
                }
            }
            #undef OAK_CTRL_STEP
            float maxCorr = vmaxvq_f32((float32x4_t)vMaxCorr);
            for (; o < obsN; o++) {
                float cc = fmaf(g.trackingRate, x[o] * previous[o] - chCorr[o], chCorr[o]);
                chCorr[o] = cc; maxCorr = fmaxf(maxCorr, cc);
                float *ctrl = control + o * OAK_CELL_STRIDE; float32x4_t c0 = (float32x4_t)oak_load4(ctrl), ch = vdupq_n_f32(x[o] - previous[o]);
                oak_store4(ctrl, (float4)vfmaq_f32(c0, vRate, vfmaq_f32(vnegq_f32(c0), actLow, ch)));
                if (wideAct) { float32x4_t c1 = (float32x4_t)oak_load4(ctrl + 4u); oak_store4(ctrl + 4u, (float4)vfmaq_f32(c1, vRate, vfmaq_f32(vnegq_f32(c1), actHigh, ch))); }
            }
            a->prevCorr = a->obsCorr;
            a->obsCorr = maxCorr >= 1.0f ? 1.0f : maxCorr * maxCorr;
        } else encodeCtrlSlow(a, x, previous, 1.0f / (float)a->stepCount, g.trackingRate);
    }
}

static __attribute__((noinline)) void perceive(OakAgent *a, const float *obs) {
    OakGeometry g = a->g;
    uint32_t obsN = a->observationCount, metaAge = a->stepCount < OAK_FEATURE_TIMESCALE ? a->stepCount : OAK_FEATURE_TIMESCALE;
    float standardized[OAK_OBSERVATION_LIMIT + 1u], inverseAge = 1.0f / ((float)metaAge + 1.0f), floorRate = 1.0f / ((float)a->stepCount + 1.0f);
    float *x = features(a, a->slot);
    float4 vInvAge = splat4(inverseAge), vFloorRate = splat4(floorRate), vMetaRate = splat4(g.metaRate);
    float4 vOneMinusDecay = splat4(1.0f - g.trackingRate), vAlphaFloor = splat4(g.trackingRate * g.trackingRate), vRidgeFloor = splat4(g.ridgeFloor);
    float4 vZero = splat4(0.0f), vOne = splat4(1.0f), vHalf = splat4(0.5f);
    uint32_t o = 0u;
    lane4 obsFinAll = (lane4)vdupq_n_u32(0xffffffffu);
    float4 stAcc4 = vZero, stMax4 = vZero;
    for (; o + 4u <= obsN; o += 4u) {
        float4 vx = oak_load4(obs + o); lane4 fin = oak_finite4(vx);
        obsFinAll &= fin;
        float4 mean = oak_load4(a->chMean + o), second = oak_load4(a->chSecond + o), alpha = oak_load4(a->chAlpha + o);
        float4 trace = oak_load4(a->chTrace + o), energy = oak_load4(a->chEnergy + o), sig = vx - mean, mg = sig * trace;
        float4 ge = oak_keepFinite4(energy, __builtin_elementwise_fma(vInvAge, __builtin_elementwise_fma(mg, mg, -energy), energy));
        float4 db = vMetaRate * mg * oak_rsqrt4_fast(__builtin_elementwise_max(ge, vZero) + vInvAge);
        float4 na = oak_keepFinite4(alpha, __builtin_elementwise_min(__builtin_elementwise_max(alpha * __builtin_elementwise_fma(db, __builtin_elementwise_fma(db, vHalf, vOne), vOne), vAlphaFloor), vOne));
        float4 nh = oak_keepFinite4(trace, __builtin_elementwise_fma(vOneMinusDecay * __builtin_elementwise_max(vZero, vOne - alpha), trace, alpha * sig));
        lane4 updMeta = fin & oak_finite4(sig);
        alpha = select4(updMeta, na, alpha); trace = select4(updMeta, nh, trace); energy = select4(updMeta, ge, energy);
        float4 rate = __builtin_elementwise_max(vFloorRate, alpha);
        mean = select4(fin, oak_keepFinite4(mean, __builtin_elementwise_fma(rate, vx - mean, mean)), mean);
        second = select4(fin, oak_keepFinite4(second, __builtin_elementwise_fma(rate, __builtin_elementwise_fma(vx, vx, -second), second)), second);
        oak_store4(a->chMean + o, mean); oak_store4(a->chSecond + o, second); oak_store4(a->chAlpha + o, alpha);
        oak_store4(a->chTrace + o, trace); oak_store4(a->chEnergy + o, energy);
        float4 stdv = (vx - mean) * oak_rsqrt4(__builtin_elementwise_max(__builtin_elementwise_fma(-mean, mean, second), vZero) + vRidgeFloor);
        stAcc4 += stdv; stMax4 = __builtin_elementwise_max(stMax4, __builtin_elementwise_abs(stdv));
        oak_store4(standardized + o, stdv); oak_store4(x + o, oak_finiteOrZero4(stdv));
    }
    int allObsFin = vminvq_u32((uint32x4_t)obsFinAll) != 0u;
    float stAcc = oak_sum4(stAcc4), stMax = vmaxvq_f32((float32x4_t)stMax4);
    for (; o < obsN; o++) {
        if (!isfinite(obs[o])) allObsFin = 0;
        Channel c = { a->chMean[o], a->chSecond[o], a->chAlpha[o], a->chBeta[o], a->chTrace[o], a->chEnergy[o] };
        standardized[o] = standardizeChannel(&c, obs[o], inverseAge, floorRate, g.metaRate, g.trackingRate, a->betaFloor, g.ridgeFloor);
        a->chMean[o] = c.mean; a->chSecond[o] = c.second; a->chAlpha[o] = c.alpha; a->chBeta[o] = c.beta; a->chTrace[o] = c.trace; a->chEnergy[o] = c.energy;
        stAcc += standardized[o]; stMax = fmaxf(stMax, fabsf(standardized[o]));
        x[o] = finiteOrZero(standardized[o]);
    }
    standardized[obsN] = standardizeChannel(&a->rewardChannel, a->reward, inverseAge, floorRate, g.metaRate, g.trackingRate, a->betaFloor, g.ridgeFloor);
    for (uint32_t d = 0u; d < OAK_ACTION_PARAMS; d++) x[obsN + d] = finiteOrZero(a->actionSignal[d]);
    x[obsN + OAK_ACTION_PARAMS] = finiteOrZero(standardized[obsN]);
    x[obsN + OAK_ACTION_PARAMS + 1u] = 1.0f;
    static const float TRACE_RATE[OAK_TRACE_LEVELS] = { 0.5f, 0.25f, 0.125f, 0.0625f, 0.03125f, 0.015625f, 0.0078125f };
    float *restrict traces = x + g.projected, *restrict tLevel = a->traceLevel;
    int traceSafe = allObsFin && ((stAcc - stAcc) == 0.0f) && stMax <= 1e12f && a->traceBound <= 1e12f;
    if (__builtin_expect(traceSafe, 1)) {
        a->traceBound = fmaxf(a->traceBound, stMax) * (1.0f + 1e-6f);
        for (o = 0u; o + 4u <= obsN; o += 4u) {
            float4 st = oak_load4(standardized + o);
            #pragma clang loop unroll(full)
            for (uint32_t m = 0u; m < OAK_TRACE_LEVELS; m++) {
                uint32_t idx = m * obsN + o; float4 lv = oak_load4(tLevel + idx);
                float4 upd = __builtin_elementwise_fma(splat4(TRACE_RATE[m]), st - lv, lv);
                oak_store4(tLevel + idx, upd); oak_store4(traces + idx, upd);
            }
        }
        for (; o < obsN; o++) {
            float st = standardized[o];
            #pragma clang loop unroll(full)
            for (uint32_t m = 0u; m < OAK_TRACE_LEVELS; m++) {
                uint32_t idx = m * obsN + o; float lv = tLevel[idx];
                float upd = lv + TRACE_RATE[m] * (st - lv);
                tLevel[idx] = upd; traces[idx] = upd;
            }
        }
    } else perceiveTraceSlow(a, traces, standardized, obs, obsN);
    float rwAbs = fabsf(x[obsN + OAK_ACTION_PARAMS]);
    a->featBound[a->slot] = (traceSafe && rwAbs <= 1e12f && a->projBound <= 1e20f) ? fmaxf(fmaxf(a->traceBound, rwAbs), 1.0f) : INFINITY;
    encode(a, x, obs, traceSafe);
    uint32_t inputs = g.inputs, sources = a->sources, permStride = a->permStride;
    float *restrict px = permInputs(a, a->slot);
    const float *restrict xr = x;
    for (uint32_t p = 0u; p < OAK_PERMS; p++) {
        float *restrict dst = px + p * permStride, *restrict d = dst;
        const uint16_t *restrict perm = a->inputPerm + p * sources, *restrict pr = perm, *pe8 = perm + (sources & ~7u), *pe4 = perm + (sources & ~3u);
        for (; pr < pe8; pr += 8u, d += 8u) {
            uint64_t p0, p1; __builtin_memcpy(&p0, pr, 8u); __builtin_memcpy(&p1, pr + 4u, 8u);
            float4 v0 = { xr[p0 & 0xffffu], xr[(p0 >> 16u) & 0xffffu], xr[(p0 >> 32u) & 0xffffu], xr[p0 >> 48u] };
            float4 v1 = { xr[p1 & 0xffffu], xr[(p1 >> 16u) & 0xffffu], xr[(p1 >> 32u) & 0xffffu], xr[p1 >> 48u] };
            oak_store4(d, v0); oak_store4(d + 4u, v1);
        }
        for (; pr < pe4; pr += 4u, d += 4u) {
            uint64_t p4; __builtin_memcpy(&p4, pr, 8u);
            float4 v0 = { xr[p4 & 0xffffu], xr[(p4 >> 16u) & 0xffffu], xr[(p4 >> 32u) & 0xffffu], xr[p4 >> 48u] };
            oak_store4(d, v0);
        }
        for (uint32_t k = sources & ~3u; k < sources; k++) dst[k] = xr[perm[k]];
        oak_store4(dst + sources, oak_load4(dst));
    }
    float *restrict unit = x + inputs, *restrict energies = a->unitEnergy + a->slot * OAK_UNITS;
    const float *restrict proj = a->projection; const uint16_t *restrict uBlk = a->unitBlock;
    for (uint32_t j = 0u; j < OAK_UNITS; j += 4u) {
        float4 ss[4], ee[4];
        for (uint32_t u = 0u; u < 4u; u++) {
            const float *restrict weight = proj + (j + u) * OAK_FANIN;
            const uint16_t *restrict blk = uBlk + (j + u) * OAK_FANIN_BLOCKS;
            uint64_t b0, b1; __builtin_memcpy(&b0, blk, 8u); __builtin_memcpy(&b1, blk + 4u, 8u);
            uint16_t o0 = (uint16_t)b0, o1 = (uint16_t)(b0 >> 16u), o2 = (uint16_t)(b0 >> 32u), o3 = (uint16_t)(b0 >> 48u);
            uint16_t o4 = (uint16_t)b1, o5 = (uint16_t)(b1 >> 16u), o6 = (uint16_t)(b1 >> 32u), o7 = (uint16_t)(b1 >> 48u);
            float4 iv0 = oak_load4_off(px, o0), iv1 = oak_load4_off(px, o1), iv2 = oak_load4_off(px, o2), iv3 = oak_load4_off(px, o3);
            float4 s0 = oak_load4(weight) * iv0, s1 = oak_load4(weight + 4u) * iv1, s2 = oak_load4(weight + 8u) * iv2, s3 = oak_load4(weight + 12u) * iv3;
            float4 e0 = iv0 * iv0, e1 = iv1 * iv1, e2 = iv2 * iv2, e3 = iv3 * iv3;
            iv0 = oak_load4_off(px, o4); iv1 = oak_load4_off(px, o5); iv2 = oak_load4_off(px, o6); iv3 = oak_load4_off(px, o7);
            s0 = __builtin_elementwise_fma(oak_load4(weight + 16u), iv0, s0); e0 = __builtin_elementwise_fma(iv0, iv0, e0);
            s1 = __builtin_elementwise_fma(oak_load4(weight + 20u), iv1, s1); e1 = __builtin_elementwise_fma(iv1, iv1, e1);
            s2 = __builtin_elementwise_fma(oak_load4(weight + 24u), iv2, s2); e2 = __builtin_elementwise_fma(iv2, iv2, e2);
            s3 = __builtin_elementwise_fma(oak_load4(weight + 28u), iv3, s3); e3 = __builtin_elementwise_fma(iv3, iv3, e3);
            ss[u] = (s0 + s1) + (s2 + s3); ee[u] = (e0 + e1) + (e2 + e3);
        }
        oak_store4(unit + j, oak_hsum4(ss[0], ss[1], ss[2], ss[3]));
        oak_store4(energies + j, oak_hsum4(ee[0], ee[1], ee[2], ee[3]));
    }
    uint32_t half = OAK_UNITS / 2u;
    float *gates = a->unitGate + a->slot * half;
    float4 vGate = splat4(a->obsCorr * a->obsCorr);
    for (uint32_t k = 0u; k < half; k += 4u) {
        float4 t0 = oak_tanh4(oak_load4(unit + k)), t1 = oak_tanh4(oak_load4(unit + half + k));
        oak_store4(unit + k, t0); oak_store4(gates + k, t1);
        oak_store4(unit + half + k, t1 * __builtin_elementwise_fma(vGate, t0 - vOne, vOne));
    }
}

static inline __attribute__((always_inline)) void evalCells(const CellRow *restrict tab, const uint32_t *restrict idx, uint32_t n, int wide, float *restrict val, float4 *restrict pL, float4 *restrict pH, float4 *restrict qL, float4 *restrict qH) {
    float v = *val; float4 pl = *pL, ph = *pH, ql = *qL, qh = *qH;
    uint32_t pe = n > 8u ? n - 8u : 0u, t = 0u;
    for (; t < pe; t++) {
        __builtin_prefetch(tab + idx[t + 8u], 0, 3);
        const CellRow *s = tab + idx[t]; const float *p = s->policy;
        v += s->value; pl += oak_load4(p); ql += oak_load4(s->q);
        if (wide) { ph += oak_load4(p + 4u); qh += oak_load4(s->q + 4u); }
    }
    for (; t < n; t++) {
        const CellRow *s = tab + idx[t]; const float *p = s->policy;
        v += s->value; pl += oak_load4(p); ql += oak_load4(s->q);
        if (wide) { ph += oak_load4(p + 4u); qh += oak_load4(s->q + 4u); }
    }
    *val = v; *pL = pl; *pH = ph; *qL = ql; *qH = qh;
}

static __attribute__((noinline)) void evaluate(OakAgent *a) {
    OakGeometry g = a->g;
    uint32_t count = g.features, actions = a->actionCount, head = a->learner;
    const uint32_t *curCells = a->cellActive + a->slot * OAK_CELLS, *prevCells = a->cellActive + (1u - a->slot) * OAK_CELLS;
    uint32_t cellCount = OAK_TUPLES + 2u * a->observationCount, prevTuples = a->stepCount > 0u ? cellCount : 0u;
    const CellRow *cellTable = a->cell;
    if (prevTuples > 0u) {
        uint32_t pre = prevTuples < 8u ? prevTuples : 8u;
        for (uint32_t i = 0u; i < pre; i++) __builtin_prefetch(cellTable + prevCells[i], 1, 3);
    }
    const float *current = features(a, a->slot), *previous = features(a, 1u - a->slot);
    float currentValue, previousValue;
    dotProduct2(a->valueWeight, current, previous, count, &currentValue, &previousValue);
    float4 cellLow = splat4(0.0f), cellHigh = cellLow, qLow = cellLow, qHigh = cellLow;
    if (!a->box) {
        if (actions <= 4u) evalCells(cellTable, curCells, cellCount, 0, &currentValue, &cellLow, &cellHigh, &qLow, &qHigh);
        else evalCells(cellTable, curCells, cellCount, 1, &currentValue, &cellLow, &cellHigh, &qLow, &qHigh);
    } else {
        for (uint32_t t = 0u; t < OAK_TUPLES; t++) {
            if (t + 8u < OAK_TUPLES) __builtin_prefetch(cellTable + curCells[t + 8u], 0, 3);
            const CellRow *s = cellTable + curCells[t]; const float *p = s->policy;
            currentValue += s->value; cellLow += oak_load4(p);
            if (actions > 4u) cellHigh += oak_load4(p + 4u);
        }
        for (uint32_t t = OAK_TUPLES; t < cellCount; t++) {
            if (t + 8u < cellCount) __builtin_prefetch(cellTable + curCells[t + 8u], 0, 3);
            currentValue += cellTable[curCells[t]].value;
        }
    }
    if (prevTuples > 0u) {
        uint32_t pe = prevTuples > 8u ? prevTuples - 8u : 0u, t = 0u;
        for (; t < pe; t++) { __builtin_prefetch(cellTable + prevCells[t + 8u], 1, 3); previousValue += cellTable[prevCells[t]].value; }
        for (; t < prevTuples; t++) previousValue += cellTable[prevCells[t]].value;
    }
    float cellLogit[8] = { cellLow.x, cellLow.y, cellLow.z, cellLow.w, cellHigh.x, cellHigh.y, 0.0f, 0.0f };
    const float *policy = headPolicy(a);
    uint32_t d = 0u;
    if (actions >= 4u) {
        float v0, v1, v2, v3;
        dotRows4(policy, policy + count, policy + 2u * count, policy + 3u * count, current, count, &v0, &v1, &v2, &v3);
        a->logit[0] = v0 + cellLogit[0]; a->logit[1] = v1 + cellLogit[1]; a->logit[2] = v2 + cellLogit[2]; a->logit[3] = v3 + cellLogit[3];
        d = 4u;
    }
    for (; d + 2u <= actions; d += 2u) {
        float v0, v1; dotRows2(policy + d * count, policy + (d + 1u) * count, current, count, &v0, &v1);
        a->logit[d] = v0 + cellLogit[d]; a->logit[d + 1u] = v1 + cellLogit[d + 1u];
    }
    for (; d < actions; d++) a->logit[d] = dotProduct(policy + d * count, current, count) + cellLogit[d];
    float avgPrev = a->averageReward, delta = finiteOrZero(a->reward - avgPrev + currentValue - previousValue);
    a->td = delta;
    a->averageReward = keepFinite(avgPrev, avgPrev + g.trackingRate * delta);
    float squareRate = trackingRateFor(a->stepCount, OAK_FEATURE_TIMESCALE);
    a->tdSquare = keepFinite(a->tdSquare, a->tdSquare + squareRate * (delta * delta - a->tdSquare));
    float tdScale = oak_rsqrt(a->tdSquare);
    a->tdInv = tdScale; a->tdNormalized = finiteOrZero(delta * tdScale);
    float advPrev = a->advantageCenter, advErr = delta - advPrev, rewardErr = a->reward - avgPrev;
    a->rewardSquare = keepFinite(a->rewardSquare, a->rewardSquare + squareRate * (rewardErr * rewardErr - a->rewardSquare));
    float rewardGap = a->reward - a->rewardMean;
    a->rewardMean = keepFinite(a->rewardMean, a->rewardMean + squareRate * rewardGap);
    a->rewardVariance = keepFinite(a->rewardVariance, a->rewardVariance + squareRate * (rewardGap * rewardGap - a->rewardVariance));
    float assured = a->rewardMean - sqrtf(fmaxf(a->rewardVariance, 0.0f) * squareRate);
    float aspPrev = a->stepCount == 0u ? assured : a->aspiration;
    a->aspiration = keepFinite(aspPrev, fmaxf(assured, aspPrev + g.metaRate * (a->rewardMean - aspPrev)));
    float shortfall = finiteOrZero(fmaxf(a->aspiration - a->rewardMean, 0.0f));
    float trust = 1.0f;
    if (HEAD_TRUST[head]) { float inf = a->rewardSquare / a->tdSquare; trust = inf >= 1.0f ? 1.0f : sqrtf(inf); }
    int useTd = HEAD_SCALE[head] == SCALE_TD || (a->box && HEAD_NATURAL[head]);
    float scale = useTd ? tdScale : 1.0f;
    a->advantage = finiteOrZero((advErr - (HEAD_ASPIRE[head] ? shortfall : 0.0f)) * scale * trust);
    a->advantageCenter = keepFinite(advPrev, advPrev + g.trackingRate * advErr);
    uint32_t metaAge = a->stepCount < OAK_FEATURE_TIMESCALE ? a->stepCount : OAK_FEATURE_TIMESCALE;
    Channel *meta = a->actorMeta + head;
    if (!a->box || useTd) metaStep(meta, a->advantage, 1.0f / ((float)metaAge + 1.0f), g.metaRate, g.trackingRate, a->betaFloor);
    a->actorStep = finiteOrZero(meta->alpha * a->advantage / OAK_ACTOR_ALPHA_DENOM);
    oak_store4(a->actorCredit, oak_load4(a->actorCreditStaging));
    a->actorCredit[4] = a->actorCreditStaging[4]; a->actorCredit[5] = a->actorCreditStaging[5];
    a->qCellStep = 0.0f;
    if (a->box) return;
    oak_store4(a->qCellValue, qLow); a->qCellValue[4] = qHigh.x; a->qCellValue[5] = qHigh.y;
    float qBest = a->qCellValue[0];
    for (uint32_t i = 1u; i < actions; i++) qBest = fmaxf(qBest, a->qCellValue[i]);
    if (a->stepCount == 0u) return;
    float qDelta = finiteOrZero(a->reward - a->qCellGain + qBest - a->qCellTaken);
    a->qCellGain = keepFinite(a->qCellGain, a->qCellGain + g.trackingRate * qDelta);
    a->qCellNoise = keepFinite(a->qCellNoise, a->qCellNoise + g.trackingRate * (qDelta * qDelta - a->qCellNoise));
    a->qCellStep = finiteOrZero(g.ridgeFloor * qDelta / (1.0f + g.ridgeFloor * (float)cellCount));
    a->qCellAct = a->action;
}

static inline uint32_t cellQAction(const OakAgent *a, const float *policy, float *behaviour) {
    uint32_t actions = a->actionCount, first = 0u;
    for (uint32_t d = 1u; d < actions; d++) if (a->qCellValue[d] > a->qCellValue[first]) first = d;
    float gap = a->qCellValue[first], spread = 0.0f;
    for (uint32_t d = 0u; d < actions; d++) { float w = (d == first ? 1.0f : 0.0f) - policy[d]; gap -= policy[d] * a->qCellValue[d]; spread += w * w; }
    if (gap * gap <= a->qCellNoise * spread) return OAK_POLICY_MAX;
    *behaviour = 1.0f; return first;
}
static inline uint32_t untriedActions(const TableRowHead *row, uint32_t actions) { uint32_t u = 0u; for (uint32_t d = 0u; d < actions; d++) u |= (row->visits[d] == 0u ? 1u : 0u) << d; return u; }
static inline uint32_t knownUntried(const TableRowHead *row, uint32_t tag, uint32_t actions) { return row->owner == tag ? untriedActions(row, actions) : (1u << actions) - 1u; }
static inline float exactRowValue(const TableRowHead *row, uint8_t t8, uint32_t actions) {
    if (row->owner == 0u || tag8(row->owner) != t8) return 0.0f;
    float best = -INFINITY; for (uint32_t d = 0u; d < actions; d++) if (row->visits[d] != 0u) best = fmaxf(best, row->q[d]);
    return best == -INFINITY ? 0.0f : best;
}
static inline void psPush(OakAgent *a, uint32_t cell, uint32_t tag, float prio) {
    if (!(prio > 0.0f) || !isfinite(prio)) return;
    uint32_t count = a->psCount, minIdx = 0u; float minVal = INFINITY;
    for (uint32_t i = 0u; i < count; i++) {
        if (a->psCell[i] == cell && a->psTag[i] == tag) { if (prio > a->psPrio[i]) a->psPrio[i] = prio; return; }
        if (a->psPrio[i] < minVal) { minVal = a->psPrio[i]; minIdx = i; }
    }
    uint32_t slot = count < OAK_DYNA_QUEUE ? a->psCount++ : prio > minVal ? minIdx : OAK_DYNA_QUEUE;
    if (slot < OAK_DYNA_QUEUE) { a->psCell[slot] = (uint16_t)cell; a->psTag[slot] = tag; a->psPrio[slot] = prio; }
}
static inline void recordPredecessor(ExactRow *dst, uint32_t prevCell, uint32_t act) {
    uint32_t meta = dst->predMeta, count = (meta >> 12u) & 7u;
    for (uint32_t i = 0u; i < count; i++) if (dst->predCell[i] == prevCell && ((meta >> (3u * i)) & 7u) == act) return;
    if (count < OAK_DYNA_PREDS) { dst->predCell[count] = (uint16_t)prevCell; dst->predMeta = (uint16_t)(((count + 1u) << 12u) | (meta & 0x0fffu) | (act << (3u * count))); }
    else { for (uint32_t i = 0u; i < 3u; i++) dst->predCell[i] = dst->predCell[i + 1u]; dst->predCell[3] = (uint16_t)prevCell; dst->predMeta = (uint16_t)((OAK_DYNA_PREDS << 12u) | ((meta & 0x0fffu) >> 3u) | (act << 9u)); }
}
static void claimRow(OakAgent *a, uint32_t key, uint32_t cell, TableRowHead *row, uint32_t tag) {
    uint32_t actions = a->actionCount, untried = untriedActions(row, actions), hadOwner = row->owner != 0u;
    if (untried != (1u << actions) - 1u) a->qOpen[key] -= (uint32_t)__builtin_popcount(untried);
    for (uint32_t d = 0u; d < actions; d++) {
        if (((untried >> d) & 1u) == 0u) { a->qTried[key] -= 1u; a->qSum[key] -= row->q[d]; }
        row->q[d] = 0.0f; row->visits[d] = 0u;
        if (key == 0u) { ExactRow *ex = &a->exactTable[cell]; ex->modelProb[d][0] = ex->modelProb[d][1] = ex->modelProb[d][2] = 0u; }
    }
    if (key == 0u && hadOwner) a->exactTable[cell].predMeta = 0u;
    row->owner = tag;
}

static __attribute__((noinline)) void dynaSweep(OakAgent *a) {
    uint32_t actions = a->actionCount, minCov = 255u - (255u / OAK_DYNA_QUEUE);
    float threshSq = a->qNoise[0] * a->g.ridgeFloor;
    for (uint32_t iter = 0u; iter < OAK_DYNA_STEPS && a->psCount > 0u; iter++) {
        uint32_t count = a->psCount, bestIdx = 0u; float bestPrio = a->psPrio[0];
        for (uint32_t i = 1u; i < count; i++) if (a->psPrio[i] > bestPrio) { bestPrio = a->psPrio[i]; bestIdx = i; }
        uint32_t cell = a->psCell[bestIdx], tag = a->psTag[bestIdx];
        a->psCount = --count;
        if (bestIdx < count) { a->psCell[bestIdx] = a->psCell[count]; a->psTag[bestIdx] = a->psTag[count]; a->psPrio[bestIdx] = a->psPrio[count]; }
        const ExactRow *target = &a->exactTable[cell];
        if (target->h.owner != tag) continue;
        uint8_t targetT8 = tag8(tag); uint32_t meta = target->predMeta, pCount = (meta >> 12u) & 7u;
        for (uint32_t i = 0u; i < pCount; i++) {
            uint32_t pCell = target->predCell[i], pAct = (meta >> (3u * i)) & 7u;
            ExactRow *pex = &a->exactTable[pCell]; uint32_t pTag = pex->h.owner;
            if (pTag == 0u || pex->h.visits[pAct] == 0u) continue;
            uint32_t pSelf = pex->modelProb[pAct][2], sumP = pex->modelProb[pAct][0] + pex->modelProb[pAct][1] + pSelf;
            if (sumP < minCov) continue;
            uint32_t matched = 0u, valid = 1u; float boot = 0.0f;
            for (uint32_t b = 0u; b < 2u; b++) {
                uint32_t pb = pex->modelProb[pAct][b]; if (pb == 0u) continue;
                uint32_t nCell = pex->modelNext[pAct][b]; uint8_t nTag = pex->modelTag[pAct][b];
                if (nCell == cell && nTag == targetT8) matched = 1u;
                const TableRowHead *nrow = &a->exactTable[nCell].h;
                if (nrow->owner != 0u && tag8(nrow->owner) != nTag) { valid = 0u; break; }
                boot += (float)pb * exactRowValue(nrow, nTag, actions);
            }
            if (!matched || !valid) continue;
            uint8_t pT8 = tag8(pTag); float prevMax = exactRowValue(&pex->h, pT8, actions);
            if (pSelf > 0u) boot += (float)pSelf * prevMax;
            boot /= (float)sumP;
            float gain = a->qTried[0] ? a->qSum[0] / (float)a->qTried[0] : 0.0f;
            float val = pex->h.q[pAct] + 0.5f * finiteOrZero(pex->modelReward[pAct] - gain + boot - pex->h.q[pAct]);
            if (isfinite(val)) { a->qSum[0] += val - pex->h.q[pAct]; pex->h.q[pAct] = val; }
            float diff = fabsf(exactRowValue(&pex->h, pT8, actions) - prevMax);
            if ((pex->predMeta >> 12u) != 0u && diff * diff > threshSq) psPush(a, pCell, pTag, diff);
        }
    }
}

static __attribute__((noinline)) void learnTable(OakAgent *a, uint32_t key) {
    uint32_t actions = a->actionCount, act = a->action;
    uint32_t next = a->qCell[a->slot * OAK_KEYS + key], previous = a->qCell[(1u - a->slot) * OAK_KEYS + key];
    uint32_t nextTag = a->qTag[a->slot * OAK_KEYS + key], previousTag = a->qTag[(1u - a->slot) * OAK_KEYS + key];
    TableRowHead *row = tableRow(a, key, previous);
    if (row->owner != previousTag) claimRow(a, key, previous, row, previousTag);
    uint32_t firstTry = row->visits[act] == 0u;
    if (firstTry) {
        a->qTried[key] += 1u;
        if (untriedActions(row, actions) == (1u << actions) - 1u) a->qOpen[key] += actions - 1u;
        else a->qOpen[key] -= 1u;
    }
    const TableRowHead *successor = tableRowConst(a, key, next);
    uint32_t successorUntried = knownUntried(successor, nextTag, actions);
    float bootstrap = -INFINITY;
    for (uint32_t d = 0u; d < actions; d++) if (((successorUntried >> d) & 1u) == 0u) bootstrap = fmaxf(bootstrap, successor->q[d]);
    bootstrap = bootstrap == -INFINITY ? 0.0f : bootstrap;
    float delta = finiteOrZero(a->reward - a->qSum[key] / (float)a->qTried[key] + bootstrap - row->q[act]);
    a->qNoise[key] = keepFinite(a->qNoise[key], a->qNoise[key] + a->g.trackingRate * (delta * delta - a->qNoise[key]));
    uint32_t trackModel = key == 0u && a->qTried[0] < OAK_EXACT;
    uint32_t visitsInt = row->visits[act] < 65535u ? (uint32_t)row->visits[act] + 1u : 65535u;
    row->visits[act] = (uint16_t)visitsInt;
    float step = visitsInt == 1u ? 1.0f : visitsInt >= OAK_FEATURE_TIMESCALE ? a->g.trackingRate : 1.0f / (float)visitsInt;
    float value = row->q[act] + step * delta;
    if (isfinite(value)) { a->qSum[key] += value - row->q[act]; row->q[act] = value; }
    if (key == 1u) {
        uint32_t hop = successorUntried ? 1u : 255u; const HistRow *sh = &a->histTable[next];
        for (uint32_t d = 0u; d < actions && hop > 1u && a->qOpen[1]; d++) if (sh->distance[d] + 1u < hop) hop = sh->distance[d] + 1u;
        a->histTable[previous].distance[act] = (uint8_t)hop; return;
    }
    ExactRow *ex = &a->exactTable[previous]; float reward = finiteOrZero(a->reward);
    ex->modelReward[act] = firstTry ? reward : ex->modelReward[act] + step * (reward - ex->modelReward[act]);
    uint32_t newBranch = 0u;
    if (firstTry) {
        if (next == previous) { ex->modelProb[act][0] = ex->modelProb[act][1] = 0u; ex->modelProb[act][2] = 255u; }
        else { ex->modelNext[act][0] = (uint16_t)next; ex->modelTag[act][0] = tag8(nextTag); ex->modelProb[act][0] = 255u; ex->modelProb[act][1] = ex->modelProb[act][2] = 0u; newBranch = 1u; }
    } else {
        float mr = visitsInt >= OAK_DYNA_QUEUE ? (1.0f / (float)OAK_DYNA_QUEUE) : 1.0f / (float)visitsInt, dec = 1.0f - mr;
        float p0 = (float)ex->modelProb[act][0] * (1.0f / 255.0f) * dec, p1 = (float)ex->modelProb[act][1] * (1.0f / 255.0f) * dec, pSelf = (float)ex->modelProb[act][2] * (1.0f / 255.0f) * dec;
        if (next == previous) { if (ex->modelProb[act][2] == 0u) newBranch = 1u; pSelf += mr; }
        else {
            uint8_t nt8 = tag8(nextTag);
            if (ex->modelProb[act][0] > 0u && ex->modelNext[act][0] == next && ex->modelTag[act][0] == nt8) p0 += mr;
            else if (ex->modelProb[act][1] > 0u && ex->modelNext[act][1] == next && ex->modelTag[act][1] == nt8) p1 += mr;
            else if (p0 <= p1) { ex->modelNext[act][0] = (uint16_t)next; ex->modelTag[act][0] = nt8; p0 = mr; newBranch = 1u; }
            else { ex->modelNext[act][1] = (uint16_t)next; ex->modelTag[act][1] = nt8; p1 = mr; newBranch = 1u; }
            if (p1 > p0) {
                uint16_t tn = ex->modelNext[act][0]; uint8_t tt = ex->modelTag[act][0]; float tp = p0;
                ex->modelNext[act][0] = ex->modelNext[act][1]; ex->modelTag[act][0] = ex->modelTag[act][1]; p0 = p1;
                ex->modelNext[act][1] = tn; ex->modelTag[act][1] = tt; p1 = tp;
            }
        }
        ex->modelProb[act][0] = (uint8_t)(p0 * 255.0f + 0.5f); ex->modelProb[act][1] = (uint8_t)(p1 * 255.0f + 0.5f); ex->modelProb[act][2] = (uint8_t)(pSelf * 255.0f + 0.5f);
    }
    if (trackModel) {
        if (previous != next && (newBranch || (a->exactTable[next].predMeta >> 12u) == 0u) && (successor->owner == 0u || successor->owner == nextTag))
            recordPredecessor(&a->exactTable[next], previous, act);
        uint32_t shift = newBranch && !firstTry; float move = fabsf(step * delta);
        if (move * move > a->qNoise[0] * a->g.ridgeFloor || shift) {
            float prio = move + (shift ? sqrtf(fmaxf(a->qNoise[0], 0.0f)) : 0.0f);
            if ((ex->predMeta >> 12u) != 0u) psPush(a, previous, previousTag, prio);
            if (shift && previous != next && (a->exactTable[next].predMeta >> 12u) != 0u) psPush(a, next, nextTag, prio);
        }
        if (a->psCount > 0u) dynaSweep(a);
    }
}

static inline uint32_t greedyTablePick(const TableRowHead *row, uint32_t actions, const float *policy, float noise) {
    uint32_t first = 0u;
    for (uint32_t d = 1u; d < actions; d++) if (row->q[d] > row->q[first]) first = d;
    float horizon = (float)OAK_FEATURE_TIMESCALE, gap = row->q[first], spread = 0.0f;
    for (uint32_t d = 0u; d < actions; d++) {
        float w = (d == first ? 1.0f : 0.0f) - policy[d];
        gap -= policy[d] * row->q[d]; spread += w * w / fminf((float)row->visits[d], horizon);
    }
    return gap * gap > noise * spread ? first : OAK_POLICY_MAX;
}
static inline uint32_t tableAction(const OakAgent *a, uint32_t arm, const uint32_t *words, const float *policy, float *behaviour) {
    uint32_t actions = a->actionCount, key = TABLE_KEY[arm], cell = a->qCell[a->slot * OAK_KEYS + key];
    const TableRowHead *row = tableRowConst(a, key, cell);
    uint32_t unknown = knownUntried(row, a->qTag[a->slot * OAK_KEYS + key], actions);
    if (TABLE_MODE[arm] == TABLE_GREEDY) {
        if (unknown) return OAK_POLICY_MAX;
        uint32_t pick = greedyTablePick(row, actions, policy, a->qNoise[key]);
        if (pick < actions) *behaviour = 1.0f;
        return pick;
    }
    if (unknown == (1u << actions) - 1u) return OAK_POLICY_MAX;
    uint32_t nearest = 255u, frontier = 0u; const HistRow *hrow = &a->histTable[cell];
    for (uint32_t d = 0u; d < actions && a->qOpen[key]; d++) if (hrow->distance[d] < nearest) nearest = hrow->distance[d];
    for (uint32_t d = 0u; d < actions && TABLE_MODE[arm] == TABLE_FRONTIER; d++) frontier |= (nearest < 255u && hrow->distance[d] == nearest ? 1u : 0u) << d;
    uint32_t pool = unknown ? unknown : frontier ? frontier : (1u << actions) - 1u;
    *behaviour = unknown ? 1.0f / (float)__builtin_popcount(unknown) : 1.0f;
    uint32_t chosen = 0u, chosenWord = 0u; float best = -INFINITY;
    for (uint32_t d = 0u; d < actions; d++) {
        if (((pool >> d) & 1u) == 0u) continue;
        float val = unknown ? 0.0f : row->q[d];
        if (val > best || (val == best && words[d] > chosenWord)) { best = val; chosenWord = words[d]; chosen = d; }
    }
    return chosen;
}
static __attribute__((noinline)) uint32_t planAction(OakAgent *a, const uint32_t *words, const float *policy) {
    uint32_t actions = a->actionCount, start = a->qCell[a->slot * OAK_KEYS], startTag = a->qTag[a->slot * OAK_KEYS];
    uint32_t startUntried = knownUntried(&a->exactTable[start].h, startTag, actions);
    if (a->planIndex < a->planLength && a->planCells[a->planIndex] == start && startUntried != (1u << actions) - 1u) return a->planActions[a->planIndex++];
    a->planLength = a->planIndex = 0u;
    if (startUntried == (1u << actions) - 1u) return OAK_POLICY_MAX;
    uint32_t target = OAK_EXACT, targetAction = 0u;
    if (a->qOpen[0] > 0u) {
        uint32_t gen = ++a->planGeneration, head = 0u, tail = 1u, minProb = 255u / (1u + OAK_TRACE_LEVELS);
        a->planQueue[0] = (uint16_t)start; a->planStamp[start] = gen;
        while (head < tail && target == OAK_EXACT) {
            uint32_t cell = a->planQueue[head++]; const ExactRow *ex = &a->exactTable[cell];
            uint32_t untried = untriedActions(&ex->h, actions), pick = 0u;
            for (uint32_t d = 0u; d < actions; d++)
                if (((untried >> d) & 1u) && (target == OAK_EXACT || words[d] > pick)) { pick = words[d]; target = cell; targetAction = d; }
            for (uint32_t d = 0u; d < actions && target == OAK_EXACT; d++) {
                if (ex->h.visits[d] == 0u) continue;
                for (uint32_t b = 0u; b < 2u && tail < OAK_PLAN_BUDGET; b++) {
                    if (ex->modelProb[d][b] < minProb) continue;
                    uint32_t next = ex->modelNext[d][b], nextOwner = a->exactTable[next].h.owner;
                    if ((nextOwner != 0u && tag8(nextOwner) != ex->modelTag[d][b]) || a->planStamp[next] == gen) continue;
                    a->planStamp[next] = gen; a->planParent[next] = (uint16_t)cell; a->planParentAction[next] = (uint8_t)d; a->planQueue[tail++] = (uint16_t)next;
                }
            }
        }
    }
    if (target == OAK_EXACT) return greedyTablePick(&a->exactTable[start].h, actions, policy, a->qNoise[0]);
    uint32_t len = 1u; for (uint32_t cell = target; cell != start; cell = a->planParent[cell]) len++;
    a->planLength = len; a->planCells[len - 1u] = (uint16_t)target; a->planActions[len - 1u] = (uint8_t)targetAction;
    for (uint32_t cell = target, i = len - 1u; cell != start; cell = a->planParent[cell]) { i--; a->planCells[i] = a->planParent[cell]; a->planActions[i] = a->planParentAction[cell]; }
    return a->planActions[a->planIndex++];
}

static __attribute__((noinline)) void sampleAction(OakAgent *a, uint32_t seed, uint32_t step, float *ctrl) {
    uint32_t actions = a->actionCount;
    if (!a->box) {
        u4 rw = philox4x32(seed, RNG_STREAM_ACT, step, 0u, 0u, 0u);
        u4 rw1 = (actions > 4u && a->activeHead >= OAK_HEADS) ? philox4x32(seed, RNG_STREAM_ACT, step, 1u, 0u, 0u) : rw;
        uint32_t words[8] = { rw.x, rw.y, rw.z, rw.w, rw1.x, rw1.y, rw1.z, rw1.w };
        float logit[OAK_POLICY_MAX], weight[OAK_POLICY_MAX], logitMax = -FLT_MAX, total = 0.0f;
        for (uint32_t d = 0u; d < actions; d++) { logit[d] = finiteOrZero(a->logit[d]); logitMax = fmaxf(logitMax, logit[d]); }
        for (uint32_t d = 0u; d < actions; d++) { weight[d] = oak_exp(logit[d] - logitMax); total += weight[d]; }
        float invTotal = 1.0f / total, draw = u01(rw.x) * total, acc = 0.0f;
        uint32_t sampled = 0u;
        for (uint32_t d = 0u; d + 1u < actions; d++) { acc += weight[d]; sampled += (uint32_t)(!(draw < acc)); }
        float behaviour = weight[sampled] * invTotal;
        if (a->activeHead >= OAK_HEADS) {
            float policy[OAK_POLICY_MAX]; for (uint32_t d = 0u; d < actions; d++) policy[d] = weight[d] * invTotal;
            uint32_t arm = a->activeHead - OAK_HEADS, chosen;
            if (TABLE_MODE[arm] == TABLE_PLAN) { chosen = planAction(a, words, policy); behaviour = chosen < actions ? 1.0f : behaviour; }
            else if (TABLE_MODE[arm] == TABLE_CELLQ) chosen = cellQAction(a, policy, &behaviour);
            else chosen = tableAction(a, arm, words, policy, &behaviour);
            sampled = chosen < actions ? chosen : sampled;
        }
        a->qCellTaken = a->qCellValue[sampled];
        for (uint32_t d = 0u; d < actions; d++) ctrl[d] = d == 0u ? (float)sampled : 0.0f;
        a->action = sampled;
        float chosen = weight[sampled] * invTotal, ratio = a->activeHead >= OAK_HEADS ? chosen / behaviour : 1.0f;
        for (uint32_t d = 0u; d < actions; d++) {
            float credit = (d == sampled ? 1.0f : 0.0f) - weight[d] * invTotal;
            a->actionSignal[d] = credit;
            a->actorCreditStaging[d] = ratio * (HEAD_NATURAL[a->learner] ? finiteOrZero(((d == sampled ? 1.0f / chosen : 0.0f) - 1.0f) / (float)actions) : credit);
        }
        return;
    }
    float sigma = HEAD_NATURAL[a->learner] ? oak_rsqrt((float)(actions * OAK_ACTION_PARAMS)) : 1.0f / (float)OAK_ACTION_PARAMS;
    for (uint32_t pair = 0u; 2u * pair < actions; pair++) {
        u4 r = philox4x32(seed, RNG_STREAM_ACT, step, pair >> 1u, 0u, 0u);
        uint32_t w0 = (pair & 1u) ? r.z : r.x, w1 = (pair & 1u) ? r.w : r.y;
        float radius = sqrtf(-2.0f * oak_log(u01open(w0))), sine, cosine;
        oak_sincos(2.0f * OAK_PI * u01(w1), &sine, &cosine);
        float normals[2] = { radius * cosine, radius * sine };
        for (uint32_t k = 0u; k < 2u && 2u * pair + k < actions; k++) {
            uint32_t d = 2u * pair + k;
            float mu = oak_tanh(finiteOrZero(a->logit[d])), sample = finiteOrZero(mu + normals[k] * sigma);
            float clipped = fmaxf(-1.0f, fminf(1.0f, sample));
            ctrl[d] = a->actionSignal[d] = clipped; a->actionSignal[actions + d] = 0.0f;
            a->actorCreditStaging[d] = (sample - mu) * (1.0f - mu * mu);
        }
    }
}

static inline __attribute__((always_inline)) void updateCellRowsFast(CellRow *restrict tab, const uint32_t *restrict idx, uint32_t n, uint32_t qAct, float vD, float qS, float4 sL, float4 sH, int wide, int hasQ) {
    uint32_t pe = n > 8u ? n - 8u : 0u, t = 0u;
    for (; t < pe; t++) {
        __builtin_prefetch(tab + idx[t + 8u], 1, 3);
        CellRow *s = tab + idx[t]; float *c = s->policy;
        s->value += vD;
        if (hasQ) s->q[qAct] += qS;
        oak_store4(c, oak_load4(c) + sL);
        if (wide) oak_store4(c + 4u, oak_load4(c + 4u) + sH);
    }
    for (; t < n; t++) {
        CellRow *s = tab + idx[t]; float *c = s->policy;
        s->value += vD;
        if (hasQ) s->q[qAct] += qS;
        oak_store4(c, oak_load4(c) + sL);
        if (wide) oak_store4(c + 4u, oak_load4(c + 4u) + sH);
    }
}
static __attribute__((noinline, cold)) void addRowSlowAll(float *restrict row, const float *restrict z, float scale, uint32_t count, float *restrict bound) {
    float mx = *bound;
    for (uint32_t k = 0u; k < count; k++) { row[k] = keepFinite(row[k], fmaf(z[k], scale, row[k])); mx = fmaxf(mx, fabsf(row[k])); }
    *bound = mx;
}
static __attribute__((noinline, cold)) void addPolicySlowAll(OakAgent *a, float *restrict policy, const float *restrict pTrace, float scale, uint32_t actions, uint32_t count) {
    float pb = 0.0f, zb = 0.0f;
    for (uint32_t d = 0u; d < actions; d++) { addRowSlowAll(policy + d * count, pTrace + d * count, scale, count, &pb); for (uint32_t k = 0u; k < count; k++) zb = fmaxf(zb, fabsf(pTrace[d * count + k])); }
    a->polBound[a->learner] = pb; a->pTraceBound = zb;
}
static inline __attribute__((always_inline)) void addPolicyRow(float *restrict row, const float *restrict z, float4 scale, uint32_t count) {
    uint32_t k = 0u;
    for (; k + 16u <= count; k += 16u) {
        float4 n0 = __builtin_elementwise_fma(oak_load4(z + k), scale, oak_load4(row + k)), n1 = __builtin_elementwise_fma(oak_load4(z + k + 4u), scale, oak_load4(row + k + 4u));
        float4 n2 = __builtin_elementwise_fma(oak_load4(z + k + 8u), scale, oak_load4(row + k + 8u)), n3 = __builtin_elementwise_fma(oak_load4(z + k + 12u), scale, oak_load4(row + k + 12u));
        oak_store4(row + k, n0); oak_store4(row + k + 4u, n1); oak_store4(row + k + 8u, n2); oak_store4(row + k + 12u, n3);
    }
    for (; k < count; k += 8u) {
        float4 n0 = __builtin_elementwise_fma(oak_load4(z + k), scale, oak_load4(row + k)), n1 = __builtin_elementwise_fma(oak_load4(z + k + 4u), scale, oak_load4(row + k + 4u));
        oak_store4(row + k, n0); oak_store4(row + k + 4u, n1);
    }
}
static __attribute__((noinline, cold)) void updateProjAllSlow(OakAgent *a, const float *restrict px, const float *restrict signals) {
    float mx = 0.0f;
    for (uint32_t j = 0u; j < OAK_UNITS; j++) for (uint32_t b = 0u; b < OAK_FANIN_BLOCKS; b++) {
        float *w = a->projection + j * OAK_FANIN + 4u * b; float4 o = oak_load4(w);
        float4 nw = oak_keepFinite4(o, __builtin_elementwise_fma(splat4(signals[j]), oak_load4_off(px, a->unitBlock[j * OAK_FANIN_BLOCKS + b]), o));
        oak_store4(w, nw); mx = fmaxf(mx, vmaxvq_f32((float32x4_t)__builtin_elementwise_abs(nw)));
    }
    a->projBound = mx;
}
#define OAK_UPD_PROJ_ROW(W, PX, BLK, SV, LANE) do { \
    uint64_t b0_, b1_; __builtin_memcpy(&b0_, (BLK), 8u); __builtin_memcpy(&b1_, (BLK) + 4u, 8u); \
    float32x4_t p0_ = (float32x4_t)oak_load4_off((PX), (uint16_t)b0_), p1_ = (float32x4_t)oak_load4_off((PX), (uint16_t)(b0_ >> 16u)), p2_ = (float32x4_t)oak_load4_off((PX), (uint16_t)(b0_ >> 32u)), p3_ = (float32x4_t)oak_load4_off((PX), (uint16_t)(b0_ >> 48u)); \
    oak_store4((W), (float4)vfmaq_laneq_f32((float32x4_t)oak_load4((W)), p0_, (SV), (LANE))); oak_store4((W) + 4u, (float4)vfmaq_laneq_f32((float32x4_t)oak_load4((W) + 4u), p1_, (SV), (LANE))); \
    oak_store4((W) + 8u, (float4)vfmaq_laneq_f32((float32x4_t)oak_load4((W) + 8u), p2_, (SV), (LANE))); oak_store4((W) + 12u, (float4)vfmaq_laneq_f32((float32x4_t)oak_load4((W) + 12u), p3_, (SV), (LANE))); \
    float32x4_t p4_ = (float32x4_t)oak_load4_off((PX), (uint16_t)b1_), p5_ = (float32x4_t)oak_load4_off((PX), (uint16_t)(b1_ >> 16u)), p6_ = (float32x4_t)oak_load4_off((PX), (uint16_t)(b1_ >> 32u)), p7_ = (float32x4_t)oak_load4_off((PX), (uint16_t)(b1_ >> 48u)); \
    oak_store4((W) + 16u, (float4)vfmaq_laneq_f32((float32x4_t)oak_load4((W) + 16u), p4_, (SV), (LANE))); oak_store4((W) + 20u, (float4)vfmaq_laneq_f32((float32x4_t)oak_load4((W) + 20u), p5_, (SV), (LANE))); \
    oak_store4((W) + 24u, (float4)vfmaq_laneq_f32((float32x4_t)oak_load4((W) + 24u), p6_, (SV), (LANE))); oak_store4((W) + 28u, (float4)vfmaq_laneq_f32((float32x4_t)oak_load4((W) + 28u), p7_, (SV), (LANE))); \
} while (0)
static __attribute__((noinline, cold)) void rescanCellBound(OakAgent *a) {
    float mx = 0.0f;
    for (uint32_t i = 0u; i < OAK_TABLE; i++) {
        mx = fmaxf(mx, fabsf(a->cell[i].value));
        for (uint32_t d = 0u; d < OAK_POLICY_MAX; d++) {
            mx = fmaxf(mx, fmaxf(fabsf(a->cell[i].q[d]), fabsf(a->cell[i].policy[d])));
            for (uint32_t h = 0u; h < OAK_HEADS; h++) if (h != a->learner) mx = fmaxf(mx, fabsf(a->cellPolicyBank[h][i][d]));
        }
    }
    a->cellBound = mx;
}

static __attribute__((noinline)) void learn(OakAgent *a) {
    OakGeometry g = a->g;
    uint32_t count = g.features, inputs = g.inputs, actions = a->actionCount;
    const float *restrict x = features(a, 1u - a->slot);
    float *restrict vTrace = a->valueTrace;
    float4 decay = splat4(OAK_ELIGIBILITY_DECAY), e0 = splat4(0.0f), e1 = e0;
    for (uint32_t k = 0u; k < count; k += 8u) {
        float4 z0 = __builtin_elementwise_fma(oak_load4(vTrace + k), decay, oak_load4(x + k));
        float4 z1 = __builtin_elementwise_fma(oak_load4(vTrace + k + 4u), decay, oak_load4(x + k + 4u));
        oak_store4(vTrace + k, z0); oak_store4(vTrace + k + 4u, z1);
        e0 = __builtin_elementwise_fma(z0, z0, e0); e1 = __builtin_elementwise_fma(z1, z1, e1);
    }
    const uint32_t *restrict prevCells = a->cellActive + (1u - a->slot) * OAK_CELLS;
    uint32_t prevTuples = a->stepCount > 0u ? OAK_TUPLES + 2u * a->observationCount : 0u;
    uint32_t policyTuples = a->stepCount == 0u ? 0u : a->box ? OAK_TUPLES : prevTuples;
    float eSum = oak_sum4(e0 + e1);
    float alpha = g.ridgeFloor, valueDelta = finiteOrZero(alpha * a->td / (1.0f + alpha * (eSum + (float)prevTuples)));
    if (valueDelta != 0.0f) {
        float dVal = fabsf(valueDelta) * sqrtf(fmaxf(eSum, 0.0f)) * 1.001f;
        if (__builtin_expect((eSum - eSum) == 0.0f && eSum <= 1e30f && dVal <= 1e30f && a->valBound <= 1e30f, 1)) {
            a->valBound += dVal; addPolicyRow(a->valueWeight, vTrace, splat4(valueDelta), count);
        } else { a->valBound = 0.0f; addRowSlowAll(a->valueWeight, vTrace, valueDelta, count, &a->valBound); }
    }
    float *restrict policy = headPolicy(a), *restrict pTrace = a->policyTrace;
    float pDecS = fmaf(a->obsCorr, HEAD_DECAY[a->learner] - OAK_ELIGIBILITY_DECAY, OAK_ELIGIBILITY_DECAY);
    float4 pDec = splat4(pDecS);
    float change[OAK_POLICY_MAX], changeSquare = 0.0f, cMax = 0.0f;
    for (uint32_t i = 0u; i < actions; i++) cMax = fmaxf(cMax, fabsf(a->actorCredit[i]));
    float xMax = a->featBound[1u - a->slot];
    a->pTraceBound = (xMax <= 1e30f && cMax <= 1e30f) ? fmaf(pDecS, a->pTraceBound, cMax * xMax) * 1.0001f : INFINITY;
    uint32_t d = 0u;
    if (actions >= 4u) {
        float *restrict z0p = pTrace, *restrict z1p = pTrace + count, *restrict z2p = pTrace + 2u * count, *restrict z3p = pTrace + 3u * count;
        float4 c0 = splat4(a->actorCredit[0]), c1 = splat4(a->actorCredit[1]), c2 = splat4(a->actorCredit[2]), c3 = splat4(a->actorCredit[3]);
        float4 a0 = splat4(0.0f), a1 = a0, b0 = a0, b1 = a0, g0 = a0, g1 = a0, h0 = a0, h1 = a0;
        for (uint32_t k = 0u; k < count; k += 8u) {
            float4 x0 = oak_load4(x + k), x1 = oak_load4(x + k + 4u);
            float4 za0 = __builtin_elementwise_fma(oak_load4(z0p + k), pDec, c0 * x0), za1 = __builtin_elementwise_fma(oak_load4(z0p + k + 4u), pDec, c0 * x1);
            float4 zb0 = __builtin_elementwise_fma(oak_load4(z1p + k), pDec, c1 * x0), zb1 = __builtin_elementwise_fma(oak_load4(z1p + k + 4u), pDec, c1 * x1);
            float4 zc0 = __builtin_elementwise_fma(oak_load4(z2p + k), pDec, c2 * x0), zc1 = __builtin_elementwise_fma(oak_load4(z2p + k + 4u), pDec, c2 * x1);
            float4 zd0 = __builtin_elementwise_fma(oak_load4(z3p + k), pDec, c3 * x0), zd1 = __builtin_elementwise_fma(oak_load4(z3p + k + 4u), pDec, c3 * x1);
            oak_store4(z0p + k, za0); oak_store4(z0p + k + 4u, za1); oak_store4(z1p + k, zb0); oak_store4(z1p + k + 4u, zb1);
            oak_store4(z2p + k, zc0); oak_store4(z2p + k + 4u, zc1); oak_store4(z3p + k, zd0); oak_store4(z3p + k + 4u, zd1);
            a0 = __builtin_elementwise_fma(za0, x0, a0); a1 = __builtin_elementwise_fma(za1, x1, a1);
            b0 = __builtin_elementwise_fma(zb0, x0, b0); b1 = __builtin_elementwise_fma(zb1, x1, b1);
            g0 = __builtin_elementwise_fma(zc0, x0, g0); g1 = __builtin_elementwise_fma(zc1, x1, g1);
            h0 = __builtin_elementwise_fma(zd0, x0, h0); h1 = __builtin_elementwise_fma(zd1, x1, h1);
        }
        float4 hs = oak_hsum4(a0 + a1, b0 + b1, g0 + g1, h0 + h1);
        float hv[4] = { hs.x, hs.y, hs.z, hs.w };
        for (uint32_t i = 0u; i < 4u; i++) { change[i] = fmaf((float)policyTuples, a->actorCredit[i], hv[i]); changeSquare = fmaf(change[i], change[i], changeSquare); }
        d = 4u;
    }
    for (; d + 2u <= actions; d += 2u) {
        float *restrict za = pTrace + d * count, *restrict zb = za + count;
        float4 ca = splat4(a->actorCredit[d]), cb = splat4(a->actorCredit[d + 1u]), a0 = splat4(0.0f), a1 = a0, b0 = a0, b1 = a0;
        for (uint32_t k = 0u; k < count; k += 8u) {
            float4 x0 = oak_load4(x + k), x1 = oak_load4(x + k + 4u);
            float4 za0 = __builtin_elementwise_fma(oak_load4(za + k), pDec, ca * x0), za1 = __builtin_elementwise_fma(oak_load4(za + k + 4u), pDec, ca * x1);
            float4 zb0 = __builtin_elementwise_fma(oak_load4(zb + k), pDec, cb * x0), zb1 = __builtin_elementwise_fma(oak_load4(zb + k + 4u), pDec, cb * x1);
            oak_store4(za + k, za0); oak_store4(za + k + 4u, za1); oak_store4(zb + k, zb0); oak_store4(zb + k + 4u, zb1);
            a0 = __builtin_elementwise_fma(za0, x0, a0); a1 = __builtin_elementwise_fma(za1, x1, a1);
            b0 = __builtin_elementwise_fma(zb0, x0, b0); b1 = __builtin_elementwise_fma(zb1, x1, b1);
        }
        change[d] = fmaf((float)policyTuples, a->actorCredit[d], oak_sum4(a0 + a1));
        change[d + 1u] = fmaf((float)policyTuples, a->actorCredit[d + 1u], oak_sum4(b0 + b1));
        changeSquare = fmaf(change[d], change[d], changeSquare);
        changeSquare = fmaf(change[d + 1u], change[d + 1u], changeSquare);
    }
    for (; d < actions; d++) {
        float *restrict z = pTrace + d * count; float4 cr = splat4(a->actorCredit[d]), c0 = splat4(0.0f), c1 = c0;
        for (uint32_t k = 0u; k < count; k += 8u) {
            float4 x0 = oak_load4(x + k), x1 = oak_load4(x + k + 4u);
            float4 z0 = __builtin_elementwise_fma(oak_load4(z + k), pDec, cr * x0), z1 = __builtin_elementwise_fma(oak_load4(z + k + 4u), pDec, cr * x1);
            oak_store4(z + k, z0); oak_store4(z + k + 4u, z1);
            c0 = __builtin_elementwise_fma(z0, x0, c0); c1 = __builtin_elementwise_fma(z1, x1, c1);
        }
        change[d] = fmaf((float)policyTuples, a->actorCredit[d], oak_sum4(c0 + c1));
        changeSquare = fmaf(change[d], change[d], changeSquare);
    }
    float actorStep = a->actorStep, a2 = actorStep * actorStep;
    float policyStep = (actorStep == 0.0f || (fabsf(actorStep) <= 1.0f && changeSquare <= 1.0f) || (a2 >= 1e-36f && a2 * changeSquare <= 0.999999f))
        ? actorStep : finiteOrZero(actorStep / fmaxf(1.0f, fabsf(actorStep) * sqrtf(changeSquare)));
    float qStep = a->qCellStep; uint32_t qAct = a->qCellAct, head = a->learner;
    if (prevTuples > 0u && (policyStep != 0.0f || valueDelta != 0.0f || qStep != 0.0f)) {
        uint32_t pre = prevTuples < 8u ? prevTuples : 8u;
        for (uint32_t i = 0u; i < pre; i++) __builtin_prefetch(a->cell + prevCells[i], 1, 3);
    }
    if (policyStep != 0.0f) {
        float dPol = fabsf(policyStep) * a->pTraceBound * 1.001f;
        if (__builtin_expect(dPol <= 1e30f && a->polBound[head] <= 1e30f, 1)) {
            a->polBound[head] += dPol;
            addPolicyRow(policy, pTrace, splat4(policyStep), actions * count);
        } else addPolicySlowAll(a, policy, pTrace, policyStep, actions, count);
    }
    if (prevTuples > 0u) {
        CellRow *cellTable = a->cell;
        float dCell = fmaxf(fmaxf(fabsf(valueDelta), fabsf(qStep)), fabsf(policyStep) * cMax) * 1.001f;
        int cellSafe = dCell <= 1e30f && a->cellBound <= 1e30f;
        if (cellSafe) a->cellBound += dCell;
        if (policyStep != 0.0f) {
            float4 stepLow = ((float4){ a->actorCredit[0], a->actorCredit[1], a->actorCredit[2], a->actorCredit[3] }) * policyStep;
            float4 stepHigh = ((float4){ a->actorCredit[4], a->actorCredit[5], 0.0f, 0.0f }) * policyStep;
            if (__builtin_expect(cellSafe, 1)) {
                if (!a->box) {
                    if (actions <= 4u) updateCellRowsFast(cellTable, prevCells, policyTuples, qAct, valueDelta, qStep, stepLow, stepHigh, 0, 1);
                    else updateCellRowsFast(cellTable, prevCells, policyTuples, qAct, valueDelta, qStep, stepLow, stepHigh, 1, 1);
                } else {
                    if (actions <= 4u) updateCellRowsFast(cellTable, prevCells, policyTuples, qAct, valueDelta, qStep, stepLow, stepHigh, 0, 0);
                    else updateCellRowsFast(cellTable, prevCells, policyTuples, qAct, valueDelta, qStep, stepLow, stepHigh, 1, 0);
                }
            } else {
                for (uint32_t t = 0u; t < policyTuples; t++) {
                    if (t + 8u < policyTuples) __builtin_prefetch(cellTable + prevCells[t + 8u], 1, 3);
                    CellRow *s = cellTable + prevCells[t]; float *c = s->policy;
                    if (valueDelta != 0.0f) s->value = keepFinite(s->value, s->value + valueDelta);
                    if (qStep != 0.0f) s->q[qAct] = keepFinite(s->q[qAct], s->q[qAct] + qStep);
                    float4 lo = oak_load4(c); oak_store4(c, oak_keepFinite4(lo, lo + stepLow));
                    if (actions > 4u) { float4 hi = oak_load4(c + 4u); oak_store4(c + 4u, oak_keepFinite4(hi, hi + stepHigh)); }
                }
            }
            if (valueDelta != 0.0f) for (uint32_t t = policyTuples; t < prevTuples; t++) {
                if (t + 8u < prevTuples) __builtin_prefetch(cellTable + prevCells[t + 8u], 1, 3);
                float *v = &cellTable[prevCells[t]].value; *v = keepFinite(*v, *v + valueDelta);
            }
        } else if (valueDelta != 0.0f || qStep != 0.0f) {
            for (uint32_t t = 0u; t < prevTuples; t++) {
                if (t + 8u < prevTuples) __builtin_prefetch(cellTable + prevCells[t + 8u], 1, 3);
                CellRow *s = cellTable + prevCells[t];
                if (valueDelta != 0.0f) s->value = keepFinite(s->value, s->value + valueDelta);
                if (qStep != 0.0f) s->q[qAct] = keepFinite(s->q[qAct], s->q[qAct] + qStep);
            }
        }
        if (!cellSafe) rescanCellBound(a);
    }
    float corrGate = a->obsCorr * a->obsCorr;
    int useTd = HEAD_SCALE[a->learner] == SCALE_TD || (a->box && HEAD_NATURAL[a->learner]);
    float actor = finiteOrZero(a->advantage * (!useTd ? fmaf(corrGate, a->tdInv - 1.0f, 1.0f) : 1.0f));
    float critic = a->tdNormalized * a->tdInv;
    if (actor == 0.0f && critic == 0.0f) return;
    const float *restrict unit = x + inputs, *restrict gates = a->unitGate + (1u - a->slot) * (OAK_UNITS / 2u);
    const float *restrict energies = a->unitEnergy + (1u - a->slot) * OAK_UNITS, *restrict px = permInputs(a, 1u - a->slot);
    float signals[OAK_UNITS]; uint32_t half = OAK_UNITS / 2u;
    float4 vCritic = splat4(critic), vActor = splat4(actor), vTrack = splat4(fmaf(corrGate, g.hiddenRate - g.trackingRate, g.trackingRate));
    float4 vOne = splat4(1.0f), vZero = splat4(0.0f), vGate = splat4(a->prevCorr * a->prevCorr);
    float4 evAcc = vZero, evMax4 = vZero, sigMax4 = vZero;
    const float *restrict polIn = policy + inputs, *restrict vwIn = a->valueWeight + inputs;
    for (uint32_t k = 0u; k < half; k += 8u) {
        float4 c0 = vZero, c1 = vZero, c2 = vZero, c3 = vZero;
        const float *pr = polIn + k;
        for (uint32_t i = 0u; i < actions; i++, pr += count) {
            float4 ac = splat4(a->actorCredit[i]);
            c0 = __builtin_elementwise_fma(ac, oak_load4(pr), c0); c1 = __builtin_elementwise_fma(ac, oak_load4(pr + 4u), c1);
            c2 = __builtin_elementwise_fma(ac, oak_load4(pr + half), c2); c3 = __builtin_elementwise_fma(ac, oak_load4(pr + half + 4u), c3);
        }
        float4 top0 = __builtin_elementwise_fma(vCritic, oak_load4(vwIn + k), vActor * c0), top1 = __builtin_elementwise_fma(vCritic, oak_load4(vwIn + k + 4u), vActor * c1);
        float4 top2 = __builtin_elementwise_fma(vCritic, oak_load4(vwIn + half + k), vActor * c2), top3 = __builtin_elementwise_fma(vCritic, oak_load4(vwIn + half + k + 4u), vActor * c3);
        float4 t0 = oak_load4(unit + k), t1 = oak_load4(unit + k + 4u), g0 = oak_load4(gates + k), g1 = oak_load4(gates + k + 4u);
        float4 ev0 = oak_load4(energies + k), ev1 = oak_load4(energies + k + 4u), ev2 = oak_load4(energies + half + k), ev3 = oak_load4(energies + half + k + 4u);
        float4 n0 = __builtin_elementwise_fma(top2 * vGate, g0, top0) * (vOne - t0 * t0) * vTrack, n1 = __builtin_elementwise_fma(top3 * vGate, g1, top1) * (vOne - t1 * t1) * vTrack;
        float4 n2 = (top2 * __builtin_elementwise_fma(vGate, t0 - vOne, vOne)) * (vOne - g0 * g0) * vTrack, n3 = (top3 * __builtin_elementwise_fma(vGate, t1 - vOne, vOne)) * (vOne - g1 * g1) * vTrack;
        float4 s0 = select4((lane4)(ev0 > vZero), oak_finiteOrZero4(n0 * oak_rcp4_fast(ev0)), vZero), s1 = select4((lane4)(ev1 > vZero), oak_finiteOrZero4(n1 * oak_rcp4_fast(ev1)), vZero);
        float4 s2 = select4((lane4)(ev2 > vZero), oak_finiteOrZero4(n2 * oak_rcp4_fast(ev2)), vZero), s3 = select4((lane4)(ev3 > vZero), oak_finiteOrZero4(n3 * oak_rcp4_fast(ev3)), vZero);
        oak_store4(signals + k, s0); oak_store4(signals + k + 4u, s1); oak_store4(signals + half + k, s2); oak_store4(signals + half + k + 4u, s3);
        evAcc += (ev0 + ev2) + (ev1 + ev3);
        evMax4 = __builtin_elementwise_max(evMax4, __builtin_elementwise_max(__builtin_elementwise_max(ev0, ev1), __builtin_elementwise_max(ev2, ev3)));
        sigMax4 = __builtin_elementwise_max(sigMax4, __builtin_elementwise_max(__builtin_elementwise_max(__builtin_elementwise_abs(s0), __builtin_elementwise_abs(s1)), __builtin_elementwise_max(__builtin_elementwise_abs(s2), __builtin_elementwise_abs(s3))));
    }
    float evChk = oak_sum4(evAcc), evMax = vmaxvq_f32((float32x4_t)evMax4), sigMax = vmaxvq_f32((float32x4_t)sigMax4);
    float dProj = sigMax * sqrtf(fmaxf(evMax, 0.0f)) * 1.001f;
    if (__builtin_expect((evChk - evChk) == 0.0f && evMax <= 1e30f && dProj <= 1e30f && a->projBound <= 1e30f, 1)) {
        a->projBound += dProj;
        for (uint32_t j = 0u; j < OAK_UNITS; j += 4u) {
            float32x4_t sv = (float32x4_t)oak_load4(signals + j);
            OAK_UPD_PROJ_ROW(a->projection + j * OAK_FANIN, px, a->unitBlock + j * OAK_FANIN_BLOCKS, sv, 0);
            OAK_UPD_PROJ_ROW(a->projection + (j + 1u) * OAK_FANIN, px, a->unitBlock + (j + 1u) * OAK_FANIN_BLOCKS, sv, 1);
            OAK_UPD_PROJ_ROW(a->projection + (j + 2u) * OAK_FANIN, px, a->unitBlock + (j + 2u) * OAK_FANIN_BLOCKS, sv, 2);
            OAK_UPD_PROJ_ROW(a->projection + (j + 3u) * OAK_FANIN, px, a->unitBlock + (j + 3u) * OAK_FANIN_BLOCKS, sv, 3);
        }
    } else updateProjAllSlow(a, px, signals);
}

uint32_t oak_stateWords(void) { return (uint32_t)(sizeof(OakAgent) / sizeof(uint32_t)); }

void oak_init(uint32_t *state, uint32_t lane, uint32_t env, uint32_t seed, const float *obs) {
    (void)lane; (void)obs;
    OakAgent *a = (OakAgent *)state; __builtin_memset(a, 0, sizeof *a);
    uint32_t obsN = env_obs(env); OakGeometry g = oak_geometry(obsN);
    a->g = g; a->betaFloor = 2.0f * oak_log(g.trackingRate);
    a->observationCount = obsN; a->actionCount = env_actions(env); a->box = env_box(env);
    uint32_t direct = OAK_PROJECTED_COUNT(obsN), sources = direct + OAK_TRACE_LEVELS * obsN;
    uint32_t permStride = ((sources + 4u + 3u) / 4u) * 4u;
    a->sources = sources; a->permStride = permStride;
    for (uint32_t p = 0u; p < OAK_PERMS; p++) {
        uint16_t *perm = a->inputPerm + p * sources;
        for (uint32_t i = 0u; i < sources; i++) perm[i] = (uint16_t)(i < direct ? i : g.projected + (i - direct));
        for (uint32_t i = sources - 1u; i > 0u; i--) {
            u4 r = philox4x32(seed, RNG_STREAM_ACT, p * OAK_SOURCES_MAX + i, 11u, 0u, 0u);
            uint32_t pick = (uint32_t)(((uint64_t)r.x * (i + 1u)) >> 32u);
            uint16_t tmp = perm[i]; perm[i] = perm[pick]; perm[pick] = tmp;
        }
    }
    if (sources >= OAK_FANIN) {
        for (uint32_t j = 0u; j < OAK_UNITS; j++) {
            uint8_t used[OAK_INPUT_MAX] = {0}; uint16_t *dst = a->unitBlock + j * OAK_FANIN_BLOCKS;
            for (uint32_t b = 0u; b < OAK_FANIN_BLOCKS; b++) {
                uint32_t p = b & (OAK_PERMS - 1u), bestOff = 0u, bestOverlap = 5u;
                const uint16_t *perm = a->inputPerm + p * sources;
                for (uint32_t tries = 0u; tries < 8u && bestOverlap > 0u; tries++) {
                    u4 r = philox4x32(seed, RNG_STREAM_ACT, (j * OAK_FANIN_BLOCKS + b) * 8u + tries, 12u, 0u, 0u);
                    uint32_t off = (uint32_t)(((uint64_t)r.x * sources) >> 32u), ov = 0u;
                    for (uint32_t t = 0u; t < 4u; t++) ov += used[perm[(off + t) % sources]];
                    if (ov < bestOverlap) { bestOverlap = ov; bestOff = off; }
                }
                for (uint32_t t = 0u; t < 4u; t++) used[perm[(bestOff + t) % sources]] = 1u;
                dst[b] = (uint16_t)((p * permStride + bestOff) * (uint32_t)sizeof(float));
            }
        }
    } else {
        for (uint32_t j = 0u; j < OAK_UNITS; j++) {
            uint16_t *dst = a->unitBlock + j * OAK_FANIN_BLOCKS;
            for (uint32_t b = 0u; b < OAK_FANIN_BLOCKS; b += 4u) {
                u4 r = philox4x32(seed, RNG_STREAM_ACT, j * OAK_FANIN_BLOCKS + b, 12u, 0u, 0u);
                uint32_t pick[4] = { r.x, r.y, r.z, r.w };
                for (uint32_t u = 0u; u < 4u; u++)
                    dst[b + u] = (uint16_t)((((b + u) & (OAK_PERMS - 1u)) * permStride + (uint32_t)(((uint64_t)pick[u] * sources) >> 32u)) * (uint32_t)sizeof(float));
            }
        }
    }
    float fanIn = oak_rsqrt((float)OAK_FANIN), pMax = 0.0f;
    for (uint32_t n = 0u; n < OAK_UNITS * OAK_FANIN; n += 4u) {
        float4 w = normalFrom4(philox4x32(seed, RNG_STREAM_ACT, n, 8u, 0u, 0u), philox4x32(seed, RNG_STREAM_ACT, n, 9u, 0u, 0u)) * fanIn;
        oak_store4(a->projection + n, w); pMax = fmaxf(pMax, vmaxvq_f32((float32x4_t)__builtin_elementwise_abs(w)));
    }
    a->projBound = pMax;
    uint32_t hash = seed * 2654435761u + 977u, action = hash % OAK_POLICY_MAX;
    float sign = ((hash >> 16u) & 1u) == 0u ? 1.0f : -1.0f;
    for (uint32_t h = 0u; h < OAK_HEADS; h++) {
        a->policy[(h * OAK_POLICY_MAX + action) * g.features] = sign * (1.0f / (float)OAK_ACTION_PARAMS);
        a->polBound[h] = 1.0f;
    }
    float logAlpha = oak_log(g.ridgeFloor);
    for (uint32_t o = 0u; o < obsN; o++) { a->chAlpha[o] = g.ridgeFloor; a->chBeta[o] = logAlpha; }
    a->rewardChannel.alpha = g.ridgeFloor; a->rewardChannel.beta = logAlpha;
    for (uint32_t h = 0u; h < OAK_HEADS; h++) a->actorMeta[h].alpha = 1.0f;
}

static __attribute__((noinline)) void rotate(OakAgent *a, uint32_t seed, uint32_t step) {
    uint32_t head = a->activeHead;
    float blockMean = a->blockReward / (float)OAK_BLOCK, rate = 1.0f / (a->headWeight[head] + 1.0f);
    a->blockCount += 1u;
    a->headMean[head] = keepFinite(a->headMean[head], a->headMean[head] + rate * (blockMean - a->headMean[head]));
    a->headSquare[head] = keepFinite(a->headSquare[head], a->headSquare[head] + rate * (blockMean * blockMean - a->headSquare[head]));
    a->headWeight[head] += 1.0f; a->headBlocks[head] += 1u; a->blockReward = 0.0f; a->blockSteps = 0u;
    float strongest = -1.0f;
    for (uint32_t o = 0u; o < a->observationCount; o++) {
        float s = 0.0f; for (uint32_t d = 0u; d < a->actionCount; d++) s += fabsf(a->control[o * OAK_CELL_STRIDE + d]);
        if (s > strongest) { strongest = s; a->self = o; }
    }
    uint32_t arms = a->box ? OAK_HEADS : OAK_ARMS, next = arms;
    if ((a->blockCount & (a->blockCount - 1u)) == 0u)
        for (uint32_t h = 0u; h < arms; h++)
            if (h != head && (h >= OAK_HEADS || h == a->learner) && a->headBlocks[h] >= 2u) {
                a->headMean[h] = a->headSquare[h] = a->headWeight[h] = 0.0f; a->headBlocks[h] = 0u;
            }
    for (uint32_t h = 0u; h < arms && next == arms; h++) if (a->headBlocks[h] < 2u) next = h;
    if (next == arms) {
        float4 low = normalFrom4(philox4x32(seed, RNG_STREAM_ACT, step, 4u, 0u, 0u), philox4x32(seed, RNG_STREAM_ACT, step, 5u, 0u, 0u));
        float4 high = normalFrom4(philox4x32(seed, RNG_STREAM_ACT, step, 6u, 0u, 0u), philox4x32(seed, RNG_STREAM_ACT, step, 7u, 0u, 0u));
        float draw[8] = { low.x, low.y, low.z, low.w, high.x, high.y, high.z, high.w }, bestSample = -INFINITY, pooled = 0.0f, weight = 0.0f;
        for (uint32_t h = 0u; h < arms; h++) { pooled += a->headWeight[h] * fmaxf(a->headSquare[h] - a->headMean[h] * a->headMean[h], 0.0f); weight += a->headWeight[h]; }
        pooled /= weight; next = 0u;
        for (uint32_t h = 0u; h < arms; h++) {
            float own = fmaxf(a->headSquare[h] - a->headMean[h] * a->headMean[h], 0.0f);
            float sample = a->headMean[h] + draw[h] * sqrtf((a->headWeight[h] * own + pooled) / ((a->headWeight[h] + 1.0f) * a->headWeight[h]));
            if (sample > bestSample) { bestSample = sample; next = h; }
        }
    }
    if (next < OAK_HEADS && next != a->learner) {
        uint32_t count = a->g.features;
        for (uint32_t k = 0u; k < count; k++) a->valueTrace[k] = 0.0f;
        for (uint32_t k = 0u; k < a->actionCount * count; k++) a->policyTrace[k] = 0.0f;
        float *dst = &a->cellPolicyBank[a->learner][0][0], *src = &a->cellPolicyBank[next][0][0];
        for (uint32_t i = 0u; i < OAK_TABLE; i++, dst += OAK_CELL_STRIDE, src += OAK_CELL_STRIDE) {
            float *cp = a->cell[i].policy;
            oak_store4(dst, oak_load4(cp)); oak_store4(dst + 4u, oak_load4(cp + 4u));
            oak_store4(cp, oak_load4(src)); oak_store4(cp + 4u, oak_load4(src + 4u));
        }
        a->pTraceBound = 0.0f; a->learner = next;
    }
    a->activeHead = next;
}

__attribute__((noinline)) void oak_act(uint32_t *state, uint32_t lane, uint32_t env, uint32_t seed, uint32_t step, const float *obs, float reward, uint32_t done, float *ctrl) {
    (void)lane; (void)env; (void)done;
    OakAgent *a = (OakAgent *)state;
    a->blockReward += finiteOrZero(reward);
    if (++a->blockSteps == OAK_BLOCK) rotate(a, seed, step);
    a->reward = reward; a->stepCount = step; a->slot = step & 1u;
    perceive(a, obs);
    evaluate(a);
    if (!a->box && step > 0u) for (uint32_t key = 0u; key < OAK_KEYS; key++) learnTable(a, key);
    sampleAction(a, seed, step, ctrl);
    learn(a);
    a->stepCount = step + 1u;
}
