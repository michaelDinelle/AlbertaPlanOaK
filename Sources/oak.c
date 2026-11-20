#include "geometry.h"
#include "policy.h"

_Static_assert(OAK_LANES == 1u, "");
_Static_assert(OAK_POLICY_MAX == 6u, "");
_Static_assert(OAK_UNITS % 8u == 0u, "");
_Static_assert(OAK_FEATURE_MAX % 8u == 0u, "");
_Static_assert(OAK_TUPLES == 23u, "");

static inline float4 oak_load4(const float *p) { float4 v; __builtin_memcpy(&v, p, sizeof v); return v; }
static inline void oak_store4(float *p, float4 v) { __builtin_memcpy(p, &v, sizeof v); }
static inline lane4 oak_finite4(float4 v) { return (lane4)vcaltq_f32((float32x4_t)v, vdupq_n_f32(INFINITY)); }
static inline float oak_sum4(float4 v) { return (v.x + v.y) + (v.z + v.w); }
static inline float4 oak_keepFinite4(float4 oldv, float4 newv) { return select4(oak_finite4(newv), newv, oldv); }
static inline float4 oak_finiteOrZero4(float4 v) { return select4(oak_finite4(v), v, splat4(0.0f)); }

static inline float4 oak_rsqrt4(float4 x) {
    float32x4_t v = (float32x4_t)x;
    float32x4_t r = vrsqrteq_f32(v);
    r = vmulq_f32(r, vrsqrtsq_f32(vmulq_f32(v, r), r));
    r = vmulq_f32(r, vrsqrtsq_f32(vmulq_f32(v, r), r));
    return select4((lane4)(x > splat4(0.0f)), (float4)r, splat4(0.0f));
}

static inline float4 oak_rsqrt4_fast(float4 x) {
    float32x4_t v = (float32x4_t)x;
    float32x4_t r = vrsqrteq_f32(v);
    r = vmulq_f32(r, vrsqrtsq_f32(vmulq_f32(v, r), r));
    return select4((lane4)(x > splat4(0.0f)), (float4)r, splat4(0.0f));
}

static inline float4 oak_rcp4_fast(float4 x) {
    float32x4_t v = (float32x4_t)x;
    float32x4_t r = vrecpeq_f32(v);
    r = vmulq_f32(r, vrecpsq_f32(v, r));
    return (float4)r;
}

static inline float4 oak_hsum4(float4 a, float4 b, float4 c, float4 d) {
    float32x4_t p0 = vpaddq_f32((float32x4_t)a, (float32x4_t)b);
    float32x4_t p1 = vpaddq_f32((float32x4_t)c, (float32x4_t)d);
    return (float4)vpaddq_f32(p0, p1);
}

static inline float4 oak_tanh4(float4 x) {
    float4 a = __builtin_elementwise_abs(x);
    float4 x2 = a * a;
    float4 num = a * __builtin_elementwise_fma(x2, __builtin_elementwise_fma(x2, __builtin_elementwise_fma(x2, splat4(1.0f), splat4(378.0f)), splat4(17325.0f)), splat4(135135.0f));
    float4 den = __builtin_elementwise_fma(x2, __builtin_elementwise_fma(x2, __builtin_elementwise_fma(x2, splat4(28.0f), splat4(3150.0f)), splat4(62370.0f)), splat4(135135.0f));
    float4 t = select4((lane4)(a > splat4(5.0f)), splat4(1.0f), num * oak_rcp4_fast(den));
    float4 signedT = (float4)(((lane4)t) | (((lane4)x) & 0x80000000u));
    return select4((lane4)(x != x), x, signedT);
}

typedef struct {
    float mean, second, alpha, beta, trace, energy;
} Channel;

typedef struct {
    float q[OAK_POLICY_MAX];
    float pad;
    float value;
} CellVQ;
_Static_assert(sizeof(CellVQ) == 32u, "");

typedef struct {
    float q[OAK_POLICY_MAX];
    uint32_t owner;
    uint16_t visits[OAK_POLICY_MAX];
} TableRowHead;

typedef struct {
    TableRowHead h;
    float modelReward[OAK_POLICY_MAX];
    uint16_t modelNext[OAK_POLICY_MAX][2];
    uint16_t predCell[OAK_DYNA_PREDS];
    uint16_t predMeta;
    uint8_t modelProb[OAK_POLICY_MAX][3];
    uint8_t modelTag[OAK_POLICY_MAX][2];
} ExactRow;
_Static_assert(sizeof(ExactRow) == 128u, "");

typedef struct {
    TableRowHead h;
    uint8_t distance[OAK_POLICY_MAX];
    uint8_t pad[18];
} HistRow;
_Static_assert(sizeof(HistRow) == 64u, "");

typedef struct {
    float feature[2u * OAK_FEATURE_MAX];
    float permInput[2u * OAK_PERMS * OAK_PERM_STRIDE];
    float valueWeight[OAK_FEATURE_MAX], valueTrace[OAK_FEATURE_MAX];
    float policy[OAK_HEADS * OAK_POLICY_MAX * OAK_FEATURE_MAX], policyTrace[OAK_POLICY_MAX * OAK_FEATURE_MAX];
    float projection[OAK_UNITS * OAK_FANIN], unitEnergy[2u * OAK_UNITS], unitGate[OAK_UNITS];
    uint16_t inputPerm[OAK_PERMS * OAK_SOURCES_MAX], unitBlock[OAK_UNITS * OAK_FANIN_BLOCKS];
    float traceLevel[OAK_TRACE_LEVELS * OAK_OBSERVATION_LIMIT], traceMean[OAK_TRACE_LEVELS * OAK_OBSERVATION_LIMIT], traceSecond[OAK_TRACE_LEVELS * OAK_OBSERVATION_LIMIT];
    float settledMean[OAK_TRACE_LEVELS * OAK_OBSERVATION_LIMIT], settledSecond[OAK_TRACE_LEVELS * OAK_OBSERVATION_LIMIT];
    float control[OAK_OBSERVATION_LIMIT * OAK_CELL_STRIDE];
    uint32_t cellActive[2u * OAK_CELLS];
    float chMean[OAK_OBSERVATION_LIMIT], chSecond[OAK_OBSERVATION_LIMIT], chAlpha[OAK_OBSERVATION_LIMIT], chBeta[OAK_OBSERVATION_LIMIT], chTrace[OAK_OBSERVATION_LIMIT], chEnergy[OAK_OBSERVATION_LIMIT], chCorr[OAK_OBSERVATION_LIMIT];
    Channel rewardChannel, actorMeta[OAK_HEADS];
    float headMean[OAK_ARMS], headSquare[OAK_ARMS], headWeight[OAK_ARMS], blockReward, obsCorr, prevCorr;
    uint32_t headBlocks[OAK_ARMS], activeHead, learner, blockSteps, blockCount, self;
    float actionSignal[OAK_ACTION_PARAMS];
    float actorCredit[OAK_POLICY_MAX], actorCreditStaging[OAK_POLICY_MAX], logit[OAK_POLICY_MAX];
    float reward, td, tdNormalized, tdInv, tdSquare, advantage, advantageCenter, averageReward, actorStep, betaFloor;
    float rewardMean, rewardVariance, aspiration, rewardSquare, rewardMagnitude;
    OakGeometry g;
    uint32_t observationCount, actionCount, box, stepCount, slot, sources, permStride;
    uint32_t qCell[2u * OAK_KEYS], qTag[2u * OAK_KEYS], action, qCellAct;
    float qSum[OAK_KEYS], qNoise[OAK_KEYS];
    uint32_t qTried[OAK_KEYS], qOpen[OAK_KEYS];
    float qCellValue[OAK_POLICY_MAX], qCellTaken, qCellGain, qCellNoise, qCellStep;
    uint32_t psTag[OAK_DYNA_QUEUE], psCount;
    float psPrio[OAK_DYNA_QUEUE];
    uint16_t psCell[OAK_DYNA_QUEUE];
    uint32_t planGeneration, planLength, planIndex;
    _Alignas(128) CellVQ cellVQ[OAK_TABLE];
    _Alignas(128) float cellPolicy[OAK_HEADS * OAK_TABLE * OAK_CELL_STRIDE];
    _Alignas(128) ExactRow exactTable[OAK_EXACT];
    _Alignas(128) HistRow histTable[OAK_HIST];
    uint16_t planQueue[OAK_PLAN_BUDGET], planCells[OAK_PLAN_BUDGET];
    uint8_t planActions[OAK_PLAN_BUDGET];
    uint16_t planParent[OAK_EXACT];
    uint8_t planParentAction[OAK_EXACT];
    uint32_t planStamp[OAK_EXACT];
} OakAgent;

enum { SCALE_TD, SCALE_RAW, SCALE_MAGNITUDE };
static const uint8_t HEAD_SCALE[OAK_HEADS] = { SCALE_TD, SCALE_RAW, SCALE_RAW };
static const uint8_t HEAD_NATURAL[OAK_HEADS] = { 0u, 1u, 0u };
static const uint8_t HEAD_ASPIRE[OAK_HEADS] = { 1u, 1u, 0u };
static const uint8_t HEAD_TRUST[OAK_HEADS] = { 0u, 0u, 1u };
static const float HEAD_DECAY[OAK_HEADS] = { 0.5f, 0.75f, 0.875f };
static const uint8_t TABLE_KEY[OAK_TABLE_ARMS] = { 0u, 1u, 0u, 1u, 0u };
enum { TABLE_GREEDY, TABLE_FRONTIER, TABLE_PLAN, TABLE_CELLQ };
static const uint8_t TABLE_MODE[OAK_TABLE_ARMS] = { TABLE_GREEDY, TABLE_GREEDY, TABLE_PLAN, TABLE_FRONTIER, TABLE_CELLQ };

static inline float *features(OakAgent *a, uint32_t slot) { return a->feature + slot * a->g.features; }
static inline float *permInputs(OakAgent *a, uint32_t slot) { return a->permInput + slot * OAK_PERMS * a->permStride; }
static inline float *headPolicy(OakAgent *a) { return a->policy + a->learner * OAK_POLICY_MAX * a->g.features; }
static inline float *headCellBase(OakAgent *a) { return a->cellPolicy + a->learner * OAK_TABLE * OAK_CELL_STRIDE; }
static inline TableRowHead *tableRow(OakAgent *a, uint32_t key, uint32_t cell) {
    return key == 0u ? &a->exactTable[cell].h : &a->histTable[cell].h;
}
static inline const TableRowHead *tableRowConst(const OakAgent *a, uint32_t key, uint32_t cell) {
    return key == 0u ? &a->exactTable[cell].h : &a->histTable[cell].h;
}
static inline uint8_t tag8(uint32_t tag) {
    return (uint8_t)(((tag ^ (tag >> 8u) ^ (tag >> 16u) ^ (tag >> 24u)) & 0xffu) | 1u);
}

static inline float trackingRateFor(uint32_t step, uint32_t cap) {
    return fmaxf(1.0f / ((float)step + 1.0f), 1.0f / (float)(cap > 1u ? cap : 1u));
}

static inline float dotProduct(const float *x, const float *y, uint32_t n) {
    float4 s0 = splat4(0.0f), s1 = s0, s2 = s0, s3 = s0;
    uint32_t k = 0u;
    for (; k + 16u <= n; k += 16u) {
        s0 = __builtin_elementwise_fma(oak_load4(x + k), oak_load4(y + k), s0);
        s1 = __builtin_elementwise_fma(oak_load4(x + k + 4u), oak_load4(y + k + 4u), s1);
        s2 = __builtin_elementwise_fma(oak_load4(x + k + 8u), oak_load4(y + k + 8u), s2);
        s3 = __builtin_elementwise_fma(oak_load4(x + k + 12u), oak_load4(y + k + 12u), s3);
    }
    for (; k < n; k += 4u) s0 = __builtin_elementwise_fma(oak_load4(x + k), oak_load4(y + k), s0);
    return oak_sum4((s0 + s1) + (s2 + s3));
}

static inline void dotProduct2(const float *w, const float *x, const float *y, uint32_t n, float *ox, float *oy) {
    float4 a0 = splat4(0.0f), a1 = a0, a2 = a0, a3 = a0;
    float4 b0 = splat4(0.0f), b1 = b0, b2 = b0, b3 = b0;
    uint32_t k = 0u;
    for (; k + 16u <= n; k += 16u) {
        float4 w0 = oak_load4(w + k), w1 = oak_load4(w + k + 4u), w2 = oak_load4(w + k + 8u), w3 = oak_load4(w + k + 12u);
        a0 = __builtin_elementwise_fma(w0, oak_load4(x + k), a0);
        b0 = __builtin_elementwise_fma(w0, oak_load4(y + k), b0);
        a1 = __builtin_elementwise_fma(w1, oak_load4(x + k + 4u), a1);
        b1 = __builtin_elementwise_fma(w1, oak_load4(y + k + 4u), b1);
        a2 = __builtin_elementwise_fma(w2, oak_load4(x + k + 8u), a2);
        b2 = __builtin_elementwise_fma(w2, oak_load4(y + k + 8u), b2);
        a3 = __builtin_elementwise_fma(w3, oak_load4(x + k + 12u), a3);
        b3 = __builtin_elementwise_fma(w3, oak_load4(y + k + 12u), b3);
    }
    for (; k < n; k += 4u) {
        float4 w0 = oak_load4(w + k);
        a0 = __builtin_elementwise_fma(w0, oak_load4(x + k), a0);
        b0 = __builtin_elementwise_fma(w0, oak_load4(y + k), b0);
    }
    *ox = oak_sum4((a0 + a1) + (a2 + a3));
    *oy = oak_sum4((b0 + b1) + (b2 + b3));
}

static inline void dotRows2(const float *r0, const float *r1, const float *x, uint32_t n, float *o0, float *o1) {
    float4 a0 = splat4(0.0f), a1 = a0, a2 = a0, a3 = a0;
    float4 b0 = splat4(0.0f), b1 = b0, b2 = b0, b3 = b0;
    uint32_t k = 0u;
    for (; k + 16u <= n; k += 16u) {
        float4 x0 = oak_load4(x + k), x1 = oak_load4(x + k + 4u), x2 = oak_load4(x + k + 8u), x3 = oak_load4(x + k + 12u);
        a0 = __builtin_elementwise_fma(oak_load4(r0 + k), x0, a0);
        b0 = __builtin_elementwise_fma(oak_load4(r1 + k), x0, b0);
        a1 = __builtin_elementwise_fma(oak_load4(r0 + k + 4u), x1, a1);
        b1 = __builtin_elementwise_fma(oak_load4(r1 + k + 4u), x1, b1);
        a2 = __builtin_elementwise_fma(oak_load4(r0 + k + 8u), x2, a2);
        b2 = __builtin_elementwise_fma(oak_load4(r1 + k + 8u), x2, b2);
        a3 = __builtin_elementwise_fma(oak_load4(r0 + k + 12u), x3, a3);
        b3 = __builtin_elementwise_fma(oak_load4(r1 + k + 12u), x3, b3);
    }
    for (; k < n; k += 4u) {
        float4 x0 = oak_load4(x + k);
        a0 = __builtin_elementwise_fma(oak_load4(r0 + k), x0, a0);
        b0 = __builtin_elementwise_fma(oak_load4(r1 + k), x0, b0);
    }
    *o0 = oak_sum4((a0 + a1) + (a2 + a3));
    *o1 = oak_sum4((b0 + b1) + (b2 + b3));
}

static inline void dotRows4(const float *r0, const float *r1, const float *r2, const float *r3, const float *x, uint32_t n, float *o0, float *o1, float *o2, float *o3) {
    float4 a0 = splat4(0.0f), a1 = a0, b0 = a0, b1 = a0, c0 = a0, c1 = a0, d0 = a0, d1 = a0;
    for (uint32_t k = 0u; k < n; k += 8u) {
        float4 x0 = oak_load4(x + k), x1 = oak_load4(x + k + 4u);
        a0 = __builtin_elementwise_fma(oak_load4(r0 + k), x0, a0);
        b0 = __builtin_elementwise_fma(oak_load4(r1 + k), x0, b0);
        c0 = __builtin_elementwise_fma(oak_load4(r2 + k), x0, c0);
        d0 = __builtin_elementwise_fma(oak_load4(r3 + k), x0, d0);
        a1 = __builtin_elementwise_fma(oak_load4(r0 + k + 4u), x1, a1);
        b1 = __builtin_elementwise_fma(oak_load4(r1 + k + 4u), x1, b1);
        c1 = __builtin_elementwise_fma(oak_load4(r2 + k + 4u), x1, c1);
        d1 = __builtin_elementwise_fma(oak_load4(r3 + k + 4u), x1, d1);
    }
    float4 s = oak_hsum4(a0 + a1, b0 + b1, c0 + c1, d0 + d1);
    *o0 = s.x;
    *o1 = s.y;
    *o2 = s.z;
    *o3 = s.w;
}

static inline __attribute__((always_inline)) void metaStep(Channel *c, float signal, float inverseAge, float metaRate, float decay, float betaFloor) {
    float metaGradient = signal * c->trace;
    float gradEnergy = keepFinite(c->energy, c->energy + inverseAge * (metaGradient * metaGradient - c->energy));
    float normalizedMetaGradient = metaGradient * oak_rsqrt(fmaxf(gradEnergy, 0.0f) + inverseAge);
    float nextBeta = keepFinite(c->beta, oak_clamp(c->beta + metaRate * normalizedMetaGradient, betaFloor, 0.0f));
    float nextAlpha = keepFinite(c->alpha, oak_exp(nextBeta));
    float nextH = keepFinite(c->trace, (1.0f - decay) * fmaxf(0.0f, 1.0f - c->alpha) * c->trace + c->alpha * signal);
    if (isfinite(signal)) {
        c->alpha = nextAlpha;
        c->beta = nextBeta;
        c->trace = nextH;
        c->energy = gradEnergy;
    }
}

static inline __attribute__((always_inline)) float standardizeChannel(Channel *c, float x, float inverseAge, float floorRate, float metaRate, float decay, float betaFloor, float ridgeFloor) {
    int finite = isfinite(x);
    float signal = x - c->mean;
    if (finite) metaStep(c, signal, inverseAge, metaRate, decay, betaFloor);
    float rate = fmaxf(floorRate, c->alpha);
    if (finite) {
        c->mean = keepFinite(c->mean, c->mean + rate * (x - c->mean));
        c->second = keepFinite(c->second, c->second + rate * (x * x - c->second));
    }
    return (x - c->mean) * oak_rsqrt(fmaxf(c->second - c->mean * c->mean, 0.0f) + ridgeFloor);
}

static inline uint32_t tableCell(uint32_t hash, uint32_t bits) {
    hash ^= hash >> 15u;
    hash *= 0x2C1B3C6Du;
    hash ^= hash >> 12u;
    return hash >> (32u - bits);
}

static inline uint32_t cellCode(float value, uint32_t fine) {
    return fine ? (uint32_t)oak_clamp(floorf(value + 3.5f), 0.0f, 6.0f) : (uint32_t)oak_clamp(floorf(value + 1.5f), 0.0f, 2.0f);
}

static inline lane4 cellCode4(float4 value, uint32_t fine) {
    float4 bias = splat4(fine ? 3.5f : 1.5f), hi = splat4(fine ? 6.0f : 2.0f);
    float4 f = __builtin_elementwise_floor(value + bias);
    float4 c = __builtin_elementwise_min(__builtin_elementwise_max(f, splat4(0.0f)), hi);
    return (lane4)vcvtq_u32_f32((float32x4_t)c);
}

static inline uint32x4_t mixHash4(uint32x4_t h) {
    h = veorq_u32(h, vshrq_n_u32(h, 15));
    h = vmulq_n_u32(h, 0x2C1B3C6Du);
    h = veorq_u32(h, vshrq_n_u32(h, 12));
    return vshrq_n_u32(h, 32 - OAK_TABLE_BITS);
}

static inline uint32x4_t fnvStep4(uint32x4_t h, uint32x4_t a, uint32x4_t b, uint32x4_t c, uint32x4_t d) {
    uint32x4_t ab0 = vzip1q_u32(a, b), ab1 = vzip2q_u32(a, b);
    uint32x4_t cd0 = vzip1q_u32(c, d), cd1 = vzip2q_u32(c, d);
    uint32x4_t t0 = vreinterpretq_u32_u64(vzip1q_u64(vreinterpretq_u64_u32(ab0), vreinterpretq_u64_u32(cd0)));
    uint32x4_t t1 = vreinterpretq_u32_u64(vzip2q_u64(vreinterpretq_u64_u32(ab0), vreinterpretq_u64_u32(cd0)));
    uint32x4_t t2 = vreinterpretq_u32_u64(vzip1q_u64(vreinterpretq_u64_u32(ab1), vreinterpretq_u64_u32(cd1)));
    uint32x4_t t3 = vreinterpretq_u32_u64(vzip2q_u64(vreinterpretq_u64_u32(ab1), vreinterpretq_u64_u32(cd1)));
    h = vmulq_n_u32(veorq_u32(h, t0), 0x01000193u);
    h = vmulq_n_u32(veorq_u32(h, t1), 0x01000193u);
    h = vmulq_n_u32(veorq_u32(h, t2), 0x01000193u);
    h = vmulq_n_u32(veorq_u32(h, t3), 0x01000193u);
    return h;
}

static __attribute__((noinline)) void encode(OakAgent *a, const float *x, const float *obs) {
    OakGeometry g = a->g;
    uint32_t observationCount = a->observationCount;
    uint32_t traceTotal = OAK_TRACE_LEVELS * observationCount;
    uint32_t fineCode[(1u + OAK_TRACE_LEVELS) * OAK_OBSERVATION_LIMIT], coarseCode[(1u + OAK_TRACE_LEVELS) * OAK_OBSERVATION_LIMIT], settledCode[OAK_TRACE_LEVELS * OAK_OBSERVATION_LIMIT];
    uint32x4_t vTwo = vdupq_n_u32(2u), vEight = vdupq_n_u32(8u), vNine = vdupq_n_u32(9u), vZeroU = vdupq_n_u32(0u);
    uint32_t o = 0u;
    for (; o + 4u <= observationCount; o += 4u) {
        float4 v = oak_load4(x + o);
        uint32x4_t fc = (uint32x4_t)cellCode4(v, 1u);
        uint32x4_t cc = vminq_u32(vqsubq_u32(fc, vTwo), vTwo);
        vst1q_u32(fineCode + o, fc);
        vst1q_u32(coarseCode + o, cc);
    }
    for (; o < observationCount; o++) {
        uint32_t fc = cellCode(x[o], 1u);
        fineCode[o] = fc;
        coarseCode[o] = fc <= 2u ? 0u : fc >= 4u ? 2u : 1u;
    }
    float momentRate = trackingRateFor(a->stepCount, OAK_FEATURE_TIMESCALE), settleRate = 1.0f / ((float)a->stepCount + 1.0f);
    float ridge = a->stepCount < 2u * OAK_CAPACITY ? 1.0f / ((float)a->stepCount + 1.0f) : g.ridgeFloor;
    float4 vMoment = splat4(momentRate), vSettle = splat4(settleRate), vRidge = splat4(ridge), vZero = splat4(0.0f), vQuarter = splat4(0.25f);
    uint32_t c = 0u;
    for (; c + 4u <= traceTotal; c += 4u) {
        float4 value = oak_load4(a->traceLevel + c);
        float4 mean = oak_load4(a->traceMean + c), second = oak_load4(a->traceSecond + c);
        float4 center = oak_load4(a->settledMean + c), spread = oak_load4(a->settledSecond + c);
        float4 sq = value * value;
        mean = oak_keepFinite4(mean, __builtin_elementwise_fma(vMoment, value - mean, mean));
        second = oak_keepFinite4(second, __builtin_elementwise_fma(vMoment, sq - second, second));
        center = oak_keepFinite4(center, __builtin_elementwise_fma(vSettle, value - center, center));
        spread = oak_keepFinite4(spread, __builtin_elementwise_fma(vSettle, sq - spread, spread));
        oak_store4(a->traceMean + c, mean);
        oak_store4(a->traceSecond + c, second);
        oak_store4(a->settledMean + c, center);
        oak_store4(a->settledSecond + c, spread);
        float4 norm = oak_finiteOrZero4((value - mean) * oak_rsqrt4_fast(__builtin_elementwise_max(__builtin_elementwise_fma(-mean, mean, second), vZero) + vRidge));
        uint32x4_t fc = (uint32x4_t)cellCode4(norm, 1u);
        uint32x4_t cc = vaddq_u32(vminq_u32(vqsubq_u32(fc, vTwo), vTwo), vEight);
        vst1q_u32(fineCode + observationCount + c, fc);
        vst1q_u32(coarseCode + observationCount + c, cc);
        float4 sd = value - center;
        float4 sv = __builtin_elementwise_max(__builtin_elementwise_fma(-center, center, spread), vZero) + vRidge;
        lane4 sfin = oak_finite4(sd) & oak_finite4(sv);
        float4 sd2 = sd * sd, sq4 = vQuarter * sv;
        lane4 slt = sfin & (lane4)(sd < vZero) & (lane4)(sd2 > sq4);
        lane4 sge = sfin & (lane4)(sd > vZero) & (lane4)(sd2 >= sq4);
        vst1q_u32(settledCode + c, vsubq_u32(vaddq_u32(vNine, (uint32x4_t)slt), (uint32x4_t)sge));
    }
    for (; c < traceTotal; c++) {
        float value = a->traceLevel[c], mean = a->traceMean[c], second = a->traceSecond[c], center = a->settledMean[c], spread = a->settledSecond[c];
        mean = keepFinite(mean, mean + momentRate * (value - mean));
        second = keepFinite(second, second + momentRate * (value * value - second));
        center = keepFinite(center, center + settleRate * (value - center));
        spread = keepFinite(spread, spread + settleRate * (value * value - spread));
        a->traceMean[c] = mean;
        a->traceSecond[c] = second;
        a->settledMean[c] = center;
        a->settledSecond[c] = spread;
        float norm = finiteOrZero((value - mean) * oak_rsqrt(fmaxf(second - mean * mean, 0.0f) + ridge));
        uint32_t fc = cellCode(norm, 1u);
        fineCode[observationCount + c] = fc;
        coarseCode[observationCount + c] = (fc <= 2u ? 0u : fc >= 4u ? 2u : 1u) + 8u;
        float sd = value - center, sv = fmaxf(spread - center * center, 0.0f) + ridge;
        uint32_t sc = 9u;
        if (isfinite(sd) && isfinite(sv)) {
            float sd2 = sd * sd, sq4 = 0.25f * sv;
            if (sd < 0.0f && sd2 > sq4) sc = 8u;
            else if (sd > 0.0f && sd2 >= sq4) sc = 10u;
        }
        settledCode[c] = sc;
    }
    uint32_t codedTotal = (1u + OAK_TRACE_LEVELS) * observationCount;
    uint32_t *cells = a->cellActive + a->slot * OAK_CELLS;
    static const uint32_t TUPLE_INIT[24] = {
        1u * 0x9E3779B9u, 2u * 0x9E3779B9u, 3u * 0x9E3779B9u, 4u * 0x9E3779B9u,
        5u * 0x9E3779B9u, 6u * 0x9E3779B9u, 7u * 0x9E3779B9u, 8u * 0x9E3779B9u,
        9u * 0x9E3779B9u, 10u * 0x9E3779B9u, 11u * 0x9E3779B9u, 12u * 0x9E3779B9u,
        13u * 0x9E3779B9u, 14u * 0x9E3779B9u, 15u * 0x9E3779B9u, 16u * 0x9E3779B9u,
        17u * 0x9E3779B9u, 18u * 0x9E3779B9u, 19u * 0x9E3779B9u, 20u * 0x9E3779B9u,
        21u * 0x9E3779B9u, 22u * 0x9E3779B9u, 23u * 0x9E3779B9u, 0u
    };
    uint32x4_t h0 = vld1q_u32(TUPLE_INIT), h1 = vld1q_u32(TUPLE_INIT + 4), h2 = vld1q_u32(TUPLE_INIT + 8);
    uint32x4_t h3 = vld1q_u32(TUPLE_INIT + 12), h4 = vld1q_u32(TUPLE_INIT + 16), h5 = vld1q_u32(TUPLE_INIT + 20);
    o = 0u;
    for (; o + 4u <= observationCount; o += 4u) {
        uint32x4_t co = vld1q_u32(coarseCode + o), fi = vld1q_u32(fineCode + o);
        uint32x4_t cf0 = vzip1q_u32(co, fi), cf1 = vzip2q_u32(co, fi);
        uint32x4_t vcf[4] = {
            vreinterpretq_u32_u64(vdupq_laneq_u64(vreinterpretq_u64_u32(cf0), 0)),
            vreinterpretq_u32_u64(vdupq_laneq_u64(vreinterpretq_u64_u32(cf0), 1)),
            vreinterpretq_u32_u64(vdupq_laneq_u64(vreinterpretq_u64_u32(cf1), 0)),
            vreinterpretq_u32_u64(vdupq_laneq_u64(vreinterpretq_u64_u32(cf1), 1))
        };
        uint32x4_t vcc[4] = { vdupq_laneq_u32(co, 0), vdupq_laneq_u32(co, 1), vdupq_laneq_u32(co, 2), vdupq_laneq_u32(co, 3) };
        for (uint32_t s = 0u; s < 4u; s++) {
            h0 = vmulq_n_u32(veorq_u32(h0, vcf[s]), 0x01000193u);
            h1 = vmulq_n_u32(veorq_u32(h1, vcf[s]), 0x01000193u);
            h2 = vmulq_n_u32(veorq_u32(h2, vcf[s]), 0x01000193u);
            h3 = vmulq_n_u32(veorq_u32(h3, vcf[s]), 0x01000193u);
            h4 = vmulq_n_u32(veorq_u32(h4, vcc[s]), 0x01000193u);
            h5 = vmulq_n_u32(veorq_u32(h5, vcc[s]), 0x01000193u);
        }
    }
    for (; o < observationCount; o++) {
        uint32_t co = coarseCode[o], fi = fineCode[o];
        uint32x4_t vcf = { co, fi, co, fi };
        uint32x4_t vcc = vdupq_n_u32(co);
        h0 = vmulq_n_u32(veorq_u32(h0, vcf), 0x01000193u);
        h1 = vmulq_n_u32(veorq_u32(h1, vcf), 0x01000193u);
        h2 = vmulq_n_u32(veorq_u32(h2, vcf), 0x01000193u);
        h3 = vmulq_n_u32(veorq_u32(h3, vcf), 0x01000193u);
        h4 = vmulq_n_u32(veorq_u32(h4, vcc), 0x01000193u);
        h5 = vmulq_n_u32(veorq_u32(h5, vcc), 0x01000193u);
    }
    uint32_t h00 = vgetq_lane_u32(h0, 0), h01 = vgetq_lane_u32(h0, 1);
    o = 0u;
    for (; o + 4u <= observationCount; o += 4u) {
        uint32_t o1 = observationCount + o, o2 = 2u * observationCount + o, o3 = 3u * observationCount + o;
        uint32_t o4 = 4u * observationCount + o, o5 = 5u * observationCount + o, o6 = 6u * observationCount + o, o7 = 7u * observationCount + o;
        h0 = fnvStep4(h0, vZeroU, vZeroU, vld1q_u32(coarseCode + o1), vaddq_u32(vld1q_u32(fineCode + o1), vEight));
        h1 = fnvStep4(h1, vld1q_u32(coarseCode + o2), vaddq_u32(vld1q_u32(fineCode + o2), vEight), vld1q_u32(coarseCode + o3), vaddq_u32(vld1q_u32(fineCode + o3), vEight));
        h2 = fnvStep4(h2, vld1q_u32(coarseCode + o4), vaddq_u32(vld1q_u32(fineCode + o4), vEight), vld1q_u32(coarseCode + o5), vaddq_u32(vld1q_u32(fineCode + o5), vEight));
        h3 = fnvStep4(h3, vld1q_u32(coarseCode + o6), vaddq_u32(vld1q_u32(fineCode + o6), vEight), vld1q_u32(coarseCode + o7), vaddq_u32(vld1q_u32(fineCode + o7), vEight));
        h4 = fnvStep4(h4, vld1q_u32(settledCode + o), vld1q_u32(settledCode + o1), vld1q_u32(settledCode + o2), vld1q_u32(settledCode + o3));
        h5 = fnvStep4(h5, vld1q_u32(settledCode + o4), vld1q_u32(settledCode + o5), vld1q_u32(settledCode + o6), vZeroU);
    }
    for (; o < observationCount; o++) {
        uint32_t o1 = observationCount + o, o2 = 2u * observationCount + o, o3 = 3u * observationCount + o;
        uint32_t o4 = 4u * observationCount + o, o5 = 5u * observationCount + o, o6 = 6u * observationCount + o, o7 = 7u * observationCount + o;
        uint32x4_t v0 = { 0u, 0u, coarseCode[o1], fineCode[o1] + 8u };
        uint32x4_t v1 = { coarseCode[o2], fineCode[o2] + 8u, coarseCode[o3], fineCode[o3] + 8u };
        uint32x4_t v2 = { coarseCode[o4], fineCode[o4] + 8u, coarseCode[o5], fineCode[o5] + 8u };
        uint32x4_t v3 = { coarseCode[o6], fineCode[o6] + 8u, coarseCode[o7], fineCode[o7] + 8u };
        uint32x4_t v4 = { settledCode[o], settledCode[o1], settledCode[o2], settledCode[o3] };
        uint32x4_t v5 = { settledCode[o4], settledCode[o5], settledCode[o6], 0u };
        h0 = vmulq_n_u32(veorq_u32(h0, v0), 0x01000193u);
        h1 = vmulq_n_u32(veorq_u32(h1, v1), 0x01000193u);
        h2 = vmulq_n_u32(veorq_u32(h2, v2), 0x01000193u);
        h3 = vmulq_n_u32(veorq_u32(h3, v3), 0x01000193u);
        h4 = vmulq_n_u32(veorq_u32(h4, v4), 0x01000193u);
        h5 = vmulq_n_u32(veorq_u32(h5, v5), 0x01000193u);
    }
    h0 = vsetq_lane_u32(h00, h0, 0);
    h0 = vsetq_lane_u32(h01, h0, 1);
    uint32_t mixedOut[24];
    vst1q_u32(mixedOut, mixHash4(h0));
    vst1q_u32(mixedOut + 4, mixHash4(h1));
    vst1q_u32(mixedOut + 8, mixHash4(h2));
    vst1q_u32(mixedOut + 12, mixHash4(h3));
    vst1q_u32(mixedOut + 16, mixHash4(h4));
    vst1q_u32(mixedOut + 20, mixHash4(h5));
    for (uint32_t tuple = 0u; tuple < OAK_TUPLES; tuple++) cells[tuple] = mixedOut[tuple];
    const CellVQ *vq = a->cellVQ;
    const float *cp = headCellBase(a);
    for (uint32_t t = 0u; t < 8u; t++) {
        uint32_t idx = cells[t];
        __builtin_prefetch(vq + idx, 0, 1);
        __builtin_prefetch(cp + idx * OAK_CELL_STRIDE, 0, 1);
    }
    o = 0u;
    uint32x4_t vStep4 = { 0u, 1u, 2u, 3u };
    for (; o + 4u <= observationCount; o += 4u) {
        uint32x4_t idx = vaddq_u32(vdupq_n_u32(OAK_TUPLES + 1u + o), vStep4);
        uint32x4_t fc = vld1q_u32(fineCode + o);
        uint32x4_t h = vmulq_n_u32(veorq_u32(vmulq_n_u32(idx, 0x9E3779B9u), fc), 0x01000193u);
        vst1q_u32(cells + OAK_TUPLES + o, mixHash4(h));
    }
    for (; o < observationCount; o++) {
        uint32_t hash = ((OAK_TUPLES + 1u + o) * 0x9E3779B9u ^ fineCode[o]) * 0x01000193u;
        hash ^= hash >> 15u;
        hash *= 0x2C1B3C6Du;
        hash ^= hash >> 12u;
        cells[OAK_TUPLES + o] = hash >> (32u - OAK_TABLE_BITS);
    }
    if (a->stepCount > 0u) {
        const float *previous = features(a, 1u - a->slot);
        float controlRate = 1.0f / (float)a->stepCount;
        float4 vRate = splat4(controlRate);
        float4 actLow = { a->actionSignal[0], a->actionSignal[1], a->actionSignal[2], a->actionSignal[3] };
        float4 actHigh = { a->actionSignal[4], a->actionSignal[5], 0.0f, 0.0f };
        uint32_t hiAct = a->actionCount > 4u;
        float maxCorr = 0.0f;
        for (o = 0u; o < observationCount; o++) {
            float xo = x[o], po = previous[o];
            float cc = a->chCorr[o];
            cc = keepFinite(cc, fmaf(g.trackingRate, xo * po - cc, cc));
            a->chCorr[o] = cc;
            maxCorr = fmaxf(maxCorr, cc);
            float4 ch = splat4(xo - po);
            float *ctrl = a->control + o * OAK_CELL_STRIDE;
            float4 c0 = oak_load4(ctrl);
            oak_store4(ctrl, oak_keepFinite4(c0, __builtin_elementwise_fma(vRate, __builtin_elementwise_fma(actLow, ch, -c0), c0)));
            if (hiAct) {
                float4 c1 = oak_load4(ctrl + 4u);
                oak_store4(ctrl + 4u, oak_keepFinite4(c1, __builtin_elementwise_fma(vRate, __builtin_elementwise_fma(actHigh, ch, -c1), c1)));
            }
        }
        a->prevCorr = a->obsCorr;
        a->obsCorr = maxCorr >= 1.0f ? 1.0f : maxCorr * maxCorr;
    }
    uint32_t selfBits;
    __builtin_memcpy(&selfBits, obs + a->self, sizeof selfBits);
    uint32x4_t vSelf = vdupq_n_u32(selfBits);
    o = 0u;
    for (; o + 4u <= observationCount; o += 4u) {
        uint32x4_t idx = vaddq_u32(vdupq_n_u32(OAK_TUPLES + OAK_OBSERVATION_LIMIT + 1u + o), vStep4);
        uint32x4_t wb;
        __builtin_memcpy(&wb, obs + o, sizeof wb);
        uint32x4_t h = vmulq_n_u32(veorq_u32(vmulq_n_u32(idx, 0x9E3779B9u), vSelf), 0x01000193u);
        h = vmulq_n_u32(veorq_u32(h, wb), 0x01000193u);
        vst1q_u32(cells + OAK_TUPLES + observationCount + o, mixHash4(h));
    }
    for (; o < observationCount; o++) {
        uint32_t worldBits;
        __builtin_memcpy(&worldBits, obs + o, sizeof worldBits);
        uint32_t hash = ((((OAK_TUPLES + OAK_OBSERVATION_LIMIT + 1u + o) * 0x9E3779B9u ^ selfBits) * 0x01000193u) ^ worldBits) * 0x01000193u;
        hash ^= hash >> 15u;
        hash *= 0x2C1B3C6Du;
        hash ^= hash >> 12u;
        cells[OAK_TUPLES + observationCount + o] = hash >> (32u - OAK_TABLE_BITS);
    }
    if (a->box) return;
    uint32_t exact = 0x9E3779B9u;
    o = 0u;
    for (; o + 4u <= observationCount; o += 4u) {
        uint32x4_t b;
        __builtin_memcpy(&b, obs + o, sizeof b);
        b = veorq_u32(b, vshrq_n_u32(b, 16));
        b = vmulq_n_u32(b, 0x85EBCA6Bu);
        b = veorq_u32(b, vshrq_n_u32(b, 13));
        exact = (exact ^ vgetq_lane_u32(b, 0)) * 0x01000193u;
        exact = (exact ^ vgetq_lane_u32(b, 1)) * 0x01000193u;
        exact = (exact ^ vgetq_lane_u32(b, 2)) * 0x01000193u;
        exact = (exact ^ vgetq_lane_u32(b, 3)) * 0x01000193u;
    }
    for (; o < observationCount; o++) {
        uint32_t bits;
        __builtin_memcpy(&bits, obs + o, sizeof bits);
        bits ^= bits >> 16u;
        bits *= 0x85EBCA6Bu;
        bits ^= bits >> 13u;
        exact = (exact ^ bits) * 0x01000193u;
    }
    uint32_t history = exact;
    for (uint32_t k = observationCount; k < codedTotal; k++) history = (history ^ fineCode[k]) * 0x01000193u;
    uint32_t exactCell = tableCell(exact, OAK_EXACT_BITS), histCell = tableCell(history, OAK_HIST_BITS);
    a->qCell[a->slot * OAK_KEYS] = exactCell;
    a->qCell[a->slot * OAK_KEYS + 1u] = histCell;
    a->qTag[a->slot * OAK_KEYS] = exact;
    a->qTag[a->slot * OAK_KEYS + 1u] = history;
    __builtin_prefetch(&a->exactTable[exactCell], 0, 1);
    __builtin_prefetch(&a->histTable[histCell], 0, 1);
}

static __attribute__((noinline)) void perceive(OakAgent *a, const float *obs) {
    OakGeometry g = a->g;
    uint32_t observationCount = a->observationCount;
    float standardized[OAK_OBSERVATION_LIMIT + 1u];
    uint32_t metaAge = a->stepCount < OAK_FEATURE_TIMESCALE ? a->stepCount : OAK_FEATURE_TIMESCALE;
    float inverseAge = 1.0f / ((float)metaAge + 1.0f);
    float floorRate = 1.0f / ((float)a->stepCount + 1.0f);
    float *x = features(a, a->slot);
    float alphaFloor = g.trackingRate * g.trackingRate;
    float4 vInvAge = splat4(inverseAge), vFloorRate = splat4(floorRate), vMetaRate = splat4(g.metaRate);
    float4 vOneMinusDecay = splat4(1.0f - g.trackingRate), vAlphaFloor = splat4(alphaFloor), vRidgeFloor = splat4(g.ridgeFloor);
    float4 vZero = splat4(0.0f), vOne = splat4(1.0f), vHalf = splat4(0.5f);
    uint32_t o = 0u;
    for (; o + 4u <= observationCount; o += 4u) {
        float4 vx = oak_load4(obs + o);
        lane4 fin = oak_finite4(vx);
        float4 mean = oak_load4(a->chMean + o), second = oak_load4(a->chSecond + o);
        float4 alpha = oak_load4(a->chAlpha + o);
        float4 trace = oak_load4(a->chTrace + o), energy = oak_load4(a->chEnergy + o);
        float4 sig = vx - mean;
        float4 mg = sig * trace;
        float4 ge = oak_keepFinite4(energy, __builtin_elementwise_fma(vInvAge, __builtin_elementwise_fma(mg, mg, -energy), energy));
        float4 db = vMetaRate * mg * oak_rsqrt4_fast(__builtin_elementwise_max(ge, vZero) + vInvAge);
        float4 fac = __builtin_elementwise_fma(db, __builtin_elementwise_fma(db, vHalf, vOne), vOne);
        float4 na = oak_keepFinite4(alpha, __builtin_elementwise_min(__builtin_elementwise_max(alpha * fac, vAlphaFloor), vOne));
        float4 nh = oak_keepFinite4(trace, __builtin_elementwise_fma(vOneMinusDecay * __builtin_elementwise_max(vZero, vOne - alpha), trace, alpha * sig));
        lane4 updMeta = fin & oak_finite4(sig);
        alpha = select4(updMeta, na, alpha);
        trace = select4(updMeta, nh, trace);
        energy = select4(updMeta, ge, energy);
        float4 rate = __builtin_elementwise_max(vFloorRate, alpha);
        mean = select4(fin, oak_keepFinite4(mean, __builtin_elementwise_fma(rate, vx - mean, mean)), mean);
        second = select4(fin, oak_keepFinite4(second, __builtin_elementwise_fma(rate, __builtin_elementwise_fma(vx, vx, -second), second)), second);
        oak_store4(a->chMean + o, mean);
        oak_store4(a->chSecond + o, second);
        oak_store4(a->chAlpha + o, alpha);
        oak_store4(a->chTrace + o, trace);
        oak_store4(a->chEnergy + o, energy);
        float4 stdv = (vx - mean) * oak_rsqrt4(__builtin_elementwise_max(__builtin_elementwise_fma(-mean, mean, second), vZero) + vRidgeFloor);
        oak_store4(standardized + o, stdv);
        oak_store4(x + o, oak_finiteOrZero4(stdv));
    }
    for (; o < observationCount; o++) {
        Channel c = { a->chMean[o], a->chSecond[o], a->chAlpha[o], a->chBeta[o], a->chTrace[o], a->chEnergy[o] };
        standardized[o] = standardizeChannel(&c, obs[o], inverseAge, floorRate, g.metaRate, g.trackingRate, a->betaFloor, g.ridgeFloor);
        a->chMean[o] = c.mean;
        a->chSecond[o] = c.second;
        a->chAlpha[o] = c.alpha;
        a->chBeta[o] = c.beta;
        a->chTrace[o] = c.trace;
        a->chEnergy[o] = c.energy;
        x[o] = finiteOrZero(standardized[o]);
    }
    standardized[observationCount] = standardizeChannel(&a->rewardChannel, a->reward, inverseAge, floorRate, g.metaRate, g.trackingRate, a->betaFloor, g.ridgeFloor);
    for (uint32_t d = 0u; d < OAK_ACTION_PARAMS; d++) x[observationCount + d] = finiteOrZero(a->actionSignal[d]);
    x[observationCount + OAK_ACTION_PARAMS] = finiteOrZero(standardized[observationCount]);
    x[observationCount + OAK_ACTION_PARAMS + 1u] = 1.0f;
    float *traces = x + g.projected;
    o = 0u;
    for (; o + 4u <= observationCount; o += 4u) {
        float4 st = oak_load4(standardized + o);
        lane4 fin = oak_finite4(oak_load4(obs + o));
        for (uint32_t m = 0u; m < OAK_TRACE_LEVELS; m++) {
            float4 vRate = splat4(1.0f / (float)(2u << m));
            uint32_t idx = m * observationCount + o;
            float4 lv = oak_load4(a->traceLevel + idx);
            float4 next = __builtin_elementwise_fma(vRate, st - lv, lv);
            float4 updated = select4(fin, next, lv);
            oak_store4(a->traceLevel + idx, updated);
            oak_store4(traces + idx, oak_finiteOrZero4(updated));
        }
    }
    for (; o < observationCount; o++) {
        float st = standardized[o];
        int fin = isfinite(obs[o]);
        for (uint32_t m = 0u; m < OAK_TRACE_LEVELS; m++) {
            float rate = 1.0f / (float)(2u << m);
            uint32_t idx = m * observationCount + o;
            float lv = a->traceLevel[idx];
            float next = lv + rate * (st - lv);
            a->traceLevel[idx] = fin ? next : lv;
            traces[idx] = finiteOrZero(a->traceLevel[idx]);
        }
    }
    encode(a, x, obs);
    uint32_t inputs = g.inputs, sources = a->sources, permStride = a->permStride;
    float *px = permInputs(a, a->slot);
    for (uint32_t p = 0u; p < OAK_PERMS; p++) {
        float *dst = px + p * permStride;
        const uint16_t *perm = a->inputPerm + p * sources;
        uint32_t k = 0u;
        for (; k + 4u <= sources; k += 4u) {
            float v0 = x[perm[k]], v1 = x[perm[k + 1u]], v2 = x[perm[k + 2u]], v3 = x[perm[k + 3u]];
            dst[k] = v0;
            dst[k + 1u] = v1;
            dst[k + 2u] = v2;
            dst[k + 3u] = v3;
        }
        for (; k < sources; k++) dst[k] = x[perm[k]];
        dst[sources] = dst[0];
        dst[sources + 1u] = dst[1];
        dst[sources + 2u] = dst[2];
        dst[sources + 3u] = dst[3];
    }
    float *unit = x + inputs, *energies = a->unitEnergy + a->slot * OAK_UNITS;
    uint32_t half = OAK_UNITS / 2u;
    float *gates = a->unitGate + a->slot * half;
    float4 vGate = splat4(a->obsCorr * a->obsCorr);
    for (uint32_t j = 0u; j < OAK_UNITS; j += 4u) {
        float4 ss[4], ee[4];
        for (uint32_t u = 0u; u < 4u; u++) {
            const float *weight = a->projection + (j + u) * OAK_FANIN;
            const uint16_t *blk = a->unitBlock + (j + u) * OAK_FANIN_BLOCKS;
            float4 s0 = splat4(0.0f), s1 = s0, s2 = s0, s3 = s0;
            float4 e0 = s0, e1 = s0, e2 = s0, e3 = s0;
            for (uint32_t b = 0u; b < OAK_FANIN_BLOCKS; b += 4u) {
                float4 i0 = oak_load4(px + blk[b]), i1 = oak_load4(px + blk[b + 1u]);
                float4 i2 = oak_load4(px + blk[b + 2u]), i3 = oak_load4(px + blk[b + 3u]);
                s0 = __builtin_elementwise_fma(oak_load4(weight + 4u * b), i0, s0);
                s1 = __builtin_elementwise_fma(oak_load4(weight + 4u * b + 4u), i1, s1);
                s2 = __builtin_elementwise_fma(oak_load4(weight + 4u * b + 8u), i2, s2);
                s3 = __builtin_elementwise_fma(oak_load4(weight + 4u * b + 12u), i3, s3);
                e0 = __builtin_elementwise_fma(i0, i0, e0);
                e1 = __builtin_elementwise_fma(i1, i1, e1);
                e2 = __builtin_elementwise_fma(i2, i2, e2);
                e3 = __builtin_elementwise_fma(i3, i3, e3);
            }
            ss[u] = (s0 + s1) + (s2 + s3);
            ee[u] = (e0 + e1) + (e2 + e3);
        }
        float4 sv = oak_hsum4(ss[0], ss[1], ss[2], ss[3]);
        float4 ev = oak_hsum4(ee[0], ee[1], ee[2], ee[3]);
        float4 tv = oak_tanh4(sv);
        if (j < half) {
            oak_store4(unit + j, tv);
        } else {
            uint32_t k = j - half;
            oak_store4(gates + k, tv);
            float4 m = __builtin_elementwise_fma(vGate, oak_load4(unit + k) - vOne, vOne);
            oak_store4(unit + j, tv * m);
        }
        oak_store4(energies + j, ev);
    }
}

static __attribute__((noinline)) void evaluate(OakAgent *a) {
    OakGeometry g = a->g;
    uint32_t count = g.features, actions = a->actionCount, head = a->learner;
    const float *current = features(a, a->slot), *previous = features(a, 1u - a->slot);
    float currentValue, previousValue;
    dotProduct2(a->valueWeight, current, previous, count, &currentValue, &previousValue);
    const uint32_t *currentCells = a->cellActive + a->slot * OAK_CELLS, *previousCells = a->cellActive + (1u - a->slot) * OAK_CELLS;
    uint32_t cellCount = OAK_TUPLES + 2u * a->observationCount, previousTuples = a->stepCount > 0u ? cellCount : 0u;
    CellVQ *vq = a->cellVQ;
    const float *cp = headCellBase(a);
    float4 cellLow = splat4(0.0f), cellHigh = cellLow, qLow = cellLow, qHigh = cellLow;
    if (!a->box) {
        if (actions <= 4u) {
            for (uint32_t t = 0u; t < cellCount; t++) {
                if (t + 8u < cellCount) {
                    uint32_t pIdx = currentCells[t + 8u];
                    __builtin_prefetch(vq + pIdx, 0, 1);
                    __builtin_prefetch(cp + pIdx * OAK_CELL_STRIDE, 0, 1);
                }
                uint32_t idx = currentCells[t];
                const CellVQ *s = vq + idx;
                currentValue += s->value;
                cellLow += oak_load4(cp + idx * OAK_CELL_STRIDE);
                qLow += oak_load4(s->q);
            }
        } else {
            for (uint32_t t = 0u; t < cellCount; t++) {
                if (t + 8u < cellCount) {
                    uint32_t pIdx = currentCells[t + 8u];
                    __builtin_prefetch(vq + pIdx, 0, 1);
                    __builtin_prefetch(cp + pIdx * OAK_CELL_STRIDE, 0, 1);
                }
                uint32_t idx = currentCells[t];
                const CellVQ *s = vq + idx;
                const float *p = cp + idx * OAK_CELL_STRIDE;
                currentValue += s->value;
                cellLow += oak_load4(p);
                cellHigh += oak_load4(p + 4u);
                qLow += oak_load4(s->q);
                qHigh += oak_load4(s->q + 4u);
            }
        }
    } else {
        for (uint32_t t = 0u; t < OAK_TUPLES; t++) {
            if (t + 8u < OAK_TUPLES) {
                uint32_t pIdx = currentCells[t + 8u];
                __builtin_prefetch(vq + pIdx, 0, 1);
                __builtin_prefetch(cp + pIdx * OAK_CELL_STRIDE, 0, 1);
            }
            uint32_t idx = currentCells[t];
            const float *p = cp + idx * OAK_CELL_STRIDE;
            currentValue += vq[idx].value;
            cellLow += oak_load4(p);
            if (actions > 4u) cellHigh += oak_load4(p + 4u);
        }
        for (uint32_t t = OAK_TUPLES; t < cellCount; t++) {
            if (t + 8u < cellCount) __builtin_prefetch(vq + currentCells[t + 8u], 0, 1);
            currentValue += vq[currentCells[t]].value;
        }
    }
    for (uint32_t t = 0u; t < 8u && t < previousTuples; t++) __builtin_prefetch(vq + previousCells[t], 0, 1);
    for (uint32_t t = 0u; t < previousTuples; t++) {
        if (t + 8u < previousTuples) __builtin_prefetch(vq + previousCells[t + 8u], 0, 1);
        previousValue += vq[previousCells[t]].value;
    }
    float cellLogit[8] = { cellLow.x, cellLow.y, cellLow.z, cellLow.w, cellHigh.x, cellHigh.y, 0.0f, 0.0f };
    const float *policy = headPolicy(a);
    uint32_t d = 0u;
    if (actions >= 4u) {
        float v0, v1, v2, v3;
        dotRows4(policy, policy + count, policy + 2u * count, policy + 3u * count, current, count, &v0, &v1, &v2, &v3);
        a->logit[0] = v0 + cellLogit[0];
        a->logit[1] = v1 + cellLogit[1];
        a->logit[2] = v2 + cellLogit[2];
        a->logit[3] = v3 + cellLogit[3];
        d = 4u;
    }
    for (; d + 2u <= actions; d += 2u) {
        float v0, v1;
        dotRows2(policy + d * count, policy + (d + 1u) * count, current, count, &v0, &v1);
        a->logit[d] = v0 + cellLogit[d];
        a->logit[d + 1u] = v1 + cellLogit[d + 1u];
    }
    for (; d < actions; d++) a->logit[d] = dotProduct(policy + d * count, current, count) + cellLogit[d];
    float averageRewardPrev = a->averageReward;
    float delta = finiteOrZero(a->reward - averageRewardPrev + currentValue - previousValue);
    a->td = delta;
    float averageRewardNew = averageRewardPrev + g.trackingRate * delta;
    a->averageReward = isfinite(averageRewardNew) ? averageRewardNew : averageRewardPrev;
    float squareRate = trackingRateFor(a->stepCount, OAK_FEATURE_TIMESCALE);
    a->tdSquare = keepFinite(a->tdSquare, a->tdSquare + squareRate * (delta * delta - a->tdSquare));
    float tdScale = oak_rsqrt(a->tdSquare);
    a->tdInv = tdScale;
    a->tdNormalized = finiteOrZero(delta * tdScale);
    float advantageCenterPrev = a->advantageCenter;
    float advantageError = delta - advantageCenterPrev;
    float rewardRate = trackingRateFor(a->stepCount, OAK_FEATURE_TIMESCALE);
    float rewardError = a->reward - averageRewardPrev;
    a->rewardSquare = keepFinite(a->rewardSquare, a->rewardSquare + rewardRate * (rewardError * rewardError - a->rewardSquare));
    a->rewardMagnitude = keepFinite(a->rewardMagnitude, a->rewardMagnitude + rewardRate * (fabsf(rewardError) - a->rewardMagnitude));
    float rewardGap = a->reward - a->rewardMean;
    a->rewardMean = keepFinite(a->rewardMean, a->rewardMean + rewardRate * rewardGap);
    a->rewardVariance = keepFinite(a->rewardVariance, a->rewardVariance + rewardRate * (rewardGap * rewardGap - a->rewardVariance));
    float assured = a->rewardMean - sqrtf(fmaxf(a->rewardVariance, 0.0f) * rewardRate);
    float aspirationPrev = a->stepCount == 0u ? assured : a->aspiration;
    a->aspiration = keepFinite(aspirationPrev, fmaxf(assured, aspirationPrev + g.metaRate * (a->rewardMean - aspirationPrev)));
    float shortfall = finiteOrZero(fmaxf(a->aspiration - a->rewardMean, 0.0f));
    float informed = a->rewardSquare / a->tdSquare;
    float trust = informed >= 1.0f ? 1.0f : sqrtf(informed);
    float scale = HEAD_SCALE[head] == SCALE_TD ? tdScale : HEAD_SCALE[head] == SCALE_RAW ? 1.0f : a->rewardMagnitude / a->rewardSquare;
    a->advantage = finiteOrZero((advantageError - (HEAD_ASPIRE[head] ? shortfall : 0.0f)) * scale * (HEAD_TRUST[head] ? trust : 1.0f));
    a->advantageCenter = keepFinite(advantageCenterPrev, advantageCenterPrev + g.trackingRate * advantageError);
    uint32_t metaAge = a->stepCount < OAK_FEATURE_TIMESCALE ? a->stepCount : OAK_FEATURE_TIMESCALE;
    Channel *meta = a->actorMeta + head;
    if (!a->box) metaStep(meta, a->advantage, 1.0f / ((float)metaAge + 1.0f), g.metaRate, g.trackingRate, a->betaFloor);
    a->actorStep = finiteOrZero(meta->alpha * a->advantage / OAK_ACTOR_ALPHA_DENOM);
    oak_store4(a->actorCredit, oak_load4(a->actorCreditStaging));
    a->actorCredit[4] = a->actorCreditStaging[4];
    a->actorCredit[5] = a->actorCreditStaging[5];
    a->qCellStep = 0.0f;
    if (a->box) return;
    oak_store4(a->qCellValue, qLow);
    a->qCellValue[4] = qHigh.x;
    a->qCellValue[5] = qHigh.y;
    float qBest = a->qCellValue[0];
    for (uint32_t aidx = 1u; aidx < actions; aidx++) qBest = fmaxf(qBest, a->qCellValue[aidx]);
    if (a->stepCount == 0u) return;
    float qDelta = finiteOrZero(a->reward - a->qCellGain + qBest - a->qCellTaken);
    a->qCellGain = keepFinite(a->qCellGain, a->qCellGain + g.trackingRate * qDelta);
    a->qCellNoise = keepFinite(a->qCellNoise, a->qCellNoise + g.trackingRate * (qDelta * qDelta - a->qCellNoise));
    a->qCellStep = finiteOrZero(g.ridgeFloor * qDelta / (1.0f + g.ridgeFloor * (float)cellCount));
    a->qCellAct = a->action;
}

static inline uint32_t cellQAction(const OakAgent *a, const float *policy, float *behaviour) {
    uint32_t actions = a->actionCount, first = 0u;
    for (uint32_t d = 1u; d < actions; d++)
        if (a->qCellValue[d] > a->qCellValue[first]) first = d;
    float gap = a->qCellValue[first], spread = 0.0f;
    for (uint32_t d = 0u; d < actions; d++) {
        float weight = (d == first ? 1.0f : 0.0f) - policy[d];
        gap -= policy[d] * a->qCellValue[d];
        spread += weight * weight;
    }
    if (gap * gap <= a->qCellNoise * spread) return OAK_POLICY_MAX;
    *behaviour = 1.0f;
    return first;
}

static inline uint32_t untriedActions(const TableRowHead *row, uint32_t actions) {
    uint32_t untried = 0u;
    for (uint32_t d = 0u; d < actions; d++) untried |= (row->visits[d] == 0u ? 1u : 0u) << d;
    return untried;
}

static inline uint32_t knownUntried(const TableRowHead *row, uint32_t tag, uint32_t actions) {
    return row->owner == tag ? untriedActions(row, actions) : (1u << actions) - 1u;
}

static inline float exactRowValue(const TableRowHead *row, uint8_t t8, uint32_t actions) {
    if (row->owner == 0u || tag8(row->owner) != t8) return 0.0f;
    float best = -INFINITY;
    for (uint32_t d = 0u; d < actions; d++)
        if (row->visits[d] != 0u) best = fmaxf(best, row->q[d]);
    return best == -INFINITY ? 0.0f : best;
}

static inline void psPush(OakAgent *a, uint32_t cell, uint32_t tag, float prio) {
    if (!(prio > 0.0f) || !isfinite(prio)) return;
    uint32_t count = a->psCount;
    uint32_t minIdx = 0u;
    float minVal = INFINITY;
    for (uint32_t i = 0u; i < count; i++) {
        if (a->psCell[i] == cell && a->psTag[i] == tag) {
            if (prio > a->psPrio[i]) a->psPrio[i] = prio;
            return;
        }
        if (a->psPrio[i] < minVal) {
            minVal = a->psPrio[i];
            minIdx = i;
        }
    }
    if (count < OAK_DYNA_QUEUE) {
        a->psCell[count] = (uint16_t)cell;
        a->psTag[count] = tag;
        a->psPrio[count] = prio;
        a->psCount = count + 1u;
    } else if (prio > minVal) {
        a->psCell[minIdx] = (uint16_t)cell;
        a->psTag[minIdx] = tag;
        a->psPrio[minIdx] = prio;
    }
}

static inline void recordPredecessor(ExactRow *dst, uint32_t prevCell, uint32_t act) {
    uint32_t meta = dst->predMeta;
    uint32_t count = (meta >> 12u) & 7u;
    for (uint32_t i = 0u; i < count; i++)
        if (dst->predCell[i] == prevCell && ((meta >> (3u * i)) & 7u) == act) return;
    if (count < OAK_DYNA_PREDS) {
        dst->predCell[count] = (uint16_t)prevCell;
        dst->predMeta = (uint16_t)(((count + 1u) << 12u) | (meta & 0x0fffu) | (act << (3u * count)));
    } else {
        dst->predCell[0] = dst->predCell[1];
        dst->predCell[1] = dst->predCell[2];
        dst->predCell[2] = dst->predCell[3];
        dst->predCell[3] = (uint16_t)prevCell;
        dst->predMeta = (uint16_t)((OAK_DYNA_PREDS << 12u) | ((meta & 0x0fffu) >> 3u) | (act << 9u));
    }
}

static void claimRow(OakAgent *a, uint32_t key, uint32_t cell, TableRowHead *row, uint32_t tag) {
    uint32_t actions = a->actionCount, untried = untriedActions(row, actions);
    uint32_t hadOwner = row->owner != 0u;
    if (untried != (1u << actions) - 1u) a->qOpen[key] -= (uint32_t)__builtin_popcount(untried);
    for (uint32_t d = 0u; d < actions; d++) {
        if (((untried >> d) & 1u) == 0u) {
            a->qTried[key] -= 1u;
            a->qSum[key] -= row->q[d];
        }
        row->q[d] = 0.0f;
        row->visits[d] = 0u;
        if (key == 0u) {
            ExactRow *ex = &a->exactTable[cell];
            ex->modelProb[d][0] = 0u;
            ex->modelProb[d][1] = 0u;
            ex->modelProb[d][2] = 0u;
        }
    }
    if (key == 0u && hadOwner) a->exactTable[cell].predMeta = 0u;
    row->owner = tag;
}

static __attribute__((noinline)) void dynaSweep(OakAgent *a) {
    uint32_t actions = a->actionCount;
    uint32_t minCov = 255u - (255u / OAK_DYNA_QUEUE);
    float threshSq = a->qNoise[0] * a->g.ridgeFloor;
    for (uint32_t iter = 0u; iter < OAK_DYNA_STEPS && a->psCount > 0u; iter++) {
        uint32_t count = a->psCount, bestIdx = 0u;
        float bestPrio = a->psPrio[0];
        for (uint32_t i = 1u; i < count; i++)
            if (a->psPrio[i] > bestPrio) {
                bestPrio = a->psPrio[i];
                bestIdx = i;
            }
        uint32_t cell = a->psCell[bestIdx], tag = a->psTag[bestIdx];
        count -= 1u;
        a->psCount = count;
        if (bestIdx < count) {
            a->psCell[bestIdx] = a->psCell[count];
            a->psTag[bestIdx] = a->psTag[count];
            a->psPrio[bestIdx] = a->psPrio[count];
        }
        const ExactRow *target = &a->exactTable[cell];
        if (target->h.owner != tag) continue;
        uint8_t targetT8 = tag8(tag);
        uint32_t meta = target->predMeta;
        uint32_t pCount = (meta >> 12u) & 7u;
        for (uint32_t i = 0u; i < pCount; i++) {
            uint32_t pCell = target->predCell[i], pAct = (meta >> (3u * i)) & 7u;
            ExactRow *pex = &a->exactTable[pCell];
            uint32_t pTag = pex->h.owner;
            if (pTag == 0u || pex->h.visits[pAct] == 0u) continue;
            uint32_t p0 = pex->modelProb[pAct][0], p1 = pex->modelProb[pAct][1], pSelf = pex->modelProb[pAct][2];
            uint32_t sumP = p0 + p1 + pSelf;
            if (sumP < minCov) continue;
            uint32_t matched = 0u, valid = 1u;
            float boot = 0.0f;
            for (uint32_t b = 0u; b < 2u; b++) {
                uint32_t pb = pex->modelProb[pAct][b];
                if (pb == 0u) continue;
                uint32_t nCell = pex->modelNext[pAct][b];
                uint8_t nTag = pex->modelTag[pAct][b];
                if (nCell == cell && nTag == targetT8) matched = 1u;
                const TableRowHead *nrow = &a->exactTable[nCell].h;
                if (nrow->owner != 0u && tag8(nrow->owner) != nTag) {
                    valid = 0u;
                    break;
                }
                boot += (float)pb * exactRowValue(nrow, nTag, actions);
            }
            if (!matched || !valid) continue;
            uint8_t pT8 = tag8(pTag);
            float prevMax = exactRowValue(&pex->h, pT8, actions);
            if (pSelf > 0u) boot += (float)pSelf * prevMax;
            boot /= (float)sumP;
            float gain = a->qTried[0] ? a->qSum[0] / (float)a->qTried[0] : 0.0f;
            float delta = finiteOrZero(pex->modelReward[pAct] - gain + boot - pex->h.q[pAct]);
            float val = pex->h.q[pAct] + 0.5f * delta;
            if (isfinite(val)) {
                a->qSum[0] += val - pex->h.q[pAct];
                pex->h.q[pAct] = val;
            }
            float nextMax = exactRowValue(&pex->h, pT8, actions);
            float diff = fabsf(nextMax - prevMax);
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
    for (uint32_t d = 0u; d < actions; d++)
        if (((successorUntried >> d) & 1u) == 0u) bootstrap = fmaxf(bootstrap, successor->q[d]);
    bootstrap = bootstrap == -INFINITY ? 0.0f : bootstrap;
    float gain = a->qSum[key] / (float)a->qTried[key];
    float delta = finiteOrZero(a->reward - gain + bootstrap - row->q[act]);
    a->qNoise[key] = keepFinite(a->qNoise[key], a->qNoise[key] + a->g.trackingRate * (delta * delta - a->qNoise[key]));
    uint32_t trackModel = key == 0u && a->qTried[0] < OAK_EXACT;
    uint32_t visitsInt = row->visits[act] < 65535u ? (uint32_t)row->visits[act] + 1u : 65535u;
    row->visits[act] = (uint16_t)visitsInt;
    float visits = (float)visitsInt;
    float step = fmaxf(1.0f / visits, a->g.trackingRate), value = row->q[act] + step * delta;
    if (isfinite(value)) {
        a->qSum[key] += value - row->q[act];
        row->q[act] = value;
    }
    if (key == 1u) {
        uint32_t hop = successorUntried ? 1u : 255u;
        const HistRow *succHist = &a->histTable[next];
        for (uint32_t d = 0u; d < actions && hop > 1u && a->qOpen[1]; d++) hop = succHist->distance[d] + 1u < hop ? succHist->distance[d] + 1u : hop;
        a->histTable[previous].distance[act] = (uint8_t)hop;
        return;
    }
    ExactRow *ex = &a->exactTable[previous];
    float modelRate = fmaxf(1.0f / visits, 1.0f / (float)OAK_DYNA_QUEUE);
    float reward = finiteOrZero(a->reward);
    ex->modelReward[act] = firstTry ? reward : ex->modelReward[act] + step * (reward - ex->modelReward[act]);
    uint32_t newBranch = 0u;
    if (firstTry) {
        if (next == previous) {
            ex->modelProb[act][0] = 0u;
            ex->modelProb[act][1] = 0u;
            ex->modelProb[act][2] = 255u;
        } else {
            ex->modelNext[act][0] = (uint16_t)next;
            ex->modelTag[act][0] = tag8(nextTag);
            ex->modelProb[act][0] = 255u;
            ex->modelProb[act][1] = 0u;
            ex->modelProb[act][2] = 0u;
            newBranch = 1u;
        }
    } else {
        float decay = 1.0f - modelRate;
        float p0 = (float)ex->modelProb[act][0] * (1.0f / 255.0f) * decay;
        float p1 = (float)ex->modelProb[act][1] * (1.0f / 255.0f) * decay;
        float pSelf = (float)ex->modelProb[act][2] * (1.0f / 255.0f) * decay;
        if (next == previous) {
            if (ex->modelProb[act][2] == 0u) newBranch = 1u;
            pSelf += modelRate;
        } else {
            uint8_t nt8 = tag8(nextTag);
            if (ex->modelProb[act][0] > 0u && ex->modelNext[act][0] == next && ex->modelTag[act][0] == nt8) {
                p0 += modelRate;
            } else if (ex->modelProb[act][1] > 0u && ex->modelNext[act][1] == next && ex->modelTag[act][1] == nt8) {
                p1 += modelRate;
            } else if (p0 <= p1) {
                ex->modelNext[act][0] = (uint16_t)next;
                ex->modelTag[act][0] = nt8;
                p0 = modelRate;
                newBranch = 1u;
            } else {
                ex->modelNext[act][1] = (uint16_t)next;
                ex->modelTag[act][1] = nt8;
                p1 = modelRate;
                newBranch = 1u;
            }
            if (p1 > p0) {
                uint16_t tn = ex->modelNext[act][0];
                uint8_t tt = ex->modelTag[act][0];
                float tp = p0;
                ex->modelNext[act][0] = ex->modelNext[act][1];
                ex->modelTag[act][0] = ex->modelTag[act][1];
                p0 = p1;
                ex->modelNext[act][1] = tn;
                ex->modelTag[act][1] = tt;
                p1 = tp;
            }
        }
        ex->modelProb[act][0] = (uint8_t)(p0 * 255.0f + 0.5f);
        ex->modelProb[act][1] = (uint8_t)(p1 * 255.0f + 0.5f);
        ex->modelProb[act][2] = (uint8_t)(pSelf * 255.0f + 0.5f);
    }
    if (trackModel) {
        if (previous != next && (newBranch || (a->exactTable[next].predMeta >> 12u) == 0u) && (successor->owner == 0u || successor->owner == nextTag))
            recordPredecessor(&a->exactTable[next], previous, act);
        uint32_t shift = newBranch && !firstTry;
        float move = fabsf(step * delta);
        if (move * move > a->qNoise[0] * a->g.ridgeFloor || shift) {
            float prio = move + (shift ? sqrtf(fmaxf(a->qNoise[0], 0.0f)) : 0.0f);
            if ((ex->predMeta >> 12u) != 0u) psPush(a, previous, previousTag, prio);
            if (shift && previous != next && (a->exactTable[next].predMeta >> 12u) != 0u) psPush(a, next, nextTag, prio);
        }
        if (a->psCount > 0u) dynaSweep(a);
    }
}

static inline uint32_t tableAction(const OakAgent *a, uint32_t arm, const uint32_t *words, const float *policy, float *behaviour) {
    uint32_t actions = a->actionCount, key = TABLE_KEY[arm], cell = a->qCell[a->slot * OAK_KEYS + key];
    const TableRowHead *row = tableRowConst(a, key, cell);
    uint32_t unknown = knownUntried(row, a->qTag[a->slot * OAK_KEYS + key], actions);
    if (TABLE_MODE[arm] == TABLE_GREEDY) {
        if (unknown) return OAK_POLICY_MAX;
        uint32_t first = 0u;
        for (uint32_t d = 1u; d < actions; d++)
            if (row->q[d] > row->q[first]) first = d;
        float horizon = (float)OAK_FEATURE_TIMESCALE, gap = row->q[first], spread = 0.0f;
        for (uint32_t d = 0u; d < actions; d++) {
            float weight = (d == first ? 1.0f : 0.0f) - policy[d];
            gap -= policy[d] * row->q[d];
            spread += weight * weight / fminf((float)row->visits[d], horizon);
        }
        if (gap * gap <= a->qNoise[key] * spread) return OAK_POLICY_MAX;
        *behaviour = 1.0f;
        return first;
    }
    if (unknown == (1u << actions) - 1u) return OAK_POLICY_MAX;
    uint32_t nearest = 255u, frontier = 0u;
    const HistRow *hrow = &a->histTable[cell];
    for (uint32_t d = 0u; d < actions && a->qOpen[key]; d++) nearest = hrow->distance[d] < nearest ? hrow->distance[d] : nearest;
    for (uint32_t d = 0u; d < actions && TABLE_MODE[arm] == TABLE_FRONTIER; d++) frontier |= (nearest < 255u && hrow->distance[d] == nearest ? 1u : 0u) << d;
    uint32_t pool = unknown ? unknown : frontier ? frontier : (1u << actions) - 1u;
    *behaviour = unknown ? 1.0f / (float)__builtin_popcount(unknown) : 1.0f;
    uint32_t chosen = 0u, chosenWord = 0u;
    float best = -INFINITY;
    for (uint32_t d = 0u; d < actions; d++) {
        if (((pool >> d) & 1u) == 0u) continue;
        float value = unknown ? 0.0f : row->q[d];
        if (value > best || (value == best && words[d] > chosenWord)) {
            best = value;
            chosenWord = words[d];
            chosen = d;
        }
    }
    return chosen;
}

static __attribute__((noinline)) uint32_t planAction(OakAgent *a, const uint32_t *words, const float *policy) {
    uint32_t actions = a->actionCount, start = a->qCell[a->slot * OAK_KEYS];
    uint32_t startTag = a->qTag[a->slot * OAK_KEYS];
    uint32_t startUntried = knownUntried(&a->exactTable[start].h, startTag, actions);
    if (a->planIndex < a->planLength && a->planCells[a->planIndex] == start && startUntried != (1u << actions) - 1u) return a->planActions[a->planIndex++];
    a->planLength = 0u;
    a->planIndex = 0u;
    if (startUntried == (1u << actions) - 1u) return OAK_POLICY_MAX;
    uint32_t target = OAK_EXACT, targetAction = 0u;
    if (a->qOpen[0] > 0u) {
        uint32_t generation = ++a->planGeneration, head = 0u, tail = 1u;
        uint32_t minBranchProb = 255u / (1u + OAK_TRACE_LEVELS);
        a->planQueue[0] = (uint16_t)start;
        a->planStamp[start] = generation;
        while (head < tail && target == OAK_EXACT) {
            uint32_t cell = a->planQueue[head++];
            const ExactRow *ex = &a->exactTable[cell];
            uint32_t untried = untriedActions(&ex->h, actions), pick = 0u;
            for (uint32_t d = 0u; d < actions; d++)
                if (((untried >> d) & 1u) && (target == OAK_EXACT || words[d] > pick)) {
                    pick = words[d];
                    target = cell;
                    targetAction = d;
                }
            for (uint32_t d = 0u; d < actions && target == OAK_EXACT; d++) {
                if (ex->h.visits[d] == 0u) continue;
                for (uint32_t b = 0u; b < 2u && tail < OAK_PLAN_BUDGET; b++) {
                    if (ex->modelProb[d][b] < minBranchProb) continue;
                    uint32_t next = ex->modelNext[d][b];
                    uint32_t nextOwner = a->exactTable[next].h.owner;
                    if ((nextOwner != 0u && tag8(nextOwner) != ex->modelTag[d][b]) || a->planStamp[next] == generation) continue;
                    a->planStamp[next] = generation;
                    a->planParent[next] = (uint16_t)cell;
                    a->planParentAction[next] = (uint8_t)d;
                    a->planQueue[tail++] = (uint16_t)next;
                }
            }
        }
    }
    if (target == OAK_EXACT) {
        const TableRowHead *row = &a->exactTable[start].h;
        uint32_t first = 0u;
        for (uint32_t d = 1u; d < actions; d++)
            if (row->q[d] > row->q[first]) first = d;
        float horizon = (float)OAK_FEATURE_TIMESCALE, gap = row->q[first], spread = 0.0f;
        for (uint32_t d = 0u; d < actions; d++) {
            float weight = (d == first ? 1.0f : 0.0f) - policy[d];
            gap -= policy[d] * row->q[d];
            spread += weight * weight / fminf((float)row->visits[d], horizon);
        }
        return gap * gap > a->qNoise[0] * spread ? first : OAK_POLICY_MAX;
    }
    uint32_t length = 1u;
    for (uint32_t cell = target; cell != start; cell = a->planParent[cell]) length++;
    a->planLength = length;
    a->planCells[length - 1u] = (uint16_t)target;
    a->planActions[length - 1u] = (uint8_t)targetAction;
    for (uint32_t cell = target, i = length - 1u; cell != start; cell = a->planParent[cell]) {
        i--;
        a->planCells[i] = a->planParent[cell];
        a->planActions[i] = a->planParentAction[cell];
    }
    return a->planActions[a->planIndex++];
}

static __attribute__((noinline)) void sampleAction(OakAgent *a, uint32_t seed, uint32_t step, float *ctrl) {
    uint32_t actions = a->actionCount;
    if (!a->box) {
        u4 rw = philox4x32(seed, RNG_STREAM_ACT, step, 0u, 0u, 0u);
        u4 rw1 = (actions > 4u && a->activeHead >= OAK_HEADS) ? philox4x32(seed, RNG_STREAM_ACT, step, 1u, 0u, 0u) : rw;
        uint32_t words[8] = { rw.x, rw.y, rw.z, rw.w, rw1.x, rw1.y, rw1.z, rw1.w };
        float logit[OAK_POLICY_MAX], weight[OAK_POLICY_MAX];
        float logitMax = -FLT_MAX;
        for (uint32_t d = 0u; d < actions; d++) {
            logit[d] = finiteOrZero(a->logit[d]);
            logitMax = fmaxf(logitMax, logit[d]);
        }
        float total = 0.0f;
        for (uint32_t d = 0u; d < actions; d++) {
            weight[d] = oak_exp(logit[d] - logitMax);
            total += weight[d];
        }
        float invTotal = 1.0f / total;
        float draw = u01(rw.x) * total, acc = 0.0f;
        uint32_t sampled = actions - 1u;
        for (uint32_t d = 0u; d < actions; d++) {
            acc += weight[d];
            if (draw < acc) {
                sampled = d;
                break;
            }
        }
        float behaviour = weight[sampled] * invTotal;
        if (a->activeHead >= OAK_HEADS) {
            float policy[OAK_POLICY_MAX];
            for (uint32_t d = 0u; d < actions; d++) policy[d] = weight[d] * invTotal;
            uint32_t arm = a->activeHead - OAK_HEADS, chosen;
            if (TABLE_MODE[arm] == TABLE_PLAN) {
                chosen = planAction(a, words, policy);
                behaviour = chosen < actions ? 1.0f : behaviour;
            } else if (TABLE_MODE[arm] == TABLE_CELLQ) chosen = cellQAction(a, policy, &behaviour);
            else chosen = tableAction(a, arm, words, policy, &behaviour);
            sampled = chosen < actions ? chosen : sampled;
        }
        a->qCellTaken = a->qCellValue[sampled];
        for (uint32_t d = 0u; d < actions; d++) ctrl[d] = d == 0u ? (float)sampled : 0.0f;
        a->action = sampled;
        float chosen = weight[sampled] * invTotal;
        float ratio = a->activeHead >= OAK_HEADS ? chosen / behaviour : 1.0f;
        for (uint32_t d = 0u; d < actions; d++) {
            float credit = (d == sampled ? 1.0f : 0.0f) - weight[d] * invTotal;
            a->actionSignal[d] = credit;
            a->actorCreditStaging[d] = ratio * (HEAD_NATURAL[a->learner] ? finiteOrZero(((d == sampled ? 1.0f / chosen : 0.0f) - 1.0f) / (float)actions) : credit);
        }
        return;
    }
    for (uint32_t pair = 0u; 2u * pair < actions; pair++) {
        u4 r = philox4x32(seed, RNG_STREAM_ACT, step, pair >> 1u, 0u, 0u);
        uint32_t w0 = (pair & 1u) ? r.z : r.x;
        uint32_t w1 = (pair & 1u) ? r.w : r.y;
        float radius = sqrtf(-2.0f * oak_log(u01open(w0)));
        float sine, cosine;
        oak_sincos(2.0f * OAK_PI * u01(w1), &sine, &cosine);
        float normals[2] = { radius * cosine, radius * sine };
        for (uint32_t k = 0u; k < 2u && 2u * pair + k < actions; k++) {
            uint32_t d = 2u * pair + k;
            float mu = oak_tanh(finiteOrZero(a->logit[d]));
            float sample = finiteOrZero(mu + normals[k] / (float)OAK_ACTION_PARAMS);
            float clipped = fmaxf(-1.0f, fminf(1.0f, sample));
            ctrl[d] = clipped;
            a->actionSignal[d] = clipped;
            a->actionSignal[actions + d] = 0.0f;
            a->actorCreditStaging[d] = (sample - mu) * (1.0f - mu * mu);
        }
    }
}

static __attribute__((noinline)) void learn(OakAgent *a) {
    OakGeometry g = a->g;
    uint32_t count = g.features, inputs = g.inputs, actions = a->actionCount;
    const float *x = features(a, 1u - a->slot);
    float4 decay = splat4(OAK_ELIGIBILITY_DECAY);
    float4 e0 = splat4(0.0f), e1 = e0;
    for (uint32_t k = 0u; k < count; k += 8u) {
        float4 z0 = __builtin_elementwise_fma(oak_load4(a->valueTrace + k), decay, oak_load4(x + k));
        float4 z1 = __builtin_elementwise_fma(oak_load4(a->valueTrace + k + 4u), decay, oak_load4(x + k + 4u));
        oak_store4(a->valueTrace + k, z0);
        oak_store4(a->valueTrace + k + 4u, z1);
        e0 = __builtin_elementwise_fma(z0, z0, e0);
        e1 = __builtin_elementwise_fma(z1, z1, e1);
    }
    const uint32_t *previousCells = a->cellActive + (1u - a->slot) * OAK_CELLS;
    uint32_t previousTuples = a->stepCount > 0u ? OAK_TUPLES + 2u * a->observationCount : 0u;
    uint32_t policyTuples = a->stepCount == 0u ? 0u : a->box ? OAK_TUPLES : previousTuples;
    float alpha = g.ridgeFloor;
    float valueDelta = finiteOrZero(alpha * a->td / (1.0f + alpha * (oak_sum4(e0 + e1) + (float)previousTuples)));
    if (valueDelta != 0.0f) {
        float4 scale = splat4(valueDelta);
        for (uint32_t k = 0u; k < count; k += 8u) {
            float4 w0 = oak_load4(a->valueWeight + k), n0 = __builtin_elementwise_fma(oak_load4(a->valueTrace + k), scale, w0);
            float4 w1 = oak_load4(a->valueWeight + k + 4u), n1 = __builtin_elementwise_fma(oak_load4(a->valueTrace + k + 4u), scale, w1);
            oak_store4(a->valueWeight + k, select4(oak_finite4(n0), n0, w0));
            oak_store4(a->valueWeight + k + 4u, select4(oak_finite4(n1), n1, w1));
        }
    }
    float *policy = headPolicy(a);
    float4 policyDecay = splat4(fmaf(a->obsCorr, HEAD_DECAY[a->learner] - OAK_ELIGIBILITY_DECAY, OAK_ELIGIBILITY_DECAY));
    float change[OAK_POLICY_MAX];
    float changeSquare = 0.0f;
    uint32_t d = 0u;
    if (actions >= 4u) {
        float *z0p = a->policyTrace, *z1p = a->policyTrace + count, *z2p = a->policyTrace + 2u * count, *z3p = a->policyTrace + 3u * count;
        float4 c0 = splat4(a->actorCredit[0]), c1 = splat4(a->actorCredit[1]), c2 = splat4(a->actorCredit[2]), c3 = splat4(a->actorCredit[3]);
        float4 a0 = splat4(0.0f), a1 = a0, b0 = a0, b1 = a0, g0 = a0, g1 = a0, h0 = a0, h1 = a0;
        for (uint32_t k = 0u; k < count; k += 8u) {
            float4 x0 = oak_load4(x + k), x1 = oak_load4(x + k + 4u);
            float4 za0 = __builtin_elementwise_fma(oak_load4(z0p + k), policyDecay, c0 * x0);
            float4 za1 = __builtin_elementwise_fma(oak_load4(z0p + k + 4u), policyDecay, c0 * x1);
            float4 zb0 = __builtin_elementwise_fma(oak_load4(z1p + k), policyDecay, c1 * x0);
            float4 zb1 = __builtin_elementwise_fma(oak_load4(z1p + k + 4u), policyDecay, c1 * x1);
            float4 zc0 = __builtin_elementwise_fma(oak_load4(z2p + k), policyDecay, c2 * x0);
            float4 zc1 = __builtin_elementwise_fma(oak_load4(z2p + k + 4u), policyDecay, c2 * x1);
            float4 zd0 = __builtin_elementwise_fma(oak_load4(z3p + k), policyDecay, c3 * x0);
            float4 zd1 = __builtin_elementwise_fma(oak_load4(z3p + k + 4u), policyDecay, c3 * x1);
            oak_store4(z0p + k, za0);
            oak_store4(z0p + k + 4u, za1);
            oak_store4(z1p + k, zb0);
            oak_store4(z1p + k + 4u, zb1);
            oak_store4(z2p + k, zc0);
            oak_store4(z2p + k + 4u, zc1);
            oak_store4(z3p + k, zd0);
            oak_store4(z3p + k + 4u, zd1);
            a0 = __builtin_elementwise_fma(za0, x0, a0);
            a1 = __builtin_elementwise_fma(za1, x1, a1);
            b0 = __builtin_elementwise_fma(zb0, x0, b0);
            b1 = __builtin_elementwise_fma(zb1, x1, b1);
            g0 = __builtin_elementwise_fma(zc0, x0, g0);
            g1 = __builtin_elementwise_fma(zc1, x1, g1);
            h0 = __builtin_elementwise_fma(zd0, x0, h0);
            h1 = __builtin_elementwise_fma(zd1, x1, h1);
        }
        float4 hs = oak_hsum4(a0 + a1, b0 + b1, g0 + g1, h0 + h1);
        change[0] = fmaf((float)policyTuples, a->actorCredit[0], hs.x);
        change[1] = fmaf((float)policyTuples, a->actorCredit[1], hs.y);
        change[2] = fmaf((float)policyTuples, a->actorCredit[2], hs.z);
        change[3] = fmaf((float)policyTuples, a->actorCredit[3], hs.w);
        changeSquare = fmaf(change[0], change[0], changeSquare);
        changeSquare = fmaf(change[1], change[1], changeSquare);
        changeSquare = fmaf(change[2], change[2], changeSquare);
        changeSquare = fmaf(change[3], change[3], changeSquare);
        d = 4u;
    }
    for (; d + 2u <= actions; d += 2u) {
        float *za = a->policyTrace + d * count;
        float *zb = a->policyTrace + (d + 1u) * count;
        float4 ca = splat4(a->actorCredit[d]), cb = splat4(a->actorCredit[d + 1u]);
        float4 a0 = splat4(0.0f), a1 = a0, b0 = a0, b1 = a0;
        for (uint32_t k = 0u; k < count; k += 8u) {
            float4 x0 = oak_load4(x + k), x1 = oak_load4(x + k + 4u);
            float4 za0 = __builtin_elementwise_fma(oak_load4(za + k), policyDecay, ca * x0);
            float4 za1 = __builtin_elementwise_fma(oak_load4(za + k + 4u), policyDecay, ca * x1);
            float4 zb0 = __builtin_elementwise_fma(oak_load4(zb + k), policyDecay, cb * x0);
            float4 zb1 = __builtin_elementwise_fma(oak_load4(zb + k + 4u), policyDecay, cb * x1);
            oak_store4(za + k, za0);
            oak_store4(za + k + 4u, za1);
            oak_store4(zb + k, zb0);
            oak_store4(zb + k + 4u, zb1);
            a0 = __builtin_elementwise_fma(za0, x0, a0);
            a1 = __builtin_elementwise_fma(za1, x1, a1);
            b0 = __builtin_elementwise_fma(zb0, x0, b0);
            b1 = __builtin_elementwise_fma(zb1, x1, b1);
        }
        change[d] = fmaf((float)policyTuples, a->actorCredit[d], oak_sum4(a0 + a1));
        change[d + 1u] = fmaf((float)policyTuples, a->actorCredit[d + 1u], oak_sum4(b0 + b1));
        changeSquare = fmaf(change[d], change[d], changeSquare);
        changeSquare = fmaf(change[d + 1u], change[d + 1u], changeSquare);
    }
    for (; d < actions; d++) {
        float *z = a->policyTrace + d * count;
        float4 credit = splat4(a->actorCredit[d]), c0 = splat4(0.0f), c1 = c0;
        for (uint32_t k = 0u; k < count; k += 8u) {
            float4 x0 = oak_load4(x + k), x1 = oak_load4(x + k + 4u);
            float4 z0 = __builtin_elementwise_fma(oak_load4(z + k), policyDecay, credit * x0);
            float4 z1 = __builtin_elementwise_fma(oak_load4(z + k + 4u), policyDecay, credit * x1);
            oak_store4(z + k, z0);
            oak_store4(z + k + 4u, z1);
            c0 = __builtin_elementwise_fma(z0, x0, c0);
            c1 = __builtin_elementwise_fma(z1, x1, c1);
        }
        change[d] = fmaf((float)policyTuples, a->actorCredit[d], oak_sum4(c0 + c1));
        changeSquare = fmaf(change[d], change[d], changeSquare);
    }
    float actorStep = a->actorStep;
    float policyStep = finiteOrZero(actorStep / fmaxf(1.0f, fabsf(actorStep) * sqrtf(changeSquare)));
    float qStep = a->qCellStep;
    uint32_t qAct = a->qCellAct;
    if (previousTuples > 0u) {
        CellVQ *vq = a->cellVQ;
        if (policyStep != 0.0f) {
            float4 creditLow = { a->actorCredit[0], a->actorCredit[1], a->actorCredit[2], a->actorCredit[3] };
            float4 creditHigh = { a->actorCredit[4], a->actorCredit[5], 0.0f, 0.0f };
            float4 stepLow = creditLow * policyStep, stepHigh = creditHigh * policyStep;
            float *cp = headCellBase(a);
            if (valueDelta != 0.0f && qStep != 0.0f) {
                for (uint32_t t = 0u; t < 8u && t < policyTuples; t++) __builtin_prefetch(cp + previousCells[t] * OAK_CELL_STRIDE, 1, 1);
                if (actions <= 4u) {
                    for (uint32_t t = 0u; t < policyTuples; t++) {
                        if (t + 8u < policyTuples) {
                            uint32_t pIdx = previousCells[t + 8u];
                            __builtin_prefetch(vq + pIdx, 1, 1);
                            __builtin_prefetch(cp + pIdx * OAK_CELL_STRIDE, 1, 1);
                        }
                        uint32_t idx = previousCells[t];
                        CellVQ *s = vq + idx;
                        float nv = s->value + valueDelta;
                        s->value = isfinite(nv) ? nv : s->value;
                        float nq = s->q[qAct] + qStep;
                        s->q[qAct] = isfinite(nq) ? nq : s->q[qAct];
                        float *c = cp + idx * OAK_CELL_STRIDE;
                        float4 low = oak_load4(c), nextLow = low + stepLow;
                        oak_store4(c, select4(oak_finite4(nextLow), nextLow, low));
                    }
                } else {
                    for (uint32_t t = 0u; t < policyTuples; t++) {
                        if (t + 8u < policyTuples) {
                            uint32_t pIdx = previousCells[t + 8u];
                            __builtin_prefetch(vq + pIdx, 1, 1);
                            __builtin_prefetch(cp + pIdx * OAK_CELL_STRIDE, 1, 1);
                        }
                        uint32_t idx = previousCells[t];
                        CellVQ *s = vq + idx;
                        float nv = s->value + valueDelta;
                        s->value = isfinite(nv) ? nv : s->value;
                        float nq = s->q[qAct] + qStep;
                        s->q[qAct] = isfinite(nq) ? nq : s->q[qAct];
                        float *c = cp + idx * OAK_CELL_STRIDE;
                        float4 low = oak_load4(c), nextLow = low + stepLow;
                        float4 high = oak_load4(c + 4u), nextHigh = high + stepHigh;
                        oak_store4(c, select4(oak_finite4(nextLow), nextLow, low));
                        oak_store4(c + 4u, select4(oak_finite4(nextHigh), nextHigh, high));
                    }
                }
            } else {
                for (uint32_t t = 0u; t < 8u && t < policyTuples; t++) __builtin_prefetch(cp + previousCells[t] * OAK_CELL_STRIDE, 1, 1);
                for (uint32_t t = 0u; t < policyTuples; t++) {
                    if (t + 8u < policyTuples) {
                        uint32_t pIdx = previousCells[t + 8u];
                        if (valueDelta != 0.0f || qStep != 0.0f) __builtin_prefetch(vq + pIdx, 1, 1);
                        __builtin_prefetch(cp + pIdx * OAK_CELL_STRIDE, 1, 1);
                    }
                    uint32_t idx = previousCells[t];
                    if (valueDelta != 0.0f) {
                        float *v = &vq[idx].value;
                        float next = *v + valueDelta;
                        *v = isfinite(next) ? next : *v;
                    }
                    if (qStep != 0.0f) {
                        float *q = &vq[idx].q[qAct];
                        float next = *q + qStep;
                        *q = isfinite(next) ? next : *q;
                    }
                    float *c = cp + idx * OAK_CELL_STRIDE;
                    float4 low = oak_load4(c), nextLow = low + stepLow;
                    oak_store4(c, select4(oak_finite4(nextLow), nextLow, low));
                    if (actions > 4u) {
                        float4 high = oak_load4(c + 4u), nextHigh = high + stepHigh;
                        oak_store4(c + 4u, select4(oak_finite4(nextHigh), nextHigh, high));
                    }
                }
            }
            if (valueDelta != 0.0f) {
                for (uint32_t t = policyTuples; t < previousTuples; t++) {
                    if (t + 8u < previousTuples) __builtin_prefetch(vq + previousCells[t + 8u], 1, 1);
                    float *v = &vq[previousCells[t]].value;
                    float next = *v + valueDelta;
                    *v = isfinite(next) ? next : *v;
                }
            }
        } else if (valueDelta != 0.0f || qStep != 0.0f) {
            for (uint32_t t = 0u; t < previousTuples; t++) {
                if (t + 8u < previousTuples) __builtin_prefetch(vq + previousCells[t + 8u], 1, 1);
                uint32_t idx = previousCells[t];
                if (valueDelta != 0.0f) {
                    float *v = &vq[idx].value;
                    float next = *v + valueDelta;
                    *v = isfinite(next) ? next : *v;
                }
                if (qStep != 0.0f) {
                    float *q = &vq[idx].q[qAct];
                    float next = *q + qStep;
                    *q = isfinite(next) ? next : *q;
                }
            }
        }
    }
    if (policyStep != 0.0f) {
        float4 policyScale = splat4(policyStep);
        d = 0u;
        if (actions >= 4u) {
            float *r0 = policy, *r1 = policy + count, *r2 = policy + 2u * count, *r3 = policy + 3u * count;
            const float *z0 = a->policyTrace, *z1 = a->policyTrace + count, *z2 = a->policyTrace + 2u * count, *z3 = a->policyTrace + 3u * count;
            for (uint32_t k = 0u; k < count; k += 8u) {
                float4 w00 = oak_load4(r0 + k), n00 = __builtin_elementwise_fma(oak_load4(z0 + k), policyScale, w00);
                float4 w01 = oak_load4(r0 + k + 4u), n01 = __builtin_elementwise_fma(oak_load4(z0 + k + 4u), policyScale, w01);
                float4 w10 = oak_load4(r1 + k), n10 = __builtin_elementwise_fma(oak_load4(z1 + k), policyScale, w10);
                float4 w11 = oak_load4(r1 + k + 4u), n11 = __builtin_elementwise_fma(oak_load4(z1 + k + 4u), policyScale, w11);
                float4 w20 = oak_load4(r2 + k), n20 = __builtin_elementwise_fma(oak_load4(z2 + k), policyScale, w20);
                float4 w21 = oak_load4(r2 + k + 4u), n21 = __builtin_elementwise_fma(oak_load4(z2 + k + 4u), policyScale, w21);
                float4 w30 = oak_load4(r3 + k), n30 = __builtin_elementwise_fma(oak_load4(z3 + k), policyScale, w30);
                float4 w31 = oak_load4(r3 + k + 4u), n31 = __builtin_elementwise_fma(oak_load4(z3 + k + 4u), policyScale, w31);
                oak_store4(r0 + k, select4(oak_finite4(n00), n00, w00));
                oak_store4(r0 + k + 4u, select4(oak_finite4(n01), n01, w01));
                oak_store4(r1 + k, select4(oak_finite4(n10), n10, w10));
                oak_store4(r1 + k + 4u, select4(oak_finite4(n11), n11, w11));
                oak_store4(r2 + k, select4(oak_finite4(n20), n20, w20));
                oak_store4(r2 + k + 4u, select4(oak_finite4(n21), n21, w21));
                oak_store4(r3 + k, select4(oak_finite4(n30), n30, w30));
                oak_store4(r3 + k + 4u, select4(oak_finite4(n31), n31, w31));
            }
            d = 4u;
        }
        for (; d + 2u <= actions; d += 2u) {
            float *ra = policy + d * count, *rb = policy + (d + 1u) * count;
            const float *za = a->policyTrace + d * count, *zb = a->policyTrace + (d + 1u) * count;
            for (uint32_t k = 0u; k < count; k += 8u) {
                float4 wa0 = oak_load4(ra + k), na0 = __builtin_elementwise_fma(oak_load4(za + k), policyScale, wa0);
                float4 wa1 = oak_load4(ra + k + 4u), na1 = __builtin_elementwise_fma(oak_load4(za + k + 4u), policyScale, wa1);
                float4 wb0 = oak_load4(rb + k), nb0 = __builtin_elementwise_fma(oak_load4(zb + k), policyScale, wb0);
                float4 wb1 = oak_load4(rb + k + 4u), nb1 = __builtin_elementwise_fma(oak_load4(zb + k + 4u), policyScale, wb1);
                oak_store4(ra + k, select4(oak_finite4(na0), na0, wa0));
                oak_store4(ra + k + 4u, select4(oak_finite4(na1), na1, wa1));
                oak_store4(rb + k, select4(oak_finite4(nb0), nb0, wb0));
                oak_store4(rb + k + 4u, select4(oak_finite4(nb1), nb1, wb1));
            }
        }
        for (; d < actions; d++) {
            float *row = policy + d * count;
            const float *z = a->policyTrace + d * count;
            for (uint32_t k = 0u; k < count; k += 8u) {
                float4 w0 = oak_load4(row + k), n0 = __builtin_elementwise_fma(oak_load4(z + k), policyScale, w0);
                float4 w1 = oak_load4(row + k + 4u), n1 = __builtin_elementwise_fma(oak_load4(z + k + 4u), policyScale, w1);
                oak_store4(row + k, select4(oak_finite4(n0), n0, w0));
                oak_store4(row + k + 4u, select4(oak_finite4(n1), n1, w1));
            }
        }
    }
    float corrGate = a->obsCorr * a->obsCorr;
    float actorScale = HEAD_SCALE[a->learner] == SCALE_RAW ? fmaf(corrGate, a->tdInv - 1.0f, 1.0f) : 1.0f;
    float actor = finiteOrZero(a->advantage * actorScale);
    float critic = a->tdNormalized * a->tdInv;
    if (actor == 0.0f && critic == 0.0f) return;
    const float *unit = x + inputs;
    uint32_t half = OAK_UNITS / 2u;
    const float *gates = a->unitGate + (1u - a->slot) * half;
    const float *energies = a->unitEnergy + (1u - a->slot) * OAK_UNITS;
    const float *px = permInputs(a, 1u - a->slot);
    float signals[OAK_UNITS];
    float4 vCritic = splat4(critic), vActor = splat4(actor), vTrack = splat4(fmaf(corrGate, sqrtf(g.trackingRate * g.ridgeFloor) - g.trackingRate, g.trackingRate)), vOne = splat4(1.0f), vZero = splat4(0.0f);
    float4 vGate = splat4(a->prevCorr * a->prevCorr);
    for (uint32_t k = 0u; k < half; k += 4u) {
        float4 cr0 = splat4(0.0f), cr1 = cr0;
        for (uint32_t aidx = 0u; aidx < actions; aidx++) {
            float4 ac = splat4(a->actorCredit[aidx]);
            const float *prow = policy + aidx * count + inputs;
            cr0 = __builtin_elementwise_fma(ac, oak_load4(prow + k), cr0);
            cr1 = __builtin_elementwise_fma(ac, oak_load4(prow + half + k), cr1);
        }
        float4 top0 = __builtin_elementwise_fma(vCritic, oak_load4(a->valueWeight + inputs + k), vActor * cr0);
        float4 top1 = __builtin_elementwise_fma(vCritic, oak_load4(a->valueWeight + inputs + half + k), vActor * cr1);
        float4 t = oak_load4(unit + k), gv = oak_load4(gates + k);
        float4 ev0 = oak_load4(energies + k), ev1 = oak_load4(energies + half + k);
        float4 m = __builtin_elementwise_fma(vGate, t - vOne, vOne);
        float4 num0 = __builtin_elementwise_fma(top1 * vGate, gv, top0) * (vOne - t * t) * vTrack;
        float4 num1 = (top1 * m) * (vOne - gv * gv) * vTrack;
        oak_store4(signals + k, select4((lane4)(ev0 > vZero), oak_finiteOrZero4(num0 * oak_rcp4_fast(ev0)), vZero));
        oak_store4(signals + half + k, select4((lane4)(ev1 > vZero), oak_finiteOrZero4(num1 * oak_rcp4_fast(ev1)), vZero));
    }
    for (uint32_t j = 0u; j < OAK_UNITS; j++) {
        float signal = signals[j];
        if (signal == 0.0f) continue;
        float *weight = a->projection + j * OAK_FANIN;
        const uint16_t *blk = a->unitBlock + j * OAK_FANIN_BLOCKS;
        float4 scaled = splat4(signal);
        for (uint32_t b = 0u; b < OAK_FANIN_BLOCKS; b += 4u) {
            float4 w0 = oak_load4(weight + 4u * b), n0 = __builtin_elementwise_fma(scaled, oak_load4(px + blk[b]), w0);
            float4 w1 = oak_load4(weight + 4u * b + 4u), n1 = __builtin_elementwise_fma(scaled, oak_load4(px + blk[b + 1u]), w1);
            float4 w2 = oak_load4(weight + 4u * b + 8u), n2 = __builtin_elementwise_fma(scaled, oak_load4(px + blk[b + 2u]), w2);
            float4 w3 = oak_load4(weight + 4u * b + 12u), n3 = __builtin_elementwise_fma(scaled, oak_load4(px + blk[b + 3u]), w3);
            oak_store4(weight + 4u * b, select4(oak_finite4(n0), n0, w0));
            oak_store4(weight + 4u * b + 4u, select4(oak_finite4(n1), n1, w1));
            oak_store4(weight + 4u * b + 8u, select4(oak_finite4(n2), n2, w2));
            oak_store4(weight + 4u * b + 12u, select4(oak_finite4(n3), n3, w3));
        }
    }
}

uint32_t oak_stateWords(void) { return (uint32_t)(sizeof(OakAgent) / sizeof(uint32_t)); }

void oak_init(uint32_t *state, uint32_t lane, uint32_t env, uint32_t seed, const float *obs) {
    (void)lane; (void)obs;
    OakAgent *a = (OakAgent *)state;
    __builtin_memset(a, 0, sizeof *a);
    uint32_t observationCount = env_obs(env);
    OakGeometry g = oak_geometry(observationCount);
    a->g = g;
    a->betaFloor = 2.0f * oak_log(g.trackingRate);
    a->observationCount = observationCount;
    a->actionCount = env_actions(env);
    a->box = env_box(env);
    uint32_t direct = OAK_PROJECTED_COUNT(observationCount), sources = direct + OAK_TRACE_LEVELS * observationCount;
    uint32_t permStride = ((sources + 4u + 3u) / 4u) * 4u;
    a->sources = sources;
    a->permStride = permStride;
    for (uint32_t p = 0u; p < OAK_PERMS; p++) {
        uint16_t *perm = a->inputPerm + p * sources;
        for (uint32_t i = 0u; i < sources; i++) perm[i] = (uint16_t)(i < direct ? i : g.projected + (i - direct));
        for (uint32_t i = sources - 1u; i > 0u; i--) {
            u4 r = philox4x32(seed, RNG_STREAM_ACT, p * OAK_SOURCES_MAX + i, 11u, 0u, 0u);
            uint32_t pick = (uint32_t)(((uint64_t)r.x * (i + 1u)) >> 32u);
            uint16_t tmp = perm[i];
            perm[i] = perm[pick];
            perm[pick] = tmp;
        }
    }
    if (sources >= OAK_FANIN) {
        for (uint32_t j = 0u; j < OAK_UNITS; j++) {
            uint8_t used[OAK_INPUT_MAX];
            for (uint32_t k = 0u; k < g.inputs; k++) used[k] = 0u;
            uint16_t *dst = a->unitBlock + j * OAK_FANIN_BLOCKS;
            for (uint32_t b = 0u; b < OAK_FANIN_BLOCKS; b++) {
                uint32_t p = b & (OAK_PERMS - 1u);
                const uint16_t *perm = a->inputPerm + p * sources;
                uint32_t bestOff = 0u, bestOverlap = 5u;
                for (uint32_t tries = 0u; tries < 8u && bestOverlap > 0u; tries++) {
                    u4 r = philox4x32(seed, RNG_STREAM_ACT, (j * OAK_FANIN_BLOCKS + b) * 8u + tries, 12u, 0u, 0u);
                    uint32_t off = (uint32_t)(((uint64_t)r.x * sources) >> 32u);
                    uint32_t overlap = 0u;
                    for (uint32_t t = 0u; t < 4u; t++) overlap += used[perm[(off + t) % sources]];
                    if (overlap < bestOverlap) {
                        bestOverlap = overlap;
                        bestOff = off;
                    }
                }
                for (uint32_t t = 0u; t < 4u; t++) used[perm[(bestOff + t) % sources]] = 1u;
                dst[b] = (uint16_t)(p * permStride + bestOff);
            }
        }
    } else {
        for (uint32_t j = 0u; j < OAK_UNITS; j++) {
            uint16_t *dst = a->unitBlock + j * OAK_FANIN_BLOCKS;
            for (uint32_t b = 0u; b < OAK_FANIN_BLOCKS; b += 4u) {
                u4 r = philox4x32(seed, RNG_STREAM_ACT, j * OAK_FANIN_BLOCKS + b, 12u, 0u, 0u);
                uint32_t pick[4] = { r.x, r.y, r.z, r.w };
                for (uint32_t u = 0u; u < 4u; u++) {
                    uint32_t p = (b + u) & (OAK_PERMS - 1u);
                    uint32_t off = (uint32_t)(((uint64_t)pick[u] * sources) >> 32u);
                    dst[b + u] = (uint16_t)(p * permStride + off);
                }
            }
        }
    }
    float fanIn = oak_rsqrt((float)OAK_FANIN);
    for (uint32_t n = 0u; n < OAK_UNITS * OAK_FANIN; n += 4u) {
        u4 r0 = philox4x32(seed, RNG_STREAM_ACT, n, 8u, 0u, 0u), r1 = philox4x32(seed, RNG_STREAM_ACT, n, 9u, 0u, 0u);
        oak_store4(a->projection + n, normalFrom4(r0, r1) * fanIn);
    }
    uint32_t hash = seed * 2654435761u + 977u;
    uint32_t action = hash % OAK_POLICY_MAX;
    float sign = ((hash >> 16u) & 1u) == 0u ? 1.0f : -1.0f;
    for (uint32_t h = 0u; h < OAK_HEADS; h++) a->policy[(h * OAK_POLICY_MAX + action) * g.features] = sign * (1.0f / (float)OAK_ACTION_PARAMS);
    float logAlpha = oak_log(g.ridgeFloor);
    for (uint32_t o = 0u; o < observationCount; o++) {
        a->chAlpha[o] = g.ridgeFloor;
        a->chBeta[o] = logAlpha;
    }
    a->rewardChannel.alpha = g.ridgeFloor;
    a->rewardChannel.beta = logAlpha;
    for (uint32_t h = 0u; h < OAK_HEADS; h++) a->actorMeta[h].alpha = 1.0f;
}

static __attribute__((noinline)) void rotate(OakAgent *a, uint32_t seed, uint32_t step) {
    uint32_t head = a->activeHead;
    float blockMean = a->blockReward / (float)OAK_BLOCK;
    a->blockCount += 1u;
    float rate = 1.0f / (a->headWeight[head] + 1.0f);
    a->headMean[head] = keepFinite(a->headMean[head], a->headMean[head] + rate * (blockMean - a->headMean[head]));
    a->headSquare[head] = keepFinite(a->headSquare[head], a->headSquare[head] + rate * (blockMean * blockMean - a->headSquare[head]));
    a->headWeight[head] += 1.0f;
    a->headBlocks[head] += 1u;
    a->blockReward = 0.0f;
    a->blockSteps = 0u;
    float strongest = -1.0f;
    for (uint32_t o = 0u; o < a->observationCount; o++) {
        float strength = 0.0f;
        for (uint32_t d = 0u; d < a->actionCount; d++) strength += fabsf(a->control[o * OAK_CELL_STRIDE + d]);
        if (strength > strongest) {
            strongest = strength;
            a->self = o;
        }
    }
    uint32_t arms = a->box ? OAK_HEADS : OAK_ARMS, next = arms;
    if ((a->blockCount & (a->blockCount - 1u)) == 0u)
        for (uint32_t h = 0u; h < arms; h++)
            if (h != head && (h >= OAK_HEADS || h == a->learner) && a->headBlocks[h] >= 2u) {
                a->headMean[h] = 0.0f;
                a->headSquare[h] = 0.0f;
                a->headWeight[h] = 0.0f;
                a->headBlocks[h] = 0u;
            }
    for (uint32_t h = 0u; h < arms && next == arms; h++)
        if (a->headBlocks[h] < 2u) next = h;
    if (next == arms) {
        u4 r0 = philox4x32(seed, RNG_STREAM_ACT, step, 4u, 0u, 0u), r1 = philox4x32(seed, RNG_STREAM_ACT, step, 5u, 0u, 0u);
        u4 r2 = philox4x32(seed, RNG_STREAM_ACT, step, 6u, 0u, 0u), r3 = philox4x32(seed, RNG_STREAM_ACT, step, 7u, 0u, 0u);
        float4 low = normalFrom4(r0, r1), high = normalFrom4(r2, r3);
        float draw[8] = { low.x, low.y, low.z, low.w, high.x, high.y, high.z, high.w };
        float bestSample = -INFINITY, pooled = 0.0f, weight = 0.0f;
        for (uint32_t h = 0u; h < arms; h++) {
            pooled += a->headWeight[h] * fmaxf(a->headSquare[h] - a->headMean[h] * a->headMean[h], 0.0f);
            weight += a->headWeight[h];
        }
        pooled /= weight;
        next = 0u;
        for (uint32_t h = 0u; h < arms; h++) {
            float own = fmaxf(a->headSquare[h] - a->headMean[h] * a->headMean[h], 0.0f);
            float spread = sqrtf((a->headWeight[h] * own + pooled) / ((a->headWeight[h] + 1.0f) * a->headWeight[h]));
            float sample = a->headMean[h] + draw[h] * spread;
            if (sample > bestSample) {
                bestSample = sample;
                next = h;
            }
        }
    }
    if (next < OAK_HEADS && next != a->learner) {
        uint32_t count = a->g.features;
        for (uint32_t k = 0u; k < count; k++) a->valueTrace[k] = 0.0f;
        for (uint32_t k = 0u; k < a->actionCount * count; k++) a->policyTrace[k] = 0.0f;
        a->learner = next;
    }
    a->activeHead = next;
}

__attribute__((noinline)) void oak_act(uint32_t *state, uint32_t lane, uint32_t env, uint32_t seed, uint32_t step, const float *obs, float reward, uint32_t done, float *ctrl) {
    (void)lane; (void)env; (void)done;
    OakAgent *a = (OakAgent *)state;
    a->blockReward += finiteOrZero(reward);
    a->blockSteps += 1u;
    if (a->blockSteps == OAK_BLOCK) rotate(a, seed, step);
    a->reward = reward;
    a->stepCount = step;
    a->slot = step & 1u;
    perceive(a, obs);
    evaluate(a);
    if (!a->box && step > 0u)
        for (uint32_t key = 0u; key < OAK_KEYS; key++) learnTable(a, key);
    sampleAction(a, seed, step, ctrl);
    learn(a);
    a->stepCount = step + 1u;
}
