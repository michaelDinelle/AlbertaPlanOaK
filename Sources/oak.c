#include "geometry.h"
#include "policy.h"

_Static_assert(OAK_LANES == 1u, "");
_Static_assert(OAK_POLICY_MAX == 6u, "");
_Static_assert(OAK_UNITS % 4u == 0u, "");
_Static_assert(OAK_FEATURE_MAX % 4u == 0u, "");

static inline float4 oak_load4(const float *p) { float4 v; __builtin_memcpy(&v, p, sizeof v); return v; }
static inline float4 oak_exp4(float4 x) {
    float4 n = __builtin_elementwise_floor(__builtin_elementwise_fma(x, splat4(1.44269504f), splat4(0.5f)));
    float4 f = __builtin_elementwise_fma(-n, splat4(0.693145751953125f), x);
    f = __builtin_elementwise_fma(-n, splat4(1.428606765330187e-06f), f);
    float4 p = splat4(1.0f / 5040.0f);
    p = __builtin_elementwise_fma(p, f, splat4(1.0f / 720.0f));
    p = __builtin_elementwise_fma(p, f, splat4(1.0f / 120.0f));
    p = __builtin_elementwise_fma(p, f, splat4(1.0f / 24.0f));
    p = __builtin_elementwise_fma(p, f, splat4(1.0f / 6.0f));
    p = __builtin_elementwise_fma(p, f, splat4(0.5f));
    p = __builtin_elementwise_fma(p, f, splat4(1.0f));
    p = __builtin_elementwise_fma(p, f, splat4(1.0f));
    float4 bounded = __builtin_elementwise_min(__builtin_elementwise_max(n, splat4(-149.0f)), splat4(127.0f));
    float4 scale = (float4)((lane4)(__builtin_convertvector(bounded, int4) + 127) << 23u);
    float4 r = select4((lane4)(n > splat4(127.0f)), (p * splat4(2.0f)) * splat4(wordFloat(254u << 23u)), p * scale);
    r = select4((lane4)(x < splat4(-87.33654f)), splat4(0.0f), r);
    r = select4((lane4)(x > splat4(88.72283f)), splat4(INFINITY), r);
    return select4((lane4)(x != x), x, r);
}
static inline float4 oak_log4(float4 x) {
    lane4 bits = (lane4)x;
    int4 exponent = (int4)((bits >> 23u) & 255u) - 127;
    float4 m = (float4)((bits & 0x007FFFFFu) | 0x3F800000u);
    lane4 halve = (lane4)(m > splat4(1.41421356f));
    m = select4(halve, m * splat4(0.5f), m);
    exponent += (int4)(halve & 1u);
    float4 s = (m - splat4(1.0f)) / (m + splat4(1.0f));
    float4 s2 = s * s;
    float4 p = splat4(1.0f / 11.0f);
    p = __builtin_elementwise_fma(p, s2, splat4(1.0f / 9.0f));
    p = __builtin_elementwise_fma(p, s2, splat4(1.0f / 7.0f));
    p = __builtin_elementwise_fma(p, s2, splat4(1.0f / 5.0f));
    p = __builtin_elementwise_fma(p, s2, splat4(1.0f / 3.0f));
    p = __builtin_elementwise_fma(p, s2, splat4(1.0f));
    float4 r = __builtin_elementwise_fma(__builtin_convertvector(exponent, float4), splat4(0.693147180559945f), splat4(2.0f) * s * p);
    r = select4((lane4)(x == splat4(INFINITY)), splat4(INFINITY), r);
    r = select4((lane4)(x == splat4(0.0f)), splat4(-INFINITY), r);
    r = select4((lane4)(x < splat4(0.0f)), splat4(NAN), r);
    return select4((lane4)(x != x), x, r);
}
static inline float4 oak_tanh4(float4 x) {
    float4 a = __builtin_elementwise_abs(x);
    float4 a2 = a * a;
    float4 small = a * __builtin_elementwise_fma(a2, __builtin_elementwise_fma(a2, splat4(2.0f / 15.0f), splat4(-1.0f / 3.0f)), splat4(1.0f));
    float4 e = oak_exp4(splat4(2.0f) * a);
    float4 wide = splat4(1.0f) - splat4(2.0f) / (e + splat4(1.0f));
    float4 t = select4((lane4)(a > splat4(9.0f)), splat4(1.0f), wide);
    t = select4((lane4)(a < splat4(0.04f)), small, t);
    t = select4((lane4)(x < splat4(0.0f)), -t, t);
    return select4((lane4)(x != x), x, t);
}
static inline void oak_store4(float *p, float4 v) { __builtin_memcpy(p, &v, sizeof v); }
static inline lane4 oak_finite4(float4 v) { return (lane4)(((lane4)v & 0x7fffffffu) < 0x7f800000u); }
static inline float oak_sum4(float4 v) { return (v.x + v.y) + (v.z + v.w); }

typedef struct {
    float mean, second, alpha, beta, trace, energy;
} Channel;

typedef struct {
    float feature[2u * OAK_FEATURE_MAX];
    float valueWeight[OAK_FEATURE_MAX], valueTrace[OAK_FEATURE_MAX];
    float policy[OAK_HEADS * OAK_POLICY_MAX * OAK_FEATURE_MAX], policyTrace[OAK_POLICY_MAX * OAK_FEATURE_MAX];
    float projection[OAK_UNITS * OAK_FANIN], unitInput[2u * OAK_UNITS * OAK_FANIN], unitEnergy[2u * OAK_UNITS];
    uint32_t unitSource[OAK_UNITS * OAK_FANIN];
    float traceLevel[OAK_TRACE_LEVELS * OAK_OBSERVATION_LIMIT], traceMean[OAK_TRACE_LEVELS * OAK_OBSERVATION_LIMIT], traceSecond[OAK_TRACE_LEVELS * OAK_OBSERVATION_LIMIT];
    float settledMean[OAK_TRACE_LEVELS * OAK_OBSERVATION_LIMIT], settledSecond[OAK_TRACE_LEVELS * OAK_OBSERVATION_LIMIT];
    float cellValue[OAK_TABLE], cellPolicy[OAK_HEADS * OAK_TABLE * OAK_CELL_STRIDE];
    float headMean[OAK_ARMS], headSquare[OAK_ARMS], headWeight[OAK_ARMS], blockReward;
    uint32_t headBlocks[OAK_ARMS], activeHead, learner, blockSteps, blockCount, self;
    float control[OAK_OBSERVATION_LIMIT * OAK_POLICY_LANES];
    uint32_t cellActive[2u * OAK_CELLS];
    Channel actorMeta[OAK_HEADS];
    float channelMean[OAK_CHANNELS], channelSecond[OAK_CHANNELS], channelAlpha[OAK_CHANNELS], channelBeta[OAK_CHANNELS], channelTrace[OAK_CHANNELS], channelEnergy[OAK_CHANNELS];
    float actionSignal[OAK_ACTION_PARAMS];
    float actorCredit[OAK_POLICY_MAX], actorCreditStaging[OAK_POLICY_MAX], logit[OAK_POLICY_MAX];
    float reward, td, tdNormalized, tdSquare, advantage, advantageCenter, averageReward, actorStep, betaFloor;
    float rewardMean, rewardVariance, aspiration, rewardSquare, rewardMagnitude;
    OakGeometry g;
    uint32_t observationCount, actionCount, box, stepCount, slot;
    uint32_t qCell[2u * OAK_KEYS], qTag[2u * OAK_KEYS], action;
    float qSum[OAK_KEYS], qNoise[OAK_KEYS];
    uint32_t qTried[OAK_KEYS], qOpen[OAK_KEYS];
    uint8_t qDistance[OAK_KEYS * OAK_QTABLE * OAK_CELL_STRIDE];
    uint16_t modelNext[OAK_QTABLE * OAK_CELL_STRIDE], planParent[OAK_QTABLE], planQueue[OAK_PLAN_BUDGET], planCells[OAK_PLAN_BUDGET];
    uint8_t modelVaried[OAK_QTABLE * OAK_CELL_STRIDE], planParentAction[OAK_QTABLE], planActions[OAK_PLAN_BUDGET];
    float modelReward[OAK_QTABLE * OAK_CELL_STRIDE];
    uint32_t planStamp[OAK_QTABLE], planGeneration, planLength, planIndex;
    float qTable[OAK_KEYS * OAK_QTABLE * OAK_QROW];
    float qCellWeight[OAK_TABLE * OAK_CELL_STRIDE], qCellValue[OAK_POLICY_MAX], qCellTaken, qCellGain, qCellNoise;
    uint32_t memoryExact[OAK_MEMORY_RING], memorySum[OAK_MEMORY_RING], memoryWindow[OAK_MEMORY_LEVELS], memoryKey[2u * OAK_MEMORY_KEYS], memoryMatched[2];
    float memoryQ[OAK_TABLE * OAK_CELL_STRIDE], memoryValue[OAK_POLICY_MAX], memoryTaken, memoryGain, memoryNoise;
} OakAgent;

enum { SCALE_TD, SCALE_RAW, SCALE_MAGNITUDE };
static const uint8_t HEAD_SCALE[OAK_HEADS] = { SCALE_TD, SCALE_RAW, SCALE_RAW };
static const uint8_t HEAD_NATURAL[OAK_HEADS] = { 0u, 1u, 0u };
static const uint8_t HEAD_ASPIRE[OAK_HEADS] = { 1u, 1u, 0u };
static const uint8_t HEAD_TRUST[OAK_HEADS] = { 0u, 0u, 1u };
static const uint8_t TABLE_KEY[OAK_TABLE_ARMS] = { 0u, 1u, 0u, 1u, 0u };
enum { TABLE_GREEDY, TABLE_FRONTIER, TABLE_PLAN, TABLE_CELLQ };
static const uint8_t TABLE_MODE[OAK_TABLE_ARMS] = { TABLE_GREEDY, TABLE_GREEDY, TABLE_PLAN, TABLE_FRONTIER, TABLE_CELLQ };

static inline float *features(OakAgent *a, uint32_t slot) { return a->feature + slot * OAK_FEATURE_MAX; }
static inline float *headPolicy(OakAgent *a) { return a->policy + a->learner * OAK_POLICY_MAX * OAK_FEATURE_MAX; }
static inline float *headCell(OakAgent *a, uint32_t cell) { return a->cellPolicy + (a->learner * OAK_TABLE + cell) * OAK_CELL_STRIDE; }

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

static inline void metaStep(Channel *c, float signal, float inverseAge, float metaRate, float decay, float betaFloor) {
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

static inline float4 keep4(float4 old, float4 next) { return select4(oak_finite4(next), next, old); }
static inline float4 oak_rsqrt4(float4 x) { return splat4(1.0f) / __builtin_elementwise_sqrt(x); }
static inline float4 standardizeChannels(OakAgent *a, uint32_t o, float4 x, float inverseAge, float floorRate, float metaRate, float decay, float betaFloor, float ridgeFloor) {
    float4 mean = oak_load4(a->channelMean + o), second = oak_load4(a->channelSecond + o), alpha = oak_load4(a->channelAlpha + o);
    float4 beta = oak_load4(a->channelBeta + o), trace = oak_load4(a->channelTrace + o), energy = oak_load4(a->channelEnergy + o);
    lane4 finite = oak_finite4(x);
    float4 signal = x - mean;
    float4 metaGradient = signal * trace;
    float4 gradEnergy = keep4(energy, energy + splat4(inverseAge) * (metaGradient * metaGradient - energy));
    float4 normalized = metaGradient * oak_rsqrt4(__builtin_elementwise_max(gradEnergy, splat4(0.0f)) + splat4(inverseAge));
    float4 nextBeta = keep4(beta, __builtin_elementwise_min(__builtin_elementwise_max(beta + splat4(metaRate) * normalized, splat4(betaFloor)), splat4(0.0f)));
    float4 nextAlpha = keep4(alpha, oak_exp4(nextBeta));
    float4 nextTrace = keep4(trace, splat4(1.0f - decay) * __builtin_elementwise_max(splat4(0.0f), splat4(1.0f) - alpha) * trace + alpha * signal);
    lane4 commit = finite & oak_finite4(signal);
    alpha = select4(commit, nextAlpha, alpha);
    beta = select4(commit, nextBeta, beta);
    trace = select4(commit, nextTrace, trace);
    energy = select4(commit, gradEnergy, energy);
    float4 rate = __builtin_elementwise_max(splat4(floorRate), alpha);
    mean = select4(finite, mean + rate * (x - mean), mean);
    second = select4(finite, second + rate * (x * x - second), second);
    oak_store4(a->channelMean + o, mean);
    oak_store4(a->channelSecond + o, second);
    oak_store4(a->channelAlpha + o, alpha);
    oak_store4(a->channelBeta + o, beta);
    oak_store4(a->channelTrace + o, trace);
    oak_store4(a->channelEnergy + o, energy);
    return (x - mean) * oak_rsqrt4(__builtin_elementwise_max(second - mean * mean, splat4(0.0f)) + splat4(ridgeFloor));
}

static inline uint32_t tableCell(uint32_t hash) {
    hash ^= hash >> 15u;
    hash *= 0x2C1B3C6Du;
    hash ^= hash >> 12u;
    return hash >> (32u - OAK_QTABLE_BITS);
}

static inline uint32_t memoryMix(uint32_t salt, uint32_t now, uint32_t first, uint32_t second) {
    uint32_t hash = (((salt * 0x9E3779B9u ^ now) * 0x01000193u ^ first) * 0x01000193u ^ second) * 0x01000193u;
    hash ^= hash >> 15u;
    hash *= 0x2C1B3C6Du;
    hash ^= hash >> 12u;
    return hash;
}

static inline float *memoryRow(OakAgent *a, uint32_t key) { return a->memoryQ + (key >> (32u - OAK_TABLE_BITS)) * OAK_CELL_STRIDE; }

static inline uint32_t memoryOwner(const float *row) {
    uint32_t owner;
    __builtin_memcpy(&owner, row + OAK_CELL_STRIDE - 1u, sizeof owner);
    return owner;
}

static inline lane4 fineCode4(float4 value) {
    return __builtin_convertvector(__builtin_elementwise_min(__builtin_elementwise_max(__builtin_elementwise_floor(value + splat4(3.5f)), splat4(0.0f)), splat4(6.0f)), lane4);
}

static inline lane4 coarseCode4(float4 value) {
    return __builtin_convertvector(__builtin_elementwise_min(__builtin_elementwise_max(__builtin_elementwise_floor(value + splat4(1.5f)), splat4(0.0f)), splat4(2.0f)), lane4);
}

static inline float4 finite4(float4 v) { return select4(oak_finite4(v), v, splat4(0.0f)); }

static inline lane4 mixCells4(lane4 hash) {
    hash ^= hash >> 15u;
    hash *= 0x2C1B3C6Du;
    hash ^= hash >> 12u;
    return hash >> (32u - OAK_TABLE_BITS);
}

static __attribute__((noinline)) void encode(OakAgent *a, const float *x, const float *obs) {
    OakGeometry g = a->g;
    uint32_t observationCount = a->observationCount;
    float coded[(1u + OAK_TRACE_LEVELS) * OAK_OBSERVATION_LIMIT];
    uint32_t fineCode[(1u + OAK_TRACE_LEVELS) * OAK_OBSERVATION_LIMIT], coarseCode[(1u + OAK_TRACE_LEVELS) * OAK_OBSERVATION_LIMIT], settledCode[OAK_TRACE_LEVELS * OAK_OBSERVATION_LIMIT];
    for (uint32_t o = 0u; o < observationCount; o += 4u) oak_store4(coded + o, oak_load4(x + o));
    float4 momentRate = splat4(trackingRateFor(a->stepCount, OAK_FEATURE_TIMESCALE)), settleRate = splat4(1.0f / ((float)a->stepCount + 1.0f));
    float4 ridge = splat4(a->stepCount < 2u * OAK_CAPACITY ? 1.0f / ((float)a->stepCount + 1.0f) : g.ridgeFloor);
    for (uint32_t c = 0u; c < OAK_TRACE_LEVELS * observationCount; c += 4u) {
        float4 value = oak_load4(a->traceLevel + c), mean = oak_load4(a->traceMean + c), second = oak_load4(a->traceSecond + c);
        float4 center = oak_load4(a->settledMean + c), spread = oak_load4(a->settledSecond + c);
        mean = keep4(mean, mean + momentRate * (value - mean));
        second = keep4(second, second + momentRate * (value * value - second));
        center = keep4(center, center + settleRate * (value - center));
        spread = keep4(spread, spread + settleRate * (value * value - spread));
        oak_store4(a->traceMean + c, mean);
        oak_store4(a->traceSecond + c, second);
        oak_store4(a->settledMean + c, center);
        oak_store4(a->settledSecond + c, spread);
        oak_store4(coded + observationCount + c, finite4((value - mean) * oak_rsqrt4(__builtin_elementwise_max(second - mean * mean, splat4(0.0f)) + ridge)));
        lane4 settled = coarseCode4(finite4((value - center) * oak_rsqrt4(__builtin_elementwise_max(spread - center * center, splat4(0.0f)) + ridge)));
        __builtin_memcpy(settledCode + c, &settled, sizeof settled);
    }
    for (uint32_t c = 0u; c < (1u + OAK_TRACE_LEVELS) * observationCount; c += 4u) {
        float4 value = oak_load4(coded + c);
        lane4 fine = fineCode4(value), coarse = coarseCode4(value);
        __builtin_memcpy(fineCode + c, &fine, sizeof fine);
        __builtin_memcpy(coarseCode + c, &coarse, sizeof coarse);
    }
    uint32_t *cells = a->cellActive + a->slot * OAK_CELLS;
    lane4 chain[OAK_TUPLE_LANES / 4u], fineLane[OAK_TUPLE_LANES / 4u], ledLane[OAK_TUPLE_LANES / 4u];
    for (uint32_t v = 0u; v < OAK_TUPLE_LANES / 4u; v++)
        for (uint32_t r = 0u; r < 4u; r++) {
            uint32_t tuple = 4u * v + r;
            chain[v][r] = (tuple + 1u) * 0x9E3779B9u;
            fineLane[v][r] = (tuple & 1u) && tuple < OAK_TRACKED_TUPLES ? ~0u : 0u;
            ledLane[v][r] = tuple >= 2u && tuple < OAK_TUPLES ? ~0u : 0u;
        }
    for (uint32_t o = 0u; o < observationCount; o++) {
        lane4 fine = fineCode[o], coarse = coarseCode[o];
        for (uint32_t v = 0u; v < OAK_TUPLE_LANES / 4u; v++) chain[v] = (chain[v] ^ ((fine & fineLane[v]) | (coarse & ~fineLane[v]))) * 0x01000193u;
    }
    for (uint32_t o = 0u; o < observationCount; o++) {
        uint32_t code[OAK_TUPLE_LANES] = { 0u };
        for (uint32_t tuple = 2u; tuple < OAK_TRACKED_TUPLES; tuple++) code[tuple] = ((tuple & 1u) ? fineCode : coarseCode)[(tuple >> 1u) * observationCount + o] + 8u;
        for (uint32_t tuple = OAK_TRACKED_TUPLES; tuple < OAK_TUPLES; tuple++) code[tuple] = settledCode[(tuple - OAK_TRACKED_TUPLES) * observationCount + o] + 8u;
        for (uint32_t v = 0u; v < OAK_TUPLE_LANES / 4u; v++) {
            lane4 codes;
            __builtin_memcpy(&codes, code + 4u * v, sizeof codes);
            lane4 next = (chain[v] ^ codes) * 0x01000193u;
            chain[v] = (next & ledLane[v]) | (chain[v] & ~ledLane[v]);
        }
    }
    for (uint32_t v = 0u; v < OAK_TUPLE_LANES / 4u; v++) {
        lane4 cell = mixCells4(chain[v]);
        __builtin_memcpy(cells + 4u * v, &cell, sizeof cell);
    }
    lane4 ahead = { 0u, 1u, 2u, 3u };
    for (uint32_t o = 0u; o < observationCount; o += 4u) {
        lane4 fine;
        __builtin_memcpy(&fine, fineCode + o, sizeof fine);
        lane4 cell = mixCells4(((OAK_TUPLES + 1u + o + ahead) * 0x9E3779B9u ^ fine) * 0x01000193u);
        __builtin_memcpy(cells + OAK_TUPLES + o, &cell, sizeof cell);
    }
    const float *previous = features(a, 1u - a->slot);
    float4 controlRate = splat4(1.0f / (float)a->stepCount);
    for (uint32_t o = 0u; o < observationCount && a->stepCount > 0u; o++) {
        float4 change = splat4(x[o] - previous[o]);
        for (uint32_t d = 0u; d < a->actionCount; d += 4u) {
            float *c = a->control + o * OAK_POLICY_LANES + d;
            float4 old = oak_load4(c);
            oak_store4(c, keep4(old, old + controlRate * (oak_load4(a->actionSignal + d) * change - old)));
        }
    }
    uint32_t selfBits;
    __builtin_memcpy(&selfBits, obs + a->self, sizeof selfBits);
    for (uint32_t o = 0u; o < observationCount; o += 4u) {
        lane4 world;
        __builtin_memcpy(&world, obs + o, sizeof world);
        lane4 cell = mixCells4(((((OAK_TUPLES + OAK_OBSERVATION_LIMIT + 1u + o + ahead) * 0x9E3779B9u ^ selfBits) * 0x01000193u) ^ world) * 0x01000193u);
        __builtin_memcpy(cells + OAK_TUPLES + observationCount + o, &cell, sizeof cell);
    }
    if (a->box) return;
    uint32_t exact = 0x9E3779B9u, sum = 0u;
    for (uint32_t o = 0u; o < observationCount; o++) {
        uint32_t bits;
        __builtin_memcpy(&bits, obs + o, sizeof bits);
        bits ^= bits >> 16u;
        bits *= 0x85EBCA6Bu;
        bits ^= bits >> 13u;
        exact = (exact ^ bits) * 0x01000193u;
        uint32_t salted = (bits ^ (o + 1u) * 0x9E3779B9u) * 0x2C1B3C6Du;
        sum += salted ^ salted >> 15u;
    }
    uint32_t ring = OAK_MEMORY_RING - 1u, now = a->stepCount & ring, last = (now - 1u) & ring;
    for (uint32_t m = 0u; m < OAK_MEMORY_LEVELS; m++) a->memoryWindow[m] += a->memorySum[last] - a->memorySum[(last - (2u << m)) & ring];
    a->memoryExact[now] = exact;
    a->memorySum[now] = sum;
    uint32_t *memory = a->memoryKey + a->slot * OAK_MEMORY_KEYS;
    for (uint32_t m = 0u; m < OAK_MEMORY_LEVELS; m++) {
        uint32_t lag = 2u << m;
        memory[m] = memoryMix(m + 1u, exact, a->memoryExact[(now - lag) & ring], a->memoryExact[(now - lag - 1u) & ring]);
        memory[OAK_MEMORY_LEVELS + m] = memoryMix(OAK_MEMORY_LEVELS + m + 1u, exact, a->memoryWindow[m], 0u);
    }
    uint32_t history = exact;
    for (uint32_t c = observationCount; c < (1u + OAK_TRACE_LEVELS) * observationCount; c++) history = (history ^ fineCode[c]) * 0x01000193u;
    a->qCell[a->slot * OAK_KEYS] = tableCell(exact);
    a->qCell[a->slot * OAK_KEYS + 1u] = tableCell(history);
    a->qTag[a->slot * OAK_KEYS] = exact;
    a->qTag[a->slot * OAK_KEYS + 1u] = history;
}

static __attribute__((noinline)) void perceive(OakAgent *a, const float *obs) {
    OakGeometry g = a->g;
    uint32_t observationCount = a->observationCount;
    float standardized[OAK_CHANNELS];
    uint32_t metaAge = a->stepCount < OAK_FEATURE_TIMESCALE ? a->stepCount : OAK_FEATURE_TIMESCALE;
    float inverseAge = 1.0f / ((float)metaAge + 1.0f);
    float floorRate = 1.0f / ((float)a->stepCount + 1.0f);
    for (uint32_t o = 0u; o <= observationCount; o += 4u) {
        float4 value;
        for (uint32_t r = 0u; r < 4u; r++) value[r] = o + r < observationCount ? obs[o + r] : o + r == observationCount ? a->reward : 0.0f;
        oak_store4(standardized + o, standardizeChannels(a, o, value, inverseAge, floorRate, g.metaRate, g.trackingRate, a->betaFloor, g.ridgeFloor));
    }
    float *x = features(a, a->slot);
    for (uint32_t o = 0u; o < observationCount; o++) x[o] = finiteOrZero(standardized[o]);
    for (uint32_t d = 0u; d < OAK_ACTION_PARAMS; d++) x[observationCount + d] = finiteOrZero(a->actionSignal[d]);
    x[observationCount + OAK_ACTION_PARAMS] = finiteOrZero(standardized[observationCount]);
    x[observationCount + OAK_ACTION_PARAMS + 1u] = 1.0f;
    for (uint32_t k = OAK_PROJECTED_COUNT(observationCount); k < g.projected; k++) x[k] = 0.0f;
    float *traces = x + g.projected;
    for (uint32_t m = 1u; m <= OAK_TRACE_LEVELS; m++) {
        float rate = 1.0f / (float)(1u << m);
        float *level = a->traceLevel + (m - 1u) * observationCount;
        for (uint32_t o = 0u; o < observationCount; o++) {
            float next = level[o] + rate * (standardized[o] - level[o]);
            level[o] = isfinite(obs[o]) ? next : level[o];
            traces[(m - 1u) * observationCount + o] = finiteOrZero(level[o]);
        }
    }
    for (uint32_t k = g.projected + OAK_TRACE_LEVELS * observationCount; k < g.inputs; k++) x[k] = 0.0f;
    encode(a, x, obs);
    float *unit = x + g.inputs, *gathered = a->unitInput + a->slot * OAK_UNITS * OAK_FANIN;
    for (uint32_t j = 0u; j < OAK_UNITS; j++) {
        const float *weight = a->projection + j * OAK_FANIN;
        const uint32_t *source = a->unitSource + j * OAK_FANIN;
        float4 sum = splat4(0.0f), energy = sum;
        for (uint32_t k = 0u; k < OAK_FANIN; k += 4u) {
            float4 in = { x[source[k]], x[source[k + 1u]], x[source[k + 2u]], x[source[k + 3u]] };
            oak_store4(gathered + j * OAK_FANIN + k, in);
            sum = __builtin_elementwise_fma(oak_load4(weight + k), in, sum);
            energy = __builtin_elementwise_fma(in, in, energy);
        }
        unit[j] = oak_sum4(sum);
        a->unitEnergy[a->slot * OAK_UNITS + j] = oak_sum4(energy);
    }
    for (uint32_t j = 0u; j < OAK_UNITS; j += 4u) oak_store4(unit + j, oak_tanh4(oak_load4(unit + j)));
}

static __attribute__((noinline)) void evaluate(OakAgent *a) {
    OakGeometry g = a->g;
    uint32_t count = g.features, actions = a->actionCount;
    const float *current = features(a, a->slot), *previous = features(a, 1u - a->slot);
    float currentValue = dotProduct(a->valueWeight, current, count);
    float previousValue = dotProduct(a->valueWeight, previous, count);
    const uint32_t *currentCells = a->cellActive + a->slot * OAK_CELLS, *previousCells = a->cellActive + (1u - a->slot) * OAK_CELLS;
    uint32_t cellCount = OAK_TUPLES + 2u * a->observationCount, previousTuples = a->stepCount > 0u ? cellCount : 0u;
    float4 cellLow = splat4(0.0f), cellHigh = cellLow;
    for (uint32_t t = 0u; t < cellCount; t++) currentValue += a->cellValue[currentCells[t]];
    for (uint32_t t = 0u; t < (a->box ? OAK_TUPLES : cellCount); t++) {
        const float *c = headCell(a, currentCells[t]);
        cellLow += oak_load4(c);
        cellHigh += oak_load4(c + 4u);
    }
    for (uint32_t t = 0u; t < previousTuples; t++) previousValue += a->cellValue[previousCells[t]];
    float cellLogit[8] = { cellLow.x, cellLow.y, cellLow.z, cellLow.w, cellHigh.x, cellHigh.y, 0.0f, 0.0f };
    const float *policy = headPolicy(a);
    for (uint32_t d = 0u; d < actions; d++) a->logit[d] = dotProduct(policy + d * OAK_FEATURE_MAX, current, count) + cellLogit[d];
    float averageRewardOld = a->averageReward;
    float delta = finiteOrZero(a->reward - averageRewardOld + currentValue - previousValue);
    a->td = delta;
    float averageRewardNew = averageRewardOld + g.trackingRate * delta;
    a->averageReward = isfinite(averageRewardNew) ? averageRewardNew : averageRewardOld;
    float squareRate = trackingRateFor(a->stepCount, OAK_FEATURE_TIMESCALE);
    a->tdSquare = keepFinite(a->tdSquare, a->tdSquare + squareRate * (delta * delta - a->tdSquare));
    float tdScale = oak_rsqrt(a->tdSquare);
    a->tdNormalized = finiteOrZero(delta * tdScale);
    float advantageCenterOld = a->advantageCenter;
    float advantageError = delta - advantageCenterOld;
    float rewardRate = trackingRateFor(a->stepCount, OAK_FEATURE_TIMESCALE);
    float rewardError = a->reward - averageRewardOld;
    a->rewardSquare = keepFinite(a->rewardSquare, a->rewardSquare + rewardRate * (rewardError * rewardError - a->rewardSquare));
    a->rewardMagnitude = keepFinite(a->rewardMagnitude, a->rewardMagnitude + rewardRate * (fabsf(rewardError) - a->rewardMagnitude));
    float rewardGap = a->reward - a->rewardMean;
    a->rewardMean = keepFinite(a->rewardMean, a->rewardMean + rewardRate * rewardGap);
    a->rewardVariance = keepFinite(a->rewardVariance, a->rewardVariance + rewardRate * (rewardGap * rewardGap - a->rewardVariance));
    float assured = a->rewardMean - sqrtf(fmaxf(a->rewardVariance, 0.0f) * rewardRate);
    float aspirationOld = a->stepCount == 0u ? assured : a->aspiration;
    a->aspiration = keepFinite(aspirationOld, fmaxf(assured, aspirationOld + g.metaRate * (a->rewardMean - aspirationOld)));
    float shortfall = finiteOrZero(fmaxf(a->aspiration - a->rewardMean, 0.0f));
    float informed = a->rewardSquare / a->tdSquare;
    float trust = informed >= 1.0f ? 1.0f : sqrtf(informed);
    uint32_t head = a->learner;
    float scale = HEAD_SCALE[head] == SCALE_TD ? tdScale : HEAD_SCALE[head] == SCALE_RAW ? 1.0f : a->rewardMagnitude / a->rewardSquare;
    a->advantage = finiteOrZero((advantageError - (HEAD_ASPIRE[head] ? shortfall : 0.0f)) * scale * (HEAD_TRUST[head] ? trust : 1.0f));
    a->advantageCenter = keepFinite(advantageCenterOld, advantageCenterOld + g.trackingRate * advantageError);
    uint32_t metaAge = a->stepCount < OAK_FEATURE_TIMESCALE ? a->stepCount : OAK_FEATURE_TIMESCALE;
    Channel *meta = a->actorMeta + head;
    if (!a->box) metaStep(meta, a->advantage, 1.0f / ((float)metaAge + 1.0f), g.metaRate, g.trackingRate, a->betaFloor);
    a->actorStep = finiteOrZero(meta->alpha * a->advantage / OAK_ACTOR_ALPHA_DENOM);
    for (uint32_t d = 0u; d < OAK_POLICY_MAX; d++) a->actorCredit[d] = d < actions ? a->actorCreditStaging[d] : 0.0f;
    if (a->box) return;
    float4 qLow = splat4(0.0f), qHigh = qLow;
    for (uint32_t t = 0u; t < cellCount; t++) {
        const float *c = a->qCellWeight + currentCells[t] * OAK_CELL_STRIDE;
        qLow += oak_load4(c);
        qHigh += oak_load4(c + 4u);
    }
    float qValue[8] = { qLow.x, qLow.y, qLow.z, qLow.w, qHigh.x, qHigh.y, 0.0f, 0.0f }, qBest = -INFINITY;
    for (uint32_t d = 0u; d < actions; d++) {
        a->qCellValue[d] = qValue[d];
        qBest = fmaxf(qBest, qValue[d]);
    }
    const uint32_t *memory = a->memoryKey + a->slot * OAK_MEMORY_KEYS;
    float4 memoryLow = splat4(0.0f), memoryHigh = memoryLow;
    uint32_t matched = 0u;
    for (uint32_t k = 0u; k < OAK_MEMORY_KEYS; k++) {
        const float *row = memoryRow(a, memory[k]);
        if (memoryOwner(row) != memory[k]) continue;
        memoryLow += oak_load4(row);
        memoryHigh += oak_load4(row + 4u);
        matched++;
    }
    a->memoryMatched[a->slot] = matched;
    float residual[8] = { memoryLow.x, memoryLow.y, memoryLow.z, memoryLow.w, memoryHigh.x, memoryHigh.y, 0.0f, 0.0f }, memoryBest = -INFINITY;
    for (uint32_t d = 0u; d < actions; d++) {
        a->memoryValue[d] = qValue[d] + residual[d];
        memoryBest = fmaxf(memoryBest, a->memoryValue[d]);
    }
    if (a->stepCount == 0u) return;
    float qDelta = finiteOrZero(a->reward - a->qCellGain + qBest - a->qCellTaken);
    a->qCellGain = keepFinite(a->qCellGain, a->qCellGain + g.trackingRate * qDelta);
    a->qCellNoise = keepFinite(a->qCellNoise, a->qCellNoise + g.trackingRate * (qDelta * qDelta - a->qCellNoise));
    float qStep = finiteOrZero(g.ridgeFloor * qDelta / (1.0f + g.ridgeFloor * (float)cellCount));
    for (uint32_t t = 0u; t < cellCount; t++) {
        float *c = a->qCellWeight + previousCells[t] * OAK_CELL_STRIDE + a->action;
        float next = *c + qStep;
        *c = isfinite(next) ? next : *c;
    }
    float memoryDelta = finiteOrZero(a->reward - a->memoryGain + memoryBest - a->memoryTaken);
    a->memoryGain = keepFinite(a->memoryGain, a->memoryGain + g.trackingRate * memoryDelta);
    a->memoryNoise = keepFinite(a->memoryNoise, a->memoryNoise + g.trackingRate * (memoryDelta * memoryDelta - a->memoryNoise));
    float memoryStep = finiteOrZero(g.ridgeFloor * memoryDelta / (1.0f + g.ridgeFloor * (float)a->memoryMatched[1u - a->slot]));
    const uint32_t *remembered = a->memoryKey + (1u - a->slot) * OAK_MEMORY_KEYS;
    for (uint32_t k = 0u; k < OAK_MEMORY_KEYS; k++) {
        float *row = memoryRow(a, remembered[k]);
        if (memoryOwner(row) != remembered[k]) {
            for (uint32_t s = 0u; s + 1u < OAK_CELL_STRIDE; s++) row[s] = 0.0f;
            __builtin_memcpy(row + OAK_CELL_STRIDE - 1u, remembered + k, sizeof remembered[k]);
        }
        float next = row[a->action] + memoryStep;
        row[a->action] = isfinite(next) ? next : row[a->action];
    }
}

static inline uint32_t greedyOverride(const float *value, float noise, uint32_t actions, const float *policy, float *behaviour) {
    uint32_t first = 0u;
    for (uint32_t d = 1u; d < actions; d++)
        if (value[d] > value[first]) first = d;
    float gap = value[first], spread = 0.0f;
    for (uint32_t d = 0u; d < actions; d++) {
        float weight = (d == first ? 1.0f : 0.0f) - policy[d];
        gap -= policy[d] * value[d];
        spread += weight * weight;
    }
    if (gap * gap <= noise * spread) return OAK_POLICY_MAX;
    *behaviour = 1.0f;
    return first;
}

static inline uint32_t untriedActions(const float *row, uint32_t actions) {
    uint32_t untried = 0u;
    for (uint32_t d = 0u; d < actions; d++) untried |= (row[OAK_CELL_STRIDE + d] == 0.0f ? 1u : 0u) << d;
    return untried;
}

static inline uint32_t rowOwner(const float *row) {
    uint32_t owner;
    __builtin_memcpy(&owner, row + OAK_QROW - 1u, sizeof owner);
    return owner;
}

static inline uint32_t knownUntried(const float *row, uint32_t tag, uint32_t actions) {
    return rowOwner(row) == tag ? untriedActions(row, actions) : (1u << actions) - 1u;
}

static void claimRow(OakAgent *a, uint32_t key, uint32_t cell, float *row, uint32_t tag) {
    uint32_t actions = a->actionCount, untried = untriedActions(row, actions);
    if (untried != (1u << actions) - 1u) a->qOpen[key] -= (uint32_t)__builtin_popcount(untried);
    for (uint32_t d = 0u; d < actions; d++) {
        if (((untried >> d) & 1u) == 0u) {
            a->qTried[key] -= 1u;
            a->qSum[key] -= row[d];
        }
        row[d] = 0.0f;
        row[OAK_CELL_STRIDE + d] = 0.0f;
        if (key == 0u) a->modelVaried[cell * OAK_CELL_STRIDE + d] = 0u;
    }
    __builtin_memcpy(row + OAK_QROW - 1u, &tag, sizeof tag);
}

static __attribute__((noinline)) void learnTable(OakAgent *a, uint32_t key) {
    uint32_t actions = a->actionCount, act = a->action;
    uint32_t next = a->qCell[a->slot * OAK_KEYS + key], previous = a->qCell[(1u - a->slot) * OAK_KEYS + key];
    uint32_t nextTag = a->qTag[a->slot * OAK_KEYS + key], previousTag = a->qTag[(1u - a->slot) * OAK_KEYS + key];
    float *table = a->qTable + key * OAK_QTABLE * OAK_QROW;
    uint8_t *distance = a->qDistance + key * OAK_QTABLE * OAK_CELL_STRIDE;
    float *row = table + previous * OAK_QROW;
    if (rowOwner(row) != previousTag) claimRow(a, key, previous, row, previousTag);
    uint32_t firstTry = row[OAK_CELL_STRIDE + act] == 0.0f;
    if (firstTry) {
        a->qTried[key] += 1u;
        if (untriedActions(row, actions) == (1u << actions) - 1u) a->qOpen[key] += actions - 1u;
        else a->qOpen[key] -= 1u;
    }
    const float *successor = table + next * OAK_QROW;
    uint32_t successorUntried = knownUntried(successor, nextTag, actions);
    uint32_t hop = successorUntried ? 1u : 255u;
    for (uint32_t d = 0u; d < actions && hop > 1u && a->qOpen[key]; d++) hop = distance[next * OAK_CELL_STRIDE + d] + 1u < hop ? distance[next * OAK_CELL_STRIDE + d] + 1u : hop;
    float bootstrap = -INFINITY;
    for (uint32_t d = 0u; d < actions; d++)
        if (((successorUntried >> d) & 1u) == 0u) bootstrap = fmaxf(bootstrap, successor[d]);
    bootstrap = bootstrap == -INFINITY ? 0.0f : bootstrap;
    float gain = a->qSum[key] / (float)a->qTried[key];
    float delta = finiteOrZero(a->reward - gain + bootstrap - row[act]);
    a->qNoise[key] = keepFinite(a->qNoise[key], a->qNoise[key] + a->g.trackingRate * (delta * delta - a->qNoise[key]));
    float visits = row[OAK_CELL_STRIDE + act] + 1.0f;
    row[OAK_CELL_STRIDE + act] = visits;
    float step = fmaxf(1.0f / visits, a->g.trackingRate), value = row[act] + step * delta;
    if (key == 0u) {
        uint32_t pair = previous * OAK_CELL_STRIDE + act;
        float reward = finiteOrZero(a->reward);
        if (!firstTry && a->modelNext[pair] != next) a->modelVaried[pair] = 1u;
        a->modelNext[pair] = (uint16_t)next;
        a->modelReward[pair] = firstTry ? reward : a->modelReward[pair] + step * (reward - a->modelReward[pair]);
    }
    if (isfinite(value)) {
        a->qSum[key] += value - row[act];
        row[act] = value;
    }
    distance[previous * OAK_CELL_STRIDE + act] = (uint8_t)hop;
}

static inline uint32_t tableAction(const OakAgent *a, uint32_t arm, const uint32_t *words, const float *policy, float *behaviour) {
    uint32_t actions = a->actionCount, key = TABLE_KEY[arm], cell = a->qCell[a->slot * OAK_KEYS + key];
    const float *row = a->qTable + (key * OAK_QTABLE + cell) * OAK_QROW;
    uint32_t unknown = knownUntried(row, a->qTag[a->slot * OAK_KEYS + key], actions);
    if (TABLE_MODE[arm] == TABLE_GREEDY) {
        if (unknown) return OAK_POLICY_MAX;
        uint32_t first = 0u;
        for (uint32_t d = 1u; d < actions; d++)
            if (row[d] > row[first]) first = d;
        float horizon = (float)OAK_FEATURE_TIMESCALE, gap = row[first], spread = 0.0f;
        for (uint32_t d = 0u; d < actions; d++) {
            float weight = (d == first ? 1.0f : 0.0f) - policy[d];
            gap -= policy[d] * row[d];
            spread += weight * weight / fminf(row[OAK_CELL_STRIDE + d], horizon);
        }
        if (gap * gap <= a->qNoise[key] * spread) return OAK_POLICY_MAX;
        *behaviour = 1.0f;
        return first;
    }
    if (unknown == (1u << actions) - 1u) return OAK_POLICY_MAX;
    const uint8_t *distance = a->qDistance + (key * OAK_QTABLE + cell) * OAK_CELL_STRIDE;
    uint32_t nearest = 255u, frontier = 0u;
    for (uint32_t d = 0u; d < actions && a->qOpen[key]; d++) nearest = distance[d] < nearest ? distance[d] : nearest;
    for (uint32_t d = 0u; d < actions && TABLE_MODE[arm] == TABLE_FRONTIER; d++) frontier |= (nearest < 255u && distance[d] == nearest ? 1u : 0u) << d;
    uint32_t pool = unknown ? unknown : frontier ? frontier : (1u << actions) - 1u;
    *behaviour = unknown ? 1.0f / (float)__builtin_popcount(unknown) : 1.0f;
    uint32_t chosen = 0u, chosenWord = 0u;
    float best = -INFINITY;
    for (uint32_t d = 0u; d < actions; d++) {
        if (((pool >> d) & 1u) == 0u) continue;
        float value = unknown ? 0.0f : row[d];
        if (value > best || (value == best && words[d] > chosenWord)) {
            best = value;
            chosenWord = words[d];
            chosen = d;
        }
    }
    return chosen;
}

static __attribute__((noinline)) uint32_t planAction(OakAgent *a, const uint32_t *words) {
    uint32_t actions = a->actionCount, start = a->qCell[a->slot * OAK_KEYS];
    uint32_t startUntried = knownUntried(a->qTable + start * OAK_QROW, a->qTag[a->slot * OAK_KEYS], actions);
    if (a->planIndex < a->planLength && a->planCells[a->planIndex] == start && startUntried != (1u << actions) - 1u) return a->planActions[a->planIndex++];
    a->planLength = 0u;
    a->planIndex = 0u;
    if (startUntried == (1u << actions) - 1u) return OAK_POLICY_MAX;
    float gain = a->qTried[0] ? a->qSum[0] / (float)a->qTried[0] : 0.0f;
    uint32_t generation = ++a->planGeneration, head = 0u, tail = 1u, target = OAK_QTABLE, targetAction = 0u;
    a->planQueue[0] = (uint16_t)start;
    a->planStamp[start] = generation;
    while (head < tail && target == OAK_QTABLE) {
        uint32_t cell = a->planQueue[head++];
        const float *row = a->qTable + cell * OAK_QROW;
        for (uint32_t d = 0u; d < actions && target == OAK_QTABLE; d++)
            if (row[OAK_CELL_STRIDE + d] != 0.0f && !a->modelVaried[cell * OAK_CELL_STRIDE + d] && a->modelReward[cell * OAK_CELL_STRIDE + d] > gain) {
                target = cell;
                targetAction = d;
            }
        uint32_t untried = target == OAK_QTABLE ? untriedActions(row, actions) : 0u, pick = 0u;
        for (uint32_t d = 0u; d < actions; d++)
            if (((untried >> d) & 1u) && (target == OAK_QTABLE || words[d] > pick)) {
                pick = words[d];
                target = cell;
                targetAction = d;
            }
        for (uint32_t d = 0u; d < actions && target == OAK_QTABLE; d++) {
            uint32_t next = a->modelNext[cell * OAK_CELL_STRIDE + d];
            if (row[OAK_CELL_STRIDE + d] == 0.0f || a->modelVaried[cell * OAK_CELL_STRIDE + d] || a->planStamp[next] == generation || tail == OAK_PLAN_BUDGET) continue;
            a->planStamp[next] = generation;
            a->planParent[next] = (uint16_t)cell;
            a->planParentAction[next] = (uint8_t)d;
            a->planQueue[tail++] = (uint16_t)next;
        }
    }
    if (target == OAK_QTABLE) return OAK_POLICY_MAX;
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
        u4 rw1 = actions > 4u ? philox4x32(seed, RNG_STREAM_ACT, step, 1u, 0u, 0u) : rw;
        uint32_t words[8] = { rw.x, rw.y, rw.z, rw.w, rw1.x, rw1.y, rw1.z, rw1.w };
        float logit[OAK_POLICY_LANES] = { 0.0f }, weight[OAK_POLICY_LANES], score[OAK_POLICY_LANES];
        float logitMax = -FLT_MAX;
        for (uint32_t d = 0u; d < actions; d++) {
            logit[d] = finiteOrZero(a->logit[d]);
            logitMax = fmaxf(logitMax, logit[d]);
        }
        for (uint32_t d = 0u; d < actions; d += 4u) {
            float4 uniform = { u01open(words[d]), u01open(words[d + 1u]), u01open(words[d + 2u]), u01open(words[d + 3u]) };
            uniform = __builtin_elementwise_min(uniform, splat4(0x1.fffffep-1f));
            oak_store4(weight + d, oak_exp4(oak_load4(logit + d) - splat4(logitMax)));
            oak_store4(score + d, oak_load4(logit + d) - oak_log4(-oak_log4(uniform)));
        }
        float total = 0.0f;
        for (uint32_t d = 0u; d < actions; d++) total += weight[d];
        uint32_t sampled = 0u;
        float bestScore = -FLT_MAX;
        for (uint32_t d = 0u; d < actions; d++)
            if (score[d] > bestScore) {
                bestScore = score[d];
                sampled = d;
            }
        float behaviour = weight[sampled] / total;
        if (a->activeHead >= OAK_HEADS) {
            float policy[OAK_POLICY_MAX];
            for (uint32_t d = 0u; d < actions; d++) policy[d] = weight[d] / total;
            uint32_t arm = a->activeHead - OAK_HEADS, chosen;
            if (TABLE_MODE[arm] == TABLE_PLAN) {
                chosen = planAction(a, words);
                behaviour = chosen < actions ? 1.0f : behaviour;
            } else if (TABLE_MODE[arm] == TABLE_CELLQ) {
                chosen = greedyOverride(a->qCellValue, a->qCellNoise, actions, policy, &behaviour);
                if (chosen == OAK_POLICY_MAX && a->memoryMatched[a->slot] > 0u) chosen = greedyOverride(a->memoryValue, a->memoryNoise, actions, policy, &behaviour);
            }
            else chosen = tableAction(a, arm, words, policy, &behaviour);
            sampled = chosen < actions ? chosen : sampled;
        }
        a->qCellTaken = a->qCellValue[sampled];
        a->memoryTaken = a->memoryValue[sampled];
        for (uint32_t d = 0u; d < actions; d++) ctrl[d] = d == 0u ? (float)sampled : 0.0f;
        a->action = sampled;
        float chosen = weight[sampled] / total;
        float ratio = a->activeHead >= OAK_HEADS ? chosen / behaviour : 1.0f;
        for (uint32_t d = 0u; d < actions; d++) {
            float credit = (d == sampled ? 1.0f : 0.0f) - weight[d] / total;
            a->actionSignal[d] = credit;
            a->actorCreditStaging[d] = ratio * (HEAD_NATURAL[a->learner] ? finiteOrZero(((d == sampled ? 1.0f / chosen : 0.0f) - 1.0f) / (float)actions) : credit);
        }
        return;
    }
    for (uint32_t d = 0u; d < actions; d++) {
        uint32_t pair = d >> 1u;
        u4 r = philox4x32(seed, RNG_STREAM_ACT, step, pair >> 1u, 0u, 0u);
        uint32_t w0 = (pair & 1u) ? r.z : r.x;
        uint32_t w1 = (pair & 1u) ? r.w : r.y;
        float radius = sqrtf(-2.0f * oak_log(u01open(w0)));
        float sine, cosine;
        oak_sincos(2.0f * OAK_PI * u01(w1), &sine, &cosine);
        float standardNormal = ((d & 1u) == 0u) ? radius * cosine : radius * sine;
        float mu = oak_tanh(finiteOrZero(a->logit[d]));
        float sample = finiteOrZero(mu + standardNormal / (float)OAK_ACTION_PARAMS);
        float clipped = fmaxf(-1.0f, fminf(1.0f, sample));
        ctrl[d] = clipped;
        a->actionSignal[d] = clipped;
        a->actionSignal[actions + d] = 0.0f;
        a->actorCreditStaging[d] = (sample - mu) * (1.0f - mu * mu);
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
    uint32_t previousTuples = a->stepCount > 0u ? OAK_TUPLES + 2u * a->observationCount : 0u, policyTuples = a->stepCount == 0u ? 0u : a->box ? OAK_TUPLES : previousTuples;
    float alpha = g.ridgeFloor;
    float valueStep = finiteOrZero(a->td / (1.0f + alpha * (oak_sum4(e0 + e1) + (float)previousTuples)));
    for (uint32_t t = 0u; t < previousTuples; t++) {
        float *c = a->cellValue + previousCells[t];
        float next = c[0] + alpha * valueStep;
        c[0] = isfinite(next) ? next : c[0];
    }
    float4 valueScale = splat4(alpha * valueStep);
    for (uint32_t k = 0u; k < count; k += 4u) {
        float4 w = oak_load4(a->valueWeight + k), next = __builtin_elementwise_fma(oak_load4(a->valueTrace + k), valueScale, w);
        oak_store4(a->valueWeight + k, select4(oak_finite4(next), next, w));
    }
    float *policy = headPolicy(a);
    float change[OAK_POLICY_MAX];
    float changeSquare = 0.0f;
    for (uint32_t d = 0u; d < actions; d++) {
        float *z = a->policyTrace + d * OAK_FEATURE_MAX;
        float4 credit = splat4(a->actorCredit[d]), c0 = splat4(0.0f), c1 = c0;
        for (uint32_t k = 0u; k < count; k += 8u) {
            float4 x0 = oak_load4(x + k), x1 = oak_load4(x + k + 4u);
            float4 z0 = __builtin_elementwise_fma(oak_load4(z + k), decay, credit * x0);
            float4 z1 = __builtin_elementwise_fma(oak_load4(z + k + 4u), decay, credit * x1);
            oak_store4(z + k, z0);
            oak_store4(z + k + 4u, z1);
            c0 = __builtin_elementwise_fma(z0, x0, c0);
            c1 = __builtin_elementwise_fma(z1, x1, c1);
        }
        change[d] = oak_sum4(c0 + c1);
        change[d] = fmaf((float)policyTuples, a->actorCredit[d], change[d]);
        changeSquare = fmaf(change[d], change[d], changeSquare);
    }
    float actorStep = a->actorStep;
    float policyStep = finiteOrZero(actorStep / fmaxf(1.0f, fabsf(actorStep) * sqrtf(changeSquare)));
    float4 policyScale = splat4(policyStep);
    float4 creditLow = { a->actorCredit[0], a->actorCredit[1], a->actorCredit[2], a->actorCredit[3] };
    float4 creditHigh = { a->actorCredit[4], a->actorCredit[5], 0.0f, 0.0f };
    float4 stepLow = creditLow * policyStep, stepHigh = creditHigh * policyStep;
    for (uint32_t t = 0u; t < policyTuples; t++) {
        float *c = headCell(a, previousCells[t]);
        float4 low = oak_load4(c), high = oak_load4(c + 4u), nextLow = low + stepLow, nextHigh = high + stepHigh;
        oak_store4(c, select4(oak_finite4(nextLow), nextLow, low));
        oak_store4(c + 4u, select4(oak_finite4(nextHigh), nextHigh, high));
    }
    for (uint32_t d = 0u; d < actions; d++) {
        float *row = policy + d * OAK_FEATURE_MAX;
        const float *z = a->policyTrace + d * OAK_FEATURE_MAX;
        for (uint32_t k = 0u; k < count; k += 4u) {
            float4 w = oak_load4(row + k), next = __builtin_elementwise_fma(oak_load4(z + k), policyScale, w);
            oak_store4(row + k, select4(oak_finite4(next), next, w));
        }
    }
    float actor = a->advantage, critic = a->tdNormalized * oak_rsqrt(a->tdSquare);
    const float *unit = x + inputs, *gathered = a->unitInput + (1u - a->slot) * OAK_UNITS * OAK_FANIN;
    for (uint32_t j = 0u; j < OAK_UNITS; j += 4u) {
        float4 credit = splat4(0.0f);
        for (uint32_t d = 0u; d < actions; d++) credit = __builtin_elementwise_fma(splat4(a->actorCredit[d]), oak_load4(policy + d * OAK_FEATURE_MAX + inputs + j), credit);
        float4 activation = oak_load4(unit + j);
        float4 raw = (splat4(critic) * oak_load4(a->valueWeight + inputs + j) + splat4(actor) * credit) * (splat4(1.0f) - activation * activation) * splat4(g.trackingRate) / oak_load4(a->unitEnergy + (1u - a->slot) * OAK_UNITS + j);
        float4 signals = select4(oak_finite4(raw), raw, splat4(0.0f));
        for (uint32_t u = 0u; u < 4u; u++) {
            float *weight = a->projection + (j + u) * OAK_FANIN;
            const float *in = gathered + (j + u) * OAK_FANIN;
            float4 scaled = splat4(signals[u]);
            for (uint32_t k = 0u; k < OAK_FANIN; k += 4u) {
                float4 w = oak_load4(weight + k), next = __builtin_elementwise_fma(scaled, oak_load4(in + k), w);
                oak_store4(weight + k, select4(oak_finite4(next), next, w));
            }
        }
    }
}

uint32_t oak_stateWords(void) { return (uint32_t)(sizeof(OakAgent) / sizeof(uint32_t)); }

void oak_init(uint32_t *state, uint32_t lane, uint32_t env, uint32_t seed, const float *obs) {
    (void)lane; (void)obs;
    OakAgent *a = (OakAgent *)state;
    for (uint32_t w = 0u; w < oak_stateWords(); w++) state[w] = 0u;
    uint32_t observationCount = env_obs(env);
    OakGeometry g = oak_geometry(observationCount);
    a->g = g;
    a->betaFloor = 2.0f * oak_log(g.trackingRate);
    a->observationCount = observationCount;
    a->actionCount = env_actions(env);
    a->box = env_box(env);
    uint32_t direct = OAK_PROJECTED_COUNT(observationCount), sources = direct + OAK_TRACE_LEVELS * observationCount;
    float fanIn = oak_rsqrt((float)OAK_FANIN);
    for (uint32_t n = 0u; n < OAK_UNITS * OAK_FANIN; n += 4u) {
        u4 r0 = philox4x32(seed, RNG_STREAM_ACT, n, 8u, 0u, 0u), r1 = philox4x32(seed, RNG_STREAM_ACT, n, 9u, 0u, 0u), r2 = philox4x32(seed, RNG_STREAM_ACT, n, 10u, 0u, 0u);
        float4 normal = normalFrom4(r0, r1) * fanIn;
        uint32_t pick[4] = { r2.x, r2.y, r2.z, r2.w };
        for (uint32_t r = 0u; r < 4u; r++) {
            uint32_t choice = (uint32_t)(((uint64_t)pick[r] * sources) >> 32u);
            a->unitSource[n + r] = choice < direct ? choice : g.projected + (choice - direct);
            a->projection[n + r] = normal[r];
        }
    }
    uint32_t hash = seed * 2654435761u + 977u;
    uint32_t action = hash % OAK_POLICY_MAX;
    float sign = ((hash >> 16u) & 1u) == 0u ? 1.0f : -1.0f;
    for (uint32_t h = 0u; h < OAK_HEADS; h++) a->policy[(h * OAK_POLICY_MAX + action) * OAK_FEATURE_MAX] = sign * (1.0f / (float)OAK_ACTION_PARAMS);
    float logAlpha = oak_log(g.ridgeFloor);
    for (uint32_t o = 0u; o <= observationCount; o++) {
        a->channelAlpha[o] = g.ridgeFloor;
        a->channelBeta[o] = logAlpha;
    }
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
        for (uint32_t d = 0u; d < a->actionCount; d++) strength += fabsf(a->control[o * OAK_POLICY_LANES + d]);
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
        for (uint32_t k = 0u; k < OAK_POLICY_MAX * OAK_FEATURE_MAX; k++) a->policyTrace[k] = 0.0f;
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

static uint32_t oak_nonfiniteRun(const float *p, uint32_t count, uint32_t stride, uint32_t skip) {
    uint32_t bad = 0u;
    for (uint32_t i = 0u; i < count; i++) bad += i % stride != skip && !isfinite(p[i]);
    return bad;
}

uint32_t oak_nonfinite(const uint32_t *state) {
    const OakAgent *a = (const OakAgent *)state;
    const float scalars[] = { a->blockReward, a->td, a->tdNormalized, a->tdSquare, a->advantage, a->advantageCenter, a->averageReward, a->actorStep, a->betaFloor,
                              a->rewardMean, a->rewardVariance, a->aspiration, a->rewardSquare, a->rewardMagnitude, a->qCellTaken, a->qCellGain, a->qCellNoise,
                              a->memoryTaken, a->memoryGain, a->memoryNoise };
    uint32_t bad = oak_nonfiniteRun(scalars, sizeof scalars / sizeof scalars[0], 1u, 1u);
#define OAK_NONFINITE(field) bad += oak_nonfiniteRun((const float *)(const void *)a->field, (uint32_t)(sizeof a->field / (sizeof(float))), 1u, 1u)
    OAK_NONFINITE(feature);
    OAK_NONFINITE(valueWeight);
    OAK_NONFINITE(valueTrace);
    OAK_NONFINITE(policy);
    OAK_NONFINITE(policyTrace);
    OAK_NONFINITE(projection);
    OAK_NONFINITE(unitInput);
    OAK_NONFINITE(unitEnergy);
    OAK_NONFINITE(traceLevel);
    OAK_NONFINITE(traceMean);
    OAK_NONFINITE(traceSecond);
    OAK_NONFINITE(settledMean);
    OAK_NONFINITE(settledSecond);
    OAK_NONFINITE(cellValue);
    OAK_NONFINITE(cellPolicy);
    OAK_NONFINITE(headMean);
    OAK_NONFINITE(headSquare);
    OAK_NONFINITE(headWeight);
    OAK_NONFINITE(control);
    OAK_NONFINITE(actorMeta);
    OAK_NONFINITE(channelMean);
    OAK_NONFINITE(channelSecond);
    OAK_NONFINITE(channelAlpha);
    OAK_NONFINITE(channelBeta);
    OAK_NONFINITE(channelTrace);
    OAK_NONFINITE(channelEnergy);
    OAK_NONFINITE(actionSignal);
    OAK_NONFINITE(actorCredit);
    OAK_NONFINITE(actorCreditStaging);
    OAK_NONFINITE(logit);
    OAK_NONFINITE(qSum);
    OAK_NONFINITE(qNoise);
    OAK_NONFINITE(modelReward);
    OAK_NONFINITE(qCellWeight);
    OAK_NONFINITE(qCellValue);
    OAK_NONFINITE(memoryValue);
#undef OAK_NONFINITE
    bad += oak_nonfiniteRun(a->qTable, (uint32_t)(sizeof a->qTable / sizeof(float)), OAK_QROW, OAK_QROW - 1u);
    bad += oak_nonfiniteRun(a->memoryQ, (uint32_t)(sizeof a->memoryQ / sizeof(float)), OAK_CELL_STRIDE, OAK_CELL_STRIDE - 1u);
    return bad;
}
