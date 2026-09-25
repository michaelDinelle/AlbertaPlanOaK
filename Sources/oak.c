#include "geometry.h"
#include "policy.h"
#include <arm_neon.h>

#define FEATURE_PRODUCT 0u
#define FEATURE_NORMALIZED_SUM 1u
#define FEATURE_OBSERVATION_TIMESCALE 3u
#define OPERAND_BITS 14u
#define OPERAND_MASK ((1u << OPERAND_BITS) - 1u)
#define OPERAND_PRODUCT (1u << 28u)
#define OPERAND_SUM (1u << 29u)
#define OPERAND_TIMESCALE (1u << 30u)

#define AT(i) ((i) * OAK_LANES + lane)

_Static_assert(OAK_LANES == 1u, "");
_Static_assert(OAK_POLICY_MAX == 6u, "");
_Static_assert(OAK_FEATURE_CAPACITY + OAK_TRACE_CAPACITY <= (1u << OPERAND_BITS), "");

typedef float float2 __attribute__((ext_vector_type(2)));
typedef uint32_t lane2 __attribute__((ext_vector_type(2)));

static inline uint32_t oak_ctz(uint32_t x) { return (uint32_t)__builtin_ctz(x); }
static inline float4 oak_max4(float4 x, float4 y) { return __builtin_elementwise_max(x, y); }
static inline float4 oak_load4(const float *p) { float4 v; __builtin_memcpy(&v, p, sizeof v); return v; }
static inline void oak_store4(float *p, float4 v) { __builtin_memcpy(p, &v, sizeof v); }
static inline float2 oak_load2(const float *p) { float2 v; __builtin_memcpy(&v, p, sizeof v); return v; }
static inline void oak_store2(float *p, float2 v) { __builtin_memcpy(p, &v, sizeof v); }
static inline lane2 oak_finite2(float2 v) { return (lane2)(((lane2)v & 0x7fffffffu) < 0x7f800000u); }
static inline float2 select2(lane2 mask, float2 yes, float2 no) { return (float2)(((lane2)yes & mask) | ((lane2)no & ~mask)); }
static inline lane4 oak_finite4(float4 v) { return (lane4)(((lane4)v & 0x7fffffffu) < 0x7f800000u); }
static inline float4 oak_finiteOrZero4(float4 v) { return select4(oak_finite4(v), v, splat4(0.0f)); }
static inline lane4 oak_loadLanes(const uint32_t *p) { lane4 v; __builtin_memcpy(&v, p, sizeof v); return v; }
static inline void oak_storeLanes(uint32_t *p, lane4 v) { __builtin_memcpy(p, &v, sizeof v); }

typedef struct {
    float valueWeight[OAK_FEATURE_CAPACITY * OAK_LANES];
    float policy[OAK_POLICY_BANK * OAK_LANES];
    float observation[OAK_OBSERVATION_CAPACITY * OAK_LANES];
    float observationMean[OAK_OBSERVATION_CAPACITY * OAK_LANES];
    float observationSecondMoment[OAK_OBSERVATION_CAPACITY * OAK_LANES];
    float observationMeta[4u * OAK_OBSERVATION_CAPACITY * OAK_LANES];
    float features[OAK_FEATURE_STATES * OAK_LANES];
    float policyMean[OAK_ENV_ACT_MAX * OAK_LANES];
    float actionSignal[OAK_ACTION_PARAMS * OAK_LANES];
    float actorCredit[OAK_POLICY_MAX * OAK_LANES];
    float actorCreditStaging[OAK_POLICY_MAX * OAK_LANES];
    float logit[OAK_POLICY_MAX * OAK_LANES];
    float policyCoef[OAK_POLICY_MAX * OAK_LANES];
    float reward[OAK_LANES], td[OAK_LANES], advantage[OAK_LANES], advantageCenter[OAK_LANES];
    float actorMeta[4u * OAK_LANES];
    float featureUtility[OAK_FEATURE_CAPACITY * OAK_LANES];
    float featureMoments[2u * OAK_FEATURE_CAPACITY * OAK_LANES];
    float valueMeta[4u * OAK_FEATURE_CAPACITY * OAK_LANES];
    float observationTimescale[OAK_TRACE_CAPACITY];
    float rewardCenter[OAK_LANES], valueScale[OAK_LANES], previousFeatureNormSq[OAK_LANES], averageReward[OAK_LANES];
    float sampleScratch[OAK_ENV_ACT_MAX * OAK_LANES];
    float nurseryStats[4u * OAK_FEATURE_CAPACITY * OAK_LANES];
    float nurseryScore[OAK_FEATURE_CAPACITY * OAK_LANES];
    float replaceScore[3u * OAK_FEATURE_CAPACITY * OAK_LANES];
    uint32_t replaceEligible[(OAK_FEATURE_CAPACITY / 32u) * OAK_LANES];
    float replaceFloor[OAK_FEATURE_CAPACITY / 32u];
    uint32_t actorAge[OAK_LANES], replaced[OAK_LANES], replacedFeature[OAK_LANES];
    uint32_t activeFeature[OAK_ACTIVE_STATES * OAK_LANES];
    uint32_t featureOperand[OAK_FEATURE_CAPACITY], featureParent[OAK_FEATURE_CAPACITY], featureAge[OAK_FEATURE_CAPACITY];
    uint32_t nurseryOperand[OAK_FEATURE_CAPACITY], nurseryAge[OAK_FEATURE_CAPACITY];
    uint32_t nurseryParentA[OAK_FEATURE_CAPACITY], nurseryParentB[OAK_FEATURE_CAPACITY], nurseryKind[OAK_FEATURE_CAPACITY];
    float scratchValue[OAK_FEATURE_CAPACITY * OAK_LANES];
    float scratchActive[7u * OAK_ACTIVE_CAPACITY * OAK_LANES];
    float betaFloor[OAK_LANES];
    OakGeometry g[OAK_LANES];
    uint32_t observationCount[OAK_LANES], envActionCount[OAK_LANES], box[OAK_LANES], stepCount[OAK_LANES], stateSlot[OAK_LANES];
} OakAgent;

static inline float trackingRateFor(uint32_t step, uint32_t cap) {
    return fmaxf(1.0f / ((float)step + 1.0f), 1.0f / (float)(cap > 1u ? cap : 1u));
}

static inline float *policyRow(OakAgent *a, uint32_t lane, uint32_t feature) { return a->policy + feature * OAK_POLICY_MAX * OAK_LANES + lane; }
static inline float *featureValues(OakAgent *a, OakGeometry g, uint32_t slot) { return a->features + slot * (g.featureCapacity + OAK_TRACE_CAPACITY); }
static inline uint32_t *activeFeatures(OakAgent *a, OakGeometry g, uint32_t slot) { return a->activeFeature + slot * g.activeCapacity * OAK_LANES; }
static inline uint32_t perceptSourceCount(OakAgent *a, uint32_t lane) { return a->observationCount[lane] + a->envActionCount[lane] + 1u; }

static inline uint32_t featureTile(float activation) { return (uint32_t)oak_clamp(floorf(activation + 3.5f), 0.0f, 6.0f); }

static inline uint32_t zoomedFeatureTile(float activation, uint32_t zoomLevel) {
    float zoom = (float)(1u << (zoomLevel < 2u ? zoomLevel : 2u));
    return (uint32_t)oak_clamp(floorf(activation * zoom + 3.5f), 0.0f, 6.0f);
}

static inline uint32_t jointFeatureTile(float activationA, float activationB, uint32_t sourceReuse) {
    if (fabsf(activationA) >= 3.5f || fabsf(activationB) >= 3.5f) return 6u;
    activationA += ((sourceReuse & 1u) == 0u) ? (7.0f / 6.0f) : 0.0f;
    uint32_t tileA = (uint32_t)oak_clamp(floorf((activationA + 3.5f) * (3.0f / 7.0f)), 0.0f, 2.0f);
    uint32_t tileB = (uint32_t)oak_clamp(floorf((activationB + 3.5f) * (2.0f / 7.0f)), 0.0f, 1.0f);
    return tileA * 2u + tileB;
}

static inline __attribute__((always_inline)) void metaOptimizeUpdate(const OakAgent *a, uint32_t lane, float *meta, float signal, float x, uint32_t age, float metaEta, float decay) {
    if (!isfinite(signal) || !isfinite(x)) return;
    float oldAlpha = meta[AT(0u)], oldBeta = meta[AT(1u)], oldH = meta[AT(2u)], oldGradEnergy = meta[AT(3u)];
    float grad = signal * x;
    float metaGradient = grad * oldH;
    float inverseAge = 1.0f / ((float)age + 1.0f);
    float gradEnergy = keepFinite(oldGradEnergy, oldGradEnergy + inverseAge * (metaGradient * metaGradient - oldGradEnergy));
    float normalizedMetaGradient = metaGradient * oak_rsqrt(fmaxf(gradEnergy, 0.0f) + inverseAge);
    float newBeta = oldBeta + metaEta * normalizedMetaGradient;
    float beta = keepFinite(oldBeta, oak_clamp(newBeta, a->betaFloor[lane], 0.0f));
    float alpha = keepFinite(oldAlpha, oak_exp(beta));
    float featureCurvature = x * x;
    float hDecay = (1.0f - decay) * fmaxf(0.0f, 1.0f - oldAlpha * featureCurvature);
    meta[AT(0u)] = alpha;
    meta[AT(1u)] = beta;
    meta[AT(2u)] = keepFinite(oldH, hDecay * oldH + oldAlpha * grad);
    meta[AT(3u)] = gradEnergy;
}

static inline __attribute__((always_inline)) void policyDot(OakAgent *a, uint32_t lane, OakGeometry g, const uint32_t *active, const float *features, float *logit) {
    uint32_t policyCount = a->envActionCount[lane];
    for (uint32_t j = 0u; j < policyCount; j++) logit[j] = 0.0f;
    for (uint32_t i = 0u; i < g.activeCapacity; i++) {
        uint32_t index = active[AT(i)];
        float x = features[AT(index)];
        if (x == 0.0f) continue;
        const float *row = policyRow(a, lane, index);
        for (uint32_t j = 0u; j < policyCount; j++) logit[j] = fmaf(row[j * OAK_LANES], x, logit[j]);
    }
}

static __attribute__((noinline)) void predictAndControl(OakAgent *a, uint32_t lane, OakGeometry g) {
    const float *previousFeatures = featureValues(a, g, a->stateSlot[lane]);
    const float *currentFeatures = featureValues(a, g, 1u - a->stateSlot[lane]);
    const uint32_t *previousActive = activeFeatures(a, g, a->stateSlot[lane]);
    const uint32_t *currentActive = activeFeatures(a, g, 1u - a->stateSlot[lane]);
    float previousValue = 0.0f, currentValue = 0.0f;
    float valueStepDenominator = 1.0f, previousFeatureNormSq = 0.0f;
    uint32_t policyCount = a->envActionCount[lane];
    float4 logitLow = splat4(0.0f);
    float2 logitHigh = 0.0f;
    for (uint32_t i = 0u; i < g.activeCapacity; i++) {
        uint32_t previousIndex = previousActive[AT(i)], currentIndex = currentActive[AT(i)];
        float x = previousFeatures[AT(previousIndex)], y = currentFeatures[AT(currentIndex)];
        previousValue = fmaf(a->valueWeight[AT(previousIndex)], x, previousValue);
        currentValue = fmaf(a->valueWeight[AT(currentIndex)], y, currentValue);
        valueStepDenominator += a->valueMeta[AT(previousIndex * 4u)] * (x * x);
        previousFeatureNormSq = fmaf(x, x, previousFeatureNormSq);
        if (y == 0.0f) continue;
        const float *row = policyRow(a, lane, currentIndex);
        logitLow = __builtin_elementwise_fma(oak_load4(row), splat4(y), logitLow);
        logitHigh = __builtin_elementwise_fma(oak_load2(row + 4), (float2)y, logitHigh);
    }
    float logit[OAK_POLICY_MAX];
    oak_store4(logit, logitLow);
    oak_store2(logit + 4, logitHigh);
    for (uint32_t j = 0u; j < policyCount; j++) a->logit[AT(j)] = logit[j];
    a->valueScale[lane] = (isfinite(valueStepDenominator) && valueStepDenominator > 0.0f) ? (1.0f / valueStepDenominator) : 0.0f;
    a->previousFeatureNormSq[lane] = previousFeatureNormSq;
    float gauge = a->rewardCenter[lane];
    float previousDifferentialValue = previousValue - gauge;
    float currentDifferentialValue = currentValue - gauge;
    float averageRewardOld = a->averageReward[lane];
    float centeredReward = a->reward[lane] - averageRewardOld;
    float delta = centeredReward + currentDifferentialValue - previousDifferentialValue;
    delta = finiteOrZero(delta);
    a->td[lane] = delta;
    float averageRewardNew = averageRewardOld + g.trackingRate * delta;
    a->averageReward[lane] = isfinite(averageRewardNew) ? averageRewardNew : averageRewardOld;

    float advantageCenterOld = a->advantageCenter[lane];
    float advantageError = a->td[lane] - advantageCenterOld;
    a->advantage[lane] = finiteOrZero(advantageError);
    a->advantageCenter[lane] = keepFinite(advantageCenterOld, advantageCenterOld + g.trackingRate * advantageError);

    float gaugeRate = trackingRateFor(a->stepCount[lane], g.featureCapacity);
    float gaugeNew = a->rewardCenter[lane] + gaugeRate * (currentValue - a->rewardCenter[lane]);
    a->rewardCenter[lane] = keepFinite(a->rewardCenter[lane], gaugeNew);
}

static inline uint32_t operandWord(OakGeometry g, uint32_t parentA, uint32_t parentB, uint32_t kind) {
    if (kind == FEATURE_OBSERVATION_TIMESCALE) return (g.featureCapacity + parentA % g.traceChannels) | (parentB << OPERAND_BITS) | OPERAND_TIMESCALE;
    return parentA | (parentB << OPERAND_BITS) | (kind == FEATURE_PRODUCT ? OPERAND_PRODUCT : kind == FEATURE_NORMALIZED_SUM ? OPERAND_SUM : 0u);
}

static inline float4 constructedValues(const float *features, const uint32_t *operand) {
    lane4 word = oak_loadLanes(operand);
    float4 x = { features[operand[0] & OPERAND_MASK], features[operand[1] & OPERAND_MASK], features[operand[2] & OPERAND_MASK], features[operand[3] & OPERAND_MASK] };
    float4 y = { features[(operand[0] >> OPERAND_BITS) & OPERAND_MASK], features[(operand[1] >> OPERAND_BITS) & OPERAND_MASK],
                 features[(operand[2] >> OPERAND_BITS) & OPERAND_MASK], features[(operand[3] >> OPERAND_BITS) & OPERAND_MASK] };
    float4 product = x * y, sum = (x + y) * 0.7071067811865475f, gap = x * (y - x);
    return select4((lane4)((word & OPERAND_TIMESCALE) != 0u), x,
                   select4((lane4)((word & OPERAND_PRODUCT) != 0u), product, select4((lane4)((word & OPERAND_SUM) != 0u), sum, gap)));
}

static inline float4 freshRates(lane4 age) { return 1.0f / (__builtin_convertvector(age, float4) + 1.0f); }

static __attribute__((noinline)) void constructFeatures(OakAgent *a, uint32_t lane, OakGeometry g) {
    float standardized[OAK_OBSERVATION_CAPACITY];
    uint32_t observationCount = a->observationCount[lane];
    uint32_t metaAge = a->stepCount[lane] < g.featureCapacity ? a->stepCount[lane] : g.featureCapacity;
    float inverseAge = 1.0f / ((float)metaAge + 1.0f);
    float floorRate = 1.0f / ((float)a->stepCount[lane] + 1.0f);
    float metaRate = g.metaRate, decay = g.trackingRate, betaFloor = a->betaFloor[lane], ridgeFloor = g.ridgeFloor;
    float *alpha = a->observationMeta, *beta = a->observationMeta + OAK_OBSERVATION_CAPACITY * OAK_LANES;
    float *trace = a->observationMeta + 2u * OAK_OBSERVATION_CAPACITY * OAK_LANES, *energy = a->observationMeta + 3u * OAK_OBSERVATION_CAPACITY * OAK_LANES;
    for (uint32_t o = 0u; o < observationCount; o++) {
        float x = a->observation[AT(o)];
        float observationMean = a->observationMean[AT(o)];
        int finite = isfinite(x);
        float signal = x - observationMean;
        int skip = !finite || !isfinite(signal);
        float oldAlpha = alpha[AT(o)], oldBeta = beta[AT(o)], oldH = trace[AT(o)], oldGradEnergy = energy[AT(o)];
        float metaGradient = signal * oldH;
        float gradEnergy = keepFinite(oldGradEnergy, oldGradEnergy + inverseAge * (metaGradient * metaGradient - oldGradEnergy));
        float normalizedMetaGradient = metaGradient * oak_rsqrt(fmaxf(gradEnergy, 0.0f) + inverseAge);
        float newBeta = oldBeta + metaRate * normalizedMetaGradient;
        float nextBeta = keepFinite(oldBeta, oak_clamp(newBeta, betaFloor, 0.0f));
        float nextAlpha = keepFinite(oldAlpha, oak_exp(nextBeta));
        float hDecay = (1.0f - decay) * fmaxf(0.0f, 1.0f - oldAlpha * (1.0f * 1.0f));
        float nextH = keepFinite(oldH, hDecay * oldH + oldAlpha * signal);
        float alphaNow = skip ? oldAlpha : nextAlpha;
        alpha[AT(o)] = alphaNow;
        beta[AT(o)] = skip ? oldBeta : nextBeta;
        trace[AT(o)] = skip ? oldH : nextH;
        energy[AT(o)] = skip ? oldGradEnergy : gradEnergy;
        float obsRate = fmaxf(floorRate, alphaNow);
        float secondMoment = a->observationSecondMoment[AT(o)];
        float meanNext = finite ? observationMean + obsRate * (x - observationMean) : observationMean;
        float secondNext = finite ? secondMoment + obsRate * (x * x - secondMoment) : secondMoment;
        a->observationMean[AT(o)] = meanNext;
        a->observationSecondMoment[AT(o)] = secondNext;
        float var = fmaxf(secondNext - meanNext * meanNext, 0.0f);
        standardized[o] = (x - meanNext) * oak_rsqrt(var + ridgeFloor);
    }
    for (uint32_t m = 1u; m <= OAK_TRACE_LEVELS; m++) {
        float rate = 1.0f / (float)(1u << m);
        float *level = a->observationTimescale + (m - 1u) * observationCount * OAK_LANES;
        for (uint32_t o = 0u; o < observationCount; o++) {
            float next = level[AT(o)] + rate * (standardized[o] - level[AT(o)]);
            level[AT(o)] = isfinite(a->observation[AT(o)]) ? next : level[AT(o)];
        }
    }
    for (uint32_t j = 0u; j < OAK_POLICY_MAX; j++) a->actorCredit[AT(j)] = a->actorCreditStaging[AT(j)];
    __builtin_memcpy(featureValues(a, g, a->stateSlot[lane]) + g.featureCapacity, a->observationTimescale, g.traceChannels * sizeof(float));

    const float *previousFeatures = featureValues(a, g, a->stateSlot[lane]);
    float *currentFeatures = featureValues(a, g, 1u - a->stateSlot[lane]);
    uint32_t *currentActive = activeFeatures(a, g, 1u - a->stateSlot[lane]);
    uint32_t percepts = g.percepts, actionParams = OAK_ACTION_PARAMS, traceChannels = g.traceChannels;
    for (uint32_t k = percepts; k < g.activeCapacity; k++) currentFeatures[AT(currentActive[AT(k)])] = 0.0f;

    for (uint32_t i = 0u; i < percepts; i++) {
        float x;
        if (i < observationCount) {
            x = standardized[i];
        } else if (i < observationCount + actionParams) {
            x = a->actionSignal[AT(i - observationCount)];
        } else if (i == observationCount + actionParams) {
            x = a->reward[lane];
        } else if (i == observationCount + actionParams + 1u) {
            x = 1.0f;
        } else {
            uint32_t j = i - (observationCount + actionParams + 2u);
            x = (j < traceChannels) ? a->observationTimescale[AT(j)] : 0.0f;
        }
        x = finiteOrZero(x);
        currentFeatures[AT(i)] = x;
        currentActive[AT(i)] = i;
        if (a->featureAge[i] < 0xffffffffu) a->featureAge[i] += 1u;
    }

    uint32_t sourceCount = perceptSourceCount(a, lane), featureCapacity = g.featureCapacity, featureTimescale = g.featureTimescale;
    float *constructed = a->scratchValue;
    float *moment0 = a->featureMoments, *moment1 = a->featureMoments + OAK_FEATURE_CAPACITY;
    float4 settledRate = splat4(g.trackingRate);
    for (uint32_t i = percepts; i < featureCapacity; i += 4u) {
        float4 value = constructedValues(previousFeatures, a->featureOperand + i);
        lane4 age = oak_loadLanes(a->featureAge + i);
        float4 rate = select4((lane4)(age < featureTimescale), freshRates(age), settledRate);
        float4 m0 = oak_load4(moment0 + i), m1 = oak_load4(moment1 + i);
        oak_store4(constructed + i, value);
        oak_store4(moment0 + i, m0 + rate * (value - m0));
        oak_store4(moment1 + i, m1 + rate * (value * value - m1));
    }
    uint32_t directSources = sourceCount - 1u, tileFallback = observationCount + actionParams;
    uint32_t ridgeAgeLimit = 2u * percepts;
    uint32_t sourceIndex = 0u, occupancy = 0u;
    uint32_t *winners = currentActive + percepts;
    for (uint32_t ai = 0u; ai < percepts; ai++) {
        uint32_t channelA = sourceIndex < directSources ? sourceIndex : tileFallback;
        float activationA = currentFeatures[AT(channelA)];
        uint32_t tile;
        if (occupancy == 0u) {
            tile = featureTile(activationA);
        } else if ((ai & 1u) == 0u) {
            tile = zoomedFeatureTile(activationA, occupancy);
        } else {
            uint32_t channelB = (sourceIndex + occupancy) % observationCount;
            tile = jointFeatureTile(activationA, currentFeatures[AT(channelB)], occupancy);
        }
        winners[ai] = percepts + ai * OAK_CANDIDATES_PER_SLOT + tile;
        winners[ai + percepts] = percepts + (ai + percepts) * OAK_CANDIDATES_PER_SLOT + zoomedFeatureTile(activationA, occupancy + 1u);
        sourceIndex += 1u;
        if (sourceIndex == sourceCount) { sourceIndex = 0u; occupancy += 1u; }
    }
    float4 ridgeFloors = splat4(ridgeFloor), zero = splat4(0.0f);
    for (uint32_t k = 0u; k < OAK_CONSTRUCTED_PER_PERCEPT * percepts; k += 4u) {
        const uint32_t *w = winners + k;
        lane4 age = { a->featureAge[w[0]], a->featureAge[w[1]], a->featureAge[w[2]], a->featureAge[w[3]] };
        float4 mean = { moment0[w[0]], moment0[w[1]], moment0[w[2]], moment0[w[3]] };
        float4 second = { moment1[w[0]], moment1[w[1]], moment1[w[2]], moment1[w[3]] };
        float4 value = { constructed[w[0]], constructed[w[1]], constructed[w[2]], constructed[w[3]] };
        float4 var = oak_max4(second - mean * mean, zero);
        float4 ridge = select4((lane4)(age < ridgeAgeLimit), freshRates(age), ridgeFloors);
        float4 winning = oak_finiteOrZero4((value - mean) * (1.0f / __builtin_elementwise_sqrt(var + ridge)));
        lane4 aged = age + ((lane4)(age < 0xffffffffu) & 1u);
        currentFeatures[w[0]] = winning.x; currentFeatures[w[1]] = winning.y; currentFeatures[w[2]] = winning.z; currentFeatures[w[3]] = winning.w;
        a->featureAge[w[0]] = aged.x; a->featureAge[w[1]] = aged.y; a->featureAge[w[2]] = aged.z; a->featureAge[w[3]] = aged.w;
    }
}

static __attribute__((noinline)) void actorCoefficients(OakAgent *a, uint32_t lane, OakGeometry g) {
    float advantage = finiteOrZero(a->advantage[lane]);
    metaOptimizeUpdate(a, lane, a->actorMeta, advantage, 1.0f, a->actorAge[lane], g.metaRate, g.trackingRate);
    float actorAlpha = a->actorMeta[AT(0u)] / OAK_ACTOR_ALPHA_DENOM;
    float previousFeatureNormSq = a->previousFeatureNormSq[lane];
    for (uint32_t d = 0u; d < OAK_POLICY_MAX; d++) {
        float gradScale = a->actorCredit[AT(d)];
        float denom = 1.0f + actorAlpha * advantage * advantage * gradScale * gradScale * previousFeatureNormSq;
        float scale = (isfinite(denom) && denom > 0.0f) ? (1.0f / denom) : 0.0f;
        float policyCoef = actorAlpha * advantage * gradScale * scale;
        a->policyCoef[AT(d)] = finiteOrZero(policyCoef);
    }
}

static __attribute__((noinline)) void emitPolicy(OakAgent *a, uint32_t lane, OakGeometry g, int recompute) {
    if (!recompute) return;
    uint32_t policyCount = a->envActionCount[lane];
    float policyLogit[OAK_POLICY_MAX];
    policyDot(a, lane, g, activeFeatures(a, g, a->stateSlot[lane]), featureValues(a, g, a->stateSlot[lane]), policyLogit);
    for (uint32_t j = 0u; j < policyCount; j++) a->logit[AT(j)] = policyLogit[j];
}

static __attribute__((noinline)) void replaceWorstFeature(OakAgent *a, uint32_t lane, OakGeometry g) {
    const float *currentFeatures = featureValues(a, g, 1u - a->stateSlot[lane]);
    uint32_t percepts = g.percepts, featureCapacity = g.featureCapacity;
    uint32_t eligible = 0u;
    float lowestDownstream = INFINITY;
    for (uint32_t w = percepts / 32u; w < featureCapacity / 32u; w++) {
        if (a->replaceEligible[w] == 0u) continue;
        eligible += (uint32_t)oak_popcount(a->replaceEligible[w]);
        lowestDownstream = fminf(lowestDownstream, a->replaceFloor[w]);
    }
    uint32_t haveWorst = 0u, worstFeature = 0u, worstAge = 0u;
    float worstUtility = 0.0f, worstRedundancy = 0.0f, worstDownstream = 0.0f;
    for (uint32_t w = percepts / 32u; w < featureCapacity / 32u; w++) {
        if (a->replaceEligible[w] == 0u || a->replaceFloor[w] != lowestDownstream) continue;
        for (uint32_t bits = a->replaceEligible[w]; bits != 0u; bits &= bits - 1u) {
            uint32_t i = w * 32u + oak_ctz(bits);
            const float *e = a->replaceScore + i * 3u;
            float util = e[0], redundancy = e[1], downstream = e[2];
            if (downstream != lowestDownstream) continue;
            int worse = (haveWorst == 0u) || downstream < worstDownstream ||
                        (downstream == worstDownstream &&
                         (util < worstUtility || (util == worstUtility &&
                          (redundancy > worstRedundancy || (redundancy == worstRedundancy &&
                           a->featureAge[i] > worstAge)))));
            if (worse) {
                haveWorst = 1u;
                worstFeature = i;
                worstAge = a->featureAge[i];
                worstUtility = util;
                worstRedundancy = redundancy;
                worstDownstream = downstream;
            }
        }
    }
    if (a->actorAge[lane] < 0xffffffffu) a->actorAge[lane] += 1u;
    uint32_t worst = (haveWorst != 0u) ? worstFeature : percepts;
    float4 bound0 = -FLT_MAX, bound1 = bound0, bound2 = bound0, bound3 = bound0;
    for (uint32_t n = 0u; n < featureCapacity; n += 16u) {
        bound0 = oak_max4(bound0, oak_load4(a->nurseryScore + AT(n)));
        bound1 = oak_max4(bound1, oak_load4(a->nurseryScore + AT(n + 4u)));
        bound2 = oak_max4(bound2, oak_load4(a->nurseryScore + AT(n + 8u)));
        bound3 = oak_max4(bound3, oak_load4(a->nurseryScore + AT(n + 12u)));
    }
    float4 bound = oak_max4(oak_max4(bound0, bound1), oak_max4(bound2, bound3));
    float bestNurseryBound = fmaxf(fmaxf(bound[0], bound[1]), fmaxf(bound[2], bound[3]));
    int doReplace = 0;
    if (eligible > 1u && bestNurseryBound > -FLT_MAX && bestNurseryBound > worstUtility + worstDownstream) {
        float eligibleUtilSum = 0.0f, eligibleRedundancySum = 0.0f, eligibleDownstreamSum = 0.0f;
        for (uint32_t w = percepts / 32u; w < featureCapacity / 32u; w++) {
            for (uint32_t bits = a->replaceEligible[w]; bits != 0u; bits &= bits - 1u) {
                const float *e = a->replaceScore + (w * 32u + oak_ctz(bits)) * 3u;
                eligibleUtilSum += e[0];
                eligibleRedundancySum += e[1];
                eligibleDownstreamSum += e[2];
            }
        }
        float invEligible = 1.0f / (float)eligible;
        float meanEligibleUtil = eligibleUtilSum * invEligible;
        float meanEligibleRedundancy = eligibleRedundancySum * invEligible;
        float meanEligibleDownstream = eligibleDownstreamSum * invEligible;
        int lowUtility = worstUtility < meanEligibleUtil;
        int highRedundancy = worstRedundancy > meanEligibleRedundancy;
        int lowDownstreamUse = worstDownstream <= meanEligibleDownstream;
        doReplace = lowDownstreamUse && (lowUtility || highRedundancy);
    }
    if (doReplace) {
        uint32_t bestNursery = 0u;
        while (a->nurseryScore[AT(bestNursery)] != bestNurseryBound) bestNursery += 1u;
        float bestA = -FLT_MAX;
        uint32_t seedParentA = 0u;
        for (uint32_t i = 0u; i < featureCapacity; i++) {
            if (i == worst) continue;
            float parentUtility = a->featureUtility[AT(i)];
            if (!isfinite(parentUtility)) parentUtility = -FLT_MAX;
            if (parentUtility > bestA) {
                bestA = parentUtility;
                seedParentA = i;
            }
        }
        if (bestA == -FLT_MAX) seedParentA = 0u;
        float bestB = -FLT_MAX;
        uint32_t seedParentB = 1u;
        for (uint32_t i = 0u; i < featureCapacity; i++) {
            if (i == worst || i == seedParentA) continue;
            float decor = 1.0f / (1.0f + fabsf(currentFeatures[AT(i)] * currentFeatures[AT(seedParentA)]));
            float parentUtility = a->featureUtility[AT(i)] * decor;
            if (!isfinite(parentUtility)) parentUtility = -FLT_MAX;
            if (parentUtility > bestB) {
                bestB = parentUtility;
                seedParentB = i;
            }
        }
        if (bestB == -FLT_MAX) seedParentB = 1u;
        uint32_t kindNext = (a->nurseryKind[bestNursery] + 1u) % 4u;
        uint32_t parentA = a->nurseryParentA[bestNursery] % featureCapacity;
        a->featureOperand[worst] = operandWord(g, parentA, a->nurseryParentB[bestNursery] % featureCapacity, a->nurseryKind[bestNursery] % 4u);
        a->featureParent[worst] = parentA;
        a->featureAge[worst] = a->nurseryAge[bestNursery];
        a->featureMoments[AT(worst)] = a->nurseryStats[AT(bestNursery)];
        a->featureMoments[AT(OAK_FEATURE_CAPACITY + worst)] = a->nurseryStats[AT(OAK_FEATURE_CAPACITY + bestNursery)];
        a->featureUtility[AT(worst)] = a->nurseryStats[AT(2u * OAK_FEATURE_CAPACITY + bestNursery)];
        a->valueWeight[AT(worst)] = 0.0f;
        a->nurseryParentA[bestNursery] = seedParentA;
        a->nurseryParentB[bestNursery] = seedParentB;
        a->nurseryKind[bestNursery] = kindNext;
        a->nurseryAge[bestNursery] = 0u;
        a->nurseryOperand[bestNursery] = operandWord(g, seedParentA, seedParentB, kindNext);
        for (uint32_t k = 0u; k < 4u; k++) a->nurseryStats[AT(k * OAK_FEATURE_CAPACITY + bestNursery)] = 0.0f;
        a->nurseryScore[AT(bestNursery)] = -FLT_MAX;
        float initialStepSize = 1.0f / (float)g.activeCapacity;
        a->valueMeta[AT(worst * 4u + 0u)] = initialStepSize;
        a->valueMeta[AT(worst * 4u + 1u)] = oak_log(initialStepSize);
        a->valueMeta[AT(worst * 4u + 2u)] = 0.0f;
        a->valueMeta[AT(worst * 4u + 3u)] = 0.0f;
        a->replaced[lane] = 1u;
        a->replacedFeature[lane] = worst;
        for (uint32_t d = 0u; d < OAK_POLICY_MAX; d++) policyRow(a, lane, worst)[d * OAK_LANES] = 0.0f;
    } else {
        a->replaced[lane] = 0u;
    }
}

static __attribute__((noinline)) void updateUtilities(OakAgent *a, uint32_t lane, OakGeometry g) {
    uint32_t tPrev = a->stepCount[lane] - 1u;
    const float *previousFeatures = featureValues(a, g, 1u - a->stateSlot[lane]);
    const float *currentFeatures = featureValues(a, g, a->stateSlot[lane]);
    const uint32_t *previousActive = activeFeatures(a, g, 1u - a->stateSlot[lane]);
    uint32_t featureCapacity = g.featureCapacity, activeCapacity = g.activeCapacity;
    float trackingRate = trackingRateFor(tPrev, featureCapacity);
    float valueTdError = a->td[lane];
    float advantage = a->advantage[lane];
    float planningSignal = fabsf(valueTdError);
    float metaRate = g.metaRate, decay = g.trackingRate, betaFloor = a->betaFloor[lane];
    float valueStepScale = a->valueScale[lane];
    uint32_t policyCount = a->envActionCount[lane];
    float4 coefLow = oak_load4(a->policyCoef);
    float2 coefHigh = oak_load2(a->policyCoef + 4);
    lane4 learnLow = (lane4)((lane4){ 0u, 1u, 2u, 3u } < policyCount);
    lane2 learnHigh = (lane2)((lane2){ 4u, 5u } < policyCount);
    float *x = a->scratchActive, *utility = a->scratchActive + OAK_ACTIVE_CAPACITY, *age = a->scratchActive + 2u * OAK_ACTIVE_CAPACITY;
    float *alpha = a->scratchActive + 3u * OAK_ACTIVE_CAPACITY, *beta = a->scratchActive + 4u * OAK_ACTIVE_CAPACITY;
    float *trace = a->scratchActive + 5u * OAK_ACTIVE_CAPACITY, *energy = a->scratchActive + 6u * OAK_ACTIVE_CAPACITY;
    for (uint32_t s = 0u; s < activeCapacity; s++) {
        uint32_t i = previousActive[s];
        float xi = previousFeatures[i];
        float *row = policyRow(a, lane, i);
        float4 w = oak_load4(row);
        float2 wHigh = oak_load2(row + 4);
        if (xi != 0.0f || currentFeatures[i] != 0.0f) {
            float4 next = w + coefLow * xi;
            float2 nextHigh = wHigh + coefHigh * xi;
            w = select4(learnLow & oak_finite4(next), next, w);
            wHigh = select2(learnHigh & oak_finite2(nextHigh), nextHigh, wHigh);
            oak_store4(row, w);
            oak_store2(row + 4, wHigh);
        }
        float4 meta = oak_load4(a->valueMeta + i * 4u);
        float weight = a->valueWeight[i];
        float delta = valueTdError * xi * valueStepScale;
        weight = keepFinite(weight, weight + meta.x * delta);
        a->valueWeight[i] = weight;
        float actorSignal = (i == 0u) ? (fabsf(advantage) + (fabsf(advantage) + 1.0f)) : 0.0f;
        float4 magnitude = __builtin_elementwise_abs(w);
        float2 magnitudeHigh = __builtin_elementwise_abs(wHigh);
        float selectedPolicyLoad = 0.0f;
        selectedPolicyLoad += magnitude.x;
        selectedPolicyLoad += magnitude.y;
        selectedPolicyLoad += magnitude.z;
        selectedPolicyLoad += magnitude.w;
        selectedPolicyLoad += magnitudeHigh.x;
        selectedPolicyLoad += magnitudeHigh.y;
        float load = fabsf(xi) * (fabsf(weight) + selectedPolicyLoad);
        load = finiteOrZero(load);
        float featureUtilitySignal = fabsf(planningSignal * xi) + fabsf(actorSignal * xi) + load;
        utility[s] = a->featureUtility[i] + trackingRate * (featureUtilitySignal - a->featureUtility[i]);
        x[s] = xi;
        age[s] = (float)a->featureAge[i];
        alpha[s] = meta.x; beta[s] = meta.y; trace[s] = meta.z; energy[s] = meta.w;
    }
    for (uint32_t s = 0u; s < activeCapacity; s++) {
        float xi = x[s], oldAlpha = alpha[s], oldBeta = beta[s], oldH = trace[s], oldGradEnergy = energy[s];
        int skip = !isfinite(valueTdError) || !isfinite(xi);
        float grad = valueTdError * xi;
        float metaGradient = grad * oldH;
        float inverseAge = 1.0f / (age[s] + 1.0f);
        float gradEnergy = keepFinite(oldGradEnergy, oldGradEnergy + inverseAge * (metaGradient * metaGradient - oldGradEnergy));
        float normalizedMetaGradient = metaGradient * oak_rsqrt(fmaxf(gradEnergy, 0.0f) + inverseAge);
        float newBeta = oldBeta + metaRate * normalizedMetaGradient;
        float nextBeta = keepFinite(oldBeta, oak_clamp(newBeta, betaFloor, 0.0f));
        float nextAlpha = keepFinite(oldAlpha, oak_exp(nextBeta));
        float hDecay = (1.0f - decay) * fmaxf(0.0f, 1.0f - oldAlpha * (xi * xi));
        float nextH = keepFinite(oldH, hDecay * oldH + oldAlpha * grad);
        alpha[s] = skip ? oldAlpha : nextAlpha;
        beta[s] = skip ? oldBeta : nextBeta;
        trace[s] = skip ? oldH : nextH;
        energy[s] = skip ? oldGradEnergy : gradEnergy;
    }
    for (uint32_t s = 0u; s < activeCapacity; s++) {
        uint32_t i = previousActive[s];
        a->featureUtility[i] = utility[s];
        oak_store4(a->valueMeta + i * 4u, (float4){ alpha[s], beta[s], trace[s], energy[s] });
    }
    uint32_t parity = (tPrev >> 1u) & 1u;
    float *stat0 = a->nurseryStats, *stat1 = a->nurseryStats + OAK_FEATURE_CAPACITY * OAK_LANES;
    float *stat2 = a->nurseryStats + 2u * OAK_FEATURE_CAPACITY * OAK_LANES, *stat3 = a->nurseryStats + 3u * OAK_FEATURE_CAPACITY * OAK_LANES;
    uint32_t featureTimescale = g.featureTimescale, ridgeAgeLimit = 2u * g.percepts;
    float4 settledRate = splat4(decay), ridgeFloor = splat4(g.ridgeFloor), zero = splat4(0.0f), unscored = splat4(-FLT_MAX);
    for (uint32_t block = parity * OAK_CAPACITY; block < featureCapacity; block += 2u * OAK_CAPACITY) {
        for (uint32_t n = block; n < block + OAK_CAPACITY; n += 4u) {
            float4 value = constructedValues(previousFeatures, a->nurseryOperand + n);
            lane4 ageBefore = oak_loadLanes(a->nurseryAge + n);
            float4 fresh = freshRates(ageBefore);
            float4 rate = select4((lane4)(ageBefore < featureTimescale), fresh, settledRate);
            float4 ridge = select4((lane4)(ageBefore < ridgeAgeLimit), fresh, ridgeFloor);
            float4 s0 = oak_load4(stat0 + n), s1 = oak_load4(stat1 + n), s2 = oak_load4(stat2 + n), s3 = oak_load4(stat3 + n);
            s0 = s0 + rate * (value - s0);
            s1 = s1 + rate * (value * value - s1);
            float4 var = oak_max4(s1 - s0 * s0, zero);
            float4 standardizedCandidate = (value - s0) * (1.0f / __builtin_elementwise_sqrt(var + ridge));
            float4 candidateUtilitySignal = __builtin_elementwise_abs(valueTdError * standardizedCandidate);
            lane4 finite = oak_finite4(candidateUtilitySignal);
            s2 = select4(finite, s2 + trackingRate * (candidateUtilitySignal - s2), s2);
            s3 = select4(finite, s3 + trackingRate * (candidateUtilitySignal * candidateUtilitySignal - s3), s3);
            oak_store4(stat0 + n, s0); oak_store4(stat1 + n, s1); oak_store4(stat2 + n, s2); oak_store4(stat3 + n, s3);
            lane4 nurseryAge = ageBefore + ((lane4)(ageBefore < 0xffffffffu) & 1u);
            oak_storeLanes(a->nurseryAge + n, nurseryAge);
            float4 bound = s2 - __builtin_elementwise_sqrt(oak_max4(s3 - s2 * s2, zero) * decay);
            lane4 scored = (lane4)(nurseryAge >= featureCapacity) & oak_finite4(s2) & oak_finite4(bound);
            oak_store4(a->nurseryScore + n, select4(scored, bound, unscored));
        }
    }
    for (uint32_t block = g.percepts + parity * OAK_CAPACITY; block < featureCapacity; block += 2u * OAK_CAPACITY) {
        for (uint32_t w = block / 32u; w < block / 32u + OAK_CAPACITY / 32u; w++) {
            uint32_t bits = 0u;
            float4 lowest = splat4(INFINITY);
            for (uint32_t b = 0u; b < 32u; b += 4u) {
                uint32_t i = w * 32u + b;
                const uint32_t *parentA = a->featureParent + i;
                lane4 eligible = (lane4)(oak_loadLanes(a->featureAge + i) >= featureCapacity);
                float4 parent = { currentFeatures[parentA[0]], currentFeatures[parentA[1]], currentFeatures[parentA[2]], currentFeatures[parentA[3]] };
                float4 util = oak_finiteOrZero4(oak_load4(a->featureUtility + i));
                float4 redundancy = oak_finiteOrZero4(__builtin_elementwise_abs(oak_load4(currentFeatures + i) * parent));
                float4 downstreamUse = oak_finiteOrZero4(__builtin_elementwise_abs(oak_load4(a->valueWeight + i)));
                float32x4x3_t scores = { { (float32x4_t)util, (float32x4_t)redundancy, (float32x4_t)downstreamUse } };
                vst3q_f32(a->replaceScore + i * 3u, scores);
                lowest = __builtin_elementwise_min(lowest, select4(eligible, downstreamUse, splat4(INFINITY)));
                bits |= __builtin_reduce_add(eligible & (lane4){ 1u, 2u, 4u, 8u }) << b;
            }
            a->replaceEligible[w] = bits;
            a->replaceFloor[w] = __builtin_reduce_min(lowest);
        }
    }
}

static __attribute__((noinline)) void sampleDiscreteAction(OakAgent *a, uint32_t lane, uint32_t seed, uint32_t stepIdx, float *ctrl) {
    uint32_t discreteActionCount = a->envActionCount[lane];
    u4 rw = philox4x32(seed, RNG_STREAM_ACT, stepIdx, 0u, 0u, 0u);
    u4 rw1 = discreteActionCount > 4u ? philox4x32(seed, RNG_STREAM_ACT, stepIdx, 1u, 0u, 0u) : rw;
    uint32_t words[8] = { rw.x, rw.y, rw.z, rw.w, rw1.x, rw1.y, rw1.z, rw1.w };
    float behaviorLogit[OAK_POLICY_MAX], behaviorWeight[OAK_POLICY_MAX];
    float behaviorLogitMax = -FLT_MAX;
    for (uint32_t d = 0u; d < discreteActionCount; d++) {
        behaviorLogit[d] = finiteOrZero(a->logit[AT(d)]);
        behaviorLogitMax = fmaxf(behaviorLogitMax, behaviorLogit[d]);
    }
    float behaviorSoftmaxSum = 0.0f;
    for (uint32_t d = 0u; d < discreteActionCount; d++) {
        behaviorWeight[d] = oak_exp(behaviorLogit[d] - behaviorLogitMax);
        behaviorSoftmaxSum += behaviorWeight[d];
    }
    uint32_t sampledAction = 0u;
    float bestGumbelScore = -FLT_MAX;
    for (uint32_t d = 0u; d < discreteActionCount; d++) {
        float uniform = fminf(u01open(words[d]), 0x1.fffffep-1f);
        float gumbelScore = behaviorLogit[d] - oak_log(-oak_log(uniform));
        if (gumbelScore > bestGumbelScore) {
            bestGumbelScore = gumbelScore;
            sampledAction = d;
        }
    }
    for (uint32_t d = 0u; d < discreteActionCount; d++) ctrl[d] = (d == 0u) ? (float)sampledAction : 0.0f;
    for (uint32_t d = 0u; d < discreteActionCount; d++) {
        float credit = ((d == sampledAction) ? 1.0f : 0.0f) - behaviorWeight[d] / behaviorSoftmaxSum;
        a->actionSignal[AT(d)] = credit;
        a->actorCreditStaging[AT(d)] = credit;
    }
}

static __attribute__((noinline)) void sampleAction(OakAgent *a, uint32_t lane, uint32_t seed, uint32_t stepIdx, float *ctrl) {
    if (!a->box[lane]) {
        sampleDiscreteAction(a, lane, seed, stepIdx, ctrl);
        return;
    }
    uint32_t envActionCount = a->envActionCount[lane];
    for (uint32_t d = 0u; d < envActionCount; d++) {
        a->policyMean[AT(d)] = keepFinite(a->policyMean[AT(d)], oak_tanh(a->logit[AT(d)]));
        uint32_t pair = d >> 1u;
        u4 r = philox4x32(seed, RNG_STREAM_ACT, stepIdx, pair >> 1u, 0u, 0u);
        uint32_t w0 = (pair & 1u) ? r.z : r.x;
        uint32_t w1 = (pair & 1u) ? r.w : r.y;
        float uniform1 = u01open(w0);
        float uniform2 = u01(w1);
        float boxMullerRadius = sqrtf(-2.0f * oak_log(uniform1));
        float boxMullerAngle = 2.0f * OAK_PI * uniform2;
        float sine, cosine;
        oak_sincos(boxMullerAngle, &sine, &cosine);
        float standardNormal = ((d & 1u) == 0u) ? (boxMullerRadius * cosine) : (boxMullerRadius * sine);
        float mu = finiteOrZero(a->policyMean[AT(d)]);
        float sample = finiteOrZero(mu + OAK_SAMPLE_SIGMA * standardNormal);
        float clippedAction = fmaxf(-1.0f, fminf(1.0f, sample));
        ctrl[d] = clippedAction;
        a->actionSignal[AT(d)] = clippedAction;
        a->actionSignal[AT(envActionCount + d)] = 0.0f;
        a->sampleScratch[AT(d)] = sample;
    }
    for (uint32_t d = 0u; d < envActionCount; d++) {
        float sample = a->sampleScratch[AT(d)];
        float meanValue = finiteOrZero(a->policyMean[AT(d)]);
        a->actorCreditStaging[AT(d)] = (sample - meanValue) * (1.0f - meanValue * meanValue);
    }
}

uint32_t oak_stateWords(void) { return (uint32_t)(sizeof(OakAgent) / sizeof(uint32_t) / OAK_LANES); }

void oak_init(uint32_t *state, uint32_t lane, uint32_t env, uint32_t seed, const float *obs) {
    OakAgent *a = (OakAgent *)state;
    for (uint32_t w = 0u; w < oak_stateWords(); w++) state[AT(w)] = 0u;
    OakGeometry g = oak_geometry(env_obs(env));
    a->g[lane] = g;
    a->betaFloor[lane] = 2.0f * oak_log(g.trackingRate);
    a->observationCount[lane] = env_obs(env);
    a->envActionCount[lane] = env_actions(env);
    a->box[lane] = env_box(env);
    uint32_t percepts = g.percepts, traceChannels = g.traceChannels;
    float alpha = g.ridgeFloor;
    float logAlpha = oak_log(alpha);
    for (uint32_t feature = 0u; feature < g.featureCapacity; feature++) {
        uint32_t parent = feature % (feature % 4u == 3u ? traceChannels : percepts);
        a->nurseryParentA[feature] = parent;
        a->nurseryParentB[feature] = (feature * 7u + 3u) % percepts;
        a->nurseryKind[feature] = feature % 4u;
        a->nurseryOperand[feature] = operandWord(g, parent, a->nurseryParentB[feature], a->nurseryKind[feature]);
        if (feature >= percepts) {
            uint32_t source = feature - percepts;
            uint32_t parentA = source < traceChannels ? source : source % percepts;
            uint32_t parentB = source < traceChannels ? 0u : (source * 7u + 3u) % percepts;
            uint32_t kind = source < traceChannels ? FEATURE_OBSERVATION_TIMESCALE : source % 3u;
            a->featureOperand[feature] = operandWord(g, parentA, parentB, kind);
            a->featureParent[feature] = parentA;
        }
        a->valueMeta[AT(feature * 4u + 0u)] = alpha;
        a->valueMeta[AT(feature * 4u + 1u)] = logAlpha;
    }
    a->actorMeta[AT(0u)] = 1.0f;
    uint32_t hash = seed * 2654435761u + 977u;
    uint32_t action = hash % OAK_POLICY_MAX;
    float sign = ((hash >> 16u) & 1u) == 0u ? 1.0f : -1.0f;
    policyRow(a, lane, 0u)[action * OAK_LANES] = sign * (1.0f / (float)OAK_ACTION_PARAMS);
    for (uint32_t o = 0u; o < a->observationCount[lane]; o++) {
        a->observationMeta[AT(o)] = alpha;
        a->observationMeta[AT(OAK_OBSERVATION_CAPACITY + o)] = logAlpha;
    }
    for (uint32_t i = 0u; i < a->observationCount[lane]; i++) a->observation[AT(i)] = obs[i];
}

static inline __attribute__((always_inline)) void oakStep(OakAgent *a, uint32_t lane, OakGeometry g, uint32_t seed, uint32_t step, const float *obs, float reward, float *ctrl) {
    for (uint32_t i = 0u; i < a->observationCount[lane]; i++) a->observation[AT(i)] = obs[i];
    a->reward[lane] = reward;
    a->stepCount[lane] = step;
    a->stateSlot[lane] = step & 1u;
    constructFeatures(a, lane, g);
    predictAndControl(a, lane, g);
    replaceWorstFeature(a, lane, g);
    a->stepCount[lane] += 1u;
    a->stateSlot[lane] ^= 1u;
    emitPolicy(a, lane, g, a->replaced[lane] != 0u);
    sampleAction(a, lane, seed, step, ctrl);
    actorCoefficients(a, lane, g);
    updateUtilities(a, lane, g);
}

__attribute__((noinline)) void oak_act(uint32_t *state, uint32_t lane, uint32_t env, uint32_t seed, uint32_t step, const float *obs, float reward, uint32_t done, float *ctrl) {
    (void)env; (void)done;
    OakAgent *a = (OakAgent *)state;
    oakStep(a, lane, a->g[lane], seed, step, obs, reward, ctrl);
}
