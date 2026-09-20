#include "geometry.h"
#include "policy.h"

#define FEATURE_PRODUCT 0u
#define FEATURE_NORMALIZED_SUM 1u
#define FEATURE_OBSERVATION_TIMESCALE 3u

#define AT(i) ((i) * OAK_LANES + lane)

typedef struct {
    float valueWeight[OAK_FEATURE_CAPACITY * OAK_LANES];
    float policy[OAK_POLICY_BANK * OAK_LANES];
    float observation[OAK_OBSERVATION_CAPACITY * OAK_LANES];
    float observationMean[OAK_OBSERVATION_CAPACITY * OAK_LANES];
    float observationSecondMoment[OAK_OBSERVATION_CAPACITY * OAK_LANES];
    float observationMeta[4u * OAK_OBSERVATION_CAPACITY * OAK_LANES];
    float features[OAK_FEATURE_STATES * OAK_LANES];
    float policyMean[OAK_ENV_ACT_MAX * OAK_LANES];
    float policyMeanRaw[OAK_ENV_ACT_MAX * OAK_LANES];
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
    float observationTimescale[OAK_TRACE_LEVELS * OAK_OBSERVATION_CAPACITY * OAK_LANES];
    float rewardCenter[OAK_LANES], valueScale[OAK_LANES], previousFeatureNormSq[OAK_LANES], averageReward[OAK_LANES];
    float sampleScratch[OAK_ENV_ACT_MAX * OAK_LANES];
    float nurseryStats[4u * OAK_FEATURE_CAPACITY * OAK_LANES];
    float nurseryScore[OAK_FEATURE_CAPACITY * OAK_LANES];
    float replaceScore[4u * OAK_FEATURE_CAPACITY * OAK_LANES];
    uint32_t actorAge[OAK_LANES], replaced[OAK_LANES], replacedFeature[OAK_LANES];
    uint32_t activeFeature[OAK_ACTIVE_STATES * OAK_LANES];
    uint32_t featureMeta[4u * OAK_FEATURE_CAPACITY * OAK_LANES];
    uint32_t nurseryMeta[4u * OAK_FEATURE_CAPACITY * OAK_LANES];
    uint32_t traceIndex[OAK_FEATURE_CAPACITY * OAK_LANES];
    float reciprocal[OAK_FEATURE_TIMESCALE_MAX * OAK_LANES];
    float scratchValue[OAK_FEATURE_CAPACITY * OAK_LANES], scratchRate[OAK_FEATURE_CAPACITY * OAK_LANES];
    float scratchActive[7u * OAK_ACTIVE_CAPACITY * OAK_LANES];
    float betaFloor[OAK_LANES];
    OakGeometry g[OAK_LANES];
    uint32_t observationCount[OAK_LANES], envActionCount[OAK_LANES], box[OAK_LANES], stepCount[OAK_LANES], stateSlot[OAK_LANES];
} OakAgent;

static inline float trackingRateAt(device const OakAgent *a, uint32_t lane, uint32_t age) {
    return age < a->g[lane].featureTimescale ? a->reciprocal[AT(age)] : a->g[lane].trackingRate;
}

static inline float ridgeAt(device const OakAgent *a, uint32_t lane, uint32_t age) {
    return age < 2u * a->g[lane].percepts ? a->reciprocal[AT(age)] : a->g[lane].ridgeFloor;
}

static inline float trackingRateFor(uint32_t step, uint32_t cap) {
    return fmaxf(1.0f / ((float)step + 1.0f), 1.0f / (float)(cap > 1u ? cap : 1u));
}

static inline device float *policyRow(device OakAgent *a, uint32_t feature) { return a->policy + feature * OAK_POLICY_MAX * OAK_LANES; }
static inline device float *featureValues(device OakAgent *a, uint32_t lane, uint32_t slot) { return a->features + slot * a->g[lane].featureCapacity * OAK_LANES; }
static inline device uint32_t *activeFeatures(device OakAgent *a, uint32_t lane, uint32_t slot) { return a->activeFeature + slot * a->g[lane].activeCapacity * OAK_LANES; }
static inline uint32_t perceptSourceCount(device OakAgent *a, uint32_t lane) { return a->observationCount[lane] + a->envActionCount[lane] + 1u; }

static inline uint32_t featureTile(float activation) { return (uint32_t)oak_clamp(floorf(activation + 3.5f), 0.0f, 6.0f); }

static inline uint32_t tileSourceIndex(device OakAgent *a, uint32_t lane, uint32_t source) {
    return (source < a->observationCount[lane] + a->envActionCount[lane]) ? source : a->observationCount[lane] + OAK_ACTION_PARAMS;
}

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

static inline __attribute__((always_inline)) void metaOptimizeUpdate(device const OakAgent *a, uint32_t lane, device float *meta, float signal, float x, uint32_t age, float metaEta, float decay) {
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

static void policyDot(device OakAgent *a, uint32_t lane, device const uint32_t *active, device const float *features, thread float *logit) {
    for (uint32_t j = 0u; j < OAK_POLICY_MAX; j++) logit[j] = 0.0f;
    for (uint32_t i = 0u; i < a->g[lane].activeCapacity; i++) {
        uint32_t index = active[AT(i)];
        float x = features[AT(index)];
        if (x == 0.0f) continue;
        device const float *row = policyRow(a, index);
        for (uint32_t j = 0u; j < OAK_POLICY_MAX; j++) logit[j] = fmaf(row[AT(j)], x, logit[j]);
    }
}

static void predictAndControl(device OakAgent *a, uint32_t lane) {
    device const float *previousFeatures = featureValues(a, lane, a->stateSlot[lane]);
    device const float *currentFeatures = featureValues(a, lane, 1u - a->stateSlot[lane]);
    device const uint32_t *previousActive = activeFeatures(a, lane, a->stateSlot[lane]);
    device const uint32_t *currentActive = activeFeatures(a, lane, 1u - a->stateSlot[lane]);
    float previousValue = 0.0f, currentValue = 0.0f;
    float valueStepDenominator = 1.0f, previousFeatureNormSq = 0.0f;
    device float *logit = a->logit;
    for (uint32_t j = 0u; j < OAK_POLICY_MAX; j++) logit[AT(j)] = 0.0f;
    for (uint32_t i = 0u; i < a->g[lane].activeCapacity; i++) {
        uint32_t previousIndex = previousActive[AT(i)], currentIndex = currentActive[AT(i)];
        float x = previousFeatures[AT(previousIndex)], y = currentFeatures[AT(currentIndex)];
        previousValue = fmaf(a->valueWeight[AT(previousIndex)], x, previousValue);
        currentValue = fmaf(a->valueWeight[AT(currentIndex)], y, currentValue);
        valueStepDenominator += a->valueMeta[AT(previousIndex * 4u)] * (x * x);
        previousFeatureNormSq = fmaf(x, x, previousFeatureNormSq);
        if (y == 0.0f) continue;
        device const float *row = policyRow(a, currentIndex);
        for (uint32_t j = 0u; j < OAK_POLICY_MAX; j++) logit[AT(j)] = fmaf(row[AT(j)], y, logit[AT(j)]);
    }
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
    float averageRewardNew = averageRewardOld + a->g[lane].trackingRate * delta;
    a->averageReward[lane] = isfinite(averageRewardNew) ? averageRewardNew : averageRewardOld;

    float advantageCenterOld = a->advantageCenter[lane];
    float advantageError = a->td[lane] - advantageCenterOld;
    a->advantage[lane] = finiteOrZero(advantageError);
    a->advantageCenter[lane] = keepFinite(advantageCenterOld, advantageCenterOld + a->g[lane].trackingRate * advantageError);

    float gaugeRate = trackingRateFor(a->stepCount[lane], a->g[lane].featureCapacity);
    float gaugeNew = a->rewardCenter[lane] + gaugeRate * (currentValue - a->rewardCenter[lane]);
    a->rewardCenter[lane] = keepFinite(a->rewardCenter[lane], gaugeNew);
}

static inline float constructedValue(device const OakAgent *a, uint32_t lane, device const float *features, uint32_t parentA, uint32_t parentB, uint32_t kind) {
    float trace = a->observationTimescale[AT(a->traceIndex[AT(parentA)])];
    float x = features[AT(parentA)], y = features[AT(parentB)];
    float product = x * y, sum = (x + y) * 0.7071067811865475f, gap = x * (y - x);
    return kind == FEATURE_OBSERVATION_TIMESCALE ? trace : kind == FEATURE_PRODUCT ? product : kind == FEATURE_NORMALIZED_SUM ? sum : gap;
}

static void constructFeatures(device OakAgent *a, uint32_t lane) {
    float standardized[OAK_OBSERVATION_CAPACITY];
    uint32_t observationCount = a->observationCount[lane];
    uint32_t metaAge = a->stepCount[lane] < a->g[lane].featureCapacity ? a->stepCount[lane] : a->g[lane].featureCapacity;
    float inverseAge = 1.0f / ((float)metaAge + 1.0f);
    float floorRate = 1.0f / ((float)a->stepCount[lane] + 1.0f);
    float metaRate = a->g[lane].metaRate, decay = a->g[lane].trackingRate, betaFloor = a->betaFloor[lane], ridgeFloor = a->g[lane].ridgeFloor;
    device float *alpha = a->observationMeta, *beta = a->observationMeta + OAK_OBSERVATION_CAPACITY * OAK_LANES;
    device float *trace = a->observationMeta + 2u * OAK_OBSERVATION_CAPACITY * OAK_LANES, *energy = a->observationMeta + 3u * OAK_OBSERVATION_CAPACITY * OAK_LANES;
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
        device float *level = a->observationTimescale + (m - 1u) * observationCount * OAK_LANES;
        for (uint32_t o = 0u; o < observationCount; o++) {
            float next = level[AT(o)] + rate * (standardized[o] - level[AT(o)]);
            level[AT(o)] = isfinite(a->observation[AT(o)]) ? next : level[AT(o)];
        }
    }
    for (uint32_t j = 0u; j < OAK_POLICY_MAX; j++) a->actorCredit[AT(j)] = a->actorCreditStaging[AT(j)];

    device const float *previousFeatures = featureValues(a, lane, a->stateSlot[lane]);
    device float *currentFeatures = featureValues(a, lane, 1u - a->stateSlot[lane]);
    device uint32_t *currentActive = activeFeatures(a, lane, 1u - a->stateSlot[lane]);
    uint32_t percepts = a->g[lane].percepts, actionParams = OAK_ACTION_PARAMS;
    for (uint32_t k = percepts; k < a->g[lane].activeCapacity; k++) currentFeatures[AT(currentActive[AT(k)])] = 0.0f;

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
            uint32_t lvl = j / (observationCount > 1u ? observationCount : 1u);
            x = (lvl < OAK_TRACE_LEVELS) ? a->observationTimescale[AT(lvl * observationCount + (j - lvl * observationCount))] : 0.0f;
        }
        x = finiteOrZero(x);
        currentFeatures[AT(i)] = x;
        currentActive[AT(i)] = i;
        if (a->featureMeta[AT(i * 4u + 3u)] < 0xffffffffu) a->featureMeta[AT(i * 4u + 3u)] += 1u;
    }

    uint32_t sourceCount = perceptSourceCount(a, lane), featureCapacity = a->g[lane].featureCapacity;
    device float *constructed = a->scratchValue, *rate = a->scratchRate;
    for (uint32_t i = percepts; i < featureCapacity; i++) {
        device const uint32_t *meta = a->featureMeta + i * 4u * OAK_LANES;
        constructed[AT(i)] = constructedValue(a, lane, previousFeatures, meta[AT(0u)], meta[AT(1u)], meta[AT(2u)]);
        rate[AT(i)] = trackingRateAt(a, lane, meta[AT(3u)]);
    }
    device float *moment0 = a->featureMoments, *moment1 = a->featureMoments + OAK_FEATURE_CAPACITY * OAK_LANES;
    for (uint32_t i = percepts; i < featureCapacity; i++) {
        float value = constructed[AT(i)];
        moment0[AT(i)] += rate[AT(i)] * (value - moment0[AT(i)]);
        moment1[AT(i)] += rate[AT(i)] * (value * value - moment1[AT(i)]);
    }
    for (uint32_t ai = 0u; ai < percepts; ai++) {
        uint32_t sourceIndex = ai % sourceCount, occupancy = ai / sourceCount;
        uint32_t tile, tile2;
        if (occupancy == 0u) {
            tile = featureTile(currentFeatures[AT(tileSourceIndex(a, lane, sourceIndex))]);
        } else if ((ai & 1u) == 0u) {
            tile = zoomedFeatureTile(currentFeatures[AT(tileSourceIndex(a, lane, sourceIndex))], occupancy);
        } else {
            uint32_t channelA = tileSourceIndex(a, lane, sourceIndex);
            uint32_t channelB = (sourceIndex + occupancy) % observationCount;
            tile = jointFeatureTile(currentFeatures[AT(channelA)], currentFeatures[AT(channelB)], occupancy);
        }
        tile2 = zoomedFeatureTile(currentFeatures[AT(tileSourceIndex(a, lane, sourceIndex))], occupancy + 1u);
        for (uint32_t g = 0u; g < OAK_CONSTRUCTED_PER_PERCEPT; g++) {
            uint32_t winner = percepts + (ai + g * percepts) * OAK_CANDIDATES_PER_SLOT + ((g == 0u) ? tile : tile2);
            float var = fmaxf(moment1[AT(winner)] - moment0[AT(winner)] * moment0[AT(winner)], 0.0f);
            float ridge = ridgeAt(a, lane, a->featureMeta[AT(winner * 4u + 3u)]);
            float winningValue = (constructed[AT(winner)] - moment0[AT(winner)]) * oak_rsqrt(var + ridge);
            currentFeatures[AT(winner)] = finiteOrZero(winningValue);
            currentActive[AT(percepts + ai + g * percepts)] = winner;
            if (a->featureMeta[AT(winner * 4u + 3u)] < 0xffffffffu) a->featureMeta[AT(winner * 4u + 3u)] += 1u;
        }
    }
}

static void actorCoefficients(device OakAgent *a, uint32_t lane) {
    float advantage = finiteOrZero(a->advantage[lane]);
    metaOptimizeUpdate(a, lane, a->actorMeta, advantage, 1.0f, a->actorAge[lane], a->g[lane].metaRate, a->g[lane].trackingRate);
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

static void learnFromTransition(device OakAgent *a, uint32_t lane, uint32_t previousSlot) {
    device const float *previousFeatures = featureValues(a, lane, previousSlot);
    device const float *currentFeatures = featureValues(a, lane, 1u - previousSlot);
    device const uint32_t *previousActive = activeFeatures(a, lane, previousSlot);
    for (uint32_t i = 0u; i < a->g[lane].activeCapacity; i++) {
        uint32_t index = previousActive[AT(i)];
        float previousActivation = previousFeatures[AT(index)];
        if (previousActivation == 0.0f && currentFeatures[AT(index)] == 0.0f) continue;
        device float *row = policyRow(a, index);
        for (uint32_t j = 0u; j < OAK_POLICY_MAX; j++) {
            float w = row[AT(j)];
            row[AT(j)] = keepFinite(w, w + a->policyCoef[AT(j)] * previousActivation);
        }
    }
    float valueTdError = a->td[lane];
    float valueStepScale = a->valueScale[lane];
    for (uint32_t i = 0u; i < a->g[lane].activeCapacity; i++) {
        uint32_t index = previousActive[AT(i)];
        float w = a->valueWeight[AT(index)];
        float delta = valueTdError * previousFeatures[AT(index)] * valueStepScale;
        a->valueWeight[AT(index)] = keepFinite(w, w + a->valueMeta[AT(index * 4u)] * delta);
    }
}

static void emitPolicy(device OakAgent *a, uint32_t lane, int recompute) {
    float policyLogit[OAK_POLICY_MAX];
    if (recompute) policyDot(a, lane, activeFeatures(a, lane, a->stateSlot[lane]), featureValues(a, lane, a->stateSlot[lane]), policyLogit);
    else for (uint32_t j = 0u; j < OAK_POLICY_MAX; j++) policyLogit[j] = a->logit[AT(j)];
    for (uint32_t j = 0u; j < OAK_POLICY_MAX; j++) {
        float actionMean = oak_tanh(policyLogit[j]);
        a->logit[AT(j)] = policyLogit[j];
        if (!isfinite(actionMean)) actionMean = a->policyMean[AT(j)];
        a->policyMean[AT(j)] = actionMean;
        a->policyMeanRaw[AT(j)] = finiteOrZero(policyLogit[j]);
    }
}

static void replaceWorstFeature(device OakAgent *a, uint32_t lane) {
    device const float *currentFeatures = featureValues(a, lane, 1u - a->stateSlot[lane]);
    uint32_t percepts = a->g[lane].percepts, featureCapacity = a->g[lane].featureCapacity;
    float bestNurseryBound = -FLT_MAX;
    for (uint32_t n = 0u; n < featureCapacity; n++) bestNurseryBound = fmaxf(bestNurseryBound, a->nurseryScore[AT(n)]);
    uint32_t haveWorst = 0u, worstFeature = 0u, worstAge = 0u;
    float worstUtility = 0.0f, worstRedundancy = 0.0f, worstDownstream = 0.0f;
    uint32_t eligible = 0u;
    float eligibleUtilSum = 0.0f, eligibleRedundancySum = 0.0f, eligibleDownstreamSum = 0.0f;
    for (uint32_t i = percepts; i < featureCapacity; i++) {
        device const float *e = a->replaceScore + i * 4u * OAK_LANES;
        if (e[AT(3u)] <= 0.0f) continue;
        eligible += 1u;
        eligibleUtilSum += e[AT(0u)];
        eligibleRedundancySum += e[AT(1u)];
        eligibleDownstreamSum += e[AT(2u)];
        float util = e[AT(0u)], redundancy = e[AT(1u)], downstream = e[AT(2u)];
        int worse = (haveWorst == 0u) || downstream < worstDownstream ||
                    (downstream == worstDownstream &&
                     (util < worstUtility || (util == worstUtility &&
                      (redundancy > worstRedundancy || (redundancy == worstRedundancy &&
                       a->featureMeta[AT(i * 4u + 3u)] > worstAge)))));
        if (worse) {
            haveWorst = 1u;
            worstFeature = i;
            worstAge = a->featureMeta[AT(i * 4u + 3u)];
            worstUtility = util;
            worstRedundancy = redundancy;
            worstDownstream = downstream;
        }
    }
    if (a->actorAge[lane] < 0xffffffffu) a->actorAge[lane] += 1u;
    uint32_t worst = (haveWorst != 0u) ? worstFeature : percepts;
    float invEligible = 1.0f / (float)(eligible > 1u ? eligible : 1u);
    float meanEligibleUtil = eligibleUtilSum * invEligible;
    float meanEligibleRedundancy = eligibleRedundancySum * invEligible;
    float meanEligibleDownstream = eligibleDownstreamSum * invEligible;
    int lowUtility = worstUtility < meanEligibleUtil;
    int highRedundancy = worstRedundancy > meanEligibleRedundancy;
    int lowDownstreamUse = worstDownstream <= meanEligibleDownstream;
    int structurallyReplaceable = (eligible > 1u) && lowDownstreamUse && (lowUtility || highRedundancy);
    int doReplace = structurallyReplaceable && worst < featureCapacity && bestNurseryBound > -FLT_MAX &&
                    bestNurseryBound > worstUtility + worstDownstream;
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
        device const uint32_t *nursery = a->nurseryMeta + bestNursery * 4u * OAK_LANES;
        uint32_t kindNext = (nursery[AT(2u)] + 1u) % 4u;
        a->featureMeta[AT(worst * 4u + 0u)] = nursery[AT(0u)] % featureCapacity;
        a->featureMeta[AT(worst * 4u + 1u)] = nursery[AT(1u)] % featureCapacity;
        a->featureMeta[AT(worst * 4u + 2u)] = nursery[AT(2u)] % 4u;
        a->featureMeta[AT(worst * 4u + 3u)] = nursery[AT(3u)];
        a->featureMoments[AT(worst)] = a->nurseryStats[AT(bestNursery)];
        a->featureMoments[AT(OAK_FEATURE_CAPACITY + worst)] = a->nurseryStats[AT(OAK_FEATURE_CAPACITY + bestNursery)];
        a->featureUtility[AT(worst)] = a->nurseryStats[AT(2u * OAK_FEATURE_CAPACITY + bestNursery)];
        a->valueWeight[AT(worst)] = 0.0f;
        a->nurseryMeta[AT(bestNursery * 4u + 0u)] = seedParentA;
        a->nurseryMeta[AT(bestNursery * 4u + 1u)] = seedParentB;
        a->nurseryMeta[AT(bestNursery * 4u + 2u)] = kindNext;
        a->nurseryMeta[AT(bestNursery * 4u + 3u)] = 0u;
        for (uint32_t k = 0u; k < 4u; k++) a->nurseryStats[AT(k * OAK_FEATURE_CAPACITY + bestNursery)] = 0.0f;
        a->nurseryScore[AT(bestNursery)] = -FLT_MAX;
        float initialStepSize = 1.0f / (float)a->g[lane].activeCapacity;
        a->valueMeta[AT(worst * 4u + 0u)] = initialStepSize;
        a->valueMeta[AT(worst * 4u + 1u)] = oak_log(initialStepSize);
        a->valueMeta[AT(worst * 4u + 2u)] = 0.0f;
        a->valueMeta[AT(worst * 4u + 3u)] = 0.0f;
        a->replaced[lane] = 1u;
        a->replacedFeature[lane] = worst;
        for (uint32_t d = 0u; d < OAK_POLICY_MAX; d++) policyRow(a, worst)[AT(d)] = 0.0f;
    } else {
        a->replaced[lane] = 0u;
    }
}

static void updateUtilities(device OakAgent *a, uint32_t lane) {
    uint32_t tPrev = a->stepCount[lane] - 1u;
    device const float *previousFeatures = featureValues(a, lane, 1u - a->stateSlot[lane]);
    device const float *currentFeatures = featureValues(a, lane, a->stateSlot[lane]);
    device const uint32_t *previousActive = activeFeatures(a, lane, 1u - a->stateSlot[lane]);
    uint32_t featureCapacity = a->g[lane].featureCapacity, activeCapacity = a->g[lane].activeCapacity;
    float trackingRate = trackingRateFor(tPrev, featureCapacity);
    float valueTdError = a->td[lane];
    float advantage = a->advantage[lane];
    float planningSignal = fabsf(valueTdError);
    float metaRate = a->g[lane].metaRate, decay = a->g[lane].trackingRate, betaFloor = a->betaFloor[lane];
    device float *x = a->scratchActive, *utility = a->scratchActive + OAK_ACTIVE_CAPACITY * OAK_LANES, *age = a->scratchActive + 2u * OAK_ACTIVE_CAPACITY * OAK_LANES;
    device float *alpha = a->scratchActive + 3u * OAK_ACTIVE_CAPACITY * OAK_LANES, *beta = a->scratchActive + 4u * OAK_ACTIVE_CAPACITY * OAK_LANES;
    device float *trace = a->scratchActive + 5u * OAK_ACTIVE_CAPACITY * OAK_LANES, *energy = a->scratchActive + 6u * OAK_ACTIVE_CAPACITY * OAK_LANES;
    for (uint32_t s = 0u; s < activeCapacity; s++) {
        uint32_t i = previousActive[AT(s)];
        float xi = previousFeatures[AT(i)];
        float actorSignal = (i == 0u) ? (fabsf(advantage) + (fabsf(advantage) + 1.0f)) : 0.0f;
        float selectedPolicyLoad = 0.0f;
        device const float *row = policyRow(a, i);
        for (uint32_t d = 0u; d < OAK_POLICY_MAX; d++) selectedPolicyLoad += fabsf(row[AT(d)]);
        float load = fabsf(xi) * (fabsf(a->valueWeight[AT(i)]) + selectedPolicyLoad);
        load = finiteOrZero(load);
        float featureUtilitySignal = fabsf(planningSignal * xi) + fabsf(actorSignal * xi) + load;
        utility[AT(s)] = a->featureUtility[AT(i)] + trackingRate * (featureUtilitySignal - a->featureUtility[AT(i)]);
        x[AT(s)] = xi;
        age[AT(s)] = (float)a->featureMeta[AT(i * 4u + 3u)];
        alpha[AT(s)] = a->valueMeta[AT(i * 4u + 0u)]; beta[AT(s)] = a->valueMeta[AT(i * 4u + 1u)]; trace[AT(s)] = a->valueMeta[AT(i * 4u + 2u)]; energy[AT(s)] = a->valueMeta[AT(i * 4u + 3u)];
    }
    for (uint32_t s = 0u; s < activeCapacity; s++) {
        float xi = x[AT(s)], oldAlpha = alpha[AT(s)], oldBeta = beta[AT(s)], oldH = trace[AT(s)], oldGradEnergy = energy[AT(s)];
        int skip = !isfinite(valueTdError) || !isfinite(xi);
        float grad = valueTdError * xi;
        float metaGradient = grad * oldH;
        float inverseAge = 1.0f / (age[AT(s)] + 1.0f);
        float gradEnergy = keepFinite(oldGradEnergy, oldGradEnergy + inverseAge * (metaGradient * metaGradient - oldGradEnergy));
        float normalizedMetaGradient = metaGradient * oak_rsqrt(fmaxf(gradEnergy, 0.0f) + inverseAge);
        float newBeta = oldBeta + metaRate * normalizedMetaGradient;
        float nextBeta = keepFinite(oldBeta, oak_clamp(newBeta, betaFloor, 0.0f));
        float nextAlpha = keepFinite(oldAlpha, oak_exp(nextBeta));
        float hDecay = (1.0f - decay) * fmaxf(0.0f, 1.0f - oldAlpha * (xi * xi));
        float nextH = keepFinite(oldH, hDecay * oldH + oldAlpha * grad);
        alpha[AT(s)] = skip ? oldAlpha : nextAlpha;
        beta[AT(s)] = skip ? oldBeta : nextBeta;
        trace[AT(s)] = skip ? oldH : nextH;
        energy[AT(s)] = skip ? oldGradEnergy : gradEnergy;
    }
    for (uint32_t s = 0u; s < activeCapacity; s++) {
        uint32_t i = previousActive[AT(s)];
        a->featureUtility[AT(i)] = utility[AT(s)];
        a->valueMeta[AT(i * 4u + 0u)] = alpha[AT(s)]; a->valueMeta[AT(i * 4u + 1u)] = beta[AT(s)]; a->valueMeta[AT(i * 4u + 2u)] = trace[AT(s)]; a->valueMeta[AT(i * 4u + 3u)] = energy[AT(s)];
    }
    uint32_t parity = (tPrev >> 1u) & 1u;
    device float *stat0 = a->nurseryStats, *stat1 = a->nurseryStats + OAK_FEATURE_CAPACITY * OAK_LANES;
    device float *stat2 = a->nurseryStats + 2u * OAK_FEATURE_CAPACITY * OAK_LANES, *stat3 = a->nurseryStats + 3u * OAK_FEATURE_CAPACITY * OAK_LANES;
    for (uint32_t block = parity * OAK_CAPACITY; block < featureCapacity; block += 2u * OAK_CAPACITY) {
        uint32_t blockEnd = block + OAK_CAPACITY < featureCapacity ? block + OAK_CAPACITY : featureCapacity;
        float constructed[OAK_CAPACITY], rate[OAK_CAPACITY], ridge[OAK_CAPACITY];
        for (uint32_t n = block; n < blockEnd; n++) {
            device const uint32_t *nursery = a->nurseryMeta + n * 4u * OAK_LANES;
            constructed[n - block] = constructedValue(a, lane, previousFeatures, nursery[AT(0u)], nursery[AT(1u)], nursery[AT(2u)]);
            rate[n - block] = trackingRateAt(a, lane, nursery[AT(3u)]);
            ridge[n - block] = ridgeAt(a, lane, nursery[AT(3u)]);
        }
        for (uint32_t n = block; n < blockEnd; n++) {
            uint32_t j = n - block;
            float value = constructed[j], s0 = stat0[AT(n)], s1 = stat1[AT(n)], s2 = stat2[AT(n)], s3 = stat3[AT(n)];
            s0 += rate[j] * (value - s0);
            s1 += rate[j] * (value * value - s1);
            float var = fmaxf(s1 - s0 * s0, 0.0f);
            float standardizedCandidate = (value - s0) * oak_rsqrt(var + ridge[j]);
            float candidateUtilitySignal = fabsf(valueTdError * standardizedCandidate);
            int finite = isfinite(candidateUtilitySignal);
            s2 = finite ? s2 + trackingRate * (candidateUtilitySignal - s2) : s2;
            s3 = finite ? s3 + trackingRate * (candidateUtilitySignal * candidateUtilitySignal - s3) : s3;
            stat0[AT(n)] = s0; stat1[AT(n)] = s1; stat2[AT(n)] = s2; stat3[AT(n)] = s3;
            uint32_t nurseryAge = a->nurseryMeta[AT(n * 4u + 3u)];
            nurseryAge = nurseryAge < 0xffffffffu ? nurseryAge + 1u : nurseryAge;
            a->nurseryMeta[AT(n * 4u + 3u)] = nurseryAge;
            float bound = s2 - sqrtf(fmaxf(s3 - s2 * s2, 0.0f) * decay);
            a->nurseryScore[AT(n)] = (nurseryAge >= featureCapacity && isfinite(s2) && isfinite(bound)) ? bound : -FLT_MAX;
        }
    }
    for (uint32_t block = a->g[lane].percepts + parity * OAK_CAPACITY; block < featureCapacity; block += 2u * OAK_CAPACITY) {
        uint32_t blockEnd = block + OAK_CAPACITY < featureCapacity ? block + OAK_CAPACITY : featureCapacity;
        for (uint32_t i = block; i < blockEnd; i++) {
            device float *score = a->replaceScore + i * 4u * OAK_LANES;
            device const uint32_t *meta = a->featureMeta + i * 4u * OAK_LANES;
            if (meta[AT(3u)] < featureCapacity) {
                score[AT(0u)] = 0.0f; score[AT(1u)] = 0.0f; score[AT(2u)] = 0.0f; score[AT(3u)] = 0.0f;
                continue;
            }
            uint32_t parentA = meta[AT(0u)];
            float util = finiteOrZero(a->featureUtility[AT(i)]);
            float redundancy = finiteOrZero(fabsf(currentFeatures[AT(i)] * currentFeatures[AT(parentA)]));
            float downstreamUse = finiteOrZero(fabsf(a->valueWeight[AT(i)]));
            score[AT(0u)] = util; score[AT(1u)] = redundancy; score[AT(2u)] = downstreamUse; score[AT(3u)] = 1.0f;
        }
    }
}

static void sampleDiscreteAction(device OakAgent *a, uint32_t lane, uint32_t seed, uint32_t stepIdx, thread float *ctrl) {
    uint32_t discreteActionCount = a->envActionCount[lane];
    u4 rw = philox4x32(seed, RNG_STREAM_ACT, stepIdx, 0u, 0u, 0u);
    u4 rw1 = discreteActionCount > 4u ? philox4x32(seed, RNG_STREAM_ACT, stepIdx, 1u, 0u, 0u) : rw;
    uint32_t words[8] = { rw.x, rw.y, rw.z, rw.w, rw1.x, rw1.y, rw1.z, rw1.w };
    float behaviorLogitMax = -FLT_MAX;
    for (uint32_t d = 0u; d < discreteActionCount; d++) behaviorLogitMax = fmaxf(behaviorLogitMax, finiteOrZero(a->policyMeanRaw[AT(d)]));
    float behaviorSoftmaxSum = 0.0f;
    for (uint32_t d = 0u; d < discreteActionCount; d++) behaviorSoftmaxSum += oak_exp(finiteOrZero(a->policyMeanRaw[AT(d)]) - behaviorLogitMax);
    uint32_t sampledAction = 0u;
    float bestGumbelScore = -FLT_MAX;
    for (uint32_t d = 0u; d < discreteActionCount; d++) {
        float mu = finiteOrZero(a->policyMeanRaw[AT(d)]);
        float uniform = fminf(u01open(words[d]), 0x1.fffffep-1f);
        float gumbelScore = mu - oak_log(-oak_log(uniform));
        if (gumbelScore > bestGumbelScore) {
            bestGumbelScore = gumbelScore;
            sampledAction = d;
        }
    }
    for (uint32_t d = 0u; d < discreteActionCount; d++) ctrl[d] = (d == 0u) ? (float)sampledAction : 0.0f;
    for (uint32_t t = 0u; t < OAK_ACTION_PARAMS; t++) {
        float actionSignal = 0.0f;
        if (t < discreteActionCount)
            actionSignal = ((t == sampledAction) ? 1.0f : 0.0f) - oak_exp(finiteOrZero(a->policyMeanRaw[AT(t)]) - behaviorLogitMax) / behaviorSoftmaxSum;
        a->actionSignal[AT(t)] = actionSignal;
    }
    float logitMax = -FLT_MAX;
    for (uint32_t d = 0u; d < discreteActionCount; d++) logitMax = fmaxf(logitMax, finiteOrZero(a->logit[AT(d)]));
    float softmaxSum = 0.0f;
    for (uint32_t d = 0u; d < discreteActionCount; d++) softmaxSum += oak_exp(finiteOrZero(a->logit[AT(d)]) - logitMax);
    for (uint32_t d = 0u; d < OAK_POLICY_MAX; d++) {
        float credit = 0.0f;
        if (d < discreteActionCount)
            credit = ((d == sampledAction) ? 1.0f : 0.0f) - oak_exp(finiteOrZero(a->logit[AT(d)]) - logitMax) / softmaxSum;
        a->actorCreditStaging[AT(d)] = credit;
    }
}

static void sampleAction(device OakAgent *a, uint32_t lane, uint32_t seed, uint32_t stepIdx, thread float *ctrl) {
    if (!a->box[lane]) {
        sampleDiscreteAction(a, lane, seed, stepIdx, ctrl);
        return;
    }
    uint32_t envActionCount = a->envActionCount[lane];
    for (uint32_t d = 0u; d < envActionCount; d++) {
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

void oak_init(device uint32_t *state, uint32_t lane, uint32_t env, uint32_t seed, device const float *obs) {
    device OakAgent *a = (device OakAgent *)state;
    for (uint32_t w = 0u; w < oak_stateWords(); w++) state[AT(w)] = 0u;
    a->g[lane] = oak_geometry(env_obs(env));
    a->betaFloor[lane] = 2.0f * oak_log(a->g[lane].trackingRate);
    for (uint32_t n = 0u; n < a->g[lane].featureTimescale; n++) a->reciprocal[AT(n)] = 1.0f / ((float)n + 1.0f);
    a->observationCount[lane] = env_obs(env);
    a->envActionCount[lane] = env_actions(env);
    a->box[lane] = env_box(env);
    uint32_t percepts = a->g[lane].percepts, traceChannels = a->g[lane].traceChannels;
    for (uint32_t p = 0u; p < a->g[lane].featureCapacity; p++) a->traceIndex[AT(p)] = p % traceChannels;
    float alpha = a->g[lane].ridgeFloor;
    float logAlpha = oak_log(alpha);
    for (uint32_t feature = 0u; feature < a->g[lane].featureCapacity; feature++) {
        uint32_t parent = feature % (feature % 4u == 3u ? traceChannels : percepts);
        a->nurseryMeta[AT(feature * 4u + 0u)] = parent;
        a->nurseryMeta[AT(feature * 4u + 1u)] = (feature * 7u + 3u) % percepts;
        a->nurseryMeta[AT(feature * 4u + 2u)] = feature % 4u;
        a->nurseryMeta[AT(feature * 4u + 3u)] = 0u;
        if (feature >= percepts) {
            uint32_t source = feature - percepts;
            if (source < traceChannels) {
                a->featureMeta[AT(feature * 4u + 0u)] = source; a->featureMeta[AT(feature * 4u + 1u)] = 0u; a->featureMeta[AT(feature * 4u + 2u)] = 3u; a->featureMeta[AT(feature * 4u + 3u)] = 0u;
            } else {
                a->featureMeta[AT(feature * 4u + 0u)] = source % percepts;
                a->featureMeta[AT(feature * 4u + 1u)] = (source * 7u + 3u) % percepts;
                a->featureMeta[AT(feature * 4u + 2u)] = source % 3u;
                a->featureMeta[AT(feature * 4u + 3u)] = 0u;
            }
        }
        a->valueMeta[AT(feature * 4u + 0u)] = alpha;
        a->valueMeta[AT(feature * 4u + 1u)] = logAlpha;
    }
    a->actorMeta[AT(0u)] = 1.0f;
    uint32_t hash = seed * 2654435761u + 977u;
    uint32_t action = hash % OAK_POLICY_MAX;
    float sign = ((hash >> 16u) & 1u) == 0u ? 1.0f : -1.0f;
    policyRow(a, 0u)[AT(action)] = sign * (1.0f / (float)OAK_ACTION_PARAMS);
    for (uint32_t o = 0u; o < a->observationCount[lane]; o++) {
        a->observationMeta[AT(o)] = alpha;
        a->observationMeta[AT(OAK_OBSERVATION_CAPACITY + o)] = logAlpha;
    }
    for (uint32_t i = 0u; i < a->observationCount[lane]; i++) a->observation[AT(i)] = obs[i];
}

void oak_act(device uint32_t *state, uint32_t lane, uint32_t env, uint32_t seed, uint32_t step, device const float *obs, float reward, uint32_t done, thread float *ctrl) {
    (void)env; (void)done;
    device OakAgent *a = (device OakAgent *)state;
    for (uint32_t i = 0u; i < a->observationCount[lane]; i++) a->observation[AT(i)] = obs[i];
    a->reward[lane] = reward;
    a->stepCount[lane] = step;
    a->stateSlot[lane] = step & 1u;
    constructFeatures(a, lane);
    predictAndControl(a, lane);
    replaceWorstFeature(a, lane);
    a->stepCount[lane] += 1u;
    a->stateSlot[lane] ^= 1u;
    emitPolicy(a, lane, a->replaced[lane] != 0u);
    sampleAction(a, lane, seed, step, ctrl);
    actorCoefficients(a, lane);
    learnFromTransition(a, lane, 1u - a->stateSlot[lane]);
    updateUtilities(a, lane);
}
