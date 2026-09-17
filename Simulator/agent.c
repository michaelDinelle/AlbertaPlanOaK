#include <stdio.h>

#include <float.h>
#include "agent.h"

static float clampf_(float v, float lo, float hi){ return v<lo?lo:(v>hi?hi:v); }
static float keepFinite(float o, float n){ return isfinite(n) ? n : o; }
static float finiteOrZero(float v){ return isfinite(v) ? v : 0.0f; }
static float rsqrtf_(float v){ return 1.0f/sqrtf(v); }
static Record *optionRecord(Agent a, uint32_t feature, uint32_t option){
    return &a.parameters->option[feature * OAK_OPTION_COUNT + option];
}
static float *activeValues(Agent a, uint32_t slot){
    return a.memory->activeValue + slot * OAK_ACTIVE_CAPACITY;
}
static float *featureValues(Agent a, uint32_t slot){
    return a.memory->features + slot * OAK_FEATURE_CAPACITY;
}
static uint32_t *activeFeatures(Agent a, uint32_t slot){
    return a.indices->activeFeature + slot * OAK_ACTIVE_CAPACITY;
}
static uint32_t perceptSourceCount(Agent a){
    return (1u + OAK_TRACE_LEVELS) * a.observationCount + OAK_ACTION_PARAMS + 2u;
}
static uint32_t tileSourceIndex(Agent a, uint32_t source){
    return (source < a.observationCount + a.envActionCount) ? source
                                                            : a.observationCount + OAK_ACTION_PARAMS;
}
static uint32_t oakWrapTile(float f, uint32_t m){
    if (fabsf(f) < 2147483648.0f){
        int32_t r = (int32_t)f % (int32_t)m;
        return (uint32_t)(r < 0 ? r + (int32_t)m : r);
    }
    if (!isfinite(f)) return 0u;
    double q = fmod((double)f, (double)m);
    if (q < 0.0) q += (double)m;
    return (uint32_t)q;
}
static uint32_t zoomedFeatureTile(float activation, uint32_t zoomLevel){
    union { uint32_t bits; float value; } zoomBits = { (127u + zoomLevel) << 23 };
    float zoom = zoomLevel > 127u ? ldexpf(1.0f, (int)zoomLevel) : zoomBits.value;
    return oakWrapTile(floorf(activation*zoom + OAK_TILE_HALF), OAK_CANDIDATES_PER_SLOT);
}
static uint32_t jointFeatureTile(float a_, float b_, uint32_t sourceReuse){
    float candidates = (float)OAK_CANDIDATES_PER_SLOT;
    a_ += ((sourceReuse & 1u) == 0u) ? (candidates / (float)OAK_TILE_LAST) : 0.0f;
    uint32_t tileA = oakWrapTile(floorf((a_ + OAK_TILE_HALF) * ((float)OAK_JOINT_ROWS / candidates)),
                                 OAK_JOINT_ROWS);
    uint32_t tileB = oakWrapTile(floorf((b_ + OAK_TILE_HALF) * (2.0f / candidates)), 2u);
    return tileA * 2u + tileB;
}
static float normalizedObservation(Agent a, uint32_t o){
    float mean = a.memory->observationMean[o];
    float second = a.memory->observationSecondMoment[o];
    float var = second - mean*mean; if (var < 0.0f) var = 0.0f;
    return (a.memory->observation[o] - mean) * rsqrtf_(var + 1.0f/(float)(2u*OAK_PERCEPT_COUNT));
}

static float g_actQ[OAK_POLICY_MAX][OAK_FEATURE_CAPACITY];
static f4    g_actMeta[OAK_POLICY_MAX][OAK_FEATURE_CAPACITY];
static uint32_t g_actAge[OAK_POLICY_MAX][OAK_FEATURE_CAPACITY];
static int   g_actInit;
static uint32_t g_prevAct;
static uint32_t g_prevF[OAK_ACTIVE_CAPACITY];
static int   g_havePrevAct;

static uint32_t g_traceMod;
typedef struct { uint16_t pa, pb, kind, trace; } CVPlan;
static CVPlan g_cv[OAK_FEATURE_CAPACITY];
static float g_constructed[OAK_FEATURE_CAPACITY];
static float g_invAge[OAK_FEATURE_TIMESCALE];
static float g_traceRate[OAK_TRACE_LEVELS + 1u];
static inline float trackingRate(uint32_t age){
    uint32_t cap = OAK_FEATURE_TIMESCALE - 1u;
    return g_invAge[age < cap ? age : cap];
}
static inline float ridgeRate(uint32_t age){
    uint32_t cap = 2u*OAK_PERCEPT_COUNT - 1u;
    return g_invAge[age < cap ? age : cap];
}
static void metaOptimizeUpdate(f4 *meta, uint32_t i, float signal, float x, uint32_t age){
    if (!isfinite(signal) || !isfinite(x)) return;
    f4 old = meta[i];
    float oldAlpha=old.x, oldBeta=old.y, oldH=old.z, oldGradEnergy=old.w;
    float grad = signal * x;
    float metaGradient = grad * oldH;
    float gradEnergyRate = trackingRate(age);
    float gradEnergy = keepFinite(oldGradEnergy,
        oldGradEnergy + gradEnergyRate*(metaGradient*metaGradient - oldGradEnergy));
    float ge = gradEnergy > 0.0f ? gradEnergy : 0.0f;
    float normalizedMetaGradient = metaGradient * rsqrtf_(ge + gradEnergyRate);
    float newBeta = oldBeta + OAK_META_RATE * normalizedMetaGradient;
    float beta = keepFinite(oldBeta, clampf_(newBeta, 2.0f*logf(OAK_TRACKING_RATE), 0.0f));
    float alpha = (age > 0u && beta == oldBeta) ? oldAlpha : keepFinite(oldAlpha, expf(beta));
    float featureCurvature = x * x;
    float hDecay = (1.0f - OAK_TRACKING_RATE) * ((1.0f - oldAlpha*featureCurvature) > 0.0f
                                                 ? (1.0f - oldAlpha*featureCurvature) : 0.0f);
    f4 nv; nv.x=alpha; nv.y=beta; nv.z=keepFinite(oldH, hDecay*oldH + oldAlpha*grad); nv.w=gradEnergy;
    meta[i]=nv;
}
static void buildPlans(Agent a){
    uint32_t traceChannels = OAK_TRACE_LEVELS*a.observationCount;
    if (traceChannels < 1u) traceChannels = 1u;
    for (uint32_t i=0;i<OAK_FEATURE_CAPACITY;++i){
        uint32_t x=0u, y=0u, z=0u;
        if (i >= OAK_PERCEPT_COUNT){
            uint32_t source = i - OAK_PERCEPT_COUNT;
            if (source < traceChannels){ x=source; z=3u; }
            else { x=source % OAK_PERCEPT_COUNT; y=(source*7u+3u)%OAK_PERCEPT_COUNT; z=source % 3u; }
        }
        g_cv[i].kind  = (uint16_t)(z % 4u);
        g_cv[i].trace = (uint16_t)(x % g_traceMod);
        g_cv[i].pa    = (uint16_t)(x % OAK_FEATURE_CAPACITY);
        g_cv[i].pb    = (uint16_t)(y % OAK_FEATURE_CAPACITY);
    }
}
static inline float constructedFrom(Agent a, const float *features, CVPlan p){
    if (p.kind == FEATURE_OBSERVATION_TIMESCALE) return a.memory->observationTimescale[p.trace];
    float x = features[p.pa], y = features[p.pb];
    return p.kind == FEATURE_PRODUCT ? x*y
         : p.kind == FEATURE_NORMALIZED_SUM ? (x+y)*0.7071067811865475f
         : x*(y-x);
}

void constructFeatures(Agent a){
    g_traceMod = OAK_TRACE_LEVELS * a.observationCount;
    if (a.stepCount == 0u) buildPlans(a);
    float standardized[OAK_OBSERVATION_CAPACITY];
    float stepRate = 1.0f/((float)a.stepCount + 1.0f);
    for (uint32_t o = 0u; o < a.observationCount; ++o){
        float x = a.memory->observation[o];
        if (isfinite(x)){
            float observationMean = a.memory->observationMean[o];
            uint32_t age = a.stepCount < OAK_FEATURE_CAPACITY ? a.stepCount : OAK_FEATURE_CAPACITY;
            metaOptimizeUpdate(a.memory->observationMeta, o, x - observationMean, 1.0f, age);
            float r2 = a.memory->observationMeta[o].x;
            float obsRate = stepRate > r2 ? stepRate : r2;
            a.memory->observationMean[o] += obsRate * (x - observationMean);
            a.memory->observationSecondMoment[o] +=
                obsRate * (x*x - a.memory->observationSecondMoment[o]);
            standardized[o] = normalizedObservation(a, o);
            for (uint32_t m = 1u; m <= OAK_TRACE_LEVELS; ++m){
                uint32_t slot = (m-1u)*a.observationCount + o;
                float rate = g_traceRate[m];
                a.memory->observationTimescale[slot] +=
                    rate * (standardized[o] - a.memory->observationTimescale[slot]);
            }
        } else {
            standardized[o] = normalizedObservation(a, o);
        }
    }
    memcpy(a.memory->importance, a.memory->importanceStaging, sizeof a.memory->importance);
    memcpy(a.memory->optionActorCredit, a.memory->actorCreditStaging, sizeof a.memory->optionActorCredit);

    const float *previousFeatures = featureValues(a, a.stateSlot);
    float *currentFeatures = featureValues(a, 1u - a.stateSlot);
    uint32_t *currentActive = activeFeatures(a, 1u - a.stateSlot);
    float *currentActiveValue = activeValues(a, 1u - a.stateSlot);

    for (uint32_t i = 0u; i < OAK_PERCEPT_COUNT; ++i){
        float x;
        if (i < a.observationCount) x = standardized[i];
        else if (i < a.observationCount + OAK_ACTION_PARAMS) x = a.memory->actionSignal[i - a.observationCount];
        else if (i == a.observationCount + OAK_ACTION_PARAMS) x = a.memory->reward;
        else if (i == a.observationCount + OAK_ACTION_PARAMS + 1u) x = 1.0f;
        else {
            uint32_t j = i - (a.observationCount + OAK_ACTION_PARAMS + 2u);
            x = j < g_traceMod ? a.memory->observationTimescale[j] : 0.0f;
        }
        x = finiteOrZero(x);
        currentFeatures[i] = x;
        currentActive[i] = i;
        currentActiveValue[i] = x;
        if (a.indices->featureAge[i] < 0xffffffffu) a.indices->featureAge[i] += 1u;
        a.indices->featureRate[i] = trackingRate(a.indices->featureAge[i]);
    }

    for (uint32_t i = OAK_PERCEPT_COUNT; i < OAK_FEATURE_CAPACITY; ++i)
        g_constructed[i] = constructedFrom(a, previousFeatures, g_cv[i]);
    {
        const float *restrict cv = g_constructed;
        const float *restrict rt = a.indices->featureRate;
        float *restrict mx = a.memory->featureMean;
        float *restrict my = a.memory->featureSecondMoment;
        for (uint32_t i = OAK_PERCEPT_COUNT; i < OAK_FEATURE_CAPACITY; ++i){
            float c = cv[i], r = rt[i], x = mx[i], y = my[i];
            mx[i] = x + r * (c - x);
            my[i] = y + r * (c*c - y);
        }
    }
    uint32_t sourceCount = perceptSourceCount(a);
    for (uint32_t s = 0u; s < OAK_CONSTRUCTED_SLOTS; ++s){
        uint32_t sourceIndex = s % sourceCount, occupancy = s / sourceCount;
        uint32_t tile;
        if (occupancy == 0u)
            tile = zoomedFeatureTile(currentFeatures[tileSourceIndex(a, sourceIndex)], 0u);
        else if ((s & 1u) == 0u)
            tile = zoomedFeatureTile(currentFeatures[tileSourceIndex(a, sourceIndex)], occupancy);
        else {
            uint32_t chA = tileSourceIndex(a, sourceIndex);
            uint32_t chB = (sourceIndex + occupancy) % a.observationCount;
            tile = jointFeatureTile(currentFeatures[chA], currentFeatures[chB], occupancy);
        }
        uint32_t winner = OAK_PERCEPT_COUNT + s*OAK_CANDIDATES_PER_SLOT + tile;
        float wmx = a.memory->featureMean[winner], wmy = a.memory->featureSecondMoment[winner];
        float var = wmy - wmx*wmx; if (var < 0.0f) var = 0.0f;
        currentFeatures[currentActive[OAK_PERCEPT_COUNT + s]] = 0.0f;
        float standardized = finiteOrZero((g_constructed[winner] - wmx)
                           * rsqrtf_(var + ridgeRate(a.indices->featureAge[winner])));
        currentFeatures[winner] = standardized;
        currentActive[OAK_PERCEPT_COUNT + s] = winner;
        currentActiveValue[OAK_PERCEPT_COUNT + s] = standardized;
        if (a.indices->featureAge[winner] < 0xffffffffu) a.indices->featureAge[winner] += 1u;
        a.indices->featureRate[winner] = trackingRate(a.indices->featureAge[winner]);
    }
}

void predictAndControl(Agent a){
    const float *previousFeatures = featureValues(a, a.stateSlot);
    const float *currentFeatures = featureValues(a, 1u - a.stateSlot);
    const uint32_t *previousActive = activeFeatures(a, a.stateSlot);
    const uint32_t *currentActive = activeFeatures(a, 1u - a.stateSlot);
    const float *previousActiveValue = activeValues(a, a.stateSlot);
    const float *currentActiveValue = activeValues(a, 1u - a.stateSlot);
    float previousValue=0.0f, currentValue=0.0f;
    float valueStepDenominator=1.0f, previousFeatureNormSq=0.0f;
    for (uint32_t i = 0u; i < OAK_ACTIVE_CAPACITY; ++i){
        uint32_t pi = previousActive[i], ci = currentActive[i];
        float pa = previousActiveValue[i], ca = currentActiveValue[i];
        previousValue = fmaf(a.parameters->valueWeight[pi], pa, previousValue);
        currentValue  = fmaf(a.parameters->valueWeight[ci], ca, currentValue);
        valueStepDenominator += a.memory->valueAlpha[pi] * (pa*pa);
        previousFeatureNormSq = fmaf(pa, pa, previousFeatureNormSq);
    }

    float gauge = a.memory->rewardCenter;
    float previousDifferentialValue = previousValue - gauge;
    float currentDifferentialValue  = currentValue  - gauge;
    float averageRewardOld = a.memory->averageReward;
    float centeredReward = a.memory->reward - averageRewardOld;
    a.memory->td = finiteOrZero(centeredReward + currentDifferentialValue - previousDifferentialValue);
    float tracked = averageRewardOld + OAK_TRACKING_RATE*(a.memory->reward - averageRewardOld);
    a.memory->averageReward = isfinite(tracked) ? tracked : averageRewardOld;

    for (uint32_t k = 0u; k < OAK_OPTION_UPDATES; ++k){
        float optionValuePrev=0.0f, optionValueNext=0.0f;
        float policyLogit[OAK_POLICY_MAX];
        for (uint32_t j=0u;j<OAK_POLICY_MAX;++j) policyLogit[j]=0.0f;
        for (uint32_t idx = 0u; idx < OAK_ACTIVE_CAPACITY; ++idx){
            uint32_t pi = previousActive[idx], ci = currentActive[idx];
            float pa = previousActiveValue[idx], ca = currentActiveValue[idx];
            if (k != 0u && pa != 0.0f)
                optionValuePrev = fmaf(optionRecord(a, pi, k)->value, pa, optionValuePrev);
            if (ca != 0.0f){
                const Record *rc = optionRecord(a, ci, k);
                if (k != 0u) optionValueNext = fmaf(rc->value, ca, optionValueNext);
                for (uint32_t j=0u;j<OAK_POLICY_MAX;++j)
                    policyLogit[j] = fmaf(rc->policy[j], ca, policyLogit[j]);
            }
        }
        for (uint32_t j=0u;j<OAK_POLICY_MAX;++j)
            a.memory->optionLogit[k*OAK_POLICY_MAX+j]=policyLogit[j];
        float optionAverageRewardOld = a.memory->optionAverageReward[k];
        float optionTdError;
        if (k == 0u) optionTdError = a.memory->td - optionAverageRewardOld;
        else {
            float stoppingValue = currentDifferentialValue
                                + currentFeatures[k];
            float stop = (isfinite(optionValueNext) && isfinite(stoppingValue)
                          && stoppingValue >= optionValueNext) ? 1.0f : 0.0f;
            optionTdError = finiteOrZero(
                (isfinite(optionValuePrev) && isfinite(optionValueNext) && isfinite(stoppingValue))
                ? (centeredReward - optionAverageRewardOld + (1.0f-stop)*optionValueNext
                   + stop*stoppingValue - optionValuePrev) : 0.0f);
        }
        a.memory->actorAdvantage[k] = finiteOrZero(optionTdError);
        a.memory->optionAverageReward[k] =
            keepFinite(optionAverageRewardOld, optionAverageRewardOld + OAK_TRACKING_RATE*optionTdError);
    }
    {
        float stepRate = 1.0f/((float)a.stepCount + 1.0f), capRate = 1.0f/(float)OAK_FEATURE_CAPACITY;
        float gaugeRate = stepRate > capRate ? stepRate : capRate;
        float gaugeNew = a.memory->rewardCenter + gaugeRate*(currentValue - a.memory->rewardCenter);
        a.memory->rewardCenter = keepFinite(a.memory->rewardCenter, gaugeNew);
    }
    a.memory->valueScale = (isfinite(valueStepDenominator) && valueStepDenominator > 0.0f)
                           ? (1.0f/valueStepDenominator) : 0.0f;
    a.memory->previousFeatureNormSq = previousFeatureNormSq;
}

void planAllOptions(Agent a){
    for (uint32_t k = 0u; k < OAK_OPTION_UPDATES; ++k){
        float importanceRatio = a.memory->importance[k];
        float advantage = finiteOrZero(a.memory->actorAdvantage[k]);
        metaOptimizeUpdate(a.memory->actorMeta, k, advantage, 1.0f, 0u);
        float actorAlpha = a.memory->actorMeta[k].x / OAK_ACTOR_ALPHA_DENOM;
        float previousFeatureNormSq = a.memory->previousFeatureNormSq;
        for (uint32_t d=0u; d<OAK_POLICY_MAX; ++d){
            float targetGrad = a.memory->optionActorCredit[k*OAK_POLICY_MAX+d];
            float gradScale = importanceRatio*targetGrad;
            float denom = 1.0f + actorAlpha*advantage*advantage*gradScale*gradScale*previousFeatureNormSq;
            float scale = (isfinite(denom) && denom > 0.0f) ? (1.0f/denom) : 0.0f;
            a.memory->optionPolicyCoef[k*OAK_POLICY_MAX+d] = finiteOrZero(actorAlpha*advantage*gradScale*scale);
        }
    }
}

static void trackFeatureUtility(Agent a, const uint32_t *active, const float *activeValue){
    for (uint32_t z = 0u; z < OAK_ACTIVE_CAPACITY; ++z){
        uint32_t feature = active[z];
        float contribution = fabsf(a.parameters->valueWeight[feature] * activeValue[z]);
        float utility = a.memory->featureUtility[feature];
        a.memory->featureUtility[feature] = keepFinite(utility,
            utility + OAK_TRACKING_RATE * (contribution - utility));
    }
}

static void replaceLeastUsefulFeature(Agent a){
    const float *restrict utility = a.memory->featureUtility;
    const uint32_t *restrict age = a.indices->featureAge;
    uint32_t worst = OAK_FEATURE_CAPACITY;
    float worstUtility = FLT_MAX;
    for (uint32_t feature = OAK_PERCEPT_COUNT; feature < OAK_FEATURE_CAPACITY; ++feature){
        float score = age[feature] >= OAK_FEATURE_TIMESCALE ? utility[feature] : FLT_MAX;
        int better = score < worstUtility;
        worstUtility = better ? score : worstUtility;
        worst = better ? feature : worst;
    }
    if (worst == OAK_FEATURE_CAPACITY) return;

    uint32_t draw = a.stepCount*2654435761u + worst*40503u + 977u;
    g_cv[worst].kind  = (uint16_t)(draw % 3u);
    g_cv[worst].trace = (uint16_t)((draw >> 8u) % g_traceMod);
    g_cv[worst].pa    = (uint16_t)((draw >> 11u) % OAK_PERCEPT_COUNT);
    g_cv[worst].pb    = (uint16_t)((draw >> 19u) % OAK_PERCEPT_COUNT);

    a.memory->featureMean[worst] = 0.0f;
    a.memory->featureSecondMoment[worst] = 0.0f;
    a.memory->featureUtility[worst] = 0.0f;
    a.parameters->valueWeight[worst] = 0.0f;
    a.indices->featureAge[worst] = 0u;
    a.indices->featureRate[worst] = trackingRate(0u);
    for (uint32_t option = 0u; option < OAK_OPTION_COUNT; ++option){
        Record *record = optionRecord(a, worst, option);
        record->value = 0.0f;
        for (uint32_t j = 0u; j < OAK_POLICY_MAX; ++j) record->policy[j] = 0.0f;
    }
}

void learnFromTransition(Agent a, uint32_t previousSlot){
    const uint32_t policyN = a.envActionCount;
    const float *currentFeatures  = featureValues(a, 1u - previousSlot);
    const uint32_t *previousActive = activeFeatures(a, previousSlot);
    const float *previousActiveValue = activeValues(a, previousSlot);
    for (uint32_t k = 0u; k < OAK_OPTION_UPDATES; ++k){
        float actorCoefficient[OAK_POLICY_MAX];
        for (uint32_t j=0u;j<policyN;++j)
            actorCoefficient[j]=a.memory->optionPolicyCoef[k*OAK_POLICY_MAX+j];
        for (uint32_t idx=0u; idx<OAK_ACTIVE_CAPACITY; ++idx){
            uint32_t i = previousActive[idx];
            float previousActivation = previousActiveValue[idx], currentActivation = currentFeatures[i];
            if (previousActivation == 0.0f && currentActivation == 0.0f) continue;
            Record *rec = optionRecord(a, i, k);
            for (uint32_t j=0u;j<policyN;++j){
                float w = rec->policy[j];
                rec->policy[j] = keepFinite(w, w + actorCoefficient[j]*previousActivation);
            }
            if (k != 0u) rec->value = 0.0f;
        }
    }

    for (uint32_t idx=0u; idx<OAK_ACTIVE_CAPACITY; ++idx){
        uint32_t i = previousActive[idx];
        float w = a.parameters->valueWeight[i];
        float delta = a.memory->td*previousActiveValue[idx]*a.memory->valueScale;
        a.parameters->valueWeight[i] = keepFinite(w, w + a.memory->valueAlpha[i]*delta);
    }
    trackFeatureUtility(a, previousActive, previousActiveValue);
    replaceLeastUsefulFeature(a);
}

void writePrimaryPolicy(Agent a){
    static uint32_t option;
    option = (option + 1u) % OAK_OPTION_COUNT;
    uint32_t off = option*OAK_POLICY_MAX;
    for (uint32_t action=0u; action<OAK_POLICY_MAX; ++action)
        a.memory->policyMeanRaw[action] = finiteOrZero(a.memory->optionLogit[off+action]);
}

void oakInit(Agent a, uint32_t gid, float initX, float initY, float initZ, float initW){
    for (uint32_t n=0u; n<OAK_FEATURE_TIMESCALE; ++n) g_invAge[n] = 1.0f/((float)n + 1.0f);
    for (uint32_t i=0u; i<OAK_FEATURE_CAPACITY; ++i) a.indices->featureRate[i] = trackingRate(0u);
    for (uint32_t m=1u; m<=OAK_TRACE_LEVELS; ++m) g_traceRate[m] = 1.0f/(float)(1u << m);
    for (uint32_t feature=0u; feature<OAK_FEATURE_CAPACITY; ++feature){
        float s = 1.0f/(float)(2u*OAK_PERCEPT_COUNT);
        f4 om = { s, logf(s), 0.0f, 0.0f };
        if (feature < a.observationCount) a.memory->observationMeta[feature]=om;
        a.memory->valueAlpha[feature]=initY; (void)initZ;
    }
    for (uint32_t option=0u; option<OAK_OPTION_COUNT; ++option){
        optionRecord(a, option, option)->value = initX;
        a.memory->actorMeta[option].x = 1.0f;
        uint32_t hash = gid*2654435761u + option*40503u + 977u, action = hash % OAK_ENV_ACT_MAX;
        if (action < OAK_POLICY_MAX)
            optionRecord(a, 0u, option)->policy[action] = ((hash>>16u)&1u)==0u ? initW : -initW;
    }
    a.memory->reward = 0.0f;
    for (uint32_t d=0u; d<OAK_ACTION_PARAMS; ++d) a.memory->actionSignal[d]=0.0f;
    for (uint32_t i=0u;i<OAK_OPTION_COUNT*OAK_POLICY_MAX;++i) a.memory->optionActorCredit[i]=0.0f;
    for (uint32_t k=0u;k<OAK_OPTION_COUNT;++k){
        a.memory->importance[k]=1.0f; a.memory->importanceStaging[k]=1.0f;
    }
}

static float softmaxWeights(const float *raw, uint32_t count, float *logit, float *weight){
    float maxLogit = -FLT_MAX;
    for (uint32_t d=0u; d<count; ++d){
        logit[d] = finiteOrZero(raw[d]);
        if (logit[d] > maxLogit) maxLogit = logit[d];
    }
    float sum = 0.0f;
    for (uint32_t d=0u; d<count; ++d){ weight[d] = expf(logit[d] - maxLogit); sum += weight[d]; }
    return sum;
}

uint32_t oakSampleDiscrete(Agent a, uint32_t discreteActionCount, const uint32_t *words){
    float logit[OAK_POLICY_MAX], weight[OAK_POLICY_MAX];
    float behaviorSoftmaxSum = softmaxWeights(a.memory->policyMeanRaw, discreteActionCount, logit, weight);
    uint32_t sampledAction = 0u;
    float bestGumbelScore = -FLT_MAX;
    for (uint32_t d=0u; d<discreteActionCount; ++d){
        float u = ((float)(words[d & 7u] >> 8) + 1.0f) * 0x1p-24f;
        if (u > 0x1.fffffep-1f) u = 0x1.fffffep-1f;
        float g = logit[d] - logf(-logf(u));
        if (g > bestGumbelScore){ bestGumbelScore = g; sampledAction = d; }
    }
    for (uint32_t d=0u; d<OAK_ACTION_PARAMS; ++d)
        a.memory->actionSignal[d] = (d < discreteActionCount)
            ? ((d == sampledAction) ? 1.0f : 0.0f) - weight[d]/behaviorSoftmaxSum
            : 0.0f;
    float behaviorActionProb = weight[sampledAction]/behaviorSoftmaxSum;
    for (uint32_t k=0u;k<OAK_OPTION_COUNT;++k){
        uint32_t off = k*OAK_POLICY_MAX;
        float optionLogit[OAK_POLICY_MAX], optionWeight[OAK_POLICY_MAX];
        float optionSoftmaxSum = softmaxWeights(a.memory->optionLogit + off, discreteActionCount,
                                                optionLogit, optionWeight);
        for (uint32_t d=0u;d<OAK_POLICY_MAX;++d)
            a.memory->actorCreditStaging[off+d] = (d < discreteActionCount)
                ? ((d==sampledAction)?1.0f:0.0f) - optionWeight[d]/optionSoftmaxSum
                : 0.0f;
        float ratio = (optionWeight[sampledAction]/optionSoftmaxSum)/behaviorActionProb;
        float capped = ratio < (float)OAK_OPTION_COUNT ? ratio : (float)OAK_OPTION_COUNT;
        a.memory->importanceStaging[k] = isfinite(ratio) ? capped : 0.0f;
    }
    return sampledAction;
}



static void headUpdate(float *w, f4 *meta, uint32_t *age, const float *pf, float target){
    float pred=0.0f, denom=1.0f;
    for (uint32_t z=0u;z<OAK_ACTIVE_CAPACITY;++z){
        uint32_t i=g_prevF[z]; float x=pf[i];
        pred += w[i]*x;
        denom += meta[i].x * x*x;
    }
    if (!isfinite(target - pred)) return;
    float err = target - pred, scale = (denom>0.0f)?1.0f/denom:0.0f;
    for (uint32_t z=0u;z<OAK_ACTIVE_CAPACITY;++z){
        uint32_t i=g_prevF[z]; float x=pf[i];
        float v = w[i] + meta[i].x*err*x*scale;
        if (isfinite(v)) w[i]=v;
        metaOptimizeUpdate(&meta[i], 0u, err, x, age[i]);
        age[i]++;
    }
}

void oakActionModelLearn(Agent a, float reward){
    if (!g_actInit){
        float a0 = 1.0f/(float)OAK_ACTIVE_CAPACITY;
        f4 m={a0,logf(a0),0.0f,0.0f};
        for (uint32_t i=0;i<OAK_FEATURE_CAPACITY;++i)
            for (uint32_t d=0;d<OAK_POLICY_MAX;++d){
                g_actQ[d][i]=0.0f; g_actAge[d][i]=0u; g_actMeta[d][i]=m;
            }
        g_actInit=1;
    }
    if (!g_havePrevAct) return;
    const float *cf = featureValues(a, a.stateSlot);
    const uint32_t *ca = activeFeatures(a, a.stateSlot);
    float vNext=0.0f;
    const float *cav = activeValues(a, a.stateSlot);
    for (uint32_t z=0;z<OAK_ACTIVE_CAPACITY;++z){ uint32_t i=ca[z]; vNext += a.parameters->valueWeight[i]*cav[z]; }
    const float *pf = featureValues(a, 1u - a.stateSlot);
    float centered = reward - a.memory->averageReward;
    headUpdate(g_actQ[g_prevAct], g_actMeta[g_prevAct], g_actAge[g_prevAct], pf, centered + vNext);
}
void oakActionModelRemember(Agent a, uint32_t action){
    memcpy(g_prevF, activeFeatures(a, a.stateSlot), sizeof g_prevF);
    g_prevAct=action; g_havePrevAct=1;
}

static float g_qVar, g_lVar;


void oakActionModelBiasScaleFree(Agent a){
    if (!g_actInit) return;
    const uint32_t actionCount = a.envActionCount;
    const uint32_t *ca = activeFeatures(a, a.stateSlot);
    const float *cav = activeValues(a, a.stateSlot);
    float q[OAK_POLICY_MAX], v[OAK_POLICY_MAX]; float qm=0.0f;
    for (uint32_t d=0;d<actionCount;++d) v[d]=0.0f;
    for (uint32_t z=0u;z<OAK_ACTIVE_CAPACITY;++z){
        uint32_t i=ca[z]; float x=cav[z];
        for (uint32_t d=0;d<actionCount;++d) v[d] += g_actQ[d][i]*x;
    }
    for (uint32_t d=0;d<actionCount;++d){
        q[d]=isfinite(v[d])?v[d]:0.0f;
        qm += q[d]/(float)actionCount;
    }
    float lm=0.0f;
    for (uint32_t d=0;d<actionCount;++d) lm += finiteOrZero(a.memory->policyMeanRaw[d])/(float)actionCount;
    float qs=0.0f, ls=0.0f;
    for (uint32_t d=0;d<actionCount;++d){
        float dq=q[d]-qm, dl=finiteOrZero(a.memory->policyMeanRaw[d])-lm;
        qs += dq*dq; ls += dl*dl;
    }
    g_qVar += OAK_TRACKING_RATE*(qs - g_qVar);

    g_lVar += OAK_TRACKING_RATE*(ls - g_lVar);
    float qsd = sqrtf(g_qVar > 1e-30f ? g_qVar : 1e-30f);
    float lsd = sqrtf(g_lVar > 1e-30f ? g_lVar : 1e-30f);
    float gain = lsd/qsd;
    if (!isfinite(gain)) return;

    for (uint32_t d=0;d<actionCount;++d)
        a.memory->policyMeanRaw[d] = finiteOrZero(gain*(q[d]-qm));
}

