#include <stdio.h>

#include "agent.h"

static float g_actQ[OAK_FEATURE_CAPACITY][OAK_POLICY_MAX];
static f4    g_actMeta[OAK_FEATURE_CAPACITY][OAK_POLICY_MAX];
static uint32_t g_actAge[OAK_FEATURE_CAPACITY][OAK_POLICY_MAX];
static int   g_actInit;
static uint32_t g_prevAct;
static uint32_t g_prevF[OAK_ACTIVE_CAPACITY];
static float g_prevV[OAK_ACTIVE_CAPACITY];
static int   g_havePrevAct;
static float g_stash[OAK_OPTION_COUNT * 5u];
static int   g_stashDirty;

static uint32_t g_traceMod;
typedef struct { uint16_t pa, pb, kind, trace; } CVPlan;
static CVPlan g_cv[OAK_FEATURE_CAPACITY];
static void buildPlans(Agent a){
    for (uint32_t i=0;i<OAK_FEATURE_CAPACITY;++i){
        u4 m = a.indices->featureMeta[i];
        uint32_t k = m.z % 4u;
        g_cv[i].kind  = (uint16_t)k;
        g_cv[i].trace = (uint16_t)(m.x % g_traceMod);
        g_cv[i].pa    = (uint16_t)(m.x % OAK_FEATURE_CAPACITY);
        g_cv[i].pb    = (uint16_t)(m.y % OAK_FEATURE_CAPACITY);
    }
}
static inline float constructedAt(Agent a, const float *features, uint32_t i){
    CVPlan p = g_cv[i];
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
    for (uint32_t o = 0u; o < a.observationCount; ++o){
        float x = a.memory->observation[o];
        if (isfinite(x)){
            float observationMean = a.memory->observationMean[o];
            uint32_t age = a.stepCount < OAK_FEATURE_CAPACITY ? a.stepCount : OAK_FEATURE_CAPACITY;
            metaOptimizeUpdate(a.memory->observationMeta, o, x - observationMean, 1.0f, age,
                               OAK_META_RATE, OAK_TRACKING_RATE);
            float r1 = 1.0f/((float)a.stepCount + 1.0f), r2 = a.memory->observationMeta[o].x;
            float obsRate = r1 > r2 ? r1 : r2;
            a.memory->observationMean[o] += obsRate * (x - observationMean);
            a.memory->observationSecondMoment[o] +=
                obsRate * (x*x - a.memory->observationSecondMoment[o]);
            standardized[o] = normalizedObservation(a, o);
            for (uint32_t m = 1u; m <= OAK_TRACE_LEVELS; ++m){
                uint32_t slot = (m-1u)*a.observationCount + o;
                float rate = 1.0f/(float)(1u << m);
                a.memory->observationTimescale[slot] +=
                    rate * (standardized[o] - a.memory->observationTimescale[slot]);
            }
        } else {
            standardized[o] = normalizedObservation(a, o);
        }
    }
    for (uint32_t k = 0u; k < OAK_OPTION_COUNT; ++k)
        a.memory->importance[k] = a.memory->importanceStaging[k];
    for (uint32_t j = 0u; j < OAK_OPTION_COUNT*OAK_POLICY_MAX; ++j)
        a.memory->optionActorCredit[j] = a.memory->actorCreditStaging[j];

    const float *previousFeatures = featureValues(a, a.stateSlot);
    float *currentFeatures = featureValues(a, 1u - a.stateSlot);
    uint32_t *currentActive = activeFeatures(a, 1u - a.stateSlot);

    for (uint32_t i = 0u; i < OAK_PERCEPT_COUNT; ++i){
        float x;
        if (i < a.observationCount) x = standardized[i];
        else if (i < a.observationCount + OAK_ACTION_PARAMS) x = a.memory->actionSignal[i - a.observationCount];
        else if (i == a.observationCount + OAK_ACTION_PARAMS) x = a.memory->reward;
        else if (i == a.observationCount + OAK_ACTION_PARAMS + 1u) x = 1.0f;
        else {
            uint32_t j = i - (a.observationCount + OAK_ACTION_PARAMS + 2u);
            uint32_t oc = a.observationCount;
            uint32_t lvl = j / oc;
            x = (lvl < OAK_TRACE_LEVELS)
                ? a.memory->observationTimescale[lvl*oc + (j - lvl*oc)]
                : 0.0f;
        }
        x = finiteOrZero(x);
        currentFeatures[i] = x;
        currentActive[i] = i;
        if (a.indices->featureMeta[i].w < 0xffffffffu) a.indices->featureMeta[i].w += 1u;
    }

    uint32_t sourceCount = perceptSourceCount(a);
    for (uint32_t s = 0u; s < OAK_CONSTRUCTED_SLOTS; ++s){
        uint32_t sourceIndex = s % sourceCount, occupancy = s / sourceCount;
        uint32_t tile;
        if (occupancy == 0u)
#ifdef OAK_OBS0ZOOM
            tile = zoomedFeatureTile(currentFeatures[tileSourceIndex(a, sourceIndex)],
                                     (s < a.observationCount) ? (OAK_OBS0ZOOM) : 0u);
#else
            tile = zoomedFeatureTile(currentFeatures[tileSourceIndex(a, sourceIndex)], 0u);
#endif
        else if ((s & 1u) == 0u)
            tile = zoomedFeatureTile(currentFeatures[tileSourceIndex(a, sourceIndex)], occupancy);
        else {
            uint32_t chA = tileSourceIndex(a, sourceIndex);
            uint32_t chB = (sourceIndex + occupancy) % a.observationCount;
            tile = jointFeatureTile(currentFeatures[chA], currentFeatures[chB], occupancy);
        }
        uint32_t bankLo = OAK_PERCEPT_COUNT + s*OAK_CANDIDATES_PER_SLOT;
        uint32_t bankHi = bankLo + OAK_CANDIDATES_PER_SLOT;
        uint32_t winner = bankLo + tile;
        float winningValue = 0.0f;
#ifdef OAK_WINNERONLY
        {   winningValue = constructedAt(a, previousFeatures, winner);
            float rr = 1.0f/((float)a.indices->featureMeta[winner].w + 1.0f);
            float rate = rr > OAK_TRACKING_RATE ? rr : OAK_TRACKING_RATE;
            f2 moments = a.memory->featureMoments[winner];
            moments.x += rate * (winningValue - moments.x);
            moments.y += rate * (winningValue*winningValue - moments.y);
            a.memory->featureMoments[winner] = moments;
        }
#else
        for (uint32_t i = bankLo; i < bankHi; ++i){
            float constructed = constructedAt(a, previousFeatures, i);
            float rr = 1.0f/((float)a.indices->featureMeta[i].w + 1.0f);
            float rate = rr > OAK_TRACKING_RATE ? rr : OAK_TRACKING_RATE;
            f2 moments = a.memory->featureMoments[i];
            moments.x += rate * (constructed - moments.x);
            moments.y += rate * (constructed*constructed - moments.y);
            a.memory->featureMoments[i] = moments;
            if (i == winner) winningValue = constructed;
        }
#endif
        currentFeatures[currentActive[OAK_PERCEPT_COUNT + s]] = 0.0f;
        f2 wm = a.memory->featureMoments[winner];
        float var = wm.y - wm.x*wm.x; if (var < 0.0f) var = 0.0f;
        float r1 = 1.0f/((float)a.indices->featureMeta[winner].w + 1.0f);
        float r2 = 1.0f/(float)(2u*OAK_PERCEPT_COUNT);
        float ridge = r1 > r2 ? r1 : r2;
        winningValue = (winningValue - wm.x) * rsqrtf_(var + ridge);
        currentFeatures[winner] = finiteOrZero(winningValue);
        currentActive[OAK_PERCEPT_COUNT + s] = winner;
        if (a.indices->featureMeta[winner].w < 0xffffffffu) a.indices->featureMeta[winner].w += 1u;
    }
}

static void storeLearnScales(Agent a){
    const uint32_t *previousActive = activeFeatures(a, a.stateSlot);
    const float *previousFeatures = featureValues(a, a.stateSlot);
    float valueStepDenominator = 1.0f, previousFeatureNormSq = 0.0f;
    for (uint32_t i = 0u; i < OAK_ACTIVE_CAPACITY; ++i){
        uint32_t pi = previousActive[i];
        float x = previousFeatures[pi];
        float alpha = a.memory->valueMeta[pi].x;
        valueStepDenominator += alpha * (x*x);
        previousFeatureNormSq = fmaf(x, x, previousFeatureNormSq);
    }
    a.memory->valueScale = (isfinite(valueStepDenominator) && valueStepDenominator > 0.0f)
                           ? (1.0f/valueStepDenominator) : 0.0f;
    a.memory->previousFeatureNormSq = previousFeatureNormSq;
}

void predictAndControl(Agent a){
    const float *previousFeatures = featureValues(a, a.stateSlot);
    const float *currentFeatures = featureValues(a, 1u - a.stateSlot);
    const uint32_t *previousActive = activeFeatures(a, a.stateSlot);
    const uint32_t *currentActive = activeFeatures(a, 1u - a.stateSlot);
    float previousValue=0.0f, currentValue=0.0f;
    float optionValuePrevEnergy=0.0f, optionValueCrossEnergy=0.0f, optionValueNextEnergy=0.0f;
    float modelPrev=0.0f, modelCross=0.0f, modelNext=0.0f;
    for (uint32_t i = 0u; i < OAK_ACTIVE_CAPACITY; ++i){
        uint32_t pi = previousActive[i], ci = currentActive[i];
        float pa = previousFeatures[pi], ca = currentFeatures[ci];
        previousValue = fmaf(a.parameters->valueWeight[pi], pa, previousValue);
        currentValue  = fmaf(a.parameters->valueWeight[ci], ca, currentValue);
        float ap = a.memory->optionValueMeta[pi].x;
        optionValuePrevEnergy  = fmaf(ap*pa, pa, optionValuePrevEnergy);
        optionValueCrossEnergy = fmaf(ap*pa, currentFeatures[pi], optionValueCrossEnergy);
        float acps = a.memory->optionValueMeta[ci].x;
        optionValueNextEnergy  = fmaf(acps*ca, ca, optionValueNextEnergy);
        float am = a.memory->modelMeta[pi].x;
        modelPrev  = fmaf(am*pa, pa, modelPrev);
        modelCross = fmaf(am*pa, currentFeatures[pi], modelCross);
        float amc = a.memory->modelMeta[ci].x;
        modelNext  = fmaf(amc*ca, ca, modelNext);
    }
    a.memory->currentValue = currentValue;
    a.memory->previousModelEnergy = modelPrev;
    a.memory->crossModelEnergy = modelCross;
    a.memory->currentModelEnergy = modelNext;

    float gauge = a.memory->rewardCenter;
    float previousDifferentialValue = previousValue - gauge;
    float currentDifferentialValue  = currentValue  - gauge;
    float averageRewardOld = a.memory->averageReward;
    float centeredReward = a.memory->reward - averageRewardOld;
    {
        float delta = centeredReward + currentDifferentialValue - previousDifferentialValue;
        delta = finiteOrZero(delta);
        a.memory->td = delta;
        float nr = averageRewardOld + OAK_TRACKING_RATE*(a.memory->reward - averageRewardOld);
        a.memory->averageReward = isfinite(nr) ? nr : averageRewardOld;
    }
    float observedReward = finiteOrZero(a.memory->reward);

    for (uint32_t k = 0u; k < OAK_OPTION_COUNT; ++k){
        if (a.indices->parked[k] == 0u) continue;
        a.memory->continuation[k]=0.0f; a.memory->optionTd[k]=0.0f; a.memory->actorAdvantage[k]=0.0f;
    }
    for (uint32_t u = 0u; u < OAK_OPTION_UPDATES; ++u){
        uint32_t k = updatedOption(a.stepCount, u, u % OAK_SIMD_WIDTH);
        int slotLive = a.indices->parked[k] == 0u;
        uint32_t slot = k;
        float rewardHat=0.0f, durationHat=0.0f, nextValueHat=0.0f;
        float optionValuePrev=0.0f, optionValueNext=0.0f;
        float previousNextValue=0.0f, previousRewardHat=0.0f,
              previousDurationHat=0.0f;
        float policyLogit[OAK_POLICY_MAX];
        for (uint32_t j=0u;j<OAK_POLICY_MAX;++j) policyLogit[j]=0.0f;
        for (uint32_t idx = 0u; idx < OAK_ACTIVE_CAPACITY; ++idx){
            uint32_t pi = previousActive[idx], ci = currentActive[idx];
            float pa = previousFeatures[pi], ca = currentFeatures[ci];
            if (slotLive && pa != 0.0f){
                const Record *rp = optionRecord(a, pi, slot);
                optionValuePrev        = fmaf(rp->value, pa, optionValuePrev);
                previousNextValue      = fmaf(rp->next, pa, previousNextValue);
                previousRewardHat      = fmaf(rp->reward, pa, previousRewardHat);
                previousDurationHat    = fmaf(rp->duration, pa, previousDurationHat);
            }
            if (slotLive && ca != 0.0f){
                const Record *rc = optionRecord(a, ci, slot);
                optionValueNext = fmaf(rc->value, ca, optionValueNext);
                rewardHat       = fmaf(rc->reward, ca, rewardHat);
                durationHat     = fmaf(rc->duration, ca, durationHat);
                nextValueHat    = fmaf(rc->next, ca, nextValueHat);
                for (uint32_t j=0u;j<OAK_POLICY_MAX;++j)
                    policyLogit[j] = fmaf(rc->policy[j], ca, policyLogit[j]);
            }
        }
        if (!slotLive) continue;
        uint32_t optionFeatureIndex = a.indices->optionFeature[k] % OAK_FEATURE_CAPACITY;
        g_stash[k*5u+0u]=previousNextValue;
        g_stash[k*5u+2u]=previousRewardHat;
        g_stash[k*5u+3u]=previousDurationHat;
        for (uint32_t j=0u;j<OAK_POLICY_MAX;++j)
            a.memory->optionLogit[k*OAK_POLICY_MAX+j]=policyLogit[j];

        float stoppingBonus = 0.0f;
        float stoppingValue = currentDifferentialValue
                            + (1.0f + stoppingBonus)*currentFeatures[optionFeatureIndex];
        int subtaskFinite = isfinite(optionValuePrev) && isfinite(optionValueNext) && isfinite(stoppingValue);
        float stop = (isfinite(optionValueNext) && isfinite(stoppingValue) && stoppingValue >= optionValueNext)
                     ? 1.0f : 0.0f;
        if (k == 0u) stop = 0.0f;
        float continuation = 1.0f - stop;
        a.memory->continuation[k] = continuation;

        float optionAverageRewardOld = a.memory->optionAverageReward[k];
        float optionTdError = subtaskFinite
            ? (centeredReward - optionAverageRewardOld + continuation*optionValueNext
               + stop*stoppingValue - optionValuePrev) : 0.0f;
        optionTdError = finiteOrZero(optionTdError);
        if (k == 0u) optionTdError = a.memory->td - optionAverageRewardOld;
        a.memory->optionTd[k] = optionTdError;
        a.memory->actorAdvantage[k] = finiteOrZero(optionTdError);
        a.memory->optionAverageReward[k] =
            keepFinite(optionAverageRewardOld, optionAverageRewardOld + OAK_TRACKING_RATE*optionTdError);
        float importanceRatio = a.memory->importance[k];
        {
            float prev = a.memory->optionFitness[k];
            a.memory->optionFitness[k] = keepFinite(prev,
                prev + OAK_TRACKING_RATE*importanceRatio*(observedReward - prev));
        }
        float bootstrapEnergy = optionValuePrevEnergy - 2.0f*continuation*optionValueCrossEnergy
                              + continuation*continuation*optionValueNextEnergy;
        float denom = fmaf(importanceRatio, bootstrapEnergy, 1.0f);
        a.memory->optionValueScale[k] = (isfinite(denom) && denom > 0.0f) ? (1.0f/denom) : 0.0f;
        float auxDenom = fmaf(importanceRatio, optionValuePrevEnergy, 1.0f);
        a.memory->predictedReward[k]=rewardHat;
        a.memory->predictedDuration[k]=durationHat;
        a.memory->predictedNextValue[k]=nextValueHat;
    }
    {
        float gaugeRate = trackingRateFor(a.stepCount, OAK_FEATURE_CAPACITY);
        float gaugeNew = a.memory->rewardCenter + gaugeRate*(currentValue - a.memory->rewardCenter);
        a.memory->rewardCenter = keepFinite(a.memory->rewardCenter, gaugeNew);
    }
    storeLearnScales(a);
}

static void planOption(Agent a, uint32_t k, float previousNextValueDot, float optionNextValueDot,
                       float previousRewardHat, float previousDurationHat){
    float continuation = a.memory->continuation[k];
    float rewardHat = a.memory->predictedReward[k], durationHat = a.memory->predictedDuration[k],
          currentNextValueHat = a.memory->predictedNextValue[k];
    float currentValue = a.memory->currentValue;
    float modelPrevEnergy = a.memory->previousModelEnergy;
    float modelBootstrapNorm = a.memory->previousModelEnergy
        - 2.0f*continuation*a.memory->crossModelEnergy
        + continuation*continuation*a.memory->currentModelEnergy;
    if (modelBootstrapNorm < 0.0f) modelBootstrapNorm = 0.0f;
    modelBootstrapNorm = finiteOrZero(modelBootstrapNorm);
    if (!isfinite(modelPrevEnergy) || modelPrevEnergy < 0.0f) modelPrevEnergy = 0.0f;
    float modelAverageRewardOld = finiteOrZero(a.memory->optionModelAverageReward[k]);
    float rewardModelTd = (k==0u) ? 0.0f
        : ((a.memory->planningReward - modelAverageRewardOld) + continuation*rewardHat - previousRewardHat);
    float importanceRatio = a.memory->importance[k];
    if (isfinite(a.memory->planningReward))
        a.memory->optionModelAverageReward[k] = 0.0f;
    float durationTd = (k==0u) ? 0.0f : (1.0f + continuation*durationHat - previousDurationHat);
    float optionNextValueTd = continuation*currentNextValueHat
                            + (1.0f - continuation)*currentValue - previousNextValueDot;
    int modelFinite = isfinite(rewardModelTd) && isfinite(durationTd) && isfinite(durationHat)
                   && isfinite(optionNextValueTd);
    rewardModelTd = finiteOrZero(rewardModelTd);
    durationTd = finiteOrZero(durationTd);
    optionNextValueTd = finiteOrZero(optionNextValueTd);
    float mse = optionNextValueTd*optionNextValueTd + rewardModelTd*rewardModelTd;
    if (modelFinite && isfinite(mse)){
        a.memory->optionPriority[k] = sqrtf(mse > 0.0f ? mse : 0.0f);
        a.memory->optionPriority[k] = 0.0f;
    } else {
        float debt = fabsf(a.memory->optionPriority[k]);
        if (debt <= 0.0f) debt = 1.0f;
        a.memory->optionPriority[k] = -debt;
    }
}

void planAllOptions(Agent a, uint32_t previousSlot){
    const uint32_t *previousActive = activeFeatures(a, previousSlot);
    const float *previousFeatures = featureValues(a, previousSlot);
    for (uint32_t k = 0u; k < OAK_OPTION_COUNT; ++k){
        if (a.indices->parked[k] == 0u) continue;
        a.memory->optionPriority[k]=0.0f;
    }
    for (uint32_t u = 0u; u < OAK_OPTION_UPDATES; ++u){
        uint32_t k = updatedOption(a.stepCount - 1u, u, u % OAK_SIMD_WIDTH);
        if (a.indices->parked[k] != 0u) continue;
        uint32_t slot = k;
        float p0,p1,p2,p3;
        if (g_stashDirty){
            p0=p1=p2=p3=0.0f;
            for (uint32_t idx=0u; idx<OAK_ACTIVE_CAPACITY; ++idx){
                uint32_t pi = previousActive[idx];
                float pa = previousFeatures[pi];
                if (pa == 0.0f) continue;
                const Record *rp = optionRecord(a, pi, slot);
                p0=fmaf(rp->next,pa,p0); p1=fmaf(rp->nextAux,pa,p1);
                p2=fmaf(rp->reward,pa,p2); p3=fmaf(rp->duration,pa,p3);
            }
        } else {
            p0=g_stash[k*5u+0u]; p1=g_stash[k*5u+1u]; p2=g_stash[k*5u+2u]; p3=g_stash[k*5u+3u];
        }
        planOption(a, k, p0, p1, p2, p3);
        float importanceRatio = a.memory->importance[k];
        float advantage = finiteOrZero(a.memory->actorAdvantage[k]);
        metaOptimizeUpdate(a.memory->actorMeta, k, advantage, 1.0f, a.indices->optionAge[k],
                           OAK_META_RATE, OAK_TRACKING_RATE);
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

void learnFromTransition(Agent a, const float *previousFeatures, const float *currentFeatures,
                         uint32_t previousSlot, uint32_t policyN){
    const uint32_t *previousActive = activeFeatures(a, previousSlot);
    const uint32_t activeFeatureCount = OAK_ACTIVE_CAPACITY;
    for (uint32_t sb = 0u; sb < OAK_OPTION_UPDATES; ++sb){
        uint32_t slot = updatedOption(a.stepCount - 1u, sb, sb % OAK_SIMD_WIDTH);
        uint32_t k = slot;
        int optionLive = a.indices->parked[k] == 0u;
        float importanceRatio=0.0f, continuation=0.0f;

        float actorCoefficient[OAK_POLICY_MAX];
        for (uint32_t j=0u;j<policyN;++j) actorCoefficient[j]=0.0f;
        float optionValueStepScale=0.0f, td=0.0f;
        if (optionLive){
            importanceRatio = a.memory->importance[k];
            continuation = a.memory->continuation[k];
            optionValueStepScale = a.memory->optionValueScale[k];
            td = a.memory->optionTd[k];
            for (uint32_t j=0u;j<policyN;++j)
                actorCoefficient[j]=a.memory->optionPolicyCoef[k*OAK_POLICY_MAX+j];
        }
        int baseOption = (k == 0u);
        for (uint32_t idx=0u; idx<activeFeatureCount; ++idx){
            uint32_t i = previousActive[idx];
            float previousActivation = previousFeatures[i], currentActivation = currentFeatures[i];
            if (previousActivation == 0.0f && currentActivation == 0.0f) continue;
            if (!optionLive) continue;
            Record *recv = optionRecord(a, i, slot);
            Record rec = *recv;
            for (uint32_t j=0u;j<policyN;++j){
                float w = rec.policy[j];
                rec.policy[j] = keepFinite(w, w + actorCoefficient[j]*previousActivation);
            }
            if (!baseOption){
                float g = (continuation > 0.0f) ? 0.0f : (td*previousActivation);
                float value = rec.value;
                rec.value = 0.0f; (void)value; (void)g;
            }
            float next=rec.next, duration=rec.duration,
                  rewardWeight=rec.reward;
            rec.next = 0.0f; (void)next;
            rec.duration = 0.0f; (void)duration;
            rec.reward = 0.0f; (void)rewardWeight;
            *recv = rec;
        }
    }

    float valueTdError = a.memory->td;
    float valueStepScale = a.memory->valueScale;
    for (uint32_t idx=0u; idx<activeFeatureCount; ++idx){
        uint32_t i = previousActive[idx];
        float w = a.parameters->valueWeight[i];
        float delta = valueTdError*previousFeatures[i]*valueStepScale;
        a.parameters->valueWeight[i] = keepFinite(w, w + a.memory->valueMeta[i].x*delta);
    }
}

void writePrimaryPolicy(Agent a){
#ifdef OAK_GPI

    uint32_t best = 0u; float bestV = -3.402823466e38f; int found = 0;
    for (uint32_t k=0u;k<OAK_OPTION_COUNT;++k){
        if (a.indices->parked[k]!=0u) continue;
        float q = a.memory->predictedReward[k] + a.memory->predictedNextValue[k];
        if (!isfinite(q)) continue;
        if (!found || q > bestV){ bestV=q; best=k; found=1; }
    }
    uint32_t off = best*OAK_POLICY_MAX;
    for (uint32_t action=0u; action<OAK_POLICY_MAX; ++action){
        a.memory->policyMeanRaw[action]= finiteOrZero(a.memory->optionLogit[off+action]);
    }
#else
    for (uint32_t action=0u; action<OAK_POLICY_MAX; ++action){
        a.memory->policyMeanRaw[action]=finiteOrZero(a.memory->optionLogit[action]);
    }
#endif
}

void oakInit(Agent a, uint32_t gid, float initX, float initY, float initZ, float initW){
    uint32_t traceChannels = OAK_TRACE_LEVELS*a.observationCount;
    if (traceChannels < 1u) traceChannels = 1u;
    for (uint32_t feature=0u; feature<OAK_FEATURE_CAPACITY; ++feature){
        if (feature >= OAK_PERCEPT_COUNT){
            uint32_t source = feature - OAK_PERCEPT_COUNT;
            u4 fm;
            if (source < traceChannels){ fm.x=source; fm.y=0u; fm.z=3u; fm.w=0u; }
            else { fm.x=source % OAK_PERCEPT_COUNT; fm.y=(source*7u+3u)%OAK_PERCEPT_COUNT;
                   fm.z=source % 3u; fm.w=0u; }
            a.indices->featureMeta[feature]=fm;
        }
        if (feature < a.observationCount){
            float s = 1.0f/(float)(2u*OAK_PERCEPT_COUNT);
            f4 om = { s, logf(s), 0.0f, 0.0f };
            a.memory->observationMeta[feature]=om;
        }
        f4 meta = { initY, initZ, 0.0f, 0.0f };
        a.memory->valueMeta[feature]=meta;
        a.memory->optionValueMeta[feature]=meta;
        a.memory->modelMeta[feature]=meta;
    }
    for (uint32_t option=0u; option<OAK_OPTION_COUNT; ++option){
        a.indices->optionFeature[option]=option;
        optionRecord(a, option, option)->value = initX;
        a.memory->actorMeta[option].x = 1.0f;
        uint32_t hash = gid*2654435761u + option*40503u + 977u;
        uint32_t action = hash % OAK_POLICY_MAX;
        float sign = ((hash >> 16u) & 1u)==0u ? 1.0f : -1.0f;
        optionRecord(a, 0u, option)->policy[action] = sign*initW;
    }
    a.memory->reward = 0.0f;
    for (uint32_t d=0u; d<OAK_ACTION_PARAMS; ++d) a.memory->actionSignal[d]=0.0f;
    for (uint32_t i=0u;i<OAK_OPTION_COUNT*OAK_POLICY_MAX;++i) a.memory->optionActorCredit[i]=0.0f;
    for (uint32_t k=0u;k<OAK_OPTION_COUNT;++k){
        a.memory->importance[k]=1.0f; a.memory->importanceStaging[k]=1.0f; a.indices->parked[k]=0u;
    }
}

uint32_t oakSampleDiscrete(Agent a, uint32_t discreteActionCount, const uint32_t *words){
    float behaviorLogitMax = -3.402823466e38f;
    for (uint32_t d=0u; d<discreteActionCount; ++d){
        float v = finiteOrZero(a.memory->policyMeanRaw[d]);
        if (v > behaviorLogitMax) behaviorLogitMax = v;
    }
    float behaviorSoftmaxSum = 0.0f;
    for (uint32_t d=0u; d<discreteActionCount; ++d)
        behaviorSoftmaxSum += expf(finiteOrZero(a.memory->policyMeanRaw[d]) - behaviorLogitMax);
    uint32_t sampledAction = 0u;
    float bestGumbelScore = -3.402823466e38f;
    for (uint32_t d=0u; d<discreteActionCount; ++d){
        float mu = finiteOrZero(a.memory->policyMeanRaw[d]);
        float u = ((float)(words[d & 7u] >> 8) + 1.0f) * 0x1p-24f;
        if (u > 0x1.fffffep-1f) u = 0x1.fffffep-1f;
        float g = mu - logf(-logf(u));
        if (g > bestGumbelScore){ bestGumbelScore = g; sampledAction = d; }
    }
    for (uint32_t d=0u; d<OAK_ACTION_PARAMS; ++d){
        float s = 0.0f;
        if (d < discreteActionCount)
            s = ((d == sampledAction) ? 1.0f : 0.0f)
              - expf(finiteOrZero(a.memory->policyMeanRaw[d]) - behaviorLogitMax)/behaviorSoftmaxSum;
        a.memory->actionSignal[d]=s;
    }
    for (uint32_t k=0u;k<OAK_OPTION_COUNT;++k){
        uint32_t off = k*OAK_POLICY_MAX;
        float optionLogitMax = -3.402823466e38f;
        for (uint32_t d=0u;d<discreteActionCount;++d){
            float v = finiteOrZero(a.memory->optionLogit[off+d]);
            if (v > optionLogitMax) optionLogitMax = v;
        }
        float optionSoftmaxSum = 0.0f;
        for (uint32_t d=0u;d<discreteActionCount;++d)
            optionSoftmaxSum += expf(finiteOrZero(a.memory->optionLogit[off+d]) - optionLogitMax);
        for (uint32_t d=0u;d<OAK_POLICY_MAX;++d){
            float credit = 0.0f;
            if (d < discreteActionCount)
                credit = ((d==sampledAction)?1.0f:0.0f)
                    - expf(finiteOrZero(a.memory->optionLogit[off+d]) - optionLogitMax)/optionSoftmaxSum;
            a.memory->actorCreditStaging[off+d]=credit;
        }
        float behaviorActionProb =
            expf(finiteOrZero(a.memory->policyMeanRaw[sampledAction]) - behaviorLogitMax)/behaviorSoftmaxSum;
        float optionActionProb =
            expf(finiteOrZero(a.memory->optionLogit[off+sampledAction]) - optionLogitMax)/optionSoftmaxSum;
        float ratio = optionActionProb/behaviorActionProb;
        float capped = ratio < (float)OAK_OPTION_COUNT ? ratio : (float)OAK_OPTION_COUNT;
        a.memory->importanceStaging[k] = isfinite(ratio) ? capped : 0.0f;
    }
    return sampledAction;
}

void oakSetStashDirty(int d){ g_stashDirty = d; }

#ifdef OAK_ALIGN
#define OAK_PX(z,i) g_prevV[z]
#else
#define OAK_PX(z,i) pf[i]
#define MZ0 0u
#endif
static void headUpdate(float (*w)[OAK_POLICY_MAX], f4 (*meta)[OAK_POLICY_MAX],
                       uint32_t (*age)[OAK_POLICY_MAX], const float *pf, uint32_t act, float target){
    float pred=0.0f, denom=1.0f;
    for (uint32_t z=0u;z<OAK_ACTIVE_CAPACITY;++z){
        uint32_t i=g_prevF[z]; float x=OAK_PX(z,i);
        pred += w[i][act]*x;
        denom += meta[i][act].x * x*x;
    }
    float err = target - pred;
    if (!isfinite(err)) return;
    float scale = (denom>0.0f)?1.0f/denom:0.0f;
    for (uint32_t z=0u;z<OAK_ACTIVE_CAPACITY;++z){
        uint32_t i=g_prevF[z]; float x=OAK_PX(z,i);
        float v = w[i][act] + meta[i][act].x*err*x*scale;
        if (isfinite(v)) w[i][act]=v;
        metaOptimizeUpdate(&meta[i][act], 0u, err, x, age[i][act], OAK_META_RATE, OAK_TRACKING_RATE);
        age[i][act]++;
    }
}

void oakActionModelLearn(Agent a, float reward){
    if (!g_actInit){
        float a0 = (OAK_MODEL_STEP);
        f4 m={a0,logf(a0),0.0f,0.0f};
        for (uint32_t i=0;i<OAK_FEATURE_CAPACITY;++i)
            for (uint32_t d=0;d<OAK_POLICY_MAX;++d){
                g_actQ[i][d]=0.0f; g_actAge[i][d]=0u; g_actMeta[i][d]=m;
            }
        g_actInit=1;
    }
    if (!g_havePrevAct) return;
    const float *cf = featureValues(a, a.stateSlot);
    const uint32_t *ca = activeFeatures(a, a.stateSlot);
    float vNext=0.0f;
    for (uint32_t z=0;z<OAK_ACTIVE_CAPACITY;++z){ uint32_t i=ca[z]; vNext += a.parameters->valueWeight[i]*cf[i]; }
    const float *pf = featureValues(a, 1u - a.stateSlot);
    float centered = reward - a.memory->averageReward;
#ifdef OAK_ADVANTAGE
    float vPrev=0.0f;
    for (uint32_t z=0;z<OAK_ACTIVE_CAPACITY;++z){ uint32_t i=g_prevF[z]; vPrev += a.parameters->valueWeight[i]*g_prevV[z]; }
#endif
#ifdef OAK_ADVANTAGE
    headUpdate(g_actQ, g_actMeta, g_actAge, pf, g_prevAct, centered + vNext - vPrev);
#else
    headUpdate(g_actQ, g_actMeta, g_actAge, pf, g_prevAct, centered + vNext);
#endif
}
void oakActionModelRemember(Agent a, uint32_t action){
    const uint32_t *ca = activeFeatures(a, a.stateSlot);
    const float *cv = featureValues(a, a.stateSlot);
    for (uint32_t z=0;z<OAK_ACTIVE_CAPACITY;++z){ g_prevF[z]=ca[z]; g_prevV[z]=cv[ca[z]]; }
    g_prevAct=action; g_havePrevAct=1;
}

static float g_qVar, g_lVar;

#define OAK_NACT (a.envActionCount)

float oakSlot0Q(Agent a){
    const uint32_t *ca = activeFeatures(a, a.stateSlot);
    const float *cf = featureValues(a, a.stateSlot);
    uint32_t w = ca[OAK_PERCEPT_COUNT];
    return g_actQ[w][0] * cf[w];
}
uint32_t oakSlot0Idx(Agent a){ return activeFeatures(a, a.stateSlot)[OAK_PERCEPT_COUNT]; }

float oakValueNow(Agent a){
    const uint32_t *ca = activeFeatures(a, a.stateSlot);
    const float *cf = featureValues(a, a.stateSlot);
    float v=0.0f;
    for (uint32_t z=0;z<OAK_ACTIVE_CAPACITY;++z){ uint32_t i=ca[z]; v += a.parameters->valueWeight[i]*cf[i]; }
    return v;
}
void oakModelQ(Agent a, float *out){
    const uint32_t *ca = activeFeatures(a, a.stateSlot);
    const float *cf = featureValues(a, a.stateSlot);
    for (uint32_t d=0;d<3;++d){ float v=0.0f;
        for (uint32_t z=0u;z<OAK_ACTIVE_CAPACITY;++z){ uint32_t i=ca[z]; v += g_actQ[i][d]*cf[i]; }
        out[d]=v; }
}

void oakDriftReport(Agent a, int seg){
    const uint32_t *ca = activeFeatures(a, a.stateSlot);
    double vw=0, aq=0, fv=0, fm=0; float mx=0;
    for (uint32_t z=0;z<OAK_ACTIVE_CAPACITY;++z){ uint32_t i=ca[z];
        vw += fabsf(a.parameters->valueWeight[i]);
        for (uint32_t d=0;d<3;++d) aq += fabsf(g_actQ[i][d]);
        float x=fabsf(featureValues(a,a.stateSlot)[i]); fv += x; if(x>mx) mx=x;
        fm += fabsf(a.memory->featureMoments[i].x);
    }
    uint32_t n=OAK_ACTIVE_CAPACITY;
    printf("    seg%d |w_V| %.4f  |w_Q| %.4f  |phi| %.3f (max %.1f)  |mom| %.3f  qVar %.3g  rbar %+.6f\n",
           seg, vw/n, aq/(3*n), fv/n, mx, fm/n, g_qVar, a.memory->averageReward);
}

void oakStepSizeReport(Agent a){
    const uint32_t *ca = activeFeatures(a, a.stateSlot);
    float v[OAK_ACTIVE_CAPACITY], q[OAK_ACTIVE_CAPACITY];
    for (uint32_t z=0;z<OAK_ACTIVE_CAPACITY;++z){ uint32_t i=ca[z];
        v[z]=a.memory->valueMeta[i].x; q[z]=g_actMeta[i][0].x; }
    for (uint32_t i=1;i<OAK_ACTIVE_CAPACITY;++i){ float t=v[i]; int j=i-1;
        while(j>=0&&v[j]>t){v[j+1]=v[j];j--;} v[j+1]=t; }
    for (uint32_t i=1;i<OAK_ACTIVE_CAPACITY;++i){ float t=q[i]; int j=i-1;
        while(j>=0&&q[j]>t){q[j+1]=q[j];j--;} q[j+1]=t; }
    uint32_t n=OAK_ACTIVE_CAPACITY;
    printf("  step sizes over %u active features (min/p25/median/p75/max)\n", n);
    printf("    critic  %.3g %.3g %.3g %.3g %.3g   spread %.1fx\n",
           v[0],v[n/4],v[n/2],v[3*n/4],v[n-1], v[n-1]/(v[0]>0?v[0]:1e-30f));
    printf("    model   %.3g %.3g %.3g %.3g %.3g   spread %.1fx\n",
           q[0],q[n/4],q[n/2],q[3*n/4],q[n-1], q[n-1]/(q[0]>0?q[0]:1e-30f));
}

void oakActionModelBiasScaleFree(Agent a){
    if (!g_actInit) return;
    const float *cf = featureValues(a, a.stateSlot);
    const uint32_t *ca = activeFeatures(a, a.stateSlot);
    float q[OAK_POLICY_MAX]; float qm=0.0f;
    for (uint32_t d=0;d<OAK_NACT;++d){
        float v=0.0f;
        for (uint32_t z=0u;z<OAK_ACTIVE_CAPACITY;++z){ uint32_t i=ca[z]; v += g_actQ[i][d]*cf[i]; }
        q[d]=isfinite(v)?v:0.0f;
        qm += q[d]/(float)OAK_NACT;
    }
    float lm=0.0f;
    for (uint32_t d=0;d<OAK_NACT;++d) lm += finiteOrZero(a.memory->policyMeanRaw[d])/(float)OAK_NACT;
    float qs=0.0f, ls=0.0f;
    for (uint32_t d=0;d<OAK_NACT;++d){
        float dq=q[d]-qm, dl=finiteOrZero(a.memory->policyMeanRaw[d])-lm;
        qs += dq*dq; ls += dl*dl;
    }
    g_qVar += OAK_TRACKING_RATE*(qs - g_qVar);

    g_lVar += OAK_TRACKING_RATE*(ls - g_lVar);
#ifdef OAK_LOCALNORM
    float qsd = sqrtf(qs > 1e-30f ? qs : 1e-30f);
#else
    float qsd = sqrtf(g_qVar > 1e-30f ? g_qVar : 1e-30f);
#endif
    float lsd = sqrtf(g_lVar > 1e-30f ? g_lVar : 1e-30f);
    float gain = lsd/qsd;
    if (!isfinite(gain)) return;

    for (uint32_t d=0;d<OAK_NACT;++d)
        a.memory->policyMeanRaw[d] = finiteOrZero(gain*(q[d]-qm));
}
#undef OAK_NACT

