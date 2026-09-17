
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

static float constructedValue(Agent a, const float *features, uint32_t parentA, uint32_t parentB,
                              uint32_t kind){
    kind %= 4u;
    if (kind == FEATURE_OBSERVATION_TIMESCALE)
        return a.memory->observationTimescale[parentA % (OAK_TRACE_LEVELS * OAK_TRACE_CHANNELS(a))];
    float x = features[parentA % OAK_FEATURE_CAPACITY], y = features[parentB % OAK_FEATURE_CAPACITY];
    return kind == FEATURE_PRODUCT ? x*y
         : kind == FEATURE_NORMALIZED_SUM ? (x+y)*0.7071067811865475f
         : x*(y-x);
}

void constructFeatures(Agent a){
    float standardized[OAK_OBSERVATION_CAPACITY];
    for (uint32_t o = 0u; o < OAK_TRACE_CHANNELS(a); ++o){
        float x = (o < a.observationCount) ? a.memory->observation[o] : a.memory->reward;
        if (isfinite(x)){
            float observationMean = a.memory->observationMean[o];
            uint32_t age = a.stepCount < OAK_FEATURE_CAPACITY ? a.stepCount : OAK_FEATURE_CAPACITY;
            metaOptimizeUpdate(a.memory->observationMeta, o, x - observationMean, 1.0f, age,
                               OAK_META_RATE, OAK_TRACKING_RATE);
            float r1 = 1.0f/((float)a.stepCount + 1.0f), r2 = a.memory->observationMeta[o].x;
#ifdef OAK_STABLE_PERCEPTION
            r2 = OAK_TRACKING_RATE*OAK_TRACKING_RATE;
#endif
            float obsRate = r1 > r2 ? r1 : r2;
            a.memory->observationMean[o] += obsRate * (x - observationMean);
            a.memory->observationSecondMoment[o] +=
                obsRate * (x*x - a.memory->observationSecondMoment[o]);
            standardized[o] = normalizedObservation(a, o);
            for (uint32_t m = 1u; m <= OAK_TRACE_LEVELS; ++m){
                uint32_t slot = (m-1u)*OAK_TRACE_CHANNELS(a) + o;
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
            uint32_t oc = OAK_TRACE_CHANNELS(a) > 1u ? OAK_TRACE_CHANNELS(a) : 1u;
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

    for (uint32_t s = 0u; s < OAK_CONSTRUCTED_SLOTS; ++s){
        uint32_t sourceCount = perceptSourceCount(a);
        uint32_t sourceIndex = s % sourceCount, occupancy = s / sourceCount;
        uint32_t tile;
        if (occupancy == 0u)
            tile = zoomedFeatureTile(currentFeatures[tileSourceIndex(a, sourceIndex)], 0u);
        else if ((s & 1u) == 0u)
            tile = zoomedFeatureTile(currentFeatures[tileSourceIndex(a, sourceIndex)], occupancy);
        else {
            uint32_t chA = tileSourceIndex(a, sourceIndex);
#ifdef OAK_JOINTALL
            uint32_t chB = (sourceIndex + occupancy) % OAK_PERCEPT_COUNT;
#else
            uint32_t chB = (sourceIndex + occupancy) % a.observationCount;
#endif
            tile = jointFeatureTile(currentFeatures[chA], currentFeatures[chB], occupancy);
        }
        uint32_t bankLo = OAK_PERCEPT_COUNT + s*OAK_CANDIDATES_PER_SLOT;
        uint32_t bankHi = bankLo + OAK_CANDIDATES_PER_SLOT;
        uint32_t winner = bankLo + tile;
        float winningValue = 0.0f;
        for (uint32_t i = bankLo; i < bankHi; ++i){
            u4 meta = a.indices->featureMeta[i];
            float constructed = constructedValue(a, previousFeatures, meta.x, meta.y, meta.z);
            float rr = 1.0f/((float)meta.w + 1.0f);
            float rate = rr > OAK_TRACKING_RATE ? rr : OAK_TRACKING_RATE;
            f2 moments = a.memory->featureMoments[i];
            moments.x += rate * (constructed - moments.x);
            moments.y += rate * (constructed*constructed - moments.y);
            a.memory->featureMoments[i] = moments;
            if (i == winner) winningValue = constructed;
            currentFeatures[i] = 0.0f;
        }
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
        float nr = averageRewardOld + OAK_TRACKING_RATE*delta;
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
        float optionValuePrev=0.0f, optionValueNext=0.0f, optionValueAuxDot=0.0f;
        float rewardHat=0.0f, durationHat=0.0f, nextValueHat=0.0f;
        float previousNextValue=0.0f, previousNextValueAuxDot=0.0f, previousRewardHat=0.0f,
              previousDurationHat=0.0f, twinPrevDot=0.0f;
        float policyLogit[OAK_POLICY_MAX];
        for (uint32_t j=0u;j<OAK_POLICY_MAX;++j) policyLogit[j]=0.0f;
        for (uint32_t idx = 0u; idx < OAK_ACTIVE_CAPACITY; ++idx){
            uint32_t pi = previousActive[idx], ci = currentActive[idx];
            float pa = previousFeatures[pi], ca = currentFeatures[ci];
            if (slotLive && pa != 0.0f){
                const Record *rp = optionRecord(a, pi, slot);
                optionValuePrev        = fmaf(rp->value, pa, optionValuePrev);
                optionValueAuxDot      = fmaf(rp->aux, pa, optionValueAuxDot);
                previousNextValue      = fmaf(rp->next, pa, previousNextValue);
                previousNextValueAuxDot= fmaf(rp->nextAux, pa, previousNextValueAuxDot);
                previousRewardHat      = fmaf(rp->reward, pa, previousRewardHat);
                previousDurationHat    = fmaf(rp->duration, pa, previousDurationHat);
                twinPrevDot            = fmaf(rp->twin, pa, twinPrevDot);
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
        g_stash[k*5u+1u]=previousNextValueAuxDot;
        g_stash[k*5u+2u]=previousRewardHat;
        g_stash[k*5u+3u]=previousDurationHat;
        g_stash[k*5u+4u]=twinPrevDot;
        for (uint32_t j=0u;j<OAK_POLICY_MAX;++j)
            a.memory->optionLogit[k*OAK_POLICY_MAX+j]=policyLogit[j];

        float stoppingBonus = a.memory->bonusWeight[optionFeatureIndex]
                            - a.parameters->valueWeight[optionFeatureIndex];
        if (!isfinite(stoppingBonus) || stoppingBonus < 0.0f) stoppingBonus = 0.0f;
        float stoppingValue = currentDifferentialValue
                            + (1.0f + stoppingBonus)*currentFeatures[optionFeatureIndex];
        int subtaskFinite = isfinite(optionValuePrev) && isfinite(optionValueNext) && isfinite(stoppingValue);
        float stop = (isfinite(optionValueNext) && isfinite(stoppingValue) && stoppingValue >= optionValueNext)
                     ? 1.0f : 0.0f;
        if (k == 0u) stop = 0.0f;
        float continuation = 1.0f - stop;
        a.memory->continuation[k] = continuation;
        if (stop > 0.0f) a.indices->lastStopAge[k] = 0u;

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
        a.memory->optionValueDot[k] = finiteOrZero(optionValueAuxDot);
        float auxDenom = fmaf(importanceRatio, optionValuePrevEnergy, 1.0f);
        a.memory->optionValueAuxScale[k] = (isfinite(auxDenom) && auxDenom > 0.0f) ? (1.0f/auxDenom) : 0.0f;
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
        a.memory->optionModelAverageReward[k] = modelAverageRewardOld
            + trackingRateFor(a.stepCount, OAK_FEATURE_CAPACITY)*importanceRatio
              *(a.memory->planningReward - modelAverageRewardOld);
    float durationTd = (k==0u) ? 0.0f : (1.0f + continuation*durationHat - previousDurationHat);
    float optionNextValueTd = continuation*currentNextValueHat
                            + (1.0f - continuation)*currentValue - previousNextValueDot;
    int modelFinite = isfinite(rewardModelTd) && isfinite(durationTd) && isfinite(durationHat)
                   && isfinite(optionNextValueTd);
    rewardModelTd = finiteOrZero(rewardModelTd);
    durationTd = finiteOrZero(durationTd);
    optionNextValueTd = finiteOrZero(optionNextValueTd);
    a.memory->optionRewardTd[k]=rewardModelTd;
    a.memory->optionDurationTd[k]=durationTd;
    a.memory->optionNextValueTd[k]=optionNextValueTd;
    a.memory->optionNextValueDot[k]=finiteOrZero(optionNextValueDot);
    float nvDenom = fmaf(importanceRatio, modelBootstrapNorm, 1.0f);
    a.memory->optionNextValueScale[k] = (isfinite(nvDenom) && nvDenom > 0.0f) ? (1.0f/nvDenom) : 0.0f;
    float nvAuxDenom = fmaf(importanceRatio, modelPrevEnergy, 1.0f);
    a.memory->optionNextValueAuxScale[k] = (isfinite(nvAuxDenom) && nvAuxDenom > 0.0f) ? (1.0f/nvAuxDenom) : 0.0f;
    float mse = optionNextValueTd*optionNextValueTd + rewardModelTd*rewardModelTd;
    if (modelFinite && isfinite(mse)){
        a.memory->optionPriority[k] = sqrtf(mse > 0.0f ? mse : 0.0f);
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
        a.memory->optionRewardTd[k]=0.0f; a.memory->optionDurationTd[k]=0.0f;
        a.memory->optionNextValueTd[k]=0.0f; a.memory->optionNextValueDot[k]=0.0f;
        a.memory->optionNextValueScale[k]=0.0f; a.memory->optionNextValueAuxScale[k]=0.0f;
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
    const uint32_t *currentActive  = activeFeatures(a, 1u - previousSlot);
    const uint32_t activeFeatureCount = OAK_ACTIVE_CAPACITY;
    for (uint32_t sb = 0u; sb < OAK_OPTION_UPDATES; ++sb){
        uint32_t slot = updatedOption(a.stepCount - 1u, sb, sb % OAK_SIMD_WIDTH);
        uint32_t k = slot;
        int optionLive = a.indices->parked[k] == 0u;
        float importanceRatio=0.0f, continuation=0.0f;
        float optionValueAuxDot=0.0f, optionValueStepScale=0.0f, td=0.0f;
        float nextValueAuxDot=0.0f, nextValueStepScale=0.0f, nextValueTdError=0.0f;
        float rewardModelTdError=0.0f, durationTdError=0.0f;
        float optionValueAuxResidual=0.0f, optionValueAuxStepScale=0.0f;
        float nextValueAuxResidual=0.0f, nextValueAuxStepScale=0.0f;
        float twinPrevDot=0.0f, twinResidual=0.0f;
        float actorCoefficient[OAK_POLICY_MAX];
        for (uint32_t j=0u;j<policyN;++j) actorCoefficient[j]=0.0f;
        {
            float dot=0.0f;
            if (g_stashDirty){
                for (uint32_t idx=0u; idx<activeFeatureCount; ++idx){
                    uint32_t i0 = previousActive[idx];
                    float x0 = previousFeatures[i0];
                    if (!optionLive || x0 == 0.0f) continue;
                    dot = fmaf(optionRecord(a,i0,slot)->twin, x0, dot);
                }
            } else if (optionLive) dot = g_stash[k*5u+4u];
            twinPrevDot = finiteOrZero(dot);
        }
        if (optionLive){
            importanceRatio = a.memory->importance[k];
            continuation = a.memory->continuation[k];
            optionValueAuxDot = a.memory->optionValueDot[k];
            optionValueStepScale = a.memory->optionValueScale[k];
            td = a.memory->optionTd[k];
            nextValueAuxDot = a.memory->optionNextValueDot[k];
            nextValueStepScale = a.memory->optionNextValueScale[k];
            nextValueTdError = a.memory->optionNextValueTd[k];
            rewardModelTdError = a.memory->optionRewardTd[k];
            durationTdError = a.memory->optionDurationTd[k];
            optionValueAuxResidual = td - optionValueAuxDot;
            optionValueAuxStepScale = a.memory->optionValueAuxScale[k];
            nextValueAuxResidual = nextValueTdError - nextValueAuxDot;
            nextValueAuxStepScale = a.memory->optionNextValueAuxScale[k];
            twinResidual = rewardModelTdError - twinPrevDot;
            for (uint32_t j=0u;j<policyN;++j)
                actorCoefficient[j]=a.memory->optionPolicyCoef[k*OAK_POLICY_MAX+j];
        }
        int baseOption = (k == 0u);
        for (uint32_t idx=0u; idx<activeFeatureCount; ++idx){
            uint32_t i = previousActive[idx];
            float previousActivation = previousFeatures[i], currentActivation = currentFeatures[i];
            if (previousActivation == 0.0f && currentActivation == 0.0f) continue;
            float aP = a.memory->optionValueMeta[i].x, aM = a.memory->modelMeta[i].x;
            if (!optionLive) continue;
            Record *recv = optionRecord(a, i, slot);
            Record rec = *recv;
            for (uint32_t j=0u;j<policyN;++j){
                float w = rec.policy[j];
#ifdef OAK_ACTOR_FORGET
                rec.policy[j] = keepFinite(w, w + actorCoefficient[j]*previousActivation
                                              - (OAK_ACTOR_FORGET)*w);
#else
                rec.policy[j] = keepFinite(w, w + actorCoefficient[j]*previousActivation);
#endif
            }
            if (!baseOption){
                float g = (continuation > 0.0f)
                    ? (optionValueAuxDot*(previousActivation - continuation*currentActivation))
                    : (td*previousActivation);
                float value = rec.value, aux = rec.aux;
                rec.value = keepFinite(value, value + aP*importanceRatio*g*optionValueStepScale);
                rec.aux   = keepFinite(aux, aux + aP*importanceRatio*optionValueAuxResidual
                                                  *previousActivation*optionValueAuxStepScale);
            }
            float nextValueGradient = (continuation > 0.0f)
                ? (nextValueAuxDot*(previousActivation - continuation*currentActivation))
                : (nextValueTdError*previousActivation);
            float next=rec.next, nextAux=rec.nextAux, duration=rec.duration,
                  rewardWeight=rec.reward, twin=rec.twin;
            rec.next = keepFinite(next, next + aM*importanceRatio*nextValueGradient*nextValueStepScale);
            rec.nextAux = keepFinite(nextAux, nextAux + aM*importanceRatio*nextValueAuxResidual
                                                        *previousActivation*nextValueAuxStepScale);
            rec.duration = keepFinite(duration, duration + aM*importanceRatio*durationTdError
                                                *previousActivation*nextValueStepScale*0.05f);
            float rewardModelGradient = (continuation > 0.0f)
                ? (twinPrevDot*(previousActivation - continuation*currentActivation))
                : (rewardModelTdError*previousActivation);
            rec.reward = keepFinite(rewardWeight, rewardWeight + aM*importanceRatio
                                                  *rewardModelGradient*nextValueStepScale);
            rec.twin = keepFinite(twin, twin + aM*importanceRatio*twinResidual
                                               *previousActivation*nextValueAuxStepScale);
            *recv = rec;
        }
        int learnsNewFeatures = optionLive && continuation > 0.0f && importanceRatio > 0.0f;
        for (uint32_t wb=0u; wb<OAK_CONSTRUCTED_SLOTS; ++wb){
            uint32_t i = currentActive[OAK_PERCEPT_COUNT + wb];
            if (wasActive(previousActive, OAK_PERCEPT_COUNT, i)) continue;
            float currentActivation = currentFeatures[i];
            float aP = a.memory->optionValueMeta[i].x, aM = a.memory->modelMeta[i].x;
            if (!learnsNewFeatures) continue;
            Record *rec = optionRecord(a, i, slot);
            if (!baseOption){
                float value = rec->value;
                rec->value = keepFinite(value, value + aP*importanceRatio
                    *(-optionValueAuxDot*continuation*currentActivation)*optionValueStepScale);
            }
            float next=rec->next, rewardWeight=rec->reward;
            rec->next = keepFinite(next, next + aM*importanceRatio
                *(-nextValueAuxDot*continuation*currentActivation)*nextValueStepScale);
            rec->reward = keepFinite(rewardWeight, rewardWeight + aM*importanceRatio
                *(-twinPrevDot*continuation*currentActivation)*nextValueStepScale);
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

void emitOptionActions(Agent a, int recomputeLogits){
    const uint32_t *active = activeFeatures(a, a.stateSlot);
    const float *currentFeatures = featureValues(a, a.stateSlot);
    for (uint32_t k=0u;k<OAK_OPTION_COUNT;++k){
        if (a.indices->parked[k]==0u) continue;
        for (uint32_t j=0u;j<OAK_POLICY_MAX;++j){
            a.memory->optionPolicyMean[k*OAK_POLICY_MAX+j]=0.0f;
            a.memory->optionLogit[k*OAK_POLICY_MAX+j]=0.0f;
        }
    }
    for (uint32_t k=0u;k<OAK_OPTION_COUNT;++k){
        int slotLive = a.indices->parked[k]==0u;
        float policyLogit[OAK_POLICY_MAX];
        for (uint32_t j=0u;j<OAK_POLICY_MAX;++j) policyLogit[j]=0.0f;
        if (recomputeLogits){
            for (uint32_t idx=0u; idx<OAK_ACTIVE_CAPACITY; ++idx){
                uint32_t i = active[idx];
                float x = currentFeatures[i];
                if (!slotLive || x == 0.0f) continue;
                const Record *row = optionRecord(a, i, k);
                for (uint32_t j=0u;j<OAK_POLICY_MAX;++j)
                    policyLogit[j]=fmaf(row->policy[j], x, policyLogit[j]);
            }
        } else if (slotLive){
            for (uint32_t j=0u;j<OAK_POLICY_MAX;++j)
                policyLogit[j]=a.memory->optionLogit[k*OAK_POLICY_MAX+j];
        }
        if (!slotLive) continue;
        for (uint32_t j=0u;j<OAK_POLICY_MAX;++j){
            uint32_t oi = k*OAK_POLICY_MAX+j;
            float actionMean = tanhf(policyLogit[j]);
            if (recomputeLogits) a.memory->optionLogit[oi]=policyLogit[j];
            if (!isfinite(actionMean)) actionMean = a.memory->optionPolicyMean[oi];
            a.memory->optionPolicyMean[oi]=actionMean;
        }
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
        a.memory->policyMean[action]   = a.memory->optionPolicyMean[off+action];
        a.memory->policyMeanRaw[action]= finiteOrZero(a.memory->optionLogit[off+action]);
    }
#else
    for (uint32_t action=0u; action<OAK_POLICY_MAX; ++action){
        a.memory->policyMean[action]=a.memory->optionPolicyMean[action];
        a.memory->policyMeanRaw[action]=finiteOrZero(a.memory->optionLogit[action]);
    }
#endif
}

void updateUtilities(Agent a){
    uint32_t tPrev = a.stepCount - 1u;
    const float *previousFeatures = featureValues(a, 1u - a.stateSlot);
    float trackingRate = trackingRateFor(tPrev, OAK_FEATURE_CAPACITY);
    const uint32_t *previousActive = activeFeatures(a, 1u - a.stateSlot);
    float valueTdError = a.memory->td;
    for (uint32_t i=0u;i<OAK_FEATURE_CAPACITY;++i) a.memory->optionTdByFeature[i]=0.0f;
    for (uint32_t k=0u;k<OAK_OPTION_COUNT;++k){
        float use = fabsf(a.memory->optionTd[k]);
        if (isfinite(use))
            a.memory->optionTdByFeature[a.indices->optionFeature[k] % OAK_FEATURE_CAPACITY] += use;
    }
    uint32_t keyboardFeature = a.indices->optionFeature[0] % OAK_FEATURE_CAPACITY;
    float imp0 = a.memory->importance[0]; if (imp0 < 0.0f) imp0 = 0.0f;
    float keyboardUse = finiteOrZero(fabsf(a.memory->actorAdvantage[0]) + imp0);
    for (uint32_t activeSlot=0u; activeSlot<OAK_ACTIVE_CAPACITY; ++activeSlot){
        uint32_t i = previousActive[activeSlot];
        float optionValueSignal = a.memory->optionTdByFeature[i];
        float keyboardSignal = (i == keyboardFeature) ? keyboardUse : 0.0f;
        float planningSignal = fabsf(valueTdError);
        float selectedPolicyLoad = 0.0f;
        const Record *selected = optionRecord(a, i, 0u);
        for (uint32_t d=0u;d<OAK_POLICY_MAX;++d) selectedPolicyLoad += fabsf(selected->policy[d]);
        float load = fabsf(previousFeatures[i]) * (fabsf(a.parameters->valueWeight[i]) + selectedPolicyLoad);
        load = finiteOrZero(load);
        float featureUtilitySignal = fabsf(planningSignal*previousFeatures[i])
            + fabsf((optionValueSignal + keyboardSignal)*previousFeatures[i]) + load;
        a.memory->featureUtility[i] += trackingRate*(featureUtilitySignal - a.memory->featureUtility[i]);
        {
            float bonusHighWater = a.memory->bonusWeight[i];
            float valueWeight = a.parameters->valueWeight[i];
            a.memory->bonusWeight[i] = keepFinite(bonusHighWater,
                bonusHighWater + OAK_TRACKING_RATE*(valueWeight - bonusHighWater));
        }
        uint32_t age = a.indices->featureMeta[i].w;
        metaOptimizeUpdate(a.memory->valueMeta, i, valueTdError, previousFeatures[i], age,
                           OAK_META_RATE, OAK_TRACKING_RATE);
        uint32_t metaOptimizeSlot = tPrev % OAK_SIMD_WIDTH;
        if (activeSlot == metaOptimizeSlot){
            uint32_t mo = tPrev % OAK_OPTION_COUNT;
            metaOptimizeUpdate(a.memory->optionValueMeta, i, a.memory->optionTd[mo], previousFeatures[i],
                               age, OAK_META_RATE, OAK_TRACKING_RATE);
            metaOptimizeUpdate(a.memory->modelMeta, i, a.memory->optionRewardTd[mo], previousFeatures[i],
                               age, OAK_META_RATE, OAK_TRACKING_RATE);
        }
    }
    for (uint32_t k=0u;k<OAK_OPTION_COUNT;++k){
        float err = a.memory->optionPriority[k];
        float keyboardMass = (k==0u?1.0f:0.0f);
        float sig = a.memory->optionTd[k]*a.memory->optionTd[k]
                  + keyboardMass*fabsf(a.memory->actorAdvantage[k]) - err;
        a.memory->optionUtility[k] += trackingRate*(sig - a.memory->optionUtility[k]);
        if (a.indices->optionAge[k] < 0xffffffffu) a.indices->optionAge[k] += 1u;
        if (a.indices->lastStopAge[k] < 0xffffffffu) a.indices->lastStopAge[k] += 1u;
    }
}

void oakInit(Agent a, uint32_t gid, float initX, float initY, float initZ, float initW){
    uint32_t traceChannels = OAK_TRACE_LEVELS*a.observationCount;
    if (traceChannels < 1u) traceChannels = 1u;
    for (uint32_t feature=0u; feature<OAK_FEATURE_CAPACITY; ++feature){
        uint32_t den = (feature % 4u == 3u) ? traceChannels : OAK_PERCEPT_COUNT;
        uint32_t parent = feature % den;
        u4 nm = { parent, (feature*7u+3u) % OAK_PERCEPT_COUNT, feature % 4u, 0u };
        a.indices->nurseryMeta[feature] = nm;
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
        a.memory->optionUtility[option]=(float)OAK_FEATURE_CAPACITY;
        a.memory->actorMeta[option].x = 1.0f;
        uint32_t hash = gid*2654435761u + option*40503u + 977u;
#ifdef OAK_OPTINIT
        uint32_t action = hash % (a.envActionCount ? a.envActionCount : OAK_POLICY_MAX);
#else
        uint32_t action = hash % OAK_POLICY_MAX;
#endif
        float sign = ((hash >> 16u) & 1u)==0u ? 1.0f : -1.0f;
        optionRecord(a, 0u, option)->policy[action] = sign*initW;
    }
    a.memory->reward = 0.0f;
    for (uint32_t d=0u; d<OAK_ACTION_PARAMS; ++d) a.memory->actionSignal[d]=0.0f;
    for (uint32_t i=0u;i<OAK_OPTION_COUNT*OAK_POLICY_MAX;++i) a.memory->optionActorCredit[i]=0.0f;
    for (uint32_t k=0u;k<OAK_OPTION_COUNT;++k){
        a.memory->importance[k]=1.0f; a.memory->importanceStaging[k]=1.0f; a.indices->parked[k]=0u;
    }
    a.memory->inverseFeatureNorm = rsqrtf_((float)OAK_FEATURE_CAPACITY);
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

void oakActionModelLearn(Agent a, float reward){
    if (!g_actInit){
#ifdef OAK_MODEL_STEP
        float a0 = (OAK_MODEL_STEP);
#else
        float a0 = 1.0f/(float)OAK_ACTIVE_CAPACITY;
#endif
        for (uint32_t i=0;i<OAK_FEATURE_CAPACITY;++i)
            for (uint32_t d=0;d<OAK_POLICY_MAX;++d){
#ifdef OAK_OPTIMISTIC
                g_actQ[i][d]=(OAK_OPTIMISTIC);
#else
                g_actQ[i][d]=0.0f;
#endif
                g_actAge[i][d]=0u;
                f4 m={a0,logf(a0),0.0f,0.0f}; g_actMeta[i][d]=m;
            }
        g_actInit=1;
    }
    if (!g_havePrevAct) return;
    const float *cf = featureValues(a, a.stateSlot);
    const uint32_t *ca = activeFeatures(a, a.stateSlot);
    float vNext=0.0f;
    for (uint32_t z=0;z<OAK_ACTIVE_CAPACITY;++z){ uint32_t i=ca[z]; vNext += a.parameters->valueWeight[i]*cf[i]; }
#ifdef OAK_MYOPIC
    float target = (reward - a.memory->averageReward);
    (void)vNext;
#else
    float target = (reward - a.memory->averageReward) + vNext;
#endif
    const float *pf = featureValues(a, 1u - a.stateSlot);
#ifdef OAK_ALIGN
#define OAK_PV(z,i) g_prevV[z]
    (void)pf;
#else
#define OAK_PV(z,i) pf[i]
#endif
    float pred=0.0f, denom=1.0f;
    for (uint32_t z=OAK_MODEL_Z0;z<OAK_MODEL_ZEND;++z){
        uint32_t i=g_prevF[z]; float xv=OAK_PV(z,i);
        pred += g_actQ[i][g_prevAct]*xv;
        denom += g_actMeta[i][g_prevAct].x * xv*xv;
    }
    float err = target - pred;
    if (!isfinite(err)) return;
    float scale = (denom>0.0f)?1.0f/denom:0.0f;
    for (uint32_t z=OAK_MODEL_Z0;z<OAK_MODEL_ZEND;++z){
        uint32_t i=g_prevF[z]; float x=OAK_PV(z,i);
        float v = g_actQ[i][g_prevAct] + g_actMeta[i][g_prevAct].x*err*x*scale;
#ifdef OAK_MODEL_FORGET
        v -= (OAK_MODEL_FORGET)*g_actQ[i][g_prevAct];
#endif
        if (isfinite(v)) g_actQ[i][g_prevAct]=v;
        metaOptimizeUpdate(&g_actMeta[i][g_prevAct], 0u, err, x, g_actAge[i][g_prevAct],
                           OAK_META_RATE, OAK_TRACKING_RATE);
#ifdef OAK_MODEL_FLOOR
        if (g_actMeta[i][g_prevAct].x < (OAK_MODEL_FLOOR)) g_actMeta[i][g_prevAct].x = (OAK_MODEL_FLOOR);
#endif
        g_actAge[i][g_prevAct]++;
    }
}
void oakActionModelRemember(Agent a, uint32_t action){
    const uint32_t *ca = activeFeatures(a, a.stateSlot);
    const float *cv = featureValues(a, a.stateSlot);
    for (uint32_t z=0;z<OAK_ACTIVE_CAPACITY;++z){ g_prevF[z]=ca[z]; g_prevV[z]=cv[ca[z]]; }
    g_prevAct=action; g_havePrevAct=1;
}

static float g_qVar, g_lVar; static float g_qMean[OAK_POLICY_MAX];

float oakActionQForward(Agent a){
    if(!g_actInit) return 0.0f;
    const float *cf=featureValues(a,a.stateSlot);
    const uint32_t *ca=activeFeatures(a,a.stateSlot);
    float q[OAK_POLICY_MAX]; float m=0.0f;
    for(uint32_t d=0;d<OAK_POLICY_MAX;++d){ float v=0.0f;
        for(uint32_t z=0;z<OAK_ACTIVE_CAPACITY;++z){uint32_t i=ca[z]; v+=g_actQ[i][d]*cf[i];}
        q[d]=isfinite(v)?v:0.0f; m+=q[d]/(float)OAK_POLICY_MAX; }
    float turn = q[1]>q[2]?q[1]:q[2];
    (void)m; return q[0]-turn;
}
#define OAK_NACT (a.envActionCount)
#ifdef OAK_MODEL_SLOTSONLY
#define OAK_MZ0 OAK_PERCEPT_COUNT
#elif defined(OAK_MODEL_PERCEPTONLY)
#define OAK_MZ0 0u
#else
#define OAK_MZ0 0u
#endif
void oakActionModelBiasScaleFree(Agent a){
    if (!g_actInit) return;
    const float *cf = featureValues(a, a.stateSlot);
    const uint32_t *ca = activeFeatures(a, a.stateSlot);
    float q[OAK_POLICY_MAX]; float qm=0.0f;
    for (uint32_t d=0;d<OAK_NACT;++d){
        float v=0.0f;
        for (uint32_t z=OAK_MODEL_Z0;z<OAK_MODEL_ZEND;++z){ uint32_t i=ca[z]; v += g_actQ[i][d]*cf[i]; }
        q[d]=isfinite(v)?v:0.0f; qm += q[d]/(float)OAK_NACT;
    }
#ifdef OAK_PERACTION_CENTER

    for (uint32_t d=0;d<OAK_NACT;++d){
        g_qMean[d] += OAK_TRACKING_RATE*(q[d]-g_qMean[d]);
        q[d] -= g_qMean[d];
    }
    qm = 0.0f;
#endif
    float lm=0.0f;
    for (uint32_t d=0;d<OAK_NACT;++d) lm += finiteOrZero(a.memory->policyMeanRaw[d])/(float)OAK_NACT;
    float qs=0.0f, ls=0.0f;
    for (uint32_t d=0;d<OAK_NACT;++d){
        float dq=q[d]-qm, dl=finiteOrZero(a.memory->policyMeanRaw[d])-lm;
        qs += dq*dq; ls += dl*dl;
    }
    g_qVar += OAK_TRACKING_RATE*(qs - g_qVar);
    g_lVar += OAK_TRACKING_RATE*(ls - g_lVar);
    float qsd = sqrtf(g_qVar > 1e-30f ? g_qVar : 1e-30f);
    float lsd = sqrtf(g_lVar > 1e-30f ? g_lVar : 1e-30f);
#ifdef OAK_FIXED_GAIN
    float gain = (OAK_FIXED_GAIN)/qsd;
#else
    float gain = lsd/qsd;
#endif
    if (!isfinite(gain)) return;
#ifdef OAK_MODEL_ONLY

    for (uint32_t d=0;d<OAK_NACT;++d)
        a.memory->policyMeanRaw[d] = finiteOrZero(gain*(q[d]-qm));
#else
    for (uint32_t d=0;d<OAK_NACT;++d)
        a.memory->policyMeanRaw[d] = finiteOrZero(a.memory->policyMeanRaw[d] + gain*(q[d]-qm));
#endif
}
#undef OAK_NACT
#undef OAK_MZ0

void oakActionModelBias(Agent a, float weight){
    if (!g_actInit || weight<=0.0f) return;
    const float *cf = featureValues(a, a.stateSlot);
    const uint32_t *ca = activeFeatures(a, a.stateSlot);
    float q[OAK_POLICY_MAX]; float mean=0.0f;
    for (uint32_t d=0;d<OAK_POLICY_MAX;++d){
        float v=0.0f;
        for (uint32_t z=OAK_MODEL_Z0;z<OAK_MODEL_ZEND;++z){ uint32_t i=ca[z]; v += g_actQ[i][d]*cf[i]; }
        q[d]=isfinite(v)?v:0.0f; mean += q[d]/(float)OAK_POLICY_MAX;
    }
    for (uint32_t d=0;d<OAK_POLICY_MAX;++d)
        a.memory->policyMeanRaw[d] = finiteOrZero(a.memory->policyMeanRaw[d] + weight*(q[d]-mean));
}

void updateNursery(Agent a){
    uint32_t tPrev = a.stepCount - 1u;
    const float *previousFeatures = featureValues(a, 1u - a.stateSlot);
    const float *currentFeatures  = featureValues(a, a.stateSlot);
    float trackingRate = trackingRateFor(tPrev, OAK_FEATURE_CAPACITY);
    uint32_t off = (((tPrev >> 1u) & 1u) != 0u) ? OAK_SIMD_WIDTH : 0u;
    for (uint32_t blk = off; blk < OAK_FEATURE_CAPACITY; blk += 2u*OAK_SIMD_WIDTH){
        uint32_t hi = blk + OAK_SIMD_WIDTH; if (hi > OAK_FEATURE_CAPACITY) hi = OAK_FEATURE_CAPACITY;
        for (uint32_t n = blk; n < hi; ++n){
            u4 nursery = a.indices->nurseryMeta[n];
            float constructed = constructedValue(a, previousFeatures, nursery.x, nursery.y, nursery.z);
            uint32_t age = nursery.w;
            float r1 = 1.0f/((float)age + 1.0f);
            float rate = r1 > OAK_TRACKING_RATE ? r1 : OAK_TRACKING_RATE;
            float *st = &a.memory->nurseryStats[n*4u];
            st[0] += rate*(constructed - st[0]);
            st[1] += rate*(constructed*constructed - st[1]);
            float var = st[1] - st[0]*st[0]; if (var < 0.0f) var = 0.0f;
            float r2 = 1.0f/(float)(2u*OAK_PERCEPT_COUNT);
            float ridge = r1 > r2 ? r1 : r2;
            float standardizedCandidate = (constructed - st[0]) * rsqrtf_(var + ridge);
            float sig = fabsf(a.memory->td * standardizedCandidate);
            if (isfinite(sig)){
                st[2] += trackingRate*(sig - st[2]);
                st[3] += trackingRate*(sig*sig - st[3]);
            }
            if (age < 0xffffffffu) a.indices->nurseryMeta[n].w = age + 1u;
        }
    }
    for (uint32_t i=0u;i<OAK_FEATURE_CAPACITY;++i) a.memory->optionUseByFeature[i]=0.0f;
    for (uint32_t k=0u;k<OAK_OPTION_COUNT;++k){
        float use = a.memory->optionUtility[k] + (k==0u?1.0f:0.0f);
        if (isfinite(use))
            a.memory->optionUseByFeature[a.indices->optionFeature[k] % OAK_FEATURE_CAPACITY] += use;
    }
    for (uint32_t n=0u;n<OAK_FEATURE_CAPACITY;++n){
        float nurseryUtility = a.memory->nurseryStats[n*4u+2u];
        float lcb = -3.402823466e38f;
        uint32_t nurseryAge = a.indices->nurseryMeta[n].w;
        if (nurseryAge >= OAK_FEATURE_CAPACITY && isfinite(nurseryUtility)){
            float v2 = a.memory->nurseryStats[n*4u+3u] - nurseryUtility*nurseryUtility;
            if (v2 < 0.0f) v2 = 0.0f;
            float r1 = 1.0f/((float)nurseryAge + 1.0f);
            float rate = r1 > OAK_TRACKING_RATE ? r1 : OAK_TRACKING_RATE;
            float v = nurseryUtility - sqrtf(v2*rate);
            if (isfinite(v)) lcb = v;
        }
        a.memory->nurseryScore[n]=lcb;
    }
    for (uint32_t blk = OAK_PERCEPT_COUNT + off; blk < OAK_FEATURE_CAPACITY; blk += 2u*OAK_SIMD_WIDTH){
        uint32_t hi = blk + OAK_SIMD_WIDTH; if (hi > OAK_FEATURE_CAPACITY) hi = OAK_FEATURE_CAPACITY;
        for (uint32_t i = blk; i < hi; ++i){
            float *rs = &a.memory->replaceScore[i*4u];
            u4 meta = a.indices->featureMeta[i];
            if (meta.w < OAK_FEATURE_CAPACITY){ rs[0]=rs[1]=rs[2]=rs[3]=0.0f; continue; }
            uint32_t parentA = meta.x % OAK_FEATURE_CAPACITY;
            float util = finiteOrZero(a.memory->featureUtility[i]);
            float redundancy = finiteOrZero(fabsf(currentFeatures[i]*currentFeatures[parentA]));
            float downstreamUse = finiteOrZero(a.memory->optionUseByFeature[i]
                                               + fabsf(a.parameters->valueWeight[i]));
            rs[0]=util; rs[1]=redundancy; rs[2]=downstreamUse; rs[3]=1.0f;
        }
    }
}

static void zeroOptionColumn(Agent a, uint32_t k){
    for (uint32_t i=0u;i<OAK_FEATURE_CAPACITY;++i){
        Record *rec = optionRecord(a, i, k);
        rec->value=0.0f; rec->aux=0.0f; rec->reward=0.0f;
        rec->duration=0.0f; rec->next=0.0f; rec->nextAux=0.0f;
    }
}

void replaceWorstFeature(Agent a){
    const float *currentFeatures = featureValues(a, 1u - a.stateSlot);
    a.memory->planningReward = a.memory->reward;

    uint32_t bestNursery = 0xffffffffu; float bestNurseryBound = -3.402823466e38f;
    for (uint32_t n=0u;n<OAK_FEATURE_CAPACITY;++n){
        float b = a.memory->nurseryScore[n];
        if (b > bestNurseryBound){ bestNurseryBound = b; bestNursery = n; }
    }
    uint32_t haveWorst=0u, worstFeature=0xffffffffu, worstAge=0u;
    float worstUtility=0.0f, worstRedundancy=0.0f, worstDownstream=0.0f;
    uint32_t eligible=0u; float eUtil=0.0f, eRed=0.0f, eDown=0.0f;
    for (uint32_t i=OAK_PERCEPT_COUNT; i<OAK_FEATURE_CAPACITY; ++i){
        const float *e = &a.memory->replaceScore[i*4u];
        if (e[3] <= 0.0f) continue;
        eligible += 1u; eUtil += e[0]; eRed += e[1]; eDown += e[2];
        float util=e[0], redundancy=e[1], downstream=e[2];
        uint32_t age = a.indices->featureMeta[i].w;
        int worse = (haveWorst==0u) || downstream < worstDownstream
          || (downstream==worstDownstream && util < worstUtility)
          || (downstream==worstDownstream && util==worstUtility && redundancy > worstRedundancy)
          || (downstream==worstDownstream && util==worstUtility && redundancy==worstRedundancy && age > worstAge)
          || (downstream==worstDownstream && util==worstUtility && redundancy==worstRedundancy && age==worstAge && i < worstFeature);
        if (worse){ haveWorst=1u; worstFeature=i; worstAge=age;
                    worstUtility=util; worstRedundancy=redundancy; worstDownstream=downstream; }
    }
    if (!haveWorst){ worstFeature = 0xffffffffu; worstDownstream=0.0f; worstUtility=0.0f; worstRedundancy=0.0f; }

    {
        uint32_t staleLimit = OAK_FEATURE_CAPACITY*OAK_OPTION_COUNT;
        uint32_t parkedStale=0xffffffffu, unparkedStale=0xffffffffu;
        for (uint32_t k2=1u;k2<OAK_OPTION_COUNT;++k2){
            if (a.indices->lastStopAge[k2] < staleLimit) continue;
            if (a.indices->parked[k2]!=0u){ if(k2<parkedStale) parkedStale=k2; }
            else { if(k2<unparkedStale) unparkedStale=k2; }
        }
        uint32_t staleOption = (parkedStale!=0xffffffffu)?parkedStale
                             : ((unparkedStale!=0xffffffffu)?unparkedStale:0u);
        if (staleOption != 0u){
            uint32_t k=staleOption;
            zeroOptionColumn(a,k);
            g_stashDirty = 1;
            a.indices->optionAge[k]=0u; a.indices->lastStopAge[k]=0u;
            a.memory->optionUtility[k]=(float)OAK_FEATURE_CAPACITY;
            a.memory->actorAdvantage[k]=0.0f;
            a.memory->optionAverageReward[k]=0.0f;
            a.memory->optionModelAverageReward[k]=0.0f;
            a.indices->parked[k]=0u;
            uint32_t feat = a.indices->optionFeature[k] % OAK_FEATURE_CAPACITY;
            optionRecord(a,feat,k)->value = a.memory->inverseFeatureNorm;
        }
    }

    uint32_t candidateStart = OAK_PERCEPT_COUNT;
    float invEligible = 1.0f/(float)(eligible>1u?eligible:1u);
    float meanU = eUtil*invEligible, meanR = eRed*invEligible, meanD = eDown*invEligible;
    int lowUtility = worstUtility < meanU;
    int highRedundancy = worstRedundancy > meanR;
    int lowDownstreamUse = worstDownstream <= meanD;
    int structurallyReplaceable = (eligible > 1u) && lowDownstreamUse && (lowUtility || highRedundancy);
    uint32_t worst = (worstFeature != 0xffffffffu) ? worstFeature : candidateStart;
    int doReplace = structurallyReplaceable && worst < OAK_FEATURE_CAPACITY
                 && bestNursery != 0xffffffffu && bestNurseryBound > worstUtility + worstDownstream;

    a.indices->replaceSlot[0]=0u;
    if (doReplace){
        float bestA=-3.402823466e38f; uint32_t seedParentA=0u; int foundA=0;
        for (uint32_t i=0u;i<OAK_FEATURE_CAPACITY;++i){
            if (i==worst) continue;
            float pu = a.memory->featureUtility[i];
            if (!isfinite(pu)) continue;
            if (!foundA || pu > bestA){ bestA=pu; seedParentA=i; foundA=1; }
        }
        float bestB=-3.402823466e38f; uint32_t seedParentB=(OAK_FEATURE_CAPACITY>1u)?1u:0u; int foundB=0;
        for (uint32_t i=0u;i<OAK_FEATURE_CAPACITY;++i){
            if (i==worst || i==seedParentA) continue;
            float decor = 1.0f/(1.0f + fabsf(currentFeatures[i]*currentFeatures[seedParentA]));
            float pu = a.memory->featureUtility[i]*decor;
            if (!isfinite(pu)) continue;
            if (!foundB || pu > bestB){ bestB=pu; seedParentB=i; foundB=1; }
        }
        uint32_t trackingOption=0xffffffffu, leastDemandOption=0xffffffffu; float leastDemand=3.402823466e38f;
        for (uint32_t ok=1u;ok<OAK_OPTION_COUNT;++ok){
            float use = a.memory->optionUtility[ok] + a.memory->optionPriority[ok];
            if (!isfinite(use)) use = 3.402823466e38f;
            if (a.indices->optionFeature[ok]==worst && ok<trackingOption) trackingOption=ok;
            if (use < leastDemand){ leastDemand=use; leastDemandOption=ok; }
        }
        uint32_t replacedOption = (trackingOption!=0xffffffffu) ? trackingOption
            : ((leastDemandOption==0xffffffffu)?1u:leastDemandOption);
        u4 nursery = a.indices->nurseryMeta[bestNursery];
        uint32_t birth = a.indices->featureBirthCount[worst] + 1u;
        a.indices->featureBirthCount[worst]=birth;
        u4 fm = { nursery.x % OAK_FEATURE_CAPACITY, nursery.y % OAK_FEATURE_CAPACITY,
                  nursery.z % 4u, nursery.w };
        a.indices->featureMeta[worst]=fm;
        f2 mom = { a.memory->nurseryStats[bestNursery*4u], a.memory->nurseryStats[bestNursery*4u+1u] };
        a.memory->featureMoments[worst]=mom;
        a.memory->featureUtility[worst]=a.memory->nurseryStats[bestNursery*4u+2u];
        a.parameters->valueWeight[worst]=0.0f;
        a.memory->bonusWeight[worst]=0.0f;
        u4 nn = { seedParentA, seedParentB, (nursery.z+1u)%4u, 0u };
        a.indices->nurseryMeta[bestNursery]=nn;
        for (int q=0;q<4;++q) a.memory->nurseryStats[bestNursery*4u+q]=0.0f;
        float initialStepSize = 1.0f/(float)OAK_ACTIVE_CAPACITY;
        f4 fresh = { initialStepSize, logf(initialStepSize), 0.0f, 0.0f };
        a.memory->valueMeta[worst]=fresh; a.memory->optionValueMeta[worst]=fresh;
        a.memory->modelMeta[worst]=fresh;
        uint32_t k = replacedOption;
        a.indices->optionAge[k]=0u; a.indices->optionFeature[k]=worst;
        a.memory->optionUtility[k]=(float)OAK_FEATURE_CAPACITY;
        a.memory->actorAdvantage[k]=0.0f;
        a.memory->optionAverageReward[k]=0.0f;
        a.memory->optionModelAverageReward[k]=0.0f;
        a.memory->optionFitness[k]=0.0f;
        a.indices->lastStopAge[k]=0u;
        a.indices->replaceSlot[0]=1u; a.indices->replaceSlot[1]=worst;
        a.indices->replaceSlot[2]=k; a.indices->replaceSlot[3]=birth;
    } else if ((a.stepCount % OAK_OPTION_COUNT) == 0u){
        float meanFit=0.0f;
        for (uint32_t k2=0u;k2<OAK_OPTION_COUNT;++k2) meanFit += a.memory->optionFitness[k2];
        meanFit *= OAK_INV_OPTION_COUNT;
        uint32_t replacedOption=0u; float worstFitness=3.402823466e38f;
        for (uint32_t k2=1u;k2<OAK_OPTION_COUNT;++k2){
            if (a.indices->optionAge[k2] < OAK_FEATURE_CAPACITY) continue;
            float f = a.memory->optionFitness[k2];
            if (f < worstFitness){ worstFitness=f; replacedOption=k2; }
        }
        if (replacedOption!=0u && worstFitness<0.0f && worstFitness<meanFit){
            a.indices->optionAge[replacedOption]=0u;
            a.indices->lastStopAge[replacedOption]=0u;
            a.memory->optionFitness[replacedOption]=0.0f;
            a.memory->optionAverageReward[replacedOption]=0.0f;
            a.memory->optionModelAverageReward[replacedOption]=0.0f;
            a.memory->actorAdvantage[replacedOption]=0.0f;
            a.indices->replaceSlot[0]=2u; a.indices->replaceSlot[2]=replacedOption;
            a.indices->replaceSlot[3]=a.stepCount;
        }
    }

    uint32_t replacement = a.indices->replaceSlot[0];
    if (replacement != 0u){
        uint32_t worstF = a.indices->replaceSlot[1], k = a.indices->replaceSlot[2],
                 birth = a.indices->replaceSlot[3];
        float policyPerturbScale = 1.0f/(float)(OAK_ACTION_PARAMS>1u?OAK_ACTION_PARAMS:1u);
        g_stashDirty = 1;
        a.indices->parked[k]=0u;
        for (uint32_t i=0u;i<OAK_FEATURE_CAPACITY;++i){
            Record *rec = optionRecord(a,i,k);
            rec->value=0.0f; rec->aux=0.0f; rec->reward=0.0f;
            rec->duration=0.0f; rec->next=0.0f; rec->nextAux=0.0f;
            const Record *base = optionRecord(a,i,0u);
            for (uint32_t d=0u; d<OAK_POLICY_MAX; ++d){
                uint32_t h = (i*2654435761u) ^ (d*40503u) ^ (k*2246822519u)
                           ^ (birth*3266489917u) ^ a.stepCount;
                h ^= h >> 13; h *= 2654435761u; h ^= h >> 16;
                float hashedNoise = (float)(h & 0xffffu)/32768.0f - 1.0f;
                rec->policy[d] = base->policy[d]
                    + hashedNoise*policyPerturbScale*rsqrtf_((float)OAK_ACTIVE_CAPACITY);
            }
        }
        if (replacement == 1u)
            for (uint32_t ok=0u; ok<OAK_OPTION_COUNT; ++ok)
                memset(optionRecord(a, worstF, ok), 0, sizeof(Record));
        uint32_t ofi = (replacement==1u) ? worstF
                     : (a.indices->optionFeature[k] % OAK_FEATURE_CAPACITY);
        optionRecord(a, ofi, k)->value = a.memory->inverseFeatureNorm;
    }
}
