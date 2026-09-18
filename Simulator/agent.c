#include <stdio.h>

#include <float.h>
#include "agent.h"

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
static float normalizedObservation(Agent a, uint32_t o){
    float mean = a.memory->observationMean[o];
    float second = a.memory->observationSecondMoment[o];
    float var = second - mean*mean; if (var < 0.0f) var = 0.0f;
    return (a.memory->observation[o] - mean) * rsqrtf_(var + 1.0f/(float)(2u*OAK_PERCEPT_COUNT));
}

static float g_symbolValue[OAK_OBSERVATION_CAPACITY][OAK_SYMBOLS];
static uint32_t g_symbolCount[OAK_OBSERVATION_CAPACITY], g_symbol[OAK_OBSERVATION_CAPACITY];
static float g_immediate[OAK_POLICY_MAX][OAK_OBSERVATION_CAPACITY][OAK_SYMBOLS];
static float g_context = 1.0f, g_canonical[OAK_SYMBOLS];
static uint32_t g_stale, g_gate;
static uint32_t encodeSymbol(uint32_t o, float x){
    uint32_t n = g_symbolCount[o];
    for (uint32_t c = 0u; c < n; ++c) if (x == g_symbolValue[o][c]) return c;
    if (n == OAK_SYMBOLS) return OAK_SYMBOLS;
    g_symbolValue[o][n] = x; g_symbolCount[o] = n + 1u; return n;
}
static uint32_t g_prevAct;
static int   g_havePrevAct;

static uint32_t g_traceMod;
typedef struct { uint16_t pa, pb, kind, trace; } CVPlan;
static CVPlan g_cv[OAK_FEATURE_CAPACITY];
static float g_constructed[OAK_FEATURE_CAPACITY];
static uint16_t g_live[OAK_FEATURE_CAPACITY], g_liveAt[OAK_FEATURE_CAPACITY];
static uint32_t g_liveCount;
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
            float obsRate = stepRate > OAK_VALUE_STEP ? stepRate : OAK_VALUE_STEP;
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

    for (uint32_t o = 0u; o < a.observationCount; ++o) g_symbol[o] = encodeSymbol(o, a.memory->observation[o]);
    uint32_t winners[OAK_CONSTRUCTED_SLOTS];
    for (uint32_t s = 0u; s < OAK_CONSTRUCTED_SLOTS; ++s)
        winners[s] = OAK_PERCEPT_COUNT + s*OAK_CANDIDATES_PER_SLOT
                   + (g_symbol[s] < OAK_SYMBOLS ? g_symbol[s]
                      : oakWrapTile(floorf(currentFeatures[s] + OAK_TILE_HALF), OAK_CANDIDATES_PER_SLOT));
    for (uint32_t n = 0u; n < g_liveCount; ++n){
        uint32_t i = g_live[n];
        float value = constructedFrom(a, previousFeatures, g_cv[i]);
        g_constructed[i] = value;
        float rate = a.indices->featureRate[i];
        float mean = a.memory->featureMean[i], second = a.memory->featureSecondMoment[i];
        a.memory->featureMean[i] = mean + rate * (value - mean);
        a.memory->featureSecondMoment[i] = second + rate * (value*value - second);
    }
    for (uint32_t s = 0u; s < OAK_CONSTRUCTED_SLOTS; ++s){
        uint32_t winner = winners[s];
        if (!a.indices->featureAge[winner]){
            float value = constructedFrom(a, previousFeatures, g_cv[winner]);
            g_constructed[winner] = value;
            a.memory->featureMean[winner] = value;
            a.memory->featureSecondMoment[winner] = value*value;
            g_liveAt[winner] = (uint16_t)(g_liveCount + 1u);
            g_live[g_liveCount++] = (uint16_t)winner;
        }
        float mean = a.memory->featureMean[winner], second = a.memory->featureSecondMoment[winner];
        float var = second - mean*mean; if (var < 0.0f) var = 0.0f;
        currentFeatures[currentActive[OAK_PERCEPT_COUNT + s]] = 0.0f;
        float standardized = finiteOrZero((g_constructed[winner] - mean)
                           * rsqrtf_(var + ridgeRate(a.indices->featureAge[winner])));
        currentFeatures[winner] = standardized;
        currentActive[OAK_PERCEPT_COUNT + s] = winner;
        currentActiveValue[OAK_PERCEPT_COUNT + s] = standardized;
        if (a.indices->featureAge[winner] < 0xffffffffu) a.indices->featureAge[winner] += 1u;
        a.indices->featureRate[winner] = trackingRate(a.indices->featureAge[winner]);
    }
}

void predictAndControl(Agent a){
    const uint32_t *previousActive = activeFeatures(a, a.stateSlot);
    const uint32_t *currentActive = activeFeatures(a, 1u - a.stateSlot);
    const float *previousActiveValue = activeValues(a, a.stateSlot);
    const float *currentActiveValue = activeValues(a, 1u - a.stateSlot);
    float previousValue=0.0f, currentValue=0.0f;
    float valueStepDenominator=1.0f, previousFeatureNormSq=0.0f;
    for (uint32_t i = 0u; i < OAK_ACTIVE_CAPACITY; ++i){
        uint32_t pi = previousActive[i], ci = currentActive[i];
        float pa = previousActiveValue[i], ca = currentActiveValue[i];
        previousValue = fmaf(a.parameters->value[pi].weight, pa, previousValue);
        currentValue  = fmaf(a.parameters->value[ci].weight, ca, currentValue);
        valueStepDenominator += OAK_VALUE_STEP * (pa*pa);
        previousFeatureNormSq = fmaf(pa, pa, previousFeatureNormSq);
    }

    float averageRewardOld = a.memory->averageReward;
    float centeredReward = a.memory->reward - averageRewardOld;
    a.memory->td = finiteOrZero(centeredReward + currentValue - previousValue);
    float tracked = averageRewardOld + OAK_TRACKING_RATE*(a.memory->reward - averageRewardOld);
    a.memory->averageReward = isfinite(tracked) ? tracked : averageRewardOld;

    for (uint32_t k = 0u; k < OAK_OPTION_UPDATES; ++k){
        float policyLogit[OAK_POLICY_MAX];
        for (uint32_t j=0u;j<OAK_POLICY_MAX;++j) policyLogit[j]=0.0f;
        for (uint32_t idx = 0u; idx < OAK_ACTIVE_CAPACITY; ++idx){
            float ca = currentActiveValue[idx];
            if (ca == 0.0f) continue;
            const Record *rc = optionRecord(a, currentActive[idx], k);
            for (uint32_t j=0u;j<OAK_POLICY_MAX;++j)
                policyLogit[j] = fmaf(rc->policy[j], ca, policyLogit[j]);
        }
        for (uint32_t j=0u;j<OAK_POLICY_MAX;++j)
            a.memory->optionLogit[k*OAK_POLICY_MAX+j]=policyLogit[j];
        float optionAverageRewardOld = a.memory->optionAverageReward[k];
        float optionTdError = a.memory->td - optionAverageRewardOld;
        a.memory->actorAdvantage[k] = finiteOrZero(optionTdError);
        a.memory->optionAverageReward[k] =
            keepFinite(optionAverageRewardOld, optionAverageRewardOld + OAK_TRACKING_RATE*optionTdError);
    }
    a.memory->valueScale = (isfinite(valueStepDenominator) && valueStepDenominator > 0.0f)
                           ? (1.0f/valueStepDenominator) : 0.0f;
    a.memory->previousFeatureNormSq = previousFeatureNormSq;
}

void planAllOptions(Agent a){
    for (uint32_t k = 0u; k < OAK_OPTION_UPDATES; ++k){
        float importanceRatio = a.memory->importance[k];
        float advantage = finiteOrZero(a.memory->actorAdvantage[k]);
        float actorAlpha = 1.0f / OAK_ACTOR_ALPHA_DENOM;
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

static void replaceLeastUsefulFeature(Agent a){
    const uint32_t *restrict age = a.indices->featureAge;
    uint32_t worst = OAK_FEATURE_CAPACITY;
    float worstUtility = FLT_MAX;
    for (uint32_t n = 0u; n < g_liveCount; ++n){
        uint32_t feature = g_live[n];
        float score = age[feature] >= OAK_FEATURE_TIMESCALE ? a.parameters->value[feature].utility : FLT_MAX;
        int better = score < worstUtility;
        worstUtility = better ? score : worstUtility;
        worst = better ? feature : worst;
    }
    if (worst == OAK_FEATURE_CAPACITY) return;
    { uint32_t at = g_liveAt[worst] - 1u, last = g_live[--g_liveCount];
      g_live[at] = (uint16_t)last; g_liveAt[last] = (uint16_t)(at + 1u); g_liveAt[worst] = 0u; }

    uint32_t draw = a.stepCount*2654435761u + worst*40503u + 977u;
    g_cv[worst].kind  = (uint16_t)(draw % 3u);
    g_cv[worst].trace = (uint16_t)((draw >> 8u) % g_traceMod);
    g_cv[worst].pa    = (uint16_t)((draw >> 11u) % OAK_PERCEPT_COUNT);
    g_cv[worst].pb    = (uint16_t)((draw >> 19u) % OAK_PERCEPT_COUNT);

    a.memory->featureMean[worst] = 0.0f;
    a.memory->featureSecondMoment[worst] = 0.0f;
    a.parameters->value[worst].utility = 0.0f;
    a.parameters->value[worst].weight = 0.0f;
    a.indices->featureAge[worst] = 0u;
    a.indices->featureRate[worst] = trackingRate(0u);
    for (uint32_t option = 0u; option < OAK_OPTION_COUNT; ++option)
        for (uint32_t j = 0u; j < OAK_POLICY_MAX; ++j)
            optionRecord(a, worst, option)->policy[j] = 0.0f;
}

void learnFromTransition(Agent a, uint32_t previousSlot){
    const uint32_t policyN = a.envActionCount;
    const float *currentFeatures  = featureValues(a, 1u - previousSlot);
    const uint32_t *previousActive = activeFeatures(a, previousSlot);
    const float *previousActiveValue = activeValues(a, previousSlot);
    float actorCoefficient[OAK_OPTION_UPDATES][OAK_POLICY_MAX];
    for (uint32_t k=0u;k<OAK_OPTION_UPDATES;++k)
        for (uint32_t j=0u;j<policyN;++j)
            actorCoefficient[k][j]=a.memory->optionPolicyCoef[k*OAK_POLICY_MAX+j];
    for (uint32_t idx=0u; idx<OAK_ACTIVE_CAPACITY; ++idx){
        uint32_t i = previousActive[idx];
        float x = previousActiveValue[idx];
        if (x != 0.0f || currentFeatures[i] != 0.0f)
            for (uint32_t k=0u;k<OAK_OPTION_UPDATES;++k){
                Record *rec = optionRecord(a, i, k);
                for (uint32_t j=0u;j<policyN;++j){
                    float w = rec->policy[j];
                    rec->policy[j] = keepFinite(w, w + actorCoefficient[k][j]*x);
                }
            }
        float w = a.parameters->value[i].weight;
        float updated = keepFinite(w, w + OAK_VALUE_STEP*a.memory->td*x*a.memory->valueScale);
        a.parameters->value[i].weight = updated;
        float utility = a.parameters->value[i].utility;
        a.parameters->value[i].utility = keepFinite(utility,
            utility + OAK_TRACKING_RATE * (fabsf(updated*x) - utility));
    }
    replaceLeastUsefulFeature(a);
}

void oakInit(Agent a, uint32_t gid, float initW){
    for (uint32_t n=0u; n<OAK_FEATURE_TIMESCALE; ++n) g_invAge[n] = 1.0f/((float)n + 1.0f);
    for (uint32_t i=0u; i<OAK_FEATURE_CAPACITY; ++i) a.indices->featureRate[i] = trackingRate(0u);
    for (uint32_t m=1u; m<=OAK_TRACE_LEVELS; ++m) g_traceRate[m] = 1.0f/(float)(1u << m);
    for (uint32_t option=0u; option<OAK_OPTION_COUNT; ++option){
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




void oakActionModelLearn(Agent a, float reward){
    g_stale++;
    if (reward != 0.0f && g_symbol[g_gate] < OAK_SYMBOLS){
        g_stale = 0u;
        float bestGate = 0.0f;
        for (uint32_t o = 0u; o < a.observationCount; ++o){
            float spread = 0.0f;
            for (uint32_t d = 0u; d < a.envActionCount; ++d)
                for (uint32_t c = 0u; c < g_symbolCount[o]; ++c)
                    spread += g_immediate[d][o][c]*g_immediate[d][o][c];
            if (spread > bestGate){ bestGate = spread; g_gate = o; }
        }
        uint32_t s = g_symbol[g_gate];
        float canonical = g_canonical[s];
        if (canonical != 0.0f){
            float context = reward / canonical;
            g_context = keepFinite(g_context, context);
        }
        if (canonical == 0.0f) g_canonical[s] = keepFinite(0.0f, reward / g_context);
    }
    a.memory->reward = finiteOrZero(reward / g_context);
    if (!g_havePrevAct) return;
    for (uint32_t o = 0u; o < a.observationCount; ++o) if (g_symbol[o] < OAK_SYMBOLS){
        float *m = &g_immediate[g_prevAct][o][g_symbol[o]];
        *m = keepFinite(*m, *m + OAK_TRACKING_RATE*(a.memory->reward - *m));
    }
}
void oakActionModelRemember(Agent a, uint32_t action){ g_prevAct=action; g_havePrevAct=1; }

static float g_qVar, g_lVar;


void oakActionModelBiasScaleFree(Agent a){
    const uint32_t actionCount = a.envActionCount;
    for (uint32_t d=0u; d<OAK_POLICY_MAX; ++d)
        a.memory->policyMeanRaw[d] = finiteOrZero(a.memory->optionLogit[d]);
    float q[OAK_POLICY_MAX]; float qm=0.0f;
    float trust = 1.0f/(1.0f + (float)g_stale*OAK_TRACKING_RATE);
    for (uint32_t d=0;d<actionCount;++d){
        float immediate = 0.0f;
        for (uint32_t o = 0u; o < a.observationCount; ++o)
            if (g_symbol[o] < OAK_SYMBOLS) immediate += g_immediate[d][o][g_symbol[o]];
        q[d] = finiteOrZero(g_context*immediate);
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
    float gain = sqrtf(g_lVar > 1e-30f ? g_lVar : 1e-30f)/sqrtf(g_qVar > 1e-30f ? g_qVar : 1e-30f);
    if (!isfinite(gain)) return;
    for (uint32_t d=0;d<actionCount;++d)
        a.memory->policyMeanRaw[d] = finiteOrZero(gain*(q[d]-qm)*trust);
}
