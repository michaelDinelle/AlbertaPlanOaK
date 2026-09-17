
#ifndef OAKAGENT_H
#define OAKAGENT_H
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>

#ifndef OAK_PERCEPTS
#define OAK_PERCEPTS 32u
#endif
#ifndef OAK_SLOTS
#define OAK_SLOTS 48u
#endif
#ifndef OAK_CANDIDATES
#define OAK_CANDIDATES 15u
#endif
#ifndef OAK_OPTIONS
#define OAK_OPTIONS 2u
#endif
#ifndef OAK_UPDATES
#define OAK_UPDATES 2u
#endif
#ifndef OAK_MODEL_STEP
#define OAK_MODEL_STEP 0.03f
#endif
#ifndef OAK_TRACES
#define OAK_TRACES 7u
#endif
#ifndef OAK_ENV_OBS_MAX
#define OAK_ENV_OBS_MAX 12u
#endif
#ifndef OAK_ENV_ACT_MAX
#define OAK_ENV_ACT_MAX 6u
#endif

#define OAK_SIMD_WIDTH 32u
#define OAK_PLAN_PERIOD 32u
#define OAK_CANDIDATES_PER_SLOT OAK_CANDIDATES
#define OAK_TRACE_LEVELS OAK_TRACES
#define OAK_TILE_LAST (OAK_CANDIDATES_PER_SLOT - 1u)
#define OAK_JOINT_ROWS (OAK_TILE_LAST / 2u)
#define OAK_TILE_HALF (0.5f * (float)OAK_CANDIDATES_PER_SLOT)
#define OAK_ACTION_PARAMS (2u * OAK_ENV_ACT_MAX)
#define OAK_POLICY_MAX OAK_ENV_ACT_MAX
#define OAK_OBSERVATION_CAPACITY OAK_ENV_OBS_MAX
#define OAK_SOURCE_COUNT ((1u + OAK_TRACE_LEVELS) * OAK_OBSERVATION_CAPACITY + OAK_ACTION_PARAMS + 2u)
#define OAK_SOURCE_CAPACITY (((OAK_SOURCE_COUNT + OAK_SIMD_WIDTH - 1u) / OAK_SIMD_WIDTH) * OAK_SIMD_WIDTH)
#define OAK_PERCEPT_COUNT OAK_PERCEPTS
#define OAK_CONSTRUCTED_SLOTS OAK_SLOTS
#define OAK_FEATURE_CAPACITY (OAK_PERCEPT_COUNT + OAK_CONSTRUCTED_SLOTS * OAK_CANDIDATES_PER_SLOT)
#define OAK_ACTIVE_CAPACITY (OAK_PERCEPT_COUNT + OAK_CONSTRUCTED_SLOTS)

#define OAK_OPTION_COUNT OAK_OPTIONS
#define OAK_OPTION_UPDATES OAK_UPDATES
#define OAK_OPTION_BLOCKS (OAK_OPTION_COUNT / OAK_SIMD_WIDTH)
#define OAK_UPDATE_BLOCKS (OAK_OPTION_UPDATES / OAK_SIMD_WIDTH)
#define OAK_FEATURE_TIMESCALE (OAK_PERCEPT_COUNT * (1u + OAK_CANDIDATES_PER_SLOT))
#define OAK_TRACKING_RATE (1.0f / (float)OAK_FEATURE_TIMESCALE)
#define OAK_SAMPLE_SIGMA (1.0f / (float)OAK_ACTION_PARAMS)
#ifndef OAK_ACTOR_ALPHA_DENOM
#define OAK_ACTOR_ALPHA_DENOM ((float)(OAK_ACTION_PARAMS * OAK_OPTION_COUNT))
#endif
#define OAK_INV_OPTION_COUNT (1.0f / (float)OAK_OPTION_COUNT)
#ifndef OAK_META_RATE
#define OAK_META_RATE (OAK_TRACKING_RATE / (float)(OAK_FEATURE_TIMESCALE + OAK_ACTION_PARAMS))
#endif
#define OAK_FEATURE_STATES (2u * OAK_FEATURE_CAPACITY)
#define OAK_POLICY_BANK (OAK_OPTION_COUNT * OAK_POLICY_MAX)
#define OAK_FEATURE_STATS (4u * OAK_FEATURE_CAPACITY)
#define OAK_ACTIVE_STATES (2u * OAK_ACTIVE_CAPACITY)

#define FEATURE_PRODUCT 0u
#define FEATURE_NORMALIZED_SUM 1u
#define FEATURE_ZOOMED 2u
#define FEATURE_OBSERVATION_TIMESCALE 3u

typedef struct { float x, y, z, w; } f4;
#ifndef JBW_EXACT_H
typedef struct { uint32_t x, y, z, w; } u4;
#endif
typedef struct { float x, y; } f2;

typedef struct {
    float value, aux, reward, duration, next, nextAux;
    float policy[OAK_POLICY_MAX];
} Record;

typedef struct {
    float valueWeight[OAK_FEATURE_CAPACITY];
    Record option[OAK_FEATURE_CAPACITY * OAK_OPTION_COUNT];
} AgentParameters;

typedef struct {
    float observation[OAK_OBSERVATION_CAPACITY];
    float observationMean[OAK_OBSERVATION_CAPACITY];
    float observationSecondMoment[OAK_OBSERVATION_CAPACITY];
    f4 observationMeta[OAK_OBSERVATION_CAPACITY];
    float features[OAK_FEATURE_STATES];
    float policyMeanRaw[OAK_POLICY_MAX];
    float actionSignal[OAK_ACTION_PARAMS];
    float optionActorCredit[OAK_POLICY_BANK];
    float importance[OAK_OPTION_COUNT];
    float reward;
    float td;
    float optionTd[OAK_OPTION_COUNT];
    f2 featureMoments[OAK_FEATURE_CAPACITY];
    f4 valueMeta[OAK_FEATURE_CAPACITY];
    f4 optionValueMeta[OAK_FEATURE_CAPACITY];
    f4 modelMeta[OAK_FEATURE_CAPACITY];
    float actorAdvantage[OAK_OPTION_COUNT];
    float observationTimescale[OAK_TRACE_LEVELS * OAK_OBSERVATION_CAPACITY];
    float continuation[OAK_OPTION_COUNT];
    float rewardCenter;
    float optionValueScale[OAK_OPTION_COUNT];
    float valueScale;
    float previousFeatureNormSq;
    float optionAverageReward[OAK_OPTION_COUNT];
    float optionFitness[OAK_OPTION_COUNT];
    f4 actorMeta[OAK_OPTION_COUNT];
    float importanceStaging[OAK_OPTION_COUNT];
    float actorCreditStaging[OAK_POLICY_BANK];
    float optionLogit[OAK_POLICY_BANK];
    float optionPolicyCoef[OAK_POLICY_BANK];
    float averageReward;
} AgentMemory;

typedef struct {
    uint32_t optionFeature[OAK_OPTION_COUNT];
    uint32_t optionAge[OAK_OPTION_COUNT];
    uint32_t activeFeature[OAK_ACTIVE_STATES];
    uint32_t parked[OAK_OPTION_COUNT];
    u4 featureMeta[OAK_FEATURE_CAPACITY];
} AgentIndices;

typedef struct {
    uint32_t observationCount, envActionCount, stepCount, stateSlot;
    AgentParameters *parameters;
    AgentMemory *memory;
    AgentIndices *indices;
} Agent;

static inline float keepFinite(float o, float n){ return isfinite(n) ? n : o; }
static inline float finiteOrZero(float v){ return isfinite(v) ? v : 0.0f; }
static inline float clampf_(float v, float lo, float hi){ return v<lo?lo:(v>hi?hi:v); }
static inline float rsqrtf_(float v){ return 1.0f/sqrtf(v); }
static inline float trackingRateFor(uint32_t step, uint32_t cap){
    float a = 1.0f/((float)step + 1.0f), b = 1.0f/(float)(cap>1u?cap:1u);
    return a>b?a:b;
}
static inline Record *optionRecord(Agent a, uint32_t feature, uint32_t option){
    return &a.parameters->option[feature * OAK_OPTION_COUNT + option];
}
static inline float *featureValues(Agent a, uint32_t slot){
    return a.memory->features + slot * OAK_FEATURE_CAPACITY;
}
static inline uint32_t *activeFeatures(Agent a, uint32_t slot){
    return a.indices->activeFeature + slot * OAK_ACTIVE_CAPACITY;
}
static inline uint32_t perceptSourceCount(Agent a){
#ifdef OAK_OBSCYCLE
    return a.observationCount;
#else
    return (1u + OAK_TRACE_LEVELS) * a.observationCount + OAK_ACTION_PARAMS + 2u;
#endif
}
static inline uint32_t updatedOption(uint32_t transition, uint32_t budgetSlot, uint32_t threadIndex){
    uint32_t block = budgetSlot / OAK_SIMD_WIDTH;
    if (block != 0u)
        block = 1u + (transition * (OAK_UPDATE_BLOCKS - 1u) + block - 1u) % (OAK_OPTION_BLOCKS - 1u);
    return block * OAK_SIMD_WIDTH + threadIndex;
}
static inline uint32_t tileSourceIndex(Agent a, uint32_t source){
    return (source < a.observationCount + a.envActionCount) ? source
                                                            : a.observationCount + OAK_ACTION_PARAMS;
}
static inline uint32_t oakWrapTile(float f, uint32_t m){
    if (!isfinite(f)) return 0u;
    double q = fmod((double)f, (double)m);
    if (q < 0.0) q += (double)m;
    return (uint32_t)q;
}
static inline uint32_t zoomedFeatureTile(float activation, uint32_t zoomLevel){
    float zoom = ldexpf(1.0f, (int)zoomLevel);
    return oakWrapTile(floorf(activation*zoom + OAK_TILE_HALF), OAK_CANDIDATES_PER_SLOT);
}
static inline uint32_t jointFeatureTile(float a_, float b_, uint32_t sourceReuse){
    float candidates = (float)OAK_CANDIDATES_PER_SLOT;
    a_ += ((sourceReuse & 1u) == 0u) ? (candidates / (float)OAK_TILE_LAST) : 0.0f;
    uint32_t tileA = oakWrapTile(floorf((a_ + OAK_TILE_HALF) * ((float)OAK_JOINT_ROWS / candidates)),
                                 OAK_JOINT_ROWS);
    uint32_t tileB = oakWrapTile(floorf((b_ + OAK_TILE_HALF) * (2.0f / candidates)), 2u);
    return tileA * 2u + tileB;
}
static inline int wasActive(const uint32_t *prevActive, uint32_t percepts, uint32_t i){
    return i < percepts || prevActive[percepts + (i - percepts) / OAK_CANDIDATES_PER_SLOT] == i;
}
static inline float normalizedObservation(Agent a, uint32_t o){
    float mean = a.memory->observationMean[o];
    float second = a.memory->observationSecondMoment[o];
    float var = second - mean*mean; if (var < 0.0f) var = 0.0f;
    return (a.memory->observation[o] - mean) * rsqrtf_(var + 1.0f/(float)(2u*OAK_PERCEPT_COUNT));
}
static inline void metaOptimizeUpdate(f4 *meta, uint32_t i, float signal, float x, uint32_t age,
                                      float metaEta, float decay){
    if (!isfinite(signal) || !isfinite(x)) return;
    f4 old = meta[i];
    float oldAlpha=old.x, oldBeta=old.y, oldH=old.z, oldGradEnergy=old.w;
    float grad = signal * x;
    float metaGradient = grad * oldH;
    float gradEnergyRate = 1.0f/((float)age + 1.0f);
    if (gradEnergyRate < decay) gradEnergyRate = decay;
    float gradEnergy = keepFinite(oldGradEnergy,
        oldGradEnergy + gradEnergyRate*(metaGradient*metaGradient - oldGradEnergy));
    float ridge = 1.0f/((float)age + 1.0f);
    if (ridge < decay) ridge = decay;
    float ge = gradEnergy > 0.0f ? gradEnergy : 0.0f;
    float normalizedMetaGradient = metaGradient * rsqrtf_(ge + ridge);
    float newBeta = oldBeta + metaEta * normalizedMetaGradient;
    float beta = keepFinite(oldBeta, clampf_(newBeta, 2.0f*logf(decay), 0.0f));
    float alpha = keepFinite(oldAlpha, expf(beta));
    float featureCurvature = x * x;
    float hDecay = (1.0f - decay) * ((1.0f - oldAlpha*featureCurvature) > 0.0f
                                     ? (1.0f - oldAlpha*featureCurvature) : 0.0f);
    f4 nv; nv.x=alpha; nv.y=beta; nv.z=keepFinite(oldH, hDecay*oldH + oldAlpha*grad); nv.w=gradEnergy;
    meta[i]=nv;
}
#endif
