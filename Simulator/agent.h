
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
#define OAK_SLOTS 6u
#endif
#ifndef OAK_CANDIDATES
#define OAK_CANDIDATES 31u
#endif
#ifndef OAK_OPTIONS
#define OAK_OPTIONS 1u
#endif
#ifndef OAK_UPDATES
#define OAK_UPDATES 1u
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

#define OAK_CANDIDATES_PER_SLOT OAK_CANDIDATES
#define OAK_TRACE_LEVELS OAK_TRACES
#define OAK_TILE_HALF (0.5f * (float)OAK_CANDIDATES_PER_SLOT)
#define OAK_ACTION_PARAMS (2u * OAK_ENV_ACT_MAX)
#define OAK_POLICY_MAX 3u
#define OAK_OBSERVATION_CAPACITY OAK_ENV_OBS_MAX
#define OAK_PERCEPT_COUNT OAK_PERCEPTS
#define OAK_CONSTRUCTED_SLOTS OAK_SLOTS
#define OAK_FEATURE_CAPACITY (OAK_PERCEPT_COUNT + OAK_CONSTRUCTED_SLOTS * OAK_CANDIDATES_PER_SLOT)
#define OAK_ACTIVE_CAPACITY (OAK_PERCEPT_COUNT + OAK_CONSTRUCTED_SLOTS)

#define OAK_OPTION_COUNT OAK_OPTIONS
#define OAK_OPTION_UPDATES OAK_UPDATES
#define OAK_FEATURE_TIMESCALE (OAK_PERCEPT_COUNT * (1u + OAK_CANDIDATES_PER_SLOT))
#define OAK_TRACKING_RATE (1.0f / (float)OAK_FEATURE_TIMESCALE)
#define OAK_VALUE_STEP (1.0f / (float)(2u*OAK_PERCEPT_COUNT))
#define OAK_HEAD_STEP (1.0f / (float)OAK_ACTIVE_CAPACITY)
#define OAK_SYMBOLS 16u
#ifndef OAK_ACTOR_ALPHA_DENOM
#define OAK_ACTOR_ALPHA_DENOM ((float)(OAK_ACTION_PARAMS * OAK_OPTION_COUNT))
#endif
#define OAK_FEATURE_STATES (2u * OAK_FEATURE_CAPACITY)
#define OAK_POLICY_BANK (OAK_OPTION_COUNT * OAK_POLICY_MAX)
#define OAK_ACTIVE_STATES (2u * OAK_ACTIVE_CAPACITY)

#define FEATURE_PRODUCT 0u
#define FEATURE_NORMALIZED_SUM 1u
#define FEATURE_OBSERVATION_TIMESCALE 3u

typedef struct {
    float policy[OAK_POLICY_MAX];
} __attribute__((aligned(16))) Record;

typedef struct {
    struct { float weight; float utility; } value[OAK_FEATURE_CAPACITY];
    Record option[OAK_FEATURE_CAPACITY * OAK_OPTION_COUNT];
} AgentParameters;

typedef struct {
    float observation[OAK_OBSERVATION_CAPACITY];
    float observationMean[OAK_OBSERVATION_CAPACITY];
    float observationSecondMoment[OAK_OBSERVATION_CAPACITY];
    float features[OAK_FEATURE_STATES];
    float activeValue[OAK_ACTIVE_STATES];
    float policyMeanRaw[OAK_POLICY_MAX];
    float actionSignal[OAK_ACTION_PARAMS];
    float optionActorCredit[OAK_POLICY_BANK];
    float importance[OAK_OPTION_COUNT];
    float reward;
    float td;
    float featureMean[OAK_FEATURE_CAPACITY];
    float featureSecondMoment[OAK_FEATURE_CAPACITY];
    float actorAdvantage[OAK_OPTION_COUNT];
    float observationTimescale[OAK_TRACE_LEVELS * OAK_OBSERVATION_CAPACITY];
    float valueScale;
    float previousFeatureNormSq;
    float optionAverageReward[OAK_OPTION_COUNT];
    float importanceStaging[OAK_OPTION_COUNT];
    float actorCreditStaging[OAK_POLICY_BANK];
    float optionLogit[OAK_POLICY_BANK];
    float optionPolicyCoef[OAK_POLICY_BANK];
    float averageReward;
} AgentMemory;

typedef struct {
    uint32_t activeFeature[OAK_ACTIVE_STATES];
    uint32_t featureAge[OAK_FEATURE_CAPACITY];
    float featureRate[OAK_FEATURE_CAPACITY];
} AgentIndices;

typedef struct {
    uint32_t observationCount, envActionCount, stepCount, stateSlot;
    AgentParameters *parameters;
    AgentMemory *memory;
    AgentIndices *indices;
} Agent;

void constructFeatures(Agent);
void predictAndControl(Agent);
void planAllOptions(Agent);
void learnFromTransition(Agent, uint32_t);
void oakInit(Agent, uint32_t, float);
uint32_t oakSampleDiscrete(Agent, uint32_t, const uint32_t*);
void oakActionModelLearn(Agent, float);
void oakActionModelRemember(Agent, uint32_t);
void oakActionModelBiasScaleFree(Agent);
#endif
