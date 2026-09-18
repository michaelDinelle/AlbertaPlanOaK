
#ifndef OAKAGENT_H
#define OAKAGENT_H
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>

#define OAK_OBSERVATION_CAPACITY 12u
#define OAK_POLICY_MAX 3u
#define OAK_SYMBOLS 16u
#define OAK_FEATURE_TIMESCALE 2048u
#define OAK_TRACKING_RATE (1.0f / (float)OAK_FEATURE_TIMESCALE)

typedef struct {
    float observation[OAK_OBSERVATION_CAPACITY];
    float policyMeanRaw[OAK_POLICY_MAX];
    float reward;
} AgentMemory;

typedef struct {
    uint32_t observationCount, envActionCount;
    AgentMemory *memory;
} Agent;

void constructFeatures(Agent);
void oakInit(Agent);
uint32_t oakSampleDiscrete(Agent, uint32_t, const uint32_t*);
void oakActionModelLearn(Agent, float);
void oakActionModelRemember(Agent, uint32_t);
void oakActionModelBiasScaleFree(Agent);
#endif
