#include <stdlib.h>
#include "agent.h"
#include "rng.h"

#define RANDOM_AGENT_STREAM 7u

typedef struct { uint32_t seed, actionCount, step; } RandomAgent;

void *agentCreate(AgentShape shape, uint32_t seed){
    RandomAgent *agent = calloc(1, sizeof *agent);
    agent->seed = seed;
    agent->actionCount = shape.actionCount;
    return agent;
}

uint32_t agentStep(void *handle, const float *observation, float reward){
    (void)observation;
    (void)reward;
    RandomAgent *agent = handle;
    u4 draw = philox4x32(agent->seed, RNG_KEY_AGENT,
                         (u4){ agent->step++, RANDOM_AGENT_STREAM, 0u, 0u });
    return draw.x % agent->actionCount;
}
