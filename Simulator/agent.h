#ifndef AGENT_H
#define AGENT_H
#include <stdint.h>

typedef struct { uint32_t observationCount, actionCount; } AgentShape;

void    *agentCreate(AgentShape shape, uint32_t seed);
uint32_t agentStep(void *agent, const float *observation, float reward);

#endif
