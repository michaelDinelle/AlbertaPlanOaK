#include "policy.h"

void constant_act(uint32_t env, float *ctrl) {
    for (uint32_t d = 0u; d < env_actions(env); d++) ctrl[d] = 0.0f;
}
