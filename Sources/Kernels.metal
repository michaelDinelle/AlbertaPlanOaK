#include <metal_stdlib>
using namespace metal;

kernel void encode_state(
    device const float *basis [[buffer(0)]],
    device const float *signal [[buffer(1)]],
    device float *state [[buffer(2)]],
    constant uint &signalDim [[buffer(3)]],
    constant uint &rank [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= rank) {
        return;
    }

    float sum = 0.0f;
    uint rowOffset = gid * signalDim;
    for (uint column = 0; column < signalDim; column++) {
        sum += basis[rowOffset + column] * signal[column];
    }
    state[gid] = tanh(sum);
}

kernel void policy_action(
    device const float *policy [[buffer(0)]],
    device const float *state [[buffer(1)]],
    device const float *plannedBias [[buffer(2)]],
    device float *action [[buffer(3)]],
    constant uint &rank [[buffer(4)]],
    constant uint &actDim [[buffer(5)]],
    constant uint &policyStride [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= actDim) {
        return;
    }

    float sum = plannedBias[gid];
    uint rowOffset = gid * policyStride;
    for (uint column = 0; column < rank; column++) {
        sum += policy[rowOffset + column] * state[column];
    }
    action[gid] = tanh(sum);
}
