#include <metal_stdlib>
using namespace metal;

kernel void encode_state(
    device const float *basis [[buffer(0)]],
    device const float *signal [[buffer(1)]],
    device float *state [[buffer(2)]],
    constant uint &signalDim [[buffer(3)]],
    constant uint &rank [[buffer(4)]],
    constant uint &stateStride [[buffer(5)]],
    constant uint &signalOffsetElement [[buffer(6)]],
    constant uint &stateOffsetElement [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint feature_idx = gid.x;
    uint batch_idx = gid.y;
    
    if (feature_idx >= rank) {
        return;
    }

    float sum = 0.0f;
    uint basisOffset = feature_idx * signalDim;
    uint signalOffset = signalOffsetElement + batch_idx * signalDim;
    for (uint column = 0; column < signalDim; column++) {
        sum += basis[basisOffset + column] * signal[signalOffset + column];
    }
    state[stateOffsetElement + batch_idx * stateStride + feature_idx] = tanh(sum);
}

kernel void policy_action(
    device const float *policy [[buffer(0)]],
    device const float *state [[buffer(1)]],
    device const float *plannedBias [[buffer(2)]],
    device float *action [[buffer(3)]],
    constant uint &rank [[buffer(4)]],
    constant uint &actDim [[buffer(5)]],
    constant uint &policyStride [[buffer(6)]],
    constant uint &stateStride [[buffer(7)]],
    constant uint &stateOffsetElement [[buffer(8)]],
    constant uint &plannedBiasOffsetElement [[buffer(9)]],
    constant uint &actionOffsetElement [[buffer(10)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint action_idx = gid.x;
    uint batch_idx = gid.y;

    if (action_idx >= actDim) {
        return;
    }

    float sum = plannedBias[plannedBiasOffsetElement + batch_idx * actDim + action_idx];
    uint rowOffset = action_idx * policyStride;
    uint stateOffset = stateOffsetElement + batch_idx * stateStride;
    for (uint column = 0; column < rank; column++) {
        sum += policy[rowOffset + column] * state[stateOffset + column];
    }
    action[actionOffsetElement + batch_idx * actDim + action_idx] = tanh(sum);
}
