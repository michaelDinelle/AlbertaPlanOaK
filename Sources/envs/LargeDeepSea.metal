#pragma METAL fp math_mode(safe)

inline void largeDeepSea_obs(uint row, uint column, uint lastAction, thread float *obs) {
    deepSeaGrid_obs(row, column, lastAction, 15, 15.0f, obs);
}

inline float largeDeepSea_step(device uint *words, uint action, thread float *obs) {
    uint row = words[0] & 15u, column = (words[0] >> 4u) & 15u;
    uint mapBitIndex = row * 16u + column;
    uint rightwardAction = (words[1u + (mapBitIndex >> 5u)] >> (mapBitIndex & 31u)) & 1u;
    uint3 nextState;
    float reward = deepSeaGrid_advance(row, column, rightwardAction, action,
                                       15, 15.0f, -0.000625f, &nextState, obs);
    words[0] = nextState.x | (nextState.y << 4u) | (nextState.z << 8u);
    return reward;
}
