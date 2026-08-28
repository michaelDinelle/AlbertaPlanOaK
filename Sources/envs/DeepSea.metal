#pragma METAL fp math_mode(safe)

inline void deepSeaGrid_obs(uint row, uint column, uint lastAction, int extent,
                            float gridExtent, thread float *obs) {
    obs[0] = float(row) / gridExtent;
    obs[1] = float(column) / gridExtent;
    obs[2] = float(extent - int(column)) / gridExtent;
    obs[3] = float(extent - int(row)) / gridExtent;
    obs[4] = (column == uint(extent)) ? 1.0f : 0.0f;
    obs[5] = (column == 0u) ? 1.0f : 0.0f;
    obs[6] = float(lastAction);
    obs[7] = 1.0f;
}

inline float deepSeaGrid_advance(uint row, uint column, uint rightwardAction, uint action,
                                 int extent, float gridExtent, float moveCost,
                                 thread uint3 *nextState, thread float *obs) {
    int columnDelta = ((action & 1u) == rightwardAction) ? 1 : -1;
    int nextColumn = clamp(int(column) + columnDelta, 0, extent);
    uint nextRow = row + 1u;
    float reward = ((action & 1u) != 0u) ? moveCost : 0.0f;
    uint nextLastAction = action & 1u;
    if (nextRow >= uint(extent)) {
        if (nextColumn == extent) reward = reward + 1.0f;
        nextRow = 0u;
        nextColumn = 0;
    }
    *nextState = uint3(nextRow, uint(nextColumn), nextLastAction);
    deepSeaGrid_obs(nextRow, uint(nextColumn), nextLastAction, extent, gridExtent, obs);
    return reward;
}

inline void deepSea_obs(uint row, uint column, uint lastAction, thread float *obs) {
    deepSeaGrid_obs(row, column, lastAction, 7, 7.0f, obs);
}

inline float deepSea_step(device uint *words, uint action, thread float *obs) {
    uint row = words[0], column = words[1];
    uint mapBitIndex = row * 8u + column;
    uint rightwardAction = (mapBitIndex < 32u) ? ((words[3] >> mapBitIndex) & 1u)
                              : ((words[4] >> (mapBitIndex - 32u)) & 1u);
    uint3 nextState;
    float reward = deepSeaGrid_advance(row, column, rightwardAction, action,
                                       7, 7.0f, -0.00125f, &nextState, obs);
    words[0] = nextState.x;
    words[1] = nextState.y;
    words[2] = nextState.z;
    return reward;
}
