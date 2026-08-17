#pragma METAL fp math_mode(safe)

inline void largeDeepSea_obs(uint row, uint column, uint lastAction, thread float *obs) {
    float gridExtent = 15.0f;
    obs[0] = float(row) / gridExtent;
    obs[1] = float(column) / gridExtent;
    obs[2] = float(15 - int(column)) / gridExtent;
    obs[3] = float(15 - int(row)) / gridExtent;
    obs[4] = (column == 15u) ? 1.0f : 0.0f;
    obs[5] = (column == 0u) ? 1.0f : 0.0f;
    obs[6] = float(lastAction);
    obs[7] = 1.0f;
    for (int i = 8; i < 17; i++) obs[i] = 0.0f;
}

inline float largeDeepSea_step(thread uint *words, uint action, thread float *obs) {
    uint row = words[0] & 15u, column = (words[0] >> 4u) & 15u;
    uint mapBitIndex = row * 16u + column;
    uint rightwardAction = (words[1u + (mapBitIndex >> 5u)] >> (mapBitIndex & 31u)) & 1u;
    int columnDelta = ((action & 1u) == rightwardAction) ? 1 : -1;
    int nextColumn = clamp(int(column) + columnDelta, 0, 15);
    uint nextRow = row + 1u;
    float reward = ((action & 1u) != 0u) ? -0.000625f : 0.0f;
    uint nextLastAction = action & 1u;
    if (nextRow >= 15u) {
        if (nextColumn == 15) reward = reward + 1.0f;
        nextRow = 0u;
        nextColumn = 0;
    }
    words[0] = nextRow | (uint(nextColumn) << 4u) | (nextLastAction << 8u);
    largeDeepSea_obs(nextRow, uint(nextColumn), nextLastAction, obs);
    return reward;
}
