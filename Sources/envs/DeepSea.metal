#pragma METAL fp math_mode(safe)

inline void deepSea_obs(uint row, uint column, uint lastAction, thread float *obs) {
    float gridExtent = 7.0f;
    obs[0] = float(row) / gridExtent;
    obs[1] = float(column) / gridExtent;
    obs[2] = float(7 - int(column)) / gridExtent;
    obs[3] = float(7 - int(row)) / gridExtent;
    obs[4] = (column == 7u) ? 1.0f : 0.0f;
    obs[5] = (column == 0u) ? 1.0f : 0.0f;
    obs[6] = float(lastAction);
    obs[7] = 1.0f;
}

inline float deepSea_step(device uint *words, uint action, thread float *obs) {
    uint row = words[0], column = words[1];
    uint mapBitIndex = row * 8u + column;
    uint rightwardAction = (mapBitIndex < 32u) ? ((words[3] >> mapBitIndex) & 1u)
                              : ((words[4] >> (mapBitIndex - 32u)) & 1u);
    int columnDelta = ((action & 1u) == rightwardAction) ? 1 : -1;
    int nextColumn = clamp(int(column) + columnDelta, 0, 7);
    uint nextRow = row + 1u;
    float reward = ((action & 1u) != 0u) ? -0.00125f : 0.0f;
    uint nextLastAction = action & 1u;
    if (nextRow >= 7u) {
        if (nextColumn == 7) reward = reward + 1.0f;
        nextRow = 0u;
        nextColumn = 0;
    }
    words[0] = nextRow;
    words[1] = uint(nextColumn);
    words[2] = nextLastAction;
    deepSea_obs(nextRow, uint(nextColumn), nextLastAction, obs);
    return reward;
}
